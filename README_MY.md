# My Scripts

Notes for the scripts under `examples/` that I added on top of the upstream
`handover-sim2real` codebase.

## Environment

All scripts assume the `pch2r_dev` conda env is active and that
`OMG-Planner` / `GA-DDPG` are present at the repo root. Two env vars need to
be set so the scripts can find them:

```bash
export OMG_PLANNER_DIR=/home/pradyun/h2r/handover-sim2real/OMG-Planner
export GADDPG_DIR=/home/pradyun/h2r/handover-sim2real/GA-DDPG
```

To make these persist on every `conda activate pch2r_dev`:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/handover_env.sh <<'EOF'
export OMG_PLANNER_DIR=/home/pradyun/h2r/handover-sim2real/OMG-Planner
export GADDPG_DIR=/home/pradyun/h2r/handover-sim2real/GA-DDPG
EOF
```

**Pretrained PointNet++ encoder (for `MODEL.pc_pretrained`).** Phase-1 training
warm-starts its PointNet++ encoder from a pretrained checkpoint. Two choices:

- **Recommended — the handover-sim2real CVPR2023 encoder** (5-channel
  xyz+ycb+hand, trained on *this* task). Already present locally under
  `output/cvpr2023_models/`; the default config uses
  `.../2022-10-16_08-48-30_finetune_5_s0_train/DDPG_state_feat_PandaYCBEnv_latest`.
  All **68/68** encoder tensors transfer (first SA conv matches, `(64,8)`). If you
  don't have it, fetch with `./output/fetch_cvpr2023_models.sh`.
- **Fallback — the GA-DDPG grasp encoder** (4-channel xyz+gripper-flag). Only
  **67/68** transfer (first SA conv reinitialized: ours `(64,8)` vs GA-DDPG's
  `(64,7)`), and its one flag means "robot gripper", not "object/hand". Download
  once into `GA-DDPG/output/demo_model/`:

  ```bash
  cd GA-DDPG && gdown 'https://drive.google.com/uc?id=1erCIgqI2FvX-0B7ulg8qJs7eC2K73Smu' -O model.zip \
    && unzip -q -o model.zip -d . && rm model.zip && cd ..
  # (equivalent to GA-DDPG/experiments/scripts/download_model.sh, but gdown handles
  #  Google Drive's confirm token; plain wget grabs the HTML page instead.)
  ```

Train with `--pc-pretrained none` to skip warm-starting and train the encoder
from random init.

---

## Collecting the BC dataset

`examples/collect_bc_dataset.py` runs the OMG planner over every scene in the
chosen split, steps the simulator through each plan, and saves the
per-step observations + expert actions to an HDF5 file.

```bash
python examples/collect_bc_dataset.py \
    --cfg-file examples/pretrain.yaml \
    --output   output/bc_dataset/train.h5 \
    --split    train
```

Useful flags:

| flag | default | purpose |
|---|---|---|
| `--split` | `train` | one of `train` / `val` / `test` (defined by `BENCHMARK.SETUP` in the cfg) |
| `--num-episodes` | all scenes in the split | cap collection to N scenes for a quick run |
| `--seed` | `0` | controls episode shuffling and `PointListener` randomness |
| `--freeze-partial-pointcloud` | off | freeze the cloud to an early frame and hold it for the whole episode (see [Freeze point cloud](#freeze-point-cloud-experimental)) |
| `--freeze-at-step` | `0` | which policy step's cloud to freeze (`0` = the very first step) |

Output layout (one HDF5 file):

```
file.attrs               split, seed, num_episodes, dim metadata
episode_NNNNN/
  ├── point_clouds        float32 [T, 1024, 5]   xyz + ycb_flag + hand_flag, in EE frame
  ├── robot_states        float32 [T, 32]        joint_pos(9)+joint_vel(9)+ee_pose(7)+gripper(1)+prev_act(6)
  └── expert_actions      float32 [T, 7]         Δpos(3)+Δeuler(3)+gripper_cmd(1)
  attrs: scene_idx, num_steps
```

The script logs "OMG planner failed — skipped" for scenes where OMG cannot
find a collision-free grasp; this is expected and just drops the episode.

---

## Freeze point cloud (experimental)

**Motivation.** The depth camera is eye-in-hand (mounted on the gripper). As the
gripper closes in, less of the object stays in the narrow FOV, so the per-step
cloud becomes a **shrinking close-up** of a smaller slice of the object — a view
that may be off-distribution vs. the full objects the policy is meant to grasp.

> **Empirical note (what is and isn't happening).** Inspecting the collected
> clouds: the **object** is *not* duplicate-padded — it keeps its full 896
> unique points at every step; what shrinks is its real *visible extent* (bbox
> diagonal), because a close-up sliver still projects to thousands of pixels. The
> thing that *does* get duplicate-padded (by `regularize_pc_point_count`, which
> oversamples with replacement) is the **hand**, once it leaves the frame
> (e.g. 128 → 2 unique points). So this is mainly a *coverage* issue, not an
> object-duplication artifact.

**What the switch does.** When enabled, the cloud captured at policy step
`FREEZE_PARTIAL_POINTCLOUD_AT_STEP` (**default 0 = the very first step**, gripper
far / object fully in view) is **frozen and held for the rest of the episode** —
no per-step updates, no trigger condition. Because `acc_points` is stored in
world frame and re-projected into the current gripper pose each step, the frozen
cloud stays geometrically correct as the gripper moves. Valid for **static**
hand+object (the `YCB_MANO_START_FRAME=last` setting); for a moving target it
would go stale. (Safety: if the object happens not to be in view at the target
step, the freeze defers to the first later step where it is.)

Implemented in `PointListener._update_acc_points` (`handover_sim2real/policy.py`),
gated by two config keys (`handover_sim2real/config.py`), default **off** so the
original (live, per-step) behaviour is byte-for-byte unchanged:

| config key | default | meaning |
|---|---|---|
| `POLICY.FREEZE_PARTIAL_POINTCLOUD` | `False` | master switch |
| `POLICY.FREEZE_PARTIAL_POINTCLOUD_AT_STEP` | `0` | which policy step's cloud to freeze and hold |

**Where you need the flag — and where you don't.** The freeze happens at
point-cloud *generation* time and is **baked into** the stored clouds. So the
flag is needed wherever the sim **renders** clouds, and is a no-op (and rejected)
where they're **read from disk**:

| stage | script | flag needed? |
|---|---|---|
| collect BC dataset | `collect_bc_dataset.py` | **yes** — bakes frozen clouds into the HDF5 |
| collect DAgger round | `collect_dagger_dataset.py` | **yes** — regenerates clouds live |
| train | `train_bc.py` | **no** — reads the already-frozen clouds from the HDF5 |
| rollout / eval | `rollout_bc_policy.py` | **yes** — regenerates clouds live |

> **Train and eval must match.** A policy trained on frozen-cloud data must be
> rolled out (and DAgger-collected) with the flag on, or it sees a different
> point-cloud distribution than it trained on. The collection scripts record
> `freeze_partial_pointcloud` / `freeze_at_step` into the HDF5 `attrs` so you can
> check what a dataset was built with.

End-to-end example:

```bash
# 1. collect frozen-cloud datasets (note: flag ON)
for split in train val test; do
  python examples/collect_bc_dataset.py --cfg-file examples/pretrain.yaml \
      --output output/bc_dataset/${split}_frozen_pc.h5 --split $split \
      --freeze-partial-pointcloud
done

# 2. train — NO freeze flag; just point at the frozen HDF5s (full paths)
python examples/train_bc.py --cfg-file examples/configs/bc_phase1.yaml \
    --run-name frozen_pc_run1 \
    --train-h5 output/bc_dataset/train_frozen_pc.h5 \
    --val-h5   output/bc_dataset/val_frozen_pc.h5 --num-epochs 100

# 3. roll out — flag ON again, to match training
python examples/rollout_bc_policy.py --run-dir output/bc_runs/frozen_pc_run1 \
    --cfg-file examples/pretrain.yaml --freeze-partial-pointcloud --benchmark --no-render
```

**Choosing the freeze step.** `--freeze-at-step 0` (default) means the policy
navigates the *entire* approach on the initial **far, low-density** view — it
never sees finer object detail as it closes in. If that's too coarse,
`--freeze-at-step N` freezes a later frame (closer, denser, but still before the
object exits the FOV) without any code change. Pick N from the live extent curve:
the object's visible bbox diagonal is roughly flat for the first ~14 steps and
only collapses in the last ~5 (see below), so any N in that flat region is "fully
observed".

**Verifying the freeze took effect.** Because the cloud is now held in world
frame, after re-collecting the object's bbox diagonal should be **constant**
across all steps (and the hand's `obj_uniq`/extent too), instead of shrinking at
the end. Quick check on a collected file:

```python
import h5py, numpy as np
with h5py.File("output/bc_dataset/train_frozen_pc.h5", "r") as f:
    assert f.attrs["freeze_partial_pointcloud"]            # flag actually recorded
    pc = f["episode_00001/point_clouds"][:]               # [T, N, 5]
    for t in (0, pc.shape[0] // 2, pc.shape[0] - 1):
        obj = pc[t][pc[t][:, 3] == 1][:, :3]
        print(t, "obj bbox diag:", np.linalg.norm(obj.max(0) - obj.min(0)))
    # frozen: the three diagonals are identical; live (un-frozen): the last shrinks
```

---

## Visualizing a BC dataset episode

`examples/visualize_bc_dataset.py` supports two modes.

### Static (matplotlib)

Plots the point cloud at t=0,T/2,T-1, the EE trajectory, joint positions, and
the action streams. No simulator needed.

```bash
# random episode
python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train.h5

# specific episode
python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train.h5 --episode 0
```

### Simulator replay (PyBullet GUI)

Opens the PyBullet GUI, re-resets the saved scene, re-runs OMG to recover the
joint trajectory, and steps the sim while overlaying the saved point cloud
as coloured debug points (orange = YCB, blue = hand, grey = background).

```bash
python examples/visualize_bc_dataset.py \
    --dataset  output/bc_dataset/train.h5 \
    --mode     replay \
    --cfg-file examples/pretrain.yaml \
    --episode  0
```

Once the first replay finishes the window stays open. Click on the PyBullet
window to give it keyboard focus, then:

| key | action |
|---|---|
| **R** | reset the scene and replay |
| **Q** | quit |

`Ctrl-C` in the terminal also works.

---

## Visualizing YCB object grasps

`examples/visualize_grasps.py` shows a YCB object's mesh together with the
6-DoF grasp candidates from OMG's two grasp databases:

- `OMG-Planner/data/grasps/simulated/<obj>.npy` — ACRONYM-style sim-sampled
  grasps (orange). This is the set OMG actually loads in `load_grasp_set`.
- `OMG-Planner/data/grasps/graspit/<obj>_grasp_pose.txt` — older
  human-authored set (blue). Only available for ~10 YCB objects.

Each grasp is drawn as a Panda-gripper stick figure (back of hand → palm →
fingers).

```bash
# default: up to 100 of each
python examples/visualize_grasps.py --object 025_mug

# more orange grasps, fewer blue
python examples/visualize_grasps.py --object 025_mug --max-simulated 500 --max-graspit 50

# only the human-authored set (set simulated to 0)
python examples/visualize_grasps.py --object 025_mug --max-simulated 0 --max-graspit 9999

# show grasps in OMG's working frame (post-rotZ(pi/2) offset)
python examples/visualize_grasps.py --object 025_mug --apply-omg-transform
```

Flags:

| flag | default | purpose |
|---|---|---|
| `--object` | (required) | YCB id, e.g. `025_mug`, `005_tomato_soup_can` |
| `--max-simulated` | `100` | random subsample of simulated grasps |
| `--max-graspit` | `100` | random subsample of graspit grasps |
| `--apply-omg-transform` | off | post-multiply simulated grasps by `rotZ(π/2)` to match OMG's internal frame |
| `--seed` | `0` | controls the random subsample |

Note on what you're looking at: the grasp pose `T[:3, 3]` is the location of
the **panda_hand** link, *not* the fingertips. The Franka fingertips sit
~11 cm forward along +z. The script draws the gripper with the real Panda
dimensions, so the rendered fingertips land where they would actually close
on the object.

Objects with a graspit set available (the rest only have the simulated set):

```
003_cracker_box      009_gelatin_box        021_bleach_cleanser
004_sugar_box        010_potted_meat_can    024_bowl
005_tomato_soup_can  019_pitcher_base       025_mug
006_mustard_bottle
```

---

## Phase-1 BC training

`examples/train_bc.py` trains the single-frame behaviour-cloning policy
(PointNet++ scene encoder + robot-state MLP → policy head, SmoothL1 on the
6-D Δpose + BCE on the gripper bit). All hyperparameters live in
`examples/configs/bc_phase1.yaml`; the CLI lets you override the common ones
without editing the yaml.

```bash
# fresh run on the full dataset
python examples/train_bc.py \
    --cfg-file examples/configs/bc_phase1.yaml \
    --run-name phase1_full

# resume the same run from its last checkpoint
python examples/train_bc.py \
    --cfg-file examples/configs/bc_phase1.yaml \
    --run-name phase1_full \
    --resume   output/bc_runs/phase1_full/checkpoints/last.pt

# quick experiment without editing the yaml
python examples/train_bc.py \
    --cfg-file examples/configs/bc_phase1.yaml \
    --run-name lr_sweep_1e4 \
    --batch-size 32 \
    --num-epochs 30
```

CLI overrides (all optional except `--cfg-file`):

| flag | overrides | purpose |
|---|---|---|
| `--run-name` | — | sub-folder of `output/bc_runs/`; default = timestamp |
| `--train-h5` | `DATA.train_h5` | switch to a different training file |
| `--val-h5` | `DATA.val_h5` | switch to a different val file |
| `--device` | `TRAIN.device` | `cuda` / `cpu` |
| `--num-epochs` | `TRAIN.num_epochs` | quick truncation for sanity runs |
| `--batch-size` | `TRAIN.batch_size` | useful for tuning to GPU memory |
| `--use-prev-act` / `--no-prev-act` | `MODEL.use_prev_act` | keep / drop the trailing `prev_action(6)` channels of the robot state |
| `--pc-pretrained PATH` | `MODEL.pc_pretrained` | state-feat checkpoint to init the PointNet++ encoder — CVPR2023 handover (default) or GA-DDPG grasp; `none` to disable |
| `--freeze-pc` / `--no-freeze-pc` | `MODEL.freeze_pc` | freeze the PC encoder (train robot MLP + head only) / fine-tune it |
| `--resume` | — | path to a `.pt` checkpoint to continue from |

**`MODEL.use_prev_act` (default `false`).** The stored robot state ends with the
previous action (`prev_act(6)`). It is ~0.9 correlated with the target action
(the OMG path is smooth), so keeping it lets the policy copy the last action and
ignore the point cloud (the copycat / causal-confusion failure). The data is
always stored full-width (32-D); this flag only decides whether the model slices
those 6 channels off before its robot encoder, so you can toggle it **without
re-collecting**. The choice is saved into the run's `config.yaml`, and
`analyze_bc_run.py` / `rollout_bc_policy.py` read it back so the model matches the
checkpoint.

**`MODEL.pc_pretrained` + `MODEL.freeze_pc`.** The PointNet++ encoder trains from
random init unless you warm-start it. The default points at the **handover-sim2real
CVPR2023 encoder**
(`output/cvpr2023_models/2022-10-16_08-48-30_finetune_5_s0_train/DDPG_state_feat_PandaYCBEnv_latest`)
— a 5-channel (xyz+ycb+hand) encoder trained on this exact task, so all **68/68**
encoder tensors transfer (first SA conv matches, `(64,8)`). Pointing it at the
**GA-DDPG grasp** encoder instead transfers only **67/68** (its cloud is 4-channel
xyz+gripper-flag, so the first SA conv `(64,7)` is reinitialized to our `(64,8)`,
and that flag means "robot gripper", not "object/hand"). `freeze_pc: false`
(default) **fine-tunes** the encoder from that init; `freeze_pc: true` freezes it
(BN kept in eval) and trains only the robot MLP + head (~235k params). Pretrained
init runs on **fresh runs only**; on `--resume` the weights come from the resume
checkpoint.

Why the CVPR2023 encoder and not GA-DDPG's? This repo *is* the CVPR2023
handover-sim2real paper. Its policy point cloud is 5-channel xyz+object+hand (see
`handover_sim2real/policy.py`), identical to ours, and it trains PointNet++ from
scratch end-to-end — no GA-DDPG weights are loaded (the pretrain stage runs with
`pretrained_path=None`). So its own `state_feat` is a same-task, same-semantics,
exact-shape init; GA-DDPG's grasp encoder is only a weaker proxy.

Each run writes to `output/bc_runs/<run-name>/`:

```
config.yaml          # resolved hyperparameters that produced this run
normalization.npz    # per-channel mean/std used for inputs and targets
log.csv              # one row per epoch with all losses and metrics
checkpoints/
  ├── last.pt        # latest weights (saved every TRAIN.save_every epochs)
  └── best.pt        # weights at the epoch with the lowest val total loss
```

The CSV columns are `epoch, lr, wall_s` plus `train_*` and `val_*` prefixed
versions of: `pose_loss`, `gripper_loss`, `total`, `pose_l1`, `pose_pos_l1`,
`pose_rot_l1`, `gripper_acc`. The `_l1` and `gripper_acc` columns are the
ones to watch — they're in interpretable units (normalized-pose L1; binary
accuracy) and don't shift with loss-weighting changes.

A few non-obvious behaviours worth knowing:

- **Resume keeps the original normalization.** If the run dir already has a
  `normalization.npz`, `--resume` loads it instead of recomputing — so the
  resumed model sees the same input distribution it was trained on.
- **`drop_last=True` on train, `False` on val.** Avoids a B=1 final training
  batch (BatchNorm chokes) while still validating on every sample.
- **No tensorboard/wandb.** Just CSV. For plots:
  `pandas.read_csv("output/bc_runs/<run>/log.csv").plot(x="epoch", y="val_total")`.

---

## Analyzing a finished run

`examples/analyze_bc_run.py` reads a run dir and plots with matplotlib (the
figure window is interactive — pan/zoom; pass `--save` to also drop PNGs into
the run dir). Two analyses, selected with `--mode`:

- **`curves`** — loss / metric curves over epochs from `log.csv`. Prints the
  last-epoch summary and the best `val_total` epoch, then shows a 6-panel
  figure (total loss, pose loss, gripper loss, pose-L1, pos-vs-rot L1,
  gripper accuracy). Cheap, no GPU.
- **`predict`** — loads `best.pt` (falls back to `last.pt`), runs
  `model.predict()` on a dataset split, and compares the policy's output to
  the stored expert action. Prints a per-episode error table
  (`pos_l1` in metres, `rot_l1` in radians, `grip_acc`) plus a weighted mean,
  then opens an **interactive 8-panel viewer** for one episode: the 6 Δpose
  components (expert vs policy), the gripper command, and per-step error
  magnitude. `--episode` just sets the starting episode — you then page through
  the whole split without re-running:

  | key / control | action |
  |---|---|
  | **→** / **n** / **Next ▶** button | next episode |
  | **←** / **p** / **◀ Prev** button | previous episode |
  | **s** | save the current episode as `predict_episode_NNNNN.png` |
  | **q** / **Esc** | close |

  The title shows `episode_NNNNN (scene S, T steps) [i/N]` so you always know
  where you are. (Predictions for all episodes are computed once up front, so
  paging is instant.)

```bash
# training curves only
python examples/analyze_bc_run.py --run-dir output/bc_runs/phase1_full --mode curves

# qualitative predicted-vs-expert for one val episode
python examples/analyze_bc_run.py --run-dir output/bc_runs/phase1_full \
    --mode predict --split val --episode 3

# both (default), and also save PNGs into the run dir
python examples/analyze_bc_run.py --run-dir output/bc_runs/phase1_full --save
```

| flag | default | purpose |
|---|---|---|
| `--run-dir` | (required) | `output/bc_runs/<name>` |
| `--mode` | `both` | `curves` / `predict` / `both` |
| `--split` | `val` | dataset split for `predict` (looked up in the run's config.yaml) |
| `--dataset` | — | explicit HDF5 path, overrides `--split` |
| `--episode` | first | starting episode for the interactive viewer (page with ←/→) |
| `--checkpoint` | best→last | explicit `.pt` to load |
| `--device` | `cuda` | inference device |
| `--save` | off | also save the figures as PNGs into the run dir |

`predict` works in **real units**: it feeds raw pc/robot-state into
`model.predict()`, which normalizes the input and denormalizes the output,
so `pos_l1` is in metres and `rot_l1` in radians. The CSV's `pose_l1` is in
*normalized* units — the two are not directly comparable.

> A low `pos_l1` here only means the policy matches the expert **on states
> the expert visited**. It does not prove the policy can do the task on its
> own — for that, see the closed-loop rollout below.

---

## Closed-loop policy rollout (the real test)

`examples/rollout_bc_policy.py` drives the robot with the **policy's own**
predicted actions, step by step, in the simulator — no OMG planner, no expert
trajectory. Each step: build (point_cloud, robot_state) exactly as at
collection time → `model.predict()` → Δee-pose ∘ current-ee-pose → IK →
target joint position → step the sim. This is what exposes covariate shift:
the policy has to live with its own accumulated error, which the
predicted-vs-expert plots cannot show.

**Grasp-and-back hand-off (same as Phase-2).** The first time the policy commands
the gripper to close (`action[6] < 0.5`), the rollout hands off to the scripted
`grasp_and_back` (close in place, `standoff_offset = 0`, then carry to
`GOAL_CENTER`) and lets the benchmark decide success; never closing within
`--max-steps` is a failure. This makes the official `SUCCESS` reachable (it needs
the object grasped **and** carried to the goal — approach alone never scores).
`--benchmark` reports **success rate**, **grasp rate** (both fingers gripping the
object during `grasp_and_back`, via `GraspBenchmarkWrapper.grasped_active()`), and
**commanded-close** count — the same diagnostic split described under
[Phase-2 → Closed-loop rollout](#closed-loop-rollout).

```bash
python examples/rollout_bc_policy.py \
    --run-dir  output/bc_runs/run2 \
    --cfg-file examples/pretrain.yaml \
    --scene    0

# headless (prints the step-by-step ee→ycb distance and final status, no GUI)
python examples/rollout_bc_policy.py \
    --run-dir  output/bc_runs/run2 \
    --cfg-file examples/pretrain.yaml \
    --scene 0 --no-render
```

In the PyBullet window:

| key | action |
|---|---|
| **R** | re-roll the same scene |
| **N** | advance to the next scene and roll out |
| **Q** | quit |

| flag | default | purpose |
|---|---|---|
| `--run-dir` | (required) | trained run to load (`best.pt`, falling back to `last.pt`) |
| `--cfg-file` | (required) | simulator config, e.g. `examples/pretrain.yaml` |
| `--scene` | `0` | scene index to roll out |
| `--max-steps` | `30` | cap on policy steps before stopping |
| `--device` | `cuda` | inference device |
| `--no-render` | off | run headless (no GUI, no point overlay) |
| `--benchmark` | off | headless eval over many scenes → success rate + mean/median final ee→ycb |
| `--num-scenes` | all | benchmark: how many scenes to roll out |
| `--show-goal-grasp` | off | overlay the grasp OMG planned to reach for the scene (green gripper) — see below |
| `--show-grasp-set` | off | also draw the full filtered grasp candidate set (faint grey); implies `--show-goal-grasp` |
| `--freeze-partial-pointcloud` | off | freeze the cloud to an early frame — **must match** the policy's training data (see [Freeze point cloud](#freeze-point-cloud-experimental)) |
| `--freeze-at-step` | `0` | which policy step's cloud to freeze (`0` = the very first step) |

Each step prints the predicted Δpos, the gripper command (open/close), and
the end-effector → YCB distance so you can watch the arm converge (or not).
At the end it reports the benchmark status
(`SUCCESS` / `HUMAN_CONTACT` / `OBJECT_DROP` / `TIMEOUT`) and the final
ee→ycb distance. The live point cloud the policy sees is overlaid as coloured
debug points (orange = YCB, blue = hand, grey = background).

### Seeing the grasp the policy is aiming for

`--show-goal-grasp` overlays the grasp the OMG planner targeted for the scene as
a **green** gripper wireframe, so you can watch the policy's gripper converge to
it (or stall short / drift away). Add `--show-grasp-set` to also draw the full
filtered candidate set in **faint grey** — the green one is the grasp OMG
selected from that set.

```bash
python examples/rollout_bc_policy.py \
    --run-dir  output/bc_runs/phase1_full \
    --cfg-file examples/pretrain.yaml \
    --scene 0 --show-goal-grasp --show-grasp-set
```

The OMG goal is deterministic per scene, so for the static handover this is
exactly the grasp the expert demonstrations aimed at — the target the policy is
imitating. The marker is drawn once (the object is static) and stays up for the
whole roll; on **R**/**N** it is cleared and redrawn for the new scene. Notes:

- The green gripper marks the **final grasp** (the pose the gripper closes at),
  not the pre-grasp standoff OMG uses as its RL goal — so it should sit right on
  the object, overlapping the selected grey candidate. A small offset from every
  grey gripper is just IK residual and is harmless.
- Ignored under `--no-render` / `--benchmark` (the overlay needs the GUI), so it
  adds no planner overhead to headless eval.
- Running OMG only plans — it does **not** step the sim — so the rollout itself
  is unaffected by the overlay.

### Headless benchmark (success rate over many scenes)

`--benchmark` rolls the policy out over many scenes with no GUI and prints a
summary — the metric to watch when comparing models (e.g. before/after DAgger):

```bash
python examples/rollout_bc_policy.py \
    --run-dir  output/bc_runs/run2 \
    --cfg-file examples/pretrain.yaml \
    --benchmark --num-scenes 20 --no-render
# ==== benchmark ====
# success rate  : 3/20 = 15.0%
# final ee→ycb  : mean 0.41 m  median 0.39 m  ...
```

---

## Iterative DAgger (fixing covariate shift)

Plain BC only ever sees expert (OMG) states, so it has no labels for the
off-distribution states the policy actually visits in closed loop. That's why
rollouts collapse to one canonical motion that stalls short of the object and
then drifts. **DAgger** fixes this: roll out the *current* policy, query the OMG
expert at each visited state (re-planned from the current joints), add those
`(state → expert action)` pairs to the dataset, and retrain — repeat.

### Full loop (one command)

`examples/run_dagger.sh` runs K rounds end-to-end. Each round collects with the
latest policy, then retrains a fresh run on `train.h5` **+** all DAgger files so
far. Round 1 rolls out `run2` by default.

```bash
# defaults: ITERS=3, NUM_EPISODES=50, MAX_STEPS=25, starts from output/bc_runs/run2
bash examples/run_dagger.sh

# quick first signal (smaller, faster)
ITERS=2 NUM_EPISODES=30 MAX_STEPS=25 bash examples/run_dagger.sh
```

Outputs: `output/bc_dataset/dagger_iter{i}.h5` and runs
`output/bc_runs/dagger_iter{i}/`. The final policy is `dagger_iter{ITERS}`.

Overridable env vars: `ITERS`, `NUM_EPISODES` (scenes per round), `MAX_STEPS`,
`NUM_EPOCHS`, `SEED`, `DEVICE`, `PREV_RUN` (policy used in round 1),
`BASE_TRAIN_H5`, `VAL_H5`, `SIM_CFG`, `TRAIN_CFG`.

**β schedule.** By default β is **annealed across rounds**: round 1 uses
`BETA_START` (`0.5`), round `ITERS` uses `BETA_END` (`0.0`), linearly in between
(e.g. 0.50 → 0.25 → 0.00 for `ITERS=3`). Mixing in the expert early keeps a weak
policy's rollouts inside the useful funnel; pure DAgger at the end collects the
policy's own off-distribution states. The per-round β is printed each round. Pass
a fixed `BETA=<x>` to override the schedule with one value for all rounds.

### Measure the improvement

```bash
python examples/rollout_bc_policy.py --run-dir output/bc_runs/run2 \
    --cfg-file examples/pretrain.yaml --benchmark --no-render          # baseline
python examples/rollout_bc_policy.py --run-dir output/bc_runs/dagger_iter3 \
    --cfg-file examples/pretrain.yaml --benchmark --no-render          # after DAgger
```

### Running the steps manually

Each round is just two commands; run them yourself for full control.

1. **Collect** a DAgger round with the current policy (`examples/collect_dagger_dataset.py`):

   ```bash
   python examples/collect_dagger_dataset.py \
       --run-dir  output/bc_runs/run2 \
       --cfg-file examples/pretrain.yaml \
       --output   output/bc_dataset/dagger_iter1.h5 \
       --split    train 
   ```

   It drives the sim with the policy's own actions and, at each step, labels the
   visited state with OMG (re-planned from the current joints). Output is the
   **same HDF5 schema** as `collect_bc_dataset.py`.

   | flag | default | purpose |
   |---|---|---|
   | `--run-dir` | (required) | policy to roll out (`best.pt` → `last.pt`) |
   | `--cfg-file` | (required) | simulator config, e.g. `examples/pretrain.yaml` |
   | `--output` | (required) | output HDF5 path |
   | `--num-episodes` | all scenes | scenes to roll out this round |
   | `--max-steps` | `RL_MAX_STEP` | policy steps per scene |
   | `--beta` | `0.0` | prob. of executing the *expert* action instead of the policy (0 = pure DAgger) |
   | `--query-every` | `1` | record an OMG label every K steps |
   | `--drop-past-standoff` / `--no-drop-past-standoff` | **on** | stop recording at the pre-grasp standoff plane; the final reach + close come from `train.h5` |
   | `--dynamic-horizon` / `--static-horizon` | **on** | pick the OMG re-plan length from the EE→standoff distance (constant step size) vs the old `max(RL_MAX_STEP−step, 5)` |
   | `--ee-step` | `0.04` | target per-step EE displacement (m) for `--dynamic-horizon` (≈ the demos' median step) |
   | `--reach-tail` `--min-free` `--max-horizon` | `5` `3` `40` | horizon shaping: `round(dist/ee_step)` free steps (≥`min_free`) + `reach_tail`, capped at `max_horizon` |
   | `--close-pos-thresh` / `--close-rot-thresh` | `0.02` / `0.34` | grasp-reached close-trigger — **only active with `--no-drop-past-standoff`** |
   | `--egl` | off | headless EGL/NVIDIA renderer — **must match** the renderer the policy's data was built with |
   | `--render` | off | show the PyBullet GUI (default headless) |
   | `--freeze-partial-pointcloud` | off | freeze the cloud to an early frame — **must match** the policy's training data (see [Freeze point cloud](#freeze-point-cloud-experimental)) |
   | `--freeze-at-step` | `0` | which policy step's cloud to freeze (`0` = the very first step) |

   **What DAgger records (defaults):** the **approach to the pre-grasp standoff**
   only — gripper always open. Two defaults make this clean:

   - **`--dynamic-horizon`** sets the OMG re-plan length from the current
     EE→standoff distance, so each recorded first-step delta is ~`--ee-step` m at
     *every* distance. This matches the demos' per-step scale and removes the
     late-step "big jump" labels the old fixed-budget horizon produced once the
     free portion collapsed (a stuck-far policy could otherwise be told to lunge
     40–50 cm in one step).
   - **`--drop-past-standoff`** stops recording once the policy crosses the
     standoff plane (8 cm before the grasp). The final straight reach + gripper
     close are owned by `train.h5`, and OMG's *"retreat to the standoff"* labels
     (which it produces for states overshooting past it) are never recorded.

   `use_standoff` stays ON, so DAgger aims at the **same grasp** the demos used.
   To restore the older behavior: `--static-horizon` (step-shrinking horizon) and
   `--no-drop-past-standoff` (record all the way in; `--close-pos-thresh` then adds
   a CLOSE label at policy-visited near-grasp states). Cost note: OMG re-plans
   ~`max_steps`× per scene (vs once in `collect_bc_dataset.py`), so collection is
   the slow step — start with a small `--num-episodes`.

2. **Retrain** on the aggregate with `--dagger-h5` (repeatable):

   ```bash
   python examples/train_bc.py \
       --cfg-file examples/configs/bc_phase1.yaml \
       --run-name dagger_iter1 \
       --train-h5 output/bc_dataset/train.h5 \
       --dagger-h5 output/bc_dataset/dagger_iter1.h5 \
       --num-epochs 50
   ```

   `--dagger-h5` accepts several files (`dagger_iter1.h5 dagger_iter2.h5 ...`);
   normalization is recomputed over the full aggregate and per-file episode
   counts are printed.

> **Backwards compatibility:** all of this is additive. `collect_bc_dataset.py`
> is unchanged, and `train_bc.py` / `rollout_bc_policy.py` / `analyze_bc_run.py`
> behave exactly as before when you don't pass the new flags — a single-file run
> still records `DATA.train_h5` as a plain string. The previous workflow
> (collect → `train_bc.py` → `rollout_bc_policy.py`) runs as it always did.

---

## Phase-2: temporal transformer + ACT

A **second, parallel pipeline** that keeps the Phase-1 MLP policy intact as a
baseline. It adds memory and action chunking on top of the *same* PointNet++
encoder (no surgery — the spatial reasoning stays inside PointNet++):

1. a **temporal transformer** over the last `T` frames of
   `[scene_feat ⊕ robot_feat]` tokens (so it can read hand velocity/intent), and
2. a **CVAE-style Action-Chunking Transformer (ACT)** head that predicts a chunk
   of `k` future actions, executed in closed loop with **temporal ensembling**
   (reduces compounding error, smoother control).

Same action layout and losses as Phase-1 (SmoothL1 on the 6-D Δpose, BCE on the
gripper), now over the chunk, plus the CVAE **KL** term. **No re-collection** —
the windowed dataset reads the existing Phase-1 HDF5 files.

| concern | Phase-1 | Phase-2 (ACT) |
|---|---|---|
| obs | single frame | history of `T` frames (default 4) |
| head | MLP → 1 action | CVAE transformer → chunk of `k` actions (default 8) |
| params | ~1.77M | ~8.9M (same 1.53M PointNet++; +7.1M transformer) |
| files | `models.py`, `dataset.py`, `losses.py`, `trainer.py` | `models_act.py`, `dataset_seq.py`, `losses_act.py`, `trainer_act.py`, `sampler.py` |
| entry / config | `train_bc.py`, `bc_phase1.yaml` | `train_act.py`, `act_phase2.yaml` |
| analyze / rollout | `analyze_bc_run.py`, `rollout_bc_policy.py` | `analyze_act_run.py`, `rollout_act_policy.py` |

> **Runtime env.** The BC/ACT stack needs `h5py` + `torch`(cuda) + `pointnet2_ops`
> — use conda env **`pch2r_dev`** (or `pch2r`); `handover-rs` lacks `h5py`. As
> always, `export GADDPG_DIR=…/GA-DDPG` and put the repo root on `PYTHONPATH`.
> PointNet++'s CUDA ops are **GPU-only**, so `--device cuda` and AMP keeps the
> encoder in fp32 automatically.

### Train

```bash
# fresh run
python examples/train_act.py \
    --cfg-file examples/configs/act_phase2.yaml \
    --train-h5 output/bc_dataset/train.h5 \
    --val-h5   output/bc_dataset/val.h5 \
    --run-name act_run1

# resume the same run from its last checkpoint
# (re-pass --val-h5: it isn't remembered from the run — see note below)
python examples/train_act.py \
    --cfg-file examples/configs/act_phase2.yaml \
    --run-name act_run1 \
    --val-h5   output/bc_dataset/val.h5 \
    --resume   output/bc_runs/act_run1/checkpoints/last.pt
```

On `--resume` the trainer continues the epoch counter and restores the optimizer
/ scheduler / AMP-scaler and `best_val_loss` from the checkpoint, reuses the run's
existing `normalization.npz` (so the train distribution can't drift), and
**skips** the `pc_pretrained` warm-start (the encoder weights come from the resume
checkpoint). `log.csv` is appended to, not truncated. **Important:** the config is
re-read from `--cfg-file` (and the run's `config.yaml` is overwritten with it), so
**re-pass any CLI overrides** you used on the original run — e.g. `--val-h5`,
`--history-len`, `--chunk-len`, `--no-cvae`. In particular `act_phase2.yaml` ships
with an empty `val_h5`, so omitting `--val-h5` silently drops validation.

Run-dir layout (`output/bc_runs/<name>/`: `config.yaml`, `normalization.npz`,
`log.csv`, `checkpoints/{last,best}.pt`) is identical to Phase-1, and `--resume`
/ `--dagger-h5` work the same way. CLI overrides add `--history-len`,
`--chunk-len`, and `--use-cvae` / `--no-cvae` on top of the Phase-1 set.

Key `act_phase2.yaml` knobs (sized for ~700 episodes × ~20 steps — do **not**
import ACT's stock `k=100` / `d_model=512`; the episodes are only ~20 steps):

| field | default | purpose |
|---|---|---|
| `MODEL.history_len` | `4` | `T` observation frames (`1` disables the temporal part) |
| `MODEL.chunk_len` | `8` | `k` future actions predicted per step |
| `MODEL.d_model` / `n_heads` / `enc_layers` / `dec_layers` | `256` / `4` / `3` / `3` | transformer size (kept small for the data regime) |
| `MODEL.latent_dim` / `use_cvae` | `32` / `true` | CVAE latent; `use_cvae=false` ⇒ `z=0`, KL term vanishes |
| `MODEL.use_prev_act` | `false` | drop `prev_act(6)` (copycat guard; also removes the prev-act-under-chunking ambiguity) |
| `LOSS.kl_weight` | `10.0` | β on the KL term |
| `EXEC.mode` / `ensemble_m` | `ensemble` / `0.01` | closed-loop execution: temporal ensembling vs `open_loop` |
| `TRAIN.mixed_precision` | `true` | AMP on the transformer; PointNet++ forced fp32 internally |

**Single model, post-hoc ablations.** Built as one network with switches so you
can attribute results without rewriting: `--history-len 1` (no temporal),
`--no-cvae` (no latent), `EXEC.mode=open_loop` (no ensembling).

### Analyze

`examples/analyze_act_run.py` mirrors `analyze_bc_run.py` (and reuses its
interactive predicted-vs-expert viewer). `curves` adds a **KL panel**; `predict`
reproduces the *deployed* per-step action (history buffer → chunk → ensembling)
on the chosen split and compares it to the expert.

```bash
python examples/analyze_act_run.py --run-dir output/bc_runs/act_run1 --mode curves
python examples/analyze_act_run.py --run-dir output/bc_runs/act_run1 \
    --mode predict --split val --episode 0
```

### Closed-loop rollout

`examples/rollout_act_policy.py` drives the robot with the policy's own chunked
actions (obs ring buffer of length `T`, temporal ensembling), same `--benchmark`
/ GUI controls as the Phase-1 rollout.

**Grasp-and-back hand-off.** The first time the policy commands the gripper to
close (`action[6] < 0.5`), the rollout hands off to the paper's scripted
`grasp_and_back` (close in place — `standoff_offset = 0` — then carry to
`GOAL_CENTER`) and lets the benchmark decide success. If the policy never closes
within `--max-steps`, the episode is a failure. This is what makes the official
`SUCCESS` reachable at all (it requires the object grasped **and** carried to the
goal; just approaching can never score). `--benchmark` now reports three things:
**success rate** (grasped + carried to goal), **grasp rate** (both Panda fingers
gripping the object during `grasp_and_back`), and **commanded-close** count — so
you can tell the failure modes apart:

- low **commanded close** → the policy never decides to grasp (approach stalls);
- **commanded close** high but **grasp rate** ≈ 0 → it grasps too early / on air;
- **grasp rate** > **success rate** → it grasps but doesn't reach the goal.

> **Grasp metric — why not `ycb.released`.** `ycb.released` also fires on the
> *passive* release path (an open gripper merely bumping the object), so it
> over-counts. The clean signal is the benchmark's own "both fingers in
> force-contact with the object" test, which the base wrapper computes privately;
> `handover_sim2real/eval_wrapper.py::GraspBenchmarkWrapper.grasped_active()`
> re-exposes it, and the rollouts only count it **during `grasp_and_back`** (a
> real grasp can't happen with the gripper open during approach). Both rollouts
> construct the env with this subclass.

```bash
python examples/rollout_act_policy.py \
    --run-dir  output/bc_runs/act_base \
    --cfg-file examples/pretrain.yaml \
    --scene 0 --exec-mode receding

python examples/rollout_act_policy.py \
    --run-dir  output/bc_runs/act_run1 \
    --cfg-file examples/pretrain.yaml \
    --benchmark --num-scenes 20 --no-render
```

**Execution strategy (`--exec-mode`).** Temporal ensembling happens **at rollout
time**; by default the mode is read from the run's `config.yaml` (`EXEC.mode`).
Override it without editing the config to test how much ensembling helps or hurts:

| `--exec-mode` | behavior |
|---|---|
| `ensemble` *(default)* | temporal ensembling — average overlapping chunk predictions (can over-smooth → more conservative) |
| `receding` | predict every step, execute **only the first action** — no ensembling, fully reactive |
| `open_loop` | execute the whole `k`-action chunk before re-predicting |

```bash
# disable ensembling (fully reactive) and benchmark — isolates the ensembling effect
python examples/rollout_act_policy.py \
    --run-dir  output/bc_runs/act_base \
    --cfg-file examples/pretrain.yaml \
    --exec-mode receding --benchmark --num-scenes 27 --no-render
```

`--ensemble-m` likewise overrides the ensembling rate (`EXEC.ensemble_m`).

### Two things to watch

- **Val split stays episode-level.** Keep `--val-h5` a *separate* file (already
  episode-disjoint). Overlapping chunks + smooth OMG trajectories make a
  step-level split leak targets and lie about val loss.
- **Posterior collapse is benign.** With a deterministic OMG teacher the CVAE
  latent has little multimodality to capture, so `train_kl_loss` may drift toward
  0 — the decoder just falls back to deterministic. `analyze_act_run.py --mode
  curves` flags this; it means the CVAE isn't contributing, not that training broke.

### DAgger (ACT)

Same covariate-shift fix as [Phase-1 DAgger](#iterative-dagger-fixing-covariate-shift),
adapted to the ACT policy. The OMG labelling and the **HDF5 schema are identical**
— each round records one single frame per step (windowing into histories happens
later, at train time), so `train_act.py --dagger-h5` aggregates the rounds with no
conversion and `BCSequenceDataset` pools episodes across all files. The **only**
difference from Phase-1 collection is that the sim is driven by the ACT policy
(observation history ring buffer of length `T` → predicted chunk → the run's
`EXEC` strategy), exactly as in the closed-loop rollout.

New files: `examples/collect_dagger_act_dataset.py` (collect) and
`examples/run_dagger_act.sh` (the full loop). `T` / `k` / `EXEC.mode` are read
back from the rolled-out run's `config.yaml`.

The loop is **self-contained** — it does the whole DAgger schedule:

- **round 0** — train a base ACT policy on **expert data only** (`train.h5` +
  `val.h5`, no DAgger data), named `act_base`.
- **round i** — roll out the current policy, label with OMG → `dagger_act_iter{i}.h5`,
  retrain a fresh run on `train.h5` + all DAgger files so far. The next round
  rolls out that run.

```bash
# full schedule with the defaults (all scenes, full horizon, 50 epochs/run)
bash examples/run_dagger_act.sh

# quick smoke version (small collection so a round finishes fast)
ITERS=2 NUM_EPISODES=20 bash examples/run_dagger_act.sh
```

Overridable env vars:

| var | default | meaning |
|---|---|---|
| `ITERS` | `3` | number of DAgger rounds |
| `NUM_EPISODES` | *(empty)* | scenes rolled out per round; empty = **all** scenes |
| `MAX_STEPS` | *(empty)* | policy steps per scene; empty = `RL_MAX_STEP` (**no cap**) |
| `BASE_EPOCHS` | `100` | training epochs for the round-0 base run |
| `NUM_EPOCHS` | `50` | training epochs per DAgger round (1..ITERS) |
| `BETA_START` / `BETA_END` | `0.5` / `0.0` | β annealed linearly round 1→`ITERS` (0.50→0.25→0.00 for `ITERS=3`); per-round β is logged in the manifest |
| `BETA` | *(unset)* | set to force a single fixed β for all rounds (overrides the schedule) |
| `BASE_RUN` | `act_base` | name of the round-0 base run |
| `PREV_RUN` | *(empty)* | set to an existing ACT run to **skip round 0** and start from it |
| `FORCE` | `0` | `1` = ignore existing outputs and redo every step (disables resume) |
| `SEED` / `DEVICE` | `0` / `cuda` | — |
| `BASE_TRAIN_H5` / `VAL_H5` | `…/train.h5` / `…/val.h5` | expert data |
| `SIM_CFG` / `TRAIN_CFG` | `pretrain.yaml` / `act_phase2.yaml` | sim / training config |

> **Defaults note.** All-scenes + full-horizon maximizes the DAgger data and
> matches the deployment horizon, but collection is the slow step — OMG re-plans
> once per recorded step, over every scene — so a round with all ~700 scenes is
> expensive. For a first end-to-end test, set a small `NUM_EPISODES`.

**Resumable.** Stop the script (Ctrl-C) and re-run the *same command* to continue
where it left off: a completed collection (a `dagger_act_iter{i}.h5` that already
holds episodes) is detected and skipped, and an interrupted or finished training
is continued via `train_act.py --resume` (a finished run resumes to a no-op).
`FORCE=1` ignores all existing outputs and redoes every step.

**Run manifest.** Every invocation prints — and writes to
`output/bc_dataset/dagger_act_run_<timestamp>.meta.txt` — a manifest of **all
flags** (each tagged `[default]` or `[passed]`), all input/output files, and the
**exact `python … ` command** run for each collection and training step. The
manifest is also copied into each produced run dir as `dagger_manifest.txt`, so
every policy carries a record of how it was trained (alongside its own
`config.yaml`, which holds the model/optimizer hyperparameters).

Outputs `output/bc_dataset/dagger_act_iter{i}.h5` and runs
`output/bc_runs/{act_base, dagger_act_iter{i}}/`. Or run the steps manually:

```bash
# 1. collect one round with the current ACT policy
python examples/collect_dagger_act_dataset.py \
    --run-dir  output/bc_runs/act_run1 \
    --cfg-file examples/pretrain.yaml \
    --output   output/bc_dataset/dagger_act_iter1.h5 \
    --split    train --num-episodes 50 --max-steps 25

# 2. retrain ACT on the aggregate (repeatable: pass several --dagger-h5 files)
python examples/train_act.py \
    --cfg-file examples/configs/act_phase2.yaml \
    --run-name dagger_act_iter1 \
    --train-h5 output/bc_dataset/train.h5 \
    --val-h5   output/bc_dataset/val.h5 \
    --dagger-h5 output/bc_dataset/dagger_act_iter1.h5 \
    --num-epochs 100
```

> **Smoke-test the sim path first.** Collection (sim + OMG + ACT rollout with the
> history buffer / ensembling) is the part that depends on a working OMG +
> GA-DDPG setup. Before committing to a full schedule, verify it on 2 scenes
> against an existing run — finishes in well under a minute:
>
> ```bash
> python examples/collect_dagger_act_dataset.py \
>     --run-dir  output/bc_runs/act_run1 \
>     --cfg-file examples/pretrain.yaml \
>     --output   /tmp/dagger_act_smoke.h5 \
>     --num-episodes 2 --max-steps 6
> ```
>
> Expect `OMG replan failures : 0` and 2 episodes saved; the output HDF5 is the
> same schema as `train.h5`, so `train_act.py --dagger-h5` ingests it directly.

As in Phase-1, DAgger records the **approach to the pre-grasp standoff** only
(gripper open), and `collect_dagger_act_dataset.py` carries the **same defaults**:
`--dynamic-horizon` (constant ~`--ee-step` m steps, no late-step big jumps) and
`--drop-past-standoff` (stop at the standoff plane; the final reach + close stay
learned from `train.h5`). `use_standoff` stays ON so it aims at the same grasp as
the demos. The same escape hatches apply (`--static-horizon`,
`--no-drop-past-standoff` + `--close-pos-thresh`). If your dataset was collected
with the frozen-cloud option, pass `--freeze-partial-pointcloud` (and
`--freeze-at-step`) so the visited states match the policy's training
distribution.

> **Quick sanity check on a collected round.** Per-step `|Δpos|` should sit near
> `--ee-step` (≈ 0.04 m) at *all* step indices — not climb at the end — and most
> episodes should report `Reached standoff`. A flat ~0.035 m profile with `max ≈
> 0.05 m` (vs. the old data's 0.5 m late-step spikes) means the dynamic horizon is
> working.

---

## Phase-3: online RL (TD3 + BC blend)

A **third pipeline** that fine-tunes the Phase-1 reactive policy with
reinforcement learning **blended with BC** — a single-process port of the
GA-DDPG training scheme this codebase is built on. BC/DAgger only ever imitates
the OMG planner; RL adds a value signal from the **actual task reward** (did it
reach and close on the grasp), so the policy can improve *beyond* the planner and
on its **own** error distribution instead of only the states OMG demonstrates.

The blend ≈ **TD3+BC**: a twin critic learns `Q(s,a)` from a sparse terminal
grasp-success reward (contact-hold by default; see `reward_mode`), and the actor
is updated by a small policy-gradient term
(`−λ·Q`) mixed with BC terms (`(1−λ)·[pose SmoothL1 + gripper BCE]`) on the
OMG-labelled states — exactly GA-DDPG's `mix_policy_ratio` form, BC-dominated
early (`λ` ramps `0.1→0.2`).

**Design (locked).**

| decision | choice |
|---|---|
| integration | fully online off-policy, **single-process synchronous** loop (rollout → replay buffer → gradient steps → repeat); no Ray |
| reward | a committed close is **terminal**, scored by `RL.reward_mode`. **`stable_grasp`** (default, paper-faithful): hold the close `hold_steps` and reward `1` iff the object is *secured* — handover-sim's release handshake fired **and** not dropped (contact-hold, the analog of GA-DDPG's lift test; **no OMG grasp pose**). **`proximity`**: `1` iff the EE is within (`0.02 m`, `0.34 rad`) of the OMG grasp pose. Benchmark human-contact/drop or the horizon → terminal-`0`. No carry-to-goal / no `grasp_and_back` |
| RL action | **7-D = Δpose(6) + gripper logit(1)** (normalized pose); exec OPEN iff logit ≥ 0. The gripper is **learned**, not a heuristic |
| gripper learning | the close logit is supervised by **both** (a) a **dense proximity BCE label** — `P(open)=1` far / `0` within grasp proximity — and (b) the sparse close reward; they **agree**, so there's no "always-open" fight. **Training-only** (uses the privileged OMG grasp pose); at deployment the policy closes from its **own logit** — nothing uses the distance |
| demo pool | optional **permanent** pool of pure-OMG **close-at-grasp** demos (`collect_rl_demos.py`), sampled at `demo_frac` per batch (DDPGfD) — supplies the terminal `+1`s the online expert path structurally can't make |
| hand-collision filter | OMG's stock integration never put the human hand in the planning scene, so it picked/approached **hand-colliding grasps** → `FAILURE_HUMAN_CONTACT` mid-approach. The paper's fix — filter grasp candidates against the hand — is now implemented (on by default): OMG prunes grasps whose gripper comes within `hand_collision_thresh` (`0.08 m`) of the MANO hand at the **final** grasp and re-selects a hand-free one, or **skips** the scene if none exists. Applies to demo collection **and** the online OMG expert |
| clock | remaining steps `(max_steps−step)/max_steps` in **both** actor and critic (injected at the fused-feature level, so BC warm-start stays a clean 1:1 load) |
| warm-start | actor reproduces the **full** BC head — pose **and** gripper — at init (verified `<1e-4`); critic encoders copied from the BC run |
| algorithm | deterministic actor + TD3 target smoothing, twin clipped-double-Q, `γ=0.95`, `τ=5e-3`, delayed actor, Bellman ⊕ Monte-Carlo-return blend |
| goal-auxiliary | a head on **both** encoders regresses the EE-relative **final grasp pose** (pos+rot6d) — an aux-only regularizer (`aux_weight=0.5`, output unused) that gives the PointNet++ a dense "where's the grasp" signal for the sparse-reward regime; re-adds the paper's `gₜ` as the buffer's `goal_pose` |

Because the clock is **in the state**, hitting the step horizon is a genuine
terminal — so there's a single `terminal` flag (success/failure/horizon), no
separate truncation handling.

Files (`handover_sim2real/rl/`): `actor.py` (`RLActor` + `warm_start_from_bc` +
goal-aux head), `critic.py` (twin `QNetwork` + goal-aux head), `replay_buffer.py`
(+ `save_demo_transitions`/`load_demo_buffer` for the demo pool),
`td3bc_trainer.py` (`TD3BCTrainer.update` — pose-BC + **gripper BCE** + PG + aux —
+ target soft-update; **the PG term drives the pose channels only — the gripper
logit is fed to the critic detached, trained by BCE toward the proximity label
alone. Without this the unclamped near-binary logit rode `dQ/dlogit` into the
critic's OOD region and blew up to ~5e4 (rl_run4: `q_pi`→950 then collapse at
iter ~95, eval close-rate stuck at 0); the BCE label already encodes "close near
the grasp" = the reward-earning behavior, so no PG signal is lost**),
`rollout_worker.py` (sim rollout, reuses the collectors'
state/IK/geometry helpers + the grasp-proximity reward + per-episode MC returns +
the EE→grasp aux target + the proximity gripper label; **`rollout_episode` does the
policy rollout with a reverse-curriculum warm start — first `expert_initial_steps`
follow the COMMITTED step-0 OMG trajectory by index so the policy takes over near
the grasp — and generates PLAN-TRACKING BC labels: the label at step `t` is the
delta from the CURRENT state to `plan[t]` of the committed plan (GA-DDPG
`expert_plan[int(step)]`), with DAgger replans re-fitting only the plan TAIL to
the drifted state (`dagger_ratio`, never within `dagger_tail_guard` steps of the
plan end) — the OLD label (`plan[0]` of a fresh short-horizon replan every step)
had a stationary attractor at the OMG standoff, 0.08 m short of the grasp, and
taught the policy to hover there (the rl_run7 plateau);
`expert_rollout_episode` is the full-EXPERT rollout (whole traj → close →
reward) used both for a fraction of online episodes AND as `collect_rl_demos.py`'s
one source of truth**). Entries
`examples/train_rl.py` + `examples/collect_rl_demos.py`, config
`examples/configs/rl_phase1.yaml`, curve plotter `examples/plot_rl_run.py`,
held-out eval `examples/rollout_rl_policy.py`.

### Seed a demo pool (recommended)

Under the sparse grasp-proximity reward the online expert path (OMG replan, first
waypoint) **never reaches the grasp within the horizon**, so it produces zero
`+1`s — the only early successes come from the warm-started policy happening to
close in the right place. To guarantee a supply of successes, collect a permanent
pool of pure-OMG demos **once** and mix it into every run:

```bash
python examples/collect_rl_demos.py \
    --sim-cfg examples/pretrain.yaml \
    --rl-cfg  examples/configs/rl_phase1.yaml \
    --bc-run  output/bc_runs/dagger_iter_2_3 \
    --out     output/rl_demos/train.h5 \
    --split   train                 # NOTE: no --egl (see below)
```

It plays the **full** OMG trajectory (so the EE actually reaches ~0.005 m of the
grasp) then appends a **close-at-grasp** transition (`+1`), **streaming** native
RL transitions to an HDF5 file one episode at a time. `train_rl.py --demos` loads
it into a non-evicting pool and samples `demo_frac` (default `0.25`) of every
batch from it (DDPGfD-style), so the `+1`s are never evicted or drowned.

> **⚠ Do NOT use `--egl` for a long run — it OOM-kills.** Measured: pybullet's EGL
> renderer **leaks ~85 MB of GPU memory per scene** (each `reset` loads/removes the
> YCB + MANO meshes and `removeBody` doesn't free them in the EGL plugin — a
> C-level pybullet bug, not fixable here). VRAM climbs to the 8 GB cap in ~80
> scenes, RSS tracks it (7→11 GB), and the OS kills the process. `torch.cuda`'s own
> allocator stays flat (~1 MB) — it's *raw* EGL GPU memory, so `empty_cache()`
> can't help. **Run without `--egl`** (CPU TinyRenderer): leak-free, **~1.1 GB flat
> RSS**, ~5 s/scene, and the depth-based cloud is nearly identical. Keep the same
> renderer choice across the whole pipeline (BC collection, this, RL train/eval).
> Streaming to HDF5 is still worth it: memory-bounded, and a crash/OOM (a SIGKILL
> no `try/except` catches) leaves every episode already on disk.

### Hand-collision grasp filter

The stock OMG integration only ever added the **YCB object** to the planning
scene — never the human hand — so OMG was blind to it and would happily pick (and
drive toward) a grasp whose gripper passes through the hand. That fires the
benchmark's `FAILURE_HUMAN_CONTACT` mid-approach, ending the episode before the
close — the main reason `collect_rl_demos.py` dropped ~half its scenes (the
gripper visibly collides with the hand in the replay).

The paper filters grasp candidates that collide with the hand *before* planning.
That's now implemented and **on by default** for demo collection and the online
OMG/DAgger expert: OMG prunes any grasp whose gripper control points come within
`RL.hand_collision_thresh` of the MANO hand at the **final** grasp pose (not the
8 cm-back standoff — checking the standoff prunes nothing), then re-selects a
hand-free grasp. If **every** grasp collides (no hand-free grasp exists for that
scene), OMG's goal set empties → the scene is **skipped** rather than adding a
hand-collision demo.

The threshold is **`0.08 m`**, calibrated live: the gripper is scored by 6 sparse
control points against the hand's ~50 skeleton joints, a metric that under-reads
true mesh clearance by a few cm — `0.04` left marginal grasps that still made
contact, `0.10` over-pruned good scenes. Effect on scenes 0–9 (train): OFF → 5
`HUMAN_CONTACT`; ON → 2 recovered (clear grasp re-selected), 3 skipped (no
hand-free grasp), collector `7/7 closed-at-grasp, 0 collisions`. This is
**grasp-selection** filtering, not trajectory avoidance — sufficient here because
the collisions were grasp-driven; an approach-sweep collision with an otherwise
clear final grasp would need adding the hand as an OMG obstacle. Disable with
`RL.hand_collision_filter: false`.

### Train

```bash
python examples/train_rl.py --sim-cfg examples/pretrain.yaml \
    --rl-cfg examples/configs/rl_phase1.yaml \
    --bc-run output/bc_runs/dagger_iter_2_3 \
    --demos output/rl_demos/train_h30.h5 --run-name rl_run1


# resume from a checkpoint
python examples/train_rl.py \
    --sim-cfg examples/pretrain.yaml --rl-cfg examples/configs/rl_phase1.yaml \
    --bc-run  output/bc_runs/dagger_iter_2_3 --run-name rl_run1 \
    --resume  output/rl_runs/rl_run1/checkpoints/last.pt
```

**`--bc-run` is required** — it's the seed for everything: the actor is
warm-started to reproduce the BC policy's pose head, the critic's encoders are
copied from it, and its `normalization.npz` + model dims (`config.yaml`) are
reused so the RL nets live in the same normalized action space. This matters
because a **from-scratch** policy under a sparse reward essentially never
succeeds → no learning signal; warm-starting from a competent BC policy is what
makes it tractable (GA-DDPG does the same via its BC-heavy early schedule).

| flag | default | purpose |
|---|---|---|
| `--sim-cfg` | (required) | simulator config, e.g. `examples/pretrain.yaml` |
| `--rl-cfg` | `examples/configs/rl_phase1.yaml` | RL/loop hyperparameters |
| `--bc-run` | (required) | trained BC run to warm-start from (actor + critic encoders + normalizer + dims) |
| `--demos` | — | HDF5 (`.h5`) demo pool from `collect_rl_demos.py` — legacy `.npz` also loads (permanent, mixed in at `LOOP.demo_frac`) |
| `--run-name` | `rl_run1` | sub-folder of `--out-root` (`output/rl_runs/`) |
| `--split` | `train` | scene split to roll out on |
| `--num-iters` | ` ` | override the iteration count |
| `--egl` | off | **avoid** — the EGL renderer leaks ~85 MB GPU/scene and OOMs a long training run; use the leak-free CPU renderer (omit `--egl`) and keep it consistent with the demo pool |
| `--resume` | — | checkpoint `.pt` to continue from |

Key `rl_phase1.yaml` knobs (the scene/encoder/head dims are taken from
`--bc-run`, so they're *not* here):

| field | default | purpose |
|---|---|---|
| `RL.gamma` / `RL.tau` | `0.95` / `0.005` | discount (must be <1) / target soft-update rate |
| `RL.policy_noise` / `noise_clip` | `0.2` / `0.5` | TD3 target-smoothing noise (normalized units) |
| `RL.act_limit` | `5.0` | normalized-action clamp |
| `RL.policy_delay` | `2` | TD3 delayed actor / target updates |
| `RL.mc_blend` | `0.5` | critic target = `(1−b)·Bellman + b·MC-return` (propagates rare sparse successes) |
| `RL.reward_mode` / `RL.hold_steps` | `stable_grasp` / `3` | close-scoring: `stable_grasp` (contact-hold — hold `hold_steps` policy-steps, require released + not dropped) or `proximity` (EE near the OMG grasp pose) |
| `RL.hand_collision_filter` / `hand_collision_thresh` / `hand_points_radius` | `true` / `0.08` / `0.35` | filter OMG grasps that collide with the hand (see [Hand-collision grasp filter](#hand-collision-grasp-filter)); prune radius (m) at the final grasp / cutoff (m) that keeps MANO links near the object |
| `RL.pose_loss` | `smooth_l1` | actor pose-BC loss form. `smooth_l1` = L1 on raw normalized `[Δpos ‖ Δeuler]`. `pm` = point-matching on gripper control points (GA-DDPG). **rl_run10 tried `pm` and REVERTED:** at our sub-0.34-rad deltas PM weights rotation ~6× *less* than `smooth_l1` (the small rotation-channel std makes normalized-L1 a strong rotation weighter — rot/pos 6.6× at the success thresholds vs PM's 1.2×), so it regressed `eval_min_rot`. The "euler-L1 is a bad metric" argument only bites at large/wraparound angles |
| `RL.bc_weight` / `RL.gripper_bc_weight` | `2.0` / `1.0` | weight on the actor pose-BC term (plan-tracking online-DAgger labels, at the policy's OWN visited states) / gripper BCE. **NOTE:** the gripper logit is fed to the critic **detached**, trained by BCE only (an unclamped logit rode `dQ/dlogit` into the critic's OOD region and blew up to ~5e4 in rl_run4) |
| `RL.gripper_close_weight_max` | `10.0` | class-balance the gripper BCE: "close" (near-grasp) is the rare label per batch, so plain BCE learns only "open" and the gripper stays pinned open (`gprob≈1.0`) — upweight close examples to ~match the open mass, capped here |
| `RL.gripper_label_smooth` | `0.1` | **gripper-drift fix (rl_run11)**: label-smooth the gripper BCE targets (1→1−ε open, 0→ε close). The "open" label dominates every batch (policy rarely reaches the close zone → `n_close≈0` online), so plain BCE drives the logit → +∞ until it saturates (`grip_logit` 14→33 over run9) — then `sigmoid≈1`, the gradient vanishes, and close states can't pull it back, so the gripper gets **stuck open** (run9 last.pt: close-rate 4%, all timeout). Smoothing gives it a finite anchor (`logit(0.9)≈+2.2`) so it stays bounded + responsive. `0` = off (run9 behaviour) |
| `RL.aux_weight` | `0.5` | goal-auxiliary grasp-pose loss (both nets; `0` disables the aux head) |
| `RL.pg_normalize` / `RL.alpha` | `true` / `0.1` | `true` = TD3+BC normalization (λ = α ÷ mean-abs-Q); **must** be `true` — a fixed λ diverges (rl_run1 hit Q≈1.6e5). `alpha=0.1` is run9's value. (rl_run10 raised it `→0.25` to push the reach; it did *not* help and the run regressed — revisit only as an isolated single change, watching `q_pi` vs `q_mean`) |
| `RL.mix_start` / `mix_end` / `mix_ramp` | `0.1` / `0.2` / `50000` | RL-weight ramp (BC-dominated early) |
| `LOOP.capacity` | `20000` | online replay transitions (~0.8 GB at 1024×5 clouds) |
| `LOOP.pretrain_updates` | `2000` | **offline** gradient updates on the demo pool only, before any rollout — calibrates the critic so PG doesn't hit a random Q-head (the rl_run1 divergence root). `0` disables; fresh runs only |
| `LOOP.demo_frac_start` / `_end` / `_ramp` | `0.5` / `0.3` / `1000` | scheduled demo fraction per batch: anneal `start→end` over `_ramp` iters, then hold at `_end` (a permanent floor — the frozen demo pool is the strongest on-manifold signal). `_end` raised `0.1→0.3`: annealing it away let the policy drift toward its own zero-reward rollouts (rl_run5 decayed after pretrain). Falls back to constant `LOOP.demo_frac` (`0.25`) if `_start` unset |
| `LOOP.expert_episode_frac` | `0.25` | fraction of online episodes that are **full-EXPERT rollouts** (GA-DDPG non-explore): play the whole OMG traj → close → reward, a guaranteed fresh `+1` into the **online** buffer (the frozen pool goes stale vs the drifting policy). This is what keeps `buf_pos>0`; `0` disables |
| `LOOP.expert_initial_steps` | `28` | **reverse-curriculum warm start** (GA-DDPG `expert_initial`): each policy episode first follows the committed OMG plan BY INDEX for some steps, so the policy takes over NEAR the grasp and practises the settle+close (where it can earn `+1`). This is the STARTING upper bound of the takeover window; see the anneal knobs below |
| `LOOP.expert_initial_anneal_iters` / `expert_initial_end` / `expert_initial_window` | `2800` / `2` / `6` (PC; cluster `350`) | **anneal the curriculum**: the takeover window `[hi−window .. hi]` slides `hi: expert_initial_steps→expert_initial_end` linearly over `anneal_iters`, so early episodes take over right at the grasp (master the reach-tail descent+close first → earn reward → grow the critic's high-Q region from the grasp outward) and late episodes are near-from-scratch. run9 used `550` (27% of a 2000-iter run); **rl_run13 raised it to `2800` (~70% of a 4000-iter run) — a slower, higher-fraction curriculum: ~5× more updates per rung, and only ~30% from-scratch tail (the part that declined in run9/11/12).** rl_run10 tried a *wider* final band (`end 8`) and regressed (under-practiced the `ei=0` eval), so the band stays `[0..2]`. **Omit `_anneal_iters` for the old uniform sampling** (rl_run8 stalled ~0.08 m) |
| `LOOP.rollout_max_steps` | `30` | episode horizon (clock denominator + the worker's OMG plan horizon); `0`=`cfg.RL_MAX_STEP` (`20`). Raised to `30`: the from-scratch policy needs ~30 steps to approach (at step 20 it's still 0.18–0.34 m out) AND the curriculum needs room to place the policy near the grasp with steps left to finish. **The demo pool must be recollected at the SAME horizon** (clock is `remain/max_steps`; `collect_rl_demos` reads this knob) — mixing a 20-step pool with 30-step online feeds the critic inconsistent clocks |
| `LOOP.warmup_episodes` / `warmup_beta` | `0` / `1.0` | seed the online buffer expert-heavy before training (now `0` — demo pool + offline pretrain + expert episodes replace it) |
| `LOOP.episodes_per_iter` / `updates_per_iter` | `2` / `100` | rollout ↔ update ratio per iter |
| `LOOP.noise_std` | `0.1` | rollout exploration noise |
| `LOOP.dagger_ratio` / `dagger_min_step` / `dagger_tail_guard` | `0.5` / `5` / `8` | **DAgger TAIL replans** (GA-DDPG `get_flags` + `expert_plan(step=remaining)`): per-step prob of re-fitting the REST of the committed plan to the policy's drifted state over the REMAINING steps (spliced in), allowed only for `dagger_min_step < step < len(plan) − dagger_tail_guard`. The tail guard keeps the standoff→grasp reach labels **committed**. **rl_run12 tried `1.0`** (feasible mid-traj labels, to bound the `|a_pose|` inflation) — **negative** (didn't bound `|a_pose|` or move the 0.05 m reach ceiling); reverted to `0.5`. Training only; eval never replans |
| `LOOP.beta_start` / `beta_end` / `beta_ramp_iters` | `0.5` / `0.0` / `500` | mid-episode **single-step** corrective expert-execution prob, annealed (DAgger β) — executes the committed plan's CURRENT waypoint, snapping the EE back onto the plan. The full-trajectory expert behaviour is `expert_episode_frac` + `expert_initial_steps`, not β |
| `LOOP.eval_every` / `eval_episodes` | `50` / `50` | deterministic success-rate eval cadence (`eval_episodes` raised `20→50` so a low single-digit rate is resolvable and `eval_min_pos` is usable) |

Run-dir layout (`output/rl_runs/<name>/`):

```
rl_config.yaml       # resolved RL/loop hyperparameters
bc_config.yaml       # the BC run's config (provenance of the warm-start)
normalization.npz    # copied from --bc-run (shared action/state normalization)
log.csv              # one row per iter (curves live here)
checkpoints/
  ├── last.pt        # actor + critic + targets + optimizers + iter (saved every LOOP.save_every)
  └── best.pt        # weights at the best eval success rate
```

### Training curves

`train_rl.py` writes one row per iter to `output/rl_runs/<name>/log.csv`:
`iter, buffer, beta, roll_succ, roll_len, roll_ret, critic_loss, q_mean,
target_mean, actor_loss, pg_loss, bc_loss, grip_loss, aux_c, aux_a, lam,
n_expert, … , eval_succ, best_succ` (fresh run truncates a stale log; `--resume`
appends). `grip_loss` is the gripper BCE (proximity label). **Diagnostic columns
(to see *why* success moves or doesn't):**
- `roll_min_pos` / `eval_min_pos` / `eval_min_rot` — closest EE→grasp distance the
  policy actually reaches (rollout / deterministic eval). Compare to the 0.02 m
  close threshold: this separates "can't reach" from "reaches but won't close."
- `roll_close` / `eval_close` — fraction of episodes that commit a close.
- `roll_skip`, `roll_miss` / `roll_timeout` / `roll_fail` (+ `eval_*`) — terminal
  failure breakdown (OMG/hand-filter skip; grasp-miss; timeout; contact/drop).
- `q_pi` — Q on the **policy's own** action (vs `q_mean` on stored actions; a large
  gap = the actor exploiting the critic OOD).
- `a_absmean` (actor Δpose magnitude — saturation at `act_limit`?), `grip_logit`
  (mean gripper logit — strongly positive ⇒ the policy rarely closes), `buf_pos`
  (online-buffer +reward fraction — 0 until the policy earns its own successes).

`examples/plot_rl_run.py` renders a 2×3 `curves.png` (success + buffer-`+`-frac;
closest approach vs the 0.02 m thresh; close-commit rate; critic; `q_mean` vs
`q_pi`; actor losses + gripper logit):

```bash
python examples/plot_rl_run.py output/rl_runs/rl_run1          # writes curves.png in the run dir
python examples/plot_rl_run.py output/rl_runs/rl_run1 --show   # also opens a window
```

Watch a live run without waiting for it to finish:

```bash
tail -f output/rl_runs/rl_run1/log.csv
watch -n 30 'python examples/plot_rl_run.py output/rl_runs/rl_run1'
```

The metrics that matter: **`eval_succ`** (vs the BC baseline from
`rollout_bc_policy.py --benchmark`) and **`q_mean`** — a runaway `q_mean` is the
classic sparse-reward critic blow-up.

### Evaluate on held-out scenes (val / test)

RL needs no `val.h5`/`test.h5` *files* — there's no fixed dataset to overfit to
(the replay buffer is generated online), so the BC-style val-loss curve doesn't
apply. But held-out **scenes** do matter, and `train_rl.py`'s in-loop `eval_succ`
runs on the **same `--split` it rolls out on** — so that number is *train-scene*
success, not generalization (the console line says "in-loop, not held-out").

For the real number, benchmark a checkpoint on a different split with
`examples/rollout_rl_policy.py`. It loads the checkpoint's **actor only**
(critic/aux are training-time), rolls out **deterministically** (β=0, no noise),
and prints success rate + mean length + a status breakdown:

```bash
# checkpoint selection: pick the run/checkpoint with the best VAL success
python examples/rollout_rl_policy.py \
    --rl-run  output/rl_runs/rl_run1 \
    --sim-cfg examples/pretrain.yaml \
    --split   val --checkpoint best

# final number: report on TEST once, at the end
python examples/rollout_rl_policy.py \
    --rl-run  output/rl_runs/rl_run1 \
    --sim-cfg examples/pretrain.yaml \
    --split   test --num-scenes 100         # no --egl (it leaks ~85 MB/scene → OOM)

# WATCH it live in the pybullet GUI, stepping scenes BY HAND (best checkpoint):
python examples/rollout_rl_policy.py \
    --rl-run  output/rl_runs/rl_run7 \
    --sim-cfg examples/pretrain.yaml \
    --split   train --render --scene 0 --checkpoint best
#   focus the GUI window, then:  [n]/→ next scene   [p]/← prev   [r]eplay   [q]uit
#   the OMG goal grasp is drawn as a GREEN gripper wireframe each scene, so you can
#   watch the policy converge to (or miss) it. Swap --checkpoint best|last.
#   --render needs a display (run at the machine or over X-forwarding/VNC).
```

`--render` is **interactive and does not auto-advance** — you step through scenes with the keys above (headless, without `--render`, still auto-advances and prints a success rate). The eval uses the **training horizon** (`LOOP.rollout_max_steps` from the run's saved `rl_config.yaml`, else `cfg.RL_MAX_STEP`) so the rollout matches how the policy trained.

| flag | default | purpose |
|---|---|---|
| `--rl-run` | (required) | RL run dir (`output/rl_runs/<name>`) |
| `--sim-cfg` | (required) | simulator config, e.g. `examples/pretrain.yaml` |
| `--split` | `val` | held-out split to eval on (`val` for selection, `test` for the final number) |
| `--num-scenes` | all (5 with `--render`) | cap on scenes |
| `--scene` | `0` | first scene index to roll out — handy with `--render` |
| `--max-steps` | training horizon (`30`) | policy steps per episode. Also the **clock denominator**, so a value ≠ the trained horizon feeds an off-distribution clock (fine for eyeballing "how far with more/less time", not a fair number) |
| `--checkpoint` | `best` | `best` / `last` |
| `--render` | off | open the **pybullet GUI**, step scenes **interactively** ([n]/→ next, [p]/← prev, [r]eplay, [q]uit) with the OMG goal grasp drawn in green — no auto-advance (needs a display) |
| `--egl` | off | **avoid** — the EGL renderer leaks ~85 MB GPU/scene and OOMs a multi-scene eval; use the leak-free CPU renderer (omit) |

Protocol: **train** on the train split → **select** `best.pt`/run by
`--split val` → **report** once with `--split test`. (`best.pt` written during
training is chosen by the in-loop *train-scene* signal, so it's a heuristic; use
the val benchmark to make the real selection.)

### Two things to watch

- **Sparse reward + clock is the hard regime** (it's exactly what GA-DDPG avoids
  with `remain_timestep` conditioning of the *value*). The mitigations are baked
  in — BC warm-start, `mc_blend=0.5`, a small `λ`, MC returns. If success stalls,
  the first knob is `RL.mc_blend↑`; the structural escape hatch is to give the
  clock to the **critic only** (keeps the actor reactive).
- **The reward uses privileged, training-only signals.** `stable_grasp` reads
  handover-sim's contact/release/drop state; `proximity` and the gripper BCE label
  read the OMG grasp pose. Both are free in sim, unavailable on a real robot —
  fine, because they only shape *training*: the deployed policy closes from its
  **own learned logit** (a function of the point cloud + proprioception), never
  from a distance or a contact oracle. This is exactly why the paper learns a
  grasp classifier instead of "close at the last step."
- **Don't use `--egl` for any multi-scene run** (collect, train, or eval). pybullet's
  EGL renderer leaks ~85 MB of GPU memory per scene (mesh load/remove on each reset)
  and OOM-kills long runs; the CPU TinyRenderer is leak-free (~1.1 GB flat) and the
  depth-based cloud is nearly identical. Keep the renderer choice consistent across
  the whole pipeline.

> **Status.** Smoke-tested end to end — the RL core (buffer incl. the gripper-label
> fields + TD3+BC update + gripper BCE + `gripper_bc_weight=0` path + goal-aux +
> demo-npz round-trip), the real networks (forward, full-head warm-start
> equivalence, grad-flow), `collect_rl_demos.py` (full-playback → close-at-grasp
> `+1`s), and a live `train_rl.py --demos` run (warm-start → demo pool → warmup →
> train iters with finite `grip_loss` → checkpoint) all pass — under **both**
> `reward_mode=proximity` and `reward_mode=stable_grasp` (contact-hold: demos
> register genuine held grasps, online rollouts run the hold-and-check cleanly).
> The **hand-collision grasp filter** is also verified live (scenes 0–9: turns 5
> `HUMAN_CONTACT` scenes into 2 recovered + 3 skipped, collector `7/7`
> closed-at-grasp with zero collisions).
>
> **Result so far (rl_run7, 800 iters, ~7 h + full B fixes above): online RL did
> NOT solve the grasp — best held-out `eval_succ ≈ 0.05` (noise-level).** The
> chain of diagnosis:
> - The **BC base itself is broken in closed loop** — `dagger_iter_2_3` grasps
>   **2%** under its own benchmark (and 0% under the RL `stable_grasp` eval), even
>   though its per-step val metrics look fine (`val_pose_l1 0.167`, `gripper_acc
>   98.6%`). Classic covariate shift: **low per-step loss ≠ closed-loop
>   competence**, and a val-loss-selected checkpoint hides it.
> - It is **not** a scale/normalization bug (step-0 policy vs OMG delta ratio
>   `1.04`, cos `1.00`) and the policy does **reach** in the gross sense (`cos≈1.0`,
>   `d_grasp` falls 0.6→~0.1 m) — but it stalls/hovers ~0.06–0.11 m short and never
>   parks inside (0.02 m, 0.34 rad), so the gripper (correctly) never fires and
>   every episode TIMEOUTs.
>
> **ROOT CAUSE FOUND (2026-07-07), fix implemented: the DAgger/BC label field had a
> stationary attractor at the OMG standoff — 0.08 m short of the grasp.** The old
> per-step label was `plan[0]` of a **fresh** OMG replan whose horizon floored at
> `min_free=3`: OMG routes every plan through a standoff (`standoff_dist=0.08`)
> with the last 5 waypoints as the standoff→grasp reach, so `plan[0]` **never
> sampled the reach tail** and its magnitude Zeno-decayed to ~0 at the standoff
> (and pointed BACKWARD past it — exactly where the warm start hands over). The
> policy was literally taught to hover 0.08–0.11 m short; the observed hover
> distances match `standoff_dist` almost exactly. The Phase-1 DAgger collector
> (`collect_dagger_dataset.py`) used the SAME labeling (its `drop_past_standoff`
> workaround even stops recording at the standoff), which is what broke the BC
> base. So the "single-frame policy can't settle" conclusion was **confounded** —
> the settle was never demonstrated at on-policy states.
>
> **The fix (GA-DDPG-faithful, in `rollout_worker.py` + `rl_phase1.yaml`):**
> 1. **Plan-tracking labels**: OMG plans ONCE (committed plan); label at step `t` =
>    delta from the CURRENT state to `plan[t]` — the target advances through the
>    standoff INTO the grasp. DAgger replans re-fit only the plan TAIL (horizon =
>    remaining steps, no floor) inside `(dagger_min_step, len−dagger_tail_guard)`,
>    so reach labels stay committed. Bonus: OMG runs ~6×/episode instead of 30×.
> 2. **BC-dominated blend** (`alpha 1.0→0.1`): GA-DDPG's actor loss is
>    `0.9·BC + 0.1·(−minQ)` — BC leads throughout; ours was effectively inverted.
>
> Probe-verified with the rl_run7 (hovering) actor: at its stall states the new
> labels are **13–16 cm pointing at the grasp** (cos ≈ +1.0) where the old ones
> said "stay"; a warmup-25 episode walks the reach tail to `min_pos=0.016 m /
> min_rot=0.014 rad` with healthy ~2.6 cm labels.
>
> **rl_run8 (fixed labels + `alpha 0.1`, ~260 iters): PARTIAL WIN, exposed the next
> wall.** What the fix delivered: the standoff hover is gone, `eval_min_rot`
> trends down `0.73→0.45` (orientation now learned; was pinned in run7), and
> `roll_min_pos` hits **0.007 m** — when the curriculum hands it an on-plan
> near-grasp pose, the policy *settles cleanly into the grasp*. The blend is stable
> (`q_pi` tracks `q_mean`, no divergence). **But from-scratch `eval_min_pos` still
> plateaus ~0.17–0.20 m, all TIMEOUT, `eval_succ=0`.** Per-step probe (ei=0):
> the policy commands a near-constant ~3 cm/step while the plan-tracking label
> grows to demand catch-up, so it **undershoots ~15–25 %/step, compounding**, and
> arrives at the reach-tail boundary (~0.08 m) *off-plan*. A horizon sweep on the
> same actor (H=30/45/60) improves reach `0.11→0.066 m` then **saturates** — it's
> not pure runway; the policy **stalls ~0.08 m** because the reach-tail behavior
> was only trained from expert-visited *on-plan* states (covariate shift localized
> to the last 8 cm), and there's no reward anywhere the from-scratch policy reaches
> (`buf_pos≈0.008`) so the weak PG can't pull it in.
>
> **rl_run9 (annealed reverse curriculum): BREAKTHROUGH — 0 → 30%.** rl_run8's
> `expert_initial` was *uniform* from iter 0 and never sequenced the mastery; run9
> slides the takeover window from a tight band at the grasp (`ei 22–28` @ it 0) out
> to near-scratch (`ei 0–2` by it 550), so the endgame is mastered and rewarded
> **first** and the critic's high-Q region grows outward for PG to exploit as
> harder starts are introduced. Result: **best `eval_succ 0.30` @ iter 700**
> (`min_pos 0.057`, `close 0.35`), tracking `ei_hi` almost perfectly — reach broke
> the 0.08 m stall (→0.05–0.07 m), closes fire. **`best.pt` (iter 700, 30% train)
> is the current policy.** It peaked at anneal-completion then declined; a
> best-vs-last head-to-head (60 scenes) showed last.pt at **4%** — the *approach*
> stays clean but the **gripper drifts open** (`grip_logit` 14→33) and stops
> closing. Note: 2000 iters ≈ **15 h** (not 7).
>
> **rl_run10 (PM loss + `alpha 0.25` + `bc_weight 80` + slower/wider curriculum):
> NEGATIVE, reverted.** Worse on all three (`eval_succ 0.15`, `min_pos 0.072`,
> `min_rot 0.378`). Root causes: **the PM-loss premise was backwards** — at our
> sub-0.34-rad deltas the normalized `smooth_l1` already weights rotation ~6×
> *more* than PM (small rotation std), so PM *regressed* rotation; the wider
> curriculum band `[0..8]` under-practiced the from-scratch eval; `alpha 0.25`
> didn't help reach. Process lesson: **changed four things at once → couldn't
> isolate.** Config reverted to run9's exact values.
>
> **rl_run11 (IMPLEMENTED, ready): one isolated change on top of run9 — gripper
> BCE label smoothing** (`gripper_label_smooth 0.1`) to fix the open-drift. The
> "open" label dominates every batch, so plain BCE saturates the logit open until
> its gradient vanishes and the gripper is stuck; smoothing gives it a finite
> anchor (`logit(0.9)≈+2.2`) so it stays responsive. Smoke: the logit drops from
> the warm-start ~14 to **+2.9 and declining** (not climbing to 33), and the
> gripper **fires** (`close 0.50` on warm-started episodes). Touches only the
> gripper BCE — clean single-variable vs run9. **Watch:** `grip_logit` staying
> ~2–3 (not climbing) and `eval_close` holding up *late* (run9 decayed to 4%).
> Caveat: the gripper shares the actor trunk with the pose head, so if it still
> drifts the next step is a separate gripper head. **Remaining blocker after the
> gripper: reach 0.05→0.02 m.** Watch any checkpoint live with
> `rollout_rl_policy.py --render`.

## Plotting the training curves (on the cluster)

`examples/plot_rl_run.py <run_dir>` reads the run's `log.csv` and writes
`<run_dir>/curves.png`. On DelftBlue there are two gotchas: the login node's bare
`python` is the system Python 2 (the script needs 3.7+), and `conda` isn't on the
PATH unless you source its hook — so call the env's Python by full path. The login
node is also headless, so force the Agg backend (no display):

```bash
# 1. headless plotting backend (no display on the login node)
export MPLBACKEND=Agg
# 2. the env's Python by full path (bare `python` is system Python 2)
export PLOT_PY=/home/pradyunsharma/.conda/envs/pch2r_dev/bin/python
# 3. run the plotter
$PLOT_PY examples/plot_rl_run.py output/rl_runs/rl_run13
# -> wrote output/rl_runs/rl_run13/curves.png   (open it in the VS Code file explorer)
```

Swap `rl_run13` for any run. Optional `~/.bashrc` helper so it's one word:

```bash
plotrun() { MPLBACKEND=Agg /home/pradyunsharma/.conda/envs/pch2r_dev/bin/python \
    ~/h2r/handover-sim2real/examples/plot_rl_run.py "output/rl_runs/$1"; }
#   usage (from the repo root):  plotrun rl_run13
```
