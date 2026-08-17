# Phase 4 — DAgger

`examples/train_dagger_phase4.py` — Algorithm 3.1 of Ross et al. (2011) as one
Python loop. Sim, OMG planner and model stay resident across iterations.

Design notes: [`docs/thesis_phase4_dagger.md`](docs/thesis_phase4_dagger.md).
Everything else (envf setup, Phases 1–3, cluster sync): [`README_MY.md`](README_MY.md).

## Setup

```bash
module load miniconda3                                # cluster only
source "$(conda info --base)/etc/profile.d/conda.sh"  # `conda activate` alone won't work
conda activate pch2r_dev
export GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner
```

Everything needs a GPU (`OMG-Planner/omg/config.py` calls `.cuda()` at import).
Exceptions: `filter_demos.py` and the plotting scripts.

## Runbook

Each stage consumes a file the previous one wrote. Two pin rules → **two parallel
dataset families**; pick one and keep it consistent all the way through. Every
file below is a valid `--dataset` / `--grasp-pin-table` value.

| | rule A — `furthest_from_hand` | rule B — `omg` |
|---|---|---|
| pin table, train | `output/grasp_pin_table_train.json` | `output/grasp_pin_table_train_omg.json` |
| pin table, val | `output/grasp_pin_table_val.json` | `output/grasp_pin_table_val_omg.json` |
| train, raw | `output/bc_dataset/train_pinned.h5` (623) | `output/bc_dataset/train_pinned_omg.h5` (623) |
| train, filtered | `output/bc_dataset/train_pinned_ok.h5` (497) | `output/bc_dataset/train_pinned_omg_ok.h5` (472) |
| train, dropped scenes | `output/bc_dataset/train_pinned_ok.json` (126) | `output/bc_dataset/train_pinned_omg_ok.json` (151) |
| val, raw | `output/bc_dataset/val_pinned.h5` (29) | `output/bc_dataset/val_pinned_omg.h5` (29) |
| val, filtered | `output/bc_dataset/val_pinned_ok.h5` (21) | `output/bc_dataset/val_pinned_omg_ok.h5` (17) |
| val, dropped scenes | `output/bc_dataset/val_pinned_ok.json` (8) | `output/bc_dataset/val_pinned_omg_ok.json` (12) |

Counts in brackets are episodes (or dropped scenes for the `.json`). **Raw
contains failed demonstrations** — `train_pinned_omg.h5` episode 0 is scene 0,
which is a drop. Filtered is what training uses. `--episode` indexes the *file*,
so after filtering it no longer equals `scene_idx` (episode 0 of
`train_pinned_omg_ok.h5` is scene 1); the true scene is on each episode group's
`scene_idx` attr.

### 1. Pin one grasp per scene

```bash
python examples/build_grasp_pin_table.py --cfg-file examples/configs/dagger_phase4.yaml --split train --mode furthest_from_hand --tol 0.01 --out output/grasp_pin_table_train.json
python examples/build_grasp_pin_table.py --cfg-file examples/configs/dagger_phase4.yaml --split train --mode omg --out output/grasp_pin_table_train_omg.json

# SLURM
sbatch --export=ALL,SPLIT=train,MODE=furthest_from_hand,OUT=output/grasp_pin_table_train.json examples/slurm/build_pin_table.sbatch
```

Repeat with `--split val`. `--tol 0.01` only matters for rule A (at `tol 0` the
argmax moves the grasp 8.5 cm for 0.1 mm of clearance). Rebuild whenever
`valid_grasp_dict_path` or `hand_collision_filter` changes.

### 2. Collect the demonstrations

```bash
python examples/collect_bc_dataset.py --cfg-file examples/pretrain.yaml --split train --valid-grasp-dict examples/valid_grasp_dict_005.pkl --grasp-pin-table output/grasp_pin_table_train.json --output output/bc_dataset/train_pinned.h5

# SLURM, chained on the pin table
PIN=$(sbatch --parsable --export=ALL,SPLIT=train examples/slurm/build_pin_table.sbatch) && sbatch --dependency=afterok:$PIN --export=ALL,SPLIT=train,PIN=output/grasp_pin_table_train.json examples/slurm/collect_bc_demos.sbatch
```

`--cfg-file` must equal `SIM.cfg_file` in the dagger yaml.

### 3. Drop the failed demonstrations

~20% of episodes fail (plan clips the object on the lateral approach). An episode
has no CLOSE label iff it failed — that is the filter.

```bash
python examples/filter_demos.py --in output/bc_dataset/train_pinned.h5 --out output/bc_dataset/train_pinned_ok.h5
# run 1: 497 kept, 126 dropped (20.2%).  --dry-run for counts only

# sanity: does the expert close where the config says?
python examples/analyze_close_labels.py --demos output/bc_dataset/train_pinned.h5 --pin-table output/grasp_pin_table_train.json
# median 4.6 mm / 0.0003 rad; 99.4% inside both thresholds
```

Writes `*_ok.h5` **and** `*_ok.json` (the excluded scene list — without it DAgger
re-attempts the broken scenes every iteration).

> Dropping these makes the remainder easier by construction; a success rate
> measured afterwards is not comparable to one on the full split.

### 4. Wire the outputs into the config

All four move together:

```yaml
SIM:
  grasp_pin_table: output/grasp_pin_table_train.json
  exclude_scenes:  output/bc_dataset/train_pinned_ok.json
TRAIN:
  base_train_h5:   output/bc_dataset/train_pinned_ok.h5
  val_h5:          output/bc_dataset/val_pinned.h5
```

`exclude_scenes` is subtracted from the collection *and* eval pools. Swapping the
pin table without rebuilding the dataset silently produces nonsense.

### 5. Look at the demonstrations

Needs a display — run on the PC. **Dataset and pin table must be the matching
pair**, or the green gripper is drawn at a grasp the episode never aimed at.

```bash
# rule A, filtered — what training actually sees
python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train_pinned_ok.h5 --grasp-pin-table output/grasp_pin_table_train.json --mode replay --cfg-file examples/pretrain.yaml --show-goal-grasp --episode 0

# rule B, filtered
python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train_pinned_omg_ok.h5 --grasp-pin-table output/grasp_pin_table_train_omg.json --mode replay --cfg-file examples/pretrain.yaml --show-goal-grasp --episode 0

# swap in the raw *.h5 to see the ~20-24% that failed (scene 0 is one of them)

# which file episode maps to which scene
python -c "
import h5py
with h5py.File('output/bc_dataset/train_pinned_omg_ok.h5','r') as f:
    for i,k in enumerate(sorted(f)): print(i, '-> scene', f[k].attrs['scene_idx'])
"
```

Green = pinned grasp, cyan = pre-grasp standoff.

| option | values | default | what it does |
|---|---|---|---|
| `--dataset` | any of the 8 HDF5s in the family table above; also an RL demo pool (`.h5`/`.npz`) from `collect_rl_demos.py` | *required* | what to visualise |
| `--grasp-pin-table` | the 4 pin tables in the family table above | `None` | per-scene committed grasp; must match the dataset's split *and* rule |
| `--mode` | `static` \| `replay` | `static` | `static` = matplotlib, no GPU; `replay` = PyBullet sim |
| `--cfg-file` | `examples/pretrain.yaml` (or `pretrain_multicam_wlr.yaml` for run 5) | `None` | simulator yaml — required for `--mode replay` |
| `--episode` | `0 … num_episodes-1` | random | index into the **file** (≠ `scene_idx` after filtering) |
| `--replay-source` | `states` \| `omg` | `states` | `states` = drive sim through recorded states (use for DAgger data); `omg` = re-plan the expert |
| `--show-goal-grasp` | flag | off | green goal grasp `traj[-1]` + cyan standoff `traj[-5]`; runs OMG once |
| `--show-grasp-set` | flag | off | also draw all candidates (thin grey) |
| `--max-grasp-set` | int | `40` | cap on candidates drawn |
| `--show-expert-arrows` | flag | off | per-step Δpos shaft (green open, red close) + Δeuler triad |
| `--arrow-scale` | float | `3.0` | arrow exaggeration (a step is only ~3-4 cm) |
| `--valid-grasp-dict` | path \| `''` \| `none` | `examples/valid_grasp_dict_005.pkl` | grasp dict OMG loads, so the overlay matches collection |
| `--seed` | int | `None` | RNG seed |

### 6. Train

```bash
python examples/train_dagger_phase4.py --cfg-file examples/configs/dagger_phase4.yaml --run-name dagger4_run2

# SLURM
sbatch --time=14:00:00 --export=ALL,RUN=dagger4_run2 examples/slurm/train_dagger_phase4.sbatch
```

Re-run the identical command to **resume** (done iterations skipped, untrained
collection reused, interrupted fit continues from `last.pt`). Omit `BASE=`
whenever the dataset changed.

Run dir `output/dagger_runs/<name>/`: `state.json`, `dagger_log.csv`,
`data/dagger_iter_NN.h5`, `iters/iter_NN/`, `best/`, `last/`. The last two are
standalone run dirs — `rollout_bc_policy.py` loads either directly
(`rollout_act_policy.py` instead, if `train_cfg` pointed at `act_phase2.yaml`).

### 6b. Roll out a trained checkpoint

```bash
# WATCH IT — interactive PyBullet.  R = re-roll  N = next  Q = quit
python examples/rollout_bc_policy.py --run-dir output/dagger_runs/dagger4_run7/best --cfg-file examples/pretrain.yaml --max-steps 50 --scenes-from-run output/dagger_runs/dagger4_run7 --show-goal-grasp --grasp-pin-table output/grasp_pin_table_train_omg.json

# headless success rate over the 472 scenes the run trained on
python examples/rollout_bc_policy.py --run-dir output/dagger_runs/dagger4_run7/best --cfg-file examples/pretrain.yaml --max-steps 50 --scenes-from-run output/dagger_runs/dagger4_run7 --benchmark --no-render --egl

# the 151 scenes whose expert demo FAILED — what the policy was never taught
python examples/rollout_bc_policy.py --run-dir output/dagger_runs/dagger4_run7/best --cfg-file examples/pretrain.yaml --max-steps 50 --scenes output/bc_dataset/train_pinned_omg_ok.json --benchmark --no-render --egl
```

- **`--scenes-from-run`** rebuilds the run's own pool (pin-table keys −
  `exclude_scenes`, same as `dagger/setup.py`). Without it the sweep is
  `range(720)` and includes the 97 scenes OMG cannot plan for and the 151 whose
  expert failed — neither of which the policy ever saw.
- **`--max-steps` must match `DAGGER.max_steps`** (50); the default 30 cuts the
  approach off.
- **`--grasp-pin-table` must be the run's `SIM.grasp_pin_table`.** Overlay only,
  never the policy — but rules A/B differ by 2.2 cm on scene 0, above
  `close_pos_thresh`, so the wrong one makes a correct rollout look like a miss.
- `--show-goal-grasp` is a no-op under `--no-render`.
- The benchmark number does **not** reproduce the in-loop `success_rate`
  (different path: no `stable_grasp` hold, no pinning). Use `eval_dagger_run.py`
  for a comparable figure. Neither is held out if `EVAL.holdout: false`.

| option | values | default | what it does |
|---|---|---|---|
| `--run-dir` | `<run>/best`, `<run>/last`, or any `iters/iter_NN` | *required* | must contain `config.yaml`, `normalization.npz`, `checkpoints/best.pt` |
| `--cfg-file` | the run's `SIM.cfg_file` | *required* | simulator yaml |
| `--scenes-from-run` | a Phase-4 run dir or its `config.yaml` | `None` | restrict to that run's training pool; mutually exclusive with `--scenes` |
| `--scenes` | `3,7,12`, or a path to a JSON list of ints | `None` | explicit scene set (e.g. a `*_ok.json` = the dropped scenes) |
| `--scene` | any id in the pool | `0` | where interactive mode starts; falls back to the pool's first |
| `--max-steps` | int | `30` | policy steps — set to the run's `DAGGER.max_steps` |
| `--benchmark` | flag | off | sweep scenes headless, print success / grasp / dist |
| `--num-scenes` | int | all | cap on how many of the selected scenes to sweep |
| `--no-render` | flag | off | headless |
| `--egl` | flag | off | GPU offscreen renderer when headless (else software) |
| `--show-goal-grasp` | flag | off | green gripper at the goal grasp |
| `--show-grasp-set` | flag | off | also draw all candidates (faint grey) |
| `--grasp-pin-table` | the run's `SIM.grasp_pin_table` | `None` | which grasp the overlay draws |
| `--device` | `cuda` \| `cpu` | `cuda` | PointNet++ needs CUDA |

If `best/` arrived from the cluster with only `best.pt`, copy `config.yaml` and
`normalization.npz` in from the winning `iters/iter_NN/` (`state.json` names it
under `best.iter`) — that is exactly what `export_run_dir` writes.

### 6c. Faster on the cluster

Eval is not on the critical path — the loop never waits on it. Split it off:

```bash
# trainer, eval skipped
sbatch --time=12:00:00 --export=ALL,RUN=dagger4_run6,CFG=examples/configs/dagger_phase4_omg_nojoint.yaml,EXTRA=--no-eval examples/slurm/train_dagger_phase4.sbatch

# scorer alongside — polls, exits when the trainer stops
sbatch --export=ALL,RUN=dagger4_run6 examples/slurm/eval_dagger_run.sbatch
```

`eval_dagger_run.py` rebuilds everything from the run's own `config.yaml`, so its
numbers match the in-loop ones; it writes `<run>/eval_log.csv`, skips scored
iterations, and is safe to resubmit mid-training. `plot_dagger_run.py` splices it
in automatically. `WATCH=0,PUBLISH=1` runs it once at the end and exports
`<run>/best` — never while the trainer is running.

Collection parallelises exactly (π̂ᵢ frozen, scenes drawn up front). The sbatch
requests 22 CPUs and derives `WORKERS = cpus - 2`; **20 is the number to want** —
it divides `episodes_per_iter: 100` evenly. Every worker needs a CUDA context
(PointNet++ `furthest_point_sample` has no CPU kernel), but collection is
CPU-bound, so a second GPU would not help.

| 15-iteration run | projected | speedup |
|---|---|---|
| as-is (4 CPUs, serial, in-loop eval) | 8.92 h | 1.00× |
| eval in its own job | 7.26 h | 1.23× |
| + 20 collection workers | 4.25 h | 2.10× |

The rest is the Follow-The-Leader refit, which re-fits the whole aggregate every
iteration by design.

### 7. Read the results

```bash
# per ITERATION — did it learn? safe mid-training
python examples/plot_dagger_run.py output/dagger_runs/dagger4_run2

# per EPOCH, all refits — is each fit healthy, is the aggregate getting harder?
python examples/plot_dagger_epochs.py output/dagger_runs/dagger4_run2

# one iteration in full Phase-1 depth (every iters/iter_NN/ is a complete BC run dir)
python examples/analyze_bc_run.py --run-dir output/dagger_runs/dagger4_run2/iters/iter_05 --mode both

# per-epoch CSVs aren't in the usual sync (they sit beside the checkpoints)
rsync -avP --include='*/' --include='log.csv' --exclude='*' pradyunsharma@login.delftblue.tudelft.nl:/home/pradyunsharma/h2r/handover-sim2real/output/dagger_runs/dagger4_run2/iters/ output/dagger_runs/dagger4_run2/iters/
```

Columns that carry the most weight:

| column | read it as |
|---|---|
| `close_rate` ≥ `near_rate`, `close` ≥ `grasp` ≥ `success` | nested — the *gaps* localise the failure |
| `eval_min_pos` / `eval_min_rot` | closest EE→grasp over the episode; defined even at 0 success. Rotation binds first |
| `chance_rate` | ever reached a pose where closing would be correct (both tolerances, same step) |
| `close_success_rate` | success given closed — when it decides to grasp, is it right? |
| `missed_rate` / `miss_given_chance` | had the chance, came away with nothing |
| `mean_label_pos` | standoff-stall detector — collapse away from `ee_step` is the only direct signal |
| `goal_switch` | must be 0 with a pin table — grasp moving *within* an episode |
| `grasp_mismatch` | must be 0 — grasp moving *between* iterations (`goal_switch` can't see this) |
| `dropped_tail` | should be 0 — nonzero means the expert still collides on scenes stage 3 missed |
| `D_dagger_frac` vs `success_rate` | separates "DAgger helps" from "more data helps" |

In `epoch_curves.png` the sawtooth is just the optimizer restarting under
Follow-The-Leader — read the envelope. Compare loss *floors*, not endpoints: the
base fit gets `base_epochs` (100), refits get `iter_epochs` (25).

## DART

Off unless `dart_ratio > 0`. On a fraction of approach steps the executed action
is replaced by a random task-space jump; the following steps carry the expert's
recovery. Phase 4 stores only `(state, π*(state))` and replans every step, so
unlike GA-DDPG (`core/ddpg.py`) and Phase 3 (`rl/td3bc_trainer.py`) there is no
perturbed row to mask and no tail to splice.

| key | default | what it does |
|---|---|---|
| `dart_ratio` | `0.0` | per-step probability; `0` draws no random number, so a DART-free run is bit-identical to a pre-DART one |
| `dart_max_dist` | `0.20` | upper edge of the trigger band, metres from the standoff |
| `dart_pos_mag` | `0.04` | ± metres per axis (= `ee_step`) |
| `dart_rot_mag` | `0.2` | ± radians per axis |

Trigger is **distance to the standoff**, not a step index (every step replans, so
step count doesn't track progress). Lower edge is `reach_commit_dist` — jolting
below it would destroy the close label. A jolt never steals a step where the
policy asked to close, so `policy_closed` / `c_policy_close` stay comparable.

Watch `dart` (jolts fired — 0 at a nonzero ratio means the band is never
entered), `dart_env_done` as a fraction of it (climbing = `dart_pos_mag` is
knocking the object out of the hand), and `c_max_steps`.

## Config reference

`EVAL.success_mode`, both imported from `rl/rollout_worker.py`:

- **`stable_grasp`** (default) — hold shut `hold_steps`, then require secured:
  release handshake fired ∧ not dropped ∧ no human contact. No OMG calls.
- **`proximity`** — EE within (`close_pos_thresh`, `close_rot_thresh`) at the
  close; scores label agreement, not task outcome.

The benchmark's own `EpisodeStatus.SUCCESS` can never fire — Phase 4 ends the
episode at the close, there is no carry-to-goal.

Label ordering inside a step: capture `(pc, rs)` → replan → proximity check →
**write label** → query policy → execute whoever the β coin picked → step sim.
The learner's action structurally cannot enter D. Consequences:

- A premature policy close is **relabelled** `[expert_delta, OPEN]`, not copied.
  `stop_on_policy_close: false` then keeps the episode rolling; only the gripper
  bit is overridden, so the state distribution stays on-policy.
- On a collision, if the *policy* drove the step the pair is kept (most valuable
  kind); if the *expert* drove it the colliding action **is** the label — that
  one pair is dropped, counted by `dropped_tail`.

## Run variants

| run | config | pinned grasp | cameras | robot MLP input |
|---|---|---|---|---|
| 2 | `dagger_phase4.yaml` | furthest from hand | wrist | 26 dims |
| 3 | `dagger_phase4_omg.yaml` | OMG's own pick | wrist | 26 dims |
| 4 | `dagger_phase4_omg_nojoint.yaml` | OMG's own pick | wrist | **8 dims** |
| 5 | `dagger_phase4_omg_nojoint_multicam.yaml` | OMG's own pick | **wrist+left+right** | 8 dims |

- **2 vs 3** — pin rule. The two rules disagree on 63% of train scenes.
- **3 vs 4** — robot-state width (`MODEL.drop_joint_state` strips
  `joint_pos(9)+joint_vel(9)`, leaving `[ee_pos(3), ee_orn_wxyz(4), gripper(1)]`).
  Do **not** pass `BASE=`: a 26-dim checkpoint will not load into an 8-dim model.
- **4 vs 5** — cameras. The only variant needing new data.

Runs 6–10 branch off run 4 as a 2×2 with two controls:

| run | config | vs | β schedule | iters | `dart_ratio` |
|---|---|---|---|---|---|
| 4 | `dagger_phase4_omg_nojoint.yaml` | — | linear 0.75→0.10 | 15 | off (control) |
| 6 | `dagger_phase4_dart02.yaml` | 4 | linear 0.75→0.10 | 15 | **0.2** |
| 7 | `dagger_phase4_dart05.yaml` | 4 | linear 0.75→0.10 | 15 | **0.5** |
| 8 | `dagger_phase4_beta_ext.yaml` | 4 | **piecewise 1.0→0.5→0.3** | **25** | off (control) |
| 9 | `dagger_phase4_beta_ext_dart02.yaml` | 8 | piecewise 1.0→0.5→0.3 | 25 | **0.2** |
| 10 | `dagger_phase4_beta_ext_dart05.yaml` | 8 | piecewise 1.0→0.5→0.3 | 25 | **0.5** |

```bash
# runs 6, 7 — ~4.8 h each, eval inline
sbatch --time=08:00:00 --export=ALL,RUN=dagger4_run6,CFG=examples/configs/dagger_phase4_dart02.yaml examples/slurm/train_dagger_phase4.sbatch
sbatch --time=08:00:00 --export=ALL,RUN=dagger4_run7,CFG=examples/configs/dagger_phase4_dart05.yaml examples/slurm/train_dagger_phase4.sbatch

# runs 8, 9, 10 — ~10 h each, eval inline
sbatch --time=16:00:00 --export=ALL,RUN=dagger4_run8,CFG=examples/configs/dagger_phase4_beta_ext.yaml examples/slurm/train_dagger_phase4.sbatch
sbatch --time=16:00:00 --export=ALL,RUN=dagger4_run9,CFG=examples/configs/dagger_phase4_beta_ext_dart02.yaml examples/slurm/train_dagger_phase4.sbatch
sbatch --time=16:00:00 --export=ALL,RUN=dagger4_run10,CFG=examples/configs/dagger_phase4_beta_ext_dart05.yaml examples/slurm/train_dagger_phase4.sbatch

# share run 4's base fit across all five (identical iteration 0, saves ~14 min each)
ls output/dagger_runs/dagger4_run4/iters/iter_00/checkpoints/best.pt
#   ...,BASE=output/dagger_runs/dagger4_run4/iters/iter_00
```

`piecewise` β = two linear segments joined at `beta_knee` as a fraction of
`num_iters` (25 iters → join at 16: 1.00→0.50 over 1–16, 0.50→0.30 over 17–25).
Costs: iteration 1 at β=1.00 is pure expert, so the first on-policy data arrives
at iteration 2; and Follow-The-Leader makes 25 iterations **7.1 h of refits**
against run 4's 3.0 h — 2.4×, not 1.7×.

### Run 5 — re-collect under three cameras

`SIM.cfg_file` becomes `examples/pretrain_multicam_wlr.yaml`.
`COMPUTE_ROBOT_POINT_STATE` stays **off** so classes stay `[object, hand]`,
`pc_channels` stays 5, and the CVPR2023 warm start still transfers 68/68 tensors.
A wrist-only base dataset cannot be aggregated with 3-camera DAgger data.

```bash
python examples/viz_cameras_pybullet.py --sim-cfg examples/pretrain_multicam_wlr.yaml

sbatch --export=ALL,SIM_CFG=examples/pretrain_multicam_wlr.yaml,SPLIT=train,OUT=output/bc_dataset/train_pinned_omg_wlr.h5,PIN=output/grasp_pin_table_train_omg.json examples/slurm/collect_bc_demos.sbatch
sbatch --export=ALL,SIM_CFG=examples/pretrain_multicam_wlr.yaml,SPLIT=val,OUT=output/bc_dataset/val_pinned_omg_wlr.h5,PIN=output/grasp_pin_table_val_omg.json examples/slurm/collect_bc_demos.sbatch

python examples/filter_demos.py --in output/bc_dataset/train_pinned_omg_wlr.h5 --out output/bc_dataset/train_pinned_omg_wlr_ok.h5 --scenes-out output/bc_dataset/train_pinned_omg_wlr_ok.json
python examples/filter_demos.py --in output/bc_dataset/val_pinned_omg_wlr.h5 --out output/bc_dataset/val_pinned_omg_wlr_ok.h5

sbatch --time=12:00:00 --export=ALL,RUN=dagger4_run5,CFG=examples/configs/dagger_phase4_omg_nojoint_multicam.yaml examples/slurm/train_dagger_phase4.sbatch
```

Free consistency check: `train_pinned_omg_wlr_ok.json` must list the **same 151
scenes** as `train_pinned_omg_ok.json`. A different count means something other
than the cameras changed.

## Harvesting a scratch run's logs back into the repo

Runs now write to `/scratch` (`OUT_ROOT` in `examples/slurm/train_dagger_phase4.sbatch`),
because `/home` is a hard 30 GB quota and fills silently — a full `/home` kills a
job with exit code 6 and **no traceback**. The consequence is that a run's logs no
longer appear in the repo on their own: `output/dagger_runs/<RUN>/` does not exist
here until you copy the small files back. `examples/harvest_run.py` does that.

It is a **manual snapshot, not a symlink or a live view** — nothing in the repo
updates by itself. Run it before every `git add`, and again once a run finishes,
or you will commit a half-finished log.

```bash
# one run
python examples/harvest_run.py --run-dir /scratch/$USER/handover-sim2real/output/dagger_runs/dagger4_run15

# every run under scratch (dagger_runs, rl_runs, bc_runs) — the usual form
python examples/harvest_run.py --all --scratch-root /scratch/$USER/handover-sim2real

# preview without writing
python examples/harvest_run.py --all --scratch-root /scratch/$USER/handover-sim2real --dry-run

git add output/dagger_runs/dagger4_run15
git commit -m "run15 logs"
```

Each run mirrors into `output/<kind>/<RUN>/` at the same path it would have had if
written in-repo, so `iters/iter_00/log.csv` lands where you expect. Typical cost is
**~600 KB per run**; a full `--all` sweep copies ~1.8 MB and leaves ~26 GB of
checkpoints and replay data on scratch.

What it copies is a strict **allow-list**: `dagger_log.csv`, `eval_log.csv`,
`log.csv`, `config.yaml`, `state.json`, `grasp_registry.json`, `source.txt`,
`iters/*/log.csv`, `iters/*/config.yaml`, and all `*.png`. Checkpoints (`.pt`) and
replay data (`.h5`) can never be picked up — a deny-list raises rather than copies,
so widening the patterns carelessly fails loudly instead of quietly filling `/home`.

Safe to run **while a job is still going** (it only ever writes into the repo, never
into the run directory) and idempotent (size + mtime compare), so re-running just
extends the CSVs.

The `output/.gitignore` rules `!dagger_runs/*/*.csv`, `*.json`, `*.yaml`, `*.png`
are negations that explicitly un-ignore these files, so harvested metadata is
trackable with no `.gitignore` change. Note `git check-ignore -v` prints the
matching rule even when it is a negation — use the exit code (`-q`; 0 = ignored)
if you need to test a path.

Checkpoints stay on scratch by design. Pull those to the PC separately — see the
sync section in `README_MY.md`. **`/scratch` is periodically purged by age**, so
harvesting is what makes a run's record survive.
