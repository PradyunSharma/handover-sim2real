# Phase 5 — grasp conditioning and regrasping

Phase 4 pins one grasp per scene and trains a policy that never goes there.
Across runs 4–16, `near_rate` — closed within 2 cm / 0.34 rad of the pinned grasp
— sat at 0.00–0.08 while `success_rate` reached 0.80. The policy learned to come
away with the object from wherever it happened to arrive, and
[`docs/run_index.md`](docs/run_index.md) already named the cause: *the observation
never says which orientation to arrive at.* Run 13's auxiliary head was the weak
test of that hypothesis, and it did not move `near_rate` either.

Phase 5 is the strong version. The goal grasp becomes an **input** to the policy,
and each scene carries **four** well-separated pinned grasps instead of one, so a
failed handover can be retried under a different grasp. That is regrasping — and
it is only meaningful if the policy's behaviour actually changes when the
commanded grasp changes, which is why `cond_track` is the column to read first.

Phase 4 is untouched: `handover_sim2real/dagger5/`, `handover_sim2real/bc5/` and
`examples/*_p5.py` / `*_multi.py` are a full fork, so all 21 recorded Phase-4 runs
stay byte-reproducible. Everything else (env setup, Phases 1–3, cluster sync):
[`README_MY.md`](README_MY.md). Phase 4 itself: [`README_PHASE4.md`](README_PHASE4.md).

## Setup

```bash
module load miniconda3                                # cluster only
source "$(conda info --base)/etc/profile.d/conda.sh"  # `conda activate` alone won't work
conda activate pch2r_dev
export GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner
```

Everything needs a GPU (`OMG-Planner/omg/config.py` calls `.cuda()` at import, and
PointNet++ has no CPU kernel). Exceptions: `select_pinned_grasps.py`,
`analyze_grasp_separation.py` and the plotters.

## The two design decisions worth understanding first

**Conditioning is load-bearing, not a feature.** With four demonstrations per
scene, the same `(point cloud, robot state)` carries four different expert
labels. An unconditioned regression can only predict their mean — which is a
valid action for none of the four, and strictly worse than a Phase-4 policy.
Telling the policy which grasp it is being asked to reach is what makes the
mapping single-valued again. `train_dagger_phase5.py` refuses to start if the pin
table holds several grasps and `MODEL.grasp_cond` is off.

The corollary is that the network *ignoring* the conditioning is a regression
risk, not a neutral outcome, and that is what `cond_track` exists to detect
before you spend another run on it.

**Flip twins are the trap in "pick four grasps as far apart as possible".**
`omg/planner.py`'s `augment_flip_grasp` appends, for every grasp, a duplicate
rotated π about `panda_joint7` — π about the gripper's own approach axis, under
which a parallel-jaw gripper is symmetric. The twin is *the same physical grasp*
sitting at the maximum possible rotation distance, so a naive max-min selector
picks twins first and returns four poses that are really two, with contradictory
rotation labels for one target. `dagger5/grasp_select.py` quotients that out: the
metric is the mean displacement of the gripper's six control points minimised
over the two-element flip group, so a twin measures exactly 0 m and can never be
chosen as a "distant" grasp.

## Runbook

Four stages, each consuming what the previous one wrote. The last stage's three
outputs move together — swapping one without the others silently produces
nonsense.

### 1. Build the candidate table (K = 8 per scene)

```bash
python examples/build_grasp_pin_table_multi.py \
    --cfg-file examples/configs/dagger_phase5_run1.yaml \
    --split train --k 8 --sep-floor 0.02 \
    --out output/grasp_cand_table_train_p5.json

# SLURM
sbatch --export=ALL,SPLIT=train,OUT=output/grasp_cand_table_train_p5.json \
    examples/slurm/build_pin_table_multi.sbatch
```

Repeat with `--split val`. Slot 0 is seeded with `env.get_omg_goal_idx()`, so it
is byte-identical to a Phase-4 `--mode omg` pin and `succ_g0` stays comparable
with run 16; slots 1–7 are greedy farthest-point sampling under the
flip-invariant metric.

**K = 8 and not 4 on purpose.** About a quarter of pinned demonstrations fail —
the plan clips the object on the lateral approach — so demanding four of four
would keep only ~0.76⁴ ≈ 33 % of scenes. Over-provisioning to eight and choosing
the surviving four in stage 3 keeps roughly 90 %, for one extra collection pass.

Read the summary it prints. "fewer than 4 distinct" counts scenes that cannot
survive stage 3 whatever collection does, and is the first honest estimate of the
Phase-5 scene budget. Starting point: of 720 s0/train scenes, 623 plan at all
(`024_bowl` fails on all 40, `037_scissors` on 19 of 40) and 592 have a goal set
of at least four.

### 2. Collect one demonstration per (scene, candidate)

```bash
python examples/collect_bc_dataset_multi.py \
    --cfg-file examples/pretrain_multicam_wr.yaml \
    --split train \
    --valid-grasp-dict examples/valid_grasp_dict_005.pkl \
    --grasp-pin-table output/grasp_cand_table_train_p5.json \
    --output output/bc_dataset/train_p5_k8.h5

# SLURM, chained on the table
PIN=$(sbatch --parsable --export=ALL,SPLIT=train examples/slurm/build_pin_table_multi.sbatch) \
  && sbatch --dependency=afterok:$PIN \
     --export=ALL,SPLIT=train,SIM_CFG=examples/pretrain_multicam_wr.yaml,OUT=output/bc_dataset/train_p5_k8.h5,PIN=output/grasp_cand_table_train_p5.json \
     examples/slurm/collect_bc_demos_p5.sbatch
```

`--cfg-file` must equal `SIM.cfg_file` in the Phase-5 config. Budget K× Phase 4:
~5000 episodes at K = 8, still serial.

Two differences from Phase 4's collector. Pinning is **mandatory** — an episode
whose pin does not match is skipped rather than quietly falling back to OMG's own
selection, because `grasp_idx` is supposed to name the target. And every episode
group gains `grasp_idx` plus `grasp_pose_world`, the 4×4 the episode actually flew
to, read back from the planner rather than copied from the table. Storing the pose
means the dataset carries its own conditioning target and rebuilding a pin table
can never retarget a collection made against the old one.

### 3. Prune K → 4, keeping only the demonstrations that worked

```bash
python examples/select_pinned_grasps.py \
    --demos output/bc_dataset/train_p5_k8.h5 \
    --cand-table output/grasp_cand_table_train_p5.json \
    --n-final 4 --sep-floor 0.02 \
    --out output/bc_dataset/train_p5           # --dry-run for the attrition only
```

Writes three files, all of which the config points at:

| file | config key |
|---|---|
| `output/bc_dataset/train_p5.h5` | `TRAIN.base_train_h5` |
| `output/bc_dataset/train_p5.json` | `SIM.grasp_pin_table` |
| `output/bc_dataset/train_p5_excluded.json` | `SIM.exclude_scenes` |

This is the single place that decides what the training set contains. It applies
`filter_demos.py`'s completion test verbatim in effect (an episode has a CLOSE
label iff it ran to the end) plus the malformed-cloud check that killed run 19,
then re-runs the max-min selection **over the survivors** rather than taking the
first four by FPS rank — if slots 1 and 2 failed, a fresh pass over exactly the
poses that worked gives the best four available. `slot0_is_omg` records per scene
whether OMG's own pick survived, which is what makes a slot-0-only comparison
against run 16 legitimate.

`filter_demos.py` does not need to run first, and its `--scenes-out` list is wrong
for Phase 5 anyway: it drops a whole scene when any single slot fails.

Repeat for val. **Known pre-existing bug worth noting while rebuilding val:**
`valid_grasp_dict_005.pkl` has exactly 720 keys and `run_omg_planner` passes the
split-relative index straight through (`omg/planner.py:490`), so val tables have
always been built with the train dict's entries 0–35. `utils.py:34` guards `SETUP`
but not `SPLIT`. Phase 5 inherits this.

### 4. Check the data before spending a trainer on it

```bash
python examples/analyze_grasp_separation.py --demos output/bc_dataset/train_p5.h5
```

Twenty minutes, no GPU, and it answers the question that decides whether the whole
idea can work: for a given scene, how different are the four expert
demonstrations? If they are nearly identical the conditioning input has nothing
to predict from, and the failure is in the data rather than the network.

Read the **profile**, not the headline. The four grasps share a free approach and
are meant to diverge only into the reach, so a low overall informative fraction is
the geometry, not a bug — it is exactly why `DATA.reach_tail_weight: 2.5` exists.
A profile that is flat at ~0 all the way to the last step means the pin did not
apply or the slots collapsed onto one grasp, and that is a bug to find here.

### 5. Train

```bash
# shakedown FIRST — see the checklist below
sbatch --time=01:00:00 --export=ALL,RUN=dagger5_smoke,CFG=examples/configs/dagger_phase5_smoke.yaml \
    examples/slurm/train_dagger_phase5.sbatch

# run 1 — eval split off into its own job
sbatch --time=20:00:00 --export=ALL,RUN=dagger5_run1,\
SCRATCH_ROOT=/scratch/$USER/handover-sim2real examples/slurm/train_dagger_phase5.sbatch
sbatch --export=ALL,RUN=dagger5_run1 examples/slurm/eval_dagger_run_p5.sbatch
```

Re-run the identical command to resume. Run dir layout is Phase 4's exactly.

The smoke run is not optional: Phase 5 changes the unit of work from a scene to a
`(scene, grasp)` pair in six places, and this is what proves all six agree.

| smoke check | expected |
|---|---|
| `grasp_mismatch` | `0` — a (scene, slot) never re-aimed between iterations |
| `goal_switch` | `0` — and never moved within an episode |
| `grasp_idx` values in `data/dagger_iter_01.h5` | four distinct |
| `cond_goal_spread` | `> 0` — the commanded grasps really are separated |
| `cond_track` | any number; a blank means the diagnostic is not wired up |

### 6. Read the results

```bash
python examples/plot_dagger_run_p5.py output/dagger_runs/dagger5_run1
```

Three figures now: `curves.png` and `curves_diag.png` as in Phase 4, plus
**`curves_p5.png`**, which is the one this phase is about.

`cond_track` first. It is the mean pairwise spread of the four final EE poses
divided by the spread of the four commanded grasps, under the same flip-invariant
metric the selection used. Near 1 means the policy separates the four conditions
as much as their targets are separated; near 0 means it does the same thing
whatever it is told — the multi-modal averaging failure, which makes regrasping
inert however good `success_rate` looks.

Read it against `near_rate`, because the pair localises the problem. **Both low**
means the conditioning is being ignored, and the fix is FiLM over the fused
feature rather than concatenation. **`cond_track` high with `near_rate` low**
means the policy separates the conditions but tracks none of them, which is a
reach-endgame problem and points at merging in run 21's open-loop pre-grasp
commit. Either way it is readable from the first run, which is the point.

`retry_at_k` is the regrasping headline — success with k attempts at different
grasps, in FPS order — and it is free, derived from the same episodes with no
extra rollouts. It assumes each retry restarts from home, which is true of this
evaluation and *not* of a real deployment where attempt 2 begins wherever attempt
1 stopped, so read it as a ceiling.

`succ_g0..3` / `near_g0..3` are the per-slot rates; slot 0 is OMG's own pick, so
`succ_g0` is the number comparable with a Phase-4 run and the spread across slots
says how much harder the deliberately-separated grasps are.

And the standing caveat from `docs/run_index.md`: the Phase-4 noise floor on 100
eval scenes is **±0.115** (six identical base fits spanning 0.32–0.62). Any
`success_rate` difference against run 16 smaller than ~0.15 is unreadable. Prefer
`near_rate` and `cond_track`, which sit near zero today and have room to move.

### 6b. Roll out a checkpoint

```bash
# watch one scene under two different commanded grasps — the eyeball cond_track
python examples/rollout_bc_policy_p5.py --run-dir output/dagger_runs/dagger5_run1/best \
    --cfg-file examples/pretrain_multicam_wr.yaml --max-steps 50 \
    --scenes-from-run output/dagger_runs/dagger5_run1 \
    --grasp-pin-table output/bc_dataset/train_p5.json --show-goal-grasp --grasp-idx 0
#   ... then --grasp-idx 3

# headless conditional table over every scene x every slot
python examples/rollout_bc_policy_p5.py --run-dir output/dagger_runs/dagger5_run1/best \
    --cfg-file examples/pretrain_multicam_wr.yaml --max-steps 50 \
    --scenes-from-run output/dagger_runs/dagger5_run1 \
    --grasp-pin-table output/bc_dataset/train_p5.json \
    --benchmark --all-grasps --no-render --egl
```

`--grasp-pin-table` is no longer overlay-only: a conditioned checkpoint gets its
input from it, so the script refuses to run without one. `--max-steps` must match
`DAGGER.max_steps` (50).

## What changed against Phase-4 run 16

Run 1 is a **combination run**, not a controlled test — conditioning, the aux
head, the cameras, the encoder init and the epoch budget all move at once.

| | run 16 | Phase 5 run 1 | why |
|---|---|---|---|
| `SIM.cfg_file` | `pretrain_multicam_wlr` | `pretrain_multicam_wr` | wrist + right, matching what `sim2real/` can produce |
| `SIM.grasp_pin_table` | 1 grasp/scene | **4** | the phase |
| `DAGGER.episodes_per_iter` | 100 | **200** | 50 scenes × 4 slots |
| `MODEL.grasp_cond` | — | **true** | goal grasp as input, rot6d(6)+trans(3) in the EE frame |
| `MODEL.aux_head` | true | **false** | forced by the above |
| `MODEL.policy_hidden` | `[256,256]` | `[512,256]` | fused vector 512 → 640 |
| `TRAIN.pc_pretrained` | CVPR2023 | **empty** | PointNet++ from scratch |
| `TRAIN.iter_epochs` | 25 | **12** | holds wall clock at 2.2× the data |
| `TRAIN.train_from_scratch` | true | **false** | warm start each iteration |
| `TRAIN.init_ckpt` | (inert) | **`best`** | borrow the previous iteration's best |
| `EVAL.every` / `num_scenes` | 1 / 100 | **0 / 50** | scorer as its own job; ×4 slots = 200 episodes |

**The aux head and the conditioning move together and cannot be separated.** Once
the grasp is an input in the EE frame, the aux head's target *is* its own input
and it learns an identity map. The confound is real; it is stated rather than
hidden.

### Sampling: 50 scenes × 4, not 200 independent draws

`sample_pairs` draws `m // num_grasps` scenes and rolls out every grasp of each.
Two rollouts that differ *only* in the commanded grasp — same scene, same start
state, same cloud — are the only data that can teach the network the goal input
matters, and drawing them together puts them in the same iteration and the same
batches. The cost is scene coverage: 50 distinct scenes per iteration against run
16's 100.

### Epochs come down, not up

The aggregate reaches ~220k steps by iteration 25, 2.2× run 16's 99.5k — not 4×,
because `D_dagger_frac` reaches 0.90 and the base set is only a tenth of the
total. Run 16 spent 25 epochs per refit for ~34.4 M sample-passes and ~11.9 h of
`train_s`; 12 epochs here spends ~39 M, so the wall clock lands near run 16's
despite the extra data. Warm-starting is what makes 12 enough.

`base_epochs` stays at 100: iteration 0 sees 4× the data *and* a randomly
initialised PointNet++, so 100 is if anything the floor. Budget ~78 min against
run 16's ~19.

### Capacity is deliberately not scaled with the data

2.04 M parameters against run 16's 1.96 M — the head widening and the grasp
encoder very nearly cancel the aux head that was removed. The new axis of
variation is a 9-D pose, not new perceptual complexity: same 18 objects, same
cameras, in fact *fewer* scenes, and 87 % of the parameters sit in a PointNet++
whose job has not changed. Phase-3 runs 41 (deeper) and 42 (smaller) bracket run
28 on both sides and neither moved the number beyond the ±0.115 floor. A capacity
bump is a clean single-variable run 2 if the train loss plateaus high — not
something to confound run 1 with.

## Architecture

```
point_cloud [B,1024,5] ─► PointNet++      ─► scene [B,256] ┐
robot_state [B,32]     ─► RobotEncoder    ─► robot [B,256] ├─► [B,640] ─► PolicyHead ─► [B,7]
goal grasp  [B,9]      ─► GraspEncoder    ─► grasp [B,128] ┘
```

Injected at the **fused** level, following `rl/actor.py`'s `clock_dim`, whose
docstring gives the reason: it keeps the two existing encoders shape-identical, so
a Phase-4 checkpoint's `pc_encoder` and `robot_encoder` still load 1:1.

The conditioning vector is `rot6d(6) + translation(3)` — the goal grasp in the
**current EE frame**, recomputed every step because the world grasp is fixed but
the EE moves, so it is the residual pose the policy still has to null out. It is
frame-consistent with the point cloud, and it is the same quantity run 13's aux
head was asked to predict. rot6d rather than the aux head's quaternion because
the direction of use differs: as a regression target the double cover (q ≡ −q) is
absorbed by the geodesic loss, but as a network **input** it makes the function
the encoder must learn discontinuous.

Not normalized — it is already an EE-relative displacement in metres, centred near
zero by construction, and normalizing it would make it depend on statistics the
real robot cannot reproduce.

Sized at 128, about a fifth of the fused vector: enough not to be drowned by 512
dims of perception, not so much that the head can solve the task by servoing to
the goal and ignoring the cloud. That shortcut does exist — it would fly straight
through the human's hand, which is precisely what OMG's labels do not do, so the
loss pushes back. Watch `f_human_contact` anyway.

## Sim2real consequence

`sim2real/my_policy_runner.py` has no grasp source, and the real rig has no pin
table and no OMG goal set. `bc/models.py`'s own block diagram notes that the
original design had an *AnyGrasp/Grasp-MLP* branch which was removed; Phase 5
reintroduces that dependency. It is arguably the right dependency — regrasping on
hardware needs a grasp proposer regardless — but a Phase-5 checkpoint cannot be
deployed without one. `goal_target_from_state`'s `valid` channel is kept so the
branch stays maskable.

## Deferred

**True chained retry**, where attempt 2 starts from where attempt 1 stopped rather
than from home, with any object or hand disturbance persisting. That is the honest
deployment metric; `retry_at_k` as computed here is its ceiling.

**A held-out grasp split** — train on slots 0–2, evaluate on slot 3 — which would
answer whether the conditioning generalises to a grasp never demonstrated for that
scene. `EVAL.holdout` currently holds out scenes, not grasps.

Neither is in run 1.
