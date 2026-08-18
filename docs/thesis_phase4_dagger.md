# Phase 4

Reference config: `examples/configs/dagger_phase4.yaml`.
Entry point: `examples/train_dagger_phase4.py`.
Library: `handover_sim2real/dagger/`.

Phase 4 runs **Algorithm 3.1 of Ross et al. (2011)** rather than the
shell-driven approximation of Phase 1/2 or the inline RL variant of Phase 3.
Everything happens in one Python process: the simulator, the OMG planner and the
model stay resident across iterations.

The learner is the **Phase-1 single-frame policy** by default
(`TRAIN.train_cfg: examples/configs/bc_phase1.yaml`). The loop itself is
policy-agnostic — point `train_cfg` at `act_phase2.yaml` and the same loop drives
the Phase-2 temporal/chunking policy. The kind is inferred from that config
(`MODEL.chunk_len` present => ACT) and hidden behind `PolicyRunner`, so neither
the collector nor the evaluator contains policy-specific code: the single-frame
runner is stateless between steps, the ACT runner owns the T-frame ring buffer
and the chunk-execution strategy.

---

## 1. The loop

```
D    <- expert demonstrations (TRAIN.base_train_h5)      # the paper's beta_1 = 1 iteration
pi_1 <- train(D)                                         # iteration 0, "the base run"
for i = 1..N:                                            # DAGGER.num_iters
    pi_i^mix = beta_i * pi* + (1 - beta_i) * pi_i        # per-step Bernoulli mixture
    sample m T-step trajectories with pi_i^mix           # m = DAGGER.episodes_per_iter
    D_i = {(s, pi*(s)) : s visited}
    D <- D u D_i                                         # aggregate; nothing is dropped
    pi_{i+1} <- train(D)                                 # fresh fit on the union (FTL)
    evaluate pi_{i+1} on held-out scenes
keep the BEST-on-eval policy and the LAST policy
```

Three things this gets right that the earlier phases did not:

**m is a parameter, and it is meant to be small.** The paper's finite-sample
result is `N = O(T^2 log(1/delta))` with **`m = O(1)`**; Super Tux Kart used *one
lap per iteration* over 20 iterations. What each iteration buys is a *new state
distribution* — more episodes under the *same* policy are just more samples of
the same one. `run_dagger.sh` defaulted to `ITERS=3` over the whole 700-scene
split (m ~ 350, N = 3), which is the inverse allocation. The Phase-4 default is
`num_iters: 20`, `episodes_per_iter: 20`.

**The aggregate is never truncated.** Iteration `i` trains on
`[base_train_h5, dagger_iter_01.h5, ..., dagger_iter_i.h5]`, pooled uniformly by
`BCDataset` (or `BCSequenceDataset` for an ACT learner — both accept a list of
files). This is the Follow-The-Leader step the no-regret guarantee
rests on. (Phase 3's replay buffer is a FIFO ring, so its DAgger labels are
*evicted* — a sliding window, not an aggregate.)

**Both the best and the last policy are kept.** The paper returns
`best pi_i on validation`, not `pi_N`; FTL only guarantees that *some* policy in
the sequence is good. `EVAL` rolls the policy closed-loop over held-out scenes
after each iteration; `best/` and `last/` are standalone run dirs that
`rollout_bc_policy.py --run-dir` (or `rollout_act_policy.py` for an ACT
learner) loads directly. **Iteration 0 is evaluated
too** — the base policy is part of the sequence and can win.

`beta_schedule: indicator` is the paper's `beta_i = I(i = 1)`, reported in
Section 5 as the best-performing choice: the base run already *is* the beta = 1
iteration, so every DAgger iteration runs at beta = 0. `exponential`
(`beta_p^i`), `linear` and `constant` are also available.

---

## 2. What Phase 4 changes about the labels

| | Phase 1/2 collectors | **Phase 4** | Phase 3 (online) |
|---|---|---|---|
| expert query | full replan every recorded step | **full replan every step** | plan once at step 0 + gated tail splices |
| label | `plan[0]` | **`plan[0]` on approach, committed reach tail by index at the standoff** | `plan[t]` of the committed plan |
| labelled region | approach only (cut at the standoff plane) | **the whole episode, approach + reach + close** | the whole episode |
| gripper label | always OPEN (close-trigger opt-in) | **CLOSE once inside the distance threshold** | separate proximity BCE head |
| grasp set | OMG default / runtime 0.08 m filter | **paper's offline `valid_grasp_dict_005.pkl`** | same offline dict (run 21+) |

### 2.1 Replan every step; label = `plan[0]`

```python
horizon = dynamic_replan_horizon(obs, standoff_pose, ee_step, reach_tail, min_free, max_horizon)
plan, _ = env.run_omg_planner(horizon, scene_idx, reset_scene=(step == 0))
expert_delta = env.convert_target_joint_position_to_action(plan[0])   # [6], m / rad
```

`run_omg_planner` sets `traj.start = panda.body.dof_state`, so with
`reset_scene=False` the plan genuinely starts from the *drifted* configuration —
that is what makes it a `pi*(s)` query rather than a re-derivation of the
original plan. Querying at **every** step is what Algorithm 3.1 asks for; the
cost is one OMG plan per *approach* step — replanning stops once the reach is
committed (§2.2), so ~40 plans per episode, ~800 per iteration at m = 20.

### 2.2 The standoff-plane cutoff is gone — replaced by a committed reach

Phase 1/2 stopped recording at the pre-grasp standoff, because a replan from
beyond it returns a backward *retreat to the standoff* label. Phase 4 records all
the way in. That needs a fix, because **`plan[0]` labelling structurally cannot
express the final reach**.

Replanning from the standoff (measured, scene 60):

```
horizon= 8  ||delta to plan[i]||:  0.001 0.002 0.003 0.005 | 0.020 0.036 0.052 0.068
horizon=20  ||delta to plan[i]||:  0.000 ... 0.005          | 0.020 0.036 0.052 0.068
                                   ^ free portion (dead)      ^ the reach tail
```

The reach is *always* the last `reach_tail` entries, and `plan[0]` is *always* in
the free portion — which has nothing left to do once the EE is at the standoff,
because OMG puts the standoff ramp in its own goal set. Driving with **beta = 1
(pure expert, no policy involved)**:

```
 step | d(ee,standoff) | d(ee,grasp) | horizon | ||plan[0] dpos||
   38 |     0.0180     |   0.0750    |     8   |      0.0040
   50 |     0.0031     |   0.0667    |     8   |      0.0003
   69 |     0.0022     |   0.0661    |     8   |      0.0000   <- stalled
```

The EE parks 6.6 cm short of the grasp and every further label is ~0 — the same
Zeno stall that produced the rl_run7 hover in Phase 3. Two non-fixes, both
measured: `min_free: 0` (horizon collapses to the reach tail) still stalls at
0.0640; taking "the first waypoint that actually moves" reaches 0.036 m and then
**limit-cycles** 0.036 ↔ 0.052, because past the standoff OMG's nearest goal is
behind the EE and it retreats.

**The fix.** On arrival at the standoff the reach is *committed*:

```python
if ||ee - standoff|| <= reach_commit_dist:          # 0.05 m
    committed_reach = plan[-reach_tail:]            # freeze; stop replanning
    reach_i = first index whose delta >= reach_skip_eps   # tail[0] IS the standoff
...
reach_i = min(reach_i, len(committed_reach) - 1)    # hold the grasp past the end
label   = delta_from_current_ee_to(committed_reach[reach_i])
```

No further replanning, so no retreat and no oscillation. `convert_target_joint_
position_to_action` still recomputes the delta from the **current** EE every
step, so the labels stay corrective under drift — only the *target* stops moving.
Past the end the final waypoint (the grasp) is held, so the label shrinks
monotonically until the close threshold fires.

Verified with beta = 1 across scenes 0/60/61/137/200: every one commits the
reach, walks it in 6 steps, lands **4.3–4.8 mm** from the grasp with ~0 rotation
error, and terminates on `CLOSE_LABEL` in 38–46 steps.

> **Note on `max_steps`.** 50, not 30. The EE moves ~0.02 m/step — IK tracking
> under `steps_action_repeat` realises only ~60% of a commanded ~0.035 m delta —
> and a far-start scene is ~0.78 m out, so the standoff alone is ~36 steps away.
> The demonstrations need only 20 steps because `collect_bc_dataset.py` follows
> the plan **by index**, where each successive waypoint is further ahead;
> replanning and taking step 0 undershoots by construction.

### 2.3 Distance-triggered gripper-close label

At every step the EE pose is compared with the grasp pose the *current* plan aims
at (OMG `traj[-1]`, re-read each step so the close decision and the approach
labels always refer to the same target; `goal_switch` counts re-selections):

```python
pos_err, rot_err = ee_grasp_pose_error(obs, grasp_pose)
if pos_err <= close_pos_thresh and rot_err <= close_rot_thresh:
    label = [0, 0, 0, 0, 0, 0, 0.0]        # zero pose delta + gripper CLOSE
```

Defaults `0.02 m` / `0.34 rad`. This is the endgame supervision the demos supply
as a single terminal transition, now available at whatever pose the *policy*
actually reaches. It also survives a failed replan (the grasp pose is cached).

`stop_on_close_label: true` ends the episode at the first close label, mirroring
the demonstrations; `false` keeps rolling and emits a close label at every
in-threshold step.

### 2.4 Plans to the standoff **and** beyond

`use_standoff` stays on, so an OMG trajectory is
`[free portion -> standoff] + [reach_tail waypoints -> grasp]`. Inside the
planner this is a *goal-set* construction, not a post-hoc append
(`omg/planner.py`):

```python
pose_standoff[:, 0, 2, 3] = -standoff_dist * np.linspace(0, 1, reach_tail_len, endpoint=False)
standoff_grasp_global = pose_grasp_global @ pose_standoff
```

The goal is a 5-pose ramp retreating along the gripper's local −z axis, and IK is
solved for the whole ramp — which is what forces the trajectory to arrive **along
the grasp approach axis** instead of diving in sideways and clipping the object
with the fingers. Note `endpoint=False`: the offsets are
`0.8 × standoff_dist × [0, .25, .5, .75, 1]`, so `traj[-5]` is **0.064 m** back,
not 0.08 (measured across scenes 0/60/61/137). The horizon is
distance-proportional:

```
free    = max(round(||ee - standoff|| / ee_step), min_free)
horizon = min(free + reach_tail, max_horizon)
```

so the recorded `plan[0]` stays at the demonstrations' ~4 cm per-step scale at
every distance (no late-step "big jump" labels), and the final reach is always
present in the plan. The step-0 plan has no standoff yet and falls back to
`cfg.RL_MAX_STEP` (20) — the horizon `collect_bc_dataset.py` planned the demos
with.

### 2.5 One committed grasp per scene

OMG does **not** keep a fixed goal grasp. `omg/planner.py` (`ol_alg='Proj'`):

```python
proj_dist = np.linalg.norm((traj.start - goal_set) * link_smooth_weight, axis=-1)
traj.goal_idx = np.argmin(proj_dist)          # recomputed on EVERY plan
```

`traj.start` is the *current joint configuration*, so the selection is nearest-
grasp in **joint space** and is re-decided every replan. Measured over 90 replans
under ±15 cm / ±0.5 rad perturbation: **32 selected a different grasp**, one scene
cycled through four, and the target moved by up to 10 cm. Two consequences:

- **Within an episode**, labels can point at grasp A for ten steps and grasp B for
  the next ten — an inconsistency DAgger cannot average away.
- **Across phases**, the demonstrations plan once from the home configuration and
  commit to whatever argmin picked there, which need not match what DAgger picks
  from a drifted one.

Both are fixed by pinning one grasp per scene, shared by every phase:

```bash
python examples/build_grasp_pin_table.py --out output/grasp_pin_table.json --tol 0.01
```

`SIM.grasp_pin_table` loads it; `pin_goal_grasp(idx)` prunes the goal set to that
single entry, after which the argmin is constant by construction. Verified: goal
switches per episode drop from 0/0/1/1/3 to **0/0/0/0/0**, and DAgger's final
grasp matches the table exactly (0.0000 m) on every scene tried.

**Selection rule** (`--mode furthest_from_hand`, default): argmax of the
gripper-to-MANO-hand clearance — the same geometry the hand-collision filter
thresholds, used as a continuous score instead of a reject test. Over 265 scenes
it differs from OMG's pick on 174, for a median clearance gain of **1.7 cm** at a
median grasp displacement of **5.9 cm**. That displacement is the cost: you are
moving the grasp a long way for a modest clearance gain, and the far grasp may be
a worse grip or kinematically harder. `--tol` resolves near-ties toward OMG's
natural pick (at `0.01` it eliminated every "moved >5 cm to gain <5 mm" case).

Three implementation traps, all found by hitting them:

- `target_obj.grasps` holds the **standoff** configs, not the grasps. The real
  grasps are `reach_grasps[:, -1]` — OMG's own filter comments on this ("Check the
  FINAL grasp ... not the standoff configs ... which sit 8 cm back and never
  collide"). Ranking clearance on the wrong array measures it 6.4 cm too early.
- `grasp_potentials` is not parallel to `grasps` under `ol_alg='Proj'`; indexing
  it raises.
- `traj.goal_idx` must be reset when pruning — `goal_set_projection` indexes
  `reach_grasps[goal_idx]` before `setup_goal_set` recomputes it.
- `cfg.goal_idx = k` does **not** work: planner.py:223 overwrites it whenever
  `ol_alg == 'Proj'`.

Track switches via `env.get_omg_goal_idx()`, not by comparing grasp poses:
`flip_grasp` augments the set with wrist-flipped duplicates that share an EE
position, so a pose comparison misses those switches entirely.

The table must be **rebuilt** whenever `valid_grasp_dict_path` or the runtime
hand-collision filter changes — those change which candidates exist. A stale
table fails loudly via `grasp_pin_match_tol` rather than silently retargeting.

### 2.5 Phase-3 grasp filtering

`SIM.valid_grasp_dict_path` wires the paper's offline hand-collision-filtered
grasp dict into `cfg.omg_config` **before** the env is built (`OMGPlanner.__init__`
copies omg_config onto the global `omg_cfg`, so setting it later is a no-op). It
keeps ~716/720 s0 scenes versus ~351 for our runtime 0.08 m filter, which is why
Phase 3 switched to it at run 21. `hand_collision_filter` stays `false` —
stacking both double-prunes the goal set.

---

## 3. Following the learner's own policy

Two switches control how faithfully the rollout tracks the learner:

- `stop_on_policy_close` (default **true**): the policy commands a close, so
  execute it and end the episode — in deployment the episode really is over. The
  *label* recorded at that state is still the approach delta (or a close label if
  the threshold was met), which is exactly the correction for a premature close.
  `false` overrides the gripper bit and keeps approaching: more labels per
  episode, but the trajectory is no longer one the policy would have produced.
- `beta`: with probability `beta_i` a step executes `plan[0]` instead of the
  policy's action. Labelling is independent of beta — an expert-executed step is
  still labelled, and a policy-executed step is still labelled.

---

## 4. Refitting

`train_from_scratch: true` (default) is the FTL step: a new model, a
normalizer recomputed over the whole aggregate, PointNet++ warm-started from the
CVPR2023 encoder, `iter_epochs` epochs. `false` warm-starts from the previous
iteration **and reuses its normalizer** (the head's output scale is defined by
the normalizer, so warm-starting under freshly recomputed stats would silently
reinterpret it) — cheaper, but no longer Follow-The-Leader.

Two different "best" notions coexist and both are useful:

| | selects | scope |
|---|---|---|
| `checkpoints/best.pt` | lowest `val_h5` loss | within one iteration's training run |
| `<run>/best/` | highest `EVAL.select_on` | across DAgger iterations |

Iteration-level selection is lexicographic — `select_on`, then `grasp_rate`, then
`near_rate`, then negated final EE→object distance — because early on every
iteration scores zero success and the tie-breaks are what separate "closed on the
object" from "closed on air" and "got close" from "never approached".

### 4.1 What counts as a success

**Phase 4 scores the Phase-3 criterion, not the handover-sim benchmark's.** The
benchmark's `EpisodeStatus.SUCCESS` requires the hand to dwell 0.1 s inside a
15 cm ball at `GOAL_CENTER` after the human releases — i.e. it scores a
*carry-to-goal*. Phase 4, like Phase 3, ends the episode at the committed close;
nothing ever drives a retreat, so that flag could never fire and would score every
episode 0. Both modes are imported from `rl/rollout_worker.py` rather than
reimplemented, so the number Phase 4 reports and the reward Phase 3 optimises
cannot drift apart.

| `EVAL.success_mode` | test | measures |
|---|---|---|
| `stable_grasp` (default) | hold the gripper shut `hold_steps` policy-steps, then require `ycb.released` ∧ ¬dropped ∧ no human contact | task outcome, physics |
| `proximity` | EE within (`close_pos_thresh`, `close_rot_thresh`) of the grasp pose at the close | label agreement — the *same* predicate the collector uses to emit its CLOSE label |

`stable_grasp` leans on handover-sim's release handshake: the human only lets go
once the robot's fingers really grip the object inside the release contact region,
so "released and not dropped after a hold" is a genuine grasp rather than a
proximity proxy. It needs no grasp pose, so eval makes **no OMG calls**.
`proximity` needs one — free from the grasp pin table (§2.5), otherwise one step-0
plan per episode.

Four rates are logged every iteration, ordered by how much each demands, so
reading them left to right localises where episodes are lost:

| | |
|---|---|
| `close_rate` | the policy committed a close at all |
| `near_rate` | …within the CLOSE tolerances of the pinned grasp |
| `grasp_rate` | …and both fingers ended the hold on the object (`grasped_active()`) |
| `success_rate` | …and the object was secured |

The gaps are the diagnosis: `close − near` is closing in the wrong place,
`near − grasp` is closing at the right pose but not gripping, `grasp − success`
is gripping and then losing it.

Note that `near_rate` and `success_rate` can disagree, and the disagreement is
informative. Measured on the Phase-1 `phase1_full` policy over 8 scenes, scene 200
scored `stable_grasp` = 1 but sat 9.2 cm / 1.6 rad from the **pinned** grasp: the
policy secured the object from a different approach than the one the pin table
commits to. That is expected for a policy trained on unpinned demonstrations, and
it is precisely what re-collecting against the pin table is meant to remove.

---

## 5. Run layout and resume

```
output/dagger_runs/<name>/
    config.yaml        the resolved Phase-4 config
    state.json         completed iterations + best-so-far (drives resume)
    dagger_log.csv     one row per iteration: data stats + eval metrics
    data/              dagger_iter_NN.h5  — D_i, in the Phase-1/2 BC schema
    iters/iter_NN/     a full policy run dir per iteration
    best/  last/       standalone snapshots (checkpoints/best.pt + config + normalizer)
```

Re-running the same command resumes: completed iterations are skipped, a
finished-but-untrained collection is reused, and an interrupted training
continues from its `last.pt`. Episodes are streamed to HDF5 one at a time, so an
interrupted iteration keeps everything already collected.

The eval scenes are taken evenly spread across the split (not a trailing block)
and, with `EVAL.holdout: true`, removed from the collection pool — so
best-on-validation is measured on scenes DAgger never labelled.

### 5.1 What is logged, and what each group answers

`dagger_log.csv` is one row per iteration, appended and flushed as the loop runs,
so it can be plotted mid-run. `examples/plot_dagger_run.py <run>` renders it into
`curves.png` + `curves_diag.png` and is read-only — safe to run against a live
run. Columns are grouped by the question they answer:

| group | columns | the question |
|---|---|---|
| eval rates | `close_rate` `near_rate` `grasp_rate` `success_rate` | did it learn |
| eval conversion | `chance_rate` `close_success_rate` `missed_rate` `miss_given_chance` | did it get a chance, and take it |
| eval geometry | `eval_min_pos` `eval_min_rot` `mean_pos_err` `mean_rot_err` `mean_dist` `mean_close_step` | how close did it ever get, and how wrong was the close |
| eval outcomes | `f_grasp_ok` `f_grasp_miss` `f_no_release` `f_drop` `f_timeout` `f_human_contact` | which failure mode, as fractions summing to 1 |
| collection | `reached_standoff` `reached_grasp` `policy_closed` `mean_min_pos` `mean_min_rot` `c_*` | how far the **learner's own** rollouts got |
| labels | `mean_label_pos` `tiny_labels` `close_labels` `approach_labels` | is the expert still producing usable labels |
| planner | `omg_fail` `goal_switch` `pinned` | is the label source healthy |
| grasp consistency | `revisits` `grasp_mismatch` `max_grasp_drift` | did a revisited scene aim at the same grasp |
| aggregate | `D_episodes` `D_steps` `D_dagger_frac` `aggregate_files` | how big and how on-policy is D |
| refit | `epochs` `train_loss` `val_loss` `*_grip_acc` `best_val_loss` | is the fit degrading as D grows |
| mixing | `beta` `expert_steps` `m` | was the schedule actually applied |
| cost | `collect_s` `train_s` `eval_s` `wall_s` | where the time goes |

Six of these are load-bearing rather than decorative:

**The four eval rates are nested** — `close ≥ near`, `close ≥ grasp ≥ success` —
so the *vertical gaps* localise the failure without a second experiment:
`close − near` is closing in the wrong place, `near − grasp` is closing at the
right pose but not gripping, `grasp − success` is gripping and then losing it.

**`mean_label_pos` is the stall detector.** The standoff failure mode (§3) is
approach labels decaying towards zero while the EE parks short of the grasp. Step
counts and success rates look normal while it happens; the mean label
displacement collapsing away from `ee_step` is the only direct signal, and
`tiny_labels` counts the labels that carry no information at all.

**`goal_switch` should be flat zero** once the pin table is loaded. Anything else
means the table is stale or failing to match, and the labels are quietly pointing
at two different grasps within one episode.

**`D_dagger_frac` is what separates the two hypotheses.** Success rising with
`D_steps` is consistent with "more data helps"; success rising while the
on-policy share rises is the DAgger claim. Plotting them together is the only way
to tell them apart from a single run.

**`eval_min_pos` / `eval_min_rot` still move when every rate reads 0.** They are
the closest the EE came to the grasp over the *whole* episode, so they are
defined even when the policy never closes — unlike `mean_pos_err`, which is NaN
for exactly the episodes that need explaining. Early in a run they are usually
the only columns doing any work. `eval_min_rot` matters as much as `eval_min_pos`:
Phase 3 repeatedly found rotation to be the binding constraint (rl_run10 regressed
`eval_min_rot` while position looked fine), and the measured Phase-1 policy sat at
9.2 cm / **1.6 rad** — inside no reasonable rotation tolerance.

**The conversion columns separate two failures a success rate conflates.**
`chance_rate` is the fraction of episodes that ever reached a pose where closing
*would* have been correct (both tolerances satisfied at the *same* step — not
`min_pos` and `min_rot` independently, which can be met several steps apart and
would over-report the opportunity). Against it:

| reading | meaning |
|---|---|
| `chance_rate` low | the *reach* is the problem; the trigger is irrelevant |
| `chance_rate` high, `close_rate` low | it gets there and never pulls the trigger |
| `close_rate` high, `close_success_rate` low | it pulls the trigger in the wrong places |
| `miss_given_chance` high | it had real opportunities and threw them away |

`close_success_rate` is success *conditioned on having closed* — "when it decides
to grasp, is it right?" — which is a different question from `success_rate`, and
the one that isolates the gripper head from the reach.

**`grasp_mismatch` must stay 0.** The pin table *enforces* one grasp per scene;
this *verifies* it, which is not the same thing. A pin can silently fail to apply
(scene absent from the table, a stored pose that no longer matches because the
grasp set changed, an OMG failure at step 0) and each leaves OMG's re-selection in
force without stopping the run. Over 20 iterations a scene is collected several
times: if it aimed at grasp A in iteration 3 and grasp B in iteration 11, D holds
two contradictory label sets for the same states. `goal_switch` cannot see this —
it is per-episode. `GraspRegistry` records the grasp each scene first aimed at,
persists it to `<run>/grasp_registry.json` so it survives a resume, and reports a
mismatch rather than correcting it: the run continues, but the contradiction is
visible in the CSV instead of sitting silently in the aggregate.

The collection metrics lead the eval metrics: the learner reaching the standoff
on its own shows up an iteration or two before held-out success moves, so a flat
success curve with a rising `reached_standoff` means the loop is working and the
horizon or the endgame is the bottleneck, not the aggregation.

---

## 6. Scoring a finished run on the **test** split

**Nothing in `dagger_log.csv` is a generalisation number.** Every Phase-4 run
sets `SIM.split: train`, and every run after the first sets `EVAL.holdout:
false`, which puts the eval scenes back into the collection pool. So the
`success_rate` column is the policy's performance on scenes it was trained on.
Measured on run 16, that gap is about eleven points: 80.0% logged, **69.2%** on
the test split.

`examples/eval_run_scenes.py` rolls a run's checkpoint over a whole split, one
row per scene. It reuses `build_phase4_context(<run>/config.yaml)` — the same
call the training loop makes — so the simulator, the thresholds and the success
criterion cannot drift from what the run itself used. Only the split changes.

```bash
export OMG_PLANNER_DIR=$PWD/OMG-Planner GADDPG_DIR=$PWD/GA-DDPG
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/OMG-Planner/orocos_kinematics_dynamics/orocos_kdl/release/lib"

python examples/eval_run_scenes.py \
    --run-dir output/dagger_runs/dagger4_run16 \
    --split test \
    --grasp-pin-table output/grasp_pin_table_test_omg.json \
    --exclude-scenes none \
    --box-probe \
    --out-prefix output/dagger_runs/dagger4_run16/scene_eval_test
```

`--run-dir` is the RUN root, not a checkpoint directory; the script resolves
`best/` itself (`--from last` for the last policy). Roughly 6 s per scene with
`--box-probe`, so ~13 min for the 130 usable test scenes.

**Scene indices are split-relative.** `HandoverBenchmarkWrapper` rebuilds
`_scene_ids` per split, so scene 7 of test is not scene 7 of train. Two inputs
are numbered in that space and both must move with the split:

* `--grasp-pin-table` — a train table applied to test pins an unrelated grasp on
  every scene, and it doubles as the usable-scene filter, so it changes the
  denominator as well as the target. Build one per split with
  `examples/slurm/build_pin_table.sbatch`.
* `--exclude-scenes` — the failed-expert list is train-relative and has no test
  equivalent, so pass `none`.

### 6.1 Reading the report

```
  success_rate           69.2%   secured grasps / all scenes
  close_rate             76.9%   the policy committed a close at all (100/130)
  close_success_rate     90.0%   of those closes, how many held
  grasp_rate             66.9%   fingers on the object after the hold
  ------------------------------------------------------------------
  opportunity            77.7%   >=1 ray(s) hit the object AND a real close there secured it (101/130)
  success | opportunity  89.1%   ...and the policy converted it  <- conversion
  miss | opportunity     10.9%   ...and it did not
  probe_pass_rate        80.2%   of 339 geometric gate passes, how many a REAL close confirmed
  ok success ⊆ opportunity holds on every scene.
```

The top block is the nested chain — read left to right to localise where
episodes are lost.

**`opportunity` is a two-tier counterfactual**, not a geometric test. Tier one
is a 7x7 grid of rays cast pad-to-pad across the open jaws
(`dagger/grasp_box.py`), thresholded at a single ray (`--min-rays`, default 1).
Tier two, enabled by `--box-probe`, settles every step that clears the gate by
**actually closing the gripper**, holding it, scoring it with the evaluator's own
criterion, and then rewinding the simulator to the exact prior state
(`dagger/grasp_probe.py`). A step counts only if grasping right there would
genuinely have secured the object — which folds in everything the rays cannot
see, above all the human hand, which carries a collision filter that makes it
invisible to them.

Two consequences worth knowing before quoting these numbers:

* **Without `--box-probe` the figure is an upper bound**, since a pose where
  closing would collide with the human still counts. The report labels which
  definition produced it.
* **With `--box-probe`, `success | opportunity` is tautologically equal to the
  "closed on an opportunity step" rate**, because an opportunity is *defined* as
  a step where closing secures the object. Quote one of them, not both. The
  informative content is the gap between `opportunity` and `success_rate`: on run
  16's test split, 101 of 130 episodes had a real chance, 90 converted, and the
  other 29 never manufactured one at all — 15 of those died on human contact and
  13 on a drop, mostly before ever commanding a close. The bottleneck is the
  approach, not the grasp decision.

The invariant **success ⊆ opportunity** must hold scene by scene: the policy's
own successful close is itself proof that a graspable moment existed. When it
breaks, the report names the offending `scene_idx` values so they can be replayed
one at a time. Historically it broke because the gate was set at half the pad
(`--min-frac 0.5`, what runs 1-20 logged), which recovers only 80% of the
episodes that actually secured the object. Pass `--min-frac 0.5 --min-rays 1`
without `--box-probe` to reproduce a `dagger_log.csv`-comparable figure.

**In pre-grasp mode the opportunity block does not apply.** With
`DAGGER.target: pregrasp` the episode ends 6.4 cm short of the object, so the
jaws are never around it while the policy is still deciding and `opportunity`
reads near zero *by construction*. Every conditional on it then rests on a
denominator of one or two episodes and swings between 0 and 1. Read
`box_after_rate` instead — the same question asked after the blind push, with
the fingers still open — and expect the invariant warning to fire legitimately.

Outputs are `<prefix>.csv` (one row per scene), `.json` (the aggregate, including
the gate settings that produced it) and `.png` (six diagnostic panels).

---

## 7. Watching a single scene

`examples/rollout_bc_policy.py` drives the simulator with the policy's own
actions and a PyBullet window. Use it to see *why* a scene in the CSV failed.

```bash
python examples/rollout_bc_policy.py \
    --run-dir output/dagger_runs/dagger4_run16/best \
    --cfg-file examples/pretrain_multicam_wlr.yaml \
    --split test --scene 30 --max-steps 50 \
    --grasp-pin-table output/grasp_pin_table_test_omg.json --show-goal-grasp
```

Here `--run-dir` is a CHECKPOINT directory (`<run>/best`, `<run>/last`, or an
`<run>/iters/iter_NN`), not the run root. `--cfg-file` must be the sim config the
run trained with (`SIM.cfg_file` in `<run>/config.yaml`) — a different camera set
feeds the policy a point cloud it never saw. `--split` matters for the same
reason it does above: without it, scene 30 of test is silently scene 30 of train.
`--show-goal-grasp` draws the pinned grasp in green; pass the same pin table, or
OMG re-picks its own goal and a correct rollout looks like a miss.

Pass several scenes with `--scenes 30,86,32` and step through them in the window:
**N** = next scene, **R** = re-roll the same one, **Q** = quit.

### 7.1 Pre-grasp policies need the open-loop endgame

For a run with `DAGGER.target: pregrasp` (run 21 onward), channel 6 of the action
does **not** mean "close the fingers". It means "I am at the standoff — commit
the endgame", and the CVPR2023 open-loop push must run before the hold. Closing
in place instead shuts the jaws 6.4 cm short of the object and scores every
episode a miss, which looks exactly like a policy that cannot grasp.

`--target` controls this and defaults to `auto`, which walks up from `--run-dir`
looking for the Phase-4 config (the one with a `DAGGER` block) and reads
`DAGGER.target`, `forward_dist` and `forward_steps` out of it. The resolved
endgame is printed at startup, so check that line before trusting a rollout:

```
Endgame: PRE-GRASP — channel 6 commits a blind 0.064 m push over 4 sub-steps, then the hold
```

So run 21 needs nothing extra beyond the right sim config:

```bash
python examples/rollout_bc_policy.py \
    --run-dir output/dagger_runs/dagger4_run21/best \
    --cfg-file examples/pretrain_multicam_wr.yaml \
    --split test --scene 21 --max-steps 50 \
    --grasp-pin-table output/grasp_pin_table_test_omg.json --show-goal-grasp
```

Force it with `--target pregrasp` (and optionally `--forward-dist` /
`--forward-steps`) when the config is not where auto-detection can find it, e.g.
a checkpoint copied out of its run directory.

If the push itself ends the episode — a lateral swing into the pre-grasp can hit
the hand or knock the object loose — that is reported under the benchmark's own
status rather than as `GRASP_MISS`, so an over-long `forward_dist` cannot hide
behind the policy's success rate.

---

## 8. Where this sits relative to the other phases

| | Phase 1/2 | Phase 3 | **Phase 4** |
|---|---|---|---|
| driver | bash | inline in the RL loop | **Python** |
| aggregation | union of HDF5 files, retrain fresh | FIFO ring buffer (evicts) | **union of HDF5 files, retrain fresh** |
| m / N | ~350 / 3 | n/a (continuous) | **20 / 20 (configurable)** |
| expert query | every recorded step, approach only | occasional gated tail splice | **every step, whole episode** |
| policy selection | last round | best online eval | **best-on-held-out + last** |
| success criterion | benchmark carry-to-goal | close + hold, object secured | **same as Phase 3 (shared code)** |
| objective | supervised BC | TD3+BC | **supervised BC (single frame by default; ACT optional)** |
