# Run index

Every training run in `output/`, what makes it different, and what it scored.

**How this was built.** The "what changed" column is a diff of each run's *saved*
config (`rl_config.yaml` / `config.yaml` in the run dir) against its immediate
predecessor, cross-checked against the hand-written header of the matching
`examples/configs/*.yaml`. Scores come from the run's own `log.csv`. Nothing here
is from memory.

**Two caveats that apply throughout.**

`--demos` and `--bc-run` are passed on the sbatch command line and are **not**
recorded in the saved config, so two runs listed as "identical config" may still
have differed in their demonstration pool. Where that matters it is flagged.

Phase-4 eval uses `holdout: false` — the eval scenes are also collected on, so
those success rates are training-set performance. A generalization number needs
`best/` rolled out on `BENCHMARK.SPLIT=test`.

---

## Phase 4 — DAgger (`output/dagger_runs/`)

`peak` / `final` are `success_rate` over 100 eval scenes.

| run | peak | final | what is different |
|---|---|---|---|
| **dagger4_run1** | 0.13 @it4 | 0.04 | First real run. `beta_schedule: indicator` (β=0 from iteration 1), m=20 episodes/iter, 50 epochs, unfiltered `train_pinned.h5`. **Failed**: produced ONE close label in 20 iterations — β=0 meant the learner never reached the grasp, so D got no endgame data at all. |
| **dagger4_run2** | 0.21 @it2 | 0.06 | Fixes run 1: `beta linear 0.75→0.10`, m=20→**100**, epochs 50→25, and the base dataset filtered to `train_pinned_ok.h5` (126 failed demos dropped). Pin rule `furthest_from_hand`. Close labels went from 1-per-run to ~86-per-iteration. |
| **dagger4_run3** | 0.35 @it10 | 0.17 | = run 2 with **one** key changed: pin rule `furthest_from_hand` → **`omg`** (the grasp reachable with least arm motion from home). Config diff verified 1 of 51 keys. |
| **dagger4_run4** | 0.51 @it0 | 0.27 | = run 3 with **one** key changed: `train_cfg` → `bc_phase1_nojoint.yaml`, i.e. `MODEL.drop_joint_state: true`. Robot-MLP input **26 → 8 dims** (`ee_pos`, `ee_orn`, `gripper`); joint_pos(9)+joint_vel(9) removed. |
| **dagger4_run5** | 0.59 @it8 | 0.37 | = run 4 with the **camera set**: `SIM.cfg_file` → `pretrain_multicam_wlr.yaml` (wrist + left + right), plus the re-collected `*_omg_wlr_*` demos it requires. `COMPUTE_ROBOT_POINT_STATE` stays off, so still 2 classes / `pc_channels: 5`. |
| **dagger4_run6** | 0.65 @it4 | 0.18 | = run 4 + **DART** at `dart_ratio: 0.2`. On 20% of approach steps inside (0.05, 0.20] m of the standoff, the executed action becomes a random ±0.04 m / ±0.2 rad jump, so the following steps carry the expert's recovery from off-plan. 4 of 55 keys differ from run 4. |
| **dagger4_run7** | **0.69 @it6** | 0.34 | = run 6 with `dart_ratio: 0.5`. 1 key differs from run 6. Phase 3 found 0.2 ≫ 0.5 (rl_run15 0.458 vs rl_run20 0.25), but that was tuned against a 7-step *index* window, so it does not transfer by assumption. |
| **dagger4_run8** | 0.48 @it25 | 0.48 | = run 4 with `num_iters: 15 → 25` and β `linear 0.75→0.10` → **`piecewise 1.0→0.5→0.3`** (knee at iteration 16). No DART — this is the control for runs 9/10. 6 of 53 keys differ from run 4. |
| **dagger4_run9** | **0.70 @it24** | 0.47 | = run 8 + `dart_ratio: 0.2`. 4 keys differ from run 8. Note the interaction: the DART coin is drawn *before* the β coin, so jolts fire even at β = 1.00 — runs 9/10 keep some off-distribution coverage in the early iterations that run 8 has none of. |
| **dagger4_run10** | 0.60 @it4 | 0.47 | = run 8 + `dart_ratio: 0.5`. 1 key differs from run 9. |
| **dagger4_run11** | *queued* | | = run 10 with `num_iters: 25 → 20` and `train_cfg` → `bc_phase4_pm.yaml`: **`LOSS.pose_loss: pm`**, GA-DDPG's point-matching pose loss on the denormalized action, at `pm_weight: 7.0`. First of three tests of the off-pose finding below. |
| **dagger4_run12** | *queued* | | = run 11 with the pose loss reverted to `smooth_l1`, plus **DART inside the committed reach** (`dart_reach_ratio: 0.3`, ±0.012 m / ±0.3 rad, rejection-sampled against the cloud), and `dart_rot_mag` 0.2 → 0.3. |
| **dagger4_run13** | *queued* | | = run 11 with the pose loss reverted to `smooth_l1`, plus an **auxiliary goal-grasp head** (`MODEL.aux_head`, `LOSS.aux_weight: 1.0`) predicting `[quat, trans]` of the pinned grasp in the current EE frame — GA-DDPG's `extra_pred`, which this policy had dropped. |
| **dagger4_run14** | *queued* | | = run 11 with the pose loss reverted to `smooth_l1`, plus **reach-tail oversampling** (`DATA.reach_tail_weight: 2.5` on a `WeightedRandomSampler`). 2.5 is derived, not picked: it is the weight at which the last 5 steps' share of pose gradient (11.1%) is restored to their share of the data (23.8%). Third of the three single-variable tests. |
| **dagger4_run15** | *queued* | | **Combination run, not an experiment.** = run 14's family with everything stacked: `pm` + aux head + reach-tail weighting, three-camera demos (`pretrain_multicam_wlr` and the `*_omg_wlr_ok` datasets), `num_iters` 25, `dart_ratio` 0.3 with reach-tail DART, DART from step 0, β `piecewise 1.0→0.5→0.3`, `EVAL.every: 1`. Nine changes moving together, so a gain attributes to none of them; runs 11–14 are the controls that make it readable. First run to log the geometric grasp-opportunity metric. |
| **dagger4_run16** | *queued* | | = run 15 with **one** key changed: `beta_schedule: piecewise → constant`, `beta_start: 0.75`. A flat 75/25 expert mix that never anneals — trades DAgger's distribution-shift guarantee for a steady supply of close labels to the last iteration. Clean one-variable comparison against run 15. |
| **dagger4_run17** | *queued* | | = run 16 with **one** key changed: `train_cfg` → `bc_phase4_all_prevact.yaml`, i.e. `MODEL.use_prev_act: false → true`. Robot-MLP input **8 → 14 dims** — the previous *executed* 6-D delta comes back in. The one temporal signal available to a clockless single-frame policy, and the first test of it since run 4; also the copycat risk that kept it off. **Read on `success_rate`, not `val_loss`** — copycat improves both losses. Matched baseline is run 16. |
| **dagger4_run18** | *queued* | | = run 16 with **one** key added: `DAGGER.derive_standoff: false → true`. The pre-grasp standoff is derived from the pinned grasp (`grasp @ translate(0,0,-0.064)`) instead of read off the plan as `FK(traj[-5])`. The two are the **same pose to 5.6e-7 m** — OMG builds the ramp in Cartesian and appends its IK chain without re-optimising it — so the reach commit does **not** move. The only behavioural effect is that step 0 finally has a standoff and sizes its horizon by distance instead of the flat `first_horizon: 20`. A/B on 6 scenes: the one scene whose dynamic horizon still equalled 20 was the one scene that reproduced byte-for-byte, 6/6. Close to a reseed of run 16 — read against the 0.115 noise floor. Forks from 16, not 17. Plus **`max_steps: 50 → 70`** (13.9% of run 16's collection episodes died on the 50-step ceiling while still alive; the benchmark's own limit is 86.7 steps, so 70 spends 81% of it. `EVAL.max_steps` deliberately stays 50 to keep `success_rate` comparable — safe because the BC policy is clockless). Plus **`outcome_check: true`** — per-episode collection outcomes on the evaluator's taxonomy (`c_success_rate`, `co_*`), which needed the close to actually be executed and held since the collector previously `break`s before executing it. Verified D_i is byte-identical with the flag on. Plus **β linear 0.90 → 0.75** (starts above run 16, lands exactly on its flat value, so late iterations stay comparable). Plus **real DART** (`dart_mode: dart_noise`) — Laskey et al. 2017 instead of the uniform jolt: Gaussian noise *added* to the supervisor action with Σ estimated from the learner–supervisor error (Eq. 3) and trace-rescaled (Eq. 4), α annealed 3.0 → 0.5. Measured Σ̂ shows the old jolt's rotation noise was ~2.5× too large and the true covariance is anisotropic. Plus **wrist camera only** (`pretrain.yaml` + the `*_omg_ok` wrist-only base/val/exclusion pair, 472 scenes), reverting runs 15–17's three-camera rig. **Seven changes — attribution is gone by construction; this is a combination run like run 15, not an experiment.** |
| **dagger4_run19** | *blocked* | | = run 16 with **the camera changed and nothing else**: `SIM.cfg_file` → `pretrain_right.yaml`, i.e. **`CAMERAS: ["right"]`** — one fixed side camera, **no wrist/eye-in-hand view at all**. Keeps run 16's flat β=0.75, `max_steps: 50`, and crucially run 16's **original jolt DART** (`dart_mode` unset → defaults to `jolt`; run 18 gated that path behind the flag but removed none of it). So this is the only clean one-variable test of the viewpoint — runs 15/16 moved the camera together with eight other things. Plus `outcome_check: true`, now standard for every new run. Motivation is deployment: `sim2real/` drives a fixed RealSense, so wrist-camera numbers overstate what is reachable on the real robot. **Blocked on data** — no right-camera base dataset exists; `train/val_pinned_omg_right{,_ok}.h5` must be collected against `pretrain_right.yaml` first. Verified the camera builds and yields a sane 1024×5 cloud (896 object / 128 hand, object centroid 0.70–0.78 m from the EE). |
| **dagger4_run20** | *queued* | | = run 16 with **one** key changed: `TRAIN.train_from_scratch: true → false`. Each iteration **warm-starts from the previous iteration's `last.pt`** instead of re-initialising. The training SET is unchanged — still the full aggregate D = base ∪ D₁ ∪ … ∪ Dᵢ for 25 epochs — so this changes the *initialisation*, not the objective; it is **not** sequential fine-tuning on the new shard. Motivated by run 16's own per-iteration logs: `train_pose_pm_mm` starts at **~25 mm on every one of the 26 iterations** (iteration 0's random init is 25.53 mm) and ends at ~14 mm, and from iter_08 on the final train loss is pinned at 0.152–0.158 for **eighteen** iterations while |D| grows 38k → 100k steps. Iteration 0, given 100 epochs, reaches **3.31 mm** — so the per-iteration fits stop far short of the aggregate's argmin and re-pay the same descent 26 times. **Two things to read, both of which could sink it.** (1) The optimiser and LR schedule are *not* carried over — a fresh `CosineAnnealingLR(T_max=25)` hits every warm start with lr = 9.96e-4. Check epoch-0 `train_pose_pm_mm` in `iters/iter_NN/log.csv`: ~14 mm = the warm start held, ~25 mm = the restart wiped it, and the fix is a lower `OPTIM.lr`, not a re-run. (2) The **normalizer freezes** — warm starting loads the previous normalization.npz rather than recomputing over the grown aggregate (deliberate: the output scale is defined by the normalizer), so the base run's statistics propagate through all 25 iterations while run 16 renormalised over a set ending 10× larger and 70% DAgger. That is the one difference not attributable to initialisation. **Also no longer Follow-The-Leader** — a warm start reaches a different local optimum and the sequence becomes path-dependent on shard order; though run 16 never found the argmin either. Plus `outcome_check: true` (logging only, D_i byte-identical), now standard. Cost is unchanged by construction (~11.8 h training); cutting `iter_epochs` is the follow-up run, not this one. |
| dagger4_smoke | — | — | Shakedown only. m=4, 2 iterations, 6 eval scenes. Not a result. |

Runs 6–10 are a 2×2 with two DART-free controls: `{run 4's β schedule, the
extended one} × {DART 0.2, DART 0.5}`, with runs 4 and 8 as the controls. **The
2×2 did not resolve** — see the noise floor below. Runs 11–13 are three
independent single-variable tests of the off-pose finding, sharing run 10 as
control.

### The noise floor, finally measured — and it invalidates the 2×2

All six of runs 4, 6, 7, 8, 9, 10 train iteration 0 from an **identical**
configuration: same `train_cfg`, same `base_train_h5`, same `val_h5`,
`base_epochs: 100`, `seed: 0`. DART cannot touch it (it perturbs collection at
iterations ≥ 1) and neither can the β schedule or `num_iters`. Verified key by key.

Their six iteration-0 success rates:

```
0.32   0.34   0.39   0.49   0.51   0.62        mean 0.445   sd 0.115   range 0.30
```

Binomial sampling on 100 scenes at p ≈ 0.45 would give sd 0.050. The observed
0.115 is **2.3× that**, so most of the spread is training nondeterminism producing
genuinely different policies, not eval sampling.

A ±0.115 floor is larger than every difference in the 2×2. Peak success
(0.51 / 0.65 / 0.69 / 0.48 / 0.70 / 0.60) cannot be read as a DART ranking, and
run 4's own best iteration being its *base fit* is hard to read as anything but
noise. This is the Phase-4 analogue of runs 28 vs 38, and it was predicted in this
document before the runs went out; the prediction was correct and the cost was the
whole family.

What is still readable, because it is counted rather than scored: DART at 0.5
costs collection horizon (`c_max_steps` 84 → 215 between runs 4 and 7, ~7.5
jolts/episode, steps/episode 28.5 → 35.1), while eval-time `f_timeout` moves the
*other* way (0.109 → 0.073). And `reached_grasp` collapses in **all** arms, DART or
not — that pathology is not what DART addresses.

### The finding that reframes Phase 4: the policy never reaches the pinned grasp

Across runs 4/6/7/8/9/10 — 69 evaluations of 100 scenes each — `near_rate` (closed
within the label tolerances of 0.02 m / 0.34 rad) never exceeded **0.12**, and was
usually 0.00–0.04. Run 7's best checkpoint scores 0.69 success at `near_rate`
0.06. **Essentially every success is a grasp at some other pose that the object
happened to survive.**

`eval_min_rot` — the closest the gripper's *orientation* ever comes to the pinned
grasp, over the whole episode — never went below **0.43 rad** in any evaluation of
any run, against a 0.34 threshold. Not a spread; a plateau.

The metric is sound, checked two ways. The scene is static
(`MANO_SIMULATION_MODE: disable_control_and_move_by_reset`), so the world-frame
pinned pose stays valid all episode. And the expert, scored by the *same* function
against the *same* poses during collection, reaches 0.014–0.023 m / 0.04–0.10 rad.

| closest approach ever reached | expert | policy | threshold |
|---|---|---|---|
| position | 0.018 m | 0.076 m | 0.02 |
| rotation | 0.07 rad | 0.60 rad | 0.34 |

**Rotation error integrates; position error does not.** Run 7's best checkpoint has
a per-step rotation error of 0.0230 rad over ~25-step episodes — 0.0230 × 25 =
0.57 rad, and its measured `eval_min_rot` is 0.559. Position would integrate to
0.22 m by the same arithmetic, but `eval_min_pos` is 0.076, because the wrist
camera sees where the object is. There is closed-loop feedback on position and
none on orientation.

Three candidate causes, one per run 11–13:

1. **The loss cannot see rotation properly.** Per-channel z-scoring makes all six
   action channels unit-variance, but 1σ of translation moves the gripper control
   points 16–22 mm while 1σ of rotation moves them 2.2–4.9 mm. → run 11 (PM loss).
   *Note this cuts against the obvious reading: z-scoring already over-weights
   rotation 4–8× relative to its physical effect, so PM will shift weight toward
   translation.*
2. **The demonstrations never show orientation being corrected near the object.**
   OMG aligns the wrist during the free approach then reaches in a straight line:
   per-axis mean `|drot|` over the last four demonstration steps is 0.0005–0.0136
   rad against 0.050–0.059 during the approach. → run 12 (DART in the reach).
3. **The observation never says which orientation to arrive at.** No goal-pose
   head, and `pc_pretrained` loads an encoder that *was* trained with one before
   being fine-tuned without it for 100 epochs × N iterations. → run 13 (aux head).

Correlations over 55 eval points sharpen the failure modes: `f_drop` rises with
distance from the grasp (r = +0.48 vs `eval_min_pos`) while `f_human_contact`
*falls* (r = −0.31). Hanging back gives drops and timeouts; pressing in gives hand
contact. Only an orientation-correct approach breaks that trade-off, which is why
success plateaus around 0.5–0.7 regardless of what else is changed.

### The earlier findings — the joint state, and the β floor

**Dropping the joint state is the single largest effect measured in this project.**
Runs 3 and 4 train iteration 0 on the *same* HDF5 file; the only difference is
`drop_joint_state`. The base BC policy went **0.09 → 0.51** — before any DAgger
iteration ran at all. The joint configuration was acting as a scene-identity
handle: with ~350 usable scenes the arm pose is close to a scene ID, and the
network was indexing off it instead of reading the point cloud.

The corroborating detail is that run 4 fits the expert data **worse**:
`best_val_loss` runs 0.12 → 0.25 against run 3's 0.07 → 0.11, while success is 3–4×
higher throughout. Validation loss was substantially measuring memorization.

It also fixed the failure mode run 3 was stuck on. `f_timeout` falls from
0.21–0.63 (run 3) to 0.03–0.16 (run 4): the approach, not the grasp, was what the
joint state was breaking.

**Run 3's late decline is the β floor.** Its `reached_grasp` collapses 87 → 73 →
60 → 45 over iterations 10–15 as β anneals to 0.10, and success follows (0.35 →
0.26 → 0.17). Run 4 shows no such collapse over the same span. A floor nearer
0.25 is worth testing.

---

## Phase 3 — online RL, TD3+BC (`output/rl_runs/`)

`peak` / `final` are `eval_succ`. Run 44 was never launched.

### Bring-up (workstation, 2000-iter granularity)

| run | peak | final | what is different |
|---|---|---|---|
| 1 | 0.00 | 0.00 | Baseline `rl_phase1.yaml`. `pg_normalize: false` → diverged. |
| 2 | 0.00 | 0.00 | `pg_normalize: true`. |
| 3 | 0.00 | 0.00 | Adds offline pretrain (2000 updates), demo-frac ramp 0.5→0.1, `warmup_episodes` 20→0. |
| 4 | 0.00 | 0.00 | `eval_episodes` 20→50. |
| 5 | 0.05 | 0.00 | Config identical to run 4 (rerun). |
| 6 | 0.00 | 0.00 | `demo_frac_end` 0.1→0.3, `expert_episode_frac` 0.25, `expert_initial_steps` 15. |
| 7 | 0.05 | 0.05 | `expert_initial_steps` 25, `rollout_max_steps` 30, `alpha` 2.5→1.0, `bc_weight` 2.0, `gripper_close_weight_max` 10. |
| 8 | 0.00 | 0.00 | DAgger tail splice on: `dagger_ratio` 0.5, `dagger_min_step` 5, `dagger_tail_guard` 8; `alpha` 0.1. |
| 9 | 0.30 | 0.00 | Expert-initial **anneal** (550 iters, 28→2 steps, window 6). First run to reach 0.30. |
| 10 | 0.15 | 0.05 | Anneal 1000 iters, `bc_weight` 2→**80**, `pose_loss` → `pm`, `alpha` 0.25. |

### Cluster rescale (16 workers, 250-iter granularity)

| run | peak | final | what is different |
|---|---|---|---|
| 11 | 0.25 | 0.13 | Pure re-granulation for 16 parallel workers: `episodes_per_iter` 2→16, `num_iters` 2000→250, `updates_per_iter` →800. Same RL dynamics. |
| 12 | 0.17 | 0.04 | `dagger_ratio` 0.5→1.0. |
| 13 | 0.21 | 0.04 | Longer: `num_iters` 250→500, curriculum ramps stretched to match. |
| 14 | 0.17 | 0.17 | Back to run-11 length; adds dense shaping (`shaping_pos_weight` 1.0, `rot` 0.3). |
| **15** | **0.458** | 0.13 | Shaping **off**; **DART** on (`dart_ratio` 0.2, steps 15–22, pos 0.04 / rot 0.2). **Highest peak of any RL run.** |
| 16 | 0.25 | 0.00 | `dart_mode: expert`, `dart_ratio` 0.5, range 5–24, `expert_episode_frac` 0.5. |
| 17 | 0.29 | 0.13 | `dart_mode: both`. |
| 18 | 0.21 | 0.13 | `dart_mode: policy`, `action_reg_weight` 0.1. |
| 19 | 0.25 | 0.13 | `dart_mode: both`, action-reg removed. |
| 20 | 0.25 | 0.13 | `dart_mode: policy`, `dart_ratio` 0.5. |
| 21 | 0.27 | 0.08 | Paper's **offline hand-collision filter** (`valid_grasp_dict_005.pkl`); runtime 0.08 m filter off. |
| 22 | 0.21 | 0.08 | Reverts to the runtime filter (A/B against 21). |

### Widening curriculum family

| run | peak | final | what is different |
|---|---|---|---|
| 23 | 0.19 | 0.13 | **Widening** reverse curriculum: hold `expert_initial_hi` = 24, anneal `lo` 22→2 over 150 iters, so near-grasp practice is retained instead of abandoned. |
| 24 | 0.19 | 0.10 | = 23 with `dart_mode: both`, `dart_ratio` 0.5. |
| 25 | 0.21 | 0.02 | = 23 with `num_iters` 250→700 (long post-curriculum hold). |
| 26 | 0.19 | 0.08 | Flat `demo_frac: 0.05` (ramp disabled), 250 iters. |
| 27 | 0.21 | 0.06 | run-25 base, 250 iters, **`warm_start: false`** — actor+critic from scratch. |
| **28** | **0.375** | 0.02 | = 27 + **`drop_joint_state: true`** (robot MLP 26→8). The Phase-3 origin of the flag that later transformed Phase-4 run 4. Peaked at iter 133 then decayed. |
| 29 | 0.00 | 0.00 | Paper's loss recipe: `pg_normalize false`, `pose_loss pm`, `mc_blend 0`, `aux 1.0`, `bc 1.0`. **Diverged** (actor exploited Q; `a_absmean` → 1e8). |
| 30 | 0.29 | 0.13 | ACT actor (`arch: act`, 4-frame history, `chunk_len` 1) on run-28's base. |
| 31 | 0.00 | 0.00 | run 29 + `pg_normalize: true` — the minimal stability fix. Still 0. |

### Anti-decay probes (all vs run 28)

| run | peak | final | what is different |
|---|---|---|---|
| 32 | 0.21 | 0.17 | Buffer 20k→**100k**, `demo_frac` 0.7 flat, `expert_initial_lo_end` 2→10. |
| 33 | 0.10 | 0.00 | **Multicam** `[left, right]`, 3 classes → `pc_channels` 6. Collapsed to 0. |
| 34 | 0.19 | 0.08 | `demo_frac` 0.9, `num_iters` 500. |
| 35 | 0.17 | 0.04 | `demo_frac` back to 0.3, `lo_end` 15. |
| 36 | 0.27 | 0.08 | Buffer 100k, `lo_end` 2, 500 iters. |
| 37 | 0.27 | 0.06 | **Multicam `[wrist, left]`**, 6-ch. Best fixed-camera run — did not collapse (cf. 33). |
| **38** | 0.34 | **0.30** | **Config byte-identical to run 28.** Highest *final* of any RL run. See below. |
| 39 | 0.23 | 0.13 | Fine-grained: 3 episodes/iter, 1333 iters, 150 updates/iter. |
| 40 | 0.25 | 0.08 | Buffer **200k** (near-non-evicting), 500 iters, curriculum stretched over the first half. |
| 41 | 0.31 | 0.08 | **Deeper** net: `policy_hidden` / `q_hidden` [256,256] → [256,256,256,256]. 100k buffer, 500 iters. |
| 42 | 0.33 | 0.06 | **Smaller** net: heads → [128,128], `feature_dim` 256→128. Pairs with 41 as a two-sided capacity A/B. |
| 43 | 0.23 | 0.00 | Extend the winner: run 37 (wrist+left) + 100k buffer + 500 iters. |
| 45 | 0.29 | 0.10 | `aux_weight` 0.5→**5.0** (goal-auxiliary grasp-pose head), 100k buffer. |
| 46 | 0.23 | 0.19 | `aux_weight` **25.0** (~co-dominant with the BC term). |
| 47 | 0.29 | 0.06 | = run 28 with `RL.gamma` 0.95→**0.99** (effective horizon 20→100 against `rollout_max_steps` 30). |

Not results: `rl_smoke` (pipeline shakedown) and `rl_diag16` (a diagnostic pass
over run 16, not a training run).

### The result that recontextualises this section

**Run 38's saved config is byte-identical to run 28's**, and the two behaved
completely differently:

```
iter      7    28    49    70    91   112   133   154   175   196   217   238
run 28  0.04  0.00  0.02  0.00  0.02  0.19  0.38  0.06  0.10  0.06  0.02  0.02
run 38  0.00  0.02  0.06  0.16  0.28  0.16  0.28  0.32  0.34  0.28  0.34  0.14
```

Run 28 spiked at iteration 133 and decayed to ~0.02. Run 38 climbed steadily and
held 0.28–0.34 for the second half.

Runs 32, 34, 35, 36, 40, 41 and 42 were all designed to cure run 28's
"post-curriculum decay" — buffer size, curriculum shape, network capacity. That
decay **did not reproduce on an identical config**. So run-to-run variance at 250
iterations is large enough to swamp every single-knob comparison in that family,
and none of those seven runs can be read as evidence for or against its knob.

Caveat: `--demos` is not recorded in the saved config, so an undetected
difference in the demonstration pool between 28 and 38 cannot be ruled out from
the run directories alone. Either way, the conclusion for future work is the
same — **repeat before attributing**, and prefer a paired A/B over a single run.

---

## Phase 1 / 2 — BC and offline DAgger (`output/bc_runs/`)

| run | kind | what it is |
|---|---|---|
| `phase1_full` | MLP | The original Phase-1 behaviour-cloning fit on `train.h5`, 100 epochs. |
| `run2` | MLP | Phase-1 refit with `use_prev_act: false` recorded explicitly. |
| `dagger_iter1`, `dagger_iter2`, `dagger_iter12` | MLP | First manual/offline DAgger rounds — collect a round, retrain, 50 epochs each. |
| `dagger_iter_2_1/2/3` | MLP | Second offline DAgger series. `dagger_iter_2_3` is the BC run most Phase-3 jobs warm-start from. |
| `frozen_pc_run1`, `frozen_pc_dagger_iter1/2` | MLP | `FREEZE_PARTIAL_POINTCLOUD` experiment — capture the cloud at a fixed step and hold it. |
| `act_base`, `act_run1` | ACT | Phase-2 temporal transformer + action chunking, 100 epochs on `train.h5`. |
| `dagger_act_iter1/2/3` | ACT | Offline DAgger rounds for the ACT policy. |

---

## Config → run map

Configs without a numbered run file were shared across several runs.

| config | used by |
|---|---|
| `rl_phase1.yaml` | runs 1–10 (workstation) |
| `rl_phase1_cluster.yaml`, `_r15both`, `_ratio05`, `_w3`, `_areg`, `_both`, `_vgd` | runs 11–22 |
| `rl_phase1_cluster_r23…r47.yaml` | the correspondingly numbered run |
| `dagger_phase4.yaml` | dagger4_run1, run2 |
| `dagger_phase4_omg.yaml` | dagger4_run3 |
| `dagger_phase4_omg_nojoint.yaml` | dagger4_run4 |
| `dagger_phase4_omg_nojoint_multicam.yaml` | dagger4_run5 |
| `dagger_phase4_dart02.yaml`, `_dart05.yaml` | dagger4_run6, run7 |
| `dagger_phase4_beta_ext.yaml` | dagger4_run8 |
| `dagger_phase4_beta_ext_dart02.yaml`, `_dart05.yaml` | dagger4_run9, run10 |
| `dagger_phase4_pm.yaml` | dagger4_run11 |
| `dagger_phase4_reachdart.yaml` | dagger4_run12 |
| `dagger_phase4_aux.yaml` | dagger4_run13 |
| `dagger_phase4_reachw.yaml` | dagger4_run14 |
| `dagger_phase4_all.yaml` | dagger4_run15 |
| `dagger_phase4_all_beta075.yaml` | dagger4_run16 |
| `dagger_phase4_all_beta075_prevact.yaml` | dagger4_run17 |
| `dagger_phase4_all_beta075_derivstandoff.yaml` | dagger4_run18 |
| `dagger_phase4_right_cam.yaml` | dagger4_run19 |
| `dagger_phase4_all_beta075_warmstart.yaml` | dagger4_run20 |
| `bc_phase1.yaml` / `bc_phase1_nojoint.yaml` | Phase-4 `TRAIN.train_cfg` |
| `bc_phase4_pm.yaml` | run 11 `TRAIN.train_cfg` — `LOSS.pose_loss: pm`, `pm_weight: 7.0` |
| `bc_phase4_aux.yaml` | run 13 `TRAIN.train_cfg` — `MODEL.aux_head`, `LOSS.aux_weight: 1.0` |
| `bc_phase4_reachw.yaml` | run 14 `TRAIN.train_cfg` — `DATA.reach_tail_weight: 2.5` |
| `bc_phase4_all.yaml` | runs 15, 16 `TRAIN.train_cfg` — pm + aux + reach-tail weighting |
| `bc_phase4_all_prevact.yaml` | run 17 `TRAIN.train_cfg` — same, `MODEL.use_prev_act: true` |

---

## Standing conclusions

**`drop_joint_state` is the highest-value change found.** Phase-3 run 28 (0.375,
the second-best RL peak) and Phase-4 run 4 (0.51 base, 5.7× run 3) both come from
it. Joint position and velocity are redundant with `ee_pose` under an EE-frame
action, and scene-correlated enough to memorize.

**Validation loss is a poor selection target once the joint state is dropped.**
Run 4 has roughly double run 3's `best_val_loss` and 3–4× its success rate.
`EVAL.ckpt: best` selects on val loss, which now measures partly the wrong thing.

**Expert demo failure rate does not predict policy performance.** The `omg` pin
rule produces *more* failed demonstrations than `furthest_from_hand` (24.2% vs
20.2%) and a distinctly better policy — it is the easier target for a learner to
track even though it is the harder one for the planner to fly.

**Single runs are not decisive in EITHER phase, and the floor is now measured.**
Phase 3 had runs 28 vs 38 (byte-identical configs, 0.02 vs 0.30) as an accident.
Phase 4 has six deliberate samples of the same base fit: **0.32 – 0.62, sd 0.115**,
2.3× the binomial expectation. Any Phase-4 comparison resting on a success-rate
difference smaller than ~0.15 on 100 scenes is unreadable, which retrospectively
covers the whole 6–10 DART family. Budget a repeat before attributing an effect,
and prefer metrics with a floor near zero — `near_rate`, `eval_min_rot` — which
have room to move and are not dominated by this variance.

**Success rate is measuring the wrong thing.** `stable_grasp` rewards coming away
with the object from anywhere; the demonstrations teach one specific pinned pose.
The policy solved the first and ignored the second — 0.69 success at `near_rate`
0.06. Any claim that this work "learns the demonstrated grasp" needs `near_rate`
reported beside the success rate, and right now that number is ~0.

**Orientation has no feedback loop; position does.** Per-step rotation error
integrates almost exactly (0.0230 rad × ~25 steps = 0.57 vs measured 0.559) while
position error does not (would be 0.22 m, measured 0.076). The wrist camera
localizes the object but nothing in the observation names the orientation to
arrive at. This is the standing explanation for the ~0.5–0.7 plateau, and runs
11–13 test its three candidate causes.
