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
| **dagger4_run4** | **0.51 @it0** | 0.36 | = run 3 with **one** key changed: `train_cfg` → `bc_phase1_nojoint.yaml`, i.e. `MODEL.drop_joint_state: true`. Robot-MLP input **26 → 8 dims** (`ee_pos`, `ee_orn`, `gripper`); joint_pos(9)+joint_vel(9) removed. |
| **dagger4_run5** | 0.53 @it2 | 0.46 (it4, running) | = run 4 with the **camera set**: `SIM.cfg_file` → `pretrain_multicam_wlr.yaml` (wrist + left + right), plus the re-collected `*_omg_wlr_*` demos it requires. `COMPUTE_ROBOT_POINT_STATE` stays off, so still 2 classes / `pc_channels: 5`. |
| **dagger4_run6** | *queued* | | = run 4 + **DART** at `dart_ratio: 0.2`. On 20% of approach steps inside (0.05, 0.20] m of the standoff, the executed action becomes a random ±0.04 m / ±0.2 rad jump, so the following steps carry the expert's recovery from off-plan. 4 of 55 keys differ from run 4. |
| **dagger4_run7** | *queued* | | = run 6 with `dart_ratio: 0.5`. 1 key differs from run 6. Phase 3 found 0.2 ≫ 0.5 (rl_run15 0.458 vs rl_run20 0.25), but that was tuned against a 7-step *index* window, so it does not transfer by assumption. |
| **dagger4_run8** | *queued* | | = run 4 with `num_iters: 15 → 25` and β `linear 0.75→0.10` → **`piecewise 1.0→0.5→0.3`** (knee at iteration 16). No DART — this is the control for runs 9/10. 6 of 53 keys differ from run 4. |
| **dagger4_run9** | *queued* | | = run 8 + `dart_ratio: 0.2`. 4 keys differ from run 8. Note the interaction: the DART coin is drawn *before* the β coin, so jolts fire even at β = 1.00 — runs 9/10 keep some off-distribution coverage in the early iterations that run 8 has none of. |
| **dagger4_run10** | *queued* | | = run 8 + `dart_ratio: 0.5`. 1 key differs from run 9. |
| dagger4_smoke | — | — | Shakedown only. m=4, 2 iterations, 6 eval scenes. Not a result. |

Runs 6–10 are a 2×2 with two DART-free controls: `{run 4's β schedule, the
extended one} × {DART 0.2, DART 0.5}`, with runs 4 and 8 as the controls. If
DART's effect has the same sign under both schedules it is a property of DART;
if it flips, it is an interaction with how much of the collection the expert
drives. **Caveat that applies to all four DART readings:** Phase 4 has no
measured run-to-run noise floor — rl_run28 vs rl_run38 (byte-identical configs,
0.02 vs 0.30) shows Phase 3 had enough seed variance to swamp an effect this
size, and nothing rules that out here. A repeat of run 4 under a different seed
costs the same as one DART variant and would make all four readable.

### The two findings that matter

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
| `bc_phase1.yaml` / `bc_phase1_nojoint.yaml` | Phase-4 `TRAIN.train_cfg` |

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

**Single 250-iteration RL runs are not decisive.** See runs 28 vs 38.
