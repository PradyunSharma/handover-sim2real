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

`best iter` / `final` are `success_rate` over 100 eval scenes (peak checkpoint
vs. last iteration). All runs below have now completed — runs 11–20 were
previously logged here as `queued`/`blocked`; their `dagger_log.csv` and
`state.json` are in fact filled through their configured `num_iters`, so the
real numbers replace the placeholders. Run 21 alone has no run directory yet
(config only) and has not been executed.

DART ratio is `dart_ratio` (free approach) / `dart_reach_ratio` (inside the
committed reach). DART variant is `dart_mode`: **replace** ("jolt" — the
executed action is swapped for a random ±pos/±rot jump, unset defaults to
this) vs. **add noise** (`dart_noise` — Gaussian noise is added on top of the
supervisor's own action, Σ estimated from learner–supervisor error, Laskey et
al. 2017). Aux task is `MODEL.aux_head` — an auxiliary head predicting the
pinned grasp `[quat, trans]` in the current EE frame. β params are
`beta_start`→(`beta_mid`→)`beta_end` for the given `beta_schedule`.

| run | camera | iters | init | β | DART free | DART reach | DART variant | aux task | loss | model flags | best iter | final | Δ from predecessor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **run1** | wrist | 20 | scratch | indicator (β≈0) | — | — | — | no | smooth_l1 | joint state kept; m=20/iter, 50ep | 0.13 @it4 [^holdout] | 0.04 | First real run. Failed: β=0 gave one close label total. |
| **run2** | wrist | 15 | scratch | linear 0.75→0.10 | — | — | — | no | smooth_l1 | joint state kept | 0.21 @it2 | 0.11 | = run1: β floor + filtered dataset. |
| **run3** | wrist | 15 | scratch | linear 0.75→0.10 | — | — | — | no | smooth_l1 | joint state kept | 0.35 @it10 | 0.17 | = run2: pin rule furthest_from_hand → omg. |
| **run4** | wrist | 15 | scratch | linear 0.75→0.10 | — | — | — | no | smooth_l1 | nojoint | 0.51 @it0 | 0.27 | = run3: drop joint state (single largest effect in the project). |
| **run5** | wrist+L+R | 15 | scratch | linear 0.75→0.10 | — | — | — | no | smooth_l1 | nojoint | 0.59 @it8 | 0.37 | = run4: 3-camera. |
| **run6** | wrist | 15 | scratch | linear 0.75→0.10 | 0.2 | — | replace | no | smooth_l1 | nojoint | 0.65 @it4 | 0.18 | = run4: + DART 0.2. |
| **run7** | wrist | 15 | scratch | linear 0.75→0.10 | 0.5 | — | replace | no | smooth_l1 | nojoint | 0.69 @it6 | 0.34 | = run6: DART 0.5. |
| **run8** | wrist | 25 | scratch | piecewise 1.0→0.5→0.3 | — | — | — | no | smooth_l1 | nojoint | 0.48 @it25 | 0.48 | = run4: 25 iters, β piecewise (DART-free control). |
| **run9** | wrist | 25 | scratch | piecewise 1.0→0.5→0.3 | 0.2 | — | replace | no | smooth_l1 | nojoint | 0.70 @it24 | 0.47 | = run8: + DART 0.2. |
| **run10** | wrist | 25 | scratch | piecewise 1.0→0.5→0.3 | 0.5 | — | replace | no | smooth_l1 | nojoint | 0.60 @it4 | 0.47 | = run8: + DART 0.5. |
| **run11** | wrist | 20 | scratch | piecewise 1.0→0.5→0.3 | 0.5 | — | replace | no | pm (w=7) | nojoint | 0.60 @it18 | 0.34 | = run10: loss → pm. |
| **run12** | wrist | 20 | scratch | piecewise 1.0→0.5→0.3 | 0.5 | 0.3 | replace | no | smooth_l1 | nojoint | 0.74 @it6 | 0.56 | = run11: loss reverted; + reach-tail DART. |
| **run13** | wrist | 20 | scratch | piecewise 1.0→0.5→0.3 | 0.5 | — | replace | yes | smooth_l1 | nojoint | 0.68 @it18 | 0.38 | = run11: loss reverted; + aux grasp head. |
| **run14** | wrist | 20 | scratch | piecewise 1.0→0.5→0.3 | 0.2 | — | replace | no | smooth_l1 | nojoint, reach-tail ×2.5 | 0.69 @it10 | 0.42 | = run11: loss reverted; + reach-tail oversampling. |
| **run15** | wrist+L+R | 25 | scratch | piecewise 1.0→0.5→0.3 | 0.3 | 0.3 | replace | yes | pm (w=7) | nojoint, reach-tail ×2.5 | 0.81 @it8 | 0.76 | Combination run (9 changes stacked): pm+aux+reachw+3cam+DART. Not a clean test. |
| **run16** | wrist+L+R | 25 | scratch | constant 0.75 | 0.3 | 0.3 | replace | yes | pm (w=7) | nojoint, reach-tail ×2.5 | 0.80 @it16 | 0.79 | = run15: β → constant 0.75 (no anneal). |
| **run17** | wrist+L+R | 25 | scratch | constant 0.75 | 0.3 | 0.3 | replace | yes | pm (w=7) | nojoint, reach-tail ×2.5, prev-act (14-D state) | 0.57 @it15 | 0.40 | = run16: + prev-action input. |
| **run18** | wrist | 25 | scratch | linear 0.90→0.75 | 0.3 | 0.3 | add noise (α 3.0→0.5) | yes | pm (w=7) | nojoint, reach-tail ×2.5, derive_standoff, max_steps=70, outcome_check | 0.78 @it14 | 0.69 | Combination run off run16 (7 changes): wrist-only + Gaussian DART + derive_standoff + max_steps 70 + outcome_check + β 0.90→0.75. |
| **run19** | right | 25 | scratch | constant 0.75 | 0.3 | 0.3 | replace | yes | pm (w=7) | nojoint, reach-tail ×2.5, outcome_check | 0.72 @it19 | 0.43 | = run16: camera → right-only, fixed side view, nothing else. |
| **run20** | wrist+L+R | 25 | warm-start | constant 0.75 | 0.3 | 0.3 | replace | yes | pm (w=7) | nojoint, reach-tail ×2.5, outcome_check | 0.76 @it8 | 0.71 | = run16: warm-start each iter instead of from-scratch. |
| **run21** | wrist+right | 25 | warm-start (last.pt) | constant 0.75 | 0.3 | — | replace | yes | pm (w=7) | nojoint, reach-tail ×2.5, derive_standoff, outcome_check, target=pregrasp, settle=1, **PointNet++ from scratch** | not yet run | not yet run | = run16: reach made open-loop (blind commit push instead of a learned grasp). Landing error = arrival error one-for-one (corr 0.990) — the blind push cannot re-measure anything, so whatever offset the gripper has at the commit it still has at the grasp. The arm realises a flat 0.775 of any commanded translation in one 150 ms control window (p1 0.740, p99 0.809, no small-command deadband), and the base collector never re-commands a waypoint, so the demos pass the pre-grasp 11.6 mm short. `commit_settle_steps: 1` re-commands the frozen pre-grasp once, taking that to ~2.5 mm — better than the demos' own closed-loop reach (4.6 mm from the pinned grasp) — while keeping a 9 mm margin between the last APPROACH state and the COMMIT state, which a 1024-point cloud can resolve. More settle steps collapse that margin and the policy stops committing at all. Requires D_0 **re-collected** by `collect_pregrasp_demos.py` (Phase-4 collector at beta=1) rather than truncated, since the truncated set labels the 11.6 mm state COMMIT and would contradict every shard; that collection doubles as the smoke test, since its expert success rate is this run's ceiling. Reach-band DART off — its magnitudes were derived for 1.6 cm reach steps 5 cm from the object, and the window is now 11.4 cm out. Camera drops to **wrist + right** (`pretrain_multicam_wr.yaml`), matching what `sim2real/` can actually produce. **PointNet++ is initialised randomly** (`TRAIN.pc_pretrained:` left empty, the config route to what `train_bc.py` exposes as `--pc-pretrained none`) — the first run ever to not borrow the CVPR2023 encoder, which was trained on a different rig; affects iteration 0 only. Each iteration then **warm-starts from the previous iteration's `last.pt`** (`train_from_scratch: false`, `init_ckpt: last`), as in run 20 — the chain starts at run 21's OWN iteration 0 (`base_run: null`), nothing is borrowed from another run. That warm start matters more here than in run 20: a random encoder refit for 25 epochs every iteration would never recover what iteration 0's 100 epochs bought. **Four changes moving together with the endgame — this is a combination run, not a controlled test.** |
| dagger4_smoke | wrist | 2 | scratch | indicator (β≈0) | — | — | — | no | smooth_l1 | joint state kept | — | — | Shakedown only. m=4, 6 eval scenes. Not a result. |

[^holdout]: run1 is the only run with `EVAL.holdout: true` — a genuine held-out
split. Every other run evaluates with `holdout: false` (training-set
performance; see the caveat at the top of this doc), so run1's number is not
directly comparable to the rest on that axis alone — it also failed outright
for the β=0 reason stated above.

Runs 6–10 are a 2×2 with two DART-free controls: `{run 4's β schedule, the
extended one} × {DART 0.2, DART 0.5}`, with runs 4 and 8 as the controls. **The
2×2 did not resolve** — see the noise floor below. Runs 11–13 are three
independent single-variable tests of the off-pose finding, sharing run 10 as
control.

### Rolling out any of these runs

Set the environment once per shell, from the repo root:

```bash
export OMG_PLANNER_DIR=$PWD/OMG-Planner GADDPG_DIR=$PWD/GA-DDPG
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/OMG-Planner/orocos_kinematics_dynamics/orocos_kdl/release/lib"

R=output/dagger_runs
RO="python examples/rollout_bc_policy.py --max-steps 50 --show-goal-grasp \
    --grasp-pin-table output/grasp_pin_table_train_omg.json"
```

Then one line per run — the **only** thing that changes is `--cfg-file`, which
must match the camera rig the run trained on, and `--scene`:

```bash
# --- wrist only -----------------------------------------------------------
$RO --run-dir $R/dagger4_run3/best  --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run4/best  --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run6/best  --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run7/best  --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run8/best  --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run9/best  --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run10/best --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run11/best --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run12/best --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run13/best --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run14/best --cfg-file examples/pretrain.yaml --scene 0
$RO --run-dir $R/dagger4_run18/best --cfg-file examples/pretrain.yaml --scene 0

# --- wrist + left + right -------------------------------------------------
$RO --run-dir $R/dagger4_run5/best  --cfg-file examples/pretrain_multicam_wlr.yaml --scene 0
$RO --run-dir $R/dagger4_run15/best --cfg-file examples/pretrain_multicam_wlr.yaml --scene 0
$RO --run-dir $R/dagger4_run16/best --cfg-file examples/pretrain_multicam_wlr.yaml --scene 0
$RO --run-dir $R/dagger4_run17/best --cfg-file examples/pretrain_multicam_wlr.yaml --scene 0
$RO --run-dir $R/dagger4_run20/best --cfg-file examples/pretrain_multicam_wlr.yaml --scene 0

# --- right only (no wrist) ------------------------------------------------
$RO --run-dir $R/dagger4_run19/best --cfg-file examples/pretrain_right.yaml --scene 0

# --- wrist + right, PRE-GRASP endgame (see below) -------------------------
$RO --run-dir $R/dagger4_run21/best --cfg-file examples/pretrain_multicam_wr.yaml --scene 0

# --- runs 1-2 only: the OLDER pin table ----------------------------------
python examples/rollout_bc_policy.py --max-steps 50 --show-goal-grasp \
    --grasp-pin-table output/grasp_pin_table_train.json \
    --run-dir $R/dagger4_run1/best --cfg-file examples/pretrain.yaml --scene 0
python examples/rollout_bc_policy.py --max-steps 50 --show-goal-grasp \
    --grasp-pin-table output/grasp_pin_table_train.json \
    --run-dir $R/dagger4_run2/best --cfg-file examples/pretrain.yaml --scene 0
```

In the PyBullet window: **N** = next scene, **R** = re-roll the same one,
**Q** = quit. Pass `--scenes 30,86,32` to walk a specific list with **N**.

Four things about that template are load-bearing:

**`--cfg-file` must match the run's camera rig.** It sets `CAMERAS`, which
determines what the point cloud contains. A mismatch is not a mild degradation —
the policy gets an input distribution it never saw, and a healthy run looks
broken. The grouping above is `SIM.cfg_file` read straight out of each
`<run>/config.yaml`.

**`--max-steps 50`** because the script defaults to 30 while every Phase-4 run
evaluated at `EVAL.max_steps: 50`. At 30 a policy that commits late is cut off
and reads as a `TIMEOUT` that the run itself never had.

**The pin table is the overlay's ground truth, not the policy's.** It only
affects the green `--show-goal-grasp` gripper — but with the wrong one that
gripper is drawn at a grasp the run was never aiming at, so a correct rollout
looks like a miss. Runs 1–2 predate the `_omg` pin rule and need
`grasp_pin_table_train.json`; runs 3–21 use `grasp_pin_table_train_omg.json`.

**`--split` and the pin table move together.** All the commands above are on the
train split, which is where these runs were collected and evaluated. To roll out
a *test*-split scene, add `--split test` **and** switch to
`output/grasp_pin_table_test_omg.json` in the same edit — scene indices are
numbered within a split, so scene 30 of test is not scene 30 of train.

**Run 21 needs the open-loop endgame, and gets it automatically.** With
`DAGGER.target: pregrasp`, channel 6 of the action means "commit the blind push",
not "close the fingers"; closing in place would shut the jaws 6.4 cm short of the
object on every episode. `--target` defaults to `auto`, which reads
`DAGGER.target` out of `<run>/config.yaml`, so nothing extra is needed — but
check the startup line says `Endgame: PRE-GRASP` before trusting the result.
Force it with `--target pregrasp` if the checkpoint has been copied away from its
run directory. See `docs/thesis_phase4_dagger.md` §7.1.

**Most weights are not on the workstation.** Only runs **1, 2, 7, 12, 13, 15, 16**
have a local `best/checkpoints/`; everything else has logs only and needs the
snapshot pulled from the cluster first:

```bash
rsync -avz delftblue:/scratch/pradyunsharma/handover-sim2real/output/dagger_runs/dagger4_runNN/best/ \
    output/dagger_runs/dagger4_runNN/best/
```

### Scoring any of these runs on the **test** split

**None of the numbers in the table above are generalisation numbers.** Every
Phase-4 run sets `SIM.split: train`, and every run except run 1 sets
`EVAL.holdout: false`, which puts the eval scenes back into the collection pool
— so the `best iter` / `final` columns are performance on scenes the policy was
trained on. Measured on run 16 that gap is eleven points: **0.80 logged, 0.692 on
test**. `examples/eval_run_scenes.py` produces the honest figure.

Same environment block as above, then:

```bash
R=output/dagger_runs
EV="python examples/eval_run_scenes.py --split test --box-probe \
    --grasp-pin-table output/grasp_pin_table_test_omg.json --exclude-scenes none"
```

**Unlike the rollout commands, there is no `--cfg-file` here.** This script
rebuilds the simulator from `<run>/config.yaml` through the same
`build_phase4_context` the training loop uses, so the camera rig, the thresholds,
`EVAL.max_steps` and the success criterion all come from the run itself and
cannot drift from what it was scored with in-loop. Only the *split* is
overridden. That also means `--run-dir` takes the **run root** here, not
`<run>/best` — the script resolves the checkpoint itself (`--from last` for the
final policy instead of the best one).

```bash
$EV --run-dir $R/dagger4_run1  --out-prefix $R/dagger4_run1/scene_eval_test
$EV --run-dir $R/dagger4_run2  --out-prefix $R/dagger4_run2/scene_eval_test
$EV --run-dir $R/dagger4_run3  --out-prefix $R/dagger4_run3/scene_eval_test
$EV --run-dir $R/dagger4_run4  --out-prefix $R/dagger4_run4/scene_eval_test
$EV --run-dir $R/dagger4_run5  --out-prefix $R/dagger4_run5/scene_eval_test
$EV --run-dir $R/dagger4_run6  --out-prefix $R/dagger4_run6/scene_eval_test
$EV --run-dir $R/dagger4_run7  --out-prefix $R/dagger4_run7/scene_eval_test
$EV --run-dir $R/dagger4_run8  --out-prefix $R/dagger4_run8/scene_eval_test
$EV --run-dir $R/dagger4_run9  --out-prefix $R/dagger4_run9/scene_eval_test
$EV --run-dir $R/dagger4_run10 --out-prefix $R/dagger4_run10/scene_eval_test
$EV --run-dir $R/dagger4_run11 --out-prefix $R/dagger4_run11/scene_eval_test
$EV --run-dir $R/dagger4_run12 --out-prefix $R/dagger4_run12/scene_eval_test
$EV --run-dir $R/dagger4_run13 --out-prefix $R/dagger4_run13/scene_eval_test
$EV --run-dir $R/dagger4_run14 --out-prefix $R/dagger4_run14/scene_eval_test
$EV --run-dir $R/dagger4_run15 --out-prefix $R/dagger4_run15/scene_eval_test
$EV --run-dir $R/dagger4_run16 --out-prefix $R/dagger4_run16/scene_eval_test
$EV --run-dir $R/dagger4_run17 --out-prefix $R/dagger4_run17/scene_eval_test
$EV --run-dir $R/dagger4_run18 --out-prefix $R/dagger4_run18/scene_eval_test
$EV --run-dir $R/dagger4_run19 --out-prefix $R/dagger4_run19/scene_eval_test
$EV --run-dir $R/dagger4_run20 --out-prefix $R/dagger4_run20/scene_eval_test

# --- run 21 is DIFFERENT: pre-grasp endgame, see below --------------------
python examples/eval_run_scenes.py --split test \
    --grasp-pin-table output/grasp_pin_table_test_omg.json --exclude-scenes none \
    --min-frac 0.5 \
    --run-dir $R/dagger4_run21 --out-prefix $R/dagger4_run21/scene_eval_test
```

Roughly 10–13 min per run over the 130 usable test scenes. Outputs are
`<prefix>.csv` (one row per scene), `.json` (the aggregate plus the gate settings
that produced it) and `.png`.

**Run 21 drops `--box-probe` and pins `--min-frac 0.5`, for two different
reasons.** With `DAGGER.target: pregrasp` the episode ends 6.4 cm short of the
object, so the jaws are never around it while the policy is still deciding and
`opportunity` is near zero *by construction* — the probe refines steps that clear
the geometric gate and almost none do, and the counterfactual it asks ("would
closing here secure it?") is not even in that policy's action space, whose
commit is a blind push. Separately, `eval_run_scenes.py` always overrides
`EVAL.box.min_frac` from the CLI, whose default is now the permissive one-ray
gate; `box_after_rate` — the one usable opportunity-style number in pre-grasp
mode — is computed with those same box params, so `--min-frac 0.5` is what keeps
the test figure comparable to the 0.50 → 0.65 trajectory in run 21's own
`dagger_log.csv`. The report grows a pre-grasp block for this run:

```
  reach after push    pos 0.0564 m   rot 0.9012 rad   (where the blind 0.064 m push LANDED, vs the grasp)
  object in jaws after   52.2%   of the commits, how many put the object between the open pads
```

and the `!!` invariant warning will fire legitimately, because successes exceed
the near-zero opportunity count.

**Runs 1–2: the pose columns are against the wrong pin.** Those two runs were
collected with the `furthest_from_hand` grasp rule
(`grasp_pin_table_train.json`), and no non-`_omg` table exists for the test
split. So `near_rate`, `chance_rate`, `pos_err` and `min_pos` will be measured
against a grasp they were never taught to aim at. `success_rate`, `grasp_rate`,
`close_rate` and the whole opportunity block are pin-free and unaffected — and
both runs failed outright anyway (0.13 and 0.21), so this is a footnote rather
than a blocker.

**Add `--min-frac 0.5` to the grasp-mode runs too if you want
`dagger_log.csv`-comparable box columns.** The default one-ray gate plus
`--box-probe` is the *better* opportunity measure — it is the only setting under
which `success ⊆ opportunity` actually holds — but it is not the definition runs
1–20 logged. Use the default for a correct opportunity figure, `--min-frac 0.5
--min-rays 1` without `--box-probe` to reproduce the training curve's meaning.
Both are recorded in the output JSON, so the two cannot be confused later.

The same weights caveat applies: only runs **1, 2, 7, 12, 13, 15, 16, 21** have a
local snapshot, and the rest need the rsync above first. Full documentation of
the report and its metrics is in `docs/thesis_phase4_dagger.md` §6.

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

## Regrasp — approach-direction conditioning (`output/dagger_runs/`)

Runbook and design notes: [`README_REGRASP.md`](../README_REGRASP.md). Code is a
full fork (`handover_sim2real/regrasp/`, `regrasp_bc/`, `examples/*regrasp*.py`),
so nothing above changes.

The policy is told **which side to come in from** — a unit vector `d` injected
per-point into the cloud as `d·n_i` and `d·normalize(p_i−c)` — not which pose to
reach. On failure it is commanded a different side. `pc_channels` 5 → 7,
`GraspEncoder` deleted, no global conditioning branch.

Columns are Phase 4's, so a Regrasp row can be read straight across against a
Phase-4 one, plus three this phase needs: `config`, `dirs/scene` (how many
directions a scene demonstrates) and `command` (what vector the policy is
actually told). β params are `beta_start`→`beta_end` for the given
`beta_schedule`.

| run | config | camera | iters | m | dirs/scene | command | init | β | DART free | DART reach | DART variant | aux task | loss | model flags | best iter | final | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **regrasp_run1** | `regrasp_run1.yaml` | wrist+right | **19 of 25** | 354 → 176 | 1–2 (max-separated pair) | grasp axis `−R[:,2]` | warm-start (**best**.pt), PointNet++ from scratch | constant 0.75 | 0.3 | 0.3 | replace | no | pm (w=7) | nojoint, reach-tail ×2.5 (window **5**), **direction_cond**, head [256,256] | 8 (`success 0.354`) | `success 0.139` | Stopped at 19 by choice. `m` was 354 for iterations 1–5 and 176 for 6–19 — a budget correction, so **`D_steps` changes slope at iteration 6**. Base set 1087 episodes / 20386 steps over 617 scenes (471 paired, 146 single). Eval/collection read `best.pt`, the warm start read `best.pt` too. |
| **regrasp_run2** | `regrasp_run2.yaml` | wrist+right | 20 (planned) | 400 → ~259 | **1–4 (one per bin)** | **bin axis** | warm-start (**last**.pt), PointNet++ from scratch | **linear 1.0→0.75** | 0.3 | 0.3 | replace | **yes** (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | not yet run | not yet run | = run 1 with the protocol fixed and two method changes. See below. |

### Result: DAgger did not move success; the conditioning is read but not obeyed

| metric | first 5 iters | last 5 iters | Δ | chance |
|---|---|---|---|---|
| `success_rate` | 0.210 | 0.203 | **−0.008** | — |
| `dir_track` | 0.424 | 0.453 | +0.029 | — |
| `bin_hit_rate` | 0.261 | 0.306 | +0.046 | **0.25** |
| `bin_diag_rate` | 0.476 | 0.509 | +0.033 | **0.25** |
| `cond_sep` | 0.677 | 0.632 | −0.045 | — |

**Success is flat.** Every movement (range 0.114–0.354) sits inside the ±0.115
noise floor, measured on 100 scenes where this eval has 50 scenes / 79 episodes —
so the true floor here is wider still. There is no trend to read.

**The two direction metrics disagree, and that is the finding.** `bin_diag_rate`
≈ 0.50 against a chance level of 0.25: the approach *axis* matches the commanded
bin about twice as often as random. `bin_hit_rate` ≈ 0.28, i.e. **at chance**: the
side the gripper actually *arrives from* does not follow the command. The policy
orients roughly as told and approaches from wherever it likes. A single
"does it follow" number would have averaged these into something misleading;
`dir_err` (orientation) and `sector_err` (which side) are separate for this reason.

**Retry works, modestly and consistently.** `retry@2 − retry@1` is **+0.054 mean,
positive in 17 of 20 iterations**. At iteration 0 the two were identical
(0.400/0.400) — what an ignored command looks like — and the gap opening is the
clearest evidence that the second attempt does something genuinely different from
the first.

**Do not compare `success 0.20` against run 16's 0.80.** Run 16 was "grasp it from
wherever"; this is "grasp it from *there*", which is strictly harder, on a
1087-episode base set. `near_rate` reads ~0 throughout and is **dead** under this
methodology — it measures distance to a pose the policy is never given.

### Measured before the run: k = 6 is really k = 4

Over 623 planning scenes and 28339 goal-set grasps on s0/train, confirmed against
the full goal set and replicated on val:

| bin | scenes reaching it |
|---|---|
| `+x` free end | 490 (78.7 %) |
| `+z` top-down | 415 (66.6 %) |
| `+y` lateral | 366 (58.7 %) |
| `−y` lateral | 325 (52.2 %) |
| `−x` over the fingers | **12 (1.9 %)** |
| `−z` from beneath | **0 (0.0 %)** |

`−z` is geometrically impossible (object held above a table); `−x` at 0.19 % of
all grasps is the hand-collision filtering working as designed. Both bins are kept
and masked at runtime so the code stays rig-agnostic, but the retry ladder has
**four** live rungs, `chained_retry_at_k` saturates at k=4, and `succ_bin_−x` /
`succ_bin_−z` are NaN rather than zero. `k` is a test-time knob up to 20 (39.4° at
k=21, below the ~40° where bins stop being independent hypotheses).

### Caveats on every number above

- **`EVAL.holdout: False`** — the 50 eval scenes are also collected on, so success
  rates are optimistic.
- **Stopped at 19 of 25**, and `m` changed at iteration 6.
- `cond_delta` rose **0.73 → 1.38** on held-out val during the base fit, so the
  channels demonstrably reach the network and generalise. `bin_hit` at chance says
  they do not reach the *action* — which is precisely the pre-registered case for
  adding a small global `d` branch as run 2, since the SA layers max-pool and two
  scalars among ten input features are what pooling discards.

### regrasp_run2 — configured, not yet run

`examples/configs/regrasp_run2.yaml`, submitted with
`examples/slurm/train_regrasp.sbatch` (`--time=24:00:00`, ~20–21 h budgeted).
Two changes to the METHOD, four to the PROTOCOL, and nothing else.

**Method.** *One demonstration per bin*, not a maximally-separated pair. Four
contrasting commands on one observation break the scene→action confound harder
than two, and — the reason that matters for the figures — they populate every
bin's evaluation sample rather than only the two a scene happened to draw. The
train direction table gives 617 scenes and **1596 (scene, bin) pairs**, mean 2.59
per scene, split +x 490 / +y 366 / −y 325 / +z 415, against run 1's 1087 demos.

*The command is the bin axis*, not the demonstrated grasp's own approach
direction. This is a correctness fix: `retry.next_direction` has no grasp at
deployment and commands `to_world(BINS[b], anchor_R)`, while run 1 trained and
scored on `−R_grasp[:,2]`. The pinned grasp sits a **measured median 18.4° from
its bin axis** (p90 38.5, max 45 = the bin half-width), so run 1 carried a
train/deploy skew of the same order as the effect it was measuring — and
`dir_err` could not see it, because it scored against the same shifted target.
The demonstration is now the goal-set grasp *closest to that axis*, and the
residual is recorded per episode as `demo_off_deg`.

The consequence is that a failed pin is no longer recoverable. Run 1 re-derived
the command from whatever the expert flew, so a miss simply relabelled the
episode; now the command does not move, so a miss leaves an episode captioned
`+z` whose trajectory approaches from elsewhere, and it cannot be re-binned
because the scene already has a demonstration for the bin it flew to.
`SIM.demo_ok_table` — written by `audit_regrasp_demos.py --write-ok` — drops
those (scene, bin) pairs from **training and evaluation**, by pruning the pin
table so every consumer sees the same set.

**Protocol.**

| | run 1 | run 2 | why |
|---|---|---|---|
| `m` | 354 → 176 mid-run | uniform (~259 from 100 scenes) | `num_grasps` is a MIN and read 1 on a mixed table; fixed with `max_grasps`, so \|D\| no longer changes slope at iteration 6 |
| eval logging | pooled + `succ_bin_*` | **the full rate family per bin** (~180 columns) | pooled `success_rate` averages four physically different commands |
| iterations | stopped at 19/25 | 20, planned | flat from iteration 6 on, and the marginal iteration is ~40 min of refit against a change inside the noise floor |
| checkpoint | eval/collect `best`, warm start `last` | **`last` everywhere** | "the policy at iteration i" named two sets of weights |

`EVAL.holdout` stays **false** in both, deliberately: the in-loop curves are a
clean read of "is the policy getting better at what it is being taught", and the
held-out number now comes from `examples/eval_regrasp_testset.py` on the s0
**test** split instead. Also changed: `EVAL.every` 0 → 1 (scoring back inside the
loop, Phase-4 style), `num_scenes` 50 → 100, `base_epochs` 40 → 50, `iter_epochs`
12 → 15, β constant 0.75 → **linear 1.0→0.75**, and the **auxiliary head on**
(`bc_regrasp_run2.yaml`, `aux_weight: 1.0`) — legitimate again now that the
policy is told a direction rather than the pose the head predicts.

**`last` is not a bookkeeping change.** Across all 26 refits of Phase-4 runs 16
and 20 the best epoch was **never** the last epoch: the val minimum lands at
epoch 1–9 of 25 and val loss then climbs (run 20 iteration 19: 0.3643 at epoch 3,
0.4272 at epoch 24). `last.pt` is a visibly more overfit checkpoint, so run 2's
absolute rates may sit below run 1's for that reason alone.

**β = 1.0 makes iteration 1 a pure-expert round.** The learner is queried every
step but never moves the arm, so D₁ adds 259 clean demonstrations on fresh scenes
rather than on-policy states; the distribution shift starts at iteration 2, and
iteration 1's collection curves are not comparable with later ones.

**Figures.** `curves.png` → `training_curve.png` (4×3, one row per direction),
plus `debug_dagger.png`, `media_curves.png`, a 2×3 `curves_regrasp.png` carrying
*ended in the commanded bin* per bin, and — after the run —
`test_set_evaluation.png` from the test-split script, which also reports the
**chained** retry ladder beside the independent `retry@k`. See
[`README_REGRASP.md`](../README_REGRASP.md) steps 8–9.

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
| `dagger_phase4_all_beta075_pregrasp.yaml` | dagger4_run21 |
| `bc_phase1.yaml` / `bc_phase1_nojoint.yaml` | Phase-4 `TRAIN.train_cfg` |
| `bc_phase4_pm.yaml` | run 11 `TRAIN.train_cfg` — `LOSS.pose_loss: pm`, `pm_weight: 7.0` |
| `bc_phase4_aux.yaml` | run 13 `TRAIN.train_cfg` — `MODEL.aux_head`, `LOSS.aux_weight: 1.0` |
| `bc_phase4_reachw.yaml` | run 14 `TRAIN.train_cfg` — `DATA.reach_tail_weight: 2.5` |
| `bc_phase4_all.yaml` | runs 15, 16 `TRAIN.train_cfg` — pm + aux + reach-tail weighting |
| `bc_phase4_all_prevact.yaml` | run 17 `TRAIN.train_cfg` — same, `MODEL.use_prev_act: true` |
| `dagger_phase5_run1.yaml` | p5_run1 (`examples/train_regrasp.py`) |
| `dagger_phase5_smoke.yaml` | Phase-5 shakedown — 2 iters, m=8, 3 eval scenes |
| `bc_phase5_cond.yaml` | p5_run1 `TRAIN.train_cfg` — `MODEL.grasp_cond`, aux head off, head `[512,256]` |

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
