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
Phase-4 one, plus five this phase needs: `config`, `dirs/scene` (how many
directions a scene demonstrates), `command` (what vector the policy is actually
told), `d noise` (`DATA.d_noise_deg` in the learner config — how far the
commanded direction is perturbed during training) and `epochs` (`TRAIN.base_epochs` / `TRAIN.iter_epochs` — the fit at
iteration 0 and the refit every round after). β params are
`beta_start`→`beta_end` for the given `beta_schedule`.

**`epochs` only means what it says next to `init`.** Under `train_from_scratch:
true` the per-iteration budget is a full refit from the pretrained encoder, and
under a warm start it is a continuation of the previous iteration's weights — so
"15 epochs" is a much larger fraction of the training a warm-started run gets
than of a from-scratch one, and the two numbers are not comparable across rows
with different `init`. Phase-4 run 16, for reference, was 100 / 25 from scratch.

| run | config | camera | iters | m | dirs/scene | command | d noise | init | epochs (base/iter) | β | DART free | DART reach | DART variant | aux task | loss | model flags | best iter | final | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **regrasp_run1** | `regrasp_run1.yaml` | wrist+right | **19 of 25** | 354 → 176 | 1–2 (max-separated pair) | grasp axis `−R[:,2]` | 12° | warm-start (**best**.pt), PointNet++ from scratch | 40 / 12 | constant 0.75 | 0.3 | 0.3 | replace | no | pm (w=7) | nojoint, reach-tail ×2.5 (window **5**), **direction_cond**, head [256,256] | 8 (`success 0.354`) | `success 0.139` | Stopped at 19 by choice. `m` was 354 for iterations 1–5 and 176 for 6–19 — a budget correction, so **`D_steps` changes slope at iteration 6**. Base set 1087 episodes / 20386 steps over 617 scenes (471 paired, 146 single). Eval/collection read `best.pt`, the warm start read `best.pt` too. |
| **regrasp_run2** | `regrasp_run2.yaml` | wrist+right | **19 of 25** (running) | 400 → ~259 | **1–4 (one per bin)** | **bin axis** | 12° | **scratch (FTL)**, PointNet++ random every iter | 50 / 15 | **linear 0.9→0.75** | 0.3 | 0.3 | replace | **yes** (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | **13** (`success 0.563`) | it19 `success 0.487`, `dir_track 0.722`, `bin_diag 0.924` | = run 1 with the protocol fixed and two method changes. **The best run in the phase**, and the first at cluster scale: `success 0.143 → 0.563`, `bin_diag 0.517 → 0.971` (it 16) against a chance level of 0.25. First cluster run to survive the manager/worker pin-table fix (`c47e551`), and the run that confirmed it. See below. |
| **regrasp_fast1** | `regrasp_run2_fast.yaml` | wrist+right | **8 of 8** | 126 → 138 | **1–4 (one per bin)** | **bin axis** | 12° | warm-start (**last**.pt), PointNet++ from scratch | **20 / 6** | **linear 1.0→0.75** | 0.3 | 0.3 | replace | **yes** (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | 8 (`success 0.457`) | `success 0.457`, `dir_track 0.611`, `bin_diag 0.771` | Run 2's method at a fifth of the compute, on this laptop rather than DelftBlue, while the cluster was in maintenance. Five numbers turned down and nothing else: `num_iters` 20→8, `episodes_per_iter` 400→200, `base_epochs` 50→20, `iter_epochs` 15→6, `EVAL.num_scenes` 100→40. Base set 1596 episodes / 30028 steps over 617 scenes; ended at \|D\| 2624 episodes / 64257 steps, 53% of it on-policy. **First run in which DAgger moved anything.** See below. |
| **regrasp3_fast1** | `regrasp_run3_fast.yaml` | wrist+right | **6 of 6** | 368 → 356 | **3 per bin** (7.42/scene) | **bin axis** | 12° | warm-start (**last**.pt), PointNet++ from scratch | **15 / 15** | **linear 1.0→0.75** | 0.3 | 0.3 | replace | **yes** (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | **4** (`success 0.565`) | `success 0.500`, `dir_track 0.687`, `bin_diag 0.855` | Three demonstrations per bin instead of one, on the same 617 scenes. Base set **4578 episodes / 86315 steps**, 2.87× regrasp_fast1's, collected in 4 concurrent `--shard i/4` processes; ended at \|D\| 6730 episodes / 157463 steps. Laptop run, `iter_epochs` held at run 3's 15 so `num_iters` came down to 6. **Past regrasp_fast1's FINAL numbers from iteration 1 on.** See below. |
| **regrasp_run3** | `regrasp_run3.yaml` | wrist+right | 25 (planned) | 1200 → ~742 | **1–4 bins × 3 grasps** (~7.4 slots) | **bin axis** | 12° | **scratch (FTL)**, PointNet++ random every iter | 50 / 15 | **linear 0.9→0.75** | 0.3 | 0.3 | replace | yes (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | not yet run | not yet run | = run 2 with **three demonstrations per bin** instead of one, and nothing else: same frame, same k=6, same command, same network. Base set 1596 → **4578** demos over the same 617 scenes (91% of (scene, bin) pairs have ≥3 goal-set members). `episodes_per_iter` goes 400 → 1200 to hold the scene count at 100 — `max_grasps` triples, and leaving it at 400 would have drawn 33 scenes instead. See below. |
| **regrasp_run4** | `regrasp_run4.yaml` | wrist+right | 25 (planned) | 1200 → ~742 | **1–4 bins × 3 grasps** (~7.4 slots) | **bin axis** | **0° (OFF)** | **scratch (FTL)**, PointNet++ random every iter | 50 / 15 | **linear 0.9→0.5** | 0.3 | 0.3 | replace | yes (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | not yet run | not yet run | = run 3 with the direction-vector noise removed (`bc_regrasp_run4.yaml`, `d_noise_deg` 12 → 0) **and** `beta_end` 0.75 → 0.5. Shares run 3's pin tables and shards — nothing to re-collect. **Two changes, so not a clean ablation**; see below. |
| **regrasp_run5** | `regrasp_run5.yaml` | wrist+right | 25 (planned) | 1200 → ~742 | **1–4 bins × 3 grasps** (~7.4 slots) | **bin axis** | 12° | **warm-start (best.pt)**, PointNet++ from scratch at iter 0 only | 50 / 15 | **linear 0.9→0.75** | 0.3 | 0.3 | replace | yes (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | not yet run | not yet run | = run 3 with `train_from_scratch` **false**. ONE change — a clean ablation, unlike run 4. Shares run 3's pin tables and shards. Under a warm start `iter_epochs: 15` is 15 MORE epochs on a trained network rather than a complete refit, so no early dip. See below. |
| **regrasp_run6** | `regrasp_run6.yaml` | wrist+right | 25 (planned) | **7404 → ~4578 (ALL 617 scenes)** | **1–4 bins × 3 grasps** (~7.4 slots) | **bin axis** | **0° (OFF)** | **scratch (FTL)**, PointNet++ random every iter | 50 / 15 | **linear 0.9→0.5** | 0.3 | 0.3 | replace | yes (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | not yet run | not yet run | = run 4 with `episodes_per_iter` 1200 → **7404** = every scene, every iteration. Maximum on-policy coverage: \|D\| passes half DAgger data after ONE iteration instead of thirteen. **~223 h ≈ 9.3 days of GPU and ~47 GB of scratch** — the refit is quadratic in the iteration count, so `num_iters: 8` costs ~30 h and gets most of it. Shares run 3/4's tables and shards. See below. |
| **regrasp_run7** | `regrasp_run7.yaml` | wrist+right | **17 of 25** (running) | 400 → ~259 | **1–4 (one per bin)** | **bin axis** | **0° (OFF)** | **scratch (FTL)**, PointNet++ random every iter | 50 / 15 | **linear 0.9→0.5** | 0.3 | 0.3 | replace | yes (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | **12** (`success 0.555`) | it17 `success 0.496`, `dir_track 0.704`, `bin_diag 0.903` | = run 2 with **the same two changes run 4 made to run 3** (`bc_regrasp_run4.yaml`, `d_noise_deg` 12 → 0, **and** `beta_end` 0.75 → 0.5). Shares run 2's pin tables and shards. **The noise transformed the BASE FIT and nothing after it**: iteration 0 went `success 0.143 → 0.370` against run 2, and by iteration 7 the two runs are indistinguishable. Shows the FTL dip textbook-clean (0.370 → 0.244 at it 1, back above base by it 5). See below. |
| **regrasp_run8** | `regrasp_run8.yaml` | wrist+right | **19 of 25** (running) | 400 → ~259 | **1–4 (one per bin)** | **bin axis** | **0° (OFF)** | **warm-start (best.pt)**, PointNet++ from scratch at iter 0 only | 50 / 15 | **linear 0.9→0.5** | 0.3 | 0.3 | replace | yes (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | **14** (`success 0.487`) | it19 `success 0.475`, `dir_track 0.704`, `bin_diag 0.908` | = run 7 with `train_from_scratch` **false**. ONE change — the clean warm-start ablation at one demo per bin. **The warm start removes the FTL dip and buys nothing**: no early trough, `train_loss` half run 7's (0.14 vs 0.235), and success/`dir_track`/`bin_diag` all at or below run 7's from iteration 7 on. Its iteration 0 is configured identically to run 7's and scored 0.282 against 0.370 — **that 0.088 gap is the run-to-run noise floor** for this setup and the yardstick for everything else here. See below. |
| **regrasp_run9** | `regrasp_run9.yaml` | wrist+right | 25 (planned) | 400 → ~259 | **1–4 (one per bin)** | **train: grasp axis `−R[:,2]` / deploy: BIN CENTROID** | **0° (OFF)** | **scratch (FTL)**, PointNet++ random every iter | 50 / 15 | **linear 0.9→0.75** | 0.3 | 0.3 | replace | yes (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | not yet run | not yet run | **The first run whose training label and deployment command are different vectors.** = run 2 + `SIM.command_deploy: bin_centroid` + `bc_regrasp_run9.yaml` (`d_noise_deg` 12 → 0, `DATA.d_source` `d_world` → `d_grasp_world`). Trains on the grasp's own approach axis — no quantisation, no perturbation — and deploys on the unit mean of each bin's assigned `d_anchor`, which needs no grasp and so is producible on the robot. Reuses run 2's tables and shards **verbatim**: both vectors have always been written per episode, so this is a relabelling. Read it as **run 1 done properly**, not as a run-2 variant. See below. |
| **regrasp_run10** | `regrasp_run10.yaml` | wrist+right | 25 (planned) | 600 → TBD | **1–6 (one per bin, ALL SIX LIVE)** | bin axis, but **`d` = centroid → fingertip** | **0° (OFF)** | **scratch (FTL)**, PointNet++ random every iter | 50 / 15 | **linear 0.9→0.75** | 0.3 | 0.3 | replace | yes (w=1.0) | pm (w=7) | nojoint, reach-tail ×2.5 (window 5), **direction_cond**, head [256,256] | not yet run | not yet run | **The first run to change what `d` MEANS.** = run 2 + `SIM.d_rule: grasp_offset` + `bc_regrasp_run4.yaml` (noise off). `d` is no longer `−R_grasp[:,2]` but the direction from the object centroid to the midpoint between the fingertips — a function of the grasp's **position**, not its orientation. **This unlocks the two dead bins**: `−z` goes from 0 scenes to 235 and `−x` from 12 to 191, because you cannot *approach* from beneath a held object but you can close your fingers on its underside. Retry ladder gains two rungs; chance level moves 1/4 → 1/6. **Needs the whole upstream chain rebuilt** — table, assignment, base collection, audit. **Also the first run under `SIM.reach_filter`**: (scene, bin) pairs whose demonstration never reached its grasp are dropped from D, from collection and from eval (on the run-2 shard that was 30% of pairs). Deliberately NOT a single-change test against run 2. See below. |

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

**Replicated on the held-out s0 TEST split** (built 2026-08-25, 130 of 144 scenes
planning): `+x` 94, `−x` **6**, `+y` 70, `−y` 76, `+z` 100, `−z` **0**. Same
shape — four live bins, `−x` at 4.6 % of scenes, `−z` at zero — so the finding is
a property of the task and the rig, not of the split it was measured on.

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

### regrasp_run2 — the best run in the phase (19 of 25)

**RESULT.** `success 0.143 → 0.563` (iteration 13), `dir_track 0.405 → 0.737`,
`bin_diag 0.517 → 0.971` (iteration 16) against a chance level of 0.25. Latest
point, iteration 19: `success 0.487`, `dir_track 0.722`, `bin_diag 0.924`. |D|
has grown to 6452 episodes / 194264 steps.

**This is the first Regrasp run at cluster scale**, on 100 eval scenes (~259
episodes, ~65 per live bin) rather than the fast runs' 20–40, so the per-bin
numbers are readable for the first time — a single point's binomial standard
error is ~0.05 here against ~0.08 on `regrasp3_fast1`.

Two things that look alarming and are not. `train_loss` rose 0.111 → 0.235 and
flattened: that is from-scratch FTL refitting a growing, increasingly on-policy
aggregate, exactly what the schedule predicts. `val_loss` rose monotonically
0.261 → 0.352 while success more than tripled: the val set is expert-only and
frozen while D turns on-policy, and `corr(val_loss, success)` was measured at
+0.71 / +0.83 on the two fast runs. Neither is a warning.

It is also the run that **confirmed the manager/worker pin-table fix**
(`c47e551`): its eval set contains scenes 127, 253 and 616 — the exact three
computed to kill a parallel worker under the old code — and it has scored them
19 times without incident.

The standing caveat holds: `EVAL.holdout: false`, so these are train-split rates.
The held-out number comes from `eval_regrasp_testset.py` on the s0 test split.

---

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

### regrasp_run7 — the direction noise transformed the base fit and nothing after it (17 of 25)

**RESULT.** The comparison against run 2 is the one this run was built for, and
it splits cleanly in two.

| iter | run 2 `succ` | **run 7 `succ`** | run 2 `bin_diag` | **run 7 `bin_diag`** |
|---|---|---|---|---|
| 0 (base fit) | 0.143 | **0.370** | 0.517 | 0.521 |
| 1 | 0.198 | 0.244 | 0.567 | 0.588 |
| 3 | 0.269 | 0.357 | 0.592 | 0.803 |
| 5 | 0.265 | 0.370 | 0.794 | 0.866 |
| 7 | 0.416 | 0.508 | 0.882 | 0.933 |
| 12 | 0.500 | 0.555 | 0.945 | 0.920 |
| best | **0.563** (it 13) | **0.555** (it 12) | 0.971 (it 16) | 0.933 |
| latest | 0.487 (it 19) | 0.496 (it 17) | 0.924 | 0.903 |

**Iteration 0 is a clean single-variable comparison, and it is the finding.** β
does not enter until iteration 1, so the base fit isolates `d_noise_deg`
perfectly — and turning the noise off took `success` from **0.143 to 0.370**,
two and a half times, on the identical shard with the identical 50 epochs.

**But DAgger erases it.** By iteration 7 the two runs are indistinguishable and
they stay that way: the peaks are 0.563 and 0.555, the latest points 0.487 and
0.496. Whatever the perturbation was costing, on-policy data repairs it within
about six rounds. The honest reading is that `d_noise_deg: 12` hurts a *pure BC*
fit substantially and is irrelevant to a converged DAgger run — which also means
it is not the lever worth pulling on this task.

**How much of that 0.227 is real?** Run 8's iteration 0 is configured
*identically* to run 7's — same `bc_regrasp_run4.yaml`, same shard, same 50
epochs, same seed — and scored **0.282** against run 7's 0.370. So the run-to-run
spread on a base fit is about **0.088**, and the two noise-off samples average
0.326 against noise-on's single 0.143. That is roughly a 2σ separation from n=1
on one side, so treat it as strong evidence rather than a measurement.

**One thing that did not move.** `bin_diag` at iteration 0 is 0.517 vs 0.521 and
`dir_track` actually favours run 2 (0.405 vs 0.333). The noise was costing
*grasp success*, not *command compliance* — which is the opposite of the
mechanism the augmentation was introduced to protect.

**Caveat.** `beta_end` moves too (0.75 → 0.5), so everything from iteration 1
onward has two candidate causes. Since the two runs converge, the β difference is
evidently doing nothing visible either — but the clean statement is confined to
iteration 0.

---

`examples/configs/regrasp_run7.yaml`. The same two changes run 4 made to run 3,
applied to run 2 instead — verified as exactly two value diffs against
`regrasp_run2.yaml`:

| | run 2 | run 7 |
|---|---|---|
| `TRAIN.train_cfg` | `bc_regrasp_run2.yaml` (`d_noise_deg` 12°) | **`bc_regrasp_run4.yaml`** (`d_noise_deg` **0°**) |
| `DAGGER.beta_end` | 0.75 | **0.5** |

**Shares run 2's data entirely** — `regrasp_pins_train.json`,
`train_regrasp.h5` / `val_regrasp.h5`, `regrasp_demos_train_ok.json`, the
one-member per-bin tables and *not* the `_p3` ones. Nothing to re-collect; the
only cost is training time.

**Why it is worth a run rather than a footnote: it closes a 2×2 that run 4 left
half-open.**

| | noise 12°, `beta_end` 0.75 | noise 0°, `beta_end` 0.5 |
|---|---|---|
| **1 demo/bin** | run 2 | **run 7** |
| **3 demos/bin** | run 3 | run 4 |

The two treatments are expected to interact with the demo count in *opposite*
directions, and neither pair of runs alone can say so.

**The noise.** At three demos per bin the network already sees three different
grasps under one command, so `d_noise_deg: 12` is a third source of spread on a
command that is deliberately many-to-one — removing it there mostly sharpens an
already-blurred target. At **one** demo per bin the command maps to a single
trajectory, and the perturbation is the only thing standing between the policy
and memorising it. Run 7 is therefore the arm of the ablation where turning the
noise off could plausibly **hurt**, which is what makes it informative rather
than confirmatory.

**The β floor.** 0.5 puts half of every late trajectory under the learner's own
control. The on-policy *fraction* is identical in runs 4 and 7, but run 7
collects ~259 episodes an iteration against run 4's ~742, so a third as many
expert labels are available to fit those on-policy states. If 0.5 is too
aggressive anywhere, it is here — watch `c_success_rate` against run 2's from
iteration ~15 on.

**Not a clean ablation, deliberately.** Two things move against run 2, exactly as
two move in run 4 against run 3. That is the point: run 7 − run 2 and run 4 −
run 3 are the *same* comparison at two demo counts, and they can be read against
each other. Separating noise from β still needs the fifth cell — (noise off,
`beta_end` 0.75) — at whichever demo count the pair turns out to disagree on.

**What would falsify it.** Run 7 losing to run 2 on `dir_err` / `bin_diag_rate`
while run 4 beats run 3 on the same metrics means the augmentation substitutes
for demonstration diversity and should be kept wherever that diversity is absent.
Both pairs moving the same way means the noise is simply good or simply bad and
the demo count is irrelevant to it. Run 7 collapsing late while run 4 does not
points at `beta_end: 0.5` against the smaller aggregate, not at the noise.

---

### regrasp_run8 — the warm start removes the FTL dip and buys nothing (19 of 25)

**RESULT.** A clean single-variable comparison against run 7, and it answers the
question the pair was built to ask.

| iter | run 7 (scratch) | **run 8 (warm)** | run 7 `train_loss` | **run 8 `train_loss`** |
|---|---|---|---|---|
| 0 (base fit) | 0.370 | 0.282 | 0.110 | 0.110 |
| 1 | **0.244** ← dip | 0.290 | 0.220 | 0.149 |
| 2 | 0.256 | 0.328 | 0.229 | 0.147 |
| 3 | 0.357 | **0.450** | 0.232 | 0.151 |
| 5 | 0.370 | 0.340 | 0.235 | 0.141 |
| 7 | **0.508** | 0.324 | 0.237 | 0.142 |
| 12 | 0.555 | 0.441 | 0.235 | 0.146 |
| best | **0.555** (it 12) | **0.487** (it 14) | | |
| latest | 0.496 (it 17) | 0.475 (it 19) | 0.233 | 0.140 |

**The predicted FTL dip is real and it is exactly where the config said it would
be.** Run 7 falls from 0.370 to 0.244 at iteration 1 — 15 epochs from a random
PointNet++ over an aggregate barely larger than the base set — and does not clear
its base fit again until iteration 5. Run 8 never dips: 0.282 → 0.290 → 0.328 →
0.450, monotone.

**And then run 7 wins it all back.** By iteration 7 run 7 is at 0.508 against run
8's 0.324, and it stays ahead through iteration 17. On the conditioning metrics
the two are identical — `dir_track` 0.704 vs 0.704, `bin_diag` 0.903 vs 0.908 —
so the warm start is not trading command-following for success either.

**The train/val split says why.** Run 8's `train_loss` sits at ~0.14 against run
7's ~0.235, so the warm-started network fits the aggregate visibly better and
generalises no better at all. That is the signature of a model continuing its own
earlier solution rather than re-deriving one: it inherits what it already
believed, which is precisely the failure Follow-The-Leader's fresh refit exists
to prevent.

**Read the gaps against 0.088.** Run 8's iteration 0 uses the identical config to
run 7's and scored 0.282 against 0.370, so ~0.088 is the run-to-run noise floor
here. The final gap (0.496 vs 0.475) sits *below* it and should be read as no
difference; the iteration-7 gap (0.508 vs 0.324) is more than double it and is
real.

**What this licenses.** At one demonstration per bin, from-scratch FTL is the
right default: it costs about five iterations of early dip and then matches or
beats the warm start, with a cleaner theoretical story. Run 5 is the same
ablation at three demos per bin, where run 3's base set is 2.87× larger and the
dip should therefore be shallower — if run 5 also fails to beat run 3, the
warm-start question is settled for this phase.

---

`examples/configs/regrasp_run8.yaml`. **One change** against run 7:
`TRAIN.train_from_scratch` true → **false**. The one-demo-per-bin twin of run 5,
and like it a clean single-variable comparison. Shares run 2/7's tables and
shards, so the only cost is training time.

The flag is now tested at both demo counts, with matching conditions along each
row:

| | from-scratch FTL | warm start (`best.pt`) | |
|---|---|---|---|
| **3 demos/bin** | run 3 | run 5 | *(noise 12°, β→0.75)* |
| **1 demo/bin** | run 7 | run 8 | *(noise 0°, β→0.50)* |

Each **row** is internally clean — 5 against 3, and 8 against 7, each move
exactly one flag. The rows are *not* comparable to each other, because the noise
and β settings differ between them; that comparison is runs 2/3/4/7 above.

**The FTL dip it removes is deeper here than in run 5.** Run 7's base set is
30k steps against run 3's 86k, so a from-scratch refit at `iter_epochs: 15` has
under half the data per epoch that run 3's does. In gradient steps at batch 64:

| | run 7 (scratch) | run 8 (warm) |
|---|---|---|
| iteration 0 | 50 × 30k = **23k** | same |
| iteration 1 | 15 × 34k = **8k** — a third | 8k *on top of* iteration 0 |
| iteration 25 | 15 × 238k = **56k** | 56k on top of everything before |

If the warm start is going to win anywhere, the small aggregate is where.

`iter_epochs` is left at 15 rather than cut for the warm start: reducing it
because "a continued fit needs less" would make run 8 differ from run 7 in two
things and destroy the ablation, and the epochs cost the same wall clock either
way.

**`init_ckpt: best` becomes live here**, where it is inert in run 7. Every
iteration continues from the previous iteration's `best.pt` — the same file
`EVAL.ckpt: best` scores and the collection workers roll out — so there is one
set of weights per iteration, and the chain starts at this run's own iteration 0
(`base_run: null`). A bad `best` selection now *propagates* instead of being
discarded at the next refit, which is the cost of the warm start and worth
watching under `beta_end: 0.5`.

**What would falsify it.** Run 8 ahead early and converged with run 7 by
iteration ~15 means the warm start buys only the dip back, and FTL's fresh-fit
argument survives. Run 8 ahead to the end means FTL loses more to the 15-epoch
budget than it gains, and every future regrasp run should warm-start. Run 8 ahead
early and **behind** late is the inherited-mistake failure FTL exists to prevent,
arriving as the aggregate turns majority on-policy.

---

### regrasp_run10 — `d` from the grasp's position, not its orientation, configured, not yet run

`examples/configs/regrasp_run10.yaml`. **The first run in the phase to change
what `d` means.** Runs 1–9 argued about which *vector* to hand the policy; this
one changes the *question* the vector answers.

| | definition | what it says |
|---|---|---|
| `approach_axis` (runs 1–9) | `d = −R_grasp[:,2]` | which side the gripper **comes from** — a property of the grasp's **orientation** alone |
| `grasp_offset` (run 10) | `d = normalize(T_grasp · [0,0,0.1122] − c)` | which part of the object the fingers **close on** — a property of the grasp's **position** relative to the object |

0.1122 m is the fingertip end of the Panda pads (`grasp_box.py`, from
`meshes/collision/finger.obj`: the joints originate at 0.0584 and the pad runs
0.0362–0.0538 finger-local). GA-DDPG's control points put the fingertips at
0.105. The midpoint *between* the two fingertips lies on the gripper's local z
axis by symmetry, so only the depth has to be named.

**It is not a refinement, and the measurement says so.** Over the 3477 goal-set
grasps recorded in `direction_table_train.json`:

| definition | `+x` | `−x` | `+y` | `−y` | `+z` | `−z` | within 45° |
|---|---|---|---|---|---|---|---|
| `approach_axis` | 1162 | 20 | 822 | 684 | 789 | **0** | 100.0% |
| `grasp_offset` | 714 | **392** | 637 | 450 | 418 | **526** | 90.2% |

scenes reaching each bin: 490/12/365/324/415/**0** against
248/**191**/241/205/142/**235**.

**`−z` goes from dead to the third-largest bin and `−x` from 12 scenes to 191.**
That is physically right rather than a bug: you cannot *approach* an object from
beneath while it is held over a table — which is exactly why `−z` has been empty
since the feasibility census — but you can perfectly well close your fingers on
its underside. So the retry ladder gains two rungs, `chained_retry_at_k`
saturates at 6, `assign_direction_demos.py` runs **without** `--drop-bins`, and
`max_grasps` becomes 6 (hence `episodes_per_iter` 400 → 600).

**The cost is conditioning.** The fingertip point sits *on* the object, so
`tip − c` is short: a median 3.85 cm, and **14.4% of grasps land inside 2 cm**,
where the direction is centroid noise rather than geometry — and the centroid
comes from a partial, occluded cloud. `d_min_offset: 0.02` drops those at
table-build time and records the count as `_meta.n_short_offset`.
`approach_axis` has no such failure mode, which is the honest argument for the
older rule.

**The conservative fallback is one number.** `d_point_depth: 0.0` puts the point
at the **palm origin**: a median 12.48 cm from the centroid (minimum 4.55, never
degenerate) and a median 14.49° from `−R[:,2]`. Still position-derived and still
orientation-independent — but it answers the `approach_axis` question, so it
would *not* unlock `−z`.

**Everything upstream has to be rebuilt, and it cannot be done offline.**
`d_rule` is decided in `build_direction_table.py` — it is what fills `d_anchor`,
and every later stage reads that field. The existing table stores only the
members that survived the 45° cutoff under the *old* rule — 3477 of 32107,
**10.8%** — so relabelling it would be a badly biased subsample of the goal sets.
The chain is: build table → assign → collect → audit → run.
`build_regrasp_context` refuses a table whose `_meta.d_rule` disagrees with the
config, because that failure is silent: bins populated by one rule and
`bin_realized` measured by the other disagree on most episodes, the miscaption
filter empties the aggregate, and the symptom is "the collection produced almost
nothing".

**Run 10 is also the first run under the reach filter, and that is a second
change against run 2.** `SIM.reach_filter` (default on from this point) drops
every (scene, bin) pair whose base demonstration never reached the grasp it aimed
at — terminal pose outside `close_pos_thresh` / `close_rot_thresh`. Measured on
the run-2 shard it removes **480 of 1596 episodes (30.1%)**, taking D from 1596
ep / 30028 steps to 1098 / 23048, the pin table from 617 scenes / 1596 pairs to
558 / 1097, and pairable scenes from 490 to 340. The filter is applied in five
places from one predicate (`regrasp_bc/dataset.py:episode_status`) — the pin
table prune covering DAgger collection and in-loop eval, plus `BCDataset`,
`compute_normalization_stats` and the `D_episodes`/`D_steps` log columns.
Refitting the normalizer over the filtered set moves `action_std` down 2–8% per
axis and `state_std` on the EE pose up 3–7%, because the dropped episodes are
truncated mid-approach and carry the large "keep approaching" deltas.

**This is an accepted confound, not an oversight.** Run 10 was configured as
"run 2 with `d_rule` changed"; it is now "run 2 with `d_rule` changed **and** the
reach filter on", so a difference against run 2 is not attributable to `d_rule`
alone. The decision was to take the better data rather than preserve the
comparison. What that costs is stated here so nobody later reads run 10 as a
clean `grasp_offset` test — the clean one would be a re-run of run 2 under the
filter, which nobody has paid for.

**The gate that actually matters for run 10.** The filter's pass rate has been
measured on the *4-bin* run-2 shard only. Run 10's premise is that `grasp_offset`
unlocks `−z` (235 scenes) and `−x` (191), and those are precisely the directions
whose demonstrations are most likely to fail — closing on an object's underside
is harder for the planner than approaching its free end. If `−z` and `−x` have
poor reach rates, the filter deletes exactly the data the run exists to test, and
the result would read as "the new bins do not help" when the cause is that they
were never trained on. Measure it before submitting: `build_demo_table.py` then
`analyze_demo_bins.py` on the `_off` shard, and read the per-bin `reach` column.

**What would falsify it.** The premise is that "close on this part of the
object" is a *more learnable* instruction than "come from this side", because it
names a target the policy can see in its own cloud rather than a wrist
orientation it has to infer — `d·r = d · normalize(pᵢ − c)` becomes almost a
direct readout of the target. So `bin_diag_rate` should rise against run 7's
0.903 and `dir_err` should fall. `bin_diag_rate` at chance means the per-point
channels cannot express a position-derived command. High `bin_diag_rate` with
low `success_rate` in the `−z`/`−x` rows means the policy obediently drives at an
underside it cannot grasp from — the feasibility mask's problem, not the
conditioning's. **Note the chance level moves, 1/4 → 1/6**, so `bin_diag_rate` is
not comparable across the change without saying which chance it is against.

---

### regrasp_run9 — train on the true direction, deploy on the bin's centroid, configured, not yet run

`examples/configs/regrasp_run9.yaml`. **The first run in the phase whose training
label and deployment command are different vectors**, and the first that needs
the distinction to exist at all.

Every Regrasp episode has always carried two unit vectors: `d_world`, the
direction it was *commanded*, and `d_grasp_world` = `−R_grasp[:,2]`, the approach
axis the expert *actually flew*. Runs 1–8 made those the same vector, so nothing
ever had to choose between them. Run 9 chooses:

| | training label | deployment command |
|---|---|---|
| **run 1** | grasp axis | bin axis `BINS[b]` |
| **runs 2–8** | bin axis | bin axis |
| **run 9** | grasp axis | **bin centroid** |

The centroid is the unit mean of every `d_anchor` assigned to a bin, over the
pin table *after* `demo_ok_table` prunes it — so it summarises the
demonstrations the run is actually trained on. Six vectors, no grasp needed, so
it deploys on the robot exactly as `BINS` does. Measured over the 1576 surviving
(scene, bin) pairs:

| bin | centroid (anchor frame) | angle to `BINS[b]` | n |
|---|---|---|---|
| `+x` | `[ 0.9909, −0.0095,  0.1341]` | 7.73° | 486 |
| `+y` | `[ 0.2184,  0.9640,  0.1520]` | **15.43°** | 359 |
| `−y` | `[ 0.2058, −0.9699,  0.1304]` | **14.10°** | 319 |
| `+z` | `[ 0.1365,  0.0245,  0.9903]` | 7.97° | 412 |

Every bin leans toward `+x`/`+z` — the region the hand-collision filter leaves
feasible — so `BINS[b]` is not the middle of its own members, and the lateral
bins miss by twice what the front and top ones do. Commanding the centroid moves
the deployment vector to where the demonstrations are:

| train label → deploy command | median | p90 | max |
|---|---|---|---|
| grasp axis → bin axis *(run 1)* | 18.45° | 38.36° | 44.98° |
| grasp axis → **bin centroid** *(run 9)* | **16.20°** | **32.68°** | 55.08° |

**Be honest about the size of that: 2.25° of median and 5.68° of p90.** The
centroid does not remove the skew, and run 2's header is still right that a skew
of this order is comparable to the effect the conditioning is supposed to have.
What run 9 really moves is *where the same ~16–18° of angular budget is spent* —
runs 2–8 have zero skew and ~18.45° of label noise (told a sector, shown one
grasp inside it); run 9 has zero label noise (the label *is* the demonstration)
and ~16.20° of skew. Which end a policy tolerates better has never been measured
here. That is the question.

**Three things run 9 has that run 1 did not**, which is why it is the cleaner
test of the same idea rather than a repeat: the per-bin table (one demonstration
per reachable direction, not a max-separated pair), the demo filter, and — the
substantive one — **DAgger collection under the deployment command**. Run 1
rolled out conditioned on the grasp axis, so its on-policy states were states no
deployment ever visits. Run 9's collector conditions the learner on the centroid
while captioning the episode with the grasp axis, which is what DAgger actually
asks for: visit the states the deployed policy visits, label them with the
expert.

**No re-collection.** Run 2's tables and shards are the inputs verbatim — both
attrs are on all 1596 train and 69 val episodes, none zero-norm, checked. Base
collection replays an OMG plan with no policy in the loop, so `d_world` never
influenced a recorded state there. The miscaption filter stays on even though its
justification does not apply under a grasp-axis label, so run 9's aggregate is
the same 1575 episodes run 2's was and the two differ in captions and nothing
else.

**There is no single-change reference run.** Against run 2 this is two changes
(`d_noise_deg` 12 → 0 and the label/command rule); against run 7 it is also two
(`beta_end` 0.5 → 0.75 and the same rule). Run 9 keeps run 2's β because it is
"run 2 with the `d` changes" — matching run 7's 0.5 would buy a clean pair at the
cost of no longer building on the best run in the phase. What makes that
tolerable is that **one of the two is already measured**: run 7 against run 2
isolated `d_noise_deg` exactly, and it moved the base fit (0.143 → 0.370) and
nothing after it — by iteration 7 the two are indistinguishable. So from
iteration 7 on, run 9 against run 2 reads as one change. **The base fit does not:
compare it with run 7's 0.370, not run 2's 0.143.** Run 8's identically
configured iteration 0 scored 0.282 against run 7's 0.370, so **0.088 is the
noise floor** and nothing smaller is readable.

**What would falsify it.** `dir_err` should improve if precise labels beat an
exactly matched command. `dir_err` at run 7's level with `bin_diag_rate_b*`
falling means the skew dominates and run 2's argument wins — the response is then
to re-collect with `--command bin_centroid` so training and deployment agree *on
the centroid*, a third arm this run's data makes cheap to justify.

**One bug this run's plumbing fixed, affecting every earlier run's chained
number.** `chained_retry._run_attempt` conditioned the policy on
`approach_direction(grasp_pose)` — run 1's rule — while `chained_retry_scene`
picked `to_world(BINS[b], anchor_R)` and recorded *that* on the attempt. So from
run 2 on, the retry ladder commanded one vector and reported another, and no
metric could show it because `att.d_world` was the reported one. **Every
`chained_retry_at_k` produced before this fix was measured under the grasp axis
regardless of config.** Single-shot `success_rate` is unaffected.

---

### regrasp_run5 — the warm-start ablation, configured, not yet run

`examples/configs/regrasp_run5.yaml`. **One change** against run 3:
`TRAIN.train_from_scratch` true → **false**. A clean single-variable comparison,
which run 4 is not. Shares run 3's pin tables and shards, so the only cost is
training time.

**The same `iter_epochs: 15` means two different things**, which is the whole
point. Under FTL the model discards the previous weights and refits from a
random PointNet++ (`pc_pretrained` is empty), so 15 epochs is a *complete* fit
budget. Under a warm start it continues the previous iteration's `best.pt` —
encoder included — so 15 epochs is 15 *more* epochs. In gradient steps at batch
64:

| | run 3 (scratch) | run 5 (warm) |
|---|---|---|
| iteration 0 | 50 × 86k = **67k** | same |
| iteration 1 | 15 × 95k = **22k** — a third | 22k *on top of* iteration 0 |
| iteration 25 | 15 × 530k = **125k** | 125k on top of everything before |

So run 3 should dip below its base fit in the early iterations and recover as
\|D\| grows; run 5 should not dip at all.

**Both are worth running.** FTL is what Ross et al. specify and what Phase-4
run 16 (0.80, the best run in the project) did, and it has a real claim behind
it: a fresh fit on the whole aggregate cannot inherit an early iteration's
mistakes. The warm start is what both fast runs used, and is the only
configuration in which this project has yet seen `bin_diag_rate` climb past 0.8 —
`regrasp3_fast1` reached success 0.565 / `bin_diag` 0.848 by iteration 4 and
0.891 by iteration 5. This pair settles which matters more here.

---

### regrasp_run4 — the direction-noise ablation, configured, not yet run

`examples/configs/regrasp_run4.yaml`. Two changes against run 3 and nothing else:

| | run 3 | run 4 |
|---|---|---|
| `d_noise_deg` | 12° | **0°** |
| `beta_end` | 0.75 | **0.5** |

**Shares run 3's data entirely** — same p3 pin tables, same collected shards, so
the only cost is training time.

**What the noise was for.** At evaluation the policy is commanded a bin axis,
`to_world(BINS[b], anchor_R)`, exactly. In training it is shown a demonstration
whose realised approach direction sits a measured **median 18.4° from that axis**
(p90 38.5, max 45 = the bin half-width), because the demo is the goal-set grasp
*closest* to the axis rather than one on it. `d_noise_deg: 12` perturbs the
command so the policy sees a spread rather than one exact vector, on the theory
that this buys tolerance to the gap.

**The case against, which is why run 4 exists.** The deployment command is
noiseless and exact. Perturbing it in training may simply blur the conditioning —
teaching "somewhere near this direction" when the retry machine will only ever
ask for the axis itself. If run 4 improves `dir_err` or `bin_diag_rate`, the
augmentation was costing the precision it was meant to protect.

**A detail that sharpens it.** The perturbation is seeded on the EPISODE KEY, so
within a run each episode draws the *same* offset in every epoch. It is a fixed
relabelling, not a fresh sample per epoch — closer to label noise than to
augmentation, and a weaker regulariser than the flag name suggests.

**NOT A CLEAN ABLATION, and this is the caveat to carry.** `beta_end` moves too.
0.5 is outside the range anything on this task has run at: Phase-4 run 16 (0.80,
the best in the project) and regrasp run 1 both held 0.75 **constant**, and every
regrasp run to date has floored at 0.75. At 0.5, half of every late trajectory is
the learner's own actions — double run 16's on-policy fraction. A difference
against run 3 therefore has two candidate causes, and separating them needs a
third run at (noise off, `beta_end` 0.75).

---

### regrasp_run6 — every scene every iteration, configured, not yet run

`examples/configs/regrasp_run6.yaml`. **One change** against run 4:
`DAGGER.episodes_per_iter` 1200 → **7404** = 617 scenes × `max_grasps` 12.
`sample_scenes` caps at `len(pool)`, so any value at or above the cap means "all
of them" and self-corrects downward if `demo_ok_table` prunes the pool further.
Everything else is run 4 verbatim, and it shares run 3/4's pin tables and shards.

**What it tests.** Every other run samples 100 of 617 scenes a round, so a given
scene is revisited roughly once every six iterations and the aggregate grows
slowly. Here every scene is rolled out under the *current* policy every round:
maximum on-policy coverage, and the DAgger share of \|D\| passes half after a
**single** iteration (110k new steps against an 86k base) instead of after
thirteen. If DAgger's benefit is bounded by how much on-policy data it sees, this
is the run that shows it.

**It is also ~9 days of GPU.** Read this before submitting:

| | per iteration |
|---|---|
| collect | 4578 episodes / 20 workers ≈ 14 min |
| eval | 742 episodes / 20 workers ≈ 2 min |
| refit | 15 epochs × \|D\|, and \|D\| grows by 110k steps **every** iteration instead of 18k |

\|D\| after 25 iterations is **2.84M steps** against run 4's 530k; the refit total
is ~215 h against ~45 h, the whole run ~223 h ≈ 9.3 days over ten chained 24 h
jobs, and scratch holds ~47 GB of DAgger shards.

The refit cost is **quadratic in the iteration count** here — every round refits
from scratch over an aggregate that grew by 110k — so it is dominated by the last
few iterations. `num_iters: 8` costs ~30 h and still drives the on-policy share
past 0.9. If this run is worth doing at all it is probably worth doing at 8, and
that is a one-line change.

---

### regrasp_run3 — configured, not yet run

`examples/configs/regrasp_run3.yaml`. **Six config diffs from run 2, five of them
paths**; the only behavioural one is `episodes_per_iter`. The frame, `k`, the
command, the network, the loss, DART and the β schedule are all run 2's, verbatim.

**The change.** Each (scene, bin) carries **three** demonstrations instead of
one — the three goal-set grasps closest to that bin's axis, or all of them where
the bin holds fewer. 91% of (scene, bin) pairs on s0/train have three or more
members, and the assignment emits **4578** demonstrations against run 2's 1596,
over the same 617 scenes.

**What it teaches is not what run 2's per-bin change taught.** The three share
one command, so they do nothing further to break the scene→action confound —
that job belongs to the four *different* commands a scene supplies, unchanged
from run 2. What they teach is that a direction does not name a single pose,
which is true and is exactly the situation at deployment: `retry.next_direction`
issues a bin axis, and every grasp inside that bin is a correct answer to it.

**The risk is mode-averaging, and it is worth stating before the run rather than
diagnosing after.** `pose_loss: pm` is unimodal; three valid targets for one
input get averaged, and the mean of three grasps need not be a grasp. This is the
same failure that made multi-grasp Phase-4 data unusable without conditioning,
arriving now from within a single command rather than across several. It is not a
certainty — the three members lie within 45° of one axis rather than anywhere on
the sphere, so the modes are far closer together than Phase-4's were. The
signature to watch is `cond_sep` and the per-bin `bin_diag_rate` falling *while*
`train_loss` also falls. If that appears, `--per-bin 2` is the response, not an
architecture change.

**Two consequences that are not obvious from the diff.**

`max_grasps` goes 4 → 12, and `sample_pairs` draws `episodes_per_iter //
max_grasps` scenes. Leaving `m` at 400 would therefore have drawn **33** scenes
per iteration instead of 100, holding the episode count almost constant (~248 vs
~259) while quietly collapsing scene diversity — which is this project's measured
bottleneck, not episode count. Hence 1200, which holds 100 scenes and lets the
episode count rise to ~742. That is 2.9× run 2's collection *and* a |D| growing
2.9× faster, so budget roughly 2.5–3× run 2's wall clock. 800 (66 scenes, ~495
episodes) is the honest middle if that does not fit.

The pin table's slot order is **member-major** — `+x, +y, −y, +z, +x, …` rather
than `+x, +x, +x, +y, …` — and that is load-bearing rather than cosmetic. The
retry ladder reads a scene's slots in order, so bin-major grouping would make
`retry@2` a second attempt at the *same* direction under an *identical* command:
the policy would do the identical thing and `retry_at_k` would collapse into a
measure of simulator noise. Verified on the assignment: 0 of 617 scenes have a
repeated direction inside their first `n_bins` slots.

**Prerequisite: the direction table must be rebuilt.** `--members-per-bin` is a
build-time setting and run 2's table recorded one member per bin, so it cannot
serve this run — the assignment refuses rather than silently emitting one. Build
with 5 recorded and use 3, so trying 2 or 4 later is an offline re-run in seconds
instead of another 20-minute simulator pass.

**One known gap.** `GraspPinTable.keep_only` filters `(scene, bin)` pairs, not
slots, so `demo_ok_table` keeps or drops all three members of a bin together. At
one member per bin that was exactly right; at three, a bin whose first member
demonstrated cleanly but whose third had a pin miss survives whole. The audit's
`command vs bin axis` fingerprint (median 0.00° on run 2's clean shard) is the
check, and `BCDataset`'s per-episode `bin_assigned != bin_realized` filter is the
second line of defence.

### regrasp_fast1 — run 2's method at a fifth of the compute, and the first run where DAgger moved

DelftBlue went into a seven-day cluster-wide maintenance reservation before run 2
could start, so run 2's config was copied to `regrasp_run2_fast.yaml` with five
numbers turned down and **nothing else changed** — same one-demo-per-bin data,
same bin-axis command, same aux head, same β schedule, same `last` everywhere —
and run on this laptop instead. It completed all 8 iterations. Treat every level
below as unreliable and every *direction* as the thing worth reading: 40 eval
scenes give ~105 episodes and ~26 per bin, so a single per-bin point carries a
binomial standard error near 0.09.

| metric | first 4 iters (0–3) | last 4 iters (5–8) | Δ | run 1's Δ | chance |
|---|---|---|---|---|---|
| `success_rate` | 0.302 | 0.421 | **+0.119** | −0.008 | — |
| `grasp_rate` | 0.293 | 0.412 | +0.119 | — | — |
| `close_rate` | 0.369 | 0.493 | +0.124 | — | — |
| `dir_err` | 51.98° | 35.22° | **−16.8°** | — | — |
| `dir_track` | 0.422 | 0.609 | **+0.186** | +0.029 | — |
| `bin_diag_rate` | 0.562 | 0.745 | **+0.183** | +0.033 | **0.25** |
| `bin_hit_rate` | 0.236 | 0.298 | +0.062 | +0.046 | **0.25** |
| `cond_sep` | 0.542 | 0.594 | +0.052 | −0.045 | — |
| `retry@1` | 0.369 | 0.406 | +0.037 | — | — |
| `retry@4` | 0.481 | 0.606 | +0.125 | — | — |

**The conditioning is now obeyed, not merely read.** `bin_diag_rate` runs
0.467 → 0.771 across the eight iterations against a chance level of 0.25, where
run 1 sat essentially flat at 0.476 → 0.509 over nineteen. `dir_err` falls from
62° to 35°, i.e. from "barely better than a random axis" to "within a bin
half-width of the commanded one". Run 1's headline conclusion — *the channels
reach the network but not the action* — does not survive the two method changes,
and this is the result the phase existed to get.

**Success moved too, which run 1's does not let you assume.** 0.30 → 0.46 is
larger than the ±0.115 noise floor, but that floor was measured on 100 scenes and
this eval has 40, so a single-run success comparison is still not something to
lean on. What makes it more than noise is that it moves *together* with
`grasp_rate`, `close_rate` and the retry ladder, all monotone-ish over the same
iterations, and that `dir_track` — which starts near its floor and has room to
move — carries the same trend far outside its own error.

**`bin_hit_rate` is still the lagging metric, and the per-bin split says why.**
Pooled it reads 0.276 at the end, barely above chance, exactly as in run 1 — but
the four bins are nothing alike:

| bin | n | `success` | `dir_track` | `bin_diag` | `bin_hit` | `f_human_contact` |
|---|---|---|---|---|---|---|
| `+x` free end | 28 | 0.250 | 0.612 | 0.821 | **0.536** | 0.286 |
| `+y` lateral | 25 | **0.720** | 0.656 | 0.760 | 0.280 | 0.040 |
| `−y` lateral | 23 | 0.435 | 0.562 | 0.696 | 0.130 | 0.043 |
| `+z` top-down | 29 | 0.448 | 0.608 | 0.793 | 0.138 | 0.103 |

The gripper *orients* correctly for all four bins (`bin_diag` 0.70–0.82) but only
*arrives from* the commanded side for `+x`. `dir_err` and `sector_err` were split
apart for precisely this reason and the split is earning its keep. Note also that
`+x` — the free end, away from the giver, nominally the easiest approach and the
best-represented bin in the data at 490 scenes — has both the **lowest** success
and by far the highest human-contact failure rate. That is worth a look rather
than an explanation; nothing in the design predicts it.

**Retry is doing real work now.** At iteration 8 the ladder reads
0.375 / 0.600 / 0.700 / 0.700 — a **+0.325** spread from one attempt to four,
against run 1's `retry@2 − retry@1` of +0.054. Independent hypotheses that
actually differ are what the whole retry story needs, and this is the first
direct evidence of them.

**It is a combination run, not a controlled test.** Five things moved against
run 1 at once — the bin-axis command, one demo per bin instead of a separated
pair, the aux head on, β scheduled from 1.0, and `last` everywhere — so nothing
here attributes the gain to any single change. The two method changes are the
likely cause and the bin-axis fix is the one with a mechanism (run 1 trained on a
target a measured median 18.4° from the vector it would be commanded with at
deployment), but that is reasoning, not measurement.

**Caveats specific to how this run was executed.**

- `EVAL.holdout: false`, as in run 1 — the 40 eval scenes are also collected on,
  so these are train-set rates. The held-out number still has to come from
  `examples/eval_regrasp_testset.py`, which needs `output/direction_table_test.json`
  — the build that failed on DelftBlue (job 2656) and has not been re-run.
- Val loss climbs monotonically 0.255 → 0.297 while train loss falls, and
  `EVAL.ckpt: last` means the scored checkpoint is the progressively more overfit
  one. The success rate rose anyway, which is a second data point for the standing
  conclusion that val loss is a poor selection target here.
- The laptop suspended twice mid-run, breaking CUDA both times (`nvidia_uvm`
  needed reloading) and killing the process; the run resumed from `state.json`
  with no data lost. **Iteration 6's `collect_s` and `train_s` read 0** because
  its collection and refit had already completed before the crash — its wall time
  is not comparable with the others.
- `near_rate` is 0.000 throughout and remains dead under this methodology, as in
  run 1: it measures distance to a pose the policy is never given.

**What this licenses.** Running the full `regrasp_run2.yaml` on DelftBlue when the
reservation lifts, unchanged. The fast run was scoped to answer "does the
plumbing work and does `bin_diag_rate` leave chance" and it answered both yes; it
cannot settle levels, and the per-bin `bin_hit` disparity above is the thing the
full run should be read for first.

### regrasp3_fast1 — three demonstrations per bin (COMPLETE, 6 of 6)

Finished 2026-08-25 on 20 eval scenes (~138 episodes, ~35 per live bin). Final
`success 0.500`, `dir_track 0.687`, `bin_diag 0.855`; best iteration **4** at
`success 0.565`. \|D\| ended at 6730 episodes / 157463 steps.

Same 617 scenes and the same frame, k, command, network, loss, DART and β as run
2. The one method change is **three demonstrations per bin instead of one**: for
each (scene, bin) the three goal-set grasps closest to that bin's axis, giving
**4578 base episodes / 86315 steps**, 2.87× regrasp_fast1's. Collected in ~1.2 h
by four concurrent `collect_regrasp_demos --shard i/4` processes, read by the
trainer as a list rather than merged. `iter_epochs` was held at run 3's 15, which
is what forced `num_iters` down to 6.

| iter | `succ` | `dir_track` | `dir_err` | `bin_diag` | `retry@1→@4` |
|---|---|---|---|---|---|
| 0 (base fit) | 0.406 | 0.344 | 59.0° | 0.406 | 0.55 → 0.60 |
| 1 | 0.435 | 0.586 | 37.2° | 0.739 | 0.35 → 0.60 |
| 2 | 0.435 | 0.658 | 30.8° | 0.739 | 0.50 → 0.70 |
| 3 | 0.478 | 0.686 | 28.2° | 0.826 | 0.50 → 0.70 |
| 4 | **0.565** | 0.715 | **25.7°** | 0.848 | — |
| 5 | 0.500 | 0.707 | 26.4° | **0.891** | 0.30 → 0.70 |
| 6 (final) | 0.500 | 0.687 | 28.2° | 0.855 | 0.40 → 0.70 |
| *regrasp_fast1, it 8 (final)* | *0.457* | *0.611* | — | *0.771* | *0.375 → 0.70* |

**It passed regrasp_fast1's FINAL numbers at iteration 1 and never gave them
back.** One demonstration per bin took eight DAgger rounds to reach `bin_diag`
0.771; three per bin reached 0.739 in ONE round and peaked at **0.891** by the
fifth, against a chance level of 0.25. `dir_err` fell 59.0° → 25.7° and closed at
28.2°.

**Everything of value arrived in the first three rounds.** Between iterations 3
and 6 `succ` moved 0.478 → 0.500, `dir_track` 0.686 → 0.687 and `dir_err` 28.2° →
28.2° — on ~138 episodes those are all inside the binomial noise. The gain is
front-loaded, and that is the single most useful thing this run says about how to
spend the cluster budget.

**Success peaked at iteration 4 (0.565) and settled at 0.500** while `bin_diag`
went on climbing to iteration 5. A 0.065 move on 138 episodes is inside the
noise, so read the peak as "somewhere around 0.5" rather than as a maximum — but
the two metrics did stop moving together, which is the first sign in this run
that following the command and succeeding are separable.

**The mode-averaging risk did not materialise, and that is the result.** Three
valid grasps sharing one command under a unimodal `pm` loss could have been
averaged into an invalid one; the signature would have been `bin_diag` and
`cond_sep` FALLING while `train_loss` fell. Train loss fell (0.156 → 0.129) and
`bin_diag` rose 0.406 → 0.855; `cond_sep` rose 0.407 → 0.587. The direction
table's own warning that **37% of bins have their members within 2 cm** — i.e.
those extra demonstrations are near-duplicates — evidently costs nothing either.

**`bin_hit_rate` is still the lagging metric but it moved most at the end**:
0.435 pooled against fast1's 0.276, and the per-bin split is no longer one bin
carrying it. Final iteration:

| bin | n | `succ` | `dir_track` | `bin_diag` | `bin_hit` |
|---|---|---|---|---|---|
| `+x` free end | 36 | 0.500 | 0.666 | 0.833 | **0.833** |
| `+y` lateral | 35 | 0.600 | 0.545 | 0.686 | 0.171 |
| `−y` lateral | 24 | **0.750** | 0.770 | **1.000** | 0.625 |
| `+z` top-down | 43 | 0.279 | **0.773** | 0.930 | 0.209 |

`−y` ends at `bin_diag` **1.000** — every episode finished with its approach axis
in the commanded bin — and arrives from the commanded side 62.5% of the time
where fast1 read 0.130. So "only `+x` arrives correctly" was a property of the
thinner dataset, not of the method.

**`+y` is the standing outlier**: near-highest success, `bin_hit` six times lower
than `+x`. It succeeds by coming in from somewhere other than where it was told,
consistently, across every iteration — worth understanding before run 3 proper.

**Per-bin numbers churn hard at n ≈ 35, and the last iteration is the warning.**
`+z` went 0.628 → 0.279 in success between iterations 5 and 6 while its
`bin_diag` barely moved (1.000 → 0.930), and `+x` went 0.250 → 0.500 the other
way. Those are ±0.35 swings on ~40 episodes with the pooled rate flat at 0.500;
read the per-bin success column as an ordering hint, not a measurement. The full
run's 100 scenes (~65/bin) is what makes it readable.

**Caveats.** 20 eval scenes, `EVAL.holdout: false`, one run. Iteration 2's
`collect_s` reads 0 because the run was killed and resumed at that point, so its
wall time is not comparable. `near_rate` is 0.000 for iterations 0–4 and 0.065 /
0.029 at 5 and 6 — effectively dead under this methodology, as in every Regrasp
run.

**What this changed for run 3 proper.** The per-bin count is worth keeping at 3 —
the mode-averaging question is settled. The open question is the budget split:
the gain arrived almost entirely in the first three rounds and the last three
were flat, so on the cluster the marginal iteration is worth less than the
config's `num_iters: 25` assumes, and `iter_epochs: 15` is buying more than the
iteration count is.

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
| `regrasp_run1.yaml` | regrasp_run1 (`examples/train_regrasp.py`) |
| `regrasp_run2.yaml` | regrasp_run2 — configured, awaiting the cluster |
| `regrasp_run2_fast.yaml` | regrasp_fast1 — = run 2 with five numbers turned down, laptop-sized |
| `regrasp_run3.yaml` | regrasp_run3 — = run 2 with three demonstrations per bin |
| `regrasp_run3_fast.yaml` | regrasp3_fast1 — = run 3, laptop-sized (6 iters, 15/15 epochs) |
| `regrasp_run4.yaml` | regrasp_run4 — = run 3 + noise off + `beta_end` 0.5 |
| `regrasp_run5.yaml` | regrasp_run5 — = run 3 + warm start |
| `regrasp_run6.yaml` | regrasp_run6 — = run 4 + every scene every iteration (~9 days) |
| `regrasp_run7.yaml` | regrasp_run7 — = run 2 + noise off + `beta_end` 0.5 (run 4's changes at 1 demo/bin) |
| `regrasp_run8.yaml` | regrasp_run8 — = run 7 + warm start (run 5's ablation at 1 demo/bin) |
| `regrasp_run9.yaml` | regrasp_run9 — = run 2 + train on the grasp axis, deploy on the bin centroid |
| `bc_regrasp_run9.yaml` | run 9 `TRAIN.train_cfg` — = `bc_regrasp_run4.yaml` + `DATA.d_source: d_grasp_world` |
| `regrasp_run10.yaml` | regrasp_run10 — = run 2 + `d` from the grasp's position (`SIM.d_rule: grasp_offset`), noise off |
| `regrasp_smoke.yaml` | Regrasp shakedown — 2 iters, m=8, 3 eval scenes |
| `bc_regrasp.yaml` | regrasp_run1 `TRAIN.train_cfg` — `direction_cond`, aux head **off**, head `[256,256]` |
| `bc_regrasp_run2.yaml` | regrasp_run2, regrasp_run3, regrasp_run5 and both fast runs `TRAIN.train_cfg` — = `bc_regrasp.yaml` + aux head **on** (`aux_weight: 1.0`) |
| `bc_regrasp_run4.yaml` | regrasp_run4, run6, run7, run8 `TRAIN.train_cfg` — = `bc_regrasp_run2.yaml` with `d_noise_deg` 12 → **0** |

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
