# Runbook — Regrasp runs 19 and 20 on DelftBlue

Hand this to an agent on the cluster. It covers **both** runs because neither
answers anything alone.

**Standing rule for whoever executes this: do not `git commit` or `git add`
anything unprompted. Propose the command and let the user run it.**

---

## 0. What these two runs are

Both descend from **run 9** (`d_rule: approach_axis`, `command_deploy:
bin_centroid`, `d_source: d_grasp_world`, FTL from scratch, 25 iterations).

| | vs its parent | changes | needs a rebuild? |
|---|---|---|---|
| **run 19** | run 9 | `DAGGER.shield: true` **and** `TRAIN.iter_epochs` 15 → 20 | **no** |
| **run 20** | run 19 | `SIM.anchor_hand_ref: wrist → base` | **yes, everything upstream** |

### Order is not negotiable: run 19 first

Run 20 against **run 9** confounds three changes. Only run 20 against **run 19**
isolates the frame. Run 19 is also the cheaper of the two (no rebuild), so it is
first on both counts.

Run 19 is itself **two changes and not a clean ablation** — the shield acts on
*which states get collected*, the epochs on *how hard the aggregate is fitted*.
If run 19 beats run 9, that is not yet a result about either mechanism. The two
runs that would settle it are `run 9 + shield only` and `run 9 + epochs only`,
and neither exists yet. Say so in any write-up.

### ⚠ Read this before reading any number

**Expect run 20 to score about the same as run 19 in simulation, and treat that
as success.** Only 7.7% of grasps change bin between the wrist and base frames,
so ~92% of commands are the same vector and the sim task is barely different. A
flat result means deployability was bought for free.

Run 20's motivation is **deployment, not sim**. Runs 1–19 name their bins in the
MANO-wrist frame; the rig has no MANO, so `my_regrasp_policy_runner` substitutes
the robot base and then applies the run's *wrist*-frame `command_axes.json`
centroids under a *base*-frame anchor rotation. Run 9 on hardware showed the
signature — the per-bin **ordering** did not match sim:

| bin | sim, held-out | run 9 on the robot |
|---|---|---|
| `+x` | **0.532** (best) | bad — drove at the table |
| `+y` | 0.471 | the only one that half-worked |
| `+z` | 0.420 | not reported |
| `−y` | **0.290** (worst) | bad |

A uniform sim2real loss lowers every bin. It cannot **reorder** them. A rotated
frame can. Run 20 removes the substitution by training in the frame the rig
computes.

A large sim **regression** in run 20 would be the surprising outcome, and would
mean the base frame is genuinely harder to condition on rather than merely
differently labelled.

### The baselines, and the noise floor

Run 9's in-loop eval is **train-set** (`EVAL.holdout: false`, 100 scenes):

- iterations 0–4: 0.222 · iterations 5–13: 0.459 (the climb)
- **iterations 14–25: mean 0.504, sd 0.053, trend +0.47 pp/iter — flat**
- selected best 0.592 @it23, which is **+0.088 over the plateau mean**

Run 9's held-out score at that same checkpoint (129 scenes, 340 episodes):
**0.4324**.

**The measured run-to-run noise floor is 0.088.** An identical rerun of run 7's
iteration 0 scored 0.282 against 0.370. On 100 eval scenes the binomial 95% band
is ±0.098; on the 340-episode test split it is ±0.053.

**Nothing below ~0.09 on the in-loop eval is a result.** Every regrasp run from 2
to 12 (0.555–0.619) sits inside that band. Do not report run 19 or run 20 beating
run 9 unless the gap clears it, and prefer the plateau mean over the selected
best when comparing.

### Run 9's failure budget — what run 19's shield is aimed at

Held-out, 340 episodes, every episode's terminal outcome:

| outcome | share |
|---|---|
| success | 43.2% |
| knocked the object down | **26.8%** |
| hit the human | **19.1%** |
| no release | 7.6% |
| timeout | 3.2% |
| closed on nothing | **0.0%** |

Collisions are 45.9% of episodes and 81% of all failures. `box_chance` is 44.7%
and `box_taken` 82.2%: fewer than half of episodes ever get the object between
the jaws, and when they do it converts. **The approach destroys the scene before
the grasp is attempted** — that is what the shield targets.

---

## 1. Prerequisites

```bash
cd ~/h2r/handover-sim2real          # wherever the repo lives on the cluster
git log --oneline -3                # confirm the run 19/20 commit is present
ls examples/configs/regrasp_run19.yaml examples/configs/regrasp_run20.yaml
ls examples/slurm/regrasp_run20_all.sbatch examples/slurm/train_regrasp.sbatch
```

Run 19 additionally needs run 9's substrate already on the cluster:

```bash
ls -la output/regrasp_pins_train.json \
       output/regrasp_pins_train_excluded.json \
       output/regrasp_demos_train_ok.json
ls -la "$SCRATCH_ROOT/output/bc_dataset/train_regrasp.h5" \
       "$SCRATCH_ROOT/output/bc_dataset/val_regrasp.h5"
```

If those are missing, run 19 cannot start — it rebuilds nothing. Run 20 builds
its own from scratch and needs none of them.

---

## 2. ⚠ Disk, before anything else

`/home` is a **hard 30 GB quota that fills silently**. There is no ENOSPC
traceback — Python cannot write one to a full disk — so the job dies with **exit
code 6 and an empty `.err` file**, potentially five hours into a collection.

Each run needs about 5 GB (base shards ~0.6, DAgger shards ~3.1, checkpoints
~1.2, tables ~0.003). Run 20 needs its own copy of the base shards, so **budget
~10 GB for the pair.**

```bash
df -h "$HOME" | tail -1
du -sh "$SCRATCH_ROOT/output/bc_dataset" "$SCRATCH_ROOT/output/dagger_runs" 2>/dev/null
```

Below ~12 GB free, clear space before submitting. Old `iters/*/checkpoints` of
finished runs are the usual candidates — **ask the user first**, they are not
reproducible.

---

## 3. RUN 19

### 3.1 Preflight

```bash
bash examples/slurm/preflight_regrasp.sh regrasp_run19
```

It checks disk, environment, inputs, and config-vs-table consistency
(`d_rule`, `d_point_depth`, `anchor_hand_ref`). Missing inputs are reported as a
plan, not an error — but for run 19 they are an error, because nothing rebuilds
them. All four sections must pass.

### 3.2 Submit — one job

Run 9 measured **13.4 h** over 26 iterations (30.9 min/iteration mean, rising to
52.6 min as the aggregate grows). The shield is free — one numpy sweep per step.
`iter_epochs` 15 → 20 is +33% on the training half. **Budget 16–17 h.**

```bash
sbatch --time=20:00:00 \
  --export=ALL,RUN=regrasp_run19,CFG=examples/configs/regrasp_run19.yaml,SCRATCH_ROOT=$HOME/h2r-runs \
  examples/slurm/train_regrasp.sbatch
```

**20 h, not 24.** A 24 h request is DelftBlue's maximum, so the scheduler can
never backfill it into a gap; with fairshare depressed after a few GPU jobs that
is the difference between starting in an hour and starting in two days. 20 h
gives ~18% buffer over the estimate.

Do **not** override `--cpus-per-task`, `--gpus-per-task` or the partition. That
block is load-bearing: deviating from it produces `AssocMaxGRESPerJob` even
though the request sits inside the association limit.

The run is resumable from `state.json`. If 20 h is not enough, resubmit the
identical command.

### 3.3 Monitor

```bash
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.10l %R"
tail -f slurm_logs/regrasp_<jobid>.out
```

### 3.4 Gates — in this order

**Gate 1 — the shield's firing rate. Check this at iteration 1, ~40 minutes in.
Do not let the job run overnight without it.**

```bash
python3 - <<'PY'
import csv
r = list(csv.DictReader(open("$SCRATCH_ROOT/output/dagger_runs/regrasp_run19/dagger_log.csv")))[0]
steps, exp = float(r["steps"]), float(r["expert_steps"])
sp, sj, sb = float(r["shield_policy"]), float(r["shield_jolt_reject"]), float(r["shield_blind"])
learner = steps - exp
print(f"steps {steps:.0f}  expert {exp:.0f}  learner-driven {learner:.0f}")
print(f"shield_policy {sp:.0f}  -> {100*sp/max(learner,1):.1f}% of the learner's own control taken")
print(f"shield_jolt_reject {sj:.0f}   shield_blind {sb:.0f}   dart {r['dart']}")
PY
```

- **Under ~15%** — healthy. The shield is catching the tail, β is effectively
  still 0.75, and run 19 is a shield test.
- **15–40%** — usable but note it. The effective β is meaningfully above 0.75 and
  any gain is partly a higher-β effect.
- **Over ~40%** — **stop and report.** This is a high-β run wearing a shield, not
  a shield test. The honest comparison would be run 9 re-run at the matched
  effective β. Do not let it burn 17 hours first.

Also check `shield_jolt_reject` against `dart`: if it approaches 5× the jolt
count, every draw is being refused and jolts are being skipped rather than
shielded — the same reading `dart_reject` gets.

**Gate 2 — iteration 0 reproduces run 9's base fit.** Iteration 0 is the base fit
plus DAgger round 1; the shield cannot change the base fit, but `iter_epochs`
does not apply to it either (`base_epochs: 50` is unchanged). Expect
`success_rate` near run 9's iteration 0 of **0.235**, within the 0.088 floor. A
much lower number means the substrate is wrong, not that the run is bad.

**Gate 3 — the recovery from the FTL dip, iterations 1–3.** Under
`train_from_scratch` each refit starts from a random PointNet++, so early
iterations sit below the base fit and recover as D grows. Run 9 took until
iteration 5 to clear 0.35. **Run 19 fits 20 epochs instead of 15 specifically to
shorten this**, so iterations 1–3 clearing run 9's 0.206 / 0.235 / 0.202 is the
first sign the epoch change did anything.

**Gate 4 — the plateau, iterations 14–25.** This is the actual result. Compare
the **mean over iterations 14–25**, not the max:

```bash
python3 - <<'PY'
import csv, statistics as st
for run in ("regrasp_run9", "regrasp_run19"):
    try:
        s = [float(r["success_rate"]) for r in
             csv.DictReader(open(f"$SCRATCH_ROOT/output/dagger_runs/{run}/dagger_log.csv"))]
    except FileNotFoundError:
        continue
    pl = s[14:]
    if pl:
        print(f"{run:16} iters 14+: mean {st.mean(pl):.3f} sd {st.pstdev(pl):.3f} "
              f"max {max(pl):.3f}  (n={len(pl)})")
print("noise floor 0.088 — a difference below it is not a result")
PY
```

### 3.5 Plot

```bash
python examples/plot_regrasp_run.py "$SCRATCH_ROOT/output/dagger_runs/regrasp_run19"
```

Writes `training_curve.png`, `curves_regrasp.png`, `curves_diag.png` into the run
dir. Report the plateau mean, the per-bin `succ_bin_*`, and the Gate 1 shield
fraction together — the last one is what makes the others interpretable.

---

## 4. RUN 20

Only start this once run 19 has a plateau number. Run 20's whole purpose is the
comparison against it.

### 4.1 Preflight

```bash
bash examples/slurm/preflight_regrasp.sh regrasp_run20
```

On a first run the `_aabase` inputs will be **missing, and that is expected** —
the sbatch builds them. Check 1 (disk) and check 2 (environment) must pass.

### 4.2 Submit

```bash
sbatch examples/slurm/regrasp_run20_all.sbatch
```

That is the whole recipe. The script's own `#SBATCH --time` is already
`20:00:00`; it takes `SCRATCH_ROOT` from the environment or defaults to
`$HOME/h2r-runs`.

**It will probably not finish in one allocation:**

| stage | |
|---|---|
| direction table, train | 1.5 h |
| direction table, val | 0.2 h |
| assign per-bin demos | seconds |
| collect base, train | 5.0 h |
| collect base, val | 0.3 h |
| audit | minutes |
| train, 25 iterations | 16–17 h |
| **total** | **23–24 h** |

24 h is the hard maximum, so nothing fits with margin. **The script is
idempotent**: every stage is skipped when its output exists, training resumes
from `state.json`, and both collections and both table builds write
`<path>.partial` and rename only after a zero exit — so a kill cannot leave a
truncated shard the skip-check mistakes for finished.

**Resubmit the identical command to continue. Expect two submissions.**

If the queue is bad, `sbatch --time=08:00:00 ...` works fine — the script just
completes fewer stages per pass.

### 4.3 Gates

**Gate A — the per-bin histogram, printed by the assignment stage ~1.7 h in.
This is the cheap gate and it gates 5 h of collection.**

The base frame **relabels** bins rather than renaming them, so the histogram
should resemble run 9's, with ~7.7% of grasps moved:

| bin | run 9 (wrist) | run 20 (base) expectation |
|---|---|---|
| `+x` | 486 | ~450–520 |
| `+y` | 359 | ~330–390 |
| `−y` | 319 | ~290–350 |
| `+z` | 412 | ~390–430 |
| `−x`, `−z` | 12, 0 | still ~0 |

**A wildly different shape — in particular `+x` and `−x` swapping mass — means
the sign convention is inverted** (`c − p_base` instead of `p_base − c`) and
every bin label with it. Stop and report; do not spend the collection.

`−x` and `−z` staying empty is correct and expected: `−z` is geometrically
impossible with the object held above a table, and `−x` (over the giver's
fingers) is what the hand-collision filtering removes. They are kept in the bin
set, empty, because dropping them would renumber bins and break every historical
`succ_bin_*` and the retry ladder.

**Gate B — `anchor_mode` on the collected shards.** Under `base` the hysteretic
wrist→base fallback is skipped entirely, so every episode should report
`base_primary`, never `base` or `wrist`. A single `wrist` means the config did
not take.

**Gate C — the shield rate**, exactly as run 19's Gate 1. It should be close to
run 19's; a large divergence is itself informative, since the frame changes which
directions get commanded.

**Gate D — the plateau against run 19.** Same script as run 19's Gate 4, with
`regrasp_run20` added. **Flat is the expected and good outcome** — see §0.

### 4.4 Reading run 20

Report three things, in this order:

1. **Is the sim plateau within 0.088 of run 19's?** If yes, the frame change is
   free in sim and the run succeeded.
2. **Did the per-bin ordering change?** Compare `succ_bin_*` between 19 and 20.
   Under the base frame the bins mean different physical directions, so some
   reordering is expected and is not a regression.
3. **`command_axes.json`** — the six bin centroids are now in the base frame.
   These are what a deployment will issue. Record them; they are the artifact the
   robot actually consumes.

---

## 5. Scoring on the held-out test split

### Run 19 — works as-is

Run 19 keeps run 9's wrist frame and `approach_axis` rule, so the existing test
table matches.

```bash
RUN=regrasp_run19 sbatch --time=03:00:00 examples/slurm/eval_regrasp_testset.sbatch
```

Measured cost: **4.4 s/episode, 340 episodes = ~25 min/iteration**, so all 26
iterations is ~10.8 h. Chain four 3 h passes, or score only the best few
iterations with `ITERS=20,21,22,23,24,25`. The sweep is resumable per iteration.

### ⚠ Run 20 — the test table must be rebuilt first

`output/regrasp_pins_test.json` records **no `anchor_hand_ref` in its `_meta`**,
which `setup.resolve_anchor_ref` reads as *wrist by construction*. Scoring run 20
against it will be **refused**, and correctly so — it would mis-caption every
bin.

Rebuild the test table under the base frame first (~25 min, 144 scenes):

```bash
# PROPOSE, do not run unprompted
python examples/build_direction_table.py \
    --cfg-file examples/pretrain_multicam_wr.yaml \
    --split test --out output/direction_table_test_aabase.json \
    --d-rule approach_axis --anchor-hand-ref base

python examples/assign_direction_demos.py \
    --table output/direction_table_test_aabase.json \
    --out output/regrasp_pins_test_aabase --mode per-bin
```

Then point the eval at it. The existing test split is 129 scenes / 340 episodes
with `demos_per_bin [94, 0, 70, 76, 100, 0]`; the rebuilt one should be close but
**will not be identical**, so run 19 and run 20 test numbers are on slightly
different populations. Say so when comparing them.

---

## 6. Known failure modes

### 6.1 Exit code 6, empty `.err`
Disk. `/home` filled mid-write. See §2. Free space and resubmit — the idempotent
script picks up.

### 6.2 `AssocMaxGRESPerJob` in `squeue`'s reason column
A known false reason code on DelftBlue, forum-confirmed. If the request matches
the sbatch's own block, wait — it schedules.

### 6.3 `[cfg] SIM.anchor_hand_ref: 'base' but ... was built with 'wrist'`
The guard working. Either a run-20 stage is pointed at run 9's tables, or an
`_aabase` path is wrong. Check that every `_aabase` file in the config exists and
that `_meta.anchor_hand_ref` reads `base`:

```bash
python3 -c "import json; print(json.load(open('output/regrasp_pins_train_aabase.json'))['_meta'].get('anchor_hand_ref'))"
```

### 6.4 A similar refusal naming `d_rule`
Same class. Run 20's tables must record `approach_axis`; run 18's `_bframe`
tables record `grasp_offset` and are **not** interchangeable despite sharing the
frame.

### 6.5 `OMG_PLANNER_DIR is not set`
`env_setup` imports OMG at module scope even when nothing plans. The sbatch
exports it; a manual invocation must too.

### 6.6 A stage re-runs that should have been skipped
The skip checks are `[ ! -f <path> ]` on the exact configured path. A partially
written file was renamed, or `SCRATCH_ROOT` differs between passes. Compare the
`REGRASP_DATA=` line in both job logs.

### 6.7 Iteration count stalls across a pass
`state.json` is not advancing. Check the tail of the previous job's `.out` for a
traceback; a pass that dies during a refit leaves the iteration incomplete and
redoes it.

### 6.8 `shield_policy` is `-1` in the log
The collector returned no shield counters — an old checkout. Confirm
`handover_sim2real/regrasp/collector.py` defines `_step_is_safe`.

---

## 7. Quick reference

```bash
# environment
cd ~/h2r/handover-sim2real
export SCRATCH_ROOT=$HOME/h2r-runs
conda activate pch2r_dev

# ---- run 19: no rebuild, one job, ~16-17 h ------------------------------
bash examples/slurm/preflight_regrasp.sh regrasp_run19
sbatch --time=20:00:00 \
  --export=ALL,RUN=regrasp_run19,CFG=examples/configs/regrasp_run19.yaml,SCRATCH_ROOT=$HOME/h2r-runs \
  examples/slurm/train_regrasp.sbatch

# ---- run 20: full rebuild, idempotent, ~23-24 h, expect two submissions --
bash examples/slurm/preflight_regrasp.sh regrasp_run20
sbatch examples/slurm/regrasp_run20_all.sbatch
sbatch examples/slurm/regrasp_run20_all.sbatch     # again, after the first ends

# ---- progress -----------------------------------------------------------
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.10l %R"
tail -f slurm_logs/regrasp_<jobid>.out
tail -f slurm_logs/rg20_<jobid>.out
wc -l "$SCRATCH_ROOT"/output/dagger_runs/regrasp_run{19,20}/dagger_log.csv

# ---- plots --------------------------------------------------------------
python examples/plot_regrasp_run.py "$SCRATCH_ROOT/output/dagger_runs/regrasp_run19"
python examples/plot_regrasp_run.py "$SCRATCH_ROOT/output/dagger_runs/regrasp_run20"
```

### The four numbers to report

1. **Gate 1 shield fraction** — `shield_policy / (steps − expert_steps)`. Without
   it nothing else is interpretable.
2. **Plateau mean over iterations 14–25**, not the selected best.
3. **The gap against the 0.088 noise floor.** Below it, say "no measurable
   difference", not "slightly better".
4. **Run 20's per-bin `succ_bin_*`** and its `command_axes.json` — the latter is
   what the robot will consume.
