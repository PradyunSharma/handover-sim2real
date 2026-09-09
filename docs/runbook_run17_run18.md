# Runbook — Regrasp runs 18 and 17 on DelftBlue

Self-contained: the reader is assumed to have the repo and cluster access but
**not** the conversation that designed these runs. Rationale and measurements
live in each config's header and in [`docs/run_index.md`](run_index.md); this
file is the sequence of actions, the gates between them, and what to do when a
gate fails.

**Standing rules for whoever executes this**

- Do **not** `git commit` or `git add` anything unprompted. Propose the command
  and let the user run it.
- Do **not** launch a job the runbook does not list. Every `sbatch` here is
  written out in full.
- If a config header disagrees with this file, **the config header wins** — it
  is next to the code that reads it.

---

## 0. What these two runs are

Run 16 ran and failed: headline `success_rate` peaked at 0.4419 against run
11's 0.6186, and twenty DAgger iterations never beat iteration 0. The
post-mortem (`docs/run_index.md`, run 16 section) found the cause was the anchor
frame's azimuth reference, `SIM.anchor_hand_ref: hand_centroid` — the centroid
of the segmented hand cloud, which sits *on* the object the hand is holding and
therefore has almost no lever arm. Consequences, all measured on run 16's own
shards: the frame rotated a median 10.65° within one episode (p90 145°, 15.8%
past 90°), `bin_assigned == bin_realized` fell from 99% to 40%, and only
**16.2%** of each DAgger shard reached training against run 11's 57.7%.

**Run 18** = run 16 with **one key changed**, `anchor_hand_ref: hand_centroid →
base`. The anchor's x axis becomes `normalize(horizontal(p_base − c))` — the
object→robot azimuth, with the giver removed from the definition entirely. Lever
arm 61.3 cm with a hard 41.8 cm floor, so live re-anchoring drifts 1.49° per
episode instead of 10.65°, and it needs nothing the real rig lacks.

**Run 17** = run 18's frame **plus** `d_rule: grasp_offset → location_extent`,
where `d = m·u` is an extent-normalized *location* on the object rather than a
direction. Read run 17 **against run 18**, not against run 16.

### Run 18 first, then run 17

Run 18 is the baseline run 17 has to be interpreted against, and it is the
cheaper failure: if the frame repair does not restore the DAgger keep-rate,
`location_extent` is untestable and run 17 should not be launched. The two runs
share no upstream artifacts (the `d_rule` change re-bins everything), so
there is nothing to reuse in either direction — sequencing costs nothing.

### ⚠ One open decision, unresolved at the time of writing

Run 18 does **not** fix the second data-loss cause. `grasp_offset` is
`normalize(grasp_point − c)`, a chord of only **3.83 cm**, and with a live `c`
the observed object centroid migrates 3.42 cm over an episode, so **24.0%** of
chords fall under `SIM.d_min_offset` (0.02 m) by the close and the episode loses
its direction label entirely. Freezing the centroid the `d`-rule measures from
takes that to **0.0%**; it was left out so run 18 tests the frame alone.

Expected keep-rate: run 16 **16%** → run 18 **~44%** → run 11's **57.7%** only
with the freeze as well.

**If the user has since asked for that fix**, the configs will carry a new key
and this paragraph is stale — check `git log` on
`examples/configs/regrasp_run18.yaml` and trust the config. Otherwise run as
written.

---

## 1. Prerequisites

```bash
cd ~/h2r/handover-sim2real          # or wherever the repo lives
git log --oneline -3                # confirm the expected commit is checked out
```

The runs need these to be present and current — all were added or changed for
run 18 and run 17:

| file | why it matters |
|---|---|
| `handover_sim2real/regrasp/anchor.py` | the `base` reference itself |
| `handover_sim2real/regrasp/setup.py` | `resolve_anchor_ref`, the config/table guard |
| `handover_sim2real/regrasp/evaluator.py` | the in-table metric columns |
| `examples/build_direction_table.py` | `--anchor-hand-ref` (**new** — run 16 had no such flag) |
| `examples/collect_regrasp_demos.py` | `--anchor-hand-ref` (**new**) |
| `examples/train_regrasp.py` | `success_rate_in_table`, corrected `D_episodes` |
| `examples/configs/regrasp_run18.yaml` | the run |
| `examples/configs/regrasp_run17.yaml` | the run |
| `examples/slurm/regrasp_run18_all.sbatch` | all five stages, idempotent |
| `examples/slurm/regrasp_run17_all.sbatch` | all five stages, idempotent |
| `examples/slurm/preflight_regrasp.sh` | the pre-submission check |

Quick check that the new flag exists — if this prints nothing, the checkout is
too old and nothing below will work:

```bash
python examples/build_direction_table.py --help | grep anchor-hand-ref
```

Environment (the sbatch sets these itself; needed only for interactive checks):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate pch2r_dev
export GADDPG_DIR=$PWD/GA-DDPG
export OMG_PLANNER_DIR=$PWD/OMG-Planner
export SCRATCH_ROOT=$HOME/h2r-runs
export REGRASP_DATA=$SCRATCH_ROOT/output
```

---

## 2. ⚠ Disk, before anything else

**Everything goes on `/home`, which is a hard 30 GB quota that fills SILENTLY.**
There is no `ENOSPC` traceback — Python cannot write one to a full disk — so the
job dies with **exit code 6 and an empty `.err` file**, six hours into a
collection. This is the single most expensive failure available here and the one
that reports nothing.

Each run needs about **5 GB**:

```
base shards (train ~570 MB + val ~23 MB)     ~0.6 GB
run dir: data/   (25 DAgger shards)          ~3.1 GB
run dir: iters/  (per-iteration checkpoints) ~1.2 GB
direction + pin tables                       ~0.003 GB
```

Run 16's own directory is **4.0 GB** (3.0 GB of it `data/`). Two new runs plus
run 16 is ~14 GB before anything else on `/home` is counted.

```bash
quota -s
df -h $HOME | tail -1
du -sh $HOME/h2r-runs/output/dagger_runs/* 2>/dev/null | sort -rh | head
```

If free space is under ~12 GB, propose to the user that run 16's DAgger shards
be archived or deleted — they are reproducible and no longer needed, since run
16 is superseded:

```bash
# PROPOSE, do not run unprompted
du -sh $HOME/h2r-runs/output/dagger_runs/regrasp_run16/data
rm -rf $HOME/h2r-runs/output/dagger_runs/regrasp_run16/data   # frees ~3 GB
```

Keep `dagger_log.csv`, `state.json`, `config.yaml` and the PNGs — the
post-mortem numbers are read from those.

---

## 3. RUN 18

### 3.1 Preflight

```bash
bash examples/slurm/preflight_regrasp.sh regrasp_run18
```

Four checks, in the order they would kill the job: disk, environment, declared
inputs, and config-vs-table consistency. Expected on a first run:

- `[1/4]` OK, with ≥ 5 GB free
- `[2/4]` OK — `GADDPG_DIR`, `OMG_PLANNER_DIR`, torch + CUDA, h5py
- `[3/4]` **`NOTE  N input(s) missing` — this is EXPECTED.** Stages 1–4 of the
  sbatch build exactly those. It is only a problem if you intended to reuse an
  existing table.
- `[4/4]` `not built yet — nothing to compare (fine)`

**Do not submit if `[1/4]` or `[2/4]` fails.** A `[4/4]` MISMATCH means a stale
table exists at the run's paths — see §5.3.

### 3.2 Submit — request per PHASE, never 24 h

**DelftBlue backfills short jobs ahead of long ones.** Total wall clock here is
dominated by queue time, not compute, so asking for what a phase actually needs
is the biggest lever available. A 24 h request queues behind almost everything;
a 7–9 h request starts far sooner. **Do not request 24 h for any of these jobs.**

Measured stage costs, from run 11's own `wall_s` (`wall(i) = 0.78·i + 6.6 min`)
scaled for the ~7× larger eval:

| stage | work | resumable? |
|---|---|---|
| 1a direction table, train | 1.5 h | no |
| 1b direction table, val | 0.2 h | no |
| 2 assign per-bin demos | seconds | no |
| 3a **collect train** | **5.0 h** | **no — indivisible** |
| 3b collect val | 0.3 h | no |
| 4 audit the base shard | ~5 min | no |
| **phases 1–4 total** | **7.1 h** | |
| 5 train, 25 iterations | 16.1 h | **yes, per iteration** |

Phase 5 is the only resumable stage (`state.json` records the last completed
iteration), so it is the only one worth splitting. Phases 1–4 go in one job
because stage 3a cannot be interrupted — see §5.7.

```bash
# ---- phase 1-4: tables, assignment, collection, audit --------------------
# 7.1 h of work. The script's own #SBATCH --time is already 09:00:00 (27%
# buffer), so no override is needed here.
JA=$(sbatch --parsable examples/slurm/regrasp_run18_all.sbatch)
echo "phases 1-4 = $JA"

# ---- phase 5: training, three chained passes of 7 h ---------------------
# 16.1 h of work against 21 h of capacity. Each pass re-runs the skip checks
# in seconds, then resumes training from state.json.
J=$JA
for i in 1 2 3; do
    J=$(sbatch --parsable --time=07:00:00 --dependency=afterany:$J \
            examples/slurm/regrasp_run18_all.sbatch)
    echo "training pass $i = $J"
done
```

**`afterany`, not `afterok`.** A pass that hits its wall clock exits non-zero,
which is expected; under `afterok` the next pass would never start.

A pass that finds all 25 iterations already done exits in seconds, so a spare
tail job costs a queue slot and nothing else. If training finishes early, cancel
what is left rather than leaving it queued:

```bash
squeue -u $USER -o "%.10i %.12j %.8T %.10l %R"
scancel <jobid>          # any still-pending pass that is no longer needed
```

**Sizing rule if you change anything.** Give a phase its measured work plus
~25%, rounded up to the half hour. Never round up to 24 h "to be safe" — the
queue penalty is larger than the risk, and every stage here is either idempotent
or `.partial`-guarded, so a kill costs re-work rather than corruption.

### 3.3 Monitor

```bash
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.10l %R"
tail -f slurm_logs/rg18_${J1}.out
```

Stage banners look like `=== [14:03:22] 1a  direction table (train)  ~1.5 h ===`.

### 3.4 Gates — check these, in this order

**Gate A — the direction table's per-bin histogram** (end of stage 1a, before
5 h of collection is spent against it). The table build prints
`anchor azimuth from: base` in its header; confirm that line appears. Then:

Stage 2 of the sbatch already prints this histogram, so read it from the job
log first:

```bash
sed -n '/2train  assign per-bin demos/,/^=== /p' slurm_logs/rg18_${J1}.out
```

To re-print it later, `--out` is a required argument even under `--dry-run`
(which writes nothing) — so point it at a throwaway path, never at the real pin
table:

```bash
python examples/assign_direction_demos.py \
    --table output/direction_table_train_bframe.json \
    --out /tmp/probe_bframe --dry-run
```

Expect **all six bins populated** and no bin near zero. Under `grasp_offset`,
`−z` and `−x` are real (they held 283 and 247 grasps at run 11's depth) — this
is *not* `approach_axis`, where they are empty. **A bin at or near zero is a
stop:** report it and do not proceed, because `--drop-bins` is deliberately not
passed and an empty bin means the frame or the rule is wrong.

**Gate B — the base shard's keep-rate** (after stage 4). This is the number run
18 exists to move. The audit prints it; or measure directly:

```bash
python - <<'PY'
import h5py, numpy as np, sys
sys.path.insert(0,"."); sys.path.insert(0,"examples")
from handover_sim2real.regrasp_bc.dataset import episode_status, DROP_STATUSES
import os
p = os.path.expandvars("$REGRASP_DATA/bc_dataset/train_regrasp_bframe.h5")
f = h5py.File(p, "r"); n = k = mis = nr = dz = 0
for key in f:
    g = f[key]; n += 1
    st = episode_status(g)
    if st == "miscaptioned": mis += 1; continue
    if st == "no_reach":     nr  += 1; continue
    dw = np.asarray(g.attrs.get("d_grasp_world", [0,0,0]), float)
    if np.linalg.norm(dw) < 1e-6: dz += 1; continue
    k += 1
print(f"{p}\n  episodes {n}  KEPT {k} ({k/n:.1%})"
      f"  miscaptioned {mis/n:.1%}  no_reach {nr/n:.1%}  dir_zero {dz/n:.1%}")
PY
```

Interpretation:

| miscaptioned | meaning |
|---|---|
| **≤ ~3%** | the frame repair worked. Proceed. |
| **> 10%** | the table and the shard are in different frames — the exact run-16 failure. **Stop and report.** |

`dir_zero` around **20–25%** is expected and is the known unfixed cause (§0).
`dir_zero` near 0% means the freeze was applied after all — note it and proceed.

**Gate C — the first eval, iteration 0.** In
`$SCRATCH_ROOT/output/dagger_runs/regrasp_run18/dagger_log.csv`:

```bash
R=$SCRATCH_ROOT/output/dagger_runs/regrasp_run18
python - <<PY
import csv
r = list(csv.DictReader(open("$R/dagger_log.csv")))[0]
for k in ("iter","success_rate","success_rate_in_table","n_in_table",
          "dir_err_median","D_episodes"):
    print(f"  {k:24s} {r.get(k)}")
PY
```

`success_rate_in_table` at iteration 0 should be near **0.50–0.52** (run 11
scored 0.500, run 16 0.5207 — the base fit was never the problem). Much below
0.45 means something upstream is wrong; stop and report.

Note **`success_rate` and `success_rate_in_table` are different populations.**
`full_bin_coverage: true` means ~80% of eval episodes command a bin the scene
never demonstrates, and `success_rate` averages all of them.
**`success_rate_in_table` is the only one comparable with runs 1–15** and with
run 11's 0.6186.

**Gate D — the DAgger keep-rate, from iteration 2 onward.** `D_episodes` now
reports what the fit actually saw (it previously overstated it by 58% on run 16).
The per-iteration delta should be **~95–110 episodes** on ~225 collected. If it
is nearer 40, the frame repair did not take — stop and report.

```bash
python - <<PY
import csv
rows = list(csv.DictReader(open("$R/dagger_log.csv")))
prev = None
for r in rows:
    d = int(r["D_episodes"]) - prev if prev is not None else None
    print(f"  it {r['iter']:>2}  collected {r['episodes']:>4}  "
          f"D_episodes {r['D_episodes']:>5}  added {d if d is not None else '-'}")
    prev = int(r["D_episodes"])
PY
```

### 3.5 Plot and report

```bash
python examples/plot_regrasp_run.py $SCRATCH_ROOT/output/dagger_runs/regrasp_run18
```

Five PNGs land in the run dir. What to read, and in what order:

- **`curves_diag.png`**, bottom-right panel — `eval — demonstrated bins` (red
  dashed) is the curve to compare against run 11. `eval — all commanded bins`
  (green) is the diluted one.
- **`training_curve.png`** — two success-stage columns per bin: *ALL scenes* and
  *DEMONSTRATED scenes*. The second is the runs-1–15-comparable one.
- **`curves_regrasp.png`** — `success per commanded direction — DEMONSTRATED
  scenes` bottom-left; the *ALL scenes* panel bottom-right carries both series
  (solid = every scene, dotted ^ = demonstrated), and the gap between a bin's
  two lines is its generalisation cost.
- **`curves_regrasp2.png`** — same without the side panel.

Report back: best `success_rate_in_table` and its iteration, the final value,
`dir_err_median`, the six `succ_bin_*`, and the `D_episodes` per-iteration delta.
The bar to clear is **run 11's 0.6186 @ it22**.

---

## 4. RUN 17

Only after run 18 has produced a `success_rate_in_table` curve. Identical
procedure, three substitutions:

| | run 18 | run 17 |
|---|---|---|
| preflight | `preflight_regrasp.sh regrasp_run18` | `preflight_regrasp.sh regrasp_run17` |
| sbatch | `regrasp_run18_all.sbatch` | `regrasp_run17_all.sbatch` |
| artifact suffix | `_bframe` | `_loc` |
| log prefix | `slurm_logs/rg18_*` | `slurm_logs/rg17_*` |

```bash
bash examples/slurm/preflight_regrasp.sh regrasp_run17
JA=$(sbatch --parsable examples/slurm/regrasp_run17_all.sbatch)     # phases 1-4, 9 h
J=$JA
for i in 1 2 3; do
    J=$(sbatch --parsable --time=07:00:00 --dependency=afterany:$J \
            examples/slurm/regrasp_run17_all.sbatch)
done
```

Same phase costs — run 17's stages are the same work, only the `d_rule` differs.

### Two differences in the gates

**Gate A gains a null bin.** `location_extent` sends grasps whose eccentricity
is below `d_m_min` (0.15) to a **null** bin rather than an axis — they express
no location preference. `assign_direction_demos` reports these as
`n_null_bin`, counted separately from `n_short_offset` (same command as Gate A,
with `_loc` paths and its own throwaway `--out`). A moderate null count is
correct; the six axis bins should still all be populated.

**Gate B's `dir_zero` is more dangerous here, not less.** `location_extent`
shares the same `v = grasp_point − c`, so it inherits the chord collapse — but
where `grasp_offset` *drops* the episode, `location_extent` silently shrinks `m`
toward zero instead. A drifting centroid therefore produces a plausible wrong
magnitude rather than a visible loss. Record `dir_n_null` from the eval columns
every iteration and report if it climbs.

### Reading run 17

Compare against **run 18**, both on `success_rate_in_table`. Run 17 vs run 16 is
confounded — it differs by both the rule and the frame. If run 18 itself
underperforms run 11, say so and stop: `location_extent` cannot be evaluated
against a broken baseline.

---

## 5. Known failure modes

### 5.1 Exit code 6, empty `.err` file

`/home` filled. Nothing else produces this signature.

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
df -h $HOME | tail -1
```

Free space (§2), then resubmit — the sbatch is idempotent and will skip
completed stages.

### 5.2 `AssocMaxGRESPerJob` in `squeue`'s reason column

A **known false reason code** on DelftBlue, forum-confirmed. The job is queued
normally. Do not change the resource request in response to it. `--gpus-per-task`
is mandatory on this cluster and is already set.

### 5.3 `[cfg] SIM.anchor_hand_ref: 'base' but ... was built with 'wrist'`

`resolve_anchor_ref` doing its job — this is the guard added *because* run 16
had no such check. A stale table exists at the run's paths. Either the paths
were reused from another run, or a table build was interrupted between stages.

```bash
ls -la output/direction_table_*_bframe.json output/regrasp_pins_*_bframe.json
python -c "import json;print(json.load(open('output/regrasp_pins_train_bframe.json'))['_meta'])"
```

Fix by deleting the stale artifacts and letting the sbatch rebuild them — do
**not** edit the config to match the table, which would run the experiment the
table describes rather than the one intended:

```bash
# PROPOSE, do not run unprompted
rm output/direction_table_{train,val}_bframe.json \
   output/regrasp_pins_{train,val}_bframe.json \
   output/regrasp_pins_{train,val}_bframe_excluded.json \
   output/regrasp_demos_train_bframe_ok.json
rm $REGRASP_DATA/bc_dataset/{train,val}_regrasp_bframe.h5
```

### 5.4 A similar refusal naming `d_rule`, `d_point_depth`, `d_m_min` or `d_extent_pct`

`resolve_d_rule`, same class of problem, same fix. All of these are written into
the table's `_meta` and compared at load.

### 5.5 `OMG_PLANNER_DIR is not set`

Only collection and the table build need OMG. The sbatch exports it; an
interactive check does not. Export it (§1) and retry.

### 5.6 A stage re-runs that should have been skipped

Both collections and both table builds write to `<out>.partial` and rename only
on a zero exit, so a killed stage leaves the `.partial` behind and re-runs from
scratch next pass. That is correct. A leftover `.partial` is safe to delete and
is not read by anything:

```bash
ls -la output/*_bframe.json.partial $REGRASP_DATA/bc_dataset/*_bframe.h5.partial 2>/dev/null
```

### 5.7 A collection was killed mid-flight

Stage 3a takes 5 h and **cannot resume** — the collector replays OMG plans scene
by scene and keeps no cursor. A killed collection re-runs in full. This is why
phases 1–4 share one 9 h job rather than being split further: a 4 h request would
kill stage 3a every time and never make progress.

If the run keeps dying inside stage 3a, check the wall clock actually granted:

```bash
sacct -j <jobid> --format=JobID,JobName%14,State,ExitCode,Timelimit,Elapsed
```

### 5.8 Iteration count stalls across a chained pass

Check that pass 2 actually started and that `state.json` advanced:

```bash
python -c "import json;d=json.load(open('$R/state.json'));print(len(d['iterations']),'iterations recorded')"
sacct -u $USER --starttime today --format=JobID,JobName%14,State,ExitCode,Elapsed
```

---

## 6. Quick reference

```bash
# environment
cd ~/h2r/handover-sim2real
source ~/anaconda3/etc/profile.d/conda.sh && conda activate pch2r_dev
export GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner
export SCRATCH_ROOT=$HOME/h2r-runs REGRASP_DATA=$SCRATCH_ROOT/output

# run 18 — phases 1-4 (9 h, the script default), then 3 training passes of 7 h
bash examples/slurm/preflight_regrasp.sh regrasp_run18
J=$(sbatch --parsable examples/slurm/regrasp_run18_all.sbatch)
for i in 1 2 3; do
    J=$(sbatch --parsable --time=07:00:00 --dependency=afterany:$J \
            examples/slurm/regrasp_run18_all.sbatch)
done

# run 17, after run 18 has results — identical shape
bash examples/slurm/preflight_regrasp.sh regrasp_run17
J=$(sbatch --parsable examples/slurm/regrasp_run17_all.sbatch)
for i in 1 2 3; do
    J=$(sbatch --parsable --time=07:00:00 --dependency=afterany:$J \
            examples/slurm/regrasp_run17_all.sbatch)
done

# progress, plots
squeue -u $USER
python examples/plot_regrasp_run.py $SCRATCH_ROOT/output/dagger_runs/regrasp_run18
```

**The number that matters is `success_rate_in_table`, not `success_rate`.**
Run 11's 0.6186 @ it22 is the bar.

---

## 7. Scoring a finished run on the held-out test split

Separate from the two runs above, and cheap — no collection, no training, just
rollouts of checkpoints that already exist. `examples/eval_regrasp_testset.py`
scores **every iteration** of a finished run on a whole split and writes
`<split>_log.csv` plus `<split>_eval.png`.

```bash
# every iteration of run 9 on the held-out s0 TEST split (130 scenes)
python examples/eval_regrasp_testset.py \
    --run-dir output/dagger_runs/regrasp_run9 \
    --split test --ckpt last --iters all
```

Add `--chained` to also run true chained retry (the rewind machinery, attempt 2
starting where attempt 1 left the arm) — it roughly doubles the runtime and adds
the dashed `CHAINED @ k` curves to the retry panel.

Budget ~130 scenes x live bins x iterations rollouts. For run 9 (26 iterations,
4 live bins) that is ~13.5k episodes; at run 11's eval rate expect **6-9 h**, so
request **`--time=09:00:00`** and use `--iters 0,5,10,15,20,25` for a first pass
if you want a curve shape in ~2 h. The script resumes: rows already in the CSV
are skipped unless `--force`.

`test_eval.png` carries, per iteration:

- rows 0-1: the six `curves_regrasp` panels, drawn by the same
  `plot_regrasp_run.draw_conditioning` the training figure uses
- rows 2+: one row per bin — success stages, chance vs conversion, approach
  error to the grasp

**Run 9 predates `EVAL.full_bin_coverage`**, so its `succ_bin_all_*` columns are
empty and the *ALL scenes* panel correctly renders the "coverage was off" note
instead of a misleading blank grid. The *DEMONSTRATED scenes* panel is the real
per-bin curve for that run.
