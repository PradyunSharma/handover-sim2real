# Runbook — Regrasp runs 9 and 10 on DelftBlue

Everything needed to execute these two runs, written to be self-contained: the
reader is assumed to have the repo and cluster access but **not** the
conversation that designed them. Rationale and measurements live in each
config's header and in [`docs/run_index.md`](run_index.md); this file is the
sequence of actions, the gates between them, and what to do when a gate fails.

**Standing rule for whoever executes this: do not `git commit` or `git add`
anything unprompted. Propose the command and let the user run it.**

---

## 0. What these two runs are

Both descend from `regrasp_run2`, the best run in the phase (`success 0.563` at
iteration 13). Each changes one idea and keeps everything else byte-identical:
one demonstration per reachable bin, `train_from_scratch: true`, `EVAL.every: 1`,
25 iterations, β 0.9 → 0.75, the aux head, DART on both bands.

### Run 9 — train on the grasp's own direction, deploy on the bin's centroid

Every Regrasp episode carries two unit vectors: `d_world` (what the episode was
*commanded*) and `d_grasp_world` = `−R_grasp[:,2]` (the approach axis the expert
*actually flew*). Runs 1–8 made those the same vector. Run 9 separates them:

- **training label** = the grasp's own axis. No quantisation, no perturbation.
- **deployment command** = the bin's *empirical centroid*, the unit mean of every
  `d_anchor` assigned to that bin over the pruned pin table. Six vectors, no
  grasp needed, so it is producible on the robot exactly as `BINS` is.

Measured over the 1576 surviving (scene, bin) pairs, the residual angle between
training label and deployment command:

| train label → deploy command | median | p90 | max |
|---|---|---|---|
| grasp axis → bin axis *(run 1)* | 18.45° | 38.36° | 44.98° |
| grasp axis → **bin centroid** *(run 9)* | **16.20°** | **32.68°** | 55.08° |

So the centroid buys 2.25° of median. It does *not* remove the skew — read run 9
as **run 1 done properly**, not as a run-2 variant. What it really moves is where
the angular budget sits: runs 2–8 have zero skew and ~18.45° of label noise
(told a sector, shown one grasp inside it); run 9 has zero label noise and
~16.20° of skew.

**Run 9 needs no new data.** Run 2's tables and shards are the inputs verbatim.

### Run 10 — `d` from the grasp's position, not its orientation

| | definition | what it says |
|---|---|---|
| `approach_axis` (runs 1–9) | `d = −R_grasp[:,2]` | which side the gripper **comes from** — the grasp's **orientation** |
| `grasp_offset` (run 10) | `d = normalize(T_grasp · [0,0,0.1122] − c)` | which part of the object the fingers **close on** — the grasp's **position** relative to the object |

`c` is the observed object point-cloud centroid (the anchor frame's origin);
0.1122 m is the fingertip end of the Panda pads.

**This is not a refinement, it is a different question, and it unlocks the two
dead bins.** Over the 3477 goal-set grasps in `direction_table_train.json`:

| definition | `+x` | `−x` | `+y` | `−y` | `+z` | `−z` | within 45° |
|---|---|---|---|---|---|---|---|
| `approach_axis` | 1162 | 20 | 822 | 684 | 789 | **0** | 100.0% |
| `grasp_offset` | 714 | **392** | 637 | 450 | 418 | **526** | 90.2% |

Scenes reaching each bin: 490/12/365/324/415/**0** → 248/**191**/241/205/142/**235**.

`−z` goes from dead to the third-largest bin and `−x` from 12 scenes to 191 —
physically right, because you cannot *approach* a held object from beneath but
you can close your fingers on its underside. Consequences: six live bins not
four, `chained_retry_at_k` saturates at 6, chance level for `bin_diag_rate`
moves 1/4 → 1/6, `max_grasps` becomes 6.

**Run 10 needs the whole upstream chain rebuilt** and it cannot be done offline —
see §4.

---

## 1. Preconditions

### 1.1 The code must be the version that supports these runs

These runs depend on code added alongside them. Verify it arrived:

```bash
cd ~/handover-sim2real          # adjust to the actual checkout path
python - <<'PY'
import importlib, pathlib, sys
ok = True
from handover_sim2real.regrasp import directions as D
for name in ("DirectionRule", "grasp_direction", "command_direction",
             "centroid_axes", "FINGERTIP_DEPTH", "D_RULES"):
    if not hasattr(D, name):
        print(f"MISSING directions.{name}"); ok = False
from handover_sim2real.regrasp import setup as S
for name in ("resolve_command_axes", "resolve_d_rule", "COMMAND_MODES"):
    if not hasattr(S, name):
        print(f"MISSING setup.{name}"); ok = False
from handover_sim2real.regrasp.grasp_pin import GraspPinTable
if not hasattr(GraspPinTable, "bin_centroids"):
    print("MISSING GraspPinTable.bin_centroids"); ok = False
from handover_sim2real.regrasp.collector import CollectParams
from handover_sim2real.regrasp.evaluator import EvalParams
for C in (CollectParams, EvalParams):
    for f in ("command_axes", "d_rule"):
        if not hasattr(C(), f):
            print(f"MISSING {C.__name__}.{f}"); ok = False
import inspect
from handover_sim2real.regrasp_bc.dataset import BCDataset
if "d_source" not in inspect.signature(BCDataset.__init__).parameters:
    print("MISSING BCDataset(d_source=)"); ok = False
for p in ("examples/configs/regrasp_run9.yaml",
          "examples/configs/bc_regrasp_run9.yaml",
          "examples/configs/regrasp_run10.yaml",
          "examples/configs/bc_regrasp_run4.yaml",
          "examples/check_regrasp_inputs.py"):
    if not pathlib.Path(p).exists():
        print(f"MISSING {p}"); ok = False
print("CODE OK" if ok else "CODE INCOMPLETE — pull the commit that adds these")
sys.exit(0 if ok else 1)
PY
```

If this fails, stop. Nothing below will behave correctly.

### 1.2 Environment for the interactive steps

The sbatch scripts set this up themselves, but the assignment, audit and check
steps below run on the login node and need it:

```bash
module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pch2r_dev
cd ~/handover-sim2real
export GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet"
```

### 1.3 Where the data goes

All run output and all HDF5 shards go under `$HOME`, per the user's requirement.
The small JSON tables stay in the repo (they are version-controlled inputs).

```bash
export RUNS=$HOME/h2r-runs
```

`SCRATCH_ROOT=$RUNS` in the sbatch calls sets both `OUT_ROOT`
(`$RUNS/output/dagger_runs`) and `REGRASP_DATA` (`$RUNS/output`).

> **`/home` on DelftBlue is a hard 30 GB quota that fills SILENTLY.** Python
> cannot write a traceback to a full disk, so the symptom is exit code 6 and a
> log that just stops. Check headroom before every stage that writes:
> `du -sh $HOME`. Budgets are in §6.

---

## 2. Run 9 — stage the base shards

Run 9 reuses run 2's shards. They live on `/scratch` and must be copied to home.

**Check first — do not copy blindly.**

```bash
ls -la /scratch/$USER/handover-sim2real/output/bc_dataset/{train,val}_regrasp.h5 \
       $RUNS/output/bc_dataset/{train,val}_regrasp.h5 2>&1
```

As of 2026-08-28 the `/scratch` copies are `train_regrasp.h5` 652,190,252 bytes
and `val_regrasp.h5` 27,006,213 bytes, and home holds neither — so the copy is
needed. If home already has them, check whether they are the *same* files before
re-copying:

```bash
rsync -ahcn --itemize-changes \
  /scratch/$USER/handover-sim2real/output/bc_dataset/{train,val}_regrasp.h5 \
  $RUNS/output/bc_dataset/
```

`-n` is dry-run and transfers nothing; `-c` checksums both sides. **No output
means identical — skip the copy.** `-c` is not optional here: a file copied
while a job was still writing arrives with correct size and mtime but
zero-filled data blocks, and rsync's default quick check passes it. That has
already cost this project a day.

**Copy:**

```bash
du -sh $HOME                                    # need ~6.5 GB free
mkdir -p $RUNS/output/bc_dataset
rsync -ahc --progress \
  /scratch/$USER/handover-sim2real/output/bc_dataset/{train,val}_regrasp.h5 \
  $RUNS/output/bc_dataset/
```

**Gate — the copies must be readable, not merely the right size:**

```bash
python - "$RUNS/output/bc_dataset/train_regrasp.h5" \
         "$RUNS/output/bc_dataset/val_regrasp.h5" <<'PY'
import h5py, sys, numpy as np
for p in sys.argv[1:]:
    try:
        with h5py.File(p, "r") as f:
            ks = [k for k in f if k.startswith("episode_")]
            dg = sum(1 for k in ks if "d_grasp_world" in f[k].attrs
                     and np.linalg.norm(f[k].attrs["d_grasp_world"]) > 0.5)
            print(f"OK   {p}: {len(ks)} episodes, d_grasp_world usable on {dg}")
    except Exception as e:
        print(f"BAD  {p}: {type(e).__name__}: {e}")
PY
```

**Expect** 1596 episodes on train and 69 on val, with `d_grasp_world` usable on
**all** of them. `d_grasp_world` is the attr run 9 trains on — a non-zero count
on every episode is exactly why run 9 needs no re-collection. If the counts
differ, report them before submitting rather than proceeding.

> Do **not** copy these from a laptop. The laptop's copies are a separate
> collection (649,926,542 / 27,140,279 bytes). Run 2 trained on the `/scratch`
> copies, and run 9 must reuse the same data for the comparison to mean anything.

---

## 3. Run 9 — check and submit

```bash
REGRASP_DATA=$RUNS/output python examples/check_regrasp_inputs.py regrasp_run9
```

**Expect `All inputs present.` and, in the report:**

- `d_rule: approach_axis   (table: approach_axis)`
- `TRAIN.train_cfg  examples/configs/bc_regrasp_run9.yaml`, `d_noise_deg=0.0`
- `command: deploy=bin_centroid  label=d_grasp_world   (DIFFERENT vectors ...)`
- `-> beta 0.9 -> 0.75, 25 iters, m=400, scratch (FTL)`

If `command:` reads anything else, the wrong learner config is wired up. If it
reads `deploy=bin_centroid label=d_world`, `train_regrasp.py` will refuse to
start — that combination captions the base and DAgger halves of the aggregate
under different rules.

**Submit:**

```bash
sbatch --time=16:00:00 \
  --export=ALL,RUN=regrasp_run9,CFG=examples/configs/regrasp_run9.yaml,SCRATCH_ROOT=$RUNS \
  examples/slurm/train_regrasp.sbatch
```

16 h rather than 24: the estimate is ~13.5 h (§6) and a shorter request
backfills far better — measured on this account, priority fell 46× after four
GPU allocations in one afternoon. The run is resumable, so a timeout costs only
the queue wait; resubmit the identical command and completed iterations are
skipped.

**Early sanity check, once the job starts:**

```bash
tail -f slurm_logs/regrasp_<jobid>.out
```

Look for `[command] deploy on the BIN CENTROID, not the bin axis; offset from
each axis (deg): +x 7.7  +y 15.4  -y 14.1  +z 8.0`. Those four numbers are the
run's premise. If they are all ~0.0, the centroid resolved to `BINS` and the run
is silently a repeat of run 7.

Also expect `[command] deploy=bin_centroid   label=d_grasp_world   (DIFFERENT
vectors: ...)` and `[direction] d_grasp_world resolved for N episodes`.

---

## 4. Run 10 — rebuild the whole upstream chain

**None of run 2's data is reusable and it cannot be relabelled offline.**
`d_rule` is decided in `build_direction_table.py` — it is what fills `d_anchor`,
and every later stage reads that field. The existing table stores only the
goal-set members that survived the 45° cutoff under the *old* rule: 3477 of
32107, **10.8%**. Rebuilding from it would be a badly biased subsample.

`build_regrasp_context` refuses a table whose `_meta.d_rule` disagrees with the
config, because the failure it prevents is silent: bins populated by one rule
with `bin_realized` measured by the other disagree on most episodes, the
dataset's miscaption filter empties the aggregate, and the symptom is "the
collection produced almost nothing".

### 4.1 Direction tables (~30–60 min each, GPU; run both in parallel)

```bash
sbatch --export=ALL,SPLIT=train,OUT=output/direction_table_train_off.json,\
EXTRA="--members-per-bin 5 --d-rule grasp_offset --d-min-offset 0.02" \
  examples/slurm/build_direction_table.sbatch

sbatch --export=ALL,SPLIT=val,OUT=output/direction_table_val_off.json,\
EXTRA="--members-per-bin 5 --d-rule grasp_offset --d-min-offset 0.02" \
  examples/slurm/build_direction_table.sbatch
```

`--d-min-offset 0.02` drops grasps whose fingertip point lands within 2 cm of
the centroid, where the direction is centroid noise rather than geometry. The
fingertip sits a median 3.85 cm out and **14.4% of grasps are inside 2 cm**, so
this filter is doing real work; the count lands in `_meta.n_short_offset`.

### 4.2 GATE — read the histogram before going further

This is the first honest count over the **full** goal sets (~32k grasps), not
the 3477 recorded members the §0 table was measured on. One decision depends on
it: which bins, if any, to drop.

```bash
python - output/direction_table_train_off.json <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["_meta"]
print("d_rule          :", m.get("d_rule", "approach_axis (key absent)"),
      m.get("d_point_depth"), m.get("d_min_offset"))
print("scenes ok       :", m.get("n_ok"), "of", m.get("n_scenes"))
print("dropped (short) :", m.get("n_short_offset", 0))
n_ok = max(m.get("n_ok", 1), 1)
print(f"\n{'bin':<18} {'grasps':>8} {'scenes':>8} {'% of scenes':>12}")
for i, nm in enumerate(m["bin_names"]):
    g, s = m["goal_set_bin_histogram"][i], m["scenes_with_bin"][i]
    print(f"{nm:<18} {g:>8} {s:>8} {100*s/n_ok:>11.1f}%")
print("\nmean bins/scene : %.2f" % (sum(m["scenes_with_bin"]) / n_ok))
thin = [m["bin_names"][i] for i, s in enumerate(m["scenes_with_bin"])
        if s < 0.05 * n_ok]
print("THIN/DEAD bins  :", thin or "none — keep all six")
print("=> --drop-bins='" + ",".join(thin) + "'")
PY
```

**Decision — which bins to drop.** A bin reachable by only a handful of scenes is
not a retry hypothesis; run 2 dropped `−z` (0 scenes) and `−x` (12 of 623, 1.9%)
for exactly that reason. The 5% threshold above is a starting point, not a rule —
look at the actual numbers. The expectation for `grasp_offset` is that **nothing
needs dropping**: all six bins should be well populated, with `−z` and `−x`
substantial.

**If `−z` comes back near zero on the full goal sets, run 10's premise did not
survive contact with the data.** Report that before collecting — §4.4 is the
expensive step, and a `grasp_offset` run with four live bins is a much weaker
experiment than the one this was designed as.

**Escape hatch if `n_short_offset` is alarming** (well above ~15% of goal-set
members dropped): `SIM.d_point_depth: 0.0` moves the point to the palm origin,
a median 12.48 cm from the centroid (minimum 4.55 — never degenerate) and a
median 14.49° from `−R[:,2]`. Still position-derived and orientation-independent,
but it answers the `approach_axis` question, so it would **not** unlock `−z`.
Changing it means rebuilding from §4.1.

> Do **not** try to set `episodes_per_iter` from this script. The direction
> table's `scenes_with_bin` counts every bin with at least one scene, including
> ones you are about to drop — on run 2's table that reads 5 live bins when the
> run actually had 4. `max_grasps` comes from the *pin table*, after §4.3.

### 4.3 Assignment (seconds, no simulator)

```bash
python examples/assign_direction_demos.py \
    --table output/direction_table_train_off.json \
    --out   output/regrasp_pins_train_off --mode per-bin

python examples/assign_direction_demos.py \
    --table output/direction_table_val_off.json \
    --out   output/regrasp_pins_val_off --mode per-bin
```

> **No `--drop-bins`.** Run 2 passed `--drop-bins='-z_beneath,-x_over_fingers'`
> because those bins were empty under `approach_axis`. Under `grasp_offset` they
> hold 526 and 392 grasps, and dropping them discards roughly a third of the data
> this run exists to use.

If §4.2 said some bin must be dropped, add `--drop-bins='<names>'` to the *train*
command using the string it printed, and the same to val.

Writes `<out>.json` and `<out>_excluded.json`. The direction table's `_meta`
(including `d_rule`) rides forward into the pin table, which is what the
downstream cross-check reads.

**Now set `episodes_per_iter`.** The pin table is the authoritative source for
`max_grasps` — it reflects the drop-bins decision, which the direction table does
not:

```bash
python - output/regrasp_pins_train_off.json <<'PY'
import sys, collections
from handover_sim2real.regrasp.grasp_pin import GraspPinTable
from handover_sim2real.regrasp import directions as D
t = GraspPinTable(sys.argv[1])
n = collections.Counter(int(g["bin"]) for gs in t.entries.values() for g in gs)
print("d_rule        :", (t.meta or {}).get("d_rule", "approach_axis (default)"))
print("scenes        :", len(t.entries), " demos:", sum(n.values()))
print("demos per bin :", {D.BIN_SHORT[b]: n[b] for b in sorted(n)})
print("max_grasps    :", t.max_grasps,
      "  (num_grasps, a MIN, reads", t.num_grasps, "- never use it)")
print("=> DAGGER.episodes_per_iter = 100 x max_grasps =", 100 * t.max_grasps)
PY
```

`regrasp_run10.yaml` currently says `600`, assuming six. **If this prints a
different number, edit `DAGGER.episodes_per_iter` to match.** Leaving it wrong
does not error — `sample_pairs` draws `m // max_grasps` scenes, so it silently
draws the wrong number per iteration. On run 2's table this script prints 400,
which is exactly what that config carries; that is the check that it is right.

The same number is *s*'s denominator for the budgets in §6: multiply the demos
count by `100/len(scenes)` to get episodes per iteration, or just read the
`[regrasp] ... N collectable (scene, direction) pairs` line from the job log.

### 4.4 Base collection (the long pole, ~2.5–4 h train)

```bash
du -sh $HOME                                    # need ~8–11 GB free for run 10

sbatch --export=ALL,SPLIT=train,SCRATCH_ROOT=$RUNS,\
PIN=output/regrasp_pins_train_off.json,\
OUT=$RUNS/output/bc_dataset/train_regrasp_off.h5 \
  examples/slurm/collect_regrasp_demos.sbatch

sbatch --export=ALL,SPLIT=val,SCRATCH_ROOT=$RUNS,\
PIN=output/regrasp_pins_val_off.json,\
OUT=$RUNS/output/bc_dataset/val_regrasp_off.h5 \
  examples/slurm/collect_regrasp_demos.sbatch
```

Measured collection rate: **5.57 s/episode** serial on a V100 (3.62 s/episode on
a laptop). The job prints `d rule: grasp_offset (...)  (from the pin table's
_meta)` at startup — check that line; if it says `approach_axis`, the pin table
is the wrong one.

### 4.5 Audit

```bash
python examples/audit_regrasp_demos.py \
    --demos $RUNS/output/bc_dataset/train_regrasp_off.h5 \
    --write-ok output/regrasp_demos_train_off_ok.json
```

The audit exits 1 on a schema failure, and that is the right moment to stop —
training on a shard the audit rejects wastes the whole allocation.

Run 2's equivalent kept 1576 of 1596 (scene, bin) pairs, i.e. ~1.2% lost to pin
failures. **A much larger loss here is a signal, not noise**: it would mean
`bin_realized` disagrees with `bin_assigned` systematically, which points at a
`d_rule` mismatch between the table and the collector rather than at bad luck.

---

## 5. Run 10 — check and submit

```bash
REGRASP_DATA=$RUNS/output python examples/check_regrasp_inputs.py regrasp_run10
```

**Expect `All inputs present.` and, in the report:**

- `d_rule: grasp_offset ... depth 11.22 cm, min offset 2.0 cm` with
  `(table: grasp_offset)` — **not** `** MISMATCH **`
- `TRAIN.train_cfg  examples/configs/bc_regrasp_run4.yaml`, `d_noise_deg=0.0`
- `command: deploy=bin_axis  label=d_world   (SAME vector)`
- `-> beta 0.9 -> 0.75, 25 iters, m=<100 x max_grasps>, scratch (FTL)`

**Submit:**

```bash
sbatch --time=24:00:00 \
  --export=ALL,RUN=regrasp_run10,CFG=examples/configs/regrasp_run10.yaml,SCRATCH_ROOT=$RUNS \
  examples/slurm/train_regrasp.sbatch
```

24 h here rather than 16, because the episode count is not known in advance and
the run scales with it (§6).

**Early sanity check:** the log should show `[d_rule] grasp_offset (d = centroid
-> gripper point at 11.22 cm, min 2.0 cm)` and
`[regrasp] up to 6 direction(s) per scene -> N collectable (scene, direction)
pairs`.

---

## 6. Budgets

Timing is a fit to run 2's own `wall_s` column over its 20 iterations, dropping
two resumed ones that logged short:

> **wall(i) = 1.66·i + 8.0 minutes** → iteration 1 takes 10 min, iteration 25
> takes 50, and 25 iterations total **12.3 h**. Add ~1.1 h for the base fit.

The linear growth is the Follow-The-Leader refit over an aggregate that grows
~7.7k steps per iteration.

Disk is measured from run 2's 4.4 GB on disk: **179 MB per iteration** for the
DAgger shard, **46 MB** for the two checkpoints, plus 46 MB for the `best/` and
`last/` exports — 225 MB per iteration.

| | run 9 | run 10 |
|---|---|---|
| episodes/iteration | ~259 (measured) | 100 × mean bins/scene — **unknown until §4.2** |
| 25 DAgger iterations | 12.3 h | 12.3 h × *s* |
| base fit | ~1.1 h | ~1.1 h × *s* |
| **training total** | **~13.5 h** | **~13.5 h × *s*** |
| upstream (§4) | none | ~4–5 h |
| run directory | 5.7 GB | 5.7 GB × *s* |
| base shards | 0.68 GB (copied) | ~0.7 GB × *s* |
| **disk total** | **~6.4 GB** | **~6.4 GB × *s*** |

where ***s* = run 10's episodes per iteration ÷ 259**. The 10.8% subsample
suggested ~2.0 bins/scene but is biased low; at 3.5 bins/scene (*s* ≈ 1.35) run
10 is **~18 h and ~8.6 GB**, at 4.5 (*s* ≈ 1.7) it is **~23 h and ~11 GB**.
Compute *s* properly from §4.2's output before committing to a `--time`.

**Both runs together are 13–17 GB of a 30 GB hard quota.** If home is not nearly
empty, run them sequentially and harvest the first before starting the second:

```bash
python examples/harvest_run.py --run-dir $RUNS/output/dagger_runs/regrasp_run9
rm -rf $RUNS/output/dagger_runs/regrasp_run9/data     # ~4.5 GB of DAgger shards
```

The shards are the bulk and are reproducible from the tables plus the seed; the
checkpoints, logs and figures are what needs keeping.

---

## 7. Monitoring and post-run

```bash
python examples/status_regrasp_run.py $RUNS/output/dagger_runs/regrasp_run9
bash examples/watch_regrasp.sh -f slurm_logs/regrasp_<jobid>.out

# figures (safe while running — read-only)
python examples/plot_regrasp_run.py  $RUNS/output/dagger_runs/regrasp_run9
python examples/plot_regrasp_fits.py $RUNS/output/dagger_runs/regrasp_run9
```

Held-out numbers are not in these jobs — `EVAL.holdout: false` means the in-loop
scenes are also collected on, so the curves are train-set rates by design. Score
the s0 test split afterwards, which needs test-split tables built the same way
(and for run 10, built with `--d-rule grasp_offset`):

```bash
python examples/eval_regrasp_testset.py \
    --run-dir $RUNS/output/dagger_runs/regrasp_run9 --chained
```

### What to read first

**Run 9.** `dir_err` should *improve* if precise labels are worth more than an
exactly matched command. `dir_err` at run 7's level with `bin_diag_rate_b*`
falling means the skew dominates and run 2's argument wins — the response is
then to re-collect with `--command bin_centroid` so training and deployment
agree *on the centroid*, a third arm this run's data makes cheap to justify.

**Run 10.** `bin_diag_rate` should rise against run 7's 0.903 and `dir_err`
should fall, because "close on this part of the object" names a target visible
in the policy's own point cloud. `bin_diag_rate` at chance means the per-point
channels cannot express a position-derived command. High `bin_diag_rate` with
low `success_rate` in the `−z`/`−x` rows means the policy obediently drives at
an underside it cannot grasp from — the feasibility mask's problem, not the
conditioning's.

**Note the chance level moves for run 10**: 1/6 rather than 1/4, so
`bin_diag_rate` is not comparable across the change without saying which chance
level it is against.

### Comparison runs, honestly

Neither run has a single-change reference.

- Run 9 vs run 2 = two changes (noise off, label/command rule).
  Run 9 vs run 7 = two changes (β 0.5→0.75, same rule).
- Run 10 vs run 2 = two changes (noise off, `d_rule`).
  Run 10 vs run 7 = two changes (β, `d_rule`).

What makes that tolerable is that run 7 already isolated `d_noise_deg` exactly:
it moved the base fit (0.143 → 0.370) and **nothing after it** — by iteration 7
run 2 and run 7 are indistinguishable. So from iteration 7 on, both new runs
against run 2 read as one change. **The base fits do not: compare them with run
7's 0.370, not run 2's 0.143.**

**The noise floor is 0.088.** Run 8's iteration 0 is configured identically to
run 7's and scored 0.282 against 0.370. Nothing smaller than that gap is
readable anywhere in this table.

---

## 8. Things that will go wrong, and what they mean

| symptom | cause | fix |
|---|---|---|
| job dies, exit code 6, log truncated, no traceback | `/home` quota full | `du -sh $HOME`, harvest and delete a run's `data/` |
| `SystemExit: SIM.d_rule: grasp_offset but ... was built under 'approach_axis'` | run 10 pointed at run 2's tables | rebuild from §4.1 |
| `SystemExit: SIM.command_deploy: ... with DATA.d_source: d_world` | run 9's learner config not wired up | `TRAIN.train_cfg` must be `bc_regrasp_run9.yaml` |
| collection "produced almost nothing"; audit drops most pairs | `d_rule` mismatch between table and collector | check the collector's startup `d rule:` line |
| `[command] ... offset from each axis (deg): +x 0.0 +y 0.0 ...` | centroid resolved to `BINS`; run 9 is silently run 7 | pin table has no `d_anchor` — rebuild the assignment |
| `_csv.Error: line contains NUL` / `BadZipFile` / `Cannot load file containing pickled data` when plotting | files rsync'd mid-write | re-fetch with `rsync -c`; the plotters tolerate and skip |
| `AssocMaxGRESPerJob` | known false reason code on DelftBlue | forum-confirmed; ignore, the job is queued |

**Do not** change the resource block in any `.sbatch`. It is copied verbatim
from a block that passes `sbatch --test-only`; deviating from it produces
`AssocMaxGRESPerJob` even for requests well inside the association limit.
`--gpus-per-task` is mandatory (DelftBlue rejects `--gres`/`--gpus`).

---

## 9. Quick reference

```bash
# environment
conda activate pch2r_dev && cd ~/handover-sim2real
export GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet"
export RUNS=$HOME/h2r-runs

# run 9  (data already exists; just stage the shards)
rsync -ahc /scratch/$USER/handover-sim2real/output/bc_dataset/{train,val}_regrasp.h5 \
           $RUNS/output/bc_dataset/
REGRASP_DATA=$RUNS/output python examples/check_regrasp_inputs.py regrasp_run9
sbatch --time=16:00:00 --export=ALL,RUN=regrasp_run9,\
CFG=examples/configs/regrasp_run9.yaml,SCRATCH_ROOT=$RUNS \
  examples/slurm/train_regrasp.sbatch

# run 10  (four build stages first — see §4, and STOP at the §4.2 gate)
sbatch --export=ALL,SPLIT=train,OUT=output/direction_table_train_off.json,\
EXTRA="--members-per-bin 5 --d-rule grasp_offset --d-min-offset 0.02" \
  examples/slurm/build_direction_table.sbatch
# ... §4.2 gate, §4.3 assign, §4.4 collect, §4.5 audit ...
REGRASP_DATA=$RUNS/output python examples/check_regrasp_inputs.py regrasp_run10
sbatch --time=24:00:00 --export=ALL,RUN=regrasp_run10,\
CFG=examples/configs/regrasp_run10.yaml,SCRATCH_ROOT=$RUNS \
  examples/slurm/train_regrasp.sbatch
```
