# Runbook — Regrasp run 10, training only

Supersedes §4–§5 of `runbook_run9_run10.md` for run 10. The upstream chain there
— direction table, assignment, base collection, audit — is **done**; this book
starts at the point where those artifacts exist and only the DAgger run is left.

One thing changed under run 10 since that book was written, and it changes what
the run trains on, so it gets a gate of its own before submission.

---

## 0. What is different from the old runbook

`SIM.reach_filter` now exists and **defaults to on**. It drops every
`(scene, bin)` pair whose base demonstration never reached the grasp it was
aiming at — terminal EE pose outside `close_pos_thresh` / `close_rot_thresh` of
`grasp_pose_world`.

This is not the same question `demo_ok_table` asks. That one asks whether a
demonstration's **caption** is honest (`bin_realized == bin_assigned`, pin
landed) and passes ~99%. This asks whether the expert **arrived**, and on the
run-2 shard only 69.9% did. The other 30% stop a mean 108 mm short, run 14 steps
against 21, and end still commanding "keep approaching" because `c_env_done`
truncated them. They were in `D`, and their pairs were re-collected and re-scored
every iteration.

Measured on the run-2 shard (`train_regrasp.h5`, four live bins):

```
D                1596 ep / 30028 steps   ->   1098 ep / 23048 steps
pin table         617 scenes / 1596 pairs ->   558 scenes / 1097 pairs
pairable (>= 2)   490 scenes             ->   340 scenes
```

**Nothing upstream needs rebuilding.** The filter is derived at train time from
`TRAIN.base_train_h5`; it is not a file. One predicate
(`regrasp_bc/dataset.py:episode_status`) drives five consumers: the pin-table
prune (which covers DAgger collection *and* in-loop eval), `BCDataset`,
`compute_normalization_stats`, and the `D_episodes`/`D_steps` log columns.

**Accepted consequence.** Run 10 was configured as "run 2 with `d_rule`
changed". It is now that plus the reach filter, so a difference against run 2 is
not attributable to `d_rule` alone. That was decided deliberately — better data
over a cleaner comparison — and it is recorded in `run_index.md` so nobody later
reads run 10 as a clean `grasp_offset` test.

---

## 1. Preconditions

### 1.1 The code must include the reach filter

```bash
cd ~/handover-sim2real
python - <<'PY'
import inspect, pathlib, sys
ok = True
from handover_sim2real.regrasp import reach as R
for n in ("reached", "reach_ok_pairs", "terminal_pose_error",
          "DEFAULT_POS_THRESH", "DEFAULT_ROT_THRESH"):
    if not hasattr(R, n):
        print(f"MISSING reach.{n}"); ok = False
from handover_sim2real.regrasp_bc import dataset as DS
for n in ("episode_status", "DROP_STATUSES"):
    if not hasattr(DS, n):
        print(f"MISSING dataset.{n}"); ok = False
for n in ("reach_filter", "reach_pos_thresh", "reach_rot_thresh"):
    if n not in inspect.signature(DS.BCDataset.__init__).parameters:
        print(f"MISSING BCDataset({n}=)"); ok = False
    if n not in inspect.signature(DS.compute_normalization_stats).parameters:
        print(f"MISSING compute_normalization_stats({n}=)"); ok = False
from handover_sim2real.regrasp.grasp_pin import GraspPinTable
if not hasattr(GraspPinTable, "keep_only"):
    print("MISSING GraspPinTable.keep_only"); ok = False
for p in ("examples/build_demo_table.py", "examples/analyze_demo_bins.py"):
    if not pathlib.Path(p).exists():
        print(f"MISSING {p}"); ok = False
print("CODE OK" if ok else "CODE INCOMPLETE — pull the commit that adds reach.py")
sys.exit(0 if ok else 1)
PY
```

Then run §1.1 of `runbook_run9_run10.md` as well — that check covers `d_rule`,
`command_axes` and the run-10 configs, and none of it is superseded.

### 1.2 Environment and paths

Unchanged from `runbook_run9_run10.md` §1.2–1.3:

```bash
module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pch2r_dev
cd ~/handover-sim2real
export GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet"
export RUNS=$HOME/h2r-runs
```

> `/home` on DelftBlue is a hard 30 GB quota that fills SILENTLY — the symptom is
> exit code 6 and a log that just stops. `du -sh $HOME` before submitting.

### 1.3 The upstream artifacts must still be there

```bash
ls -la $RUNS/output/bc_dataset/train_regrasp_off.h5 \
       $RUNS/output/bc_dataset/val_regrasp_off.h5
ls -la output/regrasp_pins_train_off.json \
       output/regrasp_pins_train_off_excluded.json \
       output/regrasp_demos_train_off_ok.json \
       output/direction_table_train_off.json
```

All five must exist. `_meta.d_rule` in the direction table must read
`grasp_offset` — §2 re-checks that as part of the input report.

---

## 2. GATE — what the filter does to the `_off` shard

**Do not skip this, and do not submit before reading it.**

Everything measured above is from the *four-bin* run-2 shard. Run 10's entire
premise is that `grasp_offset` unlocks `−z` (0 → 235 scenes) and `−x`
(12 → 191), and those are precisely the directions whose demonstrations are most
likely to fail: closing the fingers on an object's underside is harder for the
planner than approaching its free end. If those two bins have poor reach rates,
the filter deletes exactly the data this run exists to test, and the result reads
as "the new bins do not help" when the truth is they were never trained on.

```bash
python examples/build_demo_table.py \
    --demos $RUNS/output/bc_dataset/train_regrasp_off.h5

python examples/analyze_demo_bins.py \
    output/bc_dataset/tables/train_regrasp_off_demo_success_detail.csv
```

Read three things in that output.

**The per-bin `reach` column.** Six live bins, and the rate for `−z` and `−x`
against the rate for `+x`/`+z`. For reference the run-2 shard ran `+x` 75.9%,
`+z` 78.3%, `+y` 64.5%, `−y` 56.3% — so the spread between the easiest and
hardest live bin was already ~20 points.

**The grasp-poses-per-scene histogram, `reach` column.** Its `0` row is the
scenes that lose every demonstration and drop out of the table entirely. Its
`>= 2 (can pair)` line is the count of scenes that can still supply the
same-observation/different-command contrast — the only data that forces the
conditioning channels to be read.

**Total surviving pairs**, which sets the episode budget in §4.

### Decision rule

| what the gate shows | do |
|---|---|
| `−z` and `−x` reach rates within ~15 pts of `+x`/`+z`, `>= 2` above ~250 scenes | **submit as configured** — §3 |
| `−z` or `−x` reach rate collapsed (say under 40%) | the filter is eating the run's subject. Submit with `reach_filter: false` and note run 10 as pre-filter, or raise `reach_pos_thresh` and re-gate — but say which in the run log |
| `>= 2` below ~150 scenes | too little pairing left to test conditioning at all. Do not submit; the fix is more base collection, not a threshold |

If you opt out, it is one line in `examples/configs/regrasp_run10.yaml`:

```yaml
SIM:
  reach_filter: false
```

---

## 3. Check and submit

```bash
REGRASP_DATA=$RUNS/output python examples/check_regrasp_inputs.py regrasp_run10
```

**Expect `All inputs present.` and, in the report:**

- `d_rule: grasp_offset ... depth 11.22 cm, min offset 2.0 cm` with
  `(table: grasp_offset)` — **not** `** MISMATCH **`
- `TRAIN.train_cfg  examples/configs/bc_regrasp_run4.yaml`, `d_noise_deg=0.0`
- `command: deploy=bin_axis  label=d_world   (SAME vector)`
- `reach_filter=ON  (0.02 m / 0.34 rad)`
- `-> beta 0.9 -> 0.75, 25 iters, m=600, scratch (FTL)`

**Submit:**

```bash
sbatch --time=24:00:00 \
  --export=ALL,RUN=regrasp_run10,CFG=examples/configs/regrasp_run10.yaml,SCRATCH_ROOT=$RUNS \
  examples/slurm/train_regrasp.sbatch
```

### The first two minutes of the log

Four lines, in this order. Any one missing means stop and read §6.

```
[d_rule] grasp_offset (d = centroid -> gripper point at 11.22 cm, min 2.0 cm)
[reach-filter] <base>.h5: N/M episodes reached their grasp (0.02 m / 0.34 rad)
[grasp-pin] kept ... scenes ...                       <- the keep_only report
[regrasp] up to K direction(s) per scene -> P collectable (scene, direction) pairs
```

`N/M` must match the gate's total. `K` is `max_grasps` **after** both prunes — if
a bin lost all of its pairs, `K` can be below 6, and since `n_scenes = m //
max_grasps` that silently changes how many scenes an iteration draws. `P` is the
pool the run actually samples from.

Then, at the first fit:

```
[reach-filter] DROPPED X episode(s) that never reached their grasp ...
[direction]    DROPPED Y episode(s) whose bin_realized != bin_assigned
[normalize]    fit over Z episode(s); X+Y dropped ...
```

`Z` must equal the `D_episodes` in `dagger_log.csv` row 0. If it does not, the
normalizer and the dataset have come apart and the head's output scale is defined
by data the model never sees — kill the run.

---

## 4. Budget

The old book's §6 fit assumed the unfiltered pair count. Scale it by what
survived the gate: wall clock is close to linear in episodes per iteration.

```
iterations        25
episodes/iter     m = 600, expanded per scene -> depends on max_grasps after the prune
```

With the run-2 shard's 31% pair loss as the reference, expect roughly 0.7× the
pre-filter estimate. The run is resumable — a 24 h timeout costs only the queue
wait, since resubmitting the identical command skips completed iterations.

`REGRASP_DATA` shards are the disk risk, not the run dir. Re-check `du -sh $HOME`
after iteration 1 and extrapolate.

---

## 5. Monitoring and what to read first

```bash
python examples/status_regrasp_run.py $RUNS/output/dagger_runs/regrasp_run10
bash examples/watch_regrasp.sh -f slurm_logs/regrasp_<jobid>.out

# read-only, safe while running
python examples/plot_regrasp_run.py  $RUNS/output/dagger_runs/regrasp_run10
python examples/plot_regrasp_fits.py $RUNS/output/dagger_runs/regrasp_run10
```

**`bin_diag_rate` first.** It should rise against run 7's 0.903, because "close on
this part of the object" names a target visible in the policy's own cloud —
`d·r = d · normalize(pᵢ − c)` becomes almost a direct readout. **Chance is 1/6
here, not 1/4**, so the number is not comparable to earlier runs without saying
which chance level it is against.

**Then the `−z` / `−x` rows.** High `bin_diag_rate` with low `success_rate` there
means the policy obediently drives at an underside it cannot grasp from — a
feasibility-mask problem, not a conditioning one.

**`D_episodes` and `D_dagger_frac`.** These now count post-filter episodes, so
they are not comparable with runs 1–9's columns. Say so when quoting them.

**Do not compare `val_loss` with runs 1–9.** The val shard is filtered too, so
the quantity changed. Within run 10 it is still a valid selection signal.

`EVAL.holdout: false`, so the in-loop curves are train-set rates by design.
Held-out numbers need the test-split tables built with `--d-rule grasp_offset`:

```bash
python examples/eval_regrasp_testset.py \
    --run-dir $RUNS/output/dagger_runs/regrasp_run10 --chained
```

---

## 6. Failure modes specific to this run

**`SIM.reach_filter is on but TRAIN.base_train_h5 does not exist`** — a
`SystemExit` at startup, before the queue is spent. `REGRASP_DATA` did not
resolve, or the `_off` shard is not staged. Check §1.3.

**`[reach-filter]` line never appears** — the code predates the filter. Re-run
§1.1; do not assume it defaulted off.

**`max_grasps` drops below 6** — legitimate, and it means a bin lost every pair
to one of the two prunes. It changes `n_scenes = m // max_grasps`, so the
iteration draws more scenes and more episodes than the budget assumed. Read `P`
in the startup line rather than computing it.

**The aggregate comes out nearly empty** — the classic `d_rule` mismatch, not the
reach filter: bins populated by one rule and `bin_realized` measured by the other
disagree on most episodes, so the miscaption filter drains D.
`build_regrasp_context` refuses that combination, so it should be caught at
startup; if it is not, check `_meta.d_rule` in the direction table.

**Normalizer/dataset episode counts disagree** — see §3. Kill it.

---

## 7. Quick reference

```bash
# environment
conda activate pch2r_dev && cd ~/handover-sim2real
export GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet"
export RUNS=$HOME/h2r-runs

# gate (§2) — REQUIRED before submitting
python examples/build_demo_table.py --demos $RUNS/output/bc_dataset/train_regrasp_off.h5
python examples/analyze_demo_bins.py output/bc_dataset/tables/train_regrasp_off_demo_success_detail.csv

# check and submit (§3)
REGRASP_DATA=$RUNS/output python examples/check_regrasp_inputs.py regrasp_run10
sbatch --time=24:00:00 \
  --export=ALL,RUN=regrasp_run10,CFG=examples/configs/regrasp_run10.yaml,SCRATCH_ROOT=$RUNS \
  examples/slurm/train_regrasp.sbatch

# monitor (§5)
python examples/status_regrasp_run.py $RUNS/output/dagger_runs/regrasp_run10
```
