# Runbook — scoring regrasp_run9 on the held-out s0 test split

Self-contained: the reader is assumed to have the repo and cluster access but
**not** the conversation that motivated this. Design rationale lives in
`examples/eval_regrasp_testset.py`'s docstring and in
[`docs/run_index.md`](run_index.md); this file is the sequence of actions, the
gates, and what to do when one fails.

**Standing rules for whoever executes this**

- Do **not** `git commit` or `git add` anything unprompted. Propose the command
  and let the user run it.
- Do **not** launch a job this file does not list.
- This is **read-only with respect to the run**: it writes two new files into
  the run directory (`test_log.csv`, `test_eval.png`) and touches nothing else.
  No collection, no training, no checkpoints modified.

---

## 0. What this answers, and why it is worth 11 hours

**Every number in run 9's `training_curve.png` is a train-set number.** Its
config has `EVAL.holdout: False` with `EVAL.num_scenes: 100`, so the in-loop
evaluation ran on a `np.linspace` subsample of the *train* split — 100 scenes of
~617 — and **those scenes were also collected on**. Run 9's headline
`success 0.5924 @it23` is therefore optimistic by an unknown margin and must
never be reported as a held-out result.

This produces the first genuinely held-out number for run 9: every scene of the
s0 **test** split, every iteration, under the run's own saved config so the only
thing that differs is the data.

**Iteration 23 is already scored** — one row is in `test_log.csv` from an
earlier single-iteration run. The sweep below skips it automatically; 25
iterations remain, ~10.4 h.

| | in-loop (`training_curve.png`) | this (`test_eval.png`) |
|---|---|---|
| split | train | **test** |
| scenes | 100 of ~617, linspace subsample | **129, all of them** |
| collected on? | **yes** | no |
| episodes/iteration | 238 | 340 |
| run 9 result | `success 0.5924 @it23`, final 0.4958 | *what this measures* |

### What it produces

```
<run>/test_log.csv     one row per iteration, the same ~420-column schema as
                       dagger_log.csv, so the two are spliceable column-for-column
<run>/test_eval.png    rows 0-1: the six panels of curves_regrasp.png
                       rows 2+ : one row per bin — success stages,
                                 chance vs conversion, approach error to the grasp
```

---

## 1. Prerequisites

### 1.1 The test-split pin table — the only hard dependency

```bash
ls -la output/regrasp_pins_test.json output/regrasp_pins_test_excluded.json
```

Nothing in this runbook builds them and the evaluation cannot start without
them. If they are missing, the script exits printing the two commands that would
build them (a `build_direction_table.py --split test` pass, ~0.3 h, then
`assign_direction_demos.py`). **They exist in the local repo — confirm they were
actually synced to the cluster**, because this is the failure that wastes a
queue wait.

Contents, for reference — 129 scenes, 340 (scene, slot) pairs, four live bins:

```
+x_free_end   94      -x_over_fingers  dropped (1.9% feasible)
+z_top_down  100      -z_beneath       dropped (0% feasible)
+y_lateral    70
-y_lateral    76
```

### 1.2 `d_rule` compatibility — checked automatically, worth understanding

`regrasp_pins_test.json` carries no `d_rule` in its `_meta`, so it reads as
**`approach_axis`** by construction. Run 9's config also has no `d_rule` key, so
it too defaults to `approach_axis`. **They match, and the run will start.**

This is specific to runs 1–9. Runs 10+ use `grasp_offset` and 17 uses
`location_extent`; scoring one of those on test needs a test table rebuilt under
that rule, and `resolve_d_rule` will refuse the job rather than silently caption
the bins under the wrong definition. Same for `resolve_anchor_ref` and
`anchor_hand_ref`.

### 1.3 The run directory

```bash
ls -d $HOME/h2r-runs/output/dagger_runs/regrasp_run9 output/dagger_runs/regrasp_run9 2>/dev/null
ls output/dagger_runs/regrasp_run9/iters | wc -l        # expect 26
ls output/dagger_runs/regrasp_run9/iters/iter_23/checkpoints/
```

Run 9 predates the `$SCRATCH_ROOT` convention, so it may live in either place.
The sbatch tries `$SCRATCH_ROOT/output/dagger_runs/$RUN` first, falls back to the
repo's `./output/dagger_runs/$RUN`, and prints which it used. If it is somewhere
else entirely, pass `RUN_DIR=/full/path` instead of `RUN=`.

`--ckpt last` needs `last.pt` in each `iters/iter_NN/checkpoints/`. An iteration
whose checkpoint cannot be loaded is **skipped with a message**, not fatal.

### 1.4 Environment (the sbatch sets these itself)

```bash
cd ~/h2r/handover-sim2real
source ~/anaconda3/etc/profile.d/conda.sh && conda activate pch2r_dev
export GADDPG_DIR=$PWD/GA-DDPG
export OMG_PLANNER_DIR=$PWD/OMG-Planner      # imported at module scope even
                                             # though evaluation never plans
export SCRATCH_ROOT=$HOME/h2r-runs
```

Disk: two small files, a few MB. The `/home` quota is not a concern here.

---

## 2. Timing

**Measured, not derived.** Iteration 23 of run 9 on this split scored 340
episodes in **1497 s**:

```
129 test scenes x 2.64 slots = 340 episodes per iteration
340 x 4.4 s                  = ~25 min per iteration
x 26 iterations              = ~10.8 h total
```

The script is **serial** — it calls `evaluate_policy` directly, with no worker
pool — which is why a single iteration costs 25 minutes of wall clock rather
than the ~90 s the in-loop evaluation took on a comparable episode count.

> An earlier version of this file estimated 7.4 h, derived from the in-loop
> `eval_s` divided by what was read as an 8-worker pool. That 8 was
> `TRAIN.num_workers` — the **dataloader** worker count — not the eval pool,
> which is set by `--num-workers` on `train_regrasp.py` and was larger. The
> measured 25 min/iteration above is the number to plan against.

`--chained` adds up to `--max-attempts` (4) rollouts per scene plus rewind
replay, roughly **2.5×** the above. Do not combine it with `--iters all` on a
first pass.

### Why four 3 h jobs and not one 12 h job

`write_log` rewrites the CSV after **every** iteration and `read_done` skips
rows already in it, so **a wall-clock kill costs at most the ~25 min iteration
in flight** and nothing else. DelftBlue backfills short jobs ahead of long ones,
and total wall clock here is dominated by queue time. Four 3 h passes give 12 h
of capacity for 10.8 h of work, and every job queues as a short job.

**Do not request 24 h.** Do not request 8 h either — there is no benefit, since
the sweep resumes for free.

---

## 3. Submit

`sbatch` does **not** accept `VAR=value` as a positional argument. Use one of
these two forms.

```bash
# ---- all 26 iterations: four chained 3 h passes -------------------------
J=$(sbatch --parsable --export=ALL,RUN=regrasp_run9 \
        examples/slurm/eval_regrasp_testset.sbatch)
echo "pass 1 = $J"
for i in 2 3 4; do
    J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,RUN=regrasp_run9 \
            examples/slurm/eval_regrasp_testset.sbatch)
    echo "pass $i = $J"
done
```

**`afterany`, not `afterok`.** A pass that hits its wall clock exits non-zero,
which is expected; under `afterok` the next pass would never start. A pass that
finds every iteration already scored prints `nothing to score` and exits in
seconds, so a spare tail job costs a queue slot and nothing else.

### Cheaper first passes

```bash
# ~2.5 h — six points, enough to see the curve shape (needs --time=03:00:00)
sbatch --export=ALL,RUN=regrasp_run9,ITERS=0,5,10,15,20,25 \
       examples/slurm/eval_regrasp_testset.sbatch

# ~25 min — one iteration (23 is already scored; pick another)
sbatch --time=01:00:00 --export=ALL,RUN=regrasp_run9,ITERS=11 \
       examples/slurm/eval_regrasp_testset.sbatch

# ~75 min — three points, and the figure draws lines rather than dots
sbatch --time=02:00:00 --export=ALL,RUN=regrasp_run9,ITERS=0,11,23 \
       examples/slurm/eval_regrasp_testset.sbatch
```

A later `--iters all` fills in whatever is missing and re-scores nothing, so
these compose: start with the six-point pass, decide whether the full sweep is
worth it, then submit it.

### Other knobs

| variable | default | effect |
|---|---|---|
| `RUN` | `regrasp_run9` | run name under `$OUT_ROOT` |
| `RUN_DIR` | *(derived)* | full path, overrides `RUN` |
| `SPLIT` | `test` | `test` / `val` / `train` (`train` = the FULL train set) |
| `ITERS` | `all` | `all` or a comma list |
| `CKPT` | `last` | `last` or `best` |
| `CHAINED` | *(off)* | `1` adds true chained retry, ~2.5× slower |

---

## 4. Monitor

```bash
squeue -u $USER -o "%.10i %.14j %.8T %.10M %.10l %R"
tail -f slurm_logs/testeval_${J}.out
```

Each iteration prints one line as it completes:

```
[iter 23] .../iters/iter_23
  success=0.412 grasp=0.399 close=0.518  (1043s)
  per-bin success / ended-in-commanded-bin:  +x 0.45/0.88  +y 0.39/0.74  ...
```

Progress at any time, without waiting for the job:

```bash
R=output/dagger_runs/regrasp_run9         # or the $SCRATCH_ROOT path
tail -n +2 $R/test_log.csv | wc -l        # iterations scored so far
```

---

## 5. Gates

**Gate 1 — the first iteration completes in ~25 min.** If it takes much longer,
the estimate is wrong for this machine and the 3 h passes will each land ~9
iterations instead of ~7; harmless, but add a fifth pass. If it takes 5
minutes, check `num_episodes` in the CSV — a number well below 340 means the pin
table is smaller than expected and the score is over a different population.

```bash
python - <<'PY'
import csv
r = list(csv.DictReader(open("output/dagger_runs/regrasp_run9/test_log.csv")))[0]
for k in ("iter","split","num_scenes","num_episodes","eval_s",
          "success_rate","dir_err_median"):
    print(f"  {k:16s} {r.get(k)}")
PY
```

Expect `num_scenes 129`, `num_episodes 340`, `split test`.

**Gate 2 — `dir_err_median` should be small.** Run 9 trains on the grasp's own
axis (`DATA.d_source: d_grasp_world`) and its in-loop `dir_err_median` is
**10–12°**. A test value in that range says the conditioning transfers. A value
near 45° or above means something is mis-captioned rather than that the policy
generalises badly — stop and check §1.2.

**Gate 3 — the held-out number against the train-set one.** Run 9's in-loop best
is `0.5924 @it23`. A test value below that is *expected and correct* — the point
of the exercise is to measure how much. A test value **above** it is suspicious
and worth investigating before reporting.

---

## 6. Reading the results

```bash
python examples/eval_regrasp_testset.py \
    --run-dir output/dagger_runs/regrasp_run9 --split test --plot-only
```

The sbatch already does this at the end of every pass, so a partial curve is
available after the first job.

### Noise floor — read this before believing any gap

340 episodes per iteration, four bins of 70–100:

| quantity | n | 1σ at p≈0.5 | 95% interval |
|---|---|---|---|
| pooled `success_rate` | 340 | ±2.7 pp | **±5.3 pp** |
| `succ_bin_+z` | 100 | ±5.0 pp | ±9.8 pp |
| `succ_bin_+x` | 94 | ±5.2 pp | ±10.1 pp |
| `succ_bin_-y` | 76 | ±5.7 pp | ±11.2 pp |
| `succ_bin_+y` | 70 | ±6.0 pp | ±11.7 pp |
| `retry_at_k` | 129 scenes | ±4.4 pp | ±8.6 pp |

**A per-bin difference under ~10 pp is not readable.** Iteration-to-iteration
wobble of that size is sampling noise, not learning.

### Two panels that will look wrong and are not

- **"success per commanded direction — ALL scenes"** renders the note
  *"EVAL.full_bin_coverage was off"* instead of curves. Correct: that key
  arrived at run 16, so run 9 logged no `succ_bin_all_*`. The
  **DEMONSTRATED scenes** panel is the real per-bin curve for this run.
- **"arrived from the COMMANDED side"** is saturated and near-useless. It
  measures object-centroid → **wrist**, ~13 cm behind the fingers, so a perfect
  grasp already reads ~27° against a 30° threshold and the ceiling is 0.57–0.69
  rather than 1.0. A bin far below the others still means something; the
  absolute level does not.

### What to report

Best `success_rate` and its iteration, the final value, `dir_err_median`, the
four `succ_bin_*`, and `retry_at_4`. State plainly that this is the **held-out
test** number and that run 9's `0.5924 @it23` is a train-set number on scenes it
was also collected on. Add the row to `docs/run_index.md`.

---

## 7. Failure modes

### 7.1 `output/regrasp_pins_test.json not found`

§1.1. The script prints the two build commands. Building the test table is a
simulator pass of its own (~0.3 h for 130 scenes) and needs `OMG_PLANNER_DIR`.

### 7.2 `[cfg] SIM.d_rule: ... but ... was built under 'approach_axis'`

You are scoring a run 10+ policy against the runs-1–9 test table. §1.2. Build a
test table under that run's rule; do not edit the config to match the table.

### 7.3 `[skip] cannot load last.pt`

That iteration's checkpoint is missing or corrupt. Non-fatal — the sweep
continues and the CSV simply has no row for it. Check
`ls iters/iter_NN/checkpoints/`; `--ckpt best` may exist where `last` does not.

### 7.4 `nothing to score (use --force to re-score)`

Every requested iteration is already in the CSV. Expected on a tail pass. To
deliberately re-score, add `--force` — which discards the whole CSV and starts
over, so prefer deleting the specific rows if you only want some.

### 7.5 `AssocMaxGRESPerJob` in `squeue`'s reason column

A **known false reason code** on DelftBlue, forum-confirmed. The job is queued
normally. Do not change the resource request in response to it.

### 7.6 The job dies with no traceback

Check the time limit actually granted, and whether it was the wall clock:

```bash
sacct -j <jobid> --format=JobID,JobName%14,State,ExitCode,Timelimit,Elapsed
```

A wall-clock kill is expected and harmless here — resubmit and it resumes. An
exit code 6 with an empty `.err` would mean a full disk, but this job writes
only a few MB, so that points at something else on `/home`.

---

## 8. Quick reference

```bash
cd ~/h2r/handover-sim2real
ls -la output/regrasp_pins_test.json          # the one prerequisite

# all 26 iterations, four chained 3 h passes (~10.8 h of work)
J=$(sbatch --parsable --export=ALL,RUN=regrasp_run9 \
        examples/slurm/eval_regrasp_testset.sbatch)
for i in 2 3 4; do
    J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,RUN=regrasp_run9 \
            examples/slurm/eval_regrasp_testset.sbatch)
done

# progress / results
squeue -u $USER
tail -n +2 output/dagger_runs/regrasp_run9/test_log.csv | wc -l
python examples/eval_regrasp_testset.py \
    --run-dir output/dagger_runs/regrasp_run9 --split test --plot-only
```

**This is a held-out number. Run 9's in-loop `0.5924 @it23` is not.**

### Known gap, if the 7 h is annoying

The script is serial. The in-loop evaluation was parallelised across the
collection worker pool for exactly this reason — `run_eval`'s docstring records
that as ~18.5 h saved over run 3 — and the machinery
(`pool.evaluate(run_dir, ckpt, jobs, params)`) already exists. Wiring a
`--num-workers` flag into `eval_regrasp_testset.py` would take this from 10.8 h
to **under 1 h** on 8 workers and make `--chained --iters all` viable. Not done;
ask before assuming it is.
