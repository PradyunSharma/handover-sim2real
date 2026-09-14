# Runbook — scoring regrasp runs 19, 21, 22 and 23 on the held-out s0 test split

Self-contained: the reader is assumed to have the repo and cluster access but
**not** the conversation that motivated this. Design rationale lives in
`examples/eval_regrasp_testset.py`'s docstring and in
[`docs/run_index.md`](run_index.md); this file is the sequence of actions, the
gates, and what to do when one fails.

**Standing rules for whoever executes this**

- Do **not** `git commit` or `git add` anything unprompted. Propose the command
  and let the user run it.
- Do **not** launch a job this file does not list.
- This is **read-only with respect to the run**: it writes three new files into
  the run directory (`test_log.csv`, `test_eval.png`, `test_summary.png`) and
  touches nothing else. No collection, no training, no checkpoints modified.

---

## 0. What this answers

Every number in a run's `training_curve.png` is a **train-set** number. All four
configs carry `EVAL.holdout: false`, so the in-loop evaluation ran on a
`np.linspace` subsample of the *train* split and **those scenes were also
collected on**. This produces the first genuinely held-out number: every scene
of the s0 **test** split, under the run's own saved config, so the only thing
that differs is the data.

|  | in-loop (`training_curve.png`) | this (`test_summary.png`) |
|---|---|---|
| split | train | **test** |
| scenes | 100 of ~617, linspace subsample | **129, all of them** |
| collected on? | **yes** | no |
| episodes/iteration | ~238 | ~340 |

### What it produces

```
<run>/test_log.csv       one row per iteration, 476 columns, the dagger_log
                         schema plus the chained and adaptive-ladder blocks, so
                         the two files are diffable column for column
<run>/test_eval.png      curves against DAgger iteration. Rows 0-1 the
                         conditioning panels; rows 2+ one per commanded bin,
                         four wide: success stages / chance vs conversion /
                         approach error / EVAL OUTCOMES
<run>/test_summary.png   THE FIGURE TO READ. Per-bin bars for one iteration:
                         handover success, grasp commit, ended-in-commanded-bin,
                         raw direction error in degrees, outcome taxonomy
```

**Read `test_summary.png` first.** Every panel on `test_eval.png` is a curve
against iteration, and the stacked-area ones need at least two x points to
render — so on a single-checkpoint sweep four of its panels come out blank with
a legend over them. The bar figure asks the same questions at n=1.

---

## 1. Compatibility — three things must agree, and one of the four runs does not

The split's pin table supplies each scene's `anchor_R` and `centroid_world`, and
both are computed from the **observed** point cloud. A table built under a
different camera set therefore names the bins in a frame the policy cannot
reproduce. Three fields have to match the run's config:

| | `cfg_file` | `d_rule` | `anchor_hand_ref` |
|---|---|---|---|
| `regrasp_pins_test.json` | `pretrain_multicam_wr.yaml` | `approach_axis`¹ | `wrist`¹ |
| run 19 | `pretrain_multicam_wr.yaml` ✅ | `approach_axis` ✅ | `wrist` ✅ |
| run 22 | `pretrain_multicam_wr.yaml` ✅ | `approach_axis` ✅ | `wrist` ✅ |
| run 23 | `pretrain_multicam_wr.yaml` ✅ | `approach_axis` ✅ | `wrist` ✅ |
| **run 21** | **`pretrain_right.yaml` ❌** | `approach_axis` ✅ | `wrist` ✅ |

¹ absent from `_meta`, which reads as these defaults by construction.

**Runs 19, 22 and 23 are ready.** The script refuses run 21 against this table
rather than reporting a number measured against a command the policy was never
given. Two options:

**(a)** Build the matching table first — a separate sim pass, one OMG plan per
scene, roughly 40 min on a GPU node:

```bash
python examples/build_direction_table.py --split test \
    --cfg-file examples/pretrain_right.yaml \
    --anchor-hand-ref wrist --d-rule approach_axis \
    --out output/direction_table_test_right.json
python examples/assign_direction_demos.py \
    --table output/direction_table_test_right.json \
    --out output/regrasp_pins_test_right
```

No `--drop-bins`: run 21's train table (`regrasp_pins_train_right.json`) has
`dropped_bins: []` and **five** live bins, so it commands `-x` as well and the
test table must too. Then submit with
`PIN_TABLE=output/regrasp_pins_test_right.json`.

**(b)** `ALLOW_CFG=1`, and write the caveat into whatever reports the number.
Run 21's figures are then not comparable with 19/22/23's.

### 1.2 The command vector

All four runs use `SIM.command_deploy: bin_centroid`: the policy is conditioned
on the **empirical mean** of each bin's assigned `d_anchor`, which is a property
of the table that mean was computed from. `resolve_command_axes` handed the test
table returns the *test split's* centroids — 2.9° off run 19's `+x`, 6.0° off its
`+z`, 5.9° off its `-y`.

The script therefore loads the run's own `command_axes.json`, the same file
`sim2real/my_regrasp_policy_runner.py` reads on the robot. **Nothing to set; it
is the default**, and it prints the offset it corrected on startup:

```
[command] conditioning on output/dagger_runs/regrasp_run19/command_axes.json
          (bin_centroid); offset from the split's own centroids (deg):
          +x 2.9  -x 0.0  +y 3.7  -y 5.9  +z 6.0  -z 0.0
```

`--split-axes` opts out, which answers "what would this split's own geometry
have commanded" and is **not** the deployment number.

---

## 2. Gates, in order

### Gate 1 — the tables exist

```bash
ls -la output/regrasp_pins_test.json output/regrasp_pins_test_excluded.json
```

Missing → the script exits with the two build commands. Do not improvise them.

### Gate 2 — the run directory resolves

```bash
RUN=regrasp_run19
ls -la $SCRATCH_ROOT/output/dagger_runs/$RUN/config.yaml \
       output/dagger_runs/$RUN/config.yaml 2>&1
ls output/dagger_runs/$RUN/command_axes.json
```

The sbatch tries `$OUT_ROOT/$RUN` then `output/dagger_runs/$RUN`. If neither has
`config.yaml`, set `RUN_DIR` explicitly — do not copy files around.

### Gate 3 — one iteration first, never the whole sweep

```bash
RUN=regrasp_run19 ITERS=25 sbatch examples/slurm/eval_regrasp_testset.sbatch
```

~25 min. **Check three things in the `.out` before submitting the sweep:**

1. the `[command] conditioning on .../command_axes.json` line is present and the
   offsets are single digits;
2. `bins : +x > +z > +y > -y` lists the directions you expect;
3. the `per-bin` line has a non-zero `n=` for every bin.

A zero `n=` on a live bin means the table and the config disagree about which
bins exist, and every per-bin number downstream is then over the wrong
population.

### Gate 4 — the sweep

Iteration count and cost, at the measured 25 min per iteration (340 episodes ×
4.4 s, run 9 on this cluster — do not re-derive it from `eval_s`):

| run | iteration dirs | ~wall clock | 3 h passes |
|---|---|---|---|
| 19 | 36 (0-35) | 15.0 h | 6 |
| 21 | 35 (0-34) | 14.6 h | 6 |
| 22 | 21 (0-20) | 8.8 h | 4 |
| 23 | 22 (0-21) | 9.2 h | 4 |

**The sweep is resumable per iteration** — `write_log` rewrites the CSV after
each one and `read_done` skips what is already there — so chain short jobs
rather than asking for one long one. DelftBlue backfills short jobs ahead of
long ones and a kill costs at most the 25 minutes in flight.

```bash
RUN=regrasp_run19
J=$(sbatch --parsable --export=ALL,RUN=$RUN \
        examples/slurm/eval_regrasp_testset.sbatch)
for i in $(seq 5); do                     # 5 more = 18 h capacity for 15 h
  J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,RUN=$RUN \
          examples/slurm/eval_regrasp_testset.sbatch)
done
```

If you only want the headline and not the curve, score the two or three
iterations that matter instead of all of them — `ITERS=25` for run 19's best,
plus its last. That is 50 min rather than 15 h, and `test_summary.png` is a
single-iteration figure anyway.

A pass that finds everything scored prints `nothing to score` and exits in
seconds, so a spare tail job costs a queue slot and nothing else.

---

## 3. The adaptive direction ladder

`retry_at_k` has always been reduced over `directions.RETRY_LADDER` —
`+x, +z, +y, -y, -z, -x` — a **fixed** order hardcoded from run 11's measured
per-bin success. That is the right constant for comparing runs to each other and
the wrong one for asking what a deployment achieves: a robot does not know run
11, it knows what has worked on the objects it has seen.

`ADAPTIVE=1` keeps a Beta-Bernoulli posterior per bin, updates it after every
episode, and hands the next scene the order standing at that moment.
`adaptive_retry_at_k` is reduced over that per-scene order. It is **causal**: the
order for scene *i* is fixed before scene *i* is rolled out, so it never ranks by
a rate computed from the episode it is about to score.

```bash
RUN=regrasp_run19 ITERS=25 ADAPTIVE=1 BINS='+x,+z,+y,-y' \
    sbatch examples/slurm/eval_regrasp_testset.sbatch
```

`BINS` is the order to **start** from, worth `--rank-prior-strength` (default 6)
episodes per bin as pseudo-counts; after roughly that much evidence the data
takes over. It is a preference, not a filter — live bins you do not name are
appended behind the ones you do, so nothing stops being evaluated. `ONLY_BINS`
is the filter.

### Read the two sweeps differently

| | what the ranker changes | what moves |
|---|---|---|
| independent (default) | the **reduction** only — every (scene, bin) pair is rolled out whatever the order | `adaptive_retry_at_k`, and nothing else |
| `--chained` / `STOP_SUCC=1` | **which rollouts happen** — attempt 2 runs only if attempt 1 failed | `mean_attempts`, and the per-bin denominators |

A higher `adaptive_retry_at_k` in the independent sweep is **the same rollouts
ordered better, not a better policy**. Per-bin success, `dir_err` and the
outcome taxonomy are order-invariant there and will not move.

Under `STOP_SUCC=1` the bins ranked last are sampled only on the scenes the
earlier bins failed, so their success rates are conditioned on difficulty and
are **not comparable across bins**. Read `rank_n_b*` before quoting any of them.

### Modes

| `RANK_MODE` | behaviour |
|---|---|
| `fixed` | never updates. The control condition — diff everything else against it. |
| `mean` | posterior mean. Greedy; locks onto a bin that goes 1-for-1 early. |
| `ucb` | **default.** mean + `c` posterior standard deviations, so an under-sampled bin keeps an exploration bonus. Deterministic. |
| `thompson` | samples the posterior. Randomised — two runs of the same sweep give different ladders, which is why it is not the default. |

### Sanity check on synthetic data

60 scenes, 2 bins each, true per-bin rates `+x 0.75, +y 0.45, -y 0.30, +z 0.20`,
with a deliberately worst-first prior `+z > -y > +y > +x`:

| mode | `@1` | `@2` | stability | final order |
|---|---|---|---|---|
| `fixed` | 0.310 | 0.698 | 1.00 | `+z\|-y\|+y\|+x` |
| `ucb` | 0.496 | 0.659 | 0.84 | `+x\|+y\|-y\|+z` |
| `mean` | 0.496 | 0.682 | 0.83 | `+x\|+y\|-y\|+z` |
| `thompson` | 0.512 | 0.674 | 0.31 | `+x\|+y\|-y\|+z` |

All three adaptive modes recover `+x` as the best bin and lift `@1` from 0.31 to
~0.50. **`@2` barely moves** — with two bins per scene, ordering cannot change
an OR over both, so the whole gain lives at k=1. That is the expected signature;
an adaptive `@2` far above the fixed one would mean something is wrong with the
reduction, not that ranking helped.

`fixed` **must** read stability 1.00 and 0 reorders. It cannot move by
construction, so anything else is a bug in the metric.

---

## 4. What each requested number is called

| question | column | per bin |
|---|---|---|
| handover success | `success_rate` | `succ_bin_{b}` (demo'd), `succ_bin_all_{b}` (every scene) |
| grasp commit — "closed \| in jaws" | `box_taken_rate` | `box_taken_rate_b{b}` |
| …and the chance it converted | `box_chance_rate` | `box_chance_rate_b{b}` |
| ended in the commanded bin | `bin_diag_rate` | `bin_diag_rate_b{b}` |
| arrived from the commanded side | `bin_hit_rate` | `bin_hit_rate_b{b}` |
| direction tracking (index) | `dir_track` | `dir_track_b{b}` |
| **raw direction error, degrees** | `dir_err` | `dir_err_b{b}` |
| failure mode, of the eval set | `f_grasp_ok` … `f_timeout` | `f_*_b{b}` |
| failure mode, of the **failures** | `ff_grasp_miss` … `ff_timeout` | `ff_*_b{b}` |
| retry, fixed ladder | `retry_at_{1..6}` | — |
| retry, learned ladder | `adaptive_retry_at_{1..6}` | — |
| the learned ladder itself | `rank_order` | `rank_pos_b{b}`, `rank_post_b{b}`, `rank_n_b{b}` |

`dir_track` is `1 - dir_err/90` and carries no information `dir_err` does not.
Quote the **degrees**: 0.75 tracking is 22.5°, and whether that is good is a
question about grippers and tolerances that only the degree reading lets you ask.
Two marks are drawn on every `dir_err` panel — 30° is `BIN_HIT_DEG`, the margin
`bin_hit_rate` counts at; **45° is the octahedral Voronoi half-angle**, above
which the achieved direction is nearer some *other* bin than the one commanded,
i.e. the command was inverted rather than tracked loosely.

---

## 5. Failure modes

| symptom | cause | action |
|---|---|---|
| `SystemExit: [cfg] ... was built under SIM.cfg_file` | run 21 against the wr table | §1, option (a) or (b) |
| `not found. Build the test-split tables first` | Gate 1 | run the two printed commands |
| `no command_axes.json at ...` | run predates the file (runs ≤ 8) | it used `bin_axis`; pass `--split-axes` |
| `[command] ... was written under SIM.command_deploy: X but the run's config says Y` | one of the two is stale | do not guess — find which was edited |
| four blank stacked-area panels | one-iteration sweep | read `test_summary.png`; this is expected |
| a bin reads `n=0` | table/config disagree on live bins | stop, Gate 3 |
| `--only-bins ... leaves nothing` | asked for `-x`/`-z` on a table that drops them | those bins are unreachable on s0/train — 0 and 12 scenes of 623 |
| exit 6, empty `.err` | `/home` hit its 30 GB quota | `df -h $HOME`; this is a known DelftBlue failure |
| `AssocMaxGRESPerJob` in `squeue` | known **false** reason code | wait; forum-confirmed, not a real limit |

---

## 6. Reporting

Quote `success_rate_in_table` as the headline, not `success_rate` — under
`FULL_BINS=1` the latter is ~80 % bins the scene never demonstrated and is not
comparable with any earlier run. State the iteration, the split, the scene count
and the episode count; all four are in the suptitle of `test_summary.png`.

Keep the **noise floor** in view: 0.088 run-to-run on this pipeline
(`regrasp_runs_inside_noise_floor`). Any `success_rate` difference smaller than
~0.09 between two runs is unreadable, and "best iteration" selected as a max over
25-35 scored iterations is itself a selection artifact. `dir_err` and
`bin_diag_rate` start far from their floors and have room to move; read those
first.
