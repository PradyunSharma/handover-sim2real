# Runbook — scoring a Phase-4 DAgger run on the held-out s0 test split

Self-contained: the reader is assumed to have the repo and cluster access but
**not** the conversation that motivated this. Design rationale lives in
`examples/eval_dagger_testset.py`'s docstring; this file is the sequence of
actions, the gates, and what to do when one fails. The Regrasp twin is
[`runbook_testeval_run19_23.md`](runbook_testeval_run19_23.md).

**Standing rules for whoever executes this**

- Do **not** `git commit` or `git add` anything unprompted. Propose the command
  and let the user run it.
- Do **not** launch a job this file does not list.
- This is **read-only with respect to the run**: it writes three new files into
  the run directory (`test_log.csv`, `test_eval.png`, `test_summary.png`) and
  touches nothing else.

---

## 0. What this answers, and why it did not exist before

**Phase 4 has never had a held-out number.** `eval_dagger_run.py` takes no
`--split` — it rebuilds the run's own context from the run's own `config.yaml`,
and every Phase-4 config carries

```yaml
SIM:   { split: train }
EVAL:  { num_scenes: 100, holdout: false }
```

so the in-loop curve and the standalone re-score are the **same measurement**: a
`np.linspace` subsample of 100 of ~623 usable **train** scenes, all of which were
also collected on. `dagger4_run19`'s headline has been quoted as a result; it is
a train-set number on a sixth of the train set.

|  | in-loop (`curves.png`) | this (`test_summary.png`) |
|---|---|---|
| split | train | **test** |
| scenes | 100 of ~623, linspace subsample | **144, all of them** |
| collected on? | **yes** | no |
| episodes/iteration | 100 | 144 |

Phase 4 conditions the policy on **nothing** — no goal pose, no direction — so
this script is much smaller than the Regrasp one. Everything that file exists for
beyond the split repoint (per-bin panels, `dir_err`, the retry ladder, the
adaptive ranker) has no referent here.

### What it produces

```
<run>/test_log.csv       one row per iteration — the eval_log.csv schema plus the
                         failure-conditioned ff_* block and this script's own
                         provenance columns (pin_table, excluded_applied)
<run>/test_eval.png      five curves against DAgger iteration: stage rates,
                         chance vs conversion, approach error, and the outcome
                         taxonomy under both denominators
<run>/test_summary.png   THE FIGURE TO READ. One iteration, as bars.
```

---

## 1. Compatibility — one thing must agree, and it is not in the config

`SIM.grasp_pin_table` decides which grasp each close is scored against **and
which scenes are usable at all**. A table built under different settings scores a
different task, silently.

**The pin rule is recorded nowhere except the table itself.** There is no
`grasp_pin_mode` key in any Phase-4 `config.yaml`; the rule lives in the table's
`_meta.mode` and, by convention, in its filename. So the script checks the split
table against **the run's own table**, field by field — not against the config,
which has nothing to say. (A version that read the rule from the config with the
split table's own value as the fallback compared every field against itself and
passed unconditionally. It waved through `grasp_pin_table_val.json`, which is
`furthest_from_hand`, against run 19's `omg`.)

Both rules exist in `output/`, four characters apart in the name:

| file | `mode` | scenes |
|---|---|---|
| `grasp_pin_table_train_omg.json` | `omg` | 720 — **run 19's** |
| `grasp_pin_table_test_omg.json` | `omg` | **144 — use this** |
| `grasp_pin_table_val_omg.json` | `omg` | 36 |
| `grasp_pin_table_val.json` | `furthest_from_hand` | — |

Scoring an `omg` run against a `furthest_from_hand` table measures agreement with
a grasp the policy was never taught: `near_rate`, `chance_rate` and `pos_err` all
collapse for a reason that says nothing about the policy. The script refuses it.

**`dagger4_run19` is ready** — `grasp_pin_table_test_omg.json` agrees on all six
of `mode`, `tol`, `setup`, `hand_collision_filter`, `hand_collision_thresh` and
`valid_grasp_dict_path`. A run pinned with `furthest_from_hand` needs a
`furthest_from_hand` test table built first; there is currently none.

### 1.2 `exclude_scenes` is a train artifact and is dropped

`SIM.exclude_scenes` points at `output/bc_dataset/train_pinned_omg_right_ok.json`
— the **train** scenes whose base demonstration succeeded. Nothing was collected
on test, so it has nothing to say there, and applying it would drop test scenes
because a *train* scene with the same integer id failed. The script pops it on
test/val and keeps it on train, where it is the right filter for exactly the same
reason. The banner says which happened, and `excluded_applied` records it in the
CSV.

---

## 2. Gates, in order

### Gate 1 — the table exists

```bash
ls -la output/grasp_pin_table_test_omg.json
```

Missing → the script exits with the build command.

### Gate 2 — the run directory resolves

```bash
RUN=dagger4_run19
ls -la $SCRATCH_ROOT/output/dagger_runs/$RUN/config.yaml \
       output/dagger_runs/$RUN/config.yaml 2>&1
```

The sbatch tries `$OUT_ROOT/$RUN` then `output/dagger_runs/$RUN`. If neither has
`config.yaml`, set `RUN_DIR` explicitly — do not copy files around.

### Gate 3 — one iteration first

```bash
RUN=dagger4_run19 ITERS=19 sbatch examples/slurm/eval_dagger_testset.sbatch
```

~11 min. **Check three things in the `.out` before the sweep:**

1. `[pin] grasp_pin_table_test_omg.json agrees with the run's ...` is present —
   not a WARNING, not a "cannot be checked";
2. `scenes : 144  (the in-loop eval used 100 of the train split)`;
3. `exclude : dropped (train artifact)`.

### Gate 4 — the sweep

144 episodes × 4.4 s ≈ **11 min per iteration**, so run 19's 26 iterations are
~4.6 h — two 3 h passes. (Phase 4 is one episode per scene; Regrasp's 340
episodes come from ~2.6 direction slots per scene.) Resumable per iteration, so
chain short jobs:

```bash
RUN=dagger4_run19
J=$(sbatch --parsable --export=ALL,RUN=$RUN examples/slurm/eval_dagger_testset.sbatch)
J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,RUN=$RUN \
        examples/slurm/eval_dagger_testset.sbatch)
```

If you only want the headline, `ITERS=19` for run 19's best-on-train plus its
last is 22 minutes rather than 4.6 hours, and `test_summary.png` is a
single-iteration figure anyway.

---

## 3. Reading the result — one trap in the figure

**The stage rates are NOT a funnel**, and both `plot_dagger_run.py` ("the nested
rates") and the evaluator's own docstring say they are. Reading the code:

```python
success = held                      # grasp_held_after_hold — survived hold+release
grasped = env.grasped_active()      # both fingers in contact, at that instant
near    = pos_err <= 0.02 and rot_err <= 0.34
```

Three **independent** predicates evaluated at the same moment. An object can
survive the hold while the contact query reads False, so `success_rate >
grasp_rate` is normal — measured on run 19, it happens in **17 of 26**
iterations. `near_rate` sits at 0.01–0.04 while `grasp_rate` sits at 0.5–0.7, so
it is not a stage between them either. This script's panels are titled "stage
rates (independent, not nested)" for that reason; do not read a gap between two
bars as episodes lost between two stages.

The genuinely conditional chain is the commit one, and it is drawn as such:
`box_chance_rate` (a chance existed) → `box_taken_rate` (it was taken) →
`box_success_rate` (it paid off), each conditional on the one before.

### What each requested number is called

| question | column |
|---|---|
| handover success | `success_rate` |
| grasp commit — "closed \| in jaws" | `box_taken_rate` |
| …and the chance it converted | `box_chance_rate` |
| …and whether the commit paid off | `box_success_rate` |
| failure mode, of the eval set | `f_grasp_ok` … `f_timeout` |
| failure mode, of the **failures** | `ff_grasp_miss` … `ff_timeout` |
| approach error | `eval_min_pos` / `eval_min_rot`, `mean_pos_err` / `mean_rot_err` |

`ff_*` is computed by this script — Phase 4's evaluator emits only `f_*`.
`f_timeout` moves whenever success moves and so cannot separate "got worse" from
"failed differently"; `ff_timeout` is the shape of the failure and is the one to
read when a success rate drops.

---

## 4. Failure modes

| symptom | cause | action |
|---|---|---|
| `[pin] ... was built under different settings` | wrong pin rule | §1 — build a matching table, or `ALLOW_PIN=1` and own the caveat |
| `[pin] ... is the 'val' split's table, but --split is 'test'` | `PIN_TABLE` names the wrong split | drop the override; the default derives it |
| `[pin] the run's own table ... is not readable from here` | the run's train table was moved or deleted | the rule cannot be checked at all — verify `mode` by hand |
| `not found. Build the test-split pin table first` | Gate 1 | run the printed command |
| two blank stacked-area panels | one-iteration sweep | read `test_summary.png`; expected |
| `cannot load best.pt` on some iterations | that iteration published no `best/` | it is skipped, not fatal — check `EVAL.every` |
| exit 6, empty `.err` | `/home` hit its 30 GB quota | `df -h $HOME`; known DelftBlue failure |
| `AssocMaxGRESPerJob` in `squeue` | known **false** reason code | wait; forum-confirmed |

---

## 5. Reporting

State the iteration, the split, the scene count and the checkpoint — all four are
in the suptitle of `test_summary.png`. Say **`best.pt`**, because that is the
Phase-4 default and it differs from Regrasp, which moved to `last.pt` at run 2.

Keep the noise floor in view. Phase 4's own six-sample base-fit spread is
**0.32–0.62, sd 0.115** on 100 scenes, so any comparison resting on a
success-rate difference smaller than ~0.15 is unreadable — and "best iteration"
picked as a max over 26 scored iterations is itself a selection artifact. On 144
test scenes the binomial standard error at p≈0.7 is ±0.038, which is the *floor*,
not the whole variance.

And keep `near_rate` beside `success_rate`. Run 19 reads success ~0.72 at
`near_rate` ~0.04: the policy comes away with the object from almost anywhere and
almost never lands on the pinned pose. Any claim that this work "learns the
demonstrated grasp" needs that second number in the same sentence.
