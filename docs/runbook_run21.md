# Runbook — regrasp_run21 (DelftBlue)

**Audience: the agent executing this on DelftBlue.** You have the repo, a SLURM
account, and this file. Follow it top to bottom.

> **Standing rule: do NOT `git commit` or `git add` anything unprompted.**
> Propose the command and let the user run it. This applies to every stage below,
> including the tables and shards this run produces.

---

## 0. What run 21 is

Run 19 plus seven changes. Two are bug fixes, four are new behaviour, one is the
rig:

| # | change | where | kind |
|---|---|---|---|
| 1 | `DAGGER.replan_after_pin: true` | config | fix — step-0 expert label |
| 2 | worker/manager pin-table agreement | `regrasp/setup.py` | fix — **no config key** |
| 3 | `DAGGER.plane_gate: true` | config | new — only the expert enters the reach corridor |
| 4 | `DAGGER.max_steps: 50 → 80` | config | new — horizon |
| 5 | `EVAL.max_steps: 50 → 80` | config | new — scored to match |
| 6 | `SIM.cfg_file: pretrain_right.yaml` | config | new — **right camera only, no wrist** |
| 7 | `DAGGER.num_iters: 25 → 35` | config | new — **and the beta ramp with it** |

### Change 7 moves beta, not just the iteration count

`num_iters` is the **denominator of the linear beta schedule**:

```
beta(i) = beta_start + (beta_end - beta_start) * (i - 1) / (num_iters - 1)
```

So 35 does not append ten iterations at the floor — it **stretches the handover**.
The endpoints are unchanged (0.900 → 0.750); what changes is where you are along
the way:

| iteration | run 21 beta | run 19 beta |
|---|---|---|
| 1  | 0.900 | 0.900 |
| 10 | 0.860 | 0.844 |
| 20 | 0.816 | 0.781 |
| 25 | 0.794 | **0.750** (its last) |
| 35 | **0.750** | — |

The learner takes over more slowly throughout, reaching run 19's final beta only
at iteration 35. **Do not compare run 21's iteration N against run 19's iteration
N** — at equal iteration index they are at different betas.

The reason for going to 35: run 19 was still rising when it stopped. Its best
iteration was its **last** (0.743 @ it25) and iters 14–20 averaged 0.648, so 25
was a stopping point rather than a plateau. Run 9, by contrast, went flat from
iteration 14.

**This is not a clean ablation and is not meant to be.** Neither was run 19,
which moved the shield and `iter_epochs` together. If run 21 beats run 19 the
per-change counters below say which mechanism moved; if it loses, change 6 is
the first suspect.

### Change 2 has no config key, on purpose

`build_regrasp_context` never applied the reach prune that `train_regrasp.py`
applies to the manager's copy, and `keep_only` **renumbers slots** — so the
manager and its twenty workers disagreed about what every slot index *meant*,
while the pool is shipped `(scene, slot)` and not the table.

Measured on run 19: **246 of 1097 manager slots (22.4%)** named a different bin
in the worker, and **898 of 4883 episodes (18.4%)** landed on pairs the reach
filter had excluded. Those failed `reached()` **77.1%** of the time against
**31.1%** for pairs that belonged in the pool — 692 of run 19's 1933 dropped
episodes, **36% of all the collection waste**.

There is no version of the loop that should keep this, so it is unconditional.
**Run 19's numbers are not reproducible on this code. That is intended.**

---

## 1. Why the whole pipeline re-runs

**Because of the camera change, and the reason is not obvious.**

`build_direction_table.py` records `centroid_world` as the **observed point
cloud's** centroid — its own docstring says so — not the object pose. That is
deliberate: it is the same quantity the policy sees. But it means dropping the
wrist camera moves the centroid by centimetres, moves the anchor frame with it,
and **re-bins grasps**. Run 9/19's tables caption the wrong bins here.

**Nothing in the repo would have told you.** Every shard records `cameras` in
its attrs and no loader compares them. `resolve_anchor_ref` guards the anchor
reference and `resolve_d_rule` the direction rule; until run 21 nothing guarded
the camera set. `preflight_regrasp.sh` check 4 now compares `cfg_file` against
the table's `_meta` — **run it**.

Everything is rebuilt under the `_right` suffix, distinct from run 9/19's
unsuffixed files and run 20's `_aabase`. Run 21 does **not** replace run 19's
files, it **adds** ~5 GB alongside them.

---

## 2. Preflight — do not skip

```bash
cd ~/h2r/handover-sim2real
examples/slurm/preflight_regrasp.sh regrasp_run21
```

Four checks. **Check 1 (disk) and check 2 (environment) must pass before you
submit anything.**

- **Check 1, disk.** `/home` is a hard **30 GB quota that fills silently** —
  Python cannot write an ENOSPC traceback to a full disk, so the job dies with
  exit code 6 and an **empty `.err` file**. Losing 5 h of collection this way is
  the most expensive failure available here and it reports nothing. Run 21 needs
  ~5 GB on top of whatever run 19 left.
- **Check 3, inputs.** On a first pass every `_right` input is missing. **That
  is expected** — stages 1–4 build exactly those. Only a problem if you meant to
  reuse an existing table.
- **Check 4, consistency.** Compares `d_rule`, `d_point_depth`,
  `anchor_hand_ref` **and now `cfg_file`** against the pin table's `_meta`. On a
  first pass it prints "not built yet — nothing to compare (fine)". On later
  passes all four must say `agree`.

---

## 3. Submit

```bash
sbatch examples/slurm/regrasp_run21_all.sbatch
```

That is the whole recipe. The script is **idempotent**: every stage is skipped
when its output exists and `train_regrasp.py` resumes from `state.json`.

### It will not finish in one allocation

```
  direction table train    1.5 h
  direction table val      0.2 h
  assign per-bin demos     seconds
  collect base train       5.0 h
  collect base val         0.3 h
  audit                    minutes
  train, 35 iterations    25-28 h
  -------------------------------
  TOTAL                   32-35 h
```

Run 19 measured ~16 h for 25 iterations, but per-iteration cost **rises with
|D|** — run 9 logged 30.9 min mean against 52.6 min by iteration 26 — so
iterations 26–35 add roughly 10 h, not 40% of 16.

24 h is DelftBlue's hard maximum, so nothing fits. **Resubmit the identical
command** and pass 2 picks up where pass 1 was killed. Expect two or three
passes. To chain them without waiting:

```bash
J1=$(sbatch --parsable examples/slurm/regrasp_run21_all.sbatch)
J2=$(sbatch --parsable --dependency=afterany:$J1 examples/slurm/regrasp_run21_all.sbatch)
sbatch --dependency=afterany:$J2 examples/slurm/regrasp_run21_all.sbatch
```

`afterany`, not `afterok` — pass 1 being killed on the wall clock is the
*expected* outcome, not a failure.

**`--time=20:00:00`, not 24.** A 24 h request is the maximum, so the scheduler
can never backfill it into a gap; with fairshare depressed after a few GPU jobs
that is the difference between starting in an hour and starting in two days.
If the queue is bad, `sbatch --time=08:00:00 ...` works fine — the script just
does fewer stages per pass.

### Why a short job is safe

Both collections and both table builds write to `<path>.partial` and rename only
after a zero exit. A job killed mid-collection leaves **no file** the
`[ ! -f ... ]` guard would mistake for a finished one. Without that, pass 2
would train on a truncated shard and say nothing.

---

## 4. Gates — what to read, and when

### Gate 1 — the per-bin histogram, after stage 2 (~2 h in)

Prints inline, *before* 5 h of simulator time is spent against it.

Run 9 measured **+x 486, +y 359, −y 319, +z 412**, with **−x and −z empty**.
The camera change moves the centroid, so expect drift. The pass condition:

- −x and −z **stay empty** (−z is geometrically impossible with the object above
  a table; −x is over the giver's fingers, which the hand filtering removes)
- the ordering **+x > +z > +y > −y** holds
- totals within roughly ±15% of run 9's

A wildly different shape means the centroid moved far enough to break the frame.
**Stop and report** — the run would not be comparable to anything.

### Gate 2 — override rate, iteration 1

```
(shield_policy + plane_policy) / (steps - expert_steps)
```

How much of the learner's control was taken away. This raises the **effective
beta** above the configured 0.75, and a run whose learner is overridden half the
time is not the run the config describes.

`plane_policy` is **new and untested at scale**. If it is a large fraction, the
gate is doing more than intended — loosen `DAGGER.plane_gate_margin` (a positive
value lets the hand dip slightly into the corridor) rather than turning the gate
off. Report the number before changing anything.

### Gate 3 — `pin_goal_moved / episodes`, iteration 1

Measures the bug `replan_after_pin` fixes: episodes where the goal *index*
actually changed across the pin, i.e. where run 19's step-0 label was genuinely
wrong. Large fraction → the bug was widespread. Small → scene 95 was unlucky and
change 1 will not move the headline.

### Gate 4 — `reached_grasp / episodes`

Run 19 averaged **0.604** over iterations 1–25 (at its own beta schedule). This should rise toward **~0.69
on arithmetic alone** — the pool fix removes the 18% of episodes that were
failing 77% of the time.

**Do not read that rise as learning.** It is bookkeeping. Report it as such.

---

## 5. Reading the result

`success_rate` is **not comparable with runs 1–20**. This run scores at 80 steps;
they scored at 50. Compare the held-out test-set number instead, and pass it a
matching `--max-steps`.

**Beta differs at equal iteration index** (see change 7), so an iteration-by-
iteration overlay against run 19 compares two different on-policy fractions. The
comparable points are the ENDPOINTS — run 19 it25 and run 21 it35 are both at
beta 0.750.

The **noise floor is 0.088** — an identical rerun of run 7's iteration 0 scored
0.282 vs 0.370. Run 19 peaked at **0.743 (it25)**, mean **0.648** over iters
14–20. Only a gain past ~0.09 is readable.

**Losing the wrist camera is expected to cost success rate in sim.** Every run to
date has leaned on the eye-in-hand view for object localisation. The argument for
change 6 is not that it helps in simulation — it is that what remains is what the
real rig actually has. A run 21 that merely holds near run 19 while seeing one
fixed camera is a good outcome; say so plainly rather than presenting it as a
regression.

`EVAL.holdout` is still `false`, so the eval scenes are collected on. These are
optimistic, train-set numbers — comparable to run 9/19 and to nothing outside it.

### New in `dagger_log.csv`

| column | meaning |
|---|---|
| `replan_pin` | episodes that re-planned after the pin |
| `pin_goal_moved` | of those, how many changed goal index — **the bug's incidence** |
| `plane_policy` | policy steps refused at the standoff plane |
| `plane_jolt_reject` | jolt **redraws** at the plane (can exceed `dart`) |

### New in each episode's HDF5 group

`step_drivers`, `int8[T]`, index-aligned with `expert_actions`. Diagnostic only —
nothing reads it.

```
-1 none (label recorded, episode ended before acting)
 0 expert       1 policy       2 dart_free
 3 dart_reach   4 shield       5 plane_gate
```

Collapse 0/4/5 for "expert-driven". The overrides are kept separate from plain
expert steps because a shielded step *executes* π\*, but got there because the
learner proposed something refused — a different event from β choosing the expert.

---

## 6. Failure modes seen before

| symptom | cause | fix |
|---|---|---|
| exit code 6, **empty `.err`** | `/home` quota full | free space, resubmit; run preflight first |
| `AssocMaxGRESPerJob` in the queue reason | **known false reason code** (forum-confirmed) | wait; it is not a real limit |
| job never starts, 24 h request | maximum-length jobs cannot be backfilled | resubmit at `--time=20:00:00` or less |
| `ModuleNotFoundError: omg` | `OMG_PLANNER_DIR` unset or wrong | the sbatch exports it; check the `:-` default did not silently take an empty value |
| "asked to act with no direction set" | pin table missing the slot the manager emitted | **should be impossible after change 2** — if you see it, stop and report, the fix did not take |
| preflight check 4 says `cfg_file MISMATCH` | reusing run 9/19's wrist+right tables | delete the `_right` table and let stage 1 rebuild it |

---

## 7. What to report back

1. Preflight output, all four checks.
2. Gate 1 histogram against run 9's numbers.
3. Gates 2–4 at iteration 1, with the raw counts, not just ratios.
4. Final `success_rate` per iteration, **stated as not comparable to runs 1–20**.
5. Anything in section 6 you hit.

Do not commit or stage anything. Hand the user the commands.
