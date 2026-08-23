# Regrasp — approach-direction conditioning

The policy is told **which side to come in from**, not which pose to reach. If the
handover fails, it is told a different side and tries again.

    conditioning   a unit vector d, injected PER-POINT into the cloud
    command set    k = 6 octahedral bins in a gravity-aligned, hand-anchored frame
    retry          on failure, command the surviving bin furthest from those tried

Code lives in `handover_sim2real/regrasp/` and `handover_sim2real/regrasp_bc/`.
Nothing outside those two packages is modified.

---

## Why a direction and not a pose

Commanding a **goal grasp pose** carries three costs, all of which get worse with
scale:

- **It needs a grasp proposer at deployment.** The real rig has no pin table.
- **Coverage is per object.** "These grasps for *this* object" does not transfer,
  so the 351-scene ceiling stays the bottleneck.
- **`k` is baked into the data.** Changing the number of retry hypotheses means
  relabelling and retraining.

A direction is object-agnostic, so coverage is required **across the dataset**
rather than per object; and because the network reads a continuous vector rather
than a one-hot, `k` is a test-time knob.

**It also makes the wrist-flip ambiguity impossible.** `augment_flip_grasp`
appends twins rotated π about the gripper's own +z, which leaves column 2 of R
untouched — so `d = −R[:,2]` is *identical* for a grasp and its twin, measured at
`0.000e+00` over 2000 random poses. A pose-space metric ranks those twins as
maximally separated and will happily select two commands that are one physical
grasp; a direction cannot.

---

## The conditioning

**Anchor frame** (recomputed per step), where `c` is the object point-cloud
centroid and `p_wrist` the giver's wrist:

```
z = world up (gravity)
x = normalize(horizontal(c - p_wrist))     # object side, away from the giver
y = z × x
```

Degenerate when the hand sits over the object: fall back to base→object, with
**two thresholds** (engage below 4 cm, release above 8 cm) so the mode latches and
cannot chatter mid-approach. Handedness is **not** mirrored — DexYCB is 542/540
right/left in our collection, so the data covers both.

**The command** is `d = −R_grasp[:,2]` rotated into that frame — the negated
gripper approach axis, pointing from the object outward toward where the gripper
comes from. The sign is verified: advancing 10 cm along `+R[:,2]` shrinks a
scene's candidate spread from 0.0851 m to 0.0340 m on **100 %** of 486 scenes.

**Injection is per-point, and only per-point.** For each point `p_i` with
estimated normal `n_i`:

```
d · n_i                     is this surface facing the incoming gripper?
d · normalize(p_i - c)      is this point on the side I am approaching from?
```

so the model input is **7 channels**: `xyz | ycb | hand | d·n | d·r`. There is no
global `d` branch — `GraspEncoder` is deleted and `fused_dim` returns to
`2 × feature_dim`.

Three facts make this cheap to get right:

1. **Dot products are frame-invariant**, so everything happens in the EE frame
   where the cloud already lives. The anchor never reaches the network — it only
   decides *bin labels* and the retry bookkeeping.
2. **`c` comes from the cloud itself**, identical at collection, training and
   deployment, with no ground truth.
3. **Normals must not ride through `PointListener`.** `se3_transform_pc` rotates
   only rows `:3` and *copies* the rest, so a normal carried as an extra row
   silently stays in the wrong frame. Estimating at the end, in the EE frame,
   sidesteps it and leaves `policy.py` — shared with the rest of the repo —
   untouched.

Normals are kNN-PCA from the observed cloud (`scipy.spatial.cKDTree`; `open3d`
exists only on the robot PC), sign-oriented outward from `c`.

---

## MEASURED: k = 6 is really k = 4

`analyze_direction_feasibility.py` runs the census **offline** — the DexYCB cache
already holds extrinsics-transformed hand and object poses, and the MANO wrist is
`joint7`'s origin in each subject/side URDF. Seconds, no GPU. Confirmed against
the full goal set by the direction table:

| bin | goal-set grasps | scenes reaching it |
|---|---|---|
| `+x` free end | 11358 | 490 (78.7 %) |
| `+z` top-down | 8653 | 415 (66.6 %) |
| `+y` lateral | 4314 | 366 (58.7 %) |
| `−y` lateral | 3959 | 325 (52.2 %) |
| **`−x` over the fingers** | **55** | **12 (1.9 %)** |
| **`−z` from beneath** | **0** | **0 (0.0 %)** |

Replicates on val (`−z` 0/29, `−x` 0/29). `−z` is geometrically impossible — the
object is held above a table. `−x` at 0.19 % of all grasps is the hand-collision
filtering working as designed.

**Both bins are kept and masked at runtime**, so the code stays rig-agnostic and
`−z` becomes real on a table-less setup with no change. But the retry ladder has
**four** live rungs, `chained_retry_at_k` saturates at k = 4, and `succ_bin_−x` /
`succ_bin_−z` are NaN rather than zero.

`k` is a test-time knob **up to 20**: minimum separation falls 78.1° at k=6 to
41.0° at k=20 and 39.4° at k=21, below the ~40° where bins stop being independent
hypotheses. Note the Fibonacci lattice at k=6 gives only 78.1° against the
octahedron's 90°, so `BINS` is strictly better there.

---

## Why two demonstrations per scene

**This is a property of the dataset that no architecture reaches.** At one demo
per scene, `d` is a deterministic function of the observation across the whole
dataset, so the network can drive the loss to its floor by learning
scene → action and ignoring the conditioning entirely. Two demos of one scene
under different `d` map the **same observation to two different actions**, which
is the only thing that forces the channels to be read.

Pairs are selected by **realised** separation, not bin-axis separation — a grasp
may sit up to 45° from its bin's axis, so two grasps in 90°-separated bins can
realise 35° apart. Selecting on the axis overstated the contrast; on the realised
directions the median went 90° → **116°** and the minimum 35° → **41°**.

---

## Runbook

### Everything at once (DelftBlue)

```bash
bash examples/slurm/regrasp_pipeline.sh
```

One command on a **login node**. It runs the offline assignment inline, then
submits six SLURM jobs wired together with `--dependency=afterok`, and returns in
about a second:

```
T  train direction table ──> A  collect train demos ──┐
V  val   direction table ──> B  collect val demos   ──┴─> D  smoke ─> E  train
C  test  direction table ─────────────────────────────────────┴───> F  test eval
```

T, V and C are normally **skipped** — the direction tables are tracked in git —
so the usual case is A, B and C going in together. They exist as stages so a
checkout whose `output/` was never populated builds them rather than stopping.

It cannot be one job: the chain is ~30 h and DelftBlue caps at 24. Splitting it
also means A, B and C run in **parallel**, a failure costs only its own stage,
and the smoke failing stops the chain by construction.

Two things it handles that are easy to get wrong by hand. It **detects a stale
shard** — run 1's `.h5` sits at exactly the path run 2 wants, and its `d_world`
is the grasp axis rather than the bin axis (a median 22° apart), so the pipeline
fingerprints it, moves it to `*.run1-stale.h5`, and re-collects. And it
**derives** `demo_ok_table` and the test pin table inside the jobs that need
them, so no manual step sits between two multi-hour jobs.

`--dry-run` prints the whole graph without submitting. `--force` resubmits
everything. Re-running after a failure resubmits only what is missing.

**Where the data lands.** `SCRATCH_ROOT` (default `/scratch/$USER/handover-sim2real`)
takes the two things that are large:

| | size | where |
|---|---|---|
| run dir — checkpoints + 20 iteration shards | ~3.5 GB | **scratch** |
| base HDF5 shards — `train_regrasp.h5`, `val_regrasp.h5` | ~1.1 GB | **scratch** |
| direction tables, pin tables, `demo_ok` list | ~5 MB | repo |

The split is by size and by kind, not by convenience. `/home` is a **hard 30 GB
quota that fills silently** — the job dies with exit code 6 and no traceback,
because Python cannot write one to a full disk. `/scratch` is **purged by age**,
so the small JSON tables stay in the repo where they are version-controlled: they
are inputs, and losing them would cost both a re-collection *and* the record of
what was collected.

Configs name the shards as `${REGRASP_DATA}/bc_dataset/...`, which
`handover_sim2real/regrasp/setup.py` expands at load and which defaults to the
in-repo `output/`, so nothing changes when running off the cluster. Set
`REGRASP_DATA=output` to keep everything in the repo, or point it anywhere else.

Note the base shards are on scratch too, so a purge costs ~4 h of re-collection —
`harvest_run.py` pulls back only the small metadata. Copy the shards aside if you
want to reuse them for run 3.

The stages below are the same work done by hand, and are what to read when a
stage fails.


### 1. Feasibility census (offline, seconds)

```bash
python examples/analyze_direction_feasibility.py --split train
```

The go/no-go gate. A bin no scene can reach is not a retry hypothesis.

### 2. Direction table (one sim pass, ~20 min on a laptop GPU)

```bash
python examples/build_direction_table.py --split train --out output/direction_table_train.json
```

Per scene: wrist, **observed** centroid, anchor frame, and for each bin the
goal-set grasp closest to its axis, with per-bin counts over the *full* goal set.

The `--cfg-file` must match what the demos will be collected with — the centroid
comes from the cloud, so the camera set and renderer change it.

### 3. Demo assignment (offline, seconds)

```bash
python examples/assign_direction_demos.py --table output/direction_table_train.json \
    --out output/regrasp_pins_train --mode per-bin --drop-bins='-z_beneath,-x_over_fingers'
```

Pure combinatorics, so `k`, the mode and the separation floor can all be
re-decided without touching the cluster. Writes the pin table and the exclusion
list.

`--mode per-bin` (run 2 on, the default) gives each scene **one demonstration for
every bin it can reach**, each the goal-set grasp *closest to that bin's axis* —
which is the vector the policy is commanded with, so the demonstration is as
close to the instruction as the goal set allows. On s0/train: 617 scenes,
**1596 demos**, mean 2.59 per scene, split +x 490 / +y 366 / −y 325 / +z 415.

`--mode pair` reproduces run 1: the single maximally-separated pair, 1088 demos.
Both break the scene→action confound; per-bin breaks it harder and, more
importantly, populates every bin's *evaluation* sample instead of only the two a
scene happened to draw.

### 4. Collect

```bash
python examples/collect_regrasp_demos.py --cfg-file examples/pretrain_multicam_wr.yaml --split train --grasp-pin-table output/regrasp_pins_train.json --output output/bc_dataset/train_regrasp.h5
```

~3.5 h for train at 1596 demos, ~5 min for val. Repeat with `--split val`.

### 5. Audit, and write the usable-pair list

```bash
python examples/audit_regrasp_demos.py --demos output/bc_dataset/train_regrasp.h5 \
    --write-ok output/regrasp_demos_train_ok.json
```

**The number to read is `informative steps`** — the fraction of steps where a
scene's demos actually differ. Run 1's: **median 1.000, mean 0.921**. A flat
profile there means the idea fails in the *data*, before any architecture
question.

`--write-ok` is not optional for run 2. It records the (scene, **bin**) pairs
collection actually demonstrated — dropping any where OMG could not plan, or
where the pin missed and the expert flew into a different sector. Point
`SIM.demo_ok_table` at it and those pairs leave **both** the collection pool and
the eval set.

Why that matters more than it did in run 1: the command is now the bin axis, so a
missed pin does not change what the policy is *told*, only what it is *shown* —
leaving an episode captioned `+z` whose trajectory comes in from elsewhere. It
cannot be re-binned either, because the scene already has a demonstration for the
bin it actually flew to. Without the filter the policy trains on three directions
and gets scored on four.

### 6. Train

```bash
# base fit only — answers "is the conditioning read?" in ~1 h
python examples/train_regrasp.py --cfg-file examples/configs/regrasp_run2.yaml --run-name regrasp_base --num-iters 0

# the full DAgger loop
python examples/train_regrasp.py --cfg-file examples/configs/regrasp_run2.yaml --run-name regrasp_run2
```

Resumable — re-run the same command. ~50 min/iteration on a laptop; ~13 h for the
whole of `regrasp_run2.yaml` on a DelftBlue V100 with 20 collection workers:

```bash
sbatch --time=20:00:00 --export=ALL,RUN=regrasp_run2,\
    SCRATCH_ROOT=/scratch/$USER/handover-sim2real examples/slurm/train_regrasp.sbatch
```

`regrasp_run2.yaml` is `regrasp_run1.yaml` with four changes and nothing else:
20 iterations instead of 25, `base_epochs` 40 → 50, `EVAL.every` 0 → 1, and
`EVAL.holdout` false → **true**. That last one means run 2's rates are genuinely
held out and are therefore **not comparable with run 1's absolute numbers** —
run 1 collected on its own eval scenes.

### 7. Score

Run 2 scores itself: `EVAL.every: 1` puts evaluation back inside the loop, where
Phase 4 had it. It costs ~300 s an iteration (50 scenes × the 1.58 directions the
average scene can supply ≈ 79 episodes), which is 1.7 h on a 13 h run — and it
means `EVAL.ckpt: best` actually selects on `success_rate` rather than on val loss,
because the metrics exist before the next iteration starts.

To split it out instead, set `EVAL.every: 0` and run the scorer alongside:

```bash
python examples/eval_regrasp_run.py --run-dir output/dagger_runs/regrasp_run2 --iters all
```

Writes `<run>/eval_log.csv`. Safe to run WHILE training continues — it only reads
what the trainer writes, and its own CSV is written atomically, so a concurrent
plot never sees a partial file. Add `--watch` to poll for new iterations, or
`--iters 0,5,10,15,19` for a subset if you want the trend sooner. The plotter
handles either layout.

### 8. Plot

```bash
python examples/plot_regrasp_run.py output/dagger_runs/regrasp_run2
```

Five figures, written into the run directory:

| file | what it shows |
|---|---|
| `training_curve.png` | 4×3, **one row per commanded direction**: success stages, chance vs conversion, approach error. The main figure — see below for why it is per bin |
| `curves_regrasp.png` | 2×3, **the one to read first**: is the policy using the command, `retry@k`, **ended in the commanded bin per bin**, per-bin success, per-bin tracking, arrived-from-the-commanded-side per bin |
| `debug_dagger.png` | 3×4 loop machinery: per-bin collection progress, per-bin failure profile, the pooled outcome stack, pin consistency, \|D\| growth, refit loss |
| `curves_diag.png` | 2×4 machinery health: label collapse, endgame, planner failures, β mixing, close timing, wall-clock split |
| `media_curves.png` | the presentation cut — the same panels pooled over directions, trimmed to five that fit on a slide |

A sixth, `test_set_evaluation.png`, comes from step 9 and is not written by this
script.

**Why the main figure is per bin.** A pooled `success_rate` averages four
physically different commands, so a policy that solves `+x` and ignores `+z`
plots identically to one that is mediocre at both — and telling those apart is
the entire question. The rows are the four directions this dataset can reach,
read from the log rather than assumed.

`curves_regrasp.png` only appears once eval metrics exist — the figure is gated
on `dir_track` being finite, so plotting a run with no scored iterations
silently gives you four figures instead of five. A panel whose columns are
missing entirely (an older log) says so in grey text rather than rendering an
empty axis that reads as a genuine zero.

The plotter splices `eval_log.csv` into the trainer's own log automatically
(`[plot] merged N rows`), reads only, and overwrites in place — so re-run it as
scoring progresses. Add `--show` to open the figures instead of only saving them.

Full-environment form, if you are not in an activated shell:

```bash
cd ~/h2r/handover-sim2real && GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet" python examples/plot_regrasp_run.py output/dagger_runs/regrasp_run2
```

### 9. Test-set evaluation (after training)

```bash
sbatch --export=ALL,RUN=regrasp_run2,CHAINED=1,"ITERS=0,5,10,15,20" examples/slurm/eval_regrasp_testset.sbatch
```

Writes `<run>/test_eval_log.csv` and `<run>/test_set_evaluation.png` — the
`training_curve.png` layout (one row per commanded direction) plus a
conditioning row: *ended in the commanded bin* per bin, *arrived from the
commanded side* per bin, and `retry@k` with the chained curve overlaid.

**This is the held-out number.** The in-loop curves run with
`EVAL.holdout: false` on the train split, so they are train-set rates by design
— a clean read of "is the policy getting better at what it is being taught",
and nothing more. This script scores the s0 **test** split, which the run has
never seen in any form.

**Two evaluations, and they answer different questions.** *Independent* (always)
rolls every (scene, bin) pair fresh from home; its `retry@k` is an OR over
independent attempts and therefore a **ceiling** — it assumes a failed attempt
costs nothing. *Chained* (`--chained`) runs the retry ladder for real: attempt 1
fails, the arm rewinds to 30 % of the trajectory it just flew, and attempt 2 is
commanded from there. The gap between the two curves is what the reset-based
version was giving away. Costs ~1.6× the independent sweep.

s0/test is 144 scenes, ~124 of which plan, so ~320 episodes an iteration: ~21 min
independent, ~35 min chained. `ITERS` is the first thing to set — five points
spread across the run show the trend for a tenth of the cost, and the CSV is
incremental so the rest can be filled in later. `--plot-only` re-renders from the
CSV with no simulator, which runs on a laptop.

Prerequisite — the test split needs its own tables, which is a GPU sim pass:

```bash
sbatch --export=ALL,SPLIT=test,OUT=output/direction_table_test.json examples/slurm/build_direction_table.sbatch
python examples/assign_direction_demos.py --table output/direction_table_test.json --out output/regrasp_pins_test --mode per-bin --drop-bins='-z_beneath,-x_over_fingers'
```

There is deliberately no `demo_ok_table` on test: nothing is collected there, so
the pin table itself is the feasibility statement.

### 10. Chained retry (single scenes, with rendering)

```bash
python examples/eval_regrasp_retry.py --run-dir output/dagger_runs/regrasp_run2 --scenes 10,12,40 --render
```

On failure, rewind to 30 % of the failed trajectory and command the next
direction from the ladder. `--render` draws the bin rays styled by state
(commanded bright, spent dim, available faint) and each attempt's path.

---

## Reading the results

| metric | means |
|---|---|
| **`dir_err`** | angle between the command and the achieved approach axis. The headline. |
| `dir_track` | `1 − dir_err/90°`. 1 follows, 0 ignores. |
| `bin_hit_rate` | arrived from the commanded side within 30°. Replaces `near_rate`. |
| `bin_diag_rate` | confusion-matrix diagonal. Collapsing to `1/k` = the policy goes the same way whatever it is told. |
| `cond_sep` | spread of what it DID over spread of what it was TOLD. |
| `chained_retry_at_k` | the regrasping headline. Saturates at k=4 here. |
| `cond_delta` | in the *training* log: does the action change when the command is shuffled. Non-zero at init; **decaying toward zero is the failure signature**. |

**`near_rate` is dead.** It measured distance to a pinned pose the policy is no
longer given, and reads ~0. That is not a regression.

---

## Gotchas

**`GraspPinTable.num_grasps` is a MIN and reads 1** on a Regrasp table, which
mixes 1- and 2-direction scenes. Use `max_grasps` to size arrays and
`num_grasps_for(scene)` to iterate. Getting this wrong silently discards every
paired second demonstration — the mechanism the whole design rests on.

**Shard completeness.** `DaggerHDF5Writer` writes `complete=False` at open and
`True` only on a clean close; the resume path requires it. Without that flag a
resume reuses a partial shard as if it were whole. Shards predating the flag have
no attr and are treated as complete.

**`attempted` is world vectors, never bin indices.** The anchor rotates with the
human, so a bin index names a different physical direction before and after.
Invisible under `YCB_MANO_START_FRAME: last` (static hand); appears on hardware.

**8 channels on disk, 7 into the model.** `d` is perturbed at training time, so
`d`-dependent channels cannot be baked into the file — the normals are stored and
the dot products are built at load.

**`EVAL.holdout`** decides whether the eval scenes are also collected on.
`regrasp_run1.yaml` inherited `false` from Phase 5, so every rate that run
produced is a train-set rate and optimistic. `regrasp_run2.yaml` sets it `true`,
which is why the two runs' absolute numbers are not comparable.

---

## Run 1

19 of 25 DAgger iterations (stopped by choice), 1087 base episodes / 20386 steps.
`m ≈ 354` for iterations 1–5 and `≈ 176` for 6–19 — a deliberate budget
correction, which is why `D_steps` changes slope at iteration 6.

Base fit: `val_total` best **0.1851 at epoch 16**, degrading to 0.239 by 99 with a
4.9× train/val gap — hence `base_epochs: 40`. `cond_delta` **rose 0.73 → 1.38** on
held-out val, so the conditioning is read and generalises.

`DATA.reach_tail` went 10 → **5** on measurement: the divergence profile is flat
(0.083 over the last ten steps vs 0.093 over 10–15, a ratio of **0.89×**). A wider
window only pays when the demonstrations share a free approach and diverge into
the reach; two directions come in from opposite sides from step 0, so weighting
the endgame was starving the free approach where obstacle avoidance is learned.

**Headline: `success_rate` 0.210 (first 5) → 0.203 (last 5)**, a change of −0.008
and well inside the ±0.115 noise floor. The readable finding is the pair
`bin_diag_rate ≈ 0.50` against a chance of 0.25 while `bin_hit_rate` sits **at**
chance: the policy orients as told and then approaches from wherever it likes.

---

## Run 2 (configured, not yet run)

`examples/configs/regrasp_run2.yaml`. Same architecture, same conditioning, same
data — run 1's ambiguity was about the RUN, not the method, and run 2 removes the
four reasons it could not be read:

| | run 1 | run 2 | why |
|---|---|---|---|
| `m` | 354 → 176 mid-run | uniform | `num_grasps` is a MIN and read 1 on a mixed table; fixed with `max_grasps`, so \|D\| no longer changes slope at iteration 6 |
| eval | pooled over 4 directions | **per bin** | "solves +x, ignores +z" and "mediocre at both" gave the same number |
| `EVAL.holdout` | `false` | **`true`** | run 1 collected on its own eval scenes; its rates are train-set rates |
| iterations | stopped at 19/25 | 20, planned | flat from iteration 6 on, and the marginal iteration is ~40 min of refit against a change inside the noise floor |

`base_epochs` also goes 40 → 50 (margin, not need — the measured val minimum is
epoch 16) and `EVAL.every` 0 → 1, which puts scoring back inside the loop and
makes `EVAL.ckpt: best` select on `success_rate` rather than on val loss.

**Run 2's absolute rates will be lower than run 1's and that is not a
regression** — `holdout: true` is a different measurement. Compare trends and
per-bin spread, not levels.

The pre-registered branch: if `dir_track` stays near its floor with
`bin_diag_rate` at 0.25 across all four rows, the per-point channels are being
washed out by the SA-layer max-pooling, and run 3 adds a small global-`d` branch.
The gap between runs 2 and 3 is then the measurement of how much the pooling
destroys.

---

## Deferred

**A retreat controller.** The chained rewind resets the simulator, which rewinds
the human's DexYCB playback too — so `chained_retry_at_k` is still an upper
bound, just a much tighter one than the reset-based `retry_at_k`. The honest
version drives the arm back along its own joint path without resetting.

**A held-out BIN split** — train on three bins, evaluate on the fourth. This is
the only real test of whether the conditioning *interpolates*, and therefore of
the claim that makes `k` a test-time knob worth anything.
