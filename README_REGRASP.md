# Regrasp — command reference

Every script in the Regrasp phase, with its arguments. Design notes, results and
rationale live in [`docs/run_index.md`](docs/run_index.md) and in the config
headers under `examples/configs/regrasp_run*.yaml`.

To execute a specific run end to end on the cluster, follow its runbook instead —
those carry the gates, expected outputs and failure modes this file does not:
[`docs/runbook_run9_run10.md`](docs/runbook_run9_run10.md).

Code lives in `handover_sim2real/regrasp/` and `handover_sim2real/regrasp_bc/`.

Prefix any command with the environment if you are not in an activated shell:

```bash
cd ~/h2r/handover-sim2real
export GADDPG_DIR=$PWD/GA-DDPG OMG_PLANNER_DIR=$PWD/OMG-Planner
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet"
```

---

## 0. Whole pipeline (DelftBlue)

```bash
bash examples/slurm/regrasp_pipeline.sh
bash examples/slurm/regrasp_pipeline.sh --dry-run
bash examples/slurm/regrasp_pipeline.sh --force
bash examples/slurm/regrasp_inventory.sh
```

`SCRATCH_ROOT` (default `/scratch/$USER/handover-sim2real`) holds run dirs and
HDF5 shards; JSON tables stay in the repo.

---

## 1. Feasibility census

```bash
python examples/analyze_direction_feasibility.py --split train
```

`--split {train,val,test}`

## 2. Direction table

```bash
python examples/build_direction_table.py --split train --out output/direction_table_train.json
python examples/build_direction_table.py --split train --members-per-bin 5 --out output/direction_table_train.json

# run 10: d from the grasp's POSITION (centroid -> fingertip), not its orientation
python examples/build_direction_table.py --split train --members-per-bin 5 \
    --d-rule grasp_offset --d-min-offset 0.02 \
    --out output/direction_table_train_off.json
```

`--cfg-file` `--split` `--out` `--num-scenes` `--start` `--k` `--max-angle`
`--valid-grasp-dict` `--members-per-bin` `--d-rule {approach_axis,grasp_offset}`
`--d-point-depth` `--d-min-offset` `--egl` `--seed`

`--d-rule` decides what a bin *means* and is written into the table's `_meta`;
every later stage inherits it and refuses a mismatch. Under `grasp_offset` all
six bins are populated (`−z` and `−x` are dead only under `approach_axis`), so
read the printed histogram before assigning or collecting.

```bash
sbatch --export=ALL,SPLIT=test,OUT=output/direction_table_test.json examples/slurm/build_direction_table.sbatch
```

## 3. Demo assignment

```bash
python examples/assign_direction_demos.py --table output/direction_table_train.json \
    --out output/regrasp_pins_train --mode per-bin --drop-bins='-z_beneath,-x_over_fingers'

python examples/assign_direction_demos.py --table output/direction_table_train.json \
    --out output/regrasp_pins_train_p3 --per-bin 3 --drop-bins='-z_beneath,-x_over_fingers'
```

```bash
# run 10: NO --drop-bins — under grasp_offset all six bins carry demonstrations
python examples/assign_direction_demos.py --table output/direction_table_train_off.json \
    --out output/regrasp_pins_train_off --mode per-bin
```

`--table` `--out` `--min-bins` `--max-angle` `--drop-bins` `--min-sep-deg`
`--mode {per-bin,pair}` `--per-bin` `--dry-run`

Writes `<out>.json` and `<out>_excluded.json`. The table's `_meta` (including
`d_rule`) rides forward into the pin table.

## 4. Collect

```bash
python examples/collect_regrasp_demos.py --cfg-file examples/pretrain_multicam_wr.yaml \
    --split train --grasp-pin-table output/regrasp_pins_train.json \
    --output output/bc_dataset/train_regrasp.h5

python examples/collect_regrasp_demos.py ... --shard 0/4
```

`--cfg-file` `--output` `--split` `--num-scenes` `--num-episodes` `--seed`
`--valid-grasp-dict` `--grasp-pin-table` `--shard i/n` `--grasps-per-scene`
`--command {bin_axis,bin_centroid,grasp_axis}`
`--d-rule {approach_axis,grasp_offset}` `--d-point-depth`
`--freeze-partial-pointcloud` `--freeze-at-step` `--egl`

`--command` must match `SIM.command_deploy` in the run config that trains on the
shard. Ignored by a run whose learner sets `DATA.d_source: d_grasp_world`.
`--d-rule` defaults to the pin table's own `_meta`, which is what you want —
pass it only to override a table with no recorded rule.

```bash
sbatch --export=ALL,SPLIT=train examples/slurm/collect_regrasp_demos.sbatch
```

Measured: 3.62 s/episode on this laptop, 5.57 s/episode on a V100 node (serial).

## 5. Audit

```bash
python examples/audit_regrasp_demos.py --demos output/bc_dataset/train_regrasp.h5 \
    --write-ok output/regrasp_demos_train_ok.json
```

`--demos` `--write-ok`

## 5b. Per-(scene, grasp) success table

```bash
python examples/build_demo_table.py --demos output/bc_dataset/train_regrasp.h5
python examples/build_demo_table.py --demos output/bc_dataset/val_regrasp.h5
```

`--demos` (one or more shards) `--out` `--criterion {reach,close_label,caption,all}`
`--by {grasp,bin}` `--pos-thresh` `--rot-thresh` `--no-detail`

Writes `output/bc_dataset/tables/<stem>_demo_success.csv` — rows are scenes,
columns are grasp indices, `1` where the demonstration succeeded and `0` where it
did not. **Blank means the pair was never collected**, which is a different fact
from a failure:

```
scene,0,1,2,3
32,1,1,1,1
52,1,1,0,1
10,1,0,,
```

Columns are grasp indices rather than bins because a bin does not identify a
demonstration — run 3's `--per-bin 3` gives each bin three grasps with three
outcomes. `--by bin` gives the bin view, where a cell is the AND over that bin's
grasps.

**`--criterion reach` (the default) is not what `demo_ok_table` measures.** The
audit asks whether the *caption* is valid (`bin_realized == bin_assigned`, pin
landed) and passes **1575 of 1596**. This asks whether the expert actually
reached the grasp it aimed at, and **1116 of 1596 (69.9%)** did — the other 480
stop a median 89 mm short and run 14 steps against 21. Both are real filters for
different jobs, and the gap between them is the 30% of the base set that is in
`D` while demonstrating nothing.

`--criterion close_label` uses the expert's own final CLOSE label instead of the
geometry; the two agree on **99.6%** of episodes, which is why the number is
believable rather than a threshold artifact.

The companion `*_demo_success_detail.csv` carries one row per episode — scene,
grasp index, bin (as an index and as an `axis` label: `+x`, `+y`, `-y`, …),
terminal position and rotation error, step count and every criterion — so any `0`
in the matrix can be traced.

## 5c. Demos per direction

```bash
python examples/analyze_demo_bins.py
python examples/analyze_demo_bins.py --by realized
python examples/analyze_demo_bins.py output/bc_dataset/tables/*_detail.csv
```

Positional: one or more `*_detail.csv` (default: the train table; several are
merged). `--criterion {reach,close_label,caption,all}` `--by {assigned,realized}`

Per direction: demonstrations, distinct scenes, how many reached, and the rate.
This is the question the success matrix **cannot** answer, because a column there
is a SLOT and slots pack densely over whichever bins a scene reaches — `grasp_idx`
1 is `+y` on one scene and `+z` on another, so counting columns counts something
else. Reads the CSV from 5b rather than the shard, so it needs no h5py and runs on
a login node.

`--by realized` regroups on where each demo actually went instead of what it was
commanded; the two differ exactly on the episodes where the pin was refused
(19 of 1596 on run 2's base set).

Bins with no demonstrations print at zero rather than being omitted — they are
directions the policy will EXTRAPOLATE into if the feasibility mask ever admits
them at test time. Under a `--drop-bins='-z_beneath,-x_over_fingers'` table that
is `-x` and `-z`, and the trailing line reports the live count that caps
`chained_retry_at_k`.

## 5d. The reach filter (on by default)

`demo_ok_table` asks whether a demonstration's **caption** is honest and passes
**1576/1596**. It says nothing about whether the expert arrived. `SIM.reach_filter`
asks that second question — terminal pose within `close_pos_thresh` /
`close_rot_thresh` of the target grasp — and passes **1116/1596 (69.9%)**.

```yaml
SIM:
  reach_filter: true        # default; false reproduces runs 1-9 exactly
  reach_pos_thresh: 0.02    # m,   mirrors DAGGER.close_pos_thresh
  reach_rot_thresh: 0.34    # rad, mirrors DAGGER.close_rot_thresh
```

**No config edit is needed** — it defaults on, so every run from here forward gets
it. It applies in three places from one criterion
([`handover_sim2real/regrasp/reach.py`](handover_sim2real/regrasp/reach.py)):

| where | mechanism |
|---|---|
| DAgger collection | `pin_table.keep_only(reach_ok_pairs(base_train_h5))` in `train_regrasp.py` |
| in-loop eval | same prune — `eval_jobs` reads slot counts from that table |
| training data | `BCDataset`, per episode, via `dataset.episode_status` |
| normalization | `compute_normalization_stats`, same predicate |
| `D_episodes` / `D_steps` | `dataset_size(files, log_filt)`, same predicate |

`episode_status` is the single predicate all four of the non-table consumers route
through. They diverged once: the normalizer was fit over every episode in the
shard while the dataset trained on a filtered subset, so the head's output scale
was defined by data the model never saw. At 19 miscaptioned episodes that was
noise; at 479 unreached ones it is 30% of the shard, and the dropped episodes are
systematically different — truncated mid-approach, so their labels skew toward
large "keep approaching" deltas and away from small reach-phase ones. Refitting
moves `action_std` down 2–8% per axis and `state_std` on the EE pose up 3–7%.

The table prune alone is **not** enough: the base shard on disk still holds the
episode and the loader reads every episode in the file. That is the same leak the
caption filter had before it grew a loader-side twin (`D_episodes` 1596, not
1575). Both halves, or neither.

Measured effect on `regrasp_pins_train.json`:

```
raw              617 scenes  1596 pairs
after demo_ok    617 scenes  1576 pairs
after reach      558 scenes  1097 pairs     <- 30% of the pairs, 59 whole scenes
slots per scene  1:222  2:175  3:119  4:42

D (train_regrasp.h5)   1596 ep / 30028 steps  ->  1098 ep / 23048 steps
```

A warm-started or resumed run **keeps its existing normalizer** (`train_regrasp.py`
loads rather than recomputes), so turning this on mid-chain does not silently
rescale a head. Only a fresh base fit refits.

**Two consequences to plan around.** Unpaired scenes go from 127/617 (21%) to
222/558 (40%), so a larger share of the data cannot break the conditioning
confound — check `cond_delta` on the paired subset rather than assuming it holds.
And mean pairs per scene falls 2.59 → 1.97, so `episodes_per_iter` (divided by
`max_grasps`, still 4) yields fewer episodes per iteration than before; re-read
the collection log's episode count before comparing wall clock with an earlier run.

`val_loss` is comparable across runs that share this setting and **not** against
runs 1-9, whose val set carried the unreached demonstrations.

## 6. Check a config's inputs

```bash
REGRASP_DATA=/scratch/$USER/handover-sim2real/output \
  python examples/check_regrasp_inputs.py regrasp_run3 regrasp_run4 regrasp_run5
```

Positional: config names, no path and no `.yaml`. Env: `REGRASP_DATA`.

## 7. Train

```bash
# base fit only
python examples/train_regrasp.py --cfg-file examples/configs/regrasp_run2.yaml \
    --run-name regrasp_base --num-iters 0

# full DAgger loop
python examples/train_regrasp.py --cfg-file examples/configs/regrasp_run2.yaml \
    --run-name regrasp_run2 --num-workers 8
```

`--cfg-file` `--run-name` `--out-root` `--num-iters` `--episodes-per-iter`
`--base-run` `--device` `--seed` `--num-workers` `--worker-device` `--no-eval`

```bash
sbatch --time=24:00:00 \
  --export=ALL,RUN=regrasp_run7,CFG=examples/configs/regrasp_run7.yaml,SCRATCH_ROOT=/scratch/$USER/handover-sim2real \
  examples/slurm/train_regrasp.sbatch
```

sbatch vars: `RUN` `CFG` `SCRATCH_ROOT` `OUT_ROOT` `REGRASP_DATA` `BASE`
`WORKERS` `EXTRA`. Resumable — resubmit the identical command.

## 8. Score (only when `EVAL.every: 0`)

```bash
python examples/eval_regrasp_run.py --run-dir output/dagger_runs/regrasp_run2 --iters all
```

`--run-dir` `--out` `--iters` `--num-scenes` `--ckpt {best,last}` `--device`
`--seed` `--force` `--watch` `--poll-s` `--timeout-s` `--publish-best`

## 9. Test-set evaluation

```bash
python examples/eval_regrasp_testset.py --run-dir output/dagger_runs/regrasp3_fast1 \
    --iters 4 --ckpt last --chained
```

`--run-dir` `--out` `--iters` `--split {test,val,train}` `--pin-table`
`--exclude-scenes` `--demo-ok-table` `--num-scenes` `--ckpt {best,last}`
`--chained` `--rewind-frac` `--max-attempts` `--device` `--seed` `--force`
`--plot-only` `--pos-thresh` `--rot-thresh`

```bash
sbatch --export=ALL,RUN=regrasp_run2,CHAINED=1,"ITERS=0,5,10,15,20" examples/slurm/eval_regrasp_testset.sbatch
```

Needs the test tables first — steps 2 and 3 with `--split test`.

---

## Plotting

```bash
python examples/plot_regrasp_run.py output/dagger_runs/regrasp_run2
```

Writes `training_curve.png`, `curves_regrasp.png`, `debug_dagger.png`,
`curves_diag.png`, `media_curves.png`. `--show` opens them.

```bash
python examples/plot_regrasp_fits.py output/dagger_runs/regrasp_run2
```

Per-epoch train/val for every refit → `fit_curves.png`.
`--metric` `--iters 0,4,8` `--out` `--logy`

```bash
python examples/plot_regrasp_bin_spread.py \
    output/regrasp_pins_train.json output/regrasp_pins_train_p3.json \
    --labels run2,run3 --bins-deg 3.75
```

Within-bin angular spread of the assigned grasps — to the fixed bin axis (row 1)
and to each bin's empirical centroid (row 2) → `output/bin_spread.png`.
`--labels` `--out` `--bins-deg`

```bash
python examples/status_regrasp_run.py output/dagger_runs/regrasp_run2
watch -n 300 python examples/status_regrasp_run.py output/dagger_runs/regrasp_run2
```

```bash
bash examples/watch_regrasp.sh -f slurm_logs/regrasp_<jobid>.out
```

---

## Rendering and replay (PyBullet GUI)

### One iteration's policy, one scene, one bin

```bash
python examples/rollout_regrasp_policy.py \
    --run-dir output/dagger_runs/regrasp_run2/iters/iter_13 \
    --cfg-file examples/pretrain_multicam_wr.yaml \
    --grasp-pin-table output/regrasp_pins_train.json \
    --scene 32 --bin 4 --show-goal-grasp
```

`--run-dir` takes any iteration directory (`<run>/iters/iter_NN`) or `<run>/best`
/ `<run>/last`; there is no checkpoint flag — it loads `checkpoints/best.pt` and
falls back to `last.pt`. GUI is on by default; `--no-render` disables it.

**Use `--bin` for a rollout.** A rollout does not replay anything — it issues a
command, and the command is built from the bin and the anchor alone:
`to_world(axes[bin], anchor_R)`, where `axes` is the bin axes or, under
`SIM.command_deploy: bin_centroid` (run 9), the bins' empirical centroids. So
the bin *fully determines* what the policy is told, and `--bin` is the only way
to issue the same command across scenes — slot 1 is `+y` on one scene and `−y`
on another.

Where a bin holds several grasps (a `--per-bin 3` table: scene 32's `+x` is
slots 0, 4 and 8) the command is identical for all of them; the slot only
decides which grasp `--show-goal-grasp` draws and which pose gets pinned for
scoring. The first — the closest to the bin axis — is used. `--grasp-idx` still
names an exact grasp when that matters, and it is the *only* selector that
identifies a recorded demonstration (see the replay section below).

`--bin` needs `--grasp-pin-table`, and errors with the scene's actual bin list
when that scene has no demonstration for the bin.

To see what a scene offers before rolling it:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from handover_sim2real.regrasp.grasp_pin import GraspPinTable
from handover_sim2real.regrasp import directions as D
t=GraspPinTable('output/regrasp_pins_train.json'); s=32
print(', '.join(f'--bin {t.bin_of(s,g)} ({D.BIN_SHORT[t.bin_of(s,g)]}) = slot {g}'
                for g in range(t.num_grasps_for(s))))"
```

`--run-dir` `--cfg-file` `--scene` `--scenes 10,12,40` `--num-scenes`
`--scenes-from-run` `--max-steps` `--hold-steps` `--dwell-steps` `--device`
`--no-render` `--benchmark` `--show-goal-grasp` `--show-grasp-set`
`--bin N` `--grasp-idx N` `--all-grasps` `--show-pred-grasp` `--grasp-pin-table`
`--command {bin_axis,bin_centroid,grasp_axis}` `--show-anchor-frame`
`--show-bin-sphere` `--bin-sphere-radius` `--bin-sphere-points`
`--freeze-partial-pointcloud` `--freeze-at-step` `--egl`

`--command` must match the run's `SIM.command_deploy` (`bin_axis` for runs 2–8,
`bin_centroid` for run 9, `grasp_axis` for run 1) or you are watching a
different command than the one that was scored. The resolved vectors are in
`<run>/command_axes.json`.

Conditioning overlays (both need `--grasp-pin-table`):

```bash
python examples/rollout_regrasp_policy.py \
    --run-dir output/dagger_runs/regrasp_run2/iters/iter_13 \
    --cfg-file examples/pretrain_multicam_wr.yaml \
    --grasp-pin-table output/regrasp_pins_train.json \
    --scene 32 --grasp-idx 1 \
    --show-anchor-frame --show-bin-sphere --show-goal-grasp
```

`--show-anchor-frame` draws x/y/z at the object centroid (x away from the giver's
wrist, z world up); `--show-bin-sphere` draws a point shell coloured by bin with
a labelled ray down each bin axis, colours matching `plot_regrasp_run.py`. Both
read `anchor_R` / `centroid_world` from the pin table's `scene_meta`.

Scenes with all four live bins under the run-2 table: **32, 52, 91, 92, 94**
(slots `0=+x 1=+y 2=−y 3=+z`; under a `--per-bin 3` table the same four bins
occupy 12 slots, `0–3`, `4–7`, `8–11`).

### Replay recorded demonstrations — base shard or a run's DAgger buffer

Same tool for both. The base shard `output/bc_dataset/train_regrasp.h5` holds the
expert demonstrations; every DAgger iteration writes `<run>/data/dagger_iter_NN.h5`
holding the episodes that iteration collected. Replaying drives the robot through
the *recorded* rollout and overlays the saved point cloud, so it shows what
actually happened, not a fresh rollout.

```bash
# base demonstrations — scene 32, its 4th grasp
python examples/visualize_bc_dataset.py \
    --dataset output/bc_dataset/train_regrasp.h5 \
    --mode replay --cfg-file examples/pretrain_multicam_wr.yaml \
    --grasp-pin-table output/regrasp_pins_train.json \
    --scene 32 --grasp-idx 3 --show-goal-grasp --show-expert-arrows

# one iteration's DAgger buffer — same selectors
python examples/visualize_bc_dataset.py \
    --dataset output/dagger_runs/regrasp_run2/data/dagger_iter_13.h5 \
    --mode replay --cfg-file examples/pretrain_multicam_wr.yaml \
    --grasp-pin-table output/regrasp_pins_train.json \
    --scene 52 --grasp-idx 3 --show-goal-grasp --show-expert-arrows
```

#### The conditioning overlay — anchor frame, bin sphere, and `d`

Add `--show-anchor-frame --show-bin-sphere --show-d` to any replay. All three are
drawn at the **observed object centroid**, recomputed from the episode's own
step-0 cloud exactly as the collector built it — episodes store `anchor_R` but not
the origin it is pinned at, and the object's pose would sit a few cm from the
visible-surface centroid the conditioning channels actually use.

`--show-d` draws three vectors, because they are not the same thing and drawing
only one is how you convince yourself the conditioning is fine when it is not:

| colour | vector |
|---|---|
| white | what the episode was **commanded** (`d_world`) — what the network read |
| yellow | this shard's `d_rule` applied to the grasp it **flew**; under run 10 that is `grasp_offset`, centroid → fingertip midpoint, with the offset segment and endpoint drawn |
| grey | `−R_grasp[:,2]` (`d_grasp_world`), runs 1–9's `approach_axis`, for contrast |

The rule is read from `_meta.d_rule` in `--grasp-pin-table`, so a run-10 table
draws `grasp_offset` and a run-2 table draws `approach_axis` with no flag.
Override with `--d-rule / --d-point-depth / --d-min-offset`. The angles between
the vectors are printed, and under `grasp_offset` so is the centroid → fingertip
offset with a warning when it falls below `d_min_offset` — that is the case where
`d` is centroid noise rather than geometry.

```bash
# run 10's demos, with the full conditioning overlay
python examples/visualize_bc_dataset.py \
    --dataset $RUNS/output/bc_dataset/train_regrasp_off.h5 \
    --mode replay --cfg-file examples/pretrain_multicam_wr.yaml \
    --grasp-pin-table output/regrasp_pins_train_off.json \
    --scene 32 --grasp-idx 0 \
    --show-goal-grasp --show-anchor-frame --show-bin-sphere --show-d
```

The same overlays are available on `rollout_regrasp_policy.py`; both scripts draw
them from `handover_sim2real/regrasp/viz.py`, so a bin is the same colour in both
views and in every figure.

`--scene` + `--grasp-idx` resolve to the flat `--episode` index and print what
they picked:

```
[select] scene 32 grasp_idx 3 bin 4 (+z) -> --episode 56
```

When nothing matches, the error says what the shard *does* hold — for the scene
if it's present, and the scene list if it isn't (a DAgger shard carries only the
~100 scenes its iteration drew, not the whole split).

`--bin` is available as a filter for "anything in this direction", but where
several grasps share the bin it lists them all and takes the first:

```
[select] 3 episodes match; taking the first. All of them:
    --episode     0  scene    0  grasp_idx  0  bin 0 (+x)  steps  14
    --episode     1  scene    0  grasp_idx  1  bin 0 (+x)  steps  21
    --episode     2  scene    0  grasp_idx  2  bin 0 (+x)  steps  21
```

> **Select a replay by GRASP INDEX, never by bin.** A bin does not identify a
> demonstration. Run 2's table gives each bin one grasp, but run 3's `--per-bin 3`
> gives it three — on scene 32, `+x` is slots 0, 4 *and* 8, three different grasp
> poses with three different trajectories. The episode's `grasp_idx` attr is the
> slot it flew, and that is the definitive selector; `bin_assigned` is context.
> (For a *rollout* the opposite holds — see the previous section: there the bin
> is what determines the command, and no grasp is being replayed at all.)

`--grasp-pin-table` makes `--show-goal-grasp` draw **this episode's** grasp: the
replay pins the episode's own `grasp_idx` before reading the pose back from OMG.
Without the table, OMG re-selects its goal by `argmin` over the goal set and the
overlay can land on a different grasp than the episode aimed at — measured at up
to 20.5 cm away. Each loaded episode prints its `scene_idx`, `grasp_idx` and
`bin`, so the overlay can be checked against the caption.

`--replay-source states` (the default) follows the stored `robot_states`, which
is the **policy's** path for DAgger data and the only mode where the cloud lines
up. `--replay-source omg` re-plans the expert instead and matches the *offline*
set only — on DAgger data it shows a different trajectory than the one recorded.

Keys in the window: `R` replay, `N` / `P` next / previous episode (reloads that
episode's scene and grasp overlay), `Q` quit.

`--dataset` `--scene N` `--grasp-idx N` `--bin N` `--episode N`
`--mode {static,replay}` `--cfg-file` `--replay-source {states,omg}`
`--show-expert-arrows` / `--no-expert-arrows` `--arrow-scale` `--show-goal-grasp`
`--show-grasp-set` `--max-grasp-set` `--valid-grasp-dict` `--grasp-pin-table`
`--seed`

`--mode static` needs no simulator — it plots the clouds and trajectory with
matplotlib, which is the fast way to check a shard is intact.

**`--episode` is a position in the file** and says nothing about what the
episode is — `--scene` / `--grasp-idx` above exist so you never have to translate
by hand. To browse what a shard contains before picking one:

```bash
python -c "
import h5py, collections, sys; sys.path.insert(0,'.')
from handover_sim2real.regrasp import directions as D
f=h5py.File('output/dagger_runs/regrasp_run2/data/dagger_iter_13.h5','r')
ks=[k for k in f if k.startswith('episode_')]
n=collections.Counter(int(f[k].attrs.get('bin_assigned',-1)) for k in ks)
g=collections.Counter(int(f[k].attrs['grasp_idx']) for k in ks)
print(len(ks),'episodes over',
      len({int(f[k].attrs['scene_idx']) for k in ks}),'scenes')
print('  by bin  :', {D.BIN_SHORT[b]: c for b,c in sorted(n.items()) if b>=0})
print('  by slot :', dict(sorted(g.items())))"
```

```
268 episodes over 100 scenes
  by bin  : {'+x': 82, '+y': 65, '-y': 54, '+z': 67}
  by slot : {0: 100, 1: 86, 2: 58, 3: 24}
```

### One scene through the retry ladder

```bash
python examples/eval_regrasp_retry.py \
    --run-dir output/dagger_runs/regrasp_run2 \
    --scenes 10,12,40 --render
```

`--run-dir` `--policy-dir` `--ckpt {best,last}` `--out` `--rewind-frac`
`--rewind-mode` `--budget` `--max-attempts` `--num-scenes` `--scenes`
`--replay-tol` `--save-clouds` `--device` `--seed` `--quiet`

Render group: `--render` `--pace` `--replay-pace` `--pause-s` `--show-cloud`
`--no-path` `--no-grasp-markers` `--no-step`

---

## Local laptop pipeline

```bash
bash examples/run_regrasp3_local.sh
```

Stages T/A/V/C/D/E, fail-fast, idempotent.

## Harvest a run off scratch

```bash
python examples/harvest_run.py --run-dir /scratch/$USER/handover-sim2real/output/dagger_runs/regrasp_run2
```
