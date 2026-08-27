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

### One iteration's policy on one scene

```bash
python examples/rollout_regrasp_policy.py \
    --run-dir output/dagger_runs/regrasp_run2/iters/iter_13 \
    --cfg-file examples/pretrain_multicam_wr.yaml \
    --scene 10 --show-goal-grasp
```

`--run-dir` takes any iteration directory (`<run>/iters/iter_NN`) or `<run>/best`
/ `<run>/last`; there is no checkpoint flag — it loads `checkpoints/best.pt` and
falls back to `last.pt`. GUI is on by default; `--no-render` disables it.
`--grasp-idx` picks which commanded direction (bin slot) to roll out.

`--run-dir` `--cfg-file` `--scene` `--scenes 10,12,40` `--num-scenes`
`--scenes-from-run` `--max-steps` `--hold-steps` `--dwell-steps` `--device`
`--no-render` `--benchmark` `--show-goal-grasp` `--show-grasp-set`
`--grasp-idx N` `--all-grasps` `--show-pred-grasp` `--grasp-pin-table`
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

Scenes with all four live bins: **32, 52, 91, 92, 94** (`0=+x 1=+y 2=−y 3=+z`).

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
