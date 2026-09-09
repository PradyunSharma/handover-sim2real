#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RUN 17, EVERY STAGE, IN ONE JOB — and everything on /home, not /scratch.
#
#     sbatch examples/slurm/regrasp_run17_all.sbatch
#
# IT WILL NOT FINISH IN ONE 24 h ALLOCATION, and that is not a bug in the
# script. Measured from run 11's own `wall_s` (fit `wall(i) = 0.78*i + 6.6 min`)
# and scaled for run 16's ~7x larger eval:
#
#     direction table train    1.5 h
#     direction table val      0.2 h
#     collect train            5.0 h
#     collect val              0.3 h
#     train, 25 iterations    16.1 h
#     ------------------------------
#     TOTAL                   23.1 h        DelftBlue hard maximum: 24 h
#
# 0.9 h of margin on a 23 h estimate is not margin. So this script is IDEMPOTENT
# instead: every stage is skipped when its output already exists, and
# `train_regrasp.py` resumes from `state.json`. Submit it twice, chained, and
# pass 2 picks up wherever pass 1 was killed:
#
#     J1=$(sbatch --parsable examples/slurm/regrasp_run17_all.sbatch)
#     sbatch --dependency=afterany:$J1 examples/slurm/regrasp_run17_all.sbatch
#
# `afterany`, NOT `afterok`: pass 1 is EXPECTED to die on the wall clock, which
# is a non-zero exit. Under `afterok` pass 2 would never start.
#
# WHY ONE JOB IS WORSE THAN FOUR, in case the trade is ever revisited. Each stage
# wants different resources — the table build is one GPU and 4 cores, collection
# is CPU-wide, training is GPU plus workers — so a single allocation has to ask
# for the maximum any stage needs and waste the rest for the other 80% of the
# time. The staged version in the run-16 section of docs/run_index.md also fails
# cheaper: a bad direction table costs 1.5 h there and 6.5 h here, because here
# it is only discovered after collection has already run against it.
#
# ── EVERYTHING ON /home ──────────────────────────────────────────────────────
# `SCRATCH_ROOT` is a misnomer inherited from train_regrasp.sbatch: it sets both
# `OUT_ROOT` (the run directory) and `REGRASP_DATA` (the HDF5 shards), and it can
# point anywhere. Here it points into $HOME.
#
# ⚠ /home IS A HARD 30 GB QUOTA THAT FILLS SILENTLY. There is no ENOSPC
# traceback — Python cannot write one to a full disk — so the job dies with exit
# code 6 and an empty error file. Run 17 needs about 5 GB:
#
#     base shards (train 570 MB + val 23 MB)          ~0.6 GB
#     run dir: data/ (25 DAgger shards)               ~3.1 GB
#     run dir: iters/ (per-iteration checkpoints)     ~1.2 GB
#     direction + pin tables                          ~0.003 GB
#
# `preflight_regrasp.sh regrasp_run17` checks the quota BEFORE any of this starts. Do not skip
# it: losing 6 h of collection to a full disk is the single most expensive
# failure available here, and it is the one that reports nothing.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-rg17
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/rg17_%j.out
#SBATCH --error=slurm_logs/rg17_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2023r1 || true
module load cuda/11.7 || true
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pch2r_dev}"
export PYTHONUNBUFFERED=1
export GADDPG_DIR="${GADDPG_DIR:-$PWD/GA-DDPG}"
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-$PWD/OMG-Planner}"

# ── the one knob: where the big files live ──────────────────────────────────
# NOT /scratch. /scratch is purged by age; the ask was /home. Both the run dir
# and the shards follow this.
export SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/h2r-runs}"
export REGRASP_DATA="${REGRASP_DATA:-$SCRATCH_ROOT/output}"
export OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT/output/dagger_runs}"
mkdir -p "$REGRASP_DATA/bc_dataset" "$OUT_ROOT" slurm_logs output

RUN=regrasp_run17
CFG=examples/configs/regrasp_run17.yaml
SIM_CFG=examples/pretrain_multicam_wr.yaml

# `d_rule`, its two `location_extent` parameters, the DEPTH and THE ANCHOR
# REFERENCE all have to match examples/configs/regrasp_run17.yaml exactly, or
# `resolve_d_rule` / `resolve_anchor_ref` refuse the run.
D_ARGS="--d-rule location_extent --d-point-depth 0.1034 --d-min-offset 0.02 --d-m-min 0.15 --d-extent-pct 95.0 --anchor-hand-ref base"

DT_TRAIN=output/direction_table_train_loc.json
DT_VAL=output/direction_table_val_loc.json
PIN_TRAIN=output/regrasp_pins_train_loc.json
PIN_VAL=output/regrasp_pins_val_loc.json
H5_TRAIN="$REGRASP_DATA/bc_dataset/train_regrasp_loc.h5"
H5_VAL="$REGRASP_DATA/bc_dataset/val_regrasp_loc.h5"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "SCRATCH_ROOT=$SCRATCH_ROOT   REGRASP_DATA=$REGRASP_DATA"
df -h "$HOME" | tail -1

stage () { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

# ── 1. direction tables ─────────────────────────────────────────────────────
# Run 17 changes BOTH `d_rule` (grasp_offset -> location_extent) and the anchor
# reference (hand_centroid -> base), so nothing upstream of run 16 or run 18 is
# reusable. `d_rule` decides which grasp each bin gets and the frame decides what
# the bins ARE; `resolve_d_rule` and `resolve_anchor_ref` both refuse a stale
# table, so a mismatch fails immediately rather than after the collection.
if [ ! -f "$DT_TRAIN" ]; then
    stage "1a  direction table (train)  ~1.5 h"
    python examples/build_direction_table.py --cfg-file "$SIM_CFG" \
        --split train --out "$DT_TRAIN" $D_ARGS
else
    echo "[skip] $DT_TRAIN exists"
fi

if [ ! -f "$DT_VAL" ]; then
    stage "1b  direction table (val)  ~0.2 h"
    python examples/build_direction_table.py --cfg-file "$SIM_CFG" \
        --split val --out "$DT_VAL" $D_ARGS
else
    echo "[skip] $DT_VAL exists"
fi

# ── 2. the per-bin assignment ───────────────────────────────────────────────
# NO --drop-bins. Under `approach_axis` the pipeline drops `-z` and `-x` because
# they are empty; under `grasp_offset` they hold 283 and 247 grasps and dropping
# them would silently discard a THIRD of the data. Inline rather than inside the
# collection job so the per-bin histogram is printed before 5 h of simulator
# time is spent against it.
for sp in train val; do
    dt="$DT_TRAIN"; pin="$PIN_TRAIN"
    [ "$sp" = val ] && { dt="$DT_VAL"; pin="$PIN_VAL"; }
    if [ ! -f "$pin" ]; then
        stage "2$sp  assign per-bin demos ($sp)"
        python examples/assign_direction_demos.py \
            --table "$dt" --out "${pin%.json}" --mode per-bin
    else
        echo "[skip] $pin exists"
    fi
done

# ── 3. base collection ──────────────────────────────────────────────────────
for sp in train val; do
    h5="$H5_TRAIN"; pin="$PIN_TRAIN"
    [ "$sp" = val ] && { h5="$H5_VAL"; pin="$PIN_VAL"; }
    if [ ! -f "$h5" ]; then
        stage "3$sp  collect base demonstrations ($sp)  ~5 h train / 0.3 h val"
        # `--d-rule`/`--d-point-depth` default to the pin table's `_meta`, but
        # they are passed explicitly: the shard's captions are what training
        # reads, and a silent default is the one thing that must not decide them.
        python examples/collect_regrasp_demos.py --cfg-file "$SIM_CFG" \
            --split "$sp" --output "$h5" --grasp-pin-table "$pin" \
            --d-rule location_extent --d-point-depth 0.1034 --anchor-hand-ref base \
            --valid-grasp-dict examples/valid_grasp_dict_005.pkl
    else
        echo "[skip] $h5 exists"
    fi
done

# ── 4. the audit ────────────────────────────────────────────────────────────
# `SIM.demo_ok_table` drops the (scene, bin) pairs whose demonstration never
# reached its grasp. train_regrasp.sbatch derives this itself when absent, but
# doing it here keeps the gate visible in the log rather than buried in the
# training job's preamble.
OK=output/regrasp_demos_train_loc_ok.json
if [ ! -f "$OK" ]; then
    stage "4  audit the base shard -> $OK"
    python examples/audit_regrasp_demos.py --demos "$H5_TRAIN" --write-ok "$OK"
else
    echo "[skip] $OK exists"
fi

# ── 5. the run ──────────────────────────────────────────────────────────────
# Resumable: `state.json` records the last completed iteration, so a wall-clock
# kill costs only the iteration in flight. This is the stage pass 2 continues.
stage "5  train (25 iterations)  ~16.1 h"
python examples/train_regrasp.py --cfg-file "$CFG" --run-name "$RUN" \
    --out-root "$OUT_ROOT" \
    --num-workers "$(( SLURM_CPUS_PER_TASK - 2 ))"

stage "done"
df -h "$HOME" | tail -1
