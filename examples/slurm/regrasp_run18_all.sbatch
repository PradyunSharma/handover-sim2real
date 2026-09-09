#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RUN 18, EVERY STAGE, IN ONE JOB — and everything on /home, not /scratch.
#
#     sbatch examples/slurm/regrasp_run18_all.sbatch
#
# IT WILL NOT FINISH IN ONE ALLOCATION, and that is not a bug in the script.
# Measured from run 11's own `wall_s` (fit `wall(i) = 0.78*i + 6.6 min`) and
# scaled for the ~7x larger eval:
#
#     STAGE                          work     indivisible?
#     1a direction table train        1.5 h   yes
#     1b direction table val          0.2 h   yes
#     2  assign per-bin demos        ~0.0 h   yes (seconds)
#     3a collect train                5.0 h   YES - see below
#     3b collect val                  0.3 h   YES
#     4  audit the base shard        ~0.1 h   yes
#     5  train, 25 iterations        16.1 h   no, resumable PER ITERATION
#     -------------------------------------
#        phases 1-4                   7.1 h
#        phase 5                     16.1 h
#
# ── DO NOT REQUEST 24 h ──────────────────────────────────────────────────────
# DelftBlue's scheduler backfills SHORT jobs ahead of long ones, so a 24 h
# request queues behind almost everything while a 7-9 h request starts far
# sooner. Total wall clock is dominated by queue time, not compute, so asking
# for what a phase actually needs is the single biggest lever available.
#
# `#SBATCH --time` below is 9 h: enough for phases 1-4 in one job (7.1 h of work,
# 27% buffer). Phase 5 is resumable per iteration, so it goes in SHORTER chained
# jobs rather than one long one. Override on the command line:
#
#     # phases 1-4, one job
#     J=$(sbatch --parsable examples/slurm/regrasp_run18_all.sbatch)
#     # phase 5, three chained passes of 7 h
#     for i in 1 2 3; do
#       J=$(sbatch --parsable --time=07:00:00 --dependency=afterany:$J \
#              examples/slurm/regrasp_run18_all.sbatch)
#     done
#
# `afterany`, NOT `afterok`: a pass that hits its wall clock exits non-zero,
# which is EXPECTED, and under `afterok` the next pass would never start. A pass
# that finds all 25 iterations already done exits in seconds, so an unneeded
# tail job costs a queue slot and nothing else.
#
# ── WHAT MAKES SHORT JOBS SAFE ──────────────────────────────────────────────
# Every stage is skipped when its output already exists, and `train_regrasp.py`
# resumes from `state.json`, so a wall-clock kill in phase 5 costs only the
# iteration in flight.
#
# THE COLLECTION STAGES ARE THE EXCEPTION and are why `.partial` exists. The
# collector writes one HDF5 group per episode as it goes, so a job killed
# mid-collection leaves a real, openable, INCOMPLETE shard — and a bare
# `[ ! -f "$h5" ]` check would call it finished. Every later stage would then
# train on a truncated shard silently. Both collections and both table builds
# therefore write to `<out>.partial` and rename only after a zero exit. A killed
# collection re-runs from scratch; it never half-succeeds.
#
# WHY ONE JOB RATHER THAN FIVE, in case the trade is ever revisited. Each stage
# wants different resources — the table build is one GPU and 4 cores, collection
# is CPU-wide, training is GPU plus workers — so a single allocation asks for the
# maximum any stage needs and wastes the rest. Against that: five submissions
# means five queue waits, and the phase split above already recovers most of the
# benefit. The staged version in the run-16 section of docs/run_index.md also
# fails cheaper: a bad direction table costs 1.5 h there and 6.5 h here, because
# here it is only discovered after collection has run against it.
#
# ── EVERYTHING ON /home ──────────────────────────────────────────────────────
# `SCRATCH_ROOT` is a misnomer inherited from train_regrasp.sbatch: it sets both
# `OUT_ROOT` (the run directory) and `REGRASP_DATA` (the HDF5 shards), and it can
# point anywhere. Here it points into $HOME.
#
# ⚠ /home IS A HARD 30 GB QUOTA THAT FILLS SILENTLY. There is no ENOSPC
# traceback — Python cannot write one to a full disk — so the job dies with exit
# code 6 and an empty error file. Run 18 needs about 5 GB:
#
#     base shards (train 570 MB + val 23 MB)          ~0.6 GB
#     run dir: data/ (25 DAgger shards)               ~3.1 GB
#     run dir: iters/ (per-iteration checkpoints)     ~1.2 GB
#     direction + pin tables                          ~0.003 GB
#
# `preflight_regrasp.sh regrasp_run18` checks the quota BEFORE any of this starts. Do not skip
# it: losing 6 h of collection to a full disk is the single most expensive
# failure available here, and it is the one that reports nothing.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-rg18
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/rg18_%j.out
#SBATCH --error=slurm_logs/rg18_%j.err

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

RUN=regrasp_run18
CFG=examples/configs/regrasp_run18.yaml
SIM_CFG=examples/pretrain_multicam_wr.yaml

# `d_rule`, the DEPTH and THE ANCHOR REFERENCE are what make run 18's tables
# incompatible with run 16's. `resolve_d_rule` and `resolve_anchor_ref` refuse
# the run if the config and the table disagree on any of them, so these four
# values must match examples/configs/regrasp_run18.yaml exactly.
D_ARGS="--d-rule grasp_offset --d-point-depth 0.1034 --d-min-offset 0.02 --anchor-hand-ref base"

DT_TRAIN=output/direction_table_train_bframe.json
DT_VAL=output/direction_table_val_bframe.json
PIN_TRAIN=output/regrasp_pins_train_bframe.json
PIN_VAL=output/regrasp_pins_val_bframe.json
H5_TRAIN="$REGRASP_DATA/bc_dataset/train_regrasp_bframe.h5"
H5_VAL="$REGRASP_DATA/bc_dataset/val_regrasp_bframe.h5"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "SCRATCH_ROOT=$SCRATCH_ROOT   REGRASP_DATA=$REGRASP_DATA"
df -h "$HOME" | tail -1

stage () { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

# ── 1. direction tables ─────────────────────────────────────────────────────
# The ANCHOR FRAME decides which bin every grasp is filed under, and run 18
# changes it (`hand_centroid` -> `base`). Run 16's tables therefore caption the
# wrong bins for run 18 and must be rebuilt: measured, 7.7% of grasps change bin
# between the wrist and base frames and far more between hand_centroid and base,
# which the tables were never told about at all. `--anchor-hand-ref` is NEW —
# run 16 had no such flag, which is exactly why its table and its rollouts
# disagreed. `setup.resolve_anchor_ref` now refuses a run whose config and table
# differ, so a stale table fails in the first second instead of after 20 hours.
if [ ! -f "$DT_TRAIN" ]; then
    stage "1a  direction table (train)  ~1.5 h"
    rm -f "$DT_TRAIN.partial"
    python examples/build_direction_table.py --cfg-file "$SIM_CFG" \
        --split train --out "$DT_TRAIN.partial" $D_ARGS
    mv "$DT_TRAIN.partial" "$DT_TRAIN"
else
    echo "[skip] $DT_TRAIN exists"
fi

if [ ! -f "$DT_VAL" ]; then
    stage "1b  direction table (val)  ~0.2 h"
    rm -f "$DT_VAL.partial"
    python examples/build_direction_table.py --cfg-file "$SIM_CFG" \
        --split val --out "$DT_VAL.partial" $D_ARGS
    mv "$DT_VAL.partial" "$DT_VAL"
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
        # ---- WRITE ASIDE, RENAME ON SUCCESS -------------------------------
        # The collector creates one HDF5 group PER EPISODE as it goes, so a job
        # killed on the wall clock leaves a real, openable, INCOMPLETE file --
        # and the `[ ! -f "$h5" ]` check above would treat it as finished. Every
        # later stage would then train on a truncated shard with no warning.
        # Collecting into `.partial` and renaming only after a zero exit makes
        # the check honest, which is what allows this script to be submitted
        # with a time limit shorter than the whole pipeline.
        rm -f "$h5.partial"
        python examples/collect_regrasp_demos.py --cfg-file "$SIM_CFG" \
            --split "$sp" --output "$h5.partial" --grasp-pin-table "$pin" \
            --d-rule grasp_offset --d-point-depth 0.1034 --anchor-hand-ref base \
            --valid-grasp-dict examples/valid_grasp_dict_005.pkl
        mv "$h5.partial" "$h5"
    else
        echo "[skip] $h5 exists"
    fi
done

# ── 4. the audit ────────────────────────────────────────────────────────────
# `SIM.demo_ok_table` drops the (scene, bin) pairs whose demonstration never
# reached its grasp. train_regrasp.sbatch derives this itself when absent, but
# doing it here keeps the gate visible in the log rather than buried in the
# training job's preamble.
OK=output/regrasp_demos_train_bframe_ok.json
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
