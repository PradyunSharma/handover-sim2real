#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RUN 20, EVERY STAGE, IN ONE JOB — run 19 rebuilt in the ROBOT-BASE frame.
#
#     sbatch examples/slurm/regrasp_run20_all.sbatch
#
# ONE change against run 19: `SIM.anchor_hand_ref: wrist -> base`, so the
# anchor's azimuth is `horizontal(p_base - c)` — object centroid to robot base —
# with z still world up. The shield stays on and `iter_epochs` stays at 20.
#
# WHY THE WHOLE PIPELINE AND NOT JUST THE TRAINER. The anchor decides which BIN
# each grasp is filed under, so run 9's wrist-captioned tables caption the wrong
# bins here. `setup.resolve_anchor_ref` REFUSES a run whose `SIM.anchor_hand_ref`
# disagrees with its table's recorded one — the guard that would have caught run
# 16 — so tables, pin assignment and base shards are all rebuilt under `_aabase`
# (approach_axis + base anchor). That suffix is deliberately NOT run 18's
# `_bframe`: same frame, different substrate (run 18 is `grasp_offset` at depth
# 0.1034), and the two must not share files.
#
# ── IT WILL PROBABLY NOT FINISH IN ONE ALLOCATION ────────────────────────────
#
#     direction table train    1.5 h
#     direction table val      0.2 h
#     assign per-bin demos     seconds
#     collect base train       5.0 h
#     collect base val         0.3 h
#     audit                    minutes
#     train, 25 iterations    16-17 h        (run 19's budget: run 9 measured
#     -------------------------------         13.4 h, +33% on the training half
#     TOTAL                   23-24 h         for iter_epochs 15 -> 20)
#
# 24 h is DelftBlue's hard maximum, so there is no allocation this fits with
# margin. THIS SCRIPT IS IDEMPOTENT INSTEAD: every stage is skipped when its
# output already exists, and `train_regrasp.py` resumes from `state.json`. So
# the single command above is the whole recipe — RESUBMIT THE IDENTICAL COMMAND
# and pass 2 picks up wherever pass 1 was killed. Expect two submissions.
#
# ── WHY --time=20:00:00 AND NOT 24 ───────────────────────────────────────────
# A 24 h request cannot be backfilled at all — it is the maximum, so the
# scheduler can never slot it into a gap. 20 h costs ~4 h of progress per pass
# and buys a materially better queue position; with fairshare depressed after a
# few GPU jobs that is the difference between starting in an hour and starting
# in two days. Shorten it further (`--time=08:00:00`) if the queue is bad — the
# script does not care, it just does fewer stages per pass.
#
# ── WHAT MAKES A SHORT JOB SAFE ─────────────────────────────────────────────
# Both collections and both table builds write to `<path>.partial` and rename
# only after a zero exit. A job killed mid-collection therefore leaves NO file
# the `[ ! -f ... ]` guard would mistake for a finished one — without that, pass
# 2 would train on a truncated shard and say nothing.
#
# ── EVERYTHING ON /home ─────────────────────────────────────────────────────
# `SCRATCH_ROOT` sets both `OUT_ROOT` (the run directory) and `REGRASP_DATA`
# (the HDF5 shards), and here it points into $HOME.
#
# /home IS A HARD 30 GB QUOTA THAT FILLS SILENTLY. There is no ENOSPC traceback
# — Python cannot write one to a full disk — so the job dies with exit code 6
# and an empty error file. Run 20 needs about 5 GB:
#
#     base shards (train ~570 MB + val ~23 MB)        ~0.6 GB
#     run dir: data/ (25 DAgger shards)               ~3.1 GB
#     run dir: iters/ (per-iteration checkpoints)     ~1.2 GB
#     direction + pin tables                          ~0.003 GB
#
#     examples/slurm/preflight_regrasp.sh regrasp_run20
#
# checks the quota BEFORE any of this starts, and compares the config's
# `d_rule` / `anchor_hand_ref` against the tables'. Do not skip it: losing 5 h
# of collection to a full disk is the most expensive failure available here, and
# it is the one that reports nothing.
#
# ── WHAT TO READ FIRST WHEN IT RUNS ─────────────────────────────────────────
# Gate 1, after the assignment stage: the per-bin histogram. Under `base` the
# bins are RELABELLED, not merely renamed — expect ~7.7% of grasps to move, so a
# histogram close to run 9's (+x 486, +y 359, -y 319, +z 412, -x/-z empty) is the
# pass. A wildly different shape means the sign convention is inverted and every
# bin label with it.
#
# Gate 2, iteration 1 of training: `shield_policy / (steps - expert_steps)`.
# That is how much of the learner's control the shield took, and it raises the
# EFFECTIVE beta above the configured 0.75.
#
# AND NOTE WHAT SUCCESS LOOKS LIKE: run 20 should score about the SAME as run 19
# in sim. ~92% of commands are the same vector between the two frames, so a flat
# result is the good outcome — it means deployability was bought for free. The
# number that moves is on the robot.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-rg20
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/rg20_%j.out
#SBATCH --error=slurm_logs/rg20_%j.err

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

RUN=regrasp_run20
CFG=examples/configs/regrasp_run20.yaml
SIM_CFG=examples/pretrain_multicam_wr.yaml

# THE ANCHOR REFERENCE is what makes run 20's tables incompatible with run 9's.
# `resolve_d_rule` and `resolve_anchor_ref` refuse the run if the config and the
# table disagree, so these two values must match regrasp_run20.yaml exactly.
#
# NO --d-point-depth / --d-min-offset here, unlike run 18: those are
# `grasp_offset` parameters and mean nothing under `approach_axis`, where
# `d = -R_grasp[:,2]` needs no depth along the gripper axis and has no chord to
# reject as too short.
D_ARGS="--d-rule approach_axis --anchor-hand-ref base"

DT_TRAIN=output/direction_table_train_aabase.json
DT_VAL=output/direction_table_val_aabase.json
PIN_TRAIN=output/regrasp_pins_train_aabase.json
PIN_VAL=output/regrasp_pins_val_aabase.json
H5_TRAIN="$REGRASP_DATA/bc_dataset/train_regrasp_aabase.h5"
H5_VAL="$REGRASP_DATA/bc_dataset/val_regrasp_aabase.h5"

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
# NO --drop-bins, and unlike run 18 that is not a data-preservation argument.
# Under `approach_axis` `-x` and `-z` really ARE empty — `-z` is geometrically
# impossible with the object held above a table, and `-x` (over the giver's
# fingers) is what the hand-collision filtering exists to remove — so run 9
# measured 0 and 12 scenes for them out of 623. Dropping them would change the
# BIN NUMBERING, and every historical `succ_bin_*` and the retry ladder are
# indexed by the octahedral constant. They stay in, empty, exactly as in run 9,
# and `command_axes.json` marks them by carrying the raw axis instead of a
# centroid.
#
# THIS IS ALSO GATE 1. Inline rather than inside the collection job so the
# per-bin histogram prints before 5 h of simulator time is spent against it. The
# base frame RELABELS bins rather than renaming them, so compare against run 9's
# (+x 486, +y 359, -y 319, +z 412) — ~7.7% of grasps should move. A wildly
# different shape means the sign convention is inverted.
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
            --d-rule approach_axis --anchor-hand-ref base \
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
OK=output/regrasp_demos_train_aabase_ok.json
if [ ! -f "$OK" ]; then
    stage "4  audit the base shard -> $OK"
    python examples/audit_regrasp_demos.py --demos "$H5_TRAIN" --write-ok "$OK"
else
    echo "[skip] $OK exists"
fi

# ── 5. the run ──────────────────────────────────────────────────────────────
# Resumable: `state.json` records the last completed iteration, so a wall-clock
# kill costs only the iteration in flight. This is the stage pass 2 continues.
stage "5  train (25 iterations)  ~16-17 h"
python examples/train_regrasp.py --cfg-file "$CFG" --run-name "$RUN" \
    --out-root "$OUT_ROOT" \
    --num-workers "$(( SLURM_CPUS_PER_TASK - 2 ))"

stage "done"
df -h "$HOME" | tail -1
