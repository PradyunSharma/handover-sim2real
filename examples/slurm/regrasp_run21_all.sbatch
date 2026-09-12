#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RUN 21, EVERY STAGE, IN ONE JOB — run 19 with two bug fixes, a standoff gate,
# a longer horizon, and NO WRIST CAMERA.
#
#     sbatch examples/slurm/regrasp_run21_all.sbatch
#
# Six changes against run 19. Two are corrections, three are new behaviour, one
# is the rig:
#
#   DAGGER.replan_after_pin  false -> true    step-0 label bug (config key)
#   (no key)                                  worker/manager pin-table bug
#   DAGGER.plane_gate        false -> true    only the expert enters the reach
#   DAGGER.max_steps         50 -> 80         horizon
#   EVAL.max_steps           50 -> 80         scored to match
#   SIM.cfg_file             wrist+right -> RIGHT ONLY
#
# ── WHY THE WHOLE PIPELINE AND NOT JUST THE TRAINER ─────────────────────────
#
# THE CAMERA CHANGE IS WHAT FORCES IT, and the reason is not obvious. The
# direction table records `centroid_world` as the OBSERVED point cloud's
# centroid — build_direction_table.py says so in its own docstring — not the
# object pose. That is deliberate: it is the same quantity the policy will see.
# But it means dropping the wrist camera moves the centroid by centimetres,
# moves the anchor frame with it, and RE-BINS grasps. Run 9/19's tables caption
# the wrong bins here.
#
# AND NOTHING WOULD HAVE TOLD YOU. Every shard records `cameras` in its attrs
# and no loader compares them; `resolve_anchor_ref` guards the anchor reference
# and `resolve_d_rule` the direction rule, but until run 21 nothing guarded the
# camera set. `preflight_regrasp.sh` now checks `cfg_file` against the table's
# `_meta` — run it, it is the only thing standing between you and a run that
# trains wrist+right base demonstrations mixed with right-only DAgger data and
# reports nothing.
#
# Everything is rebuilt under the `_right` suffix, kept distinct from run 9/19's
# unsuffixed files and from run 20's `_aabase`.
#
# ── IT WILL NOT FINISH IN ONE ALLOCATION ────────────────────────────────────
#
#     direction table train    1.5 h
#     direction table val      0.2 h
#     assign per-bin demos     seconds
#     collect base train       5.0 h
#     collect base val         0.3 h
#     audit                    minutes
#     train, 25 iterations    17-20 h        run 19 measured ~16 h at 50 steps;
#     -------------------------------        max_steps 50 -> 80 lengthens only
#     TOTAL                   24-27 h        the episodes that were hitting the
#                                            cap (13.9% in run 19), while the
#                                            pool fix REMOVES ~18% of collection
#                                            that was going to unreachable pairs
#
# 24 h is DelftBlue's hard maximum, so nothing fits. THIS SCRIPT IS IDEMPOTENT:
# every stage is skipped when its output exists and `train_regrasp.py` resumes
# from `state.json`. RESUBMIT THE IDENTICAL COMMAND — expect two or three passes.
#
# ── WHY --time=20:00:00 AND NOT 24 ───────────────────────────────────────────
# A 24 h request is the maximum, so the scheduler can never backfill it into a
# gap. 20 h costs ~4 h of progress per pass and buys a materially better queue
# position. Shorten it further if the queue is bad — the script does not care.
#
# ── WHAT MAKES A SHORT JOB SAFE ─────────────────────────────────────────────
# Both collections and both table builds write to `<path>.partial` and rename
# only on a zero exit, so a job killed mid-collection leaves NO file the
# `[ ! -f ... ]` guard would mistake for a finished one.
#
# ── DISK ────────────────────────────────────────────────────────────────────
# `SCRATCH_ROOT` sets both `OUT_ROOT` and `REGRASP_DATA`, pointing into $HOME.
# /home IS A HARD 30 GB QUOTA THAT FILLS SILENTLY — no ENOSPC traceback, just
# exit code 6 and an empty .err. Run 21 needs ~5 GB, and note it does NOT
# replace run 19's files, it ADDS to them: check free space before starting.
#
#     examples/slurm/preflight_regrasp.sh regrasp_run21
#
# ── WHAT TO READ FIRST ──────────────────────────────────────────────────────
# Gate 1, after the assignment stage: the per-bin histogram. Run 9 measured
# +x 486, +y 359, -y 319, +z 412, with -x and -z empty. The camera change moves
# the centroid, so expect some drift — but -x/-z must stay empty and the
# ordering must hold. A wildly different shape means the centroid moved far
# enough to break the frame, and the run is not comparable to anything.
#
# Gate 2, iteration 1: `(shield_policy + plane_policy) / (steps - expert_steps)`.
# That is how much of the learner's control was taken away, and it raises the
# EFFECTIVE beta above the configured 0.75. `plane_policy` is new and untested
# at scale — if it is a large fraction, the gate is doing more than intended and
# `plane_gate_margin` should be loosened rather than the gate turned off.
#
# Gate 3, iteration 1: `pin_goal_moved / episodes`. This measures the step-0
# label bug `replan_after_pin` fixes. Small means scene 95 was unlucky.
#
# AND NOTE WHAT SUCCESS LOOKS LIKE. `reached_grasp / episodes` should rise from
# run 19's 0.604 toward ~0.69 ON ARITHMETIC ALONE — the pool fix removes the 18%
# of episodes that were failing 77% of the time. Do not read that as learning.
# `success_rate` is NOT comparable with runs 1-20: this run scores at 80 steps,
# they scored at 50. Losing the wrist camera is expected to COST success rate in
# sim; the argument for it is that what remains is what the rig actually has.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-rg21
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/rg21_%j.out
#SBATCH --error=slurm_logs/rg21_%j.err

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

RUN=regrasp_run21
CFG=examples/configs/regrasp_run21.yaml
# RIGHT CAMERA ONLY. This single line is what forces stages 1-4 to re-run; it
# must match `SIM.cfg_file` in regrasp_run21.yaml exactly, and preflight check 4
# compares it against the table's `_meta`.
SIM_CFG=examples/pretrain_right.yaml

# THE ANCHOR REFERENCE is what makes run 20's tables incompatible with run 9's.
# `resolve_d_rule` and `resolve_anchor_ref` refuse the run if the config and the
# table disagree, so these two values must match regrasp_run20.yaml exactly.
#
# NO --d-point-depth / --d-min-offset here, unlike run 18: those are
# `grasp_offset` parameters and mean nothing under `approach_axis`, where
# `d = -R_grasp[:,2]` needs no depth along the gripper axis and has no chord to
# reject as too short.
D_ARGS="--d-rule approach_axis --anchor-hand-ref wrist"

DT_TRAIN=output/direction_table_train_right.json
DT_VAL=output/direction_table_val_right.json
PIN_TRAIN=output/regrasp_pins_train_right.json
PIN_VAL=output/regrasp_pins_val_right.json
H5_TRAIN="$REGRASP_DATA/bc_dataset/train_regrasp_right.h5"
H5_VAL="$REGRASP_DATA/bc_dataset/val_regrasp_right.h5"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "SCRATCH_ROOT=$SCRATCH_ROOT   REGRASP_DATA=$REGRASP_DATA"
df -h "$HOME" | tail -1

stage () { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

# ── 1. direction tables ─────────────────────────────────────────────────────
# THE ANCHOR REFERENCE IS UNCHANGED HERE — run 21 keeps run 19's `wrist`, and
# `--anchor-hand-ref wrist` below says so explicitly rather than relying on a
# default. It is run 20, not this one, that moves the frame to `base`.
#
# WHAT FORCES THE REBUILD IS THE CAMERA. The anchor frame's ORIGIN is
# `centroid_world`, and build_direction_table.py records that as the centroid of
# the points THE CAMERAS ACTUALLY RETURNED — a visible-surface centroid, not the
# object's pose. Dropping the wrist camera changes which surface is visible, so
# the centroid moves by centimetres, the anchor moves with it, and grasps land in
# different bins. Run 9/19's tables caption the wrong bins for a right-only run.
#
# AND NOTHING VALIDATED THIS UNTIL NOW. `resolve_anchor_ref` guards the anchor
# reference and `resolve_d_rule` the direction rule, but the camera set was
# unguarded everywhere: shards record `cameras` in their attrs and no loader
# compares them. preflight_regrasp.sh check 4 now compares `cfg_file` against the
# table's `_meta`, which is the only thing that stops this run silently reusing
# wrist+right tables.
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
            --d-rule approach_axis --anchor-hand-ref wrist \
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
OK=output/regrasp_demos_train_right_ok.json
if [ ! -f "$OK" ]; then
    stage "4  audit the base shard -> $OK"
    python examples/audit_regrasp_demos.py --demos "$H5_TRAIN" --write-ok "$OK"
else
    echo "[skip] $OK exists"
fi

# ── 5. the run ──────────────────────────────────────────────────────────────
# Resumable: `state.json` records the last completed iteration, so a wall-clock
# kill costs only the iteration in flight. This is the stage pass 2 continues.
stage "5  train (25 iterations)  ~17-20 h"
python examples/train_regrasp.py --cfg-file "$CFG" --run-name "$RUN" \
    --out-root "$OUT_ROOT" \
    --num-workers "$(( SLURM_CPUS_PER_TASK - 2 ))"

stage "done"
df -h "$HOME" | tail -1
