#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SCORE A FINISHED PHASE-4 DAgger RUN ON A HELD-OUT SPLIT, every iteration.
#
#     RUN=dagger4_run19 sbatch examples/slurm/eval_dagger_testset.sbatch
#
# Knobs, all optional, all read from the environment so one file serves every
# run (`--export` them or put them inline as above):
#
#     NO_PIN     1 to drop the pin table and score ALL 144 test scenes
#                (pose diagnostics go NaN — see the script's --no-pin-table)
#     FORCE      1 to re-score iterations already in the CSV (default off)
#     OUT        CSV filename (default <split>_log.csv)
#     RUN        run NAME under $OUT_ROOT            (default dagger4_run19)
#     RUN_DIR    full path, overrides RUN            (default $OUT_ROOT/$RUN)
#     SPLIT      test | val | train                  (default test)
#     ITERS      all | 0,5,10,...                     (default all)
#     CKPT       best | last                         (default: the run's EVAL.ckpt)
#     PIN_TABLE  override the split's pin table      (default: derived from the
#                                                     run's, split substituted)
#     ALLOW_PIN  1 to ignore a pin-provenance mismatch (default off — see below)
#     NUM_SCENES cap the scene count                 (default: all of them)
#
# ── WHY THIS EXISTS ──────────────────────────────────────────────────────────
# Phase 4 has never had a held-out number. `eval_dagger_run.py` takes no
# `--split`: it rebuilds the run's own context from the run's own config.yaml,
# and every Phase-4 config carries `SIM.split: train`, `EVAL.holdout: false`,
# `EVAL.num_scenes: 100`. So the in-loop curve and the standalone re-score are
# the SAME measurement — 100 of ~623 usable TRAIN scenes, all of which were also
# collected on. dagger4_run19's headline has been quoted as a result; it is a
# train-set number.
#
# This scores all 144 test scenes, under the run's own saved config, so the only
# thing that differs is the data.
#
# ── DAGGER4_RUN19 IS READY; CHECK THE PIN RULE FOR ANY OTHER RUN ────────────
# `output/grasp_pin_table_test_omg.json` (144 scenes, built 2026-08-17) matches
# dagger4_run19's `grasp_pin_table_train_omg.json` on all six provenance fields:
# mode, tol, setup, hand_collision_filter, hand_collision_thresh and
# valid_grasp_dict_path. The script checks that TABLE AGAINST TABLE rather than
# against the config, because a Phase-4 config does not record the pin rule at
# all — it lives only in the table's `_meta.mode` and in the filename.
#
# THAT MATTERS BECAUSE BOTH RULES EXIST IN `output/`, four characters apart:
#     grasp_pin_table_test_omg.json      mode: omg                 <- run 19
#     grasp_pin_table_val.json           mode: furthest_from_hand
# Scoring an `omg` run against a `furthest_from_hand` table measures agreement
# with a grasp the policy was never taught; `near_rate`, `chance_rate` and
# `pos_err` all collapse for a reason that says nothing about the policy. The
# script refuses it. A run pinned with `furthest_from_hand` needs a
# `furthest_from_hand` test table built first — there is currently none.
#
# ── COST ─────────────────────────────────────────────────────────────────────
# 144 test scenes, one episode each (Phase 4 is unconditioned — no slots, no
# bins), against Regrasp's 340. At the measured 4.4 s per episode that is
# ~11 min per iteration, so run 19's 26 iterations are ~4.6 h: two 3 h passes.
#
# THE SWEEP IS RESUMABLE PER ITERATION. `write_log` rewrites the CSV after each
# one and `read_done` skips what is already there, so a wall-clock kill costs at
# most the ~11 min in flight. Chain short jobs rather than asking for one long
# one — DelftBlue backfills short ahead of long.
#
#     RUN=dagger4_run19
#     J=$(sbatch --parsable --export=ALL,RUN=$RUN \
#             examples/slurm/eval_dagger_testset.sbatch)
#     J=$(sbatch --parsable --dependency=afterany:$J --export=ALL,RUN=$RUN \
#             examples/slurm/eval_dagger_testset.sbatch)
#
# A pass that finds everything scored prints "nothing to score" and exits in
# seconds, so a spare tail job costs a queue slot and nothing else.
#
# ── ONE PREREQUISITE ─────────────────────────────────────────────────────────
#     ls -la output/grasp_pin_table_test_omg.json
# Nothing here builds it; the script exits with the command that would.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-p4testeval
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/p4testeval_%j.out
#SBATCH --error=slurm_logs/p4testeval_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2023r1 || true
module load cuda/11.7 || true
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pch2r_dev}"
export PYTHONUNBUFFERED=1
export GADDPG_DIR="${GADDPG_DIR:-$PWD/GA-DDPG}"
# `env_setup` imports OMG at module scope even though evaluation never plans —
# except on scenes absent from the pin table, where `_resolve_grasp_pose` falls
# back to one step-0 plan. Either way the variable has to be set or the import
# fails before the first rollout.
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-$PWD/OMG-Planner}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/h2r-runs}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT/output/dagger_runs}"
mkdir -p slurm_logs

RUN="${RUN:-dagger4_run19}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$RUN}"
SPLIT="${SPLIT:-test}"
ITERS="${ITERS:-all}"

# THE RUN DIR IS THE ONE THING THIS CANNOT GUESS. Older campaigns wrote into the
# repo's own `output/`, later ones into $SCRATCH_ROOT. Try both rather than
# failing on a path the user never chose.
if [ ! -f "$RUN_DIR/config.yaml" ] && [ -f "output/dagger_runs/$RUN/config.yaml" ]; then
    echo "[note] $RUN_DIR has no config.yaml; using ./output/dagger_runs/$RUN"
    RUN_DIR="output/dagger_runs/$RUN"
fi
if [ ! -f "$RUN_DIR/config.yaml" ]; then
    echo "FAIL: no config.yaml under $RUN_DIR — set RUN_DIR explicitly." >&2
    exit 2
fi

# An ARRAY, expanded as "${EXTRA[@]+"${EXTRA[@]}"}": under `set -u`, bash 4.3 and
# earlier treat an EMPTY array's [@] expansion as an unbound variable and abort,
# so the DEFAULT invocation would be the one that dies. DelftBlue's images do not
# have to agree on the bash version and losing a 3 h job to that is not worth the
# two characters saved.
EXTRA=()
[ "${NO_PIN:-0}" = "1" ] && EXTRA+=(--no-pin-table)
[ "${FORCE:-0}" = "1" ] && EXTRA+=(--force)
[ -n "${OUT:-}" ]      && EXTRA+=(--out "$OUT")
[ -n "${CKPT:-}" ]       && EXTRA+=(--ckpt "$CKPT")
[ -n "${PIN_TABLE:-}" ]  && EXTRA+=(--pin-table "$PIN_TABLE")
[ -n "${NUM_SCENES:-}" ] && EXTRA+=(--num-scenes "$NUM_SCENES")
[ "${ALLOW_PIN:-0}" = "1" ] && EXTRA+=(--allow-pin-mismatch)

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "run_dir=$RUN_DIR  split=$SPLIT  iters=$ITERS"
echo "extra: ${EXTRA[*]+"${EXTRA[*]}"}"
echo "already scored: $(tail -n +2 "$RUN_DIR/${SPLIT}_log.csv" 2>/dev/null | wc -l) iteration(s)"
df -h "$HOME" | tail -1

python examples/eval_dagger_testset.py \
    --run-dir "$RUN_DIR" --split "$SPLIT" --iters "$ITERS" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
rc=$?

# Re-render both figures from whatever the CSV holds, even if the sweep was cut
# short — a partial curve is worth looking at and costs seconds. On a
# one-iteration sweep only `<split>_summary.png` says anything: a stacked AREA
# needs two x points to render at all.
echo
echo "=== plotting from $RUN_DIR/${SPLIT}_log.csv ==="
python examples/eval_dagger_testset.py \
    --run-dir "$RUN_DIR" --split "$SPLIT" --plot-only || true

echo
echo "scored: $(tail -n +2 "$RUN_DIR/${SPLIT}_log.csv" 2>/dev/null | wc -l) iteration(s)"
exit $rc
