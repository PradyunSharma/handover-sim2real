#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SCORE A RUN UNDER THE OFFICIAL HANDOVERSIM BENCHMARK PROTOCOL.
#
#     RUN=regrasp_run19 ITER=25 sbatch examples/slurm/eval_benchmark_protocol.sbatch
#
#     RUN        run NAME under $OUT_ROOT      (default regrasp_run19)
#     RUN_DIR    full path, overrides RUN
#     ITER       iteration to score            (default: the run's own best)
#     CKPT       best | last                   (default: the run's EVAL.ckpt)
#     SPLIT      test | val | train            (default test)
#     START_FRAME  first | last                (default first — the benchmark)
#     YCB_MODE   all | grasp_only              (default all — the benchmark)
#     BINS       regrasp direction order, e.g. '+y,+z,+x,-y'
#     NUM_SCENES cap, for a smoke test (skips the official pass)
#     RES_DIR    output directory
#     FORCE      1 to re-run scenes already on disk
#
# ── WHAT THIS MEASURES THAT THE OTHER TWO SCRIPTS DO NOT ────────────────────
# `eval_regrasp_testset.py` and `eval_dagger_testset.py` score `stable_grasp` —
# close, hold, object secured. The benchmark wants that AND the carry: the hand
# inside a 15 cm ball at GOAL_CENTER for 0.1 s, all within 13 s. benchmark-
# SUCCESS is a STRICT SUBSET of ours, so every rate this repo has published is
# an upper bound on the number this produces.
#
# The metrics are computed by `handover.benchmark_evaluator.evaluate()` — their
# code, on the official per-scene `.npz` files this writes — so a row from here
# and a row from `all_cvpr2023_results_eval.sh` are the same row.
#
# ── THREE WAYS OUR SETUP DIFFERS, AND THE DEFAULTS UNDO ALL THREE ───────────
#   the carry          we never did it          -> this script does it
#   YCB_MANO_START_FRAME  every config: `last`  -> default `first`
#   YCB_LOAD_MODE         every config: grasp_only -> default `all`
#
# The defaults reproduce the published benchmark, which means the policy is run
# OUT OF DISTRIBUTION on two axes it was never trained for. Expect the number to
# be well below the `stable_grasp` rate; that gap is the finding, not a bug.
#
# To separate the three causes, run the ladder:
#   START_FRAME=last YCB_MODE=grasp_only   -> the carry alone
#   START_FRAME=last YCB_MODE=all          -> + clutter
#   (defaults)                             -> + the moving human = the real number
#
# ── COST ───────────────────────────────────────────────────────────────────
# Up to 13 s of sim per scene versus the ~7.5 s our 50-step eval used, plus the
# retreat, and 144 scenes. Budget ~25-35 min per run; one 3 h slot is ample and
# the sweep RESUMES — an existing `{:03d}.npz` is loaded rather than re-run
# unless FORCE=1.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-bench
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/bench_%j.out
#SBATCH --error=slurm_logs/bench_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2023r1 || true
module load cuda/11.7 || true
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pch2r_dev}"
export PYTHONUNBUFFERED=1
export GADDPG_DIR="${GADDPG_DIR:-$PWD/GA-DDPG}"
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-$PWD/OMG-Planner}"
# `benchmark_evaluator` imports from the handover package by name, so its own
# directory has to be importable for the official pass to run.
export PYTHONPATH="${PYTHONPATH:-}:$PWD/handover-sim"

export SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/h2r-runs}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT/output/dagger_runs}"
mkdir -p slurm_logs

RUN="${RUN:-regrasp_run19}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$RUN}"
if [ ! -f "$RUN_DIR/config.yaml" ] && [ -f "output/dagger_runs/$RUN/config.yaml" ]; then
    echo "[note] $RUN_DIR has no config.yaml; using ./output/dagger_runs/$RUN"
    RUN_DIR="output/dagger_runs/$RUN"
fi
if [ ! -f "$RUN_DIR/config.yaml" ]; then
    echo "FAIL: no config.yaml under $RUN_DIR — set RUN_DIR explicitly." >&2
    exit 2
fi

# An ARRAY, expanded as "${EXTRA[@]+"${EXTRA[@]}"}": under `set -u`, bash 4.3 and
# earlier abort on an empty array's [@] expansion, which would kill the DEFAULT
# invocation.
EXTRA=()
[ -n "${ITER:-}" ]        && EXTRA+=(--iter "$ITER")
[ -n "${CKPT:-}" ]        && EXTRA+=(--ckpt "$CKPT")
[ -n "${START_FRAME:-}" ] && EXTRA+=(--start-frame "$START_FRAME")
[ -n "${YCB_MODE:-}" ]    && EXTRA+=(--ycb-load-mode "$YCB_MODE")
[ -n "${BINS:-}" ]        && EXTRA+=(--bins "$BINS")
[ -n "${NUM_SCENES:-}" ]  && EXTRA+=(--num-scenes "$NUM_SCENES")
[ -n "${RES_DIR:-}" ]     && EXTRA+=(--res-dir "$RES_DIR")
[ "${FORCE:-0}" = "1" ]   && EXTRA+=(--force)

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "run_dir=$RUN_DIR  split=${SPLIT:-test}"
echo "extra: ${EXTRA[*]+"${EXTRA[*]}"}"
df -h "$HOME" | tail -1

python examples/eval_benchmark_protocol.py \
    --run-dir "$RUN_DIR" --split "${SPLIT:-test}" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
exit $?
