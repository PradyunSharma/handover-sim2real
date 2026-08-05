#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — score a Phase-4 DAgger run's iterations, OUTSIDE the loop.
#
#   Run it alongside the trainer (it polls for new iterations):
#     sbatch --export=ALL,RUN=dagger4_run6 examples/slurm/eval_dagger_run.sbatch
#   Or once, after the run finishes:
#     sbatch --export=ALL,RUN=dagger4_run6,WATCH=0,PUBLISH=1 \
#         examples/slurm/eval_dagger_run.sbatch
#
# WHY A SEPARATE JOB. Evaluation is ~19% of a DAgger run's wall clock and nothing
# in the loop waits on it: train_dagger_phase4.py assigns cur_run_dir = iter_dir
# BEFORE the eval block, so the next iteration collects with the freshly-refit
# policy regardless; the metrics feed only checkpoint selection and reporting.
# Set EVAL.every: 0 in the run config and let this job carry the cost in parallel.
#
# It is read-only on everything the trainer writes. The only file it creates is
# <run>/eval_log.csv, written atomically via a .tmp rename, so a plotter reading
# it mid-write never sees a partial file. Iterations already scored are skipped,
# so it is restartable and safe to resubmit.
#
# PUBLISH=1 also exports <run>/best. Leave it OFF while the trainer is running —
# the trainer writes there too, and two writers is a race. Use it in the
# after-the-fact invocation.
#
# ONE GPU, FEW CORES: this is a serial loop over eval scenes and, unlike
# collection, there is no worker pool for it — it is off the critical path, so
# its own latency does not matter.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-dag4-eval
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/dagger4_eval_%j.out
#SBATCH --error=slurm_logs/dagger4_eval_%j.err

set -euo pipefail
mkdir -p slurm_logs

# ── environment (identical to collect_demos.sbatch — keep in sync) ──────────
module load miniconda3 2>/dev/null || true
source "$(conda info --base 2>/dev/null || echo /apps/generic/miniforge3/25.11.0)/etc/profile.d/conda.sh"
conda activate pch2r_dev
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-$PWD/OMG-Planner}"
export GADDPG_DIR="${GADDPG_DIR:-$PWD/GA-DDPG}"
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/OMG-Planner/orocos_kinematics_dynamics/orocos_kdl/release/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="7.0"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Unbuffered: a 24 h run must show progress live, and a crash must leave a log
# that stops where it actually failed rather than at an arbitrary buffer edge.
export PYTHONUNBUFFERED=1

# ── parameters (override via sbatch --export=ALL,...) ───────────────────────
RUN="${RUN:?set RUN=<run name>, e.g. --export=ALL,RUN=dagger4_run6}"
OUT_ROOT="${OUT_ROOT:-output/dagger_runs}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${RUN}}"
WATCH="${WATCH:-1}"                   # 1 = poll for new iterations while training runs
POLL_S="${POLL_S:-300}"
# Stop watching after this long with no new iteration. Sized so a slow late
# iteration (collection + a ~30 min from-scratch refit on a 55k-step aggregate)
# cannot be mistaken for the trainer having finished.
TIMEOUT_S="${TIMEOUT_S:-5400}"
PUBLISH="${PUBLISH:-0}"               # 1 = also export <run>/best (NOT while training)
NUM_SCENES="${NUM_SCENES:-}"          # override EVAL.num_scenes
EXTRA="${EXTRA:-}"

if [ ! -f "${RUN_DIR}/config.yaml" ]; then
    echo "ERROR: ${RUN_DIR}/config.yaml not found — is that a Phase-4 run dir?" >&2
    exit 1
fi

ARGS=(--run-dir "${RUN_DIR}")
[ "${WATCH}" = "1" ] && ARGS+=(--watch --poll-s "${POLL_S}" --timeout-s "${TIMEOUT_S}")
[ "${PUBLISH}" = "1" ] && ARGS+=(--publish-best)
[ -n "${NUM_SCENES}" ] && ARGS+=(--num-scenes "${NUM_SCENES}")

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "run-dir=${RUN_DIR}  watch=${WATCH}  publish=${PUBLISH}  extra=${EXTRA:-none}"

srun python examples/eval_dagger_run.py "${ARGS[@]}" ${EXTRA}
