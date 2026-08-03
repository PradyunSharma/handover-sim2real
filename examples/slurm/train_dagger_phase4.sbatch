#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — Phase-4 DAgger (examples/train_dagger_phase4.py).
# STEP 3, after the pin table (step 1) and the pinned demonstrations (step 2).
#
#   Default run:
#     sbatch --export=ALL,RUN=dagger4_run1 examples/slurm/train_dagger_phase4.sbatch
#   Short shakedown FIRST (2 iterations, 4 episodes, 6 eval scenes, ~20 min).
#   ALWAYS override --time here: the 24 h default cannot be backfilled, and with
#   fairshare depressed after a few GPU jobs that is the difference between
#   starting in an hour and starting in two days (measured: priority fell 46x
#   after four allocations in one afternoon, pushing the estimate to +2.5 days).
#     sbatch --time=00:45:00 --export=ALL,RUN=dagger4_smoke,CFG=examples/configs/dagger_phase4_smoke.yaml \
#         examples/slurm/train_dagger_phase4.sbatch
#   From an existing base run (skips the 100-epoch base fit):
#     sbatch --export=ALL,RUN=dagger4_run2,BASE=output/bc_runs/phase1_full \
#         examples/slurm/train_dagger_phase4.sbatch
#   Watch:  tail -f slurm_logs/dagger4_<jobid>.out
#   Plot (safe while running — read-only):
#     python examples/plot_dagger_run.py output/dagger_runs/<RUN>
#
# RESUMABLE. Re-running the SAME command continues: completed iterations are
# skipped, a finished-but-untrained collection is reused, and an interrupted
# training resumes from its last.pt. So a 24 h timeout is not a loss — just
# resubmit the identical command and it picks up where it stopped.
#
# Budget (20 iterations x m=20 episodes, eval every 2 over 100 scenes):
#   collect ~15-20 min/iter (an OMG replan every step is the cost)
#   refit   ~5-10 min/iter  (grows with |D|; FTL refits from scratch each time)
#   eval    ~10 min, every other iteration
#   => ~11-13 h. Fits 24 h, but not with room to spare — check the log at 20 h
#      and resubmit if it timed out mid-run.
#
# Unlike steps 1-2 this job is genuinely GPU-bound during the refits (PointNet++),
# on top of needing CUDA for OMG at all.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-dagger4
# Resource block copied VERBATIM from collect_demos.sbatch, which passes
# `sbatch --test-only`. Deviating from it produced AssocMaxGRESPerJob even though
# the request (gres/gpu=1) sits well inside the association limit, so treat this
# block as load-bearing, not as defaults to tune. --gpus-per-task is MANDATORY
# (DelftBlue rejects --gres/--gpus outright); --nodes=1 is a warning, not an error.
#SBATCH --partition=gpu-v100          # see train_rl.sbatch for why not gpu-a100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # main process + TRAIN.num_workers DataLoader workers
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G              # 16G total
#SBATCH --output=slurm_logs/dagger4_%j.out
#SBATCH --error=slurm_logs/dagger4_%j.err

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
CFG="${CFG:-examples/configs/dagger_phase4.yaml}"
RUN="${RUN:-dagger4_run1}"
OUT_ROOT="${OUT_ROOT:-output/dagger_runs}"
BASE="${BASE:-}"                      # existing base run dir; empty => train it here
EXTRA="${EXTRA:-}"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cfg=${CFG}  run=${RUN}  out-root=${OUT_ROOT}  base=${BASE:-none}  extra=${EXTRA:-none}"

BASE_ARG=()
[ -n "${BASE}" ] && BASE_ARG=(--base-run "${BASE}")

srun python examples/train_dagger_phase4.py \
    --cfg-file "${CFG}" \
    --run-name "${RUN}" \
    --out-root "${OUT_ROOT}" \
    "${BASE_ARG[@]}" \
    ${EXTRA}
