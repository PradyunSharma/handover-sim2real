#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — re-collect the OMG expert demonstrations
# (examples/collect_bc_dataset.py) with the Phase-3/4 grasp set AND the per-scene
# pin table. This is STEP 2, and it DEPENDS on the pin table from STEP 1
# (examples/slurm/build_pin_table.sbatch).
#
#   Train split (D_0 for Phase 4):
#     sbatch --export=ALL,SPLIT=train,OUT=output/bc_dataset/train_pinned.h5,PIN=output/grasp_pin_table_train.json \
#         examples/slurm/collect_bc_demos.sbatch
#   Val split (within-run best.pt selection):
#     sbatch --export=ALL,SPLIT=val,OUT=output/bc_dataset/val_pinned.h5,PIN=output/grasp_pin_table_val.json \
#         examples/slurm/collect_bc_demos.sbatch
#   Chained behind the table build (preferred — no manual waiting):
#     JID=$(sbatch --parsable --export=ALL,SPLIT=train,OUT=output/grasp_pin_table_train.json \
#             examples/slurm/build_pin_table.sbatch)
#     sbatch --dependency=afterok:$JID --export=ALL,SPLIT=train,OUT=output/bc_dataset/train_pinned.h5,PIN=output/grasp_pin_table_train.json \
#         examples/slurm/collect_bc_demos.sbatch
#   Watch:  tail -f slurm_logs/bcdemos_<jobid>.out
#
# WHY RE-COLLECT. The existing output/bc_dataset/train.h5 was collected with
# NEITHER grasp filter wired in, while Phase 3/4 use valid_grasp_dict_005.pkl.
# Measured over 72 scenes: 21% of scenes had a DIFFERENT goal grasp between the
# demos and DAgger (median 10.7 cm apart, max 20.5 cm — all far outside the 2 cm
# close threshold). Aggregating those two label sets is exactly the inconsistency
# DAgger cannot average away.
#
# SERIAL — one env, one scene at a time (~720 train scenes) → plan for many hours.
# The h5 is written incrementally, so a crash/timeout keeps every episode already
# collected. No --egl: it leaks ~85 MB/scene (OOM over a full split), AND the point
# cloud is renderer-dependent — this MUST match SIM.egl in dagger_phase4.yaml
# (false), or the DAgger policy sees a different input distribution than it trained on.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-bcdemos
# A GPU IS MANDATORY — not for a neural network (the expert is OMG, the cloud
# comes from PyBullet's CPU renderer), but for OMG ITSELF: importing omg.config
# calls .cuda() at module scope (config.py:222) and the SDF collision cost is
# CUDA throughout. A CPU partition dies at import with "Found no NVIDIA driver".
#SBATCH --partition=gpu-v100          # see train_rl.sbatch for why not gpu-a100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
# Resource block copied VERBATIM from collect_demos.sbatch, which passes
# `sbatch --test-only`. Deviating from it produced AssocMaxGRESPerJob even though
# the request (gres/gpu=1) sits well inside the association limit (gres/gpu=2),
# so treat this block as load-bearing, not as defaults to tune. Specifically:
# --gpus-per-task is MANDATORY (DelftBlue rejects --gres/--gpus outright), and
# --nodes=1 is only a WARNING on gpu-v100 (it is an error on compute-p1, but a
# CPU partition cannot run this job at all — OMG needs CUDA).
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # serial script — a few cores for PyBullet/python
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G              # 16G total
#SBATCH --output=slurm_logs/bcdemos_%j.out
#SBATCH --error=slurm_logs/bcdemos_%j.err

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
# Unbuffered: without this, python's stdout is block-buffered when redirected to
# a file, so a 48 h collection shows no progress until it exits — and a crash
# leaves a truncated log that stops at an arbitrary point.
export PYTHONUNBUFFERED=1

# ── parameters (override via sbatch --export=ALL,...) ───────────────────────
SIM_CFG="${SIM_CFG:-examples/pretrain.yaml}"   # MUST match SIM.cfg_file in dagger_phase4.yaml
SPLIT="${SPLIT:-train}"
OUT="${OUT:-output/bc_dataset/${SPLIT}_pinned.h5}"
VGD="${VGD:-examples/valid_grasp_dict_005.pkl}"          # Phase-3/4 grasp set
PIN="${PIN:-output/grasp_pin_table_${SPLIT}.json}"       # one committed grasp per scene
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"                    # e.g. EXTRA="--num-episodes 20" for a smoke run

if [ ! -f "${PIN}" ]; then
    echo "ERROR: pin table ${PIN} not found. Run build_pin_table.sbatch first" >&2
    exit 1
fi

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "sim-cfg=${SIM_CFG}  split=${SPLIT}  out=${OUT}"
echo "vgd=${VGD}  pin=${PIN}  seed=${SEED}  extra=${EXTRA:-none}"
mkdir -p "$(dirname "${OUT}")"

srun python examples/collect_bc_dataset.py \
    --cfg-file          "${SIM_CFG}" \
    --output            "${OUT}" \
    --split             "${SPLIT}" \
    --seed              "${SEED}" \
    --valid-grasp-dict  "${VGD}" \
    --grasp-pin-table   "${PIN}" \
    ${EXTRA}
