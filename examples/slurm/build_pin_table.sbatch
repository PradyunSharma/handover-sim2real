#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — build the per-scene grasp pin table
# (examples/build_grasp_pin_table.py). This is STEP 1: the demonstration
# collection and the Phase-4 DAgger loop both load the table, so it has to exist
# before either runs.
#
#   Train split (the one Phase 4 uses):
#     sbatch --export=ALL,SPLIT=train,OUT=output/grasp_pin_table_train.json \
#         examples/slurm/build_pin_table.sbatch
#   Val split (for the val demos used in within-run checkpoint selection):
#     sbatch --export=ALL,SPLIT=val,OUT=output/grasp_pin_table_val.json \
#         examples/slurm/build_pin_table.sbatch
#   Smoke first (~20 scenes, ~2 min — proves the env before burning 24 h):
#     sbatch --export=ALL,SPLIT=train,OUT=output/pin_smoke.json,EXTRA="--num-scenes 20" \
#         examples/slurm/build_pin_table.sbatch
#   Watch:  tail -f slurm_logs/pintable_<jobid>.out
#
# ONE TABLE PER SPLIT. Scene indices are split-relative, so a train table applied
# to val pins the wrong grasps; loading a mismatched table is a hard error.
#
# SERIAL — one env, one OMG plan per scene (~720 train scenes at ~1-3 s each →
# ~30-60 min). The JSON is rewritten every 25 scenes, so a timeout keeps most of
# the work. No --egl (leaks ~85 MB/scene → OOM over a full split), and the table
# does not depend on the renderer anyway.
#
# REBUILD whenever SIM.valid_grasp_dict_path or hand_collision_filter changes —
# those change which grasp candidates exist.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-pintable
#SBATCH --partition=gpu-v100          # see train_rl.sbatch for why not gpu-a100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=04:00:00
# No --nodes: compute-p1 rejects a per-node request that is not --exclusive
# ("Specify '--ntasks' and '--cpus-per-task' instead"). --ntasks=1 already lands
# this on a single node, so --nodes was redundant on the GPU partitions too.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3968M           # 15.5G total. NOT 4G: compute-p1 caps
                                      # mem-per-cpu at 190464MB/48 = 3968MB and
                                      # rejects the job outright. 3968 is under
                                      # the gpu-v100 cap (5333) too, so this one
                                      # value works on either partition.
#SBATCH --output=slurm_logs/pintable_%j.out
#SBATCH --error=slurm_logs/pintable_%j.err

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

# ── parameters (override via sbatch --export=ALL,...) ───────────────────────
CFG="${CFG:-examples/configs/dagger_phase4.yaml}"   # its SIM block defines the grasp set
SPLIT="${SPLIT:-train}"
OUT="${OUT:-output/grasp_pin_table_${SPLIT}.json}"
MODE="${MODE:-furthest_from_hand}"    # furthest_from_hand | omg
TOL="${TOL:-0.01}"                    # near-tie tolerance (m), resolved toward OMG's
                                      # pick. 0 would move a grasp 8.5 cm to gain
                                      # 0.1 mm of clearance; 0.01 suppresses that.
EXTRA="${EXTRA:-}"                    # e.g. EXTRA="--num-scenes 20"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cfg=${CFG}  split=${SPLIT}  out=${OUT}  mode=${MODE}  tol=${TOL}  extra=${EXTRA:-none}"
mkdir -p "$(dirname "${OUT}")"

srun python examples/build_grasp_pin_table.py \
    --cfg-file "${CFG}" \
    --split    "${SPLIT}" \
    --out      "${OUT}" \
    --mode     "${MODE}" \
    --tol      "${TOL}" \
    ${EXTRA}
