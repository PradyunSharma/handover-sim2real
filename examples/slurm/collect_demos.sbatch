#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — recollect expert RL demos (collect_rl_demos.py).
#
#   Smoke (~20 scenes, validates a new grasp filter quickly):
#     sbatch --export=ALL,RL_CFG=examples/configs/rl_phase1_cluster_vgd.yaml,OUT=output/rl_demos/smoke_vgd.h5,EXTRA="--num-scenes 20" \
#         examples/slurm/collect_demos.sbatch
#   Full train split:
#     sbatch --export=ALL,RL_CFG=examples/configs/rl_phase1_cluster_vgd.yaml,OUT=output/rl_demos/train_h30_vgd.h5 \
#         examples/slurm/collect_demos.sbatch
#   Watch:  tail -f slurm_logs/demos_<jobid>.out
#
# SERIAL — one env, one scene at a time (~720 train scenes) → plan for many hours.
# The h5 is written incrementally, so a crash/timeout keeps every episode already
# collected. Runs on a V100 node with the default --device cuda: collection itself
# is expert-only (OMG + PyBullet, CPU-bound), but cuda is the environment this repo
# is proven on here. No --egl (leaks ~85 MB/scene → OOM over a full split).
# NOTE: this occupies 1 of the account's 2 concurrent GPUs while it runs.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-demos
#SBATCH --partition=gpu-v100          # see train_rl.sbatch for why not gpu-a100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # serial script — a few cores for PyBullet/python
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G              # 16G total
#SBATCH --output=slurm_logs/demos_%j.out
#SBATCH --error=slurm_logs/demos_%j.err

set -euo pipefail
mkdir -p slurm_logs

# ── environment (identical to train_rl.sbatch — keep in sync) ───────────────
module load miniconda3 2>/dev/null || true
source "$(conda info --base 2>/dev/null || echo /apps/generic/miniforge3/25.11.0)/etc/profile.d/conda.sh"
conda activate pch2r_dev
# DO NOT TRUST AN INHERITED VALUE. `${VAR:-default}` only fires when VAR is
# UNSET, and sbatch --export=ALL carries the submitting shell's environment onto
# the compute node — so an OMG_PLANNER_DIR exported from some other directory in
# that login session silently wins over the $PWD default and travels with the
# job. add_sys_path_from_env (utils.py:10) then asserts only that the variable
# EXISTS and appends it verbatim, so the run gets as far as printing its config
# banner and dies at `import omg` inside gym.make() with
# "ModuleNotFoundError: No module named 'omg'" — a message that names a
# dependency when the real problem is a path. Validate, then fall back.
for _v in OMG_PLANNER_DIR:omg GADDPG_DIR:core; do
    _name="${_v%%:*}"; _pkg="${_v##*:}"
    _cur="$(eval echo "\${${_name}:-}")"
    _def="$PWD/$( [ "$_name" = OMG_PLANNER_DIR ] && echo OMG-Planner || echo GA-DDPG )"
    if [ -n "${_cur}" ] && [ ! -d "${_cur}/${_pkg}" ]; then
        echo "WARNING: inherited ${_name}=${_cur} has no ${_pkg}/ — using ${_def}" >&2
        _cur=""
    fi
    export "${_name}=${_cur:-${_def}}"
    _res="$(eval echo "\${${_name}}")"
    if [ ! -d "${_res}/${_pkg}" ]; then
        echo "ERROR: ${_name}=${_res} has no ${_pkg}/ package." >&2
        echo "  If the directory is empty this is an unpopulated git submodule:" >&2
        echo "    git submodule update --init --recursive" >&2
        exit 1
    fi
    echo "${_name}=${_res}  (${_pkg}/ present)"
done
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/OMG-Planner/orocos_kinematics_dynamics/orocos_kdl/release/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="7.0"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── parameters (override via sbatch --export=ALL,...) ───────────────────────
SIM_CFG="${SIM_CFG:-examples/pretrain.yaml}"   # pretrain_multicam.yaml for the fixed-camera 6-ch cloud
RL_CFG="${RL_CFG:-examples/configs/rl_phase1_cluster_vgd.yaml}"
OUT="${OUT:-output/rl_demos/train_h30_vgd.h5}"
BC_RUN="${BC_RUN:-output/bc_runs/dagger_iter_2_3}"   # supplies the (camera-independent) action/state normalizer
SPLIT="${SPLIT:-train}"
EXTRA="${EXTRA:-}"                    # e.g. EXTRA="--num-scenes 20" for a smoke run

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "sim-cfg=${SIM_CFG}  rl-cfg=${RL_CFG}  out=${OUT}  split=${SPLIT}  extra=${EXTRA:-none}"
mkdir -p "$(dirname "${OUT}")"

srun python examples/collect_rl_demos.py \
    --sim-cfg "${SIM_CFG}" \
    --bc-run  "${BC_RUN}" \
    --rl-cfg  "${RL_CFG}" \
    --out     "${OUT}" \
    --split   "${SPLIT}" \
    ${EXTRA}
