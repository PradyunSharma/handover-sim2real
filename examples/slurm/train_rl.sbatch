#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — Phase-3 RL training (train_rl.py) with parallel rollouts.
#
#   Submit:   sbatch examples/slurm/train_rl.sbatch
#   Watch:    squeue --me         # queue/running state
#             tail -f slurm_logs/rl_<jobid>.out
#   Cancel:   scancel <jobid>
#
# Docs: https://doc.dhpc.tudelft.nl/delftblue/Slurm-scheduler/
#
# This runs ONE learner (on the GPU) + WORKERS parallel rollout worker processes
# (on the CPUs) — the analog of the paper's num_remotes. Keep --num-workers ==
# episodes_per_iter in the RL config (rl_phase1_cluster.yaml is tuned for 16), and
# request cpus-per-task = WORKERS + 2 (learner + main loop want a couple of cores).
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-rl
#SBATCH --partition=gpu-v100          # V100 nodes. NOTE: the education-me-msc-ro
                                      # account is NOT granted gpu-a100 (any GPU
                                      # request there -> pending AssocMaxGRESPerJob);
                                      # it CAN get a full V100S 32GB here with qos=normal.
#SBATCH --account=education-me-msc-ro  # DelftBlue allocation (sacctmgr show assoc user=$USER)
#SBATCH --qos=normal                  # account has normal,reservation (NOT gpu-a100-small)
#SBATCH --time=24:00:00               # education GPU cap is 24h; checkpoints let you --resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18            # 16 rollout workers + 2 for learner/main (V100 node = 48 cores/4 GPUs)
#SBATCH --gpus-per-task=1             # 1 V100S (32GB) for the learner
#SBATCH --mem-per-cpu=4G              # 18 * 4G = 72G (envs are ~1.1G CPU-renderer each)
#SBATCH --output=slurm_logs/rl_%j.out
#SBATCH --error=slurm_logs/rl_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@tudelft.nl

set -euo pipefail
mkdir -p slurm_logs

# ── environment ─────────────────────────────────────────────────────────────
# Adjust to your setup. On DelftBlue conda usually comes from the miniconda3
# module; if you installed your own miniconda, source that instead.
module load miniconda3 2>/dev/null || true
# module load cuda/11.7 || true       # only if your torch build needs system CUDA

# conda's `activate` wrapper is gone in miniforge; source the hook, then `conda activate`.
# `conda info --base` resolves once the module is on PATH; the literal path is a fallback.
source "$(conda info --base 2>/dev/null || echo /apps/generic/miniforge3/25.11.0)/etc/profile.d/conda.sh"
conda activate pch2r_dev              # the project's conda env (README_MY.md)

# OMG-Planner / GA-DDPG locations (README_MY.md). If your conda activate.d already
# exports these, these lines are harmless no-ops.
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-$PWD/OMG-Planner}"
export GADDPG_DIR="${GADDPG_DIR:-$PWD/GA-DDPG}"

# 'handover' (handover-sim/) and 'handover_sim2real' live in the repo and are NOT
# pip-installed, so put the repo root and handover-sim on PYTHONPATH (README_MY.md).
# A non-login batch shell does not inherit the PYTHONPATH your interactive shell sets.
export PYTHONPATH="$PWD:$PWD/handover-sim:$PWD/handover-sim/mano_pybullet:${PYTHONPATH:-}"

# Native shared libs the loader must find at runtime:
#  - $CONDA_PREFIX/lib : libassimp.so (pyassimp mesh loader) + a modern libstdc++
#    (GLIBCXX_3.4.29+) needed by the cluster-rebuilt CUDA/C++ extensions.
#  - orocos_kdl/release/lib : liborocos-kdl.so.1.4, needed by OMG-Planner's PyKDL
#    (also needs the `sip` pkg installed in the env).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PWD/OMG-Planner/orocos_kinematics_dynamics/orocos_kdl/release/lib:${LD_LIBRARY_PATH:-}"

# pointnet2_ops (GA-DDPG PointNet++) JIT-compiles its CUDA _ext on first import via
# torch.utils.cpp_extension.load (needs `ninja` + `nvcc`, both in pch2r_dev). Pin the
# arch to the V100 (sm_70) so the build is deterministic and the compiled extension in
# ~/.cache/torch_extensions is shared across the learner + all workers (compiled once).
export TORCH_CUDA_ARCH_LIST="7.0"

# One math thread per process — the manager sets these too, but exporting here
# guarantees every child (and the OMG planner's numpy/scipy) is pinned.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── run ─────────────────────────────────────────────────────────────────────
WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))     # leave 2 cores for learner + main loop
RUN_NAME="${RUN_NAME:-rl_run11}"           # override: sbatch --export=ALL,RUN_NAME=rl_run12 ...
# RL config path — overridable so different runs use different configs WITHOUT editing
# the shared file (a --resume re-reads --rl-cfg, so editing it mid-run would change the
# resumed run's behavior). Override: sbatch --export=ALL,RUN_NAME=rl_run17,RL_CFG=examples/configs/rl_phase1_cluster_both.yaml ...
RL_CFG="${RL_CFG:-examples/configs/rl_phase1_cluster.yaml}"
# Demo pool — overridable like RL_CFG so a run can pair its hand-filter with the
# matching demos WITHOUT editing this file. run15 (0.08 m runtime filter): train_h30.h5.
# run21/run23 (paper valid_grasp_dict): train_h30_vgd.h5.
DEMOS="${DEMOS:-output/rl_demos/train_h30.h5}"

echo "run=${RUN_NAME}"
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cpus-per-task=${SLURM_CPUS_PER_TASK}  rollout workers=${WORKERS}"

# NOTE: --worker-device MUST be cuda. The actor's PointNet++ encoder uses
# pointnet2_ops furthest-point-sampling, which is CUDA-only (no CPU kernel), so a
# cpu worker crashes on every POLICY episode and it's silently counted as skipped
# (skip=N, buf=0, no learning). Workers share the one GPU (N small actor contexts +
# the learner); fine on a 32GB V100 at 16 workers.
echo "rl-cfg=${RL_CFG}"
echo "demos=${DEMOS}"
srun python examples/train_rl.py \
    --sim-cfg examples/pretrain.yaml \
    --rl-cfg  "${RL_CFG}" \
    --bc-run  output/bc_runs/dagger_iter_2_3 \
    --demos   "${DEMOS}" \
    --run-name "${RUN_NAME}" \
    --num-workers "${WORKERS}" \
    --worker-device cuda \
    --episodes-per-iter "${WORKERS}" \
    --updates-per-iter "$(( WORKERS * 50 ))"

# Resume after a walltime kill: add
#   --resume output/rl_runs/${RUN_NAME}/checkpoints/last.pt
