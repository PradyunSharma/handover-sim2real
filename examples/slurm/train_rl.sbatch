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
#SBATCH --partition=gpu-a100          # A100 partition (see `sinfo`); or `gpu`
#SBATCH --account=education-me-msc-ro  # DelftBlue allocation (sacctmgr show assoc user=$USER)
#SBATCH --qos=normal
#SBATCH --time=24:00:00               # walltime cap; checkpoints let you --resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18            # 16 rollout workers + 2 for learner/main
#SBATCH --gpus-per-task=1             # 1 A100 for the learner
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
module load 2023r1 || true
module load miniconda3 || true
# module load cuda/11.7 || true       # only if your torch build needs system CUDA

source activate pch2r_dev             # the project's conda env (README_MY.md)

# OMG-Planner / GA-DDPG locations (README_MY.md). If your conda activate.d already
# exports these, these lines are harmless no-ops.
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-$PWD/OMG-Planner}"
export GADDPG_DIR="${GADDPG_DIR:-$PWD/GA-DDPG}"

# One math thread per process — the manager sets these too, but exporting here
# guarantees every child (and the OMG planner's numpy/scipy) is pinned.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── run ─────────────────────────────────────────────────────────────────────
WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))     # leave 2 cores for learner + main loop
RUN_NAME="${RUN_NAME:-rl_run11}"           # override: sbatch --export=ALL,RUN_NAME=rl_run12 ...

echo "run=${RUN_NAME}"
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cpus-per-task=${SLURM_CPUS_PER_TASK}  rollout workers=${WORKERS}"

srun python examples/train_rl.py \
    --sim-cfg examples/pretrain.yaml \
    --rl-cfg  examples/configs/rl_phase1_cluster.yaml \
    --bc-run  output/bc_runs/dagger_iter_2_3 \
    --demos   output/rl_demos/train_h30.h5 \
    --run-name "${RUN_NAME}" \
    --num-workers "${WORKERS}" \
    --worker-device cpu \
    --episodes-per-iter "${WORKERS}" \
    --updates-per-iter "$(( WORKERS * 50 ))"

# Resume after a walltime kill: add
#   --resume output/rl_runs/${RUN_NAME}/checkpoints/last.pt
