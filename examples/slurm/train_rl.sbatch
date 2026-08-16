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
# Simulator config — overridable. Default = egocentric wrist cloud (pretrain.yaml).
# For the fixed-camera 6-ch cloud (run33) pass SIM_CFG=examples/pretrain_multicam.yaml
# (and an RL config with DATA.pc_channels: 6). MUST match the cloud the DEMOS were
# collected with (a 6-ch multicam demo pool needs the multicam sim cfg, and vice-versa).
SIM_CFG="${SIM_CFG:-examples/pretrain.yaml}"
# Overfit / capability probe — cap rollouts+eval to the FIRST N scenes. 0 = all scenes.
# Pair with a demo pool collected on the same first-N scenes (collect_demos EXTRA=--num-scenes N).
NUM_SCENES="${NUM_SCENES:-0}"

# ── where the run writes — /home has a HARD 30 GB quota on DelftBlue ─────────
# Hitting it kills the job with exit code 6 and NO traceback: Python cannot flush
# one to a full disk, so the only symptom is a truncated .out file. An RL run is
# GBs of checkpoints, so send it to /scratch (5 TiB) instead.
#
# SCRATCH_ROOT is the ONE knob. Empty (the default) keeps the in-repo behaviour
# unchanged, so nothing differs when running outside the cluster:
#   sbatch --export=ALL,RUN_NAME=rl_run40,\
#       SCRATCH_ROOT=/scratch/pradyunsharma/handover-sim2real examples/slurm/train_rl.sbatch
# OUT_ROOT can still be set directly and wins over SCRATCH_ROOT if you need a
# path that is not <root>/output/rl_runs.
#
# /scratch is PERIODICALLY PURGED BY AGE. Checkpoints are meant to be transient
# there, but the run's metadata is the experiment record — bring it back into git
# with:  python examples/harvest_run.py --run-dir "${OUT_ROOT}/${RUN_NAME}"
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
if [ -n "${SCRATCH_ROOT}" ]; then
    OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/output/rl_runs}"
else
    OUT_ROOT="${OUT_ROOT:-output/rl_runs}"
fi
mkdir -p "${OUT_ROOT}"

echo "run=${RUN_NAME}  out-root=${OUT_ROOT}"
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cpus-per-task=${SLURM_CPUS_PER_TASK}  rollout workers=${WORKERS}"

# NOTE: --worker-device MUST be cuda. The actor's PointNet++ encoder uses
# pointnet2_ops furthest-point-sampling, which is CUDA-only (no CPU kernel), so a
# cpu worker crashes on every POLICY episode and it's silently counted as skipped
# (skip=N, buf=0, no learning). Workers share the one GPU (N small actor contexts +
# the learner); fine on a 32GB V100 at 16 workers.
echo "rl-cfg=${RL_CFG}"
echo "sim-cfg=${SIM_CFG}"
echo "demos=${DEMOS}"
echo "num-scenes=${NUM_SCENES} (0=all)"
srun python examples/train_rl.py \
    --sim-cfg "${SIM_CFG}" \
    --rl-cfg  "${RL_CFG}" \
    --bc-run  output/bc_runs/dagger_iter_2_3 \
    --demos   "${DEMOS}" \
    --run-name "${RUN_NAME}" \
    --out-root "${OUT_ROOT}" \
    --num-workers "${WORKERS}" \
    --worker-device cuda \
    --episodes-per-iter "${WORKERS}" \
    --updates-per-iter "$(( WORKERS * 50 ))" \
    --num-scenes "${NUM_SCENES}"

# Resume after a walltime kill: add
#   --resume ${OUT_ROOT}/${RUN_NAME}/checkpoints/last.pt
# NOT output/rl_runs/... — with SCRATCH_ROOT set the run does not live in the
# repo, and a hardcoded path here would silently resume from nothing (or from a
# stale same-named run in home). Resubmit with the SAME SCRATCH_ROOT/OUT_ROOT and
# the path resolves itself.
