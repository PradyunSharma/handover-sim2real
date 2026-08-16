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
#   EVALUATION IS NO LONGER IN THIS JOB when the config sets EVAL.every: 0. It is
#   not on the critical path (train_dagger_phase4.py hands the next collection the
#   fresh policy before eval runs), so it costs ~19% of wall clock for nothing.
#   Score the iterations alongside, in a second job:
#     sbatch --export=ALL,RUN=dagger4_run6 examples/slurm/eval_dagger_run.sbatch
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
#SBATCH --cpus-per-task=22            # 20 parallel collection workers + 2 for the
                                      # main loop and the refit's DataLoader. 20 DIVIDES
                                      # DAGGER.episodes_per_iter=100 (5 episodes each,
                                      # 100% utilization); 16 would leave some workers
                                      # with 7. Past ~20 the gain is minutes on a whole
                                      # run and the queue wait is worse — collection
                                      # drops to ~1 min/iteration and the FTL refit
                                      # becomes the entire cost. A V100 node has 48
                                      # cores across 4 GPUs, so 64 is NOT schedulable.
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
# Unbuffered: a 24 h run must show progress live, and a crash must leave a log
# that stops where it actually failed rather than at an arbitrary buffer edge.
export PYTHONUNBUFFERED=1

# ── parameters (override via sbatch --export=ALL,...) ───────────────────────
CFG="${CFG:-examples/configs/dagger_phase4.yaml}"
RUN="${RUN:-dagger4_run1}"

# ── where the run writes — /home has a HARD 30 GB quota on DelftBlue ─────────
# Hitting it kills the job with exit code 6 and NO traceback: Python cannot flush
# one to a full disk, so the only symptom is a truncated .out file. A DAgger run
# is ~42 MB fixed + ~78 MB per iteration (41 MB checkpoints + 27 MB .h5), so 25
# iterations is ~2.0 GB and a handful of runs fills home. Send it to /scratch.
#
# SCRATCH_ROOT is the ONE knob, shared with train_rl.sbatch and
# eval_dagger_run.sbatch. Empty (the default) keeps the in-repo behaviour, so
# nothing differs when running outside the cluster:
#   sbatch --export=ALL,RUN=dagger4_run15,CFG=examples/configs/dagger_phase4_all.yaml,\
#       SCRATCH_ROOT=/scratch/pradyunsharma/handover-sim2real \
#       examples/slurm/train_dagger_phase4.sbatch
# OUT_ROOT can still be set directly and wins, for a path that is not
# <root>/output/dagger_runs.
#
# INPUTS ARE UNAFFECTED. The config's grasp_pin_table and bc_dataset paths are
# read relative to the repo as before; only the run's OUTPUT moves.
#
# /scratch is PERIODICALLY PURGED BY AGE, so bring the small metadata back into
# git when the run finishes (or alongside it — the script is safe mid-run):
#   python examples/harvest_run.py --run-dir "${OUT_ROOT}/${RUN}"
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
if [ -n "${SCRATCH_ROOT}" ]; then
    OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/output/dagger_runs}"
else
    OUT_ROOT="${OUT_ROOT:-output/dagger_runs}"
fi
mkdir -p "${OUT_ROOT}"
BASE="${BASE:-}"                      # existing base run dir; empty => train it here
WORKERS="${WORKERS:-$(( SLURM_CPUS_PER_TASK - 2 ))}"   # parallel collection processes
EXTRA="${EXTRA:-}"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cfg=${CFG}  run=${RUN}  out-root=${OUT_ROOT}  base=${BASE:-none}  extra=${EXTRA:-none}"
echo "cpus-per-task=${SLURM_CPUS_PER_TASK}  collection workers=${WORKERS}"

BASE_ARG=()
[ -n "${BASE}" ] && BASE_ARG=(--base-run "${BASE}")

srun python examples/train_dagger_phase4.py \
    --cfg-file "${CFG}" \
    --run-name "${RUN}" \
    --out-root "${OUT_ROOT}" \
    --num-workers "${WORKERS}" \
    "${BASE_ARG[@]}" \
    ${EXTRA}
