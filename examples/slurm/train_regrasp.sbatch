#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — Phase-5 DAgger (examples/train_regrasp.py):
# grasp-conditioned policy, four pinned grasps per scene, regrasping.
# STEP 4, after the candidate table (1), the K-per-scene demonstrations (2) and
# select_pinned_grasps.py (3).
#
#   Default run — request 20 h, NOT the 24 h default (see the budget below):
#     sbatch --time=20:00:00 --export=ALL,RUN=dagger5_run1 examples/slurm/train_regrasp.sbatch
#   Short shakedown FIRST (2 iterations, m=8, 3 eval scenes x 4 grasps, ~30 min).
#   It is not optional here: Phase 5 changes the unit of work from a scene to a
#   (scene, grasp) pair in six places, and the smoke config's checklist is what
#   proves all six agree before a 20 h job queues behind them.
#   ALWAYS override --time here: the 24 h default cannot be backfilled, and with
#   fairshare depressed after a few GPU jobs that is the difference between
#   starting in an hour and starting in two days (measured: priority fell 46x
#   after four allocations in one afternoon, pushing the estimate to +2.5 days).
#     sbatch --time=01:00:00 --export=ALL,RUN=regrasp_smoke,CFG=examples/configs/regrasp_smoke.yaml \
#         examples/slurm/train_regrasp.sbatch
#   From an existing base run (skips the 100-epoch base fit):
#     sbatch --export=ALL,RUN=dagger5_run2,BASE=output/dagger_runs/dagger5_run1/iters/iter_00 \
#         examples/slurm/train_regrasp.sbatch
#   EVALUATION IS NO LONGER IN THIS JOB when the config sets EVAL.every: 0. It is
#   not on the critical path (train_regrasp.py hands the next collection the
#   fresh policy before eval runs), so it costs ~19% of wall clock for nothing.
#   Score the iterations alongside, in a second job:
#     sbatch --export=ALL,RUN=dagger5_run1 examples/slurm/eval_regrasp_run.sbatch
#   Watch:  tail -f slurm_logs/regrasp_<jobid>.out
#   Plot (safe while running — read-only):
#     python examples/plot_regrasp_run.py output/dagger_runs/<RUN>
#
# RESUMABLE. Re-running the SAME command continues: completed iterations are
# skipped, a finished-but-untrained collection is reused, and an interrupted
# training resumes from its last.pt. So a 24 h timeout is not a loss — just
# resubmit the identical command and it picks up where it stopped.
#
# Budget (25 iterations x m=200 episodes, eval SPLIT OFF into its own job):
#   base fit ~78 min      4x Phase-4's base data AND a random PointNet++ init
#   collect  ~2-3 min/iter with 20 workers (an OMG replan every step is the cost)
#   refit    grows with |D| to ~50 min at iteration 25; ~13.5 h in total
#   => ~15-16 h. Request --time=20:00:00. Resumable, so a timeout is not a loss.
#
# The aggregate reaches ~220k steps, 2.2x Phase-4 run 16's 99.5k, which is why
# iter_epochs drops 25 -> 12: total sample-passes then land within 13% of run
# 16's and the wall clock is comparable despite the extra data.
#
# DISK. ~119 MB per iteration at m=200 (checkpoints + the .h5 shard), so ~3 GB
# for the run. /home is a hard 30 GB quota that fills SILENTLY — always pass
# SCRATCH_ROOT.
#
# Unlike steps 1-2 this job is genuinely GPU-bound during the refits (PointNet++),
# on top of needing CUDA for OMG at all.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-regrasp
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
                                      # DAGGER.episodes_per_iter=200 (10 episodes each,
                                      # 100% utilization); 16 would leave some workers
                                      # with 13. Past ~20 the gain is minutes on a whole
                                      # run and the queue wait is worse — collection
                                      # drops to ~1 min/iteration and the FTL refit
                                      # becomes the entire cost. A V100 node has 48
                                      # cores across 4 GPUs, so 64 is NOT schedulable.
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G              # 16G total
#SBATCH --output=slurm_logs/regrasp_%j.out
#SBATCH --error=slurm_logs/regrasp_%j.err

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
CFG="${CFG:-examples/configs/regrasp_run1.yaml}"
RUN="${RUN:-dagger5_run1}"

# ── where the run writes — /home has a HARD 30 GB quota on DelftBlue ─────────
# Hitting it kills the job with exit code 6 and NO traceback: Python cannot flush
# one to a full disk, so the only symptom is a truncated .out file. A DAgger run
# is ~42 MB fixed + ~78 MB per iteration (41 MB checkpoints + 27 MB .h5), so 25
# iterations is ~2.0 GB and a handful of runs fills home. Send it to /scratch.
#
# SCRATCH_ROOT is the ONE knob, shared with train_rl.sbatch and
# eval_dagger_run.sbatch. Empty (the default) keeps the in-repo behaviour, so
# nothing differs when running outside the cluster:
#   sbatch --export=ALL,RUN=dagger5_run1,CFG=examples/configs/regrasp_run1.yaml,\
#       SCRATCH_ROOT=/scratch/pradyunsharma/handover-sim2real \
#       examples/slurm/train_regrasp.sbatch
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

srun python examples/train_regrasp.py \
    --cfg-file "${CFG}" \
    --run-name "${RUN}" \
    --out-root "${OUT_ROOT}" \
    --num-workers "${WORKERS}" \
    "${BASE_ARG[@]}" \
    ${EXTRA}
