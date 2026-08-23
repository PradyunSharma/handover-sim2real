#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — Regrasp DAgger (examples/train_regrasp.py):
# approach-direction-conditioned policy, ONE demonstration per reachable bin,
# retry by re-commanding. STEP 4, after the direction table (1), the demo
# assignment (2) and the base demonstrations + audit (3) — see README_REGRASP.md.
#
#   Default run — request 24 h (see the budget below; run 2 does not fit in 20):
#     sbatch --time=24:00:00 --export=ALL,RUN=regrasp_run2 examples/slurm/train_regrasp.sbatch
#   Short shakedown FIRST (2 iterations, m=8, 3 eval scenes, ~30 min). It is not
#   optional: Regrasp makes the unit of work a (scene, direction) pair in six
#   places, and the smoke config's checklist is what proves all six agree before
#   a 24 h job queues behind them.
#   ALWAYS override --time here: the 24 h default cannot be backfilled, and with
#   fairshare depressed after a few GPU jobs that is the difference between
#   starting in an hour and starting in two days (measured: priority fell 46x
#   after four allocations in one afternoon, pushing the estimate to +2.5 days).
#     sbatch --time=01:00:00 --export=ALL,RUN=regrasp_smoke,CFG=examples/configs/regrasp_smoke.yaml \
#         examples/slurm/train_regrasp.sbatch
#   From an existing base run (skips the 50-epoch base fit):
#     sbatch --export=ALL,RUN=regrasp_run3,BASE=output/dagger_runs/regrasp_run2/iters/iter_00 \
#         examples/slurm/train_regrasp.sbatch
#   Watch:  tail -f slurm_logs/regrasp_<jobid>.out
#   Plot (safe while running — read-only; writes five figures):
#     python examples/plot_regrasp_run.py output/dagger_runs/<RUN>
#
# EVALUATION IS BACK INSIDE THIS JOB. regrasp_run2.yaml sets `EVAL.every: 1`.
# Run 1 split it out because eval is off the critical path (train_regrasp.py
# hands the next collection the fresh policy before eval runs). It is back
# because with the split job the metrics arrive AFTER the next iteration has
# started, so `EVAL.ckpt` + `select_on: success_rate` never actually got to
# select — run 1's warm starts were decided on val loss alone.
# To go back to the split: set `EVAL.every: 0` and submit
#     sbatch --export=ALL,RUN=<RUN> examples/slurm/eval_regrasp_run.sbatch
# alongside. The plotter reads both layouts.
#
# THE HELD-OUT NUMBER IS NOT IN THIS JOB EITHER. `EVAL.holdout: false` means the
# in-loop scenes are also collected on, so the curves are TRAIN-SET rates by
# design. Score the s0 test split afterwards:
#     python examples/eval_regrasp_testset.py --run-dir <run> --chained
#
# RESUMABLE. Re-running the SAME command continues: completed iterations are
# skipped, a finished-but-untrained collection is reused, and an interrupted
# training resumes from its last.pt. So a 24 h timeout is not a loss — just
# resubmit the identical command and it picks up where it stopped.
#
# Budget (20 iterations, m=400 -> ~259 episodes, eval IN the loop):
#   base fit ~65 min      50 epochs on ~56k steps, random PointNet++ init
#   collect  ~4 min/iter with 20 workers (an OMG replan every step is the cost)
#   refit    15 epochs on a |D| growing to ~238k; ~45 min by iteration 20, ~13 h
#   eval     ~16 min/iter (100 scenes x 2.59 directions)     ~5.5 h
#   => ~20-21 h. REQUEST --time=24:00:00, NOT 20. Resumable, so a timeout costs
#      only the queue wait; if the queue is the problem, EVAL.every: 2 saves
#      2.75 h and halves the number of points on every curve.
#
# `episodes_per_iter: 400` is 100 scenes x max_grasps=4, and a scene supplies one
# episode per direction it can REACH (mean 2.59), so the real count is ~259. The
# refit and eval figures are extrapolated from run 1's LAPTOP timings — treat
# them as an upper bound on a V100.
#
# DISK. ~160 MB per iteration at ~259 episodes (checkpoints + the .h5 shard), so
# ~3.5 GB for the run. /home is a hard 30 GB quota that fills SILENTLY — always pass
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
                                      # main loop and the refit's DataLoader. The actual
                                      # episode count is ~259, not the configured m=400
                                      # (a scene contributes one episode per direction it
                                      # can reach, and the mean is 2.59 of 4), so no worker
                                      # count divides it exactly; 20 leaves the stragglers
                                      # at 13 while most take 12. Past ~20 the gain is
                                      # minutes on a whole
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
CFG="${CFG:-examples/configs/regrasp_run2.yaml}"
RUN="${RUN:-regrasp_run2}"

# ── where the run writes — /home has a HARD 30 GB quota on DelftBlue ─────────
# Hitting it kills the job with exit code 6 and NO traceback: Python cannot flush
# one to a full disk, so the only symptom is a truncated .out file. A DAgger run
# is ~42 MB fixed + ~78 MB per iteration (41 MB checkpoints + 27 MB .h5), so 25
# iterations is ~2.0 GB and a handful of runs fills home. Send it to /scratch.
#
# SCRATCH_ROOT is the ONE knob, shared with train_rl.sbatch and
# eval_dagger_run.sbatch. Empty (the default) keeps the in-repo behaviour, so
# nothing differs when running outside the cluster:
#   sbatch --export=ALL,RUN=regrasp_run2,CFG=examples/configs/regrasp_run2.yaml,\
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

# ── REGRASP_DATA: the HDF5 shards follow the run onto /scratch ───────────────
# The configs name their shards as `${REGRASP_DATA}/bc_dataset/...`, which
# expands at config load (handover_sim2real/regrasp/setup.py) and defaults to the
# in-repo `output/`. Pointing it at scratch alongside OUT_ROOT is what keeps the
# ~1.1 GB base set off /home's HARD 30 GB quota — a quota that fills SILENTLY and
# kills the job with exit code 6 and no traceback, because Python cannot write
# one to a full disk. The small JSON tables stay in the repo on purpose: they are
# inputs, they are worth version-controlling, and /scratch is purged by age.
export REGRASP_DATA="${REGRASP_DATA:-${SCRATCH_ROOT:+${SCRATCH_ROOT}/output}}"
export REGRASP_DATA="${REGRASP_DATA:-output}"
mkdir -p "${REGRASP_DATA}/bc_dataset"
echo "REGRASP_DATA=${REGRASP_DATA}"

BASE="${BASE:-}"                      # existing base run dir; empty => train it here
WORKERS="${WORKERS:-$(( SLURM_CPUS_PER_TASK - 2 ))}"   # parallel collection processes
EXTRA="${EXTRA:-}"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cfg=${CFG}  run=${RUN}  out-root=${OUT_ROOT}  base=${BASE:-none}  extra=${EXTRA:-none}"
echo "cpus-per-task=${SLURM_CPUS_PER_TASK}  collection workers=${WORKERS}"

# ── derive SIM.demo_ok_table if the config asks for one and it is missing ────
# It is a deterministic function of the base shard — the (scene, bin) pairs the
# collection actually demonstrated — so making the job produce it removes a
# manual step from between two 3-hour jobs, which is exactly the kind of gap a
# pipeline forgets. Regenerated only when ABSENT: an existing file is the record
# of what a particular collection achieved and must not be silently rewritten
# from a shard that has since been extended.
DEMO_OK="$(python - "${CFG}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print((c.get("SIM") or {}).get("demo_ok_table") or "")
PY
)"
BASE_H5="$(python - "${CFG}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
print((c.get("TRAIN") or {}).get("base_train_h5") or "")
PY
)"
if [ -n "${DEMO_OK}" ] && [ ! -f "${DEMO_OK}" ]; then
    if [ ! -f "${BASE_H5}" ]; then
        echo "ERROR: ${CFG} wants ${DEMO_OK} but ${BASE_H5} does not exist —" >&2
        echo "  collect the base demonstrations first." >&2
        exit 1
    fi
    echo "[prep] ${DEMO_OK} missing — deriving it from ${BASE_H5}"
    # The audit EXITS 1 on a schema failure, and that is the right moment to
    # stop: training on a shard the audit rejects wastes the whole allocation.
    python examples/audit_regrasp_demos.py --demos "${BASE_H5}" --write-ok "${DEMO_OK}"
fi

BASE_ARG=()
[ -n "${BASE}" ] && BASE_ARG=(--base-run "${BASE}")

srun python examples/train_regrasp.py \
    --cfg-file "${CFG}" \
    --run-name "${RUN}" \
    --out-root "${OUT_ROOT}" \
    --num-workers "${WORKERS}" \
    "${BASE_ARG[@]}" \
    ${EXTRA}
