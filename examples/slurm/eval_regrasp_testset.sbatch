#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — score a finished Regrasp run on the HELD-OUT s0 TEST
# split (examples/eval_regrasp_testset.py). STEP 5, after training.
#
#   START HERE — a few iterations, independent + the chained ladder (~2 h). ITERS
#   must be quoted, or sbatch parses the commas as more --export assignments:
#     sbatch --export=ALL,RUN=regrasp_run2,CHAINED=1,"ITERS=0,5,10,15,20" \
#         examples/slurm/eval_regrasp_testset.sbatch
#   The final policy only (~20 min):
#     sbatch --export=ALL,RUN=regrasp_run2,CHAINED=1,ITERS=20 \
#         examples/slurm/eval_regrasp_testset.sbatch
#   Every iteration, independent only (~7.5 h) — a full test-split curve:
#     sbatch --export=ALL,RUN=regrasp_run2 examples/slurm/eval_regrasp_testset.sbatch
#   Watch:  tail -f slurm_logs/regrasp_test_%j.out
#
# RESUMABLE AND INCREMENTAL. Iterations already in test_eval_log.csv are skipped,
# so scoring five now and the rest later costs nothing extra, and the figure is
# re-rendered from whatever the CSV holds. --plot-only re-renders without the
# simulator at all, which runs fine on a laptop.
#
# WHY THIS IS NOT THE IN-LOOP NUMBER. The training curves run with
# `EVAL.holdout: false` on the TRAIN split, so their scenes are also collected
# on — by design, because that makes them a clean read of "is the policy getting
# better at what it is being taught". This job scores the benchmark's own test
# split, which the run has never seen in any form. Expect lower numbers, and do
# NOT read the difference as a regression.
#
# COST, AND WHY ITERS IS THE FIRST THING TO SET. s0/test is 8 subjects x 20
# sequences x 2 hands = 160 scenes, 144 after the two skipped objects, of which
# ~124 plan. At the measured 2.59 reachable bins per scene that is ~320 episodes
# an iteration at ~4 s each: ~21 min independent, ~35 min chained. All 21
# iterations is 7.5 h / 12 h. The test curve does not need every point — five
# iterations spread across the run show the trend for a tenth of the cost.
#
# PREREQUISITE — the test split needs its own tables, and that is a GPU sim pass:
#     sbatch --export=ALL,SPLIT=test,OUT=output/direction_table_test.json \
#         examples/slurm/build_direction_table.sbatch
#     python examples/assign_direction_demos.py --table output/direction_table_test.json \
#         --out output/regrasp_pins_test --mode per-bin \
#         --drop-bins='-z_beneath,-x_over_fingers'
# There is deliberately NO demo_ok_table on test: nothing is collected there, so
# the pin table itself is the feasibility statement.
#
# ONE GPU, FEW CORES: a serial loop over test scenes with no worker pool. It is
# off every critical path, so its own latency does not matter.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-regrasp-test
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=13:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/regrasp_test_%j.out
#SBATCH --error=slurm_logs/regrasp_test_%j.err

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
RUN="${RUN:?set RUN=<run name>, e.g. --export=ALL,RUN=regrasp_run2}"
# Same SCRATCH_ROOT knob as train_regrasp.sbatch — pass the SAME value the
# trainer was submitted with, or this job looks for the run in the repo and exits
# with "config.yaml not found". Empty default = unchanged in-repo behaviour.
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
if [ -n "${SCRATCH_ROOT}" ]; then
    OUT_ROOT="${OUT_ROOT:-${SCRATCH_ROOT}/output/dagger_runs}"
else
    OUT_ROOT="${OUT_ROOT:-output/dagger_runs}"
fi
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${RUN}}"

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

SPLIT="${SPLIT:-test}"
ITERS="${ITERS:-all}"
CKPT="${CKPT:-last}"                  # matches EVAL.ckpt from run 2 on
CHAINED="${CHAINED:-0}"               # 1 = also run the retry ladder
REWIND="${REWIND:-0.30}"
MAXATT="${MAXATT:-4}"
EXTRA="${EXTRA:-}"

if [ ! -f "${RUN_DIR}/config.yaml" ]; then
    echo "ERROR: ${RUN_DIR}/config.yaml not found — is that a Regrasp run dir?" >&2
    exit 1
fi
# ── derive the split's pin table if the direction table is there ─────────────
# `assign_direction_demos.py` is pure combinatorics over the direction table —
# seconds, no simulator — so a missing pin table is not a reason to fail a job
# that has already waited in the queue. The DIRECTION table is the real
# prerequisite, because building that needs the planner and a GPU pass.
DIR_TABLE="${DIR_TABLE:-output/direction_table_${SPLIT}.json}"
PINS="output/regrasp_pins_${SPLIT}.json"
if [ ! -f "${PINS}" ]; then
    if [ ! -f "${DIR_TABLE}" ]; then
        echo "ERROR: neither ${PINS} nor ${DIR_TABLE} exists." >&2
        echo "  Build the ${SPLIT}-split direction table first (a GPU sim pass):" >&2
        echo "    sbatch --export=ALL,SPLIT=${SPLIT},OUT=${DIR_TABLE} \\" >&2
        echo "        examples/slurm/build_direction_table.sbatch" >&2
        exit 1
    fi
    echo "[prep] ${PINS} missing — assigning it from ${DIR_TABLE}"
    python examples/assign_direction_demos.py --table "${DIR_TABLE}" \
        --out "output/regrasp_pins_${SPLIT}" --mode per-bin \
        --drop-bins='-z_beneath,-x_over_fingers'
fi

ARGS=(--run-dir "${RUN_DIR}" --split "${SPLIT}" --iters "${ITERS}" --ckpt "${CKPT}")
[ "${CHAINED}" = "1" ] && ARGS+=(--chained --rewind-frac "${REWIND}" --max-attempts "${MAXATT}")

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "run-dir=${RUN_DIR}  split=${SPLIT}  iters=${ITERS}  ckpt=${CKPT}  chained=${CHAINED}"

srun python examples/eval_regrasp_testset.py "${ARGS[@]}" ${EXTRA}
