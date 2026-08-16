#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — build the Phase-5 grasp CANDIDATE table, K maximally
# separated grasps per scene (examples/build_grasp_pin_table_multi.py). STEP 1
# of four; select_pinned_grasps.py later prunes K down to the final N=4.
#
#   Train split:
#     sbatch --export=ALL,SPLIT=train,OUT=output/grasp_cand_table_train_p5.json \
#         examples/slurm/build_pin_table_multi.sbatch
#   Val split (for the val demos used in within-run best.pt selection):
#     sbatch --export=ALL,SPLIT=val,OUT=output/grasp_cand_table_val_p5.json \
#         examples/slurm/build_pin_table_multi.sbatch
#   Smoke first (~20 scenes, ~2 min — proves the env before burning 24 h):
#     sbatch --export=ALL,SPLIT=train,OUT=output/pin_smoke_p5.json,EXTRA="--num-scenes 20" \
#         examples/slurm/build_pin_table_multi.sbatch
#
# K=8 AND NOT 4 ON PURPOSE. About a quarter of pinned demonstrations fail (the
# plan clips the object on the lateral approach), so demanding four of four would
# keep ~0.76^4 = 33% of scenes. Over-provision here, collect all K, and let stage
# 3 choose the four that actually worked.
#
# READ THE SUMMARY IT PRINTS. "fewer than 4 distinct" is the count of scenes that
# cannot survive stage 3 no matter how collection goes, and it is the first
# honest estimate of the Phase-5 scene budget.
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
#SBATCH --job-name=h2r-pintab5
# A GPU IS MANDATORY — not for a neural network, but for OMG itself. Importing
# `omg.config` runs get_global_param() at module scope, which calls .cuda()
# (config.py:222); and the planner's SDF collision cost (sdf_tools.py, cost.py)
# is CUDA throughout, ~15 unconditional .cuda() calls on the hot path. A CPU
# partition dies at import with "Found no NVIDIA driver". Do not "fix" this by
# moving to `compute` — that was tried, and it is the planner, not the policy.
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
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G              # 16G total
#SBATCH --output=slurm_logs/pintable5_%j.out
#SBATCH --error=slurm_logs/pintable5_%j.err

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
# Unbuffered: without this, python's stdout is block-buffered when redirected to
# a file, so a job that dies mid-run shows a truncated log that stops at an
# arbitrary point and looks like a hang at the wrong line.
export PYTHONUNBUFFERED=1

# ── parameters (override via sbatch --export=ALL,...) ───────────────────────
CFG="${CFG:-examples/configs/dagger_phase5_run1.yaml}"  # its SIM block defines the grasp set
SPLIT="${SPLIT:-train}"
OUT="${OUT:-output/grasp_cand_table_${SPLIT}_p5.json}"
K="${K:-8}"                           # candidates collected; 4 survive stage 3
NFINAL="${NFINAL:-4}"                 # recorded in _meta; only affects reporting here
SEP="${SEP:-0.02}"                    # minimum control-point separation, metres.
                                      # Under the FLIP-INVARIANT metric, so a
                                      # wrist-flip twin reads as distance 0 and can
                                      # never be selected as a "distant" grasp.
EXTRA="${EXTRA:-}"                    # e.g. EXTRA="--num-scenes 20"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cfg=${CFG}  split=${SPLIT}  out=${OUT}  k=${K}  n_final=${NFINAL}  sep=${SEP}  extra=${EXTRA:-none}"
mkdir -p "$(dirname "${OUT}")"

srun python examples/build_grasp_pin_table_multi.py \
    --cfg-file  "${CFG}" \
    --split     "${SPLIT}" \
    --out       "${OUT}" \
    --k         "${K}" \
    --n-final   "${NFINAL}" \
    --sep-floor "${SEP}" \
    ${EXTRA}
