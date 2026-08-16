#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — collect the Phase-5 expert demonstrations
# (examples/collect_bc_dataset_multi.py): ONE PER (scene, candidate grasp), so K
# times as many episodes as Phase 4 collected. STEP 2 of four, and it DEPENDS on
# the candidate table from STEP 1 (examples/slurm/build_pin_table_multi.sbatch).
#
#   Train split:
#     sbatch --export=ALL,SPLIT=train,SIM_CFG=examples/pretrain_multicam_wr.yaml,OUT=output/bc_dataset/train_p5_k8.h5,PIN=output/grasp_cand_table_train_p5.json \
#         examples/slurm/collect_bc_demos_p5.sbatch
#   Val split (within-run best.pt selection):
#     sbatch --export=ALL,SPLIT=val,SIM_CFG=examples/pretrain_multicam_wr.yaml,OUT=output/bc_dataset/val_p5_k8.h5,PIN=output/grasp_cand_table_val_p5.json \
#         examples/slurm/collect_bc_demos_p5.sbatch
#   Chained behind the table build (preferred — no manual waiting):
#     JID=$(sbatch --parsable --export=ALL,SPLIT=train,OUT=output/grasp_cand_table_train_p5.json \
#             examples/slurm/build_pin_table_multi.sbatch)
#     sbatch --dependency=afterok:$JID --export=ALL,SPLIT=train,SIM_CFG=examples/pretrain_multicam_wr.yaml,OUT=output/bc_dataset/train_p5_k8.h5,PIN=output/grasp_cand_table_train_p5.json \
#         examples/slurm/collect_bc_demos_p5.sbatch
#   Watch:  tail -f slurm_logs/bcdemos5_<jobid>.out
#
# BUDGET K TIMES PHASE 4. At K=8 over ~623 planning scenes that is ~5000 episodes
# against Phase 4's 623. Still serial and still resumable only by restarting, so
# request the full 24 h and expect to resubmit once.
#
# SIM_CFG MUST BE pretrain_multicam_wr.yaml for run 1 — wrist + right, matching
# SIM.cfg_file in dagger_phase5_run1.yaml. A camera mismatch between the base set
# and the DAgger shards is the one inconsistency the aggregate cannot average
# away, and it fails silently: training looks fine and the policy is wrong only
# at deployment.
#
# WHY RE-COLLECT. The existing output/bc_dataset/train.h5 was collected with
# NEITHER grasp filter wired in, while Phase 3/4 use valid_grasp_dict_005.pkl.
# Measured over 72 scenes: 21% of scenes had a DIFFERENT goal grasp between the
# demos and DAgger (median 10.7 cm apart, max 20.5 cm — all far outside the 2 cm
# close threshold). Aggregating those two label sets is exactly the inconsistency
# DAgger cannot average away.
#
# SERIAL — one env, one scene at a time (~720 train scenes) → plan for many hours.
# The h5 is written incrementally, so a crash/timeout keeps every episode already
# collected. No --egl: it leaks ~85 MB/scene (OOM over a full split), AND the point
# cloud is renderer-dependent — this MUST match SIM.egl in dagger_phase5_run1.yaml
# (false), or the DAgger policy sees a different input distribution than it trained on.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-bcdem5
# A GPU IS MANDATORY — not for a neural network (the expert is OMG, the cloud
# comes from PyBullet's CPU renderer), but for OMG ITSELF: importing omg.config
# calls .cuda() at module scope (config.py:222) and the SDF collision cost is
# CUDA throughout. A CPU partition dies at import with "Found no NVIDIA driver".
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
#SBATCH --cpus-per-task=4             # serial script — a few cores for PyBullet/python
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G              # 16G total
#SBATCH --output=slurm_logs/bcdemos5_%j.out
#SBATCH --error=slurm_logs/bcdemos5_%j.err

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
# a file, so a 48 h collection shows no progress until it exits — and a crash
# leaves a truncated log that stops at an arbitrary point.
export PYTHONUNBUFFERED=1

# ── parameters (override via sbatch --export=ALL,...) ───────────────────────
SIM_CFG="${SIM_CFG:-examples/pretrain_multicam_wr.yaml}"  # MUST match SIM.cfg_file in the phase-5 config
SPLIT="${SPLIT:-train}"
OUT="${OUT:-output/bc_dataset/${SPLIT}_p5_k8.h5}"
VGD="${VGD:-examples/valid_grasp_dict_005.pkl}"           # Phase-3/4/5 grasp set
PIN="${PIN:-output/grasp_cand_table_${SPLIT}_p5.json}"    # K candidates per scene
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"                    # e.g. EXTRA="--num-scenes 5" for a smoke run

if [ ! -f "${PIN}" ]; then
    echo "ERROR: candidate table ${PIN} not found. Run build_pin_table_multi.sbatch first" >&2
    exit 1
fi

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "sim-cfg=${SIM_CFG}  split=${SPLIT}  out=${OUT}"
echo "vgd=${VGD}  pin=${PIN}  seed=${SEED}  extra=${EXTRA:-none}"
mkdir -p "$(dirname "${OUT}")"

srun python examples/collect_bc_dataset_multi.py \
    --cfg-file          "${SIM_CFG}" \
    --output            "${OUT}" \
    --split             "${SPLIT}" \
    --seed              "${SEED}" \
    --valid-grasp-dict  "${VGD}" \
    --grasp-pin-table   "${PIN}" \
    ${EXTRA}
