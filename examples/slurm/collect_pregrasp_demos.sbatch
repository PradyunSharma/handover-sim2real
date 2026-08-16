#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — the PRE-GRASP demonstration set D_0
# (examples/collect_pregrasp_demos.py), for a run with DAGGER.target: pregrasp.
#
# This REPLACES step 2 (collect_bc_demos.sbatch) for such a run. It still depends
# on the pin table from step 1 (build_pin_table.sbatch), which already exists as
# output/grasp_pin_table_{train,val}_omg.json.
#
#   Train split (D_0):
#     sbatch --time=04:00:00 --export=ALL,CFG=examples/configs/dagger_phase4_all_beta075_pregrasp.yaml,OUT=output/bc_dataset/train_pregrasp_settled.h5 \
#         examples/slurm/collect_pregrasp_demos.sbatch
#   Val split (selects best.pt within each iteration):
#     sbatch --time=01:00:00 --export=ALL,CFG=examples/configs/dagger_phase4_all_beta075_pregrasp.yaml,SPLIT=val,PIN=output/grasp_pin_table_val_omg.json,EXCLUDE=none,OUT=output/bc_dataset/val_pregrasp_settled.h5 \
#         examples/slurm/collect_pregrasp_demos.sbatch
#   6-scene shakedown first (~5 min) — the script is otherwise unexercised:
#     sbatch --time=00:30:00 --export=ALL,CFG=...,LIMIT=6,OUT=/tmp/pregrasp_smoke.h5 \
#         examples/slurm/collect_pregrasp_demos.sbatch
#   Watch:  tail -f slurm_logs/pregrasp_<jobid>.out
#
# WHY IT EXISTS. `make_pregrasp_demos.py` can derive a pre-grasp set by truncating
# the grasp demos for free, but it cannot produce a SETTLED terminal state — the
# demonstrations never hold the pre-grasp, so their last row sits ~11.6 mm short
# and is labelled COMMIT there. With DAGGER.commit_settle_steps > 0 the shards
# label that same state APPROACH, and the aggregate then holds two contradictory
# commit labels for one state. This produces D_0 through the SAME collector, at
# beta = 1, so every commit label in the run is written by one rule.
#
# IT IS ALSO THE SMOKE TEST. outcome_check executes the blind reach and the close
# on every episode and scores it, so the summary at the end is the EXPERT's own
# success rate with no learner involved — the ceiling for the run. Read it before
# submitting the training job.
#
# SERIAL — one env, one scene at a time. Measured from run 16: ~15 s of
# single-worker CPU per episode on the three-camera rig, so ~2 h for the 472
# usable train scenes, and less here since wrist+right is one render cheaper.
# Ask for 4 h; the val split is a few dozen scenes and wants 1 h. The h5 is
# written incrementally, so a timeout keeps everything already collected — but
# there is no resume, so a truncated file must be re-collected from scratch.
#
# NO --egl, deliberately: the point cloud is renderer-dependent and this MUST
# match SIM.egl in the run config (false), or the policy trains on a different
# input distribution than DAgger later collects.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-pregrasp
# A GPU IS MANDATORY — not for a network (the expert is OMG, the cloud comes from
# PyBullet's CPU renderer) but for OMG itself: importing omg.config calls .cuda()
# at module scope and the SDF collision cost is CUDA throughout. A CPU partition
# dies at import with "Found no NVIDIA driver".
#SBATCH --partition=gpu-v100          # see train_rl.sbatch for why not gpu-a100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
# Resource block copied VERBATIM from collect_bc_demos.sbatch, which passes
# `sbatch --test-only`. Deviating from it produced AssocMaxGRESPerJob even though
# the request sits well inside the association limit, so treat it as load-bearing.
# --gpus-per-task is MANDATORY (DelftBlue rejects --gres/--gpus outright).
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # serial script — a few cores for PyBullet/python
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G              # 16G total
#SBATCH --output=slurm_logs/pregrasp_%j.out
#SBATCH --error=slurm_logs/pregrasp_%j.err

set -euo pipefail
mkdir -p slurm_logs

# ── environment (identical to collect_bc_demos.sbatch — keep in sync) ───────
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
export PYTHONUNBUFFERED=1

# ── parameters (override via sbatch --export=ALL,...) ───────────────────────
CFG="${CFG:-examples/configs/dagger_phase4_all_beta075_pregrasp.yaml}"
OUT="${OUT:-output/bc_dataset/train_pregrasp_settled.h5}"
SPLIT="${SPLIT:-}"                    # empty => whatever SIM.split says (train)
PIN="${PIN:-}"                        # REQUIRED with SPLIT=val (split-relative!)
EXCLUDE="${EXCLUDE:-}"                # 'none' disables; the list is train-relative
LIMIT="${LIMIT:-}"                    # first N scenes only, for a shakedown
SEED="${SEED:-0}"

# The dataset is an INPUT to training and is read relative to the repo, so unlike
# a run's output it does NOT go to /scratch. It is ~100-150 MB, which /home's
# 30 GB quota absorbs; the quota problem is checkpoints, not datasets.
mkdir -p "$(dirname "${OUT}")"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "cfg=${CFG}  out=${OUT}  split=${SPLIT:-<from cfg>}  pin=${PIN:-<from cfg>}"
echo "exclude=${EXCLUDE:-<from cfg>}  limit=${LIMIT:-all}  seed=${SEED}"

ARGS=(--cfg "${CFG}" --output "${OUT}" --seed "${SEED}")
[ -n "${SPLIT}" ]   && ARGS+=(--split "${SPLIT}")
[ -n "${PIN}" ]     && ARGS+=(--grasp-pin-table "${PIN}")
[ -n "${EXCLUDE}" ] && ARGS+=(--exclude-scenes "${EXCLUDE}")
[ -n "${LIMIT}" ]   && ARGS+=(--limit "${LIMIT}")

srun python examples/collect_pregrasp_demos.py "${ARGS[@]}"
