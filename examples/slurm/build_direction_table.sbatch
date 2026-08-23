#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DelftBlue SLURM job — build the Regrasp DIRECTION table: per scene, the anchor
# frame plus the best goal-set grasp realising each approach-direction bin
# (examples/build_direction_table.py). STEP 1 of four; assign_direction_demos.py
# then picks two bins per scene OFFLINE, with no simulator.
#
#   Train split:
#     sbatch --export=ALL,SPLIT=train,OUT=output/direction_table_train.json \
#         examples/slurm/build_direction_table.sbatch
#   Val split (for the val demos used in within-run best.pt selection):
#     sbatch --export=ALL,SPLIT=val,OUT=output/direction_table_val.json \
#         examples/slurm/build_direction_table.sbatch
#   Smoke first (~20 scenes, ~2 min — proves the env before burning the full pass):
#     sbatch --export=ALL,SPLIT=train,OUT=output/direction_smoke.json,EXTRA="--num-scenes 20" \
#         examples/slurm/build_direction_table.sbatch
#
# WHAT THIS RECORDS, and why it is not the old K-candidate table. Per scene: the
# giver's wrist, the OBSERVED object centroid, the anchor frame, and — for each
# of the k approach-direction bins — the goal-set grasp closest to that bin's
# axis. Every goal-set member is binned (not an FPS subsample of 8), so the
# per-bin counts are the real feasibility census.
#
# THE TABLE NOW DEPENDS ON THE RENDERER. This is the opposite of what the old
# K-candidate builder needed, and the reason is the centroid: it comes from the
# observed point CLOUD, so the camera set and the renderer change it. SIM_CFG
# MUST match what the demonstrations will be collected with, or the anchor frame
# a bin was assigned in differs from the frame the policy sees at run time.
# Still no --egl (leaks ~85 MB/scene -> OOM over a full split), which matches
# `egl: false` in the run config.
#
# READ THE PER-BIN SUMMARY IT PRINTS. It is the GO/NO-GO gate: a bin no scene can
# reach is not a retry hypothesis, and finding that out here costs minutes rather
# than a 20-hour collection. The offline preview
# (`examples/analyze_direction_feasibility.py`, no simulator at all) already says
# to expect -z at 0% and -x at ~2% on s0/train; this pass confirms it against the
# FULL goal set rather than the FPS-selected 8.
#   Watch:  tail -f slurm_logs/dirtable_<jobid>.out
#
# ONE TABLE PER SPLIT. Scene indices are split-relative, so a train table applied
# to val anchors the wrong scenes.
#
# SERIAL — one env, one OMG plan per scene (~720 train scenes at ~1-3 s each →
# ~30-60 min). The JSON is written once at the end; use EXTRA="--num-scenes N"
# to smoke it first.
#
# REBUILD whenever SIM.valid_grasp_dict_path, hand_collision_filter, the camera
# set or the renderer changes — the first two change which grasps exist, the last
# two change the centroid.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-dirtab
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
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G              # 16G total
#SBATCH --output=slurm_logs/dirtable_%j.out
#SBATCH --error=slurm_logs/dirtable_%j.err

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
#
# NOTE the --cfg-file here is the SIMULATOR yaml, not a Regrasp run config. It
# MUST match what the demonstrations will be collected with: the object centroid
# recorded per scene comes from the observed point CLOUD, and the cloud is
# renderer- and camera-dependent. A mismatch puts a silent offset between the
# anchor frame a bin was assigned in and the frame the policy sees.
SIM_CFG="${SIM_CFG:-examples/pretrain_multicam_wr.yaml}"
SPLIT="${SPLIT:-train}"
OUT="${OUT:-output/direction_table_${SPLIT}.json}"
K="${K:-6}"                           # 6 = the octahedral bins
MAXANG="${MAXANG:-45}"                # Voronoi half-angle at k=6, so each grasp
                                      # falls in exactly one bin
VGD="${VGD:-examples/valid_grasp_dict_005.pkl}"
EXTRA="${EXTRA:-}"                    # e.g. EXTRA="--num-scenes 20"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "sim_cfg=${SIM_CFG}  split=${SPLIT}  out=${OUT}  k=${K}  max_angle=${MAXANG}  extra=${EXTRA:-none}"
mkdir -p "$(dirname "${OUT}")"

srun python examples/build_direction_table.py \
    --cfg-file         "${SIM_CFG}" \
    --split            "${SPLIT}" \
    --out              "${OUT}" \
    --k                "${K}" \
    --max-angle        "${MAXANG}" \
    --valid-grasp-dict "${VGD}" \
    ${EXTRA}
