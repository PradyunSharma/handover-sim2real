#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SCORE A FINISHED REGRASP RUN ON A HELD-OUT SPLIT, every iteration.
#
#     RUN=regrasp_run9 sbatch examples/slurm/eval_regrasp_testset.sbatch
#
# Knobs, all optional, all read from the environment so one file serves every
# run (`--export` them or put them inline as above):
#
#     RUN       run NAME under $OUT_ROOT           (default regrasp_run9)
#     RUN_DIR   full path, overrides RUN           (default $OUT_ROOT/$RUN)
#     SPLIT     test | val | train                 (default test)
#     ITERS     all | 0,5,10,...                    (default all)
#     CKPT      last | best                        (default last)
#     CHAINED   1 to add true chained retry        (default off)
#
# ── WHY 3 h AND NOT 12 ────────────────────────────────────────────────────────
# MEASURED, on this split, on this cluster: iteration 23 of run 9 scored 340
# episodes in 1497 s -> 4.4 s per episode, 25 min per iteration.
#
#     129 test scenes x 2.64 slots = 340 episodes per iteration
#     340 x 4.4 s                  = ~25 min per iteration
#     x 26 iterations              = ~10.8 h total
#
# An earlier estimate here said 7.4 h, derived from the in-loop `eval_s` divided
# by what was read as an 8-worker pool. That 8 was `TRAIN.num_workers`, the
# DATALOADER worker count; the eval pool is `--num-workers` on train_regrasp.py
# and was larger. The measured number above supersedes it — do not re-derive.
#
# `--chained` adds up to `--max-attempts` rollouts per scene plus rewind replay,
# roughly 2.5x the above.
#
# THE SWEEP IS RESUMABLE PER ITERATION. `write_log` rewrites the CSV after every
# iteration and `read_done` skips rows already in it, so a wall-clock kill costs
# at most the ~25 min iteration in flight — nothing else. Short chained jobs are
# therefore strictly better than one long one: DelftBlue backfills short jobs
# ahead of long ones, and the downside of being killed is 25 minutes.
#
#     J=$(sbatch --parsable examples/slurm/eval_regrasp_testset.sbatch)
#     for i in 1 2 3; do
#       J=$(sbatch --parsable --dependency=afterany:$J \
#              examples/slurm/eval_regrasp_testset.sbatch)
#     done
#
# FOUR 3 h passes = 12 h of capacity for 10.8 h of work, ~7 iterations a pass. A
# pass that finds every iteration already scored prints "nothing to score" and
# exits in seconds, so a spare tail job costs a queue slot and nothing else.
#
# ── ONE PREREQUISITE ────────────────────────────────────────────────────────
# The SPLIT's pin table must exist: `output/regrasp_pins_<split>.json` and
# `output/regrasp_pins_<split>_excluded.json`. Nothing here builds them, and the
# script exits with the two commands that would. Check before submitting:
#
#     ls -la output/regrasp_pins_test.json output/regrasp_pins_test_excluded.json
#
# Note `regrasp_pins_test.json` carries no `d_rule` in its `_meta`, so it reads
# as `approach_axis` — which is what runs 1-9 used. Scoring a `grasp_offset` run
# (10+) on test needs a test table rebuilt under that rule, and
# `resolve_d_rule` will refuse the run rather than mis-caption the bins.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h2r-testeval
#SBATCH --partition=gpu-v100
#SBATCH --account=education-me-msc-ro
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/testeval_%j.out
#SBATCH --error=slurm_logs/testeval_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2023r1 || true
module load cuda/11.7 || true
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-pch2r_dev}"
export PYTHONUNBUFFERED=1
export GADDPG_DIR="${GADDPG_DIR:-$PWD/GA-DDPG}"
# `env_setup` imports OMG at module scope even though evaluation never plans, so
# the variable has to be set or the import fails before the first rollout.
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-$PWD/OMG-Planner}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/h2r-runs}"
export REGRASP_DATA="${REGRASP_DATA:-$SCRATCH_ROOT/output}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH_ROOT/output/dagger_runs}"
mkdir -p slurm_logs

RUN="${RUN:-regrasp_run9}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/$RUN}"
SPLIT="${SPLIT:-test}"
ITERS="${ITERS:-all}"
CKPT="${CKPT:-last}"

# THE RUN DIR IS THE ONE THING THIS CANNOT GUESS. Older campaigns wrote into the
# repo's own `output/`, later ones into $SCRATCH_ROOT. Try both rather than
# failing on a path the user never chose.
if [ ! -f "$RUN_DIR/config.yaml" ] && [ -f "output/dagger_runs/$RUN/config.yaml" ]; then
    echo "[note] $RUN_DIR has no config.yaml; using ./output/dagger_runs/$RUN"
    RUN_DIR="output/dagger_runs/$RUN"
fi
if [ ! -f "$RUN_DIR/config.yaml" ]; then
    echo "FAIL: no config.yaml under $RUN_DIR — set RUN_DIR explicitly." >&2
    exit 2
fi

EXTRA=""
[ "${CHAINED:-0}" = "1" ] && EXTRA="--chained"

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "run_dir=$RUN_DIR  split=$SPLIT  iters=$ITERS  ckpt=$CKPT  chained=${CHAINED:-0}"
echo "already scored: $(tail -n +2 "$RUN_DIR/${SPLIT}_log.csv" 2>/dev/null | wc -l) iteration(s)"
df -h "$HOME" | tail -1

python examples/eval_regrasp_testset.py \
    --run-dir "$RUN_DIR" \
    --split "$SPLIT" --ckpt "$CKPT" --iters "$ITERS" $EXTRA
rc=$?

# Re-render the figure from whatever the CSV holds, even if the sweep was cut
# short — a partial curve is worth looking at and costs seconds.
echo
echo "=== plotting from $RUN_DIR/${SPLIT}_log.csv ==="
python examples/eval_regrasp_testset.py \
    --run-dir "$RUN_DIR" --split "$SPLIT" --plot-only || true

echo
echo "scored: $(tail -n +2 "$RUN_DIR/${SPLIT}_log.csv" 2>/dev/null | wc -l) iteration(s)"
exit $rc
