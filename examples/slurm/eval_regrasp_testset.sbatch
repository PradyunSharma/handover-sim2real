#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SCORE A FINISHED REGRASP RUN ON A HELD-OUT SPLIT, every iteration.
#
#     RUN=regrasp_run9 sbatch examples/slurm/eval_regrasp_testset.sbatch
#
# Knobs, all optional, all read from the environment so one file serves every
# run (`--export` them or put them inline as above):
#
#     FORCE     1 to re-score iterations already in the CSV (default off)
#     OUT       CSV filename, e.g. test_chained_log.csv  (default <split>_log.csv)
#     RUN       run NAME under $OUT_ROOT           (default regrasp_run9)
#     RUN_DIR   full path, overrides RUN           (default $OUT_ROOT/$RUN)
#     SPLIT     test | val | train                 (default test)
#     ITERS     all | 0,5,10,...                    (default all)
#     CKPT      last | best                        (default last)
#     CHAINED   1 to add true chained retry        (default off)
#     BINS      attempt order, e.g. '+z,+x,-y,+y'  (default RETRY_LADDER)
#     ONLY_BINS restrict to these bins entirely    (default: all live bins)
#     ADAPTIVE  1 to re-rank the bins online       (default off)
#     RANK_MODE fixed|mean|ucb|thompson            (default ucb, with ADAPTIVE)
#     FULL_BINS 1 to command every bin on every scene  (default off)
#     STOP_SUCC 1 to stop a scene at its first success (default off)
#     AXES      command_axes.json to condition on  (default the run's own)
#     ALLOW_CFG 1 to ignore a cfg_file mismatch    (default off — see below)
#
# ── RUNS 19, 22 AND 23 ARE READY; RUN 21 IS NOT ─────────────────────────────
# `regrasp_pins_test.json` was built under `examples/pretrain_multicam_wr.yaml`
# with `d_rule: approach_axis` and `anchor_hand_ref: wrist`. Runs 19, 22 and 23
# match on all three and score against it as-is.
#
# Run 21 trained under `examples/pretrain_right.yaml` (right camera only) and
# its TRAIN table is `regrasp_pins_train_right.json`. The pin table's per-scene
# `anchor_R` and `centroid_world` come from the OBSERVED cloud, so a table built
# from a two-camera fusion names the bins in a frame a one-camera policy cannot
# reproduce. The script REFUSES that pairing rather than reporting it. Build the
# matching table first (a separate sim pass, one plan per scene):
#
#     python examples/build_direction_table.py --split test \
#         --cfg-file examples/pretrain_right.yaml \
#         --anchor-hand-ref wrist --d-rule approach_axis \
#         --out output/direction_table_test_right.json
#     python examples/assign_direction_demos.py \
#         --table output/direction_table_test_right.json \
#         --out output/regrasp_pins_test_right
#
# Note NO `--drop-bins` there: `regrasp_pins_train_right.json` has
# `dropped_bins: []` and five live bins, so run 21 commands `-x` as well and the
# test table must too. Then submit with
# `PIN_TABLE=output/regrasp_pins_test_right.json`.
#
# ── WHICH VECTOR THE POLICY IS CONDITIONED ON ───────────────────────────────
# All four runs use `SIM.command_deploy: bin_centroid`, which means the command
# is the EMPIRICAL MEAN of each bin's assigned `d_anchor` — and that mean is a
# property of the table it is computed from. The script therefore loads the
# RUN's saved `command_axes.json` rather than re-deriving it from the test
# table, which sits 2.9 deg off run 19's `+x` and 6.0 deg off its `+z`. Nothing
# to set; it is the default. `--split-axes` is the opt-out.
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
# as `approach_axis` — which is what runs 1-9 and 19-23 used. Scoring a
# `grasp_offset` run (10-18) on test needs a test table rebuilt under that rule,
# and `resolve_d_rule` will refuse the run rather than mis-caption the bins.
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

# Built as an ARRAY, not a string: a `--bins '+z,+x'` value word-splits the
# moment it goes through `$EXTRA` unquoted, and `+z` then becomes a bad run-dir.
#
# Expanded below as `"${EXTRA[@]+"${EXTRA[@]}"}"`, not the plain `"${EXTRA[@]}"`.
# Under `set -u` bash 4.3 and earlier treat an EMPTY array's `[@]` expansion as
# an unbound variable and abort — so the default invocation, with no optional
# knobs set, would be the one that dies. Bash 4.4+ fixed it; DelftBlue's login
# and compute images do not have to agree, and losing a 3 h job to that is not
# worth the two characters saved.
EXTRA=()
[ "${FORCE:-0}" = "1" ] && EXTRA+=(--force)
[ -n "${OUT:-}" ]      && EXTRA+=(--out "$OUT")
[ "${CHAINED:-0}" = "1" ]   && EXTRA+=(--chained)
[ "${ADAPTIVE:-0}" = "1" ]  && EXTRA+=(--adaptive --rank-mode "${RANK_MODE:-ucb}")
[ "${FULL_BINS:-0}" = "1" ] && EXTRA+=(--full-bin-coverage)
[ "${STOP_SUCC:-0}" = "1" ] && EXTRA+=(--stop-on-success)
[ "${ALLOW_CFG:-0}" = "1" ] && EXTRA+=(--allow-cfg-mismatch)
[ -n "${BINS:-}" ]          && EXTRA+=(--bins "$BINS")
[ -n "${ONLY_BINS:-}" ]     && EXTRA+=(--only-bins "$ONLY_BINS")
[ -n "${AXES:-}" ]          && EXTRA+=(--command-axes "$AXES")
[ -n "${PIN_TABLE:-}" ]     && EXTRA+=(--pin-table "$PIN_TABLE")
[ -n "${EXCLUDE_SCENES:-}" ] && EXTRA+=(--exclude-scenes "$EXCLUDE_SCENES")

echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "run_dir=$RUN_DIR  split=$SPLIT  iters=$ITERS  ckpt=$CKPT  chained=${CHAINED:-0}"
echo "extra: ${EXTRA[*]+"${EXTRA[*]}"}"
echo "already scored: $(tail -n +2 "$RUN_DIR/${SPLIT}_log.csv" 2>/dev/null | wc -l) iteration(s)"
df -h "$HOME" | tail -1

python examples/eval_regrasp_testset.py \
    --run-dir "$RUN_DIR" \
    --split "$SPLIT" --ckpt "$CKPT" --iters "$ITERS" "${EXTRA[@]+"${EXTRA[@]}"}"
rc=$?

# Re-render both figures from whatever the CSV holds, even if the sweep was cut
# short — a partial curve is worth looking at and costs seconds. `--plot-only`
# writes `<split>_eval.png` (curves against iteration) and `<split>_summary.png`
# (per-bin bars for one iteration); on a one-iteration sweep only the second
# says anything, because a stacked AREA needs two x points to render.
echo
echo "=== plotting from $RUN_DIR/${SPLIT}_log.csv ==="
python examples/eval_regrasp_testset.py \
    --run-dir "$RUN_DIR" --split "$SPLIT" --plot-only \
    ${OUT:+--out "$OUT"} || true

echo
echo "scored: $(tail -n +2 "$RUN_DIR/${SPLIT}_log.csv" 2>/dev/null | wc -l) iteration(s)"
exit $rc
