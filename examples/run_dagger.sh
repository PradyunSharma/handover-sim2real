#!/usr/bin/env bash
#
# Iterative DAgger for Phase-1 BC.
#
# Each round: roll out the current policy + label visited states with OMG
# (collect_dagger_dataset.py), then retrain a fresh run on the aggregate
# [train.h5 + dagger_iter1..i.h5] (train_bc.py --dagger-h5). The next round
# rolls out the run just trained.
#
# Override any setting via env vars, e.g.:
#   ITERS=2 NUM_EPISODES=30 bash examples/run_dagger.sh
#
# Each script is independently runnable; this is just the loop.
set -euo pipefail

SIM_CFG="${SIM_CFG:-examples/pretrain.yaml}"             # simulator config
TRAIN_CFG="${TRAIN_CFG:-examples/configs/bc_phase1.yaml}" # BC training config
BASE_TRAIN_H5="${BASE_TRAIN_H5:-output/bc_dataset/train.h5}"
VAL_H5="${VAL_H5:-output/bc_dataset/val.h5}"
PREV_RUN="${PREV_RUN:-output/bc_runs/run2}"   # policy rolled out in round 1
ITERS="${ITERS:-3}"
NUM_EPISODES="${NUM_EPISODES:-}"             # empty = all scenes in the split
MAX_STEPS="${MAX_STEPS:-}"                   # empty = collector default (RL_MAX_STEP)
NUM_EPOCHS="${NUM_EPOCHS:-50}"
# Per-round beta schedule: round 1 -> BETA_START, round ITERS -> BETA_END,
# linearly interpolated. High beta early keeps the weak policy's rollouts in the
# useful funnel (more expert mixing); beta 0 at the end is pure DAgger on the
# policy's own off-distribution states. Pass BETA to force one fixed value.
BETA_START="${BETA_START:-0.5}"               # beta at round 1
BETA_END="${BETA_END:-0.0}"                   # beta at round ITERS
BETA="${BETA:-}"                              # set to force a fixed beta (no schedule)
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
DAGGER_DIR="${DAGGER_DIR:-output/bc_dataset}"

export GADDPG_DIR="${GADDPG_DIR:-GA-DDPG}"
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-OMG-Planner}"

# beta for DAgger round $1 (1..ITERS). Fixed when BETA is passed, else the
# linear BETA_START->BETA_END schedule (ITERS=1 collapses to BETA_END).
round_beta() {
  if [ -n "$BETA" ]; then echo "$BETA"; return; fi
  LC_ALL=C awk -v s="$BETA_START" -v e="$BETA_END" -v i="$1" -v n="$ITERS" \
    'BEGIN{ if (n <= 1) printf "%.4f", e; else printf "%.4f", s + (e - s) * (i - 1) / (n - 1) }'
}
if [ -n "$BETA" ]; then echo "DAgger beta: $BETA (fixed)"
else echo "DAgger beta schedule: ${BETA_START} -> ${BETA_END} linear over ${ITERS} round(s)"; fi

DAGGER_FILES=()
for i in $(seq 1 "$ITERS"); do
  echo "================ DAgger round ${i}/${ITERS} ================"
  OUT_H5="${DAGGER_DIR}/dagger_iter_2_${i}.h5"
  RUN_NAME="dagger_iter_2_${i}"
  BETA_I="$(round_beta "$i")"

  echo "[round ${i}] collect with policy '${PREV_RUN}' -> ${OUT_H5}  (beta=${BETA_I})"
  EP_FLAG=(); [ -n "$NUM_EPISODES" ] && EP_FLAG=(--num-episodes "$NUM_EPISODES")
  MS_FLAG=(); [ -n "$MAX_STEPS" ]    && MS_FLAG=(--max-steps "$MAX_STEPS")
  python examples/collect_dagger_dataset.py \
    --run-dir  "$PREV_RUN" \
    --cfg-file "$SIM_CFG" \
    --output   "$OUT_H5" \
    --split    train \
    "${EP_FLAG[@]}" "${MS_FLAG[@]}" \
    --beta "$BETA_I" --seed "$SEED" --device "$DEVICE"

  DAGGER_FILES+=("$OUT_H5")

  echo "[round ${i}] train '${RUN_NAME}' on ${BASE_TRAIN_H5} + ${DAGGER_FILES[*]}"
  python examples/train_bc.py \
    --cfg-file "$TRAIN_CFG" \
    --run-name "$RUN_NAME" \
    --train-h5 "$BASE_TRAIN_H5" \
    --val-h5   "$VAL_H5" \
    --dagger-h5 "${DAGGER_FILES[@]}" \
    --num-epochs "$NUM_EPOCHS" \
    --device "$DEVICE"

  PREV_RUN="output/bc_runs/${RUN_NAME}"
done

echo "DAgger done. Final policy: ${PREV_RUN}"
