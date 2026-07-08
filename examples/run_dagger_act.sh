#!/usr/bin/env bash
#
# Iterative DAgger for Phase-2 ACT — self-contained, resumable, with a manifest.
#
#   round 0 : train a base ACT policy on expert data only (train.h5 + val.h5).
#   round i : roll out the current policy + label visited states with OMG
#             (collect_dagger_act_dataset.py), then retrain a fresh ACT run on
#             the aggregate [train.h5 + dagger_act_iter1..i.h5]. The next round
#             rolls out that run.
#
# RESUMABLE: stop with Ctrl-C and re-run the same command — a completed
#   collection (a dagger HDF5 that already holds episodes) is detected and
#   skipped, and an interrupted/finished training is continued via
#   train_act.py --resume. Set FORCE=1 to ignore existing outputs and redo all.
#
# MANIFEST: every invocation writes a manifest (all flags with default/passed
#   origin, all files, and the exact commands run) to the dagger dir, and copies
#   it into each produced run dir as dagger_manifest.txt — so each policy carries
#   a record of exactly how it was trained (alongside its own config.yaml).
#
# Override any setting via env vars, e.g.:
#   ITERS=2 NUM_EPISODES=30 bash examples/run_dagger_act.sh
set -euo pipefail

# ---- capture which settings were PASSED (set in env) vs left at DEFAULT ----
declare -A ORIGIN
for v in SIM_CFG TRAIN_CFG BASE_TRAIN_H5 VAL_H5 ITERS NUM_EPISODES MAX_STEPS \
         BASE_EPOCHS NUM_EPOCHS BETA BETA_START BETA_END SEED DEVICE DAGGER_DIR \
         BASE_RUN PREV_RUN FORCE GADDPG_DIR OMG_PLANNER_DIR; do
  if [ -n "${!v+x}" ]; then ORIGIN[$v]="passed"; else ORIGIN[$v]="default"; fi
done

SIM_CFG="${SIM_CFG:-examples/pretrain.yaml}"               # simulator config
TRAIN_CFG="${TRAIN_CFG:-examples/configs/act_phase2.yaml}"  # ACT training config
BASE_TRAIN_H5="${BASE_TRAIN_H5:-output/bc_dataset/train.h5}"
VAL_H5="${VAL_H5:-output/bc_dataset/val.h5}"
ITERS="${ITERS:-3}"
NUM_EPISODES="${NUM_EPISODES:-}"   # empty = ALL scenes in the split
MAX_STEPS="${MAX_STEPS:-}"         # empty = RL_MAX_STEP, no cap
BASE_EPOCHS="${BASE_EPOCHS:-100}"  # epochs for the round-0 base run
NUM_EPOCHS="${NUM_EPOCHS:-50}"     # epochs per DAgger round (1..ITERS)
# Per-round beta schedule: round 1 -> BETA_START, round ITERS -> BETA_END,
# linearly interpolated. Mixing in the expert early (high beta) keeps the weak
# policy's rollouts inside the useful funnel; pure DAgger (beta 0) at the end
# collects the policy's own off-distribution states. Passing BETA explicitly
# overrides the schedule with a single fixed value for all rounds (back-compat).
BETA_START="${BETA_START:-0.5}"    # beta at round 1
BETA_END="${BETA_END:-0.0}"        # beta at round ITERS
BETA="${BETA:-}"                   # set to a value to force fixed beta (no schedule)
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
DAGGER_DIR="${DAGGER_DIR:-output/bc_dataset}"
BASE_RUN="${BASE_RUN:-act_base}"   # name of the round-0 base run
PREV_RUN="${PREV_RUN:-}"           # set to skip round 0 and roll out this run
FORCE="${FORCE:-0}"                # 1 = ignore existing outputs, redo everything
export GADDPG_DIR="${GADDPG_DIR:-GA-DDPG}"
export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-OMG-Planner}"

mkdir -p "$DAGGER_DIR"
METAFILE="${DAGGER_DIR}/dagger_act_run_$(date +%Y%m%d_%H%M%S).meta.txt"
: > "$METAFILE"
log() { echo "$*" | tee -a "$METAFILE"; }
row() { printf '   %-16s : %s   [%s]\n' "$1" "$2" "${ORIGIN[$3]}" | tee -a "$METAFILE"; }

# per-round collection caps (omitted entirely when empty → all scenes / full horizon)
COLLECT_OPTS=()
if [ -n "$NUM_EPISODES" ]; then COLLECT_OPTS+=(--num-episodes "$NUM_EPISODES"); fi
if [ -n "$MAX_STEPS" ];    then COLLECT_OPTS+=(--max-steps "$MAX_STEPS"); fi

# resolve the round-1 rollout policy (own a base run unless PREV_RUN is provided)
OWN_BASE=0
if [ -z "$PREV_RUN" ]; then PREV_RUN="output/bc_runs/${BASE_RUN}"; OWN_BASE=1; fi

# beta for DAgger round $1 (1..ITERS). Fixed when BETA is passed, else the
# linear BETA_START->BETA_END schedule (ITERS=1 collapses to BETA_END).
round_beta() {
  if [ -n "$BETA" ]; then echo "$BETA"; return; fi
  LC_ALL=C awk -v s="$BETA_START" -v e="$BETA_END" -v i="$1" -v n="$ITERS" \
    'BEGIN{ if (n <= 1) printf "%.4f", e; else printf "%.4f", s + (e - s) * (i - 1) / (n - 1) }'
}
if [ -n "$BETA" ]; then BETA_DESC="$BETA (fixed)"
else BETA_DESC="${BETA_START} → ${BETA_END} linear over ${ITERS} round(s)"; fi

# ---- manifest header: every flag (value + default/passed), and all files ----
log "============================================================"
log " DAgger (ACT) run manifest"
log "============================================================"
log " date        : $(date '+%Y-%m-%d %H:%M:%S')"
log " git commit  : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
log " interpreter : $(command -v python || echo '<python not on PATH>')"
log " manifest    : ${METAFILE}"
log " resume mode : $([ "$FORCE" = 1 ] && echo 'FORCE — redo all' || echo 'on — skip completed steps')"
log "------------------------------------------------------------"
log " parameters (value [origin]):"
row "GADDPG_DIR"      "$GADDPG_DIR"                          GADDPG_DIR
row "OMG_PLANNER_DIR" "$OMG_PLANNER_DIR"                     OMG_PLANNER_DIR
row "DEVICE"          "$DEVICE"                              DEVICE
row "SEED"            "$SEED"                                SEED
row "SIM_CFG"         "$SIM_CFG"                             SIM_CFG
row "TRAIN_CFG"       "$TRAIN_CFG"                           TRAIN_CFG
row "BASE_TRAIN_H5"   "$BASE_TRAIN_H5"                       BASE_TRAIN_H5
row "VAL_H5"          "$VAL_H5"                              VAL_H5
row "ITERS"           "$ITERS"                               ITERS
row "BASE_RUN"        "$BASE_RUN"                            BASE_RUN
row "PREV_RUN"        "$PREV_RUN"                            PREV_RUN
row "BASE_EPOCHS"     "$BASE_EPOCHS"                         BASE_EPOCHS
row "NUM_EPOCHS"      "$NUM_EPOCHS"                          NUM_EPOCHS
row "NUM_EPISODES"    "${NUM_EPISODES:-ALL scenes}"          NUM_EPISODES
row "MAX_STEPS"       "${MAX_STEPS:-RL_MAX_STEP (no cap)}"   MAX_STEPS
row "BETA schedule"   "$BETA_DESC"                           BETA
row "DAGGER_DIR"      "$DAGGER_DIR"                          DAGGER_DIR
row "FORCE"           "$FORCE"                               FORCE
log "------------------------------------------------------------"
log " files produced this run:"
if [ "$OWN_BASE" -eq 1 ]; then log "   run  : output/bc_runs/${BASE_RUN}   (base, expert data only)"; fi
AGG="${BASE_TRAIN_H5}"
for i in $(seq 1 "$ITERS"); do
  AGG="${AGG} + ${DAGGER_DIR}/dagger_act_iter${i}.h5"
  log "   data : ${DAGGER_DIR}/dagger_act_iter${i}.h5   (round ${i}, beta=$(round_beta "$i"))"
  log "   run  : output/bc_runs/dagger_act_iter${i}   (train on: ${AGG})"
done
log "============================================================"

# ---- helpers ----

# is a collected dagger HDF5 complete (exists and holds >=1 episode)?
h5_done() {
  [ -f "$1" ] || return 1
  python - "$1" <<'PYEOF' 2>/dev/null
import sys, h5py
try:
    with h5py.File(sys.argv[1], "r") as f:
        sys.exit(0 if int(f.attrs.get("num_episodes", 0)) > 0 else 1)
except Exception:
    sys.exit(1)
PYEOF
}

# train_act for a run, resuming from its last.pt when present (unless FORCE).
# $1 = run name; remaining args are passed through (e.g. --dagger-h5 a.h5 b.h5).
train_run() {
  local run_name="$1"; shift
  local ckpt="output/bc_runs/${run_name}/checkpoints/last.pt"
  local epochs="$NUM_EPOCHS"
  if [ "$run_name" = "$BASE_RUN" ]; then epochs="$BASE_EPOCHS"; fi
  local resume=()
  if [ "$FORCE" != "1" ] && [ -f "$ckpt" ]; then
    resume=(--resume "$ckpt"); log "   [resume] ${run_name} from ${ckpt}"
  fi
  local cmd=(python examples/train_act.py --cfg-file "$TRAIN_CFG" --run-name "$run_name"
             --train-h5 "$BASE_TRAIN_H5" --val-h5 "$VAL_H5" --num-epochs "$epochs"
             --device "$DEVICE" "$@" ${resume[@]+"${resume[@]}"})
  log "   \$ ${cmd[*]}"
  "${cmd[@]}"
  cp "$METAFILE" "output/bc_runs/${run_name}/dagger_manifest.txt" 2>/dev/null || true
}

# ---- round 0: base policy on expert data only (skipped if PREV_RUN given) ----
if [ "$OWN_BASE" -eq 1 ]; then
  log "================ round 0: base ACT policy '${BASE_RUN}' (expert data only) ================"
  train_run "$BASE_RUN"
else
  log "================ round 0 skipped — rolling out provided PREV_RUN=${PREV_RUN} ================"
fi

# ---- rounds 1..ITERS: collect with current policy, retrain on the aggregate ----
DAGGER_FILES=()
for i in $(seq 1 "$ITERS"); do
  OUT_H5="${DAGGER_DIR}/dagger_act_iter${i}.h5"
  RUN_NAME="dagger_act_iter${i}"
  log "================ DAgger (ACT) round ${i}/${ITERS} ================"

  BETA_I="$(round_beta "$i")"
  log "   beta (round ${i}/${ITERS}) = ${BETA_I}"
  if [ "$FORCE" != "1" ] && h5_done "$OUT_H5"; then
    log "   [skip] collection — ${OUT_H5} already complete"
  else
    cmd=(python examples/collect_dagger_act_dataset.py --run-dir "$PREV_RUN"
         --cfg-file "$SIM_CFG" --output "$OUT_H5" --split train
         ${COLLECT_OPTS[@]+"${COLLECT_OPTS[@]}"} --beta "$BETA_I" --seed "$SEED" --device "$DEVICE")
    log "   \$ ${cmd[*]}"
    "${cmd[@]}"
  fi
  DAGGER_FILES+=("$OUT_H5")

  train_run "$RUN_NAME" --dagger-h5 "${DAGGER_FILES[@]}"
  PREV_RUN="output/bc_runs/${RUN_NAME}"
done

log "DAgger (ACT) done. Final policy: ${PREV_RUN}"
log "Manifest: ${METAFILE}  (also copied into each run dir as dagger_manifest.txt)"
