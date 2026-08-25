#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RUN 3, FAST VARIANT, END TO END ON THIS LAPTOP. One command, ~8.4 h.
#
#   bash examples/run_regrasp3_local.sh
#
# Six stages, fail-fast, every one idempotent: re-running after a crash skips
# what already completed and picks up where it stopped. That matters more than
# elegance here — the machine has twice lost CUDA to a suspend mid-run, and the
# recovery has to be "run the same command again", not a decision tree at 3 a.m.
#
#   T  direction tables      train + val, --members-per-bin 5      ~11 min
#   A  assignment            --per-bin 3                            ~1 min
#   V  val collection        ~200 episodes, serial                 ~12 min
#   C  train collection      4578 episodes, 4 CONCURRENT SHARDS    ~69 min
#   D  audits + merge        per shard, --write-ok                  ~5 min
#   E  training              base fit + 6 iterations               ~406 min
#
# WHY THE SHARDING. Base collection is serial at a MEASURED 3.62 s/episode, so
# 4578 episodes is ~4.6 h in one process. Four concurrent shards bring it to
# ~1.2 h. They are round-robin over SCENES (scene_idx %% 4), so each holds whole
# scenes and the four files are read directly by the trainer as a list; nothing
# is ever merged.
#
# The 3.62 s/ep is the HDF5 birth-to-last-write span over its episode count, and
# it replicates across run 2's train (1596 ep, 96 min) and val (69 ep, 4 min) to
# within 0.3%. Training is the long pole here, not collection: `iter_epochs: 15`
# over an 86k-step aggregate costs 32-47 min PER ROUND.
#
# SUSPEND KILLS THIS. Not the script's fault and not fixable from inside it: a
# suspend breaks the CUDA context and every stage dies. If you want it to
# survive the night untouched:
#
#   sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
#   # and afterwards:  sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
#
# A `systemd-inhibit` wrapper is NOT enough — it was in place for fast1 and the
# machine slept anyway, because an explicit sleep request overrides it.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

export GADDPG_DIR="$ROOT/GA-DDPG"
export OMG_PLANNER_DIR="$ROOT/OMG-Planner"
export PYTHONPATH="$ROOT:$ROOT/handover-sim:$ROOT/handover-sim/mano_pybullet"
export REGRASP_DATA="${REGRASP_DATA:-$ROOT/output}"
export PYTHONUNBUFFERED=1          # or `tail -f` shows only stderr for hours

PY="${PY:-/home/pradyun/anaconda3/envs/pch2r_dev/bin/python}"
RUN_NAME="${RUN_NAME:-regrasp3_fast1}"
CFG="examples/configs/regrasp_run3_fast.yaml"
SIMCFG="examples/pretrain_multicam_wr.yaml"
VGD="examples/valid_grasp_dict_005.pkl"
NSHARD=4
WORKERS="${WORKERS:-4}"

LOGDIR="$ROOT/output/logs/$RUN_NAME"
mkdir -p "$LOGDIR" "$REGRASP_DATA/bc_dataset"
STATUS="$LOGDIR/STATUS"
MAIN="$LOGDIR/pipeline.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN"; }
stage() { echo "$1" > "$STATUS"; say "=== STAGE $1 ==="; }
die()  { say "FAILED in stage $(cat "$STATUS" 2>/dev/null): $*"; echo "FAILED" > "$STATUS"; exit 1; }

T0=$(date +%s)
say "run 3 fast — $RUN_NAME"
say "  python  $PY"
say "  data    $REGRASP_DATA"
say "  logs    $LOGDIR"

# ---- preflight: the two things that have actually broken overnight ----------
$PY - <<'EOF' || die "CUDA is not usable. If the machine suspended: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
import torch, sys
torch.cuda.init()
print(f"CUDA OK — {torch.cuda.device_count()} device(s), "
      f"{torch.cuda.get_device_name(0)}")
EOF
FREE_GB=$(df -BG --output=avail "$REGRASP_DATA" | tail -1 | tr -dc '0-9')
say "  disk    ${FREE_GB} GB free"
[ "$FREE_GB" -ge 12 ] || die "need ~12 GB for the shards (base set is ~1.9 GB, plus 8 DAgger rounds)"

# ---- T: direction tables ----------------------------------------------------
# --members-per-bin 5 is a BUILD-time setting and run 2's tables recorded 1, so
# they cannot serve this run. Rebuilt only if the existing file is not already
# deep enough — a probe on _meta, not on existence, because the run-2 file has
# the right NAME and the wrong CONTENT.
stage T
for SPLIT in train val; do
  TBL="output/direction_table_${SPLIT}.json"
  DEEP=$($PY -c "
import json,sys
try: print(int(json.load(open('$TBL'))['_meta'].get('members_per_bin',1)))
except Exception: print(0)")
  if [ "$DEEP" -ge 3 ]; then
    say "  $TBL already has $DEEP members/bin — skipping"
  else
    say "  building $TBL (has $DEEP members/bin, need >=3)"
    $PY examples/build_direction_table.py --split "$SPLIT" \
        --cfg-file "$SIMCFG" --valid-grasp-dict "$VGD" \
        --members-per-bin 5 --out "$TBL" \
        >"$LOGDIR/table_${SPLIT}.log" 2>&1 || die "table $SPLIT (see $LOGDIR/table_${SPLIT}.log)"
    grep -E "spread within a bin|WARNING" "$LOGDIR/table_${SPLIT}.log" | tee -a "$MAIN" || true
  fi
done

# ---- A: assignment ----------------------------------------------------------
# Seconds. Always re-run: it is cheap, and a stale pin table is the failure that
# cost a day last time (run 1's pair table sitting under run 2's filename).
stage A
for SPLIT in train val; do
  $PY examples/assign_direction_demos.py \
      --table "output/direction_table_${SPLIT}.json" \
      --out "output/regrasp_pins_${SPLIT}_p3" --mode per-bin --per-bin 3 \
      --drop-bins='-z_beneath,-x_over_fingers' \
      >"$LOGDIR/assign_${SPLIT}.log" 2>&1 || die "assign $SPLIT (see $LOGDIR/assign_${SPLIT}.log)"
  grep -E "^  (kept|demos|short of|demos per bin)" "$LOGDIR/assign_${SPLIT}.log" | tee -a "$MAIN" || true
done

# ---- V: val collection (serial, small) --------------------------------------
stage V
VAL_H5="$REGRASP_DATA/bc_dataset/val_regrasp_p3.h5"
if [ -s "$VAL_H5" ]; then
  say "  $VAL_H5 exists — skipping (delete it to recollect)"
else
  # h5py opens "w" and TRUNCATES, so collection is not resumable: write to a
  # .part and rename only on success, or a crash leaves a short file that looks
  # complete to the check above.
  $PY examples/collect_regrasp_demos.py --cfg-file "$SIMCFG" --split val \
      --grasp-pin-table output/regrasp_pins_val_p3.json \
      --valid-grasp-dict "$VGD" \
      --output "${VAL_H5}.part" \
      >"$LOGDIR/collect_val.log" 2>&1 || die "val collection (see $LOGDIR/collect_val.log)"
  mv "${VAL_H5}.part" "$VAL_H5"
  say "  val collected"
fi

# ---- C: train collection, 4 concurrent shards -------------------------------
stage C
PIDS=(); NEED=()
for i in $(seq 0 $((NSHARD-1))); do
  SH="$REGRASP_DATA/bc_dataset/train_regrasp_p3.s${i}.h5"
  if [ -s "$SH" ]; then say "  shard $i exists — skipping"; continue; fi
  NEED+=("$i")
  ( $PY examples/collect_regrasp_demos.py --cfg-file "$SIMCFG" --split train \
        --grasp-pin-table output/regrasp_pins_train_p3.json \
        --valid-grasp-dict "$VGD" --shard "${i}/${NSHARD}" \
        --output "${SH}.part" \
      && mv "${SH}.part" "$SH" ) >"$LOGDIR/collect_train_s${i}.log" 2>&1 &
  PIDS+=($!)
  say "  shard $i launched (pid ${PIDS[-1]})"
  sleep 20     # stagger the sim builds; four at once has OOM'd an 8 GB card
done
FAIL=0
for p in "${PIDS[@]:-}"; do [ -n "$p" ] && { wait "$p" || FAIL=1; }; done
[ "$FAIL" -eq 0 ] || die "a train shard failed — see $LOGDIR/collect_train_s*.log"
for i in $(seq 0 $((NSHARD-1))); do
  [ -s "$REGRASP_DATA/bc_dataset/train_regrasp_p3.s${i}.h5" ] || die "shard $i missing after collection"
done
say "  all $NSHARD train shards present"

# ---- D: audits --------------------------------------------------------------
# `command vs bin axis` is the line that matters: it must read ~0.00 deg. A
# non-zero median means episodes are captioned with one direction and flown to
# another, which is the failure that silently poisoned a whole collection once.
# `audit_regrasp_demos --demos` opens ONE file, so each shard is audited on its
# own and the four ok-tables are merged. That is sound rather than a workaround:
# the shards PARTITION the scenes (round-robin, whole scenes each), the ok-table
# is keyed by scene, and every per-scene statistic the audit reports is computed
# within a scene. Nothing is pooled across shards that needs to be.
stage D
: > "$LOGDIR/audit_train.log"
for i in $(seq 0 $((NSHARD-1))); do
  $PY examples/audit_regrasp_demos.py \
      --demos "$REGRASP_DATA/bc_dataset/train_regrasp_p3.s${i}.h5" \
      --write-ok "$LOGDIR/ok_s${i}.json" \
      >>"$LOGDIR/audit_train.log" 2>&1 || die "train audit shard $i (see $LOGDIR/audit_train.log)"
done
$PY - "$LOGDIR" "$NSHARD" <<'EOF' | tee -a "$MAIN" || die "merging the ok-tables"
import json, sys
logdir, n = sys.argv[1], int(sys.argv[2])
# THE FILE IS NESTED: {"_meta": {...}, "ok": {scene: [bin, ...]}}. Merging the
# top level instead of ["ok"] silently produces {"_meta": [keys], "ok": [scene
# ids]} — still valid JSON, still loads, and blows up much later inside
# GraspPinTable.keep_only with a ValueError about dictionary update sequences.
# Read ["ok"], write ["ok"], and assert the shape on the way out.
merged, dup, metas = {}, 0, []
for i in range(n):
    doc = json.load(open(f"{logdir}/ok_s{i}.json"))
    if "ok" not in doc or not isinstance(doc["ok"], dict):
        raise SystemExit(f"ok_s{i}.json is not the expected "
                         f"{{_meta, ok:{{scene:[bins]}}}} shape")
    metas.append(doc.get("_meta", {}))
    for s, bins in doc["ok"].items():
        if s in merged:                      # impossible if the shards partition
            dup += 1                          # scenes; loud if they ever stop
            merged[s] = sorted(set(merged[s]) | set(bins))
        else:
            merged[s] = list(bins)
out = {"_meta": {"stage": "run_regrasp3_local.sh merge",
                 "sources": [m.get("source") for m in metas],
                 "n_scenes": len(merged),
                 "n_pairs": sum(len(v) for v in merged.values())},
       "ok": merged}
assert isinstance(out["ok"], dict) and all(
    isinstance(v, list) for v in out["ok"].values()), "merged shape is wrong"
json.dump(out, open("output/regrasp_demos_train_p3_ok.json", "w"))
print(f"  ok-table: {out['_meta']['n_scenes']} scenes, "
      f"{out['_meta']['n_pairs']} (scene, bin) pairs kept"
      + (f"   WARNING {dup} scenes appeared in more than one shard" if dup else ""))
EOF
# Prove the merged file is loadable by the thing that consumes it, HERE rather
# than six seconds into training.
$PY - <<'EOF' || die "the merged ok-table is not usable by GraspPinTable"
import json, sys
sys.path.insert(0, ".")
from handover_sim2real.regrasp.grasp_pin import load_grasp_pin_table
ok = json.load(open("output/regrasp_demos_train_p3_ok.json"))["ok"]
t = load_grasp_pin_table("output/regrasp_pins_train_p3.json", match_tol=0.02)
rep = t.keep_only(ok, verbose=False)
print(f"  keep_only: {rep['demos_after']}/{rep['demos_before']} slots kept over "
      f"{rep['scenes_after']} scenes ({len(rep['dropped_scenes'])} emptied)")
assert rep["demos_after"] > 0, "the filter kept nothing"
EOF
grep -E "command vs bin axis|informative steps" "$LOGDIR/audit_train.log" | tee -a "$MAIN" || true
if grep -qE "command vs bin axis.*(FAIL|[1-9][0-9]*\.[0-9]+ deg)" "$LOGDIR/audit_train.log"; then
  say "  WARNING command vs bin axis is NOT ~0 — mis-captioned episodes may have survived"
  say "  (keep_only filters (scene,bin) not slots; see regrasp_run3.yaml)"
fi

# ---- E: training ------------------------------------------------------------
# Resumes from state.json if a previous attempt got part way, so re-running the
# whole script after a crash costs nothing already paid for.
stage E
say "  training $RUN_NAME — base fit + 6 iterations at iter_epochs 15, ~6.8 h"
$PY examples/train_regrasp.py --cfg-file "$CFG" --run-name "$RUN_NAME" \
    --num-workers "$WORKERS" 2>&1 | tee -a "$LOGDIR/train.log" || die "training (see $LOGDIR/train.log)"

# ---- figures ----------------------------------------------------------------
$PY examples/plot_regrasp_run.py "output/dagger_runs/$RUN_NAME" \
    >>"$LOGDIR/train.log" 2>&1 || say "  (plotting failed; the CSV is intact)"

echo "DONE" > "$STATUS"
say "DONE in $(( ($(date +%s) - T0) / 60 )) min"
say "  figures: output/dagger_runs/$RUN_NAME/{training_curve,curves_regrasp,debug_dagger,media_curves}.png"
say "  status : $PY examples/status_regrasp_run.py output/dagger_runs/$RUN_NAME"
