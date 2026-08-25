#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# WHAT ALREADY EXISTS FOR A REGRASP RUN, AND IS IT USABLE.
#
#     bash examples/slurm/regrasp_inventory.sh
#
# Answers "what survived the last attempt" before you resubmit. It reads and
# reports; it never writes, submits or deletes anything.
#
# EXISTENCE IS NOT USABILITY, which is the whole reason this is not `ls`. Three
# separate things can be true of a file sitting at the right path:
#
#   a direction table can be run 1's (k=6 but one member per bin) or run 3's
#     (five members) — both load, both work, they differ in what assignment can
#     be derived from them
#   a pin table can be run 1's max-separated PAIR (1088 demos) rather than run
#     2's one-per-bin (1596) — same filename, different experiment
#   an HDF5 shard can be INCOMPLETE (a killed collection) or collected under run
#     1's command rule (d_world = the grasp axis, not the bin axis), and either
#     one silently trains the wrong thing
#
# So each is probed for what it actually contains. For the SUBMIT decision run
# `regrasp_pipeline.sh --dry-run`, which applies the same probes and then prints
# which stages it would skip.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER/handover-sim2real}"
REGRASP_DATA="${REGRASP_DATA:-$SCRATCH_ROOT/output}"
PY="${PY:-python}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }

bold "Regrasp inventory"
echo "  repo    : $PWD"
echo "  scratch : $REGRASP_DATA"
echo

bold "direction tables (git-tracked, small)"
for sp in train val test; do
    f="output/direction_table_$sp.json"
    if [ -f "$f" ]; then
        $PY - "$f" "$sp" <<'PROBE'
import json, sys, os
f, sp = sys.argv[1], sys.argv[2]
m = json.load(open(f))["_meta"]
mb = m.get("members_per_bin", 1)
print(f"  OK   {sp:<5} k={m.get('k')}  members/bin={mb}  "
      f"scenes={m.get('n_ok')}/{m.get('n_scenes')}  "
      f"{os.path.getsize(f)/1e6:.1f} MB  built {m.get('built')}")
PROBE
    else
        echo "  --   $sp   MISSING  (a ~20 min GPU job; the pipeline builds it)"
    fi
done
echo

bold "pin tables + demo_ok (git-tracked, small)"
for sp in train val test; do
    f="output/regrasp_pins_$sp.json"
    if [ -f "$f" ]; then
        $PY - "$f" "$sp" <<'PROBE'
import json, sys
f, sp = sys.argv[1], sys.argv[2]
m = json.load(open(f)).get("_meta", {})
mode, n = m.get("mode"), m.get("n_demos")
flag = "" if mode == "per-bin" else "   <-- NOT per-bin: run 1's PAIR table, wrong for run 2+"
print(f"  OK   {sp:<5} mode={mode}  per_bin={m.get('per_bin') or 1}  demos={n}{flag}")
PROBE
    else
        echo "  --   $sp   MISSING  (seconds; derived from the direction table)"
    fi
done
f=output/regrasp_demos_train_ok.json
if [ -f "$f" ]; then
    $PY - "$f" <<'PROBE'
import json, sys
d = json.load(open(sys.argv[1])); m = d.get("_meta", {}); ok = d.get("ok", {})
print(f"  OK   demo_ok  {m.get('n_pairs', sum(len(v) for v in ok.values()))} pairs "
      f"over {len(ok)} scenes   from {m.get('source')}")
print("       NOTE derived from ONE collection. If the shard is re-collected "
      "this is stale;\n            the pipeline removes it automatically when it "
      "retires a shard.")
PROBE
else
    echo "  --   demo_ok MISSING  (train_regrasp.sbatch derives it from the shard)"
fi
echo

bold "HDF5 shards (scratch, large)"
shopt -s nullglob
# h5py is in the `pch2r_dev` conda env, not in a DelftBlue login node's default
# python. Check ONCE and degrade to a size listing, rather than letting every
# shard print its own ModuleNotFoundError traceback — 40 identical stack traces
# bury the JSON results above them and say nothing the first one did not.
HAVE_H5=1
$PY -c "import h5py, numpy" 2>/dev/null || HAVE_H5=0
if [ "$HAVE_H5" = "0" ]; then
    echo "  (h5py not importable by '$PY' — sizes only, no completeness or rule check)"
    echo "  For the full probe:  module load miniconda3 && conda activate pch2r_dev"
    ls -la "$REGRASP_DATA"/bc_dataset/*.h5 2>/dev/null \
        | awk '{printf "  %-40s %8.2f GB  %s %s\n", $9, $5/1e9, $6, $7}' \
        || echo "  (none in $REGRASP_DATA/bc_dataset/)"
    echo
    bold "run directories (scratch)"
    for d in "$REGRASP_DATA"/dagger_runs/*/; do
        n=$(grep -c . "$d/dagger_log.csv" 2>/dev/null || echo 0)
        echo "  $(basename "$d"):  $(( n > 0 ? n - 1 : 0 )) iterations logged"
    done
    echo
    bold "queue"; squeue -u "$USER" 2>/dev/null || true
    echo
    echo "Next: conda activate pch2r_dev, re-run this, then"
    echo "      bash examples/slurm/regrasp_pipeline.sh --dry-run"
    exit 0
fi
# Only the Regrasp shards by default. The bc_dataset directory accumulates every
# phase's data — 40-odd Phase-1/4 files that predate `bin_assigned` and can only
# ever report "unknown", burying the two lines this script exists to show.
# ALL=1 lists everything.
PAT='*regrasp*.h5'
[ "${ALL:-0}" = "1" ] && PAT='*.h5'
found=0
for h in "$REGRASP_DATA"/bc_dataset/$PAT; do
    found=1
    $PY - "$h" <<'PROBE'
import sys, os
import h5py, numpy as np
# Load directions.py AS A FILE, not through `handover_sim2real.regrasp`. The
# package __init__ pulls in the simulator, which prints four lines of gym
# deprecation warning per import — once per shard, drowning the table. The module
# itself is pure numpy by design (see its header) and imports standalone.
import importlib.util as _u
_s = _u.spec_from_file_location(
    "_rg_dirs", os.path.join("handover_sim2real", "regrasp", "directions.py"))
D = _u.module_from_spec(_s); _s.loader.exec_module(D)
p = sys.argv[1]
try:
    with h5py.File(p, "r") as f:
        eps = [k for k in f if k.startswith("episode")]
        done = f.attrs.get("complete", None)
        # Run 1 vs run 2+ rule: under the bin-axis command d_world IS
        # to_world(BINS[b], anchor_R), so the angle is 0. Run 1's is ~20 deg.
        off = []
        for k in eps[:32]:
            a = f[k].attrs
            b, R, d = int(a.get("bin_assigned", -1)), a.get("anchor_R"), a.get("d_world")
            if b >= 0 and R is not None and d is not None:
                off.append(float(D.angle_between(np.asarray(d),
                                                 D.to_world(D.BINS[b], np.asarray(R)))))
    med = float(np.median(off)) if off else float("nan")
    state = ("INCOMPLETE (killed mid-collection)" if done is not None and not done
             else "complete" if done else "complete (legacy, no flag)")
    rule = ("bin axis  [run 2+]" if med == med and med < 1.0
            else f"grasp axis {med:.1f} deg  [RUN 1 — stale for run 2+]"
            if med == med else "unknown")
    print(f"  {os.path.basename(p):<34} {len(eps):>5} ep  "
          f"{os.path.getsize(p)/1e9:.2f} GB  {state}  {rule}")
except Exception as e:
    print(f"  {os.path.basename(p):<34} UNREADABLE: {e}")
PROBE
done
[ "$found" = "0" ] && echo "  (none in $REGRASP_DATA/bc_dataset/)"
echo

bold "run directories (scratch)"
found=0
for d in "$REGRASP_DATA"/dagger_runs/*/; do
    found=1
    n=$(grep -c . "$d/dagger_log.csv" 2>/dev/null || echo 0)
    echo "  $(basename "$d"):  $(( n > 0 ? n - 1 : 0 )) iterations logged"
done
[ "$found" = "0" ] && echo "  (none)"
echo

bold "queue"
squeue -u "$USER" 2>/dev/null || echo "  (squeue unavailable — not on a cluster)"
echo
echo "Next: bash examples/slurm/regrasp_pipeline.sh --dry-run"
echo "      (same probes, then prints which stages it would submit)"
