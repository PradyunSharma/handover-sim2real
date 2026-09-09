#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# WILL THIS RUN START, OR WILL IT DIE SIX HOURS IN? Run this on a DelftBlue
# LOGIN node before submitting anything. It touches no GPU, finishes in seconds.
#
#     bash examples/slurm/preflight_regrasp.sh regrasp_run18
#     bash examples/slurm/preflight_regrasp.sh regrasp_run17
#
# It checks the four things that kill this run, in the order they would kill it:
#
#   1. DISK. /home is a HARD 30 GB quota that fills SILENTLY — no ENOSPC
#      traceback, because Python cannot write one to a full disk. The job dies
#      with exit code 6 and an EMPTY error file, six hours into a collection.
#      This is the check that matters most and the one nothing else performs.
#   2. ENVIRONMENT. conda env, GADDPG_DIR, OMG_PLANNER_DIR, torch+CUDA, h5py.
#      A missing env var fails in the first second, which is cheap — but it
#      fails AFTER the queue wait, which is not.
#   3. INPUTS. Delegated to `check_regrasp_inputs.py`, which reads the paths OUT
#      OF THE CONFIG rather than from a remembered filename, so it cannot
#      disagree with what the job will open. Missing inputs are EXPECTED on a
#      first run — the sbatch builds them — so they are reported as a plan, not
#      as an error.
#   4. CONSISTENCY. `d_rule`, `d_point_depth` AND `anchor_hand_ref` in the
#      config against the pin table's `_meta`, when the table exists. These are
#      the ones that produce a plausible, wrong result rather than a crash.
#      0.1122 vs 0.1034 re-bins 8% of grasps. The anchor reference is worse:
#      run 16 set `hand_centroid` while `build_direction_table.py` hardcoded the
#      MANO wrist, and since those two frames are roughly ANTI-ALIGNED,
#      `bin_assigned == bin_realized` fell 99% -> 40%, the miscaption filter ate
#      44% of every DAgger shard, and training saw 19% of what was collected.
#      Twenty iterations, plausible curves, never beat iteration 0.
#      `setup.resolve_anchor_ref` now refuses this at load; the check is here so
#      it is caught before the queue wait rather than after it.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

RUN="${1:-}"
if [ -z "$RUN" ]; then
    echo "usage: bash examples/slurm/preflight_regrasp.sh <run-name>" >&2
    echo "   e.g. bash examples/slurm/preflight_regrasp.sh regrasp_run18" >&2
    exit 2
fi
CFG="examples/configs/${RUN}.yaml"
if [ ! -f "$CFG" ]; then
    echo "no such config: $CFG" >&2
    exit 2
fi
export SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/h2r-runs}"
export REGRASP_DATA="${REGRASP_DATA:-$SCRATCH_ROOT/output}"
NEED_GB=5

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
warn() { printf '  \033[33mNOTE\033[0m  %s\n' "$*"; }

echo "preflight: $RUN   ($CFG)"
echo "  REGRASP_DATA = $REGRASP_DATA"
echo "  OUT_ROOT     = $SCRATCH_ROOT/output/dagger_runs"
echo

# ── 1. disk ─────────────────────────────────────────────────────────────────
echo "[1/4] disk on /home (need ~${NEED_GB} GB free)"
if command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null | grep -q .; then
    quota -s 2>/dev/null | sed 's/^/        /'
fi
avail_kb=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
avail_gb=$(( avail_kb / 1024 / 1024 ))
used_pct=$(df -Pk "$HOME" | awk 'NR==2 {print $5}')
if [ "$avail_gb" -ge "$NEED_GB" ]; then
    ok "$avail_gb GB free on $HOME ($used_pct used)"
else
    bad "only $avail_gb GB free on $HOME, need ~$NEED_GB GB ($used_pct used)"
    warn "biggest directories under \$HOME:"
    du -sh "$HOME"/* 2>/dev/null | sort -rh | head -5 | sed 's/^/        /'
fi
if ! mkdir -p "$REGRASP_DATA/bc_dataset" 2>/dev/null; then
    bad "cannot create $REGRASP_DATA/bc_dataset"
else
    ok "$REGRASP_DATA/bc_dataset is writable"
fi

# ── 2. environment ──────────────────────────────────────────────────────────
echo
echo "[2/4] environment"
[ -n "${GADDPG_DIR:-}" ] && [ -d "${GADDPG_DIR}" ] \
    && ok "GADDPG_DIR=$GADDPG_DIR" || bad "GADDPG_DIR unset or missing (export GADDPG_DIR=\$PWD/GA-DDPG)"
[ -n "${OMG_PLANNER_DIR:-}" ] && [ -d "${OMG_PLANNER_DIR}" ] \
    && ok "OMG_PLANNER_DIR=$OMG_PLANNER_DIR" || warn "OMG_PLANNER_DIR unset (only collection needs it)"
python - <<'PY' && ok "python deps import" || bad "python deps missing — is the conda env active?"
import importlib, sys
miss = [m for m in ("torch", "h5py", "yaml", "numpy", "scipy", "matplotlib")
        if not importlib.util.find_spec(m)]
if miss:
    print("        missing:", ", ".join(miss)); sys.exit(1)
import torch
print(f"        torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
PY

# ── 3. inputs ───────────────────────────────────────────────────────────────
echo
echo "[3/4] inputs declared by $CFG"
out="$(python examples/check_regrasp_inputs.py "$RUN" 2>&1)"
echo "$out" | sed 's/^/      /'
n_missing=$(echo "$out" | grep -c '\*\* MISSING \*\*')
if [ "$n_missing" -eq 0 ]; then
    ok "every input present — the sbatch will skip straight to training"
else
    warn "$n_missing input(s) missing. EXPECTED on a first run: stages 1-4 of"
    warn "regrasp_run16_all.sbatch build exactly these. Only a problem if you"
    warn "meant to reuse an existing table."
fi

# ── 4. depth consistency ────────────────────────────────────────────────────
echo
echo "[4/4] config vs pin table (the silent-wrong-answer check)"
python - "$CFG" <<'PY'
import json, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))["SIM"]
pin = cfg["grasp_pin_table"]
if not os.path.exists(pin):
    print(f"        {pin} not built yet — nothing to compare (fine)")
    raise SystemExit(0)
m = (json.load(open(pin)).get("_meta") or {})
# (config value, table value, label). Defaults are what a table PREDATING each
# key was built with, so an old table compares equal instead of failing.
checks = [
    ("d_rule",          cfg.get("d_rule", "approach_axis"),
                        m.get("d_rule", "approach_axis")),
    ("d_point_depth",   float(cfg.get("d_point_depth", 0.1122)),
                        float(m.get("d_point_depth", 0.1122))),
    ("anchor_hand_ref", cfg.get("anchor_hand_ref", "wrist"),
                        m.get("anchor_hand_ref", "wrist")),
]
if str(cfg.get("d_rule")) == "location_extent":
    checks += [
        ("d_m_min",     float(cfg.get("d_m_min", 0.15)),
                        float(m.get("d_m_min", 0.15))),
        ("d_extent_pct", float(cfg.get("d_extent_pct", 95.0)),
                         float(m.get("d_extent_pct", 95.0))),
    ]
bad = [(k, a, b) for k, a, b in checks if a != b]
for k, a, b in checks:
    print(f"        {'MISMATCH' if (k, a, b) in bad else 'agree   '}  "
          f"{k:16s} config={a!r:20s} table={b!r}")
if bad:
    print("        The table's bins were NAMED under its values. Rebuild it with")
    print("        the config's, or the run is scored against a different frame")
    print("        and/or a different definition of `d` than it collected under.")
    raise SystemExit(1)
PY
[ $? -eq 0 ] && ok "no depth/rule mismatch" || bad "config and pin table disagree"

echo
if [ "$fail" -eq 0 ]; then
    printf '\033[32mPREFLIGHT PASSED\033[0m — submit with:\n'
    echo "  J1=\$(sbatch --parsable examples/slurm/${RUN}_all.sbatch)"
    echo "  sbatch --dependency=afterany:\$J1 examples/slurm/${RUN}_all.sbatch"
else
    printf '\033[31mPREFLIGHT FAILED\033[0m — fix the FAIL lines above before submitting.\n'
    exit 1
fi
