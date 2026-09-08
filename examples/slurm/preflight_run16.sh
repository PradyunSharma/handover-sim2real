#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# WILL RUN 16 START, OR WILL IT DIE SIX HOURS IN? Run this on a DelftBlue LOGIN
# node before submitting anything. It touches no GPU and finishes in seconds.
#
#     bash examples/slurm/preflight_run16.sh
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
#   4. CONSISTENCY. `d_point_depth` in the config against the pin table's
#      `_meta`, when the table exists. This is the one that produces a
#      plausible, wrong result rather than a crash: 0.1122 vs 0.1034 re-bins 8%
#      of grasps, and until recently `resolve_d_rule` compared only the rule
#      NAME and would have let it through.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.."

RUN=regrasp_run16
CFG=examples/configs/regrasp_run16.yaml
export SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/h2r-runs}"
export REGRASP_DATA="${REGRASP_DATA:-$SCRATCH_ROOT/output}"
NEED_GB=5

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
warn() { printf '  \033[33mNOTE\033[0m  %s\n' "$*"; }

echo "preflight: $RUN"
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
want = (cfg.get("d_rule", "approach_axis"),
        float(cfg.get("d_point_depth", 0.1122)))
if not os.path.exists(pin):
    print(f"        {pin} not built yet — nothing to compare (fine)")
    raise SystemExit(0)
m = (json.load(open(pin)).get("_meta") or {})
got = (m.get("d_rule", "approach_axis"), float(m.get("d_point_depth", 0.1122)))
if want == got:
    print(f"        config and table agree: {got[0]} @ {got[1]}")
else:
    print(f"        MISMATCH  config={want}  table={got}")
    print("        Rebuild the table at the config's depth, or the run is scored")
    print("        against bins captioned by a different definition of `d`.")
    raise SystemExit(1)
PY
[ $? -eq 0 ] && ok "no depth/rule mismatch" || bad "config and pin table disagree"

echo
if [ "$fail" -eq 0 ]; then
    printf '\033[32mPREFLIGHT PASSED\033[0m — submit with:\n'
    echo "  J1=\$(sbatch --parsable examples/slurm/regrasp_run16_all.sbatch)"
    echo "  sbatch --dependency=afterany:\$J1 examples/slurm/regrasp_run16_all.sbatch"
else
    printf '\033[31mPREFLIGHT FAILED\033[0m — fix the FAIL lines above before submitting.\n'
    exit 1
fi
