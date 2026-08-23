#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# THE WHOLE REGRASP RUN, SUBMITTED ONCE. Run this on a DelftBlue LOGIN node:
#
#     bash examples/slurm/regrasp_pipeline.sh
#
# It submits six chained SLURM jobs and returns in about a second. Nothing after
# this needs a human until the figures exist.
#
#     bash examples/slurm/regrasp_pipeline.sh --dry-run     # print, submit nothing
#     RUN=regrasp_run3 bash examples/slurm/regrasp_pipeline.sh
#     SKIP_SMOKE=1 bash examples/slurm/regrasp_pipeline.sh  # if it already passed
#
# WHY SIX JOBS AND NOT ONE. The chain is ~30 h of wall clock and DelftBlue's
# maximum is 24, so a single allocation cannot hold it. SLURM dependencies are
# the mechanism that exists for this, and they are strictly better than one long
# job anyway: a failure costs only the stage that failed, the independent stages
# run in PARALLEL rather than in sequence, and each stage asks for the resources
# it actually needs instead of the maximum any stage needs.
#
#     A  collect train demos      GPU  ~4 h   ──┐
#     B  collect val demos        GPU  ~20 m  ──┼──> D  smoke   ──> E  train
#     C  build test dir. table    GPU  ~2 h   ──┼──────────────────────┴──> F  test eval
#                                               │
#     (A, B, C start immediately and in parallel)
#
# D also DERIVES `demo_ok_table` from A's shard, and F derives the test pin table
# from C's direction table — both are seconds of pure combinatorics, and folding
# them into the job that needs them is what removes the two manual steps that
# used to sit between the long jobs.
#
# IDEMPOTENT. Every stage whose output already exists is skipped, so re-running
# this after a partial failure resubmits only what is missing. That matters most
# for A: 4 h of collection should never be repeated because a later stage died.
# `--force` overrides and resubmits everything.
#
# WHAT IT DOES NOT DO. It does not wait, and it does not check results. Watch it
# with `squeue -u $USER` and read `slurm_logs/`. If the smoke FAILS the chain
# stops there by construction (`afterok`), which is the point of running it.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root, wherever this was invoked from

# ── knobs ────────────────────────────────────────────────────────────────────
RUN="${RUN:-regrasp_run2}"
CFG="${CFG:-examples/configs/regrasp_run2.yaml}"
SMOKE_CFG="${SMOKE_CFG:-examples/configs/regrasp_smoke.yaml}"
SIM_CFG="${SIM_CFG:-examples/pretrain_multicam_wr.yaml}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$USER/handover-sim2real}"
TRAIN_H5="${TRAIN_H5:-output/bc_dataset/train_regrasp.h5}"
VAL_H5="${VAL_H5:-output/bc_dataset/val_regrasp.h5}"
TEST_ITERS="${TEST_ITERS:-0,5,10,15,20}"   # which iterations the test sweep scores
SKIP_SMOKE="${SKIP_SMOKE:-0}"
FORCE="${FORCE:-0}"
DRY=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --force)   FORCE=1 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# `sbatch --parsable` prints the job id and nothing else, which is what makes the
# dependency chain expressible. In --dry-run the command is printed and a fake id
# returned, so the whole chain can be inspected without touching the queue.
submit() {
    local desc="$1"; shift
    if [ "$DRY" = "1" ]; then
        printf '  \033[2m%s\033[0m\n' "sbatch $*" >&2
        # A counter would not work: `submit` runs inside $( ), i.e. a subshell,
        # so any increment is discarded. Name the placeholder after the stage
        # instead, which also makes the printed dependency graph readable.
        echo "<${desc%% *}>"
        return
    fi
    local jid
    jid="$(sbatch --parsable "$@")"
    note "$desc -> job $jid"
    echo "$jid"
}

# ── preflight: the two things this pipeline cannot produce ───────────────────
# The TRAIN direction table is the output of a sim pass that is not part of this
# chain (it is built once and reused across runs), and the configs are the
# experiment itself. Failing here costs a second; failing inside job A costs a
# queue wait plus four hours.
for f in output/direction_table_train.json output/direction_table_val.json \
         "$CFG" "$SMOKE_CFG" "$SIM_CFG"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
mkdir -p slurm_logs output/bc_dataset

say "Regrasp pipeline   run=$RUN   cfg=$CFG"
note "scratch : $SCRATCH_ROOT"
[ "$DRY" = "1" ] && note "DRY RUN — nothing will be submitted"

# ── stage 0: the per-bin assignment (seconds, here, not in a job) ────────────
# Pure combinatorics over the direction table. Running it inline means a mistake
# in the bin set or the drop list surfaces NOW, with its histogram printed, and
# not four hours later inside a collection job.
if [ "$FORCE" = "1" ] || [ ! -f output/regrasp_pins_train.json ]; then
    say "[0] assigning train demos (one per reachable bin)"
    [ "$DRY" = "1" ] || python examples/assign_direction_demos.py \
        --table output/direction_table_train.json \
        --out output/regrasp_pins_train --mode per-bin \
        --drop-bins='-z_beneath,-x_over_fingers'
else
    note "[0] output/regrasp_pins_train.json exists — skipping"
fi
if [ "$FORCE" = "1" ] || [ ! -f output/regrasp_pins_val.json ]; then
    [ "$DRY" = "1" ] || python examples/assign_direction_demos.py \
        --table output/direction_table_val.json \
        --out output/regrasp_pins_val --mode per-bin \
        --drop-bins='-z_beneath,-x_over_fingers' >/dev/null
    note "[0] assigned val demos"
else
    note "[0] output/regrasp_pins_val.json exists — skipping"
fi

# ── is an existing shard RUN 2's, or run 1's left lying in the same path? ────
# "The file exists" is not "the file is right". Run 1's shards sit at exactly
# these paths and are wrong for run 2 in two independent ways: they hold the
# max-separated PAIR rather than one demo per bin, and their `d_world` is the
# grasp's own axis rather than the bin's. Skipping collection because the path is
# occupied would train run 2 on run 1's data and produce a plausible, wrong
# result — the worst failure mode available here.
#
# The fingerprint is exact: under run 2's rule `d_world` IS `to_world(BINS[b],
# anchor_R)`, so the angle between them is 0; under run 1's it is a median 22 deg
# (measured on the existing val shard). A handful of episodes settles it.
shard_is_current() {
    python - "$1" <<'PROBE' 2>/dev/null
import sys
import h5py, numpy as np
sys.path.insert(0, ".")
from handover_sim2real.regrasp import directions as D
with h5py.File(sys.argv[1], "r") as f:
    eps = [k for k in f if k.startswith("episode_")]
    if not eps or not bool(f.attrs.get("complete", True)):
        raise SystemExit(1)
    off = []
    for k in eps[:32]:
        a = f[k].attrs
        b, R, d = int(a.get("bin_assigned", -1)), a.get("anchor_R"), a.get("d_world")
        if b < 0 or R is None or d is None:
            raise SystemExit(1)
        off.append(float(D.angle_between(np.asarray(d),
                                         D.to_world(D.BINS[b], np.asarray(R)))))
    raise SystemExit(0 if float(np.median(off)) < 1.0 else 1)
PROBE
}

# Never deleted, only moved aside: a stale shard is hours of simulator time and
# the only copy of what run 1 trained on.
retire_stale() {
    local f="$1" to="${1%.h5}.run1-stale.h5"
    say "    ! $f was collected under RUN 1's rule (command = grasp axis)."
    note "      moving it to $to and re-collecting"
    [ "$DRY" = "1" ] || mv "$f" "$to"
}

# ── stages A, B, C: independent, submitted together ──────────────────────────
say "[A-C] collection and the test-split table (parallel)"
DEPS=()

# The flags, not just the move, so --dry-run reports the jobs it WOULD submit
# rather than the ones it would submit if the move it did not perform had.
TRAIN_STALE=0; VAL_STALE=0
if [ -f "$TRAIN_H5" ] && [ "$FORCE" != "1" ] && ! shard_is_current "$TRAIN_H5"; then
    retire_stale "$TRAIN_H5"; TRAIN_STALE=1
fi
if [ -f "$VAL_H5" ] && [ "$FORCE" != "1" ] && ! shard_is_current "$VAL_H5"; then
    retire_stale "$VAL_H5"; VAL_STALE=1
fi
# The (scene, bin) filter is derived from the train shard, so a re-collection
# invalidates it too. train_regrasp.sbatch regenerates it when absent.
if { [ ! -f "$TRAIN_H5" ] || [ "$TRAIN_STALE" = "1" ]; } \
        && [ -f output/regrasp_demos_train_ok.json ]; then
    note "    ! output/regrasp_demos_train_ok.json is stale with it — removing"
    [ "$DRY" = "1" ] || rm -f output/regrasp_demos_train_ok.json
fi

if [ "$FORCE" = "1" ] || [ "$TRAIN_STALE" = "1" ] || [ ! -f "$TRAIN_H5" ]; then
    A=$(submit "A collect train (~4 h)" --time=20:00:00 \
        --export=ALL,SPLIT=train,SIM_CFG="$SIM_CFG",OUT="$TRAIN_H5",PIN=output/regrasp_pins_train.json \
        examples/slurm/collect_regrasp_demos.sbatch)
    DEPS+=("$A")
else
    note "A $TRAIN_H5 exists — skipping collection"
fi

if [ "$FORCE" = "1" ] || [ "$VAL_STALE" = "1" ] || [ ! -f "$VAL_H5" ]; then
    B=$(submit "B collect val (~20 min)" --time=03:00:00 \
        --export=ALL,SPLIT=val,SIM_CFG="$SIM_CFG",OUT="$VAL_H5",PIN=output/regrasp_pins_val.json \
        examples/slurm/collect_regrasp_demos.sbatch)
    DEPS+=("$B")
else
    note "B $VAL_H5 exists — skipping collection"
fi

# C is needed only by F, so it is deliberately NOT in DEPS — the training chain
# must not wait on a table it never reads.
C=""
if [ "$FORCE" = "1" ] || [ ! -f output/direction_table_test.json ]; then
    C=$(submit "C build test direction table (~2 h)" --time=06:00:00 \
        --export=ALL,SPLIT=test,SIM_CFG="$SIM_CFG",OUT=output/direction_table_test.json \
        examples/slurm/build_direction_table.sbatch)
else
    note "C output/direction_table_test.json exists — skipping"
fi

# `--dependency=afterok:a:b` waits for BOTH and refuses to start if either fails,
# which is what makes a broken stage stop the chain instead of feeding garbage
# forward. An empty dependency list means the predecessors were all skipped.
dep_of() {
    local ids=("$@") out=""
    for i in "${ids[@]}"; do [ -n "$i" ] && out="${out}:${i}"; done
    [ -n "$out" ] && echo "--dependency=afterok${out}" || true
}

# ── stage D: the shakedown ───────────────────────────────────────────────────
# Two iterations, m=8, three eval scenes. It also derives `demo_ok_table` from
# A's shard (train_regrasp.sbatch does that when the file is missing), so this
# job is what proves BOTH the wiring and the derivation before E commits 24 h.
PREV=()
if [ "$SKIP_SMOKE" = "1" ]; then
    note "[D] SKIP_SMOKE=1 — no shakedown (the chain will not catch a wiring error)"
    PREV=("${DEPS[@]}")
else
    say "[D] shakedown"
    D=$(submit "D smoke (~30 min)" --time=01:30:00 $(dep_of "${DEPS[@]:-}") \
        --export=ALL,RUN=regrasp_smoke,CFG="$SMOKE_CFG",SCRATCH_ROOT="$SCRATCH_ROOT" \
        examples/slurm/train_regrasp.sbatch)
    PREV=("$D")
fi

# ── stage E: the run ─────────────────────────────────────────────────────────
say "[E] the run"
E=$(submit "E train $RUN (~20-21 h)" --time=24:00:00 $(dep_of "${PREV[@]:-}") \
    --export=ALL,RUN="$RUN",CFG="$CFG",SCRATCH_ROOT="$SCRATCH_ROOT" \
    examples/slurm/train_regrasp.sbatch)

# ── stage F: the held-out number ─────────────────────────────────────────────
# Waits on E (the checkpoints) and C (the test direction table), and derives the
# test pin table itself. `--chained` runs the real retry ladder beside the
# independent sweep; TEST_ITERS keeps it to a few points, because the test curve
# needs a trend and not every iteration.
say "[F] test-split evaluation"
# ITERS IS EXPORTED THROUGH THE ENVIRONMENT, NOT THROUGH --export=ALL,K=V.
# sbatch splits that list on commas ITSELF, so `ITERS=0,5,10` arrives as
# ITERS=0 plus three bare names — shell quoting cannot prevent it, because the
# whole thing is one argv word by the time sbatch sees it. `--export=ALL` already
# carries the submitting shell's environment, so setting the variable here is
# both simpler and immune.
F=$(ITERS="$TEST_ITERS" submit "F test eval, iters=$TEST_ITERS (~2 h)" \
    $(dep_of "$E" "$C") \
    --export=ALL,RUN="$RUN",CHAINED=1,SCRATCH_ROOT="$SCRATCH_ROOT" \
    examples/slurm/eval_regrasp_testset.sbatch)

# ── what to do next ──────────────────────────────────────────────────────────
RD="$SCRATCH_ROOT/output/dagger_runs/$RUN"
cat <<EOF

$(say "Submitted. Nothing further is needed until it finishes.")
  watch     squeue -u $USER
  logs      tail -f slurm_logs/regrasp_*.out
  plot      python examples/plot_regrasp_run.py $RD
            (safe at any time — read-only on the CSV)
  figures   $RD/{training_curve,curves_regrasp,debug_dagger,curves_diag,media_curves}.png
            $RD/test_set_evaluation.png            (after F)
  harvest   python examples/harvest_run.py --run-dir $RD
            /scratch is purged by age — pull the CSVs and figures into git.

  If a stage fails, fix it and re-run this script: every completed stage is
  skipped, so only the missing work is resubmitted.
EOF
