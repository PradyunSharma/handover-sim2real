#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Readable progress from a Regrasp training log.
#
#   bash examples/watch_regrasp.sh                    # last 20 lines, then exit
#   bash examples/watch_regrasp.sh -f                 # follow live
#   bash examples/watch_regrasp.sh -f output/logs/foo/train.log
#
# WHY THIS EXISTS. The trainer's per-epoch summary and the collector's episode
# counter are already in the log, but `tail` shows neither: tqdm redraws its
# progress bar with a CARRIAGE RETURN and no newline, so hundreds of bar frames
# and the one line you want share a single physical line. Translating \r to \n
# splits them apart, and then the interesting lines can be grepped out.
#
# The filter keeps exactly the run-1-style view:
#   epoch NNN  train_total=... | val_total=...  (Ns)
#   [iter NN/NN]  beta=...  m=... episodes over ... scenes
#   [collect] / [policy] / the episode counter
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

FOLLOW=0
[ "${1:-}" = "-f" ] && { FOLLOW=1; shift; }
LOG="${1:-output/logs/regrasp3_fast1/train.log}"
[ -f "$LOG" ] || { echo "no log at $LOG"; exit 1; }

# `[iter` and `[collect]` are literal brackets, so the alternation is anchored on
# the words rather than on a bracket class that grep -E would read as a set.
KEEP='train_total|\[iter [0-9]|\[collect\]|\[policy\]|episodes=[0-9]|^Steps/epoch|^Run dir|^\[train\] (warm start|MODEL)|^=====+$'

if [ "$FOLLOW" -eq 1 ]; then
  # stdbuf on BOTH stages: tr and grep block-buffer when their output is a pipe,
  # which would hold your lines hostage for 4 kB — the same buffering that made
  # a collection look dead for an hour once.
  tail -n 2000 -f "$LOG" \
    | stdbuf -o0 tr '\r' '\n' \
    | stdbuf -o0 grep --line-buffered -E "$KEEP"
else
  tr '\r' '\n' < "$LOG" | grep -E "$KEEP" | tail -n "${LINES_OUT:-20}"
fi
