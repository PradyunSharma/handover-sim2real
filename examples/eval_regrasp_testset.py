"""
Score a finished Regrasp run on a WHOLE split — held-out test, or the full train set.

    python examples/eval_regrasp_testset.py --run-dir output/dagger_runs/regrasp_run2
    python examples/eval_regrasp_testset.py --run-dir output/dagger_runs/regrasp_run2 \\
        --split train                    # every train scene, not the eval subsample
    python examples/eval_regrasp_testset.py --run-dir output/dagger_runs/regrasp_run2 \\
        --iters 0,5,10,15,20 --chained
    python examples/eval_regrasp_testset.py --run-dir output/dagger_runs/regrasp_run2 \\
        --split train --plot-only        # re-render from an existing CSV

WHY THIS IS A SEPARATE SCRIPT AND NOT A FLAG ON THE TRAINER

The in-loop evaluation runs with `EVAL.holdout: false` on a `np.linspace`
SUBSAMPLE of the train split — 100 scenes of ~617 in run 2, 40 in the fast run —
and those scenes are also collected on. That is deliberate: a fixed subsample
makes consecutive iterations a PAIRED comparison, so a trend is readable at a
sample size where a single point is not. But it is a train-set number on a tenth
of the train set, and it must never be reported as anything else.

This script answers the other two questions over a WHOLE split, once, after
training — after, because it needs the entire iteration sequence and because
putting it inline would double an already-long job for a number nobody acts on
mid-run.

  --split test   (the default) scenes the run has never seen in any form: not
      held out from a pool, held out by the benchmark's own s0 split. THE
      generalisation number.

  --split train  every scene the run could have collected on, at ~6x the in-loop
      sample. Still not held out — it is the in-loop curve's own question asked
      at a sample size where per-BIN rates stop being noise. Use it to decide
      whether a per-bin gap seen in `training_curve.png` is real, and never as
      evidence the policy generalises. `demo_ok_table` is applied here (see
      `main`) so the sweep scores the pairs the policy was actually taught.

TWO EVALUATIONS, AND THEY ANSWER DIFFERENT QUESTIONS

  INDEPENDENT (always)  Every (scene, bin) pair is a fresh rollout from home. This
      is what the per-bin panels are built from, and it is the honest measure of
      "given this command, can the policy execute it". `retry_at_k` derived from
      it is an OR over independent attempts and therefore a CEILING: it assumes
      a failed attempt costs nothing and leaves the world untouched.

  CHAINED (--chained)   The retry ladder, run for real: attempt 1 fails, the arm
      rewinds to 30% of the trajectory it just flew, and attempt 2 is commanded
      from there. `chained_retry_at_k` is the number a deployment would see, and
      the GAP against the independent `retry_at_k` is what the reset-based
      version was giving away. Costs roughly 1.6x the independent sweep (most
      scenes stop after one or two attempts), so it is opt-in.

      Still an upper bound, and the reason is written down rather than hidden:
      the rewind resets the simulator, which rewinds the human's DexYCB playback
      with it. A real retreat controller drives the arm back along its own joint
      path without touching the human. That is the deferred experiment.

OUTPUT, named by split so a train sweep cannot overwrite a test one

  <run>/<split>_eval_log.csv     one row per iteration, the same ~340-column
                                 schema as dagger_log.csv, so the two are
                                 directly diffable column for column
  <run>/<split>_set_evaluation.png  the training_curve.png layout — one ROW per
                                 commanded direction, columns success stages /
                                 chance vs conversion / approach error — plus a
                                 conditioning row: ended-in-the-commanded-bin per
                                 bin, arrived-from-the-commanded-side per bin,
                                 and retry@k with the chained curve overlaid when
                                 --chained was used.

THE COMMAND VECTOR COMES FROM THE RUN, NOT FROM THE SPLIT

`SIM.command_deploy: bin_centroid` — runs 9 and 19-23 — means "condition the
policy on the EMPIRICAL MEAN of each bin's assigned `d_anchor`", and
`resolve_command_axes` computes that mean from whichever pin table it is handed.
Handed the TEST table it returns the test split's centroids, which are not the
vectors the policy was trained to obey: measured between `regrasp_pins_test.json`
and run 19's saved axes they differ by 2.9 deg on `+x` and 6.0 deg on `+z`. The
policy would then be scored on a command no part of its training ever issued,
and nothing would look wrong.

So this script loads the axes from the RUN's own `command_axes.json` — the same
file `sim2real/my_regrasp_policy_runner.py` reads on the robot — and overrides
whatever the split's table implies. `--command-axes` points it elsewhere;
`--split-axes` restores the old behaviour for the one case where it is the right
question (asking what the test split's own geometry would have commanded).

ADAPTIVE DIRECTION ORDERING (--adaptive)

`retry_at_k` has always been reduced over `directions.RETRY_LADDER`, a FIXED
order hardcoded from run 11's per-bin success. That is the right constant for
comparing runs and the wrong one for asking what a deployment achieves: a robot
does not know run 11, it knows what has worked on the objects it has seen.

`--bins` passes the order to start from and `--adaptive` lets the sweep revise
it, scene by scene, from a Beta-Bernoulli posterior per bin (see
`regrasp/bin_ranker.py`). The order used on scene i is the order standing BEFORE
scene i is rolled out, so `adaptive_retry_at_k` is causal — it never ranks by a
rate computed from the episode it is about to score.

READ THE TWO SWEEPS DIFFERENTLY, because the ranker does very different work in
each. In the INDEPENDENT sweep every (scene, bin) pair is rolled out whatever the
order, so per-bin success, `dir_err` and the outcome taxonomy are
order-invariant: the ranker changes the REDUCTION and nothing else, and a higher
`adaptive_retry_at_k` is the same rollouts ordered better, not a better policy.
Under `--chained` or `--stop-on-success` the order decides which rollouts happen
at all, so it buys real attempts — `mean_attempts` is the number that moves —
and the per-bin denominators stop being comparable, which `rank_n_b*` reports.

CHECKPOINT. `last`, not `best`, matching the run's own `EVAL.ckpt`. Every Phase-4
run scored `best.pt`; from run 2 on the whole pipeline — collection, in-loop eval,
warm start and this script — reads the same `last.pt`, so "the policy at iteration
i" names one set of weights everywhere instead of three.

PREREQUISITE: a direction table and pin table for the split, which are a separate
sim pass — except on train, where the run's own tables already exist and are
reused as-is:

    python examples/build_direction_table.py --split test \\
        --out output/direction_table_test.json
    python examples/assign_direction_demos.py --table output/direction_table_test.json \\
        --out output/regrasp_pins_test --drop-bins='-z_beneath,-x_over_fingers'

There is deliberately NO `demo_ok_table` on test, and there deliberately IS one
on train. The file records which (scene, bin) pairs base COLLECTION managed to
demonstrate. Nothing is collected on test, so the pin table alone states
feasibility there — it was built by calling the planner on every scene, and
filtering test by the train split's collection failures would score on a set
defined by an unrelated run. On train the pairs OMG could not demonstrate are
pairs the policy was never taught, so scoring them charges the policy for
directions absent from its data and breaks comparability with the in-loop curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from handover_sim2real.regrasp import load_policy_runner       # noqa: E402
from handover_sim2real.regrasp import directions as _D         # noqa: E402
from handover_sim2real.regrasp.bin_ranker import (             # noqa: E402
    BinRanker, parse_bin_sequence,
)
from handover_sim2real.regrasp.evaluator import (              # noqa: E402
    aggregate_eval_rows, eval_num_grasps, eval_one, off_table_slot,
)
from handover_sim2real.regrasp.setup import (                     # noqa: E402
    build_regrasp_context, expand_config_paths,
)

from eval_regrasp_run import iteration_dirs                    # noqa: E402
from train_regrasp import LOG_FIELDS, eval_columns             # noqa: E402


# The test CSV is the dagger_log schema plus its own provenance, so a column
# means the same thing in both files and `diff <(cut -f...) ...` is meaningful.
TEST_FIELDS = (["iter", "run_dir", "ckpt", "split", "num_scenes", "num_episodes",
                "eval_s"]
               + [c for c in LOG_FIELDS
                  if c not in ("iter", "beta", "m", "is_best", "collect_s",
                               "train_s", "eval_s", "wall_s")
                  and not c.startswith("c_")]
               + ["chained_retry_at_1", "chained_retry_at_2",
                  "chained_retry_at_3", "chained_retry_at_4",
                  "solved_rate", "mean_attempts", "mean_attempts_to_success",
                  "signal_human_rate", "mean_branch_step", "replay_err_mean"]
               + [f"chain_succ_bin_{b}" for b in range(len(_D.BINS))]
               + [f"chain_n_bin_{b}" for b in range(len(_D.BINS))]
               # ---- THE CHAIN'S OWN RATE FAMILY, all `chain_`-prefixed ------
               # The prefix is not cosmetic. `chained_metrics` used to emit
               # `dir_err` and `dir_track` unprefixed — the same names the
               # INDEPENDENT evaluation writes — so a --chained row silently
               # overwrote the single-shot direction numbers with the chain's,
               # and one column header stood over two different measurements.
               # Everything the chain produces is namespaced so that class of
               # collision cannot recur.
               + ["chain_n_attempts", "chain_n_fail", "chain_dir_err",
                  "chain_dir_track", "chain_close_rate", "chain_grasp_rate",
                  "chain_box_chance_rate", "chain_box_taken_rate",
                  "chain_miss_given_box"]
               + [f"chain_f_{k}" for k in ("grasp_ok", "grasp_miss",
                                           "no_release", "drop", "timeout",
                                           "human_contact")]
               + [f"chain_ff_{k}" for k in ("grasp_miss", "no_release", "drop",
                                            "timeout", "human_contact")]
               + [f"chain_dir_err_b{b}" for b in range(len(_D.BINS))]
               + [f"chain_dir_track_b{b}" for b in range(len(_D.BINS))]
               + [f"chain_bin_diag_b{b}" for b in range(len(_D.BINS))]
               + [f"chain_n_realized_b{b}" for b in range(len(_D.BINS))]
               # ---- THE ADAPTIVE LADDER (--adaptive / --bins) --------------
               # `adaptive_retry_at_k` sits BESIDE `retry_at_k` rather than
               # replacing it. They are the same episodes reduced under two
               # different attempt orders — the fixed `RETRY_LADDER` and the one
               # the sweep learned — and the GAP between them is the entire
               # measurement. Overwriting the fixed column would delete the
               # baseline the adaptive number is only meaningful against, and
               # would silently break every comparison against runs 16-20.
               + [f"adaptive_retry_at_{k}" for k in range(1, len(_D.BINS) + 1)]
               + [f"adaptive_retry_n_{k}" for k in range(1, len(_D.BINS) + 1)]
               + [f"adaptive_retry_all_at_{k}" for k in range(1, len(_D.BINS) + 1)]
               + [f"adaptive_retry_all_n_{k}" for k in range(1, len(_D.BINS) + 1)]
               + ["rank_mode", "rank_order", "rank_order_idx",
                  "rank_seq_requested", "rank_stable_frac", "rank_reorders",
                  "mean_attempts_indep", "solved_rate_indep",
                  "stop_on_success", "full_bin_coverage", "command_axes_src"]
               + [f"rank_pos_b{b}" for b in range(len(_D.BINS))]
               + [f"rank_post_b{b}" for b in range(len(_D.BINS))]
               + [f"rank_emp_b{b}" for b in range(len(_D.BINS))]
               + [f"rank_n_b{b}" for b in range(len(_D.BINS))])


def _glue_negative_values(argv=None) -> list:
    """`--bins -y,+x` -> `--bins=-y,+x`, so half the bin set is nameable.

    argparse takes any token starting with `-` as an option name unless it
    parses as a negative NUMBER, and `-y,+x` does not. So `--bins -y,+x` fails
    with "expected one argument" and `--only-bins -x,-z` with it — which is two
    of the six bins, including `-y`, one of the four live ones. Quoting does not
    help; only `--bins=-y,+x` does, and nobody reads far enough into a help text
    to learn that before hitting it.

    Rewriting the two affected flags here removes the trap without touching
    argparse's behaviour for anything else: only `--bins` and `--only-bins` are
    rewritten, only when the next token begins with `-`, and the `=` form
    already works and is left alone.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--bins", "--only-bins") and i + 1 < len(argv) \
                and argv[i + 1].startswith("-"):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


# EVERY CHAINED COLUMN IS `chain_`-PREFIXED, asserted rather than trusted.
# The row builder merges `chained` into the same dict as the independent
# metrics, so an unprefixed key that happens to match an independent column
# overwrites it with no error and no visible sign — which is exactly what
# `dir_err` and `dir_track` did before they were renamed. This turns a silent
# wrong number into an import-time failure.
_CHAINED_OWN = [c for c in TEST_FIELDS
                if c.startswith(("chain_", "chained_"))
                or c in ("solved_rate", "mean_attempts",
                         "mean_attempts_to_success", "signal_human_rate",
                         "mean_branch_step", "replay_err_mean")]
assert len(set(_CHAINED_OWN)) == len(_CHAINED_OWN), "duplicate chained column"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="output/dagger_runs/<name>")
    p.add_argument("--out", default=None,
                   help="default <split>_log.csv, so a train sweep cannot "
                        "silently overwrite a test one")
    p.add_argument("--iters", default="all",
                   help="'all' or a comma list, e.g. 0,5,10,15,20")
    p.add_argument("--split", default="test", choices=["test", "val", "train"],
                   help="handover-sim split to score on. `train` scores the FULL "
                        "training set — every scene the run could have collected "
                        "on, not the ~100-scene subsample the in-loop eval uses. "
                        "It is not a held-out number and must never be reported "
                        "as one; it answers 'how well did it learn what it was "
                        "taught', at a sample size where per-bin rates are "
                        "actually readable.")
    p.add_argument("--pin-table", default=None,
                   help="default output/regrasp_pins_<split>.json")
    p.add_argument("--exclude-scenes", default=None,
                   help="default output/regrasp_pins_<split>_excluded.json")
    p.add_argument("--demo-ok-table", default=None,
                   help="TRAIN ONLY, defaults to the run's own. Restricts scoring "
                        "to the (scene, bin) pairs base collection actually "
                        "demonstrated — without it a full-train sweep scores "
                        "directions the policy was never taught. Pass '' to "
                        "score every planned pair instead.")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="cap the number of scenes (default: all of them)")
    p.add_argument("--ckpt", default="last", choices=["best", "last"],
                   help="which checkpoint of each iteration to score (default last, "
                        "matching everything else in the run)")
    p.add_argument("--chained", action="store_true",
                   help="also run the retry LADDER, where attempt 2 starts from "
                        "where attempt 1 stopped rather than from home")

    # ---- which directions, in which order --------------------------------
    g = p.add_argument_group(
        "direction sequence",
        "Which bins are commanded and in what order. `--bins` is a PREFERENCE "
        "(unnamed live bins are appended behind the named ones, so nothing "
        "stops being evaluated); `--only-bins` is the filter.")
    g.add_argument("--bins", default=None,
                   help="the attempt order to start from, best first, e.g. "
                        "'+z,+x,-y,+y' or '4,0,3,2'. Short names, long names "
                        "and indices are all accepted, and a value starting "
                        "with '-' works with or without the '=' form. Default: "
                        "directions.RETRY_LADDER, which is what every run since "
                        "16 reduced retry_at_k over.")
    g.add_argument("--only-bins", default=None,
                   help="evaluate ONLY these bins. Unlike --bins this drops the "
                        "rest of the split entirely, so per-bin columns for the "
                        "excluded directions come back blank rather than "
                        "measured.")
    g.add_argument("--full-bin-coverage", action="store_true",
                   help="command every live bin on every scene, not only the "
                        "ones that scene demonstrates. This is the situation "
                        "the retry ladder actually creates — it commands a "
                        "direction BECAUSE the previous one failed, not because "
                        "a demo exists — and it is what `succ_bin_all_*` "
                        "measures against `succ_bin_*`. Needs "
                        "EVAL.success_mode: stable_grasp; `proximity` scores "
                        "distance to a pinned grasp and an off-table bin has "
                        "none.")
    g.add_argument("--adaptive", action="store_true",
                   help="re-rank the bins online from their running success as "
                        "the scenes are swept. See the module docstring: in the "
                        "INDEPENDENT sweep this changes the retry_at_k "
                        "reduction and nothing else; under --chained or "
                        "--stop-on-success it changes which rollouts happen.")
    g.add_argument("--rank-mode", default="ucb",
                   choices=["fixed", "mean", "ucb", "thompson"],
                   help="how the online ranker scores a bin (default ucb: "
                        "posterior mean + c standard deviations, so a bin tried "
                        "twice keeps an exploration bonus and one tried eighty "
                        "times is trusted at its mean). `mean` is greedy and "
                        "locks onto an early 1-for-1 bin; `fixed` holds --bins "
                        "and is the control condition; `thompson` is "
                        "randomised and needs --seed to reproduce.")
    g.add_argument("--rank-prior-strength", type=float, default=6.0,
                   help="how many EPISODES the passed --bins order is worth, as "
                        "Beta pseudo-counts (default 6)")
    g.add_argument("--rank-ucb-c", type=float, default=1.0,
                   help="--rank-mode ucb: posterior standard deviations added "
                        "to the mean")
    g.add_argument("--stop-on-success", action="store_true",
                   help="stop a scene at its first success in the INDEPENDENT "
                        "sweep too, instead of rolling every bin. Cheaper and "
                        "makes mean_attempts a real number, but the bins ranked "
                        "last are then sampled only on the hard scenes, so "
                        "per-bin rates STOP BEING COMPARABLE ACROSS BINS — read "
                        "rank_n_b* before quoting any of them.")

    # ---- which command vector ---------------------------------------------
    p.add_argument("--command-axes", default=None,
                   help="command_axes.json to condition on (default: the run's "
                        "own, beside config.yaml). These are the vectors the "
                        "policy was TRAINED to obey; deriving them from the "
                        "split's own pin table instead commands something the "
                        "run never issued — 2.9-6.0 deg off on the test split.")
    p.add_argument("--split-axes", action="store_true",
                   help="derive the command axes from the SPLIT's pin table "
                        "rather than the run's saved ones. Answers 'what would "
                        "this split's own geometry have commanded', which is a "
                        "real question and is not the deployment number.")
    p.add_argument("--allow-cfg-mismatch", action="store_true",
                   help="proceed even when the split's pin table was built "
                        "under a different SIM.cfg_file than the run — see the "
                        "guard in `main` for why that is normally fatal")
    p.add_argument("--rewind-frac", type=float, default=0.30,
                   help="--chained: how far back along the failed trajectory to "
                        "resume from, as a fraction of its length")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true", help="re-score iterations already in the CSV")
    p.add_argument("--plot-only", action="store_true",
                   help="skip evaluation, re-render the figures from the CSV")
    p.add_argument("--summary-iter", type=int, default=None,
                   help="which iteration <split>_summary.png shows (default: "
                        "the highest success_rate in the CSV). The run's own "
                        "`best/` was selected on the TRAIN subsample and is not "
                        "necessarily best here, which is why this defaults to a "
                        "re-selection rather than to that.")
    p.add_argument("--pos-thresh", type=float, default=0.02)
    p.add_argument("--rot-thresh", type=float, default=0.34)
    return p.parse_args(_glue_negative_values())


def read_done(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(r["iter"]): r for r in csv.DictReader(f) if r.get("iter")}


def write_log(path: Path, rows: dict) -> None:
    """Atomic via .tmp rename, so a plotter reading mid-write never sees half a
    file — the same discipline eval_regrasp_run.py uses."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEST_FIELDS)
        w.writeheader()
        for i in sorted(rows):
            w.writerow({k: rows[i].get(k, "") for k in TEST_FIELDS})
    tmp.replace(path)


def scene_bins(ctx, scene: int, *, allowed, full_coverage: bool) -> dict:
    """`{bin: slot}` for one scene — which directions can be commanded on it.

    A NON-NEGATIVE slot is a DEMONSTRATED (scene, bin) pair: the table holds a
    grasp realising that direction, so the pose diagnostics (`pos_err`,
    `box_chance`) are defined and the row counts toward `succ_bin_*`.

    A NEGATIVE slot is `off_table_slot(b)` — the bin commanded on a scene that
    demonstrates no grasp for it. `eval_one` decodes it, plans nothing, and
    scores `stable_grasp` on the rollout alone. Those rows carry real success and
    direction numbers with the pose columns NaN, and they are the population
    `succ_bin_all_*` is over. Only produced under `--full-bin-coverage`, because
    without it they would silently change what `success_rate` means.
    """
    n_here = int(ctx.pin_table.num_grasps_for(int(scene)) or 0)
    out = {}
    for gi in range(n_here):
        b = ctx.pin_table.bin_of(int(scene), gi)
        if b is None or int(b) not in allowed:
            continue
        out.setdefault(int(b), int(gi))
    if full_coverage:
        for b in allowed:
            out.setdefault(int(b), off_table_slot(int(b)))
    return out


def run_independent(ctx, runner, scenes, *, ranker, allowed, full_coverage,
                    stop_on_success, verbose=True):
    """The single-shot sweep, driven SCENE BY SCENE so the ranker can learn.

    Returns `(rows, ladders, attempts_per_scene, global_orders)`:
    `ladders[scene]` is the attempt order that stood BEFORE that scene was
    rolled out, restricted to the bins it can realise, and `global_orders` is the
    same ranking UNRESTRICTED, one entry per scene in sweep order — the first
    drives the causal `adaptive_retry_at_k` reduction, the second measures how
    much the ranking itself churned.

    WHY THIS REPLACES `evaluate_policy` RATHER THAN WRAPPING IT. That function
    takes the whole job list up front and reduces at the end, which is exactly
    the shape that cannot express "the order for scene i depends on scenes
    1..i-1". The episodes are identical — the same `eval_one` on the same
    `(scene, slot)` pairs — and they are handed to the same
    `aggregate_eval_rows`, so every column in the CSV keeps its meaning. What
    changes is only the ORDER they are run in and the fact that the ranker sees
    each outcome as it lands.

    ORDER-INVARIANCE, and the one place it breaks. With `stop_on_success` off,
    every feasible (scene, bin) pair is rolled out regardless of order, so the
    row SET is identical to what `evaluate_policy` would have produced and every
    aggregate is unchanged. With it on, a scene stops at its first success, so
    the bins ranked last are sampled only on scenes the earlier bins failed —
    their success rates are then conditioned on difficulty and are not
    comparable with the bins above them. That is a real bias, it is the price of
    the cheaper sweep, and `rank_n_b*` is what makes it visible.
    """
    rows, ladders, attempts, global_orders = [], {}, {}, []
    n_done = 0
    total = 0
    for scene in scenes:
        total += len(scene_bins(ctx, int(scene), allowed=allowed,
                                full_coverage=full_coverage))
    for scene in scenes:
        per = scene_bins(ctx, int(scene), allowed=allowed,
                         full_coverage=full_coverage)
        if not per:
            continue
        # THE ORDER IS READ BEFORE ANY EPISODE OF THIS SCENE RUNS. Reading it
        # after, or re-reading it between attempts, would let the scene's own
        # first outcome reorder its own remaining attempts — a leak small enough
        # to be invisible in the plot and large enough to make
        # `adaptive_retry_at_1` meaningless.
        order = ranker.order(feasible=per.keys())
        ladders[int(scene)] = list(order)
        # THE UNRESTRICTED ORDER, recorded separately. `order` above is the
        # ranking INTERSECTED with what this scene can realise, and most scenes
        # realise a different pair — so consecutive `order`s differ whenever the
        # feasible SET differs, whether or not the ranking moved. Measured on a
        # 60-scene synthetic sweep in `fixed` mode, where the ranking cannot move
        # by construction, that read 0.24 "stability". `rank_stable_frac` is a
        # statement about the RANKER, so it is computed from this instead.
        global_orders.append(tuple(ranker.order()))
        n_att = 0
        for b in order:
            row = eval_one(ctx.sim, runner, int(scene), per[b],
                           params=ctx.eval_params, pin_table=ctx.pin_table)
            rows.append(row)
            n_att += 1
            n_done += 1
            ranker.observe(int(b), bool(row.get("success")), episode=n_done)
            if verbose:
                print(f"    eval [{n_done:4d}/{total}] scene={scene:4d} "
                      f"{_D.BIN_SHORT[int(b)]:>2}"
                      f"{'*' if per[b] < 0 else ' '} "
                      f"success={int(bool(row.get('success')))} "
                      f"grasped={int(bool(row.get('grasped')))} "
                      f"close@{row.get('close_step')} {row.get('reason')}")
            if stop_on_success and row.get("success"):
                break
        attempts[int(scene)] = n_att
        if verbose and ranker.mode != "fixed" and n_done % 40 < n_att:
            print(f"      [rank] {ranker.describe()}")
    return rows, ladders, attempts, global_orders


def write_episodes(path: Path, rows, ladders, *, iteration: int,
                   d_rule=None) -> int:
    """One CSV row per EPISODE — the thing the aggregate cannot answer from.

    `<stem>_episodes.csv`, appended per iteration. The aggregate log carries
    per-bin fractions but not their cross-tabulation, so a question like "of the
    scenes no direction solved, what were the failures" — 8 scenes, 32 episodes
    on run 19's 143-scene sweep — could not be answered from what was saved, and
    the rows that could answer it were discarded with `m.pop("rows")`. 572 rows
    at ~15 fields is a few tens of kilobytes; there is no reason to throw them
    away.

    `dir_err` and `bin_realized` are derived here exactly as `_dir_block` derives
    them (the row's own `d_achieved` first, the rule's recomputation second,
    then `from_world` + `bin_of` in the row's anchor frame), so a per-episode
    value and the per-bin mean it contributes to cannot disagree.
    """
    from handover_sim2real.regrasp import directions as _D2
    cols = ["iter", "scene_idx", "bin", "bin_idx", "slot", "in_table",
            "ladder_pos", "success", "grasped", "closed", "reason",
            "close_step", "box_chance", "box_taken", "dir_err",
            "bin_realized", "pos_err", "rot_err"]
    new = not path.exists()
    n = 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            sc, b = int(r["scene_idx"]), int(r.get("bin_idx", -1))
            order = ladders.get(sc, [])
            de, rb = float("nan"), -1
            d_cmd, ee, c = r.get("d_world"), r.get("ee_final"), r.get("centroid_world")
            if d_cmd is not None and ee is not None:
                ach = r.get("d_achieved")
                if ach is None and d_rule is not None:
                    ach = d_rule.of(np.asarray(ee),
                                    None if c is None else np.asarray(c))
                if ach is not None and float(np.linalg.norm(ach)) >= _D2.D_ZERO_EPS:
                    de = float(_D2.angle_between(d_cmd, np.asarray(ach)))
                    R = r.get("anchor_R")
                    if R is not None:
                        rb = int(_D2.bin_of(_D2.from_world(np.asarray(ach),
                                                            np.asarray(R))))
            w.writerow({
                "iter": iteration, "scene_idx": sc,
                "bin": _D2.BIN_SHORT[b] if b >= 0 else "",
                "bin_idx": b, "slot": int(r.get("grasp_idx", -1)),
                "in_table": int(r.get("in_table", 1)),
                "ladder_pos": (order.index(b) if b in order else -1),
                "success": int(bool(r.get("success"))),
                "grasped": int(bool(r.get("grasped"))),
                "closed": int(bool(r.get("closed"))),
                "reason": r.get("reason", ""),
                "close_step": int(r.get("close_step", -1)),
                "box_chance": int(bool(r.get("box_chance"))),
                "box_taken": int(bool(r.get("box_taken"))),
                "dir_err": "" if de != de else round(de, 3),
                "bin_realized": rb,
                "pos_err": ("" if r.get("pos_err") is None
                            or r["pos_err"] != r["pos_err"]
                            else round(float(r["pos_err"]), 5)),
                "rot_err": ("" if r.get("rot_err") is None
                            or r["rot_err"] != r["rot_err"]
                            else round(float(r["rot_err"]), 4)),
            })
            n += 1
    return n


def adaptive_metrics(rows, ladders, attempts, *, ranker, stop_on_success,
                     global_orders=None):
    """`adaptive_retry_at_k` and friends — the ladder's own reduction.

    THE CAUSAL REDUCTION. For each scene, walk the order that stood before it
    was rolled out and ask whether any of the first k attempts succeeded. It is
    the same arithmetic `_regrasp_metrics` does over `RETRY_LADDER`, with the
    per-scene order substituted for the global constant, and it is honest
    precisely because that order was fixed before the outcomes existed.

    `rank_stable_frac` is how often consecutive scenes were handed the same
    UNRESTRICTED ranking. Near 1.0 means the ranker converged early and the
    adaptive number is effectively a fixed ladder that happens to differ from
    `RETRY_LADDER`; well below it means the order was still moving when the
    split ran out, which is a reason to distrust the ranking rather than the
    policy. It is deliberately NOT computed from the per-scene ladders: those are
    the ranking intersected with each scene's feasible bins, so they differ
    whenever the feasible SET differs and would read ~0.24 even in `fixed` mode,
    where the ranking cannot move at all.

    UNDER `stop_on_success` A SCENE HAS NO OUTCOME for the bins it never
    reached. Those are treated as "not attempted" rather than as failures: the
    scene stopped because it had already SUCCEEDED, so counting the unreached
    rungs as failures would push `adaptive_retry_at_k` DOWN on exactly the
    scenes that went best. `adaptive_retry_n_k` carries how many scenes actually
    offered k rungs.
    """
    # TWO POPULATIONS, AND MIXING THEM IS THE TRAP THIS GUARDS AGAINST.
    # `_regrasp_metrics` computes `retry_at_k` over IN-TABLE rows only, so it is
    # the 129 s0-test scenes that demonstrate at least one bin, and a scene
    # lacking the ladder's k-th rung contributes a miss. Under
    # `--full-bin-coverage` every scene carries all four bins, so an adaptive
    # reduction over ALL rows gives each scene four real rungs while the fixed
    # ladder still counts only demonstrated pairs — measured on the 143-scene
    # run, `retry_n_k` 94/77/50/40 against 143/143/143/143. Reading those two
    # curves against each other would credit the ranker with the coverage.
    #
    # So `adaptive_retry_at_k` is computed over the SAME in-table rows the fixed
    # ladder uses, and the all-scenes version is reported separately under its
    # own name. The comparison the figures draw is then like for like.
    def _by_scene(rs):
        out = {}
        for r in rs:
            b = int(r.get("bin_idx", -1))
            if b >= 0:
                out.setdefault(int(r["scene_idx"]), {})[b] = bool(r["success"])
        return out

    in_table = [r for r in rows if int(r.get("in_table", 1)) == 1]
    by_scene_all = _by_scene(rows)
    by_scene = _by_scene(in_table)
    n_scenes = max(len(by_scene), 1)
    out = {}
    for k in range(1, len(_D.BINS) + 1):
        hits = 0
        deep = 0
        for sc, per in by_scene.items():
            order = ladders.get(int(sc), [])
            rungs = [b for b in order[:k] if b in per]
            # A scene that stopped early offers fewer rungs than the ladder
            # names. It counts toward `hits` (it succeeded) but not toward the
            # k-deep denominator, which is the population where "k attempts"
            # was genuinely available.
            if len(rungs) == k:
                deep += 1
            if any(per[b] for b in rungs):
                hits += 1
        out[f"adaptive_retry_at_{k}"] = hits / n_scenes
        out[f"adaptive_retry_n_{k}"] = deep
        # ...and the same over EVERY evaluated scene, off-table rows included.
        # This is the deployment question — the ladder commands a direction
        # BECAUSE the last one failed, not because a demo exists — and it is not
        # comparable with `retry_at_k`, which is why it has its own name.
        na = max(len(by_scene_all), 1)
        ha = sum(1 for sc, per in by_scene_all.items()
                 if any(per[b] for b in ladders.get(int(sc), [])[:k] if b in per))
        out[f"adaptive_retry_all_at_{k}"] = ha / na
        out[f"adaptive_retry_all_n_{k}"] = na
    # OVER THE UNRESTRICTED ORDER, not the per-scene one — see `run_independent`.
    # Falls back to the per-scene ladders only when a caller did not supply it,
    # which cannot happen from `main` and keeps the signature usable in a test.
    seq = ([tuple(o) for o in global_orders] if global_orders is not None
           else [tuple(ladders[s]) for s in sorted(ladders)])
    same = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    out["rank_stable_frac"] = (same / (len(seq) - 1)) if len(seq) > 1 else 1.0
    out["rank_reorders"] = (len(seq) - 1 - same) if len(seq) > 1 else 0
    att = [attempts[s] for s in sorted(attempts)]
    out["mean_attempts_indep"] = float(np.mean(att)) if att else float("nan")
    out["solved_rate_indep"] = (sum(1 for per in by_scene.values()
                                    if any(per.values())) / n_scenes)
    out["stop_on_success"] = int(bool(stop_on_success))
    out.update(ranker.report())
    return out


def run_chained(ctx, runner, scenes, *, rewind_frac: float, max_attempts: int,
                ranker=None, allowed=None):
    """The retry ladder over the test scenes. Returns chained_metrics()'s dict.

    `ranker`, when given, supplies the attempt ORDER and is updated from every
    attempt's outcome — so the chain is the one sweep where the online ranking
    does real work rather than re-reducing finished episodes. A scene stops at
    its first success by construction here, so a good order genuinely costs
    fewer rollouts and `mean_attempts` is what moves.

    The ranker is SHARED with the independent sweep on purpose when both run: by
    the time the chain starts it already carries the whole split's per-bin
    evidence, so the chain is ordered by what the single-shot pass learned. That
    is a stronger prior than the chain could build on its own (it sees at most
    one or two bins per scene, and only on the scenes that failed), and it is
    still causal with respect to the chain's own outcomes.
    """
    from handover_sim2real.regrasp.chained_retry import (
        RetryParams, chained_metrics, chained_retry_scene,
    )

    per_scene = {}
    for n, scene in enumerate(scenes):
        n_here = ctx.pin_table.num_grasps_for(int(scene))
        # bin -> the pose realising it. The chain still needs a pose per attempt
        # (OMG's CLOSE label and the geometric scores are measured against one);
        # it is simply no longer what the policy is told.
        pose_of_bin = {}
        for gi in range(int(n_here)):
            b = ctx.pin_table.bin_of(int(scene), gi)
            if b is not None and (allowed is None or int(b) in allowed):
                pose_of_bin[int(b)] = ctx.pin_table.pose(int(scene), gi)
        if not pose_of_bin:
            continue
        meta = ctx.pin_table.scene_meta.get(int(scene), {})
        anchor_R = meta.get("anchor_R")
        prior = (ranker.order(feasible=pose_of_bin.keys())
                 if ranker is not None else None)
        atts = chained_retry_scene(
            ctx.sim, runner, int(scene), pose_of_bin,
            params=ctx.eval_params,
            retry=RetryParams(max_attempts=int(max_attempts),
                              rewind_frac=float(rewind_frac)),
            anchor_R=np.asarray(anchor_R) if anchor_R is not None else None,
            feasible=sorted(pose_of_bin), prior=prior)
        per_scene[int(scene)] = atts
        if ranker is not None:
            for a in atts:
                b = getattr(a, "bin_idx", None)
                if b is not None and int(b) >= 0:
                    ranker.observe(int(b), bool(getattr(a, "success", False)))
        if (n + 1) % 10 == 0:
            print(f"    [chain] {n + 1}/{len(scenes)} scenes", flush=True)
    return chained_metrics(per_scene, max_attempts=int(max_attempts))


def _report_iteration(m, adaptive, chained, args, *, eval_s, ranker, bins):
    """The per-iteration console block — every number the user asked to see.

    Printed as well as logged because a sweep is long enough that reading the
    CSV afterwards is a second session, and the whole point of the per-bin split
    is to notice a collapsed direction while there is still time to stop.

    THE FOUR HEADLINE LINES, in the order they should be read:

      handover   `success_rate` is the Phase-3 `stable_grasp` criterion — the
                 object left the human, ended in the jaws, and was still there
                 after the release. That is the handover, not a proxy for it.
      commit     `box_taken_rate` is "closed | in jaws": of the episodes that
                 ever had the object geometrically between the open fingers,
                 how many actually committed a close. `box_chance_rate` is the
                 denominator, so the pair separates "never got the chance" from
                 "got it and did not take it" — the same split
                 `_panel_opportunity` draws.
      direction  `bin_diag_rate` (ended in the commanded bin) and `dir_err`
                 (degrees off the commanded vector), per bin. `dir_track` is
                 `1 - dir_err/90` and is printed because every run since 2
                 reported it; the degrees are what to quote.
      ladder     retry@k under the FIXED `RETRY_LADDER` and, when --adaptive is
                 on, under the order the sweep learned. The gap between them is
                 what online ranking bought.
    """
    print(f"  HANDOVER success={m['success_rate']:.3f}   "
          f"grasp={m['grasp_rate']:.3f} close={m['close_rate']:.3f}   "
          f"({eval_s:.0f}s)")
    print(f"  COMMIT   in-jaws chance={m.get('box_chance_rate', float('nan')):.3f} "
          f"-> closed|in-jaws={m.get('box_taken_rate', float('nan')):.3f}   "
          f"(missed|in-jaws={m.get('miss_given_box', float('nan')):.3f})")
    print("  per-bin  "
          + "  |  ".join(
              f"{_D.BIN_SHORT[b]} n={int(m.get(f'n_b{b}', 0) or 0):3d} "
              f"succ={m.get(f'succ_bin_{b}', float('nan')):.2f} "
              f"in-bin={m.get(f'bin_diag_rate_b{b}', float('nan')):.2f} "
              f"err={m.get(f'dir_err_b{b}', float('nan')):5.1f}deg "
              f"track={m.get(f'dir_track_b{b}', float('nan')):.2f}"
              for b in bins))
    # The failure taxonomy over the episodes that came away with nothing. `ff_*`
    # stacks to 1.0 over the FAILURES, so it separates two bins failing at the
    # same rate for different reasons — which pooled `f_*` cannot.
    if m.get("n_fail"):
        from train_regrasp import EVAL_FAIL_REASONS, reason_columns
        frac = reason_columns(m.get("reasons_fail"), EVAL_FAIL_REASONS,
                              denom=m.get("n_fail"))
        print(f"  FAILURES (of {int(m['n_fail'])} failed episodes)  "
              + "  ".join(
                  f"{lbl} {frac.get(k, 0.0):.2f}"
                  for k, lbl in (("ff_timeout", "never closed"),
                                 ("ff_grasp_miss", "closed, not secured"),
                                 ("ff_drop", "drop"),
                                 ("ff_no_release", "no release"),
                                 ("ff_human_contact", "human contact"))))
    ladder = "/".join(f"{m.get(f'retry_at_{k}', float('nan')):.2f}"
                      for k in range(1, 5))
    print(f"  retry@1-4  fixed ladder "
          f"({'>'.join(_D.BIN_SHORT[b] for b in _D.RETRY_LADDER[:4])}): {ladder}")
    if args.adaptive:
        ad = "/".join(f"{adaptive.get(f'adaptive_retry_at_{k}', float('nan')):.2f}"
                      for k in range(1, 5))
        print(f"             ADAPTIVE ({adaptive.get('rank_order', '')}): {ad}"
              f"   stable={adaptive.get('rank_stable_frac', float('nan')):.2f} "
              f"reorders={adaptive.get('rank_reorders', 0)}")
        print(f"             posterior {ranker.describe()}")
    if args.stop_on_success:
        print(f"             mean attempts/scene="
              f"{adaptive.get('mean_attempts_indep', float('nan')):.2f}  "
              f"solved={adaptive.get('solved_rate_indep', float('nan')):.3f}")
    if chained:
        print("  CHAINED retry@k: "
              + "/".join(f"{chained.get(f'chained_retry_at_{k}', float('nan')):.2f}"
                         for k in range(1, args.max_attempts + 1)))


def _fig_stem(args) -> str:
    """The basename both figures are written under.

    DERIVED FROM `--out`, NOT FROM `--split`, and that is a bug fix. The figures
    used to be named `<split>_eval.png` / `<split>_summary.png` regardless, so a
    sweep writing `--out test144_log.csv` silently overwrote the figures of the
    earlier `test_log.csv` sweep — and the sbatch's `--plot-only` re-render then
    read `test_log.csv` (the file it was NOT given) and overwrote them a second
    time with the older data. Both happened on the 144-scene runs: the console
    said "wrote test_eval.png" while the CSV in hand was `test144_log.csv`.
    """
    out = getattr(args, "out", None)
    if not out:
        return str(args.split)
    stem = Path(out).name
    for suffix in ("_log.csv", ".csv"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def plot(run_root: Path, log_path: Path, args) -> None:
    """`<split>_eval.png` — the conditioning figure plus the per-bin diagnostics.

    LAYOUT. Two blocks, stacked:

      rows 0-1   the conditioning panels of `curves_regrasp.png`, drawn by
                 `plot_regrasp_run.draw_conditioning` — the same function the
                 training figure calls, so the two are comparable panel for
                 panel. On this figure the retry panel additionally carries the
                 CHAINED curves, which only this script produces, and the
                 adaptive ladder when --adaptive was used.
      rows 2..   one row per commanded bin, FOUR panels wide:
                   success stages        close -> near -> grasp -> success
                   chance vs conversion  did it get a chance, did it take it
                   approach error        how near the EE actually came
                   EVAL outcomes         HOW it failed, stacked to 1.0

    THE FOURTH COLUMN IS NEW AND IT IS THE LAST COLUMN OF `training_curve.png`.
    "It failed" and "how it failed" are the same question at two resolutions,
    and the test figure had only the first — so a bin whose success dropped gave
    no way to tell a policy that never closed from one that closed and dropped,
    which are opposite fixes. The COLLECTION twin of that column (`co_*`) is
    deliberately absent: nothing is collected on a held-out split, so those
    columns do not exist and drawing an empty frame would say "collection
    failed" rather than "there was none".

    The panels are IMPORTED, never reimplemented. A test figure that draws its
    rates even slightly differently from the training one is a figure you cannot
    hold up next to it, which is the only thing it is for.

    Every axis title is prefixed with the split, because the whole hazard of this
    figure is someone reading a full-TRAIN number as a held-out one.
    """
    import matplotlib.pyplot as plt
    import plot_regrasp_run as P

    if not log_path.exists():
        raise SystemExit(f"no {log_path} — run without --plot-only first")
    num, n = P._load(log_path)
    if n == 0:
        raise SystemExit(f"{log_path} has no rows yet")
    if n == 1:
        # Every panel on this figure is a curve against DAgger iteration, and
        # the stacked-area ones (`_panel_outcomes`) need TWO x points to render
        # at all — at n=1 they draw an invisible zero-width polygon under a full
        # legend, which reads as "nothing was measured". Say so once rather than
        # letting four blank panels imply it.
        print(f"[plot] {log_path.name} has ONE iteration, so every curve panel "
              f"is a single point and the stacked-area ones render blank. "
              f"{_fig_stem(args)}_summary.png is the figure to read.")
    it = num("iter")
    ctx = P._Ctx(num, it, args, "grasp")
    bins = P._bins_to_plot(num)
    TAG = f"{args.split.upper()}: "

    NCOL = 4
    nrow = 2 + max(len(bins), 1)
    fig, ax = plt.subplots(nrow, NCOL, figsize=(6.2 * NCOL, 3.7 * nrow),
                           squeeze=False)

    # ---- rows 0-1: the conditioning block, identical to curves_regrasp -----
    # `err` is the raw direction error in DEGREES, the panel `track` rescales
    # into an index. Both are drawn: the ratio is what runs 1-20 reported, the
    # degrees are the number to quote.
    P.draw_conditioning({"retry": ax[0][0], "ended": ax[0][1],
                         "side": ax[0][2], "err": ax[0][3],
                         "succ":  ax[1][0], "succ_all": ax[1][1],
                         "track": ax[1][2]},
                        ctx, num, it, bins, tag=TAG)
    _panel_ladder(ax[1][3], ctx, num, it, TAG)

    # ---- rows 2..: the per-bin diagnostics from training_curve.png ---------
    for r, b in enumerate(bins):
        sfx, name = f"_b{b}", P._bin_title(b)
        P._panel_nested(ax[2 + r][0], ctx, sfx,
                        title=f"{TAG}{name} — success stages")
        P._panel_opportunity(ax[2 + r][1], ctx, sfx,
                             title=f"{TAG}{name} — chance vs conversion")
        P._panel_approach(ax[2 + r][2], ctx, sfx,
                          title=f"{TAG}{name} — approach error to the grasp")
        # Legend on the first row only: a stacked area has no empty corner, so
        # four identical legends cover four panels' worth of data to say one
        # thing.
        P._panel_outcomes(ax[2 + r][3], ctx, sfx=sfx, legend=(r == 0),
                          title=f"{TAG}{name} — EVAL outcomes "
                                f"(policy alone, beta=0)")

    P._fix_x(fig, it)
    held = "HELD-OUT " if args.split != "train" else "FULL "
    fig.suptitle(f"Regrasp on the {held}{args.split} split — {run_root.name}"
                 f"   [{int(num('num_scenes')[-1]) if P._finite(num('num_scenes')) else '?'} scenes]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.03 / nrow * 2])
    out = run_root / f"{_fig_stem(args)}_eval.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    plot_summary(run_root, num, n, args, bins)


def _panel_ladder(a, ctx, num, it, tag=""):
    """retry@k under the FIXED ladder vs the order the sweep LEARNED.

    Only this script can draw it: `retry_at_k` is reduced over
    `directions.RETRY_LADDER`, a constant chosen from run 11's per-bin success,
    while `adaptive_retry_at_k` is reduced over the order an online ranker
    arrived at on this split with this checkpoint. Same episodes, two orders —
    so the VERTICAL GAP is exactly what ranking the directions by their observed
    success is worth, with the policy held fixed.

    `rank_stable_frac` rides on the right axis because it is the panel's own
    caveat: near 1.0 the ranker settled early and the adaptive curve is a
    different fixed ladder; well below it the order was still moving when the
    split ran out, and the adaptive number is then a statement about 129 scenes
    rather than about the policy.
    """
    import plot_regrasp_run as P

    drew = False
    for k, col in zip((1, 2, 3, 4),
                      ("tab:blue", "tab:green", "tab:orange", "tab:red")):
        ys = num(f"retry_at_{k}")
        if P._finite(ys):
            P._plot(a, it, ys, "-", marker="o", ms=3, lw=1.2 + 0.2 * k,
                    color=col, alpha=0.55, label=f"fixed ladder @ {k}")
            drew = True
        ya = num(f"adaptive_retry_at_{k}")
        if P._finite(ya):
            P._plot(a, it, ya, "-", marker="D", ms=3.5, lw=1.2 + 0.2 * k,
                    color=col, label=f"ADAPTIVE @ {k}")
            drew = True
    if drew:
        a.set_ylim(-0.02, 1.02)
    P._note_empty(a, "no adaptive ladder in this log\n(run with --adaptive)")
    P._grid(a, f"{tag}retry@k: fixed ladder (o, faded) vs learned order (D)",
            ylabel="fraction of scenes solved")
    st = num("rank_stable_frac")
    if P._finite(st):
        a2 = a.twinx()
        P._plot(a2, it, st, ":", marker="s", ms=3, color="0.35",
                label="ladder stability")
        a2.set_ylim(-0.02, 1.02)
        a2.set_ylabel("consecutive scenes with the same order", fontsize=7,
                      color="0.35")
        a2.tick_params(axis="y", labelcolor="0.35", labelsize=7)
    P._legend(a, loc="lower right", ncol=2)


def plot_summary(run_root: Path, num, n, args, bins) -> None:
    """`<split>_summary.png` — the per-bin bars for ONE iteration.

    WHY A SECOND FIGURE. Everything above is a curve against DAgger iteration,
    which is the right shape for a training run and the wrong one for a test
    report: a test sweep is often a single checkpoint, where a line plot of one
    point conveys nothing, and even over ten iterations the question being asked
    is "what does the finished policy do", not "how did it get there". This is
    the answer to that question, and it is the figure to put in a document.

    THE ITERATION SHOWN is the one with the highest `success_rate` in the CSV,
    with its number in every title — not the last row, because a sweep scored at
    `--iters 0,5,10` has no meaningful "last", and not a hardcoded best-of,
    because the run's own `best/` was selected on the TRAIN subsample and need
    not be the best here. `--summary-iter` overrides it.

    FOUR PANELS, all per commanded bin, all with the episode count on the bar so
    a rate over eleven episodes cannot be read as one over a hundred:

      success      the handover (`stable_grasp`), demonstrated scenes and — when
                   --full-bin-coverage ran — every scene, side by side. The gap
                   between the two bars is the generalisation-to-an-undemonstrated
                   -direction cost, which is the number that decides whether `k`
                   is really a test-time knob.
      commit       in-jaws chance -> closed | in jaws. Two bars, because the
                   conversion alone cannot distinguish a policy that never got
                   the object between its fingers from one that got it and
                   froze.
      direction    ended in the commanded bin, and arrived from the commanded
                   side, against the 1/len(live bins) chance line.
      dir_err      the RAW error in degrees with the 30-deg bin_hit margin and
                   the 45-deg "nearer another bin" line drawn across it.
      outcomes     the eval failure taxonomy, stacked to 1.0 per bin.

    THE FIFTH PANEL IS HERE BECAUSE THE CURVE FIGURE CANNOT SHOW IT. The outcome
    taxonomy is drawn there as a stacked AREA, which needs at least two x points
    — and a test sweep is routinely ONE checkpoint, where every area panel
    renders as an invisible zero-width polygon with a legend over it. Measured on
    run 9's single-iteration test log: the `f_*` columns are all present and
    populated, and all four outcome panels came out blank. A stacked BAR asks the
    identical question at n=1, so the taxonomy survives the case the figure is
    most often used in.
    """
    import matplotlib.pyplot as plt
    import plot_regrasp_run as P

    succ = num("success_rate")
    it = num("iter")
    idx = None
    want_it = getattr(args, "summary_iter", None)
    if want_it is not None:
        cand = [i for i, v in enumerate(it) if v == want_it]
        if not cand:
            raise SystemExit(f"--summary-iter {want_it} is not in the CSV "
                             f"(have {[int(v) for v in it if v == v]})")
        idx = cand[0]
    else:
        fin = [(v, i) for i, v in enumerate(succ) if v == v]
        idx = max(fin)[1] if fin else n - 1
    ITER = int(it[idx]) if it[idx] == it[idx] else -1

    def at(key, default=float("nan")):
        ys = num(key)
        v = ys[idx] if idx < len(ys) else default
        return v if v == v else default

    SH = [_D.BIN_SHORT[b] for b in bins]
    x = np.arange(len(bins), dtype=float)
    w = 0.38
    NP = 5
    fig, ax = plt.subplots(1, NP, figsize=(5.8 * NP, 4.8), squeeze=False)
    ax = ax[0]
    COL = [P._BIN_COLOURS_BY_BIN[b] for b in bins]

    def _count_labels(a, xs, ys, ns):
        for xi, yi, ni in zip(xs, ys, ns):
            if yi != yi:
                continue
            a.text(xi, min(yi + 0.03, 1.02), f"n={int(ni)}", ha="center",
                   va="bottom", fontsize=7, color="0.35")

    # ---- 1. the handover ---------------------------------------------------
    demo = [at(f"succ_bin_{b}") for b in bins]
    alls = [at(f"succ_bin_all_{b}") for b in bins]
    n_demo = [at(f"n_bin_{b}", 0) for b in bins]
    # DRAW THE SECOND SERIES ONLY IF IT DIFFERS. Without
    # `--full-bin-coverage` there are no off-table rows, so `succ_bin_all_b`
    # is computed over the same episodes as `succ_bin_b` and is equal to it to
    # every digit — measured on run 9's test log, all four bins. A hatched bar
    # of identical height beside each solid one says nothing and reads as a
    # second measurement, which is worse than saying nothing.
    has_all = (any(v == v for v in alls)
               and any(abs(a_ - d) > 1e-9
                       for a_, d in zip(alls, demo) if a_ == a_ and d == d))
    a = ax[0]
    a.bar(x - (w / 2 if has_all else 0), demo, w if has_all else 0.6,
          color=COL, label="scenes that DEMO the bin")
    if has_all:
        a.bar(x + w / 2, alls, w, color=COL, alpha=0.45, hatch="//",
              label="ALL scenes (off-table commands included)")
    _count_labels(a, x - (w / 2 if has_all else 0), demo, n_demo)
    a.set_ylim(0, 1.05)
    a.set_title("HANDOVER success per commanded direction\n"
                "(stable_grasp — released, held, not dropped)", fontsize=10)
    a.set_ylabel("fraction of that bin's episodes")
    a.legend(fontsize=7, loc="upper right")

    # ---- 2. the commit -----------------------------------------------------
    a = ax[1]
    chance = [at(f"box_chance_rate_b{b}") for b in bins]
    taken = [at(f"box_taken_rate_b{b}") for b in bins]
    a.bar(x - w / 2, chance, w, color="tab:blue", label="object in jaws (chance)")
    a.bar(x + w / 2, taken, w, color="tab:green", label="closed | in jaws (commit)")
    a.set_ylim(0, 1.05)
    a.set_title("GRASP COMMIT: did it get the chance, did it take it\n"
                "(the pair separates 'never in the jaws' from 'froze')",
                fontsize=10)
    a.legend(fontsize=7, loc="upper right")

    # ---- 3. did it go where it was told ------------------------------------
    a = ax[2]
    ended = [at(f"bin_diag_rate_b{b}") for b in bins]
    side = [at(f"bin_hit_rate_b{b}") for b in bins]
    a.bar(x - w / 2, ended, w, color="tab:purple",
          label="ended in the COMMANDED bin")
    a.bar(x + w / 2, side, w, color="tab:olive",
          label="arrived from the COMMANDED side")
    # The chance line is over the bins actually COMMANDED here, not a hardcoded
    # 1/4 or 1/6 — those differ per run (run 21's table has five live bins) and
    # a wrong line misleads in both directions. It is still only a rough floor:
    # the live bins are clustered, so a shuffled command scores above 1/k.
    a.axhline(1.0 / max(len(bins), 1), color="0.4", ls="--", lw=1.1,
              label=f"1/{len(bins)} (uniform guess over the commanded bins)")
    a.set_ylim(0, 1.05)
    a.set_title("DIRECTION: ended in the commanded bin\n"
                "(orientation) and arrived from its side (position)",
                fontsize=10)
    a.legend(fontsize=7, loc="upper right")

    # ---- 4. the raw error --------------------------------------------------
    a = ax[3]
    err = [at(f"dir_err_b{b}") for b in bins]
    a.bar(x, err, 0.6, color=COL)
    for xi, yi in zip(x, err):
        if yi == yi:
            a.text(xi, yi + 0.8, f"{yi:.1f}", ha="center", va="bottom",
                   fontsize=8)
    a.axhline(_D.BIN_HIT_DEG, color="0.45", ls=":", lw=1.2,
              label=f"{_D.BIN_HIT_DEG:.0f} deg (bin_hit margin)")
    a.axhline(45.0, color="tab:red", ls="--", lw=1.2,
              label="45 deg (nearer another bin)")
    top = max([v for v in err if v == v] + [50.0])
    a.set_ylim(0, top * 1.25)
    a.set_title("RAW direction error (deg, lower is better)\n"
                "angle between the commanded d and the achieved approach axis",
                fontsize=10)
    a.set_ylabel("degrees")
    a.legend(fontsize=7, loc="upper right")

    # ---- 5. HOW it failed --------------------------------------------------
    # `f_*` are fractions of that bin's eval episodes and stack to 1.0, so the
    # green band at the bottom IS `succ_bin_all_b` and the bands above it
    # partition the rest. Same taxonomy, same colours and same order as
    # `plot_regrasp_run._OUTCOMES`, so this bar and the training figure's area
    # can be read against each other without translating.
    a = ax[4]
    bottom = np.zeros(len(bins))
    drew = False
    for key, lbl, col in P._OUTCOMES:
        vals = np.array([at(f"{key}_b{b}", float("nan")) for b in bins])
        if not np.any(np.isfinite(vals)):
            continue
        v = np.where(np.isfinite(vals), vals, 0.0)
        a.bar(x, v, 0.6, bottom=bottom, color=col, label=lbl)
        bottom += v
        drew = True
    if drew:
        a.set_ylim(0, 1.05)
    else:
        a.text(0.5, 0.5, "no outcome taxonomy in this log\n"
                         "(needs the f_* columns; added after run 8)",
               ha="center", va="center", transform=a.transAxes,
               fontsize=9, color="0.45")
    a.set_title("EVAL OUTCOMES per commanded direction\n"
                "(policy alone, beta=0; stacks to 1.0)", fontsize=10)
    a.set_ylabel("fraction of that bin's episodes")
    a.legend(fontsize=7, loc="upper right", ncol=2)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(SH)
        a.grid(alpha=0.3, axis="y")
        a.set_xlabel("commanded direction")

    held = "HELD-OUT " if args.split != "train" else "FULL "
    # `_load` coerces every column to float, so the ladder — a string like
    # `+z|+x|-y|+y` — comes back NaN through `num` and has to be read from the
    # CSV directly.
    ladder = ""
    order_txt = _csv_cell(run_root, args, ITER, "rank_order")
    if order_txt:
        ladder = f"   [learned ladder: {order_txt}]"
    fig.suptitle(
        f"Regrasp on the {held}{args.split} split — {run_root.name} "
        f"@ iteration {ITER}   "
        f"[{int(at('num_scenes', 0))} scenes, {int(at('num_episodes', 0))} "
        f"episodes]{ladder}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = run_root / f"{_fig_stem(args)}_summary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _csv_cell(run_root: Path, args, iteration: int, col: str) -> str:
    """One string cell out of the log. `plot_regrasp_run._load` coerces every
    column to float, which turns `+z|+x|-y` into NaN — fine for every other
    column and useless for the one that names the ladder."""
    path = run_root / (args.out or f"{args.split}_log.csv")
    if not path.exists():
        return ""
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("iter") and int(r["iter"]) == int(iteration):
                return str(r.get(col, "") or "")
    return ""


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_dir)
    # `<split>_log.csv` / `<split>_eval.png`. The earlier names were
    # `<split>_eval_log.csv` / `<split>_set_evaluation.png`; an existing CSV
    # under the old name is ADOPTED rather than ignored, so a part-finished
    # sweep is not re-run from scratch after the rename.
    log_path = run_root / (args.out or f"{args.split}_log.csv")
    if args.out is None and not log_path.exists():
        legacy = run_root / f"{args.split}_eval_log.csv"
        if legacy.exists():
            print(f"[compat] adopting {legacy.name} -> {log_path.name}")
            log_path = legacy
    if args.plot_only:
        plot(run_root, log_path, args)
        return

    cfg_path = run_root / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config.yaml in {run_root} — is that a Regrasp run dir?")
    with cfg_path.open() as f:
        cfg4 = expand_config_paths(yaml.safe_load(f))

    # ---- repoint the run's own config at the test split ---------------------
    # Everything else — the cameras, the standoff, the success criterion, the
    # box geometry, the close thresholds — comes from the run's saved config, so
    # the test number differs from the training-curve number in the DATA and in
    # nothing else. Reconstructing an EvalParams by hand here is how the two
    # quietly stop being comparable.
    sim = cfg4["SIM"]
    sim["split"] = args.split
    sim["grasp_pin_table"] = (args.pin_table
                              or f"output/regrasp_pins_{args.split}.json")
    sim["exclude_scenes"] = (args.exclude_scenes
                             or f"output/regrasp_pins_{args.split}_excluded.json")
    # THE demo_ok_table IS SPLIT-DEPENDENT, and getting it wrong is silent.
    #
    # On test/val nothing was ever collected, so there is no collection outcome
    # to filter by and the pin table alone states feasibility — it was built by
    # calling the planner on every scene. Filtering test by the TRAIN split's
    # collection failures would score on a set defined by an unrelated run.
    #
    # On train the opposite holds. The run itself trained with the table applied,
    # so the (scene, bin) pairs OMG could not demonstrate are pairs the policy
    # was never taught; scoring them would charge the policy for directions
    # absent from its data and make the full-train number incomparable with the
    # in-loop curve it is supposed to extend.
    if args.split == "train":
        if args.demo_ok_table is not None:
            if args.demo_ok_table:
                sim["demo_ok_table"] = args.demo_ok_table
            else:
                sim.pop("demo_ok_table", None)
        # else: whatever the run's own config carried, which is the right default
    else:
        sim.pop("demo_ok_table", None)

    # `reach_filter` IS SPLIT-DEPENDENT FOR THE SAME REASON, and it is the other
    # half of the same prune: `demo_ok_table` asks whether the base caption was
    # honest, this asks whether the base demonstration reached its grasp. Both
    # name pairs in the TRAIN pin table, so on test/val -- where nothing was
    # collected -- neither has anything to say and applying either would score
    # the split against an unrelated run's collection outcome.
    #
    # It must be popped EXPLICITLY: build_regrasp_context defaults it ON, and
    # would otherwise try to prune the TEST table with TRAIN reach pairs (or
    # refuse outright when the base shard is not reachable from here).
    if args.split != "train":
        sim["reach_filter"] = False

    for path_key in ("grasp_pin_table", "exclude_scenes"):
        if not Path(sim[path_key]).exists():
            raise SystemExit(
                f"{sim[path_key]} not found. Build the {args.split}-split tables "
                f"first:\n"
                f"  python examples/build_direction_table.py --split {args.split} "
                f"--out output/direction_table_{args.split}.json\n"
                f"  python examples/assign_direction_demos.py "
                f"--table output/direction_table_{args.split}.json "
                f"--out output/regrasp_pins_{args.split} "
                f"--drop-bins='-z_beneath,-x_over_fingers'")

    ev = cfg4.setdefault("EVAL", {})
    # Every test scene, not an np.linspace subsample: the split is already small
    # and already held out, so there is nothing to hold back from it. `holdout`
    # is meaningless here for the same reason — nothing collects on test.
    ev["num_scenes"] = int(args.num_scenes) if args.num_scenes else 10 ** 6
    ev["holdout"] = False
    seed = int(args.seed if args.seed is not None
               else cfg4.get("DAGGER", {}).get("seed", 0))

    ctx = build_regrasp_context(cfg4, seed=seed)
    n_ep = len(ctx.pin_table.pairs(ctx.eval_scenes))

    # ---- GUARD: was the split's table built under the run's camera rig? -----
    # The pin table stores `anchor_R` and `centroid_world` per scene, and both
    # are derived from the OBSERVED point cloud — so they are a function of which
    # cameras were on. `regrasp_pins_test.json` was built under
    # `pretrain_multicam_wr.yaml` (wrist + right); run 21 trained under
    # `pretrain_right.yaml` (right only). Scoring run 21 against that table
    # anchors every bin in a frame computed from a cloud the policy never sees,
    # which is the same class of silent mismatch `resolve_anchor_ref` refuses one
    # axis over — and `preflight_regrasp.sh` check 4 exists because it already
    # cost a run once.
    #
    # Nothing crashes if you ignore it; the rates simply stop meaning what they
    # say. Hence a refusal with the build command rather than a warning.
    tbl_cfg = str((getattr(ctx.pin_table, "meta", {}) or {}).get("cfg_file", ""))
    run_cfg = str(sim.get("cfg_file", ""))
    if tbl_cfg and run_cfg and tbl_cfg != run_cfg and not args.allow_cfg_mismatch:
        stem = f"output/regrasp_pins_{args.split}_<suffix>"
        raise SystemExit(
            f"[cfg] {sim['grasp_pin_table']} was built under SIM.cfg_file "
            f"{tbl_cfg!r}, but {run_root.name} trained under {run_cfg!r}.\n"
            f"The table's per-scene `anchor_R` and `centroid_world` come from "
            f"the OBSERVED cloud, so a different camera set names the bins in a "
            f"different frame and every per-bin rate below is measured against "
            f"a command this policy was never given.\n"
            f"Build a matching table (mirror the flags the run's own TRAIN "
            f"table was built with — `_meta` in {sim['grasp_pin_table']} and in "
            f"the run's train table records them):\n"
            f"  python examples/build_direction_table.py --split {args.split} "
            f"--cfg-file {run_cfg} "
            f"--anchor-hand-ref {sim.get('anchor_hand_ref', 'wrist')} "
            f"--d-rule {sim.get('d_rule', 'approach_axis')} "
            f"--out output/direction_table_{args.split}_<suffix>.json\n"
            f"  python examples/assign_direction_demos.py "
            f"--table output/direction_table_{args.split}_<suffix>.json "
            f"--out {stem} --drop-bins=<same as the train table's "
            f"_meta.dropped_bins>\n"
            f"then pass --pin-table {stem}.json --exclude-scenes "
            f"{stem}_excluded.json.\n"
            f"Or --allow-cfg-mismatch to proceed anyway and own the caveat.")

    # ---- THE COMMAND VECTOR COMES FROM THE RUN, NOT FROM THE SPLIT ----------
    # See the module docstring. `build_regrasp_context` has just resolved
    # `command_axes` from THIS split's pin table, which under
    # `command_deploy: bin_centroid` means the test split's own per-bin
    # centroids — 2.9 deg off run 19's `+x` and 6.0 deg off its `+z`. The policy
    # was trained to obey the run's vectors, so those are what it gets.
    axes_src = "split pin table"
    if not args.split_axes:
        axes_path = Path(args.command_axes or (run_root / "command_axes.json"))
        if not axes_path.exists():
            if args.command_axes:
                raise SystemExit(f"no command_axes.json at {axes_path}")
            print(f"[command] no {axes_path} — falling back to the split's own "
                  f"centroids. Runs 9 and 19+ all save one; a run without it "
                  f"predates the file and used SIM.command_deploy: bin_axis.")
        else:
            with axes_path.open() as f:
                saved = json.load(f)
            saved_mode = str(saved.get("mode", ""))
            cfg_mode = str(sim.get("command_deploy", "bin_axis"))
            if saved_mode and saved_mode != cfg_mode:
                raise SystemExit(
                    f"[command] {axes_path} was written under "
                    f"SIM.command_deploy: {saved_mode!r} but the run's config "
                    f"says {cfg_mode!r}. One of the two is stale and guessing "
                    f"which would silently pick the command vector.")
            new_axes = np.asarray(saved["axes"], dtype=np.float64)
            old_axes = np.asarray(ctx.eval_params.command_axes,
                                  dtype=np.float64)
            off = _D.angle_between(new_axes, old_axes)
            ctx.eval_params.command_axes = new_axes
            axes_src = str(axes_path)
            # "offset from what build_regrasp_context resolved", which is the
            # split's own centroids under `bin_centroid` and the raw octahedral
            # axes under `bin_axis`. Naming it "the split's centroids"
            # unconditionally would mislabel the second case, where the number
            # printed is the run's well-known offset-from-axis (run 19: +x 8.4,
            # +y 16.3) rather than anything about the split.
            was = ("the split's own centroids" if cfg_mode == "bin_centroid"
                   else "the raw bin axes")
            print(f"[command] conditioning on {axes_path} "
                  f"({saved_mode or 'bin_axis'}); offset from {was} (deg): "
                  + "  ".join(f"{_D.BIN_SHORT[b]} {off[b]:.1f}"
                              for b in range(len(_D.BINS))
                              if np.isfinite(off[b])))

    # ---- which bins, and in what order --------------------------------------
    # The live set is whatever the split's table can actually realise. Asking for
    # a bin it cannot is not an error — `--bins` is a preference and the
    # unrealisable entries simply never come up — but `--only-bins` narrowing to
    # an empty set is, because it would score nothing and report zeros.
    table_bins = sorted({int(b)
                         for sc in ctx.pin_table.entries
                         for gi in range(int(ctx.pin_table.num_grasps_for(int(sc)) or 0))
                         if (b := ctx.pin_table.bin_of(int(sc), gi)) is not None})
    if args.full_bin_coverage:
        if str(ctx.eval_params.success_mode) != "stable_grasp":
            raise SystemExit(
                f"--full-bin-coverage commands bins a scene never demonstrated, "
                f"so there is no pinned grasp to measure against, and this run's "
                f"EVAL.success_mode is {ctx.eval_params.success_mode!r}. Only "
                f"`stable_grasp` scores a rollout without one.")
        table_bins = sorted(set(table_bins) | set(_D.LIVE_BINS))
    allowed = set(table_bins)
    if args.only_bins:
        want_only = set(parse_bin_sequence(args.only_bins))
        allowed &= want_only
        if not allowed:
            raise SystemExit(
                f"--only-bins {args.only_bins!r} leaves nothing: this table "
                f"realises {[_D.BIN_SHORT[b] for b in table_bins]}.")
    # The ranker itself is built PER ITERATION inside the loop, not here: its
    # posterior summarises one checkpoint's per-bin behaviour, and carrying one
    # across iterations would seed iteration 25's ladder with iteration 5's
    # evidence. `seq` is the shared starting order; everything else is per-run.
    seq = parse_bin_sequence(args.bins, live=allowed)

    held_out = args.split != "train"
    print("=" * 78)
    print(f"Regrasp {'HELD-OUT' if held_out else 'FULL-TRAIN'} eval   "
          f"run={run_root.name}")
    print(f"  split       : {args.split}   "
          + ("(held out by the benchmark, never collected on)" if held_out else
             "(COLLECTED ON — this is a train-set number, not a generalisation "
             "one)"))
    print(f"  scenes      : {len(ctx.eval_scenes)} -> {n_ep} independent episodes "
          f"per iteration")
    print(f"  checkpoint  : {args.ckpt}     success={ctx.eval_params.success_mode}")
    print(f"  chained     : {'ON' if args.chained else 'off'}"
          + (f"   rewind {args.rewind_frac:.0%}, "
             f"max {args.max_attempts} attempts" if args.chained else ""))
    print(f"  command     : {axes_src}")
    print("  bins        : "
          + " > ".join(_D.BIN_SHORT[b] for b in seq)
          + (f"   (only {sorted(_D.BIN_SHORT[b] for b in allowed)})"
             if args.only_bins else "")
          + ("   [+ off-table: every live bin on every scene]"
             if args.full_bin_coverage else ""))
    print("  ordering    : "
          + (f"ADAPTIVE {args.rank_mode} "
             f"(prior {args.rank_prior_strength:g} episodes"
             + (f", c={args.rank_ucb_c:g}" if args.rank_mode == "ucb" else "")
             + ")" if args.adaptive
             else "fixed — the passed sequence, held")
          + ("   [stop at first success: per-bin rates become "
             "difficulty-conditioned]" if args.stop_on_success else ""))
    print(f"  pinning     : {ctx.pin_table.describe()}")
    print(f"  writing     : {log_path}")
    print("=" * 78)

    want = None if args.iters == "all" else {int(x) for x in args.iters.split(",")}
    # `done` rather than `rows`: the per-iteration loop below now owns a
    # `rows` of its own (the EPISODE rows the adaptive sweep produces), and
    # the two shadowing each other is exactly the kind of silent aliasing
    # that writes one iteration's episodes into the CSV as if they were
    # iterations.
    done = {} if args.force else read_done(log_path)

    todo = [(i, d) for i, d in iteration_dirs(run_root)
            if (want is None or i in want) and i not in done]
    if not todo:
        print("nothing to score (use --force to re-score)")
    for i, run_dir in todo:
        t0 = time.time()
        print(f"\n[iter {i:02d}] {run_dir}")
        try:
            runner, _ = load_policy_runner(run_dir, args.device, ckpt=args.ckpt)
        except Exception as e:                              # noqa: BLE001
            print(f"  [skip] cannot load {args.ckpt}.pt: {type(e).__name__}: {e}")
            continue
        # A FRESH RANKER PER ITERATION. The posterior summarises one
        # checkpoint's per-bin behaviour, and iteration 25's ladder must not be
        # seeded by iteration 5's — the whole claim of `adaptive_retry_at_k` is
        # that it is what THIS policy would have learned on the split.
        it_ranker = BinRanker(
            prior_order=tuple(seq),
            mode=("fixed" if not args.adaptive else args.rank_mode),
            prior_strength=float(args.rank_prior_strength),
            ucb_c=float(args.rank_ucb_c), seed=seed)
        rows, ladders, attempts, gorders = run_independent(
            ctx, runner, ctx.eval_scenes, ranker=it_ranker, allowed=allowed,
            full_coverage=args.full_bin_coverage,
            stop_on_success=args.stop_on_success,
            verbose=bool(ctx.eval_params.verbose))
        m = aggregate_eval_rows(rows, ctx.eval_params,
                                eval_num_grasps(ctx.pin_table))
        m.pop("rows", None)
        n_ep = write_episodes(
            run_root / f"{_fig_stem(args)}_episodes.csv", rows, ladders,
            iteration=i, d_rule=getattr(ctx.eval_params, "d_rule", None))
        print(f"  [episodes] appended {n_ep} rows to "
              f"{_fig_stem(args)}_episodes.csv")
        adaptive = adaptive_metrics(rows, ladders, attempts, ranker=it_ranker,
                                    stop_on_success=args.stop_on_success,
                                    global_orders=gorders)
        chained = {}
        if args.chained:
            chained = run_chained(ctx, runner, ctx.eval_scenes,
                                  rewind_frac=args.rewind_frac,
                                  max_attempts=args.max_attempts,
                                  ranker=it_ranker if args.adaptive else None,
                                  allowed=allowed)
        del runner
        if args.device != "cpu":
            import torch
            torch.cuda.empty_cache()
        eval_s = time.time() - t0

        row = {k: "" for k in TEST_FIELDS}
        row.update({"iter": i, "run_dir": str(run_dir), "ckpt": args.ckpt,
                    "split": args.split, "num_scenes": len(ctx.eval_scenes),
                    "num_episodes": int(m.get("n", 0)),
                    "eval_s": round(eval_s, 1),
                    "rank_seq_requested":
                        "|".join(_D.BIN_SHORT[b] for b in seq),
                    "full_bin_coverage": int(bool(args.full_bin_coverage)),
                    "command_axes_src": axes_src})
        row.update({k: v for k, v in eval_columns(m).items() if k in row})
        # Strings ride through unchanged; numbers are rounded and a non-finite
        # one BLANKS rather than writing `nan`, matching `_r` in the trainer so
        # an unvisited bin reads as absent rather than as a measured zero.
        for src in (adaptive, chained):
            for k, v in src.items():
                if k not in row:
                    continue
                if isinstance(v, (int, float, np.floating, np.integer)):
                    row[k] = ("" if not np.isfinite(float(v))
                              else round(float(v), 4))
                else:
                    row[k] = v
        done[i] = row
        write_log(log_path, done)

        _report_iteration(m, adaptive, chained, args, eval_s=eval_s,
                          ranker=it_ranker, bins=sorted(allowed))

    if done:
        print(f"\nwrote {log_path}  ({len(done)} iterations)")
        plot(run_root, log_path, args)
    else:
        print("\nnothing scored — no completed iterations found in state.json")


if __name__ == "__main__":
    main()
