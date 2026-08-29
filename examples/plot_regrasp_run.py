"""
Plot a Regrasp DAgger run from its dagger_log.csv.

    python examples/plot_regrasp_run.py output/dagger_runs/regrasp_run1
    python examples/plot_regrasp_run.py output/dagger_runs/regrasp_run1 --show

Safe to run WHILE the loop is going — it only reads the CSV, which is appended
and flushed once per iteration. If the run scored its iterations in a separate
job (`EVAL.every: 0`), <run>/eval_log.csv is spliced in by iteration, so the
figures look the same either way.

Five figures, and the split between them is by QUESTION, not by convenience.

TRAINING CURVE (<run>/training_curve.png) — 4x5, "did it learn, and for which
direction". One ROW per commanded bin, five columns:

  • success stages   close -> near -> grasp -> success. Nested (close >= near,
                     close >= grasp >= success), so the VERTICAL GAPS are the
                     diagnosis: close-near is closing in the wrong place,
                     near-grasp is closing right but not gripping, grasp-success
                     is gripping and then losing it.
  • chance vs        did the policy GET a chance and did it TAKE it. Leads with
    conversion       the geometric test (object material really between the open
                     jaws) rather than agreement with the pinned pose. A flat
                     success rate with a rising chance rate means the reach is
                     solved and the trigger is not.
  • approach error   EE -> grasp: the CLOSEST reached over the episode (solid;
                     defined even when the policy never closes) and the error AT
                     the close (dashed), position on the left axis and rotation
                     on the right. Phase-3 experience is that ROTATION binds
                     first.
  • COLLECTION       how the DAgger episodes of this direction ended.
    outcomes
  • EVAL outcomes    how the evaluation episodes of this direction ended, same
                     taxonomy and colours.

  THE LAST TWO ARE DIFFERENT POPULATIONS AND ARE ADJACENT ON PURPOSE. Collection
  runs the beta mixture (0.9->0.75, so most steps are the OMG expert's) with DART
  injected and `expert_after_commit` forcing the reach; eval runs the policy
  alone at beta=0, no expert, no DART, on its own 100-scene draw that is never
  written to a shard. So column 4 is how the LOOP fails and column 5 is how the
  POLICY fails, and they move independently — run 12's eval stack barely shifted
  while `co_timeout` took over its collection stack. Two asymmetries stop them
  being comparable as LEVELS: a collection episode is only ever scored if a close
  fires, so everything that timed out or was killed lands in column 4's grey and
  brown rather than in any grasp category; and with `stop_on_policy_close: false`
  the close that fires in collection is the EXPERT's geometric trigger, not the
  learner's decision.

  Column 4 needs DAGGER.outcome_check and the per-bin `co_*_b{b}` columns, which
  postdate run 12. It cannot be backfilled: the taxonomy comes from pushing and
  holding in the simulator, and a shard records no outcome.

WHY PER BIN, AND WHY THIS IS THE MAIN FIGURE. A pooled `success_rate` averages
four physically different commands. A policy that solves `+x` and ignores `+z`
plots identically to one that is mediocre at both, and only the first is
evidence the conditioning is being read — which is the entire question this
phase exists to answer. The rows are the four directions the dataset can
actually reach (`-x` is demonstrable by 12 of 623 scenes and `-z` by none), and
which four is read from the log rather than assumed.

DIAGNOSTIC (<run>/curves_diag.png) — 2x4, "is the machinery healthy":
  • labels     — mean approach-label displacement vs ee_step, and the count of
                 degenerate (~zero) labels. A collapse towards 0 is the standoff
                 stall returning; it is invisible in step counts.
  • endgame    — CLOSE labels produced and reach-tail steps followed: whether the
                 expert is still teaching the part of the task that matters. Also
                 the premature closes the policy commanded (each one is relabelled
                 OPEN, never copied into D), and the pairs dropped because an
                 EXPERT step collided and ended the episode — that red line should
                 sit at 0, or exclude_scenes is missing scenes the expert breaks.
  • planner    — OMG failures, goal-grasp switches, pinned episodes. With the pin
                 table loaded, goal_switch should be flat 0; anything else means
                 the table is stale or not matching.
  • mixing     — beta and the expert's share of executed steps, plus (DART runs
                 only) the share taken by random jolts and the fraction of those
                 jolts that ended the episode. That last line is the magnitude
                 tripwire: if it climbs, dart_pos_mag is knocking the object out
                 of the hand rather than displacing the gripper.
  • closing    — when each side decides to close: the learner's mean close step
                 during collection vs at eval, against the horizon.
  • cost       — wall-clock split by phase, so the expensive part is visible.

DEBUG DAGGER (<run>/debug_dagger.png) — 3x4, "is the LOOP working". The per-bin
outcome stacks moved to training_curve.png, next to the success rate they
explain; what stays here is about the DAgger DATA rather than the policy.
  • row 1      — collection, per commanded direction: how far the learner's own
                 rollouts got. This is DAgger's state distribution shifting, and
                 it moves BEFORE eval success does — and it can move for one
                 direction and not another, which pooling hides.
  • row 2      — HOW CLOSE the collection episodes actually got, per direction:
                 the mean over each iteration of the CLOSEST the EE came to the
                 grasp, position and rotation, against their close thresholds.
                 The training figure asks this of the eval episodes; this asks
                 it of the data that decides what D contains, and the two can
                 disagree completely. Run 12 held `eval_min_pos` near 0.105 m
                 while its collection stalled at 0.131 m against run 11's
                 0.049 m — invisible on the eval figure, because eval carries no
                 DART. A flat line here means D has stopped gaining states near
                 the grasp and no further iteration will help.

                 `c_min_pos_b{b}` / `c_min_rot_b{b}` postdate run 12, but unlike
                 the outcome stacks they ARE recoverable: the shards hold
                 `robot_states` and `grasp_pose_world`, so
                 examples/backfill_collect_err.py recomputes them into
                 <run>/collect_err.csv, which is spliced in here. That
                 recomputation reads ~2-3 mm high (see the script) because the
                 collector minimises over steps a shard does not keep; the level
                 shifts slightly, no trend does.

  • row 3      — the pooled outcome stack, the pin-consistency check
                 (`grasp_mismatch` must stay 0 — a nonzero bar means D holds
                 contradictory labels for one scene), |D| growth with the
                 on-policy share, and the refit's train/val loss. Under FTL each
                 refit point is a fresh fit on a bigger, more diverse D, so a
                 RISING train loss means the aggregate is becoming
                 self-inconsistent, which is the failure DAgger cannot average
                 away.

CONDITIONING (<run>/curves_regrasp.png) — 2x3, "is the conditioning doing
anything". Read this one first.
  • is it USING the command — the pooled headline. `dir_track`
                 (1 - mean dir_err / 90 deg) is 1 when the gripper ends on the
                 axis it was told to come in on and 0 when it ignores the
                 command; `bin_diag_rate` is the discrete version and
                 `bin_hit_rate` asks about the SIDE the gripper arrived from
                 rather than its orientation. Chance for both is 1/4 with four
                 live bins, drawn as the dotted line — a rate sitting ON it means
                 the policy orients freely and the conditioning is inert.
  • retry@k    — success with k attempts at different directions. The regrasping
                 headline, derived from the same episodes at no extra cost. It
                 assumes each retry restarts from home, which is true of this
                 evaluation and NOT of a real deployment, where attempt 2 begins
                 wherever attempt 1 stopped. Read it as a ceiling; the chained
                 version is in eval_regrasp_testset.py.
  • ended in the COMMANDED bin, PER BIN — the literal question, and the panel the
                 phase turns on: told `+z`, what fraction of episodes ended with
                 the gripper's approach axis in `+z`? Per bin rather than pooled
                 because a pooled 0.50 is equally consistent with "follows all
                 four commands half the time" and "nails +x and ignores the
                 rest", and only the second is a reason to change anything.
  • per-bin success — one curve per commanded direction. The spread is how much
                 the direction matters: curves on top of each other mean
                 retrying is four draws from one distribution.
  • per-bin tracking — `dir_track` per direction, the continuous companion to the
                 panel above it.
  • arrived from the COMMANDED side, PER BIN — orientation and side fail
                 independently: a gripper can point the right way on the wrong
                 side of the object. Run 1 measured exactly that pooled
                 (bin_diag ~0.50, bin_hit at chance); per bin is where it becomes
                 actionable.

MEDIA (<run>/media_curves.png) — the presentation cut. The same data and the
same panel functions, POOLED over directions and trimmed to five panels that fit
on a slide. Nothing in it is new; the per-bin split, the pin check, |D| growth
and the refit loss are all dropped as internal questions.

TARGET AWARENESS (run 21 on). `DAGGER.target` is read from <run>/config.yaml,
because under `pregrasp` several columns keep their NAMES and change their
MEANING — the episode ends 6.4 cm short of the grasp, so `eval_min_pos`,
`mean_pos_err` and `chance_rate` are measured against the PRE-GRASP. A pre-grasp
min-pos-err of 0.005 is not twenty times better than run 16's 0.10; it is a
different quantity, and plotting the two under one label invites exactly the
wrong comparison. Every affected panel is relabelled, and the pooled panels gain
`mean_reach_pos_err` (diamonds) — where the BLIND push ended up relative to the
GRASP, and the only line comparable to a grasp-mode run's min pos err.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml
matplotlib.use("Agg")  # headless-safe; overridden by --show below
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# `directions` is pure numpy — no torch, no sim, no h5py — so importing it costs
# this script nothing and keeps the bin names and the LIVE_BINS default in ONE
# place. Duplicating them here as string literals is exactly the kind of copy
# that survives a bin-set change and silently mislabels every figure afterwards.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handover_sim2real.regrasp import directions as _D          # noqa: E402


def _load(log_path: Path, eval_log: Path | None = None,
          extra_logs: tuple[Path, ...] = ()):
    cols: dict[str, list] = {}
    with log_path.open() as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)

    n_rows = len(cols.get("iter", []))

    def splice(path: Path) -> None:
        """Fill BLANK cells from a sidecar CSV keyed on `iter`.

        Two sidecars use this. `eval_log.csv` carries the eval columns when
        EVAL.every: 0 put scoring in a separate job; `collect_err.csv` carries
        per-bin collection error recomputed from the shards for a run that
        predates those columns (examples/backfill_collect_err.py).

        Only blanks are filled, so an in-loop number always wins and a run that
        logged its own values is never overwritten by a recomputation.
        """
        with path.open() as f:
            by_iter = {r["iter"]: r for r in csv.DictReader(f) if r.get("iter")}
        if not by_iter:
            return
        keys = set().union(*(set(r) for r in by_iter.values())) - {"iter", "run_dir"}
        filled = 0
        for k in keys:
            col = cols.setdefault(k, [""] * n_rows)
            for i, it in enumerate(cols.get("iter", [])):
                if i < len(col) and not str(col[i]).strip():
                    v = by_iter.get(str(it), {}).get(k, "")
                    if str(v).strip():
                        col[i] = v
                        filled += 1
        print(f"[plot] merged {len(by_iter)} rows / {filled} cells from "
              f"{path.name}")

    if eval_log is not None and eval_log.exists():
        splice(eval_log)
    for extra in extra_logs:
        if extra is not None and extra.exists():
            splice(extra)

    def num(key):
        """Column as floats. Absent column (an older log) or blank cell -> NaN,
        which matplotlib renders as a gap rather than a spurious zero."""
        vals = cols.get(key)
        if not vals:
            return [float("nan")] * n_rows
        out = []
        for v in vals:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out

    return num, n_rows


def _finite(ys) -> bool:
    return any(y == y for y in ys)


def _plot(ax, xs, ys, style="-", **kw):
    """Plot a series, DROPPING missing points rather than passing NaN through.

    Matplotlib breaks a line at every NaN, so a column that is only filled on
    some iterations — every eval column when EVAL.every > 1, every collection
    column on iteration 0 — would render as isolated markers while the legend
    advertised a connected line. Dropping the gaps instead joins the points that
    do exist, which is what these series mean: the same policy sequence, sampled
    less often, not a discontinuity in it.

    Returns False if there was nothing to draw, so callers can skip the label.
    """
    pts = [(x, y) for x, y in zip(xs, ys) if y == y]
    if not pts:
        return False
    ax.plot([p[0] for p in pts], [p[1] for p in pts], style, **kw)
    return True


def _stack(ax, xs, series, labels, colors):
    """Stacked area over only the iterations that HAVE data.

    `stackplot` cannot take NaN, and substituting 0 is worse than useless here:
    an iteration with no eval (EVAL.every > 1) would stack to zero and render as
    a spike down to the axis, which reads as "everything failed" rather than
    "not measured". So drop those rows entirely and stack the rest.
    """
    keep = [i for i in range(len(xs)) if any(s[i] == s[i] for s in series)]
    if not keep:
        return False
    clean = [[(s[i] if s[i] == s[i] else 0.0) for i in keep] for s in series]
    ax.stackplot([xs[i] for i in keep], *clean, labels=labels, colors=colors,
                 alpha=0.85)
    return True


def _grid(ax, title, xlabel="DAgger iteration", ylabel=None):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)


def _legend(ax, **kw):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, **kw)


def _fix_x(fig, it):
    """Iterations are integers, and a panel whose series are all-NaN (no closes
    yet, say) would otherwise fall back to a 0..1 axis that silently disagrees
    with its neighbours."""
    lo, hi = min(it), max(it)
    pad = 0.2 if hi > lo else 0.5
    for a in fig.axes:
        a.set_xlim(lo - pad, hi + pad)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))


# ── PER-BIN PANELS ───────────────────────────────────────────────────────────
#
# Every eval panel is drawn once per commanded direction, and the SAME function
# draws the pooled version on the media figure. That is deliberate: a per-bin
# panel that differs from its aggregate in some detail of what it plots is a
# figure that cannot be read against itself, and the only reliable way to keep
# them identical is for there to be one implementation.
#
# `sfx` selects the column family: "" is the pooled eval set, "_b3" is the
# episodes commanded to approach from `-y`. Columns that exist only pooled (the
# auxiliary head, the pinned-pose reference pair, the pre-grasp push) draw only
# when `sfx` is empty, so a per-bin panel is a strict subset rather than a panel
# with mysterious gaps.


# Columns that were called something else before the per-bin family existed.
# `succ_bin_{b}` IS `success_rate_b{b}` — the same reduction over the same rows —
# so accepting it keeps run 1's figures readable instead of rendering four empty
# rows for the only run that predates the schema.
_LEGACY = {f"success_rate_b{b}": f"succ_bin_{b}" for b in range(len(_D.BINS))}
_LEGACY["n_b0"] = "n_bin_0"

# One colour per BIN INDEX, so `+y` is the same green on every panel of every
# figure. Indexed by bin, not by position in `bins`, because a run that never
# commands `-y` must not silently shift `+z` onto `-y`'s colour.
_BIN_COLOURS_BY_BIN = ("tab:blue", "tab:brown", "tab:green", "tab:orange",
                       "tab:red", "tab:purple")


class _Ctx:
    """What every panel needs: the columns, the x axis, and which pose the run
    was steering to."""

    def __init__(self, num, it, args, target):
        self.num, self.it, self.args = num, it, args
        self.target = target
        self.pregrasp = target == "pregrasp"
        self.TGT = "pre-grasp" if self.pregrasp else "grasp"

    def get(self, key):
        """A column, falling back to its pre-rename name if the new one is empty."""
        ys = self.num(key)
        if not _finite(ys) and key in _LEGACY:
            ys = self.num(_LEGACY[key])
        return ys


def _note_empty(a, msg="not recorded in this run's log"):
    """Say WHY a panel is blank.

    An empty axis reads as "the measurement is zero", which is a claim about the
    policy. These panels are blank because the columns did not exist when the run
    was logged, which is a claim about the CSV — a distinction worth one line of
    grey text.
    """
    if not a.get_lines() and not a.collections:
        a.text(0.5, 0.5, msg, transform=a.transAxes, ha="center", va="center",
               fontsize=8, color="0.55", style="italic")
        return True
    return False


def _bins_to_plot(num, k=4):
    """Which directions this run actually commanded, in bin order.

    Ranked by episode count and cut at `k`, because the octahedral set has six
    bins and this dataset can reach four — four blank rows would be four rows of
    nothing. Sorted back into bin order afterwards so the rows read
    +x, +y, -y, +z rather than by popularity. Falls back to `LIVE_BINS` for a log
    with no per-bin counts at all.
    """
    tot = {}
    for b in range(len(_D.BINS)):
        # `n_b{b}` is the new column; `n_bin_{b}` is what run 1 wrote. Either
        # answers "did this bin get episodes", so accept both rather than
        # rendering an empty figure for a log that predates the rename.
        ns = [v for v in num(f"n_b{b}") if v == v]
        if not any(ns):
            ns = [v for v in num(f"n_bin_{b}") if v == v]
        if sum(ns) > 0:
            tot[b] = sum(ns)
    if not tot:
        return list(_D.LIVE_BINS)
    return sorted(sorted(tot, key=lambda b: -tot[b])[:k])


def _bin_title(b: str | int) -> str:
    return f"{_D.BIN_SHORT[b]} ({_D.BIN_NAMES[b].split('_', 1)[1].replace('_', ' ')})"


def _rung(num, k: int) -> str:
    """" — mostly +x (79%)" for the k-th rung of the retry ladder, else "".

    THE RUNG IS NOT ONE DIRECTION. `retry_at_k` walks each scene's pin slots in
    ascending bin order, so rung 1 is `+x` for a scene that can reach `+x` and
    `+y` for one that cannot — the rung is a mixture across scenes. Naming the
    modal bin AND its share is the only honest label: a share near 1.0 means the
    ladder really is that direction, and a share near 0.4 is the reader's warning
    not to read the curve as "then it tried +y".

    The composition is a property of the pin table and the eval scene set, not of
    the checkpoint, so it is constant across iterations and the modal bin is
    taken over the whole run rather than per point.
    """
    bs = [b for b in num(f"retry_bin_{k}") if b == b]
    fr = [f for f in num(f"retry_bin_frac_{k}") if f == f]
    if not bs:
        return ""
    b = int(round(max(set(bs), key=bs.count)))
    if not 0 <= b < len(_D.BIN_SHORT):
        return ""
    if not fr:
        return f" — mostly {_D.BIN_SHORT[b]}"
    return f" — mostly {_D.BIN_SHORT[b]} ({sum(fr) / len(fr):.0%})"


def _panel_nested(a, c: _Ctx, sfx="", title=None):
    """close -> near -> grasp -> success, which are NESTED. The vertical GAPS are
    the diagnosis: close-near is closing in the wrong place, near-grasp is closing
    right but not gripping, grasp-success is gripping and then losing it."""
    for key, label, style in (("close_rate", "close", ":"),
                              ("near_rate", "near (pose ok)", "-."),
                              ("grasp_rate", "grasp", "--"),
                              ("success_rate", "success", "-")):
        ys = c.get(key + sfx)
        if _finite(ys):
            _plot(a, c.it, ys, style, marker="o", ms=3, label=label,
                  lw=2 if key == "success_rate" else 1.2)
    a.set_ylim(-0.02, 1.02)
    _note_empty(a)
    _grid(a, title or "success stages", ylabel="fraction of episodes")
    _legend(a, loc="upper left")


def _panel_opportunity(a, c: _Ctx, sfx="", title=None, lean=False):
    """Did it get a chance, and did it take it.

    LEADS WITH THE GEOMETRIC TEST (box_*, regrasp/grasp_box.py): object material
    actually between the open jaws, which counts an off-pose grasp as the
    opportunity it is. The pinned-pose pair (chance_rate / miss_given_chance) is
    a faint reference only — it gates on agreement with the pin and reads
    0.03-0.05 in runs succeeding 60-70% of the time, so it measures pin
    agreement, not opportunity.
    """
    for key, label, style, col, lw in (
            ("box_chance_rate", "object in jaws", "-", "tab:blue", 1.4),
            ("box_taken_rate", "closed | in jaws", "-", "tab:green", 2.0),
            ("miss_given_box", "no grasp | in jaws", "-", "tab:red", 1.4),
            ("close_success_rate", "success | closed", "--", "tab:olive", 1.2),
            ("mean_box_frac", "mean jaw occupancy", ":", "tab:gray", 1.0)):
        # LEAN keeps only the three that carry the story. `success | closed` is a
        # fourth conditioning on the same episodes and crowds the panel, and
        # `mean jaw occupancy` tracks `object in jaws` so closely in these runs
        # that it reads as a duplicate line.
        if lean and key in ("close_success_rate", "mean_box_frac"):
            continue
        ys = c.get(key + sfx)
        if _finite(ys):
            _plot(a, c.it, ys, style, marker="o", ms=3, color=col, label=label, lw=lw)
    if not sfx and not lean:
        # Pre-grasp mode: the jaws cannot contain the object while the policy is
        # still deciding — it stops 6.4 cm short — so box_chance_rate above reads
        # ~0 BY CONSTRUCTION and box_taken_rate is NaN with it. `box_after_rate`
        # is this mode's conversion measure.
        ys = c.get("box_after_rate")
        if _finite(ys):
            _plot(a, c.it, ys, "-D", ms=4, color="tab:purple", lw=2.0,
                  label="in jaws AFTER push | committed")
        for key, label, col in (
                ("chance_rate", f"at pinned {c.TGT} (ref)", "tab:blue"),
                ("miss_given_chance", "missed | pinned (ref)", "tab:red")):
            ys = c.get(key)
            if _finite(ys):
                _plot(a, c.it, ys, ":", marker="", color=col, label=label,
                      lw=1.0, alpha=0.4)
    a.set_ylim(-0.02, 1.02)
    _note_empty(a)
    _grid(a, title or "opportunity vs conversion", ylabel="fraction")
    _legend(a, loc="upper left")


def _panel_approach(a, c: _Ctx, sfx="", title=None):
    """EE -> target: the CLOSEST reached over the episode (solid; defined even
    when the policy never closes) and the error AT the close (dashed), position
    on the left axis and rotation on the right. Phase-3 experience is that
    ROTATION binds first."""
    args, TGT = c.args, c.TGT
    mp, mr = c.get("eval_min_pos" + sfx), c.get("eval_min_rot" + sfx)
    pe, re_ = c.get("mean_pos_err" + sfx), c.get("mean_rot_err" + sfx)
    if _finite(mp):
        _plot(a, c.it, mp, "-o", ms=3, color="tab:blue",
              label=f"min pos err to {TGT} (m)")
    if _finite(pe):
        _plot(a, c.it, pe, "--o", ms=3, color="tab:cyan", alpha=0.8,
              label=(f"pos err to {TGT} at commit (m)" if c.pregrasp
                     else "pos err at close (m)"))
    if not sfx:
        # Pre-grasp mode: where the BLIND push ended up relative to the GRASP.
        # THIS, not `mp`, is the line comparable to a grasp-mode run's min pos
        # err — the only one measured against the pose the gripper has to end on.
        rp = c.get("mean_reach_pos_err")
        if _finite(rp):
            _plot(a, c.it, rp, "-D", ms=3, color="tab:purple",
                  label="pos err to GRASP after blind push (m)")
        # Auxiliary goal-grasp head: how far the network's BELIEF about the grasp
        # is from the pinned pose, on the same axes as the gripper's own error on
        # purpose. An accurate prediction alongside a gripper that still arrives
        # far away means the information is in the features and the action head
        # is not using it; both large means the observation cannot support the
        # target at all.
        ap = c.get("aux_pos_mm")
        if _finite(ap):
            _plot(a, c.it, (np.asarray(ap, dtype=float) / 1000.0).tolist(), ":^",
                  ms=3, color="tab:green", alpha=0.9,
                  label="aux: predicted grasp pos err (m)")
    a.axhline(args.pos_thresh, color="tab:blue", ls=":", lw=1,
              label=f"close thresh {args.pos_thresh} m")
    a.set_ylim(bottom=0)
    a.set_ylabel("position error (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    if _finite(mr):
        _plot(a2, c.it, mr, "-s", ms=3, color="tab:red",
              label=f"min rot err to {TGT} (rad)")
    if _finite(re_):
        _plot(a2, c.it, re_, "--s", ms=3, color="tab:orange", alpha=0.8,
              label=(f"rot err to {TGT} at commit (rad)" if c.pregrasp
                     else "rot err at close (rad)"))
    if not sfx:
        rr = c.get("mean_reach_rot_err")
        if _finite(rr):
            _plot(a2, c.it, rr, "-D", ms=3, color="tab:brown", alpha=0.9,
                  label="rot err to GRASP after blind push (rad)")
        ar = c.get("aux_rot_deg")
        if _finite(ar):
            _plot(a2, c.it, np.radians(np.asarray(ar, dtype=float)).tolist(), ":^",
                  ms=3, color="tab:olive", alpha=0.9,
                  label="aux: predicted grasp rot err (rad)")
    a2.axhline(args.rot_thresh, color="tab:red", ls=":", lw=1,
               label=f"close thresh {args.rot_thresh} rad")
    a2.set_ylim(bottom=0)
    a2.set_ylabel("rotation error (rad)", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    if not any(_finite(y) for y in (mp, mr, pe, re_)):
        # The two dotted threshold lines are always drawn, so the generic
        # empty-axis test cannot fire here — ask the data directly.
        a.text(0.5, 0.5, "not recorded in this run's log", transform=a.transAxes,
               ha="center", va="center", fontsize=8, color="0.55", style="italic")
    _grid(a, title or f"EE -> {c.TGT}: closest (solid) vs at the close (dashed)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        # LOWER left, not upper right. Both axes are forced to start at 0 while
        # the errors live near the top of their range (~0.1 m on a 0-0.11 axis),
        # so the bottom-left corner is the only reliably empty one — upper right
        # sat on top of the curves in every run so far.
        a.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower left", ncol=2,
                 framealpha=0.85)


# The eval outcome taxonomy, drawn two ways. `f_*` are fractions of the eval set
# and stack to 1.0; `ff_*` drop the success and are fractions of the FAILURES, so
# they stack to 1.0 over "of the episodes that came away with nothing, what went
# wrong". The second is the one that separates two bins failing at the same rate
# for different reasons, which is exactly what a per-bin figure is for.
_OUTCOMES = (("f_grasp_ok", "secured", "tab:green"),
             ("f_grasp_miss", "closed, not secured", "tab:olive"),
             ("f_no_release", "no release", "tab:orange"),
             ("f_drop", "drop", "tab:red"),
             ("f_human_contact", "human contact", "tab:purple"),
             ("f_timeout", "never closed", "tab:gray"))
_FAILURES = (("ff_grasp_miss", "closed, not secured", "tab:olive"),
             ("ff_no_release", "no release", "tab:orange"),
             ("ff_drop", "drop", "tab:red"),
             ("ff_human_contact", "human contact", "tab:purple"),
             ("ff_timeout", "never closed / timed out", "tab:gray"))
# The COLLECTION twin of `_OUTCOMES`, same taxonomy and same colours so the two
# rows can be read against each other at a glance. Two differences that matter:
# these are RAW COUNTS over `c_episodes_b{b}` rather than fractions, and the
# taxonomy has one extra category — BENCH_TIMEOUT, which eval cannot produce
# because it stops at EVAL.max_steps well inside the benchmark's own limit.
_COLLECT_OUTCOMES = (("co_grasp_ok", "secured", "tab:green"),
                     ("co_grasp_miss", "closed, not secured", "tab:olive"),
                     ("co_no_release", "no release", "tab:orange"),
                     ("co_drop", "drop", "tab:red"),
                     ("co_human_contact", "human contact", "tab:purple"),
                     ("co_bench_timeout", "benchmark timeout", "tab:brown"),
                     ("co_timeout", "never closed", "tab:gray"))


def _panel_outcomes(a, c: _Ctx, sfx="", title=None, failures_only=False,
                    legend=True):
    series = _FAILURES if failures_only else _OUTCOMES
    if _stack(a, c.it, [c.get(k + sfx) for k, _, _ in series],
              [lb for _, lb, _ in series], [col for _, _, col in series]):
        a.set_ylim(0, 1)
    _note_empty(a)
    _grid(a, title or ("failure profile (of the episodes that failed)"
                       if failures_only else "eval outcomes"),
          ylabel="fraction")
    # A stacked area has no empty corner, so four identical legends across a row
    # cover four panels' worth of data to say the same thing once. Callers drawing
    # a row pass legend=False on all but the first.
    if legend:
        _legend(a, loc="lower left", ncol=2)


def _panel_collect_outcomes(a, c: _Ctx, b=None, title=None, legend=True):
    """How the COLLECTION episodes ended, per direction — the twin of
    `_panel_outcomes`, on the DAgger data rather than the eval set.

    The two rows answer different questions and neither substitutes for the
    other. Eval is beta=0 with no DART, so `f_*` is how the POLICY fails.
    Collection is the beta mixture (0.9->0.75, so mostly the expert) with DART
    injected and `expert_after_commit` forcing the reach, so `co_*` is how the
    LOOP fails — and a loop that stops producing arrivals starves D of exactly
    the states the policy needs, without eval showing anything for an iteration
    or two. Run 12 is the case: its eval stack barely moved while `co_timeout`
    took over the collection stack.

    Counts, not fractions (see bin_collect_fields), so divide by
    `c_episodes_b{b}` here. Requires DAGGER.outcome_check and postdates run 12;
    an older log has neither the pooled nor the per-bin columns and gets a note.
    """
    denom = c.get(f"c_episodes_b{b}" if b is not None else "episodes")
    series = []
    for key, _, _ in _COLLECT_OUTCOMES:
        ys = c.get(key + (f"_b{b}" if b is not None else ""))
        # -1 marks "this bin collected nothing", which is a gap, not a zero.
        series.append([(y / d if (d and d > 0 and y == y and y >= 0)
                        else float("nan")) for y, d in zip(ys, denom)])
    if _stack(a, c.it, series, [lb for _, lb, _ in _COLLECT_OUTCOMES],
              [col for _, _, col in _COLLECT_OUTCOMES]):
        a.set_ylim(0, 1)
    _note_empty(a, "per-bin collection outcomes not in this run's log\n"
                   "(needs DAGGER.outcome_check; added after run 12)")
    _grid(a, title or "collection outcomes", ylabel="fraction")
    if legend:
        _legend(a, loc="lower left", ncol=2)


def _panel_collect_error(a, c: _Ctx, b=None, title=None, legend=True):
    """How close the COLLECTION episodes got, per direction — the DAgger-data
    twin of `_panel_approach`.

    `mean_min_pos` / `mean_min_rot` are the CLOSEST the EE came to the target
    over each episode, averaged over that bin's episodes. Closest rather than
    terminal, and so defined even for an episode that never closed — which is
    most of them in a bad run, and exactly the ones a terminal-only statistic
    would silently exclude.

    This is the row that reads a DART regression at a glance. Run 12's
    `eval_min_pos` sat at 0.105 m all run while its collection stalled at
    0.131 m against run 11's 0.049 m: the eval figure could not show it, because
    eval carries no DART. A flat line here means D has stopped gaining states
    near the grasp, and nothing downstream will improve.

    Position on the left axis, rotation on the right, both against their close
    thresholds — the same layout as the eval panel so the two are comparable by
    eye. Per-bin columns postdate run 12; `backfill_collect_err.py` recomputes
    them from an existing run's shards.
    """
    args = c.args
    sfx = f"_b{b}" if b is not None else ""
    mp = c.get(f"c_min_pos{sfx}") if b is not None else c.get("mean_min_pos")
    mr = c.get(f"c_min_rot{sfx}") if b is not None else c.get("mean_min_rot")
    if _finite(mp):
        _plot(a, c.it, mp, "-o", ms=3, color="tab:blue",
              label="closest EE->target (m)")
    a.axhline(args.pos_thresh, color="tab:blue", ls=":", lw=1,
              label=f"close thresh {args.pos_thresh} m")
    a.set_ylim(bottom=0)
    a.set_ylabel("position error (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
    a2 = a.twinx()
    if _finite(mr):
        _plot(a2, c.it, mr, "--s", ms=3, color="tab:red",
              label="closest rotation (rad)")
    a2.axhline(args.rot_thresh, color="tab:red", ls=":", lw=1,
               label=f"close thresh {args.rot_thresh} rad")
    a2.set_ylim(bottom=0)
    a2.set_ylabel("rotation error (rad)", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    if not (_finite(mp) or _finite(mr)):
        # Two threshold lines are always drawn, so the generic empty-axis test
        # cannot fire — ask the data.
        a.text(0.5, 0.5, "per-bin collection error not in this run's log\n"
                         "(run examples/backfill_collect_err.py <run>)",
               transform=a.transAxes, ha="center", va="center", fontsize=8,
               color="0.55", style="italic")
    _grid(a, title or "collection: closest EE -> target")
    if legend:
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        if h1 or h2:
            a.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower left", ncol=2,
                     framealpha=0.85)


def _panel_collection(a, c: _Ctx, b=None, title=None, lean=False, legend=True):
    """How far the LEARNER's own rollouts got during collection. This is DAgger's
    state distribution shifting, and it moves BEFORE eval success does.

    `b` selects a bin's `c_*_b{b}` counters; None is the pooled pair. The
    per-bin counters are raw counts against `c_episodes_b{b}`, and the collector
    writes -1 (not 0) for a bin it collected nothing for, so a reused shard —
    which reports no breakdown at all — reads as a gap rather than as a
    collapse.
    """
    if b is None:
        eps = c.get("episodes")
        keys = (("reached_standoff", "reached standoff"),
                ("reached_grasp", "reached pre-grasp (COMMIT fired)"
                 if c.pregrasp else "reached grasp (CLOSE fired)"),
                ("policy_closed",
                 "committed on its own" if c.pregrasp else "closed on its own"))
    else:
        eps = c.get(f"c_episodes_b{b}")
        keys = ((f"c_reached_standoff_b{b}", "reached standoff"),
                (f"c_reached_grasp_b{b}", "reached pre-grasp (COMMIT fired)"
                 if c.pregrasp else "reached grasp (CLOSE fired)"),
                (f"c_policy_closed_b{b}", "closed on its own"),
                (f"c_success_b{b}", "secured (beta mixture)"))
    if lean:
        # `closed on its own` is a collection-side counter that needs beta to
        # interpret, and the presentation figure no longer shows beta.
        keys = keys[:2]
    for key, label in keys:
        ys = c.get(key)
        frac = [(y / e if (e and e > 0 and y == y and y >= 0) else float("nan"))
                for y, e in zip(ys, eps)]
        if _finite(frac):
            _plot(a, c.it, frac, "-o", ms=3, label=label)
    a.set_ylim(-0.02, 1.02)
    _note_empty(a)
    _grid(a, title or "collection: how far the LEARNER got",
          ylabel="fraction of episodes")
    if legend:
        _legend(a, loc="upper left")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="output/dagger_runs/<name>")
    p.add_argument("--show", action="store_true")
    p.add_argument("--ee-step", type=float, default=0.04,
                   help="DAGGER.ee_step, drawn as the reference label scale")
    p.add_argument("--pos-thresh", type=float, default=0.02)
    p.add_argument("--rot-thresh", type=float, default=0.34)
    args = p.parse_args()

    run = Path(args.run_dir)
    log_path = run / "dagger_log.csv"
    if not log_path.exists():
        raise SystemExit(f"no log at {log_path}")

    num, n = _load(log_path, eval_log=run / "eval_log.csv",
                   extra_logs=(run / "collect_err.csv",))
    if n == 0:
        raise SystemExit(f"{log_path} has no rows yet")
    it = num("iter")

    # ---- which pose the run was steering to (DAGGER.target, run 21 on) ----
    # Under target: pregrasp the episode ends 6.4 cm short of the grasp, so
    # eval_min_pos / mean_pos_err / chance_rate are measured against the PRE-GRASP
    # and box_chance_rate is ~0 by construction. Plotting those under the labels
    # this file used for runs 1-20 would invite exactly the wrong comparison —
    # a pre-grasp "min pos err" of 0.005 is not twenty times better than run 16's
    # 0.10, it is a different quantity. Read from the run's own saved config so a
    # log can never be plotted under the wrong labels.
    target = "grasp"
    cfg_path = run / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open() as f:
            target = str(((yaml.safe_load(f) or {}).get("DAGGER", {}) or {})
                         .get("target", "grasp"))
    pregrasp = target == "pregrasp"

    # Every panel reads its columns and its labels through this — including the
    # "pre-grasp" / "grasp" relabelling, which is now `ctx.TGT`.
    ctx = _Ctx(num, it, args, target)
    bins = _bins_to_plot(num)

    # ── TRAINING CURVE (<run>/training_curve.png) — one ROW per direction ────
    #
    # The pooled version of this figure is an average over four physically
    # different commands, and that average is exactly what hides the phase's
    # result: a policy that solves `+x` and ignores `+z` plots identically to one
    # that is mediocre at both, and only the first is evidence the conditioning
    # is doing anything. So the three eval panels are drawn once per bin, and
    # every row shares its y limits with the others by construction (all four are
    # fractions on [0, 1], and the approach row's axes are set from the data).
    #
    # The run-machinery panels that used to share this grid — the pin-consistency
    # check, |D| growth, the refit loss, and the collection curves — moved to
    # debug_dagger.png. None of them are results; all of them need the diagnostic
    # figure's context to mean anything, and here they competed for attention
    # with the four rows that ARE the result.
    # COLUMNS 4 AND 5 ARE THE FAILURE MODES, and they belong here rather than on
    # the diagnostic figure: "it failed" and "how it failed" are the same
    # question asked at two resolutions, and splitting them across two files
    # meant reading a success rate on one and the reason for it on another.
    # Column 4 is the COLLECTION episodes and column 5 the EVAL ones — different
    # populations (see _panel_collect_outcomes), adjacent so the pair is legible.
    nrow = max(len(bins), 1)
    fig, ax = plt.subplots(nrow, 5, figsize=(28, 3.7 * nrow), squeeze=False)
    for r, b in enumerate(bins):
        sfx = f"_b{b}"
        name = _bin_title(b)
        _panel_nested(ax[r][0], ctx, sfx, title=f"{name} — success stages")
        _panel_opportunity(ax[r][1], ctx, sfx,
                           title=f"{name} — chance vs conversion")
        _panel_approach(ax[r][2], ctx, sfx,
                        title=f"{name} — approach error to the {ctx.TGT}")
        _panel_collect_outcomes(
            ax[r][3], ctx, b=b, legend=(r == 0),
            title=f"{name} — COLLECTION outcomes (beta mix + DART)")
        _panel_outcomes(ax[r][4], ctx, sfx=sfx, legend=(r == 0),
                        title=f"{name} — EVAL outcomes (policy alone, beta=0)")

    _fix_x(fig, it)
    fig.suptitle(f"Regrasp — {run.name}   [eval, by commanded direction]"
                 + (f"   [target: {target} — geometry is measured to the "
                    f"PRE-GRASP, not the grasp]" if pregrasp else ""),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.03 / nrow * 2])
    main_out = run / "training_curve.png"
    fig.savefig(main_out, dpi=140)
    print(f"wrote {main_out}")

    # ── DIAGNOSTIC ──────────────────────────────────────────────────────────
    fig2, bx = plt.subplots(2, 4, figsize=(21, 8))

    # labels
    a = bx[0][0]
    lp = num("mean_label_pos")
    if _finite(lp):
        _plot(a, it, lp, "-o", ms=3, color="tab:blue", label="mean approach label (m)")
    a.axhline(args.ee_step, color="k", ls=":", lw=1, label=f"ee_step {args.ee_step} m")
    a.set_ylim(bottom=0)
    a.set_ylabel("label displacement (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    ty = num("tiny_labels")
    if _finite(ty):
        _plot(a2, it, ty, "-s", ms=3, color="tab:red", label="degenerate (~0) labels")
    a2.set_ylabel("count", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    _grid(a, "expert label scale (stall detector)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")

    # endgame coverage
    a = bx[0][1]
    for key, label in (("close_labels", "COMMIT labels" if pregrasp
                        else "CLOSE labels"),
                       ("reach_steps", "committed pre-grasp hold steps"
                        if pregrasp else "committed-reach steps"),
                       # Of those hold steps, how many were the convergence
                       # servo. Read against commit_settle_steps x the episodes
                       # that committed: a shortfall means episodes are entering
                       # the tolerance ball and then being ended before they can
                       # settle, which puts the arrival error — and therefore the
                       # landing error — straight back.
                       ("settle_steps", "settle steps"),
                       ("policy_close_cmds", "premature commit cmds"
                        if pregrasp else "premature close cmds")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, "-o", ms=3, label=label)
    # Pairs dropped because an EXPERT step ended the episode (the lateral-approach
    # collision). Should be ~0: a nonzero line means exclude_scenes is not
    # catching the scenes where the expert itself still collides.
    ys = num("dropped_tail")
    if _finite(ys):
        _plot(a, it, ys, "-s", ms=3, color="tab:red",
              label="expert-collision pairs dropped")
    _grid(a, "endgame coverage in D_i", ylabel="labels this iteration")
    _legend(a, loc="upper left")

    # planner health
    a = bx[0][2]
    for key, label in (("omg_fail", "OMG plan failures"),
                       ("goal_switch", "goal-grasp switches"),
                       ("pinned", "episodes pinned")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, "-o", ms=3, label=label)
    _grid(a, "planner / grasp pinning", ylabel="count this iteration")
    _legend(a, loc="upper left")

    # when does it close — and how long it had to decide. `mean_box_steps` is in
    # the same units (policy steps), so it belongs on this axis: a long window
    # sitting above the close step means the policy loitered in the jaws-occupied
    # state rather than never reaching it, which box_taken_rate alone cannot say.
    a = bx[1][0]
    for key, label in (("mean_policy_close_step", "collection (learner)"),
                       ("mean_close_step", "eval")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, "-o", ms=3, label=label)
    ys = num("mean_box_steps")
    if _finite(ys):
        _plot(a, it, ys, "--s", ms=3, color="tab:gray",
              label="opportunity window (steps)")
    _grid(a, "step at which the policy commits" if pregrasp
             else "step at which the policy closes", ylabel="policy step")
    _legend(a, loc="upper left")

    # beta / expert mixing
    a = bx[1][1]
    b = num("beta")
    if _finite(b):
        _plot(a, it, b, "-o", ms=3, color="tab:purple", label="beta (expert prob.)")
    steps = num("steps")
    exp = num("expert_steps")
    frac = [(e / s if (s and s > 0 and e == e and e >= 0) else float("nan"))
            for e, s in zip(exp, steps)]
    if _finite(frac):
        _plot(a, it, frac, "--s", ms=3, color="tab:brown", label="expert share of steps")
    # DART belongs here: it is the OTHER thing that overrides the executed action.
    # Absent (or all -1) on a run collected without it, so `_finite` keeps the
    # panel unchanged for every pre-DART run.
    dart = num("dart")
    if _finite(dart) and any(d == d and d > 0 for d in dart):
        dfrac = [(d / s if (s and s > 0 and d == d and d >= 0) else float("nan"))
                 for d, s in zip(dart, steps)]
        _plot(a, it, dfrac, "-^", ms=3, color="tab:olive",
              label="DART share of steps")
        # The magnitude tripwire: jolts that ended the episode, as a fraction of
        # jolts fired. Climbing means dart_pos_mag is knocking the object out of
        # the hand rather than displacing the gripper.
        ended = num("dart_env_done")
        if _finite(ended):
            efrac = [(e / d if (d and d > 0 and e == e and e >= 0) else float("nan"))
                     for e, d in zip(ended, dart)]
            if _finite(efrac):
                _plot(a, it, efrac, ":v", ms=3, color="tab:red",
                      label="jolts that ended the episode")
    a.set_ylim(-0.02, 1.02)
    _grid(a, "expert mixing", ylabel="fraction")
    _legend(a, loc="upper right")

    # cost
    a = bx[1][2]
    _stack(a, it, [num("collect_s"), num("train_s"), num("eval_s")],
           ["collect", "train", "eval"],
           ["tab:blue", "tab:orange", "tab:green"])
    _grid(a, "wall clock per iteration", ylabel="seconds")
    _legend(a, loc="upper left")

    # ---- collection outcomes (DAGGER.outcome_check; empty for runs 1-17) ----
    # Deliberately the same taxonomy and the same colours as the eval-outcome
    # panel on debug_dagger.png, so the two stacks can be read against each
    # other (`_OUTCOMES` is the shared definition of both). The
    # difference between them IS the beta mixture: this one is what the
    # expert/learner blend achieved on the collection scenes, that one is what the
    # policy achieves alone on the eval scenes.
    a = bx[0][3]
    series = [("co_grasp_ok", "secured", "tab:green"),
              ("co_grasp_miss", "closed, not secured", "tab:olive"),
              ("co_no_release", "no release", "tab:orange"),
              ("co_drop", "drop", "tab:red"),
              ("co_human_contact", "human contact", "tab:purple"),
              ("co_bench_timeout", "benchmark timeout", "tab:brown"),
              ("co_timeout", "never closed", "tab:gray")]
    if _stack(a, it, [num(k) for k, _, _ in series],
              [l for _, l, _ in series], [c for _, _, c in series]):
        a.set_ylim(0, 1)
    _grid(a, "collection outcomes (fraction of kept episodes)")
    _legend(a, loc="lower left", ncol=2)

    # ---- collection success vs eval success ----
    # The GAP is the quantity of interest, not either line. Collection runs at
    # beta (mostly expert) on the training pool; eval runs the policy alone on the
    # eval scenes. A collection line that stays high while eval sags says the data
    # is still good and the policy is not absorbing it; both sagging together says
    # the expert itself is failing on these scenes and the labels are the problem.
    a = bx[1][3]
    cs, es, bt = num("c_success_rate"), num("success_rate"), num("beta")
    if _finite(cs):
        _plot(a, it, cs, "-o", ms=3, color="tab:blue",
              label="collection (beta mixture)")
    if _finite(es):
        _plot(a, it, es, "-s", ms=3, color="tab:green", label="eval (policy alone)")
    if _finite(bt):
        _plot(a, it, bt, ":", color="tab:gray", label="beta (expert share)")
    a.set_ylim(0, 1)
    _grid(a, "success: collection vs eval", ylabel="fraction")
    _legend(a, loc="lower left")

    _fix_x(fig2, it)
    fig2.suptitle(f"Phase-5 DAgger diagnostics — {run.name}"
                  + (f"   [target: {target}]" if pregrasp else ""), fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    diag_out = run / "curves_diag.png"
    fig2.savefig(diag_out, dpi=140)
    print(f"wrote {diag_out}")

    # ── DEBUG DAGGER (<run>/debug_dagger.png) — the loop's own machinery ─────
    #
    # Everything here is a question about the RUN rather than about the policy:
    # is the learner's state distribution moving, which way are the failures
    # going, did the pin hold, is the aggregate growing, is the refit healthy.
    # They shared a grid with the results until now and lost every time.
    fig4, dx = plt.subplots(3, 4, figsize=(21, 12), squeeze=False)

    # Row 1 — collection, per commanded direction. DAgger's state distribution
    # shifting is the thing that moves BEFORE eval success does, and it can move
    # for one direction and not another; pooled, that is invisible.
    for j in range(4):
        a = dx[0][j]
        if j < len(bins):
            b = bins[j]
            _panel_collection(a, ctx, b=b, legend=(j == 0),
                              title=f"{_D.BIN_SHORT[b]} — collection: how far "
                                    f"the LEARNER got")
        else:
            a.axis("off")

    # Row 2 — HOW CLOSE THE COLLECTION EPISODES ACTUALLY GOT, per direction. The
    # training figure's third column asks this of the EVAL episodes; this asks it
    # of the DAgger data, which is the population that decides what D contains.
    #
    # It is the panel that reads run 12 at a glance. `eval_min_pos` stayed near
    # 0.105 m there while collection stalled at 0.131 m against run 11's 0.049 m,
    # and the eval figure cannot show that because eval never had DART in it. If
    # this row flattens, D has stopped gaining states near the grasp and no
    # amount of further iteration will help.
    for j in range(4):
        a = dx[1][j]
        if j < len(bins):
            b = bins[j]
            _panel_collect_error(
                a, ctx, b=b, legend=(j == 0),
                title=f"{_D.BIN_SHORT[b]} — COLLECTION: closest EE -> "
                      f"{ctx.TGT}")
        else:
            a.axis("off")

    # Row 3 — the pooled outcome stack, then the three run-machinery panels that
    # used to sit on the main grid. The per-bin outcome rows moved to
    # training_curve.png, where the success rate they explain lives.
    _panel_outcomes(dx[2][0], ctx, title="eval outcomes (all bins, fraction of "
                                         "the eval set)")

    # Did a revisited scene still aim at the same grasp? The pin enforces it;
    # this checks it actually held. `grasp_mismatch` must stay 0 — a nonzero bar
    # means D holds contradictory labels for one scene.
    a = dx[2][1]
    for key, label, col in (("revisits", "scenes revisited", "tab:blue"),
                            ("grasp_mismatch", "grasp CHANGED", "tab:red")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, "-o", ms=3, color=col, label=label)
    a.set_ylabel("episodes", fontsize=8)
    a2 = a.twinx()
    dr = num("max_grasp_drift")
    if _finite(dr):
        _plot(a2, it, dr, "--s", ms=3, color="tab:purple", label="max drift (m)")
    a2.set_ylabel("max drift (m)", fontsize=8, color="tab:purple")
    a2.tick_params(axis="y", labelcolor="tab:purple", labelsize=7)
    _grid(a, "same grasp per scene? (should be 0 changed)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    # |D| in steps and the on-policy share. Success plotted against this is what
    # separates "DAgger helps" from "more data helps". NOTE the slope changes if
    # `m` changed mid-run.
    a = dx[2][2]
    ds = num("D_steps")
    if _finite(ds):
        _plot(a, it, ds, "-o", ms=3, color="tab:blue", label="|D| (steps)")
    a.set_ylabel("labelled steps in D", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    fr = num("D_dagger_frac")
    if _finite(fr):
        _plot(a2, it, fr, "-s", ms=3, color="tab:orange", label="on-policy share")
    a2.set_ylabel("DAgger fraction of D", fontsize=8, color="tab:orange")
    a2.set_ylim(0, 1)
    a2.tick_params(axis="y", labelcolor="tab:orange", labelsize=7)
    _grid(a, "the aggregate D the refit sees")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    # Under FTL each point is a fresh fit on a bigger, more diverse D, so a
    # RISING train loss means the aggregate is becoming self-inconsistent —
    # the one failure DAgger cannot average away.
    a = dx[2][3]
    for key, label, style in (("train_loss", "train", "-"),
                              ("val_loss", "val", "--"),
                              ("best_val_loss", "best val", ":")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, style, marker="o", ms=3, label=label)
    a.set_yscale("log")
    a.set_ylabel("loss (log)", fontsize=8)
    a2 = a.twinx()
    for key, label, style in (("train_grip_acc", "train grip acc", "-"),
                              ("val_grip_acc", "val grip acc", "--")):
        ys = num(key)
        if _finite(ys):
            _plot(a2, it, ys, style, marker="^", ms=3, color="tab:green",
                  label=label, alpha=0.7 if key == "val_grip_acc" else 1.0)
    a2.set_ylabel("gripper acc", fontsize=8, color="tab:green")
    a2.set_ylim(0, 1.02)
    a2.tick_params(axis="y", labelcolor="tab:green", labelsize=7)
    _grid(a, "the refit on the growing aggregate")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")

    _fix_x(fig4, it)
    fig4.suptitle(f"Regrasp DAgger machinery — {run.name}", fontsize=12)
    fig4.tight_layout(rect=[0, 0, 1, 0.975])
    debug_out = run / "debug_dagger.png"
    fig4.savefig(debug_out, dpi=140)
    print(f"wrote {debug_out}")

    # ── REGRASP (<run>/curves_regrasp.png) — "is the conditioning doing anything" ──
    # Its own figure rather than more panels on the training curve, because these
    # answer a different question and are blank for every Phase-4 run.
    if _finite(num("dir_track")) or _finite(num("retry_at_2")):
        c_get = ctx.get       # legacy-aware column reader, as the panels use
        _BIN_COLOURS = [_BIN_COLOURS_BY_BIN[b] for b in bins]
        fig3, ax3 = plt.subplots(2, 3, figsize=(19, 9))

        # THE panel, and the one to read first. `dir_track` is 1 - mean(dir_err)
        # / 90 deg: 1 = the gripper ends on the axis it was told to come in on,
        # 0 = it ignores the command. `bin_diag_rate` is the discrete version —
        # how often the REALISED bin is the commanded one — and `bin_hit_rate`
        # asks about the SIDE the gripper arrived from rather than its
        # orientation. Chance for both is 1/4 on this dataset (four live bins),
        # drawn as the dotted line, and a rate sitting ON it means the policy is
        # orienting freely and the conditioning is inert.
        a = ax3[0][0]
        for key, label, style, col, lw in (
                ("dir_track", "dir_track (1 - mean dir_err / 90 deg)", "-",
                 "tab:purple", 2.2),
                ("bin_diag_rate", "realised bin == commanded bin", "-",
                 "tab:blue", 1.6),
                ("bin_hit_rate", "arrived from the commanded side", "--",
                 "tab:green", 1.4),
                ("cond_sep", "cond_sep (what it DID / what it was TOLD)", "-.",
                 "tab:orange", 1.4)):
            ys = num(key)
            if _finite(ys):
                _plot(a, it, ys, style, marker="o", ms=3, color=col, label=label,
                      lw=lw)
        a.axhline(0.25, color="0.6", ls=":", lw=1.0)
        # BELOW the line, not above: `bin_hit_rate` sits within a few points of
        # chance in every run so far, so a label above 0.25 lands on the curve
        # it is annotating.
        a.text(0.01, 0.235, "chance (4 live bins)", fontsize=7, color="0.4",
               va="top", transform=a.get_yaxis_transform())
        a.set_ylim(-0.02, 1.3)
        _grid(a, "is the policy USING the commanded direction?",
              ylabel="ratio / fraction")
        _legend(a, loc="upper left")

        # retry@k — the regrasping headline. Derived from the same episodes, so
        # it costs nothing; it assumes each retry restarts from home, which is
        # true of this evaluation and not of a real deployment. Read as a ceiling.
        a = ax3[0][1]
        for k, col in zip(range(1, 5), ("tab:blue", "tab:green", "tab:orange",
                                        "tab:red")):
            ys = num(f"retry_at_{k}")
            if _finite(ys):
                _plot(a, it, ys, "-", marker="o", ms=3, lw=1.6 + 0.2 * k,
                      color=col, label=f"success @ {k} attempt(s){_rung(num, k)}")
        a.set_ylim(-0.02, 1.02)
        _grid(a, "regrasping: success with k tries", ylabel="fraction of eval scenes")
        _legend(a, loc="lower right")

        # ---- COMMANDED BIN vs REALISED BIN, one curve per commanded bin ----
        # THE literal question: told to come in from `+z`, what fraction of
        # episodes actually ENDED with the gripper's approach axis in `+z`?
        # `bin_diag_rate_b{b}` is the b-th diagonal entry of the confusion matrix
        # normalised by that bin's episode count, so each curve is a per-bin
        # accuracy against a chance level of 1/4.
        #
        # Reading it per bin rather than pooled is the point. A pooled 0.50
        # is consistent with two very different policies: one that follows all
        # four commands half the time, and one that nails `+x` (which is 31% of
        # the eval episodes and the easiest direction) while ignoring the rest.
        # Only the second is a reason to change the architecture, and only this
        # panel tells them apart.
        a = ax3[0][2]
        for b, col in zip(bins, _BIN_COLOURS):
            ys = c_get(f"bin_diag_rate_b{b}")
            if _finite(ys):
                _plot(a, it, ys, "-", marker="o", ms=3, lw=1.8, color=col,
                      label=_D.BIN_SHORT[b])
        a.axhline(0.25, color="0.6", ls=":", lw=1.0)
        a.text(0.01, 0.235, "chance (4 live bins)", fontsize=7, color="0.4",
               va="top", transform=a.get_yaxis_transform())
        a.set_ylim(-0.02, 1.02)
        _note_empty(a)
        _grid(a, "ended in the COMMANDED bin (per bin)",
              ylabel="fraction of that bin's episodes")
        _legend(a, loc="upper left", ncol=2)

        # ...and the same question about WHICH SIDE the gripper came from rather
        # than which way it pointed. A gripper can be correctly oriented on the
        # wrong side of the object and vice versa, so the two panels fail
        # independently: orientation right / side wrong is what run 1 measured
        # pooled (bin_diag ~0.50, bin_hit at chance), and per bin is where it
        # becomes actionable.
        a = ax3[1][2]
        for b, col in zip(bins, _BIN_COLOURS):
            ys = c_get(f"bin_hit_rate_b{b}")
            if _finite(ys):
                _plot(a, it, ys, "-", marker="o", ms=3, lw=1.8, color=col,
                      label=_D.BIN_SHORT[b])
        a.axhline(0.25, color="0.6", ls=":", lw=1.0)
        a.set_ylim(-0.02, 1.02)
        _note_empty(a)
        _grid(a, "arrived from the COMMANDED side (per bin)",
              ylabel="fraction of that bin's episodes")
        _legend(a, loc="upper left", ncol=2)

        # Per-BIN success, not per-slot. Slot k means "this scene's k-th chosen
        # direction" and is not comparable across scenes; bin k is a fixed
        # physical direction and is. The spread between these curves is how much
        # the direction matters — if `+x` and `+z` separate, the retry ladder has
        # something to work with; if they sit on top of each other, retrying is
        # just four draws from one distribution.
        a = ax3[1][0]
        for b, col in zip(bins, _BIN_COLOURS):
            ys = c_get(f"success_rate_b{b}")       # falls back to run 1's succ_bin_
            if _finite(ys):
                _plot(a, it, ys, "-", marker="o", ms=3, lw=1.6, color=col,
                      label=_D.BIN_SHORT[b])
        a.set_ylim(-0.02, 1.02)
        _grid(a, "success per commanded direction",
              ylabel="fraction of that bin's episodes")
        _legend(a, loc="upper left", ncol=2)

        # ...and whether it FOLLOWED each direction, which is the other half. A
        # bin can succeed because the policy ignored it and did the easy thing;
        # that shows up as a high success rate on the left with a dir_track here
        # no better than the bins it is beating.
        a = ax3[1][1]
        for b, col in zip(bins, _BIN_COLOURS):
            ys = c_get(f"dir_track_b{b}")
            if _finite(ys):
                _plot(a, it, ys, "-", marker="o", ms=3, lw=1.6, color=col,
                      label=_D.BIN_SHORT[b])
        a.set_ylim(-0.02, 1.02)
        _note_empty(a)
        _grid(a, "direction tracking per bin (1 - dir_err / 90 deg)",
              ylabel="ratio")
        _legend(a, loc="upper left", ncol=2)

        _fix_x(fig3, it)
        fig3.suptitle(f"Regrasp conditioning — {run.name}", fontsize=12)
        fig3.tight_layout(rect=[0, 0, 1, 0.96])
        regrasp_out = run / "curves_regrasp.png"
        fig3.savefig(regrasp_out, dpi=140)
        print(f"wrote {regrasp_out}")

    # ── MEDIA (<run>/media_curves.png) — the presentation cut ────────────────
    #
    # The same data and the same panel functions, POOLED over directions and
    # trimmed to the five that carry the story for an outside reader. Nothing
    # here is new; what it buys is a figure that fits on a slide, which the 4x3
    # per-bin grid above does not. Dropped on purpose: the per-bin split (an
    # internal question), the pin-consistency check (a correctness check, not a
    # result), |D| growth, and the refit loss.
    fig5 = plt.figure(figsize=(16.5, 8))
    gs = fig5.add_gridspec(2, 6)
    mx = [[fig5.add_subplot(gs[0, 0:2]), fig5.add_subplot(gs[0, 2:4]),
           fig5.add_subplot(gs[0, 4:6])],
          [fig5.add_subplot(gs[1, 1:3]), fig5.add_subplot(gs[1, 3:5])]]
    _panel_nested(mx[0][0], ctx, title="Eval: the nested success rates")
    _panel_opportunity(mx[0][1], ctx, lean=True,
                       title="Approach opportunity vs grasp conversion")
    _panel_approach(mx[0][2], ctx, title=f"Approach error to the {ctx.TGT}")
    _panel_outcomes(mx[1][0], ctx, title="Eval outcomes (fraction of episodes)")
    _panel_collection(mx[1][1], ctx, lean=True, title="Data collection")

    _fix_x(fig5, it)
    fig5.suptitle(f"Regrasp — {run.name}", fontsize=12)
    fig5.tight_layout(rect=[0, 0, 1, 0.96])
    media_out = run / "media_curves.png"
    fig5.savefig(media_out, dpi=140)
    print(f"wrote {media_out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
