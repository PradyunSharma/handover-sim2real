"""
Presentation-facing version of examples/plot_dagger_run.py.

    python examples/plot_dagger_media.py output/dagger_runs/dagger4_run16
    python examples/plot_dagger_media.py output/dagger_runs/dagger4_run16 --show

Same data and the same CSV reader as plot_dagger_run.py, re-labelled for an
outside reader and trimmed to the five panels that carry the story. The
diagnostic figure is not reproduced here — use plot_dagger_run.py for that.

Writes to output/media/<run>_curves.png rather than into the run directory, so
regenerating figures for a talk never touches a run's own artefacts and the
figures for several runs sit side by side.

WHAT CHANGED versus plot_dagger_run.py, and why each panel is worth keeping:
  • nested rates      — grasp commanded / near the labelled pose / grasp
                        succeeded / success. The VERTICAL GAPS are the
                        diagnosis, which is why all four stay.
  • approach vs grasp — the geometric opportunity (object really between the
                        open jaws) and its conversion. The pinned-pose
                        reference pair and `mean jaw occupancy` are dropped:
                        the first measures agreement with the pin rather than
                        opportunity, and the second tracks `box_chance_rate` so
                        closely in these runs that it reads as a duplicate line.
                        `success | closed` goes too — it is a fourth
                        conditioning on the same episodes and crowds the panel.
  • approach metrics  — position (left axis) and rotation (right) error, closest
                        reached over the episode and at the moment the grasp was
                        commanded, plus the auxiliary head's prediction error
                        when the run has one.
  • eval outcomes     — the failure-mode breakdown as a stacked area.
  • data collection   — how far the learner's own rollouts got. `closed on its
                        own` is dropped: it is a collection-side counter that
                        needs beta to interpret, which this figure no longer
                        shows.
Dropped entirely: grasp-consistency (a correctness check on the pin, not a
result), dataset growth, and refit loss — all three are run-machinery panels
that need the diagnostic figure's context to mean anything.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")  # headless-safe; overridden by --show below
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _load(log_path: Path, eval_log: Path | None = None):
    cols: dict[str, list] = {}
    with log_path.open() as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)

    n_rows = len(cols.get("iter", []))

    # With EVAL.every: 0 the loop leaves every eval column blank and the metrics
    # live in eval_log.csv instead (examples/eval_dagger_run.py, a separate job).
    # Splice them in by iteration so the plots look the same either way. Only
    # BLANK cells are filled — an in-loop number always wins, so a run that did
    # both never has its own numbers overwritten by a re-scored pool.
    if eval_log is not None and eval_log.exists():
        with eval_log.open() as f:
            by_iter = {r["iter"]: r for r in csv.DictReader(f) if r.get("iter")}
        if by_iter:
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
                  f"{eval_log.name}")

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


def _grid(ax, title, xlabel="Training iteration", ylabel=None):
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
    """Iterations are integers, and a panel whose series are all-NaN (no grasp
    commands yet, say) would otherwise fall back to a 0..1 axis that silently
    disagrees with its neighbours."""
    lo, hi = min(it), max(it)
    pad = 0.2 if hi > lo else 0.5
    for a in fig.axes:
        a.set_xlim(lo - pad, hi + pad)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="output/dagger_runs/<name>")
    p.add_argument("--show", action="store_true")
    p.add_argument("--out", default=None,
                   help="output PNG (default output/media/<run>_curves.png)")
    p.add_argument("--pos-thresh", type=float, default=0.02)
    p.add_argument("--rot-thresh", type=float, default=0.34)
    args = p.parse_args()

    run = Path(args.run_dir)
    log_path = run / "dagger_log.csv"
    if not log_path.exists():
        raise SystemExit(f"no log at {log_path}")

    num, n = _load(log_path, eval_log=run / "eval_log.csv")
    if n == 0:
        raise SystemExit(f"{log_path} has no rows yet")
    it = num("iter")

    # Five panels: three on top, two below. A plain 2x3 would leave the bottom
    # row hanging off the left with a hole on the right, so the grid is laid out
    # in SIX half-width columns and each panel spans two of them — the top row
    # from 0, the bottom row offset by one, which centres it under the top.
    fig = plt.figure(figsize=(16.5, 8))
    gs = fig.add_gridspec(2, 6)
    ax = [[fig.add_subplot(gs[0, 0:2]),
           fig.add_subplot(gs[0, 2:4]),
           fig.add_subplot(gs[0, 4:6])],
          [fig.add_subplot(gs[1, 1:3]),
           fig.add_subplot(gs[1, 3:5])]]

    # ---- the nested rates ----------------------------------------------------
    a = ax[0][0]
    for key, label, style in (("close_rate", "Grasp commanded", ":"),
                              ("near_rate", "Near labelled grasp pose", "-."),
                              ("grasp_rate", "Grasp succeeded", "--"),
                              ("success_rate", "Handover success", "-")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, style, marker="o", ms=3, label=label,
                  lw=2 if key == "success_rate" else 1.2)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "Eval on training data", ylabel="Fraction of eval scenes")
    _legend(a, loc="upper left")

    # ---- geometric opportunity vs its conversion ----------------------------
    # The box_* family (dagger/grasp_box.py) asks whether object material was
    # really between the open jaws, so an off-pose grasp counts as the
    # opportunity it is — unlike the pinned-pose test, which gates on agreement
    # with the pin and is not drawn here.
    a = ax[0][1]
    for key, label, style, col, lw in (
            ("box_chance_rate", "Object in jaws", "-", "tab:blue", 1.4),
            ("box_taken_rate", "Grasp commanded | In jaws", "-", "tab:green", 2.0),
            ("miss_given_box", "Not grasped | In jaws", "-", "tab:red", 1.4)):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, style, marker="o", ms=3, color=col, label=label, lw=lw)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "Policy's approach success vs Policy's grasp command success",
          ylabel="Fraction")
    # The conversion line sits at ~0.9, the opportunity line at ~0.7 and the miss
    # line at ~0.1, so every corner is occupied. The clear band is the gap
    # between the miss line and the opportunity line — anchor into it explicitly.
    _legend(a, loc="center right", bbox_to_anchor=(1.0, 0.30))

    # ---- approach error: closest reached, and where it was when it committed --
    a = ax[0][2]
    mp, mr = num("eval_min_pos"), num("eval_min_rot")
    pe, re_ = num("mean_pos_err"), num("mean_rot_err")
    if _finite(mp):
        _plot(a, it, mp, "-o", ms=3, color="tab:blue", label="Min pos err (m)")
    if _finite(pe):
        _plot(a, it, pe, "--o", ms=3, color="tab:cyan", alpha=0.8,
              label="Pos err at grasp command (m)")
    # Auxiliary goal-grasp head (run 13 on): how far the network's BELIEF about
    # the grasp is from the pinned pose. Plotted on the same axes as the gripper's
    # actual error on purpose — the comparison is the whole point. If the head
    # predicts the pose accurately while the gripper still arrives far away, the
    # information is present in the features and the action head is not using it;
    # if both are large, the observation cannot support the target at all.
    # Absent (all-NaN) on runs without the head, so nothing changes for those.
    ap = num("aux_pos_mm")
    if _finite(ap):
        _plot(a, it, (np.asarray(ap, dtype=float) / 1000.0).tolist(), ":^", ms=3,
              color="tab:green", alpha=0.9,
              label="Aux: predicted grasp pos err (m)")
    a.axhline(args.pos_thresh, color="tab:blue", ls=":", lw=1,
              label=f"Close thresh {args.pos_thresh} m")
    a.set_ylim(bottom=0)
    a.set_ylabel("Position error (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    if _finite(mr):
        _plot(a2, it, mr, "-s", ms=3, color="tab:red", label="Min rot err (rad)")
    if _finite(re_):
        _plot(a2, it, re_, "--s", ms=3, color="tab:orange", alpha=0.8,
              label="Rot err at grasp command (rad)")
    ar = num("aux_rot_deg")
    if _finite(ar):
        _plot(a2, it, np.radians(np.asarray(ar, dtype=float)).tolist(), ":^", ms=3,
              color="tab:olive", alpha=0.9,
              label="Aux: predicted grasp rot err (rad)")
    a2.axhline(args.rot_thresh, color="tab:red", ls=":", lw=1,
               label=f"Close thresh {args.rot_thresh} rad")
    a2.set_ylim(bottom=0)
    a2.set_ylabel("Rotation error (rad)", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    _grid(a, "Policy's approach metrics")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        # Seven entries, and both error families sit in the upper half of their
        # axes. Two columns keep the box short enough to clear the auxiliary
        # curves, and the right half keeps it off the aux-rotation line's dip at
        # iteration 0.
        a.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right", ncol=2,
                 framealpha=0.9)

    # ---- outcome breakdown ---------------------------------------------------
    a = ax[1][0]
    series = [("f_grasp_ok", "Secured", "tab:green"),
              ("f_grasp_miss", "Closed, not secured", "tab:olive"),
              ("f_no_release", "No release", "tab:orange"),
              ("f_drop", "Drop", "tab:red"),
              ("f_human_contact", "Human contact", "tab:purple"),
              ("f_timeout", "Never closed", "tab:gray")]
    if _stack(a, it, [num(k) for k, _, _ in series],
              [l for _, l, _ in series], [c for _, _, c in series]):
        a.set_ylim(0, 1)
    _grid(a, "Eval outcomes (fraction of scenes)")
    _legend(a, loc="lower left", ncol=2)

    # ---- how far the learner's own rollouts got during collection ------------
    a = ax[1][1]
    eps = num("episodes")
    for key, label in (
            ("reached_standoff", "Reached standoff"),
            ("reached_grasp", "Reached labelled grasp pose and grasp commanded")):
        ys = num(key)
        frac = [(y / e if (e and e > 0 and y == y and y >= 0) else float("nan"))
                for y, e in zip(ys, eps)]
        if _finite(frac):
            _plot(a, it, frac, "-o", ms=3, label=label)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "Data collection", ylabel="Fraction of episodes")
    _legend(a, loc="upper left")

    _fix_x(fig, it)
    fig.tight_layout()

    out = (Path(args.out) if args.out
           else run.parent.parent / "media" / f"{run.name}_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
