"""
What the six Regrasp direction bins ARE, and what the data puts in them.

    python examples/plot_regrasp_bins.py --run regrasp_run21
    python examples/plot_regrasp_bins.py --table output/regrasp_pins_train.json
    python examples/plot_regrasp_bins.py                  # geometry only, no data

THE BINS ARE A PARTITION OF DIRECTIONS, NOT A LIST OF SIX ARROWS, and every
confusion about them comes from reading the arrows and forgetting the cells.
`bin_of` is an argmin over angles, so the six octahedral axes induce six Voronoi
cells of 45 deg half-angle covering the whole sphere: EVERY direction is in a
bin, including one 44 deg off the axis that names it. The top-left panel draws
that partition directly — a few thousand directions, each coloured by the bin it
falls in — because a picture of the cells is the only way to see that the label
is a sector and not a ray.

WHAT THIS IS NOT. `plot_regrasp_bin_spread.py` compares the angular spread of
several pin tables against both the fixed axis and the empirical centroid; this
is a single-table overview of the bin SET. `analyze_demo_bins.py` counts
demonstrations per bin from a shard's detail CSV. Neither draws the geometry.

A NOTE ON THE COLOURS. Bin -> hue is fixed repo-wide (`plot_regrasp_run.py`'s
`_BIN_COLOURS_BY_BIN`, `regrasp/viz.py`'s `BIN_RGB`), so a bin is the same colour
here, in every run figure, and in the PyBullet overlays. That mapping does not
survive a colour-vision check — `+y` green and `-y` orange are ~indistinguishable
under protanopia — so NOTHING HERE IS ENCODED BY COLOUR ALONE: every bin carries
its `+x` / `-x` text on every panel, and the 3-D scatter varies marker shape too.
Repainting is not an option worth taking unilaterally; it would silently
reinterpret every figure already produced.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from handover_sim2real.regrasp import directions as D

# Repo-wide, see the header.
BIN_COLOURS = ("tab:blue", "tab:brown", "tab:green", "tab:orange",
               "tab:red", "tab:purple")
# Secondary encoding for the scatter, so identity survives the CVD failure.
BIN_MARKERS = ("o", "s", "^", "v", "D", "P")


def _load_table(path):
    """(meta, per-grasp rows) from a pin table, or (None, []) if there is none."""
    if not path:
        return None, []
    with open(path) as f:
        raw = json.load(f)
    meta = raw.get("_meta", {})
    rows = []
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        for slot, g in enumerate(v.get("grasps", [])):
            d = g.get("d_anchor")
            if d is None:
                continue
            rows.append((int(k), slot, int(g.get("bin", -1)),
                         np.asarray(d, dtype=np.float64),
                         float(g.get("angle_to_axis_deg", np.nan))))
    return meta, rows


def _sphere(ax, n_points, axis_len=1.0, label_size=12):
    """The partition itself: `n_points` directions, each coloured by its bin.

    POINTS RATHER THAN PATCHES. Drawing six spherical polygons would need the
    cell boundaries computed and projected; sampling the sphere and calling the
    SAME `bin_of` the collector calls draws whatever the code actually does,
    including any future `k`. If the picture and the assignment ever disagree,
    that is a real bug rather than a drawing artefact.
    """
    v = D.fibonacci_directions(int(n_points))
    b = np.array([D.bin_of(x) for x in v])
    for i in range(len(D.BINS)):
        m = b == i
        # depthshade off: it darkens the far hemisphere, and a cell that
        # changes shade around the sphere reads as two different cells.
        ax.scatter(v[m, 0], v[m, 1], v[m, 2], s=4, alpha=0.6,
                   color=BIN_COLOURS[i], linewidths=0, depthshade=False,
                   zorder=1)

    for i, a in enumerate(D.BINS):
        ax.quiver(0, 0, 0, *(a * axis_len * 1.28), color=BIN_COLOURS[i],
                  arrow_length_ratio=0.12, linewidth=2.0, zorder=6)
        ax.text(*(a * (axis_len * 1.28 + 0.26)), D.BIN_SHORT[i],
                color=BIN_COLOURS[i], fontsize=label_size,
                ha="center", va="center", fontweight="bold", zorder=12,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                          pad=0.8))
    _cube(ax)


def _cube(ax, r=1.62):
    """A bare 3-D frame: no panes, no ticks, no box.

    `set_axis_off` rather than clearing the ticks one by one — mplot3d draws
    the pane EDGES regardless of pane alpha, which came out as a black polygon
    behind the sphere.

    NO SEPARATE `anchor x/y/z` LABELS, deliberately. The bin axes ARE the anchor
    axes — bin `+x` is the anchor frame's +x by construction — so a second set
    of captions on the same three rays says nothing and collides with the first.
    The panel title carries the frame instead.
    """
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    # Off the default (-60): at -60 the +y axis runs almost straight into the
    # screen and its label lands on the sphere's silhouette.
    ax.view_init(elev=20, azim=-38)


def _panel_cells(ax, n_points):
    _sphere(ax, n_points)
    # 45 deg is not a constant to look up: with every pair of the octahedral
    # set 90 deg apart, the Voronoi half-angle is half the pair separation.
    half = 0.5 * float(np.min(D.angle_between(D.BINS[0], D.BINS[1:])))
    ax.set_title("the six bins are a PARTITION — drawn in the ANCHOR frame\n"
                 f"{len(D.BINS)} cells, {half:.0f}° half-angle, "
                 "whole sphere covered", fontsize=10, pad=2)


def _panel_data(ax, rows, n_points):
    """Where a table's pinned grasps actually sit inside those cells."""
    _cube(ax)
    for i, a in enumerate(D.BINS):
        ax.quiver(0, 0, 0, *(a * 1.28), color=BIN_COLOURS[i],
                  arrow_length_ratio=0.12, linewidth=1.5, alpha=0.5, zorder=6)
        ax.text(*(a * 1.54), D.BIN_SHORT[i], color=BIN_COLOURS[i],
                fontsize=12, ha="center", va="center", fontweight="bold",
                zorder=12,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                          pad=0.8))
    for i in range(len(D.BINS)):
        v = np.array([r[3] for r in rows if r[2] == i])
        if not len(v):
            continue
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=14, alpha=0.7,
                   color=BIN_COLOURS[i], marker=BIN_MARKERS[i],
                   linewidths=0, zorder=2)
    ax.set_title(f"where the {len(rows)} pinned grasps sit\n"
                 "one point per demonstration, at its own `d`",
                 fontsize=10, pad=2)


def _bars(ax, counts, title, note=None):
    """One measure, one axis, values written on the bars.

    Horizontal so the bin labels read left-to-right at full size — they are the
    secondary encoding that carries identity when the hues do not.
    """
    y = np.arange(len(D.BINS))
    ax.barh(y, counts, color=[BIN_COLOURS[i] for i in range(len(D.BINS))],
            height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([D.BIN_SHORT[i] for i in range(len(D.BINS))],
                       fontsize=10)
    ax.invert_yaxis()
    hi = max(1.0, float(np.max(counts)))
    for i, c in enumerate(counts):
        ax.text(c + hi * 0.02, i, f"{int(c)}", va="center", fontsize=9,
                color="0.25")
    ax.set_xlim(0, hi * 1.18)
    ax.set_title(title, fontsize=10, pad=4)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if note:
        ax.set_xlabel(note, fontsize=8, color="0.4")


def _panel_angles(ax, rows, max_angle):
    """How far inside its cell each demonstration sits.

    THE LABEL IS A SECTOR AND THE DEMO IS ONE POINT IN IT — this is the size of
    that gap, and it is honest label noise rather than error: the policy is
    commanded the bin AXIS (`retry.next_direction` has no grasp to read) and
    shown a grasp up to `max_angle_deg` away from it.
    """
    edges = np.linspace(0, 45, 24)
    any_drawn = False
    for i in range(len(D.BINS)):
        a = np.array([r[4] for r in rows if r[2] == i and np.isfinite(r[4])])
        if not len(a):
            continue
        ax.hist(a, bins=edges, histtype="step", linewidth=1.6,
                color=BIN_COLOURS[i], label=f"{D.BIN_SHORT[i]}  (n={len(a)})")
        any_drawn = True
    # ANNOTATED AT THE FLOOR, not the ceiling: the distributions peak at the
    # left and the two thresholds sit at 30 and 45, so the top of the axis is
    # where the legend has to go and the bottom is empty.
    lo = ax.get_ylim()[0]
    ax.axvline(D.BIN_HIT_DEG, color="0.35", linestyle="--", linewidth=1.2)
    ax.text(D.BIN_HIT_DEG - 1.6, lo, f"BIN_HIT_DEG {D.BIN_HIT_DEG:g}° ",
            fontsize=7.5, color="0.35", va="bottom", ha="right")
    if max_angle:
        ax.axvline(max_angle, color="0.15", linewidth=1.2)
        ax.text(max_angle - 0.6, lo, f"table cut {max_angle:g}° ",
                fontsize=7.5, color="0.15", va="bottom", ha="right")
    ax.set_xlabel("angle from the grasp's `d` to the bin axis it was assigned "
                  "(deg)", fontsize=8.5)
    ax.set_ylabel("grasps", fontsize=8.5)
    ax.set_title("the label is a SECTOR, the demo is one point in it",
                 fontsize=10, pad=4)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if any_drawn:
        ax.legend(fontsize=7.5, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), frameon=True, framealpha=0.92,
                  edgecolor="none", handlelength=1.4, columnspacing=1.2)


def _panel_text(ax, meta, rows, table):
    ax.axis("off")
    live = [i for i in range(len(D.BINS))
            if any(r[2] == i for r in rows)] if rows else list(D.LIVE_BINS)
    lines = [
        "THE BIN SET",
        f"  k = {len(D.BINS)}   octahedral, {D.BIN_SHORT[:len(D.BINS)]}",
        "  every pair 90° apart -> Voronoi half-angle 45°",
        f"  BIN_HIT_DEG = {D.BIN_HIT_DEG:g}°  (margin against boundary noise)",
        f"  NEIGHBOUR_DEG = {D.NEIGHBOUR_DEG:g}°  -> `neighbours` returns only",
        "     the bin itself, so the retry ladder stops only after all six",
        "",
        "ORDER IS LOAD-BEARING",
        "  `bin` is stored in every episode attr and keyed in the grasp",
        "  registry. Append, never insert.",
        "",
        f"LIVE BINS HERE: {', '.join(D.BIN_SHORT[i] for i in live)}"
        f"   -> chance 1/{max(len(live), 1)} = {1.0 / max(len(live), 1):.3f}",
    ]
    if meta:
        lines += [
            "",
            f"TABLE  {os.path.basename(table)}",
            f"  d_rule        {meta.get('d_rule')}",
            f"  max_angle_deg {meta.get('max_angle_deg')}",
            f"  centroid      {meta.get('centroid_source')}",
            f"  hand ref      {meta.get('anchor_hand_ref')}",
            f"  scenes {meta.get('n_scenes')}  ok {meta.get('n_ok')}  "
            f"demos {meta.get('n_demos')}",
        ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.2,
            family="monospace", color="0.15", transform=ax.transAxes)


def main():
    p = argparse.ArgumentParser(
        description="Visualise the Regrasp direction bins and a table's "
                    "occupancy of them.")
    p.add_argument("--run", default=None,
                   help="Regrasp run name — derives the pin table from the "
                        "run's own config snapshot")
    p.add_argument("--run-root", default=None)
    p.add_argument("--table", default=None,
                   help="pin table JSON. Implied by --run; omit both for the "
                        "geometry alone.")
    p.add_argument("--out", default=None,
                   help="output PNG (default output/regrasp_bins.png, or "
                        "output/regrasp_bins_<run>.png with --run)")
    p.add_argument("--points", type=int, default=6000,
                   help="directions sampled to paint the cells (default 6000)")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    table = args.table
    if args.run and not table:
        from handover_sim2real.regrasp import resolve_run
        spec = resolve_run(args.run, run_root=args.run_root)
        print(spec.describe())
        table = spec.pin_table
    out = args.out or (f"output/regrasp_bins_{args.run}.png" if args.run
                       else "output/regrasp_bins.png")

    meta, rows = _load_table(table)
    if table:
        print(f"[table] {table}: {len(rows)} pinned grasps")

    if not rows:
        # Geometry only — the two data panels would be blank, and a blank panel
        # reads as "no data in these bins" rather than "no table given".
        fig = plt.figure(figsize=(11.5, 4.8))
        _panel_cells(fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False), args.points)
        _panel_text(fig.add_subplot(1, 2, 2), meta, rows, table or "")
    else:
        fig = plt.figure(figsize=(16.5, 9.6))
        _panel_cells(fig.add_subplot(2, 3, 1, projection="3d", computed_zorder=False), args.points)
        _panel_data(fig.add_subplot(2, 3, 2, projection="3d", computed_zorder=False), rows,
                    args.points)
        _panel_angles(fig.add_subplot(2, 3, 3), rows,
                      float(meta.get("max_angle_deg") or 0.0))

        gs = meta.get("goal_set_bin_histogram")
        if gs:
            _bars(fig.add_subplot(2, 3, 4), np.asarray(gs, dtype=float),
                  "candidate grasps in the goal sets",
                  "before pinning — what the scenes could demonstrate")
        dpb = meta.get("demos_per_bin")
        if dpb is None:
            dpb = [sum(1 for r in rows if r[2] == i)
                   for i in range(len(D.BINS))]
        _bars(fig.add_subplot(2, 3, 5), np.asarray(dpb, dtype=float),
              "pinned demonstrations",
              "what the shard actually contains")
        _panel_text(fig.add_subplot(2, 3, 6), meta, rows, table or "")

    fig.suptitle("Regrasp direction bins"
                 + (f" — {args.run}" if args.run else ""),
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}  ({fig.get_size_inches()[0]:.0f}x"
          f"{fig.get_size_inches()[1]:.0f} in @ {args.dpi} dpi)")


if __name__ == "__main__":
    main()
