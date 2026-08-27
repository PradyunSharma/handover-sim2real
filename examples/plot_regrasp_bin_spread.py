"""
How tightly do a bin's assigned grasps cluster, and around WHAT.

    python examples/plot_regrasp_bin_spread.py \
        output/regrasp_pins_train.json output/regrasp_pins_train_p3.json

    python examples/plot_regrasp_bin_spread.py output/regrasp_pins_*.json \
        --labels run2,run3 --out output/bin_spread.png

WHAT THIS MEASURES, AND WHY IT IS TWO QUESTIONS RATHER THAN ONE.

The policy is commanded a bin's FIXED AXIS — `to_world(BINS[b], anchor_R)` —
because at deployment `retry.next_direction` has no grasp to read. The
demonstrations for that bin are the goal-set grasps NEAREST that axis, not
grasps ON it. The residual angle is honest label noise: the policy is told a
sector and shown one grasp inside it.

  ROW 1, angle to the FIXED BIN AXIS. The label noise the conditioning has to
  tolerate, and the quantity `demo_off_deg` records per episode. This is what
  train/deploy skew means here.

  ROW 2, angle to the bin's EMPIRICAL CENTROID (the normalised mean of the
  `d_anchor` actually assigned to it). Same members, different reference. It
  answers "how tight is the cluster" WITHOUT the offset — because a bin whose
  members sit 20 deg off-axis but 5 deg from each other is a very different
  object from one scattered 20 deg around the axis, and row 1 alone cannot tell
  them apart.

The printed BIAS column is the angle between those two references, and it is the
number the two rows exist to separate. A large bias with a small row-2 spread
means the assignment is systematically pulling to one side of the sector — the
policy would then be commanded a direction the data never demonstrates, which no
amount of training fixes and which a per-bin `dir_err` would report as a floor it
cannot get under.

WHY COMPARE run 2's TABLE WITH run 3's. `--per-bin 3` keeps the three nearest
members instead of one, so it necessarily reaches FURTHER from the axis. How much
further is the cost of the extra demonstrations, and it is the quantity behind
run 3's mode-averaging risk: three targets under one command are resolved by a
unimodal `pm` loss by averaging, and the wider they sit the less the mean is a
grasp. Run 3's own header says to read the within-bin spread before trusting the
extra members.

Reads the pin tables only — no simulator, no GPU, seconds. `d_anchor` and
`angle_to_axis_deg` are written by assign_direction_demos.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handover_sim2real.regrasp import directions as D          # noqa: E402


def bin_vectors(path: Path):
    """{bin_idx: (N, 3) d_anchor} for one pin table, plus its scene count.

    `d_anchor` is stored per grasp, so nothing is re-derived here — the vectors
    are exactly what the assignment committed to and what the collector flew.
    """
    t = json.load(open(path))
    per: dict[int, list] = {}
    scenes = 0
    for k, v in t.items():
        if k.startswith("_"):
            continue
        scenes += 1
        for g in v.get("grasps", []):
            b, d = g.get("bin"), g.get("d_anchor")
            if b is None or d is None:
                continue
            per.setdefault(int(b), []).append(np.asarray(d, dtype=np.float64))
    return {b: np.asarray(v) for b, v in sorted(per.items())}, scenes


def _draw(a, datas, colours, labels, edges) -> None:
    """All tables' distributions in ONE panel, as SIDE-BY-SIDE filled bars.

    Two choices here, both forced by what these histograms actually look like.

    GROUPED, NOT OVERLAID. The distributions being compared overlap almost
    completely — the medians differ by 1-3 deg — so translucent filled bars blend
    over most of their range into a third colour matching NEITHER legend swatch.
    Passing the datasets to `hist` TOGETHER makes matplotlib split each bin into
    one solid bar per table, so every bar is exactly the colour its legend entry
    shows and nothing overlaps. That is why this takes all the data at once
    rather than being called per table.

    DENSITY. run 3's table holds ~3x run 2's grasps (1404 vs 490 in `+x`), so on
    a raw-count axis run 2's bars would sit in run 3's shadow. The question is
    whether the SHAPE moved, not how many rows each file has.
    """
    keep = [(d, c, l) for d, c, l in zip(datas, colours, labels)
            if d is not None and len(d)]
    if not keep:
        return
    a.hist([d for d, _, _ in keep], bins=edges, density=True,
           color=[c for _, c, _ in keep], label=[l for _, _, l in keep])
    for d, c, _ in keep:
        a.axvline(float(np.median(d)), color=c, ls="--", lw=1.2)


def spreads(vecs: np.ndarray, b: int):
    """(angles to the fixed bin axis, angles to the empirical centroid, bias)."""
    axis = D.BINS[b]
    centroid = D.normalize(vecs.mean(axis=0))
    to_axis = np.array([float(D.angle_between(v, axis)) for v in vecs])
    to_cent = np.array([float(D.angle_between(v, centroid)) for v in vecs])
    return to_axis, to_cent, float(D.angle_between(centroid, axis))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", nargs="+", help="regrasp_pins_<split>[_p3].json")
    p.add_argument("--labels", default=None,
                   help="comma list, one per table (default: file stems)")
    p.add_argument("--out", default="output/bin_spread.png")
    p.add_argument("--bins-deg", type=float, default=2.5,
                   help="histogram bin width in degrees (default 2.5)")
    args = p.parse_args()

    tables = [Path(t) for t in args.tables]
    labels = (args.labels.split(",") if args.labels
              else [t.stem.replace("regrasp_pins_", "") for t in tables])
    if len(labels) != len(tables):
        raise SystemExit(f"{len(labels)} labels for {len(tables)} tables")

    loaded = [bin_vectors(t) for t in tables]

    # Only bins that ANY table populates. On this dataset that is four of six:
    # `-z` is geometrically impossible above a table and `-x` is what the hand
    # collision filter removes, so both are empty by construction rather than by
    # accident, and plotting them would be four empty panels.
    live = sorted({b for per, _ in loaded for b in per})
    ncol = len(live) + 1                                    # + a pooled column
    fig, ax = plt.subplots(2, ncol, figsize=(3.6 * ncol, 6.4), squeeze=False)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    edges = np.arange(0.0, 46.0 + args.bins_deg, args.bins_deg)

    print(f"\n{'table':<10} {'bin':>4} {'n':>6} {'median':>7} {'p90':>7} "
          f"{'max':>7} | {'med':>7} {'p90':>7} | {'bias':>6}")
    print(f"{'':<10} {'':>4} {'':>6} {'--- to fixed axis ---':>23} | "
          f"{'to centroid':>15} | {'':>6}")

    # Every table's data is gathered per panel BEFORE anything is drawn, because
    # grouped bars need all the datasets in one `hist` call (see `_draw`).
    pooled = [[[], []] for _ in tables]
    for col, b in enumerate(live):
        panel = [[None] * len(tables), [None] * len(tables)]
        legend = [[None] * len(tables), [None] * len(tables)]
        for ti, ((per, nsc), lab) in enumerate(zip(loaded, labels)):
            v = per.get(b)
            if v is None or not len(v):
                continue
            to_axis, to_cent, bias = spreads(v, b)
            pooled[ti][0].append(to_axis)
            pooled[ti][1].append(to_cent)
            print(f"{lab:<10} {D.BIN_SHORT[b]:>4} {len(v):>6} "
                  f"{np.median(to_axis):>7.1f} {np.percentile(to_axis, 90):>7.1f} "
                  f"{to_axis.max():>7.1f} | {np.median(to_cent):>7.1f} "
                  f"{np.percentile(to_cent, 90):>7.1f} | {bias:>6.1f}")
            for row, data in ((0, to_axis), (1, to_cent)):
                panel[row][ti] = data
                legend[row][ti] = (f"{lab} (n={len(v)}, "
                                   f"med {np.median(data):.1f}°)")
        for row, ref in ((0, "fixed bin axis"), (1, "bin centroid")):
            a = ax[row][col]
            _draw(a, panel[row], colours, legend[row], edges)
            a.set_title(f"{D.BIN_SHORT[b]} — to {ref}", fontsize=10)
            a.set_xlabel("angle (deg)", fontsize=8)
            a.set_ylabel("density", fontsize=8)
            a.tick_params(labelsize=7)
            a.grid(alpha=0.3, lw=0.5)
            a.legend(fontsize=7)

    for row, ref in ((0, "fixed bin axis"), (1, "bin centroid")):
        allv = [np.concatenate(pooled[ti][row]) if pooled[ti][row] else None
                for ti in range(len(tables))]
        _draw(ax[row][ncol - 1], allv, colours,
              [None if v is None else
               f"{labels[ti]} (n={len(v)}, med {np.median(v):.1f}°)"
               for ti, v in enumerate(allv)], edges)
    for row, ref in ((0, "fixed bin axis"), (1, "bin centroid")):
        a = ax[row][ncol - 1]
        a.set_title(f"ALL BINS — to {ref}", fontsize=10)
        a.set_xlabel("angle (deg)", fontsize=8)
        a.set_ylabel("density", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.3, lw=0.5)
        a.legend(fontsize=7)

    # 45 deg is the Voronoi half-angle: a member further than this from the axis
    # would belong to a different bin, so it is the hard ceiling on row 1 and the
    # reason the x range stops there.
    for a in ax.ravel():
        a.axvline(45.0, color="0.4", ls=":", lw=1.0)

    fig.suptitle("Regrasp — within-bin angular spread of the assigned grasps "
                 "(dotted line = 45° Voronoi half-angle)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
