"""
Plot the PER-EPOCH training curves of a Phase-4 DAgger run.

    python examples/plot_dagger_epochs.py output/dagger_runs/dagger4_run1
    python examples/plot_dagger_epochs.py output/dagger_runs/dagger4_run1 --show

This is the companion to plot_dagger_run.py. That one reads dagger_log.csv, which
has ONE row per DAgger iteration and therefore only the last-epoch loss of each
refit. This one reads the per-epoch log.csv that the BC trainer writes inside
every iteration dir, so you get the same pos / rot / gripper breakdown Phase 1
gives you via analyze_bc_run.py — but for all N refits at once.

Each iteration is a run dir in its own right (config.yaml, checkpoints/,
normalization.npz, log.csv), so for a single iteration you can just use the
Phase-1 tool directly and get its predicted-vs-expert viewer too:

    python examples/analyze_bc_run.py \\
        --run-dir output/dagger_runs/dagger4_run1/iters/iter_05 --mode both

Two figures, because under Follow-The-Leader every iteration is a FRESH fit on a
bigger aggregate and the two obvious questions need different axes:

  <run>/epoch_curves.png          the whole training history end to end, epochs
                                  laid out consecutively with a grey line at each
                                  iteration boundary. The sawtooth is expected —
                                  that is the optimizer restarting. What you are
                                  looking for is the ENVELOPE: the floor each
                                  refit reaches should hold roughly flat. A floor
                                  that climbs iteration after iteration means D is
                                  accumulating labels the network cannot satisfy
                                  simultaneously, which is the one failure mode
                                  aggregation cannot average away.

  <run>/epoch_curves_overlay.png  every refit on a shared 0..E axis, coloured
                                  dark (early) to bright (late). This answers
                                  "is the fit getting harder as |D| grows" by
                                  making the curves directly comparable: if the
                                  bright curves sit above the dark ones, later
                                  aggregates are harder to fit, not just longer.

NOTE the two loss scales are not comparable across iterations in an absolute
sense — the base fit runs base_epochs (100) and the refits run iter_epochs (25 or
50), so the base curve simply has more time to descend. Compare floors, not
endpoints at equal epoch counts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; overridden by --show below
import matplotlib.pyplot as plt
import numpy as np


# title, metric stem (train_<stem> / val_<stem>), y-scale
PANELS = [
    ("Total loss",              "total",        "log"),
    ("Pose loss (SmoothL1)",    "pose_loss",    "log"),
    ("Gripper loss (BCE)",      "gripper_loss", "log"),
    ("Pose L1 (normalized)",    "pose_l1",      "log"),
    ("Pos / rot L1 (train)",    "_posrot",      "log"),
    ("Gripper accuracy",        "gripper_acc",  "linear"),
]


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    """log.csv -> column name -> float array, NaN for blanks (val_every gaps)."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = {}
    for key in rows[0].keys():
        vals = []
        for r in rows:
            v = r.get(key, "")
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(np.nan)
        cols[key] = np.asarray(vals, dtype=np.float64)
    return cols


def load_iters(run: Path) -> list[tuple[int, dict[str, np.ndarray]]]:
    iters_dir = run / "iters"
    if not iters_dir.is_dir():
        raise SystemExit(
            f"No {iters_dir}.\n"
            f"The per-epoch logs live in the iteration dirs, which the summary sync\n"
            f"skips (they sit next to the checkpoints). Pull just the CSVs with:\n"
            f"    rsync -avP --include='*/' --include='log.csv' --exclude='*' \\\n"
            f"      pradyunsharma@login.delftblue.tudelft.nl:"
            f"/home/pradyunsharma/h2r/handover-sim2real/{run}/iters/ \\\n"
            f"      {iters_dir}/")
    out = []
    for d in sorted(iters_dir.glob("iter_*")):
        log = d / "log.csv"
        if not log.exists():
            print(f"[skip] {d.name}: no log.csv")
            continue
        cols = _read_csv(log)
        if cols:
            out.append((int(d.name.split("_")[1]), cols))
    if not out:
        raise SystemExit(f"No readable log.csv under {iters_dir}")
    return out


def _finite(xs, ys):
    """Drop NaN pairs so sparse val series draw as lines, not isolated dots."""
    m = np.isfinite(ys)
    return np.asarray(xs)[m], np.asarray(ys)[m]


def _series(cols: dict, stem: str, split: str) -> np.ndarray | None:
    return cols.get(f"{split}_{stem}")


# ── figure 1: consecutive epochs, whole run ─────────────────────────────────

def plot_concat(iters, run: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))

    # x offset of each iteration on the shared axis
    offsets, acc = {}, 0
    for it, cols in iters:
        offsets[it] = acc
        acc += len(cols["epoch"])
    total = acc

    for ax, (title, stem, scale) in zip(axes.flat, PANELS):
        for it, cols in iters:
            x = offsets[it] + cols["epoch"]
            if stem == "_posrot":
                for key, colour, lab in (("pose_pos_l1", "tab:blue", "pos"),
                                         ("pose_rot_l1", "tab:orange", "rot")):
                    s = _series(cols, key, "train")
                    if s is not None:
                        ax.plot(*_finite(x, s), "-", lw=1.0, color=colour,
                                label=lab if it == iters[0][0] else None)
            else:
                for split, colour, lab in (("train", "tab:blue", "train"),
                                           ("val", "tab:orange", "val")):
                    s = _series(cols, stem, split)
                    if s is not None:
                        ax.plot(*_finite(x, s), "-", lw=1.0, color=colour,
                                label=lab if it == iters[0][0] else None)

        # iteration boundaries
        for it, _ in iters[1:]:
            ax.axvline(offsets[it], color="0.75", lw=0.6, zorder=0)

        # panel 0 carries the iteration ruler on top; leave room for it
        ax.set_title(title, fontsize=10, pad=26 if ax is axes.flat[0] else 6)
        ax.set_xlabel("cumulative epoch")
        ax.set_xlim(0, total)
        ax.set_yscale(scale)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # iteration numbers along the top of the first panel
    ax0 = axes.flat[0]
    top = ax0.secondary_xaxis("top")
    top.set_xticks([offsets[it] for it, _ in iters])
    top.set_xticklabels([str(it) for it, _ in iters], fontsize=6)
    top.set_xlabel("DAgger iteration", fontsize=8)

    fig.suptitle(f"Per-epoch training curves, all refits — {run.name}   "
                 f"({len(iters)} iterations, {total} epochs total)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


# ── figure 2: refits overlaid on a shared epoch axis ────────────────────────

def plot_overlay(iters, run: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    cmap = plt.get_cmap("viridis")
    lo, hi = iters[0][0], iters[-1][0]
    norm = plt.Normalize(lo, hi)

    for ax, (title, stem, scale) in zip(axes.flat, PANELS):
        for it, cols in iters:
            colour = cmap(norm(it))
            x = cols["epoch"]
            if stem == "_posrot":
                for key, style in (("pose_pos_l1", "-"), ("pose_rot_l1", "--")):
                    s = _series(cols, key, "train")
                    if s is not None:
                        ax.plot(*_finite(x, s), style, lw=1.0, color=colour, alpha=0.85)
            else:
                s = _series(cols, stem, "train")
                if s is not None:
                    ax.plot(*_finite(x, s), "-", lw=1.0, color=colour, alpha=0.85)
                s = _series(cols, stem, "val")
                if s is not None:
                    ax.plot(*_finite(x, s), "--", lw=0.9, color=colour, alpha=0.55)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch within iteration")
        ax.set_yscale(scale)
        ax.grid(True, alpha=0.3)

    # explicit layout: fig.colorbar(ax=axes) does its own and tight_layout fights it
    fig.subplots_adjust(left=0.05, right=0.90, top=0.88, bottom=0.08,
                        hspace=0.38, wspace=0.22)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 label="DAgger iteration", fraction=0.02, pad=0.02)
    fig.suptitle(f"Refits overlaid — {run.name}   "
                 f"solid = train, dashed = val (pos solid / rot dashed in panel 5)",
                 fontsize=13)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


# ── console summary ─────────────────────────────────────────────────────────

def summarize(iters) -> None:
    print(f"\n{'iter':>5s} {'epochs':>7s} {'train_end':>10s} {'train_min':>10s} "
          f"{'val_end':>9s} {'val_min':>9s} {'grip_acc':>9s}")
    for it, cols in iters:
        tr = cols.get("train_total", np.array([np.nan]))
        va = cols.get("val_total", np.array([np.nan]))
        ga = cols.get("train_gripper_acc", np.array([np.nan]))

        def _min(a):
            f = a[np.isfinite(a)]
            return f.min() if len(f) else np.nan

        def _end(a):
            f = a[np.isfinite(a)]
            return f[-1] if len(f) else np.nan

        print(f"{it:5d} {len(cols['epoch']):7d} {_end(tr):10.4f} {_min(tr):10.4f} "
              f"{_end(va):9.4f} {_min(va):9.4f} {_end(ga):9.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="output/dagger_runs/<name>")
    p.add_argument("--show", action="store_true", help="open the figures too")
    args = p.parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)

    run = Path(args.run_dir)
    iters = load_iters(run)
    summarize(iters)
    plot_concat(iters, run, run / "epoch_curves.png")
    plot_overlay(iters, run, run / "epoch_curves_overlay.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
