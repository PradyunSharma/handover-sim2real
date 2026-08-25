"""
Per-EPOCH train/val curves for every DAgger iteration of a Regrasp run.

    python examples/plot_regrasp_fits.py output/dagger_runs/regrasp_fast1
    python examples/plot_regrasp_fits.py output/dagger_runs/regrasp_fast1 --metric val_cond_delta
    python examples/plot_regrasp_fits.py output/dagger_runs/regrasp_fast1 --iters 0,4,8

WHAT THIS SHOWS THAT dagger_log.csv CANNOT. `dagger_log.csv` records ONE row per
iteration — the end-of-fit `train_loss` and `val_loss`. That is a single point
per refit, so it can say the aggregate got harder but not whether a fit
converged, overfit, or was still descending when the epoch budget ran out. The
per-epoch rows live in `<run>/iters/iter_NN/log.csv`, written by BCTrainer, and
this reads those.

READ IT FOR THREE THINGS.

  DID THE FIT CONVERGE. A val curve still falling at the last epoch means
  `iter_epochs` is too small and the refit is leaving accuracy on the table. One
  that bottomed early and climbed is overfitting the aggregate — and under
  `EVAL.ckpt: last` the scored weights are the overfit ones, deliberately, so
  this panel is where you see how much that costs.

  IS THE AGGREGATE BECOMING SELF-INCONSISTENT. Under Follow-The-Leader every
  refit is a fresh fit on a bigger D. Curves that shift UP bodily from one
  iteration to the next mean later data contradicts earlier data — the one
  failure DAgger cannot average away. Curves that shift up because D simply got
  more diverse look the same at the last epoch and different in shape, which is
  why the whole curve is plotted rather than its endpoint.

  WHERE `best.pt` WOULD HAVE LANDED. The val minimum is marked. In runs 2 and 3
  `EVAL.ckpt: last`, so `best.pt` is written and never read; the gap between the
  marked minimum and the right-hand end is exactly what that choice gives up.

DEFAULT METRIC IS `*_total`, the optimised objective (pose + gripper + aux). Any
column of the per-epoch log works: `--metric val_cond_delta` for whether the
network reads the direction channels, `--metric val_pose_pos_m` for metres of
position error, `--metric val_aux_pos_mm` for the auxiliary head.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def iteration_logs(run: Path):
    """[(iter_index, rows)] for every iters/iter_NN/log.csv, in iteration order."""
    out = []
    for d in sorted((run / "iters").glob("iter_*")):
        p = d / "log.csv"
        if not p.exists():
            continue
        with p.open() as f:
            rows = [r for r in csv.DictReader(f) if r.get("epoch", "").strip()]
        if rows:
            out.append((int(d.name.split("_")[1]), rows))
    return out


def series(rows, key):
    """(epochs, values) for `key`, skipping blanks so a partial log still plots."""
    xs, ys = [], []
    for r in rows:
        v = r.get(key, "")
        if v.strip():
            try:
                ys.append(float(v))
                xs.append(int(r["epoch"]))
            except ValueError:
                pass
    return xs, ys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="output/dagger_runs/<name>")
    p.add_argument("--metric", default="total",
                   help="bare name -> train_<m>/val_<m> are overlaid (default "
                        "'total'); or an explicit column like 'val_cond_delta', "
                        "which is plotted alone")
    p.add_argument("--iters", default="all",
                   help="'all' or a comma list, e.g. 0,4,8")
    p.add_argument("--out", default=None, help="default <run>/fit_curves.png")
    p.add_argument("--logy", action="store_true", help="log-scale the y axis")
    args = p.parse_args()

    run = Path(args.run_dir)
    logs = iteration_logs(run)
    if not logs:
        raise SystemExit(f"no iters/iter_*/log.csv under {run}")
    if args.iters != "all":
        want = {int(x) for x in args.iters.split(",")}
        logs = [(i, r) for i, r in logs if i in want]
        if not logs:
            raise SystemExit(f"--iters {args.iters} matched none of "
                             f"{[i for i, _ in iteration_logs(run)]}")

    # A bare name means "the train and val versions of this", which is the
    # common case and the reason the flag is not just a column name.
    explicit = args.metric.startswith(("train_", "val_"))
    pairs = ([(args.metric, args.metric, "-")] if explicit else
             [(f"train_{args.metric}", "train", "-"),
              (f"val_{args.metric}", "val", "--")])

    n = len(logs)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow),
                           squeeze=False)

    # One shared y range across every panel. Per-panel autoscaling would make a
    # fit that shifted bodily upward look identical to one that did not, and
    # that shift is the FTL self-consistency signal this figure exists for.
    lo, hi = float("inf"), float("-inf")
    for _, rows in logs:
        for key, _, _ in pairs:
            _, ys = series(rows, key)
            if ys:
                lo, hi = min(lo, min(ys)), max(hi, max(ys))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    ylim = (lo - pad, hi + pad) if lo < float("inf") else None

    for k, (it, rows) in enumerate(logs):
        a = ax[k // ncol][k % ncol]
        drew = False
        for key, label, style in pairs:
            xs, ys = series(rows, key)
            if not ys:
                continue
            drew = True
            a.plot(xs, ys, style, marker="o", ms=3, lw=1.5, label=label)
            # Mark where best.pt would have been taken. Under EVAL.ckpt: last
            # it is not, and the distance to the right-hand end is the cost.
            if key.startswith("val_"):
                j = min(range(len(ys)), key=lambda i: ys[i])
                a.plot([xs[j]], [ys[j]], "v", ms=8, color="tab:red", zorder=5,
                       label=f"val min @ep{xs[j]}")
        if not drew:
            a.text(0.5, 0.5, f"no '{args.metric}' column", ha="center",
                   va="center", transform=a.transAxes, fontsize=9, color="0.5")
        if ylim:
            a.set_ylim(*ylim)
        if args.logy:
            a.set_yscale("log")
        a.grid(alpha=0.3, lw=0.5)
        a.set_title(f"iteration {it}"
                    + ("  (base fit)" if it == 0 else ""), fontsize=10)
        a.set_xlabel("epoch", fontsize=8)
        a.set_ylabel(args.metric, fontsize=8)
        a.tick_params(labelsize=7)
        a.legend(fontsize=7, loc="best")

    for k in range(len(logs), nrow * ncol):
        ax[k // ncol][k % ncol].axis("off")

    fig.suptitle(f"{run.name} — per-epoch {args.metric}, one panel per refit "
                 f"(shared y axis)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out) if args.out else run / "fit_curves.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # The endpoint summary the figure cannot be read off precisely.
    print(f"\n  {'iter':>4}  {'epochs':>6}  {'val min':>8}  {'@ep':>4}  "
          f"{'val last':>8}  {'gap':>7}")
    for it, rows in logs:
        key = pairs[-1][0] if explicit else f"val_{args.metric}"
        xs, ys = series(rows, key)
        if not ys:
            continue
        j = min(range(len(ys)), key=lambda i: ys[i])
        print(f"  {it:>4}  {len(ys):>6}  {ys[j]:>8.4f}  {xs[j]:>4}  "
              f"{ys[-1]:>8.4f}  {ys[-1] - ys[j]:>+7.4f}")


if __name__ == "__main__":
    main()
