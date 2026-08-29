#!/usr/bin/env python
"""Recompute per-bin COLLECTION approach error for a run that predates the columns.

    python examples/backfill_collect_err.py output/dagger_runs/regrasp_run12
    python examples/backfill_collect_err.py --run regrasp_run12
    python examples/backfill_collect_err.py output/dagger_runs/regrasp_run9 --check

`c_min_pos_b{b}` / `c_min_rot_b{b}` — how close the DAgger episodes of each
commanded direction actually got to the grasp — were added to `dagger_log.csv`
after run 12. Runs 1-12 have the pooled `mean_min_pos` and nothing per bin.

THE DATA IS NOT LOST. Every shard stores `robot_states` and the episode's own
`grasp_pose_world` and `bin_realized`, and the collector's metric is exactly
`reach.terminal_pose_error` applied per step, so the per-bin means recompute from
disk. This writes them to `<run>/collect_err.csv`, which plot_regrasp_run.py
splices in by iteration the same way it splices `eval_log.csv` — only into blank
cells, so a run that logged its own numbers is never overwritten.

ONE KNOWN DIFFERENCE, MEASURED RATHER THAN ASSUMED. The collector updates
`min_pos` on EVERY simulator step, at the top of the step and before the label
decision; a shard only holds the steps whose label was kept. So it misses the
step an episode broke on, and the pair popped from the tail when an EXPERT step
tripped ENV_DONE. The recomputed minimum is therefore taken over a subset and can
only be >= the true one.

    run 12, 25 iterations:  +2.93 mm mean, +4.94 mm worst, on values near 120 mm

A consistent ~2% high bias, in one direction, on every iteration — so it shifts
the level slightly and changes no trend, no comparison between bins, and no
comparison between runs backfilled the same way. Do NOT mix a backfilled series
with a natively-logged one in the same comparison. `--check` prints this diff for
whatever run you point it at; run it before trusting a new backfill.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handover_sim2real.regrasp import directions as _rg_dirs  # noqa: E402
from handover_sim2real.regrasp import reach as _rg_reach      # noqa: E402

N_BINS = len(_rg_dirs.BINS)


def episode_min_err(rs, grasp_pose_world) -> tuple[float, float]:
    """(closest position error, closest rotation error) over the episode.

    `terminal_pose_error` applied at every recorded step and minimised, which is
    what the collector does online — see its `min_pos, min_rot = min(...)` line.
    """
    best_p, best_r = float("inf"), float("inf")
    for step in rs:
        p, r = _rg_reach.terminal_pose_error(step, grasp_pose_world)
        best_p, best_r = min(best_p, p), min(best_r, r)
    return best_p, best_r


def scan_shard(path: Path) -> dict:
    """Per-bin and pooled error lists for one `dagger_iter_NN.h5`."""
    import h5py

    per_bin = {b: {"min_pos": [], "min_rot": []} for b in range(N_BINS)}
    pooled = {"min_pos": [], "min_rot": []}
    unjudgeable = 0
    with h5py.File(path, "r") as f:
        for key in f:
            grp = f[key]
            if "grasp_pose_world" not in grp.attrs:
                # Predates the attr; cannot be judged, and counting it as 0
                # would drag every mean toward the grasp.
                unjudgeable += 1
                continue
            rs = np.asarray(grp["robot_states"])
            if rs.size == 0:
                unjudgeable += 1
                continue
            p, r = episode_min_err(rs, np.asarray(grp.attrs["grasp_pose_world"]))
            pooled["min_pos"].append(p)
            pooled["min_rot"].append(r)
            # `bin_realized`, matching how the collector buckets its per-bin
            # counters — the direction the episode actually flew, not the one it
            # was asked for.
            b = int(grp.attrs.get("bin_realized", -1))
            if 0 <= b < N_BINS:
                per_bin[b]["min_pos"].append(p)
                per_bin[b]["min_rot"].append(r)
    return {"per_bin": per_bin, "pooled": pooled, "unjudgeable": unjudgeable}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", default=None,
                   help="output/dagger_runs/<name>")
    p.add_argument("--run", default=None,
                   help="run NAME instead of a path (resolved like the viewer's "
                        "--run: $REGRASP_DATA/dagger_runs, $RUNS/..., ./output/...)")
    p.add_argument("--run-root", default=None)
    p.add_argument("--out", default=None,
                   help="output CSV (default <run>/collect_err.csv)")
    p.add_argument("--check", action="store_true",
                   help="also recompute the POOLED mean_min_pos and diff it "
                        "against the logged column, to size the "
                        "labelled-steps-only difference")
    args = p.parse_args()

    if args.run:
        from handover_sim2real.regrasp.runspec import resolve_run
        run = resolve_run(args.run, run_root=args.run_root).run_dir
    elif args.run_dir:
        run = Path(args.run_dir)
    else:
        raise SystemExit("give a run directory, or --run <name>.")

    shards = sorted((run / "data").glob("dagger_iter_*.h5"))
    if not shards:
        raise SystemExit(f"no shards under {run / 'data'}")

    logged = {}
    log_path = run / "dagger_log.csv"
    if log_path.exists():
        with log_path.open() as f:
            logged = {r["iter"]: r for r in csv.DictReader(f) if r.get("iter")}

    fields = ["iter"] + [f"c_{k}_b{b}" for b in range(N_BINS)
                         for k in ("min_pos", "min_rot")]
    rows, checks = [], []
    for path in shards:
        stem = path.stem.rsplit("_", 1)[-1]
        if not stem.isdigit():
            continue
        it = int(stem)
        s = scan_shard(path)
        row = {"iter": it}
        for b in range(N_BINS):
            for k in ("min_pos", "min_rot"):
                m = _mean(s["per_bin"][b][k])
                row[f"c_{k}_b{b}"] = "" if m is None else round(m, 4)
        rows.append(row)

        n = len(s["pooled"]["min_pos"])
        mp = _mean(s["pooled"]["min_pos"])
        note = f" ({s['unjudgeable']} unjudgeable)" if s["unjudgeable"] else ""
        if args.check:
            try:
                was = float(logged.get(str(it), {}).get("mean_min_pos", ""))
            except ValueError:
                was = float("nan")
            checks.append((it, was, mp))
            print(f"  iter {it:2d}  {n:4d} eps{note}   recomputed "
                  f"{mp:.4f} m   logged {was:.4f} m   diff "
                  f"{abs(mp - was) * 1000:6.2f} mm")
        else:
            print(f"  iter {it:2d}  {n:4d} eps{note}   pooled min_pos "
                  f"{mp:.4f} m")

    out = Path(args.out) if args.out else run / "collect_err.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} iterations)")

    if checks:
        diffs = [abs(m - w) for _, w, m in checks if w == w]
        if diffs:
            print(f"pooled mean_min_pos, recomputed vs logged: max diff "
                  f"{max(diffs) * 1000:.2f} mm, mean "
                  f"{sum(diffs) / len(diffs) * 1000:.2f} mm over "
                  f"{len(diffs)} iterations")
    print("plot_regrasp_run.py picks this up automatically on the next run.")


if __name__ == "__main__":
    main()
