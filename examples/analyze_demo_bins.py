"""
Per-bin demonstration counts for a Regrasp shard, read from the detail CSV.

    python examples/analyze_demo_bins.py
    python examples/analyze_demo_bins.py output/bc_dataset/tables/*_detail.csv

Two numbers per approach direction: how many demonstrations it carries, and how
many of them actually reached the grasp they were aiming at.

WHY THIS DOES NOT READ THE SHARD. `build_demo_table.py` prints a per-bin block
already, but it needs the .h5 and h5py to do it. This reads the CSV that script
wrote, so re-slicing by direction costs no I/O, runs on a login node, and works on
the artifact that actually gets copied off the cluster.

THE SLOT INDEX IS NOT A DIRECTION, which is the whole reason this file exists.
Slots pack densely over whichever bins a scene reaches, so `grasp_idx` 1 is `+y`
on one scene and `+z` on another. Counting columns of the success matrix answers
a different question than counting bins, and only the latter tells you whether a
direction has enough demonstrations to be learnable.

A BIN WITH NO DEMONSTRATIONS IS PRINTED ANYWAY, at zero. It is a direction the
policy will EXTRAPOLATE into if the feasibility mask ever admits it at test time,
so its absence is a finding rather than a blank.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handover_sim2real.regrasp.directions import BIN_SHORT      # noqa: E402

DEFAULT_CSV = "output/bc_dataset/tables/train_regrasp_demo_success_detail.csv"
CRITERIA = ("reach", "close_label", "caption", "all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="*", default=[DEFAULT_CSV],
                   help=f"one or more *_detail.csv (default: {DEFAULT_CSV}). "
                        "Several are merged, which is what a sharded collection "
                        "needs.")
    p.add_argument("--criterion", default="reach", choices=list(CRITERIA),
                   help="what counts as successful. reach (default): the "
                        "terminal pose was within the close thresholds of the "
                        "target grasp.")
    p.add_argument("--by", default="assigned", choices=["assigned", "realized"],
                   help="which bin to group on. assigned (default): the "
                        "direction the demo was COMMANDED to take. realized: "
                        "where it actually went — the two differ exactly where "
                        "the pin was refused.")
    return p.parse_args()


def read_rows(paths) -> list[dict]:
    """Whitespace-tolerant read of one or more detail CSVs.

    Earlier versions of `build_demo_table.py` column-aligned the file, so both
    keys and values can carry padding; a freshly written one carries none.
    Stripping both means either reads the same and no file needs regenerating.
    """
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({(k or "").strip(): (v or "").strip()
                             for k, v in r.items()})
    return rows


def axis_of(row: dict, which: str) -> str:
    """The row's direction label.

    Prefers the `axis` column, which `build_demo_table.py` now writes — but that
    column labels `bin_assigned` only, so `--by realized` always goes through the
    index. Falls back to the index for CSVs written before `axis` existed, so
    this runs on a file that is already on disk.
    """
    if which == "assigned" and row.get("axis"):
        return row["axis"]
    b = int(row[f"bin_{which}"])
    if b < 0:
        return "?"                       # no pin table, or a Phase-4 shard
    return BIN_SHORT[b] if b < len(BIN_SHORT) else f"b{b}"


def main() -> None:
    args = parse_args()
    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    missing = [c for c in (args.criterion, f"bin_{args.by}") if c not in rows[0]]
    if missing:
        raise SystemExit(f"{args.csv[0]} has no column(s) {missing}; "
                         f"found {sorted(rows[0])}")

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # [demos, passed]
    scenes: dict[str, set] = defaultdict(set)
    for r in rows:
        a = axis_of(r, args.by)
        tally[a][0] += 1
        tally[a][1] += int(r[args.criterion])
        scenes[a].add(int(r["scene"]))

    # Canonical axis order first (including bins nothing demonstrated), then any
    # label the octahedral set does not name — a fibonacci table's `b7`, or `?`.
    order = list(BIN_SHORT) + sorted(a for a in tally if a not in BIN_SHORT)

    n_scenes = len({int(r["scene"]) for r in rows})
    print(f"\n{len(rows)} demos over {n_scenes} scenes   "
          f"bin = bin_{args.by}   success = {args.criterion} == 1")
    print(f"  from {', '.join(str(p) for p in args.csv)}\n")
    print(f"  {'bin':<4} {'demos':>7} {'scenes':>7} {'success':>8} {'rate':>7}")
    for a in order:
        d, k = tally.get(a, [0, 0])
        if d == 0:
            print(f"  {a:<4} {0:>7} {0:>7} {0:>8} {'—':>7}"
                  f"   <- never demonstrated")
            continue
        print(f"  {a:<4} {d:>7} {len(scenes[a]):>7} {k:>8} {100 * k / d:>6.1f}%")

    d = sum(v[0] for v in tally.values())
    k = sum(v[1] for v in tally.values())
    print(f"  {'-' * 4} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 7}")
    print(f"  {'all':<4} {d:>7} {n_scenes:>7} {k:>8} {100 * k / d:>6.1f}%")

    live = sum(1 for a in BIN_SHORT if tally.get(a, [0])[0])
    print(f"\n  live bins: {live}/{len(BIN_SHORT)} — the retry ladder has {live} "
          f"rungs, and chained_retry_at_k saturates at k={live}.")

    # ---- how many grasp poses each scene carries -----------------------------
    # Two histograms over the SAME scenes. The first counts what the table
    # offered; the second counts what survived `--criterion`, and its `0` row is
    # the scenes that drop out entirely because no demonstration on them worked.
    #
    # THE `>= 2` LINE IS THE ONE THAT MATTERS. A scene with a single demonstration
    # teaches reaching and grasping, but `d` is then a deterministic function of
    # its observation and it cannot break the conditioning confound; only a scene
    # with two or more contributes the same-observation/different-command contrast
    # that forces the channels to be read. Filtering on `reach` moves scenes from
    # the second group into the first, so the rate below is the real cost of the
    # filter and the number to weigh against the 30% of bad demonstrations it
    # removes.
    total_of: dict[int, int] = defaultdict(int)
    succ_of: dict[int, int] = defaultdict(int)
    for r in rows:
        s = int(r["scene"])
        total_of[s] += 1
        succ_of[s] += int(r[args.criterion])

    hist_all = Counter(total_of.values())
    # Every scene appears, including the ones whose successes are 0 — that count
    # IS the finding, and a Counter over successes alone would silently omit it.
    hist_ok = Counter(succ_of[s] for s in total_of)
    top = max(hist_all) if hist_all else 0

    print("\n  grasp poses per scene")
    print(f"    {'':<22}{'all':>5}{args.criterion:>12}")
    for n in range(0, top + 1):
        # The `all` column can never have a 0 row — a scene is only in the file
        # because it has at least one episode — so it prints as a dash rather
        # than a misleading zero.
        a_cell = "—" if n == 0 else f"{hist_all.get(n, 0)}"
        label = f"{n} pose" + ("" if n == 1 else "s")
        print(f"    {label:<22}{a_cell:>5}{hist_ok.get(n, 0):>12}"
              + ("   <- no usable demonstration" if n == 0 and hist_ok.get(0)
                 else ""))
    n_all2 = sum(c for n, c in hist_all.items() if n >= 2)
    n_ok2 = sum(c for n, c in hist_ok.items() if n >= 2)
    n_ok1 = sum(c for n, c in hist_ok.items() if n >= 1)
    print(f"    {'-' * 39}")
    print(f"    {'scenes':<22}{n_scenes:>5}{n_ok1:>12}")
    print(f"    {'>= 2 (can pair)':<22}{n_all2:>5}{n_ok2:>12}"
          f"   {100 * n_all2 / n_scenes:.0f}% -> "
          f"{100 * n_ok2 / n_scenes:.0f}% of all scenes")


if __name__ == "__main__":
    main()
