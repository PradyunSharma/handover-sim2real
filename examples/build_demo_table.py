"""
Per-(scene, grasp) success matrix for a Regrasp demonstration shard.

    python examples/build_demo_table.py --demos output/bc_dataset/train_regrasp.h5

Writes `output/bc_dataset/tables/<stem>_demo_success.csv`: one row per scene, one
column per grasp index, `1` where that demonstration succeeded and `0` where it
did not. A pair the table never asked for is left BLANK, not zero — "never
collected" and "collected and failed" are different facts and a 0 in both places
would make the file unreadable.

    scene,0,1,2,3
    0,1,,,
    3,1,0,,
    32,1,1,1,0

COLUMNS ARE GRASP INDICES, NOT BINS, and that is the point of the file. A bin
does not identify a demonstration: run 2's `--per-bin 1` table gives each bin one
grasp, but run 3's `--per-bin 3` gives it three, so on scene 32 the `+x` bin is
slots 0, 4 and 8 — three poses, three trajectories, three separate outcomes. The
slot is what the collector iterated and what the episode records in `grasp_idx`.
`--by bin` gives the bin view instead, where a cell is the AND over that bin's
grasps, which is the right reading for "can this scene be demonstrated from this
direction at all".

WHAT "SUCCESSFUL" MEANS HERE, AND WHY IT IS NOT `demo_ok_table`.

`audit_regrasp_demos.py --write-ok` asks whether the CAPTION is valid —
`bin_realized == bin_assigned` and the pin landed. On run 2's base set that
passes 1575 of 1596 (98.7%). It is the right filter for its job (a mis-captioned
episode teaches "when told +z, approach from -y") but it says nothing about
whether the demonstration ACCOMPLISHED anything.

This file asks the substantive question: did the expert actually reach the grasp
it was aiming at. Measured on the same shard, **1116 of 1596 (69.9%)** did. The
other 480 stop a median 128 mm short, run 14 steps instead of 21, and end still
commanding "keep approaching" — `c_env_done` fired and the episode was truncated.
They were 30% of the base set and they were in `D`, because the demo_ok filter
does not look for this. `SIM.reach_filter` (on by default, see
`handover_sim2real/regrasp/reach.py`) now removes them from D, from DAgger
collection and from the in-loop eval, using this same criterion.

TWO INDEPENDENT CRITERIA, AND THEY AGREE ON 99.6% OF EPISODES:

  reach        terminal EE pose within `--pos-thresh` / `--rot-thresh` of
               `grasp_pose_world`. The default, and the geometric truth.
  close_label  the last expert action commands a gripper CLOSE
               (`expert_actions[-1][6] < 0.5`). The collector appends that
               transition only when the plan ran to completion, so it is the
               DATA's own statement that the expert believed it had arrived.

That they agree is worth knowing rather than assuming: it means neither is an
artifact of a threshold. `--criterion caption` reproduces `demo_ok_table`, and
`--criterion all` requires reach AND caption — the strictest reading, 1096.

Reads the shard only. No simulator, no GPU, seconds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handover_sim2real.regrasp import directions as D       # noqa: E402

CRITERIA = ("reach", "close_label", "caption", "all")


def axis_label(b: int) -> str:
    """`bin_assigned` -> the readable axis, for the detail CSV's `axis` column.

    The slot index is NOT a direction — slots pack densely over whichever bins a
    scene reaches, so slot 1 is `+y` on one scene and `+z` on another. This column
    is what makes the detail file groupable by direction without a join against
    the pin table. Empty for -1 (no pin table / a Phase-4 shard); `b<n>` past the
    octahedral set, where a fibonacci table's bins have no axis name.
    """
    if b < 0:
        return ""
    return D.BIN_SHORT[b] if b < len(D.BIN_SHORT) else f"b{b}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", nargs="+", required=True,
                   help="one or more shards. Several are merged, which is what a "
                        "`--shard i/n` collection needs — the pieces hold whole "
                        "scenes, so no scene is split across files.")
    p.add_argument("--out", default=None,
                   help="default output/bc_dataset/tables/<stem>_demo_success.csv")
    p.add_argument("--criterion", default="reach", choices=list(CRITERIA),
                   help="reach (default): terminal pose within the close "
                        "thresholds. close_label: the expert emitted a CLOSE. "
                        "caption: bin_realized == bin_assigned and the pin "
                        "landed, i.e. what demo_ok_table records. all: reach "
                        "AND caption.")
    p.add_argument("--by", default="grasp", choices=["grasp", "bin"],
                   help="grasp (default): one column per grasp index, the "
                        "collector's unit of work. bin: one column per "
                        "direction, cell = AND over that bin's grasps.")
    p.add_argument("--pos-thresh", type=float, default=0.02,
                   help="metres; mirrors DAGGER.close_pos_thresh")
    p.add_argument("--rot-thresh", type=float, default=0.34,
                   help="radians (~19.5 deg); mirrors DAGGER.close_rot_thresh")
    p.add_argument("--no-detail", action="store_true",
                   help="skip the companion per-episode CSV. It is what makes a "
                        "0 in the matrix answerable, so keep it unless the file "
                        "count matters.")
    return p.parse_args()


def episode_rows(paths, pos_thresh: float, rot_thresh: float) -> list[dict]:
    """One dict per episode: what it was, and every success criterion for it."""
    from transforms3d.quaternions import quat2mat

    out = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for k in sorted(x for x in f if x.startswith("episode_")):
                g, a = f[k], f[k].attrs
                rs = np.asarray(g["robot_states"])
                act = np.asarray(g["expert_actions"])
                G = np.asarray(a["grasp_pose_world"], dtype=np.float64)
                # Terminal EE pose. rs[18:21] xyz, rs[21:25] quaternion wxyz —
                # in SIM WORLD, the same frame the pin table's poses are in, so
                # the two are directly comparable with no extra transform.
                p_err = float(np.linalg.norm(rs[-1, 18:21] - G[:3, 3]))
                R = quat2mat(rs[-1, 21:25])
                cos = (np.trace(R.T @ G[:3, :3]) - 1.0) / 2.0
                r_err = float(np.arccos(np.clip(cos, -1.0, 1.0)))
                ba = int(a.get("bin_assigned", -1))
                br = int(a.get("bin_realized", -1))
                reach = bool(p_err < pos_thresh and r_err < rot_thresh)
                # The gripper channel of the LAST label. The base collector
                # appends the close transition only when the plan ran to
                # completion, so this is the data's own "the expert arrived".
                closed = bool(float(act[-1, 6]) < 0.5)
                caption = bool(ba >= 0 and ba == br and int(a.get("pin_ok", 1)))
                out.append({
                    "file": Path(path).name, "episode": int(k.split("_")[1]),
                    "scene": int(a["scene_idx"]),
                    "grasp_idx": int(a.get("grasp_idx", 0)),
                    "bin_assigned": ba, "bin_realized": br,
                    "axis": axis_label(ba),
                    "pin_ok": int(a.get("pin_ok", -1)),
                    "steps": int(len(act)),
                    "pos_err_m": round(p_err, 6),
                    "rot_err_rad": round(r_err, 6),
                    "reach": int(reach), "close_label": int(closed),
                    "caption": int(caption), "all": int(reach and caption),
                })
    return out


def main() -> None:
    args = parse_args()
    rows = episode_rows(args.demos, args.pos_thresh, args.rot_thresh)
    if not rows:
        raise SystemExit(f"no episodes found in {args.demos}")

    key = "grasp_idx" if args.by == "grasp" else "bin_assigned"
    # AND over duplicates. Under `--by grasp` there is exactly one episode per
    # (scene, slot) so this is a no-op; under `--by bin` it is the whole point —
    # a bin counts as demonstrated only if every one of its grasps was.
    cell: dict[tuple[int, int], int] = {}
    for r in rows:
        c = int(r[key])
        if c < 0:
            continue
        k = (r["scene"], c)
        cell[k] = min(cell.get(k, 1), int(r[args.criterion]))

    scenes = sorted({s for s, _ in cell})
    cols = sorted({c for _, c in cell})

    stem = Path(args.demos[0]).stem.split(".")[0]
    out = Path(args.out) if args.out else (
        Path("output/bc_dataset/tables") / f"{stem}_demo_success.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    head = ("scene" if args.by == "grasp"
            else "scene")  # same label; the columns carry the meaning
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([head] + ([str(c) for c in cols] if args.by == "grasp"
                             else [D.BIN_SHORT[c] for c in cols]))
        for s in scenes:
            # Blank, not 0, for a pair the table never asked for.
            w.writerow([s] + ["" if (s, c) not in cell else cell[(s, c)]
                              for c in cols])

    if not args.no_detail:
        det = out.with_name(out.stem + "_detail.csv")
        with det.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["scene"], r["grasp_idx"])))

    # ---- summary -----------------------------------------------------------
    n = len(rows)
    print(f"\n{n} episodes over {len(scenes)} scenes, "
          f"{len(cols)} {'grasp slot' if args.by == 'grasp' else 'bin'}(s)")
    print(f"  thresholds: {args.pos_thresh} m / {args.rot_thresh} rad\n")
    print(f"  {'criterion':<12} {'passed':>7} {'rate':>8}")
    for c in CRITERIA:
        k = sum(r[c] for r in rows)
        print(f"  {c:<12} {k:>7} {100*k/n:>7.1f}%"
              + ("   <- used for the matrix" if c == args.criterion else ""))
    agree = sum(1 for r in rows if r["reach"] == r["close_label"]) / n
    print(f"\n  reach vs close_label agree on {100*agree:.1f}% of episodes")

    by_bin = defaultdict(lambda: [0, 0])
    for r in rows:
        b = int(r["bin_assigned"])
        if b < 0:
            continue
        by_bin[b][0] += int(r[args.criterion])
        by_bin[b][1] += 1
    if by_bin:
        print(f"\n  per bin ({args.criterion}):")
        for b in sorted(by_bin):
            k, t = by_bin[b]
            print(f"    {D.BIN_SHORT[b]:<3} {k:5d}/{t:<5d} {100*k/t:6.1f}%")

    fail = [r for r in rows if not r[args.criterion]]
    if fail:
        pe = np.array([r["pos_err_m"] for r in fail])
        st = np.array([r["steps"] for r in fail])
        ok_st = np.array([r["steps"] for r in rows if r[args.criterion]])
        print(f"\n  the {len(fail)} failures stop a median "
              f"{np.median(pe)*1000:.0f} mm short and run "
              f"{np.median(st):.0f} steps against {np.median(ok_st):.0f}")

    print(f"\nwrote {out}")
    if not args.no_detail:
        print(f"wrote {out.with_name(out.stem + '_detail.csv')}   "
              f"(per episode: axis, pos/rot error, steps, every criterion)")


if __name__ == "__main__":
    main()
