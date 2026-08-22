"""
The Regrasp go/no-go gate: which approach-direction bins are reachable at all?

Runs OFFLINE — no simulator, no GPU, no planner, seconds not hours — because
every input is already on disk:

    handover-sim/handover/data/dex-ycb-cache/pose_*.npz    hand + object trajectories
    handover-sim/handover/data/assets/<subject>_<side>/    the MANO wrist offset
    output/grasp_cand_table_*.json                         the candidate grasps

WHY THIS IS THE GATE AND NOT A SANITY CHECK. The whole premise of Regrasp is that
`k` is a test-time knob: command a direction, and if that grasp fails, command a
different one. A bin no scene can reach is not a retry hypothesis — it is a
direction the training data will never cover and the policy will be extrapolating
into. Finding that out costs a 20-hour collection if you find it afterwards and
about a minute if you find it here.

WHAT IT CANNOT TELL YOU. The candidate table holds the FPS-selected 8 of a median
49-grasp goal set, chosen by a pose metric that is blind to approach direction. So
a bin could be reachable in the full goal set and absent from this subsample. FPS
maximises diversity, which biases the subsample TOWARD spread rather than away, so
a bin that reads zero here is very unlikely to be real elsewhere — but "zero"
should still be confirmed against the full goal set when the direction table is
built. Read a small count as "check this", not as "this many".

The object centroid here is the object's ORIGIN from the DexYCB pose, whereas at
runtime the anchor uses the observed point-cloud centroid. They differ by the
offset between an object's origin and its visible surface centre, a few cm. That
moves the anchor's x-axis by a few degrees at the wrist-object distances involved
and cannot flip a 90-deg-separated bin assignment, so it is fine for a
feasibility census and is NOT fine for labelling — the real table build uses the
observed centroid.

    python examples/analyze_direction_feasibility.py
    python examples/analyze_direction_feasibility.py --split val --k 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from handover_sim2real.regrasp import anchor as A            # noqa: E402
from handover_sim2real.regrasp import directions as D        # noqa: E402

_DATA = _REPO / "handover-sim" / "handover" / "data"
_CACHE = _DATA / "dex-ycb-cache"
_ASSETS = _DATA / "assets"

TABLE_HEIGHT = 0.92                       # handover/config.py:31; added to BOTH the
                                          # hand (mano.py:37) and the object
                                          # (ycb.py:74), so it cancels in c - wrist
                                          # but matters for the base fallback.
PANDA_BASE = np.array([0.61, -0.50, 0.875])       # ENV.PANDA_BASE_POSITION
EVAL_SKIP_OBJECT = (0, 15)


def scene_ids(setup: str = "s0", split: str = "train") -> list[int]:
    """Reproduce HandoverBenchmarkWrapper's scene list without building an env."""
    if setup != "s0":
        raise SystemExit("only s0 is reproduced here; others need the wrapper")
    if split == "train":
        seq = {i for i in range(100) if i % 5 != 4}
        subj = set(range(10))
    elif split in ("val", "test"):
        seq = {i for i in range(100) if i % 5 == 4}
        subj = {0, 1} if split == "val" else set(range(2, 10))
    else:
        raise SystemExit(f"unknown split {split}")
    return [i for i in range(1000)
            if (i // 5) % 20 not in EVAL_SKIP_OBJECT
            and (i // 100) in subj and (i % 100) in seq]


_WRIST_RE = re.compile(
    r'<joint name="joint7".*?<origin xyz="([^"]+)"', re.S)


def wrist_offsets() -> dict:
    """{'<subject>_<side>': [3]} — joint7's origin, i.e. the MANO wrist in link6.

    Link 7 is the wrist/palm. NOT link 0 (a massless base pinned at the world
    origin) and NOT link 6 (the floating-base root, which sits ~9 cm off the
    actual wrist — exactly this offset).
    """
    out = {}
    for d in sorted(_ASSETS.iterdir()):
        urdf = d / "mano.urdf"
        if not urdf.exists():
            continue
        m = _WRIST_RE.search(urdf.read_text())
        if m:
            out[d.name] = np.array([float(x) for x in m.group(1).split()])
    return out


def scene_anchor(scene_id: int, offsets: dict):
    """(R_anchor, meta) for one scene at its START frame, or None if unusable."""
    meta_p = _CACHE / f"meta_{scene_id:03d}.json"
    pose_p = _CACHE / f"pose_{scene_id:03d}.npz"
    if not (meta_p.exists() and pose_p.exists()):
        return None
    meta = json.loads(meta_p.read_text())
    z = np.load(pose_p)
    pose_m, pose_y = z["pose_m"], z["pose_y"]

    frame = len(pose_m) - 1                      # ENV.YCB_MANO_START_FRAME: last
    side = meta["mano_sides"][0]
    subject = meta["name"].split("/")[0]

    # MANO's validity window. `mano.reset` only builds the body when the frame is
    # inside it; outside, `env.mano.body` is None and there IS no wrist to anchor
    # to. This is the check that decides whether the wrist frame is usable at all.
    nz = np.nonzero(np.any(pose_m[:, 0, :] != 0.0, axis=1))[0]
    hand_present = bool(len(nz) and nz[0] <= frame <= nz[-1])

    # cached pose_m is dstack((translation, 48 euler)); the first 3 euler are the
    # floating-base rotation. See dex_ycb.py's cache builder.
    t = pose_m[frame, 0, 0:3].astype(np.float64).copy()
    e = pose_m[frame, 0, 3:6].astype(np.float64)
    t[2] += TABLE_HEIGHT
    off = offsets.get(f"{subject}_{side}")
    wrist = None
    if hand_present and off is not None:
        wrist = t + Rot.from_euler("XYZ", e).as_matrix() @ off

    # cached pose_y is dstack((translation, euler)); grasped object only.
    c = pose_y[frame, int(meta["ycb_grasp_ind"]), 0:3].astype(np.float64).copy()
    c[2] += TABLE_HEIGHT

    R, m = A.anchor_rotation(c, wrist, PANDA_BASE, A.AnchorState())
    m.update({"hand_present": hand_present, "side": side,
              "centroid": c, "wrist": wrist})
    return R, m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="output/grasp_cand_table_train_p5.json")
    p.add_argument("--split", default="train")
    p.add_argument("--k", type=int, default=6,
                   help="6 = the octahedral bins; anything else = Fibonacci")
    p.add_argument("--out", default=None, help="write the per-scene census as JSON")
    args = p.parse_args()

    bins = D.BINS if args.k == 6 else D.fibonacci_directions(args.k)
    names = (D.BIN_NAMES if args.k == 6
             else tuple(f"fib{i}" for i in range(args.k)))

    tbl = json.load(open(args.table))
    tbl.pop("_meta", None)
    ids = scene_ids("s0", args.split)

    per_scene, counts = {}, np.zeros(len(bins), dtype=np.int64)
    scenes_with = np.zeros(len(bins), dtype=np.int64)
    n_no_hand = n_fallback = n_no_cache = n_no_cands = 0
    pair_sep, pair_bins = [], Counter()
    sides = Counter()

    offsets = wrist_offsets()
    for idx, sid in enumerate(ids):
        entry = tbl.get(str(idx))
        got = scene_anchor(sid, offsets)
        if got is None:
            n_no_cache += 1
            continue
        R, m = got
        sides[m["side"]] += 1
        if not m["hand_present"]:
            n_no_hand += 1
        if m["mode"] == "base":
            n_fallback += 1
        if not isinstance(entry, dict) or not entry.get("grasps"):
            n_no_cands += 1
            continue

        T = np.array([g["ee_pose_world"] for g in entry["grasps"]], dtype=np.float64)
        d_anchor = np.array([D.from_world(D.approach_direction(t), R) for t in T])
        feas = sorted({D.bin_of(d, bins) for d in d_anchor})
        for b in feas:
            scenes_with[b] += 1
        for d in d_anchor:
            counts[D.bin_of(d, bins)] += 1
        per_scene[idx] = {"scene_id": sid, "bins": feas, "mode": m["mode"],
                          "side": m["side"], "hand_present": m["hand_present"]}

        pair = D.most_separated_pair(feas, empties=scenes_with, bins=bins)
        if pair is not None:
            per_scene[idx]["pair"] = list(pair)
            pair_sep.append(float(D.angle_between(bins[pair[0]], bins[pair[1]])))
            pair_bins[tuple(sorted(pair))] += 1

    n = len(per_scene)
    print("=" * 74)
    print(f"Regrasp direction feasibility   split={args.split}  k={args.k}")
    print(f"  scenes in split      : {len(ids)}")
    print(f"  usable (cache+cands) : {n}")
    print(f"  hand ABSENT at start : {n_no_hand}"
          + ("   <-- the wrist anchor is undefined for these" if n_no_hand else ""))
    print(f"  anchor fell back     : {n_fallback}  (hand directly over/under object)")
    print(f"  handedness           : {dict(sides)}")
    print("=" * 74)

    print(f"\n  {'bin':<18} {'candidates':>11} {'scenes reaching it':>20}")
    for i, nm in enumerate(names):
        flag = ""
        if scenes_with[i] == 0:
            flag = "   <-- UNREACHABLE"
        elif scenes_with[i] < 0.05 * max(n, 1):
            flag = "   <-- thin"
        print(f"  {nm:<18} {counts[i]:>11} {scenes_with[i]:>13} "
              f"({100*scenes_with[i]/max(n,1):4.1f}%){flag}")

    live = int((scenes_with > 0).sum())
    print(f"\n  live bins: {live} of {len(bins)}"
          + ("" if live == len(bins) else
             f"   -> the retry ladder has {live} hypotheses, not {len(bins)}"))

    if pair_sep:
        ps = np.asarray(pair_sep)
        print(f"\n  scenes able to supply a PAIR : {len(ps)}/{n} "
              f"({100*len(ps)/max(n,1):.1f}%)")
        print(f"  pair separation (deg)        : median {np.median(ps):.0f}  "
              f"min {ps.min():.0f}  antipodal {100*(ps>179).mean():.0f}%")
        print("  most common pairs:")
        for (a, b), c in pair_bins.most_common(6):
            print(f"    {names[a]:<18} + {names[b]:<18} {c:>4}  "
                  f"({D.angle_between(bins[a], bins[b]):.0f} deg)")

    print("\n  A pair is what breaks the confound: at ONE demo per scene, d is a\n"
          "  deterministic function of the observation across the dataset and the\n"
          "  network can ignore the conditioning entirely. Scenes that cannot\n"
          "  supply two separated bins contribute nothing to that.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"_meta": {"split": args.split, "k": args.k, "table": args.table,
                       "counts": counts.tolist(),
                       "scenes_with": scenes_with.tolist(),
                       "n_no_hand": n_no_hand, "n_fallback": n_fallback},
             **{str(k): v for k, v in per_scene.items()}}, indent=1, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
