"""
Drop failed demonstrations, and record which scenes they came from.

An episode in a BC dataset has NO gripper-close label if and only if it failed:
`collect_bc_dataset.py` plays the OMG plan and appends the closure transition
only `if not done`, and `done` at that point means the benchmark terminated the
episode — human contact, object drop, or timeout. Measured on train_pinned.h5:
126 of 623 episodes, 20%.

Those episodes are the approach phase of a handover that went wrong, and they are
currently training data. Observed cause: the planned trajectory clips the object
while translating laterally into the pre-grasp pose, knocking it out of the hand.
Imitating that is worse than not training on it at all.

This writes two things:

  <out>.h5     the same dataset with the failed episodes removed
  <out>.json   the scene indices they came from, for DAGGER/SIM.exclude_scenes —
               so the DAgger loop neither collects on nor evaluates on scenes
               whose expert is broken

    python examples/filter_demos.py \\
        --in  output/bc_dataset/train_pinned.h5 \\
        --out output/bc_dataset/train_pinned_ok.h5

NOTE ON WHAT THIS COSTS. Dropping ~20% of scenes makes the remaining set easier
by construction: every scene where the expert itself collides is gone. A success
rate measured afterwards is on that easier subset and is NOT comparable to one
measured on the full split — say so when reporting it. The alternative fix is to
make OMG avoid the object during the lateral approach, which addresses the cause
rather than the symptom but is a much larger change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--scenes-out", default=None,
                   help="JSON list of dropped scene indices (default: <out>.json)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be dropped, write nothing")
    args = p.parse_args()

    dst = Path(args.dst)
    scenes_out = Path(args.scenes_out) if args.scenes_out else dst.with_suffix(".json")

    keep, drop, drop_scenes = [], [], []
    with h5py.File(args.src, "r") as f:
        attrs = dict(f.attrs)
        for key in sorted(k for k in f if k.startswith("episode")):
            acts = f[key]["expert_actions"][:]
            scene = int(f[key].attrs["scene_idx"])
            if (acts[:, 6] < 0.5).any():          # has a CLOSE label -> completed
                keep.append(key)
            else:
                drop.append(key)
                drop_scenes.append(scene)

        n = len(keep) + len(drop)
        print(f"{args.src}")
        print(f"  episodes        : {n}")
        print(f"  kept (completed): {len(keep)}  ({100*len(keep)/max(n,1):.1f}%)")
        print(f"  dropped (failed): {len(drop)}  ({100*len(drop)/max(n,1):.1f}%)")
        print(f"  distinct scenes dropped: {len(set(drop_scenes))}")
        if args.dry_run:
            print("\n--dry-run: nothing written")
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst, "w") as g:
            for k, v in attrs.items():
                g.attrs[k] = v
            g.attrs["num_episodes"] = len(keep)
            g.attrs["filtered_from"] = str(args.src)
            g.attrs["filter_rule"] = "dropped episodes with no CLOSE label (= benchmark failure)"
            steps = 0
            for i, key in enumerate(keep):
                src_grp = f[key]
                out = g.create_group(f"episode_{i:05d}")
                for k, v in src_grp.attrs.items():
                    out.attrs[k] = v
                for name in ("point_clouds", "robot_states", "expert_actions"):
                    out.create_dataset(name, data=src_grp[name][:], compression="gzip")
                steps += len(src_grp["expert_actions"])

    with scenes_out.open("w") as fh:
        json.dump(sorted(set(drop_scenes)), fh)

    print(f"\nwrote {dst}   ({len(keep)} episodes, {steps} steps)")
    print(f"wrote {scenes_out}  ({len(set(drop_scenes))} scene indices to exclude)")
    print(f"\nPoint TRAIN.base_train_h5 at the filtered .h5 and "
          f"SIM.exclude_scenes at the .json.")


if __name__ == "__main__":
    main()
