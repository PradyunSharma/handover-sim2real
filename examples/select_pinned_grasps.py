"""
Prune the Phase-5 K-candidate collection down to N pinned grasps per scene.

This is stage 3 of the Phase-5 data pipeline and the single place that decides
what the training set contains:

    build_grasp_pin_table_multi.py   K=8 diverse candidates per scene
    collect_bc_dataset_multi.py      one demonstration per (scene, candidate)
    select_pinned_grasps.py          <- keep the N=4 most separated that WORKED

**Why the selection happens here and not in the table builder.** About a quarter
of pinned demonstrations fail — the OMG plan clips the object while translating
laterally into the pre-grasp — and the builder cannot know which. Choosing four
up front and hoping all four plan would keep ~0.76^4 = 33% of scenes. Collecting
eight and choosing four afterwards keeps ~90%, at the cost of one extra
collection pass.

**The completion test is `filter_demos.py`'s**, verbatim in effect: an episode has
a gripper-CLOSE label if and only if it ran to the end, because
`collect_bc_dataset_multi.py` appends the closure transition only `if not done`.
Episodes whose clouds are not [T, n_pts, C] are also dropped — a short cloud
kills the DataLoader's collate from inside a worker with nothing naming the
episode (it killed Phase-4 run 19 three minutes in). So `filter_demos.py` does not
need to be run before this; running it first would be harmless but its
`--scenes-out` list is wrong for Phase 5, since it drops a whole scene when any
single slot fails.

**Re-running the max-min selection over the survivors** (rather than taking the
first four by FPS rank) matters: if slots 1 and 2 failed, the survivors 0, 3, 4, 5
may be poorly spread, and a fresh farthest-point pass over exactly the poses that
worked gives the best four available. The seed is the surviving candidate with the
lowest original FPS rank, so slot 0 stays OMG's own pick wherever that one
survived — `slot0_is_omg` records per scene whether it did, which is what makes a
slot-0-only comparison against Phase-4 run 16 legitimate.

Writes three files:

    <out>.json          the final N-grasp pin table (SIM.grasp_pin_table)
    <out>.h5            the pruned dataset, N episodes per scene (TRAIN.base_train_h5)
    <out>_excluded.json scenes that could not supply N (SIM.exclude_scenes)

    python examples/select_pinned_grasps.py \\
        --demos output/bc_dataset/train_p5_k8.h5 \\
        --cand-table output/grasp_cand_table_train_p5.json \\
        --n-final 4 --sep-floor 0.02 \\
        --out output/bc_dataset/train_p5

NOTE ON WHAT THIS COSTS, inherited from `filter_demos.py`: dropping failed
demonstrations makes the remainder easier by construction, and Phase 5 drops
harder still because a scene needs N successes rather than one. A success rate
measured afterwards is on that easier subset and is not comparable with one
measured over the full split — say so when reporting it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Import grasp_select DIRECTLY, not as handover_sim2real.dagger5.grasp_select.
# The module itself needs only numpy, but `from pkg.mod import ...` executes the
# package __init__ first, and dagger5/__init__.py pulls gym, pybullet and the
# handover envs — so a stage-3 prune that touches nothing but HDF5 and numpy
# would otherwise demand the full simulator PYTHONPATH. Putting the package
# directory on sys.path bypasses __init__ entirely and keeps this script (and
# analyze_grasp_separation.py) runnable on a login node with a bare env.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "handover_sim2real" / "dagger5"))

import h5py
import numpy as np

from grasp_select import grasp_distance_matrix, select_diverse_grasps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", required=True, help="the raw K-candidate HDF5")
    p.add_argument("--cand-table", required=True,
                   help="the K-candidate table the demos were collected against")
    p.add_argument("--out", required=True,
                   help="output prefix; writes <out>.h5, <out>.json, <out>_excluded.json")
    p.add_argument("--n-final", type=int, default=4)
    p.add_argument("--sep-floor", type=float, default=0.02,
                   help="minimum separation (m) between the kept grasps")
    p.add_argument("--dry-run", action="store_true",
                   help="report the attrition, write nothing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    h5_out, tbl_out = out.with_suffix(".h5"), out.with_suffix(".json")
    excl_out = out.parent / f"{out.name}_excluded.json"

    cand = json.load(open(args.cand_table))
    cand_meta = cand.pop("_meta", {})

    # ── pass 1: read every episode's outcome and target ──────────────────────
    per_scene = defaultdict(list)     # scene -> [(episode_key, grasp_idx, pose, ok)]
    n_total = n_malformed = n_failed = 0
    with h5py.File(args.demos, "r") as f:
        file_attrs = dict(f.attrs)
        keys = sorted(k for k in f if k.startswith("episode"))
        # Modal cloud width, as filter_demos.py does — taken from the file rather
        # than hardcoded so this works for any uniform_num_pts.
        widths = defaultdict(int)
        for k in keys:
            widths[int(f[k]["point_clouds"].shape[1])] += 1
        n_pts = max(widths, key=widths.get) if widths else 0

        for k in keys:
            g = f[k]
            n_total += 1
            scene = int(g.attrs["scene_idx"])
            gi = int(g.attrs["grasp_idx"])
            pose = np.asarray(g.attrs["grasp_pose_world"], dtype=np.float64)
            ok = True
            if int(g["point_clouds"].shape[1]) != n_pts:
                ok, n_malformed = False, n_malformed + 1
            elif not (g["expert_actions"][:, 6] < 0.5).any():
                ok, n_failed = False, n_failed + 1
            per_scene[scene].append((k, gi, pose, ok))

    # ── pass 2: choose N per scene out of the slots that worked ──────────────
    table = {"_meta": {**cand_meta,
                       "phase": 5,
                       "n_final": int(args.n_final),
                       "sep_floor_m": float(args.sep_floor),
                       "selected_from": str(args.demos),
                       "cand_table": str(args.cand_table),
                       "selected": time.strftime("%Y-%m-%d %H:%M:%S")}}
    chosen = {}                       # scene -> [(episode_key, orig_slot, pose)]
    excluded, reasons = [], defaultdict(int)
    seps_kept = []
    for scene in sorted(per_scene):
        survivors = [(k, gi, pose) for (k, gi, pose, ok) in per_scene[scene] if ok]
        if len(survivors) < args.n_final:
            excluded.append(scene)
            reasons["too_few_successful_demos"] += 1
            continue

        # Re-run farthest-point sampling over exactly the poses that worked,
        # seeded at the lowest surviving FPS rank (slot 0 = OMG's pick when it
        # survived), so slot 0 stays comparable with Phase-4 run 16.
        survivors.sort(key=lambda t: t[1])
        poses = np.stack([p for (_, _, p) in survivors])
        idxs, seps = select_diverse_grasps(
            poses, seed_idx=0, k=args.n_final, sep_floor=args.sep_floor,
            dist=grasp_distance_matrix(poses))
        if len(idxs) < args.n_final:
            excluded.append(scene)
            reasons["too_few_separated_grasps"] += 1
            continue

        picked = [survivors[i] for i in idxs]
        chosen[scene] = picked
        seps_kept.extend(s for s in seps[1:] if np.isfinite(s))
        table[str(scene)] = {
            "n_candidates": len(per_scene[scene]),
            "n_successful": len(survivors),
            "slot0_is_omg": bool(picked[0][1] == 0),
            "grasps": [{"ee_pose_world": pose.tolist(),
                        "orig_slot": int(gi),
                        "fps_rank": rank,
                        "min_sep_m": (None if not np.isfinite(seps[rank])
                                      else float(seps[rank]))}
                       for rank, (_, gi, pose) in enumerate(picked)],
        }

    n_scenes = len(per_scene)
    n_kept = len(chosen)
    print(f"{args.demos}")
    print(f"  episodes              : {n_total}  over {n_scenes} scenes")
    print(f"  dropped (failed)      : {n_failed}  ({100*n_failed/max(n_total,1):.1f}%)")
    print(f"  dropped (malformed)   : {n_malformed}")
    print(f"  scenes kept           : {n_kept}/{n_scenes}  "
          f"({100*n_kept/max(n_scenes,1):.1f}%)")
    print(f"  scenes excluded       : {len(excluded)}  {dict(reasons)}")
    print(f"  base episodes         : {n_kept * args.n_final}")
    if seps_kept:
        s = np.asarray(seps_kept)
        print(f"  separation kept (m)   : median {np.median(s):.4f}  "
              f"min {s.min():.4f}  p10 {np.percentile(s, 10):.4f}")
    n_omg0 = sum(1 for v in table.values()
                 if isinstance(v, dict) and v.get("slot0_is_omg"))
    print(f"  slot 0 == OMG's pick  : {n_omg0}/{n_kept}  "
          f"(the subset comparable with Phase-4 run 16)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    # ── write the pruned dataset ─────────────────────────────────────────────
    h5_out.parent.mkdir(parents=True, exist_ok=True)
    steps = ep_idx = 0
    with h5py.File(args.demos, "r") as f, h5py.File(h5_out, "w") as g:
        for k, v in file_attrs.items():
            g.attrs[k] = v
        g.attrs["filtered_from"] = str(args.demos)
        g.attrs["filter_rule"] = (
            "kept the n_final most-separated demonstrations per scene among those "
            "with a CLOSE label and [T, n_pts, C] clouds; scenes that could not "
            "supply n_final were dropped entirely")
        g.attrs["grasps_per_scene"] = int(args.n_final)
        g.attrs["grasp_pin_table"] = str(tbl_out)
        for scene in sorted(chosen):
            for slot, (key, orig_slot, pose) in enumerate(chosen[scene]):
                src = f[key]
                dst = g.create_group(f"episode_{ep_idx:05d}")
                for kk, vv in src.attrs.items():
                    dst.attrs[kk] = vv
                # Renumber to the final 0..N-1 the policy will be conditioned on,
                # keeping the original for traceability back to the K-candidate
                # collection.
                dst.attrs["grasp_idx"] = slot
                dst.attrs["orig_grasp_idx"] = int(orig_slot)
                for name in ("point_clouds", "robot_states", "expert_actions"):
                    dst.create_dataset(name, data=src[name][:], compression="gzip")
                steps += len(src["expert_actions"])
                ep_idx += 1
        g.attrs["num_episodes"] = ep_idx

    with tbl_out.open("w") as fh:
        json.dump(table, fh, indent=1)
    with excl_out.open("w") as fh:
        json.dump(sorted(excluded), fh)

    print(f"\nwrote {h5_out}    ({ep_idx} episodes, {steps} steps)")
    print(f"wrote {tbl_out}   ({n_kept} scenes x {args.n_final} grasps)")
    print(f"wrote {excl_out}  ({len(excluded)} scene indices to exclude)")
    print(f"\nPoint TRAIN.base_train_h5 at the .h5, SIM.grasp_pin_table at the "
          f".json, and SIM.exclude_scenes at the _excluded.json. All three move "
          f"together — swapping one without the others silently produces nonsense.")


if __name__ == "__main__":
    main()
