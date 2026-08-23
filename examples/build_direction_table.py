"""
Stage 1 of the Regrasp pipeline: the per-scene direction table.

One simulator pass over the split. Per scene it records everything the offline
pair assignment needs, so that assignment becomes pure combinatorics — seconds,
re-runnable with a different `k` or a different tie-break, with no GPU.

    per scene
      wrist_world        the giver's wrist at the start frame (MANO link 7)
      centroid_world     the OBSERVED object point-cloud centroid, not the pose
      anchor_R           the frame the bins live in
      anchor_mode        "wrist" | "base" -- did the degenerate fallback fire
      mano_side          "left" | "right"
      goal_set_size      how many grasps OMG offered
      bins[]             for EACH reachable bin: the best grasp realising it,
                         its d_anchor, its angle to the bin axis, its hand
                         clearance, and how many goal-set members fell in it

WHY THIS REPLACES `build_grasp_pin_table_multi.py`. That one stored the
FPS-selected 8 of a median-49 goal set, chosen by a pose metric that is blind to
approach direction, plus no wrist and no centroid. So the anchor could not be
built from it at all, and a bin present in the full goal set could be missing
from the subsample. `examples/analyze_direction_feasibility.py` gets a useful
census out of it anyway, but the numbers there are explicitly provisional; these
are the ones to trust.

WHY THE OBSERVED CENTROID AND NOT THE OBJECT POSE. At runtime the anchor and the
`d . normalize(p_i - c)` channel both use `pc[pc[:,3] > 0.5, :3].mean(0)` — the
centroid of the points the cameras actually returned, which is a visible-surface
centroid offset from the object's origin by a few cm. Recording the pose-derived
centroid here would put a systematic offset between the bin a demonstration was
ASSIGNED and the frame the policy sees, for no benefit. One `env.reset` plus one
cloud read per scene buys the consistency.

SIZE. Six bin entries per scene rather than every goal-set member, so the JSON
stays a few hundred KB rather than tens of MB, and the full-goal-set census is
preserved as per-bin counts.

    python examples/build_direction_table.py --split train \\
        --out output/direction_table_train.json

    # SLURM (needs a GPU partition: OMG plans on the GPU)
    sbatch --export=ALL,SPLIT=train,OUT=output/direction_table_train.json \\
        examples/slurm/build_direction_table.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from handover_sim2real.regrasp import anchor as A                  # noqa: E402
from handover_sim2real.regrasp import directions as D              # noqa: E402
from handover_sim2real.regrasp.channels import object_centroid     # noqa: E402
from handover_sim2real.regrasp.env_setup import (                  # noqa: E402
    build_sim_cfg, build_sim_context, preflight,
)
from collect_bc_dataset import _point_cloud                        # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg-file", default="examples/pretrain_multicam_wr.yaml",
                   help="simulator config; MUST match what the demos will be "
                        "collected with — the cloud is renderer-dependent and the "
                        "centroid comes from the cloud")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--out", default=None)
    p.add_argument("--num-scenes", type=int, default=None,
                   help="cap, for a smoke run")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--k", type=int, default=6, help="6 = octahedral bins")
    p.add_argument("--max-angle", type=float, default=45.0,
                   help="a grasp counts toward a bin only within this angle of "
                        "its axis (45 = the Voronoi half-angle at k=6, so every "
                        "grasp lands in exactly one bin)")
    p.add_argument("--valid-grasp-dict", default="examples/valid_grasp_dict_005.pkl")
    p.add_argument("--egl", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preflight()
    out = Path(args.out or f"output/direction_table_{args.split}.json")

    sim_cfg_d = {
        "cfg_file": args.cfg_file,
        "split": args.split,
        "egl": bool(args.egl),
        "valid_grasp_dict_path": args.valid_grasp_dict,
        "hand_collision_filter": False,
        "use_standoff": True,
        "standoff_dist": 0.08,
    }
    cfg = build_sim_cfg(sim_cfg_d)
    sim = build_sim_context(cfg, sim_cfg_d, seed=args.seed)
    env = sim.env

    bins = D.BINS if args.k == 6 else D.fibonacci_directions(args.k)
    names = (list(D.BIN_NAMES) if args.k == 6
             else [f"fib{i}" for i in range(args.k)])
    base = np.asarray(cfg.ENV.PANDA_BASE_POSITION, dtype=np.float64)

    n_total = sim.num_scenes
    lo = int(args.start)
    hi = min(n_total, lo + args.num_scenes) if args.num_scenes else n_total

    table = {"_meta": {
        "phase": "regrasp", "schema": "direction-table-v1",
        "split": args.split, "setup": str(cfg.BENCHMARK.SETUP),
        "cfg_file": args.cfg_file, "k": int(args.k),
        "bin_names": names, "bins": np.asarray(bins).tolist(),
        "max_angle_deg": float(args.max_angle),
        "valid_grasp_dict_path": args.valid_grasp_dict,
        "centroid_source": "observed point cloud (ycb channel), EE frame -> world",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}

    scenes_with = np.zeros(len(bins), dtype=np.int64)
    goal_hist = np.zeros(len(bins), dtype=np.int64)
    n_ok = n_no_plan = n_no_hand = n_fallback = n_no_object = 0
    modes, sides = Counter(), Counter()
    t0 = time.time()

    print("=" * 74)
    print(f"Regrasp direction table   split={args.split}  scenes {lo}..{hi - 1}")
    print(f"  k={args.k}  max_angle={args.max_angle} deg  -> {out}")
    print("=" * 74)

    for idx in range(lo, hi):
        obs = env.reset(idx=idx)
        sim.point_listener.reset()

        # The anchor's inputs, read at the START frame -- which is the whole
        # episode under YCB_MANO_START_FRAME: last, since the hand is static there.
        wrist = A.wrist_world(env)
        side = A.handedness(env)
        if side:
            sides[side] += 1
        if wrist is None:
            n_no_hand += 1

        pc = _point_cloud(obs, sim.point_listener, sim.panda_base_inv_tf)
        c_ee = object_centroid(pc, fallback_to_all=False)
        if c_ee is None:
            n_no_object += 1
            table[str(idx)] = None
            print(f"  [{idx:4d}] no object points — cannot anchor")
            continue
        c_world = A.centroid_to_world(
            c_ee, obs, sim.panda_base_inv_tf,
            cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)

        R, meta = A.anchor_rotation(c_world, wrist, base, A.AnchorState())
        modes[meta["mode"]] += 1
        if meta["mode"] == "base":
            n_fallback += 1

        plan, _ = env.run_omg_planner(int(cfg.RL_MAX_STEP), idx, reset_scene=True)
        if plan is None:
            n_no_plan += 1
            table[str(idx)] = None
            continue
        poses = np.asarray(env.goal_set_ee_poses(), dtype=np.float64)
        if len(poses) == 0:
            n_no_plan += 1
            table[str(idx)] = None
            continue

        # EVERY goal-set member, not an FPS subsample -- this is the whole point.
        d_world = np.stack([D.approach_direction(T) for T in poses])
        d_anchor = np.stack([D.from_world(d, R) for d in d_world])
        ang = D.angles_to_bins(d_anchor, bins)              # [n_grasps, k]
        assign = np.argmin(ang, axis=1)
        best_ang = ang[np.arange(len(ang)), assign]

        entries = []
        for b in range(len(bins)):
            members = np.flatnonzero((assign == b) & (best_ang <= args.max_angle))
            goal_hist[b] += len(members)
            if not len(members):
                continue
            # Closest to the bin axis: the most canonical realisation of "come
            # from this direction", and the least likely to be re-binned by a
            # small anchor difference at collection time.
            pick = int(members[int(np.argmin(best_ang[members]))])
            entries.append({
                "bin": b, "bin_name": names[b],
                "ee_pose_world": poses[pick].tolist(),
                "d_anchor": d_anchor[pick].tolist(),
                "d_world": d_world[pick].tolist(),
                "angle_to_axis_deg": float(best_ang[pick]),
                "goal_set_idx": pick,
                "n_members": int(len(members)),
            })
            scenes_with[b] += 1

        table[str(idx)] = {
            "goal_set_size": int(len(poses)),
            "wrist_world": None if wrist is None else wrist.tolist(),
            "centroid_world": c_world.tolist(),
            "anchor_R": R.tolist(),
            "anchor_mode": meta["mode"],
            "anchor_horiz_norm": float(meta["horiz_norm"]),
            "mano_side": side,
            "hand_present": wrist is not None,
            "bins": entries,
        }
        n_ok += 1
        if (idx - lo) % 25 == 0 or idx == hi - 1:
            el = time.time() - t0
            done = idx - lo + 1
            print(f"  [{idx:4d}] {done}/{hi - lo}  ok={n_ok} noplan={n_no_plan}  "
                  f"bins={[e['bin_name'].split('_')[0] for e in entries]}  "
                  f"({el / max(done, 1):.1f}s/scene)")

    table["_meta"].update({
        "n_scenes": hi - lo, "n_ok": n_ok, "n_no_plan": n_no_plan,
        "n_no_object": n_no_object, "n_no_hand": n_no_hand,
        "n_anchor_fallback": n_fallback,
        "scenes_with_bin": scenes_with.tolist(),
        "goal_set_bin_histogram": goal_hist.tolist(),
        "handedness": dict(sides), "elapsed_s": round(time.time() - t0, 1),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))

    print("\n" + "=" * 74)
    print(f"  planned            : {n_ok}/{hi - lo}   (no plan {n_no_plan}, "
          f"no object points {n_no_object})")
    print(f"  hand absent        : {n_no_hand}      anchor fallback: {n_fallback}")
    print(f"  handedness         : {dict(sides)}")
    print(f"\n  {'bin':<18} {'goal-set grasps':>16} {'scenes reaching it':>20}")
    for b, nm in enumerate(names):
        flag = "   <-- UNREACHABLE" if scenes_with[b] == 0 else (
            "   <-- thin" if scenes_with[b] < 0.05 * max(n_ok, 1) else "")
        print(f"  {nm:<18} {goal_hist[b]:>16} {scenes_with[b]:>13} "
              f"({100 * scenes_with[b] / max(n_ok, 1):4.1f}%){flag}")
    live = int((scenes_with > 0).sum())
    print(f"\n  live bins: {live}/{len(bins)}   retry ladder has {live} rungs")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")
    print("Next: examples/assign_direction_demos.py — pure combinatorics over this "
          "file, no simulator.")


if __name__ == "__main__":
    main()
