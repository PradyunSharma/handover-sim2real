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
    p.add_argument("--members-per-bin", type=int, default=5,
                   help="how many goal-set grasps to RECORD per bin, closest to "
                        "the bin axis first (run 2 recorded 1). Recording more "
                        "than are wanted is the point: `assign_direction_demos "
                        "--per-bin N` then picks N of them OFFLINE, so changing "
                        "how many demonstrations a bin gets costs seconds "
                        "instead of another simulator pass. 5 covers the "
                        "3-per-bin setting with headroom at ~5x the pose "
                        "payload (a few MB).")
    # ---- WHAT `d` IS DERIVED FROM. See directions.py for the measurement. ----
    p.add_argument("--d-rule", default="approach_axis",
                   choices=["approach_axis", "grasp_offset"],
                   help="approach_axis (default, runs 1-9): d = -R_grasp[:,2], "
                        "'which side the gripper comes from'. grasp_offset "
                        "(run 10): d = centroid -> the point between the "
                        "fingertips, 'which part of the object the fingers "
                        "close on' — independent of the grasp's ORIENTATION. "
                        "THEY ARE NOT THE SAME QUESTION: measured on s0/train, "
                        "grasp_offset makes -z the third-largest bin (526 "
                        "grasps over 235 scenes) where approach_axis has ZERO, "
                        "because you cannot approach from beneath a held object "
                        "but you can close on its underside. Rebuilds the whole "
                        "assignment, so a table built under one rule cannot be "
                        "used by a run configured for the other — the rule is "
                        "written into `_meta` and checked downstream.")
    p.add_argument("--d-point-depth", type=float, default=None,
                   help="grasp_offset only: metres along the gripper's local +z "
                        "to the point d is measured to. Default 0.1122 = the "
                        "fingertip end of the Panda pads. 0.1034 is the pad "
                        "centre; 0.0 is the PALM ORIGIN, which is 12.5 cm from "
                        "the centroid (never degenerate) and a median 14.5 deg "
                        "from -R[:,2] — i.e. position-derived but answering the "
                        "approach_axis question. Use it if the fingertip rule "
                        "proves too noisy.")
    p.add_argument("--d-min-offset", type=float, default=0.02,
                   help="grasp_offset only: drop a grasp whose point lands "
                        "closer than this to the centroid, where the direction "
                        "is centroid noise rather than geometry. Measured: the "
                        "fingertip sits a median 3.85 cm out and 14.4%% of "
                        "grasps are inside 2 cm. 0 disables.")
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

    # The rule, resolved ONCE and written into `_meta`. Every downstream stage
    # compares against it rather than assuming, because a table built under one
    # rule and consumed under the other is silently wrong: the bins are
    # populated by different grasps and nothing about the file's shape says so.
    rule = D.DirectionRule(
        rule=str(args.d_rule),
        depth=(D.FINGERTIP_DEPTH if args.d_point_depth is None
               else float(args.d_point_depth)),
        min_offset=(float(args.d_min_offset)
                    if args.d_rule == "grasp_offset" else 0.0))

    table = {"_meta": {
        "phase": "regrasp", "schema": "direction-table-v1",
        "split": args.split, "setup": str(cfg.BENCHMARK.SETUP),
        "cfg_file": args.cfg_file, "k": int(args.k),
        "bin_names": names, "bins": np.asarray(bins).tolist(),
        "max_angle_deg": float(args.max_angle),
        "valid_grasp_dict_path": args.valid_grasp_dict,
        "centroid_source": "observed point cloud (ycb channel), EE frame -> world",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        **rule.as_meta(),
    }}

    scenes_with = np.zeros(len(bins), dtype=np.int64)
    goal_hist = np.zeros(len(bins), dtype=np.int64)
    member_spread: list[float] = []
    n_ok = n_no_plan = n_no_hand = n_fallback = n_no_object = n_short = 0
    modes, sides = Counter(), Counter()
    t0 = time.time()

    print("=" * 74)
    print(f"Regrasp direction table   split={args.split}  scenes {lo}..{hi - 1}")
    print(f"  k={args.k}  max_angle={args.max_angle} deg  -> {out}")
    print(f"  d = {rule.describe()}")
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
        #
        # `d_rule` decides what `d` MEANS (directions.py): the approach axis
        # `-R[:,2]`, or the direction from the object centroid to the point
        # between the fingertips. THIS IS THE STAGE THAT DECIDES IT for the whole
        # pipeline — every later stage reads `d_anchor` out of this file or
        # recomputes it under the rule recorded in `_meta`, so the rule cannot be
        # changed downstream without rebuilding here.
        #
        # `grasp_offset` can return None (the fingertip point lands closer to the
        # centroid than `--d-min-offset`, where the direction is centroid noise
        # rather than geometry). Those grasps are dropped from the assignment
        # rather than binned, and counted.
        d_list = [rule.of(T, c_world) for T in poses]
        keepmask = np.array([d is not None for d in d_list])
        n_short += int((~keepmask).sum())
        if not keepmask.any():
            n_no_plan += 1
            table[str(idx)] = None
            continue
        d_world = np.stack([d if d is not None else np.zeros(3) for d in d_list])
        d_anchor = np.stack([D.from_world(d, R) for d in d_world])
        ang = D.angles_to_bins(d_anchor, bins)              # [n_grasps, k]
        assign = np.argmin(ang, axis=1)
        best_ang = ang[np.arange(len(ang)), assign]
        best_ang = np.where(keepmask, best_ang, np.inf)     # drop the degenerate

        entries = []
        for b in range(len(bins)):
            members = np.flatnonzero((assign == b) & (best_ang <= args.max_angle))
            goal_hist[b] += len(members)
            if not len(members):
                continue
            # Closest to the bin axis: the most canonical realisation of "come
            # from this direction", and the least likely to be re-binned by a
            # small anchor difference at collection time.
            order = members[np.argsort(best_ang[members], kind="stable")]
            pick = int(order[0])
            # MEMBERS, ordered by the same rule as `pick`. Run 2 recorded only
            # the head of this list and could therefore demonstrate a bin exactly
            # once; recording the first few lets `assign_direction_demos
            # --per-bin N` emit N demonstrations of ONE command without another
            # simulator pass. The ordering is load-bearing — the assignment takes
            # a PREFIX of it, so `members[0]` must stay the closest-to-axis grasp
            # that run 2 used, which is what keeps `--per-bin 1` byte-identical
            # to run 2's table.
            keep = order[:max(1, int(args.members_per_bin))]
            entries.append({
                "bin": b, "bin_name": names[b],
                "ee_pose_world": poses[pick].tolist(),
                "d_anchor": d_anchor[pick].tolist(),
                "d_world": d_world[pick].tolist(),
                "angle_to_axis_deg": float(best_ang[pick]),
                "goal_set_idx": pick,
                "n_members": int(len(members)),
                "members": [{
                    "ee_pose_world": poses[int(j)].tolist(),
                    "d_anchor": d_anchor[int(j)].tolist(),
                    "d_world": d_world[int(j)].tolist(),
                    "angle_to_axis_deg": float(best_ang[int(j)]),
                    "goal_set_idx": int(j),
                } for j in keep],
            })
            # HOW DIFFERENT THE RECORDED MEMBERS ACTUALLY ARE. Three
            # demonstrations of one command are worth collecting only if they are
            # three different grasps; if the goal set holds near-duplicates, this
            # buys 3x the episodes, 3x the mode-averaging under a unimodal loss,
            # and no new information. Palm-origin spread in metres is the cheap
            # read on that, printed in the summary so it is seen before a
            # 3x-longer collection is launched rather than after.
            if len(keep) > 1:
                pts = np.stack([poses[int(j)][:3, 3] for j in keep])
                member_spread.append(float(np.linalg.norm(
                    pts[:, None, :] - pts[None, :, :], axis=-1).max()))
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
        # grasp_offset only: goal-set members dropped because the point landed
        # inside `--d-min-offset` of the centroid. Zero under approach_axis.
        "n_short_offset": int(n_short),
        "scenes_with_bin": scenes_with.tolist(),
        "goal_set_bin_histogram": goal_hist.tolist(),
        "members_per_bin": int(args.members_per_bin),
        "handedness": dict(sides), "elapsed_s": round(time.time() - t0, 1),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))

    print("\n" + "=" * 74)
    print(f"  planned            : {n_ok}/{hi - lo}   (no plan {n_no_plan}, "
          f"no object points {n_no_object})")
    print(f"  hand absent        : {n_no_hand}      anchor fallback: {n_fallback}")
    if rule.needs_centroid():
        print(f"  d = {rule.describe()}")
        print(f"  dropped (offset < {rule.min_offset * 100:.1f} cm): {n_short} "
              f"goal-set members — the direction there is centroid noise")
    print(f"  handedness         : {dict(sides)}")
    print(f"\n  {'bin':<18} {'goal-set grasps':>16} {'scenes reaching it':>20}")
    for b, nm in enumerate(names):
        flag = "   <-- UNREACHABLE" if scenes_with[b] == 0 else (
            "   <-- thin" if scenes_with[b] < 0.05 * max(n_ok, 1) else "")
        print(f"  {nm:<18} {goal_hist[b]:>16} {scenes_with[b]:>13} "
              f"({100 * scenes_with[b] / max(n_ok, 1):4.1f}%){flag}")
    live = int((scenes_with > 0).sum())
    print(f"\n  live bins: {live}/{len(bins)}   retry ladder has {live} rungs")

    # ARE THE EXTRA MEMBERS ACTUALLY DIFFERENT GRASPS? Recording N per bin is
    # worth a 3x-longer collection only if the N are distinct; near-duplicates
    # buy 3x the episodes, 3x the mode-averaging under a unimodal regression
    # loss, and nothing else. Read this BEFORE launching the collection.
    if member_spread:
        ms = np.asarray(member_spread)
        print(f"\n  members recorded   : up to {args.members_per_bin} per bin, "
              f"{len(ms)} bins with more than one")
        print(f"  spread within a bin: median {np.median(ms) * 100:.1f} cm   "
              f"p10 {np.percentile(ms, 10) * 100:.1f}   "
              f"max {ms.max() * 100:.1f}   (palm origins, widest pair)")
        thin = float((ms < 0.02).mean())
        if thin > 0.25:
            print(f"  WARNING {100 * thin:.0f}% of bins spread under 2 cm — those "
                  f"extra demonstrations are near-duplicates of the first and "
                  f"will mostly add mode-averaging, not information.")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")
    print("Next: examples/assign_direction_demos.py — pure combinatorics over this "
          "file, no simulator.")


if __name__ == "__main__":
    main()
