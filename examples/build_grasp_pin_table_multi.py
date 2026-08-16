"""
Build the Phase-5 grasp candidate table: K maximally-separated grasps per scene.

Phase 4 pinned ONE grasp per scene (`build_grasp_pin_table.py`) for the reason
documented there: OMG re-decides `goal_idx = argmin ||traj.start - goal_set[i]||`
on every plan, so an unpinned target moves mid-episode and labels that point at
grasp A for ten steps and grasp B for the next ten are exactly the inconsistency
DAgger cannot average away.

Phase 5 keeps the pinning and adds a second axis: the policy is *conditioned* on
which grasp it is being asked to reach, so each scene needs several. This script
emits K candidates per scene, in farthest-point-sampling order:

    slot 0   OMG's own pick from the home configuration — byte-identical to
             `build_grasp_pin_table.py --mode omg`, so Phase-5 slot 0 stays
             comparable with Phase-4 run 16.
    slot 1.. greedily the grasp furthest from everything chosen so far, under
             the flip-invariant control-point metric in `dagger5/grasp_select.py`.

**Why K = 8 when only 4 are wanted.** About a quarter of pinned demonstrations
fail — the plan clips the object on the lateral approach — and grasps chosen for
separation will be no easier than OMG's natural pick. Demanding that all four of
four plan successfully would keep only ~0.76^4 = 33% of scenes. So over-provision
here, collect demonstrations for all K, and let `select_pinned_grasps.py` choose
the final four from the slots that actually worked. The extra collection pass is
cheap next to a 15 h trainer.

**Why the metric is not naive SE(3).** `omg/planner.py:augment_flip_grasp` appends
a wrist-flipped duplicate of every grasp — pi about the gripper's own approach
axis, under which a parallel-jaw gripper is symmetric. A twin is the same physical
grasp at maximal rotation distance, so a naive max-min selector picks twins first
and returns four poses that are really two. See `dagger5/grasp_select.py`.

The table stores each grasp's **world EE pose**, not its index: the goal set is
rebuilt per episode, its ordering is an IK artifact, and `omg/planner.py` shuffles
the survivors with `np.random.choice`. At load time the pose is matched back.

Usage:
    python examples/build_grasp_pin_table_multi.py \\
        --cfg-file examples/configs/dagger_phase5_run1.yaml \\
        --split train --k 8 --sep-floor 0.02 \\
        --out output/grasp_cand_table_train_p5.json

    # SLURM
    sbatch --export=ALL,SPLIT=train,OUT=output/grasp_cand_table_train_p5.json \\
        examples/slurm/build_pin_table_multi.sbatch

The table MUST be rebuilt whenever the grasp set changes — i.e. when
`SIM.valid_grasp_dict_path` or the runtime `hand_collision_filter` settings change
— because those change which candidates exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import yaml

from handover_sim2real.dagger5 import build_sim_cfg, build_sim_context
from handover_sim2real.dagger5.grasp_select import (
    grasp_distance_matrix,
    select_diverse_grasps,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg-file", default="examples/configs/dagger_phase5_run1.yaml",
                   help="Phase-5 config; its SIM block defines the grasp set")
    p.add_argument("--out", default="output/grasp_cand_table_train_p5.json")
    p.add_argument("--k", type=int, default=8,
                   help="candidates per scene (over-provision; 4 survive)")
    p.add_argument("--n-final", type=int, default=4,
                   help="how many the pipeline will eventually keep — recorded in "
                        "_meta and used only to report the at-risk scene count")
    p.add_argument("--sep-floor", type=float, default=0.02,
                   help="minimum control-point separation (m) between candidates; "
                        "FPS stops when the next best is closer than this")
    p.add_argument("--num-scenes", type=int, default=None, help="cap for a quick run")
    p.add_argument("--split", default=None, help="override SIM.split")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg5 = yaml.safe_load(open(args.cfg_file))
    sim_d = dict(cfg5["SIM"])
    if args.split:
        sim_d["split"] = args.split
    sim_cfg = build_sim_cfg(sim_d)
    sim = build_sim_context(sim_cfg, sim_d, seed=0)
    env = sim.env

    n = env.num_scenes if args.num_scenes is None else min(env.num_scenes, args.num_scenes)
    print(f"Building grasp candidate table: {n} scenes  split={sim_cfg.BENCHMARK.SPLIT}  "
          f"k={args.k}  sep_floor={args.sep_floor} m")
    print(f"  grasp set: valid_grasp_dict={sim_d.get('valid_grasp_dict_path')}  "
          f"hand_collision_filter={sim_d.get('hand_collision_filter')}")

    table = {
        "_meta": {
            "mode": "omg_seeded_fps",
            "metric": "flip_invariant_control_point",
            "k_candidates": int(args.k),
            "n_final": int(args.n_final),
            "sep_floor_m": float(args.sep_floor),
            "split": sim_cfg.BENCHMARK.SPLIT,
            "setup": sim_cfg.BENCHMARK.SETUP,
            "valid_grasp_dict_path": sim_d.get("valid_grasp_dict_path"),
            "hand_collision_filter": bool(sim_d.get("hand_collision_filter", False)),
            "hand_collision_thresh": float(sim_d.get("hand_collision_thresh", 0.08)),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": 5,
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_nohand = n_short = 0
    counts, min_seps = [], []
    t0 = time.time()
    for sc in range(n):
        env.reset(idx=sc)
        sim.point_listener.reset()
        plan, _ = env.run_omg_planner(int(sim_cfg.RL_MAX_STEP), sc, reset_scene=True)
        if plan is None:
            n_fail += 1
            table[str(sc)] = None
            continue

        poses = env.goal_set_ee_poses()
        clr = env.grasp_hand_clearances()
        omg_i = int(env.get_omg_goal_idx())
        if clr is None or len(clr) == 0:
            n_nohand += 1
            clr = None

        dist = grasp_distance_matrix(poses)
        idxs, seps = select_diverse_grasps(
            poses, seed_idx=omg_i, k=args.k, sep_floor=args.sep_floor, dist=dist)

        # Fewer than n_final distinct grasps means this scene can never supply a
        # full set. Keep the entry anyway — select_pinned_grasps.py is the single
        # place that decides what to drop, and it wants to report why.
        if len(idxs) < args.n_final:
            n_short += 1

        table[str(sc)] = {
            "goal_set_size": int(len(poses)),
            "omg_idx": omg_i,
            "n_candidates": len(idxs),
            "grasps": [
                {
                    "ee_pose_world": poses[i].tolist(),      # the durable key
                    "fps_rank": rank,
                    "is_omg_pick": bool(i == omg_i),
                    "goal_set_idx": int(i),                  # advisory only
                    "hand_clearance_m": (None if clr is None else float(clr[i])),
                    "min_sep_m": (None if not np.isfinite(s) else float(s)),
                }
                for rank, (i, s) in enumerate(zip(idxs, seps))
            ],
        }
        n_ok += 1
        counts.append(len(idxs))
        min_seps.extend(s for s in seps[1:] if np.isfinite(s))

        if (sc + 1) % 25 == 0 or sc == n - 1:
            print(f"  [{sc+1:4d}/{n}] planned={n_ok} omg_failed={n_fail} "
                  f"short(<{args.n_final})={n_short} ({time.time()-t0:.0f}s)", flush=True)
            with out.open("w") as f:
                json.dump(table, f, indent=1)

    with out.open("w") as f:
        json.dump(table, f, indent=1)

    print(f"\nDone -> {out}")
    print(f"  scenes with candidates : {n_ok}/{n}")
    print(f"  OMG failed (no entry)  : {n_fail}")
    print(f"  no hand (clearance nan): {n_nohand}")
    print(f"  fewer than {args.n_final} distinct : {n_short}  "
          f"(these cannot survive select_pinned_grasps.py)")
    if counts:
        c = np.asarray(counts)
        print(f"  candidates/scene       : median {int(np.median(c))}  "
              f"min {c.min()}  max {c.max()}  "
              f"(=={args.k} on {int((c == args.k).sum())} scenes)")
    if min_seps:
        s = np.asarray(min_seps)
        print(f"  separation achieved (m): median {np.median(s):.4f}  "
              f"p10 {np.percentile(s, 10):.4f}  max {s.max():.4f}")


if __name__ == "__main__":
    main()
