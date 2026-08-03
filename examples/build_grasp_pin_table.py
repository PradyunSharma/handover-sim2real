"""
Build the per-scene grasp pin table: ONE committed grasp per scene.

Why this exists. OMG re-decides its goal grasp on every plan —
`goal_idx = argmin ||traj.start - goal_set[i]||` in joint space
(`OMG-Planner/omg/planner.py`, `ol_alg='Proj'`), where `traj.start` is the
CURRENT joint configuration. Phase-4 DAgger replans every step with the policy
driving, so the target grasp can move mid-episode: measured over 90 replans under
+-15 cm / +-0.5 rad perturbation, 32 selected a different grasp, one scene cycled
through four of them, and the target shifted by up to 10 cm. The demonstrations
have the same problem in reverse — they plan once from the home configuration, so
they commit to whatever argmin picked there, which need not be what DAgger picks
later.

Labels that point at grasp A for ten steps and grasp B for the next ten are
exactly the inconsistency DAgger cannot average away. This table pins one grasp
per scene, and both the demonstration collector and the DAgger collector load it,
so every episode of every phase aims at the same pose.

Selection rule (`--mode`):
    furthest_from_hand  (default) argmax of the gripper-to-MANO-hand clearance —
                        the same geometry the hand-collision filter thresholds,
                        used here as a continuous score instead of a reject test.
    omg                 whatever OMG's own argmin picks from the home config.

`--tol` breaks near-ties in favour of OMG's pick: among grasps within `tol`
metres of the best clearance, keep OMG's choice. A pure argmax (tol=0) will
happily move the grasp 8.5 cm to gain 0.1 mm of clearance; ~0.01 avoids that.

The table stores the chosen grasp's **world EE pose**, not its index. The goal set
is rebuilt per episode and its ordering is an IK artifact, so an index is not a
durable key; at load time the pose is matched back to the nearest goal-set entry.

Usage:
    python examples/build_grasp_pin_table.py \\
        --cfg-file examples/configs/dagger_phase4.yaml \\
        --out output/grasp_pin_table.json

    # ties resolved toward OMG's natural pick
    python examples/build_grasp_pin_table.py --tol 0.01 --out ...

The table MUST be rebuilt whenever the grasp set changes — i.e. when
`SIM.valid_grasp_dict_path` or the runtime `hand_collision_filter` settings
change — because those change which candidates exist.
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

from handover_sim2real.dagger import build_sim_cfg, build_sim_context


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg-file", default="examples/configs/dagger_phase4.yaml",
                   help="Phase-4 config; its SIM block defines the grasp set")
    p.add_argument("--out", default="output/grasp_pin_table.json")
    p.add_argument("--mode", default="furthest_from_hand",
                   choices=["furthest_from_hand", "omg"])
    p.add_argument("--tol", type=float, default=0.0,
                   help="near-tie tolerance (m) resolved toward OMG's pick")
    p.add_argument("--num-scenes", type=int, default=None, help="cap for a quick run")
    p.add_argument("--split", default=None, help="override SIM.split")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg4 = yaml.safe_load(open(args.cfg_file))
    sim_d = dict(cfg4["SIM"])
    if args.split:
        sim_d["split"] = args.split
    sim_cfg = build_sim_cfg(sim_d)
    sim = build_sim_context(sim_cfg, sim_d, seed=0)
    env = sim.env

    n = env.num_scenes if args.num_scenes is None else min(env.num_scenes, args.num_scenes)
    print(f"Building grasp pin table: {n} scenes  split={sim_cfg.BENCHMARK.SPLIT}  "
          f"mode={args.mode}  tol={args.tol}")
    print(f"  grasp set: valid_grasp_dict={sim_d.get('valid_grasp_dict_path')}  "
          f"hand_collision_filter={sim_d.get('hand_collision_filter')}")

    table = {
        "_meta": {
            "mode": args.mode,
            "tol": args.tol,
            "split": sim_cfg.BENCHMARK.SPLIT,
            "setup": sim_cfg.BENCHMARK.SETUP,
            "valid_grasp_dict_path": sim_d.get("valid_grasp_dict_path"),
            "hand_collision_filter": bool(sim_d.get("hand_collision_filter", False)),
            "hand_collision_thresh": float(sim_d.get("hand_collision_thresh", 0.08)),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = n_nohand = n_changed = 0
    gains, shifts = [], []
    t0 = time.time()
    for sc in range(n):
        env.reset(idx=sc)
        sim.point_listener.reset()
        plan, _ = env.run_omg_planner(int(sim_cfg.RL_MAX_STEP), sc, reset_scene=True)
        if plan is None:
            n_fail += 1
            table[str(sc)] = None
            continue

        clr = env.grasp_hand_clearances()
        poses = env.goal_set_ee_poses()
        omg_i = env.get_omg_goal_idx()
        if clr is None or len(clr) == 0:
            # No hand in this scene: nothing to be furthest from, so keep OMG's
            # pick. Still pinned, which is the point.
            n_nohand += 1
            idx, clearance = omg_i, None
        else:
            idx = int(env.select_goal_grasp(mode=args.mode, tol=args.tol))
            clearance = float(clr[idx])
            gains.append(float(clr[idx] - clr[omg_i]))
            shifts.append(float(np.linalg.norm(poses[idx][:3, 3] - poses[omg_i][:3, 3])))
            n_changed += int(idx != omg_i)

        table[str(sc)] = {
            "ee_pose_world": poses[idx].tolist(),   # the durable key
            "goal_set_size": int(len(poses)),
            "omg_idx": int(omg_i),
            "chosen_idx": int(idx),                 # advisory only
            "hand_clearance_m": clearance,
        }
        n_ok += 1

        if (sc + 1) % 25 == 0 or sc == n - 1:
            print(f"  [{sc+1:4d}/{n}] pinned={n_ok} omg_failed={n_fail} "
                  f"changed_vs_omg={n_changed} ({time.time()-t0:.0f}s)", flush=True)
            with out.open("w") as f:
                json.dump(table, f, indent=1)

    with out.open("w") as f:
        json.dump(table, f, indent=1)

    print(f"\nDone -> {out}")
    print(f"  scenes pinned        : {n_ok}/{n}")
    print(f"  OMG failed (no entry): {n_fail}")
    print(f"  no hand (kept OMG)   : {n_nohand}")
    print(f"  differs from OMG pick: {n_changed}")
    if gains:
        g = np.asarray(gains); s = np.asarray(shifts)
        print(f"  clearance gain (m)   : median {np.median(g):.4f}  max {g.max():.4f}")
        print(f"  grasp moved (m)      : median {np.median(s):.4f}  max {s.max():.4f}")
        print(f"  moved >5 cm for <5 mm gain: {int(((s > 0.05) & (g < 0.005)).sum())}"
              f"  (raise --tol to suppress these)")


if __name__ == "__main__":
    main()
