#!/usr/bin/env python3
"""Derive a PRE-GRASP demonstration set from an existing grasp one — no simulator.

`DAGGER.target: pregrasp` (run 21) moves the terminal state 6.4 cm back along the
approach: the learner drives to the standoff, commits, and a blind feed-forward
push does the rest (`handover_sim2real/dagger/pregrasp.py`). The DAgger shards
are collected that way by the collector itself, but D_0 — the expert
demonstrations the base fit is trained on — was collected against the grasp and
would otherwise teach the policy to reach in and close, contradicting every label
the run then aggregates on top of it.

THE RE-COLLECTION IS UNNECESSARY, because the pre-grasp episode is a PREFIX of
the grasp episode. `collect_bc_dataset.collect_episode` plays the OMG plan by
index and appends one close, so an episode of T steps decomposes as

    row i  in [0, len(plan))   state before waypoint i, label = delta to it
    row len(plan)              state at the grasp,      label = CLOSE

and OMG's last `reach_tail` waypoints are the standoff ramp, so waypoint
`len(plan) - reach_tail` IS the standoff. The state recorded one row later is
therefore the state at the standoff — the exact state a pre-grasp collection
would have terminated on, with the exact point cloud it would have observed.
Keeping rows 0..len(plan)-reach_tail+1 and rewriting that last row's action as
CLOSE is the whole conversion:

    keep = T - reach_tail + 1        (17 of 21 at the defaults)

Doing it this way is not just cheaper, it is BETTER: run 21's approach labels,
scenes and clouds become bit-identical to run 16's, so the two runs differ only
in the endgame and a difference in outcome cannot be a difference in data.

SELF-CHECK. The script measures, per episode, the displacement from the kept
terminal state to the state the demonstration actually closed at, decomposed in
the terminal EE frame. The axial component is what `DAGGER.forward_dist` has to
be; the lateral component is correction the blind push cannot make, and is the
honest upper bound on what this run gives away. On the 472 kept episodes of
train_pinned_omg_ok.h5 that reads 0.0642 m axial / 0.0088 m lateral.

    python examples/make_pregrasp_demos.py \
        --input  output/bc_dataset/train_pinned_omg_wlr_ok.h5 \
        --output output/bc_dataset/train_pinned_omg_wlr_ok_pregrasp.h5
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np
from transforms3d.quaternions import quat2mat

# robot_state layout (collect_bc_dataset._robot_state): joint_pos(9) +
# joint_vel(9) + ee_xyz(3) + ee_wxyz(4) + gripper_norm(1) + prev_act(6)
EE_POS = slice(18, 21)
EE_QUAT = slice(21, 25)
CLOSE_LABEL = np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="grasp-target HDF5 to convert")
    p.add_argument("--output", required=True, help="pre-grasp HDF5 to write")
    p.add_argument("--reach-tail", type=int, default=5,
                   help="OMG cfg.reach_tail_length — how many trailing plan "
                        "waypoints are the standoff ramp (default 5)")
    return p.parse_args()


def reach_geometry(rs, keep):
    """(axial, lateral, d_rot) from the kept terminal state to the closing state.

    Decomposed in the terminal EE frame, because that is the frame the blind push
    is commanded in: `axial` is the +z distance `forward_dist` has to cover, and
    `lateral` is the part of the demonstrated reach that is off that axis and so
    cannot be reproduced without looking.
    """
    p0 = rs[keep - 1, EE_POS].astype(np.float64)
    p1 = rs[-1, EE_POS].astype(np.float64)
    R0 = quat2mat(rs[keep - 1, EE_QUAT].astype(np.float64))
    R1 = quat2mat(rs[-1, EE_QUAT].astype(np.float64))
    d = p1 - p0
    axial = float(d @ R0[:, 2])
    lateral = float(np.linalg.norm(d - axial * R0[:, 2]))
    cos = (np.trace(R0.T @ R1) - 1.0) / 2.0
    return axial, lateral, float(np.arccos(np.clip(cos, -1.0, 1.0)))


def main():
    args = parse_args()
    tail = int(args.reach_tail)

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        names = sorted(n for n in fin
                       if isinstance(fin[n], h5py.Group)
                       and "expert_actions" in fin[n])
        kept = skipped = 0
        geom = []
        for name in names:
            grp = fin[name]
            acts = grp["expert_actions"][:]
            T = len(acts)
            # Only an episode that ran the plan to the end has the ramp structure
            # this conversion assumes. `filter_demos.py` already drops the rest
            # ("dropped episodes with no CLOSE label"), so this should never fire
            # on an *_ok.h5 — it is here so that it fails loudly if it does,
            # rather than silently truncating an episode at the wrong pose.
            if T <= tail or float(acts[-1, 6]) >= 0.5:
                print(f"  skip {name}: T={T}, ends_closed={acts[-1, 6] < 0.5} "
                      f"— not a completed plan")
                skipped += 1
                continue

            keep = T - tail + 1
            rs = grp["robot_states"][:]
            geom.append(reach_geometry(rs, keep))

            acts = acts[:keep].copy()
            acts[-1] = CLOSE_LABEL          # commit, 6.4 cm short of the grasp
            out = fout.create_group(name)
            for k, v in grp.attrs.items():
                out.attrs[k] = v
            out.attrs["num_steps"] = keep
            out.create_dataset("point_clouds", data=grp["point_clouds"][:keep],
                               compression="gzip")
            out.create_dataset("robot_states", data=rs[:keep], compression="gzip")
            out.create_dataset("expert_actions", data=acts, compression="gzip")
            kept += 1

        fout.attrs["num_episodes"] = kept
        # PROVENANCE. A pre-grasp file and a grasp file have the identical schema
        # and are silently mixable, and mixing them puts two contradictory close
        # labels on the same states. These attrs are the only thing that tells
        # them apart, so they are not optional decoration.
        fout.attrs["target"] = "pregrasp"
        fout.attrs["derived_from"] = str(args.input)
        fout.attrs["reach_tail"] = tail
        fout.attrs["conversion"] = (
            "truncated to T - reach_tail + 1 rows; last action -> CLOSE. The "
            "terminal state is the OMG standoff (traj[-reach_tail]); the reach "
            "beyond it is DAGGER.forward_dist, executed open loop.")

        g = np.asarray(geom, dtype=np.float64)
        if len(g):
            # Recorded so a run can be audited against the data it came from
            # without re-deriving this: forward_dist should be the axial mean.
            fout.attrs["reach_axial_mean"] = float(g[:, 0].mean())
            fout.attrs["reach_lateral_mean"] = float(g[:, 1].mean())

    print(f"\nepisodes kept    : {kept}   (skipped {skipped})")
    print(f"output           : {args.output}")
    if len(g):
        print("\nremaining reach from the new terminal state to the grasp, in that "
              "state's EE frame:")
        for label, col in zip(("axial (+z)", "lateral", "d_rot (rad)"), g.T):
            print(f"  {label:12s} mean {col.mean():.4f}  median "
                  f"{np.median(col):.4f}  p90 {np.percentile(col, 90):.4f}  "
                  f"max {col.max():.4f}")
        print(f"\n-> set DAGGER.forward_dist to ~{g[:, 0].mean():.3f} "
              f"(default 0.064 = standoff_dist * (1 - 1/reach_tail))")
        print(f"-> the lateral column is correction the blind push cannot make")


if __name__ == "__main__":
    main()
