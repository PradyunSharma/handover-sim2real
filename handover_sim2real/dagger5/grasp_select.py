"""Picking N maximally-separated grasps out of OMG's goal set (Phase 5).

Phase 4 pinned ONE grasp per scene. Phase 5 pins four, and conditions the policy
on which of them it is currently being asked to reach, so a failed handover can be
retried under a different grasp. That needs an answer to "how far apart are two
grasps", and the obvious answers are all wrong here.

**Position-only Euclidean is wrong** because two grasps at the same point with
perpendicular approach axes are completely different grasps. **Naive SE(3)
(position + geodesic rotation) is worse than wrong**: `omg/planner.py`'s
`augment_flip_grasp` deliberately appends, for every grasp, a duplicate rotated by
pi about `panda_joint7` — i.e. pi about the gripper's own approach axis. A
parallel-jaw gripper is symmetric under that rotation, so the twin is *the same
physical grasp*, yet it sits at the maximum possible rotation distance. A max-min
selector run on a naive metric picks flip twins first and returns four poses that
are really two, with contradictory rotation labels for the same physical target.
`grasp_pin.py` records the empirical signature of this (a handful of episodes
closing 3.1413 rad from their pin while p99 was 0.0029 rad).

So the metric here is the mean displacement of the gripper's six control points,
**minimised over the two-element flip group** — a proper quotient metric on
SE(3)/Z2. It is in metres, it weights orientation by its real effect on where the
fingers end up (the same argument that motivates `LOSS.pose_loss: pm`), and it
returns 0 for a flip twin, which is exactly what a selector needs.

The control points are `hand_finger_point` from `GA-DDPG/core/utils.py` — the same
six the hand-collision filter and `grasp_hand_clearances` use
(`train_env.py:_hand_grasp_collision_mask`), transposed to row-major. Note these
are NOT the array in `bc5/losses.py:_GRIPPER_CONTROL_POINTS`, which places the
finger offsets on x instead of y; the two differ by a 90-degree twist about the
approach axis. Both happen to be flip-invariant, so either would work here, but
this file follows the collision filter because that is the set with a physical
claim attached to it.
"""

from __future__ import annotations

import numpy as np

# Rows, metres, panda_hand frame. == GA-DDPG's hand_finger_point.T. The first two
# rows are both the wrist origin (that duplication is in the original array), so
# the origin carries weight 2/6 in the mean — harmless, and kept so the numbers
# here are comparable with the clearance figures in the pin tables.
GRIPPER_CONTROL_POINTS = np.array(
    [[0.0,  0.000, 0.000],
     [0.0,  0.000, 0.000],
     [0.0,  0.053, 0.075],
     [0.0, -0.053, 0.075],
     [0.0,  0.053, 0.105],
     [0.0, -0.053, 0.105]], dtype=np.float64)          # [6, 3]

# The gripper's approach axis is +z in the hand frame, so the jaw-swap symmetry is
# pi about z. Applied on the RIGHT (T @ FLIP) so it is a rotation in the gripper's
# own frame, not the world's.
FLIP = np.array([[-1.0, 0.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0, 0.0],
                 [0.0,  0.0, 1.0, 0.0],
                 [0.0,  0.0, 0.0, 1.0]], dtype=np.float64)

SYMMETRY_GROUP = (np.eye(4), FLIP)


def control_points(poses):
    """(..., 4, 4) world EE poses -> (..., 6, 3) world control points."""
    poses = np.asarray(poses, dtype=np.float64)
    R = poses[..., :3, :3]
    t = poses[..., :3, 3]
    return np.einsum("...ij,kj->...ki", R, GRIPPER_CONTROL_POINTS) + t[..., None, :]


def grasp_distance(a, b):
    """Flip-invariant control-point distance between two 4x4 world EE poses, in
    metres. Broadcasts over leading dimensions.

    Returns ~0 for a wrist-flip twin, which is the whole point — see the module
    docstring. Mean (not sum) over the six points, so the number reads as "how far
    the gripper moved", comparable with `hand_clearance_m` and the close
    thresholds.
    """
    cp_a = control_points(a)                                    # [..., 6, 3]
    best = None
    for g in SYMMETRY_GROUP:
        cp_b = control_points(np.asarray(b, dtype=np.float64) @ g)
        d = np.linalg.norm(cp_a - cp_b, axis=-1).mean(axis=-1)   # [...]
        best = d if best is None else np.minimum(best, d)
    return best


def grasp_distance_matrix(poses):
    """(n, 4, 4) -> (n, n) symmetric flip-invariant distances, zero diagonal."""
    poses = np.asarray(poses, dtype=np.float64)
    n = len(poses)
    if n == 0:
        return np.zeros((0, 0))
    cp = control_points(poses)                                  # [n, 6, 3]
    cp_flip = control_points(poses @ FLIP)                      # [n, 6, 3]
    # ||cp[i] - cp[j]|| and ||cp[i] - cp_flip[j]||, minimised elementwise.
    d0 = np.linalg.norm(cp[:, None] - cp[None, :], axis=-1).mean(-1)
    d1 = np.linalg.norm(cp[:, None] - cp_flip[None, :], axis=-1).mean(-1)
    d = np.minimum(d0, d1)
    d = np.minimum(d, d.T)          # the flip term is not symmetric on its own
    np.fill_diagonal(d, 0.0)
    return d


def select_diverse_grasps(poses, seed_idx=0, k=8, sep_floor=0.02, dist=None):
    """Greedy farthest-point sampling over a goal set, seeded at `seed_idx`.

    `poses` is (n, 4, 4) world EE poses — pass `env.goal_set_ee_poses()`, which is
    already IK-feasible, obstacle-free and hand-collision-filtered. `seed_idx`
    should be `env.get_omg_goal_idx()` so that slot 0 is byte-identical to the
    Phase-4 `--mode omg` pin and Phase 5 slot 0 stays comparable with run 16.

    Stops early when the next-best candidate is closer than `sep_floor` to
    something already chosen, so the length of the result IS the number of
    physically distinct grasps this scene can offer. A scene returning fewer than
    4 cannot be used.

    Returns `(indices, min_seps)`: the chosen indices in FPS order, and for each
    the distance to its nearest already-chosen neighbour (`inf` for the seed).
    """
    poses = np.asarray(poses, dtype=np.float64)
    n = len(poses)
    if n == 0:
        return [], []
    d = grasp_distance_matrix(poses) if dist is None else np.asarray(dist)

    seed = int(seed_idx) if 0 <= int(seed_idx) < n else 0
    chosen = [seed]
    seps = [float("inf")]
    # Distance from every candidate to the nearest chosen one, maintained
    # incrementally (this is what makes FPS O(k n) rather than O(k n^2)).
    near = d[:, seed].copy()
    while len(chosen) < int(k):
        near[chosen] = -1.0
        j = int(np.argmax(near))
        if near[j] < float(sep_floor):
            break
        chosen.append(j)
        seps.append(float(near[j]))
        near = np.minimum(near, d[:, j])
    return chosen, seps


def pairwise_mean_distance(poses):
    """Mean over the n(n-1)/2 unordered pairs of `grasp_distance`. The denominator
    of the `cond_track` diagnostic (and, with EE poses instead of grasps, its
    numerator). Returns nan for fewer than two poses."""
    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) < 2:
        return float("nan")
    d = grasp_distance_matrix(poses)
    iu = np.triu_indices(len(poses), k=1)
    return float(d[iu].mean())
