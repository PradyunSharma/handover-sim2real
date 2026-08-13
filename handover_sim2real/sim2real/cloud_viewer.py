"""Live 3D view of the exact point cloud the policy is being fed.

Enabled with `my_policy_runner.py --show-cloud`, alongside the 2D segmentation
overlay that is always on. The two answer different questions and neither
replaces the other: the 2D overlay says whether the hand was *segmented*, this
says whether the resulting points landed in the right *place*.

WHAT IS DRAWN, and why each piece earns its place:

  the 1024 policy points   Not the pre-sampling clouds — the literal [1024, 5]
                           tensor, split by its own one-hot channels. If a class
                           was under-full and got padded by repetition, you see
                           the duplicates piled up, which is the honest picture
                           of what the network received.
  the gripper wireframe    The cloud is in the panda_hand frame, whose origin is
                           invisible in a bare scatter plot. Without a gripper to
                           refer to, "the object is 8 cm ahead of the fingers"
                           and "the object is 8 cm behind them" look identical —
                           and the second one means the calibration is inverted.
                           Drawn from GA-DDPG's control points, so it is the same
                           geometry the point-matching loss scored during
                           training.
  the exclusion box        Only when robot exclusion is on. A box you cannot see
                           is a box you cannot tell is eating the object.

COLOUR MODES, toggled with 'c' in the 3D window:

  by class    object vs hand — what the network's one-hot channels say.
  by camera   which rig each point came from. This is the multi-camera
              diagnostic: fused into one cloud, a tripod whose extrinsics are
              wrong looks like a slightly noisy object, and the only way to see
              the two sets sitting apart from each other is to colour them
              differently. Expect the two cameras' points to OVERLAP on the
              shared surfaces; a rigid offset between them is a calibration
              error, and its size is the error.

The window is updated in-place (never re-added), so your rotation/zoom survives
across frames. It is also rate-limited: Open3D re-uploads the whole buffer on
every update, and doing that at camera rate steals time from the control loop
for a view no human can read that fast.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# GA-DDPG's Panda control points (core/utils.get_control_point_tensor), in the
# panda_hand frame: fingertips at z = 0.105, finger bases at z = 0.075, spread
# +/-0.053 in x. Hard-coded rather than imported so the viewer stays usable
# without $GADDPG_DIR on sys.path.
_FINGER_HALF_X = 0.053
_FINGER_BASE_Z = 0.075
_FINGER_TIP_Z = 0.105

# Mid-grey. Shared by this viewer and test_perception_viz so the two windows
# match — one definition, because they are meant to be the same picture.
#
# Kept on the dark side of mid deliberately: test_perception_viz draws the raw
# scene cloud in near-white, and the lighter this gets the more those dots wash
# out against it. This value keeps white context, orange object and green hand
# all legible at once. Raise it toward 0.6 if you prefer a lighter field and can
# live with fainter context dots.
BACKGROUND = (0.38, 0.38, 0.40)

CLASS_COLOURS = {
    "object": (1.00, 0.45, 0.10),   # orange
    "hand": (0.10, 0.85, 0.35),     # green
}
# Distinct hues for the rigs, in --cameras order.
CAMERA_COLOURS = [
    (0.20, 0.60, 1.00),   # wrist  — blue
    (1.00, 0.85, 0.10),   # tripod — yellow
    (0.90, 0.30, 0.90),   # third  — magenta
]


def _gripper_lineset(o3d):
    """A wireframe Panda gripper at the origin of the panda_hand frame."""
    pts = np.array([
        [0.0, 0.0, 0.0],                                    # 0 flange origin
        [0.0, 0.0, _FINGER_BASE_Z],                         # 1 between the bases
        [+_FINGER_HALF_X, 0.0, _FINGER_BASE_Z],             # 2
        [-_FINGER_HALF_X, 0.0, _FINGER_BASE_Z],             # 3
        [+_FINGER_HALF_X, 0.0, _FINGER_TIP_Z],              # 4
        [-_FINGER_HALF_X, 0.0, _FINGER_TIP_Z],              # 5
    ])
    lines = [[0, 1], [2, 3], [2, 4], [3, 5]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.85, 0.85, 0.90]]), (len(lines), 1)))
    return ls


def _box_lineset(o3d, box, colour=(0.45, 0.13, 0.13)):
    # Deliberately dim. The box is the largest object in the scene (0.32 m deep
    # against a ~0.1 m gripper) and at full saturation it dominates the view it
    # is only context for.
    """Wireframe of the robot-exclusion box, in the panda_hand frame."""
    xs = (-box.half_x, box.half_x)
    ys = (-box.half_y, box.half_y)
    zs = (box.z_min, box.z_max)
    pts = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    # indices follow the x-major, y, z nesting above
    lines = [[0, 1], [2, 3], [4, 5], [6, 7],
             [0, 2], [1, 3], [4, 6], [5, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([colour]), (len(lines), 1)))
    return ls


# PolicyCloudViewer lived here: a legacy-Visualizer window that drew the policy
# cloud alone. It is gone because it was the runner's SECOND implementation of
# the same view, and the two had drifted — it never gained the white raw-scene
# cloud, so --show-cloud could not answer the question the view exists for
# (is the coloured cloud in the right PLACE?). Both the runner and
# test_perception_viz.py now open dual_cloud_window.DualCloudWindow.
#
# What remains in this module is the shared vocabulary that window still uses:
# the colours, the gripper and exclusion-box wireframes, and source_for_cloud.


def source_for_cloud(oi: np.ndarray, hi: np.ndarray,
                     object_source: np.ndarray, hand_source: np.ndarray,
                     num_object: int, num_hand: int) -> Optional[np.ndarray]:
    """Map build_policy_cloud's sampled indices back to per-row rig indices.

    Returns None when provenance is unavailable (e.g. an empty class, whose rows
    are zero-padding and belong to no camera), which the viewer treats as "fall
    back to colouring by class".
    """
    if len(oi) != num_object or len(hi) != num_hand:
        return None
    if len(object_source) == 0 or len(hand_source) == 0:
        return None
    return np.concatenate([object_source[oi], hand_source[hi]])
