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


class PolicyCloudViewer:
    """Non-blocking Open3D window showing the policy's input cloud.

    Deliberately fail-soft. This is a diagnostic on a robot control loop, so a
    viewer problem — no Open3D installed, no display, window closed by hand —
    disables the view and lets the run continue rather than taking the robot
    down with it. `self.alive` reports whether it is still showing anything.
    """

    def __init__(self, enabled: bool = True, camera_names: Sequence[str] = (),
                 exclusion_box=None, update_hz: float = 10.0,
                 point_size: float = 4.0):
        self.enabled = enabled
        self.alive = False
        self.colour_mode = "class"
        self._camera_names = list(camera_names)
        self._exclusion_box = exclusion_box
        self._min_dt = 1.0 / max(update_hz, 1e-3)
        self._last_draw = 0.0
        self._framed = False
        self._o3d = None
        self._vis = None
        self._pcd = None

        if not enabled:
            return
        try:
            import open3d as o3d
        except Exception as e:                       # pragma: no cover
            print(f"[viz] --show-cloud needs open3d, which failed to import "
                  f"({e}). Continuing without the 3D view.")
            self.enabled = False
            return

        self._o3d = o3d
        try:
            self._vis = o3d.visualization.VisualizerWithKeyCallback()
            if not self._vis.create_window("policy point cloud (panda_hand frame)",
                                           width=1000, height=760):
                raise RuntimeError("create_window failed (no display?)")

            self._pcd = o3d.geometry.PointCloud()
            self._vis.add_geometry(self._pcd)
            self._vis.add_geometry(_gripper_lineset(o3d))
            if exclusion_box is not None:
                self._vis.add_geometry(_box_lineset(o3d, exclusion_box))
            self._vis.add_geometry(
                o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

            opt = self._vis.get_render_option()
            opt.background_color = np.asarray(BACKGROUND)
            opt.point_size = point_size

            self._vis.register_key_callback(ord("C"), self._toggle_colour)
            self.alive = True
            print("[viz] 3D cloud window open. 'c' toggles colour by "
                  "class / camera. Closing it does not stop the run.")
        except Exception as e:                       # pragma: no cover
            print(f"[viz] could not open the 3D window ({e}); continuing without it.")
            self.enabled = False
            self.alive = False

    # ── colouring ────────────────────────────────────────────────────────────

    def _toggle_colour(self, _vis) -> bool:
        self.colour_mode = "camera" if self.colour_mode == "class" else "class"
        print(f"[viz] colour by {self.colour_mode}")
        self._last_draw = 0.0        # force a redraw on the next update
        return False

    def _colours(self, pc: np.ndarray, source: Optional[np.ndarray]) -> np.ndarray:
        n_obj = int(pc[:, 3].sum())
        cols = np.empty((len(pc), 3), dtype=np.float64)
        if self.colour_mode == "camera" and source is not None and len(source) == len(pc):
            for i in range(max(len(self._camera_names), 1)):
                cols[source == i] = CAMERA_COLOURS[i % len(CAMERA_COLOURS)]
        else:
            cols[:n_obj] = CLASS_COLOURS["object"]
            cols[n_obj:] = CLASS_COLOURS["hand"]
        return cols

    def _frame_view(self) -> None:
        """Fit the scene, then swing to an oblique angle. Once, on the first
        real cloud.

        ORDER MATTERS, and getting it wrong produced an empty window on the
        robot. Open3D derives its view scale from the bounding box of the
        geometry present when that geometry was ADDED — and the cloud is added
        empty at construction time (hence Open3D's "number of points is 0"
        warning). Setting zoom/lookat against that stale box frames a ~0.3 m
        scene containing only the gripper and the exclusion box, while a real
        wrist-camera cloud sits at z = 0.4-1.0 m in the hand frame: entirely
        outside the frustum, so the gripper drew and the points did not.

        `reset_view_point(True)` refits to what is in the scene NOW, which is
        why it has to come first. Only then is the oblique orientation applied,
        because set_front/set_up rotate about the fitted centre. Deliberately no
        set_lookat: the fitted centre already covers cloud and gripper together,
        and overriding it re-introduces exactly the framing bug above.

        The oblique angle is not cosmetic either. Every object here is
        axis-aligned, so Open3D's default straight-down-an-axis view renders the
        gripper — which lies entirely in the y=0 plane — as a SINGLE LINE, and
        the exclusion box as two nested squares. Both were verified to look
        exactly like that.
        """
        try:
            self._vis.reset_view_point(True)     # refit to the real extent FIRST
            vc = self._vis.get_view_control()
            vc.set_front([0.55, -0.75, 0.36])    # camera sits +x, -y, above
            vc.set_up([0.0, 0.0, 1.0])           # +z (the gripper axis) is up
            vc.set_zoom(0.85)
        except Exception:                        # pragma: no cover
            self._vis.reset_view_point(True)

    # ── per-frame ────────────────────────────────────────────────────────────

    def update(self, pc: np.ndarray, source: Optional[np.ndarray] = None) -> None:
        """Show a [N, 5] policy cloud. `source` is one rig index per row."""
        import time

        if not self.alive:
            return
        now = time.time()
        if now - self._last_draw < self._min_dt:
            return
        self._last_draw = now

        try:
            o3d = self._o3d
            self._pcd.points = o3d.utility.Vector3dVector(
                pc[:, :3].astype(np.float64))
            self._pcd.colors = o3d.utility.Vector3dVector(self._colours(pc, source))
            self._vis.update_geometry(self._pcd)
            if not self._framed and float(np.ptp(pc[:, :3], axis=0).max()) > 1e-3:
                # Frame on the first cloud with real extent, then never again so
                # the user's own rotation and zoom survive. The extent guard
                # matters because a degenerate first cloud (a class padded
                # entirely with zeros) would otherwise fit the view to a single
                # point and leave it stuck there for the whole run.
                self._frame_view()
                self._framed = True
        except Exception as e:                       # pragma: no cover
            print(f"[viz] update failed ({e}); disabling the 3D view.")
            self.alive = False

    def poll(self) -> None:
        """Pump the window's events. Call once per control-loop iteration."""
        if not self.alive:
            return
        try:
            if not self._vis.poll_events():
                # The user closed the window. That is a request to stop looking,
                # not to stop the robot.
                print("[viz] 3D window closed; the run continues.")
                self.close()
                return
            self._vis.update_renderer()
        except Exception:                            # pragma: no cover
            self.alive = False

    def close(self) -> None:
        self.alive = False
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
            self._vis = None


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
