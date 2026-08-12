#!/usr/bin/env python3
"""Watch the perception stack, live, with nothing attached to the robot.

    python test_perception_viz.py                       # both cameras, live pose
    python test_perception_viz.py --cameras wrist       # wrist only
    python test_perception_viz.py --no-ros              # no robot connection

Opens exactly what `my_policy_runner.py --show-cloud` opens — a 2D window per
camera with the segmentation overlay, plus the 3D cloud — and nothing else. No
policy is loaded and no ROS topic is ever published to, so this cannot move the
arm. It exists to answer two questions before you trust a rollout:

  is the hand segmented correctly?   -> the 2D overlays
  did the points land in the right   -> the 3D window
  place once deprojected and fused?

WHY THE WHITE CONTEXT CLOUD. The 3D view draws TWO clouds. The big coloured one
is the [1024, 5] tensor the policy would receive — object orange, hand green.
Behind it, in small white dots, is the raw deprojected scene from every camera,
merged in the same frame. The coloured cloud alone cannot tell you whether it is
in the right *place*, because it has nothing to be in the right place relative
to: a hand cloud that has been rigidly displaced by a bad extrinsic still looks
like a perfectly good hand cloud. Against the white scene — table, arm,
background, all of it geometry you can recognise — a displacement is obvious,
and for two cameras the white clouds must OVERLAP. Where they do not, the gap is
the calibration error.

The two are drawn at different point sizes, which is the reason this script uses
Open3D's O3DVisualizer rather than the legacy Visualizer that cloud_viewer.py
uses: the legacy renderer has ONE global point size for the whole scene, so
"small white dots behind large coloured ones" is not expressible there.

THE FIXED CAMERA NEEDS THE ROBOT POSE. A wrist camera's extrinsics are constant,
so its cloud is correct with no robot connection at all. A tripod camera's are
`inv(T_base_hand) @ T_base_cam`, so without a live pose its points are placed
against a fictitious arm position and the fused picture is meaningless. This
script therefore subscribes to the pose topic read-only, and refuses to draw a
fixed camera under --no-ros rather than showing you a confident, wrong overlap.
"""

from __future__ import annotations

import argparse
import copy
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

SIM2REAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SIM2REAL_DIR.parents[2]
sys.path.insert(0, str(SIM2REAL_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "hands-segmentation-pytorch"))

import open3d as o3d  # noqa: E402
import open3d.core as o3c  # noqa: E402
from open3d.visualization import gui, rendering  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from cloud_viewer import (  # noqa: E402
    BACKGROUND,
    CAMERA_COLOURS,
    CLASS_COLOURS,
    _box_lineset,
    _gripper_lineset,
)
from pointcloud_multicam import (  # noqa: E402
    NUM_HAND_POINTS,
    NUM_OBJECT_POINTS,
    ROBOT_EXCLUSION,
    HandSegmenter,
    MultiCameraPerception,
    build_policy_cloud,
    build_rigs,
    overlay_mask,
)
from cloud_viewer import source_for_cloud  # noqa: E402
from transforms import transform_points  # noqa: E402

# Same defaults as the runner, so what you see here is what it would see.
ROSBRIDGE_HOST = "172.16.0.7"
ROSBRIDGE_PORT = 9090
CURRENT_POSE_TOPIC = "/cartesian_pose"
POSE_MSG_TYPE = "geometry_msgs/PoseStamped"

T_HAND_CAM_NOMINAL = np.array([
    [0.0, -1.0, 0.0, 0.036],
    [1.0,  0.0, 0.0, 0.000],
    [0.0,  0.0, 1.0, 0.036],
    [0.0,  0.0, 0.0, 1.000],
], dtype=np.float64)

CONTEXT_COLOUR = (0.80, 0.80, 0.84)

# Mouse-drag behaviour, cycled with 'r'. These are genuinely different controls,
# not settings of one control, and the difference is what "I can revolve around a
# point but cannot rotate the scene" means:
#
#   ROTATE_CAMERA_SPHERE  arcball. The drag tumbles the view freely in any
#                         direction, including roll. Nothing is pinned, so any
#                         orientation is reachable. Default, because a fused
#                         point cloud has no meaningful "up" — you are judging
#                         whether two clouds coincide in 3D, and that needs
#                         looking down arbitrary axes.
#   ROTATE_CAMERA         turntable. Orbits about the look-at point with the UP
#                         VECTOR PINNED to +z, so the horizon never tilts. Good
#                         for keeping the gripper's axis upright and staying
#                         oriented; it simply cannot roll, which is why it felt
#                         like rotation was missing.
#   ROTATE_MODEL          spins the geometry instead of the camera. Same freedom
#                         as the arcball, opposite mental model.
# Order = cycle order, and the FIRST is the default. Turntable leads because it
# is the one whose drag direction feels conventional; the arcball tumbles freely
# but Open3D drags it in the opposite sense, which reads as inverted controls if
# you are not expecting it. Free rotation is one keypress away rather than
# imposed.
ROTATE_MODES = [
    ("turntable (up pinned, conventional drag)", "ROTATE_CAMERA"),
    ("arcball (free rotation, inverted drag)", "ROTATE_CAMERA_SPHERE"),
    ("rotate model", "ROTATE_MODEL"),
]


# ── robot pose (read-only) ───────────────────────────────────────────────────

class PoseListener:
    """Subscribes to the EE pose. Never advertises, never publishes."""

    def __init__(self, host: str, port: int, topic: str, msg_type: str):
        import roslibpy

        self._roslibpy = roslibpy
        self.client = roslibpy.Ros(host=host, port=port)
        self._topic_name, self._msg_type = topic, msg_type
        self._topic = None
        self._lock = threading.Lock()
        self._msg = None

    def start(self, timeout_s: float = 10.0) -> None:
        self.client.run()
        t0 = time.time()
        while not self.client.is_connected:
            if time.time() - t0 > 5.0:
                raise SystemExit(
                    f"No rosbridge at {self.client.host}:{self.client.port}. "
                    "Start it, or pass --no-ros (wrist camera only).")
            time.sleep(0.05)
        self._topic = self._roslibpy.Topic(self.client, self._topic_name, self._msg_type)
        self._topic.subscribe(self._cb)

        t0 = time.time()
        while self.latest() is None:
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"Connected, but no message on {self._topic_name}.")
            time.sleep(0.05)

    def _cb(self, msg) -> None:
        with self._lock:
            self._msg = msg

    def latest(self) -> Optional[dict]:
        with self._lock:
            return copy.deepcopy(self._msg) if self._msg is not None else None

    def stop(self) -> None:
        for fn in (lambda: self._topic and self._topic.unsubscribe(),
                   self.client.terminate):
            try:
                fn()
            except Exception:
                pass


def pose_msg_to_matrix(msg: dict) -> np.ndarray:
    p, o = msg["pose"]["position"], msg["pose"]["orientation"]
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([o["x"], o["y"], o["z"], o["w"]]).as_matrix()
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    return T


# ── the 3D window ────────────────────────────────────────────────────────────

def _fixed_size(xyz: np.ndarray, n: int) -> np.ndarray:
    """Exactly n points: subsample if more, repeat if fewer.

    Needed because in-place geometry updates cannot GROW an allocation —
    O3DVisualizer rejects an update whose point count exceeds the count the
    geometry was created with ("cannot be updated because the number of points
    exceeds the existing point count") and silently keeps the old cloud. Padding
    by repetition is invisible: the duplicates land exactly on real points.
    """
    if len(xyz) == n:
        return xyz
    if len(xyz) > n:
        return xyz[np.random.choice(len(xyz), n, replace=False)]
    pad = np.random.choice(len(xyz), n - len(xyz), replace=True)
    return np.concatenate([xyz, xyz[pad]], axis=0)


class DualCloudWindow:
    """O3DVisualizer showing the policy cloud over a small-dot context cloud.

    Both clouds are allocated ONCE at a fixed size and thereafter updated in
    place. The first version rebuilt them every frame with
    remove_geometry/add_geometry, which re-uploads to the GPU each time and was
    a large part of why the window stuttered. `update_geometry` needs tensor
    geometry (o3d.t.geometry.PointCloud), hence the tensor clouds below.
    """

    def __init__(self, camera_names, exclusion_box=None, width=1100, height=820,
                 context_point_size=1.0, policy_point_size=5.0,
                 context_max=30000, policy_size=1024):
        self._camera_names = list(camera_names)
        self.colour_mode = "class"
        self.show_context = True
        self.alive = False
        self._framed = False
        self._context_max = int(context_max)
        self._policy_size = int(policy_size)
        self._ctx_visible = True
        self._pol_visible = True
        # Tensor clouds handed to update_geometry are kept alive here until the
        # renderer has certainly consumed them — see _push().
        self._retain: list = []
        self._fov = 60.0
        self._centre = None       # orbit centre, set when the view is first framed

        self._app = gui.Application.instance
        self._app.initialize()
        self._vis = o3d.visualization.O3DVisualizer(
            "perception: policy cloud (large) over raw scene (small white)",
            width, height)
        self._vis.show_settings = False
        # ORDER AND SKYBOX BOTH MATTER. O3DVisualizer draws a lit skybox by
        # default, and it renders OVER whatever set_background asks for — so
        # set_background alone silently does nothing and the window keeps its
        # default pale-blue gradient. The skybox has to be turned off for the
        # colour to be visible at all. (The legacy Visualizer in cloud_viewer.py
        # has no skybox, which is why the same one-liner works there.)
        self._vis.show_skybox(False)
        self._vis.set_background([*BACKGROUND, 1.0], None)
        self._rot_idx = 0
        self._apply_rotate_mode()

        self._m_ctx = rendering.MaterialRecord()
        self._m_ctx.shader = "defaultUnlit"
        self._m_ctx.point_size = float(context_point_size)
        self._m_pol = rendering.MaterialRecord()
        self._m_pol.shader = "defaultUnlit"
        self._m_pol.point_size = float(policy_point_size)
        self._m_line = rendering.MaterialRecord()
        self._m_line.shader = "unlitLine"
        self._m_line.line_width = 2.0

        self._vis.add_geometry("gripper", _gripper_lineset(o3d), self._m_line)
        if exclusion_box is not None:
            self._vis.add_geometry("exclusion", _box_lineset(o3d, exclusion_box),
                                   self._m_line)
        self._vis.add_geometry(
            "origin", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

        # Allocate at full size up front; every later update writes exactly this
        # many points. Parked at the origin and hidden until real data arrives.
        self._vis.add_geometry("context", self._tpcd(
            np.zeros((self._context_max, 3), np.float32),
            np.tile(CONTEXT_COLOUR, (self._context_max, 1)).astype(np.float32)),
            self._m_ctx)
        # Allocated with the OBJECT colour, not zeros. Zeros are black, and any
        # moment where positions have been uploaded but colours have not — a
        # partially applied update — then renders as black points. Seeding with
        # a real colour means the worst case is a stale hue rather than a
        # startling black speckle.
        self._vis.add_geometry("policy", self._tpcd(
            np.zeros((self._policy_size, 3), np.float32),
            np.tile(CLASS_COLOURS["object"], (self._policy_size, 1)).astype(np.float32)),
            self._m_pol)
        self._vis.show_geometry("context", False)
        self._vis.show_geometry("policy", False)
        self._ctx_visible = self._pol_visible = False

        self._app.add_window(self._vis)
        self.alive = True

    # -- geometry helpers --

    @staticmethod
    def _tpcd(xyz: np.ndarray, colours: np.ndarray):
        p = o3d.t.geometry.PointCloud()
        p.point.positions = o3c.Tensor(np.ascontiguousarray(xyz, dtype=np.float32))
        p.point.colors = o3c.Tensor(np.ascontiguousarray(colours, dtype=np.float32))
        return p

    def _policy_colours(self, pc: np.ndarray, source) -> np.ndarray:
        n_obj = int(pc[:, 3].sum())
        cols = np.empty((len(pc), 3), dtype=np.float32)
        if self.colour_mode == "camera" and source is not None and len(source) == len(pc):
            for i in range(max(len(self._camera_names), 1)):
                cols[source == i] = CAMERA_COLOURS[i % len(CAMERA_COLOURS)]
        else:
            cols[:n_obj] = CLASS_COLOURS["object"]
            cols[n_obj:] = CLASS_COLOURS["hand"]
        return cols

    # -- interaction --

    def _apply_rotate_mode(self) -> None:
        label, attr = ROTATE_MODES[self._rot_idx]
        try:
            self._vis.mouse_mode = getattr(gui.SceneWidget.Controls, attr)
            self.rotate_mode_label = label
        except Exception as e:                       # pragma: no cover
            print(f"[viz] could not set mouse mode {attr}: {e}")
            self.rotate_mode_label = "unknown"

    def cycle_rotate_mode(self) -> None:
        self._rot_idx = (self._rot_idx + 1) % len(ROTATE_MODES)
        self._apply_rotate_mode()
        print(f"[viz] drag mode: {self.rotate_mode_label}")

    def roll(self, degrees: float) -> None:
        """Spin the view about the axis you are looking along.

        This is the degree of freedom the turntable cannot reach: it pins the
        up-vector, so a drag can orbit but never tilt the horizon. Rather than
        forcing the arcball — which reaches every orientation but whose drag
        direction reads as inverted — roll is exposed as its own key, where the
        direction is unambiguous.

        The camera POSITION and VIEW DIRECTION are left exactly as they are and
        only `up` is rotated, so the view does not jump. The orbit centre is
        re-placed along the current view axis at its existing distance, which
        keeps subsequent turntable drags behaving the same way.
        """
        try:
            m = np.asarray(self._vis.scene.camera.get_model_matrix())
            eye = m[:3, 3].astype(float)
            up = m[:3, 1].astype(float)
            forward = -m[:3, 2].astype(float)         # OpenGL: camera looks down -Z

            d = float(np.linalg.norm(eye - self._centre)) if self._centre is not None else 1.0
            d = max(d, 1e-3)
            axis = forward / (np.linalg.norm(forward) + 1e-12)
            up_new = Rot.from_rotvec(axis * np.deg2rad(degrees)).apply(up)

            self._vis.setup_camera(self._fov, (eye + forward * d).tolist(),
                                   eye.tolist(), up_new.tolist())
            self._vis.post_redraw()
        except Exception as e:                       # pragma: no cover
            print(f"[viz] roll failed: {e}")

    def _show(self, name: str, want: bool, current: bool) -> bool:
        if want != current:
            self._vis.show_geometry(name, want)
        return want

    # -- per frame --

    def update(self, policy_pc: Optional[np.ndarray], source,
               context_xyz: Optional[np.ndarray]) -> None:
        """Push a frame. A frame with nothing in it HOLDS the previous cloud.

        Holding rather than hiding is deliberate and is what stopped the cloud
        blinking out several times a second. `fused.usable` goes false whenever
        the segmenter loses the hand for a single frame — which is often — and
        hiding the geometry on those frames made the whole cloud disappear and
        come back constantly. The staleness is still reported honestly, in the
        2D captions (`STALE`) and the console, so nothing is being concealed;
        only the flicker is gone.
        """
        if not self.alive:
            return
        flags = rendering.Scene.UPDATE_POINTS_FLAG | rendering.Scene.UPDATE_COLORS_FLAG

        if self.show_context and context_xyz is not None and len(context_xyz):
            xyz = _fixed_size(np.asarray(context_xyz, np.float32), self._context_max)
            self._push("context", xyz,
                       np.tile(CONTEXT_COLOUR, (self._context_max, 1)).astype(np.float32),
                       flags)
            self._ctx_visible = self._show("context", True, self._ctx_visible)
        elif not self.show_context:
            self._ctx_visible = self._show("context", False, self._ctx_visible)

        if policy_pc is not None and len(policy_pc) == self._policy_size:
            self._push("policy",
                       np.ascontiguousarray(policy_pc[:, :3], np.float32),
                       self._policy_colours(policy_pc, source), flags)
            self._pol_visible = self._show("policy", True, self._pol_visible)
            if not self._framed:
                self._frame(policy_pc[:, :3])
                self._framed = True

        # Ask for a repaint explicitly. Geometry changed outside an input event,
        # and the window is being driven by run_one_tick() from our own loop
        # rather than by Application.run(), so nothing else guarantees a redraw.
        self._vis.post_redraw()

    def _push(self, name: str, xyz: np.ndarray, colours: np.ndarray,
              flags: int) -> None:
        """Update a geometry in place, keeping the uploaded buffer ALIVE.

        The tensor cloud handed to update_geometry must outlive the call. Open3D
        hands the buffer to Filament, which consumes it on the render thread,
        so a cloud built inline and dropped as soon as update_geometry returns
        can be freed while the renderer is still reading it — which shows up as
        points flashing black or picking up garbage colours. Holding the last
        few frames' clouds costs a couple of megabytes and removes the race.
        """
        cloud = self._tpcd(xyz, colours)
        self._retain.append(cloud)
        if len(self._retain) > 4:
            self._retain.pop(0)
        self._vis.update_geometry(name, cloud, flags)

    def _frame(self, xyz: np.ndarray) -> None:
        """Aim at the POLICY cloud, not the whole scene.

        The context cloud can sprawl metres across a lab; framing to it would
        shrink the gripper and the object — the things being judged — to a few
        pixels. Bounds come from the policy cloud plus the gripper, and the eye
        sits obliquely so the axis-aligned geometry does not degenerate (the
        gripper is planar and collapses to a line viewed down y).
        """
        try:
            pts = np.vstack([xyz, np.array([[0, 0, 0], [0, 0, 0.105]])])
            centre = pts.mean(axis=0)
            radius = max(float(np.linalg.norm(pts - centre, axis=1).max()), 0.15)
            direction = np.array([0.55, -0.75, 0.36])
            direction /= np.linalg.norm(direction)
            eye = centre + direction * radius * 3.0
            self._vis.setup_camera(self._fov, centre, eye, [0.0, 0.0, 1.0])
            self._centre = centre
        except Exception:
            self._vis.reset_camera_to_default()

    def tick(self) -> bool:
        """Pump the GUI once. False once the window has been closed."""
        if not self.alive:
            return False
        try:
            if not self._app.run_one_tick():
                self.alive = False
                return False
        except Exception:                            # pragma: no cover
            self.alive = False
            return False
        return True

    def close(self) -> None:
        if self.alive:
            self.alive = False
            try:
                self._vis.close()
                self._app.run_one_tick()
            except Exception:
                pass


# ── raw scene cloud ──────────────────────────────────────────────────────────

def context_cloud(rig, depth_m: np.ndarray, T_base_hand: np.ndarray,
                  stride: int, radius_m: float) -> np.ndarray:
    """Everything the camera sees, in panda_hand, clipped to a usable radius.

    Unsegmented and unlabelled on purpose — this is the recognisable geometry
    (table, arm, walls) that makes a misplaced coloured cloud visible. Clipped
    because a tripod at 1.5 m otherwise contributes the whole room, which
    dominates the view scale for no diagnostic gain.
    """
    xyz, _, _ = rig.camera.depth_to_pointcloud(
        depth_m=depth_m, color_bgr=None, mask=None, stride=stride,
        min_depth=rig.params.min_depth_m, max_depth=rig.params.max_depth_m)
    if not len(xyz):
        return np.zeros((0, 3), dtype=np.float32)
    pts = transform_points(rig.hand_from_camera(T_base_hand).astype(np.float32), xyz)
    return pts[np.linalg.norm(pts, axis=1) < radius_m]


# ── perception worker ────────────────────────────────────────────────────────

class PerceptionWorker(threading.Thread):
    """Runs the whole perception pass off the GUI thread, publishing snapshots.

    Only ever produces numpy arrays. Open3D geometry is built on the main thread
    from those arrays, because the renderer and its geometry are not safe to
    touch from two threads. Likewise every OpenCV window stays on the main
    thread; this returns the images and lets the caller draw them.

    Publishes a complete, self-consistent snapshot under one lock, so the drawer
    can never mix a point cloud from one frame with the captions from another.
    """

    def __init__(self, perception, rigs, pose, context_stride, context_radius,
                 window, max_hz: float = 15.0):
        super().__init__(daemon=True)
        self._perception = perception
        self._rigs = rigs
        self._pose = pose
        self._stride = context_stride
        self._radius = context_radius
        self._window = window
        # Throttled on purpose. Unthrottled this thread pins a core or two
        # (segmentation, deprojection, the fusion), and on a laptop that starves
        # the render thread it is meant to be feeding — the window stutters
        # again, for a completely different reason than before. Nothing consumes
        # snapshots faster than the draw rate anyway.
        self._min_dt = 1.0 / max(max_hz, 1e-3)
        self._lock = threading.Lock()
        self._snap = None
        # NOT self._stop: threading.Thread already defines an internal _stop()
        # that join() calls during teardown, and shadowing it with an Event
        # makes every join() raise "'Event' object is not callable".
        self._stop_evt = threading.Event()
        self.ok = True
        self.error = None

    def latest(self):
        with self._lock:
            return self._snap

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5.0)

    def run(self):
        try:
            while not self._stop_evt.is_set():
                t0 = time.time()
                T_base_hand = (np.eye(4) if self._pose is None
                               else pose_msg_to_matrix(self._pose.latest()))

                fused = self._perception.observe(T_base_hand)

                policy_pc = source = None
                if fused.usable:
                    policy_pc, oi, hi = build_policy_cloud(
                        fused.object_xyz, fused.hand_xyz, return_index=True)
                    source = source_for_cloud(oi, hi, fused.object_source,
                                              fused.hand_source,
                                              NUM_OBJECT_POINTS, NUM_HAND_POINTS)

                ctx = None
                if self._window.show_context:
                    parts = [context_cloud(r, self._perception.last_depths[r.name],
                                           T_base_hand, self._stride, self._radius)
                             for r in self._rigs]
                    parts = [q for q in parts if len(q)]
                    ctx = np.concatenate(parts) if parts else None

                snap = {
                    "policy_pc": policy_pc,
                    "source": source,
                    "context": ctx,
                    "usable": fused.usable,
                    "summary": fused.summary(),
                    "per_camera": fused.per_camera,
                    "frames": dict(self._perception.last_frames),
                }
                with self._lock:
                    self._snap = snap

                # Yield the rest of the budget to the GUI. Interruptible, so
                # stop() still returns promptly.
                self._stop_evt.wait(max(0.0, self._min_dt - (time.time() - t0)))
        except Exception as e:                       # pragma: no cover
            self.error = e
        finally:
            self.ok = False


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cameras", default="wrist,tripod")
    p.add_argument("--calib-session", default=None,
                   help="hand-eye session providing T_base_color.npy for fixed cameras")
    p.add_argument("--hand-eye", default=None,
                   help="4x4 .npy T_hand_cam for the WRIST camera (default: the "
                        "sim's nominal mount, which is not a calibration)")
    p.add_argument("--hand-seg-ckpt",
                   default=str(SIM2REAL_DIR / "checkpoint" / "cp1" / "checkpoint.ckpt"))
    p.add_argument("--no-ros", action="store_true",
                   help="do not connect to rosbridge. Wrist camera only: a fixed "
                        "camera's placement depends on the live robot pose.")
    p.add_argument("--rosbridge-host", default=ROSBRIDGE_HOST)
    p.add_argument("--rosbridge-port", type=int, default=ROSBRIDGE_PORT)
    p.add_argument("--context-stride", type=int, default=6,
                   help="subsampling of the white scene cloud (higher = sparser)")
    p.add_argument("--context-radius", type=float, default=1.2,
                   help="clip the white cloud to this radius around panda_hand")
    p.add_argument("--draw-hz", type=float, default=10.0,
                   help="how often the 3D geometry is re-uploaded. The event "
                        "loop always runs flat out; only this is throttled, "
                        "because uploading is the expensive part.")
    p.add_argument("--context-max", type=int, default=30000,
                   help="hard cap on white-cloud points. The geometry is "
                        "allocated once at this size and cannot grow later.")
    p.add_argument("--no-context", action="store_true",
                   help="start with the white scene cloud hidden ('w' toggles)")
    p.add_argument("--no-robot-exclusion", action="store_true")
    p.add_argument("--seconds", type=float, default=None,
                   help="auto-exit after this long (smoke test)")
    # No --screenshot: O3DVisualizer's export_current_image and the lower-level
    # Scene.render_to_image are both queued on Filament's render thread, and
    # neither completes when the app is driven by run_one_tick() from our own
    # loop — verified, the callback simply never fires. A flag that reports
    # success and writes nothing is worse than no flag. Use an external grab
    # (xwd, or your desktop's screenshot key) on the 3D window.
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    args = p.parse_args()

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    fixed = [c for c in camera_names if c != "wrist"]
    if fixed and args.calib_session is None:
        raise SystemExit(
            f"--cameras includes fixed camera(s) {', '.join(fixed)} but no "
            "--calib-session; their pose in the base frame is not guessable.")
    if fixed and args.no_ros:
        raise SystemExit(
            f"--no-ros cannot place a fixed camera. {', '.join(fixed)} needs the "
            "live EE pose (its extrinsics are inv(T_base_hand) @ T_base_cam), so "
            "without it the fused cloud would be confidently wrong. Either drop "
            "--no-ros, or use --cameras wrist.")

    import torch
    from torchvision import transforms

    from model import HandSegModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    T_hand_cam = (np.load(args.hand_eye).astype(np.float64) if args.hand_eye
                  else T_HAND_CAM_NOMINAL)
    if args.hand_eye is None:
        print("[calib] wrist T_hand_cam is the SIM's nominal mount, not a "
              "calibration of your D435.")

    seg_model = HandSegModel.load_from_checkpoint(
        args.hand_seg_ckpt, map_location="cpu").to(device).eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    rigs = build_rigs(
        camera_names, T_hand_cam_wrist=T_hand_cam, fixed_session=args.calib_session,
        color_size=(args.camera_width, args.camera_height),
        depth_size=(args.camera_width, args.camera_height), fps=args.camera_fps,
        exclude_robot=not args.no_robot_exclusion)

    pose = None
    window = None
    try:
        if not args.no_ros:
            pose = PoseListener(args.rosbridge_host, args.rosbridge_port,
                                CURRENT_POSE_TOPIC, POSE_MSG_TYPE)
            pose.start()
            print(f"pose: subscribed to {CURRENT_POSE_TOPIC} (read-only)")
        else:
            print("pose: --no-ros, using identity (valid for the wrist camera only)")

        for rig in rigs:
            rig.camera.start()
            print(f"[camera] {rig.name:8s} serial={rig.serial}  {rig.kind}")

        perception = MultiCameraPerception(
            rigs, HandSegmenter(seg_model, preprocess, device))

        window = DualCloudWindow(
            [r.name for r in rigs],
            exclusion_box=(ROBOT_EXCLUSION
                           if (not args.no_robot_exclusion
                               and any(r.exclude_robot for r in rigs)) else None),
            context_max=args.context_max)
        window.show_context = not args.no_context

        print("\nkeys (in any OpenCV window):"
              "  c = colour by class/camera   w = white scene cloud on/off"
              "   z / x = roll the view   r = drag mode   q = quit")
        print(f"drag mode: {window.rotate_mode_label}")
        print("  left-drag orbits, scroll zooms; 'z'/'x' tilt the horizon, which "
              "is the one thing the turntable cannot do.")
        print(f"  'r' cycles: {' -> '.join(m[0] for m in ROTATE_MODES)}\n")

        # PERCEPTION RUNS OFF THE GUI THREAD. This is the whole reason the window
        # is usable. One pass — two camera grabs, a segmentation forward, the
        # deprojection and the fusion — takes ~70 ms, and when it sat in the same
        # loop as gui.run_one_tick() the GUI got exactly ONE event tick per pass.
        # At ~14 Hz a trackpad drag is sampled so coarsely that most of the
        # gesture is dropped, which is why rotation barely responded and the
        # window felt stuck. With the work on a worker thread the main loop does
        # nothing but pump events, so the camera stays smooth no matter how slow
        # perception is.
        worker = PerceptionWorker(perception, rigs, pose, args.context_stride,
                                  args.context_radius, window)
        worker.start()

        t_start = time.time()
        draws = 0
        last_draw = 0.0
        last_2d = 0.0
        min_draw_dt = 1.0 / max(args.draw_hz, 1e-3)
        try:
            while worker.ok:
                if args.seconds is not None and time.time() - t_start > args.seconds:
                    break

                now = time.time()
                snap = worker.latest()

                # Geometry upload is rate-limited independently of both the
                # camera and the event loop: it is the expensive part, and no
                # one can read a point cloud faster than this anyway.
                if snap is not None and now - last_draw >= min_draw_dt:
                    window.update(snap["policy_pc"], snap["source"], snap["context"])
                    last_draw = now
                    draws += 1
                    if draws % 30 == 0:
                        print(f"[{draws}] {snap['summary']}"
                              + ("" if snap["usable"] else "   (holding: a class is empty)"))

                if snap is not None and now - last_2d >= 1.0 / 15.0:
                    for rig in rigs:
                        pair = snap["frames"].get(rig.name)
                        if pair is None:
                            continue
                        color_bgr, mask = pair
                        view = overlay_mask(color_bgr, mask)
                        d = snap["per_camera"][rig.name]
                        txt = [
                            f"{rig.name}  ({rig.kind})  serial {rig.serial}",
                            f"obj={d['object']}  hand={d['hand']}"
                            + (f"  -{d['robot_pts_removed']} robot"
                               if d["robot_pts_removed"] else "")
                            + ("  STALE" if d["used_last_hand"] or d["used_last_object"]
                               else ""),
                            f"mask px={int(mask.sum())}",
                        ]
                        if not snap["usable"]:
                            txt.append("NOT USABLE - a class is empty")
                        for j, t in enumerate(txt):
                            cv2.putText(view, t, (10, 24 + 22 * j),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                        (0, 255, 255) if j else (0, 255, 0), 1,
                                        cv2.LINE_AA)
                        cv2.imshow(f"cam: {rig.name}", view)
                    last_2d = now

                # cv2.waitKey(1) both pumps the OpenCV windows and yields ~1 ms,
                # so this spins fast without burning a core outright.
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("c"):
                    window.colour_mode = ("camera" if window.colour_mode == "class"
                                          else "class")
                    print(f"colour by {window.colour_mode}")
                    last_draw = 0.0
                if key == ord("w"):
                    window.show_context = not window.show_context
                    print(f"white scene cloud {'on' if window.show_context else 'off'}")
                    last_draw = 0.0
                if key == ord("r"):
                    window.cycle_rotate_mode()
                if key == ord("z"):
                    window.roll(-10.0)
                if key == ord("x"):
                    window.roll(+10.0)

                if not window.tick():      # window closed
                    break
        finally:
            worker.stop()
            if worker.error is not None:
                print(f"perception thread stopped: {worker.error}")
    finally:
        if window is not None:
            window.close()
        cv2.destroyAllWindows()
        for rig in rigs:
            try:
                rig.camera.stop()
            except Exception:
                pass
        if pose is not None:
            pose.stop()
        print("stopped.")


if __name__ == "__main__":
    main()
