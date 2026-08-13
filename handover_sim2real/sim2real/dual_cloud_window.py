"""The shared 3D perception window: policy cloud over the raw scene.

Lives on its own so that `my_policy_runner.py --show-cloud` and
`test_perception_viz.py` open the SAME window rather than two that resemble each
other. A debugging view that renders differently from the thing being debugged
is worse than no view: every difference between them is a difference you have to
hold in your head while deciding whether what you are looking at is a fault.

The window draws TWO clouds at different point sizes. The big coloured one is
the [1024, 5] tensor the policy receives — object orange, hand green. Behind it,
in small white dots, is the raw deprojected scene from every camera, merged in
the same frame. The coloured cloud alone cannot tell you whether it is in the
right PLACE, because it has nothing to be in the right place relative to: a hand
cloud rigidly displaced by a bad extrinsic still looks like a perfectly good hand
cloud. Against the white scene — table, arm, background, geometry you recognise —
a displacement is obvious, and with two cameras the white clouds must OVERLAP.
Where they do not, the gap is the calibration error.

Two point sizes is why this is not the legacy Visualizer in cloud_viewer.py:
that renderer has one global point size for the whole scene. It is gui.Window +
SceneWidget rather than the higher-level O3DVisualizer because only the former
can be given a keyboard handler — see DualCloudWindow._on_key.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SIM2REAL_DIR = Path(__file__).resolve().parent
if str(_SIM2REAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM2REAL_DIR))

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
from transforms import transform_points  # noqa: E402

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

# Open3D reports keys as gui.KeyName enums; the main loop's handler is written
# against the single characters cv2.waitKey returns. Mapping here keeps the two
# windows on one set of actions instead of two parallel handlers.
INTERACTIVE_KEYS = "cwrzxq"


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


def _key_char(key) -> Optional[str]:
    """A gui.KeyEvent key -> the character cv2.waitKey would have given.

    Open3D reports letters as gui.KeyName.A..Z whose values happen to be the
    lowercase ASCII codes, but only the keys the main loop acts on are
    translated: an unrecognised key must be reported as unhandled so the
    SceneWidget still gets its own (arrow keys, modifiers).
    """
    try:
        char = chr(int(key))
    except (TypeError, ValueError):                  # pragma: no cover
        return None
    if char in INTERACTIVE_KEYS:
        return char
    if int(key) == int(gui.KeyName.ESCAPE):
        return "q"
    return None


class DualCloudWindow:
    """The policy cloud over a small-dot raw-scene cloud, in one window.

    Both clouds are allocated ONCE at a fixed size and thereafter updated in
    place. The first version rebuilt them every frame with
    remove_geometry/add_geometry, which re-uploads to the GPU each time and was
    a large part of why the window stuttered. `update_geometry` needs tensor
    geometry (o3d.t.geometry.PointCloud), hence the tensor clouds below.

    `enabled=False` builds nothing and turns every method into a no-op, so a
    caller that only sometimes wants a window (the runner, behind --show-cloud)
    does not need a null object or a guard at each call site.
    """

    def __init__(self, camera_names, exclusion_box=None, width=1100, height=820,
                 context_point_size=1.0, policy_point_size=5.0,
                 context_max=30000, policy_size=1024, enabled=True):
        self._camera_names = list(camera_names)
        self.colour_mode = "class"
        self.show_context = True
        self.alive = False
        self.enabled = bool(enabled)
        self.rotate_mode_label = "disabled"
        self._keys = []
        if not self.enabled:
            return
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

        # (self._keys is initialised above, before the `enabled` early return.)
        # Keys pressed while THIS window has focus, drained by the main loop so
        # the 3D window and the OpenCV windows drive the same actions. See
        # _on_key for why this is a gui.Window rather than an O3DVisualizer.
        self._app = gui.Application.instance
        self._app.initialize()

        # BUILT ON gui.Window RATHER THAN O3DVisualizer, AND THAT IS THE WHOLE
        # REASON. O3DVisualizer derives from WindowBase, not gui.Window: it has
        # no set_on_key and exposes no SceneWidget, so it cannot be given a
        # keyboard handler at all. The roll keys therefore only worked while an
        # OpenCV window held focus — you had to click the camera view to rotate
        # the cloud. gui.Window + SceneWidget is a little more assembly (layout
        # and camera are manual below) and can receive keys.
        self._win = self._app.create_window(
            "perception: policy cloud (large) over raw scene (small white)",
            width, height)
        self._widget = gui.SceneWidget()
        self._widget.scene = rendering.Open3DScene(self._win.renderer)
        self._scene = self._widget.scene
        self._win.add_child(self._widget)
        self._win.set_on_layout(self._on_layout)
        self._win.set_on_key(self._on_key)
        self._win.set_on_close(self._on_close)

        # ORDER AND SKYBOX BOTH MATTER. The scene draws a lit skybox by default,
        # and it renders OVER whatever set_background asks for — so
        # set_background alone silently does nothing and the window keeps its
        # default pale-blue gradient. The skybox has to be turned off for the
        # colour to be visible at all. (The legacy Visualizer in cloud_viewer.py
        # has no skybox, which is why the same one-liner works there.)
        self._scene.show_skybox(False)
        self._scene.set_background([*BACKGROUND, 1.0])
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

        m_mesh = rendering.MaterialRecord()
        m_mesh.shader = "defaultUnlit"

        self._scene.add_geometry("gripper", _gripper_lineset(o3d), self._m_line)
        if exclusion_box is not None:
            self._scene.add_geometry("exclusion", _box_lineset(o3d, exclusion_box),
                                     self._m_line)
        self._scene.add_geometry(
            "origin", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05),
            m_mesh)

        # Allocate at full size up front; every later update writes exactly this
        # many points. Parked at the origin and hidden until real data arrives.
        #
        # add_downsampled_copy_for_fast_rendering is OFF for both clouds. That
        # copy is what the scene draws while the camera is moving, and
        # update_geometry only writes the full-resolution buffer — so with it on,
        # every drag would show a stale cloud and snap to the current one on
        # release. Neither cloud is large enough to need it.
        self._scene.add_geometry("context", self._tpcd(
            np.zeros((self._context_max, 3), np.float32),
            np.tile(CONTEXT_COLOUR, (self._context_max, 1)).astype(np.float32)),
            self._m_ctx, False)
        # Allocated with the OBJECT colour, not zeros. Zeros are black, and any
        # moment where positions have been uploaded but colours have not — a
        # partially applied update — then renders as black points. Seeding with
        # a real colour means the worst case is a stale hue rather than a
        # startling black speckle.
        self._scene.add_geometry("policy", self._tpcd(
            np.zeros((self._policy_size, 3), np.float32),
            np.tile(CLASS_COLOURS["object"], (self._policy_size, 1)).astype(np.float32)),
            self._m_pol, False)
        self._scene.show_geometry("context", False)
        self._scene.show_geometry("policy", False)
        self._ctx_visible = self._pol_visible = False

        # A default view, so the window is not staring into an empty frustum
        # before the first cloud arrives and sets a real one.
        self._widget.setup_camera(
            self._fov,
            o3d.geometry.AxisAlignedBoundingBox([-0.3, -0.3, -0.1], [0.3, 0.3, 0.5]),
            [0.0, 0.0, 0.2])
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

    # -- window plumbing --

    def _on_layout(self, ctx) -> None:
        """SceneWidget fills the window. gui.Window does no layout on its own."""
        self._widget.frame = self._win.content_rect

    def _on_close(self) -> bool:
        self.alive = False
        return True                                  # allow the close

    def _on_key(self, event) -> bool:
        """Queue key presses for the main loop.

        This is the point of building on gui.Window: the 3D window can now act
        on z/x and the rest without the OpenCV window holding focus. Keys are
        QUEUED rather than acted on here so that both windows drive one code
        path — the main loop already handles these, and duplicating the actions
        would be two behaviours to keep in step.

        RETURNS A PLAIN BOOL, and the type is not incidental. The two set_on_key
        overloads in Open3D disagree: gui.Window's takes
        `Callable[[KeyEvent], bool]` where True means "stop dispatching", while
        gui.SceneWidget's takes an EventCallbackResult. Returning the enum from
        the Window handler — which is what this did first — fails to convert on
        the way back into C++ on every real keypress, and the window is torn
        down. It could not be caught by calling this method from Python, because
        that path never crosses the binding; test_on_key_contract asserts the
        type instead.

        False for anything unrecognised, so arrow keys and modifiers still reach
        the SceneWidget's own camera controls.
        """
        if event.type != gui.KeyEvent.Type.DOWN:
            return False
        char = _key_char(event.key)
        if char is None:
            return False
        self._keys.append(char)
        return True

    def drain_keys(self) -> list:
        keys, self._keys = self._keys, []
        return keys

    # -- interaction --

    def _apply_rotate_mode(self) -> None:
        label, attr = ROTATE_MODES[self._rot_idx]
        try:
            self._widget.set_view_controls(getattr(gui.SceneWidget.Controls, attr))
            self.rotate_mode_label = label
        except Exception as e:                       # pragma: no cover
            print(f"[viz] could not set mouse mode {attr}: {e}")
            self.rotate_mode_label = "unknown"

    def cycle_rotate_mode(self) -> None:
        if not self.enabled:
            return
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
        if not self.enabled:
            return
        try:
            m = np.asarray(self._scene.camera.get_model_matrix())
            eye = m[:3, 3].astype(float)
            up = m[:3, 1].astype(float)
            forward = -m[:3, 2].astype(float)         # OpenGL: camera looks down -Z

            d = float(np.linalg.norm(eye - self._centre)) if self._centre is not None else 1.0
            d = max(d, 1e-3)
            axis = forward / (np.linalg.norm(forward) + 1e-12)
            up_new = Rot.from_rotvec(axis * np.deg2rad(degrees)).apply(up)

            self._widget.look_at((eye + forward * d).tolist(),
                                 eye.tolist(), up_new.tolist())
            self._win.post_redraw()
        except Exception as e:                       # pragma: no cover
            print(f"[viz] roll failed: {e}")

    def _show(self, name: str, want: bool, current: bool) -> bool:
        if want != current:
            self._scene.show_geometry(name, want)
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
        self._win.post_redraw()

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
        self._scene.scene.update_geometry(name, cloud, flags)

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
            # Bounds come from THESE points, not from the scene's bounding box.
            # That is the fix for the window that opened empty: the scene's box
            # is whatever was present when geometry was added, which is nothing,
            # and framing against it put a real cloud outside the frustum.
            bounds = o3d.geometry.AxisAlignedBoundingBox(
                (centre - radius).astype(np.float64),
                (centre + radius).astype(np.float64))
            self._widget.setup_camera(self._fov, bounds, centre.astype(np.float32))
            self._widget.look_at(centre.astype(np.float32),
                                 eye.astype(np.float32),
                                 np.array([0.0, 0.0, 1.0], np.float32))
            self._centre = centre
        except Exception as e:                       # pragma: no cover
            print(f"[viz] framing failed, using scene bounds: {e}")
            self._widget.setup_camera(self._fov, self._scene.bounding_box,
                                      self._scene.bounding_box.get_center())

    def tick(self) -> bool:
        """Pump the GUI once. False once the window has been closed.

        Disabled returns True: there is no window, so there is nothing to have
        been closed, and a caller looping on `while window.tick()` must not stop
        merely because it asked for no view.
        """
        if not self.enabled:
            return True
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
                self._win.close()
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


def selftest() -> None:
    """Check the key-handler contract. No cameras, no ROS, no window.

    This exists because of a bug it would have caught. gui.Window.set_on_key is
    `Callable[[KeyEvent], bool]`, while gui.SceneWidget.set_on_key takes an
    EventCallbackResult; the handler returned the enum, which fails to convert
    on the way back into C++ and tore the window down on every real keypress.

    A smoke test that CALLED _on_key from Python could not catch it — that path
    never crosses the binding, so the enum came back happily and the window
    stayed up. What distinguishes the two is the return TYPE, so that is what is
    asserted here. _on_key touches nothing but self._keys, so it runs unbound
    against a stub and needs no display.
    """
    class _Stub:
        def __init__(self):
            self._keys = []

    class _Ev:
        def __init__(self, key, kind=None):
            self.key = key
            self.type = kind if kind is not None else gui.KeyEvent.Type.DOWN

    cases = [
        (gui.KeyName.Z, "z", True),
        (gui.KeyName.X, "x", True),
        (gui.KeyName.C, "c", True),
        (gui.KeyName.W, "w", True),
        (gui.KeyName.R, "r", True),
        (gui.KeyName.Q, "q", True),
        (gui.KeyName.ESCAPE, "q", True),
        (gui.KeyName.LEFT, None, False),     # must reach the SceneWidget
        (gui.KeyName.UP, None, False),
    ]
    for key, want_char, want_handled in cases:
        stub = _Stub()
        got = DualCloudWindow._on_key(stub, _Ev(key))
        assert type(got) is bool, (
            f"{key}: handler returned {type(got).__name__} ({got!r}); "
            "gui.Window.set_on_key is Callable[[KeyEvent], bool] and anything "
            "else fails to convert back into C++, destroying the window")
        assert got == want_handled, f"{key}: handled={got}, wanted {want_handled}"
        queued = (stub._keys or [None])[0]
        assert queued == want_char, f"{key}: queued {queued!r}, wanted {want_char!r}"
        print(f"  {str(key):22s} -> handled={str(got):5s} queued={queued!r}")

    # Key-up must never act, or every press would fire twice.
    stub = _Stub()
    got = DualCloudWindow._on_key(stub, _Ev(gui.KeyName.Z, gui.KeyEvent.Type.UP))
    assert type(got) is bool and got is False and stub._keys == [], got
    print("  key UP                 -> ignored, nothing queued")
    print("key contract ok")


def exit_without_finalizing() -> None:
    """Leave the process without running interpreter finalization.

        Fatal Python error: take_gil: PyMUTEX_LOCK(gil->mutex) failed
        Python runtime state: finalizing

    Open3D's Filament renderer and roslibpy's Twisted reactor both keep threads
    that outlive main() and re-enter Python. When CPython starts finalizing, the
    GIL those threads are waiting on is destroyed under them, and the process
    either hangs or aborts with a core dump — after every piece of work is done
    and `stopped.` has printed. It is a teardown artefact, not a failure of the
    run, and it is not specific to this window: the O3DVisualizer this script
    used previously hangs the same way in the same environment.

    There is nothing left to finalize by this point — the worker thread is
    stopped, cameras are closed, the pose subscription is torn down, and the only
    thing os._exit skips that we care about is flushing our own streams, which is
    done explicitly first.
    """
    import os

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(0)

