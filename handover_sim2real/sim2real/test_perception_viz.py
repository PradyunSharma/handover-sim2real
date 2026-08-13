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

The two are drawn at different point sizes, which is why this script does not use
the legacy Visualizer that cloud_viewer.py does: that renderer has ONE global
point size for the whole scene, so "small white dots behind large coloured ones"
is not expressible there. It uses gui.Window + SceneWidget rather than the
higher-level O3DVisualizer because only the former can be given a keyboard
handler — see DualCloudWindow.

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

from scipy.spatial.transform import Rotation as Rot  # noqa: E402

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
from dual_cloud_window import (  # noqa: E402
    ROTATE_MODES,
    DualCloudWindow,
    context_cloud,
    exit_without_finalizing,
    selftest,
)

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
                    "arm_clusters": fused.arm_clusters,
                    "arm_dropped": fused.arm_dropped,
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
    # Mirrors my_policy_runner, so what this window shows is what the policy is
    # fed. A debugging view running different perception settings from the thing
    # being debugged is worse than no view at all.
    p.add_argument("--no-cluster", action="store_true",
                   help="object class from the crop sphere alone (pre-clustering "
                        "behaviour, tight radii restored)")
    p.add_argument("--wrist-seg-px", type=int, default=None)
    p.add_argument("--fixed-seg-px", type=int, default=None)
    p.add_argument("--hand-margin-px", type=int, default=None,
                   help="pixels around the hand mask belonging to neither class "
                        "(default 5 wrist / 4 fixed). 0 disables, which brings "
                        "back the object<->forearm bridge and its flicker.")
    p.add_argument("--no-arm-rejection", action="store_true",
                   help="keep object-class blobs behind the hand (the forearm)")
    p.add_argument("--arm-offset", type=float, default=0.07,
                   help="metres behind the hand, along hand->robot-base, before "
                        "a blob is called forearm. This window prints every "
                        "blob's offset, which is how you pick this number.")
    p.add_argument("--arm-lateral", type=float, default=0.12,
                   help="metres sideways of that axis before a point is called "
                        "forearm. This is what catches an INCLINED arm, which "
                        "the axis term alone scores near zero.")
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
    p.add_argument("--selftest", action="store_true",
                   help="check the key-handler contract and exit. No cameras, "
                        "no ROS, no window.")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

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

    rigs = build_rigs(
        camera_names, T_hand_cam_wrist=T_hand_cam, fixed_session=args.calib_session,
        color_size=(args.camera_width, args.camera_height),
        depth_size=(args.camera_width, args.camera_height), fps=args.camera_fps,
        exclude_robot=not args.no_robot_exclusion,
        cluster=not args.no_cluster,
        wrist_seg_px=args.wrist_seg_px, fixed_seg_px=args.fixed_seg_px,
        hand_margin_px=args.hand_margin_px)

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
            rigs, HandSegmenter(seg_model, device),
            arm_rejection=not args.no_arm_rejection,
            arm_offset_m=args.arm_offset,
            arm_lateral_m=args.arm_lateral)

        window = DualCloudWindow(
            [r.name for r in rigs],
            exclusion_box=(ROBOT_EXCLUSION
                           if (not args.no_robot_exclusion
                               and any(r.exclude_robot for r in rigs)) else None),
            context_max=args.context_max)
        window.show_context = not args.no_context

        print("\nkeys (in ANY window, including the 3D one):"
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
        last_blobs = None
        blob_changes = 0
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
                    # Blob count is tracked every frame, not every 30th, because
                    # the arm problem showed up as FLICKER — the shell of
                    # hand-surface points the mask misses fuses the object blob
                    # to the forearm blob, and whether it is continuous varies
                    # frame to frame. A count that changes while the scene is
                    # still is the signature; a sampled print would miss it.
                    n_blobs = len(snap["arm_clusters"])
                    if n_blobs != last_blobs and last_blobs is not None:
                        blob_changes += 1
                    last_blobs = n_blobs

                    if draws % 30 == 0:
                        # Each entry is (points, metres toward the robot from the
                        # hand, fraction of the blob behind the threshold). The
                        # forearm sits clearly negative. If object and arm do not
                        # separate here, no threshold will separate them.
                        blobs = "  ".join(f"{n}@{o:+.3f}/lat{r:.3f}/{f:.2f}"
                                          for n, o, r, f in snap["arm_clusters"][:4])
                        print(f"[{draws}] {snap['summary']}"
                              + (f"   blobs: {blobs}" if blobs else "")
                              + (f"   blob-count changed {blob_changes}x"
                                 if blob_changes else "")
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
                            # The declustered count is the whole point of
                            # looking: it is the table and background mass that
                            # would otherwise be object points. A zero here
                            # every frame means clustering is not biting.
                            f"declustered={d['cluster_dropped']}"
                            + (f"  [{d['cluster_fallback']}]"
                               if d["cluster_fallback"] else "")
                            + f"   seg={rig.params.seg_input_px}px",
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
                # so this spins fast without burning a core outright. The 3D
                # window's keys are drained into the SAME list, so every action
                # works from whichever window happens to have focus rather than
                # only from the camera views.
                key = cv2.waitKey(1) & 0xFF
                pressed = [chr(key)] if 32 <= key < 127 else []
                if key == 27:
                    pressed.append("q")
                pressed.extend(window.drain_keys())

                quit_requested = False
                for k in pressed:
                    if k == "q":
                        quit_requested = True
                    elif k == "c":
                        window.colour_mode = ("camera" if window.colour_mode == "class"
                                              else "class")
                        print(f"colour by {window.colour_mode}")
                        last_draw = 0.0
                    elif k == "w":
                        window.show_context = not window.show_context
                        print(f"white scene cloud "
                              f"{'on' if window.show_context else 'off'}")
                        last_draw = 0.0
                    elif k == "r":
                        window.cycle_rotate_mode()
                    elif k == "z":
                        window.roll(-10.0)
                    elif k == "x":
                        window.roll(+10.0)
                if quit_requested:
                    break

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
    exit_without_finalizing()
