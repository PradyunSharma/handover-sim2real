#!/usr/bin/env python3
"""Capture image + robot pose pairs into a session.

    python capture_image_and_pose.py --session 2026-08-11 --role tripod

Move the arm, let it come to rest, press 's'. Each save writes
<session>/images/NNNN.png and appends the matching pose to
<session>/robot_poses.json. 'q' or Esc quits.

Aim for 15-20 pairs with LARGE ROTATION changes between them — AX=XB recovers
the camera's orientation only from relative rotations, so translating the arm
around without turning the wrist yields an ill-conditioned solve however many
captures it contains. calibrate.py reports the diversity it got.

Live board detection is overlaid, so a capture that the solver would later throw
away is visible at the time you take it rather than afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import roslibpy

import calib_common as cc
import calib_config as cfg

# The RealSense wrapper lives one level up and is shared with the runner; this
# script used to carry a third copy of it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import RealSenseCamera  # noqa: E402


def pose_stamped_to_T(msg: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    p, o = msg["pose"]["position"], msg["pose"]["orientation"]
    R = Rot.from_quat([o["x"], o["y"], o["z"], o["w"]]).as_matrix()
    return cc.make_T(R, np.array([p["x"], p["y"], p["z"]]))


def franka_state_to_T(msg: dict) -> np.ndarray:
    flat = msg.get("O_T_EE")
    if flat is None or len(flat) != 16:
        raise ValueError("message has no valid O_T_EE")
    return np.array(flat, dtype=np.float64).reshape(4, 4, order="F")   # column-major


class PoseSubscriber:
    def __init__(self, host, port, topic, msg_type, source):
        self.client = roslibpy.Ros(host=host, port=port)
        self.topic_name, self.msg_type, self.source = topic, msg_type, source
        self.topic: Optional[roslibpy.Topic] = None
        self._lock = threading.Lock()
        self._T: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def start(self):
        self.client.run()
        t0 = time.time()
        while not self.client.is_connected:
            if time.time() - t0 > 5.0:
                raise SystemExit(f"No rosbridge at {self.client.host}:{self.client.port}")
            time.sleep(0.05)
        self.topic = roslibpy.Topic(self.client, self.topic_name, self.msg_type)
        self.topic.subscribe(self._cb)

    def _cb(self, msg):
        try:
            T = franka_state_to_T(msg) if self.source == "franka_state" else pose_stamped_to_T(msg)
        except Exception:
            return
        with self._lock:
            self._T, self._t = T, time.time()

    def latest(self):
        with self._lock:
            return (None if self._T is None else self._T.copy()), self._t

    def stop(self):
        for fn in (lambda: self.topic and self.topic.unsubscribe(), self.client.terminate):
            try:
                fn()
            except Exception:
                pass


def next_index(session) -> int:
    """One past the highest index in EITHER the images or the pose file.

    Deriving this from images/ alone is what allowed a cleared image folder to
    restart numbering at 0001 while the append-only JSON kept the old entries —
    silently pairing old poses with new pictures. Taking the max of both means a
    half-cleared session can never overwrite.
    """
    idx = 0
    if session.images.is_dir():
        for p in session.images.glob("*.png"):
            try:
                idx = max(idx, int(p.stem))
            except ValueError:
                continue
    if session.poses_json.exists():
        for entry in json.loads(session.poses_json.read_text()):
            try:
                idx = max(idx, int(Path(entry["image"]).stem))
            except (ValueError, KeyError):
                continue
    return idx + 1


def append_pose(path: Path, entry: dict) -> None:
    entries = json.loads(path.read_text()) if path.exists() else []
    entries.append(entry)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_session_arg(p)
    cc.add_camera_args(p)
    p.add_argument("--rosbridge-host", default=cfg.ROSBRIDGE_HOST)
    p.add_argument("--rosbridge-port", type=int, default=cfg.ROSBRIDGE_PORT)
    p.add_argument("--pose-topic", default=cfg.POSE_TOPIC)
    p.add_argument("--pose-type", default=cfg.POSE_TYPE)
    p.add_argument("--pose-source", default=cfg.POSE_SOURCE,
                   choices=["franka_state", "pose_stamped"])
    p.add_argument("--reset", action="store_true",
                   help="delete this session's images AND robot_poses.json together, "
                        "then start clean. Clearing only one by hand desyncs them and "
                        "silently mispairs poses with pictures.")
    args = p.parse_args()

    session = cc.resolve_session(args.session, create=True)

    if args.reset:
        n_img = len(list(session.images.glob("*.png")))
        # Both halves together, always. Clearing only one is precisely the
        # desync that pairs old poses with new pictures.
        for img_path in session.images.glob("*.png"):
            img_path.unlink()
        session.poses_json.unlink(missing_ok=True)
        print(f"[session] reset {session.name}: removed {n_img} image(s) and robot_poses.json")

    problems = cc.check_session_pairing(session)
    if problems:
        raise SystemExit(
            f"Session {session.name!r} has a broken image/pose pairing:\n  "
            + "\n  ".join(problems)
            + "\n\nCapturing on top of this would add good data to bad. Either:\n"
              f"  python capture_image_and_pose.py --session {session.name} --reset\n"
              "  (discards this session's captures and starts clean), or\n"
              "  use a fresh --session name."
        )

    serial = cc.resolve_serial(args.serial, args.role)

    existing = next_index(session) - 1
    if existing:
        print(f"[session] {session.name} already holds {existing} captures — "
              "appending. Use a NEW --session if the camera has moved since.")

    # Interior corners of the ChArUco grid — the maximum the detector can return.
    max_corners = (cfg.BOARD.squares_x - 1) * (cfg.BOARD.squares_y - 1)

    K = D = board = dictionary = None
    if session.intrinsics_json.exists():
        K, D, _ = cc.load_intrinsics(session.intrinsics_json)
        board, dictionary = cc.build_board()
    else:
        print("[warn] no color_intrinsics.json in this session yet — capturing "
              "without live board feedback. Run generate_color_intrinsics.py first "
              "to see whether each shot is usable.")

    cam = RealSenseCamera(color_size=(cfg.STREAM.width, cfg.STREAM.height),
                          depth_size=(cfg.STREAM.width, cfg.STREAM.height),
                          fps=cfg.STREAM.fps, serial=serial)
    sub = PoseSubscriber(args.rosbridge_host, args.rosbridge_port,
                         args.pose_topic, args.pose_type, args.pose_source)
    cam.start()
    sub.start()

    idx = next_index(session)
    print(f"session dir : {session.root}")
    print(f"pose topic  : {args.pose_topic} [{args.pose_source}]")
    print("\n's' save   'q'/Esc quit\n")

    try:
        while True:
            color, _, ts_ms = cam.get_frames()
            T, t_recv = sub.latest()
            pose_ok = T is not None and t_recv is not None and \
                (time.time() - t_recv) <= cfg.MAX_POSE_AGE_S

            corners, tilt = -1, None
            if board is not None:
                res = cc.detect_board(color, K, D, board, dictionary)
                corners = res["num_charuco"] if res else 0
                if res:
                    tilt = cc.board_tilt_deg(res["T_cam_board"])

            view = color.copy()
            board_ok = corners < 0 or corners >= cfg.MIN_CHARUCO_CORNERS
            if corners < 0:
                corner_text, corner_colour = "board corners: n/a", (0, 255, 0)
            else:
                quality = ("good" if corners >= cfg.GOOD_CHARUCO_CORNERS
                           else "marginal — move closer" if board_ok
                           else f"TOO FEW (need {cfg.MIN_CHARUCO_CORNERS})")
                corner_text = f"corners: {corners}/{max_corners}  {quality}"
                corner_colour = ((0, 255, 0) if corners >= cfg.GOOD_CHARUCO_CORNERS
                                 else (0, 200, 255) if board_ok else (0, 0, 255))
            if tilt is None:
                tilt_text, tilt_colour = "", None
            elif tilt >= cfg.GOOD_BOARD_TILT_DEG:
                tilt_text, tilt_colour = f"tilt: {tilt:.0f} deg  good", (0, 255, 0)
            else:
                tilt_text = (f"tilt: {tilt:.0f} deg  TOO FLAT — angle the board "
                             f"(want >{cfg.GOOD_BOARD_TILT_DEG:.0f})")
                tilt_colour = (0, 200, 255)

            lines = [
                (f"next: {idx:04d}.png   saved: {idx - 1}", (0, 255, 0)),
                (f"pose: {'ok' if pose_ok else 'MISSING/STALE — is roscore up?'}",
                 (0, 255, 0) if pose_ok else (0, 0, 255)),
                (corner_text, corner_colour),
            ]
            if tilt_colour is not None:
                lines.append((tilt_text, tilt_colour))
            for i, (text, colour) in enumerate(lines):
                cv2.putText(view, text, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, colour, 2, cv2.LINE_AA)
            cv2.imshow("capture", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key != ord("s"):
                continue

            if not pose_ok:
                print("not saved: robot pose missing or stale")
                continue
            if not board_ok:
                print(f"not saved: only {corners} ChArUco corners "
                      f"(need {cfg.MIN_CHARUCO_CORNERS}) — this shot would be discarded")
                continue

            name = f"{idx:04d}.png"
            if not cv2.imwrite(str(session.images / name), color):
                print(f"failed to write {name}")
                continue
            try:
                append_pose(session.poses_json, {
                    "image": name,
                    "T_base_gripper": T.tolist(),
                    "image_timestamp_ms": float(ts_ms),
                    "pose_receive_time_unix": float(t_recv),
                    "pose_topic": args.pose_topic,
                    "pose_source": args.pose_source,
                })
            except Exception as e:
                print(f"failed to append pose ({e}); removing image")
                (session.images / name).unlink(missing_ok=True)
                continue

            print(f"saved {name}  ({corners} corners)")
            idx += 1
    finally:
        sub.stop()
        cam.stop()
        cv2.destroyAllWindows()
        n = next_index(session) - 1
        print(f"\n{n} captures in {session.root}")
        if n < cfg.MIN_SAMPLES:
            print(f"calibrate.py needs at least {cfg.MIN_SAMPLES}.")


if __name__ == "__main__":
    main()
