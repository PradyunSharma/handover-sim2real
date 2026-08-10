#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import argparse
import json
import math
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import roslibpy


# ----------------------------
# RealSense camera
# ----------------------------

@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    coeffs: Tuple[float, float, float, float, float]


class RealSenseCamera:
    def __init__(
        self,
        color_size: Tuple[int, int] = (640, 480),
        depth_size: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ) -> None:
        self.color_size = color_size
        self.depth_size = depth_size
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)

        self.depth_scale: Optional[float] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.started = False

    def start(self) -> None:
        self.config.enable_stream(
            rs.stream.color,
            self.color_size[0],
            self.color_size[1],
            rs.format.bgr8,
            self.fps,
        )
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_size[0],
            self.depth_size[1],
            rs.format.z16,
            self.fps,
        )

        profile = self.pipeline.start(self.config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        for _ in range(10):
            self.pipeline.wait_for_frames()

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()

        if not color_frame:
            raise RuntimeError("Failed to get initial aligned color frame.")

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        self.intrinsics = CameraIntrinsics(
            width=intr.width,
            height=intr.height,
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            distortion_model=str(intr.model),
            coeffs=tuple(float(c) for c in intr.coeffs[:5]),
        )

        self.started = True

    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    def get_frames(self) -> Tuple[np.ndarray, np.ndarray, float]:
        if not self.started:
            raise RuntimeError("Camera not started. Call start() first.")

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get aligned color/depth frames.")

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)

        timestamp_ms = float(color_frame.get_timestamp())
        return color_bgr, depth_m, timestamp_ms

    def get_intrinsics(self) -> CameraIntrinsics:
        if self.intrinsics is None:
            raise RuntimeError("Intrinsics unavailable. Call start() first.")
        return self.intrinsics


# ----------------------------
# SE(3) helpers
# ----------------------------

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def quat_normalize(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    return [x / n, y / n, z / n, w / n]


def quat_to_rotation_matrix(q):
    x, y, z, w = quat_normalize(q)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)
    return R


def pose_stamped_msg_to_T(msg: dict) -> np.ndarray:
    p = msg["pose"]["position"]
    o = msg["pose"]["orientation"]

    t = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    q = [o["x"], o["y"], o["z"], o["w"]]
    R = quat_to_rotation_matrix(q)

    return make_T(R, t)


def franka_state_msg_to_T(msg: dict) -> np.ndarray:
    flat = msg.get("O_T_EE", None)
    if flat is None or len(flat) != 16:
        raise ValueError("Message does not contain valid O_T_EE.")
    return np.array(flat, dtype=np.float64).reshape(4, 4, order="F")


# ----------------------------
# rosbridge subscriber
# ----------------------------

class RobotPoseSubscriber:
    def __init__(
        self,
        host: str,
        port: int,
        topic_name: str,
        topic_type: str,
        pose_source: str,
    ) -> None:
        self.host = host
        self.port = port
        self.topic_name = topic_name
        self.topic_type = topic_type
        self.pose_source = pose_source

        self.client = roslibpy.Ros(host=self.host, port=self.port)
        self.topic: Optional[roslibpy.Topic] = None

        self._lock = threading.Lock()
        self._latest_T_base_gripper: Optional[np.ndarray] = None
        self._latest_receive_time: Optional[float] = None
        self._latest_raw_msg: Optional[dict] = None

    def start(self) -> None:
        self.client.run()

        timeout_s = 5.0
        t0 = time.time()
        while not self.client.is_connected:
            if time.time() - t0 > timeout_s:
                raise RuntimeError(
                    f"Could not connect to rosbridge at ws://{self.host}:{self.port}"
                )
            time.sleep(0.05)

        self.topic = roslibpy.Topic(
            self.client,
            self.topic_name,
            self.topic_type,
        )
        self.topic.subscribe(self._callback)

    def stop(self) -> None:
        if self.topic is not None:
            try:
                self.topic.unsubscribe()
            except Exception:
                pass

        try:
            self.client.terminate()
        except Exception:
            pass

    def _callback(self, msg: dict) -> None:
        try:
            if self.pose_source == "franka_state":
                T = franka_state_msg_to_T(msg)
            elif self.pose_source == "pose_stamped":
                T = pose_stamped_msg_to_T(msg)
            else:
                return
        except Exception:
            return

        with self._lock:
            self._latest_T_base_gripper = T
            self._latest_receive_time = time.time()
            self._latest_raw_msg = msg

    def get_latest_pose(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            T = None if self._latest_T_base_gripper is None else self._latest_T_base_gripper.copy()
            t = self._latest_receive_time
        return T, t

    def is_connected(self) -> bool:
        return bool(self.client.is_connected)


# ----------------------------
# File helpers
# ----------------------------

def get_next_image_index(images_dir: Path) -> int:
    existing = sorted(images_dir.glob("*.png"))
    if not existing:
        return 1

    max_idx = 0
    for path in existing:
        try:
            idx = int(path.stem)
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return max_idx + 1


def make_depth_vis(depth_m: np.ndarray, max_depth_m: float = 1.5) -> np.ndarray:
    depth_vis = depth_m.copy()
    depth_vis[~np.isfinite(depth_vis)] = 0.0
    depth_vis = np.clip(depth_vis, 0.0, max_depth_m)
    depth_vis = (depth_vis / max_depth_m * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_vis


def load_existing_pose_entries(json_path: Path) -> list:
    if not json_path.exists():
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} exists but is not a JSON list.")
    return data


def append_robot_pose_entry(
    json_path: Path,
    image_filename: str,
    T_base_gripper: np.ndarray,
    image_timestamp_ms: float,
    pose_receive_time_unix: float,
    pose_topic: str,
    pose_source: str,
) -> None:
    entries = load_existing_pose_entries(json_path)
    entries.append(
        {
            "image": image_filename,
            "T_base_gripper": T_base_gripper.tolist(),
            "image_timestamp_ms": float(image_timestamp_ms),
            "pose_receive_time_unix": float(pose_receive_time_unix),
            "pose_topic": pose_topic,
            "pose_source": pose_source,
        }
    )

    tmp_path = json_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(entries, f, indent=2)
    tmp_path.replace(json_path)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture RealSense images and append matching robot poses over rosbridge."
    )
    parser.add_argument("--rosbridge-host", type=str, default="172.16.0.7")
    parser.add_argument("--rosbridge-port", type=int, default=9090)

    parser.add_argument(
        "--pose-source",
        choices=["franka_state", "pose_stamped"],
        default="franka_state",
        help="How to interpret the subscribed robot pose topic.",
    )
    parser.add_argument(
        "--pose-topic",
        type=str,
        default="/franka_state_controller/franka_states",
        help="Robot pose topic.",
    )
    parser.add_argument(
        "--pose-type",
        type=str,
        default="franka_msgs/FrankaState",
        help="ROS message type of the pose topic.",
    )

    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument("--poses-json", type=str, default="robot_poses.json")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--max-pose-age",
        type=float,
        default=1.0,
        help="Maximum allowed age in seconds of latest received pose when saving.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    poses_json = Path(args.poses_json)
    next_idx = get_next_image_index(images_dir)

    cam = RealSenseCamera(
        color_size=(args.width, args.height),
        depth_size=(args.width, args.height),
        fps=args.fps,
    )
    pose_sub = RobotPoseSubscriber(
        host=args.rosbridge_host,
        port=args.rosbridge_port,
        topic_name=args.pose_topic,
        topic_type=args.pose_type,
        pose_source=args.pose_source,
    )

    cam.start()
    pose_sub.start()

    print(f"Connected to rosbridge at {args.rosbridge_host}:{args.rosbridge_port}")
    print(f"Subscribed to {args.pose_topic} [{args.pose_type}]")
    print(f"Pose source mode: {args.pose_source}")
    print(f"Saving images to: {images_dir.resolve()}")
    print(f"Appending poses to: {poses_json.resolve()}")
    print()
    print("Controls:")
    print("  s  -> save image and append pose")
    print("  q  -> quit")
    print("  ESC -> quit")

    try:
        while True:
            color_bgr, depth_m, image_ts_ms = cam.get_frames()
            depth_vis = make_depth_vis(depth_m, max_depth_m=1.5)

            latest_pose, latest_pose_t = pose_sub.get_latest_pose()
            pose_ok = (
                latest_pose is not None
                and latest_pose_t is not None
                and (time.time() - latest_pose_t) <= args.max_pose_age
            )

            overlay = color_bgr.copy()
            cv2.putText(
                overlay,
                f"next save: {next_idx:04d}.png",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"rosbridge: {'connected' if pose_sub.is_connected() else 'disconnected'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if pose_sub.is_connected() else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"latest pose: {'ok' if pose_ok else 'missing/stale'}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if pose_ok else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "press 's' to save image+pose, 'q' or ESC to quit",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("color", overlay)
            cv2.imshow("depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if not pose_ok:
                    print("Not saving: latest robot pose is missing or stale.")
                    continue

                image_filename = f"{next_idx:04d}.png"
                image_path = images_dir / image_filename

                ok = cv2.imwrite(str(image_path), color_bgr)
                if not ok:
                    print(f"Failed to save image: {image_path}")
                    continue

                try:
                    append_robot_pose_entry(
                        json_path=poses_json,
                        image_filename=image_filename,
                        T_base_gripper=latest_pose,
                        image_timestamp_ms=image_ts_ms,
                        pose_receive_time_unix=latest_pose_t,
                        pose_topic=args.pose_topic,
                        pose_source=args.pose_source,
                    )
                except Exception as e:
                    print(f"Failed to append pose JSON: {e}")
                    try:
                        image_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue

                print(f"Saved {image_path} and appended pose to {poses_json}")
                next_idx += 1

            elif key == ord("q") or key == 27:
                break

    finally:
        pose_sub.stop()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()