from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs


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

        # Warm up a few frames for auto-exposure/stability.
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
        """
        Returns:
            color_bgr: uint8 HxWx3
            depth_m: float32 HxW in meters, aligned to color
            timestamp_ms: float
        """
        if not self.started:
            raise RuntimeError("Camera not started. Call start() first.")

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get aligned color/depth frames.")

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
        depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)

        timestamp_ms = float(color_frame.get_timestamp())
        return color_bgr, depth_m, timestamp_ms

    def get_intrinsics(self) -> CameraIntrinsics:
        if self.intrinsics is None:
            raise RuntimeError("Intrinsics unavailable. Call start() first.")
        return self.intrinsics

    def depth_to_pointcloud(
        self,
        depth_m: np.ndarray,
        color_bgr: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        stride: int = 1,
        min_depth: float = 0.05,
        max_depth: float = 2.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Convert aligned depth image to point cloud in camera frame.

        Args:
            depth_m: HxW float32 depth in meters
            color_bgr: optional HxWx3 uint8 image aligned to depth
            mask: optional HxW bool/uint8 mask; points kept where mask is True/nonzero
            stride: sample every N pixels
            min_depth, max_depth: keep points within this range

        Returns:
            points_xyz: Nx3 float32 in camera frame (meters)
            colors_rgb: optional Nx3 float32 in [0,1]
            pixel_indices: Nx2 int32 as (v, u)
        """
        intr = self.get_intrinsics()

        if depth_m.ndim != 2:
            raise ValueError("depth_m must be HxW.")
        h, w = depth_m.shape
        if h != intr.height or w != intr.width:
            raise ValueError(
                f"Depth shape {depth_m.shape} does not match intrinsics "
                f"({intr.height}, {intr.width})."
            )

        if mask is None:
            valid = np.ones_like(depth_m, dtype=bool)
        else:
            valid = mask.astype(bool)

        valid &= np.isfinite(depth_m)
        valid &= depth_m > min_depth
        valid &= depth_m < max_depth

        if stride > 1:
            subsample = np.zeros_like(valid, dtype=bool)
            subsample[::stride, ::stride] = True
            valid &= subsample

        v, u = np.nonzero(valid)
        z = depth_m[v, u]

        x = (u.astype(np.float32) - intr.cx) * z / intr.fx
        y = (v.astype(np.float32) - intr.cy) * z / intr.fy

        points_xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        pixel_indices = np.stack([v, u], axis=1).astype(np.int32)

        colors_rgb: Optional[np.ndarray] = None
        if color_bgr is not None:
            if color_bgr.shape[:2] != depth_m.shape:
                raise ValueError("color_bgr must be aligned to depth_m.")
            colors_bgr = color_bgr[v, u].astype(np.float32) / 255.0
            colors_rgb = colors_bgr[:, ::-1].copy()

        return points_xyz, colors_rgb, pixel_indices