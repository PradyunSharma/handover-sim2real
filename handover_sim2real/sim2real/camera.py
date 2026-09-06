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


@dataclass(frozen=True)
class CameraModel:
    """The handful of things that differ between D400 bodies and matter here.

    Intrinsics are read off the device and need no table. These four do not
    appear in any stream profile, and getting them wrong is quiet rather than
    loud — a min_z set for the wrong body does not error, it just returns
    garbage depth in the near field and calls it a measurement.

    THE BASELINE IS WHY YOU WOULD SWAP THE BODY AT ALL. Stereo depth noise goes
    as sigma_z = z^2 * sigma_disparity / (f * baseline), so the D455's 95 mm
    against the D435's 50 mm is very nearly a halving of depth error at the same
    range, for the same sensor and the same disparity quality. No resolution
    change on a D435 comes close to that; the measured spread across every mode
    it offers was about 12%.

    MIN_Z IS THE PRICE, and it is the one that can bite. A longer baseline means
    the two imagers stop overlapping sooner, so the D455 sees nothing closer
    than about 0.4 m where the D435 manages 0.1. That is comfortable for a
    tripod at 0.6-1.5 m and disqualifying for a wrist camera at 0.3 m.

    RGB_HFOV_DEG matters because this pipeline aligns depth TO COLOUR, so the
    colour frame clips the cloud. On a D435 that is a real loss — 69 deg of
    colour against 87 of depth, and only 55 in the 4:3 modes. The D455's colour
    is WIDER than its depth, so alignment throws nothing away.
    """
    name: str
    baseline_m: float
    min_z_m: float
    max_range_m: float
    rgb_hfov_deg: float


CAMERA_MODELS = {
    "d435": CameraModel("D435", 0.050, 0.10, 3.0, 69.0),
    "d435i": CameraModel("D435i", 0.050, 0.10, 3.0, 69.0),
    "d415": CameraModel("D415", 0.055, 0.16, 3.0, 69.0),
    "d455": CameraModel("D455", 0.095, 0.40, 6.0, 90.0),
}
DEFAULT_CAMERA_MODEL = "d435"


def model_from_device_name(device_name: str) -> Optional[str]:
    """'Intel RealSense D455' -> 'd455'. None if it is not one we have a row for.

    Matched longest-key-first so 'd435i' is not swallowed by the 'd435' prefix.
    """
    low = str(device_name).lower().replace(" ", "")
    for key in sorted(CAMERA_MODELS, key=len, reverse=True):
        if key in low:
            return key
    return None


class RealSenseCamera:
    def __init__(
        self,
        color_size: Tuple[int, int] = (640, 480),
        depth_size: Tuple[int, int] = (640, 480),
        fps: int = 30,
        serial: Optional[str] = None,
    ) -> None:
        self.color_size = color_size
        self.depth_size = depth_size
        self.fps = fps
        # With more than one RealSense attached, librealsense binds whichever it
        # enumerates first. Pin the device when it matters — the wrist and the
        # tripod camera are not interchangeable.
        self.serial = serial

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)

        self.depth_scale: Optional[float] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        # Filled from the device at start(). Detected rather than declared: the
        # body is a physical fact the driver already knows, and a flag that has
        # to be repeated across calibration and deployment is a flag that will
        # eventually disagree with the hardware in one of them.
        self.device_name: Optional[str] = None
        self.model_key: Optional[str] = None
        self.started = False

    @property
    def model(self) -> CameraModel:
        """The body's constants, falling back to the D435 row if unrecognised."""
        return CAMERA_MODELS[self.model_key or DEFAULT_CAMERA_MODEL]

    def start(self) -> None:
        if self.serial is not None:
            self.config.enable_device(str(self.serial))
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

        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self.device_name = str(device.get_info(rs.camera_info.name))
        self.model_key = model_from_device_name(self.device_name)

        # Warm up a few frames for auto-exposure/stability.
        #
        # THIS IS WHERE A BAD LINK SHOWS UP, and it is worth catching here
        # because the bare librealsense message ("Frame didn't arrive within
        # 5000") reads like a timeout to tune rather than a fault to fix. Note
        # what has already succeeded by this line: the device enumerated, and
        # pipeline.start() accepted both modes — which it validates against the
        # device's own profile list, without a single frame flowing. So getting
        # this far says nothing about bandwidth or power, and anything that only
        # reads stream profiles (generate_color_intrinsics.py) will happily
        # succeed against a camera that cannot stream at all.
        try:
            for _ in range(10):
                self.pipeline.wait_for_frames()
        except RuntimeError as err:
            usb = "unknown"
            try:
                usb = str(device.get_info(rs.camera_info.usb_type_descriptor))
            except Exception:
                pass
            raise RuntimeError(
                f"{self.device_name or 'RealSense'} accepted color "
                f"{self.color_size[0]}x{self.color_size[1]} + depth "
                f"{self.depth_size[0]}x{self.depth_size[1]} @ {self.fps}fps but "
                f"delivered no frames ({err}).\n"
                f"  USB link reports: {usb}\n"
                "  The modes are valid, so this is a link or power fault, not a "
                "config one. In order of likelihood: a USB2 or charge-only "
                "cable (the link must read 3.x); a hub or front-panel port that "
                "cannot supply enough power; another process holding the "
                "device; two cameras on one controller. A D455 draws more than "
                "a D435 and is less forgiving of a marginal link.\n"
                "  Colour alone is roughly half the bandwidth: if --stream "
                "color works and this does not, the link is the constraint."
            ) from err

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