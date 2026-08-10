from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CloudExtractionResult:
    hand_xyz: np.ndarray
    object_xyz: np.ndarray
    hand_center: Optional[np.ndarray]
    debug: dict[str, Any]


def _ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"hand_mask must be HxW, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def _sample_or_pad_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Uniformly sample or repeat points to get exactly num_points rows.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got shape {points.shape}")

    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    if n == num_points:
        return points.astype(np.float32, copy=False)

    if n > num_points:
        idx = np.random.choice(n, size=num_points, replace=False)
        return points[idx].astype(np.float32, copy=False)

    idx = np.random.choice(n, size=num_points - n, replace=True)
    padded = np.concatenate([points, points[idx]], axis=0)
    return padded.astype(np.float32, copy=False)


def extract_hand_object_clouds(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    hand_mask: np.ndarray,
    cam,
    last_hand_xyz: Optional[np.ndarray] = None,
    last_object_xyz: Optional[np.ndarray] = None,
    crop_radius_m: float = 0.12,
    object_max_radius_m: float = 0.10,
    min_depth_m: float = 0.10,
    max_depth_m: float = 1.50,
    full_cloud_stride: int = 2,
    hand_cloud_stride: int = 1,
    min_hand_points: int = 100,
    min_object_points: int = 80,
) -> CloudExtractionResult:
    """
    Extract hand and object point clouds in the external camera frame.

    Assumptions:
    - hand_mask is aligned with color/depth.
    - cam.depth_to_pointcloud(...) returns:
        points_xyz: Nx3
        colors_rgb: optional Nx3
        pixel_indices: Nx2 as (v, u)

    Returns:
    - hand_xyz: hand points in camera frame
    - object_xyz: object points in camera frame
    - hand_center: 3D crop center used this frame, or None
    - debug: counts and fallback info
    """
    hand_mask = _ensure_uint8_mask(hand_mask)

    debug: dict[str, Any] = {
        "used_last_hand": False,
        "used_last_object": False,
        "hand_center": None,
        "raw_hand_points": 0,
        "crop_points": 0,
        "crop_hand_points": 0,
        "crop_object_points": 0,
    }

    # 1) Hand cloud directly from the 2D hand mask.
    hand_points_xyz, _, _ = cam.depth_to_pointcloud(
        depth_m=depth_m,
        color_bgr=color_bgr,
        mask=hand_mask,
        stride=hand_cloud_stride,
        min_depth=min_depth_m,
        max_depth=max_depth_m,
    )
    hand_points_xyz = hand_points_xyz.astype(np.float32, copy=False)
    debug["raw_hand_points"] = int(len(hand_points_xyz))

    if len(hand_points_xyz) >= min_hand_points:
        hand_center = np.median(hand_points_xyz, axis=0).astype(np.float32)
    else:
        hand_center = None

    # If the hand is too weak this frame, fall back early if possible.
    if hand_center is None:
        if last_hand_xyz is not None and len(last_hand_xyz) > 0:
            debug["used_last_hand"] = True
            hand_points_xyz = last_hand_xyz.astype(np.float32, copy=False)
            hand_center = np.median(hand_points_xyz, axis=0).astype(np.float32)
        else:
            hand_points_xyz = np.zeros((0, 3), dtype=np.float32)

    if hand_center is not None:
        debug["hand_center"] = hand_center.copy()

    # 2) Full cloud for local crop around the hand.
    full_points_xyz, _, pixel_indices = cam.depth_to_pointcloud(
        depth_m=depth_m,
        color_bgr=color_bgr,
        mask=None,
        stride=full_cloud_stride,
        min_depth=min_depth_m,
        max_depth=max_depth_m,
    )
    full_points_xyz = full_points_xyz.astype(np.float32, copy=False)
    pixel_indices = pixel_indices.astype(np.int32, copy=False)

    if hand_center is None or len(full_points_xyz) == 0:
        # Cannot crop meaningfully; use fallbacks if available.
        if last_object_xyz is not None and len(last_object_xyz) > 0:
            object_xyz = last_object_xyz.astype(np.float32, copy=False)
            debug["used_last_object"] = True
        else:
            object_xyz = np.zeros((0, 3), dtype=np.float32)

        return CloudExtractionResult(
            hand_xyz=hand_points_xyz,
            object_xyz=object_xyz,
            hand_center=hand_center,
            debug=debug,
        )

    dist_to_center = np.linalg.norm(full_points_xyz - hand_center[None, :], axis=1)
    crop_keep = dist_to_center < crop_radius_m

    crop_points_xyz = full_points_xyz[crop_keep]
    crop_pixel_indices = pixel_indices[crop_keep]
    debug["crop_points"] = int(len(crop_points_xyz))

    if len(crop_points_xyz) == 0:
        if last_object_xyz is not None and len(last_object_xyz) > 0:
            object_xyz = last_object_xyz.astype(np.float32, copy=False)
            debug["used_last_object"] = True
        else:
            object_xyz = np.zeros((0, 3), dtype=np.float32)

        return CloudExtractionResult(
            hand_xyz=hand_points_xyz,
            object_xyz=object_xyz,
            hand_center=hand_center,
            debug=debug,
        )

    # 3) Re-label cropped 3D points using the 2D hand mask.
    vv = crop_pixel_indices[:, 0]
    uu = crop_pixel_indices[:, 1]
    crop_hand_flags = hand_mask[vv, uu] > 0

    hand_crop_xyz = crop_points_xyz[crop_hand_flags].astype(np.float32, copy=False)
    object_crop_xyz = crop_points_xyz[~crop_hand_flags].astype(np.float32, copy=False)

    debug["crop_hand_points"] = int(len(hand_crop_xyz))
    debug["crop_object_points"] = int(len(object_crop_xyz))

    # 4) Keep object points reasonably close to the hand center
    # to suppress table/background leakage.
    if len(object_crop_xyz) > 0:
        obj_dist = np.linalg.norm(object_crop_xyz - hand_center[None, :], axis=1)
        object_crop_xyz = object_crop_xyz[obj_dist < object_max_radius_m]

    # 5) Choose final hand cloud.
    if len(hand_crop_xyz) >= min_hand_points:
        hand_xyz = hand_crop_xyz
    elif len(hand_points_xyz) >= min_hand_points:
        hand_xyz = hand_points_xyz
    elif last_hand_xyz is not None and len(last_hand_xyz) > 0:
        hand_xyz = last_hand_xyz.astype(np.float32, copy=False)
        debug["used_last_hand"] = True
    else:
        hand_xyz = hand_crop_xyz  # possibly empty

    # 6) Choose final object cloud with fallback.
    if len(object_crop_xyz) >= min_object_points:
        object_xyz = object_crop_xyz.astype(np.float32, copy=False)
    elif last_object_xyz is not None and len(last_object_xyz) > 0:
        object_xyz = last_object_xyz.astype(np.float32, copy=False)
        debug["used_last_object"] = True
    else:
        object_xyz = object_crop_xyz.astype(np.float32, copy=False)

    return CloudExtractionResult(
        hand_xyz=hand_xyz.astype(np.float32, copy=False),
        object_xyz=object_xyz.astype(np.float32, copy=False),
        hand_center=hand_center,
        debug=debug,
    )


def build_policy_point_tensor(
    hand_xyz: np.ndarray,
    object_xyz: np.ndarray,
    num_hand_points: int = 128,
    num_object_points: int = 896,
) -> np.ndarray:
    """
    Build the policy input cloud with one-hot labels:
      hand points  -> [x, y, z, 1, 0]
      object points-> [x, y, z, 0, 1]

    Returns:
      (num_hand_points + num_object_points, 5) float32
    """
    hand_fixed = _sample_or_pad_points(hand_xyz, num_hand_points)
    obj_fixed = _sample_or_pad_points(object_xyz, num_object_points)

    hand_labels = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (num_hand_points, 1))
    obj_labels = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (num_object_points, 1))

    hand_feat = np.concatenate([hand_fixed, hand_labels], axis=1)
    obj_feat = np.concatenate([obj_fixed, obj_labels], axis=1)

    return np.concatenate([hand_feat, obj_feat], axis=0).astype(np.float32)