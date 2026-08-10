from __future__ import annotations

import numpy as np


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 rigid transform.
    T maps points from frame b -> a.
    Returns T^{-1}, which maps points from frame a -> b.
    """
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4,4), got {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(T: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 transform to Nx3 points.
    """
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4,4), got {T.shape}")
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points_xyz.shape}")

    n = points_xyz.shape[0]
    if n == 0:
        return points_xyz.astype(np.float32, copy=False)

    ones = np.ones((n, 1), dtype=np.float64)
    points_h = np.concatenate([points_xyz.astype(np.float64), ones], axis=1)
    transformed_h = (T @ points_h.T).T
    return transformed_h[:, :3].astype(np.float32)


def camera_to_ee_transform(
    T_base_cam: np.ndarray,
    T_base_ee: np.ndarray,
) -> np.ndarray:
    """
    Build T_ee_cam from:
      T_base_cam: camera -> base
      T_base_ee: ee -> base
    Returns:
      T_ee_cam: camera -> ee
    """
    T_ee_base = invert_transform(T_base_ee)
    T_ee_cam = T_ee_base @ T_base_cam
    return T_ee_cam
