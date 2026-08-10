#!/usr/bin/env python3
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ----------------------------
# Basic SE(3) helpers
# ----------------------------

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec.reshape(3))


def T_to_rvec_tvec(T: np.ndarray):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    return rvec, tvec


def rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.trace(R)
    val = (trace - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


# ----------------------------
# Quaternion helpers for averaging
# ----------------------------

def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)

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


def average_rotations(rotations):
    # Markley quaternion averaging
    A = np.zeros((4, 4), dtype=np.float64)
    q_ref = None

    for R in rotations:
        q = R_to_quat_xyzw(R)
        if q_ref is None:
            q_ref = q.copy()
        if np.dot(q, q_ref) < 0:
            q = -q
        A += np.outer(q, q)

    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    if q_avg[3] < 0:
        q_avg = -q_avg
    return quat_xyzw_to_R(q_avg)


# ----------------------------
# ChArUco helpers
# ----------------------------

def get_aruco_dictionary(dictionary_name: str):
    name_to_id = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    return cv2.aruco.getPredefinedDictionary(name_to_id[dictionary_name])


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str,
    use_legacy_pattern: bool = False,
):
    dictionary = get_aruco_dictionary(dictionary_name)

    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length_m,
            marker_length_m,
            dictionary,
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_length_m,
            marker_length_m,
            dictionary,
        )

    if use_legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)

    return board, dictionary


def detect_charuco_pose_and_corners(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board,
    dictionary,
    min_charuco_corners: int = 6,
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    charuco_corners = None
    charuco_ids = None
    marker_ids = None

    if hasattr(cv2.aruco, "CharucoDetector"):
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.cameraMatrix = camera_matrix
        charuco_params.distCoeffs = dist_coeffs

        detector_params = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "CORNER_REFINE_NONE"):
            detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        else:
            detector_params = cv2.aruco.DetectorParameters_create()
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=detector_params
            )

        if marker_ids is None or len(marker_ids) == 0:
            return None

        if not hasattr(cv2.aruco, "interpolateCornersCharuco"):
            return None

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )
        if retval is None or retval < min_charuco_corners:
            return None

    if charuco_ids is None or len(charuco_ids) < min_charuco_corners:
        return None

    if hasattr(board, "matchImagePoints"):
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        obj_points = board.chessboardCorners[charuco_ids.flatten(), :]
        img_points = charuco_corners.reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        obj_points = obj_points.reshape(-1, 1, 3)
        img_points = img_points.reshape(-1, 1, 2)

    if not ok:
        return None

    T_cam_board = rvec_tvec_to_T(rvec, tvec)

    return {
        "T_cam_board": T_cam_board,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "obj_points": obj_points,
        "img_points": img_points,
        "num_markers": 0 if marker_ids is None else len(marker_ids),
        "num_charuco": len(charuco_ids),
    }


# ----------------------------
# IO
# ----------------------------

def load_intrinsics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K, D


def load_robot_poses(json_path):
    with open(json_path, "r") as f:
        items = json.load(f)

    image_names = []
    T_base_gripper_list = []
    for item in items:
        image_names.append(item["image"])
        T = np.array(item["T_base_gripper"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Bad shape for {item['image']}: {T.shape}")
        T_base_gripper_list.append(T)

    return image_names, T_base_gripper_list


# ----------------------------
# Main validation
# ----------------------------

def main():
    # Files
    intrinsics_path = "color_intrinsics.json"
    poses_path = "robot_poses.json"
    images_dir = Path("images")
    t_base_color_path = "T_base_color.npy"

    # Board definition: CHANGE ONLY IF YOUR PRINTED BOARD DIFFERS
    squares_x = 8
    squares_y = 8
    square_length_m = 0.021
    marker_length_m = 0.016
    dictionary_name = "DICT_5X5_50"
    use_legacy_pattern = True

    K, D = load_intrinsics(intrinsics_path)
    image_names, T_base_gripper_list = load_robot_poses(poses_path)
    T_base_color = np.load(t_base_color_path)

    board, dictionary = create_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_length_m,
        marker_length_m=marker_length_m,
        dictionary_name=dictionary_name,
        use_legacy_pattern=use_legacy_pattern,
    )

    valid = []
    for img_name, T_base_gripper in zip(image_names, T_base_gripper_list):
        image_path = images_dir / img_name
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping {img_name}: image not found")
            continue

        result = detect_charuco_pose_and_corners(
            image_bgr=image,
            camera_matrix=K,
            dist_coeffs=D,
            board=board,
            dictionary=dictionary,
            min_charuco_corners=6,
        )
        if result is None:
            print(f"Skipping {img_name}: ChArUco board not found")
            continue

        T_cam_board = result["T_cam_board"]
        T_gripper_board = invert_T(T_base_gripper) @ T_base_color @ T_cam_board

        valid.append({
            "image": img_name,
            "T_base_gripper": T_base_gripper,
            "T_cam_board": T_cam_board,
            "T_gripper_board": T_gripper_board,
            "charuco_corners": result["charuco_corners"],
            "charuco_ids": result["charuco_ids"],
            "obj_points": result["obj_points"],
            "img_points": result["img_points"],
            "num_markers": result["num_markers"],
            "num_charuco": result["num_charuco"],
        })

    if len(valid) < 3:
        raise RuntimeError(f"Too few valid samples: {len(valid)}")

    # Estimate a constant T_gripper_board
    translations = np.array([v["T_gripper_board"][:3, 3] for v in valid], dtype=np.float64)
    rotations = [v["T_gripper_board"][:3, :3] for v in valid]

    t_ref = translations.mean(axis=0)
    R_ref = average_rotations(rotations)
    T_gripper_board_ref = make_T(R_ref, t_ref)

    # Metrics
    trans_errors_mm = []
    rot_errors_deg = []
    reproj_errors_px = []

    for v in valid:
        T_gb_i = v["T_gripper_board"]
        dtrans_mm = 1000.0 * np.linalg.norm(T_gb_i[:3, 3] - t_ref)
        drot_deg = rotation_angle_deg(R_ref.T @ T_gb_i[:3, :3])

        trans_errors_mm.append(dtrans_mm)
        rot_errors_deg.append(drot_deg)

        # Predict board pose in camera using calibrated T_base_color and constant T_gripper_board_ref
        T_cam_board_pred = invert_T(T_base_color) @ v["T_base_gripper"] @ T_gripper_board_ref
        rvec_pred, tvec_pred = T_to_rvec_tvec(T_cam_board_pred)

        obj_points = v["obj_points"]
        img_points_meas = v["img_points"].reshape(-1, 2)

        proj_points, _ = cv2.projectPoints(obj_points, rvec_pred, tvec_pred, K, D)
        proj_points = proj_points.reshape(-1, 2)

        err = np.linalg.norm(proj_points - img_points_meas, axis=1)
        reproj_rmse = float(np.sqrt(np.mean(err ** 2)))
        reproj_errors_px.append(reproj_rmse)

    trans_errors_mm = np.array(trans_errors_mm)
    rot_errors_deg = np.array(rot_errors_deg)
    reproj_errors_px = np.array(reproj_errors_px)

    print("\n=== Calibration validation summary ===")
    print(f"Valid samples: {len(valid)}")
    print()
    print("Gripper->board consistency:")
    print(f"  translation error mean   : {trans_errors_mm.mean():.3f} mm")
    print(f"  translation error median : {np.median(trans_errors_mm):.3f} mm")
    print(f"  translation error max    : {trans_errors_mm.max():.3f} mm")
    print(f"  rotation error mean      : {rot_errors_deg.mean():.3f} deg")
    print(f"  rotation error median    : {np.median(rot_errors_deg):.3f} deg")
    print(f"  rotation error max       : {rot_errors_deg.max():.3f} deg")
    print()
    print("Image reprojection:")
    print(f"  reprojection RMSE mean   : {reproj_errors_px.mean():.3f} px")
    print(f"  reprojection RMSE median : {np.median(reproj_errors_px):.3f} px")
    print(f"  reprojection RMSE max    : {reproj_errors_px.max():.3f} px")
    print()

    print("Per-image results:")
    for img_name, te, re, pe in zip(
        [v["image"] for v in valid],
        trans_errors_mm,
        rot_errors_deg,
        reproj_errors_px,
    ):
        print(f"  {img_name}: trans={te:.3f} mm, rot={re:.3f} deg, reproj={pe:.3f} px")

    np.save("T_gripper_board_ref.npy", T_gripper_board_ref)
    print("\nSaved reference board mount transform to T_gripper_board_ref.npy")


if __name__ == "__main__":
    main()