import json
import numpy as np
import cv2


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


def T_to_R_t(T: np.ndarray):
    return T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()


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
    if dictionary_name not in name_to_id:
        raise ValueError(f"Unsupported dictionary: {dictionary_name}")
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

def detect_charuco_pose(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board,
    dictionary,
    min_charuco_corners: int = 6,
    debug: bool = False,
):
    """
    Returns T_cam_target = ^cT_t for a ChArUco board, or None if not enough detections.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    charuco_corners = None
    charuco_ids = None

    # Preferred path for newer OpenCV versions
    if hasattr(cv2.aruco, "CharucoDetector"):
        try:
            detector = cv2.aruco.CharucoDetector(board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        except Exception:
            charuco_corners, charuco_ids = None, None

    # Fallback path for older OpenCV versions
    if charuco_ids is None:
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

        if hasattr(cv2.aruco, "interpolateCornersCharuco"):
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
        else:
            # No legacy interpolation API available
            return None

    if charuco_ids is None or len(charuco_ids) < min_charuco_corners:
        return None

    # Pose estimation
    if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            camera_matrix,
            dist_coeffs,
            None,
            None,
        )
        if not ok:
            return None
    else:
        # Newer OpenCV deprecates some old pose helpers; solvePnP works too.
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
            # Older-style fallback using board chessboard corners
            obj_points = board.chessboardCorners[charuco_ids.flatten(), :]
            img_points = charuco_corners.reshape(-1, 2)
            ok, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not ok:
            return None

    return rvec_tvec_to_T(rvec, tvec)


# ----------------------------
# Hand-eye calibration
# ----------------------------

def calibrate_fixed_camera_to_robot_base(
    robot_base_T_gripper_list,
    cam_T_target_list,
    method=cv2.CALIB_HAND_EYE_TSAI,
):
    """
    Eye-to-hand calibration.

    Inputs:
      robot_base_T_gripper_list: list of ^baseT_gripper from Franka FK
      cam_T_target_list:         list of ^camT_target from vision

    For eye-to-hand, use inverted robot poses:
      ^gripperT_base = inverse(^baseT_gripper)

    Then pass:
      R_gripper2base := rotations of ^gripperT_base
      R_target2cam   := rotations of ^camT_target

    The returned result is interpreted here as ^baseT_cam.
    """
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for T_base_gripper, T_cam_target in zip(robot_base_T_gripper_list, cam_T_target_list):
        T_gripper_base = invert_T(T_base_gripper)

        R_gb, t_gb = T_to_R_t(T_gripper_base)
        R_ct, t_ct = T_to_R_t(T_cam_target)

        R_gripper2base.append(R_gb)
        t_gripper2base.append(t_gb)
        R_target2cam.append(R_ct)
        t_target2cam.append(t_ct)

    R_out, t_out = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method,
    )

    T_base_cam = make_T(R_out, t_out.reshape(3))
    return T_base_cam


# ----------------------------
# Realsense depth/color composition
# ----------------------------

def compose_base_T_depth_from_base_T_color(
    T_base_color: np.ndarray,
    rs_extrinsics_depth_to_color: dict,
) -> np.ndarray:
    """
    RealSense SDK gives depth->color extrinsics as rotation + translation.
    That is ^colorT_depth, so:

        ^baseT_depth = ^baseT_color @ ^colorT_depth
    """
    R = np.array(rs_extrinsics_depth_to_color["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.array(rs_extrinsics_depth_to_color["translation"], dtype=np.float64).reshape(3)
    T_color_depth = make_T(R, t)
    return T_base_color @ T_color_depth


# ----------------------------
# IO helpers
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
        assert T.shape == (4, 4)
        T_base_gripper_list.append(T)

    return image_names, T_base_gripper_list


# ----------------------------
# Main example
# ----------------------------

def run_example():
    # 1) Load color intrinsics
    K, D = load_intrinsics("color_intrinsics.json")

    # 2) Create ChArUco board
    # CHANGE THESE TO MATCH YOUR PRINTED BOARD
    squares_x = 8
    squares_y = 8
    square_length_m = 0.021
    marker_length_m = 0.016
    dictionary_name = "DICT_5X5_50"

    board, dictionary = create_charuco_board(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_length_m,
        marker_length_m=marker_length_m,
        dictionary_name=dictionary_name,
        use_legacy_pattern=True,
    )

    # 3) Load synchronized robot poses and image names
    image_names, T_base_gripper_list = load_robot_poses("robot_poses.json")

    cam_T_target_list = []
    valid_robot_poses = []

    for img_name, T_base_gripper in zip(image_names, T_base_gripper_list):
        image = cv2.imread(f"images/{img_name}")
        if image is None:
            print(f"Skipping {img_name}: image not found")
            continue

        T_cam_target = detect_charuco_pose(
            image_bgr=image,
            camera_matrix=K,
            dist_coeffs=D,
            board=board,
            dictionary=dictionary,
            min_charuco_corners=6,
            debug=True,
        )
        if T_cam_target is None:
            print(f"Skipping {img_name}: ChArUco board not found")
            continue

        cam_T_target_list.append(T_cam_target)
        valid_robot_poses.append(T_base_gripper)

    if len(valid_robot_poses) < 10:
        raise RuntimeError(
            f"Too few valid calibration samples ({len(valid_robot_poses)}). "
            "Collect more poses with better board visibility."
        )

    # 4) Calibrate fixed camera to base
    T_base_color = calibrate_fixed_camera_to_robot_base(
        robot_base_T_gripper_list=valid_robot_poses,
        cam_T_target_list=cam_T_target_list,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    print("T_base_color =")
    print(T_base_color)

    # 5) Compose to depth frame if needed
    rs_depth_to_color = {
        # Replace with values from pyrealsense2 depth_stream.get_extrinsics_to(color_stream)
        "rotation": [1, 0, 0,
                     0, 1, 0,
                     0, 0, 1],
        "translation": [0, 0, 0],
    }

    T_base_depth = compose_base_T_depth_from_base_T_color(
        T_base_color,
        rs_depth_to_color,
    )

    print("T_base_depth =")
    print(T_base_depth)

    np.save("T_base_color.npy", T_base_color)
    np.save("T_base_depth.npy", T_base_depth)


if __name__ == "__main__":
    run_example()