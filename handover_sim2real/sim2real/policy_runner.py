#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import roslibpy
import torch
from scipy.spatial.transform import Rotation as Rot
from torchvision import transforms

# -----------------------------------------------------------------------------
# Directory layout assumptions
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "h2r"
HANDOVER_RS_ROOT = PROJECT_ROOT / "handover-rs"
HANDOVER_SIM2REAL_ROOT = PROJECT_ROOT / "handover-sim2real"
HANDOVER_SIM_ROOT = HANDOVER_SIM2REAL_ROOT / "handover-sim"
HANDS_SEG_ROOT = PROJECT_ROOT / "hands-segmentation-pytorch"

DEFAULT_MODEL_DIR = (
    HANDOVER_SIM2REAL_ROOT
    / "output"
    / "cvpr2023_models"
    / "2022-10-14_03-01-32_finetune_1_s0_train"
)
DEFAULT_GRASP_DIR = HANDOVER_SIM2REAL_ROOT / "output" / "grasp_trigger_PRE_2"
DEFAULT_HAND_SEG_CKPT = HANDOVER_RS_ROOT / "checkpoint" / "checkpoint.ckpt"

# Sanity checks
if not HANDOVER_RS_ROOT.exists():
    raise FileNotFoundError(f"handover-rs not found: {HANDOVER_RS_ROOT}")
if not HANDOVER_SIM2REAL_ROOT.exists():
    raise FileNotFoundError(f"handover-sim2real not found: {HANDOVER_SIM2REAL_ROOT}")
if not HANDOVER_SIM_ROOT.exists():
    raise FileNotFoundError(f"handover-sim not found: {HANDOVER_SIM_ROOT}")
if not (HANDOVER_SIM_ROOT / "handover").exists():
    raise FileNotFoundError(
        f"'handover' package folder not found under: {HANDOVER_SIM_ROOT}"
    )

# Put local project dirs on sys.path
sys.path.insert(0, str(HANDOVER_RS_ROOT))
sys.path.insert(0, str(HANDS_SEG_ROOT))
sys.path.insert(0, str(HANDOVER_SIM2REAL_ROOT))
sys.path.insert(0, str(HANDOVER_SIM_ROOT))

print("Using paths:")
print("  HANDOVER_RS_ROOT      =", HANDOVER_RS_ROOT)
print("  HANDOVER_SIM2REAL_ROOT=", HANDOVER_SIM2REAL_ROOT)
print("  HANDOVER_SIM_ROOT     =", HANDOVER_SIM_ROOT)

from camera import RealSenseCamera  # noqa: E402
from pointcloud_pipeline import (  # noqa: E402
    build_policy_point_tensor,
    extract_hand_object_clouds,
)
from transforms import transform_points  # noqa: E402
from model import HandSegModel  # noqa: E402

# handover-sim2real imports
from handover_sim2real.utils import add_sys_path_from_env  # noqa: E402

add_sys_path_from_env("GADDPG_DIR")

from handover_sim2real.config import get_cfg  # noqa: E402
from core.bc import BC  # noqa: E402
from core.ddpg import DDPG  # noqa: E402
from core.utils import make_nets_opts_schedulers, unpack_action  # noqa: E402
from env.panda_scene import PandaTaskSpace6D  # noqa: E402
from experiments.config import cfg_from_file  # noqa: E402

# -----------------------------------------------------------------------------
# ROS bridge defaults
# -----------------------------------------------------------------------------
ROSBRIDGE_HOST = "172.16.0.7"
ROSBRIDGE_PORT = 9090

CURRENT_POSE_TOPIC = "/cartesian_pose"
TARGET_POSE_TOPIC = "/equilibrium_pose"
POSE_MSG_TYPE = "geometry_msgs/PoseStamped"

# -----------------------------------------------------------------------------
# Safety / runtime knobs
# -----------------------------------------------------------------------------
MIN_TARGET_X_M = 0.0  # do not go behind base / toward rear wall
MIN_TARGET_Z_M = 0.0  # do not go below table plane

CAMERA_WIDTH = 424
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

CROP_RADIUS_M = 0.12
OBJECT_MAX_RADIUS_M = 0.10
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.50
FULL_CLOUD_STRIDE = 2
HAND_CLOUD_STRIDE = 1
MIN_HAND_POINTS = 100
MIN_OBJECT_POINTS = 80

NUM_HAND_POINTS = 128
NUM_OBJECT_POINTS = 896

# Slowdown / rate limiting
MAX_TRANS_STEP_M = 0.005      # 3 mm per control cycle
MAX_ROT_STEP_DEG = 5.0        # 1 degree per control cycle

# Fixed tripod-camera calibration.
# Interpret this as camera frame -> robot base frame.
CALIB_T_BASE_CAM = np.array([
    [-0.99655429,  0.01054811, -0.08226951,  0.58932334],
    [ 0.08208897,  0.26739797, -0.96008319,  0.51493328],
    [ 0.01187164, -0.96352844, -0.26734248,  0.92565771],
    [ 0.,          0.,          0.,          1.        ]
], dtype=np.float64)

GRASP_PRED_THRESHOLD = 0.9
GRASP_MODEL_SUFFIX = "epoch_20000"

PRINT_EVERY_SEC = 0.5


# -----------------------------------------------------------------------------
# ROS state callback
# -----------------------------------------------------------------------------
current_msg: Optional[dict] = None


def pose_cb(msg: dict) -> None:
    global current_msg
    current_msg = msg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    mask = largest_component(mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def pose_msg_to_matrix(msg: dict) -> np.ndarray:
    pose = msg["pose"]
    pos = np.array(
        [
            pose["position"]["x"],
            pose["position"]["y"],
            pose["position"]["z"],
        ],
        dtype=np.float64,
    )
    quat = np.array(
        [
            pose["orientation"]["x"],
            pose["orientation"]["y"],
            pose["orientation"]["z"],
            pose["orientation"]["w"],
        ],
        dtype=np.float64,
    )
    quat_norm = np.linalg.norm(quat)
    if quat_norm <= 1e-12:
        raise ValueError("Received zero-norm quaternion from /cartesian_pose")
    quat = quat / quat_norm

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rot.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T

def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def matrix_to_pose_msg_like(template_msg: dict, T: np.ndarray, seq: int) -> dict:
    target = copy.deepcopy(template_msg)

    now = time.time()
    secs = int(now)
    nsecs = int((now - secs) * 1e9)

    quat = Rot.from_matrix(T[:3, :3]).as_quat()

    target["header"]["seq"] = seq
    target["header"]["stamp"] = {"secs": secs, "nsecs": nsecs}

    target["pose"]["position"]["x"] = float(T[0, 3])
    target["pose"]["position"]["y"] = float(T[1, 3])
    target["pose"]["position"]["z"] = float(T[2, 3])

    target["pose"]["orientation"]["x"] = float(quat[0])
    target["pose"]["orientation"]["y"] = float(quat[1])
    target["pose"]["orientation"]["z"] = float(quat[2])
    target["pose"]["orientation"]["w"] = float(quat[3])

    return target


def clamp_target_pose(T: np.ndarray) -> np.ndarray:
    T = T.copy()
    T[0, 3] = max(float(T[0, 3]), MIN_TARGET_X_M)
    T[2, 3] = max(float(T[2, 3]), MIN_TARGET_Z_M)
    return T

def limit_target_pose_step(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    max_trans_step_m: float = MAX_TRANS_STEP_M,
    max_rot_step_deg: float = MAX_ROT_STEP_DEG,
) -> np.ndarray:
    """
    Limit how far the target pose is allowed to move away from the current pose
    in one control cycle.
    """
    limited = target_pose.copy()

    # ----- Translation limit -----
    cur_t = current_pose[:3, 3]
    tgt_t = target_pose[:3, 3]
    dt = tgt_t - cur_t
    dt_norm = np.linalg.norm(dt)

    if dt_norm > max_trans_step_m and dt_norm > 1e-9:
        dt = dt / dt_norm * max_trans_step_m

    limited[:3, 3] = cur_t + dt

    # ----- Rotation limit -----
    cur_R = Rot.from_matrix(current_pose[:3, :3])
    tgt_R = Rot.from_matrix(target_pose[:3, :3])

    rel_R = cur_R.inv() * tgt_R
    rotvec = rel_R.as_rotvec()
    angle = np.linalg.norm(rotvec)

    max_angle = np.deg2rad(max_rot_step_deg)

    if angle > max_angle and angle > 1e-9:
        rotvec = rotvec / angle * max_angle
        rel_R_limited = Rot.from_rotvec(rotvec)
        limited[:3, :3] = (cur_R * rel_R_limited).as_matrix()
    else:
        limited[:3, :3] = target_pose[:3, :3]

    return limited


def action_to_target_pose(current_ee_pose: np.ndarray, action: np.ndarray) -> np.ndarray:
    delta_ee_pose = unpack_action(action)
    target_ee_pose = current_ee_pose @ delta_ee_pose
    return target_ee_pose


def overlay_mask(color_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = color_bgr.copy()
    hand_color = np.zeros_like(color_bgr)
    hand_color[:, :, 1] = 255
    blended = (alpha * hand_color + (1.0 - alpha) * overlay).astype(np.uint8)
    overlay = np.where(mask[..., None] > 0, blended, overlay)
    return overlay


def load_hand_segmenter(device: str, checkpoint: Path) -> tuple[HandSegModel, transforms.Compose]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Hand segmentation checkpoint not found: {checkpoint}")

    model = HandSegModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, preprocess


def load_policy_agents(model_dir: Path, grasp_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"Main policy model dir not found: {model_dir}")
    if not grasp_dir.exists():
        raise FileNotFoundError(f"Grasp model dir not found: {grasp_dir}")

    model_cfg = get_cfg()
    cfg_from_file(
        filename=str(model_dir / "config.yaml"),
        dict=model_cfg,
        reset_model_spec=False,
        merge_to_cn_dict=True,
    )

    agent = DDPG(model_cfg.RL_TRAIN.feature_input_dim, PandaTaskSpace6D(), model_cfg.RL_TRAIN)
    net_dict = make_nets_opts_schedulers(model_cfg.RL_MODEL_SPEC, model_cfg.RL_TRAIN)
    agent.setup_feature_extractor(net_dict)
    agent.load_model(str(model_dir))

    grasp_cfg = cfg_from_file(filename=str(grasp_dir / "config.yaml"), no_merge=True)
    grasp_agent = BC(grasp_cfg.RL_TRAIN.feature_input_dim, PandaTaskSpace6D(), grasp_cfg.RL_TRAIN)
    grasp_net_dict = make_nets_opts_schedulers(grasp_cfg.RL_MODEL_SPEC, grasp_cfg.RL_TRAIN)
    grasp_agent.setup_feature_extractor(grasp_net_dict)
    grasp_agent.load_model(str(grasp_dir), surfix=GRASP_MODEL_SUFFIX)

    return model_cfg, agent, grasp_agent


def warm_up_agents(agent, grasp_agent) -> None:
    dummy_state = [
        (np.zeros((5, 1024), dtype=np.float32), np.array([], dtype=np.float32)),
        None,
        None,
        None,
    ]
    agent.select_action(dummy_state)
    grasp_agent.select_action_grasp(dummy_state, GRASP_PRED_THRESHOLD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HandoverSim2Real policy on physical robot via rosbridge.")
    parser.add_argument("--rosbridge-host", type=str, default=ROSBRIDGE_HOST)
    parser.add_argument("--rosbridge-port", type=int, default=ROSBRIDGE_PORT)
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--grasp-dir", type=str, default=str(DEFAULT_GRASP_DIR))
    parser.add_argument("--hand-seg-ckpt", type=str, default=str(DEFAULT_HAND_SEG_CKPT))
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_dir = Path(args.model_dir).expanduser().resolve()
    grasp_dir = Path(args.grasp_dir).expanduser().resolve()
    hand_seg_ckpt = Path(args.hand_seg_ckpt).expanduser().resolve()

    print(f"Main policy dir: {model_dir}")
    print(f"Grasp policy dir: {grasp_dir}")
    print(f"Hand seg ckpt: {hand_seg_ckpt}")

    hand_seg_model, hand_seg_preprocess = load_hand_segmenter(device, hand_seg_ckpt)

    os.chdir(HANDOVER_SIM2REAL_ROOT)
    print(f"Changed working directory to: {Path.cwd()}")

    model_cfg, agent, grasp_agent = load_policy_agents(model_dir, grasp_dir)
    warm_up_agents(agent, grasp_agent)

    client = roslibpy.Ros(host=args.rosbridge_host, port=args.rosbridge_port)
    client.run()

    for _ in range(50):
        if client.is_connected:
            break
        time.sleep(0.1)

    if not client.is_connected:
        raise RuntimeError(
            f"Could not connect to rosbridge at {args.rosbridge_host}:{args.rosbridge_port}"
        )

    sub = roslibpy.Topic(client, CURRENT_POSE_TOPIC, POSE_MSG_TYPE)
    pub = roslibpy.Topic(client, TARGET_POSE_TOPIC, POSE_MSG_TYPE)

    sub.subscribe(pose_cb)
    pub.advertise()

    print(f"Connected to rosbridge at {args.rosbridge_host}:{args.rosbridge_port}")
    print(f"Subscribed to {CURRENT_POSE_TOPIC}")
    print(f"Publishing to {TARGET_POSE_TOPIC}")
    print("Waiting for current pose...")

    timeout_s = 10.0
    t0 = time.time()
    while current_msg is None and (time.time() - t0) < timeout_s:
        time.sleep(0.05)

    if current_msg is None:
        sub.unsubscribe()
        pub.unadvertise()
        client.terminate()
        raise RuntimeError(f"No message received on {CURRENT_POSE_TOPIC} through rosbridge")

    cam = RealSenseCamera(
        color_size=(args.camera_width, args.camera_height),
        depth_size=(args.camera_width, args.camera_height),
        fps=args.camera_fps,
    )
    cam.start()

    print("Runner started. Press 'q' or Esc in the image window to quit.")

    last_hand_xyz = None
    last_object_xyz = None
    publish_seq = 0
    last_print_time = 0.0

    try:
        while True:
            if current_msg is None:
                time.sleep(0.01)
                continue

            color_bgr, depth_m, _ = cam.get_frames()
            current_pose_msg = copy.deepcopy(current_msg)
            current_ee_pose = pose_msg_to_matrix(current_pose_msg)

            # -----------------------------------------------------------------
            # Hand segmentation
            # -----------------------------------------------------------------
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            x = hand_seg_preprocess(color_rgb).unsqueeze(0).to(device, non_blocking=True)

            with torch.inference_mode():
                logits = hand_seg_model(x)
                pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            hand_mask = cv2.resize(
                pred,
                (color_bgr.shape[1], color_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            hand_mask = normalize_mask(hand_mask)

            # -----------------------------------------------------------------
            # Point cloud extraction
            # -----------------------------------------------------------------
            result = extract_hand_object_clouds(
                color_bgr=color_bgr,
                depth_m=depth_m,
                hand_mask=hand_mask,
                cam=cam,
                last_hand_xyz=last_hand_xyz,
                last_object_xyz=last_object_xyz,
                crop_radius_m=CROP_RADIUS_M,
                object_max_radius_m=OBJECT_MAX_RADIUS_M,
                min_depth_m=MIN_DEPTH_M,
                max_depth_m=MAX_DEPTH_M,
                full_cloud_stride=FULL_CLOUD_STRIDE,
                hand_cloud_stride=HAND_CLOUD_STRIDE,
                min_hand_points=MIN_HAND_POINTS,
                min_object_points=MIN_OBJECT_POINTS,
            )

            hand_xyz = result.hand_xyz
            object_xyz = result.object_xyz

            if len(hand_xyz) > 0:
                last_hand_xyz = hand_xyz
            if len(object_xyz) > 0:
                last_object_xyz = object_xyz

            # -----------------------------------------------------------------
            # Tripod camera frame -> current EE/policy frame
            # current_ee_pose is T_base_ee
            # CALIB_T_BASE_CAM is T_base_cam
            # Therefore:
            #   T_ee_cam = T_ee_base @ T_base_cam
            # -----------------------------------------------------------------
            T_ee_base = invert_transform(current_ee_pose)
            T_ee_cam = T_ee_base @ CALIB_T_BASE_CAM

            hand_policy = transform_points(T_ee_cam.astype(np.float32), hand_xyz)
            object_policy = transform_points(T_ee_cam.astype(np.float32), object_xyz)

            have_valid_obs = (len(hand_policy) > 0) and (len(object_policy) > 0)

            if have_valid_obs:
                point_state = build_policy_point_tensor(
                    hand_policy,
                    object_policy,
                    num_hand_points=NUM_HAND_POINTS,
                    num_object_points=NUM_OBJECT_POINTS,
                ).T.astype(np.float32)

                state = [
                    (point_state, np.array([], dtype=np.float32)),
                    None,
                    None,
                    None,
                ]

                action, _, _, _ = agent.select_action(state)
                grasp_pred = bool(np.squeeze(grasp_agent.select_action_grasp(state, GRASP_PRED_THRESHOLD)))

                target_ee_pose = action_to_target_pose(current_ee_pose, action)
                target_ee_pose = limit_target_pose_step(current_ee_pose, target_ee_pose)
                target_ee_pose = clamp_target_pose(target_ee_pose)
            else:
                # No segmented point cloud perceived -> hold current pose
                grasp_pred = False
                target_ee_pose = current_ee_pose.copy()

            target_msg = matrix_to_pose_msg_like(current_pose_msg, target_ee_pose, publish_seq)
            publish_seq += 1
            pub.publish(roslibpy.Message(target_msg))

            overlay = overlay_mask(color_bgr, hand_mask)
            dbg = result.debug
            status = (
                f"hand={len(hand_policy)} obj={len(object_policy)} "
                f"fallback_h={dbg['used_last_hand']} fallback_o={dbg['used_last_object']} "
                f"grasp_pred={grasp_pred}"
            )
            cv2.putText(
                overlay,
                status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("policy_runner_overlay", overlay)
            cv2.imshow("policy_runner_hand_mask", (hand_mask * 255).astype(np.uint8))

            now = time.time()
            if now - last_print_time > PRINT_EVERY_SEC:
                pos = target_ee_pose[:3, 3]
                print(
                    f"target xyz=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) "
                    f"hand={len(hand_policy):4d} obj={len(object_policy):4d} "
                    f"grasp_pred={grasp_pred}",
                    flush=True,
                )
                last_print_time = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

        try:
            sub.unsubscribe()
        except Exception:
            pass
        try:
            pub.unadvertise()
        except Exception:
            pass
        try:
            client.terminate()
        except Exception:
            pass

        print("Stopped.")


if __name__ == "__main__":
    main()