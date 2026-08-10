#!/usr/bin/env python3
"""
Run the Phase-1/4 BC policy (DAgger run 12, checkpoint/cp2) on the physical FR3.

This is the sibling of policy_runner.py, which drives the CVPR2023 GA-DDPG
model. The two policies are NOT interchangeable — three things differ, and each
one silently produces garbage if carried over from the other script:

  1. POINT CLOUD ORDER/LABELS. cp2 was trained by PointListener with the object
     class first: 896 object points [x,y,z,1,0] then 128 hand points
     [x,y,z,0,1], 1024 total. policy_runner.py's build_policy_point_tensor emits
     the opposite (hand first, hand=[1,0]). See build_bc_point_tensor below.
  2. ACTION SCALING. GA-DDPG's select_action returns a task-space action that
     unpack_action rescales through PandaTaskSpace6D. cp2's targets came from
     train_env.convert_target_joint_position_to_action, which is a RAW SE(3)
     delta between FK poses — no scaling. unpack_action is still the right
     matrix builder (euler2mat 'sxyz' + translation), just applied to a delta
     that is already in metres/radians.
  3. FRAME. The cloud and the delta both live in the panda_hand frame (link 8 of
     panda_gripper_hand_camera.urdf) — the hand mounting flange, NOT the
     fingertip TCP. See --ee-offset-z.

Observation contract (verified against output/bc_dataset/train_pinned_omg_ok.h5
attrs and handover_sim2real/policy.py::PointListener):

    point_cloud  [1024, 5]  xyz + ycb_flag + hand_flag, in the panda_hand frame
    robot_state  [32]       joint_pos(9)+joint_vel(9)+ee_xyz(3)+ee_wxyz(4)
                            +gripper_norm(1)+prev_act(6). The EE pose is in the
                            SIM WORLD frame, not the panda base frame — see
                            T_SIMWORLD_BASE. Run 12 sets drop_joint_state=true
                            and use_prev_act=false, so only rs[18:26] reaches
                            the network — see build_robot_state.
    action       [7]        dpos(3)+deuler(3) in the panda_hand frame, plus a
                            gripper bit where 1 = stay open, 0 = CLOSE.

Usage:
    # look but don't touch: perception + policy, publishes nothing
    python my_policy_runner.py --dry-run

    # safe bring-up: home first, then one step per SPACE press
    python my_policy_runner.py --home --step-mode

    # full closed loop, gripper live
    python my_policy_runner.py --home --enable-gripper
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import roslibpy
import torch
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp
from torchvision import transforms

# -----------------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------------
SIM2REAL_DIR = Path(__file__).resolve().parent
HANDOVER_SIM2REAL_ROOT = SIM2REAL_DIR.parents[1]
PROJECT_ROOT = HANDOVER_SIM2REAL_ROOT.parent
HANDS_SEG_ROOT = PROJECT_ROOT / "hands-segmentation-pytorch"

DEFAULT_POLICY_DIR = SIM2REAL_DIR / "checkpoint" / "cp2"
DEFAULT_HAND_SEG_CKPT = SIM2REAL_DIR / "checkpoint" / "cp1" / "checkpoint.ckpt"

if not HANDS_SEG_ROOT.exists():
    raise FileNotFoundError(
        f"hands-segmentation-pytorch not found: {HANDS_SEG_ROOT}\n"
        "It must sit alongside handover-sim2real (it provides HandSegModel)."
    )

sys.path.insert(0, str(HANDS_SEG_ROOT))
sys.path.insert(0, str(HANDOVER_SIM2REAL_ROOT))
sys.path.insert(0, str(SIM2REAL_DIR))

from camera import RealSenseCamera  # noqa: E402
from pointcloud_pipeline import (  # noqa: E402
    _sample_or_pad_points,
    extract_hand_object_clouds,
)
from transforms import invert_transform, transform_points  # noqa: E402
from model import HandSegModel  # noqa: E402

from handover_sim2real.utils import add_sys_path_from_env  # noqa: E402

# BCPolicy's PointNet++ backbone comes from GA-DDPG, so $GADDPG_DIR has to be on
# sys.path before handover_sim2real.bc imports it. add_sys_path_from_env asserts
# the variable is set; on the robot PC the checkout is always the sibling
# GA-DDPG/ directory, so default to it rather than making every invocation
# export it. An explicit GADDPG_DIR still wins.
if "GADDPG_DIR" not in os.environ:
    _gaddpg = HANDOVER_SIM2REAL_ROOT / "GA-DDPG"
    if not (_gaddpg / "core").is_dir():
        raise FileNotFoundError(
            f"GADDPG_DIR is not set and no GA-DDPG checkout at {_gaddpg}. "
            "The policy's PointNet++ backbone lives there.")
    os.environ["GADDPG_DIR"] = str(_gaddpg)
add_sys_path_from_env("GADDPG_DIR")

# NOT handover_sim2real.dagger.policy_io: its package __init__ imports
# .env_setup, which imports the `handover` simulator. The robot PC has no reason
# to carry the sim, so load_bc_policy below mirrors policy_io.build_policy's
# field reads instead. Keep the two in sync if MODEL ever grows a field.
from handover_sim2real.bc import BCPolicy, Normalizer  # noqa: E402
from core.utils import unpack_action  # noqa: E402

# -----------------------------------------------------------------------------
# ROS bridge
# -----------------------------------------------------------------------------
ROSBRIDGE_HOST = "172.16.0.7"
ROSBRIDGE_PORT = 9090

CURRENT_POSE_TOPIC = "/cartesian_pose"
TARGET_POSE_TOPIC = "/equilibrium_pose"
POSE_MSG_TYPE = "geometry_msgs/PoseStamped"

# franka_ros gripper. grasp/move are actionlib actions, but roslibpy 2.0 dropped
# its actionlib client, so we publish the goal messages onto the action's goal
# topic directly — actionlib is plain topics underneath, and a fire-and-forget
# goal is all this needs. joint_states carries the two finger positions in
# metres (0 .. 0.04 each).
GRIPPER_GRASP_GOAL_TOPIC = "/franka_gripper/grasp/goal"
GRIPPER_GRASP_GOAL_TYPE = "franka_gripper/GraspActionGoal"
GRIPPER_MOVE_GOAL_TOPIC = "/franka_gripper/move/goal"
GRIPPER_MOVE_GOAL_TYPE = "franka_gripper/MoveActionGoal"
GRIPPER_STATE_TOPIC = "/franka_gripper/joint_states"
GRIPPER_STATE_TYPE = "sensor_msgs/JointState"

GRIPPER_MAX_FINGER_M = 0.04   # sim: gripper_norm = joint_pos[7] / 0.04
GRASP_WIDTH_M = 0.0           # close all the way; epsilon/force do the work
GRASP_SPEED = 0.05
GRASP_FORCE = 20.0
GRASP_EPSILON_INNER = 0.04
GRASP_EPSILON_OUTER = 0.04

# -----------------------------------------------------------------------------
# Frames
# -----------------------------------------------------------------------------
# Sim wrist camera, from panda_gripper_hand_camera.urdf + Panda._t3d_hand_to_camera
# (handover-sim/handover/panda.py:110): the pinhole camera frame sits at
# (0.036, 0, 0.036) in panda_hand with a +90 deg rotation about z. Both the sim's
# deprojection and RealSenseCamera.depth_to_pointcloud use the OpenCV convention
# (x right, y down, z forward), so this matrix maps one onto the other directly.
#
# WARNING: this is the SIM's nominal mount, not a calibration of your D435. Every
# point the policy sees is biased by however far your real mount deviates. Pass a
# measured hand-eye matrix with --hand-eye <T_hand_cam.npy> as soon as you have one.
T_HAND_CAM_NOMINAL = np.array([
    [0.0, -1.0, 0.0, 0.036],
    [1.0,  0.0, 0.0, 0.000],
    [0.0,  0.0, 1.0, 0.036],
    [0.0,  0.0, 0.0, 1.000],
], dtype=np.float64)

# robot_state[18:25] is the EE pose in the SIM WORLD frame, not the panda base
# frame. collect_bc_dataset._robot_state reads body.link_state raw, while only
# _point_cloud applies panda_base_inv_tf — so the cloud is base/EE-relative but
# the state vector is not. The sim stands the panda on a table at
# ENV.PANDA_BASE_POSITION=(0.61,-0.50,0.875) with ENV.PANDA_BASE_ORIENTATION =
# a +90 deg yaw, so a real base-frame pose has to be mapped through this before
# the policy sees it. Skipping it puts ee_z ~1.1 m and the yaw 90 deg outside
# the training distribution (normalization.npz has ee mean [0.614,-0.092,1.475],
# std [0.118,0.160,0.123]) — the network still emits a confident action, it is
# just answering a question about a robot standing somewhere else.
T_SIMWORLD_BASE = np.array([
    [0.0, -1.0, 0.0,  0.610],
    [1.0,  0.0, 0.0, -0.500],
    [0.0,  0.0, 1.0,  0.875],
    [0.0,  0.0, 0.0,  1.000],
], dtype=np.float64)

# ENV.PANDA_INITIAL_POSITION, the pose every training episode starts from. Given
# here as the panda_hand pose in the BASE frame (pybullet FK on
# panda_gripper_hand_camera.urdf at that joint config), because /equilibrium_pose
# is the only interface this script has. The joint config itself is
#   (0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785) + fingers at 0.04
# which is inside the FR3 joint limits — use it directly if you have a
# joint-space controller, which is the safer way to home.
T_BASE_HAND_HOME = np.array([
    [0.87758249,  0.00034942,  0.47942555,  0.14609343],
    [0.00039816, -0.99999992,  0.0,         0.0       ],
    [0.47942551,  0.00019089, -0.87758256,  0.70596832],
    [0.0,         0.0,         0.0,         1.0       ],
], dtype=np.float64)

HOME_JOINTS = (0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785)

# Homing is an interpolated Cartesian move, not a jump: the controller gets a
# sequence of waypoints this far apart so the arm sweeps a predictable path.
HOME_STEP_TRANS_M = 0.02
HOME_STEP_ROT_DEG = 5.0
HOME_SETTLE_TIMEOUT_S = 3.0

# z offset from panda_hand to whatever frame /cartesian_pose publishes.
#
# MEASURED ON THIS ROBOT AS ZERO. /cartesian_pose matches O_T_EE to 4e-6 m, and
# franka_states reports F_T_EE = translation (0,0,0) with a -45 deg z rotation —
# NOT the Franka Hand default of (0,0,0.1034) with that rotation. Flange + Rz(-45)
# is exactly how panda_gripper_hand_camera.urdf defines panda_hand, so this
# controller already publishes the policy's frame. Confirmed independently:
# pybullet FK of panda_hand at the robot's reported q lands within 0.06 mm of
# /cartesian_pose.
#
# Set --ee-offset-z 0.1034 if you ever reconfigure the EE to the fingertip TCP
# (F_T_EE translation becomes 0.1034); leaving it wrong is a silent, constant
# 10 cm error in both the observation and every commanded target.
DEFAULT_EE_OFFSET_Z = 0.0

# -----------------------------------------------------------------------------
# Perception (same pipeline as policy_runner.py)
# -----------------------------------------------------------------------------
# Color and depth are requested as separate streams and only then aligned, so
# they need not match — but each must be a mode the device actually offers.
# policy_runner.py asks for 424x240 on both; the D435 offers that for COLOR only
# (its depth modes start at 256x144 / 480x270), so that pair fails device-side
# with a bare "Couldn't resolve requests". 640x480 is valid for both.
# depth_to_pointcloud validates against the COLOR intrinsics because align()
# resamples depth to the color grid, so the depth resolution set here only
# governs what the sensor streams.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
CAMERA_FPS = 30

CROP_RADIUS_M = 0.12
OBJECT_MAX_RADIUS_M = 0.10
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.50
FULL_CLOUD_STRIDE = 2
HAND_CLOUD_STRIDE = 1
MIN_HAND_POINTS = 100
MIN_OBJECT_POINTS = 80

# Run 12's split: POINT_STATE_YCB_RATIO = 0.875 of uniform_num_pts = 1024.
NUM_OBJECT_POINTS = 896
NUM_HAND_POINTS = 128

# -----------------------------------------------------------------------------
# Safety
# -----------------------------------------------------------------------------
MIN_TARGET_X_M = 0.0    # never behind the base
MIN_TARGET_Z_M = 0.0    # never below the table plane

# Per-policy-step ceiling. cp2's action_std is ~0.016-0.021 m and ~0.06-0.08 rad,
# so a well-behaved step is well under these; they exist to catch a blown
# prediction, not to shape normal motion.
MAX_STEP_TRANS_M = 0.05
MAX_STEP_ROT_DEG = 20.0

# Step-and-settle convergence.
SETTLE_POS_TOL_M = 0.005
SETTLE_ROT_TOL_DEG = 3.0
SETTLE_TIMEOUT_S = 2.0
SETTLE_POLL_S = 0.02

MAX_POLICY_STEPS = 50   # dagger/evaluator.py EvalParams.max_steps

# -----------------------------------------------------------------------------
# ROS state
# -----------------------------------------------------------------------------
current_msg: Optional[dict] = None
gripper_finger_m: Optional[float] = None


def pose_cb(msg: dict) -> None:
    global current_msg
    current_msg = msg


def gripper_state_cb(msg: dict) -> None:
    """Mean finger position in metres, 0 (closed) .. 0.04 (open)."""
    global gripper_finger_m
    pos = msg.get("position") or []
    if len(pos) >= 2:
        gripper_finger_m = 0.5 * (float(pos[0]) + float(pos[1]))
    elif len(pos) == 1:
        gripper_finger_m = float(pos[0])


# -----------------------------------------------------------------------------
# Mask helpers
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


def overlay_mask(color_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = color_bgr.copy()
    hand_color = np.zeros_like(color_bgr)
    hand_color[:, :, 1] = 255
    blended = (alpha * hand_color + (1.0 - alpha) * overlay).astype(np.uint8)
    return np.where(mask[..., None] > 0, blended, overlay)


# -----------------------------------------------------------------------------
# Pose helpers
# -----------------------------------------------------------------------------
def pose_msg_to_matrix(msg: dict) -> np.ndarray:
    pose = msg["pose"]
    pos = np.array([pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]],
                   dtype=np.float64)
    quat = np.array([pose["orientation"]["x"], pose["orientation"]["y"],
                     pose["orientation"]["z"], pose["orientation"]["w"]], dtype=np.float64)
    n = np.linalg.norm(quat)
    if n <= 1e-12:
        raise ValueError(f"Zero-norm quaternion on {CURRENT_POSE_TOPIC}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rot.from_quat(quat / n).as_matrix()
    T[:3, 3] = pos
    return T


def matrix_to_pose_msg_like(template_msg: dict, T: np.ndarray, seq: int) -> dict:
    target = copy.deepcopy(template_msg)

    now = time.time()
    secs = int(now)
    quat = Rot.from_matrix(T[:3, :3]).as_quat()

    target["header"]["seq"] = seq
    target["header"]["stamp"] = {"secs": secs, "nsecs": int((now - secs) * 1e9)}

    target["pose"]["position"]["x"] = float(T[0, 3])
    target["pose"]["position"]["y"] = float(T[1, 3])
    target["pose"]["position"]["z"] = float(T[2, 3])

    target["pose"]["orientation"]["x"] = float(quat[0])
    target["pose"]["orientation"]["y"] = float(quat[1])
    target["pose"]["orientation"]["z"] = float(quat[2])
    target["pose"]["orientation"]["w"] = float(quat[3])
    return target


def z_offset_transform(dz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = dz
    return T


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(translation error in m, rotation error in rad) between two 4x4 poses."""
    dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rel = Rot.from_matrix(a[:3, :3]).inv() * Rot.from_matrix(b[:3, :3])
    return dt, float(np.linalg.norm(rel.as_rotvec()))


def clamp_action_delta(action6: np.ndarray) -> tuple[np.ndarray, bool]:
    """Cap one policy step's translation and rotation. Returns (action, clamped)."""
    a = np.asarray(action6, dtype=np.float64).copy()
    clamped = False

    t_norm = float(np.linalg.norm(a[:3]))
    if t_norm > MAX_STEP_TRANS_M and t_norm > 1e-9:
        a[:3] *= MAX_STEP_TRANS_M / t_norm
        clamped = True

    # The delta's euler angles are 'sxyz' (transforms3d, via unpack_action), so
    # scale in rotvec space rather than scaling the angles directly.
    rotvec = Rot.from_matrix(unpack_action(a)[:3, :3]).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    max_angle = np.deg2rad(MAX_STEP_ROT_DEG)
    if angle > max_angle and angle > 1e-9:
        capped = Rot.from_rotvec(rotvec / angle * max_angle).as_matrix()
        a[3:6] = Rot.from_matrix(capped).as_euler("xyz")
        clamped = True

    return a, clamped


def clamp_target_pose(T: np.ndarray) -> np.ndarray:
    T = T.copy()
    T[0, 3] = max(float(T[0, 3]), MIN_TARGET_X_M)
    T[2, 3] = max(float(T[2, 3]), MIN_TARGET_Z_M)
    return T


# -----------------------------------------------------------------------------
# Observation assembly
# -----------------------------------------------------------------------------
def build_bc_point_tensor(object_xyz: np.ndarray, hand_xyz: np.ndarray) -> np.ndarray:
    """[1024, 5] in cp2's layout: 896 object then 128 hand.

    Mirrors PointListener._process_pointcloud + point_states_to_state, where the
    class index i sets channel 3+i and the merge order follows the env's class
    order (object first). Channel 3 = ycb_flag, channel 4 = hand_flag.
    """
    obj_fixed = _sample_or_pad_points(object_xyz, NUM_OBJECT_POINTS)
    hand_fixed = _sample_or_pad_points(hand_xyz, NUM_HAND_POINTS)

    obj_labels = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (NUM_OBJECT_POINTS, 1))
    hand_labels = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (NUM_HAND_POINTS, 1))

    return np.concatenate(
        [np.concatenate([obj_fixed, obj_labels], axis=1),
         np.concatenate([hand_fixed, hand_labels], axis=1)],
        axis=0,
    ).astype(np.float32)


def build_robot_state(T_base_hand: np.ndarray, gripper_norm: float) -> np.ndarray:
    """The 32-D vector, with only the channels run 12 actually reads filled in.

    Takes the panda_hand pose in the REAL BASE frame and re-expresses it in the
    sim world frame, which is what the dataset stored (see T_SIMWORLD_BASE).

    drop_joint_state=true + use_prev_act=false means BCPolicy._select_robot_state
    keeps rs[18:26] and nothing else, so joint_pos/joint_vel (0:18) and prev_act
    (26:32) are left at zero deliberately — they are sliced away before the
    encoder, and the normalizer is elementwise, so their value cannot reach the
    network. _assert_state_layout() enforces that those two flags really are set.
    """
    T_simworld_hand = T_SIMWORLD_BASE @ T_base_hand

    rs = np.zeros(32, dtype=np.float32)
    rs[18:21] = T_simworld_hand[:3, 3]
    quat_xyzw = Rot.from_matrix(T_simworld_hand[:3, :3]).as_quat()
    rs[21] = quat_xyzw[3]           # w first: the dataset stores ee_wxyz
    rs[22:25] = quat_xyzw[:3]
    rs[25] = np.clip(gripper_norm, 0.0, 1.0)
    return rs


def _assert_state_layout(run_cfg: dict) -> None:
    m = run_cfg["MODEL"]
    if not bool(m.get("drop_joint_state", False)) or bool(m.get("use_prev_act", True)):
        raise ValueError(
            "This runner fills only robot_state[18:26] (ee pose + gripper). The "
            "checkpoint's config has drop_joint_state="
            f"{m.get('drop_joint_state')} / use_prev_act={m.get('use_prev_act')}, "
            "so it also reads joint state and/or prev_action — which the real "
            "robot does not provide here. Feeding zeros would be silently wrong."
        )


# -----------------------------------------------------------------------------
# Gripper
# -----------------------------------------------------------------------------
def _action_goal_msg(goal_id: str, goal: dict) -> dict:
    """Wrap an actionlib goal payload in its ActionGoal envelope."""
    now = time.time()
    secs = int(now)
    stamp = {"secs": secs, "nsecs": int((now - secs) * 1e9)}
    return {
        "header": {"seq": 0, "stamp": stamp, "frame_id": ""},
        "goal_id": {"stamp": stamp, "id": goal_id},
        "goal": goal,
    }


class FrankaGripper:
    """franka_ros gripper over rosbridge. A no-op when disabled."""

    def __init__(self, client: Optional[roslibpy.Ros], enabled: bool):
        self.enabled = bool(enabled and client is not None)
        self._grasp = self._move = None
        self._seq = 0
        if self.enabled:
            self._grasp = roslibpy.Topic(
                client, GRIPPER_GRASP_GOAL_TOPIC, GRIPPER_GRASP_GOAL_TYPE)
            self._move = roslibpy.Topic(
                client, GRIPPER_MOVE_GOAL_TOPIC, GRIPPER_MOVE_GOAL_TYPE)
            self._grasp.advertise()
            self._move.advertise()

    def close(self) -> None:
        if not self.enabled:
            print("[gripper] CLOSE commanded (disabled — pass --enable-gripper to act)")
            return
        self._seq += 1
        self._grasp.publish(roslibpy.Message(_action_goal_msg(
            f"my_policy_runner_grasp_{self._seq}",
            {
                "width": GRASP_WIDTH_M,
                "epsilon": {"inner": GRASP_EPSILON_INNER,
                            "outer": GRASP_EPSILON_OUTER},
                "speed": GRASP_SPEED,
                "force": GRASP_FORCE,
            })))
        print(f"[gripper] grasp goal sent (width={GRASP_WIDTH_M} force={GRASP_FORCE}N)")

    def open(self) -> None:
        if not self.enabled:
            return
        self._seq += 1
        self._move.publish(roslibpy.Message(_action_goal_msg(
            f"my_policy_runner_move_{self._seq}",
            {"width": 2 * GRIPPER_MAX_FINGER_M, "speed": GRASP_SPEED})))

    def shutdown(self) -> None:
        for topic in (self._grasp, self._move):
            if topic is not None:
                try:
                    topic.unadvertise()
                except Exception:
                    pass


def read_gripper_norm(assume_open: bool) -> float:
    """Normalized finger position for robot_state[25]: 1 = open, 0 = closed.

    Falls back to 1.0 when /franka_gripper/joint_states is not being published —
    the policy only ever sees an open gripper during approach anyway, since the
    episode ends at the close.
    """
    if gripper_finger_m is None:
        return 1.0 if assume_open else 0.0
    return float(np.clip(gripper_finger_m / GRIPPER_MAX_FINGER_M, 0.0, 1.0))


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_hand_segmenter(device: str, checkpoint: Path):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Hand segmentation checkpoint not found: {checkpoint}")

    model = HandSegModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def load_bc_policy(policy_dir: Path, ckpt: str, device: str):
    """Load the run dir's BCPolicy. Returns (model, run_cfg).

    Mirrors handover_sim2real.dagger.policy_io.{read_run_cfg,build_policy,
    load_policy_runner} for the "bc" branch, minus the simulator dependency.
    The load is strict, so any config/checkpoint mismatch raises here rather
    than producing a subtly wrong policy.
    """
    run_cfg = yaml.safe_load((policy_dir / "config.yaml").read_text())
    if "chunk_len" in run_cfg["MODEL"] or "history_len" in run_cfg["MODEL"]:
        raise ValueError(
            f"{policy_dir} describes an ACT policy; this runner drives the "
            "single-frame BC policy only (no history buffer or chunk execution).")

    norm_path = policy_dir / "normalization.npz"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"{norm_path} is missing — the policy's action/state scaling is part "
            "of its definition; rolling out without it produces garbage.")

    m, d = run_cfg["MODEL"], run_cfg["DATA"]
    model = BCPolicy(
        pc_channels=int(d["pc_channels"]),
        robot_state_dim=int(d["robot_state_dim"]),
        action_dim=int(d["action_dim"]),
        feature_dim=int(m["feature_dim"]),
        robot_hidden=int(m["robot_hidden"]),
        policy_hidden=tuple(m["policy_hidden"]),
        pointnet_scale=int(m["pointnet_scale"]),
        pointnet_radius=float(m["pointnet_radius"]),
        pointnet_nclusters=int(m["pointnet_nclusters"]),
        use_prev_act=bool(m.get("use_prev_act", True)),
        drop_joint_state=bool(m.get("drop_joint_state", False)),
        joint_state_dim=int(m.get("joint_state_dim", 18)),
        freeze_pc=bool(m.get("freeze_pc", False)),
        aux_head=bool(m.get("aux_head", False)),
        aux_dim=int(m.get("aux_dim", 7)),
        aux_hidden=tuple(m.get("aux_hidden", (256, 256))),
        normalizer=Normalizer.load(str(norm_path)),
    ).to(device)

    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"[policy] {ckpt} (epoch {payload.get('epoch', '?')})  bc (single frame)")
    return model, run_cfg


@torch.no_grad()
def policy_act(model, pc: np.ndarray, rs: np.ndarray, device: str) -> np.ndarray:
    """[7] deployable action: BCPolicy.predict denormalizes ch0..5 and
    hard-thresholds the gripper logit to {0, 1}. Same call BCRunner.act makes."""
    pc_t = torch.from_numpy(pc).float().unsqueeze(0).to(device)
    rs_t = torch.from_numpy(rs).float().unsqueeze(0).to(device)
    return model.predict(pc_t, rs_t)[0].cpu().numpy().astype(np.float32)


def warm_up_policy(model, device: str) -> None:
    """One dummy forward so the first real step isn't paying CUDA init."""
    policy_act(model,
               np.zeros((NUM_OBJECT_POINTS + NUM_HAND_POINTS, 5), dtype=np.float32),
               np.zeros(32, dtype=np.float32), device)


# -----------------------------------------------------------------------------
# Control
# -----------------------------------------------------------------------------
def settle(pub, template_msg: dict, T_base_ctrl_target: np.ndarray, seq: int,
           timeout_s: float) -> tuple[bool, float, float]:
    """Publish a target and block until the EE reaches it.

    cp2 is a single-frame Markov policy trained on ~2 cm steps where the sim
    robot fully reached each waypoint before the next observation. Streaming
    targets at camera rate would feed it mid-motion states it never saw in
    training, so we hold each target until convergence (or timeout — an
    impedance controller has steady-state error and may never hit tolerance).

    Returns (converged, final_pos_err, final_rot_err).
    """
    pub.publish(roslibpy.Message(
        matrix_to_pose_msg_like(template_msg, T_base_ctrl_target, seq)))

    rot_tol = np.deg2rad(SETTLE_ROT_TOL_DEG)
    t0 = time.time()
    dt = drot = float("inf")

    while time.time() - t0 < timeout_s:
        time.sleep(SETTLE_POLL_S)
        if current_msg is None:
            continue
        T_now = pose_msg_to_matrix(current_msg)
        dt, drot = pose_error(T_now, T_base_ctrl_target)
        if dt < SETTLE_POS_TOL_M and drot < rot_tol:
            return True, dt, drot

    return False, dt, drot


def go_home(pub, T_ctrl_hand: np.ndarray, T_hand_ctrl: np.ndarray,
            seq: int) -> int:
    """Drive the arm to the sim's episode-start pose, in interpolated steps.

    Every training episode began at ENV.PANDA_INITIAL_POSITION, so the policy
    has only ever seen states downstream of it; starting anywhere else is
    already off-distribution on step 0.

    This is a CARTESIAN move because /equilibrium_pose is the only interface
    here, which means the elbow ends up wherever the controller's nullspace puts
    it rather than at the sim's joint config, and the straight-line path is not
    collision-checked. Keep the workspace clear. If you have a joint-space
    controller, commanding HOME_JOINTS is strictly better.

    Returns the next publish sequence number.
    """
    if current_msg is None:
        raise RuntimeError("No pose received; cannot home.")

    T_start = pose_msg_to_matrix(current_msg) @ T_ctrl_hand
    dist, ang = pose_error(T_start, T_BASE_HAND_HOME)
    n = max(int(np.ceil(max(dist / HOME_STEP_TRANS_M,
                           np.rad2deg(ang) / HOME_STEP_ROT_DEG))), 1)

    print(f"[home] {dist*100:.1f} cm / {np.rad2deg(ang):.1f} deg away — "
          f"{n} interpolated waypoints")

    key_rots = Rot.from_matrix(np.stack([T_start[:3, :3], T_BASE_HAND_HOME[:3, :3]]))
    slerp = Slerp([0.0, 1.0], key_rots)

    for i in range(1, n + 1):
        s = i / n
        T_way = np.eye(4)
        T_way[:3, :3] = slerp(s).as_matrix()
        T_way[:3, 3] = (1 - s) * T_start[:3, 3] + s * T_BASE_HAND_HOME[:3, 3]

        ok, dp, dr = settle(pub, current_msg, clamp_target_pose(T_way @ T_hand_ctrl),
                            seq, HOME_SETTLE_TIMEOUT_S)
        seq += 1
        if not ok:
            print(f"[home] waypoint {i}/{n} timed out: {dp*1000:.1f} mm "
                  f"{np.rad2deg(dr):.1f} deg residual", flush=True)

    T_end = pose_msg_to_matrix(current_msg) @ T_ctrl_hand
    dp, dr = pose_error(T_end, T_BASE_HAND_HOME)
    print(f"[home] done — {dp*1000:.1f} mm / {np.rad2deg(dr):.1f} deg from home")
    return seq


# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the DAgger run-12 BC policy (checkpoint/cp2) on the FR3.")
    p.add_argument("--rosbridge-host", type=str, default=ROSBRIDGE_HOST)
    p.add_argument("--rosbridge-port", type=int, default=ROSBRIDGE_PORT)
    p.add_argument("--policy-dir", type=str, default=str(DEFAULT_POLICY_DIR),
                   help="run dir holding config.yaml, normalization.npz, best.pt")
    p.add_argument("--ckpt", type=str, default=None,
                   help="explicit .pt path (default: <policy-dir>/best.pt)")
    p.add_argument("--hand-seg-ckpt", type=str, default=str(DEFAULT_HAND_SEG_CKPT))
    p.add_argument("--hand-eye", type=str, default=None,
                   help="4x4 .npy T_hand_cam (camera -> panda_hand). Defaults to "
                        "the sim's nominal wrist mount, which is NOT a "
                        "calibration of your D435.")
    p.add_argument("--ee-offset-z", type=float, default=DEFAULT_EE_OFFSET_Z,
                   help="z offset from panda_hand to the frame /cartesian_pose "
                        "publishes. Measured as 0 on this robot (F_T_EE has no "
                        "translation, so it already publishes panda_hand); use "
                        "0.1034 only if the EE is reconfigured to the TCP.")
    p.add_argument("--enable-gripper", action="store_true",
                   help="actually command franka_gripper on a CLOSE prediction")
    p.add_argument("--step-mode", action="store_true",
                   help="SAFE MODE: preview each predicted step and only execute "
                        "it when you press SPACE. Nothing moves unprompted.")
    p.add_argument("--home", action="store_true",
                   help="drive to the sim's episode-start pose before running "
                        "(interpolated Cartesian move — keep the workspace clear)")
    p.add_argument("--home-only", action="store_true",
                   help="home and exit, without running the policy")
    p.add_argument("--max-steps", type=int, default=MAX_POLICY_STEPS)
    p.add_argument("--settle-timeout", type=float, default=SETTLE_TIMEOUT_S)
    p.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    p.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    p.add_argument("--depth-width", type=int, default=DEPTH_WIDTH,
                   help="must be a depth mode the device offers — the D435 has "
                        "no 424x240 depth, unlike its color stream")
    p.add_argument("--depth-height", type=int, default=DEPTH_HEIGHT)
    p.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    p.add_argument("--dry-run", action="store_true",
                   help="run perception + policy and print targets, publish nothing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError(
            "No CUDA device. The PointNet++ backbone in BCPolicy has GPU-only "
            "CUDA ops and cannot run on CPU.")
    print(f"Using device: {device}")

    if args.dry_run and (args.home or args.home_only):
        # Homing is the one thing that has to publish; silently skipping it and
        # falling through to the policy loop would be a nasty surprise.
        raise SystemExit("--home/--home-only need a live connection and cannot "
                         "run under --dry-run, which publishes nothing.")

    policy_dir = Path(args.policy_dir).expanduser().resolve()
    ckpt = args.ckpt or str(policy_dir / "best.pt")
    hand_seg_ckpt = Path(args.hand_seg_ckpt).expanduser().resolve()

    T_hand_cam = (np.load(args.hand_eye).astype(np.float64) if args.hand_eye
                  else T_HAND_CAM_NOMINAL)
    if T_hand_cam.shape != (4, 4):
        raise ValueError(f"--hand-eye must be 4x4, got {T_hand_cam.shape}")
    if args.hand_eye is None:
        print("[calib] WARNING: using the SIM's nominal wrist mount for T_hand_cam. "
              "Pass --hand-eye <T_hand_cam.npy> once you have calibrated the D435.")

    print(f"Policy dir     : {policy_dir}")
    print(f"Checkpoint     : {ckpt}")
    print(f"Hand seg ckpt  : {hand_seg_ckpt}")
    print(f"EE offset z    : {args.ee_offset_z:.4f} m (panda_hand -> published frame)")
    print(f"Gripper        : {'ENABLED' if args.enable_gripper else 'disabled'}")

    hand_seg_model, hand_seg_preprocess = load_hand_segmenter(device, hand_seg_ckpt)
    policy, run_cfg = load_bc_policy(policy_dir, ckpt, device)
    _assert_state_layout(run_cfg)
    warm_up_policy(policy, device)

    # panda_hand <-> the frame the controller publishes.
    T_hand_ctrl = z_offset_transform(args.ee_offset_z)
    T_ctrl_hand = invert_transform(T_hand_ctrl)

    client = None
    sub = pub = gripper_sub = None
    cam = None
    gripper = FrankaGripper(None, False)
    publish_seq = 0   # shared by homing and the policy loop

    try:
        if not args.dry_run:
            client = roslibpy.Ros(host=args.rosbridge_host, port=args.rosbridge_port)
            client.run()
            for _ in range(50):
                if client.is_connected:
                    break
                time.sleep(0.1)
            if not client.is_connected:
                raise RuntimeError(
                    f"Could not connect to rosbridge at "
                    f"{args.rosbridge_host}:{args.rosbridge_port}")

            sub = roslibpy.Topic(client, CURRENT_POSE_TOPIC, POSE_MSG_TYPE)
            pub = roslibpy.Topic(client, TARGET_POSE_TOPIC, POSE_MSG_TYPE)
            gripper_sub = roslibpy.Topic(client, GRIPPER_STATE_TOPIC, GRIPPER_STATE_TYPE)

            sub.subscribe(pose_cb)
            gripper_sub.subscribe(gripper_state_cb)
            pub.advertise()
            gripper = FrankaGripper(client, args.enable_gripper)

            print(f"Connected to rosbridge at {args.rosbridge_host}:{args.rosbridge_port}")
            print("Waiting for current pose...")
            t0 = time.time()
            while current_msg is None and time.time() - t0 < 10.0:
                time.sleep(0.05)
            if current_msg is None:
                raise RuntimeError(f"No message on {CURRENT_POSE_TOPIC}")

            T0 = pose_msg_to_matrix(current_msg) @ T_ctrl_hand
            print(f"Start panda_hand pose: xyz=({T0[0,3]:+.3f}, {T0[1,3]:+.3f}, "
                  f"{T0[2,3]:+.3f})  <- sanity-check this against the real flange "
                  "before trusting --ee-offset-z")

            if args.home or args.home_only:
                hp = T_BASE_HAND_HOME[:3, 3]
                print(f"About to move the arm to the sim's start pose, base frame "
                      f"xyz=({hp[0]:+.3f}, {hp[1]:+.3f}, {hp[2]:+.3f}), as an "
                      "interpolated Cartesian path. CLEAR THE WORKSPACE.")
                if input("Type 'go' to home: ").strip().lower() != "go":
                    print("Homing declined; exiting without moving.")
                    return
                publish_seq = go_home(pub, T_ctrl_hand, T_hand_ctrl, publish_seq)
                if args.home_only:
                    return

        cam = RealSenseCamera(
            color_size=(args.camera_width, args.camera_height),
            depth_size=(args.depth_width, args.depth_height),
            fps=args.camera_fps,
        )
        try:
            cam.start()
        except RuntimeError as err:
            # librealsense's two failure modes here look alike but mean opposite
            # things, and neither message names the stream at fault.
            modes = (f"color {args.camera_width}x{args.camera_height} + depth "
                     f"{args.depth_width}x{args.depth_height} @ {args.camera_fps}fps")
            if "resolve" in str(err).lower():
                hint = ("the device does not offer that combination. Color and "
                        "depth mode lists differ — the D435 has 424x240 color but "
                        "no 424x240 depth. `rs-enumerate-devices -m` lists both.")
            else:
                hint = ("the modes were accepted but no frames arrived, which is "
                        "a link/power fault rather than a config one. Check that "
                        "usb_type_descriptor reads 3.x: a D435 that enumerates at "
                        "2.1 (charge-only cable, USB2 port, or a hub) often "
                        "advertises modes it then cannot stream at all.")
            raise RuntimeError(f"RealSense failed to start {modes} ({err}) — {hint}") from err

        if args.step_mode:
            print(f"STEP MODE ({args.max_steps} policy steps max). The overlay "
                  "previews each predicted action; press SPACE in the image "
                  "window to execute one step, 'h' to re-home, 'q' to quit.")
        else:
            print(f"CONTINUOUS MODE ({args.max_steps} policy steps max). Steps "
                  "execute as soon as they are predicted. 'h' to re-home, "
                  "'q' or Esc to quit.")

        last_hand_xyz = None
        last_object_xyz = None
        step = 0
        stop_reason = "max steps reached"

        while step < args.max_steps:
            color_bgr, depth_m, _ = cam.get_frames()

            if args.dry_run:
                T_base_hand = np.eye(4)
            else:
                if current_msg is None:
                    time.sleep(0.01)
                    continue
                T_base_hand = pose_msg_to_matrix(copy.deepcopy(current_msg)) @ T_ctrl_hand

            # ---- hand segmentation ----
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            x = hand_seg_preprocess(color_rgb).unsqueeze(0).to(device, non_blocking=True)
            with torch.inference_mode():
                pred = hand_seg_model(x).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            hand_mask = cv2.resize(pred, (color_bgr.shape[1], color_bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            hand_mask = normalize_mask(hand_mask)

            # ---- hand / object clouds, camera frame ----
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
            if len(result.hand_xyz) > 0:
                last_hand_xyz = result.hand_xyz
            if len(result.object_xyz) > 0:
                last_object_xyz = result.object_xyz

            # ---- camera frame -> panda_hand frame ----
            # Eye-in-hand: the cloud needs no robot pose at all, only the fixed
            # mount transform. (The tripod setup in policy_runner.py had to go
            # through the base frame; this one does not.)
            hand_policy = transform_points(T_hand_cam.astype(np.float32), result.hand_xyz)
            object_policy = transform_points(T_hand_cam.astype(np.float32), result.object_xyz)
            have_obs = len(hand_policy) > 0 and len(object_policy) > 0

            # ---- PREDICT (never moves the robot) ----
            # Prediction is separated from execution so step mode can show you
            # what the policy wants to do before anything is published. In
            # continuous mode the two phases just run back to back.
            grasp_close = False
            clamped = False
            delta6 = None
            T_base_ctrl_target = None

            if have_obs:
                pc = build_bc_point_tensor(object_policy, hand_policy)
                rs = build_robot_state(T_base_hand, read_gripper_norm(assume_open=True))

                action = policy_act(policy, pc, rs, device)   # [7], ch6 in {0, 1}
                grasp_close = bool(action[6] < 0.5)

                if not grasp_close:
                    delta6, clamped = clamp_action_delta(action[:6])
                    T_base_hand_target = T_base_hand @ unpack_action(delta6)
                    T_base_ctrl_target = clamp_target_pose(T_base_hand_target @ T_hand_ctrl)

            # ---- display ----
            overlay = overlay_mask(color_bgr, hand_mask)
            dbg = result.debug
            lines = [
                f"step {step}/{args.max_steps}  obj={len(object_policy)} "
                f"hand={len(hand_policy)} fb_h={dbg['used_last_hand']} "
                f"fb_o={dbg['used_last_object']}",
            ]
            if not have_obs:
                lines.append("NO OBSERVATION - holding")
            elif grasp_close:
                lines.append("PENDING: CLOSE GRIPPER")
            else:
                lines.append(
                    f"PENDING: d={np.linalg.norm(delta6[:3])*100:.1f}cm "
                    f"r={np.rad2deg(np.linalg.norm(Rot.from_matrix(unpack_action(delta6)[:3, :3]).as_rotvec())):.1f}deg"
                    + ("  CLAMPED" if clamped else ""))
            lines.append("SPACE=execute  h=home  q=quit" if args.step_mode
                         else "q=quit  h=home")

            for i, text in enumerate(lines):
                cv2.putText(overlay, text, (10, 22 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255) if i == 1 else (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("my_policy_runner_overlay", overlay)
            cv2.imshow("my_policy_runner_hand_mask", (hand_mask * 255).astype(np.uint8))

            # ---- key handling ----
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop_reason = "user quit"
                break
            if key == ord("h"):
                if args.dry_run:
                    print("[home] ignored in --dry-run")
                else:
                    publish_seq = go_home(pub, T_ctrl_hand, T_hand_ctrl, publish_seq)
                continue

            # ---- EXECUTE ----
            # Step mode gates every single motion on SPACE; nothing the policy
            # predicts reaches the robot until you ask for it.
            if not have_obs:
                time.sleep(0.01)   # nothing segmented; does NOT consume a step
                continue
            if args.step_mode and key != ord(" "):
                continue

            if grasp_close:
                print(f"[{step:02d}] policy commanded CLOSE", flush=True)
                gripper.close()
                stop_reason = "policy closed the gripper"
                break

            pos = T_base_ctrl_target[:3, 3]
            print(f"[{step:02d}] target xyz=({pos[0]:+.3f}, {pos[1]:+.3f}, "
                  f"{pos[2]:+.3f})  |d|={np.linalg.norm(delta6[:3]):.4f}m "
                  f"obj={len(object_policy):4d} hand={len(hand_policy):4d}"
                  f"{'  CLAMPED' if clamped else ''}", flush=True)

            if not args.dry_run:
                ok, dpos, drot = settle(pub, current_msg, T_base_ctrl_target,
                                        publish_seq, args.settle_timeout)
                publish_seq += 1
                if not ok:
                    print(f"     settle timeout: {dpos*1000:.1f}mm "
                          f"{np.rad2deg(drot):.1f}deg residual", flush=True)
            step += 1

        print(f"Episode ended after {step} policy steps: {stop_reason}")

    finally:
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

        for topic, fn in ((sub, "unsubscribe"), (gripper_sub, "unsubscribe"),
                          (pub, "unadvertise")):
            if topic is not None:
                try:
                    getattr(topic, fn)()
                except Exception:
                    pass
        gripper.shutdown()
        if client is not None:
            try:
                client.terminate()
            except Exception:
                pass

        print("Stopped.")


if __name__ == "__main__":
    main()
