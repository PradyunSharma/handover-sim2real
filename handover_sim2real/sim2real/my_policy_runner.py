#!/usr/bin/env python3
"""
Run a Phase-1/4 BC policy (checkpoint/run12, run16, run19) on the physical FR3.

Two checkpoints, one runner. They share the observation contract below and
differ only in how many viewpoints fill the cloud:

    cp2   DAgger run 12, wrist camera only.  --cameras wrist          (default)
    cp3   DAgger run 16, wrist+left+right.   --cameras wrist,tripod

cp3's extra views exist because the wrist camera loses the object during the
last few centimetres of the approach — exactly the phase Phase-4 DAgger was
meant to fix. Running cp3 off the wrist alone therefore removes the information
it was trained to use, so pass --cameras. The fusion lives in
pointcloud_multicam.py, which documents where the real rig cannot match the
simulator's segmentation oracle.

This is the sibling of policy_runner.py, which drives the CVPR2023 GA-DDPG
model. The two policies are NOT interchangeable — three things differ, and each
one silently produces garbage if carried over from the other script:

  1. POINT CLOUD ORDER/LABELS. Both cp2 and cp3 were trained by PointListener
     with the object class first: 896 object points [x,y,z,1,0] then 128 hand
     points [x,y,z,0,1], 1024 total. policy_runner.py's
     build_policy_point_tensor emits the opposite (hand first, hand=[1,0]). See
     pointcloud_multicam.build_policy_cloud.
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

    # cp3, both cameras fused (needs a validated hand-eye session)
    python my_policy_runner.py --run run16 \
           --cameras wrist,tripod --calib-session session_02 --home --step-mode
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import cv2
import numpy as np
import roslibpy
import torch
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp

# -----------------------------------------------------------------------------
# Directory layout
# -----------------------------------------------------------------------------
SIM2REAL_DIR = Path(__file__).resolve().parent
HANDOVER_SIM2REAL_ROOT = SIM2REAL_DIR.parents[1]
PROJECT_ROOT = HANDOVER_SIM2REAL_ROOT.parent
HANDS_SEG_ROOT = PROJECT_ROOT / "hands-segmentation-pytorch"

# NAMED AFTER THE DAgger RUN THEY CAME FROM, not cp1/cp2/cp3. The old names said
# nothing: "cp3" does not tell you what it was trained on, which checkpoint
# inside the run it is, or how to find the log that explains its behaviour — and
# a rollout labelled "cp3" is unreproducible six months later. run16 does all
# three, because output/dagger_runs/dagger4_run16/ is right there.
#
# checkpoint/cp1 is deliberately NOT renamed: it is the hand SEGMENTATION model
# (HandSegModel, 2021), not a policy, and has no run behind it.
CHECKPOINT_DIR = SIM2REAL_DIR / "checkpoint"
DEFAULT_RUN = "run12"
DEFAULT_POLICY_DIR = CHECKPOINT_DIR / DEFAULT_RUN
DEFAULT_HAND_SEG_CKPT = CHECKPOINT_DIR / "cp1" / "checkpoint.ckpt"


def available_runs() -> list[str]:
    """Installed policy folders, for the --run help text and its error message.

    A folder counts only if it holds all three files a policy needs, so a
    half-copied checkpoint is not offered as a choice.
    """
    if not CHECKPOINT_DIR.is_dir():
        return []
    return sorted(
        d.name for d in CHECKPOINT_DIR.iterdir()
        if d.is_dir() and all((d / f).exists() for f in
                              ("config.yaml", "normalization.npz", "best.pt")))

if not HANDS_SEG_ROOT.exists():
    raise FileNotFoundError(
        f"hands-segmentation-pytorch not found: {HANDS_SEG_ROOT}\n"
        "It must sit alongside handover-sim2real (it provides HandSegModel)."
    )

sys.path.insert(0, str(HANDS_SEG_ROOT))
sys.path.insert(0, str(HANDOVER_SIM2REAL_ROOT))
sys.path.insert(0, str(SIM2REAL_DIR))

from pointcloud_multicam import (  # noqa: E402
    NUM_HAND_POINTS,
    NUM_OBJECT_POINTS,
    PC_CHANNELS,
    FINGER_EXCLUSION_MODES,
    ROBOT_EXCLUSION,
    CALIB_DIR,
    HandSegmenter,
    MultiCameraPerception,
    apply_camera_model,
    check_session_camera,
    build_policy_cloud,
    build_rigs,
    overlay_mask,
)
from sam2_segmenter import (  # noqa: E402
    add_segmentation_args,
    build_segmenter,
    describe_segmenter,
)
from cloud_viewer import source_for_cloud  # noqa: E402
from dual_cloud_window import (  # noqa: E402
    DualCloudWindow,
    context_cloud,
    exit_without_finalizing,
)
from transforms import invert_transform  # noqa: E402
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

# Carries F_T_EE, which is the ONLY authority on which frame /cartesian_pose is
# publishing. See measure_ee_offset_z.
FRANKA_STATE_TOPIC = "/franka_state_controller/franka_states"
FRANKA_STATE_TYPE = "franka_msgs/FrankaState"
GRIPPER_HOMING_GOAL_TOPIC = "/franka_gripper/homing/goal"
GRIPPER_HOMING_GOAL_TYPE = "franka_gripper/HomingActionGoal"
GRIPPER_STOP_GOAL_TOPIC = "/franka_gripper/stop/goal"
GRIPPER_STOP_GOAL_TYPE = "franka_gripper/StopActionGoal"
# Liveness, not existence. actionlib publishes a GoalStatusArray on this topic
# continuously while the server is up, so one message is proof something is
# listening for goals — which advertising the goal topic ourselves is not.
GRIPPER_GRASP_STATUS_TOPIC = "/franka_gripper/grasp/status"
GRIPPER_STATUS_TYPE = "actionlib_msgs/GoalStatusArray"

# THESE TOPICS COME FROM franka_ros, NOT FROM THE IMPEDANCE CONTROLLER. The
# `load_gripper:=True` on
#
#   roslaunch franka_human_friendly_controllers \
#       cartesian_variable_impedance_controller.launch robot_ip:=... load_gripper:=True
#
# reaches franka_control.launch in that package, which does
# `<include file="$(find franka_gripper)/launch/franka_gripper.launch" if="$(arg
# load_gripper)">`. So the gripper is served by the stock franka_gripper node and
# the interface is the stock one: grasp/move/homing/stop action servers plus
# /franka_gripper/joint_states. The package's own reference client
# (python/LfD/panda.py) publishes GraspActionGoal to /franka_gripper/grasp/goal
# exactly as below, which is the confirmation that goal topics work here without
# an action client.
GRIPPER_MAX_FINGER_M = 0.04   # sim: gripper_norm = joint_pos[7] / 0.04
GRASP_WIDTH_M = 0.0           # close all the way; epsilon/force do the work
GRASP_SPEED = 0.05
GRASP_FORCE = 20.0
# WIDE ON PURPOSE. libfranka calls an object grasped only if the final finger
# distance d satisfies width - inner < d < width + outer, and reports failure
# otherwise. With width = 0 that makes epsilon the maximum object thickness the
# grasp will admit — at 0.04 anything thicker than 40 mm came back as a failed
# grasp, which covers a good part of the YCB set. The reference client in the
# controller package uses 0.3 for the same reason, i.e. "any width counts". The
# fingers close and clamp identically either way; this only decides what the
# action result says.
GRASP_EPSILON_INNER = 0.3
GRASP_EPSILON_OUTER = 0.3

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
HOME_REFINE_PASSES = 6      # backstop; the refine exits as soon as it is inside
                            # HOME_REFINE_TOL_M, which is normally the first pass

# HOME IS A STARTING POSE, NOT A PRECISION TARGET, and asking for more precision
# than the arm can deliver is what made homing hunt.
#
# This was 2 mm. The arm has a ~17 mm standing droop and an under-travel gain
# that varies 0.44-0.92 between consecutive moves, so 2 mm is below what it can
# reliably land: each refine pass overshot, the next corrected back, and with six
# passes of up to CREEP_MAX_NUDGES each that is a lot of small motions around
# home. Reported from hardware as the arm "trying to find exact home position but
# not able to".
#
# Nothing downstream needs the precision. Every training episode began at
# ENV.PANDA_INITIAL_POSITION, but the policy is single-frame and closed-loop —
# it observes wherever the arm actually is and steps from there, so a couple of
# centimetres at t=0 is a slightly different starting state, not an error that
# accumulates. 2 cm is the operator's call on this rig.
# HOW LONG AFTER A CLOSE THE ARM TAKES ITSELF HOME. Long enough for the fingers
# to finish closing and for the grasp to be seen to hold before the object is
# carried anywhere — franka_gripper's grasp action is asynchronous, so the
# CLOSE command returns well before the fingers have stopped — and short enough
# that the run does not stall waiting for a human. It is a deadline the main
# loop watches, never a sleep, so every key stays live throughout.
#
# 3.0 and not the 5.0 this started at. The fingers are the thing being waited
# for and they are quick: GRASP_SPEED is 0.05 m/s over at most ~40 mm of travel
# per finger, so under a second even from fully open. The rest was margin for
# watching the grasp hold, and in --exp-mode that margin is paid once per
# attempt with the operator already watching.
AUTO_HOME_DELAY_S = 3.0

HOME_REFINE_TOL_M = 0.020   # --home-tol
HOME_WAY_TOL_M = 0.008      # intermediate waypoints are a path, not a target

# STREAMED HOMING. The waypoint loop already knew the intermediate poses are "a
# path, not a target" — HOME_WAY_TOL_M says so — but it still called settle() on
# each one, and settle() ends by waiting for the arm to hold still for
# SETTLE_STILL_HOLD_S. So a 30 cm home executed as fifteen 2 cm moves separated
# by fifteen dead stops. The path was smooth; the traversal of it was not.
#
# Streaming publishes the same interpolated path on a clock instead, so the arm
# is always being told to go somewhere slightly ahead of where it is and never
# arrives anywhere until the end. Unlike the policy loop this cannot deadlock:
# the waypoints are absolute and pre-computed, so they advance whether or not
# the arm keeps up, and no break-away lead is needed to make progress.
#
# Speed is set here rather than derived from the waypoint spacing, and the
# spacing then follows from the publish rate. That is the right way round — it
# means changing the rate changes only smoothness, never how fast the arm moves
# across the room.
#
# 0.12 m/s and not the 0.08 this started at. A streamed path is followed with a
# standing lag of roughly speed * tau, and tau on this controller is ~0.12 s, so
# going half again as fast puts the arm ~14 mm behind the commanded equilibrium
# instead of ~10 mm. That is a pull, not an error that accumulates:
# clamp_command_lead in the stream loop bounds the equilibrium to
# MAX_COMMAND_LEAD_M (100 mm) from the measured pose, so an arm that cannot keep
# up is left behind rather than yanked. The refine pass that follows is a static
# target and unaffected, which is why "how fast it crosses the room" and "where
# it ends up" stay separate questions.
HOME_SPEED_M_S = 0.12       # Cartesian speed of the streamed path
HOME_STREAM_DT_S = 0.05     # publish period; spacing = speed * this = 6 mm

# Fallback z offset from panda_hand to whatever frame /cartesian_pose publishes,
# used ONLY when the robot does not publish F_T_EE. `measure_ee_offset_z` reads
# the real value at startup; see there for why this is no longer a constant.
#
# THIS CONSTANT WAS ONCE 0 AND WAS ONCE RIGHT, which is the whole cautionary
# tale. franka_states then reported F_T_EE = (0,0,0) with a -45 deg z rotation,
# and flange + Rz(-45) is exactly how panda_gripper_hand_camera.urdf defines
# panda_hand — so the controller was publishing the policy's frame, confirmed
# to 0.06 mm against pybullet FK. The robot's configured end effector later
# became the Franka Hand TCP, F_T_EE became (0,0,0.1034), and every one of
# those statements silently stopped being true.
#
# 0 is kept as the fallback rather than 0.1034 because a robot that publishes no
# F_T_EE at all is most likely not a Franka Hand setup either.
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

# The crop radii, depth limits, strides and point floors used to live here as
# module constants. They are per-camera now — a wrist view at 0.3 m and a tripod
# view at 1.0 m are not the same measurement problem — so they live in
# pointcloud_multicam.{WRIST_PARAMS,FIXED_PARAMS}. One definition, because two
# copies of a cloud-extraction parameter that must agree is exactly how the
# board spec got out of sync in the calibration scripts.

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
# 3.0, not 2.0: creep does its correcting INSIDE one settle() rather than across
# repeated calls, so the budget that used to cover one command now has to cover
# the whole approach — initial travel, a handful of nudges at >= CREEP_DEAD_S +
# CREEP_STALL_HOLD_S apart, and the final hold. A 39 mm step measures ~1.2 s of
# that, so 2.0 s left no margin and would have turned slow steps into timeouts.
SETTLE_TIMEOUT_S = 3.0
SETTLE_POLL_S = 0.02

# ── abort ────────────────────────────────────────────────────────────────────
# 't' at any moment. THIS IS A SOFT STOP AND NOT AN EMERGENCY STOP — say so
# plainly, because a key that claims to stop a robot will be trusted in exactly
# the situation where being wrong matters. /equilibrium_pose is the only
# interface here, so the strongest thing available is to re-command the
# equilibrium AT THE ARM'S MEASURED POSE: the position error goes to zero, the
# controller stops pulling, and the arm holds roughly where it stands. It does
# not brake, it does not go rigid, and it still sags by the standing droop
# (~17 mm). The button on the wall is the emergency stop.
#
# The reason this needs machinery at all rather than another `elif` in the key
# handler is that the key handler does not run while the robot is moving. A
# policy step can sit inside settle() for up to 9 s, homing streams for several,
# and a gripper open blocks for up to 15. A stop that is only noticed after the
# motion it was meant to interrupt has finished is not a stop, so the blocking
# loops poll for it themselves through `poll_stop()`.
stop_requested = False
_stop_poller: Optional[Callable[[], bool]] = None
# WHICH key raised the stop, and WHEN. `stop_requested` stays the single
# question every blocking loop asks — "should I stop?" — and all five of them
# want the same answer for 't' and for --exp-mode's 'f'. What differs is only
# what the MAIN LOOP does next: 't' is a user stop, 'f' is a RECORDED policy
# failure, and a session whose log cannot tell those apart has no failure data.
# `stop_at` is the PRESS time, which is the honest end of an attempt's clock —
# the loop does not reach its dispatch again for another perception pass.
stop_reason_key: Optional[str] = None
stop_at: Optional[float] = None
# Which keys may raise a stop from INSIDE a motion. 't' always. --exp-mode adds
# 'f' for the duration of an attempt and removes it again, because during the
# verdict wait the same key means "the grasp failed" and must NOT abort the arm
# carrying the object home. Mutated by ExperimentSession, read on every poll.
STOP_KEYS = {"t"}
# Keys seen by the inner poller that were NOT a stop key. cv2.waitKey
# consumes what it reads, so without this a SPACE or a 'q' pressed during a
# motion would be swallowed by the poll rather than merely delayed by it.
_swallowed_keys: list[str] = []


def request_stop(key: str = "t", when: Optional[float] = None) -> None:
    """Raise the abort, remembering why and when.

    FIRST KEY WINS. A 't' pressed while the arm is still unwinding from an 'f'
    must not rewrite why we stopped — the reason is what gets recorded, and the
    first press is the one that describes what the operator saw.
    """
    global stop_requested, stop_reason_key, stop_at
    if not stop_requested:
        stop_reason_key = key
        stop_at = time.time() if when is None else when
    stop_requested = True


def clear_stop() -> None:
    """Acknowledge an abort. Called only after the arm has actually been frozen."""
    global stop_requested, stop_reason_key, stop_at
    stop_requested = False
    stop_reason_key = None
    stop_at = None


def poll_stop() -> bool:
    """Has a stop key been pressed? Safe to call from inside a motion loop."""
    if stop_requested:
        return True
    if _stop_poller is not None:
        try:
            if _stop_poller():
                # BOTH CONTRACTS. The real poller raises through request_stop()
                # itself, because only it knows which key was pressed and when.
                # A poller that merely returns True — the older contract, and
                # what the tests install — still stops the motion, defaulting to
                # 't'. request_stop is first-key-wins, so this cannot overwrite
                # a reason the poller already recorded.
                request_stop("t")
        except Exception:
            pass                    # a viewer that has gone away must not abort
    return stop_requested

# Motion-stopped criterion — the one that actually fires on this robot. Per-poll
# movement below these for SETTLE_STILL_HOLD_S means the arm has finished
# responding to the target, standing impedance droop and all. Measured droop is
# ~17 mm, so the reached-the-target test above never passes; see settle().
#
# Sized from measurement, and it has to fit between two bounds:
#   noise floor  /cartesian_pose publishes at ~989 Hz; with the arm stationary
#                the worst per-message delta over 6 s was 0.0065 mm / 0.0011 deg.
#   motion floor homing moved 2 cm per waypoint in >3 s, i.e. of order 0.1 mm per
#                20 ms poll, so anything at or above that reads as "still" while
#                the arm is genuinely travelling.
# 0.05 mm sits ~8x above the noise and ~2x below the slowest observed motion.
# Erring tight is deliberate: failing to detect stillness costs a timeout (slow
# but safe), while declaring it early hands the policy a mid-motion observation.
SETTLE_STILL_POS_M = 0.00005     # 0.05 mm between polls == 2.5 mm/s
SETTLE_STILL_ROT_DEG = 0.01      # == 0.5 deg/s
# 0.30, not 0.15: homing reported 4.9 mm at the moment settle() returned, but the
# arm was at 3.54 mm when re-measured later and then drifted only 0.011 mm over
# 25 s. So ~1.4 mm of tail was escaping past the quiet window. A longer hold
# catches more of it, and biases the droop estimate less. Costs 0.15 s a step.
SETTLE_STILL_HOLD_S = 0.30       # must stay quiet this long

# Droop compensation. The Cartesian impedance controller settles at
#   A = E - D
# where E is the commanded equilibrium and D the standing offset it needs to
# hold the arm against gravity. Measured on this robot: D = (3.7, 3.9, 16.7) mm,
# confirmed by two homing runs that both ended exactly |D| from home regardless
# of where they started.
#
# This matters far more than it looks. Each step commands E = A_measured + delta,
# so the arm lands at A + delta - D: the ACHIEVED motion is delta - D. With
# delta ~20-29 mm and D ~17.6 mm mostly -z, a commanded rise executes as a
# descent. Compensating means commanding E = target + D so that A = target.
#
# D is re-estimated after every settle as (commanded - measured), which is a
# stable fixed point: once compensation is exact the estimate stops moving.
# Translation only — rotational droop measured 0.9 deg against a ~3.4-4.6 deg
# per-step rotation, so it slows convergence rather than reversing it, and
# composing rotational corrections carries more risk than it buys.
# Correction gain, NOT a smoothing weight — hence values above 1.
#
# The controller is not `A = E - D` with a fixed D. Measured at two commands:
#     E = home                     -> D = (3.74, 3.87, 16.74) mm
#     E = home + (5.9,4.5,20.5) mm -> D = (7.30, 5.05, 23.69) mm
# Commanding 20.5 mm more in z bought only 13.55 mm of motion, i.e. the plant is
#     A = G*E + b,  G = diag(~0.40, ~0.74, ~0.66)
# It under-travels every commanded displacement rather than sitting at a fixed
# offset from it.
#
# The update below is integral feedback, so its fixed point is exact for ANY G;
# only the rate depends on it, as (1 - alpha*G) per pass. alpha=0.8 with g=0.66
# gives 0.47 — which is exactly the ~0.55 per-pass shrink observed on the robot,
# and why three refine passes could not reach home. alpha ~ 1/g is the deadbeat
# choice; 1.3 gives 0.48 / 0.14 / 0.04 on the three measured axes.
#
# Stability needs 0 < alpha*g < 2, so alpha < 2/0.74 = 2.7. 1.3 keeps margin at
# both ends.
DROOP_EMA_ALPHA = 1.3
# Bounding the OFFSET ITSELF is the wrong constraint: whatever the mechanism, the
# offset needed to hold a pose is not constant, so a tight cap on it binds and
# each step then achieves less than the last.  What to bound instead is how far
# the commanded equilibrium sits ahead of where the arm actually IS, since that
# governs how hard the controller pulls and how far it could lunge on a bad
# estimate.  Both values below are SAFETY BACKSTOPS chosen not to bind in normal
# operation (observed offsets ~24 mm, steps ~25 mm), not tuned parameters.
#
# WHY THEY ARE NOT TUNED: the underlying behaviour is not yet identified. Two
# resting measurements 20 mm apart are consistent with an affine under-travel
# (A = G*E + b, G ~ 0.4-0.74), but extrapolating that across the workspace
# predicts the arm could barely move in x, which the robot plainly contradicts.
# Stiction fits the same data at least as well — the arm halts once the impedance
# force drops below static friction, leaving a variable shortfall rather than a
# fixed ratio — and unlike the affine model it also explains why iterating on a
# static target converges on hardware (homing: 17.6 -> 9.8 -> 4.9 -> 3.5 mm).
# Distinguishing them needs commanded displacements measured at full rest.
# Until then: iterate, bound generously, and report the residual every step.
MAX_COMMAND_LEAD_M = 0.10
MAX_DROOP_COMP_M = 0.30

# Per-policy-step convergence. Each step's target is static, so it is iterated to
# the same way homing is — see move_to(). 3 mm is ~10% of a typical 25 mm step,
# well inside what the closed-loop policy corrects for on the next observation,
# and the passes cap keeps a step from stalling the episode if the arm is blocked.
STEP_CONVERGE_PASSES = 3

# How close a step has to land before the policy is allowed to look again.
#
# RELATIVE to the step, not absolute, and that is the whole point. Two things
# push in opposite directions and a single number cannot serve both:
#
#   the approach   Steps are 30-50 mm and the arm is nowhere near the object.
#                  Precision here buys nothing — the loop is closed, so a short
#                  step is just a short step, and the next observation is of
#                  wherever the arm actually is. What precision COSTS here is
#                  corrections, and corrections are what the arm visibly does
#                  after the big move.
#   the placement  Near the object the policy commands millimetres, and there a
#                  correction is the point.
#
# The fraction handles both, and it has a property a fixed tolerance does not:
# it can never exceed the step, so a step can never be satisfied without moving.
# A fixed 20 mm tolerance would make any commanded step under 20 mm a complete
# no-op — the arm would sit still while the policy believed it had moved.
#
# 0.6 is where the corrections stop, measured against a plant carrying this
# robot's own gain jitter (see below): 3.2 commands per step at 5 mm, 2.0 at
# 8 mm, 1.0 at 0.6 of the step. The arm then executes ~75% of each commanded
# step in one continuous motion instead of ~91% in three or four bursts.
#
# WHY THE JITTER FORCES THIS. Four consecutive steps on the robot measured
# under-travel gains of 0.51, 0.44, 0.85, 0.92 — the plant is genuinely different
# every move, because the direction changes and friction and the arm's
# configuration change with it. No feed-forward can land a first command
# precisely against that; precision requires measuring and correcting, and
# correcting requires stopping. The trade is real and this is which side of it
# to be on for a closed-loop policy.
STEP_CONVERGE_TOL_FRAC = 0.6
STEP_CONVERGE_TOL_MIN_M = 0.004
STEP_CONVERGE_TOL_M = 0.005     # only a default for callers with no step size


def step_tolerance(step_m: float, frac: float = STEP_CONVERGE_TOL_FRAC) -> float:
    """Convergence tolerance for a commanded step of `step_m`."""
    return max(STEP_CONVERGE_TOL_MIN_M, frac * float(step_m))

# -----------------------------------------------------------------------------
# WHY THIS LOOP IS SLOWER THAN THE PAPER'S, which is a design difference and not
# a performance bug. Worth stating plainly because the instinct is to blame the
# network, and the network is not the cost.
#
# CVPR2023 (Christen et al., arXiv 2303.17592) runs a FIXED-RATE loop. Its
# rollout is, per policy step:
#
#     action = policy(obs)
#     tjp    = IK(current_ee_pose @ delta)
#     for _ in range(steps_action_repeat):    # 0.15 s / 0.001 s = 150 substeps
#         obs = env.step(tjp)
#
# POLICY.TIME_ACTION_REPEAT = 0.15, so it re-observes and re-predicts at 6.7 Hz
# and NEVER checks whether the arm arrived. It cannot get stuck waiting, and the
# arm is permanently chasing a target that has already moved — which is what
# makes it look continuous. Undershoot needs no correction pass because the next
# action is a fresh delta from wherever the arm actually got to; the error is
# absorbed by the next prediction rather than by a settling loop.
#
# This runner instead waits for convergence before it will look again: move_to()
# runs settle() until the arm is inside tolerance and has held still for
# SETTLE_STILL_HOLD_S. That buys a clean single-frame observation — the policy is
# Markov and its robot_state is an EE pose that would otherwise be paired with a
# cloud captured while the arm was moving — and it costs the entire step time.
# Perception is ~43 ms and inference ~10 ms against a settle measured in seconds.
#
# So the honest trade is: fixed-rate is smooth and matches training, wait-for-
# arrival is easier to reason about on hardware and lets step mode gate every
# motion. Nothing below makes the wait cheap; it only makes it shorter.
# -----------------------------------------------------------------------------
# Creep correction — why the arm used to move in stop-go-stop-go bursts
# -----------------------------------------------------------------------------
# The convergence above is right; the way it was applied was not. move_to() ran
# up to STEP_CONVERGE_PASSES separate settle() calls per policy step, and every
# settle() waits for the arm to come to a COMPLETE stop — SETTLE_STILL_HOLD_S =
# 0.30 s with no motion at all. So one policy step executed as: lunge, dead stop,
# twitch, dead stop, twitch, dead stop. Three full stops per step, by
# construction, which is exactly the "moves a certain distance then does some
# fine motions" that shows up on the robot. Logged steps are ~39 mm and pass 1
# lands well short of that, so passes 2 and 3 fired on essentially every step.
#
# The fix is not a looser tolerance — that would just leave the step short. It is
# to stop waiting for a dead stop before correcting. A Cartesian impedance
# controller accepts a new equilibrium at any instant; it has no notion of the
# arm being "between" commands. So when the arm STALLS short of the target —
# still for CREEP_STALL_HOLD_S, a quarter of the full hold — the lead is
# lengthened and republished right there, and the arm resumes from where it
# stands. The correction becomes a decelerating creep onto the target inside one
# continuous motion, with a single full stop at the end.
#
# That last stop is not negotiable: cp2/cp3 are single-frame Markov policies
# trained on states where the sim robot had fully reached its waypoint, so the
# observation that follows a step must be of a stationary arm.
# The nudge fires off a STALL — the arm stopped — and not off deceleration,
# which would be the obvious way to remove the pause entirely. Deceleration was
# rejected on purpose: while the arm is still travelling, (target - position)
# overstates what is left to do, because the command already outstanding is going
# to close most of it. Leading by a fraction of that overstated error commands
# past the target, and the correction for an overshoot is a REVERSAL — a twitch,
# which is the exact thing being removed. Predicting the landing point instead
# would need the controller's time constant, which is not identified. Waiting for
# the stall costs CREEP_STALL_HOLD_S of dead time and buys an error term that
# needs no prediction at all.
CREEP_STALL_HOLD_S = 0.05    # stillness that means "stalled", not "arrived"
CREEP_GAIN = 0.6             # fraction of the remaining error added per nudge
CREEP_MAX_NUDGES = 12        # backstop; MAX_COMMAND_LEAD_M and the settle
                             # timeout bound this too

# Minimum lead increment when a nudge produced NO MOTION AT ALL.
#
# A proportional nudge is the right shape while the arm is still responding, and
# useless once it is not: if the arm is 3 mm from the target and will not move
# until the equilibrium is ~17 mm past it, then adding 0.6 x 3 mm buys 1.8 mm of
# an 14 mm gap, and the next nudge buys 1.8 mm again. Measured, that is a dozen
# nudges to finish the last few millimetres, each paying its own latency wait —
# the last mile costing more than the whole rest of the move.
#
# No motion is a different piece of information from a short move: it says the
# lead is under the break-away threshold, and nothing about how far under. So it
# gets an absolute floor instead of a proportional one, and the arm walks up to
# break-away in a bounded number of steps regardless of how small the remaining
# error is.
#
# 10 mm, and the size is the whole point — 4 mm was tried first and measured to
# do NOTHING. The floor has to be a real fraction of the standing offset it is
# walking up to (~17 mm on this robot) or the settle times out before it gets
# there: on a 25 mm standing offset with 6 mm to travel, a 4 mm floor timed out
# after 9 nudges still 5.9 mm short, and a 10 mm floor converged in 4 nudges and
# 1.9 s. Across the plant sweep the same change cut the worst plant's mid-move
# dead time from 585 ms to 253 ms and its reversals from 1.9 per step to 0.6.
CREEP_BREAKAWAY_M = 0.010
# Stall detection is armed by SEEING THE ARM MOVE, not by a fixed delay. A fixed
# delay long enough to cover the worst round trip is dead time on every nudge;
# "it moved, then it stopped" is unambiguous the moment it happens. The delay
# survives only as the fallback for a command that produces no motion at all —
# already at the equilibrium, or blocked — where there is no motion to wait for.
#
# It is a CEILING on a self-calibrating gate, not a fixed wait. Fixed is the
# wrong shape: short enough to be cheap (0.12 s) and a 0.15 s round trip nudges
# before the arm has moved at all, applies two leads to one error and overshoots
# by 22 mm with a reversal to correct it — measured. Long enough to be safe
# (0.30 s) and every nudge that fails to break the arm loose pays 0.30 s of dead
# time, which on the last few millimetres of a step is most of the step —
# measured too, at ~1 s per step.
#
# So the gate is the round trip the link has actually shown, once one command has
# been seen to land; this only bounds it before that and if the link is slow.
CREEP_DEAD_S = 0.30
CREEP_LATENCY_MARGIN_S = 0.02   # added to the observed round trip

# THERE IS DELIBERATELY NO CAP TYING THE LEAD TO THE REMAINING ERROR.
#
# One was written and removed, and it is worth saying why, because it is an
# appealing idea: the lead accumulates while the error shrinks, so a lead sized
# for a 39 mm step is oversized for the 4 mm at the end of it, and bounding it by
# "what the remaining distance can absorb" reads as obviously right.
#
# It is not right. The lead a move needs is set by the controller's break-away
# threshold, which has nothing to do with how far is left to go — an arm 4 mm
# from its target with a 25 mm standing offset needs a lead of 25 mm, and a cap
# that forbids it deadlocks. Swept across the plants consistent with this robot's
# measurements, no slack value was uniformly good: 20 mm fixed the low-friction
# case and made a 25 mm standing offset fail to converge on every single step;
# 35 mm fixed that and broke the low-friction case instead. That is the signature
# of a constant tuned to a plant, and the plant is not identified. See the
# MAX_COMMAND_LEAD_M note above for the same conclusion reached the same way.
#
# What bounds the lead instead is MAX_COMMAND_LEAD_M — how far the equilibrium
# may sit ahead of where the arm actually IS — which is a statement about how
# hard the controller is asked to pull, and true regardless of which plant model
# is right.

# Feed-forward lead along the direction of travel.
#
# The vector droop estimate above cannot do this job, and that is the structural
# reason three passes were needed on EVERY step rather than converging away after
# the first few. A standing offset is a property of a POSE; the shortfall that
# makes a commanded move fall short is a property of a DIRECTION — friction
# opposes travel. Each policy step travels somewhere new, so a vector learned on
# the last step's direction is stale, and can point the wrong way outright when
# the policy reverses. Carrying it forward then costs a correction instead of
# saving one.
#
# A scalar magnitude applied along the current move's own direction cannot point
# the wrong way. At worst it is too long or too short along an axis the arm is
# travelling anyway, and the creep closes the difference. That is the whole
# reason this is a scalar: it exists only to make the FIRST command land close so
# few nudges are needed, and correctness never rests on it.
GAIN_BETA = 0.35             # EMA weight on each move's measured gain
GAIN_MIN = 0.10              # floor; see observe_move

# A GAIN IS ONLY A GAIN IF THE ARM HAS FINISHED MOVING, and `--control rate` by
# definition never lets it.
#
# observe_move divides what a command achieved by what it asked for. settle()
# waits for the arm to stop before reading that, so the division is honest.
# The fixed-rate loop dwells for a fixed period and reads whatever the arm has
# reached by then — which with tau ~ 0.12 s against a 0.15 s period is 71% of
# the exponential, so a true gain of 0.85 measures as ~0.46. The estimator then
# inverts it, and 1/0.46 - 1 asks for a lead of 1.2x travel where the honest
# answer is 0.18x.
#
# Measured on hardware, minutes apart on the same arm: homing (which waits)
# reported gain 0.85, while the rate loop reported 0.10-0.30 and drove the lead
# to 171 mm on 25 mm policy steps — nearly 7x the move it was leading.
#
# So the fixed-rate loop no longer teaches the estimator; it only reads it. The
# lead then comes from moves that actually settled — homing before every
# episode, and any settle/step-mode run — and `lead_for`'s travel/scale term
# takes it down to the right size for a 25 mm step. Rate mode keeps its
# break-away, which is what covers a small step against a stall band and is
# driven by the arm being stuck rather than by any gain estimate.
# The cap that follows from that. A lead may be at most 1.25x the move it is
# leading PLUS a fixed allowance, and the two terms are not interchangeable:
#
#   the ratio    1.25x is what a gain of 0.444 asks for, and 0.44 is the worst
#                this arm has ever honestly measured. Past that is not a stiff
#                arm, it is the broken measurement above.
#   the offset   a multiplicative gain CANNOT represent the standing offset —
#                this controller settles ~17 mm short of any target, so an 8 mm
#                command against it moves the arm not at all, and the lead that
#                gets it moving is necessarily larger than the step. Capping by
#                ratio alone starves exactly those small moves: measured, an
#                8 mm step then fell 27.5 mm short of its goal over 60 ticks.
#
# Together they bound the 171 mm lead seen on 25 mm steps down to 51 mm, while
# leaving a small step the headroom it needs to break away at all.
#
# Under-leading is self-correcting — the loop is closed, so the next observation
# simply sees a shorter step. Over-leading is not: the policy fights it. That
# asymmetry is why this is a cap and not a tuned gain.
MAX_LEAD_TRAVEL_RATIO = 1.25
LEAD_STALL_ALLOWANCE_M = 0.020
TRAVEL_LEAD_MIN_M = 0.005    # moves shorter than this do not inform the estimate

# 80 and not the evaluator's 50 (dagger/evaluator.py EvalParams.max_steps).
# The real arm needs more of them for the same motion: it executes a fraction
# of each commanded step — measured 0.3 of the rotation and 0.4-0.8 of the
# translation under --control rate — where sim executes all of it, so an
# approach that converges in 30 sim steps has been taking 40-60 here and
# timing out at 50 with the gripper still short of the object.
MAX_POLICY_STEPS = 80

# handover_sim2real/config.py: POLICY.TIME_ACTION_REPEAT = 0.15 against
# SIM.TIME_STEP = 0.001, i.e. 150 substeps per policy step. That is the rate the
# policy was trained and evaluated at, so it is the default for --control rate
# rather than anything tuned here.
RATE_CONTROL_HZ = 1.0 / 0.15

# Displacement over one tick below which the arm counts as not having moved, and
# the fixed-rate loop starts adding break-away lead. Well above the pose noise
# and well below anything a real step achieves, so it separates "stuck" from
# "slow" rather than firing on both.
RATE_STUCK_M = 0.0005

# How much of a tick's motion has to lie ALONG the heading that was commanded
# before that tick counts as a measurement of the controller's gain. cos >= 0.5
# is 60 degrees, which is generous: real under-travel is collinear with the
# command and only perpendicular drift eats into it.
#
# THIS IS THE GUARD THAT KEEPS THE LEAD FROM RUNNING AWAY, and the failure it
# stops was measured on hardware, not imagined. `observe_move` reads
# (achieved / commanded) as "the fraction of a commanded displacement this arm
# executes". That is only what it means if the arm was trying to execute it.
# Near the object the policy reverses constantly, so the arm spends a tick
# unwinding the PREVIOUS heading: it travels 20 mm and projects 1.7 mm onto the
# heading it was just given. Read as a gain that is 0.06, and the estimator
# answers with a lead of fifteen times the step.
#
# It compounds, because a larger lead makes `commanded` larger while `achieved`
# stays governed by the whipsaw, so the next sample is smaller still. Observed
# in one 58-step episode: gain 0.78 -> 0.10 (the floor) and the lead 29 mm ->
# 160 mm, with the arm overshooting its target by up to 7.4 mm and reversing
# between consecutive policy steps. A sign test alone does not catch it — the
# samples that do the damage are small and POSITIVE.
RATE_ALIGN_FRAC = 0.5

# HOW MUCH LEAD THE FIXED-RATE LOOP MAY USE. Unlike settle's cap this is a
# stability condition, not a sanity bound, and it is what stops the arm buzzing
# once the gripper is close to the object.
#
# settle() can afford a generous lead because it checks arrival: an over-led
# first command is walked back by the creep nudges inside the same move. The
# fixed-rate loop has no arrival test at all, by design. It is therefore a bare
# proportional loop — each tick commands (target - now) * (1 + r), the arm
# executes a fraction g of that within the dwell, and the error left over scales
# by
#
#     e_next / e = 1 - g * (1 + r)
#
# every tick. Measured on this arm, g spans 0.44 to 0.85 depending on direction
# and load. Evaluated across that band:
#
#   r = 1.25  (MAX_LEAD_TRAVEL_RATIO)   -0.91 at g = 0.85 — the error flips sign
#             every tick and sheds only 9% of itself per flip, which at 6.7 Hz
#             is about a second of visible oscillation per correction. Near the
#             object the policy re-injects error faster than that, so it never
#             dies down. This is the shaking, and nothing about it needs a bug:
#             it is what this cap does against a well-behaved arm.
#   r = 5.25  what a 5 mm travel gets once `s` has grown, because the cap's
#             fixed LEAD_STALL_ALLOWANCE_M term stops shrinking with the move:
#             1.25*5mm + 20mm. -1.75 to -4.31, i.e. divergent — the equilibrium
#             slams from one side to the other against clamp_command_lead.
#   r = 0.5   -0.27 to +0.34: the error at least halves every tick at BOTH ends
#             of the measured band. Both bounds are tight, so this is the middle
#             of the usable interval and not a value tuned to one run.
#
# The stall allowance is dropped here for the same reason. It exists because a
# STATIC target sits behind this controller's ~17 mm standing droop, so a small
# absolute command moves the arm not at all. The rate loop does not command
# static targets: every tick rebuilds the target as a delta off a freshly
# measured pose, so the droop is in the anchor and the measurement alike and
# cancels. What is left is the dwell-limited lag, which is multiplicative. A
# genuine stall — stiction, a blocked joint — is still covered, by break-away in
# RateCommander.command, which fires on the arm not moving and clears the moment
# it does, instead of being paid on every step whether or not it is needed.
RATE_MAX_LEAD_RATIO = 0.5

# How long each step-mode iteration idles pumping the 3D window's events before
# recomputing perception. One iteration of perception + policy is ~50 ms
# (measured: tripod get_frames 33 ms, hand segmentation 24 ms at 384 px overlapped
# with it, full observe() 33-47 ms, policy cloud assembly free), so without this
# the viewer is pumped at ~20 Hz and a trackpad drag is sampled too coarsely to
# orbit smoothly. Any keypress breaks the pump immediately, so this never delays
# SPACE.
#
# It applies ONLY when a 3D window actually exists AND --step-mode is on. Both
# guards matter. In continuous mode the loop rate is the control rate and must
# not be traded away for a debug view. And without --show-cloud there is no
# window to pump — tick() and drain_keys() return immediately — so the pump
# would be 200 ms of idle per iteration buying nothing, which is 4x the loop's
# entire real workload and drops the camera preview from ~20 Hz to ~4 Hz.
VIEWER_PUMP_S = 0.20

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
# largest_component / normalize_mask moved to pointcloud_multicam, which applies
# them inside HandSegmenter so every camera's mask is cleaned identically.


# overlay_mask lives in pointcloud_multicam alongside normalize_mask, so the
# runner and test_perception_viz render the mask identically.


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


def measure_ee_offset_z(client, timeout_s: float = 5.0) -> Optional[float]:
    """Read the panda_hand -> published-frame z offset off the robot.

    WHY THIS IS MEASURED AND NOT A CONSTANT. It used to be a constant, correct
    at the time it was written, and it went stale without a single error: the
    robot's configured end effector changed to the Franka Hand TCP, `F_T_EE`
    became (0, 0, 0.1034), and `/cartesian_pose` started publishing a frame
    103.4 mm ahead of `panda_hand` along the approach axis.

    Nothing complains, because the observation and the commanded target are
    BOTH in the wrong frame and stay self-consistent. What breaks is the one
    relationship that is not: where the point cloud sits relative to the
    gripper. An object just beyond the fingertips (panda_hand z ~ 0.16) renders
    at z ~ 0.06 — between the finger boxes — and the policy closes about 10 cm
    short of it, which is exactly how this was found on hardware.

    `F_T_EE` is published every control cycle and is the authority, so the
    constant is only ever a fallback for a robot that does not publish it.

    Returns None when the topic never arrives or the transform is not a pure
    z offset — the latter because this whole correction is `Tz(-dz)`, and
    silently applying it to a transform with x/y translation would trade a
    known error for a subtler one.
    """
    import roslibpy

    got: dict[str, Any] = {}
    topic = roslibpy.Topic(client, FRANKA_STATE_TOPIC, FRANKA_STATE_TYPE)
    try:
        topic.subscribe(lambda m: got.setdefault("msg", m))
        t0 = time.time()
        while "msg" not in got and time.time() - t0 < timeout_s:
            time.sleep(0.05)
    finally:
        try:
            topic.unsubscribe()
        except Exception:
            pass

    if "msg" not in got:
        return None
    raw = got["msg"].get("F_T_EE")
    if raw is None or len(raw) != 16:
        return None

    # libfranka ships 4x4 transforms COLUMN-major.
    F = np.asarray(raw, dtype=np.float64).reshape(4, 4).T
    tx, ty, tz = F[:3, 3]
    if abs(tx) > 1e-6 or abs(ty) > 1e-6:
        print(f"[frames] F_T_EE has x/y translation ({tx * 1e3:.1f}, "
              f"{ty * 1e3:.1f}) mm, which a z offset cannot express. "
              "Falling back to --ee-offset-z.")
        return None
    return float(tz)


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


def clamp_command_lead(T_command: np.ndarray, T_current: np.ndarray,
                       max_lead_m: float = MAX_COMMAND_LEAD_M) -> np.ndarray:
    """Keep a commanded equilibrium within max_lead_m of the arm's actual pose.

    Droop compensation deliberately commands past the target, and with G < 1 the
    required overshoot grows as the arm works outward. This is the safety bound on
    that: it limits how hard the impedance controller is ever asked to pull,
    without capping the offset itself (which legitimately grows — see
    MAX_COMMAND_LEAD_M).
    """
    lead = T_command[:3, 3] - T_current[:3, 3]
    n = float(np.linalg.norm(lead))
    if n <= max_lead_m or n < 1e-9:
        return T_command
    out = T_command.copy()
    out[:3, 3] = T_current[:3, 3] + lead * (max_lead_m / n)
    return out


def clip_lead_to_error(lead: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Forbid a lead that points against the way the arm still has to go.

    A lead exists to get the arm somewhere; one aimed the other way is wrong
    whatever model produced it, so this needs no assumption about the controller
    and cannot be tuned incorrectly.

    It is here because of a real deadlock, and the deadlock is not obvious. The
    lead accumulates while the arm falls short, which is right. But if a move
    then OVERSHOOTS, the error reverses while the accumulated lead does not: the
    equilibrium is still tens of millimetres beyond the target, which is still
    ahead of the arm, so the controller keeps pulling the arm further the wrong
    way. Nudging shrinks the lead by only a fraction of a now-small error, so it
    takes more nudges to walk the equilibrium back behind the arm than the nudge
    budget allows. Measured: 8.4 mm past the target, all twelve nudges spent,
    never settling — and since the estimate only learns from settled moves, every
    later step repeated it exactly. Twelve commands per step, forever.

    Clipping the opposing component puts the equilibrium at the target at worst,
    which the arm can always fall short of in the correct direction.
    """
    n = float(np.linalg.norm(err))
    if n < 1e-12:
        return lead
    u = np.asarray(err, dtype=np.float64) / n
    along = float(np.asarray(lead, dtype=np.float64) @ u)
    if along >= 0.0:
        return lead
    return lead - along * u     # remove only the opposing part


def freeze_arm(pub, seq: int, times: int = 3) -> int:
    """Re-command the equilibrium at the arm's MEASURED pose. Returns next seq.

    With the equilibrium on top of the arm the position error is zero, so the
    controller generates no restoring force and the arm stops where it stands.
    That is the entire mechanism, and its limits follow from it: the arm is not
    braked and not stiffened, so it will still sag by the standing droop and can
    still be pushed. Published `times` over rather than once because a single
    dropped websocket frame would leave the previous target — the one we are
    trying to abandon — as the live command.
    """
    if pub is None or current_msg is None:
        return seq
    T_now = pose_msg_to_matrix(current_msg)
    for _ in range(times):
        pub.publish(roslibpy.Message(
            matrix_to_pose_msg_like(current_msg, T_now, seq)))
        seq += 1
    return seq


def clamp_target_pose(T: np.ndarray) -> np.ndarray:
    T = T.copy()
    T[0, 3] = max(float(T[0, 3]), MIN_TARGET_X_M)
    T[2, 3] = max(float(T[2, 3]), MIN_TARGET_Z_M)
    return T


# -----------------------------------------------------------------------------
# Observation assembly
# -----------------------------------------------------------------------------
# build_bc_point_tensor moved to pointcloud_multicam.build_policy_cloud, which
# is now the single place the [1024, 5] tensor is assembled for one camera or
# several. The layout is unchanged: 896 object rows then 128 hand rows, channel
# 3 = ycb_flag, channel 4 = hand_flag.


# HOW FAR THE OBJECT IS FROM THE GRIPPER, in the frame the policy sees.
# Measured over run 19's own training clouds: the object centroid sits a median
# 11 cm from the panda_hand origin, p95 23 cm of extent. That is the range every
# head in the network was fitted on, and the gripper head in particular
# saturates outside it — pushing a training cloud 10 cm further out takes its
# logit from -3.8 to +17.6, which is the regime a real run reports when it will
# not close. So this number decides between two completely different problems:
# the arm has not arrived (large, and the refusal is CORRECT), or the cloud is
# misplaced relative to the gripper (small in reality, large here).
SIM_OBJECT_RANGE_M = 0.11


# The other half of the shape. Run 19's close states have an object EXTENT of
# 11 cm median, 23 cm at p95 — a YCB object. A 40 cm carton is outside that
# whatever its distance, and distance alone cannot show it.
SIM_OBJECT_EXTENT_M = 0.23


def _range_txt(xyz, extent: bool = False) -> str:
    if xyz is None or len(xyz) < 3:
        return ""
    a = np.asarray(xyz, np.float64)
    d = float(np.linalg.norm(np.median(a, axis=0)))
    out = f" @{d*100:4.1f}cm" + ("!" if d > 4.0 * SIM_OBJECT_RANGE_M else "")
    if extent:
        e = float(np.ptp(a, axis=0).max())
        out += f" x{e*100:4.1f}cm" + ("!" if e > SIM_OBJECT_EXTENT_M else "")
    return out


def _grip_txt(adapter) -> str:
    """The close decision as a NUMBER on the HUD, not just as a behaviour.

    `grip` below 0 is a close. Reference points, measured on each policy's own
    close-labelled training states: phase-4 run 19 sits at -1.5 when it means
    it, regrasp run 19 at -3.8. So a reading hovering at -0.2 while the object
    is between the fingers is a marginal decision, and +3 is a refusal — and
    those want opposite fixes.
    """
    g = getattr(adapter, "grip_logit", None)
    if g is None or g != g:                    # None or NaN
        return ""
    return f"  grip {g:+.2f}" + ("  CLOSE" if g < 0 else "")


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
        self._client = client
        # UNIQUE PER PROCESS. The counter alone restarts at 0 every run, so two
        # runs sent byte-identical goal ids. actionlib's ActionServer discards a
        # goal whose id is already in its status list, so a repeat can be dropped
        # in silence — and this interface has no action client to notice. The pid
        # costs nothing and removes the class of bug.
        self._tag = f"my_policy_runner_{os.getpid()}"
        self._grasp = self._move = self._homing = self._stop = None
        self._seq = 0
        if self.enabled:
            self._grasp = roslibpy.Topic(
                client, GRIPPER_GRASP_GOAL_TOPIC, GRIPPER_GRASP_GOAL_TYPE)
            self._move = roslibpy.Topic(
                client, GRIPPER_MOVE_GOAL_TOPIC, GRIPPER_MOVE_GOAL_TYPE)
            self._homing = roslibpy.Topic(
                client, GRIPPER_HOMING_GOAL_TOPIC, GRIPPER_HOMING_GOAL_TYPE)
            self._grasp.advertise()
            self._move.advertise()
            self._stop = roslibpy.Topic(
                client, GRIPPER_STOP_GOAL_TOPIC, GRIPPER_STOP_GOAL_TYPE)
            self._homing.advertise()
            self._stop.advertise()

    def server_is_up(self, timeout_s: float = 3.0) -> bool:
        """Is anything actually listening for grasp goals?

        WORTH CHECKING, BECAUSE THE FAILURE IS SILENT. Publishing a goal to a
        topic no action server has subscribed to succeeds: rosbridge takes the
        message, nobody acts on it, and nothing anywhere reports a problem. The
        run then looks normal right up to the CLOSE that does not happen — with
        a human holding an object, which is an expensive place to find out.

        The state this catches is real and easy to reach: launching the
        controller WITHOUT load_gripper:=True leaves /franka_gripper/joint_states
        publishing (the joint_state_publisher source list still names it) while
        every grasp/move/homing topic is absent. So the width readback looks
        healthy and the commands go nowhere.

        Testing the status topic rather than the goal topic is the point.
        actionlib publishes a GoalStatusArray continuously while a server is up,
        so one message proves a server exists; the goal topic would show up as
        soon as WE advertised it, which proves only that we are running.
        """
        if not self.enabled:
            return False
        seen = []
        topic = roslibpy.Topic(self._client, GRIPPER_GRASP_STATUS_TOPIC,
                               GRIPPER_STATUS_TYPE)
        try:
            topic.subscribe(lambda _msg: seen.append(1))
            t0 = time.time()
            while not seen and time.time() - t0 < timeout_s:
                time.sleep(0.05)
        finally:
            try:
                topic.unsubscribe()
            except Exception:
                pass
        return bool(seen)

    def home(self) -> None:
        """Calibrate the finger travel. Needed once per gripper power cycle.

        franka_gripper.launch does NOT do this for you, and until it has run the
        reported width is uncalibrated — which matters here beyond the grasp
        itself, because `read_gripper_norm` feeds both robot_state[25] and the
        position of the finger exclusion boxes. It takes a few seconds and moves
        the fingers through their full range, so it is opt-in rather than
        automatic: nothing should move at startup that the flags did not ask for.
        """
        if not self.enabled:
            print("[gripper] homing skipped (disabled)")
            return
        self._seq += 1
        self._homing.publish(roslibpy.Message(_action_goal_msg(
            f"{self._tag}_homing_{self._seq}", {})))
        print("[gripper] homing goal sent (fingers will open and close fully)")

    def close(self) -> None:
        if not self.enabled:
            print("[gripper] CLOSE commanded (disabled — pass --enable-gripper to act)")
            return
        self._seq += 1
        self._grasp.publish(roslibpy.Message(_action_goal_msg(
            f"{self._tag}_grasp_{self._seq}",
            {
                "width": GRASP_WIDTH_M,
                "epsilon": {"inner": GRASP_EPSILON_INNER,
                            "outer": GRASP_EPSILON_OUTER},
                "speed": GRASP_SPEED,
                "force": GRASP_FORCE,
            })))
        print(f"[gripper] grasp goal sent (width={GRASP_WIDTH_M} force={GRASP_FORCE}N)")

    def wait_for_width(self, want_m: float, timeout_s: float = 8.0,
                       tol_m: float = 0.004) -> bool:
        """Block until the fingers reach `want_m` each, or give up.

        Goals go out over the goal TOPIC, so there is no result to wait on — the
        only feedback available is /franka_gripper/joint_states. Watching it is
        the difference between knowing a command worked and assuming it did,
        which for the gripper is the difference between the policy being told
        the truth about robot_state[25] and being told 1.0 because nothing
        contradicted it.
        """
        if not self.enabled:
            return False
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if (gripper_finger_m is not None
                    and abs(gripper_finger_m - want_m) <= tol_m):
                return True
            if poll_stop():
                return False
            time.sleep(0.05)
        return False

    def prepare(self, home_first: bool) -> bool:
        """Put the gripper in the state the policy was trained to see: OPEN.

        WHY THIS EXISTS. Nothing used to open the gripper at startup, and the
        episode ENDS on a close — so every run after one that actually grasped
        began with the fingers shut. Two things then go wrong at once and
        neither announces itself:

          * robot_state[25] is the normalised finger position, and it reads 0.
            The policy was trained on approaches where it is 1.0 throughout,
            because the episode terminates at the close. Being told the gripper
            is already shut from step 0 is off-distribution before the first
            action, which is what "the arm moves in a weird way and not toward
            the object" looks like.
          * finger_q_m is 0, so the two finger exclusion boxes MEET (inner_y is
            -0.6 mm) and cover the whole jaw volume. Object points between the
            jaws are then deleted — the exact failure GraspRegion exists to
            prevent, arriving through the gripper state instead.

        Homing runs AFTER the open, not before. Homing drives the fingers
        through their full range, and doing that while they are clamped on an
        object is how a homing goal quietly fails — which is the other half of
        "it homes the first time and never again".
        """
        if not self.enabled:
            return False
        target = GRIPPER_MAX_FINGER_M
        self.open()
        if not self.wait_for_width(target):
            print(f"[gripper] open did not complete: width now "
                  f"{'unknown' if gripper_finger_m is None else f'{gripper_finger_m*1000:.1f} mm'}",
                  flush=True)
        if home_first:
            self.home()
            if not self.wait_for_width(target, timeout_s=15.0):
                print("[gripper] homing did not finish inside 15 s — the "
                      "fingers may be obstructed, or the goal was dropped.",
                      flush=True)
                return False
            print("[gripper] homed", flush=True)
            # Homing ends open, but say so explicitly rather than assume it.
            self.open()
            self.wait_for_width(target)
        return self.wait_for_width(target, timeout_s=1.0)

    def stop(self) -> None:
        """Abort whatever the gripper is doing. Leaves the fingers where they are.

        franka_gripper's stop action cancels the goal in flight, which matters
        during an abort for one case in particular: a grasp that is closing on
        something it should not be closing on.
        """
        if not self.enabled or self._stop is None:
            return
        self._seq += 1
        self._stop.publish(roslibpy.Message(_action_goal_msg(
            f"{self._tag}_stop_{self._seq}", {})))

    def open(self) -> None:
        if not self.enabled:
            return
        self._seq += 1
        self._move.publish(roslibpy.Message(_action_goal_msg(
            f"{self._tag}_move_{self._seq}",
            {"width": 2 * GRIPPER_MAX_FINGER_M, "speed": GRASP_SPEED})))

    def shutdown(self) -> None:
        for topic in (self._grasp, self._move, self._homing, self._stop):
            if topic is not None:
                try:
                    topic.unadvertise()
                except Exception:
                    pass


class ObservationNotUsable(RuntimeError):
    """The adapter cannot act on THIS frame, but the run is fine.

    THE DIFFERENCE BETWEEN THIS AND A BUG IS THE WHOLE POINT. `fused.usable` is
    "both classes non-empty" — one point clears it — while anything that wants a
    centroid needs three, and the regrasp adapter needs one to anchor its
    direction. A class that came out of `reject_arm_clusters` with two points
    left therefore satisfies the loop's gate and not the adapter's, and the
    adapter used to answer that with a bare RuntimeError, which killed the
    process. Mid-session that costs the whole recording.

    Holding the frame is the established response to a class that came out too
    thin — it is what `[arm:all-arm]` does, and for the same reason: a stall is
    visible and recoverable, while acting on two points is a silently wrong
    command. Worse here, because the regrasp direction is latched ONCE per
    episode and would steer every remaining step of the attempt.

    Raised by the adapter, caught in the predict block, and it costs no step.
    """


def _warn_hold(msg: str, every_s: float = 1.0) -> None:
    """Rate-limited notice that a frame was skipped. Once a second is enough.

    At the control rate an unrate-limited print is seven lines a second of the
    same sentence, which buries the step log it is supposed to annotate.
    """
    now = time.time()
    if now - getattr(_warn_hold, "_last", 0.0) >= every_s:
        _warn_hold._last = now
        print(f"[hold] {msg}", flush=True)


def read_gripper_norm(assume_open: bool) -> float:
    """Normalized finger position for robot_state[25]: 1 = open, 0 = closed.

    Falls back to 1.0 when /franka_gripper/joint_states is not being published —
    the policy only ever sees an open gripper during approach anyway, since the
    episode ends at the close.
    """
    if gripper_finger_m is None:
        return 1.0 if assume_open else 0.0
    return float(np.clip(gripper_finger_m / GRIPPER_MAX_FINGER_M, 0.0, 1.0))


def aperture_mm(finger_m: Optional[float]) -> Optional[float]:
    """One finger's travel -> the gap between the jaws, in millimetres.

    `gripper_finger_m` is the half-sum of the two prismatic joints, i.e. how far
    ONE finger has moved, which is what robot_state[25] normalises. The opening
    an object has to fit through is twice it. Kept as a named function rather
    than a `2 * 1000 *` at each call site because the factor of two between the
    two conventions is exactly the kind of thing that gets dropped once and then
    shows up as an object that measures 17 mm in the log and 34 mm with a
    caliper. None in, None out: before the first /franka_gripper/joint_states
    message there is no reading, and an empty CSV cell says that honestly where
    a 0.0 would read as a closed gripper.
    """
    return None if finger_m is None else 2000.0 * float(finger_m)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_hand_segmenter(device: str, checkpoint: Path):
    """The hand model alone. Its input transform is per camera now, so it is
    built inside HandSegmenter (see pointcloud_multicam.make_seg_preprocess)
    rather than fixed once here at 256."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Hand segmentation checkpoint not found: {checkpoint}")

    model = HandSegModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model = model.to(device)
    model.eval()
    return model


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
            "of its definition; rolling out without it produces garbage.\n\n"
            "It is per-run and cannot be borrowed from another checkpoint: the "
            "stats come from that run's own training aggregate, so cp2's "
            "normalizer describes a different dataset (wrist-only vs wlr) and "
            "would mis-scale every action.\n"
            "Fetch the one that belongs to this run, e.g. for cp3:\n"
            "  rsync -avP delftblue:/scratch/pradyunsharma/handover-sim2real/"
            "output/dagger_runs/dagger4_run16/best/normalization.npz \\\n"
            f"    {norm_path}")

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
def grip_logit_of(model) -> float:
    """The gripper logit behind the {0,1} bit `predict` returns, or NaN.

    THE BIT HIDES THE ONLY NUMBER THAT MATTERS WHEN IT WILL NOT CLOSE. `predict`
    thresholds `sigmoid(logit) > 0.5`, so a policy that is one millivolt from
    closing and one that is refusing outright produce the identical observation
    on the robot: an open gripper. Negative commands a close, and the magnitude
    says how sure it is — on run 19's own close-labelled states the mean is
    -3.8, so a reading near zero means "marginal" and +3 means "actively
    refusing", which are different problems with different fixes.
    """
    raw = getattr(model, "last_raw", None)
    if raw is None:
        return float("nan")
    return float(raw.reshape(-1, raw.shape[-1])[0, 6])


def policy_act(model, pc: np.ndarray, rs: np.ndarray, device: str) -> np.ndarray:
    """[7] deployable action: BCPolicy.predict denormalizes ch0..5 and
    hard-thresholds the gripper logit to {0, 1}. Same call BCRunner.act makes."""
    pc_t = torch.from_numpy(pc).float().unsqueeze(0).to(device)
    rs_t = torch.from_numpy(rs).float().unsqueeze(0).to(device)
    return model.predict(pc_t, rs_t)[0].cpu().numpy().astype(np.float32)


def warm_up_policy(model, device: str) -> None:
    """One dummy forward so the first real step isn't paying CUDA init."""
    policy_act(model,
               np.zeros((NUM_OBJECT_POINTS + NUM_HAND_POINTS, PC_CHANNELS),
                        dtype=np.float32),
               np.zeros(32, dtype=np.float32), device)


# -----------------------------------------------------------------------------
# Control
# -----------------------------------------------------------------------------
class DroopCompensator:
    """Feed-forward estimate of how far past a target to command.

    Two estimates live here, and which one is used depends on the control path:

      `d`  a standing offset VECTOR, used by the legacy multi-pass path
           (`--no-creep`). `compensate` applies it; `update` re-estimates it by
           integral feedback from what the last command achieved.

      `g`  the fraction of a commanded displacement the arm actually executes,
           used by the creep path. `lead_for` turns it into a lead; `observe_move`
           re-estimates it from what a move's first command achieved.

    A GAIN and not a lead magnitude, which is what this held first. A lead is a
    distance, so it is only ever right for the move it was learned on: correct
    for a 40 mm step, far too large for the 4 mm correction at the end of one,
    and unbounded above as steps grow. A gain is dimensionless. It says the same
    true thing about the controller at every scale, and `lead_for` recovers the
    right distance for whatever error is actually in front of it.

    They are deliberately not mixed. Disabled, all of it is a no-op: the creep
    still converges, it just starts each move from a zero lead and pays a nudge
    or two for it.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.d = np.zeros(3, dtype=np.float64)
        self.samples = 0
        # 1.0 means "executes what it is told", i.e. no lead. Starting there
        # rather than at a guess means the first move of an episode is commanded
        # honestly and the estimate is built from what it does, instead of the
        # arm being thrown at a lead nobody has measured yet.
        self.g = 1.0
        self.g_samples = 0
        self.s = 0.0           # lead magnitude for a full-step first command, m
        # The travel `s` was measured over. A lead magnitude is only meaningful
        # at the scale it was learned at, and until this existed nothing recorded
        # what that scale was — so `lead_for` applied a full-step lead to a 2 mm
        # correction. See lead_for.
        self.s_scale = 0.0

    def compensate(self, T_target: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return T_target
        T = T_target.copy()
        T[:3, 3] = T[:3, 3] + self.d
        return T

    def gain(self) -> float:
        """Fraction of a commanded displacement this arm actually executes."""
        return self.g if self.enabled else 1.0

    def lead_for(self, travel: np.ndarray,
                 max_ratio: float = MAX_LEAD_TRAVEL_RATIO,
                 stall_allowance: bool = True) -> np.ndarray:
        """The lead for the FIRST command of a move covering `travel`.

        `max_ratio` and `stall_allowance` are the caller's, because the two
        callers are not asking the same question. settle() wants the biggest
        lead that is not absurd and corrects the rest; the fixed-rate loop has
        no correction pass, so its cap has to keep the closed loop damped and
        its allowance has to be zero — see RATE_MAX_LEAD_RATIO. The defaults are
        settle's, so this reads unchanged there.

        A magnitude along the direction of travel, learned from what previous
        first commands needed. It is a magnitude and not a gain because a gain
        has to be re-applied at other scales to be useful and this one is not
        clean enough to survive that — see observe_move. It is a magnitude along
        TRAVEL and not a vector because friction opposes the direction of motion,
        so a vector learned on the last move points the wrong way as soon as the
        policy reverses, which during a handover it does constantly.

        NEVER LARGER THAN THE SCALE IT WAS MEASURED AT. `observe_move` refuses to
        learn from a move shorter than TRAVEL_LEAD_MIN_M — "a short move says
        nothing about the gain" — and for a long time nothing enforced the
        matching rule when APPLYING it. That asymmetry had a specific symptom.
        `go_home`'s refine is a move of a few millimetres, and its first command
        got the full lead learned over ~30 mm policy steps, which
        MAX_DROOP_COMP_M lets reach hundreds. The arm shot past home, the next
        pass shot back, and homing hunted for all six refine passes — while
        `observe_move` ignored every one of those moves as too short to learn
        from, so it could not correct itself out of it either.

        Reported from hardware as homing working the first time and hunting
        afterwards, which is exactly the shape of the bug: the first home runs
        with s = 0 because no episode has been flown yet.

        Scaling by travel/scale is the physically right relation, not a fudge:
        the lead that lands a move is `e * (1/g - 1)`, linear in the distance
        covered. Clamped at 1.0 rather than extrapolated up, because a magnitude
        that does not extrapolate is the property observe_move chose it for.
        """
        if not self.enabled or self.s <= 0.0:
            return np.zeros(3, dtype=np.float64)
        n = float(np.linalg.norm(travel))
        if n < 1e-9:
            return np.zeros(3, dtype=np.float64)
        mag = self.s
        if self.s_scale > 1e-9:
            mag *= min(1.0, n / self.s_scale)
        mag = min(mag, max_ratio * n
                  + (LEAD_STALL_ALLOWANCE_M if stall_allowance else 0.0))
        return mag * (np.asarray(travel, dtype=np.float64) / n)

    def observe_move(self, commanded0: float, shortfall: np.ndarray,
                     travel: np.ndarray) -> None:
        """Learn from what the move's FIRST command achieved.

        Every move measures the controller's under-travel directly, and this uses
        that rather than assuming it. The first command asked the arm to cover
        `commanded0` and it covered `e0 - shortfall`, so

            g = (e0 - shortfall) / commanded0

        is the fraction of a commanded displacement this arm actually executes,
        here, now, in this direction. Landing `e0` therefore wants a commanded
        `e0 / g`, i.e. a lead of `e0/g - e0`. That has the right fixed point for
        free: if the command already landed on target, g = e0/commanded0 and the
        formula returns the lead that command used.

        `commanded0` is what was ACTUALLY PUBLISHED, measured from the pose, not
        the lead this code intended. The two differ whenever clamp_command_lead
        truncates the command, and using the intended value there is a genuine
        runaway: g comes out too small, the ideal lead too large, the next
        command is truncated harder, and the estimate walks to its cap and stays
        there. Measured, that turned a 3-command step into a 20-command one.

        Everything about this is measured per move, which is the point. The
        previous version corrected by the raw shortfall instead — deliberately
        under-correcting so the estimate would approach from below and never
        overshoot — and that is safe but far too slow: it closes only a fraction
        g of the gap per move, so at g = 0.5 with the EMA on top it needed ten to
        fifteen moves. On hardware that reads as the corrections never going
        away, because an episode is fifty steps and the arm is still converging
        for a third of it.

        Only the along-travel component is used. The perpendicular part is real —
        gravity does not care which way the arm is going — but is not separable
        from the direction-dependent part in a single move, and carrying a
        mis-attributed vector forward is the failure this whole scheme exists to
        avoid. The creep re-derives it each move for the cost of a nudge.
        """
        if not self.enabled:
            return
        e0 = float(np.linalg.norm(travel))
        if e0 < TRAVEL_LEAD_MIN_M:
            return              # a short move says nothing about the gain
        u = np.asarray(travel, dtype=np.float64) / e0
        short_along = float(np.asarray(shortfall, dtype=np.float64) @ u)

        commanded = float(commanded0)
        if commanded <= 1e-6:
            return
        # Floored, not merely guarded against zero: an arm that barely moved
        # gives a gain near zero and so a lead near infinity, and the honest
        # reading of "it hardly moved" is "lead a good deal more", not "lead by
        # forty metres". The floor turns that into a bounded step the next move
        # refines. Capped at 1.0 because a controller that over-travels is not
        # something this compensation should try to exploit.
        g_obs = float(np.clip((e0 - short_along) / commanded, GAIN_MIN, 1.0))

        # Store the LEAD this move wanted, not the gain that implies it. The gain
        # is the honest way to compute it — but it is contaminated by the
        # standing offset (achieved = g*commanded - offset, so a measured gain is
        # always g - offset/commanded, an underestimate) and re-applying that
        # underestimate at a different scale over-leads badly. Measured on a
        # 17 mm offset it settled at 0.35 against a true 0.50, over-led the first
        # command, and cost seven commands a step walking the overshoot back.
        #
        # A lead magnitude does not extrapolate, so it cannot be wrong that way.
        # It is only ever applied to the FIRST command of a move, and first
        # commands are always one full policy step, which is the scale it was
        # measured at. The corrections after it accumulate from the actual error.
        ideal = e0 / g_obs - e0
        if ideal <= 0.0:
            return              # this move wanted to be aimed backwards; ignore
        self.s = ((1.0 - GAIN_BETA) * self.s
                  + GAIN_BETA * min(ideal, MAX_DROOP_COMP_M))
        # Track the scale alongside the magnitude, on the same EMA, so the two
        # always describe the same moves. `lead_for` needs it to avoid handing a
        # full-step lead to a millimetre-scale correction.
        self.s_scale = (1.0 - GAIN_BETA) * self.s_scale + GAIN_BETA * e0
        self.g = g_obs
        self.g_samples += 1

    def update(self, T_commanded: np.ndarray, T_measured: np.ndarray) -> None:
        """D = commanded - measured, EMA-blended and capped.

        A fixed point once compensation is exact: commanding target + d and
        landing on target measures d again, so the estimate holds steady.
        """
        if not self.enabled:
            return
        d_meas = T_commanded[:3, 3] - T_measured[:3, 3]
        self.d = (1.0 - DROOP_EMA_ALPHA) * self.d + DROOP_EMA_ALPHA * d_meas
        n = float(np.linalg.norm(self.d))
        if n > MAX_DROOP_COMP_M:
            self.d *= MAX_DROOP_COMP_M / n
        self.samples += 1

    def describe(self) -> str:
        if not self.enabled:
            return "droop compensation OFF"
        if self.g_samples:
            return (f"lead={self.s*1000:.0f} mm along travel (gain {self.g:.2f})"
                    f"  n={self.g_samples}")
        return (f"droop=({self.d[0]*1000:+.1f}, {self.d[1]*1000:+.1f}, "
                f"{self.d[2]*1000:+.1f}) mm  n={self.samples}")


class SettleResult(NamedTuple):
    settled: bool
    pos_err: float          # to the TARGET, not to the command — the log should
    rot_err: float          # show the standing offset rather than hide it
    next_seq: int
    nudges: int


def settle(pub, template_msg: dict, T_base_ctrl_target: np.ndarray, seq: int,
           timeout_s: float, droop: "DroopCompensator | None" = None,
           tol_m: float = SETTLE_POS_TOL_M, creep: bool = True) -> SettleResult:
    """Command a target and block until the arm has stopped ON it.

    cp2/cp3 are single-frame Markov policies trained on steps where the sim robot
    fully reached each waypoint before the next observation. Streaming targets at
    camera rate would feed them mid-motion states they never saw in training, so
    each target is held until the motion it caused is over.

    A Cartesian impedance controller only generates force through position error,
    so holding the arm against gravity requires a permanent offset — measured on
    this robot at ~17 mm, almost all of it -z — and the arm will therefore STALL
    short of any target it is simply told to go to. There are two ways to deal
    with that and this function does both, selected by `creep`:

      creep=True (default)
        Accept stalling as the normal end of a command rather than the end of
        the move. When the arm has been still for CREEP_STALL_HOLD_S but is not
        yet within `tol_m`, lengthen the lead and republish immediately. The arm
        resumes from where it stands, so the move reads as one decelerating
        approach instead of a series of twitches separated by dead stops.
        Convergence means still AND on target.

      creep=False
        The original behaviour: publish once, and declare convergence when the
        pose stops changing, wherever that happens to be. Getting the rest of
        the way was the caller's problem, and move_to() solved it by commanding
        again — which is what made the motion lumpy. Kept for comparison and as
        an escape hatch.

    Convergence is judged on TRANSLATION plus stillness. Rotation is measured and
    returned but does not gate, because the lead is translation-only (rotational
    droop measured 0.9 deg against 3.4-4.6 deg steps): gating on a residual this
    function cannot correct would turn every step into a timeout. This is not a
    regression — the stillness path, the one that actually fired before, ignored
    rotation entirely.
    """
    rot_tol = np.deg2rad(SETTLE_ROT_TOL_DEG)
    still_rot_tol = np.deg2rad(SETTLE_STILL_ROT_DEG)
    target_xyz = T_base_ctrl_target[:3, 3]

    T_start = pose_msg_to_matrix(current_msg) if current_msg is not None else None
    travel = (target_xyz - T_start[:3, 3]) if T_start is not None else np.zeros(3)

    if not creep:
        lead = (droop.compensate(T_base_ctrl_target)[:3, 3] - target_xyz
                if droop is not None else np.zeros(3))
    else:
        # The FIRST command is sized from the learned gain, which is honest
        # here and only here: the gain was measured on moves of exactly this
        # scale — a full policy step — so applying it at that scale is applying
        # it where it was identified. The corrections that follow accumulate
        # instead, because by then the error is small and the standing offset
        # dominates it, and a gain cannot represent an additive offset. Using the
        # gain all the way down under-leads the tail badly: measured, 10 commands
        # a step against 1.8.
        lead = droop.lead_for(travel) if droop is not None else np.zeros(3)


    def publish(seq: int) -> tuple[np.ndarray, int]:
        """Command target+lead, bounded so the equilibrium never sits more than
        MAX_COMMAND_LEAD_M ahead of where the arm actually is — the bound that
        governs how hard the controller pulls, and the one backstop that holds
        however wrong the lead gets."""
        T_cmd = T_base_ctrl_target.copy()
        T_cmd[:3, 3] = target_xyz + lead
        if current_msg is not None:
            T_cmd = clamp_command_lead(T_cmd, pose_msg_to_matrix(current_msg))
        pub.publish(roslibpy.Message(
            matrix_to_pose_msg_like(template_msg, T_cmd, seq)))
        return T_cmd, seq + 1

    T_command, seq = publish(seq)
    # What the FIRST command actually ASKED FOR, along the direction of travel,
    # read back off the published pose so a clamp is included ...
    u0 = travel / max(float(np.linalg.norm(travel)), 1e-12)
    commanded0 = float((T_command[:3, 3] - (T_start[:3, 3] if T_start is not None
                                            else T_command[:3, 3])) @ u0)
    first_rest = None           # ... and where it left the arm. See
                                # DroopCompensator.observe_move for why the
                                # estimate must come from these two and not from
                                # whatever the move ends up accumulating.
    t0 = t_pub = time.time()
    latency = None              # round trip, once the link has demonstrated one
    armed_at = t0 + CREEP_DEAD_S
    moved = False               # has the arm responded to the live command yet?
    dt = drot = float("inf")
    T_prev = None
    still_since = None
    settled = False
    nudges = 0

    while time.time() - t0 < timeout_s:
        time.sleep(SETTLE_POLL_S)
        # Checked here because this loop is where a policy step spends its time,
        # so this is where an abort has to be noticed to mean anything.
        if poll_stop():
            break
        if current_msg is None:
            continue
        T_now = pose_msg_to_matrix(current_msg)
        dt, drot = pose_error(T_now, T_base_ctrl_target)

        # There is deliberately NO in-tolerance early exit here. One used to
        # exist — "arrived, stop waiting" — and it was unsound: being within
        # tolerance at one sample says nothing about whether the arm is stopping
        # there or passing through at speed. It could never fire before, because
        # an uncompensated command always stalls short and never reaches
        # tolerance at all; with the lead sized to actually arrive, it fires
        # mid-transit, and returning there hands the policy an observation of a
        # moving arm — the exact thing this function exists to prevent. Measured
        # with an oversized CREEP_GAIN, it let a heavy overshoot report 2.5 mm.
        # Convergence is in tolerance AND stopped, below, with no shortcut.

        if T_prev is not None:
            step_t, step_r = pose_error(T_prev, T_now)
            if step_t < SETTLE_STILL_POS_M and step_r < still_rot_tol:
                if still_since is None:
                    still_since = time.time()
            else:
                if not moved:
                    # First motion after this command: the round trip is at most
                    # this, so later commands need not assume the worst. It is an
                    # over-estimate — it includes however long the arm took to
                    # move a measurable amount — which is the safe direction.
                    latency = time.time() - t_pub
                moved = True
                still_since = None      # moved again; restart the quiet window
        T_prev = T_now

        if still_since is None:
            continue
        quiet = time.time() - still_since
        if first_rest is None and quiet >= CREEP_STALL_HOLD_S:
            first_rest = T_now[:3, 3].copy()

        if not creep:
            if quiet >= SETTLE_STILL_HOLD_S:
                settled = True          # stopped == done, wherever it stopped
                break
            continue

        if dt < tol_m:
            if quiet >= SETTLE_STILL_HOLD_S:
                settled = True
                break
            continue

        # Stillness before the arm has reacted to the live command is latency,
        # not a stall, and nudging on it would apply two leads to one error and
        # overshoot. Once motion has been seen, a stop is a stop immediately.
        # This gates only the NUDGE: in tolerance and stopped is unambiguous
        # whenever it happens, and delaying that would be dead time for nothing.
        if not moved and time.time() < armed_at:
            continue

        # Stalled short. Lengthen the lead and go again, without waiting for the
        # full hold: the controller does not need the arm at rest to accept a new
        # equilibrium, and waiting for one is precisely what made the motion
        # stop-go-stop-go.
        if quiet >= CREEP_STALL_HOLD_S and nudges < CREEP_MAX_NUDGES:
            err_vec = target_xyz - T_now[:3, 3]
            step = CREEP_GAIN * err_vec
            if not moved and float(np.linalg.norm(step)) < CREEP_BREAKAWAY_M:
                # The last command moved the arm not at all, so the lead is under
                # break-away and the remaining error says nothing about by how
                # much.
                step = err_vec / max(float(np.linalg.norm(err_vec)), 1e-12)
                step = step * CREEP_BREAKAWAY_M
            lead = clip_lead_to_error(lead + step, err_vec)
            n = float(np.linalg.norm(lead))
            if n > MAX_DROOP_COMP_M:
                lead *= MAX_DROOP_COMP_M / n
            T_command, seq = publish(seq)
            nudges += 1
            t_pub = time.time()
            gate = (CREEP_DEAD_S if latency is None
                    else min(latency + CREEP_LATENCY_MARGIN_S, CREEP_DEAD_S))
            armed_at = t_pub + gate
            moved = False
            still_since = None
            T_prev = None

    # Learn only from a converged move: a timed-out one is still travelling, so
    # its lead is transit, not the offset the pose needs, and would corrupt the
    # estimate.
    if droop is not None and settled and current_msg is not None:
        if creep:
            if first_rest is None:      # converged before ever coming to rest
                first_rest = pose_msg_to_matrix(current_msg)[:3, 3]
            droop.observe_move(commanded0, target_xyz - first_rest, travel)
        else:
            droop.update(T_command, pose_msg_to_matrix(current_msg))

    return SettleResult(settled, dt, drot, seq, nudges)


class RateCommander:
    """Fixed-rate control: publish once per tick, never wait for arrival.

    This is the CVPR2023 loop (arXiv 2303.17592), transplanted. Its rollout is

        action = policy(obs)
        tjp    = IK(current_ee_pose @ delta)
        for _ in range(int(0.15 / 0.001)):      # POLICY.TIME_ACTION_REPEAT
            obs = env.step(tjp)

    — one command, a fixed 0.15 s of execution, then look again, with no test of
    whether the arm got there. The arm is permanently chasing a target that has
    already moved, which is what makes the motion continuous instead of
    stop-start, and undershoot needs no correction pass because the NEXT action
    is a fresh delta from wherever the arm actually reached. Error is absorbed by
    the next prediction rather than by a settling loop.

    WHAT IS PUBLISHED IS EXACTLY WHAT settle() PUBLISHES FIRST: target plus the
    droop lead along travel, bounded by clamp_command_lead. Fixed-rate mode is
    then precisely "settle's first command, then stop waiting", which keeps one
    definition of a well-formed command instead of two.

    THE LEAD IS NOT OPTIONAL HERE, and the reason is worth stating because
    dropping it looks safe. The target is rebuilt every tick from the MEASURED
    pose, so a command the arm is too stiff to execute does not accumulate: the
    equilibrium is re-placed at the same physical spot, the arm stays put, the
    observation does not change, the policy predicts the same delta, and the loop
    deadlocks. In settle() the creep nudges break that; here the standing lead is
    the only thing that does. It is why `--no-droop-compensation` and
    `--control rate` together are a bad combination near the object, where the
    deltas are smallest.

    The estimator keeps learning across ticks, from the same quantities settle
    uses: what the last command asked for along travel, and how far the arm got
    by the time the next tick came round.
    """

    def __init__(self, pub, hz: float, droop: "DroopCompensator | None"):
        self.pub = pub
        self.period = 1.0 / max(float(hz), 1e-3)
        self.droop = droop
        self._next_tick = None
        self._pending = None        # (commanded0, start_xyz, u0, want)
        # Extra lead accumulated while the arm is not moving. See _stuck below:
        # this is creep's break-away, spread across ticks instead of across
        # nudges inside one settle.
        self._stuck_lead = 0.0

    def reset(self) -> None:
        """Forget the tick clock and the outstanding command.

        Between episodes the arm is homed and the gripper re-opened, so the
        pending command's start pose is meaningless and feeding it to the droop
        estimator would attribute a homing move to a policy step. The tick clock
        has to restart too, or the first tick of the new episode inherits an
        overdue deadline and fires with no dwell at all.
        """
        self._next_tick = None
        self._pending = None
        self._stuck_lead = 0.0

    def command(self, T_base_ctrl_target: np.ndarray, seq: int) -> tuple[int, float]:
        """Publish one target. Returns (next_seq, seconds slept since last tick).

        The sleep is what sets the control period, and it is taken AFTER
        publishing so the arm is moving during it rather than after it. It is
        also the whole reason perception and inference cost nothing in this mode:
        they happen inside a period that would otherwise be idle, so anything
        under the period is free.
        """
        now = time.time()
        if current_msg is None or poll_stop():
            return seq, 0.0
        # The template is read live, exactly as move_to passes current_msg into
        # settle: it carries the frame_id and stamp fields the controller expects
        # and only the pose is overwritten.
        template = current_msg
        T_now = pose_msg_to_matrix(template)

        # Learn from the PREVIOUS tick before issuing the next: this is the only
        # moment we know both what was asked and what it achieved.
        if self._pending is not None and self.droop is not None:
            # THE TWO DISTANCES ARE NOT THE SAME AND observe_move NEEDS BOTH.
            # `want` is the displacement the policy actually asked for;
            # `commanded0` is what got published, which is `want` plus the lead
            # and possibly truncated by clamp_command_lead. The gain it derives
            # is (achieved / commanded), so passing the commanded distance as
            # the desired one would make every tick look like a perfect move and
            # freeze the estimate at its starting value.
            commanded0, start_xyz, u0, want = self._pending
            disp = T_now[:3, 3] - start_xyz
            moved = float(np.linalg.norm(disp))
            achieved = float(disp @ u0)
            # A TICK THE ARM SPENT GOING SOMEWHERE ELSE IS NOT A GAIN
            # MEASUREMENT. The test is whether the motion was MOSTLY the one
            # commanded — see RATE_ALIGN_FRAC — and not merely whether its sign
            # was right: the samples that walk the lead to its cap are small and
            # positive, an arm 80 degrees off the heading it was just handed.
            #
            # A true STALL is exempt and must stay so. There `moved` is under
            # the pose noise, the ratio is decided by nothing, and the sample is
            # real evidence that this arm does not execute what it is told —
            # which is how the estimator learns a stiff arm at all. Break-away
            # below handles the rest of that case.
            aligned = moved < RATE_STUCK_M or achieved >= RATE_ALIGN_FRAC * moved
            if commanded0 > 1e-6 and want > 1e-6 and aligned:
                self.droop.observe_move(commanded0, u0 * (want - achieved),
                                        u0 * want)
            # BREAK-AWAY, and it is load-bearing on small steps. observe_move
            # ignores any move under TRAVEL_LEAD_MIN_M (5 mm) because a short
            # move is a terrible gain estimator — right for settle(), where the
            # creep nudges get a small step home instead. Fixed-rate has no
            # nudges, so on a 4 mm step against a stall band the estimator is
            # never fed, the lead stays zero, the arm never moves, the
            # observation never changes, and the policy re-issues the same
            # delta forever. Measured: 0.00 mm over 60 ticks.
            #
            # So growth is driven by the arm being stuck, not by the estimator.
            # One break-away per stalled tick, exactly as creep adds one per
            # stalled nudge, and cleared as soon as the arm moves — the same
            # re-derive-per-move discipline, for the same reason: friction
            # depends on direction, so a lead earned going one way is not
            # evidence about the next.
            #
            # STUCK MEANS THE ARM DID NOT MOVE, which is the DISTANCE it
            # covered and not the component of it along the last command. The
            # two differ exactly when the policy reverses, and near the object
            # it reverses constantly: the period after a reversal is spent
            # decelerating and turning round, so the arm can travel centimetres
            # while projecting under half a millimetre onto the direction it
            # was asked for one tick ago. Read through `achieved` that is a
            # stall, and break-away then adds 10 mm of lead for it — pushing
            # HARDER into the new direction precisely because the arm was busy
            # obeying the old one. Every reversal earns another 10 mm, the cap
            # is reached in about 1.5 s, and the equilibrium then sits 100 mm
            # from the arm in a direction that alternates at the control rate.
            # That is a bang-bang limit cycle, and on hardware it is the
            # shaking people report when the gripper gets close to the object.
            #
            # Static friction, which is the only thing break-away exists to
            # overcome, does not care which way the arm is going — an arm that
            # moved at all has already broken away. So the norm is not merely
            # safer here, it is the quantity the mechanism was always about.
            # AND A TICK NOBODY ASKED TO MOVE IS NOT A STALL EITHER, which is
            # the other half of the same confusion and the one that survives
            # every fix above. `moved` alone cannot tell an arm that REFUSED to
            # go from an arm that has ARRIVED and was asked for nothing — and
            # near the object the policy's deltas are fractions of a
            # millimetre, so every arrival reads as a stall.
            #
            # Traced in simulation once the lead was damped enough for the arm
            # to converge at all: 60 mm closed to 0.03 mm in three ticks, the
            # fourth asked for 0.03 mm, the arm delivered 0.035 mm, break-away
            # called that a stall and added 10 mm — kicking the gripper 7.1 mm
            # back off the object. Reconverge in two ticks, get kicked again:
            # a +-7 mm limit cycle at 1.7 Hz that only ever appears at the end
            # of an approach, which is exactly where the shaking was reported.
            #
            # Static friction is a thing that opposes a REQUESTED motion. If
            # nothing was requested there is nothing to break away from, and
            # the lead can only push the arm off a target it had reached. So
            # the request has to clear the same floor the motion does.
            if want > RATE_STUCK_M and moved < RATE_STUCK_M:
                self._stuck_lead = min(self._stuck_lead + CREEP_BREAKAWAY_M,
                                       MAX_COMMAND_LEAD_M)
            else:
                self._stuck_lead = 0.0
        self._pending = None

        target_xyz = T_base_ctrl_target[:3, 3]
        travel = target_xyz - T_now[:3, 3]
        # Damped, and with no stall allowance: this loop never checks arrival,
        # so its lead is the loop gain. See RATE_MAX_LEAD_RATIO.
        lead = (self.droop.lead_for(travel, max_ratio=RATE_MAX_LEAD_RATIO,
                                    stall_allowance=False)
                if self.droop is not None else np.zeros(3))
        n_travel = float(np.linalg.norm(travel))
        if self._stuck_lead > 0.0 and n_travel > 1e-9:
            lead = lead + (travel / n_travel) * self._stuck_lead

        T_cmd = T_base_ctrl_target.copy()
        T_cmd[:3, 3] = target_xyz + lead
        T_cmd = clamp_command_lead(T_cmd, T_now)
        self.pub.publish(roslibpy.Message(
            matrix_to_pose_msg_like(template, T_cmd, seq)))

        n = float(np.linalg.norm(travel))
        if n > 1e-9:
            u0 = travel / n
            self._pending = (float((T_cmd[:3, 3] - T_now[:3, 3]) @ u0),
                             T_now[:3, 3].copy(), u0, n)

        # Hold the period from the tick BEFORE, not from now, so perception and
        # inference are inside the budget rather than added to it. A tick that
        # overran does not then try to claw the time back.
        if self._next_tick is None:
            self._next_tick = now
        self._next_tick += self.period
        slept = self._next_tick - time.time()
        if slept > 0:
            time.sleep(slept)
        else:
            # Fell behind: perception plus inference exceeded the period. Resync
            # rather than accumulate a debt that would make every later tick
            # instant and turn this into an unthrottled loop.
            self._next_tick = time.time()
        return seq + 1, max(slept, 0.0)


def move_to(pub, T_target_ctrl: np.ndarray, seq: int, timeout_s: float,
            droop: "DroopCompensator | None", max_passes: int, tol_m: float,
            label: str = "", creep: bool = True) -> tuple[int, float, float, int]:
    """Command a STATIC pose until the arm actually reaches it.

    With creep=True a single settle() already converges, because the correction
    happens inside the move; the extra passes then cost nothing and exist only
    for the case where a settle timed out — the arm blocked, or the target
    outside what MAX_COMMAND_LEAD_M will pull for. Each retry is a fresh move
    from wherever the last one stalled, so a transient obstruction does not end
    the episode.

    With creep=False the passes ARE the correction, and this is the loop that
    made the motion lumpy: every pass waits for a dead stop before commanding
    again, so a step executes as lunge / stop / twitch / stop / twitch / stop.
    It converges — homing showed 17.6 -> 2.97 -> 0.94 mm — it just converges
    visibly, at the joints.

    Returns (next_seq, pos_err, rot_err, passes_used, commands_issued).

    `commands_issued` is what tells you how the motion actually LOOKED: it is the
    number of separate equilibrium commands the arm responded to, so 1 is a
    single continuous move and 3 is a jump plus two visible adjustments. Without
    it the log cannot distinguish "converged in one go" from "converged after
    three corrections", which are the same final millimetres and completely
    different to watch.
    """
    dp = dr = float("inf")
    used = 0
    commands = 0
    for i in range(max_passes):
        if poll_stop():
            break
        res = settle(pub, current_msg, T_target_ctrl, seq, timeout_s, droop,
                     tol_m=tol_m if creep else SETTLE_POS_TOL_M, creep=creep)
        dp, dr, seq = res.pos_err, res.rot_err, res.next_seq
        commands += 1 + res.nudges
        used = i + 1
        if dp < tol_m:
            break
        if label and i + 1 < max_passes:
            print(f"{label} pass {i+1}: {dp*1000:.1f} mm out, "
                  f"{droop.describe() if droop else ''}", flush=True)
    return seq, dp, dr, used, commands


def go_home(pub, T_ctrl_hand: np.ndarray, T_hand_ctrl: np.ndarray,
            seq: int, droop: "DroopCompensator | None" = None,
            creep: bool = True, stream: bool = True,
            speed_m_s: float = HOME_SPEED_M_S,
            tol_m: float = HOME_REFINE_TOL_M) -> int:
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

    # Streaming picks its own, much finer, waypoint spacing: nothing is paid per
    # waypoint any more, so the only reason to space them out is gone and finer
    # is strictly smoother. The stepwise path keeps HOME_STEP_TRANS_M, where each
    # waypoint costs a settle and a hold.
    if stream:
        step_m = max(float(speed_m_s) * HOME_STREAM_DT_S, 1e-4)
        n = max(int(np.ceil(max(dist / step_m,
                                np.rad2deg(ang) / HOME_STEP_ROT_DEG))), 1)
        print(f"[home] {dist*100:.1f} cm / {np.rad2deg(ang):.1f} deg away — "
              f"streaming {n} waypoints at {speed_m_s*100:.0f} cm/s "
              f"(~{n * HOME_STREAM_DT_S:.1f} s, continuous)")
    else:
        n = max(int(np.ceil(max(dist / HOME_STEP_TRANS_M,
                                np.rad2deg(ang) / HOME_STEP_ROT_DEG))), 1)
        print(f"[home] {dist*100:.1f} cm / {np.rad2deg(ang):.1f} deg away — "
              f"{n} interpolated waypoints, settling at each")

    key_rots = Rot.from_matrix(np.stack([T_start[:3, :3], T_BASE_HAND_HOME[:3, :3]]))
    slerp = Slerp([0.0, 1.0], key_rots)

    next_tick = time.time()
    for i in range(1, n + 1):
        if poll_stop():
            print("[home] aborted", flush=True)
            return seq
        s = i / n
        T_way = np.eye(4)
        T_way[:3, :3] = slerp(s).as_matrix()
        T_way[:3, 3] = (1 - s) * T_start[:3, 3] + s * T_BASE_HAND_HOME[:3, 3]
        T_cmd = clamp_target_pose(T_way @ T_hand_ctrl)

        if stream:
            # The standing droop offset only — no travel lead. Consecutive
            # waypoints already pull the arm forward, which is what a lead is
            # for; adding one on top would aim past the end of the path and
            # arrive with momentum at exactly the pose we then want to hold.
            if droop is not None:
                T_cmd = droop.compensate(T_cmd)
            # Bounds how far the equilibrium may run ahead of the arm, and so how
            # hard the controller pulls, if the arm cannot keep up with the
            # commanded speed. Without it a too-fast path becomes a growing
            # position error and a growing force.
            T_cmd = clamp_command_lead(T_cmd, pose_msg_to_matrix(current_msg))
            pub.publish(roslibpy.Message(
                matrix_to_pose_msg_like(current_msg, T_cmd, seq)))
            seq += 1
            next_tick += HOME_STREAM_DT_S
            sleep_s = next_tick - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.time()
            continue

        # Intermediate waypoints get a loose tolerance on purpose: they are a
        # path, not a destination, and converging each one to millimetres would
        # add a creep and a full hold per waypoint for no benefit. The refine
        # below is where home is actually reached.
        res = settle(pub, current_msg, T_cmd,
                     seq, HOME_SETTLE_TIMEOUT_S, droop,
                     tol_m=HOME_WAY_TOL_M, creep=creep)
        seq = res.next_seq
        if not res.settled:
            print(f"[home] waypoint {i}/{n} timed out: {res.pos_err*1000:.1f} mm "
                  f"{np.rad2deg(res.rot_err):.1f} deg residual", flush=True)

    # Refinement. The interpolation above commands the first waypoint before any
    # droop has been observed, so the arm lands short and the estimate is only
    # learned on the way. Re-commanding home now closes that gap — and doubles as
    # the estimator's calibration, so the policy loop starts with a converged
    # value instead of learning it during your first real steps.
    # Unconditional when streaming, where it is not a refinement but the only
    # thing that lands the pose at all: nothing along a streamed path ever waits
    # for the arm, so it arrives trailing the last waypoint by whatever the droop
    # and its own lag amount to.
    if stream or creep or (droop is not None and droop.enabled):
        seq, _, _, _, _ = move_to(pub, clamp_target_pose(T_BASE_HAND_HOME @ T_hand_ctrl),
                               seq, HOME_SETTLE_TIMEOUT_S, droop,
                               HOME_REFINE_PASSES, tol_m,
                               "[home] refine", creep=creep)

    T_end = pose_msg_to_matrix(current_msg) @ T_ctrl_hand
    dp, dr = pose_error(T_end, T_BASE_HAND_HOME)
    print(f"[home] done — {dp*1000:.1f} mm / {np.rad2deg(dr):.1f} deg from home"
          + (f"  [{droop.describe()}]" if droop is not None else ""))
    return seq


# -----------------------------------------------------------------------------
class Phase4Policy:
    """The default policy this runner drives, behind the adapter interface.

    THE ADAPTER EXISTS SO A SECOND POLICY DOES NOT MEAN A SECOND RUNNER. The
    regrasp policy differs from this one in three narrow places — how it is
    loaded, that it needs a commanded direction per episode, and that its cloud
    carries two extra channels — against roughly seven hundred lines of camera
    bring-up, calibration checking, homing, gripper handling, abort polling and
    viewer plumbing that are identical. Copying those would fork every hardware
    fix in them, and this repo already has one pair of runners that diverged
    exactly that way (see "Why the two runners differ").

    Everything below is a no-op hook for this policy and a real one for regrasp.
    """

    # Does `--run` / `--policy-dir` / `--ckpt` actually name what this adapter
    # loads? False for the regrasp policy, which is named by --regrasp-run and
    # ignores all three — and a startup banner that printed them anyway would
    # put "Policy dir: checkpoint/run12" at the top of a session log driven by
    # regrasp_run19. Someone reading that log a month later has no way to know
    # it is vestigial.
    uses_policy_dir = True

    def load(self, args, device: str, policy_dir: Path, ckpt: str) -> None:
        self.device = device
        self.model, self.run_cfg = load_bc_policy(policy_dir, ckpt, device)
        _assert_state_layout(self.run_cfg)
        warm_up_policy(self.model, device)

    def reset(self) -> None:
        """Called at every episode start."""

    def act(self, pc: np.ndarray, rs: np.ndarray, *, fused=None,
            T_base_hand=None) -> np.ndarray:
        a = policy_act(self.model, pc, rs, self.device)
        self.grip_logit = grip_logit_of(self.model)
        return a

    def hud(self) -> str:
        """Extra text for the on-screen HUD, or empty."""
        return ""

    # ---- --exp-mode hooks ---------------------------------------------------

    # DOES THIS POLICY TAKE A PER-ATTEMPT COMMAND? Read once, before any
    # camera opens, to decide whether a session is a list of commands
    # (--exp-bins) or just a count of attempts (--exp-attempts). Asking the
    # class is better than calling set_command and catching the refusal: a
    # SystemExit raised to be caught is a control flow nobody reading
    # set_command would expect.
    TAKES_COMMAND = False

    def set_command(self, label: str) -> None:
        """Point the policy at the next command in an --exp-mode sequence."""
        raise SystemExit(
            f"this policy takes no command: it grasps a held object however it "
            f"likes, so {label!r} means nothing to it. --exp-bins names "
            "DIRECTIONS, which only the direction-conditioned regrasp policy "
            "reads. Run my_regrasp_policy_runner.py.")

    def validate_commands(self, labels) -> None:
        """Refuse a whole sequence up front, before a camera opens or anything moves.

        A bad label at position 3 must not surface after two attempts have
        already been run, recorded and timed.
        """
        if labels:
            self.set_command(labels[0])         # the same refusal, one message

    def record_spec(self):
        """(extra steps.csv columns, extra binary array spec) for the recorder."""
        return (), {}

    def record_step(self):
        """(extra csv values, extra arrays) for the step just taken."""
        return {}, {}

    def record_outcome_fields(self):
        """Extra attempts.csv columns this policy fills in at an attempt's end.

        Pinned here rather than taken from the first row, for the reason every
        other column list in the recorder is pinned: two attempts with
        different natural key sets would mis-align against a header written
        from whichever landed first.
        """
        return ()

    def record_outcome(self):
        """Those columns' values, measured from the LAST observation acted on.

        Called once, from `ExperimentSession.attempt_ended`, so it sees the
        pose and the cloud the policy stopped at — a grasp pose after a close,
        and wherever it ran out of budget after a timeout.
        """
        return {}


class ExperimentSession:
    """The --exp-mode sequencer: which command is next, and what phase we are in.

    IT DECIDES AND RECORDS; IT DOES NOT MOVE THE ROBOT. Every freeze, home,
    gripper open and target publish stays in `main()`'s loop, where all of them
    already live. That split is the whole point of this class being a class: the
    two runners share one loop precisely so they cannot diverge in how the arm
    is driven, and a sequencer that published would be a second driver of it.

    Five phases, each pinned to loop state that already exists rather than to a
    parallel set of flags that would have to be kept in step with it:

        IDLE     before the first 's'      armed=False  episode_over=False
        RUNNING  an attempt is live        armed=True   episode_over=False
        VERDICT  closed, awaiting p/f      armed=False  episode_over=True
        BETWEEN  outcome written           armed=False  episode_over=True
        DONE     every command judged      armed=False  episode_over=True

    "Frozen after a failure" is deliberately NOT a sixth phase: it is BETWEEN
    with `froze` set on the row just written. The loop state is identical, and a
    second phase would be one more thing to keep in step for no gain — the HUD
    tells them apart from the last row.
    """

    def __init__(self, labels, max_steps: int, recorder=None, adapter=None):
        self.labels = list(labels)
        # A SESSION IS A LIST OF ATTEMPTS; A COMMAND PER ATTEMPT IS OPTIONAL.
        #
        # The regrasp policy takes a direction, so its session is one attempt
        # per bin and the bin is what names each one. The plain Phase-4 policy
        # takes nothing — it grasps a held object however it likes — so its
        # session is just a count, `[None] * n`, and attempts are named by
        # their number. Everything else about the sequencer is identical, which
        # is why this is one label that may be None rather than two classes.
        #
        # Derived rather than passed so there is one source of truth: a caller
        # that hands over labels has already said which kind of session it is.
        self.commanded = any(l is not None for l in self.labels)
        self.noun = "bin" if self.commanded else "attempt"
        self.max_steps = int(max_steps)
        self.recorder = recorder
        # Only ever asked for `record_outcome()`, once per attempt. The session
        # still drives nothing — that is main()'s loop — it just carries the
        # one measurement only the policy can make.
        self.adapter = adapter
        # ONE timestamp for the whole session. The recorder reads this rather
        # than taking its own: two independently computed stamps put the outcome
        # table and the per-attempt files under different names.
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.cursor = 0
        self.phase = "IDLE"
        self.cur: Optional[dict] = None
        self.rows: list[dict] = []
        self._tries: dict[int, int] = {}
        # Tightest per-finger travel seen since the attempt ended. See
        # note_gripper.
        self._grip_min_m: Optional[float] = None
        # Set by `attempt_ended` when it also decides the verdict, so the two
        # halves of one ending print as one block. See `_say`.
        self._pending_head = self._pending_landed = None

    # ---- queries ------------------------------------------------------------

    @property
    def next_label(self) -> Optional[str]:
        return None if self.cursor >= len(self.labels) else self.labels[self.cursor]

    @property
    def has_next(self) -> bool:
        """Is there an attempt left? NOT the same as `next_label is not None`.

        In an uncommanded session every label IS None, so testing the label for
        existence would report a finished session from the very first attempt.
        The cursor is the only thing that knows.
        """
        return self.cursor < len(self.labels)

    def tag(self, row: Optional[dict]) -> str:
        """How one attempt is NAMED in the terminal: its command, or its number."""
        if not row:
            return "?"
        return row.get("bin") or f"#{row.get('attempt')}"

    def _next_tag(self) -> str:
        """The same, for the attempt that has not started yet."""
        return self.next_label if self.commanded else f"#{self.cursor + 1}"

    def elapsed(self) -> Optional[float]:
        if self.cur is None:
            return None
        return (self.cur["t_end"] or time.time()) - self.cur["t_start"]

    # ---- the jaws -----------------------------------------------------------

    def note_gripper(self, finger_m: Optional[float]) -> None:
        """One float compare per control iteration, from main()'s loop.

        HOW TIGHT THE FINGERS GOT IS NOT RECOVERABLE ANY OTHER WAY. It is the
        measured thickness of whatever they caught — the fingers are commanded
        to 0.0 m, so they stop on the object or they stop on each other — which
        is the difference between "the policy closed" and "the policy grasped".
        But sampling it once at the verdict does not get it: the operator can
        press 'p' a tenth of a second after the grasp goal goes out, while the
        fingers are still travelling, and steps.csv only carries whatever the
        last iteration happened to sample. A running minimum is independent of
        when the human's hand moved.

        Only while an attempt is in flight or owes a verdict, so the release
        that follows a pass — and the fully open jaws between attempts — cannot
        contaminate the next row.
        """
        if finger_m is None or self.cur is None:
            return
        if self._grip_min_m is None or finger_m < self._grip_min_m:
            self._grip_min_m = float(finger_m)

    def _grip_min_mm(self) -> Optional[float]:
        return aperture_mm(self._grip_min_m)

    # ---- what the operator reads -------------------------------------------

    RULE = "\u2500" * 66

    def _say(self, head: str, *lines: str, rule: bool = True) -> None:
        """One block: a headline, some facts, and what to press. Always that.

        THE TERMINAL IS THE ONLY INTERFACE DURING AN ATTEMPT — the operator's
        hands are near the robot and their eyes are not on the HUD — so the
        rule is one block per TRANSITION and nothing per step. Everything the
        session prints goes through here, which is what stops it drifting back
        into the scatter of one-off prints this replaced: a headline you can
        find by scrolling, the facts that headline needs, and a NEXT line that
        never has to be remembered.
        """
        out = [f"\n{self.RULE}" if rule else ""]
        out.append(f"  {head}")
        out += [f"    {ln}" for ln in lines if ln]
        nxt = self.next_line()
        if nxt:
            out.append(f"  NEXT  {nxt}")
        print("\n".join(x for x in out if x != ""), flush=True)

    def next_line(self) -> str:
        """The keys that do something RIGHT NOW, in the order you would use them.

        Only the live ones. Listing 'p' during an approach or 'f' before there
        is anything to fail teaches the operator to stop reading the line.
        """
        n = len(self.labels)
        if self.phase == "RUNNING":
            return "f = policy failed (stops the arm)   t = void   q = quit"
        if self.phase == "VERDICT":
            return "p = PASS   f = FAIL   (the arm is carrying it home)"
        if self.phase == "IDLE":
            return f"s = start 1/{n}" + (f": {self.labels[0]}"
                                         if self.commanded else "")
        last = self.rows[-1] if self.rows else None
        bits = []
        if self.has_next:
            bits.append(f"s = start {self.cursor + 1}/{n}"
                        + (f": {self.next_label}" if self.commanded else ""))
        if last is not None and self._can_retry():
            bits.append(f"r = retry {self.tag(last)}")
        if self.rows and self.rows[-1].get("froze"):
            bits.append("h = home first")
        bits.append("q = quit")
        return "   ".join(bits)

    def tally(self) -> str:
        n = lambda v: sum(1 for r in self.rows if r["verdict"] == v)
        out = f"{n('pass')} pass / {n('fail')} fail"
        return out + (f" / {n('void')} void" if n("void") else "")

    # ---- transitions --------------------------------------------------------

    def _can_retry(self) -> bool:
        """Is there a finished attempt whose bin the cursor could be rolled back to?

        False after a VOID, because 't' never advanced the cursor — the bin is
        already the next one up and 'r' would be a no-op dressed as an action.
        """
        if self.phase not in ("BETWEEN", "DONE") or not self.rows:
            return False
        return int(self.rows[-1]["attempt"]) - 1 < self.cursor

    def retry_requested(self) -> bool:
        """'r'. Re-offer the bin that just finished. True when the cursor moved.

        A RETRY DOES NOT ERASE THE ATTEMPT IT REPEATS. The failed row stays in
        `rows` and on disk with its own `try` number, and the retry is written
        beside it — so a bin retried three times shows three rows, and the
        session's success rate is over attempts rather than over whichever
        attempt the operator decided to keep. That is the same rule 't' already
        follows, and it is the difference between a log and a highlight reel.

        Only the cursor moves. Nothing is republished, no arm state changes:
        the next 's' simply commands the same bin again.
        """
        n = len(self.labels)
        if self.phase == "RUNNING":
            self._say(f"RUNNING {self.tag(self.cur)} \u2014 nothing to retry yet",
                      "'f' records a failure, 't' voids the attempt; either "
                      "then offers 'r'.")
            return False
        if self.phase == "VERDICT":
            self._say(f"VERDICT OWED for {self.tag(self.cur)}",
                      f"judge it first \u2014 a retry after 'f' re-offers the "
                      f"same {self.noun}.")
            return False
        if not self.rows:
            self._say("NOTHING TO RETRY", "no attempt has finished yet.")
            return False
        last = self.rows[-1]
        if not self._can_retry():
            # Two ways to be here and the message has to fit both: a VOID never
            # consumed the bin, and a retry already rolled the cursor back.
            why = ("'t' does not consume an attempt"
                   if last["verdict"] == "void" else "'r' already rolled back")
            self._say(f"{self.tag(last)} IS ALREADY NEXT",
                      f"{why}, so there is nothing further to re-offer.")
            return False
        self.cursor = int(last["attempt"]) - 1
        self.phase = "BETWEEN"
        self._say(f"RETRY {self.tag(last)}  \u2014  attempt "
                  f"{self.cursor + 1}/{n}, try {int(last['try']) + 1}",
                  f"the {last['verdict']} just recorded is KEPT; this is an "
                  "extra attempt, not a replacement.")
        return True

    def start_requested(self) -> bool:
        """'s'. May the next attempt start? False, with a printed reason, if not.

        A PREDICATE AND NOT THE LABEL, because None is a legal label: an
        uncommanded session's every attempt is None, so returning the label and
        testing it for None would refuse every 's' in such a session. The
        caller reads `next_label` itself, and only calls set_command when there
        is something to command.

        REFUSING IS AS MUCH OF THE MACHINE AS ACCEPTING. 's' during a verdict
        wait would start the next attempt with the last one unjudged, and after
        the final one it must do nothing at all — that is what ends a session
        rather than rolling it on.
        """
        if self.phase == "RUNNING":
            self._say(f"ALREADY RUNNING {self.tag(self.cur)}")
            return False
        if self.phase == "VERDICT":
            self._say(f"VERDICT OWED for {self.tag(self.cur)} "
                      f"({self.cur['attempt']}/{len(self.labels)})")
            return False
        if not self.has_next:
            self._say(f"SESSION COMPLETE — {len(self.labels)} {self.noun}s, "
                      f"{self.tally()}")
            return False
        return True

    def attempt_started(self, t_start: Optional[float] = None) -> str:
        label = self.labels[self.cursor]
        self._tries[self.cursor] = self._tries.get(self.cursor, 0) + 1
        self.cur = {"session_id": self.session_id, "attempt": self.cursor + 1,
                    "n_attempts": len(self.labels),
                    "try": self._tries[self.cursor], "bin": label,
                    "t_start": time.time() if t_start is None else t_start,
                    "max_steps": self.max_steps, "t_end": None,
                    "elapsed_s": None, "ending": None, "steps": None,
                    "verdict": None, "t_verdict": None, "froze": False,
                    "grip_end_mm": None, "grip_min_mm": None,
                    "grip_verdict_mm": None}
        self.phase = "RUNNING"
        self._grip_min_m = None
        # 'f' MUST INTERRUPT A settle()/move_to() IN PROGRESS, and only while an
        # attempt is running: during the verdict wait the same key is a verdict
        # and must not abort the arm carrying the object home.
        STOP_KEYS.add("f")
        self._say(f"ATTEMPT {self.cur['attempt']}/{len(self.labels)}"
                  + (f" · bin {label}" if label else "")
                  + f" · try {self._tries[self.cursor]}",
                  f"clock running · max {self.max_steps} steps · {self.tally()}")
        if self.recorder is not None:
            self.recorder.on_attempt_start(dict(self.cur))
        return label

    def attempt_ended(self, ending: str, steps: int, verdict=None,
                      t_end=None, froze: bool = False) -> None:
        STOP_KEYS.discard("f")
        if self.cur is None:
            print(f"[exp] {ending} with no attempt in flight — ignored",
                  flush=True)
            return
        self.cur.update(t_end=time.time() if t_end is None else t_end,
                        ending=ending, steps=int(steps), froze=bool(froze),
                        grip_end_mm=aperture_mm(gripper_finger_m))
        self.cur["elapsed_s"] = self.cur["t_end"] - self.cur["t_start"]
        # RE-ARMED HERE, not left running from the approach. The minimum that
        # means anything is the one the CLOSING makes; a minimum spanning the
        # approach would just report the same open jaws with extra steps.
        self._grip_min_m = None
        # WHERE THE GRIPPER ACTUALLY ENDED UP, taken now and not at the verdict.
        # The verdict comes after the carry home, by which time the pose that
        # answers "which side did it approach from" is gone. Wrapped because a
        # measurement is never worth losing the outcome row over.
        if self.adapter is not None:
            try:
                self.cur.update(self.adapter.record_outcome() or {})
            except Exception as exc:                      # noqa: BLE001
                print(f"[exp] outcome measurement failed ({exc}) — the row is "
                      "written without it", flush=True)
        # WHICH SIDE IT ACTUALLY CAME FROM belongs in this block and not a
        # later one: a grasp that succeeded from the wrong bin is a different
        # result from one that succeeded from the commanded one, and the
        # verdict key cannot tell them apart.
        rb, de = self.cur.get("bin_realized"), self.cur.get("dir_err_deg")
        landed = ""
        if rb:
            hit = self.cur.get("bin_hit")
            landed = (f"commanded {self.tag(self.cur)}, landed {rb}"
                      + ("" if de is None else f" ({de:.0f}\u00b0 off)")
                      + ("" if hit is None else
                         ("  \u2713 bin hit" if hit else "  \u2717 wrong sector")))
        w = self.cur["grip_end_mm"]
        if verdict is None:
            self.phase = "VERDICT"
            self._say(f"{self.tag(self.cur)} CLOSED · {self.cur['elapsed_s']:.1f}s "
                      f"· {steps} steps",
                      landed,
                      "the arm is carrying the object home; a key pressed "
                      "during that move is read when it finishes")
        else:
            self._pending_head = (
                f"{self.tag(self.cur)} {ending.upper()} · "
                f"{self.cur['elapsed_s']:.1f}s · {steps} steps"
                + ("" if w is None else f" · jaws {w:.0f} mm"))
            self._pending_landed = landed
            self.verdict_given(verdict)

    def verdict_given(self, verdict: str) -> bool:
        """Close the row out and advance. True when the fingers are HOLDING."""
        if self.cur is None:
            return False
        self.cur.update(verdict=verdict, t_verdict=time.time(),
                        grip_verdict_mm=aperture_mm(gripper_finger_m),
                        grip_min_mm=self._grip_min_mm())
        row = dict(self.cur)
        self.rows.append(row)
        held = row["ending"] == "close"
        self.cur = None
        self.cursor += 1
        self.phase = "DONE" if self.cursor >= len(self.labels) else "BETWEEN"
        if self.recorder is not None:
            self.recorder.on_attempt_done(row)
        # THE JAWS ARE PART OF THE VERDICT because they are the check on it:
        # "pass" beside a 0.4 mm closure is an operator who judged too early,
        # or an object that was never between the fingers.
        mn, at = row["grip_min_mm"], row["grip_verdict_mm"]
        jaws = "" if mn is None else (
            f"jaws closed to {mn:.1f} mm"
            + ("" if at is None or abs(at - mn) < 0.5
               else f", {at:.1f} mm at the verdict \u2014 it slipped"))
        # `attempt_ended` stashes these when it decides the verdict itself (a
        # timeout or an 'f'), so that ending gets ONE block rather than two.
        # No verdict in the fallback head: `_say` appends it below, and
        # "+x PASS ... -> PASS" reads like two different facts.
        head = getattr(self, "_pending_head", None) or (
            f"{self.tag(row)} · {row['elapsed_s']:.1f}s · {row['steps']} steps")
        landed = getattr(self, "_pending_landed", None) or ""
        self._pending_head = self._pending_landed = None
        self._say(f"{head}  \u2192  {verdict.upper()}",
                  landed, jaws,
                  ("ARM FROZEN where it stopped" if row["froze"] else ""),
                  f"session so far: {self.tally()}")
        return held

    def attempt_voided(self, steps: int) -> None:
        """'t' during an attempt. The command is RE-OFFERED, not consumed.

        A user stop is not a policy failure — 'f' is — and a slip of the finger
        should not cost an attempt out of the session.
        """
        STOP_KEYS.discard("f")
        if self.cur is None:
            return
        self.cur.update(t_end=time.time(), ending="user_stop",
                        steps=int(steps), verdict="void", froze=True,
                        t_verdict=time.time(),
                        grip_end_mm=aperture_mm(gripper_finger_m),
                        grip_verdict_mm=aperture_mm(gripper_finger_m))
        # grip_min_mm stays empty: a void never closed, so there is no closure
        # to report, and the approach's open jaws written into that column
        # would read as a grasp that stopped at 80 mm.
        self.cur["elapsed_s"] = self.cur["t_end"] - self.cur["t_start"]
        row = dict(self.cur)
        self.rows.append(row)
        self.cur = None
        self.phase = "BETWEEN"                  # the cursor does NOT advance
        if self.recorder is not None:
            self.recorder.on_attempt_done(row)
        self._say(f"{self.tag(row)} VOIDED by 't' · {row['elapsed_s']:.1f}s · "
                  f"{steps} steps",
                  f"a user stop is not a policy failure, so the {self.noun} is "
                  "NOT consumed",
                  "ARM FROZEN where it stopped",
                  f"session so far: {self.tally()}")

    def close(self, reason: str) -> dict:
        """From main()'s finally. Nothing after this runs: __main__ leaves
        through os._exit(0), which flushes stdout and stderr and nothing else."""
        STOP_KEYS.discard("f")
        if self.cur is not None:
            # Read BEFORE the update, because "did this attempt reach a close"
            # is what decides whether the tracked minimum means anything, and
            # the update is about to overwrite `ending` with "abandoned".
            closed = self.cur["ending"] == "close"
            self.cur.update(
                t_end=self.cur["t_end"] or time.time(),
                ending=self.cur["ending"] or "abandoned",
                steps=self.cur["steps"] or 0, verdict="abandoned",
                # Whatever was already captured at the ending, else the jaws as
                # they are now — quitting mid-approach still records an opening.
                grip_end_mm=(self.cur["grip_end_mm"]
                             if self.cur["grip_end_mm"] is not None
                             else aperture_mm(gripper_finger_m)),
                grip_verdict_mm=aperture_mm(gripper_finger_m),
                # Only if the fingers were actually asked to close. Abandoning
                # mid-approach leaves a minimum that is just the open jaws.
                grip_min_mm=self._grip_min_mm() if closed else None)
            self.cur["elapsed_s"] = self.cur["t_end"] - self.cur["t_start"]
            self.rows.append(dict(self.cur))
            if self.recorder is not None:
                self.recorder.on_attempt_done(self.rows[-1])
            self.cur = None
        summary = {"session_id": self.session_id, "reason": reason,
                   "labels": list(self.labels), "rows": list(self.rows),
                   "attempted": self.cursor, "n_attempts": len(self.labels)}
        # THE LABEL COLUMN ONLY EARNS ITS WIDTH IF THERE ARE LABELS. With no
        # command it would hold "#2" against a row that already begins "2.2" —
        # the same number twice, in a table whose whole job is to be read at a
        # glance after a long session.
        cmd = self.commanded
        print(f"\n{self.RULE}\n  SESSION {self.session_id} ENDED — {reason}\n"
              f"    {self.cursor}/{len(self.labels)} {self.noun}s · "
              f"{self.tally()}\n"
              f"  {'':2s}{'#':>4s} " + (f"{self.noun[:3]:>3s} " if cmd else "")
              + f"{'ending':>11s} {'time':>6s} "
              f"{'steps':>6s} {'jaws':>7s} {'land':>4s} {'off':>4s}  verdict")
        for r in self.rows:
            mn = r.get("grip_min_mm")
            rb = r.get("bin_realized") or "--"
            de = r.get("dir_err_deg")
            print(f"  {'':2s}{r['attempt']}.{r['try']:<2d} "
                  + (f"{self.tag(r):>3s} " if cmd else "")
                  + f"{str(r['ending']):>11s} {r['elapsed_s']:5.1f}s "
                  f"{r['steps']:6d} "
                  f"{'     --' if mn is None else f'{mn:5.1f}mm'} "
                  f"{rb:>4s}"
                  f"{'    ' if de is None else f'{de:4.0f}'}  {r['verdict']}")
        if self.recorder is not None:
            self.recorder.on_session_end(summary)
        return summary

    # ---- the HUD ------------------------------------------------------------

    def status_line(self, step: int, max_steps: int,
                    auto_home_in: Optional[float] = None) -> str:
        n = len(self.labels)
        if self.phase == "IDLE":
            return (f"EXP 0/{n} — press 's' for 1/{n}"
                    + (f": {self.next_label}" if self.commanded else ""))
        if self.phase == "RUNNING":
            return (f"RUNNING {self.cur['attempt']}/{n} {self.tag(self.cur)}   "
                    f"t={self.elapsed():.1f}s   step {step}/{max_steps}")
        if self.phase == "VERDICT":
            # THE LIVE CLOSURE, so the verdict is a reading and not a guess.
            # The fingers are commanded to 0.0 m: settling near zero means they
            # met each other and the object is not in them, whatever it looked
            # like from across the room.
            mn = self._grip_min_mm()
            s = (f"CLOSED {self.cur['attempt']}/{n} {self.tag(self.cur)} at "
                 f"{self.cur['elapsed_s']:.2f}s / {self.cur['steps']} steps"
                 + ("" if mn is None else f"   jaws {mn:.1f} mm")
                 + " — VERDICT? 'p'=pass 'f'=fail")
            return s + ("" if auto_home_in is None
                        else f"   CARRYING HOME IN {auto_home_in:.1f}s")
        if self.phase == "DONE":
            return f"SESSION COMPLETE — {n}/{n}, {self.tally()}. 'q' to end."
        last = self.rows[-1] if self.rows else {}
        head = ("FAILED" if last.get("verdict") == "fail" else
                "VOIDED" if last.get("verdict") == "void" else "PASS")
        return (f"{head} {last.get('attempt')}/{n} {self.tag(last)} "
                f"({last.get('ending')}, {last.get('elapsed_s', 0.0):.2f}s)"
                + ("  ARM FROZEN — 'h' to home, then" if last.get("froze")
                   else ".")
                + f" 's' for {self.cursor + 1}/{n}"
                + (f": {self.next_label}" if self.commanded else "") + "   "
                f"[{self.tally()}]")

    def key_line(self) -> str:
        """The HUD legend. Derived from `next_line` so the overlay and the
        terminal cannot drift apart, plus the keys that are live in EVERY phase
        and so are left out of the terminal's NEXT line as noise."""
        base = self.next_line()
        if self.phase == "RUNNING":
            return base
        return base + ("   h = home   t = STOP" if self.phase != "VERDICT"
                       else "   h = home now   t = cancel the carry")


def _exp_bins(text: str) -> list[str]:
    """'+x,+y,-y' -> ['+x', '+y', '-y']. Order is the session; repeats are legal."""
    labels = [t.strip() for t in str(text).split(",") if t.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("--exp-bins is empty")
    return labels


def build_parser() -> argparse.ArgumentParser:
    """Every flag, as a parser rather than parsed arguments.

    Split out so `my_regrasp_policy_runner.py` can add its own flags to THIS
    parser instead of restating fifty-odd of them. Two runners with two
    hand-maintained copies of the same flag list is how they end up disagreeing
    about a default, and the disagreement shows up as a robot behaving
    differently for no visible reason.
    """
    p = argparse.ArgumentParser(
        description="Run a Phase-4 BC policy from checkpoint/<run> on the FR3.")
    p.add_argument("--rosbridge-host", type=str, default=ROSBRIDGE_HOST)
    p.add_argument("--rosbridge-port", type=int, default=ROSBRIDGE_PORT)
    p.add_argument("--run", type=str, default=None, metavar="NAME",
                   help="policy to run, by the DAgger run it came from: "
                        f"{', '.join(available_runs()) or 'none installed'}. "
                        "Shorthand for --policy-dir checkpoint/NAME. The "
                        "folders are named after the run because that is the "
                        "only name that identifies which policy a rollout used "
                        "— 'cp3' does not say what it was trained on.")
    p.add_argument("--policy-dir", type=str, default=None,
                   help="run dir holding config.yaml, normalization.npz, "
                        "best.pt. Use --run for one inside checkpoint/; this is "
                        f"for a dir anywhere else. Default: {DEFAULT_RUN}.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="explicit .pt path (default: <policy-dir>/best.pt)")
    p.add_argument("--hand-seg-ckpt", type=str, default=str(DEFAULT_HAND_SEG_CKPT))
    p.add_argument("--hand-eye", type=str, default=None,
                   help="4x4 .npy T_hand_cam (camera -> panda_hand). Defaults to "
                        "the sim's nominal wrist mount, which is NOT a "
                        "calibration of your D435.")
    p.add_argument("--ee-offset-z", type=float, default=None,
                   help="z offset from panda_hand to the frame /cartesian_pose "
                        "publishes. DEFAULT IS TO READ IT off the robot's own "
                        "F_T_EE, which is the only authority on it: 0 when the "
                        "controller publishes panda_hand, 0.1034 when the EE is "
                        "configured to the Franka Hand TCP. Pass a value only to "
                        "override that, and expect a loud warning if it "
                        "disagrees with the robot.")
    p.add_argument("--enable-gripper", action="store_true",
                   help="actually command franka_gripper on a CLOSE prediction. "
                        "Needs the controller launched with load_gripper:=True, "
                        "which is what starts the franka_gripper node.")
    p.add_argument("--home-gripper", action="store_true",
                   help="send a homing goal at startup (implies "
                        "--enable-gripper). The Franka Hand needs this once per "
                        "power cycle before its reported width is calibrated, "
                        "and franka_gripper.launch does not do it for you. The "
                        "fingers open and close fully, so keep them clear.")
    p.add_argument("--control", choices=("settle", "rate"), default="settle",
                   help="how a predicted step reaches the robot. 'settle' "
                        "(default) commands the target and BLOCKS until the arm "
                        "has arrived and held still, which is where ~95%% of a "
                        "step goes. 'rate' is the CVPR2023 loop: publish once, "
                        "dwell --rate-hz, look again, never check arrival — "
                        "continuous motion, and the regime the policy was "
                        "actually trained in.")
    p.add_argument("--rate-hz", type=float, default=RATE_CONTROL_HZ,
                   help=f"policy rate for --control rate (default "
                        f"{RATE_CONTROL_HZ:.2f} Hz = the paper's "
                        "POLICY.TIME_ACTION_REPEAT of 0.15 s). Ignored by "
                        "--control settle.")
    p.add_argument("--rate-anchor", choices=("obs", "now"), default="obs",
                   help="which pose the policy's delta hangs off. 'obs' "
                        "(default, and what every run so far used) is the pose "
                        "the observation was taken at, matching sim. 'now' "
                        "re-reads the pose at publish time, ~120 ms later, so "
                        "each step gets a full delta of fresh travel from where "
                        "the arm actually is — try it if the arm stops and "
                        "lurches or reverses near the object, and judge it on "
                        "the moved=/res= columns.")
    p.add_argument("--step-mode", action="store_true",
                   help="SAFE MODE: preview each predicted step and only execute "
                        "it when you press SPACE. Nothing moves unprompted.")
    p.add_argument("--home", action="store_true",
                   help="drive to the sim's episode-start pose before running "
                        "(interpolated Cartesian move — keep the workspace clear)")
    p.add_argument("--home-speed", type=float, default=HOME_SPEED_M_S,
                   help=f"Cartesian speed of the homing path in m/s (default "
                        f"{HOME_SPEED_M_S}). The path is streamed on a clock, so "
                        "this sets how fast the arm crosses the workspace; the "
                        "publish rate only sets how finely.")
    p.add_argument("--home-tol", type=float, default=HOME_REFINE_TOL_M,
                   help=f"how close to home is close enough, in metres (default "
                        f"{HOME_REFINE_TOL_M}). Home is a starting pose, not a "
                        "target: the policy observes wherever the arm actually "
                        "is and steps from there. Asking for less than the arm "
                        "can reliably land — it has ~17 mm of standing droop and "
                        "a gain that varies 2x between moves — makes the refine "
                        "overshoot and hunt instead of stopping.")
    p.add_argument("--auto-home-delay", type=float, default=AUTO_HOME_DELAY_S,
                   metavar="SEC",
                   help=f"seconds to wait after the policy CLOSES the gripper "
                        f"before homing on its own (default {AUTO_HOME_DELAY_S}). "
                        "The wait is a deadline, not a sleep: perception, the "
                        "overlays and every key keep running through it, and "
                        "'t', 'h' or 's' all cancel the pending move. Only a "
                        "close schedules one — a stop, a max-steps ending or a "
                        "quit never does.")
    p.add_argument("--no-auto-home", action="store_true",
                   help="never home on its own after a close; wait for 'h', "
                        "which is how this behaved before. Worth having while "
                        "you are debugging a policy that closes in the wrong "
                        "place, since the automatic move drags the arm out of "
                        "the pose you wanted to look at.")
    p.add_argument("--exp-mode", action="store_true",
                   help="run a SESSION: a sequence of attempts, one per 's', "
                        "each timed and recorded. A policy that takes a "
                        "command wants --exp-bins; one that does not (this "
                        "runner's) wants --exp-attempts, default 1. "
                        "Implies --home, --home-gripper and --enable-gripper. "
                        "'f' fails the running attempt and stops the arm where "
                        "it stands; after a close the arm carries the object "
                        "home and waits for 'p' (pass) or 'f' (fail); 'q' ends "
                        "the session once every command has been judged.")
    p.add_argument("--exp-bins", type=_exp_bins, default=None, metavar="A,B,C",
                   help="the ORDERED sequence of commands, one attempt each: "
                        "--exp-bins +x,+y,+z,-y. Repeats are legal and mean "
                        "run it again. What a label may BE is the policy's "
                        "business — the regrasp runner takes directions and "
                        "refuses any its run has no demonstrations for, up "
                        "front, before a camera opens.")
    p.add_argument("--exp-attempts", type=int, default=None, metavar="N",
                   help="how many attempts this session is, when the policy "
                        "takes no command (default 1 — one grasp per session). "
                        "The sequencer, the timing, the recording, the verdict "
                        "keys and 'r' to retry are the same either way; only "
                        "what names an attempt differs. Mutually exclusive "
                        "with --exp-bins, which says the same thing by "
                        "listing the commands.")
    p.add_argument("--exp-out", type=str, default=None, metavar="DIR",
                   help="where the session's records go. A subdirectory named "
                        "for the session timestamp is created inside it. Only "
                        "read under --exp-mode; without it nothing is recorded, "
                        "which is what you want for a rehearsal.")
    p.add_argument("--exp-no-depth", action="store_true",
                   help="skip the depth stream. Depth is ~94%% of the data "
                        "(~260 MB per attempt per two cameras against ~15 MB "
                        "without) and it is the ONLY thing that makes the raw "
                        "cloud reconstructable offline — the video and the "
                        "labels cannot be turned back into 3D. Use this for "
                        "long rehearsals, not for data you intend to keep.")
    p.add_argument("--exp-video-fps", type=float, default=None, metavar="FPS",
                   help="the frame rate declared in the mp4 headers. Defaults "
                        "to --rate-hz under --control rate (so playback is "
                        "real time) and 10 otherwise. It is NOMINAL: one video "
                        "frame is written per recorded step and the true "
                        "timestamps are in steps.csv, so this changes playback "
                        "speed and nothing about which frame is which.")
    p.add_argument("--home-stepwise", action="store_true",
                   help="home the old way: settle on every 2 cm waypoint. That "
                        "is one dead stop per waypoint — fifteen for a 30 cm "
                        "home — which is what made homing stutter. Kept as an "
                        "escape hatch and for comparison.")
    p.add_argument("--home-only", action="store_true",
                   help="home and exit, without running the policy")
    p.add_argument("--no-droop-compensation", action="store_true",
                   help="command equilibrium poses raw. The impedance controller "
                        "settles ~17 mm short of any target, so each step then "
                        "executes as (delta - droop) rather than delta.")
    p.add_argument("--step-tol-frac", type=float, default=STEP_CONVERGE_TOL_FRAC,
                   help="how close a step must land, as a fraction of the step "
                        "itself. Lower is more precise and less smooth: the arm "
                        "corrects more, and every correction is a visible stop "
                        f"and restart. Default {STEP_CONVERGE_TOL_FRAC}; 0.15 "
                        "restores tight per-step convergence.")
    p.add_argument("--no-creep", action="store_true",
                   help="revert to the multi-pass step correction: command, wait "
                        "for a dead stop, re-command, repeat. Converges to the "
                        "same place but the arm visibly stops and restarts two "
                        "or three times per policy step.")
    p.add_argument("--settle-timeout", type=float, default=SETTLE_TIMEOUT_S,
                   help="per-move budget. Creep corrects inside this window "
                        "rather than across repeated calls, so it wants room: "
                        f"default {SETTLE_TIMEOUT_S} s.")
    p.add_argument("--max-steps", type=int, default=MAX_POLICY_STEPS)
    p.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    p.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    p.add_argument("--depth-width", type=int, default=DEPTH_WIDTH,
                   help="must be a depth mode the device offers — the D435 has "
                        "no 424x240 depth, unlike its color stream")
    p.add_argument("--depth-height", type=int, default=DEPTH_HEIGHT)
    p.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    p.add_argument("--camera-model", choices=("d435", "d435i", "d415", "d455"),
                   default=None,
                   help="override the camera body instead of detecting it from "
                        "the device. Only affects the near-depth floor (a D455 "
                        "sees nothing closer than 0.40 m against a D435's 0.10) "
                        "and what is printed. Detection is normally right; use "
                        "this if the driver's device name is unhelpful.")
    p.add_argument("--serial", action="append", default=None,
                   metavar="ROLE=SERIAL",
                   help="override one role's serial without editing "
                        "calib_config.py, e.g. --serial tripod=419122270338 "
                        "after swapping the tripod camera. Repeatable.")
    p.add_argument("--camera-serial", type=str, default=None,
                   help="RealSense serial for the WRIST camera. With two D435s "
                        "attached, librealsense otherwise binds whichever "
                        "enumerates first. Ignored unless --cameras is just "
                        "'wrist'; otherwise serials come from calib_config.")
    p.add_argument("--cameras", type=str, default="wrist",
                   help="comma-separated camera roles to fuse, e.g. "
                        "'wrist,tripod'. cp2 (run 12) is a WRIST-ONLY policy and "
                        "wants 'wrist'; cp3 (run 16) was trained on wrist+left+"
                        "right and wants 'wrist,tripod'. Roles other than "
                        "'wrist' are fixed cameras and need --calib-session.")
    p.add_argument("--calib-session", type=str, default=None,
                   help="hand-eye session under 'camera calibration/sessions/' "
                        "providing T_base_color.npy for the fixed camera(s).")
    p.add_argument("--per-camera-cap", type=int, default=None,
                   help="cap each camera's contribution per class before the "
                        "union. Default (unset) matches the simulator, which "
                        "concatenates raw so a nearer view dominates.")
    p.add_argument("--show-cloud", action="store_true",
                   help="open the same 3D window test_perception_viz.py opens: "
                        "the [1024, 5] policy cloud in large coloured dots over "
                        "the raw deprojected scene from every camera in small "
                        "white ones, in panda_hand, with a gripper wireframe. "
                        "Keys work in any window — c = colour by class/camera, "
                        "w = white cloud on/off, z/x = roll, r = drag mode.")
    p.add_argument("--cloud-update-hz", type=float, default=10.0,
                   help="redraw rate for --show-cloud. Open3D re-uploads the "
                        "whole buffer each update, so pushing this to camera "
                        "rate costs the control loop for no readable gain.")
    p.add_argument("--context-stride", type=int, default=6,
                   help="pixel stride for the white raw-scene cloud. Higher is "
                        "sparser and cheaper; it is only a backdrop.")
    p.add_argument("--context-radius", type=float, default=1.2,
                   help="clip the white cloud to this radius around panda_hand. "
                        "A tripod at 1.5 m otherwise contributes the whole room, "
                        "which dominates the view scale for no diagnostic gain.")
    p.add_argument("--context-max", type=int, default=30000,
                   help="hard cap on white-cloud points. The geometry is "
                        "allocated once at this size and cannot grow later.")
    p.add_argument("--no-context", action="store_true",
                   help="start with the white raw-scene cloud hidden ('w' "
                        "toggles it).")
    p.add_argument("--no-robot-exclusion", action="store_true",
                   help="keep points inside the gripper box in the object class. "
                        "The sim excluded the arm by segmentation id; without "
                        "this box a side camera labels the approaching gripper "
                        "as object.")
    p.add_argument("--finger-boxes", choices=("split", "span", "off"),
                   default="split",
                   help="how to exclude the FINGERS, which the gripper box "
                        "deliberately spares. 'split' (default) is two boxes on "
                        "the finger bodies only, leaving the jaw gap for the "
                        "object. 'span' merges them into one box running "
                        "through the gap: it cannot miss a mis-calibrated "
                        "finger point, but it also deletes the object once the "
                        "object is between the jaws, which is the last few "
                        "steps of every episode. 'off' disables the cut without "
                        "touching --no-robot-exclusion.")
    p.add_argument("--no-cluster", action="store_true",
                   help="define the object class by the crop sphere alone, as "
                        "before. Default is to keep only the points 3D-connected "
                        "to the hand, which removes the table and lets the "
                        "radius be loose enough to hold a long object. Turning "
                        "this off also restores the old tight radii, so the two "
                        "settings are a real A/B.")
    p.add_argument("--hand-margin-px", type=int, default=None,
                   help="pixels around the hand mask belonging to neither "
                        "class (default 5 wrist / 4 fixed). Removes the shell "
                        "of hand-surface points the mask misses, which "
                        "otherwise bridges the held object to the forearm and "
                        "makes arm rejection flicker. 0 disables.")
    p.add_argument("--no-arm-rejection", action="store_true",
                   help="keep object-class blobs behind the hand. The hand "
                        "model segments hands, not arms, so the forearm falls "
                        "into the object class and is anatomically connected to "
                        "the hand — connectivity cannot remove it, only which "
                        "side of the hand it sits on can.")
    p.add_argument("--arm-offset", type=float, default=0.07,
                   help="how far behind the hand, along the hand->robot-base "
                        "axis, a blob must sit before it is called forearm. "
                        "Larger keeps more (safer for an object held with its "
                        "body toward the human), smaller cuts more arm.")
    p.add_argument("--arm-lateral", type=float, default=0.18,
                   help="how far SIDEWAYS of that axis a point may sit and "
                        "still be object. Without this an inclined forearm "
                        "scores near zero on the axis and survives; with it, "
                        "the kept region is a capsule around the hand. Larger "
                        "keeps a long object held across the view, smaller cuts "
                        "more arm.")
    p.add_argument("--arm-below", type=float, default=0.10,
                   help="reject object points more than this far BELOW the hand. "
                        "A forearm descends to the elbow; an object being "
                        "offered does not hang under the hand holding it. This "
                        "is what allows --arm-lateral to be wide enough to keep "
                        "an object held ACROSS the hand-to-robot axis.")
    add_segmentation_args(p)
    p.add_argument("--wrist-seg-px", type=int, default=None,
                   help="hand-segmentation input size for the wrist camera "
                        "(default 256, what cp1 was trained at). "
                        "--segmentation sam2 ignores this: SAM2 has its own "
                        "fixed input size.")
    p.add_argument("--fixed-seg-px", type=int, default=None,
                   help="hand-segmentation input size for the fixed camera(s) "
                        "(default 384). At ~1 m the hand is a few dozen pixels "
                        "at 256 and its mask falls through min_hand_points. On "
                        "this GPU 384 costs ~10 ms per pass and 512 ~25 ms.")
    p.add_argument("--dry-run", action="store_true",
                   help="run perception + policy and print targets, publish nothing")
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main(adapter: "Phase4Policy | None" = None,
         args: "argparse.Namespace | None" = None) -> None:
    # `adapter` swaps WHICH policy is driven without touching anything about how
    # the robot is driven; `args` lets a caller supply a parser of its own that
    # extends build_parser(). Both default to the Phase-4 behaviour exactly.
    adapter = adapter or Phase4Policy()
    args = args if args is not None else parse_args()

    # ---- --exp-mode: refuse the contradictions before anything opens --------
    exp_labels = list(getattr(args, "exp_bins", None) or [])
    if args.exp_mode:
        # WHAT IT IMPLIES, and why it implies rather than requires. A session
        # whose fingers were never homed reports an uncalibrated width for every
        # attempt, which poisons robot_state[25] and the finger exclusion boxes
        # silently — see FrankaGripper.prepare. Making that a flag you can
        # forget makes it a failure mode you will not see.
        if args.dry_run:
            raise SystemExit("--exp-mode drives the robot and times attempts; "
                             "--dry-run publishes nothing. Pass one.")
        if args.home_only:
            raise SystemExit("--exp-mode and --home-only are contradictory.")
        if args.step_mode:
            raise SystemExit(
                "--exp-mode and --step-mode are contradictory: the elapsed time "
                "an attempt reports would be however long you took to press "
                "SPACE. Run the session continuously.")
        if args.no_auto_home:
            raise SystemExit(
                "--exp-mode homes after every close, carrying the object, and "
                "waits for your verdict. --no-auto-home deletes exactly that "
                "step. Use --auto-home-delay to change WHEN, not whether.")
        if exp_labels and args.exp_attempts is not None:
            raise SystemExit(
                f"--exp-bins ({len(exp_labels)}) and --exp-attempts "
                f"({args.exp_attempts}) both say how long this session is. "
                "Pass one.")
        # A COMMANDED SESSION IS A LIST; AN UNCOMMANDED ONE IS A COUNT.
        #
        # Asked of the class rather than by calling set_command and catching
        # its SystemExit. The refusal is still what an operator sees if they
        # pass --exp-bins to a policy that takes no command — validate_commands
        # below — because that message explains which runner they wanted,
        # which a generic "this policy takes no command" would not.
        if getattr(adapter, "TAKES_COMMAND", False):
            if not exp_labels:
                raise SystemExit("--exp-mode needs --exp-bins, e.g. "
                                 "--exp-bins +x,+y,+z,-y")
        elif exp_labels:
            adapter.validate_commands(exp_labels)      # refuses, and says why
        else:
            n_att = 1 if args.exp_attempts is None else int(args.exp_attempts)
            if n_att < 1:
                raise SystemExit(f"--exp-attempts {n_att} is not a session.")
            # None and not a placeholder string: `bin` is a column in
            # attempts.csv, and a made-up label there would read like a command
            # the policy was given. Empty is the truth.
            exp_labels = [None] * n_att
        args.home = True             # home the ARM at startup
        args.home_gripper = True     # ... and CALIBRATE the fingers
        args.enable_gripper = True   # a close that does nothing is not an attempt
        print(f"[exp] session of {len(exp_labels)} attempt"
              f"{'' if len(exp_labels) == 1 else 's'}"
              + (f": {', '.join(exp_labels)}" if exp_labels[0] else "")
              + "\n[exp] --home, --home-gripper and --enable-gripper are "
                "implied.")
        if exp_labels[0] is not None:
            adapter.validate_commands(exp_labels)  # BEFORE any camera or motion
    elif exp_labels or args.exp_out is not None or args.exp_attempts is not None:
        raise SystemExit("--exp-bins / --exp-attempts / --exp-out are only "
                         "read under --exp-mode.")

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

    if args.run and args.policy_dir:
        raise SystemExit(
            f"--run {args.run} and --policy-dir {args.policy_dir} both name a "
            "policy. Pass one.")
    if args.run:
        policy_dir = (CHECKPOINT_DIR / args.run)
        if not policy_dir.is_dir():
            raise SystemExit(
                f"No checkpoint/{args.run}. Installed: "
                f"{', '.join(available_runs()) or '(none)'}.\n"
                "The folders are named after the DAgger run they came from; "
                "copy best.pt, normalization.npz and config.yaml from "
                f"output/dagger_runs/dagger4_{args.run}/best/ to add one.")
    else:
        policy_dir = Path(args.policy_dir or DEFAULT_POLICY_DIR)
    policy_dir = policy_dir.expanduser().resolve()
    ckpt = args.ckpt or str(policy_dir / "best.pt")
    hand_seg_ckpt = Path(args.hand_seg_ckpt).expanduser().resolve()

    T_hand_cam = (np.load(args.hand_eye).astype(np.float64) if args.hand_eye
                  else T_HAND_CAM_NOMINAL)
    if T_hand_cam.shape != (4, 4):
        raise ValueError(f"--hand-eye must be 4x4, got {T_hand_cam.shape}")

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not camera_names:
        raise SystemExit("--cameras is empty")
    # Only warn about the wrist mount if a wrist camera is actually in use —
    # T_hand_cam is unused otherwise, and warning about it would be noise.
    if args.hand_eye is None and "wrist" in camera_names:
        print("[calib] WARNING: using the SIM's nominal wrist mount for T_hand_cam. "
              "Pass --hand-eye <T_hand_cam.npy> once you have calibrated the D435.")
    fixed_names = [c for c in camera_names if c != "wrist"]
    if fixed_names and args.calib_session is None:
        raise SystemExit(
            f"--cameras includes fixed camera(s) {', '.join(fixed_names)} but no "
            "--calib-session. A fixed camera's pose in the base frame is not "
            "guessable; without it its points land arbitrarily in the hand frame "
            "and the fused cloud is worse than the wrist camera alone.")
    if "wrist" not in camera_names:
        # Allowed, but it is a bigger departure than dropping one of three views,
        # so say what changes rather than just letting it run. Not an error: the
        # fusion is viewpoint-agnostic and this is a legitimate way to isolate
        # the tripod path.
        print(
            "\n[warn] no wrist camera in --cameras. Two things change:\n"
            "  * cp2 was trained wrist-ONLY and cp3 on wrist+left+right, so every\n"
            "    checkpoint here saw the eye-in-hand view. Running without it is a\n"
            "    larger input-distribution shift than dropping a side view.\n"
            "  * the wrist camera is the only one whose extrinsics are constant.\n"
            "    With fixed cameras alone, EVERY point is placed through\n"
            "    inv(T_base_hand) @ T_base_color, so the hand-eye calibration and\n"
            "    the reported EE pose now sit in series with the whole cloud —\n"
            "    there is no pose-independent view left to anchor it.\n"
            f"    Calibration in use: session {args.calib_session!r}.\n")

    if adapter.uses_policy_dir:
        print(f"Policy dir     : {policy_dir}")
        print(f"Checkpoint     : {ckpt}")
    print(f"Hand seg ckpt  : {hand_seg_ckpt}")
    print("EE offset z    : "
          + (f"{args.ee_offset_z:.4f} m (OVERRIDE)" if args.ee_offset_z is not None
             else "from the robot's F_T_EE")
          + " (panda_hand -> published frame)")
    print(f"Gripper        : {'ENABLED' if args.enable_gripper else 'disabled'}")
    print(f"Droop comp     : {'off' if args.no_droop_compensation else 'on'}")
    if args.control == "rate":
        print(f"Step anchored  : {args.rate_anchor}"
              + ("  (the pose the observation was taken at, as in sim)"
                 if args.rate_anchor == "obs" else
                 "  (re-read at publish time, ~120 ms later — A/B)"))
    # None means "never", which is what the CLOSE branch tests. Resolved here so
    # there is one answer rather than a flag pair re-read at the call site.
    auto_home_s = (None if args.no_auto_home or args.dry_run
                   else max(0.0, float(args.auto_home_delay)))
    print("Auto-home      : "
          + (f"{auto_home_s:.1f}s after a policy CLOSE" if auto_home_s is not None
             else "off, --dry-run publishes nothing" if args.dry_run
             else "off (press 'h')"))
    print(f"Step motion    : {'multi-pass (stops between passes)' if args.no_creep else 'creep (one stop per step)'}")
    print(f"Cameras        : {', '.join(camera_names)}"
          + (f"  (calib session {args.calib_session})" if fixed_names else ""))

    droop = DroopCompensator(enabled=not args.no_droop_compensation)

    # Not loaded under --segmentation sam2: cp1 is 40M parameters and 9-26 ms a
    # pass on a GPU that is about to hold SAM2, Grounding DINO and the policy,
    # and in that mode nothing would ever call it.
    hand_seg_model = (load_hand_segmenter(device, hand_seg_ckpt)
                      if args.segmentation == "hand-net" else None)
    adapter.load(args, device, policy_dir, ckpt)

    # panda_hand <-> the frame the controller publishes. Provisional: the real
    # value is read off F_T_EE once rosbridge is up, a few lines below. Only
    # --dry-run keeps this one, and it never uses it (T_base_hand is identity).
    ee_offset_z = (args.ee_offset_z if args.ee_offset_z is not None
                   else DEFAULT_EE_OFFSET_Z)
    T_hand_ctrl = z_offset_transform(ee_offset_z)
    T_ctrl_hand = invert_transform(T_hand_ctrl)

    client = None
    sub = pub = gripper_sub = None
    rigs: list = []          # referenced by the finally block before it is filled
    viewer = None
    # Same reason as `rigs`: the finally block closes these, and a failure
    # during camera bring-up must not turn into a NameError that hides it.
    exp = rec = None
    stop_reason = "did not start"
    gripper = FrankaGripper(None, False)
    publish_seq = 0   # shared by homing and the policy loop

    try:
        # ---- cameras ----
        # BEFORE ROS AND BEFORE HOMING, deliberately. Opening a camera is
        # reversible and costs nothing; homing MOVES THE ARM. With the order
        # reversed, asking for a camera that is not plugged in meant the robot
        # drove to the home pose, sat there, and only then did the run die on
        # "No device connected" — leaving the arm somewhere it was moved to for
        # a run that never started. Everything that can be checked without
        # touching the robot is now checked first.
        #
        # With more than one RealSense attached, librealsense binds whichever
        # device enumerates first, and opening the wrong one is silent: the
        # policy just receives a viewpoint it never saw in training. Serials come
        # from calib_config.CAMERA_SERIALS so the wrist/tripod assignment has one
        # definition on this machine; --camera-serial overrides the wrist entry
        # for the single-camera case.
        serials = None
        if args.camera_serial is not None:
            if camera_names != ["wrist"]:
                raise SystemExit(
                    "--camera-serial names one device but --cameras asks for "
                    f"{', '.join(camera_names)}. Set the serials in "
                    "'camera calibration/calib_config.py' instead — that is "
                    "where every other script reads them from.")
            serials = {"wrist": args.camera_serial}
        elif camera_names == ["wrist"]:
            import pyrealsense2 as _rs
            _devs = [(d.get_info(_rs.camera_info.serial_number),
                      d.get_info(_rs.camera_info.usb_type_descriptor))
                     for d in _rs.context().query_devices()]
            if len(_devs) > 1:
                listing = "\n  ".join(f"{s}  usb {u}" for s, u in _devs)
                print(f"[camera] {len(_devs)} RealSense devices attached:\n  "
                      f"{listing}\n[camera] using the 'wrist' serial from "
                      "calib_config.CAMERA_SERIALS.")

        if args.serial:
            # Layered ON TOP of calib_config's table rather than replacing it,
            # so overriding the tripod cannot silently unset the wrist. This is
            # the flag for "I swapped the body on the tripod" — the serial is
            # what identifies a camera, and a new body is a new serial.
            sys.path.insert(0, str(CALIB_DIR))
            import calib_config as _cfg          # noqa: E402
            serials = dict(serials if serials is not None else _cfg.CAMERA_SERIALS)
            for item in args.serial:
                role, _, sn = item.partition("=")
                if not sn:
                    raise SystemExit(
                        f"--serial wants ROLE=SERIAL, got {item!r}. Roles: "
                        f"{', '.join(sorted(_cfg.CAMERA_SERIALS))}.")
                serials[role.strip()] = sn.strip()
                print(f"[camera] {role.strip()} serial -> {sn.strip()} "
                      "(override; calib_config.py untouched)")

        rigs = build_rigs(
            camera_names,
            T_hand_cam_wrist=T_hand_cam,
            fixed_session=args.calib_session,
            serials=serials,
            color_size=(args.camera_width, args.camera_height),
            depth_size=(args.depth_width, args.depth_height),
            fps=args.camera_fps,
            exclude_robot=not args.no_robot_exclusion,
            cluster=not args.no_cluster,
            wrist_seg_px=args.wrist_seg_px,
            fixed_seg_px=args.fixed_seg_px,
            hand_margin_px=args.hand_margin_px,
            camera_model=args.camera_model,
        )

        for rig in rigs:
            try:
                rig.camera.start()
            except RuntimeError as err:
                # librealsense's two failure modes here look alike but mean
                # opposite things, and neither message names the stream at fault.
                modes = (f"color {args.camera_width}x{args.camera_height} + depth "
                         f"{args.depth_width}x{args.depth_height} @ {args.camera_fps}fps")
                if "no device connected" in str(err).lower():
                    # NOT a link fault, whatever the generic message below says.
                    # librealsense raises this when the requested SERIAL is not
                    # among the attached devices, so the useful answer is which
                    # ones are — the previous wording sent the reader off
                    # checking cables for a camera that was simply unplugged.
                    try:
                        import pyrealsense2 as _rs
                        attached = [
                            (d.get_info(_rs.camera_info.serial_number),
                             d.get_info(_rs.camera_info.name))
                            for d in _rs.context().query_devices()]
                    except Exception:
                        attached = []
                    listing = ("\n  ".join(f"{s}  {n}" for s, n in attached)
                               if attached else "(none)")
                    hint = (f"no device with that serial is attached. Attached "
                            f"now:\n  {listing}\n"
                            "Serials come from 'camera calibration/"
                            "calib_config.py'. If the camera you want IS in that "
                            f"list, ask for it by role: --cameras <role> (this "
                            f"run asked for {', '.join(camera_names)}).")
                elif "resolve" in str(err).lower():
                    hint = ("the device does not offer that combination. Color "
                            "and depth mode lists differ — the D435 has 424x240 "
                            "color but no 424x240 depth. `rs-enumerate-devices "
                            "-m` lists both.")
                elif len(rigs) > 1:
                    hint = ("the modes were accepted but no frames arrived. With "
                            "two D435s this is usually USB BANDWIDTH, not the "
                            "cable: two 640x480 colour+depth streams at 30 fps "
                            "exceed what one USB3 controller reliably carries. "
                            "Put the cameras on separate controllers (not just "
                            "separate ports), or drop --camera-fps to 15. Check "
                            "usb_type_descriptor reads 3.x for BOTH.")
                else:
                    hint = ("the modes were accepted but no frames arrived, which "
                            "is a link/power fault rather than a config one. "
                            "Check that usb_type_descriptor reads 3.x: a D435 "
                            "that enumerates at 2.1 (charge-only cable, USB2 "
                            "port, or a hub) often advertises modes it then "
                            "cannot stream at all.")
                raise RuntimeError(
                    f"RealSense '{rig.name}' (serial {rig.serial}) failed to "
                    f"start {modes} ({err}) — {hint}") from err
            # After start(), because the body is read off the live device. This
            # is what makes swapping a D455 onto the tripod a hardware change
            # rather than a code change: the near-depth floor follows the body.
            print(f"[camera] {apply_camera_model(rig, args.camera_model)}")
            if rig.kind == "fixed" and args.calib_session:
                mismatch = check_session_camera(args.calib_session, rig)
                if mismatch:
                    raise SystemExit(
                        f"\n[calibration] CAMERA MISMATCH\n    {mismatch}\n\n"
                        "    In 'camera calibration/':\n"
                        "      python generate_color_intrinsics.py --session "
                        "<new> --role tripod\n"
                        "      python capture_image_and_pose.py --session <new> "
                        "--role tripod\n"
                        "      python calibrate.py --session <new>\n"
                        "      python validate_calibration.py --session <new>\n")
            p = rig.params
            print(f"[camera] {rig.name:8s} serial={rig.serial}  {rig.kind}  "
                  f"seg={p.seg_input_px}px  "
                  + (f"cluster={p.cluster_voxel_m * 1e3:.0f}mm" if p.cluster_voxel_m
                     else "cluster=off")
                  + f"  margin={p.hand_margin_px}px"
                  + f"  r_obj={p.object_max_radius_m:.2f}m")

        # A FIXED CAMERA CANNOT BE PLACED WITHOUT THE LIVE POSE, dry run or not.
        # Its chain is inv(T_base_hand) @ T_base_color, so substituting identity
        # does not degrade the geometry, it changes the frame: the cloud lands in
        # the BASE frame while the gripper wireframe and the exclusion boxes are
        # still drawn at the origin of the frame the cloud is supposed to be in.
        # The result is a view with the boxes down at the robot's mounting plate,
        # i.e. below the table, which reads as a calibration fault and is not one.
        #
        # So --dry-run still SUBSCRIBES when a fixed camera is present. It stays
        # a dry run in the only sense that matters: `pub` is never advertised, so
        # there is nothing that could command the arm.
        needs_pose = any(r.T_base_cam is not None for r in rigs)
        if not args.dry_run or needs_pose:
            client = roslibpy.Ros(host=args.rosbridge_host, port=args.rosbridge_port)
            client.run()
            for _ in range(50):
                if client.is_connected:
                    break
                time.sleep(0.1)
            if not client.is_connected:
                if args.dry_run:
                    print(f"\n[pose] *** NO ROSBRIDGE AT "
                          f"{args.rosbridge_host}:{args.rosbridge_port} ***\n"
                          "    A fixed camera needs the live robot pose to be "
                          "placed at all. Without\n"
                          "    it the cloud is in the BASE frame while the "
                          "gripper and boxes are drawn\n"
                          "    at the hand frame's origin — expect them below "
                          "the table. Wrist-only\n"
                          "    runs are unaffected.\n", flush=True)
                    client = None
                else:
                    raise RuntimeError(
                        f"Could not connect to rosbridge at "
                        f"{args.rosbridge_host}:{args.rosbridge_port}")

        if client is not None:
            sub = roslibpy.Topic(client, CURRENT_POSE_TOPIC, POSE_MSG_TYPE)
            sub.subscribe(pose_cb)
            if not args.dry_run:
                pub = roslibpy.Topic(client, TARGET_POSE_TOPIC, POSE_MSG_TYPE)
                pub.advertise()
                gripper_sub = roslibpy.Topic(
                    client, GRIPPER_STATE_TOPIC, GRIPPER_STATE_TYPE)
                gripper_sub.subscribe(gripper_state_cb)
            # Never armed in a dry run, whatever --enable-gripper says: the
            # point of the flag is that nothing can act on the robot.
            gripper = FrankaGripper(
                client, (args.enable_gripper or args.home_gripper)
                and not args.dry_run)

            print(f"Connected to rosbridge at {args.rosbridge_host}:{args.rosbridge_port}"
                  + ("  (READ-ONLY: dry run, subscribed for the robot pose only)"
                     if args.dry_run else ""))

            # WHICH FRAME IS /cartesian_pose PUBLISHING? Asked of the robot, not
            # assumed, because getting it wrong is a constant ~10 cm error that
            # produces no symptom except the policy grasping short.
            measured = measure_ee_offset_z(client)
            if measured is None:
                print(f"[frames] no {FRANKA_STATE_TOPIC} — falling back to "
                      f"ee_offset_z = {ee_offset_z:.4f} m. If the arm grasps "
                      "short, this is the first thing to check.")
            elif args.ee_offset_z is None:
                ee_offset_z = measured
                frame = ("panda_hand" if abs(measured) < 1e-4
                         else f"{measured * 1000:.1f} mm ahead of panda_hand")
                print(f"[frames] F_T_EE says /cartesian_pose publishes {frame}; "
                      f"ee_offset_z = {measured:.4f} m")
            elif abs(measured - args.ee_offset_z) > 1e-3:
                print(f"\n[frames] *** --ee-offset-z {args.ee_offset_z:.4f} "
                      f"DISAGREES WITH THE ROBOT ***\n"
                      f"    F_T_EE reports {measured:.4f} m. Using your value.\n"
                      f"    A {abs(measured - args.ee_offset_z) * 1000:.0f} mm "
                      "error here moves the whole cloud along the approach\n"
                      "    axis, so the object renders nearer the fingers than "
                      "it is and the\n"
                      "    policy closes short. Drop the flag to use the "
                      "robot's own value.\n", flush=True)
            T_hand_ctrl = z_offset_transform(ee_offset_z)
            T_ctrl_hand = invert_transform(T_hand_ctrl)

            print("Waiting for current pose...")
            t0 = time.time()
            while current_msg is None and time.time() - t0 < 10.0:
                time.sleep(0.05)
            if current_msg is None:
                if not args.dry_run:
                    raise RuntimeError(f"No message on {CURRENT_POSE_TOPIC}")
                print(f"[pose] nothing on {CURRENT_POSE_TOPIC}; the fixed "
                      "camera's cloud will be in the BASE frame and the boxes "
                      "will draw below the table.")

            if gripper.enabled and not gripper.server_is_up():
                print("\n[gripper] *** NO GRASP ACTION SERVER ***\n"
                      f"    Nothing is publishing {GRIPPER_GRASP_STATUS_TOPIC}, so "
                      "the goals this\n"
                      "    sends will be accepted by rosbridge and acted on by "
                      "nobody. The\n"
                      "    usual cause is a controller launched without "
                      "load_gripper:=True —\n"
                      "    note that /franka_gripper/joint_states keeps "
                      "publishing either way,\n"
                      "    so the width readback is NOT evidence the gripper is "
                      "commandable.\n"
                      "    Continuing; the CLOSE will simply not happen.\n",
                      flush=True)

            # OPEN FIRST, ALWAYS. The episode ends on a close, so any run after
            # one that grasped starts with the fingers shut — and a shut gripper
            # lies to the policy through robot_state[25] and collapses the finger
            # exclusion boxes onto the object. See FrankaGripper.prepare.
            if gripper.enabled:
                gripper.prepare(home_first=args.home_gripper)

            # Whether the width is actually arriving matters more than it looks:
            # read_gripper_norm falls back to 1.0 (open) in silence, which is
            # right for the approach but would place the finger exclusion boxes
            # at the open position no matter where the fingers really are.
            if gripper_finger_m is None:
                print("[gripper] WARNING: nothing on "
                      f"{GRIPPER_STATE_TOPIC} — width falls back to fully open. "
                      "Check the controller was launched with load_gripper:=True.")
            else:
                print(f"[gripper] finger width {gripper_finger_m * 1000:.1f} mm "
                      f"per finger ({read_gripper_norm(True):.2f} normalised)")
                # A closed gripper is not cosmetic, and without --enable-gripper
                # there is nothing this can do about it except say so.
                if read_gripper_norm(True) < 0.8:
                    print("[gripper] *** THE GRIPPER IS NOT OPEN ***\n"
                          "    robot_state[25] reads closed for the whole "
                          "approach, which the policy\n"
                          "    never saw in training (the episode ENDS at the "
                          "close), and the finger\n"
                          "    exclusion boxes collapse onto the object and "
                          "delete it. Expect the arm\n"
                          "    to move oddly and not toward the object. Re-run "
                          "with --enable-gripper\n"
                          "    (which opens it at startup), or open it by hand.",
                          flush=True)

            T0 = pose_msg_to_matrix(current_msg) @ T_ctrl_hand
            print(f"Start panda_hand pose: xyz=({T0[0,3]:+.3f}, {T0[1,3]:+.3f}, "
                  f"{T0[2,3]:+.3f})  <- sanity-check this against the real flange "
                  "before trusting --ee-offset-z")

            if args.home or args.home_only:
                hp = T_BASE_HAND_HOME[:3, 3]
                # No prompt: --home IS the confirmation, and there is nothing to
                # decide at this point that was not decided by typing the flag.
                # The gate that matters moved to where a gate is actually useful
                # — the policy does not start until you press 's', with the
                # cameras and the cloud already up so you can see what it is
                # about to act on. Homing without --home is still one key ('h')
                # away in the loop.
                print(f"Homing to the sim's start pose, base frame "
                      f"xyz=({hp[0]:+.3f}, {hp[1]:+.3f}, {hp[2]:+.3f}), as an "
                      "interpolated Cartesian path. CLEAR THE WORKSPACE.",
                      flush=True)
                publish_seq = go_home(pub, T_ctrl_hand, T_hand_ctrl, publish_seq,
                                      droop, creep=not args.no_creep,
                                      stream=not args.home_stepwise,
                                      speed_m_s=args.home_speed,
                                      tol_m=args.home_tol)
                if args.home_only:
                    return

        # The finger cut is gated on the housing box as well, because both are
        # "keep the robot out of the object class" and having one on with the
        # other off is a state nobody asks for on purpose.
        finger_boxes = (FINGER_EXCLUSION_MODES[args.finger_boxes]
                        if (not args.no_robot_exclusion
                            and any(r.exclude_robot for r in rigs))
                        else None)
        perception = MultiCameraPerception(
            rigs,
            build_segmenter(args, rigs, device, hand_seg_model=hand_seg_model),
            per_camera_cap=args.per_camera_cap,
            arm_rejection=not args.no_arm_rejection,
            arm_offset_m=args.arm_offset,
            arm_lateral_m=args.arm_lateral,
            arm_below_m=args.arm_below,
            finger_exclusion=finger_boxes,
        )
        print(describe_segmenter(args))
        if args.segmentation == "sam2":
            print("[perception] arm rejection: off (the object mask already "
                  "excludes the forearm)")
        else:
            print(f"[perception] arm rejection: "
                  + (f"on, capsule {args.arm_offset:.3f} m behind the hand / "
                     f"{args.arm_lateral:.3f} m sideways"
                     if not args.no_arm_rejection else "off"))
        # Printed rather than left implicit: span mode changes what the policy
        # sees at contact, and a run whose log does not say which was on is a
        # run you cannot compare to another.
        if finger_boxes is None:
            print("[perception] finger boxes: off")
        elif finger_boxes.span_gap:
            print("[perception] finger boxes: SPAN — one box through the jaw "
                  "gap; the object goes with the fingers at contact")
        else:
            print("[perception] finger boxes: split — two boxes, jaw gap spared")

        # The SAME window test_perception_viz.py opens, from the same module.
        # These were two implementations that drew the same data differently
        # until the runner's lacked the white raw-scene cloud, which is the one
        # thing that shows whether the coloured cloud is in the right PLACE. A
        # debug view that renders unlike the thing being debugged is worth less
        # than one that renders identically, so there is now only one.
        viewer = DualCloudWindow(
            camera_names=[r.name for r in rigs],
            # Draw the box only when it is actually filtering, so what you see
            # is what is running.
            exclusion_box=(ROBOT_EXCLUSION
                           if (not args.no_robot_exclusion
                               and any(r.exclude_robot for r in rigs))
                           else None),
            # The same object the filter is using, so the wireframe can never
            # show you a mode you are not running.
            finger_boxes=finger_boxes,
            context_max=args.context_max,
            enabled=args.show_cloud,
        )
        viewer.show_context = not args.no_context
        cloud_min_dt = 1.0 / max(args.cloud_update_hz, 1e-3)
        last_cloud_draw = 0.0
        if args.show_cloud:
            print("cloud keys (in ANY window): c = colour by class/camera   "
                  "w = white scene cloud on/off   z / x = roll   r = drag mode")

        if args.step_mode:
            print(f"STEP MODE ({args.max_steps} policy steps max). The overlay "
                  "previews each predicted action; press SPACE in the image "
                  "window to execute one step, 'h' to re-home, 't' to STOP, "
                  "'q' to quit.")
        else:
            print(f"CONTINUOUS MODE ({args.max_steps} policy steps max). Steps "
                  "execute as soon as they are predicted. 'h' to re-home, "
                  "'q' or Esc to quit.")
        print("Press 's' in any window to start the policy. Until then the "
              "cameras and the cloud run but the robot is not commanded.",
              flush=True)

        # Fixed-rate control, or None for the settle path. Built here rather
        # than at parse time because it needs the publisher, and it holds state
        # across ticks (the tick clock and the last command, which is what the
        # droop estimate is learned from).
        rate_cmd = None
        if args.control == "rate" and not args.dry_run and pub is not None:
            rate_cmd = RateCommander(pub, args.rate_hz, droop)
            print(f"CONTROL: fixed rate, {args.rate_hz:.2f} Hz "
                  f"({1000/args.rate_hz:.0f} ms/step) — one command per step, "
                  "no arrival check. The arm keeps moving between steps; this "
                  "is the loop the policy was trained in.", flush=True)
            if droop is not None and not droop.enabled:
                print("CONTROL: WARNING — droop compensation is off. In this "
                      "mode the target is rebuilt from the measured pose every "
                      "tick, so a command too small to move the arm repeats "
                      "forever instead of accumulating. Expect stalls on the "
                      "small steps near the object.", flush=True)
        elif args.control == "rate":
            print("CONTROL: --control rate ignored (--dry-run or no robot)",
                  flush=True)
        else:
            print("CONTROL: settle — each step blocks until the arm arrives and "
                  "holds still. Smooth motion is --control rate.", flush=True)

        # THE ABORT POLLER. Installed here because it needs both key sources.
        # It runs from inside settle(), go_home() and the gripper waits, where
        # the main loop's key handling is not running at all — which is exactly
        # when the robot is moving and an abort is worth having.
        def _poll_keys() -> bool:
            hit = False
            now = time.time()
            k = cv2.waitKey(1) & 0xFF
            if 32 <= k < 127:
                c = chr(k)
                if c in STOP_KEYS:
                    request_stop(c, now)
                    hit = True
                else:
                    _swallowed_keys.append(c)
            elif k == 27:
                _swallowed_keys.append("q")
            if viewer.enabled:
                for c in viewer.drain_keys():
                    if c in STOP_KEYS:
                        request_stop(c, now)
                        hit = True
                    else:
                        _swallowed_keys.append(c)
            return hit

        globals()["_stop_poller"] = _poll_keys

        # ---- the session and its recorder ----------------------------------
        # Constructed here, before announce_end and start_episode, so both can
        # close over `exp`. The recorder is optional: --exp-mode without
        # --exp-out runs the whole key flow and writes nothing, which is what a
        # rehearsal wants.
        exp = (ExperimentSession(exp_labels, args.max_steps, adapter=adapter)
               if args.exp_mode else None)
        if exp is not None and args.exp_out:
            import exp_recorder as _expr
            # NOT the checkpoint's DATA.pc_channels. `policy_pc.f4` records
            # build_policy_cloud's output, whose width is fixed at 5 for every
            # policy; a regrasp run's 7-channel cloud is a separate stream the
            # adapter declares itself. Passing 7 in here declared a stride the
            # writer never used. See adapter_spec.
            fields, spec = _expr.adapter_spec(adapter)
            fps = (args.exp_video_fps if args.exp_video_fps
                   else (args.rate_hz if args.control == "rate" else 10.0))
            rec = _expr.ExpRecorder(
                Path(args.exp_out).expanduser() / exp.session_id, rigs,
                record_depth=not args.exp_no_depth, video_fps=fps,
                extra_fields=fields, arrays_spec=spec,
                attempt_csv_fields=_expr.attempt_fields(adapter),
                manifest=_expr.build_manifest(
                    args, rigs, adapter, exp.session_id,
                    extra={"transforms": {
                        "T_SIMWORLD_BASE": T_SIMWORLD_BASE.tolist(),
                        "T_ctrl_hand": T_ctrl_hand.tolist(),
                        "ee_offset_z": float(ee_offset_z)}}),
                row_constants={"runner": Path(sys.argv[0]).stem,
                               "control": args.control,
                               "cameras": ",".join(r.name for r in rigs),
                               "segmentation": args.segmentation,
                               "policy_run": str(getattr(adapter, "run_dir",
                                                         policy_dir)),
                               "anchor_ref": getattr(adapter, "anchor_ref", ""),
                               "d_rule": getattr(adapter, "d_rule", "")})
            exp.recorder = rec

        step = 0
        stop_reason = "max steps reached"
        # Previous step's target and arm pose, for the overshoot diagnostic in
        # the step log. Reset per episode with everything else.
        prev_target = prev_arm = None
        # The episode ending does not end the program. The robot stops taking
        # commands, but perception, the overlays and the 3D window keep running
        # so the scene the policy stopped on stays inspectable. 'q' is the only
        # way out. This matters most on a CLOSE: the frame the policy decided to
        # grasp from is exactly the one worth looking at, and it used to be torn
        # down the instant it appeared.
        episode_over = False
        # And it does not START until you say so. Perception, the overlays and
        # the 3D window come up first and run un-armed, so the scene the policy
        # will act on is on screen BEFORE anything can be published — which is
        # the moment you actually want a gate, unlike the old "type go to home"
        # prompt that fired at a blank terminal with no view of anything.
        # Continuous mode needs this or the first step lands the instant the
        # window opens; step mode gets it too, so both modes start the same way.
        armed = False

        # WHEN THE ARM WILL HOME ITSELF, or None for "it will not". Set only by
        # the CLOSE branch: a grasp is the one episode ending where the next
        # thing you want is always the same, and standing over the keyboard to
        # type 'h' with an object in the fingers is the part worth deleting.
        #
        # A DEADLINE, NOT A SLEEP. The loop keeps grabbing frames, segmenting
        # and reading keys through the whole wait, so 't' still stops, 'h' still
        # homes now, 's' still starts the next episode — and each of those
        # CLEARS this, because a key press that meant something else must not be
        # followed by the arm moving on its own a moment later. That is the
        # whole safety argument for the feature: nothing here can move the robot
        # that a keypress in the preceding seconds did not leave scheduled.
        auto_home_at: Optional[float] = None

        def announce_end(reason: str) -> None:
            print(f"Episode ended after {step} policy steps: {reason}")
            if exp is not None:
                return      # the session prints what to press; "'s' to run
                            # another episode" is false during a verdict wait
            print("Episode over - camera and cloud windows stay live. "
                  "'h' to re-home, 's' to run another episode, 'q' to quit.",
                  flush=True)

        def release_object() -> None:
            """Open the fingers at a moment the operator CHOSE.

            start_episode() re-opens them anyway — it has to, because a shut
            gripper reads as robot_state[25] = 0 for the whole next approach and
            collapses the finger exclusion boxes onto the object. But that
            moment is the next 's', with the arm at home 0.7 m up and about to
            move: the object would drop from there while the operator's
            attention was on the arm. Doing it on the verdict instead costs
            nothing, happens while the arm is stationary, and makes prepare()'s
            wait return on its first poll rather than after a full open.
            """
            if not gripper.enabled:
                return
            print("[exp] releasing — take the object.", flush=True)
            gripper.open()
            gripper.wait_for_width(GRIPPER_MAX_FINGER_M, timeout_s=4.0)

        def start_episode() -> None:
            """Begin a new episode without restarting the process.

            EVERYTHING PER-EPISODE IS RESET AND EVERYTHING LEARNED IS KEPT. The
            step budget, the stop reason and the stale-cloud caches belong to the
            episode that just ended; the droop estimate does not — it is a
            property of the arm, took several moves to converge, and throwing it
            away would make the first steps of every later episode lumpy again.

            The gripper is re-opened here rather than in the 'h' handler, because
            it has to happen whether or not you homed first. An episode ends on a
            CLOSE, so a second episode would otherwise begin with the fingers
            shut — which reads to the policy as robot_state[25] = 0 for the whole
            approach, and collapses the finger exclusion boxes onto the object.
            That is the same failure that made consecutive PROCESS runs behave
            oddly; restarting in-process reintroduces it by a shorter path.
            """
            nonlocal step, episode_over, stop_reason, armed, auto_home_at
            nonlocal prev_target, prev_arm
            prev_target = prev_arm = None
            # Pressing 's' inside the post-close wait means "go again from
            # here", not "go again and then wander home mid-approach".
            auto_home_at = None
            step = 0
            stop_reason = "max steps reached"
            episode_over = False
            armed = True
            # Stale per-camera clouds are kept across frames so a momentary
            # segmentation dropout does not blank the observation. Across
            # EPISODES they are a lie: the scene has changed.
            perception.reset()
            adapter.reset()
            if rate_cmd is not None:
                rate_cmd.reset()
            if gripper.enabled:
                gripper.prepare(home_first=False)
            print(f"\n=== new episode (max {args.max_steps} steps) ===",
                  flush=True)

        while True:
            if not episode_over and step >= args.max_steps:
                episode_over = True
                announce_end(stop_reason)
                if exp is not None and exp.phase == "RUNNING":
                    # A TIMEOUT IS A RECORDED FAILURE, and it stops the arm
                    # where it stands for the same reason 'f' does: the pose the
                    # policy ran out of budget in is the pose worth looking at.
                    # freeze_arm does not block, so this is safe at the loop
                    # top; it is also necessary under --control rate, where the
                    # last commanded target is one the arm is still travelling
                    # toward.
                    publish_seq = freeze_arm(pub, publish_seq)
                    gripper.stop()
                    armed = False
                    auto_home_at = None
                    exp.attempt_ended("timeout", steps=step, verdict="fail",
                                      froze=True)

            # Keyed on whether a pose EXISTS, not on --dry-run. A dry run with a
            # fixed camera subscribes precisely so this branch can be the real
            # one; identity is the last resort, and it silently reframes the
            # cloud rather than merely making it stale.
            if current_msg is not None:
                T_base_hand = pose_msg_to_matrix(copy.deepcopy(current_msg)) @ T_ctrl_hand
            elif args.dry_run:
                T_base_hand = np.eye(4)
            else:
                time.sleep(0.01)
                continue

            # ---- segment, deproject and fuse every camera into panda_hand ----
            # The pose is read BEFORE the frames are grabbed and used to place
            # the fixed camera's points, so a stale pose shifts that camera's
            # cloud bodily. The wrist camera is immune — its extrinsics are
            # constant — which is the practical reason it stays the anchor view.
            # One gripper reading per iteration, shared by the cloud and the
            # robot state. Reading it twice would let the two disagree by a
            # frame, and the finger exclusion boxes are placed from it.
            gripper_norm = read_gripper_norm(assume_open=True)
            if exp is not None:
                # The SAME reading the cloud and robot_state are built from, on
                # purpose: a second read of the global would be a frame out and
                # the recorded closure would not be the one the policy saw.
                exp.note_gripper(gripper_finger_m)
            _t = time.time()
            fused = perception.observe(
                T_base_hand, gripper_norm * GRIPPER_MAX_FINGER_M)
            ms_obs = (time.time() - _t) * 1e3
            ms_pol = 0.0
            object_policy = fused.object_xyz
            hand_policy = fused.hand_xyz
            have_obs = fused.usable

            # ---- PREDICT (never moves the robot) ----
            # Prediction is separated from execution so step mode can show you
            # what the policy wants to do before anything is published. In
            # continuous mode the two phases just run back to back.
            grasp_close = False
            clamped = False
            delta6 = None
            T_base_ctrl_target = None
            # Reset per iteration like the others. It was not, which was
            # harmless while only the execute path read it (that path cannot
            # run on a CLOSE step) — but anything reading it on a CLOSE step or
            # the first iteration would get the previous step's target or an
            # UnboundLocalError, and the recorder reads it every step.
            T_base_hand_target = None

            if have_obs:
                # return_index costs nothing and is what lets the 3D view colour
                # each of the 1024 rows by the camera it came from — the sampling
                # is otherwise where provenance is lost.
                pc, oi, hi = build_policy_cloud(object_policy, hand_policy,
                                                return_index=True)
                if viewer.enabled and time.time() - last_cloud_draw >= cloud_min_dt:
                    # Deprojecting every camera's full depth frame is the one
                    # genuinely expensive part of drawing, so it is behind the
                    # same throttle as the redraw and skipped entirely when the
                    # white cloud is hidden. It reuses perception.last_depths —
                    # the exact frames this observation came from, rather than a
                    # fresh grab, which would be a different instant.
                    ctx = None
                    if viewer.show_context:
                        parts = [context_cloud(rig, perception.last_depths[rig.name],
                                               T_base_hand, args.context_stride,
                                               args.context_radius)
                                 for rig in rigs
                                 if rig.name in perception.last_depths]
                        parts = [q for q in parts if len(q)]
                        ctx = np.concatenate(parts) if parts else None
                    viewer.update(pc, source_for_cloud(
                        oi, hi, fused.object_source, fused.hand_source,
                        NUM_OBJECT_POINTS, NUM_HAND_POINTS), ctx)
                    last_cloud_draw = time.time()
                rs = build_robot_state(T_base_hand, gripper_norm)

                _t = time.time()
                try:
                    action = adapter.act(pc, rs, fused=fused,
                                         T_base_hand=T_base_hand)  # [7], ch6
                except ObservationNotUsable as exc:
                    # HOLD THE FRAME RATHER THAN DIE. The observation cleared
                    # `usable` and still was not enough for the adapter — see
                    # ObservationNotUsable. Treating it as "nothing segmented"
                    # reuses the path that already exists for that: no target is
                    # published, no step is consumed, and the next perception
                    # pass is a fresh chance. Under --exp-mode the alternative
                    # was losing the session to a traceback.
                    have_obs = False
                    action = None
                    _warn_hold(f"skipped a frame: {exc}")
                ms_pol = (time.time() - _t) * 1e3

                grasp_close = bool(have_obs and action[6] < 0.5)
                if have_obs and not grasp_close:
                    delta6, clamped = clamp_action_delta(action[:6])
                    # WHICH POSE THE DELTA HANGS OFF, and it is not obvious.
                    #
                    # The action is "from where you are, move like this", and in
                    # sim there is no gap between the two readings of "where you
                    # are": the observation and the command are the same
                    # instant. On the robot they are not. `T_base_hand` was read
                    # BEFORE observe(), and observe() plus inference is ~120 ms
                    # of a 150 ms period — so by the time this target is
                    # published the arm has covered most of a step under the
                    # PREVIOUS command, and the pose the delta is hung off is
                    # that far behind it.
                    #
                    # While consecutive deltas agree, that costs nothing: the
                    # arm is travelling the right way and simply keeps going.
                    # When they disagree — which near the object is most ticks —
                    # the new target can land BEHIND the arm, so the arm stops
                    # or reverses inside the period, `achieved` collapses, and
                    # the lead estimator is fed a stall that never happened.
                    # `--rate-anchor now` re-reads the pose here instead, giving
                    # every step a full delta of fresh travel from wherever the
                    # arm actually is.
                    #
                    # DEFAULT IS `obs`, which is what the policy was trained
                    # against and what every run so far used. This is an A/B, to
                    # be judged on the `moved=`/`res=` columns, not a fix that
                    # has been shown to be one.
                    T_anchor = T_base_hand
                    if args.rate_anchor == "now" and current_msg is not None:
                        T_anchor = (pose_msg_to_matrix(copy.deepcopy(current_msg))
                                    @ T_ctrl_hand)
                    T_base_hand_target = T_anchor @ unpack_action(delta6)
                    T_base_ctrl_target = clamp_target_pose(T_base_hand_target @ T_hand_ctrl)

            # ---- RECORD ----
            # Before the display block on purpose. `overlay_mask` copies today,
            # but the HUD path draws with cv2.putText into the first rig's
            # `view`, which is one refactor away from drawing into the frame
            # being recorded. Recording first makes the ordering explicit
            # rather than lucky.
            if rec is not None:
                rec.on_step(
                    step=step, t_mono=_t, perception=perception, fused=fused,
                    T_base_hand=T_base_hand, gripper_norm=gripper_norm,
                    pc=pc if have_obs else None,
                    robot_state=rs if have_obs else None,
                    action=action if have_obs else None,
                    have_obs=have_obs, armed=armed,
                    executed=bool(armed and not episode_over and have_obs),
                    grasp_close=grasp_close, clamped=clamped,
                    ms_obs=ms_obs, ms_pol=ms_pol, adapter=adapter,
                    grip_logit=getattr(adapter, "grip_logit", float("nan")),
                    target=T_base_hand_target,
                    d_world=getattr(adapter, "d_world", None))
                rec.check()

            # ---- display ----
            # One overlay window per camera, so a camera that has stopped
            # contributing (occluded, mis-segmented, or unplugged) is visible
            # rather than being silently averaged into the union.
            overlay = None
            for rig in rigs:
                color_bgr, hand_mask = perception.last_frames[rig.name]
                view = overlay_mask(color_bgr, hand_mask)
                # The object mask, where there is one, tinted separately. With
                # two masks the interesting failure is them disagreeing about
                # the same pixels, and one colour cannot show that.
                obj_mask = perception.last_object_masks.get(rig.name)
                if obj_mask is not None:
                    view = overlay_mask(view, obj_mask, colour=(0, 0, 255))
                d = fused.per_camera[rig.name]
                cv2.putText(view, f"{rig.name}  obj={d['object']} hand={d['hand']}"
                            + (f"  -{d['cluster_dropped']} declust"
                               if d["cluster_dropped"] else "")
                            + (f"  [{d['cluster_fallback']}]"
                               if d["cluster_fallback"] not in (None, "disabled")
                               else "")
                            + (f"  -{d['robot_pts_removed']} robot"
                               if d["robot_pts_removed"] else "")
                            + (f"  -{d['finger_pts_removed']} finger"
                               if d["finger_pts_removed"] else "")
                            + ("  STALE" if d["used_last_hand"] or d["used_last_object"]
                               else "")
                            # A tracker being re-seeded every frame produces
                            # entirely plausible point counts, so this is the
                            # only place that failure is visible.
                            + ("  RESEED" if d.get("seg_reseeded") else "")
                            + (f"  x{d['seg_reseeds']}"
                               if d.get("seg_reseeds") else ""),
                            (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1, cv2.LINE_AA)
                if overlay is None:
                    overlay = view          # the wrist view carries the HUD below
                else:
                    cv2.imshow(f"cam: {rig.name}", view)

            lines = [
                f"step {step}/{args.max_steps}  "
                f"obj={len(object_policy)}{_range_txt(object_policy)} "
                f"hand={len(hand_policy)}{_range_txt(hand_policy)}   "
                f"{fused.summary()}",
            ]
            if exp is not None:
                # Index 1, so it inherits the existing yellow highlight and the
                # adapter's direction line sits right under it. That pairing is
                # the visual confirmation that set_command took effect: if
                # adapter.hud() still names the previous command after an 's',
                # the re-point failed.
                lines.append(exp.status_line(
                    step, args.max_steps,
                    auto_home_in=(None if auto_home_at is None
                                  else max(0.0, auto_home_at - time.time()))))
            extra = adapter.hud()
            if extra:
                lines.append(extra)
            if rec is not None:
                lines.append(rec.hud())
            if episode_over and exp is None:
                left = (None if auto_home_at is None
                        else max(0.0, auto_home_at - time.time()))
                lines.append(
                    f"EPISODE OVER: {stop_reason}  -  's' to run again"
                    + ("" if left is None
                       else f"   AUTO-HOME IN {left:.1f}s ('t' cancels)"))
            elif not armed and exp is None:
                # Still shows what the policy WOULD do, so you can watch the
                # prediction settle before arming rather than after.
                lines.append("NOT STARTED - press 's'"
                             + ("" if not have_obs else
                                ("   (would CLOSE)" if grasp_close else
                                 f"   (would move {np.linalg.norm(delta6[:3])*100:.1f}cm)")
                                + _grip_txt(adapter)))
            elif not have_obs:
                lines.append("NO OBSERVATION - holding")
            elif grasp_close:
                lines.append(f"PENDING: CLOSE GRIPPER{_grip_txt(adapter)}")
            else:
                lines.append(
                    f"PENDING: d={np.linalg.norm(delta6[:3])*100:.1f}cm "
                    f"r={np.rad2deg(np.linalg.norm(Rot.from_matrix(unpack_action(delta6)[:3, :3]).as_rotvec())):.1f}deg"
                    + ("  CLAMPED" if clamped else "")
                    + _grip_txt(adapter))
            lines.append(
                exp.key_line() if exp is not None
                else "s=new episode  h=home  t=STOP  q=quit" if episode_over
                else "s=start  h=home  t=STOP  q=quit" if not armed
                else ("SPACE=execute  t=STOP  h=home  q=quit" if args.step_mode
                      else "t=STOP  q=quit  h=home"))

            for i, text in enumerate(lines):
                cv2.putText(overlay, text, (10, 22 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255) if i == 1 else (0, 255, 0), 1, cv2.LINE_AA)
            # Named for its camera like the others, so with several cameras it is
            # obvious which stream you are looking at. Keys are read from this
            # window (cv2.waitKey is global, but this is the one with the HUD).
            cv2.imshow(f"cam: {rigs[0].name}  [keys here]", overlay)

            # ---- key handling, with the 3D window pumped while we idle ----
            # A single poll() per iteration gives the viewer ~10 Hz, because one
            # iteration is a camera grab plus a segmentation forward plus a
            # policy forward. At 10 Hz a trackpad drag is sampled so coarsely
            # that the view barely responds, which reads as "laggy and won't
            # rotate". In STEP MODE the loop is waiting for a human anyway, so
            # that wait is spent pumping the window at full rate instead of
            # re-running perception nobody asked for. Any keypress breaks out
            # immediately, so it costs no responsiveness at the keyboard.
            #
            # In continuous mode the pump is skipped: there the loop rate IS the
            # control rate, and slowing it to make a debug view smoother would
            # be the wrong trade.
            key = 255
            view_keys: list = []
            # ... and skipped when there is no 3D window either, which is the
            # common case: without --show-cloud, tick() and drain_keys() are both
            # no-ops, so the pump is 200 ms of pure idle per iteration on top of
            # ~50 ms of real work. That alone took the camera preview from ~20 Hz
            # to ~4 Hz, which reads as "perception is running slow" even though
            # perception measures 33-47 ms.
            # Before the episode starts and after it ends the loop is waiting
            # for a human in BOTH modes — there is no control rate left to
            # protect — so continuous mode joins step mode in spending that wait
            # on the 3D window. That is the whole point of the window being up:
            # an un-pumped cloud you cannot orbit is not much better than a
            # closed one, and setting up the scene before pressing 's' is
            # exactly when you want to orbit it.
            pump_until = time.time() + (VIEWER_PUMP_S
                                        if (args.step_mode or episode_over
                                            or not armed)
                                        and viewer.enabled
                                        else 0.0)
            while True:
                viewer.tick()
                # The 3D window's keys are drained into the same list the OpenCV
                # windows feed, so every action works from whichever window has
                # focus. Without this the roll keys only worked while a camera
                # view was focused, which meant clicking away from the cloud to
                # rotate the cloud.
                view_keys.extend(viewer.drain_keys())
                k = cv2.waitKey(1) & 0xFF
                if k != 255 or view_keys:
                    key = k
                    break
                if time.time() >= pump_until:
                    break

            pressed = [chr(key)] if 32 <= key < 127 else []
            if key == 27:
                pressed.append("q")
            pressed.extend(view_keys)
            # Anything the abort poller read while the robot was moving. It has
            # to consume keys to see 't', so it hands the rest back here instead
            # of eating them.
            if _swallowed_keys:
                pressed.extend(_swallowed_keys)
                _swallowed_keys.clear()
            # A stop raised from INSIDE a motion never reached `pressed`: the
            # poller had to consume the key to see it. Put it back, so there is
            # ONE dispatch table rather than two, and so which key it was
            # survives — 't' is a user stop and 'f' is a recorded policy
            # failure, and they must not collapse into each other here.
            if stop_requested and stop_reason_key and stop_reason_key not in pressed:
                pressed.append(stop_reason_key)

            # View-only keys never touch the robot, so they are handled here and
            # dropped before the step keys below see them.
            for k in pressed:
                if k == "c":
                    viewer.colour_mode = ("camera" if viewer.colour_mode == "class"
                                          else "class")
                    print(f"colour by {viewer.colour_mode}")
                    last_cloud_draw = 0.0
                elif k == "w":
                    viewer.show_context = not viewer.show_context
                    print(f"white scene cloud "
                          f"{'on' if viewer.show_context else 'off'}")
                    last_cloud_draw = 0.0
                elif k == "m":
                    # 'm' and not 'r'. --exp-mode needs 'r' for RETRY, which is
                    # an action on the experiment and far more consequential
                    # than cycling how a mouse drag orbits the view. Moved
                    # unconditionally rather than per-mode, so a key means one
                    # thing everywhere.
                    viewer.cycle_rotate_mode()
                elif k == "z":
                    viewer.roll(-10.0)
                elif k == "x":
                    viewer.roll(+10.0)

            if "t" in pressed:
                # FIRST, before anything else in this branch: a stop that left a
                # pending auto-home armed would freeze the arm and then drive
                # it home a few seconds later, which is the exact opposite of
                # what 't' means.
                auto_home_at = None
                publish_seq = freeze_arm(pub, publish_seq)
                clear_stop()
                gripper.stop()
                if not episode_over:
                    stop_reason = "stopped by user ('t')"
                    episode_over = True
                    announce_end(stop_reason)
                armed = False
                if exp is not None and exp.phase == "RUNNING":
                    # 't' VOIDS the attempt rather than failing it: a failure is
                    # something the POLICY did and 'f' records that. The command
                    # is re-offered, so the next 's' retries the same one.
                    exp.attempt_voided(steps=step)
                print("[STOP] equilibrium re-commanded at the current pose. "
                      "This is a SOFT stop — the arm is holding, not braked. "
                      "Use the hardware E-stop if it is not enough.", flush=True)
                continue
            # ---- EXPERIMENT MODE: the outcome keys ----
            # AFTER 't', which outranks everything, and BEFORE 's', because
            # _swallowed_keys can deliver several keys in one batch and an 'f'
            # arriving alongside an 's' must be the reason THIS attempt ended,
            # not a key the next one inherits. It consumes the iteration either
            # way: "f s" must not start the next attempt before you have
            # re-homed by hand, and 'h' is dispatched below 's'.
            # RETRY, before 's': `_swallowed_keys` can deliver several keys in
            # one batch, and an "r s" batch must roll the cursor back BEFORE
            # 's' reads which bin is next — otherwise the retry silently starts
            # the following bin instead.
            if exp is not None and "r" in pressed:
                exp.retry_requested()
                continue
            if exp is not None and ("f" in pressed or "p" in pressed):
                verdict = "fail" if "f" in pressed else "pass"
                if exp.phase == "RUNNING":
                    if verdict == "pass":
                        print("[exp] 'p' means nothing yet — the policy has not "
                              "closed. 'f' fails it, 't' voids it.", flush=True)
                    else:
                        # THE KEY'S OWN PRESS TIME, not now: when 'f' arrived
                        # through the abort poller it interrupted a settle(),
                        # and `stop_at` is when that happened. Read it BEFORE
                        # clear_stop() wipes it.
                        t_f = stop_at if stop_at is not None else time.time()
                        auto_home_at = None
                        publish_seq = freeze_arm(pub, publish_seq)
                        clear_stop()
                        gripper.stop()
                        stop_reason = "policy failed ('f')"
                        episode_over = True
                        armed = False
                        announce_end(stop_reason)
                        exp.attempt_ended("policy_fail", steps=step,
                                          verdict="fail", t_end=t_f, froze=True)
                elif exp.phase == "VERDICT":
                    # A VERDICT DOES NOT STOP THE ARM, so this branch leaves
                    # `auto_home_at` alone — unlike 't', 's' and 'h', all of
                    # which clear it. An 'f' here says the GRASP failed, not
                    # that the carry-home should abort; press 't' as well if you
                    # want the arm stopped too.
                    if exp.verdict_given(verdict) and auto_home_at is None:
                        release_object()
                else:
                    print(f"[exp] nothing to judge ({exp.phase.lower()}) — "
                          "'s' starts the next attempt.", flush=True)
                continue

            if "q" in pressed:
                stop_reason = "user quit"
                break
            # Read from `pressed` rather than `key` so it works from the 3D
            # window too — which is where you will be looking when you decide
            # the scene is ready.
            if "s" in pressed:
                if exp is not None:
                    if exp.start_requested():       # False + a printed reason
                        # None whenever the policy takes no command — see
                        # ExperimentSession.__init__. Asking for it BEFORE
                        # attempt_started, which is what advances past it.
                        label = exp.next_label
                        if label is not None:
                            adapter.set_command(label)
                        # ALWAYS through start_episode(), never the bare
                        # `armed = True` path: adapter.act() runs on every
                        # iteration that has an observation, including before
                        # 's', so a regrasp adapter has already latched its
                        # direction off the idle scene. start_episode()'s
                        # adapter.reset() drops that latch, and its `continue`
                        # means the re-latch happens on a fresh observation
                        # with the NEW command.
                        start_episode()
                        exp.attempt_started()       # clock starts after the
                                                    # gripper open, not before
                    continue
                if episode_over:
                    start_episode()
                    # Fresh observation before acting. start_episode() re-opens
                    # the gripper, and the frame above it was segmented and its
                    # robot_state read while the fingers were still shut.
                    continue
                elif not armed:
                    armed = True
                    print("Policy started.", flush=True)
            if "h" in pressed:
                if exp is not None and exp.phase == "RUNNING":
                    print("[exp] refusing to home mid-attempt — the clock is "
                          "running. 'f' fails it, 't' voids it, then 'h'.",
                          flush=True)
                    continue
                auto_home_at = None          # you did it by hand; nothing left to do
                if args.dry_run:
                    print("[home] ignored in --dry-run")
                else:
                    publish_seq = go_home(pub, T_ctrl_hand, T_hand_ctrl,
                                          publish_seq, droop,
                                          creep=not args.no_creep,
                                          stream=not args.home_stepwise,
                                          speed_m_s=args.home_speed,
                                          tol_m=args.home_tol)
                continue

            # ---- AUTO-HOME AFTER A CLOSE ----
            # Placed after every key branch on purpose, so a key pressed on this
            # same iteration is honoured and this is not. `go_home` blocks while
            # it streams, and its own abort poller reads 't' throughout, so the
            # move is interruptible once it starts as well as before.
            if auto_home_at is not None and time.time() >= auto_home_at:
                # No --dry-run guard: `auto_home_s` is already None there, so a
                # deadline is never scheduled in the first place.
                auto_home_at = None
                print("[home] auto-homing now (policy closed the gripper)",
                      flush=True)
                publish_seq = go_home(pub, T_ctrl_hand, T_hand_ctrl,
                                      publish_seq, droop,
                                      creep=not args.no_creep,
                                      stream=not args.home_stepwise,
                                      speed_m_s=args.home_speed,
                                      tol_m=args.home_tol)
                # The verdict may have been given during the 5 s delay, in
                # which case the release was deferred to here so the object is
                # let go at HOME rather than at the grasp pose.
                if exp is not None and exp.phase != "VERDICT":
                    release_object()
                continue

            # ---- EXECUTE ----
            # Step mode gates every single motion on SPACE; nothing the policy
            # predicts reaches the robot until you ask for it.
            #
            # Before 's' and after the episode, everything above this line still
            # runs and nothing below it does: the policy keeps predicting and
            # the HUD keeps showing what it would do, but no target is ever
            # published and no step is consumed. 'h' is the one exception,
            # handled above — it is how you re-home before arming, and the usual
            # next thing you want after a grasp.
            if episode_over or not armed:
                continue
            if not have_obs:
                time.sleep(0.01)   # nothing segmented; does NOT consume a step
                continue
            if args.step_mode and key != ord(" "):
                continue

            if grasp_close:
                print(f"[{step:02d}] policy commanded CLOSE{_grip_txt(adapter)}",
                      flush=True)
                t_close = time.time()        # before the goal goes out
                gripper.close()
                stop_reason = "policy closed the gripper"
                episode_over = True
                announce_end(stop_reason)
                if exp is not None:
                    armed = False
                    # verdict=None means "a verdict is owed" -> phase VERDICT.
                    exp.attempt_ended("close", steps=step, verdict=None,
                                      t_end=t_close)
                if auto_home_s is not None:
                    auto_home_at = time.time() + auto_home_s
                    # LOUD, because the arm is now going to move with nobody
                    # touching anything. The delay exists so the fingers have
                    # settled on the object before it is carried anywhere, and
                    # so there is a window to change your mind.
                    print(f"[home] AUTO-HOME in {auto_home_s:.1f}s — the arm "
                          "will move on its own. 't' cancels, 'h' homes now, "
                          "'s' starts the next episode instead.", flush=True)
                continue

            pos = T_base_ctrl_target[:3, 3]
            # fused.summary() carries the per-camera split and the declustered
            # count. Without it a clean object cloud and one that fell back to
            # the sphere every frame print identically.
            # WHERE THE ARM ACTUALLY IS, and how it did against the LAST target.
            #
            # Without this the log cannot tell an oscillating controller from an
            # oscillating observation, because the target it prints is
            # arm_pose (*) delta — so a wobbling target is equally consistent
            # with the arm overshooting and with the policy changing its mind.
            # `res` is the signed residual to the previous target along the
            # direction that target was in: negative means the arm went PAST it.
            # BOTH IN panda_hand. `pos` above is the CONTROL frame, 103.4 mm
            # ahead of the hand on this robot, so measuring the arm against it
            # would report that constant offset as a residual on every step.
            arm = T_base_hand[:3, 3]
            hand_target = T_base_hand_target[:3, 3]
            diag = ""
            if prev_target is not None:
                to_prev = prev_target - prev_arm
                n_prev = float(np.linalg.norm(to_prev))
                if n_prev > 1e-9:
                    u = to_prev / n_prev
                    got = float((arm - prev_arm) @ u)
                    diag = (f"  arm=({arm[0]:+.3f},{arm[1]:+.3f},{arm[2]:+.3f})"
                            f" moved={got*1000:+6.1f}/{n_prev*1000:5.1f}mm"
                            f" res={(n_prev - got)*1000:+6.1f}mm")
            prev_target, prev_arm = hand_target.copy(), arm.copy()

            print(f"[{step:02d}] target xyz=({pos[0]:+.3f}, {pos[1]:+.3f}, "
                  f"{pos[2]:+.3f})  |d|={np.linalg.norm(delta6[:3]):.4f}m "
                  f"obj={len(object_policy):4d}{_range_txt(object_policy, True)}"
                  f" hand={len(hand_policy):4d}{_range_txt(hand_policy)}"
                  # HERE TOO, not only on the overlay. The HUD is drawn into an
                  # image; the terminal is what gets copied into a bug report,
                  # and "it would not close" is not a report without this number.
                  f"{_grip_txt(adapter)}"
                  f"  {fused.summary()}"
                  f"{'  CLAMPED' if clamped else ''}{diag}", flush=True)

            if not args.dry_run and rate_cmd is not None:
                _t = time.time()
                publish_seq, slept = rate_cmd.command(
                    T_base_ctrl_target, publish_seq)
                ms_move = (time.time() - _t) * 1e3
                print(f"     published, dwelt {slept*1000:.0f}ms "
                      f"(obs {ms_obs:.0f} + policy {ms_pol:.0f} inside the "
                      f"{1000/args.rate_hz:.0f}ms period)"
                      + (f"; {droop.describe()}" if droop is not None else ""),
                      flush=True)
            elif not args.dry_run:
                tol = step_tolerance(float(np.linalg.norm(delta6[:3])),
                                     args.step_tol_frac)
                _t = time.time()
                publish_seq, dpos, drot, passes, commands = move_to(
                    pub, T_base_ctrl_target, publish_seq, args.settle_timeout,
                    droop, STEP_CONVERGE_PASSES, tol,
                    creep=not args.no_creep)
                ms_move = (time.time() - _t) * 1e3
                # Always logged, not only on failure. How many commands the arm
                # was given IS what the motion looks like — 1 is one continuous
                # move, 3 is the jump-plus-two-adjustments people report — and
                # the lead beside it says whether that count is going to come
                # down: a lead that has stopped changing while commands stay
                # above 1 means the estimate has saturated, not converged.
                # WHERE THE STEP ACTUALLY GOES. Printed every step because the
                # answer is counter-intuitive and people reach for the network
                # first: perception and inference are tens of milliseconds, and
                # waiting for the arm to stop is seconds. See the note above
                # settle() on why we wait at all, and what the CVPR2023 loop
                # does instead (a fixed 0.15 s dwell, no convergence check).
                total = ms_obs + ms_pol + ms_move
                print(f"     moved in {commands} command"
                      f"{'' if commands == 1 else 's'}, "
                      f"{dpos*1000:.1f}mm short of {tol*1000:.0f}mm"
                      + (f", {droop.describe()}" if droop is not None else ""),
                      flush=True)
                print(f"     {total/1000:.2f}s  =  obs {ms_obs:.0f}ms + policy "
                      f"{ms_pol:.0f}ms + motion {ms_move:.0f}ms "
                      f"({100*ms_move/max(total,1e-9):.0f}% waiting for the arm)",
                      flush=True)
            step += 1

        # Only the quit path reaches here un-announced; an episode that ended on
        # its own said so at the time and has been idling ever since.
        if not episode_over:
            print(f"Episode ended after {step} policy steps: {stop_reason}")

    finally:
        # FIRST, before the cameras, the viewer or ROS. Nothing here depends on
        # any of them, and `__main__` leaves through exit_without_finalizing()
        # — os._exit(0), which flushes stdout and stderr and NOTHING ELSE. No
        # atexit hook runs and no daemon thread gets another slice, so a flush
        # that has not happened by the time this returns has not happened at
        # all. Putting it first means a failure in camera teardown cannot cost
        # a session's data.
        if exp is not None:
            try:
                exp.close(stop_reason)
            except Exception:
                traceback.print_exc()
        if rec is not None:
            try:
                rec.close()
            except Exception:
                traceback.print_exc()

        for rig in rigs:
            try:
                rig.camera.stop()
            except Exception:
                pass
        if viewer is not None:
            viewer.close()
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
    # Only on a clean return: an exception must still print its traceback and
    # exit normally. See exit_without_finalizing for why this is not just exit().
    exit_without_finalizing()
