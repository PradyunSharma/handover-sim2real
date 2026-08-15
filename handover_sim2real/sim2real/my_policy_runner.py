#!/usr/bin/env python3
"""
Run a Phase-1/4 BC policy (checkpoint/cp2 or cp3) on the physical FR3.

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
    python my_policy_runner.py --policy-dir checkpoint/cp3 \
           --cameras wrist,tripod --calib-session session_02 --home --step-mode
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple, Optional

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

from pointcloud_multicam import (  # noqa: E402
    NUM_HAND_POINTS,
    NUM_OBJECT_POINTS,
    PC_CHANNELS,
    ROBOT_EXCLUSION,
    HandSegmenter,
    MultiCameraPerception,
    build_policy_cloud,
    build_rigs,
    overlay_mask,
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
HOME_REFINE_PASSES = 6      # re-command home once the droop estimate exists
HOME_REFINE_TOL_M = 0.002   # stop refining below this (was SETTLE_POS_TOL_M=5 mm)
HOME_WAY_TOL_M = 0.008      # intermediate waypoints are a path, not a target

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
# 5 mm, not 3. This is the other half of the smoothness fix and it is the half
# the symptom actually pointed at: "the arm moves a certain distance and then it
# does some fine motions" IS the tail of a 39 mm step being chased down to 3 mm.
# The last few millimetres are the expensive part — the arm has to be led far
# enough past the target to break loose at all, so every millimetre down there
# costs a correction — and swept across the plausible plants, tightening 5 mm to
# 3 mm roughly triples the corrections per step and quadruples the time the arm
# spends stopped mid-move, to buy about 1.5 mm.
#
# It is not accuracy that is being given up, because the loop is closed: the next
# observation is of where the arm actually IS, so a 5 mm residual is not an error
# the policy is blind to, it is a slightly shorter step. Measured, 95% of each
# commanded step is still executed. 5 mm is also well inside cp2's own action
# scale (action_std 16-21 mm), so it is small compared with the resolution at
# which the policy is steering in the first place.
STEP_CONVERGE_TOL_M = 0.005

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
TRAVEL_LEAD_BETA = 0.5       # EMA weight on each converged move's lead
TRAVEL_LEAD_MIN_M = 0.005    # moves shorter than this do not inform the estimate
TRAVEL_LEAD_MAX_M = 0.05     # cap, same spirit as MAX_DROOP_COMP_M

MAX_POLICY_STEPS = 50   # dagger/evaluator.py EvalParams.max_steps

# How long each step-mode iteration idles pumping the 3D window's events before
# recomputing perception. One iteration of perception + policy is ~80-100 ms, so
# without this the viewer is pumped at ~10 Hz and a trackpad drag is sampled far
# too coarsely to orbit smoothly. Any keypress breaks the pump immediately, so
# this never delays SPACE. Only used in --step-mode: in continuous mode the loop
# rate is the control rate and must not be traded away for a debug view.
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

      `s`  a lead MAGNITUDE applied along the direction of the current move,
           used by the creep path. `initial_lead` applies it; `observe_travel`
           re-estimates it from the lead a converged move actually needed.

    They are deliberately not mixed. Under creep the standing offset is
    re-derived within every move by the creep itself, which is strictly more
    robust than carrying a vector across moves that go in different directions —
    see the TRAVEL_LEAD_* block. Disabled, all four are no-ops: the creep still
    converges, it just starts each move from a zero lead and pays a nudge or two
    for it.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.d = np.zeros(3, dtype=np.float64)
        self.samples = 0
        self.s = 0.0            # travel-direction lead, metres
        self.s_samples = 0

    def compensate(self, T_target: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return T_target
        T = T_target.copy()
        T[:3, 3] = T[:3, 3] + self.d
        return T

    def initial_lead(self, travel: np.ndarray) -> np.ndarray:
        """The lead to command before the arm has moved, for a move covering
        `travel` (target minus current position)."""
        if not self.enabled or self.s <= 0.0:
            return np.zeros(3, dtype=np.float64)
        n = float(np.linalg.norm(travel))
        if n < 1e-9:
            return np.zeros(3, dtype=np.float64)
        return self.s * (np.asarray(travel, dtype=np.float64) / n)

    def observe_travel(self, lead_used: np.ndarray, travel: np.ndarray) -> None:
        """Learn from a move that converged, by projecting the lead it ended up
        needing onto the direction it travelled.

        Only the along-travel component is kept. The perpendicular part is real —
        gravity does not care which way the arm is going — but it is not
        separable from the direction-dependent part in a single move, and
        carrying a mis-attributed vector forward is the failure this whole scheme
        exists to avoid. The creep re-derives it each move for the cost of a
        nudge.
        """
        if not self.enabled:
            return
        n = float(np.linalg.norm(travel))
        if n < TRAVEL_LEAD_MIN_M:
            return              # a short move says nothing about travel lead
        along = float(np.asarray(lead_used, dtype=np.float64) @ (travel / n))
        if along <= 0.0:
            # The move finished needing a lead pointing BACKWARDS along the way
            # it went, which happens when it overshot and the corrections walked
            # the lead back past zero. That is evidence about that move, not
            # evidence that no lead is needed, and folding it in as a zero halves
            # a perfectly good estimate: measured, one such move dropped a
            # converged 22 mm lead to 11 mm and cost seven commands to rebuild.
            return
        self.s = ((1.0 - TRAVEL_LEAD_BETA) * self.s
                  + TRAVEL_LEAD_BETA * min(along, TRAVEL_LEAD_MAX_M))
        self.s_samples += 1

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
        if self.s_samples:
            return f"lead={self.s*1000:.1f} mm along travel  n={self.s_samples}"
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
        lead = droop.initial_lead(travel) if droop is not None else np.zeros(3)

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
            lead = lead + step
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
            droop.observe_travel(lead, travel)
        else:
            droop.update(T_command, pose_msg_to_matrix(current_msg))

    return SettleResult(settled, dt, drot, seq, nudges)


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

    Returns (next_seq, pos_err, rot_err, passes_used).
    """
    dp = dr = float("inf")
    used = 0
    for i in range(max_passes):
        res = settle(pub, current_msg, T_target_ctrl, seq, timeout_s, droop,
                     tol_m=tol_m if creep else SETTLE_POS_TOL_M, creep=creep)
        dp, dr, seq = res.pos_err, res.rot_err, res.next_seq
        used = i + 1
        if dp < tol_m:
            break
        if label and i + 1 < max_passes:
            print(f"{label} pass {i+1}: {dp*1000:.1f} mm out, "
                  f"{droop.describe() if droop else ''}", flush=True)
    return seq, dp, dr, used


def go_home(pub, T_ctrl_hand: np.ndarray, T_hand_ctrl: np.ndarray,
            seq: int, droop: "DroopCompensator | None" = None,
            creep: bool = True) -> int:
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

        # Intermediate waypoints get a loose tolerance on purpose: they are a
        # path, not a destination, and converging each one to millimetres would
        # add a creep and a full hold per waypoint for no benefit. The refine
        # below is where home is actually reached.
        res = settle(pub, current_msg, clamp_target_pose(T_way @ T_hand_ctrl),
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
    if creep or (droop is not None and droop.enabled):
        seq, _, _, _ = move_to(pub, clamp_target_pose(T_BASE_HAND_HOME @ T_hand_ctrl),
                               seq, HOME_SETTLE_TIMEOUT_S, droop,
                               HOME_REFINE_PASSES, HOME_REFINE_TOL_M,
                               "[home] refine", creep=creep)

    T_end = pose_msg_to_matrix(current_msg) @ T_ctrl_hand
    dp, dr = pose_error(T_end, T_BASE_HAND_HOME)
    print(f"[home] done — {dp*1000:.1f} mm / {np.rad2deg(dr):.1f} deg from home"
          + (f"  [{droop.describe()}]" if droop is not None else ""))
    return seq


# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a Phase-4 BC policy (checkpoint/cp2 or cp3) on the FR3.")
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
    p.add_argument("--no-droop-compensation", action="store_true",
                   help="command equilibrium poses raw. The impedance controller "
                        "settles ~17 mm short of any target, so each step then "
                        "executes as (delta - droop) rather than delta.")
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
    p.add_argument("--arm-lateral", type=float, default=0.12,
                   help="how far SIDEWAYS of that axis a point may sit and "
                        "still be object. Without this an inclined forearm "
                        "scores near zero on the axis and survives; with it, "
                        "the kept region is a capsule around the hand. Larger "
                        "keeps a long object held across the view, smaller cuts "
                        "more arm.")
    p.add_argument("--wrist-seg-px", type=int, default=None,
                   help="hand-segmentation input size for the wrist camera "
                        "(default 256, what cp1 was trained at).")
    p.add_argument("--fixed-seg-px", type=int, default=None,
                   help="hand-segmentation input size for the fixed camera(s) "
                        "(default 384). At ~1 m the hand is a few dozen pixels "
                        "at 256 and its mask falls through min_hand_points. On "
                        "this GPU 384 costs ~10 ms per pass and 512 ~25 ms.")
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

    print(f"Policy dir     : {policy_dir}")
    print(f"Checkpoint     : {ckpt}")
    print(f"Hand seg ckpt  : {hand_seg_ckpt}")
    print(f"EE offset z    : {args.ee_offset_z:.4f} m (panda_hand -> published frame)")
    print(f"Gripper        : {'ENABLED' if args.enable_gripper else 'disabled'}")
    print(f"Droop comp     : {'off' if args.no_droop_compensation else 'on'}")
    print(f"Step motion    : {'multi-pass (stops between passes)' if args.no_creep else 'creep (one stop per step)'}")
    print(f"Cameras        : {', '.join(camera_names)}"
          + (f"  (calib session {args.calib_session})" if fixed_names else ""))

    droop = DroopCompensator(enabled=not args.no_droop_compensation)

    hand_seg_model = load_hand_segmenter(device, hand_seg_ckpt)
    policy, run_cfg = load_bc_policy(policy_dir, ckpt, device)
    _assert_state_layout(run_cfg)
    warm_up_policy(policy, device)

    # panda_hand <-> the frame the controller publishes.
    T_hand_ctrl = z_offset_transform(args.ee_offset_z)
    T_ctrl_hand = invert_transform(T_hand_ctrl)

    client = None
    sub = pub = gripper_sub = None
    rigs: list = []          # referenced by the finally block before it is filled
    viewer = None
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
            p = rig.params
            print(f"[camera] {rig.name:8s} serial={rig.serial}  {rig.kind}  "
                  f"seg={p.seg_input_px}px  "
                  + (f"cluster={p.cluster_voxel_m * 1e3:.0f}mm" if p.cluster_voxel_m
                     else "cluster=off")
                  + f"  margin={p.hand_margin_px}px"
                  + f"  r_obj={p.object_max_radius_m:.2f}m")

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
                publish_seq = go_home(pub, T_ctrl_hand, T_hand_ctrl, publish_seq,
                                      droop, creep=not args.no_creep)
                if args.home_only:
                    return

        perception = MultiCameraPerception(
            rigs,
            HandSegmenter(hand_seg_model, device),
            per_camera_cap=args.per_camera_cap,
            arm_rejection=not args.no_arm_rejection,
            arm_offset_m=args.arm_offset,
            arm_lateral_m=args.arm_lateral,
        )
        print(f"[perception] arm rejection: "
              + (f"on, capsule {args.arm_offset:.3f} m behind the hand / "
                 f"{args.arm_lateral:.3f} m sideways"
                 if not args.no_arm_rejection else "off"))

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
                  "window to execute one step, 'h' to re-home, 'q' to quit.")
        else:
            print(f"CONTINUOUS MODE ({args.max_steps} policy steps max). Steps "
                  "execute as soon as they are predicted. 'h' to re-home, "
                  "'q' or Esc to quit.")

        step = 0
        stop_reason = "max steps reached"

        while step < args.max_steps:
            if args.dry_run:
                T_base_hand = np.eye(4)
            else:
                if current_msg is None:
                    time.sleep(0.01)
                    continue
                T_base_hand = pose_msg_to_matrix(copy.deepcopy(current_msg)) @ T_ctrl_hand

            # ---- segment, deproject and fuse every camera into panda_hand ----
            # The pose is read BEFORE the frames are grabbed and used to place
            # the fixed camera's points, so a stale pose shifts that camera's
            # cloud bodily. The wrist camera is immune — its extrinsics are
            # constant — which is the practical reason it stays the anchor view.
            fused = perception.observe(T_base_hand)
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
                rs = build_robot_state(T_base_hand, read_gripper_norm(assume_open=True))

                action = policy_act(policy, pc, rs, device)   # [7], ch6 in {0, 1}
                grasp_close = bool(action[6] < 0.5)

                if not grasp_close:
                    delta6, clamped = clamp_action_delta(action[:6])
                    T_base_hand_target = T_base_hand @ unpack_action(delta6)
                    T_base_ctrl_target = clamp_target_pose(T_base_hand_target @ T_hand_ctrl)

            # ---- display ----
            # One overlay window per camera, so a camera that has stopped
            # contributing (occluded, mis-segmented, or unplugged) is visible
            # rather than being silently averaged into the union.
            overlay = None
            for rig in rigs:
                color_bgr, hand_mask = perception.last_frames[rig.name]
                view = overlay_mask(color_bgr, hand_mask)
                d = fused.per_camera[rig.name]
                cv2.putText(view, f"{rig.name}  obj={d['object']} hand={d['hand']}"
                            + (f"  -{d['cluster_dropped']} declust"
                               if d["cluster_dropped"] else "")
                            + (f"  [{d['cluster_fallback']}]"
                               if d["cluster_fallback"] not in (None, "disabled")
                               else "")
                            + (f"  -{d['robot_pts_removed']} robot"
                               if d["robot_pts_removed"] else "")
                            + ("  STALE" if d["used_last_hand"] or d["used_last_object"]
                               else ""),
                            (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1, cv2.LINE_AA)
                if overlay is None:
                    overlay = view          # the wrist view carries the HUD below
                else:
                    cv2.imshow(f"cam: {rig.name}", view)

            lines = [
                f"step {step}/{args.max_steps}  obj={len(object_policy)} "
                f"hand={len(hand_policy)}   {fused.summary()}",
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
            pump_until = time.time() + (VIEWER_PUMP_S if args.step_mode else 0.0)
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
                elif k == "r":
                    viewer.cycle_rotate_mode()
                elif k == "z":
                    viewer.roll(-10.0)
                elif k == "x":
                    viewer.roll(+10.0)

            if "q" in pressed:
                stop_reason = "user quit"
                break
            if key == ord("h"):
                if args.dry_run:
                    print("[home] ignored in --dry-run")
                else:
                    publish_seq = go_home(pub, T_ctrl_hand, T_hand_ctrl,
                                          publish_seq, droop,
                                          creep=not args.no_creep)
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
            # fused.summary() carries the per-camera split and the declustered
            # count. Without it a clean object cloud and one that fell back to
            # the sphere every frame print identically.
            print(f"[{step:02d}] target xyz=({pos[0]:+.3f}, {pos[1]:+.3f}, "
                  f"{pos[2]:+.3f})  |d|={np.linalg.norm(delta6[:3]):.4f}m "
                  f"obj={len(object_policy):4d} hand={len(hand_policy):4d}"
                  f"  {fused.summary()}"
                  f"{'  CLAMPED' if clamped else ''}", flush=True)

            if not args.dry_run:
                publish_seq, dpos, drot, passes = move_to(
                    pub, T_base_ctrl_target, publish_seq, args.settle_timeout,
                    droop, STEP_CONVERGE_PASSES, STEP_CONVERGE_TOL_M,
                    creep=not args.no_creep)
                if dpos >= STEP_CONVERGE_TOL_M:
                    print(f"     step short by {dpos*1000:.1f}mm "
                          f"{np.rad2deg(drot):.1f}deg after {passes} passes",
                          flush=True)
            step += 1

        print(f"Episode ended after {step} policy steps: {stop_reason}")

    finally:
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
