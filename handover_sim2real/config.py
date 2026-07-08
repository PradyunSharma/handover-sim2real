# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import easysim

from handover.config import cfg
from yacs.config import CfgNode as CN

_C = cfg

_C_handover_config = _C.clone()

# ---------------------------------------------------------------------------- #
# Policy config
# ---------------------------------------------------------------------------- #
_C.POLICY = CN()


_C.POLICY.TIME_ACTION_REPEAT = 0.15

_C.POLICY.TIME_CLOSE_GRIPPER = 0.5

_C.POLICY.BACK_STEP_SIZE = 0.03

_C.POLICY.POINT_STATE_YCB_RATIO = 0.875

# ---------------------------------------------------------------------------- #
# Experimental: freeze the eye-in-hand point cloud to an early frame and hold it
# for the whole episode, instead of using the live cloud (which degrades to a
# shrinking close-up of the object as the gripper approaches). See
# PointListener._update_acc_points. Default off -> original (live) behavior.
#
# The cloud captured at policy step FREEZE_PARTIAL_POINTCLOUD_AT_STEP (default 0 =
# the very first step, gripper far / object fully in view) is frozen and
# re-projected into the current gripper frame every later step. No trigger
# condition. Points are stored in world frame, so for a STATIC hand+object the
# frozen cloud stays geometrically correct as the gripper moves; for a moving
# target it would go stale.
_C.POLICY.FREEZE_PARTIAL_POINTCLOUD = False

_C.POLICY.FREEZE_PARTIAL_POINTCLOUD_AT_STEP = 0


def get_cfg(handover_config_only=False):
    if not handover_config_only:
        cfg = _C
    else:
        cfg = _C_handover_config
    return cfg.clone()
