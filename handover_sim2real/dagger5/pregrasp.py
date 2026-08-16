"""The open-loop endgame: the blind reach + close of the CVPR2023 policy.

Runs 1-20 asked the learner to produce the whole motion, standoff -> grasp
included, and scored the close where the policy fired it. This module implements
the alternative the original paper uses (`handover_sim2real/policy.py`,
`HandoverSim2RealPolicy.grasp_and_back`): the learner is responsible only for the
APPROACH, it signals "commit" instead of "close", and the last few centimetres
are a fixed feed-forward motion that consumes no perception.

    ... policy steps ...  ->  commit (action[6] < 0.5)
                              -> push `dist` m along the gripper's OWN local +z
                              -> close, hold, score

`grasp_and_back` computes `reach_goal = R @ [0, 0, 0.08] + pos` from the pose at
the commit, interpolates 5 points, drops the first, and plays the remaining 4 by
index at one `steps_action_repeat` each, holding the orientation fixed. This is
the same motion, expressed through the `action_to_target_joint` interface the
rest of Phase 4 drives the arm with, so the reach lands on the arm exactly the
way a policy step does.

WHY 0.064 m AND NOT THE PAPER'S 0.08. OMG's standoff ramp is
`grasp @ translate(0, 0, -standoff_dist * (1 - 1/reach_tail))` — 0.064 m at the
defaults, not 0.08 (see `collector.derived_standoff_pose`). Measured on the 472
kept demonstrations of `train_pinned_omg_ok.h5`, the displacement from the state
the arm is in after the standoff waypoint to the state it closes at is

    along the EE's local +z   mean 0.0642 m   median 0.0652   p10-p90 0.054-0.073
    lateral (off-axis)        mean 0.0088 m   median 0.0077   p90 0.0152, max 0.042
    orientation change        mean 0.027 rad  median 0.023    p90 0.050

so 0.064 is what the expert's own reach travels, and the paper's 0.08 would
overshoot by 1.6 cm. The lateral row is the price of going open loop: the
demonstrated reach is not a pure axial push, and ~9 mm of the correction it makes
is unavailable to a motion that cannot see. That is inside `close_pos_thresh`
(0.02 m) but it is not nothing, and it is the number to weigh a drop in success
against.

FROZEN TARGET, NOT FROZEN JOINT ANGLES. The world-frame goal is computed once, at
the commit, and never revised — no observation, no grasp pose, nothing the real
robot would not have. The per-substep command is then the delta from the CURRENT
EE to the next point on that frozen line, which is what a Cartesian controller is
given on hardware and what the collector's committed reach already does ("the
delta is still recomputed from the CURRENT EE each step; only the target stops
moving"). Playing 4 pre-solved IK configurations by index instead, as
`grasp_and_back` does, would bake this env's tracking lag into the waypoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet
from transforms3d.euler import mat2euler

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from rollout_bc_policy import action_to_target_joint  # noqa: E402


def forward_dist_default(standoff_dist: float, reach_tail: int) -> float:
    """The push that exactly undoes the standoff offset.

    Same expression as `collector.derived_standoff_pose`'s offset, so the two
    cannot drift apart: the standoff is the grasp backed off by this much along
    the grasp's local -z, therefore this much along local +z is the way back.
    """
    return float(standoff_dist) * (1.0 - 1.0 / int(reach_tail))


def ee_pose_world(obs) -> np.ndarray:
    """4x4 world pose of the hand link, the frame every delta action is in."""
    body = obs["panda_body"]
    link = obs["panda_link_ind_hand"]
    pos = np.asarray(body.link_state[0, link, 0:3], dtype=np.float64)
    quat_xyzw = np.asarray(body.link_state[0, link, 3:7], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(
        pybullet.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    T[:3, 3] = pos
    return T


def open_loop_reach(env, obs, steps_action_repeat: int, *,
                    dist: float, num_steps: int = 4):
    """Blind push `dist` m along the gripper's local +z. Returns (obs, done, status).

    The gripper stays OPEN throughout — closing is the caller's next move
    (`grasp_held_after_hold`), exactly as the paper closes only after the reach.
    Aborts early if the benchmark ends the episode mid-push (a lateral swing into
    the pre-grasp can knock the object out of the hand), and reports that through
    `done` / `status` so the caller can file the episode under the benchmark's own
    failure rather than as a bad grasp.
    """
    num_steps = max(int(num_steps), 1)
    ee0 = ee_pose_world(obs)
    goal = ee0.copy()
    goal[:3, 3] = ee0[:3, 3] + ee0[:3, :3] @ np.array([0.0, 0.0, float(dist)])

    done, status = False, 0
    for k in range(1, num_steps + 1):
        way = ee0.copy()
        way[:3, 3] = ee0[:3, 3] + (goal[:3, 3] - ee0[:3, 3]) * (k / num_steps)
        delta = np.linalg.inv(ee_pose_world(obs)) @ way
        action = np.concatenate([
            delta[:3, 3], np.asarray(mat2euler(delta[:3, :3])), [1.0],
        ]).astype(np.float32)
        target_jp = action_to_target_joint(action, obs)
        for _ in range(int(steps_action_repeat)):
            obs, _, done, info = env.step(target_jp)
            if done:
                status = int(info.get("status", 0))
                break
        if done:
            break
    return obs, done, status
