"""
Collect the Phase-5 offline BC dataset: K demonstrations per scene, one per
candidate grasp, using OMG Planner as expert teacher.

This is `collect_bc_dataset.py` with the unit of work changed from a scene to a
`(scene, grasp_idx)` pair. Phase 5 conditions the policy on which grasp it is
being asked to reach, so every scene needs a demonstration for each of its
candidate grasps (`build_direction_table.py`). Nothing else about the
observation, the action or the file schema changes — the extra episode attrs are
additive, so `filter_demos.py` and the visualizers still work unmodified.

Pinning is MANDATORY here, unlike Phase 4. Without a successful pin the episode's
target is whatever OMG's argmin picked, which is not the slot we asked for, and an
episode whose `grasp_idx` attr lies about its own target poisons the conditioning.
Such episodes are skipped and counted.

For each `(scene, grasp)` the OMG Planner generates a joint-space trajectory.
We step through that trajectory and record at each policy step:

  point_clouds  [T, N, C]  accumulated scene point cloud in EE frame
                            N=1024 pts, C=5 (xyz + ycb_flag + hand_flag)
  robot_states  [T, R=32]  joint_pos(9) + joint_vel(9) + EE_pose_wxyz(7)
                            + gripper_norm(1) + prev_action(6)
  expert_actions [T, A=7]  delta_EE(6: Δpos+Δeuler) + gripper_cmd(1 binary)
                            gripper_cmd=1 during approach, =0 at pre-grasp close

Dataset: HDF5, one group per episode  episode_NNNNN/
  ├── point_clouds   float32 [T, N, C]
  ├── robot_states   float32 [T, 32]
  └── expert_actions float32 [T, 7]
  attrs: scene_idx, num_steps, grasp_idx, grasp_pose_world

`grasp_pose_world` is the 4x4 pose the episode actually aimed at, read back from
the planner after pinning rather than copied from the table. Storing the pose and
not just the slot means the training dataset carries its own conditioning target:
`regrasp_bc/dataset.py` never has to look a table up, so rebuilding a pin table can no
longer silently retarget a dataset collected against the old one.

Usage:
    python examples/collect_regrasp_demos.py \\
        --cfg-file examples/pretrain_multicam_wr.yaml  \\
        --output   output/bc_dataset/train_p5_k8.h5 \\
        --valid-grasp-dict examples/valid_grasp_dict_005.pkl \\
        --grasp-pin-table output/grasp_cand_table_train_p5.json \\
        [--split train]  [--num-episodes 200]  [--seed 0]  [--grasps-per-scene 8]

The pretrain.yaml already sets YCB_MANO_START_FRAME=last (hand is stationary)
which is the correct setting for static handover.
"""

import argparse
import gym
import h5py
import numpy as np
import os
import pybullet

import handover          # registers HandoverHandCameraPointStateEnv-v1 etc.
import handover_sim2real # registers HandoverSim2RealTrainEnv-v1

from handover.benchmark_wrapper import HandoverBenchmarkWrapper, EpisodeStatus
from handover_sim2real.config import get_cfg
from handover_sim2real.regrasp import anchor as _rg_anchor
from handover_sim2real.regrasp import channels as _rg_channels
from handover_sim2real.regrasp import directions as _rg_directions
from handover_sim2real.policy import PointListener
from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("GADDPG_DIR")

from core.utils import tf_quat, unpack_pose
from experiments.config import cfg_from_file


# ── dimension constants ──────────────────────────────────────────────────────
ROBOT_STATE_DIM = 32   # joint_pos(9) + joint_vel(9) + EE(7) + gripper(1) + prev_act(6)
ACTION_DIM      = 7    # Δpos(3) + Δeuler(3) + gripper_cmd(1)
NUM_PTS         = 1024
# 8 on disk (xyz|ycb|hand|nx|ny|nz), 7 into the model (…|d.n|d.r). The two
# conditioning channels depend on the COMMAND, which is perturbed at training
# time, so only the d-independent half — the normals — can be stored.
PC_CHANNELS     = 8


# ── helpers ──────────────────────────────────────────────────────────────────

def _ee_pose_mat(panda_body, link_ind_hand, panda_base_inv_tf):
    """4×4 EE pose in panda-base frame."""
    pos = panda_body.link_state[0, link_ind_hand, 0:3]
    orn = panda_body.link_state[0, link_ind_hand, 3:7]
    pos, orn = pybullet.multiplyTransforms(*panda_base_inv_tf, pos, orn)
    return unpack_pose(np.hstack([pos, tf_quat(orn)]))  # 4×4


def _robot_state(obs, prev_action_6d):
    """Compose 32-D robot state vector."""
    body = obs["panda_body"]
    link = obs["panda_link_ind_hand"]

    joint_pos = np.asarray(body.dof_state[0, :, 0], dtype=np.float32)   # [9]
    joint_vel = np.asarray(body.dof_state[0, :, 1], dtype=np.float32)   # [9]

    ee_pos     = np.asarray(body.link_state[0, link, 0:3], dtype=np.float32)  # [3]
    ee_orn_xyzw = np.asarray(body.link_state[0, link, 3:7], dtype=np.float32) # [4]
    ee_orn_wxyz = tf_quat(ee_orn_xyzw)                                         # [4] wxyz

    # Finger position (joint 7 or 8) normalised to [0,1]: 1=open, 0=closed
    gripper_norm = np.array([joint_pos[7] / 0.04], dtype=np.float32)  # [1]

    return np.concatenate(
        [joint_pos, joint_vel, ee_pos, ee_orn_wxyz, gripper_norm, prev_action_6d]
    ).astype(np.float32)  # [32]


def ee_grasp_pose_error(obs, grasp_pose_world):
    """(pos_err_m, rot_err_rad) between the current EE and a world-frame grasp pose.

    `grasp_pose_world` is a 4x4 matrix (e.g. env.get_omg_goal_grasp_pose(), the
    OMG traj[-1] the gripper closes at). Used by the DAgger collectors to decide
    when the policy has reached a graspable state and a CLOSE label should be
    recorded instead of another OMG approach step.
    """
    body = obs["panda_body"]
    link = obs["panda_link_ind_hand"]
    ee_pos       = np.asarray(body.link_state[0, link, 0:3], dtype=np.float64)
    ee_quat_xyzw = np.asarray(body.link_state[0, link, 3:7], dtype=np.float64)
    ee_R = np.asarray(pybullet.getMatrixFromQuaternion(ee_quat_xyzw)).reshape(3, 3)

    g_pos = np.asarray(grasp_pose_world[:3, 3], dtype=np.float64)
    g_R   = np.asarray(grasp_pose_world[:3, :3], dtype=np.float64)

    pos_err = float(np.linalg.norm(ee_pos - g_pos))
    cos = (np.trace(ee_R.T @ g_R) - 1.0) / 2.0
    rot_err = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    return pos_err, rot_err


def standoff_plane(grasp_pose_world, standoff_pose_world):
    """(point, normal) of the plane through the standoff, normal pointing toward
    the grasp (the approach direction). Both inputs are 4x4 world poses
    (env.get_omg_goal_grasp_pose() / env.get_omg_standoff_pose()). A point is
    'at/past the standoff' (toward the grasp) when (ee_pos - point)·normal >= 0.
    """
    g_pos = np.asarray(grasp_pose_world[:3, 3], dtype=np.float64)
    s_pos = np.asarray(standoff_pose_world[:3, 3], dtype=np.float64)
    axis = g_pos - s_pos
    n = np.linalg.norm(axis)
    normal = axis / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    return s_pos, normal


def ee_at_or_past_standoff(obs, point, normal):
    """True when the current EE is at/past the standoff plane (grasp side)."""
    body = obs["panda_body"]
    link = obs["panda_link_ind_hand"]
    ee_pos = np.asarray(body.link_state[0, link, 0:3], dtype=np.float64)
    return float(np.dot(ee_pos - point, normal)) >= 0.0


def dynamic_replan_horizon(obs, standoff_pose_world, ee_step, reach_tail,
                           min_free, max_horizon):
    """OMG horizon (num_steps) so the planner's free portion (current -> standoff)
    advances ~`ee_step` metres per step — i.e. a *distance*-proportional length
    instead of a fixed count. This keeps the recorded first-step delta at the
    demonstrations' per-step scale regardless of how far the policy is, avoiding
    both the late-step 'big jump' labels (too few steps) and the 'too-small step'
    over-resolution of a fixed horizon.

        free_steps = round( ‖ee - standoff‖ / ee_step )   (>= min_free)
        horizon    = free_steps + reach_tail               (<= max_horizon)

    `reach_tail` (=5) is added back because OMG folds the standoff->grasp reach
    into the last `reach_tail` waypoints; we only steer the free portion. It
    mirrors OMG's cfg.reach_tail_length and is not ours to pick: the reach is
    sampled at standoff_dist/reach_tail_length = 1.6 cm, NOT at `ee_step`, which
    governs the free portion alone.
    """
    body = obs["panda_body"]
    link = obs["panda_link_ind_hand"]
    ee_pos = np.asarray(body.link_state[0, link, 0:3], dtype=np.float64)
    s_pos  = np.asarray(standoff_pose_world[:3, 3], dtype=np.float64)
    dist = float(np.linalg.norm(ee_pos - s_pos))
    free = max(int(round(dist / max(ee_step, 1e-6))), int(min_free))
    return int(min(free + int(reach_tail), int(max_horizon)))


def _point_cloud(obs, point_listener, panda_base_inv_tf):
    """Accumulated point cloud processed by PointListener → [N, C] float32."""
    raw = obs["callback_get_point_states"]()   # list of [N_i, 3] arrays
    raw_T = [ps.T for ps in raw]              # list of [3, N_i]
    ee_mat = _ee_pose_mat(
        obs["panda_body"], obs["panda_link_ind_hand"], panda_base_inv_tf
    )
    state = point_listener.point_states_to_state(raw_T, ee_mat)
    pc_CN = state[0][0]                       # [C, N]
    return pc_CN.T.astype(np.float32)         # [N, C]


# ── episode collection ────────────────────────────────────────────────────────

def collect_episode(env, point_listener, cfg, scene_idx,
                    panda_base_inv_tf, steps_action_repeat, pin_table=None,
                    grasp_idx=0, command_axes="BINS", d_rule=None):
    """
    Run one episode and return a dict of arrays, or None if it could not be run.

    The approach plays the full OMG trajectory so the gripper actually reaches
    the grasp pose. One final gripper-close transition is appended at the end.

    `pin_table` (examples/build_direction_table.py) forces slot `grasp_idx`
    of this scene. OMG otherwise picks `argmin ||traj.start - goal_set[i]||` from
    the home configuration, which need not be what DAgger later picks from a
    drifted one — measured, that disagreement affects ~1 scene in 5, by a median
    of 10.7 cm, and the aggregate then holds two conflicting targets for the same
    scene. Pinning needs a plan first (to build the goal set), then a replan to
    the pinned grasp.

    Unlike Phase 4, a failed pin aborts the episode: the whole point of Phase 5 is
    that `grasp_idx` names the target, and an episode that quietly fell back to
    OMG's own choice would be mislabelled.
    """
    obs = env.reset(idx=scene_idx)
    point_listener.reset()

    expert_plan, _ = env.run_omg_planner(cfg.RL_MAX_STEP, scene_idx)
    if expert_plan is None:
        return None

    pin_ok = True
    if pin_table is not None:
        # A FAILED PIN NO LONGER LOSES THE EPISODE. Phase 5 returned None here
        # because `grasp_idx` NAMED the conditioning target; under Regrasp the
        # command is derived from whatever the planner actually flies to, so the
        # episode is still a correct demonstration — of a different direction.
        # `bin_assigned` vs `bin_realized` records that. This recovers the ~1%
        # irreducible failures caused by the goal set being re-drawn.
        pin_ok = bool(pin_table.apply(env, scene_idx, grasp_idx))
        expert_plan, _ = env.run_omg_planner(cfg.RL_MAX_STEP, scene_idx,
                                             reset_scene=False)
        if expert_plan is None:
            return None

    # The pose the episode is actually flying to, read back from the planner
    # rather than copied from the table — after pinning they agree to within
    # match_tol, and this is the one that is true by construction.
    grasp_pose = np.asarray(env.get_omg_goal_grasp_pose(), dtype=np.float32)

    stop_step  = len(expert_plan)
    prev_act6d = np.zeros(6, dtype=np.float32)

    point_clouds   = []
    robot_states   = []
    expert_actions = []
    done = False
    info = {}

    # The anchor frame, once, from the step-0 cloud — see regrasp/anchor.py. The
    # hand is static under this config, so one read is the whole episode.
    anchor_R = anchor_mode = None
    wrist = _rg_anchor.wrist_world(env)
    mano_side = _rg_anchor.handedness(env)
    _ba = pin_table.bin_of(scene_idx, grasp_idx) if pin_table is not None else None
    _bin_assigned = None if _ba is None else int(_ba)
    # "BINS" rather than a bare default array: None is a MEANINGFUL value for
    # `command_axes` (it selects run 1's grasp-axis rule), so "omitted" and
    # "grasp_axis" must not be the same request.
    if isinstance(command_axes, str):
        command_axes = _rg_directions.BINS
    d_rule = d_rule or _rg_directions.DirectionRule()
    # What the expert actually flies, under the run's `d_rule`. It is NOT the
    # command any more — see the step-0 block below — but it IS what
    # `DATA.d_source: d_grasp_world` trains on. Under `grasp_offset` it needs the
    # object centroid, which only exists once the step-0 cloud has arrived, so
    # this is finalised inside the loop rather than here.
    d_grasp_world = None
    d_world = None
    c_w = None

    for step in range(stop_step):
        pc5   = _point_cloud(obs, point_listener, panda_base_inv_tf)
        pc, _ = _rg_channels.pack_stored_cloud(pc5)
        if step == 0:
            c_ee = _rg_channels.object_centroid(pc5, fallback_to_all=False)
            if c_ee is not None:
                c_w = _rg_anchor.centroid_to_world(
                    c_ee, obs, panda_base_inv_tf,
                    cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)
                anchor_R, _am = _rg_anchor.anchor_rotation(
                    c_w, wrist, np.asarray(cfg.ENV.PANDA_BASE_POSITION),
                    _rg_anchor.AnchorState())
                anchor_mode = _am["mode"]
                # ---- THE COMMAND, WHICHEVER RULE `--command` NAMES -----------
                # It has to be computed HERE and not above, because the anchor
                # frame only exists once the step-0 cloud has given us the object
                # centroid. The default is the bin axis: `retry.next_direction`
                # has no grasp to read at deployment and issues
                # `to_world(BINS[b], anchor_R)`, so labelling with `-R_grasp[:,2]`
                # instead is a MEASURED median-18-degree train/deploy skew, the
                # same order as the effect the conditioning is supposed to have.
                #
                # THE DAGGER COLLECTOR MAKES THE SAME CALL, and that is the trap
                # this comment exists to flag: base demonstrations come from THIS
                # file, DAgger shards from handover_sim2real/regrasp/collector.py.
                # Changing one and not the other produces an aggregate whose
                # halves are captioned under different rules — invisible in every
                # rate, and caught only by `audit_regrasp_demos.py`'s
                # `command vs bin axis` line. Both now go through
                # `directions.command_direction`, so there is one definition
                # rather than two copies.
                #
                # NOTE the base shard's `d_world` is NOT what a run 9-style
                # config trains on. This file replays an OMG plan, so nothing
                # here is conditioned on anything — `d_world` is a caption, and
                # `DATA.d_source: d_grasp_world` reads the other attr instead.
                # That is why run 9 needs no re-collection.
                d_grasp_world = d_rule.of(grasp_pose, c_w)
                _d = _rg_directions.command_direction(
                    _bin_assigned, anchor_R, grasp_pose=grasp_pose,
                    axes=command_axes)
                # Falls back to the rule's own direction when the command cannot
                # be formed, so an episode always has SOME caption or none at
                # all — never a zero vector standing in for one.
                d_world = _d if _d is not None else d_grasp_world
        rs    = _robot_state(obs, prev_act6d)
        delta = env.convert_target_joint_position_to_action(expert_plan[step])  # [6]
        act   = np.concatenate([delta, [1.0]]).astype(np.float32)  # gripper open

        point_clouds.append(pc)
        robot_states.append(rs)
        expert_actions.append(act)

        prev_act6d = delta.copy()

        for _ in range(steps_action_repeat):
            obs, _, done, info = env.step(expert_plan[step])
            if done:
                break
        if done:
            break

    if len(expert_actions) == 0:
        return None

    # Append gripper-close transition at the pre-grasp pose
    if not done:
        pc, _ = _rg_channels.pack_stored_cloud(
            _point_cloud(obs, point_listener, panda_base_inv_tf))
        rs = _robot_state(obs, prev_act6d)
        close_act = np.concatenate(
            [np.zeros(6, dtype=np.float32), [0.0]]
        ).astype(np.float32)

        point_clouds.append(pc)
        robot_states.append(rs)
        expert_actions.append(close_act)

    return {
        "point_clouds":   np.array(point_clouds,   dtype=np.float32),  # [T, N, C]
        "robot_states":   np.array(robot_states,   dtype=np.float32),  # [T, 32]
        "expert_actions": np.array(expert_actions, dtype=np.float32),  # [T, 7]
        "scene_idx":      scene_idx,
        "grasp_idx":      int(grasp_idx),
        "grasp_pose_world": grasp_pose,                                # [4, 4]
        # ---- Regrasp conditioning + provenance ----
        "d_world":        (np.zeros(3, dtype=np.float32) if d_world is None
                           else np.asarray(d_world, dtype=np.float32)),
        # ...and what the expert actually flew, under the run's `d_rule`, which
        # is a different vector from the command: the demonstration sits a
        # median ~18 deg from its bin's axis (max 45, the bin half-width). That
        # gap is the label noise the conditioning has to tolerate, recorded so it
        # can be measured rather than assumed — and it is what
        # `DATA.d_source: d_grasp_world` trains on.
        "d_grasp_world":  (np.zeros(3, dtype=np.float32) if d_grasp_world is None
                           else np.asarray(d_grasp_world, dtype=np.float32)),
        "demo_off_deg":   (float("nan")
                           if (d_world is None or d_grasp_world is None)
                           else float(_rg_directions.angle_between(
                               d_world, d_grasp_world))),
        # `x or -1` would map bin 0 (+x, the MOST common bin) to -1, because 0
        # is falsy. Explicit None check, always.
        "bin_assigned":   _bin_assigned if _bin_assigned is not None else -1,
        # The bin the DEMONSTRATION lands in — derived from the grasp, not from
        # the command, so it still disagrees with `bin_assigned` exactly when the
        # pin missed. Deriving it from `d_world` would make it trivially equal.
        "bin_realized":   (-1 if anchor_R is None or d_grasp_world is None else
                           int(_rg_directions.bin_of(
                               _rg_directions.from_world(d_grasp_world,
                                                         anchor_R)))),
        "anchor_R":       (np.eye(3, dtype=np.float32) if anchor_R is None
                           else np.asarray(anchor_R, dtype=np.float32)),
        # The anchor's ORIGIN — see the DAgger collector. With it, switching
        # `d_rule` is a relabelling pass rather than a re-collection.
        "centroid_world": (np.full(3, np.nan, dtype=np.float32) if c_w is None
                           else np.asarray(c_w, dtype=np.float32)),
        "anchor_mode":    str(anchor_mode or "unset"),
        "wrist_world":    (np.full(3, np.nan, dtype=np.float32) if wrist is None
                           else np.asarray(wrist, dtype=np.float32)),
        "mano_side":      str(mano_side or "unknown"),
        "pin_ok":         int(pin_ok),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Collect offline BC dataset from OMG Planner.")
    p.add_argument("--cfg-file",       required=True, help="path to config yaml (e.g. pretrain.yaml)")
    p.add_argument("--output",         required=True, help="output HDF5 path")
    p.add_argument("--split",          default="train", choices=["train", "val", "test"])
    p.add_argument("--num-scenes",     type=int, default=None, help="cap on scenes (quick runs)")
    p.add_argument("--num-episodes",   type=int, default=None,
                   help="cap on (scene, grasp) pairs, applied after --num-scenes")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--valid-grasp-dict", default=None,
                   help="paper's offline hand-collision-filtered grasp dict "
                        "(examples/valid_grasp_dict_005.pkl). MUST match what the "
                        "DAgger/RL phases use, or they aim at different grasps.")
    p.add_argument("--grasp-pin-table", required=True,
                   help="per-scene candidate grasps "
                        "(examples/build_direction_table.py). Required: it "
                        "is what defines the slots this script iterates over.")
    p.add_argument("--shard", default=None,
                   help="'i/n' — collect only scenes with scene_idx %% n == i, "
                        "writing an independent HDF5. Base collection is SERIAL "
                        "(~8 s/episode) and is the long pole of a run: at three "
                        "demonstrations per bin it is ~10 h on one process. Four "
                        "shards run concurrently cut that to ~2.6 h, and the "
                        "trainer takes a LIST for TRAIN.base_train_h5, so the "
                        "pieces never need merging. Round-robin over scenes "
                        "rather than a contiguous block because scenes carry "
                        "different slot counts, and interleaving balances them "
                        "without measuring anything. Each shard holds WHOLE "
                        "scenes, so no episode is split across files.")
    p.add_argument("--grasps-per-scene", type=int, default=None,
                   help="cap on slots per scene (default: every slot the table "
                        "holds, i.e. K from build_direction_table.py)")
    p.add_argument("--d-rule", default=None,
                   choices=["approach_axis", "grasp_offset"],
                   help="what `d` is derived from. DEFAULT: read from the pin "
                        "table's `_meta`, which is what you want — the table's "
                        "bins were populated under one rule and measuring the "
                        "realised bin by another turns every episode into a "
                        "miscaption. Pass it only to override a table with no "
                        "recorded rule.")
    p.add_argument("--d-point-depth", type=float, default=None,
                   help="grasp_offset only; default: the table's, else 0.1122")
    p.add_argument("--command", default="bin_axis",
                   choices=["bin_axis", "bin_centroid", "grasp_axis"],
                   help="which rule writes the `d_world` CAPTION on each episode. "
                        "MUST match SIM.command_deploy in the run config that "
                        "trains on this shard, or the aggregate's halves are "
                        "captioned under different rules. bin_axis (default) is "
                        "runs 2-8; grasp_axis is run 1. Note this is only a "
                        "caption here — this script replays an OMG plan, so "
                        "nothing is conditioned on it, and a run whose learner "
                        "sets DATA.d_source: d_grasp_world ignores it entirely.")
    p.add_argument("--freeze-partial-pointcloud", action="store_true",
                   help="experimental: freeze the cloud to an early frame and hold "
                        "it for the whole episode, instead of the live cloud that "
                        "shrinks to a close-up as the gripper approaches")
    p.add_argument("--freeze-at-step", type=int, default=None,
                   help="which policy step's cloud to freeze and hold "
                        "(default: config value, 0 = the very first step)")
    p.add_argument("--egl", action="store_true",
                   help="render the offscreen hand camera with the EGL GPU renderer "
                        "(NVIDIA dGPU here) instead of the DIRECT-mode CPU software "
                        "renderer. The point cloud is renderer-dependent, so rollout "
                        "MUST later use the same renderer this dataset was built with.")
    return p.parse_args()


def main():
    args = parse_args()
    # This script reaches gym.make() without importing regrasp.env_setup,
    # so the OMG_PLANNER_DIR guard has to be asked for explicitly. Without
    # it a bad path surfaces as `ModuleNotFoundError: No module named 'omg'`
    # from inside the gym registry, which names a dependency when the real
    # problem is a path.
    from handover_sim2real.regrasp.env_setup import preflight
    preflight()

    cfg = get_cfg()
    cfg_from_file(filename=args.cfg_file, dict=cfg, merge_to_cn_dict=True)
    cfg.BENCHMARK.SPLIT = args.split
    if args.freeze_partial_pointcloud:
        cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD = True
    if args.freeze_at_step is not None:
        cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD_AT_STEP = args.freeze_at_step
    if args.egl:
        cfg.SIM.BULLET.USE_EGL = True   # GPU (NVIDIA) offscreen camera, headless
    np.random.seed(args.seed)

    # The paper's offline hand-collision grasp filter must be on cfg.omg_config
    # BEFORE the env is built: OMGPlanner.__init__ copies omg_config onto the
    # global omg_cfg, so setting it afterwards is a no-op. Pass the SAME dict the
    # DAgger/RL phases use or the two aim at different grasp sets (measured: ~1
    # scene in 5 disagrees, median 10.7 cm apart).
    # DEFAULT IT FROM THE PIN TABLE, which records the dict it was built with.
    # Omitting the flag used to mean "no filter", so the goal set at collection
    # was a DIFFERENT set from the one the table's poses came from and the pins
    # could not match: `[grasp-pin] pinned pose not in the current goal set`,
    # then OMG's own pick, then — under bin-axis conditioning — an episode
    # captioned with a direction it did not fly. The table knows the answer, the
    # env has to be built before the table is normally loaded, and a bare
    # `json.load` costs nothing, so read it here rather than requiring a human to
    # keep two command lines in sync.
    if not args.valid_grasp_dict and args.grasp_pin_table:
        try:
            import json as _json
            with open(args.grasp_pin_table) as _f:
                _meta = (_json.load(_f).get("_meta") or {})
            _from_table = _meta.get("valid_grasp_dict_path")
        except Exception as _e:                                # noqa: BLE001
            _from_table = None
            print(f"[valid_grasp_dict] could not read {args.grasp_pin_table}: {_e}")
        if _from_table:
            args.valid_grasp_dict = _from_table
            print(f"[valid_grasp_dict] not given — taking the table's own: "
                  f"{_from_table}")

    if args.valid_grasp_dict:
        from handover_sim2real.utils import resolve_valid_grasp_dict_path
        _vgd = resolve_valid_grasp_dict_path(
            {"valid_grasp_dict_path": args.valid_grasp_dict}, cfg.BENCHMARK.SETUP)
        if _vgd is not None:
            cfg.omg_config["valid_grasp_dict_path"] = _vgd
            print(f"[valid_grasp_dict] paper hand-collision filter ON: {_vgd}")

    env            = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    point_listener = PointListener(cfg, seed=args.seed)

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION
    )
    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

    num_scenes = env.num_scenes
    if args.num_scenes is not None:
        num_scenes = min(num_scenes, args.num_scenes)

    print(f"Collecting from {num_scenes} scenes  split={args.split}  seed={args.seed}")
    print(f"Output: {args.output}")

    from handover_sim2real.regrasp import load_grasp_pin_table
    pin_table = load_grasp_pin_table(
        args.grasp_pin_table,
        sim_cfg_block={"valid_grasp_dict_path": args.valid_grasp_dict,
                       "hand_collision_filter": False,
                       "split": args.split})   # scene indices are split-relative

    # The caption rule, resolved once. No `keep_only` has run here — this script
    # is what PRODUCES the shard the demo filter is later derived from — so a
    # `bin_centroid` caption is over the full assignment rather than the pruned
    # one. That is a ~1% difference and it is recorded in the shard's attrs
    # below, but it is the reason `bin_centroid` belongs on the DEPLOYMENT side
    # rather than here.
    from handover_sim2real.regrasp.setup import resolve_command_axes
    command_axes = resolve_command_axes(pin_table, args.command)

    # THE `d` RULE COMES FROM THE TABLE unless overridden. The table's bins were
    # populated under one rule; deriving `bin_realized` under another would
    # disagree with `bin_assigned` on most episodes and the demo audit would
    # discard the collection. Reading it from the file is what makes that
    # impossible to get wrong from the command line.
    _m = getattr(pin_table, "meta", {}) or {}
    d_rule = _rg_directions.DirectionRule(
        rule=str(args.d_rule or _m.get("d_rule", "approach_axis")),
        depth=float(args.d_point_depth if args.d_point_depth is not None
                    else _m.get("d_point_depth", _rg_directions.FINGERTIP_DEPTH)),
        min_offset=float(_m.get("d_min_offset", 0.0)))
    print(f"d rule: {d_rule.describe()}"
          + ("  (from the pin table's _meta)" if args.d_rule is None else
             "  (CLI OVERRIDE)"))

    # The work list: every (scene, slot) the table offers, scene-major, so a
    # partial run still covers whole scenes and `--num-episodes` stays readable.
    shard_i = shard_n = None
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in str(args.shard).split("/"))
        except ValueError:
            raise SystemExit(f"--shard wants 'i/n', got {args.shard!r}")
        if not 0 <= shard_i < shard_n:
            raise SystemExit(f"--shard needs 0 <= i < n, got {shard_i}/{shard_n}")

    jobs = []
    for scene_idx in range(num_scenes):
        if shard_n is not None and scene_idx % shard_n != shard_i:
            continue
        n_slots = pin_table.num_grasps_for(scene_idx)
        if args.grasps_per_scene is not None:
            n_slots = min(n_slots, int(args.grasps_per_scene))
        jobs.extend((scene_idx, g) for g in range(n_slots))
    if args.num_episodes is not None:
        jobs = jobs[:int(args.num_episodes)]
    n_sc = len({s for s, _ in jobs})
    print(f"  {len(jobs)} (scene, grasp) pairs over {n_sc} scenes"
          + (f"   [shard {shard_i}/{shard_n}]" if shard_n else ""))

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    ep_idx      = 0
    omg_skipped = 0

    with h5py.File(args.output, "w") as f:
        # File-level metadata
        f.attrs["split"]           = args.split
        f.attrs["seed"]            = args.seed
        f.attrs["num_pts"]         = NUM_PTS
        f.attrs["pc_channels"]     = PC_CHANNELS
        f.attrs["robot_state_dim"] = ROBOT_STATE_DIM
        f.attrs["action_dim"]      = ACTION_DIM
        f.attrs["action_format"]   = "delta_pos(3)+delta_euler(3)+gripper_cmd(1)"
        f.attrs["robot_state_fmt"] = "joint_pos(9)+joint_vel(9)+ee_xyz(3)+ee_wxyz(4)+gripper_norm(1)+prev_act(6)"
        f.attrs["pc_format"]       = ("xyz(3)+ycb(1)+hand(1)+normal(3) in EE "
                                      "frame; d.n and d.r are built at load")
        f.attrs["schema"]          = "regrasp-v1"
        f.attrs["freeze_partial_pointcloud"]  = cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD
        f.attrs["freeze_at_step"]             = cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD_AT_STEP
        # PROVENANCE — which grasp set these labels aim at. Without it a dataset
        # is indistinguishable from one collected against a different goal set,
        # and comparing it to a pin table silently measures the mismatch between
        # two grasp choices instead of whatever you meant to measure.
        f.attrs["valid_grasp_dict"] = str(args.valid_grasp_dict or "")
        f.attrs["grasp_pin_table"]  = str(args.grasp_pin_table or "")
        # PROVENANCE — which OBSERVATION produced this cloud. Until this existed,
        # a wrist-only and a wrist+left+right collection were indistinguishable
        # from their attrs: both carry pc_channels 5, num_pts 1024 and the same
        # pc_format, so the FILENAME was the only evidence. Aggregating two
        # camera configs is the one inconsistency DAgger cannot average away, and
        # it fails silently — training looks fine and the policy is wrong only at
        # deployment.
        #
        # The resolved CAMERA LIST, not just the cfg path: a path stays true
        # while the file it names is edited, so the name alone can drift away
        # from what was actually collected. The renderer is recorded for the same
        # reason — EGL and the CPU TinyRenderer do not produce identical clouds.
        _hcps = cfg.ENV.HANDOVER_HAND_CAMERA_POINT_STATE_ENV
        f.attrs["sim_cfg_file"]   = str(args.cfg_file or "")
        f.attrs["cameras"]        = ",".join(_hcps.CAMERAS)
        f.attrs["compute_mano"]   = bool(_hcps.COMPUTE_MANO_POINT_STATE)
        f.attrs["compute_robot"]  = bool(_hcps.COMPUTE_ROBOT_POINT_STATE)
        f.attrs["renderer"]       = "egl" if cfg.SIM.BULLET.USE_EGL else "tiny"

        f.attrs["phase"]            = 5
        f.attrs["grasps_per_scene"] = int(pin_table.num_grasps)
        # Which rule wrote `d_world` on every episode in this file. Recorded so a
        # shard cannot be trained on under a config that assumes a different one
        # without the mismatch being answerable after the fact.
        f.attrs["command_rule"] = str(args.command)
        # ...and what `d` itself was derived from. A shard collected under one
        # `d_rule` and trained under a config expecting the other is captioned
        # by a different question entirely; recorded so it is answerable.
        for _k, _v in d_rule.as_meta().items():
            f.attrs[_k] = _v

        for job_i, (scene_idx, grasp_idx) in enumerate(jobs):
            episode = collect_episode(
                env, point_listener, cfg, scene_idx,
                panda_base_inv_tf, steps_action_repeat, pin_table=pin_table,
                grasp_idx=grasp_idx, command_axes=command_axes, d_rule=d_rule,
            )

            if episode is None:
                omg_skipped += 1
                print(f"  [{job_i+1:5d}/{len(jobs)}] scene {scene_idx} g{grasp_idx}: "
                      f"OMG planner failed or pin did not match — skipped")
                continue

            T = len(episode["expert_actions"])
            grp = f.create_group(f"episode_{ep_idx:05d}")
            grp.attrs["scene_idx"] = episode["scene_idx"]
            grp.attrs["num_steps"] = T
            # Phase-5 additions. `grasp_idx` is the slot the sampler asked for;
            # `grasp_pose_world` is where the episode actually flew, so the
            # dataset carries its own conditioning target and never needs the
            # table at training time.
            grp.attrs["grasp_idx"]        = episode["grasp_idx"]
            grp.attrs["grasp_pose_world"] = episode["grasp_pose_world"]
            # Regrasp: `d_world` is the one BCDataset requires; the rest are
            # provenance. `wrist_world` in particular lets the anchor frame be
            # re-derived under a different definition as a relabelling pass.
            for _k in ("d_world", "d_grasp_world", "demo_off_deg",
                       "bin_assigned", "bin_realized", "anchor_R",
                       "anchor_mode", "wrist_world", "mano_side", "pin_ok",
                       "centroid_world"):
                if _k in episode:
                    grp.attrs[_k] = episode[_k]
            grp.create_dataset("point_clouds",   data=episode["point_clouds"],   compression="gzip")
            grp.create_dataset("robot_states",   data=episode["robot_states"],   compression="gzip")
            grp.create_dataset("expert_actions", data=episode["expert_actions"], compression="gzip")
            ep_idx += 1

            if (job_i + 1) % 40 == 0 or job_i == len(jobs) - 1:
                print(f"  [{job_i+1:5d}/{len(jobs)}]  episodes saved: {ep_idx}"
                      f"  steps this ep: {T}", flush=True)

        f.attrs["num_episodes"] = ep_idx

    print(f"\nDone.")
    print(f"  Episodes saved : {ep_idx}/{len(jobs)}")
    print(f"  Skipped        : {omg_skipped}")
    print(f"  Dataset        : {args.output}")


if __name__ == "__main__":
    main()
