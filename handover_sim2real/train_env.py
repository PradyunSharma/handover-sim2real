# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import numpy as np
import pybullet
import random

from handover.benchmark_wrapper import EpisodeStatus
from handover.handover_env import HandoverHandCameraPointStateEnv
from scipy.spatial.transform import Rotation as Rot

from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("OMG_PLANNER_DIR")

from omg.config import cfg as omg_cfg
from omg.core import PlanningScene
from ycb_render.robotPose import robot_pykdl

add_sys_path_from_env("GADDPG_DIR")

from core.utils import (
    anchor_seeds,
    hand_finger_point,
    inv_lookat,
    inv_relative_pose,
    mat2euler,
    mat2quat,
    pack_pose,
    pack_pose_rot_first,
    ros_quat,
    rotZ,
    se3_inverse,
    tf_quat,
    unpack_pose,
    wrap_value,
)


class OMGPlanner:
    def __init__(self, cfg):
        self._cfg = cfg

        for key, val in self._cfg.items():
            setattr(omg_cfg, key, val)

        omg_cfg.get_global_path()

        # Enforce determinism. This accounts for the call of random.sample() in
        # `Robot.load_collision_points()` in `OMG-Planner/omg/core.py`.
        random.seed(0)

        self._scene = PlanningScene(omg_cfg)

    def reset_scene(self, names, poses):
        for name in list(self._scene.env.names):
            self._scene.env.remove_object(name, lazy=True)
        assert len(self._scene.env.objects) == 0

        for name, pose in zip(names, poses):
            self._scene.env.add_object(name, pose[:3], pose[3:], compute_grasp=False)
        self._scene.env.combine_sdfs()

        self._grasp_computed = False

    def plan_to_target(self, start_conf, target_name, num_steps, scene_idx):
        self._scene.traj.start = start_conf
        self._scene.env.set_target(target_name)

        omg_cfg.timesteps = num_steps
        omg_cfg.get_global_param(steps=omg_cfg.timesteps)

        if not hasattr(self._scene, "planner"):
            self._scene.reset(scene_idx=scene_idx)
        else:
            if self._grasp_computed:
                self._scene.env.objects[0].compute_grasp = False
            self._scene.update_planner(scene_idx=scene_idx)

        if not self._grasp_computed:
            self._grasp_computed = True

        info = self._scene.step()
        traj = self._scene.planner.history_trajectories[-1]

        if len(info) == 0:
            traj = None

        return traj, info

    def get_grasp_poses(self):
        return self._scene.env.objects[self._scene.env.target_idx].grasps_poses


class HandoverSim2RealTrainEnv(HandoverHandCameraPointStateEnv):
    def init(self):
        super().init()

        self._panda_base_invert_transform = pybullet.invertTransform(
            self._cfg.ENV.PANDA_BASE_POSITION, self._cfg.ENV.PANDA_BASE_ORIENTATION
        )
        self._panda_base_pose = (
            self.cfg.ENV.PANDA_BASE_POSITION
            + self.cfg.ENV.PANDA_BASE_ORIENTATION[3:]
            + self.cfg.ENV.PANDA_BASE_ORIENTATION[:3]
        )

        self._panda_kinematics = robot_pykdl.robot_kinematics(None, data_path=self.cfg.ROOT_DIR)

        self._omg_planner = OMGPlanner(self._cfg.omg_config)

        # Human-hand grasp filter (paper §: filter grasps that collide with the
        # hand). Off by default so existing pipelines are unchanged; enable per
        # run via set_hand_collision_filter(). Registered on the OMG scene env so
        # OMG's setup_goal_set prunes hand-colliding grasps before it plans.
        self._hand_collision_filter = False
        self._hand_collision_thresh = 0.08   # m, gripper control point -> hand point
        self._hand_points_radius = 0.35      # m, keep hand links within this of the object
        self._omg_planner._scene.env.external_grasp_filter = self._hand_grasp_collision_mask

    def post_reset(self, env_ids, scene_id):
        self._omg_planner_goal_pose = None
        self._omg_grasp_pose = None

        return super().post_reset(env_ids, scene_id)

    def callback_get_reward_post_status(self, reward, status):
        if status == EpisodeStatus.SUCCESS:
            reward = 1.0
        else:
            reward = 0.0
        return reward

    def _get_ee_pose(self):
        pos = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 0:3]
        orn = self.panda.body.link_state[0, self.panda.LINK_IND_HAND, 3:7]
        ee_pose = np.hstack((pos, tf_quat(orn)))
        return ee_pose

    def randomize_arm_init(self, near, far):
        pos = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 0:3].tolist()
        orn = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 3:7].tolist()
        ycb_pose = pos + orn[3:] + orn[:3]
        panda_base_to_ycb_pose = inv_relative_pose(ycb_pose, self._panda_base_pose)
        panda_base_to_ycb_trans = panda_base_to_ycb_pose[:3, 3]

        outer_loop_num = 20
        inner_loop_num = 5

        for _ in range(outer_loop_num):
            theta = np.random.uniform(low=0, high=np.pi * 2 / 3)
            phi = np.random.uniform(low=np.pi / 2, high=np.pi * 3 / 2)
            r = np.random.uniform(low=near, high=far)
            pos = np.array(
                [
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta),
                ]
            )

            position = (
                panda_base_to_ycb_trans + pos + np.random.uniform(low=-0.03, high=0.03, size=3)
            )
            position[0] = np.clip(position[0], +0.0, +0.5)
            position[1] = np.clip(position[1], -0.3, +0.3)
            position[2] = np.clip(position[2], +0.2, +0.6)

            pos = position - panda_base_to_ycb_trans
            up = np.array([0.0, 0.0, -1.0])

            for _ in range(inner_loop_num):
                R = np.matmul(inv_lookat(pos, 2 * pos, up), rotZ(-np.pi / 2)[:3, :3])
                orientation = ros_quat(mat2quat(R))
                anchor_idx = np.random.randint(len(anchor_seeds))
                q_out = self._panda_kinematics.inverse_kinematics(
                    position, orientation=orientation, seed=anchor_seeds[anchor_idx]
                )
                if q_out is not None:
                    break

        if q_out is not None:
            q_out = q_out.tolist() + [0.04, 0.04]

        return q_out

    def set_initial_joint_position(self, initial_joint_position):
        self.panda.body.initial_dof_position = initial_joint_position

    def get_ee_to_ycb_distance(self):
        ee_pos = self._get_ee_pose()[:3]
        ycb_pos = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 0:3].numpy()
        ee_to_ycb_distance = np.linalg.norm(ee_pos - ycb_pos)
        return ee_to_ycb_distance

    def _joint_to_world_ee_pose(self, joint_position):
        """Packed world-frame panda_hand pose for a joint configuration."""
        panda_base_to_ee = self._panda_kinematics.forward_kinematics_parallel(
            joint_values=wrap_value(joint_position)[None], offset=False
        )[0, 7]
        return pack_pose(np.matmul(unpack_pose(self._panda_base_pose), panda_base_to_ee))

    def set_hand_collision_filter(self, enable=True, thresh=0.08, points_radius=0.35):
        """Enable/configure the human-hand grasp filter for OMG planning. When on,
        OMG's goal set is pruned of grasps whose gripper geometry comes within
        `thresh` (m) of the MANO hand, so the planner reaches a hand-free grasp
        (the paper's collision-checking of grasp candidates against the hand).
        `points_radius` (m) keeps only hand links within that distance of the
        target object, dropping the MANO URDF's virtual floating-base links."""
        self._hand_collision_filter = bool(enable)
        self._hand_collision_thresh = float(thresh)
        self._hand_points_radius = float(points_radius)

    def _mano_hand_points_world(self):
        """(M, 3) world-frame points sampling the human hand (MANO link origins),
        or None if no hand is present. Restricted to links near the target object
        so the floating-base virtual links (strung from the world origin to the
        wrist) don't become spurious obstacles near the object."""
        mano = getattr(self, "mano", None)
        body = getattr(mano, "body", None) if mano is not None else None
        if body is None or body.link_state is None:
            return None
        ls = body.link_state[0, :, 0:3]
        pts = np.asarray(ls.cpu().numpy() if hasattr(ls, "cpu") else ls, dtype=np.float32)

        obj = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 0:3]
        obj = np.asarray(obj.cpu().numpy() if hasattr(obj, "cpu") else obj, dtype=np.float32)
        pts = pts[np.linalg.norm(pts - obj[None], axis=1) < self._hand_points_radius]
        return pts if len(pts) > 0 else None

    def _hand_grasp_collision_mask(self, goal_set):
        """Boolean [n] mask (True = keep) for OMG's IK'd goal-set joint configs:
        rejects a grasp if any gripper control point (`hand_finger_point`, placed
        at the grasp's world EE pose) comes within `_hand_collision_thresh` of a
        MANO hand point. Registered as the OMG env's `external_grasp_filter`;
        returns all-True (no-op) unless the filter is enabled and a hand exists."""
        goal_set = np.asarray(goal_set)
        n = len(goal_set)
        if not self._hand_collision_filter or n == 0:
            return np.ones(n, dtype=bool)
        hand_pts = self._mano_hand_points_world()
        if hand_pts is None:
            return np.ones(n, dtype=bool)

        keep = np.ones(n, dtype=bool)
        for i, q in enumerate(goal_set):
            ee = unpack_pose(self._joint_to_world_ee_pose(q))          # 4x4 world
            gp = (np.matmul(ee[:3, :3], hand_finger_point) + ee[:3, 3:4]).T  # [6, 3]
            dmin = np.linalg.norm(gp[:, None, :] - hand_pts[None, :, :], axis=-1).min()
            keep[i] = dmin > self._hand_collision_thresh
        return keep

    def run_omg_planner(self, num_steps, scene_idx, reset_scene=True):
        if reset_scene:
            names = []
            poses = []
            for i in range(len(self.ycb.ids)):
                names += [self.ycb.CLASSES[self.ycb.ids[i]]]
                pos = self.ycb.pose[-1, i, 0:3]
                orn = self.ycb.pose[-1, i, 3:6]
                orn = Rot.from_euler("XYZ", orn).as_quat()
                pos, orn = pybullet.multiplyTransforms(*self._panda_base_invert_transform, pos, orn)
                poses += [pos + orn[3:] + orn[:3]]

            self._omg_planner.reset_scene(names, poses)

        start_conf = self.panda.body.dof_state[0, :, 0]
        target_name = self.ycb.CLASSES[self.ycb.ids[0]]

        traj, info = self._omg_planner.plan_to_target(start_conf, target_name, num_steps, scene_idx)

        if traj is None:
            print("Planning not run due to empty goal set.")
        else:
            # traj[-5] is the pre-grasp standoff (reach_tail_length=5,
            # standoff_dist=0.08 m) used as the RL goal; traj[-1] is the actual
            # grasp the gripper closes at. Keep both: the standoff for the RL
            # observation/reward, the grasp pose for visualization.
            self._omg_planner_goal_pose = self._joint_to_world_ee_pose(traj[-5])
            self._omg_grasp_pose = self._joint_to_world_ee_pose(traj[-1])

        return traj, info

    def get_omg_goal_grasp_pose(self):
        """4x4 world-frame pose of the grasp the OMG planner last planned to
        reach — the *final* trajectory waypoint (the pose the gripper closes
        at), not the pre-grasp standoff. None if the last run_omg_planner()
        found no goal. Deterministic per scene, so for a static handover this is
        the grasp the expert demonstrations aimed at — what the policy imitates."""
        if self._omg_grasp_pose is None:
            return None
        return unpack_pose(self._omg_grasp_pose)

    def get_omg_standoff_pose(self):
        """4x4 world-frame pose of the pre-grasp standoff (traj[-5], 8 cm back
        along the reach from the grasp) the OMG planner last planned to. None if
        the last run_omg_planner() found no goal. Used by DAgger collection to
        define the standoff plane: the approach phase the policy must imitate
        ends here; the final straight reach + close come from the demonstrations."""
        if self._omg_planner_goal_pose is None:
            return None
        return unpack_pose(self._omg_planner_goal_pose)

    def get_grasp_poses_world(self):
        """(N, 4, 4) world-frame poses of the filtered grasp candidates the OMG
        planner selects its goal from: the object-frame grasps (object ->
        panda_hand) placed at the live target-object pose. Requires
        run_omg_planner() to have populated the grasp set first."""
        grasps_obj = np.asarray(self._omg_planner.get_grasp_poses())
        if grasps_obj.ndim != 3:
            return np.empty((0, 4, 4))
        pos = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 0:3].tolist()
        orn = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 3:7].tolist()
        ycb_pose = pos + orn[3:] + orn[:3]
        return np.matmul(unpack_pose(ycb_pose), grasps_obj)

    def convert_target_joint_position_to_action(self, target_joint_position):
        current_joint_position = self.panda.body.dof_state[0, :, 0]
        current_ee_pose = self._panda_kinematics.forward_kinematics_parallel(
            joint_values=wrap_value(current_joint_position)[None], offset=False
        )[0, 7]
        target_ee_pose = self._panda_kinematics.forward_kinematics_parallel(
            joint_values=wrap_value(target_joint_position)[None], offset=False
        )[0, 7]
        delta_ee_pose = np.matmul(se3_inverse(current_ee_pose), target_ee_pose)
        action = np.hstack((delta_ee_pose[:3, 3], mat2euler(delta_ee_pose[:3, :3])))
        return action

    def get_ee_to_goal_pose(self, nearest=False):
        if nearest:
            return self._get_ee_to_nearest_goal_pose()
        ee_pose = self._get_ee_pose()
        ee_to_goal_pose = pack_pose_rot_first(
            inv_relative_pose(self._omg_planner_goal_pose, ee_pose)
        )
        return ee_to_goal_pose

    def _get_ee_to_nearest_goal_pose(self):
        ycb_to_goal_poses = self._omg_planner.get_grasp_poses()

        pos = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 0:3].tolist()
        orn = self.ycb.bodies[self.ycb.ids[0]].link_state[0, 6, 3:7].tolist()
        ycb_pose = pos + orn[3:] + orn[:3]
        ee_pose = self._get_ee_pose()
        goal_poses = np.matmul(unpack_pose(ycb_pose), ycb_to_goal_poses)
        ee_to_goal_poses = np.matmul(se3_inverse(unpack_pose(ee_pose)), goal_poses)

        point = hand_finger_point
        point_goal_poses = (
            np.matmul(ee_to_goal_poses[:, :3, :3], hand_finger_point) + ee_to_goal_poses[:, :3, 3:4]
        )
        index = np.argmin(np.mean(np.sum(np.abs(point - point_goal_poses), axis=1), axis=-1))
        ee_to_nearest_goal_pose = pack_pose_rot_first(ee_to_goal_poses[index])

        return ee_to_nearest_goal_pose
