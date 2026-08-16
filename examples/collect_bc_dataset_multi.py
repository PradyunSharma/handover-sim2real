"""
Collect the Phase-5 offline BC dataset: K demonstrations per scene, one per
candidate grasp, using OMG Planner as expert teacher.

This is `collect_bc_dataset.py` with the unit of work changed from a scene to a
`(scene, grasp_idx)` pair. Phase 5 conditions the policy on which grasp it is
being asked to reach, so every scene needs a demonstration for each of its
candidate grasps (`build_grasp_pin_table_multi.py`). Nothing else about the
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
`bc5/dataset.py` never has to look a table up, so rebuilding a pin table can no
longer silently retarget a dataset collected against the old one.

Usage:
    python examples/collect_bc_dataset_multi.py \\
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
from handover_sim2real.policy import PointListener
from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("GADDPG_DIR")

from core.utils import tf_quat, unpack_pose
from experiments.config import cfg_from_file


# ── dimension constants ──────────────────────────────────────────────────────
ROBOT_STATE_DIM = 32   # joint_pos(9) + joint_vel(9) + EE(7) + gripper(1) + prev_act(6)
ACTION_DIM      = 7    # Δpos(3) + Δeuler(3) + gripper_cmd(1)
NUM_PTS         = 1024
PC_CHANNELS     = 5    # xyz + ycb_flag + hand_flag


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
                    grasp_idx=0):
    """
    Run one episode and return a dict of arrays, or None if it could not be run.

    The approach plays the full OMG trajectory so the gripper actually reaches
    the grasp pose. One final gripper-close transition is appended at the end.

    `pin_table` (examples/build_grasp_pin_table_multi.py) forces slot `grasp_idx`
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

    if pin_table is not None:
        if not pin_table.apply(env, scene_idx, grasp_idx):
            return None
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

    for step in range(stop_step):
        pc    = _point_cloud(obs, point_listener, panda_base_inv_tf)
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
        pc = _point_cloud(obs, point_listener, panda_base_inv_tf)
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
                        "(examples/build_grasp_pin_table_multi.py). Required: it "
                        "is what defines the slots this script iterates over.")
    p.add_argument("--grasps-per-scene", type=int, default=None,
                   help="cap on slots per scene (default: every slot the table "
                        "holds, i.e. K from build_grasp_pin_table_multi.py)")
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
    # This script reaches gym.make() without importing dagger5.env_setup,
    # so the OMG_PLANNER_DIR guard has to be asked for explicitly. Without
    # it a bad path surfaces as `ModuleNotFoundError: No module named 'omg'`
    # from inside the gym registry, which names a dependency when the real
    # problem is a path.
    from handover_sim2real.dagger5.env_setup import preflight
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

    from handover_sim2real.dagger5 import load_grasp_pin_table
    pin_table = load_grasp_pin_table(
        args.grasp_pin_table,
        sim_cfg_block={"valid_grasp_dict_path": args.valid_grasp_dict,
                       "hand_collision_filter": False,
                       "split": args.split})   # scene indices are split-relative

    # The work list: every (scene, slot) the table offers, scene-major, so a
    # partial run still covers whole scenes and `--num-episodes` stays readable.
    jobs = []
    for scene_idx in range(num_scenes):
        n_slots = pin_table.num_grasps_for(scene_idx)
        if args.grasps_per_scene is not None:
            n_slots = min(n_slots, int(args.grasps_per_scene))
        jobs.extend((scene_idx, g) for g in range(n_slots))
    if args.num_episodes is not None:
        jobs = jobs[:int(args.num_episodes)]
    print(f"  {len(jobs)} (scene, grasp) pairs over {num_scenes} scenes")

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
        f.attrs["pc_format"]       = "xyz(3)+ycb_flag(1)+hand_flag(1) in EE frame"
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

        for job_i, (scene_idx, grasp_idx) in enumerate(jobs):
            episode = collect_episode(
                env, point_listener, cfg, scene_idx,
                panda_base_inv_tf, steps_action_repeat, pin_table=pin_table,
                grasp_idx=grasp_idx,
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
