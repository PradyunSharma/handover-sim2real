"""
Visualize a BC dataset episode.  Two modes:

  static  (default) — plots point clouds and trajectories using matplotlib.
                       No simulator required.

  replay            — loads the simulator, resets to the correct scene, then
                       replays the stored expert actions while overlaying the
                       saved point cloud as coloured debug points in PyBullet.
                       Useful for sanity-checking that dataset matches the sim.

Usage — static (random episode):
    python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train.h5

Usage — static (specific episode):
    python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train.h5 --episode 0

Usage — simulator replay:
    python examples/visualize_bc_dataset.py \
        --dataset output/bc_dataset/train.h5 \
        --mode replay \
        --cfg-file examples/pretrain.yaml \
        --episode 0
"""

import argparse
import h5py
import numpy as np
import sys


# ── helpers ──────────────────────────────────────────────────────────────────

def load_episode(dataset_path, ep_idx=None):
    """Return (metadata_dict, arrays_dict) for one episode."""
    with h5py.File(dataset_path, "r") as f:
        keys = [k for k in f.keys() if k.startswith("episode_")]
        if len(keys) == 0:
            raise RuntimeError(f"No episodes found in {dataset_path}")

        if ep_idx is None:
            ep_idx = np.random.randint(len(keys))
        key = f"episode_{ep_idx:05d}"
        if key not in f:
            raise KeyError(f"Episode {key} not found; dataset has {len(keys)} episodes")

        grp  = f[key]
        meta = dict(grp.attrs)
        data = {
            "point_clouds":   grp["point_clouds"][:],   # [T, N, C]
            "robot_states":   grp["robot_states"][:],   # [T, 32]
            "expert_actions": grp["expert_actions"][:], # [T, 7]
        }
        file_meta = dict(f.attrs)
    return meta, data, file_meta, ep_idx


# ── static visualisation ──────────────────────────────────────────────────────

def visualize_static(dataset_path, ep_idx):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    meta, data, file_meta, ep_idx = load_episode(dataset_path, ep_idx)
    pc  = data["point_clouds"]    # [T, N, C]
    rs  = data["robot_states"]    # [T, 32]
    act = data["expert_actions"]  # [T, 7]
    T   = len(act)

    print(f"Episode {ep_idx}  scene_idx={meta['scene_idx']}  steps={T}")
    print(f"  point_cloud shape : {pc.shape}")
    print(f"  robot_state shape : {rs.shape}")
    print(f"  expert_action shape: {act.shape}")

    # ── EE trajectory from robot state (indices 18:21 = ee_xyz) ──────────────
    ee_xyz = rs[:, 18:21]  # joint_pos(9)+joint_vel(9)+ee_pos(3)

    # ── actions ──────────────────────────────────────────────────────────────
    delta_pos   = act[:, 0:3]
    delta_euler = act[:, 3:6]
    gripper_cmd = act[:, 6]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"BC Dataset — episode {ep_idx}  (scene {meta['scene_idx']}, {T} steps)",
                 fontsize=13)

    # 1. 3-D point cloud at first, middle, last step
    sample_steps = sorted(set([0, T // 2, T - 1]))
    for col, t in enumerate(sample_steps):
        ax = fig.add_subplot(2, 5, col + 1, projection="3d")
        pts = pc[t]                  # [N, C]
        xyz = pts[:, :3]
        # colour by semantic flag: YCB=orange, hand=blue, background=grey
        ycb_flag  = pts[:, 3] > 0.5
        hand_flag = pts[:, 4] > 0.5
        colours = np.full((len(pts), 3), 0.6)  # grey background
        colours[ycb_flag]  = [1.0, 0.5, 0.0]  # orange
        colours[hand_flag] = [0.2, 0.4, 1.0]  # blue
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=colours, s=1, alpha=0.6)
        ax.set_title(f"Point cloud t={t}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.tick_params(labelsize=6)

    # 2. EE trajectory 3-D
    ax4 = fig.add_subplot(2, 5, 4, projection="3d")
    ax4.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], "o-", ms=3)
    ax4.scatter(*ee_xyz[0], color="green", s=50, label="start", zorder=5)
    ax4.scatter(*ee_xyz[-1], color="red",   s=50, label="end",   zorder=5)
    ax4.set_title("EE trajectory (world frame)", fontsize=9)
    ax4.set_xlabel("x"); ax4.set_ylabel("y"); ax4.set_zlabel("z")
    ax4.legend(fontsize=7)
    ax4.tick_params(labelsize=6)

    # 3. Joint positions over time
    ax5 = fig.add_subplot(2, 5, 5)
    for j in range(7):
        ax5.plot(rs[:, j], label=f"j{j}")
    ax5.set_title("Joint positions (arm)", fontsize=9)
    ax5.set_xlabel("step"); ax5.set_ylabel("rad")
    ax5.legend(fontsize=5, ncol=2)

    # 4. Delta position actions
    ax6 = fig.add_subplot(2, 5, 6)
    ax6.plot(delta_pos[:, 0], label="Δx")
    ax6.plot(delta_pos[:, 1], label="Δy")
    ax6.plot(delta_pos[:, 2], label="Δz")
    ax6.set_title("Expert Δposition", fontsize=9)
    ax6.set_xlabel("step"); ax6.legend(fontsize=7)

    # 5. Delta rotation actions
    ax7 = fig.add_subplot(2, 5, 7)
    ax7.plot(delta_euler[:, 0], label="Δroll")
    ax7.plot(delta_euler[:, 1], label="Δpitch")
    ax7.plot(delta_euler[:, 2], label="Δyaw")
    ax7.set_title("Expert Δrotation (euler)", fontsize=9)
    ax7.set_xlabel("step"); ax7.legend(fontsize=7)

    # 6. Gripper command
    ax8 = fig.add_subplot(2, 5, 8)
    ax8.step(range(T), gripper_cmd, where="post")
    ax8.set_ylim(-0.1, 1.1)
    ax8.set_yticks([0, 1]); ax8.set_yticklabels(["close", "open"])
    ax8.set_title("Gripper command", fontsize=9)
    ax8.set_xlabel("step")

    # 7. Gripper state (from robot state)
    # layout: joint_pos(9)+joint_vel(9)+ee_pos(3)+ee_orn(4)+gripper(1)+prev_act(6)
    #         indices 0-8    9-17        18-20     21-24     25         26-31
    ax9 = fig.add_subplot(2, 5, 9)
    ax9.plot(rs[:, 25], label="gripper norm")
    ax9.set_ylim(-0.05, 1.05)
    ax9.set_title("Gripper state (normalised)", fontsize=9)
    ax9.set_xlabel("step")

    # 8. Action magnitude
    ax10 = fig.add_subplot(2, 5, 10)
    mag = np.linalg.norm(act[:, :6], axis=1)
    ax10.plot(mag)
    ax10.set_title("Action magnitude ‖Δ‖", fontsize=9)
    ax10.set_xlabel("step")

    plt.tight_layout()
    plt.show()


# ── simulator replay ──────────────────────────────────────────────────────────

def visualize_replay(dataset_path, ep_idx, cfg_file):
    import gym
    import pybullet
    import time

    import handover
    import handover_sim2real

    from handover.benchmark_wrapper import HandoverBenchmarkWrapper
    from handover_sim2real.config import get_cfg
    from handover_sim2real.utils import add_sys_path_from_env

    add_sys_path_from_env("GADDPG_DIR")
    from experiments.config import cfg_from_file

    meta, data, file_meta, ep_idx = load_episode(dataset_path, ep_idx)
    scene_idx  = int(meta["scene_idx"])
    expert_act = data["expert_actions"]    # [T, 7]  (we only need the Δee part)
    saved_pc   = data["point_clouds"]      # [T, N, C]
    T = len(expert_act)

    print(f"Replaying episode {ep_idx}  scene_idx={scene_idx}  steps={T}")

    cfg = get_cfg()
    cfg_from_file(filename=cfg_file, dict=cfg, merge_to_cn_dict=True)
    cfg.SIM.RENDER = True  # open PyBullet GUI

    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))

    # Run OMG planner to get original expert joint trajectory for replay
    obs = env.reset(idx=scene_idx)
    expert_plan, _ = env.run_omg_planner(cfg.RL_MAX_STEP, scene_idx)
    if expert_plan is None:
        print("OMG planner failed for this scene — cannot replay.")
        return

    stop_step           = len(expert_plan)
    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION
    )

    from handover_sim2real.utils import add_sys_path_from_env
    add_sys_path_from_env("GADDPG_DIR")
    from core.utils import tf_quat, unpack_pose, se3_transform_pc
    from scipy.spatial.transform import Rotation as Rot

    panda_base_pos = np.array(cfg.ENV.PANDA_BASE_POSITION)
    panda_base_orn = np.array(cfg.ENV.PANDA_BASE_ORIENTATION)  # xyzw
    R_base = Rot.from_quat(panda_base_orn).as_matrix()

    def play_once(obs):
        debug_ids = []
        for step in range(min(stop_step, T)):
            link_ind = obs["panda_link_ind_hand"]
            pos_world = obs["panda_body"].link_state[0, link_ind, 0:3]
            orn_world = obs["panda_body"].link_state[0, link_ind, 3:7]
            pos_base, orn_base = pybullet.multiplyTransforms(
                *panda_base_inv_tf, pos_world, orn_world
            )
            ee_mat = unpack_pose(np.hstack([pos_base, tf_quat(orn_base)]))

            # Point cloud is stored in EE frame → transform to world for display.
            pts_ee   = saved_pc[step, :, :3]
            pts_base = se3_transform_pc(ee_mat, pts_ee.T).T
            pts_world = (R_base @ pts_base.T).T + panda_base_pos

            ycb_flag  = saved_pc[step, :, 3] > 0.5
            hand_flag = saved_pc[step, :, 4] > 0.5
            colours = np.full((len(pts_world), 3), 0.6)
            colours[ycb_flag]  = [1.0, 0.5, 0.0]
            colours[hand_flag] = [0.3, 0.5, 1.0]

            for dbg_id in debug_ids:
                pybullet.removeUserDebugItem(dbg_id)
            debug_ids.clear()

            idx_show = np.random.choice(
                len(pts_world), size=min(200, len(pts_world)), replace=False
            )
            dbg_id = pybullet.addUserDebugPoints(
                pts_world[idx_show].tolist(),
                colours[idx_show].tolist(),
                pointSize=4,
            )
            debug_ids.append(dbg_id)

            done = False
            for _ in range(steps_action_repeat):
                obs, _, done, _ = env.step(expert_plan[step])
                if done:
                    break

            time.sleep(0.05)
            if done:
                break

        # Clear debug points once playback ends so the next replay starts clean.
        for dbg_id in debug_ids:
            pybullet.removeUserDebugItem(dbg_id)
        return obs

    obs = play_once(obs)

    print("Replay finished.")
    print("In the PyBullet window:  R = replay,  Q = quit.")

    R_KEY = ord('r')
    Q_KEY = ord('q')
    try:
        while True:
            keys = pybullet.getKeyboardEvents()
            if R_KEY in keys and keys[R_KEY] & pybullet.KEY_WAS_TRIGGERED:
                print("Replaying...")
                obs = env.reset(idx=scene_idx)
                obs = play_once(obs)
                print("Replay finished.  Press R to replay again, Q to quit.")
            if Q_KEY in keys and keys[Q_KEY] & pybullet.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualise a BC dataset episode.")
    p.add_argument("--dataset",  required=True, help="path to HDF5 dataset file")
    p.add_argument("--episode",  type=int, default=None,
                   help="episode index (default: random)")
    p.add_argument("--mode",     default="static", choices=["static", "replay"],
                   help="static=matplotlib plots, replay=PyBullet simulator")
    p.add_argument("--cfg-file", default=None,
                   help="config yaml (required for --mode replay)")
    p.add_argument("--seed",     type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    if args.mode == "static":
        visualize_static(args.dataset, args.episode)
    else:
        if args.cfg_file is None:
            print("Error: --cfg-file is required for --mode replay")
            sys.exit(1)
        visualize_replay(args.dataset, args.episode, args.cfg_file)


if __name__ == "__main__":
    main()
