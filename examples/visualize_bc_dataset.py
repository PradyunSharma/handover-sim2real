"""
Visualize a BC dataset episode.  Two modes:

  static  (default) — plots point clouds and trajectories using matplotlib.
                       No simulator required.

  replay            — loads the simulator, resets to the correct scene, then
                       drives the robot through the recorded rollout while
                       overlaying the saved (EE-frame) point cloud as coloured
                       debug points in PyBullet. Two drivers (--replay-source):
                         states (default) — follows the stored robot_states, so it
                           faithfully reproduces the recorded rollout (the POLICY's
                           path for DAgger data, the expert's for the offline set),
                           and the point cloud lines up with the object/hand.
                         omg — re-plans the OMG expert and steps that instead; only
                           matches the OFFLINE expert dataset. For DAgger data this
                           shows the expert (not the policy) and the cloud will not
                           align — use 'states'.
                       At each step it also draws the full EXPERT-action label (the
                       OMG target):
                         - translation Δpos as a shaft from the current EE with a
                           3-D arrowhead at the TIP (where to go) — green=gripper
                           open, red=gripper close;
                         - rotation Δeuler as a small orientation triad at the tip
                           (X=yellow, Y=magenta, Z=cyan) = the commanded gripper
                           orientation.
                       Both Δpos and Δangle are exaggerated by --arrow-scale (the
                       per-step deltas are only ~cm / a few degrees). Arrows persist
                       through the rollout AND stay on screen after it ends (cleared
                       only when you press R to replay). Disable with
                       --no-expert-arrows.

Usage — static (random episode):
    python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train.h5

Usage — static (specific episode):
    python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train.h5 --episode 0

Usage — simulator replay:
    python examples/visualize_bc_dataset.py \
        --dataset output/bc_dataset/train.h5 \
        --mode replay \
        --cfg-file examples/pretrain.yaml \
        --episode 0 \
        [--show-goal-grasp]   # overlay the grasp (green) + standoff (cyan) gripper

Also accepts an RL demo pool (examples/collect_rl_demos.py) — a streamed `.h5` or
a legacy `.npz`: episodes are split at terminal==1, the normalized pose is
denormalized, and the gripper bit is taken from the stored logit (OPEN iff >= 0).
Same static / replay modes:
    python examples/visualize_bc_dataset.py --dataset output/rl_demos/train.h5 \
        --mode replay --cfg-file examples/pretrain.yaml --episode 0
"""

import argparse
import h5py
import numpy as np
import sys


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_episode_flat(full, sl, has, dataset_path, ep_idx=None):
    """Load one episode from a flat RL demo pool (examples/collect_rl_demos.py) —
    an npz or a streamed HDF5. Per-transition arrays are split into episodes at
    terminal==1; `full(key)`->whole array, `sl(key,a,b)`->row slice (lazy for h5),
    `has(key)`->bool abstract the backend. Denormalizes the stored pose back to
    real Δpos/Δeuler (via action_mean/std) and rebuilds a 7-D expert_actions with
    a gripper bit (OPEN iff the stored logit >= 0), so static + replay work."""
    term = np.asarray(full("terminal")).reshape(-1)
    ends = np.where(term >= 0.5)[0]                       # each episode ends at a terminal
    if len(ends) == 0:
        raise RuntimeError(f"No episodes (no terminal flags) in {dataset_path}")
    starts = np.concatenate([[0], ends[:-1] + 1])
    if ep_idx is None:
        ep_idx = int(np.random.randint(len(ends)))
    if ep_idx >= len(ends):
        raise KeyError(f"Episode {ep_idx} not found; pool has {len(ends)} episodes")
    a, b = int(starts[ep_idx]), int(ends[ep_idx]) + 1

    action = np.asarray(sl("action", a, b))              # [T,7] normalized pose + gripper logit
    mean = np.asarray(full("action_mean")) if has("action_mean") else np.zeros(6, np.float32)
    std  = np.asarray(full("action_std"))  if has("action_std")  else np.ones(6, np.float32)
    pose_real = action[:, :6] * std + mean               # -> real Δpos/Δeuler
    grip_bit  = (action[:, 6] >= 0.0).astype(np.float32)  # 1=open, 0=close
    expert_actions = np.concatenate([pose_real, grip_bit[:, None]], axis=1).astype(np.float32)

    scene_idx = int(np.asarray(sl("scene_idx", a, a + 1))[0]) if has("scene_idx") else 0
    meta = {"scene_idx": scene_idx, "num_steps": b - a}
    data = {"point_clouds": np.asarray(sl("pc", a, b)),
            "robot_states": np.asarray(sl("rs", a, b)),
            "expert_actions": expert_actions}
    reward_end = float(np.asarray(sl("reward", b - 1, b)).reshape(-1)[0])
    print(f"[rl-demo] episode {ep_idx}/{len(ends)}  scene_idx={scene_idx}  "
          f"steps={b - a}  reward@end={reward_end:.0f}")
    return meta, data, {"rl_demo": True}, ep_idx


def load_episode(dataset_path, ep_idx=None):
    """Return (metadata_dict, arrays_dict, file_meta, ep_idx) for one episode.
    Accepts a BC per-episode HDF5, or an RL demo pool (flat npz OR streamed HDF5)
    — auto-detected by extension and, for HDF5, by a top-level `terminal`
    dataset (flat RL pool) vs `episode_*` groups (BC dataset)."""
    if str(dataset_path).endswith(".npz"):
        d = np.load(dataset_path)
        return _load_episode_flat(lambda k: d[k], lambda k, a, b: d[k][a:b],
                                  lambda k: k in d.files, dataset_path, ep_idx)
    with h5py.File(dataset_path, "r") as f:
        if "terminal" in f:                              # flat RL demo pool (streamed)
            return _load_episode_flat(lambda k: f[k][:], lambda k, a, b: f[k][a:b],
                                      lambda k: k in f, dataset_path, ep_idx)
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

def visualize_replay(dataset_path, ep_idx, cfg_file, source="states",
                     show_expert=True, arrow_scale=3.0, show_goal_grasp=False):
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
    scene_idx    = int(meta["scene_idx"])
    saved_pc     = data["point_clouds"]      # [T, N, C]  EE-frame cloud per step
    robot_states = data["robot_states"]      # [T, 32]  joint_pos(9)+... per step
    expert_act   = data["expert_actions"]    # [T, 7]  OMG label at each visited state
    T = len(saved_pc)

    is_dagger = bool(file_meta.get("dagger", False))
    print(f"Replaying episode {ep_idx}  scene_idx={scene_idx}  steps={T}  "
          f"source={source}  (dagger={is_dagger})")
    if source == "omg" and is_dagger:
        print("  NOTE: --replay-source omg re-plans the OMG expert and steps THAT, "
              "not the\n        policy's recorded states — for DAgger data the robot "
              "and the point\n        cloud will NOT match the recorded rollout. Use "
              "--replay-source states.")

    cfg = get_cfg()
    cfg_from_file(filename=cfg_file, dict=cfg, merge_to_cn_dict=True)
    cfg.SIM.RENDER = True  # open PyBullet GUI

    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))

    obs = env.reset(idx=scene_idx)

    # 'omg' mode re-plans the expert and drives the sim along it (only faithful for
    # the OFFLINE expert dataset). 'states' mode drives the sim through the stored
    # robot_states, faithfully reproducing whatever rollout was recorded. We also
    # run OMG (once, from the reset config = same grasp the data aimed at) when
    # --show-goal-grasp is set, just to read back the goal grasp / standoff poses.
    expert_plan = None
    if source == "omg" or show_goal_grasp:
        expert_plan, _ = env.run_omg_planner(cfg.RL_MAX_STEP, scene_idx)
        if expert_plan is None:
            if source == "omg":
                print("OMG planner failed — cannot replay (--replay-source omg).")
                return
            print("OMG planner failed — cannot draw --show-goal-grasp.")

    stop_step           = len(expert_plan) if (source == "omg" and expert_plan is not None) else T
    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION
    )

    from handover_sim2real.utils import add_sys_path_from_env
    add_sys_path_from_env("GADDPG_DIR")
    from core.utils import tf_quat, unpack_pose, se3_transform_pc, euler2mat
    from scipy.spatial.transform import Rotation as Rot

    panda_base_pos = np.array(cfg.ENV.PANDA_BASE_POSITION)
    panda_base_orn = np.array(cfg.ENV.PANDA_BASE_ORIENTATION)  # xyzw
    R_base = Rot.from_quat(panda_base_orn).as_matrix()

    def draw_gripper(pose_mat, colour, line_ids, line_width=2.0):
        """Panda parallel-jaw wireframe at 4x4 world pose (same convention as
        rollout_bc_policy.draw_gripper / visualize_grasps.gripper_segments)."""
        from visualize_grasps import gripper_segments
        for p, q in gripper_segments(pose_mat):
            line_ids.append(pybullet.addUserDebugLine(
                p.tolist(), q.tolist(), lineColorRGB=colour, lineWidth=line_width))

    # Goal-grasp overlay (static per scene → drawn once, kept up the whole session).
    goal_ids = []
    if show_goal_grasp:
        goal_mat     = env.get_omg_goal_grasp_pose()   # traj[-1], the grasp pose
        standoff_mat = env.get_omg_standoff_pose()     # traj[-5], the pre-grasp pose
        if goal_mat is not None:
            draw_gripper(goal_mat, [0.0, 1.0, 0.0], goal_ids, 3.0)        # green
            print(f"  goal grasp (green)    pos={goal_mat[:3, 3].round(3)}")
        if standoff_mat is not None:
            draw_gripper(standoff_mat, [0.0, 1.0, 1.0], goal_ids, 2.0)    # cyan
            print(f"  pre-grasp standoff (cyan) pos={standoff_mat[:3, 3].round(3)}"
                  f"  — where approach-only DAgger labels stop")
        if goal_mat is None and standoff_mat is None:
            print("  --show-goal-grasp: OMG found no goal grasp — nothing to draw.")

    # Expert-action arrows live in the enclosing scope so they stay on screen
    # after playback ends; they're wiped only when a new replay (R) starts.
    arrow_ids = []

    def play_once(obs):
        debug_ids = []
        for aid in arrow_ids:
            pybullet.removeUserDebugItem(aid)
        arrow_ids.clear()
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

            # Expert-action arrow: where OMG (the label) says to move the EE from
            # this policy-visited state. act[:3] is the Δpos in the EE frame; map it
            # to world via the current EE pose and draw a (scaled) line. Green =
            # gripper-open label, red = gripper-close label. Arrows persist so the
            # whole correction "field" along the rollout stays visible.
            if show_expert:
                act = expert_act[step]
                cur_base = ee_mat[:3, 3]
                tgt_base = ee_mat[:3, :3] @ act[:3] + cur_base
                cur_w = R_base @ cur_base + panda_base_pos
                tgt_w = R_base @ tgt_base + panda_base_pos
                tgt_w = cur_w + arrow_scale * (tgt_w - cur_w)   # exaggerate (Δ≈3-4cm)
                colour = [0.0, 0.9, 0.0] if act[6] >= 0.5 else [0.9, 0.0, 0.0]
                # Shaft (current EE → expert target) + a 3-D arrowhead at the TIP
                # so the start (gripper) vs end (where to go) is unambiguous.
                arrow_ids.append(pybullet.addUserDebugLine(
                    cur_w.tolist(), tgt_w.tolist(), colour, lineWidth=3))
                d = tgt_w - cur_w
                L = float(np.linalg.norm(d))
                if L > 1e-6:
                    dh = d / L
                    ref = np.array([0.0, 0.0, 1.0]) if abs(dh[2]) < 0.9 \
                        else np.array([1.0, 0.0, 0.0])
                    p1 = np.cross(dh, ref); p1 /= np.linalg.norm(p1)
                    p2 = np.cross(dh, p1)
                    base_h = tgt_w - 0.25 * L * dh          # arrowhead length 25%
                    for pv in (p1, p2):
                        for s in (1.0, -1.0):
                            head = base_h + s * 0.12 * L * pv
                            arrow_ids.append(pybullet.addUserDebugLine(
                                tgt_w.tolist(), head.tolist(), colour, lineWidth=3))

                # Rotation part of the label (act[3:6], Δeuler in the EE frame):
                # draw the commanded gripper orientation as a triad at the tip.
                # X=yellow, Y=magenta, Z=cyan (kept distinct from the green/red
                # translation shaft). The Δangle is exaggerated by arrow_scale too,
                # so the (few-degree) per-step rotation is actually visible.
                R_delta = euler2mat(act[3], act[4], act[5])          # EE-frame, sxyz
                rotvec = Rot.from_matrix(R_delta).as_rotvec() * arrow_scale
                R_delta_ex = Rot.from_rotvec(rotvec).as_matrix()
                tgt_rot_world = R_base @ (ee_mat[:3, :3] @ R_delta_ex)
                axis_cols = ([1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0])
                for a in range(3):
                    ax_end = tgt_w + 0.04 * tgt_rot_world[:, a]
                    arrow_ids.append(pybullet.addUserDebugLine(
                        tgt_w.tolist(), ax_end.tolist(), axis_cols[a], lineWidth=2))

            # Advance the sim to the next recorded state. 'states': drive the
            # panda toward the next stored joint config (faithfully reproduces the
            # recorded rollout — policy states for DAgger, expert for offline).
            # 'omg': step the freshly re-planned expert trajectory (old behavior).
            done = False
            if source == "omg":
                target = expert_plan[step]
            else:
                target = robot_states[min(step + 1, T - 1), :9]
            for _ in range(steps_action_repeat):
                obs, _, done, _ = env.step(target)
                if done:
                    break

            time.sleep(0.05)
            if done:
                break

        # Clear only the (per-step) point cloud when playback ends; the expert
        # arrows stay on screen so the finished correction field is inspectable.
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
    p.add_argument("--dataset",  required=True,
                   help="BC HDF5 dataset, or an RL demo pool (.h5/.npz) from collect_rl_demos.py")
    p.add_argument("--episode",  type=int, default=None,
                   help="episode index (default: random)")
    p.add_argument("--mode",     default="static", choices=["static", "replay"],
                   help="static=matplotlib plots, replay=PyBullet simulator")
    p.add_argument("--cfg-file", default=None,
                   help="config yaml (required for --mode replay)")
    p.add_argument("--replay-source", default="states", choices=["states", "omg"],
                   help="replay mode driver: 'states' drives the sim through the "
                        "recorded robot_states (faithful — use this for DAgger data); "
                        "'omg' re-plans the expert and steps that (old behavior, only "
                        "matches the offline expert dataset).")
    p.add_argument("--no-expert-arrows", dest="show_expert", action="store_false",
                   help="hide the per-step expert-action (OMG label) arrows in replay.")
    p.add_argument("--arrow-scale", type=float, default=3.0,
                   help="exaggeration factor for the expert-action arrows "
                        "(the per-step Δpos is only ~3-4 cm; default 3×).")
    p.add_argument("--show-goal-grasp", action="store_true",
                   help="overlay the gripper pose OMG planned to reach — green = "
                        "goal grasp (traj[-1]), cyan = pre-grasp standoff (traj[-5], "
                        "where approach-only DAgger labels stop). Runs OMG once.")
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
        visualize_replay(args.dataset, args.episode, args.cfg_file, args.replay_source,
                         args.show_expert, args.arrow_scale, args.show_goal_grasp)


if __name__ == "__main__":
    main()
