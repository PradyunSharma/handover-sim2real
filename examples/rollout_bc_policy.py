"""
Roll out a trained Phase-1 BC policy in the simulator (closed loop).

Unlike examples/visualize_bc_dataset.py --mode replay (which replays OMG's
stored joint trajectory), this script drives the robot with the *policy's own*
predicted actions, step by step:

    obs ─► build (point_cloud, robot_state) exactly as during collection
        ─► model.predict() ─► Δee-pose (6) + gripper bit (1)
        ─► Δee-pose ∘ current ee-pose ─► IK ─► target joint position
        ─► step the sim steps_action_repeat times
        ─► repeat

This is the real qualitative test: a low pose-L1 in analyze_bc_run.py only
says the policy matches the expert *on states the expert visited*. Closed-loop
rollout shows whether the policy actually reaches and grasps the object when
it has to live with its own accumulated error (covariate shift).

The live point cloud the policy sees is overlaid as coloured debug points
(orange = YCB, blue = hand, grey = background).

Usage:
    python examples/rollout_bc_policy.py \
        --run-dir  output/bc_runs/phase1_full \
        --cfg-file examples/pretrain.yaml \
        --scene    0 [--show-goal-grasp] [--show-grasp-set]

Add --show-goal-grasp to overlay the gripper pose OMG planned to reach for the
scene (green wireframe). The OMG goal is deterministic per scene, so for a
static handover it is exactly the grasp the expert demos aimed at — the target
the policy is imitating; watch the live gripper converge to or miss it.
--show-grasp-set additionally draws the full filtered candidate set (faint grey).

In the PyBullet window:  R = re-roll the same scene,  N = next scene,  Q = quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sibling example modules importable (collect_bc_dataset) regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gym
import numpy as np
import pybullet
import torch
import yaml

import handover            # noqa: F401  registers envs
import handover_sim2real   # noqa: F401  registers envs

from handover.benchmark_wrapper import HandoverBenchmarkWrapper, EpisodeStatus
from handover_sim2real.config import get_cfg
from handover_sim2real.policy import PointListener, HandoverSim2RealPolicy
from handover_sim2real.eval_wrapper import GraspBenchmarkWrapper
from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("GADDPG_DIR")
from core.utils import unpack_action, tf_quat, unpack_pose, se3_transform_pc  # noqa: E402
from experiments.config import cfg_from_file  # noqa: E402

# Reuse the exact observation -> (robot_state, point_cloud) builders used at
# collection time so the policy sees the same representation it trained on.
from collect_bc_dataset import _robot_state, _point_cloud  # noqa: E402


# ── model loading ────────────────────────────────────────────────────────────

def load_policy(run_dir: Path, device: str):
    from handover_sim2real.bc import BCPolicy, Normalizer

    with (run_dir / "config.yaml").open() as f:
        rcfg = yaml.safe_load(f)

    norm_path = run_dir / "normalization.npz"
    normalizer = Normalizer.load(norm_path) if norm_path.exists() else None
    if normalizer is None:
        print("WARNING: no normalization.npz — rollout will likely be garbage.")

    m, d = rcfg["MODEL"], rcfg["DATA"]
    model = BCPolicy(
        pc_channels        = int(d["pc_channels"]),
        robot_state_dim    = int(d["robot_state_dim"]),
        action_dim         = int(d["action_dim"]),
        feature_dim        = int(m["feature_dim"]),
        robot_hidden       = int(m["robot_hidden"]),
        policy_hidden      = tuple(m["policy_hidden"]),
        pointnet_scale     = int(m["pointnet_scale"]),
        pointnet_radius    = float(m["pointnet_radius"]),
        pointnet_nclusters = int(m["pointnet_nclusters"]),
        use_prev_act       = bool(m.get("use_prev_act", True)),
        freeze_pc          = bool(m.get("freeze_pc", False)),
        normalizer         = normalizer,
    ).to(device)

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (epoch {payload.get('epoch', '?')})")
    return model


# ── action application ─────────────────────────────────────────────────────

def action_to_target_joint(action, obs):
    """Δee-pose(6)+gripper(1) → target joint position via IK.

    Mirrors HandoverSim2RealPolicy.convert_action_to_target_joint_position,
    plus sets the fingers from the predicted gripper bit (≥0.5 → open).
    """
    pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
    orn = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 3:7].numpy()
    ee_pose = unpack_pose(np.hstack((pos, tf_quat(orn))))

    target_ee_pose = np.matmul(ee_pose, unpack_action(action[:6]))
    tpos = target_ee_pose[:3, 3]
    from scipy.spatial.transform import Rotation as Rot
    torn = Rot.from_matrix(target_ee_pose[:3, :3]).as_quat()

    tjp = pybullet.calculateInverseKinematics(
        obs["panda_body"].contact_id[0], obs["panda_link_ind_hand"] - 1, tpos, torn
    )
    tjp = np.array(tjp)
    finger = 0.04 if action[6] >= 0.5 else 0.0
    tjp[7:9] = finger
    return tjp


# ── point cloud overlay ─────────────────────────────────────────────────────

def draw_pointcloud(pc_ee, ee_mat, R_base, panda_base_pos, debug_ids):
    """Overlay the policy-input cloud (EE frame, [N,5]) in world frame."""
    for d in debug_ids:
        pybullet.removeUserDebugItem(d)
    debug_ids.clear()

    pts_ee   = pc_ee[:, :3]
    pts_base = se3_transform_pc(ee_mat, pts_ee.T).T
    pts_world = (R_base @ pts_base.T).T + panda_base_pos

    ycb  = pc_ee[:, 3] > 0.5
    hand = pc_ee[:, 4] > 0.5
    colours = np.full((len(pts_world), 3), 0.6)
    colours[ycb]  = [1.0, 0.5, 0.0]
    colours[hand] = [0.3, 0.5, 1.0]

    idx = np.random.choice(len(pts_world), size=min(200, len(pts_world)), replace=False)
    debug_ids.append(pybullet.addUserDebugPoints(
        pts_world[idx].tolist(), colours[idx].tolist(), pointSize=4))


# ── goal-grasp overlay ───────────────────────────────────────────────────────

def draw_gripper(pose_mat, colour, line_ids, line_width=2.0):
    """Draw a Panda parallel-jaw gripper wireframe at 4x4 world pose `pose_mat`.

    Reuses the gripper stick-figure geometry from visualize_grasps.py (origin =
    panda_hand, +z approach, ±y fingers) — the same convention the FK goal pose
    and the object-frame grasp set are expressed in. Appends the created line
    ids to `line_ids` so the caller can remove them later.
    """
    # Lazy import so the headless/benchmark path never pulls in matplotlib.
    from visualize_grasps import gripper_segments
    for p, q in gripper_segments(pose_mat):
        line_ids.append(pybullet.addUserDebugLine(
            p.tolist(), q.tolist(), lineColorRGB=colour, lineWidth=line_width))


# ── rollout ──────────────────────────────────────────────────────────────────

def rollout(env, model, point_listener, gb_policy, scene_idx, device,
            panda_base_inv_tf, steps_action_repeat, max_steps,
            R_base, panda_base_pos, draw=True,
            show_goal_grasp=False, show_grasp_set=False,
            omg_steps=None, goal_marker_ids=None):
    obs = env.reset(idx=scene_idx)
    point_listener.reset()

    # Optionally overlay the grasp OMG planned to reach for this scene. The OMG
    # goal is deterministic per scene, so for a static handover it is exactly
    # the grasp the expert demonstrations aimed at. Drawn once (the object is
    # static) and left up for the whole roll so you can see the gripper close
    # the gap; replaced on the next roll via the shared goal_marker_ids list.
    if show_goal_grasp:
        if goal_marker_ids is None:
            goal_marker_ids = []
        for d in goal_marker_ids:
            pybullet.removeUserDebugItem(d)
        goal_marker_ids.clear()

        env.run_omg_planner(omg_steps or max_steps, scene_idx)  # plans, no sim step
        if show_grasp_set:
            try:
                for T in env.get_grasp_poses_world():
                    draw_gripper(T, [0.55, 0.55, 0.55], goal_marker_ids, 1.0)
            except Exception as e:  # viz only — never abort the rollout
                print(f"  (could not draw grasp set: {e})")
        goal_mat = env.get_omg_goal_grasp_pose()
        if goal_mat is not None:
            draw_gripper(goal_mat, [0.0, 1.0, 0.0], goal_marker_ids, 3.0)
            print(f"  OMG goal grasp (green) at pos={goal_mat[:3, 3].round(3)}")
        else:
            print("  OMG found no goal grasp for this scene — nothing to draw.")

    debug_ids = []
    status = 0
    done = False
    info = {}
    dist = float("nan")
    close_step = -1
    grasped = False

    # prev_action matches collection: zeros at step 0, then the previous step's
    # raw 6-D Δee action. model.predict() returns denormalized (real-unit)
    # actions, so action[:6] is in the same space as the expert delta that was
    # stored as prev_act during collection.
    prev_act6d = np.zeros(6, dtype=np.float32)

    print(f"\n--- scene {scene_idx} ---")
    for step in range(max_steps):
        pc = _point_cloud(obs, point_listener, panda_base_inv_tf)   # [N,5] EE frame
        rs = _robot_state(obs, prev_act6d)                          # [32]

        pc_t = torch.from_numpy(pc).float().unsqueeze(0).to(device)
        rs_t = torch.from_numpy(rs).float().unsqueeze(0).to(device)
        action = model.predict(pc_t, rs_t)[0].cpu().numpy()         # [7] real units
        prev_act6d = action[:6].astype(np.float32)                  # for next step's robot_state

        ee_pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
        ycb_pos = env.ycb.bodies[env.ycb.ids[0]].link_state[0, 6, 0:3].numpy()
        dist = np.linalg.norm(ee_pos - ycb_pos)

        # First time the policy commands a grasp (gripper close), hand off to the
        # scripted grasp-and-back: close in place (standoff 0) and carry to
        # GOAL_CENTER, then let the benchmark decide success.
        if action[6] < 0.5:
            close_step = step
            print(f"  step {step:3d}  GRASP commanded  ee→ycb={dist:.3f} m  → grasp_and_back")
            gb_policy.reset()
            gb_done = False
            while not gb_done and not done:
                gb_action, gb_done = gb_policy.grasp_and_back(obs)
                for _ in range(steps_action_repeat):
                    obs, _, done, info = env.step(gb_action)
                    if done:
                        break
                status = info.get("status", 0)
                grasped = grasped or env.grasped_active()
            break

        # Approach step (gripper still open).
        if draw:
            link = obs["panda_link_ind_hand"]
            pw = obs["panda_body"].link_state[0, link, 0:3]
            ow = obs["panda_body"].link_state[0, link, 3:7]
            pb, ob = pybullet.multiplyTransforms(*panda_base_inv_tf, pw, ow)
            ee_mat = unpack_pose(np.hstack([pb, tf_quat(ob)]))
            draw_pointcloud(pc, ee_mat, R_base, panda_base_pos, debug_ids)

        tjp = action_to_target_joint(action, obs)
        print(f"  step {step:3d}  Δpos={action[:3].round(3)}  grip=open   ee→ycb={dist:.3f} m")

        for _ in range(steps_action_repeat):
            obs, _, done, info = env.step(tjp)
            if done:
                break
        status = info.get("status", 0)
        if done:
            break
        if draw:
            time.sleep(0.03)

    # Report outcome.
    flags = []
    if status & EpisodeStatus.SUCCESS:               flags.append("SUCCESS")
    if status & EpisodeStatus.FAILURE_HUMAN_CONTACT: flags.append("HUMAN_CONTACT")
    if status & EpisodeStatus.FAILURE_OBJECT_DROP:   flags.append("OBJECT_DROP")
    if status & EpisodeStatus.FAILURE_TIMEOUT:       flags.append("TIMEOUT")
    closed = f"close@{close_step}" if close_step >= 0 else "never closed"
    print(f"  result: {' | '.join(flags) if flags else 'no-success'}  "
          f"({closed}, grasped={grasped}, ee→ycb={dist:.3f} m)")
    for d in debug_ids:
        pybullet.removeUserDebugItem(d)
    return status, dist, grasped, close_step


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir",  required=True, help="output/bc_runs/<name>")
    p.add_argument("--cfg-file", required=True, help="simulator config (e.g. examples/pretrain.yaml)")
    p.add_argument("--scene",    type=int, default=0, help="scene index to roll out")
    p.add_argument("--max-steps", type=int, default=30, help="max policy steps")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--no-render", action="store_true",
                   help="run headless (no GUI, no point overlay)")
    p.add_argument("--benchmark", action="store_true",
                   help="headless eval over many scenes: prints success rate + mean ee→ycb")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="benchmark: number of scenes to roll out (default: all)")
    p.add_argument("--show-goal-grasp", action="store_true",
                   help="overlay the gripper pose OMG planned to reach for the "
                        "scene (green wireframe) — the grasp the expert demos "
                        "aimed at. Ignored in --no-render / --benchmark.")
    p.add_argument("--show-grasp-set", action="store_true",
                   help="also draw the full filtered OMG grasp candidate set "
                        "(faint grey); implies --show-goal-grasp.")
    p.add_argument("--freeze-partial-pointcloud", action="store_true",
                   help="experimental: freeze the cloud to an early frame and hold "
                        "it for the whole episode. MUST match the setting the "
                        "dataset was collected/trained with.")
    p.add_argument("--freeze-at-step", type=int, default=None,
                   help="which policy step's cloud to freeze and hold "
                        "(default: config value, 0 = the very first step)")
    p.add_argument("--egl", action="store_true",
                   help="headless only: use the EGL GPU renderer for the offscreen "
                        "hand camera (else DIRECT-mode software fallback). The point "
                        "cloud is renderer-dependent — MUST match how the dataset was "
                        "collected, or the policy sees a different distribution.")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    cfg = get_cfg()
    cfg_from_file(filename=args.cfg_file, dict=cfg, merge_to_cn_dict=True)
    if args.freeze_partial_pointcloud:
        cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD = True
    if args.freeze_at_step is not None:
        cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD_AT_STEP = args.freeze_at_step
    render = not args.no_render
    cfg.SIM.RENDER = render
    # The offscreen hand camera's renderer changes the point cloud (GUI hardware
    # GL vs DIRECT software vs EGL), which can flip the borderline gripper-close
    # decision. Keep it consistent with how the dataset was COLLECTED. --egl opts
    # into the EGL GPU renderer when headless (else DIRECT software fallback).
    if args.egl and not render:
        cfg.SIM.BULLET.USE_EGL = True

    env = GraspBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    point_listener = PointListener(cfg, seed=0)
    model = load_policy(run_dir, args.device)

    # Scripted grasp-and-back, reused from the paper's policy (close + carry to
    # GOAL_CENTER). Standoff 0 → close at the policy's pose, not 8 cm beyond it.
    gb_policy = HandoverSim2RealPolicy(cfg, None, None, 0.0)
    gb_policy._standoff_offset = np.zeros(3, dtype=np.float32)

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)
    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

    from scipy.spatial.transform import Rotation as Rot
    panda_base_pos = np.array(cfg.ENV.PANDA_BASE_POSITION)
    R_base = Rot.from_quat(np.array(cfg.ENV.PANDA_BASE_ORIENTATION)).as_matrix()

    scene = args.scene
    want_goal = args.show_goal_grasp or args.show_grasp_set
    goal_marker_ids = []  # shared across re-rolls so old markers get cleared

    def do_rollout(s, draw=render):
        return rollout(env, model, point_listener, gb_policy, s, args.device,
                       panda_base_inv_tf, steps_action_repeat, args.max_steps,
                       R_base, panda_base_pos, draw=draw,
                       show_goal_grasp=(want_goal and draw),
                       show_grasp_set=args.show_grasp_set,
                       omg_steps=cfg.RL_MAX_STEP,
                       goal_marker_ids=goal_marker_ids)

    # Headless benchmark: roll out many scenes, report success / grasp / dist.
    if args.benchmark:
        n = min(args.num_scenes or env.num_scenes, env.num_scenes)
        succ, grasped_n, closed_n, dists = 0, 0, 0, []
        for s in range(n):
            status, dist, grasped, close_step = do_rollout(s, draw=False)
            if status & EpisodeStatus.SUCCESS:
                succ += 1
            if grasped:
                grasped_n += 1
            if close_step >= 0:
                closed_n += 1
            dists.append(dist)
        dists = np.array(dists, dtype=np.float32)
        print("\n==== benchmark ====")
        print(f"policy         : {run_dir}")
        print(f"scenes         : {n}")
        print(f"success rate   : {succ}/{n} = {succ / n:.1%}  (grasped + carried to goal)")
        print(f"grasp rate     : {grasped_n}/{n} = {grasped_n / n:.1%}  (both fingers gripping object)")
        print(f"commanded close: {closed_n}/{n}")
        print(f"ee→ycb         : mean {np.nanmean(dists):.3f} m  median {np.nanmedian(dists):.3f} m  "
              f"min {np.nanmin(dists):.3f}  max {np.nanmax(dists):.3f}")
        return

    do_rollout(scene)

    if not render:
        return

    print("\nIn the PyBullet window:  R = re-roll,  N = next scene,  Q = quit.")
    R_KEY, N_KEY, Q_KEY = ord('r'), ord('n'), ord('q')
    try:
        while True:
            keys = pybullet.getKeyboardEvents()
            if R_KEY in keys and keys[R_KEY] & pybullet.KEY_WAS_TRIGGERED:
                do_rollout(scene)
                print("R = re-roll,  N = next scene,  Q = quit.")
            if N_KEY in keys and keys[N_KEY] & pybullet.KEY_WAS_TRIGGERED:
                scene = (scene + 1) % env.num_scenes
                do_rollout(scene)
                print("R = re-roll,  N = next scene,  Q = quit.")
            if Q_KEY in keys and keys[Q_KEY] & pybullet.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
