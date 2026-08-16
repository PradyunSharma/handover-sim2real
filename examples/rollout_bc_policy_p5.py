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
--show-pred-grasp overlays what the run-13 auxiliary head THINKS the grasp is
(magenta), redrawn each step. With --show-goal-grasp you get belief vs truth in
one view.

In the PyBullet window:  R = re-roll the same scene,  N = next scene,  Q = quit.
"""

from __future__ import annotations

import argparse
import json
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
from handover_sim2real.policy import PointListener
from handover_sim2real.eval_wrapper import GraspBenchmarkWrapper
from handover_sim2real.utils import add_sys_path_from_env
# Module scope, not inside load_policy: rollout() below needs it every step.
from handover_sim2real.bc5.dataset import goal_cond_from_state

add_sys_path_from_env("GADDPG_DIR")
from core.utils import unpack_action, tf_quat, unpack_pose, se3_transform_pc  # noqa: E402
from experiments.config import cfg_from_file  # noqa: E402

# Reuse the exact observation -> (robot_state, point_cloud) builders used at
# collection time so the policy sees the same representation it trained on.
from collect_bc_dataset import _robot_state, _point_cloud  # noqa: E402


# ── model loading ────────────────────────────────────────────────────────────

def scenes_from_run(path, num_scenes: int):
    """The scene pool a Phase-4 run actually collected and trained on.

    Mirrors handover_sim2real.dagger5.setup.build_phase5_context exactly: the
    grasp pin table's key set is the FIRST filter (scenes OMG can plan for at
    all — ~13% of the s0 train split has no reachable goal set), and
    SIM.exclude_scenes is the second (scenes whose expert demonstration ended in
    a benchmark failure). Rolling out over range(num_scenes) instead scores the
    policy on both groups, which it never saw in training.
    """
    p = Path(path)
    cfg_path = p / "config.yaml" if p.is_dir() else p
    if not cfg_path.exists():
        raise SystemExit(f"--scenes-from-run: no config at {cfg_path}")
    with cfg_path.open() as f:
        sim = (yaml.safe_load(f) or {}).get("SIM", {}) or {}

    pin = sim.get("grasp_pin_table")
    if pin and Path(pin).exists():
        with open(pin) as f:
            raw = json.load(f)
        usable = {int(k) for k, v in raw.items() if k != "_meta" and v is not None}
    else:
        if pin:
            print(f"WARNING: pin table {pin} missing — pool NOT filtered by it")
        usable = set(range(num_scenes))

    excl = sim.get("exclude_scenes")
    if excl and Path(excl).exists():
        with open(excl) as f:
            usable -= {int(s) for s in json.load(f)}
    elif excl:
        print(f"WARNING: exclude_scenes {excl} missing — failed scenes KEPT")

    return sorted(s for s in usable if 0 <= s < num_scenes)


def resolve_scenes(args, num_scenes: int):
    """Explicit scene list to roll out, or None to mean range(num_scenes)."""
    if args.scenes and args.scenes_from_run:
        raise SystemExit("pass --scenes or --scenes-from-run, not both")

    if args.scenes_from_run:
        ids = scenes_from_run(args.scenes_from_run, num_scenes)
        print(f"Scene pool from {args.scenes_from_run}: {len(ids)} scenes")
    elif args.scenes:
        pth = Path(args.scenes)
        if pth.exists():
            with pth.open() as f:
                ids = sorted({int(s) for s in json.load(f)})
        else:
            ids = sorted({int(s) for s in args.scenes.replace(" ", "").split(",") if s})
        ids = [s for s in ids if 0 <= s < num_scenes]
        print(f"Scene pool: {len(ids)} scenes")
    else:
        return None

    if not ids:
        raise SystemExit("scene selection is empty")
    return ids


def load_policy(run_dir: Path, device: str):
    from handover_sim2real.bc5 import BCPolicy, Normalizer

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
        drop_joint_state   = bool(m.get("drop_joint_state", False)),
        joint_state_dim    = int(m.get("joint_state_dim", 18)),
        freeze_pc          = bool(m.get("freeze_pc", False)),
        # Run 13's auxiliary goal-grasp head. Must be built when the config says
        # so or the checkpoint will not strict-load — and it is what
        # --show-pred-grasp reads.
        aux_head           = bool(m.get("aux_head", False)),
        aux_dim            = int(m.get("aux_dim", 7)),
        aux_hidden         = tuple(m.get("aux_hidden", (256, 256))),
        # Phase 5's goal-grasp conditioning. Unlike the aux head this changes
        # policy_head's in_dim, so getting it wrong is a shape error at load
        # rather than a silent behavioural difference.
        grasp_cond         = bool(m.get("grasp_cond", False)),
        grasp_cond_dim     = int(m.get("grasp_cond_dim", 9)),
        grasp_hidden       = int(m.get("grasp_hidden", 128)),
        grasp_feat_dim     = int(m.get("grasp_feat_dim", 128)),
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

def rollout(env, model, point_listener, scene_idx, device,
            panda_base_inv_tf, steps_action_repeat, max_steps,
            R_base, panda_base_pos, draw=True,
            show_goal_grasp=False, show_grasp_set=False,
            omg_steps=None, goal_marker_ids=None, pin_table=None,
            hold_steps=3, dwell_steps=20, show_pred_grasp=False,
            grasp_idx=0):
    obs = env.reset(idx=scene_idx)

    # Phase 5: which of the scene's pinned grasps this roll is commanded to
    # reach. It is BOTH the policy's conditioning and what the green overlay
    # draws, so watching slot 0 and slot 3 on the same scene is the eyeball
    # version of the cond_track diagnostic. Requires the pin table — without one
    # there is no such thing as "grasp 3" and a conditioned policy has nothing to
    # be conditioned on.
    cond_goal = None if pin_table is None else pin_table.pose(scene_idx, grasp_idx)
    if getattr(model, "grasp_cond", False) and cond_goal is None:
        raise SystemExit(
            f"this checkpoint is grasp-conditioned (MODEL.grasp_cond: true) but "
            f"no pinned pose is available for scene {scene_idx} slot {grasp_idx}. "
            f"Pass --grasp-pin-table (the run's SIM.grasp_pin_table).")

    # Optionally overlay the grasp the policy is supposed to be aiming at. Drawn
    # once (the object is static) and left up for the whole roll so you can watch
    # the gripper close the gap; replaced on the next roll via goal_marker_ids.
    #
    # PASS THE PIN TABLE the policy was trained with, or this overlay lies. OMG
    # re-decides its goal on every plan, and on a Phase-4 dataset the pinned grasp
    # differs from OMG's free pick on 63% of train scenes — so without the table
    # the green gripper lands somewhere the policy was never taught to go, and a
    # correct rollout looks like a miss.
    if show_goal_grasp:
        if goal_marker_ids is None:
            goal_marker_ids = []
        for d in goal_marker_ids:
            pybullet.removeUserDebugItem(d)
        goal_marker_ids.clear()

        env.run_omg_planner(omg_steps or max_steps, scene_idx)  # plans, no sim step
        if pin_table is not None and pin_table.apply(env, scene_idx, grasp_idx):
            # Pruning the goal set renumbers it, so replan to re-resolve the goal
            # index against the pinned grasp. reset_scene=False keeps the scene.
            env.run_omg_planner(omg_steps or max_steps, scene_idx, reset_scene=False)
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

    # AFTER the overlay, never before: reset() reseeds the GLOBAL numpy stream
    # (policy.PointListener.reset -> np.random.seed) and the per-step point-cloud
    # subsample draws from that same global stream (np.random.choice of 1024
    # points). OMG's planner also draws from it (omg/planner.py goal selection and
    # sampling), so planning between the reseed and step 0 shifts the stream and
    # the policy sees a DIFFERENT 1024-point sample — same scene, same physics,
    # divergent rollout. Reseeding here makes --show-goal-grasp purely visual.
    point_listener.reset()

    debug_ids = []
    pred_marker_ids: list = []
    status = 0
    done = False
    info = {}
    dist = float("nan")
    close_step = -1
    grasped = False
    success = False
    reason = None

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
        # Recomputed every step from the RAW rs, exactly as BCDataset does at
        # training time — the world grasp is fixed but the EE moves, so the
        # EE-frame residual the policy is conditioned on changes.
        cond_t = None
        if getattr(model, "grasp_cond", False):
            cond_t = torch.from_numpy(
                goal_cond_from_state(rs, cond_goal)).float().unsqueeze(0).to(device)
        action = model.predict(pc_t, rs_t, cond_t)[0].cpu().numpy()  # [7] real units
        prev_act6d = action[:6].astype(np.float32)                  # for next step's robot_state

        # ----- run-13 auxiliary head: where does the policy THINK the grasp is --
        # Redrawn every step: unlike the pinned overlay (static, the scene does
        # not move) this is a live belief and watching it settle — or not — is the
        # point. Costs one extra forward pass, render mode only.
        #
        # Purely diagnostic: `action` above is what drives the arm, and it came
        # from predict() before this ran. Nothing here can change the rollout.
        if show_pred_grasp and getattr(model, "aux_head", None) is not None:
            with torch.no_grad():
                _, goal = model.forward_aux(
                    pc_t, model.normalizer.normalize_state(rs_t)
                    if model.normalizer is not None else rs_t)
            g = goal[0].cpu().numpy()                    # [quat_wxyz(4) ‖ trans(3)]
            # EE frame -> world. The head predicts the grasp RELATIVE TO THE
            # CURRENT EE, which is exactly how the training target was built
            # (bc/dataset.py::goal_target_from_state), so composing with the live
            # EE pose is the inverse of that construction.
            link = obs["panda_link_ind_hand"]
            ee_mat = unpack_pose(np.hstack([
                obs["panda_body"].link_state[0, link, 0:3].numpy(),
                tf_quat(obs["panda_body"].link_state[0, link, 3:7].numpy())]))
            T_rel = unpack_pose(np.hstack([g[4:7], g[:4]]))   # pos-first packing
            for d in pred_marker_ids:
                pybullet.removeUserDebugItem(d)
            pred_marker_ids.clear()
            draw_gripper(ee_mat @ T_rel, [1.0, 0.0, 1.0], pred_marker_ids, 2.0)

        ee_pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
        ycb_pos = env.ycb.bodies[env.ycb.ids[0]].link_state[0, 6, 0:3].numpy()
        dist = np.linalg.norm(ee_pos - ycb_pos)

        # First time the policy commands a grasp (gripper close): close IN PLACE
        # and hold. No carry to GOAL_CENTER — the episode ends at the committed
        # close, exactly as it does during collection and in the Phase-4
        # evaluator, so the number here means the same thing as the training
        # one. `grasp_held_after_hold` is the same function rl/rollout_worker.py
        # scores with, so this cannot drift from the `stable_grasp` criterion.
        if action[6] < 0.5:
            # Lazy: rollout_worker imports action_to_target_joint from THIS
            # module, so a top-level import here would be circular.
            from handover_sim2real.rl.rollout_worker import grasp_held_after_hold
            close_step = step
            print(f"  step {step:3d}  GRASP commanded  ee→ycb={dist:.3f} m  "
                  f"→ close + hold {hold_steps}")
            held, obs = grasp_held_after_hold(
                env, obs, steps_action_repeat, hold_steps)
            grasped = bool(env.grasped_active())
            success = held
            reason = ("GRASP_OK" if held else
                      "DROP" if bool(getattr(env, "_dropped", False)) else
                      "NO_RELEASE" if not bool(env.ycb.released) else
                      "GRASP_MISS")

            # VIEWING ONLY, render mode only. The hold above is 3 policy-steps —
            # a blink on screen — so keep the shut gripper live a while longer so
            # the grasp is actually watchable. Nothing here touches success /
            # grasped / reason: those are already decided, and this must not
            # become a lift test by the back door.
            if draw and dwell_steps > 0:
                shut = np.concatenate([np.zeros(6, np.float32), [0.0]]).astype(np.float32)
                for _ in range(int(dwell_steps)):
                    jp = action_to_target_joint(shut, obs)
                    for _ in range(steps_action_repeat):
                        obs, _, d_, _ = env.step(jp)
                        if d_:
                            break
                    time.sleep(0.03)
                    if d_:
                        break
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

    # Report outcome. `reason` is set at the close; otherwise the episode ran out
    # of steps or the env ended it, and the status bits say which.
    if reason is None:
        if status & EpisodeStatus.FAILURE_HUMAN_CONTACT:
            reason = "HUMAN_CONTACT"
        elif status & EpisodeStatus.FAILURE_OBJECT_DROP:
            reason = "DROP"
        else:
            reason = "TIMEOUT"
    closed = f"close@{close_step}" if close_step >= 0 else "never closed"
    print(f"  result: {'SUCCESS' if success else 'FAIL'} [{reason}]  "
          f"({closed}, grasped={grasped}, ee→ycb={dist:.3f} m)")
    for d in debug_ids:
        pybullet.removeUserDebugItem(d)
    return success, reason, dist, grasped, close_step


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir",  required=True, help="output/bc_runs/<name>")
    p.add_argument("--cfg-file", required=True, help="simulator config (e.g. examples/pretrain.yaml)")
    p.add_argument("--scene",    type=int, default=0, help="scene index to roll out")
    p.add_argument("--max-steps", type=int, default=30, help="max policy steps")
    p.add_argument("--hold-steps", type=int, default=3,
                   help="stable_grasp: policy-steps to hold the gripper shut after "
                        "the close before checking the object is secured. Match the "
                        "run's EVAL.hold_steps (default 3).")
    p.add_argument("--dwell-steps", type=int, default=20,
                   help="GUI only: extra policy-steps to keep the gripper shut on "
                        "screen after the grasp has already been scored, so the "
                        "close is watchable. 0 disables. Never affects the result, "
                        "and is skipped entirely under --no-render/--benchmark.")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--no-render", action="store_true",
                   help="run headless (no GUI, no point overlay)")
    p.add_argument("--benchmark", action="store_true",
                   help="headless eval over many scenes: prints success rate + mean ee→ycb")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="benchmark: number of scenes to roll out (default: all)")
    p.add_argument("--scenes", default=None,
                   help="restrict to an explicit scene set instead of range(n): "
                        "either '3,7,12' or a path to a JSON list of ints. "
                        "--num-scenes then caps how many of THESE are rolled out, "
                        "and 'N' in the GUI steps through them.")
    p.add_argument("--scenes-from-run", default=None,
                   help="restrict to the scenes a Phase-4 run actually trained on: "
                        "pass its run dir (or config.yaml) and the pool is rebuilt "
                        "the way handover_sim2real.dagger5.setup does — the grasp pin "
                        "table's keys minus SIM.exclude_scenes. Rolling out over "
                        "range(n) instead includes scenes OMG cannot plan for and "
                        "scenes whose expert demo failed, both of which the policy "
                        "never saw and neither of which it can be expected to solve.")
    p.add_argument("--show-goal-grasp", action="store_true",
                   help="overlay the gripper pose OMG planned to reach for the "
                        "scene (green wireframe) — the grasp the expert demos "
                        "aimed at. Ignored in --no-render / --benchmark.")
    p.add_argument("--show-grasp-set", action="store_true",
                   help="also draw the full filtered OMG grasp candidate set "
                        "(faint grey); implies --show-goal-grasp.")
    p.add_argument("--grasp-idx", type=int, default=0,
                   help="which pinned grasp to command (Phase 5). 0 is OMG's own "
                        "pick and matches a Phase-4 rollout; rolling the same "
                        "scene at 0 and 3 is the visual form of cond_track.")
    p.add_argument("--all-grasps", action="store_true",
                   help="--benchmark only: sweep every pinned slot of every "
                        "scene, i.e. the conditional table retry@k comes from")
    p.add_argument("--show-pred-grasp", action="store_true",
                   help="overlay the grasp the policy's AUXILIARY HEAD predicts "
                        "(magenta wireframe), redrawn every step so you can watch "
                        "the belief settle. Needs a checkpoint trained with "
                        "MODEL.aux_head (run 13 on); silently inert otherwise. "
                        "Pair with --show-goal-grasp to see prediction (magenta) "
                        "against ground truth (green): the gap between them is "
                        "what the aux head is being asked to close, and it is a "
                        "different quantity from the gap between the gripper and "
                        "the green pose. Ignored in --no-render / --benchmark.")
    p.add_argument("--grasp-pin-table", default=None,
                   help="per-scene committed grasp(s) (examples/build_grasp_pin_table_multi.py). "
                        "Pass the table the policy was TRAINED with, or the green "
                        "goal gripper is drawn at OMG's free pick instead — which "
                        "differs from the pinned grasp on 63%% of train scenes, "
                        "making a correct rollout look like a miss.")
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

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)
    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

    from scipy.spatial.transform import Rotation as Rot
    panda_base_pos = np.array(cfg.ENV.PANDA_BASE_POSITION)
    R_base = Rot.from_quat(np.array(cfg.ENV.PANDA_BASE_ORIENTATION)).as_matrix()

    scene_ids = resolve_scenes(args, env.num_scenes)
    scene = args.scene
    if scene_ids is not None and scene not in scene_ids:
        scene = scene_ids[0]
        print(f"scene {args.scene} is not in the pool — starting at {scene}")
    want_goal = args.show_goal_grasp or args.show_grasp_set
    goal_marker_ids = []  # shared across re-rolls so old markers get cleared

    pin_table = None
    if args.grasp_pin_table:
        from handover_sim2real.dagger5 import load_grasp_pin_table
        pin_table = load_grasp_pin_table(args.grasp_pin_table)
        print(f"Grasp pin table: {args.grasp_pin_table}")

    def do_rollout(s, draw=render, g=None):
        return rollout(env, model, point_listener, s, args.device,
                       panda_base_inv_tf, steps_action_repeat, args.max_steps,
                       R_base, panda_base_pos, draw=draw,
                       show_goal_grasp=(want_goal and draw),
                       show_grasp_set=args.show_grasp_set,
                       omg_steps=cfg.RL_MAX_STEP,
                       goal_marker_ids=goal_marker_ids,
                       pin_table=pin_table,
                       hold_steps=args.hold_steps,
                       dwell_steps=args.dwell_steps,
                       show_pred_grasp=(args.show_pred_grasp and draw),
                       grasp_idx=(args.grasp_idx if g is None else g))

    # Headless benchmark: roll out many scenes, report success / grasp / dist.
    if args.benchmark:
        ids = scene_ids if scene_ids is not None else list(range(env.num_scenes))
        if args.num_scenes:
            ids = ids[:args.num_scenes]
        succ, grasped_n, closed_n, dists = 0, 0, 0, []
        reasons = {}
        # --all-grasps sweeps every pinned slot of every scene, which is the
        # conditional table `succ_g*` and `retry_at_k` are computed from. Without
        # it only --grasp-idx is rolled, so a Phase-5 benchmark reports one slot.
        n_slots = (pin_table.num_grasps if (args.all_grasps and pin_table) else 1)
        jobs = ([(s, g) for s in ids for g in range(n_slots)] if n_slots > 1
                else [(s, args.grasp_idx) for s in ids])
        n = len(jobs)
        for s, g in jobs:
            success, reason, dist, grasped, close_step = do_rollout(s, draw=False, g=g)
            if success:
                succ += 1
            if grasped:
                grasped_n += 1
            if close_step >= 0:
                closed_n += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            dists.append(dist)
        dists = np.array(dists, dtype=np.float32)
        print("\n==== benchmark ====")
        print(f"policy         : {run_dir}")
        print(f"scenes         : {n}")
        print(f"success rate   : {succ}/{n} = {succ / n:.1%}  "
              f"(stable_grasp: closed, held {args.hold_steps}, object secured)")
        print(f"grasp rate     : {grasped_n}/{n} = {grasped_n / n:.1%}  (both fingers gripping object)")
        print(f"commanded close: {closed_n}/{n}")
        print(f"reasons        : {dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}")
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
                if scene_ids is None:
                    scene = (scene + 1) % env.num_scenes
                else:
                    i = scene_ids.index(scene) if scene in scene_ids else -1
                    scene = scene_ids[(i + 1) % len(scene_ids)]
                do_rollout(scene)
                print("R = re-roll,  N = next scene,  Q = quit.")
            if Q_KEY in keys and keys[Q_KEY] & pybullet.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
