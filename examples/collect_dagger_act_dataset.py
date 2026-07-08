"""
Collect a DAgger dataset for the Phase-2 ACT policy: roll out the trained ACT
policy, label visited states with OMG.

The Phase-2 analog of collect_dagger_dataset.py. The DAgger idea is unchanged —
drive the sim with the *policy's own* actions and label every visited state with
the OMG action the expert would take there — but the policy here is the temporal
/ chunking ACTPolicy, so action selection uses an observation **history ring
buffer** of length T, a predicted **chunk** of k actions, and the run's EXEC
strategy (temporal ensembling by default), exactly as in rollout_act_policy.

What gets RECORDED is still one single frame per step in the *same HDF5 schema*
as collect_bc_dataset.py (windowing into histories happens later, at training
time), so train_act.py can aggregate it via `--dagger-h5` with no conversion:

    point_clouds   float32 [T, 1024, 5]   xyz + ycb_flag + hand_flag (EE frame)
    robot_states   float32 [T, 32]        joint_pos(9)+joint_vel(9)+ee(7)+grip(1)+prev_act(6)
    expert_actions float32 [T, 7]         Δpos(3)+Δeuler(3)+gripper_cmd(1)  (OMG label)

Only the *executed* action (which states get visited) comes from the ACT policy;
the recorded label is the OMG single-step expert delta (gripper OPEN). By default
DAgger labels the *approach to the pre-grasp standoff* only:
  --dynamic-horizon    : OMG re-plan length scales with the EE->standoff distance
                         (~--ee-step m/step), so labels stay at the demo step
                         scale instead of late-step big jumps.
  --drop-past-standoff : recording stops at the standoff plane; the final reach +
                         gripper close come from the demonstrations.
use_standoff stays ON (same grasp as the demos). --no-drop-past-standoff (+
--close-pos-thresh) restores the grasp-reached close-trigger; --static-horizon
restores the old step-shrinking horizon.

Usage:
    python examples/collect_dagger_act_dataset.py \\
        --run-dir  output/bc_runs/act_run1 \\
        --cfg-file examples/pretrain.yaml \\
        --output   output/bc_dataset/dagger_act_iter1.h5 \\
        [--split train] [--num-episodes 50] [--max-steps 25] \\
        [--beta 0.0] [--query-every 1] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling example modules

import gym
import h5py
import numpy as np
import pybullet
import torch
import yaml

import handover            # noqa: F401  registers envs
import handover_sim2real   # noqa: F401  registers envs

from handover.benchmark_wrapper import HandoverBenchmarkWrapper
from handover_sim2real.config import get_cfg
from handover_sim2real.policy import PointListener
from handover_sim2real.bc import TemporalEnsembler
from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("GADDPG_DIR")
from experiments.config import cfg_from_file  # noqa: E402

# Reuse the exact state builders + schema constants, the IK helper, and the ACT
# policy loader + history stacker — so DAgger states are byte-identical to what
# training/inference see.
from collect_bc_dataset import (  # noqa: E402
    _point_cloud, _robot_state, ee_grasp_pose_error,
    standoff_plane, ee_at_or_past_standoff, dynamic_replan_horizon,
    ROBOT_STATE_DIM, ACTION_DIM, NUM_PTS, PC_CHANNELS,
)
from rollout_bc_policy import action_to_target_joint  # noqa: E402
from rollout_act_policy import load_policy, stack_history  # noqa: E402


# ── one DAgger episode (ACT policy) ────────────────────────────────────────────

def collect_dagger_act_episode(env, model, point_listener, cfg, scene_idx, device,
                               panda_base_inv_tf, steps_action_repeat, max_steps,
                               beta, query_every, rng, T, k, exec_cfg,
                               close_pos_thresh, close_rot_thresh,
                               drop_past_standoff, dynamic_horizon,
                               ee_step, reach_tail, min_free, max_horizon):
    """Roll out the ACT policy on `scene_idx`, labelling visited states with OMG.

    Labels the *approach to the pre-grasp standoff* only (gripper OPEN). With
    `dynamic_horizon`, the OMG re-plan length is chosen from the EE->standoff
    distance so each first-step delta is ~`ee_step` m (no late-step big jumps).
    With `drop_past_standoff`, recording stops at the standoff plane — the final
    reach + close come from the demonstrations (no backward labels). use_standoff
    stays ON so DAgger aims at the same grasp the demos used. The close-trigger
    (`close_pos_thresh`) only applies when `drop_past_standoff` is off.

    Returns (episode_dict | None, n_omg_fail, n_close, reached_standoff). None
    when the scene is unusable (OMG fails on the first step, or nothing recorded).
    """
    obs = env.reset(idx=scene_idx)
    point_listener.reset()

    mode = exec_cfg.get("mode", "ensemble")
    ens = (TemporalEnsembler(chunk_len=k, m=float(exec_cfg.get("ensemble_m", 0.01)))
           if mode == "ensemble" else None)
    if ens is not None:
        ens.reset()
    pending: list[np.ndarray] = []   # open-loop action queue

    prev_act6d = np.zeros(6, dtype=np.float32)
    pc_buf, rs_buf = [], []
    point_clouds, robot_states, expert_actions = [], [], []
    n_omg_fail = 0
    n_close = 0
    reached_standoff = False
    standoff_pose = None          # cached 4x4 standoff (traj[-5]) from first plan
    plane_pt = plane_n = None     # standoff plane (point, normal toward grasp)
    done = False

    for step in range(max_steps):
        # State exactly as the policy / dataset sees it.
        pc = _point_cloud(obs, point_listener, panda_base_inv_tf)
        rs = _robot_state(obs, prev_act6d)
        pc_buf.append(pc); rs_buf.append(rs)

        # Stop once the policy reaches the standoff plane (final reach + close
        # are demonstrated offline; re-planning past it yields backward labels).
        if drop_past_standoff and plane_pt is not None and \
                ee_at_or_past_standoff(obs, plane_pt, plane_n):
            reached_standoff = True
            break

        # Query OMG (re-plan from the current joints) when we need a label.
        is_record_step = (step % query_every == 0)
        need_expert = is_record_step or beta > 0.0

        if dynamic_horizon and standoff_pose is not None:
            horizon = dynamic_replan_horizon(
                obs, standoff_pose, ee_step, reach_tail, min_free, max_horizon)
        elif dynamic_horizon:
            horizon = int(cfg.RL_MAX_STEP)
        else:
            horizon = max(cfg.RL_MAX_STEP - step, reach_tail)

        expert_plan = expert_delta = None
        if need_expert:
            expert_plan, _ = env.run_omg_planner(
                horizon, scene_idx, reset_scene=(step == 0)
            )
            if expert_plan is None:
                n_omg_fail += 1
                if step == 0:
                    return None, n_omg_fail, n_close, reached_standoff
            else:
                expert_delta = env.convert_target_joint_position_to_action(
                    expert_plan[0]
                )  # [6]
                if plane_pt is None:
                    g = env.get_omg_goal_grasp_pose()
                    s = env.get_omg_standoff_pose()
                    if g is not None and s is not None:
                        standoff_pose = s
                        plane_pt, plane_n = standoff_plane(g, s)

        # Record the labelled (state, expert action) pair.
        if is_record_step and expert_delta is not None:
            at_grasp = False
            if not drop_past_standoff and close_pos_thresh > 0.0:
                grasp_pose = env.get_omg_goal_grasp_pose()  # world 4x4 traj[-1]
                if grasp_pose is not None:
                    pos_err, rot_err = ee_grasp_pose_error(obs, grasp_pose)
                    at_grasp = (pos_err <= close_pos_thresh
                                and rot_err <= close_rot_thresh)

            point_clouds.append(pc)
            robot_states.append(rs)
            if at_grasp:
                expert_actions.append(
                    np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)  # CLOSE
                )
                n_close += 1
                break   # graspable state reached — end the episode
            expert_actions.append(
                np.concatenate([expert_delta, [1.0]]).astype(np.float32)
            )

        # ACT policy action: stack the last T frames -> chunk -> EXEC strategy.
        # Computed every step (advancing the ensembler / open-loop queue) so the
        # policy's bookkeeping stays consistent even on β-mixed expert steps.
        if mode == "ensemble" or not pending:
            pc_hist = stack_history(pc_buf, T)[None]
            rs_hist = stack_history(rs_buf, T)[None]
            pc_t = torch.from_numpy(pc_hist).float().to(device)
            rs_t = torch.from_numpy(rs_hist).float().to(device)
            chunk = model.predict(pc_t, rs_t)[0].cpu().numpy()   # [k,7], ch6=prob
        if mode == "ensemble":
            policy_action = ens.step(chunk)
        else:
            if not pending:
                pending = [a.copy() for a in chunk]
            policy_action = pending.pop(0)
            policy_action[6] = 1.0 if policy_action[6] >= 0.5 else 0.0

        # Choose the action to EXECUTE (β=0 → always policy = pure DAgger).
        use_expert = expert_plan is not None and rng.uniform() < beta
        if use_expert:
            target_jp  = expert_plan[0]
            exec_delta = expert_delta
        else:
            target_jp  = action_to_target_joint(policy_action, obs)
            exec_delta = policy_action[:6].astype(np.float32)

        prev_act6d = exec_delta.copy()

        for _ in range(steps_action_repeat):
            obs, _, done, _ = env.step(target_jp)
            if done:
                break
        if done:
            break

    if len(expert_actions) == 0:
        return None, n_omg_fail, n_close, reached_standoff

    return {
        "point_clouds":   np.array(point_clouds,   dtype=np.float32),
        "robot_states":   np.array(robot_states,   dtype=np.float32),
        "expert_actions": np.array(expert_actions, dtype=np.float32),
        "scene_idx":      scene_idx,
    }, n_omg_fail, n_close, reached_standoff


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir",      required=True, help="trained ACT policy run dir (e.g. output/bc_runs/act_run1)")
    p.add_argument("--cfg-file",     required=True, help="simulator config (e.g. examples/pretrain.yaml)")
    p.add_argument("--output",       required=True, help="output HDF5 path")
    p.add_argument("--split",        default="train", choices=["train", "val", "test"])
    p.add_argument("--num-episodes", type=int, default=None, help="max scenes to roll out")
    p.add_argument("--max-steps",    type=int, default=None, help="policy steps per scene (default: RL_MAX_STEP)")
    p.add_argument("--beta",         type=float, default=0.0,
                   help="prob. of executing the EXPERT action instead of the policy (0 = pure DAgger)")
    p.add_argument("--query-every",  type=int, default=1, help="record an OMG label every K steps")
    p.add_argument("--close-pos-thresh", type=float, default=0.02,
                   help="record a gripper-CLOSE label (and end the episode) once the "
                        "policy's EE is within this many metres of the OMG grasp pose. "
                        "0 disables (gripper always labelled open, as before).")
    p.add_argument("--close-rot-thresh", type=float, default=0.4,
                   help="orientation tolerance (rad) paired with --close-pos-thresh "
                        "for the grasp-reached close trigger (~0.4 rad ≈ 23°). Only "
                        "applies when --no-drop-past-standoff is set.")
    # --- approach-only DAgger: distance-proportional horizon + standoff drop ---
    p.add_argument("--drop-past-standoff", dest="drop_past_standoff",
                   action="store_true", default=True,
                   help="stop recording once the policy reaches the pre-grasp "
                        "standoff plane; the final reach + close come from the "
                        "demonstrations (default on, kills backward labels).")
    p.add_argument("--no-drop-past-standoff", dest="drop_past_standoff",
                   action="store_false",
                   help="disable the standoff drop (record all the way in).")
    p.add_argument("--dynamic-horizon", dest="dynamic_horizon",
                   action="store_true", default=True,
                   help="choose the OMG re-plan length from the EE->standoff "
                        "distance so each first-step delta is ~--ee-step m "
                        "(default on, avoids late-step big-jump labels).")
    p.add_argument("--static-horizon", dest="dynamic_horizon",
                   action="store_false",
                   help="use the old step-shrinking horizon max(RL_MAX_STEP-step,5).")
    p.add_argument("--ee-step", type=float, default=0.04,
                   help="target per-step EE displacement (m) for --dynamic-horizon "
                        "(default 0.04 ≈ the demonstrations' median step).")
    p.add_argument("--reach-tail", type=int, default=5,
                   help="OMG reach_tail_length added back to the free-portion horizon.")
    p.add_argument("--min-free", type=int, default=3,
                   help="minimum free (approach) steps in the dynamic horizon.")
    p.add_argument("--max-horizon", type=int, default=40,
                   help="cap on the dynamic OMG horizon (num_steps).")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--render",       action="store_true", help="show the PyBullet GUI (default headless)")
    p.add_argument("--freeze-partial-pointcloud", action="store_true",
                   help="experimental: freeze the cloud to an early frame and hold "
                        "it for the whole episode. MUST match the setting the policy "
                        "was trained with.")
    p.add_argument("--freeze-at-step", type=int, default=None,
                   help="which policy step's cloud to freeze and hold "
                        "(default: config value, 0 = the very first step)")
    p.add_argument("--egl", action="store_true",
                   help="headless only: EGL GPU renderer (NVIDIA) for the offscreen "
                        "hand camera instead of the CPU software renderer — MUST match "
                        "the renderer the policy's training data was built with.")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = get_cfg()
    cfg_from_file(filename=args.cfg_file, dict=cfg, merge_to_cn_dict=True)
    cfg.BENCHMARK.SPLIT = args.split
    cfg.SIM.RENDER = bool(args.render)
    if args.egl and not args.render:
        cfg.SIM.BULLET.USE_EGL = True
    if args.freeze_partial_pointcloud:
        cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD = True
    if args.freeze_at_step is not None:
        cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD_AT_STEP = args.freeze_at_step

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    env            = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    point_listener = PointListener(cfg, seed=args.seed)
    model, exec_cfg = load_policy(Path(args.run_dir), args.device)

    # History/chunk sizes come from the trained run's config.
    with (Path(args.run_dir) / "config.yaml").open() as f:
        rcfg = yaml.safe_load(f)
    T = int(rcfg["MODEL"]["history_len"])
    k = int(rcfg["MODEL"]["chunk_len"])

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION
    )
    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)
    max_steps = args.max_steps if args.max_steps is not None else cfg.RL_MAX_STEP

    num_scenes = env.num_scenes
    if args.num_episodes is not None:
        num_scenes = min(num_scenes, args.num_episodes)

    print(f"DAgger (ACT) collection: {num_scenes} scenes  split={args.split}  "
          f"policy={args.run_dir}  T={T} k={k} exec={exec_cfg.get('mode')}  "
          f"beta={args.beta}  query_every={args.query_every}  max_steps={max_steps}")
    print(f"  drop_past_standoff={args.drop_past_standoff}  "
          f"dynamic_horizon={args.dynamic_horizon}  ee_step={args.ee_step}  "
          f"reach_tail={args.reach_tail}  min_free={args.min_free}  "
          f"max_horizon={args.max_horizon}")
    print(f"Output: {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    ep_idx = 0
    omg_skipped = 0
    omg_fail_total = 0
    close_total = 0
    reached_standoff_total = 0

    with h5py.File(args.output, "w") as f:
        f.attrs["split"]           = args.split
        f.attrs["seed"]            = args.seed
        f.attrs["num_pts"]         = NUM_PTS
        f.attrs["pc_channels"]     = PC_CHANNELS
        f.attrs["robot_state_dim"] = ROBOT_STATE_DIM
        f.attrs["action_dim"]      = ACTION_DIM
        f.attrs["action_format"]   = "delta_pos(3)+delta_euler(3)+gripper_cmd(1)"
        f.attrs["dagger"]          = True
        f.attrs["source_run_dir"]  = str(args.run_dir)
        f.attrs["policy_type"]     = "act"
        f.attrs["history_len"]     = T
        f.attrs["chunk_len"]       = k
        f.attrs["exec_mode"]       = exec_cfg.get("mode", "ensemble")
        f.attrs["beta"]            = args.beta
        f.attrs["freeze_partial_pointcloud"] = cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD
        f.attrs["freeze_at_step"]            = cfg.POLICY.FREEZE_PARTIAL_POINTCLOUD_AT_STEP
        f.attrs["query_every"]     = args.query_every
        f.attrs["max_steps"]       = max_steps
        f.attrs["close_pos_thresh"] = args.close_pos_thresh
        f.attrs["close_rot_thresh"] = args.close_rot_thresh
        f.attrs["drop_past_standoff"] = args.drop_past_standoff
        f.attrs["dynamic_horizon"]    = args.dynamic_horizon
        f.attrs["ee_step"]            = args.ee_step
        f.attrs["reach_tail"]         = args.reach_tail
        f.attrs["min_free"]           = args.min_free
        f.attrs["max_horizon"]        = args.max_horizon

        for scene_idx in range(num_scenes):
            episode, n_fail, n_close, reached = collect_dagger_act_episode(
                env, model, point_listener, cfg, scene_idx, args.device,
                panda_base_inv_tf, steps_action_repeat, max_steps,
                args.beta, args.query_every, rng, T, k, exec_cfg,
                args.close_pos_thresh, args.close_rot_thresh,
                args.drop_past_standoff, args.dynamic_horizon,
                args.ee_step, args.reach_tail, args.min_free, args.max_horizon,
            )
            omg_fail_total += n_fail
            close_total += n_close
            reached_standoff_total += int(reached)

            if episode is None:
                omg_skipped += 1
                print(f"  [{scene_idx+1:4d}/{num_scenes}] unusable (OMG setup failed) — skipped")
                continue

            T_ep = len(episode["expert_actions"])
            grp = f.create_group(f"episode_{ep_idx:05d}")
            grp.attrs["scene_idx"] = episode["scene_idx"]
            grp.attrs["num_steps"] = T_ep
            grp.create_dataset("point_clouds",   data=episode["point_clouds"],   compression="gzip")
            grp.create_dataset("robot_states",   data=episode["robot_states"],   compression="gzip")
            grp.create_dataset("expert_actions", data=episode["expert_actions"], compression="gzip")
            ep_idx += 1

            if (scene_idx + 1) % 10 == 0 or scene_idx == num_scenes - 1:
                print(f"  [{scene_idx+1:4d}/{num_scenes}]  episodes saved: {ep_idx}"
                      f"  steps this ep: {T_ep}")

        f.attrs["num_episodes"] = ep_idx
        f.attrs["num_close_labels"] = close_total
        f.attrs["num_reached_standoff"] = reached_standoff_total

    print(f"\nDone.")
    print(f"  Episodes saved        : {ep_idx}")
    print(f"  Reached standoff      : {reached_standoff_total} (episodes that hit the standoff plane)")
    print(f"  Close labels          : {close_total} (only when --no-drop-past-standoff)")
    print(f"  Scenes skipped        : {omg_skipped}")
    print(f"  OMG replan failures   : {omg_fail_total} (steps without a label)")
    print(f"  Dataset               : {args.output}")


if __name__ == "__main__":
    main()
