"""
Roll out a trained Phase-2 ACT policy in the simulator (closed loop).

Like examples/rollout_bc_policy.py, but for the temporal/chunking ACTPolicy:

    keep a ring buffer of the last T observations
        ─► stack to (pc_hist[1,T,N,5], rs_hist[1,T,32])
        ─► model.predict() ─► chunk[k,7]  (ch6 = gripper probability)
        ─► TemporalEnsembler (EXEC.mode=ensemble) or open-loop queue
        ─► Δee-pose ∘ current ee-pose ─► IK ─► target joint position
        ─► step the sim, repeat

Reuses the exact obs builders (_robot_state/_point_cloud from collect_bc_dataset)
and the IK / point-overlay helpers from rollout_bc_policy, so the policy sees the
same representation it trained on.

Usage:
    python examples/rollout_act_policy.py \
        --run-dir  output/bc_runs/act_full \
        --cfg-file examples/pretrain.yaml \
        --scene    0
    # headless eval over many scenes
    python examples/rollout_act_policy.py --run-dir ... --cfg-file ... --benchmark

In the PyBullet window:  R = re-roll,  N = next scene,  Q = quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling example modules

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
from handover_sim2real.bc import TemporalEnsembler
from handover_sim2real.utils import add_sys_path_from_env

add_sys_path_from_env("GADDPG_DIR")
from experiments.config import cfg_from_file  # noqa: E402

# Reuse the collection-time obs builders and the BC rollout's IK / overlay.
from collect_bc_dataset import _robot_state, _point_cloud           # noqa: E402
from rollout_bc_policy import action_to_target_joint, draw_pointcloud  # noqa: E402


# ── model loading ────────────────────────────────────────────────────────────

def load_policy(run_dir: Path, device: str):
    from handover_sim2real.bc import ACTPolicy, Normalizer

    with (run_dir / "config.yaml").open() as f:
        rcfg = yaml.safe_load(f)

    norm_path = run_dir / "normalization.npz"
    normalizer = Normalizer.load(norm_path) if norm_path.exists() else None
    if normalizer is None:
        print("WARNING: no normalization.npz — rollout will likely be garbage.")

    m, d = rcfg["MODEL"], rcfg["DATA"]
    model = ACTPolicy(
        pc_channels        = int(d["pc_channels"]),
        robot_state_dim    = int(d["robot_state_dim"]),
        action_dim         = int(d["action_dim"]),
        feature_dim        = int(m["feature_dim"]),
        robot_hidden       = int(m["robot_hidden"]),
        d_model            = int(m["d_model"]),
        n_heads            = int(m["n_heads"]),
        enc_layers         = int(m["enc_layers"]),
        dec_layers         = int(m["dec_layers"]),
        cvae_enc_layers    = int(m.get("cvae_enc_layers", 2)),
        dropout            = float(m.get("dropout", 0.1)),
        history_len        = int(m["history_len"]),
        chunk_len          = int(m["chunk_len"]),
        latent_dim         = int(m["latent_dim"]),
        use_cvae           = bool(m.get("use_cvae", True)),
        use_prev_act       = bool(m.get("use_prev_act", False)),
        pointnet_scale     = int(m["pointnet_scale"]),
        pointnet_radius    = float(m["pointnet_radius"]),
        pointnet_nclusters = int(m["pointnet_nclusters"]),
        normalizer         = normalizer,
    ).to(device)

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    exec_cfg = rcfg.get("EXEC", {"mode": "ensemble", "ensemble_m": 0.01})
    print(f"Loaded {ckpt_path} (epoch {payload.get('epoch', '?')})  "
          f"T={m['history_len']} k={m['chunk_len']} exec={exec_cfg.get('mode')}")
    return model, exec_cfg


# ── observation history ──────────────────────────────────────────────────────

def stack_history(buf: list[np.ndarray], T: int) -> np.ndarray:
    """Last T entries of buf, oldest→newest; left-pad by repeating the oldest."""
    recent = buf[-T:]
    if len(recent) < T:
        recent = [recent[0]] * (T - len(recent)) + recent
    return np.stack(recent, axis=0)


# ── rollout ──────────────────────────────────────────────────────────────────

def rollout(env, model, exec_cfg, point_listener, gb_policy, scene_idx, device,
            panda_base_inv_tf, steps_action_repeat, max_steps,
            R_base, panda_base_pos, T, k, draw=True):
    obs = env.reset(idx=scene_idx)
    point_listener.reset()

    mode = exec_cfg.get("mode", "ensemble")
    ensembler = (TemporalEnsembler(chunk_len=k, m=float(exec_cfg.get("ensemble_m", 0.01)))
                 if mode == "ensemble" else None)
    if ensembler is not None:
        ensembler.reset()
    pending: list[np.ndarray] = []   # open-loop action queue

    pc_buf: list[np.ndarray] = []
    rs_buf: list[np.ndarray] = []
    prev_act6d = np.zeros(6, dtype=np.float32)

    debug_ids = []
    status, done, info, dist = 0, False, {}, float("nan")
    close_step = -1
    grasped = False
    print(f"\n--- scene {scene_idx} ---")
    for step in range(max_steps):
        pc = _point_cloud(obs, point_listener, panda_base_inv_tf)   # [N,5] EE frame
        rs = _robot_state(obs, prev_act6d)                          # [32]
        pc_buf.append(pc); rs_buf.append(rs)

        # predict a fresh chunk every step except in open_loop (which drains the
        # previous chunk before re-predicting).
        if mode != "open_loop" or not pending:
            pc_hist = stack_history(pc_buf, T)[None]                # [1,T,N,5]
            rs_hist = stack_history(rs_buf, T)[None]                # [1,T,32]
            pc_t = torch.from_numpy(pc_hist).float().to(device)
            rs_t = torch.from_numpy(rs_hist).float().to(device)
            chunk = model.predict(pc_t, rs_t)[0].cpu().numpy()      # [k,7], ch6=prob

        if mode == "ensemble":
            action = ensembler.step(chunk)                          # [7], ch6={0,1}
        elif mode == "receding":
            # No ensembling: predict every step, execute only the first action
            # (fully reactive — isolates the temporal-ensembling effect).
            action = chunk[0].copy()
            action[6] = 1.0 if action[6] >= 0.5 else 0.0
        else:  # open_loop: execute the whole chunk before re-predicting
            if not pending:
                pending = [a.copy() for a in chunk]
            action = pending.pop(0)
            action[6] = 1.0 if action[6] >= 0.5 else 0.0            # threshold prob
        prev_act6d = action[:6].astype(np.float32)

        ee_pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
        ycb_pos = env.ycb.bodies[env.ycb.ids[0]].link_state[0, 6, 0:3].numpy()
        dist = np.linalg.norm(ee_pos - ycb_pos)

        # First time the policy commands a grasp (gripper close), hand off to the
        # scripted grasp-and-back: close the gripper in place (standoff 0) and
        # carry to GOAL_CENTER, then let the benchmark decide success.
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
            from core.utils import unpack_pose, tf_quat
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
    p.add_argument("--scene",    type=int, default=0)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--device",   default="cuda")
    p.add_argument("--no-render", action="store_true", help="headless (no GUI/overlay)")
    p.add_argument("--benchmark", action="store_true",
                   help="headless eval over many scenes: success rate + mean ee→ycb")
    p.add_argument("--num-scenes", type=int, default=None)
    p.add_argument("--exec-mode", choices=["ensemble", "open_loop", "receding"], default=None,
                   help="override EXEC.mode from the run config: ensemble=temporal "
                        "ensembling (default); open_loop=execute the full k-chunk before "
                        "re-predicting; receding=predict every step but execute only the "
                        "first action (NO ensembling, fully reactive)")
    p.add_argument("--ensemble-m", type=float, default=None,
                   help="override EXEC.ensemble_m (temporal-ensembling rate)")
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
    model, exec_cfg = load_policy(run_dir, args.device)

    # CLI overrides for the execution strategy (don't mutate the loaded dict).
    exec_cfg = dict(exec_cfg)
    if args.exec_mode is not None:
        exec_cfg["mode"] = args.exec_mode
    if args.ensemble_m is not None:
        exec_cfg["ensemble_m"] = args.ensemble_m
    print(f"EXEC (effective): mode={exec_cfg.get('mode')}  ensemble_m={exec_cfg.get('ensemble_m')}")

    with (run_dir / "config.yaml").open() as f:
        rcfg = yaml.safe_load(f)
    T = int(rcfg["MODEL"]["history_len"])
    k = int(rcfg["MODEL"]["chunk_len"])

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)
    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

    from scipy.spatial.transform import Rotation as Rot
    panda_base_pos = np.array(cfg.ENV.PANDA_BASE_POSITION)
    R_base = Rot.from_quat(np.array(cfg.ENV.PANDA_BASE_ORIENTATION)).as_matrix()

    # Scripted grasp-and-back, reused from the paper's policy (close + carry to
    # GOAL_CENTER). Standoff 0 → close at the policy's pose, not 8 cm beyond it.
    gb_policy = HandoverSim2RealPolicy(cfg, None, None, 0.0)
    gb_policy._standoff_offset = np.zeros(3, dtype=np.float32)

    def do_rollout(s, draw=render):
        return rollout(env, model, exec_cfg, point_listener, gb_policy, s, args.device,
                       panda_base_inv_tf, steps_action_repeat, args.max_steps,
                       R_base, panda_base_pos, T, k, draw=draw)

    if args.benchmark:
        n = min(args.num_scenes or env.num_scenes, env.num_scenes)
        succ, grasped_n, closed_n, dists = 0, 0, 0, []
        for s in range(n):
            st, dist, grasped, close_step = do_rollout(s, draw=False)
            if st & EpisodeStatus.SUCCESS:
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
        print(f"ee→ycb         : mean {np.nanmean(dists):.3f} m  median {np.nanmedian(dists):.3f} m")
        return

    do_rollout(args.scene)
    if not render:
        return

    print("\nIn the PyBullet window:  R = re-roll,  N = next scene,  Q = quit.")
    R_KEY, N_KEY, Q_KEY = ord('r'), ord('n'), ord('q')
    scene = args.scene
    try:
        while True:
            keys = pybullet.getKeyboardEvents()
            if R_KEY in keys and keys[R_KEY] & pybullet.KEY_WAS_TRIGGERED:
                do_rollout(scene)
            if N_KEY in keys and keys[N_KEY] & pybullet.KEY_WAS_TRIGGERED:
                scene = (scene + 1) % env.num_scenes
                do_rollout(scene)
            if Q_KEY in keys and keys[Q_KEY] & pybullet.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
