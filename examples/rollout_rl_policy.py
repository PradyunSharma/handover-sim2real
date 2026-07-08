"""
Benchmark a trained Phase-3 RL policy on held-out scenes.

`train_rl.py` rolls out AND its in-loop eval run on the SAME `--split` (train),
so that number is train-scene success, not generalization. This script loads an
RL checkpoint's actor and evaluates it deterministically (no exploration noise,
no expert, no learning) over a chosen split — use `val` for checkpoint selection
and `test` for the final number.

    # headless benchmark (success rate over a split):
    python examples/rollout_rl_policy.py \\
        --rl-run   output/rl_runs/rl_run1 \\
        --sim-cfg  examples/pretrain.yaml \\
        --split    val --num-scenes 100

    # WATCH it live in the pybullet GUI, stepping scenes by hand:
    python examples/rollout_rl_policy.py --rl-run output/rl_runs/rl_run7 \\
        --sim-cfg examples/pretrain.yaml --split train --render --scene 0
    #   [n]/→ next scene, [p]/← prev, [r]eplay, [q]uit; green wireframe = OMG goal
    #   grasp. Swap --checkpoint best|last to compare.

Reuses the training rollout worker (same gripper heuristic + observation
builders), so eval matches how the policy was trained. The actor is the only
network needed — the critic / aux heads are irrelevant at deployment.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

import gym
import numpy as np
import pybullet
import torch
import yaml

import handover            # noqa: F401  registers envs
import handover_sim2real   # noqa: F401  registers envs

from handover.benchmark_wrapper import HandoverBenchmarkWrapper
from handover_sim2real.config import get_cfg
from handover_sim2real.policy import PointListener
from handover_sim2real.utils import add_sys_path_from_env

from handover_sim2real.rl import RLActor
from handover_sim2real.rl.rollout_worker import RolloutWorker

add_sys_path_from_env("GADDPG_DIR")
from experiments.config import cfg_from_file   # noqa: E402
from rollout_bc_policy import draw_gripper      # noqa: E402  (GUI grasp overlay)


def _draw_grasp(grasp_world, marker_ids):
    """Overlay the OMG goal grasp as a green Panda-gripper wireframe. Clears the
    previous scene's markers first (world-frame debug lines persist across steps)."""
    for mid in marker_ids:
        try:
            pybullet.removeUserDebugItem(mid)
        except Exception:
            pass
    marker_ids.clear()
    if grasp_world is not None:
        draw_gripper(grasp_world, [0.0, 1.0, 0.0], marker_ids, 3.0)


# key -> action for interactive GUI navigation
_NAV = {ord("n"): "next", ord("p"): "prev", ord("r"): "replay", ord("q"): "quit",
        pybullet.B3G_RIGHT_ARROW: "next", pybullet.B3G_LEFT_ARROW: "prev"}


def _wait_key():
    """Block (keeping the GUI responsive) until the user presses a nav key."""
    while True:
        ev = pybullet.getKeyboardEvents()
        for code, act in _NAV.items():
            if code in ev and (ev[code] & pybullet.KEY_WAS_TRIGGERED):
                return act
        time.sleep(0.03)


def load_rl_actor(rl_run: Path, checkpoint: str, device: str):
    """Build the RLActor from the run's saved dims and load the checkpoint's
    actor weights + normalizer. Dims come from bc_config.yaml (encoder/head) and
    rl_config.yaml (clock). Loads with strict=False so an actor built with the
    aux head still accepts a checkpoint trained without it (aux is unused here)."""
    from handover_sim2real.bc import Normalizer

    with (rl_run / "rl_config.yaml").open() as f:
        rlcfg = yaml.safe_load(f)
    with (rl_run / "bc_config.yaml").open() as f:
        bccfg = yaml.safe_load(f)
    m, d, rlm = bccfg["MODEL"], bccfg["DATA"], rlcfg["MODEL"]

    actor = RLActor(
        pc_channels        = int(d["pc_channels"]),
        robot_state_dim    = int(d["robot_state_dim"]),
        feature_dim        = int(m["feature_dim"]),
        robot_hidden       = int(m["robot_hidden"]),
        policy_hidden      = tuple(m["policy_hidden"]),
        pointnet_scale     = int(m["pointnet_scale"]),
        pointnet_radius    = float(m["pointnet_radius"]),
        pointnet_nclusters = int(m["pointnet_nclusters"]),
        use_prev_act       = bool(m.get("use_prev_act", False)),
        clock_dim          = int(rlm["clock_dim"]),
    ).to(device)

    ckpt_path = rl_run / "checkpoints" / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        ckpt_path = rl_run / "checkpoints" / "last.pt"
    payload = torch.load(ckpt_path, map_location=device)
    missing, unexpected = actor.load_state_dict(payload["trainer"]["actor"], strict=False)
    actor.eval()
    normalizer = Normalizer.load(str(rl_run / "normalization.npz"))
    print(f"Loaded actor from {ckpt_path} (iter {payload.get('iter', '?')}, "
          f"train best_succ={payload.get('best_succ', float('nan')):.3f})")
    aux_missing = [k for k in missing if "aux_head" not in k]
    if aux_missing or unexpected:
        print(f"  state_dict: missing(non-aux)={aux_missing} unexpected={list(unexpected)}")
    return actor, normalizer, rlcfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rl-run",    required=True, help="RL run dir (output/rl_runs/<name>)")
    p.add_argument("--sim-cfg",   required=True, help="simulator config, e.g. examples/pretrain.yaml")
    p.add_argument("--split",     default="val", choices=["train", "val", "test"])
    p.add_argument("--num-scenes", type=int, default=None, help="cap on scenes (default: all in the split)")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--device",    default="cuda")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--egl",       action="store_true",
                   help="headless NVIDIA renderer — match the renderer the policy trained on")
    p.add_argument("--render",    action="store_true",
                   help="open the pybullet GUI and step scenes INTERACTIVELY (does "
                        "NOT auto-advance): [n]/→ next, [p]/← prev, [r]eplay, [q]uit. "
                        "The OMG goal grasp is drawn as a green gripper each scene.")
    p.add_argument("--scene",     type=int, default=None,
                   help="scene index to START on (default 0) — handy with --render")
    p.add_argument("--max-steps", type=int, default=None,
                   help="policy steps per episode (default = the run's training "
                        "horizon, rollout_max_steps=30). NOTE: this is also the "
                        "clock denominator, so a value != the trained horizon feeds "
                        "the policy an off-distribution clock — fine for eyeballing "
                        "how far it gets with more/less time, not a fair eval number.")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = get_cfg()
    cfg_from_file(filename=args.sim_cfg, dict=cfg, merge_to_cn_dict=True)
    cfg.BENCHMARK.SPLIT = args.split
    cfg.SIM.RENDER = bool(args.render)        # True -> pybullet GUI window
    if args.egl and not args.render:
        cfg.SIM.BULLET.USE_EGL = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    env            = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    point_listener = PointListener(cfg, seed=args.seed)
    actor, normalizer, rlcfg = load_rl_actor(Path(args.rl_run), args.checkpoint, args.device)
    # hand-collision grasp filter for the OMG expert — match the training run.
    if bool(rlcfg["RL"].get("hand_collision_filter", True)):
        env.set_hand_collision_filter(
            enable=True,
            thresh=float(rlcfg["RL"].get("hand_collision_thresh", 0.08)),
            points_radius=float(rlcfg["RL"].get("hand_points_radius", 0.35)))

    start = int(args.scene) if args.scene is not None else 0
    if args.num_scenes is not None:
        n = args.num_scenes
    elif args.render:
        n = 5                                    # watching live: a handful by default
    else:
        n = env.num_scenes
    n = min(n, env.num_scenes - start)
    # by default match the horizon the policy TRAINED with (clock is normalized by
    # max_steps); --max-steps overrides it (falls back to cfg.RL_MAX_STEP if the run
    # predates the rollout_max_steps knob).
    if args.max_steps is not None:
        rollout_max_steps = int(args.max_steps)
    else:
        rollout_max_steps = int(rlcfg.get("LOOP", {}).get("rollout_max_steps", 0)) or int(cfg.RL_MAX_STEP)
    worker = RolloutWorker(
        env, point_listener, cfg, normalizer, args.device,
        max_steps=rollout_max_steps,
        gamma=float(rlcfg["RL"].get("gamma", 0.95)),
        act_limit=float(rlcfg["RL"].get("act_limit", 5.0)),
        reward_mode=str(rlcfg["RL"].get("reward_mode", "proximity")),
        hold_steps=int(rlcfg["RL"].get("hold_steps", 3)))
    # eval never DAgger-replans (rollout_episode dagger_ratio defaults to 0):
    # the reward is scored against the step-0 goal grasp, one OMG plan/episode.

    print(f"Eval: scenes {start}..{start+n-1}  split={args.split}  "
          f"max_steps={rollout_max_steps}  policy={args.rl_run} ({args.checkpoint})"
          f"{'  [GUI]' if args.render else ''}")

    # ---- interactive GUI: one scene at a time, stepped by keypress ----
    if args.render:
        print("Controls (focus the GUI window):  [n]/→ next   [p]/← prev   "
              "[r]eplay   [q]uit")
        marker_ids: list = []
        i = start
        while True:
            print(f"\n--- scene {i} ---  (green wireframe = OMG goal grasp)")
            _draw_grasp(None, marker_ids)          # clear last scene's overlay
            _, st = worker.rollout_episode(
                actor, i, rng, beta=0.0, expert_initial_steps=0, noise_std=0.0,
                on_grasp=lambda g: _draw_grasp(g, marker_ids))
            if st.get("skipped"):
                print("  OMG skip — no hand-free grasp for this scene")
            else:
                print(f"  {st['reason']}  succ={st['success']}  len={st['length']}  "
                      f"min_pos={st.get('min_pos', float('nan')):.3f} m  "
                      f"min_rot={st.get('min_rot', float('nan')):.3f} rad")
            act = _wait_key()
            if act == "quit":
                break
            elif act == "next":
                i = min(i + 1, env.num_scenes - 1)
            elif act == "prev":
                i = max(i - 1, 0)
            # "replay" keeps the same scene index
        return

    # ---- headless benchmark: auto-advance over the scenes, report success ----
    succ = kept = length = 0
    statuses = Counter()
    for i in range(start, start + n):
        _, st = worker.rollout_episode(actor, i, rng, beta=0.0,
                                       expert_initial_steps=0, noise_std=0.0)
        if st.get("skipped"):
            statuses["OMG_SKIP"] += 1
            print(f"  scene {i:4d}: OMG skip (no hand-free grasp)")
            continue
        kept += 1
        succ += st["success"]
        length += st["length"]
        statuses[st["reason"]] += 1                 # GRASP_OK / GRASP_MISS / TIMEOUT / ...
        print(f"  scene {i:4d}: {st['reason']:12s} succ={st['success']} "
              f"len={st['length']} min_pos={st.get('min_pos', float('nan')):.3f}m "
              f"| running {succ}/{kept} = {100.0*succ/max(kept,1):.1f}%")

    print("\n==== RL benchmark ====")
    print(f"split           : {args.split}")
    print(f"grasp-success   : {succ}/{kept} = {100.0 * succ / max(kept, 1):.1f}%"
          f"  (closed within grasp proximity)")
    print(f"mean ep length  : {length / max(kept, 1):.1f} steps")
    print(f"outcome breakdown:")
    for name, c in statuses.most_common():
        print(f"    {name:16s} {c}")


if __name__ == "__main__":
    main()
