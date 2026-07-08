"""
Collect Phase-3 RL demonstrations (a permanent demo pool for train_rl.py).

Unlike the offline BC collector (examples/collect_bc_dataset.py, which writes the
[T,...] per-episode HDF5 that BC/ACT train on), this produces **native RL
transitions**: it plays the FULL OMG trajectory (so the EE actually reaches the
grasp — the replan-first-waypoint online expert does not) and **commits the close
the moment the EE enters grasp proximity**, ending each episode in the terminal
+1 that the online expert path structurally cannot generate.

Every transition carries the same fields the online rollout worker stores (same
point-cloud / robot-state builders, same clock, same reward, the proximity
gripper label), so the pool is format-compatible with the online FIFO. Output is
**streamed to HDF5 one episode at a time** (via DemoHDF5Writer): memory stays
bounded over hundreds of scenes and a crash/OOM keeps every episode already on
disk. train_rl.py loads it into a non-evicting demo pool sampled at a fixed
fraction (DDPGfD-style) — so the +1 successes are never evicted or drowned.

    python examples/collect_rl_demos.py \\
        --sim-cfg  examples/pretrain.yaml \\
        --rl-cfg   examples/configs/rl_phase1.yaml \\
        --bc-run   output/bc_runs/dagger_iter_2_3 \\
        --out      output/rl_demos/train.h5 \\
        [--split train] [--num-scenes N]

NOTE: run WITHOUT `--egl`. pybullet's EGL renderer leaks ~85 MB of GPU memory per
scene (each reset loads/removes the YCB + MANO meshes and removeBody doesn't free
them in the plugin) and OOM-kills a long collection; the CPU TinyRenderer is
leak-free (~1.1 GB flat) and barely slower. Keep the SAME renderer choice across
the whole pipeline (BC collection, this, RL training/eval) — the depth-based
cloud is nearly identical either way, but stay consistent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gym
import numpy as np
import torch
import yaml

import handover            # noqa: F401  registers envs
import handover_sim2real   # noqa: F401  registers envs

from handover.benchmark_wrapper import HandoverBenchmarkWrapper
from handover_sim2real.config import get_cfg
from handover_sim2real.policy import PointListener
from handover_sim2real.utils import add_sys_path_from_env

from handover_sim2real.rl.replay_buffer import DemoHDF5Writer
# The demo pool is produced by the SAME code path as the online full-expert
# rollout (RolloutWorker.expert_rollout_episode), so demos and online experts are
# byte-identical — one source of truth for the reward-critical playback+close logic.
from handover_sim2real.rl.rollout_worker import RolloutWorker

add_sys_path_from_env("GADDPG_DIR")
from experiments.config import cfg_from_file        # noqa: E402
from rollout_bc_policy import load_policy           # noqa: E402  (action normalizer)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-cfg",   required=True, help="simulator config (e.g. examples/pretrain.yaml)")
    p.add_argument("--rl-cfg",    default="examples/configs/rl_phase1.yaml",
                   help="RL config (gamma / act_limit / close thresholds)")
    p.add_argument("--bc-run",    required=True, help="BC run dir — supplies the action normalizer")
    p.add_argument("--out",       required=True, help="output HDF5 path (e.g. output/rl_demos/train.h5)")
    p.add_argument("--split",     default="train", choices=["train", "val", "test"])
    p.add_argument("--num-scenes", type=int, default=None, help="cap on scenes (default: all)")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--egl",       action="store_true",
                   help="headless NVIDIA (EGL) renderer. WARNING: pybullet's EGL "
                        "renderer LEAKS ~85 MB of GPU memory per scene (bodies "
                        "loaded/removed each reset) -> OOM on a long run. Leave it "
                        "OFF (CPU TinyRenderer) for a full collection; it's leak-free "
                        "(~1.1 GB flat) and barely slower (~5 s/scene).")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.rl_cfg) as f:
        rlcfg = yaml.safe_load(f)
    r = rlcfg["RL"]

    cfg = get_cfg()
    cfg_from_file(filename=args.sim_cfg, dict=cfg, merge_to_cn_dict=True)
    cfg.BENCHMARK.SPLIT = args.split
    cfg.SIM.RENDER = False
    if args.egl:
        cfg.SIM.BULLET.USE_EGL = True
        print("[warning] --egl: pybullet's EGL renderer leaks ~85 MB GPU/scene "
              "(both VRAM and host RSS) and will OOM a long collection. Prefer "
              "running WITHOUT --egl (leak-free CPU renderer) for the full pool.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env            = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    # paper-faithful hand-collision grasp filter: OMG prunes grasps whose gripper
    # would collide with the human hand, so demos don't drive into the hand
    # (scenes where every grasp collides are skipped when OMG's goal set empties).
    if bool(r.get("hand_collision_filter", True)):
        env.set_hand_collision_filter(
            enable=True,
            thresh=float(r.get("hand_collision_thresh", 0.08)),
            points_radius=float(r.get("hand_points_radius", 0.35)))
    point_listener = PointListener(cfg, seed=args.seed)
    # only the BC run's action normalizer is needed (execution is pure OMG).
    normalizer = load_policy(Path(args.bc_run), args.device).normalizer

    # one source of truth: the demo pool is produced by the SAME method the online
    # loop uses for full-expert rollouts, so demos and online experts are identical.
    # horizon must MATCH training (clock is normalized by max_steps) — take it from
    # LOOP.rollout_max_steps if set, else cfg.RL_MAX_STEP.
    rollout_max_steps = int(rlcfg.get("LOOP", {}).get("rollout_max_steps", 0)) or int(cfg.RL_MAX_STEP)
    worker = RolloutWorker(
        env, point_listener, cfg, normalizer, args.device,
        max_steps=rollout_max_steps, gamma=float(r.get("gamma", 0.95)),
        act_limit=float(r.get("act_limit", 5.0)),
        close_pos_thresh=float(r.get("close_pos_thresh", 0.02)),
        close_rot_thresh=float(r.get("close_rot_thresh", 0.34)),
        reward_mode=str(r.get("reward_mode", "proximity")),
        hold_steps=int(r.get("hold_steps", 3)))

    n = env.num_scenes if args.num_scenes is None else min(env.num_scenes, args.num_scenes)

    # Stream to HDF5 one episode at a time: memory stays bounded over hundreds of
    # scenes AND an OOM/crash (a SIGKILL no try/except can catch) still keeps every
    # episode already on disk. The ~5 GB baseline (CUDA + OMG SDF cache + pybullet)
    # is what actually OOMs here, not the transitions — so also free RAM / cap
    # --num-scenes if the machine is tight (a demo pool doesn't need every scene).
    out_path = Path(args.out)
    if out_path.suffix != ".h5":
        out_path = out_path.with_suffix(".h5")
        print(f"[note] streaming to {out_path} (HDF5) for bounded memory + crash safety")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Collecting demos: {n} scenes  split={args.split}  out={out_path}")

    writer = DemoHDF5Writer(str(out_path), extra={
        "action_mean": np.asarray(normalizer.action_mean, dtype=np.float32),
        "action_std":  np.asarray(normalizer.action_std,  dtype=np.float32)})

    kept = succ = skipped = 0
    interrupted = False
    try:
        for i in range(n):
            trans, st = worker.expert_rollout_episode(i)
            if st.get("skipped"):
                skipped += 1
                continue
            for t in trans:
                t["scene_idx"] = float(i)   # for per-episode replay in the visualizer
            writer.append(trans)            # -> disk immediately (flushed)
            kept += 1
            succ += st["success"]
            if (i + 1) % 20 == 0 or i == n - 1:
                print(f"  [{i+1:4d}/{n}] episodes={kept} transitions={writer.num_transitions} "
                      f"closed-at-grasp={succ}/{kept} skipped(OMG)={skipped}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[interrupted] Ctrl-C after {kept} episodes — partial pool already on disk.")
    finally:
        n_trans = writer.num_transitions
        writer.close()

    if kept == 0:
        out_path.unlink(missing_ok=True)
        print("No demos collected. Nothing written.")
        return

    tag = "PARTIAL " if interrupted else ""
    print(f"\nDone. {tag}{kept} demo episodes, {n_trans} transitions "
          f"({succ} closed-at-grasp) -> {out_path}")


if __name__ == "__main__":
    main()
