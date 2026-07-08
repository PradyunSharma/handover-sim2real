"""
Phase-3 RL training — online TD3 + BC (GA-DDPG-style blend) on the reactive
Phase-1 policy.

Single-process synchronous loop: roll out the current actor in the handover sim,
store transitions (sparse terminal grasp-proximity reward) in a replay buffer,
and take TD3+BC gradient steps. RL refines the 6-DoF Δpose (pose-BC toward OMG)
and the gripper logit (a proximity-synthesized BCE label + the close reward).
Both actor and critic are conditioned on a remaining-steps clock. Warm-starts the
actor + critic encoders from a trained BC run so RL begins from a competent policy.

Optionally seeds a permanent demo pool (examples/collect_rl_demos.py) sampled at a
fixed fraction alongside the online FIFO, so the +1 close-at-grasp successes are
always present.

Usage:
    python examples/train_rl.py \\
        --sim-cfg  examples/pretrain.yaml \\
        --rl-cfg   examples/configs/rl_phase1.yaml \\
        --bc-run   output/bc_runs/dagger_iter_2_3 \\
        --run-name rl_run1 \\
        [--demos output/rl_demos/train.npz] \\
        [--split train] [--num-iters 2000] [--device cuda] [--resume ...]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from collections import Counter
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

from handover_sim2real.rl import RLActor, QNetwork, ReplayBuffer, TD3BCTrainer
from handover_sim2real.rl.replay_buffer import load_demo_buffer
from handover_sim2real.rl.rollout_worker import RolloutWorker

add_sys_path_from_env("GADDPG_DIR")
from experiments.config import cfg_from_file        # noqa: E402
from rollout_bc_policy import load_policy            # noqa: E402


# ── build actor + critic from the BC run (dims must match for warm-start) ────

def build_networks(bc_run: Path, rlcfg: dict, device: str):
    bc_model = load_policy(bc_run, device)            # has .normalizer attached
    with (bc_run / "config.yaml").open() as f:
        rcfg = yaml.safe_load(f)
    m, d = rcfg["MODEL"], rcfg["DATA"]
    rlm = rlcfg["MODEL"]

    common = dict(
        pc_channels        = int(d["pc_channels"]),
        robot_state_dim    = int(d["robot_state_dim"]),
        feature_dim        = int(m["feature_dim"]),
        robot_hidden       = int(m["robot_hidden"]),
        pointnet_scale     = int(m["pointnet_scale"]),
        pointnet_radius    = float(m["pointnet_radius"]),
        pointnet_nclusters = int(m["pointnet_nclusters"]),
        use_prev_act       = bool(m.get("use_prev_act", False)),
        clock_dim          = int(rlm["clock_dim"]),
    )
    actor = RLActor(policy_hidden=tuple(m["policy_hidden"]), **common)
    critic = QNetwork(q_hidden=tuple(rlm["q_hidden"]), **common)

    actor.warm_start_from_bc(bc_model)
    critic.warm_start_encoders_from_bc(bc_model)
    print(f"[warm-start] actor reproduces BC head (pose + gripper); critic "
          f"encoders copied from {bc_run}")
    return actor, critic, bc_model.normalizer


# ── rollout / eval helpers ───────────────────────────────────────────────────

def _mean(xs):
    xs = [x for x in xs if x == x]            # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def _agg_reasons(counter):
    """Compact failure breakdown: grasp-miss / timeout / other-failure (human
    contact + drop, whatever `_status_name` calls them)."""
    known = {"GRASP_OK", "GRASP_MISS", "TIMEOUT", "EMPTY"}
    return {"miss": counter.get("GRASP_MISS", 0),
            "timeout": counter.get("TIMEOUT", 0),
            "fail": sum(v for k, v in counter.items() if k not in known)}


def expert_initial_range(it: int, loop: dict) -> tuple[int, int]:
    """(lo, hi) inclusive bounds for the per-episode reverse-curriculum takeover
    step — how many steps of committed-OMG-playback warm-start precede the policy.
    ANNEALED window (GA-DDPG's `expert_initial` made into a real curriculum): early
    the takeover is a tight window right at the grasp, so the policy first masters
    the reach-tail descent + close (earns reward → grows the critic's high-Q region
    from the grasp outward); the window then slides toward 0 so the policy learns
    to reach-and-correct from progressively farther, OFF-plan starts with the
    endgame Q already in place for PG to pull it in. `uniform (0, expert_initial_steps)`
    (the OLD behavior) never sequenced this — the intermediate takeovers failed for
    lack of a mastered endgame, so they never earned reward and the high-Q region
    never extended past the last ~2 steps (rl_run8: eval stalls ~0.08 m, buf_pos≈0).
    Falls back to the old uniform window when `expert_initial_anneal_iters` unset."""
    hi0 = int(loop.get("expert_initial_steps", 0))
    if hi0 <= 0 or "expert_initial_anneal_iters" not in loop:
        return 0, hi0
    hi1 = int(loop.get("expert_initial_end", 0))
    ramp = max(int(loop["expert_initial_anneal_iters"]), 1)
    f = min(max(it, 0) / ramp, 1.0)
    hi = int(round(hi0 + (hi1 - hi0) * f))
    win = int(loop.get("expert_initial_window", 6))
    return max(0, hi - win), hi


def collect(worker, actor, buffer, n_eps, rng, num_scenes,
            beta, noise_std, ei_lo, ei_hi, expert_episode_frac=0.0,
            dagger_ratio=0.0):
    """Collect n_eps episodes into the online buffer. A fraction
    `expert_episode_frac` are FULL-expert rollouts (GA-DDPG non-explore): the OMG
    trajectory is played to the grasp + closed → a guaranteed fresh success that
    keeps buf_pos>0. The rest are policy (explore) episodes with a reverse-
    curriculum warm start of a RANDOM `[ei_lo, ei_hi]` committed-OMG-playback
    steps (see `expert_initial_range`), so the policy practices finishing from near
    the grasp and can earn its own +1. roll_* aggregates the POLICY episodes only
    (the progress signal); expert episodes are counted separately."""
    succ = length = ret = kept = skipped = closed = 0
    exp_kept = exp_succ = 0
    minpos, reasons = [], Counter()
    for _ in range(n_eps):
        scene = int(rng.randint(num_scenes))
        if expert_episode_frac > 0.0 and rng.uniform() < expert_episode_frac:
            trans, st = worker.expert_rollout_episode(scene)
        else:
            ei = int(rng.randint(ei_lo, ei_hi + 1)) if ei_hi > 0 else 0
            trans, st = worker.rollout_episode(
                actor, scene, rng, beta=beta,
                expert_initial_steps=ei, noise_std=noise_std,
                dagger_ratio=dagger_ratio)
        if st.get("skipped"):
            skipped += 1
            continue
        buffer.add_episode(trans)
        kept += 1
        if st.get("expert_episode"):
            exp_kept += 1; exp_succ += st["success"]
            continue                    # keep roll_* a policy-only progress signal
        succ += st["success"]; length += st["length"]; ret += st["return"]
        closed += st.get("closed", 0)
        minpos.append(st.get("min_pos", float("nan")))
        reasons[st.get("reason", "?")] += 1
    n = max(kept - exp_kept, 1)         # policy episodes only
    return {"succ_rate": succ / n, "mean_len": length / n, "mean_ret": ret / n,
            "kept": kept, "skipped": skipped, "close_rate": closed / n,
            "min_pos": _mean(minpos), "reasons": reasons,
            "exp_kept": exp_kept, "exp_succ_rate": exp_succ / max(exp_kept, 1)}


def evaluate(worker, actor, rng, n_eps, num_scenes):
    succ = kept = closed = 0
    minpos, minrot, reasons = [], [], Counter()
    for i in range(n_eps):
        scene = i % num_scenes
        _, st = worker.rollout_episode(
            actor, scene, rng, beta=0.0, expert_initial_steps=0, noise_std=0.0)
        if st.get("skipped"):
            continue
        succ += st["success"]; kept += 1; closed += st.get("closed", 0)
        minpos.append(st.get("min_pos", float("nan")))
        minrot.append(st.get("min_rot", float("nan")))
        reasons[st.get("reason", "?")] += 1
    n = max(kept, 1)
    return {"succ": succ / n, "kept": kept, "close_rate": closed / n,
            "min_pos": _mean(minpos), "min_rot": _mean(minrot), "reasons": reasons}


def mix_batch(online, demo, batch_size, demo_frac, device):
    """Sample a batch, drawing a fixed `demo_frac` from the permanent demo pool
    (DDPGfD-style) and the rest from the online FIFO. Falls back to online-only
    when there is no demo pool."""
    if demo is None or len(demo) == 0 or demo_frac <= 0.0:
        return online.sample(batch_size, device)
    if len(online) == 0:
        return demo.sample(batch_size, device)
    n_demo = min(len(demo), max(1, int(round(demo_frac * batch_size))))
    n_on   = batch_size - n_demo
    if n_on <= 0:
        return demo.sample(batch_size, device)
    b_on = online.sample(n_on, device)
    b_de = demo.sample(n_demo, device)
    return {k: torch.cat([b_on[k], b_de[k]], dim=0) for k in b_de}


def demo_frac_at(it: int, loop: dict) -> float:
    """Per-iter demo fraction: linear `demo_frac_start -> demo_frac_end` over
    `demo_frac_ramp` iters, then held at the end value (a permanent floor). Falls
    back to the constant `demo_frac` when the schedule knobs aren't set."""
    if "demo_frac_start" not in loop:
        return float(loop.get("demo_frac", 0.25))
    a = float(loop["demo_frac_start"])
    b = float(loop.get("demo_frac_end", 0.1))
    ramp = max(int(loop.get("demo_frac_ramp", 1000)), 1)
    return a + (b - a) * min(max(it, 0) / ramp, 1.0)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-cfg",  required=True, help="simulator config (e.g. examples/pretrain.yaml)")
    p.add_argument("--rl-cfg",   default="examples/configs/rl_phase1.yaml")
    p.add_argument("--bc-run",   required=True, help="BC run dir to warm-start from")
    p.add_argument("--demos",    default=None,
                   help="npz demo pool from collect_rl_demos.py (permanent, mixed in at demo_frac)")
    p.add_argument("--run-name", default="rl_run1")
    p.add_argument("--out-root", default="output/rl_runs")
    p.add_argument("--split",    default="train", choices=["train", "val", "test"])
    p.add_argument("--device",   default="cuda")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--num-iters", type=int, default=None, help="override LOOP.num_iters")
    p.add_argument("--render",   action="store_true")
    p.add_argument("--egl",      action="store_true")
    p.add_argument("--resume",   default=None, help="path to a checkpoint .pt to resume")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.rl_cfg) as f:
        rlcfg = yaml.safe_load(f)
    loop = rlcfg["LOOP"]
    num_iters = args.num_iters if args.num_iters is not None else int(loop["num_iters"])

    # sim cfg
    cfg = get_cfg()
    cfg_from_file(filename=args.sim_cfg, dict=cfg, merge_to_cn_dict=True)
    cfg.BENCHMARK.SPLIT = args.split
    cfg.SIM.RENDER = bool(args.render)
    if args.egl and not args.render:
        cfg.SIM.BULLET.USE_EGL = True
        print("[warning] --egl: pybullet's EGL renderer leaks ~85 MB GPU/scene; over "
              "a long training run this OOMs. Prefer running WITHOUT --egl (leak-free "
              "CPU renderer) and keep the renderer consistent with the demo pool.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    env            = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    # hand-collision grasp filter for the online OMG/DAgger expert (match demos).
    if bool(rlcfg["RL"].get("hand_collision_filter", True)):
        env.set_hand_collision_filter(
            enable=True,
            thresh=float(rlcfg["RL"].get("hand_collision_thresh", 0.08)),
            points_radius=float(rlcfg["RL"].get("hand_points_radius", 0.35)))
    point_listener = PointListener(cfg, seed=args.seed)
    num_scenes     = env.num_scenes

    actor, critic, normalizer = build_networks(Path(args.bc_run), rlcfg, args.device)
    trainer = TD3BCTrainer(actor, critic, normalizer, rlcfg, args.device)

    buffer = ReplayBuffer(
        capacity=int(loop["capacity"]),
        num_pts=int(rlcfg["DATA"]["num_pts"]),
        pc_channels=int(rlcfg["DATA"]["pc_channels"]))

    # permanent demo pool (optional): pure-OMG close-at-grasp successes, sampled
    # at a fixed fraction so the +1 transitions are never evicted or drowned.
    demo_pool = None
    if args.demos:
        demo_pool = load_demo_buffer(args.demos)
        print(f"[demos] loaded {len(demo_pool)} transitions from {args.demos} "
              f"(demo_frac {demo_frac_at(0, loop):.2f} -> "
              f"{demo_frac_at(10**9, loop):.2f})")

    rollout_max_steps = int(loop.get("rollout_max_steps", 0)) or int(cfg.RL_MAX_STEP)
    worker = RolloutWorker(
        env, point_listener, cfg, normalizer, args.device,
        max_steps=rollout_max_steps, gamma=trainer.gamma,
        act_limit=trainer.act_limit,
        reward_mode=str(rlcfg["RL"].get("reward_mode", "proximity")),
        hold_steps=int(rlcfg["RL"].get("hold_steps", 3)),
        dagger_min_step=int(loop.get("dagger_min_step", 5)),
        dagger_tail_guard=int(loop.get("dagger_tail_guard", 8)))
    dagger_ratio = float(loop.get("dagger_ratio", 0.5))

    # run dir
    run_dir = Path(args.out_root) / args.run_name  
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (run_dir / "rl_config.yaml").open("w") as f:
        yaml.safe_dump(rlcfg, f)
    if normalizer is not None:
        normalizer.save(str(run_dir / "normalization.npz"))
    bc_cfg = Path(args.bc_run) / "config.yaml"
    if bc_cfg.exists():
        shutil.copy(bc_cfg, run_dir / "bc_config.yaml")
    log_path = run_dir / "log.csv"

    start_iter = 0
    best_succ = -1.0
    if args.resume:
        payload = torch.load(args.resume, map_location=args.device)
        trainer.load_state_dict(payload["trainer"])
        start_iter = int(payload.get("iter", 0)) + 1
        best_succ = float(payload.get("best_succ", -1.0))
        print(f"Resumed from {args.resume}: iter {start_iter}, best_succ={best_succ:.3f}")

    if start_iter == 0 and log_path.exists():
        log_path.unlink()   # fresh run — don't stack onto a stale curve

    print(f"Run dir      : {run_dir}")
    print(f"Scenes       : {num_scenes}  split={args.split}  max_steps={rollout_max_steps}")
    print(f"Warm-start   : {args.bc_run}")
    print(f"Iters        : {start_iter} -> {num_iters}   "
          f"(episodes/iter={loop['episodes_per_iter']}, updates/iter={loop['updates_per_iter']})")

    # ----- offline demo pre-training (calibrate critic + actor on the demo pool
    # BEFORE any rollout, so the PG term doesn't hit a random critic) -----
    pretrain_updates = int(loop.get("pretrain_updates", 0))
    if start_iter == 0 and pretrain_updates > 0 and demo_pool is not None:
        t0 = time.time()
        print(f"[pretrain] {pretrain_updates} offline updates on "
              f"{len(demo_pool)} demo transitions (demo-only)...")
        for u in range(pretrain_updates):
            st = trainer.update(demo_pool.sample(int(loop["batch_size"]), args.device))
            if (u + 1) % max(pretrain_updates // 10, 1) == 0:
                print(f"  [pretrain {u+1:6d}/{pretrain_updates}] "
                      f"critic={st.get('critic_loss', float('nan')):.4f} "
                      f"q={st.get('q_mean', float('nan')):.3f} "
                      f"actor={st.get('actor_loss', float('nan')):.4f} "
                      f"bc={st.get('bc_loss', float('nan')):.4f} "
                      f"grip={st.get('grip_loss', float('nan')):.4f}")
        print(f"[pretrain] done ({time.time()-t0:.0f}s)")

    # ----- warmup (seed the buffer, expert-heavy) -----
    if start_iter == 0 and int(loop["warmup_episodes"]) > 0:
        t0 = time.time()
        wlo, whi = expert_initial_range(0, loop)
        w = collect(worker, trainer.actor, buffer, int(loop["warmup_episodes"]),
                    rng, num_scenes, beta=float(loop["warmup_beta"]),
                    noise_std=float(loop["noise_std"]),
                    ei_lo=wlo, ei_hi=whi, dagger_ratio=dagger_ratio)
        print(f"[warmup] buffer={len(buffer)}  succ={w['succ_rate']:.2f}  "
              f"kept={w['kept']} skipped={w['skipped']}  ({time.time()-t0:.0f}s)")

    # ----- main loop -----
    beta_ramp = max(int(loop["beta_ramp_iters"]), 1)
    for it in range(start_iter, num_iters):
        frac = min(it / beta_ramp, 1.0)
        beta = float(loop["beta_start"]) + (float(loop["beta_end"]) - float(loop["beta_start"])) * frac

        ei_lo, ei_hi = expert_initial_range(it, loop)   # reverse-curriculum window
        c = collect(worker, trainer.actor, buffer, int(loop["episodes_per_iter"]),
                    rng, num_scenes, beta=beta, noise_std=float(loop["noise_std"]),
                    ei_lo=ei_lo, ei_hi=ei_hi,
                    expert_episode_frac=float(loop.get("expert_episode_frac", 0.0)),
                    dagger_ratio=dagger_ratio)

        df = demo_frac_at(it, loop)      # scheduled demo fraction this iter
        stats = {}
        astats = {}   # last update that included an actor step (policy_delay)
        for _ in range(int(loop["updates_per_iter"])):
            if len(buffer) < int(loop["batch_size"]) and demo_pool is None:
                break
            batch = mix_batch(buffer, demo_pool, int(loop["batch_size"]),
                              df, args.device)
            stats = trainer.update(batch)
            if "actor_loss" in stats:
                astats = stats

        # online-buffer positive-reward fraction (should be ~0 until the policy
        # earns its own successes — the demos are sampled separately).
        nb = len(buffer)
        buf_pos = float((buffer.reward[:nb] > 0).mean()) if nb > 0 else float("nan")

        if it % 10 == 0 or it == num_iters - 1:
            print(f"[it {it:5d}] buf={len(buffer):6d} beta={beta:.2f} df={df:.2f} "
                  f"ei={ei_lo}-{ei_hi} "
                  f"roll_succ={c['succ_rate']:.2f} roll_minpos={c['min_pos']:.3f} "
                  f"close={c['close_rate']:.2f} exp={c['exp_kept']}/{c['exp_succ_rate']:.2f} "
                  f"skip={c['skipped']} "
                  f"q={stats.get('q_mean', float('nan')):.3f} "
                  f"q_pi={astats.get('q_pi', float('nan')):.1f} "
                  f"bc={astats.get('bc_loss', float('nan')):.4f} "
                  f"|a|={astats.get('a_absmean', float('nan')):.2f} "
                  f"glogit={astats.get('grip_logit', float('nan')):+.2f}")

        # eval (optional this iter) — captured into the CSV row below
        eval_succ = eval_min_pos = eval_min_rot = eval_close = ""
        eval_miss = eval_timeout = eval_fail = ""
        if int(loop["eval_every"]) > 0 and it % int(loop["eval_every"]) == 0 and it > start_iter:
            ev = evaluate(worker, trainer.actor, rng,
                          int(loop["eval_episodes"]), num_scenes)
            succ = ev["succ"]
            eval_succ = round(succ, 4)
            eval_min_pos = round(ev["min_pos"], 4) if ev["min_pos"] == ev["min_pos"] else ""
            eval_min_rot = round(ev["min_rot"], 4) if ev["min_rot"] == ev["min_rot"] else ""
            eval_close = round(ev["close_rate"], 3)
            rb = _agg_reasons(ev["reasons"])
            eval_miss, eval_timeout, eval_fail = rb["miss"], rb["timeout"], rb["fail"]
            # NOTE: in-loop eval is on the SAME (--split) scenes we roll out on —
            # a training-scene progress signal, NOT held-out generalization. For
            # val/test success use examples/rollout_rl_policy.py.
            print(f"  [eval it {it}] success={succ:.3f} min_pos={ev['min_pos']:.3f}m "
                  f"min_rot={ev['min_rot']:.2f}rad close_rate={ev['close_rate']:.2f} "
                  f"[miss={rb['miss']} timeout={rb['timeout']} fail={rb['fail']}] "
                  f"over {ev['kept']} {args.split}-scene rollouts")
            if succ > best_succ:
                best_succ = succ
                _save(run_dir / "checkpoints" / "best.pt", trainer, it, best_succ)

        # one CSV row per iter (curves live here) — plot log.csv
        rr = _agg_reasons(c["reasons"])
        _log_row(log_path, {
            "iter": it, "buffer": len(buffer), "beta": round(beta, 4),
            "ei_hi": ei_hi, "ei_lo": ei_lo,
            "roll_succ": round(c["succ_rate"], 4), "roll_len": round(c["mean_len"], 2),
            "roll_ret": round(c["mean_ret"], 4),
            "critic_loss": stats.get("critic_loss", float("nan")),
            "q_mean": stats.get("q_mean", float("nan")),
            "target_mean": stats.get("target_mean", float("nan")),
            "actor_loss": astats.get("actor_loss", float("nan")),
            "pg_loss": astats.get("pg_loss", float("nan")),
            "bc_loss": astats.get("bc_loss", float("nan")),
            "grip_loss": astats.get("grip_loss", float("nan")),
            "aux_c": stats.get("aux_loss_c", float("nan")),
            "aux_a": astats.get("aux_loss_a", float("nan")),
            "lam": astats.get("lam", float("nan")),
            "n_expert": astats.get("n_expert", 0),
            # tier-1: rollout geometry / failure mode
            "roll_min_pos": round(c["min_pos"], 4) if c["min_pos"] == c["min_pos"] else "",
            "roll_close": round(c["close_rate"], 3), "roll_skip": c["skipped"],
            "roll_miss": rr["miss"], "roll_timeout": rr["timeout"], "roll_fail": rr["fail"],
            # full-expert (non-explore) episodes this iter — the fresh +reward anchor
            "exp_kept": c["exp_kept"], "exp_succ": round(c["exp_succ_rate"], 3),
            # tier-1: deterministic eval geometry / failure mode (eval iters only)
            "eval_min_pos": eval_min_pos, "eval_min_rot": eval_min_rot,
            "eval_close": eval_close, "eval_miss": eval_miss,
            "eval_timeout": eval_timeout, "eval_fail": eval_fail,
            # tier-2: value / actor internals
            "q_pi": astats.get("q_pi", float("nan")),
            "a_absmean": astats.get("a_absmean", float("nan")),
            "grip_logit": astats.get("grip_logit", float("nan")),
            "buf_pos": round(buf_pos, 5) if buf_pos == buf_pos else "",
            "eval_succ": eval_succ, "best_succ": round(best_succ, 4),
        })

        if int(loop["save_every"]) > 0 and (it % int(loop["save_every"]) == 0 or it == num_iters - 1):
            _save(run_dir / "checkpoints" / "last.pt", trainer, it, best_succ)

    print(f"Done. Final policy: {run_dir}  best_eval_success={best_succ:.3f}")


def _save(path, trainer, it, best_succ):
    torch.save({"trainer": trainer.state_dict(), "iter": it,
                "best_succ": best_succ}, path)


def _log_row(path, row: dict) -> None:
    """Append one row to log.csv (writes the header on first call)."""
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
