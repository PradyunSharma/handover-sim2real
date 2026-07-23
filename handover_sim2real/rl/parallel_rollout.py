"""
Parallel rollout collection for Phase-3 RL — the cluster analog of the paper's
`num_remotes` Ray actors (examples/train.py), for the single-process learner in
examples/train_rl.py.

WHY. The learner is cheap (a small PointNet++ actor/critic on the GPU); the
wall-clock cost is ROLLOUT collection — PyBullet stepping + OMG trajectory
optimization, both CPU-bound and, in the serial loop, run one episode at a time.
On a workstation that's fine; on a many-core cluster node it leaves ~all cores
idle. This module fans rollouts across `num_workers` persistent worker processes:
each worker owns its OWN handover env + OMG planner + a CPU copy of the actor, the
manager broadcasts fresh actor weights before each collection and gathers the
produced transitions, and the learner keeps the GPU to itself.

DETERMINISM. The MANAGER (train_rl.collect(), using the run's single rng) still
decides, per episode, the scene / expert-vs-policy / expert_initial_steps exactly
as the serial loop does, and stamps each job with a unique rng seed for the
in-rollout randomness (exploration noise, DAgger tail replans). So the sequence of
scenes/kinds is unchanged; only the per-episode noise stream is drawn from an
independent, seed-reproducible RandomState per worker (bitwise identity with the
serial path is neither possible nor meaningful once rollouts run concurrently).

THREADS. Each worker pins itself to a single thread (torch.set_num_threads(1) +
OMP/MKL/BLAS=1, set in the parent before spawn so children inherit it at import),
so `num_workers` ≈ the number of CPU cores you request from SLURM. Worker actor
inference runs on CPU by default (leaves the GPU entirely to the learner and
sidesteps CUDA-in-subprocess fragility); pass worker_device="cuda" only if the GPU
has headroom for N extra contexts.
"""

from __future__ import annotations

import os
import queue
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml


# ── actor/normalizer rebuild (worker side) ───────────────────────────────────
# Mirrors train_rl.build_networks' `common` dict, but only the ACTOR is needed in
# a rollout worker (no critic), and the weights are pushed from the learner every
# collection — so this only has to reproduce the ARCHITECTURE + the normalizer.

def build_actor_and_normalizer(bc_run: Path, rlcfg: dict, device: str):
    from handover_sim2real.rl import RLActor
    from handover_sim2real.bc import Normalizer

    with (bc_run / "config.yaml").open() as f:
        rcfg = yaml.safe_load(f)
    m, d = rcfg["MODEL"], rcfg["DATA"]
    rlm = rlcfg["MODEL"]

    norm_path = bc_run / "normalization.npz"
    normalizer = Normalizer.load(norm_path) if norm_path.exists() else None

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
    actor.eval()
    return actor, normalizer


def cpu_state_dict(module) -> dict:
    """Actor weights as CPU tensors, ready to ship to workers over a Queue."""
    return {k: v.detach().to("cpu") for k, v in module.state_dict().items()}


# ── one job → one episode (worker side) ──────────────────────────────────────

def _run_job(worker, actor, job: dict):
    """Execute one rollout job. `kind` ∈ {expert, policy, eval}. Returns the
    (transitions, stats) tuple the ReplayBuffer / accounting expects."""
    scene = int(job["scene"])
    kind = job["kind"]
    rng = np.random.RandomState(int(job.get("seed", 0)))
    if kind == "expert":
        return worker.expert_rollout_episode(
            scene, rng, dart_ratio=float(job.get("dart_ratio", 0.0)))
    if kind == "eval":
        return worker.rollout_episode(actor, scene, rng, beta=0.0,
                                      expert_initial_steps=0, noise_std=0.0)
    # policy (training) episode
    return worker.rollout_episode(
        actor, scene, rng,
        beta=float(job.get("beta", 0.0)),
        noise_std=float(job.get("noise_std", 0.0)),
        expert_initial_steps=int(job.get("expert_initial_steps", 0)),
        dagger_ratio=float(job.get("dagger_ratio", 0.0)),
        dart_ratio=float(job.get("dart_ratio", 0.0)))


# ── worker process entry point ───────────────────────────────────────────────

def _worker_main(worker_id, sim_cfg, rlcfg, bc_run, split, seed,
                 rollout_kwargs, worker_device, egl, task_q, result_q):
    """Persistent worker: build the env + actor once, then loop pulling
    (weights, jobs) batches off `task_q`, syncing weights and returning the
    produced (transitions, stats) list on `result_q`."""
    try:
        # Reuse the collectors' import + env setup verbatim (byte-identical states).
        import gym
        import handover            # noqa: F401  registers envs
        import handover_sim2real   # noqa: F401  registers envs

        from handover.benchmark_wrapper import HandoverBenchmarkWrapper
        from handover_sim2real.config import get_cfg
        from handover_sim2real.policy import PointListener
        from handover_sim2real.utils import (
            add_sys_path_from_env, resolve_valid_grasp_dict_path)
        from handover_sim2real.rl.rollout_worker import RolloutWorker

        add_sys_path_from_env("GADDPG_DIR")
        from experiments.config import cfg_from_file        # noqa: F401

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = get_cfg()
        cfg_from_file(filename=sim_cfg, dict=cfg, merge_to_cn_dict=True)
        cfg.BENCHMARK.SPLIT = split
        cfg.SIM.RENDER = False
        if egl:
            cfg.SIM.BULLET.USE_EGL = True

        rl = rlcfg["RL"]
        # paper's offline hand-collision filter (valid_grasp_dict): set on omg_config
        # BEFORE the env is built (see examples/train_rl.py for the rationale).
        _vgd = resolve_valid_grasp_dict_path(rl, cfg.BENCHMARK.SETUP)
        if _vgd is not None:
            cfg.omg_config["valid_grasp_dict_path"] = _vgd
        env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
        # our aggressive runtime filter (0.08 m); off in valid_grasp_dict configs.
        if bool(rl.get("hand_collision_filter", True)):
            env.set_hand_collision_filter(
                enable=True,
                thresh=float(rl.get("hand_collision_thresh", 0.08)),
                points_radius=float(rl.get("hand_points_radius", 0.35)))
        point_listener = PointListener(cfg, seed=seed)

        actor, normalizer = build_actor_and_normalizer(
            Path(bc_run), rlcfg, worker_device)
        worker = RolloutWorker(env, point_listener, cfg, normalizer,
                               worker_device, **rollout_kwargs)

        result_q.put(("ready", worker_id, int(env.num_scenes)))
    except Exception:
        result_q.put(("error", worker_id, traceback.format_exc()))
        return

    while True:
        msg = task_q.get()
        if msg is None:
            break
        weights, jobs = msg
        if weights is not None:
            actor.load_state_dict(weights)
            actor.eval()
        out = []
        for job in jobs:
            try:
                out.append(_run_job(worker, actor, job))
            except Exception:
                # A single bad scene must not kill a multi-hour run: report it as
                # a skipped episode and keep going.
                out.append((None, {"skipped": True,
                                    "error": traceback.format_exc(),
                                    "scene_idx": job.get("scene")}))
        result_q.put(("result", worker_id, out))


# ── manager (learner side) ───────────────────────────────────────────────────

class ParallelRolloutManager:
    """Spawns `num_workers` persistent rollout workers and dispatches jobs to
    them. `rollout(weights, jobs)` broadcasts the (CPU) actor weights, round-robins
    the jobs, and returns the flat list of (transitions, stats) — the same tuples
    the serial RolloutWorker returns, so train_rl's accounting is unchanged."""

    def __init__(self, num_workers, sim_cfg, rl_cfg, bc_run, split, base_seed,
                 rollout_kwargs, worker_device="cpu", egl=False,
                 start_timeout=1800.0, job_timeout=3600.0):
        self.num_workers = int(num_workers)
        self.job_timeout = float(job_timeout)

        # Pin child math libs to one thread each (set before spawn so children
        # inherit it at import). setdefault: never override an explicit user value.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(var, "1")

        ctx = mp.get_context("spawn")
        self.result_q = ctx.Queue()
        self.task_qs = []
        self.procs = []
        for wid in range(self.num_workers):
            tq = ctx.Queue()
            p = ctx.Process(
                target=_worker_main,
                args=(wid, sim_cfg, rl_cfg, str(bc_run), split,
                      int(base_seed) + 1 + wid, rollout_kwargs, worker_device,
                      bool(egl), tq, self.result_q),
                daemon=True)
            p.start()
            self.task_qs.append(tq)
            self.procs.append(p)

        # Barrier: every worker must finish building its env before we dispatch.
        self.num_scenes = None
        ready = 0
        while ready < self.num_workers:
            try:
                tag, wid, payload = self.result_q.get(timeout=start_timeout)
            except queue.Empty:
                raise RuntimeError(
                    "rollout workers did not come up within "
                    f"{start_timeout:.0f}s (check OMG_PLANNER_DIR / GADDPG_DIR / "
                    "the sim env vars are exported to the job)")
            if tag == "error":
                self.close()
                raise RuntimeError(
                    f"rollout worker {wid} failed to start:\n{payload}")
            self.num_scenes = payload
            ready += 1

    def _get_one(self):
        while True:
            try:
                return self.result_q.get(timeout=self.job_timeout)
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(self.procs)
                        if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f"rollout worker(s) died mid-run: {dead}")
                # still alive, just slow (OMG can be) — keep waiting.

    def rollout(self, weights, jobs):
        """Broadcast `weights` to all workers, run `jobs` across them, and return
        the flat list of (transitions, stats)."""
        if not jobs:
            return []
        buckets = [[] for _ in range(self.num_workers)]
        for i, j in enumerate(jobs):
            buckets[i % self.num_workers].append(j)
        # Send to ALL workers (empty bucket → they just sync weights and reply []),
        # so every worker's actor stays current and the result count is fixed.
        for wid in range(self.num_workers):
            self.task_qs[wid].put((weights, buckets[wid]))
        results = []
        for _ in range(self.num_workers):
            tag, wid, payload = self._get_one()
            if tag == "error":
                raise RuntimeError(f"rollout worker {wid} crashed:\n{payload}")
            results.extend(payload)
        return results

    def close(self):
        for tq in self.task_qs:
            try:
                tq.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
