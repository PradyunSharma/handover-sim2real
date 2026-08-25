"""Parallel D_i collection for Phase 5 — the DAgger analog of rl/parallel_rollout.py.

WHY. Measured on run 2/run 3, collection is ~44% of a DAgger run's wall clock
(759 s per 100-episode iteration) and it runs one episode at a time on one core.
Profiling a single collection step:

    point-cloud render + deproject   84.6 ms   52%
    OMG replan (horizon 10)          74.2 ms   45%
    PointNet++ policy forward         2.2 ms    1.3%
    env.step (PyBullet)               1.3 ms    0.8%

so it is CPU-bound work — rendering and trajectory optimization — with the GPU
almost idle. Fanning episodes across worker processes is the whole win. (Every
worker still needs a CUDA context: `furthest_point_sample` has no CPU kernel, so
`worker_device` must be "cuda". That is a correctness constraint, not a
throughput one.)

WHY THIS IS EXACT, NOT APPROXIMATE. In Phase-3 RL the actor updates continuously,
so concurrent rollouts see stale weights and the parallel path is only
*equivalent in distribution*. DAgger has no such problem: pi_hat_i is FROZEN for
the whole of iteration i (weights change only at the refit), and the manager
draws all m scenes up front. Every worker loads the same checkpoint from disk
once per iteration and no episode reads another's result, so parallel collection
is the same algorithm as serial up to which RandomState each episode draws from.

DETERMINISM. The MANAGER picks the (scene, grasp) pairs (train_regrasp's
single rng, unchanged) and stamps each job with its own seed, so the sequence is
identical to the serial path. Results are returned to the caller in JOB ORDER,
not completion order, so the HDF5 episode order and the GraspRegistry's
first-seen bookkeeping do not depend on which worker happened to finish first.

THREADS. Each worker pins itself to one thread (torch + OMP/MKL/BLAS), set in the
parent before spawn so children inherit it at import. So `num_workers` ~ the
cores you asked SLURM for, minus a couple for the main process.

SIZING. Pick a worker count that DIVIDES episodes_per_iter: the manager
round-robins, so the iteration takes ceil(m / W) episode-times. At Phase 5's
m=200, W=20 gives 10 each (100% utilization) while W=16 gives 13 for some
workers (96%). Past
~20 the returns are tiny — collection is already down to minutes and the
from-scratch refit dominates.
"""

from __future__ import annotations

import contextlib
import io
import os
import queue
import traceback
from pathlib import Path

import numpy as np
import torch.multiprocessing as mp


# ── worker process entry point ───────────────────────────────────────────────

def _worker_main(worker_id, cfg4, seed, worker_device, task_q, result_q):
    """Persistent worker: build the env ONCE, then loop pulling jobs.

    A job batch is
    (run_dir, ckpt, beta, params, iteration, [(idx, scene, grasp_idx, seed), ...]).
    The policy is (re)loaded whenever run_dir/ckpt changes — i.e. once per DAgger
    iteration — because building the env is the expensive part (~1.3 s of SDF
    load per scene reset on top of a one-off construction), not loading weights.
    """
    try:
        import torch

        from handover_sim2real.regrasp.collector import collect_dagger_episode
        from handover_sim2real.regrasp.evaluator import eval_one
        from handover_sim2real.regrasp.policy_io import load_policy_runner
        from handover_sim2real.regrasp.setup import build_regrasp_context

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # verbose=False: 20 workers each printing the pin-table banner is noise.
        ctx = build_regrasp_context(cfg4, seed=seed, verbose=False)
        result_q.put(("ready", worker_id, int(ctx.sim.num_scenes)))

        runner = None
        loaded = None
        while True:
            job = task_q.get()
            if job is None:
                break
            # A 7-tuple carries an explicit kind; a 6-tuple is a collection
            # job, kept so nothing outside this file has to change at once.
            if len(job) == 7:
                kind, run_dir, ckpt, beta, params, iteration, items = job
            else:
                kind = "collect"
                run_dir, ckpt, beta, params, iteration, items = job

            if loaded != (run_dir, ckpt):
                del runner
                if worker_device != "cpu":
                    torch.cuda.empty_cache()
                # Silenced: the manager already logs which checkpoint drives the
                # iteration, and 20 workers x 15 iterations is 300 identical lines.
                with contextlib.redirect_stdout(io.StringIO()):
                    runner, _ = load_policy_runner(Path(run_dir), worker_device,
                                                   ckpt=ckpt)
                loaded = (run_dir, ckpt)

            out = []
            if kind == "eval":
                # EVAL, not collection. Same env, same frozen checkpoint, a
                # different per-episode call: `eval_one` rolls the policy with
                # NO beta mixture and returns a metrics row instead of an
                # episode. It draws no random numbers, so `ep_seed` is ignored
                # and the parallel result is bit-identical to the serial one.
                for idx, scene, grasp_i, _ in items:
                    row = eval_one(ctx.sim, runner, int(scene), int(grasp_i),
                                   params=params, pin_table=ctx.pin_table)
                    out.append((int(idx), row, None))
            else:
                for idx, scene, grasp_i, ep_seed in items:
                    rng = np.random.RandomState(ep_seed)
                    episode, st = collect_dagger_episode(
                        ctx.sim, runner, int(scene), rng=rng, beta=float(beta),
                        params=params, pin_table=ctx.pin_table,
                        grasp_idx=int(grasp_i))
                    out.append((int(idx), episode, st))
            result_q.put(("done", worker_id, out))
    except Exception:                                    # noqa: BLE001
        try:
            result_q.put(("error", worker_id, traceback.format_exc()))
        except Exception:                                # noqa: BLE001, S110
            pass


# ── manager ──────────────────────────────────────────────────────────────────

class ParallelDaggerCollector:
    """Worker pool for D_i collection.

    `collect(run_dir, ckpt, pairs, seeds, beta, params, iteration)` returns
    `[(episode | None, stats), ...]` in the order `pairs` was given — the same
    sequence the serial loop in collector.collect_iteration produces, so the
    caller's HDF5 writing, aggregation and registry checks are unchanged.
    """

    def __init__(self, num_workers, cfg4, base_seed=0, worker_device="cuda",
                 start_timeout=1800.0, job_timeout=7200.0):
        self.num_workers = int(num_workers)
        self.job_timeout = float(job_timeout)

        # Pin child math libs to one thread each. setdefault so an explicit user
        # value is never overridden; set BEFORE spawn so children inherit it.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(var, "1")

        ctx = mp.get_context("spawn")
        self.result_q = ctx.Queue()
        self.task_qs, self.procs = [], []
        for wid in range(self.num_workers):
            tq = ctx.Queue()
            p = ctx.Process(
                target=_worker_main,
                args=(wid, cfg4, int(base_seed) + 1 + wid, worker_device,
                      tq, self.result_q),
                daemon=True)
            p.start()
            self.task_qs.append(tq)
            self.procs.append(p)

        # Barrier: every worker must have its env up before we dispatch anything,
        # otherwise a crash-on-startup surfaces as a mysterious hang mid-iteration.
        self.num_scenes = None
        ready = 0
        while ready < self.num_workers:
            try:
                tag, wid, payload = self.result_q.get(timeout=start_timeout)
            except queue.Empty:
                self.close()
                raise RuntimeError(
                    f"only {ready}/{self.num_workers} collection workers came up "
                    f"within {start_timeout:.0f}s (check OMG_PLANNER_DIR / "
                    "GADDPG_DIR are exported to the job, and that the GPU has "
                    "room for one CUDA context per worker)")
            if tag == "error":
                self.close()
                raise RuntimeError(f"collection worker {wid} failed to start:\n{payload}")
            self.num_scenes = payload
            ready += 1
        print(f"[parallel] {self.num_workers} collection workers up "
              f"(policy on {worker_device}); num_scenes={self.num_scenes}")

    def _get_one(self):
        while True:
            try:
                return self.result_q.get(timeout=self.job_timeout)
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(self.procs)
                        if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f"collection worker(s) died mid-iteration: {dead}")
                # alive, just slow — OMG can be. Keep waiting.

    def _fan_out(self, kind, run_dir, ckpt, pairs, seeds, beta, params, iteration):
        """Round-robin `pairs` over the workers and gather in JOB ORDER.

        Shared by `collect` and `evaluate`: the two differ only in what the
        worker calls per item, and returning by job index rather than completion
        order is what keeps either path independent of which worker finished
        first.
        """
        jobs = [(i, s, g, sd) for i, ((s, g), sd) in enumerate(zip(pairs, seeds))]
        buckets = [[] for _ in range(self.num_workers)]
        for k, j in enumerate(jobs):
            buckets[k % self.num_workers].append(j)
        # Dispatch to ALL workers, including empty buckets, so the reply count is
        # fixed at num_workers and a short iteration cannot leave the gather loop
        # waiting on a worker that was never given anything.
        for wid in range(self.num_workers):
            self.task_qs[wid].put((kind, str(run_dir), str(ckpt), float(beta),
                                   params, int(iteration), buckets[wid]))
        got = {}
        for _ in range(self.num_workers):
            tag, wid, payload = self._get_one()
            if tag == "error":
                raise RuntimeError(f"{kind} worker {wid} crashed:\n{payload}")
            for idx, a, b in payload:
                got[idx] = (a, b)
        return [got[k] for k in range(len(jobs))]

    def evaluate(self, run_dir, ckpt, pairs, params, iteration=0):
        """[(scene, gi), ...] -> [row, ...] in the order given.

        WHAT IS ACTUALLY GUARANTEED, AND WHAT IS NOT.
        GUARANTEED: `_eval_episode` draws no random numbers, every episode
        resets the sim to its own scene, and results are reassembled by JOB
        INDEX rather than completion order — so the row sequence, and therefore
        every rate, per-bin block and retry@k, does not depend on which worker
        finished first.
        NOT VERIFIED: bit-identity with the serial path. The equivalence test
        was attempted and abandoned (the 2-worker pool stalled in env
        construction), so this is an argument from the code, not a measurement.
        It is also bounded from below by GPU nondeterminism, which makes two
        SERIAL evals of the same checkpoint differ at some digit anyway — so
        "identical to serial" was never the right bar. Treat parallel eval as
        equivalent to re-running eval, not to replaying it.

        Eval was the second-largest cost of a run and the only serial one:
        measured on run 3's sizing, 742 episodes/iteration x 25 iterations is
        ~18.5 h on a node whose 20 collection workers sit idle throughout.
        """
        if not pairs:
            return []
        pairs = [(int(p[0]), int(p[1])) for p in pairs]
        out = self._fan_out("eval", run_dir, ckpt, pairs, [0] * len(pairs),
                            0.0, params, iteration)
        return [row for row, _ in out]

    def collect(self, run_dir, ckpt, pairs, seeds, beta, params, iteration):
        """`pairs` is [(scene_idx, grasp_idx), ...] — the Phase-5 work unit. A
        bare list of scene ids is read as slot 0, so a Phase-4-shaped call still
        works."""
        if not pairs:
            return []
        pairs = [(int(p), 0) if isinstance(p, (int, np.integer))
                 else (int(p[0]), int(p[1])) for p in pairs]
        return self._fan_out("collect", run_dir, ckpt, pairs, seeds, beta,
                             params, iteration)

    def close(self):
        for tq in self.task_qs:
            try:
                tq.put(None)
            except Exception:                            # noqa: BLE001, S110
                pass
        for p in self.procs:
            p.join(timeout=15.0)
            if p.is_alive():
                p.terminate()
