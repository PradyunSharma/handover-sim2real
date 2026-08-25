"""
Phase-4: DAgger (Ross et al., 2011, Algorithm 3.1) as a single Python loop.

Phase 1/2 drove DAgger from bash (`run_dagger.sh` / `run_dagger_act.sh`) over
one-shot collector + trainer scripts. Phase 4 runs the whole algorithm in one
process: collect -> aggregate -> refit -> evaluate -> repeat, with the env, the
planner and the model all held in memory across iterations.

The learner is the Phase-1 SINGLE-FRAME policy by default. The loop itself is
policy-agnostic — point TRAIN.train_cfg at act_phase2.yaml and the same loop
drives the Phase-2 chunking policy; the kind is inferred from that config.

    D   <- expert demonstrations (TRAIN.base_train_h5)     # the beta_1 = 1 iteration
    pi_1 <- train(D)                                       # iteration 0, the "base run"
    for i = 1..N:
        pi_i^mix = beta_i*pi* + (1-beta_i)*pi_i
        sample m = episodes_per_iter T-step trajectories with pi_i^mix
        D_i = {(s, pi*(s)) : s visited by pi_i^mix}
        D <- D u D_i                                       # aggregate; nothing is dropped
        pi_{i+1} <- train(D)                               # fresh fit on the union (FTL)
        evaluate pi_{i+1} on the held-out eval scenes
    keep the BEST-on-eval policy and the LAST policy

Phase-4-specific collection rules (see handover_sim2real/dagger/collector.py):
  * OMG is re-planned from the CURRENT drifted config at EVERY step; the label is
    that plan's first waypoint.
  * The plan runs to the pre-grasp standoff AND the reach beyond it.
  * No standoff-plane cutoff — the whole approach is labelled.
  * A gripper-CLOSE label is emitted once the EE is within
    (close_pos_thresh, close_rot_thresh) of the plan's grasp pose.
  * Grasp candidates come from Phase-3's filtered grasp dict.

Usage:
    python examples/train_dagger_phase4.py \\
        --cfg-file examples/configs/dagger_phase4.yaml \\
        --run-name dagger4_run1

    # resume: re-run the SAME command. Completed iterations are skipped, a
    # finished-but-untrained collection is reused, and an interrupted training
    # continues from its last.pt.

Run dir (output/dagger_runs/<run_name>/):
    config.yaml       the resolved Phase-4 config
    state.json        completed iterations + best-so-far (drives resume)
    dagger_log.csv    one row per iteration (data + eval metrics)
    data/             dagger_iter_NN.h5  — D_i, in the Phase-1/2 BC schema
    iters/iter_NN/    a full policy run dir per iteration
    best/  last/      standalone snapshots, loadable by rollout_bc_policy.py
                      (or rollout_act_policy.py for an ACT learner)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling example modules

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from handover_sim2real.regrasp_bc import (
    ACTTrainer,
    BCDataset,
    BCSequenceDataset,
    BCTrainer,
    Normalizer,
    compute_normalization_stats,
)
from handover_sim2real.regrasp import (
    build_box_params,
    CollectParams,
    EvalParams,
    build_policy,
    build_sim_cfg,
    build_sim_context,
    collect_iteration,
    dart_alpha_at,
    dart_scaled_sigma,
    evaluate_policy,
    export_run_dir,
    GraspRegistry,
    load_grasp_pin_table,
    load_policy_runner,
    policy_kind,
)
from handover_sim2real.regrasp.evaluator import (               # noqa: E402
    aggregate_eval_rows, eval_jobs, eval_num_grasps,
)
from handover_sim2real.regrasp.pregrasp import forward_dist_default
from handover_sim2real.regrasp.setup import (                     # noqa: E402
    build_regrasp_context, expand_config_paths, scene_pools,
)


# ── config plumbing ──────────────────────────────────────────────────────────

def load_yaml(path) -> dict:
    """Read a config and expand `${REGRASP_DATA}` in every path it contains.

    The sbatch scripts set that variable to `$SCRATCH_ROOT/output` so the HDF5
    shards stay off /home's 30 GB quota; it defaults to `output`, so a config
    with no `$` in it — every Regrasp config before run 2 — is unaffected.
    """
    with open(path) as f:
        return expand_config_paths(yaml.safe_load(f))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def beta_at(i: int, d: dict, num_iters: int) -> float:
    """beta for DAgger iteration `i` (1-based).

    The paper's beta_1 = 1 iteration is the base run on the expert
    demonstrations, so iteration i here is the paper's iteration i+1 — hence the
    exponential schedule starts at p^1, not p^0.
    """
    schedule = str(d.get("beta_schedule", "indicator")).lower()
    if schedule == "indicator":
        return 0.0
    if schedule == "constant":
        return float(d.get("beta_start", 0.0))
    if schedule == "exponential":
        return float(d.get("beta_p", 0.5)) ** i
    if schedule == "linear":
        lo, hi = float(d.get("beta_start", 0.5)), float(d.get("beta_end", 0.0))
        if num_iters <= 1:
            return hi
        return lo + (hi - lo) * (i - 1) / (num_iters - 1)
    if schedule == "piecewise":
        # Two linear segments joined at `beta_knee` (a FRACTION of num_iters, so
        # the shape survives a change of num_iters): beta_start -> beta_mid over
        # the first segment, beta_mid -> beta_end over the second. A single
        # `linear` cannot express "hand over slowly, then hold a floor", which is
        # the response to run 3's collapse — reached_grasp fell 87 -> 45 over
        # iterations 10-15 as beta annealed to 0.10, and success followed it down.
        lo = float(d.get("beta_start", 1.0))
        mid = float(d.get("beta_mid", 0.5))
        hi = float(d.get("beta_end", 0.0))
        knee = float(d.get("beta_knee", 0.66))
        if num_iters <= 1:
            return hi
        # k is the LAST iteration of the first segment, so beta_at(k) == beta_mid
        # exactly and the second segment starts one step below it. Clamped to
        # [1, num_iters-1] so both segments always have somewhere to go.
        k = min(max(int(round(knee * num_iters)), 1), num_iters - 1)
        if i <= k:
            return lo + (mid - lo) * (i - 1) / max(k - 1, 1)
        return mid + (hi - mid) * (i - k) / max(num_iters - k, 1)
    raise ValueError(f"Unknown DAGGER.beta_schedule '{schedule}'")


def sample_scenes(pool: list[int], m: int, mode: str, rng, cursor: int) -> tuple[list[int], int]:
    """m scenes for one iteration. Returns (scenes, new cursor)."""
    m = min(int(m), len(pool))
    if mode == "sequential":
        idx = [(cursor + j) % len(pool) for j in range(m)]
        return [pool[j] for j in idx], (cursor + m) % len(pool)
    return [int(s) for s in rng.choice(pool, size=m, replace=False)], cursor


def sample_pairs(pool: list[int], m: int, num_grasps: int, mode: str, rng,
                 cursor: int, pin_table=None) -> tuple[list[tuple[int, int]], int]:
    """~m (scene, direction) pairs for one iteration. Returns (pairs, cursor).

    Draws scenes and rolls out EVERY direction of each, rather than drawing m
    independent pairs. The reason is what the conditioning has to learn: two
    rollouts that differ only in the commanded direction — same scene, same start
    state, same point cloud — are the only data that can teach the network the
    command matters, and drawing them together puts them in the same iteration
    and the same batches. Independent draws would spread that contrast across
    the whole run.

    PER-SCENE COUNTS, NOT A UNIFORM ONE. A Regrasp table mixes scenes that can
    supply two separated directions with scenes that can only supply one (471 and
    146 respectively on s0/train). Expanding `range(num_grasps)` uniformly would
    either invent a slot 1 that does not exist, or — with `num_grasps` read from
    `GraspPinTable.num_grasps`, which is a MIN — collapse to slot 0 everywhere
    and silently discard every paired second demonstration. So the expansion asks
    the table per scene, and `m` becomes a target rather than an exact count.
    """
    num_grasps = max(int(num_grasps), 1)
    n_scenes = max(int(m) // num_grasps, 1)
    scenes, cursor = sample_scenes(pool, n_scenes, mode, rng, cursor)
    if pin_table is None:
        return [(int(s), g) for s in scenes for g in range(num_grasps)], cursor
    return [(int(s), g) for s in scenes
            for g in range(max(pin_table.num_grasps_for(int(s)), 1))], cursor


def base_files(trn: dict) -> list[str]:
    """`TRAIN.base_train_h5` as a LIST, accepting one path or several.

    Base collection is serial (~8 s/episode) and is the long pole of a run — at
    three demonstrations per bin it is ~10 h in one process. `collect_regrasp_demos
    --shard i/n` splits it across concurrent processes, and each writes its own
    HDF5. Taking a list here means those pieces are read directly and never
    merged: no 2.6 GB copy, no merge step to get wrong, and `episode_counts()`
    still prints one line per file so a missing shard is visible rather than
    silently reducing the training set.

    Order is preserved as written, and every path is checked here rather than at
    the first batch — a typo in shard 3 of 4 would otherwise train happily on
    three quarters of the data and report nothing.
    """
    v = trn["base_train_h5"]
    files = [v] if isinstance(v, str) else list(v)
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(
            "TRAIN.base_train_h5 lists files that do not exist:\n  "
            + "\n  ".join(missing)
            + "\nA shard that failed to collect must be re-run, not dropped: "
              "training on the survivors is a different and smaller experiment.")
    return files


# ── per-iteration training (a fresh fit on the whole aggregate) ──────────────

def run_eval(pool, sim, run_dir, ckpt, device, eval_scenes, eval_params,
             pin_table, iteration=0):
    """Evaluate one checkpoint, on the worker pool when there is one.

    THE POOL IS THE SAME ONE COLLECTION USES, and it is otherwise IDLE here:
    collection has finished, the refit is done, and eval is the only thing left
    in the iteration. Measured on run 3's sizing that idle time is ~44 min an
    iteration and ~18.5 h over the run, which made eval the second-largest cost
    of a DAgger run and the only serial one.

    ORDER-STABLE, NOT BIT-IDENTICAL. `_eval_episode` draws no random numbers
    and every episode resets the sim to its own scene, and rows come back in JOB
    ORDER, so no metric depends on which worker finished first. That is the
    property this needs. It is NOT a claim of bit-identity with the serial path:
    that was never measured, and GPU nondeterminism means two serial evals of
    one checkpoint already disagree at some digit. Read a parallel eval as
    equivalent to re-running eval, not to replaying it.

    Falls back to the serial loop when there is no pool, which is what a
    `--num-workers 0` run and every local smoke test take.
    """
    if pool is None:
        runner, _ = load_policy_runner(Path(run_dir), device, ckpt=ckpt)
        out = evaluate_policy(sim, runner, eval_scenes, params=eval_params,
                              pin_table=pin_table)
        del runner
        if device != "cpu":
            torch.cuda.empty_cache()
        return out
    jobs = eval_jobs(eval_scenes, pin_table)
    rows = pool.evaluate(str(run_dir), str(ckpt), jobs, eval_params,
                         iteration=iteration)
    return aggregate_eval_rows(rows, eval_params, eval_num_grasps(pin_table))


def train_on_aggregate(train_cfg: dict, train_files: list[str], val_h5: str | None,
                       run_dir: Path, *, epochs: int, device: str,
                       num_workers: int, init_from: Path | None = None,
                       init_ckpt: str = "last", seed: int = 0) -> Path:
    """Train one model on D = base_train_h5 u all D_i collected so far.

    Works for either learner: the Phase-1 single-frame policy (BCDataset +
    BCTrainer) or the Phase-2 ACT policy (windowed BCSequenceDataset +
    ACTTrainer). Which one is decided by the config, not by a separate flag.

    `init_from` (a previous iteration's run dir) warm-starts the weights AND
    reuses that run's normalizer — the network's output scale is defined by the
    normalizer, so warm-starting under freshly recomputed stats would silently
    reinterpret the head. Pass None for the paper-faithful fresh fit.

    `init_ckpt` picks WHICH of that run's checkpoints seeds the chain:
        "last"  the end of the cosine schedule — the most converged fit on the
                aggregate, and what run 20 used.
        "best"  the epoch with the lowest val loss, i.e. early-stopped.
    "best" is the one that matches the rest of the loop: EVAL.ckpt is `best`, so
    collection and evaluation already run on best.pt. With "last" the weights
    carried forward are not the weights the next shard was collected with, which
    makes the chain and the data come from two different policies. Falls back to
    last.pt when best.pt is absent (a run with no val set never writes one).

    Returns the run dir. Resumes automatically from checkpoints/last.pt.
    """
    cfg = copy.deepcopy(train_cfg)
    cfg["DATA"]["train_h5"] = train_files if len(train_files) > 1 else train_files[0]
    cfg["DATA"]["val_h5"] = val_h5
    cfg["TRAIN"]["num_epochs"] = int(epochs)
    cfg["TRAIN"]["device"] = device
    cfg["TRAIN"]["num_workers"] = int(num_workers)
    cfg["TRAIN"]["seed"] = int(seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    set_seed(seed)

    resume_ckpt = run_dir / "checkpoints" / "last.pt"
    resume = resume_ckpt.exists()
    norm_path = run_dir / "normalization.npz"

    if resume and norm_path.exists():
        normalizer = Normalizer.load(norm_path)
        print(f"[train] resuming {run_dir.name} from {resume_ckpt.name}")
    elif init_from is not None:
        normalizer = Normalizer.load(Path(init_from) / "normalization.npz")
        print(f"[train] warm start from {init_from} (reusing its normalizer)")
    else:
        print(f"[train] computing normalization over {len(train_files)} file(s) ...")
        normalizer = compute_normalization_stats(cfg["DATA"]["train_h5"])
    normalizer.save(norm_path)

    kind = policy_kind(cfg)
    if kind == "bc":
        # Auxiliary goal-grasp target (run 13), only when the head exists and the
        # weight is nonzero. "auto" resolves each shard's pin table from its own
        # attrs — the base h5 has always carried it and the DAgger shards do from
        # run 12 on, which matters here because the aggregate mixes both.
        goal_table = None
        if bool(cfg["MODEL"].get("aux_head", False)) and \
                float(cfg.get("LOSS", {}).get("aux_weight", 0.0)) > 0.0:
            goal_table = cfg["DATA"].get("grasp_pin_table") or "auto"
        # REGRASP conditioning. No table needed: every episode carries its own
        # `d_world` attr, so the dataset reads its own command and a rebuilt pin
        # table cannot retarget an existing collection.
        direction_cond = bool(cfg["MODEL"].get("direction_cond", True))
        # Perturb the command by this much, TRAIN ONLY. Weighting or perturbing
        # val would change what val_loss means across epochs.
        d_noise = float(cfg["DATA"].get("d_noise_deg", 0.0))
        # Reach-tail oversampling (run 14). TRAIN ONLY — weighting val would
        # change what val_loss means and break comparability across runs. Under
        # conditioning it matters more, not less: the four grasps of a scene look
        # alike during the free approach and only diverge in the reach, so the
        # reach tail is where most of the conditioning signal lives.
        train_ds = BCDataset(cfg["DATA"]["train_h5"], normalizer=normalizer,
                             goal_table=goal_table,
                             direction_cond=direction_cond, d_noise_deg=d_noise,
                             reach_tail_weight=float(cfg["DATA"].get(
                                 "reach_tail_weight", 1.0)),
                             reach_tail=int(cfg["DATA"].get("reach_tail", 5)))
        val_ds = (BCDataset(val_h5, normalizer=normalizer, goal_table=goal_table,
                            direction_cond=direction_cond, d_noise_deg=0.0)
                  if val_h5 and os.path.exists(val_h5) else None)
    else:
        T = int(cfg["MODEL"]["history_len"])
        k = int(cfg["MODEL"]["chunk_len"])
        train_ds = BCSequenceDataset(cfg["DATA"]["train_h5"], history_len=T,
                                     chunk_len=k, normalizer=normalizer)
        val_ds = (BCSequenceDataset(val_h5, history_len=T, chunk_len=k,
                                    normalizer=normalizer)
                  if val_h5 and os.path.exists(val_h5) else None)
    print(f"[train] {kind} policy | D = {len(train_ds)} steps / "
          f"{train_ds.num_episodes} episodes")
    for path, ne in train_ds.episode_counts():
        print(f"          {path}: {ne} episodes")

    pin = device != "cpu"
    train_sampler = (WeightedRandomSampler(train_ds.sample_weights,
                                           num_samples=len(train_ds),
                                           replacement=True)
                     if getattr(train_ds, "sample_weights", None) else None)
    train_dl = DataLoader(train_ds, batch_size=int(cfg["TRAIN"]["batch_size"]),
                          shuffle=(train_sampler is None), sampler=train_sampler,
                          num_workers=int(num_workers),
                          pin_memory=pin, drop_last=True)
    val_dl = (DataLoader(val_ds, batch_size=int(cfg["TRAIN"]["batch_size"]),
                         shuffle=False, num_workers=int(num_workers),
                         pin_memory=pin, drop_last=False)
              if val_ds is not None else None)

    model = build_policy(cfg, normalizer)

    if not resume:
        if init_from is not None:
            ckpt_path = Path(init_from) / "checkpoints" / f"{init_ckpt}.pt"
            if not ckpt_path.exists():
                # Only reachable for init_ckpt="best" on a run with no val set.
                # Announced rather than silent: a chain seeded from last.pt when
                # best.pt was asked for is a different experiment.
                print(f"[train] {ckpt_path.name} absent — warm starting from "
                      f"last.pt instead")
                ckpt_path = Path(init_from) / "checkpoints" / "last.pt"
            payload = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(payload["model"])
            print(f"[train] warm start weights from {ckpt_path} "
                  f"(epoch {payload.get('epoch', '?')})")
        else:
            pc_pre = cfg.get("MODEL", {}).get("pc_pretrained")
            if pc_pre and os.path.exists(pc_pre):
                model.load_pretrained_pc_encoder(pc_pre)
            elif pc_pre:
                print(f"[train] WARNING pc_pretrained not found, PC encoder from "
                      f"scratch: {pc_pre}")
            else:
                # Deliberate (run 21): MODEL.pc_pretrained left empty. Logged so
                # the run's stdout distinguishes "chose a random encoder" from
                # "the checkpoint path was quietly dropped from the config".
                print("[train] MODEL.pc_pretrained empty — PointNet++ encoder "
                      "initialised FROM SCRATCH")

    trainer_cls = BCTrainer if kind == "bc" else ACTTrainer
    trainer = trainer_cls(model, train_dl, val_dl, cfg, run_dir=str(run_dir))
    if resume:
        trainer.resume_from(str(resume_ckpt))
    trainer.train()

    del trainer, model, train_dl, val_dl, train_ds, val_ds
    if device != "cpu":
        torch.cuda.empty_cache()
    return run_dir


# ── state / logging ─────────────────────────────────────────────────────────

def score_tuple(m: dict, select_on: str) -> list[float]:
    """Lexicographic selection score for "return best pi_i on validation".

    Primary is EVAL.select_on. Early in training every iteration scores 0
    success, so the tie-breaks matter, ordered by how much they demand:
    grasp_rate separates "closed on the object" from "closed on air", near_rate
    separates "closed at the right pose" from "closed anywhere", and the negated
    final EE->object distance separates "got close" from "never approached".
    Strictly-greater comparison keeps the EARLIEST iteration on a full tie.
    """
    dist = float(m.get("mean_dist", float("nan")))
    return [
        float(m.get(select_on, 0.0)),
        float(m.get("grasp_rate", 0.0)),
        float(m.get("near_rate", 0.0)),
        -(dist if np.isfinite(dist) else 1e9),
    ]


def maybe_update_best(state: dict, run_root: Path, run_dir: Path, iteration: int,
                      metrics: dict | None, select_on: str, ckpt: str) -> bool:
    """Publish `run_dir` as best/ if it beats the incumbent. Returns True if so."""
    if metrics is None:
        return False
    score = score_tuple(metrics, select_on)
    incumbent = (state.get("best") or {}).get("score_tuple")
    # A stored tuple of a different length is from a run made before the metric
    # set changed; its numbers are not comparable, so ignore it rather than let
    # Python's shorter-list-is-smaller rule silently decide the comparison.
    if incumbent is not None and len(incumbent) == len(score) and score <= list(incumbent):
        return False
    state["best"] = {"iter": iteration, "metric": select_on,
                     "score": score[0], "score_tuple": score,
                     "run_dir": str(run_dir)}
    export_run_dir(run_dir, run_root / "best", ckpt=ckpt,
                   note=f"DAgger iteration {iteration} — best {select_on}={score[0]:.4f}")
    print(f"  [best] new best {select_on}={score[0]:.4f} (iteration {iteration})")
    return True


def load_state(path: Path) -> dict:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"iterations": [], "best": None, "cursor": 0}


def _json_safe(o):
    """json.dump `default=` for numpy leaking into the state dict.

    `state["iterations"][k]["collect"]` is the collector's aggregate handed
    through verbatim, so ANY numpy value a future metric puts there would kill
    the run at the first save — which is exactly what run 18 hit: the DART Sigma
    estimator added two 6x6 ndarrays and json.dump raised four levels down
    (state -> iterations -> rec -> collect). Converting here rather than only
    fixing that one metric means the next one cannot reproduce it.

    Deliberately narrow: numpy scalars and arrays only. Anything else still
    raises, because a genuinely unserialisable object in the resume state is a
    bug worth surfacing, not worth stringifying.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):      # np.float64/int64/bool_ etc.
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_state(path: Path, state: dict) -> None:
    # tmp + atomic replace: if the dump raises, the PREVIOUS state.json is left
    # intact rather than truncated, so a crashed run still resumes.
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2, default=_json_safe)
    tmp.replace(path)


# Fixed column set. Rows are written by two call sites (the base iteration and
# the DAgger iterations) whose natural key sets differ; deriving fieldnames from
# whichever row lands first silently mis-aligns every later row against the
# header, so pin the columns here and let missing keys blank out.
#
# Grouped by the question each group answers — see examples/plot_dagger_run.py,
# which renders exactly these:
#   1. did it learn        eval rates + pose errors on held-out scenes
#   2. why did it fail     eval outcome breakdown (fractions, stack to 1)
#   3. is DAgger working   collection stats: how far the LEARNER got on its own
#   4. are labels sane     expert-label scale + coverage of the endgame
#   5. is the fit healthy  train/val loss of the refit on the growing aggregate
#   6. what did it cost    wall time split by phase
LOG_FIELDS = [
    # -- identity / schedule
    "iter", "beta", "m",
    # -- (3) collection: what the learner's own rollouts looked like
    "episodes", "steps", "skipped", "pinned",
    "reached_standoff", "reached_grasp", "reach_steps", "settle_steps",
    "omg_fail", "goal_switch", "expert_steps", "policy_closed",
    "policy_close_cmds", "dropped_tail", "dart", "dart_env_done",
    "dart_reach", "dart_reject",
    # DART-paper noise (DAGGER.dart_mode: dart_noise). `dart_sigma_trace` is
    # tr(Sigma_hat), the measured learner-supervisor error, and is logged in BOTH
    # modes — in a jolt run it is the counterfactual "what the noise would have
    # been". `dart_noise_steps` is how many expert steps were actually perturbed.
    "dart_sigma_trace", "dart_noise_steps", "dart_alpha",
    "mean_min_pos", "mean_min_rot", "mean_policy_close_step",
    "c_close_label", "c_policy_close", "c_max_steps", "c_env_done", "c_no_labels",
    "c_omg_fail0",
    # -- collection OUTCOME (DAGGER.outcome_check; 0 for runs that leave it off).
    # `c_*` above is the loop's terminal cause; these are what happened to the
    # handover, on the evaluator's taxonomy, as fractions of the kept episodes so
    # they stack to 1.0. `c_success_rate` is the same stable-grasp criterion as
    # `success_rate`, but measured on the beta MIXTURE, not on the policy alone.
    "c_success_rate",
    "co_grasp_ok", "co_grasp_miss", "co_no_release", "co_drop",
    "co_human_contact", "co_bench_timeout", "co_timeout",
    # -- the blind endgame (DAGGER.target: pregrasp; blank in every other run).
    # Where the feed-forward push landed relative to the grasp it could not see.
    # Read against `mean_min_pos`, which in that mode is the distance to the
    # PRE-GRASP: a small mean_min_pos with a large c_reach_pos means the policy
    # is arriving correctly and `forward_dist` is wrong, which is a one-line fix
    # rather than a failed run.
    "c_reach_pos", "c_reach_rot",
    # -- did a revisited scene still aim at the same grasp (pin verification)
    "revisits", "grasp_mismatch", "max_grasp_drift",
    # -- (4) labels
    "close_labels", "approach_labels", "mean_label_pos", "tiny_labels",
    # -- aggregate D the refit sees
    "aggregate_files", "D_episodes", "D_steps", "D_dagger_frac",
    # -- (5) the refit itself
    "epochs", "train_loss", "train_grip_acc", "val_loss", "val_grip_acc",
    "best_val_loss", "aux_pos_mm", "aux_rot_deg", "aux_pm_mm",
    # -- (1) eval
    "success_rate", "grasp_rate", "near_rate", "close_rate",
    "close_success_rate", "chance_rate", "missed_rate", "miss_given_chance",
    "box_chance_rate", "box_taken_rate", "box_missed_rate", "miss_given_box",
    "mean_box_steps", "mean_box_frac",
    # DAGGER.target: pregrasp only. `box_after_rate` is that mode's conversion
    # measure: of the episodes where the policy committed, how many had the
    # object between the open jaws AFTER the blind push. `box_chance_rate` is
    # near 0 there by construction — the policy stops 6.4 cm short, so the object
    # is never in the jaws while it is still deciding.
    "mean_reach_pos_err", "mean_reach_rot_err", "box_after_rate",
    "eval_min_pos", "eval_min_rot",
    "mean_dist", "mean_pos_err", "mean_rot_err", "mean_close_step",
    # -- PHASE 5. The three columns this phase exists to move.
    #
    # `cond_track` FIRST, because it decides whether the rest means anything: the
    # mean pairwise spread of the four final EE poses over the spread of the four
    # commanded grasps. Near 1 = the policy separates the conditions; near 0 = it
    # does the same thing whatever it is told, which is the multi-modal averaging
    # failure and makes regrasping inert however good success_rate looks. If it
    # sits below ~0.3 while near_rate stays flat, concatenation is not enough and
    # the conditioning should move to FiLM.
    #
    # `retry_at_k` is the regrasping headline: success given k attempts at
    # different grasps, in FPS order, derived from the same episodes at no extra
    # cost. It assumes each retry restarts from home, so read it as a ceiling.
    #
    # `succ_g*` / `near_g*` are the per-slot rates. Slot 0 is OMG's own pick, so
    # `succ_g0` is the column comparable with a Phase-4 run; the spread across
    # slots says how much harder the deliberately-separated grasps are.
    # Regrasp: direction metrics replace cond_track / near_g*. `dir_err` is the
    # headline (angle between the command and the achieved approach axis);
    # `bin_hit_rate` is the near_rate analogue; `bin_diag_rate` collapsing means
    # the policy goes the same way whatever it is told.
    "dir_err", "dir_err_median", "dir_track", "sector_err",
    "bin_hit_rate", "bin_diag_rate", "cond_sep",
    "retry_at_1", "retry_at_2", "retry_at_3", "retry_at_4",
    # WHICH DIRECTION EACH RUNG OF THE LADDER WAS. The ladder walks pin slots in
    # ascending bin order, so rung k is a MIXTURE across scenes rather than one
    # direction — `retry_bin_k` is the modal bin and `retry_bin_frac_k` its
    # share. A share well under 1.0 means the ladder is not a fixed direction
    # sequence and the retry curve must not be read as "then it tried +y".
    "retry_bin_1", "retry_bin_2", "retry_bin_3", "retry_bin_4",
    "retry_bin_frac_1", "retry_bin_frac_2", "retry_bin_frac_3", "retry_bin_frac_4",
    "succ_bin_0", "succ_bin_1", "succ_bin_2",
    "succ_bin_3", "succ_bin_4", "succ_bin_5",
    "n_bin_0", "n_bin_1", "n_bin_2", "n_bin_3", "n_bin_4", "n_bin_5",
    "succ_g0", "succ_g1",
    # -- (2) eval outcome breakdown, fractions of the eval set
    "f_grasp_ok", "f_grasp_miss", "f_no_release", "f_drop", "f_timeout",
    "f_human_contact",
    # ...and the same breakdown over the FAILURES only, which is the profile
    # rather than the level. See EVAL_FAIL_REASONS.
    "ff_grasp_miss", "ff_no_release", "ff_drop", "ff_timeout", "ff_human_contact",
    # -- (6)
    "is_best", "collect_s", "train_s", "eval_s", "wall_s",
]
# NOTE: ~180 more columns are APPENDED to this list below, once the per-bin
# helpers exist. They are appended rather than interleaved because the block
# would otherwise bury the ~130 columns a human reads by eye.

# collector `reason` -> column, and evaluator `reason` -> column. Anything not
# listed is dropped from the CSV rather than silently mis-filed, so a new reason
# string shows up as a gap in the stack plot instead of a wrong count.
COLLECT_REASONS = {"CLOSE_LABEL": "c_close_label", "POLICY_CLOSE": "c_policy_close",
                   "MAX_STEPS": "c_max_steps", "ENV_DONE": "c_env_done",
                   "NO_LABELS": "c_no_labels", "OMG_FAIL_STEP0": "c_omg_fail0"}
EVAL_REASONS = {"GRASP_OK": "f_grasp_ok", "GRASP_MISS": "f_grasp_miss",
                "NO_RELEASE": "f_no_release", "DROP": "f_drop",
                "TIMEOUT": "f_timeout", "HUMAN_CONTACT": "f_human_contact"}
# Collection outcomes (DAGGER.outcome_check). Same taxonomy as EVAL_REASONS plus
# BENCH_TIMEOUT, which eval cannot produce: eval stops at EVAL.max_steps (50) well
# inside the benchmark's own 86.7-step limit, whereas collection now runs to 70 and
# an episode that also stalls can be killed by the benchmark instead.
COLLECT_OUTCOMES = {"GRASP_OK": "co_grasp_ok", "GRASP_MISS": "co_grasp_miss",
                    "NO_RELEASE": "co_no_release", "DROP": "co_drop",
                    "HUMAN_CONTACT": "co_human_contact",
                    "BENCH_TIMEOUT": "co_bench_timeout", "TIMEOUT": "co_timeout"}
# The same taxonomy conditioned on FAILURE — denominator is the episodes that did
# not come away with the object, not the eval set. `f_timeout` answers "how much
# of the eval set timed out", which moves whenever success moves; `ff_timeout`
# answers "of the ones that failed, how many timed out", which is the failure
# PROFILE and is what tells `+z` apart from `+x` when both succeed 20% of the
# time. GRASP_OK is absent by construction: it is the success.
EVAL_FAIL_REASONS = {"GRASP_MISS": "ff_grasp_miss", "NO_RELEASE": "ff_no_release",
                     "DROP": "ff_drop", "TIMEOUT": "ff_timeout",
                     "HUMAN_CONTACT": "ff_human_contact"}

# ── PER-BIN COLUMNS ──────────────────────────────────────────────────────────
# The Regrasp figures draw every eval panel once per commanded direction, so the
# whole rate family is logged per bin with a `_b{b}` suffix. `n_bins` is the
# BIN SET's size (6), not the number this dataset can reach (4): a fixed header
# keeps two runs diffable and lets a plotter ask for a column without first
# asking whether the run happened to command that direction. The two dead bins
# write blanks.
N_BINS = 6
BIN_EVAL_KEYS = (
    "n",                                                    # episodes in the bin
    "close_rate", "near_rate", "grasp_rate", "success_rate",         # nested
    "box_chance_rate", "box_taken_rate", "miss_given_box",           # opportunity
    "close_success_rate", "mean_box_frac",
    "eval_min_pos", "eval_min_rot", "mean_pos_err", "mean_rot_err",  # approach
    "dir_err", "dir_track", "sector_err", "bin_hit_rate", "bin_diag_rate",
)
BIN_COLLECT_KEYS = ("episodes", "reached_standoff", "reached_grasp",
                    "policy_closed", "success")


def bin_eval_fields() -> list[str]:
    """Every per-bin eval column, in a fixed order. Shared with
    eval_regrasp_run.py so the two CSVs stay spliceable column-for-column."""
    out = []
    for b in range(N_BINS):
        out += [f"{k}_b{b}" for k in BIN_EVAL_KEYS]
        out += [f"{c}_b{b}" for c in EVAL_REASONS.values()]
        out += [f"{c}_b{b}" for c in EVAL_FAIL_REASONS.values()]
    return out


def bin_collect_fields() -> list[str]:
    """Every per-bin COLLECTION column — the learner's own rollouts, split by the
    direction they were driven under."""
    return [f"c_{k}_b{b}" for b in range(N_BINS) for k in BIN_COLLECT_KEYS]


# Deferred to here because the helpers above have to exist first (see the note
# at the end of the LOG_FIELDS literal).
LOG_FIELDS += bin_collect_fields() + bin_eval_fields()


def reason_columns(reasons: dict, mapping: dict, denom: int | None = None) -> dict:
    """Spread a {reason: count} dict over its fixed columns.

    `denom` converts to fractions (eval, so the categories stack to 1.0); None
    keeps raw counts (collection, where `m` varies with the config).
    """
    out = {v: 0 for v in mapping.values()}
    for reason, count in (reasons or {}).items():
        # `_status_name` can OR two failures into "DROP|HUMAN_CONTACT"; charge
        # the episode to each so no episode goes missing from the breakdown.
        for part in str(reason).split("|"):
            col = mapping.get(part)
            if col is not None:
                out[col] += count
    if denom:
        out = {k: round(v / denom, 4) for k, v in out.items()}
    return out


def _r(value, nd: int = 4):
    """Round for the CSV, but leave a missing/NaN value BLANK rather than writing
    'nan' — a blank cell plots as a gap, 'nan' can parse as a data point."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return round(v, nd) if np.isfinite(v) else ""


def eval_columns(m: dict | None) -> dict:
    """The eval half of a log row: rates, pose errors, outcome breakdown."""
    if not m:
        return {}
    out = {k: _r(m.get(k)) for k in (
        "success_rate", "grasp_rate", "near_rate", "close_rate",
        "close_success_rate", "chance_rate", "missed_rate", "miss_given_chance",
        "box_chance_rate", "box_taken_rate", "box_missed_rate", "miss_given_box",
        "mean_box_steps", "mean_box_frac",
        "mean_reach_pos_err", "mean_reach_rot_err", "box_after_rate",
        "eval_min_pos", "eval_min_rot",
        "mean_dist", "mean_pos_err", "mean_rot_err", "mean_close_step",
        # Regrasp — blank rather than 0 for bins a run never commanded, so an
        # unvisited direction does not read as a genuine zero success rate.
        "dir_err", "dir_err_median", "dir_track", "sector_err",
        "bin_hit_rate", "bin_diag_rate", "cond_sep",
        "retry_at_1", "retry_at_2", "retry_at_3", "retry_at_4",
        "retry_bin_frac_1", "retry_bin_frac_2", "retry_bin_frac_3",
        "retry_bin_frac_4",
        "succ_bin_0", "succ_bin_1", "succ_bin_2",
        "succ_bin_3", "succ_bin_4", "succ_bin_5",
        "n_bin_0", "n_bin_1", "n_bin_2", "n_bin_3", "n_bin_4", "n_bin_5",
        "succ_g0", "succ_g1")}
    # The modal bin of each rung is an INDEX, not a rate — blank when the rung
    # collected nothing, never 0, since 0 is a real bin (`+x`).
    for k in range(1, 5):
        v = m.get(f"retry_bin_{k}")
        out[f"retry_bin_{k}"] = "" if v is None else int(v)
    out.update(reason_columns(m.get("reasons"), EVAL_REASONS, denom=m.get("n") or None))
    out.update(reason_columns(m.get("reasons_fail"), EVAL_FAIL_REASONS,
                              denom=m.get("n_fail") or None))
    # ---- the same family, once per commanded direction ----------------------
    # `n_b{b}` stays an integer count (it is the denominator every other column
    # in the block is over, and reading it as a rate would be a mistake); the
    # rest go through `_r`, so an unvisited bin blanks rather than reading 0.
    for b in range(N_BINS):
        out[f"n_b{b}"] = int(m.get(f"n_b{b}", 0) or 0)
        for k in BIN_EVAL_KEYS[1:]:
            out[f"{k}_b{b}"] = _r(m.get(f"{k}_b{b}"))
        # A bin with no episodes gets BLANKS, not the zeros `reason_columns`
        # returns when handed no denominator — a stacked area reading 0.0 across
        # every category says "nothing failed", which is the opposite of "nothing
        # was measured".
        nb, nf = out[f"n_b{b}"], int(m.get(f"n_fail_b{b}", 0) or 0)
        for c in EVAL_REASONS.values():
            out[f"{c}_b{b}"] = ""
        for c in EVAL_FAIL_REASONS.values():
            out[f"{c}_b{b}"] = ""
        if nb:
            out.update({f"{c}_b{b}": v for c, v in reason_columns(
                m.get(f"reasons_b{b}"), EVAL_REASONS, denom=nb).items()})
        if nf:
            out.update({f"{c}_b{b}": v for c, v in reason_columns(
                m.get(f"reasons_fail_b{b}"), EVAL_FAIL_REASONS, denom=nf).items()})
    return out


def collect_columns(c: dict) -> dict:
    """The collection half: how far the LEARNER's own rollouts got, and whether
    the expert labels on them are still usable."""
    return {
        "episodes": c["episodes"], "steps": c["steps"], "skipped": c["skipped"],
        "pinned": c.get("pinned", -1),
        "reached_standoff": c.get("reached_standoff", -1),
        "reached_grasp": c.get("reached_grasp", -1),
        "reach_steps": c.get("n_reach_steps", -1),
        "settle_steps": c.get("n_settle_steps", -1),
        "omg_fail": c["n_omg_fail"], "goal_switch": c["n_goal_switch"],
        "expert_steps": c["n_expert_steps"], "policy_closed": c["policy_closed"],
        "policy_close_cmds": c.get("n_policy_close_cmds", -1),
        "dropped_tail": c.get("n_dropped_tail", -1),
        "dart": c.get("n_dart", -1),
        "dart_env_done": c.get("n_dart_env_done", -1),
        "dart_reach": c.get("n_dart_reach", -1),
        "dart_reject": c.get("n_dart_reject", -1),
        "dart_sigma_trace": _r(c.get("dart_sigma_trace"), 6),
        "dart_noise_steps": c.get("n_dart_noise", -1),
        "dart_alpha": _r(c.get("dart_alpha"), 4),
        "mean_min_pos": _r(c.get("mean_min_pos")),
        "mean_min_rot": _r(c.get("mean_min_rot")),
        "c_reach_pos": _r(c.get("mean_reach_pos_err")),
        "c_reach_rot": _r(c.get("mean_reach_rot_err")),
        "mean_policy_close_step": _r(c.get("mean_policy_close_step"), 2),
        "close_labels": c.get("n_close_labels", -1),
        "approach_labels": c.get("n_approach_labels", -1),
        "mean_label_pos": _r(c.get("mean_label_pos")),
        "tiny_labels": c.get("n_tiny_labels", -1),
        "revisits": c.get("n_revisits", -1),
        "grasp_mismatch": c.get("n_grasp_mismatch", -1),
        "max_grasp_drift": _r(c.get("max_grasp_drift")),
        **reason_columns(c.get("reasons"), COLLECT_REASONS),
        # Fractions of KEPT episodes (denom=episodes), so the co_* block stacks to
        # 1.0 and is directly readable against the eval f_* block. All zero when
        # DAGGER.outcome_check is off, which is what runs 1-17 record.
        "c_success_rate": _r(
            (c.get("success", 0) / c["episodes"]) if c.get("episodes") else None),
        **reason_columns(c.get("outcomes"), COLLECT_OUTCOMES,
                         denom=c.get("episodes") or None),
        # ---- the same counters split by the direction driven ----------------
        # Raw counts, not fractions: `c_episodes_b{b}` is the denominator, and
        # the plotter divides. -1 for a bin with no episodes so a reused shard
        # (which reports no per-bin breakdown at all) is distinguishable from a
        # direction that genuinely collected nothing.
        **{f"c_{k}_b{b}": int((c.get("per_bin", {}).get(b) or {}).get(k, -1))
           for b in range(N_BINS) for k in BIN_COLLECT_KEYS},
    }


def dataset_size(files: list[str]) -> tuple[int, int]:
    """(episodes, steps) across the HDF5 files the refit will train on."""
    n_ep = n_steps = 0
    for path in files:
        if not path or not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            for name in f:
                grp = f[name]
                if isinstance(grp, h5py.Group) and "num_steps" in grp.attrs:
                    n_ep += 1
                    n_steps += int(grp.attrs["num_steps"])
    return n_ep, n_steps


def read_train_log(run_dir: Path) -> dict:
    """Last-epoch and best metrics from an iteration's own per-epoch log.csv.

    Read back from disk rather than returned by the trainer so the columns are
    filled on RESUME too, when training is skipped entirely.
    """
    path = Path(run_dir) / "log.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    def get(row, key):
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return float("nan")

    last = rows[-1]
    vals = [get(r, "val_total") for r in rows]
    finite = [v for v in vals if np.isfinite(v)]
    return {
        "epochs": len(rows),
        "train_loss": _r(get(last, "train_total")),
        "train_grip_acc": _r(get(last, "train_gripper_acc")),
        "val_loss": _r(get(last, "val_total")),
        "val_grip_acc": _r(get(last, "val_gripper_acc")),
        "best_val_loss": _r(min(finite)) if finite else "",
        # Auxiliary goal-grasp head (run 13). NaN — and so an empty CSV cell —
        # on every run without the head, which is what keeps the column harmless
        # in older runs' plots.
        "aux_pos_mm": _r(get(last, "val_aux_pos_mm")),
        "aux_rot_deg": _r(get(last, "val_aux_rot_deg")),
        "aux_pm_mm": _r(get(last, "val_aux_pm_mm")),
    }


def log_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg-file", default="examples/configs/dagger_phase4.yaml")
    p.add_argument("--run-name", default=None,
                   help="subdir of --out-root; default: timestamped")
    p.add_argument("--out-root", default="output/dagger_runs")
    p.add_argument("--num-iters", type=int, default=None, help="override DAGGER.num_iters")
    p.add_argument("--episodes-per-iter", type=int, default=None,
                   help="override DAGGER.episodes_per_iter (m in the paper)")
    p.add_argument("--base-run", default=None,
                   help="override TRAIN.base_run: start from an existing ACT run")
    p.add_argument("--device", default=None, help="override TRAIN.device")
    p.add_argument("--seed", type=int, default=None, help="override DAGGER.seed")
    p.add_argument("--num-workers", type=int, default=1,
                   help="parallel COLLECTION worker processes. 1 = the original "
                        "serial loop. Set ~ --cpus-per-task minus 2, and prefer a "
                        "value that DIVIDES DAGGER.episodes_per_iter (at m=100, 20 "
                        "gives 5 episodes each; 16 gives 7 to some workers).")
    p.add_argument("--worker-device", default="cuda", choices=["cpu", "cuda"],
                   help="device for a collection worker's policy. MUST be cuda: "
                        "PointNet++'s furthest_point_sample has no CPU kernel. "
                        "Left as a flag only so the failure is explicit.")
    p.add_argument("--no-eval", action="store_true", help="disable evaluation (EVAL.every=0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg4 = load_yaml(args.cfg_file)
    sim_cfg_d, dag, trn, ev = cfg4["SIM"], cfg4["DAGGER"], cfg4["TRAIN"], cfg4["EVAL"]

    if args.num_iters is not None:
        dag["num_iters"] = args.num_iters
    if args.episodes_per_iter is not None:
        dag["episodes_per_iter"] = args.episodes_per_iter
    if args.base_run is not None:
        trn["base_run"] = args.base_run
    if args.device is not None:
        trn["device"] = args.device
    if args.seed is not None:
        dag["seed"] = args.seed
    if args.no_eval:
        ev["every"] = 0

    device = str(trn.get("device", "cuda"))
    seed = int(dag.get("seed", 0))
    num_iters = int(dag["num_iters"])
    set_seed(seed)

    run_name = args.run_name or time.strftime("dagger4_%Y%m%d_%H%M%S")
    run_root = Path(args.out_root) / run_name
    (run_root / "data").mkdir(parents=True, exist_ok=True)
    (run_root / "iters").mkdir(exist_ok=True)
    with (run_root / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg4, f, sort_keys=False)

    state = load_state(run_root / "state.json")
    done_iters = {int(r["iter"]): r for r in state["iterations"]}

    # ----- simulator (one env: OMG for collection, benchmark for eval) -----
    sim_cfg = build_sim_cfg(sim_cfg_d)
    sim = build_sim_context(sim_cfg, sim_cfg_d, seed=seed)

    # One committed grasp per scene, shared with the demonstrations. Without it
    # OMG re-selects its goal on every replan and the target can move mid-episode.
    # Loaded BEFORE the pools because its key set is also the list of scenes the
    # expert can plan for, which is what both pools should be drawn from.
    pin_table = load_grasp_pin_table(
        sim_cfg_d.get("grasp_pin_table"),
        match_tol=float(sim_cfg_d.get("grasp_pin_match_tol", 0.02)),
        sim_cfg_block=sim_cfg_d)
    usable = set(pin_table.entries) if pin_table is not None else None
    # Phase 5: how many pinned grasps every scene carries. The sampler expands
    # each drawn scene into this many episodes and the evaluator scores each eval
    # scene under all of them, so it multiplies both `episodes_per_iter`'s
    # interpretation and the eval cost. 1 makes the loop behave exactly as Phase
    # 4 did, which is what a Phase-4 pin table gives.
    # MAX, not `num_grasps` (a MIN, which reads 1 on a Regrasp table because the
    # table mixes 1- and 2-direction scenes). With 1 here, sample_pairs computes
    # n_scenes = m // 1 = m and then expands each scene to its real direction
    # count, so `episodes_per_iter: 200` silently produced 354 episodes — 1.77x
    # the requested budget, and roughly double the wall clock.
    num_grasps = pin_table.max_grasps if pin_table is not None else 1

    # Scenes whose EXPERT is broken — the planned trajectory collides with the
    # object while translating into the pre-grasp pose, so the demonstration
    # terminated in a benchmark failure (examples/filter_demos.py writes this
    # list). Excluding them here removes them from the collection pool AND the
    # eval set: DAgger cannot learn anything useful on a scene where the expert
    # crashes, and scoring the policy on one caps the achievable rate for no
    # reason. NOTE the eval set then covers an easier subset — say so when
    # reporting, it is not comparable to a number over the full split.
    excl_path = sim_cfg_d.get("exclude_scenes")
    if excl_path:
        with open(excl_path) as f:
            excluded = {int(s) for s in json.load(f)}
        before = len(usable) if usable is not None else sim.num_scenes
        usable = (usable or set(range(sim.num_scenes))) - excluded
        n_excluded = before - len(usable)
        print(f"[exclude_scenes] {excl_path}: dropping {n_excluded} of "
              f"{before} scenes whose expert demonstration failed")
    else:
        n_excluded = 0

    pool, eval_scenes = scene_pools(sim.num_scenes, ev, usable=usable)

    train_cfg = load_yaml(trn["train_cfg"])
    # TRAIN.pc_pretrained overrides MODEL.pc_pretrained in the train config. Same
    # mechanism examples/train_bc.py exposes as `--pc-pretrained none`: an empty
    # value means initialise the PointNet++ encoder RANDOMLY instead of from the
    # CVPR2023 state-feat checkpoint. Exposed here so a run can make that choice
    # without forking a 230-line train config for one key — the key is absent by
    # default, so every existing run is untouched.
    #
    # Only iteration 0 is affected. With TRAIN.train_from_scratch: false the later
    # iterations warm-start from the previous iteration's checkpoint, and that
    # path never consults pc_pretrained.
    if "pc_pretrained" in trn:
        train_cfg.setdefault("MODEL", {})["pc_pretrained"] = trn["pc_pretrained"] or None
        print(f"[cfg] TRAIN.pc_pretrained override -> "
              f"{trn['pc_pretrained'] or 'NONE (encoder from scratch)'}")
    learner = policy_kind(train_cfg)   # "bc" (single frame) | "act" (chunking)

    # THE Phase-5 consistency check. With several pinned grasps per scene and an
    # unconditioned policy, the same observation carries several different expert
    # labels and the regression can only predict their mean — a policy that is
    # worse than a Phase-4 one and gives no hint why. Cheap to check, expensive
    # to discover 15 hours in.
    model_cfg = train_cfg.get("MODEL", {})
    # A RENAMED KEY MUST NOT FAIL OPEN. `.get("grasp_cond", False)` on a config
    # that no longer has the key returns False, the guard never fires, and the
    # check that exists specifically to stop an unconditioned fit on multi-modal
    # data becomes a no-op. So: reject the dead key outright.
    if "grasp_cond" in model_cfg:
        raise SystemExit(
            f"[cfg] {trn['train_cfg']} sets MODEL.grasp_cond, which Regrasp "
            f"removed — the conditioning is now per-point in the cloud, not an "
            f"MLP branch. Use MODEL.direction_cond and DATA.pc_channels: 7.")
    direction_cond = bool(model_cfg.get("direction_cond", True))
    if num_grasps > 1 and not direction_cond:
        raise SystemExit(
            f"[cfg] the pin table gives {num_grasps} directions per scene but "
            f"{trn['train_cfg']} has MODEL.direction_cond: false. The dataset "
            f"would hold {num_grasps} contradictory labels per state and the fit "
            f"could only regress their mean.")
    if int(train_cfg.get("DATA", {}).get("pc_channels", 0)) != 7:
        raise SystemExit(
            f"[cfg] direction conditioning needs DATA.pc_channels: 7 "
            f"(xyz|ycb|hand|d.n|d.r); {trn['train_cfg']} says "
            f"{train_cfg.get('DATA', {}).get('pc_channels')}.")

    collect_params = CollectParams(
        max_steps=int(dag.get("max_steps", 30)),
        close_pos_thresh=float(dag.get("close_pos_thresh", 0.02)),
        close_rot_thresh=float(dag.get("close_rot_thresh", 0.34)),
        stop_on_close_label=bool(dag.get("stop_on_close_label", True)),
        stop_on_policy_close=bool(dag.get("stop_on_policy_close", True)),
        ee_step=float(dag.get("ee_step", 0.04)),
        reach_tail=int(dag.get("reach_tail", 5)),
        min_free=int(dag.get("min_free", 3)),
        max_horizon=int(dag.get("max_horizon", 40)),
        # The step-0 plan has no standoff to size a distance-proportional horizon
        # from, so it falls back to the horizon the demonstrations were planned
        # with (collect_bc_dataset.py uses cfg.RL_MAX_STEP).
        first_horizon=int(dag.get("first_horizon") or sim_cfg.RL_MAX_STEP),
        # Planner-free pre-grasp (run 18). Default false, so a config that omits
        # the key collects exactly what runs 1-17 collected. standoff_dist is
        # mirrored from the SIM block — it is OMG's own constant, not a second
        # free parameter, and reading it from anywhere else would let the
        # derivation disagree with the waypoints the planner actually returns.
        derive_standoff=bool(dag.get("derive_standoff", False)),
        standoff_dist=float(sim_cfg_d.get("standoff_dist", 0.08)),
        # Per-episode outcome scoring (run 18). Default false => runs 1-17 spend
        # no extra sim steps and log zeros in the co_* columns. hold_steps is
        # taken from the EVAL block so the collection and eval criteria cannot
        # drift apart — they are the same `grasp_held_after_hold` call.
        outcome_check=bool(dag.get("outcome_check", False)),
        hold_steps=int(ev.get("hold_steps", 3)),
        # Where the episode ends (run 21). "grasp" is runs 1-20, so a config that
        # omits the key labels exactly what they labelled. forward_dist defaults
        # to the standoff offset itself rather than to a literal, so it tracks
        # standoff_dist / reach_tail instead of silently disagreeing with them.
        target=str(dag.get("target", "grasp")),
        forward_dist=float(dag.get("forward_dist") or forward_dist_default(
            float(sim_cfg_d.get("standoff_dist", 0.08)),
            int(dag.get("reach_tail", 5)))),
        forward_steps=int(dag.get("forward_steps", 4)),
        # 0 = commit on the step the tolerance is first met, i.e. runs 1-20.
        commit_settle_steps=int(dag.get("commit_settle_steps", 0)),
        # DART-paper noise (run 18). "jolt" is the runs 1-17 behaviour and the
        # default, so an unset key changes nothing. `dart_sigma` is seeded from
        # state.json on resume so a restarted run keeps the covariance it had
        # rather than dropping back to the bootstrap.
        dart_mode=str(dag.get("dart_mode", "jolt")),
        dart_alpha_scale=float(dag.get("dart_alpha_scale", 3.0)),
        dart_alpha_end=float(dag.get("dart_alpha_end",
                                     dag.get("dart_alpha_scale", 3.0))),
        dart_noise_ratio=float(dag.get("dart_noise_ratio", 1.0)),
        dart_reach_pos_scale=float(dag.get("dart_reach_pos_scale", 0.3005)),
        dart_reach_rot_scale=float(dag.get("dart_reach_rot_scale", 1.0)),
        dart_sigma=(np.asarray(state["dart_sigma"], dtype=np.float64)
                    if state.get("dart_sigma") is not None else None),
        reach_commit_dist=float(dag.get("reach_commit_dist", 0.05)),
        reach_skip_eps=float(dag.get("reach_skip_eps", 0.01)),
        expert_after_commit=bool(dag.get("expert_after_commit", False)),
        # DART. Off by default: `dart_ratio: 0` draws no random number, so a run
        # that omits these keys collects exactly what it collected before DART
        # existed.
        dart_ratio=float(dag.get("dart_ratio", 0.0)),
        dart_max_dist=float(dag.get("dart_max_dist", 0.20)),
        dart_pos_mag=float(dag.get("dart_pos_mag", 0.04)),
        dart_rot_mag=float(dag.get("dart_rot_mag", 0.2)),
        # DART inside the committed reach (run 12). Same "0 draws nothing"
        # contract, so leaving it unset keeps a config bit-identical to runs 4-11.
        dart_reach_ratio=float(dag.get("dart_reach_ratio", 0.0)),
        dart_reach_pos_mag=float(dag.get("dart_reach_pos_mag", 0.01202)),
        dart_reach_rot_mag=float(dag.get("dart_reach_rot_mag", 0.01549)),
        dart_reach_max_tries=int(dag.get("dart_reach_max_tries", 5)),
        dart_reach_clearance=float(dag.get("dart_reach_clearance", 0.01)),
        dart_reach_path_steps=int(dag.get("dart_reach_path_steps", 4)),
    )
    # (pin_table is loaded above, before the scene pools, because its key set
    # defines which scenes the expert can plan for.)
    # Verification, not enforcement: the pin can silently fail to apply (scene
    # missing from the table, stale pose, OMG failure at step 0), and a scene
    # revisited in a later iteration would then be labelled towards a DIFFERENT
    # grasp. `goal_switch` cannot see that — it is per-episode.
    registry = GraspRegistry(run_root / "grasp_registry.json",
                             tol=float(sim_cfg_d.get("grasp_pin_match_tol", 0.02)))

    # Parallel D_i collection. Built AFTER the main env so a startup failure in a
    # worker is reported before any iteration begins. The main env stays: the
    # evaluator still uses it, and it is what `sim.num_scenes` is read from.
    collector_pool = None
    if int(args.num_workers) > 1:
        from handover_sim2real.regrasp.parallel import ParallelDaggerCollector
        collector_pool = ParallelDaggerCollector(
            num_workers=int(args.num_workers), cfg4=cfg4, base_seed=seed,
            worker_device=str(args.worker_device))
        m_ep = int(dag["episodes_per_iter"])
        rounds = -(-m_ep // int(args.num_workers))
        print(f"[parallel] m={m_ep} over {args.num_workers} workers -> "
              f"{rounds} episode(s) each at the slowest worker "
              f"({m_ep / (rounds * int(args.num_workers)) * 100:.0f}% utilization)")

    # Success is the PHASE-3 criterion (hold the close, object secured), not the
    # benchmark's carry-to-GOAL_CENTER SUCCESS — Phase 4 has no retreat, so the
    # benchmark's flag could never fire. The proximity thresholds are taken from
    # the DAgger block so `proximity` mode scores exactly the predicate the
    # collector uses to emit its CLOSE label.
    # Geometric opportunity test. Defaults describe the Panda pads and are not
    # per-run knobs; min_frac/open_thresh DEFINE the metric, so changing either
    # makes box_* incomparable across runs — see dagger/grasp_box.py.
    box_params = build_box_params(ev)
    eval_params = EvalParams(
        max_steps=int(ev.get("max_steps", 30)),
        success_mode=str(ev.get("success_mode", "stable_grasp")),
        hold_steps=int(ev.get("hold_steps", 3)),
        close_pos_thresh=collect_params.close_pos_thresh,
        close_rot_thresh=collect_params.close_rot_thresh,
        box_check=bool(ev.get("box_check", True)),
        box=box_params,
        # Taken from the COLLECTION params, not re-read from the EVAL block: a
        # policy trained to stop at the pre-grasp and evaluated as if it stopped
        # at the grasp would score 0 for a reason that has nothing to do with the
        # policy, and there is no configuration in which the two should differ.
        target=collect_params.target,
        forward_dist=collect_params.forward_dist,
        forward_steps=collect_params.forward_steps,
        standoff_dist=collect_params.standoff_dist,
        reach_tail=collect_params.reach_tail,
        verbose=bool(ev.get("verbose", False)))
    eval_every = int(ev.get("every", 1))
    eval_ckpt = str(ev.get("ckpt", "best"))
    select_on = str(ev.get("select_on", "success_rate"))

    print("=" * 78)
    print(f"Phase-4 DAgger   run={run_name}")
    if usable is None:
        print(f"  scenes        : {sim.num_scenes} in split={sim_cfg.BENCHMARK.SPLIT}"
              f"  [NO pin table: pool unfiltered, expect ~13% of episodes to abort "
              f"with OMG_FAIL_STEP0]")
    else:
        no_plan = sim.num_scenes - len(usable) - n_excluded
        print(f"  scenes        : {sim.num_scenes} in split={sim_cfg.BENCHMARK.SPLIT}"
              f" -> {len(usable)} usable  ({no_plan} have no OMG goal set"
              + (f", {n_excluded} excluded as failed demos" if n_excluded else "") + ")")
    print(f"  pool / eval   : {len(pool)} / {len(eval_scenes)}"
          f"{'  [eval held out of the pool]' if ev.get('holdout', True) else '  [eval scenes also collected on]'}")
    print(f"  learner       : {learner}"
          f"{' (single frame)' if learner == 'bc' else ' (temporal + chunking)'}"
          f"   <- {trn['train_cfg']}")
    print(f"  iterations    : {num_iters}  x  m={dag['episodes_per_iter']} episodes")
    print(f"  beta schedule : {dag.get('beta_schedule')}  "
          f"(iter1={beta_at(1, dag, num_iters):.3f}, "
          f"iter{num_iters}={beta_at(num_iters, dag, num_iters):.3f})")
    print("                  "
          + " ".join(f"{beta_at(j, dag, num_iters):.2f}"
                     for j in range(1, num_iters + 1)))
    print(f"  grasp pinning : "
          f"{pin_table.describe() if pin_table else 'OFF (OMG re-selects every replan)'}")
    print(f"  close label   : pos<={collect_params.close_pos_thresh} m  "
          f"rot<={collect_params.close_rot_thresh} rad")
    print(f"  DART          : "
          + (f"p={collect_params.dart_ratio} per approach step within "
             f"({collect_params.reach_commit_dist}, {collect_params.dart_max_dist}] m "
             f"of the standoff; +-{collect_params.dart_pos_mag} m / "
             f"+-{collect_params.dart_rot_mag} rad"
             if collect_params.dart_ratio > 0.0 else "OFF"))
    print(f"  DART (reach)  : "
          + (f"p={collect_params.dart_reach_ratio} per committed-reach step; "
             f"+-{collect_params.dart_reach_pos_mag} m / "
             f"+-{collect_params.dart_reach_rot_mag} rad, rejection-sampled "
             f"(clearance {collect_params.dart_reach_clearance} m, "
             f"{collect_params.dart_reach_max_tries} tries)"
             if collect_params.dart_reach_ratio > 0.0 else "OFF"))
    print(f"  eval success  : {eval_params.success_mode}"
          + (f" (hold {eval_params.hold_steps} steps, then released & not dropped)"
             if eval_params.success_mode == "stable_grasp"
             else " (EE within the CLOSE tolerances of the grasp pose)")
          + "  [Phase-3 criterion; no carry-to-goal]")
    print(f"  refit         : {'from scratch (FTL)' if trn.get('train_from_scratch', True) else 'warm start'}")
    print(f"  run dir       : {run_root}")
    print("=" * 78)

    log_path = run_root / "dagger_log.csv"
    dagger_files: list[str] = []

    # ----- iteration 0: the base policy (the paper's beta_1 = 1 iteration) -----
    if trn.get("base_run"):
        base_dir = Path(trn["base_run"])
        print(f"[iter 00] using existing base run {base_dir}")
    elif 0 in done_iters:
        base_dir = Path(done_iters[0]["run_dir"])
        print(f"[iter 00] already trained: {base_dir}")
    else:
        print(f"[iter 00] training the base policy on {trn['base_train_h5']} "
              f"({trn['base_epochs']} epochs)")
        base_dir = train_on_aggregate(
            train_cfg, base_files(trn), trn.get("val_h5"),
            run_root / "iters" / "iter_00", epochs=int(trn["base_epochs"]),
            device=device, num_workers=int(trn.get("num_workers", 2)), seed=seed)

    if 0 not in done_iters:
        rec = {"iter": 0, "run_dir": str(base_dir), "dagger_h5": None,
               "beta": 1.0, "eval": None}
        state["iterations"].append(rec)
        done_iters[0] = rec
        save_state(run_root / "state.json", state)

    # The paper returns the best policy over the WHOLE sequence pi_1..pi_N, and
    # pi_1 (the base run on expert data alone) is part of that sequence — so it
    # has to be scored too, or a DAgger iteration wins the comparison by default.
    if eval_every > 0 and eval_scenes and done_iters[0].get("eval") is None:
        runner, _ = load_policy_runner(base_dir, device, ckpt=eval_ckpt)
        # The real episode count, not scenes x max_grasps: a per-bin table gives
        # each scene only the directions it can reach (mean 2.59, max 4), so the
        # product OVERSTATES the sweep by ~55% and makes the first timing estimate
        # of the run wrong in the direction that matters.
        _n_eval_ep = (len(pin_table.pairs(eval_scenes)) if pin_table is not None
                      else len(eval_scenes))
        print(f"[iter 00] evaluating the base policy on {len(eval_scenes)} scenes "
              f"= {_n_eval_ep} episodes (per-scene direction counts) ...")
        base_metrics = run_eval(collector_pool, sim, base_dir, eval_ckpt, device,
                                eval_scenes, eval_params, pin_table, iteration=0)
        base_metrics.pop("rows")
        print(f"[iter 00] success={base_metrics['success_rate']:.3f} "
              f"grasp={base_metrics['grasp_rate']:.3f} "
              f"near={base_metrics['near_rate']:.3f} "
              f"close={base_metrics['close_rate']:.3f} "
              f"chance={base_metrics['chance_rate']:.3f} "
              f"min_pos={base_metrics['eval_min_pos']:.3f} m "
              f"min_rot={base_metrics['eval_min_rot']:.3f} rad")
        done_iters[0]["eval"] = base_metrics
        best0 = maybe_update_best(state, run_root, base_dir, 0, base_metrics,
                                  select_on, eval_ckpt)
        done_iters[0]["is_best"] = best0
        export_run_dir(base_dir, run_root / "last", ckpt=eval_ckpt,
                       note="DAgger iteration 0 (base policy)")
        save_state(run_root / "state.json", state)
        d_ep, d_steps = dataset_size(base_files(trn))
        row0 = {
            "iter": 0, "beta": 1.0, "m": 0, "episodes": 0, "steps": 0,
            "skipped": 0, "aggregate_files": 1,
            "D_episodes": d_ep, "D_steps": d_steps, "D_dagger_frac": 0.0,
            "is_best": int(best0), "wall_s": 0.0,
        }
        row0.update(read_train_log(base_dir))
        row0.update(eval_columns(base_metrics))
        log_row(log_path, row0)

    cur_run_dir = Path(done_iters[max(done_iters)]["run_dir"])
    dagger_files = [r["dagger_h5"] for r in state["iterations"] if r.get("dagger_h5")]

    # ----- DAgger iterations -----
    for i in range(1, num_iters + 1):
        if i in done_iters:
            cur_run_dir = Path(done_iters[i]["run_dir"])
            print(f"[iter {i:02d}] already complete -> {cur_run_dir}")
            continue

        t0 = time.time()
        beta = beta_at(i, dag, num_iters)
        rng = np.random.RandomState(seed * 10_000 + i)
        pairs, state["cursor"] = sample_pairs(
            pool, int(dag["episodes_per_iter"]), num_grasps,
            str(dag.get("scene_sampling", "random")), rng,
            int(state.get("cursor", 0)), pin_table=pin_table)

        n_scenes_drawn = len({s for s, _ in pairs})
        n_paired = sum(1 for s in {s for s, _ in pairs}
                       if sum(1 for ss, _ in pairs if ss == s) > 1)
        print(f"\n{'='*78}\n[iter {i:02d}/{num_iters}]  beta={beta:.3f}  "
              f"m={len(pairs)} episodes over {n_scenes_drawn} scenes "
              f"({n_paired} paired)  policy={cur_run_dir}\n{'='*78}")

        # --- D_i: roll out the CURRENT policy, label every visited state ---
        t_collect = time.time()
        h5_path = run_root / "data" / f"dagger_iter_{i:02d}.h5"
        # REUSE ONLY A SHARD THAT FINISHED. This path exists so a run that died
        # between collection and the refit does not re-collect an hour of work.
        # But `exists()` alone cannot tell a finished shard from an interrupted
        # one, and it silently reused an 82-of-354-episode iteration and would
        # have recorded it as complete — the aggregate would then be permanently
        # short by most of an iteration, with nothing in any log saying so.
        #
        # `complete` is written only by DaggerHDF5Writer.close(). Shards from
        # before the flag existed have no attr; those are treated as complete,
        # because they were produced by a loop that could not record an unfinished
        # one, and re-collecting them would discard good data.
        reuse = False
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                n_ep = int(f.attrs.get("num_episodes", 0))
                done = f.attrs.get("complete", None)
            legacy = done is None
            reuse = bool(done) or legacy
            if not reuse:
                print(f"  [collect] {h5_path.name} is INCOMPLETE ({n_ep} episodes, "
                      f"complete=False) — the run was interrupted mid-collection. "
                      f"Discarding and re-collecting.")
                h5_path.unlink()
        if reuse:
            print(f"  [collect] reusing existing {h5_path.name} ({n_ep} episodes"
                  + ("; legacy shard with no `complete` flag)" if legacy else ")"))
            cstats = {"episodes": n_ep, "steps": -1, "skipped": -1,
                      "reached_standoff": -1, "n_reach_steps": -1,
                      "reached_grasp": -1, "n_omg_fail": -1, "n_goal_switch": -1,
                      "n_expert_steps": -1, "policy_closed": -1,
                      "n_policy_close_cmds": -1, "n_dropped_tail": -1,
                      "n_dart": -1, "n_dart_env_done": -1,
                      "n_dart_reach": -1, "n_dart_reject": -1,
                      "mean_min_pos": float("nan"), "mean_min_rot": float("nan"),
                      "mean_policy_close_step": float("nan"),
                      "n_close_labels": -1, "n_approach_labels": -1,
                      "mean_label_pos": float("nan"), "n_tiny_labels": -1,
                      "n_revisits": -1, "n_grasp_mismatch": -1,
                      "max_grasp_drift": float("nan"),
                      "reasons": {}, "path": str(h5_path)}
        else:
            print(f"  [collect] {len(pairs)} episodes -> {h5_path.name}")
            if collector_pool is not None:
                # Per-episode seeds drawn from the iteration rng, so the scene
                # sequence AND the seed sequence are what the serial path would
                # have produced; only which RandomState each episode reads from
                # differs (bitwise identity is not meaningful once episodes run
                # concurrently). Aggregation still happens in collect_iteration.
                ep_seeds = [int(rng.randint(0, 2 ** 31 - 1)) for _ in pairs]
                results = collector_pool.collect(
                    cur_run_dir, eval_ckpt, pairs, ep_seeds, beta,
                    collect_params, i)
                cstats = collect_iteration(
                    None, None, pairs, h5_path, rng=rng,
                    beta=beta, params=collect_params, pin_table=pin_table,
                    registry=registry, iteration=i, results=results)
            else:
                runner, _ = load_policy_runner(cur_run_dir, device, ckpt=eval_ckpt)
                cstats = collect_iteration(
                    sim, runner, pairs, h5_path, rng=rng,
                    beta=beta, params=collect_params, pin_table=pin_table,
                    registry=registry, iteration=i)
                del runner
                if device != "cpu":
                    torch.cuda.empty_cache()
            print(f"  [collect] episodes={cstats['episodes']} steps={cstats['steps']} "
                  f"standoff={cstats.get('reached_standoff', -1)} "
                  f"grasp={cstats['reached_grasp']} pinned={cstats.get('pinned', -1)} "
                  f"skipped={cstats['skipped']} "
                  + (f"dart={cstats.get('n_dart', -1)}"
                     f"(ended={cstats.get('n_dart_env_done', -1)}) "
                     if collect_params.dart_ratio > 0.0 else "")
                  + (f"dart_reach={cstats.get('n_dart_reach', -1)}"
                     f"(rejected={cstats.get('n_dart_reject', -1)}) "
                     if collect_params.dart_reach_ratio > 0.0 else "")
                  + f"omg_fail={cstats['n_omg_fail']} reasons={cstats['reasons']}")

        # alpha governing THIS iteration's noise (iteration 1 uses the psi_0
        # bootstrap instead, since no error has been measured yet).
        cstats["dart_alpha"] = dart_alpha_at(
            i, num_iters, collect_params.dart_alpha_scale,
            collect_params.dart_alpha_end)

        # --- DART: re-estimate Sigma from THIS iteration's learner-supervisor
        # discrepancy and hand it to the next one (paper Alg. 1 steps 3-5).
        # Runs whatever the mode, so a jolt run still records what the noise would
        # have been; only dart_noise consumes it.
        sig_hat = cstats.get("dart_sigma_hat")
        if sig_hat is not None:
            tr_hat = float(np.trace(sig_hat))
            # The ANCHOR is the first iteration's measured error, tr(Sigma_hat_1),
            # and never moves — so a learner that gets worse is not answered with
            # more noise (the paper's Eq. 4 safeguard). Persisted so a resumed run
            # re-anchors to the same value rather than to whichever iteration it
            # happened to restart on.
            if state.get("dart_trace_anchor") is None:
                state["dart_trace_anchor"] = tr_hat
                print(f"  [dart] anchoring to iteration {i:02d}: "
                      f"tr(Sigma_hat_1)={tr_hat:.3e}")
            # alpha is annealed across iterations (dart_alpha_scale ->
            # dart_alpha_end), so the noise LEVEL decays as the learner improves
            # while its SHAPE keeps tracking the current error directions. i+1
            # because the Sigma estimated now is the one the NEXT iteration uses.
            alpha_next = dart_alpha_at(i + 1, num_iters,
                                       collect_params.dart_alpha_scale,
                                       collect_params.dart_alpha_end)
            sigma_next = dart_scaled_sigma(
                sig_hat, alpha_next * state["dart_trace_anchor"])
            collect_params = replace(collect_params, dart_sigma=sigma_next)
            state["dart_sigma"] = sigma_next.tolist()
            state["dart_alpha"] = float(alpha_next)
            if collect_params.dart_mode == "dart_noise":
                print(f"  [dart] tr(Sigma_hat)={tr_hat:.3e}  alpha[{i+1:02d}]="
                      f"{alpha_next:.3f} -> Sigma^alpha diag sd "
                      f"pos={np.sqrt(np.diag(sigma_next)[:3]).round(4).tolist()} m "
                      f"rot={np.sqrt(np.diag(sigma_next)[3:]).round(4).tolist()} rad "
                      f"(noised {cstats.get('n_dart_noise', 0)} expert steps)")

        collect_s = time.time() - t_collect

        if cstats["episodes"] == 0:
            print(f"  [iter {i:02d}] no episodes collected — stopping.")
            break
        dagger_files.append(str(h5_path))

        # --- refit on the aggregate D = base u D_1 u ... u D_i ---
        t_train = time.time()
        iter_dir = run_root / "iters" / f"iter_{i:02d}"
        init_from = None if bool(trn.get("train_from_scratch", True)) else cur_run_dir
        # Which checkpoint seeds the warm start. "last" is run 20's behaviour and
        # the default, so a config that omits the key is unchanged; "best" makes
        # the chain carry forward the same weights EVAL.ckpt and the collector
        # already use. Inert when train_from_scratch is true.
        init_ckpt = str(trn.get("init_ckpt", "last"))
        train_on_aggregate(
            train_cfg, base_files(trn) + dagger_files, trn.get("val_h5"),
            iter_dir, epochs=int(trn["iter_epochs"]), device=device,
            num_workers=int(trn.get("num_workers", 2)), init_from=init_from,
            init_ckpt=init_ckpt, seed=seed + i)
        cur_run_dir = iter_dir
        train_s = time.time() - t_train

        # --- evaluate: DAgger returns the BEST policy over the sequence ---
        t_eval = time.time()
        eval_s = 0.0          # stays 0 on the iterations EVAL.every skips
        emetrics = None
        if eval_every > 0 and eval_scenes and (i % eval_every == 0 or i == num_iters):
            _njobs = len(eval_jobs(eval_scenes, pin_table))
            print(f"  [eval] {len(eval_scenes)} scenes = {_njobs} episodes"
                  + (f" on {collector_pool.num_workers} workers ..."
                     if collector_pool is not None else " (serial) ..."))
            emetrics = run_eval(collector_pool, sim, iter_dir, eval_ckpt, device,
                                eval_scenes, eval_params, pin_table, iteration=i)
            emetrics.pop("rows")
            print(f"  [eval] success={emetrics['success_rate']:.3f} "
                  f"grasp={emetrics['grasp_rate']:.3f} "
                  f"near={emetrics['near_rate']:.3f} "
                  f"close={emetrics['close_rate']:.3f} "
                  f"(success|close={emetrics['close_success_rate']:.3f})")
            print(f"  [eval] chance={emetrics['chance_rate']:.3f} "
                  f"missed={emetrics['missed_rate']:.3f} "
                  f"min_pos={emetrics['eval_min_pos']:.3f} m "
                  f"min_rot={emetrics['eval_min_rot']:.3f} rad "
                  f"ee->ycb={emetrics['mean_dist']:.3f} m")
            # Geometric opportunity — the pin-free reading. box_chance is the
            # denominator, taken|box the conversion.
            print(f"  [eval] box_chance={emetrics['box_chance_rate']:.3f} "
                  f"(taken|box={emetrics['box_taken_rate']:.3f} "
                  f"missed|box={emetrics['miss_given_box']:.3f}) "
                  f"window={emetrics['mean_box_steps']:.1f} steps "
                  f"occ={emetrics['mean_box_frac']:.3f}")
        eval_s = time.time() - t_eval

        # --- best / last snapshots ---
        export_run_dir(iter_dir, run_root / "last", ckpt=eval_ckpt,
                       note=f"DAgger iteration {i} (last)")
        is_best = maybe_update_best(state, run_root, iter_dir, i, emetrics,
                                    select_on, eval_ckpt)

        wall = time.time() - t0
        rec = {"iter": i, "run_dir": str(iter_dir), "dagger_h5": str(h5_path),
               "beta": beta, "pairs": pairs, "collect": cstats,
               "eval": emetrics, "is_best": is_best, "wall_s": round(wall, 1)}
        state["iterations"].append(rec)
        done_iters[i] = rec
        save_state(run_root / "state.json", state)

        # D as the refit actually saw it: demonstrations + every D_j so far.
        # `D_dagger_frac` is the axis that separates "DAgger helps" from "more
        # data helps" — success against |D| with the on-policy share alongside.
        d_ep, d_steps = dataset_size(base_files(trn) + dagger_files)
        _, dag_steps = dataset_size(dagger_files)

        row = {
            "iter": i, "beta": round(beta, 4), "m": len(pairs),
            "aggregate_files": len(dagger_files) + 1,
            "D_episodes": d_ep, "D_steps": d_steps,
            "D_dagger_frac": _r(dag_steps / d_steps if d_steps else float("nan")),
            "is_best": int(is_best),
            "collect_s": round(collect_s, 1), "train_s": round(train_s, 1),
            "eval_s": round(eval_s, 1), "wall_s": round(wall, 1),
        }
        row.update(collect_columns(cstats))
        row.update(read_train_log(iter_dir))
        row.update(eval_columns(emetrics))
        log_row(log_path, row)

    if collector_pool is not None:
        collector_pool.close()

    print("\n" + "=" * 78)
    print(f"Done. {len(dagger_files)} DAgger iterations aggregated.")
    if (run_root / "last").exists():
        print(f"  last : {run_root / 'last'}")
    if state["best"]:
        b = state["best"]
        print(f"  best : {run_root / 'best'}  "
              f"(iteration {b['iter']}, {b['metric']}={b['score']:.4f})")
    else:
        print("  best : not selected — score the iterations separately:\n"
              f"         python examples/eval_dagger_run.py --run-dir {run_root} "
              f"--publish-best")
    print(f"  log  : {log_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
