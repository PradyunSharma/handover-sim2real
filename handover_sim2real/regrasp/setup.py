"""One place that turns a Phase-5 config into everything a run needs.

`examples/train_regrasp.py` and `examples/eval_regrasp_run.py` both have to
build the SAME simulator, the SAME pin table, the SAME scene pools and the SAME
EvalParams — otherwise a success rate produced by the standalone evaluator is not
comparable to one produced inside the loop, which is the entire point of moving
evaluation out of it.

Rather than duplicate ~60 lines in two scripts and hope they stay in sync, both
call `build_regrasp_context(cfg)` here. The eval scene set in particular is
`np.linspace` over the *usable* scenes, so it depends on the pin table's key set
and the exclusion list; get either wrong and you silently score a different pool.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from handover_sim2real.regrasp.env_setup import build_sim_cfg, build_sim_context
from handover_sim2real.regrasp.evaluator import EvalParams
from handover_sim2real.regrasp.grasp_box import build_box_params
from handover_sim2real.regrasp.grasp_pin import load_grasp_pin_table
from handover_sim2real.regrasp.pregrasp import forward_dist_default


# ── WHERE THE BIG FILES LIVE ────────────────────────────────────────────────
#
# `${REGRASP_DATA}` in any config path expands from the environment, defaulting
# to the in-repo `output/`. On DelftBlue the sbatch scripts set it to
# `$SCRATCH_ROOT/output`, which moves the ~1.1 GB of HDF5 shards off /home — a
# HARD 30 GB quota that fills SILENTLY, killing the job with exit code 6 and no
# traceback, because Python cannot write one to a full disk.
#
# THE SPLIT IS BY SIZE AND REGENERABILITY, NOT BY KIND. The shards and the run
# directories are large and reproducible from a table plus a seed, so they go to
# scratch. The JSON tables are a few megabytes, are INPUTS rather than outputs,
# and are worth keeping in git — so configs name them plainly and they stay in
# the repo. /scratch is purged by age; the consequence of getting this backwards
# is either a full quota or a four-hour re-collection.
#
# Expansion happens at LOAD, so the config saved into a run directory records the
# resolved path. That is deliberate: `<run>/config.yaml` is a record of what the
# run actually read, and a run whose data has since been purged should say where
# the data was, not where it might be re-created.
DATA_ROOT_VAR = "REGRASP_DATA"
DEFAULT_DATA_ROOT = "output"


def data_root() -> str:
    return os.environ.get(DATA_ROOT_VAR) or DEFAULT_DATA_ROOT


def expand_config_paths(cfg):
    """Expand `${REGRASP_DATA}` and any other env var in every string in `cfg`.

    Recursive and in-place-safe (returns a new structure). A config with no `$`
    anywhere — every Regrasp config before run 2 — is returned unchanged, so this
    is free to apply everywhere and cannot alter an older run's meaning.
    """
    os.environ.setdefault(DATA_ROOT_VAR, DEFAULT_DATA_ROOT)
    if isinstance(cfg, dict):
        return {k: expand_config_paths(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [expand_config_paths(v) for v in cfg]
    if isinstance(cfg, str) and "$" in cfg:
        return os.path.expandvars(cfg)
    return cfg


def scene_pools(num_scenes: int, ev: dict,
                usable: set | None = None) -> tuple[list[int], list[int]]:
    """(collection pool, eval scenes).

    `usable` restricts BOTH to scenes the expert can actually plan for — pass the
    grasp pin table's keys. Roughly 13% of the s0 train split (97 of 720) has no
    reachable goal set, and OMG fails at step 0 there: the episode is abandoned
    with reason OMG_FAIL_STEP0 having produced no labels. Without the filter
    DAgger rediscovers those scenes on every iteration that samples one, so
    `episodes_per_iter: 100` yields ~86 episodes rather than 100, and eval draws
    ~13 scenes for which no grasp pose exists (their pose metrics come out NaN
    while success/grasp/close still count, so the denominators silently differ
    between columns).

    The pin table is the right filter by construction, not by assumption: it was
    built by calling `run_omg_planner(..., reset_scene=True)` on each scene —
    exactly the call the collector makes at step 0 — and recording the ones that
    returned a plan. If planning is deterministic, a scene absent from the table
    would fail here too. `c_omg_fail0` in dagger_log.csv stays as the check: with
    the filter on it should be 0, and anything else is evidence that OMG's
    outcome varies run to run.

    Eval scenes are spread evenly over the usable set (np.linspace) rather than
    taken as a prefix, so the same config always yields the same held-out scenes
    — which is what lets examples/eval_regrasp_run.py reproduce the in-loop pool.
    """
    ids = sorted(usable) if usable is not None else list(range(num_scenes))
    n_eval = min(int(ev.get("num_scenes", 100)), len(ids))
    eval_scenes = sorted({ids[i] for i in
                          np.linspace(0, len(ids) - 1, n_eval).astype(int).tolist()})
    if bool(ev.get("holdout", True)):
        pool = [s for s in ids if s not in set(eval_scenes)]
    else:
        pool = list(ids)
    return pool, eval_scenes


@dataclass
class RegraspContext:
    """Everything both entry points need, built once from the resolved config."""

    sim: Any                    # SimContext (env + point listener + transforms)
    sim_cfg: Any                # the yacs sim config
    pin_table: Any              # GraspPinTable | None
    pool: list                  # scenes DAgger collects on
    eval_scenes: list           # scenes DAgger is scored on
    num_grasps: int             # pinned grasps per scene (Phase 5; 1 = Phase 4)
    eval_params: EvalParams
    eval_ckpt: str
    select_on: str
    n_excluded: int
    usable: set | None
    # None when SIM.demo_ok_table is unset; otherwise the prune report from
    # GraspPinTable.keep_only — how many (scene, bin) pairs base collection
    # actually demonstrated, and which scenes it emptied.
    demo_ok: dict | None = None


def build_regrasp_context(cfg4: dict, *, seed: int = 0,
                         verbose: bool = True) -> RegraspContext:
    """Build the simulator + pools + eval settings a Phase-5 config describes.

    `cfg4` is the resolved Phase-5 config (the dict saved as <run>/config.yaml),
    so passing that file back in reproduces a run's setup exactly.
    """
    sim_cfg_d = cfg4["SIM"]
    dag = cfg4["DAGGER"]
    ev = cfg4.get("EVAL", {})

    # build_sim_cfg and load_grasp_pin_table print banners unconditionally. One
    # copy is informative; twenty (one per collection worker) is noise that hides
    # the run's own output. Only stdout is captured — exceptions still propagate.
    sink = contextlib.redirect_stdout(io.StringIO()) if not verbose \
        else contextlib.nullcontext()
    with sink:
        sim_cfg = build_sim_cfg(sim_cfg_d)
        sim = build_sim_context(sim_cfg, sim_cfg_d, seed=seed)

        pin_table = load_grasp_pin_table(
            sim_cfg_d.get("grasp_pin_table"),
            match_tol=float(sim_cfg_d.get("grasp_pin_match_tol", 0.02)),
            sim_cfg_block=sim_cfg_d)

    # ---- (scene, BIN) pairs base collection actually demonstrated -----------
    # `exclude_scenes` below is scene-granular and cannot express "this scene is
    # fine for +x and +z but OMG could not plan the -y grasp". Run 2 needs that:
    # it asks for one demonstration per reachable bin, and a per-bin planner
    # failure would otherwise leave the policy trained on three directions and
    # EVALUATED on four, scoring an extrapolation as if it were a regression.
    #
    # Applied to the pin table rather than checked at each call site, so the
    # collection pool, the eval loop, `bin_of` and `max_grasps` cannot disagree
    # about which pairs exist. Written by audit_regrasp_demos.py --write-ok from
    # the base shard, so it reflects what was collected, not what was hoped for.
    demo_ok_path = sim_cfg_d.get("demo_ok_table")
    demo_ok_report = None
    if demo_ok_path and pin_table is not None:
        with open(demo_ok_path) as f:
            raw_ok = json.load(f)
        demo_ok_report = pin_table.keep_only(
            raw_ok.get("ok", raw_ok), verbose=verbose)

    usable = set(pin_table.entries) if pin_table is not None else None

    n_excluded = 0
    excl_path = sim_cfg_d.get("exclude_scenes")
    if excl_path:
        with open(excl_path) as f:
            excluded = {int(s) for s in json.load(f)}
        before = len(usable) if usable is not None else sim.num_scenes
        usable = (usable or set(range(sim.num_scenes))) - excluded
        n_excluded = before - len(usable)
        if verbose:
            print(f"[exclude_scenes] {excl_path}: dropping {n_excluded} of "
                  f"{before} scenes whose expert demonstration failed")

    pool, eval_scenes = scene_pools(sim.num_scenes, ev, usable=usable)

    # Thresholds come from the DAGGER block on purpose: `proximity` mode must
    # score exactly the predicate the collector uses to emit its CLOSE label.
    eval_params = EvalParams(
        max_steps=int(ev.get("max_steps", 30)),
        success_mode=str(ev.get("success_mode", "stable_grasp")),
        hold_steps=int(ev.get("hold_steps", 3)),
        close_pos_thresh=float(dag.get("close_pos_thresh", 0.02)),
        close_rot_thresh=float(dag.get("close_rot_thresh", 0.34)),
        box_check=bool(ev.get("box_check", True)),
        box=build_box_params(ev),
        # Also from the DAGGER block, and for a stronger version of the same
        # reason: a policy trained to stop at the pre-grasp and evaluated as if
        # it stopped at the grasp scores 0 for a reason that has nothing to do
        # with the policy. There is no configuration in which these should differ
        # between collection and eval, so there is only one place to set them.
        target=str(dag.get("target", "grasp")),
        forward_dist=float(dag.get("forward_dist") or forward_dist_default(
            float(sim_cfg_d.get("standoff_dist", 0.08)),
            int(dag.get("reach_tail", 5)))),
        forward_steps=int(dag.get("forward_steps", 4)),
        standoff_dist=float(sim_cfg_d.get("standoff_dist", 0.08)),
        reach_tail=int(dag.get("reach_tail", 5)),
        verbose=bool(ev.get("verbose", False)))

    # Phase 5. `pool` and `eval_scenes` stay SCENE lists, not (scene, grasp)
    # pairs: the sampler expands each drawn scene into all of its slots, and the
    # eval set stays the same np.linspace over scenes it has always been — only
    # each of those scenes is now scored under every grasp. Keeping the pools
    # scene-shaped means the resume logic, the exclusion list and
    # eval_regrasp_run.py's reproduction of the in-loop pool are all unchanged.
    # MAX, not min: a Regrasp table mixes 1- and 2-grasp scenes, and
    # `num_grasps` (a min) would read 1 and make every `range(num_grasps)`
    # loop drop the paired second demonstration. This value sizes metric
    # arrays; ITERATION must use pin_table.num_grasps_for(scene).
    num_grasps = pin_table.max_grasps if pin_table is not None else 1
    if verbose:
        n_pool_pairs = (len(pin_table.pairs(pool)) if pin_table is not None
                        else len(pool))
        n_eval_pairs = (len(pin_table.pairs(eval_scenes)) if pin_table is not None
                        else len(eval_scenes))
        print(f"[regrasp] up to {num_grasps} direction(s) per scene -> "
              f"{n_pool_pairs} collectable (scene, direction) pairs, "
              f"{n_eval_pairs} eval episodes per scored iteration")

    return RegraspContext(
        sim=sim, sim_cfg=sim_cfg, pin_table=pin_table,
        pool=pool, eval_scenes=eval_scenes, num_grasps=num_grasps,
        eval_params=eval_params,
        eval_ckpt=str(ev.get("ckpt", "best")),
        select_on=str(ev.get("select_on", "success_rate")),
        n_excluded=n_excluded, usable=usable,
        demo_ok=demo_ok_report)
