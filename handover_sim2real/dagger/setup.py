"""One place that turns a Phase-4 config into everything a run needs.

`examples/train_dagger_phase4.py` and `examples/eval_dagger_run.py` both have to
build the SAME simulator, the SAME pin table, the SAME scene pools and the SAME
EvalParams — otherwise a success rate produced by the standalone evaluator is not
comparable to one produced inside the loop, which is the entire point of moving
evaluation out of it.

Rather than duplicate ~60 lines in two scripts and hope they stay in sync, both
call `build_phase4_context(cfg)` here. The eval scene set in particular is
`np.linspace` over the *usable* scenes, so it depends on the pin table's key set
and the exclusion list; get either wrong and you silently score a different pool.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from handover_sim2real.dagger.env_setup import build_sim_cfg, build_sim_context
from handover_sim2real.dagger.evaluator import EvalParams
from handover_sim2real.dagger.grasp_pin import load_grasp_pin_table


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
    — which is what lets examples/eval_dagger_run.py reproduce the in-loop pool.
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
class Phase4Context:
    """Everything both entry points need, built once from the resolved config."""

    sim: Any                    # SimContext (env + point listener + transforms)
    sim_cfg: Any                # the yacs sim config
    pin_table: Any              # GraspPinTable | None
    pool: list                  # scenes DAgger collects on
    eval_scenes: list           # scenes DAgger is scored on
    eval_params: EvalParams
    eval_ckpt: str
    select_on: str
    n_excluded: int
    usable: set | None


def build_phase4_context(cfg4: dict, *, seed: int = 0,
                         verbose: bool = True) -> Phase4Context:
    """Build the simulator + pools + eval settings a Phase-4 config describes.

    `cfg4` is the resolved Phase-4 config (the dict saved as <run>/config.yaml),
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
        verbose=bool(ev.get("verbose", False)))

    return Phase4Context(
        sim=sim, sim_cfg=sim_cfg, pin_table=pin_table,
        pool=pool, eval_scenes=eval_scenes, eval_params=eval_params,
        eval_ckpt=str(ev.get("ckpt", "best")),
        select_on=str(ev.get("select_on", "success_rate")),
        n_excluded=n_excluded, usable=usable)
