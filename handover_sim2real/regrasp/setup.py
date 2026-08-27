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

from handover_sim2real.regrasp import directions as _rg_directions
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


# ── WHAT THE POLICY IS TOLD, AND WHERE THAT IS DECIDED ──────────────────────
#
# `SIM.command_deploy` names the rule that turns a bin into the unit vector the
# policy is CONDITIONED ON whenever it acts — DAgger rollouts, in-loop eval, the
# retry ladder, the test set, the rollout viewer. It is one key because those
# five must agree; a run in which any two of them disagree reports nothing
# unusual, it simply tells the policy one thing and scores it on another.
#
#   bin_axis      `BINS[b]`, the geometric axis. Runs 2-8, and the default, so a
#                 config that omits the key behaves exactly as run 2 did.
#   bin_centroid  the empirical mean of that bin's assigned `d_anchor` over the
#                 PRUNED pin table. Same six-vector, deployment-computable
#                 object; 8-17 deg closer to what the demonstrations show.
#   grasp_axis    `-R_grasp[:,2]`. Run 1. NOT DEPLOYABLE — there is no grasp at
#                 test time — so it is here for reproducing run 1 and for
#                 diagnostics, never for a number that claims to be a deployment
#                 rate.
#
# The TRAINING label is a separate key, `DATA.d_source` in the learner config,
# because the two are genuinely independent: an episode is a rollout under one
# rule and a demonstration captioned under another. Run 9 is the first run in
# which they differ. `train_regrasp.py` refuses the one combination that is
# always wrong — see its `d_source` guard.
COMMAND_MODES = ("bin_axis", "bin_centroid", "grasp_axis")


def resolve_command_axes(pin_table, mode: str = "bin_axis", *,
                         verbose: bool = True):
    """`SIM.command_deploy` -> the [k, 3] axis set, or None for `grasp_axis`.

    None is the grasp-axis rule rather than an error: `directions.command_
    direction` treats a missing axis set as "derive it from the pose", which is
    the same fallback an old Phase-5-shaped pin table already takes.

    MUST be called after `GraspPinTable.keep_only` — the centroid summarises the
    training assignment, and the demo filter is what decides that.
    """
    mode = str(mode or "bin_axis")
    if mode not in COMMAND_MODES:
        raise SystemExit(f"[cfg] SIM.command_deploy must be one of "
                         f"{list(COMMAND_MODES)}, got {mode!r}")
    if mode == "grasp_axis":
        return None
    if mode == "bin_axis" or pin_table is None:
        return _rg_directions.BINS.copy()
    axes = pin_table.bin_centroids()
    if verbose:
        off = _rg_directions.angle_between(axes, _rg_directions.BINS)
        print("[command] deploy on the BIN CENTROID, not the bin axis; "
              "offset from each axis (deg): "
              + "  ".join(f"{_rg_directions.BIN_SHORT[b]} {off[b]:.1f}"
                          for b in _rg_directions.LIVE_BINS))
    return axes


def resolve_d_rule(pin_table, sim_cfg_d: dict, *, verbose: bool = True):
    """`SIM.d_rule` -> a `DirectionRule`, CROSS-CHECKED against the pin table.

    THE TABLE IS AUTHORITATIVE, and this function exists because of that. The
    table's bins were populated by one rule: `assign_direction_demos.py` filed
    each grasp under the bin nearest its `d_anchor`, and `d_anchor` came out of
    `build_direction_table.py` under whatever `--d-rule` it was given. A run that
    derives `bin_realized` under the OTHER rule disagrees with `bin_assigned` on
    most episodes, the dataset's miscaption filter then drops most of the
    aggregate, and the symptom is "the collection produced almost nothing" —
    which names neither the config key nor the table.

    So a config that states `SIM.d_rule` must agree with the table, and one that
    omits it inherits the table's. A table with no recorded rule predates this
    key and is `approach_axis` by construction.
    """
    meta = (getattr(pin_table, "meta", {}) or {}) if pin_table is not None else {}
    from_table = _rg_directions.DirectionRule(
        rule=str(meta.get("d_rule", "approach_axis")),
        depth=float(meta.get("d_point_depth", _rg_directions.FINGERTIP_DEPTH)),
        min_offset=float(meta.get("d_min_offset", 0.0)))
    asked = sim_cfg_d.get("d_rule")
    if asked is None:
        if verbose and from_table.needs_centroid():
            print(f"[d_rule] {from_table.describe()}  (from the pin table)")
        return from_table
    want = _rg_directions.DirectionRule.from_cfg(sim_cfg_d)
    if pin_table is not None and want.rule != from_table.rule:
        raise SystemExit(
            f"[cfg] SIM.d_rule: {want.rule} but "
            f"{getattr(pin_table, 'path', 'the pin table')} was built under "
            f"{from_table.rule!r}. The table's bins were populated by that rule, "
            f"so `bin_realized` computed under yours would disagree with "
            f"`bin_assigned` on most episodes and the miscaption filter would "
            f"discard the aggregate. Rebuild the table with "
            f"`build_direction_table.py --d-rule {want.rule}` (and re-assign, "
            f"re-collect, re-audit), or set SIM.d_rule: {from_table.rule}.")
    if verbose:
        print(f"[d_rule] {want.describe()}")
    return want


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
    # [k, 3] the axis set every command is built from, or None for the grasp-axis
    # rule. Mirrored onto `eval_params` so a worker that only receives params
    # still commands the same thing; kept here too because the collector's params
    # are built by train_regrasp.py, not by this function.
    command_axes: Any = None
    command_mode: str = "bin_axis"
    # `directions.DirectionRule` — what `d` is derived FROM. Distinct from
    # `command_axes`, which is what a BIN turns into: the rule decides what a bin
    # means, the axes decide which vector names it.
    d_rule: Any = None


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

    # AFTER the demo filter, before anything that commands: the centroid rule
    # summarises the assignment that survived `keep_only`.
    command_mode = str(sim_cfg_d.get("command_deploy", "bin_axis"))
    command_axes = resolve_command_axes(pin_table, command_mode, verbose=verbose)
    d_rule = resolve_d_rule(pin_table, sim_cfg_d, verbose=verbose)

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
        # What every commanded direction is built from. On the params rather than
        # looked up per call site because the parallel evaluator SHIPS these to
        # its workers — a worker that resolved its own would be one config edit
        # away from scoring a different command than the manager collected under.
        command_axes=command_axes,
        # ...and what `d` MEANS, so `dir_err` and `bin_realized` are measured in
        # the same terms the command is issued in.
        d_rule=d_rule,
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
        demo_ok=demo_ok_report,
        command_axes=command_axes, command_mode=command_mode, d_rule=d_rule)
