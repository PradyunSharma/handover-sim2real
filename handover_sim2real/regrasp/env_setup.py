"""
Phase-5 simulator construction — ONE env serving both collection and evaluation.

The DAgger collector needs `run_omg_planner()` (only on
`HandoverSim2RealTrainEnv-v1`) and the evaluator needs `grasped_active()` (only
on `GraspBenchmarkWrapper`). `GraspBenchmarkWrapper` subclasses
`HandoverBenchmarkWrapper`, so wrapping the train env yields both and keeps a
single PyBullet connection per process — building a second env for eval would
mean a second connection and a second OMG SDF cache.

Grasp filtering follows PHASE 3 (run 21 onwards): the paper's OFFLINE
hand-collision-filtered grasp dict (`examples/valid_grasp_dict_005.pkl`) is
wired into `cfg.omg_config` BEFORE the env is built, because
`OMGPlanner.__init__` copies omg_config onto the *global* `omg_cfg` at
construction time — setting it afterwards has no effect. Our aggressive runtime
0.08 m filter stays OFF in that configuration (leaving both on double-filters
and drops ~half the scenes).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gym
import pybullet

import handover  # noqa: F401  registers the base handover envs
import handover_sim2real  # noqa: F401  registers HandoverSim2RealTrainEnv-v1

from handover_sim2real.config import get_cfg
from handover_sim2real.eval_wrapper import GraspBenchmarkWrapper
from handover_sim2real.policy import PointListener
from handover_sim2real.utils import add_sys_path_from_env, resolve_valid_grasp_dict_path

add_sys_path_from_env("GADDPG_DIR")

from experiments.config import cfg_from_file  # noqa: E402

# OMG_PLANNER_DIR is added here too, not left to handover_sim2real/train_env.py.
#
# WHY. train_env.py:15 adds it at MODULE level, but train_env is only imported
# when gym.make() resolves the registry entry point — i.e. deep inside
# build_sim_context, ~60 lines and one config load after the process started. So
# a broken OMG_PLANNER_DIR surfaced as a ModuleNotFoundError from the middle of
# environment construction, after the run had already printed its banner and
# looked healthy. Doing it at import makes the path setup a fact about process
# startup instead of a side effect of a lazy registration.
#
# THIS IS HYGIENE, NOT A FIX. add_sys_path_from_env only asserts that the
# variable EXISTS (utils.py:10) and then appends its value verbatim — it never
# checks that the directory contains an `omg` package. A variable pointing at the
# wrong place passes that assert and fails at `import omg` regardless of when the
# path was added. `assert_omg_importable` below is the part that actually catches
# it, and it is called right after.
add_sys_path_from_env("OMG_PLANNER_DIR")


def assert_omg_importable() -> str:
    """Fail loudly, and specifically, when $OMG_PLANNER_DIR has no omg package.

    The failure this exists for looks like a missing dependency and is not one:
    `sys.path` DOES contain $OMG_PLANNER_DIR (the assert in add_sys_path_from_env
    passed), the directory it names simply has no `omg/` in it. Two ways that
    happens on a cluster, and the message distinguishes them:

      * the variable was INHERITED. Every sbatch here writes
        `export OMG_PLANNER_DIR="${OMG_PLANNER_DIR:-$PWD/OMG-Planner}"`, and `:-`
        only fires when the variable is unset — so with `--export=ALL`, a stale
        value exported from some other directory in the submitting shell wins
        over the $PWD default and is carried onto the compute node.
      * the SUBMODULE is not populated. OMG-Planner is a git submodule; `git
        pull` moves the recorded gitlink but does not fetch its contents, so the
        directory can exist and be empty after a clone without --recursive.

    Returns the resolved root so a caller can log it. Raises rather than printing,
    because this runs at IMPORT — 20 collection workers each printing a banner is
    noise, and the exception text below already carries everything a preflight
    print would have shown. `preflight()` is the opt-in success log.
    """
    root = os.environ.get("OMG_PLANNER_DIR", "")
    if not root:
        raise RuntimeError("OMG_PLANNER_DIR is not set")
    pkg = Path(root) / "omg" / "__init__.py"
    if pkg.exists():
        return root

    exists = Path(root).is_dir()
    listing = ""
    if exists:
        try:
            entries = sorted(p.name for p in Path(root).iterdir())[:12]
            listing = f"\n  it contains: {entries or '(EMPTY)'}"
        except OSError:
            pass
    raise RuntimeError(
        f"OMG_PLANNER_DIR={root!r} has no omg/__init__.py, so `import omg` will "
        f"fail once gym.make() loads handover_sim2real/train_env.py.\n"
        f"  the directory {'exists' if exists else 'DOES NOT EXIST'}{listing}\n"
        f"  cwd is {os.getcwd()}\n"
        f"This is NOT a missing dependency. Two causes, both cluster-specific:\n"
        f"  1. INHERITED VALUE. The sbatch default is "
        f"${{OMG_PLANNER_DIR:-$PWD/OMG-Planner}}, and `:-` does not fire when the "
        f"variable is already set. With --export=ALL a stale value from the "
        f"submitting shell overrides the $PWD default. Fix: `unset "
        f"OMG_PLANNER_DIR` before sbatch, or re-export it from the repo root.\n"
        f"  2. UNPOPULATED SUBMODULE. OMG-Planner is a git submodule; `git pull` "
        f"moves the gitlink but does not fetch contents. Fix: `git submodule "
        f"update --init --recursive OMG-Planner`.\n"
        f"Check which: `ls $OMG_PLANNER_DIR/omg | head` on the SAME node.")


assert_omg_importable()


def preflight() -> None:
    """Log where the two vendored trees actually resolved. Call once from an
    entry point (not from the workers).

    Worth having in every job log even when nothing is wrong: the signature of
    the failure this module guards against is GADDPG_DIR resolving fine while
    OMG_PLANNER_DIR does not, IN THE SAME PROCESS, and that is only visible if
    both are recorded. `has <pkg>/` is the line that matters —
    `add_sys_path_from_env` (utils.py:10) asserts only that the variable exists,
    never that it points anywhere useful.
    """
    for var, pkg in (("GADDPG_DIR", "core"), ("OMG_PLANNER_DIR", "omg")):
        root = os.environ.get(var)
        if root is None:
            print(f"[preflight] {var:16s} = <UNSET>")
            continue
        print(f"[preflight] {var:16s} = {root}   is_dir={Path(root).is_dir()}  "
              f"has {pkg}/={(Path(root) / pkg / '__init__.py').exists()}")
    print(f"[preflight] cwd = {os.getcwd()}")


@dataclass
class SimContext:
    """Everything the collector and evaluator need from the simulator side."""

    cfg: Any
    env: GraspBenchmarkWrapper
    point_listener: PointListener
    panda_base_inv_tf: tuple
    steps_action_repeat: int

    @property
    def num_scenes(self) -> int:
        return int(self.env.num_scenes)


def build_sim_cfg(sim: dict):
    """Simulator config for Phase 4, with the Phase-3 grasp filtering wired in.

    `sim` is the SIM block of the Phase-4 config:
        cfg_file                path to the GA-DDPG-style sim yaml (pretrain.yaml)
        split                   train | val | test
        egl                     EGL GPU renderer for the offscreen hand camera
        valid_grasp_dict_path   paper's offline hand-collision filter (Phase 3)
        use_standoff            OMG plans to the standoff AND the reach beyond it
        standoff_dist           ramp EXTENT, not the standoff distance: OMG spaces
                                reach_tail_length poses over
                                linspace(0,1,n,endpoint=False)*standoff_dist, so
                                the furthest sits at standoff_dist*(1-1/n).
                                Default 0.08 with n=5 => standoff at 0.064 m.

    The renderer choice changes the point cloud, so it MUST match how the base
    dataset was collected or the policy sees a different input distribution.
    """
    cfg = get_cfg()
    cfg_from_file(filename=sim["cfg_file"], dict=cfg, merge_to_cn_dict=True)

    cfg.BENCHMARK.SPLIT = str(sim.get("split", "train"))
    # Headless by default — collection and training must never open a window, and
    # no saved run config sets this key, so every existing run is unaffected.
    # `render: true` is opt-in from a viewer (eval_regrasp_retry.py --render)
    # and has to be decided HERE: the GUI-vs-DIRECT bullet connection is made
    # inside gym.make(), so setting cfg.SIM.RENDER after build_sim_context is too
    # late to have any effect.
    cfg.SIM.RENDER = bool(sim.get("render", False))
    if bool(sim.get("egl", False)):
        cfg.SIM.BULLET.USE_EGL = True

    # Plan to the standoff AND beyond it: OMG's trajectory is
    # [free portion -> standoff] + [reach_tail waypoints -> grasp], so traj[-5]
    # is the pre-grasp standoff and traj[-1] is the pose the gripper closes at.
    # Phase 4 labels the whole thing (no standoff-plane cutoff), so the reach
    # segment must exist in the plan.
    cfg.omg_config["use_standoff"] = bool(sim.get("use_standoff", True))
    cfg.omg_config["standoff_dist"] = float(sim.get("standoff_dist", 0.08))

    vgd = resolve_valid_grasp_dict_path(sim, cfg.BENCHMARK.SETUP)
    if vgd is not None:
        cfg.omg_config["valid_grasp_dict_path"] = vgd
        print(f"[valid_grasp_dict] paper hand-collision filter ON: {vgd}")

    return cfg


def build_sim_context(cfg, sim: dict, seed: int = 0) -> SimContext:
    """Build the env + the per-episode helpers, after `build_sim_cfg`."""
    env = GraspBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))

    # Our runtime hand-collision filter. Off whenever the paper's offline dict is
    # in use (they filter the same thing; stacking them is double filtering).
    if bool(sim.get("hand_collision_filter", False)):
        env.set_hand_collision_filter(
            enable=True,
            thresh=float(sim.get("hand_collision_thresh", 0.08)),
            points_radius=float(sim.get("hand_points_radius", 0.35)),
        )
        print(
            "[hand_collision_filter] runtime filter ON at "
            f"{float(sim.get('hand_collision_thresh', 0.08)):.3f} m"
        )

    point_listener = PointListener(cfg, seed=seed)

    # NOTE: no scripted grasp-and-back here. Phase 4 scores the PHASE-3 criterion
    # (hold the close, object secured), not the benchmark's carry-to-GOAL_CENTER
    # SUCCESS, so nothing ever drives the retreat. See dagger/evaluator.py.

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION
    )
    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

    return SimContext(
        cfg=cfg,
        env=env,
        point_listener=point_listener,
        panda_base_inv_tf=panda_base_inv_tf,
        steps_action_repeat=steps_action_repeat,
    )
