"""
Phase-4 simulator construction — ONE env serving both collection and evaluation.

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

from dataclasses import dataclass
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
    cfg.SIM.RENDER = False
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
