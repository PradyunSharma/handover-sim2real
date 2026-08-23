"""
Chained regrasping: retry a failed handover from PART-WAY along the failed
trajectory, under a different pinned grasp.

WHAT THIS IS FOR, and how it differs from the `retry_at_k` already in
`evaluator.py`. That one rolls all four grasps of a scene as four INDEPENDENT
episodes, each from a full `env.reset()`, and then asks "did any of them
succeed". It is a ceiling: it charges nothing for the failed attempt, because
attempt 2 never sees the world attempt 1 left behind. This module runs the
actual sequence instead —

    roll grasp 0 -> fail -> rewind to `rewind_frac` of that trajectory
                 -> switch to grasp 1 -> keep rolling -> fail -> rewind again
                 -> ... until success or the grasps run out

— and the branch point is a state the policy really reached, not the home pose.

HOW THE REWIND IS IMPLEMENTED, and the one thing it is NOT. Every executed step
records the target joint position it was driven with, so a rewind is
`env.reset(idx=scene)` followed by a replay of the first `r` recorded targets.
The physics is deterministic under a fixed command sequence, so the replay lands
on the same state the original attempt passed through — `replay_err` measures
that per attempt and should read ~0.

The alternative was `pybullet.saveState`/`restoreState`, and it was rejected:
the env's Python-level state is not in the PyBullet snapshot. `mano._frame`,
`ycb._frame`, `ycb._released`, `_release_step_counter_*`, `_elapsed_steps` and
`_dropped` all live outside it, and `ycb.release()` additionally converts the
object from a kinematic to a dynamic body. Restoring bullet alone would rewind
the arm while leaving the human's playback index — and the release handshake —
where the failure left them. Replay moves every one of those through its normal
code path, so there is no list of counters to keep in sync as handover-sim
changes.

THE HONEST CAVEAT, stated here because it bounds what the number means: THE
HUMAN REWINDS TOO. The hand is a DexYCB playback keyed on `mano._frame`, so a
reset-and-replay puts the human back where they were at step `r` as well. A real
regrasp does not get that — the person keeps moving while the robot backs off,
and the object is 2-3 s further along its handover. So this is still an upper
bound, just a much tighter one than the reset-based `retry_at_k`: the robot pays
for its failed approach, the human does not. The version that charges both is a
retreat controller that drives the arm back along its own recorded joint path
WITHOUT resetting, and it is deliberately not built here — it needs gripper
reopening, a collision-safe reverse servo, and a policy for the failures that
already ended the episode (human contact and object drop leave a world there is
no retrying from at all).

BUDGET. `shared` (the default) gives each attempt `max_steps` policy steps in
TOTAL including the replayed prefix, so every attempt spans the same horizon as
a plain eval episode and `chained_retry_at_1` is directly comparable with
`succ_g0`. `fresh` gives the resumed segment a full `max_steps` of its own. Both
stay inside the benchmark's own limit, which is 13.0 s / 0.15 s = 86 policy
steps against a configured horizon of 50.

WHICH TRAJECTORY THE REWIND SLICES. `previous` (the default) rewinds into the
attempt that just failed — its trajectory is the replayed prefix plus whatever
the new grasp did with it, so the branch point creeps along as attempts
accumulate. `first` always branches from the same point on attempt 0, giving one
common prefix and N-1 sibling branches. They coincide when the attempts come out
similar lengths and diverge when a retry runs long.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from collect_bc_dataset import (  # noqa: E402
    _point_cloud, _robot_state, ee_grasp_pose_error,
)
from rollout_bc_policy import action_to_target_joint  # noqa: E402

from handover_sim2real.rl.rollout_worker import (  # noqa: E402
    _status_name, grasp_held_after_hold,
)
from handover_sim2real.regrasp.collector import derived_standoff_pose  # noqa: E402
from handover_sim2real.regrasp.evaluator import (  # noqa: E402
    EvalParams, _ee_mat_from_state,
)
from handover_sim2real.regrasp import directions as _rg_dirs  # noqa: E402
from handover_sim2real.regrasp import retry as _rg_retry  # noqa: E402
from handover_sim2real.regrasp.grasp_box import grasp_opportunity  # noqa: E402
from handover_sim2real.regrasp.pregrasp import open_loop_reach  # noqa: E402

REWIND_MODES = ("previous", "first")
BUDGETS = ("shared", "fresh")

# Failures that leave no world to retry from. `retryable` records this per
# attempt: a chained retry after HUMAN_CONTACT or DROP is only possible at all
# because the rewind resets the simulator, and a retreat-based implementation
# could not offer it. Reported, never enforced — filtering on it would silently
# change the denominator of every rate below.
TERMINAL_REASONS = ("HUMAN_CONTACT", "DROP")


@dataclass
class RetryParams:
    """How the chain backs up and how much rope each attempt gets."""

    rewind_frac: float = 0.30       # branch at floor(frac * len(trajectory))
    rewind_mode: str = "previous"   # previous | first  (see the module docstring)
    budget: str = "shared"          # shared | fresh
    max_attempts: int = 0           # 0 = every grasp the pin table holds
    replay_tol: float = 0.005       # m of EE divergence before the replay is doubted
    save_clouds: bool = False       # store [T, N, C] clouds as well as states
    verbose: bool = True

    def __post_init__(self):
        if self.rewind_mode not in REWIND_MODES:
            raise ValueError(f"rewind_mode must be one of {REWIND_MODES}, "
                             f"got {self.rewind_mode!r}")
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}, got {self.budget!r}")
        if not 0.0 <= self.rewind_frac < 1.0:
            raise ValueError("rewind_frac must be in [0, 1) — 1.0 would branch at "
                             "the failure itself, which is not a rewind")


def _pose_error(ee_mat, grasp_pose) -> tuple[float, float]:
    """`ee_grasp_pose_error` for an EE given as a 4x4 rather than as an `obs`.

    Same convention, deliberately: the prefix of a retried attempt is REPLAYED,
    so there is no `obs` to read for those steps, but `robot_states` carries the
    EE pose (rs[18:21] xyz, rs[21:25] wxyz) and `min_pos` / `min_rot` are only
    meaningful if they cover the whole trajectory the attempt executed — including
    the part it inherited.
    """
    ee_R, ee_t = np.asarray(ee_mat)[:3, :3], np.asarray(ee_mat)[:3, 3]
    g_R, g_t = np.asarray(grasp_pose)[:3, :3], np.asarray(grasp_pose)[:3, 3]
    cos = (np.trace(ee_R.T @ g_R) - 1.0) / 2.0
    return (float(np.linalg.norm(ee_t - g_t)),
            float(np.arccos(np.clip(cos, -1.0, 1.0))))


@dataclass
class Attempt:
    """One grasp's rollout: the trajectory it executed and how it ended.

    `rows` is the WHOLE trajectory — replayed prefix first, then the steps this
    attempt's policy actually chose — so an attempt is self-contained and the
    stored episode reads like any other. `prefix_len` says where the seam is.
    """

    scene_idx: int = 0
    attempt: int = 0
    grasp_idx: int = 0
    rows: list = field(default_factory=list)
    branch_step: int = 0        # index in the SOURCE trajectory it branched at
    prefix_len: int = 0         # steps replayed (== branch_step, kept separate
                                # so a clamped branch is visible)
    replay_err: float = 0.0     # m between the replayed EE and the recorded one
    budget_steps: int = 0
    success: int = 0
    grasped: int = 0
    closed: int = 0
    near: int = 0
    close_step: int = -1
    pos_err: float = float("nan")
    rot_err: float = float("nan")
    min_pos: float = float("nan")
    min_rot: float = float("nan")
    reach_pos_err: float = float("nan")
    reach_rot_err: float = float("nan")
    had_chance: int = 0
    box_chance: int = 0
    box_taken: int = 0
    dist: float = float("nan")
    status: int = 0
    reason: str = ""
    ee_final: object = None
    grasp_pose: object = None
    d_world: object = None        # the command this attempt was given
    bin_idx: int = -1             # which bin that direction is
    dir_err: float = float("nan")  # angle to the achieved approach axis, degrees
    stop_reason: str = ""         # SIGNAL_HUMAN when the ladder ran out

    @property
    def retryable(self) -> bool:
        """Would a NON-resetting retreat have had a world left to retry in?"""
        return not any(t in self.reason for t in TERMINAL_REASONS)

    @property
    def executed_jp(self) -> list:
        """Target joints of the steps that actually stepped the env, in order.

        The closing step drives no plain `env.step` — the hold and the scoring
        take over — so it contributes no target and is excluded here. That is
        also why a branch can never land on it.
        """
        return [r["jp"] for r in self.rows if r["jp"] is not None]


def _ee_pos(obs) -> np.ndarray:
    return np.asarray(
        obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3],
        dtype=np.float64)


def _run_attempt(sim, runner, scene_idx: int, *, grasp_pose, target_pose,
                 params: EvalParams, prefix_rows: list, prefix_jp: list,
                 expect_ee, budget_steps: int, save_clouds: bool,
                 attempt: int = 0, grasp_idx: int = 0, branch_step: int = 0,
                 n_attempts: int = 1, viz=None) -> Attempt:
    """Reset, replay `prefix_jp`, then roll the policy under `grasp_pose`.

    The replay is silent about the policy: it re-executes recorded joint targets,
    never re-queries the network. That is what makes the prefix reproducible even
    though `PointListener` subsamples the cloud with a live RNG — the physics
    depends only on the command sequence, and the policy is not in the loop until
    the branch point is reached.

    `viz` is an optional `chain_viz.ChainViz`. Its hooks are called AFTER the
    reset, never before: `env.reset` can drop through to `pybullet.resetSimulation`
    when the scene's bodies change, and that removes every debug item — so an
    overlay drawn ahead of the reset would silently vanish on some scenes and
    survive on others.
    """
    env = sim.env
    pregrasp = str(params.target) == "pregrasp"
    if target_pose is None:
        target_pose = grasp_pose

    att = Attempt(scene_idx=int(scene_idx), attempt=int(attempt),
                  grasp_idx=int(grasp_idx), branch_step=int(branch_step),
                  prefix_len=len(prefix_jp), budget_steps=int(budget_steps),
                  grasp_pose=grasp_pose)

    obs = env.reset(idx=int(scene_idx))
    sim.point_listener.reset()
    if viz is not None:
        viz.begin_attempt(int(attempt), int(grasp_idx), int(branch_step),
                          int(n_attempts))

    # ---- replay the inherited prefix ----------------------------------------
    replay_done = False
    for i, jp in enumerate(prefix_jp):
        for _ in range(sim.steps_action_repeat):
            obs, _, replay_done, _info = env.step(jp)
            if replay_done:
                break
        if viz is not None:
            viz.replay_step(i, _ee_pos(obs))
        if replay_done:
            break
    if viz is not None and prefix_jp and not replay_done:
        viz.mark_branch(_ee_pos(obs))
    if replay_done:
        # The prefix is a strict prefix of a trajectory that survived those
        # steps, so a deterministic replay cannot end early. If it does, the
        # determinism assumption is broken and every number downstream is
        # suspect — say so rather than carry on.
        att.reason = "REPLAY_DIVERGED"
        att.rows = list(prefix_rows)
        return att
    if expect_ee is not None:
        got = np.asarray(
            obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3],
            dtype=np.float64)
        att.replay_err = float(np.linalg.norm(got - np.asarray(expect_ee)))

    # ---- resume: the policy takes over, conditioned on the NEW grasp --------
    runner.reset()
    # The command is the DIRECTION, derived from the grasp exactly as the
    # collector derives it. `grasp_pose` stays around because the expert's CLOSE
    # label and the geometric scores are still measured against it — it just is
    # no longer what the policy is told.
    runner.set_direction(None if grasp_pose is None
                         else _rg_dirs.approach_direction(grasp_pose))

    rows = list(prefix_rows)
    # The inherited steps are re-scored against THIS attempt's grasp: the arm
    # really did pass through those poses, and `min_pos` over only the resumed
    # segment would hide an approach that was already close before the switch.
    min_pos = min_rot = float("inf")
    had_chance = False
    for r in rows:
        if grasp_pose is not None:
            pe, re_ = _pose_error(_ee_mat_from_state(r["rs"]), target_pose)
            min_pos, min_rot = min(min_pos, pe), min(min_rot, re_)
            if pe <= params.close_pos_thresh and re_ <= params.close_rot_thresh:
                had_chance = True

    prev_act6d = (np.asarray(prefix_rows[-1]["act"][:6], dtype=np.float32)
                  if prefix_rows else np.zeros(6, dtype=np.float32))
    status, done, info = 0, False, {}
    dist = float("nan")
    pos_err = rot_err = float("nan")
    reach_pos_err = reach_rot_err = float("nan")
    box_chance = box_taken = False
    close_step = -1
    success = grasped = False
    ee_final = None
    reason = ""

    for step in range(int(budget_steps)):
        pc = _point_cloud(obs, sim.point_listener, sim.panda_base_inv_tf)
        rs = _robot_state(obs, prev_act6d)
        ee_final = _ee_mat_from_state(rs)
        action = runner.act(pc, rs)
        prev_act6d = action[:6].astype(np.float32)

        opportunity = False
        if params.box_check:
            opportunity, _frac = grasp_opportunity(env, params.box)
            box_chance |= bool(opportunity)

        ee_pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
        ycb_pos = env.ycb.bodies[env.ycb.ids[0]].link_state[0, 6, 0:3].numpy()
        dist = float(np.linalg.norm(ee_pos - ycb_pos))

        if target_pose is not None:
            pe, re_ = ee_grasp_pose_error(obs, target_pose)
            min_pos, min_rot = min(min_pos, pe), min(min_rot, re_)
            if pe <= params.close_pos_thresh and re_ <= params.close_rot_thresh:
                had_chance = True

        row = {"rs": np.asarray(rs, dtype=np.float32),
               "act": np.asarray(action, dtype=np.float32),
               "ee_pos": np.asarray(ee_pos, dtype=np.float64),
               "jp": None,
               "pc": np.asarray(pc, dtype=np.float32) if save_clouds else None}

        if viz is not None:
            viz.step(step, ee_pos, int(grasp_idx), obs=obs, pc=pc)

        if action[6] < 0.5:
            # ---- the policy committed ----
            close_step = len(rows)
            box_taken = bool(opportunity)
            if target_pose is not None:
                pos_err, rot_err = ee_grasp_pose_error(obs, target_pose)
            rows.append(row)               # closing step: no jp, never replayed

            if pregrasp:
                obs, pushed_done, st = open_loop_reach(
                    env, obs, sim.steps_action_repeat,
                    dist=params.forward_dist, num_steps=params.forward_steps)
                if grasp_pose is not None:
                    reach_pos_err, reach_rot_err = ee_grasp_pose_error(obs, grasp_pose)
                if pushed_done:
                    status, reason = st, _status_name(st)
                    break

            held, obs = grasp_held_after_hold(
                env, obs, sim.steps_action_repeat, params.hold_steps)
            grasped = bool(env.grasped_active())
            near = bool(pos_err <= params.close_pos_thresh
                        and rot_err <= params.close_rot_thresh)
            success = held if params.success_mode == "stable_grasp" else near
            if success:
                reason = "GRASP_OK"
            elif bool(getattr(env, "_dropped", False)):
                reason = "DROP"
            elif not bool(env.ycb.released):
                reason = "NO_RELEASE"
            else:
                reason = "GRASP_MISS"
            break

        target_jp = action_to_target_joint(action, obs)
        row["jp"] = np.asarray(target_jp, dtype=np.float64)
        rows.append(row)
        for _ in range(sim.steps_action_repeat):
            obs, _, done, info = env.step(target_jp)
            if done:
                break
        status = info.get("status", 0)
        if done:
            reason = _status_name(status)
            break
    else:
        reason = "TIMEOUT"

    att.rows = rows
    att.success = int(success)
    att.grasped = int(grasped)
    att.closed = int(close_step >= 0)
    att.near = int(bool(pos_err <= params.close_pos_thresh
                        and rot_err <= params.close_rot_thresh))
    att.close_step = int(close_step)
    att.pos_err, att.rot_err = float(pos_err), float(rot_err)
    att.min_pos = float(min_pos) if np.isfinite(min_pos) else float("nan")
    att.min_rot = float(min_rot) if np.isfinite(min_rot) else float("nan")
    att.reach_pos_err, att.reach_rot_err = float(reach_pos_err), float(reach_rot_err)
    att.had_chance = int(had_chance)
    att.box_chance, att.box_taken = int(box_chance), int(box_taken)
    att.dist = float(dist)
    att.status, att.reason = int(status), reason
    att.ee_final = ee_final
    return att


def chained_retry_scene(sim, runner, scene_idx: int, pose_of_bin, *,
                        params: EvalParams, retry: RetryParams,
                        anchor_R=None, feasible=None, viz=None) -> list:
    """Run the chain on one scene. Returns the attempts made, in order.

    THE SEQUENCE COMES FROM THE RETRY LADDER, NOT FROM A SLOT LIST. Phase 5 took
    `grasp_poses[k]` for attempt k, which tied the number of attempts to the
    number of pinned grasps and made `attempt == grasp_idx` by construction.
    Under Regrasp the two are independent: the ladder can command up to
    `retry.max_attempts` directions drawn from the k bins, while a scene supplies
    at most two DEMONSTRATED ones. `retry.py` picks each next direction as the
    surviving bin furthest from everything already attempted.

    `pose_of_bin` maps bin index -> a 4x4 world grasp realising that direction.
    A pose is still needed per attempt because OMG's CLOSE label and the
    geometric scores are measured against one; it is simply no longer what the
    policy is told. Bins with no pose are skipped — commanding a direction this
    scene cannot realise would score the policy on an impossible instruction.

    Stops at the first success, so a scene solved on the first direction costs
    exactly one rollout.
    """
    pregrasp = str(params.target) == "pregrasp"
    anchor_R = np.eye(3) if anchor_R is None else np.asarray(anchor_R, dtype=np.float64)
    # Only bins this scene can actually realise, intersected with any caller
    # restriction (on s0/train that is {+x, +y, -y, +z}: -z is reachable by no
    # scene and -x by twelve).
    have = {int(b) for b, p in dict(pose_of_bin).items() if p is not None}
    if feasible is not None:
        have &= {int(b) for b in feasible}
    rp = _rg_retry.RetryParams(max_attempts=int(retry.max_attempts or 4))
    st = _rg_retry.RetryState()

    if viz is not None:
        viz.begin_scene(int(scene_idx), dict(pose_of_bin), anchor_R)

    attempts: list[Attempt] = []
    prefix_rows: list = []
    prefix_jp: list = []
    expect_ee = None
    branch_step = 0

    k = 0
    while True:
        d_world, bin_idx, info = _rg_retry.next_direction(
            st, anchor_R, rp, feasible=have)
        if d_world is None:
            # The ladder is out of hypotheses. That is a RESULT — the robot
            # should ask the person to re-present the object — not an error, so
            # it is recorded on the last attempt rather than dropped.
            if attempts:
                attempts[-1].stop_reason = info["why"]
            if retry.verbose:
                print(f"    scene {scene_idx:4d}: {info['stop']} ({info['why']}) "
                      f"after {st.attempts} attempt(s)")
            break

        grasp_pose = dict(pose_of_bin).get(bin_idx)
        target_pose = grasp_pose
        if pregrasp and grasp_pose is not None:
            target_pose = derived_standoff_pose(
                grasp_pose, params.standoff_dist, params.reach_tail)

        budget = (params.max_steps - len(prefix_jp)
                  if retry.budget == "shared" else params.max_steps)
        att = _run_attempt(
            sim, runner, scene_idx, grasp_pose=grasp_pose,
            target_pose=target_pose, params=params,
            prefix_rows=prefix_rows, prefix_jp=prefix_jp, expect_ee=expect_ee,
            budget_steps=max(1, int(budget)), save_clouds=retry.save_clouds,
            attempt=k, grasp_idx=bin_idx, branch_step=branch_step,
            n_attempts=rp.max_attempts, viz=viz)
        att.d_world = np.asarray(d_world, dtype=np.float64)
        att.bin_idx = int(bin_idx)
        if att.ee_final is not None:
            att.dir_err = float(_rg_dirs.angle_between(
                d_world, _rg_dirs.approach_direction(np.asarray(att.ee_final))))
        attempts.append(att)
        if viz is not None:
            viz.end_attempt(att)

        if retry.verbose:
            seam = (f"  (branched at {att.branch_step}, replay_err "
                    f"{att.replay_err * 1000:.1f} mm)" if att.prefix_len else "")
            print(f"    scene {scene_idx:4d} attempt {k} "
                  f"{_rg_dirs.BIN_NAMES[bin_idx]:<16} "
                  f"{'OK ' if att.success else '-- '}{att.reason:<14} "
                  f"steps={len(att.rows):2d} dir_err={att.dir_err:5.1f} "
                  f"min_pos={att.min_pos:.3f}{seam}")

        if att.success:
            break
        st.mark(d_world, bin_idx, att.reason)
        k += 1

        # ---- pick the branch point for the next direction ----
        src = attempts[0] if retry.rewind_mode == "first" else attempts[-1]
        exec_jp = src.executed_jp
        if not exec_jp:
            # The source never executed a plain step (it closed immediately, or
            # died on step 0). There is nothing to rewind INTO, so the next grasp
            # starts from home — which is the reset protocol, recorded as such.
            prefix_rows, prefix_jp, expect_ee, branch_step = [], [], None, 0
            continue
        r = int(math.floor(retry.rewind_frac * len(src.rows)))
        r = max(0, min(r, len(exec_jp)))
        prefix_jp = exec_jp[:r]
        prefix_rows = [dict(row) for row in src.rows[:r]]
        expect_ee = src.rows[r]["ee_pos"] if r < len(src.rows) else None
        branch_step = r

    if viz is not None:
        viz.end_scene()
    return attempts


def chained_metrics(per_scene: dict, max_attempts: int = 4) -> dict:
    """Aggregate the chains. `per_scene` maps scene_idx -> [Attempt].

    `chained_retry_at_k` is the headline and is the number to put beside
    `evaluator`'s `retry_at_k`: same question, but each attempt after the first
    starts from a state the policy actually drove itself into. The gap between
    the two is what the reset-based version was giving away.

    `chained_retry_at_1` should reproduce `succ_g0` up to rollout noise, because
    attempt 0 IS the slot-0 episode. If it does not, the chain is not starting
    where the independent evaluation starts and nothing below is comparable.
    """
    if not per_scene:
        return {}
    n_scenes = len(per_scene)
    # `max_attempts`, NOT a slot count. Phase 5 tied the two together because
    # attempt k WAS slot k; the ladder decouples them, so the metric arrays are
    # sized by how many attempts are allowed, not by how many directions a scene
    # happened to be pinned with.
    out = {"n_scenes": n_scenes, "max_attempts": int(max_attempts)}

    for k in range(1, max_attempts + 1):
        hits = sum(1 for atts in per_scene.values()
                   if any(a.success for a in atts[:k]))
        out[f"chained_retry_at_{k}"] = hits / n_scenes

    # Success rate OF each attempt, over the scenes that got that far. Denominator
    # shrinks with k (a solved scene never makes attempt 2), so this is "given the
    # first k grasps failed, does grasp k work" — a conditional, not a per-slot
    # rate, and not comparable with `succ_g{k}` from the independent evaluation.
    for k in range(max_attempts):
        reached = [atts[k] for atts in per_scene.values() if len(atts) > k]
        out[f"attempt_{k}_n"] = len(reached)
        out[f"attempt_{k}_succ"] = (sum(a.success for a in reached) / len(reached)
                                    if reached else float("nan"))

    solved = [atts for atts in per_scene.values() if any(a.success for a in atts)]
    out["solved_rate"] = len(solved) / n_scenes
    out["mean_attempts"] = float(np.mean([len(a) for a in per_scene.values()]))
    out["mean_attempts_to_success"] = (
        float(np.mean([len(a) for a in solved])) if solved else float("nan"))

    # How much of the failed approach each retry actually inherited. A branch
    # step of 0 means the chain degenerated to a plain reset for that attempt.
    branches = [a.branch_step for atts in per_scene.values() for a in atts[1:]]
    out["mean_branch_step"] = float(np.mean(branches)) if branches else float("nan")
    out["n_branch_at_home"] = int(sum(1 for b in branches if b == 0))

    # Replay fidelity. This is the assumption the whole module rests on, so it is
    # a reported metric and not an assert: a large max means the sim did not
    # reproduce the recorded trajectory and the branch states are not the ones
    # the policy really visited.
    errs = [a.replay_err for atts in per_scene.values() for a in atts if a.prefix_len]
    out["replay_err_mean"] = float(np.mean(errs)) if errs else float("nan")
    out["replay_err_max"] = float(np.max(errs)) if errs else float("nan")
    out["n_replay_diverged"] = int(sum(
        1 for atts in per_scene.values() for a in atts
        if a.reason == "REPLAY_DIVERGED"))

    # Of the failures that triggered a retry, how many left a world a NON-
    # resetting retreat could have retried in. The complement is the share of
    # this metric's advantage that comes purely from rewinding the simulator.
    retried_after = [atts[k] for atts in per_scene.values()
                     for k in range(len(atts) - 1)]
    out["retryable_frac"] = (
        float(np.mean([a.retryable for a in retried_after]))
        if retried_after else float("nan"))

    # Direction tracking across the whole ladder: did the policy go where each
    # successive command pointed, or does it do the same thing every attempt —
    # which would make the retry inert however good the OR-over-attempts looks.
    errs = [a.dir_err for atts in per_scene.values() for a in atts
            if a.dir_err == a.dir_err]
    out["dir_err"] = float(np.mean(errs)) if errs else float("nan")
    out["dir_track"] = float(1.0 - np.mean(errs) / 90.0) if errs else float("nan")
    by_bin = {}
    for atts in per_scene.values():
        for a in atts:
            if a.bin_idx >= 0:
                by_bin.setdefault(a.bin_idx, []).append(a.success)
    for b, v in sorted(by_bin.items()):
        out[f"chain_succ_bin_{b}"] = float(np.mean(v))
        out[f"chain_n_bin_{b}"] = len(v)
    # How often the ladder ran out rather than the policy simply failing. A high
    # rate here means the feasibility mask is too tight, not that the policy is
    # bad, and the two must not be read as the same thing.
    n_sig = sum(1 for atts in per_scene.values()
                if atts and atts[-1].stop_reason)
    out["signal_human_rate"] = n_sig / n_scenes

    reasons = {}
    for atts in per_scene.values():
        for a in atts:
            reasons[a.reason] = reasons.get(a.reason, 0) + 1
    out["reasons"] = reasons
    return out
