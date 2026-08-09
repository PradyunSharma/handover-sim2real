"""
Closed-loop evaluation, used for DAgger's "return best pi_i on validation" step.

SUCCESS IS THE PHASE-3 CRITERION, NOT THE BENCHMARK'S. Phase 4, like Phase 3, is
about reaching the grasp and committing the close at the right pose — there is no
carry-to-goal, so `EpisodeStatus.SUCCESS` (which requires the hand to dwell inside
a 15 cm ball at GOAL_CENTER) can never fire and would score every episode 0. Both
success modes are imported verbatim from `handover_sim2real.rl.rollout_worker`
rather than reimplemented, so the number Phase 4 reports and the reward Phase 3
optimises cannot drift apart:

    stable_grasp (default, the mode every recent Phase-3 run uses)
        After the committed close, hold the gripper shut in place for
        `hold_steps` policy-steps, then require the object to be SECURED:
        handover-sim's release handshake fired (`ycb.released` — the human only
        lets go once the robot really grips) AND no drop AND no human-contact
        failure during the hold.  -> `grasp_held_after_hold`

    proximity
        The EE is within (close_pos_thresh, close_rot_thresh) of the grasp pose
        at the moment of the close. Pure geometry, no physics. This is exactly
        the predicate the Phase-4 collector uses to emit its CLOSE label, so it
        measures label agreement rather than task outcome. -> `ee_grasp_pose_error`

The grasp pose that `proximity` scores against comes from the grasp pin table
when one is loaded (free — it stores the world pose), otherwise from a single
step-0 OMG plan. `stable_grasp` needs no grasp pose at all, so with it the
evaluator makes no OMG calls and an eval sweep stays cheap enough to run after
every DAgger iteration.

OPPORTUNITY IS MEASURED TWO WAYS. `chance_rate` gates on proximity to the pinned
grasp and is therefore a pin-agreement measure, not an opportunity measure — it
reads 0.03-0.05 in runs that succeed 60-70% of the time, because the policy's
grasps are real but off-pose. `box_chance_rate` / `box_taken_rate` instead ask,
per step, whether object material actually sits between the open finger pads
(`dagger/grasp_box.py`, ray-cast against ground-truth collision geometry). Both
are reported; the geometric pair is the honest one, the pinned pair is kept so
runs 4-14 stay comparable.

Protocol per scene:
  * observe -> runner.act() -> Delta-ee-pose -> IK -> step
  * the first time the policy commands a close, score it (hold + secured check),
    and end the episode
  * otherwise step until the horizon or a benchmark failure (human contact/drop)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from collect_bc_dataset import (  # noqa: E402
    _point_cloud, _robot_state, ee_grasp_pose_error,
)
from rollout_bc_policy import action_to_target_joint  # noqa: E402

# The Phase-3 success test itself — imported, never reimplemented.
from handover_sim2real.rl.rollout_worker import (  # noqa: E402
    _status_name, grasp_held_after_hold,
)
from handover_sim2real.dagger.grasp_box import (  # noqa: E402
    BoxParams, grasp_opportunity,
)

SUCCESS_MODES = ("stable_grasp", "proximity")


@dataclass
class EvalParams:
    max_steps: int = 50
    success_mode: str = "stable_grasp"   # see the module docstring
    hold_steps: int = 3                  # stable_grasp: policy-steps held shut
    close_pos_thresh: float = 0.02       # proximity: metres
    close_rot_thresh: float = 0.34       # proximity: radians (~19.5 deg)
    # Geometric opportunity test (see dagger/grasp_box.py). Runs ALONGSIDE the
    # pinned-pose `had_chance`, never replacing it, so `chance_rate` keeps the
    # meaning it had in runs 4-14 and the two remain comparable.
    box_check: bool = True
    box: BoxParams = None
    verbose: bool = False

    def __post_init__(self):
        if self.success_mode not in SUCCESS_MODES:
            raise ValueError(f"success_mode must be one of {SUCCESS_MODES}, "
                             f"got {self.success_mode!r}")
        if self.box is None:
            self.box = BoxParams()


def _resolve_grasp_pose(sim, scene_idx: int, pin_table):
    """World 4x4 of the grasp this scene's close is scored against.

    Prefers the pin table (it stores the committed grasp's world pose, so this
    costs nothing and is by construction the same pose the collector labelled
    towards). Falls back to one step-0 OMG plan, as Phase 3 does. None if
    neither is available — the caller then reports NaN pose errors.
    """
    if pin_table is not None:
        entry = pin_table.entries.get(int(scene_idx))
        if entry is not None:
            return np.asarray(entry["ee_pose_world"], dtype=np.float64)

    plan, _ = sim.env.run_omg_planner(
        int(sim.cfg.RL_MAX_STEP), int(scene_idx), reset_scene=True)
    if plan is None:
        return None
    return sim.env.get_omg_goal_grasp_pose()


def _eval_episode(sim, runner, scene_idx, *, params: EvalParams,
                  grasp_pose=None) -> dict:
    env = sim.env
    obs = env.reset(idx=scene_idx)
    sim.point_listener.reset()
    runner.reset()

    prev_act6d = np.zeros(6, dtype=np.float32)
    status, done, info = 0, False, {}
    dist = float("nan")
    pos_err = rot_err = float("nan")
    # Closest the EE ever came to the grasp, over the WHOLE episode. Unlike the
    # at-close errors these exist even when the policy never closes, which is
    # exactly the case where a success rate of 0 tells you nothing.
    min_pos = min_rot = float("inf")
    # Was the EE ever inside BOTH tolerances AT THE SAME STEP — i.e. was there a
    # step at which closing would have been a correct grasp. Deliberately not
    # `min_pos <= t and min_rot <= t`, which can be satisfied at two different
    # steps and would over-report the opportunity.
    had_chance = False
    # The GEOMETRIC opportunity: was the object ever really between the open
    # jaws (dagger/grasp_box.py), independent of the pin. `box_taken` is the one
    # that answers "given a chance, did it take it" — it is only set when the
    # close is commanded ON such a step, not merely in an episode that had one.
    box_chance = False
    box_taken = False
    box_steps = 0
    box_frac_max = 0.0
    close_step = -1
    success = False
    grasped = False
    reason = ""

    for step in range(params.max_steps):
        pc = _point_cloud(obs, sim.point_listener, sim.panda_base_inv_tf)
        rs = _robot_state(obs, prev_act6d)
        action = runner.act(pc, rs)          # [7], ch6 in {0,1}
        prev_act6d = action[:6].astype(np.float32)

        # Read from `obs`, i.e. the state the policy just acted FROM — so an
        # opportunity is scored against the pose at which the decision was made,
        # not one the action has already moved away from.
        opportunity = False
        if params.box_check:
            opportunity, box_frac = grasp_opportunity(env, params.box)
            box_frac_max = max(box_frac_max, box_frac)
            if opportunity:
                box_chance = True
                box_steps += 1

        ee_pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
        ycb_pos = env.ycb.bodies[env.ycb.ids[0]].link_state[0, 6, 0:3].numpy()
        dist = float(np.linalg.norm(ee_pos - ycb_pos))

        if grasp_pose is not None:
            pe, re_ = ee_grasp_pose_error(obs, grasp_pose)
            min_pos, min_rot = min(min_pos, pe), min(min_rot, re_)
            if pe <= params.close_pos_thresh and re_ <= params.close_rot_thresh:
                had_chance = True

        if action[6] < 0.5:
            # ---- the policy committed a close: score it (Phase-3 criterion) ----
            close_step = step
            box_taken = bool(opportunity)

            # Geometry FIRST, from the pose the close was committed at — the hold
            # moves the fingers, so this has to be read before it runs.
            if grasp_pose is not None:
                pos_err, rot_err = ee_grasp_pose_error(obs, grasp_pose)
            near = bool(pos_err <= params.close_pos_thresh
                        and rot_err <= params.close_rot_thresh)

            # The hold runs in BOTH modes: it does not affect the proximity score
            # (already read above) and it is what makes `grasped` meaningful.
            held, obs = grasp_held_after_hold(
                env, obs, sim.steps_action_repeat, params.hold_steps)
            grasped = bool(env.grasped_active())

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
        for _ in range(sim.steps_action_repeat):
            obs, _, done, info = env.step(target_jp)
            if done:
                break
        status = info.get("status", 0)
        if done:
            reason = _status_name(status)    # human contact / drop / bench timeout
            break
    else:
        reason = "TIMEOUT"                   # horizon reached, never closed

    return {
        "scene_idx": int(scene_idx),
        "success": int(success),
        "grasped": int(grasped),
        "closed": int(close_step >= 0),
        "near": int(bool(pos_err <= params.close_pos_thresh
                         and rot_err <= params.close_rot_thresh)),
        "close_step": close_step,
        "dist": dist,
        "pos_err": pos_err,
        "rot_err": rot_err,
        "min_pos": min_pos if np.isfinite(min_pos) else float("nan"),
        "min_rot": min_rot if np.isfinite(min_rot) else float("nan"),
        "had_chance": int(had_chance),
        # Reached a graspable pose but did not come away with the object —
        # either never closed there, or closed and lost it. Separates "never got
        # there" from "got there and blew it", which a success rate cannot.
        "missed": int(had_chance and not success),
        # ---- geometric opportunity (grasp_box.py), pin-independent ----
        "box_chance": int(box_chance),      # object was ever in the open jaws
        "box_taken": int(box_taken),        # ...and the close was commanded there
        "box_missed": int(box_chance and not success),
        "box_steps": int(box_steps),        # how long the window stayed open
        "box_frac_max": float(box_frac_max),  # best jaw occupancy seen, for
                                              # recalibrating min_frac offline
        "status": int(status),
        "reason": reason,
    }


def evaluate_policy(sim, runner, scenes, *, params: EvalParams,
                    pin_table=None) -> dict:
    """Roll the policy over `scenes` and aggregate. Returns rates in [0, 1].

    `success_rate` is the Phase-3 criterion selected by `params.success_mode`.
    The others are diagnostics that split a failure into its stage:
        close_rate      the policy committed a close at all
        near_rate       ...and did it within the CLOSE-label tolerances
                        (NaN-safe: 0 for scenes with no grasp pose to compare to)
        grasp_rate      ...and both fingers ended the hold on the object
        success_rate    ...and the object was secured (release, no drop)
    Reading them left to right localises where the policy is losing episodes.

    Opportunity is reported TWICE, against two different definitions, because
    they disagree and the disagreement is itself the finding:
        chance_rate     the EE was within tolerance of the PINNED grasp
        box_chance_rate the object was geometrically between the open jaws
    The first is near-zero in runs that succeed 60-70% of the time (it is really
    measuring pin agreement); the second counts off-pose grasps as the
    opportunities they are. `box_taken_rate` is the conversion of the latter.
    """
    rows = []
    for i, scene in enumerate(scenes):
        # proximity needs a grasp pose; stable_grasp only wants one if a pin
        # table makes it free (it is a diagnostic there, not the score).
        grasp_pose = None
        if params.success_mode == "proximity" or pin_table is not None:
            grasp_pose = _resolve_grasp_pose(sim, int(scene), pin_table)
            if grasp_pose is None and params.success_mode == "proximity":
                print(f"    [eval] scene {scene}: no grasp pose (OMG failed and no "
                      f"pin entry) — proximity cannot score it; counted as failure")

        row = _eval_episode(sim, runner, int(scene), params=params,
                            grasp_pose=grasp_pose)
        rows.append(row)
        if params.verbose:
            print(f"    eval [{i+1:3d}/{len(scenes)}] scene={scene:4d} "
                  f"success={row['success']} grasped={row['grasped']} "
                  f"close@{row['close_step']} pos_err={row['pos_err']:.3f} "
                  f"ee->ycb={row['dist']:.3f} "
                  f"box={row['box_chance']}/{row['box_taken']}"
                  f"@{row['box_steps']}st({row['box_frac_max']:.2f}) "
                  f"{row['reason']}")

    n = max(len(rows), 1)

    def _mean(key, where=lambda r: True):
        vals = [r[key] for r in rows if where(r) and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    # Outcome breakdown as FRACTIONS of the eval set, so the categories stack to
    # 1.0 and plot directly as an area chart of where episodes are being lost.
    # `reason` is exclusive by construction (one break per episode).
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    n_closed = sum(r["closed"] for r in rows)
    n_chance = sum(r["had_chance"] for r in rows)
    n_box = sum(r["box_chance"] for r in rows)

    return {
        "n": len(rows),
        "success_rate": sum(r["success"] for r in rows) / n,
        "grasp_rate": sum(r["grasped"] for r in rows) / n,
        "close_rate": n_closed / n,
        "near_rate": sum(r["near"] for r in rows) / n,
        # CONDITIONAL on having closed: "when it decides to grasp, is it right?"
        # Distinct from success_rate, which is over all episodes and so conflates
        # a bad grasp with never trying. NaN when nothing closed.
        "close_success_rate": (sum(r["success"] for r in rows) / n_closed
                               if n_closed else float("nan")),
        # Reached a graspable pose at some step...
        "chance_rate": n_chance / n,
        # ...and still did not secure the object. `missed_rate` is over ALL
        # episodes; `miss_given_chance` is the conditional — the fraction of
        # real opportunities the policy threw away.
        "missed_rate": sum(r["missed"] for r in rows) / n,
        "miss_given_chance": (sum(r["missed"] for r in rows) / n_chance
                              if n_chance else float("nan")),
        # ---- the GEOMETRIC opportunity, pin-independent (grasp_box.py) ----
        # Unlike chance_rate this does not require agreement with the pinned
        # pose, so an off-pose grasp — which is most of what the policy does —
        # counts as the opportunity it is.
        #
        # NaN, not 0, when the check is disabled: a 0 here would read as "the
        # policy never got a chance", which is a claim about the policy rather
        # than about what was measured. NaN writes a BLANK cell (see `_r` in
        # train_dagger_phase4) and plots as a gap.
        "box_chance_rate": (n_box / n) if params.box_check else float("nan"),
        # THE headline: given that the object really was between the open jaws,
        # how often did the policy command the close there. Conditional on the
        # chance, so it separates "never got a chance" from "got one and did not
        # take it" — the distinction success_rate cannot make. NaN when nothing
        # ever presented a chance.
        "box_taken_rate": (sum(r["box_taken"] for r in rows) / n_box
                           if (n_box and params.box_check) else float("nan")),
        # ...and, taken or not, how often that chance failed to become a grasp.
        "box_missed_rate": (sum(r["box_missed"] for r in rows) / n
                            if params.box_check else float("nan")),
        "miss_given_box": (sum(r["box_missed"] for r in rows) / n_box
                           if (n_box and params.box_check) else float("nan")),
        # How many policy-steps the window stayed open, over the episodes that
        # had one. A long declined window is a much stronger indictment of the
        # close decision than a one-step flicker, and the two are indistinguish-
        # able in box_taken_rate alone.
        "mean_box_steps": (_mean("box_steps", where=lambda r: r["box_chance"])
                           if params.box_check else float("nan")),
        # Best jaw occupancy seen per episode, averaged. Continuous, so
        # `box.min_frac` can be recalibrated from logs without re-running eval.
        "mean_box_frac": (_mean("box_frac_max") if params.box_check
                          else float("nan")),
        # Closest approach over the whole episode — defined even when the policy
        # never closes, so it still moves while every rate above reads 0.
        "eval_min_pos": _mean("min_pos"),
        "eval_min_rot": _mean("min_rot"),
        "mean_dist": _mean("dist"),
        # Pose error at the close, over the episodes that CLOSED — averaging a
        # non-closing episode in would be averaging a number that does not exist.
        "mean_pos_err": _mean("pos_err"),
        "mean_rot_err": _mean("rot_err"),
        "mean_close_step": _mean("close_step", where=lambda r: r["close_step"] >= 0),
        "reasons": reasons,
        "rows": rows,
    }
