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

WITH `target: pregrasp` (run 21) THE CLOSE IS A COMMIT. The policy is only asked
for the approach, so its channel-6 zero triggers the CVPR2023 endgame — a blind
6.4 cm push along the gripper's own +z (`pregrasp.open_loop_reach`) — and only
then the hold. `success_rate` is the identical `grasp_held_after_hold` call in
both modes, which is the point: run 21's headline number is directly comparable
to run 16's even though the policy is producing 6.4 cm less of the motion. The
GEOMETRIC scores do move, because the pose the policy steers to has moved:
`pos_err` / `min_pos` / `had_chance` are measured against the standoff, and the
new `reach_pos_err` reports where the push ended up relative to the grasp.
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
from handover_sim2real.dagger5.grasp_box import (  # noqa: E402
    BoxParams, grasp_opportunity,
)
# Pre-grasp mode (EvalParams.target). Both imported, never reimplemented, so the
# pose the evaluator scores against and the endgame it executes are the same ones
# the collector labelled towards.
from handover_sim2real.dagger5.collector import derived_standoff_pose  # noqa: E402
from handover_sim2real.dagger5.pregrasp import open_loop_reach  # noqa: E402

SUCCESS_MODES = ("stable_grasp", "proximity")
TARGETS = ("grasp", "pregrasp")


@dataclass
class EvalParams:
    max_steps: int = 50
    success_mode: str = "stable_grasp"   # see the module docstring
    hold_steps: int = 3                  # stable_grasp: policy-steps held shut
    close_pos_thresh: float = 0.02       # proximity: metres
    close_rot_thresh: float = 0.34       # proximity: radians (~19.5 deg)
    # ---- where the episode ends (run 21); mirrors CollectParams.target ----
    # "grasp"    the policy's close is a close: hold it and score.
    # "pregrasp" the policy's close is a COMMIT: run the blind feed-forward reach
    #            (`pregrasp.open_loop_reach`) first, then hold and score. Success
    #            is the same `grasp_held_after_hold` in both modes, so the
    #            headline number stays directly comparable to run 16's — the
    #            policy is simply being asked for less of the motion.
    # The geometric scores move with it: `pos_err`, `min_pos` and `had_chance`
    # are measured against the STANDOFF, since that is the pose the policy is
    # steering to. `reach_pos_err` then reports where the blind push ended up
    # relative to the grasp, which is what separates "the policy stopped in the
    # wrong place" from "forward_dist is mis-set".
    target: str = "grasp"
    forward_dist: float = 0.064          # metres along the gripper's local +z
    forward_steps: int = 4               # sub-steps to spread the push over
    standoff_dist: float = 0.08          # OMG's ramp extent, for the derivation
    reach_tail: int = 5                  # OMG's cfg.reach_tail_length
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
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, "
                             f"got {self.target!r}")
        if self.box is None:
            self.box = BoxParams()


def _ee_mat_from_state(rs) -> np.ndarray:
    """(4, 4) world EE pose from the 32-D robot state: rs[18:21] = xyz,
    rs[21:25] = quaternion wxyz. Note the state is in SIM WORLD frame while the
    point cloud is EE-relative — a documented asymmetry of this layout — and the
    pin table's poses are in that same world frame, so the two are directly
    comparable without any extra transform."""
    from transforms3d.quaternions import quat2mat
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(rs[18:21], dtype=np.float64)
    T[:3, :3] = quat2mat(np.asarray(rs[21:25], dtype=np.float64))
    return T


def _resolve_grasp_pose(sim, scene_idx: int, pin_table, grasp_idx: int = 0):
    """World 4x4 of the grasp this (scene, slot)'s close is scored against.

    Prefers the pin table (it stores the committed grasp's world pose, so this
    costs nothing and is by construction the same pose the collector labelled
    towards). Falls back to one step-0 OMG plan, as Phase 3 does. None if
    neither is available — the caller then reports NaN pose errors. The fallback
    can only ever produce slot 0's answer, so with `grasp_idx > 0` and no table
    there is nothing to score against and this returns None rather than silently
    scoring the wrong grasp.
    """
    if pin_table is not None:
        pose = pin_table.pose(int(scene_idx), int(grasp_idx))
        if pose is not None:
            return pose

    if int(grasp_idx) != 0:
        return None
    plan, _ = sim.env.run_omg_planner(
        int(sim.cfg.RL_MAX_STEP), int(scene_idx), reset_scene=True)
    if plan is None:
        return None
    return sim.env.get_omg_goal_grasp_pose()


def _eval_episode(sim, runner, scene_idx, *, params: EvalParams,
                  grasp_pose=None, target_pose=None) -> dict:
    """`grasp_pose` is what the gripper must end up on; `target_pose` is what the
    POLICY is steering to. They are the same pose in grasp mode and 6.4 cm apart
    in pre-grasp mode, where the difference is covered by the blind push."""
    env = sim.env
    pregrasp = str(params.target) == "pregrasp"
    if target_pose is None:
        target_pose = grasp_pose
    obs = env.reset(idx=scene_idx)
    sim.point_listener.reset()
    runner.reset()
    # Phase 5: condition the learner on the grasp it is being SCORED against, so
    # a policy that ignores the conditioning and one that follows it are told
    # apart by the metrics rather than by which target each was shown.
    #
    # `grasp_pose`, NOT `target_pose`: the collector conditions on
    # env.get_omg_goal_grasp_pose(), which is the final grasp in both grasp and
    # pre-grasp mode — the standoff is where the policy stops, not what it was
    # told to aim at. Evaluation makes no OMG calls under stable_grasp, so this
    # comes from the pin table for free.
    runner.set_goal(grasp_pose)

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
    # Pre-grasp mode only. `reach_*` is where the BLIND push ended up relative to
    # the grasp — the one thing that cannot be inferred from the policy's own
    # pose — and `box_after` asks the same question the box test asks during a
    # grasp-mode approach, but at the only pose in this mode where the answer can
    # be yes: after the push, with the fingers still open.
    reach_pos_err = reach_rot_err = float("nan")
    box_after = False
    box_after_frac = float("nan")
    close_step = -1
    success = False
    grasped = False
    reason = ""
    # Phase 5 `cond_track`: the EE pose the episode ended at, as a 4x4 world
    # matrix. Rolling the same scene under all four grasps and asking how far
    # apart these four end up is the one diagnostic that separates "the policy
    # tracks the commanded grasp" from "the policy ignores it and regresses the
    # mean of four demonstrations" — and the second is the failure mode that
    # makes the whole regrasping premise inert.
    ee_final = None

    for step in range(params.max_steps):
        pc = _point_cloud(obs, sim.point_listener, sim.panda_base_inv_tf)
        rs = _robot_state(obs, prev_act6d)
        # World-frame EE pose, straight out of the state the policy just saw
        # (rs[18:21] xyz, rs[21:25] wxyz). Overwritten every step, so whatever
        # the episode ends on is what cond_track measures.
        ee_final = _ee_mat_from_state(rs)
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

        if target_pose is not None:
            pe, re_ = ee_grasp_pose_error(obs, target_pose)
            min_pos, min_rot = min(min_pos, pe), min(min_rot, re_)
            if pe <= params.close_pos_thresh and re_ <= params.close_rot_thresh:
                had_chance = True

        if action[6] < 0.5:
            # ---- the policy committed: score it (Phase-3 criterion) ----
            # In pre-grasp mode "committed" means the blind reach, not the close,
            # so the endgame runs between the geometry read and the hold.
            close_step = step
            box_taken = bool(opportunity)

            # Geometry FIRST, from the pose the commit was made at — the reach and
            # the hold both move the arm, so this has to be read before either.
            if target_pose is not None:
                pos_err, rot_err = ee_grasp_pose_error(obs, target_pose)
            near = bool(pos_err <= params.close_pos_thresh
                        and rot_err <= params.close_rot_thresh)

            if pregrasp:
                obs, pushed_done, st = open_loop_reach(
                    env, obs, sim.steps_action_repeat,
                    dist=params.forward_dist, num_steps=params.forward_steps)
                if grasp_pose is not None:
                    reach_pos_err, reach_rot_err = ee_grasp_pose_error(
                        obs, grasp_pose)
                if params.box_check and not pushed_done:
                    box_after, box_after_frac = grasp_opportunity(env, params.box)
                if pushed_done:
                    # The push ended the episode: the swing into the grasp knocked
                    # the object out of the hand, or tripped human contact. That is
                    # the benchmark's failure, not a grasp that missed — recording
                    # it as GRASP_MISS would hide an over-long forward_dist behind
                    # the policy's success rate.
                    status = st
                    reason = _status_name(st)
                    break

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
        "ee_final": ee_final,
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
        # ---- pre-grasp mode: what the BLIND push achieved (NaN/0 otherwise) ----
        "reach_pos_err": float(reach_pos_err),
        "reach_rot_err": float(reach_rot_err),
        "box_after": int(box_after),
        "box_after_frac": float(box_after_frac),
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
    pregrasp = str(params.target) == "pregrasp"
    # Phase 5: every scene is scored under EVERY pinned grasp, so the eval set is
    # `num_grasps` x larger than `scenes` and each episode carries the slot it was
    # conditioned on. That is what makes the per-slot rates, retry@k and
    # cond_track all fall out of one pass.
    num_grasps = int(getattr(pin_table, "num_grasps", 1) or 1)
    rows = []
    for i, scene in enumerate(scenes):
        for gi in range(num_grasps):
            # proximity needs a grasp pose; stable_grasp only wants one if a pin
            # table makes it free (it is a diagnostic there, not the score).
            grasp_pose = None
            if params.success_mode == "proximity" or pin_table is not None:
                grasp_pose = _resolve_grasp_pose(sim, int(scene), pin_table, gi)
                if grasp_pose is None and params.success_mode == "proximity":
                    print(f"    [eval] scene {scene} g{gi}: no grasp pose (OMG "
                          f"failed and no pin entry) — proximity cannot score it; "
                          f"counted as failure")

            # The pose the POLICY is steering to. Derived, never planned for: it
            # is a fixed function of the grasp (`derived_standoff_pose`, 5.6e-7 m
            # from the planner's own traj[-reach_tail]), so eval still makes no
            # OMG call it was not already making.
            target_pose = grasp_pose
            if pregrasp and grasp_pose is not None:
                target_pose = derived_standoff_pose(
                    grasp_pose, params.standoff_dist, params.reach_tail)

            row = _eval_episode(sim, runner, int(scene), params=params,
                                grasp_pose=grasp_pose, target_pose=target_pose)
            row["grasp_idx"] = gi
            row["grasp_pose"] = grasp_pose
            rows.append(row)
            if params.verbose:
                reach = (f"reach={row['reach_pos_err']:.3f}/{row['box_after']} "
                         if pregrasp else "")
                print(f"    eval [{i+1:3d}/{len(scenes)}] scene={scene:4d} "
                      f"g{gi} "
                      f"success={row['success']} grasped={row['grasped']} "
                      f"close@{row['close_step']} pos_err={row['pos_err']:.3f} "
                      f"{reach}"
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
        # ---- pre-grasp mode: did the BLIND push finish the job ----
        # Averaged over the episodes that COMMITTED, since an episode with no
        # commit has no push to measure. `mean_reach_pos_err` against
        # `mean_pos_err` splits the two failures this mode can have: a large
        # pos_err is a policy that stopped in the wrong place, a small pos_err
        # with a large reach_pos_err is a `forward_dist` that needs re-tuning.
        # `box_after_rate` is the conversion — of the commits, how many put the
        # object between the open jaws.
        "mean_reach_pos_err": _mean("reach_pos_err"),
        "mean_reach_rot_err": _mean("reach_rot_err"),
        "box_after_rate": ((sum(r["box_after"] for r in rows) / n_closed)
                           if (n_closed and params.box_check
                               and str(params.target) == "pregrasp")
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
        **_phase5_metrics(rows, num_grasps),
    }


def _phase5_metrics(rows, num_grasps: int) -> dict:
    """The three things Phase 5 exists to measure, all from one eval pass.

    **Per-slot rates** (`succ_g0..`, `near_g0..`). Slot 0 is OMG's own pick, so
    `succ_g0` is the column directly comparable with a Phase-4 run; the spread
    across slots is how much harder the deliberately-separated grasps are.

    **retry@k** — success@1..success@N over the slots in FPS order, i.e. "try
    grasp 0, and if the handover fails try grasp 1, ...". This is the regrasping
    headline, and it is free: no extra rollouts, just a different reduction over
    the episodes already run. It assumes each retry restarts from home, which is
    true of this evaluation and not of a real deployment, where attempt 2 begins
    wherever attempt 1 stopped. Read it as the ceiling.

    **cond_track** — the diagnostic that decides whether any of the above means
    anything. For each scene, how far apart the N final EE poses are, divided by
    how far apart the N commanded grasps are, both under the flip-invariant
    control-point metric. 1.0 means the policy separates the conditions as much
    as the targets are separated; 0.0 means it does the same thing whatever it is
    told, which is the multi-modal averaging failure and would make regrasping
    inert no matter how good `success_rate` looked.
    """
    from handover_sim2real.dagger5.grasp_select import pairwise_mean_distance

    if not rows:
        return {}
    out = {"num_grasps": int(num_grasps)}

    by_slot = {g: [r for r in rows if r.get("grasp_idx", 0) == g]
               for g in range(num_grasps)}
    for g, rs in by_slot.items():
        m = max(len(rs), 1)
        out[f"succ_g{g}"] = sum(r["success"] for r in rs) / m
        out[f"near_g{g}"] = sum(r["near"] for r in rs) / m

    # retry@k, and the per-scene tables the other two need.
    by_scene: dict[int, dict[int, dict]] = {}
    for r in rows:
        by_scene.setdefault(int(r["scene_idx"]), {})[int(r.get("grasp_idx", 0))] = r
    n_scenes = max(len(by_scene), 1)
    for k in range(1, num_grasps + 1):
        hits = sum(1 for per in by_scene.values()
                   if any(per[g]["success"] for g in range(k) if g in per))
        out[f"retry_at_{k}"] = hits / n_scenes

    if num_grasps < 2:
        out["cond_track"] = float("nan")
        out["cond_ee_spread"] = float("nan")
        out["cond_goal_spread"] = float("nan")
        return out

    ratios, ee_spreads, goal_spreads = [], [], []
    for per in by_scene.values():
        ees = [per[g]["ee_final"] for g in sorted(per)
               if per[g].get("ee_final") is not None]
        goals = [per[g]["grasp_pose"] for g in sorted(per)
                 if per[g].get("grasp_pose") is not None]
        if len(ees) < 2 or len(goals) < 2:
            continue
        d_ee = pairwise_mean_distance(np.stack(ees))
        d_goal = pairwise_mean_distance(np.stack(goals))
        ee_spreads.append(d_ee)
        goal_spreads.append(d_goal)
        if d_goal > 1e-6:
            ratios.append(d_ee / d_goal)
    out["cond_track"] = float(np.mean(ratios)) if ratios else float("nan")
    out["cond_ee_spread"] = float(np.mean(ee_spreads)) if ee_spreads else float("nan")
    out["cond_goal_spread"] = (float(np.mean(goal_spreads)) if goal_spreads
                               else float("nan"))
    return out
