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
from handover_sim2real.regrasp import directions as _rg_dirs  # noqa: E402
from handover_sim2real.regrasp.grasp_box import (  # noqa: E402
    BoxParams, grasp_opportunity,
)
# Pre-grasp mode (EvalParams.target). Both imported, never reimplemented, so the
# pose the evaluator scores against and the endgame it executes are the same ones
# the collector labelled towards.
from handover_sim2real.regrasp.collector import derived_standoff_pose  # noqa: E402
from handover_sim2real.regrasp.pregrasp import open_loop_reach  # noqa: E402

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
                  grasp_pose=None, target_pose=None, d_world=None) -> dict:
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
    # Condition on the DIRECTION derived from the grasp being scored, so a
    # policy that follows the command and one that ignores it are told apart by
    # the metrics rather than by which target each was shown.
    runner.set_direction(d_world)

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
    # Every scene is scored under EVERY direction it can supply, and each episode
    # carries the slot it was conditioned on — that is what makes the per-bin
    # rates, retry@k and dir_track all fall out of one pass.
    #
    # `max_grasps`, NOT `num_grasps`: the latter is a MIN over scenes and reads 1
    # on a Regrasp table (which mixes 1- and 2-direction scenes), which would
    # score only the first direction of every scene and quietly halve the eval.
    # The per-scene count drives the actual iteration.
    num_grasps = int(getattr(pin_table, "max_grasps", 0) or
                     getattr(pin_table, "num_grasps", 1) or 1)
    rows = []
    for i, scene in enumerate(scenes):
        n_here = (pin_table.num_grasps_for(int(scene))
                  if pin_table is not None else num_grasps)
        for gi in range(max(int(n_here), 1)):
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

            # THE COMMAND IS THE BIN AXIS, matching the collector and matching
            # what `retry.next_direction` issues at deployment. Run 1 used
            # `-R_grasp[:,2]` here, which is a different vector: the pinned grasp
            # sits a median 18.4 deg off its bin's axis, so eval was scoring the
            # policy on a command no deployment would ever give it — and
            # `dir_err`, which is the angle between the command and the achieved
            # axis, was measuring against that same shifted target and so hid the
            # discrepancy rather than showing it.
            meta = (pin_table.scene_meta.get(int(scene), {})
                    if pin_table is not None else {})
            anchor_R = meta.get("anchor_R")
            b = (pin_table.bin_of(int(scene), gi)
                 if pin_table is not None else None)
            if b is not None and anchor_R is not None and int(b) >= 0:
                d_world = _rg_dirs.to_world(_rg_dirs.BINS[int(b)],
                                            np.asarray(anchor_R))
            else:
                # No bin or no anchor (a Phase-5-shaped table): run 1's rule, so
                # an old pin table still scores rather than scoring nothing.
                d_world = (None if grasp_pose is None
                           else _rg_dirs.approach_direction(grasp_pose))
            row = _eval_episode(sim, runner, int(scene), params=params,
                                grasp_pose=grasp_pose, target_pose=target_pose,
                                d_world=d_world)
            row["grasp_idx"] = gi
            row["grasp_pose"] = grasp_pose
            row["d_world"] = d_world
            row["anchor_R"] = np.asarray(anchor_R) if anchor_R is not None else None
            row["centroid_world"] = meta.get("centroid_world")
            row["bin_idx"] = -1 if b is None else int(b)
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

    out = _rate_block(rows, params)
    out["rows"] = rows
    out.update(_regrasp_metrics(rows, num_grasps))
    # ---- THE SAME BLOCK AGAIN, ONE BIN AT A TIME ---------------------------
    # Pooled rates hide the thing the phase is about. `success_rate` is an
    # average over four physically different commands, and a policy that solves
    # `+x` and ignores `+z` reads identically to one that is mediocre at both.
    # Every panel of the training figure is therefore drawn per bin, which needs
    # the whole family per bin rather than just `succ_bin_*`.
    #
    # Suffix `_b{b}`, and EVERY bin gets keys — including the two that are empty
    # on this dataset. A fixed schema means the CSV header does not change with
    # what a run happened to command, so two runs stay diffable and a plotter can
    # ask for a column without first asking whether it exists. Empty bins yield
    # NaN, which `_r` writes as a blank cell and matplotlib renders as a gap.
    for b in range(len(_rg_dirs.BINS)):
        rs = [r for r in rows if int(r.get("bin_idx", -1)) == b]
        for k, v in _rate_block(rs, params).items():
            out[f"{k}_b{b}"] = v
    return out


def _rate_block(rows, params) -> dict:
    """The rate / error / outcome reduction over ONE set of eval episodes.

    Factored out so the identical arithmetic runs over the whole eval set and
    over each direction bin's slice of it — the alternative, a second
    hand-written per-bin reduction, is how a pooled number and its own breakdown
    drift apart without either looking wrong.

    An EMPTY slice returns the same keys with NaN (and `n: 0`), never zeros: a
    bin nobody commanded has an undefined success rate, and a 0 there is a claim
    about the policy rather than about what was measured.
    """
    n = max(len(rows), 1)

    def _mean(key, where=lambda r: True):
        vals = [r[key] for r in rows if where(r) and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _rate(pred, denom=None):
        if not rows:
            return float("nan")
        d = n if denom is None else denom
        return (sum(pred(r) for r in rows) / d) if d else float("nan")

    # Outcome breakdown as FRACTIONS of the eval set, so the categories stack to
    # 1.0 and plot directly as an area chart of where episodes are being lost.
    # `reason` is exclusive by construction (one break per episode).
    reasons, reasons_fail = {}, {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        if not r["success"]:
            reasons_fail[r["reason"]] = reasons_fail.get(r["reason"], 0) + 1

    n_closed = sum(r["closed"] for r in rows)
    n_chance = sum(r["had_chance"] for r in rows)
    n_box = sum(r["box_chance"] for r in rows)

    return {
        "n": len(rows),
        "success_rate": _rate(lambda r: r["success"]),
        "grasp_rate": _rate(lambda r: r["grasped"]),
        "close_rate": _rate(lambda r: r["closed"]),
        "near_rate": _rate(lambda r: r["near"]),
        # CONDITIONAL on having closed: "when it decides to grasp, is it right?"
        # Distinct from success_rate, which is over all episodes and so conflates
        # a bad grasp with never trying. NaN when nothing closed.
        "close_success_rate": (sum(r["success"] for r in rows) / n_closed
                               if n_closed else float("nan")),
        # Reached a graspable pose at some step...
        "chance_rate": _rate(lambda r: r["had_chance"]),
        # ...and still did not secure the object. `missed_rate` is over ALL
        # episodes; `miss_given_chance` is the conditional — the fraction of
        # real opportunities the policy threw away.
        "missed_rate": _rate(lambda r: r["missed"]),
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
        "box_chance_rate": (_rate(lambda r: r["box_chance"])
                            if params.box_check else float("nan")),
        # THE headline: given that the object really was between the open jaws,
        # how often did the policy command the close there. Conditional on the
        # chance, so it separates "never got a chance" from "got one and did not
        # take it" — the distinction success_rate cannot make. NaN when nothing
        # ever presented a chance.
        "box_taken_rate": (sum(r["box_taken"] for r in rows) / n_box
                           if (n_box and params.box_check) else float("nan")),
        # ...and, taken or not, how often that chance failed to become a grasp.
        "box_missed_rate": (_rate(lambda r: r["box_missed"])
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
        # Of the episodes that did NOT come away with the object, which way did
        # they fail. Denominator is the failures, not the eval set, so the
        # categories stack to 1.0 and a bin's failure PROFILE is readable
        # independently of how often that bin fails at all — the two questions
        # the pooled `f_*` fractions run together.
        "reasons": reasons,
        "reasons_fail": reasons_fail,
        "n_fail": int(sum(1 for r in rows if not r["success"])),
    }


def _regrasp_metrics(rows, num_grasps: int) -> dict:
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
    from handover_sim2real.regrasp import directions as _D

    if not rows:
        return {}
    out = {"num_grasps": int(num_grasps)}

    # ---- per-BIN rates, not per-slot ---------------------------------------
    # Slot k means "this scene's k-th chosen direction" and is not comparable
    # across scenes; bin k is a fixed physical direction and is. `succ_bin_*` is
    # therefore the column to read, and the one that shows whether -x and -z are
    # learnable at all (on this dataset they are not: 11 and 0 demonstrations).
    for b in range(len(_D.BINS)):
        rs = [r for r in rows if int(r.get("bin_idx", -1)) == b]
        out[f"succ_bin_{b}"] = (sum(r["success"] for r in rs) / len(rs)
                                if rs else float("nan"))
        out[f"n_bin_{b}"] = len(rs)
    # Kept for continuity with the Phase-5 column set, but slot-indexed and so
    # only meaningful within a scene.
    for g in range(num_grasps):
        rs = [r for r in rows if r.get("grasp_idx", 0) == g]
        m = max(len(rs), 1)
        out[f"succ_g{g}"] = sum(r["success"] for r in rs) / m

    # ---- retry@k -----------------------------------------------------------
    by_scene: dict[int, dict[int, dict]] = {}
    for r in rows:
        by_scene.setdefault(int(r["scene_idx"]), {})[int(r.get("grasp_idx", 0))] = r
    n_scenes = max(len(by_scene), 1)
    for k in range(1, num_grasps + 1):
        hits = sum(1 for per in by_scene.values()
                   if any(per[g]["success"] for g in range(k) if g in per))
        out[f"retry_at_{k}"] = hits / n_scenes

        # WHICH DIRECTION THE k-TH RUNG ACTUALLY WAS. The ladder walks a scene's
        # pin slots in table order, and `assign_direction_demos --mode per-bin`
        # emits them in ASCENDING BIN INDEX — so slot 0 is `+x` for a scene that
        # can reach `+x` and `+y` for one that cannot. A rung is therefore a
        # MIXTURE of directions across scenes, not one direction, and a legend
        # naming a single bin would be wrong for every scene in the minority.
        # Log the modal bin and its share so the figure can say "+x (79%)" and
        # remain true. `retry_bin_frac` well below 1.0 is the signal that the
        # ladder is not a fixed direction order and should not be read as one.
        slot = [r for r in rows if int(r.get("grasp_idx", 0)) == k - 1]
        if slot:
            counts: dict[int, int] = {}
            for r in slot:
                b = int(r.get("bin_idx", -1))
                counts[b] = counts.get(b, 0) + 1
            top = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
            out[f"retry_bin_{k}"] = top[0]
            out[f"retry_bin_frac_{k}"] = top[1] / len(slot)

    # ---- DID IT GO WHERE IT WAS TOLD ---------------------------------------
    # This replaces `near_rate`, which measured distance to a pinned POSE the
    # policy was never given and now reads low for a reason that says nothing
    # about the policy.
    #
    #   dir_err       angle between the commanded d and the approach axis the
    #                 gripper actually ended on. THE headline.
    #   sector_err    angle between d and the direction the gripper arrived FROM
    #                 (centroid -> EE). Genuinely different: dir_err is about
    #                 ORIENTATION, sector_err about WHICH SIDE. A gripper can be
    #                 correctly oriented on the wrong side and vice versa.
    #   bin_hit_rate  fraction with sector_err < 30 deg. The `near_rate` analogue.
    #                 30 rather than the 45-deg Voronoi half-angle, for margin.
    #   dir_track     1 - mean(dir_err)/90, so it reads like cond_track did:
    #                 1 = follows the command, 0 = ignores it.
    dir_all, conf_all = _dir_block(rows)
    out.update(dir_all)
    out["bin_confusion"] = conf_all.tolist()
    # ...and the same four numbers restricted to each commanded bin. `dir_track`
    # pooled over four directions is an average of four different questions: a
    # policy that tracks `+x` perfectly and ignores `+z` reads the same as one
    # that half-tracks both, and only the first is evidence the conditioning is
    # being read at all.
    for b in range(len(_D.BINS)):
        rs = [r for r in rows if int(r.get("bin_idx", -1)) == b]
        blk, _ = _dir_block(rs)
        for k, v in blk.items():
            out[f"{k}_b{b}"] = v

    # ---- does the behaviour CHANGE with the command -------------------------
    # cond_sep is the direction-space analogue of Phase-5's cond_track: the
    # spread of what the policy DID over the spread of what it was TOLD. It needs
    # only two conditions per scene, where cond_track wanted four.
    ratios = []
    for per in by_scene.values():
        cmds = [per[g].get("d_world") for g in sorted(per)]
        achs = [per[g].get("ee_final") for g in sorted(per)]
        ok = [(c, a) for c, a in zip(cmds, achs) if c is not None and a is not None]
        if len(ok) < 2:
            continue
        c0, c1 = ok[0][0], ok[1][0]
        a0 = _D.approach_direction(np.asarray(ok[0][1]))
        a1 = _D.approach_direction(np.asarray(ok[1][1]))
        told = float(_D.angle_between(c0, c1))
        did = float(_D.angle_between(a0, a1))
        if told > 1e-3:
            ratios.append(did / told)
    out["cond_sep"] = float(np.mean(ratios)) if ratios else float("nan")
    return out


def _dir_block(rows):
    """Did the gripper go where it was told, over one set of eval episodes.

    Returns `(metrics, confusion)`. Factored out for the same reason as
    `_rate_block`: the per-bin breakdown and the pooled headline are the same
    arithmetic, and writing it twice is how they come to disagree.

        dir_err       angle between the commanded d and the approach axis the
                      gripper actually ended on. THE headline.
        sector_err    angle between d and the direction the gripper arrived FROM
                      (centroid -> EE). Genuinely different: dir_err is about
                      ORIENTATION, sector_err about WHICH SIDE. A gripper can be
                      correctly oriented on the wrong side and vice versa.
        bin_hit_rate  fraction with sector_err < 30 deg. The `near_rate` analogue.
                      30 rather than the 45-deg Voronoi half-angle, for margin.
        dir_track     1 - mean(dir_err)/90, so 1 = follows the command, 0 =
                      ignores it.
        bin_diag_rate how often the REALISED bin is the commanded one. Collapsing
                      onto one column of the confusion matrix is the multi-modal
                      averaging failure — the policy going the same way whatever
                      it is told.
    """
    from handover_sim2real.regrasp import directions as _D

    dir_errs, sector_errs = [], []
    confusion = np.zeros((len(_D.BINS), len(_D.BINS)), dtype=np.int64)
    for r in rows:
        d_cmd = r.get("d_world")
        if d_cmd is None:
            continue
        ee = r.get("ee_final")
        if ee is None:
            continue
        achieved = _D.approach_direction(np.asarray(ee))
        dir_errs.append(float(_D.angle_between(d_cmd, achieved)))
        c = r.get("centroid_world")
        if c is not None:
            arrived = _D.normalize(np.asarray(ee)[:3, 3] - np.asarray(c))
            if float(np.linalg.norm(arrived)) > 0.5:
                sector_errs.append(float(_D.angle_between(d_cmd, arrived)))
        R = r.get("anchor_R")
        cb, rb = int(r.get("bin_idx", -1)), -1
        if R is not None:
            rb = _D.bin_of(_D.from_world(achieved, np.asarray(R)))
        if cb >= 0 and rb >= 0:
            confusion[cb, rb] += 1

    de = np.asarray(dir_errs, dtype=np.float64)
    se = np.asarray(sector_errs, dtype=np.float64)
    tot = int(confusion.sum())
    nan = float("nan")
    return {
        "dir_err": float(de.mean()) if de.size else nan,
        "dir_err_median": float(np.median(de)) if de.size else nan,
        "dir_track": float(1.0 - de.mean() / 90.0) if de.size else nan,
        "sector_err": float(se.mean()) if se.size else nan,
        "bin_hit_rate": float((se < _D.BIN_HIT_DEG).mean()) if se.size else nan,
        "bin_diag_rate": float(np.trace(confusion) / tot) if tot else nan,
    }, confusion
