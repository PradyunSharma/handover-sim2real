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
from handover_sim2real.regrasp import anchor as _rg_anchor  # noqa: E402
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
    # ---- what the policy is COMMANDED (SIM.command_deploy) ------------------
    # [k, 3] axis set, or None for the grasp-axis rule. The default is the STRING
    # "BINS", resolved below, because None is a MEANINGFUL value here — it
    # selects run 1's grasp-axis rule — and a None default would make "omitted"
    # and "grasp_axis" the same request. Every config written before this field
    # existed therefore scores exactly what it scored.
    #
    # Carried on the params rather than resolved per call site because the
    # parallel evaluator SHIPS these to its workers, and a command the manager
    # did not choose is the one bug that reports nothing unusual.
    command_axes: object = "BINS"
    # ---- what `d` is DERIVED FROM (`SIM.d_rule`) ---------------------------
    # A `directions.DirectionRule`, and it decides what "the direction the
    # gripper ACHIEVED" means — `dir_err` and `bin_realized` are measured with
    # it. Must match the rule the pin table was built under: scoring a
    # `grasp_offset` command against an `approach_axis` achievement compares two
    # different questions and reports the difference as policy error.
    d_rule: object = None
    # ---- when the anchor frame is built (`SIM.anchor_update`) ---------------
    # `latched` = once at step 0 and held, which is what every run through 15
    # did. `live` = rebuilt from the observed cloud every step, and the command
    # re-issued with it. The camera is eye-in-hand, so the object's OBSERVED
    # centroid moves as the gripper approaches even though the object does not;
    # `latched` conditions on a frame derived from the worst view of the episode
    # and a real rig has no way to reproduce it.
    anchor_update: str = "latched"
    # `SIM.anchor_hand_ref`: the MANO wrist JOINT, or the segmented hand CLOUD's
    # centroid, which is what the real rig measures from.
    anchor_hand_ref: str = "wrist"
    # ---- whether a short `grasp_offset` chord is DROPPED from the metrics ----
    # `d_min_offset` exists to stop a near-centroid fingertip from turning
    # centroid noise into a confident direction. Applying it to the MEASUREMENT
    # as well silently removes those episodes from `dir_err` and the confusion
    # matrix, so a policy that consistently stops on top of the centroid is
    # scored on the subset where it did not. False measures every episode that
    # arrived at all, at `min_offset` 0.
    dir_drop_short: bool = True
    verbose: bool = False

    def __post_init__(self):
        if isinstance(self.command_axes, str):
            self.command_axes = _rg_dirs.BINS.copy()
        if self.d_rule is None:
            self.d_rule = _rg_dirs.DirectionRule()
        if str(self.anchor_update) not in _rg_anchor.ANCHOR_UPDATES:
            raise ValueError(
                f"anchor_update must be one of {_rg_anchor.ANCHOR_UPDATES}, "
                f"got {self.anchor_update!r}")
        if str(self.anchor_hand_ref) not in _rg_anchor.ANCHOR_HAND_REFS:
            raise ValueError(
                f"anchor_hand_ref must be one of "
                f"{_rg_anchor.ANCHOR_HAND_REFS}, got {self.anchor_hand_ref!r}")
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
                  grasp_pose=None, target_pose=None, d_world=None,
                  bin_idx=None, anchor_R=None) -> dict:
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
    # THE FRAME THE EPISODE ENDED IN, which under `live` is not the one it
    # started in. `_dir_block` scores `bin_realized` against these, so a live run
    # is measured in the frame a deployment would have had at the close rather
    # than in the step-0 frame it has already left.
    live = str(params.anchor_update) == "live"
    anchor_state = _rg_anchor.AnchorState() if live else None
    anchor_final = None if anchor_R is None else np.asarray(anchor_R)
    centroid_final = None
    n_anchor_blind = 0

    for step in range(params.max_steps):
        pc = _point_cloud(obs, sim.point_listener, sim.panda_base_inv_tf)
        rs = _robot_state(obs, prev_act6d)
        # REBUILD THE FRAME AND RE-ISSUE THE COMMAND, before the policy acts on
        # this cloud, so the direction it is told matches the observation it is
        # told it about. On a step whose cloud has no object points the previous
        # frame is kept: conditioning on nothing would zero both channels, which
        # reads to the network as a valid command rather than a missing one.
        if live:
            aR, cw, ameta = _rg_anchor.anchor_from_cloud(
                pc, obs, env, sim.panda_base_inv_tf, sim.cfg, anchor_state,
                hand_ref=params.anchor_hand_ref)
            if aR is None:
                n_anchor_blind += 1
            else:
                anchor_final, centroid_final = aR, cw
                d_live = _rg_dirs.command_direction(
                    bin_idx, aR, grasp_pose=grasp_pose,
                    axes=params.command_axes)
                if d_live is not None:
                    d_world = d_live
                    runner.set_direction(d_world)
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
        # The frame and origin the episode ENDED in, and the command that was
        # standing at that moment. Under `latched` these are the step-0 values
        # handed in; under `live` they are what the last cloud with object points
        # produced. `eval_one` writes them onto the row so `_dir_block` scores in
        # the frame the policy was actually being commanded in.
        "anchor_R_final": anchor_final,
        "centroid_world_final": centroid_final,
        "d_world_final": d_world,
        "n_anchor_blind": int(n_anchor_blind),
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
                    pin_table=None, pairs=None) -> dict:
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
    num_grasps = eval_num_grasps(pin_table)
    jobs = eval_jobs(scenes, pin_table, num_grasps, pairs=pairs)
    rows = []
    for n, (scene, gi) in enumerate(jobs):
        row = eval_one(sim, runner, scene, gi, params=params, pin_table=pin_table)
        rows.append(row)
        if params.verbose:
            _print_eval_row(row, n, len(jobs), scene, gi, params)
    return aggregate_eval_rows(rows, params, num_grasps)


def eval_num_grasps(pin_table) -> int:
    """`max_grasps`, NOT `num_grasps`.

    The latter is a MIN over scenes and reads 1 on a Regrasp table (which mixes
    1- and 4-direction scenes), which would score only the first direction of
    every scene and quietly halve the eval.
    """
    return int(getattr(pin_table, "max_grasps", 0) or
               getattr(pin_table, "num_grasps", 1) or 1)


def off_table_slot(bin_idx: int) -> int:
    """A slot index meaning "command this BIN on a scene that has no demo for it".

    NEGATIVE, AND THAT IS DELIBERATE. The job protocol between the manager and
    the worker pool is a flat `(scene, slot)` pair — `parallel._fan_out` packs
    exactly two fields — so carrying a third would change the wire format of
    every job for a case that only needs one extra integer. Slots are
    non-negative by construction, so `-(bin + 1)` is unambiguous, survives the
    existing protocol untouched, and decodes in one place (`eval_one`).

    An off-table episode has NO pinned grasp: the scene demonstrates no grasp in
    that direction, which is the entire point of scoring it. `stable_grasp` does
    not need one — it scores release-and-hold — so these rows carry real success,
    grasp and direction numbers, with the pose diagnostics NaN.
    """
    return -(int(bin_idx) + 1)


def slot_bin(gi: int):
    """`(slot, bin)` for a job index: `(gi, None)` on-table, `(None, b)` off."""
    gi = int(gi)
    return (None, -gi - 1) if gi < 0 else (gi, None)


def stratified_pairs(pin_table, scenes, frac: float, *, verbose: bool = True,
                     full_bins: bool = False):
    """`[(scene, gi)]` — `frac` of EACH BIN's pairs, spread evenly, deterministic.

    THE DEFAULT EVAL IS A SCENE SAMPLE, AND THAT MAKES THE BINS WHATEVER THEY
    FALL OUT AS. `scene_pools` takes an np.linspace over usable scene ids and
    `eval_jobs` then expands each into ALL of its slots, so the per-bin counts
    are a by-product of which scenes were drawn. On run 11 that gives
    `+x 54, -x 33, +y 37, -y 25, +z 19, -z 26` — a 2.8x spread, with `+z` on 19
    episodes where one episode is 5.3 percentage points and the binomial standard
    error is +-11. Half the visible movement in the per-bin curves is that.

    This selects at the PAIR level instead: group every (scene, slot) pair by its
    bin, then take `ceil(frac * n_b)` from each. The per-bin proportions then
    match the population's by construction, the small bins are no longer starved
    relative to their own size, and `n_bin_*` becomes a stated fraction of a
    known denominator rather than an accident.

    EVENLY SPREAD, NOT RANDOMLY DRAWN — an np.linspace over each bin's sorted
    pair list, the same device `scene_pools` uses on scenes. No seed is consumed,
    so the same config always scores the same pairs and two runs are comparable
    episode by episode; a seeded draw would be reproducible only as long as
    nothing upstream of it changed its RNG consumption.

    NOTE THIS DOES NOT EQUALISE THE BINS. 20% of 19 is still 4. It makes the
    sample proportional and stable, which is a different (and honest) thing:
    equalising would require oversampling `+z` scenes that do not exist.
    """
    if pin_table is None:
        raise SystemExit(
            "EVAL.scene_select: per_bin_frac needs SIM.grasp_pin_table — the "
            "bin of every slot lives in the table, so without it there is "
            "nothing to stratify on.")
    frac = float(frac)
    if not 0.0 < frac <= 1.0:
        raise SystemExit(f"EVAL.bin_frac must be in (0, 1], got {frac}")

    by_bin: dict[int, list] = {}
    for scene in sorted(int(s) for s in scenes):
        for gi in range(pin_table.num_grasps_for(scene)):
            b = pin_table.bin_of(scene, gi)
            if b is None or int(b) < 0:
                continue
            by_bin.setdefault(int(b), []).append((scene, gi))

    out = []
    for b in sorted(by_bin):
        pool = by_bin[b]
        n = min(len(pool), max(1, int(np.ceil(frac * len(pool)))))
        idx = (np.linspace(0, len(pool) - 1, n).astype(int).tolist()
               if n > 1 else [len(pool) // 2])
        take = [pool[i] for i in sorted(set(idx))]
        out += take
        if verbose:
            print(f"[eval] bin {_rg_dirs.BIN_SHORT[b]}: {len(take)} of "
                  f"{len(pool)} pairs ({100.0 * len(take) / len(pool):.0f}%)")
    out.sort()
    scenes_used = sorted({s for s, _ in out})
    if verbose:
        print(f"[eval] per-bin {frac:.0%} sample: {len(out)} episodes over "
              f"{len(scenes_used)} scenes")

    # ---- FULL BIN COVERAGE --------------------------------------------------
    # `succ_bin_b` answers "when the policy is told `+z` ON A SCENE THAT
    # DEMONSTRATES `+z`, how often does it succeed". That is the right question
    # for "did it learn the training distribution", and the wrong one for "can I
    # deploy this" — at deployment the retry ladder commands a direction because
    # the previous one failed, not because a demonstration exists. Nothing in the
    # runs so far has ever measured a bin on a scene that did not demonstrate it.
    #
    # So every selected scene is additionally scored under EVERY bin it does not
    # already carry. Those rows get `in_table = 0`; the metric block reports
    # `succ_bin_b` over the in-table rows only (unchanged, comparable with runs
    # 1-15) and `succ_bin_all_b` over both. ONE eval pass, two populations.
    if full_bins:
        have = {(s, pin_table.bin_of(s, g)) for s, g in out}
        extra = [(s, off_table_slot(b))
                 for s in scenes_used for b in range(len(_rg_dirs.BINS))
                 if (s, b) not in have]
        out = sorted(out + extra)
        if verbose:
            print(f"[eval] full bin coverage: +{len(extra)} off-table episodes "
                  f"-> {len(out)} total ({len(scenes_used)} scenes x "
                  f"{len(_rg_dirs.BINS)} bins)")
    return out


def eval_jobs(scenes, pin_table, num_grasps=None, pairs=None):
    """[(scene, gi)] — every (scene, direction) pair scored, in order.

    `pairs`, when given, IS the job list: a pre-selected set of (scene, slot)
    pairs from `stratified_pairs`, passed through unchanged so the serial loop,
    the parallel pool and the resume path all enumerate the identical work.

    Split out so the serial loop and the parallel pool enumerate the SAME work
    in the SAME order. Results are reassembled by job index, which is what makes
    the parallel path bit-identical to the serial one rather than merely
    equivalent in distribution.
    """
    if pairs is not None:
        return [(int(s), int(g)) for s, g in pairs]
    num_grasps = eval_num_grasps(pin_table) if num_grasps is None else num_grasps
    out = []
    for scene in scenes:
        n_here = (pin_table.num_grasps_for(int(scene))
                  if pin_table is not None else num_grasps)
        out += [(int(scene), gi) for gi in range(max(int(n_here), 1))]
    return out


def eval_one(sim, runner, scene, gi, *, params: EvalParams, pin_table=None) -> dict:
    """ONE (scene, direction) episode -> its row. DETERMINISTIC: draws no RNG.

    The body of the old double loop, extracted so a worker process can call it
    on its own env. Everything beyond `sim`/`runner` is a plain argument or
    lives in `pin_table`, which every worker already builds.
    """
    pregrasp = str(params.target) == "pregrasp"
    scene = int(scene)
    # A NEGATIVE SLOT IS A BIN WITH NO DEMONSTRATION on this scene — see
    # `off_table_slot`. There is no pinned grasp to resolve, and asking for one
    # would fall through to OMG's free pick, which is a different grasp in a
    # different direction and would silently caption the row wrong.
    slot, off_bin = slot_bin(gi)
    off_table = slot is None
    if off_table and params.success_mode == "proximity":
        raise SystemExit(
            "EVAL.success_mode: proximity scores distance to a pinned grasp, and "
            "an off-table bin has none. Use stable_grasp with full bin coverage, "
            "or turn EVAL.full_bin_coverage off.")
    # proximity needs a grasp pose; stable_grasp only wants one if a pin table
    # makes it free (a diagnostic there, not the score).
    grasp_pose = None
    if not off_table and (params.success_mode == "proximity"
                          or pin_table is not None):
        grasp_pose = _resolve_grasp_pose(sim, scene, pin_table, slot)
        if grasp_pose is None and params.success_mode == "proximity":
            print(f"    [eval] scene {scene} g{gi}: no grasp pose (OMG failed "
                  f"and no pin entry) — proximity cannot score it; counted as "
                  f"failure")

    # The pose the POLICY steers to. Derived, never planned for, so eval makes
    # no OMG call it was not already making.
    target_pose = grasp_pose
    if pregrasp and grasp_pose is not None:
        target_pose = derived_standoff_pose(
            grasp_pose, params.standoff_dist, params.reach_tail)

    # THE COMMAND IS WHATEVER `SIM.command_deploy` SAYS, and it is the same
    # `command_direction` call the collector makes — a shared definition rather
    # than a duplicated one, because the two drifting apart is silent in every
    # rate: the policy is told one thing and scored on another, and `dir_err`
    # measures against the shifted target too.
    #
    # The default is the bin axis. Run 1 used `-R_grasp[:,2]`, a different
    # vector: the pinned grasp sits a median 18.4 deg off its bin's axis, so
    # eval scored the policy on a command no deployment could give it.
    meta = (pin_table.scene_meta.get(scene, {}) if pin_table is not None else {})
    anchor_R = meta.get("anchor_R")
    b = (off_bin if off_table
         else (pin_table.bin_of(scene, slot) if pin_table is not None else None))
    d_world = _rg_dirs.command_direction(
        b, None if anchor_R is None else np.asarray(anchor_R),
        grasp_pose=grasp_pose, axes=params.command_axes)

    row = _eval_episode(sim, runner, scene, params=params,
                        grasp_pose=grasp_pose, target_pose=target_pose,
                        d_world=d_world, bin_idx=b,
                        anchor_R=None if anchor_R is None else np.asarray(anchor_R))
    row["grasp_idx"] = int(gi)
    row["grasp_pose"] = grasp_pose
    row["bin_idx"] = -1 if b is None else int(b)
    # 1 = this (scene, bin) pair HAS a demonstration and is what runs 1-15
    # scored; 0 = the bin was commanded on a scene that never demonstrates it.
    # The metric block splits `succ_bin_*` on this.
    row["in_table"] = int(not off_table)
    # THE STEP-0 VALUES, kept under their historical names so a `latched` run's
    # rows are byte-identical to what they always were.
    row["d_world_0"] = d_world
    row["anchor_R_0"] = np.asarray(anchor_R) if anchor_R is not None else None
    row["centroid_world_0"] = meta.get("centroid_world")
    # WHAT THE METRICS ARE MEASURED AGAINST. Under `latched` the episode hands
    # back exactly what it was given, so these three collapse onto the step-0
    # values; under `live` they are the frame and command standing at the close.
    # EXPLICIT None CHECK, never `or`: these are ndarrays, and `arr or x`
    # evaluates the array's truth value, which raises.
    _dw = row.pop("d_world_final", None)
    row["d_world"] = d_world if _dw is None else _dw
    _aR = row.pop("anchor_R_final", None)
    row["anchor_R"] = (np.asarray(_aR) if _aR is not None
                       else (np.asarray(anchor_R) if anchor_R is not None else None))
    _cw = row.pop("centroid_world_final", None)
    row["centroid_world"] = (_cw if _cw is not None
                             else meta.get("centroid_world"))
    return row


def _print_eval_row(row, n, total, scene, gi, params) -> None:
    reach = (f"reach={row['reach_pos_err']:.3f}/{row['box_after']} "
             if str(params.target) == "pregrasp" else "")
    print(f"    eval [{n+1:3d}/{total}] scene={scene:4d} g{gi} "
          f"success={row['success']} grasped={row['grasped']} "
          f"close@{row['close_step']} pos_err={row['pos_err']:.3f} {reach}"
          f"ee->ycb={row['dist']:.3f} "
          f"box={row['box_chance']}/{row['box_taken']}"
          f"@{row['box_steps']}st({row['box_frac_max']:.2f}) {row['reason']}")


def aggregate_eval_rows(rows, params, num_grasps) -> dict:
    """rows -> the metric dict. Shared by the serial and parallel eval paths."""
    out = _rate_block(rows, params)
    out["rows"] = rows
    out.update(_regrasp_metrics(
        rows, num_grasps, getattr(params, "d_rule", None),
        drop_short=bool(getattr(params, "dir_drop_short", True))))
    # ---- THE SAME BLOCK AGAIN, ONE BIN AT A TIME ---------------------------
    # Pooled rates hide the thing the phase is about: `success_rate` averages
    # four physically different commands, and a policy that solves `+x` and
    # ignores `+z` reads identically to one mediocre at both. EVERY bin gets
    # keys, including the two empty on this dataset, so the CSV header does not
    # change with what a run happened to command. Empty bins yield NaN, which
    # `_r` writes blank and matplotlib renders as a gap.
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


def _regrasp_metrics(rows, num_grasps: int, d_rule=None,
                     drop_short: bool = True) -> dict:
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
        allb = [r for r in rows if int(r.get("bin_idx", -1)) == b]
        # TWO POPULATIONS, ONE EVAL PASS.
        #
        #   succ_bin_b      the scenes that DEMONSTRATE b. "Did it learn what it
        #                   was taught" — and the column runs 1-15 reported, so
        #                   this stays the comparable one.
        #   succ_bin_all_b  EVERY evaluated scene, commanded b whether or not a
        #                   demonstration for it exists. "What happens if I
        #                   deploy and the ladder asks for b" — which is the
        #                   situation the retry machine creates on purpose,
        #                   since it commands a direction BECAUSE the previous
        #                   one failed, not because a demo exists.
        #
        # Rows carry `in_table`, so both come out of the same rollouts and the
        # gap between them is the generalisation-to-unseen-direction number this
        # phase has never measured. Older rows have no `in_table`; they default
        # to 1, so a re-scored run 1-15 log gives succ_bin_all == succ_bin.
        rs = [r for r in allb if int(r.get("in_table", 1)) == 1]
        out[f"succ_bin_{b}"] = (sum(r["success"] for r in rs) / len(rs)
                                if rs else float("nan"))
        out[f"n_bin_{b}"] = len(rs)
        out[f"succ_bin_all_{b}"] = (sum(r["success"] for r in allb) / len(allb)
                                    if allb else float("nan"))
        out[f"n_bin_all_{b}"] = len(allb)
    # Kept for continuity with the Phase-5 column set, but slot-indexed and so
    # only meaningful within a scene.
    for g in range(num_grasps):
        rs = [r for r in rows if r.get("grasp_idx", 0) == g]
        m = max(len(rs), 1)
        out[f"succ_g{g}"] = sum(r["success"] for r in rs) / m

    # ---- retry@k -----------------------------------------------------------
    # KEYED BY BIN, WALKED IN LADDER ORDER (`directions.RETRY_LADDER`:
    # +x, +z, +y, -y, -z, -x, ordered by measured per-bin success). Runs 1-15
    # keyed by SLOT and walked ascending, which is ascending BIN INDEX and
    # therefore arbitrary with respect to which direction is worth trying first:
    # `-x`, the worst bin at 0.333, sat at index 1 and was tried second.
    #
    # IN-TABLE ROWS ONLY, so `retry_at_k` keeps the population it has always had
    # and stays comparable with runs 1-15 despite the reordering. The off-table
    # rows exist for `succ_bin_all_*`; folding them in here would change both the
    # order and the population in one step.
    by_scene: dict[int, dict[int, dict]] = {}
    for r in rows:
        if int(r.get("in_table", 1)) != 1:
            continue
        b = int(r.get("bin_idx", -1))
        if b >= 0:
            by_scene.setdefault(int(r["scene_idx"]), {})[b] = r
    n_scenes = max(len(by_scene), 1)
    for k in range(1, len(_D.RETRY_LADDER) + 1):
        rungs = _D.RETRY_LADDER[:k]
        hits = sum(1 for per in by_scene.values()
                   if any(per[b]["success"] for b in rungs if b in per))
        out[f"retry_at_{k}"] = hits / n_scenes
        # ---- THE DENOMINATOR, AND WHY IT HAS TO BE REPORTED -----------------
        # `retry_at_k` is over EVERY scene, including those with fewer than k
        # slots — for them the k-th attempt does not exist and the rate is just
        # their retry_at_(their slot count). `succ_bin_b`, by contrast, is over
        # the scenes that HAVE bin b. Different populations, so the two are not
        # comparable and `retry_at_4 >= max_b succ_bin_b` is NOT guaranteed:
        # measured on run 11 it 22, retry_at_4 = 0.75 over 100 scenes while
        # succ_bin_+x = 0.778 over the 54 scenes that have a `+x` demo, and the
        # 46 scenes without one are in the first denominator but not the second.
        #
        # Restricted to scenes that actually offer k attempts, the ladder IS
        # monotone and IS above any single bin measured on the same scenes,
        # because adding an attempt can only add a success. That is the number
        # to read as "what does retrying buy", and it is what these two columns
        # make available.
        deep = [per for per in by_scene.values()
                if sum(1 for b in rungs if b in per) == k]
        out[f"retry_n_{k}"] = len(deep)
        out[f"retry_at_{k}_deep"] = (
            sum(1 for per in deep
                if any(per[b]["success"] for b in rungs if b in per))
            / len(deep) if deep else float("nan"))
        # The ladder is FIXED now, so rung k is a definite direction rather than
        # a per-scene mixture. Kept as columns so the figures need no special
        # case, but `retry_bin_frac_k` is 1.0 by construction from here on.
        out[f"retry_bin_{k}"] = int(_D.RETRY_LADDER[k - 1])
        out[f"retry_bin_frac_{k}"] = 1.0

        # (The modal-bin computation that stood here is gone: it existed because
        # the ladder walked pin SLOTS, so rung k was a mixture of directions
        # across scenes and a legend naming one bin would have been wrong for the
        # minority. `RETRY_LADDER` fixes the order, so rung k is now one definite
        # direction and it is set above. It also ran AFTER the new assignment and
        # would have overwritten it.)

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
    dir_all, conf_all = _dir_block(rows, d_rule, drop_short=drop_short)
    out.update(dir_all)
    out["bin_confusion"] = conf_all.tolist()
    # ...and the same four numbers restricted to each commanded bin. `dir_track`
    # pooled over four directions is an average of four different questions: a
    # policy that tracks `+x` perfectly and ignores `+z` reads the same as one
    # that half-tracks both, and only the first is evidence the conditioning is
    # being read at all.
    for b in range(len(_D.BINS)):
        rs = [r for r in rows if int(r.get("bin_idx", -1)) == b]
        blk, _ = _dir_block(rs, d_rule, drop_short=drop_short)
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


def _dir_block(rows, d_rule=None, drop_short: bool = True):
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

    # THE ACHIEVED DIRECTION IS MEASURED UNDER THE SAME RULE AS THE COMMAND.
    # Under `grasp_offset` the command means "close on this part of the object",
    # so scoring it against `-R_ee[:,2]` ("which way is the wrist pointing")
    # compares two different questions and reports the difference as policy
    # error. Defaults to `approach_axis`, which is what runs 1-9 measured.
    d_rule = d_rule or _D.DirectionRule()
    # MEASURE EVERY EPISODE THAT ARRIVED, or only those whose chord cleared
    # `d_min_offset`. The threshold is a COLLECTION guard — it stops a fingertip
    # that ends on top of the centroid from minting a confident direction out of
    # centroid noise — and reusing it here quietly changes the denominator:
    # exactly the episodes that stopped short vanish from `dir_err` and from the
    # confusion matrix, so a policy whose failure mode IS stopping on the
    # centroid is scored on the subset where it did not do that. With
    # `drop_short=False` the same rule is applied at `min_offset` 0, so the
    # direction is noisy but present and the episode is counted.
    if not drop_short and d_rule.needs_centroid() and d_rule.min_offset:
        d_rule = _D.DirectionRule(rule=d_rule.rule, depth=d_rule.depth,
                                  min_offset=0.0)

    dir_errs, sector_errs = [], []
    n_short = 0
    confusion = np.zeros((len(_D.BINS), len(_D.BINS)), dtype=np.int64)
    for r in rows:
        d_cmd = r.get("d_world")
        if d_cmd is None:
            continue
        ee = r.get("ee_final")
        if ee is None:
            continue
        c = r.get("centroid_world")
        achieved = d_rule.of(np.asarray(ee), None if c is None else np.asarray(c))
        if achieved is None:
            n_short += 1
            continue
        dir_errs.append(float(_D.angle_between(d_cmd, achieved)))
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
        # HOW MANY EPISODES THE DIRECTION METRICS ABOVE DO NOT COVER. Non-zero
        # only under `grasp_offset` with `drop_short`, and it is the number that
        # says whether `dir_err` is over the whole eval set or over a filtered
        # subset of it.
        "dir_n": int(de.size),
        "dir_n_short": int(n_short),
    }, confusion
