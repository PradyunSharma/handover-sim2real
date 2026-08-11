"""
Phase-4 DAgger collection.

One episode = drive the sim with the CURRENT policy's own actions (beta-mixed
with the expert), and at every visited state query OMG for the action the expert
would take there. The (state, expert action) pairs are written in the *exact*
Phase-1/2 BC HDF5 schema, so the aggregate D can be fed straight to
`BCSequenceDataset` with no conversion:

    point_clouds   float32 [T, 1024, 5]   xyz + ycb_flag + hand_flag (EE frame)
    robot_states   float32 [T, 32]        joint_pos(9)+joint_vel(9)+ee(7)+grip(1)+prev_act(6)
    expert_actions float32 [T, 7]         dpos(3)+deuler(3)+gripper_cmd(1)

What differs from the Phase-1/2 collectors (`collect_dagger_*_dataset.py`):

  * **Replan every step, label = plan[0].** OMG is re-run from the policy's
    CURRENT (drifted) joint configuration at every step and the label is the
    first waypoint of that fresh plan — a genuine pi*(s) query at every visited
    state, which is what Algorithm 3.1 asks for. (Phase 3, by contrast, commits
    to a step-0 plan and reads plan[t] out of it.)

  * **No standoff-plane cutoff, and a COMMITTED REACH instead.** Phase 1/2
    stopped recording once the EE crossed the pre-grasp standoff, because a
    replan from beyond it yields backward "retreat to the standoff" labels.
    Phase 4 records all the way in. That needs a fix, because plan[0] labelling
    structurally cannot express the final reach:

      - OMG puts the standoff RAMP in its goal set, so the returned trajectory
        is [free portion -> standoff] + [last `reach_tail` waypoints -> grasp].
      - Once the EE is AT the standoff the free portion is stationary, so
        plan[0] decays to zero while the reach sits in the LAST waypoints.
        Measured with beta=1 (pure expert, no policy involved): the EE parks
        6.6 cm short of the grasp forever and every further label is ~0. This is
        the same Zeno stall that produced the rl_run7 hover in Phase 3.
      - Taking "the first waypoint that moves" instead does not fix it either:
        past the standoff OMG's nearest goal is behind the EE, so it retreats,
        and the EE limit-cycles between ~0.036 and ~0.052 m.

    So on arrival at the standoff (||ee - standoff|| <= `reach_commit_dist`) the
    current plan's last `reach_tail` waypoints are FROZEN and followed BY INDEX
    with no further replanning — no retreat, no oscillation. The delta is still
    recomputed from the CURRENT EE each step, so labels stay corrective under
    drift; only the target stops moving. Past the end the final waypoint (the
    grasp) is held, so the label shrinks monotonically until the close threshold
    fires. Measured after the fix: 4.3-4.8 mm from the grasp, CLOSE label
    emitted, on every scene tried.

  * **Distance-triggered gripper-close label.** At each step the EE pose is
    compared with the grasp pose the current plan aims at (OMG traj[-1]). Once
    BOTH the position and the orientation error are inside
    (close_pos_thresh, close_rot_thresh), the label becomes the closure command
    `[0,0,0, 0,0,0, 0.0]` instead of another approach delta. Phase 1/2 only did
    this as a legacy opt-in; here it is the endgame supervision.

  * **Plans to the standoff AND beyond.** The OMG horizon is distance
    proportional — `free = round(||ee - standoff|| / ee_step)` waypoints for the
    free portion plus `reach_tail` for the standoff->grasp reach OMG folds into
    the trajectory tail — so the recorded first-step delta stays at the
    demonstrations' per-step scale at every distance, and the plan always
    contains the final reach.

  * **DART is nearly free here** (`dart_ratio > 0`; Laskey et al. 2017, GA-DDPG
    `env.random_perturb`). On a fraction of approach steps the executed action is
    REPLACED by a random task-space jump (+-dart_pos_mag m, +-dart_rot_mag rad,
    gripper open), landing the EE off-plan so the FOLLOWING steps are labelled
    with the expert's recovery from there. It manufactures the off-plan near-grasp
    coverage the learner stops producing on its own once beta anneals and it has
    learned to track the plan — the states where it arrives at the standoff a few
    cm off LATERALLY and has never seen a correction.

    Two things that cost real machinery upstream cost nothing in Phase 4:

      - **No perturb flag, no masking.** GA-DDPG (`core/ddpg.py`) and Phase 3
        (`rl/td3bc_trainer.py`) must tag the perturbed row and drop it from the
        critic's Bellman fit, because there the stored action IS the artificial
        jump. Phase 4 stores only (state, pi*(state)) — the executed action is
        never written — so replacing it costs nothing and there is nothing to
        mask. The pair recorded on a jolt step is an ordinary one.
      - **No replan/splice.** GA-DDPG re-plans and splices the tail onto the
        committed prefix after each jolt. This collector already re-queries OMG
        from the current drifted configuration EVERY step, so the next iteration
        of the loop labels the perturbed state correctly by construction. (It
        also avoids GA-DDPG's ordering bug, where the recorded observation is
        pre-jolt while `expert_action` comes from the post-jolt replan.)

    The trigger is DISTANCE to the standoff, not a step index as in GA-DDPG
    (`DART_MIN_STEP`/`DART_MAX_STEP`) and Phase 3 (`[15, 22)`). Those work because
    the expert there plays a plan BY INDEX, so step count tracks progress. Here
    every step replans and `max_steps` is 50: a far scene needs ~36 steps to
    reach the standoff and a near one ~10, so a fixed step window would jolt one
    episode mid-flight and another before it had moved. The band is effectively
    (`reach_commit_dist`, `dart_max_dist`] — floored because the reach commits
    below it, and jolting during the committed reach would kick the EE back out
    of `close_pos_thresh` and destroy the close label, which is the scarcest
    thing an iteration produces.

    The jolt REPLACES a step rather than being an extra out-of-band one, so it
    costs one of the `max_steps` budget. If `c_max_steps` rises when DART is on,
    that budget is the reason: the alternative is to execute the chosen action
    first and jolt at the END of the loop body (after the `env.step` block), so
    the next iteration reads the perturbed state and the step budget is untouched.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# Reuse the offline collectors' state builders / IK / geometry so Phase-4 states
# are byte-identical to what Phase-1/2 trained on (same precedent as
# handover_sim2real/rl/rollout_worker.py).
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from collect_bc_dataset import (  # noqa: E402
    _point_cloud,
    _robot_state,
    ee_grasp_pose_error,
    dynamic_replan_horizon,
    ACTION_DIM,
    NUM_PTS,
    PC_CHANNELS,
    ROBOT_STATE_DIM,
)
from rollout_bc_policy import action_to_target_joint  # noqa: E402

# Outcome scoring, imported from the Phase-3 worker rather than reimplemented —
# the evaluator imports the same two, so a collection outcome and an eval outcome
# are decided by identical code (see CollectParams.outcome_check).
from handover_sim2real.rl.rollout_worker import (  # noqa: E402
    _status_name,
    grasp_held_after_hold,
)
# 'sxyz' axes — the same convention the actions are built with (train_env.py uses
# transforms3d's mat2euler default), so this inverts the label exactly.
from transforms3d.euler import euler2mat  # noqa: E402


CLOSE_LABEL = np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)


@dataclass
class CollectParams:
    """The DAGGER block of the Phase-4 config, resolved."""

    # 50, not 30: the EE moves ~0.02 m/step (IK tracking under
    # steps_action_repeat only realises ~60% of a commanded ~0.035 m delta), and
    # a far-start scene is ~0.78 m from the grasp. Measured: ~36 steps to reach
    # the standoff on scene 60. The demos need only 20 steps because they follow
    # the plan BY INDEX; replanning and taking step 0 undershoots.
    max_steps: int = 50
    # gripper-close label trigger (position in metres, orientation in radians;
    # 0.34 rad ~ 19.5 deg)
    close_pos_thresh: float = 0.02
    close_rot_thresh: float = 0.34
    # end the episode as soon as the close label is emitted (mirrors the single
    # terminal close transition in the expert demonstrations). False keeps
    # rolling and emits a close label at every in-threshold step.
    stop_on_close_label: bool = True
    # the policy commanded a close: execute it and end the episode (faithful --
    # in deployment the episode really is over). False overrides the policy's
    # gripper bit and keeps approaching, which yields more labels per episode
    # but no longer follows the learner's own policy.
    stop_on_policy_close: bool = True
    # distance-proportional OMG horizon (see module docstring)
    ee_step: float = 0.04
    # MUST equal OMG's cfg.reach_tail_length (config.py:92, default 5) -- it is a
    # mirror of the planner's own constant, not a free choice. OMG builds the
    # standoff ramp as exactly that many poses, so this is how many trailing
    # waypoints of the returned plan are the reach. Get it wrong and traj[-N] is
    # not the standoff, the commit freezes the wrong slice, and the horizon
    # request comes up short. Its SPACING is standoff_dist/reach_tail_length =
    # 1.6 cm, deliberately finer than ee_step (4 cm): coarse while flying to the
    # standoff, fine while reaching in.
    reach_tail: int = 5
    min_free: int = 3
    max_horizon: int = 40
    first_horizon: int = 30  # horizon of the very first plan (standoff unknown)
    # ----- planner-free pre-grasp (run 18) -----
    # false: the standoff is read from the plan (`env.get_omg_standoff_pose()` =
    # FK of traj[-reach_tail]) — the behaviour of runs 1-17.
    # true:  it is DERIVED from the pinned grasp via `derived_standoff_pose`.
    #
    # The two poses are numerically identical (5.6e-7 m; see that function), so
    # this does not move the reach commit. What changes is when the pose EXISTS:
    #   * at reset, seeded from the pin table, so the step-0 plan gets the same
    #     distance-proportional horizon every later step gets instead of the flat
    #     `first_horizon` fallback. This is the ONLY behavioural difference that
    #     fires in practice, and it touches 1 step per episode.
    #   * on a step whose replan failed, so `d_standoff` — and therefore the DART
    #     trigger band — is no longer silently unavailable. Latent robustness
    #     only: run 16 logged omg_fail = 0 over 89,645 steps, so it never fires.
    # Expect collection to be near-identical to a `false` run. If it is not,
    # something else changed.
    #
    # The pin table's stored pose is used ONLY for the reset seed, and only for
    # horizon sizing: `apply()` matches it to the nearest goal-set entry within
    # `grasp_pin_match_tol`, so it can sit up to that far from the grasp actually
    # planned to (measured 0.015 m on train scene 0, 0.0 on scenes 1-4). The
    # first successful plan replaces it with the live grasp and it is never
    # consulted again. Sizing a horizon as round(d / ee_step) with d ~ 0.4 m
    # cannot notice 1.5 cm; the commit test never sees this value.
    derive_standoff: bool = False
    # ----- per-episode outcome scoring (run 18) -----
    # false: `reason` is the loop's TERMINAL CAUSE only (CLOSE_LABEL / POLICY_CLOSE
    # / ENV_DONE / MAX_STEPS) — what runs 1-17 logged. That says why the loop
    # stopped, not whether the handover worked.
    # true:  additionally score each episode with the SAME criterion the evaluator
    # uses, so a collection success rate is comparable to `success_rate`.
    #
    # Why this needs extra work rather than a spare field: on a CLOSE_LABEL the
    # loop `break`s BEFORE executing anything, so during collection the gripper
    # NEVER actually closes and the benchmark's success is simply not observable.
    # With this on, the close is executed and held for `hold_steps` AFTER the
    # labels are frozen, then `grasp_held_after_hold` decides it. The recorded
    # (state, action) pairs are already fixed at that point, so D_i is byte-
    # identical either way — this only observes.
    #
    # Cost: hold_steps(3) x steps_action_repeat(150) = 450 sim steps on the ~70%
    # of episodes that close, against ~540,000 sim steps an iteration already
    # spends. About +6%.
    #
    # It also captures `info["status"]` from env.step, which the collector used to
    # discard — that is what splits the ENV_DONE bucket (15.8% of run 16's
    # episodes) into HUMAN_CONTACT vs DROP vs BENCH_TIMEOUT.
    outcome_check: bool = False
    hold_steps: int = 3          # must match EVAL.hold_steps to stay comparable
    # OMG's cfg.standoff_dist — a ramp EXTENT, not a distance. The effective
    # standoff is standoff_dist * (1 - 1/reach_tail) = 0.064 m at the defaults.
    # Mirrored from the SIM block so the derivation cannot drift out of sync with
    # the planner that produced the waypoints.
    standoff_dist: float = 0.08
    # ----- committed reach (see module docstring section on the endgame) -----
    # Once the EE is within `reach_commit_dist` of the standoff, freeze the
    # current plan's last `reach_tail` waypoints and follow them BY INDEX instead
    # of replanning. `reach_skip_eps` drops leading waypoints of that tail that
    # are effectively where the EE already is (the tail's first entry IS the
    # standoff, so it is a no-op at commit time).
    #
    # 0.05, not 0.02: `min_free` floors the free portion at 3 steps, so
    # ||plan[0]|| ~ d_standoff/3 and the approach labels decay as the standoff is
    # approached (measured: 0.040 at d=0.12, 0.011 at d=0.05, 0.004 at d=0.018).
    # Handing off at 0.05 keeps every recorded approach label at roughly the
    # demonstrations' per-step scale. Committing early is geometrically safe: the
    # reach tail is `pose_grasp_global @ pose_standoff`, a fixed function of the
    # chosen grasp, identical whatever distance it was planned from (verified
    # across horizons 5/8/10/20). `reach_skip_eps` then simply skips fewer
    # leading waypoints, so the first frozen target is the standoff itself.
    reach_commit_dist: float = 0.05
    # ----- expert takeover for the committed reach -----
    # Once the reach is committed, EXECUTE the expert's waypoints regardless of
    # beta (i.e. beta = 1 for the endgame only).
    #
    # Why it is tempting: committing the reach only fixes the LABEL TARGET. What
    # moves the arm is still the beta coin, so with a weak learner the EE never
    # converges and `close_pos_thresh` never fires. Run 1 measured exactly this —
    # 82/90/85 committed-reach steps in iterations 15-17 with `reached_grasp` = 0
    # throughout, and ONE close label in the entire run. Forcing the expert here
    # converts every standoff arrival into a close label.
    #
    # What it costs: the states recorded during the reach are on the EXPERT's
    # distribution, not the learner's, which is the covariate shift DAgger exists
    # to remove — reintroduced in the region that matters most. It is mitigated by
    # the takeover starting from wherever the LEARNER arrived (so the segment is
    # seeded by a learner-induced state, and varies with it), and it is only the
    # last few waypoints. The GA-DDPG/Phase-3 tail splice makes the same trade.
    #
    # It is falsifiable: if collection `reached_grasp` climbs while eval
    # `chance_rate` stays 0, the takeover is manufacturing labels the policy
    # cannot use, and the honest conclusion is that the reach itself is unlearned.
    expert_after_commit: bool = False
    reach_skip_eps: float = 0.01
    # ----- DART (see the module docstring) -----
    # Per-step probability of replacing the executed action with a random jump,
    # drawn BEFORE the beta coin so this is the unconditional rate. 0 disables
    # DART *and* leaves the rng stream untouched (the ratio is tested first, so
    # no draw happens) — a dart_ratio: 0 run is bit-identical to one built before
    # DART existed.
    dart_ratio: float = 0.0
    # Upper edge of the trigger band, in metres from the pre-grasp standoff. The
    # lower edge is `reach_commit_dist` by construction: below it the reach has
    # committed and DART is off. 0.20 puts every jolt in the last ~4-6 approach
    # steps, which is where off-plan coverage is missing and where the recovery
    # is still short enough to fit in the remaining step budget.
    dart_max_dist: float = 0.20
    # GA-DDPG's magnitudes (env/panda_scene.py random_perturb), kept deliberately:
    # 0.04 m is exactly `ee_step`, so one jolt displaces the EE by one step's
    # worth in a random direction — within what the policy could plausibly undo
    # in a step or two, which is the point.
    dart_pos_mag: float = 0.04
    dart_rot_mag: float = 0.2

    # ----- DART-paper noise injection (run 18; Laskey et al. 2017, 1703.09327) --
    # "jolt"       — runs 1-17: with probability dart_ratio, inside a distance
    #                band, REPLACE the executed action with a uniform draw from a
    #                fixed box (+-dart_pos_mag / +-dart_rot_mag).
    # "dart_noise" — the paper's actual method: on every EXPERT-driven step,
    #                execute pi*(s) + eps with eps ~ N(0, Sigma). Noise is ADDED
    #                to the supervisor's control stream rather than replacing it,
    #                it is Gaussian with a FULL covariance rather than a uniform
    #                box, and its magnitude is ESTIMATED from the learner's own
    #                error rather than hand-tuned.
    #
    # The label is the clean pi*(s) in BOTH modes, which is the one thing the
    # jolt already got right (paper: "the algorithm records what the supervisor
    # should have done, not the noisy action that was executed").
    #
    # Sigma is re-estimated every DAgger iteration from the learner-supervisor
    # discrepancy (paper Eq. 3):
    #     Sigma_hat_{k+1} = (1/T) sum_t (pi_theta(x_t) - pi*(x_t))(...)^T
    # then trace-rescaled to a fixed anticipated error level (paper Eq. 4):
    #     Sigma^alpha = (alpha / (T tr(Sigma_hat))) Sigma_hat
    # with alpha = dart_alpha_scale * T * tr(Sigma_hat_1), i.e. the noise MAGNITUDE
    # is pinned at dart_alpha_scale x the first iteration's measured error while
    # its SHAPE keeps tracking which directions the learner is currently wrong in.
    # That rescaling is the paper's safeguard against the estimate running away.
    #
    # Both quantities are already computed every step here: `policy_action[:6]`
    # (queried unconditionally, even on expert steps) and `expert_delta`, in the
    # same real units. So the estimator costs nothing but an outer product.
    dart_mode: str = "jolt"
    # Sigma^alpha for THIS iteration, 6x6, supplied by the driver. None => fall
    # back to the jolt-equivalent bootstrap (see dart_bootstrap_sigma), which is
    # what iteration 1 uses since no learner error has been measured yet.
    dart_sigma: Optional[np.ndarray] = None
    # alpha at the FIRST DAgger iteration. The paper holds alpha fixed; we anneal
    # it to `dart_alpha_end` (see dart_alpha_at) so exploration noise shrinks as
    # the learner improves, which is the same reasoning behind annealing beta.
    dart_alpha_scale: float = 3.0
    # alpha at the LAST iteration. Equal to dart_alpha_scale => the paper's fixed
    # alpha. Noise sd scales as sqrt(alpha), so 3.0 -> 0.5 is a 2.4x reduction.
    dart_alpha_end: float = 3.0
    # Fraction of expert steps to perturb. 1.0 is the paper (noise on the whole
    # control stream); lower values interpolate toward the jolt's sparse firing.
    dart_noise_ratio: float = 1.0
    # ----- per-phase scaling of Sigma inside the COMMITTED REACH -----
    # The estimator pools discrepancies over the WHOLE episode, so one Sigma
    # covers both phases — but the two phases do not tolerate the same noise. The
    # jolt encoded that explicitly: dart_reach_pos_mag (0.01202) is 3.33x smaller
    # than dart_pos_mag (0.04), because a 4 cm displacement 5 cm from the object
    # knocks it out of the hand, while rotation was left alone (both 0.3 rad).
    # Without this the reach would be perturbed at free-phase magnitude, which is
    # a regression against tuning runs 12-17 paid for.
    #
    # Applied as a congruence transform D Sigma D with
    # D = diag([pos]*3 + [rot]*3), so it scales standard deviations, preserves
    # positive-semidefiniteness, and keeps the correlation structure intact.
    # Defaults reproduce the jolt's own ratio: 0.01202/0.04 = 0.3005, 0.3/0.3 = 1.
    dart_reach_pos_scale: float = 0.3005
    dart_reach_rot_scale: float = 1.0

    # ----- DART inside the COMMITTED REACH (run 12) -----
    # The band above stops at `reach_commit_dist` by construction, so DART has
    # never fired in the reach — and the reach is exactly where the policy fails.
    # Measured over runs 4/6/7/8/9 (55 evaluations x 100 scenes): the policy's
    # closest approach to the pinned grasp is 0.069-0.113 m / 0.46-0.80 rad
    # against an expert that reaches 0.014-0.023 m / 0.04-0.10 rad, and
    # `near_rate` never exceeded 0.08. The cause is visible in the labels: OMG
    # aligns the wrist during the free approach and then reaches in a straight
    # line, so per-axis |drot| over the last four demonstration steps is
    # 0.0005-0.0136 rad against 0.050-0.059 during the approach. The
    # demonstrations contain almost no orientation correction near the object,
    # so the policy has never been shown one.
    #
    # This is safe here for a reason specific to the committed reach: the target
    # is a FROZEN JOINT CONFIGURATION commanded absolutely (`target_jp =
    # expert_target_jp`), not a delta that has to be re-derived, and `reach_i`
    # saturates at the final waypoint. So recovery is a servo to a fixed target
    # rather than an accumulating error — displace the gripper and it converges
    # back onto the grasp on its own.
    #
    # 0 disables and draws nothing, so a run with this off is bit-identical to
    # runs 4-11.
    dart_reach_ratio: float = 0.0
    # Magnitudes MATCHED TO THE DEMONSTRATIONS' OWN REACH STEPS, measured on
    # train_pinned_omg_ok over the last 5 label-producing steps of each episode:
    # per-axis mean |dpos| = 0.01202 m, mean |drot| = 0.01549 rad.
    #
    # NOTE the factor of two: uniform(-m, m) has mean |.| = m/2, so setting the
    # parameter TO the measured mean makes a jolt average HALF a reach step
    # (0.0060 m / 0.0077 rad per axis). Doubling these to 0.024 / 0.031 would
    # make it average a full reach step. Deliberately the smaller of the two —
    # it keeps the perturbed states on the reach's own scale, which is what
    # makes the recovery expressible by the next frozen waypoint and keeps the
    # resulting labels at the same magnitude as the rest of the reach.
    #
    # These DEFAULTS are the demo-matched values. Run 12 keeps the translation
    # one and overrides the rotation to 0.3: 0.0155 rad is ~0.9 deg per axis
    # against the ~0.56 rad of accumulated wrist error the run exists to teach
    # recovery from, and the demonstrations' near-object rotation scale is the
    # diagnosed PROBLEM — copying it into the jolt would reproduce it rather than
    # probe it. Kept as the default anyway because it is the measured, principled
    # number; the override belongs in the run config where it can be read.
    dart_reach_pos_mag: float = 0.01202
    dart_reach_rot_mag: float = 0.01549
    # Rejection sampling: a jolt is only executed if neither the jolted pose NOR
    # the servo back to the next reach waypoint brings the gripper's control
    # points within `dart_reach_clearance` of a hand- or object-flagged point.
    # Checking the endpoint alone is not enough — the recovery is a straight line
    # in joint space, and cutting diagonally into the object is the failure that
    # put 151 scenes on the exclusion list. After `dart_reach_max_tries` refusals
    # the step simply takes no jolt.
    #
    # Rejecting is not a biased noise model: a jolt that knocks the object out of
    # the hand ENDS the episode, so it carries no expert label and was never
    # training signal. This conditions on the support where pi* is defined.
    dart_reach_max_tries: int = 5
    dart_reach_clearance: float = 0.01   # metres
    dart_reach_path_steps: int = 4       # interpolation samples along the recovery


def _grasp_moved(prev, cur, tol: float = 1e-4) -> bool:
    """Did the planner re-select a different goal grasp between two plans?"""
    if prev is None or cur is None:
        return False
    return bool(np.linalg.norm(np.asarray(prev)[:3, 3] - np.asarray(cur)[:3, 3]) > tol)


def _ee_pos(obs) -> np.ndarray:
    return np.asarray(obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3],
                      dtype=np.float64)


def derived_standoff_pose(grasp_pose, standoff_dist: float, reach_tail: int):
    """The pre-grasp standoff as a pure function of the grasp pose — no planner.

    OMG builds its standoff ramp in CARTESIAN space and does not re-optimise it
    (planner.py:356 `standoff_grasp_global = pose_grasp_global @ pose_standoff`,
    with `pose_standoff[k, 0, 2, 3] = -standoff_dist * linspace(0, 1, n,
    endpoint=False)[k]`). The ramp's IK chain is APPENDED to the optimised free
    trajectory rather than folded into it — core.py:639 labels every index past
    `cfg.timesteps` as "standoff" precisely because it sits outside the
    optimiser's window. So `traj[-reach_tail]` is, by construction, the IK
    solution of

        grasp @ translate(0, 0, -standoff_dist * (1 - 1/reach_tail))

    i.e. the grasp backed off along its own local -z, sharing the grasp's
    rotation exactly.

    Measured against `env.get_omg_standoff_pose()` over 113 replans on 4 train
    scenes: max position gap 5.6e-7 m, max rotation gap 1.3e-6 rad, and the
    reach-commit fired on the identical step in 4/4 scenes. It is the same pose,
    to floating point.

    The point of deriving it is AVAILABILITY, not accuracy. The planner's copy
    only exists after a successful plan, so it is None at reset and is not
    refreshed on a step whose replan failed. This form needs nothing but the
    pinned grasp, which is static within an episode (measured: ||G_t - G_0|| max
    4.1e-15 m over 150 replans), so one read at step 0 is good for the whole
    episode — and it is equally available to the evaluator and the RL reward,
    neither of which runs OMG at all.
    """
    off = float(standoff_dist) * (1.0 - 1.0 / int(reach_tail))
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = -off
    return np.asarray(grasp_pose, dtype=np.float64) @ T


def dart_bootstrap_sigma(pos_mag: float, rot_mag: float) -> np.ndarray:
    """Iteration-1 noise, before any learner error has been measured.

    DART initialises with an arbitrary psi_0; the least arbitrary choice here is
    the noise runs 1-18 already used, so iteration 1 perturbs at a magnitude this
    project has tuned. A uniform draw on [-a, a] has variance a^2/3, so this is
    the diagonal Gaussian with the same per-axis variance as one jolt.

    Only iteration 1 uses it. From iteration 2 the covariance is measured, and
    the trace target is anchored to that first MEASUREMENT (tr(Sigma_hat_1)), not
    to this bootstrap — so a poor psi_0 does not propagate.
    """
    return np.diag(np.array(
        [pos_mag ** 2 / 3.0] * 3 + [rot_mag ** 2 / 3.0] * 3, dtype=np.float64))


def dart_alpha_at(iteration: int, num_iters: int,
                  alpha_start: float, alpha_end: float) -> float:
    """alpha for a given DAgger iteration, linear from start to end.

    `iteration` is 1-based and is the iteration the noise will be USED in, so the
    driver passes i+1 when it estimates Sigma at the end of iteration i. Returns
    alpha_start when num_iters <= 1.

    Noise sd scales as sqrt(alpha), not alpha — 3.0 -> 0.5 is 1.73x -> 0.71x the
    measured error, a 2.4x reduction in sd across the run.
    """
    if num_iters <= 1:
        return float(alpha_start)
    frac = min(max((int(iteration) - 1) / (int(num_iters) - 1), 0.0), 1.0)
    return float(alpha_start) + (float(alpha_end) - float(alpha_start)) * frac


def dart_scaled_sigma(sigma_hat: np.ndarray, target_trace: float) -> np.ndarray:
    """Paper Eq. 4: rescale Sigma_hat so its trace hits `target_trace`.

        Sigma^alpha = (alpha / (T tr(Sigma_hat))) Sigma_hat,  alpha/T = target_trace

    This is the safeguard that keeps the estimate from running away: the SHAPE of
    Sigma keeps tracking which action directions the learner is currently wrong
    in, while the overall MAGNITUDE stays pinned at a level chosen once. Without
    it a learner that got worse would be answered with ever more noise.
    """
    tr = float(np.trace(sigma_hat))
    if not np.isfinite(tr) or tr <= 1e-12:
        return np.asarray(sigma_hat, dtype=np.float64)
    return np.asarray(sigma_hat, dtype=np.float64) * (float(target_trace) / tr)


# The same six Panda control points the PM loss uses (bc/losses.py), in the hand
# link frame — which is also the frame the observation point cloud is expressed
# in, so a candidate delta can be checked against the cloud with no transform.
_CTRL_PTS = np.array([[0.000, 0.000, 0.000],
                      [0.000, 0.000, 0.000],
                      [0.053, 0.000, 0.075],
                      [-0.053, 0.000, 0.075],
                      [0.053, 0.000, 0.105],
                      [-0.053, 0.000, 0.105]], dtype=np.float64)


def _delta_to_points(delta) -> np.ndarray:
    """Where the gripper control points land under a 6-D EE-frame delta pose."""
    R = euler2mat(*np.asarray(delta, dtype=np.float64)[3:6])
    return _CTRL_PTS @ R.T + np.asarray(delta, dtype=np.float64)[:3]


def _path_min_dist(obstacles, a, b, path_steps: int) -> float:
    """Closest any gripper control point comes to `obstacles` while the pose is
    swept from delta `a` to delta `b`. Interpolating the 6-vector is exact in
    translation and a small-angle approximation in rotation, which is all these
    magnitudes are."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    best = np.inf
    for t in np.linspace(0.0, 1.0, max(int(path_steps), 1) + 1):
        pts = _delta_to_points(a + t * (b - a))
        d2 = ((obstacles[None, :, :] - pts[:, None, :]) ** 2).sum(-1)
        best = min(best, float(d2.min()))
    return float(np.sqrt(best))


def _jolt_is_safe(pc, jolt_delta, recover_delta, *, clearance: float,
                  path_steps: int) -> bool:
    """Would this jolt, AND the servo back to the next reach waypoint, stay clear?

    `pc` is the observation cloud [N, 5] in the EE frame: xyz + ycb_flag +
    hand_flag. The gripper control points are in the same frame, so both the
    jolted pose and the recovery path can be checked directly against it.

    Checking the jolt's ENDPOINT alone would miss the failure that matters: the
    recovery is a straight line in joint space from the displaced configuration
    to a frozen waypoint, and a lateral displacement makes that line cut through
    the object. So the whole segment is swept.

    The test is RELATIVE, not absolute, and it has to be. During the reach the
    gripper is closing on an object held in a human hand, so it is *supposed* to
    end up within a centimetre of hand and object points — an absolute floor
    rejects the expert's own trajectory. Measured on a 6-episode shakedown at
    ratio 1.0, an absolute 0.01 m floor refused 65 draws to accept 7. So the bar
    is `min(clearance, nominal)`: a jolt is allowed when it leaves the gripper no
    closer than the unperturbed reach was already going to come, and never
    closer than `clearance` when there was room to spare.

    True when nothing to avoid is in the cloud — an empty scene cannot refute a
    jolt.
    """
    obstacles = pc[(pc[:, 3] > 0.5) | (pc[:, 4] > 0.5), :3]
    if obstacles.shape[0] == 0:
        return True

    zero = np.zeros(6, dtype=np.float64)
    rec = zero if recover_delta is None else np.asarray(recover_delta, dtype=np.float64)
    nominal = _path_min_dist(obstacles, zero, rec, path_steps)
    jolted = _path_min_dist(obstacles, np.asarray(jolt_delta, dtype=np.float64),
                            rec, path_steps)
    return jolted >= min(float(clearance), nominal)


def collect_dagger_episode(sim, runner, scene_idx, *, rng,
                           beta: float, params: CollectParams, pin_table=None):
    """Roll out `runner`'s policy on `scene_idx`, labelling every visited state.

    `runner` is a PolicyRunner, so this works unchanged for the Phase-1
    single-frame policy and the Phase-2 ACT policy — the DAgger labelling does
    not depend on how the learner produces its action.

    Returns (episode | None, stats). `episode` is None when the scene is
    unusable (OMG cannot plan at step 0) or nothing was recorded.
    """
    env = sim.env
    obs = env.reset(idx=scene_idx)
    sim.point_listener.reset()
    runner.reset()

    prev_act6d = np.zeros(6, dtype=np.float32)
    point_clouds, robot_states, expert_actions = [], [], []

    def _standoff_for(g):
        """Pre-grasp standoff: derived from the grasp, or read off the plan.

        One place so the two call sites below cannot drift apart. Returns None
        exactly where the old code did, so `derive_standoff: false` is the
        pre-run-18 behaviour unchanged.
        """
        if params.derive_standoff:
            if g is None:
                return None
            return derived_standoff_pose(g, params.standoff_dist,
                                         params.reach_tail)
        return env.get_omg_standoff_pose()

    grasp_pose = None      # OMG traj[-1]: the pose the gripper closes at
    standoff_pose = None   # OMG traj[-5]: pre-grasp standoff (horizon sizing)
    # Seed the standoff BEFORE any plan exists, so the step-0 horizon is sized
    # the same way every other step's is. Uses the pin table's stored pose (the
    # only grasp available at reset) and is overwritten by the live grasp on the
    # first successful plan — see CollectParams.derive_standoff for why a
    # match_tol-sized error is immaterial to a horizon.
    if params.derive_standoff and pin_table is not None:
        entry = pin_table.entries.get(int(scene_idx))
        if entry is not None:
            standoff_pose = derived_standoff_pose(
                np.asarray(entry["ee_pose_world"], dtype=np.float64),
                params.standoff_dist, params.reach_tail)
    committed_reach = None # frozen standoff->grasp waypoints, followed by index
    reach_i = 0
    goal_idx_prev = None   # OMG's grasp SELECTION index, to detect real switches
    pinned = False
    n_omg_fail = n_goal_switch = n_expert_steps = n_close_labels = 0
    n_reach_steps = 0
    min_pos, min_rot = float("inf"), float("inf")
    # `n_policy_close_cmds` counts every premature close the policy commands;
    # `ended_on_close` is only whether one of them ENDED the episode, which is
    # what stop_on_policy_close controls.
    n_policy_close_cmds = 0
    ended_on_close = False
    policy_close_step = -1
    done = False
    reason = "MAX_STEPS"
    # Who executed the most recent step, and whether that step wrote a label.
    # Both are needed to decide, on ENV_DONE, whether the last recorded pair is
    # contaminated — see the ENV_DONE branch at the bottom of the loop.
    last_exec_expert = False
    recorded = False
    n_dropped_tail = 0
    # DART: how many jolts fired, and how many of them ended the episode. The
    # second is the magnitude tripwire — a +-4 cm jump near the standoff is close
    # to the MANO hand, and if a large fraction of jolts trip the benchmark's
    # contact/drop check then dart_pos_mag is knocking the object loose rather
    # than displacing the gripper.
    n_dart = 0
    n_dart_env_done = 0
    # Reach jolts are counted separately from approach jolts: they fire under a
    # different rule, at a different magnitude, and their risk profile is
    # different (a jolt 5 cm from the object is not a jolt 20 cm from it).
    # `n_dart_reject` counts REFUSED candidate draws, not steps — if it is near
    # zero the clearance test never binds and the magnitude can safely rise; if
    # it approaches max_tries x the jolt count, the magnitude is too large for
    # the geometry and jolts are being silently skipped.
    n_dart_reach = 0
    n_dart_reject = 0
    last_exec_dart = False
    # Per-episode OUTCOME (params.outcome_check). `reason` above stays the loop's
    # terminal cause; these are the handover's result, on the evaluator's taxonomy.
    info = {}
    outcome = ""
    ep_success = 0
    # DART Sigma estimator (paper Eq. 3): running sum of d d^T over every step
    # where BOTH the learner's and the supervisor's action exist, and the count.
    # Summed rather than averaged here so `collect_iteration` can pool across
    # episodes of different lengths without weighting short ones more heavily.
    dart_dd_sum = np.zeros((6, 6), dtype=np.float64)
    dart_dd_n = 0
    n_dart_noise = 0
    # The covariance actually used this iteration. `dart_sigma` is supplied by the
    # driver from the PREVIOUS iteration's estimate; iteration 1 has none.
    sigma = None
    if params.dart_mode == "dart_noise":
        sigma = (params.dart_sigma if params.dart_sigma is not None
                 else dart_bootstrap_sigma(params.dart_pos_mag, params.dart_rot_mag))
        sigma = np.asarray(sigma, dtype=np.float64)
    # Magnitude of every APPROACH label recorded. This is the stall detector: the
    # standoff failure mode is labels decaying to ~0 while the EE sits 6.6 cm
    # short, which is invisible in step counts but obvious here.
    label_pos_mags: list[float] = []

    for step in range(params.max_steps):
        # ----- state exactly as the policy / dataset sees it -----
        pc = _point_cloud(obs, sim.point_listener, sim.panda_base_inv_tf)
        rs = _robot_state(obs, prev_act6d)

        expert_target_jp = None
        # ||ee - standoff||, once a plan has told us where the standoff is. Drives
        # BOTH the reach commit and the DART trigger band, so it is computed once
        # rather than in two places that could drift apart. Stays None while the
        # reach is committed (no replan, so nothing updates it) and on a step
        # whose plan failed — DART is off in both cases.
        d_standoff = None
        # With a derived standoff the pose does not depend on THIS step's plan,
        # so the distance is known even when the replan fails — the one case
        # where the plan-read version leaves it None and silently disables DART.
        # Recomputed identically inside the plan-success branch below; the grasp
        # is static within an episode, so that is a no-op.
        if params.derive_standoff and standoff_pose is not None:
            d_standoff = float(np.linalg.norm(
                _ee_pos(obs) - standoff_pose[:3, 3]))
        if committed_reach is None:
            # ----- APPROACH: pi*(s) is a FRESH OMG plan from the CURRENT
            # (drifted) config, and the label is its first waypoint. The horizon
            # is distance proportional once the standoff is known, so plan[0]
            # stays at the demonstrations' per-step scale instead of collapsing
            # into one huge jump late in the episode.
            if standoff_pose is not None:
                horizon = dynamic_replan_horizon(
                    obs, standoff_pose, params.ee_step, params.reach_tail,
                    params.min_free, params.max_horizon)
            else:
                horizon = int(params.first_horizon)

            plan, _ = env.run_omg_planner(horizon, scene_idx, reset_scene=(step == 0))
            if plan is None:
                n_omg_fail += 1
                if step == 0:
                    return None, {"scene_idx": scene_idx, "skipped": True,
                                  "reason": "OMG_FAIL_STEP0", "steps": 0,
                                  "n_omg_fail": n_omg_fail}
            else:
                expert_target_jp = plan[0]
                # Track the SELECTION INDEX, not the pose: `flip_grasp` augments
                # the goal set with wrist-flipped duplicates that share an EE
                # position, so comparing poses misses those switches entirely.
                gi = env.get_omg_goal_idx()
                if goal_idx_prev is not None and gi != goal_idx_prev:
                    n_goal_switch += 1
                goal_idx_prev = gi
                new_grasp = env.get_omg_goal_grasp_pose()
                if new_grasp is not None:
                    grasp_pose = new_grasp
                new_standoff = _standoff_for(grasp_pose)
                if new_standoff is not None:
                    standoff_pose = new_standoff

                # ----- PIN THE GOAL GRASP (once, on the step-0 plan) -----
                # OMG re-decides goal_idx = argmin ||traj.start - goal_set[i]||
                # in joint space on EVERY plan, so with the policy driving the
                # target can move mid-episode (measured: 32/90 replans under
                # +-15 cm perturbation switched grasp; one scene cycled through
                # four, shifting the target up to 10 cm). Pruning the goal set to
                # the table's committed grasp makes the argmin constant, and
                # makes DAgger aim at the SAME grasp the demonstrations used.
                if step == 0 and pin_table is not None:
                    if pin_table.apply(env, scene_idx):
                        pinned = True
                        # Pruning renumbers the goal set (the pinned grasp becomes
                        # index 0), so re-baseline the switch counter or the
                        # renumbering itself reads as a switch on the next step.
                        goal_idx_prev = env.get_omg_goal_idx()
                        # the pin can change the goal, so re-read both poses
                        g = env.get_omg_goal_grasp_pose()
                        if g is not None:
                            grasp_pose = g
                        s_ = _standoff_for(grasp_pose)
                        if s_ is not None:
                            standoff_pose = s_

                # ----- COMMIT THE REACH once the EE reaches the standoff -----
                # Beyond this point replanning is actively harmful: the standoff
                # ramp is in OMG's goal set, so a plan from at/past the standoff
                # has a *stationary* free portion and plan[0] decays to zero
                # (measured: the EE parks 6.6 cm short of the grasp and every
                # further label is ~0 — the rl_run7 hover). plan[0] structurally
                # cannot express the reach: the reach is always the LAST
                # reach_tail waypoints. So freeze them and follow by index.
                # convert_target_joint_position_to_action still recomputes the
                # delta from the CURRENT EE, so the labels stay corrective under
                # drift; only the TARGET stops moving.
                if standoff_pose is not None:
                    d_standoff = float(np.linalg.norm(
                        _ee_pos(obs) - standoff_pose[:3, 3]))
                if (d_standoff is not None
                        and len(plan) >= params.reach_tail
                        and d_standoff <= params.reach_commit_dist):
                    committed_reach = np.asarray(plan[-params.reach_tail:])
                    # The tail's first entry IS the standoff, i.e. where the EE
                    # already is — skip leading no-ops so the first reach label
                    # is a real step.
                    mags = [float(np.linalg.norm(np.asarray(
                        env.convert_target_joint_position_to_action(w))[:3]))
                        for w in committed_reach]
                    reach_i = next((j for j, m in enumerate(mags)
                                    if m >= params.reach_skip_eps), 0)
                    expert_target_jp = committed_reach[reach_i]

        if committed_reach is not None:
            # ----- REACH: follow the frozen standoff->grasp waypoints by index.
            # Past the end we hold the final waypoint (the grasp itself), so the
            # label stays "go to the grasp" and shrinks monotonically until the
            # close threshold fires — no replan, so no retreat, no oscillation.
            reach_i = min(reach_i, len(committed_reach) - 1)
            expert_target_jp = committed_reach[reach_i]
            n_reach_steps += 1

        expert_delta = (
            np.asarray(env.convert_target_joint_position_to_action(expert_target_jp),
                       dtype=np.float32)          # [6] real units
            if expert_target_jp is not None else None)

        # ----- gripper-close label: EE within tolerance of the grasp pose -----
        # Uses the grasp the CURRENT plan aims at, so the close decision and the
        # approach labels always refer to the same target. Survives a failed
        # replan via the cached pose.
        at_grasp = False
        if grasp_pose is not None:
            pos_err, rot_err = ee_grasp_pose_error(obs, grasp_pose)
            min_pos, min_rot = min(min_pos, pos_err), min(min_rot, rot_err)
            at_grasp = (pos_err <= params.close_pos_thresh
                        and rot_err <= params.close_rot_thresh)

        # ----- record the labelled (state, expert action) pair -----
        if at_grasp:
            label = CLOSE_LABEL.copy()
            n_close_labels += 1
        elif expert_delta is not None:
            label = np.concatenate([expert_delta, [1.0]]).astype(np.float32)
            label_pos_mags.append(float(np.linalg.norm(expert_delta[:3])))
        else:
            label = None  # replan failed and we are not at the grasp: no label
        recorded = label is not None
        if recorded:
            point_clouds.append(pc)
            robot_states.append(rs)
            expert_actions.append(label)

        if at_grasp and params.stop_on_close_label:
            reason = "CLOSE_LABEL"
            # The labels are already appended above, so scoring here cannot change
            # what D_i contains — it only runs the sim forward to observe what the
            # close the expert just LABELLED would actually have done.
            if params.outcome_check:
                held, obs = grasp_held_after_hold(
                    env, obs, sim.steps_action_repeat, params.hold_steps)
                ep_success = int(held)
                outcome = ("GRASP_OK" if held
                           else "DROP" if bool(getattr(env, "_dropped", False))
                           else "NO_RELEASE" if not bool(env.ycb.released)
                           else "GRASP_MISS")
            break

        # ----- the policy's own action (drives the state distribution) -----
        # Queried every step even on beta-expert steps: for a stateful runner
        # (ACT's history buffer / chunk queue) that keeps its bookkeeping
        # advancing exactly as it would in deployment. For the single-frame
        # policy it is simply one forward pass.
        policy_action = runner.act(pc, rs)   # [7], ch6 in {0,1}

        # ----- DART Sigma estimate: (pi_theta(x) - pi*(x))(...)^T at this state.
        # Accumulated in BOTH modes so a jolt run still reports what the noise
        # WOULD have been — that makes the two modes comparable on the same
        # diagnostic rather than only after switching. Both vectors are the 6-D
        # real-unit delta (metres, radians), so the outer product is in the same
        # units the noise is drawn in.
        if expert_delta is not None:
            d = (policy_action[:6].astype(np.float64)
                 - np.asarray(expert_delta, dtype=np.float64))
            dart_dd_sum += np.outer(d, d)
            dart_dd_n += 1

        # ----- DART? drawn BEFORE the beta coin, so `dart_ratio` is the
        # unconditional per-step rate rather than a rate conditioned on who would
        # otherwise have driven — the point is to displace the state, and it does
        # not matter whose step is spent doing it. `params.dart_ratio > 0.0` is
        # tested FIRST so a DART-off run draws nothing and its rng stream is
        # identical to a pre-DART run's.
        #
        # `not at_grasp` closes the one hole the band leaves: with
        # stop_on_close_label off, a close label can be written before the reach
        # commits (the EE happened to land inside the thresholds), and jolting
        # then would throw away the arrival we just recorded.
        #
        # A jolt also never steals a step on which the POLICY asked to close.
        # DART overriding an approach action just displaces the state, which is
        # the point; overriding a close would suppress a TERMINAL decision, and
        # `policy_closed` / `c_policy_close` / `mean_policy_close_step` would then
        # measure something different in a DART run than in the run it is being
        # compared against. The label written above this line is the same either
        # way (an approach delta saying "do not close here"), so declining the
        # jolt costs no supervision — only the episode's continuation differs, and
        # those steps were about to end the episode anyway.
        # `dart_mode` first, so switching to dart_noise turns the jolt paths off
        # without touching their knobs — the two are alternative implementations
        # of the same idea, never stacked.
        dart = (params.dart_mode == "jolt"
                and params.dart_ratio > 0.0
                and committed_reach is None
                and not at_grasp
                and policy_action[6] >= 0.5
                and d_standoff is not None
                and d_standoff <= params.dart_max_dist
                and rng.uniform() < params.dart_ratio)

        # ----- DART inside the committed reach (run 12) -----
        # Same guards on `at_grasp` and on the policy's close, for the same
        # reasons; the band conditions are replaced by "the reach has committed".
        # The jolt is REJECTION SAMPLED: drawn, checked against the cloud
        # together with the servo back to the next frozen waypoint, redrawn on a
        # refusal, and abandoned after `dart_reach_max_tries`. See CollectParams.
        dart_reach_delta = None
        if (params.dart_mode == "jolt"
                and params.dart_reach_ratio > 0.0
                and committed_reach is not None
                and not at_grasp
                and policy_action[6] >= 0.5
                and rng.uniform() < params.dart_reach_ratio):
            # Where the servo will pull the gripper AFTER the jolt: reach_i is
            # incremented once this step executes, and the index is clamped, so
            # the recovery target is the next waypoint (or the grasp at the end).
            nxt = committed_reach[min(reach_i + 1, len(committed_reach) - 1)]
            recover_delta = np.asarray(
                env.convert_target_joint_position_to_action(nxt), dtype=np.float64)
            for _ in range(int(params.dart_reach_max_tries)):
                cand = np.concatenate([
                    rng.uniform(-params.dart_reach_pos_mag,
                                params.dart_reach_pos_mag, size=3),
                    rng.uniform(-params.dart_reach_rot_mag,
                                params.dart_reach_rot_mag, size=3),
                ])
                if _jolt_is_safe(pc, cand, recover_delta,
                                 clearance=params.dart_reach_clearance,
                                 path_steps=params.dart_reach_path_steps):
                    dart_reach_delta = cand.astype(np.float32)
                    break
                n_dart_reject += 1

        # ----- choose what to EXECUTE: pi_i = beta*pi* + (1-beta)*pi_hat -----
        # pi_i = beta*pi* + (1-beta)*pi_hat, except that the committed reach can
        # be forced onto the expert (see CollectParams.expert_after_commit): the
        # endgame is where the close labels live and where a weak learner stalls.
        forced_expert = bool(params.expert_after_commit and committed_reach is not None)
        # A reach jolt PRE-EMPTS the forced expert, exactly as an approach jolt
        # pre-empts the beta coin. Placed before the beta draw so a
        # dart_reach_ratio: 0 run consumes the rng identically to runs 4-11.
        use_expert = (not dart) and dart_reach_delta is None \
            and expert_target_jp is not None and (
                forced_expert or rng.uniform() < beta)
        if dart or dart_reach_delta is not None:
            # Random task-space jump instead of anyone's action. The gripper stays
            # OPEN (1.0): a jolt must never be the thing that closes the hand.
            # The pair recorded above this line is untouched and entirely ordinary
            # — it is (the state we were in, what the expert would do there). Only
            # the state the NEXT step starts from is perturbed, and that step gets
            # its own fresh label: a replanned pi*(s) on the approach, or the
            # delta to the next frozen waypoint inside the reach. Either way it is
            # a genuine expert query from the displaced state, never a
            # reconstruction of the pre-jolt label.
            if dart_reach_delta is not None:
                n_dart_reach += 1
                exec_delta = dart_reach_delta
            else:
                n_dart += 1
                exec_delta = np.concatenate([
                    rng.uniform(-params.dart_pos_mag, params.dart_pos_mag, size=3),
                    rng.uniform(-params.dart_rot_mag, params.dart_rot_mag, size=3),
                ]).astype(np.float32)
            target_jp = action_to_target_joint(
                np.concatenate([exec_delta, [1.0]]).astype(np.float32), obs)
        elif use_expert:
            n_expert_steps += 1
            # ----- DART: noise ON the supervisor's action, not instead of it ---
            # pi*(s) + eps, eps ~ N(0, Sigma). The label written above this line
            # is the CLEAN expert_delta and is deliberately not touched — that
            # asymmetry (visit the noisy state, learn the clean correction) is
            # the whole mechanism: the supervisor is "forced to take corrective
            # actions" from states its own noise pushed it into.
            #
            # Only the position/rotation channels are perturbed. The gripper is
            # held OPEN, as the jolt did, because a binary close is terminal here
            # and noise must never be the thing that fires it.
            if (sigma is not None and not at_grasp
                    and (params.dart_noise_ratio >= 1.0
                         or rng.uniform() < params.dart_noise_ratio)):
                # Inside the committed reach the same Sigma is scaled down (see
                # dart_reach_pos_scale) and each draw is CLEARANCE-CHECKED against
                # the observed cloud, exactly as the reach jolt was: 5 cm from the
                # object an unlucky draw is what knocks it out of the hand. On the
                # free approach neither applies — there is nothing close to hit.
                sig_step = sigma
                tries = 1
                recover_delta = None
                if committed_reach is not None:
                    D = np.diag(np.array(
                        [params.dart_reach_pos_scale] * 3
                        + [params.dart_reach_rot_scale] * 3, dtype=np.float64))
                    sig_step = D @ sigma @ D
                    nxt = committed_reach[min(reach_i + 1,
                                              len(committed_reach) - 1)]
                    recover_delta = np.asarray(
                        env.convert_target_joint_position_to_action(nxt),
                        dtype=np.float64)
                    tries = int(params.dart_reach_max_tries)

                eps = None
                for _ in range(max(tries, 1)):
                    cand = rng.multivariate_normal(np.zeros(6), sig_step)
                    if recover_delta is None or _jolt_is_safe(
                            pc, cand, recover_delta,
                            clearance=params.dart_reach_clearance,
                            path_steps=params.dart_reach_path_steps):
                        eps = cand
                        break
                    n_dart_reject += 1

                if eps is None:
                    # Every draw was refused: take the clean expert step rather
                    # than force an unsafe one. Counted in n_dart_reject above.
                    target_jp = expert_target_jp
                    exec_delta = expert_delta
                else:
                    exec_delta = (np.asarray(expert_delta, dtype=np.float64)
                                  + eps).astype(np.float32)
                    target_jp = action_to_target_joint(
                        np.concatenate([exec_delta, [1.0]]).astype(np.float32), obs)
                    n_dart_noise += 1
            else:
                target_jp = expert_target_jp
                exec_delta = expert_delta
        else:
            if policy_action[6] < 0.5:
                # Count and timestamp the premature close REGARDLESS of what we
                # do about it. Booking these inside the stop_on_policy_close
                # branch would zero `policy_closed` / `policy_close_step` the
                # moment that flag is turned off, hiding the very behaviour the
                # flag exists to keep collecting corrections for.
                n_policy_close_cmds += 1
                if policy_close_step < 0:
                    policy_close_step = step
                if params.stop_on_policy_close:
                    ended_on_close = True
                else:
                    policy_action = policy_action.copy()
                    policy_action[6] = 1.0  # override: keep approaching
            target_jp = action_to_target_joint(policy_action, obs)
            exec_delta = policy_action[:6].astype(np.float32)
        last_exec_expert = bool(use_expert)
        last_exec_dart = bool(dart or dart_reach_delta is not None)

        prev_act6d = np.asarray(exec_delta, dtype=np.float32).copy()

        for _ in range(sim.steps_action_repeat):
            obs, _, done, info = env.step(target_jp)
            if done:
                break

        # advance along the committed reach (clamped at the grasp above)
        if committed_reach is not None:
            reach_i += 1

        if ended_on_close:
            reason = "POLICY_CLOSE"
            # The POLICY committed this close, so unlike CLOSE_LABEL this scores
            # the learner's own decision — the same thing the evaluator measures,
            # just at beta > 0 and on a collection scene.
            if params.outcome_check:
                held, obs = grasp_held_after_hold(
                    env, obs, sim.steps_action_repeat, params.hold_steps)
                ep_success = int(held)
                outcome = ("GRASP_OK" if held
                           else "DROP" if bool(getattr(env, "_dropped", False))
                           else "NO_RELEASE" if not bool(env.ycb.released)
                           else "GRASP_MISS")
            break
        if done:
            reason = "ENV_DONE"
            # `info["status"]` was discarded before run 18, which is why ENV_DONE
            # was an undifferentiated bucket. It carries the benchmark's own
            # failure flags, so this is where HUMAN_CONTACT / DROP / BENCH_TIMEOUT
            # comes from — no extra simulation needed.
            if params.outcome_check:
                outcome = _status_name(int(info.get("status", 0)))
            # The benchmark just killed the episode — usually the object knocked
            # out of the hand by the lateral swing into the pre-grasp pose. WHO
            # drove this step decides whether the pair we just recorded is
            # contaminated, because the label and the executed action are only
            # the same thing on an expert step:
            #
            #   policy drove -> the colliding action was the POLICY's, and the
            #       policy's action is never recorded. The stored pair is (a state
            #       the policy drove itself into, what the expert would have done
            #       there) — the single most valuable kind of pair DAgger
            #       collects. KEEP it.
            #   expert drove -> the colliding action IS the label just written.
            #       Training on it teaches the collision. Drop that ONE pair;
            #       every earlier step of the episode is untouched.
            #   DART drove -> like the policy case, and for the same reason: the
            #       jolt is not the label, so the pair is clean. KEEP it. It is
            #       still counted below, because a jolt magnitude that routinely
            #       ends episodes is buying coverage at the cost of the episodes
            #       that would have produced the close labels.
            #
            # Only reachable on an approach label: an at_grasp step breaks above
            # before executing, so the close label can never be the one dropped.
            if last_exec_dart:
                n_dart_env_done += 1
            if last_exec_expert and recorded:
                point_clouds.pop()
                robot_states.pop()
                expert_actions.pop()
                if label_pos_mags:
                    label_pos_mags.pop()
                n_dropped_tail = 1
            break

    # Loop exhausted without closing and without the benchmark ending it — the
    # same TIMEOUT the evaluator's for/else records, so the taxonomies line up.
    if params.outcome_check and not outcome:
        outcome = "TIMEOUT"

    if len(expert_actions) == 0:
        # Reachable via the ENV_DONE pop above when the episode had exactly one
        # label and the expert's step is what ended it: nothing usable is left.
        return None, {"scene_idx": scene_idx, "skipped": True,
                      "reason": "NO_LABELS", "steps": 0, "n_omg_fail": n_omg_fail,
                      "n_dropped_tail": int(n_dropped_tail),
                      "n_dart": int(n_dart),
                      "n_dart_env_done": int(n_dart_env_done),
                      "n_dart_reach": int(n_dart_reach),
                      "n_dart_reject": int(n_dart_reject)}

    episode = {
        "point_clouds": np.asarray(point_clouds, dtype=np.float32),
        "robot_states": np.asarray(robot_states, dtype=np.float32),
        "expert_actions": np.asarray(expert_actions, dtype=np.float32),
        "scene_idx": int(scene_idx),
    }
    stats = {
        "scene_idx": int(scene_idx),
        "skipped": False,
        "steps": len(expert_actions),
        "reason": reason,
        # `reason` is WHY THE LOOP STOPPED; `outcome` is WHAT HAPPENED to the
        # handover, on the evaluator's taxonomy (GRASP_OK / GRASP_MISS /
        # NO_RELEASE / DROP / HUMAN_CONTACT / BENCH_TIMEOUT / TIMEOUT). Empty
        # unless params.outcome_check. `success` is the stable-grasp criterion, so
        # it is directly comparable to EVAL's success_rate — with the caveat that
        # a collection episode is driven by the beta MIXTURE, not by the policy.
        "outcome": outcome,
        "success": int(ep_success),
        # DART Sigma estimator, summed not averaged — see the accumulator above.
        "dart_dd_sum": dart_dd_sum,
        "dart_dd_n": int(dart_dd_n),
        "n_dart_noise": int(n_dart_noise),
        "n_omg_fail": n_omg_fail,
        "n_goal_switch": n_goal_switch,
        "n_expert_steps": n_expert_steps,
        "n_close_labels": n_close_labels,
        "n_reach_steps": n_reach_steps,
        "pinned": int(pinned),
        # The grasp the CLOSE label was scored against — what this episode's
        # labels actually aimed at. Checked across iterations by GraspRegistry;
        # `n_goal_switch` above only covers moves WITHIN this episode.
        "grasp_pose": (np.asarray(grasp_pose).tolist()
                       if grasp_pose is not None else None),
        "reached_standoff": int(committed_reach is not None),
        "reached_grasp": int(n_close_labels > 0),
        # Episodes in which the policy commanded a close at least once. Under
        # stop_on_policy_close this also ended the episode; without it the
        # episode kept going, so `n_policy_close_cmds` (how MANY steps it wanted
        # to close on) is the finer signal.
        "policy_closed": int(n_policy_close_cmds > 0),
        "n_policy_close_cmds": int(n_policy_close_cmds),
        "policy_close_step": int(policy_close_step),
        # 1 if the trailing pair was dropped because an EXPERT step ended the
        # episode. A rising count means the expert is still colliding on scenes
        # that survived filtering, i.e. exclude_scenes is not catching them.
        "n_dropped_tail": int(n_dropped_tail),
        # DART jolts fired, and how many of them ended the episode. Read as a
        # ratio: n_dart tells you the band is firing at all (0 with dart_ratio > 0
        # means dart_max_dist never triggers, or the reach commits before the EE
        # enters the band), n_dart_env_done tells you whether the magnitude is
        # destroying the episodes it was meant to enrich.
        "n_dart": int(n_dart),
        "n_dart_env_done": int(n_dart_env_done),
        # Reach jolts and the draws the clearance test refused (see the counter
        # declarations). `n_dart_env_done` covers BOTH bands, so read it against
        # n_dart + n_dart_reach.
        "n_dart_reach": int(n_dart_reach),
        "n_dart_reject": int(n_dart_reject),
        "min_pos": min_pos,
        "min_rot": min_rot,
        # Approach-label scale. `n_tiny_labels` counts labels the policy cannot
        # learn anything from (below reach_skip_eps/2); a rising count is the
        # standoff stall coming back.
        "n_approach_labels": len(label_pos_mags),
        "sum_label_pos": float(np.sum(label_pos_mags)) if label_pos_mags else 0.0,
        "n_tiny_labels": int(sum(m < 0.5 * params.reach_skip_eps
                                 for m in label_pos_mags)),
    }
    return episode, stats


# ── HDF5 writer ──────────────────────────────────────────────────────────────

class DaggerHDF5Writer:
    """Stream episodes to disk one at a time.

    Bounded memory over hundreds of episodes, and an interrupted iteration still
    leaves every episode already collected on disk (the loop's resume logic
    reads `num_episodes` back out of the attrs).
    """

    def __init__(self, path, attrs: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._f = h5py.File(self.path, "w")
        self._f.attrs["num_pts"] = NUM_PTS
        self._f.attrs["pc_channels"] = PC_CHANNELS
        self._f.attrs["robot_state_dim"] = ROBOT_STATE_DIM
        self._f.attrs["action_dim"] = ACTION_DIM
        self._f.attrs["action_format"] = "delta_pos(3)+delta_euler(3)+gripper_cmd(1)"
        self._f.attrs["dagger"] = True
        self._f.attrs["dagger_phase"] = 4
        for key, val in (attrs or {}).items():
            self._f.attrs[key] = val

    @property
    def num_episodes(self) -> int:
        return self._n

    def append(self, episode: dict) -> None:
        grp = self._f.create_group(f"episode_{self._n:05d}")
        grp.attrs["scene_idx"] = episode["scene_idx"]
        grp.attrs["num_steps"] = len(episode["expert_actions"])
        for name in ("point_clouds", "robot_states", "expert_actions"):
            grp.create_dataset(name, data=episode[name], compression="gzip")
        self._n += 1
        self._f.attrs["num_episodes"] = self._n
        self._f.flush()

    def close(self, extra: dict | None = None) -> None:
        for key, val in (extra or {}).items():
            self._f.attrs[key] = val
        self._f.attrs["num_episodes"] = self._n
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ── one DAgger iteration's worth of data ─────────────────────────────────────

def _observation_attrs(sim) -> dict:
    """Camera/renderer provenance for a shard's HDF5 attrs.

    The resolved CAMERA LIST, not just the sim cfg path: a path stays true while
    the file it names is edited, so the filename can drift away from what was
    actually collected. The renderer is here for the same reason — EGL and the
    CPU TinyRenderer do not produce identical clouds, which is why the Phase-4
    configs already warn that SIM.egl must match the base dataset.

    Degrades to empty rather than raising: this is metadata, and a missing
    attribute must never be the thing that kills a 20-hour collection.
    """
    cfg = getattr(sim, "cfg", None)
    if cfg is None:
        return {}
    try:
        hcps = cfg.ENV.HANDOVER_HAND_CAMERA_POINT_STATE_ENV
        return {
            "cameras": ",".join(hcps.CAMERAS),
            "compute_mano": bool(hcps.COMPUTE_MANO_POINT_STATE),
            "compute_robot": bool(hcps.COMPUTE_ROBOT_POINT_STATE),
            "renderer": "egl" if cfg.SIM.BULLET.USE_EGL else "tiny",
        }
    except AttributeError:
        return {}


def collect_iteration(sim, runner, scenes, out_path, *, rng,
                      beta: float, params: CollectParams, pin_table=None,
                      registry=None, iteration: int = 0,
                      progress_every: int = 5, results=None) -> dict:
    """Collect D_i: `len(scenes)` trajectories under the current mixed policy.

    `registry` (a GraspRegistry) verifies that a scene revisited in a later
    iteration still aims at the same grasp — the pin table enforces that, this
    checks it actually held.

    `results`, when given, is a pre-computed `[(episode | None, stats), ...]`
    aligned with `scenes` — what dagger/parallel.py's worker pool returns. The
    episodes were produced elsewhere but the aggregation, the HDF5 writing and
    the registry checks still happen HERE, in this one loop, so the parallel and
    serial paths cannot drift apart in what they count or what they store.
    `sim` and `runner` are then unused and may be None.
    """
    agg = {"episodes": 0, "steps": 0, "skipped": 0, "n_omg_fail": 0,
           "n_goal_switch": 0, "n_expert_steps": 0, "pinned": 0, "reached_standoff": 0,
           "reached_grasp": 0, "n_reach_steps": 0, "policy_closed": 0,
           "n_policy_close_cmds": 0, "n_dropped_tail": 0,
           "n_dart": 0, "n_dart_env_done": 0,
           "n_dart_reach": 0, "n_dart_reject": 0,
           "n_close_labels": 0, "n_approach_labels": 0, "sum_label_pos": 0.0,
           "n_tiny_labels": 0, "n_revisits": 0, "n_grasp_mismatch": 0,
           "max_grasp_drift": 0.0,
           "min_pos": [], "min_rot": [], "close_steps": [], "reasons": {},
           # params.outcome_check only. `outcomes` is keyed on the EVALUATOR's
           # taxonomy and counted over KEPT episodes only, so its counts stack to
           # `episodes` and can be read as fractions; `reasons` above is the
           # terminal cause and counts skipped episodes too.
           "outcomes": {}, "success": 0,
           # DART: pooled sum(d d^T) and step count over the whole iteration.
           # `sigma_hat` = sum / n is paper Eq. 3; the driver rescales it (Eq. 4)
           # and hands it back as the NEXT iteration's dart_sigma.
           "dart_dd_sum": np.zeros((6, 6), dtype=np.float64),
           "dart_dd_n": 0, "n_dart_noise": 0}

    writer = DaggerHDF5Writer(out_path, attrs={
        "iteration": int(iteration),
        "beta": float(beta),
        "max_steps": int(params.max_steps),
        "close_pos_thresh": float(params.close_pos_thresh),
        "close_rot_thresh": float(params.close_rot_thresh),
        "ee_step": float(params.ee_step),
        "reach_tail": int(params.reach_tail),
        "reach_commit_dist": float(params.reach_commit_dist),
        "stop_on_close_label": bool(params.stop_on_close_label),
        "stop_on_policy_close": bool(params.stop_on_policy_close),
        # Recorded even when off, so a DART-free shard is provably DART-free
        # rather than merely predating the feature.
        "dart_ratio": float(params.dart_ratio),
        "dart_max_dist": float(params.dart_max_dist),
        "dart_pos_mag": float(params.dart_pos_mag),
        "dart_rot_mag": float(params.dart_rot_mag),
        "dart_reach_ratio": float(params.dart_reach_ratio),
        "dart_reach_pos_mag": float(params.dart_reach_pos_mag),
        "dart_reach_rot_mag": float(params.dart_reach_rot_mag),
        "dart_reach_clearance": float(params.dart_reach_clearance),
        # Which pin table the labels aim at. The base collector has always written
        # this; the DAgger shards did not, which meant an aggregate could not say
        # what its `scene_idx` values were relative to. BCDataset's auxiliary
        # goal-grasp target (run 13) resolves scene_idx through exactly this file,
        # so recording it makes each shard self-describing and removes the chance
        # of pairing a train table with val indices.
        "grasp_pin_table": str(getattr(pin_table, "path", "") or ""),
        "scenes": np.asarray(scenes, dtype=np.int32),
        # Which OBSERVATION produced these clouds. The base collector records the
        # same keys; this is the other half of the aggregate, and the aggregate is
        # exactly where a camera mismatch would do its damage. Two collections
        # with different cameras are indistinguishable from every other attr —
        # same pc_channels, num_pts and pc_format — so without this the only
        # evidence is a filename, and a mixed aggregate trains silently and is
        # wrong only at deployment.
        **_observation_attrs(sim),
    })
    try:
        for i, scene in enumerate(scenes):
            if results is not None:
                episode, st = results[i]
            else:
                episode, st = collect_dagger_episode(
                    sim, runner, int(scene), rng=rng, beta=beta, params=params,
                    pin_table=pin_table)

            agg["n_omg_fail"] += st.get("n_omg_fail", 0)
            # Pooled OUTSIDE the `episode is None` guard: a skipped episode still
            # produced learner-supervisor comparisons before it was dropped, and
            # the estimator wants every one of them. (`.get` because the early
            # returns for OMG_FAIL_STEP0 / NO_LABELS carry no estimator fields.)
            if st.get("dart_dd_n"):
                agg["dart_dd_sum"] += st["dart_dd_sum"]
                agg["dart_dd_n"] += int(st["dart_dd_n"])
            agg["n_dart_noise"] += int(st.get("n_dart_noise", 0))
            agg["reasons"][st["reason"]] = agg["reasons"].get(st["reason"], 0) + 1
            if episode is None:
                agg["skipped"] += 1
            else:
                writer.append(episode)
                agg["episodes"] += 1
                agg["steps"] += st["steps"]
                if st.get("outcome"):
                    agg["outcomes"][st["outcome"]] = (
                        agg["outcomes"].get(st["outcome"], 0) + 1)
                agg["success"] += int(st.get("success", 0))
                agg["n_goal_switch"] += st["n_goal_switch"]
                agg["n_expert_steps"] += st["n_expert_steps"]
                agg["pinned"] += st["pinned"]
                agg["reached_standoff"] += st["reached_standoff"]
                agg["reached_grasp"] += st["reached_grasp"]
                agg["n_reach_steps"] += st["n_reach_steps"]
                agg["policy_closed"] += st["policy_closed"]
                agg["n_policy_close_cmds"] += st["n_policy_close_cmds"]
                agg["n_dropped_tail"] += st["n_dropped_tail"]
                agg["n_dart"] += st["n_dart"]
                agg["n_dart_env_done"] += st["n_dart_env_done"]
                agg["n_dart_reach"] += st.get("n_dart_reach", 0)
                agg["n_dart_reject"] += st.get("n_dart_reject", 0)
                agg["n_close_labels"] += st["n_close_labels"]
                agg["n_approach_labels"] += st["n_approach_labels"]
                agg["sum_label_pos"] += st["sum_label_pos"]
                agg["n_tiny_labels"] += st["n_tiny_labels"]
                if np.isfinite(st["min_pos"]):
                    agg["min_pos"].append(st["min_pos"])
                if np.isfinite(st["min_rot"]):
                    agg["min_rot"].append(st["min_rot"])
                if st["policy_close_step"] >= 0:
                    agg["close_steps"].append(st["policy_close_step"])
                if registry is not None:
                    chk = registry.check(int(scene), st.get("grasp_pose"),
                                         iteration=iteration)
                    if chk["seen"]:
                        agg["n_revisits"] += 1
                        agg["n_grasp_mismatch"] += int(chk["mismatch"])
                        if np.isfinite(chk["drift"]):
                            agg["max_grasp_drift"] = max(agg["max_grasp_drift"],
                                                         chk["drift"])

            if (i + 1) % progress_every == 0 or i == len(scenes) - 1:
                print(f"    [{i+1:3d}/{len(scenes)}] episodes={agg['episodes']} "
                      f"steps={agg['steps']} standoff={agg['reached_standoff']} "
                      f"grasp={agg['reached_grasp']} skipped={agg['skipped']}")
    finally:
        writer.close()
        if registry is not None:
            registry.save()

    def _mean(key):
        vals = agg.pop(key)
        return float(np.mean(vals)) if vals else float("nan")

    agg["mean_min_pos"] = _mean("min_pos")
    agg["mean_min_rot"] = _mean("min_rot")
    agg["mean_policy_close_step"] = _mean("close_steps")
    # Mean approach-label displacement. The demonstrations sit at ~ee_step; a
    # collapse towards 0 means the expert has stopped producing usable labels.
    agg["mean_label_pos"] = (agg["sum_label_pos"] / agg["n_approach_labels"]
                             if agg["n_approach_labels"] else float("nan"))
    agg.pop("sum_label_pos")
    agg["path"] = str(out_path)
    # DART paper Eq. 3: Sigma_hat = (1/T) sum_t (pi_theta - pi*)(pi_theta - pi*)^T.
    # Left UNSCALED here — Eq. 4's trace target belongs to the driver, which is
    # the only place that knows tr(Sigma_hat_1) from the first iteration.
    agg["dart_sigma_hat"] = (agg["dart_dd_sum"] / agg["dart_dd_n"]
                             if agg["dart_dd_n"] else None)
    # Drop the raw accumulator: the driver only ever reads `dart_sigma_hat`, and
    # this dict is embedded verbatim in state.json, where 36 floats of an
    # un-normalised intermediate are pure noise. (`dart_sigma_hat` itself stays —
    # the covariance's evolution across iterations is worth being able to read
    # back — and save_state's `default=` handles the array.)
    agg.pop("dart_dd_sum", None)
    agg["dart_sigma_trace"] = (float(np.trace(agg["dart_sigma_hat"]))
                               if agg["dart_sigma_hat"] is not None else float("nan"))
    return agg
