"""
The HandoverSim benchmark protocol — Chao et al.'s success criterion, run on our
policies so the numbers are comparable with Christen et al. (CVPR 2023).

WHY THIS IS SEPARATE FROM `dagger/evaluator.py` AND `regrasp/evaluator.py`

Those score the PHASE-3 criterion, `stable_grasp`: commit the close, hold it,
and require the object secured — released by the human, not dropped, no human
contact. The benchmark asks for all of that AND THEN SOME:

    handover-sim `HandoverStatusWrapper._check_status()`, EpisodeStatus.SUCCESS
      1. ycb.released                      -- the human let go
      2. both finger links contact the object
      3. no drop  (object on table / another YCB / below table height, unheld)
      4. no human contact above CONTACT_FORCE_THRESH (0.0 -- ANY contact)
      5. the hand link within GOAL_RADIUS (0.15 m) of GOAL_CENTER
      6. ...held there for SUCCESS_TIME_THRESH (0.1 s)
      7. ...all within MAX_EPISODE_TIME (13.0 s = 13000 sim steps = 86.7 policy
         steps at TIME_ACTION_REPEAT 0.15)

1-4 are `stable_grasp`. **5-7 are the carry to the goal, and we have never done
them** -- `regrasp/env_setup.py` says so in as many words ("no scripted
grasp-and-back here ... nothing ever drives the retreat"), and
`dagger/evaluator.py` notes that `EpisodeStatus.SUCCESS` "can never fire and
would score every episode 0".

So benchmark-SUCCESS is a STRICT SUBSET of our success: every episode the
benchmark counts, we would too, and not conversely. Our published rates are
upper bounds on the benchmark number, and the gap is whatever the retreat costs.

THE FAILURE TAXONOMY IS COARSER, AND THAT IS NOT A LOSS OF INFORMATION SO MUCH
AS A DIFFERENT QUESTION. The benchmark has three buckets; we have five. Ours map
in as:

    DROP            -> FAILURE_OBJECT_DROP
    HUMAN_CONTACT   -> FAILURE_HUMAN_CONTACT
    TIMEOUT         -> FAILURE_TIMEOUT
    NO_RELEASE      -> FAILURE_TIMEOUT      (the human never let go; the clock runs out)
    GRASP_MISS      -> FAILURE_OBJECT_DROP  (see below)

GRASP_MISS IS A DROP THAT HAS NOT LANDED YET. The evaluator assigns it only when
`ycb.released` is True (the human let go), `grasped_active()` is False (not in
the fingers) and `_dropped` is False — and `_dropped` is set only once the object
touches the table, another YCB, or falls below table height. An object that has
been released, is not held, and has not yet hit anything at the end of a 3-step
hold is in the air. Under the benchmark's 13 s clock it lands, and the wrapper
calls that FAILURE_OBJECT_DROP. (A gripper that closes on nothing while the
human still holds the object is NO_RELEASE, not GRASP_MISS — `released` gates
the two apart.)

So the benchmark's `timeout` bucket absorbs our NO_RELEASE and is much broader
than our `ff_timeout`; its `drop` absorbs our GRASP_MISS, which is small.

THE SECOND GAP, AND IT IS THE LARGER ONE: `YCB_MANO_START_FRAME`

Every config in this repo sets `YCB_MANO_START_FRAME: last`, which freezes the
human at the FINAL frame of the DexYCB sequence -- the hand already fully
extended, holding the object out, static from t=0. The benchmark's default is
`first`, where the human's reach plays out over the episode and the object is a
MOVING target the robot has to intercept. That is why every benchmark policy
waits `time_wait = 3.0 s` before acting: under `first` there is nothing to grasp
yet.

So a faithful comparison with Christen et al. needs BOTH changes, and the second
one evaluates our policies on a distribution they have never seen. `--start-frame`
makes the choice explicit rather than silent:

    first  (DEFAULT)  the benchmark as published. Comparable with Christen et al.
                      Our policies were not trained for it.
    last              our training distribution, with the benchmark's SUCCESS
                      criterion. NOT comparable with anyone else's number; it
                      isolates exactly one thing -- what the carry-to-goal
                      requirement costs a policy that can already grasp.

Report which one, always. They are different experiments.

OUTPUT IS THE OFFICIAL FORMAT, DELIBERATELY. `run_episode` records what
`handover.benchmark_runner.BenchmarkRunner._run_scene` records -- `action`,
`elapsed_time`, `elapsed_frame`, `result` -- and `write_result` lays it out as
`<res_dir>/{:03d}.npz` beside a `config.yaml`, which is exactly what
`handover.benchmark_evaluator.evaluate()` consumes. The metrics are then computed
by THEIR code, not ours. Given how much of this project's history is silent
metric drift between two implementations of the same rate, reimplementing the
one number we want to compare against would be the wrong place to save an hour.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pybullet


# `handover.benchmark_wrapper.EpisodeStatus`, duplicated as plain ints so this
# module imports on a login node without gym or the benchmark package. Asserted
# against the real enum in `check_status_enum` before any run that matters.
SUCCESS = 1
FAILURE_HUMAN_CONTACT = 2
FAILURE_OBJECT_DROP = 4
FAILURE_TIMEOUT = 8

# `handover-sim/examples/demo_trajectory.py`, the values every published
# benchmark policy is built from. `time_close_gripper` is GA-DDPG's 0.5 s rather
# than the demo trajectory's 0.2 s, because GA-DDPG is the CVPR-2023 baseline
# this is meant to be read against (`run_benchmark_gaddpg_hold.py`).
TIME_WAIT = 3.0
TIME_CLOSE_GRIPPER = 0.5
BACK_STEP_SIZE = 0.03


def check_status_enum() -> None:
    """Fail loudly if the benchmark's status bits ever move under us.

    The four constants above are copied rather than imported so the module stays
    importable without gym; that copy is a latent bug the day upstream renumbers
    them, and every rate in the output would shift silently. One import and four
    comparisons remove that.
    """
    from handover.benchmark_wrapper import EpisodeStatus as E

    for name, ours in (("SUCCESS", SUCCESS),
                       ("FAILURE_HUMAN_CONTACT", FAILURE_HUMAN_CONTACT),
                       ("FAILURE_OBJECT_DROP", FAILURE_OBJECT_DROP),
                       ("FAILURE_TIMEOUT", FAILURE_TIMEOUT)):
        theirs = int(getattr(E, name))
        if theirs != ours:
            raise SystemExit(
                f"[benchmark] EpisodeStatus.{name} is {theirs} upstream but "
                f"{ours} in handover_sim2real/benchmark.py. Every rate this "
                f"module reports would be mis-bucketed. Update the constants.")


def retreat_plan(obs, cfg, step_size: float = BACK_STEP_SIZE) -> list:
    """Joint configurations carrying the EE from where it is to GOAL_CENTER.

    A TRANSCRIPTION of `handover-sim/examples/demo_benchmark_wrapper.py:76-95`,
    which is the retreat every published benchmark policy uses -- GA-DDPG's two
    entries inherit it unchanged through `SimplePolicy.forward`. Straight line in
    CARTESIAN space, `step_size` per waypoint, position-only IK, fingers held
    shut. Deliberately not improved on: the retreat is part of the protocol, and
    a better one would make our success rate incomparable with the number it is
    being placed beside.

    Position-only IK (no target orientation) is theirs too, and it matters --
    the wrist is free to rotate on the way back, which is what keeps the goal
    reachable from the far side of the workspace.
    """
    pos = obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:3].numpy()
    goal = np.asarray(cfg.BENCHMARK.GOAL_CENTER, dtype=np.float64)
    dpos = goal - pos
    dist = float(np.linalg.norm(dpos))
    if dist < 1e-9:
        return []
    step = dpos / dist * float(step_size)
    n = int(np.ceil(dist / float(step_size)))
    out, p = [], pos.astype(np.float64).copy()
    for _ in range(n):
        p += step
        conf = np.array(pybullet.calculateInverseKinematics(
            obs["panda_body"].contact_id[0], obs["panda_link_ind_hand"] - 1, p))
        conf[7:9] = 0.0                       # fingers shut for the whole carry
        out.append(conf)
    return out


def run_episode(sim, scene_idx: int, act_fn, *, max_policy_steps: int = 0,
                time_wait: float = TIME_WAIT,
                time_close_gripper: float = TIME_CLOSE_GRIPPER,
                back_step_size: float = BACK_STEP_SIZE,
                on_close=None, verbose: bool = False) -> dict:
    """One scene under the benchmark protocol. Returns the official result dict.

    `act_fn(obs, pc, rs, step) -> action[7]` is the policy, in the 7-D form every
    evaluator here already speaks: 6-D EE delta plus a gripper bit where
    `action[6] < 0.5` means CLOSE. The first step at which it commits ends the
    approach and begins the carry; `on_close(obs, step)` fires there, for a
    caller that wants the at-commit geometry.

    THE EPISODE ENDS WHEN THE BENCHMARK SAYS IT DOES, not when we decide it has.
    `info["status"] != 0` is the only stopping condition, exactly as
    `BenchmarkRunner._run_scene` has it -- so SUCCESS, the three failures and the
    13 s timeout are all decided by `HandoverStatusWrapper`. `max_policy_steps`
    is a safety valve for the APPROACH only (0 = the benchmark's own clock), and
    it is not a scoring threshold: hitting it stops commanding new approach
    actions and lets the clock run out into FAILURE_TIMEOUT, which is what the
    benchmark would have recorded anyway.

    PHASES, in order, mirroring `SimplePolicy.forward`:
      wait      hold the start configuration for `time_wait`. Under
                `YCB_MANO_START_FRAME: first` the human is still reaching and
                there is nothing to grasp; under `last` it is dead time and
                `--time-wait 0` is the right setting.
      approach  the policy, re-queried every `steps_action_repeat` sim steps.
      close     hold the commit pose with the fingers shut for
                `time_close_gripper`, so the grasp settles before the arm moves.
      carry     replay `retreat_plan`, one waypoint per `steps_action_repeat`.
      hold      once the plan is exhausted, keep commanding its last
                configuration. The success test needs the hand to DWELL inside
                the goal ball for 0.1 s; stopping at the last waypoint would end
                the episode at the moment of arrival and score 0 for a handover
                that had in fact completed.
    """
    env = sim.env
    cfg = sim.cfg
    rep = int(sim.steps_action_repeat)
    steps_wait = int(float(time_wait) / cfg.SIM.TIME_STEP)
    steps_close = int(float(time_close_gripper) / cfg.SIM.TIME_STEP)

    obs = env.reset(idx=int(scene_idx))
    sim.point_listener.reset()

    actions, elapsed = [], []
    status, info = 0, {}
    close_frame = None
    close_action = None
    back = None
    n_policy = 0
    committed = False

    start_conf = np.asarray(cfg.ENV.PANDA_INITIAL_POSITION, dtype=np.float64)
    target_jp = start_conf.copy()
    # `use_prev_act` is off in every Regrasp and Phase-4 config, so this stays
    # zero in practice — carried anyway so the robot-state vector this module
    # builds is byte-identical to the one the evaluators build, rather than
    # nearly so.
    prev_act6d = np.zeros(6, dtype=np.float32)

    while True:
        frame = int(env.frame)

        if frame < steps_wait:
            target_jp = start_conf.copy()

        elif not committed:
            # Re-query the policy on the action-repeat boundary only, so the
            # command rate matches every other evaluator in this repo (and
            # GA-DDPG's own benchmark policy, which gates on the same modulus).
            if (frame - steps_wait) % rep == 0:
                t0 = time.perf_counter()
                pc, rs = _observe(sim, obs, prev_act6d)
                action = act_fn(obs, pc, rs, n_policy)
                elapsed.append(time.perf_counter() - t0)
                if action is not None:
                    prev_act6d = np.asarray(action[:6], dtype=np.float32)
                n_policy += 1
                if action is None or float(action[6]) < 0.5:
                    committed = True
                    close_frame = frame
                    if on_close is not None:
                        on_close(obs, n_policy - 1)
                    close_action = np.asarray(target_jp, dtype=np.float64).copy()
                    close_action[7:9] = 0.0
                    target_jp = close_action
                else:
                    target_jp = _target_joint(action, obs)
                    if max_policy_steps and n_policy >= int(max_policy_steps):
                        # Out of approach budget without a commit. Stop issuing
                        # new actions and let the benchmark clock expire — the
                        # episode is a FAILURE_TIMEOUT either way, and forcing a
                        # close here would manufacture a grasp attempt the
                        # policy never made.
                        committed = True
                        close_frame = frame
                        close_action = np.asarray(target_jp,
                                                  dtype=np.float64).copy()

        elif frame < (close_frame or 0) + steps_close:
            target_jp = close_action                      # settle the grasp

        else:
            if back is None:
                back = retreat_plan(obs, cfg, back_step_size)
                if verbose:
                    print(f"      retreat: {len(back)} waypoints")
            if back:
                i = (frame - close_frame - steps_close) // rep
                target_jp = back[min(i, len(back) - 1)]
            # `back == []` leaves target_jp at the closed commit pose, which is
            # the right degenerate answer: the EE is already at the goal.

        obs, _, _, info = env.step(target_jp)
        actions.append(np.asarray(target_jp, dtype=np.float64).copy())
        status = int(info.get("status", 0))
        if status != 0:
            break

    return {
        "action": np.asarray(actions, dtype=np.float64),
        "elapsed_time": np.asarray(elapsed, dtype=np.float64),
        "elapsed_frame": int(env.frame),
        "result": int(status),
        # Ours, not the benchmark's — ignored by `evaluate()`, kept because a
        # bare status cannot say whether the policy ever committed, and "never
        # closed" and "closed and then lost it on the way back" are the same
        # FAILURE_TIMEOUT to the benchmark and completely different problems.
        "committed": int(bool(close_frame is not None)),
        "close_frame": int(close_frame if close_frame is not None else -1),
        "n_policy_steps": int(n_policy),
        "n_retreat_waypoints": int(len(back) if back else 0),
    }


def _observe(sim, obs, prev_act6d):
    """`(point cloud, robot state)` exactly as the evaluators build them.

    Imported from `examples/collect_bc_dataset.py` rather than reimplemented:
    these two functions define the policy's entire input contract — the 1024x5
    EE-frame cloud and the 32-D robot state the real robot hard-asserts — and a
    second version of either is a sim2real gap that would show up as a bad
    benchmark number rather than as an error.
    """
    import sys
    from pathlib import Path
    ex = str(Path(__file__).resolve().parents[1] / "examples")
    if ex not in sys.path:
        sys.path.insert(0, ex)
    from collect_bc_dataset import _point_cloud, _robot_state
    pc = _point_cloud(obs, sim.point_listener, sim.panda_base_inv_tf)
    rs = _robot_state(obs, prev_act6d)
    return pc, rs


def _target_joint(action, obs):
    """`action_to_target_joint` without importing `examples/` at module scope."""
    import sys
    from pathlib import Path
    ex = str(Path(__file__).resolve().parents[1] / "examples")
    if ex not in sys.path:
        sys.path.insert(0, ex)
    from rollout_bc_policy import action_to_target_joint
    return action_to_target_joint(action, obs)


def write_result(res_dir: str, idx: int, result: dict) -> str:
    """`<res_dir>/{:03d}.npz`, the layout `benchmark_evaluator.evaluate` reads."""
    os.makedirs(res_dir, exist_ok=True)
    path = os.path.join(res_dir, "{:03d}.npz".format(int(idx)))
    np.savez(path, **{k: v for k, v in result.items()})
    return path


def status_name(status: int) -> str:
    """A readable bucket for one status. SUCCESS is exclusive by construction;
    the failures are a bitmask and CAN co-occur, so they are joined rather than
    collapsed to the first — a drop caused by a collision with the giver is both
    things and reporting only one of them loses the cause."""
    s = int(status)
    if s == SUCCESS:
        return "SUCCESS"
    parts = [n for bit, n in ((FAILURE_HUMAN_CONTACT, "HUMAN_CONTACT"),
                              (FAILURE_OBJECT_DROP, "OBJECT_DROP"),
                              (FAILURE_TIMEOUT, "TIMEOUT")) if s & bit]
    return "|".join(parts) if parts else f"UNKNOWN({s})"


def summarise(results: dict, time_step: float = 0.001) -> dict:
    """`benchmark_evaluator.evaluate`'s arithmetic, for the live console only.

    THE REPORTED NUMBER IS STILL THEIRS. This exists so a long sweep can print a
    running rate without waiting for the official pass, and because `evaluate()`
    refuses to run until every scene of the split has a file. It is the same
    arithmetic, written once, and the script asserts the two agree at the end —
    if they ever diverge, theirs is right.
    """
    if not results:
        return {}
    r = np.array([v["result"] for v in results.values()], dtype=np.int64)
    frames = np.array([v["elapsed_frame"] for v in results.values()],
                      dtype=np.float64)
    plan = np.array([float(np.sum(v["elapsed_time"])) for v in results.values()])
    succ = r == SUCCESS
    out = {
        "num_scenes": int(len(r)),
        "num_success": int(succ.sum()),
        "success_rate": float(succ.mean()),
        "failure_human_contact": float((r & FAILURE_HUMAN_CONTACT > 0).mean()),
        "failure_object_drop": float((r & FAILURE_OBJECT_DROP > 0).mean()),
        "failure_timeout": float((r & FAILURE_TIMEOUT > 0).mean()),
    }
    out["time_exec"] = (float(frames[succ].mean() * time_step) if succ.any()
                        else float("nan"))
    out["time_plan"] = float(plan[succ].mean()) if succ.any() else float("nan")
    out["time_total"] = out["time_exec"] + out["time_plan"]
    # OURS, and not in the official table: of the episodes that failed, how many
    # never committed a close at all. The benchmark calls "never grasped" and
    # "grasped and lost it on the carry" both FAILURE_TIMEOUT, and they are
    # opposite problems.
    comm = np.array([v.get("committed", 1) for v in results.values()])
    out["commit_rate"] = float(comm.mean())
    out["fail_never_committed"] = (float((comm[~succ] == 0).mean())
                                   if (~succ).any() else float("nan"))
    return out
