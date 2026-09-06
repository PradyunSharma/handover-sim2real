#!/usr/bin/env python3
"""Measure how the arm MOVES during a policy step, without a robot.

    python test_step_motion.py            # the comparison + the regressions
    python test_step_motion.py --sweep    # also sweep the plant parameters

The complaint this exists for is not accuracy, it is smoothness: "at each step
the arm moves a certain distance and then it does some fine motions". So the
thing to measure is the SHAPE of the motion, and the numbers that capture it are
how long the arm spends stopped in the MIDDLE of a step and how often it
backtracks. Pause count alone is too weak to assert on — shortening a pause from
300 ms to 40 ms leaves it unchanged while removing everything that made it
visible — so all three are measured and the dead time is the one that decides.

Everything below drives the real settle() and move_to() from my_policy_runner
against a simulated arm and a fake clock, so what is measured is the shipped
control code and not a paraphrase of it. The plant is the honest part of the
uncertainty: the true behaviour of the impedance controller is NOT identified
(see the MAX_COMMAND_LEAD_M comment in the runner — an affine under-travel and
stiction fit the same two measurements), so the model carries both mechanisms
and --sweep checks the conclusion survives across the range rather than holding
at one lucky setting. That sweep is not decoration: it is what rejected a lead
cap that looked obviously correct, worked on one plant and deadlocked another.
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import my_policy_runner as m  # noqa: E402


# -----------------------------------------------------------------------------
# Simulated arm
# -----------------------------------------------------------------------------
class Plant:
    """A Cartesian impedance arm that does not reach what it is told.

    One commanded equilibrium produces ONE exponential approach to a point that
    falls short of it, and then nothing further — the arm sits there until a new
    equilibrium arrives. That latching is the whole point: it is what makes a
    single command insufficient and correction necessary, and it is why the
    shortfall cannot be modelled as a servo that would eventually creep in on
    its own.

    Two mechanisms, because the real one is unknown:
      gain     fractional under-travel, A = p + gain*(E - p). Dominates large
               moves. Consistent with the two-point measurement in the runner.
      stall_m  a dead band the arm cannot push through, A stops stall_m short of
               E. Dominates small moves, and unlike gain alone it does not imply
               the absurd conclusion that the arm can barely move in x.
    """

    def __init__(self, p0, *, gain=0.75, stall_m=0.012, tau_s=0.12,
                 gain_jitter=0.0, noise_m=2e-6, seed=0):
        self.p = np.asarray(p0, dtype=np.float64).copy()
        self.goal = self.p.copy()
        self.gain = float(gain)
        self.stall = float(stall_m)
        self.tau = float(tau_s)
        self.jitter = float(gain_jitter)
        self.noise = float(noise_m)
        self.rng = np.random.default_rng(seed)
        self.commands = 0

    def command(self, E: np.ndarray) -> None:
        self.commands += 1
        err = np.asarray(E, dtype=np.float64) - self.p
        n = float(np.linalg.norm(err))
        if n < 1e-12:
            self.goal = self.p.copy()
            return
        # A DIFFERENT gain for every command, not one per plant.
        #
        # This is the single most important thing the model got wrong, and the
        # robot said so directly: four consecutive steps measured gains of 0.51,
        # 0.44, 0.85, 0.92. A fixed-gain plant predicted 1.2 commands per step
        # while the arm needed 3 to 6, and the gap is entirely this. It is not
        # noise to be averaged away either — the direction of travel changes
        # every step, and friction and the nullspace configuration change with
        # it, so there is a genuinely different plant each time. Anything that
        # depends on the last move predicting the next one has to survive that.
        g = self.gain
        if self.jitter > 0.0:
            g = float(np.clip(self.rng.normal(self.gain, self.jitter), 0.05, 1.0))
        travel = max(0.0, g * n - self.stall)
        self.goal = self.p + err / n * travel

    def advance(self, dt: float) -> None:
        self.p += (self.goal - self.p) * (1.0 - np.exp(-dt / self.tau))

    def measured(self) -> np.ndarray:
        return self.p + self.rng.normal(0.0, self.noise, 3)


# -----------------------------------------------------------------------------
# Fake clock, fake ROS
# -----------------------------------------------------------------------------
SUBSTEP_S = 0.002


class Harness:
    """Replaces my_policy_runner's `time` module and its publisher.

    settle() reaches the outside world through exactly three things — time.time,
    time.sleep, and pub.publish — so intercepting those three runs the real
    control loop at arbitrary speed against the plant above. current_msg is
    written on every substep, the way /cartesian_pose would.
    """

    def __init__(self, plant: Plant, R: np.ndarray | None = None):
        self.plant = plant
        self.t = 0.0
        self.R = np.eye(3) if R is None else R
        self.trace: list[tuple[float, np.ndarray]] = []
        self._publish_stamps: list[float] = []
        self._write_msg()

    # --- time module surface -------------------------------------------------
    def time(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        remaining = float(dt)
        while remaining > 1e-12:
            h = min(SUBSTEP_S, remaining)
            self.plant.advance(h)
            self.t += h
            remaining -= h
            self.trace.append((self.t, self.plant.p.copy()))
        self._write_msg()

    # --- publisher surface ---------------------------------------------------
    def publish(self, msg) -> None:
        pos = msg["pose"]["position"]
        self.plant.command(np.array([pos["x"], pos["y"], pos["z"]]))
        self._publish_stamps.append(self.t)

    # --- /cartesian_pose -----------------------------------------------------
    def _write_msg(self) -> None:
        q = _quat_from_matrix(self.R)
        p = self.plant.measured()
        m.current_msg = {
            "header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0},
                       "frame_id": "panda_link0"},
            "pose": {"position": {"x": float(p[0]), "y": float(p[1]),
                                  "z": float(p[2])},
                     "orientation": {"x": float(q[0]), "y": float(q[1]),
                                     "z": float(q[2]), "w": float(q[3])}},
        }

    # --- metrics -------------------------------------------------------------
    def motion_profile(self, still_speed=0.0025) -> tuple[int, float]:
        """(bursts of motion, seconds spent stopped BETWEEN them).

        still_speed is SETTLE_STILL_POS_M / SETTLE_POLL_S — the same 2.5 mm/s
        the runner itself calls "stopped", so this counts exactly the pauses a
        person would see, not an arbitrary smaller flicker.

        The count alone is a weak measure and easy to flatter: shortening a pause
        from 300 ms to 5 ms leaves the count unchanged while removing everything
        that made it visible. The dead time between bursts is the number that
        cannot be gamed, so both are returned and both are asserted on. Time
        before the first burst and after the last is excluded — that is command
        latency and the final hold, neither of which is stop-and-go.
        """
        segs, moving, dead = 0, False, 0.0
        started = False
        pause = 0.0
        for (t0, p0), (t1, p1) in zip(self.trace, self.trace[1:]):
            dt = t1 - t0
            v = float(np.linalg.norm(p1 - p0)) / max(dt, 1e-9)
            if v >= still_speed:
                if not moving:
                    segs += 1
                    if started:
                        dead += pause
                    started = True
                    pause = 0.0
                moving = True
            else:
                moving = False
                if started:
                    pause += dt
        return segs, dead

    def reversals(self, direction, still_speed=0.0025) -> int:
        """Bursts of motion that travel BACKWARDS along `direction`.

        The third symptom, and the most visible one: an arm that overshoots and
        comes back has to reverse, and a reversal reads as a twitch however
        short it is. Any scheme that leads past the target trades pauses for
        these, so this is measured separately rather than folded into the pause
        count, where the trade would be invisible.
        """
        u = np.asarray(direction, dtype=np.float64)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        n, back = 0, False
        for (t0, p0), (t1, p1) in zip(self.trace, self.trace[1:]):
            step = p1 - p0
            v = float(np.linalg.norm(step)) / max(t1 - t0, 1e-9)
            if v < still_speed:
                back = False
                continue
            if float(step @ u) < 0.0:
                if not back:
                    n += 1
                back = True
            else:
                back = False
        return n

    def reset_trace(self) -> None:
        self.trace.clear()
        self._publish_stamps.clear()


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_quat()


def _pose(p, R=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=np.float64)
    return T



# -----------------------------------------------------------------------------
# Driving one episode
# -----------------------------------------------------------------------------
def run_steps(*, creep: bool, n_steps=8, step_m=0.039, plant_kw=None,
              seed=0) -> dict:
    """Execute n_steps policy steps and report what the arm did.

    The step targets are taken RELATIVE TO THE MEASURED POSE, exactly as the
    runner builds them (T_base_hand @ unpack_action(delta)), so a step that
    under-travels shortens the episode's reach rather than accumulating a
    position error — which is the property that makes under-travel survivable at
    all, and would be lost by targeting absolute waypoints here.
    """
    rng = np.random.default_rng(seed)
    plant = Plant([0.45, 0.0, 0.50], seed=seed, **(plant_kw or {}))
    h = Harness(plant)

    real_time, m.time = m.time, h
    try:
        droop = m.DroopCompensator(enabled=True)
        seq = 0
        per_step = []
        for _ in range(n_steps):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            target = _pose(plant.p + d * step_m)

            h.reset_trace()
            cmds0 = plant.commands
            t0 = h.t
            seq, dp, dr, passes, _cmds = m.move_to(
                h, target, seq, m.SETTLE_TIMEOUT_S, droop,
                m.STEP_CONVERGE_PASSES, m.STEP_CONVERGE_TOL_M, creep=creep)
            bursts, dead = h.motion_profile()
            per_step.append({
                "stops": max(bursts - 1, 0),
                "dead": dead,
                "back": h.reversals(d),
                "err_mm": dp * 1000.0,
                "secs": h.t - t0,
                "commands": plant.commands - cmds0,
                "passes": passes,
            })
    finally:
        m.time = real_time
        m.current_msg = None

    return {
        "stops": float(np.mean([s["stops"] for s in per_step])),
        "max_stops": max(s["stops"] for s in per_step),
        "dead": float(np.mean([s["dead"] for s in per_step])),
        "max_dead": max(s["dead"] for s in per_step),
        "back": float(np.mean([s["back"] for s in per_step])),
        "err_mm": float(np.mean([s["err_mm"] for s in per_step])),
        "worst_mm": max(s["err_mm"] for s in per_step),
        "secs": float(np.mean([s["secs"] for s in per_step])),
        "commands": float(np.mean([s["commands"] for s in per_step])),
        "converged": sum(s["err_mm"] < m.STEP_CONVERGE_TOL_M * 1000
                         for s in per_step),
        "n": len(per_step),
    }


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
def _comparison() -> None:
    print("\n-- stop-and-go per policy step ------------------------------")
    old = run_steps(creep=False)
    new = run_steps(creep=True)

    for name, r in (("multi-pass", old), ("creep", new)):
        print(f"  {name:11s} pauses/step {r['stops']:.2f} (worst {r['max_stops']})"
              f"  stopped {r['dead']*1000:4.0f} ms mid-move"
              f"  reversals {r['back']:.2f}"
              f"   err {r['err_mm']:5.2f} mm (worst {r['worst_mm']:5.2f})"
              f"   {r['secs']:.2f} s   {r['commands']:.1f} cmds"
              f"   converged {r['converged']}/{r['n']}")

    # The bug, reproduced. If the old path does not visibly stop and restart in
    # this plant then the plant is wrong and nothing below means anything.
    assert old["stops"] >= 1.5, (
        f"multi-pass path only pauses {old['stops']:.2f}x per step here — the "
        "plant no longer reproduces the reported behaviour, so this comparison "
        "is vacuous")
    assert old["dead"] >= 0.4, (
        f"multi-pass path only sits still {old['dead']*1000:.0f} ms mid-step "
        "here; the reported symptom is not being reproduced")

    # The fix. Dead time is the assertion that matters, and the count is the
    # weaker of the two: shortening a pause from 300 ms to 40 ms leaves the
    # count unchanged while removing everything that made it visible, so a test
    # that only counted pauses could be passed by doing nothing useful.
    assert new["dead"] <= 0.15, (
        f"creep still sits still {new['dead']*1000:.0f} ms mid-step")
    assert new["dead"] < old["dead"] / 3.0, (
        f"creep dead time {new['dead']*1000:.0f} ms is not a real improvement "
        f"on {old['dead']*1000:.0f} ms")
    # ... and none of it may have been bought with accuracy, which was never the
    # complaint and is the obvious way to fake a smooth trace.
    assert new["converged"] == new["n"], (
        f"creep converged on only {new['converged']}/{new['n']} steps")
    assert new["worst_mm"] <= old["worst_mm"] + 0.5, (
        f"creep is less accurate: {new['worst_mm']:.2f} vs {old['worst_mm']:.2f} mm")
    # ... nor by trading pauses for backtracking.
    assert new["back"] <= 0.25, (
        f"creep reverses on {new['back']:.2f} steps out of 1 — it is overshooting "
        "and coming back, which is a twitch by another name")


def _travel_lead_learns() -> None:
    """The scalar lead should converge and cut the work per step.

    This is the part that replaces the droop VECTOR, so the thing worth checking
    is not just that it converges but that it converges to something useful: the
    later steps of an episode should need fewer commands than the first, because
    the first command already lands close.
    """
    print("\n-- travel lead ----------------------------------------------")
    rng = np.random.default_rng(3)
    plant = Plant([0.45, 0.0, 0.50], seed=3)
    h = Harness(plant)
    real_time, m.time = m.time, h
    try:
        droop = m.DroopCompensator(enabled=True)
        seq, cmds, leads = 0, [], []
        for _ in range(10):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            c0 = plant.commands
            res = m.settle(h, m.current_msg, _pose(plant.p + d * 0.039), seq,
                           m.SETTLE_TIMEOUT_S, droop,
                           tol_m=m.STEP_CONVERGE_TOL_M, creep=True)
            seq = res.next_seq
            cmds.append(plant.commands - c0)
            leads.append(droop.g)
    finally:
        m.time = real_time
        m.current_msg = None

    print(f"  commands per step : {cmds}")
    print(f"  measured gain     : {[round(x, 2) for x in leads]}")
    first, last = cmds[0], float(np.mean(cmds[-4:]))
    print(f"  first step {first} cmds -> last four average {last:.2f}")
    assert last < first, (
        f"the lead is not buying anything: {last:.2f} commands vs {first}")
    assert np.std(leads[-4:]) < 0.05, (
        f"gain has not settled: {[round(x, 2) for x in leads[-4:]]}")


def _lead_does_not_overshoot_a_short_correction() -> None:
    """A lead learned on full steps must not be dumped on a 2 mm refine.

    Reported from hardware: homing works the first time and hunts afterwards —
    "it looks like it is trying to find exact home position but it is not able
    to". The first home is the tell. It runs before any episode, so `s` is still
    0 and no lead is applied at all; every later home runs with `s` learned over
    ~30 mm policy steps, and `go_home`'s refine is a move of a few millimetres.

    `observe_move` already refuses to LEARN from moves under TRAVEL_LEAD_MIN_M
    ("a short move says nothing about the gain"). Nothing enforced the matching
    rule when applying it, so the refine got the full-step lead — and, being
    below that same floor, could not teach the estimator its way back out.

    Staged as the runner stages it: learn on full steps, then refine.
    """
    print("\n-- short-move lead ------------------------------------------")
    rng = np.random.default_rng(11)
    plant = Plant([0.45, 0.0, 0.50], seed=11)
    h = Harness(plant)
    real_time, m.time = m.time, h
    try:
        droop = m.DroopCompensator(enabled=True)
        seq = 0
        # An episode's worth of full-scale steps, which is what teaches `s`.
        for _ in range(8):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            res = m.settle(h, m.current_msg, _pose(plant.p + d * 0.030), seq,
                           m.SETTLE_TIMEOUT_S, droop,
                           tol_m=m.STEP_CONVERGE_TOL_M, creep=True)
            seq = res.next_seq
        learned_s, learned_scale = droop.s, droop.s_scale

        # Now the homing refine: a 3 mm correction.
        travel = np.array([0.003, 0.0, 0.0])
        lead = droop.lead_for(travel)
        full = droop.lead_for(travel / np.linalg.norm(travel) * learned_scale)
    finally:
        m.time = real_time
        m.current_msg = None

    print(f"  learned lead {learned_s*1000:.1f} mm over a "
          f"{learned_scale*1000:.1f} mm scale")
    print(f"  lead applied to a 3 mm move : {np.linalg.norm(lead)*1000:.2f} mm")
    print(f"  lead applied at full scale  : {np.linalg.norm(full)*1000:.2f} mm")

    assert learned_s > 1e-4, "staging failed: nothing was learned to over-apply"
    # The lead must never dominate the move it is leading — that is precisely
    # what makes the arm shoot past and hunt.
    assert np.linalg.norm(lead) <= np.linalg.norm(travel) * 1.01, (
        f"a 3 mm correction was given a {np.linalg.norm(lead)*1000:.1f} mm lead; "
        "this is the homing hunt")
    # And the full-scale behaviour must be untouched, or the fix has cost the
    # thing the lead exists for.
    assert abs(np.linalg.norm(full) - learned_s) < 1e-9, (
        "scaling changed the full-step lead, which is the case it was tuned on")


def _lead_cannot_dwarf_the_step() -> None:
    """A lead may exceed its step, but not by an unbounded multiple.

    `--control rate` measures the gain BEFORE the arm has finished moving —
    that is what "fixed rate" means — so the ratio it divides understates the
    gain and the lead it implies is several times too large. Observed on
    hardware minutes apart on the same arm: homing, which waits, reported gain
    0.85; the rate loop reported 0.10-0.30 and drove the lead to 171 mm on
    25 mm policy steps.

    The cap is deliberately AFFINE, and both halves are load-bearing in
    opposite directions:

      without the ratio  a 25 mm step is commanded ~196 mm and the arm shoots
                         past the object, which is what the approach oscillating
                         near the object looks like;
      without the offset a small step is starved — a multiplicative gain cannot
                         represent the ~17 mm standing offset, so an 8 mm
                         command moves the arm not at all. Measured when this
                         cap was first written as a pure ratio: an 8 mm step
                         fell 27.5 mm short of its goal over 60 ticks.
    """
    print("\n-- lead vs step ---------------------------------------------")
    d = m.DroopCompensator(enabled=True)
    worst = 0.0
    for s_mm, scale_mm, step_mm in ((171, 30, 25), (121, 30, 14),
                                    (300, 30, 25), (100, 30, 8)):
        d.s, d.s_scale = s_mm / 1000.0, scale_mm / 1000.0
        step = np.array([step_mm / 1000.0, 0.0, 0.0])
        lead = float(np.linalg.norm(d.lead_for(step)))
        bound = m.MAX_LEAD_TRAVEL_RATIO * step_mm / 1000.0 + m.LEAD_STALL_ALLOWANCE_M
        print(f"  learned {s_mm:3d} mm at scale {scale_mm} mm, step {step_mm:2d} mm"
              f" -> lead {lead*1000:5.1f} mm (bound {bound*1000:.1f})")
        assert lead <= bound + 1e-12, (
            f"a {step_mm} mm step was given a {lead*1000:.1f} mm lead")
        worst = max(worst, lead / (step_mm / 1000.0))
    print(f"  worst lead/step ratio: {worst:.2f}x")

    # The offset half: a small step must still get enough lead to break a
    # stall band, or the cap has traded an oscillation for a deadlock.
    d.s, d.s_scale = 0.100, 0.030
    small = float(np.linalg.norm(d.lead_for(np.array([0.008, 0.0, 0.0]))))
    assert small > 0.015, (
        f"an 8 mm step got only {small*1000:.1f} mm of lead — too little to "
        "break a ~17 mm standing offset, which is a deadlock, not a fix")
    print(f"  8 mm step still gets {small*1000:.1f} mm of lead to break away")


def _lead_cannot_point_backwards() -> None:
    """A reversal must not be made worse by what the previous move learned.

    This is the failure the scalar exists to prevent, and it is worth a test of
    its own because the vector version passes every straight-line check and only
    breaks when the policy turns around — which on a real handover it does
    constantly, as the human's hand moves.
    """
    print("\n-- reversal -------------------------------------------------")
    droop = m.DroopCompensator(enabled=True)
    # A +x move that was commanded 80 mm and delivered 40 mm: gain 0.5.
    droop.observe_move(0.08, np.zeros(3), np.array([0.04, 0.0, 0.0]))
    forward = droop.lead_for(np.array([0.04, 0.0, 0.0]))
    backward = droop.lead_for(np.array([-0.04, 0.0, 0.0]))
    print(f"  learned from +x move: {droop.describe()}")
    print(f"  lead(+x)={forward*1000} mm  lead(-x)={backward*1000} mm")
    assert forward[0] > 0 and backward[0] < 0, "lead did not follow the travel"
    assert np.allclose(forward, -backward), "lead is not direction-symmetric"

    # The other half: a lead must never oppose the way the arm still has to go.
    # Without this, an overshoot leaves the equilibrium beyond the target — still
    # ahead of the arm — and the controller keeps pulling it the wrong way.
    over = m.clip_lead_to_error(np.array([0.057, 0.0, 0.0]),
                                np.array([-0.0084, 0.0, 0.0]))
    print(f"  57 mm lead against an 8.4 mm overshoot -> {over[0]*1000:.1f} mm")
    assert over[0] <= 0.0, "an opposing lead survived the clip"
    keep = m.clip_lead_to_error(np.array([0.057, 0.0, 0.0]),
                                np.array([0.0084, 0.0, 0.0]))
    assert np.allclose(keep[0], 0.057), "the clip is eating a helpful lead"

    # The vector estimator, for contrast: it would command +x lead on a -x move.
    vec = m.DroopCompensator(enabled=True)
    vec.d = np.array([0.02, 0.0, 0.0])
    wrong = vec.compensate(_pose([0.0, 0.0, 0.0]))[:3, 3]
    print(f"  vector estimator on a -x move would still lead {wrong*1000} mm")
    assert wrong[0] > 0


def _latency_is_not_a_stall() -> None:
    """A slow round trip must not be read as the arm having stalled.

    Without the dead time the arm looks perfectly still for the whole of the
    controller's reaction delay, the creep scores that as a stall and nudges for
    an error the outstanding command was already going to fix. Two nudges for
    one error is an overshoot, and overshoot means a reversal — a twitch, which
    is the thing being removed.
    """
    print("\n-- latency --------------------------------------------------")
    results = {}
    for lag_s in (0.0, 0.05):
        plant = Plant([0.45, 0.0, 0.50], seed=1)
        h = Harness(plant)
        # Delay the plant's reaction to every command by lag_s.
        pending: list[tuple[float, np.ndarray]] = []
        raw_publish, raw_sleep = h.publish, h.sleep

        def publish(msg, _p=pending, _h=h):
            pos = msg["pose"]["position"]
            _p.append((_h.t + lag_s,
                       np.array([pos["x"], pos["y"], pos["z"]])))

        def sleep(dt, _p=pending, _h=h, _raw=raw_sleep):
            due = [e for e in _p if e[0] <= _h.t]
            for _, E in due:
                _h.plant.command(E)
                _p.remove((_, E))
            _raw(dt)

        h.publish, h.sleep = publish, sleep

        real_time, m.time = m.time, h
        try:
            droop = m.DroopCompensator(enabled=True)
            target = _pose(plant.p + np.array([0.0, 0.0, 0.039]))
            res = m.settle(h, m.current_msg, target, 0, m.SETTLE_TIMEOUT_S,
                           droop, tol_m=m.STEP_CONVERGE_TOL_M, creep=True)
        finally:
            m.time = real_time
            m.current_msg = None

        # Over the TRACE, not the final pose: an overshoot that is corrected
        # before the move ends is exactly the failure being looked for, and
        # reading the last sample would miss every one of them.
        overshoot = max((float(p[2]) - target[2, 3]) for _, p in h.trace)
        back = h.reversals([0.0, 0.0, 1.0])
        results[lag_s] = (res.nudges, res.pos_err * 1000, overshoot * 1000, back)
        print(f"  lag {lag_s*1000:3.0f} ms: {res.nudges} nudges, "
              f"{res.pos_err*1000:.2f} mm final, peak overshoot "
              f"{overshoot*1000:+.2f} mm, {back} reversals")

    assert results[0.05][0] <= results[0.0][0] + 2, (
        "latency is being counted as a stall: "
        f"{results[0.05][0]} nudges with lag vs {results[0.0][0]} without")
    assert results[0.05][2] < 2.0, "overshot under latency"
    assert results[0.05][3] == 0, "backtracked under latency"


def _rotation_residual_does_not_stall_the_step() -> None:
    """An uncorrectable rotation residual must not turn every step into a timeout.

    settle() judges convergence on translation now, and this is the case that
    forced the change. The lead is translation-only, so if rotation gated
    convergence too, any rotation the controller under-travelled past 3 deg would
    hold the step open until the timeout — every step, on a robot where
    rotational droop was measured at 0.9 deg and steps rotate 3.4-4.6 deg. The
    arm here simply refuses to rotate at all, which is the worst case.
    """
    print("\n-- rotation residual ----------------------------------------")
    from scipy.spatial.transform import Rotation as Rot

    plant = Plant([0.45, 0.0, 0.50], seed=7)
    h = Harness(plant)                      # arm holds identity orientation
    target = _pose(plant.p + np.array([0.0, 0.0, 0.039]),
                   R=Rot.from_euler("y", 25.0, degrees=True).as_matrix())

    real_time, m.time = m.time, h
    try:
        res = m.settle(h, m.current_msg, target, 0, m.SETTLE_TIMEOUT_S,
                       m.DroopCompensator(enabled=True),
                       tol_m=m.STEP_CONVERGE_TOL_M, creep=True)
        elapsed = h.t
    finally:
        m.time = real_time
        m.current_msg = None

    print(f"  25 deg of rotation the arm cannot do: settled={res.settled} "
          f"pos {res.pos_err*1000:.2f} mm, rot {np.rad2deg(res.rot_err):.1f} deg, "
          f"{elapsed:.2f} s of a {m.SETTLE_TIMEOUT_S:.1f} s budget")
    assert res.settled, "a rotation residual is stalling the whole step"
    assert res.pos_err < m.STEP_CONVERGE_TOL_M, "translation did not converge"
    assert elapsed < m.SETTLE_TIMEOUT_S * 0.75, (
        f"the step is running to timeout ({elapsed:.2f} s) despite converging")
    # The residual must still be REPORTED, or this silently hides a real fault.
    assert np.rad2deg(res.rot_err) > 20.0, "the rotation residual is not surfaced"


def _breakaway_is_load_bearing() -> None:
    """CREEP_BREAKAWAY_M has to be big enough to actually reach break-away.

    The case is a small residual with no learned lead yet — the arm needs the
    equilibrium ~25 mm past the target before it moves at all, and has 6 mm to
    travel, so a proportional nudge is worth 3.6 mm and gets nowhere. This exists
    because the constant was first set to 4 mm and measured to change literally
    nothing: it was still smaller than the proportional term it was meant to
    replace, so the floor never bound. A test that only ran the normal episode
    would have called that a pass.
    """
    print("\n-- break-away -----------------------------------------------")
    out = {}
    for brk in (0.004, m.CREEP_BREAKAWAY_M):
        saved, m.CREEP_BREAKAWAY_M = m.CREEP_BREAKAWAY_M, brk
        plant = Plant([0.45, 0.0, 0.50], seed=5,
                      gain=0.60, stall_m=0.025, tau_s=0.20)
        h = Harness(plant)
        real_time, m.time = m.time, h
        try:
            target = _pose(plant.p + np.array([0.0, 0.0, 0.006]))
            res = m.settle(h, m.current_msg, target, 0, m.SETTLE_TIMEOUT_S,
                           m.DroopCompensator(enabled=True),
                           tol_m=m.STEP_CONVERGE_TOL_M, creep=True)
            out[brk] = (res.settled, res.nudges, res.pos_err * 1000, h.t)
        finally:
            m.time = real_time
            m.current_msg = None
            m.CREEP_BREAKAWAY_M = saved
        print(f"  floor {brk*1000:5.1f} mm: settled={out[brk][0]} "
              f"nudges={out[brk][1]} final={out[brk][2]:.2f} mm in {out[brk][3]:.2f} s")

    assert not out[0.004][0], (
        "a 4 mm floor now converges here, so this scene no longer isolates what "
        "the floor is for and the comparison below is vacuous")
    assert out[m.CREEP_BREAKAWAY_M][0], (
        f"CREEP_BREAKAWAY_M = {m.CREEP_BREAKAWAY_M*1000:.0f} mm cannot break the "
        "arm loose from a small residual — it is too small to be worth having")


def _plants_this_robot_could_be() -> None:
    """Only the plants consistent with what this robot has actually shown.

    This is the check that matters, and it is separate from the free sweep below
    because a free grid contains plants this robot demonstrably is not, and
    tuning against those is how you end up trading away the cases that are real.

    The one hard datum is that the controller "settles ~17 mm short of any
    target". For a 40 mm command that means it achieved 23 mm, so

        gain * 0.040 - stall = 0.023

    which is a ONE-PARAMETER family, not a free grid: gain from 0.58 (where the
    stall band would go negative) up to 1.0 (a pure stall band, no under-travel
    at all). Everything in between is a candidate; nothing outside it is. That
    single constraint does more to narrow the plant than any amount of sweeping,
    and it is worth more than it looks: a plant like gain 0.50 with a 17 mm stall
    band would move the arm 3 mm on a 40 mm command, and no robot anyone is
    running a handover policy on behaves like that.
    """
    print("\n-- plants consistent with this robot's own measurement -------")
    print(f"  {'gain':>5s} {'stall':>7s} │{'cmds old':>9s}{'cmds new':>9s} │"
          f"{'dead new':>9s}{'rev new':>8s}{'err new':>8s}{'conv':>7s}")
    worst_cmds = worst_dead = 0.0
    for gain in (0.60, 0.70, 0.80, 0.90, 1.00):
        stall = 0.040 * gain - 0.023
        # With the jitter the robot actually shows. Without it this test passes
        # at 1.2 commands a step while the real arm needs 3 to 6, which is
        # exactly how the fixed-gain version of it came to be believed.
        kw = dict(gain=gain, stall_m=stall, tau_s=0.12, gain_jitter=0.22)
        saved = m.STEP_CONVERGE_TOL_M
        m.STEP_CONVERGE_TOL_M = 0.003
        old = run_steps(creep=False, n_steps=12, step_m=0.033, plant_kw=kw)
        m.STEP_CONVERGE_TOL_M = m.step_tolerance(0.033)
        new = run_steps(creep=True, n_steps=12, step_m=0.033, plant_kw=kw)
        m.STEP_CONVERGE_TOL_M = saved
        worst_cmds = max(worst_cmds, new["commands"])
        worst_dead = max(worst_dead, new["dead"])
        print(f"  {gain:5.2f} {stall*1000:6.1f}m │{old['commands']:9.1f}"
              f"{new['commands']:9.1f} │{new['dead']*1000:8.0f}m"
              f"{new['back']:8.2f}{new['worst_mm']:7.2f}m"
              f"{new['converged']:4d}/{new['n']}")
        assert old["commands"] >= 2.5, (
            f"the multi-pass path only issues {old['commands']:.1f} commands per "
            "step here, so this plant no longer reproduces the reported "
            "one-jump-plus-two-adjustments and the comparison is vacuous")
        assert new["converged"] == new["n"], (
            f"creep converged on {new['converged']}/{new['n']} at gain={gain}")

    print(f"  worst: {worst_cmds:.1f} commands/step, "
          f"{worst_dead*1000:.0f} ms stopped mid-move")
    # 1.5 because the first move or two of an episode legitimately need a
    # correction while the lead is still being learned; after that it is 1.
    assert worst_cmds <= 1.5, (
        f"creep needs {worst_cmds:.1f} commands per step on a plant this robot "
        "could actually be — the arm will still visibly stop and restart")
    assert worst_dead <= 0.05, (
        f"creep sits still {worst_dead*1000:.0f} ms mid-step on a plant this "
        "robot could actually be")


def _tolerance_can_never_skip_a_step() -> None:
    """A relative tolerance must always be smaller than the step it judges.

    This is the property that makes it safe to be as loose as it is. A FIXED
    tolerance wide enough to stop the corrections — 20 mm, measured — silently
    turns every commanded step under 20 mm into a no-op: the arm is already
    within tolerance of the target before it moves, so settle() returns at once
    and the policy is told the step happened. Near the object, where the policy
    commands millimetres, that is the whole endgame quietly not executing.
    """
    print("\n-- tolerance vs step size -----------------------------------")
    for step in (0.050, 0.033, 0.020, 0.010, 0.005, 0.002):
        tol = m.step_tolerance(step)
        flag = "" if tol < step else "   <-- WOULD SKIP"
        print(f"  step {step*1000:5.1f} mm -> tol {tol*1000:5.1f} mm{flag}")
        assert tol < step or step <= m.STEP_CONVERGE_TOL_MIN_M, (
            f"a {step*1000:.0f} mm step has a {tol*1000:.0f} mm tolerance, so it "
            "is satisfied without the arm moving at all")
    # And the floor has to hold for steps small enough that a fraction of them
    # would be below the arm's own noise.
    assert m.step_tolerance(1e-6) == m.STEP_CONVERGE_TOL_MIN_M


def _fixed_rate_tracks_and_never_deadlocks() -> None:
    """--control rate must keep the arm moving, including on tiny steps.

    Two things are being asserted, and the second is the one that matters.

    IT REACHES. Fixed-rate never checks arrival, so the guarantee is not
    per-step convergence — it is that the arm closes on a target that keeps
    being re-issued. A policy step here is a fresh delta from the MEASURED pose,
    exactly as the runner builds it, so under-travel shortens the reach rather
    than accumulating error, and what "working" means is that repeated ticks
    converge on the goal.

    IT DOES NOT DEADLOCK ON SMALL STEPS. This is the failure mode the mode
    introduces and settle() does not have. The target is rebuilt from the
    measured pose every tick, so a command the arm is too stiff to execute does
    not accumulate: the equilibrium is re-placed at the same physical spot and
    the arm never moves again. The standing lead is the only thing preventing
    it, which is why a plant with a stall band is swept here at a step size well
    inside that band — 4 mm against a 12 mm dead zone. Without the lead this
    test hangs at zero displacement, which is what makes it worth having.
    """
    for label, step_m, plant_kw in (
            ("normal step  39 mm", 0.039, {}),
            ("small step    8 mm", 0.008, {}),
            ("inside stall  4 mm", 0.004, {"stall_m": 0.012}),
            ("stiff arm    39 mm", 0.039, {"gain": 0.45, "gain_jitter": 0.22}),
    ):
        plant = Plant([0.45, 0.0, 0.50], seed=3, **plant_kw)
        h = Harness(plant)
        real_time, m.time = m.time, h
        try:
            droop = m.DroopCompensator(enabled=True)
            rc = m.RateCommander(h, m.RATE_CONTROL_HZ, droop)
            start = plant.p.copy()
            goal = start + np.array([0.0, 0.0, 1.0]) * 0.0  # set below
            direction = np.array([0.6, -0.5, 0.62])
            direction /= np.linalg.norm(direction)
            goal = start + direction * (step_m * 12)

            seq, t0 = 0, h.t
            for _ in range(60):
                err = goal - plant.p
                n = float(np.linalg.norm(err))
                if n < 0.002:
                    break
                # One policy step: a bounded delta toward the goal, taken from
                # the measured pose. Same construction as the runner's
                # T_base_hand @ unpack_action(delta).
                d = err / n * min(step_m, n)
                seq, _ = rc.command(_pose(plant.p + d), seq)
            moved = float(np.linalg.norm(plant.p - start))
            remain = float(np.linalg.norm(goal - plant.p))
        finally:
            m.time = real_time
            m.current_msg = None

        assert moved > 1e-3, (
            f"{label}: arm did not move at all ({moved*1000:.2f} mm) — the "
            "fixed-rate loop deadlocked, which is what the standing lead exists "
            "to prevent")
        assert remain < 0.006, (
            f"{label}: {remain*1000:.1f} mm short of the goal after 60 ticks "
            f"(moved {moved*1000:.0f} mm of {np.linalg.norm(goal-start)*1000:.0f})")
        print(f"  {label}: reached to {remain*1000:4.1f} mm in "
              f"{h.t - t0:4.1f} s, {plant.commands} commands, "
              f"lead {droop.s*1000:.1f} mm")

    # And the mode must be CHEAPER than settle on the same plant, or there is no
    # reason for it to exist. Compared at equal work: one policy step.
    plant = Plant([0.45, 0.0, 0.50], seed=3)
    h = Harness(plant)
    real_time, m.time = m.time, h
    try:
        droop = m.DroopCompensator(enabled=True)
        rc = m.RateCommander(h, m.RATE_CONTROL_HZ, droop)
        t0 = h.t
        rc.command(_pose(plant.p + np.array([0.02, 0.01, 0.015])), 0)
        rate_s = h.t - t0
    finally:
        m.time = real_time
        m.current_msg = None
    settle_s = run_steps(creep=True, n_steps=8)["secs"]
    assert rate_s < settle_s, (
        f"fixed rate cost {rate_s:.3f} s a step against settle's {settle_s:.3f} s")
    print(f"  per step: rate {rate_s*1000:.0f} ms vs settle "
          f"{settle_s*1000:.0f} ms ({settle_s/rate_s:.1f}x)")


def _homing_is_continuous() -> None:
    """Streamed homing must land, and must not stop on the way.

    Both halves matter and they pull against each other. Landing is easy if you
    settle every waypoint — that is what the old path did, and it is why homing
    stuttered. Not stopping is easy if you never converge — and then the arm ends
    somewhere near home rather than at it, which is off-distribution on step 0
    for a policy that only ever saw states downstream of the sim start pose.

    `dead` is seconds spent stopped in the MIDDLE of the motion, using the same
    2.5 mm/s the runner itself calls "stopped", so it counts exactly the pauses a
    person would see. The stepwise path is measured alongside so the comparison
    is against the real alternative and not against a remembered number.
    """
    out = {}
    for label, stream in (("streamed", True), ("stepwise", False)):
        plant = Plant([0.30, 0.20, 0.62], seed=1)
        h = Harness(plant)
        real_time, m.time = m.time, h
        try:
            droop = m.DroopCompensator(enabled=True)
            h.reset_trace()
            t0 = h.t
            m.go_home(h, np.eye(4), np.eye(4), 0, droop, creep=True,
                      stream=stream)
            bursts, dead = h.motion_profile()
            err = float(np.linalg.norm(plant.p - m.T_BASE_HAND_HOME[:3, 3]))
            out[label] = {"stops": max(bursts - 1, 0), "dead": dead,
                          "err_mm": err * 1000.0, "secs": h.t - t0,
                          "commands": plant.commands}
        finally:
            m.time = real_time
            m.current_msg = None
        r = out[label]
        print(f"  {label:8s}: {r['stops']:2d} stops, {r['dead']*1000:5.0f} ms "
              f"dead, landed {r['err_mm']:4.1f} mm out, {r['secs']:4.1f} s, "
              f"{r['commands']} commands")

    s, p = out["streamed"], out["stepwise"]
    # Landing is the requirement that survives whatever the plant turns out to
    # be, so it is asserted absolutely rather than relative to the old path.
    # Against the CONFIGURED tolerance, not a number: this read `< 5.0 mm` while
    # the tolerance was 2 mm, and asking the arm for more precision than it can
    # land is what made homing hunt in the first place. A test that pins the old
    # number would just re-argue for the bug.
    landed_tol_mm = m.HOME_REFINE_TOL_M * 1000.0
    assert s["err_mm"] < landed_tol_mm, (
        f"streamed homing landed {s['err_mm']:.1f} mm from home, outside the "
        f"{landed_tol_mm:.0f} mm tolerance — the refine is not closing the lag "
        "the streamed path leaves")
    # And the whole point: strictly fewer mid-motion stops than settling every
    # waypoint. Relative, because the absolute count depends on the plant.
    assert s["dead"] < p["dead"], (
        f"streamed homing spent {s['dead']*1000:.0f} ms stopped against "
        f"stepwise's {p['dead']*1000:.0f} ms — it is not smoother")
    print(f"  dead time {p['dead']*1000:.0f} -> {s['dead']*1000:.0f} ms "
          f"({p['dead']/max(s['dead'], 1e-9):.1f}x less standing still)")


def _second_home_does_not_hunt() -> None:
    """Homing after an episode must behave like homing before one.

    The reported failure was specifically the SECOND home: "the first time
    homing happens well, but afterwards ... it looks like it is trying to find
    exact home position but it is not able to". `_homing_is_continuous` could
    never have caught it, because it homes with a fresh DroopCompensator — which
    is the first-home case, and the first home was always fine.

    What differs is that an episode has taught the estimator a lead. Two things
    then went wrong, and both are asserted here: the lead was applied at full
    magnitude to the refine's millimetre-scale first command, and the refine was
    chasing a 2 mm tolerance the arm cannot land against a 17 mm droop and a
    gain that swings 2x between moves.
    """
    print("\n-- second home ----------------------------------------------")
    rng = np.random.default_rng(5)
    out = {}
    for label, learn in (("fresh droop", False), ("after an episode", True)):
        plant = Plant([0.30, 0.20, 0.62], seed=1)
        h = Harness(plant)
        real_time, m.time = m.time, h
        try:
            droop = m.DroopCompensator(enabled=True)
            if learn:
                # An episode's worth of full-scale steps, as the runner flies
                # before you press 'h' the second time.
                seq = 0
                for _ in range(8):
                    d = rng.normal(size=3)
                    d /= np.linalg.norm(d)
                    res = m.settle(h, m.current_msg, _pose(plant.p + d * 0.030),
                                   seq, m.SETTLE_TIMEOUT_S, droop,
                                   tol_m=m.STEP_CONVERGE_TOL_M, creep=True)
                    seq = res.next_seq
                plant.p = np.array([0.30, 0.20, 0.62])   # back out to where home starts
                h._write_msg()
            h.reset_trace()
            c0 = plant.commands
            m.go_home(h, np.eye(4), np.eye(4), 0, droop, creep=True, stream=True)
            bursts, dead = h.motion_profile()
            out[label] = {
                "stops": max(bursts - 1, 0),
                "dead_ms": dead * 1000.0,
                "err_mm": float(np.linalg.norm(
                    plant.p - m.T_BASE_HAND_HOME[:3, 3])) * 1000.0,
                "commands": plant.commands - c0,
                "lead_mm": droop.s * 1000.0,
            }
        finally:
            m.time = real_time
            m.current_msg = None
        r = out[label]
        print(f"  {label:17s}: {r['stops']:2d} stops, {r['dead_ms']:5.0f} ms "
              f"dead, landed {r['err_mm']:4.1f} mm out, {r['commands']} commands "
              f"(learned lead {r['lead_mm']:.0f} mm)")

    a, b = out["fresh droop"], out["after an episode"]
    assert b["lead_mm"] > 1.0, "staging failed: no lead was learned to misapply"
    tol_mm = m.HOME_REFINE_TOL_M * 1000.0
    assert b["err_mm"] < tol_mm, (
        f"the second home landed {b['err_mm']:.1f} mm out, outside {tol_mm:.0f} mm")
    # The hunt showed up as extra commands and extra stops around home. Allowing
    # a couple more than the fresh case, since a learned lead legitimately
    # changes the approach; what it may not do is multiply them.
    assert b["commands"] <= a["commands"] + 3, (
        f"the second home took {b['commands']} commands against the first's "
        f"{a['commands']} — this is the hunt")
    assert b["stops"] <= a["stops"] + 1, (
        f"the second home stopped {b['stops']} times against {a['stops']}")


def _abort_interrupts_a_motion_in_progress() -> None:
    """'t' must stop the arm DURING a step, not after it.

    The whole point of the abort machinery is that the main loop's key handling
    does not run while the robot is moving — a policy step can sit inside
    settle() for seconds. So the test that matters is not "does the flag stop the
    next step", which any `elif` would pass; it is "does a key pressed mid-motion
    end the motion it was pressed during".

    The poller is armed to fire once, part-way through, and what is asserted is
    that settle() returns EARLY and that the arm stops well short of the target
    it was commanded to. If the abort were only checked between steps, the arm
    would arrive and the elapsed time would be the full move.
    """
    for label, fire_after_s in (("abort at 0.15 s", 0.15),
                                ("abort at 0.40 s", 0.40)):
        plant = Plant([0.45, 0.0, 0.50], seed=5)
        h = Harness(plant)
        real_time, m.time = m.time, h
        try:
            m.clear_stop()
            fired = {"at": None}

            def poller() -> bool:
                if h.t >= fire_after_s:
                    fired["at"] = h.t
                    return True
                return False

            m._stop_poller = poller
            start = plant.p.copy()
            target = _pose(start + np.array([0.0, 0.0, 1.0]) * 0.0
                           + np.array([0.20, 0.0, 0.0]))   # a long 20 cm move
            t0 = h.t
            res = m.settle(h, m.current_msg, target, 0, m.SETTLE_TIMEOUT_S,
                           m.DroopCompensator(enabled=True),
                           tol_m=m.STEP_CONVERGE_TOL_M, creep=True)
            elapsed = h.t - t0
            moved = float(np.linalg.norm(plant.p - start))
        finally:
            m._stop_poller = None
            m.clear_stop()
            m.time = real_time
            m.current_msg = None

        assert fired["at"] is not None, f"{label}: poller never ran — settle is not polling"
        assert not res.settled, f"{label}: settle reported success after an abort"
        assert elapsed < fire_after_s + 0.10, (
            f"{label}: settle ran {elapsed:.2f} s, {elapsed - fire_after_s:.2f} s "
            "past the abort — it is not being noticed inside the motion")
        assert moved < 0.19, (
            f"{label}: arm still covered {moved*1000:.0f} mm of a 200 mm move")
        print(f"  {label}: settle returned at {elapsed:.2f} s, "
              f"arm stopped after {moved*1000:.0f} of 200 mm")

    # And freeze_arm must command the pose the arm is actually AT, since that is
    # the entire mechanism — an equilibrium anywhere else keeps pulling.
    plant = Plant([0.45, 0.0, 0.50], seed=5)
    h = Harness(plant)
    real_time, m.time = m.time, h
    try:
        h.publish({"pose": {"position": {"x": 0.60, "y": 0.10, "z": 0.55}}})
        h.sleep(0.5)                        # let it run toward that target
        where = plant.p.copy()
        m.freeze_arm(h, 0)
        h.sleep(1.0)                        # and nothing should happen now
        drift = float(np.linalg.norm(plant.p - where))
    finally:
        m.time = real_time
        m.current_msg = None
    assert drift < 0.002, (
        f"arm drifted {drift*1000:.1f} mm after freeze_arm — the equilibrium is "
        "not on top of the measured pose")
    print(f"  freeze_arm: {drift*1000:.2f} mm of drift over 1.0 s after the stop")


def _sweep() -> None:
    """The conclusion has to hold across the plants that fit the measurements.

    A single plant setting proving the point would only prove that a plant
    exists which proves the point — and in this case the sweep earned its keep
    twice: it rejected a lead cap that worked on one plant and deadlocked
    another, and it showed a break-away floor that was doing nothing at all.

    The gain/stall grid brackets what this robot has actually shown: a standing
    offset of ~17 mm, and an under-travel between 0.35 (implied by the homing
    convergence ratio) and 0.66 (from the two-command pair).
    """
    print("\n-- plant sweep ----------------------------------------------")
    print(f"  {'gain':>5s} {'stall':>6s} {'tau':>5s} │"
          f"{'dead old':>9s}{'dead new':>9s} │{'rev new':>8s}{'err new':>8s}"
          f"{'conv':>7s}")
    worst_rev = 0.0
    better = total = 0
    worst_point = None
    for gain in (0.35, 0.5, 0.66, 0.95):
        for stall in (0.004, 0.017, 0.025):
            for tau in (0.06, 0.12, 0.25):
                kw = dict(gain=gain, stall_m=stall, tau_s=tau)
                old = run_steps(creep=False, n_steps=5, plant_kw=kw)
                new = run_steps(creep=True, n_steps=5, plant_kw=kw)
                total += 1
                better += new["dead"] < old["dead"]
                worst_rev = max(worst_rev, new["back"])
                if worst_point is None or new["dead"] > worst_point[0]:
                    worst_point = (new["dead"], gain, stall, tau)
                assert new["converged"] == new["n"], (
                    f"creep failed to converge at gain={gain} stall={stall} "
                    f"tau={tau}: {new['converged']}/{new['n']}")
                print(f"  {gain:5.2f} {stall*1000:5.0f}m {tau:5.2f} │"
                      f"{old['dead']*1000:8.0f}m{new['dead']*1000:8.0f}m │"
                      f"{new['back']:8.2f}{new['worst_mm']:7.2f}m"
                      f"{new['converged']:4d}/{new['n']}")

    d, g, s, tau = worst_point
    print(f"  creep beats multi-pass on dead time at {better}/{total} points")
    print(f"  worst creep reversals across the sweep: {worst_rev:.2f} per step")
    print(f"  worst creep dead time: {d*1000:.0f} ms at "
          f"gain={g} stall={s*1000:.0f}mm tau={tau}")

    # This grid is a STRESS TEST, not a target, and the only thing asserted on it
    # is that nothing deadlocks — every point still converges, checked per point
    # above. Dead time and reversals are printed and not bounded, on purpose.
    #
    # Most of this grid contradicts the robot. Its own datum is that the
    # controller settles ~17 mm short of a target, i.e. gain*0.040 - stall =
    # 0.023 for a 40 mm command; a point like gain 0.35 with a 25 mm stall band
    # would move the arm NOT AT ALL on that command. Tuning to make such points
    # look good costs the points that are real: measured, one attempt to do that
    # took the plants this robot could actually be from 1.2 commands per step to
    # 6.8. _plants_this_robot_could_be() is where the bar is set, and this is
    # here to show what happens outside it.
    print("  (bounds are asserted in _plants_this_robot_could_be, not here — "
          "most of this grid contradicts the robot's own 17 mm datum)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep the plant parameters (slower)")
    args = ap.parse_args()

    _comparison()
    _travel_lead_learns()
    _lead_does_not_overshoot_a_short_correction()
    _lead_cannot_dwarf_the_step()
    _lead_cannot_point_backwards()
    _latency_is_not_a_stall()
    _rotation_residual_does_not_stall_the_step()
    _breakaway_is_load_bearing()
    _plants_this_robot_could_be()
    _tolerance_can_never_skip_a_step()
    print("\nfixed-rate control (--control rate)")
    _fixed_rate_tracks_and_never_deadlocks()
    print("\nhoming motion")
    _homing_is_continuous()
    _second_home_does_not_hunt()
    print("\nabort key ('t')")
    _abort_interrupts_a_motion_in_progress()
    if args.sweep:
        _sweep()
    print("\nall step-motion checks passed")


if __name__ == "__main__":
    main()
