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
                 noise_m=2e-6, seed=0):
        self.p = np.asarray(p0, dtype=np.float64).copy()
        self.goal = self.p.copy()
        self.gain = float(gain)
        self.stall = float(stall_m)
        self.tau = float(tau_s)
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
        travel = max(0.0, self.gain * n - self.stall)
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
            seq, dp, dr, passes = m.move_to(
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
            leads.append(droop.s * 1000.0)
    finally:
        m.time = real_time
        m.current_msg = None

    print(f"  commands per step : {cmds}")
    print(f"  learned lead (mm) : {[round(x, 1) for x in leads]}")
    first, last = cmds[0], float(np.mean(cmds[-4:]))
    print(f"  first step {first} cmds -> last four average {last:.2f}")
    assert last < first, (
        f"the lead is not buying anything: {last:.2f} commands vs {first}")
    assert np.std(leads[-4:]) < 2.0, (
        f"lead has not settled: {[round(x, 1) for x in leads[-4:]]}")


def _lead_cannot_point_backwards() -> None:
    """A reversal must not be made worse by what the previous move learned.

    This is the failure the scalar exists to prevent, and it is worth a test of
    its own because the vector version passes every straight-line check and only
    breaks when the policy turns around — which on a real handover it does
    constantly, as the human's hand moves.
    """
    print("\n-- reversal -------------------------------------------------")
    droop = m.DroopCompensator(enabled=True)
    droop.observe_travel(np.array([0.02, 0.0, 0.0]), np.array([0.04, 0.0, 0.0]))
    forward = droop.initial_lead(np.array([0.04, 0.0, 0.0]))
    backward = droop.initial_lead(np.array([-0.04, 0.0, 0.0]))
    print(f"  learned from +x move: lead(+x)={forward*1000} mm  "
          f"lead(-x)={backward*1000} mm")
    assert forward[0] > 0 and backward[0] < 0, "lead did not follow the travel"
    assert np.allclose(forward, -backward), "lead is not direction-symmetric"

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
    worst_dead = worst_rev = 0.0
    for gain in (0.35, 0.5, 0.66, 0.95):
        for stall in (0.004, 0.017, 0.025):
            for tau in (0.06, 0.12, 0.25):
                kw = dict(gain=gain, stall_m=stall, tau_s=tau)
                old = run_steps(creep=False, n_steps=5, plant_kw=kw)
                new = run_steps(creep=True, n_steps=5, plant_kw=kw)
                worst_dead = max(worst_dead, new["dead"])
                worst_rev = max(worst_rev, new["back"])
                print(f"  {gain:5.2f} {stall*1000:5.0f}m {tau:5.2f} │"
                      f"{old['dead']*1000:8.0f}m{new['dead']*1000:8.0f}m │"
                      f"{new['back']:8.2f}{new['worst_mm']:7.2f}m"
                      f"{new['converged']:4d}/{new['n']}")
    print(f"  worst creep dead time across the sweep: {worst_dead*1000:.0f} ms")
    print(f"  worst creep reversals across the sweep: {worst_rev:.2f} per step")
    # Deliberately not zero. Some backtracking survives on plants where the lead
    # a long move needs is much larger than the lead the last few millimetres
    # need, because the lead accumulates and nothing shrinks it. Removing that
    # properly means identifying the controller, which two measurements do not
    # support; the bound here is what the shipped code actually achieves.
    assert worst_dead <= 0.30, f"creep dead time reaches {worst_dead*1000:.0f} ms"
    assert worst_rev <= 1.0, f"creep reverses {worst_rev:.2f} times per step"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep the plant parameters (slower)")
    args = ap.parse_args()

    _comparison()
    _travel_lead_learns()
    _lead_cannot_point_backwards()
    _latency_is_not_a_stall()
    _rotation_residual_does_not_stall_the_step()
    _breakaway_is_load_bearing()
    if args.sweep:
        _sweep()
    print("\nall step-motion checks passed")


if __name__ == "__main__":
    main()
