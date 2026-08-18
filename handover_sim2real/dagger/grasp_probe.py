"""Counterfactual opportunity: would closing RIGHT HERE actually get the object?

THE PROBLEM WITH THE GEOMETRIC TEST ALONE. `grasp_box.grasp_opportunity` asks
whether object material sits between the open pads. That is necessary but not
sufficient, and it is wrong in both directions at once: the MANO hand is
invisible to the rays, so it reports a chance in poses where closing would
collide with the human (an over-count), while the ray grid only spans the distal
17.6 mm of the pad, so an object gripped nearer the palm registers nothing (an
under-count). Measured on run 16's test split at min_frac 0.50, it reported a
chance in 80 of 130 episodes while 90 secured the object — `box_chance_rate`
below `success_rate`, which is geometrically impossible and proves the
under-count directly.

WHAT THIS DOES INSTEAD, as a hierarchy. The cheap geometric test stays as the
GATE — it costs 49 raycasts and rejects most steps outright. Only when it passes
does this run the expensive, definitive check: close the gripper for real, hold
it, and score the result with `grasp_held_after_hold`, the same criterion the
evaluator uses to decide success. A step counts as an opportunity only if a close
there would genuinely have secured the object. That folds in everything geometry
cannot see — the human hand, the object's mass and friction, whether the fingers
would shove it out of the way, whether the release handshake fires.

    frac >= min_frac  ->  close, hold, score  ->  held ? opportunity : not

AND IT MUST LEAVE NO TRACE. The probe advances physics by
`hold_steps * steps_action_repeat` steps, closes the fingers, and can set
`ycb.released`, `_dropped` and the benchmark's status flags — so without an exact
rewind it would destroy the very episode it is measuring. `probe_grasp_here`
snapshots and restores:

  * PyBullet's full dynamics state, via saveState/restoreState.
  * The Python-side counters the wrapper chain keeps, which PyBullet knows
    nothing about: the frame counters that drive the recorded hand and object
    trajectories, the elapsed-step count, the release latch, the drop latch and
    the success dwell counter. Each is written back to the object that OWNS it —
    `gym.Wrapper.__getattr__` forwards reads down the chain but an assignment
    would silently create a shadowing attribute on the wrapper instead.
  * easysim's cached body state. `link_state` / `dof_state` / `contact` are
    collected lazily and memoised per step, so after a rewind they still hold
    post-probe values; nulling them is what makes the next read come from the
    restored simulator.

VERIFIED, NOT ASSUMED. The restore is checked before returning: the hand pose,
the object pose and the frame counter must come back to what they were. A silent
failure here would corrupt every episode after the first probe in a way that
looks like ordinary policy variance, so a mismatch raises instead.

COST. One probe is `hold_steps * steps_action_repeat` sim steps — 450 at the
defaults — against the ~50 policy steps of an episode. The gate keeps that off
most steps, but it fires on ~2 steps per episode that has a chance, so budget
roughly double the eval wall clock.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch          # easysim's update_attr_array takes a tensor of env ids

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from handover_sim2real.rl.rollout_worker import grasp_held_after_hold  # noqa: E402
from handover_sim2real.dagger.grasp_box import _bullet_client  # noqa: E402

# Python attributes that survive a PyBullet restoreState and therefore have to be
# rewound by hand. Ordered as (owner-search name); the owner is resolved at
# runtime because they live on different links of the gym wrapper chain:
# `_frame` on the base HandoverEnv, `_elapsed_steps` / `_dropped` /
# `_success_step_counter` on HandoverBenchmarkWrapper.
_ENV_ATTRS = ("_frame", "_elapsed_steps", "_dropped", "_success_step_counter")
# ...and the same for the two trajectory players, which own their own cursors.
_YCB_ATTRS = ("_frame", "_released")
_MANO_ATTRS = ("_frame",)


def _owner(obj, name):
    """The object in the wrapper chain whose own __dict__ holds `name`.

    Needed in BOTH directions, for two different reasons.

    Reading: gym's `Wrapper.__getattr__` refuses outright to forward names that
    start with an underscore ("attempted to get missing private attribute"), and
    every counter here is private, so `env._frame` raises rather than reaching
    the base env.

    Writing: `env._dropped = False` would set the attribute on whichever wrapper
    is being held, shadowing the real flag on the wrapper that owns it and
    leaving the actual episode state untouched.

    So both the snapshot and the restore resolve the owner first.
    """
    cur = obj
    while cur is not None:
        if name in vars(cur):
            return cur
        cur = vars(cur).get("env")
    raise AttributeError(f"no owner for {name!r} in the wrapper chain")


def _get(obj, name):
    return vars(_owner(obj, name))[name]


def _clone(v):
    """Detached copy of an easysim attribute, tensor or array or None."""
    if v is None:
        return None
    return v.clone() if hasattr(v, "clone") else np.array(v).copy()


def _snapshot(env):
    p = _bullet_client(env)
    sim = env.simulator
    ycb_body = env.ycb.bodies[env.ycb.ids[0]]
    state = {
        "bullet": p.saveState(),
        "env": {n: _get(env, n) for n in _ENV_ATTRS},
        "ycb": {n: _get(env.ycb, n) for n in _YCB_ATTRS},
        "mano": ({n: _get(env.mano, n) for n in _MANO_ATTRS}
                 if getattr(env, "mano", None) is not None else {}),
        # ---- CONTROLLER STATE, which saveState does NOT cover ----
        # PyBullet's saveState captures dynamics (positions, velocities, forces)
        # and nothing about how the joints are being DRIVEN. The probe commands
        # the fingers shut, and that setpoint survives restoreState, so the very
        # next env.step drives them closed again and the episode is over. Caught
        # by the acceptance test: with the probe on, 10 test scenes went 80% ->
        # 30% success with TIMEOUTs appearing (a shut gripper can never pass the
        # open_thresh gate again).
        "targets": [(b, _clone(b.dof_target_position)) for b in sim._scene.bodies],
        # ...and `ycb.release()` is not just a flag either: it rewrites the
        # object's collision filter to COLLISION_FILTER_YCB_RELEASE and zeroes
        # its dof_max_force. Both are easysim attributes pushed into Bullet, so a
        # probe that trips the human's release leaves the object permanently
        # free-falling with no motor force for the rest of the real episode.
        "ycb_body": ycb_body,
        "ycb_filter": _clone(ycb_body.get_attr_array("link_collision_filter", 0)),
        "ycb_maxforce": _clone(ycb_body.get_attr_array("dof_max_force", 0)),
    }
    # The witnesses the restore is checked against — read AFTER the ids above so
    # they describe the same instant.
    body = env.panda.body
    state["witness"] = (
        np.asarray(body.link_state[0, env.panda.LINK_IND_HAND, 0:7], dtype=np.float64).copy(),
        np.asarray(env.ycb.bodies[env.ycb.ids[0]].link_state[0, 6, 0:7],
                   dtype=np.float64).copy(),
    )
    return state


def _restore(env, state, *, verify=True, tol=1e-6):
    p = _bullet_client(env)
    p.restoreState(stateId=state["bullet"])
    p.removeState(state["bullet"])

    for n, v in state["env"].items():
        setattr(_owner(env, n), n, v)
    for n, v in state["ycb"].items():
        setattr(_owner(env.ycb, n), n, v)
    for n, v in state["mano"].items():
        setattr(_owner(env.mano, n), n, v)

    # ---- controller state, restored the way the sim's own code sets it ----
    # `dof_target_position` is re-applied on every step regardless of dirty flag,
    # so writing the value back is enough. The two release attributes go through
    # easysim's dirty-flag path and are locked after startup, so they need
    # `update_attr_array` — the same call ycb.release() uses to set them.
    for body, tgt in state["targets"]:
        body.dof_target_position = tgt
    yb = state["ycb_body"]
    if state["ycb_filter"] is not None:
        yb.update_attr_array("link_collision_filter", torch.tensor([0]),
                             state["ycb_filter"])
    if state["ycb_maxforce"] is not None:
        yb.update_attr_array("dof_max_force", torch.tensor([0]),
                             state["ycb_maxforce"])

    # easysim memoises these per step; without invalidating them the next read
    # returns the POST-probe values and the rewind is invisible to the caller.
    sim = env.simulator
    for body in sim._scene.bodies:
        body.dof_state = None
        body.link_state = None
    sim._contact = None

    if not verify:
        return
    body = env.panda.body
    ee = np.asarray(body.link_state[0, env.panda.LINK_IND_HAND, 0:7], dtype=np.float64)
    ycb = np.asarray(env.ycb.bodies[env.ycb.ids[0]].link_state[0, 6, 0:7],
                     dtype=np.float64)
    w_ee, w_ycb = state["witness"]
    d_ee, d_ycb = float(np.abs(ee - w_ee).max()), float(np.abs(ycb - w_ycb).max())
    # The pose check alone is NOT sufficient and was the reason the first version
    # passed its own verification while corrupting every episode: both known
    # leaks are latent, correct at the instant of the rewind and destructive on
    # the NEXT step. So the controller state is checked too.
    d_tgt = 0.0
    for body_, tgt in state["targets"]:
        cur = body_.dof_target_position
        if (cur is None) != (tgt is None):
            d_tgt = float("inf")
        elif cur is not None:
            d_tgt = max(d_tgt, float(np.abs(np.asarray(cur, dtype=np.float64)
                                            - np.asarray(tgt, dtype=np.float64)).max()))
    released = bool(_get(env.ycb, "_released"))
    if d_ee > tol or d_ycb > tol or d_tgt > tol:
        raise RuntimeError(
            f"grasp probe failed to rewind the simulator: EE moved {d_ee:.3e}, "
            f"object moved {d_ycb:.3e}, motor targets moved {d_tgt:.3e} "
            f"(tol {tol:.0e}), released={released}. Every episode after this "
            f"point would be silently corrupted, so this is fatal rather than a "
            f"warning. Re-run without the probe (EVAL.box_probe: false).")


def probe_grasp_here(env, obs, steps_action_repeat: int, hold_steps: int) -> bool:
    """Close, hold, score, rewind. True iff a close HERE would secure the object.

    `obs` is the observation the caller is holding; the probe steps the sim, so
    the caller must keep using its OWN `obs` afterwards rather than anything this
    returns — which is why nothing is returned but the verdict. The easysim
    bodies inside that `obs` are live handles whose caches this function
    invalidates on the way out, so they read the restored state.

    The scoring call is `grasp_held_after_hold`, imported rather than
    reimplemented: an opportunity is then true exactly when the evaluator's own
    success criterion would have been satisfied, by construction.
    """
    state = _snapshot(env)
    try:
        held, _ = grasp_held_after_hold(env, obs, int(steps_action_repeat),
                                        int(hold_steps))
        return bool(held)
    finally:
        _restore(env, state)
