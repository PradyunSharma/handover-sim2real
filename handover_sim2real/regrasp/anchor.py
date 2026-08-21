"""
The anchor frame: the gravity-aligned, hand-anchored frame the bins live in.

    z = world up (gravity)
    x = normalize(horizontal(c - p_wrist))    # object side, pointing AWAY from the giver
    y = z x x

`c` is the object point-cloud centroid and `p_wrist` the human giver's wrist. The
frame is what makes the conditioning OBJECT-AGNOSTIC: "+x" means "the free end,
away from the hand" on every object in the dataset, so coverage is required across
the dataset rather than per object. That is the whole reason collection gets
cheaper than "four grasps per object".

WHAT THE ANCHOR CAN AND CANNOT BREAK. The network never sees this frame. The
per-point channels are `d . n_i` and `d . normalize(p_i - c)`, dot products of
things all expressed in the EE frame, and dot products are frame-invariant. The
anchor is used for exactly two things: deciding which BIN a demonstration belongs
to at assignment time, and the retry machine's angular bookkeeping. So an anchor
error cannot corrupt a single network input — it can only move a near-boundary
grasp into the wrong bin. Do not over-engineer it.

THE DEGENERATE CASE. When the hand sits nearly directly above or below the
object, `horizontal(c - p_wrist)` collapses and the azimuth reference is
meaningless. Fall back to the robot-base -> object direction. The switch is
HYSTERETIC — two thresholds, not one — because a single threshold with the wrist
hovering near it would flip the frame back and forth mid-approach, and every bin
label would flip with it. `enter` < `exit`: the fallback engages below `enter` and
only releases above `exit`, so the mode latches.

STATIC HANDS HIDE ALL OF THIS. Under the active sim config
(`pretrain_multicam_wr.yaml`: `YCB_MANO_START_FRAME: last`,
`MANO_SIMULATION_MODE: disable_control_and_move_by_reset`) the MANO frame index is
clamped at `num_frames - 1` forever, so the hand is STATIC for the whole episode
and the anchor is a per-episode constant. The hysteresis is therefore untestable
in simulation as configured, and `anchor_mode` should read "wrist" on 100% of
episodes — a single "base" means the threshold is miscalibrated, not that the
fallback worked. The code still recomputes per step because a moving-hand config
and the real robot both need it.

PLAIN ARRAYS IN, PLAIN ARRAYS OUT. No `env` argument, no gym, no pybullet, no
torch. `wrist_world` / `handedness` below are the only env-aware helpers and they
are separated deliberately: on the real rig `p_wrist` comes from hand
segmentation, not from a MANO link, and `anchor_rotation` must be reusable there
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from handover_sim2real.regrasp.directions import normalize

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# The MANO wrist/palm link. NOT 0 (a massless base pinned at the world origin) and
# NOT 6 (the floating-base root, ~9 cm off the actual wrist): links 1-3 are the
# prismatic XYZ of the floating base, 4-6 the continuous rotation, and `joint7`'s
# origin IS the MANO joint-0 origin. Links 8..52 are the fingers.
MANO_WRIST_LINK = 7


@dataclass
class AnchorParams:
    """Thresholds for the degenerate-case fallback.

    The defaults are in metres of HORIZONTAL wrist-to-object offset. 0.04 / 0.08
    brackets a band roughly a hand's width wide: below 4 cm the azimuth really is
    meaningless, above 8 cm it is solid, and between them whichever mode was last
    committed keeps running.
    """

    enter: float = 0.04      # engage the fallback below this
    exit: float = 0.08       # release it only above this

    def __post_init__(self):
        if not 0.0 < self.enter < self.exit:
            raise ValueError(
                f"need 0 < enter < exit for hysteresis; got enter={self.enter}, "
                f"exit={self.exit}. Equal thresholds are a single threshold and "
                f"will chatter.")


@dataclass
class AnchorState:
    """Per-episode latch. Construct once per episode, pass to every call."""

    fallback: bool = False       # is the base-direction fallback currently engaged
    switches: int = 0            # how many times the mode flipped this episode
    history: list = field(default_factory=list)   # horizontal norms, for diagnostics

    @property
    def mode(self) -> str:
        return "base" if self.fallback else "wrist"


def horizontal(v) -> np.ndarray:
    """Project onto the world horizontal plane (drop the gravity component)."""
    v = np.asarray(v, dtype=np.float64)
    return v - np.dot(v, WORLD_UP) * WORLD_UP


def anchor_rotation(centroid, wrist, robot_base, state: AnchorState | None = None,
                    params: AnchorParams | None = None):
    """(R_anchor [3,3], meta) — columns are the anchor x, y, z in WORLD coords.

    `R_anchor @ v_anchor` gives world; `R_anchor.T @ v_world` gives anchor. Pass
    the same `state` for every step of an episode so the fallback latches.

    `wrist` may be None (no hand in the scene) — the fallback engages and `meta`
    says so, rather than raising, because an episode with no visible giver is a
    real situation and a crash there loses the whole rollout.
    """
    params = params or AnchorParams()
    state = state if state is not None else AnchorState()

    c = np.asarray(centroid, dtype=np.float64)
    base = np.asarray(robot_base, dtype=np.float64)

    if wrist is None:
        h, hn = np.zeros(3), 0.0
    else:
        h = horizontal(c - np.asarray(wrist, dtype=np.float64))
        hn = float(np.linalg.norm(h))
    state.history.append(hn)

    # Latch: cross `enter` going down to engage, cross `exit` going up to release.
    # Between the two, whatever was committed last keeps running -- that is the
    # entire point, and it is why this reads `state.fallback` before writing it.
    was = state.fallback
    if state.fallback:
        if hn > params.exit:
            state.fallback = False
    else:
        if hn < params.enter:
            state.fallback = True
    if state.fallback != was:
        state.switches += 1

    if state.fallback:
        h = horizontal(c - base)
        hn = float(np.linalg.norm(h))

    x = normalize(h)
    if float(np.linalg.norm(x)) < 0.5:
        # Both references collapsed: the object is directly over the robot base
        # AND over the wrist. Vanishingly unlikely, but a zero x would make the
        # frame singular and every bin label meaningless, so pick an arbitrary
        # but DETERMINISTIC horizontal axis and flag it loudly in meta.
        x = np.array([1.0, 0.0, 0.0])
        degenerate = True
    else:
        degenerate = False

    y = normalize(np.cross(WORLD_UP, x))
    R = np.stack([x, y, WORLD_UP], axis=1)      # columns
    return R, {"mode": state.mode, "horiz_norm": hn, "switched": state.fallback != was,
               "switches": state.switches, "degenerate": degenerate}


# ── env-aware helpers, kept separate so `anchor_rotation` stays portable ──────

def wrist_world(env):
    """The giver's wrist in world coords, or None when there is no hand.

    `env.mano.body` is None whenever the hand is not present: `MANO.reset` only
    builds the body when the frame is inside [sid, eid], and `step` tears it down
    at eid+1. Guarding is not defensive programming, it is the documented
    lifecycle. Mirrors the access pattern in `train_env._mano_hand_points_world`,
    including the torch-tensor branch.
    """
    mano = getattr(env, "mano", None)
    body = getattr(mano, "body", None) if mano is not None else None
    if body is None:
        return None
    ls = getattr(body, "link_state", None)
    if ls is None or len(ls) == 0:
        return None
    p = ls[0, MANO_WRIST_LINK, 0:3]
    return np.asarray(p.cpu().numpy() if hasattr(p, "cpu") else p, dtype=np.float64)


def handedness(env) -> str | None:
    """"left" | "right" | None. Stored per episode so the choice stays revisitable.

    The azimuth is NOT mirrored for left hands: DexYCB is ~50/50 (501 right, 499
    left across 1000 scenes), so the data covers both and mirroring would halve
    the effective diversity of the anchor. Recording it means that decision can be
    revisited from the collected data instead of by recollecting.
    """
    mano = getattr(env, "mano", None)
    body = getattr(mano, "body", None) if mano is not None else None
    name = getattr(body, "name", None) if body is not None else None
    if not name:
        return None
    return "left" if str(name).endswith("_left") else "right"


def object_centroid_world(pc_ee_xyz, ee_pose_world):
    """Object centroid in WORLD, from the EE-frame cloud and the EE pose.

    Only the anchor needs a world-frame centroid (to compare against a world-frame
    wrist). The per-point channels use the EE-frame centroid directly and never
    go through here -- see `channels.object_centroid`.
    """
    T = np.asarray(ee_pose_world, dtype=np.float64)
    p = np.asarray(pc_ee_xyz, dtype=np.float64)
    if p.size == 0:
        return None
    return T[:3, :3] @ p.mean(axis=0) + T[:3, 3]
