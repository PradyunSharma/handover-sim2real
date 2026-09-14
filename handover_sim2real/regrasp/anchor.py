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

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Only `centroid_to_world` needs it, and only at call time -- the frame maths
# above stays importable with nothing but numpy.
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

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
                    params: AnchorParams | None = None, *,
                    reference: str = "hand"):
    """(R_anchor [3,3], meta) — columns are the anchor x, y, z in WORLD coords.

    `R_anchor @ v_anchor` gives world; `R_anchor.T @ v_world` gives anchor. Pass
    the same `state` for every step of an episode so the fallback latches.

    `wrist` may be None (no hand in the scene) — the fallback engages and `meta`
    says so, rather than raising, because an episode with no visible giver is a
    real situation and a crash there loses the whole rollout.

    `reference="base"` DROPS THE HAND FROM THE DEFINITION ENTIRELY and takes the
    azimuth from the robot instead:

        x = normalize(horizontal(p_base - c))       # object -> robot

    `wrist` is then unread, the hysteresis never runs, and `meta["mode"]` is
    "base_primary" -- distinct from "base", which means the HAND reference
    collapsed and the fallback caught it. Measured over 2208 run-11 episodes:

        reference        lever arm ||horiz||    1 cm centroid error moves x by
        robot base       61.3 cm  (min 41.8)          0.65 deg  (p95 1.03)
        MANO wrist       15.7 cm  (min  6.4)          2.53 deg  (p95 4.11)
        hand centroid     9.2 cm -> 7.7 cm at close   ~5 deg

    The object is always well out in front of the robot -- it has to be, or the
    arm could not reach it -- so `||horizontal(p_base - c)||` has a HARD FLOOR
    five times the `enter` threshold and the degenerate case is not rare, it is
    impossible. That is why the latch is SKIPPED rather than merely unlikely to
    fire. Under `anchor_update: live` the frame then rotates a median 1.49 deg
    over a whole episode (p99 7.67, 0.02% past 90 deg), against 10.65 deg median
    and 15.8% past 90 deg for the hand centroid -- i.e. live re-anchoring costs
    nothing here, which it does not under either hand reference.

    SIGN: `p_base - c`, not `c - p_base`. The robot and the giver face each
    other, so this points roughly where `horizontal(c - p_wrist)` did and the bin
    LABELS stay comparable with runs 1-16 -- measured, 7.7% of grasps change bin
    under this sign against 78.6% under the other. The `mode: "base"` FALLBACK
    keeps its own opposite convention (`c - base`); it is a different code path,
    reached only when the hand reference dies, and changing it would silently
    re-label every historical fallback episode.
    """
    params = params or AnchorParams()
    state = state if state is not None else AnchorState()

    c = np.asarray(centroid, dtype=np.float64)
    base = np.asarray(robot_base, dtype=np.float64)

    # ---- the hand-free frame: no reference to the giver, so no latch ---------
    if str(reference) == "base":
        h = horizontal(base - c)
        hn = float(np.linalg.norm(h))
        state.history.append(hn)
        x = normalize(h)
        degenerate = float(np.linalg.norm(x)) < 0.5
        if degenerate:
            # The object is directly over the robot base. Unreachable in this
            # workspace (measured floor 41.8 cm), but a zero x would make the
            # frame singular, so fail the same deterministic way the hand path
            # does rather than emitting a silent NaN.
            x = np.array([1.0, 0.0, 0.0])
        y = normalize(np.cross(WORLD_UP, x))
        return np.stack([x, y, WORLD_UP], axis=1), {
            "mode": "base_primary", "horiz_norm": hn, "switched": False,
            "switches": 0, "degenerate": degenerate}

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


def points_to_world(pts_ee, obs, panda_base_inv_tf, base_pos, base_quat):
    """`[N, 3]` EE-frame points -> WORLD. `centroid_to_world` for a whole cloud.

    `d_rule: location_extent` needs the OBJECT CLOUD, not just its centroid, to
    measure `r_u` — and it needs it in the frame the grasp pose and centroid are
    in, which is world. Batched rather than a loop over `centroid_to_world`
    because it runs per step under `anchor_update: live` and rebuilds the EE
    matrix each call otherwise.

    `m` is frame-invariant (a ratio of two projections onto one axis), so this
    is a convenience, not a correctness requirement — but mixing frames between
    the pose and the cloud IS a correctness bug, and having one function makes
    that mixing hard to write by accident.
    """
    from scipy.spatial.transform import Rotation as Rot
    from collect_bc_dataset import _ee_pose_mat

    p = np.asarray(pts_ee, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.zeros((0, 3))
    ee_mat = _ee_pose_mat(obs["panda_body"], obs["panda_link_ind_hand"],
                          panda_base_inv_tf)
    p_base = p @ ee_mat[:3, :3].T + ee_mat[:3, 3]
    R_base = Rot.from_quat(np.asarray(base_quat, dtype=np.float64)).as_matrix()
    return p_base @ R_base.T + np.asarray(base_pos, dtype=np.float64)


def object_points_world(pc5, obs, panda_base_inv_tf, base_pos, base_quat):
    """The OBJECT points of an EE-frame `[N, 5]` cloud, in world.

    OBJECT ONLY — the `ycb` channel, never `hand`. Including hand points lets the
    giver's forearm inflate the measured extent, which shrinks every `m` on that
    scene and does so by an amount that depends on how much of the arm the wrist
    camera happens to see.
    """
    from handover_sim2real.regrasp import channels as _channels

    p = np.asarray(pc5, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.zeros((0, 3))
    mask = p[:, _channels.CH_YCB] > 0.5
    if not mask.any():
        return np.zeros((0, 3))
    return points_to_world(p[mask, _channels.CH_XYZ], obs, panda_base_inv_tf,
                           base_pos, base_quat)


def points_to_world(p_ee, obs, panda_base_inv_tf, base_pos, base_quat):
    """EE-frame points `[N, 3]` -> WORLD `[N, 3]`, via the panda base.

    ONLY THE ANCHOR AND THE VIEWERS NEED THIS. The per-point channels are dot
    products, so they work entirely in the EE frame and never convert anything;
    the anchor is the one place that has to compare the centroid against a
    world-frame wrist, and a viewer is the other, to draw the cloud where the
    object is.

    The chain mirrors `_point_cloud`'s in reverse. `_ee_pose_mat` gives the EE in
    the panda BASE frame (that is the pose `se3_transform_pc` was applied with),
    so it is EE -> base -> world — the same round trip
    `rollout_regrasp_policy.draw_pointcloud` performs to overlay a cloud.

    BATCHED BECAUSE THE SCALAR FORM IN A LOOP IS PATHOLOGICAL: the EE pose and
    the base rotation do not depend on the point, and rebuilding both per point
    costs a thousand scipy calls to draw one cloud.
    """
    from scipy.spatial.transform import Rotation as Rot
    from collect_bc_dataset import _ee_pose_mat

    p = np.atleast_2d(np.asarray(p_ee, dtype=np.float64))
    ee_mat = _ee_pose_mat(obs["panda_body"], obs["panda_link_ind_hand"],
                          panda_base_inv_tf)
    p_base = p @ ee_mat[:3, :3].T + ee_mat[:3, 3]
    R_base = Rot.from_quat(np.asarray(base_quat, dtype=np.float64)).as_matrix()
    return p_base @ R_base.T + np.asarray(base_pos, dtype=np.float64)


def centroid_to_world(c_ee, obs, panda_base_inv_tf, base_pos, base_quat):
    """One EE-frame point -> WORLD. `points_to_world` for a single point."""
    return points_to_world(c_ee, obs, panda_base_inv_tf,
                           base_pos, base_quat)[0]


# ── the anchor from THIS step's observation (`SIM.anchor_update`) ─────────────

ANCHOR_UPDATES = ("latched", "live")

# WHICH POINT ON THE GIVER the azimuth is measured from (`SIM.anchor_hand_ref`).
#
#   wrist          `mano.body.link_state[0, 7]` — the MANO wrist JOINT centre.
#                  Exact, and available only in simulation.
#   hand_centroid  the centroid of the segmented hand POINT CLOUD, which is what
#                  `my_regrasp_policy_runner._set_direction` uses on the real rig
#                  (`class_centroid(fused.hand_xyz)`).
#
# THESE ARE DIFFERENT POINTS. The wrist joint sits at the base of the palm; the
# cloud centroid sits out in the visible middle of the hand, and only the visible
# part at that. `wrist` therefore makes the simulator compute an azimuth the
# robot cannot reproduce — a sim2real gap in the definition of the frame itself,
# not in the perception feeding it.
#   base           NO HAND AT ALL: x = normalize(horizontal(p_base - c)), the
#                  object -> robot azimuth. The only reference whose lever arm
#                  cannot collapse (61.3 cm median, 41.8 cm floor), so `live`
#                  re-anchoring rotates the frame 1.49 deg over an episode
#                  instead of the hand centroid's 10.65 deg median / 15.8%
#                  past 90 deg. It also needs nothing the real rig lacks --
#                  no wrist detector, no hand segmentation -- which is why it
#                  is the deployable choice as well as the stable one. Cost:
#                  "+x" stops meaning "away from the giver's fingers" and
#                  starts meaning "the side facing the robot"; the frame is
#                  then nearly scene-invariant (azimuth std 7.4 deg vs the
#                  wrist's 18.4). That costs nothing the POLICY sees -- the
#                  anchor never enters a network input, see the module
#                  docstring -- it only changes what a bin is NAMED.
ANCHOR_HAND_REFS = ("wrist", "hand_centroid", "base")


def anchor_from_cloud(pc5, obs, env, panda_base_inv_tf, cfg,
                      state: AnchorState | None = None, wrist=None,
                      hand_ref: str = "wrist"):
    """`(R_anchor, centroid_world, meta)` from the cloud observed at THIS step.

    THE ORIGIN MOVES EVEN WHEN NOTHING DOES. The camera is eye-in-hand, so the
    object's OBSERVED centroid is a function of where the gripper is looking
    from: at step 0 it is a distant, heavily self-occluded slice of the object
    and by the close it is a near view of the face the fingers are on. `c` is
    therefore not a property of the scene, and the anchor built from it at step 0
    is not the anchor a deployment would compute at step 20.

    `SIM.anchor_update: latched` keeps the historical behaviour — build the frame
    once at step 0 and hold it for the episode, which is exact only if `c` is
    stationary. `live` calls this every step, which is what a real rig has to do
    because it has no step-0 privilege and no ground truth to fall back on.

    Pass the SAME `state` for every step of an episode: the fallback latch is
    hysteretic (0.04 m in, 0.08 m out) precisely so a live frame cannot chatter
    between the wrist reference and the base one mid-approach.

    Returns `(None, None, meta)` when the cloud holds no object points — a real
    situation (`policy.py`'s occlusion note), and one the caller must handle by
    keeping the previous frame rather than by conditioning on nothing.
    """
    from handover_sim2real.regrasp import channels as _channels

    c_ee = _channels.object_centroid(pc5, fallback_to_all=False)
    if c_ee is None:
        return None, None, {"mode": None, "no_centroid": True}

    c_world = centroid_to_world(
        c_ee, obs, panda_base_inv_tf,
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)
    # THE GIVER REFERENCE. Under `hand_centroid` it comes from the SAME cloud the
    # object centroid came from, so both references move together as the view
    # changes and neither is ground truth — which is the situation on hardware.
    # Falls back to the wrist only when the cloud carries no hand points at all;
    # `anchor_rotation` then engages its own base fallback if that is None too.
    # `base` reads NO hand: not the MANO link, not the cloud's hand channel. The
    # `env` argument goes unused on this path, which is the point -- it is the
    # only reference the real rig can reproduce exactly.
    if str(hand_ref) == "base":
        wrist = None
    elif str(hand_ref) == "hand_centroid":
        h_ee = _channels.hand_centroid(pc5)
        wrist = (wrist_world(env) if h_ee is None else centroid_to_world(
            h_ee, obs, panda_base_inv_tf,
            cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION))
    elif wrist is None:
        wrist = wrist_world(env)
    R, meta = anchor_rotation(
        c_world, wrist, np.asarray(cfg.ENV.PANDA_BASE_POSITION), state,
        reference=("base" if str(hand_ref) == "base" else "hand"))
    meta["no_centroid"] = False
    meta["hand_ref"] = str(hand_ref)
    return R, c_world, meta
