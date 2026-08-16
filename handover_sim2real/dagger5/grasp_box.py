"""
Geometric "was the object actually in the jaws" test, for the eval opportunity
metric.

WHY THIS EXISTS. The evaluator's original opportunity test (`had_chance`) asks
whether the EE was within (close_pos_thresh, close_rot_thresh) of the PINNED
grasp. Measured over runs 7/9/10 that reports a mean `chance_rate` of 0.033 /
0.026 / 0.049 while `success_rate` in those same runs peaks at 0.69 / 0.70 /
0.60 — it claims the policy essentially never had an opportunity in runs where it
secures the object two thirds of the time. The gate is dominated by the 0.34 rad
rotation tolerance that `eval_min_rot` never clears, so it is really measuring
agreement with the pin (the same thing `near_rate` measures), not opportunity.
Off-pose grasps are still grasps, and they are most of what the policy does.

WHAT THIS MEASURES INSTEAD. Purely local geometry, with no reference to the pin:
is there object material inside the cuboid swept between the two finger pads,
right now. That is true of an off-pose grasp and false of a policy parked next to
the object, which is exactly the discrimination the pinned test cannot make.

HOW. A grid of rays cast across the finger gap, from just inside one pad's inner
face to just inside the other's, using `rayTestBatch` on the live PyBullet
client. Ground-truth collision geometry, not the rendered point cloud: the wrist
camera sits ~10 cm from the object with the fingers straddling it at grasp range,
so a point-cloud version of this test would be visibility-conditioned in exactly
the configurations it is meant to count.

THE COLLISION-FILTER TRAP, and why the raycast needs a patch. handover-sim gives
each YCB object a one-bit collision group AND mask (`COLLISION_FILTER_YCB =
2**(i+1)`, config.py:90) so objects only collide with what they should. Bullet's
ray callback runs with group = DefaultFilter = bit 0 and tests

    (objGroup & rayMask) && (rayGroup & objMask)

and the second term is `1 & 2**(i+1)` = 0, so EVERY ray passes straight through
the object. This is silent — `rayTestBatch` returns "no hit" rather than an
error, which reads exactly like "no object in the jaws" and would have made the
metric a constant zero. pybullet's `collisionFilterMask` argument cannot fix it:
that sets the ray's mask (the first term), not its group.

So `_ycb_ray_visible` adds bit 0 to the object's MASK for the duration of the
cast and restores it in a `finally`. Its group is left alone. No
`stepSimulation` runs between the patch and the restore — the caller is between
policy steps — so no physics is ever computed under the widened mask and the
benchmark cannot observe it. The restore is unconditional, so an exception mid-
cast cannot leave the object colliding with the table (group 1) either.

The reported number is the FRACTION of the grid that hits the object, i.e. how
much of the CONTACT PAD has material behind it — a proxy for how much the
fingers would actually close on. A single ray clipping a corner is not an
opportunity in any useful sense, so callers threshold it (`min_frac`).

THE BOX IS THE PAD, NOT THE FINGER. An earlier version spanned the whole
53.7 mm finger blade, which diluted the signal roughly 2x with geometry that can
never touch anything: the blade is recessed everywhere proximal of its distal
17.6 mm, so material sitting in the throat of the jaws would be shoved by a
close rather than gripped. See BoxParams for the measured pad extent.

THE HUMAN HAND IS INVISIBLE TO THESE RAYS. MANO carries the same style of
one-bit collision filter as the object (`COLLISION_FILTER_MANO = 2**22`) and the
patch below widens ONLY the YCB object's mask, so rays pass straight through the
hand and it can never block one. This makes the metric a pure object test:
`box_chance_rate` can be true in a configuration where actually closing would
collide with the human, which is a leading failure mode here. Treat it as an
UPPER BOUND on real opportunity. Verified by firing a ray at every body's AABB
centre: plane/table/panda are ray-visible (their groups include bit 0), YCB and
MANO are not.

Those ray-visible bodies can block, by first-hit semantics — the table in
particular, if the jaws ever pass near it. Measured at a pinned grasp pose the
tally was {object: 18, nothing: 27} with no spurious blocker, which is expected
since the object is held in mid-air, but a jaw ray blocked by the table would
under-count.

FRAME. All box geometry is expressed in the `panda_hand` link frame (link 8),
the same frame the point cloud and the actions live in. From the URDF
(panda_gripper_hand_camera.urdf): both finger joints originate at z = 0.0584 and
travel along +/-y with an upper limit of 0.04, so the jaws open to 0.08 m total
and the pads run out to the TCP at z ~ 0.1034.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import pybullet


@dataclass
class BoxParams:
    """Jaw cuboid + the rule for calling one step an opportunity.

    The defaults describe the Panda finger pads and are not meant to be tuned
    per-run; `min_frac` and `open_thresh` are the two knobs that define the
    metric, and changing either changes what `box_chance_rate` means across runs.
    """

    # ---- what counts as an opportunity ----
    # Fraction of the ray grid that must hit the object — i.e. how much of the
    # CONTACT PAD has material behind it. 0.50 = 25 of 49 rays, half the pad,
    # a patch of roughly 12 x 12 mm on each face.
    #
    # This is a STRICT reading of "opportunity": the object must be squarely
    # presented across the pad, not merely clipping it. Measured over a 20-scene
    # eval of run 7's best checkpoint, that keeps 13 of the 14 episodes that
    # actually secured the object (92.9%); the one it drops peaked at 0.265, a
    # grasp that worked on a corner of the pad. On the failure side it halves the
    # false denominator: 33% of failures clear 0.50 against 67% at 0.20.
    #
    # The trade is deliberate — `box_chance_rate` now means "had a clean chance"
    # and slightly UNDER-counts, where a lower threshold over-counted marginal
    # ones. For reference the full picture on ideal pinned poses: at-grasp
    # coverage is mean 0.949 / p10 0.798 / min 0.592, and at the 6.4 cm standoff
    # it is exactly 0.000 in every scene.
    min_frac: float = 0.50
    # Gripper must still be OPEN for the step to be an opportunity — a step taken
    # mid-close is not a chance the policy is declining, and counting it would
    # inflate the denominator with states the policy has already committed from.
    # Normalised finger position (joint 7 / 0.04, as in _robot_state): 1 = open.
    open_thresh: float = 0.8

    # ---- the cuboid, in the panda_hand frame (metres) ----
    #
    # THIS IS THE CONTACT PAD, NOT THE FINGER. Measured off both
    # meshes/collision/finger.obj and meshes/visual/finger.obj, which agree: the
    # flat inner face — the only surface that can ever touch the object — spans
    # x in [-0.0088, +0.0088] and z in [0.0362, 0.0538] in finger-local coords,
    # i.e. 17.5 x 17.6 mm. It is square, and it sits at the TIP.
    #
    # Everything proximal of z = 0.036 is recessed: the minimum y on those
    # slices is 0.0116-0.0174, so the blade never reaches the object plane there.
    # Sampling it would count material the gripper cannot grip — an object
    # sitting in the throat of the jaws, which closing would shove rather than
    # hold. Adding the finger origin offset (0.0584) puts the pad at
    # z in [0.0946, 0.1122] in the hand frame.
    half_x: float = 0.0088    # half the pad width
    z_lo: float = 0.0946      # pad start  (0.0584 + 0.0362)
    z_hi: float = 0.1122      # pad end / fingertip (0.0584 + 0.0538)
    # Rays run along y, the finger travel axis, between the two inner faces.
    # Backed off by `inset` so a ray never starts inside a finger's own mesh
    # (a ray originating inside a body has undefined first-hit behaviour).
    inset: float = 0.002
    # Square region => square grid. 7 x 7 = 49 rays at 2.93 mm spacing, finer
    # than the 5 mm of the old finger-length box despite covering 1/6 the area.
    grid_x: int = 7           # fraction resolution 1/49 = 0.020
    grid_z: int = 7


def build_box_params(ev: dict) -> BoxParams:
    """BoxParams from an EVAL config block's optional `box:` sub-dict.

    Shared by train_dagger_phase4 and dagger/setup.py so the live loop and the
    standalone re-scorer cannot end up measuring against different cuboids —
    which would silently make a re-scored run incomparable to its own log.
    """
    bx = (ev or {}).get("box") or {}
    d = BoxParams()
    return BoxParams(
        min_frac=float(bx.get("min_frac", d.min_frac)),
        open_thresh=float(bx.get("open_thresh", d.open_thresh)),
        half_x=float(bx.get("half_x", d.half_x)),
        z_lo=float(bx.get("z_lo", d.z_lo)),
        z_hi=float(bx.get("z_hi", d.z_hi)),
        inset=float(bx.get("inset", d.inset)),
        grid_x=int(bx.get("grid_x", d.grid_x)),
        grid_z=int(bx.get("grid_z", d.grid_z)),
    )


def _bullet_client(env):
    """The live BulletClient behind the gym wrapper chain.

    `contact_id` and the body uids are PyBullet uids (easysim's bullet backend
    sets `body.contact_id = [self._body_ids[body.name]]`), so they index into
    this client directly. gym.Wrapper forwards `simulator` down to the
    easysim SimulatorEnv that owns it.
    """
    return env.simulator._p


@contextlib.contextmanager
def _ycb_ray_visible(p, *items):
    """Make the given (uid, easysim body) pairs visible to raycasts, then put
    them back exactly.

    See the module docstring: without this every ray silently passes through the
    object. The filter values are read off the easysim body rather than the
    config so the patch follows handover-sim's own switch to
    COLLISION_FILTER_YCB_RELEASE when the human lets go (ycb.py:release) — using
    the config constant would restore a pre-release filter onto a released
    object and quietly change what it collides with for the rest of the episode.

    The metric passes ONLY the YCB object. Passing the MANO hand as well is a
    viewer/diagnostic path (see examples/inspect_grasp_box.py): it changes what
    blocks a ray and so would change the metric's meaning.
    """
    saved = [(uid, [int(v) for v in body.get_attr_array("link_collision_filter", 0)])
             for uid, body in items]
    try:
        # Mirrors easysim's own loop (bullet.py:_set_link_collision_filter):
        # link index i takes vals[i + 1], base included at i = -1.
        for uid, vals in saved:
            for i in range(-1, len(vals) - 1):
                f = vals[i + 1]
                p.setCollisionFilterGroupMask(uid, i, f, f | 1)
        yield
    finally:
        for uid, vals in saved:
            for i in range(-1, len(vals) - 1):
                f = vals[i + 1]
                p.setCollisionFilterGroupMask(uid, i, f, f)


def _hand_frame_to_world(pts, pos, quat_xyzw):
    """[N,3] in the hand frame -> [N,3] world."""
    R = np.asarray(pybullet.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    return np.asarray(pts, dtype=np.float64) @ R.T + np.asarray(pos, dtype=np.float64)


def gripper_open_frac(env) -> float:
    """Normalised finger opening in [0, 1] (1 = fully open).

    Reads joint 7 and divides by the 0.04 travel limit — the same definition
    `_robot_state` puts in the policy's own observation, so "open" means here
    what it means to the network.
    """
    return float(env.panda.body.dof_state[0, 7, 0]) / 0.04


def jaw_rays(env, box: BoxParams, hand_pose=None, gap=None):
    """The ray grid itself: (src_world [N,3], dst_world [N,3]).

    Split out of `object_in_jaws_frac` so a viewer can DRAW exactly the rays the
    metric casts rather than a lookalike reimplementation that could drift from
    it. Returns (None, None) when the jaws are shut and there is no gap to span.

    `hand_pose` ((pos, quat_xyzw), world) and `gap` (finger half-opening, m)
    override the live robot state, which asks the counterfactual "would there be
    material in the jaws if the hand were HERE" without moving anything.
    """
    body = env.panda.body
    link = env.panda.LINK_IND_HAND

    # Live gap: the two finger joints translate along OPPOSITE y axes from a
    # shared origin, so the free span is [-q_right, +q_left]. Use the actual
    # joint values rather than the limit — a partially closed gripper has a
    # smaller box, which is the physically correct one to test.
    if gap is None:
        q_left = float(body.dof_state[0, 7, 0])
        q_right = float(body.dof_state[0, 8, 0])
    else:
        q_left = q_right = float(gap)
    y_hi = q_left - box.inset
    y_lo = -(q_right - box.inset)
    if y_hi <= y_lo:
        return None, None

    xs = np.linspace(-box.half_x, box.half_x, int(box.grid_x))
    zs = np.linspace(box.z_lo, box.z_hi, int(box.grid_z))
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    gx, gz = gx.reshape(-1), gz.reshape(-1)
    n_rays = gx.shape[0]

    src = np.stack([gx, np.full(n_rays, y_lo), gz], axis=-1)
    dst = np.stack([gx, np.full(n_rays, y_hi), gz], axis=-1)

    if hand_pose is None:
        hand_pos = body.link_state[0, link, 0:3].numpy()
        hand_quat = body.link_state[0, link, 3:7].numpy()      # xyzw
    else:
        hand_pos, hand_quat = hand_pose
    return (_hand_frame_to_world(src, hand_pos, hand_quat),
            _hand_frame_to_world(dst, hand_pos, hand_quat))


def jaw_ray_hits(env, box: BoxParams, hand_pose=None, gap=None):
    """(src_world, dst_world, hit_mask [N] bool, fraction).

    The full result of the cast, for callers that want to render it. `hit_mask`
    is True where the ray's FIRST hit was the YCB object. All-zeros arrays and
    0.0 when the jaws are shut.
    """
    p = _bullet_client(env)
    src_w, dst_w = jaw_rays(env, box, hand_pose=hand_pose, gap=gap)
    if src_w is None:
        z = np.zeros((0, 3))
        return z, z, np.zeros(0, dtype=bool), 0.0

    ycb_body = env.ycb.bodies[env.ycb.ids[0]]
    ycb_uid = int(ycb_body.contact_id[0])
    with _ycb_ray_visible(p, (ycb_uid, ycb_body)):
        hits = p.rayTestBatch(src_w.tolist(), dst_w.tolist())
    mask = np.array([int(h[0]) == ycb_uid for h in hits], dtype=bool)
    return src_w, dst_w, mask, float(mask.sum()) / float(mask.size)


def hand_ray_block(env, box: BoxParams, hand_pose=None, gap=None):
    """DIAGNOSTIC ONLY — never used by the metric. (hand_mask, fraction).

    Recasts the same grid with the MANO hand ALSO made ray-visible, and reports
    which rays hit the human first. This is the caveat the metric cannot see:
    `object_in_jaws_frac` leaves MANO invisible, so it will happily report a
    full jaw in a pose where closing would collide with the hand. Returns an
    empty mask when there is no hand this scene.
    """
    p = _bullet_client(env)
    src_w, dst_w = jaw_rays(env, box, hand_pose=hand_pose, gap=gap)
    mano = getattr(env, "mano", None)
    mano_body = getattr(mano, "body", None) if mano is not None else None
    if src_w is None or mano_body is None:
        return np.zeros(0, dtype=bool), 0.0

    ycb_body = env.ycb.bodies[env.ycb.ids[0]]
    mano_uid = int(mano_body.contact_id[0])
    with _ycb_ray_visible(p, (int(ycb_body.contact_id[0]), ycb_body),
                          (mano_uid, mano_body)):
        hits = p.rayTestBatch(src_w.tolist(), dst_w.tolist())
    mask = np.array([int(h[0]) == mano_uid for h in hits], dtype=bool)
    return mask, float(mask.sum()) / float(mask.size)


def object_in_jaws_frac(env, box: BoxParams, hand_pose=None, gap=None) -> float:
    """Fraction of the jaw cross-section with YCB object material behind it.

    0.0 when the jaws are shut (no gap to cast across) or the object is absent
    from the gap. Never raises on a degenerate gap — a closed gripper is simply
    not an opportunity.
    """
    return jaw_ray_hits(env, box, hand_pose=hand_pose, gap=gap)[3]


def grasp_opportunity(env, box: BoxParams) -> tuple[bool, float]:
    """(is this step an opportunity, jaw occupancy fraction).

    An opportunity requires BOTH that the jaws are still open and that enough of
    the object sits between them. The fraction is returned regardless so callers
    can log a continuous diagnostic and recalibrate `min_frac` after the fact
    without re-running the eval.
    """
    frac = object_in_jaws_frac(env, box)
    is_open = gripper_open_frac(env) >= box.open_thresh
    return bool(is_open and frac >= box.min_frac), frac
