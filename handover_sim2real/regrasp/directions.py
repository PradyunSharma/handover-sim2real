"""
The approach-direction conditioning manifold: bins, angles, and neighbourhoods.

`d` is a unit vector pointing **from the object outward toward where the gripper
will come from**. From a demonstration it is the negated gripper approach axis,
`d_world = -R_grasp[:, 2]`, rotated into the anchor frame (`anchor.py`).

THE SIGN IS MEASURED, NOT ASSUMED. Over the 486 scenes of
the Phase-5 candidate table holding four or more candidates, advancing 10 cm
along `+R[:, 2]` from the palm origin shrinks the spread of a scene's candidate
set from 0.0851 m to 0.0340 m — on 100% of scenes. `+z` therefore points *into*
the object, so `-R[:, 2]` is the outward direction this module talks about.

`d` IS FLIP-INVARIANT FOR FREE, WHICH MAKES A WHOLE CLASS OF BUG IMPOSSIBLE.
`augment_flip_grasp` appends twins rotated pi about the gripper's OWN +z, which
negates columns 0 and 1 of R and leaves column 2 untouched — so `d` is IDENTICAL
for a grasp and its twin, measured at max 0.000e+00 over 2000 random poses.

A pose-space metric does not have that property: the twins sit at SE(3) rotation
distance pi, so a max-min selector ranks them as maximally separated and will
select two commands that are one physical grasp, leaving the dataset holding two
contradictory rotation labels for one target. `grasp_pin.py` still carries the
rotation-disambiguation code that existed to catch it, and the measured
signature (episodes closing 3.1413 rad from their pin while p99 was 0.0029).
Under direction conditioning it cannot arise — the twins are the same command.

WHY k = 6 OCTAHEDRAL. Exact optimal packing for six points on a sphere: 90 deg
minimum separation, comfortably above the ~40 deg below which bins stop being
independent retry hypotheses. And every bin has a physical meaning, which the k=7
and k=8 optimal packings do not — they align with no axis. Bins are a FIXED
CONSTANT OF THE SYSTEM: the hand-object configuration affects the feasibility mask
and the ranking, never the bin positions.

k IS A TEST-TIME KNOB, WITH A MEASURED CEILING AT 20. The conditioning the network
reads is a continuous unit vector, never a one-hot index, so nothing in a trained
policy depends on `k`. Going from 6 bins to 12 after a bad evaluation is
`fibonacci_directions(12)` and costs no retraining and no relabelling. But the
knob has a limit: minimum separation falls off as `k` grows, and below ~40 deg
bins stop being independent retry hypotheses —

    k        6     8    10    12    14    20    21
    min sep 78.1  66.3  58.5  52.9  48.7  41.0  39.4 deg

so k = 20 is the last usable setting and k = 21 is not. Note also that the
Fibonacci lattice at k = 6 gives only 78.1 deg against the octahedron's 90 deg,
and loses the per-axis interpretability with it — at k = 6, `BINS` is strictly
better and `fibonacci_directions` is for k != 6 only.

MEASURED: THE `-z` BIN IS EMPTY ON THIS DATASET. Anchor `z` IS world up, so `d.z`
is frame-independent and the +-z bins are computable from the Phase-5 candidate
candidate table with no wrist data at all. Over 3810 candidates, 623 scenes:

    -z within 45 deg     0   ( 0.0%)      scenes with any:   0 / 623
    below-ish 45-70    154   ( 4.0%)
    lateral 70-110    1737   (45.6%)
    above-ish 110-135 1029   (27.0%)
    +z within 45 deg   890   (23.4%)      scenes with any: 402 / 623

`-z` is not starved, it is ABSENT — the object is held above a table and OMG's IK
plus collision filtering rejects every from-beneath approach. The measurement is
over the FPS-selected 8 of a median-49 goal set, and FPS *maximises* diversity, so
a direction present anywhere would be over-represented in that subsample, not
under; zero here is strong evidence of zero overall. `BIN_MINUS_Z` is kept anyway
— on a different rig, or a table-less setup, it is real — but expect the
feasibility mask to exclude it on every scene here. Confirmed against the FULL
goal set: -z 0/623 scenes and -x 12/623, so FOUR live hypotheses, not six. Confirm against the FULL goal set when the
direction table is built.

Pure numpy. No sim imports, no torch, no h5py — so this runs on a login node, in
a DataLoader worker, and eventually on the robot PC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Order is fixed and load-bearing: `bin_idx` is stored in every episode attr and
# keyed in the grasp registry, so reordering these silently reinterprets recorded
# data. Append, never insert.
BINS = np.array([
    [+1.0, 0.0, 0.0],    # 0  +x  free end — maximum finger clearance
    [-1.0, 0.0, 0.0],    # 1  -x  over the giver's fingers
    [0.0, +1.0, 0.0],    # 2  +y  lateral
    [0.0, -1.0, 0.0],    # 3  -y  lateral
    [0.0, 0.0, +1.0],    # 4  +z  top-down
    [0.0, 0.0, -1.0],    # 5  -z  from beneath   (measured empty — see the header)
], dtype=np.float64)

BIN_NAMES = ("+x_free_end", "-x_over_fingers", "+y_lateral", "-y_lateral",
             "+z_top_down", "-z_beneath")

BIN_PLUS_X, BIN_MINUS_X, BIN_PLUS_Y, BIN_MINUS_Y, BIN_PLUS_Z, BIN_MINUS_Z = range(6)

# THE FOUR BINS THIS DATASET CAN ACTUALLY REACH — a property of s0/train, NOT of
# the system. `-x` is demonstrable by 12 of 623 scenes and `-z` by none (see the
# header), so those two collect no episodes, score NaN, and would render as four
# blank panels in any per-bin figure. Reporting and plotting default to this
# tuple so a figure has one row per direction that exists; nothing in the retry
# machine, the collector or the model reads it, and on a table-less rig the right
# fix is to re-measure, not to edit this line.
LIVE_BINS = (BIN_PLUS_X, BIN_PLUS_Y, BIN_MINUS_Y, BIN_PLUS_Z)

# Short labels for figure titles and CSV headers — `BIN_NAMES` carries the
# rationale in the name and is too long for a 3.5-inch axis.
BIN_SHORT = ("+x", "-x", "+y", "-y", "+z", "-z",
             "0")     # index BIN_NULL: no location preference (location_extent only)

# The Voronoi half-angle for 90-deg-separated bins is 45 deg. `bin_hit_rate` uses
# 30 to keep margin against boundary noise, so a "hit" is unambiguous rather than
# a coin flip between two adjacent bins.
BIN_HIT_DEG = 30.0

# THE NULL BIN — "this grasp expresses no location preference".
#
# Only `location_extent` produces it. `d = m * u` with `m` the eccentricity, and
# a grasp AT the centroid has m ~ 0: there is no side to name, and forcing it
# onto whichever octahedral axis its noise happens to point at would put a
# confident label on a quantity that is not there. That is the failure the old
# `grasp_offset` rule had by construction — `normalize(p_grasp - c)` is undefined
# as the offset vanishes, so the direction flipped under millimetre
# perturbations. Retaining the magnitude fixes the geometry; this bin is what
# makes the BOOKKEEPING honest about it.
#
# 6, i.e. one past the octahedral set, so `BINS` stays exactly the six axes and
# every `range(len(BINS))` loop keeps its meaning. `BIN_SHORT` carries a seventh
# label for printing; nothing iterates its length (checked).
#
# NULL-BIN DEMONSTRATIONS ARE LABELS ONLY. They are legitimate training data —
# "grasp near the middle" is a real instruction — but the retry ladder must never
# COMMAND the null bin, because "no preference" is not a hypothesis to fall back
# to. `RETRY_LADDER` is over `BINS` and so cannot reach it.
BIN_NULL = 6

# BELOW THIS, `d` IS ZERO — no direction, not a short one.
#
# It replaces the `np.linalg.norm(d) < 0.5` test that used to mean "is this a
# valid unit vector", which was correct while every `d` was unit and is WRONG
# under `location_extent`: a legitimate command there can have magnitude 0.05,
# and 0.5 would discard four fifths of the achievable range as invalid.
D_ZERO_EPS = 1e-6


def bin_label(b) -> str:
    """`BIN_SHORT[b]`, plus a name for the null bin and for "unassigned"."""
    b = -1 if b is None else int(b)
    if b == BIN_NULL:
        return "0"                      # no location preference
    return BIN_SHORT[b] if 0 <= b < len(BIN_SHORT) else "--"

# The retry machine's neighbour-exclusion radius. See `neighbours` — for the
# octahedral set this is a no-op, and that is a fact worth knowing rather than a
# reason to change it.
NEIGHBOUR_DEG = 30.0

# ── THE RETRY LADDER: which direction to try, in what order ──────────────────
#
# `+x, +z, +y, -y, -z, -x` — bin indices (0, 4, 2, 3, 5, 1).
#
# ORDERED BY MEASURED PER-BIN SUCCESS, not by index. Runs 1-15 walked the pin
# table's SLOT order, which is ascending bin index, and that is arbitrary with
# respect to how good each direction is: `-x` is bin 1, so the WORST bin
# (0.333 on run 11 it 22) was tried SECOND whenever a scene had it, while `+z`
# (0.737) sat at index 4 and was tried nearly last. Ordering by success is the
# whole point of a ladder — the first rung should be the direction most likely
# to work.
#
# The order is from run 11 iteration 22, which is the largest well-measured
# per-bin sample under `grasp_offset`:
#
#     +x 0.778 (n=54)   +z 0.737 (n=19)   +y 0.649 (n=37)
#     -y 0.600 (n=25)   -z 0.538 (n=26)   -x 0.333 (n=33)
#
# A FIXED ORDER MAKES RUNG k A DEFINITE DIRECTION. Under the slot walk, rung k
# was a mixture across scenes (rung 1 was `+x` on only 54% of them), so the
# curve could not be read as "then it tried +y". With this it can, and
# `retry_bin_frac_k` collapses to 1.0 by construction rather than reporting a
# mixture.
#
# IT IS A CONSTANT, NOT A FIT. Re-deriving it per run from that run's own
# per-bin rates would make the ladder a function of the thing it is scoring, so
# `retry_at_k` would improve whenever the ordering got luckier — which is not a
# property of the policy. If the ranking is ever revisited it should move here,
# deliberately, and the runs either side of the change stop being comparable.
RETRY_LADDER = (0, 4, 2, 3, 5, 1)


def normalize(v, axis=-1, eps: float = 1e-12):
    """Unit vector(s). Zero-length input returns zeros rather than NaN.

    Returning zeros is deliberate: a zero direction is detectable downstream
    (`np.linalg.norm(d) < 0.5`) whereas a NaN propagates silently into `d.n`,
    then into every point of the cloud, then into the loss.
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > eps)


def approach_direction(grasp_pose) -> np.ndarray:
    """`d` for a 4x4 grasp pose, in whatever frame the pose is expressed in.

    Identical for a grasp and its wrist-flip twin (see the header), which is why
    no symmetry quotient is needed anywhere in this module.
    """
    return normalize(-np.asarray(grasp_pose, dtype=np.float64)[:3, 2])


# ── WHAT `d` IS DERIVED FROM (`SIM.d_rule`) ─────────────────────────────────
#
# Two rules, and they do NOT measure the same thing. This is the deepest knob in
# the phase — it changes what a bin MEANS, which bins are populated, and what a
# demonstration for a bin looks like — so it is recorded in the direction
# table's `_meta` and every downstream stage refuses a table built under a
# different one.
#
#   approach_axis   `d = -R_grasp[:,2]`. "WHICH SIDE THE GRIPPER COMES FROM."
#                   Runs 1-9. Flip-invariant for free (see the header), and
#                   independent of where the object's centroid is estimated to
#                   be. Its limitation is that it is a property of the gripper's
#                   ORIENTATION alone: two grasps that approach along the same
#                   axis but close on opposite ends of a long object are the
#                   same command.
#
#   grasp_offset    `d = normalize(T_grasp @ [0,0,depth] - c)`, the direction
#                   from the anchor origin (the object point-cloud centroid) to
#                   the midpoint between the fingertips. "WHICH PART OF THE
#                   OBJECT THE FINGERS CLOSE ON." Depends only on the grasp's
#                   POSITION relative to the object, never on its orientation —
#                   the same closure point reached by a rotated wrist is the
#                   same command.
#
# MEASURED ON s0/train (3477 grasps in direction_table_train.json, 623 scenes),
# and the difference is not a refinement, it is a different question:
#
#                        +x     -x     +y     -y     +z     -z   within 45 deg
#   approach_axis      1162     20    822    684    789      0      100.0%
#   grasp_offset        714    392    637    450    418    526       90.2%
#
#   scenes reaching:    490     12    365    324    415      0   (approach_axis)
#                       248    191    241    205    142    235   (grasp_offset)
#
# `-z` GOES FROM DEAD TO THE THIRD-LARGEST BIN, and `-x` from 12 scenes to 191.
# That is physically right rather than a bug: you cannot APPROACH an object from
# beneath when it is held over a table, but you can perfectly well close your
# fingers on its UNDERSIDE. So under `grasp_offset` the retry ladder has six
# live rungs instead of four and `chained_retry_at_k` saturates at 6.
#
# THE COST IS CONDITIONING. `depth` puts the point between the fingers, i.e. ON
# the object, so `tip - c` is SHORT: a median 3.85 cm, and 14.4% of grasps are
# under 2 cm — where the direction is dominated by centroid noise, and the
# centroid comes from a partial, occluded cloud. `min_offset` exists to drop
# those at table-build time. `-R[:,2]` has no such failure mode, which is the
# honest argument for the older rule.
#
# THE PALM ORIGIN (`depth: 0.0`) IS A THIRD OPTION AND IT IS NEARLY THE OLD ONE:
# median 12.48 cm from the centroid (min 4.55, never degenerate) and a median
# 14.49 deg from `-R[:,2]`. Position-derived, but it answers the approach_axis
# question. Use it if `grasp_offset` at fingertip depth proves too noisy and the
# point is only to remove the orientation dependence.
D_RULES = ("approach_axis", "grasp_offset", "location_extent")

# Metres along the gripper's local +z. The Panda's finger pads span z in
# [0.0946, 0.1122] (grasp_box.py, from meshes/collision/finger.obj: the finger
# joints originate at 0.0584 and the pad runs 0.0362..0.0538 in finger-local
# coordinates), so 0.1122 is the FINGERTIP END and 0.1034 the pad centre.
# GA-DDPG's own control points put the fingertips at 0.105. The midpoint BETWEEN
# the two fingertips lies on the local z axis by symmetry, whichever axis the
# fingers travel along, so only the depth has to be named.
FINGERTIP_DEPTH = 0.1122


def grasp_point(grasp_pose, depth: float = FINGERTIP_DEPTH) -> np.ndarray:
    """The point on the gripper's own axis at `depth` metres, in world.

    `depth=0` is the palm origin (`T[:3,3]`); `FINGERTIP_DEPTH` is the midpoint
    between the fingertips, which is where the object ends up.
    """
    T = np.asarray(grasp_pose, dtype=np.float64)
    return T[:3, 3] + float(depth) * T[:3, 2]


def grasp_direction(grasp_pose, centroid_world=None, rule: str = "approach_axis",
                    depth: float = FINGERTIP_DEPTH, min_offset: float = 0.0):
    """`d` in WORLD for one grasp, under `rule`. The single definition.

    Returns None when the rule cannot be evaluated — no pose, no centroid under
    `grasp_offset`, or an offset shorter than `min_offset`. None is deliberate
    and must not be turned into zeros by a caller: a zero direction reads to the
    network as "approach from nowhere" on every point of the cloud, which is a
    wrong label rather than a missing one.

    NOTE `grasp_offset` IS NOT FLIP-INVARIANT-BY-ACCIDENT, IT IS FLIP-INVARIANT
    BY CONSTRUCTION AND MORE. `augment_flip_grasp` rotates pi about the
    gripper's own +z, which moves neither `T[:3,3]` nor `T[:3,2]`, so the point
    and hence `d` are identical for a grasp and its twin — the same property
    `-R[:,2]` has. It is additionally invariant to ANY rotation about the
    approach axis and to wrist roll generally, which is the point of the rule.
    """
    if grasp_pose is None:
        return None
    if str(rule) not in D_RULES:
        raise ValueError(f"d_rule must be one of {list(D_RULES)}, got {rule!r}")
    if str(rule) == "approach_axis":
        return approach_direction(grasp_pose)
    if centroid_world is None:
        return None
    v = grasp_point(grasp_pose, depth) - np.asarray(centroid_world, dtype=np.float64)
    if float(np.linalg.norm(v)) < max(float(min_offset), 1e-9):
        return None
    return normalize(v)


# ── `location_extent`: d = m * u, the EXTENT-NORMALIZED LOCATION COMMAND ─────
#
# WHAT IT COMMANDS AND WHY THAT IS THE RIGHT THING TO COMMAND. The other two
# rules name the gripper's approach ORIENTATION. But orientation is largely
# determined by local geometry — a parallel jaw closes across the locally narrow
# dimension — so a policy can in principle read it off the point cloud. LOCATION
# is what the observation genuinely underdetermines: on an L-shaped object,
# "approach from above" says nothing about which limb to take, and nothing in the
# cloud resolves it either. Conditioning should carry what the observation lacks.
#
# It is also the factorization the grasp-detection literature converged on —
# GraspNet samples seed POINTS then predicts approach vectors per seed;
# Where2Act predicts per-point actionability then orientations at that point.
# Location outer, orientation conditional.
#
#     v   = grasp_point(T, depth) - c
#     u   = v / |v|                       the direction  (unit)
#     r_u = percentile((obj - c) @ u, 95) the object's extent ALONG u
#     m   = clip(|v| / r_u, 0, 1)         the eccentricity
#     d   = m * u                         NOT a unit vector
#
# `m = 0` is a grasp at the centroid, `m = 1` one at the object's extremity in
# that direction.
#
# WHY THE EXTENT AND NOT ONE RADIUS. Normalizing an anisotropic quantity by an
# isotropic scalar makes `m = 1` unreachable in most directions. With
# `R_obj = max|p_i - c|` a 6:1 banana reaches 1 along its long axis but caps near
# 0.17 laterally, so a lateral unit command asks for a location six times further
# out than the object extends — and the cap varies per object AND per direction,
# so the same commanded number means a different physical place on every scene.
# Dividing by the extent along `u` makes `m = 1` achievable in every direction by
# construction, which is what makes the magnitude comparable across bins at all.
#
# WHY RETAINING THE MAGNITUDE FIXES THE FLIP INSTABILITY. `grasp_offset` is
# `normalize(p_grasp - c)`, which is UNDEFINED as the offset vanishes: a grasp
# near the centroid projects a vanishing vector onto the unit sphere, so its
# direction is noise and flips under millimetre perturbations. Keeping the
# magnitude makes a centroid grasp `d ~ 0`, which is stable AND semantically
# right — both conditioning channels go uniformly near zero, which tells the
# policy "no side preference", which is true. Sliding from the middle of a banana
# toward one end then grows the signal smoothly from zero instead of flipping it.
# The discontinuity was in the normalization, never in the geometry.
#
# THREE DETAILS THAT DECIDE WHETHER THIS WORKS:
#
#   95th percentile, NOT max.   One depth outlier at the object silhouette
#                               otherwise sets the scale for the whole scene.
#   The OBSERVED cloud, always. A partial view underestimates the extent for
#                               back-facing directions. That bias is identical at
#                               collection, training and deployment, so it
#                               cancels. A mesh or a fused cloud would remove the
#                               bias and reintroduce a privileged signal the
#                               robot does not have — strictly worse.
#   OBJECT points only.         The `ycb` channel, never the `hand` one. Include
#                               hand points and the giver's forearm inflates the
#                               extent, shrinking every `m` on that scene.

def extent_along(obj_points, centroid, u, pct: float = 95.0) -> float:
    """The object's extent from `centroid` along unit `u`, in the cloud's frame.

    `percentile(..., 95)` of the projections rather than the max: the cloud is a
    single-frame depth observation and its silhouette carries outliers, and one
    of them setting `r_u` would rescale every `m` on the scene.

    Returns 0.0 when there are no points, or when the observed cloud has no
    positive extent along `u` at all — which happens for a genuinely back-facing
    direction on a heavily occluded view. The caller must treat 0.0 as "not
    measurable" rather than dividing by it.
    """
    p = np.asarray(obj_points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return 0.0
    proj = (p - np.asarray(centroid, dtype=np.float64)) @ \
        np.asarray(u, dtype=np.float64)
    return float(max(np.percentile(proj, float(pct)), 0.0))


def location_command(grasp_pose, centroid, obj_points, *,
                     depth: float = FINGERTIP_DEPTH, m_min: float = 0.0,
                     pct: float = 95.0):
    """`(d, m, u, info)` for `location_extent`. `d = m * u`, and is NOT unit.

    `m = 0` with `u = 0` means "no location preference": either the grasp sits on
    the centroid, or the extent along `u` could not be measured. Both are
    reported in `info` so the two are never conflated in a census.

    FRAME-AGNOSTIC, LIKE THE OTHER RULES. `m` is a ratio of two projections onto
    the same axis and `u` is a direction, so any frame works provided
    `grasp_pose`, `centroid` and `obj_points` are ALL in it. The collector passes
    world; nothing here converts.
    """
    zero = (np.zeros(3), 0.0, np.zeros(3))
    if grasp_pose is None or centroid is None or obj_points is None:
        return (*zero, {"reason": "missing input"})
    c = np.asarray(centroid, dtype=np.float64)
    v = grasp_point(grasp_pose, depth) - c
    nv = float(np.linalg.norm(v))
    if nv < D_ZERO_EPS:
        # A grasp exactly on the centroid. Genuinely null, not an error.
        return (*zero, {"reason": "at centroid", "offset": nv})
    u = v / nv
    r_u = extent_along(obj_points, c, u, pct)
    if r_u < D_ZERO_EPS:
        # The observed cloud has no extent along `u`: a back-facing direction on
        # an occluded view. `m` is not measurable, so refuse rather than invent
        # one — an unnormalized `m` would be a different quantity from every other
        # episode's and would silently widen the bin's magnitude distribution.
        return (*zero, {"reason": "no extent along u", "offset": nv, "r_u": r_u})
    # `v @ u` IS `nv` by construction (u = v/nv); written as the ratio because
    # that is what the quantity means.
    m = float(np.clip(nv / r_u, 0.0, 1.0))
    info = {"reason": "ok", "offset": nv, "r_u": r_u, "m_raw": nv / r_u,
            "clipped": bool(nv / r_u > 1.0)}
    if m < float(m_min):
        # Below `m_min` the eccentricity is real but too small to name a side.
        # `u` is returned anyway so a census can see WHICH way the sub-threshold
        # grasp leaned, but `d` is zeroed so the label carries no preference.
        info["reason"] = "below m_min"
        return np.zeros(3), m, u, info
    return m * u, m, u, info


@dataclass(frozen=True)
class DirectionRule:
    """`SIM.d_rule` + its two numbers, resolved once and carried as one object.

    Frozen and plain-numeric so it pickles to the collection and eval workers,
    and so a run's record (`<run>/config.yaml`, the direction table's `_meta`)
    can be compared for equality rather than field by field. `from_cfg` is the
    ONE place a config block turns into this, so a stage that forgets a default
    cannot silently disagree with the stage before it.
    """

    rule: str = "approach_axis"
    depth: float = FINGERTIP_DEPTH
    min_offset: float = 0.0
    # ---- `location_extent` only -------------------------------------------
    # `m_min`: below this eccentricity the grasp goes to the NULL bin and its `d`
    # is zeroed. 0.15 is the starting value — a grasp within 15% of the way to
    # the object's extremity along its own direction expresses no side.
    m_min: float = 0.15
    # The percentile that defines "the object's extent". 95, not 100: see
    # `extent_along`.
    extent_pct: float = 95.0

    def __post_init__(self):
        if self.rule not in D_RULES:
            raise ValueError(f"SIM.d_rule must be one of {list(D_RULES)}, "
                             f"got {self.rule!r}")
        if not 0.0 <= self.m_min < 1.0:
            raise ValueError(f"SIM.d_m_min must be in [0, 1), got {self.m_min}")
        if not 50.0 <= self.extent_pct <= 100.0:
            raise ValueError(f"SIM.d_extent_pct must be in [50, 100], "
                             f"got {self.extent_pct}")

    @classmethod
    def from_cfg(cls, block):
        b = block or {}
        return cls(rule=str(b.get("d_rule", "approach_axis")),
                   depth=float(b.get("d_point_depth", FINGERTIP_DEPTH)),
                   min_offset=float(b.get("d_min_offset", 0.0)),
                   m_min=float(b.get("d_m_min", 0.15)),
                   extent_pct=float(b.get("d_extent_pct", 95.0)))

    # ---- what KIND of vector this rule produces ----------------------------
    @property
    def unit(self) -> bool:
        """Is `d` a unit vector? False for `location_extent`, whose MAGNITUDE is
        half the command. Everything that touches `d` on the way to the network
        has to ask: `append_direction_channels` normalized unconditionally, which
        would have discarded the eccentricity silently and left the two channels
        identical to `grasp_offset`'s."""
        return self.rule != "location_extent"

    def needs_centroid(self) -> bool:
        return self.rule in ("grasp_offset", "location_extent")

    def needs_points(self) -> bool:
        """Does evaluating this rule need the OBJECT POINT CLOUD? Only
        `location_extent`, for `r_u`. Callers that cannot supply one get None
        rather than a silently un-normalized magnitude."""
        return self.rule == "location_extent"

    def of(self, grasp_pose, centroid_world=None, obj_points=None):
        """This rule's `d` for one grasp, or None. See `grasp_direction`.

        Under `location_extent` a NULL result (grasp at the centroid, or an
        unmeasurable extent) comes back as a ZERO VECTOR, not None: zero is the
        correct label there — "no side preference" — whereas None means "this
        rule could not be evaluated" and must not be trained on. `decompose`
        returns the two apart when the caller needs to tell them apart.
        """
        if self.rule != "location_extent":
            return grasp_direction(grasp_pose, centroid_world, self.rule,
                                   self.depth, self.min_offset)
        if grasp_pose is None or centroid_world is None or obj_points is None:
            return None
        d, _m, _u, _info = location_command(
            grasp_pose, centroid_world, obj_points,
            depth=self.depth, m_min=self.m_min, pct=self.extent_pct)
        return d

    def decompose(self, grasp_pose, centroid_world=None, obj_points=None):
        """`(d, m, u, info)`. For the two unit rules `m` is 1 and `u` is `d`, so
        a caller can store `m`/`u` unconditionally and a `location_extent` shard
        stays readable by the same code."""
        if self.rule == "location_extent":
            return location_command(grasp_pose, centroid_world, obj_points,
                                    depth=self.depth, m_min=self.m_min,
                                    pct=self.extent_pct)
        d = self.of(grasp_pose, centroid_world)
        if d is None:
            return None, float("nan"), None, {"reason": "rule not evaluable"}
        return d, 1.0, d, {"reason": "ok"}

    def bin_of(self, d, bins=None) -> int:
        """The bin `d` names, `BIN_NULL` when it names none.

        THE NULL CASE IS WHY THIS IS A METHOD. `directions.bin_of` is an argmin
        over angles and will happily return a confident bin for a 1e-9-magnitude
        vector, because `angles_to_bins` normalizes first. Under
        `location_extent` that is exactly the wrong answer: the whole point of
        keeping the magnitude is that a near-centroid grasp has no side, and the
        argmin would name whichever axis its noise leans toward.
        """
        if d is None:
            return -1
        if self.rule == "location_extent" and \
                float(np.linalg.norm(np.asarray(d, dtype=np.float64))) < D_ZERO_EPS:
            return BIN_NULL
        return bin_of(d, bins)

    def as_meta(self) -> dict:
        meta = {"d_rule": self.rule, "d_point_depth": self.depth,
                "d_min_offset": self.min_offset}
        if self.rule == "location_extent":
            meta.update({"d_m_min": self.m_min,
                         "d_extent_pct": self.extent_pct})
        return meta

    def describe(self) -> str:
        if self.rule == "approach_axis":
            return "approach_axis (d = -R_grasp[:,2])"
        if self.rule == "location_extent":
            return (f"location_extent (d = m * u, m = |p_grasp - c| / "
                    f"extent_p{self.extent_pct:g}(u), point at "
                    f"{self.depth*100:.2f} cm, null below m={self.m_min:g}) "
                    f"-- NOT a unit vector")
        return (f"grasp_offset (d = centroid -> gripper point at "
                f"{self.depth*100:.2f} cm"
                + (f", min {self.min_offset*100:.1f} cm" if self.min_offset else "")
                + ")")


def angle_between(a, b) -> np.ndarray:
    """Angle in DEGREES between unit vectors, broadcasting over leading axes."""
    a, b = normalize(a), normalize(b)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def angles_to_bins(d, bins=None) -> np.ndarray:
    """[k] angles in degrees from `d` to every bin."""
    bins = BINS if bins is None else np.asarray(bins, dtype=np.float64)
    return angle_between(np.asarray(d, dtype=np.float64)[..., None, :], bins)


def bin_of(d, bins=None) -> int:
    """Nearest bin index. Ties break toward the lower index, deterministically.

    This is only ever used for BOOKKEEPING — assignment, histograms, the retry
    machine's memory. The network is conditioned on the continuous vector and
    never sees a bin index, so a `d` sitting near a Voronoi boundary produces a
    debatable label and an unaffected input.
    """
    return int(np.argmin(angles_to_bins(d, bins)))


def neighbours(idx: int, deg: float = NEIGHBOUR_DEG, bins=None) -> set:
    """Bins within `deg` of bin `idx`, INCLUDING `idx` itself.

    For the octahedral set every pair is 90 deg apart, so at the default 30 deg
    this returns exactly `{idx}` — the exclusion removes only the attempted bin.
    That is worth stating rather than discovering: it means the retry machine's
    "no bins left" stop condition is reached only after all six have been tried,
    and the 30 deg radius is insurance for a future `k` where bins really do
    crowd, not something doing work today.
    """
    bins = BINS if bins is None else np.asarray(bins, dtype=np.float64)
    return set(np.flatnonzero(angle_between(bins[idx], bins) <= deg).tolist())


def exclude(attempted, deg: float = NEIGHBOUR_DEG, bins=None) -> list:
    """Surviving bin indices after removing `attempted` and their neighbourhoods.

    `attempted` is a list of DIRECTION VECTORS, not indices — see `retry.py`. A
    bin index is not a durable identity across an anchor-frame change, and the
    anchor moves whenever the human does.
    """
    bins = BINS if bins is None else np.asarray(bins, dtype=np.float64)
    dead: set = set()
    for d in attempted:
        dead |= neighbours(bin_of(d, bins), deg, bins)
    return [i for i in range(len(bins)) if i not in dead]


def furthest_from(attempted, available, bins=None) -> int:
    """The available bin maximising the minimum angle to anything attempted.

    This is the whole ranking function, deliberately: no clearance term, no
    weights. Any gain in `chained_retry_at_k` is then attributable to the policy
    rather than to a clever ranker. With nothing attempted yet the first
    available bin is returned, so the caller's preferred bin should lead the list.
    """
    bins = BINS if bins is None else np.asarray(bins, dtype=np.float64)
    if not len(available):
        raise ValueError("no available bins")
    if not len(attempted):
        return int(available[0])
    att = normalize(np.asarray(attempted, dtype=np.float64).reshape(-1, 3))
    scores = [float(angle_between(bins[i], att).min()) for i in available]
    return int(available[int(np.argmax(scores))])


def fibonacci_directions(k: int) -> np.ndarray:
    """[k, 3] near-uniform directions on the sphere, for any `k`.

    The escape hatch that makes `k` a test-time knob. Swap these in for `BINS`
    and nothing about a trained policy changes — the conditioning is a continuous
    vector, so a new bin is just a new query point on the same manifold. Note the
    k=6 lattice is NOT the octahedron and loses the per-axis interpretability, so
    prefer `BINS` at k=6.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    i = np.arange(k, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / k
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def to_world(d_anchor, R_anchor) -> np.ndarray:
    """Anchor-frame direction -> world."""
    return normalize(np.asarray(R_anchor, dtype=np.float64)
                     @ np.asarray(d_anchor, dtype=np.float64))


def from_world(d_world, R_anchor) -> np.ndarray:
    """World direction -> anchor frame."""
    return normalize(np.asarray(R_anchor, dtype=np.float64).T
                     @ np.asarray(d_world, dtype=np.float64))


def to_ee(d_world, ee_rotation) -> np.ndarray:
    """World direction -> the CURRENT EE frame.

    This is the one the network's channels are built from, because the cloud
    lives in the EE frame and the dot products only require `d`, `n_i` and
    `p_i - c` to agree with EACH OTHER. Nothing else about the anchor frame
    reaches the network.
    """
    return normalize(np.asarray(ee_rotation, dtype=np.float64).T
                     @ np.asarray(d_world, dtype=np.float64))


def centroid_axes(members, bins=None, magnitude: bool = False) -> np.ndarray:
    """[k, 3] the UNIT MEAN of each bin's assigned directions; `bins[b]` if empty.

    `members` is an iterable of `(bin_idx, d_anchor)` — the assignment a pin table
    committed to, in ANCHOR frame, which is the only frame in which averaging
    across scenes means anything (world-frame directions for the same bin point
    all over the place, because the anchor rotates with the giver's hand).

    WHY THIS IS NOT `BINS[b]`. The bin axis is a geometric constant; the members
    of a bin are whichever goal-set grasps happened to land nearest it, and on
    this dataset they do NOT surround it. Measured on the run-2 train table, the
    angle between a bin's axis and its members' mean is 8 deg for `+x` and `+z`
    but 14-17 deg for `+-y`, and EVERY bin leans toward `+x`/`+z` — the region
    the hand-collision filter leaves feasible. So a policy commanded `BINS[b]` is
    commanded a direction the data systematically does not demonstrate, and the
    residual is a bias rather than noise that averages out.

    That is what this is for: as a DEPLOYMENT command it is the closest thing to
    the training distribution's centre that is computable WITHOUT a grasp, which
    is the constraint the whole phase exists under. It is a dataset-level
    constant — six vectors — so it transfers to the robot the same way `BINS`
    does. See `examples/plot_regrasp_bin_spread.py`, which measures exactly this
    gap and prints it as the BIAS column.

    Empty bins fall back to the geometric axis rather than to zero: a zero
    direction is a lie the conditioning channels cannot distinguish from
    "approach from nowhere", and `-x`/`-z` are empty on this dataset by
    construction.
    """
    bins = BINS if bins is None else np.asarray(bins, dtype=np.float64)
    acc = np.zeros_like(bins)
    cnt = np.zeros(len(bins), dtype=np.int64)
    for b, d in members:
        if b is None or d is None:
            continue
        b = int(b)
        # `0 <= b < len(bins)` EXCLUDES `BIN_NULL` by construction, which is
        # what should happen: a null-bin demonstration expresses no direction, so
        # averaging it into one would drag that bin's centroid toward zero.
        if 0 <= b < len(bins):
            v = np.asarray(d, dtype=np.float64)
            # MAGNITUDE MODE KEEPS `|d|`. Under `location_extent` the magnitude
            # IS the eccentricity, so normalizing each member before averaging
            # would throw away exactly the half of the command this rule adds and
            # produce a unit axis — i.e. silently fall back to `bin_axis`.
            acc[b] += v if magnitude else normalize(v)
            cnt[b] += 1
    if not magnitude:
        out = normalize(acc)
        # `normalize` returns zeros for a zero-length input, which is exactly the
        # empty-bin case; fall back per row rather than testing counts separately.
        dead = np.linalg.norm(out, axis=-1) < 0.5
        out[dead] = bins[dead]
        return out

    # THE PLAIN MEAN, and NO axis fallback. An empty bin has no magnitude, and
    # `BINS[b]` would assert `m = 1` — the extremity of the object — which is the
    # one command this rule must never issue unprompted: it is both the most
    # aggressive location available and, on a bin with no demonstrations, one the
    # policy has never seen. Zero rows are left zero and the caller must refuse
    # to command them; `resolve_command_axes` does.
    out = np.zeros_like(bins)
    live = cnt > 0
    out[live] = acc[live] / cnt[live, None]
    return out


def command_direction(bin_idx, anchor_R, grasp_pose=None, axes=None):
    """THE world-frame command for one (bin, anchor), in one place.

    Every producer of a command calls this: the base collector, the DAgger
    collector, the evaluator, the retry ladder and the rollout viewer. That is
    the point of it existing. The command is a five-way agreement — training
    labels, on-policy rollouts, eval scoring, the retry ladder and the robot —
    and the failure mode when two of them drift is invisible in every rate: the
    policy is simply told one thing and scored on another, and `dir_err`
    measures against the shifted target too, so nothing reports it.

    `axes` selects the rule:

      `BINS`            the bin's geometric axis. Runs 2-8.
      `centroid_axes()` the bin's empirical mean over the training assignment.
                        Same information content, ~8-17 deg closer to what the
                        demonstrations actually show.
      `None`            the demonstrated grasp's own approach axis, `-R[:,2]`.
                        Run 1. NOT AVAILABLE AT DEPLOYMENT — there is no grasp
                        to read — so this is a training-time rule only, and
                        using it on both sides is what run 2 was built to stop.

    Falls back to the grasp axis whenever the bin or the anchor is missing (a
    Phase-5-shaped pin table), so an old table still collects and still scores
    rather than producing None and failing at the first `act()`.
    """
    if (axes is not None and bin_idx is not None and anchor_R is not None
            and int(bin_idx) >= 0):
        return to_world(np.asarray(axes, dtype=np.float64)[int(bin_idx)],
                        anchor_R)
    return None if grasp_pose is None else approach_direction(grasp_pose)


def histogram(directions, bins=None) -> np.ndarray:
    """[k] counts of `directions` by nearest bin. The go/no-go gate's output."""
    bins = BINS if bins is None else np.asarray(bins, dtype=np.float64)
    out = np.zeros(len(bins), dtype=np.int64)
    for d in np.asarray(directions, dtype=np.float64).reshape(-1, 3):
        out[bin_of(d, bins)] += 1
    return out


def most_separated_pair(candidates, empties=None, bins=None):
    """The maximally-separated pair of feasible bins, tie-broken toward emptier.

    `candidates` is the list of bin indices a scene can actually reach;
    `empties` an optional [k] array of current global counts, lower being more
    wanted. Returns `(i, j)` or None when fewer than two bins are feasible.

    THE PAIR MATTERS MORE THAN THE COUNT. At one demonstration per scene `d` is a
    deterministic function of the observation across the whole dataset, so the
    network can minimise loss by learning scene->action and ignoring the
    conditioning entirely. Two demonstrations of the same scene under different
    `d` map the SAME observation to two DIFFERENT actions, which is the only
    thing that forces the conditioning channels to be read. No architectural fix
    reaches a dataset-level confound, which is why this returns a pair and why
    the separation is maximised rather than the emptiness.
    """
    bins = BINS if bins is None else np.asarray(bins, dtype=np.float64)
    cand = list(dict.fromkeys(int(c) for c in candidates))
    if len(cand) < 2:
        return None
    counts = (np.zeros(len(bins)) if empties is None
              else np.asarray(empties, dtype=np.float64))
    best, best_key = None, None
    for a_i, a in enumerate(cand):
        for b in cand[a_i + 1:]:
            sep = float(angle_between(bins[a], bins[b]))
            # Maximise separation; among equally separated pairs prefer the one
            # whose bins are globally emptiest (hence the negated sum).
            key = (round(sep, 6), -(counts[a] + counts[b]))
            if best_key is None or key > best_key:
                best, best_key = (a, b), key
    return best
