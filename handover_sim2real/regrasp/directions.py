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
BIN_SHORT = ("+x", "-x", "+y", "-y", "+z", "-z")

# The Voronoi half-angle for 90-deg-separated bins is 45 deg. `bin_hit_rate` uses
# 30 to keep margin against boundary noise, so a "hit" is unambiguous rather than
# a coin flip between two adjacent bins.
BIN_HIT_DEG = 30.0

# The retry machine's neighbour-exclusion radius. See `neighbours` — for the
# octahedral set this is a no-op, and that is a fact worth knowing rather than a
# reason to change it.
NEIGHBOUR_DEG = 30.0


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
