"""
Per-point conditioning: turn `[N, 5]` + a direction into the `[N, 7]` the net sees.

The two appended channels, for each point `p_i` with surface normal `n_i` and a
commanded approach direction `d`:

    d . n_i                     is this surface facing the incoming gripper?
    d . normalize(p_i - c)      is this point on the side I am approaching from?

so the points on the commanded approach side are explicitly marked in the input
rather than left to be inferred from a global feature vector.

    stored in HDF5   [N, 8]   xyz(3) | ycb(1) | hand(1) | nx, ny, nz(3)
    fed to the model [N, 7]   xyz(3) | ycb(1) | hand(1) | d.n(1) | d.r(1)

WHY THE SPLIT. Noise augmentation on `d` happens at TRAINING time, so a
`d`-dependent channel cannot be baked into the file. The `d`-independent part —
the normals — is what gets stored, and the two dot products are computed in
`__getitem__` from the stored normals and a possibly-perturbed `d`. That also
means changing the augmentation, or the direction definition, is a relabelling
pass rather than a collection campaign.

DOT PRODUCTS ARE FRAME-INVARIANT, WHICH IS WHAT MAKES THIS CHEAP. `d`, `n_i` and
`p_i - c` only have to agree with EACH OTHER, so everything happens in the EE
frame where the cloud already lives, and no new frame plumbing is needed anywhere.
The anchor frame never reaches the network.

NORMALS MUST NOT RIDE THROUGH `PointListener`. `GA-DDPG/core/utils.py:310`
`se3_transform_pc` rotates only rows `:3` and COPIES the rest — so a normal
carried as an extra row through `_update_acc_points` would stay in the source
frame while xyz moved to the EE frame, silently. Estimating at the end, in the EE
frame, sidesteps it entirely and leaves `handover_sim2real/policy.py` — shared
with Phase 1-4 and the RL code — untouched.

CHANNEL ORDER IS APPEND-ONLY. Existing consumers index channels 3 and 4 by
hardcoded position (`collector.py`'s DART obstacle mask, `chain_viz`'s cloud
overlay, `visualize_bc_dataset`), so the new channels go at 5 and 6 and nothing
above them moves.

No `env`, no torch, no h5py. This is the module `sim2real/pointcloud_multicam.py`
will import when the 7-channel path reaches the robot.
"""

from __future__ import annotations

import numpy as np

from handover_sim2real.regrasp.directions import normalize
from handover_sim2real.regrasp.normals import DEFAULT_K, estimate_normals

# Channel indices, named so nothing has to count.
CH_XYZ = slice(0, 3)
CH_YCB = 3
CH_HAND = 4
CH_NORMAL = slice(5, 8)      # in the STORED [N, 8] layout
CH_DOT_N = 5                 # in the MODEL [N, 7] layout
CH_DOT_R = 6

STORED_CHANNELS = 8
MODEL_CHANNELS = 7


def object_centroid(pc, fallback_to_all: bool = True):
    """Centroid of the object points, in the cloud's own frame. None if undefined.

    THE EMPTY-OBJECT CASE IS LIVE, NOT THEORETICAL. `handover_sim2real/policy.py`
    records it verbatim: "Measured on scene 494 under pretrain_right.yaml: object
    0 pts, hand 93". A fixed camera can be occluded from step 0,
    `_update_acc_points` skips empty classes, so `acc_points[0]` stays empty for
    the whole episode. A NaN centroid would propagate into `d . r` for every point
    and then into the loss, which is why this returns None and the caller decides.

    Note the object/hand split is NOT always 896/128: with the hand empty the
    object slot regularizes to the full 1024, so the mask can select 1024, 896,
    or 0. Never assume the block boundary.
    """
    p = np.asarray(pc, dtype=np.float64)
    if p.size == 0:
        return None
    mask = p[:, CH_YCB] > 0.5
    if mask.any():
        return p[mask, CH_XYZ].mean(axis=0)
    if fallback_to_all and len(p):
        # Better than None for a rollout that would otherwise die: the hand points
        # are at least near the object. The caller still sees this via the returned
        # flag from `build_model_cloud`.
        return p[:, CH_XYZ].mean(axis=0)
    return None


def pack_stored_cloud(pc5, centroid=None, k: int = DEFAULT_K):
    """`[N, 5]` -> `[N, 8]` for the HDF5, estimating normals. Returns (pc8, info)."""
    p = np.asarray(pc5, dtype=np.float32)
    c = object_centroid(p) if centroid is None else np.asarray(centroid, np.float64)
    if c is None:
        n = np.zeros((len(p), 3), dtype=np.float64)
        info = {"n_points": len(p), "n_fallback": len(p), "n_degenerate": 0,
                "k": int(k), "no_centroid": True}
    else:
        n, info = estimate_normals(p[:, CH_XYZ], c, k)
        info["no_centroid"] = False
    return np.concatenate([p, n.astype(np.float32)], axis=1), info


def append_direction_channels(pc, d, normals=None, centroid=None,
                              k: int = DEFAULT_K):
    """-> `[N, 7]`. Accepts either the stored `[N, 8]` or a bare `[N, 5]`.

    With `[N, 8]` the stored normals are used and nothing is recomputed — the
    training path. With `[N, 5]` normals are estimated on the spot — the inference
    path. Both must produce the same numbers for the same geometry; see
    `normals.py` on why that holds and what would break it.
    """
    p = np.asarray(pc, dtype=np.float64)
    d = normalize(np.asarray(d, dtype=np.float64))
    if p.shape[1] not in (5, STORED_CHANNELS):
        raise ValueError(
            f"expected [N, 5] or [N, {STORED_CHANNELS}], got [N, {p.shape[1]}]")

    c = object_centroid(p) if centroid is None else np.asarray(centroid, np.float64)
    if c is None:
        c = np.zeros(3)

    if normals is not None:
        n = np.asarray(normals, dtype=np.float64)
    elif p.shape[1] == STORED_CHANNELS:
        n = p[:, CH_NORMAL]
    else:
        n, _ = estimate_normals(p[:, CH_XYZ], c, k)

    # QUANTIZE TO float32 BEFORE THE DOT PRODUCT. The training path reads normals
    # that have been through float32 HDF5 storage; the inference path computes
    # them fresh in float64 and would otherwise carry ~1e-7 more precision into
    # `d . n`. The difference is negligible against the degrees of jitter the
    # resampled cloud already contributes -- but with this line the two paths are
    # BIT-IDENTICAL rather than merely close, and an identity is cheaper to keep
    # true than an approximation is to keep bounded. (The same applies to xyz,
    # which is float32 everywhere in the real pipeline; pass float64 xyz to one
    # path and float32 to the other and the CENTROID shifts, which moves `d . r`.)
    n = n.astype(np.float32).astype(np.float64)

    r = normalize(p[:, CH_XYZ] - c)
    dot_n = n @ d
    dot_r = r @ d
    return np.concatenate(
        [p[:, :5], dot_n[:, None], dot_r[:, None]], axis=1).astype(np.float32)


def perturb_direction(d, deg: float, rng=None) -> np.ndarray:
    """Rotate `d` by `deg` degrees about a uniformly random perpendicular axis.

    Training-time augmentation. It teaches interpolation between bins and
    robustness to the fact that a test-time bin centre will not exactly match any
    demonstration's realised direction — the demo's `d` is whatever OMG actually
    flew to, which lands somewhere inside the bin, not on its axis.

    A FIXED magnitude with a random axis, not a Gaussian: it puts every sample on
    a cone of known half-angle, so "trained with 12 degrees of slop" is a
    statement about the data rather than about a tail. `deg <= 0` is the identity.
    """
    d = normalize(np.asarray(d, dtype=np.float64))
    if deg <= 0.0 or float(np.linalg.norm(d)) < 0.5:
        return d
    rng = np.random.default_rng() if rng is None else rng
    # A random vector, made perpendicular to d. Retry only in the vanishingly
    # unlikely case that it came out parallel.
    for _ in range(8):
        v = rng.normal(size=3)
        v = v - np.dot(v, d) * d
        n = np.linalg.norm(v)
        if n > 1e-8:
            axis = v / n
            break
    else:
        return d
    th = np.radians(float(deg))
    # Rodrigues, with d . axis == 0 so the third term drops out.
    return normalize(d * np.cos(th) + np.cross(axis, d) * np.sin(th))


def build_model_cloud(pc5, d_ee, k: int = DEFAULT_K):
    """The inference path in one call: `[N, 5]` + `d` in the EE frame -> `[N, 7]`.

    Returns `(pc7, info)`; `info["no_centroid"]` is the one worth logging, since
    it means `d . r` is being computed against an invented origin.
    """
    p = np.asarray(pc5, dtype=np.float64)
    c = object_centroid(p, fallback_to_all=False)
    info = {"no_centroid": c is None}
    if c is None:
        # `x or y` is a truth test, and on an ndarray that raises. The centroid is
        # an array, so this has to be an explicit None check.
        c = object_centroid(p)
        if c is None:
            c = np.zeros(3)
    n, ninfo = estimate_normals(p[:, CH_XYZ], c, k)
    info.update(ninfo)
    return append_direction_channels(p, d_ee, normals=n, centroid=c), info
