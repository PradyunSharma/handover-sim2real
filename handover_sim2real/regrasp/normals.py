"""
Per-point surface normals, estimated from the observed cloud.

kNN + PCA: the smallest eigenvector of a point's local covariance is the surface
normal, up to sign. Sign is fixed by orienting outward from the object centroid.

WHY ESTIMATED AND NOT GROUND TRUTH. The real rig has no mesh. A policy trained on
mesh normals would be trained on a signal it can never see at deployment, so the
sim2real gap would be built into the input rather than measured. `scipy.spatial
.cKDTree` is available in every conda env here (and `OMG/omg/core.py` already uses
it); `open3d` and `sklearn` exist ONLY on the robot PC, so an open3d estimator
would work in deployment and break sim collection and cluster training — exactly
backwards.

WHY PCA AND NOT `normalize(p_i - c)`. Those two would be nearly the same signal on
a convex object, which would make the two conditioning channels redundant. PCA
recovers the TRUE local surface direction — piecewise-constant across a box face —
while `normalize(p_i - c)` varies smoothly over the same face. The centroid is
used only to disambiguate the SIGN, never to define the direction. Keeping them
distinct is the whole point of having two channels.

DETERMINISM IS A HARD REQUIREMENT, AND IT IS INVISIBLE. The collector records
normals into the HDF5; `BCRunner.act` recomputes them from the live cloud at
inference. Those must agree, or the policy sees a different `n` at deployment than
it trained on, with nothing anywhere reporting a mismatch. cKDTree kNN and
`np.linalg.eigh` are both deterministic given the same input array, so the
equality holds today. ANY future switch to an approximate, randomized, or
GPU-nondeterministic estimator silently breaks it. If you change this function,
that is the invariant you are breaking.

TWO DEGENERACIES, BOTH REAL, BOTH GUARDED:

  * DUPLICATED POINTS. `regularize_pc_point_count`
    (`GA-DDPG/core/utils.py:809-813`) oversamples a short class by
    `np.random.choice` WITHOUT `replace=False`, so a cloud with fewer than
    `uniform_num_pts` real points contains exact duplicates. A neighbourhood of
    duplicates has a rank-deficient covariance and `eigh` returns an arbitrary
    eigenvector — not an error, just a wrong number.
  * COLLINEAR NEIGHBOURHOODS. A point on a thin edge or a one-pixel-wide sliver
    gets neighbours along a line; the normal is then ambiguous within a whole
    plane and the returned eigenvector is again arbitrary.

Both fall back to `normalize(p_i - c)` and are COUNTED, because a high fallback
rate means the cloud is too sparse for this to work at all and that is a fact
about the data, not a warning to suppress.

NORMALS JITTER STEP TO STEP EVEN WHEN NOTHING MOVES. `PointListener
._update_acc_points` resamples `acc_points` with `np.random.choice` on every step,
so the point SET changes between steps for identical geometry. A larger `k`
averages over more of the local patch and damps this; `DEFAULT_K = 16` is the
floor. Measure it before trusting the conditioning — `jitter_report` below exists
for exactly that, and if the step-to-step change in `d . n` is comparable to the
signal then the channel is noise.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_K = 16

# A neighbourhood is unusable when its covariance is rank-deficient. Two tests,
# because the two failures look different: everything-identical collapses the
# LARGEST eigenvalue to zero, while collinear collapses the MIDDLE one relative to
# the largest. A genuinely planar patch has a small SMALLEST eigenvalue and is
# exactly what we want, so it must not trip either test.
ABS_EPS = 1e-16          # largest eigenvalue below this: the patch is a single point
REL_EPS = 1e-6           # middle/largest below this: the patch is a line


def estimate_normals(xyz, centroid, k: int = DEFAULT_K):
    """([N, 3] unit normals, info dict) for an [N, 3] point cloud.

    `centroid` fixes the sign only. Frame-agnostic: pass both in the same frame
    and the normals come back in it. Deterministic — see the module docstring.
    """
    p = np.asarray(xyz, dtype=np.float64)
    n_pts = len(p)
    info = {"n_points": n_pts, "n_fallback": 0, "n_degenerate": 0, "k": int(k)}
    if n_pts == 0:
        return np.zeros((0, 3), dtype=np.float64), info

    c = np.asarray(centroid, dtype=np.float64)
    radial = p - c
    rn = np.linalg.norm(radial, axis=1, keepdims=True)
    # A point sitting exactly on the centroid has no outward direction. Give it a
    # fixed axis rather than a NaN; it is one point out of 1024 and a NaN here
    # would poison `d . n` for the whole cloud.
    radial_unit = np.divide(radial, rn, out=np.tile([0.0, 0.0, 1.0], (n_pts, 1)),
                            where=rn > 1e-12)

    kk = int(min(max(k, 3), n_pts))
    if kk < 3:
        info["n_fallback"] = n_pts
        return radial_unit, info

    # cKDTree.query is deterministic; ties in distance break by index, which is
    # stable for a fixed input array.
    _, idx = cKDTree(p).query(p, k=kk)
    nbr = p[idx]                                   # [N, k, 3]
    X = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", X, X) / float(kk)
    w, V = np.linalg.eigh(cov)                     # ascending eigenvalues
    normals = V[:, :, 0]                           # smallest -> surface normal

    largest = w[:, 2]
    middle = w[:, 1]
    degenerate = (largest < ABS_EPS) | (middle <= REL_EPS * np.maximum(largest, ABS_EPS))
    if degenerate.any():
        normals = np.where(degenerate[:, None], radial_unit, normals)
        info["n_degenerate"] = int(degenerate.sum())
        info["n_fallback"] = int(degenerate.sum())

    # Orient outward. `sign` of exactly 0 would zero the normal, so treat the
    # ambiguous case as "already outward" rather than deleting the vector.
    dots = np.einsum("ni,ni->n", normals, radial_unit)
    normals = np.where((dots < 0.0)[:, None], -normals, normals)

    nn = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, nn, out=radial_unit.copy(), where=nn > 1e-12)
    return normals, info


def jitter_report(clouds, centroids, k: int = DEFAULT_K) -> dict:
    """Step-to-step normal stability across a sequence of clouds of one episode.

    `PointListener` resamples the cloud every step, so consecutive steps hold
    different point SETS for the same geometry. This reports how much `n` moves as
    a result, in degrees, matched nearest-neighbour between consecutive steps.
    Run it on a frozen scene: whatever it reports there is pure sampling noise,
    and the conditioning signal has to be large against it.
    """
    angs = []
    prev_p = prev_n = None
    for xyz, c in zip(clouds, centroids):
        n, _ = estimate_normals(xyz, c, k)
        p = np.asarray(xyz, dtype=np.float64)
        if prev_p is not None and len(p) and len(prev_p):
            _, j = cKDTree(prev_p).query(p, k=1)
            d = np.abs(np.einsum("ni,ni->n", n, prev_n[j])).clip(0.0, 1.0)
            angs.append(np.degrees(np.arccos(d)))
        prev_p, prev_n = p, n
    if not angs:
        return {"n_steps": 0}
    a = np.concatenate(angs)
    return {"n_steps": len(angs) + 1, "mean_deg": float(a.mean()),
            "p90_deg": float(np.percentile(a, 90)), "max_deg": float(a.max())}
