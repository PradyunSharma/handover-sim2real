"""
PyBullet debug overlays for the Regrasp conditioning geometry.

The anchor frame, the bin sphere and the `d` vector are the three things you have
to see together to read a Regrasp episode: `d` is meaningless without the frame it
is expressed in, and the frame is hard to read without the bins painted on it.

SHARED BECAUSE TWO SCRIPTS DRAW THEM. `rollout_regrasp_policy.py` shows what a
trained policy is commanded; `visualize_bc_dataset.py` shows what a recorded
demonstration was labelled with. Those must be drawn identically or comparing
them by eye is worthless — a bin that is green in one view and orange in the
other is a bug nobody will notice.

Colours match `plot_regrasp_run.py`'s `_BIN_COLOURS_BY_BIN`, so a bin is the same
colour in the GUI as on every figure.

`pybullet` is imported lazily inside each function: this module is pulled in by
scripts that never open a GUI, and importing pybullet has side effects.
"""

from __future__ import annotations

import numpy as np

from handover_sim2real.regrasp import directions as _dirs

# tab10, as RGB.
BIN_RGB = ((0.12, 0.47, 0.71),      # +x  free end        tab:blue
           (0.55, 0.34, 0.29),      # -x  over fingers    tab:brown
           (0.17, 0.63, 0.17),      # +y  lateral         tab:green
           (1.00, 0.50, 0.05),      # -y  lateral         tab:orange
           (0.84, 0.15, 0.16),      # +z  top-down        tab:red
           (0.58, 0.40, 0.74))      # -z  from beneath    tab:purple


def draw_anchor_frame(anchor_R, origin, ids, length=0.15, width=3.0):
    """The gravity-aligned, wrist-anchored frame every direction is expressed in.

        x = horizontal(object centroid - giver's wrist)   "away from the hand"
        z = world up
        y = z x x

    Drawn at the object centroid, which is the frame's origin by construction:
    `d` points FROM the object OUTWARD toward where the gripper comes from, and
    the two conditioning channels are `d.n_i` and `d.normalize(p_i - c)`.

    Positive axes are solid and labelled; negatives are drawn at half length so
    the handedness is readable without cluttering the object.
    """
    import pybullet

    origin = np.asarray(origin, dtype=np.float64)
    R = np.asarray(anchor_R, dtype=np.float64)
    for k, (col, lab) in enumerate(((( 1.0, 0.25, 0.25), "x  away from hand"),
                                    (( 0.25, 1.0, 0.25), "y  lateral"),
                                    (( 0.35, 0.55, 1.0), "z  world up"))):
        a = R[:, k]
        ids.append(pybullet.addUserDebugLine(
            origin.tolist(), (origin + length * a).tolist(),
            lineColorRGB=list(col), lineWidth=width))
        ids.append(pybullet.addUserDebugLine(
            origin.tolist(), (origin - 0.5 * length * a).tolist(),
            lineColorRGB=[c * 0.45 for c in col], lineWidth=width * 0.5))
        ids.append(pybullet.addUserDebugText(
            lab, (origin + 1.12 * length * a).tolist(),
            textColorRGB=list(col), textSize=1.0))


def draw_bin_sphere(anchor_R, origin, ids, radius=0.10, n_points=2400,
                    point_size=3):
    """A see-through sphere around the object, painted by BIN.

    Every point is a candidate direction, coloured by which of the k bins it
    falls in — so the six Voronoi cells of the octahedral set are directly
    visible. The 45 deg cell half-angle that `bin_of` implements is the boundary
    between colours.

    POINTS RATHER THAN A MESH, deliberately. A PyBullet visual shape carries ONE
    rgba, so a genuinely segmented sphere would need six meshes; and a solid
    translucent shell hides the object it is centred on. A point shell is
    see-through by construction, needs no bodies to clean up, and goes away with
    the other debug items.
    """
    import pybullet

    origin = np.asarray(origin, dtype=np.float64)
    R = np.asarray(anchor_R, dtype=np.float64)
    d = _dirs.fibonacci_directions(int(n_points))          # anchor frame
    bins = np.array([_dirs.bin_of(v) for v in d])
    pts = origin + radius * (R @ d.T).T
    cols = np.array([BIN_RGB[b % len(BIN_RGB)] for b in bins])
    ids.append(pybullet.addUserDebugPoints(
        pts.tolist(), cols.tolist(), pointSize=int(point_size)))

    # A labelled ray down each bin's axis, so the colours have names. These are
    # the vectors `retry.next_direction` issues; at k=6 they coincide with the
    # frame axes above, which is the interpretability the octahedral set buys.
    for b in range(len(_dirs.BINS)):
        a = _dirs.to_world(_dirs.BINS[b], R)
        ids.append(pybullet.addUserDebugLine(
            origin.tolist(), (origin + radius * 1.35 * a).tolist(),
            lineColorRGB=list(BIN_RGB[b]), lineWidth=2.0))
        ids.append(pybullet.addUserDebugText(
            _dirs.BIN_SHORT[b], (origin + radius * 1.45 * a).tolist(),
            textColorRGB=list(BIN_RGB[b]), textSize=1.1))


def draw_direction(d_world, origin, ids, *, colour=(1.0, 1.0, 1.0),
                   label=None, length=0.22, width=5.0, cone=0.035):
    """One `d` as a thick labelled arrow from the anchor origin.

    `d` is a DIRECTION, so it is drawn from the frame's origin rather than from
    any point on the gripper — the length carries no information and is a
    display choice. The arrowhead is two short back-swept segments in the plane
    least aligned with `d`, which is enough to read the sense at a glance.

    Returns the arrow's tip in world, so a caller can hang extra text off it.
    """
    import pybullet

    origin = np.asarray(origin, dtype=np.float64)
    d = _dirs.normalize(np.asarray(d_world, dtype=np.float64))
    if float(np.linalg.norm(d)) < 0.5:            # normalize returns 0 on failure
        return None
    tip = origin + length * d

    ids.append(pybullet.addUserDebugLine(
        origin.tolist(), tip.tolist(), lineColorRGB=list(colour), lineWidth=width))

    # Any vector not parallel to d gives a plane to sweep the head back in.
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp = _dirs.normalize(np.cross(d, ref))
    for s in (+1.0, -1.0):
        ids.append(pybullet.addUserDebugLine(
            tip.tolist(), (tip - cone * d + s * cone * 0.6 * perp).tolist(),
            lineColorRGB=list(colour), lineWidth=width))
    if label:
        ids.append(pybullet.addUserDebugText(
            label, (tip + 0.03 * d).tolist(),
            textColorRGB=list(colour), textSize=1.2))
    return tip


def draw_grasp_point(grasp_pose, centroid_world, ids, *,
                     depth: float = _dirs.FINGERTIP_DEPTH,
                     colour=(1.0, 0.85, 0.1), size=9.0):
    """The point `grasp_offset` measures to, plus the chord it measures along.

    Under `d_rule: grasp_offset` (run 10) `d = normalize(grasp_point - c)`, so the
    vector is only interpretable next to the two things that define it: the
    fingertip midpoint at `depth` along the gripper's own +z, and the segment
    from the centroid to it. Drawing them makes the SHORT-OFFSET failure visible
    too — when that segment is a stub, `d` is centroid noise rather than
    geometry, which is what `d_min_offset` exists to reject.

    Returns the offset length in metres (`None` if there is no pose).
    """
    import pybullet

    if grasp_pose is None:
        return None
    pt = _dirs.grasp_point(grasp_pose, depth)
    c = np.asarray(centroid_world, dtype=np.float64)
    ids.append(pybullet.addUserDebugPoints(
        [pt.tolist()], [list(colour)], pointSize=float(size)))
    ids.append(pybullet.addUserDebugLine(
        c.tolist(), pt.tolist(), lineColorRGB=[v * 0.7 for v in colour],
        lineWidth=1.5))
    return float(np.linalg.norm(pt - c))
