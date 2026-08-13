from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CloudExtractionResult:
    hand_xyz: np.ndarray
    object_xyz: np.ndarray
    hand_center: Optional[np.ndarray]
    debug: dict[str, Any]


def _ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"hand_mask must be HxW, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def _sample_or_pad_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Uniformly sample or repeat points to get exactly num_points rows.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got shape {points.shape}")

    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    if n == num_points:
        return points.astype(np.float32, copy=False)

    if n > num_points:
        idx = np.random.choice(n, size=num_points, replace=False)
        return points[idx].astype(np.float32, copy=False)

    idx = np.random.choice(n, size=num_points - n, replace=True)
    padded = np.concatenate([points, points[idx]], axis=0)
    return padded.astype(np.float32, copy=False)


# A crop sphere 0.30 m across at 0.010 m voxels is 60^3 cells. This cap is three
# orders of magnitude above that: it exists to turn "a stray deprojected point at
# 40 m blew the bounding box up" into a graceful fallback rather than an
# allocation that takes the control loop down.
_MAX_CLUSTER_GRID_CELLS = 8_000_000

# A blob is "coherent" when at most this fraction of it falls on the far side of
# the arm threshold (or at least 1 - this fraction does). Anything in between is
# treated as two things stuck together and cut point by point.
#
# SET LOW ON PURPOSE, because the two errors do not cost the same. Splitting a
# genuinely long object loses only the tail beyond the threshold; keeping a
# merged blob admits the ENTIRE forearm, which is the failure being fixed. There
# is no value that separates the two cases cleanly — a long object and an
# object-plus-arm can produce the same fraction — so the tie goes to splitting.
# At 0.15 a heavily-bridged blob still read as coherent-forward, because the
# bridge itself is forward mass that dilutes the arm out of the fraction.
_BLOB_COHERENT_FRAC = 0.05

# ...and a blob at least this arm-like is dropped ENTIRELY, stub included. This
# is the one thing blob structure buys that per-point scoring cannot: a forearm
# always leaves a wedge between the hand and the lateral bound that no per-point
# rule can reach, and dropping the blob whole removes it. Set from the sweep in
# test_multicam_fusion; see _BLOB_ARM_FRAC in that file's arm-rejection test.
_BLOB_ARM_FRAC = 0.60

# How many points inside the grasp region it takes to veto that whole-blob drop.
# Not 1: a single stray point, which depth noise or a calibration error can put
# between the fingers on any frame, must not be able to rescue an entire arm.
# Not large either — at contact the object is the thing being looked at from a
# metre away, and a handful of points on it is a normal reading.
_MIN_GRASP_POINTS = 10


def _voxel_label_grid(clouds: list[np.ndarray], voxel_m: float):
    """26-connected component labels over the union of `clouds`.

    Returns `(labels, n_components, cells)` where `labels` is the dense int
    grid and `cells` holds one fancy-index tuple per input cloud, or None if the
    bounding box would need an unreasonable grid. Shared by the two filters
    below so they cannot drift apart on connectivity or quantisation.
    """
    from scipy import ndimage

    both = np.concatenate(clouds, axis=0)
    origin = both.min(axis=0)
    idx = np.floor((both - origin) / voxel_m).astype(np.int64)
    shape = tuple(int(v) for v in idx.max(axis=0) + 1)
    if int(np.prod(shape, dtype=np.int64)) > _MAX_CLUSTER_GRID_CELLS:
        return None

    cells, start = [], 0
    occupied = np.zeros(shape, dtype=bool)
    for cloud in clouds:
        cell = tuple(idx[start:start + len(cloud)].T)
        occupied[cell] = True
        cells.append(cell)
        start += len(cloud)

    labels, n_components = ndimage.label(
        occupied, structure=np.ones((3, 3, 3), dtype=bool))
    return labels, int(n_components), cells


def reject_arm_clusters(
    object_xyz: np.ndarray,
    hand_xyz: np.ndarray,
    robot_origin: np.ndarray,
    *,
    voxel_m: float,
    min_offset_m: float = 0.07,
    max_lateral_m: float = 0.12,
    grasp_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop the object-class blobs that are arm rather than held object.

    WHY CONNECTIVITY CANNOT DO THIS. `hand_connected_object_points` keeps what is
    contiguous with the hand, and the forearm is the one piece of the scene that
    is contiguous with the hand by anatomy. The segmentation model was trained to
    find hands, not arms, so everything past the wrist falls out of the hand mask
    and into the object class, permanently and by construction.

    In simulation this had no counterpart at all: the human is a MANO mesh, which
    is wrist-to-fingertips, so no forearm exists in either class of the training
    data. The arm is not a kind of object point the policy handles badly — it is
    a kind it has never seen.

    THE DISCRIMINATOR. The object is HELD, so it lies within roughly a grasp's
    reach of the hand in every direction. The forearm is ATTACHED, so it leaves
    the hand and keeps going. With

        s = (p - hand centroid) . unit(robot origin - hand centroid)
        r = |(p - hand centroid) - s * that unit vector|

    a point counts as arm when `s < -min_offset_m` OR `r > max_lateral_m`. The
    kept region is therefore a capsule around the hand: long toward the robot,
    where an object is offered; short behind it and bounded sideways, where only
    the arm goes.

    THE LATERAL BOUND IS NOT OPTIONAL, and this function shipped once without it.
    A signed projection alone measures only the component along one axis, so it
    ranks a forearm by how nearly it points away from the robot. Swept against
    inclination, with the arm rotating out of that axis:

        tilt      0    15    30    45    60    75    90 deg
        kept     6%    9%   11%   23%   42%   86%  100%

    A forearm hanging straight down from the hand is at 90 degrees and scores
    about -0.05 — inside any threshold that keeps a mug. It was reported from
    hardware exactly that way: fine with the forearm horizontal, arm labelled
    object once it tilted. Laterally it is 0.15 m or more from the hand centroid
    the whole time, which is what the second term catches.

    THREE FURTHER CHOICES WORTH DEFENDING:

      * The robot BASE, not the end effector. The EE is where the policy is
        driving to, so scoring against it closes a feedback loop: one frame that
        picks the forearm pulls the EE toward the forearm, which makes the
        forearm score better next frame. The base cannot be moved by a
        perception error. Early in an episode the arm is near home anyway, so
        the two axes nearly agree.
      * A threshold, not an argmin over blobs. "Keep the single nearest cluster"
        is the tempting version and it deletes half an object whenever occlusion
        splits it in two, which fingers and depth dropouts do. A threshold keeps
        every fragment on the grasp side and still drops the arm.
      * `min_offset_m` is well negative rather than zero. An object need not be
        offered forward: a mug gripped by the handle has its body on the human's
        side of the hand. A forearm cropped to 0.25 m has its centroid about
        0.13 m back when it is horizontal.

    WHAT THE LATERAL BOUND COSTS. An object held so that it extends sideways
    past `max_lateral_m` — a 0.24 m bar gripped at one end and pointed across the
    robot's line of sight — loses the part beyond it. That is the same tension
    `object_max_radius_m` used to carry alone, moved somewhere it can be
    anisotropic: the direction an object is plausibly offered stays generous.

    WHAT THIS CANNOT SEPARATE. Scoring is per blob, so an object resting against
    the forearm is one blob with it and shares its fate. The hand normally
    prevents that — it sits between them and is excluded from its own class — but
    that is what makes the two separable, not a guarantee that they always are.

    Returns a boolean KEEP MASK over `object_xyz`, not the points, because at
    the point this runs each row has a parallel entry saying which camera it came
    from, and returning points would silently invalidate it.
    """
    debug: dict[str, Any] = {
        "applied": False,
        "clusters": [],
        "dropped": 0,
        "fallback": None,
        "grasp_points": 0,
        "grasp_vetoed": 0,
    }
    keep_all = np.ones(len(object_xyz), dtype=bool)

    if voxel_m <= 0:
        debug["fallback"] = "disabled"
        return keep_all, debug
    if len(object_xyz) == 0 or len(hand_xyz) == 0:
        debug["fallback"] = "no-seed"
        return keep_all, debug

    hand_center = np.median(hand_xyz, axis=0)
    axis = np.asarray(robot_origin, dtype=np.float64) - hand_center
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        # The hand is on top of the robot origin; there is no "toward the robot".
        debug["fallback"] = "degenerate-axis"
        return keep_all, debug
    axis /= norm

    grid = _voxel_label_grid([object_xyz], voxel_m)
    if grid is None:
        debug["fallback"] = "grid-too-large"
        return keep_all, debug
    labels, n_components, (obj_cells,) = grid

    # The kept region is a capsule around the hand, not a half-space: long
    # TOWARD the robot, short behind, and bounded SIDEWAYS. The lateral bound is
    # what makes this work for an arm at any inclination — see the docstring.
    rel = object_xyz - hand_center
    point_offsets = rel @ axis
    lateral = np.linalg.norm(rel - point_offsets[:, None] * axis, axis=1)
    arm_like = (point_offsets < -min_offset_m) | (lateral > max_lateral_m)

    # THE GRASP REGION OVERRIDES EVERYTHING. A point between the fingers is the
    # object by definition, so it is never arm however it scores, and a blob
    # holding enough of them is never discarded whole. Without this the object
    # vanished precisely as the gripper reached it: the two clouds touch, merge
    # into one blob, the robot's own links carry it past the whole-blob arm
    # threshold, and the object leaves with them. See GraspRegion.
    if grasp_mask is None:
        grasp_mask = np.zeros(len(object_xyz), dtype=bool)
    else:
        grasp_mask = np.asarray(grasp_mask, dtype=bool)
        if len(grasp_mask) != len(object_xyz):
            raise ValueError(
                f"grasp_mask has {len(grasp_mask)} entries for "
                f"{len(object_xyz)} points")
    arm_like &= ~grasp_mask
    debug["grasp_points"] = int(grasp_mask.sum())

    point_labels = labels[obj_cells]
    keep = np.zeros(len(object_xyz), dtype=bool)
    blobs, split, vetoed = [], 0, 0
    for label in np.unique(point_labels):
        if label <= 0:
            continue
        member = point_labels == label
        offset = float(point_offsets[member].mean())
        # Median, not mean: this number is read to choose max_lateral_m, and a
        # blob with a few stray points far off-axis should not report a radius
        # no actual part of it sits at.
        lat = float(np.median(lateral[member]))
        arm_frac = float(arm_like[member].mean())

        # A blob that is mostly inside or mostly outside is decided whole, which
        # is what protects a coherent object that happens to straddle the line —
        # a mug gripped by the handle has its body on the human's side.
        #
        # A blob that is genuinely MIXED is the signature of two things fused by
        # a bridge of noise, and deciding it whole is exactly the failure this
        # guards against: the merged object+forearm blob scores -0.025 and the
        # whole arm rides in on the object's coattails. There, and only there,
        # fall back to cutting point by point.
        # A blob reaching into the grasp is cut point by point at worst, never
        # dropped whole — the per-point rule then keeps every grasp point,
        # because they were cleared of arm_like above.
        holds_grasp = int(grasp_mask[member].sum()) >= _MIN_GRASP_POINTS

        if arm_frac <= _BLOB_COHERENT_FRAC:
            keep |= member
        elif arm_frac >= _BLOB_ARM_FRAC and not holds_grasp:
            pass
        else:
            keep |= member & ~arm_like
            split += 1
            if holds_grasp and arm_frac >= _BLOB_ARM_FRAC:
                vetoed += 1

        blobs.append((member, offset, lat, arm_frac))

    # (points, offset toward the robot, median distance off that axis, arm
    # fraction). The lateral column exists because a blob can be well inside
    # min_offset_m and still be 100% arm-like, and without it that reads as
    # inexplicable.
    debug["clusters"] = sorted(
        ((int(m.sum()), round(o, 4), round(r, 4), round(f, 3))
         for m, o, r, f in blobs),
        reverse=True)
    debug["components"] = n_components
    debug["split_blobs"] = split
    debug["grasp_vetoed"] = vetoed

    if not keep.any():
        # NOTHING IN THE OBJECT CLASS LOOKS LIKE A HELD OBJECT, so return an
        # empty class and let the caller hold the frame.
        #
        # This function first did the opposite — kept the least-bad blob, on the
        # reasoning that an empty class is unusable while a dirty one is merely
        # degraded. On hardware that fallback fired on EVERY frame and quietly
        # promoted an 87%-arm blob of 862 points to be the object, which is the
        # exact contamination the rest of this function exists to prevent, now
        # invisible because it looks like a normal object cloud downstream.
        #
        # Empty is the honest answer and it is also the safe one:
        # FusedObservation.usable already goes false on an empty class, so the
        # runner holds its step and the viewer prints NOT USABLE. A stall is
        # visible and is fixed by loosening --arm-lateral or --arm-offset; an
        # arm labelled "object" is invisible and steers the policy into it.
        debug["fallback"] = "all-arm"
        return np.zeros(len(object_xyz), dtype=bool), debug

    debug["applied"] = True
    debug["dropped"] = int((~keep).sum())
    return keep, debug


def hand_connected_object_points(
    object_xyz: np.ndarray,
    hand_seed_xyz: np.ndarray,
    *,
    voxel_m: float,
    min_hand_voxel_frac: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep only the object points that form one connected body with the hand.

    WHY THIS EXISTS. The object class is defined negatively — non-hand points
    near the hand — so it is really "whatever happens to be inside a sphere".
    The table under the handover, the human's other arm and the wall behind them
    all land in that sphere and become object points. In simulation the object
    class came from a segmentation-buffer lookup on the YCB body id, so it
    contained the object and nothing else; every one of those extra points is a
    kind of input the policy has never seen, in the channel that owns 896 of its
    1024 rows.

    The physical fact the sphere ignores is that the object is BEING HELD. It is
    in contact with the hand, so in 3D it is one connected body with it, while
    the table and the background are not. Labelling by connectivity rather than
    by distance is therefore both stricter and more permissive in the right
    directions: it drops a table point 5 cm from the hand, and it keeps the far
    end of a 0.2 m drill that a radius test would amputate.

    METHOD. Voxelize hand and object together at `voxel_m`, take 26-connected
    components, and keep the object points whose voxel lands in a component the
    hand also occupies. `scipy.ndimage` does the labelling in C over a grid of a
    few tens of thousands of cells, which is far cheaper than a KD-tree
    region-grow over the ~10^4 points themselves.

    A component qualifies only if it holds `min_hand_voxel_frac` of the hand's
    voxels. A single mis-segmented hand pixel landing on the table would
    otherwise vote the entire tabletop into the object class, which is precisely
    the failure this function exists to remove.

    NOTHING IS DILATED, which was not the first design. Dilating the hand by a
    voxel before labelling looks like cheap insurance against a gap between the
    hand and the object it holds, and it doubles the reach in every direction
    including downward: on a staged scene with the hand 3 cm above a table it
    pulled in half the tabletop, while removing it cost nothing. It costs
    nothing because the two classes PARTITION the cropped pixels — every crop
    pixel is either hand or not — so wherever the object meets the hand in the
    image the two clouds are pixel-adjacent by construction, and only a band of
    dropped depth at the occlusion boundary can separate them. Measured gap
    tolerance on a held bar, undilated:

        voxel 8 mm -> 5 mm    voxel 10 mm -> 10 mm    voxel 12 mm -> 15 mm

    which is why the far camera, whose depth drops out over a wider band at
    range, gets the coarser voxel.

    WHAT THIS DOES NOT FIX. The reach that connects the object also connects
    anything else, so a hand held close enough to the table merges with it and
    this becomes a no-op. Measured on the same scene, the table separates
    cleanly above 17.5 mm of standoff at a 10 mm voxel and 22.5 mm at 12 mm.
    Below that the object class is no better than the sphere gave. Plane removal
    is the answer to that case and is deliberately not implemented here.

    Returns the kept points and a debug dict. Every failure path returns the
    input unchanged: an object cloud contaminated with table is degraded, an
    empty one is unusable, so this never subtracts its way down to nothing.
    """
    debug: dict[str, Any] = {
        "applied": False,
        "components": 0,
        "dropped": 0,
        "fallback": None,
    }

    if voxel_m <= 0:
        debug["fallback"] = "disabled"
        return object_xyz, debug
    if len(object_xyz) == 0 or len(hand_seed_xyz) == 0:
        debug["fallback"] = "no-seed"
        return object_xyz, debug

    grid = _voxel_label_grid([object_xyz, hand_seed_xyz], voxel_m)
    if grid is None:
        debug["fallback"] = "grid-too-large"
        return object_xyz, debug
    labels, n_components, (obj_cells, hand_cells) = grid

    hand_occ = np.zeros(labels.shape, dtype=bool)
    hand_occ[hand_cells] = True

    debug["components"] = n_components
    if n_components <= 1:
        # Hand, object and whatever else survived the crop are already one blob.
        # Nothing to separate; say so rather than claiming a filter ran.
        debug["applied"] = True
        return object_xyz, debug

    # One entry per occupied HAND VOXEL, not per hand point, so the vote is by
    # volume. Counting points would let a dense patch of noise outvote the hand.
    hand_labels, hand_counts = np.unique(labels[hand_occ], return_counts=True)
    valid = hand_labels > 0
    hand_labels, hand_counts = hand_labels[valid], hand_counts[valid]
    if len(hand_labels) == 0:
        debug["fallback"] = "no-hand-component"
        return object_xyz, debug

    threshold = max(1, int(np.ceil(min_hand_voxel_frac * hand_counts.sum())))
    keep_labels = hand_labels[hand_counts >= threshold]

    keep = np.isin(labels[obj_cells], keep_labels)
    n_keep = int(keep.sum())
    if n_keep == 0:
        # The object segmented into no component the hand touches. Trust the
        # sphere for this frame rather than hand the policy an empty class.
        debug["fallback"] = "no-connected-object"
        return object_xyz, debug

    debug["applied"] = True
    debug["dropped"] = len(object_xyz) - n_keep
    return object_xyz[keep], debug


def extract_hand_object_clouds(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    hand_mask: np.ndarray,
    cam,
    last_hand_xyz: Optional[np.ndarray] = None,
    last_object_xyz: Optional[np.ndarray] = None,
    crop_radius_m: float = 0.12,
    object_max_radius_m: float = 0.10,
    cluster_voxel_m: float = 0.0,
    cluster_min_hand_frac: float = 0.05,
    hand_margin_px: int = 0,
    min_depth_m: float = 0.10,
    max_depth_m: float = 1.50,
    full_cloud_stride: int = 2,
    hand_cloud_stride: int = 1,
    min_hand_points: int = 100,
    min_object_points: int = 80,
) -> CloudExtractionResult:
    """
    Extract hand and object point clouds in the external camera frame.

    Assumptions:
    - hand_mask is aligned with color/depth.
    - cam.depth_to_pointcloud(...) returns:
        points_xyz: Nx3
        colors_rgb: optional Nx3
        pixel_indices: Nx2 as (v, u)

    Returns:
    - hand_xyz: hand points in camera frame
    - object_xyz: object points in camera frame
    - hand_center: 3D crop center used this frame, or None
    - debug: counts and fallback info
    """
    hand_mask = _ensure_uint8_mask(hand_mask)

    debug: dict[str, Any] = {
        "used_last_hand": False,
        "used_last_object": False,
        "hand_center": None,
        "raw_hand_points": 0,
        "crop_points": 0,
        "crop_hand_points": 0,
        "crop_object_points": 0,
        "cluster_dropped": 0,
        "cluster_fallback": None,
        "margin_band_points": 0,
    }

    # 1) Hand cloud directly from the 2D hand mask.
    hand_points_xyz, _, _ = cam.depth_to_pointcloud(
        depth_m=depth_m,
        color_bgr=color_bgr,
        mask=hand_mask,
        stride=hand_cloud_stride,
        min_depth=min_depth_m,
        max_depth=max_depth_m,
    )
    hand_points_xyz = hand_points_xyz.astype(np.float32, copy=False)
    debug["raw_hand_points"] = int(len(hand_points_xyz))

    if len(hand_points_xyz) >= min_hand_points:
        hand_center = np.median(hand_points_xyz, axis=0).astype(np.float32)
    else:
        hand_center = None

    # If the hand is too weak this frame, fall back early if possible.
    if hand_center is None:
        if last_hand_xyz is not None and len(last_hand_xyz) > 0:
            debug["used_last_hand"] = True
            hand_points_xyz = last_hand_xyz.astype(np.float32, copy=False)
            hand_center = np.median(hand_points_xyz, axis=0).astype(np.float32)
        else:
            hand_points_xyz = np.zeros((0, 3), dtype=np.float32)

    if hand_center is not None:
        debug["hand_center"] = hand_center.copy()

    # 2) Full cloud for local crop around the hand.
    full_points_xyz, _, pixel_indices = cam.depth_to_pointcloud(
        depth_m=depth_m,
        color_bgr=color_bgr,
        mask=None,
        stride=full_cloud_stride,
        min_depth=min_depth_m,
        max_depth=max_depth_m,
    )
    full_points_xyz = full_points_xyz.astype(np.float32, copy=False)
    pixel_indices = pixel_indices.astype(np.int32, copy=False)

    if hand_center is None or len(full_points_xyz) == 0:
        # Cannot crop meaningfully; use fallbacks if available.
        if last_object_xyz is not None and len(last_object_xyz) > 0:
            object_xyz = last_object_xyz.astype(np.float32, copy=False)
            debug["used_last_object"] = True
        else:
            object_xyz = np.zeros((0, 3), dtype=np.float32)

        return CloudExtractionResult(
            hand_xyz=hand_points_xyz,
            object_xyz=object_xyz,
            hand_center=hand_center,
            debug=debug,
        )

    dist_to_center = np.linalg.norm(full_points_xyz - hand_center[None, :], axis=1)
    crop_keep = dist_to_center < crop_radius_m

    crop_points_xyz = full_points_xyz[crop_keep]
    crop_pixel_indices = pixel_indices[crop_keep]
    debug["crop_points"] = int(len(crop_points_xyz))

    if len(crop_points_xyz) == 0:
        if last_object_xyz is not None and len(last_object_xyz) > 0:
            object_xyz = last_object_xyz.astype(np.float32, copy=False)
            debug["used_last_object"] = True
        else:
            object_xyz = np.zeros((0, 3), dtype=np.float32)

        return CloudExtractionResult(
            hand_xyz=hand_points_xyz,
            object_xyz=object_xyz,
            hand_center=hand_center,
            debug=debug,
        )

    # 3) Re-label cropped 3D points using the 2D hand mask.
    #
    #    THE MARGIN BAND IS WHY THIS IS THREE-WAY RATHER THAN TWO-WAY. The mask
    #    sits a little inside the true hand silhouette — morphological opening
    #    eats the boundary, and the segmentation is upsampled from 256 or 384 by
    #    nearest neighbour — so a thin shell of genuine HAND-SURFACE points falls
    #    outside it and into the object class. That shell wraps the hand, and in
    #    doing so it connects the held object to the forearm behind the wrist,
    #    which is what `reject_arm_clusters` relies on being two separate blobs.
    #    Measured on a staged scene: 20% of that ring surviving is enough to fuse
    #    them into one blob whose centroid scores -0.025 m, comfortably inside
    #    any sane threshold, so the whole arm is kept. Since the ring's
    #    continuity flickers with the mask, so did the arm.
    #
    #    So pixels within `hand_margin_px` of the mask belong to NEITHER class.
    #    They still seed the connectivity test — dropping them outright would
    #    open a gap between the hand and the object it holds, which is the one
    #    adjacency `hand_connected_object_points` cannot do without.
    vv = crop_pixel_indices[:, 0]
    uu = crop_pixel_indices[:, 1]
    crop_hand_flags = hand_mask[vv, uu] > 0

    if hand_margin_px > 0:
        import cv2

        k = np.ones((2 * hand_margin_px + 1, 2 * hand_margin_px + 1), np.uint8)
        grown_mask = cv2.dilate(hand_mask, k)
        crop_grown_flags = grown_mask[vv, uu] > 0
    else:
        crop_grown_flags = crop_hand_flags

    hand_crop_xyz = crop_points_xyz[crop_hand_flags].astype(np.float32, copy=False)
    object_crop_xyz = crop_points_xyz[~crop_grown_flags].astype(np.float32, copy=False)
    band_crop_xyz = crop_points_xyz[
        crop_grown_flags & ~crop_hand_flags].astype(np.float32, copy=False)

    debug["crop_hand_points"] = int(len(hand_crop_xyz))
    debug["crop_object_points"] = int(len(object_crop_xyz))
    debug["margin_band_points"] = int(len(band_crop_xyz))

    # 4) Keep only the object points that form one connected body with the hand.
    #    This is what actually suppresses table and background leakage; the
    #    radius test below is a bound, not a segmentation.
    #
    #    The seed is the crop's own hand points, deliberately: they came off the
    #    same `full_cloud_stride` as the object points, so both classes have the
    #    same voxel density and connectivity means the same thing on both sides
    #    of the boundary. The stride-1 hand cloud is denser and would over-report
    #    occupancy near the hand, so it is only the fallback — bounded by the
    #    crop radius, since it is the one cloud that was never cropped and can
    #    run up a sleeve to the shoulder.
    if cluster_voxel_m > 0 and len(object_crop_xyz) > 0:
        seed = hand_crop_xyz
        if len(seed) < min_hand_points and len(hand_points_xyz) > 0:
            seed_dist = np.linalg.norm(hand_points_xyz - hand_center[None, :], axis=1)
            seed = hand_points_xyz[seed_dist < crop_radius_m]
        if len(band_crop_xyz):
            # The band reaches from the mask boundary out to where the object
            # class now begins, so including it keeps hand and object adjacent
            # across the gap the band itself opened.
            seed = np.concatenate([seed, band_crop_xyz], axis=0)
        object_crop_xyz, cluster_debug = hand_connected_object_points(
            object_crop_xyz, seed,
            voxel_m=cluster_voxel_m,
            min_hand_voxel_frac=cluster_min_hand_frac,
        )
        debug["cluster_dropped"] = cluster_debug["dropped"]
        debug["cluster_fallback"] = cluster_debug["fallback"]

    # 5) Bound the object to a sphere around the hand. With clustering on this
    #    is a backstop against a component that reaches out of the workspace,
    #    which is why `object_max_radius_m` can be set well past the largest
    #    object instead of tight enough to double as the segmentation.
    if len(object_crop_xyz) > 0:
        obj_dist = np.linalg.norm(object_crop_xyz - hand_center[None, :], axis=1)
        object_crop_xyz = object_crop_xyz[obj_dist < object_max_radius_m]

    # 6) Choose final hand cloud.
    if len(hand_crop_xyz) >= min_hand_points:
        hand_xyz = hand_crop_xyz
    elif len(hand_points_xyz) >= min_hand_points:
        hand_xyz = hand_points_xyz
    elif last_hand_xyz is not None and len(last_hand_xyz) > 0:
        hand_xyz = last_hand_xyz.astype(np.float32, copy=False)
        debug["used_last_hand"] = True
    else:
        hand_xyz = hand_crop_xyz  # possibly empty

    # 7) Choose final object cloud with fallback.
    if len(object_crop_xyz) >= min_object_points:
        object_xyz = object_crop_xyz.astype(np.float32, copy=False)
    elif last_object_xyz is not None and len(last_object_xyz) > 0:
        object_xyz = last_object_xyz.astype(np.float32, copy=False)
        debug["used_last_object"] = True
    else:
        object_xyz = object_crop_xyz.astype(np.float32, copy=False)

    return CloudExtractionResult(
        hand_xyz=hand_xyz.astype(np.float32, copy=False),
        object_xyz=object_xyz.astype(np.float32, copy=False),
        hand_center=hand_center,
        debug=debug,
    )


def build_policy_point_tensor(
    hand_xyz: np.ndarray,
    object_xyz: np.ndarray,
    num_hand_points: int = 128,
    num_object_points: int = 896,
) -> np.ndarray:
    """
    Build the policy input cloud with one-hot labels:
      hand points  -> [x, y, z, 1, 0]
      object points-> [x, y, z, 0, 1]

    Returns:
      (num_hand_points + num_object_points, 5) float32
    """
    hand_fixed = _sample_or_pad_points(hand_xyz, num_hand_points)
    obj_fixed = _sample_or_pad_points(object_xyz, num_object_points)

    hand_labels = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (num_hand_points, 1))
    obj_labels = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (num_object_points, 1))

    hand_feat = np.concatenate([hand_fixed, hand_labels], axis=1)
    obj_feat = np.concatenate([obj_fixed, obj_labels], axis=1)

    return np.concatenate([hand_feat, obj_feat], axis=0).astype(np.float32)