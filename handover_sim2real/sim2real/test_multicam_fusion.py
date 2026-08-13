#!/usr/bin/env python3
"""Check the multi-camera fusion without a robot, a policy, or a GPU.

    python test_multicam_fusion.py            # geometry only, no hardware
    python test_multicam_fusion.py --live     # also open the cameras and fuse

The geometry tests use fake cameras, so they answer the question that hardware
cannot: given a KNOWN point in the world, does each camera's chain put it at the
same place in the hand frame? A live run tells you the pipeline produces a
tensor; only this tells you the tensor means what the policy thinks it means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pointcloud_multicam import (  # noqa: E402
    NUM_HAND_POINTS,
    NUM_OBJECT_POINTS,
    ROBOT_EXCLUSION,
    CameraRig,
    FIXED_PARAMS,
    WRIST_PARAMS,
    build_policy_cloud,
)
from transforms import invert_transform  # noqa: E402


def _rig_transform_agreement() -> None:
    """The two camera kinds must map a world point to the SAME hand-frame point.

    This is the test that catches an inverted extrinsic. A flipped T_base_cam
    still yields a plausible-looking cloud in roughly the right place, so it
    survives visual inspection; it does not survive being compared against the
    wrist camera's independent answer for the same physical point.
    """
    rng = np.random.default_rng(0)

    # An arbitrary, non-degenerate robot pose (nothing axis-aligned, so a
    # transposed rotation cannot pass by symmetry).
    from scipy.spatial.transform import Rotation as Rot
    T_base_hand = np.eye(4)
    T_base_hand[:3, :3] = Rot.from_euler("xyz", [0.3, -0.7, 1.1]).as_matrix()
    T_base_hand[:3, 3] = [0.42, -0.13, 0.55]

    # Wrist camera: fixed on the hand.
    T_hand_cam_wrist = np.eye(4)
    T_hand_cam_wrist[:3, :3] = Rot.from_euler("z", np.pi / 2).as_matrix()
    T_hand_cam_wrist[:3, 3] = [0.036, 0.0, 0.036]

    # Tripod: fixed in the base frame.
    T_base_cam_fixed = np.eye(4)
    T_base_cam_fixed[:3, :3] = Rot.from_euler("xyz", [-2.0, 0.1, 0.9]).as_matrix()
    T_base_cam_fixed[:3, 3] = [1.1, 0.6, 0.9]

    class _FakeCam:
        pass

    wrist = CameraRig(name="wrist", kind="eye_in_hand", camera=_FakeCam(),
                      params=WRIST_PARAMS, T_hand_cam=T_hand_cam_wrist)
    tripod = CameraRig(name="tripod", kind="fixed", camera=_FakeCam(),
                       params=FIXED_PARAMS, T_base_cam=T_base_cam_fixed)

    # A point somewhere in the workspace, expressed in each camera's own frame.
    p_base = np.array([0.5, 0.1, 0.6, 1.0])
    p_cam_wrist = invert_transform(T_base_hand @ T_hand_cam_wrist) @ p_base
    p_cam_tripod = invert_transform(T_base_cam_fixed) @ p_base

    from transforms import transform_points
    got_wrist = transform_points(
        wrist.hand_from_camera(T_base_hand).astype(np.float32),
        p_cam_wrist[None, :3].astype(np.float32))[0]
    got_tripod = transform_points(
        tripod.hand_from_camera(T_base_hand).astype(np.float32),
        p_cam_tripod[None, :3].astype(np.float32))[0]

    expect = (invert_transform(T_base_hand) @ p_base)[:3]

    err_w = float(np.linalg.norm(got_wrist - expect))
    err_t = float(np.linalg.norm(got_tripod - expect))
    print(f"  wrist  chain error {err_w * 1e6:.3f} um")
    print(f"  tripod chain error {err_t * 1e6:.3f} um")
    print(f"  disagreement between the two cameras "
          f"{np.linalg.norm(got_wrist - got_tripod) * 1e6:.3f} um")
    # 1 um. transform_points returns float32, whose relative epsilon at a 0.5 m
    # radius is ~6e-8 m, so anything tighter is testing the dtype rather than the
    # maths. A transposed rotation or an un-inverted transform misplaces the
    # point by centimetres to metres, so this catches every failure that matters
    # with six orders of magnitude to spare.
    assert err_w < 1e-6 and err_t < 1e-6, "extrinsic chain is wrong"
    _ = rng


def _tensor_layout() -> None:
    """896 object rows first with channel 3 hot, then 128 hand rows channel 4."""
    obj = np.tile(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (500, 1))
    hand = np.tile(np.array([[-1.0, -2.0, -3.0]], dtype=np.float32), (50, 1))

    pc = build_policy_cloud(obj, hand)
    assert pc.shape == (NUM_OBJECT_POINTS + NUM_HAND_POINTS, 5), pc.shape
    assert pc.dtype == np.float32

    o, h = pc[:NUM_OBJECT_POINTS], pc[NUM_OBJECT_POINTS:]
    assert np.allclose(o[:, 3], 1.0) and np.allclose(o[:, 4], 0.0), "object one-hot"
    assert np.allclose(h[:, 3], 0.0) and np.allclose(h[:, 4], 1.0), "hand one-hot"
    # Under-full classes are padded by repetition, never with zeros: a zero row
    # is a point at the gripper origin, which is a lie the network cannot detect.
    assert np.allclose(o[:, :3], [1.0, 2.0, 3.0]), "object padded with zeros"
    assert np.allclose(h[:, :3], [-1.0, -2.0, -3.0]), "hand padded with zeros"
    print(f"  layout ok: {pc.shape}, object rows [0:{NUM_OBJECT_POINTS}] "
          f"channel 3 hot, hand rows channel 4 hot")

    empty = build_policy_cloud(np.zeros((0, 3), np.float32), hand)
    assert empty.shape == (NUM_OBJECT_POINTS + NUM_HAND_POINTS, 5)
    print("  empty object class still produces a full tensor (caller must gate "
          "on FusedObservation.usable)")


def _source_provenance() -> None:
    """Sampling must carry each point's source camera with it, not beside it.

    Built so a mix-up cannot hide: camera 0's points sit at x=0 and camera 1's at
    x=10, so any row whose label disagrees with its own coordinate is caught.
    An off-by-one in the index mapping — the plausible bug — shows up as a few
    percent mislabelled, which a spot check would miss and this does not.
    """
    from pointcloud_multicam import build_policy_cloud
    from cloud_viewer import source_for_cloud

    rng = np.random.default_rng(1)
    n0, n1 = 700, 400                      # object: two cameras, disjoint in x
    obj = np.concatenate([
        np.column_stack([np.zeros(n0), rng.normal(size=n0), rng.normal(size=n0)]),
        np.column_stack([np.full(n1, 10.0), rng.normal(size=n1), rng.normal(size=n1)]),
    ]).astype(np.float32)
    obj_src = np.concatenate([np.zeros(n0, np.int8), np.ones(n1, np.int8)])

    m0, m1 = 90, 60
    hand = np.concatenate([
        np.column_stack([np.zeros(m0), rng.normal(size=m0), rng.normal(size=m0)]),
        np.column_stack([np.full(m1, 10.0), rng.normal(size=m1), rng.normal(size=m1)]),
    ]).astype(np.float32)
    hand_src = np.concatenate([np.zeros(m0, np.int8), np.ones(m1, np.int8)])

    pc, oi, hi = build_policy_cloud(obj, hand, return_index=True)
    src = source_for_cloud(oi, hi, obj_src, hand_src,
                           NUM_OBJECT_POINTS, NUM_HAND_POINTS)
    assert src is not None and len(src) == len(pc)

    # every row labelled camera 0 must have x == 0, camera 1 must have x == 10
    bad = int(((src == 0) & (pc[:, 0] != 0.0)).sum()
              + ((src == 1) & (pc[:, 0] != 10.0)).sum())
    assert bad == 0, f"{bad} rows carry the wrong camera label"
    print(f"  {len(pc)} rows, {int((src == 0).sum())} from cam0 / "
          f"{int((src == 1).sum())} from cam1, 0 mislabelled")

    # Over-full object class (1100 > 896) must subsample without replacement;
    # under-full hand class (150 < 128 is false, so also subsample) — check the
    # ratio survives roughly, i.e. sampling is uniform and not front-biased.
    frac = float((src[:NUM_OBJECT_POINTS] == 0).mean())
    assert 0.55 < frac < 0.72, f"object sampling looks biased: cam0 frac {frac:.3f}"
    print(f"  object rows are {frac:.1%} cam0 against a {n0 / (n0 + n1):.1%} input "
          "share — sampling is uniform, not front-biased")

    # An empty class has no provenance; the viewer must fall back, not crash.
    assert source_for_cloud(np.zeros(0, np.int64), hi, np.zeros(0, np.int8),
                            hand_src, NUM_OBJECT_POINTS, NUM_HAND_POINTS) is None
    print("  empty class -> None (viewer falls back to colouring by class)")


def _robot_exclusion() -> None:
    """The box must swallow the gripper body and spare the grasp volume."""
    cases = {
        "hand housing        (0, 0, -0.03)": ([0.0, 0.0, -0.03], False),
        "wrist behind hand   (0, 0, -0.20)": ([0.0, 0.0, -0.20], False),
        "object in fingers   (0, 0, +0.10)": ([0.0, 0.0, 0.10], True),
        "fingertip plane     (0, 0, +0.075)": ([0.0, 0.0, 0.075], True),
        "object beside hand  (0.15, 0, 0)": ([0.15, 0.0, 0.0], True),
        "far forearm         (0, 0, -0.40)": ([0.0, 0.0, -0.40], True),
    }
    for label, (p, want_keep) in cases.items():
        keep = bool(ROBOT_EXCLUSION.keep_mask(np.array([p], dtype=np.float32))[0])
        assert keep == want_keep, f"{label}: kept={keep}, wanted {want_keep}"
        print(f"  {label:36s} -> {'keep' if keep else 'DROP'}")


def _hand_connected_clustering() -> None:
    """A staged handover where the right answer is known point by point.

    The scene is the one the sphere cannot handle: a hand holding a 0.24 m bar,
    over a table, with a distractor blob floating nearby. A radius test either
    keeps the table (loose) or amputates the bar (tight). Connectivity should
    keep the whole bar and drop both the table and the distractor, so the two
    assertions are checked separately — a filter that drops everything would
    pass a test that only looked at the table.
    """
    from pointcloud_pipeline import hand_connected_object_points

    from pointcloud_multicam import FIXED_PARAMS, WRIST_PARAMS

    rng = np.random.default_rng(7)

    # Hand at the origin: a solid ball of radius 0.04, not a Gaussian blob. The
    # difference matters — a Gaussian's tails have no edge, so "how far is the
    # hand from the table" would not be a well-defined quantity and the standoff
    # cases below could not be staged.
    v = rng.normal(size=(2000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    hand = (v * 0.04 * rng.uniform(0, 1, (2000, 1)) ** (1 / 3)).astype(np.float32)

    # A bar gripped at one end, running out to 0.24 m along +x. Touches the hand
    # so the two are one body, which is the physical fact being relied on.
    t = rng.uniform(0.01, 0.24, size=900)
    bar = np.stack([t,
                    rng.normal(0.0, 0.008, 900),
                    rng.normal(0.0, 0.008, 900)], axis=1).astype(np.float32)

    # Table 4 cm under the hand's underside, a plane wide enough to be well
    # inside any usable crop radius. Disconnected from the hand in 3D, and the
    # mass that used to be fed to the policy as object points.
    def make_table(z: float, n: int = 1500) -> np.ndarray:
        return np.stack([rng.uniform(-0.25, 0.25, n),
                         rng.uniform(-0.25, 0.25, n),
                         np.full(n, z) + rng.normal(0, 0.002, n)],
                        axis=1).astype(np.float32)

    table = make_table(-0.08)

    # Something else in the scene entirely — the other hand, a mug on the bench.
    distractor = rng.normal([0.0, 0.18, 0.05], 0.02, size=(400, 3)).astype(np.float32)

    object_xyz = np.concatenate([bar, table, distractor], axis=0)
    truth_is_object = np.concatenate([
        np.ones(len(bar), bool),
        np.zeros(len(table), bool),
        np.zeros(len(distractor), bool)])

    kept, debug = hand_connected_object_points(object_xyz, hand, voxel_m=0.010)
    assert debug["applied"] and debug["fallback"] is None, debug

    # Match kept rows back to their source block by identity, since clustering
    # returns a subset in input order.
    keep_mask = np.zeros(len(object_xyz), bool)
    view = {tuple(p) for p in kept.tolist()}
    for i, p in enumerate(object_xyz.tolist()):
        keep_mask[i] = tuple(p) in view

    bar_kept = keep_mask[truth_is_object].mean()
    junk_kept = keep_mask[~truth_is_object].mean()
    print(f"  components found      : {debug['components']}")
    print(f"  bar points kept       : {bar_kept:.1%}  (want ~100%)")
    print(f"  table+distractor kept : {junk_kept:.1%}  (want 0%)")
    print(f"  dropped               : {debug['dropped']} of {len(object_xyz)}")
    assert bar_kept > 0.98, f"clustering amputated the held object ({bar_kept:.1%})"
    assert junk_kept == 0.0, f"table or distractor survived ({junk_kept:.1%})"

    # The far end of the bar is 0.24 m out — past every radius the pre-clustering
    # config used, which is the amputation this is meant to end.
    far = float(np.linalg.norm(kept, axis=1).max())
    assert far > 0.20, f"expected the bar's far end near 0.24 m, got {far:.3f}"
    print(f"  furthest kept point   : {far:.3f} m "
          f"(legacy object_max_radius_m was 0.10)")

    # Every failure path must return the input untouched rather than an empty
    # class: a dirty object cloud is degraded, an empty one is unusable.
    for label, kwargs in {
        "disabled (voxel 0)": dict(voxel_m=0.0),
        "no hand seed": dict(voxel_m=0.010),
    }.items():
        seed = hand if "disabled" in label else np.zeros((0, 3), np.float32)
        out, dbg = hand_connected_object_points(object_xyz, seed, **kwargs)
        assert len(out) == len(object_xyz) and not dbg["applied"], label
        print(f"  {label:22s}-> passthrough ({dbg['fallback']})")

    # A stray hand point on the table must not vote the tabletop in. This is the
    # single-pixel-of-mis-segmentation case min_hand_voxel_frac exists for.
    stray = np.concatenate([hand, np.array([[0.1, 0.1, -0.08]], np.float32)])
    kept2, _ = hand_connected_object_points(object_xyz, stray, voxel_m=0.010)
    assert len(kept2) == len(kept), (
        f"one stray hand point pulled in {len(kept2) - len(kept)} extra points")
    print(f"  stray hand point      -> still {len(kept2)} kept")

    # The reach that connects a held object to the hand is the same reach that
    # connects the hand to anything else, so both are worth pinning down at the
    # voxel sizes actually shipped. These are the numbers quoted in
    # hand_connected_object_points' docstring; the point of asserting them is
    # that changing a voxel size silently changes both, in opposite directions.
    # Note both probes read debug["fallback"], not len(). A cloud that stayed
    # connected and a cloud that detached ENTIRELY both come back at full
    # length, because total loss trips the passthrough — so length alone cannot
    # tell "kept everything" from "lost everything", which is exactly backwards
    # for measuring reach.
    def connected(points: np.ndarray, voxel: float) -> bool:
        _, dbg = hand_connected_object_points(points, hand, voxel_m=voxel)
        assert dbg["fallback"] in (None, "no-connected-object"), dbg
        return dbg["fallback"] is None

    print("  gap the hand can bridge to its own object:")
    for label, voxel in (("wrist", WRIST_PARAMS.cluster_voxel_m),
                         ("fixed", FIXED_PARAMS.cluster_voxel_m)):
        widest = -1.0
        for gap in np.arange(0.0, 0.041, 0.0025):
            x0 = 0.04 + gap
            g = np.stack([rng.uniform(x0, x0 + 0.20, 900),
                          rng.normal(0, 0.008, 900),
                          rng.normal(0, 0.008, 900)], axis=1).astype(np.float32)
            if not connected(g, voxel):
                break
            widest = gap
        print(f"    {label} @ {voxel * 1e3:.0f} mm voxel -> {widest * 1e3:.1f} mm")
        assert widest >= 0.005, (
            f"{label}: object detaches at a {widest * 1e3:.1f} mm gap, which a "
            "band of dropped depth at the occlusion boundary will exceed")

    # ... and the standoff the table needs before it stops merging. Scanned from
    # the top down for the LAST standoff that still merges, because the boundary
    # is quantised by where the voxel grid happens to fall and a single clean
    # sample partway up would otherwise read as the answer.
    print("  table standoff before it separates from the hand:")
    for label, voxel in (("wrist", WRIST_PARAMS.cluster_voxel_m),
                         ("fixed", FIXED_PARAMS.cluster_voxel_m)):
        merges_up_to = 0.0
        for standoff in np.arange(0.060, 0.0049, -0.0025):
            if connected(make_table(-0.04 - standoff, 3000), voxel):
                merges_up_to = standoff
                break
        print(f"    {label} @ {voxel * 1e3:.0f} mm voxel -> "
              f"clean above {merges_up_to * 1e3:.1f} mm")
        assert merges_up_to <= 0.035, (
            f"{label}: table still merges at a {merges_up_to * 1e3:.0f} mm "
            "standoff — the object class is not being cleaned at all")


def _arm_rejection() -> None:
    """The forearm must go and the held object must stay, including in pieces.

    This is the case connectivity provably cannot handle: the forearm is
    anatomically continuous with the hand, so it shares the hand's component by
    construction. The scene is laid out in the panda_hand frame with the robot
    base along -z, the human offering an object toward it, and the forearm
    running the other way.
    """
    from pointcloud_pipeline import (
        _MIN_GRASP_POINTS as _MIN_GRASP,
        reject_arm_clusters,
    )

    rng = np.random.default_rng(11)
    hand = rng.normal(0.0, 0.02, size=(500, 3)).astype(np.float32)

    # Robot base 0.8 m away along -z; the object is offered toward it.
    robot_origin = np.array([0.0, 0.0, -0.8])

    def bar(centre, extent, n=600):
        return (np.array(centre, np.float32)
                + rng.uniform(-1, 1, (n, 1)) * np.array(extent, np.float32)
                + rng.normal(0, 0.006, (n, 3))).astype(np.float32)

    obj = bar([0.0, 0.0, -0.07], [0.03, 0.03, 0.04])       # toward the robot
    forearm = bar([0.0, 0.0, 0.20], [0.015, 0.015, 0.10])  # away from it

    points = np.concatenate([obj, forearm])
    truth = np.concatenate([np.ones(len(obj), bool), np.zeros(len(forearm), bool)])

    keep, dbg = reject_arm_clusters(points, hand, robot_origin, voxel_m=0.012)
    assert dbg["applied"], dbg
    print(f"  blobs (points @ offset / lateral / arm-frac): "
          + "  ".join(f"{n}@{o:+.3f}/lat{r:.3f}/{f:.2f}"
                       for n, o, r, f in dbg["clusters"]))
    print(f"  object kept  : {keep[truth].mean():.1%}  (want 100%)")
    print(f"  forearm kept : {keep[~truth].mean():.1%}  (want 0%)")
    assert keep[truth].all(), "the held object was rejected as arm"
    assert not keep[~truth].any(), "the forearm survived"

    # A fragmented object — fingers and depth dropouts split one in practice —
    # must survive whole. This is what a "keep the single nearest blob" rule
    # would get wrong, and why the test is a threshold instead of an argmin.
    near = bar([0.0, 0.0, -0.05], [0.02, 0.02, 0.02], 300)
    far = bar([0.0, 0.0, -0.14], [0.02, 0.02, 0.02], 300)
    split = np.concatenate([near, far, forearm])
    keep2, dbg2 = reject_arm_clusters(split, hand, robot_origin, voxel_m=0.012)
    assert len(dbg2["clusters"]) >= 3, f"expected 3 separate blobs, {dbg2}"
    assert keep2[:600].all(), "an argmin rule would have kept only one fragment"
    assert not keep2[600:].any(), "the forearm survived the split-object case"
    print(f"  split object : both fragments kept ({keep2[:600].sum()} pts), "
          f"forearm dropped")

    # An object held with its body toward the human — a mug by the handle —
    # scores negative and must still be kept. This is what the threshold buys
    # over a sign test, and it is the reason the default is 0.07 and not 0.
    #
    # It has to be staged clear of the forearm. The first version of this scene
    # overlapped the two, they merged into a single blob, and the test failed
    # for a reason that had nothing to do with the threshold — which is itself
    # the limitation to remember: this separates object from arm only when they
    # are separate blobs. The hand usually guarantees that, sitting between them
    # and excluded from its own class, but an object resting against the
    # forearm is scored with it and shares its fate.
    mug = bar([0.0, 0.0, 0.03], [0.03, 0.03, 0.025], 400)
    keep3, dbg3 = reject_arm_clusters(np.concatenate([mug, forearm]), hand,
                                      robot_origin, voxel_m=0.012)
    assert len(dbg3["clusters"]) == 2, (
        f"mug and forearm merged into {len(dbg3['clusters'])} blob(s); restage "
        f"the scene, this case cannot test the threshold: {dbg3}")
    print(f"  mug-by-handle: offsets "
          + " ".join(f"{o:+.3f}" for _, o, _, _ in dbg3["clusters"])
          + f", kept {keep3[:400].mean():.0%} of the mug")
    assert keep3[:400].all(), (
        "an object held with its body toward the human was rejected — the "
        "offset threshold is too tight")
    assert not keep3[400:].any(), "the forearm survived the mug case"

    # When NOTHING looks like a held object the class goes empty, and that is the
    # intended answer. The earlier version kept the least-bad blob instead; on
    # hardware that fired every frame and promoted an 87%-arm blob to be the
    # object class, silently, because downstream an arm-shaped object cloud is
    # indistinguishable from a real one. Empty trips FusedObservation.usable, so
    # the runner holds and the viewer says NOT USABLE — a visible stall the
    # operator can fix by loosening the bounds.
    behind = bar([0.0, 0.0, 0.20], [0.02, 0.02, 0.05], 300)
    keep4, dbg4 = reject_arm_clusters(behind, hand, robot_origin, voxel_m=0.012)
    assert not keep4.any() and dbg4["fallback"] == "all-arm", dbg4
    print(f"  nothing resembling a held object -> class emptied "
          f"({dbg4['fallback']}), caller holds the frame")

    for label, kwargs, seed in (
        ("disabled (voxel 0)", dict(voxel_m=0.0), hand),
        ("no hand", dict(voxel_m=0.012), np.zeros((0, 3), np.float32)),
    ):
        k, d = reject_arm_clusters(points, seed, robot_origin, **kwargs)
        assert k.all() and not d["applied"], label
        print(f"  {label:22s}-> passthrough ({d['fallback']})")

    # THE FLICKER REGRESSION. Reported from hardware as "sometimes the arm
    # appears as object, most times not". The cause is a shell of hand-surface
    # points that the mask misses, which wraps the hand and fuses the object blob
    # to the forearm blob; the ring's continuity varies frame to frame, so the
    # arm blinks in and out. `hand_margin_px` now removes that shell upstream,
    # but the decision must not DEPEND on it having worked — the mask can be off
    # by more than the margin. Sweeping the ring's coverage stands in for the
    # jitter: the answer has to be flat across it, because a rule that is right
    # at one coverage and wrong at another is precisely a flicker.
    sv = rng.normal(size=(4000, 3))
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ring = (sv * 0.047).astype(np.float32)

    # A forearm starting AT the wrist, which is where one is. The forearm staged
    # above sits further back so that the mug case could be separated from it;
    # at that distance the shell cannot reach it and the sweep below would pass
    # without ever forming the bridge it exists to test.
    wrist_arm = bar([0.0, 0.0, 0.16], [0.018, 0.018, 0.09], 600)

    print("  mask-shell bridge sweep (the flicker):")
    bridged = 0
    for coverage in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        shell = ring[rng.random(len(ring)) < coverage]
        staged = (np.concatenate([obj, wrist_arm, shell]) if len(shell)
                  else np.concatenate([obj, wrist_arm]))
        k, d = reject_arm_clusters(staged, hand, robot_origin, voxel_m=0.012)
        obj_kept, arm_kept = k[:len(obj)].mean(), k[len(obj):len(obj) * 2].mean()
        merged = len(d["clusters"]) == 1
        bridged += merged
        print(f"    shell {coverage:>4.0%}  blobs={len(d['clusters'])}"
              f"{' MERGED' if merged else '       '}  "
              f"object {obj_kept:>5.0%}  arm {arm_kept:>5.0%}")
        assert obj_kept == 1.0, (
            f"shell {coverage:.0%}: lost {1 - obj_kept:.0%} of the object")
        assert arm_kept < 0.05, (
            f"shell {coverage:.0%}: {arm_kept:.0%} of the forearm survived — "
            "the bridge fused it to the object and it was judged as one body")

    # Without this the sweep can go green by never bridging at all, which is how
    # the first version of it passed.
    assert bridged >= 4, (
        f"the shell only fused the blobs at {bridged} of 6 coverages; this "
        "scene is no longer reproducing the flicker and the sweep above proves "
        "nothing")

    # THE GRASP VETO. Reported from hardware as "the object-labelled cloud
    # disappears as the gripper closes on the object, even though hand and
    # object are clearly visible in the image". As the gripper converges, its own
    # points and the object's necessarily touch and merge into one blob; the
    # robot's links run laterally, so they carry the merged blob past
    # _BLOB_ARM_FRAC and the object was discarded with them — the object class
    # emptying exactly when the grasp became possible.
    #
    # Staged in the REAL panda_hand geometry, which the first version of this
    # scene got wrong by putting the human hand at the gripper origin: the human
    # hand is BEYOND the fingers, the object is between them, and the robot base
    # is behind. Get that backwards and the object scores as arm on its own,
    # which tests something else entirely.
    from pointcloud_multicam import GRASP_REGION

    grasp_hand = (np.array([0.0, 0.0, 0.20], np.float32)
                  + rng.normal(0, 0.02, (400, 3))).astype(np.float32)
    grasp_base = np.array([0.0, 0.0, -0.60])

    # Spans the finger gap so it touches BOTH sides: with a compact blob only
    # one slab happened to connect, and a half-merged blob sat at 0.516 arm —
    # under the threshold, so nothing was dropped and the test proved nothing.
    held = bar([0.0, 0.0, 0.095], [0.012, 0.045, 0.012], 120)
    # Robot: a thin bridge from beside the object out to the bulk of the arm,
    # which sits beyond max_lateral_m and so scores as arm. `bar` draws a LINE
    # (one scalar per point across all three extents), not a box — slabs offset
    # in z therefore never touched the object at all, and the blobs stayed
    # separate while the scene looked right on paper.
    bridge = np.concatenate([
        bar([0.0, +0.085, 0.095], [0.0, 0.045, 0.0], 60),
        bar([0.0, -0.085, 0.095], [0.0, 0.045, 0.0], 60),
    ])
    mass = np.concatenate([
        bar([0.0, +0.215, 0.095], [0.0, 0.085, 0.0], 500),
        bar([0.0, -0.215, 0.095], [0.0, 0.085, 0.0], 500),
    ])
    robot = np.concatenate([bridge, mass])
    merged = np.concatenate([held, robot])
    gmask = GRASP_REGION.contains(merged)
    assert gmask[:len(held)].sum() >= _MIN_GRASP, (
        f"scene is wrong: only {gmask[:len(held)].sum()} held points are inside "
        "the grasp region, so the veto is not being exercised")

    without, dbg_a = reject_arm_clusters(merged, grasp_hand, grasp_base,
                                         voxel_m=0.012)
    withveto, dbg_b = reject_arm_clusters(merged, grasp_hand, grasp_base,
                                          voxel_m=0.012, grasp_mask=gmask)
    assert len(dbg_a["clusters"]) == 1, (
        f"object and robot did not merge into one blob ({dbg_a['clusters']}); "
        "this scene is not reproducing the failure")
    print(f"  object merged with robot: {dbg_a['clusters']}")
    print(f"    no veto  -> object kept {without[:len(held)].mean():>5.0%}"
          f"   ({dbg_a['fallback'] or 'blob dropped whole'})")
    print(f"    veto     -> object kept {withveto[:len(held)].mean():>5.0%}"
          f"   robot kept {withveto[len(held):].mean():>5.0%}"
          f"   vetoed={dbg_b['grasp_vetoed']}")

    # The bug, reproduced: without the veto the object goes with the robot.
    assert without[:len(held)].mean() < 0.5, (
        "this scene no longer reproduces the failure — the merged blob is not "
        "being dropped whole, so the veto below proves nothing")
    # ... and fixed.
    assert withveto[:len(held)].all(), (
        f"the veto kept only {withveto[:len(held)].mean():.0%} of an object "
        "sitting between the fingers")
    assert dbg_b["grasp_vetoed"] >= 1, dbg_b
    # It must still cut the ARM MASS it was merged with, not wave the blob
    # through. Checked on the mass alone, not on all robot points: the bridge
    # sits inside the capsule, so the per-point rule keeps it with or without the
    # veto — asserting on every robot point credited the veto with capsule
    # behaviour and failed for a reason that had nothing to do with it.
    kept_mass = withveto[len(held) + len(bridge):].mean()
    assert kept_mass < 0.02, (
        f"the veto rescued {kept_mass:.0%} of the arm mass too; it spares the "
        "grasp region, not the whole blob")
    print(f"    arm mass kept {kept_mass:.0%}")

    # A few stray points between the fingers must NOT rescue an arm: depth noise
    # and calibration error put that many there routinely.
    stray = np.zeros(len(merged), bool)
    stray[np.where(gmask)[0][:_MIN_GRASP - 1]] = True
    _, dbg_c = reject_arm_clusters(merged, grasp_hand, grasp_base,
                                   voxel_m=0.012, grasp_mask=stray)
    assert dbg_c["grasp_vetoed"] == 0, (
        f"{_MIN_GRASP - 1} stray grasp points triggered the veto")
    print(f"    {_MIN_GRASP - 1} stray grasp points -> no veto")

    # INCLINATION. Reported from hardware as "fine if my forearm is horizontal,
    # but at some inclination I do see the forearm labelled as object". A signed
    # projection alone measures only the component along the hand->robot axis, so
    # it ranks an arm by how nearly it points away from the robot: at 90 degrees
    # a real forearm scored about -0.05 and was kept entirely. The lateral term
    # is what fixed it, and this sweep is the reason to keep it.
    #
    # The arm is staged as a cylinder starting AT the wrist, past the hand ball.
    # An earlier version ran it through the hand centroid, which merged it with
    # the object and measured the merged case while claiming to measure this one.
    print("  forearm inclination sweep (0 deg = straight away from the robot):")
    for deg in (0, 15, 30, 45, 60, 75, 90):
        r = np.random.default_rng(100 + deg)
        t = np.deg2rad(deg)
        d = np.array([0.0, np.sin(t), np.cos(t)])
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.cross(d, e1)
        length = r.uniform(0.075, 0.28, (600, 1))
        around = r.uniform(0, 2 * np.pi, (600, 1))
        limb = (length * d + 0.04 * np.cos(around) * e1
                + 0.04 * np.sin(around) * e2
                + r.normal(0, 0.004, (600, 3))).astype(np.float32)

        k, d_ = reject_arm_clusters(np.concatenate([obj, limb]), hand,
                                    robot_origin, voxel_m=0.012)
        obj_kept, arm_kept = k[:len(obj)].mean(), k[len(obj):].mean()
        print(f"    tilt {deg:>3}deg  blobs={len(d_['clusters'])}  "
              f"object {obj_kept:>5.0%}  arm {arm_kept:>5.0%}")
        assert obj_kept == 1.0, f"tilt {deg}: lost the object"
        assert arm_kept < 0.05, (
            f"tilt {deg}: {arm_kept:.0%} of the forearm survived. A forearm "
            "away from the hand->robot axis projects to nearly zero on it; "
            "only the lateral bound catches this one.")


def _viz_visible() -> None:
    """The 3D view must actually SHOW the cloud, not merely be handed it.

    This exists because the first version did not. Open3D scales its view from
    the bounding box of the geometry present when it was added, and the cloud is
    added empty; framing against that stale box put a real wrist-camera cloud
    (z = 0.4-1.0 m in the hand frame) outside the frustum, so the window drew the
    gripper and nothing else. The unit tests all passed — they used a cloud at
    z = 0.16 m, which happened to fall inside the stale box.

    So this renders and counts coloured pixels. Both distances are covered
    because they fail independently: near-only was the case that accidentally
    worked, far-only is the case that shipped broken.
    """
    import open3d as o3d  # noqa: F401  (import error = no viewer to test)

    from pointcloud_multicam import ROBOT_EXCLUSION, build_policy_cloud
    from dual_cloud_window import DualCloudWindow

    rng = np.random.default_rng(0)
    cases = {
        "near (object at the fingertips, z~0.13)": (
            rng.normal([0.0, 0.0, 0.13], 0.025, size=(900, 3)),
            rng.normal([0.0, 0.05, 0.20], 0.030, size=(300, 3))),
        "far  (wrist view of the scene, z~0.45-0.75)": (
            np.column_stack([rng.uniform(-0.15, 0.15, 900),
                             rng.uniform(-0.15, 0.15, 900),
                             rng.uniform(0.45, 0.75, 900)]),
            np.column_stack([rng.uniform(-0.10, 0.10, 300),
                             rng.uniform(-0.10, 0.10, 300),
                             rng.uniform(0.50, 0.80, 300)])),
        # Well outside the default camera box the window sets up before any data
        # arrives, which the two cases above are not — they sit close enough to
        # it that they stay in frame even when framing never runs, so on their
        # own they would not notice framing breaking again. A tripod's cloud
        # reaches this far in the hand frame routinely.
        "very far (tripod reach, z~1.5-2.5)": (
            np.column_stack([rng.uniform(-0.40, 0.40, 900),
                             rng.uniform(-0.40, 0.40, 900),
                             rng.uniform(1.50, 2.50, 900)]),
            np.column_stack([rng.uniform(-0.20, 0.20, 300),
                             rng.uniform(-0.20, 0.20, 300),
                             rng.uniform(1.60, 2.40, 300)])),
    }

    for label, (obj, hand) in cases.items():
        pc = build_policy_cloud(obj.astype(np.float32), hand.astype(np.float32))
        v = DualCloudWindow(["wrist"], exclusion_box=ROBOT_EXCLUSION,
                            context_max=100)
        if not v.alive:
            print("  no display; skipping")
            return
        try:
            for _ in range(5):
                v.update(pc, None, None)
                if not v.tick():
                    break
            cam = v._scene.camera
            view = np.asarray(cam.get_view_matrix())
            proj = np.asarray(cam.get_projection_matrix())
        finally:
            v.close()

        # Project the cloud through the camera's own matrices and count how much
        # lands inside the unit cube. This replaces counting coloured pixels in a
        # screenshot, which the previous window allowed and this one does not:
        # Filament queues render_to_image on its render thread, and under
        # run_one_tick it never completes (the reason --screenshot was dropped).
        #
        # It also tests the property more directly. The bug was that framing used
        # the scene's bounding box, which is empty when the geometry is added, so
        # a real cloud fell outside the frustum. "Is it inside the frustum" is
        # exactly that question; pixel counting was a proxy for it that also
        # depended on colours, tone mapping and point size.
        pts = np.concatenate([pc[:, :3], np.ones((len(pc), 1), np.float32)], axis=1)
        clip = pts @ view.T @ proj.T
        w = clip[:, 3:4]
        ok = (np.abs(w) > 1e-9).ravel()
        ndc = np.divide(clip[:, :3], w, out=np.zeros_like(clip[:, :3]), where=np.abs(w) > 1e-9)
        inside = ok & np.all(np.abs(ndc) <= 1.0, axis=1)
        frac = float(inside.mean())
        print(f"  {label}: {frac:.0%} of the cloud inside the frustum")
        assert frac > 0.90, (
            f"only {frac:.0%} of the cloud is inside the view frustum for the "
            f"{label} case — the view is framed somewhere the cloud is not.")


def _live(session: str | None, cameras: list[str]) -> None:
    import torch

    from pointcloud_multicam import HandSegmenter, MultiCameraPerception, build_rigs

    # hands-segmentation-pytorch sits BESIDE the repo, not inside it:
    # parents[2] is handover-sim2real, parents[3] is the workspace that holds it.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "hands-segmentation-pytorch"))
    from model import HandSegModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(__file__).resolve().parent / "checkpoint" / "cp1" / "checkpoint.ckpt"
    model = HandSegModel.load_from_checkpoint(str(ckpt), map_location="cpu").to(device).eval()

    T_hand_cam = np.array([[0.0, -1.0, 0.0, 0.036],
                           [1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.036],
                           [0.0, 0.0, 0.0, 1.0]])

    rigs = build_rigs(cameras, T_hand_cam_wrist=T_hand_cam, fixed_session=session)
    for rig in rigs:
        rig.camera.start()
        print(f"  started {rig.name} serial={rig.serial}")

    perception = MultiCameraPerception(rigs, HandSegmenter(model, device))
    try:
        # An identity pose is a lie for the fixed camera, so its points land
        # wherever the calibration says relative to the base — fine for a
        # plumbing check, meaningless as a geometry check. That is what the
        # offline tests above are for.
        obs = perception.observe(np.eye(4))
        print(f"  fused: object={len(obs.object_xyz)} hand={len(obs.hand_xyz)} "
              f"usable={obs.usable}")
        print(f"  per camera: {obs.summary()}")
        if obs.usable:
            pc = build_policy_cloud(obs.object_xyz, obs.hand_xyz)
            print(f"  tensor {pc.shape}, object extent "
                  f"{np.ptp(pc[:NUM_OBJECT_POINTS, :3], axis=0).round(3)} m")
    finally:
        for rig in rigs:
            rig.camera.stop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true", help="also open the cameras")
    p.add_argument("--viz", action="store_true",
                   help="also render the 3D view and assert the cloud is "
                        "actually visible in it (needs a display)")
    p.add_argument("--cameras", default="wrist,tripod")
    p.add_argument("--calib-session", default=None)
    args = p.parse_args()

    print("extrinsic chain agreement")
    _rig_transform_agreement()
    print("\npolicy tensor layout")
    _tensor_layout()
    print("\nper-point camera provenance")
    _source_provenance()
    print("\nrobot exclusion box")
    _robot_exclusion()
    print("\nhand-connected object clustering")
    _hand_connected_clustering()
    print("\nforearm rejection")
    _arm_rejection()
    print("\nall offline checks passed")

    if args.viz:
        print("\n3D view actually shows the cloud")
        _viz_visible()
        print("  visible at every distance")

    if args.live:
        print("\nlive capture")
        _live(args.calib_session, [c.strip() for c in args.cameras.split(",")])

    if args.viz or args.live:
        # A window was opened, so Open3D's Filament threads are alive and
        # interpreter finalization will hang or abort on them. See
        # dual_cloud_window.exit_without_finalizing.
        from dual_cloud_window import exit_without_finalizing
        exit_without_finalizing()


if __name__ == "__main__":
    main()
