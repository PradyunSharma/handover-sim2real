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
    import time

    import open3d as o3d  # noqa: F401  (import error = no viewer to test)

    from pointcloud_multicam import ROBOT_EXCLUSION, build_policy_cloud
    from cloud_viewer import CLASS_COLOURS, PolicyCloudViewer

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
    }

    for label, (obj, hand) in cases.items():
        pc = build_policy_cloud(obj.astype(np.float32), hand.astype(np.float32))
        v = PolicyCloudViewer(True, ["wrist"], ROBOT_EXCLUSION, update_hz=60.0)
        if not v.alive:
            print("  no display; skipping")
            return
        try:
            for _ in range(30):
                v.update(pc)
                v.poll()
                time.sleep(0.005)
            img = np.asarray(v._vis.capture_screen_float_buffer(do_render=True))
        finally:
            v.close()

        # Count pixels near each class colour. Points render as small squares, so
        # a visible 1024-point cloud is worth hundreds of pixels; a cloud outside
        # the frustum is worth exactly zero.
        n = {}
        for name, rgb in CLASS_COLOURS.items():
            d = np.linalg.norm(img - np.asarray(rgb)[None, None, :], axis=2)
            n[name] = int((d < 0.20).sum())
        print(f"  {label}: object px={n['object']}, hand px={n['hand']}")
        assert n["object"] > 200 and n["hand"] > 50, (
            f"cloud not visible for the {label} case — object px {n['object']}, "
            f"hand px {n['hand']}. The view is framed outside the cloud.")


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

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    T_hand_cam = np.array([[0.0, -1.0, 0.0, 0.036],
                           [1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.036],
                           [0.0, 0.0, 0.0, 1.0]])

    rigs = build_rigs(cameras, T_hand_cam_wrist=T_hand_cam, fixed_session=session)
    for rig in rigs:
        rig.camera.start()
        print(f"  started {rig.name} serial={rig.serial}")

    perception = MultiCameraPerception(rigs, HandSegmenter(model, preprocess, device))
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
    print("\nall offline checks passed")

    if args.viz:
        print("\n3D view actually shows the cloud")
        _viz_visible()
        print("  visible at both distances")

    if args.live:
        print("\nlive capture")
        _live(args.calib_session, [c.strip() for c in args.cameras.split(",")])


if __name__ == "__main__":
    main()
