"""Fuse several RGB-D cameras into the single [1024, 5] cloud the policy expects.

cp2 (run 12) was trained wrist-only, so `my_policy_runner.py` originally
segmented one image, deprojected it, and handed the result straight to the
network. cp3 (run 16) was trained with SIM.cfg_file
`examples/pretrain_multicam_wlr.yaml` — CAMERAS ["wrist", "left", "right"] — so
its observation is the union of three viewpoints. This module is the real-robot
counterpart of that fusion.

WHAT THE SIMULATOR ACTUALLY DOES, since this has to match it and not merely
resemble it (handover-sim/handover/handover_env.py::_get_point_states, then
handover_sim2real/policy.py::PointListener):

  1. Per camera, per class, segment and deproject. Classes are ordered and the
     order is fixed: [object, hand]. `COMPUTE_ROBOT_POINT_STATE` is False in
     that config, so the robot is NOT a class — the arm's own pixels are simply
     never deprojected by the fixed cameras.
  2. Move every camera's points into the panda_hand frame. The wrist camera's
     points are already there; the fixed cameras' are deprojected to world and
     multiplied by `t_hand_from_world`.
  3. `np.concatenate` the per-camera arrays PER CLASS. Nothing is balanced or
     weighted per camera — a camera that contributes more points has
     proportionally more influence. See "Why no per-camera balancing" below.
  4. PointListener shuffles each class (`np.random.choice` without replacement,
     which for size == N is a permutation), regularizes it to exactly 1024, then
     takes the first `int(1024 * ratio)`: 896 object, 128 hand. Concatenated,
     that is the [1024, 5] tensor, channel 3 = object one-hot, channel 4 = hand.

Steps 1-3 are reproduced below. Step 4 is `build_policy_cloud`, which collapses
the shuffle-regularize-slice into one sample-or-pad — equivalent in
distribution, since the shuffle upstream is what makes the "first N" slice
uniform in the first place.

THE PART THE SIMULATOR GETS FOR FREE AND WE DO NOT. In sim, "which points are
the object" is a segmentation-buffer lookup by body id: exact, and the robot arm
is excluded by construction. On the real robot there is no such oracle. What we
have is a hand segmentation network, so the object is defined negatively — the
non-hand points near the hand — exactly as `pointcloud_pipeline
.extract_hand_object_clouds` already does for the wrist camera. Two consequences
worth being explicit about, because they are the difference between this being a
faithful port and being merely a plausible one:

  * The robot's own gripper is not segmented away. From the wrist camera it is
    barely in frame and the sim's wrist view has the same property, so it
    mattered little for cp2. From a side camera the arm is large, and as it
    closes on the object it enters the crop radius and gets labelled "object" —
    a class the training data never contained. `ROBOT_EXCLUSION` below removes
    the gripper body using the pose we already know exactly.
  * "Object" here means "non-hand points forming one connected body with the
    hand, within `object_max_radius_m` of its centroid". The connectivity test
    (`pointcloud_pipeline.hand_connected_object_points`) is what keeps the table
    and the background out; before it existed the radius alone had to, and it
    could not do that and contain a long object at the same time. A distant
    camera sees more of the scene through the same sphere and its depth is
    noisier, so both the radius and the clustering voxel are per-camera
    parameters, not global ones.

WHY NO PER-CAMERA BALANCING, given the temptation to add it. The sim renders all
three cameras at 224x224, so a close camera contributes many more points per
object than a distant one, and the union is dominated by whichever view is
nearest. Reproducing that bias means concatenating raw, which is what step 3
does. Both real cameras here run at the same 640x480, so the near/far ratio is
preserved up to the same constant factor and raw concatenation stays faithful.
Set `per_camera_cap` if you ever run the two cameras at different resolutions —
at that point the ratio is no longer the sim's and raw concatenation is no
longer the neutral choice.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

_SIM2REAL_DIR = Path(__file__).resolve().parent
if str(_SIM2REAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM2REAL_DIR))

from pointcloud_pipeline import (  # noqa: E402
    _sample_or_pad_points,
    extract_hand_object_clouds,
    reject_arm_clusters,
)
from transforms import invert_transform, transform_points  # noqa: E402

# camera.py imports pyrealsense2 at module scope, which only the robot PC's
# `handover-rs` env has. The fusion GEOMETRY — extrinsic chains, the policy
# tensor layout, the robot exclusion box — is pure numpy and is exactly what you
# want to be able to check on a machine with no cameras attached, so the driver
# import is deferred into build_rigs() rather than paid at import time.
if TYPE_CHECKING:
    from camera import RealSenseCamera


# ── the policy tensor ────────────────────────────────────────────────────────

# POLICY.POINT_STATE_YCB_RATIO = 0.875 against RL_TRAIN.uniform_num_pts = 1024.
# pretrain_multicam_wlr.yaml deliberately keeps 2 classes, so POINT_STATE_RATIOS
# stays a 2-element list and PointListener ignores it (policy.py:34 only honours
# it above 2 entries) — the split is the same 896/128 the wrist-only runs used.
NUM_OBJECT_POINTS = 896
NUM_HAND_POINTS = 128
PC_CHANNELS = 5


def _sample_indices(n: int, target: int) -> np.ndarray:
    """Which rows `_sample_or_pad_points` would pick, as indices.

    Same three cases and the same RNG calls, so the points chosen are
    distributed identically — this exists only so that a parallel per-point
    array (which camera each point came from) can be carried through the sample
    instead of being discarded. Empty input has no indices; the caller
    substitutes zeros, as _sample_or_pad_points does.
    """
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == target:
        return np.arange(n, dtype=np.int64)
    if n > target:
        return np.random.choice(n, size=target, replace=False)
    return np.concatenate(
        [np.arange(n, dtype=np.int64),
         np.random.choice(n, size=target - n, replace=True)])


def build_policy_cloud(
    object_xyz: np.ndarray,
    hand_xyz: np.ndarray,
    num_object_points: int = NUM_OBJECT_POINTS,
    num_hand_points: int = NUM_HAND_POINTS,
    return_index: bool = False,
):
    """[1024, 5], panda_hand frame: 896 object rows then 128 hand rows.

    Channel 3 is the object one-hot and channel 4 the hand one-hot, because
    PointListener._process_pointcloud writes `point_state_[3 + i] = 1` for class
    index i and the env's class order is [object, hand]. Note this is the
    OPPOSITE of `pointcloud_pipeline.build_policy_point_tensor`, which puts hand
    first with hand = [1, 0] — that function belongs to the CVPR2023 runner and
    is not interchangeable with this one.

    With return_index, also returns the per-class row indices into the inputs, so
    a caller can map each of the 1024 rows back to the camera it came from. That
    is what lets the 3D viewer colour by source: the sampling is what would
    otherwise destroy provenance.
    """
    oi = _sample_indices(len(object_xyz), num_object_points)
    hi = _sample_indices(len(hand_xyz), num_hand_points)

    obj = (object_xyz[oi].astype(np.float32) if len(oi)
           else np.zeros((num_object_points, 3), dtype=np.float32))
    hand = (hand_xyz[hi].astype(np.float32) if len(hi)
            else np.zeros((num_hand_points, 3), dtype=np.float32))

    obj_labels = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (num_object_points, 1))
    hand_labels = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (num_hand_points, 1))

    pc = np.concatenate(
        [np.concatenate([obj, obj_labels], axis=1),
         np.concatenate([hand, hand_labels], axis=1)],
        axis=0,
    ).astype(np.float32)

    return (pc, oi, hi) if return_index else pc


# ── robot self-exclusion ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RobotExclusionBox:
    """Axis-aligned box in the panda_hand frame whose contents are discarded.

    Replaces, approximately, what `COMPUTE_ROBOT_POINT_STATE: False` does for
    free in simulation: keep the robot out of the object class. Only the fixed
    cameras need it — the wrist camera is mounted on the very body this removes
    and barely sees it.

    THE BOUNDS ARE CHOSEN TO SPARE THE GRASP. In panda_hand the fingers extend
    along +z and the fingertips sit at z = 0.075..0.105 (GA-DDPG's control
    points, core/utils.get_control_point_tensor). An object being grasped is
    therefore at z ~ 0.10, right between them. A box that swallowed the finger
    volume would delete the object at exactly the moment the side cameras exist
    to see it — so the ceiling is z = +0.02, which clears the hand housing
    (z ~ -0.06..0) and the wrist behind it while leaving the whole finger
    volume untouched. The arm further back than -0.30 m is outside
    `object_max_radius_m` of the human hand anyway and needs no box.
    """
    half_x: float = 0.09
    half_y: float = 0.09
    z_min: float = -0.30
    z_max: float = 0.02

    def keep_mask(self, points_hand: np.ndarray) -> np.ndarray:
        """True for points to KEEP, i.e. outside the box."""
        if len(points_hand) == 0:
            return np.zeros(0, dtype=bool)
        inside = (
            (np.abs(points_hand[:, 0]) < self.half_x)
            & (np.abs(points_hand[:, 1]) < self.half_y)
            & (points_hand[:, 2] > self.z_min)
            & (points_hand[:, 2] < self.z_max)
        )
        return ~inside


ROBOT_EXCLUSION = RobotExclusionBox()


@dataclass(frozen=True)
class GraspRegion:
    """The volume between the fingers, in the panda_hand frame.

    The counterpart to RobotExclusionBox: that says where the robot IS, this
    says where the object must be. Anything here at grasp time is the thing
    being grasped — that is what the fingers closing on it means — so no
    heuristic downstream is allowed to reason it away.

    It exists because one did. `reject_arm_clusters` discards a blob whole when
    enough of it scores as arm, and as the gripper converges on the object the
    two clouds necessarily touch and merge into one blob. The robot's own links
    run laterally and behind, so they score as arm in bulk; past the threshold
    the object went out with them, and the object class emptied exactly as the
    grasp became possible. Approaching is when the filter matters and contact is
    when it must not fire.

    The bounds are deliberately loose. GA-DDPG's control points put the
    fingertips at z = 0.075..0.105 and the fingers open to 0.08 total, so a tight
    box would be |xy| < 0.04, z in 0.075..0.105. Every bound here is wider,
    because this region only ever KEEPS points: being generous costs a little
    arm surviving in the object class, while being tight costs the object at the
    moment it matters most. That asymmetry also makes it robust to the
    calibration error it has to survive — with a fixed camera alone, the cloud
    is placed through inv(T_base_hand) @ T_base_color and moves relative to this
    box by the full hand-eye and pose error.
    """
    half_xy: float = 0.055
    z_min: float = 0.050
    z_max: float = 0.130

    def contains(self, points_hand: np.ndarray) -> np.ndarray:
        if len(points_hand) == 0:
            return np.zeros(0, dtype=bool)
        return (
            (np.abs(points_hand[:, 0]) < self.half_xy)
            & (np.abs(points_hand[:, 1]) < self.half_xy)
            & (points_hand[:, 2] > self.z_min)
            & (points_hand[:, 2] < self.z_max)
        )


GRASP_REGION = GraspRegion()


# ── per-camera extraction parameters ─────────────────────────────────────────

@dataclass(frozen=True)
class ExtractionParams:
    """Arguments to `extract_hand_object_clouds` for one camera.

    Split per camera because a wrist view at 0.3 m and a tripod view at 1.0 m
    are not the same measurement problem: the D435's depth noise grows roughly
    with the square of range, so the far camera needs a looser crop to keep the
    object at all, a stride that does not drown the near camera in points, and
    more segmentation resolution to find a hand that subtends a tenth of the
    pixels.
    """
    crop_radius_m: float = 0.25
    object_max_radius_m: float = 0.22
    # 0 disables clustering and restores the pure radius behaviour. See
    # WRIST_PARAMS_LEGACY.
    cluster_voxel_m: float = 0.010
    cluster_min_hand_frac: float = 0.05
    # Pixels around the hand mask belonging to neither class. Kills the shell of
    # hand-surface points that otherwise bridges the held object to the forearm.
    # In pixels rather than metres because the cause — mask erosion and nearest-
    # neighbour upsampling from the segmenter's square input — is pixel-scale
    # and does not care how far away the hand is.
    hand_margin_px: int = 5
    seg_input_px: int = 256
    min_depth_m: float = 0.10
    max_depth_m: float = 1.50
    full_cloud_stride: int = 2
    hand_cloud_stride: int = 1
    min_hand_points: int = 100
    min_object_points: int = 80


# THE RADII ARE LOOSE ON PURPOSE, and only safe because clustering is on. With
# the object defined by a sphere alone, `object_max_radius_m` had to do two
# incompatible jobs: exclude the table (wants to be small) and contain the whole
# object (wants to be large). At 0.10 m the first job won, and any object longer
# than about 20 cm held at one end — the drill, the wood block, a banana — had
# its far half deleted every frame. Connectivity now does the excluding, so the
# radius can be set past the biggest object we expect to see and act purely as a
# workspace bound.
WRIST_PARAMS = ExtractionParams()

# Tripod, ~1 m out. Wider crop and object radius because at that range a 1 cm
# depth error is normal and a sphere around a noisy hand centroid starts
# clipping the object itself. Point counts scale with solid angle, so the same
# object subtends far fewer pixels here — the minimum-points floors drop
# accordingly, otherwise every frame would fall through to the stale-cloud
# fallback and the fusion would quietly become wrist-only again.
#
# The voxel is coarser for the same reason the radii are: connectivity has to
# survive the range noise, and a gap opened by a few millimetres of depth error
# would otherwise disconnect the object from the hand that is holding it.
#
# `seg_input_px` is where most of the difference lives. The hand model sees the
# frame downscaled to a square, so a hand at 1.5 m lands on a few dozen pixels
# at 256 and the mask thins out until the hand class stops meeting its floor.
# Measured on the RTX 2000 Ada in this machine, per perception pass:
#   2 cams @256 batched            15.5 ms   (what this replaces)
#   wrist @256 + tripod @384       26.0 ms
#   wrist @256 + tripod @512       40.8 ms
# against a runner step of roughly 270 ms. 384 buys 2.25x the pixels for 4% of
# the step; 512 buys 4x for 9%. 384 is the default and --fixed-seg-px moves it.
FIXED_PARAMS = ExtractionParams(
    crop_radius_m=0.30,
    object_max_radius_m=0.26,
    cluster_voxel_m=0.012,
    # One pixel less than the wrist's. The shell the band removes is a
    # pixel-scale artefact and is the same width here, but what it COSTS is
    # metric: at 640x480 a pixel spans roughly 0.8 mm at the wrist's 0.4 m and
    # 2.5 mm at the tripod's 1.5 m, so the same 5 px eats 4 mm of object surface
    # near the wrist and 12 mm here. On a small object at range that is a real
    # fraction of what the camera can see of it.
    hand_margin_px=4,
    seg_input_px=384,
    min_depth_m=0.25,
    max_depth_m=2.50,
    full_cloud_stride=2,
    hand_cloud_stride=1,
    min_hand_points=40,
    min_object_points=30,
)

# Pre-clustering behaviour, kept so `--no-cluster` is a real A/B and not an
# approximation of one: turning connectivity off without also restoring the
# tight radii would leave the object class bounded by a 0.22 m sphere with
# nothing filtering it, which is worse than either configuration on its own.
WRIST_PARAMS_LEGACY = ExtractionParams(
    crop_radius_m=0.12, object_max_radius_m=0.10, cluster_voxel_m=0.0,
    hand_margin_px=0)

FIXED_PARAMS_LEGACY = ExtractionParams(
    crop_radius_m=0.16,
    object_max_radius_m=0.13,
    cluster_voxel_m=0.0,
    hand_margin_px=0,
    seg_input_px=384,
    min_depth_m=0.25,
    max_depth_m=2.50,
    min_hand_points=40,
    min_object_points=30,
)


# ── cameras ──────────────────────────────────────────────────────────────────

@dataclass
class CameraRig:
    """One physical camera plus the transform that puts its points in the hand.

    `kind` decides how that transform is obtained, and the two cases are
    genuinely different, not two spellings of the same thing:

      eye_in_hand  T_hand_cam is CONSTANT. The cloud needs no robot pose at all;
                   a pose error cannot contaminate it. This is the wrist camera.
      fixed        T_base_cam is constant, so T_hand_cam = inv(T_base_hand) @
                   T_base_cam varies every step and the cloud inherits the full
                   error of BOTH the hand-eye calibration and the reported EE
                   pose. This is the tripod camera.
    """
    name: str
    kind: str                                  # "eye_in_hand" | "fixed"
    camera: "RealSenseCamera"
    params: ExtractionParams
    T_hand_cam: Optional[np.ndarray] = None    # eye_in_hand only
    T_base_cam: Optional[np.ndarray] = None    # fixed only
    exclude_robot: bool = False
    serial: Optional[str] = None

    # per-frame fallback memory, exactly as the single-camera loop kept it
    _last_hand_xyz: Optional[np.ndarray] = field(default=None, repr=False)
    _last_object_xyz: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.kind == "eye_in_hand":
            if self.T_hand_cam is None:
                raise ValueError(f"{self.name}: eye_in_hand rig needs T_hand_cam")
        elif self.kind == "fixed":
            if self.T_base_cam is None:
                raise ValueError(
                    f"{self.name}: fixed rig needs T_base_cam — run the hand-eye "
                    "calibration in 'camera calibration/' and pass its "
                    "T_base_color.npy")
        else:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")

    def hand_from_camera(self, T_base_hand: np.ndarray) -> np.ndarray:
        if self.kind == "eye_in_hand":
            return self.T_hand_cam
        return invert_transform(T_base_hand) @ self.T_base_cam

    def reset(self) -> None:
        self._last_hand_xyz = None
        self._last_object_xyz = None


# ── hand segmentation ────────────────────────────────────────────────────────

def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob.

    The segmenter fires on any skin-coloured region, so a forearm, a face at the
    edge of frame, or the operator's other hand all come back as hand pixels.
    Their centroid would drag `hand_center` — and with it the object crop —
    somewhere with no object in it. Keeping one blob is what makes the centroid
    mean "the hand doing the handover".
    """
    import cv2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Binarize, keep the largest blob, then open/close to fill speckle.

    Lifted verbatim from my_policy_runner so the wrist camera's mask is
    processed exactly as it was for cp2, and the tripod camera gets the same
    treatment rather than a second, subtly different one.
    """
    import cv2

    mask = (mask > 0).astype(np.uint8)
    mask = largest_component(mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def overlay_mask(color_bgr: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    """Tint the segmented hand green over the colour image.

    Shared by the runner and test_perception_viz so both draw the mask
    identically — a debugging view that renders differently from the thing being
    debugged is worse than none.
    """
    overlay = color_bgr.copy()
    hand_color = np.zeros_like(color_bgr)
    hand_color[:, :, 1] = 255
    blended = (alpha * hand_color + (1.0 - alpha) * overlay).astype(np.uint8)
    return np.where(mask[..., None] > 0, blended, overlay)


DEFAULT_SEG_INPUT_PX = 256


def make_seg_preprocess(px: int):
    """The hand model's input transform at a given square resolution.

    256 is what cp1 was trained and previously run at; the network is fully
    convolutional so a larger input is a legitimate inference-time choice, not a
    reinterpretation of the weights. ImageNet normalization is fixed by the
    ResNet-50 backbone and is not a tunable.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((px, px)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class HandSegmenter:
    """The hand-segmentation network, run over every camera in one pass per size.

    Batching is not a micro-optimisation here. The runner's step is
    observe-then-act, and a per-image forward puts the whole network latency
    between the depth frames and the action computed from them — which is a real
    position error while the arm is moving, not just slowness.

    Cameras at the same `seg_input_px` still share one forward. Cameras at
    different resolutions cannot: a batch is one tensor and one tensor has one
    spatial size. Running the wrist at 256 and the tripod at 384 therefore costs
    two forwards, 26.0 ms against the 15.5 ms of a single batched pair — paid
    because the tripod's hand mask at 256 was thin enough to fall through the
    `min_hand_points` floor, and everything downstream is anchored on the hand
    centroid. Give both cameras the same size and this collapses back to one
    call with no code change.
    """

    def __init__(self, model, device: str, default_px: int = DEFAULT_SEG_INPUT_PX):
        self._model = model
        self._device = device
        self._default_px = int(default_px)
        self._preprocess: dict[int, Any] = {}

    def _transform(self, px: int):
        if px not in self._preprocess:
            self._preprocess[px] = make_seg_preprocess(px)
        return self._preprocess[px]

    def __call__(self, images_rgb: Sequence[np.ndarray],
                 sizes: Optional[Sequence[int]] = None) -> list[np.ndarray]:
        import cv2
        import torch

        if not images_rgb:
            return []
        if sizes is None:
            sizes = [self._default_px] * len(images_rgb)
        if len(sizes) != len(images_rgb):
            raise ValueError(
                f"got {len(images_rgb)} images but {len(sizes)} input sizes")

        groups: dict[int, list[int]] = {}
        for i, px in enumerate(sizes):
            groups.setdefault(int(px), []).append(i)

        out: list[Optional[np.ndarray]] = [None] * len(images_rgb)
        for px, indices in groups.items():
            transform = self._transform(px)
            batch = torch.stack([transform(images_rgb[i]) for i in indices])
            batch = batch.to(self._device, non_blocking=True)
            with torch.inference_mode():
                pred = self._model(batch).argmax(1).cpu().numpy().astype(np.uint8)
            for slot, i in enumerate(indices):
                img = images_rgb[i]
                m = cv2.resize(pred[slot], (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
                out[i] = normalize_mask(m)
        return out  # type: ignore[return-value]


# ── fusion ───────────────────────────────────────────────────────────────────

@dataclass
class FusedObservation:
    object_xyz: np.ndarray                 # [N, 3] panda_hand frame
    hand_xyz: np.ndarray                   # [M, 3] panda_hand frame
    per_camera: dict[str, dict[str, Any]]  # name -> counts / fallback flags
    # Which rig each point came from, as an index into `camera_names`. Carried
    # so the 3D viewer can colour by source — with two cameras fused into one
    # cloud, "the tripod's contribution is in the wrong place" is invisible
    # unless you can see the two apart.
    object_source: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    hand_source: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    camera_names: tuple = ()

    # Arm rejection is fusion-wide, not per camera, so it is reported here.
    # `arm_clusters` is (size, signed offset toward the robot) per blob, biggest
    # first — the numbers to look at when tuning the offset threshold, since a
    # forearm and a held object are only separable if their offsets differ.
    arm_dropped: int = 0
    arm_fallback: Optional[str] = None
    arm_clusters: tuple = ()
    # Points sitting between the fingers, and how many blobs that saved from the
    # whole-blob arm drop. A veto firing is the interesting event: it means the
    # object had merged with the robot and would otherwise have gone with it.
    grasp_points: int = 0
    grasp_vetoed: int = 0

    @property
    def usable(self) -> bool:
        """Both classes non-empty.

        PointListener falls back to the object cloud alone when the hand class is
        empty, but that fallback exists for a simulator that still had a correct
        object cloud. Here an empty hand class means the segmenter found nothing,
        which also means the object class — defined relative to the hand centroid
        — is whatever happened to be in front of the camera. Better to hold.
        """
        return len(self.object_xyz) > 0 and len(self.hand_xyz) > 0

    def summary(self) -> str:
        """e.g. `tripod:o517-284/h63` — 517 object points kept, 284 declustered.

        The subtraction is the number worth watching: it is the table and
        background mass that used to be fed to the policy as object points, and
        it going to zero every frame means clustering is not doing anything.
        """
        parts = []
        for name, d in self.per_camera.items():
            dropped = d.get("cluster_dropped", 0)
            parts.append(
                f"{name}:o{d['object']}"
                + (f"-{dropped}" if dropped else "")
                + f"/h{d['hand']}"
                + ("*" if d["used_last_hand"] or d["used_last_object"] else "")
            )
        if self.arm_dropped:
            parts.append(f"arm-{self.arm_dropped}")
        if self.grasp_points:
            parts.append(f"grasp{self.grasp_points}"
                         + (f"!veto{self.grasp_vetoed}" if self.grasp_vetoed else ""))
        if self.arm_fallback not in (None, "disabled"):
            parts.append(f"[arm:{self.arm_fallback}]")
        return "  ".join(parts)


class MultiCameraPerception:
    """Owns the rigs and the segmenter; turns one robot pose into one cloud.

    Usage mirrors the single-camera path it replaces:

        perception = MultiCameraPerception(rigs, segmenter)
        obs = perception.observe(T_base_hand)
        if obs.usable:
            pc = build_policy_cloud(obs.object_xyz, obs.hand_xyz)
    """

    def __init__(self, rigs: Sequence[CameraRig], segmenter: Callable,
                 per_camera_cap: Optional[int] = None,
                 arm_rejection: bool = True,
                 arm_voxel_m: float = 0.012,
                 arm_offset_m: float = 0.07,
                 arm_lateral_m: float = 0.12):
        if not rigs:
            raise ValueError("need at least one camera rig")
        self._rigs = list(rigs)
        self._segment = segmenter
        self._per_camera_cap = per_camera_cap
        # Arm rejection runs on the FUSED cloud, after every camera has been
        # brought into the hand frame, for two reasons. It needs the robot base,
        # which only exists once the pose is applied; and a blob seen by two
        # cameras is one blob only after the union — per camera, the same
        # forearm would be scored twice from two partial views.
        self._arm_rejection = arm_rejection
        self._arm_voxel_m = arm_voxel_m
        self._arm_offset_m = arm_offset_m
        self._arm_lateral_m = arm_lateral_m
        # Last (colour, hand mask) and last depth per camera, kept so a viewer
        # can draw exactly the frames this observation was computed from rather
        # than grabbing its own — which would be a different instant.
        self.last_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.last_depths: dict[str, np.ndarray] = {}

    @property
    def rigs(self) -> list[CameraRig]:
        return self._rigs

    def reset(self) -> None:
        for rig in self._rigs:
            rig.reset()

    def observe(self, T_base_hand: np.ndarray) -> FusedObservation:
        import cv2

        # 1. Grab every camera first, THEN segment. Interleaving grab/segment
        #    would stagger the exposures by a full network forward (~30 ms), so
        #    the two views would describe different instants of a moving hand.
        colors, depths = [], []
        for rig in self._rigs:
            color_bgr, depth_m, _ = rig.camera.get_frames()
            colors.append(color_bgr)
            depths.append(depth_m)

        masks = self._segment(
            [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in colors],
            sizes=[rig.params.seg_input_px for rig in self._rigs])

        object_parts, hand_parts = [], []
        object_src, hand_src = [], []
        per_camera: dict[str, dict[str, Any]] = {}

        for rig_i, (rig, color_bgr, depth_m, hand_mask) in enumerate(
                zip(self._rigs, colors, depths, masks)):
            self.last_frames[rig.name] = (color_bgr, hand_mask)
            self.last_depths[rig.name] = depth_m

            p = rig.params
            result = extract_hand_object_clouds(
                color_bgr=color_bgr,
                depth_m=depth_m,
                hand_mask=hand_mask,
                cam=rig.camera,
                last_hand_xyz=rig._last_hand_xyz,
                last_object_xyz=rig._last_object_xyz,
                crop_radius_m=p.crop_radius_m,
                object_max_radius_m=p.object_max_radius_m,
                cluster_voxel_m=p.cluster_voxel_m,
                cluster_min_hand_frac=p.cluster_min_hand_frac,
                hand_margin_px=p.hand_margin_px,
                min_depth_m=p.min_depth_m,
                max_depth_m=p.max_depth_m,
                full_cloud_stride=p.full_cloud_stride,
                hand_cloud_stride=p.hand_cloud_stride,
                min_hand_points=p.min_hand_points,
                min_object_points=p.min_object_points,
            )
            if len(result.hand_xyz) > 0:
                rig._last_hand_xyz = result.hand_xyz
            if len(result.object_xyz) > 0:
                rig._last_object_xyz = result.object_xyz

            # 2. Into the panda_hand frame — the frame the policy's cloud and its
            #    action deltas are both expressed in.
            T_hand_cam = rig.hand_from_camera(T_base_hand).astype(np.float32)
            obj_hand = transform_points(T_hand_cam, result.object_xyz)
            hand_hand = transform_points(T_hand_cam, result.hand_xyz)

            n_before = len(obj_hand)
            if rig.exclude_robot:
                # Object class only. The human hand is never inside the gripper,
                # and masking it there would just hide a calibration error we
                # would rather see.
                obj_hand = obj_hand[ROBOT_EXCLUSION.keep_mask(obj_hand)]

            if self._per_camera_cap is not None:
                obj_hand = _cap(obj_hand, self._per_camera_cap)
                hand_hand = _cap(hand_hand, self._per_camera_cap)

            if len(obj_hand):
                object_parts.append(obj_hand)
                object_src.append(np.full(len(obj_hand), rig_i, dtype=np.int8))
            if len(hand_hand):
                hand_parts.append(hand_hand)
                hand_src.append(np.full(len(hand_hand), rig_i, dtype=np.int8))

            per_camera[rig.name] = {
                "object": int(len(obj_hand)),
                "hand": int(len(hand_hand)),
                "robot_pts_removed": int(n_before - len(obj_hand)) if rig.exclude_robot else 0,
                "used_last_hand": bool(result.debug["used_last_hand"]),
                "used_last_object": bool(result.debug["used_last_object"]),
                # How much table and background connectivity threw away, and why
                # it did not run when it did not. Worth surfacing rather than
                # burying: a cluster step that silently falls back every frame
                # looks exactly like one that is working.
                "cluster_dropped": int(result.debug["cluster_dropped"]),
                "cluster_fallback": result.debug["cluster_fallback"],
                "margin_band": int(result.debug["margin_band_points"]),
            }

        # 3. Raw per-class union across cameras — see the module docstring on why
        #    this is deliberately unweighted.
        object_xyz = (np.concatenate(object_parts, axis=0) if object_parts
                      else np.zeros((0, 3), dtype=np.float32))
        hand_xyz = (np.concatenate(hand_parts, axis=0) if hand_parts
                    else np.zeros((0, 3), dtype=np.float32))
        object_source = (np.concatenate(object_src) if object_src
                         else np.zeros(0, np.int8))

        # 4. Drop the human forearm. It reaches here because it is anatomically
        #    continuous with the hand, so no connectivity test can separate it —
        #    only where it lies relative to the robot can.
        arm_debug: dict[str, Any] = {"dropped": 0, "fallback": "disabled",
                                     "clusters": [], "grasp_points": 0,
                                     "grasp_vetoed": 0}
        if self._arm_rejection:
            # The base origin, in the hand frame the cloud already lives in.
            robot_origin = invert_transform(T_base_hand)[:3, 3]
            keep, arm_debug = reject_arm_clusters(
                object_xyz, hand_xyz, robot_origin,
                voxel_m=self._arm_voxel_m, min_offset_m=self._arm_offset_m,
                max_lateral_m=self._arm_lateral_m,
                # Both are in the panda_hand frame already, so the grasp volume
                # is a fixed box here — no pose needed, and it is the one thing
                # in this function the robot knows exactly.
                grasp_mask=GRASP_REGION.contains(object_xyz))
            if not keep.all():
                object_xyz = object_xyz[keep]
                object_source = object_source[keep]

        return FusedObservation(
            object_xyz=object_xyz, hand_xyz=hand_xyz, per_camera=per_camera,
            object_source=object_source,
            hand_source=(np.concatenate(hand_src) if hand_src
                         else np.zeros(0, np.int8)),
            camera_names=tuple(r.name for r in self._rigs),
            arm_dropped=int(arm_debug["dropped"]),
            arm_fallback=arm_debug["fallback"],
            arm_clusters=tuple(arm_debug["clusters"]),
            grasp_points=int(arm_debug["grasp_points"]),
            grasp_vetoed=int(arm_debug["grasp_vetoed"]))


def _cap(points: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0 or len(points) <= cap:
        return points
    idx = np.random.choice(len(points), size=cap, replace=False)
    return points[idx]


# ── construction from the calibration folder ─────────────────────────────────

CALIB_DIR = _SIM2REAL_DIR / "camera calibration"


def load_fixed_extrinsics(session: str) -> np.ndarray:
    """T_base_color for a calibration session, with the frame check spelled out.

    The matrix maps camera -> robot BASE. It is the same base frame the runner's
    `T_base_hand` is in, because `capture_image_and_pose.py` recorded poses from
    `franka_states.O_T_EE` and on this robot F_T_EE has no translation — so
    O_T_EE IS panda_hand and both sides of `inv(T_base_hand) @ T_base_cam` agree.
    If the EE is ever reconfigured to a fingertip TCP, that stops being true and
    the calibration must be redone, not offset.
    """
    path = CALIB_DIR / "sessions" / session / "T_base_color.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"No calibration at {path}.\n"
            "The fixed camera's pose is not guessable — without it its points "
            "land somewhere arbitrary in the hand frame and the fused cloud is "
            "worse than the wrist camera alone. Run, in 'camera calibration/':\n"
            "  python generate_color_intrinsics.py --session <name> --role tripod\n"
            "  python capture_image_and_pose.py    --session <name> --role tripod\n"
            "  python calibrate.py                 --session <name>\n"
            "  python validate_calibration.py      --session <name>")
    T = np.load(path)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 matrix, got {T.shape}")
    return T.astype(np.float64)


def build_rigs(
    names: Sequence[str],
    T_hand_cam_wrist: np.ndarray,
    fixed_session: Optional[str] = None,
    serials: Optional[dict] = None,
    color_size: tuple[int, int] = (640, 480),
    depth_size: tuple[int, int] = (640, 480),
    fps: int = 30,
    exclude_robot: bool = True,
    cluster: bool = True,
    wrist_seg_px: Optional[int] = None,
    fixed_seg_px: Optional[int] = None,
    hand_margin_px: Optional[int] = None,
) -> list[CameraRig]:
    """Build rigs for the requested camera names, resolving serials and extrinsics.

    `serials` defaults to `calib_config.CAMERA_SERIALS`, so the tripod/wrist
    assignment has exactly one definition on this machine rather than one per
    script. With two RealSenses attached the serial is not optional: librealsense
    binds whichever it enumerates first, and silently calibrating against one
    camera while streaming the other produces a confident, entirely wrong cloud.
    """
    from camera import RealSenseCamera  # deferred: needs pyrealsense2

    if serials is None:
        sys.path.insert(0, str(CALIB_DIR))
        import calib_config as cfg  # noqa: E402
        serials = dict(cfg.CAMERA_SERIALS)

    wrist_params = WRIST_PARAMS if cluster else WRIST_PARAMS_LEGACY
    fixed_params = FIXED_PARAMS if cluster else FIXED_PARAMS_LEGACY
    if wrist_seg_px is not None:
        wrist_params = replace(wrist_params, seg_input_px=int(wrist_seg_px))
    if fixed_seg_px is not None:
        fixed_params = replace(fixed_params, seg_input_px=int(fixed_seg_px))
    if hand_margin_px is not None:
        wrist_params = replace(wrist_params, hand_margin_px=int(hand_margin_px))
        fixed_params = replace(fixed_params, hand_margin_px=int(hand_margin_px))

    rigs: list[CameraRig] = []
    for name in names:
        serial = serials.get(name)
        camera = RealSenseCamera(color_size=color_size, depth_size=depth_size,
                                 fps=fps, serial=serial)
        if name == "wrist":
            rigs.append(CameraRig(
                name=name, kind="eye_in_hand", camera=camera,
                params=wrist_params, T_hand_cam=T_hand_cam_wrist,
                exclude_robot=False, serial=serial))
        else:
            if fixed_session is None:
                raise ValueError(
                    f"camera {name!r} is a fixed camera and needs a calibration "
                    "session (--calib-session)")
            rigs.append(CameraRig(
                name=name, kind="fixed", camera=camera,
                params=fixed_params,
                T_base_cam=load_fixed_extrinsics(fixed_session),
                exclude_robot=exclude_robot, serial=serial))
    return rigs
