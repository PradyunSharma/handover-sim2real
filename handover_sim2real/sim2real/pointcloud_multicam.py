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
  * "Object" here means "non-hand within `object_max_radius_m` of the hand
    centroid". A distant camera sees more of the scene through that sphere than
    a wrist camera does, so its object cloud is dirtier. The radius is therefore
    a per-camera parameter, not a global one.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

_SIM2REAL_DIR = Path(__file__).resolve().parent
if str(_SIM2REAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM2REAL_DIR))

from pointcloud_pipeline import (  # noqa: E402
    _sample_or_pad_points,
    extract_hand_object_clouds,
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


# ── per-camera extraction parameters ─────────────────────────────────────────

@dataclass(frozen=True)
class ExtractionParams:
    """Arguments to `extract_hand_object_clouds` for one camera.

    Split per camera because a wrist view at 0.3 m and a tripod view at 1.0 m
    are not the same measurement problem: the D435's depth noise grows roughly
    with the square of range, so the far camera needs a looser crop to keep the
    object at all, and a stride that does not drown the near camera in points.
    """
    crop_radius_m: float = 0.12
    object_max_radius_m: float = 0.10
    min_depth_m: float = 0.10
    max_depth_m: float = 1.50
    full_cloud_stride: int = 2
    hand_cloud_stride: int = 1
    min_hand_points: int = 100
    min_object_points: int = 80


WRIST_PARAMS = ExtractionParams()

# Tripod, ~1 m out. Wider crop and object radius because at that range a 1 cm
# depth error is normal and a 0.10 m sphere around a noisy hand centroid starts
# clipping the object itself. Point counts scale with solid angle, so the same
# object subtends far fewer pixels here — the minimum-points floors drop
# accordingly, otherwise every frame would fall through to the stale-cloud
# fallback and the fusion would quietly become wrist-only again.
FIXED_PARAMS = ExtractionParams(
    crop_radius_m=0.16,
    object_max_radius_m=0.13,
    min_depth_m=0.25,
    max_depth_m=2.50,
    full_cloud_stride=2,
    hand_cloud_stride=1,
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


class HandSegmenter:
    """The hand-segmentation network, run over every camera in ONE forward pass.

    Batching is not a micro-optimisation here. The runner's step is
    observe-then-act, and with two cameras a per-image forward doubles the
    latency between the depth frames and the action computed from them — which
    is a real position error while the arm is moving, not just slowness. One
    batched call keeps it at roughly the single-camera cost.
    """

    def __init__(self, model, preprocess, device: str):
        self._model = model
        self._preprocess = preprocess
        self._device = device

    def __call__(self, images_rgb: Sequence[np.ndarray]) -> list[np.ndarray]:
        import cv2
        import torch

        if not images_rgb:
            return []
        batch = torch.stack([self._preprocess(img) for img in images_rgb])
        batch = batch.to(self._device, non_blocking=True)
        with torch.inference_mode():
            pred = self._model(batch).argmax(1).cpu().numpy().astype(np.uint8)

        out = []
        for img, mask in zip(images_rgb, pred):
            m = cv2.resize(mask, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            out.append(normalize_mask(m))
        return out


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
        return "  ".join(
            f"{name}:o{d['object']}/h{d['hand']}"
            + ("*" if d["used_last_hand"] or d["used_last_object"] else "")
            for name, d in self.per_camera.items()
        )


class MultiCameraPerception:
    """Owns the rigs and the segmenter; turns one robot pose into one cloud.

    Usage mirrors the single-camera path it replaces:

        perception = MultiCameraPerception(rigs, segmenter)
        obs = perception.observe(T_base_hand)
        if obs.usable:
            pc = build_policy_cloud(obs.object_xyz, obs.hand_xyz)
    """

    def __init__(self, rigs: Sequence[CameraRig], segmenter: Callable,
                 per_camera_cap: Optional[int] = None):
        if not rigs:
            raise ValueError("need at least one camera rig")
        self._rigs = list(rigs)
        self._segment = segmenter
        self._per_camera_cap = per_camera_cap
        self.last_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}

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

        masks = self._segment([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in colors])

        object_parts, hand_parts = [], []
        object_src, hand_src = [], []
        per_camera: dict[str, dict[str, Any]] = {}

        for rig_i, (rig, color_bgr, depth_m, hand_mask) in enumerate(
                zip(self._rigs, colors, depths, masks)):
            self.last_frames[rig.name] = (color_bgr, hand_mask)

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
            }

        # 3. Raw per-class union across cameras — see the module docstring on why
        #    this is deliberately unweighted.
        object_xyz = (np.concatenate(object_parts, axis=0) if object_parts
                      else np.zeros((0, 3), dtype=np.float32))
        hand_xyz = (np.concatenate(hand_parts, axis=0) if hand_parts
                    else np.zeros((0, 3), dtype=np.float32))

        return FusedObservation(
            object_xyz=object_xyz, hand_xyz=hand_xyz, per_camera=per_camera,
            object_source=(np.concatenate(object_src) if object_src
                           else np.zeros(0, np.int8)),
            hand_source=(np.concatenate(hand_src) if hand_src
                         else np.zeros(0, np.int8)),
            camera_names=tuple(r.name for r in self._rigs))


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

    rigs: list[CameraRig] = []
    for name in names:
        serial = serials.get(name)
        camera = RealSenseCamera(color_size=color_size, depth_size=depth_size,
                                 fps=fps, serial=serial)
        if name == "wrist":
            rigs.append(CameraRig(
                name=name, kind="eye_in_hand", camera=camera,
                params=WRIST_PARAMS, T_hand_cam=T_hand_cam_wrist,
                exclude_robot=False, serial=serial))
        else:
            if fixed_session is None:
                raise ValueError(
                    f"camera {name!r} is a fixed camera and needs a calibration "
                    "session (--calib-session)")
            rigs.append(CameraRig(
                name=name, kind="fixed", camera=camera,
                params=FIXED_PARAMS,
                T_base_cam=load_fixed_extrinsics(fixed_session),
                exclude_robot=exclude_robot, serial=serial))
    return rigs
