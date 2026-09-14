"""Promptable segmentation: Grounding DINO seeds a box, SAM2 tracks it.

WHY THIS EXISTS, since the shipped `HandSegmenter` is not broken. It segments
HANDS, and the policy's cloud is 896 object rows against 128 hand rows. Under
that backend the object class is never segmented at all — it is derived
negatively by `pointcloud_pipeline.extract_hand_object_clouds`: crop a sphere
around the hand centroid, subtract the dilated hand mask, keep what is
3D-connected to the hand, then drop blobs that look like forearm. README_SIM2REAL
calls that "the largest remaining sim2real gap in perception, larger than the
calibration", and names the case it provably cannot fix: a hand within ~2 cm of
the table merges with it, and the tabletop comes through as object.

Measured on the staged scene in `test_multicam_fusion._positive_object_mask` —
a hand holding an object resting on a table at 0.60 m — the derived class
returns 1265 points of which 905, **72%**, are table. Given an object mask it
returns the object's 400 and nothing else.

So this backend's job is to answer a question the other one cannot: which pixels
are the OBJECT. The hand mask it also produces is a bonus, and if it is merely as
good as cp1's that is fine.

THE SHAPE OF THE COMPUTATION IS FORCED BY THE BUDGET. `--control rate` gives the
whole loop 150 ms per step. Grounding DINO is a ~172M-parameter open-vocabulary
detector and cannot run per frame at that rate. SAM2 is a tracker with a memory
bank and can. So the detector runs ONCE, at the episode boundary, and SAM2
carries the masks from there — which is also why `MultiCameraPerception.reset()`
forwards to `reset()` here.

WHAT THIS BACKEND GETS WRONG THAT THE OTHER ONE CANNOT. cp1 fails LOUDLY: it
returns an empty mask, the point counts go to zero, `FusedObservation.usable` is
False and the runner holds the arm. A tracker fails SILENTLY — it latches onto
the forearm or the table edge and reports a confident mask of the wrong thing,
and the policy acts on it. That inversion is the real risk in swapping backends,
not accuracy, and `_Watchdog` below is the part that addresses it. Treat it as
load-bearing rather than as instrumentation.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from pointcloud_multicam import SegmentationResult, normalize_mask
from transforms import invert_transform

# Object ids inside a SAM2 session. Two objects share one image-encoder pass —
# the encoder is the expensive part and runs per FRAME, not per object, so
# tracking the hand as well as the object is close to free.
OBJ_HAND = 1
OBJ_OBJECT = 2

# How many past frames of per-object memory to retain. The memory attention
# reaches back `num_maskmem - 1` = 6 frames at `memory_temporal_stride_for_eval`
# = 1, and the object-pointer cross-attention reaches back
# `max_obj_ptrs_in_encoder` = 16 (sam2/modeling/sam2_base.py). 32 is double the
# deeper of the two.
#
# THIS IS NOT AN OPTIMISATION, it is what makes an unbounded stream possible at
# all. Upstream's `propagate_in_video` keeps every frame's memory because a video
# is finite; here the loop runs until the operator stops it, and each retained
# frame costs a maskmem tensor per object. Left unpruned, a ten-minute session
# is hundreds of megabytes of VRAM holding frames nothing can attend to.
MEMORY_RETENTION_FRAMES = 32

# SAM2's own preprocessing, from sam2/utils/misc.py: resize to a square
# `image_size`, scale to [0,1], then ImageNet normalize. Not tunable — they are
# the statistics the backbone was trained under.
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

DEFAULT_SAM2_MODEL = "sam2.1_hiera_tiny"
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# Config name and checkpoint filename per model key. The config path is relative
# to the sam2 package's own hydra search path, not to this repo.
SAM2_MODELS = {
    "sam2.1_hiera_tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml",
                          "sam2.1_hiera_tiny.pt"),
    "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s.yaml",
                           "sam2.1_hiera_small.pt"),
    "sam2.1_hiera_base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml",
                               "sam2.1_hiera_base_plus.pt"),
    "sam2.1_hiera_large": ("configs/sam2.1/sam2.1_hiera_l.yaml",
                           "sam2.1_hiera_large.pt"),
}

_INSTALL_HINT = (
    "The 'sam2' segmentation backend needs two packages this env does not have "
    "by default:\n"
    "    pip install 'hydra-core>=1.3.2' 'iopath>=0.1.10' 'transformers>=4.40'\n"
    "    pip install --no-deps 'git+https://github.com/facebookresearch/sam2.git'\n"
    "and the SAM 2.1 weights (mkdir first — curl -o will not create the dir):\n"
    "    mkdir -p checkpoint/sam2\n"
    "    curl -L -o checkpoint/sam2/sam2.1_hiera_tiny.pt \\\n"
    "      https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_tiny.pt\n"
    "--no-deps on sam2 is deliberate: its requirement pins will happily move "
    "numpy and torch, and the working robot loop depends on both."
)


# ── watchdog ─────────────────────────────────────────────────────────────────

@dataclass
class WatchdogLimits:
    """When a track has stopped describing the thing it was seeded on.

    Every one of these is a statement about the physical scene rather than about
    the network's confidence, which is the point: a tracker's own score is the
    thing that is wrong when a tracker is wrong. Only `min_score` consults it,
    and it is deliberately the weakest of the four.
    """
    # Below this the mask is not a measurement of anything. Deliberately generous
    # — the per-camera min_hand_points / min_object_points floors downstream are
    # the real test, and re-seeding on a brief occlusion is worse than one thin
    # frame that the stale-cloud fallback already covers.
    min_pixels: int = 60
    # A mask that trebles or thirds between two frames 150 ms apart did not
    # observe an object changing size; it changed WHICH object it is on.
    max_area_ratio: float = 3.0
    # 1.5 m/s is a brisk human reach; over a 150 ms step that is 0.22 m. A
    # centroid that moved further than this did not follow anything.
    max_centroid_jump_m: float = 0.22
    # SAM2's occlusion head. Low is a genuine "I have lost it", but it also dips
    # on legitimate partial occlusion, so it is a trigger of last resort.
    min_score: float = -2.0
    # Re-seeding costs a Grounding DINO forward. Without a floor on the interval,
    # a scene the detector cannot resolve turns every frame into a detection and
    # the control loop stops meeting its rate.
    min_reseed_interval_s: float = 1.0
    # HOW FAR A RE-SEED MAY MOVE THE OBJECT, and it is the same 0.22 m on
    # purpose — a re-seed runs on the SAME FRAME as the track it replaces, so
    # the real object has not moved at all and the whole allowance is detector
    # box noise plus the one stale frame that triggered the fault. Anything
    # beyond it is not a recovered track, it is a different object.
    #
    # THE HOLE THIS CLOSES WAS MEASURED ON HARDWARE. `note_reseed` clears the
    # centroid history, so the jump test above could not fire on the frame
    # after a re-seed: a re-seed was a free pass to relocate the object
    # anywhere in the image. Observed on a `+x` attempt at step 26 — the object
    # mask moved 140 px with ZERO overlap with the mask it replaced (all 5336
    # pixels newly object, none of them previously hand), the object cloud's
    # median leapt from 14.7 cm to 21.4 cm, and the wrong region was then
    # tracked happily for the remaining six steps while the policy, told the
    # object had retreated, hovered at 19 cm and never closed.
    max_reseed_jump_m: float = 0.22


# Prefix on the one fault that means "the REPLACEMENT is wrong", as opposed to
# "the track has drifted". The two need different answers — see `segment` — and
# the alternative to a marker is a second return value threaded through two call
# sites for one bit.
RESEED_FAULT = "re-seed landed elsewhere: "

# How much further than the rig's own object crop radius a freshly detected
# object box may sit from the hand. A 2D box's 3D median is a cruder estimate
# than a mask's, so a little slack is right; 2x the crop radius is not, because
# past the crop the deprojection stage deletes the points anyway.
SEED_DIST_SLACK = 1.4


class _Watchdog:
    """Per-camera track health. One instance per SAM2 session."""

    def __init__(self, limits: WatchdogLimits):
        self._limits = limits
        self.reset()

    def reset(self) -> None:
        self._area: dict[int, int] = {}
        self._centroid: dict[int, np.ndarray] = {}
        # The centroid as it was JUST BEFORE the current re-seed, kept so the
        # replacement can be held to it. Consumed on the first check after the
        # re-seed and not carried further: two re-seeds later it would be
        # anchoring the track to a scene that has genuinely moved on.
        self._pre_reseed: dict[int, np.ndarray] = {}
        self._last_reseed = 0.0
        self.reseeds = 0
        self.last_reason: Optional[str] = None

    def may_reseed(self, now: float) -> bool:
        return (now - self._last_reseed) >= self._limits.min_reseed_interval_s

    def note_reseed(self, now: float, reason: str) -> None:
        self._last_reseed = now
        self.reseeds += 1
        self.last_reason = reason
        # AREA is cleared and the CENTROID is carried. A re-seeded mask may
        # legitimately be a very different size — a fresh detection of a
        # partly occluded object often is — but it may not be somewhere else,
        # because it is a detection of the same frame.
        self._pre_reseed = dict(self._centroid)
        self._area.clear()
        self._centroid.clear()

    def check(self, obj_id: int, mask: np.ndarray, score: float,
              centroid_m: Optional[np.ndarray]) -> Optional[str]:
        """None if the track still looks like itself, else why not."""
        lim = self._limits
        area = int(mask.sum())
        name = "hand" if obj_id == OBJ_HAND else "object"

        if area < lim.min_pixels:
            self._area.pop(obj_id, None)
            self._centroid.pop(obj_id, None)
            return f"{name} mask collapsed to {area} px"

        prev_area = self._area.get(obj_id)
        self._area[obj_id] = area
        if prev_area is not None and prev_area >= lim.min_pixels:
            ratio = max(area / prev_area, prev_area / area)
            if ratio > lim.max_area_ratio:
                return (f"{name} mask area changed {ratio:.1f}x "
                        f"({prev_area} -> {area} px)")

        if centroid_m is not None and np.all(np.isfinite(centroid_m)):
            prev = self._centroid.get(obj_id)
            self._centroid[obj_id] = centroid_m
            # Against the previous frame normally; against the pre-re-seed
            # centroid on the first check after a re-seed, which is the case
            # the cleared history used to make unverifiable.
            base, limit, when, tag = (prev, lim.max_centroid_jump_m,
                                      "in one step", "")
            if prev is None and self._pre_reseed.get(obj_id) is not None:
                base = self._pre_reseed.pop(obj_id)
                limit, when, tag = (lim.max_reseed_jump_m,
                                    "from the track it replaced", RESEED_FAULT)
            if base is not None:
                jump = float(np.linalg.norm(centroid_m - base))
                if jump > limit:
                    return (f"{tag}{name} centroid jumped "
                            f"{jump*100:.0f} cm {when}")

        if score < lim.min_score:
            return f"{name} track score {score:.2f}"
        return None


# ── the streaming predictor ──────────────────────────────────────────────────

_STREAM_CLS: Optional[type] = None


def stream_predictor_class() -> type:
    """`SAM2VideoPredictor` with frames appended one at a time.

    THE OFFICIAL VIDEO API CANNOT TAKE A LIVE CAMERA, which is worth stating
    plainly because it looks as though it should. `init_state(video_path, ...)`
    calls `load_video_frames()` and then fixes `num_frames = len(images)`, and
    the class exposes no way to append — the public surface is `init_state`,
    `add_new_points_or_box`, `add_new_mask`, `propagate_in_video`,
    `reset_state`. Every frame must exist before tracking starts.

    Nothing about the MODEL requires that; it is an assumption in the state
    container. So this subclass builds the same state with a growing frame store
    and runs the body of `propagate_in_video`'s loop on one frame at a time. The
    three things it adds are the frame store, memory pruning, and not needing to
    know the length in advance. Everything else is upstream's.

    Defined inside a function so this module imports on a machine with no sam2
    installed — the default backend must not pay for a dependency it never uses.
    """
    global _STREAM_CLS
    if _STREAM_CLS is not None:
        return _STREAM_CLS

    import torch
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as err:                       # pragma: no cover
        raise ImportError(f"{err}\n\n{_INSTALL_HINT}") from err

    class SAM2StreamPredictor(SAM2VideoPredictor):   # type: ignore[misc,valid-type]

        @torch.inference_mode()
        def init_stream(self, height: int, width: int,
                        offload_state_to_cpu: bool = False) -> dict:
            """`init_state` for a video whose length is not yet known.

            Mirrors upstream key for key so every inherited method finds what it
            expects. Two deliberate differences: `images` is a dict keyed by
            frame index rather than a tensor, because a stream has no array to
            slice and old frames must be droppable; and there is no frame-0
            backbone warm-up, because frame 0 does not exist yet.
            """
            state: dict[str, Any] = {}
            state["images"] = {}
            state["num_frames"] = 0
            state["offload_video_to_cpu"] = False
            state["offload_state_to_cpu"] = offload_state_to_cpu
            state["video_height"] = int(height)
            state["video_width"] = int(width)
            state["device"] = self.device
            state["storage_device"] = (torch.device("cpu") if offload_state_to_cpu
                                       else self.device)
            state["point_inputs_per_obj"] = {}
            state["mask_inputs_per_obj"] = {}
            state["cached_features"] = {}
            state["constants"] = {}
            state["obj_id_to_idx"] = OrderedDict()
            state["obj_idx_to_id"] = OrderedDict()
            state["obj_ids"] = []
            state["output_dict_per_obj"] = {}
            state["temp_output_dict_per_obj"] = {}
            state["frames_tracked_per_obj"] = {}
            return state

        @torch.inference_mode()
        def append_frame(self, state: dict, img_chw: "torch.Tensor") -> int:
            """Add one preprocessed frame; returns its index.

            Only the newest frame is retained. `_get_image_feature` is the sole
            reader of `images` and is only ever asked for the frame being
            tracked; the memory bank stores ENCODED features, not pixels, so
            dropping the raw frames costs nothing and keeps a session's pixel
            footprint constant instead of linear in episode length.
            """
            idx = int(state["num_frames"])
            state["images"][idx] = img_chw
            state["num_frames"] = idx + 1
            for old in [k for k in state["images"] if k < idx]:
                del state["images"][old]
            return idx

        @torch.inference_mode()
        def track_frame(self, state: dict, frame_idx: int):
            """One iteration of `propagate_in_video`'s loop, for one frame.

            Returns (obj_ids, video_res_masks, object_score_logits). The masks
            are logits at the original camera resolution, so > 0 is the mask.
            """
            batch_size = self._get_obj_num(state)
            if batch_size == 0:
                raise RuntimeError("no objects have been prompted in this session")

            pred_masks_per_obj: list[Any] = [None] * batch_size
            scores: list[float] = [0.0] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = state["output_dict_per_obj"][obj_idx]
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    # The seed frame: its output already exists, and re-running
                    # inference on it would discard the prompt.
                    current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                    pred_masks = current_out["pred_masks"].to(
                        state["device"], non_blocking=True)
                else:
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=False,
                        run_mem_encoder=True,
                    )
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
                state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": False}
                pred_masks_per_obj[obj_idx] = pred_masks
                logits = current_out.get("object_score_logits")
                scores[obj_idx] = (float(logits.flatten()[0])
                                   if logits is not None else 0.0)

            all_pred_masks = (torch.cat(pred_masks_per_obj, dim=0)
                              if batch_size > 1 else pred_masks_per_obj[0])
            _, video_res_masks = self._get_orig_video_res_output(
                state, all_pred_masks)
            self._prune_memory(state, frame_idx)
            return list(state["obj_ids"]), video_res_masks, scores

        def _prune_memory(self, state: dict, frame_idx: int) -> None:
            """Drop memory older than anything can attend to.

            `cond_frame_outputs` is never pruned — that is the seed, and the
            memory attention treats it as always available. Only the rolling
            non-conditioning memory is bounded.
            """
            cutoff = frame_idx - MEMORY_RETENTION_FRAMES
            if cutoff <= 0:
                return
            for obj_idx in state["output_dict_per_obj"]:
                non_cond = state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
                for t in [t for t in non_cond if t < cutoff]:
                    del non_cond[t]
                tracked = state["frames_tracked_per_obj"].get(obj_idx, {})
                for t in [t for t in tracked if t < cutoff]:
                    del tracked[t]

    _STREAM_CLS = SAM2StreamPredictor
    return _STREAM_CLS


# ── the detector ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    box: np.ndarray          # [x0, y0, x1, y1] in native image pixels
    score: float
    label: str


class GroundingDinoDetector:
    """Open-vocabulary boxes from a text phrase, run once per seed.

    Through HuggingFace `transformers` rather than the original
    `IDEA-Research/GroundingDINO` package, which builds a
    `MultiScaleDeformableAttention` CUDA extension at install time. This env's
    numpy already resolves from three places (the env, `~/.local`, and ROS on
    the path); adding a source CUDA build to that is how a working robot loop
    stops importing.

    Loaded in fp16. It is ~172M parameters against an 8 GB laptop GPU that is
    also holding SAM2 and the PointNet++ policy, and it only ever produces boxes
    that are about to be quantised into a prompt — half precision is not a
    meaningful loss of anything downstream.
    """

    def __init__(self, model_id: str = DEFAULT_DINO_MODEL, device: str = "cuda",
                 box_threshold: float = 0.25, text_threshold: float = 0.25):
        try:
            import torch
            import transformers
            from transformers import (AutoProcessor,
                                      AutoModelForZeroShotObjectDetection)
        except ImportError as err:                   # pragma: no cover
            raise ImportError(f"{err}\n\n{_INSTALL_HINT}") from err

        self._torch = torch
        self._device = device
        self._box_threshold = float(box_threshold)
        self._text_threshold = float(text_threshold)
        self._processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        # `torch_dtype` became `dtype` in transformers 5. The old name is still
        # accepted there ("kept for BC") but is on its way out, and the new one
        # does not exist on 4.x — so pick by major version rather than passing
        # one and hoping. Chosen explicitly because a silently ignored dtype
        # kwarg loads 690 MB in fp32 and only shows up as VRAM pressure.
        major = int(str(transformers.__version__).split(".")[0])
        dtype_kw = {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, **dtype_kw).to(device).eval()

    def detect(self, image_rgb: np.ndarray, prompt: str) -> list[Detection]:
        """Boxes for one phrase, best first.

        Grounding DINO wants lowercase phrases terminated by a period; anything
        else silently detects less well rather than erroring, so the text is
        normalised here instead of trusting the caller.
        """
        torch = self._torch
        text = prompt.strip().lower()
        if not text.endswith("."):
            text += "."

        inputs = self._processor(images=image_rgb, text=text,
                                 return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)

        # The box-score keyword was RENAMED. transformers 4.40 takes
        # `box_threshold`; it became `threshold` later, and passing the wrong one
        # is a TypeError rather than a warning. Both are tried because the pin is
        # a floor, not a version, and this is the whole difference between the
        # backend working and it not importing.
        post = self._processor.post_process_grounded_object_detection
        kwargs = dict(text_threshold=self._text_threshold,
                      target_sizes=[image_rgb.shape[:2]])
        try:
            results = post(outputs, inputs["input_ids"],
                           threshold=self._box_threshold, **kwargs)[0]
        except TypeError:
            results = post(outputs, inputs["input_ids"],
                           box_threshold=self._box_threshold, **kwargs)[0]

        # Likewise `labels` -> `text_labels`. Newer versions return both, with
        # `labels` deprecated, so the new name is preferred where present.
        labels = results.get("text_labels", results.get("labels", []))
        out = [
            Detection(box=np.asarray(b, dtype=np.float32),
                      score=float(s),
                      label=str(t))
            for b, s, t in zip(results["boxes"].detach().cpu().numpy(),
                               results["scores"].detach().cpu().numpy(),
                               labels)
        ]
        out.sort(key=lambda d: d.score, reverse=True)
        return out


# ── the backend ──────────────────────────────────────────────────────────────

@dataclass
class Sam2Config:
    sam2_model: str = DEFAULT_SAM2_MODEL
    checkpoint_dir: Optional[Path] = None
    dino_model: str = DEFAULT_DINO_MODEL
    hand_prompt: str = "a hand."
    object_prompt: str = "an object."
    device: str = "cuda"
    limits: WatchdogLimits = field(default_factory=WatchdogLimits)
    # bfloat16 is what SAM2's own demos run under and roughly halves the encoder
    # cost. Kept switchable because "the masks got worse when we went fast" is a
    # hypothesis that has to be testable.
    autocast: bool = True


class _Session:
    """One SAM2 track per camera. Cameras are independent videos.

    The watchdog OUTLIVES the tracking state, deliberately. Its rate limiter and
    its re-seed count are properties of this camera's whole run, and rebuilding
    it alongside the state would reset the one thing that stops a scene the
    detector cannot resolve from turning every frame into a detection.
    """

    def __init__(self, name: str, limits: WatchdogLimits):
        self.name = name
        self.state: Optional[dict] = None
        self.seeded = False
        self.watchdog = _Watchdog(limits)


class Sam2Segmenter:
    """`HandSegmenter`'s contract, with an object mask as well.

    Same call signature, so `MultiCameraPerception` does not know which backend
    it holds. The difference is entirely in the returned `SegmentationResult`:
    `object` is a mask rather than None, and downstream that switches
    `extract_hand_object_clouds` off the negative derivation.

    `sizes` is accepted and IGNORED. It is cp1's square input resolution, a
    knob that exists because that network's mask thins out at range; SAM2 has
    its own fixed `image_size` and no equivalent. Accepted rather than rejected
    so the two backends stay interchangeable at the call site.
    """

    def __init__(self, rigs: Sequence[Any], config: Sam2Config,
                 detector: Optional[GroundingDinoDetector] = None):
        import torch

        self._torch = torch
        self._cfg = config
        self._rigs = list(rigs)
        self._device = config.device
        # Why `_pick_object` refused, carried the one frame from the pick to
        # `_seed`'s return so the HUD says "35 cm from the hand" rather than
        # the generic "no object box distinguishable from the hand box".
        self._last_seed_reject: Optional[str] = None

        if config.sam2_model not in SAM2_MODELS:
            raise ValueError(
                f"unknown sam2 model {config.sam2_model!r}; have "
                f"{', '.join(sorted(SAM2_MODELS))}")
        cfg_name, ckpt_name = SAM2_MODELS[config.sam2_model]
        ckpt_dir = Path(config.checkpoint_dir
                        or Path(__file__).resolve().parent / "checkpoint" / "sam2")
        ckpt = ckpt_dir / ckpt_name
        if not ckpt.exists():
            raise FileNotFoundError(
                f"SAM2 weights not found at {ckpt}\n\n{_INSTALL_HINT}")

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as err:                   # pragma: no cover
            raise ImportError(f"{err}\n\n{_INSTALL_HINT}") from err

        # Built through upstream's builder so the hydra config is resolved the
        # way upstream resolves it, then re-classed onto the streaming subclass.
        # Subclassing at build time would mean duplicating the builder.
        predictor = build_sam2_video_predictor(
            cfg_name, str(ckpt), device=config.device)
        predictor.__class__ = stream_predictor_class()
        self._predictor = predictor

        self._detector = detector or GroundingDinoDetector(
            config.dino_model, device=config.device)

        self._sessions = {rig.name: _Session(rig.name, config.limits)
                          for rig in self._rigs}
        self._img_mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
        self._img_std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
        self._img_mean = self._img_mean.to(config.device)
        self._img_std = self._img_std.to(config.device)

        # Depth and the live robot pose for the frame being segmented. Set from
        # the call arguments and read by the seed (to choose between candidate
        # boxes) and the watchdog (to measure a jump in metres rather than
        # pixels). Held as state rather than threaded through six helpers.
        self._depth: Optional[np.ndarray] = None
        self._T_base_hand: Optional[np.ndarray] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop every track. The next frame re-detects.

        Called from `MultiCameraPerception.reset()`, i.e. at every episode
        boundary. A memory bank carried across episodes is not stale the way a
        point cloud is stale — it is confidently wrong, still tracking the
        object from the handover before.
        """
        for session in self._sessions.values():
            session.state = None
            session.seeded = False
            session.watchdog.reset()

    # ── the contract ─────────────────────────────────────────────────────────

    def __call__(self, images_rgb: Sequence[np.ndarray],
                 sizes: Optional[Sequence[int]] = None,
                 depths: Optional[Sequence[np.ndarray]] = None,
                 T_base_hand: Optional[np.ndarray] = None
                 ) -> list[SegmentationResult]:
        if len(images_rgb) != len(self._rigs):
            raise ValueError(
                f"got {len(images_rgb)} images for {len(self._rigs)} rigs — the "
                "segmenter is constructed against the rigs and their order")

        torch = self._torch
        self._T_base_hand = T_base_hand
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if (self._cfg.autocast and str(self._device).startswith("cuda"))
               else _nullcontext())
        out: list[SegmentationResult] = []
        with ctx:
            for i, (rig, img) in enumerate(zip(self._rigs, images_rgb)):
                self._depth = depths[i] if depths is not None else None
                out.append(self._segment_one(rig, img))
        return out

    # ── per camera ───────────────────────────────────────────────────────────

    def _segment_one(self, rig, image_rgb: np.ndarray) -> SegmentationResult:
        h, w = image_rgb.shape[:2]
        session = self._sessions[rig.name]
        now = time.time()
        debug: dict[str, Any] = {"reseeded": False, "reason": None}
        empty = np.zeros((h, w), np.uint8)

        # At most two passes: track, and — if the track has stopped describing
        # what it was seeded on — detect again and track once more on the SAME
        # frame. A recoverable fault then costs a detection rather than a
        # dropped observation, and an unrecoverable one cannot spin.
        hand = obj = empty
        for attempt in (0, 1):
            if session.state is None:
                session.state = self._predictor.init_stream(h, w)
                session.seeded = False

            frame_idx = self._predictor.append_frame(
                session.state, self._preprocess(image_rgb))

            if not session.seeded:
                reason = self._seed(rig, session, image_rgb, frame_idx)
                if reason is not None:
                    # Nothing to track. Empty masks are the honest answer, and
                    # the runner already knows what to do with them: an unusable
                    # observation holds the arm. Inventing a mask here would be
                    # the silent failure this backend has to avoid.
                    debug["reason"] = reason
                    debug["reseeds"] = session.watchdog.reseeds
                    return SegmentationResult(hand=empty, object=empty,
                                              debug=debug)

            obj_ids, mask_logits, scores = self._predictor.track_frame(
                session.state, frame_idx)

            masks: dict[int, np.ndarray] = {}
            score_by_id: dict[int, float] = {}
            for slot, obj_id in enumerate(obj_ids):
                masks[int(obj_id)] = (
                    mask_logits[slot, 0] > 0.0).cpu().numpy().astype(np.uint8)
                score_by_id[int(obj_id)] = scores[slot]
            hand = masks.get(OBJ_HAND, empty)
            obj = masks.get(OBJ_OBJECT, empty)

            fault = None
            for obj_id, mask in ((OBJ_HAND, hand), (OBJ_OBJECT, obj)):
                fault = session.watchdog.check(
                    obj_id, mask, score_by_id.get(obj_id, 0.0),
                    self._centroid_m(rig, mask))
                if fault is not None:
                    break
            if fault is None:
                break

            debug["reason"] = fault
            if fault.startswith(RESEED_FAULT):
                # A REPLACEMENT THAT LANDED SOMEWHERE ELSE IS NOT A TRACK, and
                # it is the one fault the point floors cannot catch: the wrong
                # region is a perfectly healthy 1300-point cloud, well above
                # `min_object_points`, so nothing downstream objects and the
                # policy is steered by it for the rest of the episode. Empty
                # masks are the honest answer here, exactly as they are for a
                # failed seed above — the runner holds the arm, costs no step,
                # and the next frame is a fresh chance.
                debug["reseeds"] = session.watchdog.reseeds
                debug["refused_reseed"] = True
                return SegmentationResult(hand=empty, object=empty, debug=debug)
            if attempt == 1 or not session.watchdog.may_reseed(now):
                # Rate-limited out, or already retried. Return what was measured
                # with the fault recorded — the HUD shows it and the downstream
                # point floors will hold the arm if the masks really are junk.
                break
            session.watchdog.note_reseed(now, fault)
            session.state = None
            session.seeded = False
            debug["reseeded"] = True

        # `normalize_mask` for the hand only. It keeps the largest blob and
        # opens/closes, which is right for a hand and WRONG for an object: an
        # object can legitimately appear as two pieces with the fingers
        # occluding its middle, and keeping the largest would delete half of it
        # at exactly the moment the policy is closing on it.
        debug["hand_px"] = int(hand.sum())
        debug["object_px"] = int(obj.sum())
        debug["reseeds"] = session.watchdog.reseeds
        return SegmentationResult(hand=normalize_mask(hand), object=obj,
                                  debug=debug)

    # ── seeding ──────────────────────────────────────────────────────────────

    def _seed(self, rig, session: _Session, image_rgb: np.ndarray,
              frame_idx: int) -> Optional[str]:
        """Detect, disambiguate, prompt. Returns a reason string on failure."""
        hands = self._detector.detect(image_rgb, self._cfg.hand_prompt)
        if not hands:
            return f"no box for {self._cfg.hand_prompt!r}"
        objects = self._detector.detect(image_rgb, self._cfg.object_prompt)
        if not objects:
            return f"no box for {self._cfg.object_prompt!r}"

        hand_det = self._pick_hand(rig, hands)
        self._last_seed_reject = None
        object_det = self._pick_object(rig, objects, hand_det)
        if object_det is None:
            return (self._last_seed_reject
                    or "no object box distinguishable from the hand box")

        for obj_id, det in ((OBJ_HAND, hand_det), (OBJ_OBJECT, object_det)):
            self._predictor.add_new_points_or_box(
                inference_state=session.state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=det.box,
            )
        # Folds the prompt outputs into the tracking state. Upstream calls this
        # from `propagate_in_video`; there is no propagate here, so it is called
        # explicitly and exactly once per seed.
        self._predictor.propagate_in_video_preflight(session.state)
        session.seeded = True
        return None

    def _pick_hand(self, rig, dets: list[Detection]) -> Detection:
        """The hand nearest the robot base, not the most confident one.

        Same criterion and the same reasoning as
        `pointcloud_multicam.select_hand_component`: a hand offering an object
        is reaching toward the robot and a bystander's is not, and the BASE
        rather than the end effector because scoring against the EE closes a
        loop in which one bad frame pulls the arm toward the error and makes the
        error score better next frame. The base cannot be moved by a perception
        mistake.

        Falls back to the detector's own ranking whenever the geometry is
        unavailable, so this can only do better than confidence, never fail
        closed.
        """
        base_in_cam = self._base_in_camera(rig)
        if base_in_cam is None or len(dets) == 1:
            return dets[0]

        best, best_d = dets[0], np.inf
        for det in dets:
            xyz = self._box_centroid_cam(rig, det.box)
            if xyz is None:
                continue
            d = float(np.linalg.norm(xyz - base_in_cam))
            if d < best_d:
                best, best_d = det, d
        return best

    def _pick_object(self, rig, dets: list[Detection],
                     hand: Detection) -> Optional[Detection]:
        """The object box nearest the chosen hand in 3D, AND near enough.

        An open-vocabulary detector asked for "an object" will happily return
        the hand, the arm, and the table as well. Nearness to the hand that was
        already disambiguated is the constraint that actually applies: the
        object is BEING HELD.

        NEAREST IS NOT THE SAME AS NEAR, and this used to be a bare argmin with
        no bound, so when every candidate was wrong the least-wrong one won and
        was tracked for the rest of the episode with nothing downstream able to
        object. The visible symptom is the mask jumping on 's': `perception.
        reset()` drops the SAM2 session at every episode boundary — it must,
        because a memory bank carried across episodes is confidently wrong
        rather than merely stale — so every 's' is a fresh detection, and a
        fresh detection is exactly where an unbounded argmin does its damage.

        The bound is the rig's OWN `object_max_radius_m`, which is already the
        answer to "how far from the hand may a point be and still be the
        object" — 0.26 m for a fixed camera. Beyond it, the deprojection stage
        would delete nearly every point of this box anyway, so accepting it
        buys a track that cannot survive its own crop. `SEED_DIST_SLACK`
        loosens it a little because a 2D box's 3D median is a cruder estimate
        than a mask's, and being slightly generous here costs a little wrong
        surface while being tight costs the whole seed.

        Returning None sends `_seed` down its existing failure path: empty
        masks, the arm holds, no step is consumed, and the next frame is a
        fresh chance with the reason on the HUD.
        """
        hand_xyz = self._box_centroid_cam(rig, hand.box)
        candidates = [d for d in dets if _iou(d.box, hand.box) < 0.85]
        if not candidates:
            return None
        if hand_xyz is None:
            # No depth to judge with. Falling back to the detector's ranking is
            # the documented behaviour of every geometry helper here — it can
            # only do better than confidence, never fail closed.
            return candidates[0]

        best, best_d = None, np.inf
        for det in candidates:
            xyz = self._box_centroid_cam(rig, det.box)
            if xyz is None:
                continue
            d = float(np.linalg.norm(xyz - hand_xyz))
            if d < best_d:
                best, best_d = det, d
        if best is None:
            return candidates[0]        # depth answered for none of them
        limit = SEED_DIST_SLACK * float(getattr(
            rig.params, "object_max_radius_m", 0.26))
        if best_d > limit:
            self._last_seed_reject = (
                f"nearest object box is {best_d*100:.0f} cm from the hand "
                f"(limit {limit*100:.0f}); nothing detected is being held")
            return None
        return best

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _base_in_camera(self, rig) -> Optional[np.ndarray]:
        """Robot base origin in this camera's frame, or None."""
        T_base_cam = getattr(rig, "T_base_cam", None)
        if T_base_cam is None:
            # Eye-in-hand: the extrinsic is to the hand, so the base needs the
            # live pose. Without it there is no base to measure from.
            if self._T_base_hand is None:
                return None
            T_base_cam = np.asarray(self._T_base_hand, dtype=np.float64) @ \
                np.asarray(rig.T_hand_cam, dtype=np.float64)
        return invert_transform(np.asarray(T_base_cam, dtype=np.float64))[:3, 3]

    def _centroid_m(self, rig, mask: np.ndarray) -> Optional[np.ndarray]:
        """Median 3D point under a mask, in the camera frame.

        Median rather than mean: a mask straddling a depth edge picks up points
        metres behind the subject, and a mean would follow them off it. Same
        reasoning, and the same stride, as `select_hand_component`.

        Returns None whenever depth cannot answer — no depth passed, an empty
        mask, or too few valid pixels. Every caller treats None as "cannot
        judge" and falls back rather than failing closed.
        """
        depth = self._depth
        if depth is None or not mask.any():
            return None
        xyz, _, _ = rig.camera.depth_to_pointcloud(
            depth_m=depth, mask=mask, stride=4,
            min_depth=rig.params.min_depth_m, max_depth=rig.params.max_depth_m)
        if len(xyz) < 3:
            return None
        return np.median(xyz, axis=0).astype(np.float64)

    def _box_centroid_cam(self, rig, box: np.ndarray) -> Optional[np.ndarray]:
        """`_centroid_m` for the rectangle of a detection box."""
        if self._depth is None:
            return None
        h, w = self._depth.shape[:2]
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = 1
        return self._centroid_m(rig, mask)

    # ── preprocessing ────────────────────────────────────────────────────────

    def _preprocess(self, image_rgb: np.ndarray):
        """To SAM2's fixed square input, normalized as its backbone expects.

        Straight from `sam2/utils/misc.py`. The resize is not preserving aspect
        ratio, deliberately — that is what the model was trained on, and the
        mask comes back through `_get_orig_video_res_output`, which reverses it.
        """
        import cv2

        torch = self._torch
        size = int(self._predictor.image_size)
        img = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img).permute(2, 0, 1).float().to(self._device) / 255.0
        t -= self._img_mean
        t /= self._img_std
        return t


# ── choosing a backend ───────────────────────────────────────────────────────
#
# The flags and the factory live here, next to the backend they were added for,
# because BOTH entry points need them: `my_policy_runner.py` and
# `test_perception_viz.py`. test_perception_viz already states the principle for
# --finger-boxes — "a debugging view running different perception settings from
# the thing being debugged is worse than no view at all" — and two independently
# written constructions of the same backend is precisely how that happens.

SEGMENTATION_MODES = ("hand-net", "sam2")


def add_segmentation_args(p) -> None:
    """The `--segmentation` family, identically in every entry point."""
    p.add_argument("--segmentation", choices=SEGMENTATION_MODES,
                   default="hand-net",
                   help="which segmenter feeds the point classes. 'hand-net' "
                        "(default) is cp1, a DeepLabV3 that segments HANDS; the "
                        "object class is then derived negatively — crop a "
                        "sphere around the hand, subtract the mask, keep what "
                        "is 3D-connected — which is what fails when the object "
                        "is touching the table. 'sam2' prompts Grounding DINO "
                        "once per episode and tracks both classes with SAM2, so "
                        "the object is MEASURED and the whole derivation chain "
                        "(crop, margin band, connectivity, forearm rejection) "
                        "is bypassed. sam2 needs extra packages and weights; it "
                        "will tell you the commands if they are missing.")
    p.add_argument("--seg-hand-prompt", type=str, default="a hand.",
                   help="text prompt for the hand box (--segmentation sam2).")
    p.add_argument("--seg-object-prompt", type=str, default="an object.",
                   help="text prompt for the object box (--segmentation sam2). "
                        "NAME THE THING if you can — 'a blue mug.' is far more "
                        "reliable than 'an object.', because an open-vocabulary "
                        "detector grounds nouns and 'object' is barely one. "
                        "Lowercase, ending in a period.")
    p.add_argument("--sam2-model", choices=tuple(sorted(SAM2_MODELS)),
                   default=DEFAULT_SAM2_MODEL,
                   help=f"SAM2 variant (default {DEFAULT_SAM2_MODEL}). Bigger "
                        "is better and slower; the encoder runs once per camera "
                        "per frame, against a 150 ms step in --control rate.")
    p.add_argument("--no-sam2-autocast", action="store_true",
                   help="run SAM2 in fp32. Roughly doubles the encoder cost; "
                        "here so 'the masks got worse when we went fast' is a "
                        "testable claim rather than a suspicion.")


def build_segmenter(args, rigs, device: str, hand_seg_model=None):
    """The segmenter `--segmentation` asked for.

    `hand_seg_model` is only consulted for 'hand-net'. In sam2 mode cp1 should
    not be loaded at all — it is 40M parameters and 9-26 ms per pass on a GPU
    that is about to hold SAM2, Grounding DINO and the policy.
    """
    from pointcloud_multicam import HandSegmenter

    mode = getattr(args, "segmentation", "hand-net")
    if mode == "hand-net":
        if hand_seg_model is None:
            raise ValueError("--segmentation hand-net needs the cp1 model")
        return HandSegmenter(hand_seg_model, device)
    if mode == "sam2":
        return Sam2Segmenter(rigs, Sam2Config(
            sam2_model=args.sam2_model,
            hand_prompt=args.seg_hand_prompt,
            object_prompt=args.seg_object_prompt,
            device=device,
            autocast=not args.no_sam2_autocast,
        ))
    raise ValueError(f"unknown segmentation mode {mode!r}")


def describe_segmenter(args) -> str:
    """One startup line. A run whose log does not say which segmentation was on
    is a run that cannot be compared to another."""
    mode = getattr(args, "segmentation", "hand-net")
    if mode != "sam2":
        return ("[perception] segmentation: hand-net (cp1) — object class "
                "derived from geometry")
    return (f"[perception] segmentation: sam2 ({args.sam2_model}"
            f"{'' if not args.no_sam2_autocast else ', fp32'}) — hand "
            f"{args.seg_hand_prompt!r}, object {args.seg_object_prompt!r}; "
            "object class MEASURED, crop/connectivity/arm-rejection bypassed")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ── offline check ────────────────────────────────────────────────────────────

def selftest() -> None:
    """The track-health and seed-pick policy, with no torch and no camera.

    Only the pure-numpy decision logic is exercised, which is where every bug
    this file has had actually lived. The predictor and the detector need
    weights and a GPU and are not testable here.
    """
    import types

    lim = WatchdogLimits()
    mask = np.zeros((48, 64), np.uint8)
    mask[10:30, 10:30] = 1                      # 400 px, well over min_pixels
    at = lambda x: np.array([float(x), 0.0, 1.0])

    # ---- tracking, unchanged ------------------------------------------------
    w = _Watchdog(lim)
    assert w.check(OBJ_OBJECT, mask, 0.0, at(0.00)) is None
    assert w.check(OBJ_OBJECT, mask, 0.0, at(0.05)) is None
    f = w.check(OBJ_OBJECT, mask, 0.0, at(0.60))
    assert f and "in one step" in f, f
    assert not f.startswith(RESEED_FAULT), (
        "an ordinary drift was tagged as a re-seed fault, which would throw "
        "the frame away instead of letting the point floors judge it")

    # ---- THE CASE THAT COST AN EPISODE -------------------------------------
    # After a re-seed the history used to be cleared, so a replacement mask on
    # the far side of the image passed every test.
    w = _Watchdog(lim)
    assert w.check(OBJ_OBJECT, mask, 0.0, at(0.00)) is None
    w.note_reseed(time.time(), "object mask collapsed to 12 px")
    f = w.check(OBJ_OBJECT, mask, 0.0, at(0.31))
    assert f is not None, (
        "a re-seed 31 cm from the track it replaced was accepted — it runs on "
        "the same frame, so the object cannot have moved at all")
    assert f.startswith(RESEED_FAULT) and "from the track it replaced" in f, f

    # ... and a re-seed that lands where the object was is what re-seeding is
    # FOR, so it must pass — otherwise the guard breaks recovery.
    w = _Watchdog(lim)
    assert w.check(OBJ_OBJECT, mask, 0.0, at(0.00)) is None
    w.note_reseed(time.time(), "object track score -3.10")
    assert w.check(OBJ_OBJECT, mask, 0.0, at(0.03)) is None, (
        "a re-seed 3 cm from the old centroid was refused")
    # The anchor is consumed, not kept: two frames later the scene has moved on.
    assert w.check(OBJ_OBJECT, mask, 0.0, at(0.18)) is None

    # Area is a fault while TRACKING and allowed across a re-seed, where a
    # fresh detection of a partly occluded object legitimately differs.
    w = _Watchdog(lim)
    small = np.zeros((48, 64), np.uint8)
    small[10:20, 10:20] = 1                     # 100 px
    assert w.check(OBJ_OBJECT, mask, 0.0, at(0.0)) is None
    f = w.check(OBJ_OBJECT, small, 0.0, at(0.0))
    assert f and "area changed" in f, f
    w = _Watchdog(lim)
    assert w.check(OBJ_OBJECT, mask, 0.0, at(0.0)) is None
    w.note_reseed(time.time(), "object mask area changed 4.0x")
    assert w.check(OBJ_OBJECT, small, 0.0, at(0.0)) is None, (
        "a re-seed was refused for changing size, which is the one thing a "
        "fresh detection is allowed to do")

    # ---- the seed pick, which every 's' goes through -----------------------
    # `perception.reset()` drops the SAM2 session at each episode boundary, so
    # pressing 's' always re-detects. `_pick_object` was a bare argmin, so when
    # every candidate was wrong the least-wrong one was seeded and tracked.
    class _Cam:
        def depth_to_pointcloud(self, **kw):
            raise AssertionError("the stub must not reach depth")

    class _Rig:
        name = "tripod"
        params = types.SimpleNamespace(object_max_radius_m=0.26)
        camera = _Cam()

    seg = Sam2Segmenter.__new__(Sam2Segmenter)   # no torch, no weights
    seg._last_seed_reject = None
    boxes = {"hand": (10.0, 10.0, 40.0, 40.0),
             "near": (42.0, 10.0, 70.0, 40.0),
             "far": (300.0, 300.0, 340.0, 340.0)}
    where = {boxes["hand"]: at(0.00), boxes["near"]: at(0.12),
             boxes["far"]: at(0.55)}
    seg._box_centroid_cam = lambda rig, box: where[tuple(box)]
    D = lambda b: types.SimpleNamespace(box=boxes[b], score=0.9)
    rig, hand = _Rig(), D("hand")

    got = seg._pick_object(rig, [D("near"), D("far")], hand)
    assert got is not None and tuple(got.box) == boxes["near"], (
        "the held object 12 cm from the hand was not picked")

    seg._last_seed_reject = None
    got = seg._pick_object(rig, [D("far")], hand)
    assert got is None, (
        "a box 55 cm from the hand was seeded as the held object — nearest is "
        "not the same as near, and this is the argmin that had no bound")
    assert seg._last_seed_reject and "from the hand" in seg._last_seed_reject
    print(f"    seed refused with: {seg._last_seed_reject}")

    # A box just inside the bound must still be accepted, or the limit is
    # asserting nothing but its own tightness.
    edge = (200.0, 200.0, 240.0, 240.0)
    where[edge] = at(SEED_DIST_SLACK * 0.26 - 0.01)
    seg._last_seed_reject = None
    got = seg._pick_object(rig, [types.SimpleNamespace(box=edge, score=0.5)],
                           hand)
    assert got is not None, (
        f"a box just inside the {SEED_DIST_SLACK * 0.26 * 100:.0f} cm limit "
        "was refused")

    print("sam2: drift/collapse/area faults as before; a re-seed is held to "
          "the track it replaced; a seed is held to the hand")


if __name__ == "__main__":
    selftest()
