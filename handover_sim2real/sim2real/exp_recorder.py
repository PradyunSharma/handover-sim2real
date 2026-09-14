#!/usr/bin/env python3
"""Record a real-robot experiment session without slowing the control loop.

WHAT THIS EXISTS FOR. `--exp-mode` runs an ordered sequence of grasp attempts,
one per 's', each timed and judged. This module is the half that writes what
happened: the RGB the policy saw, the segmentation it derived, the depth every
cloud came from, every tensor that entered the network, every action that left
it, and the outcome of each attempt. The point is that a session directory can
be read six months later, by someone with numpy and nothing else, and answer
"what is this policy's success rate per direction, and what did the failures
look like".

THE BUDGET IS THE DESIGN. One iteration of the loop is ~50 ms of real work
inside a 150 ms period under `--control rate`, and 33-47 ms of that is already
`observe()`. So the control thread's entire contribution here is a handful of
`.copy()` calls and a `put_nowait` — measured at ~0.14 ms, 0.09% of the period —
and everything expensive (PNG encode, video encode, every write) happens on one
daemon writer thread. The measurements behind that split, on this machine:

    color.copy()   640x480x3 u8   0.029 ms      cv2.imencode png lvl 3  1.47 ms
    depth.copy()   640x480   f32  0.041 ms      VideoWriter.write       ~2.5 ms
    mask.copy()    640x480   u8   0.005 ms      zlib.compress(depth,1)  7.29 ms
    queue put/get                 0.001 ms      cv2.imencode png depth 11.24 ms

The right-hand column is why none of it runs where the robot is waiting. The
worker's duty cycle is ~6%, and under `--control rate` the control thread is
asleep for ~100 ms of every period, so the worker runs in genuinely idle time.

WHY NOT HDF5. There is no h5py in `handover-rs` (it is in the sim envs, not this
one), and no lz4/blosc/zstandard/imageio/PyAV either. What there is: cv2 4.11
with a bundled ffmpeg that has mpeg4 but no x264 — so `mp4v` writes and `avc1`
does not — plus numpy and the stdlib. The formats below are chosen to need
exactly that and nothing more.

WHY FIXED-STRIDE RAW RECORDS. `numpy.lib.format.open_memmap` wants the length up
front, and a preallocated tail reads back as VALID ALL-ZERO DEPTH — a silent lie
of exactly the kind `regrasp/collector.py` grew its `complete` flag to prevent.
A `.npy` header patched at close is precisely the write that does not happen
under `os._exit(0)`. Raw records of a known stride recover by
`size // record_bytes`, a torn tail is arithmetically detectable, and every
earlier record is intact. Every binary handle here is `buffering=0`, so a byte
that reached `write()` has reached the kernel and survives `os._exit`, SIGKILL
and a segfault in Filament.

TWO INDICES, AND ONLY TWO. `step` counts control iterations and indexes every
light stream (cloud, state, action, pose, CSV row). `heavy_idx` is shared by all
three heavy sinks — it IS the mp4 frame ordinal, the depth record ordinal and
the label PNG name. `steps.csv` carries both; `heavy_idx = -1` means the frame
was shed under backpressure. When nothing was dropped, which is the normal case,
they are equal and nobody has to think about it.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from pointcloud_multicam import (NUM_HAND_POINTS, NUM_OBJECT_POINTS,
                                 PC_CHANNELS)

SCHEMA = "h2r.sim2real.exp_recorder/1"

# How many un-written steps may pile up before the heavy payload is shed. Eight
# steps of two cameras is ~34 MB of queue residency, and at the measured 6% duty
# cycle it is unreachable in steady state — a backlog means an I/O stall (a full
# disk, a --exp-out on NFS or a USB stick), which is exactly the thing that must
# be shouted about rather than absorbed.
HEAVY_BACKLOG = 8

# The label image's vocabulary. Object wins where both masks claim a pixel, and
# the overlap count goes in the CSV: "two masks claim the same pixels" is a real
# failure of the segmentation and it should survive into the data rather than
# only onto the screen.
LABEL_BG, LABEL_HAND, LABEL_OBJECT = 0, 1, 2
PNG_LEVEL = 3          # the knee: level 6 costs 1.9x the CPU for 25% less data

# How an attempt ended, and what the operator made of it. `close` is the only
# ending on which the fingers are actually holding something.
ENDINGS = ("close", "policy_fail", "timeout", "user_stop", "abandoned")
VERDICTS = ("pass", "fail", "void", "abandoned")

# Refuse to start a session that cannot finish. ~260 MB per attempt per two
# cameras with depth on.
MIN_FREE_BYTES = 5 * 1024 ** 3


# ── the pinned column lists ──────────────────────────────────────────────────
# Pinned, not derived from the first row, for the reason `train_dagger_phase4`'s
# LOG_FIELDS is pinned: two call sites with different natural key sets would
# mis-align every later row against a header taken from whichever landed first.

STEP_FIELDS_BASE = [
    # identity and indexing
    "attempt", "step", "heavy_idx", "t_mono", "t_wall", "dt_ms",
    # the commanded condition, repeated on every row on purpose: a concatenated
    # steps table must be plottable without a join back to attempts.csv
    "bin", "d_world_x", "d_world_y", "d_world_z",
    # robot pose. pose_base_hand.f8 is the authority (float64 4x4); these are
    # the same pose as xyz + wxyz, for reading and plotting
    "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz", "grip_norm",
    # the policy's INPUT, summarised (policy_pc.f4 / policy_rs.f4 are exact)
    # obj_range_m is the median object point's distance from the gripper
    # origin. Run 19's training clouds sit at 0.11 m; the gripper head saturates
    # outside that, so this is the column that says whether a refusal to close
    # was correct.
    "obs_usable", "n_obj", "n_hand", "obj_range_m", "hand_range_m",
    "arm_dropped", "arm_fallback",
    "grasp_points", "grasp_vetoed",
    # the policy's OUTPUT, exact: %.9g round-trips IEEE-754 binary32
    "act_dx", "act_dy", "act_dz", "act_rx", "act_ry", "act_rz", "act_grip",
    "act_pos_mm", "act_rot_deg", "clamped", "grasp_close",
    # The logit behind grasp_close. The bit says what it did; this says how
    # close it came, which is the difference between a marginal decision and a
    # refusal and is invisible on the robot.
    "grip_logit",
    # what actually reached the robot
    "armed", "executed", "target_x", "target_y", "target_z",
    # timing: the numbers the loop already prints and has never stored
    "ms_obs", "ms_pol", "ms_rec",
    # recorder health, so a drop is in the DATA and not only on the terminal
    "q_depth", "dropped_total",
]

_PER_CAM = ("obj", "hand", "robot_rm", "finger_rm", "stale", "reseeds",
            "obj_from_mask", "lab_hand_px", "lab_obj_px", "lab_overlap_px",
            "depth_valid_px", "png_bytes")

ATTEMPT_FIELDS = [
    "session_id", "attempt", "n_attempts", "try", "bin", "dir_name",
    "ending", "verdict", "froze",
    # THE JAWS, in mm of TOTAL opening — twice the per-finger travel that
    # `grip_norm` in steps.csv normalises, because the gap an object has to fit
    # through is what you compare against a caliper.
    #
    # `grip_min_mm` is the number worth having. GRASP_WIDTH_M is 0.0, so the
    # fingers are always commanded shut and whatever they settle at IS the
    # thickness of what they caught: ~0 mm means the grasp closed on air, and
    # 34 mm means it closed on a 34 mm object. That separates "the policy
    # closed" from "the policy grasped" without a human in the loop, which is
    # exactly what `verdict` cannot do on its own. It is a minimum rather than
    # a sample because the operator can judge a tenth of a second after the
    # goal goes out, before the fingers have finished travelling.
    #
    # `grip_end_mm` is the opening at the instant the attempt ENDED. After a
    # close that is ~80 mm (the goal has only just been published), which is
    # the point: it is the real opening after an 'f' or a timeout, where the
    # fingers never moved at all. `grip_verdict_mm` is the opening when the
    # verdict was given — equal to the min while the object is still held,
    # wider if it slipped out on the way home, which is a failure mode the
    # verdict alone records as a plain "fail".
    "grip_end_mm", "grip_min_mm", "grip_verdict_mm",
    "t_start", "t_end", "elapsed_s", "t_verdict",
    "steps", "max_steps", "n_frames", "n_dropped", "hz_mean",
    "runner", "policy_run", "control", "cameras", "segmentation",
    "anchor_ref", "d_rule", "bytes_written", "complete",
]


def step_fields(camera_names, extra=()) -> list[str]:
    """The steps.csv header for THIS session. Pinned once, then never derived.

    The per-camera block is the one place the column set depends on a runtime
    flag (`--cameras`), so the discipline is preserved by fixing the list at
    recorder construction and writing it into the manifest as `steps_csv_fields`
    — identical for every row of a session, and self-describing to a reader.
    """
    return (list(STEP_FIELDS_BASE)
            + [f"{n}_{k}" for n in camera_names for k in _PER_CAM]
            + list(extra))


# ── messages ─────────────────────────────────────────────────────────────────

class _Msg:
    __slots__ = ("kind", "payload")

    def __init__(self, kind: str, payload: Any = None):
        self.kind = kind
        self.payload = payload


class _Sink:
    """Every open handle for one attempt. Opened on START, closed on DONE."""

    def __init__(self, root: Path, camera_names, fields, record_depth: bool,
                 video_fps: float, arrays_spec: dict):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.camera_names = list(camera_names)
        self.record_depth = bool(record_depth)
        self.bytes = 0
        self.n_frames = 0
        self.n_steps = 0
        self._prev_t = None

        self._csv_f = (root / "steps.csv").open("w", newline="")
        self._csv = csv.DictWriter(self._csv_f, fieldnames=fields,
                                   extrasaction="ignore")
        self._csv.writeheader()
        self._csv_f.flush()

        # buffering=0: a byte that reached write() has reached the kernel, so
        # os._exit(0) and SIGKILL cannot lose it. There is no Python-level
        # buffer to flush and therefore none to lose.
        self._bins: dict[str, Any] = {}
        self._shapes: dict[str, tuple] = {}
        for name, spec in arrays_spec.items():
            self._bins[name] = (root / name).open("wb", buffering=0)
            self._shapes[name] = tuple(spec["shape"])
        self._depth = {}
        self._video = {}
        for cam in self.camera_names:
            if self.record_depth:
                self._depth[cam] = (root / f"depth_{cam}.u16").open(
                    "wb", buffering=0)
            (root / f"labels_{cam}").mkdir(exist_ok=True)
            path = root / f"rgb_{cam}.mp4"
            vw = cv2.VideoWriter(str(path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(video_fps), (640, 480))
            # Checked, not assumed: a VideoWriter that failed to open accepts
            # every write() silently and leaves a zero-byte file.
            if not vw.isOpened():
                vw = None
                print(f"[rec] WARNING: cannot open {path} — no RGB video for "
                      f"{cam} this attempt. mp4v is the only codec this "
                      "OpenCV build has; avc1 is not available.", flush=True)
            self._video[cam] = vw

    def write_array(self, name: str, arr) -> None:
        """Append one fixed-stride record. The shape is CHECKED, not trusted.

        Every one of these files is a headerless stream of identical records,
        read back as `size // record_bytes` — so a single record of the wrong
        width does not corrupt itself, it makes every later record in the file
        unlocatable. That is worth an exception: the worker captures it, the
        main loop surfaces it, and the attempt is reported broken instead of
        being written as a stream that looks well-formed and decodes to noise.
        A shape mismatch between the manifest and the writer is exactly how
        `policy_pc.f4` ended up with two strides in one file.
        """
        f = self._bins.get(name)
        if f is None:
            return
        a = np.ascontiguousarray(arr)
        want = tuple(self._shapes[name])
        if a.shape != want:
            raise ValueError(
                f"{name}: record shape {a.shape} does not match the manifest's "
                f"{want}. A short or long record makes every LATER record in "
                "this file unlocatable, so it is refused rather than written.")
        b = a.tobytes()
        f.write(b)
        self.bytes += len(b)

    def write_row(self, row: dict) -> None:
        self._csv.writerow(row)
        self._csv_f.flush()
        self.n_steps += 1

    def close(self) -> None:
        for f in list(self._bins.values()) + list(self._depth.values()):
            try:
                f.close()
            except Exception:
                pass
        for vw in self._video.values():
            if vw is not None:
                try:
                    vw.release()      # this is what writes the moov atom
                except Exception:
                    pass
        try:
            self._csv_f.close()
        except Exception:
            pass


class _WriterThread(threading.Thread):
    """The one thread that touches disk.

    Modelled on `test_perception_viz.PerceptionWorker`, including the parts that
    look like paranoia and are not: the event is `_stop_evt` because
    `threading.Thread` already owns a `_stop()` that `join()` calls, and the
    queue is drained with a timeout rather than slept on so shutdown returns
    promptly.

    ITS HARD RULE, generalised from that precedent: it touches numpy,
    `cv2.imencode`, `cv2.VideoWriter` and file handles, and NEVER `cv2.imshow`,
    `cv2.waitKey` or Open3D. highgui and Filament are the parts that are not
    thread-safe; the codec paths are, and they release the GIL.
    """

    def __init__(self, q: "queue.Queue[_Msg]", root: Path, camera_names,
                 fields, record_depth: bool, video_fps: float,
                 arrays_spec: dict, depth_scales: dict,
                 attempt_fields=None):
        super().__init__(daemon=True, name="exp-writer")
        self._q = q
        self.root = root
        self.camera_names = list(camera_names)
        self.fields = fields
        self.attempt_fields = list(attempt_fields or ATTEMPT_FIELDS)
        self.record_depth = bool(record_depth)
        self.video_fps = float(video_fps)
        self.arrays_spec = arrays_spec
        self.depth_scales = depth_scales
        self.ok = True
        self.error: Optional[str] = None
        self.bytes = 0
        self._sink: Optional[_Sink] = None
        self._prev_t: Optional[float] = None
        # HOW MANY ATTEMPTS HAVE BEEN OPENED, which is what names their
        # directories. See _start: the bin index cannot do this job.
        self._n_started = 0
        # NOT `_stop`: threading.Thread already owns a `_stop()` that join()
        # calls, and shadowing it makes join() raise. It means ABANDON — the
        # ordinary flush is the CLOSE sentinel, which drains first.
        self._stop_evt = threading.Event()

    # ---- lifetime -----------------------------------------------------------

    def run(self) -> None:
        try:
            while True:
                try:
                    msg = self._q.get(timeout=0.25)
                except queue.Empty:
                    if self._stop_evt.is_set():
                        break
                    continue
                if msg.kind == "close":
                    break
                self._dispatch(msg)
        except Exception:
            self.ok = False
            self.error = traceback.format_exc()
            self._drain()
        finally:
            if self._sink is not None:
                self._sink.close()
                self._sink = None

    def _drain(self) -> None:
        """After a failure, keep the queue from growing without bound."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _dispatch(self, msg: _Msg) -> None:
        if msg.kind == "start":
            self._start(msg.payload)
        elif msg.kind == "step":
            self._step(msg.payload)
        elif msg.kind == "done":
            self._done(msg.payload)
        elif msg.kind == "event":
            self._event(msg.payload)

    # ---- per attempt --------------------------------------------------------

    def _start(self, row: dict) -> None:
        if self._sink is not None:
            self._sink.close()
        self._prev_t = None
        # NAMED BY THE ORDER IT RAN IN, NOT BY WHICH BIN IT WAS.
        #
        # This was `attempt_{row['attempt'] - 1:03d}`, and any bin run more
        # than once then wrote into the directory of the previous try. Every
        # handle here is opened "w"/"wb", so that is not a merge, it is a
        # silent overwrite: attempts.csv kept all the rows — ExperimentSession
        # is careful that a retry never erases the row it repeats — while the
        # video, depth, labels and policy IO underneath them were replaced by
        # whichever try happened last. Seen on a real session: four tries of
        # +x, four rows, one directory, three tries of recording gone.
        #
        # A bin can be run again for two different reasons and BOTH hit it:
        # 'r' (retry) and 't' (void, which by design does not consume the bin).
        #
        # The ordinal is the row's position in attempts.csv, so for a session
        # with no retries and no voids it is exactly `attempt - 1` and the
        # naming of every session recorded so far is unchanged. `dir_name` is
        # written into the row either way, so the mapping is explicit on disk
        # rather than a convention a reader has to re-derive.
        d = self.root / f"attempt_{self._n_started:03d}"
        self._n_started += 1
        self._sink = _Sink(d, self.camera_names, self.fields,
                           self.record_depth, self.video_fps, self.arrays_spec)
        _write_json(d / "attempt.json", dict(row, complete=False,
                                             heavy_complete=True))
        self._event({"kind": "attempt_start", "t": time.time(), **row})

    def _done(self, row: dict) -> None:
        sink, self._sink = self._sink, None
        if sink is None:
            return
        sink.close()
        el = row.get("elapsed_s") or 0.0
        full = dict(row)
        full.update(dir_name=sink.root.name, n_frames=sink.n_frames,
                    steps=row.get("steps") if row.get("steps") is not None
                    else sink.n_steps,
                    hz_mean=(sink.n_steps / el) if el > 0 else "",
                    bytes_written=sink.bytes, complete=True)
        _write_json(sink.root / "attempt.json", full)
        _append_csv(self.root / "attempts.csv", self.attempt_fields, full)
        self.bytes += sink.bytes
        self._event({"kind": "attempt_done", "t": time.time(),
                     "attempt": full.get("attempt"), "bin": full.get("bin"),
                     "ending": full.get("ending"),
                     "verdict": full.get("verdict"),
                     "n_frames": sink.n_frames, "bytes": sink.bytes})

    def _event(self, rec: dict) -> None:
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ---- per step -----------------------------------------------------------

    def _step(self, p: dict) -> None:
        sink = self._sink
        if sink is None:
            return                      # a step with no attempt open: ignore
        row = p["row"]
        t = row.get("t_mono")
        row["dt_ms"] = ("" if self._prev_t is None
                        else round((t - self._prev_t) * 1e3, 2))
        self._prev_t = t
        row["t_mono"] = f"{t:.6f}"

        for name, arr in p["arrays"].items():
            sink.write_array(name, arr)

        heavy = p.get("heavy")
        if heavy is not None:
            idx = sink.n_frames
            for cam, (color, depth, hand, obj) in heavy.items():
                lab = np.zeros(hand.shape, dtype=np.uint8)
                lab[hand > 0] = LABEL_HAND
                n_hand = int((lab == LABEL_HAND).sum())
                n_over = 0
                if obj is not None:
                    n_over = int(((lab == LABEL_HAND) & (obj > 0)).sum())
                    lab[obj > 0] = LABEL_OBJECT      # object wins the overlap
                ok, buf = cv2.imencode(
                    ".png", lab, [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
                png_bytes = 0
                if ok:
                    path = sink.root / f"labels_{cam}" / f"{idx:06d}.png"
                    path.write_bytes(buf.tobytes())
                    png_bytes = int(buf.size)
                    sink.bytes += png_bytes
                vw = sink._video.get(cam)
                if vw is not None:
                    vw.write(color)
                n_valid = 0
                if self.record_depth and depth is not None:
                    scale = float(self.depth_scales.get(cam) or 0.001)
                    raw = np.rint(depth / scale)
                    np.clip(raw, 0, 65535, out=raw)
                    raw = raw.astype(np.uint16)
                    n_valid = int((raw > 0).sum())
                    sink._depth[cam].write(raw.tobytes())
                    sink.bytes += raw.nbytes
                row[f"{cam}_lab_hand_px"] = n_hand
                row[f"{cam}_lab_obj_px"] = int((lab == LABEL_OBJECT).sum())
                row[f"{cam}_lab_overlap_px"] = n_over
                row[f"{cam}_depth_valid_px"] = n_valid
                row[f"{cam}_png_bytes"] = png_bytes
            sink.n_frames += 1

        sink.write_row(row)


def _write_json(path: Path, obj: dict) -> None:
    """Atomic, so a kill mid-write leaves the previous version, not half of one.

    The pattern is `capture_image_and_pose.py`'s: write a sibling .tmp, then
    rename over the target, which is atomic within a filesystem.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable))
    tmp.replace(path)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _append_csv(path: Path, fields, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


# ── the public face ──────────────────────────────────────────────────────────

class ExpRecorder:
    """Records a session. Every method here runs on the CONTROL thread.

    So every method here is short, allocates as little as it can, and never
    blocks: `on_step` copies four small arrays and puts a message on an
    unbounded queue. The queue is unbounded because a bounded one would either
    block the control loop or drop silently, and neither is acceptable — the
    shedding decision is made explicitly by the PRODUCER from `qsize()`, before
    the copies are even made, so a shed frame costs less than a recorded one.

    A DROPPED FRAME IS MADE VISIBLE FIVE WAYS, because a silent drop is the
    failure mode that makes a dataset quietly wrong rather than obviously
    missing: `heavy_idx = -1` in steps.csv, the HUD line, a rate-limited print,
    an events.jsonl record, and `heavy_complete: false` in attempt.json.
    """

    def __init__(self, root, rigs, *, record_depth: bool = True,
                 video_fps: float = 10.0, extra_fields=(),
                 arrays_spec: Optional[dict] = None,
                 attempt_csv_fields=None,
                 manifest: Optional[dict] = None,
                 row_constants: Optional[dict] = None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._rigs = list(rigs)
        self.camera_names = [r.name for r in self._rigs]
        self.record_depth = bool(record_depth)

        free = shutil.disk_usage(self.root).free
        if free < MIN_FREE_BYTES:
            raise SystemExit(
                f"{self.root} has {free / 1e9:.1f} GB free. A session records "
                f"~{260 if record_depth else 15} MB per attempt per two "
                "cameras; refusing to start below 5 GB. Free some space, pass "
                "--exp-no-depth, or point --exp-out somewhere else.")

        self.arrays_spec = dict(arrays_spec or {})
        self.fields = step_fields(self.camera_names, extra_fields)
        self.attempt_fields = list(attempt_csv_fields or ATTEMPT_FIELDS)
        self._q: "queue.Queue[_Msg]" = queue.Queue()
        depth_scales = {r.name: getattr(r.camera, "depth_scale", None)
                        for r in self._rigs}
        self._writer = _WriterThread(self._q, self.root, self.camera_names,
                                     self.fields, self.record_depth,
                                     video_fps, self.arrays_spec, depth_scales,
                                     attempt_fields=self.attempt_fields)
        self._writer.start()

        # Repeated on every attempts.csv row on purpose. That file is the one
        # someone opens six months later, and a row that cannot say which
        # policy produced it is the "which checkpoint was cp3" problem this
        # repo has already had once.
        self.row_constants = dict(row_constants or {})
        self.attempt: Optional[int] = None
        self.bin: Optional[str] = None
        self.n_steps = 0
        self.heavy_idx = -1
        self.dropped = 0
        # How long `on_step` took LAST time. Only the recorder can time its own
        # body, and only the next row can carry the answer — so `ms_rec` is
        # always one step behind, which is what makes it free. It is in the CSV
        # so a regression (someone moving the PNG encode onto the control
        # thread) shows up in the data rather than only in a stopwatch.
        self._last_ms = 0.0
        self._drop_notes = 0
        self._reported_error = False

        man = dict(manifest or {})
        man.update(schema=SCHEMA, complete=False,
                   steps_csv_fields=self.fields,
                   attempts_csv_fields=self.attempt_fields,
                   records={k: dict(v) for k, v in self.arrays_spec.items()},
                   cameras_recorded=self.camera_names,
                   record_depth=self.record_depth,
                   video={"container": "mp4", "fourcc": "mp4v", "lossy": True,
                          "fps_declared": float(video_fps),
                          "fps_is_nominal": True,
                          "frame_index_is": "heavy_idx"},
                   labels={"container": "png", "dtype": "uint8",
                           "values": {"0": "background", "1": "hand",
                                      "2": "object"},
                           "overlap_rule": "object wins",
                           "png_compression": PNG_LEVEL,
                           "name_index_is": "heavy_idx"})
        self._manifest = man
        _write_json(self.root / "manifest.json", man)
        (self.root / "argv.txt").write_text("\n".join(sys.argv) + "\n")
        print(f"[rec] recording to {self.root}"
              + ("" if self.record_depth else "  (no depth)"), flush=True)

    # ---- session lifecycle --------------------------------------------------

    def on_attempt_start(self, row: dict) -> None:
        self.attempt = int(row["attempt"])
        self.bin = row.get("bin")
        self.n_steps = 0
        self.heavy_idx = -1
        self._q.put_nowait(_Msg("start", dict(row)))

    def on_attempt_done(self, row: dict) -> None:
        out = dict(row)
        out.setdefault("n_dropped", self.dropped)
        out.update(self.row_constants)
        self._q.put_nowait(_Msg("done", out))
        self.attempt = None
        self.bin = None

    def on_session_end(self, summary: dict) -> None:
        self._q.put_nowait(_Msg("event", {"kind": "session_end",
                                          "t": time.time(), **summary}))
        self._manifest.update(complete=True, ended_utc=time.time(),
                              n_attempts_run=summary.get("attempted"),
                              n_dropped=self.dropped)
        _write_json(self.root / "manifest.json", self._manifest)

    # ---- per step -----------------------------------------------------------

    def on_step(self, *, step: int, t_mono: float, perception, fused,
                T_base_hand, gripper_norm, pc, robot_state, action,
                have_obs: bool, armed: bool, executed: bool,
                grasp_close: bool, clamped: bool, ms_obs: float,
                ms_pol: float, adapter, target=None, d_world=None,
                grip_logit=float("nan")) -> None:
        """Copy what the writer needs and hand it over. Nothing else.

        Called BEFORE the display block on purpose. `overlay_mask` copies today,
        but the HUD path draws with `cv2.putText` into the first rig's `view`,
        which is one refactor away from drawing into the frame being recorded.
        Recording first makes the ordering explicit rather than lucky.
        """
        if self.attempt is None:
            return
        qd = self._q.qsize()
        heavy = None
        if qd < HEAVY_BACKLOG:
            heavy = {}
            for rig in self._rigs:
                pair = perception.last_frames.get(rig.name)
                if pair is None:
                    continue
                color_bgr, hand_mask = pair
                obj = perception.last_object_masks.get(rig.name)
                depth = (perception.last_depths.get(rig.name)
                         if self.record_depth else None)
                # THE COPY ON `color_bgr` IS MANDATORY AND MUST STAY HERE.
                # camera.py's get_frames returns np.asanyarray(frame.get_data())
                # — a VIEW into librealsense-owned memory that the driver
                # recycles once the rs.frame is released. Handing the view to
                # another thread produces plausible video with occasional
                # tearing from a later frame, which looks like a camera fault.
                heavy[rig.name] = (
                    color_bgr.copy(),
                    None if depth is None else depth.copy(),
                    hand_mask.copy(),
                    None if obj is None else obj.copy(),
                )
            self.heavy_idx += 1
            hidx = self.heavy_idx
        else:
            self.dropped += 1
            hidx = -1
            self._note_drop(qd)

        arrays = self._arrays(pc, robot_state, action, T_base_hand, adapter,
                              have_obs)
        row = self._row(step=step, t_mono=t_mono, heavy_idx=hidx, fused=fused,
                        T_base_hand=T_base_hand, gripper_norm=gripper_norm,
                        action=action, have_obs=have_obs, armed=armed,
                        executed=executed, grasp_close=grasp_close,
                        clamped=clamped, ms_obs=ms_obs, ms_pol=ms_pol,
                        target=target, d_world=d_world, q_depth=qd,
                        grip_logit=grip_logit)
        extra_row, _ = _adapter_step(adapter)
        row.update(extra_row)
        row["ms_rec"] = round(self._last_ms, 3)
        self.n_steps += 1
        self._q.put_nowait(_Msg("step", {"row": row, "heavy": heavy,
                                         "arrays": arrays}))
        self._last_ms = (time.time() - t_mono) * 1e3

    def _arrays(self, pc, rs, action, T_base_hand, adapter, have_obs) -> dict:
        """One record per STEP, always — NaN-filled when there was no observation.

        Skipping records on an unusable frame would silently decouple the record
        ordinal from `step`, so `policy_pc[step]` would stop meaning what it
        says. NaN is unambiguous because a real cloud never contains one, and
        `obs_usable` in the CSV says whether the record means anything.
        """
        out = {}
        _, extra = _adapter_step(adapter)
        supplied = {"pose_base_hand.f8": T_base_hand,
                    "policy_pc.f4": pc, "policy_rs.f4": rs,
                    "policy_action.f4": action, **extra}
        for name, spec in self.arrays_spec.items():
            v = supplied.get(name)
            dt = np.dtype(spec["dtype"])
            if v is None:
                out[name] = np.full(tuple(spec["shape"]), np.nan, dtype=dt)
            else:
                out[name] = np.asarray(v, dtype=dt)
        return out

    def _row(self, *, step, t_mono, heavy_idx, fused, T_base_hand,
             gripper_norm, action, have_obs, armed, executed, grasp_close,
             clamped, ms_obs, ms_pol, target, d_world, q_depth,
             grip_logit) -> dict:
        p = T_base_hand[:3, 3] if T_base_hand is not None else (np.nan,) * 3
        q = _quat_wxyz(T_base_hand)
        a = (np.asarray(action, dtype=np.float64) if action is not None
             else np.full(7, np.nan))
        d = (np.asarray(d_world, dtype=np.float64) if d_world is not None
             else np.full(3, np.nan))
        row = {
            "attempt": self.attempt, "step": step, "heavy_idx": heavy_idx,
            # NUMERIC here, formatted by the worker after it has taken the
            # difference. Formatting on the control thread cost the worker the
            # only quantity it can compute that the producer cannot.
            "t_mono": float(t_mono), "t_wall": f"{time.time():.3f}",
            "bin": self.bin,
            "d_world_x": _g(d[0]), "d_world_y": _g(d[1]), "d_world_z": _g(d[2]),
            "ee_x": _g(p[0]), "ee_y": _g(p[1]), "ee_z": _g(p[2]),
            "ee_qw": _g(q[0]), "ee_qx": _g(q[1]), "ee_qy": _g(q[2]),
            "ee_qz": _g(q[3]),
            "grip_norm": _g(gripper_norm),
            "obs_usable": int(bool(have_obs)),
            "n_obj": len(fused.object_xyz), "n_hand": len(fused.hand_xyz),
            "obj_range_m": _median_range(fused.object_xyz),
            "hand_range_m": _median_range(fused.hand_xyz),
            "arm_dropped": getattr(fused, "arm_dropped", ""),
            "arm_fallback": getattr(fused, "arm_fallback", "") or "",
            "grasp_points": getattr(fused, "grasp_points", ""),
            "grasp_vetoed": getattr(fused, "grasp_vetoed", ""),
            "act_dx": _g(a[0]), "act_dy": _g(a[1]), "act_dz": _g(a[2]),
            "act_rx": _g(a[3]), "act_ry": _g(a[4]), "act_rz": _g(a[5]),
            "act_grip": _g(a[6]),
            "act_pos_mm": _g(float(np.linalg.norm(a[:3])) * 1e3),
            "act_rot_deg": _g(np.rad2deg(float(np.linalg.norm(a[3:6])))),
            "clamped": int(bool(clamped)), "grasp_close": int(bool(grasp_close)),
            "grip_logit": _g(grip_logit),
            "armed": int(bool(armed)), "executed": int(bool(executed)),
            "ms_obs": round(float(ms_obs), 2), "ms_pol": round(float(ms_pol), 2),
            "q_depth": q_depth, "dropped_total": self.dropped,
        }
        if target is not None:
            row["target_x"] = _g(target[0, 3])
            row["target_y"] = _g(target[1, 3])
            row["target_z"] = _g(target[2, 3])
        for name, d_ in (getattr(fused, "per_camera", None) or {}).items():
            row[f"{name}_obj"] = d_.get("object", "")
            row[f"{name}_hand"] = d_.get("hand", "")
            row[f"{name}_robot_rm"] = d_.get("robot_pts_removed", "")
            row[f"{name}_finger_rm"] = d_.get("finger_pts_removed", "")
            row[f"{name}_stale"] = int(bool(d_.get("used_last_hand")
                                            or d_.get("used_last_object")))
            row[f"{name}_reseeds"] = d_.get("seg_reseeds", "")
            row[f"{name}_obj_from_mask"] = int(bool(d_.get("object_from_mask")))
        return row

    # ---- health -------------------------------------------------------------

    def _note_drop(self, qd: int) -> None:
        self._drop_notes += 1
        if self._drop_notes <= 3 or self._drop_notes % 50 == 0:
            print(f"[rec] DROPPED a frame (queue {qd}) — the writer is not "
                  f"keeping up; {self.dropped} lost so far. Check disk space "
                  "and whether --exp-out is on a slow filesystem.", flush=True)
        self._q.put_nowait(_Msg("event", {"kind": "drop", "t": time.time(),
                                          "attempt": self.attempt,
                                          "q": qd, "total": self.dropped}))

    def check(self) -> None:
        """Once per loop iteration. A writer that died must not die silently."""
        if not self._writer.ok and not self._reported_error:
            self._reported_error = True
            print(f"[rec] WRITER FAILED — recording has stopped:\n"
                  f"{self._writer.error}", flush=True)

    def hud(self) -> str:
        if not self._writer.ok:
            return "REC FAILED - not recording"
        mb = self._writer.bytes / 1e6
        s = (f"REC a{self.attempt:02d} s={self.n_steps:03d} "
             f"f={self.heavy_idx + 1:03d} q={self._q.qsize()} {mb:.0f}MB"
             if self.attempt is not None else
             f"REC idle  {mb:.0f}MB written")
        return s + (f"  DROPPED {self.dropped}" if self.dropped else "")

    def close(self) -> None:
        """Flush and JOIN. Must run inside main()'s finally.

        `__main__` exits through `exit_without_finalizing()` — `os._exit(0)`,
        which flushes stdout and stderr and nothing else. No atexit hook runs,
        no daemon thread gets another scheduling slice, and anything still on
        this queue is gone. The CLOSE sentinel rather than `_stop_evt` is what
        makes this a flush: the event means abandon, the sentinel means drain
        everything queued before it and then exit.
        """
        try:
            self._q.put_nowait(_Msg("close"))
        except Exception:
            pass
        self._writer.join(timeout=30.0)
        if self._writer.is_alive():
            print(f"[rec] WRITER DID NOT FINISH — {self._q.qsize()} items "
                  "unwritten and LOST", flush=True)
            self._writer._stop_evt.set()
            self._writer.join(timeout=2.0)
        if self._writer.error:
            print(f"[rec] writer failed:\n{self._writer.error}", flush=True)
        print(f"[rec] {self.root}  {self._writer.bytes / 1e6:.0f} MB, "
              f"{self.dropped} frames dropped", flush=True)


def _median_range(xyz) -> str:
    if xyz is None or len(xyz) < 3:
        return ""
    return _g(np.linalg.norm(np.median(np.asarray(xyz, np.float64), axis=0)))


def _g(x) -> str:
    """%.9g — the shortest decimal that round-trips IEEE-754 binary32 exactly.

    So the CSV is not an approximation of the binary; it is the same number.
    """
    try:
        return f"{float(x):.9g}"
    except (TypeError, ValueError):
        return ""


def _quat_wxyz(T):
    """Rotation matrix -> (w, x, y, z), by hand.

    Shepperd's method rather than scipy, because this runs on the CONTROL
    thread once per step and `Rot.from_matrix(...).as_quat()` measured a
    meaningful fraction of the whole recorder's per-step budget. It is twenty
    lines of arithmetic with no import and no object construction.
    """
    if T is None:
        return (np.nan,) * 4
    m = np.asarray(T, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s2 = np.sqrt(tr + 1.0) * 2.0
        return (0.25 * s2, (m[2, 1] - m[1, 2]) / s2,
                (m[0, 2] - m[2, 0]) / s2, (m[1, 0] - m[0, 1]) / s2)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s2 = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return ((m[2, 1] - m[1, 2]) / s2, 0.25 * s2,
                (m[0, 1] + m[1, 0]) / s2, (m[0, 2] + m[2, 0]) / s2)
    if m[1, 1] > m[2, 2]:
        s2 = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 2] - m[2, 0]) / s2, (m[0, 1] + m[1, 0]) / s2,
                0.25 * s2, (m[1, 2] + m[2, 1]) / s2)
    s2 = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[1, 0] - m[0, 1]) / s2, (m[0, 2] + m[2, 0]) / s2,
            (m[1, 2] + m[2, 1]) / s2, 0.25 * s2)


def _adapter_step(adapter):
    fn = getattr(adapter, "record_step", None)
    if fn is None:
        return {}, {}
    try:
        row, arrays = fn()
        return dict(row or {}), dict(arrays or {})
    except Exception:
        return {}, {}


def attempt_fields(adapter) -> list[str]:
    """The attempts.csv header: the pinned block plus whatever the policy adds.

    Separate from `step_fields` because the two tables answer different
    questions and get written by different call sites — and separate from
    `adapter_spec` because `csv.DictWriter(extrasaction="ignore")` silently
    DROPS a key that is not in its fieldnames. An adapter that measured the
    realized bin and had it thrown away at the last moment is exactly the sort
    of quiet loss this file is written to avoid.
    """
    out = list(ATTEMPT_FIELDS)
    fn = getattr(adapter, "record_outcome_fields", None)
    if fn is not None:
        try:
            out += [c for c in (fn() or ()) if c not in out]
        except Exception:
            pass
    return out


def adapter_spec(adapter):
    """(extra csv fields, binary array spec) for whatever policy is loaded.

    The base spec is the same for every policy this runner drives; the adapter
    adds what only it knows about. Keeping it a hook is the same reason the
    adapter exists at all — `my_policy_runner` must not learn what a regrasp
    direction is.

    `policy_pc.f4` IS `build_policy_cloud`'S OUTPUT, and that shape is fixed at
    [NUM_OBJECT_POINTS + NUM_HAND_POINTS, PC_CHANNELS] whatever policy is
    driving. This used to take the channel count from the CHECKPOINT's
    `DATA.pc_channels`, which is 7 for a regrasp run — so the spec said 7 while
    every array written was 5 wide. On a usable frame the real [1024, 5] went
    out; on an unusable one `_arrays` NaN-filled at the DECLARED [1024, 7]. The
    file came out with a mixed stride, which is the one corruption
    fixed-stride records cannot survive: `size // record_bytes` stops working
    and no record after the first short one can be located. Measured on one
    attempt: 50 usable frames and 70 held ones in a file that looked like 148
    records of a 120-step attempt.

    The 7-channel cloud is a SEPARATE stream. The regrasp adapter declares it
    as `policy_pc7.f4` through the hook below, which is exactly the split this
    function exists to keep.
    """
    spec = {
        "pose_base_hand.f8": {"dtype": "<f8", "shape": [4, 4], "index": "step"},
        "policy_pc.f4": {"dtype": "<f4",
                         "shape": [NUM_OBJECT_POINTS + NUM_HAND_POINTS,
                                   PC_CHANNELS],
                         "index": "step"},
        "policy_rs.f4": {"dtype": "<f4", "shape": [32], "index": "step"},
        "policy_action.f4": {"dtype": "<f4", "shape": [7], "index": "step"},
    }
    fields: tuple = ()
    fn = getattr(adapter, "record_spec", None)
    if fn is not None:
        try:
            fields, extra = fn()
            spec.update(extra or {})
            fields = tuple(fields or ())
        except Exception:
            fields = ()
    return fields, spec


def build_manifest(args, rigs, adapter, session_id: str,
                   extra: Optional[dict] = None) -> dict:
    """Everything constant across the session, written once.

    THE TEST THIS HAS TO PASS is that a reader with only this file and the
    binaries can put a recorded depth frame into the base frame. That needs the
    colour intrinsics (the colour ones, because depth is aligned TO colour), the
    depth scale PER CAMERA (a D435 and a D455 can legitimately report different
    ones, so a session-level scale would be a latent lie), and either
    `T_base_cam` for a fixed rig or `T_hand_cam` plus the per-frame pose for the
    wrist. `ExtractionParams` goes in too: without the crop radius, the strides
    and the depth limits, reproducing the POLICY's cloud from the depth is
    guesswork.
    """
    cams = []
    for rig in rigs:
        intr = None
        try:
            i = rig.camera.get_intrinsics()
            intr = {"width": i.width, "height": i.height, "fx": i.fx,
                    "fy": i.fy, "cx": i.cx, "cy": i.cy,
                    "distortion_model": i.distortion_model,
                    "coeffs": list(i.coeffs)}
        except Exception:
            pass
        cams.append({
            "name": rig.name, "kind": rig.kind, "serial": rig.serial,
            "depth_scale_m": getattr(rig.camera, "depth_scale", None),
            "intrinsics": intr,
            "T_hand_cam": (None if rig.T_hand_cam is None
                           else np.asarray(rig.T_hand_cam).tolist()),
            "T_base_cam": (None if rig.T_base_cam is None
                           else np.asarray(rig.T_base_cam).tolist()),
            "exclude_robot": bool(rig.exclude_robot),
            "extraction_params": dataclasses.asdict(rig.params),
        })
    man = {
        "session_id": session_id,
        "created_utc": time.time(),
        "argv": list(sys.argv),
        "git": _git_head(),
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "cv2": cv2.__version__},
        "runner": Path(sys.argv[0]).stem,
        "frames": {"width": 640, "height": 480, "color_format": "bgr8",
                   "depth_dtype": "<u2", "depth_shape": [480, 640],
                   "depth_order": "C", "depth_bytes_per_record": 614400,
                   "depth_units": "raw sensor units; multiply by "
                                  "cameras[i].depth_scale_m",
                   "depth_aligned_to": "color"},
        "control": {"mode": args.control, "rate_hz": args.rate_hz,
                    "max_steps": args.max_steps,
                    "step_mode": bool(args.step_mode),
                    "gripper": bool(args.enable_gripper),
                    "auto_home_delay": args.auto_home_delay},
        "segmentation": {"backend": args.segmentation},
        "calib_session": args.calib_session,
        "cameras": cams,
        "transforms": {"T_SIMWORLD_BASE": None},
        "exp_bins": list(getattr(args, "exp_bins", None) or []),
    }
    if extra:
        man.update(extra)
    return man


def _git_head() -> dict:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=str(Path(__file__).resolve().parent),
                             capture_output=True, text=True, timeout=3)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=str(Path(__file__).resolve().parent),
                               capture_output=True, text=True, timeout=5)
        return {"commit": out.stdout.strip(),
                "dirty": bool(dirty.stdout.strip())}
    except Exception:
        return {}
