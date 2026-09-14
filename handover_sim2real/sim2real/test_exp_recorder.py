#!/usr/bin/env python3
"""Offline checks for exp_recorder. No camera, no robot, no ROS.

WHAT IS WORTH TESTING HERE is not that files appear — that is visible — but the
three properties a session's usefulness rests on and that are invisible until
the data is wrong:

  * every stream's length agrees with the step count, so `policy_pc[step]` means
    what it says;
  * a frame read back through the published reader reconstructs the cloud that
    went in, bit-for-bit on the metric depth;
  * a stalled writer sheds HEAVY frames without blocking the producer and
    without losing a single CSV row.

The third is the one that decides whether this can sit in a 150 ms control loop.

    python test_exp_recorder.py
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_recorder as R                                        # noqa: E402
import read_exp_session as RD                                   # noqa: E402


# ── fakes, kept minimal and shaped exactly like the real objects ─────────────

@dataclasses.dataclass
class _Params:
    full_cloud_stride: int = 2
    min_depth_m: float = 0.1
    max_depth_m: float = 1.5


class _Cam:
    def __init__(self, scale=0.001):
        self.depth_scale = scale

    def get_intrinsics(self):
        return types.SimpleNamespace(width=640, height=480, fx=600.0, fy=600.0,
                                     cx=320.0, cy=240.0,
                                     distortion_model="brown_conrady",
                                     coeffs=(0.0,) * 5)


def _rig(name, kind="fixed"):
    return types.SimpleNamespace(
        name=name, kind=kind, camera=_Cam(), params=_Params(),
        T_hand_cam=None if kind == "fixed" else np.eye(4),
        T_base_cam=np.eye(4) if kind == "fixed" else None,
        exclude_robot=False, serial=f"SER{name}")


class _Perc:
    def __init__(self, names, rng):
        self.last_frames = {}
        self.last_depths = {}
        self.last_object_masks = {}
        self._names = names
        self._rng = rng

    def frame(self, k):
        for n in self._names:
            color = np.full((480, 640, 3), k % 251, np.uint8)
            depth = np.full((480, 640), 0.500 + 0.001 * (k % 7), np.float32)
            hand = np.zeros((480, 640), np.uint8)
            hand[100:150, 100:200] = 1
            obj = np.zeros((480, 640), np.uint8)
            obj[120:180, 150:260] = 1          # deliberately overlaps the hand
            self.last_frames[n] = (color, hand)
            self.last_depths[n] = depth
            self.last_object_masks[n] = obj


def _fused(rng, names):
    f = types.SimpleNamespace()
    f.object_xyz = rng.normal(size=(400, 3)).astype(np.float32)
    f.hand_xyz = rng.normal(size=(120, 3)).astype(np.float32)
    f.arm_dropped = 3
    f.arm_fallback = None
    f.grasp_points = 12
    f.grasp_vetoed = 0
    f.per_camera = {n: {"object": 400, "hand": 120, "robot_pts_removed": 5,
                        "finger_pts_removed": 2, "used_last_hand": False,
                        "used_last_object": False, "seg_reseeds": 0,
                        "object_from_mask": True} for n in names}
    return f


def _args():
    return types.SimpleNamespace(
        control="rate", rate_hz=6.67, max_steps=50, step_mode=False,
        enable_gripper=True, auto_home_delay=5.0, segmentation="sam2",
        calib_session="D435", exp_bins=["+x", "-y"])


def _drive(rec, perc, fused, rng, n, names, pause=0.0):
    T = np.eye(4)
    for k in range(n):
        perc.frame(k)
        T[:3, 3] = [0.4 + 0.001 * k, 0.0, 0.5]
        rec.on_step(step=k, t_mono=time.time(), perception=perc, fused=fused,
                    T_base_hand=T, gripper_norm=1.0,
                    pc=rng.normal(size=(1024, 5)).astype(np.float32),
                    robot_state=rng.normal(size=32).astype(np.float32),
                    action=rng.normal(size=7).astype(np.float32),
                    have_obs=True, armed=True, executed=True,
                    grasp_close=False, clamped=False, ms_obs=40.0, ms_pol=9.0,
                    adapter=types.SimpleNamespace(), d_world=[1.0, 0.0, 0.0])
        if pause:
            time.sleep(pause)


# ── the checks ───────────────────────────────────────────────────────────────

def _round_trip_and_lengths(root: Path) -> None:
    """Lengths agree, and frame 42's depth reconstructs bit-exactly."""
    names = ["wrist", "tripod"]
    rigs = [_rig("wrist", "eye_in_hand"), _rig("tripod", "fixed")]
    rng = np.random.default_rng(0)
    perc, fused = _Perc(names, rng), _fused(rng, names)
    fields, spec = R.adapter_spec(types.SimpleNamespace())
    rec = R.ExpRecorder(root, rigs, video_fps=10.0, extra_fields=fields,
                        arrays_spec=spec,
                        manifest=R.build_manifest(_args(), rigs, None, "T0"),
                        row_constants={"runner": "test", "policy_run": "run9"})
    rec.on_attempt_start({"attempt": 1, "n_attempts": 2, "try": 1, "bin": "+x",
                          "t_start": time.time(), "max_steps": 50,
                          "session_id": "T0"})
    N = 60
    _drive(rec, perc, fused, rng, N, names, pause=0.02)
    rec.on_attempt_done({"attempt": 1, "bin": "+x", "ending": "close",
                         "verdict": "pass", "steps": N, "elapsed_s": 6.0,
                         "t_start": 0.0, "t_end": 6.0, "froze": False,
                         "session_id": "T0", "n_attempts": 2, "try": 1,
                         "max_steps": 50, "t_verdict": 6.5})
    rec.on_session_end({"session_id": "T0", "attempted": 1, "reason": "test"})
    rec.close()
    assert rec.dropped == 0, f"{rec.dropped} frames dropped with an idle writer"

    a = root / "attempt_000"
    man = json.loads((root / "manifest.json").read_text())
    assert man["complete"] is True, "manifest never marked complete"
    aj = json.loads((a / "attempt.json").read_text())
    assert aj["complete"] is True and aj["n_frames"] == N, aj

    import csv as _csv
    rows = list(_csv.DictReader((a / "steps.csv").open()))
    assert len(rows) == N, f"{len(rows)} csv rows for {N} steps"
    assert [int(r["heavy_idx"]) for r in rows] == list(range(N)), \
        "heavy_idx is not 0..N-1 with an idle writer"

    for name, sp in man["records"].items():
        n = int(np.prod(sp["shape"]))
        size = (a / name).stat().st_size
        stride = n * np.dtype(sp["dtype"]).itemsize
        assert size == N * stride, f"{name}: {size} B is not {N} x {stride}"

    for cam in names:
        d = np.memmap(a / f"depth_{cam}.u16", dtype="<u2",
                      mode="r").reshape(-1, 480, 640)
        assert len(d) == N, f"depth_{cam}: {len(d)} records for {N} steps"
        assert len(list((a / f"labels_{cam}").glob("*.png"))) == N
        assert (a / f"rgb_{cam}.mp4").stat().st_size > 0, "empty mp4"

    # The property that matters: metric depth reconstructs EXACTLY, in the same
    # order of operations camera.py uses, so what a reader computes is what the
    # pipeline saw rather than a close approximation of it.
    import cv2
    k = 42
    cam = next(c for c in man["cameras"] if c["name"] == "tripod")
    raw = np.memmap(a / "depth_tripod.u16", dtype="<u2",
                    mode="r").reshape(-1, 480, 640)[k]
    z = raw.astype(np.float32) * cam["depth_scale_m"]
    want = np.full((480, 640), 0.500 + 0.001 * (k % 7), np.float32)
    assert np.array_equal(z, want), (
        f"depth round-trip is not exact: max |dz| = {np.abs(z - want).max():.3e}")

    lab = cv2.imread(str(a / f"labels_tripod/{k:06d}.png"), cv2.IMREAD_UNCHANGED)
    assert lab.dtype == np.uint8 and set(np.unique(lab)) <= {0, 1, 2}, \
        f"label image is {lab.dtype} with values {np.unique(lab)}"
    # object wins the overlap, and the overlap is COUNTED rather than hidden
    assert lab[130, 170] == R.LABEL_OBJECT, "object did not win the overlap"
    assert lab[110, 110] == R.LABEL_HAND
    assert int(rows[k]["tripod_lab_overlap_px"]) == 30 * 50, \
        rows[k]["tripod_lab_overlap_px"]
    print(f"  lengths agree across {len(man['records'])} binaries, "
          f"{len(names)} depth streams and steps.csv at N={N}")
    print("  metric depth round-trips bit-exactly; labels lossless, object wins")

    # THE READER IS THE FORMAT'S TEST. Everything above reads the files the way
    # the writer laid them out, which proves the writer is self-consistent and
    # nothing else. This goes through the reader that ships with the session
    # and reconstructs the cloud in the base frame from the manifest alone.
    sess = RD.load_session(root)
    assert len(sess["attempts"]) == 1 and sess["steps"]["attempt_000"]
    out = RD.frame_cloud(sess, 0, k, "tripod")
    xyz = out["xyz_base"]
    assert len(xyz) == 480 * 640, (
        f"{len(xyz)} points reconstructed from a frame whose depth is uniformly "
        "in range — the min/max depth gate is reading the wrong numbers")
    # T_base_cam is the identity in these fakes, so z is the camera z.
    assert np.allclose(xyz[:, 2], 0.500 + 0.001 * (k % 7), atol=1e-6), (
        "the reconstructed depth is not the recorded depth")
    assert int((out["labels"] == R.LABEL_OBJECT).sum()) == 60 * 110
    assert np.array_equal(out["action"],
                          np.memmap(a / "policy_action.f4", dtype="<f4",
                                    mode="r").reshape(-1, 7)[int(rows[k]["step"])])
    print(f"  read_exp_session reconstructs frame {k}: {len(xyz)} points in "
          "the base frame, labels and action aligned")


def _a_retried_bin_keeps_both_recordings(root: Path) -> None:
    """Running one bin four times must leave four recordings, not one.

    ExperimentSession is deliberate that a retry never erases the attempt it
    repeats — the failed row stays, with its own `try` number, so the session's
    success rate is over attempts and not over whichever try the operator chose
    to keep. That promise was only half kept. attempts.csv had every row; the
    directory under them was named from the BIN index, so all four tries opened
    the same files "wb" and only the last survived.

    Seen on a real session before this was fixed: four tries of +x, four rows,
    one directory. Both keys that re-offer a bin hit it — 'r' and the 't' void,
    which by design does not consume the attempt.
    """
    names = ["tripod"]
    rigs = [_rig("tripod")]
    rng = np.random.default_rng(11)
    perc, fused = _Perc(names, rng), _fused(rng, names)
    fields, spec = R.adapter_spec(types.SimpleNamespace())
    rec = R.ExpRecorder(root, rigs, extra_fields=fields, arrays_spec=spec,
                        manifest=R.build_manifest(_args(), rigs, None, "T5"))
    # One bin, four tries: the exact shape of the session that lost its data.
    tries = (("timeout", "fail"), ("close", "pass"),
             ("user_stop", "void"), ("close", "pass"))
    for k, (ending, verdict) in enumerate(tries, start=1):
        rec.on_attempt_start({"attempt": 1, "n_attempts": 1, "try": k,
                              "bin": "+x", "t_start": time.time(),
                              "max_steps": 80, "session_id": "T5"})
        _drive(rec, perc, fused, rng, 3 + k, names, pause=0.02)
        rec.on_attempt_done({"attempt": 1, "bin": "+x", "ending": ending,
                             "verdict": verdict, "steps": 3 + k,
                             "elapsed_s": 1.0, "session_id": "T5",
                             "n_attempts": 1, "try": k, "t_start": 0.0,
                             "t_end": 1.0, "froze": False, "max_steps": 80,
                             "t_verdict": 1.0})
    rec.on_session_end({"session_id": "T5", "attempted": 1, "reason": "test"})
    rec.close()

    session = RD.load_session(root)
    rows = session["attempts"]
    dirs = sorted(d.name for d in root.glob("attempt_*") if d.is_dir())
    print(f"  4 tries of one bin -> {len(rows)} rows in {len(dirs)} dirs: "
          + ", ".join(dirs))
    assert len(dirs) == 4, (
        f"four tries of +x left {len(dirs)} director(ies) {dirs} — the later "
        "tries overwrote the earlier ones, so the rows in attempts.csv point "
        "at recordings that are no longer theirs")
    assert len(rows) == 4, f"{len(rows)} rows read back, expected 4"

    # Each row must own its directory, and own the RIGHT one: the step count is
    # unique per try here, so a crossed mapping cannot pass by coincidence.
    seen = set()
    for row in rows:
        name = row["dir_name"]
        assert name not in seen, f"{name} is claimed by two rows"
        seen.add(name)
        want = 3 + int(row["try"])
        got = len(session["steps"][name])
        assert got == want, (
            f"try {row['try']} says {want} steps but {name} holds {got} — the "
            "row and the recording under it are not the same attempt")
    print("  every row owns its own directory, matched by step count")

    # And a plain session still numbers the way every session on disk already
    # does, or this fix silently renames history.
    plain = root.parent / (root.name + "_plain")
    rec = R.ExpRecorder(plain, rigs, extra_fields=fields, arrays_spec=spec,
                        manifest=R.build_manifest(_args(), rigs, None, "T6"))
    for i in (1, 2, 3):
        rec.on_attempt_start({"attempt": i, "n_attempts": 3, "try": 1,
                              "bin": "+x", "t_start": time.time(),
                              "max_steps": 80, "session_id": "T6"})
        _drive(rec, perc, fused, rng, 3, names, pause=0.02)
        rec.on_attempt_done({"attempt": i, "bin": "+x", "ending": "close",
                             "verdict": "pass", "steps": 3, "elapsed_s": 1.0,
                             "session_id": "T6", "n_attempts": 3, "try": 1,
                             "t_start": 0.0, "t_end": 1.0, "froze": False,
                             "max_steps": 80, "t_verdict": 1.0})
    rec.on_session_end({"session_id": "T6", "attempted": 3, "reason": "test"})
    rec.close()
    for row in RD.load_session(plain)["attempts"]:
        n = int(row["attempt"])
        assert row["dir_name"] == f"attempt_{n - 1:03d}", (
            f"attempt {n} landed in {row['dir_name']} — with no retries "
            "the numbering must still be the bin index minus one, or every "
            "session already on disk is renamed by this change")
    print("  a session with no retries numbers exactly as it always did")


def _reader_refuses_a_torn_session(root: Path) -> None:
    """A killed attempt must not read back as a smaller, valid session."""
    names = ["tripod"]
    rigs = [_rig("tripod")]
    rng = np.random.default_rng(3)
    perc, fused = _Perc(names, rng), _fused(rng, names)
    fields, spec = R.adapter_spec(types.SimpleNamespace())
    rec = R.ExpRecorder(root, rigs, extra_fields=fields, arrays_spec=spec,
                        manifest=R.build_manifest(_args(), rigs, None, "T3"))
    for i in (1, 2):
        rec.on_attempt_start({"attempt": i, "n_attempts": 2, "try": 1,
                              "bin": "+x", "t_start": time.time(),
                              "max_steps": 50, "session_id": "T3"})
        _drive(rec, perc, fused, rng, 4, names, pause=0.02)
        rec.on_attempt_done({"attempt": i, "bin": "+x", "ending": "close",
                             "verdict": "pass", "steps": 4, "elapsed_s": 1.0,
                             "session_id": "T3", "n_attempts": 2, "try": 1,
                             "t_start": 0.0, "t_end": 1.0, "froze": False,
                             "max_steps": 50, "t_verdict": 1.0})
    rec.on_session_end({"session_id": "T3", "attempted": 2, "reason": "test"})
    rec.close()
    assert len(RD.load_session(root)["attempts"]) == 2

    # Now do to it what a SIGKILL mid-attempt does: files on disk, no summary
    # row, attempt.json still saying complete=false.
    import csv as _csv
    rows = list(_csv.DictReader((root / "attempts.csv").open()))
    with (root / "attempts.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=R.ATTEMPT_FIELDS,
                            extrasaction="ignore")
        w.writeheader()
        w.writerow(rows[0])
    aj = root / "attempt_001" / "attempt.json"
    d = json.loads(aj.read_text())
    d["complete"] = False
    aj.write_text(json.dumps(d))

    try:
        RD.load_session(root)
    except SystemExit as e:
        assert "attempt_001" in str(e), str(e)
        print("  a torn session is refused, naming the attempt: "
              + str(e).splitlines()[1].strip())
        return
    raise AssertionError(
        "load_session accepted a session with an attempt on disk that has no "
        "summary row — it would silently have used 1 attempt of 2")


def _backpressure_sheds_heavy_not_rows(root: Path) -> None:
    """A stalled writer must shed FRAMES, never block, and never lose a row."""
    names = ["tripod"]
    rigs = [_rig("tripod")]
    rng = np.random.default_rng(1)
    perc, fused = _Perc(names, rng), _fused(rng, names)
    fields, spec = R.adapter_spec(types.SimpleNamespace())
    rec = R.ExpRecorder(root, rigs, extra_fields=fields, arrays_spec=spec,
                        manifest=R.build_manifest(_args(), rigs, None, "T1"))
    rec.on_attempt_start({"attempt": 1, "n_attempts": 1, "try": 1, "bin": "+x",
                          "t_start": time.time(), "max_steps": 50,
                          "session_id": "T1"})

    # Stall the writer by making every message take a quarter of a second — the
    # stand-in for a disk that has stopped answering. NOT by holding the queue's
    # own mutex: `qsize()` takes that same lock, so the producer would deadlock
    # inside its own backpressure check instead of shedding.
    N = 40
    inner = rec._writer._dispatch

    def _slow(msg):
        time.sleep(0.25)
        return inner(msg)

    rec._writer._dispatch = _slow
    t0 = time.time()
    _drive(rec, perc, fused, rng, N, names)
    producer_ms = (time.time() - t0) * 1e3 / N
    rec._writer._dispatch = inner

    rec.on_attempt_done({"attempt": 1, "bin": "+x", "ending": "timeout",
                         "verdict": "fail", "steps": N, "elapsed_s": 4.0,
                         "session_id": "T1", "n_attempts": 1, "try": 1,
                         "t_start": 0.0, "t_end": 4.0, "froze": True,
                         "max_steps": 50, "t_verdict": 4.0})
    rec.on_session_end({"session_id": "T1", "attempted": 1, "reason": "test"})
    rec.close()

    import csv as _csv
    a = root / "attempt_000"
    rows = list(_csv.DictReader((a / "steps.csv").open()))
    shed = [r for r in rows if int(r["heavy_idx"]) == -1]
    assert len(rows) == N, f"{len(rows)} rows survived of {N} — a row was LOST"
    assert rec.dropped > 0 and len(shed) == rec.dropped, (
        f"shedding not recorded: {rec.dropped} counted, {len(shed)} marked")
    d = np.memmap(a / "depth_tripod.u16", dtype="<u2",
                  mode="r").reshape(-1, 480, 640)
    assert len(d) == N - rec.dropped, (
        f"{len(d)} depth records for {N} steps with {rec.dropped} shed")
    pcs = np.memmap(a / "policy_pc.f4", dtype="<f4", mode="r").reshape(-1, 1024, 5)
    assert len(pcs) == N, "the LIGHT payload must never be shed"
    assert producer_ms < 5.0, (
        f"the producer took {producer_ms:.2f} ms/step against a stalled writer "
        "— it is blocking, which would stall the control loop")
    print(f"  producer {producer_ms:.3f} ms/step against a fully stalled "
          f"writer; shed {rec.dropped}/{N} frames, lost 0 rows, kept every cloud")


def _producer_cost(root: Path) -> None:
    """The number the whole design is justified by."""
    names = ["wrist", "tripod"]
    rigs = [_rig("wrist", "eye_in_hand"), _rig("tripod")]
    rng = np.random.default_rng(2)
    perc, fused = _Perc(names, rng), _fused(rng, names)
    fields, spec = R.adapter_spec(types.SimpleNamespace())
    rec = R.ExpRecorder(root, rigs, extra_fields=fields, arrays_spec=spec,
                        manifest=R.build_manifest(_args(), rigs, None, "T2"))
    rec.on_attempt_start({"attempt": 1, "n_attempts": 1, "try": 1, "bin": "+x",
                          "t_start": time.time(), "max_steps": 50,
                          "session_id": "T2"})
    N = 120
    perc.frame(0)
    T = np.eye(4)
    pc = rng.normal(size=(1024, 5)).astype(np.float32)
    rs_ = rng.normal(size=32).astype(np.float32)
    ac = rng.normal(size=7).astype(np.float32)
    t0 = time.time()
    for k in range(N):
        rec.on_step(step=k, t_mono=time.time(), perception=perc, fused=fused,
                    T_base_hand=T, gripper_norm=1.0, pc=pc, robot_state=rs_,
                    action=ac, have_obs=True, armed=True, executed=True,
                    grasp_close=False, clamped=False, ms_obs=40.0, ms_pol=9.0,
                    adapter=types.SimpleNamespace(), d_world=[1.0, 0.0, 0.0])
        time.sleep(0.02)           # 50 Hz: well above the loop's 6.7 Hz,
                                   # and slow enough that nothing is shed, so
                                   # this measures the RECORDING path
    per_ms = (time.time() - t0) * 1e3 / N - 20.0
    rec.on_attempt_done({"attempt": 1, "bin": "+x", "ending": "close",
                         "verdict": "pass", "steps": N, "elapsed_s": 1.0,
                         "session_id": "T2", "n_attempts": 1, "try": 1,
                         "t_start": 0.0, "t_end": 1.0, "froze": False,
                         "max_steps": 50, "t_verdict": 1.0})
    rec.on_session_end({"session_id": "T2", "attempted": 1, "reason": "test"})
    rec.close()
    print(f"  control-thread cost {per_ms:.3f} ms/step, 2 cameras "
          f"({per_ms / 150.0 * 100:.2f}% of a 150 ms period), "
          f"{rec.dropped} dropped")
    assert rec.dropped == 0, (
        f"{rec.dropped} frames shed at 50 Hz — the writer cannot keep up even "
        "at 7x the control rate, so this is not measuring the recording path")
    assert per_ms < 5.0, (
        f"{per_ms:.2f} ms/step on the control thread is too much — the whole "
        "point of the writer thread is that this stays under a millisecond")


def _pc_stride_is_five_and_enforced(root: Path) -> None:
    """The declared record shape must be the one the writer actually uses.

    THIS COST A CORRUPTED STREAM. `policy_pc.f4` records
    `build_policy_cloud`'s output, which is [1024, 5] for every policy that
    exists — but the spec used to take its channel count from the CHECKPOINT's
    `DATA.pc_channels`, which is 7 for a regrasp run. So the manifest said 7,
    every usable frame wrote 5, and every HELD frame NaN-filled at 7, because
    `_arrays` fills to the declared shape. One recorded attempt came out with
    50 records of one width and 70 of another in the same headerless file.

    That is the one corruption fixed-stride records cannot survive. A truncated
    tail is fine — `size // record_bytes` finds every earlier record, which is
    why the format has no header. A short record in the MIDDLE makes every
    record after it unlocatable, and nothing about the file looks wrong: the
    attempt above read back as 148 clean records of a 120-step attempt.

    So two things are asserted. The spec is 5 wide no matter what the adapter
    says, and a record of the wrong width is REFUSED rather than written — a
    loud failure on one attempt beats a stream that decodes to noise.
    """
    fields, spec = R.adapter_spec(types.SimpleNamespace(pc_channels=7))
    got = tuple(spec["policy_pc.f4"]["shape"])
    assert got == (1024, 5), (
        f"policy_pc.f4 is declared {got}; build_policy_cloud emits [1024, 5] "
        "for every policy, and a 7-channel cloud is a separate stream")

    # And the NaN fill for a held frame uses that same width, which is the half
    # of it that actually broke.
    rec_fields = R.step_fields(["tripod"], fields)
    sink = R._Sink(root, ["tripod"], rec_fields, False, 10.0, spec)
    try:
        sink.write_array("policy_pc.f4", np.zeros((1024, 5), np.float32))
        try:
            sink.write_array("policy_pc.f4", np.zeros((1024, 7), np.float32))
        except ValueError as exc:
            assert "unlocatable" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a [1024, 7] record was accepted into a [1024, 5] stream — "
                "every later record in that file is now unfindable")
        assert (root / "policy_pc.f4").stat().st_size == 1024 * 5 * 4, (
            "the refused record still reached the file")
    finally:
        sink.close()
    print("  policy_pc.f4 is [1024, 5] whatever the adapter says, and a "
          "mismatched record is refused")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="exp_rec_test_"))
    try:
        print("round trip and stream lengths")
        _round_trip_and_lengths(root / "s0")
        print("\nrecord strides")
        _pc_stride_is_five_and_enforced(root / "s4")
        print("\nretried bin")
        _a_retried_bin_keeps_both_recordings(root / "s5")
        print("\ntorn session")
        _reader_refuses_a_torn_session(root / "s3")
        print("\nbackpressure")
        _backpressure_sheds_heavy_not_rows(root / "s1")
        print("\ncontrol-thread cost")
        _producer_cost(root / "s2")
        print("\nall exp_recorder checks passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
