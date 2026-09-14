#!/usr/bin/env python3
"""Read a session written by exp_recorder. numpy, cv2 and the stdlib, nothing else.

THIS FILE IS THE FORMAT'S TEST. If reconstructing a recorded frame needed
anything from this repo — an import, a config, a class definition — then the
session would only be readable from a working checkout of the code that wrote
it, which is not what "recorded" should mean. So the reconstruction below is
nineteen lines of arithmetic against `manifest.json`, and the only project fact
it relies on is that `camera.py` computes metric depth as
`raw.astype(np.float32) * depth_scale`. Doing it in that same order of
operations is what makes the result bit-identical to what the pipeline saw
rather than merely close to it.

    python read_exp_session.py <session_dir>              # summarise
    python read_exp_session.py <session_dir> --attempt 2 --frame 42

`load_session` REFUSES a session whose attempt directories and summary disagree.
An attempt killed mid-flight leaves its files on disk with no row in
attempts.csv, and a loader that trusted only the summary would quietly use three
attempts of four — which is the failure `regrasp/collector.py` grew its
`complete` flag to prevent, arriving by a different road.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_session(root) -> dict:
    """(manifest, attempts, per-attempt step tables). Raises on a torn session."""
    root = Path(root)
    man = json.loads((root / "manifest.json").read_text())
    rows = []
    ap = root / "attempts.csv"
    if ap.exists():
        rows = list(csv.DictReader(ap.open()))
    # TWO ROWS POINTING AT ONE DIRECTORY IS DATA LOSS, NOT A DUPLICATE KEY.
    # Built as a dict comprehension this silently kept the last row and dropped
    # the others, so a session recorded before attempt directories were named by
    # run order reads back as if the earlier tries never existed — which is
    # exactly the case where you most want to be told. The recorder cannot
    # produce this any more; sessions already on disk still can.
    by_dir: dict[str, dict] = {}
    collisions: dict[str, int] = {}
    for r in rows:
        name = r.get("dir_name")
        if not name:
            continue
        if name in by_dir:
            collisions[name] = collisions.get(name, 1) + 1
        by_dir[name] = r

    dirs = sorted(d for d in root.glob("attempt_*") if d.is_dir())
    problems = [f"{name}: {n} attempts.csv rows name this one directory, so "
                f"only the last of them still has its recording on disk"
                for name, n in sorted(collisions.items())]
    for d in dirs:
        r = by_dir.get(d.name)
        if r is None:
            problems.append(f"{d.name}: on disk but no row in attempts.csv")
            continue
        aj = d / "attempt.json"
        if not aj.exists():
            problems.append(f"{d.name}: no attempt.json")
        elif not json.loads(aj.read_text()).get("complete"):
            problems.append(f"{d.name}: attempt.json says complete=false")
    if problems:
        raise SystemExit(
            "this session does not hold what attempts.csv says it does — the "
            "process was killed mid-attempt, or it predates attempt "
            "directories being named by run order:\n  "
            + "\n  ".join(problems)
            + "\nDelete the offending attempt_* directories, or the rows that "
              "no longer have one, to read the rest.")
    if not man.get("complete"):
        print(f"[warn] {root}/manifest.json says complete=false: the session "
              "never shut down cleanly. Everything already written is still "
              "readable.")

    steps = {}
    for d in dirs:
        p = d / "steps.csv"
        steps[d.name] = list(csv.DictReader(p.open())) if p.exists() else []

    # EVERY BINARY MUST BE A WHOLE NUMBER OF RECORDS, AND THE SAME NUMBER AS
    # THE CSV HAS ROWS. Both halves are load-bearing and neither is paranoia.
    #
    # A truncated tail is expected and harmless — a hard kill loses the last
    # record and `size // record_bytes` recovers every earlier one, which is why
    # the format is headerless and fixed-stride in the first place. What is NOT
    # survivable is a record of the WRONG WIDTH in the middle, because from
    # there on no record can be located at all. That happened once, for real:
    # `policy_pc.f4` was declared [1024, 7] from the checkpoint's pc_channels
    # while the writer always produced [1024, 5], so held frames NaN-filled at
    # the declared width and usable ones did not. The file then looked like 148
    # clean records of a 120-step attempt and every one after the first held
    # frame was garbage — silently, which is the whole reason this check exists.
    # `_Sink.write_array` now refuses a mismatched record, so only sessions
    # recorded before that fix can show it.
    bad = []
    for d in dirs:
        n_rows = len(steps[d.name])
        for name, spec in (man.get("records") or {}).items():
            f = d / name
            if not f.exists():
                continue
            rec = int(np.dtype(spec["dtype"]).itemsize
                      * np.prod(spec["shape"]))
            size = f.stat().st_size
            n = size // rec
            if size % rec:
                bad.append(f"{d.name}/{name}: {size} B is not a whole number "
                           f"of {rec} B records ({size % rec} B over)")
            elif n_rows and n != n_rows:
                bad.append(f"{d.name}/{name}: {n} records for {n_rows} rows in "
                           "steps.csv — the stride does not match the manifest")
    if bad:
        raise SystemExit(
            "a binary stream does not match its declared record shape, so its "
            "records cannot be located:" + "".join("\n  " + b for b in bad)
            + "\nThe CSV, the labels, the depth and the other streams are "
              "unaffected; read those and ignore the stream named above.")
    return {"root": root, "manifest": man, "attempts": rows, "steps": steps}


def _array(att: Path, man: dict, name: str):
    """One of the fixed-stride binaries, as [N, *shape].

    A record count rather than a header, so a file torn by a kill still reads:
    every whole record survives and the remainder is arithmetically detectable.
    """
    spec = man["records"][name]
    p = att / name
    stride = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
    n, rem = divmod(p.stat().st_size, stride)
    if rem:
        print(f"[warn] {p.name}: {rem} trailing bytes — the last record was "
              f"torn by a kill; reading the {n} whole ones.")
    return np.memmap(p, dtype=spec["dtype"], mode="r",
                     shape=(n, *spec["shape"]))


def frame_cloud(session: dict, attempt: int, heavy_idx: int, camera=None):
    """The recorded frame, back in the ROBOT BASE frame. The nineteen lines."""
    import cv2

    root, man = session["root"], session["manifest"]
    att = root / f"attempt_{attempt:03d}"
    cam = next(c for c in man["cameras"]
               if camera is None or c["name"] == camera)
    K = cam["intrinsics"]
    H, W = K["height"], K["width"]

    row = next(r for r in session["steps"][att.name]
               if int(r["heavy_idx"]) == heavy_idx)
    step = int(row["step"])

    depth = np.memmap(att / f"depth_{cam['name']}.u16", dtype="<u2",
                      mode="r").reshape(-1, H, W)[heavy_idx]
    lab = cv2.imread(str(att / f"labels_{cam['name']}/{heavy_idx:06d}.png"),
                     cv2.IMREAD_UNCHANGED)
    T_bh = _array(att, man, "pose_base_hand.f8")[step]

    z = depth.astype(np.float32) * cam["depth_scale_m"]   # camera.py:220, exactly
    lo = cam["extraction_params"].get("min_depth_m", 0.1)
    hi = cam["extraction_params"].get("max_depth_m", 3.0)
    v, u = np.nonzero((z > lo) & (z < hi))
    zz = z[v, u]
    xyz_cam = np.stack([(u - K["cx"]) * zz / K["fx"],
                        (v - K["cy"]) * zz / K["fy"], zz], 1)
    T_bc = (np.asarray(cam["T_base_cam"]) if cam["kind"] == "fixed"
            else np.asarray(T_bh) @ np.asarray(cam["T_hand_cam"]))
    xyz_base = xyz_cam @ T_bc[:3, :3].T + T_bc[:3, 3]
    return {"xyz_base": xyz_base, "labels": lab[v, u], "row": row,
            "T_base_hand": np.asarray(T_bh), "camera": cam["name"],
            "policy_pc": _array(att, man, "policy_pc.f4")[step],
            "action": _array(att, man, "policy_action.f4")[step]}


def summarise(session: dict) -> None:
    man, rows = session["manifest"], session["attempts"]
    print(f"session {man.get('session_id')}  {man.get('runner')}  "
          f"{len(rows)} attempts")
    print(f"  cameras {', '.join(man.get('cameras_recorded', []))}"
          f"   depth {'on' if man.get('record_depth') else 'OFF'}"
          f"   control {man.get('control', {}).get('mode')}")
    if not rows:
        return
    print(f"  {'#':>3} {'bin':>3} {'ending':>11} {'verdict':>9} "
          f"{'secs':>7} {'steps':>6} {'frames':>7}")
    for r in rows:
        print(f"  {r['attempt']:>3} {r['bin']:>3} {r['ending']:>11} "
              f"{r['verdict']:>9} {float(r['elapsed_s']):>7.2f} "
              f"{r['steps']:>6} {r['n_frames']:>7}")
    ok = sum(1 for r in rows if r["verdict"] == "pass")
    judged = [r for r in rows if r["verdict"] in ("pass", "fail")]
    print(f"  {ok}/{len(judged)} passed"
          + (f" ({ok / len(judged) * 100:.0f}%)" if judged else ""))
    per = {}
    for r in judged:
        p, n = per.get(r["bin"], (0, 0))
        per[r["bin"]] = (p + (r["verdict"] == "pass"), n + 1)
    if per:
        print("  by command: " + "  ".join(f"{b} {p}/{n}"
                                           for b, (p, n) in sorted(per.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session")
    ap.add_argument("--attempt", type=int, default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--camera", default=None)
    a = ap.parse_args()

    s = load_session(a.session)
    summarise(s)
    if a.attempt is None:
        return
    out = frame_cloud(s, a.attempt - 1, a.frame, a.camera)
    xyz, lab = out["xyz_base"], out["labels"]
    print(f"\nattempt {a.attempt}, frame {a.frame}, camera {out['camera']}: "
          f"{len(xyz)} points in the base frame")
    print(f"  bbox x [{xyz[:, 0].min():+.3f} {xyz[:, 0].max():+.3f}]  "
          f"y [{xyz[:, 1].min():+.3f} {xyz[:, 1].max():+.3f}]  "
          f"z [{xyz[:, 2].min():+.3f} {xyz[:, 2].max():+.3f}]")
    print(f"  labels: {int((lab == 1).sum())} hand, {int((lab == 2).sum())} "
          f"object, {int((lab == 0).sum())} background")
    print(f"  action {np.round(out['action'], 4)}")


if __name__ == "__main__":
    main()
