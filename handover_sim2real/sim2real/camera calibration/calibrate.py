#!/usr/bin/env python3
"""Eye-to-hand: solve the fixed camera's pose in the robot base frame.

    python calibrate.py --session 2026-08-11

Reads <session>/{color_intrinsics.json, robot_poses.json, images/} and writes
<session>/T_base_color.npy. Run validate_calibration.py afterwards — a solve
always returns something, and only the residual says whether it is any good.

Board spec, solver and thresholds all live in calib_config.py; the SE(3),
ChArUco and hand-eye code lives in calib_common.py.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import calib_common as cc
import calib_config as cfg


def pose_diversity(T_list) -> tuple[float, float, float]:
    """min / median / max pairwise rotation, in degrees.

    Reported because it is the precondition people forget: AX=XB determines X's
    rotation only from relative rotations, so a set of near-parallel poses is
    ill-conditioned no matter how many captures it holds.
    """
    angs = []
    for i in range(len(T_list)):
        for j in range(i + 1, len(T_list)):
            rel = Rot.from_matrix(T_list[i][:3, :3]).inv() * Rot.from_matrix(T_list[j][:3, :3])
            angs.append(np.degrees(np.linalg.norm(rel.as_rotvec())))
    if not angs:
        return 0.0, 0.0, 0.0
    a = np.asarray(angs)
    return float(a.min()), float(np.median(a)), float(a.max())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_session_arg(p)
    p.add_argument("--method", type=str, default=cfg.HAND_EYE_METHOD,
                   help=f"hand-eye solver (default {cfg.HAND_EYE_METHOD})")
    p.add_argument("--compare", action="store_true",
                   help="solve with every method and report each one's residual")
    args = p.parse_args()

    session = cc.resolve_session(args.session)
    names, poses, results, _, _ = cc.load_session_samples(session)

    if len(names) < cfg.MIN_SAMPLES:
        raise SystemExit(
            f"Only {len(names)} usable captures (need {cfg.MIN_SAMPLES}). "
            "Collect more, with the board fully visible.")

    lo, mid, hi = pose_diversity(poses)
    print(f"pairwise rotation between poses: min={lo:.1f} median={mid:.1f} max={hi:.1f} deg")
    if mid < 20.0:
        print("[warn] low rotation diversity — rotate the wrist more between captures; "
              "translation alone cannot determine the camera's orientation.")

    cams = [r["T_cam_board"] for r in results]

    tilts = np.array([cc.board_tilt_deg(C) for C in cams])
    flat = int((tilts < cfg.GOOD_BOARD_TILT_DEG).sum())
    print(f"board tilt: min={tilts.min():.0f} median={np.median(tilts):.0f} "
          f"max={tilts.max():.0f} deg  ({flat}/{len(tilts)} below "
          f"{cfg.GOOD_BOARD_TILT_DEG:.0f})")
    if flat > len(tilts) / 2:
        print("[warn] most captures are near square-on to the camera. A planar board "
              "viewed head-on barely constrains its own out-of-plane rotation, which "
              "shows up directly as rotation residual — angle the board more.")

    def residual(X) -> tuple[float, float]:
        """RMS spread of T_gripper_board, which must be constant across captures."""
        G = [cc.invert_T(P) @ X @ C for P, C in zip(poses, cams)]
        t = np.array([g[:3, 3] for g in G])
        R_ref = cc.average_rotations([g[:3, :3] for g in G])
        dt = np.linalg.norm(t - t.mean(0), axis=1)
        dr = [cc.rotation_angle_deg(R_ref.T @ g[:3, :3]) for g in G]
        return float(np.sqrt((dt ** 2).mean()) * 1000), float(np.sqrt(np.mean(np.square(dr))))

    if args.compare:
        print("\nsolver comparison (T_gripper_board consistency, lower is better):")
        for name in ("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"):
            try:
                X = cc.solve_eye_to_hand(poses, cams, method=name)
            except Exception as e:
                print(f"  {name:11s} failed: {e}")
                continue
            mm, deg = residual(X)
            print(f"  {name:11s} pos RMS={mm:6.2f} mm  rot RMS={deg:5.2f} deg  "
                  f"cam={np.round(X[:3, 3], 4)}")
        print()

    T_base_color = cc.solve_eye_to_hand(poses, cams, method=args.method)
    mm, deg = residual(T_base_color)

    print(f"\nmethod: {args.method}")
    print("T_base_color =")
    print(np.array_str(T_base_color, precision=6, suppress_small=True))
    print(f"\ncamera position in base frame: {np.round(T_base_color[:3, 3], 4)} m")
    print(f"T_gripper_board consistency  : {mm:.2f} mm / {deg:.2f} deg RMS")

    np.save(session.T_base_color, T_base_color)
    print(f"\nwrote {session.T_base_color}")
    print("now run:  python validate_calibration.py --session " + session.name)


if __name__ == "__main__":
    main()
