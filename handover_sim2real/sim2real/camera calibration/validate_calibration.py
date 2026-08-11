#!/usr/bin/env python3
"""Score a calibration against the thresholds in calib_config.py.

    python validate_calibration.py --session 2026-08-11

The board is rigid on the wrist, so T_gripper_board must solve to the same pose
from every capture. That spread is the calibration's true error — the solve
itself always returns something. Also reprojects the board using the calibrated
camera and the averaged mount, giving a residual in pixels.

Writes <session>/T_gripper_board_ref.npy and exits non-zero if the calibration
misses the thresholds, so it can gate a script.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

import calib_common as cc
import calib_config as cfg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_session_arg(p)
    args = p.parse_args()

    session = cc.resolve_session(args.session)
    session.require(session.T_base_color)
    names, poses, results, K, D = cc.load_session_samples(session)
    if len(names) < 3:
        raise SystemExit(f"Too few valid samples: {len(names)}")

    T_base_color = np.load(session.T_base_color)

    # The mount, recovered independently from every capture. Never measured by
    # hand — it cancels out of AX=XB, and falls out as a by-product.
    T_gb = [cc.invert_T(P) @ T_base_color @ r["T_cam_board"]
            for P, r in zip(poses, results)]
    t_ref = np.array([g[:3, 3] for g in T_gb]).mean(axis=0)
    R_ref = cc.average_rotations([g[:3, :3] for g in T_gb])
    T_ref = cc.make_T(R_ref, t_ref)

    trans_mm, rot_deg, reproj_px = [], [], []
    for P, r, g in zip(poses, results, T_gb):
        trans_mm.append(1000.0 * np.linalg.norm(g[:3, 3] - t_ref))
        rot_deg.append(cc.rotation_angle_deg(R_ref.T @ g[:3, :3]))

        T_pred = cc.invert_T(T_base_color) @ P @ T_ref
        rvec, tvec = cc.T_to_rvec_tvec(T_pred)
        proj, _ = cv2.projectPoints(r["obj_points"], rvec, tvec, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - r["img_points"].reshape(-1, 2), axis=1)
        reproj_px.append(float(np.sqrt(np.mean(err ** 2))))

    trans_mm = np.array(trans_mm)
    rot_deg = np.array(rot_deg)
    reproj_px = np.array(reproj_px)

    print(f"\n=== session {session.name}: {len(names)} samples ===")
    print(f"{'':22s}{'mean':>9s}{'median':>9s}{'max':>9s}{'threshold':>12s}")
    rows = [
        ("gripper->board trans", trans_mm, "mm", cfg.MAX_TRANS_ERR_MM),
        ("gripper->board rot", rot_deg, "deg", cfg.MAX_ROT_ERR_DEG),
        ("reprojection RMSE", reproj_px, "px", cfg.MAX_REPROJ_PX),
    ]
    failed = []
    for label, arr, unit, thr in rows:
        flag = "" if arr.mean() <= thr else "   FAIL"
        if flag:
            failed.append(label)
        print(f"{label:22s}{arr.mean():9.3f}{np.median(arr):9.3f}{arr.max():9.3f}"
              f"{thr:9.2f} {unit}{flag}")

    print("\nper image:")
    for n, te, re_, pe in zip(names, trans_mm, rot_deg, reproj_px):
        print(f"  {n}: trans={te:6.3f} mm  rot={re_:5.3f} deg  reproj={pe:6.3f} px")

    worst = int(np.argmax(trans_mm))
    if trans_mm[worst] > 3 * np.median(trans_mm):
        print(f"\n[hint] {names[worst]} is a clear outlier ({trans_mm[worst]:.1f} mm vs "
              f"{np.median(trans_mm):.1f} median) — drop it and re-solve.")

    np.save(session.T_gripper_board, T_ref)
    print(f"\nwrote {session.T_gripper_board}")
    print(f"board mount on the wrist: {np.round(t_ref * 1000, 1)} mm "
          "(recovered, never measured)")

    if failed:
        print("\nFAILED: " + ", ".join(failed))
        print("Usual causes: too few poses, insufficient rotation diversity, a "
              "mis-measured square_length_m in calib_config.py, motion blur, or a "
              "board that shifted during capture.")
        return 1
    print("\nPASS — within calib_config thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
