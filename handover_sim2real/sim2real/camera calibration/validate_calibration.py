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

    # How far off-centre each board sat, as a fraction of the image half-diagonal.
    # Reported because unmodelled edge distortion is absorbed as board TILT and
    # shows up here as rotation residual while reprojection stays sub-pixel —
    # see GOOD_BOARD_RADIUS_FRAC.
    half_diag = 0.5 * float(np.hypot(cfg.STREAM.width, cfg.STREAM.height))
    centre = np.array([K[0, 2], K[1, 2]])
    rad_frac = np.array([
        float(np.linalg.norm(r["img_points"].reshape(-1, 2).mean(0) - centre))
        / half_diag for r in results])

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

    # WHAT THE NUMBERS MEAN WHERE THE ROBOT WORKS, because two of the three are
    # in units that do not compare across cameras and the thresholds were set on
    # one.
    #
    # Rotation is the one that matters. It is the only term that GROWS with
    # distance, and being a fixed bias it does not average away over the points
    # in a cloud the way depth noise does. Translation is a constant offset
    # everywhere and is usually the smaller of the two by the time you are a
    # metre out.
    #
    # Reprojection is in PIXELS, which is not a property of the calibration
    # alone — it is the calibration seen through this camera's focal length. A
    # wide-FOV body has coarser pixels, so the same physical error prints as a
    # smaller number, and a fixed px threshold silently means something
    # different on every camera. Converted here so it can be compared with the
    # other two, and with a previous session on a different body.
    #
    # It is also the only row that can see the INTRINSICS: trans and rot are
    # both consistency checks on T_gripper_board and are blind to fx, fy, cx, cy
    # and the distortion model, all of which are factory values here. So
    # reprojection failing while the other two pass points at the intrinsics or
    # at the chain as a whole, not at the hand-eye solve.
    fx = float(K[0, 0])
    print(f"\nat the working distance (fx = {fx:.1f} px):")
    print(f"{'':22s}{'0.6 m':>9s}{'1.0 m':>9s}{'1.5 m':>9s}")
    for label, val in (("rotation", np.deg2rad(rot_deg.mean())),
                       ("reprojection", reproj_px.mean() / fx),
                       ("translation", None)):
        if val is None:
            print(f"{label:22s}{trans_mm.mean():9.1f}{trans_mm.mean():9.1f}"
                  f"{trans_mm.mean():9.1f}   mm (constant)")
        else:
            print(f"{label:22s}{val * 600:9.1f}{val * 1000:9.1f}"
                  f"{val * 1500:9.1f}   mm")

    print("\nper image:")
    for n, te, re_, pe in zip(names, trans_mm, rot_deg, reproj_px):
        print(f"  {n}: trans={te:6.3f} mm  rot={re_:5.3f} deg  reproj={pe:6.3f} px"
              f"  r={rad_frac[names.index(n)]:.2f}")

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
        off = rad_frac > cfg.GOOD_BOARD_RADIUS_FRAC
        if off.any() and (~off).any():
            print(f"{off.sum()}/{len(rad_frac)} captures had the board past "
                  f"r={cfg.GOOD_BOARD_RADIUS_FRAC} of the half-diagonal; those "
                  f"average {rot_deg[off].mean():.3f} deg rotation residual "
                  f"against {rot_deg[~off].mean():.3f} deg for the central ones. "
                  "Unmodelled edge distortion is absorbed as board tilt, so it "
                  "lands in rotation while reprojection stays sub-pixel — "
                  "recapture nearer the middle of the frame before adding poses.")
        print("Usual causes: too few poses, insufficient rotation diversity, a "
              "mis-measured square_length_m in calib_config.py, motion blur, or a "
              "board that shifted during capture.")
        return 1
    print("\nPASS — within calib_config thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
