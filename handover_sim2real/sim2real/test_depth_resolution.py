#!/usr/bin/env python3
"""Does a higher camera resolution actually buy better depth? Measure it.

    python test_depth_resolution.py --list              # no capture, no setup
    python test_depth_resolution.py --serial 825312073923

`--list` needs nothing but the camera plugged in. It enumerates every depth and
colour mode the device offers and prints the focal length in PIXELS for each,
which is the number that decides whether a resolution change can help at all:

  * if f_px is the SAME at two widths, the smaller one is a CROP of the larger.
    Same angular resolution per pixel, same depth noise, just a narrower field
    of view. Switching between them cannot change depth accuracy.
  * if f_px SCALES with the width, the smaller one is a DOWNSCALE, and the
    larger really does resolve finer disparity.

The measurement mode answers the same question empirically, using Intel's own
methodology: point the camera at a flat wall, fit a plane to the middle of the
frame, and report the RMS of the residuals. That RMS is the depth noise.

WHY RMS ALONE WOULD MISLEAD, and what the last column is for. Depth noise from
stereo goes as

    sigma_z  =  z^2 * sigma_disparity / (f_px * baseline)

so a mode with a bigger f_px has a smaller sigma_z even if the sensor is doing
exactly as well per pixel. Comparing sigma_z across resolutions therefore
flatters the high-resolution mode by construction. The honest comparison is
sigma_disparity — the RMS converted back through that formula — which is the
resolution-independent figure of merit. If sigma_disparity is flat across modes,
the extra pixels are real. If it climbs with resolution, the ASIC is
interpolating and you are paying bandwidth for nothing.

That second case is the documented behaviour of the D435 above 848x480: depth is
generated at 848x480 and larger modes are extrapolated from it.

The scene matters more than anything this script can control. Use a flat,
untextured, matte wall filling the middle of the frame, at the distance you
actually deploy at, with the room lit the way it normally is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _hfov_deg(width: int, fx: float) -> float:
    return 2.0 * np.degrees(np.arctan(0.5 * width / fx))


def list_modes(serial: str | None) -> None:
    """Every mode the device offers, with the focal length it implies.

    Read from the stream profiles rather than started pipelines, so this costs
    nothing and cannot disturb a camera something else is using.
    """
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError("No RealSense device attached.")
    if serial is not None:
        devices = [d for d in devices
                   if d.get_info(rs.camera_info.serial_number) == str(serial)]
        if not devices:
            raise RuntimeError(f"No RealSense with serial {serial}.")

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        sn = dev.get_info(rs.camera_info.serial_number)
        print(f"\n{name}  serial {sn}")

        for sensor in dev.query_sensors():
            rows = {}
            for p in sensor.get_stream_profiles():
                if not p.is_video_stream_profile():
                    continue
                vp = p.as_video_stream_profile()
                if vp.stream_type() not in (rs.stream.depth, rs.stream.color):
                    continue
                # Same (stream, w, h) appears once per format and fps; the
                # intrinsics are identical across those, so keep one and
                # collect the frame rates.
                key = (str(vp.stream_type()).split(".")[-1], vp.width(), vp.height())
                intr = vp.get_intrinsics()
                rows.setdefault(key, [intr, set()])[1].add(vp.fps())

            if not rows:
                continue
            print(f"\n  {sensor.get_info(rs.camera_info.name)}")
            print(f"    {'stream':6s} {'mode':>10s} {'fx':>8s} {'fy':>8s} "
                  f"{'cx':>7s} {'cy':>7s} {'HFOV':>7s}  fps")
            for (stream, w, h), (intr, fpss) in sorted(
                    rows.items(), key=lambda kv: (kv[0][0], -kv[0][1], -kv[0][2])):
                print(f"    {stream:6s} {f'{w}x{h}':>10s} {intr.fx:8.1f} "
                      f"{intr.fy:8.1f} {intr.ppx:7.1f} {intr.ppy:7.1f} "
                      f"{_hfov_deg(w, intr.fx):6.1f}d  "
                      f"{','.join(str(f) for f in sorted(fpss))}")

    print("\nEqual fx at two widths => the narrower mode is a CROP: same depth "
          "noise,\nnarrower view. fx scaling with width => a real change in "
          "angular resolution.")


def _plane_rms(points: np.ndarray) -> tuple[float, np.ndarray]:
    """RMS distance to the best-fit plane, and its unit normal.

    Total-least-squares via SVD, i.e. the plane minimising perpendicular
    distance — not a z = f(x, y) regression, which would understate the error
    on a wall seen at an angle.
    """
    centred = points - points.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    resid = centred @ normal
    return float(np.sqrt(np.mean(resid ** 2))), normal


def measure(serial: str | None, modes: list[tuple[int, int]], fps: int,
            frames: int, roi_frac: float, baseline_m: float,
            equal_roi: bool = True) -> None:
    import pyrealsense2 as rs

    # EQUAL PIXEL FRACTIONS ARE THE WRONG COMPARISON, and quietly so. These
    # modes do not share a field of view — on a D435, 640x480 sees 79.6 deg
    # horizontally where 848x480 sees 90 — so the same central 20% of each frame
    # covers a DIFFERENT patch of wall: 0.72 m against 0.86 m at ~2.1 m. No real
    # wall is perfectly flat, and a smaller patch has less of whatever bow,
    # skirting or poster is on it, so the narrow-field mode wins a little for
    # free. That bias is invisible in the output and always favours the same
    # mode, which is the worst kind.
    #
    # So the ROI is pinned in ANGLE instead: roi * width / fx, held at whatever
    # the first mode implies. Every mode then fits its plane to the same cone of
    # the scene, and the only thing left varying is the sensor.
    ref_w, ref_h = modes[0]
    ref_angular = None
    if equal_roi:
        print("\nROI pinned in angle to the first mode, so every mode fits the "
              "same\npatch of wall. Pass --no-equal-roi for a fixed pixel "
              "fraction instead.")

    print(f"\n{frames} frames per mode, plane fit per frame.")
    print("Point at a flat matte wall at your working distance.\n")
    print(f"  {'mode':>10s} {'fps':>4s} {'fx':>7s} {'roi':>6s} {'patch':>7s} "
          f"{'range':>8s} {'fill':>6s} {'RMS':>8s} {'disparity':>10s}")
    print(f"  {'':>10s} {'':>4s} {'px':>7s} {'%':>6s} {'m':>7s} "
          f"{'m':>8s} {'%':>6s} {'mm':>8s} {'px RMS':>10s}")

    results = []
    for (w, h) in modes:
        pipeline = rs.pipeline()
        config = rs.config()
        if serial is not None:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        try:
            profile = pipeline.start(config)
        except Exception as exc:                      # unsupported mode
            print(f"  {f'{w}x{h}':>10s} {fps:>4d}  -- not available: "
                  f"{str(exc).splitlines()[0]}")
            continue

        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            scale = float(depth_sensor.get_depth_scale())
            vp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intr = vp.get_intrinsics()

            for _ in range(15):                       # auto-exposure settle
                pipeline.wait_for_frames()

            # Half-angle of the ROI, as a tangent: (roi * width / 2) / fx. Fix
            # it from the first mode and every later mode solves for the pixel
            # fraction that reproduces it.
            if equal_roi:
                if ref_angular is None:
                    ref_angular = roi_frac * ref_w / intr.fx
                    this_roi = roi_frac
                else:
                    this_roi = min(1.0, ref_angular * intr.fx / w)
            else:
                this_roi = roi_frac

            # Pixel grid for the ROI once, not per frame.
            x0, x1 = int(w * (0.5 - this_roi / 2)), int(w * (0.5 + this_roi / 2))
            y0, y1 = int(h * (0.5 - this_roi / 2)), int(h * (0.5 + this_roi / 2))
            uu, vv = np.meshgrid(np.arange(x0, x1, dtype=np.float64),
                                 np.arange(y0, y1, dtype=np.float64))

            rms_list, fill_list, z_list = [], [], []
            for _ in range(frames):
                f = pipeline.wait_for_frames().get_depth_frame()
                if not f:
                    continue
                z = (np.asanyarray(f.get_data())[y0:y1, x0:x1].astype(np.float64)
                     * scale)
                valid = z > 0
                fill_list.append(float(valid.mean()))
                if valid.sum() < 100:
                    continue
                zz = z[valid]
                pts = np.column_stack([
                    (uu[valid] - intr.ppx) / intr.fx * zz,
                    (vv[valid] - intr.ppy) / intr.fy * zz,
                    zz,
                ])
                rms, _ = _plane_rms(pts)
                rms_list.append(rms)
                z_list.append(float(np.median(zz)))

            if not rms_list:
                print(f"  {f'{w}x{h}':>10s} {fps:>4d}  -- no valid depth in the "
                      "ROI (too close, too dark, or pointed at nothing)")
                continue

            rms = float(np.mean(rms_list))
            z = float(np.mean(z_list))
            fill = float(np.mean(fill_list)) * 100.0
            # Invert sigma_z = z^2 * sigma_d / (f * B) to get the figure of
            # merit that does NOT improve for free with resolution.
            disp = rms * intr.fx * baseline_m / (z * z)
            patch = this_roi * w / intr.fx * z      # metres across, at range
            print(f"  {f'{w}x{h}':>10s} {fps:>4d} {intr.fx:7.1f} "
                  f"{this_roi * 100:6.1f} {patch:7.2f} {z:8.3f} "
                  f"{fill:6.1f} {rms * 1000:8.2f} {disp:10.3f}")
            results.append((w, h, intr.fx, z, rms, disp))
        finally:
            pipeline.stop()

    if len(results) < 2:
        return
    print("\nRMS is what you get. Disparity RMS is what the sensor is doing:")
    best = min(results, key=lambda r: r[5])
    for w, h, fx, z, rms, disp in results:
        tag = "  <- best per pixel" if disp == best[5] else ""
        print(f"  {w}x{h}: {rms * 1000:.2f} mm at {z:.2f} m, "
              f"{disp:.3f} px disparity{tag}")
    print("\nIf disparity RMS is flat, the extra pixels are real and the mm "
          "column is\nthe honest gain. If it rises with resolution, the larger "
          "mode is being\ninterpolated and you are paying bandwidth for noise.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="enumerate modes and their focal lengths, then exit. "
                        "Needs no wall and no setup.")
    p.add_argument("--serial", default=None,
                   help="which camera (see calib_config.CAMERA_SERIALS)")
    p.add_argument("--modes", default="640x480,848x480,1280x720",
                   help="comma-separated WxH depth modes to compare")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=30,
                   help="frames averaged per mode")
    p.add_argument("--roi", type=float, default=0.2,
                   help="central fraction of the FIRST mode's frame to fit the "
                        "plane to. Intel's guidance is a central ROI; the edges "
                        "carry lens and occlusion effects that are not depth "
                        "noise. Later modes match it in angle, not in pixels.")
    p.add_argument("--no-equal-roi", action="store_true",
                   help="use the same pixel fraction for every mode instead of "
                        "the same angular patch. These modes do not share a "
                        "field of view, so this makes narrow-field modes look "
                        "better by fitting a smaller patch of wall.")
    p.add_argument("--baseline", type=float, default=0.050,
                   help="stereo baseline in metres, 0.050 for the D435. Only "
                        "scales the disparity column; comparisons are unaffected.")
    args = p.parse_args()

    if args.list:
        list_modes(args.serial)
        return

    modes = []
    for tok in args.modes.split(","):
        w, _, h = tok.strip().partition("x")
        modes.append((int(w), int(h)))
    measure(args.serial, modes, args.fps, args.frames, args.roi, args.baseline,
            equal_roi=not args.no_equal_roi)


if __name__ == "__main__":
    main()
