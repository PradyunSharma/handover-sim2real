#!/usr/bin/env python3
"""Export the camera's colour intrinsics into a session.

    python generate_color_intrinsics.py --session 2026-08-11 --role tripod

Intrinsics are resolution-specific: whatever is exported here must match the
capture resolution and the resolution the runner streams at.
"""

from __future__ import annotations

import argparse
import json

import pyrealsense2 as rs

import calib_common as cc
import calib_config as cfg


def intrinsics_to_dict(intr) -> dict:
    return {
        "camera_matrix": [[float(intr.fx), 0.0, float(intr.ppx)],
                          [0.0, float(intr.fy), float(intr.ppy)],
                          [0.0, 0.0, 1.0]],
        "dist_coeffs": [float(x) for x in intr.coeffs[:5]],
        "width": int(intr.width),
        "height": int(intr.height),
        "model": str(intr.model),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cc.add_session_arg(p)
    cc.add_camera_args(p)
    p.add_argument("--stream", choices=["color", "depth"], default="color")
    args = p.parse_args()

    session = cc.resolve_session(args.session, create=True)
    serial = cc.resolve_serial(args.serial, args.role)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    stream = rs.stream.color if args.stream == "color" else rs.stream.depth
    fmt = rs.format.bgr8 if args.stream == "color" else rs.format.z16
    config.enable_stream(stream, cfg.STREAM.width, cfg.STREAM.height, fmt, cfg.STREAM.fps)

    profile = pipeline.start(config)
    try:
        intr = profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
        data = intrinsics_to_dict(intr)
        session.intrinsics_json.write_text(json.dumps(data, indent=2))
        print(f"wrote {session.intrinsics_json}")
        print(json.dumps(data, indent=2))
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
