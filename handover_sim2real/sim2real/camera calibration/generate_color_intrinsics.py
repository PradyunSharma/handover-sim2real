#!/usr/bin/env python3
import json
import argparse
import pyrealsense2 as rs

# run like this -> python export_realsense_intrinsics.py --stream color --width 640 --height 480 --fps 30 --output color_intrinsics.json


def intrinsics_to_dict(intr):
    return {
        "camera_matrix": [
            [float(intr.fx), 0.0, float(intr.ppx)],
            [0.0, float(intr.fy), float(intr.ppy)],
            [0.0, 0.0, 1.0],
        ],
        "dist_coeffs": [float(x) for x in intr.coeffs[:5]],
        "width": int(intr.width),
        "height": int(intr.height),
        "model": str(intr.model),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export RealSense intrinsics to color_intrinsics.json for calibrate.py"
    )
    parser.add_argument("--stream", choices=["color", "depth"], default="color",
                        help="Which stream intrinsics to export. Use color if checkerboard detection is on RGB images.")
    parser.add_argument("--width", type=int, default=640, help="Stream width")
    parser.add_argument("--height", type=int, default=480, help="Stream height")
    parser.add_argument("--fps", type=int, default=30, help="Stream fps")
    parser.add_argument("--output", type=str, default="color_intrinsics.json",
                        help="Output json file")
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()

    if args.stream == "color":
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    else:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    try:
        if args.stream == "color":
            vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        else:
            vsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        intr = vsp.get_intrinsics()
        data = intrinsics_to_dict(intr)

        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved intrinsics to: {args.output}")
        print(json.dumps(data, indent=2))

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()