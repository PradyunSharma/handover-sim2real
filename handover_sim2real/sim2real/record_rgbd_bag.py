from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record RealSense RGB-D to .bag")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .bag path. Default: recordings/<timestamp>.bag",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live RGB/depth preview while recording",
    )
    return parser.parse_args()


def make_output_path(user_path: str | None) -> Path:
    if user_path is not None:
        path = Path(user_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    out_dir = Path("recordings").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir / f"realsense_{timestamp}.bag"


def main() -> int:
    args = parse_args()
    output_path = make_output_path(args.output)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    # This is the important part: native RealSense recording.
    config.enable_record_to_file(str(output_path))

    print(f"Recording to: {output_path}")
    print("Press 'q' in the preview window or Ctrl+C in terminal to stop.")

    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale:.8f} meters/unit")

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            if args.preview:
                color_bgr = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())

                # Preview only. Do not use this color-mapped depth for reconstruction.
                depth_m = depth_raw.astype(np.float32) * depth_scale
                depth_vis = np.clip(depth_m, 0.0, 1.5)
                depth_vis = (depth_vis / 1.5 * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                cv2.imshow("color", color_bgr)
                cv2.imshow("depth_preview", depth_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error while recording: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    print(f"Saved recording: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())