from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay RealSense .bag and export full point-cloud video"
    )
    parser.add_argument("--input", type=str, required=True, help="Input .bag file")
    parser.add_argument(
        "--output",
        type=str,
        default="pointcloud_output.mp4",
        help="Output visualization video path",
    )
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video fps")
    parser.add_argument("--stride", type=int, default=2, help="Pixel stride for deprojection")
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=1.50)
    return parser.parse_args()


def depth_to_xyz(
    depth_m: np.ndarray,
    intr: rs.intrinsics,
    color_bgr: np.ndarray | None = None,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 1.5,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = depth_m.shape
    valid = np.isfinite(depth_m)
    valid &= depth_m > min_depth
    valid &= depth_m < max_depth

    if stride > 1:
        subsample = np.zeros_like(valid, dtype=bool)
        subsample[::stride, ::stride] = True
        valid &= subsample

    v, u = np.nonzero(valid)
    z = depth_m[v, u].astype(np.float32)

    x = (u.astype(np.float32) - intr.ppx) * z / intr.fx
    y = (v.astype(np.float32) - intr.ppy) * z / intr.fy
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    rgb = None
    if color_bgr is not None:
        colors_bgr = color_bgr[v, u].astype(np.float32) / 255.0
        rgb = colors_bgr[:, ::-1].copy()

    return xyz, rgb


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(input_path), repeat_playback=False)

    print(f"Reading: {input_path}")
    print(f"Writing: {output_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="PointCloud Replay",
        width=args.width,
        height=args.height,
        visible=True,
    )

    pcd = o3d.geometry.PointCloud()
    added = False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        print("Failed to open video writer.", file=sys.stderr)
        vis.destroy_window()
        return 1

    try:
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        align = rs.align(rs.stream.color)

        # Grab one frame to get intrinsics.
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to get first color frame from bag.")
        intr = color_frame.profile.as_video_stream_profile().intrinsics

        render_opt = vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.background_color = np.asarray([0.0, 0.0, 0.0])

        initialized_view = False
        frame_count = 0

        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                # End of file
                break

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            depth_units = depth_frame.get_units()
            depth_m = depth_raw.astype(np.float32) * float(depth_units)

            xyz, rgb = depth_to_xyz(
                depth_m=depth_m,
                intr=intr,
                color_bgr=color_bgr,
                stride=args.stride,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )

            if len(xyz) == 0:
                continue

            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
            if rgb is not None:
                pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

            if not added:
                vis.add_geometry(pcd)
                added = True
            else:
                vis.update_geometry(pcd)

            if not initialized_view:
                ctr = vis.get_view_control()
                ctr.set_zoom(0.7)
                initialized_view = True

            vis.poll_events()
            vis.update_renderer()

            # Capture current Open3D frame buffer and write to video.
            img = vis.capture_screen_float_buffer(do_render=False)
            img = (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        writer.release()
        vis.destroy_window()

    print(f"Saved point-cloud video to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())