from __future__ import annotations

import time

import numpy as np
import open3d as o3d

from camera import RealSenseCamera


def main() -> None:
    # Tuning knobs for speed / stability
    update_hz = 30.0          # point cloud refresh rate
    stride = 2               # 1 = dense, 2/4 = lighter
    min_depth = 0.10         # meters
    max_depth = 1.50         # meters

    cam = RealSenseCamera(color_size=(640, 480), depth_size=(640, 480), fps=30)
    cam.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RealSense Full Point Cloud", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    added = False
    last_update = 0.0

    print("Open3D window started. Close the window or press Ctrl+C in terminal to quit.")

    try:
        while True:
            color_bgr, depth_m, _ = cam.get_frames()

            now = time.time()
            if now - last_update >= 1.0 / update_hz:
                points_xyz, colors_rgb, _ = cam.depth_to_pointcloud(
                    depth_m=depth_m,
                    color_bgr=color_bgr,
                    mask=None,            # full point cloud
                    stride=stride,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )

                if len(points_xyz) > 0:
                    # Open3D expects float64 arrays
                    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
                    if colors_rgb is not None:
                        pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))

                    if not added:
                        vis.add_geometry(pcd)
                        added = True

                        render_opt = vis.get_render_option()
                        render_opt.point_size = 2.0
                        render_opt.background_color = np.asarray([0.0, 0.0, 0.0])

                        ctr = vis.get_view_control()
                        ctr.set_zoom(0.7)
                    else:
                        vis.update_geometry(pcd)

                last_update = now

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        cam.stop()


if __name__ == "__main__":
    main()