import cv2
import numpy as np

from camera import RealSenseCamera


def main() -> None:
    cam = RealSenseCamera(color_size=(640, 480), depth_size=(640, 480), fps=30)
    cam.start()

    print(cam.get_intrinsics())

    try:
        while True:
            color_bgr, depth_m, ts = cam.get_frames()

            # Full point cloud, lightly subsampled for speed.
            points_xyz, _, _ = cam.depth_to_pointcloud(
                depth_m,
                color_bgr=color_bgr,
                stride=4,
                min_depth=0.10,
                max_depth=1.50,
            )

            depth_vis = depth_m.copy()
            depth_vis[~np.isfinite(depth_vis)] = 0
            depth_vis = np.clip(depth_vis, 0.0, 1.5)
            depth_vis = (depth_vis / 1.5 * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            text = f"pts={len(points_xyz)}  ts={ts:.1f} ms"
            cv2.putText(color_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("color", color_bgr)
            cv2.imshow("depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()