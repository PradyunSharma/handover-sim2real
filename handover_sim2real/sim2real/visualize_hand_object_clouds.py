from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from torchvision import transforms

from camera import RealSenseCamera
from pointcloud_pipeline import extract_hand_object_clouds
from transforms import transform_points

HANDS_REPO = Path.home() / "h2r" / "hands-segmentation-pytorch"
CHECKPOINT = Path.home() / "h2r" / "handover-rs" / "checkpoint" / "checkpoint.ckpt"

sys.path.insert(0, str(HANDS_REPO))
from model import HandSegModel  # noqa: E402


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)


def make_pcd(points_xyz: np.ndarray, color_rgb: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if len(points_xyz) == 0:
        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        return pcd

    pts = points_xyz.astype(np.float64, copy=False)
    cols = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (len(pts), 1))
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def build_crop_cloud(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    cam: RealSenseCamera,
    hand_center: np.ndarray | None,
    crop_radius_m: float,
    min_depth_m: float,
    max_depth_m: float,
    full_cloud_stride: int,
) -> np.ndarray:
    if hand_center is None:
        return np.zeros((0, 3), dtype=np.float32)

    full_points_xyz, _, _ = cam.depth_to_pointcloud(
        depth_m=depth_m,
        color_bgr=color_bgr,
        mask=None,
        stride=full_cloud_stride,
        min_depth=min_depth_m,
        max_depth=max_depth_m,
    )

    if len(full_points_xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    dist = np.linalg.norm(full_points_xyz - hand_center[None, :], axis=1)
    keep = dist < crop_radius_m
    return full_points_xyz[keep].astype(np.float32, copy=False)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Hands repo:", HANDS_REPO)
    print("Checkpoint:", CHECKPOINT)

    if not HANDS_REPO.exists():
        raise FileNotFoundError(f"Hands repo not found: {HANDS_REPO}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = HandSegModel.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
    )
    model = model.to(device)
    model.eval()
    print("Model device:", next(model.parameters()).device)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Lower resolution keeps the visualizer responsive.
    cam = RealSenseCamera(color_size=(424, 240), depth_size=(424, 240), fps=30)
    cam.start()

    # Identity placeholder. Keep this in place so later you can replace it with
    # a real camera-optical-frame -> policy-frame transform if needed.
    T_policy_cam = np.eye(4, dtype=np.float32)

    crop_radius_m = 0.12
    object_max_radius_m = 0.10
    min_depth_m = 0.10
    max_depth_m = 1.50
    full_cloud_stride = 2
    hand_cloud_stride = 1
    min_hand_points = 100
    min_object_points = 80
    viz_update_hz = 8.0

    last_hand_xyz = None
    last_object_xyz = None
    last_update = 0.0

    vis = o3d.visualization.Visualizer()
    vis.create_window("Hand / Object Point Clouds", width=1280, height=720)

    full_pcd = o3d.geometry.PointCloud()
    crop_pcd = o3d.geometry.PointCloud()
    hand_pcd = o3d.geometry.PointCloud()
    obj_pcd = o3d.geometry.PointCloud()

    # vis.add_geometry(full_pcd)
    # vis.add_geometry(crop_pcd)
    # vis.add_geometry(hand_pcd)
    # vis.add_geometry(obj_pcd)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0.15, 0.15, 0.15])
    render_opt.point_size = 3.0

    alpha = 0.45
    geometries_added = False

    try:
        while True:
            color_bgr, depth_m, _ = cam.get_frames()
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            x = preprocess(color_rgb).unsqueeze(0).to(device, non_blocking=True)

            with torch.inference_mode():
                logits = model(x)
                pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            mask = cv2.resize(
                pred,
                (color_bgr.shape[1], color_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask > 0).astype(np.uint8)
            mask = largest_component(mask)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            result = extract_hand_object_clouds(
                color_bgr=color_bgr,
                depth_m=depth_m,
                hand_mask=mask,
                cam=cam,
                last_hand_xyz=last_hand_xyz,
                last_object_xyz=last_object_xyz,
                crop_radius_m=crop_radius_m,
                object_max_radius_m=object_max_radius_m,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                full_cloud_stride=full_cloud_stride,
                hand_cloud_stride=hand_cloud_stride,
                min_hand_points=min_hand_points,
                min_object_points=min_object_points,
            )

            hand_xyz = result.hand_xyz
            object_xyz = result.object_xyz
            hand_center = result.hand_center

            if len(hand_xyz) > 0:
                last_hand_xyz = hand_xyz
            if len(object_xyz) > 0:
                last_object_xyz = object_xyz

            crop_xyz = build_crop_cloud(
                color_bgr=color_bgr,
                depth_m=depth_m,
                cam=cam,
                hand_center=hand_center,
                crop_radius_m=crop_radius_m,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                full_cloud_stride=full_cloud_stride,
            )

            full_xyz, _, _ = cam.depth_to_pointcloud(
                depth_m=depth_m,
                color_bgr=color_bgr,
                mask=None,
                stride=4,
                min_depth=min_depth_m,
                max_depth=max_depth_m,
            )

            # Placeholder transform. Identity for now.
            full_xyz_policy = transform_points(T_policy_cam, full_xyz)
            crop_xyz_policy = transform_points(T_policy_cam, crop_xyz)
            hand_xyz_policy = transform_points(T_policy_cam, hand_xyz)
            object_xyz_policy = transform_points(T_policy_cam, object_xyz)

            now = time.time()
            if now - last_update >= 1.0 / viz_update_hz:
                full_new = make_pcd(full_xyz_policy, (0.8, 0.8, 0.8))   # brighter gray
                crop_new = make_pcd(crop_xyz_policy, (0.1, 0.5, 1.0))
                hand_new = make_pcd(hand_xyz_policy, (0.0, 1.0, 0.0))
                obj_new = make_pcd(object_xyz_policy, (1.0, 0.0, 0.0))

                full_pcd.points = full_new.points
                full_pcd.colors = full_new.colors

                crop_pcd.points = crop_new.points
                crop_pcd.colors = crop_new.colors

                hand_pcd.points = hand_new.points
                hand_pcd.colors = hand_new.colors

                obj_pcd.points = obj_new.points
                obj_pcd.colors = obj_new.colors

                total_pts = (
                    len(full_xyz_policy)
                    + len(crop_xyz_policy)
                    + len(hand_xyz_policy)
                    + len(object_xyz_policy)
                )

                if (not geometries_added) and total_pts > 0:
                    vis.add_geometry(full_pcd, reset_bounding_box=True)
                    vis.add_geometry(crop_pcd, reset_bounding_box=False)
                    vis.add_geometry(hand_pcd, reset_bounding_box=False)
                    vis.add_geometry(obj_pcd, reset_bounding_box=False)

                    ctr = vis.get_view_control()
                    ctr.set_zoom(0.7)

                    geometries_added = True
                elif geometries_added:
                    vis.update_geometry(full_pcd)
                    vis.update_geometry(crop_pcd)
                    vis.update_geometry(hand_pcd)
                    vis.update_geometry(obj_pcd)

                last_update = now

            print(
                f"full={len(full_xyz_policy)} crop={len(crop_xyz_policy)} "
                f"hand={len(hand_xyz_policy)} obj={len(object_xyz_policy)}",
                end="\r",
                flush=True,
            )

            overlay = color_bgr.copy()
            hand_color = np.zeros_like(color_bgr)
            hand_color[:, :, 1] = 255
            blended = (alpha * hand_color + (1 - alpha) * overlay).astype(np.uint8)
            overlay = np.where(mask[..., None] > 0, blended, overlay)

            dbg = result.debug
            text = (
                f"hand={len(hand_xyz_policy)} obj={len(object_xyz_policy)} "
                f"crop={len(crop_xyz_policy)} "
                f"fallback_h={dbg['used_last_hand']} fallback_o={dbg['used_last_object']}"
            )
            cv2.putText(
                overlay,
                text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("overlay", overlay)
            cv2.imshow("hand_mask", (mask * 255).astype(np.uint8))

            if not vis.poll_events():
                break
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == "__main__":
    main()