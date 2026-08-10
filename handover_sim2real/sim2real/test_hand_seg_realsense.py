from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from camera import RealSenseCamera

HANDS_REPO = Path.home() / "h2r" / "hands-segmentation-pytorch"
CHECKPOINT = Path.home() / "h2r" / "handover-rs" / "checkpoint" / "checkpoint.ckpt"

sys.path.insert(0, str(HANDS_REPO))
from model import HandSegModel


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)


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

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    cam = RealSenseCamera(color_size=(640, 480), depth_size=(640, 480), fps=30)
    # cam = RealSenseCamera(color_size=(320, 240), depth_size=(320, 240), fps=30)
    cam.start()

    alpha = 0.45

    try:
        frame_idx = 0
        last_mask = None
        seg_every = 3
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

            # Convert to binary hand mask
            mask = (mask > 0).astype(np.uint8)

            # Cleanup
            mask = largest_component(mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            mask_vis = (mask * 255).astype(np.uint8)

            overlay = color_bgr.copy()
            hand_color = np.zeros_like(color_bgr)
            hand_color[:, :, 1] = 255
            blended = (alpha * hand_color + (1 - alpha) * overlay).astype(np.uint8)
            overlay = np.where(mask[..., None] > 0, blended, overlay)

            hand_points_xyz, _, _ = cam.depth_to_pointcloud(
                depth_m=depth_m,
                color_bgr=color_bgr,
                mask=mask,
                stride=2,
                min_depth=0.10,
                max_depth=1.50,
            )

            text = f"device={device} hand_pts={len(hand_points_xyz)}"
            cv2.putText(
                overlay, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )

            cv2.imshow("rgb", color_bgr)
            cv2.imshow("hand_mask", mask_vis)
            cv2.imshow("overlay", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()