"""
Phase-1 BC loss.

The action vector has two qualitatively different parts:

  channels 0..5  continuous Δpos (3) + Δeuler (3), regressed in the
                 *normalized* space (mean=0, std=1 per channel). SmoothL1
                 (a.k.a. Huber loss) gives us L2 in the small-error regime
                 and L1 in the large-error regime — robust to a few outlier
                 expert steps without losing the gradient near zero.

  channel 6      binary gripper command {0=close, 1=open}. The policy emits
                 a logit here; BCEWithLogitsLoss is numerically stable and
                 expects logits + 0/1 targets.

Both terms are summed (with a configurable weight on the gripper term) to
form a scalar total loss. We also return the components for per-epoch
logging, plus a couple of plain-Python metrics (gripper accuracy, per-step
Δpose L1 in *normalized* units) that the trainer can dump to CSV.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# ── point-matching (PM) pose loss (GA-DDPG core/loss.py `pose_bc_loss`) ───────
# A fixed set of Panda gripper control points (2 TCP + 2 knuckle + 2 fingertip),
# in metres in the gripper frame. Instead of an L1 on the raw [Δpos(m) ‖ Δeuler(rad)]
# — which sums incommensurable units and scores rotation with a bad euler metric —
# we transform these points by BOTH the predicted and the target (R, t) and L1 the
# point displacement. That is one physically-meaningful SE(3) distance (metres of
# gripper-point motion) in which orientation is weighted by its real effect on the
# gripper — the metric 6-DoF-pose ("ADD") loss, and the fix for our weak wrist
# rotation (RL eval_min_rot plateaus ~0.4 rad vs the 0.34 close threshold).
_GRIPPER_CONTROL_POINTS = torch.tensor(
    [[0.000, 0.000, 0.000],
     [0.000, 0.000, 0.000],
     [0.053, 0.000, 0.075],
     [-0.053, 0.000, 0.075],
     [0.053, 0.000, 0.105],
     [-0.053, 0.000, 0.105]], dtype=torch.float32)   # [6, 3]


def _euler_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Batched intrinsic-free euler -> rotation matrix, R = Rz·Ry·Rx (transforms3d
    'sxyz', which is how our actions are built: `mat2euler(...)` default axes). euler
    [..., 3] -> R [..., 3, 3]."""
    ex, ey, ez = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(ex), torch.sin(ex)
    cy, sy = torch.cos(ey), torch.sin(ey)
    cz, sz = torch.cos(ez), torch.sin(ez)
    one, zero = torch.ones_like(cx), torch.zeros_like(cx)
    shape = (*ex.shape, 3, 3)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).reshape(shape)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).reshape(shape)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).reshape(shape)
    return Rz @ Ry @ Rx


def pose_pm_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    """Point-matching loss between two 6-D poses in **real units** (metres, radians;
    denormalize before calling). pred/target: [B, 6] = [Δpos(3) ‖ Δeuler(3)]. Returns
    the mean over batch of the summed-L1 gripper-control-point displacement."""
    cp = _GRIPPER_CONTROL_POINTS.to(pred_pose.device, pred_pose.dtype)  # [6,3]

    def _xf(pose: torch.Tensor) -> torch.Tensor:                 # [B,6] -> [B,6,3]
        R = _euler_to_matrix(pose[..., 3:6])                     # [B,3,3]
        pts = torch.matmul(cp.unsqueeze(0), R.transpose(-1, -2))  # [B,6,3]
        return pts + pose[..., None, :3]                         # + translation
    return (_xf(pred_pose) - _xf(target_pose)).abs().sum(-1).mean()


def bc_loss(pred: torch.Tensor,
            target: torch.Tensor,
            gripper_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    """Composite BC loss for a single batch.

    Args:
        pred:   [B, 7] raw policy outputs (BCPolicy.forward); channel 6 is a logit.
        target: [B, 7] expert action with the *same* normalization applied as
                in BCDataset (channels 0..5 normalized, channel 6 raw 0/1).
        gripper_weight: scalar multiplier on the gripper BCE term.

    Returns:
        Dict with tensors:
            pose_loss     scalar — SmoothL1 over channels 0..5
            gripper_loss  scalar — BCEWithLogits over channel 6
            total         scalar — pose_loss + gripper_weight * gripper_loss
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.shape[-1] != 7:
        raise ValueError(f"expected last dim 7, got {pred.shape[-1]}")

    pose_loss    = F.smooth_l1_loss(pred[..., :6], target[..., :6])
    gripper_loss = F.binary_cross_entropy_with_logits(pred[..., 6], target[..., 6])
    total        = pose_loss + gripper_weight * gripper_loss
    return {
        "pose_loss":    pose_loss,
        "gripper_loss": gripper_loss,
        "total":        total,
    }


@torch.no_grad()
def bc_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Per-batch diagnostic metrics — handy to log alongside the loss.

    All values returned as plain Python floats (already detached).

    pose_l1       mean |pred - target| over channels 0..5 in normalized units;
                  comparable across runs because targets are normalized to ~N(0,1).
    pose_pos_l1   same, but only Δpos channels 0..2 (still normalized units).
    pose_rot_l1   same, but only Δeuler channels 3..5.
    gripper_acc   fraction of samples where sigmoid(logit) ≥ 0.5 matches the
                  binary target. Random-init baseline is ~0.5.
    """
    err = (pred[..., :6] - target[..., :6]).abs()
    pose_l1     = err.mean().item()
    pose_pos_l1 = err[..., :3].mean().item()
    pose_rot_l1 = err[..., 3:].mean().item()

    pred_gripper = (torch.sigmoid(pred[..., 6]) >= 0.5).float()
    gripper_acc  = (pred_gripper == target[..., 6]).float().mean().item()

    return {
        "pose_l1":     pose_l1,
        "pose_pos_l1": pose_pos_l1,
        "pose_rot_l1": pose_rot_l1,
        "gripper_acc": gripper_acc,
    }
