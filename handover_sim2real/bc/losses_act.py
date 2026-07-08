"""
Phase-2 ACT loss.

Same two-part action as Phase-1 (continuous Δpose ch0..5 under SmoothL1 in
normalized space; binary gripper ch6 under BCEWithLogits), but now over a
**chunk** of k future actions, plus the CVAE **KL** term:

    total = pose_l1  +  gripper_weight · gripper_bce  +  kl_weight · KL(μ,logσ²)

Chunk slots past the episode end are padded and excluded via ``chunk_mask``
(1=real, 0=pad). KL uses the closed form against a unit Gaussian prior; it is
zero when the CVAE path is off (mu/logvar None) — handy for the use_cvae
ablation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _masked_mean(per_step: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """per_step[B,k], mask[B,k] -> scalar mean over valid slots."""
    denom = mask.sum().clamp(min=1.0)
    return (per_step * mask).sum() / denom


def act_loss(pred: torch.Tensor,
             target: torch.Tensor,
             chunk_mask: torch.Tensor,
             mu: Optional[torch.Tensor] = None,
             logvar: Optional[torch.Tensor] = None,
             gripper_weight: float = 1.0,
             kl_weight: float = 10.0) -> Dict[str, torch.Tensor]:
    """Composite ACT loss for a batch of chunks.

    Args:
        pred:        [B, k, 7] raw policy outputs (ch6 = logit).
        target:      [B, k, 7] expert chunk, normalized like the dataset
                     (ch0..5 normalized, ch6 raw 0/1).
        chunk_mask:  [B, k] 1=real action, 0=past-episode-end pad.
        mu, logvar:  [B, latent] CVAE posterior stats, or None (KL=0).
        gripper_weight, kl_weight: term multipliers.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.shape[-1] != 7:
        raise ValueError(f"expected last dim 7, got {pred.shape[-1]}")

    pose_err = F.smooth_l1_loss(pred[..., :6], target[..., :6], reduction="none")  # [B,k,6]
    pose_loss = _masked_mean(pose_err.mean(dim=-1), chunk_mask)

    grip_err = F.binary_cross_entropy_with_logits(
        pred[..., 6], target[..., 6], reduction="none")                            # [B,k]
    gripper_loss = _masked_mean(grip_err, chunk_mask)

    if mu is not None and logvar is not None:
        # KL(N(mu,sigma^2) || N(0,1)), mean over batch.
        kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()
    else:
        kl_loss = pose_loss.new_zeros(())

    total = pose_loss + gripper_weight * gripper_loss + kl_weight * kl_loss
    return {
        "pose_loss":    pose_loss,
        "gripper_loss": gripper_loss,
        "kl_loss":      kl_loss,
        "total":        total,
    }


@torch.no_grad()
def act_metrics(pred: torch.Tensor,
                target: torch.Tensor,
                chunk_mask: torch.Tensor) -> Dict[str, float]:
    """Masked per-chunk diagnostics (plain floats), comparable to bc_metrics.

    pose_l1 / pose_pos_l1 / pose_rot_l1 in normalized units; gripper_acc over
    valid slots (random-init baseline ~0.5).
    """
    err = (pred[..., :6] - target[..., :6]).abs()                # [B,k,6]
    pose_l1     = _masked_mean(err.mean(dim=-1),    chunk_mask).item()
    pose_pos_l1 = _masked_mean(err[..., :3].mean(-1), chunk_mask).item()
    pose_rot_l1 = _masked_mean(err[..., 3:].mean(-1), chunk_mask).item()

    pred_grip = (torch.sigmoid(pred[..., 6]) >= 0.5).float()     # [B,k]
    correct   = (pred_grip == target[..., 6]).float()
    gripper_acc = _masked_mean(correct, chunk_mask).item()

    return {
        "pose_l1":     pose_l1,
        "pose_pos_l1": pose_pos_l1,
        "pose_rot_l1": pose_rot_l1,
        "gripper_acc": gripper_acc,
    }
