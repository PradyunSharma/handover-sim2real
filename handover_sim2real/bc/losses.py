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


POSE_LOSSES = ("smooth_l1", "pm", "both")


def _denorm(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Normalized channels 0..5 -> real metres/radians. Identity if no stats."""
    if std is None:
        return x[..., :6]
    return x[..., :6] * std + mean


def bc_loss(pred: torch.Tensor,
            target: torch.Tensor,
            gripper_weight: float = 1.0,
            *,
            pose_loss: str = "smooth_l1",
            pm_weight: float = 1.0,
            action_mean: torch.Tensor | None = None,
            action_std: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """Composite BC loss for a single batch.

    Args:
        pred:   [B, 7] raw policy outputs (BCPolicy.forward); channel 6 is a logit.
        target: [B, 7] expert action with the *same* normalization applied as
                in BCDataset (channels 0..5 normalized, channel 6 raw 0/1).
        gripper_weight: scalar multiplier on the gripper BCE term.
        pose_loss: which pose term drives the gradient —
            smooth_l1  SmoothL1 on the NORMALIZED channels (the Phase-1 default).
                       Per-channel z-scoring makes all six unit-variance, so this
                       weights a 1-sigma rotation error the same as a 1-sigma
                       translation error even though (measured on
                       train_pinned_omg_ok) 1 sigma of translation moves the
                       gripper control points 16-22 mm and 1 sigma of rotation
                       moves them 2.2-4.9 mm.
            pm         `pose_pm_loss` on the DENORMALIZED action — GA-DDPG's
                       core/loss.py `pose_bc_loss`, which it applies in raw units
                       (its policy tanh-squashes into the action-space bounds; it
                       has no z-scoring at all). One SE(3) metric in metres, so
                       the two parts are weighted by physical effect rather than
                       by the variance of their channel.
            both       smooth_l1 + pm_weight * pm.
        pm_weight: multiplier on the PM term (ignored when pose_loss=smooth_l1).
        action_mean/action_std: [6] tensors used to denormalize before the PM
            term. Required for pm/both — without them the "metres" are z-scores
            and the whole point is lost.

    Returns:
        Dict with tensors:
            pose_loss     scalar — whichever term(s) `pose_loss` selected
            pose_sl1      scalar — the SmoothL1 term, always reported
            pose_pm       scalar — the PM term in metres (pm/both only)
            gripper_loss  scalar — BCEWithLogits over channel 6
            total         scalar — pose_loss + gripper_weight * gripper_loss
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.shape[-1] != 7:
        raise ValueError(f"expected last dim 7, got {pred.shape[-1]}")
    if pose_loss not in POSE_LOSSES:
        raise ValueError(f"pose_loss must be one of {POSE_LOSSES}, got {pose_loss!r}")
    if pose_loss != "smooth_l1" and action_std is None:
        raise ValueError(f"pose_loss={pose_loss!r} needs action_mean/action_std to "
                         f"denormalize; PM loss is only meaningful in real units")

    sl1          = F.smooth_l1_loss(pred[..., :6], target[..., :6])
    gripper_loss = F.binary_cross_entropy_with_logits(pred[..., 6], target[..., 6])

    out: Dict[str, torch.Tensor] = {"pose_sl1": sl1}
    if pose_loss == "smooth_l1":
        pose = sl1
    else:
        pm = pose_pm_loss(_denorm(pred, action_mean, action_std),
                          _denorm(target, action_mean, action_std))
        out["pose_pm"] = pm
        pose = pm_weight * pm if pose_loss == "pm" else sl1 + pm_weight * pm

    out["pose_loss"]    = pose
    out["gripper_loss"] = gripper_loss
    out["total"]        = pose + gripper_weight * gripper_loss
    return out


@torch.no_grad()
def bc_metrics(pred: torch.Tensor, target: torch.Tensor,
               action_mean: torch.Tensor | None = None,
               action_std: torch.Tensor | None = None) -> Dict[str, float]:
    """Per-batch diagnostic metrics — handy to log alongside the loss.

    All values returned as plain Python floats (already detached).

    pose_l1       mean |pred - target| over channels 0..5 in normalized units;
                  comparable across runs because targets are normalized to ~N(0,1).
    pose_pos_l1   same, but only Δpos channels 0..2 (still normalized units).
    pose_rot_l1   same, but only Δeuler channels 3..5.
    gripper_acc   fraction of samples where sigmoid(logit) ≥ 0.5 matches the
                  binary target. Random-init baseline is ~0.5.

    With action_mean/action_std, three more in REAL units. These are the ones to
    compare across runs that use different pose losses: the normalized ones above
    are denominated in the run's own channel sigmas, and `total` changes scale
    entirely between smooth_l1 (z-scores) and pm (metres).

    pose_pos_m    mean |Δpos error| in metres
    pose_rot_rad  mean |Δeuler error| in radians
    pose_pm_mm    the PM metric in millimetres — the single number that says how
                  far apart the two gripper poses actually are
    """
    err = (pred[..., :6] - target[..., :6]).abs()
    pred_gripper = (torch.sigmoid(pred[..., 6]) >= 0.5).float()

    out = {
        "pose_l1":     err.mean().item(),
        "pose_pos_l1": err[..., :3].mean().item(),
        "pose_rot_l1": err[..., 3:].mean().item(),
        "gripper_acc": (pred_gripper == target[..., 6]).float().mean().item(),
    }
    if action_std is not None:
        pr = _denorm(pred, action_mean, action_std)
        tg = _denorm(target, action_mean, action_std)
        real = (pr - tg).abs()
        out["pose_pos_m"]   = real[..., :3].mean().item()
        out["pose_rot_rad"] = real[..., 3:].mean().item()
        out["pose_pm_mm"]   = pose_pm_loss(pr, tg).item() * 1000.0
    return out
