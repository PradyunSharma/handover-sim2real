"""
Phase-3 RL actor — deterministic 6-DoF policy with a "remaining-steps" clock.

This is the Phase-1 BC backbone (PointNet++ scene encoder + robot MLP) re-used
verbatim, with two differences from `bc.models.BCPolicy`:

  1. It outputs only the **6-D pose delta** (Δpos/Δeuler in *normalized* action
     space, exactly `BCPolicy.forward()[:, :6]`). The gripper is NOT learned here
     — during RL rollouts it is set by a standoff/proximity heuristic (see
     `rl.rollout_worker`). RL only refines the approach.

  2. It is conditioned on a **clock**: `remain_norm = (max_steps - step) /
     max_steps ∈ (0, 1]`, injected at the fused-feature level (concatenated to
     `[scene_feat ⊕ robot_feat]` before the head). Injecting at the fused level
     (rather than into the robot-state vector, as GA-DDPG does) keeps the two
     encoders shape-identical to the BC policy, so warm-starting from a trained
     BC checkpoint is a clean 1:1 load.

Everything is in **normalized** action space (mean 0 / std 1 per channel, from
the BC `Normalizer`). Denormalization to real metres/radians happens only at
env-step time. `forward()` is linear (no tanh) so the warm-start below makes the
actor *exactly* reproduce the BC policy's pose output at init; callers clamp the
output to a sane range (`clamp_action`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from handover_sim2real.bc.models import (
    PointCloudEncoder, RobotEncoder, load_pretrained_pc_encoder,
)


class RLActor(nn.Module):
    def __init__(self,
                 pc_channels: int = 5,
                 robot_state_dim: int = 32,
                 feature_dim: int = 256,
                 robot_hidden: int = 128,
                 policy_hidden=(256, 256),
                 pointnet_scale: int = 1,
                 pointnet_radius: float = 0.02,
                 pointnet_nclusters: int = 32,
                 use_prev_act: bool = False,
                 prev_act_dim: int = 6,
                 clock_dim: int = 1,
                 aux_dim: int = 9):
        super().__init__()
        self.use_prev_act         = bool(use_prev_act)
        self.prev_act_dim         = int(prev_act_dim)
        self.full_robot_state_dim = int(robot_state_dim)
        self.clock_dim            = int(clock_dim)
        self.aux_dim              = int(aux_dim)
        effective_robot_dim = (self.full_robot_state_dim if self.use_prev_act
                               else self.full_robot_state_dim - self.prev_act_dim)

        self.pc_encoder = PointCloudEncoder(
            in_channels=pc_channels, model_scale=pointnet_scale,
            feature_dim=feature_dim, pointnet_radius=pointnet_radius,
            pointnet_nclusters=pointnet_nclusters,
        )
        self.robot_encoder = RobotEncoder(
            in_dim=effective_robot_dim, hidden_dim=robot_hidden,
            feature_dim=feature_dim,
        )

        # Head: [scene ⊕ robot ⊕ clock] -> 7-D action = Δpose(6, normalized) +
        # gripper logit(1). The gripper is a learned output again (like BC): the
        # policy decides WHEN to close, and the reward (+1 for closing within
        # grasp proximity) shapes that decision. Exec: gripper OPEN iff logit ≥ 0.
        layers: list[nn.Module] = []
        prev = 2 * feature_dim + self.clock_dim
        for h in policy_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, 7)]
        self.head = nn.Sequential(*layers)

        # Goal-auxiliary head: predicts the final grasp pose relative to the EE
        # as pos(3)+rot6d(6) from the scene features. Pure regularizer — its
        # output is unused at inference; the supervised loss just shapes the
        # PointNet++ encoder ("where is the grasp"), which is what makes the
        # sparse-reward RL learnable. Randomly initialized (BC warm-start skips it).
        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, self.aux_dim),
        )

        self.feature_dim = feature_dim
        self._fused_dim  = 2 * feature_dim   # where the clock is appended

    # ----- robot-state channel selection (mirrors BCPolicy) ----------------
    def _select_robot_state(self, rs: torch.Tensor) -> torch.Tensor:
        if self.use_prev_act:
            return rs
        return rs[..., : self.full_robot_state_dim - self.prev_act_dim]

    def forward(self, pc: torch.Tensor, rs_norm: torch.Tensor,
                remain_norm: torch.Tensor, return_aux: bool = False):
        """pc [B,N,C] raw, rs_norm [B,32] normalized, remain_norm [B,1] in (0,1].

        Returns the 7-D action [B,7] = Δpose(6, normalized) + gripper logit(1).
        With return_aux=True also returns the auxiliary grasp-pose prediction
        [B, aux_dim].
        """
        scene = self.pc_encoder(pc)
        robot = self.robot_encoder(self._select_robot_state(rs_norm))
        if remain_norm.dim() == 1:
            remain_norm = remain_norm.unsqueeze(-1)
        action = self.head(torch.cat([scene, robot, remain_norm], dim=-1))
        if return_aux:
            return action, self.aux_head(scene)
        return action

    # ----- warm start from a trained BC policy -----------------------------
    @torch.no_grad()
    def warm_start_from_bc(self, bc_model) -> None:
        """Copy the BC policy's weights so this actor *exactly* reproduces
        `BCPolicy.forward(pc, rs)` (all 7 dims — pose + gripper logit) at init:
        the clock is ignored at init via zero-initialized clock columns, and the
        BC gripper row is kept so the actor starts with BC's close-timing (RL
        then refines both from the reward). All learned by RL thereafter.

        Requires the BC policy to share this actor's `feature_dim`,
        `use_prev_act`, and `policy_hidden` (so the encoders + head line up).
        """
        # Encoders: 1:1 (identical modules / shapes).
        self.pc_encoder.load_state_dict(bc_model.pc_encoder.state_dict())
        self.robot_encoder.load_state_dict(bc_model.robot_encoder.state_dict())

        # Head: BCPolicy.policy_head.net is a matching Sequential with a 7-D
        # final layer (same width as ours). Copy each layer, padding only the
        # first layer with zeros for the extra clock columns.
        src = list(bc_model.policy_head.net)
        dst = list(self.head)
        assert len(src) == len(dst), (
            f"head depth mismatch: bc={len(src)} actor={len(dst)} "
            "(policy_hidden must match)")
        first_linear = True
        for s, d in zip(src, dst):
            if not isinstance(s, nn.Linear):
                continue
            if first_linear:
                # d.weight: [H, 2F+clock]; s.weight: [H, 2F]
                assert d.weight.shape[1] == s.weight.shape[1] + self.clock_dim
                d.weight.zero_()
                d.weight[:, : s.weight.shape[1]].copy_(s.weight)
                d.bias.copy_(s.bias)
                first_linear = False
            else:
                assert d.weight.shape == s.weight.shape, (d.weight.shape, s.weight.shape)
                d.weight.copy_(s.weight)
                d.bias.copy_(s.bias)

    # ----- pretrained PC encoder (fallback when no BC warm-start) ----------
    def load_pretrained_pc_encoder(self, ckpt_path: str, verbose: bool = True) -> dict:
        return load_pretrained_pc_encoder(self.pc_encoder, ckpt_path, verbose=verbose)


def clamp_action(a: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamp a normalized action to [-limit, limit] per channel."""
    return a.clamp(-limit, limit)
