"""
Phase-3 RL critic — twin Q network with a "remaining-steps" clock.

TD3-style clipped double-Q: two independent Q-heads on top of their own
PointNet++ scene encoder + robot MLP (separate from the actor's, so critic
gradients don't reshape the actor's representation). Each Q estimates the
discounted future *handover success* (sparse terminal reward) of taking the
normalized 6-D action `a` in state `(pc, rs)` with `remain` steps left.

The clock (`remain_norm`) is fed to the critic too — with a fixed episode
horizon and no clock in the state the same state would look both terminal (at
the step limit) and non-terminal, giving the value function contradictory
targets. Time-in-state removes that ambiguity, which is the whole reason we
condition the critic on it.

Inputs mirror the actor: `pc` raw, `rs_norm` normalized, `a_norm` in normalized
action space, `remain_norm ∈ (0,1]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from handover_sim2real.bc.models import (
    PointCloudEncoder, RobotEncoder, load_pretrained_pc_encoder,
)


def _mlp(in_dim: int, hidden, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    def __init__(self,
                 pc_channels: int = 5,
                 robot_state_dim: int = 32,
                 feature_dim: int = 256,
                 robot_hidden: int = 128,
                 q_hidden=(256, 256),
                 action_dim: int = 7,   # Δpose(6) + gripper logit(1)
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
        self.action_dim           = int(action_dim)
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
        q_in = 2 * feature_dim + self.action_dim + self.clock_dim
        self.q1 = _mlp(q_in, q_hidden, 1)
        self.q2 = _mlp(q_in, q_hidden, 1)
        # Goal-auxiliary head (pure regularizer): predict the EE-relative grasp
        # pose pos(3)+rot6d(6) from the scene features, shaping the value
        # encoder. Output unused for value estimation.
        self.aux_head = _mlp(feature_dim, (feature_dim // 2,), self.aux_dim)
        self.feature_dim = feature_dim

    def _select_robot_state(self, rs: torch.Tensor) -> torch.Tensor:
        if self.use_prev_act:
            return rs
        return rs[..., : self.full_robot_state_dim - self.prev_act_dim]

    def _features(self, pc, rs_norm, a_norm, remain_norm):
        """Returns (concat feature x, scene feature) — scene reused for the aux head."""
        scene = self.pc_encoder(pc)
        robot = self.robot_encoder(self._select_robot_state(rs_norm))
        if remain_norm.dim() == 1:
            remain_norm = remain_norm.unsqueeze(-1)
        x = torch.cat([scene, robot, a_norm, remain_norm], dim=-1)
        return x, scene

    def forward(self, pc, rs_norm, a_norm, remain_norm, return_aux: bool = False):
        """Returns (q1, q2), each [B, 1]. With return_aux=True also returns the
        auxiliary grasp-pose prediction [B, aux_dim]."""
        x, scene = self._features(pc, rs_norm, a_norm, remain_norm)
        q1, q2 = self.q1(x), self.q2(x)
        if return_aux:
            return q1, q2, self.aux_head(scene)
        return q1, q2

    def q1_only(self, pc, rs_norm, a_norm, remain_norm):
        """Q1 alone — used for the deterministic policy-gradient actor loss."""
        x, _ = self._features(pc, rs_norm, a_norm, remain_norm)
        return self.q1(x)

    # ----- warm start ------------------------------------------------------
    @torch.no_grad()
    def warm_start_encoders_from_bc(self, bc_model) -> None:
        """Copy the trained BC policy's encoders into the critic's (a good
        scene/robot representation to bootstrap value learning)."""
        self.pc_encoder.load_state_dict(bc_model.pc_encoder.state_dict())
        self.robot_encoder.load_state_dict(bc_model.robot_encoder.state_dict())

    def load_pretrained_pc_encoder(self, ckpt_path: str, verbose: bool = True) -> dict:
        return load_pretrained_pc_encoder(self.pc_encoder, ckpt_path, verbose=verbose)
