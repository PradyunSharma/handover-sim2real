"""
Phase-3 RL actor, ACT (Phase-2) backbone — Design A: temporal observation,
single-action RL.

This wraps the Phase-2 `bc.models_act.ACTPolicy` (PointNet++ per-frame encoder →
temporal transformer → CVAE → chunk decoder) and presents the SAME call interface
as `rl.actor.RLActor`, so the TD3+BC loop can use it as a drop-in policy:

    action[B,7] = actor(pc_hist[B,T,N,C], rs_norm_hist[B,T,32], remain_norm[B,1])

Design A keeps TD3 single-step: the actor consumes a T-frame history but emits ONE
action per step (chunk element 0). The critic, replay-buffer Bellman math, and the
policy-gradient path are unchanged single-step — the ONLY new thing is that the
actor sees the last T frames instead of one. (Chunk execution + temporal ensembling
+ CVAE is Design B; here `chunk_len=1` and `use_cvae=False` typically, but both are
config-driven so B is reachable without touching this file.)

Differences from `RLActor` that the wrapper reconciles:
  • Clock — injected inside ACTPolicy (clock_dim>0) at the per-frame token level,
    not into the robot vector. No BC warm-start alignment to preserve (arch=act is
    from-scratch only), so this is free.
  • Goal-aux head — ACTPolicy has none; the loop's `aux_weight` goal-auxiliary
    regularizer needs one. We add it here, fed by the CURRENT frame's scene feature
    (the last history frame), matching `RLActor.aux_head` semantics.
  • `history_len` attribute — the rollout worker and trainer branch on
    `getattr(actor, "history_len", 1) > 1` to build/feed the T-frame history; a plain
    `RLActor` reports 1 and the single-frame path stays byte-identical.

Everything is in normalized action space (same as RLActor); denormalization happens
at env-step time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from handover_sim2real.bc.models_act import ACTPolicy


class RLActorACT(nn.Module):
    def __init__(self,
                 pc_channels: int = 5,
                 robot_state_dim: int = 32,
                 feature_dim: int = 256,
                 robot_hidden: int = 128,
                 pointnet_scale: int = 1,
                 pointnet_radius: float = 0.02,
                 pointnet_nclusters: int = 32,
                 use_prev_act: bool = False,
                 prev_act_dim: int = 6,
                 drop_joint_state: bool = False,
                 joint_state_dim: int = 18,
                 clock_dim: int = 1,
                 aux_dim: int = 9,
                 # ── ACT (Phase-2) transformer hyperparameters ──
                 history_len: int = 4,
                 chunk_len: int = 1,
                 d_model: int = 256,
                 n_heads: int = 4,
                 enc_layers: int = 3,
                 dec_layers: int = 3,
                 cvae_enc_layers: int = 2,
                 latent_dim: int = 32,
                 use_cvae: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.history_len = int(history_len)
        self.clock_dim   = int(clock_dim)
        self.aux_dim     = int(aux_dim)
        self.feature_dim = int(feature_dim)

        self.act = ACTPolicy(
            pc_channels=pc_channels, robot_state_dim=robot_state_dim,
            action_dim=7, feature_dim=feature_dim, robot_hidden=robot_hidden,
            d_model=d_model, n_heads=n_heads, enc_layers=enc_layers,
            dec_layers=dec_layers, cvae_enc_layers=cvae_enc_layers,
            latent_dim=latent_dim, use_cvae=use_cvae, dropout=dropout,
            history_len=history_len, chunk_len=chunk_len,
            use_prev_act=use_prev_act, prev_act_dim=prev_act_dim,
            drop_joint_state=drop_joint_state, joint_state_dim=joint_state_dim,
            clock_dim=clock_dim,
            pointnet_scale=pointnet_scale, pointnet_radius=pointnet_radius,
            pointnet_nclusters=pointnet_nclusters,
        )
        # Goal-auxiliary head (pure regularizer): predict the EE-relative grasp
        # pose pos(3)+rot6d(6) from the current-frame scene feature. Output unused
        # at inference; the supervised loss shapes the PointNet++ encoder. Mirrors
        # RLActor.aux_head. Randomly initialized.
        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, self.aux_dim),
        )

    # expose the PointNet++ encoder like RLActor.pc_encoder — as a PROPERTY (not a
    # submodule alias) so state_dict() doesn't carry duplicate encoder keys.
    @property
    def pc_encoder(self):
        return self.act.pc_encoder

    def forward(self, pc_hist: torch.Tensor, rs_norm_hist: torch.Tensor,
                remain_norm: torch.Tensor, return_aux: bool = False):
        """pc_hist [B,T,N,C] raw, rs_norm_hist [B,T,32] normalized, remain_norm
        [B,1] in (0,1]. Returns the 7-D action [B,7] = Δpose(6, normalized) +
        gripper logit(1) (chunk element 0). With return_aux=True also returns the
        auxiliary grasp-pose prediction [B, aux_dim]."""
        if remain_norm.dim() == 1:
            remain_norm = remain_norm.unsqueeze(-1)
        memory, _, scene_last = self.act._encode_history(
            pc_hist, rs_norm_hist, remain_norm)
        # z = 0 (prior mean): the CVAE posterior needs the ground-truth action chunk,
        # which is unavailable at policy time — the decoder is deterministic here.
        z = torch.zeros(memory.shape[0], self.act.latent_dim,
                        device=memory.device, dtype=memory.dtype)
        pred = self.act._decode(memory, z)          # [B, chunk_len, 7]
        action = pred[:, 0, :]                       # single action [B, 7]
        if return_aux:
            return action, self.aux_head(scene_last)
        return action

    # ----- pretrained PC encoder (optional; arch=act is otherwise from-scratch) --
    def load_pretrained_pc_encoder(self, ckpt_path: str, verbose: bool = True) -> dict:
        return self.act.load_pretrained_pc_encoder(ckpt_path, verbose=verbose)
