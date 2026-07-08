"""
Phase-2 ACT policy: temporal transformer + CVAE-style action-chunking head.

Pipeline (PointNet++ kept exactly as Phase-1 — no surgery):

    for each of T history frames (shared weights, batched as [B*T, …]):
        pc[t] ─► PointCloudEncoder ─► scene_feat[t] [B, F]   (reused verbatim)
        rs[t] ─► RobotEncoder      ─► robot_feat[t] [B, F]   (reused verbatim)
        obs_tok[t] = Linear(2F→D)(scene_feat ⊕ robot_feat)   # one fused token/frame

    [obs_tok_1..T] + temporal pos ─► Transformer ENCODER ─► memory [B, T, D]

    CVAE (train only):  [CLS, proprio_cond, embed(action_chunk)] + pos
                        ─► Transformer ─► CLS ─► (μ, logσ²) ─► z = μ + σ·ε
                        inference: z = μ = 0   (CVAE encoder skipped)
    z ─► Linear(latent→D) ─► z_tok, prepended to memory ─► memory⁺ [B, T+1, D]

    k learned query tokens ─► Transformer DECODER (cross-attn memory⁺)
                           ─► Linear(D→7) ─► pred_chunk [B, k, 7]   (ch6 = logit)

Action layout matches Phase-1: ch0..5 continuous Δpos/Δeuler (regressed in
normalized space), ch6 gripper logit. The CVAE + chunking + ensembling reduce
compounding error; with a deterministic OMG teacher the latent may collapse
toward the prior (benign — the decoder falls back to deterministic).

Ablation flags: ``history_len=1`` collapses the temporal part to a single frame;
``use_cvae=False`` forces z=0 always (KL term vanishes).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .models import PointCloudEncoder, RobotEncoder, load_pretrained_pc_encoder


# ── ACT policy ───────────────────────────────────────────────────────────────

class ACTPolicy(nn.Module):
    """Temporal-transformer + CVAE action-chunking BC policy.

    forward(pc_hist, rs_hist, action_chunk) -> (pred_chunk[B,k,7], mu, logvar)
      • pred_chunk: raw outputs (ch6 = logit) — feed straight to act_loss.
      • mu, logvar: latent stats for the KL term, or None when the CVAE path is
        not used (use_cvae=False, or eval, or action_chunk is None).

    predict(pc_hist, rs_hist) -> deployable chunk[B,k,7] with z=0:
      • ch0..5 denormalized Δpose, ch6 = gripper *probability* (sigmoid, NOT
        thresholded) so a TemporalEnsembler can average probabilities across
        overlapping chunks before thresholding.
    """

    def __init__(self,
                 pc_channels: int = 5,
                 robot_state_dim: int = 32,
                 action_dim: int = 7,
                 feature_dim: int = 256,
                 robot_hidden: int = 128,
                 d_model: int = 256,
                 n_heads: int = 4,
                 enc_layers: int = 3,
                 dec_layers: int = 3,
                 cvae_enc_layers: int = 2,
                 dim_feedforward: int | None = None,
                 dropout: float = 0.1,
                 history_len: int = 4,
                 chunk_len: int = 8,
                 latent_dim: int = 32,
                 use_cvae: bool = True,
                 use_prev_act: bool = False,
                 prev_act_dim: int = 6,
                 pointnet_scale: int = 1,
                 pointnet_radius: float = 0.02,
                 pointnet_nclusters: int = 32,
                 freeze_pc: bool = False,
                 normalizer=None):
        super().__init__()
        self.action_dim   = int(action_dim)
        self.d_model      = int(d_model)
        self.history_len  = int(history_len)
        self.chunk_len    = int(chunk_len)
        self.latent_dim   = int(latent_dim)
        self.use_cvae     = bool(use_cvae)
        ff = int(dim_feedforward) if dim_feedforward else 4 * self.d_model

        # ----- robot-state channel selection (mirrors BCPolicy) -------------
        self.use_prev_act         = bool(use_prev_act)
        self.prev_act_dim         = int(prev_act_dim)
        self.full_robot_state_dim = int(robot_state_dim)
        effective_robot_dim = (self.full_robot_state_dim if self.use_prev_act
                               else self.full_robot_state_dim - self.prev_act_dim)

        # ----- per-frame encoders (reused verbatim from Phase-1) ------------
        self.pc_encoder = PointCloudEncoder(
            in_channels=pc_channels, model_scale=pointnet_scale,
            feature_dim=feature_dim, pointnet_radius=pointnet_radius,
            pointnet_nclusters=pointnet_nclusters,
        )
        self.robot_encoder = RobotEncoder(
            in_dim=effective_robot_dim, hidden_dim=robot_hidden, feature_dim=feature_dim,
        )
        self.obs_proj = nn.Linear(2 * feature_dim, self.d_model)

        # ----- temporal encoder --------------------------------------------
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.history_len, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            self.d_model, n_heads, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # ----- CVAE posterior encoder (training only) -----------------------
        if self.use_cvae:
            self.action_emb   = nn.Linear(self.action_dim, self.d_model)
            self.proprio_cond = nn.Linear(feature_dim, self.d_model)  # from current robot_feat
            self.cls_token    = nn.Parameter(torch.zeros(1, 1, self.d_model))
            # positions for [CLS, proprio_cond, k action tokens]
            self.cvae_pos     = nn.Parameter(torch.zeros(1, 2 + self.chunk_len, self.d_model))
            cvae_layer = nn.TransformerEncoderLayer(
                self.d_model, n_heads, dim_feedforward=ff, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.cvae_encoder = nn.TransformerEncoder(cvae_layer, num_layers=cvae_enc_layers)
            self.latent_head  = nn.Linear(self.d_model, 2 * self.latent_dim)
        self.latent_to_mem = nn.Linear(self.latent_dim, self.d_model)

        # ----- chunk decoder -----------------------------------------------
        self.query_emb = nn.Parameter(torch.zeros(1, self.chunk_len, self.d_model))
        dec_layer = nn.TransformerDecoderLayer(
            self.d_model, n_heads, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.action_head = nn.Linear(self.d_model, self.action_dim)

        self.normalizer = normalizer
        self._reset_parameters()

        self._pc_frozen = False
        if freeze_pc:
            self.set_pc_trainable(False)

    def _reset_parameters(self) -> None:
        for p in [self.temporal_pos, self.query_emb]:
            nn.init.trunc_normal_(p, std=0.02)
        if self.use_cvae:
            for p in [self.cls_token, self.cvae_pos]:
                nn.init.trunc_normal_(p, std=0.02)

    # ----- normalizer / pretrained / freeze plumbing -----------------------
    def set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer

    def load_pretrained_pc_encoder(self, ckpt_path: str, verbose: bool = True) -> dict:
        return load_pretrained_pc_encoder(self.pc_encoder, ckpt_path, verbose=verbose)

    def set_pc_trainable(self, trainable: bool) -> None:
        self._pc_frozen = not trainable
        for p in self.pc_encoder.parameters():
            p.requires_grad = trainable
        if self._pc_frozen:
            self.pc_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_pc_frozen", False):
            self.pc_encoder.eval()  # keep frozen BN in eval even in train mode
        return self

    def _select_robot_state(self, rs: torch.Tensor) -> torch.Tensor:
        """Drop trailing prev_action(6) channels unless use_prev_act (copycat guard)."""
        if self.use_prev_act:
            return rs
        return rs[..., : self.full_robot_state_dim - self.prev_act_dim]

    # ----- per-frame encoding shared by forward/predict --------------------
    def _encode_history(self, pc_hist: torch.Tensor, rs_hist: torch.Tensor):
        """pc_hist[B,T,N,5], rs_hist[B,T,32] -> memory[B,T,D], robot_feat_last[B,F]."""
        B, T = pc_hist.shape[:2]
        pc_flat = pc_hist.reshape(B * T, *pc_hist.shape[2:])         # [B*T, N, 5]
        rs_flat = self._select_robot_state(rs_hist.reshape(B * T, rs_hist.shape[-1]))

        # PointNet++'s custom CUDA ops (group_points) are float32-only and assert
        # on fp16 — so the encoder must run OUTSIDE autocast. The transformer half
        # still benefits from AMP. Cast scene to robot's dtype before concat so the
        # cat is dtype-consistent whether AMP is on (robot=fp16) or off (fp32).
        with torch.cuda.amp.autocast(enabled=False):
            scene = self.pc_encoder(pc_flat.float())               # [B*T, F] fp32
        robot = self.robot_encoder(rs_flat)                        # [B*T, F]
        obs = self.obs_proj(torch.cat([scene.to(robot.dtype), robot], dim=-1))
        obs = obs.reshape(B, T, self.d_model) + self.temporal_pos   # [B, T, D]
        memory = self.temporal_encoder(obs)                         # [B, T, D]

        robot_last = robot.reshape(B, T, -1)[:, -1]                 # [B, F] current frame
        return memory, robot_last

    # ----- CVAE posterior (training) ---------------------------------------
    def _encode_latent(self, action_chunk: torch.Tensor, robot_last: torch.Tensor):
        """Returns (z, mu, logvar). action_chunk[B,k,7], robot_last[B,F]."""
        B = action_chunk.shape[0]
        cls  = self.cls_token.expand(B, -1, -1)                     # [B,1,D]
        cond = self.proprio_cond(robot_last).unsqueeze(1)          # [B,1,D]
        acts = self.action_emb(action_chunk)                       # [B,k,D]
        tokens = torch.cat([cls, cond, acts], dim=1) + self.cvae_pos
        h = self.cvae_encoder(tokens)[:, 0]                        # CLS output [B,D]
        mu, logvar = self.latent_head(h).chunk(2, dim=-1)         # each [B,latent]
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    # ----- decode k-action chunk from memory + latent ----------------------
    def _decode(self, memory: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        z_tok = self.latent_to_mem(z).unsqueeze(1)                 # [B,1,D]
        memory_plus = torch.cat([z_tok, memory], dim=1)            # [B,T+1,D]
        queries = self.query_emb.expand(B, -1, -1)                 # [B,k,D]
        dec = self.decoder(queries, memory_plus)                  # [B,k,D]
        return self.action_head(dec)                              # [B,k,7]

    # ----- forward (training) ----------------------------------------------
    def forward(self, pc_hist: torch.Tensor, rs_hist: torch.Tensor,
                action_chunk: torch.Tensor | None = None):
        memory, robot_last = self._encode_history(pc_hist, rs_hist)

        if self.use_cvae and action_chunk is not None:
            z, mu, logvar = self._encode_latent(action_chunk, robot_last)
        else:
            z = torch.zeros(memory.shape[0], self.latent_dim,
                            device=memory.device, dtype=memory.dtype)
            mu = logvar = None

        pred = self._decode(memory, z)
        return pred, mu, logvar

    # ----- inference -------------------------------------------------------
    @torch.no_grad()
    def predict(self, pc_hist: torch.Tensor, rs_hist: torch.Tensor) -> torch.Tensor:
        """Deployable chunk with z=0. ch0..5 denormalized Δpose, ch6 = gripper prob."""
        self.eval()
        rs_in = (self.normalizer.normalize_state(rs_hist)
                 if self.normalizer is not None else rs_hist)
        pred, _, _ = self.forward(pc_hist, rs_in, action_chunk=None)  # [B,k,7]
        if self.normalizer is not None:
            pred = self.normalizer.denormalize_action(pred)          # ch6 untouched
        cont    = pred[..., :6]
        gripper = torch.sigmoid(pred[..., 6:7])                      # probability, not bit
        return torch.cat([cont, gripper], dim=-1)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if (p.requires_grad or not trainable_only))
