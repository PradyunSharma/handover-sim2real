"""
Phase-1 BC policy network.

Architecture (from the project block diagram, AnyGrasp/Grasp-MLP removed):

    point_cloud [B,N,5] ─► PointCloudEncoder ─► scene_feat [B,D]  ┐
                                                                   ├─► concat [B,2D] ─► PolicyHead ─► action [B,7]
    robot_state [B,32]  ─► RobotEncoder      ─► robot_feat [B,D]  ┘

The point-cloud encoder reuses GA-DDPG's PointNet++ backbone (`core.networks
.base_network`) so we don't fight nvcc again. We only borrow the 3-stage
SA-module → FC stack (returns 512 features per cloud) and project to a
configurable feature_dim.

Action layout: channels 0..5 are continuous Δpos/Δeuler (regressed under
SmoothL1 over the *normalized* targets); channel 6 is a gripper logit
(BCEWithLogitsLoss against a binary target).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from handover_sim2real.utils import add_sys_path_from_env

# GA-DDPG isn't an installed package — it lives at $GADDPG_DIR. Make sure it's
# on sys.path before we try to import from it. add_sys_path_from_env is
# idempotent on repeated calls.
add_sys_path_from_env("GADDPG_DIR")
from core.networks import base_network  # noqa: E402


# ── shared: pretrained PointNet++ encoder loader ─────────────────────────────

def load_pretrained_pc_encoder(pc_encoder: "PointCloudEncoder",
                               ckpt_path: str,
                               verbose: bool = True) -> dict:
    """Initialize a PointCloudEncoder's PointNet++ backbone from a GA-DDPG /
    handover-sim2real state-feat checkpoint.

    These checkpoints save ``{'net': OrderedDict, 'opt': ..., ...}`` where the
    policy encoder lives under ``module.encoder.*`` (and the critic encoder under
    ``module.value_encoder.*``, which we ignore). ``pc_encoder.encoder`` is the
    same ``base_network`` ModuleList, so keys line up after stripping the
    ``module.encoder.`` prefix. Any tensor whose shape doesn't match is skipped
    and left at init — notably the first SA conv when the point cloud has a
    different feature-channel count than the source. Returns a report dict;
    loads in place with strict=False.

    Shared by both BCPolicy (Phase-1 MLP) and ACTPolicy (Phase-2) so the
    warm-start logic lives in exactly one place.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload["net"] if isinstance(payload, dict) and "net" in payload else payload
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    src: dict = {}
    for k, v in sd.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        if kk.startswith("encoder."):            # policy encoder only
            src[kk[len("encoder."):]] = v

    tgt = pc_encoder.encoder.state_dict()
    loaded   = {k: v for k, v in src.items() if k in tgt and v.shape == tgt[k].shape}
    mismatch = sorted(k for k in tgt if k in src and src[k].shape != tgt[k].shape)
    missing  = sorted(k for k in tgt if k not in src)
    pc_encoder.encoder.load_state_dict(loaded, strict=False)

    if verbose:
        print(f"[pc_pretrained] loaded {len(loaded)}/{len(tgt)} encoder tensors "
              f"from {ckpt_path}")
        if mismatch:
            print(f"[pc_pretrained] reinitialized (shape mismatch): {mismatch}")
        if missing:
            print(f"[pc_pretrained] left at init (absent from ckpt): {missing}")
    return {"loaded": len(loaded), "target_total": len(tgt),
            "reinit_shape_mismatch": mismatch, "absent_from_ckpt": missing}


# ── point cloud encoder ──────────────────────────────────────────────────────

class PointCloudEncoder(nn.Module):
    """PointNet++ backbone (from GA-DDPG) + optional linear projection.

    Input:  pc [B, N, C]   xyz in first 3 channels, extra features in the rest.
    Output: feature [B, feature_dim].
    """

    def __init__(self,
                 in_channels: int = 5,
                 model_scale: int = 1,
                 feature_dim: int = 256,
                 pointnet_radius: float = 0.02,
                 pointnet_nclusters: int = 32):
        super().__init__()
        self.in_channels = in_channels
        # base_network returns nn.ModuleList([sa_modules, fc_layer]).
        # fc_layer outputs 512 * model_scale features.
        self.encoder = base_network(
            pointnet_radius, pointnet_nclusters, model_scale, in_channels
        )
        backbone_out = 512 * model_scale
        self.proj = (nn.Identity() if feature_dim == backbone_out
                     else nn.Linear(backbone_out, feature_dim))
        self.feature_dim = feature_dim

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: [B, N, C]  ->  xyz [B, N, 3], features [B, C, N]
        xyz      = pc[..., :3].contiguous()
        features = pc.transpose(1, -1).contiguous()
        for sa in self.encoder[0]:
            xyz, features = sa(xyz, features)
        # After the last SA module, features is [B, backbone_out, 1].
        z = self.encoder[1](features.squeeze(-1))
        return self.proj(z)


# ── robot state encoder ──────────────────────────────────────────────────────

class RobotEncoder(nn.Module):
    """Plain MLP: robot_state [B, 32] -> [B, feature_dim]."""

    def __init__(self,
                 in_dim: int = 32,
                 hidden_dim: int = 128,
                 feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, rs: torch.Tensor) -> torch.Tensor:
        return self.net(rs)


# ── policy head ──────────────────────────────────────────────────────────────

class PolicyHead(nn.Module):
    """Concatenated-feature MLP that produces the raw 7-D action vector.

    Channels 0..5 are continuous, channel 6 is a logit (no activation here).
    """

    def __init__(self,
                 in_dim: int,
                 hidden=(256, 256),
                 action_dim: int = 7):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── combined BC policy ───────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    """End-to-end Phase-1 BC policy.

    forward(pc, robot_state) returns the *raw* action [B, 7] — use this
    directly with bc_loss during training.

    predict(pc, robot_state) returns a deployable action where:
        • channels 0..5 are denormalized Δpos/Δeuler in metres / radians
        • channel 6 is hard-thresholded to {0, 1}
    Attach a Normalizer with `.set_normalizer(norm)` (or pass at __init__)
    so that predict() does input/output (de)normalization for you.
    """

    def __init__(self,
                 pc_channels: int = 5,
                 robot_state_dim: int = 32,
                 feature_dim: int = 256,
                 robot_hidden: int = 128,
                 policy_hidden=(256, 256),
                 action_dim: int = 7,
                 pointnet_scale: int = 1,
                 pointnet_radius: float = 0.02,
                 pointnet_nclusters: int = 32,
                 use_prev_act: bool = True,
                 prev_act_dim: int = 6,
                 freeze_pc: bool = False,
                 normalizer=None):
        super().__init__()
        # The stored robot_state always has `robot_state_dim` channels with the
        # previous action as the trailing `prev_act_dim`. When use_prev_act is
        # False those trailing channels are dropped before the robot encoder
        # (see _select_robot_state); the encoder is sized accordingly. The
        # *input* to forward/predict is always the full robot_state_dim — the
        # slice happens internally, so the normalizer can stay full-width.
        self.use_prev_act         = bool(use_prev_act)
        self.prev_act_dim         = int(prev_act_dim)
        self.full_robot_state_dim = int(robot_state_dim)
        effective_robot_dim = (self.full_robot_state_dim if self.use_prev_act
                               else self.full_robot_state_dim - self.prev_act_dim)

        self.pc_encoder = PointCloudEncoder(
            in_channels=pc_channels,
            model_scale=pointnet_scale,
            feature_dim=feature_dim,
            pointnet_radius=pointnet_radius,
            pointnet_nclusters=pointnet_nclusters,
        )
        self.robot_encoder = RobotEncoder(
            in_dim=effective_robot_dim,
            hidden_dim=robot_hidden,
            feature_dim=feature_dim,
        )
        self.policy_head = PolicyHead(
            in_dim=2 * feature_dim,
            hidden=tuple(policy_hidden),
            action_dim=action_dim,
        )
        self.feature_dim = feature_dim
        self.action_dim  = action_dim
        # Normalizer is *not* a submodule — it's a plain Python object that
        # holds numpy arrays. Stored as an attribute so checkpoints don't try
        # to pickle it into state_dict.
        self.normalizer = normalizer

        # Optionally freeze the point-cloud encoder (train only robot MLP +
        # head). Default is False — full end-to-end / fine-tuning.
        self._pc_frozen = False
        if freeze_pc:
            self.set_pc_trainable(False)

    # ----- normalizer plumbing ---------------------------------------------
    def set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer

    # ----- point-cloud encoder: pretrained init + freeze -------------------
    def load_pretrained_pc_encoder(self, ckpt_path: str, verbose: bool = True) -> dict:
        """Initialize the PointNet++ encoder from a state-feat checkpoint.

        Thin wrapper around the module-level ``load_pretrained_pc_encoder`` so
        both BCPolicy and ACTPolicy share one implementation.
        """
        return load_pretrained_pc_encoder(self.pc_encoder, ckpt_path, verbose=verbose)

    def set_pc_trainable(self, trainable: bool) -> None:
        """Freeze (trainable=False) or unfreeze the point-cloud encoder.

        When frozen we also force the encoder to eval() so its BatchNorm
        running stats stop updating — see the train() override below, which
        re-applies this every time the module is put back in train mode.
        """
        self._pc_frozen = not trainable
        for p in self.pc_encoder.parameters():
            p.requires_grad = trainable
        if self._pc_frozen:
            self.pc_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_pc_frozen", False):
            self.pc_encoder.eval()   # keep frozen BN in eval even in train mode
        return self

    # ----- robot-state channel selection -----------------------------------
    def _select_robot_state(self, rs: torch.Tensor) -> torch.Tensor:
        """Optionally drop the trailing prev_action(6) channels.

        prev_act is ~0.9 correlated with the target action (the OMG path is
        smooth), so leaving it in lets the policy 'copy' the previous action
        and ignore the point cloud (causal confusion / copycat). Dropping it
        forces the policy to read proprioception + scene instead. Input is the
        full robot_state_dim; this returns the encoder-sized slice.
        """
        if self.use_prev_act:
            return rs
        return rs[..., : self.full_robot_state_dim - self.prev_act_dim]

    # ----- forward (training) ----------------------------------------------
    def forward(self, pc: torch.Tensor, rs: torch.Tensor) -> torch.Tensor:
        scene = self.pc_encoder(pc)
        robot = self.robot_encoder(self._select_robot_state(rs))
        return self.policy_head(torch.cat([scene, robot], dim=-1))

    # ----- inference helper -------------------------------------------------
    @torch.no_grad()
    def predict(self, pc: torch.Tensor, rs: torch.Tensor) -> torch.Tensor:
        """Run inference and return a deployable action.

        If a Normalizer is attached:
          1. robot_state is normalized before the forward pass,
          2. action channels 0..5 are denormalized after.
        Gripper logit (channel 6) is passed through sigmoid and thresholded.
        """
        self.eval()
        rs_in = self.normalizer.normalize_state(rs) if self.normalizer is not None else rs
        raw   = self.forward(pc, rs_in)
        if self.normalizer is not None:
            raw = self.normalizer.denormalize_action(raw)
        cont    = raw[..., :6]
        gripper = (torch.sigmoid(raw[..., 6:7]) > 0.5).float()
        return torch.cat([cont, gripper], dim=-1)

    # ----- parameter count for sanity --------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if (p.requires_grad or not trainable_only))
