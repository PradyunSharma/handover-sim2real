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
import torch.nn.functional as F

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


# ── goal-grasp encoder (Phase 5) ─────────────────────────────────────────────

class GraspEncoder(nn.Module):
    """Plain MLP: goal grasp in the CURRENT EE frame [B, 9] -> [B, grasp_feat_dim].

    Structurally a second RobotEncoder, and injected at the same place — the
    fused feature vector, following `rl/actor.py`'s `clock_dim`, whose docstring
    gives the reason: "Injecting at the fused level (rather than into the
    robot-state vector, as GA-DDPG does) keeps the two encoders shape-identical
    to the BC policy, so warm-starting from a trained BC checkpoint is a clean
    1:1 load." The same holds here — a Phase-4 checkpoint's pc_encoder and
    robot_encoder still load 1:1 into a Phase-5 model.

    Input is rot6d(6) + translation(3), not the aux head's quaternion(4)+trans(3).
    For a regression *target* the quaternion is fine; for an *input* its double
    cover (q and -q are the same rotation) makes the mapping the MLP has to learn
    discontinuous for no benefit. rot6d — the first two columns of R, Zhou et al.
    2019 — is continuous and needs no normalization.
    """

    def __init__(self,
                 in_dim: int = 9,
                 hidden_dim: int = 128,
                 feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.in_dim = in_dim
        self.feature_dim = feature_dim

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.net(goal)


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
                 drop_joint_state: bool = False,
                 joint_state_dim: int = 18,
                 freeze_pc: bool = False,
                 aux_head: bool = False,
                 aux_dim: int = 7,
                 aux_hidden=(256, 256),
                 grasp_cond: bool = False,
                 grasp_cond_dim: int = 9,
                 grasp_hidden: int = 128,
                 grasp_feat_dim: int = 128,
                 normalizer=None):
        super().__init__()
        # The stored robot_state always has `robot_state_dim` channels, laid out
        # joint_pos(9)+joint_vel(9)+ee_pose(7)+gripper(1)+prev_act(6). Two
        # optional slices trim it before the robot encoder (see
        # _select_robot_state); the encoder is sized accordingly. The *input* to
        # forward/predict is always the full robot_state_dim — the slice happens
        # internally, so the normalizer can stay full-width.
        #
        # drop_joint_state strips the LEADING joint_pos(9)+joint_vel(9) block:
        # the action is EE-frame, so the ee_pose already in the vector carries
        # the kinematics, and joint space is redundant + scene-correlated (a
        # memorization handle). Mirrors RLActor/RLCritic's flag of the same name
        # so a Phase-3 and a Phase-4 policy can be given the same input.
        # It changes the robot-encoder in_dim, so a checkpoint trained with it
        # cannot be loaded into a model without it (and vice versa) — the run's
        # config.yaml is what keeps the two in sync.
        self.use_prev_act         = bool(use_prev_act)
        self.prev_act_dim         = int(prev_act_dim)
        self.drop_joint_state     = bool(drop_joint_state)
        self.joint_state_dim      = int(joint_state_dim)
        self.full_robot_state_dim = int(robot_state_dim)
        _tail = (self.full_robot_state_dim if self.use_prev_act
                 else self.full_robot_state_dim - self.prev_act_dim)
        _lead = self.joint_state_dim if self.drop_joint_state else 0
        effective_robot_dim = _tail - _lead

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
        # ----- GOAL-GRASP CONDITIONING (Phase 5) -----
        # THE reason Phase 5 exists. With four pinned grasps per scene the same
        # observation carries four different expert labels, so an unconditioned
        # regression has no option but to predict their mean — which is a valid
        # action for none of them. Telling the policy which grasp it is being
        # asked to reach is what makes (pc, rs, goal) -> action single-valued
        # again, and it is simultaneously the fix Phase 4 kept failing to find:
        # over runs 4-16 the policy closed within tolerance of the pinned grasp
        # on 0-8% of episodes because nothing in the observation named the pose
        # to arrive at (run 13 tried to force it via an auxiliary PREDICTION;
        # this hands it over as an INPUT).
        #
        # Sized at 128, not feature_dim: about a fifth of the fused vector, which
        # is enough not to be drowned by 512 dims of perception and not so much
        # that the head can solve the task by servoing to the goal and ignoring
        # the cloud (that shortcut exists — it would fly straight through the
        # human's hand, which is exactly what OMG's labels do not do, so the loss
        # pushes back; watch f_human_contact anyway).
        #
        # Changes policy_head's in_dim, so a Phase-4 checkpoint cannot strict-load
        # into a conditioned model — same situation as drop_joint_state and the
        # aux head, and the run's config.yaml is again the only record.
        self.grasp_cond = bool(grasp_cond)
        self.grasp_cond_dim = int(grasp_cond_dim) if self.grasp_cond else 0
        self.grasp_encoder = (GraspEncoder(in_dim=self.grasp_cond_dim,
                                           hidden_dim=grasp_hidden,
                                           feature_dim=grasp_feat_dim)
                              if self.grasp_cond else None)
        fused_dim = 2 * feature_dim + (grasp_feat_dim if self.grasp_cond else 0)

        self.policy_head = PolicyHead(
            in_dim=fused_dim,
            hidden=tuple(policy_hidden),
            action_dim=action_dim,
        )
        # ----- optional AUXILIARY GOAL-GRASP HEAD (run 13) -----
        # A second head on the SAME concatenated features predicting where the
        # goal grasp is relative to the current EE: [quat_wxyz(4) ‖ trans(3)].
        # GA-DDPG's `extra_pred` (core/networks.py), which we dropped when this
        # policy was written, and the evidence says that was the costly omission.
        #
        # Measured over runs 4/6/7/8/9: the policy's per-step rotation error is
        # 0.0230 rad and its episodes are ~25 steps, so 0.57 rad of wrist error
        # accumulates — and eval_min_rot is 0.559, i.e. rotation error integrates
        # essentially UNCORRECTED. Position error does not (0.0090 m/step would
        # integrate to 0.22 m; eval_min_pos is 0.076) because the wrist camera
        # sees where the object is. Nothing in the observation tells the policy
        # which ORIENTATION it is meant to arrive at, so this head makes that an
        # explicit prediction target and forces the scene features to carry it.
        #
        # Note `pc_pretrained` loads the CVPR2023 state_feat encoder, which WAS
        # trained with this aux loss — and we then fine-tune it for 100 epochs x
        # N iterations with no aux term, so whatever goal-pose structure it
        # carried is trained away. This puts the term back.
        #
        # Adding the head changes state_dict keys, so an aux checkpoint will not
        # strict-load into a model built without it (and vice versa) — exactly the
        # situation drop_joint_state already has, and the run's config.yaml is
        # again the only record.
        #
        # Phase-5 note: with grasp_cond on, this head's target IS its own input
        # (both are the goal grasp in the current EE frame), so it degenerates to
        # an identity map and teaches nothing. Phase-5 configs set
        # MODEL.aux_head: false / LOSS.aux_weight: 0.0. The two flags therefore
        # move together against run 16 and cannot be separated — a confound worth
        # stating rather than hiding.
        self.aux_dim = int(aux_dim) if aux_head else 0
        self.aux_head = (PolicyHead(in_dim=fused_dim,
                                    hidden=tuple(aux_hidden),
                                    action_dim=self.aux_dim)
                         if self.aux_dim else None)
        if self.aux_head is not None and self.grasp_cond:
            print("[model] WARNING aux_head is on together with grasp_cond: the "
                  "head's target is its own input, so the auxiliary loss is an "
                  "identity map and carries no signal.")
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
        """Trim the stored 32-D robot state down to the encoder's input.

        Trailing prev_action(6), dropped when use_prev_act is False: prev_act is
        ~0.9 correlated with the target action (the OMG path is smooth), so
        leaving it in lets the policy 'copy' the previous action and ignore the
        point cloud (causal confusion / copycat).

        Leading joint_pos(9)+joint_vel(9), dropped when drop_joint_state is
        True: redundant with the ee_pose that follows it, and scene-correlated.

        Input is always the full robot_state_dim; this returns the
        encoder-sized slice.
        """
        hi = (self.full_robot_state_dim if self.use_prev_act
              else self.full_robot_state_dim - self.prev_act_dim)
        lo = self.joint_state_dim if self.drop_joint_state else 0
        return rs[..., lo:hi]

    # ----- forward (training) ----------------------------------------------
    def _features(self, pc: torch.Tensor, rs: torch.Tensor,
                  goal: torch.Tensor | None = None) -> torch.Tensor:
        """Concatenated [scene ‖ robot (‖ goal)] features — one PointNet++ pass."""
        scene = self.pc_encoder(pc)
        robot = self.robot_encoder(self._select_robot_state(rs))
        parts = [scene, robot]
        if self.grasp_encoder is not None:
            if goal is None:
                raise ValueError(
                    "this policy was built with MODEL.grasp_cond: true, so every "
                    "forward/predict needs the goal grasp in the current EE frame "
                    "[B, %d]. Passing None would silently condition on nothing."
                    % self.grasp_cond_dim)
            parts.append(self.grasp_encoder(goal[..., :self.grasp_cond_dim]))
        return torch.cat(parts, dim=-1)

    def forward(self, pc: torch.Tensor, rs: torch.Tensor,
                goal: torch.Tensor | None = None) -> torch.Tensor:
        """Raw action [B, 7]. Return type is UNCHANGED by the aux head, so every
        existing caller (predict, PolicyRunner, the evaluator, the rollouts) is
        untouched whether the head is present or not.

        `goal` is required iff the model was built with grasp_cond, and is the
        goal grasp expressed in the CURRENT EE frame — rot6d(6) + trans(3), which
        is what `regrasp_bc/dataset.goal_cond_from_state` produces. It changes every step
        even though the world-frame grasp does not, because the EE moves.
        """
        return self.policy_head(self._features(pc, rs, goal))

    def forward_aux(self, pc: torch.Tensor, rs: torch.Tensor,
                    goal: torch.Tensor | None = None):
        """(action [B, 7], goal [B, 7] or None) from ONE encoder pass.

        The goal's quaternion block is L2-normalized here rather than in the loss,
        so whatever reads the head gets a unit quaternion — GA-DDPG normalizes in
        the same place (core/networks.py `extra_pred`).
        """
        feat = self._features(pc, rs, goal)
        action = self.policy_head(feat)
        if self.aux_head is None:
            return action, None
        goal = self.aux_head(feat)
        if self.aux_dim == 7:
            goal = torch.cat(
                [F.normalize(goal[..., :4], p=2, dim=-1), goal[..., 4:]], dim=-1)
        return action, goal

    # ----- inference helper -------------------------------------------------
    @torch.no_grad()
    def predict(self, pc: torch.Tensor, rs: torch.Tensor,
                goal: torch.Tensor | None = None) -> torch.Tensor:
        """Run inference and return a deployable action.

        If a Normalizer is attached:
          1. robot_state is normalized before the forward pass,
          2. action channels 0..5 are denormalized after.
        Gripper logit (channel 6) is passed through sigmoid and thresholded.

        `goal` is NOT normalized — it is already an EE-relative displacement in
        metres and radians-free (rot6d), i.e. the same kind of quantity the action
        channels are, and centred near zero by construction. Normalizing it would
        also make it depend on statistics the real robot cannot reproduce.
        """
        self.eval()
        rs_in = self.normalizer.normalize_state(rs) if self.normalizer is not None else rs
        raw   = self.forward(pc, rs_in, goal)
        if self.normalizer is not None:
            raw = self.normalizer.denormalize_action(raw)
        cont    = raw[..., :6]
        gripper = (torch.sigmoid(raw[..., 6:7]) > 0.5).float()
        return torch.cat([cont, gripper], dim=-1)

    # ----- parameter count for sanity --------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if (p.requires_grad or not trainable_only))
