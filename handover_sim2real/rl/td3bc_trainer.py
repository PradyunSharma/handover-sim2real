"""
Phase-3 RL learner — the TD3 + BC blend (the GA-DDPG-style update).

One `update(batch)` does:

  Critic (twin, clipped double-Q, TD3 target smoothing):
      a'      = clamp( actor_target(s') + clamp(noise, ±c), ±A )
      y_bell  = r + γ · (1 − terminal) · min(Q1', Q2')(s', a')
      y       = (1 − mc_blend) · y_bell + mc_blend · mc_return
      L_critic = SmoothL1(Q1, y) + SmoothL1(Q2, y) + aux_w · SmoothL1(auxᵛ, goal)

  Actor (delayed; deterministic policy-gradient blended with BC — GA-DDPG form):
      L_actor = − λ · Q1(s, π(s))
                + (1 − λ) · bc_w   · SmoothL1(π(s)[:6], a_expert)|expert       (pose)
                + (1 − λ) · grip_w · BCE(π(s)[6], gripper_label)|gripper_flag   (close)
                + aux_w · SmoothL1(auxᵖ, goal)
      λ ramps `mix_start → mix_end` over `mix_ramp` updates (BC-dominated early).
      Pose BC is applied only on OMG-labelled (expert_flag) transitions; the
      gripper BCE only where the grasp pose is known (gripper_flag). The gripper
      label is proximity-synthesized (open far / close near) so it AGREES with
      the reward — the policy still executes its own logit at deployment.
      (Set `pg_normalize=True` for the TD3+BC λ = α / mean|Q| variant instead.)

  Goal-auxiliary (aux_w>0): a small head on each net's scene encoder regresses
  the EE-relative final grasp pose (pos+rot6d = `goal_pose`); pure regularizer
  that shapes both PointNet++ encoders — the output is never used for control/value.

  Targets: Polyak soft-update after every delayed actor step.

Because the clock is in the state, `terminal` already covers the horizon limit
(no separate truncation handling). Everything is in normalized action space; the
learner never touches the sim or denormalization.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from handover_sim2real.bc.losses import pose_pm_loss


class TD3BCTrainer:
    def __init__(self, actor, critic, normalizer, cfg: dict, device: str = "cuda"):
        self.actor  = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target  = copy.deepcopy(self.actor).eval()
        self.critic_target = copy.deepcopy(self.critic).eval()
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.normalizer = normalizer
        self.device     = device
        # real-unit action stats for the point-matching pose loss (it needs metres/
        # radians; the buffer stores NORMALIZED actions, so we denormalize in-loss).
        if normalizer is not None:
            self.a_mean_t = torch.as_tensor(normalizer.action_mean, dtype=torch.float32, device=device)
            self.a_std_t  = torch.as_tensor(normalizer.action_std,  dtype=torch.float32, device=device)
        else:
            self.a_mean_t = torch.zeros(6, device=device)
            self.a_std_t  = torch.ones(6, device=device)

        r = cfg["RL"]
        # pose-BC loss form: "pm" = point-matching on gripper control points (proper
        # SE(3) metric, weights wrist rotation correctly); "smooth_l1" = the old L1 on
        # the raw normalized [Δpos ‖ Δeuler] (sums metres+radians, weak on rotation).
        self.pose_loss    = str(r.get("pose_loss", "smooth_l1"))
        self.gamma        = float(r.get("gamma", 0.95))
        self.tau          = float(r.get("tau", 5e-3))
        self.policy_noise = float(r.get("policy_noise", 0.2))
        self.noise_clip   = float(r.get("noise_clip", 0.5))
        self.act_limit    = float(r.get("act_limit", 5.0))
        self.policy_delay = int(r.get("policy_delay", 2))
        self.mc_blend     = float(r.get("mc_blend", 0.0))
        self.bc_weight    = float(r.get("bc_weight", 1.0))
        self.gripper_bc_weight = float(r.get("gripper_bc_weight", 1.0))  # gripper BCE
        # class-balance the gripper BCE: "close" (near-grasp) is the rare label in
        # every batch (the policy spends most steps far), so plain BCE learns only
        # "open" and the gripper stays pinned open (BC baseline: gprob==1.0 the whole
        # rollout, never fires). Upweight close examples to ~match the open mass,
        # capped at this factor.
        self.gripper_close_weight_max = float(r.get("gripper_close_weight_max", 10.0))
        # label-smooth the gripper BCE (targets 1->1-eps, 0->eps): the far/"open"
        # label dominates so plain BCE drives the logit to +inf until it saturates
        # (sigmoid=1, vanishing grad) and the gripper is STUCK open — the run9 late
        # decline. Smoothing bounds the logit at ~logit(1-eps) so it stays responsive
        # and close states can still fire it. 0 = off.
        self.gripper_label_smooth = float(r.get("gripper_label_smooth", 0.0))
        self.aux_weight   = float(r.get("aux_weight", 0.0))  # goal-auxiliary grasp-pose loss
        # action-magnitude regularizer on the actor's POSE output: L2 penalty
        # w·mean(a_pose²) added to the actor loss. Bounds the post-curriculum |a_pose|
        # DRIFT that inflates ~0.7→2.0 in EVERY run (→ overshooting/aggressive closes
        # that collide/drop and collapse the peak). Quadratic so it self-targets the
        # large drifted actions and is gentle in the healthy ~0.7 regime; the strong
        # BC term keeps legitimate motion. 0 = off. First isolated test of the drift.
        self.action_reg_weight = float(r.get("action_reg_weight", 0.0))
        # PG/BC blend: GA-DDPG mix schedule (default) or TD3+BC normalization.
        self.pg_normalize = bool(r.get("pg_normalize", False))
        self.alpha        = float(r.get("alpha", 2.5))
        self.mix_start    = float(r.get("mix_start", 0.1))
        self.mix_end      = float(r.get("mix_end", 0.2))
        self.mix_ramp     = int(r.get("mix_ramp", 50000))
        # potential-based reward shaping (Ng et al. 1999) on the DISTANCE to the OMG
        # grasp — an ADDITIVE, provably policy-invariant learning aid on TOP of the
        # sparse close reward. Φ(s) = −(w_pos·‖ee→grasp‖ + w_rot·angle(ee,grasp)),
        # and the shaping term γ·Φ(s')·(1−term) − Φ(s) is added to the Bellman target
        # (the (1−term) is the absorbing-Φ=0 convention). Both weights 0 = disabled
        # (the run9/11/12/13 sparse reward). NOTE: shaping enters the BELLMAN target
        # only; mc_return stays the sparse return-to-go, so with mc_blend>0 the target
        # is a (bounded) mix — lower mc_blend for a purer shaping signal if desired.
        self.shaping_pos_weight = float(r.get("shaping_pos_weight", 0.0))
        self.shaping_rot_weight = float(r.get("shaping_rot_weight", 0.0))
        self.shaping_on = (self.shaping_pos_weight > 0.0 or self.shaping_rot_weight > 0.0)

        self.actor_optim  = torch.optim.Adam(
            self.actor.parameters(),  lr=float(r.get("actor_lr", 3e-4)))
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=float(r.get("critic_lr", 3e-4)))

        self.grad_clip  = float(r.get("grad_clip", 1.0))
        self.update_step = 0

    # ----- helpers ---------------------------------------------------------
    def _norm_state(self, rs: torch.Tensor) -> torch.Tensor:
        return self.normalizer.normalize_state(rs) if self.normalizer is not None else rs

    def _potential(self, goal_pose: torch.Tensor) -> torch.Tensor:
        """Φ(s) = −(w_pos·pos_dist + w_rot·rot_angle) from the EE-relative grasp pose
        `goal_pose` [B,9] = [pos(3) ‖ rot6d(6)]. pos_dist = ‖pos‖; rot_angle is the
        geodesic angle of the rot6d (Gram-Schmidt → R, angle=arccos((tr R−1)/2))."""
        pos = goal_pose[:, :3].norm(dim=1, keepdim=True)               # [B,1]
        a1, a2 = goal_pose[:, 3:6], goal_pose[:, 6:9]
        b1 = a1 / (a1.norm(dim=1, keepdim=True) + 1e-8)
        a2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = a2 / (a2.norm(dim=1, keepdim=True) + 1e-8)
        b3 = torch.cross(b1, b2, dim=1)
        trace = b1[:, 0] + b2[:, 1] + b3[:, 2]                         # diag of [b1 b2 b3]
        rot = torch.arccos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)).unsqueeze(1)  # [B,1]
        return -(self.shaping_pos_weight * pos + self.shaping_rot_weight * rot)

    def _mix_ratio(self) -> float:
        if self.mix_ramp <= 0:
            return self.mix_end
        f = min(self.update_step / self.mix_ramp, 1.0)
        return self.mix_start + (self.mix_end - self.mix_start) * f

    # ----- one gradient update --------------------------------------------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        pc, rs        = batch["pc"], self._norm_state(batch["rs"])
        remain        = batch["remain"]
        action        = batch["action"]
        reward        = batch["reward"]
        next_pc       = batch["next_pc"]
        next_rs       = self._norm_state(batch["next_rs"])
        next_remain   = batch["next_remain"]
        terminal      = batch["terminal"]
        mc_return     = batch["mc_return"]
        expert_action = batch["expert_action"]
        expert_flag   = batch["expert_flag"]
        goal_pose     = batch["goal_pose"]
        expert_gripper = batch.get("expert_gripper")   # proximity gripper label
        gripper_flag   = batch.get("gripper_flag")     # label validity
        perturb_flag   = batch.get("perturb_flag")     # DART: artificial jump row

        # ----- critic -----
        with torch.no_grad():
            next_a = self.actor_target(next_pc, next_rs, next_remain)
            noise  = (torch.randn_like(next_a) * self.policy_noise
                      ).clamp(-self.noise_clip, self.noise_clip)
            next_a = (next_a + noise).clamp(-self.act_limit, self.act_limit)
            q1_t, q2_t = self.critic_target(next_pc, next_rs, next_a, next_remain)
            min_q = torch.min(q1_t, q2_t)
            # potential-based shaping added to the immediate reward: F = γ·Φ(s')·
            # (1−term) − Φ(s). Policy-invariant (telescopes to an endpoint constant),
            # so it can't be farmed by hovering; the sparse +1 still makes closing
            # optimal. Sparse reward untouched (buf_pos/success stay sparse).
            if self.shaping_on:
                shaping = (self.gamma * self._potential(batch["next_goal_pose"])
                           * (1.0 - terminal) - self._potential(goal_pose))
            else:
                shaping = 0.0
            y_bell = reward + shaping + self.gamma * (1.0 - terminal) * min_q
            y = (1.0 - self.mc_blend) * y_bell + self.mc_blend * mc_return

        if self.aux_weight > 0.0:
            q1, q2, aux_c = self.critic(pc, rs, action, remain, return_aux=True)
            aux_loss_c = F.smooth_l1_loss(aux_c, goal_pose)
        else:
            q1, q2 = self.critic(pc, rs, action, remain)
            aux_loss_c = torch.zeros((), device=q1.device)
        # DART: exclude perturbed transitions from the Bellman fit — their stored
        # action is an artificial random jump, so Q(s,a) → r+γQ(s') there is a
        # meaningless target that would corrupt the critic. They are kept for the
        # actor BC (the next-step plan-tracking label teaches the recovery). No-op
        # when nothing is perturbed (DART off / demo batch): keep == all-ones, so
        # the masked mean equals the plain mean.
        if perturb_flag is not None:
            keep  = (perturb_flag < 0.5).float()              # [B,1]
            denom = keep.sum().clamp_min(1.0)
            q_fit = (((F.smooth_l1_loss(q1, y, reduction="none")
                       + F.smooth_l1_loss(q2, y, reduction="none")) * keep).sum()
                     / denom)
        else:
            q_fit = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)
        critic_loss = q_fit + self.aux_weight * aux_loss_c

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()

        out = {"critic_loss": float(critic_loss.detach()),
               "q_mean": float(q1.mean().detach()),
               "target_mean": float(y.mean().detach()),
               "aux_loss_c": float(aux_loss_c.detach())}

        # ----- delayed actor + target soft-update -----
        if self.update_step % self.policy_delay == 0:
            if self.aux_weight > 0.0:
                a_pi, aux_a = self.actor(pc, rs, remain, return_aux=True)
                aux_loss_a = F.smooth_l1_loss(aux_a, goal_pose)
            else:
                a_pi = self.actor(pc, rs, remain)
                aux_loss_a = torch.zeros((), device=a_pi.device)
            # PG drives the POSE channels only. The gripper logit a_pi[:, 6] is
            # fed to the critic detached so the deterministic policy-gradient
            # -(lam*Q) cannot flow into it: an unclamped near-binary logit riding
            # dQ/dlogit into the critic's OOD region blew up to ~5e4 (rl_run4:
            # q_pi->950 in lockstep, then both collapsed at iter ~95, close-rate
            # stuck at 0). The gripper is trained by BCE toward the proximity
            # label only, which already encodes "close near the grasp" == the
            # reward-earning behavior, so no PG signal is lost.
            a_for_q = torch.cat([a_pi[:, :6], a_pi[:, 6:7].detach()], dim=1)
            q1_pi = self.critic.q1_only(pc, rs, a_for_q, remain)

            if self.pg_normalize:
                lam = self.alpha / q1_pi.abs().mean().detach().clamp_min(1e-6)
                pg_loss = -(lam * q1_pi).mean()
                bc_scale = 1.0
            else:
                lam = self._mix_ratio()
                pg_loss = -(lam * q1_pi).mean()
                bc_scale = (1.0 - lam)

            # Pose BC supervises a_pi[:, :6] toward the OMG label on labelled
            # transitions (the OMG approach carries no useful gripper signal).
            mask = (expert_flag > 0.5).squeeze(-1)
            if mask.any():
                if self.pose_loss == "pm":
                    # denormalize to real metres/radians, then point-match the
                    # gripper control points (proper 6-DoF metric; fixes rotation).
                    pred_real = a_pi[mask][:, :6] * self.a_std_t + self.a_mean_t
                    tgt_real  = expert_action[mask] * self.a_std_t + self.a_mean_t
                    bc_loss = pose_pm_loss(pred_real, tgt_real)
                else:
                    bc_loss = F.smooth_l1_loss(a_pi[mask][:, :6], expert_action[mask])
            else:
                bc_loss = a_pi.sum() * 0.0   # keep graph valid, zero contribution

            # Gripper BCE supervises the logit a_pi[:, 6] toward the proximity-
            # synthesized label (target = P(open): 1 far, 0 near) wherever the
            # grasp pose is known. It AGREES with the close reward, so it is a
            # dense accelerant, not a competing signal.
            grip_loss = torch.zeros((), device=a_pi.device)
            n_gripper = 0
            if self.gripper_bc_weight > 0.0 and gripper_flag is not None:
                gmask = (gripper_flag > 0.5).squeeze(-1)
                n_gripper = int(gmask.sum())
                if n_gripper > 0:
                    tgt = expert_gripper[gmask].squeeze(-1)   # P(open): 0 close, 1 open
                    is_close = tgt < 0.5
                    n_close = int(is_close.sum())
                    n_open = n_gripper - n_close
                    # weight close examples so their total mass ~= the open mass
                    if 0 < n_close < n_gripper:
                        w_close = min(n_open / n_close, self.gripper_close_weight_max)
                    else:
                        w_close = 1.0
                    w = torch.where(is_close, a_pi.new_tensor(w_close),
                                    a_pi.new_tensor(1.0))
                    # label smoothing: 1->1-eps (open), 0->eps (close) so the logit
                    # converges to a bounded, responsive value instead of saturating
                    # open (the run9 gripper drift). eps=0 recovers hard targets.
                    if self.gripper_label_smooth > 0.0:
                        eps = self.gripper_label_smooth
                        tgt = tgt * (1.0 - 2.0 * eps) + eps
                    grip_loss = F.binary_cross_entropy_with_logits(
                        a_pi[gmask][:, 6], tgt, weight=w)

            # bound the |a_pose| drift (quadratic → self-targets the large actions)
            action_reg = (self.action_reg_weight * a_pi[:, :6].pow(2).mean()
                          if self.action_reg_weight > 0.0
                          else torch.zeros((), device=a_pi.device))
            actor_loss = (pg_loss
                          + bc_scale * self.bc_weight * bc_loss
                          + bc_scale * self.gripper_bc_weight * grip_loss
                          + self.aux_weight * aux_loss_a
                          + action_reg)

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optim.step()

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor,  self.actor_target)

            out.update({"actor_loss": float(actor_loss.detach()),
                        "pg_loss":   float(pg_loss.detach()),
                        "bc_loss":   float(bc_loss.detach()) if torch.is_tensor(bc_loss) else 0.0,
                        "grip_loss": float(grip_loss.detach()),
                        "action_reg": float(action_reg.detach()),
                        "aux_loss_a": float(aux_loss_a.detach()),
                        "lam":       float(lam),
                        "n_expert":  int(mask.sum()),
                        "n_gripper": n_gripper,
                        # diagnostics: Q on the POLICY's own action (the OOD gap vs
                        # q_mean on stored actions), and actor-output health.
                        "q_pi":       float(q1_pi.mean().detach()),
                        "a_absmean":  float(a_pi[:, :6].detach().abs().mean()),
                        "grip_logit": float(a_pi[:, 6].detach().mean())})

        self.update_step += 1
        return out

    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        # keep buffers (e.g. BN running stats) in sync
        for b, bt in zip(net.buffers(), target.buffers()):
            bt.data.copy_(b.data)

    # ----- checkpoint ------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim":   self.actor_optim.state_dict(),
            "critic_optim":  self.critic_optim.state_dict(),
            "update_step":   self.update_step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_target.load_state_dict(sd["actor_target"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.actor_optim.load_state_dict(sd["actor_optim"])
        self.critic_optim.load_state_dict(sd["critic_optim"])
        self.update_step = int(sd.get("update_step", 0))
