"""
ACTTrainer — Phase-2 training loop.

Subclasses BCTrainer to reuse its epoch loop, validation, checkpointing, CSV
logging, scheduler and AMP machinery unchanged. Only the per-batch step differs:
the batch is a 4-tuple (pc_hist, rs_hist, action_chunk, chunk_mask), the model
returns (pred, mu, logvar), and the loss is act_loss (masked SmoothL1 + masked
BCE + KL). kl_loss flows into the CSV log automatically via the losses dict.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .losses_act import act_loss, act_metrics
from .trainer import BCTrainer, _noop_ctx


class ACTTrainer(BCTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = float(self.cfg["LOSS"].get("kl_weight", 10.0))

    def _step_batch(self, batch, train: bool):
        pc_hist, rs_hist, chunk, mask = (t.to(self.device, non_blocking=True) for t in batch)
        autocast = (torch.cuda.amp.autocast
                    if (self.use_amp and self.device != "cpu") else _noop_ctx)
        with autocast():
            # action_chunk fed only in train mode → CVAE posterior is used for KL;
            # eval uses z=0 (prior), matching deployment.
            pred, mu, logvar = self.model(pc_hist, rs_hist,
                                          action_chunk=chunk if train else None)
            losses: Dict[str, torch.Tensor] = act_loss(
                pred, chunk, mask, mu, logvar,
                gripper_weight=self.gripper_weight, kl_weight=self.kl_weight)
            metrics: Dict[str, float] = act_metrics(pred, chunk, mask)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(losses["total"]).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

        return losses, metrics, pc_hist.shape[0]
