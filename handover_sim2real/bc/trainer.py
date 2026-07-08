"""
BCTrainer — epoch loop, validation, checkpointing, CSV logging.

Designed to be invoked by examples/train_bc.py with a config dict (whatever
shape examples/configs/bc_phase1.yaml ends up taking — we only read fields
under OPTIM / TRAIN / LOSS).

Run directory layout (auto-created):

    output/bc_runs/<run_name>/
        ├── checkpoints/
        │   ├── last.pt
        │   └── best.pt          # val_loss-best
        ├── normalization.npz    # the Normalizer used to train this model
        └── log.csv              # one row per epoch with all train/val metrics

Checkpoint payload:
    {
        "epoch": int,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() or None,
        "scaler":    scaler.state_dict() or None,   # for AMP
        "best_val_loss": float,
    }

The Normalizer is saved alongside (normalization.npz) rather than inside the
.pt file so it can be loaded without instantiating the full model.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .losses import bc_loss, bc_metrics

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(it, **kw):  # noqa: D401, ANN001
        return it


# ── helpers ──────────────────────────────────────────────────────────────────

class _MetricAccum:
    """Batch-size-weighted running mean for a dict of scalar metrics."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.n: int = 0

    def update(self, metrics: Dict[str, Any], batch_size: int) -> None:
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * batch_size
        self.n += batch_size

    def average(self) -> Dict[str, float]:
        if self.n == 0:
            return {}
        return {k: v / self.n for k, v in self.sums.items()}


def _make_scheduler(name: str, optimizer: torch.optim.Optimizer,
                    num_epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    raise ValueError(f"Unknown scheduler '{name}' (expected none/cosine/step)")


# ── trainer ──────────────────────────────────────────────────────────────────

class BCTrainer:
    """Phase-1 BC training loop.

    Args:
        model:        BCPolicy (or anything with the same forward signature).
        train_loader: DataLoader yielding (pc, robot_state, expert_action).
        val_loader:   Same shape; optional. Pass None to skip validation.
        cfg:          Nested dict with OPTIM / TRAIN / LOSS sections.
        run_dir:      Where to write checkpoints + log.csv. Created if missing.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 cfg: Dict[str, Any],
                 run_dir: str):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg

        self.device         = cfg["TRAIN"].get("device", "cuda")
        self.num_epochs     = int(cfg["TRAIN"]["num_epochs"])
        self.val_every      = int(cfg["TRAIN"].get("val_every", 1))
        self.save_every     = int(cfg["TRAIN"].get("save_every", 1))
        self.grad_clip      = float(cfg["TRAIN"].get("grad_clip", 0.0))   # 0 = off
        self.use_amp        = bool(cfg["TRAIN"].get("mixed_precision", False))
        self.gripper_weight = float(cfg["LOSS"].get("gripper_weight", 1.0))

        self.run_dir = Path(run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.csv"

        self.model.to(self.device)

        opt_kind = cfg["OPTIM"].get("kind", "adamw").lower()
        opt_cls  = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[opt_kind]
        # Only optimize parameters that require grad — lets freeze_pc actually
        # exclude the frozen PointNet++ encoder from the optimizer.
        self.optimizer = opt_cls(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(cfg["OPTIM"]["lr"]),
            weight_decay=float(cfg["OPTIM"].get("weight_decay", 0.0)),
        )
        self.scheduler = _make_scheduler(
            cfg["OPTIM"].get("scheduler", "none"),
            self.optimizer,
            self.num_epochs,
        )
        self.scaler = (torch.cuda.amp.GradScaler() if self.use_amp else None)

        # Resume state — filled by resume_from() if called.
        self.start_epoch:   int   = 0
        self.best_val_loss: float = float("inf")

        # Persist the normalizer alongside checkpoints (single source of truth).
        norm = getattr(self.model, "normalizer", None)
        if norm is not None:
            norm.save(self.run_dir / "normalization.npz")

    # ----- one epoch --------------------------------------------------------
    def _step_batch(self, batch, train: bool) -> tuple[Dict[str, torch.Tensor], Dict[str, float], int]:
        pc, rs, act = (t.to(self.device, non_blocking=True) for t in batch)
        autocast = torch.cuda.amp.autocast if (self.use_amp and self.device != "cpu") else _noop_ctx
        with autocast():
            out    = self.model(pc, rs)
            losses = bc_loss(out, act, gripper_weight=self.gripper_weight)
            metrics = bc_metrics(out, act)
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
        return losses, metrics, pc.shape[0]

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        acc = _MetricAccum()
        bar = tqdm(self.train_loader, desc=f"epoch {epoch:03d} train", leave=False)
        for batch in bar:
            losses, metrics, B = self._step_batch(batch, train=True)
            row = {k: v.item() for k, v in losses.items()}
            row.update(metrics)
            acc.update(row, B)
        return acc.average()

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        acc = _MetricAccum()
        bar = tqdm(self.val_loader, desc=f"epoch {epoch:03d}  val ", leave=False)
        for batch in bar:
            losses, metrics, B = self._step_batch(batch, train=False)
            row = {k: v.item() for k, v in losses.items()}
            row.update(metrics)
            acc.update(row, B)
        return acc.average()

    # ----- checkpointing ----------------------------------------------------
    def save_checkpoint(self, name: str, epoch: int) -> None:
        path = self.run_dir / "checkpoints" / name
        payload = {
            "epoch":          epoch,
            "model":          self.model.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler":         self.scaler.state_dict()    if self.scaler    is not None else None,
            "best_val_loss":  self.best_val_loss,
        }
        torch.save(payload, path)

    def resume_from(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.scaler is not None and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch   = int(ckpt.get("epoch", -1)) + 1
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        print(f"Resumed from {path}: epoch {self.start_epoch}, best_val_loss={self.best_val_loss:.4f}")

    # ----- CSV logging ------------------------------------------------------
    def _log_row(self, row: Dict[str, Any]) -> None:
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ----- main loop --------------------------------------------------------
    def train(self) -> None:
        # Fresh run (not resuming) into a dir that already has a log: the old
        # log is stale. Truncate it so we don't stack runs in one log.csv
        # (which makes the epoch column reset 0->N and distorts plots).
        if self.start_epoch == 0 and self.log_path.exists():
            self.log_path.unlink()
        print(f"Run dir : {self.run_dir}")
        print(f"Device  : {self.device}")
        print(f"Epochs  : {self.start_epoch} → {self.num_epochs}")
        print(f"AMP     : {self.use_amp}")
        print(f"Scheduler: {self.cfg['OPTIM'].get('scheduler', 'none')}")
        n_train_steps = len(self.train_loader)
        n_val_steps   = len(self.val_loader) if self.val_loader is not None else 0
        print(f"Steps/epoch: train={n_train_steps}, val={n_val_steps}")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)

            do_val = (self.val_loader is not None) and (epoch % self.val_every == 0)
            val_metrics = self.validate(epoch) if do_val else {}

            if self.scheduler is not None:
                self.scheduler.step()

            # Log row: prefix train_/val_, drop the loss-tensor distinction.
            lr_now = self.optimizer.param_groups[0]["lr"]
            row: Dict[str, Any] = {"epoch": epoch, "lr": lr_now, "wall_s": round(time.time() - t0, 2)}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}":   v for k, v in val_metrics.items()})
            self._log_row(row)

            # Console summary.
            msg = (f"epoch {epoch:03d}  "
                   f"train_total={train_metrics.get('total', float('nan')):.4f}  "
                   f"train_grip_acc={train_metrics.get('gripper_acc', float('nan')):.3f}")
            if val_metrics:
                msg += (f"  |  val_total={val_metrics.get('total', float('nan')):.4f}  "
                        f"val_grip_acc={val_metrics.get('gripper_acc', float('nan')):.3f}")
            msg += f"  ({row['wall_s']:.1f}s)"
            print(msg)

            # Checkpointing.
            if (epoch + 1) % self.save_every == 0 or (epoch + 1) == self.num_epochs:
                self.save_checkpoint("last.pt", epoch)
            if val_metrics and val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = float(val_metrics["total"])
                self.save_checkpoint("best.pt", epoch)


# ── small util: no-op context manager used when AMP is disabled ─────────────

class _noop_ctx:
    def __enter__(self):  return None
    def __exit__(self, *a): return False
