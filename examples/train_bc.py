"""
CLI entry point for Phase-1 BC training.

Wires up examples/configs/bc_phase1.yaml → datasets → BCPolicy → BCTrainer.

Usage:
    python examples/train_bc.py --cfg-file examples/configs/bc_phase1.yaml

    # quick override of the dataset paths and run name
    python examples/train_bc.py \
        --cfg-file examples/configs/bc_phase1.yaml \
        --train-h5 output/bc_dataset/train.h5 \
        --val-h5   output/bc_dataset/val.h5 \
        --run-name phase1_full

    # resume training from the last checkpoint of a previous run
    python examples/train_bc.py \
        --cfg-file examples/configs/bc_phase1.yaml \
        --run-name phase1_full \
        --resume   output/bc_runs/phase1_full/checkpoints/last.pt

Run dir layout (auto-created at output/bc_runs/<run_name>/):
    config.yaml         — the resolved config used for this run
    normalization.npz   — Normalizer (computed from train_h5, or copied from resume)
    log.csv             — one row per epoch with all metrics
    checkpoints/{last,best}.pt
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from handover_sim2real.bc import (
    BCDataset,
    BCPolicy,
    BCTrainer,
    Normalizer,
    compute_normalization_stats,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.train_h5:
        cfg["DATA"]["train_h5"] = args.train_h5
    if args.val_h5:
        cfg["DATA"]["val_h5"] = args.val_h5
    if args.device:
        cfg["TRAIN"]["device"] = args.device
    if args.num_epochs is not None:
        cfg["TRAIN"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        cfg["TRAIN"]["batch_size"] = args.batch_size
    if args.use_prev_act is not None:
        cfg.setdefault("MODEL", {})["use_prev_act"] = bool(args.use_prev_act)
    if args.pc_pretrained is not None:
        # "none"/"" disables pretrained init (train PC encoder from scratch).
        val = args.pc_pretrained.strip()
        cfg.setdefault("MODEL", {})["pc_pretrained"] = (None if val.lower() in ("none", "")
                                                        else val)
    if args.freeze_pc is not None:
        cfg.setdefault("MODEL", {})["freeze_pc"] = bool(args.freeze_pc)
    return cfg


def build_model(cfg: dict, normalizer: Normalizer) -> BCPolicy:
    m = cfg["MODEL"]
    d = cfg["DATA"]
    return BCPolicy(
        pc_channels        = int(d["pc_channels"]),
        robot_state_dim    = int(d["robot_state_dim"]),
        action_dim         = int(d["action_dim"]),
        feature_dim        = int(m["feature_dim"]),
        robot_hidden       = int(m["robot_hidden"]),
        policy_hidden      = tuple(m["policy_hidden"]),
        pointnet_scale     = int(m["pointnet_scale"]),
        pointnet_radius    = float(m["pointnet_radius"]),
        pointnet_nclusters = int(m["pointnet_nclusters"]),
        use_prev_act       = bool(m.get("use_prev_act", True)),
        freeze_pc          = bool(m.get("freeze_pc", False)),
        normalizer         = normalizer,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg-file",   required=True,
                   help="path to yaml config (e.g. examples/configs/bc_phase1.yaml)")
    p.add_argument("--run-name",   default=None,
                   help="subdir of output/bc_runs/; default: timestamp")
    p.add_argument("--train-h5",   default=None, help="override DATA.train_h5")
    p.add_argument("--dagger-h5",  nargs="*", default=None,
                   help="extra HDF5 files to aggregate with the train set (DAgger rounds)")
    p.add_argument("--val-h5",     default=None, help="override DATA.val_h5")
    p.add_argument("--device",     default=None, help="override TRAIN.device")
    p.add_argument("--num-epochs", type=int, default=None, help="override TRAIN.num_epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override TRAIN.batch_size")
    p.add_argument("--resume",     default=None,
                   help="path to a *.pt checkpoint to resume from")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-prev-act", dest="use_prev_act", action="store_true",
                   help="override MODEL.use_prev_act: keep prev_action(6) in the robot state")
    g.add_argument("--no-prev-act",  dest="use_prev_act", action="store_false",
                   help="override MODEL.use_prev_act: drop prev_action(6) (avoids the copycat shortcut)")
    p.set_defaults(use_prev_act=None)  # None => fall back to the config value

    p.add_argument("--pc-pretrained", default=None,
                   help="override MODEL.pc_pretrained: path to a state-feat checkpoint "
                        "to init the PointNet++ encoder — CVPR2023 handover (default) or "
                        "GA-DDPG grasp ('none' to disable)")
    gf = p.add_mutually_exclusive_group()
    gf.add_argument("--freeze-pc",    dest="freeze_pc", action="store_true",
                    help="override MODEL.freeze_pc: freeze the PC encoder (train robot MLP + head only)")
    gf.add_argument("--no-freeze-pc", dest="freeze_pc", action="store_false",
                    help="override MODEL.freeze_pc: fine-tune the PC encoder")
    p.set_defaults(freeze_pc=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = apply_overrides(load_cfg(args.cfg_file), args)

    set_seed(int(cfg["TRAIN"].get("seed", 0)))

    # Resolve run dir.
    if args.run_name is None:
        args.run_name = time.strftime("phase1_%Y%m%d_%H%M%S")
    run_dir = Path("output/bc_runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Assemble the (possibly multi-file) training set: the base DATA.train_h5
    # (str or list) plus any DAgger aggregate files passed via --dagger-h5.
    base = cfg["DATA"]["train_h5"]
    train_files = [base] if isinstance(base, str) else list(base)
    if args.dagger_h5:
        train_files += list(args.dagger_h5)
    # Record in config.yaml: keep a plain string for the single-file (normal)
    # case so existing tooling is unaffected; only DAgger runs store a list.
    cfg["DATA"]["train_h5"] = train_files if len(train_files) > 1 else train_files[0]

    # Save the resolved config so the run is reproducible from its directory.
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Normalization stats. If resuming, prefer the stats already in the run dir
    # so we don't drift from the checkpoint's training distribution.
    norm_path = run_dir / "normalization.npz"
    if args.resume and norm_path.exists():
        print(f"Loading normalization from {norm_path}")
        normalizer = Normalizer.load(norm_path)
    else:
        print(f"Computing normalization stats from {cfg['DATA']['train_h5']} ...")
        normalizer = compute_normalization_stats(cfg["DATA"]["train_h5"])
        normalizer.save(norm_path)
        print(f"Saved normalization to {norm_path}")

    # Datasets + loaders.
    train_ds = BCDataset(cfg["DATA"]["train_h5"], normalizer=normalizer)
    val_ds   = (BCDataset(cfg["DATA"]["val_h5"],   normalizer=normalizer)
                if cfg["DATA"].get("val_h5") and os.path.exists(cfg["DATA"]["val_h5"])
                else None)
    print(f"Train steps: {len(train_ds)}  ({train_ds.num_episodes} episodes)")
    if len(train_files) > 1:
        for path, ne in train_ds.episode_counts():
            print(f"    {path}: {ne} episodes")
    if val_ds is not None:
        print(f"Val   steps: {len(val_ds)}  ({val_ds.num_episodes} episodes)")
    else:
        print("Val   steps: 0  (no val_h5)")

    bs       = int(cfg["TRAIN"]["batch_size"])
    nw       = int(cfg["TRAIN"]["num_workers"])
    pin      = (cfg["TRAIN"]["device"] != "cpu")
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=pin, drop_last=True)
    val_dl   = (DataLoader(val_ds, batch_size=bs, shuffle=False,
                           num_workers=nw, pin_memory=pin, drop_last=False)
                if val_ds is not None else None)

    # Model + trainer.
    model = build_model(cfg, normalizer)

    # Initialize the PointNet++ encoder from a pretrained checkpoint (fresh runs
    # only — on resume the trained weights come from the resume checkpoint).
    pc_pre = cfg.get("MODEL", {}).get("pc_pretrained")
    if pc_pre and not args.resume:
        if os.path.exists(pc_pre):
            model.load_pretrained_pc_encoder(pc_pre)
        else:
            print(f"WARNING: MODEL.pc_pretrained not found, training PC encoder "
                  f"from scratch: {pc_pre}")
    elif pc_pre and args.resume:
        print("Resuming: skipping pc_pretrained init (weights come from --resume).")

    print(f"Model parameters: {model.num_parameters():,} trainable "
          f"({model.num_parameters(trainable_only=False):,} total)")
    print(f"Robot state: use_prev_act={model.use_prev_act}  "
          f"(robot encoder input dim = "
          f"{model.full_robot_state_dim if model.use_prev_act else model.full_robot_state_dim - model.prev_act_dim})")
    print(f"PC encoder: {'FROZEN (robot MLP + head only)' if model._pc_frozen else 'trainable (fine-tune)'}")
    trainer = BCTrainer(model, train_dl, val_dl, cfg, run_dir=str(run_dir))

    if args.resume:
        trainer.resume_from(args.resume)

    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
