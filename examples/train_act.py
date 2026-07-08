"""
CLI entry point for Phase-2 ACT training.

Wires examples/configs/act_phase2.yaml → windowed datasets → ACTPolicy →
ACTTrainer. Mirrors examples/train_bc.py (same run-dir layout, normalization,
resume semantics); the differences are the windowed BCSequenceDataset and the
ACTPolicy / ACTTrainer.

Usage:
    python examples/train_act.py --cfg-file examples/configs/act_phase2.yaml \
        --train-h5 output/bc_dataset/train.h5 \
        --val-h5   output/bc_dataset/val.h5 \
        --run-name act_full

    # ablations (single model, post-hoc attribution)
    python examples/train_act.py --cfg-file examples/configs/act_phase2.yaml --history-len 1
    python examples/train_act.py --cfg-file examples/configs/act_phase2.yaml --no-cvae

Run dir layout (output/bc_runs/<run_name>/): config.yaml, normalization.npz,
log.csv, checkpoints/{last,best}.pt — identical to Phase-1.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from handover_sim2real.bc import (
    ACTPolicy,
    ACTTrainer,
    BCSequenceDataset,
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
    if args.history_len is not None:
        cfg["MODEL"]["history_len"] = args.history_len
    if args.chunk_len is not None:
        cfg["MODEL"]["chunk_len"] = args.chunk_len
    if args.use_cvae is not None:
        cfg["MODEL"]["use_cvae"] = bool(args.use_cvae)
    if args.pc_pretrained is not None:
        val = args.pc_pretrained.strip()
        cfg.setdefault("MODEL", {})["pc_pretrained"] = (None if val.lower() in ("none", "")
                                                        else val)
    if args.freeze_pc is not None:
        cfg.setdefault("MODEL", {})["freeze_pc"] = bool(args.freeze_pc)
    return cfg


def build_model(cfg: dict, normalizer: Normalizer) -> ACTPolicy:
    m, d = cfg["MODEL"], cfg["DATA"]
    return ACTPolicy(
        pc_channels        = int(d["pc_channels"]),
        robot_state_dim    = int(d["robot_state_dim"]),
        action_dim         = int(d["action_dim"]),
        feature_dim        = int(m["feature_dim"]),
        robot_hidden       = int(m["robot_hidden"]),
        d_model            = int(m["d_model"]),
        n_heads            = int(m["n_heads"]),
        enc_layers         = int(m["enc_layers"]),
        dec_layers         = int(m["dec_layers"]),
        cvae_enc_layers    = int(m.get("cvae_enc_layers", 2)),
        dropout            = float(m.get("dropout", 0.1)),
        history_len        = int(m["history_len"]),
        chunk_len          = int(m["chunk_len"]),
        latent_dim         = int(m["latent_dim"]),
        use_cvae           = bool(m.get("use_cvae", True)),
        use_prev_act       = bool(m.get("use_prev_act", False)),
        pointnet_scale     = int(m["pointnet_scale"]),
        pointnet_radius    = float(m["pointnet_radius"]),
        pointnet_nclusters = int(m["pointnet_nclusters"]),
        freeze_pc          = bool(m.get("freeze_pc", False)),
        normalizer         = normalizer,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg-file",   required=True)
    p.add_argument("--run-name",   default=None,
                   help="subdir of output/bc_runs/; default: timestamp")
    p.add_argument("--train-h5",   default=None, help="override DATA.train_h5")
    p.add_argument("--dagger-h5",  nargs="*", default=None,
                   help="extra HDF5 files to aggregate with the train set")
    p.add_argument("--val-h5",     default=None, help="override DATA.val_h5")
    p.add_argument("--device",     default=None, help="override TRAIN.device")
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--history-len", type=int, default=None, help="override MODEL.history_len (T)")
    p.add_argument("--chunk-len",   type=int, default=None, help="override MODEL.chunk_len (k)")
    p.add_argument("--resume",     default=None, help="path to a *.pt checkpoint to resume from")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-cvae", dest="use_cvae", action="store_true",
                   help="override MODEL.use_cvae: enable the CVAE latent + KL")
    g.add_argument("--no-cvae",  dest="use_cvae", action="store_false",
                   help="override MODEL.use_cvae: z=0 always (ablation)")
    p.set_defaults(use_cvae=None)

    p.add_argument("--pc-pretrained", default=None,
                   help="override MODEL.pc_pretrained ('none' to train PC encoder from scratch)")
    gf = p.add_mutually_exclusive_group()
    gf.add_argument("--freeze-pc",    dest="freeze_pc", action="store_true")
    gf.add_argument("--no-freeze-pc", dest="freeze_pc", action="store_false")
    p.set_defaults(freeze_pc=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = apply_overrides(load_cfg(args.cfg_file), args)

    set_seed(int(cfg["TRAIN"].get("seed", 0)))

    if args.run_name is None:
        args.run_name = time.strftime("act_%Y%m%d_%H%M%S")
    run_dir = Path("output/bc_runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Assemble the (possibly multi-file) training set.
    base = cfg["DATA"]["train_h5"]
    train_files = [base] if isinstance(base, str) else list(base)
    if args.dagger_h5:
        train_files += list(args.dagger_h5)
    cfg["DATA"]["train_h5"] = train_files if len(train_files) > 1 else train_files[0]

    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Normalization (identical to Phase-1: per-frame state, per-action[:6]).
    norm_path = run_dir / "normalization.npz"
    if args.resume and norm_path.exists():
        print(f"Loading normalization from {norm_path}")
        normalizer = Normalizer.load(norm_path)
    else:
        print(f"Computing normalization stats from {cfg['DATA']['train_h5']} ...")
        normalizer = compute_normalization_stats(cfg["DATA"]["train_h5"])
        normalizer.save(norm_path)
        print(f"Saved normalization to {norm_path}")

    T = int(cfg["MODEL"]["history_len"])
    k = int(cfg["MODEL"]["chunk_len"])
    train_ds = BCSequenceDataset(cfg["DATA"]["train_h5"], history_len=T, chunk_len=k,
                                 normalizer=normalizer)
    val_ds = (BCSequenceDataset(cfg["DATA"]["val_h5"], history_len=T, chunk_len=k,
                                normalizer=normalizer)
              if cfg["DATA"].get("val_h5") and os.path.exists(cfg["DATA"]["val_h5"])
              else None)
    print(f"Windows  T={T}  k={k}")
    print(f"Train steps: {len(train_ds)}  ({train_ds.num_episodes} episodes)")
    if len(train_files) > 1:
        for path, ne in train_ds.episode_counts():
            print(f"    {path}: {ne} episodes")
    print(f"Val   steps: {len(val_ds)}  ({val_ds.num_episodes} episodes)" if val_ds
          else "Val   steps: 0  (no val_h5)")

    bs  = int(cfg["TRAIN"]["batch_size"])
    nw  = int(cfg["TRAIN"]["num_workers"])
    pin = (cfg["TRAIN"]["device"] != "cpu")
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=pin, drop_last=True)
    val_dl = (DataLoader(val_ds, batch_size=bs, shuffle=False,
                         num_workers=nw, pin_memory=pin, drop_last=False)
              if val_ds is not None else None)

    model = build_model(cfg, normalizer)

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
    print(f"CVAE: {'on' if model.use_cvae else 'off (z=0)'}  |  "
          f"PC encoder: {'FROZEN' if model._pc_frozen else 'fine-tune'}")

    trainer = ACTTrainer(model, train_dl, val_dl, cfg, run_dir=str(run_dir))
    if args.resume:
        trainer.resume_from(args.resume)
    trainer.train()
    print("Done.")


if __name__ == "__main__":
    main()
