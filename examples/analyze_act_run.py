"""
Analyze a finished (or in-progress) Phase-2 ACT training run.

The Phase-2 analog of examples/analyze_bc_run.py. Same two analyses, adapted to
the temporal / action-chunking ACTPolicy:

  curves   — read log.csv and plot loss / metric curves over epochs, *including*
             the CVAE KL term (watch for posterior collapse: KL → 0). No GPU.

  predict  — load the checkpoint and reproduce the *deployed* per-step action on
             episodes from a dataset (val by default): for each step it stacks
             the last T observation frames, predicts a k-action chunk, and runs
             the run's EXEC strategy (temporal ensembling by default) to get the
             single action that would actually be executed — then compares that
             to the expert action stored in the HDF5. This is the teacher-forced
             (dataset-state) analog of the closed-loop rollout.

Usage:
    python examples/analyze_act_run.py --run-dir output/bc_runs/act_run1 --mode curves
    python examples/analyze_act_run.py --run-dir output/bc_runs/act_run1 \
        --mode predict --split val --episode 0
    python examples/analyze_act_run.py --run-dir output/bc_runs/act_run1 --mode both
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling example modules

import h5py
import numpy as np
import torch

# Reuse the Phase-1 analyzer's run-dir helpers + interactive viewer (no sim imports).
from analyze_bc_run import (          # noqa: E402
    load_run_config, pick_dataset_path, _read_log_csv, show_predictions,
)


# ── curves (adds the KL panel) ───────────────────────────────────────────────

def analyze_curves(run_dir: Path, save: bool) -> None:
    import matplotlib.pyplot as plt

    log_path = run_dir / "log.csv"
    if not log_path.exists():
        raise SystemExit(f"No log.csv in {run_dir}")
    cols = _read_log_csv(log_path)
    epoch = cols["epoch"]
    print(f"Loaded {len(epoch)} epochs from {log_path}")

    print(f"\nLast epoch ({int(epoch[-1])}):")
    for c in ["train_total", "val_total", "train_pose_l1", "val_pose_l1",
              "train_gripper_acc", "val_gripper_acc", "train_kl_loss"]:
        if c in cols:
            print(f"  {c:18s} = {cols[c][-1]:.4f}")
    if "val_total" in cols and np.isfinite(cols["val_total"]).any():
        best_i = int(np.nanargmin(cols["val_total"]))
        print(f"\nBest val_total = {cols['val_total'][best_i]:.4f} "
              f"at epoch {int(epoch[best_i])}")
    if "train_kl_loss" in cols and np.nanmax(cols["train_kl_loss"]) < 1e-3:
        print("NOTE: train_kl_loss ≈ 0 → CVAE latent collapsed (benign with a "
              "deterministic teacher; the decoder is effectively deterministic).")

    panels = [
        ("Total loss",            ["train_total", "val_total"]),
        ("Pose loss (SmoothL1)",  ["train_pose_loss", "val_pose_loss"]),
        ("Gripper loss (BCE)",    ["train_gripper_loss", "val_gripper_loss"]),
        ("KL (CVAE)",             ["train_kl_loss", "val_kl_loss"]),
        ("Pose L1 (normalized)",  ["train_pose_l1", "val_pose_l1"]),
        ("Gripper accuracy",      ["train_gripper_acc", "val_gripper_acc"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (title, names) in zip(axes.flat, panels):
        for name in names:
            if name in cols:
                ax.plot(epoch, cols[name], label=name)
        ax.set_title(title); ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"ACT training curves — {run_dir.name}", fontsize=13)
    plt.tight_layout()
    if save:
        out = run_dir / "curves.png"
        fig.savefig(out, dpi=120)
        print(f"\nSaved {out}")
    plt.show()


# ── predict ──────────────────────────────────────────────────────────────────

def build_and_load_model(cfg: dict, run_dir: Path, device: str, ckpt: str):
    from handover_sim2real.bc import ACTPolicy, Normalizer

    norm_path = run_dir / "normalization.npz"
    normalizer = Normalizer.load(norm_path) if norm_path.exists() else None
    if normalizer is None:
        print("WARNING: no normalization.npz — predictions will be in normalized units!")

    m, d = cfg["MODEL"], cfg["DATA"]
    model = ACTPolicy(
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
        normalizer         = normalizer,
    ).to(device)

    ckpt_path = ckpt or str(run_dir / "checkpoints" / "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = str(run_dir / "checkpoints" / "last.pt")
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"Loaded checkpoint {ckpt_path} (epoch {payload.get('epoch', '?')})")
    return model


def _stack_history(buf: list[np.ndarray], T: int) -> np.ndarray:
    """Last T entries, oldest→newest; left-pad by repeating the oldest."""
    recent = buf[-T:]
    if len(recent) < T:
        recent = [recent[0]] * (T - len(recent)) + recent
    return np.stack(recent, axis=0)


@torch.no_grad()
def predict_episode(model, device, pcs, rss, T, k, exec_cfg):
    """Reproduce the deployed per-step action over one episode.

    pcs[T_ep,N,5], rss[T_ep,32] raw. Mirrors rollout_act_policy: ring buffer of
    the last T frames → chunk → EXEC strategy. Returns pred[T_ep,7] real units.
    """
    from handover_sim2real.bc import TemporalEnsembler
    mode = exec_cfg.get("mode", "ensemble")
    ens = (TemporalEnsembler(chunk_len=k, m=float(exec_cfg.get("ensemble_m", 0.01)))
           if mode == "ensemble" else None)
    if ens is not None:
        ens.reset()
    pending: list[np.ndarray] = []

    pc_buf, rs_buf, out = [], [], []
    for t in range(len(pcs)):
        pc_buf.append(pcs[t]); rs_buf.append(rss[t])
        if mode == "ensemble" or not pending:
            pc_hist = _stack_history(pc_buf, T)[None]
            rs_hist = _stack_history(rs_buf, T)[None]
            pc_t = torch.from_numpy(pc_hist).float().to(device)
            rs_t = torch.from_numpy(rs_hist).float().to(device)
            chunk = model.predict(pc_t, rs_t)[0].cpu().numpy()   # [k,7], ch6=prob
        if mode == "ensemble":
            action = ens.step(chunk)
        else:
            if not pending:
                pending = [a.copy() for a in chunk]
            action = pending.pop(0)
            action[6] = 1.0 if action[6] >= 0.5 else 0.0
        out.append(action)
    return np.stack(out)  # [T_ep, 7]


def analyze_predict(run_dir: Path, cfg: dict, dataset_path: str,
                    episode: int | None, device: str, ckpt: str, save: bool) -> None:
    model = build_and_load_model(cfg, run_dir, device, ckpt)
    T = int(cfg["MODEL"]["history_len"])
    k = int(cfg["MODEL"]["chunk_len"])
    exec_cfg = cfg.get("EXEC", {"mode": "ensemble", "ensemble_m": 0.01})
    print(f"EXEC: {exec_cfg.get('mode')}  T={T} k={k}")

    with h5py.File(dataset_path, "r") as f:
        keys = sorted(kk for kk in f.keys() if kk.startswith("episode_"))
        if not keys:
            raise SystemExit(f"No episodes in {dataset_path}")

        print(f"\nPer-episode error on {dataset_path} ({len(keys)} episodes):")
        print(f"  {'episode':>8s}  {'steps':>5s}  {'pos_l1(m)':>10s}  "
              f"{'rot_l1(rad)':>11s}  {'grip_acc':>8s}")
        agg_pos, agg_rot, agg_grip, agg_n = [], [], [], 0
        per_ep, scene_of = {}, {}
        for key in keys:
            pcs = f[key]["point_clouds"][:]
            rss = f[key]["robot_states"][:]
            exp = f[key]["expert_actions"][:]
            pred = predict_episode(model, device, pcs, rss, T, k, exec_cfg)

            pos_l1 = np.abs(pred[:, :3] - exp[:, :3]).mean()
            rot_l1 = np.abs(pred[:, 3:6] - exp[:, 3:6]).mean()
            grip_acc = (pred[:, 6] == exp[:, 6]).mean()
            per_ep[key] = (pred, exp)
            scene_of[key] = int(f[key].attrs.get("scene_idx", -1))
            ep_idx = int(key.split("_")[1])
            print(f"  {ep_idx:8d}  {len(exp):5d}  {pos_l1:10.4f}  "
                  f"{rot_l1:11.4f}  {grip_acc:8.3f}")
            agg_pos.append(pos_l1 * len(exp))
            agg_rot.append(rot_l1 * len(exp))
            agg_grip.append(grip_acc * len(exp))
            agg_n += len(exp)

        print(f"\n  WEIGHTED MEAN  pos_l1={sum(agg_pos)/agg_n:.4f} m  "
              f"rot_l1={sum(agg_rot)/agg_n:.4f} rad  "
              f"grip_acc={sum(agg_grip)/agg_n:.3f}")

    show_predictions(run_dir, keys, per_ep, scene_of, episode, save)


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="output/bc_runs/<name>")
    p.add_argument("--mode", default="both", choices=["curves", "predict", "both"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--dataset", default=None, help="explicit HDF5 path (overrides --split)")
    p.add_argument("--episode", type=int, default=None,
                   help="episode index to start the per-component viewer on")
    p.add_argument("--checkpoint", default=None,
                   help="explicit .pt path; default best.pt then last.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save", action="store_true",
                   help="also save the figures as PNGs into the run dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    cfg = load_run_config(run_dir)

    if args.mode in ("curves", "both"):
        analyze_curves(run_dir, save=args.save)
    if args.mode in ("predict", "both"):
        dataset_path = pick_dataset_path(cfg, run_dir, args.split, args.dataset)
        analyze_predict(run_dir, cfg, dataset_path, args.episode,
                        args.device, args.checkpoint, save=args.save)


if __name__ == "__main__":
    main()
