"""
Analyze a finished (or in-progress) Phase-1 BC training run.

Two analyses:

  curves   — read log.csv and plot loss / metric curves over epochs.
             Tells you *whether* training is progressing and whether it
             overfits. Cheap, needs no GPU.

  predict  — load the trained checkpoint and run model.predict() on episodes
             from a dataset (val by default), comparing the policy's output
             to the expert action stored in the HDF5. Tells you *what* the
             policy actually predicts — directions and magnitudes, not just a
             scalar loss. This is the analysis that catches "regresses to the
             mean" failures the loss number hides.

Usage:
    # plot training curves
    python examples/analyze_bc_run.py --run-dir output/bc_runs/phase1_full --mode curves

    # qualitative predicted-vs-expert for one episode
    python examples/analyze_bc_run.py --run-dir output/bc_runs/phase1_full \
        --mode predict --split val --episode 0

    # both, and aggregate per-episode error across the whole val split
    python examples/analyze_bc_run.py --run-dir output/bc_runs/phase1_full --mode both
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml


# ── run-dir helpers ────────────────────────────────────────────────────────

def load_run_config(run_dir: Path) -> dict:
    with (run_dir / "config.yaml").open() as f:
        return yaml.safe_load(f)


def pick_dataset_path(cfg: dict, run_dir: Path, split: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    key = {"train": "train_h5", "val": "val_h5", "test": "test_h5"}.get(split)
    val = cfg["DATA"].get(key) if key else None
    if val:
        # train_h5 may be a list (DAgger aggregate) or a string. predict mode
        # iterates a single HDF5, so for a list default to the first file (the
        # base train set); pass --dataset to point at a specific one.
        path = val[0] if isinstance(val, list) else val
        if isinstance(val, list) and len(val) > 1:
            print(f"[analyze] {split} is a {len(val)}-file aggregate; using {path} "
                  f"(pass --dataset to pick another)")
        if os.path.exists(path):
            return path
    raise SystemExit(f"Could not resolve a {split} dataset path; pass --dataset explicitly.")


# ── curves ─────────────────────────────────────────────────────────────────

def _read_log_csv(log_path: Path) -> dict[str, np.ndarray]:
    """Read log.csv into a dict of column_name -> float array (NaN for blanks)."""
    import csv
    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{log_path} is empty")
    cols: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        vals = []
        for r in rows:
            v = r.get(key, "")
            vals.append(float(v) if v not in ("", None) else np.nan)
        cols[key] = np.array(vals, dtype=np.float64)
    return cols


def analyze_curves(run_dir: Path, save: bool) -> None:
    import matplotlib.pyplot as plt

    log_path = run_dir / "log.csv"
    if not log_path.exists():
        raise SystemExit(f"No log.csv in {run_dir}")
    cols = _read_log_csv(log_path)
    epoch = cols["epoch"]
    print(f"Loaded {len(epoch)} epochs from {log_path}")

    # Console summary of the last epoch + best val.
    print(f"\nLast epoch ({int(epoch[-1])}):")
    for c in ["train_total", "val_total", "train_pose_l1", "val_pose_l1",
              "train_gripper_acc", "val_gripper_acc"]:
        if c in cols:
            print(f"  {c:18s} = {cols[c][-1]:.4f}")
    if "val_total" in cols and np.isfinite(cols["val_total"]).any():
        best_i = int(np.nanargmin(cols["val_total"]))
        print(f"\nBest val_total = {cols['val_total'][best_i]:.4f} "
              f"at epoch {int(epoch[best_i])}")

    panels = [
        ("Total loss",            ["train_total", "val_total"]),
        ("Pose loss (SmoothL1)",  ["train_pose_loss", "val_pose_loss"]),
        ("Gripper loss (BCE)",    ["train_gripper_loss", "val_gripper_loss"]),
        ("Pose L1 (normalized)",  ["train_pose_l1", "val_pose_l1"]),
        ("Pos vs Rot L1 (val)",   ["val_pose_pos_l1", "val_pose_rot_l1"]),
        ("Gripper accuracy",      ["train_gripper_acc", "val_gripper_acc"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (title, names) in zip(axes.flat, panels):
        for name in names:
            if name in cols:
                ax.plot(epoch, cols[name], label=name)
        ax.set_title(title); ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"BC training curves — {run_dir.name}", fontsize=13)
    plt.tight_layout()

    if save:
        out = run_dir / "curves.png"
        fig.savefig(out, dpi=120)
        print(f"\nSaved {out}")
    plt.show()


# ── predict ──────────────────────────────────────────────────────────────────

def build_and_load_model(cfg: dict, run_dir: Path, device: str, ckpt: str):
    from handover_sim2real.bc import BCPolicy, Normalizer

    norm_path = run_dir / "normalization.npz"
    normalizer = Normalizer.load(norm_path) if norm_path.exists() else None
    if normalizer is None:
        print("WARNING: no normalization.npz — predictions will be in normalized units!")

    m, d = cfg["MODEL"], cfg["DATA"]
    model = BCPolicy(
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
    ).to(device)

    ckpt_path = ckpt or str(run_dir / "checkpoints" / "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = str(run_dir / "checkpoints" / "last.pt")
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"Loaded checkpoint {ckpt_path} (epoch {payload.get('epoch', '?')})")
    return model


def _episode_keys(f: h5py.File) -> list[str]:
    return sorted(k for k in f.keys() if k.startswith("episode_"))


@torch.no_grad()
def predict_episode(model, device, pcs, rss):
    """Run model.predict over a full episode. pcs [T,N,5], rss [T,32] raw."""
    pc = torch.from_numpy(pcs).float().to(device)
    rs = torch.from_numpy(rss).float().to(device)
    return model.predict(pc, rs).cpu().numpy()  # [T, 7] in real units


def analyze_predict(run_dir: Path, cfg: dict, dataset_path: str,
                    episode: int | None, device: str, ckpt: str, save: bool) -> None:
    model = build_and_load_model(cfg, run_dir, device, ckpt)

    with h5py.File(dataset_path, "r") as f:
        keys = _episode_keys(f)
        if not keys:
            raise SystemExit(f"No episodes in {dataset_path}")

        # ---- aggregate per-episode error across the whole split ----
        print(f"\nPer-episode error on {dataset_path} ({len(keys)} episodes):")
        print(f"  {'episode':>8s}  {'steps':>5s}  {'pos_l1(m)':>10s}  "
              f"{'rot_l1(rad)':>11s}  {'grip_acc':>8s}")
        agg_pos, agg_rot, agg_grip, agg_n = [], [], [], 0
        per_ep = {}
        scene_of = {}
        for k in keys:
            pcs = f[k]["point_clouds"][:]
            rss = f[k]["robot_states"][:]
            exp = f[k]["expert_actions"][:]          # raw expert action
            pred = predict_episode(model, device, pcs, rss)

            pos_l1 = np.abs(pred[:, :3] - exp[:, :3]).mean()
            rot_l1 = np.abs(pred[:, 3:6] - exp[:, 3:6]).mean()
            grip_acc = (pred[:, 6] == exp[:, 6]).mean()
            per_ep[k] = (pred, exp)
            scene_of[k] = int(f[k].attrs.get("scene_idx", -1))
            ep_idx = int(k.split("_")[1])
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


# ── shared interactive predicted-vs-expert viewer ─────────────────────────────

def show_predictions(run_dir: Path, keys: list[str],
                     per_ep: dict[str, tuple[np.ndarray, np.ndarray]],
                     scene_of: dict[str, int], episode: int | None,
                     save: bool) -> None:
    """Interactive per-episode viewer of policy (pred) vs expert action.

    per_ep maps episode_key -> (pred[T,7], expert[T,7]) in real units. Shared by
    both the Phase-1 (analyze_bc_run) and Phase-2 (analyze_act_run) analyzers.
    Navigate with ←/→ or n/p, s = save PNG, q = quit.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    # Starting position in the sorted episode list.
    if episode is None:
        pos0 = 0
    else:
        want = f"episode_{episode:05d}"
        if want in per_ep:
            pos0 = keys.index(want)
        else:
            print(f"WARNING: {want} not in this split — starting at {keys[0]}")
            pos0 = 0
    state = {"pos": pos0}

    comp_names = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.13,
                        hspace=0.35, wspace=0.30)

    def render() -> None:
        k = keys[state["pos"]]
        pred, exp = per_ep[k]
        T = len(exp)
        steps = np.arange(T)
        for ax in axes.flat:
            ax.clear()
        for i in range(6):
            ax = axes.flat[i]
            ax.plot(steps, exp[:, i],  "o-",  ms=3, label="expert", color="tab:green")
            ax.plot(steps, pred[:, i], "x--", ms=4, label="policy", color="tab:red")
            ax.set_title(comp_names[i]); ax.set_xlabel("step")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
        # gripper command (binary)
        ax = axes.flat[6]
        ax.step(steps, exp[:, 6],  where="post", label="expert", color="tab:green")
        ax.step(steps, pred[:, 6], where="post", label="policy", color="tab:red", ls="--")
        ax.set_title("gripper cmd"); ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["close", "open"])
        ax.set_xlabel("step"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        # per-step error magnitude
        ax = axes.flat[7]
        pos_err = np.linalg.norm(pred[:, :3] - exp[:, :3], axis=1)
        rot_err = np.linalg.norm(pred[:, 3:6] - exp[:, 3:6], axis=1)
        ax.plot(steps, pos_err, label="‖Δpos err‖ (m)")
        ax.plot(steps, rot_err, label="‖Δrot err‖ (rad)")
        ax.set_title("per-step error magnitude"); ax.set_xlabel("step")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        fig.suptitle(f"Policy vs expert — {run_dir.name}  {k}  "
                     f"(scene {scene_of[k]}, {T} steps)   "
                     f"[{state['pos'] + 1}/{len(keys)}]\n"
                     f"←/→ or n/p = prev/next     s = save PNG     q = quit",
                     fontsize=12)
        fig.canvas.draw_idle()

    def save_current() -> None:
        k = keys[state["pos"]]
        out = run_dir / f"predict_{k}.png"
        fig.savefig(out, dpi=120)
        print(f"Saved {out}")

    def step(delta: int) -> None:
        state["pos"] = (state["pos"] + delta) % len(keys)
        render()

    def on_key(event) -> None:
        if event.key in ("right", "n"):
            step(+1)
        elif event.key in ("left", "p"):
            step(-1)
        elif event.key == "s":
            save_current()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # On-screen Prev/Next buttons (kept on fig so they aren't GC'd).
    b_prev = Button(fig.add_axes([0.40, 0.025, 0.085, 0.05]), "◀ Prev")
    b_next = Button(fig.add_axes([0.515, 0.025, 0.085, 0.05]), "Next ▶")
    b_prev.on_clicked(lambda _e: step(-1))
    b_next.on_clicked(lambda _e: step(+1))
    fig._nav_buttons = (b_prev, b_next)

    render()
    if save:
        save_current()
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="output/bc_runs/<name>")
    p.add_argument("--mode", default="both", choices=["curves", "predict", "both"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="which dataset split to run predict on")
    p.add_argument("--dataset", default=None,
                   help="explicit HDF5 path (overrides --split lookup)")
    p.add_argument("--episode", type=int, default=None,
                   help="episode index for the detailed per-component plot")
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
