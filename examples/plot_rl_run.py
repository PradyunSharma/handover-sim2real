"""
Plot Phase-3 RL training curves from a run's log.csv.

    python examples/plot_rl_run.py output/rl_runs/rl_run1
    python examples/plot_rl_run.py output/rl_runs/rl_run1 --show

Reads <run>/log.csv (one row per iter, written by train_rl.py) and renders a
2x3 grid:
  • success   — eval + rollout success, and the online-buffer +reward fraction
  • approach  — closest EE->grasp distance reached (vs the 0.02 m close thresh):
                the reaching-vs-closing signal
  • close     — close-commit rate (does the policy ever close?)
  • critic    — critic_loss and its target mean
  • value     — q_mean (data actions) vs q_pi (policy actions; the OOD gap)
  • actor     — actor_loss / bc_loss / gripper logit / action magnitude

Saves <run>/curves.png (and shows it with --show).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; overridden by --show below
import matplotlib.pyplot as plt


def _load(log_path: Path):
    cols: dict[str, list] = {}
    with log_path.open() as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)

    n_rows = len(cols.get("iter", []))

    def num(key):
        vals = cols.get(key)
        if not vals:                      # column absent (e.g. older log) → all-NaN
            return [float("nan")] * n_rows
        out = []
        for v in vals:
            try:
                out.append(float(v))
            except (ValueError, TypeError):
                out.append(float("nan"))
        return out

    it = num("iter")
    return it, num, cols


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="RL run dir (or a path to log.csv)")
    p.add_argument("--save", default=None, help="output png (default <run>/curves.png)")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    run = Path(args.run)
    log_path = run if run.suffix == ".csv" else run / "log.csv"
    if not log_path.exists():
        raise SystemExit(f"no log.csv at {log_path} — has training written any rows yet?")

    it, num, _ = _load(log_path)
    if args.show:
        matplotlib.use("TkAgg", force=True)

    def sparse(key):                       # (x, y) for the non-NaN points of an eval col
        y = num(key)
        xy = [(x, v) for x, v in zip(it, y) if v == v]
        return [x for x, _ in xy], [v for _, v in xy]

    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    fig.suptitle(f"RL training — {log_path}")

    # [0,0] success — the metric that matters
    ex, ey = sparse("eval_succ")
    ax[0, 0].plot(it, num("roll_succ"), lw=1, alpha=0.6, label="rollout succ (policy)")
    if ey:
        ax[0, 0].plot(ex, ey, "o-", color="C3", label="eval succ")
    ax[0, 0].plot(it, num("exp_succ"), lw=1, alpha=0.5, color="C2", label="expert-episode succ")
    ax[0, 0].plot(it, num("buf_pos"), lw=1, alpha=0.5, color="C7", label="buf +reward frac")
    ax[0, 0].set_title("success rate"); ax[0, 0].set_ylim(-0.02, 1.02)
    ax[0, 0].set_xlabel("iter"); ax[0, 0].legend()

    # [0,1] approach — closest EE->grasp distance (the reaching-vs-closing signal)
    ax[0, 1].plot(it, num("roll_min_pos"), lw=1, alpha=0.6, label="rollout min pos")
    mx, my = sparse("eval_min_pos")
    if my:
        ax[0, 1].plot(mx, my, "o-", color="C3", label="eval min pos")
    ax[0, 1].axhline(0.02, ls="--", lw=1, color="k", alpha=0.5, label="close thresh 0.02 m")
    ax[0, 1].set_title("closest approach to grasp (m)"); ax[0, 1].set_xlabel("iter")
    ax[0, 1].set_ylim(bottom=0); ax[0, 1].legend()

    # [0,2] close rate — does the policy ever commit a close?
    ax[0, 2].plot(it, num("roll_close"), lw=1, alpha=0.6, label="rollout close rate")
    cx, cy = sparse("eval_close")
    if cy:
        ax[0, 2].plot(cx, cy, "o-", color="C3", label="eval close rate")
    ax[0, 2].set_title("close-commit rate"); ax[0, 2].set_ylim(-0.02, 1.02)
    ax[0, 2].set_xlabel("iter"); ax[0, 2].legend()

    # [1,0] critic
    ax[1, 0].plot(it, num("critic_loss"), label="critic_loss")
    ax[1, 0].plot(it, num("target_mean"), lw=1, alpha=0.6, label="target_mean")
    ax[1, 0].plot(it, num("aux_c"), lw=1, alpha=0.6, label="aux (grasp-pose)")
    ax[1, 0].set_title("critic"); ax[1, 0].set_xlabel("iter"); ax[1, 0].legend()

    # [1,1] value — q_mean (stored actions) vs q_pi (policy actions, right axis)
    ax[1, 1].plot(it, num("q_mean"), color="C2", label="q_mean (data a)")
    ax[1, 1].set_ylabel("q_mean"); ax[1, 1].set_xlabel("iter")
    axr = ax[1, 1].twinx()
    axr.plot(it, num("q_pi"), color="C1", lw=1, alpha=0.7, label="q_pi (policy a)")
    axr.set_ylabel("q_pi")
    ax[1, 1].set_title("value estimate  (q_pi >> q_mean = OOD gap)")
    l1, la1 = ax[1, 1].get_legend_handles_labels(); l2, la2 = axr.get_legend_handles_labels()
    ax[1, 1].legend(l1 + l2, la1 + la2, loc="upper left")

    # [1,2] actor
    ax[1, 2].plot(it, num("actor_loss"), label="actor_loss")
    ax[1, 2].plot(it, num("bc_loss"), lw=1, alpha=0.6, label="bc_loss")
    ax[1, 2].plot(it, num("grip_logit"), lw=1, alpha=0.6, label="grip logit (mean)")
    ax[1, 2].plot(it, num("a_absmean"), lw=1, alpha=0.6, label="|a_pose| mean")
    ax[1, 2].set_title("actor"); ax[1, 2].set_xlabel("iter"); ax[1, 2].legend()

    fig.tight_layout()
    out = Path(args.save) if args.save else log_path.parent / "curves.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
