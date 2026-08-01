"""
Plot Phase-3 RL training curves from a run's log.csv.

    python examples/plot_rl_run.py output/rl_runs/rl_run1
    python examples/plot_rl_run.py output/rl_runs/rl_run1 --show

Reads <run>/log.csv (one row per iter, written by train_rl.py) and renders TWO
figures.

MAIN (<run>/curves.png) — 2x3 grid:
  • success   — eval + rollout success, and the online-buffer +reward fraction
  • approach  — closest EE->grasp POSITION error (left axis, vs the 0.02 m close
                thresh) AND ROTATION error (right axis, vs the 0.34 rad thresh):
                the reaching-vs-closing signal, both DoF
  • close     — close-commit rate (does the policy ever close?)
  • critic    — critic_loss + aux (LOG left axis) vs target_mean (linear right
                axis); the two losses sit ~30x below target_mean, so on a shared
                linear axis they render as flat lines on zero
  • value     — q_mean (data actions) vs q_pi (policy actions; the OOD gap)
  • actor     — actor_loss / bc_loss / gripper logit / action magnitude

DIAGNOSTIC (<run>/curves_diag.png) — 2x2 grid of the columns the main figure
does not show:
  • failure modes — eval outcome breakdown (success / miss / contact-drop /
                    timeout) as fractions of the retained eval episodes: which
                    sub-skill is missing when success is flat
  • curriculum    — the reverse-curriculum takeover band [ei_lo, ei_hi], beta,
                    and buffer fill; without these the roll_* curves are not
                    comparable across iterations
  • blend         — pg_loss vs lam: under pg_normalize, pg_loss should stay
                    bounded near -alpha however far lam swings
  • losses        — all loss components on one LOG axis

Saves both (and shows them with --show).
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


def _positive(vals):
    """Mask non-positive entries to NaN so a log axis can take the series."""
    return [v if (v == v and v > 0) else float("nan") for v in vals]


def _twin_legend(axl, axr, loc="upper left"):
    l1, la1 = axl.get_legend_handles_labels()
    l2, la2 = axr.get_legend_handles_labels()
    axl.legend(l1 + l2, la1 + la2, loc=loc, fontsize=8)


def _eval_breakdown(it, num):
    """Per-eval outcome fractions. train_rl.py logs the failure COUNTS but not the
    number of retained episodes, so recover it from
        kept = (miss + timeout + fail) / (1 - eval_succ)
    (skipped scenes are already excluded from both sides). Returns
    (x, success, miss, fail, timeout) as fractions of kept."""
    xs, f_succ, f_miss, f_fail, f_to = [], [], [], [], []
    for x, s, mi, to, fa in zip(it, num("eval_succ"), num("eval_miss"),
                                num("eval_timeout"), num("eval_fail")):
        if s != s or mi != mi or to != to or fa != fa:
            continue                       # not an eval iter (blank cells → NaN)
        others = mi + to + fa
        if s >= 1.0:                       # everything succeeded: kept unrecoverable
            kept = others if others > 0 else float("nan")
        else:
            kept = others / (1.0 - s)
        if kept != kept or kept <= 0:
            continue
        xs.append(x)
        f_succ.append(s)
        f_miss.append(mi / kept)
        f_fail.append(fa / kept)
        f_to.append(to / kept)
    return xs, f_succ, f_miss, f_fail, f_to


def _read_alpha(run_dir: Path):
    """-alpha is the bound pg_loss should respect under pg_normalize. Read it from
    the run's saved rl_config.yaml if available; return None otherwise (the
    reference line is then simply omitted)."""
    cfg_path = run_dir / "rl_config.yaml"
    if not cfg_path.exists():
        return None
    try:
        import yaml
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        rl = cfg.get("RL", {})
        if not rl.get("pg_normalize", False):
            return None
        return float(rl.get("alpha", 2.5))
    except Exception:
        return None


# ── main figure ──────────────────────────────────────────────────────────────

def _figure_main(it, num, sparse, log_path):
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
    ax[0, 0].set_xlabel("iter"); ax[0, 0].legend(fontsize=8)

    # [0,1] approach — closest EE->grasp POSITION (left) and ROTATION (right).
    # Both are needed: a run can satisfy the rotation tolerance while still
    # stalling far outside the position tolerance (or vice versa), and the two
    # live on incompatible scales (metres vs radians) so they need twin axes.
    ax[0, 1].plot(it, num("roll_min_pos"), lw=1, alpha=0.6, color="C0",
                  label="rollout min pos")
    mx, my = sparse("eval_min_pos")
    if my:
        ax[0, 1].plot(mx, my, "o-", color="C3", label="eval min pos")
    ax[0, 1].axhline(0.02, ls="--", lw=1, color="k", alpha=0.5,
                     label="close thresh 0.02 m")
    ax[0, 1].set_ylabel("position error (m)")
    ax[0, 1].set_ylim(bottom=0)

    axr = ax[0, 1].twinx()
    rx, ry = sparse("eval_min_rot")
    if ry:
        axr.plot(rx, ry, "s--", color="C4", lw=1.2, ms=3.5, label="eval min rot")
    axr.axhline(0.34, ls=":", lw=1, color="C4", alpha=0.7,
                label="close thresh 0.34 rad")
    axr.set_ylabel("rotation error (rad)", color="C4")
    axr.tick_params(axis="y", labelcolor="C4")
    axr.set_ylim(bottom=0)

    ax[0, 1].set_title("closest approach to grasp"); ax[0, 1].set_xlabel("iter")
    # upper RIGHT: both error curves decay left-to-right, so the top-right corner
    # is the empty region (upper left sits on top of the early eval points).
    _twin_legend(ax[0, 1], axr, loc="upper right")

    # [0,2] close rate — does the policy ever commit a close?
    ax[0, 2].plot(it, num("roll_close"), lw=1, alpha=0.6, label="rollout close rate")
    cx, cy = sparse("eval_close")
    if cy:
        ax[0, 2].plot(cx, cy, "o-", color="C3", label="eval close rate")
    ax[0, 2].set_title("close-commit rate"); ax[0, 2].set_ylim(-0.02, 1.02)
    ax[0, 2].set_xlabel("iter"); ax[0, 2].legend(fontsize=8)

    # [1,0] critic — critic_loss and aux_c run ~30x below target_mean, so on one
    # linear axis both losses collapse onto zero and their (monotone) decline is
    # invisible. Losses go on a LOG left axis, target_mean on a linear right axis.
    ax[1, 0].plot(it, _positive(num("critic_loss")), color="C0", label="critic_loss")
    ax[1, 0].plot(it, _positive(num("aux_c")), color="C1", lw=1, alpha=0.8,
                  label="aux (grasp-pose)")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_ylabel("loss (log)")
    ax[1, 0].set_xlabel("iter")

    axr = ax[1, 0].twinx()
    axr.plot(it, num("target_mean"), color="C7", lw=1, alpha=0.7, label="target_mean")
    axr.set_ylabel("target_mean", color="C7")
    axr.tick_params(axis="y", labelcolor="C7")

    ax[1, 0].set_title("critic  (losses log-scale, target linear)")
    _twin_legend(ax[1, 0], axr)

    # [1,1] value — q_mean (stored actions) vs q_pi (policy actions, right axis)
    ax[1, 1].plot(it, num("q_mean"), color="C2", label="q_mean (data a)")
    ax[1, 1].set_ylabel("q_mean"); ax[1, 1].set_xlabel("iter")
    axr = ax[1, 1].twinx()
    axr.plot(it, num("q_pi"), color="C1", lw=1, alpha=0.7, label="q_pi (policy a)")
    axr.set_ylabel("q_pi")
    ax[1, 1].set_title("value estimate  (q_pi >> q_mean = OOD gap)")
    _twin_legend(ax[1, 1], axr)

    # [1,2] actor
    ax[1, 2].plot(it, num("actor_loss"), label="actor_loss")
    ax[1, 2].plot(it, num("bc_loss"), lw=1, alpha=0.6, label="bc_loss")
    ax[1, 2].plot(it, num("grip_logit"), lw=1, alpha=0.6, label="grip logit (mean)")
    ax[1, 2].plot(it, num("a_absmean"), lw=1, alpha=0.6, label="|a_pose| mean")
    ax[1, 2].set_title("actor"); ax[1, 2].set_xlabel("iter")
    ax[1, 2].legend(fontsize=8)

    fig.tight_layout()
    return fig


# ── diagnostic figure ────────────────────────────────────────────────────────

def _figure_diag(it, num, log_path, run_dir):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"RL diagnostics — {log_path}")

    # [0,0] eval failure modes — the "why did it fail" panel. A flat-zero success
    # curve is uninformative on its own; this splits it into never-closed
    # (timeout), closed-in-the-wrong-place (miss), and collided/dropped (fail).
    xs, f_succ, f_miss, f_fail, f_to = _eval_breakdown(it, num)
    if xs:
        ax[0, 0].stackplot(xs, f_succ, f_miss, f_fail, f_to,
                           labels=["success", "miss (bad close)",
                                   "contact / drop", "timeout (never closed)"],
                           colors=["C2", "C1", "C3", "C7"], alpha=0.85)
        ax[0, 0].set_ylim(0, 1)
        # the top of the stack is the (flat) timeout band, so a legend there
        # covers no detail; lower left would sit on the success band.
        ax[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    else:
        ax[0, 0].text(0.5, 0.5, "no eval rows yet", ha="center", va="center",
                      transform=ax[0, 0].transAxes)
    ax[0, 0].set_title("eval outcome breakdown (fraction of retained episodes)")
    ax[0, 0].set_xlabel("iter")

    # [0,1] curriculum / schedule state. roll_* metrics are measured under the
    # takeover band, so they are only comparable across iterations at the same
    # band — this panel is what makes them readable.
    lo, hi = num("ei_lo"), num("ei_hi")
    ax[0, 1].fill_between(it, lo, hi, color="C0", alpha=0.25,
                          label="expert takeover band")
    ax[0, 1].plot(it, hi, color="C0", lw=1, label="ei_hi")
    ax[0, 1].plot(it, lo, color="C0", lw=1, ls="--", label="ei_lo")
    ax[0, 1].set_ylabel("expert-playback steps")
    ax[0, 1].set_xlabel("iter")
    ax[0, 1].set_ylim(bottom=0)

    axr = ax[0, 1].twinx()
    axr.plot(it, num("beta"), color="C3", lw=1, alpha=0.8, label="beta (expert mix)")
    buf = num("buffer")
    buf_max = max((v for v in buf if v == v), default=0.0)
    if buf_max > 0:
        axr.plot(it, [v / buf_max if v == v else float("nan") for v in buf],
                 color="C7", lw=1, alpha=0.7, label="buffer fill (frac of max)")
    axr.set_ylabel("beta / buffer fill")
    axr.set_ylim(-0.02, 1.02)

    ax[0, 1].set_title("curriculum + schedule state")
    _twin_legend(ax[0, 1], axr, loc="center right")

    # [1,0] PG/BC blend. Under pg_normalize, lam = alpha / mean|Q(s,pi(s))| and
    # rises as Q falls, but the PRODUCT is what enters the loss: pg_loss should
    # stay bounded near -alpha. Divergence shows up here first.
    ax[1, 0].plot(it, num("pg_loss"), color="C0", label="pg_loss")
    alpha_ref = _read_alpha(run_dir)
    if alpha_ref is not None:
        ax[1, 0].axhline(-alpha_ref, ls="--", lw=1, color="k", alpha=0.5,
                         label=f"-alpha = {-alpha_ref:g}")
    ax[1, 0].set_ylabel("pg_loss")
    ax[1, 0].set_xlabel("iter")

    axr = ax[1, 0].twinx()
    axr.plot(it, num("lam"), color="C1", lw=1, alpha=0.7, label="lam")
    axr.set_ylabel("lam", color="C1")
    axr.tick_params(axis="y", labelcolor="C1")

    ax[1, 0].set_title("PG / BC blend  (pg_loss bounded by -alpha = healthy)")
    _twin_legend(ax[1, 0], axr)

    # [1,1] every loss component on one log axis — they span ~2 decades, so a
    # linear axis hides all but the largest.
    for key, label, color in (("critic_loss", "critic_loss", "C0"),
                              ("bc_loss", "bc_loss (pose)", "C1"),
                              ("grip_loss", "grip_loss (BCE)", "C2"),
                              ("aux_a", "aux actor", "C3"),
                              ("aux_c", "aux critic", "C4")):
        ax[1, 1].plot(it, _positive(num(key)), lw=1, alpha=0.85,
                      color=color, label=label)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel("loss (log)")
    ax[1, 1].set_title("loss components")
    ax[1, 1].set_xlabel("iter")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="RL run dir (or a path to log.csv)")
    p.add_argument("--save", default=None, help="output png (default <run>/curves.png)")
    p.add_argument("--save-diag", default=None,
                   help="diagnostic png (default <run>/curves_diag.png)")
    p.add_argument("--no-diag", action="store_true",
                   help="skip the diagnostic figure")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    run = Path(args.run)
    log_path = run if run.suffix == ".csv" else run / "log.csv"
    if not log_path.exists():
        raise SystemExit(f"no log.csv at {log_path} — has training written any rows yet?")
    run_dir = log_path.parent

    it, num, _ = _load(log_path)
    if args.show:
        matplotlib.use("TkAgg", force=True)

    def sparse(key):                       # (x, y) for the non-NaN points of an eval col
        y = num(key)
        xy = [(x, v) for x, v in zip(it, y) if v == v]
        return [x for x, _ in xy], [v for _, v in xy]

    fig = _figure_main(it, num, sparse, log_path)
    out = Path(args.save) if args.save else run_dir / "curves.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    if not args.no_diag:
        fig_d = _figure_diag(it, num, log_path, run_dir)
        out_d = Path(args.save_diag) if args.save_diag else run_dir / "curves_diag.png"
        fig_d.savefig(out_d, dpi=120)
        print(f"wrote {out_d}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
