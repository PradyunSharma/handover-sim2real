"""
Plot a Phase-4 DAgger run from its dagger_log.csv.

    python examples/plot_dagger_run.py output/dagger_runs/dagger4_run1
    python examples/plot_dagger_run.py output/dagger_runs/dagger4_run1 --show

Safe to run WHILE the loop is going — it only reads the CSV, which is appended
and flushed once per iteration.

Reads <run>/dagger_log.csv (one row per DAgger iteration, written by
train_dagger_phase4.py) and renders two figures.

MAIN (<run>/curves.png) — 2x4 grid, "did it learn":
  • success    — success / grasp / near / close rate on the held-out eval scenes.
                 The four are nested (close >= near, close >= grasp >= success),
                 so the VERTICAL GAPS are the diagnosis: close-near is closing in
                 the wrong place, near-grasp is closing right but not gripping,
                 grasp-success is gripping and then losing it.
  • opportunity— did the policy GET a chance and did it TAKE it: the fraction of
                 episodes that ever reached a graspable pose, the fraction that
                 commanded a close, success CONDITIONED on having closed, and the
                 episodes that reached a graspable pose and came away with
                 nothing. A flat success rate with a rising chance rate means the
                 reach is solved and the trigger is not.
  • approach   — EE->grasp error: the CLOSEST reached over the episode (solid;
                 defined even when the policy never closes) and the error AT the
                 close (dashed), position on the left axis and rotation on the
                 right. Phase-3 experience is that ROTATION binds first.
  • outcomes   — eval outcome breakdown as a stacked area (fractions summing to 1):
                 which failure mode is being traded for which as success moves.
  • learner    — how far the LEARNER's own rollouts got during collection:
                 fraction reaching the standoff / the grasp, and closing on their
                 own. This is DAgger's state distribution shifting, and it moves
                 BEFORE eval success does.
  • consistency— scenes revisited in a later iteration, and how many aimed at a
                 DIFFERENT grasp than the first time. Must stay 0: a nonzero bar
                 means D holds contradictory labels for one scene.
  • dataset    — |D| in steps (left) and the on-policy share (right). Success
                 plotted against this is what separates "DAgger helps" from
                 "more data helps".
  • fit        — train/val loss of each refit (log axis) + gripper accuracy.
                 Under FTL each point is a fresh fit on a bigger, more diverse D;
                 a RISING train loss means the aggregate is becoming
                 self-inconsistent, which is the failure DAgger cannot average away.

DIAGNOSTIC (<run>/curves_diag.png) — 2x3 grid, "is the machinery healthy":
  • labels     — mean approach-label displacement vs ee_step, and the count of
                 degenerate (~zero) labels. A collapse towards 0 is the standoff
                 stall returning; it is invisible in step counts.
  • endgame    — CLOSE labels produced and reach-tail steps followed: whether the
                 expert is still teaching the part of the task that matters.
  • planner    — OMG failures, goal-grasp switches, pinned episodes. With the pin
                 table loaded, goal_switch should be flat 0; anything else means
                 the table is stale or not matching.
  • closing    — when each side decides to close: the learner's mean close step
                 during collection vs at eval, against the horizon.
  • beta/mix   — the beta schedule and the fraction of executed steps that were
                 the expert's. Without these the collection curves are not
                 comparable across iterations.
  • cost       — wall-clock split by phase, so the expensive part is visible.

Saves both (and shows them with --show).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; overridden by --show below
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _load(log_path: Path):
    cols: dict[str, list] = {}
    with log_path.open() as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)

    n_rows = len(cols.get("iter", []))

    def num(key):
        """Column as floats. Absent column (an older log) or blank cell -> NaN,
        which matplotlib renders as a gap rather than a spurious zero."""
        vals = cols.get(key)
        if not vals:
            return [float("nan")] * n_rows
        out = []
        for v in vals:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out

    return num, n_rows


def _finite(ys) -> bool:
    return any(y == y for y in ys)


def _grid(ax, title, xlabel="DAgger iteration", ylabel=None):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)


def _legend(ax, **kw):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, **kw)


def _fix_x(fig, it):
    """Iterations are integers, and a panel whose series are all-NaN (no closes
    yet, say) would otherwise fall back to a 0..1 axis that silently disagrees
    with its neighbours."""
    lo, hi = min(it), max(it)
    pad = 0.2 if hi > lo else 0.5
    for a in fig.axes:
        a.set_xlim(lo - pad, hi + pad)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="output/dagger_runs/<name>")
    p.add_argument("--show", action="store_true")
    p.add_argument("--ee-step", type=float, default=0.04,
                   help="DAGGER.ee_step, drawn as the reference label scale")
    p.add_argument("--pos-thresh", type=float, default=0.02)
    p.add_argument("--rot-thresh", type=float, default=0.34)
    args = p.parse_args()

    run = Path(args.run_dir)
    log_path = run / "dagger_log.csv"
    if not log_path.exists():
        raise SystemExit(f"no log at {log_path}")

    num, n = _load(log_path)
    if n == 0:
        raise SystemExit(f"{log_path} has no rows yet")
    it = num("iter")

    # ── MAIN ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 4, figsize=(21, 8))

    # success
    a = ax[0][0]
    for key, label, style in (("close_rate", "close", ":"),
                              ("near_rate", "near (pose ok)", "-."),
                              ("grasp_rate", "grasp", "--"),
                              ("success_rate", "success", "-")):
        ys = num(key)
        if _finite(ys):
            a.plot(it, ys, style, marker="o", ms=3, label=label,
                   lw=2 if key == "success_rate" else 1.2)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "eval (held out): the nested rates", ylabel="fraction of eval scenes")
    _legend(a, loc="upper left")

    # opportunity vs conversion: did it get a chance, and did it take it
    a = ax[0][1]
    for key, label, style, col in (
            ("chance_rate", "had a graspable pose", "-", "tab:blue"),
            ("close_rate", "commanded a close", "--", "tab:orange"),
            ("close_success_rate", "success | closed", "-", "tab:green"),
            ("missed_rate", "had chance, no grasp", "-", "tab:red"),
            ("miss_given_chance", "missed | had chance", ":", "tab:purple")):
        ys = num(key)
        if _finite(ys):
            a.plot(it, ys, style, marker="o", ms=3, color=col, label=label)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "opportunity vs conversion", ylabel="fraction")
    _legend(a, loc="upper left")

    # approach: CLOSEST the EE ever came, and where it was when it closed
    a = ax[0][2]
    mp, mr = num("eval_min_pos"), num("eval_min_rot")
    pe, re_ = num("mean_pos_err"), num("mean_rot_err")
    if _finite(mp):
        a.plot(it, mp, "-o", ms=3, color="tab:blue", label="min pos err (m)")
    if _finite(pe):
        a.plot(it, pe, "--o", ms=3, color="tab:cyan", alpha=0.8,
               label="pos err at close (m)")
    a.axhline(args.pos_thresh, color="tab:blue", ls=":", lw=1,
              label=f"close thresh {args.pos_thresh} m")
    a.set_ylim(bottom=0)
    a.set_ylabel("position error (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    if _finite(mr):
        a2.plot(it, mr, "-s", ms=3, color="tab:red", label="min rot err (rad)")
    if _finite(re_):
        a2.plot(it, re_, "--s", ms=3, color="tab:orange", alpha=0.8,
                label="rot err at close (rad)")
    a2.axhline(args.rot_thresh, color="tab:red", ls=":", lw=1,
               label=f"close thresh {args.rot_thresh} rad")
    a2.set_ylim(bottom=0)
    a2.set_ylabel("rotation error (rad)", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    _grid(a, "EE -> grasp: closest reached (solid) vs at the close (dashed)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right")

    # outcome breakdown
    a = ax[0][3]
    series = [("f_grasp_ok", "secured", "tab:green"),
              ("f_grasp_miss", "closed, not secured", "tab:olive"),
              ("f_no_release", "no release", "tab:orange"),
              ("f_drop", "drop", "tab:red"),
              ("f_human_contact", "human contact", "tab:purple"),
              ("f_timeout", "never closed", "tab:gray")]
    ys = [num(k) for k, _, _ in series]
    if any(_finite(y) for y in ys):
        clean = [[0.0 if v != v else v for v in y] for y in ys]
        a.stackplot(it, *clean, labels=[l for _, l, _ in series],
                    colors=[c for _, _, c in series], alpha=0.85)
        a.set_ylim(0, 1)
    _grid(a, "eval outcomes (fraction of scenes)")
    _legend(a, loc="lower left", ncol=2)

    # learner progress during collection
    a = ax[1][0]
    eps = num("episodes")
    for key, label in (("reached_standoff", "reached standoff"),
                       ("reached_grasp", "reached grasp (CLOSE fired)"),
                       ("policy_closed", "closed on its own")):
        ys = num(key)
        frac = [(y / e if (e and e > 0 and y == y and y >= 0) else float("nan"))
                for y, e in zip(ys, eps)]
        if _finite(frac):
            a.plot(it, frac, "-o", ms=3, label=label)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "collection: how far the LEARNER got", ylabel="fraction of episodes")
    _legend(a, loc="upper left")

    # grasp consistency across iterations (verifies the pin actually held)
    a = ax[1][1]
    for key, label, col in (("revisits", "scenes revisited", "tab:blue"),
                            ("grasp_mismatch", "grasp CHANGED", "tab:red")):
        ys = num(key)
        if _finite(ys):
            a.plot(it, ys, "-o", ms=3, color=col, label=label)
    a.set_ylabel("episodes", fontsize=8)
    a2 = a.twinx()
    dr = num("max_grasp_drift")
    if _finite(dr):
        a2.plot(it, dr, "--s", ms=3, color="tab:purple", label="max drift (m)")
    a2.set_ylabel("max drift (m)", fontsize=8, color="tab:purple")
    a2.tick_params(axis="y", labelcolor="tab:purple", labelsize=7)
    _grid(a, "same grasp per scene? (should be 0 changed)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    # dataset growth
    a = ax[1][2]
    ds = num("D_steps")
    if _finite(ds):
        a.plot(it, ds, "-o", ms=3, color="tab:blue", label="|D| (steps)")
    a.set_ylabel("labelled steps in D", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    fr = num("D_dagger_frac")
    if _finite(fr):
        a2.plot(it, fr, "-s", ms=3, color="tab:orange", label="on-policy share")
    a2.set_ylabel("DAgger fraction of D", fontsize=8, color="tab:orange")
    a2.set_ylim(0, 1)
    a2.tick_params(axis="y", labelcolor="tab:orange", labelsize=7)
    _grid(a, "the aggregate D the refit sees")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    # fit health
    a = ax[1][3]
    for key, label, style in (("train_loss", "train", "-"),
                              ("val_loss", "val", "--"),
                              ("best_val_loss", "best val", ":")):
        ys = num(key)
        if _finite(ys):
            a.plot(it, ys, style, marker="o", ms=3, label=label)
    a.set_yscale("log")
    a.set_ylabel("loss (log)", fontsize=8)
    a2 = a.twinx()
    for key, label, style in (("train_grip_acc", "train grip acc", "-"),
                              ("val_grip_acc", "val grip acc", "--")):
        ys = num(key)
        if _finite(ys):
            a2.plot(it, ys, style, marker="^", ms=3, color="tab:green", label=label,
                    alpha=0.7 if key == "val_grip_acc" else 1.0)
    a2.set_ylabel("gripper acc", fontsize=8, color="tab:green")
    a2.set_ylim(0, 1.02)
    a2.tick_params(axis="y", labelcolor="tab:green", labelsize=7)
    _grid(a, "the refit on the growing aggregate")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")

    _fix_x(fig, it)
    fig.suptitle(f"Phase-4 DAgger — {run.name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    main_out = run / "curves.png"
    fig.savefig(main_out, dpi=140)
    print(f"wrote {main_out}")

    # ── DIAGNOSTIC ──────────────────────────────────────────────────────────
    fig2, bx = plt.subplots(2, 3, figsize=(16, 8))

    # labels
    a = bx[0][0]
    lp = num("mean_label_pos")
    if _finite(lp):
        a.plot(it, lp, "-o", ms=3, color="tab:blue", label="mean approach label (m)")
    a.axhline(args.ee_step, color="k", ls=":", lw=1, label=f"ee_step {args.ee_step} m")
    a.set_ylim(bottom=0)
    a.set_ylabel("label displacement (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    ty = num("tiny_labels")
    if _finite(ty):
        a2.plot(it, ty, "-s", ms=3, color="tab:red", label="degenerate (~0) labels")
    a2.set_ylabel("count", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    _grid(a, "expert label scale (stall detector)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")

    # endgame coverage
    a = bx[0][1]
    for key, label in (("close_labels", "CLOSE labels"),
                       ("reach_steps", "committed-reach steps")):
        ys = num(key)
        if _finite(ys):
            a.plot(it, ys, "-o", ms=3, label=label)
    _grid(a, "endgame coverage in D_i", ylabel="labels this iteration")
    _legend(a, loc="upper left")

    # planner health
    a = bx[0][2]
    for key, label in (("omg_fail", "OMG plan failures"),
                       ("goal_switch", "goal-grasp switches"),
                       ("pinned", "episodes pinned")):
        ys = num(key)
        if _finite(ys):
            a.plot(it, ys, "-o", ms=3, label=label)
    _grid(a, "planner / grasp pinning", ylabel="count this iteration")
    _legend(a, loc="upper left")

    # when does it close
    a = bx[1][0]
    for key, label in (("mean_policy_close_step", "collection (learner)"),
                       ("mean_close_step", "eval")):
        ys = num(key)
        if _finite(ys):
            a.plot(it, ys, "-o", ms=3, label=label)
    _grid(a, "step at which the policy closes", ylabel="policy step")
    _legend(a, loc="upper left")

    # beta / expert mixing
    a = bx[1][1]
    b = num("beta")
    if _finite(b):
        a.plot(it, b, "-o", ms=3, color="tab:purple", label="beta (expert prob.)")
    steps = num("steps")
    exp = num("expert_steps")
    frac = [(e / s if (s and s > 0 and e == e and e >= 0) else float("nan"))
            for e, s in zip(exp, steps)]
    if _finite(frac):
        a.plot(it, frac, "--s", ms=3, color="tab:brown", label="expert share of steps")
    a.set_ylim(-0.02, 1.02)
    _grid(a, "expert mixing", ylabel="fraction")
    _legend(a, loc="upper right")

    # cost
    a = bx[1][2]
    cs, ts, es = num("collect_s"), num("train_s"), num("eval_s")
    if any(_finite(x) for x in (cs, ts, es)):
        clean = [[0.0 if v != v else v for v in y] for y in (cs, ts, es)]
        a.stackplot(it, *clean, labels=["collect", "train", "eval"],
                    colors=["tab:blue", "tab:orange", "tab:green"], alpha=0.85)
    _grid(a, "wall clock per iteration", ylabel="seconds")
    _legend(a, loc="upper left")

    _fix_x(fig2, it)
    fig2.suptitle(f"Phase-4 DAgger diagnostics — {run.name}", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    diag_out = run / "curves_diag.png"
    fig2.savefig(diag_out, dpi=140)
    print(f"wrote {diag_out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
