"""
Plot a Phase-5 DAgger run from its dagger_log.csv.

    python examples/plot_regrasp_run.py output/dagger_runs/dagger5_run1
    python examples/plot_regrasp_run.py output/dagger_runs/dagger5_run1 --show

Safe to run WHILE the loop is going — it only reads the CSV, which is appended
and flushed once per iteration.

Reads <run>/dagger_log.csv (one row per DAgger iteration, written by
train_regrasp.py) and renders three figures.

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
                 With an auxiliary goal-grasp head (run 13 on) a third, DOTTED
                 pair is added: how far the network's PREDICTION of the grasp is
                 from the pinned pose. Read it against the solid curves — an
                 accurate prediction alongside a gripper that still arrives far
                 away means the information is in the features and the action
                 head is not using it; both large means the observation cannot
                 support the target at all. The two say different things and
                 point at different fixes.
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
                 expert is still teaching the part of the task that matters. Also
                 the premature closes the policy commanded (each one is relabelled
                 OPEN, never copied into D), and the pairs dropped because an
                 EXPERT step collided and ended the episode — that red line should
                 sit at 0, or exclude_scenes is missing scenes the expert breaks.
  • planner    — OMG failures, goal-grasp switches, pinned episodes. With the pin
                 table loaded, goal_switch should be flat 0; anything else means
                 the table is stale or not matching.
  • mixing     — beta and the expert's share of executed steps, plus (DART runs
                 only) the share taken by random jolts and the fraction of those
                 jolts that ended the episode. That last line is the magnitude
                 tripwire: if it climbs, dart_pos_mag is knocking the object out
                 of the hand rather than displacing the gripper.
  • closing    — when each side decides to close: the learner's mean close step
                 during collection vs at eval, against the horizon.
  • beta/mix   — the beta schedule and the fraction of executed steps that were
                 the expert's. Without these the collection curves are not
                 comparable across iterations.
  • cost       — wall-clock split by phase, so the expensive part is visible.

REGRASP (<run>/curves_regrasp.png) — 1x3, "is the conditioning doing anything":
  • cond_track — THE panel, and the one to read first. The mean pairwise spread
                 of the four final EE poses over the spread of the four commanded
                 grasps, under the flip-invariant control-point metric. 1.0 means
                 the policy separates the four conditions as much as their targets
                 are separated; 0.0 means it does the same thing whatever it is
                 told, which is the multi-modal averaging failure Phase 5 is built
                 to avoid and which makes regrasping inert however good
                 `success_rate` looks. `near_rate` is drawn with it because the
                 pair localises the problem: both low = the conditioning is being
                 ignored, and the fix is FiLM rather than concatenation; cond_track
                 high with near_rate low = the policy separates the conditions but
                 tracks none of them, which is a reach-endgame problem.
  • retry@k    — success with k attempts at different grasps, in FPS order. The
                 regrasping headline, derived from the same episodes at no extra
                 cost. It assumes each retry restarts from home, which is true of
                 this evaluation and NOT of a real deployment, where attempt 2
                 begins wherever attempt 1 stopped. Read it as a ceiling.
  • per-slot   — success (solid) and near (dotted) for each pinned grasp. Slot 0
                 is OMG's own pick, so `succ_g0` is the curve comparable with a
                 Phase-4 run; the spread across slots is how much harder the
                 deliberately-separated grasps are.

Drawn only when the log has the columns, so a Phase-4 log renders two figures and
a Phase-5 log renders three.

Saves them all (and shows them with --show).

TARGET AWARENESS (run 21 on). `DAGGER.target` is read from <run>/config.yaml,
because under `pregrasp` several columns keep their NAMES and change their
MEANING — the episode ends 6.4 cm short of the grasp, so `eval_min_pos`,
`mean_pos_err` and `chance_rate` are measured against the PRE-GRASP. A pre-grasp
min-pos-err of 0.005 is not twenty times better than run 16's 0.10; it is a
different quantity, and plotting the two under one label invites exactly the
wrong comparison. Every affected panel is relabelled, and three things are added:

  • `mean_reach_pos_err` on the approach panel (diamonds) — where the BLIND push
    ended up relative to the GRASP. THIS is the line comparable to a grasp-mode
    run's min pos err, because it is the only one measured against the pose the
    gripper actually has to end up on.
  • `box_after_rate` on the opportunity panel — of the episodes that committed,
    how many had the object between the open jaws after the push. In this mode
    `box_chance_rate` reads ~0 by construction (the policy stops 6.4 cm short, so
    the jaws are never occupied while it is still deciding) and `box_taken_rate`
    is NaN with it; neither is a regression.
  • `settle_steps` on the endgame panel — how much convergence servo the episodes
    actually got, against `commit_settle_steps` x the episodes that committed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml
matplotlib.use("Agg")  # headless-safe; overridden by --show below
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _load(log_path: Path, eval_log: Path | None = None):
    cols: dict[str, list] = {}
    with log_path.open() as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)

    n_rows = len(cols.get("iter", []))

    # With EVAL.every: 0 the loop leaves every eval column blank and the metrics
    # live in eval_log.csv instead (examples/eval_dagger_run.py, a separate job).
    # Splice them in by iteration so the plots look the same either way. Only
    # BLANK cells are filled — an in-loop number always wins, so a run that did
    # both never has its own numbers overwritten by a re-scored pool.
    if eval_log is not None and eval_log.exists():
        with eval_log.open() as f:
            by_iter = {r["iter"]: r for r in csv.DictReader(f) if r.get("iter")}
        if by_iter:
            keys = set().union(*(set(r) for r in by_iter.values())) - {"iter", "run_dir"}
            filled = 0
            for k in keys:
                col = cols.setdefault(k, [""] * n_rows)
                for i, it in enumerate(cols.get("iter", [])):
                    if i < len(col) and not str(col[i]).strip():
                        v = by_iter.get(str(it), {}).get(k, "")
                        if str(v).strip():
                            col[i] = v
                            filled += 1
            print(f"[plot] merged {len(by_iter)} rows / {filled} cells from "
                  f"{eval_log.name}")

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


def _plot(ax, xs, ys, style="-", **kw):
    """Plot a series, DROPPING missing points rather than passing NaN through.

    Matplotlib breaks a line at every NaN, so a column that is only filled on
    some iterations — every eval column when EVAL.every > 1, every collection
    column on iteration 0 — would render as isolated markers while the legend
    advertised a connected line. Dropping the gaps instead joins the points that
    do exist, which is what these series mean: the same policy sequence, sampled
    less often, not a discontinuity in it.

    Returns False if there was nothing to draw, so callers can skip the label.
    """
    pts = [(x, y) for x, y in zip(xs, ys) if y == y]
    if not pts:
        return False
    ax.plot([p[0] for p in pts], [p[1] for p in pts], style, **kw)
    return True


def _stack(ax, xs, series, labels, colors):
    """Stacked area over only the iterations that HAVE data.

    `stackplot` cannot take NaN, and substituting 0 is worse than useless here:
    an iteration with no eval (EVAL.every > 1) would stack to zero and render as
    a spike down to the axis, which reads as "everything failed" rather than
    "not measured". So drop those rows entirely and stack the rest.
    """
    keep = [i for i in range(len(xs)) if any(s[i] == s[i] for s in series)]
    if not keep:
        return False
    clean = [[(s[i] if s[i] == s[i] else 0.0) for i in keep] for s in series]
    ax.stackplot([xs[i] for i in keep], *clean, labels=labels, colors=colors,
                 alpha=0.85)
    return True


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

    num, n = _load(log_path, eval_log=run / "eval_log.csv")
    if n == 0:
        raise SystemExit(f"{log_path} has no rows yet")
    it = num("iter")

    # ---- which pose the run was steering to (DAGGER.target, run 21 on) ----
    # Under target: pregrasp the episode ends 6.4 cm short of the grasp, so
    # eval_min_pos / mean_pos_err / chance_rate are measured against the PRE-GRASP
    # and box_chance_rate is ~0 by construction. Plotting those under the labels
    # this file used for runs 1-20 would invite exactly the wrong comparison —
    # a pre-grasp "min pos err" of 0.005 is not twenty times better than run 16's
    # 0.10, it is a different quantity. Read from the run's own saved config so a
    # log can never be plotted under the wrong labels.
    target = "grasp"
    cfg_path = run / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open() as f:
            target = str(((yaml.safe_load(f) or {}).get("DAGGER", {}) or {})
                         .get("target", "grasp"))
    pregrasp = target == "pregrasp"
    TGT = "pre-grasp" if pregrasp else "grasp"

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
            _plot(a, it, ys, style, marker="o", ms=3, label=label,
                   lw=2 if key == "success_rate" else 1.2)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "eval (held out): the nested rates", ylabel="fraction of eval scenes")
    _legend(a, loc="upper left")

    # opportunity vs conversion: did it get a chance, and did it take it.
    #
    # LEADS WITH THE GEOMETRIC TEST (box_*, dagger/grasp_box.py): object material
    # actually between the open jaws, which counts an off-pose grasp as the
    # opportunity it is. The pinned-pose pair (chance_rate / miss_given_chance)
    # is kept as a faint reference only — it gates on agreement with the pin and
    # reads 0.03-0.05 in runs succeeding 60-70% of the time, so it measures pin
    # agreement, not opportunity. Runs before this metric existed show only the
    # faint pair, which is why both are drawn rather than one replacing the other.
    a = ax[0][1]
    for key, label, style, col, lw in (
            ("box_chance_rate", "object in jaws", "-", "tab:blue", 1.4),
            ("box_taken_rate", "closed | in jaws", "-", "tab:green", 2.0),
            ("miss_given_box", "no grasp | in jaws", "-", "tab:red", 1.4),
            ("close_success_rate", "success | closed", "--", "tab:olive", 1.2),
            ("mean_box_frac", "mean jaw occupancy", ":", "tab:gray", 1.0)):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, style, marker="o", ms=3, color=col, label=label, lw=lw)
    # Pre-grasp mode: the jaws cannot contain the object while the policy is
    # still deciding — it stops 6.4 cm short — so box_chance_rate above reads ~0
    # BY CONSTRUCTION and box_taken_rate is NaN with it. `box_after_rate` is this
    # mode's conversion measure: of the episodes that committed, how many had the
    # object between the open jaws once the blind push had run.
    ys = num("box_after_rate")
    if _finite(ys):
        _plot(a, it, ys, "-D", ms=4, color="tab:purple", lw=2.0,
              label="in jaws AFTER push | committed")
    for key, label, col in (("chance_rate", f"at pinned {TGT} (ref)", "tab:blue"),
                            ("miss_given_chance", "missed | pinned (ref)", "tab:red")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, ":", marker="", color=col, label=label,
                  lw=1.0, alpha=0.4)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "opportunity vs conversion (geometric)", ylabel="fraction")
    _legend(a, loc="upper left")

    # approach: CLOSEST the EE ever came, and where it was when it closed
    a = ax[0][2]
    mp, mr = num("eval_min_pos"), num("eval_min_rot")
    pe, re_ = num("mean_pos_err"), num("mean_rot_err")
    if _finite(mp):
        _plot(a, it, mp, "-o", ms=3, color="tab:blue",
              label=f"min pos err to {TGT} (m)")
    if _finite(pe):
        _plot(a, it, pe, "--o", ms=3, color="tab:cyan", alpha=0.8,
               label=f"pos err to {TGT} at commit (m)"
                     if pregrasp else "pos err at close (m)")
    # Pre-grasp mode: where the BLIND push ended up relative to the GRASP. This,
    # not `mp` above, is the quantity comparable to a grasp-mode run's min pos
    # err — it is the only line on this panel measured against the pose the
    # gripper actually has to end up on. The gap between it and `mp` is the
    # push's own contribution, which should be small: the two differ only by
    # forward_dist's mismatch with the true reach and by the orientation error
    # projected over 6.4 cm.
    rp, rr = num("mean_reach_pos_err"), num("mean_reach_rot_err")
    if _finite(rp):
        _plot(a, it, rp, "-D", ms=3, color="tab:purple",
              label="pos err to GRASP after blind push (m)")
    # Auxiliary goal-grasp head (run 13 on): how far the network's BELIEF about
    # the grasp is from the pinned pose. Plotted on the same axes as the gripper's
    # actual error on purpose — the comparison is the whole point. If the head
    # predicts the pose accurately while the gripper still arrives far away, the
    # information is present in the features and the action head is not using it;
    # if both are large, the observation cannot support the target at all.
    # Absent (all-NaN) on runs without the head, so nothing changes for those.
    ap = num("aux_pos_mm")
    if _finite(ap):
        _plot(a, it, (np.asarray(ap, dtype=float) / 1000.0).tolist(), ":^", ms=3,
              color="tab:green", alpha=0.9,
              label="aux: predicted grasp pos err (m)")
    a.axhline(args.pos_thresh, color="tab:blue", ls=":", lw=1,
              label=f"close thresh {args.pos_thresh} m")
    a.set_ylim(bottom=0)
    a.set_ylabel("position error (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    if _finite(mr):
        _plot(a2, it, mr, "-s", ms=3, color="tab:red",
              label=f"min rot err to {TGT} (rad)")
    if _finite(re_):
        _plot(a2, it, re_, "--s", ms=3, color="tab:orange", alpha=0.8,
                label=f"rot err to {TGT} at commit (rad)"
                      if pregrasp else "rot err at close (rad)")
    if _finite(rr):
        _plot(a2, it, rr, "-D", ms=3, color="tab:brown", alpha=0.9,
              label="rot err to GRASP after blind push (rad)")
    ar = num("aux_rot_deg")
    if _finite(ar):
        _plot(a2, it, np.radians(np.asarray(ar, dtype=float)).tolist(), ":^", ms=3,
              color="tab:olive", alpha=0.9,
              label="aux: predicted grasp rot err (rad)")
    a2.axhline(args.rot_thresh, color="tab:red", ls=":", lw=1,
               label=f"close thresh {args.rot_thresh} rad")
    a2.set_ylim(bottom=0)
    a2.set_ylabel("rotation error (rad)", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    _grid(a, f"EE -> {TGT}: closest reached (solid) vs at the "
             + ("commit (dashed), and -> grasp after the push (diamond)"
                if pregrasp else "close (dashed)"))
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
    if _stack(a, it, [num(k) for k, _, _ in series],
              [l for _, l, _ in series], [c for _, _, c in series]):
        a.set_ylim(0, 1)
    _grid(a, "eval outcomes (fraction of scenes)")
    _legend(a, loc="lower left", ncol=2)

    # learner progress during collection
    a = ax[1][0]
    eps = num("episodes")
    for key, label in (("reached_standoff", "reached standoff"),
                       ("reached_grasp", "reached pre-grasp (COMMIT fired)"
                        if pregrasp else "reached grasp (CLOSE fired)"),
                       ("policy_closed",
                        "committed on its own" if pregrasp
                        else "closed on its own")):
        ys = num(key)
        frac = [(y / e if (e and e > 0 and y == y and y >= 0) else float("nan"))
                for y, e in zip(ys, eps)]
        if _finite(frac):
            _plot(a, it, frac, "-o", ms=3, label=label)
    a.set_ylim(-0.02, 1.02)
    _grid(a, "collection: how far the LEARNER got", ylabel="fraction of episodes")
    _legend(a, loc="upper left")

    # grasp consistency across iterations (verifies the pin actually held)
    a = ax[1][1]
    for key, label, col in (("revisits", "scenes revisited", "tab:blue"),
                            ("grasp_mismatch", "grasp CHANGED", "tab:red")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, "-o", ms=3, color=col, label=label)
    a.set_ylabel("episodes", fontsize=8)
    a2 = a.twinx()
    dr = num("max_grasp_drift")
    if _finite(dr):
        _plot(a2, it, dr, "--s", ms=3, color="tab:purple", label="max drift (m)")
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
        _plot(a, it, ds, "-o", ms=3, color="tab:blue", label="|D| (steps)")
    a.set_ylabel("labelled steps in D", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    fr = num("D_dagger_frac")
    if _finite(fr):
        _plot(a2, it, fr, "-s", ms=3, color="tab:orange", label="on-policy share")
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
            _plot(a, it, ys, style, marker="o", ms=3, label=label)
    a.set_yscale("log")
    a.set_ylabel("loss (log)", fontsize=8)
    a2 = a.twinx()
    for key, label, style in (("train_grip_acc", "train grip acc", "-"),
                              ("val_grip_acc", "val grip acc", "--")):
        ys = num(key)
        if _finite(ys):
            _plot(a2, it, ys, style, marker="^", ms=3, color="tab:green", label=label,
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
    fig.suptitle(f"Phase-5 DAgger — {run.name}"
                 + (f"   [target: {target} — geometry is measured to the "
                    f"PRE-GRASP, not the grasp]" if pregrasp else ""),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    main_out = run / "curves.png"
    fig.savefig(main_out, dpi=140)
    print(f"wrote {main_out}")

    # ── DIAGNOSTIC ──────────────────────────────────────────────────────────
    fig2, bx = plt.subplots(2, 4, figsize=(21, 8))

    # labels
    a = bx[0][0]
    lp = num("mean_label_pos")
    if _finite(lp):
        _plot(a, it, lp, "-o", ms=3, color="tab:blue", label="mean approach label (m)")
    a.axhline(args.ee_step, color="k", ls=":", lw=1, label=f"ee_step {args.ee_step} m")
    a.set_ylim(bottom=0)
    a.set_ylabel("label displacement (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    ty = num("tiny_labels")
    if _finite(ty):
        _plot(a2, it, ty, "-s", ms=3, color="tab:red", label="degenerate (~0) labels")
    a2.set_ylabel("count", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    _grid(a, "expert label scale (stall detector)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")

    # endgame coverage
    a = bx[0][1]
    for key, label in (("close_labels", "COMMIT labels" if pregrasp
                        else "CLOSE labels"),
                       ("reach_steps", "committed pre-grasp hold steps"
                        if pregrasp else "committed-reach steps"),
                       # Of those hold steps, how many were the convergence
                       # servo. Read against commit_settle_steps x the episodes
                       # that committed: a shortfall means episodes are entering
                       # the tolerance ball and then being ended before they can
                       # settle, which puts the arrival error — and therefore the
                       # landing error — straight back.
                       ("settle_steps", "settle steps"),
                       ("policy_close_cmds", "premature commit cmds"
                        if pregrasp else "premature close cmds")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, "-o", ms=3, label=label)
    # Pairs dropped because an EXPERT step ended the episode (the lateral-approach
    # collision). Should be ~0: a nonzero line means exclude_scenes is not
    # catching the scenes where the expert itself still collides.
    ys = num("dropped_tail")
    if _finite(ys):
        _plot(a, it, ys, "-s", ms=3, color="tab:red",
              label="expert-collision pairs dropped")
    _grid(a, "endgame coverage in D_i", ylabel="labels this iteration")
    _legend(a, loc="upper left")

    # planner health
    a = bx[0][2]
    for key, label in (("omg_fail", "OMG plan failures"),
                       ("goal_switch", "goal-grasp switches"),
                       ("pinned", "episodes pinned")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, "-o", ms=3, label=label)
    _grid(a, "planner / grasp pinning", ylabel="count this iteration")
    _legend(a, loc="upper left")

    # when does it close — and how long it had to decide. `mean_box_steps` is in
    # the same units (policy steps), so it belongs on this axis: a long window
    # sitting above the close step means the policy loitered in the jaws-occupied
    # state rather than never reaching it, which box_taken_rate alone cannot say.
    a = bx[1][0]
    for key, label in (("mean_policy_close_step", "collection (learner)"),
                       ("mean_close_step", "eval")):
        ys = num(key)
        if _finite(ys):
            _plot(a, it, ys, "-o", ms=3, label=label)
    ys = num("mean_box_steps")
    if _finite(ys):
        _plot(a, it, ys, "--s", ms=3, color="tab:gray",
              label="opportunity window (steps)")
    _grid(a, "step at which the policy commits" if pregrasp
             else "step at which the policy closes", ylabel="policy step")
    _legend(a, loc="upper left")

    # beta / expert mixing
    a = bx[1][1]
    b = num("beta")
    if _finite(b):
        _plot(a, it, b, "-o", ms=3, color="tab:purple", label="beta (expert prob.)")
    steps = num("steps")
    exp = num("expert_steps")
    frac = [(e / s if (s and s > 0 and e == e and e >= 0) else float("nan"))
            for e, s in zip(exp, steps)]
    if _finite(frac):
        _plot(a, it, frac, "--s", ms=3, color="tab:brown", label="expert share of steps")
    # DART belongs here: it is the OTHER thing that overrides the executed action.
    # Absent (or all -1) on a run collected without it, so `_finite` keeps the
    # panel unchanged for every pre-DART run.
    dart = num("dart")
    if _finite(dart) and any(d == d and d > 0 for d in dart):
        dfrac = [(d / s if (s and s > 0 and d == d and d >= 0) else float("nan"))
                 for d, s in zip(dart, steps)]
        _plot(a, it, dfrac, "-^", ms=3, color="tab:olive",
              label="DART share of steps")
        # The magnitude tripwire: jolts that ended the episode, as a fraction of
        # jolts fired. Climbing means dart_pos_mag is knocking the object out of
        # the hand rather than displacing the gripper.
        ended = num("dart_env_done")
        if _finite(ended):
            efrac = [(e / d if (d and d > 0 and e == e and e >= 0) else float("nan"))
                     for e, d in zip(ended, dart)]
            if _finite(efrac):
                _plot(a, it, efrac, ":v", ms=3, color="tab:red",
                      label="jolts that ended the episode")
    a.set_ylim(-0.02, 1.02)
    _grid(a, "expert mixing", ylabel="fraction")
    _legend(a, loc="upper right")

    # cost
    a = bx[1][2]
    _stack(a, it, [num("collect_s"), num("train_s"), num("eval_s")],
           ["collect", "train", "eval"],
           ["tab:blue", "tab:orange", "tab:green"])
    _grid(a, "wall clock per iteration", ylabel="seconds")
    _legend(a, loc="upper left")

    # ---- collection outcomes (DAGGER.outcome_check; empty for runs 1-17) ----
    # Deliberately the same taxonomy and the same colours as the eval-outcome
    # panel on curves.png, so the two stacks can be read against each other. The
    # difference between them IS the beta mixture: this one is what the
    # expert/learner blend achieved on the collection scenes, that one is what the
    # policy achieves alone on the eval scenes.
    a = bx[0][3]
    series = [("co_grasp_ok", "secured", "tab:green"),
              ("co_grasp_miss", "closed, not secured", "tab:olive"),
              ("co_no_release", "no release", "tab:orange"),
              ("co_drop", "drop", "tab:red"),
              ("co_human_contact", "human contact", "tab:purple"),
              ("co_bench_timeout", "benchmark timeout", "tab:brown"),
              ("co_timeout", "never closed", "tab:gray")]
    if _stack(a, it, [num(k) for k, _, _ in series],
              [l for _, l, _ in series], [c for _, _, c in series]):
        a.set_ylim(0, 1)
    _grid(a, "collection outcomes (fraction of kept episodes)")
    _legend(a, loc="lower left", ncol=2)

    # ---- collection success vs eval success ----
    # The GAP is the quantity of interest, not either line. Collection runs at
    # beta (mostly expert) on the training pool; eval runs the policy alone on the
    # eval scenes. A collection line that stays high while eval sags says the data
    # is still good and the policy is not absorbing it; both sagging together says
    # the expert itself is failing on these scenes and the labels are the problem.
    a = bx[1][3]
    cs, es, bt = num("c_success_rate"), num("success_rate"), num("beta")
    if _finite(cs):
        _plot(a, it, cs, "-o", ms=3, color="tab:blue",
              label="collection (beta mixture)")
    if _finite(es):
        _plot(a, it, es, "-s", ms=3, color="tab:green", label="eval (policy alone)")
    if _finite(bt):
        _plot(a, it, bt, ":", color="tab:gray", label="beta (expert share)")
    a.set_ylim(0, 1)
    _grid(a, "success: collection vs eval", ylabel="fraction")
    _legend(a, loc="lower left")

    _fix_x(fig2, it)
    fig2.suptitle(f"Phase-5 DAgger diagnostics — {run.name}"
                  + (f"   [target: {target}]" if pregrasp else ""), fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    diag_out = run / "curves_diag.png"
    fig2.savefig(diag_out, dpi=140)
    print(f"wrote {diag_out}")

    # ── REGRASP (<run>/curves_regrasp.png) — "is the conditioning doing anything" ──
    # Its own figure rather than three more panels on curves.png, because these
    # answer a different question and are blank for every Phase-4 run.
    if _finite(num("cond_track")) or _finite(num("retry_at_2")):
        fig3, ax3 = plt.subplots(1, 3, figsize=(16, 4.4))

        # THE panel. cond_track is the mean pairwise spread of the four final EE
        # poses over the spread of the four commanded grasps. 1.0 = the policy
        # separates the conditions as much as the targets are separated; 0.0 = it
        # does the same thing whatever it is told, which is the multi-modal
        # averaging failure and makes regrasping inert however good success looks.
        # near_rate rides along because the two together say which of the two
        # failure modes is live: both low = the conditioning is ignored; cond_track
        # high and near_rate low = the policy separates the conditions but tracks
        # none of them, which is a reach-endgame problem, not a conditioning one.
        a = ax3[0]
        _plot(a, it, num("cond_track"), "-", marker="o", ms=3, lw=2.2,
              color="tab:purple", label="cond_track (ee spread / goal spread)")
        _plot(a, it, num("near_rate"), "-.", marker=".", ms=3, lw=1.2,
              color="tab:red", label="near_rate")
        a.axhline(0.3, color="0.6", ls=":", lw=1.0)
        a.text(0.01, 0.31, "below this: switch to FiLM", fontsize=7,
               color="0.4", transform=a.get_yaxis_transform())
        a.set_ylim(-0.02, 1.3)
        _grid(a, "is the policy USING the commanded grasp?", ylabel="ratio / fraction")
        _legend(a, loc="upper left")

        # retry@k — the regrasping headline. Derived from the same episodes, so
        # it costs nothing; it assumes each retry restarts from home, which is
        # true of this evaluation and not of a real deployment. Read as a ceiling.
        a = ax3[1]
        for k, col in zip(range(1, 5), ("tab:blue", "tab:green", "tab:orange",
                                        "tab:red")):
            ys = num(f"retry_at_{k}")
            if _finite(ys):
                _plot(a, it, ys, "-", marker="o", ms=3, lw=1.6 + 0.2 * k,
                      color=col, label=f"success @ {k} attempt(s)")
        a.set_ylim(-0.02, 1.02)
        _grid(a, "regrasping: success with k tries", ylabel="fraction of eval scenes")
        _legend(a, loc="lower right")

        # Per-slot rates. Slot 0 is OMG's own pick, so succ_g0 is the curve
        # comparable with a Phase-4 run; the spread is how much harder the
        # deliberately-separated grasps are.
        a = ax3[2]
        for g, col in zip(range(4), ("tab:blue", "tab:green", "tab:orange",
                                     "tab:red")):
            ys = num(f"succ_g{g}")
            if _finite(ys):
                _plot(a, it, ys, "-", marker="o", ms=3, lw=1.4, color=col,
                      label=f"grasp {g}" + (" (OMG pick)" if g == 0 else ""))
            ysn = num(f"near_g{g}")
            if _finite(ysn):
                _plot(a, it, ysn, ":", lw=1.0, color=col)
        a.set_ylim(-0.02, 1.02)
        _grid(a, "per-grasp success (solid) and near (dotted)",
              ylabel="fraction of that slot's episodes")
        _legend(a, loc="upper left", ncol=2)

        _fix_x(fig3, it)
        fig3.suptitle(f"Phase-5 grasp conditioning — {run.name}", fontsize=12)
        fig3.tight_layout(rect=[0, 0, 1, 0.94])
        regrasp_out = run / "curves_regrasp.png"
        fig3.savefig(regrasp_out, dpi=140)
        print(f"wrote {regrasp_out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
