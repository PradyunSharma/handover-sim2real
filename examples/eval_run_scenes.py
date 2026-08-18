"""Score ONE checkpoint over a whole scene pool and report EVERY SCENE.

    # the headline use: best checkpoint, all training scenes
    python examples/eval_run_scenes.py --run-dir output/dagger_runs/dagger4_run12

    # the held-out eval pool instead (reproduces the number in dagger_log.csv)
    python examples/eval_run_scenes.py --run-dir ... --scenes eval

    # a specific iteration rather than the exported best
    python examples/eval_run_scenes.py --run-dir ... --from 8

    # THE GENERALISATION NUMBER: the best checkpoint on the TEST split, scenes
    # the run never collected on or evaluated against
    python examples/eval_run_scenes.py --run-dir output/dagger_runs/dagger4_run16 \
        --split test --grasp-pin-table output/grasp_pin_table_test_omg.json \
        --exclude-scenes none

    # the honest opportunity metric: one-ray gate + a real close to confirm it
    python examples/eval_run_scenes.py --run-dir ... --split test \
        --grasp-pin-table output/grasp_pin_table_test_omg.json \
        --exclude-scenes none --box-probe

OPPORTUNITY IS A TWO-TIER TEST, and the threshold on the first tier is now ONE
RAY. `--min-rays` (default 1) is how many of the 7x7 pad-to-pad rays must hit the
object for a step to clear the geometric gate; `--min-frac` is an optional
fraction floor on top, default 0.0 and inert. `--box-probe` then settles every
step that clears the gate by closing the gripper for real, holding it, scoring it
with the evaluator's own success criterion, and rewinding the simulator. With
both on, `opportunity_rate` means what the word says: the fraction of episodes
with at least one step where the object was between the open pads AND grasping
there would genuinely have secured it.

WHY THE GATE IS NOT 0.50 ANY MORE. Runs 1-20 used a half-pad threshold. Measured
on run 16's best checkpoint over the 130-scene test split it reported a chance in
80 of 130 episodes while 90 secured the object — an opportunity rate BELOW the
success rate, which is impossible if the name is accurate. Recounting the 18
secured-but-unreported episodes off `box_frac_max` found not one at 0.0: their
peaks spread smoothly from 0.02 to 0.49, six of them just under the old gate. So
the rays did see the object every time; the threshold was refusing to call a thin
or small object a chance, because occupancy measures how CENTRED the object is on
the pad, not whether a grasp was available. Recall against secured episodes: 80%
at 0.50, 87% at 0.40, 97% at 0.10, and 100% only at one ray. Precision, which the
old threshold was really buying, now comes from `--box-probe` instead.
`print_summary` still asserts the invariant and names the scenes that break it.
The settings in force are recorded in the summary JSON, because two runs with
different gates are otherwise indistinguishable on disk.

WHY THIS EXISTS, next to eval_dagger_run.py. That script answers "how did the run
progress" — one aggregate row per ITERATION, over the held-out eval pool.
This one answers "where does the best policy actually fail" — one row per SCENE,
over the training pool, with the per-episode record that `evaluate_policy`
already computes and that every other caller throws away by popping `rows`.

Nothing about the rollout is reimplemented here. The scenes, the simulator, the
pin table and the thresholds all come from `build_phase4_context(<run>/config.yaml)`
— the same call the training loop makes — and the episodes run through
`evaluate_policy`, so a number printed here and the same number in
`dagger_log.csv` cannot drift apart. The only new code is selection of the pool,
the per-scene CSV, and the plots.

TRAINING SCENES, NOT EVAL SCENES, IS THE DEFAULT and it is deliberate. These are
the scenes the policy was collected on, so a failure here is not a
generalisation gap — it is the policy failing on data it has seen, which is a
strictly stronger statement and the one worth localising. `--scenes all` gives
both pools together.

`--scenes eval` IS NOT A GENERALISATION NUMBER, despite the name. Every Phase-4
run except run 1 sets `EVAL.holdout: false`, which puts the eval scenes back into
the collection pool — so the in-loop `success_rate` and `--scenes eval` are both
training-split performance on scenes the policy was trained on. The only honest
generalisation figure comes from `--split test`, which re-points the whole SIM
block at a different benchmark split (for SETUP s0: subjects 2-9 instead of 0-9,
at the sequence indices train never uses). See `apply_split_override` — the pin
table and exclusion list are numbered WITHIN a split and must be replaced too.

THE OPPORTUNITY COLUMNS WORK ON OLD RUNS. `opportunity` / `box_taken` are
computed by ray-casting the live simulator during the rollout
(`dagger/grasp_box.py`), not read from anything the run recorded — so runs 1-14,
which predate the metric entirely, can be scored with it retroactively. Only the
checkpoint has to exist.

COST. One episode is ~2-3 s, so a 350-scene training pool is ~15-20 min on a GPU.
`--box-probe` adds ~0.5 s per gate pass on top, which at the one-ray gate is the
larger share of the run. Read-only on the run directory apart from the three
output files it writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root -> handover_sim2real
sys.path.insert(0, _HERE)                    # examples/

from handover_sim2real.dagger import evaluate_policy, load_policy_runner  # noqa: E402
from handover_sim2real.dagger.setup import build_phase4_context           # noqa: E402


# Per-scene CSV columns: every field `_eval_episode` returns, in a fixed order so
# the file is diffable between runs. `reason` last because it is the wide one.
SCENE_FIELDS = [
    "scene_idx",
    "success", "grasped", "closed", "near",
    "close_step", "dist",
    "pos_err", "rot_err",          # reach AT THE CLOSE (NaN if it never closed)
    "min_pos", "min_rot",          # closest reach over the WHOLE episode
    "had_chance", "missed",        # pinned-pose opportunity (see the caveat below)
    # `opportunity` replaces the old `box_chance` column. Same field, honest name.
    "opportunity", "box_taken", "box_missed", "box_steps", "box_frac_max",
    "status", "reason",
]

# Outcome categories, ordered best -> worst so the strip plot reads top to bottom.
# `_status_name` can OR two failures into "DROP|HUMAN_CONTACT", so anything not
# in this list is appended at the end rather than dropped.
REASON_ORDER = ["GRASP_OK", "GRASP_MISS", "NO_RELEASE", "DROP",
                "HUMAN_CONTACT", "TIMEOUT"]
REASON_COLOR = {
    "GRASP_OK":      "#2e7d32",   # green
    "GRASP_MISS":    "#ef6c00",   # orange
    "NO_RELEASE":    "#6a1b9a",   # purple
    "DROP":          "#c62828",   # red
    "HUMAN_CONTACT": "#5d4037",   # brown
    "TIMEOUT":       "#757575",   # grey
}
_OTHER_COLOR = "#000000"


def resolve_ckpt_dir(run_root: Path, spec: str) -> tuple[Path, str]:
    """(directory to load from, checkpoint name) for --from.

    'best' / 'last' are the exported snapshots train_dagger_phase4 writes at
    <run>/best and <run>/last; an integer is an iteration under <run>/iters.
    The exported dirs carry their own config.yaml and normalization.npz, which
    is what makes them loadable standalone.
    """
    if spec in ("best", "last"):
        d = run_root / spec
        if not (d / "checkpoints").is_dir():
            raise SystemExit(
                f"{d}/checkpoints does not exist. The run may still be training "
                f"(the export is written after the winning iteration is scored) "
                f"— pass --from <iteration> instead.")
        # The snapshot dir holds exactly one file, named for what it snapshotted.
        pts = sorted(p.stem for p in (d / "checkpoints").glob("*.pt"))
        if not pts:
            raise SystemExit(f"{d}/checkpoints is empty — nothing to score.")
        return d, (spec if spec in pts else pts[0])
    try:
        i = int(spec)
    except ValueError:
        raise SystemExit(f"--from must be 'best', 'last' or an iteration number, "
                         f"got {spec!r}")
    d = run_root / "iters" / f"iter_{i:02d}"
    if not (d / "checkpoints").is_dir():
        raise SystemExit(f"no checkpoints under {d}")
    return d, "best"


def select_scenes(ctx, which: str) -> list[int]:
    if which == "train":
        return list(ctx.pool)
    if which == "eval":
        return list(ctx.eval_scenes)
    return sorted(set(ctx.pool) | set(ctx.eval_scenes))


def apply_split_override(cfg4: dict, split: str, pin: str | None,
                         exclude: str | None) -> None:
    """Re-point the run's SIM block at a DIFFERENT benchmark split.

    THE POINT: every number in dagger_log.csv is training-split performance —
    `EVAL.holdout: false` means the eval scenes were also collected on. A
    generalisation claim needs the checkpoint rolled out on scenes the run never
    saw, i.e. `--split test`, which for BENCHMARK.SETUP s0 is subjects 2-9 at
    sequence indices 4 mod 5. The policy is unchanged; only which scenes it faces.

    SCENE INDICES ARE SPLIT-RELATIVE. `HandoverBenchmarkWrapper` rebuilds
    `_scene_ids` from the split, so scene 7 of test is not scene 7 of train. Two
    inputs are numbered in that space and BOTH must move with the split, or they
    silently describe the wrong scenes:

      grasp_pin_table   keys are scene indices. A train table applied to test
                        pins an unrelated grasp on every scene. It also doubles
                        as the usable-scene filter (the ~13% of scenes with no
                        reachable goal set), so getting it wrong changes the
                        denominator as well as the target.
      exclude_scenes    the list of scenes whose expert demonstration failed.
                        Train-relative, and no test equivalent is built by
                        default, so pass 'none' unless you have one.

    Without a pin table the run still scores: `stable_grasp` needs no grasp pose
    (it is a physics check), so success/grasp/close are all still valid. What is
    lost is every POSE column — pos_err, min_pos, had_chance, missed come out NaN
    — and the pool is no longer filtered, so scenes the expert itself cannot plan
    for are counted as policy failures. That understates the policy, and the
    caller is warned rather than stopped, because an unfiltered test number is
    still a real number as long as it is labelled as one.
    """
    sim = cfg4.setdefault("SIM", {})
    sim["split"] = split
    if pin is not None:
        sim["grasp_pin_table"] = (None if pin.lower() == "none" else pin)
    if exclude is not None:
        sim["exclude_scenes"] = (None if exclude.lower() == "none" else exclude)


def write_scene_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCENE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SCENE_FIELDS})
    tmp.replace(path)


def summarise(m: dict, rows: list[dict], scenes: list[int], which: str,
              run_root: Path, ckpt_dir: Path, ckpt: str) -> dict:
    """Aggregate block, printed and saved as JSON."""
    n_close = sum(r["closed"] for r in rows)
    n_box = sum(r["opportunity"] for r in rows)
    # The scenes that break the success ⊆ opportunity invariant: the policy came
    # away with the object, yet no step of the episode was ever called an
    # opportunity. Listed by scene index (SPLIT-RELATIVE) so they can be replayed
    # one at a time — see the rollout command in print_summary's warning.
    broken = sorted(int(r["scene_idx"]) for r in rows
                    if r["success"] and not r["opportunity"])
    return {
        "run": str(run_root), "ckpt_dir": str(ckpt_dir), "ckpt": ckpt,
        "scene_set": which, "n_scenes": len(scenes),
        "success_rate": m["success_rate"],
        "close_rate": m["close_rate"],
        # THE one the user's question names: of the scenes where the policy
        # decided to close, how many became a secured grasp. Conditional, so it
        # separates "closed at the wrong place" from "never closed at all".
        "close_success_rate": m["close_success_rate"],
        "n_closed": int(n_close),
        "grasp_rate": m["grasp_rate"],
        "near_rate": m["near_rate"],
        "opportunity_rate": m["opportunity_rate"],
        # Pre-rename alias, so anything already parsing these JSONs keeps working.
        "box_chance_rate": m["box_chance_rate"],
        # THE conversion: of the scenes that had a real opportunity, how many the
        # policy came away from with the object secured. `miss_given_box` is its
        # exact complement, kept because a failure count reads more directly.
        #
        # `box_taken_rate`, `mean_box_steps` and `chance_rate` are dropped here
        # on purpose — see the note in print_summary. They are still computed by
        # `evaluate_policy` and still logged to dagger_log.csv, so nothing about
        # the training curves changes.
        "box_success_rate": m["box_success_rate"],
        "miss_given_box": m["miss_given_box"],
        "n_opportunity": int(n_box),
        "n_box_chance": int(n_box),
        # Scenes where success=1 but opportunity=0 — the invariant violations,
        # for replay. Empty is what a correct opportunity measure looks like.
        "success_without_opportunity": broken,
        "mean_box_frac": m["mean_box_frac"],
        # box_probe only: gate passes, and the fraction a real close confirmed.
        "n_probes": m.get("n_probes", 0),
        "probe_pass_rate": m.get("probe_pass_rate", float("nan")),
        "eval_min_pos": m["eval_min_pos"], "eval_min_rot": m["eval_min_rot"],
        "mean_pos_err": m["mean_pos_err"], "mean_rot_err": m["mean_rot_err"],
        "mean_close_step": m["mean_close_step"], "mean_dist": m["mean_dist"],
        "reasons": m["reasons"],
    }


def print_summary(s: dict, params) -> None:
    def pct(x):
        return "  n/a " if x != x else f"{100*x:5.1f}%"
    print("\n" + "=" * 78)
    print(f"{s['n_scenes']} {s['scene_set']} scenes   {s['ckpt']}.pt from {s['ckpt_dir']}")
    print("-" * 78)
    # One padding width for every metric line in the report, so the percentages
    # form a single column. 21 is set by the longest label, "success | opportunity".
    def row(label, value, note):
        print(f"  {label:<21} {pct(value)}   {note}")

    row("success_rate", s["success_rate"], "secured grasps / all scenes")
    row("close_rate", s["close_rate"],
        f"the policy committed a close at all ({s['n_closed']}/{s['n_scenes']})")
    row("close_success_rate", s["close_success_rate"],
        "of those closes, how many held")
    row("grasp_rate", s["grasp_rate"], "fingers on the object after the hold")
    print("-" * 78)
    # Spell out which definition produced the number, because the two differ by
    # a lot and a bare percentage does not say which one it is.
    if params.box_probe:
        defn = (f">={params.box.min_rays} ray(s) hit the object AND a real close "
                f"there secured it")
    else:
        defn = (f">={params.box.min_rays} ray(s) hit the object (UPPER BOUND — "
                f"no probe, the human hand is invisible to the rays)")
    # DELIBERATELY NOT REPORTED HERE, though `evaluate_policy` still computes
    # them and the training loop still logs them:
    #   box_taken_rate ("closed | opportunity")  With the probe on this is the
    #       SAME NUMBER as success | opportunity, by construction rather than by
    #       coincidence: opportunity means "closing here secures the object", so
    #       a close commanded on such a step must secure it in a deterministic
    #       sim — the probe ran the identical close-hold-score procedure. Two
    #       lines, one fact.
    #   mean_box_steps   How many steps the window stayed open. It counts steps
    #       that cleared the gate, not anything the policy could act on, and its
    #       value moves with the gate threshold rather than with behaviour.
    #   chance_rate      Proximity to the PINNED grasp. Reads 0.0% on runs that
    #       succeed 69% of the time, because the policy's grasps are real but
    #       nowhere near OMG's pose — it measures pin agreement, which near_rate
    #       already reports.
    row("opportunity", s["opportunity_rate"],
        f"{defn} ({s['n_opportunity']}/{s['n_scenes']})")
    row("success | opportunity", s["box_success_rate"],
        "...and the policy converted it  <- conversion")
    row("miss | opportunity", s["miss_given_box"],
        "...and it did not (the exact complement of the line above)")
    if s.get("n_probes"):
        row("probe_pass_rate", s["probe_pass_rate"],
            f"of {s['n_probes']} geometric gate passes, how many a REAL close "
            f"confirmed")

    # THE INVARIANT. A grasp cannot secure the object without there having been a
    # step at which grasping would secure the object — the policy's own close IS
    # such a step — so success ⊆ opportunity must hold, scene by scene. Any scene
    # that breaks it is a defect in the MEASURE, not in the policy, and every
    # conditional printed above rests on a denominator that excludes it. So name
    # the offenders individually rather than reporting an aggregate discrepancy:
    # they are few enough to replay, and replaying one is what settles the cause.
    broken = s.get("success_without_opportunity", [])
    bc, sr = s.get("opportunity_rate", float("nan")), s.get("success_rate", 0.0)
    if broken:
        print(f"  !! {len(broken)} scene(s) secured the object with NO step ever "
              f"counted as an opportunity ({pct(bc)} opportunity vs "
              f"{pct(sr)} success). Treat the conditionals above as conditioned "
              f"on a selected subset.")
        print(f"     scene_idx (split-relative): "
              f"{', '.join(str(x) for x in broken)}")
        print(f"     Replay one with rollout_bc_policy.py --scene <idx> to see "
              f"where the object actually sits at the close. The CSV's "
              f"box_frac_max says which cause: >0 means the rays DID see it and "
              f"the gate or the probe rejected it anyway; exactly 0 means the "
              f"object never intersected the ray volume, i.e. it was gripped "
              f"outside the pad's z-window (only the distal 17.6 mm, "
              f"z 0.0946-0.1122, is sampled).")
        if params.box.min_rays > 1 or params.box.min_frac > 0:
            print(f"     The gate is not at its most permissive "
                  f"(min_rays={params.box.min_rays}, "
                  f"min_frac={params.box.min_frac:.2f}); try --min-rays 1 "
                  f"--min-frac 0 first, or recount offline from box_frac_max in "
                  f"the CSV — a step clears fraction t exactly when "
                  f"box_frac_max >= t.")
    elif bc == bc:
        print(f"  ok success ⊆ opportunity holds on every scene.")
    print("-" * 78)
    print(f"  min reach           pos {s['eval_min_pos']:.4f} m   "
          f"rot {s['eval_min_rot']:.4f} rad   (closest approach, every episode)")
    print(f"  reach at close      pos {s['mean_pos_err']:.4f} m   "
          f"rot {s['mean_rot_err']:.4f} rad   (episodes that closed)")
    print(f"  thresholds          pos {params.close_pos_thresh:.3f} m   "
          f"rot {params.close_rot_thresh:.3f} rad")
    print("-" * 78)
    tot = max(s["n_scenes"], 1)
    for k in sorted(s["reasons"], key=lambda x: -s["reasons"][x]):
        print(f"  {k:16s} {s['reasons'][k]:4d}  {100*s['reasons'][k]/tot:5.1f}%")
    print("=" * 78)


def make_plots(rows: list[dict], s: dict, params, out_png: Path,
               sort_by: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order = list(range(len(rows)))
    if sort_by == "min_pos":
        order.sort(key=lambda i: (not np.isfinite(rows[i]["min_pos"]),
                                  rows[i]["min_pos"]))
    elif sort_by == "outcome":
        rank = {r: k for k, r in enumerate(REASON_ORDER)}
        order.sort(key=lambda i: (rank.get(rows[i]["reason"], len(REASON_ORDER)),
                                  rows[i]["min_pos"]))
    R = [rows[i] for i in order]
    x = np.arange(len(R))
    col = [REASON_COLOR.get(r["reason"], _OTHER_COLOR) for r in R]
    ok = np.array([bool(r["success"]) for r in R])

    fig, ax = plt.subplots(5, 1, figsize=(max(11, len(R) * 0.05), 17),
                           sharex=True)
    fig.suptitle(
        f"{Path(s['run']).name} — {s['ckpt']}.pt over {s['n_scenes']} "
        f"{s['scene_set']} scenes   "
        f"success {100*s['success_rate']:.1f}%   "
        f"close-success {100*s['close_success_rate']:.1f}%",
        fontsize=13)

    # ---- 1. closest the EE ever came to the pinned grasp, position -----------
    # Defined for every episode, including the ones that never closed — which is
    # exactly where success_rate carries no information at all.
    a = ax[0]
    mp = np.array([r["min_pos"] for r in R], dtype=float)
    a.bar(x, np.nan_to_num(mp, nan=0.0), color=col, width=1.0, linewidth=0)
    a.axhline(params.close_pos_thresh, color="k", ls="--", lw=1,
              label=f"close tol {params.close_pos_thresh:.3f} m")
    a.set_ylabel("min |pos| to grasp (m)")
    a.set_title("Closest approach over the episode — position "
                "(bar colour = final outcome)", fontsize=10, loc="left")
    a.legend(fontsize=8, loc="upper right")

    # ---- 2. same, rotation --------------------------------------------------
    a = ax[1]
    mr = np.array([r["min_rot"] for r in R], dtype=float)
    a.bar(x, np.nan_to_num(mr, nan=0.0), color=col, width=1.0, linewidth=0)
    a.axhline(params.close_rot_thresh, color="k", ls="--", lw=1,
              label=f"close tol {params.close_rot_thresh:.3f} rad")
    a.set_ylabel("min |rot| to grasp (rad)")
    a.set_title("Closest approach over the episode — orientation", fontsize=10,
                loc="left")
    a.legend(fontsize=8, loc="upper right")

    # ---- 3. reach AT THE CLOSE ---------------------------------------------
    # Only exists where the policy closed; a gap here is "never committed",
    # which is a different failure from "committed in the wrong place".
    a = ax[2]
    pe = np.array([r["pos_err"] for r in R], dtype=float)
    have = np.isfinite(pe)
    a.scatter(x[have & ok], pe[have & ok], s=14, c=REASON_COLOR["GRASP_OK"],
              label="closed, held")
    a.scatter(x[have & ~ok], pe[have & ~ok], s=14, c=REASON_COLOR["DROP"],
              marker="x", label="closed, lost it")
    for i in x[~have]:
        a.axvline(i, color="#cccccc", lw=0.6, zorder=0)
    a.axhline(params.close_pos_thresh, color="k", ls="--", lw=1)
    a.set_ylabel("|pos| at close (m)")
    a.set_title("Reach at the moment of the close  "
                "(grey column = never closed)", fontsize=10, loc="left")
    a.legend(fontsize=8, loc="upper right")

    # ---- 4. outcome per scene ----------------------------------------------
    a = ax[3]
    cats = REASON_ORDER + sorted({r["reason"] for r in R} - set(REASON_ORDER))
    ypos = {c: len(cats) - 1 - k for k, c in enumerate(cats)}
    a.scatter(x, [ypos[r["reason"]] for r in R], s=12, c=col)
    a.set_yticks(range(len(cats)))
    a.set_yticklabels([c for c in cats][::-1], fontsize=8)
    a.set_ylabel("outcome")
    a.set_title("Per-scene outcome", fontsize=10, loc="left")
    a.grid(axis="y", lw=0.3, alpha=0.5)

    # ---- 5. geometric grasp opportunity ------------------------------------
    # box_frac_max is the BEST jaw occupancy the episode ever reached, so the bar
    # height says how close the scene came to a chance even when it never had
    # one — and min_frac can be recalibrated off this plot without re-running.
    a = ax[4]
    bf = np.array([r["box_frac_max"] for r in R], dtype=float)
    chance = np.array([bool(r["opportunity"]) for r in R])
    taken = np.array([bool(r["box_taken"]) for r in R])
    a.bar(x, bf, width=1.0, linewidth=0,
          color=np.where(chance, "#1565c0", "#bdbdbd"))
    # Per-scene outcome of every opportunity. The middle category is empty
    # WHENEVER --box-probe is on and non-empty otherwise, which is itself the
    # readable difference between the two definitions: a probed opportunity
    # cannot be closed on and lost, because it was confirmed by that very close.
    held = taken & ok
    lost = taken & ~ok
    declined = chance & ~taken
    a.scatter(x[held], bf[held] + 0.04, s=22, marker="v",
              c=REASON_COLOR["GRASP_OK"], label="opportunity converted")
    if lost.any():
        a.scatter(x[lost], bf[lost] + 0.04, s=22, marker="v",
                  c=REASON_COLOR["GRASP_MISS"], label="closed on it, lost it")
    a.scatter(x[declined], bf[declined] + 0.04, s=22, marker="v",
              c=REASON_COLOR["DROP"], label="opportunity declined")
    # The gate line, drawn where it actually bites: the binding constraint is
    # whichever of the two thresholds is higher in fraction terms, and at the
    # default (1 ray, no floor) that is a single grid cell.
    n_rays = int(params.box.grid_x) * int(params.box.grid_z)
    gate = max(float(params.box.min_frac), params.box.min_rays / n_rays)
    a.axhline(gate, color="k", ls="--", lw=1,
              label=f"gate {gate:.3f} ({params.box.min_rays}/{n_rays} rays"
                    + (f", floor {params.box.min_frac:.2f}"
                       if params.box.min_frac > 0 else "") + ")")
    a.set_ylim(0, 1.15)
    a.set_ylabel("max jaw occupancy")
    a.set_xlabel(f"scene (sorted by {sort_by}; see the CSV for scene_idx)")
    # NaN when --no-box, or when no episode ever presented a chance; "n/a" says
    # "not measured" where a printed 0% would read as "never converted one".
    def _p(v):
        return "n/a" if v != v else f"{100 * v:.0f}%"
    a.set_title(
        f"Grasp opportunity — fraction of the pad-to-pad rays that hit the "
        f"object, best step of the episode   "
        f"[opportunity {_p(s['opportunity_rate'])}, "
        f"success | opportunity {_p(s['box_success_rate'])}]",
        fontsize=10, loc="left")
    a.legend(fontsize=8, loc="upper right")

    handles = [Line2D([], [], marker="s", ls="", color=REASON_COLOR.get(c, _OTHER_COLOR),
                      label=c) for c in cats]
    fig.legend(handles=handles, loc="lower center", ncol=len(cats), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="output/dagger_runs/<name> (must contain config.yaml)")
    p.add_argument("--from", dest="from_", default="best",
                   help="'best' (default), 'last', or an iteration number")
    p.add_argument("--split", default=None, choices=["train", "val", "test"],
                   help="roll out on a DIFFERENT benchmark split than the run "
                        "trained on — the only way to get a generalisation "
                        "number, since EVAL.holdout is false and every figure in "
                        "dagger_log.csv is training-split performance. Implies "
                        "--scenes all (the split's own pool has no train/eval "
                        "division). Scene indices are split-relative, so pass "
                        "--grasp-pin-table for the matching split and "
                        "--exclude-scenes none.")
    p.add_argument("--grasp-pin-table", default=None,
                   help="override SIM.grasp_pin_table ('none' to disable). "
                        "REQUIRED with --split unless you accept NaN pose "
                        "columns and an unfiltered pool.")
    p.add_argument("--exclude-scenes", default=None,
                   help="override SIM.exclude_scenes ('none' to disable). The "
                        "shipped list is train-relative, so pass 'none' with "
                        "--split test.")
    p.add_argument("--scenes", default="train", choices=["train", "eval", "all"],
                   help="which pool to score (default: train — the collection pool)")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="cap how many of the selected scenes are rolled out "
                        "(evenly spread, not a prefix)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override EVAL.max_steps for this sweep")
    p.add_argument("--min-rays", type=int, default=1,
                   help="how many of the 49 pad-to-pad rays must hit the object "
                        "for a step to clear the GEOMETRIC GATE. Default 1: the "
                        "gate's job is to be cheap and to miss nothing, with "
                        "precision supplied by --box-probe. Raising it re-"
                        "introduces the under-count that made opportunity read "
                        "BELOW success on run 16's test split.")
    p.add_argument("--min-frac", type=float, default=0.0,
                   help="optional fraction floor on top of --min-rays, i.e. what "
                        "share of the 7x7 grid must hit. Default 0.0 (inert). "
                        "Pass 0.5 to reproduce the gate runs 1-20 used, which is "
                        "what dagger_log.csv's box_* columns were logged with — "
                        # NB argparse %-expands help strings, so a literal
                        # percent sign must be doubled or --help raises.
                        "but note it recovers only 80%% of the episodes that "
                        "actually secured the object.")
    p.add_argument("--box-probe", action="store_true",
                   help="HIERARCHICAL opportunity test. The ray grid becomes a "
                        "cheap gate; every step that passes it is confirmed by "
                        "actually closing the gripper, holding, and scoring with "
                        "the evaluator's own success criterion — so a step "
                        "counts only if a close there would really have secured "
                        "the object. Folds in what geometry cannot see (the "
                        "human hand is invisible to the rays; mass and friction "
                        "are not modelled by them). The simulator is rewound "
                        "afterwards and the rewind is verified, so the episode "
                        "is unaffected. Roughly doubles wall clock.")
    p.add_argument("--no-box", action="store_true",
                   help="skip the ray-cast opportunity test (its columns go NaN)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-prefix", default=None,
                   help="output basename (default: <run>/scene_eval_<pool>)")
    p.add_argument("--sort", default="scene",
                   choices=["scene", "min_pos", "outcome"],
                   help="x-axis order in the plot (the CSV is always scene order)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the per-scene progress line")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_dir)
    cfg_path = run_root / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config.yaml in {run_root} — is that a Phase-4 run dir?")
    with cfg_path.open() as f:
        cfg4 = yaml.safe_load(f)

    ev = cfg4.setdefault("EVAL", {})
    if args.max_steps is not None:
        ev["max_steps"] = int(args.max_steps)
    if args.no_box:
        ev["box_check"] = False
    if args.box_probe:
        ev["box_probe"] = True
    # Threaded through the config rather than set on BoxParams afterwards, so it
    # goes through the same `build_box_params` the training loop uses and the two
    # cannot end up measuring against differently-constructed cuboids.
    ev.setdefault("box", {})["min_frac"] = float(args.min_frac)
    ev["box"]["min_rays"] = int(args.min_rays)
    seed = int(args.seed if args.seed is not None
               else cfg4.get("DAGGER", {}).get("seed", 0))

    # ---- cross-split evaluation (see apply_split_override) ----
    pool_label = args.scenes
    if args.split:
        train_split = str(cfg4.get("SIM", {}).get("split", "train"))
        apply_split_override(cfg4, args.split, args.grasp_pin_table,
                             args.exclude_scenes)
        # `holdout` splits the split's scenes into a collection pool and an eval
        # pool. On a foreign split that division is meaningless — none of them
        # were collected on — so take all of them and say so in the labels.
        ev["holdout"] = False
        ev["num_scenes"] = 10 ** 6      # scene_pools clamps to the pool size
        args.scenes = "all"
        pool_label = f"{args.split}-split"
        print(f"[split] rolling out on {args.split!r}; the run trained on "
              f"{train_split!r}")
        if args.grasp_pin_table is None:
            print(f"[split] WARNING no --grasp-pin-table given, so "
                  f"{cfg4['SIM'].get('grasp_pin_table')!r} (numbered in the "
                  f"{train_split} split) will be applied to {args.split} scene "
                  f"indices. Pass the matching table, or 'none' to disable "
                  f"pinning and accept NaN pose columns.")
        if args.exclude_scenes is None and cfg4["SIM"].get("exclude_scenes"):
            print(f"[split] WARNING exclude_scenes "
                  f"{cfg4['SIM']['exclude_scenes']!r} is numbered in the "
                  f"{train_split} split; pass --exclude-scenes none.")

    ckpt_dir, ckpt = resolve_ckpt_dir(run_root, args.from_)

    # Same call the trainer makes, so pools/thresholds/pin table are identical.
    ctx = build_phase4_context(cfg4, seed=seed)
    scenes = select_scenes(ctx, args.scenes)
    if args.num_scenes is not None and args.num_scenes < len(scenes):
        # Evenly spread rather than a prefix: scene ids are ordered by object,
        # so a prefix would sample one corner of the object set.
        idx = np.linspace(0, len(scenes) - 1, args.num_scenes).astype(int)
        scenes = [scenes[i] for i in sorted(set(idx.tolist()))]

    ctx.eval_params.verbose = not args.quiet
    src = (run_root / "best" / "source.txt")
    print("=" * 78)
    print(f"Per-scene eval   run={run_root.name}   pool={pool_label} "
          f"({len(scenes)} scenes)")
    print(f"  checkpoint  : {ckpt}.pt from {ckpt_dir}")
    if args.from_ == "best" and src.exists():
        print("  provenance  : " + src.read_text().strip().replace("\n", " | "))
    print(f"  success mode: {ctx.eval_params.success_mode}   "
          f"max_steps={ctx.eval_params.max_steps}   "
          f"hold={ctx.eval_params.hold_steps}")
    bx = ctx.eval_params.box
    if not ctx.eval_params.box_check:
        print("  opportunity : OFF")
    else:
        n_rays = int(bx.grid_x) * int(bx.grid_z)
        print(f"  opportunity : gate >= {bx.min_rays}/{n_rays} rays"
              + (f" and frac >= {bx.min_frac:.2f}" if bx.min_frac > 0 else "")
              + ("  + PROBE (every gate pass confirmed by actually closing)"
                 if ctx.eval_params.box_probe else
                 "  (NO probe — geometry only, an UPPER BOUND)"))
        # Runs 1-20 logged box_* at min_frac 0.50 with no probe. Anything else is
        # a different definition of the word, and mixing the two on one axis is
        # the mistake this note exists to prevent.
        if abs(bx.min_frac - 0.50) > 1e-9 or ctx.eval_params.box_probe:
            print(f"                NOTE this is NOT the definition "
                  f"dagger_log.csv was logged with (min_frac 0.50, no probe), so "
                  f"the opportunity block below is not comparable to the "
                  f"training curves. Pass --min-frac 0.5 --min-rays 1 without "
                  f"--box-probe for a comparable figure.")
        if bx.min_frac > 0.95:
            print(f"                WARNING at 1.00 every one of the {n_rays} "
                  f"rays must hit. Measured on IDEAL pinned grasp poses, pad "
                  f"occupancy is mean 0.949 / p10 0.798 / min 0.592, so ~10%+ of "
                  f"scenes cannot register a chance even when perfectly "
                  f"positioned. Expect opportunity to collapse and the "
                  f"conditionals below to rest on a small denominator.")
    print(f"  pinning     : {ctx.pin_table.describe() if ctx.pin_table else 'OFF'}")
    print("=" * 78)

    runner, _ = load_policy_runner(ckpt_dir, args.device, ckpt=ckpt)

    t0 = time.time()
    m = evaluate_policy(ctx.sim, runner, scenes,
                        params=ctx.eval_params, pin_table=ctx.pin_table)
    rows = m["rows"]
    dt = time.time() - t0

    prefix = Path(args.out_prefix) if args.out_prefix \
        else run_root / f"scene_eval_{pool_label}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_scene_csv(prefix.with_suffix(".csv"), rows)

    s = summarise(m, rows, scenes, pool_label, run_root, ckpt_dir, ckpt)
    s["eval_s"] = round(dt, 1)
    # Saved so the JSON says which definition of "an opportunity" produced its
    # numbers. Without it a gated-and-probed result and a bare geometric one are
    # indistinguishable on disk and will eventually be plotted on one axis.
    s["min_frac"] = (float(ctx.eval_params.box.min_frac)
                     if ctx.eval_params.box_check else None)
    s["min_rays"] = (int(ctx.eval_params.box.min_rays)
                     if ctx.eval_params.box_check else None)
    s["box_probe"] = bool(ctx.eval_params.box_probe)
    with prefix.with_suffix(".json").open("w") as f:
        json.dump(s, f, indent=2)

    print_summary(s, ctx.eval_params)
    print(f"\n  per-scene CSV : {prefix.with_suffix('.csv')}")
    print(f"  summary JSON  : {prefix.with_suffix('.json')}")
    if not args.no_plot:
        make_plots(rows, s, ctx.eval_params, prefix.with_suffix(".png"), args.sort)
        print(f"  plots         : {prefix.with_suffix('.png')}")
    print(f"  wall clock    : {dt/60:.1f} min ({dt/max(len(scenes),1):.1f} s/scene)")


if __name__ == "__main__":
    main()
