"""
Score a finished PHASE-4 DAgger run on a WHOLE split — held-out test, or the
full train set.

    python examples/eval_dagger_testset.py --run-dir output/dagger_runs/dagger4_run19
    python examples/eval_dagger_testset.py --run-dir output/dagger_runs/dagger4_run19 \\
        --iters 0,5,10,15,20,24
    python examples/eval_dagger_testset.py --run-dir output/dagger_runs/dagger4_run19 \\
        --split train                    # every train scene, not the 100-scene subsample
    python examples/eval_dagger_testset.py --run-dir output/dagger_runs/dagger4_run19 \\
        --plot-only                      # re-render from an existing CSV

WHY THIS EXISTS. Phase 4 has never had a held-out number. `eval_dagger_run.py`
takes no `--split`: it rebuilds the run's own context from the run's own
config.yaml, and every Phase-4 config carries `SIM.split: train` with
`EVAL.holdout: false` and `EVAL.num_scenes: 100`. So the in-loop curve and the
standalone re-score are the SAME measurement — a `np.linspace` subsample of 100
of ~623 usable TRAIN scenes, all of which were also collected on. That is
deliberate for tracking a run (a fixed subsample makes consecutive iterations a
paired comparison) and it is not a generalisation number, and `dagger4_run19`'s
headline has been quoted as though it were.

This is the Phase-4 twin of `eval_regrasp_testset.py`, and it is deliberately
much smaller. Phase 4 conditions the policy on NOTHING — no goal pose, no
direction — so everything that file exists for beyond the split repoint (the
per-bin panels, `dir_err`, the retry ladder, the adaptive bin ranker) has no
referent here. What is left is: point the config at the test split, refuse the
two mismatches that would silently corrupt the answer, and report the same rate
family the training curve does.

  --split test   (the default) the benchmark's own s0 test split — 144 scenes
      with a plannable goal set, none of them ever collected on. THE
      generalisation number.

  --split train  every usable train scene at ~6x the in-loop sample. Still not
      held out; it is the in-loop curve's own question asked where the binomial
      error is half as wide. Never report it as generalisation.

TWO MISMATCHES THIS REFUSES, both silent if allowed through

  THE PIN TABLE'S PROVENANCE. `SIM.grasp_pin_table` decides which grasp each
      scene's close is scored against, and which scenes are usable at all. The
      table records how it was built — `mode`, `setup`, `hand_collision_filter`,
      `hand_collision_thresh`, `valid_grasp_dict_path` — and a test table built
      under different settings is scoring a different task. Run 19 pins with
      `mode: omg` and no hand-collision filter; `grasp_pin_table_test_omg.json`
      matches on all five, which is checked rather than assumed.

  `exclude_scenes` IS A TRAIN ARTIFACT. `output/bc_dataset/train_pinned_omg_
      right_ok.json` lists the TRAIN scenes whose base demonstration succeeded.
      Nothing was collected on test, so it has nothing to say there, and
      applying it would filter the test split by an unrelated split's collection
      outcome — dropping test scenes because a train scene with the same integer
      id failed. It is popped explicitly on test/val and kept on train, where it
      is the right filter for exactly the same reason.

OUTPUT, named by split so a train sweep cannot overwrite a test one

  <run>/<split>_log.csv       one row per iteration, the eval_log.csv schema plus
                              the failure-conditioned `ff_*` block and this
                              script's own provenance columns
  <run>/<split>_eval.png      curves against DAgger iteration — the four panels
                              of curves.png's top row, plus the outcome stack
  <run>/<split>_summary.png   THE FIGURE TO READ. One iteration, as bars: the
                              nested funnel, chance vs conversion, the approach
                              errors, and the outcome taxonomy. A test sweep is
                              often a single checkpoint, where every curve panel
                              is one point and the stacked-area one renders blank.

CHECKPOINT. Defaults to the run's own `EVAL.ckpt`, which is `best` for every
Phase-4 run — unlike Regrasp, which moved to `last` at run 2 so that "the policy
at iteration i" names one set of weights everywhere. Scoring Phase 4 on `last`
is a legitimate question and `--ckpt last` asks it, but it is not what the run's
own reported numbers mean, so it is not the default.

PREREQUISITE: the split's pin table. It already exists for test — this is a
separate sim pass and was run on 2026-08-17:

    ls -la output/grasp_pin_table_test_omg.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from handover_sim2real.dagger import (                        # noqa: E402
    evaluate_policy, load_policy_runner,
)
from handover_sim2real.dagger.setup import build_phase4_context   # noqa: E402

from eval_dagger_run import EVAL_FIELDS, iteration_dirs       # noqa: E402


# The eval taxonomy, as fractions of the eval set (`f_*`, stacking to 1.0) and
# again conditioned on FAILURE (`ff_*`, stacking to 1.0 over the episodes that
# came away with nothing). Phase 4's evaluator emits the first and not the
# second — `ff_*` is computed here from `reasons` and `n_fail`.
#
# WHY BOTH. `f_timeout` answers "how much of the eval set timed out", which moves
# whenever success moves and so cannot separate a policy that got worse from one
# that failed differently. `ff_timeout` answers "of the ones that failed, how many
# timed out", which is the shape of the failure and is what says whether a drop
# in success came from never closing or from closing and dropping. Two runs at
# the same success rate for opposite reasons are indistinguishable in the first
# and obvious in the second.
EVAL_REASONS = {"GRASP_OK": "f_grasp_ok", "GRASP_MISS": "f_grasp_miss",
                "NO_RELEASE": "f_no_release", "DROP": "f_drop",
                "TIMEOUT": "f_timeout", "HUMAN_CONTACT": "f_human_contact"}
FAIL_REASONS = {"GRASP_MISS": "ff_grasp_miss", "NO_RELEASE": "ff_no_release",
                "DROP": "ff_drop", "TIMEOUT": "ff_timeout",
                "HUMAN_CONTACT": "ff_human_contact"}

# The provenance fields a split's pin table must share with the run's own. All
# five change what is being scored, none of them raise if they differ, and the
# table records every one — so the check is exact rather than a heuristic.
PIN_PROVENANCE = ("mode", "tol", "setup", "hand_collision_filter",
                  "hand_collision_thresh", "valid_grasp_dict_path")

TEST_FIELDS = (["iter", "run_dir", "ckpt", "split", "num_scenes", "n_episodes",
                "eval_s", "pin_table", "excluded_applied"]
               + [c for c in EVAL_FIELDS
                  if c not in ("iter", "run_dir", "ckpt", "num_scenes", "eval_s")]
               + ["n_fail"] + list(FAIL_REASONS.values())
               + ["opportunity_rate", "box_success_rate", "mean_box_steps",
                  "mean_box_frac", "probe_pass_rate"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="output/dagger_runs/<name>")
    p.add_argument("--out", default=None,
                   help="default <split>_log.csv, so a train sweep cannot "
                        "silently overwrite a test one")
    p.add_argument("--iters", default="all",
                   help="'all' or a comma list, e.g. 0,5,10,15,20")
    p.add_argument("--split", default="test", choices=["test", "val", "train"],
                   help="handover-sim split to score on. `train` scores the FULL "
                        "training set, not the 100-scene subsample the in-loop "
                        "eval uses; it is not a held-out number and must never "
                        "be reported as one.")
    p.add_argument("--pin-table", default=None,
                   help="default: the run's own path with the split substituted "
                        "(grasp_pin_table_train_omg.json -> ..._test_omg.json)")
    p.add_argument("--no-pin-table", action="store_true",
                   help="score EVERY scene of the split, including the ones OMG "
                        "cannot plan for. The pin table is two things at once — "
                        "the grasp each close is scored against, and the filter "
                        "deciding which scenes are usable — and `stable_grasp` "
                        "needs neither: it scores release-and-hold on the "
                        "rollout alone. Dropping it takes the s0 test split from "
                        "130 scenes to all 144. COST: the pose diagnostics "
                        "(near_rate, chance_rate, mean_pos_err, eval_min_pos) "
                        "become NaN for every scene, because there is no pinned "
                        "pose to measure against. Refused under "
                        "EVAL.success_mode: proximity, which scores that "
                        "distance and so cannot work without one.")
    p.add_argument("--allow-pin-mismatch", action="store_true",
                   help="proceed even when the split's pin table was built under "
                        "different settings than the run's — see PIN_PROVENANCE")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="cap the number of scenes (default: all of them)")
    p.add_argument("--ckpt", default=None, choices=["best", "last"],
                   help="default: the run's own EVAL.ckpt (`best` for every "
                        "Phase-4 run)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="re-score iterations already in the CSV")
    p.add_argument("--plot-only", action="store_true",
                   help="skip evaluation, re-render the figures from the CSV")
    p.add_argument("--summary-iter", type=int, default=None,
                   help="which iteration <split>_summary.png shows (default: the "
                        "highest success_rate in the CSV). The run's own `best/` "
                        "was selected on the TRAIN subsample and need not be best "
                        "here, which is why this re-selects rather than reading it.")
    return p.parse_args()


def read_done(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(r["iter"]): r for r in csv.DictReader(f) if r.get("iter")}


def write_log(path: Path, rows: dict) -> None:
    """Atomic via .tmp rename, so a plotter reading mid-write never sees half a
    file — the same discipline eval_dagger_run.py uses."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEST_FIELDS)
        w.writeheader()
        for i in sorted(rows):
            w.writerow({k: rows[i].get(k, "") for k in TEST_FIELDS})
    tmp.replace(path)


def _r(v, nd=4):
    """Round, or BLANK for a non-finite value. Never 0: a rate that was not
    measured and a rate that was measured as zero are different claims, and the
    plotters render a blank as a gap and a 0 as a data point."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return "" if not np.isfinite(f) else round(f, nd)


def outcome_columns(reasons: dict, n: int, n_fail: int) -> dict:
    """`reasons` -> the `f_*` and `ff_*` blocks.

    `_status_name` can OR two failures into "DROP|HUMAN_CONTACT"; each part is
    charged so no episode goes missing from the breakdown. That makes the
    columns sum to slightly more than 1.0 when an episode failed two ways, which
    is the honest rendering — dropping the second cause would understate it.
    """
    out = {v: 0 for v in EVAL_REASONS.values()}
    out.update({v: 0 for v in FAIL_REASONS.values()})
    for reason, count in (reasons or {}).items():
        for part in str(reason).split("|"):
            if part in EVAL_REASONS:
                out[EVAL_REASONS[part]] += count
            if part in FAIL_REASONS:
                out[FAIL_REASONS[part]] += count
    for k in EVAL_REASONS.values():
        out[k] = round(out[k] / n, 4) if n else ""
    for k in FAIL_REASONS.values():
        out[k] = round(out[k] / n_fail, 4) if n_fail else ""
    return out


def split_pin_path(run_path: str, split: str) -> str:
    """`..._train_omg.json` -> `..._test_omg.json`, and nothing cleverer.

    The split is a WHOLE path segment in every Phase-4 table name
    (`grasp_pin_table_<split>[_<rule>].json`), so substituting it is exact. It is
    done on the basename only: a run whose table lives under a directory with
    `train` in its name would otherwise get that rewritten too, which is how a
    path-munging default quietly points at a file that does not exist — or,
    worse, at one that does.
    """
    p = Path(run_path)
    for s in ("train", "val", "test"):
        if f"_{s}_" in p.name or p.name.endswith(f"_{s}.json"):
            return str(p.with_name(p.name.replace(f"_{s}", f"_{split}", 1)))
    return str(p)


def check_pin_provenance(pin_table, run_pin_path: str, path: str, *,
                         split: str, allow: bool) -> None:
    """Refuse a split table built under different settings than the RUN's table.

    THE PHASE-4 ANALOGUE OF `resolve_anchor_ref`, and the same class of bug: a
    table built differently does not crash, it silently scores a different task.

    THE REFERENCE IS THE RUN'S OWN TABLE, NOT THE RUN'S CONFIG, and that is the
    whole design of this function. A Phase-4 config does **not** record the pin
    rule — there is no `grasp_pin_mode` key anywhere in `dagger4_run19`'s
    config.yaml; the rule lives only in `_meta.mode` of the file the config
    points at, and by convention in that file's name. A version of this check
    that read the rule from the config with the split table's own value as the
    fallback would compare every field against itself and pass unconditionally.
    It did: `grasp_pin_table_val.json` carries `mode: furthest_from_hand` against
    run 19's `omg` and was waved through. Table against table is the only
    comparison with two independent sides.

    Two of the five fields matter most:

      `mode` is the pin RULE — `omg` is OMG's own pick, `furthest_from_hand` the
        deliberately harder target. Run 19 is an `omg` run; scoring it against a
        `furthest_from_hand` table measures agreement with a grasp the policy was
        never taught, and `near_rate` / `chance_rate` / `pos_err` all collapse
        for a reason that says nothing about the policy. This is not
        hypothetical — both tables exist in `output/`, four characters apart in
        the filename.

      `hand_collision_filter` decides whether grasps colliding with the giver
        were removed before pinning. Run 19 has it OFF and leans on the paper's
        offline `valid_grasp_dict_005.pkl`; a table built with it ON holds a
        strictly easier, differently-distributed target set.

    `setup`, `tol` and the collision threshold are checked for completeness. A
    guard covering only the fields that have already gone wrong catches the next
    one too late.
    """
    meta = getattr(pin_table, "meta", {}) or {}
    if not meta:
        print(f"[pin] {path} records no _meta — it predates the provenance block "
              f"and cannot be checked. Verify by hand before quoting anything "
              f"from this sweep.")
        return

    # The split table must actually BE for the split asked for. Cheap, and the
    # one error a filename substitution can make that the fields below cannot
    # see: a table correct in every build setting and for the wrong split.
    got_split = str(meta.get("split", ""))
    if got_split and got_split != str(split):
        raise SystemExit(
            f"[pin] {path} is the {got_split!r} split's table, but --split is "
            f"{split!r}. Scoring one split's scenes against another's pins is "
            f"not a caveat, it is a different experiment.")

    ref = {}
    if run_pin_path and Path(run_pin_path).exists():
        with open(run_pin_path) as f:
            ref = (json.load(f).get("_meta") or {})
    if not ref:
        print(f"[pin] the run's own table ({run_pin_path or 'unset'}) is not "
              f"readable from here, so {path} cannot be checked against it. "
              f"Verify `mode` by hand — it is the field that decides which grasp "
              f"every close is scored against, and it is recorded in neither "
              f"config.yaml nor anywhere else.")
        return

    bad = []
    for k in PIN_PROVENANCE:
        if k not in meta or k not in ref:
            continue
        a, b = meta[k], ref[k]
        if isinstance(b, bool) or isinstance(a, bool):
            same = bool(a) == bool(b)
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            same = abs(float(a) - float(b)) < 1e-9
        else:
            same = str(a) == str(b)
        if not same:
            bad.append(f"    {k:24s} {Path(path).name}={a!r}  "
                       f"{Path(run_pin_path).name}={b!r}")
    if not bad:
        print(f"[pin] {Path(path).name} agrees with the run's "
              f"{Path(run_pin_path).name} on "
              f"{', '.join(k for k in PIN_PROVENANCE if k in meta and k in ref)}")
        return
    msg = (f"[pin] {path} was built under different settings than the run's own "
           f"{run_pin_path}:\n" + "\n".join(bad)
           + "\nThe pin table decides which grasp each close is scored against "
             "and which scenes are usable at all, so this is a different task, "
             "not a different sample of the same one.\n"
             "Rebuild the split's table with the run's settings, or pass "
             "--allow-pin-mismatch and own the caveat.")
    if allow:
        print(msg.replace("[pin]", "[pin] WARNING —"))
    else:
        raise SystemExit(msg)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_dir)
    log_path = run_root / (args.out or f"{args.split}_log.csv")
    if args.plot_only:
        plot(run_root, log_path, args)
        return

    cfg_path = run_root / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config.yaml in {run_root} — is that a Phase-4 run dir?")
    with cfg_path.open() as f:
        cfg4 = yaml.safe_load(f)

    # ---- repoint the run's own config at the target split -------------------
    # Everything else — the cameras, the standoff, the success criterion, the box
    # geometry, the close thresholds — comes from the run's saved config, so the
    # test number differs from the training-curve number in the DATA and in
    # nothing else. Reconstructing an EvalParams by hand here is how the two
    # quietly stop being comparable.
    sim = cfg4["SIM"]
    run_pin = str(sim.get("grasp_pin_table", ""))
    sim["split"] = args.split
    # THE PIN TABLE IS ALSO THE SCENE FILTER, and that is why dropping it widens
    # the split. `build_phase4_context` sets `usable = set(pin_table.entries)`,
    # so the 14 s0-test scenes OMG cannot plan for are absent from the table and
    # therefore never evaluated — not because the POLICY needs a plan (it does
    # not; `stable_grasp` scores release-and-hold on the rollout alone) but
    # because the filter and the scoring reference are the same object.
    if args.no_pin_table:
        if str((cfg4.get("EVAL") or {}).get("success_mode", "stable_grasp")) \
                == "proximity":
            raise SystemExit(
                "--no-pin-table cannot work under EVAL.success_mode: proximity, "
                "which scores distance to the pinned grasp. Only stable_grasp "
                "scores a rollout without one.")
        sim["grasp_pin_table"] = None
    else:
        sim["grasp_pin_table"] = (args.pin_table
                                  or split_pin_path(run_pin, args.split))

    # `exclude_scenes` IS SPLIT-DEPENDENT, and getting it wrong is silent. It
    # lists the scenes of the TRAIN split whose base demonstration succeeded;
    # nothing was collected on test, so it has nothing to say there and applying
    # it would drop test scenes because a TRAIN scene with the same integer id
    # failed. On train it is the right filter for exactly the reason it is wrong
    # here, so it stays.
    excluded_applied = 1
    if args.split != "train":
        sim.pop("exclude_scenes", None)
        excluded_applied = 0

    if not args.no_pin_table and not Path(sim["grasp_pin_table"]).exists():
        raise SystemExit(
            f"{sim['grasp_pin_table']} not found. Build the {args.split}-split "
            f"pin table first, with the run's own pin settings:\n"
            f"  python examples/build_grasp_pin_table.py --split {args.split} "
            f"--mode {cfg4['SIM'].get('grasp_pin_mode', 'omg')} "
            f"--out {sim['grasp_pin_table']}\n"
            f"(or pass --pin-table if it lives somewhere else)")

    ev = cfg4.setdefault("EVAL", {})
    # CAPTURED BEFORE THE OVERWRITE. The banner reports what the IN-LOOP eval
    # sampled, which is the number this sweep exists to be contrasted with —
    # printing the post-overwrite value says "in-loop used 1000000 scenes",
    # which is both wrong and exactly backwards.
    inloop_n = ev.get("num_scenes")
    # Every scene of the split, not a subsample: it is already small and already
    # held out, so there is nothing to hold back from it. `holdout` is
    # meaningless here for the same reason — nothing collects on test.
    ev["num_scenes"] = int(args.num_scenes) if args.num_scenes else 10 ** 6
    ev["holdout"] = False
    seed = int(args.seed if args.seed is not None
               else cfg4.get("DAGGER", {}).get("seed", 0))

    ctx = build_phase4_context(cfg4, seed=seed)
    ckpt = args.ckpt or ctx.eval_ckpt
    if args.no_pin_table:
        print("[pin] DISABLED — every scene of the split is scored, and the pose "
              "diagnostics (near_rate, chance_rate, mean_pos_err, eval_min_pos) "
              "are NaN because there is no pinned pose to measure against.")
    else:
        # `run_pin` is the path as SAVED in the run's config, captured before it
        # was overwritten above — it is the reference the split's table is
        # checked against, and there is no other record of how the run pinned.
        check_pin_provenance(ctx.pin_table, run_pin, sim["grasp_pin_table"],
                             split=args.split, allow=args.allow_pin_mismatch)

    held_out = args.split != "train"
    print("=" * 78)
    print(f"Phase-4 {'HELD-OUT' if held_out else 'FULL-TRAIN'} eval   "
          f"run={run_root.name}")
    print(f"  split       : {args.split}   "
          + ("(held out by the benchmark, never collected on)" if held_out else
             "(COLLECTED ON — a train-set number, not a generalisation one)"))
    print(f"  scenes      : {len(ctx.eval_scenes)}"
          + (f"  (the in-loop eval used {inloop_n} of the train split)"
             if inloop_n else ""))
    print(f"  checkpoint  : {ckpt}     success={ctx.eval_params.success_mode}")
    print(f"  pin table   : {sim['grasp_pin_table'] or 'DISABLED (--no-pin-table)'}")
    print(f"  pinning     : {ctx.pin_table.describe() if ctx.pin_table else 'OFF'}")
    print(f"  exclude     : {'applied' if excluded_applied else 'dropped (train artifact)'}")
    print(f"  writing     : {log_path}")
    print("=" * 78)

    want = None if args.iters == "all" else {int(x) for x in args.iters.split(",")}
    done = {} if args.force else read_done(log_path)

    todo = [(i, d) for i, d in iteration_dirs(run_root)
            if (want is None or i in want) and i not in done]
    if not todo:
        print("nothing to score (use --force to re-score)")
    for i, run_dir in todo:
        t0 = time.time()
        print(f"\n[iter {i:02d}] {run_dir}")
        try:
            runner, _ = load_policy_runner(run_dir, args.device, ckpt=ckpt)
        except Exception as e:                              # noqa: BLE001
            print(f"  [skip] cannot load {ckpt}.pt: {type(e).__name__}: {e}")
            continue
        m = evaluate_policy(ctx.sim, runner, ctx.eval_scenes,
                            params=ctx.eval_params, pin_table=ctx.pin_table)
        rows = m.pop("rows", []) or []
        n_fail = sum(1 for r in rows if not r.get("success"))
        del runner
        if args.device != "cpu":
            import torch
            torch.cuda.empty_cache()
        eval_s = time.time() - t0

        row = {k: "" for k in TEST_FIELDS}
        row.update({"iter": i, "run_dir": str(run_dir), "ckpt": ckpt,
                    "split": args.split, "num_scenes": len(ctx.eval_scenes),
                    "n_episodes": int(m.get("n", 0)), "eval_s": round(eval_s, 1),
                    "pin_table": sim["grasp_pin_table"] or "(disabled)",
                    "excluded_applied": excluded_applied, "n_fail": n_fail})
        for k in TEST_FIELDS:
            if k in m:
                row[k] = _r(m[k])
        row.update(outcome_columns(m.get("reasons"), int(m.get("n", 0)), n_fail))
        done[i] = row
        write_log(log_path, done)
        _report_iteration(m, row, eval_s)

    if done:
        print(f"\nwrote {log_path}  ({len(done)} iterations)")
        best = max(done, key=lambda i: (float(done[i]["success_rate"])
                                        if done[i].get("success_rate") not in ("", None)
                                        else -1.0))
        print(f"best on this split: iteration {best}  "
              f"success_rate={done[best]['success_rate']}")
        plot(run_root, log_path, args)
    else:
        print("\nnothing scored — no completed iterations found in state.json")


def _report_iteration(m, row, eval_s) -> None:
    """The per-iteration console block, in the order the numbers should be read.

    handover -> commit -> approach -> failure shape. Printed as well as logged
    because a full sweep is long enough that reading the CSV afterwards is a
    second session, and a collapsed rate is worth noticing while there is still
    time to stop the job.
    """
    g = lambda k: m.get(k, float("nan"))                     # noqa: E731
    print(f"  HANDOVER success={g('success_rate'):.3f}   "
          f"grasp={g('grasp_rate'):.3f} close={g('close_rate'):.3f} "
          f"near={g('near_rate'):.3f}   ({eval_s:.0f}s)")
    print(f"  COMMIT   in-jaws chance={g('box_chance_rate'):.3f} "
          f"-> closed|in-jaws={g('box_taken_rate'):.3f} "
          f"-> secured|in-jaws={g('box_success_rate'):.3f}   "
          f"(missed|in-jaws={g('miss_given_box'):.3f})")
    print(f"  APPROACH min_pos={g('eval_min_pos'):.4f} m  "
          f"min_rot={g('eval_min_rot'):.3f} rad   "
          f"at close: pos={g('mean_pos_err'):.4f} rot={g('mean_rot_err'):.3f}")
    nf = int(row.get("n_fail") or 0)
    if nf:
        print(f"  FAILURES (of {nf} failed episodes)  "
              + "  ".join(f"{lbl} {row.get(k) or 0:.2f}"
                          for k, lbl in (("ff_timeout", "never closed"),
                                         ("ff_grasp_miss", "closed, not secured"),
                                         ("ff_drop", "drop"),
                                         ("ff_no_release", "no release"),
                                         ("ff_human_contact", "human contact"))))


# ── FIGURES ─────────────────────────────────────────────────────────────────
# The taxonomy, same names and same colours as plot_dagger_run.py's, so this
# figure and the training curve can be read against each other without a
# translation table.
_OUTCOMES = (("f_grasp_ok", "secured", "tab:green"),
             ("f_grasp_miss", "closed, not secured", "tab:olive"),
             ("f_no_release", "no release", "tab:orange"),
             ("f_drop", "drop", "tab:red"),
             ("f_human_contact", "human contact", "tab:purple"),
             ("f_timeout", "never closed", "tab:gray"))
_FAILURES = (("ff_grasp_miss", "closed, not secured", "tab:olive"),
             ("ff_no_release", "no release", "tab:orange"),
             ("ff_drop", "drop", "tab:red"),
             ("ff_human_contact", "human contact", "tab:purple"),
             ("ff_timeout", "never closed", "tab:gray"))


def plot(run_root: Path, log_path: Path, args) -> None:
    """`<split>_eval.png` — the rate family against DAgger iteration.

    The panels are the top row of `curves.png`, drawn from the same columns with
    the same labels, plus the outcome stack that lives on the training figure's
    second row. A test figure that renders its rates even slightly differently
    from the training one is a figure you cannot hold up next to it, which is the
    only thing it is for.

    Every axis title is prefixed with the split, because the whole hazard of this
    figure is someone reading a full-TRAIN number as a held-out one.
    """
    import matplotlib.pyplot as plt
    import plot_dagger_run as P

    if not log_path.exists():
        raise SystemExit(f"no {log_path} — run without --plot-only first")
    num, n = P._load(log_path)
    if n == 0:
        raise SystemExit(f"{log_path} has no rows yet")
    if n == 1:
        print(f"[plot] {log_path.name} has ONE iteration, so every curve panel "
              f"is a single point and the stacked-area ones render blank. "
              f"{args.split}_summary.png is the figure to read.")
    it = num("iter")
    TAG = f"{args.split.upper()}: "

    fig, ax = plt.subplots(1, 5, figsize=(27, 4.6), squeeze=False)
    ax = ax[0]

    # 1. the nested funnel
    a = ax[0]
    for key, label, style in (("close_rate", "close", ":"),
                              ("near_rate", "near (pose ok)", "-."),
                              ("grasp_rate", "grasp", "--"),
                              ("success_rate", "success", "-")):
        ys = num(key)
        if P._finite(ys):
            P._plot(a, it, ys, style, marker="o", ms=3, label=label,
                    lw=2 if key == "success_rate" else 1.2)
    a.set_ylim(-0.02, 1.02)
    # "stage rates", not "the nested rates" — see `plot_summary`. success and
    # grasped are independent predicates and success is the larger of the two in
    # most of run 19's iterations.
    P._grid(a, f"{TAG}stage rates (independent, not nested)",
            ylabel="fraction of eval scenes")
    P._legend(a, loc="upper left")

    # 2. chance vs conversion
    a = ax[1]
    for key, label, col, lw in (
            ("box_chance_rate", "object in jaws", "tab:blue", 1.4),
            ("box_taken_rate", "closed | in jaws", "tab:green", 2.0),
            ("box_success_rate", "secured | in jaws", "tab:cyan", 1.6),
            ("miss_given_box", "no grasp | in jaws", "tab:red", 1.4)):
        ys = num(key)
        if P._finite(ys):
            P._plot(a, it, ys, "-", marker="o", ms=3, color=col, lw=lw,
                    label=label)
    a.set_ylim(-0.02, 1.02)
    P._grid(a, f"{TAG}chance vs conversion", ylabel="fraction")
    P._legend(a, loc="upper left")

    # 3. approach error
    a = ax[2]
    for key, label, col in (("eval_min_pos", "min pos err (m)", "tab:blue"),
                            ("mean_pos_err", "pos err at close (m)", "tab:cyan")):
        ys = num(key)
        if P._finite(ys):
            P._plot(a, it, ys, "-", marker="o", ms=3, color=col, label=label)
    a.axhline(0.02, color="tab:blue", ls=":", lw=1, label="close thresh 0.02 m")
    a.set_ylim(bottom=0)
    a.set_ylabel("position error (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a2 = a.twinx()
    for key, label, col in (("eval_min_rot", "min rot err (rad)", "tab:red"),
                            ("mean_rot_err", "rot err at close (rad)", "tab:orange")):
        ys = num(key)
        if P._finite(ys):
            P._plot(a2, it, ys, "--", marker="s", ms=3, color=col, label=label)
    a2.axhline(0.34, color="tab:red", ls=":", lw=1)
    a2.set_ylim(bottom=0)
    a2.set_ylabel("rotation error (rad)", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    P._grid(a, f"{TAG}approach error to the grasp")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    if h1 or h2:
        a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")

    # 4-5. the outcome taxonomy, both readings
    for a, series, title in (
            (ax[3], _OUTCOMES, "eval outcomes (of every episode)"),
            (ax[4], _FAILURES, "failure profile (of the episodes that failed)")):
        if P._stack(a, it, [num(k) for k, _, _ in series],
                    [lb for _, lb, _ in series], [c for _, _, c in series]):
            a.set_ylim(0, 1)
        else:
            a.text(0.5, 0.5, "needs at least two scored iterations\n"
                             "(a stacked area cannot render one point)",
                   ha="center", va="center", transform=a.transAxes,
                   fontsize=9, color="0.45")
        P._grid(a, f"{TAG}{title}", ylabel="fraction")
        P._legend(a, loc="lower left", ncol=2)

    P._fix_x(fig, it)
    held = "HELD-OUT " if args.split != "train" else "FULL "
    ns = num("num_scenes")
    fig.suptitle(f"Phase-4 DAgger on the {held}{args.split} split — "
                 f"{run_root.name}"
                 f"   [{int(ns[-1]) if P._finite(ns) else '?'} scenes]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = run_root / f"{args.split}_eval.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    plot_summary(run_root, num, n, args)


def plot_summary(run_root: Path, num, n, args) -> None:
    """`<split>_summary.png` — one iteration, as bars.

    WHY A SECOND FIGURE. Everything above is a curve against DAgger iteration,
    which is the right shape for a training run and the wrong one for a test
    report: a test sweep is often a single checkpoint, where a line plot of one
    point conveys nothing and the stacked-area panels render as invisible
    zero-width polygons under a full legend. This is the answer to "what does the
    finished policy do", and it is the figure to put in a document.

    THE ITERATION SHOWN is the one with the highest `success_rate` in the CSV,
    with its number in the title — not the last row, because a sweep scored at
    `--iters 0,5,10` has no meaningful "last", and not the run's own `best/`,
    which was selected on the TRAIN subsample and need not be best here.
    `--summary-iter` overrides it.
    """
    import matplotlib.pyplot as plt

    it = num("iter")
    succ = num("success_rate")
    want_it = getattr(args, "summary_iter", None)
    if want_it is not None:
        cand = [i for i, v in enumerate(it) if v == want_it]
        if not cand:
            raise SystemExit(f"--summary-iter {want_it} is not in the CSV "
                             f"(have {[int(v) for v in it if v == v]})")
        idx = cand[0]
    else:
        fin = [(v, i) for i, v in enumerate(succ) if v == v]
        idx = max(fin)[1] if fin else n - 1
    ITER = int(it[idx]) if it[idx] == it[idx] else -1

    def at(key, default=float("nan")):
        ys = num(key)
        v = ys[idx] if idx < len(ys) else default
        return v if v == v else default

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8), squeeze=False)
    ax = ax[0]

    def _bars(a, pairs, colors, title, ylabel="fraction", ylim=(0, 1.05),
              fmt="{:.3f}"):
        labs = [lb for lb, _ in pairs]
        vals = [v for _, v in pairs]
        x = np.arange(len(labs), dtype=float)
        a.bar(x, vals, 0.6, color=colors)
        for xi, yi in zip(x, vals):
            if yi == yi:
                a.text(xi, yi + (ylim[1] - ylim[0]) * 0.015, fmt.format(yi),
                       ha="center", va="bottom", fontsize=8)
        a.set_xticks(x)
        a.set_xticklabels(labs, fontsize=8)
        if ylim:
            a.set_ylim(*ylim)
        a.set_title(title, fontsize=10)
        a.set_ylabel(ylabel, fontsize=8)
        a.grid(alpha=0.3, axis="y")

    # 1. the four stage rates — NOT a funnel, and the label says so.
    #
    # `plot_dagger_run.py` titles these "the nested rates" and the evaluator's
    # docstring describes them as a chain (`close` ...and did it within
    # tolerance ...and both fingers ended on the object ...and it was secured).
    # THEY ARE NOT NESTED. Reading the code: `success = held`, from
    # `grasp_held_after_hold`, and `grasped = env.grasped_active()` — two
    # independent predicates evaluated at the same instant, so an object can
    # survive the hold while the contact query reads False. Measured on run 19,
    # `success_rate > grasp_rate` in 17 of 26 iterations. `near` is a third
    # independent test (pose error at the close against the CLOSE-label
    # tolerances) and sits at 0.01-0.04 while grasp sits at 0.5-0.7, so it is not
    # a stage between them either.
    #
    # Drawing them under a funnel label would assert a monotonicity the data
    # contradicts in most iterations, and would make a reader treat
    # success > grasp as an error in the measurement rather than as what these
    # two columns mean. Same four numbers, honest title.
    _bars(ax[0],
          [("closed\nat all", at("close_rate")), ("near\n(pose ok)", at("near_rate")),
           ("grasped\n(contact)", at("grasp_rate")), ("SUCCESS\n(held)", at("success_rate"))],
          ["tab:gray", "tab:orange", "tab:blue", "tab:green"],
          "STAGE RATES — four INDEPENDENT tests, not a funnel\n"
          "(success = survived the hold; grasped = fingers in contact —\n"
          "neither implies the other, so success > grasped is normal)")

    # 2. the commit chain. box_chance -> box_taken -> box_success, each
    # conditional on the one before, so a gap localises a different failure.
    _bars(ax[1],
          [("in jaws\n(chance)", at("box_chance_rate")),
           ("closed |\nin jaws", at("box_taken_rate")),
           ("secured |\nin jaws", at("box_success_rate")),
           ("missed |\nin jaws", at("miss_given_box"))],
          ["tab:blue", "tab:green", "tab:cyan", "tab:red"],
          "GRASP COMMIT: did it get the chance, take it, keep it\n"
          "(the last three are CONDITIONAL on the chance)")

    # 3. approach error. Two units on one panel, so the pair is one saccade
    # apart; the threshold lines are what make either number readable.
    a = ax[2]
    pos = [("min pos\n(m)", at("eval_min_pos")), ("pos @ close\n(m)", at("mean_pos_err"))]
    rot = [("min rot\n(rad)", at("eval_min_rot")), ("rot @ close\n(rad)", at("mean_rot_err"))]
    x = np.arange(4, dtype=float)
    vals = [v for _, v in pos] + [v for _, v in rot]
    a.bar(x[:2], vals[:2], 0.6, color="tab:blue")
    a.axhline(0.02, color="tab:blue", ls=":", lw=1.2, label="close thresh 0.02 m")
    a.set_ylabel("position error (m)", fontsize=8, color="tab:blue")
    a.tick_params(axis="y", labelcolor="tab:blue")
    a.set_ylim(0, max([v for v in vals[:2] if v == v] + [0.03]) * 1.35)
    a2 = a.twinx()
    a2.bar(x[2:], vals[2:], 0.6, color="tab:red", alpha=0.75)
    a2.axhline(0.34, color="tab:red", ls="--", lw=1.2, label="close thresh 0.34 rad")
    a2.set_ylabel("rotation error (rad)", fontsize=8, color="tab:red")
    a2.tick_params(axis="y", labelcolor="tab:red")
    a2.set_ylim(0, max([v for v in vals[2:] if v == v] + [0.4]) * 1.35)
    for xi, yi, axis in list(zip(x[:2], vals[:2], [a] * 2)) + \
            list(zip(x[2:], vals[2:], [a2] * 2)):
        if yi == yi:
            axis.text(xi, yi, f"  {yi:.4f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x)
    a.set_xticklabels([lb for lb, _ in pos + rot], fontsize=8)
    a.set_title("APPROACH ERROR to the pinned grasp\n"
                "(min = closest over the episode; @ close = at the commit)",
                fontsize=10)
    a.grid(alpha=0.3, axis="y")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    # 4. the outcome taxonomy, both readings side by side as two stacked bars.
    # `f_*` over every episode and `ff_*` over the failures: the first says how
    # much of the eval set each outcome is, the second says what failure LOOKS
    # like, and only the second survives a change in the success rate.
    a = ax[3]
    for xi, series, lab in ((0.0, _OUTCOMES, "all episodes"),
                            (1.0, _FAILURES, "failures only")):
        bottom = 0.0
        for key, name, col in series:
            v = at(key)
            if v != v:
                continue
            a.bar(xi, v, 0.5, bottom=bottom, color=col,
                  label=name if xi == 0.0 else None)
            if v > 0.04:
                a.text(xi, bottom + v / 2, f"{v:.2f}", ha="center", va="center",
                       fontsize=7, color="white")
            bottom += v
    a.set_xticks([0.0, 1.0])
    a.set_xticklabels(["of ALL episodes", "of the FAILURES"], fontsize=8)
    a.set_ylim(0, 1.05)
    a.set_title("OUTCOME TAXONOMY, both denominators\n"
                "(left stacks to 1.0 over eval; right over the failures)",
                fontsize=10)
    a.set_ylabel("fraction", fontsize=8)
    a.grid(alpha=0.3, axis="y")
    a.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))

    held = "HELD-OUT " if args.split != "train" else "FULL "
    fig.suptitle(f"Phase-4 DAgger on the {held}{args.split} split — "
                 f"{run_root.name} @ iteration {ITER}   "
                 f"[{int(at('num_scenes', 0))} scenes, "
                 f"{int(at('n_episodes', 0))} episodes, "
                 f"ckpt {_csv_cell(run_root, args, ITER, 'ckpt') or '?'}]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = run_root / f"{args.split}_summary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def _csv_cell(run_root: Path, args, iteration: int, col: str) -> str:
    """One STRING cell out of the log. `plot_dagger_run._load` coerces every
    column to float, which turns `best` into NaN — fine for every other column
    and useless for the one naming the checkpoint."""
    path = run_root / (args.out or f"{args.split}_log.csv")
    if not path.exists():
        return ""
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("iter") and int(r["iter"]) == int(iteration):
                return str(r.get(col, "") or "")
    return ""


if __name__ == "__main__":
    main()
