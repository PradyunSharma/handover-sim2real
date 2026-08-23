"""
Score a finished Regrasp run on the HELD-OUT s0 TEST split.

    python examples/eval_regrasp_testset.py --run-dir output/dagger_runs/regrasp_run2
    python examples/eval_regrasp_testset.py --run-dir output/dagger_runs/regrasp_run2 \\
        --iters 0,5,10,15,20 --chained
    python examples/eval_regrasp_testset.py --run-dir output/dagger_runs/regrasp_run2 \\
        --plot-only                      # re-render from an existing CSV

WHY THIS IS A SEPARATE SCRIPT AND NOT A FLAG ON THE TRAINER

The in-loop evaluation runs with `EVAL.holdout: false` on the TRAIN split, which
means its scenes are also collected on. That is deliberate — it makes the curves
a clean read of "is the policy getting better at the thing it is being taught",
uncontaminated by the extra variance of a small held-out sample — but it is a
train-set number and must never be reported as anything else.

This script answers the other question, on scenes the run has never seen in any
form: not held out from a pool, held out by the benchmark's own s0 split. It runs
after training because it needs the whole iteration sequence and because putting
it inline would double an already-long job for a number nobody acts on mid-run.

TWO EVALUATIONS, AND THEY ANSWER DIFFERENT QUESTIONS

  INDEPENDENT (always)  Every (scene, bin) pair is a fresh rollout from home. This
      is what the per-bin panels are built from, and it is the honest measure of
      "given this command, can the policy execute it". `retry_at_k` derived from
      it is an OR over independent attempts and therefore a CEILING: it assumes
      a failed attempt costs nothing and leaves the world untouched.

  CHAINED (--chained)   The retry ladder, run for real: attempt 1 fails, the arm
      rewinds to 30% of the trajectory it just flew, and attempt 2 is commanded
      from there. `chained_retry_at_k` is the number a deployment would see, and
      the GAP against the independent `retry_at_k` is what the reset-based
      version was giving away. Costs roughly 1.6x the independent sweep (most
      scenes stop after one or two attempts), so it is opt-in.

      Still an upper bound, and the reason is written down rather than hidden:
      the rewind resets the simulator, which rewinds the human's DexYCB playback
      with it. A real retreat controller drives the arm back along its own joint
      path without touching the human. That is the deferred experiment.

OUTPUT

  <run>/test_eval_log.csv        one row per iteration, the same ~340-column
                                 schema as dagger_log.csv, so the two are
                                 directly diffable column for column
  <run>/test_set_evaluation.png  the training_curve.png layout — one ROW per
                                 commanded direction, columns success stages /
                                 chance vs conversion / approach error — plus a
                                 conditioning row: ended-in-the-commanded-bin per
                                 bin, arrived-from-the-commanded-side per bin,
                                 and retry@k with the chained curve overlaid when
                                 --chained was used.

CHECKPOINT. `last`, not `best`, matching the run's own `EVAL.ckpt`. Every Phase-4
run scored `best.pt`; from run 2 on the whole pipeline — collection, in-loop eval,
warm start and this script — reads the same `last.pt`, so "the policy at iteration
i" names one set of weights everywhere instead of three.

PREREQUISITE: a TEST-split direction table and pin table, which are a separate
sim pass:

    python examples/build_direction_table.py --split test \\
        --out output/direction_table_test.json
    python examples/assign_direction_demos.py --table output/direction_table_test.json \\
        --out output/regrasp_pins_test --drop-bins='-z_beneath,-x_over_fingers'

There is deliberately NO `demo_ok_table` on test. That file records which pairs
base COLLECTION managed to demonstrate, and nothing is collected on test — the
pin table itself is the feasibility statement, because it was built by calling
the planner on every scene. Filtering test by the train split's collection
failures would be scoring on a set defined by an unrelated run.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from handover_sim2real.regrasp import (                        # noqa: E402
    evaluate_policy, load_policy_runner,
)
from handover_sim2real.regrasp import directions as _D         # noqa: E402
from handover_sim2real.regrasp.setup import build_regrasp_context  # noqa: E402

from eval_regrasp_run import iteration_dirs                    # noqa: E402
from train_regrasp import LOG_FIELDS, eval_columns             # noqa: E402


# The test CSV is the dagger_log schema plus its own provenance, so a column
# means the same thing in both files and `diff <(cut -f...) ...` is meaningful.
TEST_FIELDS = (["iter", "run_dir", "ckpt", "split", "num_scenes", "num_episodes",
                "eval_s"]
               + [c for c in LOG_FIELDS
                  if c not in ("iter", "beta", "m", "is_best", "collect_s",
                               "train_s", "eval_s", "wall_s")
                  and not c.startswith("c_")]
               + ["chained_retry_at_1", "chained_retry_at_2",
                  "chained_retry_at_3", "chained_retry_at_4",
                  "solved_rate", "mean_attempts", "mean_attempts_to_success",
                  "signal_human_rate", "mean_branch_step", "replay_err_mean"]
               + [f"chain_succ_bin_{b}" for b in range(len(_D.BINS))]
               + [f"chain_n_bin_{b}" for b in range(len(_D.BINS))])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="output/dagger_runs/<name>")
    p.add_argument("--out", default="test_eval_log.csv")
    p.add_argument("--iters", default="all",
                   help="'all' or a comma list, e.g. 0,5,10,15,20")
    p.add_argument("--split", default="test",
                   help="handover-sim split to score on (test | val)")
    p.add_argument("--pin-table", default=None,
                   help="default output/regrasp_pins_<split>.json")
    p.add_argument("--exclude-scenes", default=None,
                   help="default output/regrasp_pins_<split>_excluded.json")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="cap the number of test scenes (default: all of them)")
    p.add_argument("--ckpt", default="last", choices=["best", "last"],
                   help="which checkpoint of each iteration to score (default last, "
                        "matching everything else in the run)")
    p.add_argument("--chained", action="store_true",
                   help="also run the retry LADDER, where attempt 2 starts from "
                        "where attempt 1 stopped rather than from home")
    p.add_argument("--rewind-frac", type=float, default=0.30,
                   help="--chained: how far back along the failed trajectory to "
                        "resume from, as a fraction of its length")
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true", help="re-score iterations already in the CSV")
    p.add_argument("--plot-only", action="store_true",
                   help="skip evaluation, re-render the figure from the CSV")
    p.add_argument("--pos-thresh", type=float, default=0.02)
    p.add_argument("--rot-thresh", type=float, default=0.34)
    return p.parse_args()


def read_done(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(r["iter"]): r for r in csv.DictReader(f) if r.get("iter")}


def write_log(path: Path, rows: dict) -> None:
    """Atomic via .tmp rename, so a plotter reading mid-write never sees half a
    file — the same discipline eval_regrasp_run.py uses."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEST_FIELDS)
        w.writeheader()
        for i in sorted(rows):
            w.writerow({k: rows[i].get(k, "") for k in TEST_FIELDS})
    tmp.replace(path)


def run_chained(ctx, runner, scenes, *, rewind_frac: float, max_attempts: int):
    """The retry ladder over the test scenes. Returns chained_metrics()'s dict."""
    from handover_sim2real.regrasp.chained_retry import (
        RetryParams, chained_metrics, chained_retry_scene,
    )

    per_scene = {}
    for n, scene in enumerate(scenes):
        n_here = ctx.pin_table.num_grasps_for(int(scene))
        # bin -> the pose realising it. The chain still needs a pose per attempt
        # (OMG's CLOSE label and the geometric scores are measured against one);
        # it is simply no longer what the policy is told.
        pose_of_bin = {}
        for gi in range(int(n_here)):
            b = ctx.pin_table.bin_of(int(scene), gi)
            if b is not None:
                pose_of_bin[int(b)] = ctx.pin_table.pose(int(scene), gi)
        if not pose_of_bin:
            continue
        meta = ctx.pin_table.scene_meta.get(int(scene), {})
        anchor_R = meta.get("anchor_R")
        atts = chained_retry_scene(
            ctx.sim, runner, int(scene), pose_of_bin,
            params=ctx.eval_params,
            retry=RetryParams(max_attempts=int(max_attempts),
                              rewind_frac=float(rewind_frac)),
            anchor_R=np.asarray(anchor_R) if anchor_R is not None else None,
            feasible=sorted(pose_of_bin))
        per_scene[int(scene)] = atts
        if (n + 1) % 10 == 0:
            print(f"    [chain] {n + 1}/{len(scenes)} scenes", flush=True)
    return chained_metrics(per_scene, max_attempts=int(max_attempts))


def plot(run_root: Path, log_path: Path, args) -> None:
    """The training_curve.png layout, on test, plus a conditioning row.

    The PANELS are imported from plot_regrasp_run rather than reimplemented. A
    test figure that draws its rates slightly differently from the training one
    is a figure you cannot hold up next to it, which is the only thing it is for.
    """
    import matplotlib.pyplot as plt
    import plot_regrasp_run as P

    if not log_path.exists():
        raise SystemExit(f"no {log_path} — run without --plot-only first")
    num, n = P._load(log_path)
    if n == 0:
        raise SystemExit(f"{log_path} has no rows yet")
    it = num("iter")
    ctx = P._Ctx(num, it, args, "grasp")
    bins = P._bins_to_plot(num)
    cols = [P._BIN_COLOURS_BY_BIN[b] for b in bins]

    nrow = max(len(bins), 1) + 1                 # + the conditioning row
    fig, ax = plt.subplots(nrow, 3, figsize=(17, 3.7 * nrow), squeeze=False)
    for r, b in enumerate(bins):
        sfx, name = f"_b{b}", P._bin_title(b)
        P._panel_nested(ax[r][0], ctx, sfx, title=f"{name} — success stages")
        P._panel_opportunity(ax[r][1], ctx, sfx, title=f"{name} — chance vs conversion")
        P._panel_approach(ax[r][2], ctx, sfx, title=f"{name} — approach error")

    # ---- the conditioning row ----------------------------------------------
    # Told to come in from bin b, how often did the gripper END there. This is
    # the diagonal of the confusion matrix per commanded bin, against a chance
    # level of 1/4 — the one number that says whether the command reached the
    # ACTION rather than merely reaching the network.
    a = ax[nrow - 1][0]
    for b, col in zip(bins, cols):
        ys = ctx.get(f"bin_diag_rate_b{b}")
        if P._finite(ys):
            P._plot(a, it, ys, "-", marker="o", ms=3, lw=1.8, color=col,
                    label=_D.BIN_SHORT[b])
    a.axhline(0.25, color="0.6", ls=":", lw=1.0)
    a.text(0.01, 0.235, "chance (4 live bins)", fontsize=7, color="0.4",
           va="top", transform=a.get_yaxis_transform())
    a.set_ylim(-0.02, 1.02)
    P._note_empty(a)
    P._grid(a, "TEST: ended in the COMMANDED bin (per bin)",
            ylabel="fraction of that bin's episodes")
    P._legend(a, loc="upper left", ncol=2)

    # ...and WHICH SIDE it came from, which fails independently of which way it
    # pointed: a gripper can be correctly oriented on the wrong side.
    a = ax[nrow - 1][1]
    for b, col in zip(bins, cols):
        ys = ctx.get(f"bin_hit_rate_b{b}")
        if P._finite(ys):
            P._plot(a, it, ys, "-", marker="o", ms=3, lw=1.8, color=col,
                    label=_D.BIN_SHORT[b])
    a.axhline(0.25, color="0.6", ls=":", lw=1.0)
    a.set_ylim(-0.02, 1.02)
    P._note_empty(a)
    P._grid(a, "TEST: arrived from the COMMANDED side (per bin)",
            ylabel="fraction of that bin's episodes")
    P._legend(a, loc="upper left", ncol=2)

    # retry@k, independent (solid) against chained (dashed). The gap IS the cost
    # of a failed attempt: the independent curve assumes attempt 2 starts from
    # home with the world untouched, the chained one starts it from where
    # attempt 1 actually left the arm.
    a = ax[nrow - 1][2]
    for k, col in zip(range(1, 5), ("tab:blue", "tab:green", "tab:orange", "tab:red")):
        ys = num(f"retry_at_{k}")
        if P._finite(ys):
            P._plot(a, it, ys, "-", marker="o", ms=3, lw=1.4 + 0.2 * k, color=col,
                    label=f"independent @ {k}")
        yc = num(f"chained_retry_at_{k}")
        if P._finite(yc):
            P._plot(a, it, yc, "--", marker="s", ms=3, lw=1.4, color=col,
                    label=f"chained @ {k}")
    a.set_ylim(-0.02, 1.02)
    P._note_empty(a)
    P._grid(a, "TEST: regrasping — success with k tries",
            ylabel="fraction of test scenes")
    P._legend(a, loc="lower right", ncol=2)

    P._fix_x(fig, it)
    fig.suptitle(f"Regrasp on the HELD-OUT {args.split} split — {run_root.name}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.03 / nrow * 2])
    out = run_root / "test_set_evaluation.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_dir)
    log_path = run_root / args.out
    if args.plot_only:
        plot(run_root, log_path, args)
        return

    cfg_path = run_root / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config.yaml in {run_root} — is that a Regrasp run dir?")
    with cfg_path.open() as f:
        cfg4 = yaml.safe_load(f)

    # ---- repoint the run's own config at the test split ---------------------
    # Everything else — the cameras, the standoff, the success criterion, the
    # box geometry, the close thresholds — comes from the run's saved config, so
    # the test number differs from the training-curve number in the DATA and in
    # nothing else. Reconstructing an EvalParams by hand here is how the two
    # quietly stop being comparable.
    sim = cfg4["SIM"]
    sim["split"] = args.split
    sim["grasp_pin_table"] = (args.pin_table
                              or f"output/regrasp_pins_{args.split}.json")
    sim["exclude_scenes"] = (args.exclude_scenes
                             or f"output/regrasp_pins_{args.split}_excluded.json")
    # No collection happens on test, so there is no collection outcome to filter
    # by; the pin table already encodes what the planner could reach.
    sim.pop("demo_ok_table", None)
    for path_key in ("grasp_pin_table", "exclude_scenes"):
        if not Path(sim[path_key]).exists():
            raise SystemExit(
                f"{sim[path_key]} not found. Build the {args.split}-split tables "
                f"first:\n"
                f"  python examples/build_direction_table.py --split {args.split} "
                f"--out output/direction_table_{args.split}.json\n"
                f"  python examples/assign_direction_demos.py "
                f"--table output/direction_table_{args.split}.json "
                f"--out output/regrasp_pins_{args.split} "
                f"--drop-bins='-z_beneath,-x_over_fingers'")

    ev = cfg4.setdefault("EVAL", {})
    # Every test scene, not an np.linspace subsample: the split is already small
    # and already held out, so there is nothing to hold back from it. `holdout`
    # is meaningless here for the same reason — nothing collects on test.
    ev["num_scenes"] = int(args.num_scenes) if args.num_scenes else 10 ** 6
    ev["holdout"] = False
    seed = int(args.seed if args.seed is not None
               else cfg4.get("DAGGER", {}).get("seed", 0))

    ctx = build_regrasp_context(cfg4, seed=seed)
    n_ep = len(ctx.pin_table.pairs(ctx.eval_scenes))

    print("=" * 78)
    print(f"Regrasp TEST-SPLIT eval   run={run_root.name}")
    print(f"  split       : {args.split}   (held out by the benchmark, never "
          f"collected on)")
    print(f"  scenes      : {len(ctx.eval_scenes)} -> {n_ep} independent episodes "
          f"per iteration")
    print(f"  checkpoint  : {args.ckpt}     success={ctx.eval_params.success_mode}")
    print(f"  chained     : {'ON' if args.chained else 'off'}"
          + (f"   rewind {args.rewind_frac:.0%}, "
             f"max {args.max_attempts} attempts" if args.chained else ""))
    print(f"  pinning     : {ctx.pin_table.describe()}")
    print(f"  writing     : {log_path}")
    print("=" * 78)

    want = None if args.iters == "all" else {int(x) for x in args.iters.split(",")}
    rows = {} if args.force else read_done(log_path)

    todo = [(i, d) for i, d in iteration_dirs(run_root)
            if (want is None or i in want) and i not in rows]
    if not todo:
        print("nothing to score (use --force to re-score)")
    for i, run_dir in todo:
        t0 = time.time()
        print(f"\n[iter {i:02d}] {run_dir}")
        try:
            runner, _ = load_policy_runner(run_dir, args.device, ckpt=args.ckpt)
        except Exception as e:                              # noqa: BLE001
            print(f"  [skip] cannot load {args.ckpt}.pt: {type(e).__name__}: {e}")
            continue
        m = evaluate_policy(ctx.sim, runner, ctx.eval_scenes,
                            params=ctx.eval_params, pin_table=ctx.pin_table)
        m.pop("rows", None)
        chained = {}
        if args.chained:
            chained = run_chained(ctx, runner, ctx.eval_scenes,
                                  rewind_frac=args.rewind_frac,
                                  max_attempts=args.max_attempts)
        del runner
        if args.device != "cpu":
            import torch
            torch.cuda.empty_cache()
        eval_s = time.time() - t0

        row = {k: "" for k in TEST_FIELDS}
        row.update({"iter": i, "run_dir": str(run_dir), "ckpt": args.ckpt,
                    "split": args.split, "num_scenes": len(ctx.eval_scenes),
                    "num_episodes": int(m.get("n", 0)),
                    "eval_s": round(eval_s, 1)})
        row.update({k: v for k, v in eval_columns(m).items() if k in row})
        for k, v in chained.items():
            if k in row and isinstance(v, (int, float, np.floating, np.integer)):
                row[k] = "" if not np.isfinite(float(v)) else round(float(v), 4)
        rows[i] = row
        write_log(log_path, rows)

        print(f"  success={m['success_rate']:.3f} grasp={m['grasp_rate']:.3f} "
              f"close={m['close_rate']:.3f}  ({eval_s:.0f}s)")
        print("  per-bin success / ended-in-commanded-bin:  "
              + "  ".join(
                  f"{_D.BIN_SHORT[b]} {m.get(f'success_rate_b{b}', float('nan')):.2f}"
                  f"/{m.get(f'bin_diag_rate_b{b}', float('nan')):.2f}"
                  for b in _D.LIVE_BINS))
        if chained:
            print("  chained retry@k: "
                  + "/".join(f"{chained.get(f'chained_retry_at_{k}', float('nan')):.2f}"
                             for k in range(1, args.max_attempts + 1)))

    if rows:
        print(f"\nwrote {log_path}  ({len(rows)} iterations)")
        plot(run_root, log_path, args)
    else:
        print("\nnothing scored — no completed iterations found in state.json")


if __name__ == "__main__":
    main()
