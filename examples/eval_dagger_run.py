"""Score a Phase-4 DAgger run's iterations OUTSIDE the training loop.

    # after (or during) a run
    python examples/eval_dagger_run.py --run-dir output/dagger_runs/dagger4_run6

    # keep pace with a job that is still training
    python examples/eval_dagger_run.py --run-dir output/dagger_runs/dagger4_run6 --watch

    # re-score everything with a bigger scene set
    python examples/eval_dagger_run.py --run-dir ... --num-scenes 200 --force \\
        --out eval_log_200.csv

WHY THIS EXISTS. Evaluation is 19% of a DAgger run's wall clock and is NOT on the
critical path. train_dagger_phase4.py sets `cur_run_dir = iter_dir` BEFORE the
eval block runs, so the next iteration collects with the freshly-refit policy
whether or not it has been scored; the eval result feeds only
`maybe_update_best`, i.e. final checkpoint selection and reporting. Running it
inline therefore just makes the loop wait. Set `EVAL.every: 0` in the run config
(or pass --no-eval) and run this instead — concurrently in a second job, or once
at the end.

COMPARABILITY. The numbers only mean the same thing as the in-loop ones if the
simulator, pin table, scene pool and thresholds are identical. That is why this
reads the run's OWN saved <run>/config.yaml and builds everything through
handover_sim2real.dagger.setup.build_phase4_context — the same call the training
loop makes. Overriding --num-scenes changes the pool on purpose; the column is
recorded in the CSV so a re-scored run cannot be mistaken for the original.

CONCURRENCY. Read-only on everything the trainer writes; the only file it creates
is <run>/eval_log.csv. Iterations already present are skipped unless --force, so
it is restartable and safe to run repeatedly while training continues.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root -> handover_sim2real
sys.path.insert(0, _HERE)                    # examples/  -> train_dagger_phase4

from handover_sim2real.dagger import (          # noqa: E402
    evaluate_policy,
    export_run_dir,
    load_policy_runner,
)
from handover_sim2real.dagger.setup import build_phase4_context   # noqa: E402


# Columns mirror the eval half of dagger_log.csv so the two can be compared or
# concatenated without a translation table.
EVAL_FIELDS = [
    "iter", "run_dir", "ckpt", "num_scenes",
    "success_rate", "grasp_rate", "near_rate", "close_rate", "close_success_rate",
    "chance_rate", "missed_rate", "miss_given_chance",
    "box_chance_rate", "box_taken_rate", "box_missed_rate", "miss_given_box",
    "mean_box_steps", "mean_box_frac",
    "eval_min_pos", "eval_min_rot", "mean_dist", "mean_pos_err", "mean_rot_err",
    "mean_close_step",
    "f_grasp_ok", "f_grasp_miss", "f_no_release", "f_drop", "f_timeout",
    "f_human_contact",
    "eval_s",
]

# Imported, not restated: `_status_name` can OR two failures into
# "DROP|HUMAN_CONTACT" and reason_columns charges both, which a naive dict lookup
# would drop. Sharing the mapping also means a new reason string cannot end up
# filed differently here than in dagger_log.csv.
from train_dagger_phase4 import EVAL_REASONS, reason_columns   # noqa: E402


def iteration_dirs(run_root: Path) -> list[tuple[int, Path]]:
    """(iteration, run_dir) for every COMPLETE iteration, base run included.

    Keyed on state.json rather than on the directory listing: train_dagger_phase4
    writes that record only after collection + refit + eval finish, so anything
    beyond `max(recorded)` belongs to an iteration still in flight and would be
    scored from a half-written checkpoint.
    """
    st_path = run_root / "state.json"
    if not st_path.exists():
        return []
    with st_path.open() as f:
        state = json.load(f)
    out = []
    for rec in state.get("iterations", []):
        i = int(rec["iter"])
        d = Path(rec["run_dir"])
        # state.json stores ABSOLUTE paths from wherever the run executed, so a
        # run dir synced off the cluster has entries that do not resolve here.
        # Fall back to the canonical layout under this run root.
        if not (d / "checkpoints").is_dir():
            d = run_root / "iters" / f"iter_{i:02d}"
        if (d / "checkpoints").is_dir():
            out.append((i, d))
    return sorted(out)


def row_from_metrics(i: int, run_dir: Path, ckpt: str, n_scenes: int,
                     m: dict, eval_s: float) -> dict:
    row = {k: "" for k in EVAL_FIELDS}
    row.update({"iter": i, "run_dir": str(run_dir), "ckpt": ckpt,
                "num_scenes": n_scenes, "eval_s": round(eval_s, 1)})
    for k in EVAL_FIELDS:
        if k in m:
            row[k] = m[k]
    row.update(reason_columns(m.get("reasons"), EVAL_REASONS,
                              denom=m.get("n") or None))
    return row


def read_done(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(r["iter"]): r for r in csv.DictReader(f) if r.get("iter")}


def write_log(path: Path, rows: dict[int, dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        w.writeheader()
        for i in sorted(rows):
            w.writerow({k: rows[i].get(k, "") for k in EVAL_FIELDS})
    tmp.replace(path)          # atomic: a concurrent plotter never sees a partial file


def score_tuple(m: dict, select_on: str) -> list[float]:
    """Same lexicographic order train_dagger_phase4.maybe_update_best uses.

    Not imported from there because this one is fed rows read back out of the
    CSV, where every value is a string and a missing metric is "" rather than
    absent — hence the coercion. The ordering itself must stay identical.
    """
    def g(k, d=0.0):
        try:
            return float(m.get(k, d) or d)
        except (TypeError, ValueError):
            return d
    dist = g("mean_dist", float("nan"))
    return [g(select_on), g("grasp_rate"), g("near_rate"),
            -dist if dist == dist else float("-inf")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="output/dagger_runs/<name> (must contain config.yaml)")
    p.add_argument("--out", default="eval_log.csv",
                   help="CSV inside the run dir (default: eval_log.csv)")
    p.add_argument("--iters", default="all",
                   help="'all' or a comma list, e.g. 0,2,4")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="override EVAL.num_scenes (changes the pool — recorded in the CSV)")
    p.add_argument("--ckpt", default=None, choices=[None, "best", "last"],
                   help="override EVAL.ckpt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="re-score iterations already in the CSV")
    p.add_argument("--watch", action="store_true",
                   help="poll for new iterations instead of exiting")
    p.add_argument("--poll-s", type=float, default=300.0,
                   help="--watch polling interval (default 300 s)")
    p.add_argument("--timeout-s", type=float, default=0.0,
                   help="--watch: give up after this long with no new iteration "
                        "(0 = never)")
    p.add_argument("--publish-best", action="store_true",
                   help="also export <run>/best from the winning iteration. Leave "
                        "OFF while the trainer is running — it writes there too.")
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
    if args.num_scenes is not None:
        ev["num_scenes"] = int(args.num_scenes)
    seed = int(args.seed if args.seed is not None
               else cfg4.get("DAGGER", {}).get("seed", 0))

    ctx = build_phase4_context(cfg4, seed=seed)
    ckpt = args.ckpt or ctx.eval_ckpt
    log_path = run_root / args.out

    print("=" * 78)
    print(f"Phase-4 standalone eval   run={run_root.name}")
    print(f"  eval scenes : {len(ctx.eval_scenes)}"
          f"{' (OVERRIDDEN)' if args.num_scenes is not None else ''}")
    print(f"  checkpoint  : {ckpt}     success={ctx.eval_params.success_mode}")
    print(f"  pinning     : {ctx.pin_table.describe() if ctx.pin_table else 'OFF'}")
    print(f"  writing     : {log_path}")
    print("=" * 78)

    want = None if args.iters == "all" else {int(x) for x in args.iters.split(",")}
    rows = {} if args.force else read_done(log_path)
    last_progress = time.time()

    while True:
        todo = [(i, d) for i, d in iteration_dirs(run_root)
                if (want is None or i in want) and i not in rows]
        for i, run_dir in todo:
            t0 = time.time()
            print(f"\n[iter {i:02d}] {run_dir}")
            try:
                runner, _ = load_policy_runner(run_dir, args.device, ckpt=ckpt)
            except Exception as e:                       # noqa: BLE001
                print(f"  [skip] cannot load {ckpt}.pt: {type(e).__name__}: {e}")
                continue
            m = evaluate_policy(ctx.sim, runner, ctx.eval_scenes,
                                params=ctx.eval_params, pin_table=ctx.pin_table)
            m.pop("rows", None)
            del runner
            if args.device != "cpu":
                import torch
                torch.cuda.empty_cache()
            eval_s = time.time() - t0
            rows[i] = row_from_metrics(i, run_dir, ckpt, len(ctx.eval_scenes),
                                       m, eval_s)
            write_log(log_path, rows)
            print(f"  success={m['success_rate']:.3f} grasp={m['grasp_rate']:.3f} "
                  f"near={m['near_rate']:.3f} close={m['close_rate']:.3f} "
                  f"chance={m['chance_rate']:.3f}  ({eval_s:.0f}s)")
            last_progress = time.time()

        if not args.watch:
            break
        if args.timeout_s and time.time() - last_progress > args.timeout_s:
            print(f"\n[watch] no new iteration for {args.timeout_s:.0f}s — stopping.")
            break
        if not todo:
            print(f"[watch] up to date ({len(rows)} scored) — sleeping {args.poll_s:.0f}s",
                  flush=True)
            time.sleep(args.poll_s)

    if rows:
        best = max(rows, key=lambda i: score_tuple(rows[i], ctx.select_on))
        print(f"\nbest: iteration {best}  {ctx.select_on}="
              f"{rows[best].get(ctx.select_on)}  ({len(rows)} iterations scored)")
        print(f"wrote {log_path}")
        if args.publish_best:
            export_run_dir(Path(rows[best]["run_dir"]), run_root / "best", ckpt=ckpt,
                           note=f"DAgger iteration {best} — best {ctx.select_on} "
                                f"(standalone eval)")
            print(f"published {run_root / 'best'}")
    else:
        print("\nnothing scored — no completed iterations found in state.json")


if __name__ == "__main__":
    main()
