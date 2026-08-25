"""
How far along is a Regrasp run, and at what pace.

    python examples/status_regrasp_run.py output/dagger_runs/regrasp_fast1
    watch -n 60 python examples/status_regrasp_run.py output/dagger_runs/regrasp_fast1

Reads `<run>/dagger_log.csv` and `<run>/config.yaml` only, so it is safe to run
against a live job — the CSV is appended and flushed once per iteration.

WHY NOT JUST `tail` THE LOG. The log tells you what the run is doing right now;
this tells you how much is left, which is a different question and the one you
actually have to plan around. It also surfaces the four numbers worth checking
mid-run without waiting for a plot:

    succ        success_rate on the eval scenes
    dir_track   1 - mean(dir_err)/90. 1 = the gripper ends on the axis it was
                told to come in on, 0 = it ignores the command
    bin_diag    how often the REALISED bin is the commanded one. Chance is 0.25
                with four live bins — a value sitting there means the
                conditioning is inert, and that is the result, not a bug
    train/val   a RISING train loss under FTL means the aggregate is becoming
                self-inconsistent, the one failure DAgger cannot average away

PACE AND ETA ARE FROM COMPLETED ITERATIONS ONLY, and iteration 0 is excluded
from the average: it is the base fit on the expert set, not a DAgger round, and
including it would drag the estimate in whichever direction `base_epochs`
happens to point. The "current iter running N min" figure is the age of the
CSV's last write, so it counts collection + refit + eval of the round in flight.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="output/dagger_runs/<name>")
    args = p.parse_args()

    run = Path(args.run_dir)
    log = run / "dagger_log.csv"
    if not log.exists():
        raise SystemExit(f"no {log} yet — the run has not finished iteration 0")

    with (run / "config.yaml").open() as f:
        total = int((yaml.safe_load(f) or {})["DAGGER"]["num_iters"])
    with log.open() as f:
        done = [r for r in csv.DictReader(f) if r.get("iter", "").strip()]
    if not done:
        raise SystemExit(f"{log} has a header but no rows yet")

    # Iteration 0 is the base fit, so completed DAgger rounds is one less.
    n = max(len(done) - 1, 0)
    walls = [float(r["wall_s"]) for r in done if r.get("wall_s", "").strip()]
    per = sum(walls[1:]) / max(len(walls) - 1, 1) if len(walls) > 1 else 0.0
    age = time.time() - os.path.getmtime(log)

    print(run)
    print(f"  iterations : {n} / {total}   ({100 * n / max(total, 1):.0f}%)")
    if per:
        left = (total - n) * per - age
        print(f"  pace       : {per / 60:.1f} min/iter   "
              f"ETA ~{max(left, 0) / 3600:.1f} h   "
              f"(current iter running {age / 60:.0f} min)")
    last = done[-1]
    print(f"  |D|        : {last.get('D_steps', '?')} steps / "
          f"{last.get('D_episodes', '?')} episodes")

    # WHERE THE TIME GOES, and whether it is going there MORE than it was.
    # `train_s` grows with |D| under FTL — every refit is a fresh fit on a bigger
    # aggregate — while `collect_s` and `eval_s` are flat (fixed episode counts).
    # So a naive mean over completed iterations UNDERSTATES the ETA, and the
    # extrapolation below uses the trend in train_s instead of its average.
    def col(k):
        return [float(r[k]) for r in done if r.get(k, "").strip()]
    cs, ts, es = col("collect_s"), col("train_s"), col("eval_s")
    if cs and ts and es:
        print(f"  last iter  : collect {cs[-1]/60:.1f} + refit {ts[-1]/60:.1f} "
              f"+ eval {es[-1]/60:.1f} = {(cs[-1]+ts[-1]+es[-1])/60:.1f} min")
        if len(ts) >= 2:
            growth = (ts[-1] - ts[0]) / max(len(ts) - 1, 1)
            rem = 0.0
            for j in range(1, total - n + 1):
                rem += cs[-1] + es[-1] + ts[-1] + growth * j
            print(f"  refit trend: +{growth:.0f} s/iter as |D| grows "
                  f"-> ~{max(rem - age, 0)/3600:.1f} h left "
                  f"({(rem/max(total-n,1))/60:.1f} min/iter ahead)")
    print()
    for r in done:
        def g(k, d="—"):
            v = r.get(k, "").strip()
            return v if v else d
        print(f"   it{g('iter'):>3}  beta={g('beta'):>5}  m={g('m'):>4}  "
              f"succ={g('success_rate'):>6}  dir_track={g('dir_track'):>6}  "
              f"bin_diag={g('bin_diag_rate'):>6}  "
              f"train={g('train_loss'):>6} val={g('val_loss'):>6}  "
              f"{float(g('wall_s', '0')) / 60:>5.1f} min "
              f"(c{float(g('collect_s','0'))/60:.0f}/t{float(g('train_s','0'))/60:.0f}"
              f"/e{float(g('eval_s','0'))/60:.0f})")


if __name__ == "__main__":
    main()
