#!/usr/bin/env python3
"""Collect the PRE-GRASP demonstration set D_0 with the Phase-4 collector itself.

WHY THIS EXISTS RATHER THAN examples/make_pregrasp_demos.py. That script derives
D_0 by truncating the grasp-target demonstrations, which is free and keeps the
approach data bit-identical to run 16's — but it cannot produce a SETTLED
terminal state, because the demonstrations never hold the pre-grasp. The base
collector plays OMG's plan by index: it commands the pre-grasp waypoint, the arm
covers ~77.5% of the distance in one 150 ms control window, and the collector
immediately commands the next waypoint instead of waiting. So the truncated
terminal row sits a median 11.6 mm short of the pre-grasp and is labelled COMMIT
there.

That is exactly the pose `DAGGER.commit_settle_steps` exists to remove, and the
two cannot coexist: a settle labels the 11.6 mm state APPROACH in every DAgger
shard while the truncated D_0 labels it COMMIT. Same state, opposite labels, one
aggregate — the inconsistency DAgger cannot average away, sitting on the one
decision the run is about.

The fix is to produce D_0 through the SAME code path as the shards, at beta = 1
(pure expert, the learner never drives), with the same `target`, the same
tolerances and the same settle. Then every commit label in the aggregate — base
and DAgger alike — is written by one rule at one distance, and the classifier
sees a clean margin between the last APPROACH state and the COMMIT state.

DART IS FORCED OFF here regardless of what the run config says. The shards want
off-distribution coverage; D_0 is the expert demonstration set and its whole job
is to be clean. Leaving `dart_ratio` at the run's 0.3 would put a third of the
approach steps on a random jolt.

THIS DOUBLES AS THE DECISIVE SMOKE TEST. With `outcome_check` on (inherited from
the run config), every episode executes the blind push and the close and is
scored on the real success criterion. So the summary at the end answers, over
every usable scene and with no policy involved, the question the whole run rests
on: from a correctly settled pre-grasp, does a blind 6.4 cm push actually secure
the object? If `GRASP_OK` is low here, no policy can rescue it and the geometry
needs revisiting before 17 hours are spent.

    # train split (the base set)
    python examples/collect_pregrasp_demos.py \
        --cfg examples/configs/dagger_phase4_all_beta075_pregrasp.yaml \
        --output output/bc_dataset/train_pregrasp_settled.h5

    # val split (selects best.pt, so it must be converted the same way)
    python examples/collect_pregrasp_demos.py \
        --cfg examples/configs/dagger_phase4_all_beta075_pregrasp.yaml \
        --split val \
        --grasp-pin-table output/grasp_pin_table_val_omg.json \
        --output output/bc_dataset/val_pregrasp_settled.h5

    # 6-scene shakedown before committing to the full pass
    python examples/collect_pregrasp_demos.py --cfg ... --limit 6 --output /tmp/smoke.h5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from handover_sim2real.dagger import CollectParams, collect_iteration  # noqa: E402
from handover_sim2real.dagger.pregrasp import forward_dist_default  # noqa: E402
from handover_sim2real.dagger.setup import build_phase4_context  # noqa: E402


class NullRunner:
    """A learner that never moves and never commits.

    At beta = 1 the expert wins every mixing draw, so this is never executed —
    but `collect_dagger_episode` still calls `act()` once per step (a stateful
    ACT runner needs its bookkeeping advanced, so the call is unconditional).
    Returning a zero delta with the gripper OPEN keeps the DART sigma estimator
    and the premature-close counters well defined without a checkpoint to load.
    """

    kind = "null"

    def reset(self) -> None:
        pass

    def act(self, pc, rs):
        return np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)

    def describe(self) -> str:
        return "null (beta=1, expert only)"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", required=True,
                   help="the RUN config (its DAGGER block defines target, "
                        "tolerances and commit_settle_steps, so D_0 and the "
                        "shards cannot be collected under different rules)")
    p.add_argument("--output", required=True, help="HDF5 to write")
    p.add_argument("--split", default=None, choices=["train", "val", "test"],
                   help="override SIM.split (default: whatever the config says)")
    p.add_argument("--grasp-pin-table", default=None,
                   help="override SIM.grasp_pin_table — REQUIRED with --split "
                        "val, since scene indices are split-relative and a train "
                        "table applied to val scenes pins the wrong grasps")
    p.add_argument("--exclude-scenes", default=None,
                   help="override SIM.exclude_scenes ('none' to disable)")
    p.add_argument("--limit", type=int, default=None,
                   help="collect only the first N usable scenes (shakedown)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.cfg) as f:
        cfg4 = yaml.safe_load(f)

    sim_cfg_d = cfg4.setdefault("SIM", {})
    if args.split:
        sim_cfg_d["split"] = args.split
    if args.grasp_pin_table:
        sim_cfg_d["grasp_pin_table"] = args.grasp_pin_table
    if args.exclude_scenes:
        sim_cfg_d["exclude_scenes"] = (None if args.exclude_scenes == "none"
                                       else args.exclude_scenes)
    if sim_cfg_d.get("split") == "val" and not args.grasp_pin_table:
        raise SystemExit(
            "--split val needs --grasp-pin-table output/grasp_pin_table_val_omg.json "
            "— scene indices are split-relative, so the train table would pin the "
            "wrong grasp on every scene")

    dag = cfg4.get("DAGGER", {}) or {}
    target = str(dag.get("target", "grasp"))
    if target != "pregrasp":
        print(f"WARNING: DAGGER.target is {target!r}, not 'pregrasp' — this will "
              f"collect a grasp-target set, which examples/collect_bc_dataset.py "
              f"already does more cheaply.")

    ctx = build_phase4_context(cfg4, seed=int(args.seed), verbose=True)

    params = CollectParams(
        max_steps=int(dag.get("max_steps", 50)),
        close_pos_thresh=float(dag.get("close_pos_thresh", 0.02)),
        close_rot_thresh=float(dag.get("close_rot_thresh", 0.34)),
        stop_on_close_label=True,
        # Inert at beta = 1 (the learner never drives), but pinned rather than
        # inherited so this cannot become load-bearing by accident.
        stop_on_policy_close=False,
        ee_step=float(dag.get("ee_step", 0.04)),
        reach_tail=int(dag.get("reach_tail", 5)),
        min_free=int(dag.get("min_free", 1)),
        max_horizon=int(dag.get("max_horizon", 40)),
        first_horizon=int(dag.get("first_horizon") or 20),
        derive_standoff=bool(dag.get("derive_standoff", False)),
        standoff_dist=float(sim_cfg_d.get("standoff_dist", 0.08)),
        target=target,
        forward_dist=float(dag.get("forward_dist") or forward_dist_default(
            float(sim_cfg_d.get("standoff_dist", 0.08)),
            int(dag.get("reach_tail", 5)))),
        forward_steps=int(dag.get("forward_steps", 4)),
        # THE POINT OF THIS SCRIPT: the base commit labels are written at the same
        # settled distance the shards use.
        commit_settle_steps=int(dag.get("commit_settle_steps", 0)),
        reach_commit_dist=float(dag.get("reach_commit_dist", 0.05)),
        reach_skip_eps=float(dag.get("reach_skip_eps", 0.01)),
        # beta = 1 already means the expert drives; making it explicit keeps the
        # committed segment on the expert even if beta is ever lowered here.
        expert_after_commit=True,
        # ---- DART OFF, deliberately, whatever the run config says ----
        dart_ratio=0.0,
        dart_reach_ratio=0.0,
        dart_mode="jolt",           # inert at ratio 0; keeps the rng stream clean
        # Scoring is what makes this a smoke test as well as a collection.
        outcome_check=bool(dag.get("outcome_check", True)),
        hold_steps=int((cfg4.get("EVAL", {}) or {}).get("hold_steps", 3)),
    )

    # EVERY usable scene, not a sample: this is the demonstration set, and
    # `scene_pools` splits the pool for DAgger's purposes, which is not what a
    # base set wants. `ctx.usable` is the pin table's keys minus exclude_scenes.
    scenes = (sorted(ctx.usable) if ctx.usable is not None
              else list(range(ctx.sim.num_scenes)))
    if args.limit:
        scenes = scenes[:int(args.limit)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Pre-grasp D_0 collection   target={params.target}  beta=1.0 (expert only)")
    print(f"  scenes        : {len(scenes)} usable in split="
          f"{sim_cfg_d.get('split', 'train')}")
    print(f"  commit        : within {params.close_pos_thresh} m / "
          f"{params.close_rot_thresh} rad of the pre-grasp, then "
          f"{params.commit_settle_steps} settle step(s)")
    print(f"  blind push    : {params.forward_dist} m over {params.forward_steps} steps")
    print(f"  DART          : off (ratio 0.0 / reach 0.0)")
    print(f"  outcome_check : {params.outcome_check}")
    print(f"  output        : {out}")
    print("=" * 78)

    t0 = time.time()
    agg = collect_iteration(
        ctx.sim, NullRunner(), scenes, out, rng=np.random.RandomState(int(args.seed)),
        beta=1.0, params=params, pin_table=ctx.pin_table, iteration=0,
        progress_every=20)
    dt = time.time() - t0

    eps = int(agg["episodes"]) or 1
    print("\n" + "=" * 78)
    print(f"episodes kept   : {agg['episodes']}  ({agg['skipped']} skipped)")
    print(f"steps           : {agg['steps']}   ({agg['steps'] / eps:.1f} per episode)")
    print(f"commit labels   : {agg['n_close_labels']}")
    print(f"settle steps    : {agg['n_settle_steps']}  "
          f"(expected {params.commit_settle_steps * agg['n_close_labels']}; a "
          f"shortfall means episodes ended before converging)")
    print(f"terminated by   : {agg['reasons']}")
    print(f"wall clock      : {dt / 60:.1f} min")

    if params.outcome_check:
        print("\n---- THE QUESTION THIS RUN RESTS ON ----")
        print(f"arrival: mean distance to the PRE-GRASP at closest approach "
              f"= {agg['mean_min_pos']:.4f} m")
        print(f"landing: mean distance to the GRASP after the blind push "
              f"= {agg['mean_reach_pos_err']:.4f} m "
              f"(rot {agg['mean_reach_rot_err']:.4f} rad)")
        print(f"success: {agg['success']}/{agg['episodes']} = "
              f"{agg['success'] / eps:.1%} secured after push + close")
        print(f"outcomes: {agg['outcomes']}")
        print("\nThis is the expert, with no learner involved, so it is the CEILING "
              "for the run.\nIf it is low, the geometry is wrong and no policy "
              "recovers it.")

    # Provenance, so a settled set can never be mistaken for a truncated one.
    import h5py
    with h5py.File(out, "a") as f:
        f.attrs["source"] = "collect_pregrasp_demos.py (beta=1, Phase-4 collector)"
        f.attrs["run_cfg"] = str(args.cfg)
        f.attrs["split"] = str(sim_cfg_d.get("split", "train"))
        f.attrs["commit_settle_steps"] = int(params.commit_settle_steps)
        f.attrs["expert_success_rate"] = float(agg["success"] / eps)

    side = out.with_suffix(".json")
    with side.open("w") as f:
        json.dump({k: v for k, v in agg.items()
                   if isinstance(v, (int, float, str, dict, list))}, f, indent=2,
                  default=str)
    print(f"\nwrote {out}\n      {side}")


if __name__ == "__main__":
    main()
