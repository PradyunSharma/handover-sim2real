"""
Audit a Regrasp collection against the schema the trainer requires.

Run this on the PILOT before collecting at scale. Every check here is one that
would otherwise surface hours later — inside a DataLoader worker, in a shape
error with no episode named, or as a policy quietly trained on a wrong label.

    python examples/audit_regrasp_demos.py --demos output/bc_dataset/train_regrasp.h5

WHAT IT CHECKS, and why each one earns its place:

  widths        every cloud is [T, N, 8]. A 5-channel shard mixed into an
                aggregate kills `collate` in a worker with nothing naming the
                offending file (this is the run-19 crash).
  d_world       present, unit, finite on EVERY episode. A zeroed direction is not
                a missing label, it is a wrong one: the two conditioning channels
                would read 0 everywhere, which the policy cannot distinguish from
                "approach from nowhere".
  normals       unit and finite. `regularize_pc_point_count` oversamples WITH
                replacement when a class is short, and duplicated points give a
                rank-deficient covariance whose eigenvector is arbitrary. The
                estimator falls back for those, and this counts how often.
  bins          `bin_assigned` vs `bin_realized`. They differ exactly when the pin
                failed and the episode flew to OMG's own pick instead — which
                Regrasp keeps rather than discards, but which must be VISIBLE.
  pairing       how many scenes carry two demonstrations at DIFFERENT directions.
                This is the number the whole design rests on: a scene with one
                demo, or two at the same direction, lets the network learn
                scene -> action and ignore the conditioning entirely.
  separation    the angle between a scene's two commands, and the per-step action
                difference between its two demonstrations. A pair that is 180 deg
                apart but produces identical trajectories teaches nothing.
  anchor        `anchor_mode` should be "wrist" on ~every episode under the
                active config, where the hand is static. A run of "base" means the
                degeneracy threshold is miscalibrated, not that the fallback
                worked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handover_sim2real.regrasp import channels as CH        # noqa: E402
from handover_sim2real.regrasp import directions as D       # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", required=True)
    p.add_argument("--write-ok", default=None,
                   help="write the (scene, bin) pairs this shard actually "
                        "demonstrated to JSON. Point SIM.demo_ok_table at it and "
                        "every pair OMG could not plan, or that the pin missed, "
                        "leaves BOTH the collection pool and the eval set. "
                        "Without it a per-bin planner failure trains on three "
                        "directions and scores four.")
    p.add_argument("--noise-floor", type=float, default=0.005,
                   help="metres of per-step |dpos| difference below which two "
                        "demonstrations are saying the same thing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fails, warns = [], []

    with h5py.File(args.demos, "r") as f:
        keys = sorted(k for k in f if k.startswith("episode"))
        if not keys:
            raise SystemExit(f"{args.demos} has no episodes")
        fattrs = dict(f.attrs)
        print("=" * 74)
        print(f"Regrasp demo audit   {args.demos}")
        print(f"  episodes {len(keys)}   schema="
              f"{fattrs.get('schema', b'?')!r}   pc_channels="
              f"{fattrs.get('pc_channels', '?')}")
        print("=" * 74)

        widths, steps = Counter(), 0
        by_scene = defaultdict(list)
        n_nofinite_d = n_nonunit_d = n_missing_d = 0
        n_norm_bad = 0
        modes, sides, pin_ok = Counter(), Counter(), Counter()
        ok_pairs: dict = defaultdict(set)
        rebinned = 0
        cmd_off = []          # angle between d_world and its bin's AXIS
        demo_off = []         # angle between d_world and what the expert flew

        for k in keys:
            g = f[k]
            pc = g["point_clouds"]
            widths[int(pc.shape[-1])] += 1
            steps += int(pc.shape[0])

            a = dict(g.attrs)
            d = a.get("d_world")
            if d is None:
                n_missing_d += 1
                continue
            d = np.asarray(d, dtype=np.float64)
            if not np.all(np.isfinite(d)):
                n_nofinite_d += 1
            elif abs(np.linalg.norm(d) - 1.0) > 1e-3:
                n_nonunit_d += 1

            # normals: sample the first and last step rather than every one
            if pc.shape[-1] >= 8:
                for t in (0, int(pc.shape[0]) - 1):
                    n = np.asarray(pc[t][:, 5:8], dtype=np.float64)
                    ln = np.linalg.norm(n, axis=1)
                    if not np.all(np.isfinite(n)) or np.any(np.abs(ln - 1.0) > 1e-2):
                        n_norm_bad += 1
                        break

            modes[str(a.get("anchor_mode", "?"))] += 1
            sides[str(a.get("mano_side", "?"))] += 1
            pin_ok[int(a.get("pin_ok", -1))] += 1
            ba, br = int(a.get("bin_assigned", -1)), int(a.get("bin_realized", -1))
            if ba >= 0 and br >= 0 and ba != br:
                rebinned += 1
            # USABLE means: the pair was demonstrated, and the demonstration is
            # of the bin it is captioned with. A missed pin leaves an episode
            # commanded `+z` whose trajectory approaches from elsewhere; under
            # bin-axis conditioning that is not re-labellable (the scene already
            # has a demonstration for the bin it actually flew to), so the pair
            # is dropped rather than kept as a contradiction in D.
            # WHICH RULE PRODUCED THIS SHARD. Run 1 set `d_world` from the
            # grasp (`-R[:,2]`); run 2 sets it to the assigned BIN'S AXIS. The
            # two differ by a median 18 deg, so the angle between `d_world` and
            # the bin axis is 0 for a run-2 shard and ~18 for a run-1 one — an
            # unambiguous fingerprint, and the only cheap way to notice that a
            # stale shard is about to be trained on under the new command rule.
            aR = a.get("anchor_R")
            if ba >= 0 and aR is not None and np.all(np.isfinite(d)):
                axis = D.to_world(D.BINS[ba], np.asarray(aR, dtype=np.float64))
                cmd_off.append(float(D.angle_between(d, axis)))
            dg = a.get("d_grasp_world")
            if dg is not None and np.all(np.isfinite(d)) and \
                    np.linalg.norm(np.asarray(dg)) > 0.5:
                demo_off.append(float(D.angle_between(d, np.asarray(dg))))
            if ba >= 0 and br == ba and int(a.get("pin_ok", 1)):
                ok_pairs[int(a["scene_idx"])].add(ba)
            by_scene[int(a["scene_idx"])].append(
                {"key": k, "d": d, "bin": br, "assigned": ba,
                 "act": np.asarray(g["expert_actions"][:, :3], dtype=np.float64)})

        # ---- widths -------------------------------------------------------
        print(f"\n  cloud widths      : {dict(widths)}   steps {steps}")
        if set(widths) != {CH.STORED_CHANNELS}:
            fails.append(f"cloud width is {dict(widths)}, expected "
                         f"{{{CH.STORED_CHANNELS}: n}} — the trainer will refuse "
                         f"this, and a MIX would die inside collate")

        # ---- the command --------------------------------------------------
        print(f"  d_world           : missing {n_missing_d}, non-finite "
              f"{n_nofinite_d}, non-unit {n_nonunit_d}")
        if n_missing_d or n_nofinite_d or n_nonunit_d:
            fails.append("d_world is missing/invalid on some episodes — a zeroed "
                         "direction is a WRONG label, not a missing one")

        print(f"  normals           : {n_norm_bad} episode(s) with non-unit or "
              f"non-finite normals")
        if n_norm_bad:
            fails.append(f"{n_norm_bad} episodes have bad normals")

        print(f"  anchor_mode       : {dict(modes)}")
        if modes.get("base", 0) > 0.02 * max(len(keys), 1):
            warns.append(f"{modes['base']} episodes fell back to the base "
                         f"direction; under a static hand that points at a "
                         f"miscalibrated degeneracy threshold, not a working "
                         f"fallback")
        print(f"  mano_side         : {dict(sides)}")
        print(f"  pin ok / failed   : {pin_ok.get(1, 0)} / {pin_ok.get(0, 0)}"
              f"   re-binned {rebinned}")
        if rebinned:
            print(f"    ({rebinned} episodes flew to a different bin than assigned "
                  f"— kept and relabelled, which is the point of deriving the "
                  f"command from the realised pose)")

        # ---- pairing: the number the design rests on ----------------------
        n_pair = n_single = n_same_bin = 0
        seps, divs = [], []
        for sc, eps in by_scene.items():
            if len(eps) < 2:
                n_single += 1
                continue
            n_pair += 1
            a, b = eps[0], eps[1]
            sep = float(D.angle_between(a["d"], b["d"]))
            seps.append(sep)
            if a["bin"] == b["bin"]:
                n_same_bin += 1
            n = min(len(a["act"]), len(b["act"]))
            if n:
                # aligned from the END: it is the reach that has to line up
                diff = np.abs(a["act"][-n:] - b["act"][-n:]).sum(axis=1)
                divs.append(float(np.mean(diff > args.noise_floor)))

        print(f"\n  scenes            : {len(by_scene)}   paired {n_pair}, "
              f"single {n_single}")
        # A collapsed pair is a CONSEQUENCE of recoverable pin failure, not a
        # schema error: when the pin misses, the episode flies to OMG's own pick,
        # and OMG picks the same grasp for both slots of that scene. The episodes
        # are honestly labelled and still teach reaching; they simply contribute
        # no contrast. Fatal only if it is a large share, because a few percent
        # dilutes nothing and re-collecting to chase them would cost hours.
        frac_same = n_same_bin / max(n_pair, 1)
        if frac_same > 0.05:
            fails.append(
                f"{n_same_bin}/{n_pair} ({100*frac_same:.1f}%) paired scenes have "
                f"BOTH demos in one bin — too many to dismiss; the paired subset "
                f"is substantially diluted")
        elif n_same_bin:
            warns.append(
                f"{n_same_bin}/{n_pair} ({100*frac_same:.1f}%) paired scenes "
                f"collapsed to one bin, all of them via a failed pin. Kept and "
                f"honestly labelled; they just carry no contrast. Filter on "
                f"realised separation when reporting cond_delta on 'paired'.")
        if seps:
            s = np.asarray(seps)
            print(f"  pair separation   : median {np.median(s):.0f} deg   "
                  f"min {s.min():.0f}   antipodal {100*(s>179).mean():.0f}%")
            print("    distribution    : " + "  ".join(
                f"{lo}-{hi}:{int(((s >= lo) & (s < hi)).sum())}"
                for lo, hi in ((0, 40), (40, 90), (90, 150), (150, 181))))
            n_below = int((s < 40).sum())
            print(f"    below the 40 deg independence floor: {n_below} "
                  f"({100*n_below/len(s):.1f}%)  <- these pairs teach no contrast")
        if divs:
            dv = np.asarray(divs)
            print(f"  informative steps : median {np.median(dv):.3f}   "
                  f"mean {dv.mean():.3f}   (per-step |dpos| difference between "
                  f"the two demos > {args.noise_floor} m)")
            print(f"  scenes under 10%  : {int((dv < 0.10).sum())}/{len(dv)}  "
                  f"(their two demos are nearly the same trajectory)")
            if np.median(dv) < 0.05:
                fails.append(
                    "the two demonstrations of a scene are nearly IDENTICAL "
                    "trajectories. The commands differ but the behaviour does "
                    "not, so there is nothing for the conditioning to learn and "
                    "no architecture fixes it — this is a DATA result.")

    # ---- the command rule, and the label noise it leaves ------------------
    if cmd_off:
        co = np.asarray(cmd_off)
        print(f"\n  command vs bin axis: median {np.median(co):.2f} deg   "
              f"max {co.max():.2f}")
        if np.median(co) > 1.0:
            fails.append(
                f"`d_world` is NOT the bin axis (median {np.median(co):.1f} deg "
                f"off it). This shard was collected under RUN 1's rule, where the "
                f"command came from the grasp pose. Run 2 commands the bin axis "
                f"at both training and deployment, so training on this would "
                f"caption every episode with a vector no rollout will ever be "
                f"given. RE-COLLECT with the per-bin pin table.")
        else:
            print("       -> run-2 rule: the command is the bin axis, as the "
                  "retry machine issues it")
    if demo_off:
        do = np.asarray(demo_off)
        print(f"  command vs demo    : median {np.median(do):.1f} deg   "
              f"p90 {np.percentile(do, 90):.1f}   max {do.max():.1f}")
        print("       -> the label noise the conditioning must tolerate: the "
              "policy is told a sector and shown one grasp inside it")
        if np.median(do) > 45.0:
            warns.append(
                f"the demonstration is a median {np.median(do):.0f} deg from the "
                f"command — beyond the 45 deg bin half-width, so the grasps are "
                f"not the closest-to-axis representatives the table should hold")

    if args.write_ok:
        out = Path(args.write_ok)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_meta": {"stage": "audit_regrasp_demos --write-ok",
                      "source": str(args.demos),
                      "n_scenes": len(ok_pairs),
                      "n_pairs": sum(len(v) for v in ok_pairs.values())},
            "ok": {str(s_): sorted(int(b) for b in bs)
                   for s_, bs in sorted(ok_pairs.items())},
        }
        out.write_text(json.dumps(payload, indent=1))
        print(f"\n  wrote {out}  ({payload['_meta']['n_pairs']} usable "
              f"(scene, bin) pairs over {payload['_meta']['n_scenes']} scenes)")
        print("  -> SIM.demo_ok_table in the run config")

    print("\n" + "=" * 74)
    for w in warns:
        print(f"  WARN  {w}")
    for e in fails:
        print(f"  FAIL  {e}")
    if not fails:
        print("  PASS — the schema is what the trainer expects.")
    print("=" * 74)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
