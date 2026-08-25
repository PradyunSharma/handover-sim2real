"""
Stage 2 of the Regrasp pipeline: choose which directions each scene demonstrates.

Pure combinatorics over `direction_table_<split>.json`. No simulator, no GPU,
seconds — which is the point: `k`, the tie-break, and the feasibility rule can all
be re-decided without touching the cluster.

    python examples/assign_direction_demos.py \\
        --table output/direction_table_train.json \\
        --out output/regrasp_pins_train --mode per-bin \\
        --drop-bins='-z_beneath,-x_over_fingers'

WHY MORE THAN ONE DEMONSTRATION PER SCENE, AND WHY IT IS THE WHOLE POINT. At ONE
demonstration per scene, `d` is a deterministic function of the observation
across the entire dataset: every (point cloud, robot state) it ever sees comes
with exactly one command, so the network can drive the loss to its floor by
learning scene -> action and ignoring the conditioning channels completely. Two
or more demonstrations of the same scene under different `d` map the SAME
observation to DIFFERENT actions, and that is the only thing that forces the
channels to be read. This is a property of the DATASET, not of the architecture
— no amount of FiLM or extra capacity reaches it.

TWO MODES.

  --mode per-bin  (run 2 on, the DEFAULT). `--per-bin N` demonstrations for EVERY
      bin the scene can reach, the N goal-set grasps CLOSEST TO THAT BIN'S AXIS.
      N defaults to 1, which is run 2; run 3 uses 3. The
      direction table already picked that representative (`angle_to_axis_deg`),
      and it is the right one precisely because the axis is what the policy is
      commanded with at deployment — `retry.next_direction` has no grasp to read,
      so it issues `to_world(BINS[b], anchor_R)`. Picking any other member of the
      bin would widen the gap between what the policy is told and what it is
      shown for no benefit. Measured on s0/train: 617 scenes, 1596 demos, mean
      2.59 per scene, split +x 490 / +y 366 / -y 325 / +z 415, with the
      demonstration a median 18.4 deg from its bin axis (p90 38.5, max 45).

  --mode pair     (run 1). The single maximally-separated pair among the feasible
      bins, tie-broken toward globally emptier bins. 1088 demos.

Per-bin supersedes pair rather than extending it: four contrasting commands on
one observation break the confound harder than two, and — the reason that
decided it — they populate every bin's EVALUATION sample instead of only the two
a scene happened to be assigned, which is what makes a per-bin figure readable.

`--per-bin N` DOES NOT BREAK THE CONFOUND FURTHER, AND IS NOT MEANT TO. N demos
of one bin share ONE command, so they map the same observation to N different
actions with nothing to distinguish them — the opposite of the paragraph above.
What they teach is that a direction does not name a single pose, which is true
and is what the policy meets at deployment. The cost is that a unimodal
regression loss (`pose_loss: pm`) resolves conflicting targets by AVERAGING them,
and the mean of three valid grasps need not be a valid grasp. Measured on
s0/train: 91% of (scene, bin) pairs have three or more goal-set members, and
`--per-bin 3` yields ~4601 demonstrations against run 2's 1596. The diagnostic if
it goes wrong is `cond_sep` and the per-bin `bin_diag_rate` falling WHILE
`train_loss` also falls — the signature of fitting a mean nobody asked for.
`--per-bin 1` reproduces run 2 exactly and is the control.

EMISSION ORDER IS MEMBER-MAJOR under `--per-bin N`: slots cycle the bins
(+x, +y, -y, +z, +x, ...) rather than exhausting one bin at a time. The retry
ladder reads a scene's slots IN ORDER, so bin-major grouping would make retry@2 a
second attempt at the same direction under an identical command — a repeat, not a
retry. See the comment at the emission site.

SCENES WITH ONE FEASIBLE BIN STILL GET COLLECTED, once. They teach reaching and
grasping and carry a valid `d`; they simply contribute nothing to breaking the
confound. `paired` and `n_demos` are recorded per scene so `cond_delta_train` can
be reported on the multi-demo subset separately and the dilution measured rather
than assumed.

MEASURED ON s0/train (623 planned scenes, 28339 goal-set grasps):

    feasible bins per scene   1:126  2:160  3:167  4:159  5:5
    able to supply a pair     491/623 (78.8%)
    pair separation           median 90 deg, 50% antipodal
    -z reachable by           0 scenes      -x by 12 (0.19% of all grasps)

so the retry ladder has FOUR live rungs, `+y/-y` is the only antipodal pair
available in practice, and about half the dataset gets a 90-degree contrast
rather than 180.

A NOTE FOR WHOEVER WIRES THIS INTO THE LOOP. The emitted table has scenes with
one grasp and scenes with two, so `GraspPinTable.num_grasps` — which is
`min(len(v))` over scenes — will read 1 and is the WRONG accessor here. Use
`pairs()` / `num_grasps_for(scene)`, which are already per-scene. Anything that
multiplies a scene count by `num_grasps` needs revisiting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handover_sim2real.regrasp import directions as D          # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", required=True, help="direction_table_<split>.json")
    p.add_argument("--out", required=True,
                   help="prefix; writes <out>.json and <out>_excluded.json")
    p.add_argument("--min-bins", type=int, default=1,
                   help="drop scenes reaching fewer than this many bins "
                        "(1 keeps single-bin scenes, 2 keeps only paired ones)")
    p.add_argument("--max-angle", type=float, default=45.0,
                   help="ignore table entries further than this from a bin axis")
    p.add_argument("--drop-bins", default="",
                   help="comma list of bin names or indices to treat as "
                        "infeasible, e.g. '-z_beneath,-x_over_fingers'")
    p.add_argument("--min-sep-deg", type=float, default=40.0,
                   help="PAIR MODE ONLY. Reject a pair whose REALISED directions "
                        "are closer than this; below ~40 deg the two commands "
                        "stop being independent hypotheses and the scene "
                        "contributes a contrast the policy cannot be expected to "
                        "resolve. The scene falls back to a single demonstration.")
    p.add_argument("--mode", choices=("per-bin", "pair"), default="per-bin",
                   help="per-bin (run 2 on, the default): a demonstration for "
                        "every bin the scene can reach, each the goal-set grasp "
                        "closest to that bin's axis. pair (run 1): the single "
                        "maximally-separated pair.")
    p.add_argument("--per-bin", type=int, default=1,
                   help="PER-BIN MODE ONLY. How many demonstrations each bin "
                        "gets, taken as a prefix of the table's `members` list "
                        "(closest to the bin axis first). 1 reproduces run 2; 3 "
                        "is run 3. Capped per bin by what the goal set actually "
                        "holds, so a bin with one member still contributes one. "
                        "Requires a table built with --members-per-bin >= N; an "
                        "older table has only the head and this silently reads "
                        "as 1, so the count is checked and reported.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def bin_members(entry: dict, n: int) -> list:
    """Up to `n` demonstrations of one bin, closest to that bin's axis first.

    `build_direction_table.py --members-per-bin M` writes a `members` list whose
    head IS the single grasp run 2 recorded, so a prefix of length 1 reproduces
    run 2 exactly and a table built before `members` existed degrades to that one
    grasp rather than failing. The caller reports the shortfall — silently
    collecting one demonstration where three were asked for would look like a
    successful run of the wrong experiment.

    Each returned dict carries the per-bin keys (`bin`, `bin_name`, `n_members`)
    that live on the parent entry, so downstream code sees one flat record per
    demonstration and does not care whether it came from `members` or the head.
    """
    head = {k: entry[k] for k in ("ee_pose_world", "d_anchor", "d_world",
                                  "angle_to_axis_deg", "goal_set_idx")}
    ms = entry.get("members") or [head]
    keep = ms[:max(1, int(n))]
    return [dict(m, bin=entry["bin"], bin_name=entry["bin_name"],
                 n_members=entry["n_members"]) for m in keep]


def main() -> None:
    args = parse_args()
    raw = json.load(open(args.table))
    meta = raw.pop("_meta", {})
    bins = np.asarray(meta.get("bins", D.BINS.tolist()), dtype=np.float64)
    names = list(meta.get("bin_names", D.BIN_NAMES))

    drop = set()
    for tok in (t.strip() for t in args.drop_bins.split(",") if t.strip()):
        drop.add(int(tok) if tok.lstrip("-").isdigit() and tok in
                 [str(i) for i in range(len(bins))] else names.index(tok))
    if drop:
        print(f"[drop] treating as infeasible: {sorted(names[i] for i in drop)}")

    scenes = {int(k): v for k, v in raw.items() if v}
    counts = np.zeros(len(bins), dtype=np.float64)      # running per-bin demo count
    table: dict = {}
    excluded, reasons = [], Counter()
    pairs_hist, seps = Counter(), []
    per_scene_bins = Counter()      # per-bin mode: scenes reaching each bin
    n_paired = n_single = n_too_close = n_short = 0

    # Scene order is fixed (sorted) so the greedy assignment is reproducible; the
    # emptiness tie-break makes it order-dependent, and an unstable order would
    # make the table irreproducible from the same inputs.
    for idx in sorted(scenes):
        s = scenes[idx]
        feas = [e for e in s["bins"]
                if e["bin"] not in drop
                and float(e["angle_to_axis_deg"]) <= args.max_angle]
        if len(feas) < max(1, args.min_bins):
            excluded.append(idx)
            reasons["no_feasible_bin" if not feas else "too_few_bins"] += 1
            continue

        by_bin = {e["bin"]: e for e in feas}
        chosen_bins = sorted(by_bin)
        if args.mode == "per-bin":
            # ONE DEMONSTRATION PER REACHABLE BIN, and no selection at all beyond
            # what the direction table already did: it stores, per bin, the
            # goal-set grasp whose realised direction is CLOSEST TO THAT BIN'S
            # AXIS (`angle_to_axis_deg`). That is the right representative
            # precisely because the axis is what the policy is commanded with —
            # picking any other member of the bin would widen the gap between
            # what the policy is told and what it is shown for no benefit.
            #
            # This supersedes pair mode rather than extending it. Pair mode
            # existed to break the scene->action confound with two contrasting
            # commands on one observation; four commands on one observation break
            # it strictly harder, and they also populate every bin's eval sample
            # instead of only the two a scene happened to be assigned.
            #
            # `--per-bin N` gives each bin N demonstrations instead of one: N
            # different expert trajectories under the SAME command, which is what
            # teaches the policy that a direction does not name a single pose.
            #
            # EMISSION ORDER IS MEMBER-MAJOR, AND THAT IS LOAD-BEARING. The retry
            # ladder walks a scene's slots in table order — `retry_at_k` is
            # "success given the first k slots" — so bin-major order
            # (+x,+x,+x,+y,...) would make retry@2 a SECOND ATTEMPT AT THE SAME
            # DIRECTION. The command is identical for both, so the policy would
            # do the identical thing, and `retry_at_k` would flatten into a
            # measure of simulator noise rather than of regrasping. Cycling the
            # bins first (+x,+y,-y,+z,+x,...) keeps rungs 1..4 four DISTINCT
            # directions exactly as in run 2, and pushes the repeats to slots 5+
            # where the ladder no longer looks.
            n_per = max(1, int(args.per_bin))
            got = {b: bin_members(by_bin[b], n_per) for b in chosen_bins}
            chosen = [got[b][m] for m in range(n_per)
                      for b in chosen_bins if m < len(got[b])]
            for b in chosen_bins:
                per_scene_bins[b] += 1
                n_short += max(0, n_per - len(got[b]))
            # `paired` COUNTS DISTINCT BINS, NOT DEMONSTRATIONS. Under
            # `--per-bin 3` a one-bin scene emits three episodes, and calling
            # that "paired" would claim a contrast that does not exist: three
            # demonstrations of ONE command teach multimodality, not
            # conditioning. Only a second BIN maps the same observation to a
            # different command, which is the confound this whole file is about.
            if len(chosen_bins) >= 2:
                n_paired += 1
                # Record the WIDEST contrast the scene supplies, so the
                # separation statistics stay comparable with pair mode's.
                seps.append(max(
                    float(D.angle_between(by_bin[a]["d_anchor"],
                                          by_bin[c]["d_anchor"]))
                    for ii, a in enumerate(chosen_bins)
                    for c in chosen_bins[ii + 1:]))
            else:
                n_single += 1
        elif len(chosen_bins) >= 2:
            # MAXIMISE THE REALISED SEPARATION, NOT THE BIN-AXIS SEPARATION.
            # A grasp only has to lie within `max_angle` (45 deg) of its bin's
            # axis, so two grasps in 90-deg-separated bins can realise anywhere
            # from ~0 to ~180 deg apart — and the pilot measured a pair 35 deg
            # apart whose BINS were 90 deg apart. Since the network is
            # conditioned on the realised `d`, not on the bin, picking by bin
            # axis systematically overstates the contrast the policy actually
            # sees. Tie-break toward globally emptier bins as before.
            best, best_key = None, None
            for ii in range(len(chosen_bins)):
                for jj in range(ii + 1, len(chosen_bins)):
                    a, b = chosen_bins[ii], chosen_bins[jj]
                    sep = float(D.angle_between(by_bin[a]["d_anchor"],
                                                by_bin[b]["d_anchor"]))
                    key = (round(sep, 6), -(counts[a] + counts[b]))
                    if best_key is None or key > best_key:
                        best, best_key = (a, b), key
            if best_key[0] < args.min_sep_deg:
                # The best available contrast is too small to be worth a second
                # demonstration: collect one and count the scene as unpaired
                # rather than pretend it breaks the confound.
                n_too_close += 1
                chosen = [by_bin[chosen_bins[0]]]
                n_single += 1
            else:
                pick = best
                chosen = [by_bin[pick[0]], by_bin[pick[1]]]
                seps.append(best_key[0])
                pairs_hist[tuple(sorted(pick))] += 1
                n_paired += 1
        elif args.mode == "pair":
            chosen = [by_bin[chosen_bins[0]]]
            n_single += 1
        for e in chosen:
            counts[e["bin"]] += 1

        table[str(idx)] = {
            "paired": len({e["bin"] for e in chosen}) >= 2,
            "n_demos": len(chosen),
            "n_bins": len({e["bin"] for e in chosen}),
            "n_feasible_bins": len(feas),
            "anchor_R": s["anchor_R"],
            "anchor_mode": s["anchor_mode"],
            "wrist_world": s["wrist_world"],
            "centroid_world": s["centroid_world"],
            "mano_side": s.get("mano_side"),
            "hand_present": s.get("hand_present", True),
            # `ee_pose_world` is the key GraspPinTable matches on; the direction
            # fields ride along untouched (scene_meta keeps non-"grasps" keys).
            "grasps": [{"ee_pose_world": e["ee_pose_world"],
                        "bin": e["bin"], "bin_name": names[e["bin"]],
                        "d_anchor": e["d_anchor"], "d_world": e["d_world"],
                        "angle_to_axis_deg": e["angle_to_axis_deg"],
                        "goal_set_idx": e["goal_set_idx"],
                        "n_members": e["n_members"]}
                       for e in chosen],
        }

    n = len(table)
    print("=" * 74)
    print(f"Regrasp pin assignment   from {args.table}")
    print(f"  scenes in table   : {len(scenes)}")
    print(f"  kept              : {n}   (paired {n_paired}, single {n_single})")
    print(f"  excluded          : {len(excluded)}  {dict(reasons)}")
    if args.mode == "per-bin" and args.per_bin > 1:
        have = int(meta.get("members_per_bin", 1) or 1)
        print(f"  demos per bin     : {args.per_bin} requested, table records up "
              f"to {have} per bin")
        if have < args.per_bin:
            print(f"  ERROR the table was built with --members-per-bin {have}, so "
                  f"at most {have} can be emitted per bin. Rebuild it:\n"
                  f"        python examples/build_direction_table.py --split "
                  f"<split> --members-per-bin {args.per_bin} --out {args.table}")
            raise SystemExit(2)
        if n_short:
            print(f"  short of {args.per_bin}          : {n_short} bin-slots — "
                  f"that bin's goal set held fewer members. Expected and "
                  f"harmless; the bin still contributes what it has.")
        print(f"  demoted to single : {n_too_close}  (best realised contrast "
              f"< {args.min_sep_deg:.0f} deg)")
    print(f"  mode              : {args.mode}")
    print(f"  demos             : {int(counts.sum())}"
          + (f"   ({counts.sum() / max(n, 1):.2f} per scene)" if n else ""))
    if seps:
        sa = np.asarray(seps)
        print(f"  pair separation   : median {np.median(sa):.0f} deg   "
              f"min {sa.min():.0f}   antipodal {100 * (sa > 179).mean():.0f}%"
              f"   (REALISED, not bin-axis)")
    print(f"\n  {'bin':<18} {'demos':>7}")
    for i, nm in enumerate(names):
        flag = "   <-- never demonstrated" if counts[i] == 0 else (
            "   <-- too few to learn" if 0 < counts[i] < 25 else "")
        print(f"  {nm:<18} {int(counts[i]):>7}{flag}")
    if pairs_hist:
        print("\n  pairs chosen:")
        for (a, b), c in pairs_hist.most_common():
            print(f"    {names[a]:<18} + {names[b]:<18} {c:>4}  "
                  f"({D.angle_between(bins[a], bins[b]):.0f} deg)")

    live = int((counts > 0).sum())
    print(f"\n  live bins: {live}/{len(bins)} -> the retry ladder has {live} rungs, "
          f"and chained_retry_at_k saturates at k={live}.")
    print("  A bin with no demonstrations is a direction the policy will")
    print("  EXTRAPOLATE into if the feasibility mask ever admits it.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl_path = out.with_suffix(".json")
    excl_path = out.parent / f"{out.name}_excluded.json"
    table["_meta"] = {
        **{k: v for k, v in meta.items() if k not in ("scenes_with_bin",)},
        "stage": "assign_direction_demos", "mode": args.mode,
        # Recorded so the pipeline's staleness probe can tell a run-2 table
        # (1 per bin) from a run-3 one (3) without reparsing every scene, and so
        # a shard collected under one can never be mixed with the other.
        "per_bin": int(args.per_bin) if args.mode == "per-bin" else None,
        "schema": "regrasp-pins-v1",
        "source_table": args.table, "min_bins": args.min_bins,
        "dropped_bins": sorted(names[i] for i in drop),
        "n_paired": n_paired, "n_single": n_single,
        "n_demos": int(counts.sum()),
        "demos_per_bin": counts.astype(int).tolist(),
        "live_bins": live,
    }
    tbl_path.write_text(json.dumps(table, indent=1))
    excl_path.write_text(json.dumps(sorted(excluded)))
    print(f"\nwrote {tbl_path}   ({n} scenes)")
    print(f"wrote {excl_path}  ({len(excluded)} scene indices)")
    print("\nPoint SIM.grasp_pin_table at the .json and SIM.exclude_scenes at the "
          "_excluded.json. NOTE the table mixes scenes with different numbers of "
          "grasps, so GraspPinTable.num_grasps (a MIN) reads 1 — use pairs() / "
          "num_grasps_for() / max_grasps, never num_grasps.")


if __name__ == "__main__":
    main()
