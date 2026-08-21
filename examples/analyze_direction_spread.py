"""
Does the Phase-5 dataset actually distinguish its four grasps?

Run this on the pruned base set BEFORE launching a 20 h trainer. It answers, in
about twenty minutes and with no GPU, the one question that decides whether
grasp conditioning can work at all:

    for a given scene, how different are the four expert demonstrations?

If they are nearly identical, the conditioning input has nothing to predict from
and no amount of training will make `cond_track` move — the failure would then be
in the DATA, not in the network, and no architectural change fixes it.

Three numbers per scene, all under the flip-invariant control-point metric from
`regrasp/grasp_select.py`:

    goal_sep       mean pairwise distance between the four TARGET grasps. This is
                   what the selection bought; it is the denominator of cond_track
                   at eval time, and a scene with a small one cannot show a large
                   trajectory difference no matter what the policy does.
    act_div(t)     mean pairwise L1 between the four expert ACTION sequences at
                   step-from-end t. Expected to be near zero early — all four
                   plans fly to roughly the same place during the free approach —
                   and to grow into the reach.
    informative    fraction of steps where act_div exceeds `--noise-floor`, i.e.
                   where the label genuinely depends on which grasp was commanded.

WHAT TO EXPECT, and how to read it. The four grasps of a scene share a free
approach and diverge only in the reach, so a low overall `informative` fraction
is NOT a failure — it is the geometry, and it is exactly why
`DATA.reach_tail_weight: 2.5` exists. What matters is the per-step profile: the
divergence must rise into the last few steps. A flat profile at zero means
something is wrong upstream (the pin did not apply, or all four slots collapsed
onto one grasp), and that is a bug to find before training, not after.

    python examples/analyze_direction_spread.py \\
        --demos output/bc_dataset/train_p5.h5

    # against the raw K-candidate set, before pruning
    python examples/analyze_direction_spread.py \\
        --demos output/bc_dataset/train_p5_k8.h5 --plot output/p5_separation.png
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# regrasp/__init__.py resolves its re-exports lazily (PEP 562), so importing
# a submodule no longer drags in gym/pybullet/handover or asserts on
# GADDPG_DIR. This script therefore imports normally. It used to insert the
# package DIRECTORY on sys.path to bypass __init__ -- a path built from
# string literals, invisible to every import-graph check, which is exactly
# how it silently broke during the dagger5 -> regrasp rename.

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import h5py
import numpy as np

from handover_sim2real.regrasp.grasp_select import pairwise_mean_distance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", required=True, help="a Phase-5 HDF5 with grasp_idx attrs")
    p.add_argument("--noise-floor", type=float, default=0.005,
                   help="metres of per-step action difference below which two "
                        "demonstrations count as saying the same thing. Default "
                        "0.005 is a quarter of the reach's own per-step scale "
                        "(~0.012 m) and an eighth of the free approach's (~0.026).")
    p.add_argument("--tail", type=int, default=10,
                   help="how many steps-from-end to profile")
    p.add_argument("--plot", default=None, help="write a PNG of the profile")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    per_scene: dict[int, dict[int, dict]] = defaultdict(dict)
    with h5py.File(args.demos, "r") as f:
        keys = sorted(k for k in f if k.startswith("episode"))
        if not keys:
            raise SystemExit(f"{args.demos} has no episodes")
        if "grasp_idx" not in f[keys[0]].attrs:
            raise SystemExit(
                f"{args.demos} has no `grasp_idx` attr — this is a Phase-4 "
                f"dataset with one grasp per scene and there is nothing to "
                f"compare. Collect with examples/collect_regrasp_demos.py.")
        for k in keys:
            g = f[k]
            per_scene[int(g.attrs["scene_idx"])][int(g.attrs["grasp_idx"])] = {
                "acts": g["expert_actions"][:, :6].astype(np.float64),
                "pose": np.asarray(g.attrs["grasp_pose_world"], dtype=np.float64),
            }

    goal_seps = []
    # act_div[t] collects, over scenes, the mean pairwise action difference at
    # step-from-end t (t=0 is the LAST step). Aligned from the end because the
    # episodes have different lengths and it is the reach that has to line up.
    act_div = defaultdict(list)
    informative_frac = []
    n_full = 0

    for scene, per in sorted(per_scene.items()):
        slots = sorted(per)
        if len(slots) < 2:
            continue
        n_full += 1
        goal_seps.append(pairwise_mean_distance(
            np.stack([per[g]["pose"] for g in slots])))

        seqs = [per[g]["acts"] for g in slots]
        n_steps = min(len(s) for s in seqs)
        per_step = []
        for t in range(min(n_steps, args.tail)):
            # -1-t: index from the END, so the reach of every episode lines up.
            v = np.stack([s[-1 - t] for s in seqs])
            d = [np.abs(v[i, :3] - v[j, :3]).sum()
                 for i in range(len(v)) for j in range(i + 1, len(v))]
            act_div[t].append(float(np.mean(d)))
            per_step.append(float(np.mean(d)))
        # Over the WHOLE episode, not just the tail — this is the headline.
        allsteps = []
        for t in range(n_steps):
            v = np.stack([s[-1 - t] for s in seqs])
            allsteps.append(float(np.mean(
                [np.abs(v[i, :3] - v[j, :3]).sum()
                 for i in range(len(v)) for j in range(i + 1, len(v))])))
        informative_frac.append(
            float(np.mean(np.asarray(allsteps) > args.noise_floor)))

    if not n_full:
        raise SystemExit("no scene has two or more grasps — nothing to compare")

    gs = np.asarray(goal_seps)
    inf = np.asarray(informative_frac)
    print(f"{args.demos}")
    print(f"  scenes with >=2 grasps : {n_full}")
    print(f"  TARGET separation (m)  : median {np.median(gs):.4f}  "
          f"p10 {np.percentile(gs, 10):.4f}  min {gs.min():.4f}")
    print(f"  informative steps      : median {np.median(inf):.3f}  "
          f"mean {inf.mean():.3f}   (per-step |dpos| difference > "
          f"{args.noise_floor} m)")
    print(f"  scenes under 10% informative: "
          f"{int((inf < 0.10).sum())}/{n_full}   "
          f"(their four demos are nearly the same trajectory)")
    print("\n  divergence profile, aligned from the END of the episode:")
    print(f"    {'step from end':>14}  {'mean |dpos| diff (m)':>21}  {'n scenes':>9}")
    for t in sorted(act_div):
        v = np.asarray(act_div[t])
        print(f"    {t:>14}  {v.mean():>21.4f}  {len(v):>9}")
    print("\n  Read the PROFILE, not the headline: the four plans share a free "
          "approach\n  and are meant to diverge only into the reach. A profile "
          "that is flat at ~0\n  means the pin did not apply or the slots "
          "collapsed onto one grasp — find that\n  before training, not after.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = sorted(act_div)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ts, [np.mean(act_div[t]) for t in ts], "-o", ms=4)
        ax[0].axhline(args.noise_floor, color="0.6", ls=":",
                      label=f"noise floor {args.noise_floor} m")
        ax[0].invert_xaxis()
        ax[0].set_xlabel("steps from the END of the episode")
        ax[0].set_ylabel("mean pairwise |dpos| difference (m)")
        ax[0].set_title("do the four demonstrations diverge?")
        ax[0].grid(alpha=0.3)
        ax[0].legend(fontsize=8)
        ax[1].hist(inf, bins=20, color="tab:purple", alpha=0.8)
        ax[1].set_xlabel("fraction of steps that distinguish the grasps")
        ax[1].set_ylabel("scenes")
        ax[1].set_title("informative-step fraction")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
