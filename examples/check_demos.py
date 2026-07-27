#!/usr/bin/env python
"""Sanity-check a collected RL demo HDF5 — mainly point-cloud CORRECTNESS (the thing that
broke silently before: the first multicam collect produced all-empty (M,0,C) clouds).
Prints a PASS/FAIL report; exits non-zero if any hard check fails.

    python examples/check_demos.py output/rl_demos/train_h30_mc_wl.h5
    python examples/check_demos.py output/rl_demos/train_h30_mc_wl.h5 --ratios 0.7 0.15 0.15
    python examples/check_demos.py output/rl_demos/train_h30.h5        # 5-ch (2-class) also fine

Run in the pch2r_dev env (needs h5py). Works for 2-class (5-ch) and 3-class (6-ch) clouds;
the number of feature classes is inferred as pc_channels - 3 (xyz).
"""
import argparse
import sys

import h5py
import numpy as np

_CLS_NAMES = {2: ["object", "hand"], 3: ["object", "hand", "robot"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", help="path to the collected demo .h5")
    ap.add_argument("--ratios", type=float, nargs="*", default=None,
                    help="expected class fractions (e.g. 0.7 0.15 0.15); default = just report")
    ap.add_argument("--num-pts", type=int, default=1024, help="expected points per cloud")
    ap.add_argument("--sample", type=int, default=3000, help="max clouds to scan for per-cloud checks")
    args = ap.parse_args()

    f = h5py.File(args.demos, "r")
    print(f"file: {args.demos}")
    print(f"keys: {list(f.keys())}")
    if "pc" not in f:
        print("FAIL: no 'pc' dataset"); sys.exit(1)

    pc = f["pc"]
    M = pc.shape[0]
    C = pc.shape[2] if pc.ndim == 3 else (pc[0].shape[-1] if M else 0)
    n_cls = C - 3
    print(f"pc shape: {pc.shape}  dtype: {pc.dtype}   -> {M} transitions, {C} channels "
          f"(xyz + {n_cls} class one-hots)")

    fails, warns = [], []

    # ---- shape / degeneracy (catches the (M,0,C) all-empty bug) ----
    if pc.ndim != 3:
        fails.append(f"pc is not a dense [M,N,C] array (ndim={pc.ndim})")
    else:
        if pc.shape[1] == 0:
            fails.append("clouds have 0 points (N=0) — ALL-EMPTY collection (the known render bug)")
        elif pc.shape[1] != args.num_pts:
            warns.append(f"N={pc.shape[1]} != expected num_pts={args.num_pts}")
        if n_cls not in (2, 3):
            warns.append(f"unusual class count {n_cls} (channels {C}); expected 5-ch(2) or 6-ch(3)")

    # ---- episode / reward stats ----
    if "terminal" in f:
        term = f["terminal"][:].astype(bool)
        n_ep = int(term.sum())
        print(f"episodes (terminals): {n_ep}   transitions/episode ~ {M / max(n_ep,1):.1f}")
    if "reward" in f:
        rew = f["reward"][:]
        pos = int((rew > 0).sum())
        print(f"positive-reward (closed-at-grasp) transitions: {pos}/{M}  "
              f"({100*pos/max(M,1):.1f}%)")
        if pos == 0:
            warns.append("NO positive-reward transitions — no +1 grasp anchor in the pool")

    # ---- per-cloud scan (sampled) ----
    idx = np.linspace(0, M - 1, min(args.sample, M)).astype(int) if M else np.array([], int)
    n_empty = n_bad_onehot = n_missing_cls = n_bad_N = 0
    cls_counts = np.zeros(max(n_cls, 1))
    gmin = np.array([np.inf] * 3); gmax = np.array([-np.inf] * 3)
    for i in idx:
        c = np.asarray(pc[i])
        if c.shape[0] == 0:
            n_empty += 1; continue
        if c.shape[0] != args.num_pts:
            n_bad_N += 1
        xyz, lab = c[:, :3], c[:, 3:]
        gmin = np.minimum(gmin, xyz.min(0)); gmax = np.maximum(gmax, xyz.max(0))
        if not np.allclose(lab.sum(1), 1.0, atol=1e-4):
            n_bad_onehot += 1
        cls = lab.argmax(1)
        present = [(cls == k).any() for k in range(n_cls)]
        if not all(present):
            n_missing_cls += 1
        for k in range(n_cls):
            cls_counts[k] += (cls == k).sum()

    scanned = len(idx)
    print(f"\nscanned {scanned} clouds:")
    print(f"  empty clouds:              {n_empty}")
    print(f"  invalid one-hot labels:    {n_bad_onehot}")
    print(f"  clouds missing a class:    {n_missing_cls}")
    print(f"  clouds with N!={args.num_pts}:        {n_bad_N}")
    if cls_counts.sum() > 0:
        frac = cls_counts / cls_counts.sum()
        names = _CLS_NAMES.get(n_cls, [f"c{k}" for k in range(n_cls)])
        print(f"  aggregate label fractions: " +
              "  ".join(f"{nm}={fr:.3f}" for nm, fr in zip(names, frac)))
        if args.ratios:
            exp = np.array(args.ratios)
            if len(exp) == n_cls and np.abs(frac - exp).max() > 0.03:
                warns.append(f"label fractions {np.round(frac,3)} deviate >0.03 from expected {exp}")
    if np.isfinite(gmin).all():
        print(f"  xyz range (EE frame):      x[{gmin[0]:.2f},{gmax[0]:.2f}] "
              f"y[{gmin[1]:.2f},{gmax[1]:.2f}] z[{gmin[2]:.2f},{gmax[2]:.2f}]")
        # EE/hand-frame clouds are centered near the origin; a big offset hints wrong frame
        if np.abs((gmin + gmax) / 2).max() > 1.0:
            warns.append("cloud not centered near origin — may not be in the EE/hand frame")

    # hard fails from the scan
    if n_empty:
        fails.append(f"{n_empty}/{scanned} clouds are empty")
    if n_bad_onehot:
        fails.append(f"{n_bad_onehot}/{scanned} clouds have invalid one-hot labels")
    if n_cls >= 3 and n_missing_cls > 0.5 * scanned:
        warns.append(f"{n_missing_cls}/{scanned} clouds miss a class (a view may not see the robot)")

    f.close()
    print("\n" + "=" * 60)
    for w in warns:
        print(f"  WARN: {w}")
    if fails:
        for x in fails:
            print(f"  FAIL: {x}")
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS" + ("  (with warnings)" if warns else ""))


if __name__ == "__main__":
    main()
