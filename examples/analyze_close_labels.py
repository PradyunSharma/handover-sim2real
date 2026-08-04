"""
Calibrate the CLOSE-label thresholds against what the EXPERT actually achieves.

The two collectors disagree about when a gripper-close label is legitimate:

  collect_bc_dataset.py   plays the whole OMG plan by index and appends CLOSE
                          UNCONDITIONALLY at the end — no distance check at all.
  dagger/collector.py     emits CLOSE only when the EE is within
                          (close_pos_thresh, close_rot_thresh) of the grasp.

So DAgger holds itself to a criterion the demonstrations were never required to
meet, and nobody has measured whether the demonstrations meet it. If the expert
typically finishes 5 cm out, then `close_pos_thresh: 0.02` is stricter than the
expert itself and DAgger can never emit a close label no matter how well the
policy behaves — which looks identical, in the logs, to the policy being bad.

This script settles it. For every demonstration episode it takes the state at the
CLOSE label, compares it to that scene's pinned grasp, and reports the error
distribution plus the pass rate at candidate thresholds.

    python examples/analyze_close_labels.py \\
        --demos output/bc_dataset/train_pinned.h5 \\
        --pin-table output/grasp_pin_table_train.json

Read the output as: if the demonstrations' own pass rate at the configured
threshold is high, the threshold is calibrated and a low `reached_grasp` in
DAgger is a real policy failure. If it is low, the threshold is the bug, and the
principled fix is to loosen it to the achievable value — NOT to drop the check,
which would let CLOSE be emitted at poses that grasp air.
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

# robot_state layout: joint_pos(9) joint_vel(9) ee_xyz(3) ee_wxyz(4) grip(1) prev(6)
EE_POS = slice(18, 21)
EE_QUAT = slice(21, 25)      # wxyz, WORLD frame (same frame as the pin table)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def rot_err(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", default="output/bc_dataset/train_pinned.h5")
    p.add_argument("--pin-table", default="output/grasp_pin_table_train.json")
    p.add_argument("--pos-thresh", type=float, default=0.02)
    p.add_argument("--rot-thresh", type=float, default=0.34)
    args = p.parse_args()

    raw = json.load(open(args.pin_table))
    pins = {int(k): np.asarray(v["ee_pose_world"], dtype=np.float64)
            for k, v in raw.items() if k != "_meta" and v is not None}
    print(f"pin table : {len(pins)} scenes  ({args.pin_table})")

    pos_e, rot_e = [], []
    n_ep = n_noclose = n_nopin = 0
    with h5py.File(args.demos, "r") as f:
        # Guard: if the demonstrations were not collected against this pin table,
        # every error below is the distance between two DIFFERENT grasp choices,
        # not the expert's convergence to its own target. Measured on the old
        # unpinned train.h5 that reads as a 7.8 cm median, which looks exactly
        # like a calibration problem and is not one.
        used_pin = str(f.attrs.get("grasp_pin_table", ""))
        if not used_pin:
            print("\n*** WARNING: this dataset records no grasp_pin_table. Either it\n"
                  "*** predates provenance logging or it was collected UNPINNED. If\n"
                  "*** unpinned, the numbers below measure the demos-vs-table grasp\n"
                  "*** mismatch, NOT threshold calibration. Re-collect with\n"
                  "*** --grasp-pin-table before trusting this.\n")
        elif used_pin != args.pin_table:
            print(f"\n*** WARNING: dataset was pinned with {used_pin!r} but this run\n"
                  f"*** compares against {args.pin_table!r}.\n")
        for key in f:
            if not key.startswith("episode"):
                continue
            n_ep += 1
            grp = f[key]
            scene = int(grp.attrs["scene_idx"])
            acts = grp["expert_actions"][:]
            # the CLOSE label is gripper channel 6 == 0
            idx = np.flatnonzero(acts[:, 6] < 0.5)
            if len(idx) == 0:
                n_noclose += 1
                continue
            if scene not in pins:
                n_nopin += 1
                continue
            rs = grp["robot_states"][int(idx[0])]
            g = pins[scene]
            pos_e.append(float(np.linalg.norm(np.asarray(rs[EE_POS], np.float64) - g[:3, 3])))
            rot_e.append(rot_err(quat_wxyz_to_mat(np.asarray(rs[EE_QUAT], np.float64)),
                                 g[:3, :3]))

    print(f"episodes  : {n_ep}   scored {len(pos_e)}   "
          f"(no CLOSE label: {n_noclose}, scene not in pin table: {n_nopin})")
    if not pos_e:
        raise SystemExit("nothing to measure")

    pos_e, rot_e = np.asarray(pos_e), np.asarray(rot_e)
    print("\nEE -> pinned grasp error AT THE EXPERT'S OWN CLOSE LABEL")
    for name, v, unit in (("position", pos_e, "m"), ("rotation", rot_e, "rad")):
        print(f"  {name:8s} median {np.median(v):.4f} {unit}   "
              f"p90 {pct(v,90):.4f}   p99 {pct(v,99):.4f}   max {v.max():.4f}")

    ok_p = pos_e <= args.pos_thresh
    ok_r = rot_e <= args.rot_thresh
    both = ok_p & ok_r
    print(f"\nPass rate at the CONFIGURED thresholds "
          f"(pos<={args.pos_thresh}, rot<={args.rot_thresh}):")
    print(f"  position only : {100*ok_p.mean():5.1f}%")
    print(f"  rotation only : {100*ok_r.mean():5.1f}%")
    print(f"  BOTH          : {100*both.mean():5.1f}%   <- what DAgger requires")

    print("\nThresholds needed to admit a given fraction of the expert's own closes:")
    for q in (50, 75, 90, 95):
        print(f"  {q:3d}% : pos<={pct(pos_e,q):.4f} m   rot<={pct(rot_e,q):.4f} rad")

    if both.mean() < 0.8:
        print(f"\n=> The demonstrations themselves clear DAgger's bar only "
              f"{100*both.mean():.0f}% of the time. The threshold is stricter than the\n"
              f"   expert achieves, so a low `reached_grasp` is at least partly the\n"
              f"   THRESHOLD's fault, not the policy's. Consider "
              f"close_pos_thresh={pct(pos_e,90):.3f}, close_rot_thresh={pct(rot_e,90):.2f}.")
    else:
        print(f"\n=> The threshold is calibrated: the expert clears it "
              f"{100*both.mean():.0f}% of the time.\n"
              f"   A low `reached_grasp` in DAgger is therefore a real policy failure,\n"
              f"   and loosening the threshold would only hide it.")


if __name__ == "__main__":
    main()
