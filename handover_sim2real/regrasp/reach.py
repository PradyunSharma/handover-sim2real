"""
Did the demonstration actually arrive? ONE definition, shared by every consumer.

`demo_ok_table` asks whether a demonstration's CAPTION is honest —
`bin_realized == bin_assigned` and the pin landed — and on the run-2 base set that
passes 1575 of 1596 (98.7%). It says nothing about whether the expert got
anywhere. Measured on the same shard, only **1116 of 1596 (69.9%)** ended within
the close thresholds of the grasp they were aiming at; the other 480 stop a mean
108 mm short, run 14 steps against 21, and end still commanding "keep
approaching" because `c_env_done` fired and truncated the episode.

Those 480 were in `D`, and their `(scene, bin)` pairs were re-collected and
re-scored every DAgger iteration. This module is the filter that removes them.

THREE CONSUMERS, ONE FUNCTION, AND THAT IS THE POINT:

  * `regrasp_bc/dataset.py`  drops the episodes from D, per episode, from the
                             attrs that ride on it — self-contained, so a shard
                             cannot be filtered against the wrong run's list.
  * `examples/train_regrasp.py`  prunes the pin table via `reach_ok_pairs`, which
                             removes the pair from DAgger collection AND from the
                             in-loop evaluation at once (every consumer reads its
                             slot count from that table).
  * `examples/build_demo_table.py`  reports the same number in its matrix.

If these three ever disagree, the aggregate holds episodes for pairs the loop no
longer collects on — which is exactly the leak the caption filter had before the
loader-side check was added (`D_episodes` 1596, not 1575).

THE THRESHOLDS MIRROR `DAGGER.close_pos_thresh` / `close_rot_thresh`, so "reached"
means here what it means everywhere else in the loop. They are not re-tuned: a
demonstration that ends further away than the distance at which the collector is
willing to command a CLOSE is, by the loop's own standard, not at the grasp.

CROSS-CHECK, NOT A THRESHOLD ARTIFACT. The independent criterion — did the expert
emit a gripper CLOSE as its last action, which the collector appends only when the
plan ran to completion — agrees with this one on 99.6% of episodes. Two
derivations, one from poses and one from action labels, landing on the same set is
what makes the 30% believable.

Pure numpy plus transforms3d for the quaternion; `reach_ok_pairs` imports h5py
lazily so the geometry stays importable without it.
"""

from __future__ import annotations

import numpy as np

# Mirrors DAGGER.close_pos_thresh / DAGGER.close_rot_thresh (0.34 rad ~ 19.5 deg).
DEFAULT_POS_THRESH = 0.02
DEFAULT_ROT_THRESH = 0.34

# `robot_state` layout is joint_pos(9) | joint_vel(9) | ee_xyz(3) | ee_wxyz(4) |
# gripper_norm(1) | prev_act(6). The EE pose is in SIM WORLD — the same frame the
# pin table stores `grasp_pose_world` in, so the two compare with no transform.
EE_XYZ = slice(18, 21)
EE_WXYZ = slice(21, 25)


def terminal_pose_error(rs_last, grasp_pose_world) -> tuple[float, float]:
    """(position error in metres, rotation error in radians) at the last step."""
    from transforms3d.quaternions import quat2mat

    rs_last = np.asarray(rs_last, dtype=np.float64)
    G = np.asarray(grasp_pose_world, dtype=np.float64)
    p_err = float(np.linalg.norm(rs_last[EE_XYZ] - G[:3, 3]))
    R = quat2mat(rs_last[EE_WXYZ])
    cos = (np.trace(R.T @ G[:3, :3]) - 1.0) / 2.0
    return p_err, float(np.arccos(np.clip(cos, -1.0, 1.0)))


def reached(rs_last, grasp_pose_world,
            pos_thresh: float = DEFAULT_POS_THRESH,
            rot_thresh: float = DEFAULT_ROT_THRESH) -> bool:
    """Did this episode end at the grasp it was aiming at?"""
    p_err, r_err = terminal_pose_error(rs_last, grasp_pose_world)
    return bool(p_err < pos_thresh and r_err < rot_thresh)


def reach_ok_pairs(paths,
                   pos_thresh: float = DEFAULT_POS_THRESH,
                   rot_thresh: float = DEFAULT_ROT_THRESH) -> tuple[dict, dict]:
    """`({scene_idx: [bin, ...]}, stats)` — the pairs whose demonstration arrived.

    The returned mapping is in `GraspPinTable.keep_only`'s format, so pruning
    collection and evaluation is one call. Keyed on the BIN, not the slot, for the
    same reason `keep_only` is: a slot index is a position within a scene and is
    renumbered by the prune, while the bin is the stable name.

    An episode with no `grasp_pose_world` attr cannot be judged and is counted as
    `unjudgeable` rather than assumed good — it is not dropped, because a shard
    predating that attr would otherwise vanish entirely. An episode whose
    `bin_assigned` is -1 (no pin table) cannot be named as a pair and is skipped.
    """
    import h5py

    if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
        paths = [paths]

    ok: dict[int, set] = {}
    stats = {"episodes": 0, "reached": 0, "failed": 0,
             "unjudgeable": 0, "unnamed": 0}
    for path in paths:
        with h5py.File(path, "r") as f:
            for k in sorted(x for x in f if x.startswith("episode_")):
                a = f[k].attrs
                stats["episodes"] += 1
                ba = int(a.get("bin_assigned", -1))
                if ba < 0:
                    stats["unnamed"] += 1
                    continue
                G = a.get("grasp_pose_world")
                if G is None:
                    stats["unjudgeable"] += 1
                    ok.setdefault(int(a["scene_idx"]), set()).add(ba)
                    continue
                if reached(f[k]["robot_states"][-1], G, pos_thresh, rot_thresh):
                    stats["reached"] += 1
                    ok.setdefault(int(a["scene_idx"]), set()).add(ba)
                else:
                    stats["failed"] += 1
    return {s: sorted(bs) for s, bs in ok.items()}, stats
