"""
Apply the per-scene grasp pin table (built by examples/build_grasp_pin_table.py).

OMG re-decides its goal grasp on every plan, so without pinning the target can
move mid-episode and can differ between the demonstrations and DAgger. The table
fixes one grasp per scene; this module applies it.

The table keys on the chosen grasp's **world EE pose**, not its index: the goal
set is rebuilt every episode and its ordering is an IK artifact (`solve_goal_set_ik`
then `flip_grasp` appends wrist-flipped duplicates), so an index is not durable.
At apply time the stored pose is matched to the nearest goal-set entry by
position, and the match is rejected if it is further than `match_tol` — that turns
"the grasp set silently changed" (a different `valid_grasp_dict`, a different
hand-collision filter) into a visible error instead of a quietly different target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GraspPinTable:
    """scene_idx -> the one grasp every phase must aim at."""

    def __init__(self, path, match_tol: float = 0.02):
        self.path = Path(path)
        with self.path.open() as f:
            raw = json.load(f)
        self.meta = raw.pop("_meta", {})
        self.entries = {int(k): v for k, v in raw.items() if v is not None}
        self.match_tol = float(match_tol)
        self.n_applied = 0
        self.n_missing = 0
        self.n_mismatch = 0

    def __len__(self) -> int:
        return len(self.entries)

    def describe(self) -> str:
        m = self.meta
        return (f"{len(self)} scenes  mode={m.get('mode')}  tol={m.get('tol')}  "
                f"split={m.get('split')}  vgd={m.get('valid_grasp_dict_path')}")

    def check_against(self, sim_cfg_block: dict) -> None:
        """Warn when the table was built under a different grasp set than the one
        now configured — the pinned pose may not exist in the current goal set."""
        m = self.meta
        if not m:
            return

        # Scene indices are SPLIT-RELATIVE, so a train table applied to the val
        # split silently pins scene 7 of val to the grasp belonging to scene 7 of
        # train. Position matching would reject most of those, but not all, so
        # this is the dangerous mismatch: it fails quietly and partially.
        want_split = str(sim_cfg_block.get("split", "")) or None
        was_split = m.get("split")
        if want_split and was_split and want_split != was_split:
            raise ValueError(
                f"[grasp-pin] table {self.path} was built for split "
                f"'{was_split}' but this run uses '{want_split}'. Scene indices "
                f"are split-relative, so this table pins the WRONG grasps. Build "
                f"one per split (build_grasp_pin_table.py --split {want_split}).")

        cur = (sim_cfg_block.get("valid_grasp_dict_path"),
               bool(sim_cfg_block.get("hand_collision_filter", False)))
        was = (m.get("valid_grasp_dict_path"), bool(m.get("hand_collision_filter", False)))
        if cur != was:
            print(f"[grasp-pin] WARNING table was built with "
                  f"valid_grasp_dict={was[0]} hand_collision_filter={was[1]}, "
                  f"but this run uses {cur[0]} / {cur[1]}. The grasp set differs, "
                  f"so pinning may fail to match. Rebuild the table.")

    def apply(self, env, scene_idx: int) -> bool:
        """Pin `env`'s goal set to this scene's committed grasp.

        Call after a successful `run_omg_planner(..., reset_scene=True)` — the
        goal set has to exist before it can be pruned, and `reset_scene` rebuilds
        it, so this is once per episode.

        Returns True if pinned. False (with the goal set left untouched, i.e.
        OMG's own selection still in force) when the scene has no entry or the
        stored pose does not match any current candidate.
        """
        entry = self.entries.get(int(scene_idx))
        if entry is None:
            self.n_missing += 1
            return False

        poses = env.goal_set_ee_poses()
        if len(poses) == 0:
            self.n_missing += 1
            return False

        target = np.asarray(entry["ee_pose_world"], dtype=np.float64)
        d = np.linalg.norm(poses[:, :3, 3] - target[None, :3, 3], axis=1)

        # Position alone does NOT identify a grasp. OMG's `flip_grasp` appends
        # wrist-flipped duplicates that share an EE position and differ by pi in
        # rotation, so argmin over position picks between a flip pair arbitrarily
        # — and then the pin aims at the twin of the grasp the table recorded.
        # Measured on train_pinned.h5: a handful of episodes closed 3.1413 rad
        # (= pi) from their pinned grasp while p99 was 0.0029 rad. Disambiguate
        # among the position candidates by ROTATION.
        near = np.flatnonzero(d <= self.match_tol)
        if len(near):
            R_t = target[:3, :3]
            cos = (np.trace(poses[near][:, :3, :3] @ R_t.T, axis1=1, axis2=2) - 1.0) / 2.0
            rot = np.arccos(np.clip(cos, -1.0, 1.0))
            idx = int(near[int(np.argmin(rot))])
            if rot.min() > 0.35:
                print(f"[grasp-pin] scene {scene_idx}: position matches to "
                      f"{d[idx]:.4f} m but the closest orientation is "
                      f"{rot.min():.3f} rad away — the stored grasp is probably a "
                      f"wrist-flip of what the goal set now holds.")
        else:
            idx = int(np.argmin(d))
        if d[idx] > self.match_tol:
            self.n_mismatch += 1
            print(f"[grasp-pin] scene {scene_idx}: pinned pose not in the current "
                  f"goal set (nearest is {d[idx]:.4f} m > match_tol "
                  f"{self.match_tol}); leaving OMG's selection. Rebuild the table "
                  f"if the grasp set changed.")
            return False

        env.pin_goal_grasp(idx)
        self.n_applied += 1
        return True

    def stats(self) -> dict:
        return {"applied": self.n_applied, "missing": self.n_missing,
                "mismatched": self.n_mismatch}


def load_grasp_pin_table(path, match_tol: float = 0.02, sim_cfg_block: dict | None = None):
    """Load a pin table, or None when `path` is falsy (pinning disabled)."""
    if not path:
        return None
    table = GraspPinTable(path, match_tol=match_tol)
    print(f"[grasp-pin] loaded {path}: {table.describe()}")
    if sim_cfg_block is not None:
        table.check_against(sim_cfg_block)
    return table
