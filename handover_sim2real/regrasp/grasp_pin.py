"""
Apply the per-scene grasp pin table (built by examples/build_direction_table.py).

OMG re-decides its goal grasp on every plan, so without pinning the target can
move mid-episode and can differ between the demonstrations and DAgger. The table
fixes the grasps every phase must aim at; this module applies one of them.

**Phase 5 stores a LIST per scene, not a single entry.** Each scene carries N
(default 4) physically distinct grasps, chosen by
`grasp_select.select_diverse_grasps`, and the unit of work everywhere downstream
is a `(scene_idx, grasp_idx)` pair rather than a scene. Slot 0 is always OMG's own
pick, so it is byte-identical to a Phase-4 `--mode omg` table and Phase-5 slot-0
numbers stay comparable with run 16.

Phase-4 tables (a bare dict per scene) still load: `_normalize_entry` wraps them
in a one-element list, so a Phase-4 table is just a Phase-5 table with N=1 and
`grasp_idx` must then always be 0.

The table keys on each chosen grasp's **world EE pose**, not its index: the goal
set is rebuilt every episode and its ordering is an IK artifact (`solve_goal_set_ik`
then `flip_grasp` appends wrist-flipped duplicates, and `omg/planner.py` shuffles
the survivors with `np.random.choice`), so an index is not durable. At apply time
the stored pose is matched to the nearest goal-set entry by position, and the match
is rejected if it is further than `match_tol` — that turns "the grasp set silently
changed" (a different `valid_grasp_dict`, a different hand-collision filter) into a
visible error instead of a quietly different target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _normalize_entry(value):
    """Phase-4 scalar entry or Phase-5 {'grasps': [...]} -> list of grasp dicts."""
    if value is None:
        return None
    if isinstance(value, dict) and "grasps" in value:
        grasps = value["grasps"]
    elif isinstance(value, list):
        grasps = value
    else:
        grasps = [value]                       # a Phase-4 table: one grasp, slot 0
    grasps = [g for g in grasps if g is not None]
    return grasps or None


class GraspPinTable:
    """scene_idx -> the N grasps every phase may aim at, in FPS order."""

    def __init__(self, path, match_tol: float = 0.02):
        self.path = Path(path)
        with self.path.open() as f:
            raw = json.load(f)
        self.meta = raw.pop("_meta", {})
        self.entries = {}
        self.scene_meta = {}
        for k, v in raw.items():
            grasps = _normalize_entry(v)
            if grasps is None:
                continue
            self.entries[int(k)] = grasps
            if isinstance(v, dict) and "grasps" in v:
                self.scene_meta[int(k)] = {kk: vv for kk, vv in v.items() if kk != "grasps"}
        self.match_tol = float(match_tol)
        self.n_applied = 0
        self.n_missing = 0
        self.n_mismatch = 0

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def num_grasps(self) -> int:
        """Slots per scene. Every scene in a built Phase-5 table has the same
        count — `select_regrasp_demos.py` drops scenes that cannot supply N — so
        this is well defined, but `num_grasps_for` is the safe accessor."""
        if not self.entries:
            return 0
        return min(len(v) for v in self.entries.values())

    @property
    def max_grasps(self) -> int:
        """The LARGEST slot count over scenes — the right width for a metric array.

        `num_grasps` is a MIN and is the wrong accessor for a Regrasp table, which
        deliberately mixes 1-grasp and 2-grasp scenes (a scene that can only reach
        one direction still contributes a demonstration). A min of 1 there would
        make every consumer that iterates `range(num_grasps)` silently drop the
        paired second demonstration — which is the entire point of the pairing.
        Iterate with `num_grasps_for(scene)`; size arrays with this.
        """
        return max((len(v) for v in self.entries.values()), default=0)

    def num_grasps_for(self, scene_idx: int) -> int:
        return len(self.entries.get(int(scene_idx), ()))

    def pairs(self, scenes=None):
        """[(scene_idx, grasp_idx)] for `scenes` (default: every scene in the
        table), scene-major. This is the Phase-5 unit of work."""
        ids = sorted(self.entries) if scenes is None else list(scenes)
        return [(int(s), g) for s in ids for g in range(self.num_grasps_for(s))]

    def pose(self, scene_idx: int, grasp_idx: int = 0):
        """(4, 4) world EE pose of one slot, or None if absent. Used by the
        evaluator and the dataset so nothing has to re-derive it from the env."""
        grasps = self.entries.get(int(scene_idx))
        if not grasps or not (0 <= int(grasp_idx) < len(grasps)):
            return None
        return np.asarray(grasps[int(grasp_idx)]["ee_pose_world"], dtype=np.float64)

    def bin_of(self, scene_idx: int, grasp_idx: int = 0):
        """The approach-direction bin index a slot was ASSIGNED, or None.

        Regrasp tables carry it per grasp (`assign_direction_pairs.py` writes it);
        a Phase-4/5 table has no such key and returns None, which the collector
        records as -1. This is the ASSIGNED bin — compare it against the episode's
        `bin_realized` to see where a failed pin sent the demonstration instead.
        """
        grasps = self.entries.get(int(scene_idx))
        if not grasps or not (0 <= int(grasp_idx) < len(grasps)):
            return None
        b = grasps[int(grasp_idx)].get("bin")
        return None if b is None else int(b)

    def describe(self) -> str:
        m = self.meta
        return (f"{len(self)} scenes x {self.num_grasps} grasps  "
                f"mode={m.get('mode')}  sep_floor={m.get('sep_floor_m')}  "
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
                f"one per split (build_direction_table.py --split {want_split}).")

        cur = (sim_cfg_block.get("valid_grasp_dict_path"),
               bool(sim_cfg_block.get("hand_collision_filter", False)))
        was = (m.get("valid_grasp_dict_path"), bool(m.get("hand_collision_filter", False)))
        if cur != was:
            print(f"[grasp-pin] WARNING table was built with "
                  f"valid_grasp_dict={was[0]} hand_collision_filter={was[1]}, "
                  f"but this run uses {cur[0]} / {cur[1]}. The grasp set differs, "
                  f"so pinning may fail to match. Rebuild the table.")

    def apply(self, env, scene_idx: int, grasp_idx: int = 0) -> bool:
        """Pin `env`'s goal set to slot `grasp_idx` of this scene.

        Call after a successful `run_omg_planner(..., reset_scene=True)` — the
        goal set has to exist before it can be pruned, and `reset_scene` rebuilds
        it, so this is once per episode.

        Returns True if pinned. False (with the goal set left untouched, i.e.
        OMG's own selection still in force) when the scene has no entry, the slot
        is out of range, or the stored pose does not match any current candidate.
        """
        target = self.pose(scene_idx, grasp_idx)
        if target is None:
            self.n_missing += 1
            return False

        poses = env.goal_set_ee_poses()
        if len(poses) == 0:
            self.n_missing += 1
            return False

        d = np.linalg.norm(poses[:, :3, 3] - target[None, :3, 3], axis=1)

        # Position alone does NOT identify a grasp. OMG's `flip_grasp` appends
        # wrist-flipped duplicates that share an EE position and differ by pi in
        # rotation, so argmin over position picks between a flip pair arbitrarily
        # — and then the pin aims at the twin of the grasp the table recorded.
        # Measured on train_pinned.h5: a handful of episodes closed 3.1413 rad
        # (= pi) from their pinned grasp while p99 was 0.0029 rad. Disambiguate
        # among the position candidates by ROTATION.
        #
        # Phase-5 note: `grasp_select` treats a flip twin as the SAME grasp (it is
        # — the gripper is symmetric), so two slots can never be twins of each
        # other and this disambiguation can never pick a different slot's pose.
        near = np.flatnonzero(d <= self.match_tol)
        if len(near):
            R_t = target[:3, :3]
            cos = (np.trace(poses[near][:, :3, :3] @ R_t.T, axis1=1, axis2=2) - 1.0) / 2.0
            rot = np.arccos(np.clip(cos, -1.0, 1.0))
            idx = int(near[int(np.argmin(rot))])
            if rot.min() > 0.35:
                print(f"[grasp-pin] scene {scene_idx}/g{grasp_idx}: position "
                      f"matches to {d[idx]:.4f} m but the closest orientation is "
                      f"{rot.min():.3f} rad away — the stored grasp is probably a "
                      f"wrist-flip of what the goal set now holds.")
        else:
            idx = int(np.argmin(d))
        if d[idx] > self.match_tol:
            self.n_mismatch += 1
            print(f"[grasp-pin] scene {scene_idx}/g{grasp_idx}: pinned pose not in "
                  f"the current goal set (nearest is {d[idx]:.4f} m > match_tol "
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
