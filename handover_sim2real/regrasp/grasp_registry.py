"""
Cross-iteration check that a (scene, grasp) target never moves.

The pin table (`grasp_pin.py`) *enforces* which grasp a scene aims at; this
*verifies* it, which is not the same thing. The pin can silently fail to apply — a scene
missing from the table, a stored pose that no longer matches any candidate
because the grasp set changed, an OMG failure at step 0 — and each of those
leaves OMG's own re-selection in force without stopping the run. Over 20
iterations the same scene is collected several times, and if it aimed at grasp A
in iteration 3 and grasp B in iteration 11 then D contains two contradictory
label sets for the same states, which is precisely the inconsistency DAgger
cannot average away.

Phase 5 keys on `(scene_idx, grasp_idx)`, not on the scene: a scene now legally
aims at four different grasps, and a scene-keyed registry would report every
alternate slot as a mismatch and drown the real signal. Within a slot the
guarantee is exactly the Phase-4 one.

So: record the grasp each (scene, slot) actually aimed at the first time it is
collected, and compare on every later visit. `goal_switch` (in the collector) catches the
target moving WITHIN an episode; this catches it moving BETWEEN iterations, which
no per-episode counter can see.

Persisted as JSON in the run dir so it survives a resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GraspRegistry:
    """(scene_idx, grasp_idx) -> the pose that pair was first collected against."""

    def __init__(self, path, tol: float = 0.02):
        self.path = Path(path)
        self.tol = float(tol)
        self.entries: dict[int, dict] = {}
        if self.path.exists():
            with self.path.open() as f:
                raw = json.load(f)
            # Keys are "scene:slot"; a bare "scene" from a Phase-4 registry
            # reads as slot 0, so an old file still loads.
            self.entries = {self._parse_key(k): v
                            for k, v in raw.items() if k != "_meta"}

    @staticmethod
    def _parse_key(k):
        s = str(k)
        return (int(s.split(":")[0]), int(s.split(":")[1])) if ":" in s else (int(s), 0)

    def check(self, scene_idx: int, pose, grasp_idx: int = 0,
              iteration: int = 0) -> dict:
        """Record or verify this (scene, grasp) pair's target.

        Returns {"seen": bool, "drift": float, "mismatch": bool}. `drift` is the
        position distance to the first-seen pose (NaN on the first visit).
        A mismatch is REPORTED, never corrected — the run continues, but the
        column in dagger_log.csv makes the contradiction visible instead of
        letting it sit in the aggregate unnoticed.
        """
        if pose is None:
            return {"seen": False, "drift": float("nan"), "mismatch": False}
        pose = np.asarray(pose, dtype=np.float64)
        key = (int(scene_idx), int(grasp_idx))
        prev = self.entries.get(key)

        if prev is None:
            self.entries[key] = {"ee_pose_world": pose.tolist(),
                                 "first_iter": int(iteration), "visits": 1}
            return {"seen": False, "drift": float("nan"), "mismatch": False}

        ref = np.asarray(prev["ee_pose_world"], dtype=np.float64)
        drift = float(np.linalg.norm(pose[:3, 3] - ref[:3, 3]))
        prev["visits"] = int(prev.get("visits", 1)) + 1
        mismatch = drift > self.tol
        if mismatch:
            print(f"[grasp-registry] scene {key[0]} g{key[1]}: grasp moved "
                  f"{drift:.4f} m since "
                  f"iteration {prev['first_iter']} (tol {self.tol}). The pin did "
                  f"not hold — D now contains two different targets for this scene.")
            prev["max_drift"] = max(float(prev.get("max_drift", 0.0)), drift)
        return {"seen": True, "drift": drift, "mismatch": mismatch}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_meta": {"tol": self.tol, "pairs": len(self.entries),
                             "scenes": len({s for s, _ in self.entries})}}
        payload.update({f"{s}:{g}": v for (s, g), v in self.entries.items()})
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=1)
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self.entries)
