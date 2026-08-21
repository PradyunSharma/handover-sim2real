"""
PyBullet GUI overlay for a chained regrasp, so the branching is something you
watch rather than something you infer from a CSV.

What ends up on screen, per scene:

  * the scene's N pinned grasps as gripper wireframes, one colour per slot. The
    grasp the current attempt is commanded to reach is drawn BRIGHT and thick,
    the others dim and thin — so "which target is live" is readable at a glance
    and the four are still visible for context.
  * the EE path of every attempt so far, each in its own slot colour, ALL LEFT UP.
    This is the whole point: a chained retry draws a trunk that forks, and the
    fork is the rewind. Paths are cleared only when the scene changes.
  * the replayed prefix retraced in grey while it re-executes, then a white
    cross-hair at the branch point.
  * a text banner naming the scene, the attempt, the commanded slot and the
    branch step, and a second line with the outcome once the attempt ends.

WHY THE OVERLAY DOES NOT ASK OMG FOR THE GOAL POSE. `rollout_bc_policy_p5.py`
draws its green gripper by planning (`run_omg_planner` -> `pin_table.apply` ->
`get_omg_goal_grasp_pose`), because there it is also proving the pin resolves
against the live goal set. Here the poses come straight from
`GraspPinTable.pose(scene, slot)`, which is the same pose the policy is
conditioned on and the same pose the evaluator scores against — so the overlay
cannot disagree with the metric, and a scene costs no planner call.

EVERY METHOD IS A NO-OP WHEN `enabled` IS FALSE, and `chained_retry` accepts
`viz=None`. Headless runs pay nothing and the core rollout stays testable without
a display.

DRAWING IS DIAGNOSTIC, NEVER LOAD-BEARING. Every pybullet call here is wrapped:
a debug-item limit or a GUI that went away must not take a rollout down halfway
through a chain. Failures print once and disable the overlay.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pybullet

# `gripper_segments` lives in examples/visualize_grasps.py. Added here rather than
# relying on a caller having done it, so `import chain_viz` is self-sufficient.
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

# One colour per grasp slot, chosen to stay distinguishable against the
# handover-sim table and to survive the dim/bright split below.
SLOT_COLOURS = (
    (0.10, 0.90, 0.20),   # 0  green   — OMG's own pick, the Phase-4-comparable slot
    (0.20, 0.65, 1.00),   # 1  blue
    (1.00, 0.60, 0.05),   # 2  orange
    (0.95, 0.25, 0.85),   # 3  magenta
)
GREY = (0.55, 0.55, 0.55)
WHITE = (1.0, 1.0, 1.0)


def slot_colour(i: int):
    return SLOT_COLOURS[int(i) % len(SLOT_COLOURS)]


def _dim(c, f: float = 0.45):
    return tuple(f * x for x in c)


class ChainViz:
    """Draws one chained-retry run. Construct once, reuse across scenes."""

    def __init__(self, *, enabled: bool = True, pace: float = 0.03,
                 replay_pace: float = 0.01, pause_s: float = 1.2,
                 show_grasps: bool = True, show_path: bool = True,
                 show_cloud: bool = False, cloud_ctx=None):
        self.enabled = bool(enabled)
        self.pace = float(pace)              # sleep per policy step
        self.replay_pace = float(replay_pace)  # sleep per replayed step
        self.pause_s = float(pause_s)        # dwell between attempts / scenes
        self.show_grasps = bool(show_grasps)
        self.show_path = bool(show_path)
        self.show_cloud = bool(show_cloud)
        # (panda_base_inv_tf, R_base, panda_base_pos) — only needed for the cloud
        self.cloud_ctx = cloud_ctx

        self._grasp_ids: list = []    # redrawn per attempt (the highlight moves)
        self._path_ids: list = []     # kept for the WHOLE scene — the fork
        self._cloud_ids: list = []    # replaced every step
        self._text_ids: list = []
        self._prev_pt = None
        self._scene = None
        self._grasp_poses = None

    # ── low-level, all failure-tolerant ──────────────────────────────────────

    def _guard(self, fn, *a, **kw):
        if not self.enabled:
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:                            # noqa: BLE001
            print(f"[viz] disabled after {type(e).__name__}: {e}")
            self.enabled = False
            return None

    def _line(self, p, q, colour, width, into):
        i = self._guard(pybullet.addUserDebugLine, list(map(float, p)),
                        list(map(float, q)), lineColorRGB=list(colour),
                        lineWidth=float(width))
        if i is not None:
            into.append(i)

    def _text(self, s, pos, colour, size=1.3):
        i = self._guard(pybullet.addUserDebugText, str(s), list(map(float, pos)),
                        textColorRGB=list(colour), textSize=float(size))
        if i is not None:
            self._text_ids.append(i)

    def _clear(self, ids):
        for i in list(ids):
            self._guard(pybullet.removeUserDebugItem, i)
        ids.clear()

    def _gripper(self, pose, colour, width, into):
        # Lazy AND guarded: visualize_grasps pulls matplotlib, which the headless
        # path never needs and which is exactly the kind of optional dependency
        # that should cost a warning rather than a dead rollout.
        try:
            from visualize_grasps import gripper_segments
        except Exception as e:                            # noqa: BLE001
            print(f"[viz] grasp markers off — cannot import gripper_segments "
                  f"({type(e).__name__}: {e})")
            self.show_grasps = False
            return
        for p, q in gripper_segments(np.asarray(pose, dtype=np.float64)):
            self._line(p, q, colour, width, into)

    # ── scene / attempt lifecycle ────────────────────────────────────────────

    def begin_scene(self, scene_idx: int, grasp_poses) -> None:
        if not self.enabled:
            return
        self._clear(self._path_ids)
        self._clear(self._grasp_ids)
        self._clear(self._cloud_ids)
        self._clear(self._text_ids)
        self._scene = int(scene_idx)
        self._grasp_poses = list(grasp_poses)
        self._prev_pt = None

    def begin_attempt(self, attempt: int, grasp_idx: int, branch_step: int,
                      n_attempts: int) -> None:
        """Redraw the grasp set with `grasp_idx` highlighted, and reset the pen.

        The grasps are redrawn rather than recoloured because pybullet debug
        lines are immutable — there is no setter for an existing item's colour.
        """
        if not self.enabled:
            return
        self._clear(self._grasp_ids)
        self._clear(self._text_ids)
        if self.show_grasps and self._grasp_poses:
            for i, pose in enumerate(self._grasp_poses):
                if pose is None:
                    continue
                live = (i == grasp_idx)
                self._gripper(pose, slot_colour(i) if live else _dim(slot_colour(i)),
                              4.0 if live else 1.0, self._grasp_ids)
            live_pose = self._grasp_poses[grasp_idx] \
                if grasp_idx < len(self._grasp_poses) else None
            if live_pose is not None:
                self._text(f"g{grasp_idx}",
                           np.asarray(live_pose)[:3, 3] + np.array([0, 0, 0.045]),
                           slot_colour(grasp_idx), 1.4)

        banner = (f"scene {self._scene}   attempt {attempt + 1}/{n_attempts}   "
                  f"-> grasp {grasp_idx}")
        if attempt > 0:
            banner += f"   (rewound to step {branch_step})"
        self._text(banner, (0.45, -0.35, 0.95), WHITE, 1.5)
        # A new pen stroke: the fork starts at the branch point, not at wherever
        # the previous attempt's path happened to end.
        self._prev_pt = None

    def replay_step(self, i: int, ee_pos) -> None:
        """Retrace the inherited prefix in grey as it re-executes."""
        if not self.enabled:
            return
        pt = np.asarray(ee_pos, dtype=np.float64)
        if self.show_path and self._prev_pt is not None:
            self._line(self._prev_pt, pt, GREY, 2.0, self._path_ids)
        self._prev_pt = pt
        if self.replay_pace:
            time.sleep(self.replay_pace)

    def mark_branch(self, ee_pos) -> None:
        """A white cross-hair where the retry takes over from the replay."""
        if not self.enabled:
            return
        p = np.asarray(ee_pos, dtype=np.float64)
        d = 0.022
        for ax in range(3):
            a, b = p.copy(), p.copy()
            a[ax] -= d
            b[ax] += d
            self._line(a, b, WHITE, 2.5, self._path_ids)

    def step(self, step: int, ee_pos, grasp_idx: int, obs=None, pc=None) -> None:
        """Extend the current attempt's path by one segment."""
        if not self.enabled:
            return
        pt = np.asarray(ee_pos, dtype=np.float64)
        if self.show_path and self._prev_pt is not None:
            self._line(self._prev_pt, pt, slot_colour(grasp_idx), 3.0, self._path_ids)
        self._prev_pt = pt
        if self.show_cloud and pc is not None and obs is not None:
            self._draw_cloud(obs, pc)
        if self.pace:
            time.sleep(self.pace)

    def _draw_cloud(self, obs, pc) -> None:
        """Overlay the EE-frame cloud the policy just saw, in world frame.

        Same transform chain `rollout_bc_policy_p5.draw_pointcloud` uses, kept
        here rather than imported so the viz module does not depend on a script.
        """
        ctx = self.cloud_ctx
        if ctx is None:
            return
        panda_base_inv_tf, R_base, panda_base_pos = ctx
        try:
            from collect_bc_dataset import _ee_pose_mat
            from core.utils import se3_transform_pc
            ee_mat = _ee_pose_mat(obs["panda_body"], obs["panda_link_ind_hand"],
                                  panda_base_inv_tf)
            pts_base = se3_transform_pc(ee_mat, np.asarray(pc)[:, :3].T).T
            pts_world = (R_base @ pts_base.T).T + panda_base_pos
        except Exception as e:                            # noqa: BLE001
            print(f"[viz] cloud overlay off after {type(e).__name__}: {e}")
            self.show_cloud = False
            return
        ycb = np.asarray(pc)[:, 3] > 0.5
        hand = np.asarray(pc)[:, 4] > 0.5
        cols = np.full((len(pts_world), 3), 0.6)
        cols[ycb] = [1.0, 0.5, 0.0]
        cols[hand] = [0.3, 0.5, 1.0]
        n = min(200, len(pts_world))
        idx = np.random.choice(len(pts_world), size=n, replace=False)
        self._clear(self._cloud_ids)
        i = self._guard(pybullet.addUserDebugPoints, pts_world[idx].tolist(),
                        cols[idx].tolist(), pointSize=4)
        if i is not None:
            self._cloud_ids.append(i)

    def end_attempt(self, att) -> None:
        """Second banner line: how this attempt ended, in green or red."""
        if not self.enabled:
            return
        colour = (0.1, 1.0, 0.2) if att.success else (1.0, 0.3, 0.25)
        self._text(f"{'SUCCESS' if att.success else 'FAILED'}  {att.reason}   "
                   f"{len(att.rows)} steps   min_pos {att.min_pos:.3f} m",
                   (0.45, -0.35, 0.88), colour, 1.4)
        if self.pause_s:
            time.sleep(self.pause_s)

    def end_scene(self) -> None:
        if self.enabled and self.pause_s:
            time.sleep(self.pause_s)

    # ── interactive stepping ─────────────────────────────────────────────────

    def wait(self, summary: str = "", keys: str = "nraq") -> str:
        """Block until one of `keys` is pressed in the GUI. Returns that key.

        Without this the window closes the moment the last scene finishes, which
        is exactly when there is something worth looking at — the completed
        trunk-and-fork with every attempt's path still drawn.

        Returns "q" immediately when the overlay is disabled or the GUI has gone
        away, so a caller can treat the return value as the decision and never
        has to check `enabled` itself. KeyboardInterrupt is caught and reported
        as "q" too: Ctrl-C at a prompt means stop, not crash, and the results
        collected so far are still worth writing.

        NOTE the keystrokes go to the PYBULLET WINDOW, not the terminal — it has
        to have focus.
        """
        if not self.enabled:
            return "q"
        labels = {"n": "next scene", "p": "previous scene",
                  "r": "re-run this scene",
                  "a": "run the rest without pausing", "q": "quit"}
        prompt = "   ".join(f"[{k.upper()}] {labels[k]}" for k in keys if k in labels)
        print(f"\n  {summary}\n  PyBullet window: {prompt}", flush=True)
        self._text(prompt, (0.45, -0.35, 0.81), (1.0, 0.95, 0.4), 1.3)

        codes = {ord(k): k for k in keys}
        try:
            while True:
                events = self._guard(pybullet.getKeyboardEvents)
                if events is None:          # _guard disabled us: the GUI is gone
                    return "q"
                for code, k in codes.items():
                    if code in events and events[code] & pybullet.KEY_WAS_TRIGGERED:
                        return k
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("  (interrupted)")
            return "q"

    def close(self) -> None:
        for ids in (self._path_ids, self._grasp_ids, self._cloud_ids, self._text_ids):
            self._clear(ids)
