#!/usr/bin/env python3
"""Run a REGRASP policy (phase 5) on the FR3, conditioned on a grasp direction.

    python my_regrasp_policy_runner.py --direction +x --cameras tripod \
        --calib-session d455 --segmentation sam2 --dry-run

Everything about driving the robot — homing, the settle/rate control modes, the
droop estimator, the gripper, the `t` abort, perception, the 3D view — is
`my_policy_runner`'s, imported rather than copied. This file is only the three
things the regrasp policy does differently.

WHAT IS DIFFERENT, in the order it bites:

1. THE COMMAND IS A DIRECTION, and it is what makes this a different task. The
   Phase-4 policy grasps a held object however it likes; this one is told WHICH
   SIDE to come from — `+x` the free end, `+y` / `-y` laterally, `+z` from above
   — and the same scene has a different right answer for each. `--direction` is
   therefore required: there is no sensible default, and conditioning on nothing
   is not a safe fallback (the two channels would read 0 everywhere, which the
   network cannot distinguish from "approach from nowhere").

2. THE DIRECTION RIDES IN THE POINT CLOUD, not in the robot state. The cloud is
   [1024, 7]: the familiar xyz + object + hand, plus `d.n` and `d.r` per point —
   the commanded direction dotted with that point's surface normal and with its
   bearing from the object centroid. Both are dot products of unit vectors, so
   they need no normalization and carry no dataset statistics, which is
   precisely what makes them portable to a real camera. `handover_sim2real
   .regrasp.channels.build_model_cloud` does it, and `BCRunner.act` calls that
   for us when handed a 5-channel cloud — so the perception stack needs no
   changes at all.

3. THE DIRECTION IS ANCHORED TO THE SCENE, not to the robot. `+x` does not mean
   the robot's +x. It means "the free end of the object", in a frame built from
   where the object is relative to the giver's hand:

       z = world up
       x = horizontal(object centroid - wrist), i.e. away from the giver
       y = z cross x

   So the same `--direction +x` follows the object around as the person turns.
   That frame needs the object centroid AND the giver's wrist, which on this rig
   come from the two segmented point classes — which is why `handover_sim2real
   .regrasp.anchor` was written to take plain arrays and no simulator.

   It is computed ONCE per episode and held, matching the simulator, where
   `runner.set_direction` is called before step 0 and never again. Recomputing
   it per frame would let the command drift as the hand moves, and the policy
   would be chasing a target that moves because it moved.

RUN 9 SPECIFICS. Run 9 is the first run whose training label and deployment
command are different vectors: it was trained on each demonstration's own
continuous approach axis (`d_grasp_world`, dequantized) and is deployed on the
empirical CENTROID of each bin rather than the bin's geometric axis. Those
centroids are per-run and live in the run's `command_axes.json` — they are not
the unit axes, and substituting `BINS` would command a direction up to ~16 deg
away from what the run expects. That file is read, not assumed; passing a run
without one is refused.

Frames: the direction command is frame-invariant — `d_ee = R_ee.T @ d_world` is
unchanged by any rotation applied to both, verified to 1e-12 — so the anchor can
be built in the real base frame. It is mapped into the sim-world frame at the
end only because `BCRunner` pairs it with `robot_state[21:25]`, which
`build_robot_state` fills in that convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import my_policy_runner as m
from my_policy_runner import T_SIMWORLD_BASE

# Pure numpy/scipy, no simulator: `channels.py` says in its own header that it
# is "the module sim2real/pointcloud_multicam.py will import when the 7-channel
# path reaches the robot", and `anchor.py` that it takes "plain arrays in, plain
# arrays out ... on the real rig p_wrist comes from hand segmentation".
from handover_sim2real.regrasp import anchor as rg_anchor          # noqa: E402
from handover_sim2real.regrasp import directions as rg_dirs        # noqa: E402
from handover_sim2real.regrasp.policy_io import load_policy_runner  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGRASP_RUN = REPO_ROOT / "output" / "dagger_runs" / "regrasp_run9" / "best"

# The four bins with demonstrations behind them. `-x` (over the giver's fingers)
# and `-z` (from beneath) are measured empty in the training assignment, and
# their entries in command_axes.json are the untouched fallback axes rather than
# a centroid of anything — commanding one would be asking for a direction the
# policy has never been trained on.
DIRECTION_CHOICES = ("+x", "+y", "-y", "+z")


def bin_index(name: str) -> int:
    """'+x' -> 0. The network never sees this; it selects which axis to command."""
    try:
        return rg_dirs.BIN_SHORT.index(name)
    except ValueError:                                   # pragma: no cover
        raise SystemExit(f"unknown direction {name!r}; have "
                         f"{', '.join(DIRECTION_CHOICES)}")


def load_command_axes(run_dir: Path, explicit: Optional[str]) -> tuple[np.ndarray, str]:
    """The six deployment axes for this run, and the mode that produced them.

    Read from the run rather than reconstructed. `SIM.command_deploy` decides
    whether a run is deployed on the bin's geometric axis or on the empirical
    centroid of the demonstrations filed under it, and for run 9 it is the
    centroid — up to ~16 deg from the axis. Rebuilding the centroids here would
    need the pin table and the demo filter that produced them, so the run writes
    the resolved answer out at setup and this reads it back.
    """
    path = Path(explicit) if explicit else (run_dir.parent / "command_axes.json")
    if not path.exists():
        raise SystemExit(
            f"No command_axes.json at {path}.\n"
            "It holds the direction vectors this run was DEPLOYED on, which for "
            "a bin_centroid run are not the unit axes — substituting those would "
            "command a direction up to ~16 deg from what the policy expects. It "
            "is written next to the run dir at setup; pass --command-axes if "
            "yours is elsewhere.")
    blob = json.loads(path.read_text())
    axes = np.asarray(blob["axes"], dtype=np.float64)
    if axes.shape != (6, 3):
        raise SystemExit(f"{path}: expected 6x3 axes, got {axes.shape}")
    return axes, str(blob.get("mode", "?"))


def class_centroid(xyz: np.ndarray) -> Optional[np.ndarray]:
    """Median of a point class, or None if there is nothing to average.

    Median rather than mean, for the same reason `select_hand_component` uses
    one: a class that has picked up a few points off a depth edge has them
    metres away, and a mean follows them there.
    """
    if xyz is None or len(xyz) < 3:
        return None
    return np.median(np.asarray(xyz, dtype=np.float64), axis=0)


class RegraspPolicy(m.Phase4Policy):
    """The regrasp policy behind my_policy_runner's adapter interface."""

    def __init__(self, direction: str, run_dir: Path, axes: np.ndarray,
                 axes_mode: str, ckpt: str = "best"):
        self.direction = direction
        self.bin = bin_index(direction)
        self.run_dir = run_dir
        self.axes = axes
        self.axes_mode = axes_mode
        self.ckpt = ckpt
        self.runner = None
        self.d_world: Optional[np.ndarray] = None
        self._anchor_state = None
        self._anchor_meta: dict = {}

    # ── adapter interface ────────────────────────────────────────────────────

    def load(self, args, device: str, policy_dir: Path, ckpt: str) -> None:
        # policy_dir/ckpt come from the Phase-4 flags and are ignored: this
        # policy is named by --regrasp-run-dir, which points at a DAgger run
        # snapshot rather than at sim2real/checkpoint/.
        self.device = device
        self.runner, self.run_cfg = load_policy_runner(
            str(self.run_dir), device=device, ckpt=self.ckpt)
        d = self.run_cfg["DATA"]
        mdl = self.run_cfg["MODEL"]
        if int(d.get("pc_channels", 5)) != 7:
            raise SystemExit(
                f"{self.run_dir} has pc_channels={d.get('pc_channels')}. This "
                "runner drives the DIRECTION-CONDITIONED regrasp policy, whose "
                "cloud is 7-channel. A 5-channel run belongs in "
                "my_policy_runner.py.")
        if not bool(mdl.get("drop_joint_state", False)) or bool(mdl.get("use_prev_act", True)):
            raise SystemExit(
                "This runner fills only robot_state[18:26] (EE pose + gripper). "
                f"The run has drop_joint_state={mdl.get('drop_joint_state')} / "
                f"use_prev_act={mdl.get('use_prev_act')}, so it also reads joint "
                "state and/or the previous action, which the real robot does not "
                "provide here. Feeding zeros would be silently wrong.")
        print(f"[regrasp] direction {self.direction} (bin {self.bin}), "
              f"axes from {self.axes_mode}: "
              f"{np.round(self.axes[self.bin], 4).tolist()} in the anchor frame")

    def reset(self) -> None:
        """Drop the latched direction; the next observation re-anchors."""
        self.d_world = None
        # A fresh latch per episode. Within one, the state is what stops the
        # anchor flickering between the wrist reference and the base fallback
        # when the hand is intermittently segmented.
        self._anchor_state = rg_anchor.AnchorState()
        self._anchor_meta = {}
        if self.runner is not None:
            self.runner.reset()

    def act(self, pc: np.ndarray, rs: np.ndarray, *, fused=None,
            T_base_hand=None) -> np.ndarray:
        if self.d_world is None:
            self._set_direction(fused, T_base_hand)
        return self.runner.act(pc, rs)

    def hud(self) -> str:
        if self.d_world is None:
            return f"dir {self.direction}: NOT ANCHORED YET"
        mode = self._anchor_meta.get("mode", "?")
        warn = "  DEGENERATE" if self._anchor_meta.get("degenerate") else ""
        return (f"dir {self.direction}  anchor={mode}  "
                f"d_world=[{self.d_world[0]:+.2f} {self.d_world[1]:+.2f} "
                f"{self.d_world[2]:+.2f}]{warn}")

    # ── the direction ────────────────────────────────────────────────────────

    def _set_direction(self, fused, T_base_hand) -> None:
        """Build the anchor frame from this observation and latch the command.

        Once per episode. The object and the giver's wrist come from the two
        segmented classes, which is the whole reason `anchor_rotation` takes
        plain arrays — in sim those two points come from the YCB body and a MANO
        link, and here they come from a camera, with nothing else changing.
        """
        if fused is None or T_base_hand is None:
            raise RuntimeError(
                "the regrasp adapter needs the fused observation and the robot "
                "pose to anchor its direction")

        # Both classes arrive in panda_hand; the anchor frame is defined against
        # world up and the robot base, so they go to the base frame first.
        def to_base(xyz):
            c = class_centroid(xyz)
            return None if c is None else (T_base_hand @ np.append(c, 1.0))[:3]

        centroid = to_base(fused.object_xyz)
        wrist = to_base(fused.hand_xyz)
        if centroid is None:
            raise RuntimeError("no object points; cannot anchor the direction")

        R_anchor, meta = rg_anchor.anchor_rotation(
            centroid, wrist, np.zeros(3), state=self._anchor_state)
        self._anchor_meta = meta

        d_base = rg_dirs.to_world(self.axes[self.bin], R_anchor)
        # Into the frame robot_state[21:25] is expressed in, because BCRunner
        # pairs the two. Only the rotation matters — d is a direction — and the
        # result is invariant to this choice as long as it MATCHES the state.
        self.d_world = rg_dirs.normalize(T_SIMWORLD_BASE[:3, :3] @ d_base)
        self.runner.set_direction(self.d_world)

        src = "wrist" if meta.get("mode") != "base" else "ROBOT BASE (no hand)"
        print(f"[regrasp] anchored on {src}: object at "
              f"{np.round(centroid, 3).tolist()}"
              + ("" if wrist is None else f", wrist at {np.round(wrist, 3).tolist()}")
              + f" -> commanding {self.direction} = "
                f"{np.round(d_base, 3).tolist()} in the base frame", flush=True)
        if meta.get("degenerate"):
            print("[regrasp] WARNING: the anchor frame is DEGENERATE — the "
                  "object is directly above both the wrist and the robot base, "
                  "so its x axis is arbitrary and every bin label with it.",
                  flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = m.build_parser()
    p.description = ("Run a direction-conditioned REGRASP policy on the FR3. "
                     "Shares my_policy_runner's control, perception and safety.")
    g = p.add_argument_group("regrasp")
    g.add_argument("--direction", choices=DIRECTION_CHOICES, required=True,
                   help="which side to grasp the object from, in the SCENE's "
                        "anchor frame (not the robot's): '+x' the free end away "
                        "from the giver, '+y'/'-y' laterally, '+z' from above. "
                        "Required — the policy is conditioned on this and has no "
                        "meaningful behaviour without it. '-x' and '-z' exist as "
                        "bins but have no demonstrations behind them.")
    g.add_argument("--regrasp-run-dir", type=str, default=str(DEFAULT_REGRASP_RUN),
                   help=f"run snapshot holding config.yaml, normalization.npz "
                        f"and checkpoints/. Default: {DEFAULT_REGRASP_RUN}")
    g.add_argument("--regrasp-ckpt", type=str, default="best",
                   help="which checkpoint inside the run dir (best | last).")
    g.add_argument("--command-axes", type=str, default=None,
                   help="command_axes.json holding the six deployment "
                        "directions. Defaults to the one beside the run dir. "
                        "Run 9 deploys on bin CENTROIDS, which are not the unit "
                        "axes — this file is how the runner knows them.")
    return p


def selftest() -> None:
    """Check the direction geometry offline — no camera, no robot, no ROS.

    The one thing that cannot be eyeballed on hardware is whether `+x` means
    what it should, because a wrong anchor frame produces a confident, smooth,
    entirely wrong approach. So the frame is asserted against the definition
    rather than against a remembered number.
    """
    import types

    axes, mode = load_command_axes(DEFAULT_REGRASP_RUN, None)
    print(f"command axes: {mode}")
    assert mode == "bin_centroid", (
        f"run 9 deploys on bin centroids; this run says {mode!r}")

    # Object out in front of the giver's hand, hand nearer the person.
    obj_hand = np.array([0.0, 0.0, 0.35])
    wrist_hand = np.array([0.05, 0.10, 0.45])
    T_base_hand = np.eye(4)
    T_base_hand[:3, 3] = [0.45, 0.0, 0.40]
    rng = np.random.default_rng(0)

    class _Fused:
        object_xyz = (rng.normal(size=(400, 3)) * 0.02 + obj_hand).astype(np.float32)
        hand_xyz = (rng.normal(size=(120, 3)) * 0.02 + wrist_hand).astype(np.float32)

    fused = _Fused()
    pc5 = np.zeros((1024, 5), np.float32)
    pc5[:896, :3] = fused.object_xyz[rng.integers(0, 400, 896)]
    pc5[:896, 3] = 1.0
    pc5[896:, :3] = fused.hand_xyz[rng.integers(0, 120, 128)]
    pc5[896:, 4] = 1.0
    rs = m.build_robot_state(T_base_hand, 1.0)

    c_b = (T_base_hand @ np.append(np.median(fused.object_xyz, 0), 1.0))[:3]
    w_b = (T_base_hand @ np.append(np.median(fused.hand_xyz, 0), 1.0))[:3]
    expect_x = rg_anchor.normalize(rg_anchor.horizontal(c_b - w_b))
    print(f"  anchor +x should be horizontal(object - wrist) = "
          f"{np.round(expect_x, 3).tolist()}")

    seen = {}
    for name in DIRECTION_CHOICES:
        a = RegraspPolicy(name, DEFAULT_REGRASP_RUN, axes, mode)
        a.load(types.SimpleNamespace(), "cuda", Path("."), "best")
        a.reset()
        act = a.act(pc5, rs, fused=fused, T_base_hand=T_base_hand)
        d_base = T_SIMWORLD_BASE[:3, :3].T @ a.d_world
        seen[name] = (d_base, act)
        print(f"  {name}: d_base={np.round(d_base, 3).tolist()}  "
              f"|d|={np.linalg.norm(d_base):.4f}  action={np.round(act, 4)}")
        assert abs(np.linalg.norm(d_base) - 1.0) < 1e-9, "d is not a unit vector"
        assert act.shape == (7,), act.shape

    # +x is the anchor's own x, up to the centroid axis's small tilt out of it.
    cos_x = float(seen["+x"][0] @ expect_x)
    assert cos_x > 0.98, (
        f"'+x' points {np.rad2deg(np.arccos(cos_x)):.1f} deg off "
        "horizontal(object - wrist) — the anchor frame is wrong, and every bin "
        "label with it")
    assert seen["+z"][0][2] > 0.95, "'+z' is not pointing up"
    for n in ("+y", "-y"):
        assert abs(seen[n][0] @ expect_x) < 0.35, f"{n} is not lateral to +x"

    # THE ANGLES BETWEEN COMMANDS MUST SURVIVE THE ANCHORING, which is the real
    # test that one orthonormal rotation was applied to all four and not, say,
    # a per-bin frame or a transposed one.
    #
    # Asserted against the run's own axes rather than against the unit bins,
    # because run 9's are empirical CENTROIDS and are not the axes: '+y' and
    # '-y' meet at 150 deg, not 180, since demonstrations in both lateral bins
    # also lean toward the free end (both carry +0.21 of x). Hard-coding -1
    # here would be asserting a geometry this run does not have.
    for a_name, b_name in (("+x", "+y"), ("+y", "-y"), ("-y", "+z"), ("+x", "+z")):
        want = float(rg_dirs.normalize(axes[bin_index(a_name)])
                     @ rg_dirs.normalize(axes[bin_index(b_name)]))
        got = float(seen[a_name][0] @ seen[b_name][0])
        print(f"  angle {a_name} to {b_name}: "
              f"{np.rad2deg(np.arccos(np.clip(got, -1, 1))):5.1f} deg "
              f"(anchor-frame {np.rad2deg(np.arccos(np.clip(want, -1, 1))):5.1f})")
        assert abs(got - want) < 1e-9, (
            f"anchoring changed the angle between {a_name} and {b_name} "
            f"({want:.4f} -> {got:.4f}) — the rotation is not orthonormal or "
            "was not applied consistently")
    # Four different commands must produce four different actions, or the
    # conditioning is not reaching the network at all.
    poses = [tuple(np.round(v[1][:6], 5)) for v in seen.values()]
    assert len(set(poses)) == len(poses), (
        "two directions produced identical actions — the conditioning channels "
        "are not reaching the policy")

    # And the invariance the frame handling relies on.
    from scipy.spatial.transform import Rotation as Rot
    Rsw = T_SIMWORLD_BASE[:3, :3]
    Ree_b = Rot.from_euler("xyz", [np.pi, 0.2, 0.7]).as_matrix()
    d_b = seen["+x"][0]
    assert np.allclose(Ree_b.T @ d_b, (Rsw @ Ree_b).T @ (Rsw @ d_b), atol=1e-12), (
        "d_ee is not frame-invariant — the base/sim-world mapping is wrong")
    print("  d_ee is frame-invariant (base vs sim-world agree to 1e-12)")
    print("\nall regrasp direction checks passed")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    args = build_parser().parse_args()
    run_dir = Path(args.regrasp_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"No run dir at {run_dir}")
    axes, axes_mode = load_command_axes(run_dir, args.command_axes)
    adapter = RegraspPolicy(args.direction, run_dir, axes, axes_mode,
                            ckpt=args.regrasp_ckpt)
    m.main(adapter=adapter, args=args)


if __name__ == "__main__":
    main()
