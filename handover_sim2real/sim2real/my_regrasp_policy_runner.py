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
   Phase-4 policy grasps a held object however it likes; this one is told which
   SIDE to come from, or which PART to take, depending on the run's `d_rule` —
   and the same scene has a different right answer for each. `--direction` is
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
   the robot's +x. It means a side of the object, in a gravity-aligned frame
   whose azimuth is measured from a reference point:

       z = world up
       x = horizontal(base - object)          --anchor-ref base (default)
       x = horizontal(object - hand centroid) --anchor-ref hand
       y = z cross x

   Both installed runs NAME their bins in a frame anchored on the MANO wrist
   JOINT, which the rig cannot read, so deployment approximates it — and `base`
   is much the closer approximation (7.7% of bin labels change against the wrist
   frame, versus the ~60% measured for the hand-cloud centroid on run 16). See
   the ANCHOR_REFS block below; this is the one place a regrasp deployment goes
   quietly wrong.

   It is computed ONCE per episode and held, matching the simulator's
   `SIM.anchor_update: latched`, where `runner.set_direction` is called before
   step 0 and never again. Recomputing it per frame would let the command drift
   as the observation changes, and the policy would be chasing a target that
   moves because it moved.

THE TWO INSTALLED RUNS TAKE DIFFERENT COMMANDS. `--regrasp-run 9` was collected
under `d_rule: approach_axis` — `d` is the gripper's approach axis, "come at the
object from this side" — and `--regrasp-run 11` under `grasp_offset`, where `d`
is the grasp point's offset from the object centroid, "grasp this part of the
object". Same flag, different question; the resolved rule is printed at startup
and the REGRASP_RUNS block below carries the measured trade between them.

Both are the kind of run whose training label and deployment command are
different vectors: trained on each demonstration's own continuous direction
(`d_grasp_world`, dequantized, `d_noise_deg: 0.0`) and deployed on the empirical
CENTROID of each bin rather than the bin's geometric axis. Those centroids are
per-run and live in the run's `command_axes.json` — they are not the unit axes,
and substituting `BINS` would command a direction up to ~16 deg away from what
the run expects. That file is read, not assumed; passing a run without one is
refused, and a bin whose entry IS the raw axis is refused as undemonstrated.

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

# Installed regrasp runs live beside the Phase-4 ones, under their own name so
# the two families cannot be confused: everything in checkpoint/ called runNN is
# Phase 4 and 5-channel, everything called regrasp_* is 7-channel and needs a
# --direction. `available_runs()` in my_policy_runner only lists a folder that
# holds best.pt at its ROOT, so a regrasp run — whose checkpoints are one level
# down, per iteration — never appears in the Phase-4 --run list.
CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoint"

# THE INSTALLED REGRASP RUNS. Both are 7-channel, both were collected under
# `examples/pretrain_multicam_wr.yaml` (wrist + right) and both trained their
# point encoder from scratch, so the two big sim2real gaps are common to them.
# What differs is `SIM.d_rule` — WHAT THE DIRECTION MEANS — and it is not a
# tuning knob, it is a different command:
#
#   9   d_rule: approach_axis   d is the gripper's APPROACH AXIS: "come at the
#       object from this side". Best iteration 23, success 0.592, and it TRACKS
#       the command well — dir_err median 10.0 deg, bin_hit 0.65, cond_sep 0.89.
#       Its `-x` and `-z` bins are empty, so only four directions are live.
#   11  d_rule: grasp_offset    d is the offset of the GRASP POINT from the
#       object centroid: "grasp THIS PART of the object". Best iteration 22,
#       success 0.619 — higher — but it tracks the command far less: bin_hit
#       0.14 and cond_sep 0.31 against run 9's 0.65 and 0.89. All six bins have
#       members. (dir_err is not comparable across the two: the rules measure
#       different quantities.)
#
# Read that trade honestly before picking. Run 11 grasps more often and obeys
# less; `cond_sep` 0.31 is close to the failure the evaluator was built to
# detect — "the policy ignores the conditioning and regresses the mean". The
# flip side is the one prediction worth testing on hardware: a policy that leans
# less on the conditioning channels leans less on the inputs the real rig
# degrades most, so run 11 may well transfer better than its sim gap suggests.
REGRASP_RUNS = {"9": "regrasp_run9", "11": "regrasp_run11"}
DEFAULT_REGRASP_RUN_NAME = "9"


def run_root(name) -> Path:
    """The installed checkpoint tree for `--regrasp-run NAME`."""
    key = str(name)
    if key not in REGRASP_RUNS:
        raise SystemExit(f"--regrasp-run must be one of "
                         f"{', '.join(sorted(REGRASP_RUNS))}, got {name!r}")
    root = CHECKPOINT_ROOT / REGRASP_RUNS[key]
    if not root.is_dir():
        raise SystemExit(
            f"regrasp run {key} is not installed at {root}. Install it from "
            f"output/dagger_runs/{REGRASP_RUNS[key]} (see that folder's "
            f"README.md for the cp -al recipe).")
    return root


# Kept for the selftest and for anything that wants the default run's tree
# without going through argparse.
REGRASP_CHECKPOINT_DIR = CHECKPOINT_ROOT / REGRASP_RUNS[DEFAULT_REGRASP_RUN_NAME]
DEFAULT_REGRASP_RUN = REGRASP_CHECKPOINT_DIR / "best"

# All six bins are OFFERED, and which ones are live is decided per run from that
# run's own command_axes.json rather than hard-coded here — run 9 has four
# (`-x` over the giver's fingers and `-z` from beneath are empty in its training
# assignment) and run 11 has all six. See `live_directions`.
DIRECTION_CHOICES = tuple(rg_dirs.BIN_SHORT[:6])

# WHICH POINT THE ANCHOR AZIMUTH IS MEASURED FROM, and it has to match the frame
# run 9's bins are NAMED in. That frame is the MANO wrist JOINT: run 9's config
# sets no `SIM.anchor_hand_ref`, the evaluator and collector both default to
# "wrist", and output/regrasp_pins_train.json records no reference — which
# `setup.resolve_anchor_ref` reads as "wrist by construction".
#
# The real rig has no wrist joint to read, so it must approximate that frame,
# and the two candidates are NOT equally good approximations of it:
#
#   base   x = horizontal(p_base - c), the object -> robot azimuth. Needs no
#          hand at all. Measured against the wrist frame, 7.7% of grasps change
#          bin (anchor.anchor_rotation's SIGN note) — the sign was chosen for
#          exactly this comparability. Lever arm 61.3 cm, floor 41.8 cm, so a
#          1 cm centroid error moves x by 0.65 deg and the degenerate fallback
#          is unreachable rather than merely unlikely.
#   hand   x = horizontal(c - c_hand) from the SEGMENTED HAND CLOUD's centroid.
#          This is the run-16 mistake, and the repo already measured what it
#          costs: a wrist-named table read in a hand-centroid frame took bin
#          agreement from 99% to 40% and cut the usable DAgger shard from 57% to
#          19%, silently (setup.resolve_anchor_ref). Lever arm 9.2 cm falling to
#          7.7 cm at the close, ~5 deg per cm of centroid error, and below 4 cm
#          `anchor_rotation` engages a fallback whose sign is OPPOSITE — a hand
#          gripping the object is exactly the geometry that triggers it.
#
# So `base` is the default: it is both the closer match to the frame run 9 was
# trained and scored in AND the only one whose reference the rig knows exactly.
# `hand` is kept because it is what "+x = away from the giver's fingers" means
# literally, and because an A/B against it is the only way to measure the claim
# above on this hardware rather than inheriting it from sim.
ANCHOR_REFS = ("base", "hand")


def bin_index(name: str) -> int:
    """'+x' -> 0. The network never sees this; it selects which axis to command."""
    try:
        return rg_dirs.BIN_SHORT.index(name)
    except ValueError:                                   # pragma: no cover
        raise SystemExit(f"unknown direction {name!r}; have "
                         f"{', '.join(DIRECTION_CHOICES)}")


def fuse_direction_value(argv: list[str]) -> list[str]:
    """Splice `--direction -y` into `--direction=-y` before argparse sees it.

    argparse reads any token starting with the prefix char as an option name
    unless it looks like a negative NUMBER, so `-y` is taken for a flag and
    `--direction` is then reported as missing its value. `+y` works and `-y`
    does not, purely because '+' is not a prefix char — a confusing thing to
    hand a user whose four legal values include two signs.

    The `=` spelling bypasses that check entirely, so joining the pair here is
    enough and argparse's own prefix handling is left alone. Only a token that
    is NOT itself a long option gets consumed, so a genuinely missing value
    still produces argparse's error instead of swallowing the next flag.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if (tok == "--direction" and i + 1 < len(argv)
                and not argv[i + 1].startswith("--")):
            out.append(f"--direction={argv[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def load_command_axes(run_dir: Path,
                      explicit: Optional[str]) -> tuple[np.ndarray, str, dict]:
    """The six deployment axes for this run, the mode, and the file's own meta.

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
    return axes, str(blob.get("mode", "?")), blob


def live_directions(axes: np.ndarray, bins=None) -> tuple[str, ...]:
    """Which of the six bins this run actually has demonstrations behind.

    `GraspPinTable.bin_centroids` leaves a bin with no surviving members at its
    raw geometric axis — "with no assignment recorded there is nothing to
    average" — so an entry that is EXACTLY the unit axis is the empty marker,
    and any real centroid differs from it in at least the last digits. That is
    why this compares for EXACT equality against the axis set rather than with a
    tolerance: a centroid can legitimately land very close to its axis (run 11's
    `-x` is 0.9992 of one) and a tolerance would call it empty. `bins` defaults
    to the canonical set; pass the file's own when you have it, so a run that
    ever ships a different axis convention is still read in its own terms.
    """
    bins = np.asarray(rg_dirs.BINS if bins is None else bins, dtype=np.float64)
    return tuple(rg_dirs.BIN_SHORT[b] for b in range(6)
                 if not np.array_equal(np.asarray(axes[b], dtype=np.float64),
                                       bins[b]))


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
                 axes_mode: str, ckpt: str = "best", anchor_ref: str = "base",
                 d_rule: str = "?", run_name: str = "?"):
        self.d_rule = str(d_rule)
        self.run_name = str(run_name)
        self.direction = direction
        self.bin = bin_index(direction)
        self.run_dir = run_dir
        self.axes = axes
        self.axes_mode = axes_mode
        self.ckpt = ckpt
        if str(anchor_ref) not in ANCHOR_REFS:
            raise SystemExit(f"--anchor-ref must be one of "
                             f"{', '.join(ANCHOR_REFS)}, got {anchor_ref!r}")
        self.anchor_ref = str(anchor_ref)
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
        print(f"[regrasp] run {self.run_name}, direction {self.direction} "
              f"(bin {self.bin}), axes from {self.axes_mode}: "
              f"{np.round(self.axes[self.bin], 4).tolist()} in the anchor frame")
        # WHAT `--direction` MEANS, printed because it is not the same question
        # in the two installed runs and no amount of tuning reconciles them.
        print(f"[regrasp] d_rule {self.d_rule}: " + {
            "approach_axis": "the direction is the gripper's APPROACH AXIS — "
                             "'come at the object from this side'",
            "grasp_offset": "the direction is the GRASP POINT's offset from the "
                            "object centroid — 'grasp this part of the object', "
                            "which is a different command from an approach side",
        }.get(self.d_rule, "unrecognised rule — the command's meaning is "
                           "whatever this run's setup defined it as"))
        print(f"[regrasp] anchor reference: {self.anchor_ref} — "
              + ("x = horizontal(base - object), the frame the rig can "
                 "reproduce exactly (7.7% bin disagreement with run 9's "
                 "wrist-named table)"
                 if self.anchor_ref == "base" else
                 "x = horizontal(object - hand cloud centroid). THIS IS THE "
                 "RUN-16 MISMATCH: run 9's bins are named in the wrist frame "
                 "and agreement there measured 40%. A/B only."))

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
        # THE CONDITIONING CHANNELS' OWN HEALTH, which nothing else reports.
        # `d . n` is only a surface-orientation signal while the PCA normals
        # hold; every degenerate neighbourhood falls back to normalize(p - c),
        # i.e. to `d . r`, collapsing the two channels into one. A real cloud
        # resampled up to 896 points from far fewer unique ones is exactly how
        # that happens, and it is invisible in the mask overlay.
        info = getattr(self.runner, "last_cloud_info", None) or {}
        nfb = int(info.get("n_fallback", 0))
        npts = int(info.get("n_points", 0)) or 1
        nrm = f"  n_fb {nfb}/{npts} ({100.0 * nfb / npts:.0f}%)" if info else ""
        if info.get("no_centroid"):
            nrm += "  NO OBJ CENTROID"
        return (f"dir {self.direction}  anchor={mode}({self.anchor_ref})  "
                f"d_world=[{self.d_world[0]:+.2f} {self.d_world[1]:+.2f} "
                f"{self.d_world[2]:+.2f}]{warn}{nrm}")

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
        # Unread on the base path, and deliberately still COMPUTED there so the
        # printout can report the azimuth the other reference would have given.
        wrist = to_base(fused.hand_xyz)
        if centroid is None:
            raise RuntimeError("no object points; cannot anchor the direction")

        # `np.zeros(3)` is the robot base, exactly, because everything here is
        # already in the BASE frame — that is what makes this reference free on
        # hardware rather than another thing to estimate.
        R_anchor, meta = rg_anchor.anchor_rotation(
            centroid, wrist, np.zeros(3), state=self._anchor_state,
            reference=("base" if self.anchor_ref == "base" else "hand"))
        self._anchor_meta = meta

        d_base = rg_dirs.to_world(self.axes[self.bin], R_anchor)
        # Into the frame robot_state[21:25] is expressed in, because BCRunner
        # pairs the two. Only the rotation matters — d is a direction — and the
        # result is invariant to this choice as long as it MATCHES the state.
        self.d_world = rg_dirs.normalize(T_SIMWORLD_BASE[:3, :3] @ d_base)
        self.runner.set_direction(self.d_world)

        src = {"base_primary": "the ROBOT BASE (hand not used)",
               "base": "the ROBOT BASE — the hand reference COLLAPSED",
               "wrist": "the hand cloud centroid"}.get(meta.get("mode"), "?")
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
                   help="which direction to command, in the SCENE's anchor frame "
                        "(not the robot's): '+x' the free end away from the "
                        "giver, '+y'/'-y' laterally, '+z' from above. Required — "
                        "the policy is conditioned on this and has no meaningful "
                        "behaviour without it. WHAT it means depends on the "
                        "run's d_rule (approach side for run 9, which part of "
                        "the object for run 11), and which bins are live depends "
                        "on the run too — run 9 has no '-x' or '-z' "
                        "demonstrations and refuses them. Both spellings work: "
                        "'--direction -y' and '--direction=-y'.")
    g.add_argument("--regrasp-run", choices=tuple(sorted(REGRASP_RUNS)),
                   default=DEFAULT_REGRASP_RUN_NAME,
                   help="which installed regrasp run to drive. '9' commands an "
                        "APPROACH AXIS and follows the command closely (sim "
                        "success 0.592, bin_hit 0.65); '11' commands a GRASP "
                        "POINT offset and grasps more often but follows the "
                        "command much less (0.619, bin_hit 0.14). They are "
                        "different commands, not two tunings of one. "
                        f"Default: {DEFAULT_REGRASP_RUN_NAME}.")
    g.add_argument("--regrasp-run-dir", type=str, default=None,
                   help="a run snapshot elsewhere, holding config.yaml, "
                        "normalization.npz and checkpoints/, with its own "
                        "command_axes.json beside it. Overrides --regrasp-run "
                        "and is mutually exclusive with --regrasp-iter.")
    g.add_argument("--regrasp-iter", type=str, default=None, metavar="N",
                   help="pick an iteration of --regrasp-run by number: "
                        "--regrasp-iter 23. Also accepts 'best' (the exported "
                        "best iteration — 23 for run 9, 22 for run 11) and "
                        "'last'. Defaults to 'best'. Success is NOT monotonic "
                        "in the iteration — see that run folder's README.")
    g.add_argument("--regrasp-ckpt", type=str, default="best",
                   help="which checkpoint inside the run dir (best | last).")
    g.add_argument("--anchor-ref", choices=ANCHOR_REFS, default="base",
                   help="which point the anchor azimuth is measured from. "
                        "'base' (default) uses horizontal(base - object), needs "
                        "no hand and is within 7.7%% of the bin labels run 9's "
                        "table was built with. 'hand' uses the segmented hand "
                        "cloud's centroid, which is the run-16 frame mismatch "
                        "(40%% bin agreement, measured) — kept for A/B only.")
    g.add_argument("--command-axes", type=str, default=None,
                   help="command_axes.json holding the six deployment "
                        "directions. Defaults to the one beside the run dir. "
                        "Run 9 deploys on bin CENTROIDS, which are not the unit "
                        "axes — this file is how the runner knows them.")
    return p


def resolve_run_dir(args) -> Path:
    """Which run directory to load: --regrasp-run + --regrasp-iter, or an explicit dir.

    `--regrasp-run-dir` wins outright and skips the registry, because it is the
    escape hatch for a run that was never installed here — it carries its own
    `command_axes.json` beside it, which is what makes it self-contained.
    """
    if args.regrasp_run_dir is not None:
        if args.regrasp_iter is not None:
            raise SystemExit(
                f"--regrasp-iter {args.regrasp_iter} and --regrasp-run-dir "
                f"{args.regrasp_run_dir} both name a policy. Pass one.")
        run_dir = Path(args.regrasp_run_dir).expanduser()
    else:
        root = run_root(args.regrasp_run)
        name = "best" if args.regrasp_iter is None else str(args.regrasp_iter).strip()
        if name in ("best", "last"):
            sub = name
        else:
            try:
                sub = f"iter_{int(name):02d}"
            except ValueError:
                raise SystemExit(
                    f"--regrasp-iter {name!r} is not a number, 'best' or 'last'")
        run_dir = root / sub
        if not run_dir.is_dir():
            have = sorted(d.name for d in root.glob("iter_*"))
            raise SystemExit(
                f"No {run_dir}.\nInstalled for run {args.regrasp_run}: "
                + (", ".join(["best", "last"] + have) if have
                   else f"nothing under {root}"))
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"No run dir at {run_dir}")
    return run_dir


def selftest() -> None:
    """Check the direction geometry offline — no camera, no robot, no ROS.

    The one thing that cannot be eyeballed on hardware is whether `+x` means
    what it should, because a wrong anchor frame produces a confident, smooth,
    entirely wrong approach. So the frame is asserted against the definition
    rather than against a remembered number.
    """
    import types

    # The CLI before the geometry. `-y` and `+y` must reach the parser alike;
    # the asymmetry that argparse imposes on them is a string bug, not a
    # geometry one, but it is the one that stopped a run on hardware.
    for spelling in (["--direction", "-y"], ["--direction=-y"],
                     ["--direction", "+y"], ["--direction=+z"]):
        want = spelling[-1].split("=")[-1]
        got = build_parser().parse_args(fuse_direction_value(spelling)).direction
        assert got == want, f"{' '.join(spelling)} parsed as {got!r}"
    assert fuse_direction_value(["--direction", "--home"]) == \
        ["--direction", "--home"], (
        "a missing --direction value must stay missing, not eat the next flag")
    print("direction spellings: '-y' and '=-y' both parse")

    axes, mode, meta = load_command_axes(DEFAULT_REGRASP_RUN, None)
    live = live_directions(axes, meta.get("bins"))
    print(f"command axes: {mode}, d_rule {meta.get('d_rule')}, live {live}")
    assert live == ("+x", "+y", "-y", "+z"), (
        f"run {DEFAULT_REGRASP_RUN_NAME}'s live bins should be the four with "
        f"demonstrations; got {live}")
    # And every installed run must be readable and self-consistent, since a
    # missing command_axes.json only surfaces at the first act() otherwise.
    for name in sorted(REGRASP_RUNS):
        a, mo, me = load_command_axes(run_root(name) / "best", None)
        print(f"  run {name}: mode {mo}, d_rule {me.get('d_rule')}, "
              f"live {live_directions(a, me.get('bins'))}")
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
    # The DEFAULT reference: the robot base, which in this frame is the origin.
    expect_x = rg_anchor.normalize(rg_anchor.horizontal(np.zeros(3) - c_b))
    # And the alternative, asserted separately below — the two are 63 deg apart
    # in this synthetic scene, which is the whole reason the choice is a flag
    # and not an implementation detail.
    expect_hand = rg_anchor.normalize(rg_anchor.horizontal(c_b - w_b))
    print(f"  anchor +x (ref=base) should be horizontal(base - object) = "
          f"{np.round(expect_x, 3).tolist()}")
    print(f"  anchor +x (ref=hand) would be horizontal(object - hand) = "
          f"{np.round(expect_hand, 3).tolist()}  "
          f"({np.rad2deg(np.arccos(np.clip(expect_x @ expect_hand, -1, 1))):.1f} "
          f"deg apart)")

    seen = {}
    for name in live:
        a = RegraspPolicy(name, DEFAULT_REGRASP_RUN, axes, mode,
                          d_rule=str(meta.get("d_rule", "?")),
                          run_name=DEFAULT_REGRASP_RUN_NAME)
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
        "horizontal(base - object) — the anchor frame is wrong, and every bin "
        "label with it")

    # THE OTHER REFERENCE MUST ACTUALLY BE A DIFFERENT FRAME. Without this the
    # flag could be wired to nothing and every check above would still pass,
    # which is precisely how the run-16 mismatch survived a whole 20-iteration
    # run without raising.
    a_hand = RegraspPolicy("+x", DEFAULT_REGRASP_RUN, axes, mode,
                           anchor_ref="hand",
                           d_rule=str(meta.get("d_rule", "?")),
                           run_name=DEFAULT_REGRASP_RUN_NAME)
    a_hand.load(types.SimpleNamespace(), "cuda", Path("."), "best")
    a_hand.reset()
    a_hand.act(pc5, rs, fused=fused, T_base_hand=T_base_hand)
    d_hand = T_SIMWORLD_BASE[:3, :3].T @ a_hand.d_world
    cos_h = float(d_hand @ expect_hand)
    assert cos_h > 0.98, (
        f"ref=hand '+x' points {np.rad2deg(np.arccos(cos_h)):.1f} deg off "
        "horizontal(object - hand centroid) — --anchor-ref hand is not doing "
        "what it says")
    sep = np.rad2deg(np.arccos(np.clip(float(d_hand @ seen["+x"][0]), -1, 1)))
    assert sep > 5.0, (
        f"the two anchor references produced the same command ({sep:.1f} deg "
        "apart) — --anchor-ref is wired to nothing")
    print(f"  ref=base and ref=hand command '+x' {sep:.1f} deg apart")
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
    args = build_parser().parse_args(fuse_direction_value(sys.argv[1:]))
    run_dir = resolve_run_dir(args)
    axes, axes_mode, axes_meta = load_command_axes(run_dir, args.command_axes)

    # REFUSE A BIN THIS RUN NEVER DEMONSTRATED. Its centroid is then the raw
    # geometric axis, so the command is well-formed and the policy will fly it
    # confidently — there is simply nothing behind it. Silence here would look
    # exactly like a policy that had learned the direction and ignored it.
    live = live_directions(axes, axes_meta.get("bins"))
    if args.direction not in live:
        raise SystemExit(
            f"--direction {args.direction} has NO demonstrations in this run: "
            f"its command_axes entry is the untouched geometric axis, not a "
            f"centroid of anything. Live directions here: "
            f"{', '.join(live)}.")

    adapter = RegraspPolicy(args.direction, run_dir, axes, axes_mode,
                            ckpt=args.regrasp_ckpt, anchor_ref=args.anchor_ref,
                            d_rule=str(axes_meta.get("d_rule", "?")),
                            run_name=str(args.regrasp_run))
    m.main(adapter=adapter, args=args)


if __name__ == "__main__":
    main()
