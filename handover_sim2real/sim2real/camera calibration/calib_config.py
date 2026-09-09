"""Every calibration parameter, in one place.

The board spec used to live in two independent copies (calibrate.py and
validate_calibration.py) that had to be edited in lockstep — editing one and not
the other validated a calibration against a different board than it was solved
with, and nothing complained. Everything now reads from here.

Only this file should need editing between calibration sessions.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── the printed board ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 8
    squares_y: int = 8
    # MEASURE THESE WITH CALIPERS. Span 8 squares and divide by 8 rather than
    # trusting the nominal value — printers rescale silently, and board size is
    # the one geometric input that does NOT cancel out of the AX=XB solve. It
    # sets metric scale, so the error passes straight through: measured on this
    # rig, a 2% size error moved the solved camera position by 9 mm, twice the
    # entire calibration residual.
    square_length_m: float = 0.02142
    marker_length_m: float = 0.0159
    dictionary: str = "DICT_5X5_50"
    # True unless the board was generated with OpenCV >= 4.6 defaults. A
    # mismatch shifts ChArUco corner IDs and yields a confident, wrong pose.
    legacy_pattern: bool = True


BOARD = BoardSpec()


# ── cameras ──────────────────────────────────────────────────────────────────

# Serial per role, so scripts can be pointed at "the tripod camera" instead of a
# number. Identify them with `python calib_common.py --preview`, which shows a
# live window per camera with the serial overlaid — jog the robot and the wrist
# one is the view that moves. Guessing wrong means calibrating the wrong camera
# with no error at any stage.
#
# QUOTE THESE. They are strings, not numbers: a serial beginning with 0 is not a
# valid Python integer literal at all, and one that happens to parse would then
# never match the string librealsense reports.
CAMERA_SERIALS: dict[str, str | None] = {
    "tripod": "825312073923",
    "wrist": "045322075902",
}

DEFAULT_ROLE = "tripod"


@dataclass(frozen=True)
class StreamSpec:
    # Intrinsics are resolution-specific. Capture, calibration and deployment
    # must all agree with whatever is exported here.
    width: int = 640
    height: int = 480
    fps: int = 30


STREAM = StreamSpec()


# ── robot pose source ────────────────────────────────────────────────────────

ROSBRIDGE_HOST = "172.16.0.7"
ROSBRIDGE_PORT = 9090

# franka_states carries O_T_EE, which on this robot IS the panda_hand frame:
# F_T_EE has no translation, only the -45 deg z rotation, and pybullet FK of
# panda_hand at the reported joints matches /cartesian_pose to 0.06 mm. So the
# calibration comes out in the same frame the policy uses, with no offset.
POSE_TOPIC = "/franka_state_controller/franka_states"
POSE_TYPE = "franka_msgs/FrankaState"
POSE_SOURCE = "franka_state"        # or "pose_stamped" for /cartesian_pose
MAX_POSE_AGE_S = 1.0


# ── solve ────────────────────────────────────────────────────────────────────

# Measured on a 10-pose set by the T_gripper_board consistency residual:
#   TSAI   8.17 mm / 1.50 deg      PARK        4.83 / 1.00
#   HORAUD 4.84 mm / 1.00 deg      DANIILIDIS  4.68 / 1.00
# All four agree on camera position to ~4 mm, so this is accuracy, not
# correctness. Tsai solves rotation then translation, so rotation error feeds
# into translation; the others solve both jointly.
HAND_EYE_METHOD = "DANIILIDIS"      # TSAI | PARK | HORAUD | ANDREFF | DANIILIDIS

# A ChArUco pose is solved from the board's INTERIOR corners:
# (squares_x-1) * (squares_y-1) = 7*7 = 49 for the default 8x8 board. All 49 when
# the board is close and sharp; fewer as it recedes, blurs, tilts edge-on or
# leaves the frame.
#
# 6 was the original floor and is far too permissive — a 6-corner PnP is
# unreliable, and it showed: in session_01 the two most distant captures
# (27 and 32 corners, 0.70 and 0.64 m) include the one with the worst
# gripper->board error, 8.9 mm against a 3.4 mm median.
MIN_CHARUCO_CORNERS = 20            # hard floor; capture refuses below this
GOOD_CHARUCO_CORNERS = 40           # capture overlay turns green at/above this

# TILT THE BOARD. A planar target viewed square-on hardly constrains its own
# out-of-plane rotation, so corner noise becomes pose error — and hand-eye
# inherits it. Measured on the `test` session: captures at >=39 deg tilt gave
# 0.13-0.23 deg rotation residual, those at <=28 deg gave 0.39-1.99 deg.
# Re-solving on tilt >= 30 alone took the rotation residual from 0.658 to
# 0.173 deg with fewer than half the captures.
#
# Guidance, not a hard gate: some square-on views are fine, they just must not
# dominate. The capture overlay turns green at or above this.
GOOD_BOARD_TILT_DEG = 30.0

# KEEP THE BOARD NEAR THE MIDDLE OF THE FRAME.
#
# The colour stream's factory intrinsics carry ZERO distortion coefficients. The
# lens is not actually distortion-free, and the residual is worst at the edges —
# where PnP absorbs it not as reprojection error, which stays sub-pixel, but as a
# tilt of the board. That lands directly in the rotation residual, which is the
# threshold hand-eye fails first.
#
# Measured on session_02, 16 captures, board detection clean throughout at
# 0.55 px mean PnP reprojection:
#
#   captures with the board in the middle half   rot residual 0.324 deg
#   captures with the board toward the edges     rot residual 0.856 deg
#   corr(distance from principal point, rot err) +0.70
#
# and not a confound: the edge captures were NEARER (-0.36 with distance),
# no more square-on (-0.13 with tilt), and detected MORE corners (+0.42).
# Re-solving on the 10 most central captures alone gave 2.43 mm / 0.382 deg —
# passing both thresholds — with pose diversity unchanged at 38 deg.
#
# Re-estimating intrinsics from the hand-eye captures does NOT fix it and makes
# it worse (0.716 -> 1.917 deg): 16 views of a small board at similar depths
# cannot separate focal length from distance, and the fit walks the principal
# point 51 px. Fixing it properly needs a dedicated intrinsics calibration —
# many views, board filling the frame, all four corners of the image, varied
# depth. Until then, capture centrally.
#
# Fraction of the half-diagonal within which the board centroid should sit.
GOOD_BOARD_RADIUS_FRAC = 0.35

# RE-SEAT THE BOARD, THEN MOVE THE ARM AROUND BEFORE CAPTURING ANYTHING.
#
# A board on a wrist mount SETTLES, and it does so during the first few poses —
# exactly the ones you are recording. In session_03 the five earliest captures
# were the five most costly in a leave-one-out, despite tilts of 33-40 deg and
# 0.55 px detection, and the recovered gripper->board translation stepped ~4 mm
# in y around the ninth capture. Neither is visible in any per-image metric; it
# shows up only as rotation residual that decreases through the session
# (corr(capture order, rot err) = -0.74).
#
# So: clamp or tape the board so it cannot rotate on the mount, then jog the arm
# through the full range you intend to capture over BEFORE the first capture. Any
# settling then happens off the record.
#
# Note the leave-one-out was flat — the best single capture to drop bought
# 0.048 deg — so this is not a bad-frame problem and cannot be fixed by
# discarding captures.

MIN_SAMPLES = 10                    # refuse to solve below this many captures


# ── acceptance thresholds (validate_calibration.py) ──────────────────────────

MAX_TRANS_ERR_MM = 3.0
MAX_ROT_ERR_DEG = 0.5
MAX_REPROJ_PX = 1.0


# ── layout ───────────────────────────────────────────────────────────────────

# One folder per capture session, so a re-calibration never mixes with the
# previous camera position. Files inside: images/, robot_poses.json,
# color_intrinsics.json, T_base_color.npy, T_gripper_board_ref.npy
SESSIONS_DIR = "sessions"
