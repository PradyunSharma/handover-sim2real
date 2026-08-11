"""Shared helpers for the calibration scripts: SE(3), ChArUco, sessions, cameras.

Previously each script carried its own copy of the SE(3) helpers, the ChArUco
board construction and the board-pose detection. The two detectors had drifted
apart — validate_calibration.py passed camera intrinsics and CORNER_REFINE_NONE
to CharucoDetector while calibrate.py did not — so the validator was not scoring
quite the same measurements the solver had used. There is now one of each.

Run this file directly to list attached RealSense cameras.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

import calib_config as cfg

HERE = Path(__file__).resolve().parent


# ── SE(3) ────────────────────────────────────────────────────────────────────

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec.reshape(3))


def T_to_rvec_tvec(T: np.ndarray):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec, T[:3, 3].reshape(3, 1)


def T_to_R_t(T: np.ndarray):
    return T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def average_rotations(rotations) -> np.ndarray:
    """Markley quaternion averaging (largest eigenvector of sum q q^T)."""
    A = np.zeros((4, 4))
    q_ref = None
    for R in rotations:
        q = _R_to_quat_xyzw(R)
        if q_ref is None:
            q_ref = q.copy()
        if np.dot(q, q_ref) < 0:
            q = -q
        A += np.outer(q, q)
    _, vecs = np.linalg.eigh(A)
    q = vecs[:, -1]
    return _quat_xyzw_to_R(-q if q[3] < 0 else q)


def _R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    # scipy is available in this env; use it rather than a hand-rolled branchy
    # conversion (the previous copy of which lived only in validate_*.py).
    from scipy.spatial.transform import Rotation as Rot
    return np.asarray(Rot.from_matrix(R).as_quat(), dtype=np.float64)


def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(np.asarray(q) / np.linalg.norm(q)).as_matrix()


# ── ChArUco ──────────────────────────────────────────────────────────────────

_DICTS = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}


def build_board(spec=None):
    """(board, dictionary) from a BoardSpec — defaults to cfg.BOARD."""
    spec = spec or cfg.BOARD
    if spec.dictionary not in _DICTS:
        raise ValueError(f"Unknown aruco dictionary {spec.dictionary!r}")
    dictionary = cv2.aruco.getPredefinedDictionary(_DICTS[spec.dictionary])
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )
    if spec.legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board, dictionary


def detect_board(image_bgr, K, D, board, dictionary,
                 min_corners: int = None):
    """Board pose in the camera frame, plus the correspondences used.

    THE one detector — solver and validator both call this, so the validation
    scores exactly the measurements the solve consumed.

    Returns a dict with T_cam_board / charuco_corners / charuco_ids /
    obj_points / img_points / num_charuco, or None if too few corners.
    """
    min_corners = cfg.MIN_CHARUCO_CORNERS if min_corners is None else min_corners
    if not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError(
            "cv2.aruco.CharucoDetector is missing — OpenCV >= 4.7 is required "
            f"(found {cv2.__version__})."
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.cameraMatrix = K
    charuco_params.distCoeffs = D
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < min_corners:
        return None

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    return {
        "T_cam_board": rvec_tvec_to_T(rvec, tvec),
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "obj_points": obj_points,
        "img_points": img_points,
        "num_charuco": int(len(charuco_ids)),
    }


def board_tilt_deg(T_cam_board: np.ndarray) -> float:
    """Angle between the board's normal and the camera's line of sight to it.

    0 deg = the board is square-on to the camera, which is the WORST case. A
    planar target viewed head-on barely constrains its own out-of-plane
    rotation — the classic planar pose ambiguity — so sub-pixel corner noise
    turns into degrees of pose error, and hand-eye inherits it.

    Measured on the `test` session, rotation residual against tilt:
        tilt >= 39 deg  ->  0.13-0.23 deg
        tilt <= 28 deg  ->  0.39-1.99 deg
    Re-solving on tilt >= 30 alone cut the rotation residual from 0.658 to
    0.173 deg using fewer than half the captures.
    """
    t = T_cam_board[:3, 3]
    n = np.linalg.norm(t)
    if n < 1e-9:
        return 0.0
    cos = abs(float(T_cam_board[:3, 2] @ (t / n)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def hand_eye_method(name: str = None) -> int:
    name = (name or cfg.HAND_EYE_METHOD).upper()
    attr = f"CALIB_HAND_EYE_{name}"
    if not hasattr(cv2, attr):
        raise ValueError(f"Unknown hand-eye method {name!r}")
    return getattr(cv2, attr)


def solve_eye_to_hand(T_base_gripper_list, T_cam_board_list, method=None):
    """T_base_cam for a FIXED camera with the board on the gripper.

    Eye-to-hand, not eye-in-hand. OpenCV's calibrateHandEye solves for
    cam2gripper; feeding it the INVERTED robot poses (base2gripper) makes the
    result cam2base instead. That swap is the whole difference between the two
    problems, and getting it wrong returns a plausible, wrong matrix.
    """
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for T_bg, T_cb in zip(T_base_gripper_list, T_cam_board_list):
        R, t = T_to_R_t(invert_T(T_bg))
        R_g2b.append(R)
        t_g2b.append(t)
        R, t = T_to_R_t(T_cb)
        R_t2c.append(R)
        t_t2c.append(t)

    R_out, t_out = cv2.calibrateHandEye(
        R_gripper2base=R_g2b, t_gripper2base=t_g2b,
        R_target2cam=R_t2c, t_target2cam=t_t2c,
        method=hand_eye_method(method),
    )
    return make_T(R_out, t_out.reshape(3))


# ── sessions ─────────────────────────────────────────────────────────────────

class Session:
    """One capture session: its own images/ and its own outputs.

    Keeping each set of 15-20 captures in its own folder is what stops a
    re-calibration from silently blending with the previous camera position —
    the capture script appends, so shared folders accumulate across setups.
    """

    def __init__(self, name: str):
        self.name = name
        self.root = HERE / cfg.SESSIONS_DIR / name

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def poses_json(self) -> Path:
        return self.root / "robot_poses.json"

    @property
    def intrinsics_json(self) -> Path:
        return self.root / "color_intrinsics.json"

    @property
    def T_base_color(self) -> Path:
        return self.root / "T_base_color.npy"

    @property
    def T_gripper_board(self) -> Path:
        return self.root / "T_gripper_board_ref.npy"

    def create(self) -> "Session":
        self.images.mkdir(parents=True, exist_ok=True)
        return self

    def require(self, *paths: Path) -> None:
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(
                f"Session {self.name!r} is missing:\n  "
                + "\n  ".join(str(p.relative_to(HERE)) for p in missing)
            )

    def __str__(self) -> str:
        return self.name


def list_sessions() -> list[str]:
    root = HERE / cfg.SESSIONS_DIR
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def resolve_session(name: str | None, create: bool = False) -> Session:
    """Named session; otherwise today's (for capture) or the newest (for the rest)."""
    if name:
        s = Session(name)
        return s.create() if create else s

    existing = list_sessions()
    if create:
        return Session(date.today().isoformat()).create()
    if not existing:
        raise SystemExit(
            f"No sessions under {cfg.SESSIONS_DIR}/. Run capture_image_and_pose.py first."
        )
    newest = max((Session(n) for n in existing), key=lambda s: s.root.stat().st_mtime)
    print(f"[session] {newest.name} (newest; pass --session to choose another)")
    return newest


def add_session_arg(parser) -> None:
    parser.add_argument("--session", type=str, default=None,
                        help="capture-session folder under sessions/. Defaults to "
                             "today's date when capturing, else the newest one.")


# ── cameras ──────────────────────────────────────────────────────────────────

def attached_cameras() -> list[dict]:
    import pyrealsense2 as rs
    out = []
    for d in rs.context().query_devices():
        out.append({
            "name": d.get_info(rs.camera_info.name),
            "serial": d.get_info(rs.camera_info.serial_number),
            "usb": d.get_info(rs.camera_info.usb_type_descriptor),
        })
    return out


def resolve_serial(explicit: str | None = None, role: str | None = None) -> str:
    """Pick a camera: explicit serial > configured role > the only one attached.

    Never guesses between two attached cameras. Calibrating the wrist camera
    while aiming the tripod at the board produces no error anywhere, just a
    confidently wrong matrix.
    """
    devices = attached_cameras()
    if not devices:
        raise SystemExit("No RealSense devices attached.")
    serials = {d["serial"] for d in devices}
    listing = "\n  ".join(f"{d['serial']}  {d['name']}  usb {d['usb']}" for d in devices)

    if explicit:
        if explicit not in serials:
            raise SystemExit(f"Serial {explicit} not attached. Attached:\n  {listing}")
        return explicit

    role = role or cfg.DEFAULT_ROLE
    configured = cfg.CAMERA_SERIALS.get(role)
    if configured is not None:
        # Tolerate an unquoted serial in the config. It cannot match the string
        # librealsense reports, and a serial with a leading zero would lose it.
        configured = str(configured)
        if configured not in serials:
            padded = configured.zfill(len(next(iter(serials))))
            if padded in serials:
                print(f"[camera] note: CAMERA_SERIALS[{role!r}] is unquoted; "
                      f"read as {padded}. Quote it in calib_config.py.")
                configured = padded
    if configured:
        if configured not in serials:
            raise SystemExit(
                f"Configured {role} serial {configured} is not attached. Attached:\n  {listing}")
        print(f"[camera] {role} = {configured} (from calib_config.CAMERA_SERIALS)")
        return configured

    if len(devices) == 1:
        only = devices[0]["serial"]
        print(f"[camera] using the only attached device: {only}")
        return only

    raise SystemExit(
        f"{len(devices)} cameras attached and no serial for role {role!r}.\n  {listing}\n"
        f"Set CAMERA_SERIALS[{role!r}] in calib_config.py, or pass --serial. "
        "Jog the robot to tell them apart — the wrist one is the view that moves."
    )


def add_camera_args(parser) -> None:
    parser.add_argument("--serial", type=str, default=None,
                        help="RealSense serial. Overrides the configured role.")
    parser.add_argument("--role", type=str, default=cfg.DEFAULT_ROLE,
                        choices=sorted(cfg.CAMERA_SERIALS),
                        help="which configured camera to use")


# Colour-only, cheapest first. Identification needs no depth, and dropping it
# roughly halves the bandwidth — which is what lets a camera stuck on a USB 2
# link show a picture at all.
_PREVIEW_MODES = [(640, 480, 30), (640, 480, 15), (424, 240, 30),
                  (424, 240, 15), (640, 480, 6), (424, 240, 6)]


def preview_cameras(seconds: float | None = None) -> None:
    """Live window per attached camera, serial burnt into the image.

    The point is to answer "which of these is on the tripod?" without guessing.
    Jog the robot: the wrist camera is the view that moves. Then put the serials
    into calib_config.CAMERA_SERIALS.
    """
    import pyrealsense2 as rs

    devices = attached_cameras()
    if not devices:
        raise SystemExit("No RealSense devices attached.")

    streams = []
    for i, dev in enumerate(devices):
        for w, h, fps in _PREVIEW_MODES:
            pipe = rs.pipeline()
            conf = rs.config()
            conf.enable_device(dev["serial"])
            conf.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            try:
                pipe.start(conf)
                pipe.wait_for_frames(timeout_ms=3000)
            except Exception as err:
                try:
                    pipe.stop()
                except Exception:
                    pass
                last = err
                continue
            title = f"{dev['serial']}  ({dev['name']}, usb {dev['usb']})"
            streams.append({"pipe": pipe, "dev": dev, "title": title, "mode": (w, h, fps)})
            print(f"[preview] {dev['serial']}: streaming {w}x{h}@{fps}  usb {dev['usb']}")
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 640, 480)
            cv2.moveWindow(title, 40 + 680 * i, 60)
            break
        else:
            print(f"[preview] {dev['serial']}: NO MODE WOULD STREAM (usb {dev['usb']}) "
                  f"— {type(last).__name__}: {last}")
            if dev["usb"].startswith("2"):
                print("           a D435 on a USB 2 link often advertises modes it "
                      "cannot stream; reseat it on a USB3 port with the Intel cable.")

    if not streams:
        raise SystemExit("No camera would stream — nothing to preview.")

    print("\nJog the robot: the WRIST camera is the view that moves.")
    print("Press 'q' or Esc in any window to quit.\n")

    t0 = time.time()
    try:
        while True:
            for s in streams:
                frames = s["pipe"].poll_for_frames()
                cf = frames.get_color_frame() if frames else None
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data()).copy()
                w, h, fps = s["mode"]
                for j, (text, scale, thick) in enumerate((
                        (s["dev"]["serial"], 1.0, 3),
                        (f"{s['dev']['name']}  usb {s['dev']['usb']}  {w}x{h}@{fps}", 0.5, 1))):
                    y = 36 + 30 * j
                    # black underlay so the text stays readable on any scene
                    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
                    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (0, 255, 0), thick, cv2.LINE_AA)
                cv2.imshow(s["title"], img)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            if seconds is not None and time.time() - t0 >= seconds:
                break
    finally:
        for s in streams:
            try:
                s["pipe"].stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


# ── intrinsics / poses IO ────────────────────────────────────────────────────

def load_intrinsics(path: Path):
    data = json.loads(Path(path).read_text())
    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K, D, data


def load_robot_poses(path: Path):
    items = json.loads(Path(path).read_text())
    names, Ts = [], []
    for it in items:
        T = np.array(it["T_base_gripper"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Bad pose shape for {it['image']}: {T.shape}")
        names.append(it["image"])
        Ts.append(T)

    # A repeated image name means two different robot poses claim the same
    # picture, so at least one pair is wrong — and hand-eye has no robustness to
    # that. It happens when images/ is cleared to start over but the append-only
    # JSON is left behind: capture restarts numbering at 0001, overwrites the old
    # pictures and appends new poses. Observed once, and it put the solved camera
    # 8.5 m from the base (4966 mm residual, versus 3.3 mm after removing the
    # five stale entries). Refuse rather than solve garbage.
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit(
            f"{Path(path).name} references {len(dupes)} image(s) more than once: "
            f"{', '.join(dupes[:5])}{' ...' if len(dupes) > 5 else ''}\n"
            "Two poses cannot share one image — the pairing is broken.\n"
            "Most likely images/ was cleared without clearing robot_poses.json.\n"
            "Fix: keep only the LAST entry per image name (the run whose pictures "
            "survived), or re-capture into a fresh --session."
        )
    return names, Ts


def check_session_pairing(session: "Session") -> list[str]:
    """Problems with the image/pose pairing, as human-readable strings."""
    problems = []
    on_disk = {p.name for p in session.images.glob("*.png")} if session.images.is_dir() else set()

    referenced: list[str] = []
    if session.poses_json.exists():
        referenced = [x["image"] for x in json.loads(session.poses_json.read_text())]

    dupes = sorted({n for n in referenced if referenced.count(n) > 1})
    if dupes:
        problems.append(f"{len(dupes)} image name(s) appear twice in robot_poses.json: "
                        f"{', '.join(dupes[:5])}")
    missing = sorted(set(referenced) - on_disk)
    if missing:
        problems.append(f"{len(missing)} pose(s) reference missing images: "
                        f"{', '.join(missing[:5])}")
    orphan = sorted(on_disk - set(referenced))
    if orphan:
        problems.append(f"{len(orphan)} image(s) have no pose: {', '.join(orphan[:5])}")
    return problems


def load_session_samples(session: Session):
    """Detect the board in every captured image. Returns (names, T_base_gripper, results)."""
    session.require(session.intrinsics_json, session.poses_json)
    K, D, meta = load_intrinsics(session.intrinsics_json)
    if (meta.get("width"), meta.get("height")) != (cfg.STREAM.width, cfg.STREAM.height):
        print(f"[warn] intrinsics are {meta.get('width')}x{meta.get('height')} but "
              f"calib_config.STREAM is {cfg.STREAM.width}x{cfg.STREAM.height}. "
              "Intrinsics are resolution-specific.")
    board, dictionary = build_board()
    names, poses = load_robot_poses(session.poses_json)

    ok_names, ok_poses, results = [], [], []
    for name, T in zip(names, poses):
        img = cv2.imread(str(session.images / name))
        if img is None:
            print(f"  skip {name}: image not found")
            continue
        res = detect_board(img, K, D, board, dictionary)
        if res is None:
            print(f"  skip {name}: board not detected")
            continue
        ok_names.append(name)
        ok_poses.append(T)
        results.append(res)

    print(f"detected the board in {len(ok_names)}/{len(names)} images")
    return ok_names, ok_poses, results, K, D


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="List attached RealSense cameras, or preview them live to see "
                    "which serial is which.")
    ap.add_argument("--preview", action="store_true",
                    help="open a live window per camera with its serial overlaid")
    ap.add_argument("--seconds", type=float, default=None,
                    help="auto-close the preview after this long")
    a = ap.parse_args()

    devs = attached_cameras()
    print(f"{len(devs)} RealSense device(s) attached:")
    for d in devs:
        print(f"  {d['serial']}  {d['name']}  usb {d['usb']}")
    print(f"\nconfigured roles: {cfg.CAMERA_SERIALS}")
    print(f"sessions: {list_sessions() or '(none)'}")

    if a.preview:
        print()
        preview_cameras(seconds=a.seconds)
    sys.exit(0)
