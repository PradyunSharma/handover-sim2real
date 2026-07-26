#!/usr/bin/env python
"""Visualize the FIXED external camera placement for the handover scene — NO sim / NO
CUDA needed (pure matplotlib + the config + the look-at math). Draws the table, panda
base, handover region, goal, and each external camera's eye + view ray + FOV frustum, so
you can eyeball coverage and tune the camera poses BEFORE collecting demos.

Run here:
    python examples/viz_external_cameras.py
    python examples/viz_external_cameras.py --sim-cfg examples/pretrain_multicam.yaml
    python examples/viz_external_cameras.py --cameras left right          # subset
Poses come from the handover config (CAMERA_{LEFT,RIGHT,BACK}_{POSITION,TARGET}); pass a
--sim-cfg to reflect any overrides in that yaml. Saves a PNG (multi-view) and also opens
an interactive window if a display is available. Edit the poses, re-run, repeat.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "handover-sim"))
sys.path.insert(0, os.path.join(_HERE, "..", "GA-DDPG"))   # for experiments.config (cfg_from_file)
os.environ.setdefault("GADDPG_DIR", os.path.join(_HERE, "..", "GA-DDPG"))

import matplotlib                               # default backend -> opens a GUI window when
import matplotlib.pyplot as plt                 # a display is available (set --save for a PNG)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection  # noqa: F401
from scipy.spatial.transform import Rotation as Rot

from handover.multicam import look_at_phys_quat
from handover_sim2real.config import get_cfg

_COLORS = {"left": "tab:blue", "right": "tab:red", "back": "tab:green"}


def frustum(eye, target, up, vfov_deg, aspect=1.0, depth=None):
    """Return (eye, 4 far corners) of the view frustum. depth defaults to |eye->target|
    so the far rectangle sits at the handover point."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    if depth is None:
        depth = np.linalg.norm(target - eye)
    R = Rot.from_quat(look_at_phys_quat(eye, target, up)).as_matrix()
    right, down, fwd = R[:, 0], R[:, 1], R[:, 2]     # +X right, +Y down, +Z forward
    hh = depth * np.tan(np.deg2rad(vfov_deg) / 2.0)  # half-height at 'depth'
    hw = hh * aspect
    center = eye + depth * fwd
    corners = np.array([center + sx * hw * right + sy * hh * down
                        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    return eye, corners


def draw_camera_3d(ax, name, eye, corners, target, color):
    ax.scatter(*eye, s=60, color=color, marker="^")
    ax.text(*eye, "  " + name, color=color, fontsize=9)
    # frustum edges eye->corners + far rectangle
    for c in corners:
        ax.plot(*zip(eye, c), color=color, lw=0.8, alpha=0.7)
    loop = np.vstack([corners, corners[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, lw=1.2, alpha=0.9)
    ax.plot(*zip(eye, target), color=color, lw=1.5, ls="--", alpha=0.9)  # view ray


def draw_camera_2d(ax, ai, bi, eye, corners, target, color):
    ax.scatter(eye[ai], eye[bi], s=50, color=color, marker="^", zorder=3)
    loop = np.vstack([corners, corners[0]])
    ax.plot(loop[:, ai], loop[:, bi], color=color, lw=1.0, alpha=0.8)
    for c in corners:
        ax.plot([eye[ai], c[ai]], [eye[bi], c[bi]], color=color, lw=0.5, alpha=0.5)
    ax.plot([eye[ai], target[ai]], [eye[bi], target[bi]], color=color, lw=1.3, ls="--")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-cfg", default=None, help="optional yaml to merge (reflects pose overrides)")
    ap.add_argument("--cameras", nargs="*", default=["left", "right", "back"],
                    help="which fixed cameras to draw (default: all three)")
    ap.add_argument("--save", default=None,
                    help="save a PNG here instead of opening a window (default: open window)")
    args = ap.parse_args()

    cfg = get_cfg()
    if args.sim_cfg:
        from experiments.config import cfg_from_file
        cfg_from_file(filename=args.sim_cfg, dict=cfg, merge_to_cn_dict=True)

    hcps = cfg.ENV.HANDOVER_HAND_CAMERA_POINT_STATE_ENV
    up = tuple(hcps.EXTERNAL_CAMERA_UP)
    vfov = float(hcps.EXTERNAL_CAMERA_VERTICAL_FOV)
    aspect = float(hcps.EXTERNAL_CAMERA_WIDTH) / float(hcps.EXTERNAL_CAMERA_HEIGHT)
    selected = [c.lower() for c in hcps.CAMERAS]      # what the run will actually use

    # scene reference points
    table_base = np.array(cfg.ENV.TABLE_BASE_POSITION, float)
    table_h = float(cfg.ENV.TABLE_HEIGHT)
    panda_base = np.array(cfg.ENV.PANDA_BASE_POSITION, float)
    goal = np.array(cfg.BENCHMARK.GOAL_CENTER, float)
    # MEASURED DexYCB object (handover) pose distribution — mean & std over a sample of
    # train scenes (read from the sim, scratchpad/dbg_pos2.py). This is the REAL region
    # the cameras must cover; ~straight in front (+Y) of the panda base.
    handover = np.array([0.516, 0.248, 1.227])
    handover_std = np.array([0.075, 0.109, 0.063])

    cams = {}
    print(f"vfov={vfov}  aspect={aspect:.2f}  handover~{tuple(np.round(handover,2))}  table_top_z={table_h}")
    for name in args.cameras:
        name = name.lower()
        eye = np.array(getattr(hcps, "CAMERA_" + name.upper() + "_POSITION"), float)
        target = np.array(getattr(hcps, "CAMERA_" + name.upper() + "_TARGET"), float)
        e, corners = frustum(eye, target, up, vfov, aspect)
        cams[name] = (eye, target, corners)
        dist = np.linalg.norm(target - eye)
        elev = np.degrees(np.arcsin((eye[2] - target[2]) / (dist + 1e-9)))
        tag = "  [SELECTED]" if name in selected else "  (available, not selected)"
        print(f"  {name:5s} eye={tuple(np.round(eye,2))} -> target={tuple(np.round(target,2))} | "
              f"dist={dist:.2f}m height_above_table={eye[2]-table_h:.2f}m elevation={elev:.0f}deg{tag}")

    # table top rectangle (approx 1.0 x 1.2 around the table base)
    tx, ty = table_base[0], table_base[1]
    tw, td = 0.5, 0.6
    table_poly = np.array([[tx - tw, ty - td, table_h], [tx + tw, ty - td, table_h],
                           [tx + tw, ty + td, table_h], [tx - tw, ty + td, table_h]])

    fig = plt.figure(figsize=(16, 5.5))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 3, 2)
    ax_yz = fig.add_subplot(1, 3, 3)

    # 3D
    ax3d.add_collection3d(Poly3DCollection([table_poly], alpha=0.15, facecolor="gray"))
    ax3d.scatter(*panda_base, s=80, color="black", marker="s"); ax3d.text(*panda_base, "  panda base")
    ax3d.scatter(*handover, s=120, color="orange", marker="*"); ax3d.text(*handover, "  handover")
    ax3d.scatter(*goal, s=60, color="purple", marker="P"); ax3d.text(*goal, "  goal")
    for name, (eye, target, corners) in cams.items():
        draw_camera_3d(ax3d, name, eye, corners, target, _COLORS.get(name, "gray"))
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z"); ax3d.set_title("3D (iso)")
    ax3d.view_init(elev=22, azim=-60)
    try:
        ax3d.set_box_aspect((1, 1, 0.6))
    except Exception:
        pass

    # 2D projections
    for ax, (ai, bi, la, lb, title) in ((ax_xy, (0, 1, "x", "y", "TOP-DOWN (x-y)")),
                                        (ax_yz, (1, 2, "y", "z", "SIDE (y-z, elevation)"))):
        rect = np.vstack([table_poly, table_poly[0]])
        ax.fill(rect[:, ai], rect[:, bi], color="gray", alpha=0.15)
        ax.scatter(panda_base[ai], panda_base[bi], color="black", marker="s", s=60, label="panda base")
        ax.scatter(handover[ai], handover[bi], color="orange", marker="*", s=140, label="handover (obj mean)")
        # ±2σ region the object actually occupies across scenes — cameras must cover this
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((handover[ai] - 2 * handover_std[ai], handover[bi] - 2 * handover_std[bi]),
                               4 * handover_std[ai], 4 * handover_std[bi],
                               fill=False, ec="orange", ls="--", lw=1.3, label="handover ±2σ"))
        ax.scatter(goal[ai], goal[bi], color="purple", marker="P", s=50, label="goal (retreat)")
        for name, (eye, target, corners) in cams.items():
            draw_camera_2d(ax, ai, bi, eye, corners, target, _COLORS.get(name, "gray"))
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
    ax_xy.legend(loc="best", fontsize=8)

    handles = [plt.Line2D([0], [0], color=_COLORS.get(n, "gray"), marker="^", lw=1.2,
                          label=n + (" [SELECTED]" if n in selected else ""))
               for n in cams]
    fig.legend(handles=handles, loc="lower center", ncol=len(cams), fontsize=9)
    fig.suptitle("External camera placement — dashed = view ray, outline = FOV at handover depth", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fig.savefig(args.save, dpi=130)
        print(f"\nsaved: {os.path.abspath(args.save)}")
    else:
        print("\nopening matplotlib window — close it to exit.")
        plt.show()


if __name__ == "__main__":
    main()
