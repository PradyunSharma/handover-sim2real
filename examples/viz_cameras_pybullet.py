#!/usr/bin/env python
"""PyBullet visualization of the fixed external cameras INSIDE the real handover scene.
Builds the sim (table + robot + posed object + hand), draws each camera's position, view
ray and FOV frustum as debug lines in the 3D world, and also renders what each camera
actually SEES. Needs a GPU + a display for the interactive GUI (run on the workstation,
not the cluster login node).

    python examples/viz_cameras_pybullet.py                         # interactive GUI (default)
    python examples/viz_cameras_pybullet.py --sim-cfg examples/pretrain_multicam.yaml
    python examples/viz_cameras_pybullet.py --cameras left right back
    python examples/viz_cameras_pybullet.py --snapshot out.png      # headless: save an overview + per-camera views, no GUI loop

Camera poses come from the handover config (CAMERA_{LEFT,RIGHT,BACK}_{POSITION,TARGET}).
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "GA-DDPG"),
           os.path.join(_HERE, "..", "handover-sim")):
    sys.path.insert(0, _p)
os.environ.setdefault("GADDPG_DIR", os.path.join(_HERE, "..", "GA-DDPG"))

import gym
import handover  # noqa: F401  (registers the env)
from handover.benchmark_wrapper import HandoverBenchmarkWrapper
from handover_sim2real.config import get_cfg

_COLORS = {"wrist": (1, 1, 0), "left": (0, 0.4, 1), "right": (1, 0.2, 0.2), "back": (0.1, 0.8, 0.1)}


def frustum_corners(eye, target, up, vfov_deg, aspect, depth):
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye; f /= (np.linalg.norm(f) + 1e-9)
    r = np.cross(f, up); r /= (np.linalg.norm(r) + 1e-9)
    u = np.cross(r, f)
    hh = depth * np.tan(np.deg2rad(vfov_deg) / 2.0); hw = hh * aspect
    c = eye + depth * f
    return [c + sx * hw * r + sy * hh * u for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def draw_camera(p, name, eye, target, corners, color):
    eye = list(map(float, eye)); target = list(map(float, target))
    p.addUserDebugLine(eye, target, color, 2.0)                     # view ray
    for c in corners:                                              # eye -> corners
        p.addUserDebugLine(eye, list(map(float, c)), color, 1.0)
    for a, b in zip(corners, corners[1:] + corners[:1]):           # far rectangle
        p.addUserDebugLine(list(map(float, a)), list(map(float, b)), color, 2.0)
    p.addUserDebugText(name, eye, color, 1.3)


def unwrap(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-cfg", default=os.path.join(_HERE, "pretrain_multicam.yaml"))
    ap.add_argument("--cameras", nargs="*", default=None, help="override which cameras to draw")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--snapshot", default=None, help="save an overview + per-camera views PNG and exit (no GUI loop)")
    args = ap.parse_args()

    cfg = get_cfg()
    if args.sim_cfg:
        from experiments.config import cfg_from_file
        cfg_from_file(filename=args.sim_cfg, dict=cfg, merge_to_cn_dict=True)
    cfg.ENV.ID = "HandoverHandCameraPointStateEnv-v1"     # base env (no OMG needed)
    cfg.BENCHMARK.SPLIT = "train"
    cfg.SIM.RENDER = args.snapshot is None                # GUI unless snapshot mode

    hcps = cfg.ENV.HANDOVER_HAND_CAMERA_POINT_STATE_ENV
    names = [c.lower() for c in (args.cameras if args.cameras else hcps.CAMERAS)]
    names = [n for n in names if n in ("left", "right", "back")]     # wrist is dynamic; skip
    up = tuple(hcps.EXTERNAL_CAMERA_UP); vfov = float(hcps.EXTERNAL_CAMERA_VERTICAL_FOV)
    aspect = float(hcps.EXTERNAL_CAMERA_WIDTH) / float(hcps.EXTERNAL_CAMERA_HEIGHT)
    # build exactly the selected fixed cameras (so their views can be rendered)
    cfg.ENV.HANDOVER_HAND_CAMERA_POINT_STATE_ENV.CAMERAS = list(names)

    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    env.reset(idx=args.scene)
    init = np.array(cfg.ENV.PANDA_INITIAL_POSITION, dtype=np.float32)
    for _ in range(10):
        obs, _, _, _ = env.step(init)
    p = unwrap(env).simulator._p

    for name in names:
        eye = np.array(getattr(hcps, "CAMERA_" + name.upper() + "_POSITION"), float)
        target = np.array(getattr(hcps, "CAMERA_" + name.upper() + "_TARGET"), float)
        depth = float(np.linalg.norm(target - eye))
        corners = frustum_corners(eye, target, up, vfov, aspect, depth)
        draw_camera(p, name, eye, target, corners, _COLORS.get(name, (1, 1, 1)))
        print(f"drew {name}: eye={tuple(np.round(eye,2))} -> target={tuple(np.round(target,2))}")

    if args.snapshot:
        # overview render of the scene, with the frustums PROJECTED onto the image
        # (pybullet debug lines don't show in getCameraImage, so we project ourselves).
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        W, H = 960, 680
        vm = p.computeViewMatrix((2.4, -1.5, 2.1), (0.4, 0.05, 1.05), (0, 0, 1))
        pm = p.computeProjectionMatrixFOV(55, W / H, 0.1, 8.0)
        rgb = np.reshape(p.getCameraImage(W, H, vm, pm)[2], (H, W, 4))[:, :, :3]
        V = np.array(vm).reshape(4, 4, order="F"); P = np.array(pm).reshape(4, 4, order="F")

        def proj(wpts):
            wpts = np.atleast_2d(np.asarray(wpts, float))
            clip = (P @ V @ np.c_[wpts, np.ones(len(wpts))].T).T
            ndc = clip[:, :3] / clip[:, 3:4]
            return np.c_[(ndc[:, 0] * 0.5 + 0.5) * W, (1 - (ndc[:, 1] * 0.5 + 0.5)) * H]

        n = len(names)
        fig, axes = plt.subplots(1, n + 1, figsize=(5.2 * (n + 1), 4.2))
        axes = np.atleast_1d(axes)
        axes[0].imshow(rgb); axes[0].set_title("scene + camera frustums"); axes[0].axis("off")
        axes[0].set_xlim(0, W); axes[0].set_ylim(H, 0)
        for name in names:
            eye = np.array(getattr(hcps, "CAMERA_" + name.upper() + "_POSITION"), float)
            target = np.array(getattr(hcps, "CAMERA_" + name.upper() + "_TARGET"), float)
            corners = frustum_corners(eye, target, up, vfov, aspect, float(np.linalg.norm(target - eye)))
            col = _COLORS.get(name, (1, 1, 1))
            pts = proj([eye] + corners + [target]); e, cs, tg = pts[0], pts[1:5], pts[5]
            axes[0].plot([e[0], tg[0]], [e[1], tg[1]], color=col, lw=1.4, ls="--")
            for c in cs:
                axes[0].plot([e[0], c[0]], [e[1], c[1]], color=col, lw=0.9)
            loop = np.vstack([cs, cs[0]]); axes[0].plot(loop[:, 0], loop[:, 1], color=col, lw=1.8)
            axes[0].scatter(e[0], e[1], color=col, s=45, zorder=5)
            axes[0].text(e[0], e[1], "  " + name, color=col, fontsize=11, weight="bold")
        inner = unwrap(env)
        for ax, name in zip(axes[1:], names):
            cam = inner._fixed_cameras[name]
            ax.imshow(cam._camera.color[0].numpy()[:, :, :3]); ax.set_title(f"{name} view"); ax.axis("off")
        fig.suptitle("Fixed cameras in the handover scene — left panel: positions + FOV; others: each camera's view", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(args.snapshot, dpi=110)
        print("saved:", os.path.abspath(args.snapshot))
        env.close(); return

    # interactive GUI
    p.resetDebugVisualizerCamera(cameraDistance=2.2, cameraYaw=50, cameraPitch=-30,
                                 cameraTargetPosition=(0.4, 0.0, 1.1))
    print("\nGUI open — rotate with the mouse. Press Enter here to exit.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    env.close()


if __name__ == "__main__":
    main()
