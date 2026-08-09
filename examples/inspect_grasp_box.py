"""
Interactive PyBullet viewer for the grasp-opportunity test (dagger/grasp_box.py).

Loads a real handover-sim scene — table, YCB object, MANO hand — plus a
FREE-FLOATING Panda gripper you can drag around with the mouse, and draws the
ray grid the metric actually casts. The terminal prints coverage whenever
something is between the fingers.

    python examples/inspect_grasp_box.py --scene 8

WHY A FLOATING GRIPPER. The real Panda is fixed-base, so PyBullet's mouse
picking cannot move it. A minimal hand+2-finger URDF is generated at startup
from the SAME meshes the real robot uses (franka_panda/meshes/collision), so the
geometry under test is identical while the body stays draggable. The real arm is
hidden and pushed out of the way.

NOTHING HERE IS A REIMPLEMENTATION. The rays come from `jaw_ray_hits`, the same
call the evaluator makes; the viewer only supplies the gripper pose and finger
gap through the `hand_pose` / `gap` overrides that already exist for testing. If
the drawing and the metric ever disagree, that is a real bug, not a viewer bug.

CONTROLS
    mouse drag        move the object, the hand, or the gripper (PyBullet's own
                      picking; hold and drag any body)
    sliders           gripper x/y/z/roll/pitch/yaw for precise placement,
                      finger opening, and the min_frac threshold
    snap_to_slider    flip to 1 to teleport the gripper to the slider pose
                      (leave at 0 to drag freely without the sliders fighting)
    reset_pose        flip to 1 to drop the gripper back at the pinned grasp

COLOURS
    green ray         first hit was the OBJECT  -> counts toward coverage
    red ray           first hit was the HUMAN HAND. Diagnostic only: the metric
                      leaves MANO invisible to rays, so these still count as
                      MISSES in `box_chance_rate`. Seeing red is exactly the
                      caveat that the geometric test over-reports opportunity.
    grey ray          hit nothing
    yellow box        the jaw cuboid the grid samples
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import pybullet  # noqa: E402

from handover_sim2real.dagger.env_setup import (  # noqa: E402
    build_sim_cfg, build_sim_context,
)
from handover_sim2real.dagger.grasp_pin import load_grasp_pin_table  # noqa: E402
from handover_sim2real.dagger.grasp_box import (  # noqa: E402
    build_box_params, jaw_ray_hits, hand_ray_block, _bullet_client,
)


# ── the floating gripper ──────────────────────────────────────────────────────

_GRIPPER_URDF = """<?xml version="1.0"?>
<robot name="floating_panda_gripper">
  <link name="panda_hand">
    <visual><geometry><mesh filename="{vis_hand}"/></geometry>
      <material name="w"><color rgba="0.85 0.85 0.88 1"/></material></visual>
    <collision><geometry><mesh filename="{col_hand}"/></geometry></collision>
    <inertial><mass value="0.73"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/></inertial>
  </link>
  <link name="panda_leftfinger">
    <visual><geometry><mesh filename="{vis_finger}"/></geometry>
      <material name="g"><color rgba="0.2 0.7 0.3 1"/></material></visual>
    <collision><geometry><mesh filename="{col_finger}"/></geometry></collision>
    <inertial><mass value="0.015"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>
  </link>
  <link name="panda_rightfinger">
    <visual><origin rpy="0 0 3.14159265359" xyz="0 0 0"/>
      <geometry><mesh filename="{vis_finger}"/></geometry>
      <material name="g2"><color rgba="0.2 0.7 0.3 1"/></material></visual>
    <collision><origin rpy="0 0 3.14159265359" xyz="0 0 0"/>
      <geometry><mesh filename="{col_finger}"/></geometry></collision>
    <inertial><mass value="0.015"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>
  </link>
  <joint name="finger_joint1" type="prismatic">
    <parent link="panda_hand"/><child link="panda_leftfinger"/>
    <origin rpy="0 0 0" xyz="0 0 0.0584"/><axis xyz="0 1 0"/>
    <limit effort="20" lower="0.0" upper="0.04" velocity="0.2"/>
  </joint>
  <joint name="finger_joint2" type="prismatic">
    <parent link="panda_hand"/><child link="panda_rightfinger"/>
    <origin rpy="0 0 0" xyz="0 0 0.0584"/><axis xyz="0 -1 0"/>
    <limit effort="20" lower="0.0" upper="0.04" velocity="0.2"/>
  </joint>
</robot>
"""


def write_gripper_urdf(out_dir: Path) -> Path:
    """Emit the hand+fingers URDF with ABSOLUTE mesh paths.

    Absolute so the file can live in a scratch dir: the real URDFs use
    `package://meshes/...`, which PyBullet resolves relative to the URDF's own
    directory and would break anywhere else.
    """
    meshes = (Path(_HERE).parent / "handover-sim" / "handover" / "data" / "assets"
              / "franka_panda" / "meshes")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "floating_panda_gripper.urdf"
    path.write_text(_GRIPPER_URDF.format(
        vis_hand=meshes / "visual" / "hand.obj",
        col_hand=meshes / "collision" / "hand.obj",
        vis_finger=meshes / "visual" / "finger.obj",
        col_finger=meshes / "collision" / "finger.obj",
    ))
    return path


# ── helpers ───────────────────────────────────────────────────────────────────

def hide_body(p, uid):
    """Make a body invisible without unloading it (the real arm)."""
    for link in range(-1, p.getNumJoints(uid)):
        p.changeVisualShape(uid, link, rgbaColor=[0, 0, 0, 0])


def go_limp(body):
    """Zero a body's actuator forces so PyBullet's mouse picking can move it.

    handover-sim position-controls the object and the hand along the recorded
    trajectory; with the motors still on, dragging fights the controller and the
    body snaps back the moment you let go.
    """
    try:
        body.dof_max_force = tuple([0.0] * len(body.get_attr_array("dof_max_force", 0)))
    except Exception:
        pass


def box_edges(box):
    """The 12 edges of the jaw cuboid in the hand frame, for drawing."""
    xs = (-box.half_x, box.half_x)
    ys = (-0.04 + box.inset, 0.04 - box.inset)
    zs = (box.z_lo, box.z_hi)
    corners = [(x, y, z) for x in xs for y in ys for z in zs]
    edges = []
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            if sum(1 for k in range(3) if abs(a[k] - b[k]) > 1e-9) == 1:
                edges.append((a, b))
    return edges


def to_world(pts, pos, quat):
    R = np.asarray(pybullet.getMatrixFromQuaternion(quat)).reshape(3, 3)
    return np.asarray(pts, dtype=np.float64) @ R.T + np.asarray(pos)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="examples/configs/dagger_phase4_reachw.yaml")
    ap.add_argument("--scene", type=int, default=None,
                    help="scene index; default = the first pinned scene")
    ap.add_argument("--scratch", default=None, help="where to write the gripper URDF")
    ap.add_argument("--hz", type=float, default=60.0)
    # Smoke-test path: same code, no window. Debug sliders read back their
    # defaults and debug lines are no-ops under DIRECT, so this exercises
    # everything except what a human would look at.
    ap.add_argument("--direct", action="store_true",
                    help="run headless (no GUI window) — for testing the script")
    ap.add_argument("--steps", type=int, default=0,
                    help="exit after N iterations instead of running forever")
    args = ap.parse_args()

    cfg4 = yaml.safe_load(open(args.config))
    sim_cfg = build_sim_cfg(cfg4["SIM"])
    sim_cfg.SIM.RENDER = not args.direct      # GUI unless --direct
    sim_cfg.SIM.BULLET.USE_EGL = False        # EGL is offscreen-only
    sim = build_sim_context(sim_cfg, cfg4["SIM"], seed=0)
    env = sim.env

    # Same call as dagger/setup.py — the pin table lives in the SIM block, and
    # match_tol/sim_cfg_block change which scenes resolve, so reading it any
    # other way would put the viewer on a different scene set than the metric.
    pin_table = load_grasp_pin_table(
        cfg4["SIM"].get("grasp_pin_table"),
        match_tol=float(cfg4["SIM"].get("grasp_pin_match_tol", 0.02)),
        sim_cfg_block=cfg4["SIM"])
    scene = args.scene
    if scene is None:
        scene = int(sorted(pin_table.entries)[0]) if pin_table else 0
    env.reset(idx=int(scene))
    p = _bullet_client(env)
    box = build_box_params(cfg4.get("EVAL", {}))

    # --- scene prep ---------------------------------------------------------
    p.setGravity(0, 0, 0)                     # nothing should fall while dragged
    hide_body(p, int(env.panda.body.contact_id[0]))
    go_limp(env.ycb.bodies[env.ycb.ids[0]])
    if getattr(env, "mano", None) is not None and env.mano.body is not None:
        go_limp(env.mano.body)

    # Temp dir by default: the generated URDF is a build artefact, not output,
    # and output/ is not ignored here so it would show up in git status.
    urdf = write_gripper_urdf(Path(args.scratch) if args.scratch
                              else Path(tempfile.gettempdir()) / "grasp_box_viewer")
    entry = pin_table.entries.get(int(scene)) if pin_table else None
    if entry is not None:
        G = np.asarray(entry["ee_pose_world"], dtype=np.float64)
        from transforms3d.quaternions import mat2quat
        w, x, y, z = mat2quat(G[:3, :3])
        start_pos, start_quat = G[:3, 3].tolist(), [x, y, z, w]
    else:
        ycb = env.ycb.bodies[env.ycb.ids[0]].link_state[0, 6, 0:3].numpy()
        start_pos, start_quat = (ycb + np.array([0, 0, 0.15])).tolist(), [1, 0, 0, 0]

    grip = p.loadURDF(str(urdf), start_pos, start_quat, useFixedBase=False)
    for link in range(-1, p.getNumJoints(grip)):
        p.changeDynamics(grip, link, linearDamping=8.0, angularDamping=8.0)
    # The gripper is a probe, not a participant: dragging it through the object
    # must not shove the object around, or the thing being measured moves as you
    # measure it. Done PER PAIR rather than by zeroing its group/mask, because
    # PyBullet's mouse picking selects bodies with a raycast — a group/mask of 0
    # would make the gripper unclickable, which is the one thing it has to be.
    for other in range(p.getNumBodies()):
        ouid = p.getBodyUniqueId(other)
        if ouid == grip:
            continue
        for la in range(-1, p.getNumJoints(grip)):
            for lb in range(-1, p.getNumJoints(ouid)):
                p.setCollisionFilterPair(grip, ouid, la, lb, 0)

    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=50,
                                 cameraPitch=-25, cameraTargetPosition=start_pos)

    # PERFORMANCE. Profiled on this scene, the per-frame COMPUTE is ~1 ms total
    # (raycast 0.26, hand diagnostic 0.51, 49 debug lines 0.09, sliders 0.01,
    # stepSimulation 0.17) — nowhere near a bottleneck. What actually costs is
    # GUI RENDERING: PyBullet opens RGB, depth and segmentation preview panes by
    # default and redraws all three every frame, on top of shadow casting, and
    # this box has Intel integrated graphics rendering a 52-link MANO hand. Kill
    # the panes and the shadows and the window becomes responsive.
    for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                 p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                 p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                 p.COV_ENABLE_SHADOWS):
        p.configureDebugVisualizer(flag, 0)

    # --- sliders ------------------------------------------------------------
    # Sliders keep their initial value alongside the id: under --direct there is
    # no GUI and readUserDebugParameter raises, so every read falls back to the
    # default and the loop still exercises the real code path.
    def slider(name, lo, hi, init):
        return (p.addUserDebugParameter(name, lo, hi, init), init)

    def read(entry):
        sid, init = entry
        try:
            return p.readUserDebugParameter(sid)
        except pybullet.error:
            return init

    rpy0 = p.getEulerFromQuaternion(start_quat)
    S = {
        "x": slider("x", start_pos[0] - 0.4, start_pos[0] + 0.4, start_pos[0]),
        "y": slider("y", start_pos[1] - 0.4, start_pos[1] + 0.4, start_pos[1]),
        "z": slider("z", start_pos[2] - 0.4, start_pos[2] + 0.4, start_pos[2]),
        "roll": slider("roll", -np.pi, np.pi, rpy0[0]),
        "pitch": slider("pitch", -np.pi, np.pi, rpy0[1]),
        "yaw": slider("yaw", -np.pi, np.pi, rpy0[2]),
        "finger": slider("finger_opening", 0.0, 0.04, 0.04),
        "min_frac": slider("min_frac", 0.0, 1.0, box.min_frac),
        "snap": slider("snap_to_slider", 0, 1, 0),
        "reset": slider("reset_pose", 0, 1, 0),
    }

    n_rays = int(box.grid_x) * int(box.grid_z)
    ray_ids = [p.addUserDebugLine([0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5])
               for _ in range(n_rays)]
    EDGES = box_edges(box)          # fixed in the hand frame — build once
    edge_ids = [p.addUserDebugLine([0, 0, 0], [0, 0, 0], [1, 1, 0])
                for _ in EDGES]
    text_id = p.addUserDebugText("", [0, 0, 0], textColorRGB=[1, 1, 1])

    print("=" * 78)
    print(f"scene {scene}   object={env.ycb.CLASSES.get(env.ycb.ids[0], '?')}")
    print(f"grid {box.grid_x}x{box.grid_z} = {n_rays} rays   min_frac={box.min_frac}"
          f"   open_thresh={box.open_thresh}")
    print("drag the object / hand / gripper with the mouse; sliders for precision")
    print("green=object  red=human hand (diagnostic, NOT counted)  grey=nothing")
    print("=" * 78)

    prev_reset = 0.0
    prev_draw = None
    last_print = (-1, -1)
    dt = 1.0 / args.hz
    it = 0
    try:
        while p.isConnected():
            it += 1
            if args.steps and it > args.steps:
                print(f"[--steps {args.steps}] done")
                break
            v = {k: read(e) for k, e in S.items()}

            if v["reset"] > 0.5 and prev_reset <= 0.5:
                p.resetBasePositionAndOrientation(grip, start_pos, start_quat)
                p.resetBaseVelocity(grip, [0, 0, 0], [0, 0, 0])
            prev_reset = v["reset"]

            if v["snap"] > 0.5:
                q = p.getQuaternionFromEuler([v["roll"], v["pitch"], v["yaw"]])
                p.resetBasePositionAndOrientation(grip, [v["x"], v["y"], v["z"]], q)
                p.resetBaseVelocity(grip, [0, 0, 0], [0, 0, 0])

            # Fingers are set KINEMATICALLY. Driving them with a motor makes the
            # free-floating base recoil (opening from the URDF's closed default
            # pushed the whole gripper off the grasp, and coverage drifted down
            # every frame). resetJointState applies no reaction force.
            gap = float(v["finger"])
            for j in (0, 1):
                p.resetJointState(grip, j, gap, targetVelocity=0.0)
                p.setJointMotorControl2(grip, j, p.POSITION_CONTROL,
                                        targetPosition=gap, force=0)

            pos, quat = p.getBasePositionAndOrientation(grip)
            src, dst, mask, frac = jaw_ray_hits(env, box, hand_pose=(pos, quat),
                                                gap=gap)
            hand_mask, hand_frac = hand_ray_block(env, box, hand_pose=(pos, quat),
                                                  gap=gap)

            # Redraw only when something actually moved: while you are reading
            # the numbers rather than dragging, this skips every debug-line call.
            moved = (prev_draw is None
                     or not np.allclose(pos, prev_draw[0], atol=1e-5)
                     or not np.allclose(quat, prev_draw[1], atol=1e-5)
                     or abs(gap - prev_draw[2]) > 1e-6)
            if moved and len(src):
                prev_draw = (np.array(pos), np.array(quat), gap)
                for k in range(n_rays):
                    if mask[k]:
                        col = [0.1, 0.9, 0.2]
                    elif len(hand_mask) and hand_mask[k]:
                        col = [0.9, 0.15, 0.15]
                    else:
                        col = [0.45, 0.45, 0.45]
                    p.addUserDebugLine(src[k].tolist(), dst[k].tolist(), col,
                                       lineWidth=2 if mask[k] else 1,
                                       replaceItemUniqueId=ray_ids[k])
                for eid, (a, b) in zip(edge_ids, EDGES):
                    aw, bw = to_world([a, b], pos, quat)
                    p.addUserDebugLine(aw.tolist(), bw.tolist(), [1, 1, 0],
                                       lineWidth=1, replaceItemUniqueId=eid)

            n_hit = int(mask.sum()) if len(mask) else 0
            opp = (frac >= v["min_frac"]) and (gap / 0.04 >= box.open_thresh)
            label = f"{frac*100:.1f}%  {'OPPORTUNITY' if opp else ''}"
            p.addUserDebugText(label, np.asarray(pos) + np.array([0, 0, 0.06]),
                               textColorRGB=[0.2, 1, 0.3] if opp else [1, 1, 1],
                               textSize=1.4, replaceItemUniqueId=text_id)

            # terminal: only on change, so it does not scroll at 60 Hz
            key = (n_hit, int(hand_mask.sum()) if len(hand_mask) else 0)
            if key != last_print:
                last_print = key
                if n_hit or key[1]:
                    flag = "  <-- OPPORTUNITY" if opp else ""
                    hand = (f"   hand blocks {key[1]}/{n_rays} "
                            f"({key[1]/n_rays*100:.1f}%)" if key[1] else "")
                    print(f"object in jaws: {n_hit}/{n_rays} rays = "
                          f"{frac*100:5.1f}%   open={gap/0.04:.2f}{hand}{flag}")
                else:
                    print("jaws empty")

            p.stepSimulation()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
