"""
Visualize a YCB object together with its grasp candidates from the
OMG-Planner data directory.

Two grasp sources are shown when available:
  - simulated/<object>.npy   (sim-sampled, ACRONYM-style; the set OMG actually
                              loads via planner.py:load_grasp_set)
  - graspit/<object>_grasp_pose.txt (older, human-authored GraspIt! set)

Each grasp is drawn as a parallel-jaw gripper stick figure
(approach axis = +z, finger axis = ±y).

Usage:
    python examples/visualize_grasps.py --object 025_mug
    python examples/visualize_grasps.py --object 025_mug --max-simulated 500 --max-graspit 50
    python examples/visualize_grasps.py --object 025_mug --apply-omg-transform
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMG_DATA = os.path.join(REPO, "OMG-Planner", "data")


# ── loaders ──────────────────────────────────────────────────────────────────

def load_obj_vertices(path):
    """Return (V, 3) ndarray of vertex positions from an OBJ file."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32)


def load_simulated_grasps(object_name):
    """Return (N, 4, 4) grasp poses in object frame, or None if missing.

    Some .npy files were pickled under Python 2 and need encoding="bytes" —
    same fallback OMG uses in planner.py:load_grasp_set.
    """
    path = os.path.join(OMG_DATA, "grasps", "simulated", f"{object_name}.npy")
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True).item()
        transforms = data["transforms"]
    except (UnicodeError, UnicodeDecodeError):
        data = np.load(path, allow_pickle=True, fix_imports=True,
                       encoding="bytes").item()
        transforms = data[b"transforms"]
    return np.asarray(transforms, dtype=np.float32)


def load_graspit_grasps(object_name):
    """
    Return (M, 4, 4) grasp poses in object frame, or None if missing.
    The graspit text file stores 16 numbers per line (row-major 4x4) followed
    by two extra metric values. Translation is in millimetres -> convert to m.
    """
    path = os.path.join(OMG_DATA, "grasps", "graspit", f"{object_name}_grasp_pose.txt")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split(",") if v.strip()]
            if len(vals) < 16:
                continue
            T = np.array(vals[:16], dtype=np.float32).reshape(4, 4)
            T[:3, 3] *= 0.001  # mm -> m
            rows.append(T)
    return np.stack(rows, axis=0) if rows else None


# ── gripper drawing ──────────────────────────────────────────────────────────

def gripper_segments(T, hand_depth=0.04, finger_z_root=0.058, finger_z_tip=0.112,
                     width=0.08):
    """
    Return line segments for a Franka Panda parallel-jaw gripper at pose T.
    Convention:
      origin (T[:3,3]) = panda_hand link  (where the saved .npy grasps anchor)
      +z = approach direction
      ±y = finger axis (gripper closes along y)
    Default dimensions follow the Panda URDF:
      • back of hand body at z=-hand_depth
      • finger bases at z=finger_z_root   (~5.8 cm forward of panda_hand)
      • finger tips  at z=finger_z_tip    (~11.2 cm forward of panda_hand)
      • max opening width=0.08 m
    """
    half = width / 2.0
    pts = {
        "wrist":      np.array([0.0,  0.0,   -hand_depth,    1.0]),
        "palm":       np.array([0.0,  0.0,    0.0,           1.0]),
        "left_base":  np.array([0.0, +half,   finger_z_root, 1.0]),
        "right_base": np.array([0.0, -half,   finger_z_root, 1.0]),
        "left_tip":   np.array([0.0, +half,   finger_z_tip,  1.0]),
        "right_tip":  np.array([0.0, -half,   finger_z_tip,  1.0]),
    }
    pw = {k: (T @ p)[:3] for k, p in pts.items()}
    return [
        (pw["wrist"],      pw["palm"]),        # back of hand body
        (pw["palm"],       pw["left_base"]),   # palm → left finger root
        (pw["palm"],       pw["right_base"]),  # palm → right finger root
        (pw["left_base"],  pw["right_base"]),  # bridge across finger bases
        (pw["left_base"],  pw["left_tip"]),    # left finger
        (pw["right_base"], pw["right_tip"]),   # right finger
    ]


def draw_grasps(ax, grasps, color, label, max_grasps=None, alpha=0.4):
    if grasps is None or len(grasps) == 0:
        return
    if max_grasps is not None and len(grasps) > max_grasps:
        idx = np.random.choice(len(grasps), size=max_grasps, replace=False)
        grasps = grasps[idx]
    labelled = False
    for T in grasps:
        for p, q in gripper_segments(T):
            ax.plot(
                [p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                color=color, alpha=alpha, lw=0.8,
                label=None if labelled else label,
            )
            labelled = True


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object",     required=True,
                    help="YCB object id, e.g. 025_mug or 005_tomato_soup_can")
    ap.add_argument("--max-simulated", type=int, default=100,
                    help="max simulated grasps to render (random subsample); "
                         "use a large number to see all")
    ap.add_argument("--max-graspit",   type=int, default=100,
                    help="max graspit (human-authored) grasps to render "
                         "(random subsample); use a large number to see all")
    ap.add_argument("--apply-omg-transform", action="store_true",
                    help="post-multiply simulated grasps by rotZ(pi/2). "
                         "This matches the offset OMG applies in "
                         "planner.py:load_grasp_set before using them.")
    ap.add_argument("--seed",       type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    mesh_path = os.path.join(OMG_DATA, "objects", args.object, "model_normalized.obj")
    if not os.path.exists(mesh_path):
        raise SystemExit(f"Mesh not found: {mesh_path}")
    verts = load_obj_vertices(mesh_path)
    print(f"Object       : {args.object}")
    print(f"  mesh verts : {len(verts)}")
    print(f"  extent (m) : {verts.ptp(axis=0)}")

    sim = load_simulated_grasps(args.object)
    git = load_graspit_grasps(args.object)
    n_sim = 0 if sim is None else len(sim)
    n_git = 0 if git is None else len(git)
    print(f"  simulated  : {n_sim} grasps")
    print(f"  graspit    : {n_git} grasps")

    if args.apply_omg_transform and sim is not None:
        # rotZ(+90°) post-multiplied — matches the offset applied at
        # OMG-Planner/omg/planner.py:493-494 before the planner uses the
        # grasps. Use this if you want to see them in OMG's working frame.
        Rz = np.array([
            [ 0.0, -1.0, 0.0, 0.0],
            [ 1.0,  0.0, 0.0, 0.0],
            [ 0.0,  0.0, 1.0, 0.0],
            [ 0.0,  0.0, 0.0, 1.0],
        ], dtype=np.float32)
        sim = np.matmul(sim, Rz)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Object as a downsampled point cloud (mesh is dense; keep it light).
    n_show = min(len(verts), 3000)
    idx = np.random.choice(len(verts), size=n_show, replace=False)
    ax.scatter(verts[idx, 0], verts[idx, 1], verts[idx, 2],
               c="0.4", s=1, alpha=0.35, label=f"{args.object} mesh")

    draw_grasps(ax, sim, color="tab:orange",
                label=f"simulated ({n_sim})", max_grasps=args.max_simulated)
    draw_grasps(ax, git, color="tab:blue",
                label=f"graspit ({n_git})",   max_grasps=args.max_graspit)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{args.object} — object + grasp poses "
                 f"(simulated≤{args.max_simulated}, graspit≤{args.max_graspit})")
    ax.legend(loc="upper left", fontsize=9)

    # Equal-ish axes around the union of mesh + grasp origins.
    pts_for_bounds = [verts]
    if sim is not None:
        pts_for_bounds.append(sim[:, :3, 3])
    if git is not None:
        pts_for_bounds.append(git[:, :3, 3])
    P   = np.vstack(pts_for_bounds)
    mid = P.mean(axis=0)
    rng = (P.max(axis=0) - P.min(axis=0)).max() * 0.6 + 0.05
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
