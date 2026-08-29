"""
Visualize a BC dataset episode.  Two modes:

  static  (default) — plots point clouds and trajectories using matplotlib.
                       No simulator required.

  replay            — loads the simulator, resets to the correct scene, then
                       drives the robot through the recorded rollout while
                       overlaying the saved (EE-frame) point cloud as coloured
                       debug points in PyBullet. Two drivers (--replay-source):
                         states (default) — follows the stored robot_states, so it
                           faithfully reproduces the recorded rollout (the POLICY's
                           path for DAgger data, the expert's for the offline set),
                           and the point cloud lines up with the object/hand.
                         omg — re-plans the OMG expert and steps that instead; only
                           matches the OFFLINE expert dataset. For DAgger data this
                           shows the expert (not the policy) and the cloud will not
                           align — use 'states'.
                       With --show-expert-arrows (OFF by default) it also draws the
                       full EXPERT-action label (the OMG target) at each step:
                         - translation Δpos as a shaft from the current EE with a
                           3-D arrowhead at the TIP (where to go) — green=gripper
                           open, red=gripper close;
                         - rotation Δeuler as a small orientation triad at the tip
                           (X=yellow, Y=magenta, Z=cyan) = the commanded gripper
                           orientation.
                       Both Δpos and Δangle are exaggerated by --arrow-scale (the
                       per-step deltas are only ~cm / a few degrees). Arrows persist
                       through the rollout AND stay on screen after it ends (cleared
                       when you press R / N / P).

                       Keys in the PyBullet window: R = replay, N / P = next /
                       previous episode (wraps around, reloads that episode's scene
                       and grasp overlay), Q = quit.

Usage — static (random episode):
    python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train.h5

Usage — static (specific episode):
    python examples/visualize_bc_dataset.py --dataset output/bc_dataset/train.h5 --episode 0

Usage — simulator replay:
    python examples/visualize_bc_dataset.py \
        --dataset output/bc_dataset/train.h5 \
        --mode replay \
        --cfg-file examples/pretrain.yaml \
        --episode 0 \
        [--show-goal-grasp]   # overlay the grasp (green) + standoff (cyan) gripper
        [--show-grasp-set]    # overlay ALL candidate grasps OMG chose from (grey) +
                              #   highlight the one it used (green)

Also accepts an RL demo pool (examples/collect_rl_demos.py) — a streamed `.h5` or
a legacy `.npz`: episodes are split at terminal==1, the normalized pose is
denormalized, and the gripper bit is taken from the stored logit (OPEN iff >= 0).
Same static / replay modes:
    python examples/visualize_bc_dataset.py --dataset output/rl_demos/train.h5 \
        --mode replay --cfg-file examples/pretrain.yaml --episode 0
"""

import argparse
import h5py
import numpy as np
import os
import sys


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_episode_flat(full, sl, has, dataset_path, ep_idx=None):
    """Load one episode from a flat RL demo pool (examples/collect_rl_demos.py) —
    an npz or a streamed HDF5. Per-transition arrays are split into episodes at
    terminal==1; `full(key)`->whole array, `sl(key,a,b)`->row slice (lazy for h5),
    `has(key)`->bool abstract the backend. Denormalizes the stored pose back to
    real Δpos/Δeuler (via action_mean/std) and rebuilds a 7-D expert_actions with
    a gripper bit (OPEN iff the stored logit >= 0), so static + replay work."""
    term = np.asarray(full("terminal")).reshape(-1)
    ends = np.where(term >= 0.5)[0]                       # each episode ends at a terminal
    if len(ends) == 0:
        raise RuntimeError(f"No episodes (no terminal flags) in {dataset_path}")
    starts = np.concatenate([[0], ends[:-1] + 1])
    if ep_idx is None:
        ep_idx = int(np.random.randint(len(ends)))
    if ep_idx >= len(ends):
        raise KeyError(f"Episode {ep_idx} not found; pool has {len(ends)} episodes")
    a, b = int(starts[ep_idx]), int(ends[ep_idx]) + 1

    action = np.asarray(sl("action", a, b))              # [T,7] normalized pose + gripper logit
    mean = np.asarray(full("action_mean")) if has("action_mean") else np.zeros(6, np.float32)
    std  = np.asarray(full("action_std"))  if has("action_std")  else np.ones(6, np.float32)
    pose_real = action[:, :6] * std + mean               # -> real Δpos/Δeuler
    grip_bit  = (action[:, 6] >= 0.0).astype(np.float32)  # 1=open, 0=close
    expert_actions = np.concatenate([pose_real, grip_bit[:, None]], axis=1).astype(np.float32)

    scene_idx = int(np.asarray(sl("scene_idx", a, a + 1))[0]) if has("scene_idx") else 0
    meta = {"scene_idx": scene_idx, "num_steps": b - a}
    data = {"point_clouds": np.asarray(sl("pc", a, b)),
            "robot_states": np.asarray(sl("rs", a, b)),
            "expert_actions": expert_actions}
    reward_end = float(np.asarray(sl("reward", b - 1, b)).reshape(-1)[0])
    print(f"[rl-demo] episode {ep_idx}/{len(ends)}  scene_idx={scene_idx}  "
          f"steps={b - a}  reward@end={reward_end:.0f}")
    return meta, data, {"rl_demo": True}, ep_idx


def count_episodes(dataset_path):
    """Number of episodes in the dataset — `episode_*` groups for a BC HDF5, or
    the number of terminal flags for a flat RL demo pool (npz / streamed h5).
    Used by the replay viewer to wrap N / P episode navigation."""
    if str(dataset_path).endswith(".npz"):
        d = np.load(dataset_path)
        return int(np.sum(np.asarray(d["terminal"]).reshape(-1) >= 0.5))
    with h5py.File(dataset_path, "r") as f:
        if "terminal" in f:                                  # flat RL demo pool
            return int(np.sum(np.asarray(f["terminal"][:]).reshape(-1) >= 0.5))
        return len([k for k in f.keys() if k.startswith("episode_")])


def resolve_episode(dataset_path, scene=None, grasp_idx=None, bin_idx=None,
                    episode=None):
    """(scene, grasp_idx) -> the flat `--episode` index, or `episode` unchanged.

    `--episode` is a POSITION IN THE FILE and says nothing about what the episode
    is. What identifies a Regrasp demonstration is the `(scene_idx, grasp_idx)`
    pair it was collected for, and translating between the two by hand is a
    lookup nobody should have to do.

    WHY `grasp_idx` AND NOT A BIN. A bin does not name a demonstration. Under a
    `--per-bin 1` table (run 2) each bin holds one grasp and the two coincide;
    under `--per-bin 3` (run 3) a bin holds three grasps with three different
    poses and three different trajectories — on scene 32, `+x` is slots 0, 4 and
    8. So the slot is the definitive selector for a REPLAY. `bin_idx` is offered
    as a filter for the common "show me anything in this direction" case, and it
    reports every match rather than silently taking one.

    (For a ROLLOUT the opposite holds: `rollout_regrasp_policy.py --bin` is the
    right selector there, because a rollout issues a command built from the bin
    and the anchor alone and no grasp is being replayed.)

    Returns an episode index. Raises SystemExit listing what the file does hold
    when the request matches nothing — that message is the point, because the
    usual cause is a shard that simply does not contain the scene.
    """
    if scene is None and grasp_idx is None and bin_idx is None:
        return episode
    if str(dataset_path).endswith(".npz"):
        raise SystemExit("--scene / --grasp-idx / --bin need a per-episode HDF5; "
                         "a flat RL demo pool has no such attrs.")
    with h5py.File(dataset_path, "r") as f:
        if "terminal" in f:
            raise SystemExit("--scene / --grasp-idx / --bin need a per-episode "
                             "HDF5; this is a flat RL demo pool.")
        rows = []
        for k in sorted(x for x in f if x.startswith("episode_")):
            a = f[k].attrs
            rows.append((int(k.split("_")[1]), int(a.get("scene_idx", -1)),
                         int(a.get("grasp_idx", 0)),
                         int(a.get("bin_assigned", -1)),
                         int(a.get("num_steps", 0))))
    hit = [r for r in rows
           if (scene is None or r[1] == int(scene))
           and (grasp_idx is None or r[2] == int(grasp_idx))
           and (bin_idx is None or r[3] == int(bin_idx))]
    short = ("+x", "-x", "+y", "-y", "+z", "-z")
    if not hit:
        scenes = sorted({r[1] for r in rows})
        want = ", ".join(x for x in (
            f"scene {scene}" if scene is not None else "",
            f"grasp_idx {grasp_idx}" if grasp_idx is not None else "",
            f"bin {bin_idx}" if bin_idx is not None else "") if x)
        msg = [f"no episode matches {want} in {dataset_path}."]
        if scene is not None and int(scene) not in scenes:
            msg.append(f"  scene {scene} is not in this shard at all — it holds "
                       f"{len(scenes)} scenes: {scenes[:15]}"
                       + (" ..." if len(scenes) > 15 else ""))
            msg.append("  (a DAgger shard holds only the ~100 scenes its "
                       "iteration drew, not the whole split)")
        else:
            here = [r for r in rows if scene is None or r[1] == int(scene)]
            msg.append(f"  scene {scene} holds: " + ", ".join(
                f"grasp_idx {r[2]}"
                + (f" (bin {r[3]} {short[r[3]]})" if 0 <= r[3] < 6 else "")
                for r in here))
        raise SystemExit("\n".join(msg))
    if len(hit) > 1:
        print(f"[select] {len(hit)} episodes match; taking the first. All of them:")
        for r in hit:
            print(f"    --episode {r[0]:5d}  scene {r[1]:4d}  grasp_idx {r[2]:2d}"
                  + (f"  bin {r[3]} ({short[r[3]]})" if 0 <= r[3] < 6 else "")
                  + f"  steps {r[4]:3d}")
    r = hit[0]
    print(f"[select] scene {r[1]} grasp_idx {r[2]}"
          + (f" bin {r[3]} ({short[r[3]]})" if 0 <= r[3] < 6 else "")
          + f" -> --episode {r[0]}")
    return r[0]


def load_episode(dataset_path, ep_idx=None):
    """Return (metadata_dict, arrays_dict, file_meta, ep_idx) for one episode.
    Accepts a BC per-episode HDF5, or an RL demo pool (flat npz OR streamed HDF5)
    — auto-detected by extension and, for HDF5, by a top-level `terminal`
    dataset (flat RL pool) vs `episode_*` groups (BC dataset)."""
    if str(dataset_path).endswith(".npz"):
        d = np.load(dataset_path)
        return _load_episode_flat(lambda k: d[k], lambda k, a, b: d[k][a:b],
                                  lambda k: k in d.files, dataset_path, ep_idx)
    with h5py.File(dataset_path, "r") as f:
        if "terminal" in f:                              # flat RL demo pool (streamed)
            return _load_episode_flat(lambda k: f[k][:], lambda k, a, b: f[k][a:b],
                                      lambda k: k in f, dataset_path, ep_idx)
        keys = [k for k in f.keys() if k.startswith("episode_")]
        if len(keys) == 0:
            raise RuntimeError(f"No episodes found in {dataset_path}")

        if ep_idx is None:
            ep_idx = np.random.randint(len(keys))
        key = f"episode_{ep_idx:05d}"
        if key not in f:
            raise KeyError(f"Episode {key} not found; dataset has {len(keys)} episodes")

        grp  = f[key]
        meta = dict(grp.attrs)
        data = {
            "point_clouds":   grp["point_clouds"][:],   # [T, N, C]
            "robot_states":   grp["robot_states"][:],   # [T, 32]
            "expert_actions": grp["expert_actions"][:], # [T, 7]
        }
        file_meta = dict(f.attrs)
    return meta, data, file_meta, ep_idx


# ── static visualisation ──────────────────────────────────────────────────────

def visualize_static(dataset_path, ep_idx):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    meta, data, file_meta, ep_idx = load_episode(dataset_path, ep_idx)
    pc  = data["point_clouds"]    # [T, N, C]
    rs  = data["robot_states"]    # [T, 32]
    act = data["expert_actions"]  # [T, 7]
    T   = len(act)

    print(f"Episode {ep_idx}  scene_idx={meta['scene_idx']}  steps={T}")
    print(f"  point_cloud shape : {pc.shape}")
    print(f"  robot_state shape : {rs.shape}")
    print(f"  expert_action shape: {act.shape}")

    # ── EE trajectory from robot state (indices 18:21 = ee_xyz) ──────────────
    ee_xyz = rs[:, 18:21]  # joint_pos(9)+joint_vel(9)+ee_pos(3)

    # ── actions ──────────────────────────────────────────────────────────────
    delta_pos   = act[:, 0:3]
    delta_euler = act[:, 3:6]
    gripper_cmd = act[:, 6]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"BC Dataset — episode {ep_idx}  (scene {meta['scene_idx']}, {T} steps)",
                 fontsize=13)

    # 1. 3-D point cloud at first, middle, last step
    sample_steps = sorted(set([0, T // 2, T - 1]))
    for col, t in enumerate(sample_steps):
        ax = fig.add_subplot(2, 5, col + 1, projection="3d")
        pts = pc[t]                  # [N, C]
        xyz = pts[:, :3]
        # colour by semantic flag: YCB=orange, hand=blue, background=grey
        ycb_flag  = pts[:, 3] > 0.5
        hand_flag = pts[:, 4] > 0.5
        colours = np.full((len(pts), 3), 0.6)  # grey background
        colours[ycb_flag]  = [1.0, 0.5, 0.0]  # orange
        colours[hand_flag] = [0.2, 0.4, 1.0]  # blue
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=colours, s=1, alpha=0.6)
        ax.set_title(f"Point cloud t={t}", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.tick_params(labelsize=6)

    # 2. EE trajectory 3-D
    ax4 = fig.add_subplot(2, 5, 4, projection="3d")
    ax4.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], "o-", ms=3)
    ax4.scatter(*ee_xyz[0], color="green", s=50, label="start", zorder=5)
    ax4.scatter(*ee_xyz[-1], color="red",   s=50, label="end",   zorder=5)
    ax4.set_title("EE trajectory (world frame)", fontsize=9)
    ax4.set_xlabel("x"); ax4.set_ylabel("y"); ax4.set_zlabel("z")
    ax4.legend(fontsize=7)
    ax4.tick_params(labelsize=6)

    # 3. Joint positions over time
    ax5 = fig.add_subplot(2, 5, 5)
    for j in range(7):
        ax5.plot(rs[:, j], label=f"j{j}")
    ax5.set_title("Joint positions (arm)", fontsize=9)
    ax5.set_xlabel("step"); ax5.set_ylabel("rad")
    ax5.legend(fontsize=5, ncol=2)

    # 4. Delta position actions
    ax6 = fig.add_subplot(2, 5, 6)
    ax6.plot(delta_pos[:, 0], label="Δx")
    ax6.plot(delta_pos[:, 1], label="Δy")
    ax6.plot(delta_pos[:, 2], label="Δz")
    ax6.set_title("Expert Δposition", fontsize=9)
    ax6.set_xlabel("step"); ax6.legend(fontsize=7)

    # 5. Delta rotation actions
    ax7 = fig.add_subplot(2, 5, 7)
    ax7.plot(delta_euler[:, 0], label="Δroll")
    ax7.plot(delta_euler[:, 1], label="Δpitch")
    ax7.plot(delta_euler[:, 2], label="Δyaw")
    ax7.set_title("Expert Δrotation (euler)", fontsize=9)
    ax7.set_xlabel("step"); ax7.legend(fontsize=7)

    # 6. Gripper command
    ax8 = fig.add_subplot(2, 5, 8)
    ax8.step(range(T), gripper_cmd, where="post")
    ax8.set_ylim(-0.1, 1.1)
    ax8.set_yticks([0, 1]); ax8.set_yticklabels(["close", "open"])
    ax8.set_title("Gripper command", fontsize=9)
    ax8.set_xlabel("step")

    # 7. Gripper state (from robot state)
    # layout: joint_pos(9)+joint_vel(9)+ee_pos(3)+ee_orn(4)+gripper(1)+prev_act(6)
    #         indices 0-8    9-17        18-20     21-24     25         26-31
    ax9 = fig.add_subplot(2, 5, 9)
    ax9.plot(rs[:, 25], label="gripper norm")
    ax9.set_ylim(-0.05, 1.05)
    ax9.set_title("Gripper state (normalised)", fontsize=9)
    ax9.set_xlabel("step")

    # 8. Action magnitude
    ax10 = fig.add_subplot(2, 5, 10)
    mag = np.linalg.norm(act[:, :6], axis=1)
    ax10.plot(mag)
    ax10.set_title("Action magnitude ‖Δ‖", fontsize=9)
    ax10.set_xlabel("step")

    plt.tight_layout()
    plt.show()


# ── simulator replay ──────────────────────────────────────────────────────────

def visualize_replay(dataset_path, ep_idx, cfg_file, source="states",
                     show_expert=False, arrow_scale=3.0, show_goal_grasp=False,
                     show_grasp_set=False, max_grasp_set=40,
                     valid_grasp_dict="examples/valid_grasp_dict_005.pkl",
                     grasp_pin_table=None, show_anchor_frame=False,
                     show_bin_sphere=False, bin_sphere_radius=0.10,
                     bin_sphere_points=2400, show_d=False, d_rule=None,
                     d_point_depth=None, d_min_offset=None):
    import gym
    import pybullet
    import time

    import handover
    import handover_sim2real

    from handover.benchmark_wrapper import HandoverBenchmarkWrapper
    from handover_sim2real.config import get_cfg
    from handover_sim2real.utils import add_sys_path_from_env, resolve_valid_grasp_dict_path
    from handover_sim2real.regrasp import anchor as _rg_anchor
    from handover_sim2real.regrasp import channels as _rg_chan
    from handover_sim2real.regrasp import directions as _rg_dirs
    from handover_sim2real.regrasp import viz as _rg_viz

    add_sys_path_from_env("GADDPG_DIR")
    from experiments.config import cfg_from_file

    # Episodes are loaded lazily (below, in load_ep) so N / P can switch between
    # them without rebuilding the simulator; we only need the count up front to
    # wrap that navigation and to resolve a random --episode.
    n_episodes = count_episodes(dataset_path)
    if n_episodes == 0:
        raise RuntimeError(f"No episodes found in {dataset_path}")
    if ep_idx is None:
        ep_idx = int(np.random.randint(n_episodes))

    cfg = get_cfg()
    cfg_from_file(filename=cfg_file, dict=cfg, merge_to_cn_dict=True)
    cfg.SIM.RENDER = True  # open PyBullet GUI

    # Match the demo pool's grasp distribution: vgd pools (*_vgd.h5) were collected
    # with OMG loading the paper's per-scene hand-collision-filtered grasp dict
    # (collect_rl_demos.py). Without it, the --show-goal-grasp/--show-grasp-set
    # overlay re-plans OMG over the FULL ACRONYM set and can pick a DIFFERENT (often
    # hand-colliding) grasp than the episode actually aimed at. Set on cfg.omg_config
    # BEFORE the env (and its OMG planner) is built. Pass ''/'none' for a non-vgd pool.
    if valid_grasp_dict and str(valid_grasp_dict).lower() != "none":
        _vgd = resolve_valid_grasp_dict_path(
            {"valid_grasp_dict_path": valid_grasp_dict}, cfg.BENCHMARK.SETUP)
        if _vgd is not None:
            cfg.omg_config["valid_grasp_dict_path"] = _vgd
            print(f"[valid_grasp_dict] OMG grasp overlay uses the vgd subset: {_vgd}")

    pin_table = None
    if grasp_pin_table:
        # REGRASP's loader, not Phase 4's, and the difference matters here: a
        # Regrasp table carries SEVERAL grasps per scene and its `apply` takes a
        # `grasp_idx`, while Phase 4's has neither. A Phase-4 table still loads
        # through it — `_normalize_entry` wraps a bare entry as a one-element
        # list — so this is strictly wider, not a change of behaviour for the
        # datasets that were already working.
        from handover_sim2real.regrasp import load_grasp_pin_table
        pin_table = load_grasp_pin_table(grasp_pin_table)

    # WHICH `d` THIS SHARD WAS LABELLED UNDER. The rule is a property of the DATA,
    # not of this viewer, and it lives in the pin table's `_meta` (put there by
    # build_direction_table.py and carried forward by assign_direction_demos.py).
    # Defaulting to it means `--show-d` draws run 10's `grasp_offset` vector for a
    # run-10 table and runs 1-9's `approach_axis` for theirs, with no flag — and
    # drawing the wrong rule is not a cosmetic error, the two differ by a median
    # 14.5 deg and disagree about which bin a grasp is in.
    _rule_meta = dict(getattr(pin_table, "meta", None) or {}) if pin_table else {}
    _rule_block = {
        "d_rule": d_rule or _rule_meta.get("d_rule", "approach_axis"),
        "d_point_depth": (d_point_depth if d_point_depth is not None
                          else _rule_meta.get("d_point_depth",
                                              _rg_dirs.FINGERTIP_DEPTH)),
        "d_min_offset": (d_min_offset if d_min_offset is not None
                         else _rule_meta.get("d_min_offset", 0.0)),
    }
    d_rule_obj = _rg_dirs.DirectionRule.from_cfg(_rule_block)
    # The rule can be named in two independent places — the run's SIM block and
    # the pin table's `_meta` — and they are built by different scripts. If they
    # ever disagree, one of them is describing data that was labelled the other
    # way, and every `d` drawn from here on is wrong by a median 14.5 deg. Say so
    # rather than picking a winner silently.
    _table_rule = _rule_meta.get("d_rule")
    if _table_rule and _table_rule != _rule_block["d_rule"]:
        print(f"[d_rule] WARNING: rule mismatch — using {_rule_block['d_rule']!r} "
              f"but {grasp_pin_table} says {_table_rule!r}. One of them does not "
              f"describe this shard; check the run's SIM.d_rule.")
    if show_d or show_anchor_frame or show_bin_sphere:
        print(f"[d_rule] {d_rule_obj.describe()}"
              + ("" if d_rule else f"   (from {grasp_pin_table or 'the default'})"))

    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))

    steps_action_repeat = int(cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION
    )

    from handover_sim2real.utils import add_sys_path_from_env
    add_sys_path_from_env("GADDPG_DIR")
    from core.utils import tf_quat, unpack_pose, se3_transform_pc, euler2mat
    from scipy.spatial.transform import Rotation as Rot

    panda_base_pos = np.array(cfg.ENV.PANDA_BASE_POSITION)
    panda_base_orn = np.array(cfg.ENV.PANDA_BASE_ORIENTATION)  # xyzw
    R_base = Rot.from_quat(panda_base_orn).as_matrix()

    def draw_gripper(pose_mat, colour, line_ids, line_width=2.0):
        """Panda parallel-jaw wireframe at 4x4 world pose (same convention as
        rollout_bc_policy.draw_gripper / visualize_grasps.gripper_segments)."""
        from visualize_grasps import gripper_segments
        for p, q in gripper_segments(pose_mat):
            line_ids.append(pybullet.addUserDebugLine(
                p.tolist(), q.tolist(), lineColorRGB=colour, lineWidth=line_width))

    def clear_ids(ids):
        for item_id in ids:
            pybullet.removeUserDebugItem(item_id)
        ids.clear()

    # ── per-episode state ─────────────────────────────────────────────────────
    # `ep` holds everything about the episode currently loaded; load_ep() swaps it
    # so N / P can walk the dataset without rebuilding the simulator. Debug items
    # are split by lifetime: goal_ids are per-SCENE (drawn at load, kept up until
    # the episode changes), arrow_ids are per-ROLLOUT (kept on screen after
    # playback ends so the finished correction field stays inspectable, wiped when
    # the next play starts).
    ep        = {}
    goal_ids  = []
    arrow_ids = []
    warned_omg_dagger = [False]

    def load_ep(idx):
        """Load episode `idx`, reset the sim to its scene, redraw the per-scene
        grasp overlay and fill `ep`. Returns the fresh obs."""
        meta, data, file_meta, idx = load_episode(dataset_path, idx)
        scene_idx    = int(meta["scene_idx"])
        # WHICH GRASP this episode flew, not which bin. A Regrasp bin holds one
        # grasp under `--per-bin 1` and three under `--per-bin 3`, so the bin
        # does not identify an episode and the slot does. Absent on Phase-4 and
        # RL-demo files, where a scene has exactly one target and 0 is right.
        grasp_idx    = int(meta.get("grasp_idx", 0))
        saved_pc     = data["point_clouds"]      # [T, N, C]  EE-frame cloud per step
        robot_states = data["robot_states"]      # [T, 32]  joint_pos(9)+... per step
        expert_act   = data["expert_actions"]    # [T, 7]  OMG label at each visited state
        T = len(saved_pc)

        is_dagger = bool(file_meta.get("dagger", False))
        # `grasp_idx` and the bin are both printed because they answer different
        # questions and are easy to confuse: the slot says WHICH GRASP this
        # episode is, the bin says which DIRECTION it was captioned under, and
        # under a 3-per-bin table three slots share one bin.
        _b = int(meta.get("bin_assigned", -1))
        _bn = ("" if _b < 0 else
               f"  bin={_b}({('+x','-x','+y','-y','+z','-z')[_b]})")
        print(f"Replaying episode {idx}/{n_episodes - 1}  scene_idx={scene_idx}  "
              f"grasp_idx={grasp_idx}{_bn}  steps={T}  source={source}  "
              f"(dagger={is_dagger})")
        if source == "omg" and is_dagger and not warned_omg_dagger[0]:
            print("  NOTE: --replay-source omg re-plans the OMG expert and steps THAT, "
                  "not the\n        policy's recorded states — for DAgger data the robot "
                  "and the point\n        cloud will NOT match the recorded rollout. Use "
                  "--replay-source states.")
            warned_omg_dagger[0] = True

        clear_ids(goal_ids)
        clear_ids(arrow_ids)
        obs = env.reset(idx=scene_idx)

        # 'omg' mode re-plans the expert and drives the sim along it (only faithful
        # for the OFFLINE expert dataset). 'states' mode drives the sim through the
        # stored robot_states, faithfully reproducing whatever rollout was recorded.
        # We also run OMG (once per episode, from the reset config = same grasp the
        # data aimed at) when --show-goal-grasp / --show-grasp-set is set, just to
        # read back the goal grasp / standoff poses and the full candidate set.
        expert_plan = None
        if source == "omg" or show_goal_grasp or show_grasp_set:
            expert_plan, _ = env.run_omg_planner(cfg.RL_MAX_STEP, scene_idx)
            # Apply the pin table if the data was collected with one. Without this,
            # OMG re-selects its goal here (argmin over the goal set, planner.py
            # under ol_alg='Proj') and the overlay can show a DIFFERENT grasp than
            # the episode actually aimed at — measured at up to 20.5 cm apart.
            # Replanning after the pin keeps `expert_plan` consistent too.
            #
            # THE EPISODE'S OWN SLOT, not slot 0. This used to call
            # `apply(env, scene_idx)`, which defaults to slot 0 — correct under
            # Phase 4, where a scene had ONE pinned grasp, and wrong for every
            # Regrasp episode with `grasp_idx != 0`. It drew a confident green
            # gripper at a grasp the episode never aimed at, which is the exact
            # failure the comment above says this block exists to prevent. Under
            # a `--per-bin 3` table three quarters of the episodes were affected.
            if expert_plan is not None and pin_table is not None:
                if pin_table.apply(env, scene_idx, grasp_idx):
                    expert_plan, _ = env.run_omg_planner(cfg.RL_MAX_STEP, scene_idx,
                                                         reset_scene=False)
            if expert_plan is None:
                # Don't bail out of the whole session — the user can still press
                # N / P to walk to an episode OMG can handle.
                print("OMG planner failed — cannot replay this episode "
                      "(--replay-source omg); press N / P for another."
                      if source == "omg" else
                      "OMG planner failed — cannot draw --show-goal-grasp.")

        ep.update(idx=idx, scene_idx=scene_idx, pc=saved_pc, rs=robot_states,
                  act=expert_act, T=T, expert_plan=expert_plan,
                  stop_step=(len(expert_plan) if expert_plan is not None else 0)
                            if source == "omg" else T)
        _draw_goal_overlay(meta, robot_states)
        _draw_conditioning_overlay(meta, saved_pc, obs)
        return obs

    def _draw_conditioning_overlay(meta, saved_pc, obs):
        """Anchor frame, bin sphere and `d`, at the anchor's own origin.

        THE ORIGIN IS RECOMPUTED, NOT STORED. Episodes carry `anchor_R` but not
        the centroid it is pinned at, so it is rebuilt here exactly as the
        collector built it: the observed-cloud centroid of the STEP-0 frame, in
        the EE frame, mapped to world. Using the object's pose instead would
        offset the drawing by a few cm from the frame the labels were actually
        computed in — the conditioning channels use the visible-surface centroid,
        so that is what has to be drawn.
        """
        if not (show_anchor_frame or show_bin_sphere or show_d):
            return
        aR = meta.get("anchor_R")
        if aR is None:
            print("  [overlay] this episode has no `anchor_R` attr (pre-Regrasp "
                  "shard) — nothing to draw.")
            return
        aR = np.asarray(aR, dtype=np.float64)

        # THE STORED ORIGIN WINS WHERE IT EXISTS. DAgger shards carry
        # `centroid_world` — the exact value the labels were computed against —
        # so recomputing it would only introduce a difference. Base shards
        # predate the attr, and there the step-0 cloud is rebuilt the way the
        # collector built it.
        c_world = meta.get("centroid_world")
        if c_world is not None:
            c_world = np.asarray(c_world, dtype=np.float64)
        else:
            c_ee = _rg_chan.object_centroid(np.asarray(saved_pc[0]),
                                            fallback_to_all=False)
            if c_ee is None:
                print("  [overlay] step 0 has no object points — cannot place "
                      "the anchor origin.")
                return
            c_world = _rg_anchor.centroid_to_world(
                c_ee, obs, panda_base_inv_tf,
                cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)

        if show_bin_sphere:
            _rg_viz.draw_bin_sphere(aR, c_world, goal_ids,
                                    radius=bin_sphere_radius,
                                    n_points=bin_sphere_points)
        if show_anchor_frame:
            _rg_viz.draw_anchor_frame(aR, c_world, goal_ids)

        if not show_d:
            return
        # THREE VECTORS, AND THEY ANSWER THREE DIFFERENT QUESTIONS. Drawing only
        # one is how you convince yourself the conditioning is fine when it is
        # not; mislabelling them is how you convince yourself it is broken when
        # it is not.
        #
        #   white   what this episode was COMMANDED (`d_world`) — the vector the
        #           network actually read
        #   yellow  THIS SHARD'S RULE applied to the grasp the episode flew,
        #           read from its own `d_grasp_world` attr
        #   grey    THE OTHER RULE, drawn for contrast
        #
        # The grey one earns its place. `approach_axis` is a property of the
        # gripper's ORIENTATION and `grasp_offset` of its POSITION, and on a
        # grasp that reaches down to close on an object's underside they are
        # nearly perpendicular — measured 88.6 deg on run 9's scene 52 slot 0,
        # where the fingertips sit 12.0 cm below the centroid while the approach
        # axis is horizontal. Seeing only one of them, the other looks like a
        # bug. Seeing both is the whole argument for run 10.
        gp = meta.get("grasp_pose_world")
        gp = None if gp is None else np.asarray(gp, dtype=np.float64)

        dw = meta.get("d_world")
        if dw is not None:
            _rg_viz.draw_direction(dw, c_world, goal_ids, colour=(1.0, 1.0, 1.0),
                                   label="d commanded")

        # The shard's own label, not a recomputation — that is what training reads.
        dg = meta.get("d_grasp_world")
        if dg is not None and np.linalg.norm(np.asarray(dg, float)) > 0.5:
            _rg_viz.draw_direction(dg, c_world, goal_ids, colour=(1.0, 0.85, 0.1),
                                   label=f"d flown ({d_rule_obj.rule})",
                                   length=0.26)
        if d_rule_obj.needs_centroid() and gp is not None:
            off = _rg_viz.draw_grasp_point(gp, c_world, goal_ids,
                                           depth=d_rule_obj.depth)
            print(f"  [d] offset centroid -> fingertips: {off * 100:.1f} cm"
                  + ("   ** below d_min_offset "
                     f"{d_rule_obj.min_offset * 100:.0f} cm — `d` here is "
                     f"centroid noise **" if off < d_rule_obj.min_offset else ""))

        other = ("grasp_offset" if d_rule_obj.rule == "approach_axis"
                 else "approach_axis")
        d_other = (_rg_dirs.DirectionRule(rule=other, depth=d_rule_obj.depth)
                   .of(gp, c_world) if gp is not None else None)
        if d_other is not None:
            _rg_viz.draw_direction(d_other, c_world, goal_ids,
                                   colour=(0.6, 0.6, 0.6),
                                   label=f"d {other} (not used)",
                                   length=0.18, width=3.0)

        if dw is not None and dg is not None:
            print(f"  [d] commanded vs flown ({d_rule_obj.rule}): "
                  f"{float(_rg_dirs.angle_between(dw, dg)):.1f} deg")
        if dg is not None and d_other is not None:
            print(f"  [d] {d_rule_obj.rule} vs {other}: "
                  f"{float(_rg_dirs.angle_between(dg, d_other)):.1f} deg"
                  f"   (orientation vs position — they are different questions)")

    def _draw_goal_overlay(meta=None, robot_states=None):
        if not (show_goal_grasp or show_grasp_set):
            return
        # THE EPISODE'S OWN GRASP, NOT A RE-PLANNED ONE. `get_omg_goal_grasp_pose`
        # is what OMG picked when THIS VIEWER replanned, which is not necessarily
        # what the episode flew, and the gap is visible:
        #
        #   * the pin matches by position within `match_tol` (2 cm) and only then
        #     disambiguates by rotation, so even a SUCCESSFUL re-pin can land on a
        #     candidate up to 2 cm from the recorded one;
        #   * where the episode's `pin_ok` is 0 the collector stored the grasp OMG
        #     actually flew to, while the viewer's re-pin may now succeed — so the
        #     overlay drew a completely different pose (11 of 1418 episodes on
        #     run 10's shard, and the symptom is "the gripper lands at a totally
        #     different orientation than the green gripper").
        #
        # `grasp_pose_world` is read back from the planner AFTER pinning at
        # collection time, so it is true by construction for this episode — and it
        # is the pose `--show-d` builds `d` from, so drawing anything else here
        # makes the two overlays disagree with each other.
        goal_mat = standoff_mat = None
        if meta is not None and meta.get("grasp_pose_world") is not None:
            goal_mat = np.asarray(meta["grasp_pose_world"], dtype=np.float64)
            # Analytic, from the grasp: OMG appends the standoff ramp in Cartesian
            # space without re-optimising it, so traj[-reach_tail] IS the grasp
            # backed off along its own -z (matched to 5.6e-7 m — see
            # regrasp/collector.py:derived_standoff_pose).
            standoff_mat = goal_mat @ np.array(
                [[1., 0., 0., 0.], [0., 1., 0., 0.],
                 [0., 0., 1., -0.064], [0., 0., 0., 1.]], dtype=np.float64)
            if robot_states is not None:
                # How far the demonstration ACTUALLY got, printed next to the
                # overlay so a gap between the green gripper and the robot is
                # named rather than left to be squinted at.
                from handover_sim2real.regrasp import reach as _rg_reach
                p_err, r_err = _rg_reach.terminal_pose_error(
                    robot_states[-1], goal_mat)
                depth = float((np.asarray(robot_states[-1][18:21])
                               - goal_mat[:3, 3]) @ goal_mat[:3, 2])
                print(f"  terminal EE vs this episode's grasp: {p_err*100:.2f} cm"
                      f" / {np.degrees(r_err):.1f} deg"
                      f"   ({depth*100:+.2f} cm along the approach axis)"
                      + ("   ** the demo did NOT reach it — truncated **"
                         if not _rg_reach.reached(robot_states[-1], goal_mat)
                         else ""))
                if not int(meta.get("pin_ok", 1)):
                    print("  NOTE pin_ok=0: the pin failed at COLLECTION, so this "
                          "episode flew to OMG's own choice, not the table's. The "
                          "green gripper is the flown grasp; the pin table names a "
                          "different one.")
        else:
            goal_mat     = env.get_omg_goal_grasp_pose()   # traj[-1]
            standoff_mat = env.get_omg_standoff_pose()     # traj[-5]

        # Full candidate set first (thin grey) so the highlighted goal draws on top:
        # every grasp OMG could pick from for this scene — the valid_grasp_dict-
        # filtered object grasps placed at the live target pose (get_grasp_poses_world).
        if show_grasp_set:
            grasps_world = env.get_grasp_poses_world()   # (N, 4, 4) world frame
            n = len(grasps_world)
            if n == 0:
                print("  --show-grasp-set: OMG loaded no candidate grasps — nothing "
                      "to draw.")
            else:
                sel = np.arange(n)
                if n > max_grasp_set:
                    sel = np.random.choice(n, size=max_grasp_set, replace=False)
                for gmat in grasps_world[sel]:
                    draw_gripper(gmat, [0.55, 0.55, 0.55], goal_ids, 1.0)  # thin grey
                print(f"  candidate grasp set (grey): drew {len(sel)}/{n} — the poses "
                      f"OMG chose its goal from")

        if goal_mat is not None:
            draw_gripper(goal_mat, [0.0, 1.0, 0.0], goal_ids, 3.0)        # green
            print(f"  goal grasp (green, the one OMG used) pos={goal_mat[:3, 3].round(3)}")
        if show_goal_grasp and standoff_mat is not None:
            draw_gripper(standoff_mat, [0.0, 1.0, 1.0], goal_ids, 2.0)    # cyan
            print(f"  pre-grasp standoff (cyan) pos={standoff_mat[:3, 3].round(3)}"
                  f"  — where approach-only DAgger labels stop")
        if goal_mat is None and standoff_mat is None:
            print("  OMG found no goal grasp — nothing to draw.")

    def play_once(obs):
        saved_pc, robot_states = ep["pc"], ep["rs"]
        expert_act, T          = ep["act"], ep["T"]
        debug_ids = []
        clear_ids(arrow_ids)
        for step in range(min(ep["stop_step"], T)):
            link_ind = obs["panda_link_ind_hand"]
            pos_world = obs["panda_body"].link_state[0, link_ind, 0:3]
            orn_world = obs["panda_body"].link_state[0, link_ind, 3:7]
            pos_base, orn_base = pybullet.multiplyTransforms(
                *panda_base_inv_tf, pos_world, orn_world
            )
            ee_mat = unpack_pose(np.hstack([pos_base, tf_quat(orn_base)]))

            # Point cloud is stored in EE frame → transform to world for display.
            pts_ee   = saved_pc[step, :, :3]
            pts_base = se3_transform_pc(ee_mat, pts_ee.T).T
            pts_world = (R_base @ pts_base.T).T + panda_base_pos

            ycb_flag  = saved_pc[step, :, 3] > 0.5
            hand_flag = saved_pc[step, :, 4] > 0.5
            colours = np.full((len(pts_world), 3), 0.6)
            colours[ycb_flag]  = [1.0, 0.5, 0.0]
            colours[hand_flag] = [0.3, 0.5, 1.0]

            for dbg_id in debug_ids:
                pybullet.removeUserDebugItem(dbg_id)
            debug_ids.clear()

            idx_show = np.random.choice(
                len(pts_world), size=min(200, len(pts_world)), replace=False
            )
            dbg_id = pybullet.addUserDebugPoints(
                pts_world[idx_show].tolist(),
                colours[idx_show].tolist(),
                pointSize=4,
            )
            debug_ids.append(dbg_id)

            # Expert-action arrow: where OMG (the label) says to move the EE from
            # this policy-visited state. act[:3] is the Δpos in the EE frame; map it
            # to world via the current EE pose and draw a (scaled) line. Green =
            # gripper-open label, red = gripper-close label. Arrows persist so the
            # whole correction "field" along the rollout stays visible.
            if show_expert:
                act = expert_act[step]
                cur_base = ee_mat[:3, 3]
                tgt_base = ee_mat[:3, :3] @ act[:3] + cur_base
                cur_w = R_base @ cur_base + panda_base_pos
                tgt_w = R_base @ tgt_base + panda_base_pos
                tgt_w = cur_w + arrow_scale * (tgt_w - cur_w)   # exaggerate (Δ≈3-4cm)
                colour = [0.0, 0.9, 0.0] if act[6] >= 0.5 else [0.9, 0.0, 0.0]
                # Shaft (current EE → expert target) + a 3-D arrowhead at the TIP
                # so the start (gripper) vs end (where to go) is unambiguous.
                arrow_ids.append(pybullet.addUserDebugLine(
                    cur_w.tolist(), tgt_w.tolist(), colour, lineWidth=3))
                d = tgt_w - cur_w
                L = float(np.linalg.norm(d))
                if L > 1e-6:
                    dh = d / L
                    ref = np.array([0.0, 0.0, 1.0]) if abs(dh[2]) < 0.9 \
                        else np.array([1.0, 0.0, 0.0])
                    p1 = np.cross(dh, ref); p1 /= np.linalg.norm(p1)
                    p2 = np.cross(dh, p1)
                    base_h = tgt_w - 0.25 * L * dh          # arrowhead length 25%
                    for pv in (p1, p2):
                        for s in (1.0, -1.0):
                            head = base_h + s * 0.12 * L * pv
                            arrow_ids.append(pybullet.addUserDebugLine(
                                tgt_w.tolist(), head.tolist(), colour, lineWidth=3))

                # Rotation part of the label (act[3:6], Δeuler in the EE frame):
                # draw the commanded gripper orientation as a triad at the tip.
                # X=yellow, Y=magenta, Z=cyan (kept distinct from the green/red
                # translation shaft). The Δangle is exaggerated by arrow_scale too,
                # so the (few-degree) per-step rotation is actually visible.
                R_delta = euler2mat(act[3], act[4], act[5])          # EE-frame, sxyz
                rotvec = Rot.from_matrix(R_delta).as_rotvec() * arrow_scale
                R_delta_ex = Rot.from_rotvec(rotvec).as_matrix()
                tgt_rot_world = R_base @ (ee_mat[:3, :3] @ R_delta_ex)
                axis_cols = ([1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0])
                for a in range(3):
                    ax_end = tgt_w + 0.04 * tgt_rot_world[:, a]
                    arrow_ids.append(pybullet.addUserDebugLine(
                        tgt_w.tolist(), ax_end.tolist(), axis_cols[a], lineWidth=2))

            # Advance the sim to the next recorded state. 'states': drive the
            # panda toward the next stored joint config (faithfully reproduces the
            # recorded rollout — policy states for DAgger, expert for offline).
            # 'omg': step the freshly re-planned expert trajectory (old behavior).
            done = False
            if source == "omg":
                target = ep["expert_plan"][step]
            else:
                target = robot_states[min(step + 1, T - 1), :9]
            for _ in range(steps_action_repeat):
                obs, _, done, _ = env.step(target)
                if done:
                    break

            time.sleep(0.05)
            if done:
                break

        # Clear only the (per-step) point cloud when playback ends; the expert
        # arrows stay on screen so the finished correction field is inspectable.
        for dbg_id in debug_ids:
            pybullet.removeUserDebugItem(dbg_id)
        return obs

    obs = load_ep(ep_idx)
    obs = play_once(obs)

    HELP = ("In the PyBullet window:  R = replay,  N / P = next / previous episode,"
            "  Q = quit.")
    print("Replay finished.")
    print(HELP)

    R_KEY, Q_KEY, N_KEY, P_KEY = ord('r'), ord('q'), ord('n'), ord('p')

    def pressed(keys, key):
        return key in keys and keys[key] & pybullet.KEY_WAS_TRIGGERED

    try:
        while True:
            keys = pybullet.getKeyboardEvents()
            if pressed(keys, R_KEY):
                print("Replaying...")
                obs = env.reset(idx=ep["scene_idx"])
                obs = play_once(obs)
                print("Replay finished.  " + HELP)
            elif pressed(keys, N_KEY) or pressed(keys, P_KEY):
                # Wrap around the dataset. load_ep re-resets the sim to the new
                # episode's scene and redraws the grasp overlay for it.
                step = 1 if pressed(keys, N_KEY) else -1
                obs = load_ep((ep["idx"] + step) % n_episodes)
                obs = play_once(obs)
                print("Replay finished.  " + HELP)
            if pressed(keys, Q_KEY):
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualise a BC dataset episode.")
    # ---- name a RUN and let everything else follow --------------------------
    # A shard does not carry the pin table its `grasp_idx` indexes or the
    # d_rule its `d_world` was labelled under, and using the wrong one is
    # SILENT: `approach_axis` on a `grasp_offset` shard draws a vector a median
    # 14.5 deg off, in a different bin. `<run_dir>/config.yaml` records both, so
    # `--run` + `--iter` derives --dataset, --cfg-file, --grasp-pin-table,
    # --d-rule, --d-point-depth and --d-min-offset. Any of those passed
    # explicitly still wins.
    p.add_argument("--run", default=None,
                   help="Regrasp run NAME (regrasp_run12) or run directory. "
                        "Derives the dataset, benchmark config, pin table and "
                        "direction rule from the run's own config.yaml — use "
                        "with --iter. Replaces --dataset.")
    p.add_argument("--iter", dest="iteration", default=None,
                   help="which DAgger iteration of --run to open: an integer, "
                        "0 or 'base' for the base demonstration set, or 'last'.")
    p.add_argument("--run-root", default=None,
                   help="where <run>/ lives, if not $REGRASP_DATA/dagger_runs, "
                        "$RUNS/output/dagger_runs or ./output/dagger_runs.")
    p.add_argument("--dataset",  default=None,
                   help="BC HDF5 dataset, or an RL demo pool (.h5/.npz) from "
                        "collect_rl_demos.py. Alternative to --run/--iter.")
    p.add_argument("--episode",  type=int, default=None,
                   help="episode index (default: random)")
    p.add_argument("--mode",     default=None, choices=["static", "replay"],
                   help="static=matplotlib plots, replay=PyBullet simulator. "
                        "Defaults to static, or to replay when --run is given "
                        "(the derived config and pin table are only used there).")
    p.add_argument("--cfg-file", default=None,
                   help="config yaml (required for --mode replay)")
    p.add_argument("--replay-source", default="states", choices=["states", "omg"],
                   help="replay mode driver: 'states' drives the sim through the "
                        "recorded robot_states (faithful — use this for DAgger data); "
                        "'omg' re-plans the expert and steps that (old behavior, only "
                        "matches the offline expert dataset).")
    p.add_argument("--show-expert-arrows", dest="show_expert", action="store_true",
                   help="draw the per-step expert-action (OMG label) arrows in replay: "
                        "the translation Δpos shaft (green=gripper open, red=close) and "
                        "the Δeuler orientation triad at its tip. OFF by default — they "
                        "clutter the view when you just want to watch the rollout.")
    p.add_argument("--no-expert-arrows", dest="show_expert", action="store_false",
                   help=argparse.SUPPRESS)  # back-compat no-op (arrows are off by default)
    p.set_defaults(show_expert=False)
    p.add_argument("--arrow-scale", type=float, default=3.0,
                   help="exaggeration factor for the expert-action arrows "
                        "(the per-step Δpos is only ~3-4 cm; default 3×).")
    p.add_argument("--show-goal-grasp", action="store_true",
                   help="overlay the gripper pose OMG planned to reach — green = "
                        "goal grasp (traj[-1]), cyan = pre-grasp standoff (traj[-5], "
                        "where approach-only DAgger labels stop). Runs OMG once.")
    p.add_argument("--show-grasp-set", action="store_true",
                   help="overlay the whole set of candidate grasp poses OMG chose "
                        "from for this scene (thin grey grippers, the valid_grasp_dict-"
                        "filtered object grasps at the live target pose) and highlight "
                        "the one it actually used (green). Runs OMG once.")
    p.add_argument("--max-grasp-set", type=int, default=40,
                   help="max candidate grasps to draw with --show-grasp-set (random "
                        "subsample to keep the GUI light; default 40).")
    p.add_argument("--valid-grasp-dict", default="examples/valid_grasp_dict_005.pkl",
                   help="per-scene hand-collision-filtered grasp dict OMG loads so the "
                        "--show-goal-grasp/--show-grasp-set overlay matches how vgd demo "
                        "pools (*_vgd.h5) were collected (default). Pass '' or 'none' to "
                        "use the FULL ACRONYM grasp set instead (non-vgd pools).")
    p.add_argument("--grasp-pin-table", default=None,
                   help="per-scene committed grasp (examples/build_grasp_pin_table.py). "
                        "Pass the SAME table the dataset was collected with, or the "
                        "--show-goal-grasp overlay may draw a different grasp than the "
                        "episode aimed at: OMG re-selects its goal on every plan "
                        "(argmin over the goal set), and the demos-vs-replan target was "
                        "measured up to 20.5 cm apart on unpinned data.")
    # ---- select an episode by WHAT IT IS, not by where it sits in the file ---
    p.add_argument("--scene", type=int, default=None,
                   help="pick the episode by scene index instead of --episode. "
                        "Combine with --grasp-idx to name one exactly.")
    p.add_argument("--grasp-idx", type=int, default=None,
                   help="which GRASP of that scene — the episode's own "
                        "`grasp_idx` attr, i.e. its slot in the pin table. THIS "
                        "IS THE DEFINITIVE SELECTOR FOR A REPLAY: a bin holds "
                        "one grasp under a --per-bin 1 table but three under "
                        "--per-bin 3 (scene 32's `+x` is slots 0, 4 and 8), so "
                        "the bin alone does not identify a demonstration.")
    p.add_argument("--bin", type=int, default=None,
                   help="filter by commanded bin (0=+x 1=-x 2=+y 3=-y 4=+z "
                        "5=-z). A convenience for 'anything in this direction' "
                        "— where several grasps share the bin it lists them all "
                        "and takes the first. Use --grasp-idx to be exact.")
    # ---- Regrasp conditioning overlays (replay mode) ------------------------
    p.add_argument("--show-anchor-frame", action="store_true",
                   help="draw the anchor frame (x away from the hand, z world "
                        "up) at the observed object centroid — the frame every "
                        "`d` is expressed in")
    p.add_argument("--show-bin-sphere", action="store_true",
                   help="draw a see-through sphere around the object painted by "
                        "BIN, with a labelled ray down each bin axis")
    p.add_argument("--bin-sphere-radius", type=float, default=0.10)
    p.add_argument("--bin-sphere-points", type=int, default=2400)
    p.add_argument("--show-d", action="store_true",
                   help="draw the conditioning vector: white = what the episode "
                        "was COMMANDED (`d_world`), yellow = this shard's d_rule "
                        "applied to the grasp it flew (run 10: grasp_offset, "
                        "centroid -> fingertip midpoint), grey = `-R_grasp[:,2]`")
    p.add_argument("--d-rule", default=None, choices=["approach_axis", "grasp_offset"],
                   help="override the rule for --show-d. Default reads "
                        "`_meta.d_rule` from --grasp-pin-table, which is what the "
                        "shard was actually labelled under")
    p.add_argument("--d-point-depth", type=float, default=None,
                   help="metres along the gripper's +z for grasp_offset "
                        "(default 0.1122, the fingertip end of the Panda pads)")
    p.add_argument("--d-min-offset", type=float, default=None,
                   help="reject grasp_offset shorter than this (run 10: 0.02)")
    p.add_argument("--seed",     type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    # ---- --run: derive the four things a shard cannot tell you itself -------
    # Explicit flags always win, so `--run regrasp_run12 --d-rule approach_axis`
    # is still how you look at the same episode under the other rule.
    if args.run:
        if args.dataset:
            raise SystemExit("pass either --run/--iter or --dataset, not both.")
        if args.iteration is None:
            raise SystemExit(
                "--run needs --iter: an integer, 0/'base' for the base "
                "demonstration set, or 'last'.")
        from handover_sim2real.regrasp.runspec import resolve_run
        spec = resolve_run(args.run, run_root=args.run_root)
        args.dataset = spec.dataset_for(args.iteration)
        if args.cfg_file is None:
            args.cfg_file = spec.cfg_file
        if args.grasp_pin_table is None:
            args.grasp_pin_table = spec.pin_table
        if args.d_rule is None:
            args.d_rule = spec.d_rule
        if args.d_point_depth is None:
            args.d_point_depth = spec.d_point_depth
        if args.d_min_offset is None:
            args.d_min_offset = spec.d_min_offset
        if args.mode is None:
            args.mode = "replay"
        print(spec.describe(args.iteration))
    elif not args.dataset:
        raise SystemExit("one of --run (with --iter) or --dataset is required.")
    if args.mode is None:
        args.mode = "static"

    # An UNSET ${RUNS} (or ${REGRASP_DATA}) expands to nothing in the shell, so a
    # copy-pasted cluster command turns into an absolute `/output/...` path and
    # h5py reports it as a bare FileNotFoundError several frames deep. Say which
    # of the two it is, here, before the simulator is built.
    if not os.path.exists(args.dataset):
        hint = ("\n  The path starts at `/`, which is what an UNSET ${RUNS} or "
                "${REGRASP_DATA} looks like\n  after the shell expands it. Either "
                "export it, or use a repo-relative path."
                if args.dataset.startswith("/output") else "")
        raise SystemExit(f"--dataset not found: {args.dataset}{hint}")
    if args.grasp_pin_table and not os.path.exists(args.grasp_pin_table):
        raise SystemExit(f"--grasp-pin-table not found: {args.grasp_pin_table}")

    # Resolved once, before either mode: both take a flat episode index, and
    # this is the only place that knows how to get one from a (scene, grasp).
    episode = resolve_episode(args.dataset, scene=args.scene,
                              grasp_idx=args.grasp_idx, bin_idx=args.bin,
                              episode=args.episode)

    if args.mode == "static":
        visualize_static(args.dataset, episode)
    else:
        if args.cfg_file is None:
            print("Error: --cfg-file is required for --mode replay")
            sys.exit(1)
        visualize_replay(args.dataset, episode, args.cfg_file, args.replay_source,
                         args.show_expert, args.arrow_scale, args.show_goal_grasp,
                         args.show_grasp_set, args.max_grasp_set, args.valid_grasp_dict,
                         args.grasp_pin_table,
                         show_anchor_frame=args.show_anchor_frame,
                         show_bin_sphere=args.show_bin_sphere,
                         bin_sphere_radius=args.bin_sphere_radius,
                         bin_sphere_points=args.bin_sphere_points,
                         show_d=args.show_d, d_rule=args.d_rule,
                         d_point_depth=args.d_point_depth,
                         d_min_offset=args.d_min_offset)


if __name__ == "__main__":
    main()
