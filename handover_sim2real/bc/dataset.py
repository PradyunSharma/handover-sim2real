"""
Dataset + normalization for offline BC training.

The HDF5 files are produced by examples/collect_bc_dataset.py with one group
per episode:

    episode_NNNNN/
        ├── point_clouds   float32 [T, 1024, 5]   xyz + ycb_flag + hand_flag (EE frame)
        ├── robot_states   float32 [T, 32]        joint_pos(9)+joint_vel(9)+ee_pose(7)+gripper(1)+prev_act(6)
        └── expert_actions float32 [T, 7]         Δpos(3)+Δeuler(3)+gripper_cmd(1, binary)
        attrs: scene_idx, num_steps

Phase 1 is single-frame, so `BCDataset.__getitem__` returns a single
(pc, robot_state, expert_action) tuple. Episodes are flattened into a flat
list of (episode_key, step) pairs at construction time.

Normalization:
  • robot_state — all 32 channels, per-channel mean/std from train split.
  • action[:6]  — continuous Δpos+Δeuler, per-channel mean/std.
  • action[6]   — binary gripper command, *never* normalized (it's the BCE target).
  • point_cloud — never normalized (xyz already in EE frame; flags are 0/1).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


GOAL_DIM = 8   # [quat_wxyz(4) ‖ trans(3) ‖ valid(1)]


def load_goal_table(path) -> dict[int, np.ndarray]:
    """scene_idx -> 4x4 world pose of that scene's pinned goal grasp.

    Reads the pin table written by examples/build_grasp_pin_table.py — the same
    file SIM.grasp_pin_table points at, so the auxiliary target is by construction
    the pose the demonstrations were labelled towards and the evaluator scores
    against. Scene indices are SPLIT-RELATIVE, so a train table must not be paired
    with val data.
    """
    import json
    with open(path) as f:
        raw = json.load(f)
    out: dict[int, np.ndarray] = {}
    for k, v in raw.items():
        if k == "_meta" or v is None:
            continue
        out[int(k)] = np.asarray(v["ee_pose_world"], dtype=np.float64)
    if not out:
        raise RuntimeError(f"no usable entries in pin table {path}")
    return out


def _se3_inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def goal_target_from_state(rs_raw: np.ndarray, grasp_world: np.ndarray) -> np.ndarray:
    """[quat_wxyz(4) ‖ trans(3) ‖ 1.0] — the goal grasp IN THE CURRENT EE FRAME.

    `rs_raw` is the UN-normalized robot_state; channels 18:21 are ee_xyz and
    21:25 are ee_wxyz (see the module docstring for the layout). Same quantity
    GA-DDPG feeds its aux head — `inv_relative_pose(cur_goal, ef_pose)` — minus
    their rotZ(pi/2), which exists only for their grasp-frame convention. Ours is
    already the hand-link frame the control points and the pin table share, so no
    extra rotation belongs here.

    Not normalized: the PM loss that consumes this works in real metres, which is
    the entire reason it is a meaningful 6-DoF metric.
    """
    from transforms3d.quaternions import mat2quat, quat2mat

    ee = np.eye(4, dtype=np.float64)
    ee[:3, 3] = np.asarray(rs_raw[18:21], dtype=np.float64)
    ee[:3, :3] = quat2mat(np.asarray(rs_raw[21:25], dtype=np.float64))

    rel = _se3_inverse(ee) @ np.asarray(grasp_world, dtype=np.float64)
    return np.concatenate([mat2quat(rel[:3, :3]), rel[:3, 3], [1.0]]).astype(np.float32)


def _as_path_list(paths) -> list[str]:
    """Normalize a single path or an iterable of paths into a list[str].

    Accepts a str / os.PathLike (treated as one file) or any iterable of them
    (e.g. the DAgger aggregate: [train.h5, dagger_iter1.h5, ...]).
    """
    if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
        return [str(paths)]
    return [str(p) for p in paths]


# ── normalization ────────────────────────────────────────────────────────────

class Normalizer:
    """Per-channel mean/std for robot_state and the continuous part of action.

    Stores numpy arrays internally; accepts/returns torch tensors for the
    apply/invert methods. A device cache avoids re-uploading the stats every
    __getitem__ when training on GPU.
    """

    def __init__(self,
                 state_mean: np.ndarray,
                 state_std:  np.ndarray,
                 action_mean: np.ndarray,
                 action_std:  np.ndarray,
                 eps: float = 1e-6):
        self.state_mean  = np.asarray(state_mean,  dtype=np.float32)
        self.state_std   = np.maximum(np.asarray(state_std,  dtype=np.float32), eps)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)   # [6]
        self.action_std  = np.maximum(np.asarray(action_std, dtype=np.float32), eps)
        assert self.state_mean.shape  == (32,), self.state_mean.shape
        assert self.action_mean.shape == (6,),  self.action_mean.shape
        self._torch_cache: dict = {}

    # ----- internal: lazy torch view of the stats on the right device -------
    def _t(self, name: str, ref: torch.Tensor) -> torch.Tensor:
        device = ref.device
        key = (name, device)
        if key not in self._torch_cache:
            self._torch_cache[key] = torch.from_numpy(getattr(self, name)).to(device)
        return self._torch_cache[key]

    # ----- robot state ------------------------------------------------------
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._t("state_mean", x)) / self._t("state_std", x)

    def denormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._t("state_std", x) + self._t("state_mean", x)

    # ----- action -----------------------------------------------------------
    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 7]. Normalizes channels 0:6, leaves channel 6 (gripper) alone."""
        cont = (x[..., :6] - self._t("action_mean", x)) / self._t("action_std", x)
        return torch.cat([cont, x[..., 6:7]], dim=-1)

    def denormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        cont = x[..., :6] * self._t("action_std", x) + self._t("action_mean", x)
        return torch.cat([cont, x[..., 6:7]], dim=-1)

    # ----- (de)serialization -----------------------------------------------
    def save(self, path: str) -> None:
        np.savez(
            path,
            state_mean=self.state_mean,
            state_std=self.state_std,
            action_mean=self.action_mean,
            action_std=self.action_std,
        )

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        d = np.load(path)
        return cls(
            state_mean=d["state_mean"],
            state_std=d["state_std"],
            action_mean=d["action_mean"],
            action_std=d["action_std"],
        )


def compute_normalization_stats(hdf5_paths) -> Normalizer:
    """Single streaming pass over the train HDF5 file(s) → Normalizer.

    `hdf5_paths` may be one path or a list of paths (the DAgger aggregate);
    stats are pooled across all files. Uses Welford-style sums of x and x² to
    avoid loading the full dataset into memory. Only channels 0–5 of
    expert_actions contribute to action stats (channel 6 is binary).
    """
    paths = _as_path_list(hdf5_paths)
    sum_s    = np.zeros(32, dtype=np.float64)
    sum_s_sq = np.zeros(32, dtype=np.float64)
    sum_a    = np.zeros(6,  dtype=np.float64)
    sum_a_sq = np.zeros(6,  dtype=np.float64)
    n = 0

    for path in paths:
        with h5py.File(path, "r") as f:
            ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
            if not ep_keys:
                raise RuntimeError(f"No episodes found in {path}")
            for k in ep_keys:
                rs  = f[k]["robot_states"][:].astype(np.float64)    # [T, 32]
                act = f[k]["expert_actions"][:].astype(np.float64)  # [T, 7]
                sum_s    += rs.sum(axis=0)
                sum_s_sq += (rs * rs).sum(axis=0)
                sum_a    += act[:, :6].sum(axis=0)
                sum_a_sq += (act[:, :6] * act[:, :6]).sum(axis=0)
                n += rs.shape[0]

    if n == 0:
        raise RuntimeError(f"All episodes in {paths} were empty")

    state_mean  = sum_s / n
    state_var   = np.maximum(sum_s_sq / n - state_mean ** 2, 0.0)
    state_std   = np.sqrt(state_var)

    action_mean = sum_a / n
    action_var  = np.maximum(sum_a_sq / n - action_mean ** 2, 0.0)
    action_std  = np.sqrt(action_var)

    return Normalizer(state_mean, state_std, action_mean, action_std)


# ── dataset ──────────────────────────────────────────────────────────────────

class BCDataset(Dataset):
    """Flat single-frame view over one or more BC HDF5 files.

    `hdf5_paths` may be a single path or a list of paths (the DAgger aggregate
    [train.h5, dagger_iter1.h5, ...]). Episodes from all files are pooled.

    `len(dataset)` = total number of policy steps across all episodes/files.
    `dataset[i]` returns one (point_cloud, robot_state, expert_action) tuple
    as float32 tensors on CPU. Robot state and action[:6] are normalized
    in-place when a Normalizer is provided; point cloud and action[6] are
    left untouched.

    Each file is opened lazily on first __getitem__ in each worker so that
    PyTorch DataLoader fork-based parallelism doesn't race on a shared handle.
    Episode keys can collide across files (every file starts at episode_00000),
    so the flat index keys on (file_idx, episode_key, step).
    """

    def __init__(self, hdf5_paths, normalizer: Normalizer | None = None,
                 goal_table=None):
        self.hdf5_paths = _as_path_list(hdf5_paths)
        self.normalizer = normalizer
        # Auxiliary goal-grasp target (run 13). When enabled, __getitem__ returns a
        # 4th element. Reconstructed per step from the stored ee_pose, so NO
        # RE-COLLECTION is needed: every dataset already carries `scene_idx` per
        # episode, and the handover config is static (MANO_SIMULATION_MODE:
        # disable_control_and_move_by_reset), so a world-frame grasp pose stays
        # valid for the whole episode.
        #
        # `goal_table` accepts:
        #   None      off — three-element items, exactly as before
        #   "auto"    resolve PER FILE from that file's `grasp_pin_table` attr.
        #             Preferred: scene indices are SPLIT-RELATIVE, so this is what
        #             makes it impossible to score val episodes against the train
        #             table. Requires the attr (base collector always writes it;
        #             DAgger shards do from run 12 on).
        #   str/Path  one table for every file
        #   dict      a pre-loaded {scene_idx: 4x4}
        self.goal_table = goal_table
        self._goal_tables: list[dict[int, np.ndarray]] | None = None
        if goal_table is not None:
            if isinstance(goal_table, dict):
                self._goal_tables = [goal_table] * len(self.hdf5_paths)
            elif str(goal_table) == "auto":
                self._goal_tables = []
                for p in self.hdf5_paths:
                    with h5py.File(p, "r") as f:
                        tp = f.attrs.get("grasp_pin_table", "")
                    tp = tp.decode() if isinstance(tp, bytes) else str(tp or "")
                    if not tp:
                        raise RuntimeError(
                            f"{p} has no `grasp_pin_table` attr, so the auxiliary "
                            f"goal target cannot be resolved automatically. Pass an "
                            f"explicit DATA.grasp_pin_table, or re-collect.")
                    self._goal_tables.append(load_goal_table(tp))
                    print(f"[aux] {Path(p).name} -> {tp}")
            else:
                one = load_goal_table(goal_table)
                self._goal_tables = [one] * len(self.hdf5_paths)

        # Build a flat (file_idx, episode_key, step) index using one-shot file
        # handles. Don't keep the handles around past __init__ — see
        # _ensure_open below.
        index: list[tuple[int, str, int]] = []
        # episode -> scene_idx, read once here rather than per __getitem__.
        scene_of: dict[tuple[int, str], int] = {}
        for fi, path in enumerate(self.hdf5_paths):
            with h5py.File(path, "r") as f:
                ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
                if not ep_keys:
                    raise RuntimeError(f"No episodes found in {path}")
                for k in ep_keys:
                    T = int(f[k].attrs["num_steps"])
                    if goal_table is not None:
                        sc = f[k].attrs.get("scene_idx")
                        if sc is None:
                            raise RuntimeError(
                                f"{path}:{k} has no scene_idx attr — the auxiliary "
                                f"goal target cannot be reconstructed without it")
                        scene_of[(fi, k)] = int(sc)
                    for t in range(T):
                        index.append((fi, k, t))
        self._index = index
        self._scene_of = scene_of
        if self._goal_tables is not None:
            missing = {(fi, s) for (fi, _), s in scene_of.items()
                       if s not in self._goal_tables[fi]}
            if missing:
                print(f"WARNING: {len(missing)} (file, scene) pair(s) have no "
                      f"pin-table entry — those goal targets are masked out "
                      f"(valid=0) and contribute nothing to the aux loss")
        # One handle slot per file, opened per-worker on first use.
        self._files: list[h5py.File | None] = [None] * len(self.hdf5_paths)

    # ----- per-worker lazy open --------------------------------------------
    def _ensure_open(self, file_idx: int) -> h5py.File:
        if self._files[file_idx] is None:
            self._files[file_idx] = h5py.File(self.hdf5_paths[file_idx], "r")
        return self._files[file_idx]

    def __getstate__(self):
        # Don't pickle the open file handles (DataLoader spawn-mode safety).
        state = self.__dict__.copy()
        state["_files"] = [None] * len(self.hdf5_paths)
        return state

    # ----- standard Dataset API --------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        fi, ep_key, t = self._index[idx]
        f = self._ensure_open(fi)
        grp = f[ep_key]

        pc  = torch.from_numpy(grp["point_clouds"][t]).float()       # [1024, 5]
        rs_raw = grp["robot_states"][t]                              # [32], RAW
        rs  = torch.from_numpy(rs_raw).float()
        act = torch.from_numpy(grp["expert_actions"][t]).float()     # [7]

        # Built from the RAW state, before normalization touches the ee_pose.
        goal = None
        if self._goal_tables is not None:
            grasp = self._goal_tables[fi].get(self._scene_of[(fi, ep_key)])
            goal = (torch.from_numpy(goal_target_from_state(rs_raw, grasp))
                    if grasp is not None
                    else torch.zeros(GOAL_DIM, dtype=torch.float32))   # valid=0

        if self.normalizer is not None:
            rs  = self.normalizer.normalize_state(rs)
            act = self.normalizer.normalize_action(act)

        if goal is None:
            return pc, rs, act
        return pc, rs, act, goal

    # ----- convenience ------------------------------------------------------
    @property
    def num_episodes(self) -> int:
        return len({(fi, ep) for fi, ep, _ in self._index})

    def episode_counts(self) -> list[tuple[str, int]]:
        """(path, num_episodes) per file — for logging the aggregate."""
        return [
            (path, len({ep for fi, ep, _ in self._index if fi == i}))
            for i, path in enumerate(self.hdf5_paths)
        ]
