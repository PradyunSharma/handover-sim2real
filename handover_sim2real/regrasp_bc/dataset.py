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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import h5py                                 # annotations only; see below
else:
    try:
        import h5py
    except ModuleNotFoundError:                 # pragma: no cover - deployment only
        # Only BCDataset and compute_normalization_stats touch HDF5. Deployment
        # environments (the robot PC runs handover_sim2real/sim2real/*) import
        # this module solely for `Normalizer`, which is four numpy arrays, and
        # have no reason to carry h5py. Leaving it None lets the import succeed
        # there; every real use below still fails on the attribute access, with
        # h5py named in the traceback.
        h5py = None
import numpy as np
import torch
from torch.utils.data import Dataset


GOAL_DIM = 8   # [quat_wxyz(4) ‖ trans(3) ‖ valid(1)]
COND_DIM = 9   # [rot6d(6) ‖ trans(3)] — the CONDITIONING vector (Phase 5)


def load_goal_table(path) -> dict[tuple[int, int], np.ndarray]:
    """(scene_idx, grasp_idx) -> 4x4 world pose of that pinned goal grasp.

    Reads the pin table written by examples/build_direction_table.py — the
    same file SIM.grasp_pin_table points at, so the target is by construction the
    pose the demonstrations were labelled towards and the evaluator scores
    against. Scene indices are SPLIT-RELATIVE, so a train table must not be paired
    with val data.

    Phase-4 tables (one grasp per scene) load as `(scene, 0)`.

    Note this is now only a FALLBACK: Phase-5 datasets carry `grasp_pose_world`
    on every episode group, so `BCDataset` reads the pose off the episode and
    never consults a table. That removes the failure mode where rebuilding a pin
    table silently retargets a dataset collected against the old one.
    """
    import json
    with open(path) as f:
        raw = json.load(f)
    out: dict[tuple[int, int], np.ndarray] = {}
    for k, v in raw.items():
        if k == "_meta" or v is None:
            continue
        grasps = v["grasps"] if isinstance(v, dict) and "grasps" in v else [v]
        for gi, g in enumerate(grasps):
            out[(int(k), gi)] = np.asarray(g["ee_pose_world"], dtype=np.float64)
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


def goal_cond_from_state(rs_raw: np.ndarray, grasp_world: np.ndarray) -> np.ndarray:
    """[rot6d(6) ‖ trans(3)] — the goal grasp IN THE CURRENT EE FRAME, as the
    Phase-5 policy's conditioning INPUT.

    Same geometry as `goal_target_from_state`; the rotation is carried as the
    first two columns of R (Zhou et al. 2019) instead of a quaternion. The reason
    is the direction of use: as a regression target the quaternion's double cover
    (q and -q name the same rotation) is handled by the geodesic/PM loss, but as a
    network INPUT it makes the function the encoder must learn discontinuous —
    two inputs that are maximally far apart in R^4 must map to the same action.
    rot6d has no such seam and needs no normalization.

    Recomputed every step, because the EE moves even though the world-frame grasp
    does not. That is the point: the conditioning is the residual pose the policy
    still has to null out.
    """
    from transforms3d.quaternions import quat2mat

    ee = np.eye(4, dtype=np.float64)
    ee[:3, 3] = np.asarray(rs_raw[18:21], dtype=np.float64)
    ee[:3, :3] = quat2mat(np.asarray(rs_raw[21:25], dtype=np.float64))

    rel = _se3_inverse(ee) @ np.asarray(grasp_world, dtype=np.float64)
    rot6d = rel[:3, :2].T.reshape(-1)          # columns 0 and 1 of R, flattened
    return np.concatenate([rot6d, rel[:3, 3]]).astype(np.float32)


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
                 goal_table=None, reach_tail_weight: float = 1.0,
                 reach_tail: int = 5, grasp_cond: bool = False):
        self.hdf5_paths = _as_path_list(hdf5_paths)
        self.normalizer = normalizer
        # Oversampling weight for the last `reach_tail` steps of every episode —
        # the standoff->grasp reach plus the CLOSE label. 1.0 leaves the dataset
        # uniform and `sample_weights` None, so nothing downstream changes.
        #
        # WHY. Measured on train_pinned_omg_ok, those steps are 23.8% of the data
        # but contribute only 11.1% of the SmoothL1 pose gradient, because the
        # expert's labels shrink into the reach (per-axis mean |dpos| 0.0120 m
        # over the last five steps against 0.0255 m during the free approach, and
        # |drot| 0.0155 rad against 0.0497). Under a loss that is quadratic in the
        # small-error regime, the phase that decides the task is the phase the
        # optimizer can most cheaply ignore.
        self.reach_tail_weight = float(reach_tail_weight)
        self.reach_tail = int(reach_tail)
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
        #
        # PHASE 5. `grasp_cond` turns the same quantity into a network INPUT
        # rather than an auxiliary target, and it is not optional the way the aux
        # head was: with four pinned grasps per scene the observation is genuinely
        # ambiguous without it. Either flag makes __getitem__ return a 5-tuple
        # (pc, rs, act, goal8, cond9), with zeros in whichever of the two is off,
        # so the arity is fixed and the trainer branches on flags rather than on
        # tuple length.
        #
        # The world-frame grasp is read from each episode's own
        # `grasp_pose_world` attr when present (Phase-5 collections always write
        # it), and only falls back to a pin-table lookup for older files. That is
        # deliberate: a dataset that carries its own target cannot be silently
        # retargeted by rebuilding the table it was collected against.
        self.grasp_cond = bool(grasp_cond)
        self.goal_table = goal_table
        self._goal_tables: list[dict[tuple[int, int], np.ndarray]] | None = None
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
        # episode -> the 4x4 world grasp pose it aimed at. Resolved once, so
        # __getitem__ only does the EE-frame transform.
        grasp_of: dict[tuple[int, str], np.ndarray] = {}
        want_grasp = grasp_cond or (goal_table is not None)
        n_from_attr = n_from_table = n_unresolved = 0
        # Per-item sampling weight, aligned with `index`. Built here because
        # step-from-end needs the episode length, which is only known while the
        # index is being walked.
        weights: list[float] = []
        for fi, path in enumerate(self.hdf5_paths):
            with h5py.File(path, "r") as f:
                ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
                if not ep_keys:
                    raise RuntimeError(f"No episodes found in {path}")
                for k in ep_keys:
                    T = int(f[k].attrs["num_steps"])
                    if want_grasp:
                        sc = f[k].attrs.get("scene_idx")
                        if sc is None:
                            raise RuntimeError(
                                f"{path}:{k} has no scene_idx attr — the goal "
                                f"grasp cannot be reconstructed without it")
                        scene_of[(fi, k)] = int(sc)
                        pose = f[k].attrs.get("grasp_pose_world")
                        if pose is not None:
                            grasp_of[(fi, k)] = np.asarray(pose, dtype=np.float64)
                            n_from_attr += 1
                        elif self._goal_tables is not None:
                            gi = int(f[k].attrs.get("grasp_idx", 0))
                            g = self._goal_tables[fi].get((int(sc), gi))
                            if g is not None:
                                grasp_of[(fi, k)] = g
                                n_from_table += 1
                            else:
                                n_unresolved += 1
                        else:
                            n_unresolved += 1
                    for t in range(T):
                        index.append((fi, k, t))
                        weights.append(self.reach_tail_weight
                                       if (T - 1 - t) < self.reach_tail else 1.0)
        self._index = index
        self._scene_of = scene_of
        self._grasp_of = grasp_of
        # None when uniform, so callers can branch on "is this dataset weighted"
        # without comparing floats.
        self.sample_weights = weights if self.reach_tail_weight != 1.0 else None
        if self.sample_weights is not None:
            n_tail = sum(1 for w in weights if w != 1.0)
            share = self.reach_tail_weight * n_tail / (
                self.reach_tail_weight * n_tail + (len(weights) - n_tail))
            print(f"[reach-weight] last {self.reach_tail} steps x"
                  f"{self.reach_tail_weight}: {n_tail}/{len(weights)} items "
                  f"({100*n_tail/len(weights):.1f}% of the data) -> "
                  f"{100*share:.1f}% of draws")
        if want_grasp:
            print(f"[goal] grasp poses resolved: {n_from_attr} from episode attrs, "
                  f"{n_from_table} from pin tables, {n_unresolved} unresolved")
            if n_unresolved:
                # Masked out (valid=0) rather than fatal, exactly as the aux head
                # already did. Under grasp_cond that is a much bigger deal — a
                # zeroed conditioning vector is a lie, not a missing label — so it
                # is loud here and the trainer refuses to start on it.
                print(f"WARNING: {n_unresolved} episode(s) have no resolvable goal "
                      f"grasp. Their conditioning would be all-zero, which the "
                      f"policy cannot distinguish from 'the grasp is exactly at "
                      f"the gripper'. Re-collect or supply DATA.grasp_pin_table.")
                if self.grasp_cond:
                    raise RuntimeError(
                        f"{n_unresolved} episode(s) have no goal grasp and "
                        f"MODEL.grasp_cond is on — refusing to train on "
                        f"fabricated conditioning.")
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
        # Fixed 5-tuple arity whenever either the aux target or the conditioning
        # is wanted; the unused one is zeros. See __init__ for why.
        if self._goal_tables is None and not self.grasp_cond:
            if self.normalizer is not None:
                rs  = self.normalizer.normalize_state(rs)
                act = self.normalizer.normalize_action(act)
            return pc, rs, act

        grasp = self._grasp_of.get((fi, ep_key))
        if grasp is None:
            goal = torch.zeros(GOAL_DIM, dtype=torch.float32)          # valid=0
            cond = torch.zeros(COND_DIM, dtype=torch.float32)
        else:
            goal = (torch.from_numpy(goal_target_from_state(rs_raw, grasp))
                    if self._goal_tables is not None
                    else torch.zeros(GOAL_DIM, dtype=torch.float32))
            cond = (torch.from_numpy(goal_cond_from_state(rs_raw, grasp))
                    if self.grasp_cond
                    else torch.zeros(COND_DIM, dtype=torch.float32))

        if self.normalizer is not None:
            rs  = self.normalizer.normalize_state(rs)
            act = self.normalizer.normalize_action(act)

        return pc, rs, act, goal, cond

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
