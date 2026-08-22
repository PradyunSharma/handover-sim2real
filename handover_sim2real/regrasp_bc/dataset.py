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

# Pure numpy, no simulator: regrasp/__init__.py resolves its re-exports lazily,
# so importing these in a DataLoader worker costs nothing beyond scipy.
from handover_sim2real.regrasp import channels as _rg_channels
from handover_sim2real.regrasp import directions as _rg_directions


GOAL_DIM = 8   # [quat_wxyz(4) ‖ trans(3) ‖ valid(1)] — the AUX HEAD's target.
               # Legitimate again under Regrasp: the policy is no longer given the
               # goal pose, so predicting it is a real auxiliary task rather than
               # an identity map. Off by default; see bc_regrasp.yaml.

# What the FILE holds and what the MODEL eats are deliberately different widths.
# `d` is perturbed at training time, so a d-dependent channel cannot be baked in.
STORED_PC_CHANNELS = _rg_channels.STORED_CHANNELS   # 8: xyz ycb hand nx ny nz
MODEL_PC_CHANNELS = _rg_channels.MODEL_CHANNELS     # 7: xyz ycb hand d.n d.r


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


def direction_in_ee_frame(rs_raw: np.ndarray, d_world: np.ndarray) -> np.ndarray:
    """The commanded approach direction, rotated into the CURRENT EE frame.

    `d_world` is constant for an episode — the pinned grasp does not move — but
    the EE does, so this is recomputed every step, exactly as the Phase-5
    conditioning was. TRANSLATION IS IRRELEVANT: `d` is a direction, so only the
    EE's rotation enters. `rs_raw[21:25]` is the EE quaternion in wxyz.

    Deriving `d_ee` here rather than storing it per step is what lets the
    perturbation below be applied once per episode: the stored quantity is the
    command, and the frame change is a pure function of the state.
    """
    from transforms3d.quaternions import quat2mat

    R_ee = quat2mat(np.asarray(rs_raw[21:25], dtype=np.float64))
    return _rg_directions.normalize(R_ee.T @ np.asarray(d_world, dtype=np.float64))


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
                 reach_tail: int = 5, direction_cond: bool = True,
                 d_noise_deg: float = 0.0):
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
        # REGRASP. The conditioning is no longer a pose handed to an MLP; it is a
        # DIRECTION injected per-point into the cloud (regrasp/channels.py). What
        # the dataset must resolve per episode is therefore `d_world`, a single
        # unit vector, and `__getitem__` rotates it into the current EE frame and
        # appends the two dot-product channels.
        self.direction_cond = bool(direction_cond)
        # Perturb `d` by this many degrees, once per EPISODE (not per step), to
        # teach interpolation between bins and robustness to the fact that a
        # test-time bin centre never exactly matches a demonstration's realised
        # direction. MUST be 0 on the val set or val loss stops being comparable
        # across epochs — train_regrasp builds both from one config block.
        self.d_noise_deg = float(d_noise_deg)
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
        # episode -> the commanded approach direction in WORLD, constant per
        # episode. Resolved once here so __getitem__ only rotates it.
        dir_of: dict[tuple[int, str], np.ndarray] = {}
        want_grasp = goal_table is not None
        n_from_attr = n_from_table = n_unresolved = 0
        n_dir = n_dir_missing = 0
        widths: set[int] = set()
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
                    widths.add(int(f[k]["point_clouds"].shape[-1]))
                    if self.direction_cond:
                        dw = f[k].attrs.get("d_world")
                        if dw is None:
                            n_dir_missing += 1
                        else:
                            dir_of[(fi, k)] = _rg_directions.normalize(
                                np.asarray(dw, dtype=np.float64))
                            n_dir += 1
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
        self._dir_of = dir_of

        # CHANNEL WIDTH. Nothing in this repo compared the h5's width against the
        # model's before, and the failure mode is nasty: mixing a pre-Regrasp
        # 5-channel shard with an 8-channel one kills the DataLoader inside
        # `collate`, in a worker, with nothing naming the offending episode —
        # exactly the run-19 crash `select_regrasp_demos.py`'s docstring
        # describes. Fatal and specific, here, at construction.
        if len(widths) > 1:
            raise RuntimeError(
                f"point-cloud width differs across the aggregate: {sorted(widths)}. "
                f"A pre-Regrasp shard has been mixed with a Regrasp one; the "
                f"DataLoader would die inside collate with no episode named. "
                f"Files: {[str(p) for p in self.hdf5_paths]}")
        self.stored_pc_channels = widths.pop() if widths else 0
        if self.direction_cond and self.stored_pc_channels != STORED_PC_CHANNELS:
            raise RuntimeError(
                f"direction conditioning needs {STORED_PC_CHANNELS}-channel clouds "
                f"(xyz|ycb|hand|normal) but the data has "
                f"{self.stored_pc_channels}. Set DATA.pc_channels to "
                f"{MODEL_PC_CHANNELS} in the MODEL and re-collect with "
                f"examples/collect_regrasp_demos.py, which writes the normals.")
        if self.direction_cond:
            print(f"[direction] d_world resolved for {n_dir} episodes"
                  + (f", MISSING on {n_dir_missing}" if n_dir_missing else "")
                  + f"; clouds {self.stored_pc_channels}ch -> model "
                    f"{MODEL_PC_CHANNELS}ch"
                  + (f", d noise {self.d_noise_deg} deg" if self.d_noise_deg else ""))
            if n_dir_missing:
                raise RuntimeError(
                    f"{n_dir_missing} episode(s) have no `d_world` attr. A zeroed "
                    f"direction is not a missing label, it is a wrong one — the "
                    f"two conditioning channels would read 0 everywhere, which the "
                    f"policy cannot tell from 'approach from nowhere'. Re-collect.")
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

        pc_raw = grp["point_clouds"][t]                              # [1024, 8]
        rs_raw = grp["robot_states"][t]                              # [32], RAW
        rs  = torch.from_numpy(rs_raw).float()
        act = torch.from_numpy(grp["expert_actions"][t]).float()     # [7]

        if self.direction_cond:
            d_world = self._dir_of[(fi, ep_key)]
            if self.d_noise_deg:
                # SEEDED ON THE EPISODE, not the step. The augmentation means
                # "the commanded direction is N degrees off", which is a property
                # of the command; a per-step draw would instead mean "the command
                # jitters within the episode", which is a different and much less
                # plausible corruption. Deriving the seed from the key rather than
                # drawing from global state also keeps a DataLoader worker's
                # output reproducible.
                seed = (hash((fi, ep_key)) ^ 0x9E3779B9) & 0x7FFFFFFF
                d_world = _rg_channels.perturb_direction(
                    d_world, self.d_noise_deg, np.random.default_rng(seed))
            d_ee = direction_in_ee_frame(rs_raw, d_world)
            # [N,8] -> [N,7]: the stored normals are used as-is, so this matches
            # what BCRunner.act computes at inference bit-for-bit.
            pc = torch.from_numpy(
                _rg_channels.append_direction_channels(pc_raw, d_ee)).float()
        else:
            pc = torch.from_numpy(np.asarray(pc_raw)).float()

        # The aux target is built from the RAW state, before normalization touches
        # the ee_pose. It is a plain 4-tuple now: with the conditioning living in
        # the cloud there is no second vector to keep the arity fixed for.
        goal = None
        if self._goal_tables is not None:
            grasp = self._grasp_of.get((fi, ep_key))
            goal = (torch.from_numpy(goal_target_from_state(rs_raw, grasp))
                    if grasp is not None
                    else torch.zeros(GOAL_DIM, dtype=torch.float32))   # valid=0

        if self.normalizer is not None:
            rs  = self.normalizer.normalize_state(rs)
            act = self.normalizer.normalize_action(act)

        return (pc, rs, act) if goal is None else (pc, rs, act, goal)

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
