"""
Windowed (sequence) dataset for the Phase-2 ACT pipeline.

Phase-1 ``BCDataset`` returns one (pc, robot_state, action) tuple per step.
ACT needs, for each anchor step ``t`` of an episode:

    • an observation **history** of the last ``T`` frames  [t-T+1 .. t]
    • an action **chunk** of the next ``k`` actions          [t .. t+k-1]

Both are carved out of the *same* per-episode HDF5 arrays produced by
examples/collect_bc_dataset.py — **no re-collection needed**, the files already
store ``point_clouds/robot_states/expert_actions`` as ``[T_ep, …]`` per episode.

Windowing rules (windows never cross an episode boundary):
  • history: source step ``s_j = clamp(t-(T-1)+j, 0, T_ep-1)`` for j=0..T-1, so
    near the start of an episode the earliest frame is repeated (a "hold at the
    initial frame" pad). Ordered oldest→newest; the last slot is the current
    frame ``t``. No attention mask needed — every history slot is a real frame.
  • chunk: action ``a_j = expert_actions[t+j]`` for ``t+j < T_ep``; the rest are
    zero-padded and flagged via ``chunk_mask`` (the loss masks them out).

Normalization mirrors BCDataset: robot_state all 32 ch, action[:6] continuous,
action[6] (binary gripper) and the point cloud left untouched.

``__getitem__`` returns float32 CPU tensors:
    pc_hist     [T, N, 5]
    rs_hist     [T, 32]
    action_chunk[k, 7]
    chunk_mask  [k]        1.0 = real action, 0.0 = past-episode-end pad
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import Normalizer, _as_path_list


class BCSequenceDataset(Dataset):
    """Windowed view over one or more BC HDF5 files for ACT training.

    Args:
        hdf5_paths:  single path or list (pooled, e.g. DAgger aggregate).
        history_len: T — number of observation frames fed to the temporal encoder.
        chunk_len:   k — number of future actions predicted per step.
        normalizer:  optional Normalizer (same one used for the MLP baseline).

    One sample per (file, episode, anchor-step), so ``len`` equals the total
    number of policy steps — same as BCDataset, just richer windows.
    """

    def __init__(self,
                 hdf5_paths,
                 history_len: int = 4,
                 chunk_len: int = 8,
                 normalizer: Normalizer | None = None):
        self.hdf5_paths = _as_path_list(hdf5_paths)
        self.history_len = int(history_len)
        self.chunk_len   = int(chunk_len)
        self.normalizer  = normalizer
        assert self.history_len >= 1, "history_len must be >= 1"
        assert self.chunk_len   >= 1, "chunk_len must be >= 1"

        # Flat (file_idx, episode_key, anchor_step, num_steps) index. num_steps is
        # cached so __getitem__ can clamp/pad without reopening attrs every call.
        index: list[tuple[int, str, int, int]] = []
        for fi, path in enumerate(self.hdf5_paths):
            with h5py.File(path, "r") as f:
                ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
                if not ep_keys:
                    raise RuntimeError(f"No episodes found in {path}")
                for key in ep_keys:
                    T_ep = int(f[key].attrs["num_steps"])
                    for t in range(T_ep):
                        index.append((fi, key, t, T_ep))
        self._index = index
        self._files: list[h5py.File | None] = [None] * len(self.hdf5_paths)

    # ----- per-worker lazy open (mirrors BCDataset) ------------------------
    def _ensure_open(self, file_idx: int) -> h5py.File:
        if self._files[file_idx] is None:
            self._files[file_idx] = h5py.File(self.hdf5_paths[file_idx], "r")
        return self._files[file_idx]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = [None] * len(self.hdf5_paths)
        return state

    # ----- Dataset API ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        fi, ep_key, t, T_ep = self._index[idx]
        grp = self._ensure_open(fi)[ep_key]
        pcs = grp["point_clouds"]        # [T_ep, N, 5]
        rss = grp["robot_states"]        # [T_ep, 32]
        acts = grp["expert_actions"]     # [T_ep, 7]

        T, k = self.history_len, self.chunk_len

        # History: source steps, clamped to repeat the first frame at episode start.
        hist_steps = [min(max(t - (T - 1) + j, 0), T_ep - 1) for j in range(T)]
        # h5py fancy-indexing wants a sorted unique list; gather then reorder.
        uniq = sorted(set(hist_steps))
        pc_u = pcs[uniq]                 # [U, N, 5]
        rs_u = rss[uniq]                 # [U, 32]
        pos  = {s: i for i, s in enumerate(uniq)}
        pc_hist = np.stack([pc_u[pos[s]] for s in hist_steps]).astype(np.float32)
        rs_hist = np.stack([rs_u[pos[s]] for s in hist_steps]).astype(np.float32)

        # Chunk: next k actions, right-pad with zeros past episode end.
        n_valid = min(k, T_ep - t)
        chunk = np.zeros((k, 7), dtype=np.float32)
        chunk[:n_valid] = acts[t:t + n_valid].astype(np.float32)
        chunk_mask = np.zeros((k,), dtype=np.float32)
        chunk_mask[:n_valid] = 1.0

        pc_hist = torch.from_numpy(pc_hist)
        rs_hist = torch.from_numpy(rs_hist)
        chunk   = torch.from_numpy(chunk)
        chunk_mask = torch.from_numpy(chunk_mask)

        if self.normalizer is not None:
            rs_hist = self.normalizer.normalize_state(rs_hist)
            # Normalize only valid rows; padded rows stay 0 (and are masked).
            if n_valid > 0:
                chunk[:n_valid] = self.normalizer.normalize_action(chunk[:n_valid])

        return pc_hist, rs_hist, chunk, chunk_mask

    # ----- convenience (mirrors BCDataset) ---------------------------------
    @property
    def num_episodes(self) -> int:
        return len({(fi, ep) for fi, ep, _, _ in self._index})

    def episode_counts(self) -> list[tuple[str, int]]:
        return [
            (path, len({ep for fi, ep, _, _ in self._index if fi == i}))
            for i, path in enumerate(self.hdf5_paths)
        ]
