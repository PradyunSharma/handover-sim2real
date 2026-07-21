"""
Off-policy replay buffer for Phase-3 RL.

A fixed-capacity ring buffer of transitions. Point clouds dominate memory
(`[N, C]` float32 per state, stored for both `s` and `s'`), so capacity is the
main memory knob: `cap` transitions ≈ `cap · N · C · 4 · 2` bytes (e.g. 20k ·
1024 · 5 · 4 · 2 ≈ 0.8 GB).

Each transition carries the standard off-policy tuple plus three Phase-3 extras:
  • `remain_norm` / `next_remain_norm` — the in-state clock (see rl.actor).
  • `terminal` — 1.0 on a TRUE episode end (success / failure / horizon reached).
     Because the clock is in the state, hitting the step horizon is a genuine
     terminal, so there is no separate "truncated" flag: `target = r` whenever
     `terminal`, else bootstrap.
  • `mc_return` — the discounted reward-to-go, precomputed per episode by the
     rollout worker; the critic target can blend Bellman with this to propagate
     rare sparse successes faster (see rl.td3bc_trainer `mc_blend`).
  • `expert_action` / `expert_flag` — the OMG pose label + whether it is valid,
     used by the actor's (pose-only) BC term.
  • `expert_gripper` / `gripper_flag` — a proximity-synthesized gripper label
     (P(open): 1 when the EE is far from the grasp, 0 when within grasp
     proximity) + whether the grasp pose is known so the label is valid. Feeds
     the actor's gripper BCE term. Training-only supervision that AGREES with the
     reward; the policy still executes its OWN logit at deployment.

Actions are stored in **normalized** space (the actor's output space).
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, num_pts: int, pc_channels: int,
                 robot_state_dim: int = 32, action_dim: int = 7,
                 expert_action_dim: int = 6, goal_dim: int = 9):
        self.capacity = int(capacity)
        self.num_pts  = int(num_pts)
        self.pc_ch    = int(pc_channels)
        self.rs_dim   = int(robot_state_dim)
        self.a_dim    = int(action_dim)             # executed action: pose(6)+gripper(1)
        self.ea_dim   = int(expert_action_dim)      # OMG pose label for the BC term (6)
        self.goal_dim = int(goal_dim)

        c, N, C = self.capacity, self.num_pts, self.pc_ch
        self.pc        = np.zeros((c, N, C), dtype=np.float32)
        self.rs        = np.zeros((c, self.rs_dim), dtype=np.float32)
        self.remain    = np.zeros((c, 1), dtype=np.float32)
        self.action    = np.zeros((c, self.a_dim), dtype=np.float32)
        self.reward    = np.zeros((c, 1), dtype=np.float32)
        self.next_pc   = np.zeros((c, N, C), dtype=np.float32)
        self.next_rs   = np.zeros((c, self.rs_dim), dtype=np.float32)
        self.next_remain = np.zeros((c, 1), dtype=np.float32)
        self.terminal  = np.zeros((c, 1), dtype=np.float32)
        self.mc_return = np.zeros((c, 1), dtype=np.float32)
        self.expert_action  = np.zeros((c, self.ea_dim), dtype=np.float32)
        self.expert_flag    = np.zeros((c, 1), dtype=np.float32)
        self.goal_pose      = np.zeros((c, self.goal_dim), dtype=np.float32)
        # EE-relative grasp at the NEXT state — Φ(s') for potential-based reward
        # shaping (trainer). Defaults to goal_pose when a caller omits it.
        self.next_goal_pose = np.zeros((c, self.goal_dim), dtype=np.float32)
        # proximity-synthesized gripper label (P(open): 1 far / 0 near) + validity
        self.expert_gripper = np.zeros((c, 1), dtype=np.float32)
        self.gripper_flag   = np.zeros((c, 1), dtype=np.float32)

        self._idx  = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def add(self, *, pc, rs, remain_norm, action, reward, next_pc, next_rs,
            next_remain_norm, terminal, mc_return, expert_action, expert_flag,
            goal_pose, next_goal_pose=None, expert_gripper=1.0, gripper_flag=0.0):
        i = self._idx
        self.pc[i]          = pc
        self.rs[i]          = rs
        self.remain[i, 0]   = remain_norm
        self.action[i]      = action
        self.reward[i, 0]   = reward
        self.next_pc[i]     = next_pc
        self.next_rs[i]     = next_rs
        self.next_remain[i, 0] = next_remain_norm
        self.terminal[i, 0] = float(terminal)
        self.mc_return[i, 0] = mc_return
        self.expert_action[i] = expert_action
        self.expert_flag[i, 0] = float(expert_flag)
        self.goal_pose[i]   = goal_pose
        self.next_goal_pose[i] = goal_pose if next_goal_pose is None else next_goal_pose
        self.expert_gripper[i, 0] = float(expert_gripper)
        self.gripper_flag[i, 0]   = float(gripper_flag)

        self._idx += 1
        if self._idx >= self.capacity:
            self._idx = 0
            self._full = True

    def add_episode(self, transitions: list[dict]) -> None:
        for t in transitions:
            self.add(**t)

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        n = len(self)
        idx = np.random.randint(0, n, size=int(batch_size))

        def t(a, dt=torch.float32):
            return torch.as_tensor(a[idx], dtype=dt, device=device)

        return {
            "pc":            t(self.pc),
            "rs":            t(self.rs),
            "remain":        t(self.remain),
            "action":        t(self.action),
            "reward":        t(self.reward),
            "next_pc":       t(self.next_pc),
            "next_rs":       t(self.next_rs),
            "next_remain":   t(self.next_remain),
            "terminal":      t(self.terminal),
            "mc_return":     t(self.mc_return),
            "expert_action": t(self.expert_action),
            "expert_flag":   t(self.expert_flag),
            "goal_pose":     t(self.goal_pose),
            "next_goal_pose": t(self.next_goal_pose),
            "expert_gripper": t(self.expert_gripper),
            "gripper_flag":   t(self.gripper_flag),
        }


# ── offline demo pool (npz) ──────────────────────────────────────────────────
# A permanent set of pure-OMG expert demonstrations (collected by
# examples/collect_rl_demos.py) that reach the grasp and CLOSE, so the buffer
# always carries the +1 successes the online expert path cannot generate. Loaded
# into its own ReplayBuffer and sampled at a fixed fraction alongside the online
# FIFO (train_rl.py) — DDPGfD-style, so the demos never get evicted or drown the
# online signal.

# the per-transition fields ReplayBuffer.add() accepts; extra npz keys (e.g.
# scene_idx, action_mean/std for the visualizer) are ignored on load.
_DEMO_BUFFER_FIELDS = (
    "pc", "rs", "remain_norm", "action", "reward", "next_pc", "next_rs",
    "next_remain_norm", "terminal", "mc_return", "expert_action", "expert_flag",
    "goal_pose", "next_goal_pose", "expert_gripper", "gripper_flag")


def save_demo_transitions(path: str, transitions: list[dict], extra: dict = None) -> None:
    """Stack a list of transition dicts (the rollout worker's format) into an
    npz. Each field becomes an array [M, ...]; scalars become [M]. `extra` adds
    file-level arrays (not per-transition), e.g. the action normalizer stats.

    Holds all transitions in RAM — fine for small pools. For a full multi-scene
    collection, stream with `DemoHDF5Writer` instead so memory stays bounded AND
    a crash/OOM (which SIGKILLs past any try/except) keeps what's on disk."""
    if not transitions:
        raise ValueError("save_demo_transitions: no transitions to save")
    keys = list(transitions[0].keys())
    stacked = {k: np.stack([np.asarray(t[k], dtype=np.float32) for t in transitions])
               for k in keys}
    if extra:
        stacked.update({k: np.asarray(v, dtype=np.float32) for k, v in extra.items()})
    np.savez_compressed(path, **stacked)


class DemoHDF5Writer:
    """Stream demo transitions to an HDF5 file, one episode at a time, so
    collection memory stays bounded regardless of scene count (like the BC
    collector's per-episode HDF5) and a mid-run crash/OOM leaves every already-
    written episode on disk. Each transition field becomes one resizable flat
    dataset `[M, ...]`; `extra` file-level arrays (action_mean/std) are written
    once at open. Read back by `load_demo_buffer` (training) and the visualizer."""

    def __init__(self, path: str, extra: dict = None):
        import h5py
        self._f = h5py.File(path, "w")
        self._dsets = None
        self._n = 0
        if extra:
            for k, v in extra.items():
                self._f.create_dataset(k, data=np.asarray(v, dtype=np.float32))

    def append(self, transitions: list[dict]) -> None:
        if not transitions:
            return
        keys = list(transitions[0].keys())
        stacked = {k: np.stack([np.asarray(t[k], dtype=np.float32) for t in transitions])
                   for k in keys}
        m = next(iter(stacked.values())).shape[0]
        if self._dsets is None:
            self._dsets = {}
            for k, arr in stacked.items():
                self._dsets[k] = self._f.create_dataset(
                    k, data=arr, maxshape=(None,) + arr.shape[1:],
                    chunks=True, compression="gzip")
        else:
            for k, arr in stacked.items():
                d = self._dsets[k]
                d.resize(self._n + m, axis=0)
                d[self._n:self._n + m] = arr
        self._n += m
        self._f.flush()          # persist each episode so an OOM can't lose it

    @property
    def num_transitions(self) -> int:
        return self._n

    def close(self) -> None:
        self._f.close()


def _reconstruct_next_goal(arrs: dict, fields: list) -> list:
    """Back-compat: demo pools collected before `next_goal_pose` existed only store
    `goal_pose`. Reconstruct Φ(s')'s input by shifting: next_goal_pose[i] =
    goal_pose[i+1] (the next transition in the episode; demos are stored in episode
    order). At a terminal it's unused (the shaping zeroes Φ(s') there), so just keep
    a valid value. No-op when the field is already present."""
    if "next_goal_pose" in arrs or "goal_pose" not in arrs:
        return fields
    gp = np.asarray(arrs["goal_pose"], dtype=np.float32)
    term = np.asarray(arrs["terminal"], dtype=np.float32).reshape(-1)
    ngp = np.roll(gp, -1, axis=0)                 # next transition's goal_pose
    ngp[term > 0.5] = gp[term > 0.5]              # terminal rows: unused, keep valid
    arrs["next_goal_pose"] = ngp
    return list(fields) + ["next_goal_pose"]


def load_demo_buffer(path: str) -> "ReplayBuffer":
    """Load a demo pool (`.h5` streamed by DemoHDF5Writer, or `.npz` from
    save_demo_transitions) into a full, non-evicting ReplayBuffer (capacity ==
    number of demos). Dims are inferred; non-buffer keys (scene_idx,
    action_mean/std) are ignored."""
    if str(path).endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "r") as f:
            M, N, C = f["pc"].shape
            buf = ReplayBuffer(
                capacity=max(int(M), 1), num_pts=int(N), pc_channels=int(C),
                robot_state_dim=int(f["rs"].shape[1]),
                action_dim=int(f["action"].shape[1]),
                expert_action_dim=int(f["expert_action"].shape[1]),
                goal_dim=int(f["goal_pose"].shape[1]))
            fields = [k for k in f.keys() if k in _DEMO_BUFFER_FIELDS]
            arrs = {k: f[k][:] for k in fields}     # one bulk read per field
        fields = _reconstruct_next_goal(arrs, fields)
        for i in range(int(M)):
            buf.add(**{k: arrs[k][i] for k in fields})
        return buf

    data = np.load(path)
    M, N, C = data["pc"].shape
    buf = ReplayBuffer(
        capacity=max(int(M), 1), num_pts=int(N), pc_channels=int(C),
        robot_state_dim=int(data["rs"].shape[1]),
        action_dim=int(data["action"].shape[1]),
        expert_action_dim=int(data["expert_action"].shape[1]),
        goal_dim=int(data["goal_pose"].shape[1]))
    fields = [k for k in data.files if k in _DEMO_BUFFER_FIELDS]
    arrs = {k: data[k] for k in fields}
    fields = _reconstruct_next_goal(arrs, fields)
    for i in range(int(M)):
        buf.add(**{k: arrs[k][i] for k in fields})
    return buf
