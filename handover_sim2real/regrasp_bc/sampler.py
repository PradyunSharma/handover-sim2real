"""
Temporal ensembling for ACT closed-loop execution.

At each timestep the policy predicts a chunk of k actions for absolute steps
[t .. t+k-1]. So the current step t is covered by up to k overlapping chunks,
predicted at creation times c0 = t, t-1, …, t-k+1. We combine them with ACT's
exponential weights.

Weight derivation (matches the reference ACT implementation): ACT weights a
prediction made at creation time c0 by ``exp(-m·c0)``; after normalizing over
the contributors for step t (which share the constant ``exp(-m·t)``), this is
equivalent to ``weight ∝ exp(m·age)`` with ``age = t - c0 ≥ 0``. So **older**
predictions get slightly more weight (temporal consistency); ``m`` small → near
uniform. m=0.01 over a k=8 chunk spans exp(0)…exp(0.07) ≈ uniform.

This combination is valid in our delta formulation: every chunk slot for
absolute step t targets the Δee-pose **in the EE frame at step t**, so the
different chunks all predict the same physical quantity (see plan/Context).

Pose channels (0..5) are averaged directly. The gripper channel (6) is expected
as a **probability** (sigmoid, from ACTPolicy.predict), averaged then
thresholded — never average the {0,1} bits.
"""

from __future__ import annotations

import numpy as np


class TemporalEnsembler:
    """Per-episode ring of recent chunk predictions; emits one action per step.

    Usage (per episode):
        ens = TemporalEnsembler(chunk_len=k, m=0.01)
        ens.reset()
        for each policy step:
            chunk = model.predict(...)[0].cpu().numpy()   # [k, 7], ch6 = prob
            action = ens.step(chunk)                      # [7], ch6 = {0,1}
    """

    def __init__(self, chunk_len: int, action_dim: int = 7,
                 m: float = 0.01, gripper_threshold: float = 0.5):
        self.k = int(chunk_len)
        self.action_dim = int(action_dim)
        self.m = float(m)
        self.gripper_threshold = float(gripper_threshold)
        self.reset()

    def reset(self) -> None:
        self._t = 0
        self._chunks: list[tuple[int, np.ndarray]] = []  # (created_t, chunk[k,7])

    def step(self, chunk: np.ndarray) -> np.ndarray:
        """Register the chunk predicted at the current step, return action for it.

        chunk: [k, action_dim] float, ch6 = gripper probability.
        returns: [action_dim] float, ch6 thresholded to {0,1}.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (self.k, self.action_dim):
            raise ValueError(f"chunk {chunk.shape} != {(self.k, self.action_dim)}")

        self._chunks.append((self._t, chunk))
        # Drop chunks too old to cover the current step.
        self._chunks = [(c0, ch) for c0, ch in self._chunks if c0 + self.k > self._t]

        preds, weights = [], []
        for c0, ch in self._chunks:
            j = self._t - c0                       # index of current step in this chunk
            if 0 <= j < self.k:
                preds.append(ch[j])
                weights.append(np.exp(self.m * j))  # j == age; older chunk → larger j
        preds = np.stack(preds, axis=0)            # [n, 7]
        w = np.asarray(weights, dtype=np.float32)
        w /= w.sum()

        action = (w[:, None] * preds).sum(axis=0)  # [7]
        out = action.copy()
        out[6] = 1.0 if action[6] >= self.gripper_threshold else 0.0
        self._t += 1
        return out
