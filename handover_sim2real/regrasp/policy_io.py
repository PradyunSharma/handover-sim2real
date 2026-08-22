"""
Load / export a policy run directory, and give the collector and the evaluator a
single per-step interface regardless of which policy they are driving.

Phase 4 is policy-agnostic: the DAgger loop (sample m trajectories, label every
visited state, aggregate, refit, evaluate) does not care whether the learner is
the Phase-1 single-frame MLP or the Phase-2 temporal/chunking ACT model. The
*kind* is inferred from the run config — `MODEL.chunk_len` present means ACT —
so there is no second place to keep in sync.

`PolicyRunner` hides the difference:

    runner.reset()                    # once per episode
    action = runner.act(pc, rs)       # [7]: dpos(3)+deuler(3)+gripper in {0,1}

The single-frame runner is stateless between steps. The ACT runner owns the
T-frame observation ring buffer and the chunk-execution strategy (temporal
ensembling / receding / open loop), so its bookkeeping advances exactly as it
would in deployment.

A run dir is the Phase-1/2 layout and stays self-contained:

    <run_dir>/config.yaml            resolved config (MODEL/DATA/... )
    <run_dir>/normalization.npz      the Normalizer this policy was fit with
    <run_dir>/checkpoints/{last,best}.pt
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from handover_sim2real.regrasp_bc import ACTPolicy, BCPolicy, Normalizer, TemporalEnsembler
from handover_sim2real.regrasp.channels import build_model_cloud
from handover_sim2real.regrasp_bc.dataset import direction_in_ee_frame


# ── which policy is this run? ────────────────────────────────────────────────

def policy_kind(run_cfg: dict) -> str:
    """"act" if the config describes a chunking/temporal policy, else "bc"."""
    m = run_cfg.get("MODEL", {})
    return "act" if ("chunk_len" in m or "history_len" in m) else "bc"


def build_policy(run_cfg: dict, normalizer: Normalizer | None):
    """Instantiate the policy a run config describes (no weights loaded).

    Mirrors examples/train_bc.py::build_model and examples/train_act.py::
    build_model, so a policy built here is state-dict-compatible with one
    trained by either script.
    """
    m, d = run_cfg["MODEL"], run_cfg["DATA"]
    if policy_kind(run_cfg) == "bc":
        return BCPolicy(
            pc_channels=int(d["pc_channels"]),
            robot_state_dim=int(d["robot_state_dim"]),
            action_dim=int(d["action_dim"]),
            feature_dim=int(m["feature_dim"]),
            robot_hidden=int(m["robot_hidden"]),
            policy_hidden=tuple(m["policy_hidden"]),
            pointnet_scale=int(m["pointnet_scale"]),
            pointnet_radius=float(m["pointnet_radius"]),
            pointnet_nclusters=int(m["pointnet_nclusters"]),
            use_prev_act=bool(m.get("use_prev_act", True)),
            drop_joint_state=bool(m.get("drop_joint_state", False)),
            joint_state_dim=int(m.get("joint_state_dim", 18)),
            freeze_pc=bool(m.get("freeze_pc", False)),
            # Must mirror train_bc.build_model: the aux head adds state_dict keys,
            # so a run-13 checkpoint will not strict-load into a policy built
            # without it. This is the path the collector and evaluator go through,
            # so omitting it here would make run 13's own checkpoints unloadable.
            aux_head=bool(m.get("aux_head", False)),
            aux_dim=int(m.get("aux_dim", 7)),
            aux_hidden=tuple(m.get("aux_hidden", (256, 256))),
            # Phase 5. Same lockstep requirement as the aux head, and with more
            # teeth: pc_channels changes the first conv's in_dim, so a mismatch
            # is a shape error at load rather than a silent behavioural
            # difference. That is the ONLY tensor whose shape depends on it.
            normalizer=normalizer,
        )
    return ACTPolicy(
        pc_channels=int(d["pc_channels"]),
        robot_state_dim=int(d["robot_state_dim"]),
        action_dim=int(d["action_dim"]),
        feature_dim=int(m["feature_dim"]),
        robot_hidden=int(m["robot_hidden"]),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        enc_layers=int(m["enc_layers"]),
        dec_layers=int(m["dec_layers"]),
        cvae_enc_layers=int(m.get("cvae_enc_layers", 2)),
        dropout=float(m.get("dropout", 0.1)),
        history_len=int(m["history_len"]),
        chunk_len=int(m["chunk_len"]),
        latent_dim=int(m["latent_dim"]),
        use_cvae=bool(m.get("use_cvae", True)),
        use_prev_act=bool(m.get("use_prev_act", False)),
        pointnet_scale=int(m["pointnet_scale"]),
        pointnet_radius=float(m["pointnet_radius"]),
        pointnet_nclusters=int(m["pointnet_nclusters"]),
        freeze_pc=bool(m.get("freeze_pc", False)),
        normalizer=normalizer,
    )


# ── per-step interface ───────────────────────────────────────────────────────

class PolicyRunner:
    """One action per step, whatever the policy underneath.

    REGRASP note: the commanded approach direction is set ONCE per attempt via
    `set_direction`, not passed to every `act`. The runner rotates it into the
    current EE frame and appends the two per-point channels itself, which keeps
    `act(pc, rs)` — and therefore every call site in the collector, the
    evaluator, the retry machine and the rollout scripts — unchanged in arity.

    `act` takes the BARE [N, 5] cloud, exactly what `_point_cloud` returns, and
    produces the [N, 7] the model eats. Normals are recomputed here rather than
    threaded in; `regrasp/normals.py` documents why that is bit-identical to what
    the collector stored, and what would break the equality.
    """

    kind = "?"

    def reset(self) -> None:
        raise NotImplementedError

    def set_direction(self, d_world) -> None:
        """The commanded approach direction in WORLD, or None.

        Set once per attempt, not per step: the direction is constant while the
        EE moves, and `act` re-expresses it in the current EE frame every call.
        Kept OFF the `act` signature for the same reason the goal pose was —
        threading it through four call sites is four chances to pass None and
        condition on nothing.
        """
        self.d_world = (None if d_world is None
                        else np.asarray(d_world, dtype=np.float64))

    def act(self, pc: np.ndarray, rs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        return self.kind


class BCRunner(PolicyRunner):
    """Phase-1 single-frame MLP: no history, no chunk, no ensembling."""

    kind = "bc"

    def __init__(self, model: BCPolicy, device: str):
        self.model = model
        self.device = device
        self.d_world = None

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, pc: np.ndarray, rs: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc)
        if pc.shape[-1] == 5:
            if self.d_world is None:
                raise RuntimeError(
                    "asked to act with no direction set — call "
                    "runner.set_direction(d_world) at the start of the attempt. "
                    "Conditioning on nothing is not a safe default: the two "
                    "channels would read 0 everywhere, which the policy cannot "
                    "tell from 'approach from nowhere'.")
            # RAW rs, exactly as BCDataset does at training time; the
            # normalization of rs happens inside predict(), after this.
            d_ee = direction_in_ee_frame(rs, self.d_world)
            pc, self.last_cloud_info = build_model_cloud(pc, d_ee)
        pc_t = torch.from_numpy(np.asarray(pc, dtype=np.float32)).unsqueeze(0).to(self.device)
        rs_t = torch.from_numpy(rs).float().unsqueeze(0).to(self.device)
        # BCPolicy.predict denormalizes ch0..5 and hard-thresholds ch6 to {0,1}.
        return self.model.predict(pc_t, rs_t)[0].cpu().numpy().astype(np.float32)

    def describe(self) -> str:
        return "bc (single frame, direction-conditioned)"


class ACTRunner(PolicyRunner):
    """Phase-2 ACT: T-frame history in, chunk of k out, executed per EXEC.mode."""

    kind = "act"

    def __init__(self, model: ACTPolicy, device: str, history_len: int,
                 chunk_len: int, exec_cfg: dict):
        self.model = model
        self.device = device
        self.T = int(history_len)
        self.k = int(chunk_len)
        self.mode = exec_cfg.get("mode", "ensemble")
        self.ensemble_m = float(exec_cfg.get("ensemble_m", 0.01))
        self.d_world = None
        self.reset()

    def set_direction(self, d_world) -> None:
        """REFUSES a direction rather than ignoring one.

        Under Phase 5 this was a harmless no-op: the conditioning was a separate
        tensor ACT simply never received. It is not harmless now — the command
        rides in the cloud's channels, and `dataset_seq`/`models_act` will happily
        consume a 7-channel cloud while nothing ever sets `d`. That would train
        and roll out on conditioning channels that are whatever the collector
        happened to write, with no error anywhere.
        """
        if d_world is not None:
            raise NotImplementedError(
                "ACT is not direction-conditioned: models_act.py has no path for "
                "the per-point command, so an ACT run would consume the channels "
                "without ever setting them. Use the single-frame BC policy, or "
                "add the conditioning to the ACT encoder first.")

    def reset(self) -> None:
        self.pc_buf: list[np.ndarray] = []
        self.rs_buf: list[np.ndarray] = []
        self.pending: list[np.ndarray] = []
        self.chunk: np.ndarray | None = None
        self.ens = (TemporalEnsembler(chunk_len=self.k, m=self.ensemble_m)
                    if self.mode == "ensemble" else None)
        if self.ens is not None:
            self.ens.reset()

    def _stack(self, buf: list[np.ndarray]) -> np.ndarray:
        """Last T entries, oldest->newest, left-padded by repeating the oldest."""
        recent = buf[-self.T:]
        if len(recent) < self.T:
            recent = [recent[0]] * (self.T - len(recent)) + recent
        return np.stack(recent, axis=0)

    @torch.no_grad()
    def act(self, pc: np.ndarray, rs: np.ndarray) -> np.ndarray:
        self.pc_buf.append(pc)
        self.rs_buf.append(rs)
        if self.mode != "open_loop" or not self.pending:
            pc_t = torch.from_numpy(self._stack(self.pc_buf)[None]).float().to(self.device)
            rs_t = torch.from_numpy(self._stack(self.rs_buf)[None]).float().to(self.device)
            self.chunk = self.model.predict(pc_t, rs_t)[0].cpu().numpy()  # [k,7], ch6=prob
        if self.mode == "ensemble":
            return self.ens.step(self.chunk).astype(np.float32)  # ch6 already {0,1}
        if self.mode == "receding":
            action = self.chunk[0].copy()
        else:  # open_loop
            if not self.pending:
                self.pending = [a.copy() for a in self.chunk]
            action = self.pending.pop(0)
        action[6] = 1.0 if action[6] >= 0.5 else 0.0
        return action.astype(np.float32)

    def describe(self) -> str:
        return f"act (T={self.T}, k={self.k}, exec={self.mode})"


# ── run-dir I/O ──────────────────────────────────────────────────────────────

def read_run_cfg(run_dir) -> dict:
    with (Path(run_dir) / "config.yaml").open() as f:
        return yaml.safe_load(f)


def load_policy_runner(run_dir, device: str, ckpt: str = "best"):
    """Load a trained policy and wrap it in the per-step interface.

    Args:
        run_dir: the run directory.
        device:  cuda | cpu.
        ckpt:    "best" | "last" | an explicit *.pt path. "best" silently falls
                 back to last.pt when the run had no validation set (in which
                 case best.pt is never written).

    Returns (runner, run_cfg).
    """
    run_dir = Path(run_dir)
    run_cfg = read_run_cfg(run_dir)

    norm_path = run_dir / "normalization.npz"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"{norm_path} is missing — the policy's action/state scaling is part "
            "of its definition; rolling out without it produces garbage."
        )
    normalizer = Normalizer.load(norm_path)

    if str(ckpt).endswith(".pt"):
        ckpt_path = Path(ckpt)
    else:
        ckpt_path = run_dir / "checkpoints" / f"{ckpt}.pt"
        if not ckpt_path.exists():
            ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found under {run_dir}/checkpoints")

    model = build_policy(run_cfg, normalizer).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()

    if policy_kind(run_cfg) == "bc":
        runner: PolicyRunner = BCRunner(model, device)
    else:
        m = run_cfg["MODEL"]
        runner = ACTRunner(model, device, int(m["history_len"]), int(m["chunk_len"]),
                           dict(run_cfg.get("EXEC", {"mode": "ensemble",
                                                     "ensemble_m": 0.01})))
    print(f"[policy] {ckpt_path} (epoch {payload.get('epoch', '?')})  {runner.describe()}")
    return runner, run_cfg


def export_run_dir(src_run_dir, dst_dir, ckpt: str = "best", note: str = "") -> None:
    """Publish a snapshot of `src_run_dir` as a standalone run dir at `dst_dir`.

    The selected checkpoint is written as `checkpoints/best.pt` so the snapshot
    loads correctly through *any* of the existing loaders (all of which try
    best.pt first), regardless of which checkpoint was selected.
    """
    src, dst = Path(src_run_dir), Path(dst_dir)
    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)

    src_ckpt = src / "checkpoints" / f"{ckpt}.pt"
    if not src_ckpt.exists():
        src_ckpt = src / "checkpoints" / "last.pt"
    shutil.copy2(src_ckpt, dst / "checkpoints" / "best.pt")

    for name in ("config.yaml", "normalization.npz", "log.csv"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    (dst / "source.txt").write_text(
        f"snapshot of: {src}\ncheckpoint : {src_ckpt.name}\n{note}\n"
    )
