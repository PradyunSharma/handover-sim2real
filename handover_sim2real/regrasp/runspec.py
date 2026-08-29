"""Resolve a Regrasp run NAME into every path and rule its data was made with.

A DAgger shard is not self-describing. To replay one correctly you need the
benchmark config it was collected under, the grasp pin table whose SLOT NUMBERING
its `grasp_idx` attribute indexes, and the direction rule its `d_world` was
labelled with — and getting any of them wrong is silent, not fatal. Hand a
`grasp_offset` shard the `approach_axis` table and `--show-d` draws a vector a
median 14.5 deg off, pointing into a different bin, with no warning.

All four facts are already recorded: `train_regrasp.py` writes the resolved
config to `<run_dir>/config.yaml` when the run starts. This module reads it back,
so the caller supplies a run name and an iteration number and nothing else:

    spec = resolve_run("regrasp_run12")
    spec.dataset_for(13)     -> .../regrasp_run12/data/dagger_iter_13.h5
    spec.cfg_file            -> examples/pretrain_multicam_wr.yaml
    spec.pin_table           -> output/regrasp_pins_train_off.json
    spec.d_rule              -> grasp_offset

THE PATHS INSIDE THAT SNAPSHOT ARE THE CLUSTER'S. `TRAIN.base_train_h5` is
absolute under `/home/pradyunsharma/h2r-runs`, which does not exist on a laptop
that rsync'd the run down. So every path is re-rooted through `data_roots()`
before it is handed back, by matching on the tail after the last `output/`
component. A path that cannot be found anywhere comes back None rather than as a
string that fails several frames deep inside h5py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# `train_regrasp.py`'s own default, and the layout every runbook uses.
_RUNS_SUBDIR = os.path.join("output", "dagger_runs")


def _repo_root() -> Path:
    """The checkout this module lives in — handover_sim2real/regrasp/../.."""
    return Path(__file__).resolve().parents[2]


def run_roots(extra: str | os.PathLike | None = None) -> list[Path]:
    """Where `<run_name>/` directories are looked for, most specific first.

    `$REGRASP_DATA` is what the sbatch exports and already ends in `output`;
    `$RUNS` is the runbooks' scratch root and does not. Both are checked so the
    same command works on the cluster and on a laptop that synced a run down.
    """
    out: list[Path] = []
    if extra:
        out.append(Path(extra))
    if os.environ.get("REGRASP_DATA"):
        out.append(Path(os.environ["REGRASP_DATA"]) / "dagger_runs")
    if os.environ.get("RUNS"):
        out.append(Path(os.environ["RUNS"]) / _RUNS_SUBDIR)
    out.append(Path.cwd() / _RUNS_SUBDIR)
    out.append(_repo_root() / _RUNS_SUBDIR)
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def data_roots(extra: str | os.PathLike | None = None) -> list[Path]:
    """Where an `output/`-relative artifact (pin table, base shard) is looked for."""
    out: list[Path] = []
    if extra:
        out.append(Path(extra))
    if os.environ.get("REGRASP_DATA"):
        out.append(Path(os.environ["REGRASP_DATA"]))
    if os.environ.get("RUNS"):
        out.append(Path(os.environ["RUNS"]) / "output")
    out.append(Path.cwd() / "output")
    out.append(_repo_root() / "output")
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def relocate_all(paths: Any, extra_root: str | os.PathLike | None = None
                 ) -> list[str]:
    """`relocate` over a scalar or a list, dropping what cannot be found.

    `TRAIN.base_train_h5` is a LIST whenever the base set was collected in
    shards (`--shard i/4`), which regrasp3_fast1 and the fast runs do.
    """
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = []
    for p in paths:
        r = relocate(str(p), extra_root)
        if r is not None:
            out.append(r)
    return out


def relocate(path: str | None, extra_root: str | os.PathLike | None = None
             ) -> str | None:
    """Find `path` here, wherever the config says it lived when the run ran.

    Tries it verbatim first — on the cluster that always wins. Otherwise splits
    on the LAST `output` component and re-joins the tail under each data root, so
    `/home/pradyunsharma/h2r-runs/output/bc_dataset/train_regrasp_off.h5` finds
    `./output/bc_dataset/train_regrasp_off.h5`. Returns None if nothing matches.
    """
    if not path:
        return None
    if os.path.exists(path):
        return str(path)
    parts = Path(path).parts
    if "output" in parts:
        i = len(parts) - 1 - parts[::-1].index("output")
        tail = Path(*parts[i + 1:])
    else:
        tail = Path(path).name
    for root in data_roots(extra_root):
        cand = root / tail
        if cand.exists():
            return str(cand)
    return None


@dataclass
class RunSpec:
    """Everything a viewer or a rollout needs, derived from a run name alone."""

    run: str
    run_dir: Path
    cfg: dict[str, Any] = field(repr=False)
    cfg_source: str                 # which yaml the SIM block came from
    cfg_file: str | None            # SIM.cfg_file — the benchmark config
    pin_table: str | None           # SIM.grasp_pin_table, re-rooted
    d_rule: str                     # SIM.d_rule, defaulting as the code does
    d_point_depth: float | None
    d_min_offset: float | None
    command_deploy: str             # SIM.command_deploy — how `d` is built at test
    valid_grasp_dict: str | None    # SIM.valid_grasp_dict_path, re-rooted
    base_h5: list[str]              # TRAIN.base_train_h5, re-rooted (may be sharded)

    # ---- checkpoints -------------------------------------------------------

    def ckpt_dir(self, which: str | int = "best") -> Path:
        """A directory `rollout_regrasp_policy.py` accepts as --run-dir.

        "best" / "last" are the run-level selections; an INTEGER (or a digit
        string, or "iter_13") is that iteration's own checkpoint under
        `<run_dir>/iters/iter_NN`, which is how you watch the policy as it was
        partway through rather than at its best.

        best/last fall back to each other, because a run killed mid-iteration
        has a `last` and no `best`.
        """
        key = str(which).strip().lower()
        if key.startswith("iter_"):
            key = key[5:]
        if key.isdigit():
            p = self.run_dir / "iters" / f"iter_{int(key):02d}"
            if p.is_dir():
                return p
            have = sorted(q.name for q in (self.run_dir / "iters").glob("iter_*")
                          ) if (self.run_dir / "iters").is_dir() else []
            raise SystemExit(
                f"{self.run}: no checkpoint for iteration {key}.\n"
                f"  looked in {self.run_dir / 'iters'}\n"
                f"  present: {', '.join(have) or 'none'}")

        order = [key] + [o for o in ("best", "last") if o != key]
        for name in order:
            p = self.run_dir / name
            if p.is_dir():
                return p
        raise SystemExit(f"{self.run}: neither {self.run_dir / 'best'} nor "
                         f"{self.run_dir / 'last'} exists — has it trained yet?")

    # ---- datasets ----------------------------------------------------------

    def iter_shards(self) -> dict[int, str]:
        """{iteration -> shard path} for every `dagger_iter_NN.h5` on disk."""
        out: dict[int, str] = {}
        for p in sorted((self.run_dir / "data").glob("dagger_iter_*.h5")):
            stem = p.stem.rsplit("_", 1)[-1]
            if stem.isdigit():
                out[int(stem)] = str(p)
        return out

    def dataset_for(self, iteration: int | str) -> str:
        """Resolve an iteration to a shard. 0 / "base" is the base demo set."""
        shards = self.iter_shards()
        if isinstance(iteration, str):
            key = iteration.strip().lower()
            if key in ("base", "0"):
                iteration = 0
            elif key == "last":
                if not shards:
                    raise SystemExit(f"{self.run}: no DAgger shards under "
                                     f"{self.run_dir / 'data'}")
                iteration = max(shards)
            else:
                try:
                    iteration = int(key)
                except ValueError:
                    raise SystemExit(
                        f"--iter must be an integer, 'base' or 'last', got "
                        f"{iteration!r}") from None

        if int(iteration) == 0:
            if not self.base_h5:
                raise SystemExit(
                    f"{self.run}: iteration 0 is the base demo set "
                    f"(`TRAIN.base_train_h5` = "
                    f"{self.cfg.get('TRAIN', {}).get('base_train_h5')!r}), which "
                    f"is not on this machine. Searched:\n  "
                    + "\n  ".join(str(r) for r in data_roots()))
            if len(self.base_h5) > 1:
                print(f"[runspec] base set is {len(self.base_h5)} shards; opening "
                      f"the first. Others: "
                      + ", ".join(os.path.basename(p) for p in self.base_h5[1:]))
            return self.base_h5[0]

        path = shards.get(int(iteration))
        if path is None:
            have = ", ".join(str(i) for i in sorted(shards)) or "none"
            raise SystemExit(
                f"{self.run}: no shard for iteration {iteration}.\n"
                f"  looked in {self.run_dir / 'data'}\n"
                f"  iterations present: {have}   (0 = the base demo set)")
        return path

    # ---- reporting ---------------------------------------------------------

    def describe(self, iteration: int | str | None = None) -> str:
        lines = [f"[run] {self.run}  ({self.run_dir})",
                 f"      config      {self.cfg_source}"]
        if iteration is not None:
            lines.append(f"      dataset     {self.dataset_for(iteration)}")
        lines += [f"      cfg_file    {self.cfg_file}",
                  f"      pin table   {self.pin_table}",
                  f"      d_rule      {self.d_rule}"
                  + (f"  (depth {self.d_point_depth}, min_offset "
                     f"{self.d_min_offset})" if self.d_rule == "grasp_offset"
                     else ""),
                  f"      command     {self.command_deploy}"]
        return "\n".join(lines)


def resolve_run(run: str, *, run_root: str | os.PathLike | None = None) -> RunSpec:
    """Find a run by name (or by path) and read back how its data was made.

    `<run_dir>/config.yaml` is the snapshot `train_regrasp.py` writes at start,
    and is preferred over `examples/configs/<run>.yaml` BECAUSE THE REPO COPY CAN
    HAVE MOVED SINCE. Run 13's config was edited after run 12 finished; reading
    the repo copy to interpret an old shard would describe a run that never
    happened. Runs 7 and 8 predate the snapshot, so the repo config is the
    fallback and the source is always printed.
    """
    cand = Path(run)
    run_dir = None
    if cand.is_dir() and ((cand / "data").exists()
                          or (cand / "config.yaml").exists()):
        run_dir = cand.resolve()
    else:
        for root in run_roots(run_root):
            p = root / run
            if p.is_dir():
                run_dir = p.resolve()
                break
    if run_dir is None:
        searched = "\n  ".join(str(r) for r in run_roots(run_root))
        raise SystemExit(f"run directory not found for {run!r}. Searched:\n  "
                         f"{searched}\n"
                         f"Set $REGRASP_DATA or $RUNS, or pass --run-root.")

    name = run_dir.name
    snap = run_dir / "config.yaml"
    repo_cfg = _repo_root() / "examples" / "configs" / f"{name}.yaml"
    if snap.exists():
        cfg_path = snap
    elif repo_cfg.exists():
        cfg_path = repo_cfg
    else:
        raise SystemExit(
            f"{name}: no config to derive from — neither {snap} nor {repo_cfg} "
            f"exists. Pass --cfg-file / --grasp-pin-table by hand.")

    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    sim = cfg.get("SIM", {}) or {}
    train = cfg.get("TRAIN", {}) or {}

    pin = relocate(sim.get("grasp_pin_table"))
    if sim.get("grasp_pin_table") and pin is None:
        print(f"[runspec] WARNING: pin table {sim['grasp_pin_table']!r} named by "
              f"{cfg_path} is not on this machine; overlays will fall back to "
              f"the default rule.")

    # `d_rule` absent means the run predates the key, which resolves to
    # approach_axis in DirectionRule.from_cfg — mirror that here so what is
    # PRINTED matches what is drawn.
    return RunSpec(
        run=name,
        run_dir=run_dir,
        cfg=cfg,
        cfg_source=str(cfg_path) + ("" if cfg_path == snap
                                    else "  (repo config — this run has no snapshot)"),
        cfg_file=sim.get("cfg_file"),
        pin_table=pin,
        d_rule=sim.get("d_rule", "approach_axis"),
        d_point_depth=sim.get("d_point_depth"),
        d_min_offset=sim.get("d_min_offset"),
        # `bin_axis` is the collector's own default and is what runs 2-8 used;
        # runs 9-13 deploy on the bin CENTROID. A rollout that takes the wrong
        # one is watching a different experiment than the one that was scored.
        command_deploy=sim.get("command_deploy", "bin_axis"),
        valid_grasp_dict=relocate(sim.get("valid_grasp_dict_path")),
        base_h5=relocate_all(train.get("base_train_h5")),
    )
