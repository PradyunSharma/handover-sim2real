"""
Copy a run's SMALL metadata off /scratch and back into the repo, for git.

    python examples/harvest_run.py --run-dir /scratch/<netid>/handover-sim2real/output/dagger_runs/dagger4_run15
    python examples/harvest_run.py --all --scratch-root /scratch/<netid>/handover-sim2real
    python examples/harvest_run.py --run-dir <...> --dry-run

WHY. On DelftBlue /home is a hard 30 GB quota and /scratch is 5 TiB, so runs now
write to /scratch (see examples/slurm/train_dagger_phase4.sbatch). But /scratch is
PERIODICALLY PURGED BY AGE. Checkpoints are big and get synced to the PC
selectively; the run's metadata — the logs and configs — is the experiment record
and belongs in git. That is ~1 MB per run against the 1-2 GB left behind.

WHAT IT COPIES is a strict ALLOW-LIST (see PATTERNS), not "everything except".
An exclude-list would silently start copying whatever a future change adds to a
run directory, and the failure mode there is filling the very quota this exists
to protect. Checkpoints (*.pt) and replay data (*.h5) can therefore never be
picked up by accident, even if they are renamed or moved.

SAFE TO RUN WHILE THE JOB IS STILL GOING. Every file it reads is either
append-only (the CSVs) or written atomically via a .tmp rename (eval_log.csv,
state.json), so a partial read is not possible for the latter and is harmless for
the former — re-run it and you get the longer file. It only ever writes into the
repo, never into the run directory, so it cannot disturb a running job.

IDEMPOTENT: a file is copied only when size or mtime differs, so re-running is
cheap and prints nothing but a summary.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Allow-list, relative to the run directory. Globs are matched with Path.glob, so
# `iters/*/log.csv` picks up every iteration without naming them.
PATTERNS = [
    # -- the experiment record ------------------------------------------------
    "dagger_log.csv",        # the DAgger loop's per-iteration metrics
    "eval_log.csv",          # written by eval_dagger_run.py, when used
    "log.csv",               # RL runs, and BC runs used as a base
    "config.yaml",           # the resolved config — reproduces the run's setup
    "state.json",            # iteration bookkeeping; also what resume reads
    "grasp_registry.json",   # which grasp each scene was pinned to
    "source.txt",
    # -- per-iteration --------------------------------------------------------
    "iters/*/log.csv",
    "iters/*/config.yaml",
    "iters/*/source.txt",
    # -- plots ----------------------------------------------------------------
    "*.png",
    "iters/*/*.png",
]

# Anything matching these is refused even if a pattern above somehow selects it.
# Belt and braces: the allow-list already excludes them, and this makes a future
# careless addition to PATTERNS fail loudly rather than quietly fill /home.
DENY_SUFFIXES = {".pt", ".pth", ".h5", ".hdf5", ".npz", ".ckpt", ".tar"}


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def harvest(run_dir: Path, dest_root: Path, dry_run: bool = False,
            verbose: bool = True) -> tuple[int, int, int]:
    """Copy the allow-listed files of `run_dir` into `dest_root/<run name>`.

    Returns (files copied, bytes copied, bytes deliberately left behind).
    """
    run_dir = run_dir.resolve()
    dest = dest_root / run_dir.name
    copied = n_bytes = 0

    wanted: list[Path] = []
    for pat in PATTERNS:
        wanted.extend(sorted(run_dir.glob(pat)))

    seen: set[Path] = set()
    for src in wanted:
        if src in seen or not src.is_file():
            continue
        seen.add(src)
        if src.suffix.lower() in DENY_SUFFIXES:
            # Unreachable via the current PATTERNS; here so that it stays
            # unreachable if PATTERNS is ever widened carelessly.
            raise RuntimeError(
                f"refusing to harvest {src}: {src.suffix} is heavy data that "
                f"belongs on scratch. Fix PATTERNS rather than DENY_SUFFIXES.")
        rel = src.relative_to(run_dir)
        dst = dest / rel
        st = src.stat()
        if dst.exists():
            d = dst.stat()
            if d.st_size == st.st_size and int(d.st_mtime) == int(st.st_mtime):
                continue                      # unchanged — idempotent no-op
        if verbose:
            print(f"  {rel}  ({_human(st.st_size)})")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)            # copy2 preserves mtime, which the
                                              # idempotence check above relies on
        copied += 1
        n_bytes += st.st_size

    left = 0
    for p in run_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in DENY_SUFFIXES:
            left += p.stat().st_size
    return copied, n_bytes, left


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="a single run directory on /scratch")
    g.add_argument("--all", action="store_true",
                   help="harvest every run under --scratch-root")
    ap.add_argument("--scratch-root", default=None,
                    help="with --all: the root holding output/{dagger_runs,rl_runs}")
    ap.add_argument("--dest", default=None,
                    help="repo destination root (default: infer dagger_runs/rl_runs "
                         "from the source path, under ./output)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be copied, write nothing")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]

    runs: list[Path] = []
    if args.run_dir:
        runs = [Path(args.run_dir)]
    else:
        if not args.scratch_root:
            ap.error("--all requires --scratch-root")
        root = Path(args.scratch_root)
        for kind in ("dagger_runs", "rl_runs", "bc_runs"):
            d = root / "output" / kind
            if d.is_dir():
                runs.extend(sorted(p for p in d.iterdir() if p.is_dir()))

    if not runs:
        raise SystemExit("no run directories found")

    tot_files = tot_bytes = tot_left = 0
    for run in runs:
        if not run.is_dir():
            print(f"[skip] {run} is not a directory")
            continue
        if args.dest:
            dest_root = Path(args.dest)
        else:
            # Mirror the source's kind directory under the repo's output/, so a
            # run harvested from scratch lands where the same run would have
            # lived had it been written in-repo.
            dest_root = repo / "output" / run.parent.name
        print(f"\n{run}\n  -> {dest_root / run.name}")
        n, b, left = harvest(run, dest_root, dry_run=args.dry_run)
        tot_files += n
        tot_bytes += b
        tot_left += left
        if n == 0:
            print("  (already up to date)")

    verb = "would copy" if args.dry_run else "copied"
    print(f"\n{verb} {tot_files} files, {_human(tot_bytes)}; "
          f"{_human(tot_left)} of checkpoints/data left on scratch")
    if args.dry_run:
        print("(--dry-run: nothing was written)")


if __name__ == "__main__":
    main()
