"""Validate a DAgger run dir after a crash, BEFORE resubmitting.

    python examples/check_run_integrity.py output/dagger_runs/dagger4_run2

WHY THIS IS NEEDED. train_dagger_phase4.py reuses an existing
data/dagger_iter_NN.h5 on the strength of `exists()` alone. DaggerHDF5Writer.append
flushes `num_episodes` after EVERY episode, so a half-written collection is
perfectly self-consistent — it just has fewer episodes than it should. Nothing
downstream notices, and that iteration silently trains on a smaller D_i.

THE DECISIVE SIGNAL is state.json, which train_dagger_phase4.py writes only after
an iteration's collection AND refit AND eval have all finished (line 864). So:

    max iteration recorded in state.json = N   =>   0..N are complete
    any data/dagger_iter_M.h5 or iters/iter_M with M > N is from the interrupted
    iteration and must go

Episode counts BELOW episodes_per_iter are normal and not evidence of damage —
scenes where OMG cannot plan are skipped and never produce an episode (run 1
sampled 20 per iteration and saved 15-20).

Checkpoints and normalization.npz are checked separately: those fail loudly on
load rather than silently, but a full disk can truncate them too.
"""

import json
import sys
import zipfile
from pathlib import Path

import h5py


def h5_episodes(p: Path):
    try:
        with h5py.File(p, "r") as f:
            groups = sum(1 for k in f if k.startswith("episode"))
            attr = int(f.attrs.get("num_episodes", -1))
            asked = len(f.attrs["scenes"]) if "scenes" in f.attrs else None
        return groups, attr, asked, None
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"


def zip_ok(p: Path):
    """.pt files are zip archives; testzip() catches truncation without a full load."""
    try:
        return zipfile.ZipFile(p).testzip() is None
    except Exception:
        return False


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = Path(sys.argv[1])
    if not run.is_dir():
        raise SystemExit(f"not a run dir: {run}")

    print(f"\n== {run} ==")

    # ---- what the loop believes is finished --------------------------------
    done, last_done = [], -1
    st_p = run / "state.json"
    if st_p.exists():
        try:
            st = json.load(st_p.open())
            done = sorted(r["iter"] for r in st.get("iterations", []))
            last_done = max(done) if done else -1
            b = st.get("best", {})
            print(f"  state.json: complete through iter {last_done}   {done}")
            if b:
                print(f"              best = iter {b.get('iter')}, "
                      f"{b.get('metric')}={b.get('score')}")
        except Exception as e:
            print(f"  state.json: CORRUPT ({type(e).__name__}) — the run would "
                  f"restart from iteration 0")
            print(f"\n  rm -r {run}      # nothing recoverable without it")
            return
    else:
        print("  state.json: absent — fresh run, nothing to clean")
        return

    stale, broken = [], []

    # ---- collections --------------------------------------------------------
    print("\n  collections")
    data = sorted((run / "data").glob("dagger_iter_*.h5")) if (run / "data").is_dir() else []
    for p in data:
        i = int(p.stem.split("_")[-1])
        groups, attr, asked, err = h5_episodes(p)
        if err:
            print(f"    [BROKEN ] {p.name:24s} {err}")
            broken.append(p)
            continue
        frac = f"{groups}/{asked} scenes" if asked else f"{groups} episodes"
        if i > last_done:
            print(f"    [STALE  ] {p.name:24s} {frac}  <- iteration never completed")
            stale.append(p)
        elif attr >= 0 and groups != attr:
            print(f"    [BROKEN ] {p.name:24s} {groups} groups vs num_episodes={attr}")
            broken.append(p)
        else:
            print(f"    [ok     ] {p.name:24s} {frac}")
    if not data:
        print("    (none)")

    # ---- iteration dirs -----------------------------------------------------
    print("\n  iteration dirs")
    iters = sorted((run / "iters").glob("iter_*")) if (run / "iters").is_dir() else []
    for d in iters:
        i = int(d.name.split("_")[-1])
        bits, ok = [], True
        for name in ("last.pt", "best.pt"):
            p = d / "checkpoints" / name
            if not p.exists():
                bits.append(f"{name}=absent")
            elif zip_ok(p):
                bits.append(f"{name}=ok")
            else:
                bits.append(f"{name}=CORRUPT")
                ok = False
        nz = d / "normalization.npz"
        if not nz.exists() or nz.stat().st_size == 0:
            bits.append("norm=MISSING")
            ok = False
        log = d / "log.csv"
        bits.append(f"{sum(1 for _ in log.open()) - 1 if log.exists() else 0} epochs")

        if not ok:
            print(f"    [BROKEN ] {d.name:10s} {'  '.join(bits)}")
            broken.append(d)
        elif i > last_done:
            print(f"    [STALE  ] {d.name:10s} {'  '.join(bits)}  <- never completed")
            stale.append(d)
        else:
            print(f"    [ok     ] {d.name:10s} {'  '.join(bits)}")
    if not iters:
        print("    (none)")

    # ---- verdict ------------------------------------------------------------
    print()
    if not stale and not broken:
        print(f"  Clean. Resubmitting will resume at iteration {last_done + 1}.")
        return
    print("  Delete these, then resubmit — the loop will redo them from scratch:\n")
    for p in stale + broken:
        print(f"    rm {'-r ' if p.is_dir() else ''}{p}")
    print(f"\n  All are for iterations > {last_done}, which state.json does NOT record "
          f"as done,\n  so nothing recorded is lost. Resume picks up at "
          f"iteration {last_done + 1}.")


if __name__ == "__main__":
    main()
