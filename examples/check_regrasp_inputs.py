"""
Every input a Regrasp run config declares, and whether it is actually there.

    python examples/check_regrasp_inputs.py regrasp_run3 regrasp_run4 regrasp_run5
    REGRASP_DATA=/scratch/$USER/handover-sim2real/output \
        python examples/check_regrasp_inputs.py regrasp_run7 regrasp_run8

WHY THIS AND NOT `ls`. The paths are read OUT OF THE CONFIG, so this cannot
disagree with what the job will open — which `ls` of a remembered filename can,
and has. Three runs sharing a data set (3/4/5, or 2/7/8) differ only in training
flags, so the check is identical for all of them and the failure mode is
submitting one against another's tables.

It also PROBES rather than just stats: a pin table's `_meta.per_bin` is what
separates run 2's one-member table from run 3's three-member one, and both live
under names that differ by four characters.

`${REGRASP_DATA}` is expanded exactly as handover_sim2real/regrasp/setup.py
expands it — default `output`, and on DelftBlue set to $SCRATCH_ROOT/output by
the sbatch scripts. Set it here to match, or the HDF5 lines will read MISSING
against the in-repo path while the shards sit on scratch.

Read-only. It never writes, derives or submits anything.
"""

from __future__ import annotations

import json
import os
import sys

import yaml

DATA = os.environ.get("REGRASP_DATA", "output")

# The five paths that move together. Swapping one without the others silently
# trains on one collection and scores against another's captions.
KEYS = [("SIM", "grasp_pin_table"), ("SIM", "exclude_scenes"),
        ("SIM", "demo_ok_table"), ("TRAIN", "base_train_h5"),
        ("TRAIN", "val_h5")]


def probe(path: str) -> str:
    """One line saying what the file CONTAINS, not merely that it exists."""
    if not path.endswith(".json"):
        return f"{os.path.getsize(path) / 1e9:.2f} GB"
    try:
        d = json.load(open(path))
    except Exception as e:                       # noqa: BLE001 — report, don't raise
        return f"UNREADABLE: {e}"
    if isinstance(d, list):
        return f"{len(d)} entries"
    m = d.get("_meta", {})
    if "mode" in m:
        return (f"mode={m.get('mode')}  per_bin={m.get('per_bin') or 1}  "
                f"demos={m.get('n_demos')}")
    if "n_pairs" in m:
        return f"pairs={m['n_pairs']} over {m.get('n_scenes')} scenes"
    return f"{len(d)} keys"


def main() -> int:
    names = sys.argv[1:]
    if not names:
        print(__doc__)
        return 2

    missing = 0
    print(f"REGRASP_DATA={DATA}")
    for name in names:
        cfg_path = f"examples/configs/{name}.yaml"
        print(f"\n=== {name} ===")
        if not os.path.exists(cfg_path):
            print(f"  ** MISSING CONFIG **  {cfg_path}")
            missing += 1
            continue
        cfg = yaml.safe_load(open(cfg_path)) or {}

        for sec, key in KEYS:
            raw = (cfg.get(sec) or {}).get(key)
            if not raw:
                print(f"  --   {sec}.{key:<15}  (not set)")
                continue
            p = os.path.expandvars(raw.replace("${REGRASP_DATA}", DATA))
            if os.path.exists(p):
                print(f"  OK   {sec}.{key:<15}  {p}")
                print(f"       {probe(p)}")
            else:
                print(f"  ** MISSING **  {sec}.{key:<15}  {p}")
                missing += 1

        # The learner config is not a data path, but `d_noise_deg` is the one
        # value that distinguishes runs sharing everything else, and reading it
        # here is how you confirm you submitted the run you meant to.
        tc = (cfg.get("TRAIN") or {}).get("train_cfg")
        if tc and os.path.exists(tc):
            d = yaml.safe_load(open(tc)) or {}
            print(f"  OK   TRAIN.train_cfg      {tc}")
            print(f"       d_noise_deg={(d.get('DATA') or {}).get('d_noise_deg')}")
        elif tc:
            print(f"  ** MISSING **  TRAIN.train_cfg      {tc}")
            missing += 1

        trn = cfg.get("TRAIN") or {}
        dag = cfg.get("DAGGER") or {}
        print(f"       -> beta {dag.get('beta_start')} -> {dag.get('beta_end')}, "
              f"{dag.get('num_iters')} iters, m={dag.get('episodes_per_iter')}, "
              f"{'scratch (FTL)' if trn.get('train_from_scratch', True) else 'warm start'}")

    print(f"\n{missing} missing input(s)." if missing else "\nAll inputs present.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
