"""Inventory the Phase-4 datasets and the files that support them.

    python examples/check_datasets.py

Answers "is everything collected, filtered and paired up" for both pinning
tracks. It opens every HDF5 and RECOUNTS the failed episodes rather than
trusting the filename, so a file that exists but was never actually filtered
shows up as UNFILTERED instead of passing silently. It also prints each pin
table's mode and split, which is how you confirm an `omg` table did not get
written over a `furthest_from_hand` one at the path SIM.grasp_pin_table uses.

A "failed" episode is one with no CLOSE label — see filter_demos.py for why that
is exactly equivalent to the benchmark having killed it.
"""

import glob, json, os
import h5py

def h5info(p):
    with h5py.File(p, "r") as f:
        eps = [k for k in f if k.startswith("episode")]
        steps = sum(int(f[k].attrs["num_steps"]) for k in eps)
        bad = sum(1 for k in eps if not (f[k]["expert_actions"][:, 6] < 0.5).any())
        pin = f.attrs.get("grasp_pin_table", "")
    return len(eps), steps, bad, pin

print("=" * 78)
print("PIN TABLES")
print("=" * 78)
for p in sorted(glob.glob("output/grasp_pin_table*.json")):
    d = json.load(open(p))
    m = d.get("_meta", {})
    ok = sum(1 for k, v in d.items() if k != "_meta" and isinstance(v, dict))
    null = sum(1 for k, v in d.items() if k != "_meta" and v is None)
    print(f"  {p}")
    print(f"      mode={m.get('mode'):<20s} split={m.get('split'):<6s} "
          f"pinned={ok:<5d} unplannable={null}")

print()
print("=" * 78)
print("DATASETS       episodes / steps / failed(no CLOSE) / pin table recorded")
print("=" * 78)
for p in sorted(glob.glob("output/bc_dataset/*pinned*.h5")):
    try:
        n, s, bad, pin = h5info(p)
        flag = "  <-- UNFILTERED" if bad else ""
        print(f"  {p:<50s} {n:5d} ep {s:7d} st  {bad:4d} bad{flag}")
        if pin:
            print(f"      built against: {pin}")
    except Exception as e:
        print(f"  {p:<50s} UNREADABLE: {e}")

print()
print("=" * 78)
print("EXCLUDE-SCENE JSONS")
print("=" * 78)
js = sorted(glob.glob("output/bc_dataset/*_ok.json"))
for p in js:
    try:
        print(f"  {p:<50s} {len(json.load(open(p))):4d} scenes excluded")
    except Exception as e:
        print(f"  {p:<50s} UNREADABLE: {e}")
if not js:
    print("  (none)")

print()
print("=" * 78)
print("COMPLETENESS  (each track needs 4 files: train h5+json, val h5+json)")
print("=" * 78)
tracks = {
    "furthest-from-hand": ("train_pinned", "val_pinned"),
    "omg":                ("train_pinned_omg", "val_pinned_omg"),
}
for name, (tr, va) in tracks.items():
    print(f"\n  {name}")
    need = [(f"output/bc_dataset/{tr}_ok.h5",   "train dataset (filtered)"),
            (f"output/bc_dataset/{tr}_ok.json", "train excluded scenes"),
            (f"output/bc_dataset/{va}_ok.h5",   "val dataset (filtered)"),
            (f"output/bc_dataset/{va}_ok.json", "val excluded scenes")]
    for path, what in need:
        print(f"    [{'x' if os.path.exists(path) else ' '}] {what:<26s} {path}")
