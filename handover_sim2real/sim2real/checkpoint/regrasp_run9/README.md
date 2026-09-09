# regrasp_run9 — direction-conditioned regrasp policy

Driven by `my_regrasp_policy_runner.py`, **not** `my_policy_runner.py`. The two
policies are not interchangeable: this one's cloud is `[1024, 7]` (the extra two
channels carry the commanded grasp direction) and it refuses to start without a
`--direction`. `checkpoint/run12`, `run16` and `run19` are all Phase-4,
`pc_channels: 5`, and have no direction input.

## Layout

```
command_axes.json      the six DEPLOYMENT directions, in the anchor frame
config.yaml            the run's own DAgger config (SIM.command_deploy: bin_centroid)
dagger_log.csv         per-iteration metrics, summarised below
best/                  iteration 23 — the exported best
last/                  iteration 25 — the final iteration
iter_00/ ... iter_25/  every iteration, each a loadable run dir
```

Each of those directories holds `config.yaml`, `normalization.npz` and
`checkpoints/{best,last}.pt`, which is exactly what `load_policy_runner`
needs. `command_axes.json` sits at THIS level on purpose — the runner looks for
it beside the run dir it was given, so it is found whether you point at
`best/`, `last/` or an `iter_NN/`.

**The `.pt` files are hard links** into `output/dagger_runs/regrasp_run9/iters/`,
so the 1.2 GB is not duplicated on disk. They read exactly like copies; deleting
either side leaves the other intact. Re-copy with `cp -a` instead of `cp -al` if
you want independent files.

## Iterations

`success_rate` is the in-loop eval on the TRAIN split (`EVAL.holdout: false`), so
these are optimistic — treat them as a ranking, not an absolute. `dir_err` is
how far, in degrees, the achieved approach ended up from the commanded one.

| iter | success | dir_err | |
|---|---|---|---|
| 0 | 0.2353 | 47.25 | **best so far** |
| 1 | 0.2059 | 46.04 |  |
| 2 | 0.2353 | 45.64 | **best so far** |
| 3 | 0.2017 | 51.65 |  |
| 4 | 0.2311 | 40.30 |  |
| 5 | 0.3950 | 20.98 | **best so far** |
| 6 | 0.4454 | 19.92 | **best so far** |
| 7 | 0.4580 | 20.40 | **best so far** |
| 8 | 0.4664 | 18.48 | **best so far** |
| 9 | 0.4412 | 19.17 |  |
| 10 | 0.4622 | 19.20 |  |
| 11 | 0.4664 | 18.21 |  |
| 12 | 0.4874 | 18.22 | **best so far** |
| 13 | 0.5084 | 16.44 | **best so far** |
| 14 | 0.5630 | 15.77 | **best so far** |
| 15 | 0.4790 | 15.35 |  |
| 16 | 0.3908 | 17.28 |  |
| 17 | 0.4832 | 16.10 |  |
| 18 | 0.5462 | 13.93 |  |
| 19 | 0.4370 | 16.82 |  |
| 20 | 0.5000 | 14.27 |  |
| 21 | 0.4958 | 15.75 |  |
| 22 | 0.5210 | 14.62 |  |
| 23 | 0.5924 | 14.68 | **best so far** |
| 24 | 0.5420 | 16.09 |  |
| 25 | 0.4958 | 15.29 |  |

Iteration **23** is what `best/` exports, at 0.5924. Note 14 (0.5630) and 18
(0.5462) are the other strong ones, and that success is not monotonic — 16 dips
to 0.3908 — so picking a late iteration is not automatically picking a good one.
