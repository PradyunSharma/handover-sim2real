# regrasp_run19 — direction-conditioned regrasp policy (the one to deploy)

Driven by `my_regrasp_policy_runner.py --regrasp-run 19`, **not**
`my_policy_runner.py`. 7-channel cloud, refuses to start without a
`--direction`. `checkpoint/run12`, `run16` and `run19` are all Phase-4,
`pc_channels: 5`, and have no direction input — note that `checkpoint/run19` and
`checkpoint/regrasp_run19` are different policies for different runners, which
is exactly why the regrasp ones carry the prefix.

## Why this one

Run 19 is run 9 with two changes: DART collision **shielding**
(`shield: true`, 1 cm clearance, 4 path steps, 5 jolt tries) and 20 training
epochs an iteration instead of 15. Same `d_rule: approach_axis`, same pin table,
same `pretrain_multicam_wr.yaml` cameras, same four live bins. It is better on
every axis that matters:

| | run 9 (iter 23) | run 11 (iter 22) | **run 19 (iter 25)** |
|---|---|---|---|
| success_rate | 0.5924 | 0.6186 | **0.7426** |
| close_rate | 0.6849 | 0.7784 | **0.8663** |
| bin_hit_rate | 0.6513 | 0.1412 | **0.6535** |
| cond_sep | 0.8899 | 0.3107 | 0.8201 |
| miss_given_box | 0.100 | 0.081 | **0.054** |
| dir_err median | 10.00 | — | 13.06 |

Run 11 grasped more often than run 9 by ignoring the command; run 19 grasps more
often *while still following it*. `cond_sep` 0.82 and `bin_hit` 0.65 say the
conditioning is being read, and `close_rate` 0.87 against run 9's 0.68 means it
fails to commit to a grasp in 13% of sim episodes rather than 31% — which is
directly the "sometimes the gripper doesn't close" symptom seen on hardware.

**The per-bin rates are also even**, which run 9's are not:

| bin | +x | +y | -y | +z |
|---|---|---|---|---|
| run 19 | 0.741 | 0.776 | 0.703 | 0.743 |
| run 9 | 0.649 | 0.630 | **0.434** | 0.632 |
| n (run 19) | 81 | 49 | 37 | 35 |

So on run 19 the direction you ask for stops being the thing that decides
whether it works. `-x` and `-z` remain empty in the training assignment and the
runner refuses them.

## Layout

```
command_axes.json      the six DEPLOYMENT directions, in the anchor frame
config.yaml            the run's own DAgger config (d_rule: approach_axis)
dagger_log.csv         per-iteration metrics, summarised below
best/                  iteration 25 — the exported best
last/                  iteration 25 — the same one; this run peaked at the end
iter_00/ ... iter_25/  every iteration, each a loadable run dir
```

Each holds `config.yaml`, `normalization.npz` and `checkpoints/{best,last}.pt`,
which is what `load_policy_runner` needs. `command_axes.json` sits at THIS level
on purpose — the runner looks for it beside the run dir it was given, so it is
found whether you point at `best/`, `last/` or an `iter_NN/`.

**The `.pt` files are hard links** into `output/dagger_runs/regrasp_run19/iters/`,
so installing all 26 iterations cost essentially nothing rather than 1.2 GB.
They read exactly like copies and deleting either side leaves the other intact;
re-copy with `cp -a` instead of `cp -al` if you want independent files.

## Still shared with runs 9 and 11, and not fixable from the deployment side

Collected under `examples/pretrain_multicam_wr.yaml` — **wrist + right
cameras** — so it expects an eye-in-hand view; deploying off a fixed camera
alone removes the only view that survives the endgame. Point encoder trained
from scratch (`pc_pretrained: null`) where the deployable Phase-4 run 19
warm-started all 68/68 tensors from the CVPR2023 encoder. And `d_noise_deg: 0.0`,
so it was never shown a direction command that was merely approximately right.

## Iterations

`success_rate` is the in-loop eval on the TRAIN split (`EVAL.holdout: false`), so
these rank the iterations rather than measuring them.

| iter | success | close | dir_err | bin_hit | cond_sep | |
|---|---|---|---|---|---|---|
| 0 | 0.3614 | 0.446 | 49.32 | 0.35 | 0.66 | **best so far** |
| 1 | 0.3416 | 0.416 | 46.97 | 0.35 | 0.87 |  |
| 2 | 0.4653 | 0.569 | 47.88 | 0.41 | 0.70 | **best so far** |
| 3 | 0.5495 | 0.668 | 45.70 | 0.37 | 0.66 | **best so far** |
| 4 | 0.5396 | 0.599 | 43.22 | 0.35 | 0.76 |  |
| 5 | 0.5000 | 0.584 | 40.51 | 0.35 | 0.73 |  |
| 6 | 0.4901 | 0.614 | 46.82 | 0.37 | 0.60 |  |
| 7 | 0.5198 | 0.648 | 40.17 | 0.38 | 0.74 |  |
| 8 | 0.5644 | 0.703 | 40.64 | 0.42 | 0.70 | **best so far** |
| 9 | 0.5396 | 0.673 | 32.92 | 0.52 | 0.83 |  |
| 10 | 0.5594 | 0.644 | 36.47 | 0.44 | 0.77 |  |
| 11 | 0.5050 | 0.634 | 38.33 | 0.40 | 0.77 |  |
| 12 | 0.5891 | 0.743 | 26.49 | 0.48 | 0.81 | **best so far** |
| 13 | 0.6188 | 0.782 | 25.12 | 0.51 | 0.82 | **best so far** |
| 14 | 0.6238 | 0.757 | 27.86 | 0.48 | 0.79 | **best so far** |
| 15 | 0.5941 | 0.733 | 24.11 | 0.51 | 0.78 |  |
| 16 | 0.6436 | 0.812 | 20.04 | 0.57 | 0.83 | **best so far** |
| 17 | 0.5990 | 0.713 | 31.98 | 0.41 | 0.75 |  |
| 18 | 0.6634 | 0.817 | 18.65 | 0.58 | 0.79 | **best so far** |
| 19 | 0.7178 | 0.832 | 20.10 | 0.60 | 0.82 | **best so far** |
| 20 | 0.6931 | 0.817 | 17.58 | 0.58 | 0.83 |  |
| 21 | 0.6683 | 0.792 | 18.74 | 0.59 | 0.81 |  |
| 22 | 0.6980 | 0.842 | 19.64 | 0.62 | 0.82 |  |
| 23 | 0.6782 | 0.852 | 16.58 | 0.62 | 0.85 |  |
| 24 | 0.6188 | 0.797 | 17.83 | 0.59 | 0.77 |  |
| 25 | 0.7426 | 0.866 | 16.12 | 0.65 | 0.82 | **best so far** |

Iteration **25** is what `best/` exports, at 0.7426, and it is also the last —
unusually, this run peaked where it ended. 19 (0.7178) and 22 (0.6980) are the
next strongest. Success is still not monotonic: 24 drops to 0.6188 immediately
before the peak, so picking a late iteration is not automatically picking a good
one.
