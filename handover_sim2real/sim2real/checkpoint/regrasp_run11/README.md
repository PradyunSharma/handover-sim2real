# regrasp_run11 — direction-conditioned regrasp policy (grasp_offset)

Driven by `my_regrasp_policy_runner.py --regrasp-run 11`, **not**
`my_policy_runner.py`. 7-channel cloud, refuses to start without a
`--direction`. `checkpoint/run12`, `run16` and `run19` are all Phase-4,
`pc_channels: 5`, and have no direction input.

## How this differs from run 9, and why it is not a tuning difference

`SIM.d_rule` is **`grasp_offset`**, where run 9's is `approach_axis`. The command
is a different question:

* run 9 — `d` is the gripper's approach axis: *come at the object from this side.*
* run 11 — `d` is the grasp point's offset from the object centroid, measured at
  `d_point_depth: 0.1122` with `d_min_offset: 0.02`: *grasp this part of the
  object.*

So `--direction +x` does not mean the same thing in the two runs, and their
`command_axes.json` files are not interchangeable. Run 11's centroids also tilt
slightly **downward** (`+x` has z = -0.20) where run 9's tilt up, and its `+y`
and `-y` are nearly antiparallel where run 9's meet at 150.5 deg.

Run 11 was also collected with a **reach filter** (`reach_filter: true`, 2 cm /
0.34 rad), 600 episodes per iteration against run 9's 400, and a different pin
table (`regrasp_pins_train_off.json`).

**All six bins are live here.** Run 9's `-x` and `-z` are empty and the runner
refuses them; run 11 demonstrates all six, so `-x` and `-z` are commandable —
though `-x` is its weakest at 0.333.

## The trade against run 9, stated plainly

At their best iterations, run 11 grasps more often and follows the command less:

| | run 9 (iter 23) | run 11 (iter 22) |
|---|---|---|
| success_rate | 0.5924 | **0.6186** |
| close_rate | 0.6849 | **0.7784** |
| bin_hit_rate | **0.6513** | 0.1412 |
| cond_sep | **0.8899** | 0.3107 |

`cond_sep` is the diagnostic the evaluator was built around: it measures how far
apart the same scene ends up under different commands, and a low value is
"the policy ignores the conditioning and regresses the mean of its
demonstrations". 0.31 is close to that failure. `dir_err` is **not** comparable
across the two runs — the rules measure different quantities.

The flip side is worth testing on hardware rather than assuming: a policy that
leans less on the conditioning channels leans less on exactly the inputs a real
camera degrades most, so run 11 may transfer better than its sim gap suggests.

## Per-bin success at iteration 22

| bin | +x | -x | +y | -y | +z | -z |
|---|---|---|---|---|---|---|
| success | 0.778 | 0.333 | 0.649 | 0.600 | 0.737 | 0.538 |
| n | 54 | 33 | 37 | 25 | 19 | 26 |

Note `-y` is 0.600 here against run 9's 0.434 — if `-y` is the direction you
need, this run is the better starting point on that axis alone.

## Layout

```
command_axes.json      the six DEPLOYMENT directions, in the anchor frame
config.yaml            the run's own DAgger config (SIM.d_rule: grasp_offset)
dagger_log.csv         per-iteration metrics, summarised below
best/                  iteration 22 — the exported best
last/                  iteration 25 — the final iteration
iter_00/ ... iter_25/  every iteration, each a loadable run dir
```

Each of those directories holds `config.yaml`, `normalization.npz` and
`checkpoints/{best,last}.pt`, which is exactly what `load_policy_runner` needs.
`command_axes.json` sits at THIS level on purpose — the runner looks for it
beside the run dir it was given, so it is found whether you point at `best/`,
`last/` or an `iter_NN/`.

**The `.pt` files are hard links** into `output/dagger_runs/regrasp_run11/iters/`,
so the 1.2 GB is not duplicated on disk — installing this cost essentially
nothing. They read exactly like copies; deleting either side leaves the other
intact. Re-copy with `cp -a` instead of `cp -al` if you want independent files.

## Shared with run 9, and unfixable from here

Both runs were collected under `examples/pretrain_multicam_wr.yaml` — **wrist +
right cameras** — so both expect an eye-in-hand view. Deploying off a fixed
camera alone removes the only view that survives the endgame. Both also trained
the point encoder from scratch (`pc_pretrained: null`), where the deployable
Phase-4 run 19 warm-started all 68/68 tensors from the CVPR2023 encoder. And
both used `d_noise_deg: 0.0`, so neither was ever shown a direction command that
was merely approximately right.

## Iterations

`success_rate` is the in-loop eval on the TRAIN split (`EVAL.holdout: false`), so
these are optimistic — treat them as a ranking, not an absolute.

| iter | success | dir_err | bin_hit | cond_sep | |
|---|---|---|---|---|---|
| 0 | 0.5000 | 49.40 | 0.08 | 0.15 | **best so far** |
| 1 | 0.3505 | 56.46 | 0.14 | 0.27 |  |
| 2 | 0.4691 | 58.20 | 0.09 | 0.19 |  |
| 3 | 0.2577 | 55.82 | 0.08 | 0.22 |  |
| 4 | 0.4536 | 41.44 | 0.11 | 0.22 |  |
| 5 | 0.3196 | 54.80 | 0.13 | 0.28 |  |
| 6 | 0.3814 | 56.64 | 0.12 | 0.26 |  |
| 7 | 0.3969 | 56.48 | 0.14 | 0.33 |  |
| 8 | 0.4381 | 55.26 | 0.12 | 0.26 |  |
| 9 | 0.3711 | 55.29 | 0.10 | 0.35 |  |
| 10 | 0.4175 | 48.70 | 0.10 | 0.33 |  |
| 11 | 0.4897 | 46.80 | 0.12 | 0.35 |  |
| 12 | 0.4691 | 49.64 | 0.12 | 0.38 |  |
| 13 | 0.4948 | 46.99 | 0.18 | 0.37 |  |
| 14 | 0.5464 | 47.04 | 0.12 | 0.31 | **best so far** |
| 15 | 0.4897 | 45.79 | 0.13 | 0.31 |  |
| 16 | 0.5464 | 42.35 | 0.11 | 0.33 |  |
| 17 | 0.4897 | 42.27 | 0.11 | 0.33 |  |
| 18 | 0.5103 | 46.44 | 0.13 | 0.33 |  |
| 19 | 0.5412 | 44.30 | 0.12 | 0.36 |  |
| 20 | 0.5361 | 41.36 | 0.13 | 0.41 |  |
| 21 | 0.5567 | 42.91 | 0.11 | 0.38 | **best so far** |
| 22 | 0.6186 | 39.43 | 0.14 | 0.31 | **best so far** |
| 23 | 0.4742 | 43.78 | 0.10 | 0.34 |  |
| 24 | 0.5000 | 45.66 | 0.13 | 0.30 |  |
| 25 | 0.5722 | 42.61 | 0.10 | 0.31 |  |

Iteration **22** is what `best/` exports, at 0.6186. 25 (0.5722) and 21 (0.5567)
are the other strong ones, and success is not monotonic — 23 drops to 0.4742
right after the peak — so a later iteration is not automatically a better one.
