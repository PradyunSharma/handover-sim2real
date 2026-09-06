# run19 — DAgger run 19 (right camera only), iteration 19

**Verified.** Strict-loads and matches its log.

| file | status |
|---|---|
| `best.pt` | run 19 iter 19, **verified** |
| `normalization.npz` | run 19's own, **verified** (md5 `9c47eb99…`) |
| `config.yaml` | the iteration's own training config, **verified** (strict-loads) |
| `source.txt` | provenance, copied from the run |

## The one thing that makes this checkpoint different

`SIM.cfg_file` is **`examples/pretrain_right.yaml`** — a single fixed side
camera. Not wrist-only like run 12, not wrist+left+right like run 16. This
policy has never seen a wrist view, and its dataset
(`train_pinned_omg_right_ok.h5`) is right-camera-only.

That makes it the natural match for a **tripod-only** rig, which is how this
robot has actually been running. run 16 scores higher in sim (0.80 against 0.72)
but has only ever seen three viewpoints at once; run 19 saw exactly one.

## best.pt — verified as iteration 19

Copied from `output/dagger_runs/dagger4_run19/best/checkpoints/best.pt`. Three
confirmations it is iteration 19 and not the last iteration:

* `best_val_loss` is **0.3420485**, matching `dagger_log.csv` row `iter 19`
  (`best_val_loss 0.342`) and no other row;
* `epoch` is 18, i.e. the best epoch inside iteration 19's 25-epoch fit;
* `best/source.txt`, written by the run itself, says
  `DAgger iteration 19 — best success_rate=0.7200`.

Iteration 19 is the last `is_best=1` row and the run's peak. Iterations 23 and
12 tie next at 0.70.

It **strict-loads all 86 tensors** (2.0M parameters) into the policy built from
`config.yaml`, and `_assert_state_layout` passes — `drop_joint_state: true`,
`use_prev_act: false`, so the runner's `robot_state[18:26]`-only fill is correct.

## normalization.npz — verified as run 19's own

md5 `9c47eb994008923a1904902b549f6320`, which differs from both run 16's
(`2cceb092…`) and run 12's (`172e234d…`), and **all four arrays** differ
numerically from run 16's. The normalizer is per-run and per-dataset, so
inheriting another run's would mis-scale every emitted action — a failure that
presents as a working policy behaving badly rather than as an error.

* `state_mean[18:21]` is `(0.622, -0.118, 1.491)`. The z near 1.5 m confirms the
  state is in the **sim world frame**, so `T_SIMWORLD_BASE` applies unchanged.
* action std is 0.016–0.020 m and 0.061–0.080 rad — the same magnitudes as run
  12 and run 16, so the runner's per-step safety clamp is still correctly sized.
* no channel has zero std.

## Run it

```bash
cd handover_sim2real/sim2real
python my_policy_runner.py --run run19 --cameras tripod \
       --calib-session d455 --home --step-mode
```

Tripod-only is the configuration this was trained for, so unlike run 16 there is
no viewpoint mismatch to reason about.
