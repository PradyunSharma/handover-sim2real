# cp3 — DAgger run 16 (wrist + left + right), iteration 16

**Complete and verified.** Nothing further to fetch.

| file | status |
|---|---|
| `config.yaml` | reconstructed, **verified** (strict-loads the checkpoint) |
| `best.pt` | run 16 iter 16, **verified** |
| `normalization.npz` | run 16's own, **verified** (md5 `2cceb092…`) |

## best.pt — verified as the right checkpoint

Copied from `output/dagger_runs/dagger4_run16/best/checkpoints/best.pt`
(md5 `d31302ee49f4a7dff0e1431e2a390ba0`). Two independent confirmations that it
really is iteration 16 and not the last iteration:

* its `best_val_loss` is **0.30812913**, matching `dagger_log.csv` row
  `iter 16` (`best_val_loss 0.3081`) and no other row;
* its `epoch` is 14, i.e. the best epoch inside iteration 16's 25-epoch fit.

Iteration 16 is the last `is_best=1` row, at `success_rate 0.80`; iterations
17–25 peak at 0.79 and never take the flag.

It **strict-loads all 86 tensors** into the policy built from `config.yaml`,
including the six `aux_head.*` tensors — which is the real proof the
reconstructed config matches the architecture the run was trained with.

## normalization.npz — verified as run 16's own

Copied from `output/dagger_runs/dagger4_run16/best/normalization.npz`
(md5 `2cceb0929b34b0aadc857cde696fb761`). Checked that it is genuinely run 16's
and not a stray copy of run 12's: the md5 differs, and **all four arrays**
(`state_mean`, `state_std`, `action_mean`, `action_std`) differ numerically.
That matters because the normalizer is per-run — run 16 was fit on
`train_pinned_omg_wlr_ok.h5` plus 25 wlr DAgger iterations, a different dataset
from run 12's — so silently inheriting cp2's would mis-scale every emitted
action, a failure that presents as a working policy behaving badly.

Two sanity checks on the values themselves:

* `state_mean[18:21]` (the EE position channels, the only ones the policy reads)
  is `(0.616, -0.116, 1.490)`. The z near 1.5 m confirms the state is in the
  **sim world frame**, not the robot base frame, so `T_SIMWORLD_BASE` in the
  runner still applies to cp3 exactly as it did to cp2.
* action std is 0.016-0.021 m and 0.06-0.08 rad — the same magnitudes as cp2, so
  the runner's per-step safety clamp is still correctly sized and needed no
  change for this checkpoint.
* no channel has zero std, so nothing divides by ~0.

## Then

```bash
cd handover_sim2real/sim2real
python my_policy_runner.py --policy-dir checkpoint/cp3 --cameras wrist,tripod \
       --home --step-mode
```

Read the "Two cameras against three" section of `README_SIM2REAL.md` first —
this checkpoint was trained on three viewpoints and you have two.
