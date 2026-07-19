# Running Phase-3 RL on DelftBlue (SLURM)

`train_rl.py` was a single-process loop: it collected rollouts **one episode at a
time** while the GPU (a tiny PointNet++ actor/critic) sat mostly idle. The cost is
the rollout — PyBullet stepping + OMG trajectory optimization, both CPU-bound. The
paper hides this with `num_remotes` parallel Ray actors; the cluster path here does
the same with a process pool.

## What changed

- **`--num-workers N`** on `train_rl.py` fans rollout collection across `N`
  persistent worker processes (each owns its own env + OMG planner + a CPU copy of
  the actor). The learner keeps the GPU. `N=1` (default) is the *exact* original
  single-process loop — nothing changes unless you ask for workers.
- **`examples/configs/rl_phase1_cluster.yaml`** rescales the loop so all 16 workers
  stay busy (`episodes_per_iter=16`, `updates_per_iter=800`, `num_iters=250`). It
  preserves total episodes (4000), total updates (200k), the replay ratio, batch
  size, and the per-update mix schedule — only the per-*iter* cadence is rescaled.
- **GPU knobs** (TF32 matmul + `cudnn.benchmark`) are enabled automatically on CUDA.

Speedup is ~linear in workers on the collection phase (the bottleneck), so ~10–15×
wall-clock on a 16-core allocation is the expectation; the GPU update phase is a
small fraction either way.

## 1. Smoke-test interactively first (2 min)

Grab a short interactive GPU allocation
([docs](https://doc.dhpc.tudelft.nl/delftblue/Slurm-interactive-jobs/)) and run a
tiny job to confirm the parallel path comes up on *your* env before burning a batch
job:

```bash
srun --partition=gpu-a100 --account=research-XX-YYY \
     --cpus-per-task=6 --gpus-per-task=1 --mem-per-cpu=4G --time=00:20:00 \
     --pty bash

# inside the allocation:
conda activate pch2r_dev
export OMG_PLANNER_DIR=$PWD/OMG-Planner GADDPG_DIR=$PWD/GA-DDPG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python examples/train_rl.py \
    --sim-cfg examples/pretrain.yaml \
    --rl-cfg  examples/configs/rl_phase1_cluster.yaml \
    --bc-run  output/bc_runs/dagger_iter_2_3 \
    --demos   output/rl_demos/train_h30.h5 \
    --run-name rl_smoke \
    --num-workers 4 --episodes-per-iter 4 --updates-per-iter 40 \
    --num-iters 3
```

You should see `[parallel] 4 rollout workers ...` then per-iter lines. If it hangs
at startup, an env var (`OMG_PLANNER_DIR` / `GADDPG_DIR`) or the conda env is wrong
— the manager prints which worker failed.

## 2. Submit the real job

Edit the two placeholders in `train_rl.sbatch` (`--account`, and the conda
activation / module lines if your setup differs), then:

```bash
sbatch examples/slurm/train_rl.sbatch
squeue --me                      # watch the queue
tail -f slurm_logs/rl_<jobid>.out
scancel <jobid>                  # to stop
```

The job requests 1 A100 + 18 CPUs and runs 16 rollout workers. Checkpoints land in
`output/rl_runs/rl_run8/checkpoints/`; if the 24 h walltime kills it, resume with
`--resume output/rl_runs/rl_run8/checkpoints/last.pt`.

## Scaling to a different core count

The shipped config is tuned for **16 workers**. To use `W` workers, keep the
invariants: `episodes_per_iter = W`, `updates_per_iter = 50·W`,
`num_iters = 4000/W`, and scale `beta_ramp_iters` / `demo_frac_ramp` /
`eval_every` / `save_every` by `2/W` relative to `rl_phase1.yaml`. The sbatch
already passes `--episodes-per-iter`/`--updates-per-iter` derived from
`--cpus-per-task`; if you change the core count, also regenerate the iter-based
ramps in the config (or pass `--num-iters`).

## Notes

- **Use `--worker-device cuda` (required, not cpu).** The `--worker-device cpu`
  default does NOT work here: the actor's PointNet++ encoder calls pointnet2_ops
  furthest-point-sampling, which is **CUDA-only** (no CPU kernel). A cpu worker
  therefore raises "CPU not supported" on every *policy* episode, which the manager
  swallows as a skip — you get `skip=N`, `buf=0`, and no learning (expert episodes
  still work, which masks it). With `cuda`, the `N` workers each hold a small actor
  context on the shared GPU alongside the learner — fine on a 32 GB V100 at 16 workers.
- Do **not** pass `--egl`: pybullet's EGL renderer leaks GPU memory per scene and
  will OOM a long run (the CPU renderer is leak-free). This matters more with many
  workers.
- `--num-workers 1` reproduces the original behavior exactly — use it to A/B.
