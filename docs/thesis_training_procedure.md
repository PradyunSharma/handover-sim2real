# Training Procedure (Phase 3)

Reference configuration throughout this section: `examples/configs/rl_phase1_cluster_r28.yaml`
(run 28). All numeric values quoted are from that file unless stated otherwise.
Phase-1 offline behaviour cloning is not covered here; the only role Phase 1 plays
below is as a supplier of network dimensions and normalisation statistics.

## 1. Structure of the training procedure

Training proceeds in three stages:

| Stage | What happens | Data source | Config keys |
|---|---|---|---|
| 0. Demonstration collection | Play the full motion-planner trajectory on every scene, commit the grasp, write transitions to disk | Simulator + OMG planner, no network | `collect_rl_demos.py` |
| 1. Offline pre-training | 2000 gradient updates on demo-only batches, before any rollout | Fixed demo pool | `LOOP.pretrain_updates: 2000` |
| 2. Online training | 250 iterations of (collect episodes → gradient updates), losses = RL + imitation | Online rollouts + demo pool | `LOOP.num_iters: 250` |

Stage 0 is run once and its output is reused across many training runs. Stages 1
and 2 use exactly the same `TD3BCTrainer.update()` call and the same loss; the
only difference is where the batch comes from.

## 2. Initialisation: from scratch

Run 28 sets `warm_start: false`. Both networks (actor, critic, both PointNet++
encoders, both robot MLPs, all heads) start from random initialisation. The
Phase-1 behaviour-cloning run directory is still passed on the command line, but
it is used only for:

- the network dimensions (`feature_dim`, `robot_hidden`, `pointnet_*`), and
- the `Normalizer` (per-channel mean/std of robot state and of the action), which
  is camera- and architecture-independent.

No behaviour-cloning weights are loaded.

Three reasons from-scratch was adopted:

1. `MODEL.drop_joint_state: true` removes joint positions and velocities from the
   robot-MLP input (26 → 8 dims: EE position 3, EE quaternion 4, gripper 1). This
   changes the first layer's input width, so behaviour-cloning weights cannot be
   loaded. `train_rl.py` raises an error if both flags are set.
2. When the camera configuration changes (`DATA.pc_channels: 5 → 6` for the
   three-class fixed-camera pipeline), the observation distribution changes and
   the egocentric 5-channel behaviour-cloning encoder does not transfer.
3. The behaviour-cloning base was measured to be weak in closed loop despite good
   per-step validation loss, so warm-starting mainly imports a covariate-shift
   failure rather than useful competence.

Optional middle ground (not used in run 28): `MODEL.pc_pretrained` seeds only the
PointNet++ backbone from a pre-trained state-feature checkpoint while leaving
heads random. Tensors whose shapes do not match are skipped.

## 3. Stage 0 — the fixed demonstration dataset

Produced by `examples/collect_rl_demos.py`. Per scene:

1. Reset the environment to that scene; run the OMG planner once with horizon
   `rollout_max_steps` to obtain a joint-space trajectory that ends at the
   selected grasp.
2. Execute the trajectory **by index** (waypoint `plan[t]` at step `t`), not by
   replanning each step. Replanning-and-taking-the-first-waypoint suffers a Zeno
   effect: each step covers only a fraction of the remaining distance, the
   end-effector stalls at the planner's standoff and never arrives, producing zero
   successful demonstrations.
3. After the last waypoint, append one **close-at-grasp** transition: zero
   displacement, gripper commanded closed, terminal reward evaluated.
4. Compute discounted Monte-Carlo returns backwards over the episode.

The reward uses `reward_mode: stable_grasp`: after the close, the gripper is held
shut for `hold_steps: 3` policy steps and reward 1 is given only if the
simulator's own release handshake fired (the human hand lets go only when the
robot has actually gripped) and the object was neither dropped nor did human
contact occur. This is the contact-hold analogue of the lift test used in
GA-DDPG, and needs no privileged grasp pose at reward time.

Each transition stores exactly the fields the online rollout worker stores — the
demo collector calls the *same* function (`RolloutWorker.expert_rollout_episode`)
that generates online expert episodes, so demonstrations and online expert data
are byte-identical in format and in reward semantics.

Practical constraints on the demo pool:

- **Horizon must match.** The clock is `(T − t)/T`; a pool collected at `T = 20`
  mixed with online data at `T = 30` gives inconsistent clock values. The
  collector reads `LOOP.rollout_max_steps` from the same config.
- **Camera configuration and channel count must match** the training run (the
  point-cloud pipeline is shared).
- **Grasp filter must match**: run 28 uses the offline hand-collision pre-filter
  (`valid_grasp_dict_005.pkl`), so the demos must be collected with the same
  filter, otherwise the scene set differs.
- Output is streamed to HDF5 one episode at a time, so memory stays bounded and a
  crash leaves all completed episodes usable.
- Collection must run **without** the EGL renderer: PyBullet's EGL plugin leaks
  roughly 85 MB of GPU memory per scene reset and OOM-kills long collections; the
  CPU rasteriser is leak-free and only slightly slower.

At training time the pool is loaded into a **non-evicting** replay buffer, so its
positive-reward transitions can never be flushed out by online data.

## 4. Stage 1 — offline pre-training on the fixed dataset

`pretrain_updates: 2000` gradient updates are taken on batches drawn entirely
from the demo pool, before a single online rollout. This runs only for a fresh
run (skipped when resuming from a checkpoint).

Purpose: calibrate the critic before the deterministic policy gradient can act on
it. With a randomly initialised critic, the actor immediately finds
out-of-distribution actions with spuriously high Q, and the resulting gradient
dominates the shared gradient-norm clip and starves the imitation term. Measured
effect: with offline pre-training, `q_mean` at online iteration 0 was 0.29 versus
0.03 without, and offline pre-training was the change that produced the first
non-zero online successes.

Because it uses the same update function, offline pre-training also fits the
actor by imitation on demonstration states, the gripper classifier, and the
auxiliary goal head — it is not critic-only.

Ablation note: the alternative "warm-up episodes" mechanism
(`warmup_episodes`, expert-heavy rollouts to seed the buffer before training) is
set to 0 in run 28. It was redundant once expert episodes were included in the
main loop.

## 5. Stage 2 — the online loop

One iteration, repeated `num_iters: 250` times:

```
for it in range(num_iters):
    beta   = schedule(it)                     # expert action mixing
    ei_lo, ei_hi = curriculum(it)             # reverse-curriculum takeover band
    df     = demo_frac(it)                    # demo fraction in each batch

    collect(episodes_per_iter = 16)           # fanned over 16 worker processes
    for _ in range(updates_per_iter = 800):
        batch = mix(online_FIFO, demo_pool, batch_size = 64, df)
        trainer.update(batch)

    if it % eval_every == 0:  evaluate(64 deterministic episodes)
```

Totals for run 28: 250 × 16 = 4000 episodes and 250 × 800 = 200 000 gradient
updates. With roughly 400 new transitions produced per iteration (16 episodes ×
~25 steps) and 800 × 64 = 51 200 samples consumed per iteration, the replay ratio
is about 128 gradient samples per fresh transition.

### 5.1 Episode types

Each of the 16 episodes per iteration is drawn as one of two kinds:

**Expert episodes** (`expert_episode_frac: 0.5`, i.e. half of them). Full
planner playback plus close, exactly as in Stage 0, but written into the *online*
buffer. These are a guaranteed fresh source of positive reward. They are
necessary because the policy earns almost no reward on its own early in training;
without them the fraction of positive-reward transitions in the online buffer
stays at zero, the critic devalues everything, and the actor drifts toward the
mean of its own zero-reward rollouts. The frozen demo pool alone does not fix
this, because it only covers expert states and not the states the current policy
actually visits. Expert episodes are excluded from the reported rollout success
rate so that metric stays a pure policy-progress signal.

**Policy episodes** (the other half). Driven by the current actor with:

- a reverse-curriculum warm start of `ei ~ Uniform[ei_lo, ei_hi]` steps of
  committed-plan playback before the policy takes over (Section 6.1);
- Gaussian exploration noise on the normalised action, `noise_std: 0.03`;
- per-step probability `beta` of executing an expert action instead of the
  policy's (Section 6.2);
- DAgger tail replanning with probability `dagger_ratio: 0.5` per step
  (Section 5.2);
- DART perturbations with probability `dart_ratio: 0.2` inside a step window
  (Section 5.3).

### 5.2 Supervision generated during rollout

Every online transition carries, in addition to $(s, a, r, s', \text{terminal})$:

| Field | How it is produced | Consumed by |
|---|---|---|
| `expert_action` (6) | Delta from the current state to the committed plan's waypoint for step $t$ | Pose imitation term |
| `expert_gripper` (1) | 1 if the EE is far from the grasp, 0 if within (0.02 m, 0.34 rad) | Gripper BCE |
| `goal_pose`, `next_goal_pose` (9) | EE-relative final grasp pose, pos + 6-D rotation | Auxiliary head, optional potential shaping |
| `mc_return` (1) | Discounted return-to-go, filled backwards after the episode | Critic target blend |
| `expert_flag`, `gripper_flag`, `perturb_flag` | Validity / masking | Loss masking |

The imitation labels are therefore **not** a fixed dataset: they are computed at
the states the current policy actually visits, which makes the imitation term
DAgger-style on-policy relabelling rather than offline behaviour cloning.

Two details matter for label quality:

- **Plan-tracking labels.** The planner is run once at step 0 and the label at
  step $t$ is the delta from the current (possibly drifted) state to `plan[t]`.
  An earlier variant — replan every step, take the first waypoint — has a
  stationary attractor at the planner's standoff roughly 8 cm short of the grasp,
  and trained the policy to hover there.
- **DAgger tail replanning.** With probability `dagger_ratio: 0.5`, and only for
  `dagger_min_step: 5 < t < len(plan) − dagger_tail_guard: 8`, the remainder of
  the plan is re-fitted from the policy's current state over the remaining step
  budget and spliced in. The tail guard prevents replanning inside the final
  approach, so the standoff-to-grasp labels stay committed. If replanning fails,
  the old plan is kept.

### 5.3 DART perturbations

Warm-start playback and DAgger both stay on the plan, so the policy never sees
recovery from a laterally off-plan state near the grasp — which is exactly the
state a from-scratch policy arrives in. DART injects that coverage:

- `dart_mode: policy` — on a policy step in `[dart_min_step: 15,
  dart_max_step: 22)`, with probability `dart_ratio: 0.2`, replace the policy
  action by a random task-space jump of ±`dart_pos_mag: 0.04` m and
  ±`dart_rot_mag: 0.2` rad, gripper open. The following steps' plan-tracking
  labels then demonstrate the recovery. The perturbed transition is flagged and
  **excluded from the critic's Bellman fit** (its stored action is artificial and
  its Q-target meaningless) but kept for the imitation term.
- `dart_mode: expert` — perturb an *expert* episode instead, replan the tail from
  the perturbed state, and let the planner execute the recovery by index. Here
  the recorded steps have consistent dynamics, so no critic masking is needed.
- `dart_mode: both` — enable both.

The window is deliberately placed just before the final approach: perturb near
the standoff, recover and realign, then descend.

### 5.4 Batch composition

`mix_batch` draws `round(df × 64)` samples from the non-evicting demo pool and the
rest from the online FIFO (`capacity: 20000` transitions). This is the DDPGfD
scheme: the rare terminal +1 transitions are guaranteed present in every batch
regardless of what the online buffer currently contains.

## 6. Curriculum and schedules

Five quantities vary over training. All are per-iteration except the
policy-gradient mix ramp, which is per-update.

### 6.1 Reverse curriculum (the main mechanism)

`expert_initial` is the number of committed-plan playback steps executed at the
start of a policy episode before the actor takes over. Two schedule modes exist.

**Sliding window** (earlier runs). The upper bound anneals
`expert_initial_steps → expert_initial_end` and a fixed-width window follows it
down. The policy first practises only the final descent and close, then
progressively farther starts.

**Widening band** (run 28, `expert_initial_lo_start` present). The upper bound is
held fixed and only the lower bound anneals:

| Key | Value | Meaning |
|---|---|---|
| `expert_initial_hi` | 24 | fixed upper bound |
| `expert_initial_lo_start` | 22 | band is [22, 24] at iteration 0 |
| `expert_initial_lo_end` | 2 | band is [2, 24] after annealing |
| `expert_initial_anneal_iters` | 150 | linear over iterations 0–150, then held |

So with a 30-step horizon, at iteration 0 the policy controls only the last 6–8
steps; by iteration 150 it controls anywhere between the last 6 and the full 30.

Rationale. The reward is sparse and terminal, and a from-scratch policy cannot
reach the grasp on its own within the horizon, so it never earns reward and
never grows a useful value function. Handing it a near-grasp state means it can
finish, close, and earn its own +1; the critic's high-value region then grows
outward from the grasp, and by the time harder starts are sampled the policy
gradient has something to pull toward. The *widening* variant (rather than
sliding) retains a thin tail of near-grasp takeovers for the whole run, because
under a sliding window the endgame skill decays once the window moves past it.

Setting the annealing keys to nothing recovers uniform sampling in
`[0, expert_initial_steps]`, which was shown to be insufficient: intermediate
takeovers fail for lack of a mastered endgame, so they earn no reward and the
value function never extends past the last couple of steps.

### 6.2 Expert action mixing (β)

Per-step probability of executing a plan waypoint instead of the policy action,
inside a policy episode. Linear `beta_start: 0.5 → beta_end: 0.0` over
`beta_ramp_iters: 63`, then zero. Early on this keeps episodes near the plan;
later the policy is fully self-driven. Note this is exploration/data-collection
only — the imitation label is computed at every step regardless of β.

### 6.3 Demonstration fraction

Linear `demo_frac_start: 0.5 → demo_frac_end: 0.3` over `demo_frac_ramp: 125`
iterations, then held at 0.3. The end value is a permanent floor, not zero: the
demo pool is the only guaranteed source of terminal +1 transitions, and annealing
it to zero removed the anchor and coincided with performance decay. If
`demo_frac_start` is absent, the code falls back to the constant `demo_frac`.

### 6.4 Policy-gradient / imitation blend (λ)

Two mutually exclusive forms:

- `pg_normalize: true` (run 28): λ = α / mean|Q|, with `alpha: 0.1`. This is the
  TD3+BC normalisation — the policy-gradient term's magnitude is bounded by α
  regardless of the scale of Q.
- `pg_normalize: false`: λ ramps `mix_start: 0.1 → mix_end: 0.2` over
  `mix_ramp: 50000` *updates*, and the imitation term is scaled by (1 − λ). This
  is the GA-DDPG form.

The normalised form was adopted after the fixed-λ form diverged repeatedly: the
policy-gradient term grows without bound as Q grows, consumes the entire shared
gradient-norm clip budget, and starves the imitation gradient (whose SmoothL1
gradient saturates at ±1). Observed divergence signature: `q_pi` rising to $10^4$–$10^5$
while `critic_loss` stayed small, and mean action magnitude reaching $10^8$. This
happened at λ ≈ 0.2, i.e. at the value used in the reference implementation, so
it is not simply a matter of choosing a smaller mix.

### 6.5 Exploration noise

Constant `noise_std: 0.03` in run 28. It is an available annealing axis but was
not annealed.

## 7. Evaluation and checkpoint selection

- In-loop evaluation every `eval_every: 7` iterations, `eval_episodes: 64`
  episodes, fully deterministic: β = 0, noise = 0, `expert_initial = 0`, no DAgger
  replanning, scenes swept in fixed order.
- This evaluation uses the **same split as the rollouts**, so it is a
  training-scene progress signal, not a generalisation measure. Held-out
  validation and test numbers come from the separate `rollout_rl_policy.py`,
  which loads a checkpoint and evaluates deterministically on `--split val|test`.
  Validation is used for checkpoint selection, test for the final number.
- `best.pt` is written whenever in-loop evaluation success improves; `last.pt`
  every `save_every: 7` iterations. Selecting `best.pt` rather than `last.pt` is
  material — in one run the two differed by 33% versus 4% success on the same
  fixed scene set.

Per-iteration diagnostics written to `log.csv` (needed because success alone is
zero for long stretches and hides which sub-skill is failing):

| Group | Columns | What they diagnose |
|---|---|---|
| Schedule state | `beta`, `ei_lo`, `ei_hi`, `lam`, demo fraction | which curriculum stage produced a given result |
| Rollout geometry | `roll_min_pos`, `roll_close`, `roll_skip`, `roll_miss`, `roll_timeout`, `roll_fail` | reaching failure vs closing failure vs collision |
| Deterministic eval | `eval_succ`, `eval_min_pos`, `eval_min_rot`, `eval_close`, failure breakdown | from-scratch capability |
| Expert anchor | `exp_kept`, `exp_succ`, `buf_pos` | whether positive reward is present in the online buffer at all |
| Learner internals | `q_mean` (stored actions), `q_pi` (policy actions), `a_absmean`, `grip_logit`, `critic_loss`, `pg_loss`, `bc_loss`, `grip_loss`, `aux_*` | divergence, out-of-distribution exploitation, gripper drift, action-magnitude drift |

The gap between `q_mean` and `q_pi` is the primary early-warning signal for
critic exploitation; `grip_logit` drifting monotonically positive is the signal
for the gripper ceasing to fire; `a_absmean` inflating from ~0.7 to ~2 is the
signal for the aggressive-close failure mode.

## 8. Known failure modes that the procedure is built around

These motivate several of the choices above and are worth stating explicitly.

1. **Actor exploitation of the critic.** Unbounded policy-gradient magnitude →
   divergence. Mitigation: `pg_normalize: true`.
2. **Gripper logit blow-up.** The unbounded gripper logit fed into the critic let
   the policy gradient ride ∂Q/∂logit into an out-of-distribution region.
   Mitigation: the gripper channel is detached before entering the critic, and is
   trained only by its supervised term.
3. **Gripper drift to permanently open.** "Far" is the overwhelmingly common
   label, so plain BCE saturates the logit open and the gripper stops firing.
   Mitigations: class balancing (`gripper_close_weight_max: 10`) and label
   smoothing (`gripper_label_smooth: 0.1`).
4. **Post-curriculum decay.** Success peaks around the end of the curriculum
   anneal and then declines. Diagnosed as distribution shift / forgetting rather
   than instability (action magnitudes stay healthy). Three candidate levers,
   each ablated: larger replay capacity, higher and fixed demonstration fraction,
   and a curriculum lower bound that never reaches zero.
5. **Zero online reward.** Without expert episodes the online buffer contains no
   positive reward and the policy decays toward its own zero-reward behaviour.

## 9. Ablation axes

The following are the knobs that change the training procedure, all
config-driven, with the values that have been used. Runs referenced by number are
the configs under `examples/configs/`.

### 9.1 Initialisation and state

| Axis | Key | Values used | Tests |
|---|---|---|---|
| Warm start | `warm_start` | true / false (run 27, 28) | whether a behaviour-cloning prior helps or imports covariate shift |
| Encoder seeding | `MODEL.pc_pretrained` | path / none | value of a pre-trained point encoder without a policy prior |
| Robot-state trimming | `MODEL.drop_joint_state` | false / true (run 28) | whether joint-space state is redundant given EE-frame actions, and whether it is a memorisation handle |
| Previous action in state | `MODEL.use_prev_act` | false / true | copycat / causal confusion |
| Camera set and channels | sim cfg `CAMERAS`, `DATA.pc_channels` | wrist 5-ch; left+right 6-ch (run 33); wrist+left 6-ch (run 37) | egocentric vs external vs mixed viewpoints |

### 9.2 Reward

| Axis | Key | Values | Tests |
|---|---|---|---|
| Reward definition | `RL.reward_mode` | `stable_grasp` / `proximity` | contact-verified grasp vs geometric proxy to the planner's grasp |
| Hold length | `RL.hold_steps` | 3 | strictness of the stability check |
| Potential shaping | `RL.shaping_pos_weight`, `RL.shaping_rot_weight` | 0 (off) / >0 | whether a policy-invariant dense distance term accelerates learning under sparse reward |

### 9.3 Loss and blend

| Axis | Key | Values | Tests |
|---|---|---|---|
| PG scaling | `RL.pg_normalize` | true / false (run 29, 31) | TD3+BC normalisation vs GA-DDPG fixed mix; stability |
| PG strength | `RL.alpha` (normalised) or `mix_start`/`mix_end` | 0.1 / 2.5; 0.1→0.2, 0.1→0.3 | RL vs imitation balance |
| Imitation weight | `RL.bc_weight` | 1.0 / 2.0 | how strongly the on-policy planner labels lead |
| Pose loss form | `RL.pose_loss` | `smooth_l1` / `pm` | raw normalised L1 vs point-matching on gripper control points (proper SE(3) metric, correct rotation weighting) |
| Auxiliary weight | `RL.aux_weight` | 0 / 0.5 / 1.0 | value of the grasp-pose regression as an encoder regulariser |
| Critic target | `RL.mc_blend` | 0.0 / 0.5 | one-step Bellman vs blend with Monte-Carlo return, for propagating rare sparse successes |
| Gripper supervision | `gripper_bc_weight`, `gripper_close_weight_max`, `gripper_label_smooth` | 1.0; 10; 0.1 | class imbalance and logit saturation |
| Action magnitude penalty | `RL.action_reg_weight` | 0 / >0 | whether the late-run action inflation causes the aggressive-close failures |
| TD3 internals | `gamma`, `tau`, `policy_noise`, `noise_clip`, `policy_delay`, `grad_clip` | 0.95, 0.005, 0.2, 0.5, 2, 1.0 | standard TD3 sensitivity |

### 9.4 Curriculum

| Axis | Key | Values | Tests |
|---|---|---|---|
| Curriculum shape | presence of `expert_initial_lo_start` | uniform / sliding / widening | whether mastery must be sequenced from the grasp outward |
| Final band width | `expert_initial_lo_end` | 2 (run 28, 36) / 10 (run 32) / 15 (run 35) | whether retaining near-grasp practice prevents post-curriculum decay |
| Anneal length | `expert_initial_anneal_iters` | 150 of 250 | curriculum speed vs consolidation time |
| Upper bound | `expert_initial_hi` | 24 of 30 steps | how much of the episode is ever expert-driven |
| Expert episode share | `expert_episode_frac` | 0.25 / 0.5 | strength of the fresh positive-reward anchor |
| β schedule | `beta_start`, `beta_end`, `beta_ramp_iters` | 0.5 → 0.0 over 63 | how fast the policy is left on its own |

### 9.5 Data and replay

| Axis | Key | Values | Tests |
|---|---|---|---|
| Demo fraction | `demo_frac` or `demo_frac_start`/`_end`/`_ramp` | constant 0.05 (run 26); ramp 0.5→0.3 (run 28, 35); fixed 0.7 (run 32); fixed 0.9 (run 34) | demo reliance vs online exploration; over-anchoring to demo scenes |
| Replay capacity | `LOOP.capacity` | 20 000 (run 28, 34, 35) / 100 000 (run 32, 36) | whether good near-grasp transitions are being evicted before they are exploited |
| Offline pre-training | `LOOP.pretrain_updates` | 0 / 2000 | critic calibration before the policy gradient acts |
| Warm-up episodes | `LOOP.warmup_episodes`, `warmup_beta` | 0 / >0 | buffer seeding, largely superseded by expert episodes |

### 9.6 Data augmentation during rollout

| Axis | Key | Values | Tests |
|---|---|---|---|
| DAgger rate | `dagger_ratio` | 0 / 0.5 | on-policy relabelling from drifted states |
| DAgger window | `dagger_min_step`, `dagger_tail_guard` | 5, 8 | keeping final-approach labels committed |
| DART mode | `dart_mode` | `policy` / `expert` / `both` (run 24) | which side of the data generation should demonstrate recovery |
| DART rate | `dart_ratio` | 0 / 0.2 / 0.5 | how much off-plan coverage is needed |
| DART window and magnitude | `dart_min_step`, `dart_max_step`, `dart_pos_mag`, `dart_rot_mag` | 15–22, 0.04 m, 0.2 rad | where and how hard to perturb |
| Exploration noise | `noise_std` | 0.03 | exploration vs label consistency |

### 9.7 Environment and scene set

| Axis | Key | Values | Tests |
|---|---|---|---|
| Hand-collision grasp filter | `valid_grasp_dict_path` vs `hand_collision_filter` + `hand_collision_thresh` | offline pre-filter (run 21 onwards) vs runtime filter at 0.04 / 0.08 / 0.10 m | how many scenes remain usable; the runtime filter at 0.08 m keeps roughly 351 of 720 training scenes versus 716 for the offline filter, so this is also a scene-diversity ablation |
| Episode horizon | `rollout_max_steps` | 20 / 30 / (40) | whether failures are running out of time or running out of skill; requires re-collecting demos to keep the clock consistent |
| Scene-set size | `--num-scenes` | all / first N | overfitting probe: can the architecture solve a small fixed set at all |
| Split | `--split` | train / val / test | generalisation |

### 9.8 Compute scaling

| Axis | Key | Values | Notes |
|---|---|---|---|
| Run length | `num_iters` | 250 / 500 (run 34) / 700 (run 25) | whether extra training consolidates or re-exposes the decay |
| Parallelism | `--num-workers`, `episodes_per_iter`, `updates_per_iter` | 16, 16, 800 | pure change of "iteration" units; total episodes, total updates and the replay ratio are held fixed when rescaling, so results are comparable across worker counts |
| Batch size | `batch_size` | 64 | gradient noise |

When changing worker count W, the intended rescaling is
`episodes_per_iter = W`, `updates_per_iter = 50 W`, `num_iters = 4000 / W`, with
per-iteration ramps (`beta_ramp_iters`, `demo_frac_ramp`, `eval_every`,
`save_every`) scaled accordingly. `mix_ramp` is per-update and is not rescaled.
`capacity` and `batch_size` are per-transition and are not rescaled.
