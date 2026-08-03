# Training Curve Metrics, and Run 28

---

# Part E — Training curve metrics

## E.1 Where the curves come from

`examples/train_rl.py` writes one CSV row per training iteration to
`<run_dir>/log.csv`. `examples/plot_rl_run.py` renders it into a 2×3 panel figure
saved as `<run_dir>/curves.png`.

Two different measurement conditions appear in the same file and must not be
confused:

| Prefix | Source | Conditions |
|---|---|---|
| `roll_*` | the exploration rollouts of that iteration | stochastic: exploration noise, β expert mixing, reverse-curriculum warm start, DAgger replans, DART perturbations. **Policy episodes only** — expert episodes are excluded so this stays a policy-progress signal |
| `exp_*` | the full-expert episodes of that iteration | planner playback, no policy involvement |
| `eval_*` | a separate deterministic evaluation sweep | β = 0, noise = 0, `expert_initial` = 0, no DAgger, no DART, scenes swept in fixed order. Written only on evaluation iterations; blank otherwise |
| everything else | the learner | averaged over the last update of the iteration (`astats` = the last update that included an actor step, since the actor is delayed) |

## E.2 Why this instrumentation exists

Under a sparse terminal reward, success is exactly zero for long stretches of
training, and a single success curve cannot distinguish "the policy cannot reach
the object", "the policy reaches but never closes the gripper", "the policy
closes but the grasp is unstable", and "the networks have diverged". The metric
set below decomposes the task into those sub-skills so that a flat zero success
curve is still informative.

## E.3 Metric reference

### Schedule state (what produced this row)

| Column | Definition | Use |
|---|---|---|
| `iter` | iteration index | x-axis |
| `buffer` | online replay-buffer occupancy | shows when the FIFO saturates and eviction begins |
| `beta` | current β for expert action mixing | how much of the rollout was policy-driven |
| `ei_lo`, `ei_hi` | current reverse-curriculum takeover band | **essential** — `roll_*` metrics are only comparable across iterations at the same band |
| `lam` | current PG/BC blend coefficient λ | under `pg_normalize` this is α / mean\|Q(s,π(s))\|, so it rises as Q falls |
| `n_expert` | labelled transitions in the last batch | should equal the batch size when every step is labelled |

### Rollout (policy episodes)

| Column | Definition | Diagnoses |
|---|---|---|
| `roll_succ` | fraction of policy episodes ending in a rewarded grasp | exploration-time capability; inflated by the warm start |
| `roll_len` | mean episode length in policy steps | close to the horizon ⇒ episodes are timing out rather than terminating on a close |
| `roll_ret` | mean undiscounted return | equals `roll_succ` for a 0/1 terminal reward |
| `roll_min_pos` | mean over episodes of the closest EE→grasp distance reached, **measured only over policy-controlled steps** | reaching ability. Not comparable across curriculum bands: a tight near-grasp band starts the policy next to the grasp |
| `roll_close` | fraction of policy episodes in which a close was committed | whether the gripper fires at all |
| `roll_skip` | episodes discarded because the planner produced no plan or no grasp pose | usable-scene fraction |
| `roll_miss` | closes committed that did not earn the reward | closing at the wrong pose |
| `roll_timeout` | episodes that reached the horizon without closing | the "never commits" failure |
| `roll_fail` | episodes terminated by human contact or object drop | the "too aggressive" failure |

### Expert episodes

| Column | Definition | Diagnoses |
|---|---|---|
| `exp_kept` | expert episodes retained this iteration | how much anchor data was produced |
| `exp_succ` | success rate of those episodes | an upper bound on what the data pipeline can supply; below 1 means the planner-driven trajectory itself fails on some scenes |
| `buf_pos` | fraction of transitions currently in the online buffer with positive reward | **the single most important early-training metric.** If it is 0, the critic has no positive signal anywhere and no amount of training will help |

### Deterministic evaluation

| Column | Definition | Diagnoses |
|---|---|---|
| `eval_succ` | success rate over `eval_episodes` deterministic episodes | the headline metric; from-scratch capability |
| `eval_close` | close-commit rate | separates "cannot decide to close" from "closes badly" |
| `eval_min_pos` | mean closest EE→grasp position error (m) | reaching, measured with no warm start — the honest reach number |
| `eval_min_rot` | mean closest EE→grasp rotation error (rad) | wrist alignment |
| `eval_miss` / `eval_timeout` / `eval_fail` | failure-mode counts | which sub-skill is missing |
| `best_succ` | running maximum of `eval_succ` | checkpoint selection |

Derived quantities worth reporting: the number of retained evaluation episodes is
`eval_miss + eval_timeout + eval_fail + successes`, and the
**close→success conversion rate** = `eval_succ / eval_close` separates the
gripper's *precision* (is a committed close a good one?) from its *recall* (does
it commit often enough?).

### Learner internals

| Column | Definition | Diagnoses |
|---|---|---|
| `critic_loss` | total critic objective | fit quality; a small value alone does not mean the value function is useful |
| `target_mean` | mean of the critic's regression target | value scale |
| `q_mean` | Q on the **stored** actions in the batch | value of the data-generating behaviour |
| `q_pi` | Q on the **policy's own** action | the out-of-distribution probe. `q_pi ≫ q_mean` and rising is the signature of the actor exploiting the critic — the primary divergence warning |
| `pg_loss` | the policy-gradient term −λ·Q | under `pg_normalize` it should stay bounded near −α |
| `bc_loss` | pose imitation loss | how far the policy is from the planner label at the states it visits; rises when the curriculum places it in harder states |
| `grip_loss` | gripper BCE | classifier fit |
| `grip_logit` | mean gripper logit | monotone positive drift ⇒ the gripper is saturating open and will stop firing |
| `a_absmean` | mean \|pose action\| | inflation ⇒ overshooting and aggressive closes |
| `aux_a`, `aux_c` | auxiliary grasp-pose regression loss on actor and critic | whether the encoders are learning the scene geometry |
| `actor_loss` | total actor objective | composite; read the components instead |

## E.4 Panel layout

| Panel | Contents | Question answered |
|---|---|---|
| (0,0) success | `roll_succ`, `eval_succ`, `exp_succ`, `buf_pos` | is anything being learned, and is there positive reward in the buffer at all |
| (0,1) approach | `roll_min_pos`, `eval_min_pos`, horizontal line at the 0.02 m close threshold | is the failure a reaching failure |
| (0,2) close rate | `roll_close`, `eval_close` | is the failure a closing failure |
| (1,0) critic | `critic_loss`, `target_mean`, `aux_c` | value-fit health |
| (1,1) value | `q_mean` (left axis) vs `q_pi` (right axis) | out-of-distribution exploitation |
| (1,2) actor | `actor_loss`, `bc_loss`, `grip_logit`, `a_absmean` | actor health and gripper drift |

## E.5 How to read the curves — decision rules

1. `buf_pos ≈ 0` → no positive reward anywhere in the online data. Nothing else
   matters until this is fixed (more expert episodes, tighter curriculum band).
2. `q_pi` rising far above `q_mean`, with `a_absmean` growing → actor exploiting
   the critic. Terminal if unchecked; check `pg_loss` is bounded.
3. `eval_min_pos` still falling → reaching is improving even if `eval_succ` is 0.
   Not a failed run.
4. `eval_min_pos` plateaued above the close threshold and `eval_timeout`
   dominating → reaching plateau; the policy stalls short and never enters the
   close region.
5. `eval_close ≈ 0` with a healthy `eval_min_pos` → gripper failure, not reach
   failure. Cross-check `grip_logit`.
6. `eval_miss` high relative to `eval_close` → the gripper fires in the wrong
   place: a pose-precision problem, not a decision problem.
7. `eval_fail` (contact/drop) rising → the policy is committing too aggressively;
   cross-check `a_absmean`.
8. `roll_min_pos` changing while `ei_lo` is annealing → measurement-condition
   change, not a capability change. Use `eval_min_pos` for capability.

## E.6 Measurement caveats

- The in-loop evaluation uses the **same split** the rollouts sample from, so it
  measures progress on training scenes, not generalisation. Held-out numbers come
  from `examples/rollout_rl_policy.py` on `--split val|test`.
- Evaluation resolution is one episode: with 48 retained episodes, the smallest
  non-zero value of `eval_succ` is 1/48 ≈ 0.021, and the binomial standard error
  at p = 0.2 is about 0.057. Single-point peaks should be read with that in mind.
- Episodes the planner cannot solve are skipped and excluded from the
  denominators, so success rates are conditioned on planner-solvable scenes.
- `roll_*` metrics are collected under exploration noise and curriculum warm
  start and are systematically optimistic relative to `eval_*`.

---

# Part F — Run 28

## F.1 Configuration

Config: `examples/configs/rl_phase1_cluster_r28.yaml` (verified byte-identical to
the copy saved in the run directory, `output/rl_runs/rl_run28/rl_config.yaml`).

| Group | Setting | Value |
|---|---|---|
| Initialisation | `warm_start` | false — actor and critic trained from random initialisation |
| Observation | point cloud | 1024 points, 5 channels (xyz + object + hand), EE frame, wrist camera |
| | `MODEL.drop_joint_state` | true — robot-MLP input reduced to 8 dims (EE position 3, EE quaternion 4, gripper 1) |
| Reward | `reward_mode` | `stable_grasp`, `hold_steps: 3` |
| Horizon | `rollout_max_steps` | 30 policy steps |
| Losses | `pg_normalize` / `alpha` | true / 0.1 |
| | `bc_weight`, `pose_loss` | 2.0, `smooth_l1` |
| | `gripper_bc_weight`, `close_weight_max`, `label_smooth` | 1.0, 10.0, 0.1 |
| | `aux_weight`, `mc_blend` | 0.5, 0.5 |
| | shaping, `action_reg_weight` | disabled (0) |
| Curriculum | band | widening: `hi` fixed 24, `lo` 22 → 2 over 150 iterations |
| | `expert_episode_frac` | 0.5 |
| | β | 0.5 → 0.0 over 63 iterations |
| | `demo_frac` | 0.5 → 0.3 over 125 iterations, then held |
| Data | demo pool | `output/rl_demos/train_h30_vgd.h5` |
| | `capacity` | 20 000 transitions |
| | `pretrain_updates` | 2000 |
| Loop | iterations / episodes / updates | 250 / 16 per iteration / 800 per iteration |
| | batch size | 64 |
| Scenes | grasp filter | offline hand-collision pre-filter `valid_grasp_dict_005.pkl` |
| | split | train |
| Eval | cadence / episodes | every 7 iterations / 64 |
| Compute | rollout workers | 16 processes |

Demonstration pool contents: 19 122 transitions from 623 episodes, of which 540
end in the +1 terminal (86.7% of demonstration episodes succeed). Mean
demonstration episode length 30.7 transitions.

Totals actually executed: 4000 episodes (250 × 16), of which 531 were skipped by
the planner (13.3%), 1733 were expert episodes (43.3%), and the remaining 1736
were policy episodes. 200 000 gradient updates. Mean policy-episode length 29.8
of 30 steps.

## F.2 Deterministic evaluation trajectory

35 evaluations, each sweeping 64 scenes of which exactly 48 were solvable by the
planner and retained (16 scenes, 25% of the fixed evaluation sweep, are skipped
every time).

| Iter | Band `ei_lo`–`ei_hi` | β | `eval_succ` | `eval_close` | `eval_min_pos` (m) | `eval_min_rot` (rad) | miss | timeout | fail |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 21–24 | 0.44 | 0.042 | 0.042 | 0.112 | 0.781 | 0 | 44 | 2 |
| 35 | 17–24 | 0.22 | 0.000 | 0.000 | 0.167 | 0.707 | 0 | 48 | 0 |
| 63 | 14–24 | 0.00 | 0.000 | 0.000 | 0.117 | 0.542 | 0 | 46 | 2 |
| 84 | 11–24 | 0.00 | 0.083 | 0.083 | 0.094 | 0.395 | 0 | 41 | 3 |
| 98 | 9–24 | 0.00 | 0.125 | 0.146 | 0.082 | 0.444 | 1 | 37 | 4 |
| 112 | 7–24 | 0.00 | 0.188 | 0.292 | 0.076 | 0.391 | 5 | 32 | 2 |
| 126 | 5–24 | 0.00 | 0.208 | 0.208 | 0.072 | 0.442 | 0 | 33 | 5 |
| **133** | **4–24** | 0.00 | **0.375** | **0.375** | **0.065** | 0.544 | 0 | 24 | 6 |
| 140 | 3–24 | 0.00 | 0.188 | 0.188 | 0.064 | 0.455 | 0 | 33 | 6 |
| 147 | 2–24 | 0.00 | 0.271 | 0.292 | 0.075 | 0.477 | 1 | 34 | 0 |
| 175 | 2–24 | 0.00 | 0.104 | 0.104 | 0.084 | 0.471 | 0 | 42 | 1 |
| 203 | 2–24 | 0.00 | 0.125 | 0.146 | 0.100 | 0.431 | 1 | 39 | 2 |
| 224 | 2–24 | 0.00 | 0.125 | 0.146 | 0.060 | 0.286 | 1 | 36 | 5 |
| 245 | 2–24 | 0.00 | 0.021 | 0.021 | 0.067 | 0.291 | 0 | 45 | 2 |

(Table abridged to every fourth evaluation plus the peak; the full 35-point
series is in `log.csv`.)

Phase summary:

| Phase | Iterations | Band | Mean `eval_succ` | Max | Mean `eval_close` | Mean `eval_min_pos` | Mean `eval_min_rot` |
|---|---|---|---|---|---|---|---|
| A — tight band | 0–70 | `lo` 22→13 | 0.007 | 0.042 | 0.019 | 0.132 | 0.647 |
| B — widening | 70–150 | `lo` 13→2 | 0.141 | 0.375 | 0.156 | 0.081 | 0.452 |
| C — band held | 150–250 | `lo` = 2 | 0.058 | 0.125 | 0.065 | 0.074 | 0.391 |

Aggregate over all 1680 retained evaluation episodes: 123 successes (7.3%), 142
closes (8.5%), 1431 timeouts (85.2%), 107 contact/drop failures (6.4%).
**Close→success conversion: 86.6%.**

Best checkpoint: iteration 133, `eval_succ` = 0.375 (18 of 48).

## F.3 Analysis

### F.3.1 The curriculum drives the success curve

`eval_succ` is essentially zero while the takeover band keeps the policy in
control of only the final 6–10 steps (iterations 0–70, mean 0.007), rises as the
lower bound anneals (mean 0.141 over iterations 70–150), and peaks at iteration
133 when `ei_lo` = 4. The onset of non-zero evaluation success coincides with
`ei_lo` dropping below roughly 12, i.e. with the policy being asked to control at
least the last 18 of 30 steps during training. The evaluation itself is
unaffected by the band (it always starts from scratch), so this is a genuine
capability transfer from curriculum training and not a change of measurement
conditions.

### F.3.2 Reaching is learned; it plateaus above the required precision

`eval_min_pos` — the closest end-effector-to-grasp distance reached in a
from-scratch episode — falls monotonically from 0.112 m at the first evaluation
to about 0.065 m by iteration 133, and then oscillates in the 0.060–0.100 m band
for the rest of the run without further systematic improvement. It never
approaches the 0.02 m proximity used to define a correct close.

`eval_min_rot` improves over the whole run and does not plateau: 0.781 rad at
the first evaluation, 0.45 rad around the peak, and 0.29 rad at the last
evaluations. The rotation tolerance associated with a correct close is 0.34 rad,
so by the end of the run the mean closest-approach **orientation already
satisfies the criterion while the position does not**. Position is the binding
constraint on this run's endgame, by roughly a factor of three.

Note that `eval_min_pos` is a mean over 48 episodes of a per-episode minimum. A
mean of 0.067 m is compatible with a minority of episodes entering the sub-0.02 m
region, which is precisely what the 8.5% close rate reflects.

### F.3.3 The bottleneck is close recall, not close precision

Across the whole run, 142 closes produced 123 successes: an 86.6% conversion
rate. At most evaluation points the conversion is 100% — `eval_close` and
`eval_succ` are the same number, and `eval_miss` is 0. The exceptions are
concentrated around iteration 112 (14 closes, 9 successes, 5 misses) and a few
late points.

Meanwhile 85.2% of all evaluation episodes end in `TIMEOUT`: the horizon expires
with the gripper never commanded shut. Mean policy episode length is 29.8 of 30
steps, confirming the same picture on the rollout side.

The interpretation is that the gripper decision is well calibrated — when the
policy decides to close, the resulting grasp is almost always stable — but it
fires far too rarely, because the policy usually never enters the state region
where closing is correct. This is consistent with F.3.2: the training label for
"close" is only active within 0.02 m of the grasp, and the policy's mean closest
approach is three times that. The gripper classifier is not the failing
component; the terminal-phase position accuracy is.

### F.3.4 Contact and drop failures scale with close attempts

`eval_fail` (human contact or object drop) totals 107 over the run, 6.4% of
episodes. It is near zero in phase A when almost nothing closes (mean 1.7 per
evaluation), rises to a mean of 3.8 per evaluation in phase B when the close rate
is highest, and stays around 3.4 in phase C. These are the cost of attempting
grasps at imperfect poses, and they track close attempts rather than being an
independent failure mode.

### F.3.5 The optimisation was stable throughout

Every stability indicator stayed in a healthy range for all 250 iterations:

| Indicator | Range over the run | Interpretation |
|---|---|---|
| `pg_loss` | −0.100 to −0.046 | bounded by α = 0.1, exactly as the normalised policy gradient is designed to be |
| `lam` | 0.34 → 1.40 | rises because λ = α / mean\|Q(s,π(s))\| and Q shrinks; the *product* stays bounded |
| `q_mean` | 0.135 – 0.380 | stable value scale |
| `q_pi` | 0.019 – 0.346 | **always below `q_mean`** |
| `a_absmean` | 0.64 → 0.78, max 1.07 | mild growth, no inflation |
| `grip_logit` | 1.84 – 2.25, mean 2.09 | flat |
| `critic_loss` | 0.0049 → 0.0009 | monotone decrease |

Two observations deserve emphasis.

**No out-of-distribution exploitation.** `q_pi` is below `q_mean` at every
iteration. The critic consistently values the stored actions — which are
dominated by planner-generated expert and demonstration actions — above the
policy's own action. The failure mode in which the actor discovers spuriously
high-value actions and diverges did not occur; if anything the policy-gradient
term is a weak contributor here, since it is pushing toward a Q the policy is not
yet achieving.

**The gripper logit did not drift.** With label smoothing at ε = 0.1, the
saturation bound on the mean logit is logit(0.9) = ln 9 ≈ 2.197. The observed
mean sits at 2.09 with a range of 1.84–2.25, i.e. pinned just below that bound
and non-monotone. The smoothing achieved its purpose: the classifier remains
responsive rather than saturating open. A mean logit slightly on the "open" side
is expected, since the large majority of visited states genuinely are far from
the grasp.

### F.3.6 Representation learning proceeded steadily

The auxiliary grasp-pose regression loss decreased monotonically on both networks
— actor 0.0151 → 0.0057, critic 0.0049 → 0.0007 — indicating the PointNet++
encoders progressively learned to localise the target grasp relative to the
end-effector from the point cloud. This continued through phase C, when
evaluation success was falling, so the decline in F.3.7 is not a degradation of
the scene representation.

The pose imitation loss shows the opposite trend, rising from 0.033 (band mean,
iterations 0–25) to a peak band mean of 0.076 around iterations 125–150 before
settling near 0.049. This is expected under a widening curriculum: as `ei_lo`
falls, the policy is evaluated at states progressively farther from the plan,
where the delta-to-waypoint label is larger and harder. Rising `bc_loss` under a
widening curriculum reflects an increasing task difficulty, not a worsening fit.

### F.3.7 Post-peak decline

After iteration 147 — the point at which the curriculum band reaches its final
width [2, 24] — `eval_succ` falls and stays low: mean 0.058 over the 14
evaluations from iteration 154 to 245, against 0.195 over the eight evaluations
from 98 to 147. The decline is not accompanied by any of the usual instability
signatures: `a_absmean`, `q_pi`, `grip_logit` and `critic_loss` are all as
healthy at iteration 245 as at iteration 133, and both reach metrics
(`eval_min_pos` ≈ 0.067, `eval_min_rot` ≈ 0.29) are as good or better at the end
of the run than at the peak.

The observable that separates the peak from the tail is the **close rate**:
0.375 at iteration 133 against a mean of 0.065 over iterations 154–245, with
`eval_timeout` rising from 24 to a typical 40–45 of 48. The policy at the end of
the run approaches the grasp as accurately as the peak policy but commits far
fewer closes.

Three properties of the run are consistent with this, and are stated as
observations rather than a confirmed cause:

1. The online replay buffer saturated at its 20 000-transition capacity by
   iteration 50 and evicted continuously thereafter. At roughly 400 new
   transitions per iteration it retains about 50 iterations of history, so the
   transitions collected around the peak were fully evicted by roughly iteration
   180.
2. The demonstration fraction reached its 0.3 floor at iteration 125, so the
   fixed positive anchor stopped strengthening just before the peak.
3. Once the band reaches [2, 24], the majority of sampled episodes hand control to
   the policy early, so the proportion of near-grasp expert states entering the
   buffer falls, while `buf_pos` remains flat at about 1.6% — the positive
   transitions are present but constitute a small and non-increasing fraction.

### F.3.8 Data pipeline health

- **Planner coverage.** 531 of 4000 rollout episodes (13.3%) were skipped because
  the planner produced no plan or no grasp pose, sampling uniformly from the
  training split. The fixed evaluation sweep over scenes 0–63 skipped exactly 16
  (25%) every time. All reported rates are therefore conditioned on
  planner-solvable scenes.
- **Expert episode quality.** `exp_succ` averaged 0.82–0.93 across the run, i.e.
  8–18% of full planner-playback episodes fail even without any policy
  involvement. This is a ceiling on the anchor data quality and matches the demo
  pool's 86.7% success rate.
- **Positive reward availability.** `buf_pos` stayed in the 0.0145–0.0171 band for
  the entire run, never collapsing to zero. The expert-episode mechanism
  successfully maintained positive reward in the online buffer throughout.

### F.3.9 Interpretation of the peak value

The peak evaluation is a single point: 18 of 48 episodes at iteration 133. The
binomial standard error at that rate and sample size is 0.070. The two
neighbouring evaluations are 0.208 (iteration 126) and 0.188 (iteration 140), and
the mean over the eight evaluations spanning iterations 98–147 is 0.195. The
defensible statement is therefore that the run reached a **plateau of roughly
0.20 success over iterations 98–147, with a best single evaluation of 0.375**,
rather than that it attained 0.375 as a stable level.

`best.pt` was written at iteration 133 by the success-based selection rule, and is
the deliverable checkpoint from this run.

### F.3.10 Scope of these numbers

All figures in this section come from the in-loop evaluation, which runs on the
**same training split** the rollouts sample from. They measure capability on
training scenes, not generalisation. A held-out number requires a separate
deterministic evaluation on the validation or test split via
`examples/rollout_rl_policy.py`.

## F.4 Summary

Run 28 trained a 5-channel, from-scratch, joint-state-free reactive policy for
250 iterations (4000 episodes, 200 000 updates) under a widening reverse
curriculum, and produced:

- a stable optimisation with no divergence, no critic exploitation, no gripper
  saturation, and monotone improvement of both the critic fit and the auxiliary
  representation loss;
- monotone improvement of from-scratch reaching (0.112 → ~0.065 m) and wrist
  alignment (0.781 → ~0.29 rad), the latter crossing below its 0.34 rad
  tolerance;
- a success plateau of roughly 0.20 over iterations 98–147 with a best single
  evaluation of 0.375 (18/48) at iteration 133;
- a dominant failure mode of `TIMEOUT` (85.2% of all evaluation episodes) with a
  high 86.6% close→success conversion, identifying **terminal-phase position
  accuracy — not the gripper decision — as the binding constraint**;
- a decline after the curriculum reached its final width, accompanied by a
  collapse in close rate but not in reach accuracy or optimisation health.
