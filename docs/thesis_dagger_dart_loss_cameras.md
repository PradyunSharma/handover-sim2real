# DAgger, DART, Loss Function, and Camera Configuration

Reference config: `examples/configs/rl_phase1_cluster_r28.yaml` (run 28).
Code locations are given per subsection.

---

# Part A — DAgger

## A.1 What problem it solves

Pure imitation from planner trajectories only ever labels states the *planner*
visits. In closed loop the learned policy drifts off that distribution and has no
label for the states it actually reaches — the standard covariate-shift failure.
DAgger (Ross et al.) fixes this by driving the simulator with the *policy's* own
actions and querying the expert for the action it *would* have taken at each
visited state.

In this project DAgger appears in two places. The Phase-3 online form (A.2–A.4)
is the one used during RL training. The offline collector (A.5) belongs to the
behaviour-cloning pipeline and is listed for completeness.

## A.2 Phase-3 online DAgger — implementation

File: `handover_sim2real/rl/rollout_worker.py`, method
`RolloutWorker.rollout_episode()`.

There is no separate DAgger data-collection pass. Every online rollout is
simultaneously an RL episode and a DAgger episode: the environment is stepped
with the policy's action, and an expert label is attached to the same transition.

**Label form: plan-tracking.** The planner is invoked **once**, at step 0, with
horizon `rollout_max_steps` (30). This produces a committed joint-space
trajectory `full_plan`. At every step $t$, the label is the end-effector delta
from the **current, possibly drifted** state to `full_plan[min(t, len-1)]`:

```python
idx           = min(step, len(full_plan) - 1)
expert_delta  = env.convert_target_joint_position_to_action(full_plan[idx])   # [6], metres/rad
expert_action = normalize(expert_delta)
expert_flag   = 1.0
```

Because the label is computed from the *current* state to a *time-indexed*
waypoint, it is corrective: if the policy lags behind, the label grows; if the
policy drifts laterally, the label points back onto the plan. The target waypoint
advances into the grasp as the episode progresses, so late in the episode the
label demonstrates the standoff-to-grasp descent.

**Every step is labelled.** GA-DDPG masks the imitation loss on non-DAgger
explore steps; here every step carries `expert_flag = 1`. This is a deliberate
deviation: with the clock in the state, a time-indexed label is self-consistent,
and the drifted-state pursuit labels are precisely the on-policy corrective
signal that offline behaviour cloning never provides.

**Rejected alternative (documented because it caused a measurable failure).** An
earlier version replanned from scratch at every step with a short horizon and
used `plan[0]` as the label. That label has a stationary attractor at the
planner's standoff, roughly 8 cm short of the grasp: the first waypoint never
enters the final reach segment, and its magnitude Zeno-decays as the standoff is
approached. The resulting policy hovered 6–11 cm short of the grasp and never
closed. Switching to plan-tracking labels removed the hover.

## A.3 Tail replanning (the "aggregation" step)

A single step-0 plan goes stale if the policy drifts far. The fix is a gated
partial replan that splices a new tail onto the committed plan:

```python
elif (dagger_ratio > 0.0
      and step > dagger_min_step
      and step < len(full_plan) - dagger_tail_guard
      and rng.uniform() < dagger_ratio):
    rest, _ = env.run_omg_planner(max_steps - step, scene_idx, reset_scene=False)
    if rest is None:
        n_omg_fail += 1                     # keep the old plan
    else:
        full_plan = concatenate([full_plan[:step], rest])
        g = env.get_omg_goal_grasp_pose()
        if g is not None:
            grasp_pose = g                  # replan may re-select the goal grasp
```

Run-28 parameters:

| Key | Value | Role |
|---|---|---|
| `dagger_ratio` | 0.5 | per-step replan probability |
| `dagger_min_step` | 5 | no replans in the first 5 steps (initial plan still fresh) |
| `dagger_tail_guard` | 8 | no replans within the last 8 steps of the plan |

Three properties of this design:

1. **Replans from the current joint configuration.** `run_omg_planner` is called
   with `reset_scene=False`, so the planner starts from `panda.body.dof_state` —
   the drifted state — which is what makes it a DAgger query rather than a
   re-derivation of the original plan.
2. **Horizon is the remaining episode budget** (`max_steps - step`), not a fixed
   short horizon. A floored short horizon reintroduces the Zeno decay: the
   per-step label magnitude must stay at least (remaining distance)/(remaining
   steps).
3. **The tail guard is load-bearing.** Replanning inside the final approach would
   overwrite the standoff-to-grasp labels with a fresh standoff approach, undoing
   exactly the labels that teach the endgame. The guard also guarantees each
   replan has enough budget for the planner's structure (free portion plus a
   ~5-step reach tail).

If the goal grasp is re-selected by a replan, `grasp_pose` is updated so the
reward proximity check, the gripper label, and the auxiliary target all remain
consistent with the labels.

## A.4 β mixing (expert action execution)

Separate from labelling: with per-step probability β the *executed* action is the
plan waypoint instead of the policy's action.

```python
warmup     = step < expert_initial_steps
use_expert = warmup or (beta > 0.0 and rng.uniform() < beta)
if use_expert:
    target_jp     = full_plan[idx]                       # follow the plan BY INDEX
    stored_action = concat([expert_action_norm, [act_limit]])   # gripper "open" logit
```

Notes:

- The stored action for an expert step is the normalised expert delta plus a
  saturated open-gripper logit (`+act_limit`), so the critic sees a valid
  in-distribution action for that transition.
- Expert steps follow the committed plan **by index**, not `plan[0]` with
  replanning — the same anti-Zeno reasoning as above.
- β is annealed `beta_start: 0.5 → beta_end: 0.0` over `beta_ramp_iters: 63`.
- The reverse-curriculum warm start (`expert_initial_steps`) uses the same code
  path, forcing expert control for the first `ei` steps unconditionally.
- Labelling is independent of β: a step executed by the expert is still labelled,
  and a step executed by the policy is still labelled.

## A.5 Offline DAgger collector (behaviour-cloning pipeline)

Files: `examples/collect_dagger_dataset.py`, `examples/collect_dagger_act_dataset.py`,
driver scripts `examples/run_dagger.sh`, `examples/run_dagger_act.sh`.

Roll out a trained BC policy, query the planner at every visited state, and write
`(state, expert action)` pairs in the *same HDF5 schema* as the original BC
dataset so `train_bc.py --dagger-h5` can aggregate them without conversion.
Two defaults shape the labels:

- `--drop-past-standoff`: recording stops at the standoff plane, so the final
  reach and the close come from demonstrations. Without it, states the policy
  overshoots past the standoff receive backward "retreat" labels.
- `--dynamic-horizon`: the replan length is chosen from the EE-to-standoff
  distance (~`--ee-step` metres per step), so label magnitudes stay at the
  demonstration scale instead of collapsing into single huge jumps late in the
  episode.

This collector is a separate offline pass and is not used by Phase-3 training,
which generates its DAgger labels inline.

---

# Part B — DART

## B.1 Motivation

Both the reverse-curriculum warm start and DAgger tail replanning keep the
end-effector essentially **on** the plan. Consequently the training data contains
almost no examples of recovering from a laterally displaced pose near the grasp —
which is exactly the state a from-scratch policy arrives in, since it accumulates
a few centimetres of error over the approach. DART (Laskey et al.; GA-DDPG's
`env.random_perturb`) manufactures that coverage by injecting noise into the
data-collection process and letting the expert demonstrate the recovery.

## B.2 Mode 1 — policy-side DART

File: `rollout_worker.py`, `rollout_episode()`. Active when
`dart_mode ∈ {policy, both}`.

On a policy step inside the window, with probability `dart_ratio`, the policy's
action is **replaced** by a random task-space jump:

```python
elif (dart_ratio > 0.0
      and dart_min_step <= step < dart_max_step
      and rng.uniform() < dart_ratio):
    is_dart = True
    exec_delta6 = concat([rng.uniform(-dart_pos_mag, dart_pos_mag, 3),
                          rng.uniform(-dart_rot_mag, dart_rot_mag, 3)])
    committed_close = False                      # gripper stays OPEN on a jolt
    action7   = concat([exec_delta6, [1.0]])
    target_jp = action_to_target_joint(action7, obs)
    stored_action = concat([normalize(exec_delta6), [act_limit]])
    # transition is written with perturb_flag = 1.0
```

The jump lands the end-effector off-plan; the **following** steps' plan-tracking
labels then demonstrate the recovery back onto the plan.

**Critic masking.** The perturbed transition is stored with `perturb_flag = 1`.
In `td3bc_trainer.update()` those rows are excluded from the Bellman fit:

```python
keep  = (perturb_flag < 0.5).float()
denom = keep.sum().clamp_min(1.0)
q_fit = ((SmoothL1(q1, y, reduction="none") + SmoothL1(q2, y, reduction="none")) * keep).sum() / denom
```

Reason: the stored action is an artificial random jump, so $Q(s,a) \to r + \gamma
Q(s')$ is a meaningless regression target there. The row is still used for the
actor's imitation term, which is the whole point of generating it. When nothing
is perturbed, `keep` is all ones and the masked mean equals the plain mean, so
this is a no-op for DART-off runs and for demo batches.

## B.3 Mode 2 — expert-side DART (GA-DDPG faithful)

File: `rollout_worker.py`, `expert_rollout_episode()`. Active when
`dart_mode ∈ {expert, both}`.

Here the jolt is applied during a full-expert episode and is **out of band** — it
is stepped through the simulator but *not recorded as a transition*:

1. Sample a random task-space jump inside the window and step the simulator with
   it (gripper open).
2. If the jolt terminated the episode (e.g. collision), abort the episode.
3. Replan the tail from the perturbed state (`run_omg_planner(max_steps - step,
   reset_scene=False)`), splice it onto the plan, update the goal grasp if
   re-selected. If replanning fails, keep the old plan and let the expert recover
   toward it.
4. Rebuild the observation, set `prev_act` to the jolt, and continue executing the
   (new) plan by index.

Every **recorded** step is therefore the expert's clean, dynamically consistent
correction from an off-plan state. No critic masking is needed, which is the
advantage over mode 1.

`dart_ratio` is passed as 0 when collecting the permanent demo pool
(`collect_rl_demos.py`), so that fixed +1 anchor stays unperturbed.

## B.4 Parameters and rationale

| Key | Run-28 value | Meaning |
|---|---|---|
| `dart_mode` | `policy` | `policy` / `expert` / `both` |
| `dart_ratio` | 0.2 | per-step perturbation probability (0 = disabled) |
| `dart_min_step` | 15 | window start |
| `dart_max_step` | 22 | window end (exclusive) |
| `dart_pos_mag` | 0.04 | ± metres, uniform per axis |
| `dart_rot_mag` | 0.2 | ± radians, uniform per axis |

Window placement: with a 30-step horizon and a ~5-step reach tail, `[15, 22)`
sits just **before** the final approach and **inside** the DAgger window. The
intent is: perturb around the standoff entry, let a tail replan produce a
feasible recovery, realign, then descend straight in. Perturbing inside the reach
tail would leave no steps to recover; perturbing early would only be re-absorbed
by the long free portion of the plan and teach nothing about the endgame.

The gripper is always commanded open during a jolt — a perturbation must never
be mistaken for a grasp commitment.

Ablations run: `dart_mode: both` with `dart_ratio: 0.5` (run 24) versus
`dart_mode: policy` with `dart_ratio: 0.2` (run 23, 28).

---

# Part C — Loss function

File: `handover_sim2real/rl/td3bc_trainer.py`, method `TD3BCTrainer.update()`.
Point-matching helper: `handover_sim2real/bc/losses.py`, `pose_pm_loss()`.

Notation: $s = (P, s^r, c)$ is the state, $a$ the stored 7-D action, $r$ the
sparse terminal reward, $d$ the terminal flag, $G$ the discounted Monte-Carlo
return, $a^\ast$ the 6-D expert pose label, $u \in \{0,1\}$ the gripper label
(1 = open), $g \in \mathbb{R}^9$ the EE-relative grasp pose.

## C.1 Critic loss

**Target action** (TD3 target policy smoothing):

$$a' = \mathrm{clamp}\big(\pi_{\bar\theta}(s') + \mathrm{clamp}(\epsilon, \pm c_\text{clip}),\ \pm A\big), \qquad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

with `policy_noise` $\sigma = 0.2$, `noise_clip` $c_\text{clip} = 0.5$,
`act_limit` $A = 5.0$.

**Optional potential-based shaping** (`shaping_pos_weight`,
`shaping_rot_weight`; both 0 in run 28, i.e. off):

$$\Phi(s) = -\big(w_p \lVert t_\text{rel} \rVert + w_r \,\theta_\text{rel}\big), \qquad F = \gamma \Phi(s')(1-d) - \Phi(s)$$

where $t_\text{rel}$ and $\theta_\text{rel}$ are the position norm and geodesic
rotation angle decoded from the 9-D `goal_pose` (Gram–Schmidt on the two rotation
columns, then $\theta = \arccos((\mathrm{tr}\,R - 1)/2)$). This is Ng et al.'s
policy-invariant shaping: it telescopes to an endpoint constant, so it cannot be
farmed by hovering, and the sparse +1 remains optimal. It enters the Bellman
target only — `mc_return` stays the unshaped sparse return.

**Target value**, blending one-step Bellman with the Monte-Carlo return:

$$y_\text{bell} = r + F + \gamma (1-d)\min_i Q_{\bar\phi_i}(s', a'), \qquad y = (1-\beta_\text{mc})\, y_\text{bell} + \beta_\text{mc}\, G$$

with $\gamma = 0.95$ and `mc_blend` $\beta_\text{mc} = 0.5$. The MC term
propagates the rare sparse successes back through an episode faster than
one-step bootstrapping alone.

**Loss**, with the DART mask $m = \mathbb{1}[\text{perturb\_flag} < 0.5]$:

$$\mathcal{L}_\text{critic} = \frac{\sum_j m_j \big[\mathrm{SmoothL1}(Q_1, y)_j + \mathrm{SmoothL1}(Q_2, y)_j\big]}{\max(\sum_j m_j, 1)} \;+\; w_\text{aux}\,\mathrm{SmoothL1}(\hat g^v, g)$$

with `aux_weight` $w_\text{aux} = 0.5$.

## C.2 Actor loss

Applied every `policy_delay` = 2 updates.

**Gripper detachment before the critic.** The 7-D policy output is split, and the
gripper channel is detached:

```python
a_for_q = torch.cat([a_pi[:, :6], a_pi[:, 6:7].detach()], dim=1)
q1_pi   = critic.q1_only(pc, rs, a_for_q, remain)
```

so the policy gradient flows only through the six pose channels. Reason: the
gripper output is an unbounded, near-binary logit; letting the policy gradient
ride $\partial Q/\partial \ell$ drove it to $\sim 5\times10^4$ and pushed the critic into
an out-of-distribution region ($Q(s,\pi(s))$ spiked to ~950, then both collapsed
and the close rate stayed at 0). No signal is lost, because the gripper's
supervised label already encodes the reward-earning behaviour.

**Policy-gradient term**, two mutually exclusive forms:

$$\lambda = \begin{cases}\dfrac{\alpha}{\overline{|Q_1(s,\pi(s))|}} & \texttt{pg\_normalize: true} \ (\alpha = 0.1),\ \ \text{bc\_scale} = 1\\[2ex] \lambda_\text{start} + (\lambda_\text{end}-\lambda_\text{start})\min\!\big(\tfrac{k}{\texttt{mix\_ramp}},1\big) & \texttt{pg\_normalize: false},\ \ \text{bc\_scale} = 1-\lambda\end{cases}$$

$$\mathcal{L}_\text{pg} = -\big(\lambda \cdot Q_1(s, \pi(s))\big)$$

The denominator in the normalised form is detached. Run 28 uses the normalised
form; the fixed-schedule form (`mix_start: 0.1 → mix_end: 0.2` over 50 000
updates) diverged repeatedly.

**Pose imitation term**, masked by `expert_flag`, two selectable metrics:

- `pose_loss: smooth_l1` (run 28) — SmoothL1 on the raw normalised 6-D vector:
  $\mathrm{SmoothL1}(\pi(s)_{0:6},\, a^\ast)$. Huber behaviour gives L2 near zero
  and L1 in the tail, so a few outlier planner steps do not dominate.
- `pose_loss: pm` — point matching in **real units**. Both prediction and target
  are denormalised to metres/radians, six fixed Panda gripper control points (two
  TCP, two knuckle, two fingertip) are transformed by each pose, and the L1
  displacement is summed over coordinates and averaged over the batch:

  $$\mathcal{L}_\text{pm} = \frac{1}{B}\sum_b \sum_{k=1}^{6}\big\lVert (R_b p_k + t_b) - (R^\ast_b p_k + t^\ast_b)\big\rVert_1$$

  with $R = R_z R_y R_x$ from the Euler channels. Motivation: SmoothL1 on
  $[\Delta\text{pos (m)} \Vert \Delta\text{euler (rad)}]$ sums incommensurable
  units and scores rotation with a poor metric; the point-matching form is a
  single physically meaningful SE(3) distance (metres of gripper-point motion) in
  which orientation is weighted by its actual effect on the gripper. It targets
  the observed rotation plateau (~0.4 rad closest approach against a 0.34 rad
  close threshold).

**Gripper term** — class-balanced, label-smoothed BCE with logits, masked by
`gripper_flag`:

```python
tgt      = expert_gripper[gmask]                 # P(open): 1 far, 0 near
is_close = tgt < 0.5
w_close  = min(n_open / n_close, gripper_close_weight_max)   # 10.0
w        = where(is_close, w_close, 1.0)
tgt      = tgt * (1 - 2*eps) + eps               # gripper_label_smooth = 0.1
grip_loss = BCEWithLogits(a_pi[gmask][:, 6], tgt, weight=w)
```

Two corrections are applied because "far/open" overwhelmingly dominates every
batch:

- **Class balancing** raises the total mass of close examples to roughly match
  the open mass, capped at 10×. Without it the classifier learns only "open" and
  the gripper never fires.
- **Label smoothing** ($\epsilon = 0.1$) bounds the converged logit. With hard
  targets, the dominant open label drives the logit toward $+\infty$ until the
  sigmoid saturates and the gradient vanishes, leaving the gripper permanently
  stuck open — the observed late-run decline where the closest approach stayed
  healthy but the close rate fell to ~4%.

Note the label agrees with the reward (close near the grasp earns +1), so this
term is a dense accelerant rather than a competing objective. It is training-only
supervision derived from the privileged grasp pose; at deployment the policy
executes its own logit and nothing consumes the distance.

**Action magnitude regulariser** (`action_reg_weight`, 0 in run 28):

$$\mathcal{L}_\text{reg} = w_\text{reg}\,\overline{\pi(s)_{0:6}^2}$$

Quadratic, so it self-targets the large drifted actions and is gentle in the
healthy regime. Motivated by the mean action magnitude inflating from ~0.7 to
~2.0 late in every run, which correlates with overshooting and aggressive closes
that collide or drop.

**Total:**

$$\mathcal{L}_\text{actor} = \mathcal{L}_\text{pg} + s_\text{bc}\, w_\text{bc}\, \mathcal{L}_\text{bc} + s_\text{bc}\, w_\text{grip}\, \mathcal{L}_\text{grip} + w_\text{aux}\, \mathrm{SmoothL1}(\hat g^\pi, g) + \mathcal{L}_\text{reg}$$

with `bc_weight` $w_\text{bc} = 2.0$, `gripper_bc_weight` $w_\text{grip} = 1.0$,
`aux_weight` $w_\text{aux} = 0.5$, and $s_\text{bc} = 1$ under `pg_normalize`.

## C.3 Auxiliary term

Present in **both** losses: `SmoothL1(aux_head(scene_features), goal_pose)`,
where `goal_pose` is the final grasp pose relative to the current end-effector,
encoded as position (3) + the first two rotation-matrix columns (6). It is a pure
regulariser — the prediction is never used for control or value estimation — that
gives the PointNet++ encoders a dense geometric signal under an otherwise sparse
terminal reward. Set `aux_weight: 0` to disable.

## C.4 Optimisation

| Item | Value |
|---|---|
| Optimiser | Adam, actor and critic separately |
| Learning rates | `actor_lr` = `critic_lr` = $3\times10^{-4}$ |
| Gradient clipping | `grad_clip: 1.0`, global norm, applied separately to each network |
| Target update | Polyak, `tau: 0.005`, on both actor and critic targets after each delayed actor step; buffers (BatchNorm statistics) are copied directly, not interpolated |
| Update ordering | critic step first, then (every 2nd update) actor step, then both soft updates |

The shared global gradient clip is why the policy-gradient/imitation balance
matters so much: an unbounded policy-gradient term consumes the whole clip budget
and starves the imitation gradient, whose SmoothL1 gradient saturates at ±1.

## C.5 Loss-related diagnostics logged per iteration

`critic_loss`, `q_mean` (Q on stored actions), `target_mean`, `actor_loss`,
`pg_loss`, `bc_loss`, `grip_loss`, `action_reg`, `aux_loss_a`, `aux_loss_c`,
`lam`, `n_expert`, `n_gripper`, `q_pi` (Q on the policy's own action),
`a_absmean`, `grip_logit`. The `q_pi` − `q_mean` gap is the out-of-distribution
exploitation warning; `grip_logit` drifting positive is the gripper-drift
warning; `a_absmean` is the action-inflation warning.

---

# Part D — Camera configuration

Config block: `ENV.HANDOVER_HAND_CAMERA_POINT_STATE_ENV` in
`handover-sim/handover/config.py`. Implementation:
`handover-sim/handover/handover_env.py` (`HandoverHandCameraPointStateEnv._get_point_states`),
`handover-sim/handover/multicam.py` (`FixedSegCamera`),
`handover-sim/handover/panda.py` (`PandaHandCamera`).
Point-cloud assembly: `handover_sim2real/policy.py` (`PointListener`).

## D.1 Selectable cameras

`CAMERAS` is a list drawn from `{"wrist", "left", "right", "back"}`. Default
`["wrist"]` reproduces the original egocentric behaviour byte-for-byte.

| Name | Type | Mount / pose | Notes |
|---|---|---|---|
| `wrist` | eye-in-hand, moves with the arm | Panda link 11, offset (+0.036, 0, +0.036) from the hand link | close-up geometry at the grasp point; **loses the object as the gripper closes in** |
| `left` | fixed external | eye (1.45, 0.25, 1.55) → target (0.52, 0.25, 1.23) | elevated side view, +x side |
| `right` | fixed external | eye (−0.40, 0.25, 1.55) → target (0.52, 0.25, 1.23) | elevated side view, −x side, symmetric to `left` |
| `back` | fixed external | eye (0.52, −0.70, 1.70) → target (0.52, 0.25, 1.23) | behind the robot |

The fixed targets are the measured mean DexYCB handover object position
(0.52, 0.25, 1.23), standard deviation roughly (0.08, 0.11, 0.06) across scenes.
`left` and `right` are placed orthogonal to the robot→human +Y approach direction
so the arm does not self-occlude the object, roughly 1 m out and 0.3 m up,
symmetric about $x = 0.52$.

## D.2 Intrinsics

| Parameter | Wrist | External (`left`/`right`/`back`) |
|---|---|---|
| Resolution | 224 × 224 | `EXTERNAL_CAMERA_WIDTH/HEIGHT`, 224 × 224 |
| Vertical FOV | 90° | `EXTERNAL_CAMERA_VERTICAL_FOV`, 90° |
| Near / far | 0.035 / 2.0 m | `EXTERNAL_CAMERA_NEAR/FAR`, 0.05 / 3.0 m |
| Up vector | — (follows the link) | `EXTERNAL_CAMERA_UP`, (0, 0, 1) |

External camera poses are overridable per run via `CAMERA_LEFT_POSITION`,
`CAMERA_LEFT_TARGET`, and the `RIGHT`/`BACK` equivalents.

External cameras use a **look-at** setup (position + target + up), which is what
`computeViewMatrix` expects; an earlier orientation-quaternion formulation aimed
the cameras at the floor. Deprojection reconstructs each pixel's view-space ray
(OpenGL convention: the camera looks along −Z, +Y up, +X right) and maps it to
world with the camera→world rotation. Verified live: the reconstructed object
cloud lands within about 4 cm of the true object pose.

## D.3 Semantic classes

`COMPUTE_MANO_POINT_STATE` and `COMPUTE_ROBOT_POINT_STATE` add classes. Class
order is fixed:

$$[\text{object},\ \text{hand}\ (\text{if COMPUTE\_MANO}),\ \text{robot}\ (\text{if COMPUTE\_ROBOT})]$$

Segmentation is by body id: the YCB object body, the MANO hand body, and the
whole Panda body. A class whose segmentation id is unavailable (e.g. the hand has
left the scene) is returned as an **empty array** rather than being dropped, so
the number of classes is constant and always matches `PointListener`'s merge
ratios.

The point-cloud channel count follows directly:

$$C = 3 + (\text{number of classes})$$

| Classes | $C$ | `DATA.pc_channels` |
|---|---|---|
| object, hand | 5 | 5 |
| object, hand, robot | 6 | 6 |

Channel 0–2 are $xyz$; channels 3.. are per-class one-hots (`point_state_[3 + i] = 1`).

## D.4 Fusion

Two code paths in `_get_point_states()`:

**Wrist-only** (`CAMERAS: ["wrist"]`, no fixed cameras). The Panda's hand camera
renders and deprojects directly into the hand frame. This path is byte-identical
to the original implementation when `COMPUTE_ROBOT_POINT_STATE` is off.

**Multi-camera** (any fixed camera selected). Each fixed camera returns
world-frame points per class; they are transformed into the **Panda hand-link
frame** — the same frame the wrist camera natively outputs — and concatenated per
class. If `wrist` is also selected, its points (already in the hand frame) are
appended to the same per-class lists. So fixed and wrist points are
indistinguishable downstream, and `PointListener` treats them identically.

`PointListener.point_states_to_state()` then merges classes to exactly
`num_pts` = 1024 points:

- **2-class path** (unchanged, egocentric): each class is truncated to
  $\lfloor \text{num\_pts} \times \text{ratio} \rfloor$ using
  `POINT_STATE_YCB_RATIO` = 0.875 for the object and the remainder for the hand.
  If the hand class is empty, the whole cloud falls back to the object class.
- **>2-class path**: each class is first regularised to `num_pts`, then its ratio
  share is taken, empty classes are skipped (rather than triggering a
  drop-everything fallback), the survivors are concatenated, and the result is
  resampled to exactly `num_pts` so the replay buffer's fixed $[1024, C]$ slot
  always fits. Ratios come from `POLICY.POINT_STATE_RATIOS`, e.g.
  `[0.7, 0.15, 0.15]` for object/hand/robot.

Finally the merged cloud is expressed in the current end-effector frame
(`se3_transform_pc(inv_ee_pose, ...)`). The cloud is **single-frame**: the
`acc_points` buffer is overwritten each step, not accumulated.

## D.5 Provided configurations and consistency requirements

| Sim config | `CAMERAS` | Classes | $C$ | Used by |
|---|---|---|---|---|
| `examples/pretrain.yaml` | `["wrist"]` (default) | object, hand | 5 | runs up to 32 |
| `examples/pretrain_multicam.yaml` | `["left", "right"]` | object, hand, robot | 6 | run 33 |
| `examples/pretrain_multicam_wl.yaml` | `["wrist", "left"]` | object, hand, robot | 6 | run 37 |

Constraints when changing the camera set:

1. `DATA.pc_channels` in the RL config **must** equal $3 + $ number of classes.
   The network and the replay buffer both read it from that block.
2. The demonstration pool must be re-collected with the same sim config — the
   cloud pipeline is shared, and a 5-channel pool cannot be mixed with 6-channel
   online data.
3. `warm_start` must be false when the channel count changes: the observation
   distribution changes and the first set-abstraction convolution has a different
   input width, so behaviour-cloning weights cannot transfer. The behaviour-cloning
   run then contributes only the camera-independent normaliser and network dims.
4. The renderer choice (EGL vs CPU rasteriser) should be identical across demo
   collection, training and evaluation. The depth-derived cloud is nearly
   identical either way, but consistency avoids a silent distribution shift — and
   EGL should be avoided anyway because of the per-scene GPU memory leak.

Placement tools: `examples/viz_external_cameras.py` (renders the external camera
views and frustums) and `examples/viz_cameras_pybullet.py` (draws camera frames
in the PyBullet GUI). Re-check placement with these after editing any pose.

## D.6 What the camera choice is testing

The wrist camera gives high-resolution egocentric geometry exactly where the
grasp happens, but it loses sight of the object during the final approach, which
is precisely the phase where the policy plateaus. Fixed external cameras see the
whole handover — object, hand and arm — throughout, but at lower effective
resolution near the grasp and with the arm potentially occluding. Run 33
(`left` + `right`) and run 37 (`wrist` + `left`) are a controlled A/B on this
trade-off: the two configs are byte-identical apart from the camera set, both
produce 3-class 6-channel clouds, and both train from scratch.
