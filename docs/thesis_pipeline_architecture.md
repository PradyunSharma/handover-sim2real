# Pipeline Architecture

This chapter defines the architecture of the Phase-3 policy: the reactive,
single-frame handover policy that is trained online with a TD3+BC actor–critic
blend. The description covers (i) the closed-loop control pipeline that connects
the simulator to the network, (ii) the observation and action interfaces, (iii)
the actor and critic networks and their auxiliary heads, and (iv) the training
architecture that surrounds them. The temporal (ACT) variant of the actor is
treated separately and is not described here.

## 1. Overview

The policy operates as a closed feedback loop at a fixed control rate. At every
policy step the simulator produces a segmented point cloud and a proprioceptive
state vector; the network maps these to a 6-DoF end-effector displacement and a
binary gripper command; the displacement is converted to a joint-space target by
inverse kinematics and held for a fixed number of simulator substeps. No motion
plan, object pose, or grasp pose is available to the policy at inference time —
all privileged quantities are used for supervision only.

```
                       ┌──────────────────────── environment ────────────────────────┐
                       │  handover-sim (PyBullet)                                    │
                       │  MANO hand playback  ·  YCB object  ·  Franka Panda          │
                       └───────┬─────────────────────────────────────────▲───────────┘
       depth + segmentation    │                                         │  q_target ∈ R^9
       from configured cameras │                                         │  (150 sim substeps)
                               ▼                                         │
        ┌──────────────────────────────────────┐              ┌──────────┴───────────┐
        │ PointListener                        │              │ IK  (pybullet)       │
        │  · per-class subsample → N = 1024    │              │  T_ee ← T_ee · Δ(a)  │
        │  · transform into EE (hand) frame    │              │  fingers ← gripper   │
        └──────────────┬───────────────────────┘              └──────────▲───────────┘
                       │ P ∈ R^{N×C}                                     │ a ∈ R^7
                       │                                                 │ (denormalised)
                       ▼                                                 │
        ┌──────────────────────────────────────────────────────────────────────────┐
        │                          π_θ  (RLActor)                                  │
        │   P ──► PointNet++ encoder ──► z_scene ∈ R^256 ──┐                        │
        │   s_r ─► robot MLP        ──► z_robot ∈ R^256 ──┼─► [·‖·‖c] ─► head ─► a │
        │   c  = remaining-steps clock ∈ (0,1] ───────────┘        (513→256→256→7) │
        │                        └──► aux head ──► ĝ ∈ R^9   (training only)        │
        └──────────────────────────────────────────────────────────────────────────┘
```

The same observation builders (`_point_cloud`, `_robot_state`) are shared by the
behaviour-cloning data collector, the online rollout worker and the evaluation
script, so the state representation is bit-identical across pre-training, online
training and evaluation.

## 2. Observation space

The policy state is the triple $s_t = (P_t,\; s^{r}_t,\; c_t)$.

**Segmented point cloud $P_t \in \mathbb{R}^{N \times C}$.** Depth images from
the configured camera set are back-projected, segmented into semantic classes,
subsampled to a fixed budget of $N = 1024$ points with fixed per-class ratios,
and expressed in the *current end-effector (hand) frame*. Because the cloud is
egocentric and re-expressed every step, it encodes the relative geometry between
gripper, object and human hand directly, which is what makes the auxiliary goal
regression of Section 5 learnable. The cloud is a **single frame**: it is not
accumulated over time, so all temporal information available to the policy comes
from the proprioceptive state and the clock.

The channel layout depends on the camera configuration:

| Configuration | $C$ | Channels | Class ratios |
|---|---|---|---|
| Wrist-only, two classes | 5 | $xyz$ + object one-hot + hand one-hot | — |
| Multi-camera, three classes | 6 | $xyz$ + object + hand + robot one-hots | $0.70 / 0.15 / 0.15$ |

The three-class variant fuses a dynamic wrist (eye-in-hand) view with one or
more fixed side cameras; fusion happens per class in the hand frame, so the
network input shape is unchanged apart from the extra one-hot channel.

**Robot state $s^{r}_t \in \mathbb{R}^{32}$.** A concatenation of joint
positions (9), joint velocities (9), end-effector position (3), end-effector
orientation as a $wxyz$ quaternion (4), the normalised finger opening (1), and
the previously executed 6-DoF displacement (6). The network consumes a
*configurable slice* of this vector:

| Flag | Effect | Effective input dim |
|---|---|---|
| — | full vector | 32 |
| `use_prev_act = false` (default) | drop the trailing previous action | 26 |
| `+ drop_joint_state = true` | additionally drop joint positions and velocities | 8 |

Dropping the previous action removes a channel that is $\approx 0.9$ correlated
with the target displacement and therefore invites causal confusion (the policy
copies its own last action and ignores the cloud). Dropping joint state is
motivated by the action being defined in the end-effector frame: the
end-effector pose already carries the task-relevant kinematics, while joint-space
values are scene-correlated and redundant. Because both flags change the width of
the robot encoder's first layer, they are only compatible with from-scratch
training (Section 6). Robot states are normalised per channel by a `Normalizer`
whose statistics are estimated on the behaviour-cloning dataset; the point cloud
is fed raw.

**Clock $c_t \in (0,1]$.** With a fixed episode horizon $T$, the remaining-step
fraction

$$c_t \;=\; \frac{T - t}{T}$$

is supplied to *both* the actor and the critic. Without it the same state would
be simultaneously terminal (at the step limit) and non-terminal, giving the value
function contradictory bootstrap targets; with it, horizon truncation becomes a
genuine terminal and a single `terminal` flag suffices. Unlike GA-DDPG, which
appends the clock to the proprioceptive vector, the clock here is injected at the
**fused-feature level** — concatenated to $[z_\text{scene} \Vert z_\text{robot}]$
immediately before the policy head. This keeps both encoders shape-identical to
the behaviour-cloning network and makes the warm-start of Section 6 an exact 1:1
weight transfer.

## 3. Action space and execution

The network emits a 7-dimensional vector

$$a_t \;=\; [\underbrace{\Delta p_x, \Delta p_y, \Delta p_z, \Delta r_x, \Delta r_y, \Delta r_z}_{\text{normalised 6-DoF displacement}},\; \underbrace{\ell}_{\text{gripper logit}}]$$

with no output nonlinearity. The first six channels are a translation and an
Euler-angle rotation of the end-effector frame, expressed in *normalised* action
units (zero mean, unit variance per channel, statistics taken from the
behaviour-cloning dataset); they are clamped to $[-A, A]$ with $A = 5$ and
denormalised only at execution time. The seventh channel is an unbounded logit;
the gripper is commanded **open iff $\ell \ge 0$**, and a commitment to close is
treated as terminal.

Execution proceeds in three stages. The denormalised displacement is composed
with the current end-effector pose, $T^\text{target}_{ee} = T_{ee} \cdot
\Delta(a_{0:6})$; PyBullet's inverse kinematics solves for a 9-DoF joint target;
and the two finger joints are overwritten with $0.04\,\text{m}$ (open) or
$0.0\,\text{m}$ (closed) according to the gripper bit. The resulting joint target
is held for

$$n_\text{repeat} \;=\; \frac{\texttt{POLICY.TIME\_ACTION\_REPEAT}}{\texttt{SIM.TIME\_STEP}} \;=\; \frac{0.15\,\text{s}}{0.001\,\text{s}} \;=\; 150$$

simulator substeps, so one policy step corresponds to $150\,\text{ms}$ of
simulated time and an episode horizon of $T = 30$ steps corresponds to
$4.5\,\text{s}$.

## 4. Actor network

The actor is a single-frame, deterministic policy $\pi_\theta(s_t) \mapsto a_t$
built from two independent encoders and a fusion head.

**Point-cloud encoder.** A PointNet++ backbone consisting of three
set-abstraction (SA) stages followed by a fully-connected stack, reused from
GA-DDPG so that pre-trained encoder weights transfer:

| Stage | Sampling | Radius | Neighbours | MLP widths |
|---|---|---|---|---|
| SA-1 | 32 centroids | 0.02 m | 64 | $C \to 64 \to 64 \to 128$ |
| SA-2 | 32 centroids | 0.04 m | 128 | $128 \to 128 \to 128 \to 256$ |
| SA-3 | global | — | — | $256 \to 256 \to 256 \to 512$ |
| FC | — | — | — | $512 \to 1024 \to 512$ (BatchNorm + ReLU) |

A linear projection maps the 512-dimensional backbone output to the shared
feature width $D = 256$, giving $z_\text{scene} \in \mathbb{R}^{256}$.

**Robot encoder.** A two-layer MLP $d_r \to 128 \to 256$ with ReLU
activations, where $d_r$ is the effective robot-state dimension from Section 2,
giving $z_\text{robot} \in \mathbb{R}^{256}$.

**Policy head.** The fused vector $[z_\text{scene} \Vert z_\text{robot} \Vert
c_t] \in \mathbb{R}^{513}$ is passed through an MLP $513 \to 256 \to 256 \to 7$
with ReLU activations and a linear output layer. The absence of a squashing
nonlinearity is deliberate: it is what allows the warm-started actor to reproduce
the behaviour-cloning policy exactly, and the output range is instead controlled
by clamping at execution and by an optional quadratic magnitude penalty on the
pose channels during training.

## 5. Critic network

The critic implements TD3's clipped double-Q estimator. It instantiates its
**own** point-cloud and robot encoders with the same topology as the actor's but
untied weights, so that value-learning gradients do not reshape the policy's
representation. The action and the clock enter at the fusion point:

$$x \;=\; [\,z^{v}_\text{scene} \,\Vert\, z^{v}_\text{robot} \,\Vert\, a_t \,\Vert\, c_t\,] \;\in\; \mathbb{R}^{520},$$

and two independent heads $Q_1, Q_2: \mathbb{R}^{520} \to 256 \to 256 \to
\mathbb{R}$ are evaluated on it. Each $Q_i$ estimates the discounted future
handover success of executing $a_t$ from $s_t$ with $c_t$ of the horizon
remaining. Only $Q_1$ is used for the deterministic policy-gradient term; the
minimum of the two targets is used for bootstrapping.

## 6. Auxiliary goal head

Both networks carry a small auxiliary head, $256 \to 128 \to 9$, attached to the
scene features. It regresses the **final grasp pose expressed relative to the
current end-effector**, encoded as position (3) plus the first two columns of the
rotation matrix (6-D continuous rotation representation):

$$g_t \;=\; \big[\,\mathbf{t}_\text{rel},\; \mathbf{R}_{\text{rel},\cdot 1},\; \mathbf{R}_{\text{rel},\cdot 2}\,\big], \qquad T_\text{rel} = T_{ee}^{-1} T_\text{grasp}.$$

The target is available only in simulation (it comes from the motion planner's
selected grasp), so the head is a **pure training-time regulariser**: its output
is never consumed for control or for value estimation. Its purpose is to give the
PointNet++ encoders a dense, geometrically meaningful learning signal in a regime
where the task reward is sparse and terminal. The representation is frame
invariant and matches the end-effector frame of the input cloud, so the quantity
is a function of the visible geometry alone. Setting the auxiliary weight to zero
disables it without any other change to the architecture.

## 7. Initialisation

Two initialisation modes are supported.

**Warm start from behaviour cloning.** The actor's encoders are loaded 1:1 from
the Phase-1 behaviour-cloning policy, and its head is copied layer by layer, with
the extra clock column of the first layer zero-initialised. The gripper row is
kept as well, so the warm-started actor reproduces the behaviour-cloning policy's
full 7-D output exactly at initialisation (verified to $\max|\Delta| < 10^{-4}$).
The critic copies the same encoders as a value-learning prior; its Q-heads and
both auxiliary heads start from random initialisation.

**From scratch.** Required whenever `drop_joint_state` or a changed point-cloud
channel count alters an encoder's input width, since the behaviour-cloning
weights no longer line up. The point-cloud encoder can still be seeded
independently from a pre-trained checkpoint (tensors whose shapes do not match
are skipped and left at initialisation).

## 8. Training architecture

The learner is deliberately kept simple and synchronous; the parallelism sits on
the data-generation side, where the cost is.

```
   ┌── worker 1 ──┐   env + planner + CPU actor copy ─┐
   ┌── worker 2 ──┐   env + planner + CPU actor copy  ├─► transitions
        …            (16 processes, 1 thread each)    │
   └── worker 16 ─┘                                   ┘
                          │                                     ▲ actor weights
                          ▼                                     │ (broadcast each iteration)
      ┌───────────────────────────────┐   ┌────────────────────┴─────────┐
      │ online replay buffer (FIFO)   │──►│  TD3 + BC learner (GPU)      │
      │ demo pool (non-evicting, HDF5)│──►│  mixed batch, demo fraction  │
      └───────────────────────────────┘   └──────────────────────────────┘
```

Each rollout worker owns a private simulator instance, motion planner and CPU
copy of the actor; the manager decides scene indices and episode types centrally
so the training schedule is unchanged relative to the serial loop, and only the
per-episode random stream differs. Transitions carry, in addition to the usual
$(s, a, r, s', \text{terminal})$ tuple, the discounted Monte-Carlo return, the
expert pose label, the auxiliary goal target at $s$ and $s'$, the gripper label,
and validity flags — the fields required by the auxiliary and imitation terms.
Successful demonstration episodes are held in a separate, non-evicting pool and
mixed into every batch at a scheduled fraction, so that the rare positive
outcomes cannot be flushed out of the FIFO.

One learner update fits the twin critic to a blend of the one-step Bellman target
and the Monte-Carlo return, and — every second update — takes a delayed actor
step whose loss combines a normalised deterministic policy gradient, a
SmoothL1 imitation term on the pose channels, a class-balanced binary
cross-entropy on the gripper logit, and the auxiliary regression. One
architecture-level detail is worth stating here because it constrains the graph
rather than the objective: the gripper logit is passed to the critic **detached**,
so the policy gradient flows only through the six pose channels. An unbounded,
near-binary logit riding $\partial Q/\partial \ell$ drives the critic into an
out-of-distribution region and destabilises both networks; the gripper is
therefore shaped exclusively by its supervised term, which already encodes the
reward-earning behaviour. Target networks for both actor and critic are updated
by Polyak averaging after every delayed actor step.

### Reference hyperparameters

| Group | Parameter | Value |
|---|---|---|
| Observation | points $N$ / channels $C$ | 1024 / 5 or 6 |
| | feature width $D$ | 256 |
| | horizon $T$ | 30 policy steps |
| Action | dimension | 7 (6 pose + 1 gripper logit) |
| | clamp $A$ | 5.0 |
| | action repeat | 150 simulator substeps |
| TD3 | $\gamma$ / $\tau$ | 0.95 / 0.005 |
| | target noise / clip | 0.2 / 0.5 |
| | policy delay | 2 |
| | Bellman–MC blend | 0.5 |
| Losses | PG coefficient $\alpha$ (normalised) | 0.1 |
| | pose imitation weight | 2.0 |
| | gripper BCE weight / label smoothing | 1.0 / 0.1 |
| | auxiliary weight | 0.5 |
| Optimisation | actor / critic learning rate | $3\times10^{-4}$ |
| | gradient-norm clip | 1.0 |
| | batch size | 64 |
| Loop | rollout workers | 16 |
| | episodes / updates per iteration | 16 / 800 |
| | replay capacity | 20 000 |
| | demonstration fraction | 0.5 → 0.3 |
