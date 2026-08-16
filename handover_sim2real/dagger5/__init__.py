"""
Phase-5: grasp-CONDITIONED DAgger with several pinned grasps per scene.

A fork of `handover_sim2real/dagger/` (Phase 4), so all 21 recorded Phase-4 runs
stay byte-reproducible while this diverges. The loop is unchanged — still
Algorithm 3.1 of Ross et al. (2011) — and so is every observation: the difference
is what a unit of work IS.

**The unit of work is a `(scene_idx, grasp_idx)` pair, not a scene.** Each scene
carries N (default 4) physically distinct pinned grasps chosen by
`grasp_select.select_diverse_grasps`, the policy is told which one it is being
asked to reach, and a failed handover can be retried under a different one. That
last part is the point: regrasping.

The conditioning is not decoration. With four demonstrations per scene the same
observation carries four different expert labels, so an unconditioned regression
can only predict their mean — a valid action for none of them. It is also the
strong form of the fix Phase 4 kept missing: over runs 4-16 the policy closed
within tolerance of the pinned grasp on 0-8% of episodes because nothing in the
observation named the pose to arrive at (run 13's auxiliary head tried to force
that via a PREDICTION; here it is an INPUT).

Six modules carry the change: `grasp_select` (new — the flip-invariant metric and
the selection), `grasp_pin` (a list per scene), `collector` (a `grasp_idx`
argument and two new episode attrs), `grasp_registry` (keyed on the pair),
`parallel` (the job tuple), `evaluator` (per-slot rates, retry@k, `cond_track`).

Phase 1/2 shipped DAgger as a *shell* loop over one-shot collector scripts;
Phase 3 folded DAgger labels inline into the RL rollout. Phase 4/5 are neither:
Algorithm 3.1 of the paper, run as a single Python loop.

    D <- expert demonstrations (train.h5)          # the beta_1 = 1 iteration
    pi_1 <- train(D)
    for i = 1..N:
        pi_i^mix = beta_i * pi* + (1 - beta_i) * pi_i
        sample m T-step trajectories with pi_i^mix   # m = episodes_per_iter
        D_i = {(s, pi*(s)) : s visited}
        D <- D u D_i                                 # aggregate, never discard
        pi_{i+1} <- train(D)                         # fresh fit on the union
        evaluate pi_{i+1}                            # -> best-on-validation
    return best pi_i, last pi_i

The loop is policy-agnostic. The learner is the Phase-1 single-frame BC policy by
default; point TRAIN.train_cfg at `act_phase2.yaml` instead and the same loop
drives the Phase-2 temporal/chunking ACT policy. The kind is inferred from the
config (`MODEL.chunk_len` present => ACT) and hidden behind `PolicyRunner`, so
neither the collector nor the evaluator has policy-specific code.

Evaluation scores the PHASE-3 success criterion, not the handover-sim benchmark's:
there is no carry-to-goal here, so `EpisodeStatus.SUCCESS` could never fire. See
`evaluator.py`.

Modules:
    env_setup  one env serving both collection (OMG) and evaluation
    policy_io  run-dir load/save + the PolicyRunner per-step interface
    collector  Phase-4 DAgger episode collection -> BC-schema HDF5
    evaluator  closed-loop eval (Phase-3 criterion) for best-policy selection
    grasp_box  ray-cast "is the object in the open jaws" opportunity test
    pregrasp   the CVPR2023 blind endgame, for DAGGER.target: pregrasp
"""

from .env_setup import SimContext, build_sim_cfg, build_sim_context
from .policy_io import (
    ACTRunner,
    BCRunner,
    PolicyRunner,
    build_policy,
    export_run_dir,
    load_policy_runner,
    policy_kind,
    read_run_cfg,
)
from .grasp_pin import GraspPinTable, load_grasp_pin_table
from .grasp_registry import GraspRegistry
from .collector import (
    CollectParams,
    collect_dagger_episode,
    collect_iteration,
    DaggerHDF5Writer,
    dart_alpha_at,
    dart_bootstrap_sigma,
    dart_scaled_sigma,
    derived_standoff_pose,
)
from .pregrasp import forward_dist_default, open_loop_reach
from .grasp_box import BoxParams, build_box_params, grasp_opportunity
from .evaluator import EvalParams, evaluate_policy
from .setup import Phase5Context, build_phase5_context, scene_pools
# NOT imported here: .parallel. It spawns worker processes that re-import this
# package, and pulling torch.multiprocessing in at package import would pay that
# cost for every consumer. train_dagger_phase4 imports it only when
# --num-workers > 1.

__all__ = [
    "SimContext",
    "build_sim_cfg",
    "build_sim_context",
    "PolicyRunner",
    "BCRunner",
    "ACTRunner",
    "build_policy",
    "policy_kind",
    "read_run_cfg",
    "load_policy_runner",
    "export_run_dir",
    "GraspPinTable",
    "load_grasp_pin_table",
    "GraspRegistry",
    "CollectParams",
    "collect_dagger_episode",
    "collect_iteration",
    "dart_alpha_at",
    "dart_bootstrap_sigma",
    "dart_scaled_sigma",
    "derived_standoff_pose",
    "forward_dist_default",
    "open_loop_reach",
    "DaggerHDF5Writer",
    "BoxParams",
    "build_box_params",
    "grasp_opportunity",
    "EvalParams",
    "evaluate_policy",
    "Phase5Context",
    "build_phase5_context",
    "scene_pools",
]
