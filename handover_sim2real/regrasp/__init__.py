"""
Regrasp: approach-direction conditioning and retry.

The policy is told **which side to come in from** — a unit vector `d` injected
PER-POINT into the cloud as `d.n_i` and `d.normalize(p_i - c)` — rather than which
pose to reach. On failure it is commanded a different side and tries again.

    directions   the k=6 octahedral bin set, angles, and the pair selector
    anchor       the gravity-aligned, hand-anchored frame the bins live in
    normals      kNN-PCA surface normals from the observed cloud
    channels     [N,5] + d -> the [N,7] the network eats
    retry        the ladder: which direction next, and when to signal the human
    grasp_pin    scene -> the grasps realising each bin (OMG still needs a pose)
    collector    DAgger episode collection -> the 8-channel HDF5 schema
    evaluator    closed-loop eval + the dir_* metrics
    chained_retry  rewind to part-way along a failed attempt and re-command

WHY A DIRECTION. It is object-agnostic, so coverage is required across the
DATASET rather than per object; it needs no grasp proposer at deployment; and
because the network reads a continuous vector rather than a one-hot, `k` is a
test-time knob (up to 20, past which bins are closer than 40 deg and stop being
independent hypotheses).

MEASURED, AND IT BOUNDS EVERYTHING HERE: on s0/train only FOUR of the six bins
are reachable. `-z` (from beneath) by 0 of 623 scenes and `-x` (over the giver's
fingers) by 12. Both are kept and masked at runtime so the code stays
rig-agnostic, but `chained_retry_at_k` saturates at k=4 and `succ_bin_-x` /
`succ_bin_-z` are NaN rather than zero.

See README_REGRASP.md for the runbook and docs/run_index.md for run 1's result.
"""

# ── LAZY re-exports (PEP 562) ────────────────────────────────────────────────
#
# Eager `from .env_setup import ...` here made importing ANY submodule pull gym,
# pybullet, the handover envs, and `add_sys_path_from_env("GADDPG_DIR")` — which
# ASSERTS, so `from handover_sim2real.regrasp.channels import build_model_cloud`
# died with "Environment variable 'GADDPG_DIR' is not set" on any machine without
# the vendored trees.
#
# That blocked three things that matter:
#   * the geometry modules (directions, anchor, normals, channels) are pure numpy
#     and are needed in DataLoader workers, on login nodes, and eventually on the
#     robot PC, none of which have or want a simulator;
#   * login-node scripts used to insert the package DIRECTORY on sys.path to
#     bypass this file entirely — a hack that silently broke on a rename, because
#     a path built from string literals is invisible to every import-graph check;
#   * `parallel.py` is still excluded below for the same class of reason.
#
# Names resolve on first attribute access instead, so `import
# handover_sim2real.regrasp` stays free and `from handover_sim2real.regrasp import
# BCRunner` costs exactly what it always did. The submodule path
# (`...regrasp.channels`) now runs only this file plus the module asked for.

_LAZY = {
    "SimContext": "env_setup", "build_sim_cfg": "env_setup",
    "build_sim_context": "env_setup",
    "ACTRunner": "policy_io", "BCRunner": "policy_io", "PolicyRunner": "policy_io",
    "build_policy": "policy_io", "export_run_dir": "policy_io",
    "load_policy_runner": "policy_io", "policy_kind": "policy_io",
    "read_run_cfg": "policy_io",
    "GraspPinTable": "grasp_pin", "load_grasp_pin_table": "grasp_pin",
    "GraspRegistry": "grasp_registry",
    "CollectParams": "collector", "collect_dagger_episode": "collector",
    "collect_iteration": "collector", "DaggerHDF5Writer": "collector",
    "dart_alpha_at": "collector", "dart_bootstrap_sigma": "collector",
    "dart_scaled_sigma": "collector", "derived_standoff_pose": "collector",
    "forward_dist_default": "pregrasp", "open_loop_reach": "pregrasp",
    "BoxParams": "grasp_box", "build_box_params": "grasp_box",
    "grasp_opportunity": "grasp_box",
    "EvalParams": "evaluator", "evaluate_policy": "evaluator",
    "RegraspContext": "setup", "build_regrasp_context": "setup",
    "scene_pools": "setup",
}


def __getattr__(name):
    """Resolve a re-exported name on first access (PEP 562)."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value          # cache: subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


# NOT re-exported, lazily or otherwise: .parallel. It spawns worker processes that
# re-import this package, and pulling torch.multiprocessing in at package import
# would pay that cost for every consumer. train_regrasp imports it only when
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
    "RegraspContext",
    "build_regrasp_context",
    "scene_pools",
    # ---- Regrasp geometry: pure numpy, no simulator, importable anywhere ----
    "directions",
    "anchor",
    "normals",
    "channels",
]
