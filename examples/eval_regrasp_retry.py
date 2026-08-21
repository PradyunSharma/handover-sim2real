"""
Evaluate CHAINED regrasping: on failure, back up along the failed trajectory and
try the next pinned grasp from there.

    python examples/eval_regrasp_retry.py \\
        --run-dir output/dagger_runs/dagger5_run1 \\
        --policy-dir output/dagger_runs/dagger5_run1/last \\
        --rewind-frac 0.30

    # watch it in a PyBullet window — use a handful of scenes, not the whole pool
    python examples/eval_regrasp_retry.py \\
        --run-dir output/dagger_runs/dagger5_run1 \\
        --policy-dir output/dagger_runs/dagger5_run1/last \\
        --scenes 12,40,88 --render

The protocol, per eval scene:

    roll the policy conditioned on grasp 0
      success -> done, one attempt
      failure -> rewind to floor(0.30 * len(trajectory)) by replaying that many
                 recorded joint targets, switch the conditioning to grasp 1, and
                 keep rolling from there
    repeat until success or the scene's four grasps are exhausted

WHAT IT ADDS OVER `retry_at_k`. The `retry_at_k` columns in `eval_log.csv` are an
OR over four INDEPENDENT episodes, each from its own `env.reset()`. Nothing is
carried from the failed attempt, so the retry never pays for it. Here attempt k
resumes from a state the policy drove itself into, which is the situation a real
regrasp is actually in. `chained_retry_at_k` is the comparable column, and the
gap against `retry_at_k` is what the reset-based version was giving away.

READ `chained_retry_at_1` FIRST AS A CONTROL. Attempt 0 is exactly the slot-0
episode, so it must reproduce `succ_g0` from the same checkpoint up to rollout
noise. If it does not, the chain is not starting where the independent
evaluation starts and no other column here is comparable with anything.

THEN READ `replay_err_max`. The rewind assumes the simulator reproduces a
recorded command sequence exactly; this is the measurement of that, in metres of
EE divergence at the branch point. It should be ~0. Anything above `--replay-tol`
means the branch states are not the ones the policy really visited.

AND READ `retryable_frac`. The rewind resets the simulator, which also rewinds
the HUMAN — the hand is a DexYCB playback keyed on a frame index. A real regrasp
gets no such thing, and after a HUMAN_CONTACT or DROP failure there is no world
left to retry in at all. `retryable_frac` is the share of retried failures that a
non-resetting retreat controller could have handled; the complement is the part
of this metric that exists only because the simulator can be rewound.

--render OPENS A PYBULLET WINDOW and draws the chain rather than summarising it.
The scene's four pinned grasps appear as gripper wireframes colour-coded by slot
(g0 green, g1 blue, g2 orange, g3 magenta), the commanded one bright and thick
and the rest dim; each attempt's EE path is drawn in its slot colour and LEFT UP
for the whole scene, so the trunk-and-fork shape of a chained retry is the thing
you actually see. The replayed prefix retraces in grey and the branch point gets
a white cross-hair. `SIM.egl` is forced off for the session — the headless GPU
renderer and an open window fight over the GL context and the hand camera comes
back black.

With a window open the run PAUSES after each scene instead of tearing the GUI
down, which is the moment there is most to look at — the finished trunk-and-fork
with every attempt's path still drawn. In the PyBullet window (it needs focus):

    N   next scene
    P   previous scene
    R   re-run this scene (same conditioning, fresh rollout RNG)
    A   run the remaining scenes without pausing
    Q   stop here — the scenes already rolled are still written

WHILE STEPPING, N BROWSES: running out of `--scenes` is not a reason to quit, so
the list extends from the usable pool on demand. `--scenes 10 --render` therefore
means "start at scene 10", and N walks on to 11, 12, ... (skipping anything the
pin table does not hold), wrapping at the end. Every scene actually rolled lands
in the outputs, so the reported `n_scenes` is what you browsed, not what you
passed. Headless runs never extend — there `--scenes` is the workload, and
quietly evaluating extra scenes would change what the rates are rates OF.

`--no-step` disables the pause and runs straight through.

Rendering is diagnostic and never load-bearing: every draw call is guarded, so a
debug-item limit or a closed window disables the overlay instead of killing a
chain halfway through. It also changes the offscreen renderer, which changes the
point cloud, which can flip a borderline close — so read numbers from a headless
run and use --render to understand them, not to produce them.

WRITES

    <out>.h5    every attempt of every scene as an `episode_%05d` group, with
                robot_states / actions / target_jp / ee_pos (+ point_clouds under
                --save-clouds). Each episode is the WHOLE trajectory — replayed
                prefix first, then the resumed segment — with `prefix_len`
                marking the seam, so it reads like any other Phase-5 episode.
    <out>.csv   one row per scene: the outcome and reason of each attempt.
    <out>.json  the aggregate metrics.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root
sys.path.insert(0, _HERE)                    # examples/

from handover_sim2real.regrasp import load_policy_runner            # noqa: E402
from handover_sim2real.regrasp.chained_retry import (               # noqa: E402
    BUDGETS, REWIND_MODES, RetryParams, chained_metrics, chained_retry_scene,
)
from handover_sim2real.regrasp.setup import build_regrasp_context    # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="output/dagger_runs/<name> — read for config.yaml, so the "
                        "simulator, pin table and scene pool match the run's own")
    p.add_argument("--policy-dir", default=None,
                   help="the run dir to load the policy from. Default: <run>/best "
                        "if it exists, else <run>/last, else the highest iters/iter_*")
    p.add_argument("--ckpt", default=None, choices=[None, "best", "last"],
                   help="override EVAL.ckpt")
    p.add_argument("--out", default=None,
                   help="output prefix; default <run>/chained_retry")
    p.add_argument("--rewind-frac", type=float, default=0.30,
                   help="branch at this fraction of the failed trajectory (0.30)")
    p.add_argument("--rewind-mode", default="previous", choices=list(REWIND_MODES),
                   help="'previous' rewinds into the attempt that just failed; "
                        "'first' always branches off attempt 0")
    p.add_argument("--budget", default="shared", choices=list(BUDGETS),
                   help="'shared' keeps each attempt inside one max_steps horizon "
                        "including the replayed prefix; 'fresh' gives the resumed "
                        "segment a full max_steps of its own")
    p.add_argument("--max-attempts", type=int, default=0,
                   help="cap the chain length (0 = every grasp in the pin table)")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="override EVAL.num_scenes (changes the pool)")
    p.add_argument("--scenes", default=None,
                   help="explicit comma list of scene indices, overriding the pool")
    p.add_argument("--replay-tol", type=float, default=0.005,
                   help="metres of EE divergence at the branch point before the "
                        "replay is reported as suspect")
    p.add_argument("--save-clouds", action="store_true",
                   help="also store the [T, N, C] point clouds (~100x the bytes)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="no per-attempt lines")

    g = p.add_argument_group(
        "visualisation",
        "Open a PyBullet window and draw the chain. Use a handful of scenes: the "
        "pacing is deliberate and the numbers are not the point here.")
    g.add_argument("--render", action="store_true",
                   help="open the GUI. The four pinned grasps are drawn as gripper "
                        "wireframes (the commanded one bright, the rest dim), each "
                        "attempt's EE path in its slot colour, the replayed prefix "
                        "in grey and the branch point as a white cross-hair. Paths "
                        "are left up for the whole scene, so the fork is visible.")
    g.add_argument("--pace", type=float, default=0.03,
                   help="seconds of sleep per policy step under --render (0.03)")
    g.add_argument("--replay-pace", type=float, default=0.01,
                   help="seconds per replayed step, so the retrace is watchable (0.01)")
    g.add_argument("--pause-s", type=float, default=1.2,
                   help="dwell after each attempt and each scene (1.2 s)")
    g.add_argument("--show-cloud", action="store_true",
                   help="also overlay the point cloud the policy sees each step "
                        "(orange = object, blue = hand). Slows the GUI noticeably.")
    g.add_argument("--no-path", action="store_true",
                   help="skip the EE trajectory lines")
    g.add_argument("--no-grasp-markers", action="store_true",
                   help="skip the gripper wireframes")
    g.add_argument("--no-step", action="store_true",
                   help="do not pause between scenes. By default --render waits "
                        "for a keypress in the PyBullet window after each scene: "
                        "N next, P previous, R re-run this scene, A run the rest "
                        "without pausing, Q stop here (results so far are still "
                        "written). While stepping N browses on into the usable "
                        "pool, so --scenes sets where to START, not a fixed list.")
    return p.parse_args()


def resolve_policy_dir(run_root: Path, explicit: str | None) -> Path:
    """Where to load weights from, preferring what the run actually published.

    `best/` only exists when the trainer had eval metrics of its own — with
    `EVAL.every: 0` (which is how Phase-5 run 1 is configured, so that scoring
    runs as a separate job) `maybe_update_best` never fires and the run publishes
    `last/` only. Falling through to the highest completed iteration covers a run
    interrupted before it wrote either.
    """
    if explicit:
        d = Path(explicit)
        if not (d / "config.yaml").exists():
            raise SystemExit(f"{d} has no config.yaml — not a policy run dir")
        return d
    for name in ("best", "last"):
        d = run_root / name
        if (d / "config.yaml").exists():
            return d
    iters = sorted((run_root / "iters").glob("iter_*"))
    for d in reversed(iters):
        if (d / "config.yaml").exists() and (d / "checkpoints").is_dir():
            return d
    raise SystemExit(
        f"no policy found under {run_root} (looked for best/, last/, iters/iter_*). "
        f"Pass --policy-dir explicitly.")


def write_h5(path: Path, per_scene: dict, save_clouds: bool) -> tuple[int, int]:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    n_ep = n_steps = 0
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "regrasp_chained_retry"
        f.attrs["note"] = ("each episode is one ATTEMPT: rows [0, prefix_len) were "
                           "replayed from the previous attempt, the rest were chosen "
                           "by the policy under this episode's grasp_idx")
        for scene in sorted(per_scene):
            for att in per_scene[scene]:
                if not att.rows:
                    continue
                g = f.create_group(f"episode_{n_ep:05d}")
                g.attrs.update({
                    "scene_idx": int(att.scene_idx),
                    "attempt": int(att.attempt),
                    "grasp_idx": int(att.grasp_idx),
                    "branch_step": int(att.branch_step),
                    "prefix_len": int(att.prefix_len),
                    "replay_err": float(att.replay_err),
                    "budget_steps": int(att.budget_steps),
                    "success": int(att.success),
                    "grasped": int(att.grasped),
                    "closed": int(att.closed),
                    "near": int(att.near),
                    "close_step": int(att.close_step),
                    "pos_err": float(att.pos_err),
                    "rot_err": float(att.rot_err),
                    "min_pos": float(att.min_pos),
                    "min_rot": float(att.min_rot),
                    "had_chance": int(att.had_chance),
                    "reason": str(att.reason),
                    "status": int(att.status),
                    "retryable": int(att.retryable),
                })
                if att.grasp_pose is not None:
                    g.attrs["grasp_pose_world"] = np.asarray(att.grasp_pose,
                                                             dtype=np.float64)
                rows = att.rows
                g.create_dataset("robot_states",
                                 data=np.stack([r["rs"] for r in rows]),
                                 compression="gzip")
                g.create_dataset("actions",
                                 data=np.stack([r["act"] for r in rows]),
                                 compression="gzip")
                g.create_dataset("ee_pos",
                                 data=np.stack([r["ee_pos"] for r in rows]),
                                 compression="gzip")
                # NaN on the closing step, which drives no plain env.step and so
                # has no target — that is also why a branch can never land there.
                width = next((len(r["jp"]) for r in rows if r["jp"] is not None), 9)
                g.create_dataset(
                    "target_jp",
                    data=np.stack([r["jp"] if r["jp"] is not None
                                   else np.full(width, np.nan) for r in rows]),
                    compression="gzip")
                if save_clouds and rows[0]["pc"] is not None:
                    g.create_dataset("point_clouds",
                                     data=np.stack([r["pc"] for r in rows]),
                                     compression="gzip")
                n_ep += 1
                n_steps += len(rows)
        f.attrs["num_episodes"] = n_ep
    return n_ep, n_steps


def write_csv(path: Path, per_scene: dict, num_grasps: int) -> None:
    fields = ["scene_idx", "n_attempts", "solved", "solved_at"]
    for k in range(num_grasps):
        fields += [f"a{k}_success", f"a{k}_reason", f"a{k}_steps",
                   f"a{k}_branch", f"a{k}_min_pos", f"a{k}_replay_err"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for scene in sorted(per_scene):
            atts = per_scene[scene]
            hit = next((i for i, a in enumerate(atts) if a.success), -1)
            row = {"scene_idx": scene, "n_attempts": len(atts),
                   "solved": int(hit >= 0), "solved_at": hit}
            for k, a in enumerate(atts):
                row.update({f"a{k}_success": a.success, f"a{k}_reason": a.reason,
                            f"a{k}_steps": len(a.rows), f"a{k}_branch": a.branch_step,
                            f"a{k}_min_pos": round(a.min_pos, 4),
                            f"a{k}_replay_err": round(a.replay_err, 5)})
            w.writerow(row)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_dir)
    cfg_path = run_root / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config.yaml in {run_root}")
    with cfg_path.open() as f:
        cfg5 = yaml.safe_load(f)

    if args.num_scenes is not None:
        cfg5.setdefault("EVAL", {})["num_scenes"] = int(args.num_scenes)
    seed = int(args.seed if args.seed is not None
               else cfg5.get("DAGGER", {}).get("seed", 0))

    # The GUI-vs-DIRECT bullet connection is made inside gym.make(), so this has
    # to be in the SIM block BEFORE build_regrasp_context — flipping cfg.SIM.RENDER
    # afterwards would do nothing at all.
    if args.render:
        cfg5.setdefault("SIM", {})["render"] = True
        if cfg5["SIM"].get("egl"):
            # EGL is the headless GPU renderer; with a window open the two fight
            # over the GL context and the hand camera comes back black.
            print("[render] disabling SIM.egl for the GUI session")
            cfg5["SIM"]["egl"] = False

    ctx = build_regrasp_context(cfg5, seed=seed)
    if ctx.pin_table is None:
        raise SystemExit(
            "this run has no grasp pin table, so there is no second grasp to "
            "retry with — chained regrasping is undefined without SIM.grasp_pin_table")
    if ctx.num_grasps < 2:
        raise SystemExit(
            f"the pin table holds {ctx.num_grasps} grasp(s) per scene. Chained "
            f"regrasping needs at least 2; this looks like a Phase-4 table.")

    policy_dir = resolve_policy_dir(run_root, args.policy_dir)
    ckpt = args.ckpt or ctx.eval_ckpt
    runner, _ = load_policy_runner(policy_dir, args.device, ckpt=ckpt)

    scenes = (sorted({int(s) for s in args.scenes.split(",")}) if args.scenes
              else list(ctx.eval_scenes))
    retry = RetryParams(rewind_frac=args.rewind_frac, rewind_mode=args.rewind_mode,
                        budget=args.budget, max_attempts=args.max_attempts,
                        replay_tol=args.replay_tol, save_clouds=args.save_clouds,
                        verbose=not args.quiet)

    out = Path(args.out) if args.out else run_root / "chained_retry"

    viz = None
    if args.render:
        from handover_sim2real.regrasp.chain_viz import ChainViz
        from scipy.spatial.transform import Rotation as Rot
        cloud_ctx = (ctx.sim.panda_base_inv_tf,
                     Rot.from_quat(np.asarray(
                         ctx.sim_cfg.ENV.PANDA_BASE_ORIENTATION)).as_matrix(),
                     np.asarray(ctx.sim_cfg.ENV.PANDA_BASE_POSITION))
        viz = ChainViz(pace=args.pace, replay_pace=args.replay_pace,
                       pause_s=args.pause_s, show_grasps=not args.no_grasp_markers,
                       show_path=not args.no_path, show_cloud=args.show_cloud,
                       cloud_ctx=cloud_ctx)

    print("=" * 78)
    print(f"Phase-5 CHAINED retry    run={run_root.name}")
    print(f"  policy      : {policy_dir}  ({ckpt})")
    print(f"  scenes      : {len(scenes)} x up to "
          f"{retry.max_attempts or ctx.num_grasps} attempts")
    print(f"  rewind      : {retry.rewind_frac:.0%} of the failed trajectory, "
          f"mode={retry.rewind_mode}, budget={retry.budget}")
    print(f"  success     : {ctx.eval_params.success_mode}   "
          f"horizon={ctx.eval_params.max_steps}   target={ctx.eval_params.target}")
    print(f"  writing     : {out}.h5 / .csv / .json")
    if viz is not None:
        print(f"  RENDER      : GUI on, pace={args.pace}s/step   "
              f"grasp colours: " +
              "  ".join(f"g{i}={n}" for i, n in
                        enumerate(("green", "blue", "orange", "magenta")[:ctx.num_grasps])))
        print("                grey = replayed prefix, white cross = branch point")
        print("  KEYS        : " + ("N next  P prev  R re-run  A run-the-rest  "
                                    "Q stop   (focus the PyBullet window; N "
                                    "browses on past --scenes)"
                                    if not args.no_step else "stepping off (--no-step)"))
    print("=" * 78)

    # Interactive stepping: with a window open, pause after each scene rather
    # than tearing the GUI down the moment the last one finishes — which is
    # precisely when there is something to look at, since every attempt's path is
    # still drawn. `--no-step` runs straight through.
    stepping = bool(viz is not None and not args.no_step)

    # While stepping, N BROWSES rather than walking a fixed list: running out of
    # `--scenes` is not a reason to quit, so the list extends from the pool on
    # demand. `--scenes 10 --render` therefore means "start at 10", and N goes to
    # the next usable scene after it. Headless runs never extend — there the list
    # is the workload, and silently evaluating extra scenes would change what a
    # reported rate is a rate OF.
    browse_pool = sorted(ctx.usable) if ctx.usable else sorted(ctx.pin_table.entries)

    def next_in_pool(after: int):
        """Smallest usable scene greater than `after`, wrapping at the end."""
        j = bisect.bisect_right(browse_pool, int(after))
        if not browse_pool:
            return None
        if j >= len(browse_pool):
            print("  [browse] end of the pool — wrapping to the first scene")
            j = 0
        return browse_pool[j]

    def advance(i: int):
        """i+1, growing the browse list from the pool. None = nothing left."""
        i += 1
        if stepping and i >= len(scenes):
            nxt = next_in_pool(scenes[-1])
            if nxt is None:
                return None
            scenes.append(nxt)
        return i

    per_scene: dict[int, list] = {}
    t0 = time.time()
    i = 0
    while i is not None and i < len(scenes):
        scene = scenes[i]
        poses = [ctx.pin_table.pose(scene, g) for g in range(ctx.num_grasps)]
        if poses[0] is None:
            print(f"  [skip] scene {scene}: no slot-0 pose in the pin table")
            i = advance(i)
            continue
        # A missing later slot truncates the chain rather than dropping the scene:
        # the scene still contributes an honest chained_retry_at_1, and cutting it
        # entirely would bias the pool toward scenes with a full grasp set.
        poses = [p for p in poses if p is not None]
        if not args.quiet:
            total = f"/{len(scenes)}" if not stepping else ""
            print(f"  [{i + 1:3d}{total}] scene {scene}")
        per_scene[scene] = chained_retry_scene(
            ctx.sim, runner, scene, poses,
            params=ctx.eval_params, retry=retry, viz=viz)

        if not stepping:
            i = advance(i)
            continue

        atts = per_scene[scene]
        hit = next((a.attempt for a in atts if a.success), -1)
        verdict = (f"scene {scene}: SOLVED on grasp {hit} after {len(atts)} "
                   f"attempt(s)" if hit >= 0 else
                   f"scene {scene}: unsolved after {len(atts)} attempt(s) "
                   f"({', '.join(a.reason for a in atts)})")
        key = viz.wait(verdict, keys="npraq")
        if key == "q":
            # Not an abort: everything rolled so far is real and gets written.
            print(f"  [quit] stopping after {len(per_scene)} scene(s)")
            break
        if key == "r":
            continue                      # same index — re-roll this scene
        if key == "p":
            i = max(0, i - 1)             # back one, never off the front
            continue
        if key == "a":
            # "Run the rest" means the rest of what was ASKED for — turning
            # stepping off here also stops advance() from extending the list, so
            # A cannot run away through all 383 scenes of the pool.
            stepping = False
        i = advance(i)

    if viz is not None:
        viz.close()
    elapsed = time.time() - t0
    m = chained_metrics(per_scene, ctx.num_grasps)
    m.update({"run_dir": str(run_root), "policy_dir": str(policy_dir), "ckpt": ckpt,
              "rewind_frac": retry.rewind_frac, "rewind_mode": retry.rewind_mode,
              "budget": retry.budget, "elapsed_s": round(elapsed, 1)})

    n_ep, n_steps = write_h5(out.with_suffix(".h5"), per_scene, args.save_clouds)
    write_csv(out.with_suffix(".csv"), per_scene, ctx.num_grasps)
    with out.with_suffix(".json").open("w") as f:
        json.dump(m, f, indent=1)

    print("\n" + "=" * 78)
    print(f"chained retry over {m['n_scenes']} scenes   ({elapsed / 60:.1f} min)")
    for k in range(1, ctx.num_grasps + 1):
        key = f"chained_retry_at_{k}"
        if key in m:
            print(f"  chained_retry@{k} : {m[key]:.3f}")
    print(f"  solved           : {m['solved_rate']:.3f}   "
          f"mean attempts {m['mean_attempts']:.2f} "
          f"({m['mean_attempts_to_success']:.2f} when solved)")
    print("  per-attempt (CONDITIONAL on the earlier grasps failing — not "
          "comparable with succ_g*):")
    for k in range(ctx.num_grasps):
        if m.get(f"attempt_{k}_n"):
            print(f"     attempt {k}: {m[f'attempt_{k}_succ']:.3f} "
                  f"over {m[f'attempt_{k}_n']} scenes")
    print(f"  branch step      : mean {m['mean_branch_step']:.1f}   "
          f"{m['n_branch_at_home']} branched at home")
    print(f"  replay err (m)   : mean {m['replay_err_mean']:.5f}  "
          f"max {m['replay_err_max']:.5f}"
          + ("   <-- ABOVE TOLERANCE, branch states are suspect"
             if (m["replay_err_max"] == m["replay_err_max"]
                 and m["replay_err_max"] > args.replay_tol) else ""))
    if m.get("n_replay_diverged"):
        print(f"  REPLAY DIVERGED  : {m['n_replay_diverged']} attempts ended during "
              f"the replay — the sim is not reproducing recorded commands")
    print(f"  retryable        : {m['retryable_frac']:.3f} of retried failures left "
          f"a world a non-resetting retreat could have used")
    print(f"  reasons          : {m['reasons']}")
    print(f"\nwrote {out}.h5  ({n_ep} attempts, {n_steps} steps)")
    print(f"wrote {out}.csv")
    print(f"wrote {out}.json")
    print("\nCompare chained_retry_at_k against retry_at_k in eval_log.csv for the "
          "same checkpoint. chained_retry_at_1 should reproduce succ_g0.")


if __name__ == "__main__":
    main()
