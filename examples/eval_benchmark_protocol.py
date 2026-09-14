"""
Score a finished run under the OFFICIAL HandoverSim benchmark protocol, so the
number can be placed beside Christen et al. (CVPR 2023).

    # Regrasp, the benchmark as published (moving human)
    python examples/eval_benchmark_protocol.py \\
        --run-dir output/dagger_runs/regrasp_run19 --iter 25 --ckpt best

    # Phase 4, same
    python examples/eval_benchmark_protocol.py \\
        --run-dir output/dagger_runs/dagger4_run19 --iter 19

    # our OWN distribution (static hand), to isolate what the carry costs
    python examples/eval_benchmark_protocol.py --run-dir ... --iter 25 \\
        --start-frame last --time-wait 0

    # re-run the official evaluator on an existing result dir
    python examples/eval_benchmark_protocol.py --evaluate-only \\
        --res-dir output/benchmark/regrasp_run19_it25_s0_test

WHAT IS DIFFERENT FROM `eval_regrasp_testset.py` / `eval_dagger_testset.py`

Those score `stable_grasp`: close, hold, object secured. The benchmark wants
that AND the carry — the hand inside a 15 cm ball at GOAL_CENTER for 0.1 s,
inside 13 s. See `handover_sim2real/benchmark.py` for the full criterion and for
why benchmark-SUCCESS is a strict subset of ours (so our published rates are
upper bounds on this one).

THE METRICS ARE COMPUTED BY THEIR CODE, NOT OURS. This writes the official
per-scene `{:03d}.npz` and a `config.yaml` into a result directory and then calls
`handover.benchmark_evaluator.evaluate()` on it — the same function
`handover-sim/examples/evaluate_benchmark.py` calls on the archived CVPR-2023
result dirs. A running summary is printed during the sweep from our own copy of
the arithmetic, and the two are cross-checked at the end; on disagreement theirs
is authoritative and this script says so.

TWO THINGS TO SET DELIBERATELY, AND BOTH CHANGE WHAT IS BEING MEASURED

  --start-frame   `first` (DEFAULT) is the benchmark as published: the human's
        reach plays out and the object is a MOVING target. `last` is what every
        config in this repo trains and evaluates on — the hand frozen fully
        extended, static from t=0. Our policies have never seen `first`. Running
        it is the honest comparison with Christen et al. and it is also an
        out-of-distribution test; running `last` isolates the cost of the carry
        alone but is comparable with nobody else's number. Say which in any
        report.

  --iter / --ckpt   ONE checkpoint, because the benchmark is a final number
        rather than a curve. Defaults to the run's own best-on-train iteration
        from `state.json`, which is not necessarily best here.

SCENES. All of them, always: `evaluate()` refuses to run unless the directory
holds exactly `env.num_scenes` files (144 on s0/test). There is no pin table and
no scene filter — the policy needs neither under this protocol — so the 14 s0
test scenes OMG cannot plan for ARE scored, as they are for every published
baseline.

REGRASP NEEDS A DIRECTION AND THE BENCHMARK GIVES IT NO SLOTS. The protocol is
one episode per scene, so exactly one command per scene has to be chosen. That
choice is `--bins`: the ranked order is walked and the first bin the scene can
anchor is commanded. The anchor frame needs only the observation (object cloud
centroid, MANO wrist, world up), so it is available on every scene including the
unplannable ones — `d = to_world(command_axes[b], anchor_R)` and nothing else.
The retry ladder is NOT expressible here: one attempt is ~7.5 s of a 13 s budget
and a second plus its rewind does not fit, so `chained_retry_at_k` for k >= 2
stays a claim under our own protocol and cannot be folded into this number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from handover_sim2real import benchmark as B                    # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None, help="output/dagger_runs/<name>")
    p.add_argument("--iter", type=int, default=None,
                   help="which iteration to score (default: the run's own best "
                        "from state.json, which was selected on the TRAIN "
                        "subsample and need not be best here)")
    p.add_argument("--ckpt", default=None, choices=["best", "last"],
                   help="default: the run's own EVAL.ckpt")
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--res-dir", default=None,
                   help="default output/benchmark/<run>_it<N>_<setup>_<split>")
    p.add_argument("--start-frame", default="first", choices=["first", "last"],
                   help="ENV.YCB_MANO_START_FRAME. `first` (default) is the "
                        "benchmark as published — the human reaches and the "
                        "object moves. `last` is this repo's training "
                        "distribution, hand frozen fully extended. Different "
                        "experiments; report which.")
    p.add_argument("--ycb-load-mode", default="all",
                   choices=["all", "grasp_only"],
                   help="ENV.YCB_LOAD_MODE. `all` (DEFAULT) is the benchmark as "
                        "published — the other YCB objects are on the table, so "
                        "they clutter the scene and can be knocked or touched "
                        "during the carry (the drop test's `contact_ycb_other` "
                        "becomes live). Every config in this repo sets "
                        "`grasp_only`, loading ONLY the target, which is easier "
                        "and is a third divergence from the benchmark alongside "
                        "the carry and the start frame. Pass `grasp_only` to "
                        "hold this repo's setting; the number is then not "
                        "comparable with published ones.")
    p.add_argument("--time-wait", type=float, default=None,
                   help="seconds held at the start configuration before the "
                        "policy acts (default 3.0 with --start-frame first, "
                        "which is what every published benchmark policy uses; "
                        "0.0 with `last`, where there is nothing to wait for)")
    p.add_argument("--time-close-gripper", type=float,
                   default=B.TIME_CLOSE_GRIPPER,
                   help="seconds the commit pose is held with the fingers shut "
                        "before the carry begins (GA-DDPG's 0.5)")
    p.add_argument("--back-step-size", type=float, default=B.BACK_STEP_SIZE,
                   help="metres per retreat waypoint (the benchmark's 0.03)")
    p.add_argument("--max-policy-steps", type=int, default=0,
                   help="cap on APPROACH policy steps, 0 = the benchmark's own "
                        "13 s clock. Not a scoring threshold: hitting it stops "
                        "new approach commands and lets the clock expire into "
                        "FAILURE_TIMEOUT, which is what would have been recorded "
                        "anyway.")
    p.add_argument("--bins", default=None,
                   help="REGRASP ONLY: the direction preference order, best "
                        "first (e.g. '+y,+z,+x,-y'). One command per scene — the "
                        "benchmark allows one episode — so the first bin in this "
                        "order that the scene can anchor is the one issued. "
                        "Default: directions.RETRY_LADDER.")
    p.add_argument("--command-axes", default=None,
                   help="REGRASP ONLY: command_axes.json (default the run's own, "
                        "which is what the policy was trained to obey)")
    p.add_argument("--num-scenes", type=int, default=None,
                   help="cap for a smoke test. The official evaluator REFUSES a "
                        "partial directory, so this prints the running summary "
                        "and skips the official pass.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--evaluate-only", action="store_true",
                   help="skip the sweep, run the official evaluator on --res-dir")
    p.add_argument("--force", action="store_true",
                   help="re-run scenes whose .npz already exists")
    return p.parse_args()


def run_kind(cfg4: dict) -> str:
    """`regrasp` or `phase4`, from the config alone.

    `SIM.grasp_pin_table` names a Regrasp table (`regrasp_pins_*`) or a Phase-4
    one (`grasp_pin_table_*`), and the Regrasp block carries keys Phase 4 has
    never had. Keyed on `command_deploy` because it is the one field that only
    ever exists on a direction-conditioned run and is present on every one of
    them, including run 2 where it defaults.
    """
    sim = cfg4.get("SIM", {}) or {}
    if "command_deploy" in sim or "anchor_hand_ref" in sim:
        return "regrasp"
    if "regrasp" in str(sim.get("grasp_pin_table", "")):
        return "regrasp"
    return "phase4"


def resolve_iter(run_root: Path, want) -> int:
    """`--iter`, or the run's own best from state.json."""
    if want is not None:
        return int(want)
    st = run_root / "state.json"
    if st.exists():
        with st.open() as f:
            state = json.load(f)
        best = (state.get("best") or {}).get("iter")
        if best is not None:
            print(f"[iter] no --iter given; using the run's own best-on-train "
                  f"iteration {int(best)} "
                  f"({(state.get('best') or {}).get('metric')}="
                  f"{(state.get('best') or {}).get('score')}). That selection "
                  f"was made on the TRAIN subsample and need not be best here.")
            return int(best)
    raise SystemExit("no --iter and no best recorded in state.json")


def iteration_dir(run_root: Path, i: int) -> Path:
    """The checkpoint directory for iteration `i`, tolerating a run synced off
    the cluster (state.json stores absolute paths from wherever it executed)."""
    st = run_root / "state.json"
    if st.exists():
        with st.open() as f:
            for rec in (json.load(f).get("iterations") or []):
                if int(rec["iter"]) == int(i):
                    d = Path(rec["run_dir"])
                    if (d / "checkpoints").is_dir():
                        return d
    d = run_root / "iters" / f"iter_{int(i):02d}"
    if (d / "checkpoints").is_dir():
        return d
    raise SystemExit(f"no checkpoints for iteration {i} under {run_root}")


def main() -> None:
    args = parse_args()

    if args.evaluate_only:
        if not args.res_dir:
            raise SystemExit("--evaluate-only needs --res-dir")
        official(Path(args.res_dir))
        return

    if not args.run_dir:
        raise SystemExit("--run-dir is required")
    run_root = Path(args.run_dir)
    cfg_path = run_root / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config.yaml in {run_root}")
    with cfg_path.open() as f:
        cfg4 = yaml.safe_load(f)
    kind = run_kind(cfg4)
    it = resolve_iter(run_root, args.iter)
    run_dir = iteration_dir(run_root, it)
    ckpt = args.ckpt or str((cfg4.get("EVAL") or {}).get("ckpt", "best"))

    # ---- the simulator, from the run's own config with three overrides ------
    # Everything else — the cameras, the point-cloud split, the action repeat —
    # stays as the run had it, so the only differences from the training-curve
    # number are the ones named here.
    sim_cfg_d = dict(cfg4["SIM"])
    sim_cfg_d["split"] = args.split
    # The pin table is BOTH the scoring reference and the scene filter. The
    # benchmark needs neither: it scores from `info["status"]`, and it requires
    # EVERY scene of the split. Dropping it is what lets the 14 unplannable s0
    # test scenes be scored, as they are for every published baseline.
    sim_cfg_d["grasp_pin_table"] = None
    sim_cfg_d.pop("exclude_scenes", None)
    sim_cfg_d.pop("demo_ok_table", None)
    sim_cfg_d["reach_filter"] = False

    if kind == "regrasp":
        from handover_sim2real.regrasp.env_setup import (
            build_sim_cfg, build_sim_context)
        from handover_sim2real.regrasp import load_policy_runner
    else:
        from handover_sim2real.dagger.env_setup import (
            build_sim_cfg, build_sim_context)
        from handover_sim2real.dagger import load_policy_runner

    seed = int(args.seed if args.seed is not None
               else (cfg4.get("DAGGER", {}) or {}).get("seed", 0))
    cfg = build_sim_cfg(sim_cfg_d)
    # THE PROTOCOL CHANGE, applied to the built config rather than the yaml so
    # it is visible in the `config.yaml` written beside the results — which is
    # the file `evaluate()` reads back to rebuild the env, so a mismatch here
    # would make the official pass score a different scene set than the sweep.
    cfg.defrost()
    cfg.ENV.YCB_MANO_START_FRAME = str(args.start_frame)
    cfg.ENV.YCB_LOAD_MODE = str(args.ycb_load_mode)
    cfg.freeze()
    sim = build_sim_context(cfg, sim_cfg_d, seed=seed)
    B.check_status_enum()

    time_wait = (args.time_wait if args.time_wait is not None
                 else (B.TIME_WAIT if args.start_frame == "first" else 0.0))

    setup = str(cfg.BENCHMARK.SETUP)
    res_dir = Path(args.res_dir or
                   f"output/benchmark/{run_root.name}_it{it}_{setup}_{args.split}")
    res_dir.mkdir(parents=True, exist_ok=True)
    # `evaluate()` reads this back to rebuild the env; it must describe the run
    # that produced the .npz files or the official pass scores a different split.
    with (res_dir / "config.yaml").open("w") as f:
        f.write(cfg.dump())

    runner, _ = load_policy_runner(run_dir, args.device, ckpt=ckpt)
    act_fn = make_act_fn(kind, sim, runner, cfg4, run_root, args)

    n_total = int(sim.num_scenes)
    todo = list(range(n_total if args.num_scenes is None
                      else min(int(args.num_scenes), n_total)))

    print("=" * 78)
    print(f"HandoverSim BENCHMARK PROTOCOL   run={run_root.name}  iter={it}  "
          f"ckpt={ckpt}  [{kind}]")
    print(f"  setup/split : {setup} / {args.split}   scenes: {len(todo)} of "
          f"{n_total}")
    print(f"  start frame : {args.start_frame}   "
          + ("(the benchmark as published — the human REACHES, the object MOVES; "
             "this run was trained on `last`)" if args.start_frame == "first"
             else "(THIS REPO'S TRAINING DISTRIBUTION — hand frozen extended. "
                  "NOT comparable with published numbers)"))
    print(f"  ycb load    : {args.ycb_load_mode}   "
          + ("(the benchmark as published — the other objects are on the table)"
             if args.ycb_load_mode == "all"
             else "(THIS REPO'S SETTING — target only, no clutter. NOT "
                  "comparable with published numbers)"))
    _ood = [n for n, v, bench in (("start-frame", args.start_frame, "first"),
                                  ("ycb-load-mode", args.ycb_load_mode, "all"))
            if v != bench]
    if _ood:
        print(f"  ** {', '.join(_ood)} differ(s) from the published benchmark — "
              f"this number is comparable across OUR runs only. **")
    print(f"  success     : hand within {cfg.BENCHMARK.GOAL_RADIUS} m of "
          f"{tuple(cfg.BENCHMARK.GOAL_CENTER)} for "
          f"{cfg.BENCHMARK.SUCCESS_TIME_THRESH} s, inside "
          f"{cfg.BENCHMARK.MAX_EPISODE_TIME} s")
    print(f"  wait/close  : {time_wait} s / {args.time_close_gripper} s   "
          f"retreat step {args.back_step_size} m")
    print(f"  writing     : {res_dir}")
    print("=" * 78)

    results = {}
    for n, idx in enumerate(todo):
        out_path = res_dir / f"{idx:03d}.npz"
        if out_path.exists() and not args.force:
            with np.load(out_path) as z:
                results[idx] = {k: z[k] for k in z.files}
            continue
        t0 = time.time()
        res = B.run_episode(
            sim, idx, act_fn, max_policy_steps=args.max_policy_steps,
            time_wait=time_wait, time_close_gripper=args.time_close_gripper,
            back_step_size=args.back_step_size)
        B.write_result(str(res_dir), idx, res)
        results[idx] = res
        s = B.summarise(results, cfg.SIM.TIME_STEP)
        print(f"  [{n + 1:3d}/{len(todo)}] scene {idx:3d}  "
              f"{B.status_name(res['result']):<14} "
              f"frames={res['elapsed_frame']:5d} "
              f"commit@{res['close_frame']:5d} "
              f"retreat={res['n_retreat_waypoints']:3d}wp  "
              f"({time.time() - t0:4.1f}s)   "
              f"running success={s['success_rate']:.3f}", flush=True)

    s = B.summarise(results, cfg.SIM.TIME_STEP)
    print("\n" + "-" * 78)
    print(f"running summary (OUR arithmetic): success {s['success_rate']:.4f} "
          f"({s['num_success']}/{s['num_scenes']})   "
          f"hand-contact {s['failure_human_contact']:.4f}  "
          f"drop {s['failure_object_drop']:.4f}  "
          f"timeout {s['failure_timeout']:.4f}")
    print(f"  commit rate {s['commit_rate']:.4f}; of the failures, "
          f"{s['fail_never_committed']:.4f} never committed a close at all "
          f"(the benchmark calls those TIMEOUT too)")

    if len(results) != n_total:
        print(f"\n[official] SKIPPED — {len(results)} of {n_total} scenes. "
              f"`benchmark_evaluator.evaluate` refuses a partial directory; "
              f"drop --num-scenes for the real number.")
        return
    official(res_dir, cross_check=s)


def make_act_fn(kind, sim, runner, cfg4, run_root, args):
    """`(obs, pc, rs, step) -> action[7]`, with the Regrasp command attached.

    PHASE 4 IS UNCONDITIONED and this is a one-liner. Regrasp has to choose ONE
    direction for the whole episode, because the benchmark allows one episode per
    scene, and it has to build the anchor frame itself: there is no pin table
    here (the protocol needs every scene, including the ones OMG cannot plan
    for), so the frame comes from the observation the way `anchor_from_cloud`
    builds it during a `live` run — object cloud centroid, MANO wrist, world up.
    """
    if kind == "phase4":
        return lambda obs, pc, rs, step: runner.act(pc, rs)

    import numpy as _np
    from handover_sim2real.regrasp import anchor as _anchor
    from handover_sim2real.regrasp import directions as _D
    from handover_sim2real.regrasp.bin_ranker import parse_bin_sequence

    # THE AXES THE POLICY WAS TRAINED TO OBEY, from the run's own file. Under
    # `command_deploy: bin_centroid` the command is the empirical mean of each
    # bin's assigned `d_anchor`, which is a property of the table it was computed
    # from — re-deriving it here from anything else would issue a vector no part
    # of training ever produced.
    axes = _D.BINS.copy()
    ap = Path(args.command_axes or (run_root / "command_axes.json"))
    if ap.exists():
        with ap.open() as f:
            axes = _np.asarray(json.load(f)["axes"], dtype=_np.float64)
        print(f"[command] conditioning on {ap}")
    else:
        print(f"[command] no {ap} — falling back to the raw bin axes "
              f"(SIM.command_deploy: bin_axis)")

    order = parse_bin_sequence(args.bins, live=_D.LIVE_BINS)
    print("[command] one direction per scene, first anchorable in: "
          + " > ".join(_D.BIN_SHORT[b] for b in order))

    state = {"d": None}
    hand_ref = str((cfg4.get("SIM") or {}).get("anchor_hand_ref", "wrist"))

    def act(obs, pc, rs, step):
        if step == 0:
            state["d"] = None
        # LATCHED AT THE FIRST STEP THAT CAN ANCHOR, matching every run in this
        # repo (`anchor_update: latched`). Under `--start-frame first` the hand
        # is still reaching when the policy starts, so the first few clouds may
        # carry no object points and the frame is simply not available yet —
        # retrying each step until it is, rather than failing the episode, is
        # what makes the moving-human setting runnable at all.
        if state["d"] is None:
            aR, _cw, _m = _anchor.anchor_from_cloud(
                pc, obs, sim.env, sim.panda_base_inv_tf, sim.cfg,
                _anchor.AnchorState(), hand_ref=hand_ref)
            if aR is not None:
                b = order[0]
                state["d"] = _D.command_direction(b, aR, grasp_pose=None,
                                                  axes=axes)
        if state["d"] is not None:
            runner.set_direction(state["d"])
        return runner.act(pc, rs)

    return act


def official(res_dir: Path, cross_check: dict | None = None) -> None:
    """Run `handover.benchmark_evaluator.evaluate` — THE number.

    Not a reimplementation and deliberately not wrapped: it writes its own
    `evaluate.log` into the directory and prints the table in the layout the
    paper's own results were reported in, so a row from here and a row from
    `all_cvpr2023_results_eval.sh` are the same row.
    """
    from handover.benchmark_evaluator import evaluate

    print("\n" + "=" * 78)
    print(f"OFFICIAL evaluation — handover.benchmark_evaluator.evaluate({res_dir})")
    print("=" * 78)
    evaluate(str(res_dir))

    if cross_check:
        # The running summary and the official pass are the same arithmetic over
        # the same files. They agreeing proves nothing; them DISAGREEING means
        # our copy has drifted, and it is the copy that is wrong.
        log = res_dir / "evaluate.log"
        if log.exists():
            txt = log.read_text()
            want = f"{100 * cross_check['success_rate']:.2f}"
            if want not in txt:
                print(f"\n[cross-check] WARNING: the running summary said "
                      f"{want}% and the official log does not contain that "
                      f"figure. THE OFFICIAL NUMBER IS THE ONE TO REPORT; "
                      f"`benchmark.summarise` has drifted and should be fixed.")
            else:
                print(f"\n[cross-check] running summary agrees with the official "
                      f"pass at {want}%.")


if __name__ == "__main__":
    main()
