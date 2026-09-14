"""
Static PyBullet view of ONE scene and the handover anchor frame it defines.

    python examples/show_handover_frame.py --run regrasp_run21 --scene 1

NO POLICY, NO EXPERT, NO EPISODE. The scene is reset and left standing: the
table, the YCB object and the MANO hand sit exactly where the benchmark puts
them at frame 0, and nothing steps. What you see is the geometry the Regrasp
conditioning is built from, with none of the motion that normally hides it.

The three viewers now draw the same things with the same functions --
`rollout_regrasp_policy.py` for a live policy, `visualize_bc_dataset.py` for a
recorded demonstration, and this one for a bare scene -- so a frame that looks
wrong in one can be checked against the others by eye. All the drawing lives in
`handover_sim2real/regrasp/viz.py`.

WHAT IS DRAWN

    red / green / blue axes   the anchor frame: x away from the giver, z world
                              up, y = z x x. Every commanded direction `d` is
                              expressed in it. Lettered `x` / `y` / `z`;
                              `--labels` spells out what each one means, at
                              the cost of text sitting over the object.
    cyan point                `c`, the object point-cloud centroid -- the
                              frame's ORIGIN.
    magenta point             `p_wrist`, the MANO wrist joint (link 7).
    thin grey line            the raw 3-D chord `c - p_wrist`.
    thick red line            its HORIZONTAL part, drawn in the centroid's
                              z-plane so it lies on the +x axis. The gap
                              between the grey and red lines is the vertical
                              component the anchor throws away.

WHERE THE NUMBERS COME FROM (`--source`)

    table   the pin table's `scene_meta` -- the exact `anchor_R`,
            `centroid_world` and `wrist_world` the demonstrations were
            captioned with. This is what training saw.
    live    recomputed here from the observed step-0 cloud through
            `anchor.anchor_from_cloud`, the same call the collector makes.

They can differ, and the difference is not a bug: the camera is eye-in-hand, so
the OBSERVED centroid depends on where the gripper happens to be looking from,
and the table's value was recorded from whatever pose that scene's episode
started at. `--source both` draws the live frame at half length alongside the
stored one so the gap is visible.

SEEING THE CLOUD INSTEAD OF THE OBJECT

`--object-alpha 0.25` fades the YCB body and turns the observed point cloud on,
which is the only way to see what the policy actually gets: a single-frame,
eye-in-hand, heavily self-occluded slice of one face, not the whole mesh the GUI
otherwise draws. Translucent is usually the more useful picture — you can see at
once whether the points lie on the surface or float off it. `--hide-object`
(alpha 0) removes the mesh entirely. Either way the body is only re-tinted,
never unloaded: it still collides and the hand still holds it.

ORDER MATTERS, and it is why this is not a one-line flag. `_get_point_states`
renders a segmentation camera when it is called, so a body faded to alpha 0
before the capture is simply not in the image and its class comes back empty.
The cloud is therefore captured at full opacity and the fade applied after, and
the opacity is restored before every capture because the tint outlives
`env.reset`.

CONTROLS
    N / P     next / previous scene (the simulator is not rebuilt)
    R         redraw
    Q         quit
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _resolve(args):
    """Fill cfg_file / pin table / d_rule from `--run`, unless given explicitly.

    Same derivation `visualize_bc_dataset.py` and `rollout_regrasp_policy.py`
    use, so "scene 1 of run 21" means the same scene, table and rule in all
    three. Explicit flags always win.
    """
    if not args.run:
        if not args.cfg_file:
            raise SystemExit("need --run, or --cfg-file for a bare scene")
        return None
    from handover_sim2real.regrasp import resolve_run

    spec = resolve_run(args.run, run_root=args.run_root)
    print(spec.describe())
    if not args.cfg_file:
        args.cfg_file = spec.cfg_file
    if not args.grasp_pin_table:
        args.grasp_pin_table = spec.pin_table
    if args.d_rule is None:
        args.d_rule = spec.d_rule
    if args.d_point_depth is None:
        args.d_point_depth = spec.d_point_depth
    return spec


def main():
    p = argparse.ArgumentParser(
        description="Static PyBullet view of a scene and its anchor frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--run", default=None,
                   help="Regrasp run name — derives --cfg-file, "
                        "--grasp-pin-table and the `d` rule from the run's "
                        "own config snapshot.")
    p.add_argument("--run-root", default=None,
                   help="where the run directories live (default: "
                        "$REGRASP_DATA/dagger_runs, then $RUNS/output/...)")
    p.add_argument("--scene", type=int, default=0,
                   help="scene index to open (default 0)")
    p.add_argument("--cfg-file", default=None,
                   help="env config, e.g. examples/pretrain_right.yaml. "
                        "Implied by --run.")
    p.add_argument("--grasp-pin-table", default=None,
                   help="pin table whose scene_meta carries anchor_R / "
                        "centroid_world / wrist_world. Implied by --run.")
    p.add_argument("--source", default="table",
                   choices=["table", "live", "both"],
                   help="where the frame comes from: the pin table's recorded "
                        "values (default), recomputed from the observed cloud, "
                        "or both drawn together.")
    p.add_argument("--grasp-idx", type=int, default=None,
                   help="also draw this slot's pinned grasp — the gripper "
                        "wireframe, and under a `grasp_offset` rule the "
                        "fingertip point `d` is measured to.")
    p.add_argument("--show-bin-sphere", action="store_true",
                   help="paint a see-through sphere around the object by BIN, "
                        "with a labelled ray down each bin axis")
    p.add_argument("--bin-sphere-radius", type=float, default=0.10)
    p.add_argument("--bin-sphere-points", type=int, default=2400)
    p.add_argument("--show-cloud", action="store_true",
                   help="draw the OBSERVED object points the centroid is the "
                        "mean of (forces the live cloud to be computed)")
    p.add_argument("--object-alpha", type=float, default=1.0,
                   help="fade the YCB object to this opacity and show its "
                        "observed point cloud: 1 leaves it alone, 0.25 makes "
                        "it translucent so the points inside it are visible, "
                        "0 makes it invisible. The body is only re-tinted — "
                        "physics, contacts and the cloud are unchanged.")
    p.add_argument("--hide-object", action="store_true",
                   help="shorthand for --object-alpha 0")
    p.add_argument("--cloud-point-size", type=float, default=4.0,
                   help="--show-cloud point size (default 4)")
    p.add_argument("--labels", action="store_true",
                   help="spell the axis labels out — `x away from hand` "
                        "rather than `x`. Off by default: the long form is "
                        "drawn over the object and hides it.")
    p.add_argument("--text-size", type=float, default=1.6,
                   help="size of the axis letters and the centroid / wrist "
                        "names (default 1.6)")
    p.add_argument("--marker-size", type=float, default=14.0,
                   help="size of the centroid and wrist dots (default 14)")
    p.add_argument("--d-rule", default=None,
                   choices=["approach_axis", "grasp_offset", "location_extent"])
    p.add_argument("--d-point-depth", type=float, default=None)
    p.add_argument("--axis-length", type=float, default=0.15,
                   help="anchor axis length in metres (default 0.15)")
    p.add_argument("--seed", type=int, default=0,
                   help="PointListener seed for the live cloud (default 0)")
    args = p.parse_args()

    _resolve(args)

    import gym
    import pybullet

    import handover          # registers the benchmark envs
    import handover_sim2real

    from handover.benchmark_wrapper import HandoverBenchmarkWrapper
    from handover_sim2real.config import get_cfg
    from handover_sim2real.policy import PointListener
    from handover_sim2real.utils import add_sys_path_from_env
    from handover_sim2real.regrasp import anchor as _rg_anchor
    from handover_sim2real.regrasp import channels as _rg_chan
    from handover_sim2real.regrasp import directions as _rg_dirs
    from handover_sim2real.regrasp import viz as _rg_viz

    add_sys_path_from_env("GADDPG_DIR")
    from experiments.config import cfg_from_file

    cfg = get_cfg()
    cfg_from_file(filename=args.cfg_file, dict=cfg, merge_to_cn_dict=True)
    cfg.SIM.RENDER = True                       # open the GUI

    pin_table = None
    if args.grasp_pin_table:
        from handover_sim2real.regrasp import load_grasp_pin_table
        pin_table = load_grasp_pin_table(args.grasp_pin_table)
        print(f"[pins] {pin_table.describe()}")

    # WHICH GIVER REFERENCE the frame is built from. A run that captioned its
    # demonstrations with `hand_centroid` and is redrawn here with `wrist` is
    # being shown a frame it never trained under, which is run 16's failure in
    # miniature — so take it from the table's own `_meta` when there is one.
    hand_ref = str((getattr(pin_table, "meta", None) or {}).get(
        "anchor_hand_ref", "wrist"))
    d_rule = _rg_dirs.DirectionRule(
        rule=args.d_rule or "approach_axis",
        **({} if args.d_point_depth is None else {"depth": args.d_point_depth}))

    if args.hide_object:
        args.object_alpha = 0.0
    if args.object_alpha < 1.0:
        # Fading the mesh without showing the points would leave an empty table.
        args.show_cloud = True

    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    panda_base_inv_tf = pybullet.invertTransform(
        cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)

    listener = PointListener(cfg, seed=args.seed)
    from collect_regrasp_demos import _point_cloud

    ids: list[int] = []

    def set_object_alpha(alpha):
        """Re-tint the YCB body to `alpha`, keeping each link's own RGB.

        The colours are read back with `getVisualShapeData` rather than
        overwritten with a constant, so a translucent object still looks like
        itself; `changeVisualShape` takes a full RGBA and passing a flat grey
        would repaint the mesh as well as fade it.

        CALL THIS *AFTER* THE CLOUD IS CAPTURED. `_get_point_states` renders a
        segmentation camera at call time (handover_env.py), so a body already
        faded to alpha 0 is not in the image and its class comes back EMPTY —
        an invisible object and no points to replace it with.

        RE-APPLIED AFTER EVERY RESET, not once at startup: a scene change can
        swap in a different YCB class and the uid goes with it. `contact_id[0]`
        is the PyBullet uid (easysim's bullet backend sets it), so it indexes
        the client directly.

        Faded, NOT removed — the object still collides and the hand still holds
        it, so the physics behind the picture is untouched.
        """
        pc = env.simulator._p
        uid = int(env.ycb.bodies[env.ycb.ids[0]].contact_id[0])
        shapes = pc.getVisualShapeData(uid)
        if not shapes:                       # no visual shapes to fade
            return 0
        for sh in shapes:
            rgb = list(sh[7])[:3]
            pc.changeVisualShape(uid, int(sh[1]),
                                 rgbaColor=rgb + [float(alpha)])
        return len(shapes)

    def clear():
        for i in ids:
            pybullet.removeUserDebugItem(i)
        ids.clear()

    def draw_gripper(pose_mat, colour, width=2.0):
        from visualize_grasps import gripper_segments
        for a, b in gripper_segments(np.asarray(pose_mat, dtype=np.float64)):
            ids.append(pybullet.addUserDebugLine(
                a.tolist(), b.tolist(), lineColorRGB=list(colour),
                lineWidth=width))

    def show(scene_idx):
        """Reset to `scene_idx` and draw. Returns the obs so nothing re-resets."""
        clear()
        obs = env.reset(idx=int(scene_idx))
        print(f"\n=== scene {scene_idx} ===")

        # FULL OPACITY FOR THE CAPTURE, every time. The tint survives
        # `env.reset` — it is a property of the body, not of the scene — so
        # without this restore the second scene you walk to is photographed
        # through the first scene's fade.
        if args.object_alpha < 1.0:
            set_object_alpha(1.0)

        # BEFORE the fade — see `set_object_alpha`. This call is what renders
        # the segmentation camera; the env then caches the result until the
        # next reset, so the dimmed object is never what the cloud is read from.
        listener.reset()
        pc5 = _point_cloud(obs, listener, panda_base_inv_tf)
        if args.object_alpha < 1.0:
            n = set_object_alpha(args.object_alpha)
            print(f"  [object] {n} visual shapes at alpha "
                  f"{args.object_alpha:g}"
                  + ("  (invisible)" if args.object_alpha <= 0 else ""))

        # ── the stored frame ────────────────────────────────────────────────
        meta = (pin_table.scene_meta.get(int(scene_idx), {})
                if pin_table is not None else {})
        aR = meta.get("anchor_R")
        cw = meta.get("centroid_world")
        have_table = aR is not None and cw is not None
        if args.source in ("table", "both") and not have_table:
            print("  [table] this scene has no scene_meta entry — falling back "
                  "to the live cloud.")

        drawn = False
        if args.source in ("table", "both") and have_table:
            aR = np.asarray(aR, dtype=np.float64)
            cw = np.asarray(cw, dtype=np.float64)
            if args.show_bin_sphere:
                _rg_viz.draw_bin_sphere(aR, cw, ids,
                                        radius=args.bin_sphere_radius,
                                        n_points=args.bin_sphere_points)
            _rg_viz.draw_anchor_frame(aR, cw, ids, length=args.axis_length,
                                      short=not args.labels,
                                      text_size=args.text_size)
            r = _rg_viz.draw_hand_anchor(meta.get("wrist_world"), cw, ids,
                                         size=args.marker_size,
                                         text_size=args.text_size)
            print(f"  [table] centroid {np.round(cw, 3)}  mode="
                  f"{meta.get('anchor_mode', '?')}  hand_ref={hand_ref}  "
                  + ("no wrist recorded (base-frame fallback)" if r is None else
                     f"horizontal wrist -> object {r * 100:.1f} cm"))
            drawn = True

        # ── the frame this observation would build right now ────────────────
        if args.source in ("live", "both") or not drawn:
            lR, lc, lmeta = _rg_anchor.anchor_from_cloud(
                pc5, obs, env, panda_base_inv_tf, cfg, hand_ref=hand_ref)
            if lR is None:
                print("  [live] the step-0 cloud holds no object points — "
                      "the camera is occluded from here, so there is no "
                      "centroid and no frame. This is a real situation, see "
                      "regrasp/channels.py.")
            else:
                # Half length when the stored frame is already up, so the two
                # are distinguishable where they nearly coincide.
                half = 0.5 if drawn else 1.0
                if args.show_bin_sphere and not drawn:
                    _rg_viz.draw_bin_sphere(lR, lc, ids,
                                            radius=args.bin_sphere_radius,
                                            n_points=args.bin_sphere_points)
                _rg_viz.draw_anchor_frame(lR, lc, ids,
                                          length=args.axis_length * half,
                                          label=not drawn,
                                          short=not args.labels,
                                          text_size=args.text_size)
                r = _rg_viz.draw_hand_anchor(
                    _rg_anchor.wrist_world(env), lc, ids,
                    size=args.marker_size, label=not drawn,
                    text_size=args.text_size)
                print(f"  [live]  centroid {np.round(lc, 3)}  mode="
                      f"{lmeta.get('mode', '?')}  "
                      + ("no wrist (base-frame fallback)" if r is None else
                         f"horizontal wrist -> object {r * 100:.1f} cm"))
                if drawn:
                    print(f"  [live vs table] centroid moved "
                          f"{np.linalg.norm(lc - cw) * 100:.1f} cm, +x "
                          f"turned "
                          f"{_rg_dirs.angle_between(lR[:, 0], aR[:, 0]):.1f} "
                          f"deg — the eye-in-hand view, not an error.")
                if not drawn:
                    aR, cw = lR, lc

        # ── the observed object points the centroid is the mean of ──────────
        if args.show_cloud:
            # DELIBERATELY NOT the centroid's cyan: the centroid has to stay
            # findable inside the cloud it is the mean of, and same-colour
            # points at a bigger size is how you lose it.
            p5 = np.asarray(pc5)
            pts = p5[p5[:, _rg_chan.CH_YCB] > 0.5][:, _rg_chan.CH_XYZ]
            if len(pts):
                w = _rg_anchor.points_to_world(
                    pts, obs, panda_base_inv_tf, cfg.ENV.PANDA_BASE_POSITION,
                    cfg.ENV.PANDA_BASE_ORIENTATION)
                ids.append(pybullet.addUserDebugPoints(
                    w.tolist(), [[0.95, 0.75, 0.30]] * len(w),
                    pointSize=float(args.cloud_point_size)))
                n_hand = int((p5[:, _rg_chan.CH_HAND] > 0.5).sum())
                print(f"  [cloud] {len(w)} object points + {n_hand} hand of "
                      f"{len(p5)} total — a single eye-in-hand frame, so this "
                      f"is one occluded face, not the mesh")
            else:
                n_hand = int((p5[:, _rg_chan.CH_HAND] > 0.5).sum())
                print(f"  [cloud] NO object points ({len(p5)} points total, "
                      f"{n_hand} of them hand). The wrist camera cannot see "
                      f"the object from the home pose in this scene — real, "
                      f"and why `object_centroid` returns None rather than a "
                      f"NaN. Try another scene with N.")

        # ── one pinned grasp, for context ───────────────────────────────────
        if args.grasp_idx is not None and pin_table is not None:
            gp = pin_table.pose(int(scene_idx), int(args.grasp_idx))
            if gp is None:
                n = pin_table.num_grasps_for(int(scene_idx))
                print(f"  [grasp] scene {scene_idx} has {n} slots — "
                      f"{args.grasp_idx} is not one of them")
            else:
                draw_gripper(gp, (0.2, 0.9, 0.2))
                b = pin_table.bin_of(int(scene_idx), int(args.grasp_idx))
                print(f"  [grasp] slot {args.grasp_idx}  bin="
                      f"{'-' if b is None else _rg_dirs.BIN_SHORT[b]}")
                if aR is not None and cw is not None:
                    dg = d_rule.of(gp, cw)
                    if dg is not None:
                        _rg_viz.draw_direction(
                            dg, cw, ids, colour=(1.0, 0.85, 0.1),
                            label=f"d ({d_rule.rule})", length=0.26)
                    if d_rule.needs_centroid():
                        off = _rg_viz.draw_grasp_point(gp, cw, ids,
                                                       depth=d_rule.depth)
                        print(f"  [grasp] centroid -> fingertips "
                              f"{off * 100:.1f} cm")
        return obs

    scene = int(args.scene)
    show(scene)
    print("\n  N / P  next / previous scene     R  redraw     Q  quit")

    # A polling loop rather than a blocking wait: PyBullet's GUI only services
    # mouse orbit while the client is alive, so sleeping on input would freeze
    # the camera the viewer exists to let you move.
    try:
        while True:
            keys = pybullet.getKeyboardEvents()
            hit = {k for k, v in keys.items() if v & pybullet.KEY_WAS_TRIGGERED}
            if ord("q") in hit:
                break
            if ord("n") in hit:
                scene += 1
                show(scene)
            elif ord("p") in hit:
                scene = max(0, scene - 1)
                show(scene)
            elif ord("r") in hit:
                show(scene)
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    env.close()


if __name__ == "__main__":
    main()
