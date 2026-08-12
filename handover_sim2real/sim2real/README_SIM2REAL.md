# Sim2Real — running the handover policy on the Franka FR3

Deploys a Phase-4 BC policy on the physical FR3 with a wrist-mounted RealSense
D435, optionally fused with a second D435 on a tripod.

There are two runners here and they drive **different policies**:

| script | policy | checkpoint |
|---|---|---|
| `policy_runner.py` | CVPR2023 GA-DDPG model | `output/cvpr2023_models/...` |
| `my_policy_runner.py` | Phase-1/4 BC policy | `checkpoint/cp2` or `cp3` |

They are **not** interchangeable — see [Why the two runners differ](#why-the-two-runners-differ).

`my_policy_runner.py` drives either BC checkpoint; they differ in how many
viewpoints fill the point cloud:

| checkpoint | run | trained cameras | val success | invocation |
|---|---|---|---|---|
| `cp2` | DAgger run 12 iter 8 | wrist only | 0.74 | `--cameras wrist` (default) |
| `cp3` | DAgger run 16 iter 16 | wrist + left + right | 0.80 | `--cameras wrist,tripod` |

Both consume the **same** `[1024, 5]` tensor — 896 object + 128 hand, `xyz` plus
two one-hot channels — because run 16's sim config deliberately keeps two point
classes. The extra cameras change which points fill those slots, not the format.

---

## Bring-up sequence

Run these in order. Each one has to work before the next is meaningful.

```bash
# 0. perception + policy only. Publishes NOTHING, robot cannot move.
python my_policy_runner.py --dry-run

# 1. homing alone, nothing else
python my_policy_runner.py --home-only

# 2. then one step per SPACE
python my_policy_runner.py --home --step-mode
```

That is cp2, wrist only. For cp3 add the second camera once its calibration
passes and the weights are fetched:

```bash
python my_policy_runner.py --policy-dir checkpoint/cp3 \
       --cameras wrist,tripod --calib-session <session> --dry-run
```

Camera serials come from `calib_config.CAMERA_SERIALS` — see
[Which camera](#which-camera-two-are-attached). Step 1 does not touch the camera
at all.

**Step 0** exercises camera → hand segmentation → point clouds → policy → target
pose without touching rosbridge. If the printed per-step `|d|` is around
0.02–0.03 m and both `obj=` and `hand=` counts are in the hundreds, the
perception stack is healthy.

**Step 1** drives the arm to the sim's episode-start pose and exits. Do this
before the policy ever has a say in where the arm goes, so you can watch the path
in isolation. It is an *interpolated* Cartesian move (2 cm / 5° per waypoint),
not a jump — but it is still a real motion, so keep the workspace clear. Check
the startup print of the `panda_hand` pose against the real flange here.

**Step 2** is the safe operating mode: the policy predicts, you see the target
previewed, and **nothing executes until you press SPACE**. Verified — 25 s of
previewing produced zero commanded steps without a keypress. Keys go to the
OpenCV window, so it must have focus.

Once you trust it, drop `--step-mode` for continuous operation, and add
`--enable-gripper` when you want it to actually grasp.

### Seeing what the policy sees

`--show-cloud` opens a live 3D view of the **literal `[1024, 5]` tensor** being
fed to the network, in the `panda_hand` frame, next to the 2D segmentation
overlay that is always on. The two answer different questions and neither
replaces the other: the overlay says whether the hand was *segmented*, the 3D
view says whether the resulting points landed in the right *place*.

```bash
python my_policy_runner.py --show-cloud --dry-run
```

You get one window per camera plus the 3D view — with `--cameras wrist,tripod`
that is three:

```
cam: wrist  [keys here]        the 2D overlay carrying the step HUD
cam: tripod                    the 2D overlay for the second camera
policy point cloud (panda_hand frame)
```

Each 2D window is captioned with that camera's own contribution
(`obj=… hand=…`, points removed by the exclusion box, and `STALE` when it fell
back to the previous frame's cloud), so a camera that has quietly stopped
contributing is visible rather than averaged into the union. **Only cameras
named in `--cameras` are opened at all** — with the default `--cameras wrist`
there is exactly one stream, no matter how many D435s are plugged in.

Drawn alongside the points: a **gripper wireframe** built from GA-DDPG's control
points (fingertips at `z = 0.105`), and the **robot exclusion box** when it is
filtering. The gripper is not decoration — the cloud is in `panda_hand`, whose
origin is invisible in a bare scatter plot, and without it "the object is 8 cm
ahead of the fingers" and "8 cm behind them" look identical. The second means
your calibration is inverted.

Press **`c`** in the 3D window to toggle colouring:

| mode | colours | what it tells you |
|---|---|---|
| by class | orange object / green hand | what the one-hot channels say — i.e. what the network thinks each point is |
| by camera | blue wrist / yellow tripod | which rig contributed each point |

Colour-by-camera is the multi-camera diagnostic. Fused into one cloud, a tripod
with bad extrinsics just looks like a slightly noisy object. Coloured by source,
the two sets should **overlap** on shared surfaces — a rigid offset between blue
and yellow *is* the calibration error, and its size is the magnitude.

Closing the 3D window does not stop the run, and the view is rate-limited
(`--cloud-update-hz`, default 10) because Open3D re-uploads the whole buffer on
every update.

**Interactivity in `--step-mode`.** One runner iteration — camera grab,
segmentation, policy forward, display — measures **3.7 Hz**, and the 3D window
was originally pumped once per iteration. A trackpad drag sampled at 3.7 Hz
barely moves the camera at all, which is why the window felt frozen rather than
merely slow. In step mode the loop is waiting for a human anyway, so each
iteration now spends `VIEWER_PUMP_S` (0.20 s) pumping the window's events before
recomputing perception: **55 Hz** measured, a 15× improvement. Any keypress
breaks the pump immediately, so SPACE is never delayed.

This is deliberately **step-mode only**. In continuous mode the loop rate *is*
the control rate, and trading it away to smooth a debug view would be the wrong
call.

**If the 3D window is ever empty while the HUD shows healthy `obj=`/`hand=`
counts**, the view is framed somewhere the cloud isn't. That was a real bug once:
Open3D scales its view from the bounding box of the geometry present when it was
*added*, and the cloud is added empty, so framing against that stale box put a
genuine wrist-camera cloud (z = 0.4–1.0 m in the hand frame) outside the
frustum — the gripper drew and the points did not. `_frame_view` now refits to
the real extent *before* applying the viewing angle.
`test_multicam_fusion.py --viz` renders and counts coloured pixels at both near
and far distances to keep it from coming back; the old code scores 0 pixels on
the far case, so the check has teeth. Dragging in the window re-frames it by
hand in any case.

### Inspecting perception on its own

`test_perception_viz.py` opens the same windows without loading a policy and
without ever publishing to a ROS topic, so it **cannot move the arm**. Use it to
judge segmentation and cloud quality before trusting a rollout.

```bash
python test_perception_viz.py --cameras wrist,tripod --calib-session <session>
python test_perception_viz.py --cameras wrist --no-ros     # no robot at all
```

Its 3D window draws **two** clouds at different point sizes:

| cloud | size | colour | what it is |
|---|---|---|---|
| policy input | large | orange object / green hand | the `[1024, 5]` tensor the network would receive |
| raw scene | small | white | every camera's full deprojected view, merged in `panda_hand` |

The white cloud is the point of the script. The coloured cloud alone cannot tell
you whether it is in the right *place* — a hand cloud rigidly displaced by a bad
extrinsic still looks like a perfectly good hand cloud. Against the white scene
(table, arm, background — geometry you recognise) a displacement is obvious, and
with two cameras **the white clouds must overlap**. Where they don't, the gap is
your calibration error.

Keys, in any OpenCV window: `c` colour by class/camera, `w` white cloud on/off,
`z`/`x` roll the view, `r` drag mode, `q` quit.
`--context-stride` and `--context-radius` control the
white cloud's density and how far out it extends (default 1.2 m around
`panda_hand`, because a tripod at 1.5 m otherwise contributes the whole room and
dominates the view scale).

**Reorienting the view.** Left-drag orbits, scroll zooms, and **`z` / `x` roll
the view** in 10° steps. Roll exists as its own key because neither built-in
Open3D control is satisfactory alone:

| mode (`r` cycles) | left-drag | drag direction |
|---|---|---|
| turntable *(default)* | orbits, up-vector **pinned to +z** — cannot roll | conventional |
| arcball | tumbles freely, roll included | **inverted** |
| rotate model | spins the geometry, not the camera | inverted |

Turntable orbits perfectly well — which is why zoom, pan and
revolve-around-a-point all work — but it pins the horizon, so a view can never
be tilted off horizontal. The arcball reaches every orientation but Open3D drags
it in the opposite sense, which reads as inverted controls. `z` / `x` give the
missing degree of freedom with an unambiguous direction: they rotate only the
camera's up-vector, leaving position and view direction untouched (verified:
eye moves 0.0 m, view direction changes 0.0°, up rotates exactly the requested
angle), so the view never jumps.

**A fixed camera needs the live robot pose.** Its extrinsics are
`inv(T_base_hand) @ T_base_color`, so `--no-ros` refuses to draw one rather than
placing its points against a fictitious arm position. The wrist camera is fine
without a robot — its extrinsics are constant.

Watch for the `STALE` marker in the 2D captions and the `*` in the console
summary. When the segmenter loses the hand, `extract_hand_object_clouds` falls
back to the previous frame's hand cloud and then crops the object around that
*stale* centroid — which can yield a very large, entirely bogus object cloud
(15k points was observed on the wrist while no hand was present). A `*` means
the observation is not fresh, whatever the point counts say.

This script uses Open3D's `O3DVisualizer` rather than the legacy `Visualizer`
that `--show-cloud` uses, because the legacy renderer has **one global point
size** for the whole scene — "small white dots behind large coloured ones" is
not expressible there.

Two O3DVisualizer quirks are worth knowing if you edit it:

* it draws a **skybox by default that renders over `set_background`**, so
  setting a background colour alone does nothing at all — `show_skybox(False)`
  is required first. The legacy viewer has no skybox, which is why the same
  one-liner works there.
* geometry colours go through Filament's **tone mapper**, so what you set is not
  what a screenshot reads back — the object orange `(255, 115, 26)` renders as
  roughly `(245, 180, 84)`. Don't colour-match rendered pixels against the
  source constants.

**Performance.** Perception runs on a worker thread and the main thread does
nothing but pump events. That is not an optimisation, it is what makes the window
usable: one perception pass takes ~70 ms, and when it shared a loop with
`run_one_tick()` the GUI got one event tick per pass, i.e. ~14 Hz. A trackpad
drag sampled at 14 Hz mostly gets dropped, which reads as "laggy and won't
rotate". Measured after the change: **82 Hz event loop** with both cameras.
Geometry is also updated in place rather than rebuilt each frame
(`--draw-hz`, default 10) — the clouds are allocated once at a fixed size, which
is why `--context-max` exists and why the white cloud is padded to it.

**If the cloud flickers or points go black**, three things caused that and are
fixed; they're worth knowing before touching this code:

* the geometry was **hidden on any frame with no data**, and `usable` goes false
  whenever the segmenter drops the hand for one frame — so the cloud blinked out
  several times a second. It now *holds* the previous cloud; staleness is still
  reported in the 2D captions and console, so nothing is concealed.
* the policy cloud was **allocated with zero (black) colours**, so a partially
  applied update rendered as black speckle. It is seeded with the object colour.
* the tensor cloud passed to `update_geometry` was a **Python temporary**, freed
  as soon as the call returned while Filament was still consuming it on the
  render thread — garbage colours. The last few frames' clouds are now retained.
  (`o3c.Tensor(ndarray)` does copy, so the numpy inputs were never the problem.)

The worker is itself throttled (15 Hz). Left unthrottled it pinned ~2.4 cores
and starved the render thread it exists to feed — the window stuttered again for
an entirely different reason. Throttling *raised* the GUI from 66 to 82 Hz and
halved CPU use, since nothing consumes snapshots faster than the draw rate.

### Keys (in the OpenCV window)

| key | action |
|---|---|
| `SPACE` | execute the previewed step (`--step-mode` only) |
| `h` | re-home mid-run |
| `q` / `Esc` | quit |

---

## Environment

Use the **`handover-rs`** conda env — it has `pyrealsense2`, `roslibpy`, and
`pytorch_lightning` (for the hand segmenter). `GADDPG_DIR` defaults to the
sibling `GA-DDPG/` checkout, so you do not need to export it.

```bash
conda activate handover-rs
cd handover_sim2real/sim2real
python my_policy_runner.py --dry-run
```

> **If you get `ModuleNotFoundError: No module named 'cv2'`** the shell resolved
> the wrong interpreter — `~/anaconda3/bin/python` is conda *base* (Python 3.8.5)
> and shadows the env's 3.10 in stale sessions. A relative path in the traceback
> (`File "my_policy_runner.py"`) is the tell, since Python ≥3.9 always prints an
> absolute one. Open a fresh terminal, or run
> `~/anaconda3/envs/handover-rs/bin/python my_policy_runner.py`.

Requires CUDA: the PointNet++ backbone has GPU-only ops and the script refuses to
start on CPU.

`hands-segmentation-pytorch/` must sit alongside `handover-sim2real/`.

---

## Checkpoints

```
checkpoint/cp1/checkpoint.ckpt   hand segmentation (HandSegModel)
checkpoint/cp2/best.pt           run 12 policy — wrist only
checkpoint/cp2/normalization.npz state/action scaling — part of the policy's definition
checkpoint/cp2/config.yaml       architecture; reconstructed, see below
checkpoint/cp3/config.yaml       run 16 architecture; reconstructed, verified
checkpoint/cp3/best.pt           run 16 iter 16 policy — wrist+left+right
checkpoint/cp3/normalization.npz run 16's own scaling — verified, not cp2's
```

`cp2/best.pt` and `normalization.npz` are md5-identical to
`output/dagger_runs/dagger4_run12/best/`. That run dir had a `config.yaml` and
the copy here did not, so it was reconstructed from the source run — trimmed to
the fields the loader reads. **Do not edit it.** Every field changes tensor
shapes, and `use_prev_act` / `drop_joint_state` in particular decide which
robot-state channels reach the network.

### cp3 — complete and verified

`cp3/config.yaml` is reconstructed from
`output/dagger_runs/dagger4_run16/iters/iter_16/config.yaml`; `best.pt` and
`normalization.npz` are copied from that run's `best/`. All three verified:

* the checkpoint **strict-loads all 86 tensors** into the policy built from the
  reconstructed config — including the six `aux_head.*` tensors, which is the
  real proof the config matches the trained architecture;
* it is **iteration 16 specifically**, not the last iteration: its
  `best_val_loss` of 0.30812913 matches only that row of `dagger_log.csv`, and
  its `epoch 14` is the best epoch inside that iteration's 25-epoch fit;
* the normalizer is **run 16's own**, not a stray copy of run 12's — different
  md5 and all four arrays differ numerically. Its `state_mean[18:21]` of
  `(0.616, -0.116, 1.490)` confirms the sim-world frame, so `T_SIMWORLD_BASE`
  applies to cp3 unchanged, and its action stds match cp2's, so the runner's
  per-step safety clamp needed no adjustment.

A normalizer is part of a policy's definition and is **per-run**: run 16 was fit
on `train_pinned_omg_wlr_ok.h5` plus 25 wlr DAgger iterations, a different
dataset from run 12's. Substituting cp2's would mis-scale every emitted action —
a failure that presents as a working policy behaving badly. The runner refuses to
start if the file is absent rather than falling back to anything.

**cp3 has an aux head, cp2 does not.** run 16 inherits `aux_head: true` from
`bc_phase4_all.yaml`. The head predicts the goal grasp in the current EE frame;
the runner never reads it, but the keys must be declared or the strict load
fails.

---

## Frames and calibration

**EE frame — measured, not assumed.** `/cartesian_pose` publishes `O_T_EE`, and
this robot's `F_T_EE` has **zero translation** with a −45° z rotation (the Franka
Hand default would be `(0, 0, 0.1034)` with that rotation). Flange + Rz(−45°) is
exactly how the URDF defines `panda_hand`, so the controller already publishes
the policy's frame. Confirmed two ways: `/cartesian_pose` matches `O_T_EE` to
4 µm, and pybullet FK of `panda_hand` at the robot's reported joints lands within
**0.06 mm**. Hence `--ee-offset-z` defaults to **0**. Only set `0.1034` if the EE
is ever reconfigured to the fingertip TCP.

That 0.06 mm agreement also means the **FR3 is kinematically identical to the
Panda** the policy was trained on, across all seven joints.

**Hand-eye — NOT calibrated.** `T_hand_cam` defaults to the *sim's* nominal wrist
mount: `t = (0.036, 0, 0.036)`, `R = Rz(+90°)`, from
`handover-sim/handover/panda.py`. Every point the policy sees is biased by
however far your real D435 mount deviates from that. This is the largest
remaining source of error. Pass a measured matrix once you have one:

```bash
python my_policy_runner.py --hand-eye T_hand_cam.npy
```

The `camera calibration/` folder solves the *other* problem — locating a **fixed**
camera in the base frame (eye-to-hand). That is the right procedure for the
tripod camera and the wrong one for the wrist; see the next section. `cp2` uses
the wrist alone, so nothing there feeds this runner today.

---

## Tripod camera calibration (eye-to-hand)

Locates a **fixed** camera in the robot base frame, producing `T_base_color.npy`.
Deployment turns that into the policy's frame per step:
`T_hand_cam = inv(T_base_hand) @ T_base_color`.

Not the same as the wrist camera, which needs *eye-in-hand* — a different
equation. This procedure applies only to a camera that does not move with the arm.

```
camera calibration/
  calib_config.py          ALL parameters — board, camera serials, thresholds
  calib_common.py          shared SE(3) / ChArUco / session / camera helpers
  generate_color_intrinsics.py
  capture_image_and_pose.py
  calibrate.py
  validate_calibration.py
  sessions/<name>/         one folder per capture set
      images/NNNN.png
      robot_poses.json
      color_intrinsics.json
      T_base_color.npy
      T_gripper_board_ref.npy
```

Every script takes `--session NAME`. Capture defaults to today's date; the others
default to the newest session. **A session is the unit of a calibration** — one
camera position, one set of captures, one result. Start a new one whenever the
camera moves, and nothing can silently blend two setups.

Run from `camera calibration/` in the `handover-rs` env (see
[Environment](#environment), including the interpreter-shadowing note).

### What the maths needs from you

Two unknowns are solved at once: the camera pose `X = T_base_cam`, and the board's
mounting `T_gripper_board`. Writing the loop closure for two captures and
eliminating the mounting gives `AX = XB`, where `A` is how the *gripper* moved and
`B` is how the *board appeared to move*. So:

* **Board position/orientation on the wrist — free.** It cancels, and is never
  measured. It falls out as a by-product (`T_gripper_board_ref.npy`), and its
  spread across captures *is* the accuracy metric.
* **Board square size — NOT free.** It sets metric scale and passes straight
  through: on this rig a **2 % size error moved the camera 9 mm**, twice the whole
  calibration residual.
* **Rotation diversity — mandatory.** Pure translation leaves `A` and `B` with
  identity rotation, and the equation then says nothing about `X`'s rotation.
  `calibrate.py` prints the diversity it got and warns below 20° median.

### 1. Physical setup (once, before any command)

1. **Print the ChArUco board** at exactly 100 % scale — no "fit to page", no
   margin scaling. Defaults are 8×8 squares, `DICT_5X5_50`, square 21 mm, marker 16 mm.
2. **Measure it with calipers.** Span 8 squares and divide by 8; do not trust the
   nominal 21 mm. Printers rescale silently. Put the result in
   `calib_config.py` → `BoardSpec.square_length_m` (and `marker_length_m`).
3. **Mount the board rigidly to the wrist** — bolted, or firmly taped to the
   flange or a fingertip. Any pose is fine. It must not shift *during* the
   session; a board that slips mid-capture corrupts every pose after it.
4. **Place and lock the tripod.** From here the camera must not move — not
   nudged, not re-aimed, not refocused. Any bump invalidates the result; start a
   new session and re-capture. Aim it so the board stays visible across a wide
   range of arm poses.
5. **Confirm you can pose the robot** — hand-guiding (brakes released, enabling
   device held) is easiest; jogging works too.

### 2. Identify the cameras (once)

```bash
python calib_common.py                      # list serials, USB link, sessions
python calib_common.py --preview            # live window per camera, serial overlaid
```

`--preview` opens one window per attached camera with its serial burnt into the
image, so you can see which is which instead of guessing. **Jog the robot — the
wrist camera is the view that moves.** Colour only, and it falls back through
lower modes, so a camera on a marginal link still shows something. `q`/`Esc`
quits; `--seconds N` auto-closes.

A camera that streams nothing is reported with its link type, e.g.

```
[preview] 045322075902: NO MODE WOULD STREAM (usb 2.1) — Frame didn't arrive within 3000
```

which is not a preview bug: a D435 on a **USB 2** link advertises modes it cannot
stream at all. Reseat it on a USB3 port with the Intel cable and re-run.

Put the serials into `calib_config.py` → `CAMERA_SERIALS`, so every script can be
told `--role tripod` rather than a number. They start as `None` on purpose: with
two D435s attached, calibrating the wrist camera while aiming the tripod at the
board raises no error anywhere and yields a confidently wrong matrix.

### 3. Export intrinsics into the session

```bash
python generate_color_intrinsics.py --session <NAME> --role tripod
```

Resolution comes from `calib_config.STREAM` (640×480). Intrinsics are
**resolution-specific**: capture, calibration and deployment must all match it.
`usb` must read `3.x` — a D435 on a 2.1 link advertises modes it cannot stream.

### 4. Capture 15–20 pose/image pairs

```bash
python capture_image_and_pose.py --session <NAME> --role tripod
```

Per capture: **move the arm → let it come to rest → press `s`**. `q`/`Esc` quits.
The overlay shows live pose freshness and detected ChArUco corner count, and a
shot the solver would later discard is **refused at capture time** rather than
silently lost.

Aim for:

* **Large rotation changes between poses** — this is what conditions the solve.
  A healthy set spans tens of degrees pairwise (session_01: 11.7–98.5°, median
  52.9°). Roll/pitch/yaw the wrist; do not merely translate.
* Board fully in frame, reasonably large, not edge-on.
* Vary distance and where it falls in the image.
* Hold still before pressing `s` — motion blur ruins corner detection.
* 15–20 pairs. `calibrate.py` refuses below `MIN_SAMPLES` (10).

Re-running against an existing session appends, and says so. That is right for
resuming an interrupted set, wrong after the camera has moved — use a new name.

### 5. Solve

```bash
python calibrate.py --session <NAME>
python calibrate.py --session <NAME> --compare    # all solvers, with residuals
```

Method comes from `calib_config.HAND_EYE_METHOD`, default `DANIILIDIS`. Measured
on session_01 by the `T_gripper_board` consistency residual:

| solver | pos RMS | rot RMS |
|---|---|---|
| TSAI | 8.14 mm | 1.50° |
| PARK / HORAUD | 4.80 mm | 0.99° |
| ANDREFF | 4.94 mm | 0.99° |
| **DANIILIDIS** | **4.56 mm** | **0.99°** |

All agree on camera position to ~4 mm, so the choice is accuracy, not
correctness. Tsai solves rotation then translation, so rotation error feeds into
translation; the others solve both jointly.

### 6. Validate — do not skip

```bash
python validate_calibration.py --session <NAME>     # exit 1 if out of spec
```

The board is rigid on the wrist, so `T_gripper_board` must solve to the same pose
from every capture. Its spread is the true error. Thresholds live in
`calib_config.py`; per-image rows are printed so a bad capture is identifiable.

Progress so far, and what each fix did:

| metric | threshold | session_01 | `test`, before | `test`, after board re-measure |
|---|---|---|---|---|
| gripper→board translation | 3.0 mm | 4.06 mm | 2.76 mm | **1.89 mm** ✓ |
| gripper→board rotation | 0.5° | 0.85° | 0.658° | 0.658° ✗ |
| reprojection RMSE | 1.0 px | 3.29 px | 2.36 px | 1.46 px ✗ |

Re-measuring the board with calipers fixed translation and most of the
reprojection, and left rotation **bit-for-bit unchanged** — which is the
diagnostic, not a disappointment. Board scale is a similarity transform: it moves
points along rays and cannot rotate anything.

**The remaining rotation error is board tilt.** Ruled out by measurement: corner
refinement (all four methods within noise, 0.658° vs 0.664°) and intrinsics (pure
PnP reprojection 0.405 px on factory intrinsics vs 0.371 px self-calibrated, no
radial trend). What does correlate is tilt, at −0.55:

| board tilt | rotation error |
|---|---|
| ≥ 39° (6 images) | 0.13 – 0.23° |
| ≤ 28° (6 images) | 0.39 – **1.99°** |

Re-solving on the tilt ≥ 30° captures alone gives **0.173°**, using fewer than
half the images. A planar target viewed square-on barely constrains its own
out-of-plane rotation, so corner noise becomes pose error and hand-eye inherits
it. The capture overlay now shows live tilt and `calibrate.py` warns when most
captures are flat.

**On the 1.0 px reprojection threshold:** even the best-tilted subset stays near
1.43 px while pure PnP is 0.4 px. That ~1 px gap is robot-side — FK absolute
accuracy and board-mount rigidity — not optical. The threshold may simply be
optimistic for this rig; translation and rotation are the numbers that reflect
what you control.

### 7. Redo the calibration when

* the tripod is moved, bumped or re-aimed — **the common case**;
* the camera is re-seated or swapped;
* the capture resolution changes (intrinsics are per-resolution).

Re-mounting the board *after* a finished calibration does **not** invalidate
`T_base_color.npy` — the mounting cancelled out. It only matters that the board
stayed put *within* a capture session.

---

## Two cameras: fusing wrist + tripod (cp3)

`pointcloud_multicam.py` is the real-robot counterpart of the simulator's
multi-camera fusion. Run it once the tripod calibration passes:

```bash
python my_policy_runner.py --policy-dir checkpoint/cp3 \
       --cameras wrist,tripod --calib-session <session> --home --step-mode
```

Check the plumbing first, without robot or GPU:

```bash
python test_multicam_fusion.py          # extrinsic chains, tensor layout, exclusion box
python test_multicam_fusion.py --live --calib-session <session>
```

### What it does

Per camera: segment the hand, deproject to the camera frame, label non-hand
points near the hand centroid as object, then transform into `panda_hand`. Then
concatenate **per class across cameras** and sample 896 object + 128 hand.

That mirrors `handover_env._get_point_states` + `PointListener` step for step,
including the deliberate absence of per-camera balancing — the sim concatenates
raw, so a nearer view legitimately dominates. Both cameras run at 640×480 here,
so that ratio is preserved. Use `--per-camera-cap` only if you run them at
different resolutions.

The wrist camera's extrinsics are constant, so its cloud needs no robot pose and
cannot be corrupted by a stale one. The tripod's chain is
`inv(T_base_hand) @ T_base_color`, so it inherits the error of **both** the
hand-eye calibration and the reported EE pose. That is why the wrist stays the
anchor view and `--cameras` refuses to drop it.

### Two honest caveats

**Two cameras against three.** cp3 was trained on wrist + left + right — two
elevated side views symmetric about the handover point. You have wrist + one
tripod. Even with a perfect calibration this is an input-distribution change, not
just a noisier observation: a whole viewpoint the network learned to rely on is
absent, and the union it sees is correspondingly sparser and one-sided. Treat
cp3-on-two-cameras as something to measure, not to assume. If it underperforms
cp2, the missing view is the first suspect — a second tripod aimed from the
opposite side is the fix, not more tuning.

**No segmentation oracle.** In sim, "which points are the object" is a lookup by
body id, and `COMPUTE_ROBOT_POINT_STATE: False` excludes the arm for free. Here
the object is defined *negatively* — non-hand points within a radius of the hand
centroid — so the robot's own gripper gets labelled "object" as it closes, a
class the training data never contained. `ROBOT_EXCLUSION` removes a box around
the gripper using the pose we already know exactly. Its ceiling is `z = +0.02` in
`panda_hand`, which clears the hand housing and wrist while **sparing the finger
volume** where a grasped object sits (fingertips are at `z = 0.075…0.105`) — a
box that swallowed the fingers would delete the object at exactly the moment the
side view exists to see it. `--no-robot-exclusion` disables it.

**Third caveat, found while testing: hand segmentation at range.** `HandSegModel`
resizes every input to 256×256 before inference. From the wrist at ~0.4 m a hand
fills a large part of the frame and survives that downscale comfortably. From the
tripod at ~1.5 m the same hand is a small patch, and after the resize it is
smaller still. In a live capture with a bare forearm visible in the tripod view,
the segmenter returned an **entirely empty mask** — the hand itself was out of
frame there, so this is not yet evidence of failure, but it is the failure mode
to watch. If `tripod:o0/h0` persists in the overlay while a hand is plainly in
that camera's view, the segmenter is the cause, not the geometry. The tripod's
`min_hand_points` floor is already lowered to 40 (from the wrist's 100) for this
reason; the real fix is to move the tripod closer or crop-and-upscale the region
of interest before segmenting.

### USB bandwidth

Two D435s streaming 640×480 colour + depth at 30 fps can exceed what one USB3
controller reliably carries. **On this machine both started and streamed
together**, so no workaround is needed today. If a second camera ever starts but
no frames arrive, that is bandwidth, not the cable — put the cameras on
**separate controllers** (not merely separate ports) or drop `--camera-fps` to
15. The runner's error message says so when more than one camera is open.

---

## Camera

### Which camera (two are attached)

With the wrist D435 and the tripod D435 both plugged in, librealsense binds
whichever enumerates first. Which serial is which is therefore load-bearing, and
it has **one definition** on this machine:

```python
# camera calibration/calib_config.py
CAMERA_SERIALS = {"tripod": "825312073923", "wrist": "045322075902"}
```

Every script reads it from there — the calibration tools, and now the runner via
`build_rigs`. **This assignment has been confirmed visually**: opening both by
serial, `045322075902` gives a close top-down view of the optical table (wrist)
and `825312073923` gives a wide view of the lab with the FR3 at the right of
frame (tripod). Re-confirm after any re-cabling:

```bash
cd "camera calibration" && python calib_common.py --preview && cd ..
```

`--preview` shows a live window per camera with its serial overlaid; jog the
robot and the wrist one is the view that moves. **Quote the serials** — one
beginning with `0` is not a valid Python integer literal.

`--camera-serial` overrides the wrist entry for the single-camera case only; with
`--cameras wrist,tripod` it is refused, because one serial cannot name two
devices. Getting the assignment wrong is silent: the policy simply receives a
viewpoint it was never trained on, geometrically wrong with no error. If instead
a camera on a bad USB link is opened you get `Frame didn't arrive`, which reads
like a hardware fault but is really the wrong device.

### Modes

Defaults are **640×480 for both** color and depth at 30 fps. The D435 offers
424×240 color but **no 424×240 depth** (its depth modes start at 256×144 /
480×270), so the 424×240 pair `policy_runner.py` uses fails device-side with a
bare `Couldn't resolve requests`. Color and depth are separate streams aligned
afterwards, so they need not match: `--depth-width` / `--depth-height` are
independent.

If the modes are accepted but no frames arrive, it is a link fault, not a config
one. Check the USB link type — `calib_common.py` prints it for every attached
device, or directly:

```python
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.serial_number),
          d.get_info(rs.camera_info.usb_type_descriptor))   # want 3.x, not 2.1
```

A D435 enumerating at **2.1** (charge-only cable, USB2 port, or a hub) advertises
modes it then cannot stream *at all* — every resolution fails, even 6 fps. Use
the Intel USB3-C cable straight into a blue USB3 port. On a good link it streams
640×480 at ~30 fps with ~67% valid depth.

---

## Gripper

`--enable-gripper` publishes `franka_gripper/GraspActionGoal` onto
`/franka_gripper/grasp/goal` (roslibpy 2.0 dropped its actionlib client, and
actionlib is plain topics underneath). Width feedback is read from
`/franka_gripper/joint_states`.

**The `franka_gripper` node must be running.** If `/franka_gripper/grasp/goal` is
not advertised, goals go nowhere silently. This does not block anything else:
with no width feedback the runner assumes "open", which is correct throughout the
approach, and the policy's gripper bit ends the episode either way.

Defaults: width 0.0, force 20 N, speed 0.05, epsilon 0.04/0.04.

---

## Safety

| limit | value | what it does |
|---|---|---|
| per-step translation | 5 cm | caps one policy step |
| per-step rotation | 20° | caps one policy step |
| workspace | x ≥ 0, z ≥ 0 | never behind the base or below the table |
| settle tolerance | 5 mm / 3°, 2 s timeout | waits for convergence before the next observation |
| episode length | 50 steps | `--max-steps` |

The step caps are guards, not shapers: the policy's `action_std` is 0.016–0.021 m
and 0.06–0.08 rad, so normal motion never reaches them. A clamped step prints
`CLAMPED` — if you see that often, something is wrong upstream.

**Step-and-settle is deliberate.** The policy is single-frame Markov, trained on
~2 cm steps where the sim robot fully reached each waypoint before the next
observation. Streaming targets at camera rate would feed it mid-motion states it
never saw in training. An impedance controller has steady-state error and may
never hit tolerance, hence the timeout.

Homing targets `ENV.PANDA_INITIAL_POSITION`, joint config
`(0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785)` with fingers at 0.04. If you have
a joint-space controller, commanding those joints directly is the **safer** way
to home than the Cartesian interpolation used here.

---

## Why the two runners differ

Three things changed between the CVPR model and this one. Each produces
plausible-looking garbage rather than an error if carried over:

1. **Point cloud order and labels.** cp2 wants object points first — 896 tagged
   `[x,y,z,1,0]`, then 128 hand points `[x,y,z,0,1]`, 1024 total.
   `build_policy_point_tensor` in `pointcloud_pipeline.py` emits the opposite
   (hand first, labels swapped); that is correct for the CVPR model. This runner
   uses its own `build_bc_point_tensor`.
2. **No task-space action scaling.** GA-DDPG rescales through `PandaTaskSpace6D`.
   cp2's targets came from `convert_target_joint_position_to_action`, a raw SE(3)
   delta between FK poses, already in metres and radians.
3. **Only 8 of 32 robot-state channels reach the network.** Run 12 sets
   `drop_joint_state=true` and `use_prev_act=false`, so `_select_robot_state`
   keeps `rs[18:26]` — EE pose plus gripper — and discards the joint block and
   previous action. The runner fills exactly those and asserts the config agrees.

Observation contract, verified against `train_pinned_omg_ok.h5` attrs and
`PointListener`:

```
point_cloud  [1024, 5]  xyz + ycb_flag + hand_flag, panda_hand frame
robot_state  [32]       ...ee_xyz(3)+ee_wxyz(4)+gripper_norm(1)...  (EE pose in BASE frame)
action       [7]        dpos(3)+deuler(3) in panda_hand, + gripper bit (1 = open, 0 = CLOSE)
```

---

## Status

Verified on hardware: camera streaming, hand segmentation, cloud extraction,
policy forward pass, action scale, and that `--step-mode` executes nothing
without a keypress.

Verified offline (`test_multicam_fusion.py`): both camera kinds place a known
world point at the same spot in `panda_hand`; the fused tensor is `[1024, 5]`
with object rows first and channel 3 hot; under-full classes pad by repetition
rather than with zeros; the exclusion box drops the hand housing and spares the
finger volume. `cp3/config.yaml` builds a policy differing from cp2's by exactly
the six aux-head tensors.

Not yet exercised against the live robot: `/equilibrium_pose` publishing, the
settle loop, homing, and the gripper goal path.

Also verified offline: the 3D viewer renders both colour modes, and per-point
camera provenance survives the 896/128 sampling with **zero mislabelled rows**
(checked against two clouds made disjoint in x, so a mix-up cannot hide).

**cp3 runs end to end**: loaded, both cameras fused, three policy steps at
3.4–4.0 cm, clean shutdown, under `--dry-run --show-cloud`.

Blocked / outstanding:

* The tripod calibration does not yet pass rotation; re-capture with the board
  tilted ≥ 30°. Until then the tripod's points are geometrically untrustworthy
  even though the plumbing works.
* The **wrist** camera is still uncalibrated (`T_hand_cam` is the sim's nominal
  mount). This is the largest remaining deployment error and it affects cp2 and
  cp3 alike.
* `franka_gripper`'s action server was not advertised when last probed, so
  `--enable-gripper` would send goals nowhere.
