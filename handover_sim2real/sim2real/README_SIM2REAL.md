# Sim2Real — running the handover policy on the Franka FR3

Deploys a Phase-4 BC policy on the physical FR3 with a wrist-mounted RealSense
D435, optionally fused with a second camera on a tripod.

Two runners, **different policies, not interchangeable** (see
[Why the two runners differ](#why-the-two-runners-differ)):

| script | policy | checkpoint |
|---|---|---|
| `policy_runner.py` | CVPR2023 GA-DDPG | `output/cvpr2023_models/...` |
| `my_policy_runner.py` | Phase-1/4 BC | `checkpoint/run12`, `run16`, `run19` |
| `my_regrasp_policy_runner.py` | regrasp BC, direction-conditioned | `checkpoint/regrasp_run9/`, `regrasp_run11/` (26 iterations each) |

`my_regrasp_policy_runner.py` **imports** `my_policy_runner` rather than copying
it — see [Regrasp](#regrasp-grasping-from-a-commanded-direction).

| `--run` | run / iter | trained cameras | val | invocation |
|---|---|---|---|---|
| `run12` | run 12 iter 8 | wrist only | 0.74 | `--run run12 --cameras wrist` (default) |
| `run16` | run 16 iter 16 | wrist + left + right | 0.80 | `--run run16 --cameras wrist,tripod` |
| `run19` | run 19 iter 19 | **right camera only** | 0.72 | `--run run19 --cameras tripod` |

`--run NAME` is shorthand for `--policy-dir checkpoint/NAME`. Folders are named
after the DAgger run because that is the only name that says what a policy was
trained on. `checkpoint/cp1` keeps its name — it is the hand *segmentation*
model, not a policy.

**`run19` matches a tripod-only rig**: trained on one fixed side camera, no wrist
view. `run16` is stronger in sim but has never seen fewer than three viewpoints.
All three consume the same `[1024, 5]` tensor — 896 object + 128 hand, `xyz` plus
two one-hot channels. Extra cameras change which points fill the slots, not the
format.

---

## Bring-up sequence

Each step must work before the next is meaningful.

```bash
# 0. perception + policy only. Publishes NOTHING, robot cannot move.
python my_policy_runner.py --dry-run

# 1. homing alone
python my_policy_runner.py --home-only

# 2. one step per SPACE
python my_policy_runner.py --home --step-mode

# two cameras, once the tripod calibration passes
python my_policy_runner.py --run run16 --cameras wrist,tripod \
       --calib-session <session> --dry-run
```

Step 0 is healthy when per-step `|d|` is 0.02–0.03 m and both `obj=` and `hand=`
are in the hundreds. Step 1 is a real Cartesian motion — keep the workspace
clear, and check the printed `panda_hand` pose against the real flange.

**`--dry-run` with a fixed camera still needs the robot.** A fixed camera's chain
is `inv(T_base_hand) @ T_base_color`, so without a pose there is no `panda_hand`
to place it in — and substituting identity does not make the cloud *stale*, it
puts it in the **base** frame while the gripper wireframe and exclusion boxes are
still drawn at the origin of the frame the cloud is meant to be in. Everything
then renders about 59 cm low and 48 cm back, i.e. **the boxes appear below the
table**, which reads as a calibration fault and is not one. So a dry run
subscribes to the pose when any fixed camera is configured. It stays a dry run in
the sense that matters: `/equilibrium_pose` is never advertised, and the gripper
is never armed. Wrist-only dry runs need no robot at all.

`--home` homes immediately with no prompt. Windows come up **un-armed**: press
`s` to start. A CLOSE or `--max-steps` ends the episode but not the program —
perception and the cloud keep running so the deciding frame stays inspectable.
`q` is the only way out.

---

## Arguments (`my_policy_runner.py`)

### Policy and connection

| flag | type / default | meaning |
|---|---|---|
| `--run NAME` | `run12` | policy by DAgger run; shorthand for `--policy-dir checkpoint/NAME` |
| `--policy-dir DIR` | — | run dir with `config.yaml`, `normalization.npz`, `best.pt` |
| `--ckpt PATH` | `<policy-dir>/best.pt` | explicit weights |
| `--hand-seg-ckpt PATH` | `checkpoint/cp1/checkpoint.ckpt` | hand segmenter; unused under `--segmentation sam2` |
| `--rosbridge-host H` | `172.16.0.7` | rosbridge address |
| `--rosbridge-port N` | `9090` | rosbridge port |
| `--dry-run` | off | run perception + policy, **publish nothing**. Still *subscribes* to the pose when a fixed camera is in use — see below |

### Motion and control

| flag | type / default | meaning |
|---|---|---|
| `--control {settle,rate}` | `settle` | `settle` blocks until the arm arrives; `rate` publishes once and dwells. See [Control mode](#control-mode-settle-vs-rate) |
| `--rate-hz N` | `6.67` | policy rate for `--control rate` (the paper's 0.15 s) |
| `--step-mode` | off | preview each step, execute only on `SPACE` |
| `--max-steps N` | `50` | episode length |
| `--step-tol-frac F` | `0.6` | arrival tolerance as a fraction of the step |
| `--settle-timeout S` | `3.0` | give up waiting for convergence |
| `--no-creep` | off | old multi-pass step motion (3 commands/step) |
| `--no-droop-compensation` | off | disable the standing-droop lead. **Deadlocks `--control rate`** |

### Homing

| flag | type / default | meaning |
|---|---|---|
| `--home` | off | drive to the sim's episode-start pose first |
| `--home-only` | off | home and exit |
| `--home-speed N` | `0.08` | Cartesian speed of the streamed path, m/s |
| `--home-tol N` | `0.02` | how close to home is close enough, m. Tighter than the arm can land makes the refine hunt |
| `--home-stepwise` | off | old behaviour: settle on every 2 cm waypoint |

### Gripper

| flag | type / default | meaning |
|---|---|---|
| `--enable-gripper` | off | act on a CLOSE. Needs `load_gripper:=True` |
| `--home-gripper` | off | homing goal at startup (implies `--enable-gripper`). Once per power cycle; fingers fully open and close |

### Frames

| flag | type / default | meaning |
|---|---|---|
| `--hand-eye PATH` | sim nominal | 4×4 `T_hand_cam` for the wrist camera. **Not calibrated by default** |
| `--ee-offset-z N` | **read from `F_T_EE`** | `panda_hand` → published frame. Pass a value only to override the robot; a mismatch warns loudly |

### Cameras

| flag | type / default | meaning |
|---|---|---|
| `--cameras LIST` | `wrist` | comma-separated roles, e.g. `wrist,tripod` |
| `--calib-session NAME` | — | session providing `T_base_color.npy`; required for any fixed camera |
| `--camera-model {d435,d435i,d415,d455}` | auto | override body detection. Sets the near-depth floor only |
| `--serial ROLE=SERIAL` | — | override one role without editing `calib_config.py` |
| `--camera-serial S` | — | wrist serial, single-camera runs only |
| `--camera-width N` `--camera-height N` | `640` / `480` | colour stream |
| `--depth-width N` `--depth-height N` | `640` / `480` | depth stream (aligned to colour afterwards) |
| `--camera-fps N` | `30` | drop to 15 if two cameras share a USB controller |

### Segmentation

| flag | type / default | meaning |
|---|---|---|
| `--segmentation {hand-net,sam2}` | `hand-net` | `sam2` **measures** the object mask instead of deriving it. See [Promptable segmentation](#promptable-segmentation-sam2--grounding-dino) |
| `--seg-hand-prompt TEXT` | `"a hand."` | sam2 only |
| `--seg-object-prompt TEXT` | `"an object."` | sam2 only. **Name the thing** — `"a blue mug."` is far more reliable |
| `--sam2-model NAME` | `sam2.1_hiera_tiny` | `tiny` / `small` / `base_plus` / `large` |
| `--no-sam2-autocast` | off | run SAM2 in fp32 (roughly doubles encoder cost) |
| `--wrist-seg-px N` | `256` | cp1 input size, wrist. Ignored by sam2 |
| `--fixed-seg-px N` | `384` | cp1 input size, fixed cameras. Ignored by sam2 |
| `--hand-margin-px N` | `5` wrist / `4` fixed | pixels around the hand mask belonging to neither class. `0` restores the flicker |

### Object class filters

| flag | type / default | meaning |
|---|---|---|
| `--no-robot-exclusion` | off | keep points inside the gripper box in the object class |
| `--finger-boxes {split,span,off}` | `split` | which finger volume leaves the object class. See [Excluding the gripper fingers](#excluding-the-gripper-fingers) |
| `--no-cluster` | off | object = crop sphere alone, tight radii restored |
| `--no-arm-rejection` | off | keep object-class blobs behind the hand |
| `--arm-offset N` | `0.07` | how far behind the hand a blob must sit to be forearm |
| `--arm-lateral N` | `0.18` | how far off the hand→base axis a point may sit and stay object |
| `--arm-below N` | `0.10` | reject object points this far below the hand. `999` disables |
| `--per-camera-cap N` | — | cap points per camera; only needed at mixed resolutions |

### 3D viewer

| flag | type / default | meaning |
|---|---|---|
| `--show-cloud` | off | open the 3D window. ~200 ms/iteration in step mode |
| `--cloud-update-hz N` | `10.0` | geometry re-upload rate |
| `--context-stride N` | `6` | white raw-scene cloud subsampling |
| `--context-radius N` | `1.2` | clip the white cloud this far around `panda_hand` |
| `--context-max N` | `30000` | hard cap; allocated once, cannot grow |
| `--no-context` | off | start with the white cloud hidden (`w` toggles) |

### Keys (any window, including the 3D one)

| key | action |
|---|---|
| `t` | **STOP** — freeze the arm immediately, from anywhere |
| `SPACE` | execute the previewed step (`--step-mode`, OpenCV window only) |
| `s` | start the policy; after an episode ends, start another |
| `h` | re-home mid-run |
| `q` / `Esc` | quit |
| `c` | colour the cloud by class vs by source camera |
| `w` | white raw-scene cloud on/off |
| `z` / `x` | roll the view |
| `r` | drag mode: turntable → arcball → rotate model |
| `n` | re-seed the tracker (`test_perception_viz.py` only) |

The last five need `--show-cloud`.

**`t` is a soft stop, not an emergency stop.** `/equilibrium_pose` is the only
interface, so the strongest available action is re-commanding the equilibrium at
the measured pose: the error goes to zero, the controller stops pulling, the arm
holds. It does not brake, does not go rigid, and still sags by the droop. The
button on the wall is the emergency stop. It works mid-motion because the
blocking loops poll for it — measured, a `t` 0.15 s into a 20 cm move returns at
0.16 s with the arm stopped after 46 mm. It also cancels any gripper goal in
flight.

---

## Environment

Use the **`handover-rs`** conda env (`pyrealsense2`, `roslibpy`,
`pytorch_lightning`). `GADDPG_DIR` defaults to the sibling `GA-DDPG/`.
`hands-segmentation-pytorch/` must sit alongside `handover-sim2real/`. CUDA is
required — the PointNet++ backbone has GPU-only ops.

```bash
conda activate handover-rs
cd handover_sim2real/sim2real
python my_policy_runner.py --dry-run
```

> `ModuleNotFoundError: No module named 'cv2'` means the shell resolved conda
> *base* (Python 3.8.5), which shadows the env's 3.10 in stale sessions. A
> relative path in the traceback is the tell. Open a fresh terminal or run
> `~/anaconda3/envs/handover-rs/bin/python my_policy_runner.py`.

---

## Checkpoints

```
checkpoint/cp1/checkpoint.ckpt   hand segmentation (HandSegModel — NOT a policy)
checkpoint/sam2/                 SAM 2.1 weights, if --segmentation sam2
checkpoint/run12/                run 12 — wrist only
checkpoint/run16/                run 16 iter 16 — wrist + left + right
checkpoint/run19/                run 19 iter 19 — right camera only
checkpoint/regrasp_run9/         REGRASP run 9 — approach_axis, all 26
                                 iterations plus best/ and last/. See its README.
checkpoint/regrasp_run11/        REGRASP run 11 — grasp_offset, same layout.
                                 A DIFFERENT command, not another tuning.
   best.pt / normalization.npz / config.yaml / source.txt
```

**A normalizer is per-run and part of the policy's definition.** Substituting
another run's mis-scales every action — a failure that presents as a working
policy behaving badly. The runner refuses to start without it. **Do not edit
`config.yaml`**: every field changes tensor shapes, and `use_prev_act` /
`drop_joint_state` decide which robot-state channels reach the network.

`run16` verified: strict-loads all 86 tensors including the six `aux_head.*`
(the real proof the config matches the trained architecture); `best_val_loss`
0.30812913 matches only iteration 16's row; the normalizer is run 16's own
(distinct md5, all four arrays differ). `run19` likewise, `best_val_loss`
0.3420485 at iteration 19. run16/run19 have an aux head, run12 does not — the
runner never reads it, but the keys must be declared or the strict load fails.

---

## Frames and calibration

**EE frame — read from the robot every run, not assumed.** `/cartesian_pose`
publishes `O_T_EE`, and which frame that *is* depends on the configured end
effector. The runner reads `F_T_EE` off `/franka_state_controller/franka_states`
at startup and derives `--ee-offset-z` from it.

This used to be a constant `0`, and was right: `F_T_EE` was `(0,0,0)` with a −45°
z rotation, which is exactly how the URDF defines `panda_hand` — confirmed to
4 µm against `O_T_EE` and 0.06 mm against pybullet FK. (That agreement also
means the **FR3 is kinematically identical to the Panda** across all seven
joints.) The configured EE later became the **Franka Hand TCP**, `F_T_EE` became
`(0, 0, 0.1034)`, and the constant went stale in silence.

**The symptom is worth recognising, because nothing errors.** Observation and
command are both in the wrong frame and stay self-consistent; the one thing that
is not is where the cloud sits *relative to the gripper*. An object just beyond
the fingertips (`panda_hand` z ≈ 0.16) renders at z ≈ 0.06 — between the finger
boxes and just clear of the housing box — and the policy closes roughly 10 cm
short. Seen on hardware exactly so.

**Wrist hand-eye — NOT calibrated.** `T_hand_cam` defaults to the sim's nominal
mount (`t = (0.036, 0, 0.036)`, `R = Rz(+90°)`). Every point is biased by however
far the real mount deviates. **This is the largest remaining source of error.**
Pass a measured matrix with `--hand-eye T_hand_cam.npy`.

`camera calibration/` solves the *other* problem — a **fixed** camera in the base
frame (eye-to-hand). Right for the tripod, wrong for the wrist.

---

## Tripod camera calibration (eye-to-hand)

Produces `T_base_color.npy`; deployment turns it into
`T_hand_cam = inv(T_base_hand) @ T_base_color` per step.

```
camera calibration/
  calib_config.py        ALL parameters — board, serials, thresholds
  calib_common.py        shared SE(3) / ChArUco / session / camera helpers
  generate_color_intrinsics.py   capture_image_and_pose.py
  calibrate.py                   validate_calibration.py
  sessions/<name>/       images/ robot_poses.json color_intrinsics.json
                         camera.json T_base_color.npy T_gripper_board_ref.npy
```

**A session is the unit of a calibration** — one camera position, one set of
captures, one result. Start a new one whenever the camera moves. Run from
`camera calibration/` in the `handover-rs` env.

### What the maths needs

`AX = XB` solves the camera pose and the board's wrist mounting at once.

* **Board position on the wrist — free.** It cancels; its spread across captures
  *is* the accuracy metric.
* **Board square size — not free.** It sets metric scale: a **2 % size error
  moved the camera 9 mm**, twice the whole residual. Measure with calipers, span
  8 squares and divide.
* **Rotation diversity — mandatory.** Pure translation says nothing about `X`'s
  rotation. `calibrate.py` warns below 20° median.

### Procedure

```bash
python calib_common.py --preview                                  # 1. identify cameras
python generate_color_intrinsics.py --session <NAME> --role tripod # 2. intrinsics
python capture_image_and_pose.py --session <NAME> --role tripod    # 3. 15-20 pairs
python calibrate.py --session <NAME>                               # 4. solve
python validate_calibration.py --session <NAME>                    # 5. exit 1 if out of spec
```

Before any of it: print the ChArUco board at exactly 100 % scale (8×8,
`DICT_5X5_50`), measure it, put the result in `BoardSpec.square_length_m`; mount
it rigidly to the wrist; place and **lock** the tripod; confirm you can pose the
robot.

Per capture: move the arm → let it rest → press `s`. Aim for large rotation
changes between poses (session_01 spanned 11.7–98.5°, median 52.9°), the board
fully in frame and **tilted ≥ 30°**, near the middle of the image, varied
distance. `calibrate.py` refuses below 10 samples.

`capture_image_and_pose.py` writes `camera.json` — role, serial, device, model.
The runner reads it back and **refuses to start if the streaming camera is not
the one the session was calibrated with**. That failure has no other symptom: the
cloud is uniformly displaced, counts look healthy, the policy acts on it.

Solver defaults to `DANIILIDIS`. Measured on session_01 by the
`T_gripper_board` residual — all agree on position to ~4 mm, so this is accuracy,
not correctness:

| solver | pos RMS | rot RMS |
|---|---|---|
| TSAI | 8.14 mm | 1.50° |
| PARK / HORAUD | 4.80 mm | 0.99° |
| ANDREFF | 4.94 mm | 0.99° |
| **DANIILIDIS** | **4.56 mm** | **0.99°** |

### Reading the validation numbers

Thresholds: translation 3.0 mm, rotation 0.5°, reprojection 1.0 px.

**Rotation is the one that matters** — it is the only term that grows with
distance, and being a bias it does not average away. Translation is a constant
offset. Reprojection is in *pixels*, which is not a property of the calibration
alone but of the focal length, so it is not comparable across camera bodies; it
is also the only row that can see the intrinsics, so it failing while the other
two pass points at the intrinsics rather than the hand-eye solve.

Two measured causes of rotation error, both fixable at capture time:

| board tilt | rotation error | | board position | rotation residual |
|---|---|---|---|---|
| ≥ 39° | 0.13–0.23° | | middle half | 0.324° |
| ≤ 28° | 0.39–1.99° | | toward the edges | 0.856° |

A planar target viewed square-on barely constrains its own out-of-plane
rotation; and the factory intrinsics carry **zero distortion coefficients**, so
edge distortion is absorbed as board tilt. Re-solving on the tilted / central
captures alone gave 0.173° and 2.43 mm / 0.382° respectively, using fewer than
half the images. Board scale is a similarity transform — re-measuring fixed
translation and left rotation bit-for-bit unchanged, which is the diagnostic.

Also: a wrist-mounted board **settles** during the first few poses. Jog the arm
through the full range before the first capture.

### Redo the calibration when

The tripod is moved, bumped or re-aimed (**the common case**); the camera is
re-seated or swapped; the capture resolution changes. Re-mounting the board
*after* a finished calibration does not invalidate it — the mounting cancelled.

### Swapping the camera body (D435 → D455)

Recalibrate; nothing else. The body is detected from the device at startup.
`min_depth_m` rises to its min-Z automatically (D455 0.40 m vs D435 0.10) and is
never lowered. **Do not put a D455 on the wrist** — min-Z 0.40 m against a
working distance of 0.3.

Worth it on the tripod: baseline 50 → 95 mm and depth noise is linear in
baseline, so error roughly halves — against ~12 % spread across every resolution
the D435 offers. RGB is 90° against 69°, and since depth is aligned *to colour*,
the colour frame is what clips the cloud.

---

## Perception

### Two cameras: fusing wrist + tripod

Per camera: segment the hand, deproject, label the object, transform into
`panda_hand`. Then concatenate **per class across cameras** and sample 896
object + 128 hand. That mirrors `handover_env._get_point_states` +
`PointListener` step for step, including the deliberate absence of per-camera
balancing — the sim concatenates raw, so a nearer view legitimately dominates.

The wrist camera's extrinsics are constant, so its cloud cannot be corrupted by a
stale robot pose. The tripod's chain inherits the error of **both** the hand-eye
calibration and the reported EE pose. Fixed cameras alone will run, but warn.

Two honest caveats: `run16` was trained on wrist + **left + right**, so two
cameras is an input-distribution change, not just a noisier observation — a
second tripod from the opposite side is the fix, not tuning. And in sim "which
points are the object" is a body-id lookup with the arm excluded for free; here
it is derived from a hand mask, so the gripper gets labelled object as it closes.

### Defining the object class

The object class owns 896 of 1024 rows and in training came from a
segmentation-buffer lookup: the object and nothing else. Reproducing that from a
hand mask is **the largest remaining sim2real gap in perception, larger than the
calibration**.

The object is *being held*, so it is one connected body with the hand while the
table is not. `hand_connected_object_points` voxelizes both classes, takes
26-connected components, and keeps object points sharing a component with the
hand. Connectivity does the excluding, so the radius is a loose workspace bound
(0.22 m wrist / 0.26 m tripod) rather than the segmentation itself.

| staged scene: hand holding a 0.24 m bar over a table | wrist, 10 mm voxel | tripod, 12 mm voxel |
|---|---|---|
| bar kept | 100 % | 100 % |
| table + distractor kept | 0 % | 0 % |
| gap it can bridge to its own object | 10 mm | 15 mm |
| table standoff before it separates | 17.5 mm | 22.5 mm |
| cost per camera per frame | ~2.7 ms | ~1.7 ms |

**Nothing is dilated** — growing the hand by one voxel pulled in half the
tabletop and removing it cost nothing, because the two classes partition the
cropped pixels and are pixel-adjacent wherever they meet. The last row is the
limit: **a hand within ~2 cm of the table still merges with it**, and there this
is a no-op. `--no-cluster` restores the old behaviour including the tight radii.

### Promptable segmentation (SAM2 + Grounding DINO)

`--segmentation sam2` **measures** the object instead of deriving it, so
everything above is skipped.

```bash
# offline first: no ROS publish, the arm cannot move. 'n' re-seeds.
python test_perception_viz.py --cameras tripod --calib-session d455 \
    --segmentation sam2 --seg-object-prompt "a blue mug."

# then the policy, in the usual order
python my_policy_runner.py --run run19 --cameras tripod --calib-session d455     --segmentation sam2 --seg-object-prompt "object in human hand." --home  --enable-gripper --show-cloud --control rate
python my_policy_runner.py --run run19 --cameras tripod --calib-session d455 \
    --segmentation sam2 --seg-object-prompt "a blue mug." --dry-run
python my_policy_runner.py --run run19 --cameras tripod --calib-session d455 \
    --segmentation sam2 --seg-object-prompt "a blue mug." --home --step-mode
```

Grounding DINO runs **once per episode** — it cannot run inside a 150 ms step;
SAM2's memory bank carries the masks from there. The episode boundary is
`perception.reset()`, so `s` re-detects. Among several hand boxes it picks the
one nearest the **robot base in 3D**, same criterion as
[Which blob is the hand](#which-blob-is-the-hand); the object is then the box
nearest that hand.

With an object mask, `extract_hand_object_clouds` skips the crop sphere, the
margin band and connectivity, and `observe()` skips forearm rejection — all four
exist only because "object" meant "not hand, near the hand". Depth range, the
radius backstop and the robot/finger boxes stay. Measured on a hand holding an
object **resting on a table** at 0.60 m, the case connectivity provably cannot
fix:

| | object points | of which table |
|---|---|---|
| derived (crop + connectivity) | 1265 | **905 (72 %)** |
| measured (object mask) | 400 | **0** |

It also decouples the classes: the object used to be defined relative to
`hand_center`, so losing the hand lost the object.

**The failure mode inverts.** `hand-net` fails loudly — empty mask, `o0/h0`,
`usable == False`, arm holds. A tracker fails *silently*, reporting a confident
mask of the wrong thing. So it re-seeds when a mask collapses below 60 px, its
area changes >3× between frames, its 3D centroid moves >0.22 m in one step
(1.5 m/s for 150 ms), or SAM2's occlusion score drops — rate-limited to once a
second. **`RESEED` and a running `xN` on the per-camera HUD** are the only place
that failure is visible; the point counts stay entirely plausible. The object
mask is tinted red against the hand's green.

Setup — nothing is installed by default, and the backend prints these if missing:

```bash
pip install 'hydra-core>=1.3.2' 'iopath>=0.1.10' 'transformers>=4.40'
pip install --no-deps 'git+https://github.com/facebookresearch/sam2.git'
mkdir -p checkpoint/sam2                       # curl -o will not create it
curl -L -o checkpoint/sam2/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

`--no-deps` stops sam2's pins moving `numpy` and `torch`. Verified unmoved here
(torch 2.5.1, numpy 1.23.5, cv2 4.11.0) — check anyway with
`python -c "import torch,numpy,cv2;print(torch.__version__,numpy.__version__,cv2.__version__)"`.
Grounding DINO goes through HuggingFace `transformers`
(`IDEA-Research/grounding-dino-tiny`, 689 MB, pulled on first use) rather than
the original package, which compiles a CUDA extension at install time. The
`>=4.40` floor spans a major version — 5.x renamed `box_threshold`→`threshold`
and `torch_dtype`→`dtype` — and the backend picks at runtime.

There is no SAM2 variant called "nano": the SAM 2.1 line is
tiny/small/base+/large. NanoSAM is a distillation of SAM *1* with no video
memory and cannot track.

> **Unmeasured.** Latency on this GPU has not been measured. The encoder runs
> once per camera per frame against a 150 ms step in `--control rate`, where the
> current segmenter costs 9.3–26.0 ms. Measure before trusting two cameras;
> `run19` is the single-camera landing zone.

### Which blob is the hand

The segmenter fires on any skin, so it returns the face and forearm too.

| camera | rule | why |
|---|---|---|
| wrist | largest blob | the hand fills the frame; bit-for-bit what run12 trained against |
| fixed | nearest the **robot base** | a whole person is in frame, and at 1 m a face rivals a hand in area |

Area selection on a tripod fails in a way that looks nothing like a mask problem.
`hand_center` is the origin of the object crop, the connectivity seed and the
arm-rejection capsule, so when the mask jumps to the face the real hand and its
object land 20–40 cm off-axis and are rejected as arm. Observed: `hand=` swinging
451↔810 between frames and the object class collapsing to 10–20 points.

The **base**, not the end effector, because the EE is where the policy is being
driven — scoring against it closes a loop in which one bad frame pulls the arm
toward the error. Falls back to area whenever geometry is unavailable.

### Excluding the robot

`ROBOT_EXCLUSION` removes a box around the gripper, ceiling `z = +0.02` in
`panda_hand` — clearing the housing and wrist while **sparing the finger volume**
where a grasped object sits (fingertips at `z = 0.075…0.105`).
`--no-robot-exclusion` disables it.

#### Excluding the gripper fingers

That ceiling leaves the fingers, which are the half of the gripper a side camera
sees best. Geometry read off `panda_gripper_hand_camera.urdf` and
`meshes/collision/finger.obj`, not guessed: each finger is `|x| ≤ 0.0105`,
`z ∈ 0.0585…0.1123`, `|y| ∈ q…q+0.0264`, with `q` the live jaw travel from
`/franka_gripper/joint_states` — the same reading that feeds `robot_state[25]`,
read once per iteration so cloud and state cannot disagree by a frame.

**The inner face gets no margin.** Every other bound is padded 8 mm, which is
free — outboard of a finger there is only more gripper. Padding inward would take
8 mm off both sides of the object exactly when it matters. The inner bound is the
measured mesh face, 0.13 mm inboard of the joint origin, rounded out to 0.6 mm.

`--finger-boxes {split,span,off}`. `split` is the pair. `span` merges them
through the gap: it cannot miss a mis-calibrated finger point, but it deletes the
object at contact — a 37 mm column in x, so a narrower object goes entirely.
Reach for `span` to find out whether finger points are hurting you; run `split`
once you know. The mode is printed at startup and the wireframe follows it.

Measured: split keeps 60/60 of a 24 mm object at the jaw centre and drops all 16
mesh corners at every jaw width; span drops all 60 but catches a finger point
7 mm inside the gap that split misses.

### The forearm, which connectivity cannot remove

`HandSegModel` segments **hands, not arms**, and the forearm is anatomically
continuous with the hand, so it is in the hand's component by construction. In
sim there was no counterpart — the human is a MANO mesh, wrist to fingertips — so
this is not a point type the policy handles badly, it is one it has never seen.

`reject_arm_clusters` scores each blob on the **fused** cloud by

```
s = (blob centroid − hand centroid) · unit(robot base − hand centroid)
```

dropping blobs with `s < −arm_offset` **or** lateral distance `> arm_lateral`
(a capsule, not a half-space) **or** more than `arm_below` under the hand.

Three deliberate choices: the **base, not the EE** (same feedback-loop argument
as above); a **threshold, not argmin**, since argmin deletes half an object
whenever occlusion splits it; and a threshold **well negative**, because a mug
gripped by the handle scores about −0.03 against a forearm's −0.20.

Why each term exists, measured:

| forearm tilt | 0° | 30° | 60° | 75° | 90° |
|---|---|---|---|---|---|
| arm kept, offset term alone | 6 % | 11 % | 42 % | 86 % | **100 %** |

A forearm hanging straight down is at 90° and scores about −0.05 — inside any
threshold that also keeps a mug. With the lateral term the same sweep is 0 % arm
and 100 % object at every tilt.

| | object held sideways | vertical forearm rejected |
|---|---|---|
| lateral 0.12 | 67 % kept | 20 % leaks |
| lateral 0.18, no below-term | 100 % kept | **56 % leaks** |
| lateral 0.18 + below-term | **100 % kept** | 20 % leaks |

`--arm-below` is what makes the wider lateral bound safe: a forearm descends to
the elbow, an offered object does not hang below the hand holding it. **That is a
real assumption** — it would fail for someone reaching up from below.
`--arm-below 999` disables it.

#### The flicker

Reported as *"sometimes the arm appears as object"*. Not the threshold: the hand
mask sits slightly inside the true silhouette, so a shell of genuine hand-surface
points falls into the object class and bridges the object to the forearm.

| shell surviving | blobs | arm kept, before | after |
|---|---|---|---|
| 0 % | 2 | 0 % | 0 % |
| 20 % | 1 | 100 % | 2 % |
| 100 % | 1 | 100 % | 2 % |

**20 % coverage fuses them**, and the merged blob scores −0.025 — nowhere near
−0.07, so no tuning would have helped. Two fixes: `hand_margin_px` removes the
shell at source (those pixels still seed connectivity, or a gap would open
between hand and object); and mixed blobs are cut **point by point** when between
5 % and 95 % behind the threshold. The residual 2 % is forearm within 7 cm of the
hand centroid.

#### `[arm:all-arm]`

When no blob survives the object class goes **empty** and the caller holds the
frame. This is deliberate. The first version kept the least-bad blob and on
hardware fired every frame, promoting an 862-point, 87 %-arm blob into the object
class — invisible, because downstream an arm-shaped object cloud looks like any
other. A stall is visible and recoverable; an arm labelled "object" steers the
policy into it. Read the blob table to tune:

```
blobs: 862@-0.092/lat0.031/0.87    <- 9.2 cm behind the hand: --arm-offset
       61@+0.054/lat0.180/1.00     <- inside the offset, 18 cm off-axis: --arm-lateral
```

#### The grasp region overrides all of it

As the gripper converges its points and the object's merge into one blob whose
bulk scores as arm, and the whole-blob drop took the object with it — the object
class emptied exactly when the grasp became possible. `GraspRegion` (`|x|,|y| <
0.055`, `z` 0.05–0.13) is never scored as arm, and a blob holding ≥10 such points
is never dropped whole. Staged, one blob of 1240 points, 82 % arm-like: object
kept 0 % → **100 %**, arm mass 0 % either way.

The bounds are looser than the hardware because this region only ever *keeps*
points: generous costs a little robot surviving, tight costs the object when it
matters most. The 10-point floor stops a single depth-noise point rescuing an
arm. `grasp<N>` and `!veto<N>` print when it fires.

### Segmentation resolution

`seg_input_px` is 256 for the wrist (what cp1 trained at) and **384 for fixed
cameras** — at 1.5 m a hand is a small patch and after the square resize its mask
falls through `min_hand_points` (observed: 103 → 63 → 40 against a floor of 40).
`HandSegModel` is fully convolutional, so a larger input is legitimate. Cameras
sharing a size share one forward.

| | ms per pass |
|---|---|
| 1 cam @256 | 9.3 |
| 2 cams @256 batched | 15.5 |
| wrist @256 + tripod @384 | 26.0 |
| wrist @256 + tripod @512 | 40.8 |

If `tripod:o0/h0` persists while a hand is plainly in view, the segmenter is the
cause, not the geometry: try `--fixed-seg-px 512` or move the tripod closer.

### Cost

| stage | |
|---|---|
| `camera.get_frames` | 33 ms — blocking on the 30 fps stream |
| hand segmentation @384 | 24 ms, overlapped with that wait |
| full `observe()` | 33–47 ms |

One iteration of real work is ~50 ms. If the loop feels slower, suspect something
idling rather than something computing.

---

## The 3D view

`--show-cloud` opens the same window `test_perception_viz.py` opens. It draws
**two** clouds: large coloured dots are the literal `[1024, 5]` tensor (orange
object, green hand), small white dots are the raw deprojected scene. The white
cloud is what makes the coloured one legible — a hand cloud rigidly displaced by
a bad extrinsic still looks like a perfectly good hand cloud, but against
recognisable geometry the displacement is obvious, and **with two cameras the
white clouds must overlap**.

`c` toggles colouring by class (what the one-hot channels say) versus by camera
(blue wrist / yellow tripod). Colour-by-camera is the multi-camera diagnostic: a
rigid offset between blue and yellow *is* the calibration error.

Also drawn: a **gripper wireframe** from GA-DDPG's control points (fingertips at
`z = 0.105`) and the exclusion boxes when filtering. The gripper is not
decoration — the cloud is in `panda_hand`, whose origin is invisible in a bare
scatter, and without it "the object is 8 cm ahead of the fingers" and "8 cm
behind" look identical. Fingers spread along **y**, which is where
`panda_finger_joint1`'s axis puts them; GA-DDPG's raw array spreads them along x
because it is in GraspNet's grasp frame, and `get_control_point_tensor(rotz=True)`
is the 90° z rotation between the two.

### Inspecting perception on its own

`test_perception_viz.py` opens the same windows without a policy and **never
publishes to ROS**, so it cannot move the arm.

```bash
python test_perception_viz.py --cameras wrist,tripod --calib-session <session>
python test_perception_viz.py --cameras wrist --no-ros     # no robot at all
python test_perception_viz.py --selftest                   # no camera/ROS/display
```

A fixed camera needs the live robot pose, so `--no-ros` refuses to draw one
rather than placing points against a fictitious arm position.

Watch for `STALE` in the 2D captions and `*` in the console. When the segmenter
loses the hand, the pipeline falls back to the previous frame's hand cloud and
crops the object around that *stale* centroid — which can yield a large, entirely
bogus object cloud (15k points observed).

Notes if you edit the viewer: perception runs on a worker thread throttled to
15 Hz (unthrottled it pinned 2.4 cores and starved the render thread it exists to
feed; throttling *raised* the GUI from 66 to 82 Hz). The key handler must return
a plain `bool` — returning `EventCallbackResult` from a `gui.Window` handler
fails to convert and tears the window down on every keypress, and calling
`_on_key` from Python cannot detect it because that path never crosses the
binding. `O3DVisualizer` draws a skybox over `set_background`, so
`show_skybox(False)` comes first; and Filament's tone mapper means rendered
pixels do not match source colour constants.

---

## Camera

Serials have **one definition**, in `camera calibration/calib_config.py`:

```python
CAMERA_SERIALS = {"tripod": "243122302229", "wrist": "045322075902"}
```

Quote them — one beginning with `0` is not a valid Python integer literal.
Re-confirm after re-cabling with `python calib_common.py --preview`: one window
per camera with its serial overlaid; jog the robot and the wrist one moves.
Getting the assignment wrong is **silent** — the policy simply receives a
viewpoint it was never trained on.

Defaults are 640×480 for both streams at 30 fps. Colour and depth are separate
streams aligned afterwards, so they need not match. The D435 offers 424×240
colour but **no 424×240 depth**, which is why `policy_runner.py`'s pair fails
device-side with `Couldn't resolve requests`.

**If modes are accepted but no frames arrive, it is a link fault, not a config
one.** A D435 enumerating at USB **2.1** advertises modes it cannot stream at all
— every resolution fails, even 6 fps. `calib_common.py` prints the link type for
every device; you want `3.x`. Two cameras at 640×480×30 can also exceed one USB3
controller: put them on **separate controllers**, not merely separate ports, or
drop `--camera-fps` to 15.

Depth resolution, measured against a 2.15 m wall with the ROI pinned in *angle*
(equal pixel counts would compare different fields of view):

| mode | plane RMS | disparity RMS |
|---|---|---|
| 640×480 | 110.01 mm | 0.458 px |
| **848×480** | **97.28 mm** | **0.444 px** |
| 1280×720 | 102.52 mm | 0.702 px |

Flat disparity from 640 → 848 means the extra pixels are real; the jump at 1280
confirms it is extrapolated above the D435's native 848×480.

---

## Gripper

`--enable-gripper` publishes `franka_gripper/GraspActionGoal` to
`/franka_gripper/grasp/goal` (roslibpy 2.0 dropped its actionlib client, and
actionlib is plain topics underneath). Width comes from
`/franka_gripper/joint_states`. Defaults: width 0.0, force 20 N, speed 0.05,
epsilon 0.3/0.3.

**The `franka_gripper` node must be running** — if the topic is not advertised,
goals go nowhere silently. With no width feedback the runner assumes "open",
which is correct throughout the approach.

---

## Control and safety

| limit | value |
|---|---|
| per-step translation | 5 cm |
| per-step rotation | 20° |
| workspace | x ≥ 0, z ≥ 0 |
| settle tolerance | 5 mm, 3 s timeout |
| command lead | 100 mm |
| episode length | 50 steps |

The step caps are guards, not shapers: `action_std` is 0.016–0.021 m and
0.06–0.08 rad, so normal motion never reaches them. Frequent `CLAMPED` means
something is wrong upstream.

Homing targets `ENV.PANDA_INITIAL_POSITION`, joints
`(0.0, -1.285, 0.0, -2.356, 0.0, 1.571, 0.785)`, fingers 0.04. **If you have a
joint-space controller, commanding those joints directly is safer** than the
Cartesian interpolation used here.

### Control mode: `settle` vs `rate`

**`--control settle`** (default) blocks until the arm has arrived and held still
for 0.3 s. That buys a clean single-frame observation — the policy is Markov, and
its `robot_state` would otherwise be paired with a cloud captured mid-motion. It
costs the entire step: **1090 ms, of which perception is ~43 ms and inference
~10 ms**.

**`--control rate`** is the CVPR2023 loop (arXiv 2303.17592): one command, a
fixed 0.15 s of execution, then look again, with no test of whether the arm got
there. The arm permanently chases a target that has already moved, which is what
makes the motion continuous; undershoot needs no correction because the next
action is a fresh delta from wherever the arm reached. **150 ms per step, 7.3×**,
and the regime the policy was trained in.

It publishes exactly what `settle` publishes *first*, so `rate` is precisely
"settle's first command, then stop waiting".

**The droop lead is not optional in `rate` mode.** The target is rebuilt from the
measured pose every tick, so a command the arm is too stiff to execute does not
accumulate — the equilibrium is re-placed at the same physical spot, the arm
stays put, the observation does not change, and the policy re-issues the same
delta forever. Measured on a 4 mm step against a 12 mm stall band: 0.00 mm over
60 ticks. First run on hardware: `--step-mode --control rate` gives the
fixed-rate command shape while `SPACE` still gates each tick.

### Homing is streamed

The waypoint loop used to call `settle()` on each 2 cm waypoint — a 30 cm home
was fifteen moves separated by fifteen dead stops. It now publishes the same
interpolated path on a clock at `--home-speed`. On a 27 cm home:

| | mid-motion stops | time stopped | landed | total |
|---|---|---|---|---|
| stepwise | 30 | 11.8 s | 1.0 mm | 24.4 s |
| streamed | 0 | 0 s | 18.9 mm | 4.1 s |

Unlike the policy loop this cannot deadlock — waypoints are absolute and
pre-computed, so they advance whether or not the arm keeps up. The final refine
is load-bearing rather than cosmetic, since nothing along a streamed path waits
for the arm. `--home-stepwise` restores the old behaviour.

**Home is a starting pose, not a precision target**, and `--home-tol` is 2 cm
because that is what this arm can land. It was 2 mm, against a ~17 mm standing
droop and an under-travel gain that swings 0.44–0.92 between consecutive moves —
so each refine pass overshot, the next corrected back, and homing hunted around
the target for up to six passes of twelve nudges each. Nothing downstream needs
the precision: the policy is single-frame and closed-loop, so it observes
wherever the arm actually is and steps from there, and a couple of centimetres
at t=0 is a different starting state rather than an error that accumulates.

**This only ever showed up on the SECOND home**, which is the diagnostic worth
remembering. The first runs before any episode, so the droop estimator has
learned nothing and the refine is commanded honestly; every later one runs with a
lead learned over ~30 mm policy steps. `test_step_motion.py` homes both ways for
that reason — testing only the fresh-estimator case is what let this through.

### Why the arm used to move in bursts

Ending each step stopped is required; stopping *three times* inside one step was
not. The old loop ran up to three commands per step, each waiting for a complete
stop. Two changes:

- **Correct without stopping first.** An impedance controller accepts a new
  equilibrium at any instant. When the arm stalls short — still for 0.05 s, not
  0.3 s — the lead is lengthened and republished right there.
- **Stop chasing the last millimetres.** The tolerance is a fraction of the step
  (`--step-tol-frac`) rather than a fixed distance. A fixed tolerance loose
  enough to stop the corrections (20 mm, measured) silently turns any step under
  20 mm into a no-op — near the object, the entire endgame quietly not executing.

| | commands per step | stopped mid-move | steps converged |
|---|---|---|---|
| multi-pass | 2.8–3.0 | 330–663 ms | 0/12 |
| creep | 1.0–1.2 | 0–12 ms | 12/12 |

**Gain jitter is why the tolerance had to move.** Four consecutive steps measured
under-travel gains of 0.51, 0.44, 0.85, 0.92 — a genuinely different plant every
move. No feed-forward lands a first command precisely against a 2× swing.

The lead is a **scalar along the direction of travel**, not a vector: friction
opposes travel, so a vector learned on the last step points the wrong way when
the policy reverses, which during a handover it does constantly. It is a
magnitude, not a gain — a measured gain is contaminated by the standing offset
and re-applying it at a different scale settled at 0.35 against a true 0.50. And
**a lead may never oppose the way the arm still has to go**, which needs no model
and fixes a real deadlock on overshoot.

**A magnitude is only valid at the scale it was measured at**, and both ends of
that have to be enforced. The estimator already refused to *learn* from moves
under 5 mm; it did not refuse to *apply* a full-step lead to one. `go_home`'s
refine is a few millimetres, so every home after the first — once an episode had
taught the estimator anything — commanded a 26 mm lead on a 3 mm correction,
overshot, came back, and hunted for all six refine passes. It could not learn its
way out either, because those same moves were below the floor. The first home
always worked, because nothing had been learned yet. The lead is now scaled by
`travel / scale_it_was_learned_at`, clamped at 1.0 so full-step behaviour is
unchanged; `test_step_motion.py` asserts both halves.

The plant is **not fully identified** — affine under-travel and stiction fit the
same two measurements. `test_step_motion.py --sweep` also runs a free grid as a
stress test, but most of it contradicts the robot, so only convergence is
asserted there; bounds are asserted on the constrained family.

---

## Regrasp: grasping from a commanded direction

`my_regrasp_policy_runner.py` drives the phase-5 regrasp policy, which is told
**which side** to take the object from.

```bash
python my_regrasp_policy_runner.py --selftest          # geometry only, no robot

python my_regrasp_policy_runner.py --direction +x --cameras tripod \
    --calib-session d455 --segmentation sam2 --seg-object-prompt "a box." --dry-run

python my_regrasp_policy_runner.py --direction +x --cameras tripod \
    --calib-session d455 --home --step-mode

python my_regrasp_policy_runner.py --regrasp-iter 23     --cameras tripod --calib-session d455 --segmentation sam2     --seg-object-prompt "a brown cuboidal box." --home --control rate --enable-gripper --home-gripper --show-cloud --direction +x
```

Every flag of `my_policy_runner` works here — same control modes, homing,
gripper, `t` abort, perception, 3D view — because the script **imports** that
runner and supplies a policy adapter rather than copying its ~700 lines of
bring-up. Only the policy differs.

| extra flag | default | meaning |
|---|---|---|
| `--direction {+x,+y,-y,+z}` | **required** | which side to approach from; `-y` and `=-y` both parse |
| `--regrasp-run {9,11}` | `9` | which installed regrasp run — different `d_rule` |
| `--regrasp-iter N` | `best` | pick an iteration by number, or `best` / `last` |
| `--anchor-ref {base,hand}` | `base` | which point the anchor azimuth is measured from |
| `--regrasp-run-dir DIR` | — | a run dir anywhere else; overrides `--regrasp-run` |
| `--regrasp-ckpt` | `best` | `best` or `last` inside the run |
| `--command-axes PATH` | beside the run dir | the deployment axes |
| `--selftest` | off | check the direction geometry offline and exit |

**`--anchor-ref` is the one place a regrasp deployment can be quietly wrong.**
Run 9's bins are *named* in an anchor frame whose azimuth reference is the MANO
wrist joint — its config sets no `SIM.anchor_hand_ref`, and the pin table records
none, which `setup.resolve_anchor_ref` reads as `wrist` by construction. The rig
has no wrist joint, so it approximates that frame, and the two approximations are
not equally good: `horizontal(base - object)` changes 7.7% of bin labels against
the wrist frame and has a 61 cm lever arm, while the segmented hand cloud's
centroid is the run-16 mismatch — measured at 40% bin agreement, on a 9 cm lever
arm that shrinks to 7.7 cm at the close and falls back to an *opposite* sign
below 4 cm. Hence the `base` default. `hand` exists to A/B that claim on this
hardware.

**The direction is anchored to the scene, not the robot.** `+x` does not mean
the robot's +x — it means the free end of the object, in a frame built from
where the object sits relative to the giver's hand:

```
z = world up          x = horizontal(object centroid - wrist)          y = z x x
```

so `+x` follows the object as the person turns. Both points come from the two
segmented classes, which is why `regrasp/anchor.py` takes plain arrays and no
simulator. The frame is computed **once per episode** and held, matching sim,
where `set_direction` is called before step 0 and never again; recomputing it
per frame would let the command drift as the hand moves. If the hand class is
empty the anchor latches to a robot-base fallback and the HUD says `anchor=base`.

**The direction rides in the point cloud**, not the robot state: the cloud is
`[1024, 7]` — the usual xyz + object + hand plus `d·n` and `d·r` per point.
Those are dot products of unit vectors, so they carry no dataset statistics and
need no normalization, which is exactly what makes them portable to a real
camera. `BCRunner.act` builds them from a 5-channel cloud, so **perception needs
no changes at all**.

**Run 9 deploys on bin centroids, not the unit axes.** It is the first run whose
training label and deployment command are different vectors — trained on each
demonstration's own continuous approach axis, deployed on the empirical centroid
of each bin. Those centroids are per-run and live in `command_axes.json`;
substituting `BINS` would command a direction up to ~16° off. The runner reads
that file and refuses a run without one. A consequence worth knowing: the
centroids are **not** mutually orthogonal — `+y` and `-y` meet at **150.5°**, not
180°, because lateral demonstrations also lean toward the free end.

`-x` (over the giver's fingers) and `-z` (from beneath) are bins but have no
demonstrations behind them, so they are not offered.

**Which run.** Two are installed, and they take a *different* command rather
than being two tunings of one. Run 9's `d_rule` is `approach_axis` — `d` is the
gripper's approach axis, "come at the object from this side". Run 11's is
`grasp_offset` — `d` is the grasp point's offset from the object centroid,
"grasp this part of the object". Their `command_axes.json` files are therefore
not interchangeable, and the runner prints which rule is in force at startup.

At their best iterations run 11 grasps more often and obeys less: success 0.619
against 0.592 and close rate 0.778 against 0.685, but `bin_hit_rate` 0.14
against 0.65 and `cond_sep` 0.31 against 0.89 — and a low `cond_sep` is the
evaluator's name for "the policy ignores the conditioning". Run 11 also has all
six bins live where run 9 has four, and its `-y` scores 0.600 against run 9's
0.434. Full comparison in `checkpoint/regrasp_run11/README.md`.

```bash
python my_regrasp_policy_runner.py --direction +x --regrasp-run 9    # iter 23
python my_regrasp_policy_runner.py --direction -y --regrasp-run 11   # iter 22
```

**Which iteration.** All 26 of each are installed, every one a loadable run dir,
selected with `--regrasp-iter`:

```bash
python my_regrasp_policy_runner.py --direction +x --regrasp-iter 23   # = run 9's best
python my_regrasp_policy_runner.py --direction +x --regrasp-run 11 --regrasp-iter 22
python my_regrasp_policy_runner.py --direction +x --regrasp-iter last --regrasp-ckpt last
```

**Success is not monotonic in the iteration**, so a later one is not
automatically a better one — run 9's 23 is its best at 0.592 while 16 dips to
0.391, and run 11's 22 is its best at 0.619 while 23 drops straight to 0.474.
The per-iteration tables are in each folder's README. Those numbers are in-loop
eval on the TRAIN split (`EVAL.holdout: false`), so treat them as a ranking
rather than an absolute.

The `.pt` files are **hard links** into `output/dagger_runs/`, so all 52
iterations cost almost nothing rather than 2.4 GB. They read exactly like copies
and deleting either side leaves the other intact; use `cp -a` instead of
`cp -al` if you ever want independent files.

**A direction with no demonstrations behind it is refused**, per run: its
`command_axes` entry is then the untouched geometric axis rather than a centroid
of anything, so the policy would fly it confidently with nothing behind it. Run
9 refuses `-x` and `-z` on those grounds; run 11 accepts all six.

`--selftest` asserts that `--direction -y` survives argparse, that both
installed runs' axis files load with the live bins they should have, that the
anchor's x axis really is `horizontal(base − object)` under the default
reference and `horizontal(object − hand)` under `--anchor-ref hand` — and that
the two are not the same frame, which is the check that would have caught the
run-16 mismatch — that every command is a unit vector producing a *different*
action (so the conditioning reaches the network), and that anchoring preserves
every pairwise angle. It needs no camera, robot or ROS.

---

## Why the two runners differ

Three things changed between the CVPR model and this one. Each produces
plausible-looking garbage rather than an error if carried over:

1. **Point cloud order and labels.** This runner wants object points first —
   896 `[x,y,z,1,0]` then 128 `[x,y,z,0,1]`. `build_policy_point_tensor` emits
   the opposite, which is correct for the CVPR model.
2. **No task-space action scaling.** GA-DDPG rescales through `PandaTaskSpace6D`;
   these targets are raw SE(3) deltas already in metres and radians.
3. **Only 8 of 32 robot-state channels reach the network.** `drop_joint_state`
   and `use_prev_act=false` keep `rs[18:26]` — EE pose plus gripper. The runner
   asserts the config agrees.

```
point_cloud  [1024, 5]  xyz + ycb_flag + hand_flag, panda_hand frame
robot_state  [32]       ...ee_xyz(3)+ee_wxyz(4)+gripper_norm(1)...  (EE pose in SIM WORLD frame)
action       [7]        dpos(3)+deuler(3) in panda_hand, + gripper bit (1 = open, 0 = CLOSE)
```

---

## Status

**Verified on hardware:** camera streaming, hand segmentation, cloud extraction,
policy forward pass, action scale, `--step-mode` executing nothing without a
keypress, and `run16` end to end under `--dry-run --show-cloud`.

**Verified offline** (`test_multicam_fusion.py`, `test_step_motion.py`): extrinsic
chains agree across camera kinds; the tensor is `[1024, 5]` with object rows
first; under-full classes pad by repetition; the exclusion and finger boxes drop
the housing and spare the jaw gap; forearm rejection keeps a split object whole
and a mug-by-the-handle while dropping a forearm; the flicker sweep asserts the
bridge actually forms (the first version passed by never bridging, which proved
nothing); per-point camera provenance survives sampling with zero mislabelled
rows; the `t` abort interrupts a motion in progress.

**Not yet exercised against the live robot:** `--control rate`, streamed homing,
the finger boxes, `run19`, gripper homing, in-process episode restart, the `t`
abort, and the entire `--segmentation sam2` path.

**Outstanding:**

* The **wrist** camera is uncalibrated (`T_hand_cam` is the sim's nominal mount).
  Largest remaining deployment error; affects every run.
* SAM2 latency is unmeasured — see [Promptable segmentation](#promptable-segmentation-sam2--grounding-dino).
* Clustering is a no-op when the hand comes within ~2 cm of the table. Plane
  removal is the fix for `hand-net`; `--segmentation sam2` sidesteps it.
* `camera.depth_to_pointcloud` deprojects with a bare pinhole model and ignores
  `intr.coeffs`. The D435 reported zeros so this cost nothing; the **D455 reports
  non-zero coefficients**, an estimated 5–7 mm systematic error at 1 m.
* `test_perception_viz.py` exits via `os._exit`. Open3D's Filament threads and
  roslibpy's Twisted reactor outlive `main()` and re-enter Python during
  finalization, hanging or aborting the process after all work is done.
