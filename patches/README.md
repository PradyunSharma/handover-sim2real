# Patches to the vendored submodules

`OMG-Planner` is pinned to upstream `NVlabs/OMG-Planner` at `0608db0`, which we
cannot push to. Local source changes therefore live here as patch files rather
than as submodule commits, and **a fresh clone does not have them** — that is the
failure this directory exists to prevent.

Apply after `git submodule update --init --recursive`:

```bash
git apply --directory=OMG-Planner patches/omg_planner_handover.patch
```

Idempotent check (prints nothing and exits 0 when already applied):

```bash
git apply --check --reverse --directory=OMG-Planner patches/omg_planner_handover.patch
```

## `omg_planner_handover.patch`

Two independent changes to two files.

### `omg/util.py` — `ycb_special_case` crashes on an empty grasp set

For `037_scissors`, `010_potted_meat_can`, `061_foam_brick`, `024_bowl` and
`025_mug` the function applies a position constraint and then reads Euler angles
out of the survivors:

```python
pose_grasp = pose_grasp[z_constraint[0]]
top_down = np.array(top_down)[:, 1]        # IndexError when top_down == []
```

The constraint can reject **every** candidate. `np.array([])` is 1-D, so `[:, 1]`
raises `IndexError: too many indices for array`, killing the process. An empty
goal set is a legitimate outcome that the caller already handles ("Planning not
run due to empty goal set"), so the patch returns it instead of crashing.

**Never fires on the s0 TRAIN split; it does fire on TEST.** That is why the bug
sat undiscovered through every Phase-4 and Phase-5 run and only surfaced when
`build_direction_table.py --split test` was first run — job 2656, a deterministic
failure that resubmitting cannot fix.

### `omg/planner.py` — the `external_grasp_filter` hook

`handover_sim2real/train_env.py:120` registers
`_hand_grasp_collision_mask` on the OMG scene env so `setup_goal_set` can prune
grasps that collide with the human hand (the paper's filter). Stock OMG has no
such call site, so **without this patch the flag silently does nothing**:
`SIM.hand_collision_filter: true` would appear to be on, the mask would never be
consulted, and the run would report a filter it never applied.

It is a no-op when the filter is off — `_hand_grasp_collision_mask` returns
all-True — so every run to date (`hand_collision_filter: false`, using the
paper's offline `valid_grasp_dict_005.pkl` instead) is unaffected either way.
The patch matters for what it prevents, not for what it currently changes.
