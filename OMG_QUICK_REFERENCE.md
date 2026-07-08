# OMG-Planner Quick Reference

## Entry Points

```bash
# Standard planning with visualization
python -m omg.core -v -f demo_scene_0

# Planning from point cloud
python -m omg.core -v -f demo_scene_0 -p

# Batch process all 100 scenes
python -m omg.core -exp -w

# Kitchen scenes with mouse interface
python -m real_world.trial_mouse -v -f kitchen0
```

## Main Classes

### `PlanningScene` - [core.py:710]
Wrapper for entire planning system
```python
scene = PlanningScene(config.cfg)
plan_info = scene.step()  # Run one planning step
scene.fast_debug_vis()     # Visualize
```

### `Planner` - [planner.py:90]
Main planning algorithm
```python
planner = Planner(env, traj, lazy=False)
planner.plan(traj)  # Execute planning
planner.history_trajectories[-1]  # Get planned trajectory
```

### `Cost` - [cost.py]
Trajectory cost computation
```python
obstacle_cost, grad = cost.batch_obstacle_cost(goals)
```

### `Optimizer` - [optimizer.py]
CHOMP trajectory optimizer
```python
optimizer = Optimizer(scene, cost)
optimizer.goal_set_projection(trajectory, gradient)
```

### `Learner` - [online_learner.py:60]
Online goal distribution learning
```python
learner = Learner(env, traj, cost)
```

## Input Data Formats

### Scene File (MATLAB format)
```python
import scipy.io as sio
scene = sio.loadmat('data/scenes/scene_0.mat')

# Contains:
scene['pose']              # [N, 4, 4] object poses
scene['path']              # [N] model paths
scene['goals']             # [M, 9] grasp configurations
scene['reach_grasps']      # [M, reach_len, 9] standoff trajectories
scene['grasp_qualities']   # [M] quality scores
scene['grasp_potentials']  # [M] cost metrics
scene['target_name']       # which object to grasp
```

### Grasp Poses (NumPy)
```python
grasp_data = np.load('data/grasps/simulated/025_mug.npy', allow_pickle=True)
transforms = grasp_data.item()["transforms"]  # [N, 4, 4] gripper poses
```

### Configuration
```python
from omg import config

config.cfg.optim_steps = 50              # Optimization iterations
config.cfg.use_standoff = True           # Pre-grasp approach
config.cfg.timesteps = 50                # Trajectory length
config.cfg.goal_set_max_num = 100        # Max grasps
```

## Output Data Formats

### Trajectory
```python
trajectory = scene.planner.history_trajectories[-1]  # [timesteps, 9]
# First 7 columns: arm joints (rad)
# Last 2 columns: gripper position (0-0.04m)
```

### Goal Sets
```python
target_obj = env.objects[env.target_idx]
target_obj.grasps           # [N, 9] joint space grasps
target_obj.reach_grasps     # [N, reach_len, 9] trajectories
target_obj.grasp_potentials # [N] quality scores
```

### Planning Info
```python
plan_info = scene.planner.info[-1]
plan_info['cost']       # Total cost
plan_info['obs']        # Obstacle cost
plan_info['smooth']     # Smoothness cost
plan_info['collision_pts']  # [timesteps, links, pts, features]
```

## Key Functions

| Function | Purpose | Location |
|----------|---------|----------|
| `load_grasp_set()` | Load .npy grasp files | planner.py:561 |
| `setup_goal_set()` | Filter collision/duplicates | planner.py:603 |
| `solve_goal_set_ik()` | IK solving | planner.py:490 |
| `plan()` | Main planning | planner.py:606 |
| `batch_obstacle_cost()` | SDF collision cost | cost.py |
| `functional_grad()` | Workspace→joint gradients | cost.py |
| `goal_set_projection()` | Goal constraint | optimizer.py |

## Data Location Map

| Data | Location | Format |
|------|----------|--------|
| Scenes | `data/scenes/scene_*.mat` | MATLAB |
| Grasps | `data/grasps/simulated/*.npy` | NumPy pickle |
| Objects | `data/objects/*/model_*.obj` | Wavefront OBJ |
| SDFs | `data/objects/*/_chomp.pth` | PyTorch tensor |
| Configs | `omg/config.py` | Python dict |

## IK Seed Configurations
```python
# 14 anchor seeds for IK solving
from core.utils import anchor_seeds  # GA-DDPG
seeds = anchor_seeds[:12]  # Use first 12 + current pose
```

## Important Parameters

```python
# Collision
epsilon = 0.2            # Obstacle padding (m)
target_epsilon = 0.1     # Target padding
clearance = 0.01         # Collision threshold
allow_collision_point = 5  # Max points in collision

# Optimization
optim_steps = 50         # Descent iterations
base_step_size = 0.1     # Learning rate
timesteps = 50           # Trajectory discretization

# Grasping
use_standoff = True      # Standoff approach
standoff_dist = 0.08     # Distance (m)
reach_tail_length = 5    # Steps from standoff to grasp

# Goal selection
goal_set_max_num = 100   # Max grasps to keep
target_hand_filter_angle = 120  # Max rotation (°)
```

## Common Workflows

### 1. Load and plan for a scene
```python
from omg.core import PlanningScene
from omg import config

config.cfg.scene_file = 'demo_scene_0'
scene = PlanningScene(config.cfg)
plan_info = scene.step()

trajectory = scene.planner.history_trajectories[-1]
```

### 2. Process grasps and get IK
```python
import numpy as np
grasp_poses = np.load('data/grasps/simulated/025_mug.npy', allow_pickle=True)
transforms = grasp_poses.item()["transforms"]

# Solve IK
reach_grasps, joint_grasps = planner.solve_goal_set_ik(
    target_obj, env, transforms, z_upsample=False
)
```

### 3. Generate expert demonstrations
```python
# Run planner multiple times
expert_trajectories = []
for scene_idx in range(100):
    config.cfg.scene_file = f'scene_{scene_idx}'
    scene = PlanningScene(config.cfg)
    plan_info = scene.step()
    expert_trajectories.append(scene.planner.history_trajectories[-1])
```

### 4. Load into replay memory (GA-DDPG)
```python
from GA-DDPG.core.replay_memory import BaseMemory

memory = BaseMemory(buffer_size=100000, args=args, name="expert")
for traj in expert_trajectories:
    memory.push(
        action=traj,
        expert_flags=1.0,  # Mark as expert
        expert_action=traj,
        ...
    )
```

## Visualization

```python
# Debug visualization
scene.fast_debug_vis(
    interact=1,              # 0: no interact, 1: interact, 2: double check
    collision_pt=True,       # Show collision points
    goal_set=True,           # Show goal set evolution
    write_video=True,        # Save to video
    nonstop=True             # Don't wait for input
)
```

## Notes

- **Robot:** Franka Panda arm (7-DOF) + parallel gripper (2-DOF)
- **Objects:** YCB dataset (25 objects pre-defined)
- **Grasp Format:** 4x4 transformation matrices (hand frame in object frame)
- **IK Solver:** PyKDL via ycb_render (Panda forward/inverse kinematics)
- **Video Output:** `.avi` format in `output_videos/` directory
- **Grid Resolution:** 0.02m for SDF computation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No grasps found | Check `.npy` file exists; verify object name |
| IK solving slow | Set `ik_parallel=True` in config |
| Memory issues | Reduce `collision_point_num` or `timesteps` |
| Collision false positives | Adjust `epsilon`, `clearance` |
| Poor plan quality | Increase `optim_steps`, tune `goal_set_max_num` |
