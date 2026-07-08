# OMG-Planner Exploration Report

## Executive Summary
The OMG (Online Motion Generation) Planner is a trajectory optimization system for robotic grasping. It generates collision-free grasp trajectories by combining pre-computed grasp poses with CHOMP-based trajectory optimization. The planner integrates with GA-DDPG for generating expert demonstrations for imitation learning.

---

## Directory Structure

### OMG-Planner Organization
```
OMG-Planner/
├── omg/                          # Main planner module
│   ├── core.py                   # Entry point, Scene and Environment classes
│   ├── planner.py               # Planner class with grasp loading & IK solving
│   ├── config.py                # Configuration parameters
│   ├── optimizer.py             # CHOMP trajectory optimizer
│   ├── cost.py                  # Cost functions (obstacle, grasp, smoothness)
│   ├── online_learner.py        # Mixture of experts for goal selection
│   ├── util.py                  # Utility functions
│   └── sdf_tools.py             # SDF manipulation
├── real_world/                  # Data generation & processing scripts
│   ├── trial.py                 # Main planning/trial execution [KEY FILE]
│   ├── gen_sdf.py               # SDF generation from meshes
│   ├── gen_xyz.py               # Point cloud generation from meshes
│   ├── gen_convex_shape.py
│   ├── process_shape.py
│   ├── convert_sdf.py
│   └── blender_process.py
├── layers/                      # Custom CUDA/PyTorch layers
│   ├── sdf_matching_loss.py
│   └── sdf_matching_loss_kernel.cu
└── data/
    ├── scenes/                  # Pre-generated scenes (100+)
    ├── grasps/
    │   ├── simulated/           # YCB object grasps as .npy files
    │   └── graspit/             # Alternative grasp source
    ├── objects/                 # 3D models, SDFs, point clouds
    └── robots/                  # Robot models and collision data
```

---

## Main Entry Points

### 1. **Standard Planning with Visualization**
```bash
python -m omg.core -v -f demo_scene_0
```
**File:** [omg/core.py](OMG-Planner/omg/core.py#L782-L820)
- Loads scene from `.mat` file
- Plans trajectory with visualization
- Supports multiple demos: `demo_scene_0`, `demo_scene_1`, kitchen scenes

### 2. **Point Cloud-Based Planning**
```bash
python -m omg.core -v -f demo_scene_0 -p
```
**File:** [omg/core.py](OMG-Planner/omg/core.py#L830-L870)
- Uses point clouds for perception
- Requires segmentation and pre-computed grasps
- Creates point-based SDF for obstacles

### 3. **Batch Processing (100 Scenes)**
```bash
python -m omg.core -exp -w
```
**File:** [omg/core.py](OMG-Planner/omg/core.py#L870-L880)
- Loops through scenes 0-99
- Generates trajectory videos
- Process all pre-computed scenes

---

## Core Components

### 1. **PlanningScene Class** [core.py](OMG-Planner/omg/core.py#L710)
Wrapper for the entire planning system
```python
scene = PlanningScene(config.cfg)
info = scene.step()  # Run planning step
scene.fast_debug_vis()  # Visualize results
```

### 2. **Planner Class** [planner.py](OMG-Planner/omg/planner.py#L90)
**Initialization & Core Methods:**
- `__init__()`: Loads grasps, sets up goal set, initializes learner
- `plan()`: Main planning function
- `load_grasp_set()`: Loads pre-computed grasps
- `setup_goal_set()`: Filters collisions, removes duplicates
- `grasp_init()`: Initializes trajectory end-points

**Key Functions:**
- `solve_goal_set_ik()`: Solves inverse kinematics for grasp poses
- `solve_and_process_ik()`: Processes IKs with filtering
- `flip_grasp()`: Augments grasps by 180° rotation
- `load_goal_from_scene()`: Loads pre-computed goals
- `load_goal_from_external()`: Uses external grasp detections

### 3. **Cost Class** [cost.py](OMG-Planner/omg/cost.py)
Computes trajectory costs during optimization
- **Obstacle Cost**: SDF-based collision penalty
- **Smoothness Cost**: Trajectory regularization
- **Grasp Cost**: End-effector pose preference
- Methods: `functional_grad()`, `batch_obstacle_cost()`, `forward_points()`

### 4. **Optimizer Class** [optimizer.py](OMG-Planner/omg/optimizer.py)
CHOMP (Covariant Hamiltonian Optimization for Motion Planning)
- Gradient descent-based trajectory refinement
- `goal_set_projection()`: Projects trajectory to feasible end-points
- Cost schedule balancing

### 5. **Learner Class** [online_learner.py](OMG-Planner/omg/online_learner.py#L60)
Online learning for goal selection
- **Mixture of Experts**: Multiple learning rates
- **Algorithm Options**: "Proj", "MD" (multiplicative descent)
- Dynamically weights goal costs during optimization

---

## Inputs to OMG Planner

### 1. **Scene Configuration** (`.mat` files)
**Location:** [data/scenes/](OMG-Planner/data/scenes/)
- Scene files: `demo_scene_0.mat`, `scene_0.mat`, ..., `scene_99.mat`

**Contents:**
```python
scene = sio.loadmat('data/scenes/scene_0.mat')
# Keys:
# - 'pose': Object/obstacle poses [N, 4, 4]
# - 'path': Model paths [N]
# - 'goals': Target grasp configurations [M, 9]
# - 'reach_grasps': Standoff reaching trajectories [M, reach_length, 9]
# - 'grasp_qualities': Quality scores [M]
# - 'grasp_potentials': Cost metrics [M]
# - 'target_name': Which object to grasp
```

### 2. **Pre-computed Grasp Poses** (`.npy` files)
**Location:** [data/grasps/simulated/](OMG-Planner/data/grasps/simulated/)
**Files:** `003_cracker_box.npy`, `025_mug.npy`, etc.

**Format:**
```python
grasp_data = np.load('data/grasps/simulated/025_mug.npy', allow_pickle=True)
grasp_dict = grasp_data.item()
pose_grasp = grasp_dict["transforms"]  # Shape: [N, 4, 4] - gripper poses in object frame
```

### 3. **Object Models**
**Location:** [data/objects/](OMG-Planner/data/objects/)
- Mesh: `model_normalized.obj` 
- SDF: `model_normalized_chomp.pth` (pre-computed)
- Extent: `model_normalized.extent.txt`
- Point cloud: `model_normalized.xyz`

### 4. **Configuration Parameters** [config.py](OMG-Planner/omg/config.py)
Over 60 tunable parameters:
- **Collision:** `epsilon`, `target_epsilon`, `clearance`, `collision_point_num`
- **Optimization:** `optim_steps`, `base_step_size`, `timesteps`
- **Grasping:** `use_standoff`, `standoff_dist`, `reach_tail_length`
- **Goal Selection:** `goal_idx`, `ol_alg` (algorithm), `goal_set_max_num`
- **Filtering:** `target_hand_filter_angle` (120°), `remove_flip_grasp`, `augment_flip_grasp`

### 5. **Point Cloud Data** (optional)
For perception-based planning with `-p` flag
- Target point cloud: Nx3 array (segmented)
- Non-target point cloud: Mx3 array (obstacles)
- Point SDF computed via KDTree

---

## Outputs from OMG Planner

### 1. **Joint Trajectories**
**Structure:** `trajectory.data` - NumPy array shape `[timesteps, 9]`
- First 7 values: Arm joint angles (radians)
- Last 2 values: Gripper finger positions (0-0.04m)
- Stored in: `planner.history_trajectories[-1]`

**Example:**
```python
traj = scene.planner.history_trajectories[-1]  # Shape: [50, 9]
# First step: current configuration
# Last step: grasp configuration
```

### 2. **Goal Sets**
```python
# Valid grasp goals
target_obj.grasps              # Shape: [N, 9] - joint space grasps
target_obj.reach_grasps        # Shape: [N, reach_length, 9] - standoff reaching
target_obj.grasp_potentials    # Shape: [N] - quality scores
target_obj.grasp_vis_points    # Visualization data
```

### 3. **Planning Information**
```python
planner.info[-1]  # Dictionary with:
# - 'cost': Final trajectory cost
# - 'obs': Obstacle cost component
# - 'smooth': Smoothness cost
# - 'grasp': Grasp quality cost
# - 'collision_pts': Collision points along trajectory [timesteps, links, points, features]
```

### 4. **Video Output** (optional with `-w` flag)
**Location:** `output_videos/`
- **Format:** `.avi` video files
- **Naming:** `demo_scene_0.avi`, `scene_0.avi`, etc.
- **Content:** 3D rendering of robot, objects, and trajectory

---

## Data Generation Pipeline

### 1. **Scene Dataset** (100 Pre-generated Scenes)
**Location:** [data/scenes/](OMG-Planner/data/scenes/)
- 100 YCB tabletop scenes with object configurations
- Each contains object poses, reachable grasps, quality metrics
- Used for training RL agent with diverse scenarios

### 2. **Grasp Generation from Grasping Pipeline**
**Source:** Simulator-based grasp synthesis

**Process:**
1. Load object mesh from YCB dataset
2. Compute grasp poses (gripper approach directions)
3. Store as 4x4 transformation matrices
4. Apply **rotation offset:** `rotZ(π/2)` for coordinate alignment
5. Special case handling for certain objects

**File:** [planner.py `load_grasp_set()`](OMG-Planner/omg/planner.py#L561)

```python
# Load simulated grasps
simulator_grasp = np.load('data/grasps/simulated/025_mug.npy', allow_pickle=True)
pose_grasp = simulator_grasp.item()["transforms"]  # Shape: [M, 4, 4]

# Apply rotation correction
offset_pose = rotZ(np.pi / 2)
pose_grasp = np.matmul(pose_grasp, offset_pose)

# Solve IK to get joint-space grasps
reach_grasps, grasps = planner.solve_goal_set_ik(target_obj, env, pose_grasp)
```

### 3. **IK Solving** [planner.py `solve_goal_set_ik()`](OMG-Planner/omg/planner.py#L490)

**Input:** Task-space grasp poses (4x4 matrices)
**Output:** Joint-space configurations + standoff reaching trajectories

**Algorithm:**
```
For each grasp pose:
  1. Apply coordinate transformations (z-upsample, y-upsample)
  2. Generate standoff poses (backward 0.08m along approach axis)
  3. Solve IK with 12 seed configurations
  4. Return multiple reach trajectories if IK succeeds
```

**Augmentation:**
- **Flip grasps:** Create 180° rotations in wrist joint space
- **Z-upsample:** 50 rotations around vertical axis
- **Y-upsample:** 10 rotations around finger axis

### 4. **Grasp Filtering** [planner.py `solve_and_process_ik()`](OMG-Planner/omg/planner.py#L243)

**Collision Filtering:**
- Check grasps against obstacles
- Keep collision-free grasps only

**Rotation Filtering:**
```python
# Filter grasps with excessive wrist rotation from start pose
R_diff = start_hand_pose.T @ target_hand_pose
angle = np.arccos((trace(R_diff) - 1) / 2) * 180 / np.pi
remove if angle > 120°  # config.target_hand_filter_angle
```

**Downward Approach Filtering:**
```python
# Remove grasps requiring gripper z-axis pointing upward
z_axis = target_hand_pose[:, :3, 0]
remove if z_axis[-1] > -0.3  # cos(70°)
```

### 5. **Goal Set Filtering** [planner.py `setup_goal_set()`](OMG-Planner/omg/planner.py#L603)

**Diversity Filtering:**
```python
unique_grasps = []
for grasp in all_grasps:
    distances = ||unique_grasps - grasp||
    if min(distances) < 0.5:  # Skip similar grasps
        continue
    unique_grasps.append(grasp)
```

**Final Sampling:**
```python
# Sample up to goal_set_max_num diverse grasps
keep = np.random.choice(unique_indices, min(len(unique), 100), replace=False)
```

### 6. **Data Processing Utilities**

**[gen_xyz.py](OMG-Planner/real_world/gen_xyz.py)**
- Generates point clouds from 3D meshes
- Extracts vertex coordinates and normals
- Normalizes geometry

**[gen_sdf.py](OMG-Planner/real_world/gen_sdf.py)**
- Converts mesh files to Signed Distance Field representations
- Uses external `sdf_gen` binary tool
- Outputs `.pth` PyTorch tensor files

---

## Expert Data Integration with GA-DDPG

### Integration Points

**1. Behavioral Cloning** [GA-DDPG/core/bc.py](GA-DDPG/core/bc.py)
- BC agent learns from OMG expert trajectories
- Supervised learning on trajectory features
- Uses same network architecture as RL policy

**2. Replay Memory** [GA-DDPG/core/replay_memory.py](GA-DDPG/core/replay_memory.py#L20)
- Stores both expert and learned trajectories
- Fields:
  ```python
  self.expert_flags       # 1.0 if from expert, 0.0 if learned
  self.expert_action      # Expert action for imitation loss
  self.next_expert_action # Next action in expert trajectory
  ```

**3. Data Flow:**
```
OMG Planner
  ↓
  Generates trajectory: [state_0 → action_0 → state_1 → ... → grasp_state]
  ↓
Replay Buffer
  ↓
  Stores with expert_flag=1
  ↓
Training Loop
  ├─ Sample expert transitions
  ├─ Compute behavioral cloning loss: ||policy_action - expert_action||²
  └─ Update policy to match expert
```

**4. Self-Supervision:**
- Optional use of expert trajectories as auxiliary training signal
- Helps stabilize RL training in early stages
- Controlled by `self_supervision` config flag

---

## Key Functions Summary

### Planner Functions
| Function | Location | Purpose |
|----------|----------|---------|
| `load_grasp_set()` | [planner.py:561](OMG-Planner/omg/planner.py#L561) | Load pre-computed grasps from .npy files |
| `setup_goal_set()` | [planner.py:603](OMG-Planner/omg/planner.py#L603) | Filter collisions & duplicates |
| `solve_goal_set_ik()` | [planner.py:490](OMG-Planner/omg/planner.py#L490) | Solve IK for grasp poses |
| `solve_and_process_ik()` | [planner.py:243](OMG-Planner/omg/planner.py#L243) | IK solving + filtering |
| `plan()` | [planner.py:606](OMG-Planner/omg/planner.py#L606) | Execute planning step |
| `grasp_init()` | [planner.py:192](OMG-Planner/omg/planner.py#L192) | Initialize trajectory endpoints |

### Cost Functions
| Function | Purpose |
|----------|---------|
| `batch_obstacle_cost()` | Compute SDF-based collision costs |
| `functional_grad()` | Workspace cost → joint space gradient |
| `forward_poses()` | Forward kinematics |
| `forward_points()` | Map collision points through FK |

### Data Generation Functions
| Function | Location | Purpose |
|----------|----------|---------|
| `gen_normal_xyz()` | [gen_xyz.py:52](OMG-Planner/real_world/gen_xyz.py#L52) | Generate point cloud from mesh |
| `generate_sdf()` | [gen_sdf.py:8](OMG-Planner/real_world/gen_sdf.py#L8) | Generate SDF from mesh |
| `recursive_load()` | [gen_xyz.py:18](OMG-Planner/real_world/gen_xyz.py#L18) | Recursively load mesh hierarchy |

---

## Configuration Parameters (Key Settings)

```python
[OMG-Planner/omg/config.py]

# Collision penalty
cfg.epsilon = 0.2                           # Obstacle padding distance
cfg.target_epsilon = 0.1                    # Target object padding
cfg.clearance = 0.01                        # Collision threshold
cfg.collision_point_num = 15                # Points per link

# Optimization
cfg.optim_steps = 50                        # Gradient descent steps
cfg.base_step_size = 0.1                    # Initial learning rate
cfg.timesteps = 50                          # Trajectory discretization

# Grasping
cfg.use_standoff = True                     # Pre-grasp approach
cfg.standoff_dist = 0.08                    # Standoff distance (m)
cfg.reach_tail_length = 5                   # Steps to grasp

# Goal selection
cfg.goal_set_proj = True                    # Use goal set projection
cfg.goal_set_max_num = 100                  # Max grasps to consider
cfg.ol_alg = "MD"                           # "Proj" or "MD"
cfg.goal_idx = -2                           # -2: min cost, -1: closest

# Filtering
cfg.target_hand_filter_angle = 120          # Max wrist rotation (°)
cfg.remove_flip_grasp = True                # Filter 180° flips
cfg.augment_flip_grasp = True               # Generate flip augmentations
```

---

## File Paths Summary

| Component | Primary Files | Location |
|-----------|---------------|----------|
| **Core Planner** | core.py, planner.py | `OMG-Planner/omg/` |
| **Optimization** | optimizer.py, cost.py | `OMG-Planner/omg/` |
| **Configuration** | config.py | `OMG-Planner/omg/` |
| **Scene Data** | *.mat files | `OMG-Planner/data/scenes/` |
| **Grasp Data** | *.npy files | `OMG-Planner/data/grasps/simulated/` |
| **Object Models** | model_*.obj, *.pth | `OMG-Planner/data/objects/` |
| **Data Generation** | gen_sdf.py, gen_xyz.py | `OMG-Planner/real_world/` |
| **Behavioral Cloning** | bc.py | `GA-DDPG/core/` |
| **Replay Memory** | replay_memory.py | `GA-DDPG/core/` |
| **Expert Integration** | trainer.py | `GA-DDPG/core/` |

---

## Summary

**OMG-Planner** is a sophisticated motion planning system that:

1. **Loads pre-computed grasps** from simulator or external sources
2. **Solves inverse kinematics** to convert task-space to joint-space
3. **Filters grasps** by collision, diversity, and reachability constraints
4. **Optimizes trajectories** using CHOMP algorithm
5. **Generates expert demonstrations** for imitation learning in GA-DDPG
6. **Supports multiple input modalities** (scene files, point clouds, detected grasps)

The system produces high-quality grasp trajectories that serve as supervision signal for training a reinforcement learning policy in the handover task.
