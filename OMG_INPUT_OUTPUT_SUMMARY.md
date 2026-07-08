# OMG Planner: Input/Output Dimensions & Expert Data

## 1. INPUT DIMENSIONS

### 1.1 Scene Configuration Input
**Source:** MATLAB `.mat` files in `OMG-Planner/data/scenes/`
- Filename: `scene_0.mat`, `demo_scene_0.mat`, etc.

**Input Structure:**
```
scene = sio.loadmat('data/scenes/scene_0.mat')

- scene['pose']              [N, 4, 4]   - Object/obstacle poses in world frame
- scene['path']              [N]         - Model file paths
- scene['goals']             [M, 9]      - Pre-computed grasp configurations (optional)
- scene['reach_grasps']      [M, R, 9]   - Standoff reaching trajectories (optional)
  where M = number of grasps, R = reach trajectory length
- scene['grasp_qualities']   [M]         - Quality scores (0-1)
- scene['grasp_potentials']  [M]         - Cost metrics
- scene['target_name']       str         - Name of target object (e.g., '025_mug')
```

### 1.2 Pre-computed Grasp Poses Input
**Source:** NumPy `.npy` files in `OMG-Planner/data/grasps/simulated/`
- Filename: `003_cracker_box.npy`, `025_mug.npy`, etc.

**Input Structure:**
```
grasp_data = np.load('data/grasps/simulated/025_mug.npy', allow_pickle=True)
grasp_dict = grasp_data.item()

- grasp_dict["transforms"]   [G, 4, 4]   - Gripper poses (hand frame in object frame)
  where G = number of grasps (typically 50-500 per object)
```

**Transform Matrix Format (4x4):**
```
[R(3,3) | t(3,1) ]
[0 0 0  |   1    ]
where:
  - R: Rotation of gripper relative to object
  - t: Translation of gripper relative to object
```

### 1.3 Object Model Data
**Source:** `OMG-Planner/data/objects/<object_id>/`

**Input Structure:**
```
model_normalized.obj                    - Mesh geometry
model_normalized_chomp.pth             - Pre-computed SDF (PyTorch tensor)
model_normalized.extent.txt            - Bounding box dimensions [3]
model_normalized.xyz                   - Point cloud representation
```

### 1.4 Configuration Parameters
**Source:** `OMG-Planner/omg/config.py`

**Key Input Parameters:**
```
cfg.timesteps = 50                      - Trajectory discretization steps
cfg.optim_steps = 50                    - Gradient descent iterations
cfg.collision_point_num = 15            - Collision checking points per link
cfg.goal_set_max_num = 100              - Maximum grasps to consider
cfg.use_standoff = True                 - Enable pre-grasp approach
cfg.standoff_dist = 0.08                - Standoff distance (meters)
cfg.reach_tail_length = 5               - Steps from standoff to grasp
```

### 1.5 Optional: Point Cloud Input (Perception Mode)
**Triggered by:** `python -m omg.core -p`

**Input Structure:**
```
target_point_cloud     [P, 3]           - Segmented target object (x,y,z)
obstacle_point_cloud   [Q, 3]           - Non-target points (obstacles)
where P, Q = number of points (typically 1000-10000)
```

---

## 2. OUTPUT DIMENSIONS

### 2.1 Joint Trajectory Output (PRIMARY OUTPUT)
**Accessible via:** `scene.planner.history_trajectories[-1]`
**Format:** NumPy array

**Output Structure:**
```
trajectory.shape = [T, 9]
where T = timesteps (default 50)

Column breakdown:
- trajectory[:, 0:7]   - 7 arm joint angles (radians)
  * Panda arm: 7-DOF
  * Range: [-2.8966, 2.8966] radians (≈±166°)
  
- trajectory[:, 7:9]   - 2 gripper finger positions (meters)
  * Parallel gripper
  * Range: [0, 0.04] meters (open/close)
```

**Example:**
```
trajectory[0]  = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785, 0.04, 0.04]  # Start config
trajectory[49] = [1.2, 0.8, 0.3, -1.4, 0.5, 1.6, 0.9, 0.0, 0.0]        # Grasp config
```

**Time Discretization:**
- Default timesteps: 50
- Each step represents: trajectory_duration / 50 (typically 0.5-1.0 seconds)

### 2.2 Goal Set Output (Internal)
**Accessible via:** `env.objects[env.target_idx]` or `target_obj`

**Output Structure:**
```
target_obj.grasps              [G', 9]   - Filtered collision-free grasps
target_obj.reach_grasps        [G', R, 9] - Standoff reaching trajectories
target_obj.grasp_potentials    [G']      - Quality scores for each grasp
target_obj.grasp_vis_points    [...]     - Visualization data

where G' ≤ goal_set_max_num (default 100)
      R = reach_tail_length (default 5 steps)
```

**Example:**
```
target_obj.grasps shape: [85, 9]                    # 85 valid grasps
target_obj.reach_grasps shape: [85, 5, 9]           # 85 grasp × 5-step reach
target_obj.grasp_potentials shape: [85]             # Cost for each
```

### 2.3 Planning Information Output (DIAGNOSTICS)
**Accessible via:** `scene.planner.info[-1]`

**Output Structure:**
```
plan_info = {
    'cost': float,                          - Total trajectory cost
    'obs': float,                           - Obstacle collision cost
    'smooth': float,                        - Trajectory smoothness cost
    'grasp': float,                         - Goal preference cost
    'collision_pts': [T, L, C, F]          - Collision point information
}

where:
  T = timesteps (50)
  L = number of links (9 for Panda)
  C = collision_point_num (typically 15)
  F = feature dimension (3D coordinates + normal)
```

**Example:**
```
plan_info = {
    'cost': 2.3456,
    'obs': 1.2,
    'smooth': 0.8,
    'grasp': 0.3456,
    'collision_pts': array of shape [50, 9, 15, 6]
}
```

### 2.4 Video Output (Optional)
**Generated by:** `-w` flag in command
**Output Location:** `OMG-Planner/output_videos/`
**Format:** `.avi` video files
**Content:** 3D rendering of robot, obstacles, and trajectory

---

## 3. HOW OMG PRODUCES EXPERT DATA (PIPELINE)

### Step 1: Load Scene
```
Input: scene_X.mat
├─ Extract object poses [N, 4, 4]
├─ Extract target object name (e.g., '025_mug')
└─ Extract pre-computed grasps [M, 9]
```

### Step 2: Load Grasp Poses
```
Input: data/grasps/simulated/<object_name>.npy
├─ Extract grasp transforms [G, 4, 4]
├─ Apply coordinate rotation: rotZ(π/2) for alignment
└─ Store as task-space grasp poses
```

### Step 3: Solve Inverse Kinematics (IK) - **File:** `planner.py:490`
```
For each grasp_pose in G grasp poses:
├─ Generate standoff pose (backward 0.08m along approach axis)
├─ Create augmented grasps:
│  ├─ Z-upsample: 50 rotations around vertical axis
│  ├─ Y-upsample: 10 rotations around finger axis
│  └─ Flip augmentation: 180° wrist rotations (optional)
├─ Solve IK for each augmented pose
│  ├─ Use 12 seed configurations from anchor points
│  ├─ Use current pose as additional seed
│  └─ Return joint-space solution if successful
└─ Generate reach trajectory from standoff to grasp
```

**Output of IK Stage:**
- reach_grasps: [G_IK, reach_tail_length, 9]  where G_IK ≤ G
- grasps: [G_IK, 9]

### Step 4: Grasp Filtering - **File:** `planner.py:243`
```
For each IK solution:

A. Collision Filtering:
   └─ Check grasp against obstacle SDFs
   └─ Keep if no collision detected

B. Rotation Filtering:
   ├─ Compute rotation difference: R_diff = start_pose.T @ target_pose
   ├─ Extract angle: θ = arccos((trace(R_diff) - 1) / 2)
   └─ Keep if θ < 120° (config: target_hand_filter_angle)

C. Downward Approach Filtering:
   ├─ Extract gripper z-axis: z = target_pose[:3, 0]  (gripper x-axis)
   └─ Keep if z[-1] < -0.3  (cos(70°), enforces downward approach)

D. Duplicate Removal:
   ├─ Compute pairwise distances between remaining grasps
   └─ Keep if distance > 0.5 from existing grasp
```

**Output of Filtering Stage:**
- unique_grasps: [G_unique, 9]  where G_unique ≤ G_IK

### Step 5: Goal Set Diversification - **File:** `planner.py:603`
```
If G_unique > goal_set_max_num (default 100):
├─ Randomly sample goal_set_max_num grasps
└─ Ensure kept grasps are well-distributed

Final goal set: [G_final, 9]  where G_final ≤ 100
```

### Step 6: Trajectory Optimization (CHOMP) - **File:** `optimizer.py`
```
Initialize trajectory:
├─ Start config: Current robot state
├─ Intermediate points: Blend between start and goal
└─ End config: Selected grasp

Iterative Optimization (50 steps):
├─ For t = 1 to optim_steps:
│  ├─ Compute workspace costs:
│  │  ├─ Obstacle cost: SDF penalty for each point
│  │  ├─ Smoothness cost: Trajectory regularization
│  │  └─ Goal cost: Preference for selected grasp
│  ├─ Convert to joint space gradient
│  ├─ Apply gradient descent: θ ← θ - step_size * ∇θ
│  └─ Project trajectory to goal set constraints
│
└─ Return optimized trajectory [50, 9]
```

**Output:** Final trajectory [50, 9]

### Step 7: Package Expert Data
```
Create expert demonstration containing:
├─ trajectory: [50, 9]              - Full state/action sequence
├─ expert_flag: 1.0                 - Mark as expert (vs. learned)
├─ expert_action: trajectory[:-1]   - For BC loss
├─ next_expert_action: trajectory[1:] - For multi-step prediction
├─ scene_info: object poses, target name
└─ metadata: costs, planning info, quality metrics
```

---

## 4. EXPERT DATA COMPOSITION

### 4.1 What is Expert Data?
Expert data consists of **demonstration trajectories** generated by the OMG planner for training the RL agent through **behavioral cloning**.

### 4.2 Single Expert Demonstration
```
{
    'state': array[50, state_dim],           - Observation sequence
    'trajectory': array[50, 9],              - Action/configuration sequence
    'expert_action': array[49, 9],           - Teacher forcing actions
    'next_expert_action': array[49, 9],      - Next-step guidance
    'expert_flag': 1.0,                      - Mark: 1.0=expert, 0.0=learned
    'cost_info': {                           - Planning diagnostics
        'total_cost': float,
        'obstacle_cost': float,
        'smoothness_cost': float,
        'grasp_cost': float
    },
    'metadata': {
        'scene_idx': int,                    - Which scene (0-99)
        'object_name': str,                  - Target object
        'grasp_idx': int,                    - Which grasp was selected
        'num_grasps_available': int          - Goal set size
    }
}
```

### 4.3 Expert Dataset Structure
**Location:** Stored in GA-DDPG replay memory
**Size:** 100 scenes × (1-5 planning attempts per scene) = ~500 trajectories

**Memory Format (GA-DDPG)** - **File:** `GA-DDPG/core/replay_memory.py`
```
class BaseMemory:
    self.state              array[buffer_size, state_dim]
    self.next_state         array[buffer_size, state_dim]
    self.action             array[buffer_size, 9]         - OMG trajectory
    self.expert_flags       array[buffer_size]            - 1.0 if expert
    self.expert_action      array[buffer_size, 9]         - BC target
    self.next_expert_action array[buffer_size, 9]         - Next-step BC target
    self.done               array[buffer_size]
    self.reward             array[buffer_size]
    self.step_type          array[buffer_size]
```

### 4.4 Expert Data Characteristics

**Trajectory Quality:**
- **Collision-free:** All grasps filtered for obstacles
- **Reachable:** IK solutions verified for robot kinematics
- **Diverse:** ~100 different grasps per object due to goal set diversity
- **Optimized:** CHOMP produces smooth, efficient paths

**Trajectory Statistics:**
```
Length:               50 timesteps
Joint bounds:        [-2.8966, 2.8966] radians (arm)
                     [0, 0.04] meters (gripper)
Typical cost range:  0.5 - 5.0 (dimensionless)
Obstacle cost:       0.0 - 3.0 (penalty)
Smoothness cost:     0.1 - 2.0 (regularization)
Planning time:       0.5 - 2.0 seconds per trajectory
```

**Diversity Sources:**
```
1. Scene diversity:      100 different table-top scenes
2. Grasp diversity:      100 grasps per object (after filtering)
3. IK augmentation:      50 z-rotations × 10 y-rotations = 500 candidates
4. Filtering diversity:  Collision, reachability, rotation limits
5. Optimization variance: Different random seeds, gradient descent paths
```

### 4.5 Integration with Learning Pipeline

**Step 1: Data Collection**
```
For scene_i in range(100):
    config.scene_file = f'scene_{scene_i}'
    scene = PlanningScene(config)
    trajectory = scene.step()
    expert_data.append({
        'trajectory': trajectory,
        'expert_flag': 1.0,
        'metadata': {...}
    })
```

**Step 2: Load into Replay Memory**
```
for exp_data in expert_data:
    memory.push(
        state=current_state,
        action=exp_data['trajectory'],
        expert_flags=1.0,
        expert_action=exp_data['trajectory'],
        next_expert_action=exp_data['trajectory'][1:],
        reward=compute_reward(...),
        done=False
    )
```

**Step 3: Behavioral Cloning Loss** - **File:** `GA-DDPG/core/bc.py`
```
For mini-batch of expert transitions:
    
    π_predictions = policy(state)           # Predicted action
    expert_actions = batch['expert_action'] # OMG action
    
    L_BC = ||π_predictions - expert_actions||²  # MSE loss
    
    Gradient: ∇L_BC = 2 * (π - expert_action)
    
    policy_params ← policy_params - α * ∇L_BC
```

**Step 4: Combined RL + BC Training** (optional)
```
L_total = λ_RL * L_RL + (1 - λ_RL) * L_BC

where λ_RL anneals from 0.0 → 1.0 during training
- Early training: Use more expert supervision (L_BC)
- Late training: Use more RL reward signal (L_RL)
```

---

## 5. SUMMARY TABLE

| Component | Input Dimensions | Output Dimensions | Format |
|-----------|-----------------|-------------------|--------|
| **Scene Config** | [N, 4, 4] poses + metadata | - | MATLAB .mat |
| **Grasp Poses** | [G, 4, 4] transforms | - | NumPy .npy |
| **SDF/Collision** | [G, 4, 4] poses | [T, C, 3] collision pts | PyTorch .pth |
| **IK Solver** | [G, 4, 4] task-space | [G', 9] joint-space | Numeric |
| **Filtering** | [G', 9] grasps | [G'', 9] filtered | Numeric |
| **Optimization** | [G'', 50, 9] trajectory | [50, 9] **FINAL** | NumPy |
| **Expert Demo** | - | [50, 9] traj + metadata | Dict/JSON |

---

## 6. KEY FILES REFERENCE

| File | Purpose | Key Function |
|------|---------|--------------|
| [omg/core.py](OMG-Planner/omg/core.py#L782) | Entry point | PlanningScene.step() |
| [omg/planner.py](OMG-Planner/omg/planner.py#L90) | Main planning | Planner.plan() |
| [omg/optimizer.py](OMG-Planner/omg/optimizer.py) | CHOMP optimizer | Optimizer.optimize() |
| [omg/cost.py](OMG-Planner/omg/cost.py) | Cost computation | Cost.batch_obstacle_cost() |
| [GA-DDPG/core/bc.py](GA-DDPG/core/bc.py) | Behavioral cloning | BC agent training |
| [GA-DDPG/core/replay_memory.py](GA-DDPG/core/replay_memory.py#L20) | Expert storage | BaseMemory class |

