"""Phase-3 RL: online TD3 + BC (GA-DDPG-style blend) on the reactive policy.

See project memory `project_handover_rl` for the locked design decisions:
online off-policy actor-critic, sparse terminal grasp-proximity reward, RL on the
6-DoF Δpose (pose-BC toward OMG) + a learned gripper logit (proximity-synthesized
BCE label + close reward), clock (remaining steps) in BOTH actor and critic,
single-process synchronous loop, optional permanent demo pool.
"""

from handover_sim2real.rl.actor import RLActor, clamp_action
from handover_sim2real.rl.critic import QNetwork
from handover_sim2real.rl.replay_buffer import ReplayBuffer
from handover_sim2real.rl.td3bc_trainer import TD3BCTrainer

__all__ = [
    "RLActor", "clamp_action", "QNetwork", "ReplayBuffer", "TD3BCTrainer",
]
