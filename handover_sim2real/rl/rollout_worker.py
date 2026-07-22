"""
Phase-3 RL rollout worker (single-process, synchronous).

Drives the handover sim with the current actor, logs off-policy transitions with
the sparse terminal handover-success reward, and returns them for the replay
buffer. Reuses the *exact* observation builders / IK / geometry helpers from the
offline collectors so states are byte-identical to what BC trained on.

Per step:
  • clock  `remain = max_steps − step`  (fed to actor/critic, normalized /max_steps)
  • actor  outputs a 7-D action = Δpose(6, normalized) + gripper logit(1); we add
    exploration noise, clamp, denormalize the pose, and set the gripper OPEN iff
    logit ≥ 0. Or, with prob β, execute one corrective OMG step (approach, gripper
    open). During the **reverse-curriculum warm start** (first `expert_initial_steps`
    of an episode) the FULL step-0 OMG trajectory is followed BY INDEX so the EE
    traverses to the grasp and the policy takes over near it (GA-DDPG expert_initial).

`expert_rollout_episode()` is the complementary non-explore rollout: it plays the
whole OMG trajectory to the grasp, closes, and scores the same reward — a
guaranteed fresh success that anchors the online buffer's +reward fraction. It is
also what the offline demo collector calls, so demos and online experts are
byte-identical.
  • the expert pose **label** (actor pose-BC term) is PLAN-TRACKING (GA-DDPG
    train_online.py `expert_plan[int(step)]`): OMG plans ONCE at step 0 (the
    committed plan) and the label at step t = the delta from the CURRENT state to
    `plan[t]`, so the label target advances THROUGH the standoff INTO the grasp
    as the episode advances. With per-step prob `dagger_ratio` (only inside
    (dagger_min_step, len(plan)−dagger_tail_guard)) the plan TAIL is re-fitted to
    the policy's drifted state over the REMAINING steps and spliced in — never
    within `dagger_tail_guard` steps of the plan end, so the standoff→grasp reach
    labels stay committed. The OLD label — `plan[0]` of a fresh short-horizon
    replan every step — had a stationary attractor at the OMG standoff (0.08 m
    short of the grasp): the first waypoint never enters the 5-step reach tail
    and its magnitude Zeno-decays at the standoff (floored free portion), which
    taught the policy to hover 0.06–0.11 m short (rl_run7 plateau, broken BC base).
    The same plan supplies the grasp pose (reward proximity check + goal-auxiliary
    target + the proximity gripper label: P(open)=1 when far from the grasp, 0
    within proximity — a training-only signal for the actor's gripper BCE, never
    an execution override).
  • DART (GA-DDPG `env.random_perturb`): on a fraction `dart_ratio` of POLICY
    steps inside [dart_min_step, dart_max_step) the executed action is REPLACED by
    a random task-space jump (±dart_pos_mag m, ±dart_rot_mag rad, gripper open),
    jolting the EE off-plan so the FOLLOWING steps' plan-tracking labels demonstrate
    the RECOVERY. This manufactures the off-plan near-grasp coverage that warm-start
    + DAgger never produce (both stay on-plan → the from-scratch policy that arrives
    a few cm off LATERALLY has never seen a near-grasp lateral correction). The
    window sits just BEFORE the reach tail and inside the DAgger window so a splice
    can re-fit a feasible recovery. The perturbed transition carries `perturb_flag`
    = 1 → the trainer DROPS it from the Bellman critic fit (its stored action is
    artificial) but KEEPS it for the actor BC. `dart_ratio` 0 = disabled.

Reward (no carry-to-goal / no benchmark SUCCESS): the episode is about reaching
the grasp and committing the close at the right pose.
  • the policy commits a close (logit < 0) →  terminal;  reward = 1 if the EE is
    within (close_pos_thresh, close_rot_thresh) of the OMG grasp pose, else 0.
  • benchmark failure (human contact / drop) → terminal, reward 0.
  • step horizon reached without closing → terminal, reward 0.
The in-state clock makes the horizon a genuine terminal (single `terminal` flag).

After the episode, discounted Monte-Carlo returns are filled in per transition
(for the critic's optional Bellman⊕MC target blend).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet
import torch
from scipy.spatial.transform import Rotation as Rot

# Reuse the offline collectors' shared helpers (state builders, IK, geometry).
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from collect_bc_dataset import (  # noqa: E402
    _point_cloud, _robot_state, ee_grasp_pose_error,
)
from rollout_bc_policy import action_to_target_joint  # noqa: E402

from handover.benchmark_wrapper import EpisodeStatus  # noqa: E402


def _status_name(s: int) -> str:
    parts = []
    if s & EpisodeStatus.FAILURE_HUMAN_CONTACT:
        parts.append("HUMAN_CONTACT")
    if s & EpisodeStatus.FAILURE_OBJECT_DROP:
        parts.append("DROP")
    if s & EpisodeStatus.FAILURE_TIMEOUT:
        parts.append("BENCH_TIMEOUT")
    return "|".join(parts) if parts else f"STATUS({s})"


def _ee_world_mat(obs) -> np.ndarray:
    body = obs["panda_body"]; link = obs["panda_link_ind_hand"]
    pos = np.asarray(body.link_state[0, link, 0:3], dtype=np.float64)
    quat = np.asarray(body.link_state[0, link, 3:7], dtype=np.float64)  # xyzw
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def _ee_to_grasp_9d(obs, grasp_world) -> np.ndarray:
    """The final grasp pose expressed relative to the current EE, as
    pos(3)+rot6d(6) (the first two columns of the rotation matrix — Zhou et al.).
    This is the goal-auxiliary target: the point cloud is in the EE frame, so the
    EE-relative grasp is a function of the visible object → learnable. Frame-
    invariant (identical whether the inputs are world or base frame)."""
    T_rel = np.linalg.inv(_ee_world_mat(obs)) @ np.asarray(grasp_world, dtype=np.float64)
    R = T_rel[:3, :3]
    return np.concatenate([T_rel[:3, 3], R[:, 0], R[:, 1]]).astype(np.float32)


def _rot6d_to_matrix(col0, col1) -> np.ndarray:
    """Invert the rot6d encoding of `_ee_to_grasp_9d`: Gram-Schmidt the two
    predicted columns back into an orthonormal rotation matrix (Zhou et al.)."""
    a1 = np.asarray(col0, dtype=np.float64)
    a2 = np.asarray(col1, dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def grasp_9d_to_world(obs, goal9) -> np.ndarray:
    """Turn an EE-relative grasp pose (pos+rot6d — the goal-aux head's output, or
    an `_ee_to_grasp_9d` target) into a world-frame 4x4, so the policy's PREDICTED
    grasp can be drawn alongside the true OMG goal grasp. Inverse of
    `_ee_to_grasp_9d`: `grasp_world = EE_world @ T_rel`."""
    goal9 = np.asarray(goal9, dtype=np.float64)
    T_rel = np.eye(4)
    T_rel[:3, :3] = _rot6d_to_matrix(goal9[3:6], goal9[6:9])
    T_rel[:3, 3] = goal9[:3]
    return _ee_world_mat(obs) @ T_rel


def _grasp_reached(obs, grasp_pose, pos_thresh, rot_thresh) -> bool:
    """True if the EE is within (pos, rot) tolerance of the OMG grasp pose —
    i.e. a close committed here is a *correct* grasp (the `proximity` reward)."""
    if grasp_pose is None:
        return False
    pos_err, rot_err = ee_grasp_pose_error(obs, grasp_pose)
    return bool(pos_err <= pos_thresh and rot_err <= rot_thresh)


def grasp_held_after_hold(env, obs, steps_action_repeat, hold_steps):
    """Contact-hold grasp-success (the `stable_grasp` reward, cf. GA-DDPG's
    lift test — no OMG grasp pose needed). After a committed close, hold the
    gripper shut in place for `hold_steps` policy-steps, then report whether the
    object ended up *secured*: handover-sim's own release handshake fired
    (`ycb.released` — the human only lets go once the robot truly grips) AND the
    object was not dropped / no human-contact failure during the hold.
    Returns (held: bool, obs)."""
    hold_action = np.concatenate([np.zeros(6, dtype=np.float32), [0.0]]).astype(np.float32)  # 0=close
    failed = False
    for _ in range(int(hold_steps)):
        hold_jp = action_to_target_joint(hold_action, obs)
        done = False
        st = 0
        for _ in range(int(steps_action_repeat)):
            obs, _, done, info = env.step(hold_jp)
            if done:
                st = int(info.get("status", 0))
                break
        failed |= bool(st & (EpisodeStatus.FAILURE_OBJECT_DROP
                             | EpisodeStatus.FAILURE_HUMAN_CONTACT))
        if done:
            break
    released = bool(env.ycb.released)
    dropped  = bool(getattr(env, "_dropped", False))
    return (released and not dropped and not failed), obs


class RolloutWorker:
    def __init__(self, env, point_listener, cfg, normalizer, device,
                 max_steps: int, gamma: float,
                 act_limit: float = 5.0,
                 close_pos_thresh: float = 0.02, close_rot_thresh: float = 0.34,
                 reward_mode: str = "proximity", hold_steps: int = 3,
                 dagger_min_step: int = 5, dagger_tail_guard: int = 8,
                 dart_min_step: int = 15, dart_max_step: int = 22,
                 dart_pos_mag: float = 0.04, dart_rot_mag: float = 0.2):
        self.env = env
        self.point_listener = point_listener
        self.cfg = cfg
        self.normalizer = normalizer
        self.device = device
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.act_limit = float(act_limit)
        self.close_pos_thresh = float(close_pos_thresh)
        self.close_rot_thresh = float(close_rot_thresh)
        self.reward_mode = str(reward_mode)      # "proximity" | "stable_grasp"
        self.hold_steps = int(hold_steps)        # stable_grasp: policy-steps to hold after close
        # DAgger tail-replan window (GA-DDPG get_flags): replans allowed only for
        # dagger_min_step < step < len(plan) - dagger_tail_guard. The guard keeps
        # the standoff->grasp reach labels committed AND guarantees every replan
        # has enough remaining budget for OMG's plan structure (free portion +
        # 5-step reach tail).
        self.dagger_min_step = int(dagger_min_step)
        self.dagger_tail_guard = int(dagger_tail_guard)
        # DART (GA-DDPG random_perturb): on a fraction `dart_ratio` of POLICY steps
        # inside [dart_min_step, dart_max_step) inject a random task-space jump
        # (±dart_pos_mag m, ±dart_rot_mag rad) instead of the policy action. The
        # jump lands the EE OFF-PLAN so the next step's plan-tracking label (and, if
        # still warming, the by-index expert) demonstrates the RECOVERY — the one
        # thing DAgger/warm-start never cover near the grasp (both stay on-plan).
        # The window sits BEFORE the reach tail (last ~5 steps) on purpose: perturb
        # around the standoff entry, recover+realign, THEN descend straight in.
        self.dart_min_step = int(dart_min_step)
        self.dart_max_step = int(dart_max_step)
        self.dart_pos_mag  = float(dart_pos_mag)
        self.dart_rot_mag  = float(dart_rot_mag)

        self.panda_base_inv_tf = pybullet.invertTransform(
            cfg.ENV.PANDA_BASE_POSITION, cfg.ENV.PANDA_BASE_ORIENTATION)
        self.steps_action_repeat = int(
            cfg.POLICY.TIME_ACTION_REPEAT / cfg.SIM.TIME_STEP)

        # numpy normalization constants (avoid tensor/device round-trips).
        self.a_mean = np.asarray(normalizer.action_mean, dtype=np.float32)
        self.a_std  = np.asarray(normalizer.action_std,  dtype=np.float32)

    def _denorm(self, a_norm: np.ndarray) -> np.ndarray:
        return a_norm * self.a_std + self.a_mean

    def _norm(self, a_real: np.ndarray) -> np.ndarray:
        return (a_real - self.a_mean) / self.a_std

    @torch.no_grad()
    def rollout_episode(self, actor, scene_idx, rng,
                        beta: float = 0.0, expert_initial_steps: int = 0,
                        noise_std: float = 0.1, on_grasp=None,
                        on_policy_grasp=None,
                        dagger_ratio: float = 0.0,
                        dart_ratio: float = 0.0) -> tuple[list[dict], dict]:
        """Roll out one episode. Returns (transitions, stats). transitions is
        empty (and stats['skipped']=True) if OMG can't plan on the first step.
        `dagger_ratio` is the per-step probability of a DAgger TAIL replan
        (training-time data collection only — leave 0 for eval so the reward is
        scored against the step-0 goal grasp and no OMG time is spent).
        `on_grasp(grasp_pose_world_4x4)`, if given, is called once the OMG goal
        grasp is known (step 0, before the robot moves) — used by the GUI viewer
        to overlay the target grasp.
        `on_policy_grasp(grasp_pose_world_4x4)`, if given, is called EVERY step
        with the actor's own goal-aux prediction (the EE-relative grasp pose it
        regresses from the point cloud, decoded to world) — the GUI viewer draws
        it so you can watch the policy's estimate converge to the true grasp. It
        costs one extra aux-head forward, so leave it None off the viewer path."""
        actor.eval()
        obs = self.env.reset(idx=scene_idx)
        self.point_listener.reset()

        prev_act6d = np.zeros(6, dtype=np.float32)
        grasp_pose = None      # OMG goal grasp (traj[-1]) — reward + aux target
        full_plan = None       # the COMMITTED OMG plan: labels + warm start follow
                               # it BY INDEX; DAgger replans splice a new TAIL
                               # (never the last dagger_tail_guard steps)

        pc = _point_cloud(obs, self.point_listener, self.panda_base_inv_tf)
        rs = _robot_state(obs, prev_act6d)

        transitions: list[dict] = []
        n_omg_fail = 0
        n_replans = 0
        n_perturb = 0
        status = 0
        done = False
        min_pos = float("inf")   # closest EE->grasp position (m) reached this episode
        min_rot = float("inf")   # closest EE->grasp rotation (rad)
        closed_any = False       # did the policy ever commit a close?

        for step in range(self.max_steps):
            remain = self.max_steps - step
            remain_norm = remain / self.max_steps

            # ----- committed OMG plan (GA-DDPG train_online.py): plan ONCE at
            # step 0; DAgger re-fits only the TAIL, never near the plan end -----
            if step == 0:
                plan, _ = self.env.run_omg_planner(
                    int(self.max_steps), scene_idx, reset_scene=True)
                if plan is None:
                    return [], {"skipped": True, "scene_idx": scene_idx}
                full_plan = np.asarray(plan)
                grasp_pose = self.env.get_omg_goal_grasp_pose()
                if grasp_pose is None:
                    return [], {"skipped": True, "scene_idx": scene_idx}
                if on_grasp is not None:
                    on_grasp(grasp_pose)
            elif (dagger_ratio > 0.0
                  and step > self.dagger_min_step
                  and step < len(full_plan) - self.dagger_tail_guard
                  and rng.uniform() < dagger_ratio):
                # DAgger tail replan (GA-DDPG `expert_plan(step=MAX_STEP-step-1)`
                # + splice): re-fit the REST of the plan to the policy's drifted
                # state, horizon = the REMAINING episode steps — no floored
                # short horizon, so per-step label magnitude stays ≥ d/remaining
                # (no Zeno decay). On failure keep the old plan.
                rest, _ = self.env.run_omg_planner(
                    int(self.max_steps - step), scene_idx, reset_scene=False)
                if rest is None:
                    n_omg_fail += 1
                else:
                    full_plan = np.concatenate(
                        [full_plan[:step], np.asarray(rest)])
                    n_replans += 1
                    g = self.env.get_omg_goal_grasp_pose()
                    if g is not None:
                        grasp_pose = g   # replan may re-select the goal grasp;
                                         # keep reward/labels/aux consistent

            # ----- expert label: PLAN-TRACKING delta from the CURRENT state to
            # the committed plan's waypoint for THIS timestep (GA-DDPG
            # `expert_plan[int(step)]`). The label target advances INTO the
            # grasp as the episode advances — near the end it demonstrates the
            # standoff->grasp reach instead of "stay at the standoff" (the old
            # fresh-replan plan[0] label Zeno-stalled there and taught the
            # rl_run7 hover). Past the plan end it holds the grasp waypoint.
            # Deliberate deviation from GA-DDPG (which BC-masks non-dagger
            # explore steps): we label EVERY step — with the clock in the state
            # the time-indexed label is self-consistent, and the drifted-state
            # pursuit labels are the on-policy corrective signal the BC base
            # never got; the tail-guarded splices keep them from going stale.
            idx = min(step, len(full_plan) - 1)
            expert_delta = np.asarray(
                self.env.convert_target_joint_position_to_action(full_plan[idx]),
                dtype=np.float32)  # [6] real units
            expert_action_norm = self._norm(expert_delta).astype(np.float32)
            expert_flag = 1.0

            # Goal-auxiliary target: EE-relative final grasp pose (pos+rot6d) for
            # THIS state's EE (computed before env.step reassigns obs).
            goal9 = _ee_to_grasp_9d(obs, grasp_pose)

            # Proximity gripper LABEL for THIS state (training-only supervision;
            # never overrides the policy's executed logit). target = P(open): 1
            # when far from the grasp, 0 within (close_pos_thresh, close_rot_thresh).
            near = _grasp_reached(obs, grasp_pose,
                                  self.close_pos_thresh, self.close_rot_thresh)
            expert_gripper = 0.0 if near else 1.0
            gripper_flag = 1.0

            # ----- actor action (normalized) + exploration noise -----
            pc_t = torch.from_numpy(pc).float().unsqueeze(0).to(self.device)
            rs_t = torch.from_numpy(rs).float().unsqueeze(0).to(self.device)
            rs_n = (self.normalizer.normalize_state(rs_t)
                    if self.normalizer is not None else rs_t)
            remain_t = torch.tensor([[remain_norm]], dtype=torch.float32,
                                    device=self.device)
            if on_policy_grasp is not None:
                # viewer path: also pull the goal-aux head's grasp-pose estimate
                # and hand it back (decoded to a world 4x4) for overlay. `obs` is
                # still the CURRENT state, so the EE frame matches the prediction.
                a_out, aux_out = actor(pc_t, rs_n, remain_t, return_aux=True)
                a_norm = a_out[0].cpu().numpy()  # [7]
                on_policy_grasp(grasp_9d_to_world(obs, aux_out[0].cpu().numpy()))
            else:
                a_norm = actor(pc_t, rs_n, remain_t)[0].cpu().numpy()  # [7]
            if noise_std > 0.0:
                a_norm = a_norm + rng.normal(0.0, noise_std, size=7).astype(np.float32)
            a_norm = np.clip(a_norm, -self.act_limit, self.act_limit).astype(np.float32)

            # ----- choose executed action: expert (warm start / β) or actor ---
            # Executed gripper: OPEN (finger 0.04) iff logit ≥ 0, else CLOSE.
            warmup = step < expert_initial_steps
            use_expert = warmup or (beta > 0.0 and rng.uniform() < beta)
            is_dart = False
            if use_expert:
                # follow the COMMITTED plan BY INDEX: the reverse-curriculum warm
                # start (GA-DDPG expert_initial) traverses the EE to the grasp so
                # the policy takes over near it (and can finish + close → earns
                # reward, buf_pos>0); a β-step snaps the EE back onto the plan
                # mid-episode. Executed action == the label (delta to plan[idx]).
                target_jp   = full_plan[idx]
                exec_delta6 = expert_delta
                # stored 7-D action: normalized expert pose + an "open" gripper logit
                stored_action = np.concatenate(
                    [expert_action_norm, [self.act_limit]]).astype(np.float32)
                committed_close = False
            elif (dart_ratio > 0.0
                  and self.dart_min_step <= step < self.dart_max_step
                  and rng.uniform() < dart_ratio):
                # DART perturbation (GA-DDPG env.random_perturb): REPLACE the policy
                # action with a random task-space jump so the EE lands off-plan and
                # the following steps' plan-tracking labels demonstrate the recovery
                # (the off-plan near-grasp coverage DAgger/warm-start never produce).
                # Gripper stays OPEN (never close on a jolt). The stored action is
                # this jump; perturb_flag=1 drops the row from the critic's Bellman
                # fit (its Q(s,a) is meaningless) while the label still feeds BC.
                is_dart = True
                n_perturb += 1
                exec_delta6 = np.concatenate([
                    rng.uniform(-self.dart_pos_mag, self.dart_pos_mag, size=3),
                    rng.uniform(-self.dart_rot_mag, self.dart_rot_mag, size=3),
                ]).astype(np.float32)
                committed_close = False
                action7 = np.concatenate([exec_delta6, [1.0]]).astype(np.float32)
                target_jp = action_to_target_joint(action7, obs)
                stored_action = np.concatenate(
                    [self._norm(exec_delta6), [self.act_limit]]).astype(np.float32)
            else:
                exec_delta6 = self._denorm(a_norm[:6]).astype(np.float32)
                committed_close = float(a_norm[6]) < 0.0
                gbit = 0.0 if committed_close else 1.0    # 0 close / 1 open
                action7 = np.concatenate([exec_delta6, [gbit]]).astype(np.float32)
                target_jp = action_to_target_joint(action7, obs)
                stored_action = a_norm

            prev_act6d = exec_delta6.copy()
            closed_any = closed_any or committed_close

            # ----- step the sim -----
            status = 0
            done = False
            for _ in range(self.steps_action_repeat):
                obs, _, done, info = self.env.step(target_jp)
                if done:
                    status = int(info.get("status", 0))
                    break

            # closest approach to the grasp (diagnostic: reaching vs closing).
            # ONLY over policy-controlled steps (step >= expert_initial_steps):
            # during the reverse-curriculum warm start the EXPERT drives the EE to
            # the grasp, so including those steps would report the expert's reach,
            # not the policy's (rl_run6: roll_min_pos looked ~0.05 while eval, which
            # has no warm start, was ~0.25). NOTE: the policy still INHERITS the
            # warm-start hand-off position, so eval_min_pos (ei=0) remains the true
            # from-scratch reach; this is the policy's closest while in control.
            if grasp_pose is not None and not warmup:
                pe, re = ee_grasp_pose_error(obs, grasp_pose)
                min_pos = min(min_pos, pe)
                min_rot = min(min_rot, re)

            # ----- reward + terminal (grasp-proximity + close; no carry) -----
            reward = 0.0
            reason = ""
            if committed_close:
                # stable_grasp: hold the close and verify the object is secured
                # (handover-sim release + no drop); proximity: EE within tol of
                # the OMG grasp pose. Both terminal; `obs` advances under the hold.
                if self.reward_mode == "stable_grasp":
                    held, obs = grasp_held_after_hold(
                        self.env, obs, self.steps_action_repeat, self.hold_steps)
                else:
                    held = _grasp_reached(obs, grasp_pose,
                                          self.close_pos_thresh, self.close_rot_thresh)
                reward = 1.0 if held else 0.0
                reason = "GRASP_OK" if held else "GRASP_MISS"
            elif done:
                reason = _status_name(status)             # human contact / drop
            elif step == self.max_steps - 1:
                reason = "TIMEOUT"

            ep_done = committed_close or done or (step == self.max_steps - 1)
            terminal = 1.0 if ep_done else 0.0

            next_pc = _point_cloud(obs, self.point_listener, self.panda_base_inv_tf)
            next_rs = _robot_state(obs, prev_act6d)
            next_remain = self.max_steps - (step + 1)
            next_remain_norm = max(next_remain, 0) / self.max_steps
            # EE-relative grasp at the NEXT state — the potential-shaping term needs
            # Φ(s') (zeroed at terminals via 1−terminal, so the value there is moot).
            next_goal9 = (_ee_to_grasp_9d(obs, grasp_pose) if grasp_pose is not None
                          else np.zeros(9, dtype=np.float32))

            transitions.append(dict(
                pc=pc, rs=rs, remain_norm=remain_norm, action=stored_action,
                reward=reward, next_pc=next_pc, next_rs=next_rs,
                next_remain_norm=next_remain_norm, terminal=terminal,
                mc_return=0.0, expert_action=expert_action_norm,
                expert_flag=expert_flag, goal_pose=goal9, next_goal_pose=next_goal9,
                expert_gripper=expert_gripper, gripper_flag=gripper_flag,
                perturb_flag=(1.0 if is_dart else 0.0)))

            pc, rs = next_pc, next_rs
            if ep_done:
                break

        # ----- discounted Monte-Carlo returns (backward) -----
        G = 0.0
        for tr in reversed(transitions):
            G = tr["reward"] + self.gamma * (1.0 - tr["terminal"]) * G
            tr["mc_return"] = G

        success = int(bool(transitions) and transitions[-1]["reward"] >= 1.0)
        stats = {
            "skipped": False,
            "scene_idx": scene_idx,
            "length": len(transitions),
            "success": success,
            "return": float(sum(tr["reward"] for tr in transitions)),
            "reason": reason if transitions else "EMPTY",
            "n_omg_fail": n_omg_fail,
            "n_replans": n_replans,
            "n_perturb": n_perturb,
            "min_pos": float(min_pos) if min_pos < float("inf") else float("nan"),
            "min_rot": float(min_rot) if min_rot < float("inf") else float("nan"),
            "closed": int(closed_any),
            "expert_episode": False,
        }
        return transitions, stats

    @torch.no_grad()
    def expert_rollout_episode(self, scene_idx, rng=None,
                               dart_ratio: float = 0.0) -> tuple[list[dict], dict]:
        """Full-EXPERT episode (GA-DDPG's non-explore rollout, `train_online.py`):
        play the ENTIRE OMG trajectory by index so the EE actually reaches the
        grasp (the online replan-first-waypoint expert Zeno-stalls and never
        arrives), then commit the close and score it with the SAME reward as the
        policy (`stable_grasp` contact-hold, or OMG-pose proximity). Every episode
        is a guaranteed fresh success/attempt that anchors the online buffer's
        +reward fraction — the online policy earns ~none on its own, so without
        this the critic has no positive signal at online states and the policy
        decays. Transition format is byte-identical to `rollout_episode` (and to
        the offline demo pool, which now calls this same method), so they mix
        freely. Returns (transitions, stats); empty + stats['skipped']=True if OMG
        can't plan / no grasp pose on this scene.

        DART (`dart_ratio>0`, GA-DDPG's FAITHFUL non-explore DART): with per-step
        prob `dart_ratio` inside [dart_min_step, dart_max_step), JOLT the EE off-plan
        by a random task-space step (±dart_pos_mag m / ±dart_rot_mag rad) — an
        OUT-OF-BAND perturbation, NOT a recorded transition — then REPLAN the OMG
        tail from the perturbed state and let the EXPERT drive the recovery by index.
        The recorded steps are thus the expert's CLEAN correction from an off-plan
        state (consistent dynamics → no critic masking needed, unlike the policy-side
        variant). This is canonical DART: noise-injected expert demonstrations that
        SHOW the recovery, manufacturing the off-plan near-grasp coverage the on-plan
        by-index playback never has. Needs `rng`; leave dart_ratio=0 for the offline
        demo pool (collect_rl_demos) so that permanent +1 anchor stays clean."""
        obs = self.env.reset(idx=scene_idx)
        self.point_listener.reset()
        plan, _ = self.env.run_omg_planner(int(self.max_steps), scene_idx,
                                           reset_scene=True)
        if plan is None:
            return [], {"skipped": True, "scene_idx": scene_idx}
        grasp_pose = self.env.get_omg_goal_grasp_pose()
        if grasp_pose is None:
            return [], {"skipped": True, "scene_idx": scene_idx}

        max_steps = int(self.max_steps)
        plan = np.asarray(plan)
        prev_act6 = np.zeros(6, dtype=np.float32)
        pc = _point_cloud(obs, self.point_listener, self.panda_base_inv_tf)
        rs = _robot_state(obs, prev_act6)

        transitions: list[dict] = []
        done = False
        min_pos = float("inf")
        min_rot = float("inf")
        n_perturb = 0
        n_omg_fail = 0
        for step in range(max_steps):
            if step >= len(plan):
                break                       # played through the grasp waypoint

            # ----- DART (GA-DDPG faithful, non-explore path): JOLT off-plan, REPLAN
            # the tail, EXPERT recovers by index. The jolt is OUT-OF-BAND (extra sim
            # step, NOT recorded); the recorded steps below are the expert's clean
            # recovery from the perturbed state (consistent dynamics → no masking).
            if (dart_ratio > 0.0 and rng is not None
                    and self.dart_min_step <= step < self.dart_max_step
                    and rng.uniform() < dart_ratio):
                n_perturb += 1
                pdelta = np.concatenate([
                    rng.uniform(-self.dart_pos_mag, self.dart_pos_mag, size=3),
                    rng.uniform(-self.dart_rot_mag, self.dart_rot_mag, size=3),
                ]).astype(np.float32)
                pjp = action_to_target_joint(
                    np.concatenate([pdelta, [1.0]]).astype(np.float32), obs)  # open
                for _ in range(self.steps_action_repeat):
                    obs, _, done, info = self.env.step(pjp)
                    if done:
                        break
                if done:
                    break                   # jolt ended the episode (collision) — abort
                rest, _ = self.env.run_omg_planner(int(max_steps - step), scene_idx,
                                                   reset_scene=False)
                if rest is None:
                    n_omg_fail += 1         # keep old plan; expert recovers toward it
                else:
                    plan = np.concatenate([plan[:step], np.asarray(rest)])
                    g = self.env.get_omg_goal_grasp_pose()
                    if g is not None:
                        grasp_pose = g
                prev_act6 = pdelta.copy()    # the jolt is the last executed action
                pc = _point_cloud(obs, self.point_listener, self.panda_base_inv_tf)
                rs = _robot_state(obs, prev_act6)
                if step >= len(plan):
                    break

            remain_norm = (max_steps - step) / max_steps
            goal9 = _ee_to_grasp_9d(obs, grasp_pose)
            near = _grasp_reached(obs, grasp_pose,
                                  self.close_pos_thresh, self.close_rot_thresh)

            # ----- OMG approach step (gripper OPEN), followed BY INDEX -----
            exec_delta6 = np.asarray(
                self.env.convert_target_joint_position_to_action(plan[step]),
                dtype=np.float32)
            stored_action = np.concatenate(
                [self._norm(exec_delta6), [self.act_limit]]).astype(np.float32)
            prev_act6 = exec_delta6.copy()
            for _ in range(self.steps_action_repeat):
                obs, _, done, info = self.env.step(plan[step])
                if done:
                    break

            pe, re = ee_grasp_pose_error(obs, grasp_pose)
            min_pos = min(min_pos, pe)
            min_rot = min(min_rot, re)

            next_pc = _point_cloud(obs, self.point_listener, self.panda_base_inv_tf)
            next_rs = _robot_state(obs, prev_act6)
            next_remain_norm = max(max_steps - (step + 1), 0) / max_steps
            next_goal9 = _ee_to_grasp_9d(obs, grasp_pose)
            transitions.append(dict(
                pc=pc, rs=rs, remain_norm=remain_norm, action=stored_action,
                reward=0.0, next_pc=next_pc, next_rs=next_rs,
                next_remain_norm=next_remain_norm, terminal=(1.0 if done else 0.0),
                mc_return=0.0, expert_action=self._norm(exec_delta6).astype(np.float32),
                expert_flag=1.0, goal_pose=goal9, next_goal_pose=next_goal9,
                expert_gripper=(0.0 if near else 1.0), gripper_flag=1.0,
                perturb_flag=0.0))
            pc, rs = next_pc, next_rs
            if done:
                break

        # ----- append the CLOSE-at-grasp transition (the +1) -----
        reason = "BENCH_DONE"
        if not done:
            step = len(transitions)
            goal9 = _ee_to_grasp_9d(obs, grasp_pose)
            action7 = np.concatenate([np.zeros(6, dtype=np.float32), [0.0]]).astype(np.float32)
            target_jp = action_to_target_joint(action7, obs)
            for _ in range(self.steps_action_repeat):
                obs, _, done, info = self.env.step(target_jp)
                if done:
                    break
            if self.reward_mode == "stable_grasp":
                held, obs = grasp_held_after_hold(
                    self.env, obs, self.steps_action_repeat, self.hold_steps)
            else:
                held = _grasp_reached(obs, grasp_pose,
                                      self.close_pos_thresh, self.close_rot_thresh)
            next_pc = _point_cloud(obs, self.point_listener, self.panda_base_inv_tf)
            next_rs = _robot_state(obs, np.zeros(6, dtype=np.float32))
            transitions.append(dict(
                pc=pc, rs=rs, remain_norm=max(max_steps - step, 0) / max_steps,
                action=np.concatenate(
                    [self._norm(np.zeros(6, dtype=np.float32)), [-self.act_limit]]).astype(np.float32),
                reward=(1.0 if held else 0.0), next_pc=next_pc, next_rs=next_rs,
                next_remain_norm=max(max_steps - step - 1, 0) / max_steps, terminal=1.0,
                mc_return=0.0, expert_action=self._norm(np.zeros(6, dtype=np.float32)).astype(np.float32),
                expert_flag=1.0, goal_pose=goal9, next_goal_pose=goal9,  # terminal: Φ(s') zeroed
                expert_gripper=0.0, gripper_flag=1.0, perturb_flag=0.0))
            reason = "GRASP_OK" if held else "GRASP_MISS"

        if not transitions:
            return [], {"skipped": True, "scene_idx": scene_idx}

        G = 0.0
        for tr in reversed(transitions):
            G = tr["reward"] + self.gamma * (1.0 - tr["terminal"]) * G
            tr["mc_return"] = G

        success = int(transitions[-1]["reward"] >= 1.0)
        stats = {
            "skipped": False,
            "scene_idx": scene_idx,
            "length": len(transitions),
            "success": success,
            "return": float(sum(tr["reward"] for tr in transitions)),
            "reason": reason,
            "n_omg_fail": n_omg_fail,
            "n_perturb": n_perturb,
            "min_pos": float(min_pos) if min_pos < float("inf") else float("nan"),
            "min_rot": float(min_rot) if min_rot < float("inf") else float("nan"),
            "closed": 1,
            "expert_episode": True,
        }
        return transitions, stats
