"""ANYmal locomotion with measured contacts and phase-conditioned footfall rewards.

The policy learns joint targets; a gait clock specifies support/swing timing.
Observations have 58 entries. Old 50-entry policies require fresh training.
"""
import os
from typing import Optional, Dict, Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class AnymalWalkingEnvImproved(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    ENV_VERSION = "contact_gait_v2"
    FOOT_ORDER = ("LF", "RF", "LH", "RH")

    def __init__(self, model_path: str, render_mode: Optional[str] = None,
                 *, gait: str = "trot"):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if gait not in ("trot", "walk"):
            raise ValueError("gait must be 'trot' or 'walk'")
        if render_mode not in (None, "human"):
            raise ValueError("Supported render modes: None, 'human'")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = 0.005
        self.model.opt.timestep = self.dt
        self.frame_skip = 4
        self.control_dt = self.dt * self.frame_skip
        self.render_mode, self.viewer = render_mode, None
        self.base_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, "base")
        self.joint_names = [f"{leg}_{joint}" for leg in self.FOOT_ORDER
                            for joint in ("HAA", "HFE", "KFE")]
        self.joint_ids = np.array([
            self._require_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names])
        self.actuator_ids = np.array([
            self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.joint_names])
        self.qpos_idxs = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_idxs = self.model.jnt_dofadr[self.joint_ids]
        self.dof_pos_limits = dict(lower=self.model.jnt_range[self.joint_ids, 0].copy(),
                                   upper=self.model.jnt_range[self.joint_ids, 1].copy())
        self.default_dof_pos = np.array([
            0.0, 0.4, -0.8, 0.0, 0.4, -0.8,
            0.0, -0.4, 0.8, 0.0, -0.4, 0.8,
        ], dtype=np.float32)
        # Allow hip/knee flexion for foot lift; keep abduction range smaller.
        self.action_scale = np.tile([0.2, 0.5, 0.7], 4).astype(np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        # velocities(6), gravity(3), command(3), joints(24), last action(12),
        # clock(2), measured contacts(4), sole clearance(4).
        self.observation_space = spaces.Box(-np.inf, np.inf, (58,), dtype=np.float32)
        self.command_lin_vel_x, self.command_lin_vel_y, self.command_ang_vel_z = 0.5, 0.0, 0.0
        # Translation is a fixed WORLD-frame command. Expose it in the base
        # frame so the policy observes heading error instead of learning circles.
        self.gait = gait
        self.gait_freq = 1.5 if gait == "trot" else 1.0
        self.stance_ratio = 0.6 if gait == "trot" else 0.75
        # Trot: LF+RH alternate with RF+LH. Walk: one foot swings at a time.
        self.phase_offsets = np.array([0.0, 0.5, 0.5, 0.0] if gait == "trot"
                                      else [0.0, 0.5, 0.75, 0.25])
        self.gait_phase, self.swing_height = 0.0, 0.07
        self.base_height_target = 0.48  # settled nominal base height

        # ANYmal C has no *_FOOT bodies: use the collidable sphere on *_SHANK.
        self.feet_geom_ids = self._find_feet()
        self.feet_body_ids = self.model.geom_bodyid[self.feet_geom_ids]
        self._foot_index = {int(g): i for i, g in enumerate(self.feet_geom_ids)}
        self.floor_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        mujoco.mj_forward(self.model, self.data)
        if (self.model.geom_type[self.floor_geom_id] != mujoco.mjtGeom.mjGEOM_PLANE
                or not np.allclose(self.data.geom_xmat[self.floor_geom_id].reshape(3, 3)[:, 2],
                                   [0, 0, 1])):
            raise ValueError("This environment requires a horizontal plane named 'floor'")
        self.floor_height = float(self.data.geom_xpos[self.floor_geom_id, 2])
        # Signed costs: no constant bonuses for zero actions or unused joints.
        self.reward_weights = {
            "tracking_lin_vel": 3.0, "tracking_ang_vel": 0.5,
            "gait_contact": 1.5, "swing_clearance": -0.5,
            "vertical_velocity": -2.0, "roll_pitch_rate": -0.05,
            "orientation": -1.0, "base_height": -5.0,
            "foot_slip": -0.25, "flight": -1.0,
            "torques": -2e-5, "joint_velocity": -5e-4,
            "action_rate": -0.01, "joint_limits": -1.0,
            "nonfoot_contact": -0.25,
            "heading": -1.0,
        }
        self.episode_length, self.max_episode_length = 0, 1000
        self.last_reward_terms, self._last_gait_metrics = {}, {}
        self._episode_start_xy = np.zeros(2)

    def _require_id(self, kind, name):
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"Required {kind.name} '{name}' is missing from model")
        return index

    def _find_feet(self):
        feet = []
        for leg in self.FOOT_ORDER:
            body = self._require_id(mujoco.mjtObj.mjOBJ_BODY, f"{leg}_SHANK")
            candidates = np.flatnonzero(
                (self.model.geom_bodyid == body)
                & (self.model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
                & ((self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)))
            if len(candidates) != 1:
                raise ValueError(f"Expected one collidable foot sphere on {leg}_SHANK")
            feet.append(candidates[0])
        return np.array(feet, dtype=int)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_idxs] = self.default_dof_pos
        self.data.qpos[:3] = [0.0, 0.0, self.floor_height + 0.65]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.ctrl[self.actuator_ids] = self.default_dof_pos
        # Bounded settle; never recurse with the same failed state and seed.
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self._check_termination():
            raise RuntimeError("ANYmal could not settle upright; check model and nominal pose")
        self.data.qvel[:] *= 0.1
        self.data.qvel[:6] += self.np_random.uniform(-0.03, 0.03, 6)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_action.fill(0.0)
        self.gait_phase = float(self.np_random.uniform(0.0, 1.0))
        self.episode_length = 0
        self._episode_start_xy = self.data.qpos[:2].copy()
        self.last_reward_terms, self._last_gait_metrics = {}, {}
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain 12 finite joint offsets")
        action = np.clip(action, -1.0, 1.0)
        # Action-rate cost supplies smoothness without hidden filter state.
        target = self.default_dof_pos + self.action_scale * action
        self.data.ctrl[self.actuator_ids] = np.clip(
            target, self.dof_pos_limits["lower"] + 0.05, self.dof_pos_limits["upper"] - 0.05)
        try:
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
            # Refresh derived state after mj_step's final integration.
            mujoco.mj_forward(self.model, self.data)
        except mujoco.FatalError:
            return np.zeros(58, dtype=np.float32), -1.0, True, False, {"error": "physics_crash"}
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            return np.zeros(58, dtype=np.float32), -1.0, True, False, {"error": "nonfinite_state"}
        self.episode_length += 1
        self.gait_phase = (self.gait_phase + self.control_dt * self.gait_freq) % 1.0
        terminated = self._check_termination()
        truncated = self.episode_length >= self.max_episode_length
        reward = self._compute_reward(action, terminated)
        self.last_action = action.copy()
        # Next observation contains the action just applied and the new phase.
        obs = self._get_obs()
        vel, contacts = self.data.qvel[:3], self._get_foot_contacts()
        info = {
            "x_pos": float(self.data.qpos[0] - self._episode_start_xy[0]),
            "x_vel": float(vel[0]), "y_vel": float(vel[1]), "z_vel": float(self.data.qvel[2]),
            "episode_length": self.episode_length, "control_dt": self.control_dt,
            "base_height": float(self.data.xpos[self.base_body_id, 2] - self.floor_height),
            "foot_contacts": contacts.astype(np.float32),
            "foot_clearance": self._get_foot_clearance().astype(np.float32),
            "flight": float(not contacts.any()),
            "diagonal_support": float(np.array_equal(contacts, [1, 0, 0, 1])
                                      or np.array_equal(contacts, [0, 1, 1, 0])),
            "reward_terms": self.last_reward_terms.copy(), **self._last_gait_metrics,
        }
        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, info

    def _gait_targets(self):
        phase = (self.gait_phase - self.phase_offsets) % 1.0
        moving = np.linalg.norm([self.command_lin_vel_x, self.command_lin_vel_y,
                                 self.command_ang_vel_z]) > 0.05
        stance = (phase < self.stance_ratio) if moving else np.ones(4, dtype=bool)
        swing = np.clip((phase - self.stance_ratio) / (1 - self.stance_ratio), 0, 1)
        clearance = self.swing_height * np.sin(np.pi * swing)
        clearance[stance] = 0.0
        return stance, clearance

    def _gait_terms(self, contacts, clearance):
        stance, target_height = self._gait_targets()
        match = float(np.mean(contacts == stance))
        # Standing and synchronous hopping score below correct alternating feet.
        contact_score = 2.0 * match - 1.0
        # Keep a cost for excessive lift. An exponential bonus saturates near
        # zero, making a 30 cm leg kick almost indistinguishable from a small
        # clearance error. Normalize by the desired peak swing height.
        height_cost = float(np.mean(np.square((clearance - target_height) / self.swing_height)))
        return contact_score, height_cost, match

    def _compute_reward(self, action, terminated):
        vel, ang = self.data.qvel[:3], self._get_base_ang_vel()
        contacts = self._get_foot_contacts()
        gait, clearance, match = self._gait_terms(contacts, self._get_foot_clearance())
        foot_vel = self._get_foot_velocities()
        slip = float(np.sum(np.square(foot_vel[contacts, :2])))
        lower, upper = self.dof_pos_limits["lower"], self.dof_pos_limits["upper"]
        mid, half = (lower + upper) / 2, (upper - lower) * 0.45
        limits = np.maximum(np.abs(self._get_dof_pos() - mid) - half, 0.0)
        terms = {
            "tracking_lin_vel": np.exp(-np.sum(np.square(
                vel[:2] - [self.command_lin_vel_x, self.command_lin_vel_y])) / 0.15),
            "tracking_ang_vel": np.exp(-(ang[2] - self.command_ang_vel_z) ** 2 / 0.25),
            "gait_contact": gait, "swing_clearance": clearance,
            "vertical_velocity": self.data.qvel[2] ** 2,  # world vertical, independent of tilt
            "roll_pitch_rate": np.sum(np.square(ang[:2])),
            "orientation": np.sum(np.square(self._get_projected_gravity()[:2])),
            "base_height": (self.data.xpos[self.base_body_id, 2]
                            - self.floor_height - self.base_height_target) ** 2,
            "foot_slip": slip, "flight": float(not contacts.any()),
            "torques": np.sum(np.square(self.data.qfrc_actuator[self.qvel_idxs])),
            "joint_velocity": np.sum(np.square(self._get_dof_vel())),
            "action_rate": np.sum(np.square(action - self.last_action)),
            "joint_limits": np.sum(limits),
            "nonfoot_contact": len(self._nonfoot_ground_bodies()),
            "heading": self._heading_cost(),
        }
        # Integrate reward rates over dt. A time limit is not a fall.
        self.last_reward_terms = {
            k: float(self.reward_weights[k] * v * self.control_dt) for k, v in terms.items()}
        self.last_reward_terms["termination"] = -1.0 if terminated else 0.0
        self._last_gait_metrics = {"gait_match": match, "foot_slip": slip}
        return float(sum(self.last_reward_terms.values()))

    def _check_termination(self):
        if self.data.xpos[self.base_body_id, 2] - self.floor_height < 0.30:
            return True
        if self._get_projected_gravity()[2] > -0.6:
            return True
        # A torso impact is a fall. Distal shank shapes can touch the floor
        # during normal foot roll, so penalize those contacts without ending
        # the episode (otherwise exploration terminates after only 2-4 steps).
        return self.base_body_id in self._nonfoot_ground_bodies()

    def _nonfoot_ground_bodies(self):
        bodies = set()
        for contact in self.data.contact:
            other = self._floor_contact_other(contact)
            if other is not None and other not in self._foot_index and contact.dist <= 0:
                bodies.add(int(self.model.geom_bodyid[other]))
        return bodies

    def _floor_contact_other(self, contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        return g2 if g1 == self.floor_geom_id else g1 if g2 == self.floor_geom_id else None

    def _get_obs(self):
        command = self._command_in_base_frame()
        return np.concatenate([
            self._get_base_lin_vel() * 2.0, self._get_base_ang_vel() * 0.25,
            self._get_projected_gravity(),
            [command[0] * 2.0, command[1] * 2.0, self.command_ang_vel_z * 0.25],
            self._get_dof_pos() - self.default_dof_pos, self._get_dof_vel() * 0.05,
            self.last_action,
            [np.sin(2 * np.pi * self.gait_phase), np.cos(2 * np.pi * self.gait_phase)],
            self._get_foot_contacts(), self._get_foot_clearance(),
        ]).astype(np.float32)

    def _get_dof_pos(self):
        return self.data.qpos[self.qpos_idxs].copy()

    def _get_dof_vel(self):
        return self.data.qvel[self.qvel_idxs].copy()

    def _get_base_lin_vel(self):
        # Free-joint translation is at the base origin, in world coordinates.
        return self.data.xmat[self.base_body_id].reshape(3, 3).T @ self.data.qvel[:3]

    def _command_in_base_frame(self):
        return self.data.xmat[self.base_body_id].reshape(3, 3).T @ np.array(
            [self.command_lin_vel_x, self.command_lin_vel_y, 0.0])

    def _heading_cost(self):
        direction = np.array([self.command_lin_vel_x, self.command_lin_vel_y])
        speed = np.linalg.norm(direction)
        if speed < 0.05 or abs(self.command_ang_vel_z) > 0.05:
            return 0.0
        forward = self.data.xmat[self.base_body_id].reshape(3, 3)[:2, 0]
        forward = forward / max(np.linalg.norm(forward), 1e-8)
        return 1.0 - float(np.clip(forward @ (direction / speed), -1.0, 1.0))

    def _get_base_ang_vel(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.base_body_id, velocity, 1)
        return velocity[:3]

    def _get_projected_gravity(self):
        return self.data.xmat[self.base_body_id].reshape(3, 3).T @ np.array([0., 0., -1.])

    def _get_foot_contacts(self):
        contacts = np.zeros(4, dtype=bool)
        for contact in self.data.contact:
            index = self._foot_index.get(self._floor_contact_other(contact))
            if index is not None and contact.dist <= 0:
                contacts[index] = True
        return contacts

    def _get_foot_clearance(self):
        # Sphere center minus radius gives sole height, not shank COM height.
        return (self.data.geom_xpos[self.feet_geom_ids, 2]
                - self.model.geom_size[self.feet_geom_ids, 0] - self.floor_height)

    def _get_foot_velocities(self):
        velocities = np.zeros((4, 3))
        for i, geom in enumerate(self.feet_geom_ids):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM,
                                    int(geom), velocity, 0)
            velocities[i] = velocity[3:]
        return velocities

    def _render_frame(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anybotics_anymal_c/scene.xml")
    parser.add_argument("--gait", choices=["trot", "walk"], default="trot")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    env = AnymalWalkingEnvImproved(args.model, "human" if args.render else None, gait=args.gait)
    try:
        obs, _ = env.reset(seed=0)
        print(f"Observation: {obs.shape}; foot geoms: {env.feet_geom_ids}")
        for step in range(100):
            obs, reward, terminated, truncated, info = env.step(np.zeros(12))
            if step % 20 == 0:
                print(f"Step {step}: reward={reward:.3f}, contacts={info['foot_contacts']}")
            if terminated or truncated:
                break
    finally:
        env.close()
