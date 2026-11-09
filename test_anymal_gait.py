"""Regression tests for the contact/gait failure, using the real ANYmal model.

Run: python -m unittest test_anymal_gait -v
"""
import unittest
from pathlib import Path

import mujoco
import numpy as np

from anymal_env_improved import AnymalWalkingEnvImproved

MODEL = str(Path(__file__).parent / "anybotics_anymal_c/scene.xml")


class GaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = AnymalWalkingEnvImproved(MODEL)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=7)

    def test_real_ground_contacts_and_airborne_feet(self):
        env = self.env
        self.assertTrue(np.all(env.feet_geom_ids >= 0))
        self.assertEqual(len(set(env.feet_geom_ids)), 4)
        np.testing.assert_array_equal(env._get_foot_contacts(), [True] * 4)
        self.assertLess(np.max(np.abs(env._get_foot_clearance())), 0.02)
        env.data.qpos[2] += 0.2
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env._get_foot_contacts().any())

    def test_foot_velocity_matches_point_jacobian(self):
        env = self.env
        env.data.qvel[:] = np.linspace(-0.8, 0.8, env.model.nv)
        mujoco.mj_forward(env.model, env.data)
        measured = env._get_foot_velocities()
        for i, geom in enumerate(env.feet_geom_ids):
            jac = np.zeros((3, env.model.nv))
            mujoco.mj_jacGeom(env.model, env.data, jac, None, int(geom))
            np.testing.assert_allclose(measured[i], jac @ env.data.qvel, atol=1e-10)

    def test_next_observation_has_current_action_and_phase(self):
        env = self.env
        phase = env.gait_phase
        action = np.linspace(-0.05, 0.05, 12, dtype=np.float32)
        obs, reward, terminated, _, info = env.step(action)
        self.assertEqual(obs.shape, (58,))
        self.assertFalse(terminated)
        np.testing.assert_array_equal(obs[36:48], action)
        expected = (phase + env.control_dt * env.gait_freq) % 1
        np.testing.assert_allclose(obs[48:50], [np.sin(2*np.pi*expected),
                                               np.cos(2*np.pi*expected)], atol=1e-7)
        np.testing.assert_array_equal(obs[50:54], info["foot_contacts"])
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))

    def test_trot_pairs_alternate_with_overlap_and_no_flight(self):
        env = self.env
        env.gait_phase = 0.25
        np.testing.assert_array_equal(env._gait_targets()[0], [1, 0, 0, 1])
        env.gait_phase = 0.75
        np.testing.assert_array_equal(env._gait_targets()[0], [0, 1, 1, 0])
        env.gait_phase = 0.05
        np.testing.assert_array_equal(env._gait_targets()[0], [1, 1, 1, 1])

    def test_alternation_beats_standing_and_hopping_over_full_cycle(self):
        env = self.env
        scores = np.zeros(3)
        for phase in np.linspace(0, 1, 200, endpoint=False):
            env.gait_phase = phase
            stance, height = env._gait_targets()
            candidates = [(stance, height), (np.ones(4, dtype=bool), np.zeros(4)),
                          (np.full(4, phase < 0.6), np.full(4, 0 if phase < 0.6 else 0.07))]
            for i, (contact, clearance) in enumerate(candidates):
                c, h, _ = env._gait_terms(contact, clearance)
                scores[i] += env.reward_weights["gait_contact"] * c + env.reward_weights["swing_clearance"] * h
        self.assertGreater(scores[0], scores[1] + 100)
        self.assertGreater(scores[0], scores[2] + 100)

    def test_walk_has_three_stance_feet_and_balanced_duty(self):
        env = AnymalWalkingEnvImproved(MODEL, gait="walk")
        try:
            supports = []
            for phase in np.linspace(0, 1, 100, endpoint=False) + 0.001:
                env.gait_phase = phase
                stance, _ = env._gait_targets()
                self.assertEqual(stance.sum(), 3)
                supports.append(stance)
            np.testing.assert_allclose(np.mean(supports, axis=0), 0.75)
        finally:
            env.close()

    def test_excessive_foot_lift_keeps_getting_more_expensive(self):
        env = self.env
        env.gait_phase = 0.25
        stance, target = env._gait_targets()
        _, exact, _ = env._gait_terms(stance, target)
        _, small_error, _ = env._gait_terms(stance, target + 0.03)
        _, high_kick, _ = env._gait_terms(stance, target + 0.25)
        self.assertEqual(exact, 0)
        self.assertGreater(high_kick, small_error * 10)
        self.assertLess(env.reward_weights["swing_clearance"], 0)

    def test_stop_command_requests_all_feet_down(self):
        env = self.env
        env.command_lin_vel_x = 0.0
        try:
            env.gait_phase = 0.25
            stance, height = env._gait_targets()
            self.assertTrue(stance.all())
            np.testing.assert_array_equal(height, np.zeros(4))
        finally:
            env.command_lin_vel_x = 0.5

    def test_bounce_and_actual_torque_are_costs(self):
        env = self.env
        env.data.qvel[2] = 0.5
        mujoco.mj_forward(env.model, env.data)
        env._compute_reward(np.zeros(12), False)
        self.assertAlmostEqual(env.last_reward_terms["vertical_velocity"], -0.01)
        # Zero actions still require motor torque to support body weight.
        self.assertLess(env.last_reward_terms["torques"], 0)

    def test_seeded_rollout_is_repeatable(self):
        env = self.env
        action = np.full(12, 0.03, dtype=np.float32)
        first = []
        env.reset(seed=19)
        for _ in range(10):
            first.append(env.step(action)[0])
        env.reset(seed=19)
        for expected in first:
            np.testing.assert_array_equal(env.step(action)[0], expected)

    def test_timeout_is_not_a_fall(self):
        env = self.env
        env.episode_length = env.max_episode_length - 1
        _, _, terminated, truncated, info = env.step(np.zeros(12))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reward_terms"]["termination"], 0)

    def test_shank_brush_does_not_terminate_upright_robot(self):
        env = self.env
        rng = np.random.default_rng(7)
        for _ in range(3):
            _, _, terminated, _, info = env.step(rng.normal(0, 0.6, 12))
        self.assertTrue(env._nonfoot_ground_bodies())
        self.assertFalse(terminated)
        self.assertLess(info["reward_terms"]["nonfoot_contact"], 0)

    def test_gym_contract(self):
        from stable_baselines3.common.env_checker import check_env
        check_env(self.env, warn=True, skip_render_check=True)

    def test_world_forward_command_cannot_be_satisfied_by_circling(self):
        env = self.env
        # Face +Y and travel +Y at 0.5 m/s: body-forward speed is perfect,
        # but this should be a world-command tracking error, visible in obs.
        env.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        env.data.qvel[:3] = [0, 0.5, 0]
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env._get_base_lin_vel(), [0.5, 0, 0], atol=1e-7)
        np.testing.assert_allclose(env._get_obs()[9:11], [0, -1], atol=1e-7)
        env._compute_reward(np.zeros(12), False)
        self.assertLess(env.last_reward_terms["tracking_lin_vel"], 0.01)
        self.assertAlmostEqual(env._heading_cost(), 1.0)
        env.data.qvel[:3] = [0.5, 0, 0]
        mujoco.mj_forward(env.model, env.data)
        env._compute_reward(np.zeros(12), False)
        self.assertAlmostEqual(env.last_reward_terms["tracking_lin_vel"], 0.06)


if __name__ == "__main__":
    unittest.main()
