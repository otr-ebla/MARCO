import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from src.envs.coverage_vector_env import MultiRobotCoverageEnv


class AxisMotionRewardTests(unittest.TestCase):
    def test_cardinal_directions_diagonals_and_no_motion(self):
        env = MultiRobotCoverageEnv.__new__(MultiRobotCoverageEnv)
        env.axis_alignment_bonus = 0.2
        env.v_max, env.dt, env._full_progress = 1.0, 0.1, 0.2
        delta = jnp.array([[.1, 0], [-.1, 0], [0, .1], [0, -.1],
                           [.07, .07], [0, 0], [.05, 0], [.1, 0], [.1, 0]])
        progress = jnp.array([.2, .2, .2, .2, .2, .2, .1, 0, -.2])
        reward = jax.jit(env._axis_motion_reward)(delta, progress)
        np.testing.assert_allclose(reward, [.2, .2, .2, .2, 0, 0, .1, 0, 0], atol=1e-6)
        env.axis_alignment_bonus = 0.0
        np.testing.assert_array_equal(env._axis_motion_reward(delta, progress), 0)

    def test_cost_is_squared_error_to_nearest_cardinal(self):
        env = MultiRobotCoverageEnv.__new__(MultiRobotCoverageEnv)
        env.axis_alignment_cost = 0.2
        env.v_max, env.dt = 1.0, 0.1
        angles = np.deg2rad([0, 30, 45, 60, 90, 120, 180, 210, 270, 330])
        delta = .1 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        cost = jax.jit(env._axis_motion_cost)(delta)
        error = np.array([0, 30, 45, 30, 0, 30, 0, 30, 0, 30])
        np.testing.assert_allclose(cost, .2 * (error / 45) ** 2, atol=1e-5)
        # Half speed pays half; rotating in place or blocked pays nothing.
        slow = env._axis_motion_cost(.05 * jnp.array([[np.cos(np.pi / 6), np.sin(np.pi / 6)]]))
        np.testing.assert_allclose(slow, [.2 * (30 / 45) ** 2 / 2], atol=1e-5)
        np.testing.assert_array_equal(env._axis_motion_cost(jnp.zeros((1, 2))), 0)

    def test_fallback_color_tracks_current_control(self):
        from src.visualize_policy import _robot_color, COLORS, _ROBOT_COLORS
        snap = {'alive': [True], 'fallback_active': [False]}
        self.assertEqual(_robot_color(snap, 0), _ROBOT_COLORS[0])
        snap['fallback_active'][0] = True
        self.assertEqual(_robot_color(snap, 0), COLORS['fallback'])
        snap['fallback_active'][0] = False
        self.assertEqual(_robot_color(snap, 0), _ROBOT_COLORS[0])
        snap['alive'][0] = False
        self.assertEqual(_robot_color(snap, 0), COLORS['dead'])
