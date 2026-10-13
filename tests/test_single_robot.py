"""Single-robot problem: configurable room, observation without teammates,
no recovery fallback and the simple coverage reward."""

import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.map_layouts import ProceduralMapLayout
from src.envs.vec_env import VecEnv
from src.train_marl import build_env_config, observation_config
from src.utils.config_parser import load_config

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'single_robot.yaml')


def single_robot_env_config(num_maps=2):
    env_cfg, _ = build_env_config(load_config(CONFIG), num_maps=num_maps)
    return env_cfg


class SingleRobotTests(unittest.TestCase):
    def test_room_size_and_depth_are_configurable(self):
        env = VecEnv(2, single_robot_env_config()).env
        self.assertEqual((env.grid_h, env.grid_w), (12, 16))
        self.assertEqual(env.num_robots, 1)
        config = observation_config(env)
        self.assertEqual((config['map_width'], config['map_height'], config['map_depth']),
                         (8.0, 6.0, 2))

        empty = ProceduralMapLayout(width=4.0, height=3.0, max_depth=0)
        real = [w for w in empty.get_walls() if w[0] > -5.0]
        self.assertEqual(len(real), 4)   # outer boundary only

    def test_observation_has_no_communication_slots(self):
        cfg = single_robot_env_config()
        env = VecEnv(2, cfg).env
        three = VecEnv(2, {**cfg, 'num_robots': 3, 'comm_slots': 2}).env
        self.assertEqual(env.comm_slots, 0)
        self.assertEqual(three.obs_vec_dim - env.obs_vec_dim, 3 * 2)
        _, obs, _, _ = VecEnv(2, cfg).reset(jax.random.PRNGKey(0))
        self.assertEqual(obs.shape, (2, 1, env.obs_dim))

    def test_no_recovery_fallback(self):
        env = VecEnv(1, single_robot_env_config()).env
        self.assertFalse(env.fallback_enabled)

    def test_simple_reward_terms(self):
        """r = alpha * new cells - revisit_cost * re-entries of visited cells - tau
        - wall_kappa * contact + completion_bonus * complete."""
        cfg = single_robot_env_config()
        env = VecEnv(1, cfg).env
        self.assertEqual(env.reward_mode, 'simple')
        step = jax.jit(env.step)
        state = env.reset(jax.random.PRNGKey(0))

        def cell(s):
            return tuple((np.asarray(s.robot_positions[0]) // env.cell_size).astype(int))
        seen_new = seen_revisit = False
        # Forward, turn on the spot by ~pi, forward again: back over covered cells.
        forward, turn = jnp.array([[1., 0.]]), jnp.array([[-1., 1.]])
        for t in range(600):
            action = turn if 25 <= t % 80 < 56 else forward   # 31 steps at 1 rad/s ~ pi
            grid_before, cell_before = np.asarray(state.coverage_grid), cell(state)
            visits_before = np.asarray(state.visit_counts[0])
            state, reward, _, _ = step(state, action)
            new = float(np.asarray(state.coverage_grid).sum() - grid_before.sum())
            col, row = cell(state)
            revisit = float((col, row) != cell_before and visits_before[row, col] > 0)
            wall = float(state.wall_hits[0])
            expected = (cfg['alpha'] * new - cfg['revisit_cost'] * revisit
                        - cfg['tau'] - cfg['wall_kappa'] * wall)
            self.assertAlmostEqual(float(reward[0]), expected, places=4)
            seen_new |= new > 0
            seen_revisit |= revisit > 0
        self.assertTrue(seen_new and seen_revisit)

if __name__ == '__main__':
    unittest.main()
