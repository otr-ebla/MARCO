import unittest
import random

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.coverage_vector_env import E2E_REWARD_DEFAULTS, MultiRobotCoverageEnv


class LocalRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(0)
        cls.env = MultiRobotCoverageEnv({
            **E2E_REWARD_DEFAULTS, 'num_maps': 1, 'num_robots': 1,
            'actor_bosco_guidance': False, 'bosco_reward_guidance': False,
        })
        cls.initial = cls.env.reset(jax.random.PRNGKey(7))
        cls.step = staticmethod(jax.jit(cls.env.step))

    def state(self, x, covered_destination=False):
        env = self.env
        grid = jnp.zeros_like(self.initial.coverage_grid).at[4, 4].set(1.)
        if covered_destination:
            grid = grid.at[4, 5].set(1.)
        return self.initial.replace(
            robot_positions=jnp.array([[x, 2.25]], dtype=jnp.float32),
            robot_headings=jnp.zeros((1,)), robot_velocities=jnp.zeros((1, 2)),
            coverage_grid=grid,
        )

    def reward(self, state, action=(1., 0.)):
        return float(self.step(state, jnp.array([action]))[1][0])

    def test_crossing_within_cell_costs_only_time(self):
        self.assertAlmostEqual(self.reward(self.state(2.1)), -.02, places=5)

    def test_revisit_is_charged_on_cell_entry(self):
        self.assertAlmostEqual(self.reward(self.state(2.45, True)), -2.02, places=5)

    def test_late_discovery_credit_grows_with_coverage(self):
        state = self.state(2.45)
        early = self.reward(state)
        remote_coverage = state.coverage_grid.at[8:, :].set(1.)
        late = self.reward(state.replace(coverage_grid=remote_coverage))
        self.assertGreater(late, early)

    def test_waiting_and_turning_cannot_earn_reward(self):
        state = self.state(2.1)
        self.assertAlmostEqual(self.reward(state, (-1., 0.)), -.02, places=5)
        self.assertAlmostEqual(self.reward(state, (-1., 1.)), -.02, places=5)

    def test_wall_contact_is_costly_and_does_not_count_as_revisit(self):
        state = self.state(.21).replace(robot_headings=jnp.array([jnp.pi]))
        self.assertAlmostEqual(self.reward(state), -2.02, places=5)

    def test_planner_does_not_change_obs_or_reward(self):
        state = self.state(2.45)
        changed = state.replace(bosco_targets=state.bosco_targets + 3.,
                                cell_assignments=jnp.ones_like(state.cell_assignments))
        np.testing.assert_array_equal(self.env.get_obs(state), self.env.get_obs(changed))
        self.assertEqual(self.reward(state), self.reward(changed))

    def test_guided_mode_cannot_use_local_reward(self):
        with self.assertRaises(ValueError):
            MultiRobotCoverageEnv(E2E_REWARD_DEFAULTS)
