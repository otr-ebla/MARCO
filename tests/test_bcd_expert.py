import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.bcd_expert import (BCDExpert, expert_steps, first_episode_summary, geodesic_table,
                                 open_edges, split_tour, sweep)
from src.envs.coverage_vector_env import MultiRobotCoverageEnv
from src.envs.vec_env import VecEnv
from src.pretrain_bc import with_policy_std
from src.train_marl import bc_coefficient

CONFIG = dict(num_maps=4, num_robots=3, n_rays=70, obs_mode='memory_comm', use_full_memory=True,
              reward_mode='sequential', max_steps=4000, fallback_enabled=False)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MultiRobotCoverageEnv(CONFIG)
        cls.expert = BCDExpert(cls.env)

    def test_tour_visits_every_free_cell_once(self):
        for m in range(self.env.num_maps):
            tour = self.expert.tour_np[m]
            tour = tour[tour >= 0]
            free = np.flatnonzero(self.env.free_mask_np[m].ravel() > .5)
            np.testing.assert_array_equal(np.sort(tour), free)

    def test_chunks_partition_the_tour_with_balanced_cost(self):
        for m in range(self.env.num_maps):
            bounds = self.expert.bounds_np[m]
            tour = self.expert.tour_np[m][:bounds[-1]]
            self.assertEqual(bounds[0], 0)
            self.assertTrue(np.all(np.diff(bounds) > 0))
            dist = np.asarray(self.expert.dist[m])
            step = np.concatenate([[1], dist[tour[:-1], tour[1:]]])
            costs = np.array([step[a:b].sum() for a, b in zip(bounds[:-1], bounds[1:])])
            # Cuts fall on tour steps: every share is within one step of equal.
            self.assertLessEqual(np.abs(costs - step.sum() / len(costs)).max(), step.max())

    def test_lanes_follow_the_longer_side_and_alternate(self):
        self.assertEqual(sweep((0, 0, 1, 2), 0), [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)])
        self.assertEqual(sweep((0, 0, 2, 1), 3)[:4], [(2, 1), (1, 1), (0, 1), (0, 0)])

    def test_thin_wall_blocks_edge_and_geodesic_detours(self):
        free = np.ones((3, 3), bool)
        # A thin wall between columns 0 and 1 on rows 0-1.
        walls = np.array([[0.48, 0.0, 0.52, 1.0]], np.float32)
        edges = open_edges(free, walls, 0.5)
        self.assertFalse(edges[0, 0, 3])
        self.assertTrue(edges[2, 0, 3])
        self.assertEqual(geodesic_table(edges)[0, 1], 5)

    def test_equal_cost_split(self):
        dist = np.abs(np.arange(6)[:, None] - np.arange(6)[None])
        np.testing.assert_array_equal(split_tour(np.arange(6), dist, 3), [0, 2, 4, 6])


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vec = VecEnv(4, CONFIG)
        cls.expert = BCDExpert(cls.vec.env)

    def test_expert_alone_covers_the_maps_without_wall_contact(self):
        state, _, _, _ = self.vec.reset(jax.random.PRNGKey(3), jnp.arange(4))
        run = expert_steps(self.expert, self.vec, 1000)
        carry = (state, self.expert.init_chunks((4,)))
        stats = []
        for _ in range(4):
            carry, out = run(carry)
            stats.append(jax.device_get(out))
        s = first_episode_summary({k: np.concatenate([o[k] for o in stats]) for k in stats[0]}, 4000)
        self.assertGreater(s['coverage'].mean(), .98)
        self.assertEqual(s['wall'].sum(), 0.)

    def test_chunks_are_assigned_at_episode_start_only(self):
        env = self.vec.env
        state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
        _, chunk, _ = self.expert.act(state, jnp.zeros(3, jnp.int32))
        self.assertEqual(sorted(np.asarray(chunk).tolist()), [0, 1, 2])
        later = state.replace(step_count=jnp.int32(5))
        _, kept, _ = self.expert.act(later, jnp.array([2, 0, 1], jnp.int32))
        np.testing.assert_array_equal(kept, [2, 0, 1])

    def test_nominal_command_is_continuous_in_the_observation(self):
        env = self.vec.env
        state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
        chunk = self.expert.assign(state)
        target, active = self.expert.targets(state, chunk)
        point = self.expert.waypoints(state, target)
        v0, w0 = self.expert.control(state, point, active)
        nudged = state.replace(robot_headings=state.robot_headings + 1e-3,
                               robot_positions=state.robot_positions + 1e-3)
        v1, w1 = self.expert.control(nudged, self.expert.waypoints(nudged, target), active)
        self.assertLess(float(jnp.max(jnp.abs(v1 - v0))), .02)
        self.assertLess(float(jnp.max(jnp.abs(w1 - w0))), .02)


class ReplayTests(unittest.TestCase):
    def test_bit_packed_observations_round_trip(self):
        from src.pretrain_bc import ReplayBuffer
        key = jax.random.PRNGKey(0)
        head = jax.random.normal(key, (4, 2, 3, 5))
        tail = (jax.random.uniform(key, (4, 2, 3, 13)) > .5).astype(jnp.float32)
        rollout = {'obs': jnp.concatenate([head, tail], -1), 'label': jnp.ones((4, 2, 3, 2)),
                   'weight': jnp.ones((4, 2, 3)), 'change': jnp.zeros((4, 2, 3))}
        buffer = ReplayBuffer(2, rollout, split=5, binary_tail=True)
        buffer.add(rollout)
        buffer.add({**rollout, 'obs': rollout['obs'] * 0.})
        idx = jnp.arange(24)
        np.testing.assert_array_equal(buffer.observations(buffer.data, idx),
                                      rollout['obs'].reshape(24, -1))
        np.testing.assert_array_equal(buffer.observations(buffer.data, idx + 24), 0.)


class ScheduleTests(unittest.TestCase):
    def test_bc_coefficient_decays_linearly_to_zero(self):
        self.assertAlmostEqual(bc_coefficient(1, .5, 100), .5)
        self.assertAlmostEqual(bc_coefficient(51, .5, 100), .25)
        self.assertEqual(bc_coefficient(200, .5, 100), 0.)
        self.assertEqual(bc_coefficient(1, 0., 100), 0.)

    def test_policy_std_is_set_within_bounds(self):
        params = {'params': {'log_std_raw': jnp.zeros(2), 'other': jnp.ones(3)}}
        out = with_policy_std(params, .3, -2.3, 1.)
        log_std = -2.3 + 3.3 * jax.nn.sigmoid(out['params']['log_std_raw'])
        np.testing.assert_allclose(jnp.exp(log_std), .3, rtol=1e-5)
        np.testing.assert_array_equal(out['params']['other'], 1.)


if __name__ == '__main__':
    unittest.main()
