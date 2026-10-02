import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.mappo import rms_init, rms_normalize, rms_update
from src.envs.coverage_vector_env import MultiRobotCoverageEnv, _FAR
from src.envs.vec_env import VecEnv
from src.models.actor_critic import Actor, LocalCritic
from src.algorithms.mappo import Transition
from src.train_marl import _episode_stats_init, _episode_stats_step


# Preserve the original geometry of the hand-authored motion fixtures.
CONFIG = dict(num_maps=1, wall_cells=0, num_robots=3, n_rays=12, obs_mode='memory_comm',
              use_full_memory=True, observation_stack=5, reward_mode='sequential',
              comm_slots=2, comm_radius=3.0, max_steps=3)


class SequentialPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MultiRobotCoverageEnv(CONFIG)
        cls.state = jax.jit(cls.env.reset)(jax.random.PRNGKey(7))

    def test_stack_order_and_reset(self):
        env, state = self.env, self.state
        np.testing.assert_array_equal(state.obs_history[:, :-1], 0)
        np.testing.assert_allclose(state.obs_history[:, -1], env._get_frame_obs(state))
        step = jax.jit(env.step)
        for _ in range(6):
            old = state
            state, _, _, _ = step(state, jnp.zeros((3, 2)))
            np.testing.assert_array_equal(state.obs_history[:, :-1], old.obs_history[:, 1:])
            np.testing.assert_allclose(state.obs_history[:, -1], env._get_frame_obs(state), atol=1e-6)
        obs = env.get_obs(state)
        heads = obs[:, :env.norm_dim].reshape(3, 5, env.frame_norm_dim)
        tails = obs[:, env.norm_dim:].reshape(3, 5, env.patch_dim)
        np.testing.assert_array_equal(jnp.concatenate([heads, tails], -1), state.obs_history)
        rms = rms_update(rms_init(env.norm_dim), obs)
        np.testing.assert_array_equal(rms_normalize(rms, obs)[:, env.norm_dim:], obs[:, env.norm_dim:])

    def test_vector_autoreset_clears_only_finished_history(self):
        vec = VecEnv(2, CONFIG)
        state, _, _, _ = vec.reset(jax.random.PRNGKey(8))
        state = state.replace(step_count=jnp.array([2, 0], jnp.int32))
        previous = state.obs_history
        state, obs, _, _, done, _, _ = vec.step(state, jnp.zeros((2, 3, 2)))
        np.testing.assert_array_equal(done, [True, False])
        np.testing.assert_array_equal(state.obs_history[0, :, :-1], 0)
        np.testing.assert_array_equal(state.obs_history[1, :, :-1], previous[1, :, 1:])
        np.testing.assert_array_equal(obs, jax.vmap(vec.env.get_obs)(state))

    def test_distant_work_visible_without_ground_truth_leak(self):
        env, state = self.env, self.state
        # Only three nearby cells and a distant cell are known. The personal map
        # must expose that distant cell even though it is outside the local crop.
        pos = jnp.array([[1.25, 1.25], [8.25, 5.25], [10.25, 5.25]])
        known = jnp.zeros_like(state.mem_known).at[:, 2, 2:5].set(1).at[:, 14, 2].set(1)
        state = state.replace(robot_positions=pos, mem_known=known,
                              mem_covered=jnp.zeros_like(known))
        first = env._get_frame_obs(state)
        changed = env._get_frame_obs(state.replace(mem_covered=known.at[:, 2, 2:5].set(0)))
        local_end = env.frame_norm_dim + 3 * env.local_coverage_size ** 2
        np.testing.assert_array_equal(first[:, :local_end], changed[:, :local_end])
        self.assertFalse(np.array_equal(first[:, local_end:], changed[:, local_end:]))
        hidden_coverage = env._get_frame_obs(state.replace(coverage_grid=jnp.ones_like(state.coverage_grid)))
        np.testing.assert_array_equal(first, hidden_coverage)
        walls = env.wall_grids
        try:
            # Alter only unknown geometry. Neither observations nor routes may change.
            field = env._known_work_field(state)
            env.wall_grids = jnp.where(known[0][None] > 0, walls, 1 - walls)
            np.testing.assert_array_equal(first, env._get_frame_obs(state))
            np.testing.assert_array_equal(field, env._known_work_field(state))
        finally:
            env.wall_grids = walls

    def test_known_routes_do_not_cross_unknown_or_walls(self):
        env, state = self.env, self.state
        walls = env.wall_grids
        try:
            env.wall_grids = jnp.zeros_like(walls).at[:, 3, 4].set(1)
            known = jnp.zeros_like(state.mem_known).at[:, 3, 2:7].set(1)
            state = state.replace(mem_known=known, mem_covered=jnp.zeros_like(known))
            field = env._known_work_field(state)
            self.assertTrue(np.all(np.asarray(field[:, 3, 4]) >= _FAR))
            self.assertTrue(np.all(np.asarray(field[:, 10, 10]) >= _FAR))
            self.assertTrue(np.all(np.asarray(field[:, 3, 3]) == 0))
        finally:
            env.wall_grids = walls

    def test_sequential_bonus_requires_fresh_adjacent_discovery(self):
        env = self.env
        state = self.state.replace(last_discovery=jnp.array([[2, 2]] * 3),
                                   sweep_direction=jnp.array([[1, 0]] * 3))
        # Straight, turn, jump over a cell.
        cells = jnp.array([[3, 2], [2, 3], [4, 2]])
        bonus = env._sequential_discovery_reward(state, cells, jnp.ones(3, bool))
        np.testing.assert_array_equal(bonus, [3, 2, 0])
        np.testing.assert_array_equal(
            env._sequential_discovery_reward(state, cells, jnp.zeros(3, bool)), 0)
        np.testing.assert_array_equal(
            env._sequential_discovery_reward(self.state, cells, jnp.ones(3, bool)), 0)

    def test_known_work_takes_priority_over_covered_frontier(self):
        env, walls = self.env, self.env.wall_grids
        try:
            env.wall_grids = jnp.zeros_like(walls)
            known = jnp.zeros_like(self.state.mem_known).at[:, 3, 2:7].set(1)
            covered = known.at[:, 3, 6].set(0)
            pos = jnp.array([[1.25, 1.75]] * 3)  # col 2, row 3
            state = self.state.replace(robot_positions=pos, mem_known=known, mem_covered=covered)
            field = env._known_work_field(state)
            np.testing.assert_array_equal(field[:, 3, 2], 4)
            np.testing.assert_array_equal(field[:, 3, 6], 0)
            # Once known work is done, the covered frontier becomes a valid target.
            field = env._known_work_field(state.replace(mem_covered=known))
            np.testing.assert_array_equal(field[:, 3, 2], 0)
        finally:
            env.wall_grids = walls

    def test_spreading_only_uses_observable_teammates(self):
        env = self.env
        state = self.state.replace(robot_positions=jnp.array([[1., 1.], [2., 1.], [9., 6.]]))
        np.testing.assert_array_equal(env._visible_teammates(state),
                                      [[False, True, False], [True, False, False], [False]*3])
        np.testing.assert_array_equal(env._visible_teammates(
            state.replace(robot_alive=jnp.array([True, False, True]))), False)

    def test_stacked_actor_and_local_critic_forward_backward(self):
        env = self.env
        obs = env.get_obs(self.state)
        kwargs = dict(vec_dim=env.obs_vec_dim, n_rays=env.n_rays, tail_dim=env.patch_dim,
                      memory_map_shape=env.memory_map_shape, observation_stack=5,
                      hidden_size=16, lidar_embed=8)
        for recurrent in (False, True):
            actor = Actor(**kwargs, recurrent=recurrent, log_std_min=-2.3)
            params = actor.init(jax.random.PRNGKey(1), obs)
            result = jax.jit(actor.apply)(params, obs)
            self.assertEqual(result[0].shape, (3, 2))
            loss, grads = jax.jit(jax.value_and_grad(
                lambda p: jnp.sum(actor.apply(p, obs)[0] ** 2)))(params)
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(grads)))
            params['params']['log_std_raw'] = jnp.full((2,), -100.)
            np.testing.assert_allclose(actor.apply(params, obs)[1], -2.3)
        critic = LocalCritic(**kwargs)
        params = critic.init(jax.random.PRNGKey(2), obs)
        self.assertEqual(jax.jit(critic.apply)(params, obs).shape, (3, 1))


class SweepRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MultiRobotCoverageEnv({
            **CONFIG, 'num_robots': 1, 'observation_stack': 1, 'dt': 0.5,
            'alpha': 0., 'progress_weight': 0., 'loiter_cost': 0., 'tau': 0.,
            'sequential_bonus': 6., 'straight_bonus': 4.,
            'boustrophedon_bonus': 8., 'revisit_cost': 2.,
        })
        cls.step = staticmethod(jax.jit(cls.env.step))

    def start(self):
        state = self.env.reset(jax.random.PRNGKey(11)).replace(
            robot_positions=jnp.array([[1.25, 1.25]]), robot_headings=jnp.zeros(1))
        state, _, _, _ = self.step(state, jnp.array([[-1., 0.]]))
        return state

    def move(self, state, heading):
        return self.step(state.replace(robot_headings=jnp.array([heading], jnp.float32)),
                         jnp.array([[1., 0.]]))[:2]

    def test_boustrophedon_rewards_real_new_cell_sequence(self):
        state = self.start()
        # East twice, shift north, reverse west, continue west.
        for heading, expected in [(0., 6.), (0., 10.), (np.pi/2, 6.),
                                  (np.pi, 14.), (np.pi, 10.)]:
            state, reward = self.move(state, heading)
            np.testing.assert_allclose(reward, expected, atol=1e-5)
        self.assertEqual(float(self.env.get_info(state)['recoverage']), 1.)

    def test_revisit_penalty_and_recoverage_ignore_dwelling(self):
        state = self.start()
        state, _ = self.move(state, 0.)
        state, reward = self.move(state, np.pi)
        np.testing.assert_allclose(reward, -2.)
        self.assertEqual(float(self.env.get_info(state)['recoverage']), 1.5)
        for _ in range(4):
            state, reward, _, _ = self.step(state, jnp.array([[-1., 1.]]))
            np.testing.assert_allclose(reward, 0.)
        self.assertEqual(float(state.cell_entries), 3.)
        self.assertEqual(float(self.env.get_info(state)['recoverage']), 1.5)
        np.testing.assert_array_equal(state.lane_return, 0)

    def test_revisit_breaks_pending_lane_reversal(self):
        state = self.start()
        for h in [0., 0., np.pi/2]:
            state, _ = self.move(state, h)
        np.testing.assert_array_equal(state.lane_return, [[-1, 0]])
        state, reward = self.move(state, -np.pi/2)  # back onto the old lane
        np.testing.assert_allclose(reward, -2.)
        np.testing.assert_array_equal(state.lane_return, 0)
        np.testing.assert_array_equal(state.sweep_run_length, 0)

    def test_turn_without_straight_run_gets_no_lane_bonus(self):
        state = self.start()
        for h in [0., np.pi/2, np.pi]:
            state, reward = self.move(state, h)
            np.testing.assert_allclose(reward, 6.)

    def test_simultaneous_team_visits_count_overlap(self):
        env = MultiRobotCoverageEnv(CONFIG)
        state = env.reset(jax.random.PRNGKey(5)).replace(robot_positions=jnp.array(
            [[1.02, 1.02], [1.48, 1.48], [4.25, 4.25]]))
        state, _, _, _ = jax.jit(env.step)(state, jnp.array([[-1., 0.]] * 3))
        info = env.get_info(state)
        self.assertEqual(float(info['cell_entries']), 3.)
        # The first two disks straddle cell corners: 4 cells each, sharing
        # (2, 2); the third lies inside one cell.
        self.assertEqual(float(info['covered_cells']), 8.)
        self.assertEqual(float(info['recoverage']), 3. / 8.)

    def test_neighbour_needs_eighty_percent_of_radius(self):
        env = MultiRobotCoverageEnv(CONFIG)
        reach = 0.2 * env.robot_radius
        # Robot 0 enters cell (col 3, row 2) by just under 80% of its radius,
        # robot 2 enters (col 3, row 6) by just over it.
        state = env.reset(jax.random.PRNGKey(5)).replace(robot_positions=jnp.array(
            [[1.5 - reach - 0.01, 1.25], [4.25, 4.25], [1.5 - reach + 0.01, 3.25]]))
        state = state.replace(coverage_grid=jnp.zeros_like(state.coverage_grid))
        state, _, _, _ = jax.jit(env.step)(state, jnp.array([[-1., 0.]] * 3))
        grid = np.asarray(state.coverage_grid)
        self.assertEqual(grid[2, 2], 1.)
        self.assertEqual(grid[2, 3], 0.)
        self.assertEqual(grid[6, 2], 1.)
        self.assertEqual(grid[6, 3], 1.)
        self.assertEqual(grid.sum(), 4.)

    def test_recoverage_survives_autoreset_in_info_and_episode_ring(self):
        vec = VecEnv(2, {**CONFIG, 'max_steps': 1})
        state, _, _, _ = vec.reset(jax.random.PRNGKey(3))
        # Simulate earlier visits, then terminate without entering another cell.
        cols, rows = jax.vmap(vec.env._pos_to_cell)(state.robot_positions)
        cell = jnp.stack([cols, rows], axis=-1)
        grid = jnp.zeros_like(state.coverage_grid).at[
            jnp.arange(2)[:, None], rows, cols].set(1.)
        covered = jnp.sum(grid, axis=(1, 2))
        state = state.replace(last_visit=cell, coverage_grid=grid, cell_entries=covered * 2)
        state, _, _, _, done, info, _ = vec.step(state, jnp.array([[[-1., 0.]] * 3] * 2))
        np.testing.assert_array_equal(done, True)
        np.testing.assert_array_equal(info['recoverage'], 2.)
        np.testing.assert_array_equal(state.cell_entries, 0.)
        np.testing.assert_array_equal(jax.vmap(vec.env.get_info)(state)['recoverage'], 1.)
        fields = {k: jnp.zeros(2) for k in Transition._fields if k != 'memory'}
        fields.update(reward=jnp.zeros((2, 3)), done=jnp.ones(2), timeout=jnp.ones(2))
        stats = jax.jit(_episode_stats_step)(
            _episode_stats_init(2), Transition(**fields), 1., info['recoverage'])
        self.assertEqual(int(stats.ring_count), 2)
        np.testing.assert_array_equal(stats.recent_recoverage[:2], 2.)


if __name__ == '__main__':
    unittest.main()


class SweepShapingTests(unittest.TestCase):
    """Boustrophedon shaping in progress mode: straight 3, turn 5, break 7."""
    @classmethod
    def setUpClass(cls):
        cls.env = MultiRobotCoverageEnv({
            **CONFIG, 'num_robots': 1, 'observation_stack': 1, 'dt': 0.5,
            'use_full_memory': False, 'reward_mode': 'progress', 'sweep_obs': True,
            'alpha': 0., 'progress_weight': 0., 'loiter_cost': 0., 'tau': 0.,
            'completion_bonus': 0., 'room_completion_bonus': 0.,
            'sweep_straight_bonus': 3., 'sweep_turn_bonus': 5., 'sweep_break_cost': 7.,
        })
        cls.step = staticmethod(jax.jit(cls.env.step))

    start = SweepRewardTests.start
    move = SweepRewardTests.move

    def east_twice(self):
        state = self.start()
        state, reward = self.move(state, 0.)
        np.testing.assert_allclose(reward, 0., atol=1e-5)   # no lane yet
        state, reward = self.move(state, 0.)
        np.testing.assert_allclose(reward, 3., atol=1e-5)   # straight ahead
        return state

    def test_leaving_open_lane_is_a_break(self):
        state = self.east_twice()
        obs = self.env.get_obs(state)[0, self.env.obs_vec_dim - 3:self.env.obs_vec_dim]
        np.testing.assert_allclose(obs, [1., 0., 1.])        # east, next cell open
        state, reward = self.move(state, np.pi / 2)
        np.testing.assert_allclose(reward, -7., atol=1e-5)

    def test_lane_end_turn_then_reverse(self):
        state = self.east_twice()
        state = state.replace(mem_covered=state.mem_covered.at[0, 2, 5].set(1.))
        obs = self.env.get_obs(state)[0, self.env.obs_vec_dim - 3:self.env.obs_vec_dim]
        np.testing.assert_allclose(obs, [1., 0., 0.])        # east, lane finished
        # Shift north at the lane end, reverse west, continue west.
        for heading, expected in [(np.pi / 2, 5.), (np.pi, 3.), (np.pi, 3.)]:
            state, reward = self.move(state, heading)
            np.testing.assert_allclose(reward, expected, atol=1e-5)
