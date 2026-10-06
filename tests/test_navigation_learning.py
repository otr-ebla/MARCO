"""Regression checks for map topology, cleanup rewards and recovery teaching."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import unittest
import pickle
import tempfile
import jax
import jax.numpy as jnp
import numpy as np

from src.envs.coverage_vector_env import OCCUPIED, MultiRobotCoverageEnv, _FAR
from src.envs.map_layouts import create_map_bank
from src.envs.vec_env import VecEnv
from src.algorithms.mappo import MAPPO, compute_gae, recovery_imitation_loss
from src.models.actor_critic import Actor, Critic
from src.train_marl import Rollout


CONFIG = dict(num_maps=1, num_robots=1, n_rays=12, obs_mode='memory_comm',
              history_cell='previous_visit', use_full_memory=True, critic_context=True,
              wall_cells=1, reward_mode='sequential',
              revisit_cost=6., revisit_end_fraction=.025, revisit_decay_power=2.,
              revisit_streak_cap=3, max_steps=100)


class NavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MultiRobotCoverageEnv(CONFIG)
        cls.state = jax.jit(cls.env.reset)(jax.random.PRNGKey(2))

    def test_walls_are_whole_cells_and_all_free_centres_are_connected(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'num_maps': 64})
        layouts = create_map_bank(64, seed=0, wall_cells=1)
        xx, yy = env._cell_centers()
        centres = np.stack([xx, yy], -1).reshape(-1, 2)
        for i, layout in enumerate(layouts):
            walls = layout.get_walls()
            real = walls[walls[:, 0] > -5.]
            np.testing.assert_allclose(real / env.cell_size, np.round(real / env.cell_size))
            self.assertTrue(np.all(real[:, 2:] - real[:, :2] >= env.cell_size))
            delta = centres[:, None] - np.clip(centres[:, None], walls[None, :, :2], walls[None, :, 2:])
            clear = np.all(np.sum(delta ** 2, -1) >= env.robot_radius ** 2, axis=-1)
            # _compute_free_mask keeps only the largest component: equality
            # proves no geometrically free cells were silently discarded.
            np.testing.assert_array_equal(clear.reshape(env.grid_h, env.grid_w), env.free_mask_np[i])
            self.assertGreater(float(env.wall_grids[i].sum()), 0.)
        smallest = int(np.argmin(np.asarray(env.free_totals)))
        count = int(env.free_totals[smallest])
        spawns = jax.jit(lambda key: env._sample_spawns(key, smallest, count))(jax.random.PRNGKey(0))
        self.assertEqual(len(np.unique(np.asarray(spawns), axis=0)), count)

    def test_wall_memory_and_physics_agree(self):
        env, state = self.env, self.state
        # Seen walls are stored as occupied cells, and only real wall cells are.
        occupied = np.asarray(state.mem_state[0] == OCCUPIED)
        self.assertGreater(occupied.sum(), 0)
        np.testing.assert_array_equal(np.asarray(env.wall_grids[0])[occupied], 1.)
        rows, cols = np.nonzero(np.asarray(env.wall_grids[0]))
        points = (jnp.array(np.stack([cols, rows], -1)) + .5) * env.cell_size
        np.testing.assert_array_equal(env._wall_collision(points, jnp.int32(0)), True)

    def test_occupied_cells_require_a_detour(self):
        env = self.env
        # A 3x3 free block whose centre (row 3, col 3) is an occupied cell.
        free = jnp.zeros((1, env.grid_h, env.grid_w), bool).at[:, 2:5, 2:5].set(True).at[:, 3, 3].set(False)
        targets = jnp.zeros_like(free).at[:, 3, 4].set(True)
        field = env._geodesic_distance(targets, free)
        self.assertEqual(float(field[0, 3, 2]), 4.)
        distance = env._work_distance(field, jnp.array([[1.25, 1.75]]))
        np.testing.assert_allclose(distance, 4.)
        # Occupying the whole middle column makes the work unreachable.
        field = env._geodesic_distance(targets, free.at[:, 2:5, 3].set(False))
        self.assertGreaterEqual(float(field[0, 3, 2]), _FAR)

    def test_revisit_cost_decays_with_time_or_coverage(self):
        env, state = self.env, self.state
        np.testing.assert_allclose(env._revisit_weight(state), 6.)
        timed = [float(env._revisit_weight(state.replace(step_count=jnp.int32(t))))
                 for t in (0, 25, 50, 75, 100)]
        self.assertTrue(all(a > b for a, b in zip(timed, timed[1:])))
        self.assertAlmostEqual(timed[-1], .15, places=5)
        np.testing.assert_allclose(env._revisit_weight(state.replace(coverage_grid=env.free_masks[0])), .15)
        np.testing.assert_allclose(env._revisit_weight(state.replace(step_count=jnp.int32(200))), .15)

    def test_actor_receives_previous_cell_but_no_planned_destination(self):
        env, state = self.env, self.state
        changed = state.replace(step_count=jnp.int32(50), revisit_streak=jnp.array([5]),
                                fallback_active=jnp.ones(1, bool), fallback_goal=jnp.array([25]))
        first = env.critic_inputs(env.get_global_state(state))[1]
        second = env.critic_inputs(env.get_global_state(changed))[1]
        self.assertEqual(first.shape, (1, env.critic_vec_dim))
        self.assertFalse(np.array_equal(first, second))
        # Planner destination and control history are not actor observations.
        np.testing.assert_array_equal(env._get_frame_obs(state), env._get_frame_obs(changed))
        history = state.replace(previous_visit=jnp.array([[2, 3]]))
        obs = env._get_frame_obs(history)
        offset = env.obs_vec_dim - 5
        np.testing.assert_allclose(obs[0, offset:offset + 3],
            [1., (1.25 - float(state.robot_positions[0, 0])) / env.map_layout.width,
             (1.75 - float(state.robot_positions[0, 1])) / env.map_layout.height], atol=1e-6)
        self.assertFalse(np.array_equal(env._get_frame_obs(state), obs))

    def test_previous_cell_updates_on_revisits_and_survives_dwelling(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'wall_cells': 0, 'dt': .5,
                                     'fallback_enabled': False})
        state = env.reset(jax.random.PRNGKey(4)).replace(
            robot_positions=jnp.array([[1.25, 1.25]]), robot_headings=jnp.zeros(1))
        step = jax.jit(env.step)
        state, *_ = step(state, jnp.array([[-1., 0.]]))
        np.testing.assert_array_equal(state.previous_visit, -1)
        state, *_ = step(state, jnp.array([[1., 0.]]))
        np.testing.assert_array_equal(state.previous_visit, [[2, 2]])
        state, *_ = step(state.replace(robot_headings=jnp.array([np.pi])), jnp.array([[1., 0.]]))
        np.testing.assert_array_equal(state.previous_visit, [[3, 2]])
        state, *_ = step(state, jnp.array([[-1., 0.]]))
        np.testing.assert_array_equal(state.previous_visit, [[3, 2]])
        np.testing.assert_array_equal(env.reset(jax.random.PRNGKey(4)).previous_visit, -1)

    def test_teacher_excludes_contacts_and_stationary_recovery(self):
        env, state = self.env, self.state
        state = state.replace(fallback_used=jnp.ones(1, bool), robot_velocities=jnp.array([[.4, .2]]))
        info = env.get_info(state)
        np.testing.assert_allclose(info['executed_action'], [[-.2, .2]], atol=1e-6)
        np.testing.assert_array_equal(info['teacher_mask'], 1.)
        for bad in (state.replace(wall_hits=jnp.ones(1)),
                    state.replace(robot_hits=jnp.ones(1)),
                    state.replace(robot_velocities=jnp.zeros((1, 2)))):
            np.testing.assert_array_equal(env.get_info(bad)['teacher_mask'], 0.)


class ImitationTests(unittest.TestCase):
    def test_only_valid_teacher_samples_produce_gradients(self):
        mean = jnp.zeros((2, 2))
        target = jnp.array([[.5, -.5], [1., 1.]])
        grad = jax.grad(recovery_imitation_loss)(mean, target, jnp.array([1., 0.]))
        self.assertLess(float(grad[0, 0]), 0.)
        self.assertGreater(float(grad[0, 1]), 0.)
        np.testing.assert_array_equal(grad[1], 0.)
        np.testing.assert_array_equal(jax.grad(recovery_imitation_loss)(mean, target, jnp.zeros(2)), 0.)

    def test_recurrent_training_learns_from_recovery_without_ppo_samples(self):
        vec = VecEnv(1, {**CONFIG, 'observation_stack': 2, 'max_steps': 2})
        env = vec.env
        actor = Actor(vec_dim=env.obs_vec_dim, n_rays=env.n_rays, tail_dim=env.patch_dim,
                      observation_stack=2, recurrent=True, hidden_size=8, lidar_embed=8,
                      memory_map_shape=env.memory_map_shape)
        algo = MAPPO(actor, Critic(hidden_size=8, map_embed=8), vec,
                     dict(n_epochs=1, recovery_imitation_coef=1.))
        actor_state, critic_state = algo.create_train_states(jax.random.PRNGKey(3))
        rollout = Rollout(algo, vec)
        carry = rollout.start(jax.random.PRNGKey(4))
        _, trajectory, value = rollout.run(actor_state.params, critic_state.params,
                                         carry, 3, jax.random.PRNGKey(5))
        self.assertTrue(np.any(np.asarray(trajectory.done)))
        self.assertEqual(trajectory.teacher_action.shape, (3, 1, 1, 2))
        advantages, returns = compute_gae(trajectory, value, .99, .95)
        trajectory = trajectory._replace(policy_mask=jnp.zeros_like(advantages),
            teacher_mask=jnp.ones_like(advantages), teacher_action=jnp.full_like(trajectory.action, .7))
        params_before = jax.tree_util.tree_map(np.array, actor_state.params)
        actor_state, _, metrics = algo.update(actor_state, critic_state, trajectory,
                                              advantages, returns, .001, .001)
        self.assertEqual(float(metrics['actor_loss']), 0.)
        self.assertEqual(float(metrics['entropy']), 0.)
        self.assertGreater(float(metrics['recovery_imitation_loss']), 0.)
        self.assertEqual(float(metrics['actor_update_fraction']), 1.)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(params_before), jax.tree_util.tree_leaves(actor_state.params))))


class GlobalCriticStackTests(unittest.TestCase):
    def test_ordered_global_frames_keep_each_robots_visit_counts_and_past_cell(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'num_robots': 2, 'critic_stack': 3,
                                     'observation_stack': 3, 'wall_cells': 0})
        state = env.reset(jax.random.PRNGKey(10))
        np.testing.assert_array_equal(state.global_history_valid, [0., 0., 1.])
        visits = state.visit_counts.at[0, 2, 2].set(1.).at[1, 3, 3].set(2.)
        state = env._push_observation(state.replace(
            visit_counts=visits, previous_visit=jnp.array([[2, 2], [3, 3]])))
        np.testing.assert_array_equal(state.global_history_valid, [0., 1., 1.])
        grid, vec = env.critic_inputs(env.get_global_state(state))
        self.assertEqual(grid.shape, (2, env.critic_channels, env.grid_h, env.grid_w))
        self.assertEqual(vec.shape, (2, env.critic_vec_dim))
        grid = np.asarray(grid).reshape(2, 3, env._critic_frame_channels, env.grid_h, env.grid_w)
        self.assertTrue(np.all(grid[:, 0] == 0.))
        np.testing.assert_allclose(grid[:, -1, 4, 2, 2], np.log1p(1) / np.log1p(env.max_steps))
        np.testing.assert_allclose(grid[:, -1, 5, 3, 3], np.log1p(2) / np.log1p(env.max_steps))
        previous = np.asarray(vec).reshape(2, 3, env._critic_frame_vec_dim)[:, -1, -6:]
        np.testing.assert_allclose(previous, [[1., 2.5 / env.grid_w, 2.5 / env.grid_h,
                                               1., 3.5 / env.grid_w, 3.5 / env.grid_h]] * 2)
        # The visit-count grid is privileged critic history, not an actor input.
        np.testing.assert_array_equal(env.get_obs(state), env.get_obs(
            state.replace(visit_counts=jnp.zeros_like(state.visit_counts))))

    def test_global_history_resets_at_timeout_and_trains_with_recurrent_actor(self):
        vec = VecEnv(1, {**CONFIG, 'num_robots': 2, 'critic_stack': 3,
                         'observation_stack': 3, 'max_steps': 2})
        env = vec.env
        actor = Actor(vec_dim=env.obs_vec_dim, n_rays=env.n_rays,
                      tail_dim=env.patch_dim, memory_map_shape=env.memory_map_shape,
                      observation_stack=3, recurrent=True, hidden_size=8, lidar_embed=8)
        algo = MAPPO(actor, Critic(hidden_size=8, map_embed=8), vec,
                     dict(n_epochs=1, recovery_imitation_coef=.5))
        actor_state, critic_state = algo.create_train_states(jax.random.PRNGKey(1))
        carry = Rollout(algo, vec).start(jax.random.PRNGKey(2))
        _, traj, bootstrap = Rollout(algo, vec).run(actor_state.params, critic_state.params,
                                                   carry, 3, jax.random.PRNGKey(3))
        np.testing.assert_array_equal(np.asarray(traj.gstate.history_valid)[:, 0],
                                      [[0., 0., 1.], [0., 1., 1.], [0., 0., 1.]])
        advantage, returns = compute_gae(traj, bootstrap, .99, .95)
        _, _, metrics = algo.update(actor_state, critic_state, traj, advantage, returns, .001, .001)
        self.assertTrue(np.isfinite(float(metrics['critic_loss'])))


class CheckpointHistoryTests(unittest.TestCase):
    def test_visualizer_restores_original_and_new_history_semantics(self):
        from src.visualize_policy import _load_checkpoint
        for metadata, expected_history, expected_walls, expected_stack in (
            ({'obs_mode': 'memory_comm'}, 'last_discovery', 0, 1),
            ({'obs_mode': 'memory_comm', 'history_cell': 'previous_visit',
              'wall_cells': 1, 'critic_context': True, 'critic_stack': 5},
             'previous_visit', 1, 5),
        ):
            with self.subTest(history=expected_history), tempfile.NamedTemporaryFile(suffix='.pkl') as file:
                pickle.dump({'actor_params': {}, 'obs_config': metadata}, file)
                file.flush()
                config = dict(CONFIG)
                _load_checkpoint(file.name, jax.devices('cpu')[0], config)
                self.assertEqual(config['history_cell'], expected_history)
                self.assertEqual(config['wall_cells'], expected_walls)
                self.assertEqual(config['critic_stack'], expected_stack)
                self.assertFalse(config['goal_obs'])

    def test_visualizer_keeps_the_map_only_for_checkpoints_trained_with_it(self):
        from src.visualize_policy import _load_checkpoint
        for metadata, expected in (
            ({'obs_mode': 'memory_comm', 'use_full_memory': True}, True),   # before memory_map_obs
            ({'obs_mode': 'memory_comm', 'use_full_memory': True, 'memory_map_obs': False}, False),
        ):
            with self.subTest(expected=expected), tempfile.NamedTemporaryFile(suffix='.pkl') as file:
                pickle.dump({'actor_params': {}, 'obs_config': metadata}, file)
                file.flush()
                config = dict(CONFIG)
                _load_checkpoint(file.name, jax.devices('cpu')[0], config)
                self.assertEqual(config['memory_map_obs'], expected)


if __name__ == '__main__':
    unittest.main()
