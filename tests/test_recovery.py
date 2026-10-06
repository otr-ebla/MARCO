import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import unittest
from unittest.mock import patch
from contextlib import nullcontext
import jax
import jax.numpy as jnp
import numpy as np

from src.envs.coverage_vector_env import COVERED, FREE, OCCUPIED, UNKNOWN, MultiRobotCoverageEnv
from src.envs.vec_env import VecEnv
from src.envs.recovery import astar, dwa, velocity_sequence, command_is_safe
from src.algorithms.mappo import MAPPO, compute_gae
from src.models.actor_critic import Actor, Critic


# These trajectory fixtures use the original seed-0 room coordinates.
CONFIG = dict(num_maps=1, wall_cells=0, num_robots=1, n_rays=36, obs_mode='memory_comm',
              reward_mode='progress', max_steps=1000, alpha=0., progress_weight=0.,
              loiter_cost=0., tau=0., spread_weight=0., completion_bonus=0.,
              room_completion_bonus=0., revisit_cost=2., fallback_cost=10.)


def set_walls(env, walls):
    """Replace the map's wall rectangles and every grid derived from them."""
    env.walls = jnp.array(walls)
    env._wall_x0, env._wall_y0 = env.walls[..., 0], env.walls[..., 1]
    env._wall_x1, env._wall_y1 = env.walls[..., 2], env.walls[..., 3]
    env.free_masks = jnp.asarray(np.stack([env._compute_free_mask(w) for w in walls]))
    env._free_flat = env.free_masks.reshape(env.num_maps, -1)
    env.free_totals = jnp.sum(env.free_masks, axis=(1, 2))
    env.wall_grids = 1. - env.free_masks
    env._cell_labels = jnp.where(env._free_flat > 0, FREE, OCCUPIED).astype(jnp.int8)
    env.room_masks = env.free_masks[:, None]
    env.room_totals = jnp.sum(env.room_masks, axis=(2, 3))


class PlannerTests(unittest.TestCase):
    def test_nearest_is_by_reachable_path_length(self):
        free = jnp.ones((7, 7), bool).at[1:, 4].set(False)
        targets = jnp.zeros_like(free).at[3, 5].set(True).at[3, 0].set(True)
        goal, waypoint = jax.jit(astar)(free, targets, jnp.int32(24))
        self.assertEqual(int(goal), 21)  # farther in Euclidean distance, shorter path
        self.assertEqual(int(waypoint), 23)

    def test_unknown_unreachable_empty_and_wall_edges(self):
        free = jnp.ones((3, 4), bool)
        targets = jnp.zeros_like(free).at[1, 2].set(True)
        edges = jnp.zeros((3, 4, 4), bool).at[:, 1, 3].set(True).at[:, 2, 2].set(True)
        for allowed, blocked in ((free.at[:, 2].set(False), jnp.zeros_like(edges)), (free, edges)):
            goal, waypoint = jax.jit(astar)(allowed, targets, jnp.int32(5), blocked)
            self.assertEqual(int(goal), -1)
            self.assertEqual(int(waypoint), 5)
        goal, _ = jax.jit(astar)(free, jnp.zeros_like(free), jnp.int32(5))
        self.assertEqual(int(goal), -1)
        goal, waypoint = jax.jit(astar)(free, targets, jnp.int32(6))
        self.assertEqual((int(goal), int(waypoint)), (6, 6))


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MultiRobotCoverageEnv(CONFIG)
        cls.step = staticmethod(jax.jit(cls.env.step))

    def state(self, env=None):
        env = env or self.env
        state = env.reset(jax.random.PRNGKey(1))
        coverage = env.free_masks[0].at[2, 8].set(0.)
        state = state.replace(robot_positions=jnp.array([[1.25, 1.25]]),
                              robot_headings=jnp.zeros(1), coverage_grid=coverage,
                              last_visit=jnp.array([[2, 2]]),
                              mem_state=env.belief_map(0, jnp.ones((1, *coverage.shape)), coverage[None]))
        return env._refresh_memory(state, jnp.array([True]))

    def test_incremental_penalty_seventh_entry_and_single_activation_charge(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'dt': 0.5})
        step = jax.jit(env.step)
        state = self.state(env)
        for index in range(1, 8):
            heading = 0. if index % 2 else np.pi
            state, reward, _, _ = step(state.replace(robot_headings=jnp.array([heading])),
                                        jnp.array([[1., 0.]]))
            np.testing.assert_allclose(reward, -2. * index - (10. if index == 7 else 0.), atol=1e-5)
            self.assertEqual(int(state.revisit_streak[0]), index)
            self.assertEqual(bool(state.fallback_active[0]), index == 7)
            self.assertFalse(bool(state.fallback_used[0]))
        self.assertEqual(int(state.fallback_count[0]), 1)
        state, _, _, _ = step(state, jnp.array([[-1., 0.]]))
        self.assertTrue(bool(state.fallback_used[0]))
        self.assertFalse(bool(state.fallback_activated[0]))
        self.assertEqual(int(state.fallback_count[0]), 1)

    def test_discovery_resets_streak_and_dwelling_does_not_charge_entries(self):
        state = self.state().replace(revisit_streak=jnp.array([4], jnp.int32))
        state, reward, _, _ = self.step(state, jnp.array([[-1., 0.]]))
        np.testing.assert_array_equal(reward, 0.)
        self.assertEqual(int(state.revisit_streak[0]), 4)
        state = state.replace(robot_positions=jnp.array([[4.01, 1.25]]))
        state, reward, _, _ = self.step(state, jnp.array([[-1., 0.]]))
        self.assertEqual(int(state.revisit_streak[0]), 0)

    def test_stall_trigger_and_no_reachable_work_does_not_charge(self):
        state = self.state().replace(no_progress_steps=jnp.array([69], jnp.int32))
        recovered, reward, _, _ = self.step(state, jnp.array([[-1., 0.]]))
        np.testing.assert_array_equal(reward, -10.)
        self.assertTrue(bool(recovered.fallback_active[0]))
        exhausted = state.replace(mem_state=self.env.belief_map(0, jnp.ones(state.mem_state.shape),
                                                                jnp.ones(state.mem_state.shape)))
        exhausted, reward, _, _ = self.step(exhausted, jnp.array([[-1., 0.]]))
        np.testing.assert_array_equal(reward, 0.)
        self.assertFalse(bool(exhausted.fallback_active[0]))

    def test_dwa_overrides_stop_policy_reaches_goal_and_releases(self):
        state = self.state().replace(fallback_active=jnp.array([True]),
                                    fallback_goal=jnp.array([2 * self.env.grid_w + 8]))
        for _ in range(160):
            state, _, _, _ = self.step(state, jnp.array([[-1., 0.]]))
            self.assertEqual(float(state.wall_hits[0]), 0.)
            if not bool(state.fallback_active[0]):
                break
        self.assertFalse(bool(state.fallback_active[0]))
        self.assertEqual(float(state.coverage_grid[2, 8]), 1.)
        self.assertEqual(int(state.revisit_streak[0]), 0)
        previous = state.robot_positions
        state, _, _, _ = self.step(state, jnp.array([[-1., 0.]]))
        np.testing.assert_array_equal(state.robot_positions, previous)
        self.assertFalse(bool(state.fallback_used[0]))

    def test_discoveries_on_route_do_not_release_before_goal_is_covered(self):
        state = self.state()
        coverage = state.coverage_grid.at[:, 5].set(0.)   # every route crosses column 5
        state = state.replace(coverage_grid=coverage,
                              mem_state=self.env.belief_map(0, jnp.ones((1, *coverage.shape)), coverage[None]),
                              fallback_active=jnp.array([True]),
                              fallback_goal=jnp.array([2 * self.env.grid_w + 8]))
        discovered_on_route = False
        for _ in range(200):
            before = float(state.coverage_grid.sum())
            state, _, _, _ = self.step(state, jnp.array([[-1., 0.]]))
            if float(state.coverage_grid[2, 8]) == 0.:
                discovered_on_route |= float(state.coverage_grid.sum()) > before
                self.assertTrue(bool(state.fallback_active[0]))
            if not bool(state.fallback_active[0]):
                break
        self.assertTrue(discovered_on_route)
        self.assertFalse(bool(state.fallback_active[0]))
        self.assertEqual(float(state.coverage_grid[2, 8]), 1.)

    def test_goal_covered_by_teammate_does_not_release_before_own_coverage(self):
        # The grid says the goal is covered, the robot's memory does not:
        # recovery keeps going until the robot's own footprint covers it.
        state = self.state()
        far = np.argwhere(np.asarray(self.env.free_masks[0]) > 0)[-1]   # keeps the episode running
        state = state.replace(coverage_grid=state.coverage_grid.at[2, 8].set(1.).at[far[0], far[1]].set(0.),
                              fallback_active=jnp.array([True]),
                              fallback_goal=jnp.array([2 * self.env.grid_w + 8]))
        state, _, _, _ = self.step(state, jnp.array([[-1., 0.]]))
        self.assertTrue(bool(state.fallback_active[0]))
        for _ in range(160):
            state, _, _, _ = self.step(state, jnp.array([[-1., 0.]]))
            if not bool(state.fallback_active[0]):
                break
        self.assertFalse(bool(state.fallback_active[0]))
        self.assertEqual(int(state.mem_state[0, 2, 8]), COVERED)

    def test_recovery_detours_around_observed_wall(self):
        self._wall_detour(stuck=False)

    def test_sequence_detours_around_wall_when_dwa_is_stuck(self):
        self._wall_detour(stuck=True)

    def _wall_detour(self, stuck):
        env = MultiRobotCoverageEnv({**CONFIG, 'n_rays': 70,
                                     'fallback_dwa_stall_steps': 3 if stuck else 30})
        # A one-cell interior wall band (column 5, rows 0-4) blocks the direct
        # path; its end leaves a detour. The belief map sees it as occupied cells.
        walls = np.asarray(env.walls).copy()
        walls[0, 4:] = [-10., -10., -10., -10.]
        walls[0, 4] = [2.5, 0., 3.0, 2.5]
        set_walls(env, walls)
        state = self.state(env)
        self.assertTrue(np.all(np.asarray(state.mem_state[0, 0:5, 5]) <= OCCUPIED))
        coverage = env.free_masks[0].at[2, 7].set(0.)
        state = state.replace(coverage_grid=coverage,
                              mem_state=env.belief_map(0, jnp.ones((1, *coverage.shape)), coverage[None]),
                              fallback_active=jnp.array([True]),
                              fallback_goal=jnp.array([2 * env.grid_w + 7]))
        controller = patch('src.envs.coverage_vector_env.dwa', return_value=jnp.array([-1., 0.])) if stuck else nullcontext()
        with controller:
            step = jax.jit(env.step)
            crossed_above_wall = False
            for _ in range(600 if stuck else 400):
                state, _, _, _ = step(state, jnp.array([[-1., 0.]]))
                self.assertEqual(float(state.wall_hits[0]), 0.)
                crossed_above_wall |= float(state.robot_positions[0, 1]) > 2.7
                if not bool(state.fallback_active[0]):
                    break
        self.assertTrue(crossed_above_wall)
        self.assertFalse(bool(state.fallback_active[0]))
        self.assertEqual(float(state.coverage_grid[2, 7]), 1.)

    def test_velocity_sequence_reaches_target_when_dwa_is_stuck(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'fallback_dwa_stall_steps': 3,
                                     'fallback_sequence_steps': 32})
        state = self.state(env).replace(
            robot_headings=jnp.array([np.pi / 2], jnp.float32),
            lidar=jnp.ones((1, env.n_rays)),
            fallback_active=jnp.array([True]),
            fallback_goal=jnp.array([2 * env.grid_w + 8]))
        # Disable only DWA, while keeping physics, planning, sensing and the
        # secondary controller real. Its queue must refill to reach this goal.
        with patch('src.envs.coverage_vector_env.dwa', return_value=jnp.array([-1., 0.])):
            step = jax.jit(env.step)
            used_sequence = False
            refills = 0
            for tick in range(500):
                state, _, _, _ = step(state, jnp.array([[-1., 0.]]))
                if tick < 3:
                    self.assertFalse(bool(state.fallback_sequence_used[0]))
                used_sequence |= bool(state.fallback_sequence_used[0])
                refills += int(state.fallback_command_index[0] == 1)
                self.assertEqual(float(state.wall_hits[0]), 0.)
                self.assertEqual(int(state.fallback_count[0]), 0)  # escalation adds no activation fee
                if not bool(state.fallback_active[0]):
                    break
        self.assertTrue(used_sequence)
        self.assertGreater(refills, 1)
        self.assertFalse(bool(state.fallback_active[0]))
        self.assertEqual(float(state.coverage_grid[2, 8]), 1.)
        self.assertEqual(int(state.fallback_command_count[0]), 0)
        self.assertFalse(bool(state.fallback_sequence[0]))

    def test_sequence_contains_turn_and_drive_commands_with_bounded_speed(self):
        env = self.env
        route = jnp.full(env.num_cells, -1, jnp.int32).at[:3].set(jnp.array([50, 51, 75]))
        commands, count = jax.jit(lambda: velocity_sequence(
            env, jnp.array([1.25, 1.25]), jnp.float32(np.pi / 2), jnp.zeros(2), route, jnp.int32(3)))()
        commands = np.asarray(commands[:int(count)])
        self.assertTrue(np.any((commands[:, 0] == 0.) & (np.abs(commands[:, 1]) > .1)))
        self.assertTrue(np.any(commands[:, 0] > .1))
        self.assertTrue(np.all((commands[:, 0] >= 0.) & (commands[:, 0] <= env.fallback_sequence_speed)))
        self.assertTrue(np.all(np.abs(commands[:, 1]) <= env.omega_max))

    def test_new_lidar_obstacle_interrupts_sequence_then_clear_scan_resumes(self):
        env = self.env
        state = self.state().replace(
            fallback_active=jnp.array([True]), fallback_sequence=jnp.array([True]),
            fallback_goal=jnp.array([2 * env.grid_w + 8]),
            fallback_command_count=jnp.array([1]),
            fallback_commands=jnp.zeros_like(self.state().fallback_commands).at[0, 0].set(jnp.array([.3, 0.])),
            lidar=jnp.ones((1, env.n_rays)))
        control = jax.jit(env._recovery_actions)
        _, _, _, clear = control(state, jnp.array([[-1., 0.]]))
        self.assertTrue(bool(clear['fallback_sequence_used'][0]))
        blocked = state.replace(lidar=state.lidar.at[0, 0].set(.24 / env.max_lidar_range))
        action, goal, used, changed = control(blocked, jnp.array([[1., 0.]]))
        self.assertTrue(bool(used[0]))  # still excluded from PPO actor updates
        self.assertTrue(bool(changed['fallback_safety_override'][0]))
        self.assertFalse(bool(changed['fallback_sequence_used'][0]))
        self.assertEqual(int(changed['fallback_command_count'][0]), 0)
        self.assertAlmostEqual(float(action[0, 0]), -1.)
        resumed = state.replace(**changed, fallback_goal=goal)
        _, _, _, resumed_control = control(resumed, jnp.array([[-1., 0.]]))
        self.assertTrue(bool(resumed_control['fallback_sequence_used'][0]))
        self.assertFalse(bool(resumed_control['fallback_safety_override'][0]))
        # A transient lidar obstacle is not written into the belief map.
        np.testing.assert_array_equal(blocked.mem_state, state.mem_state)

    def test_near_wall_rotation_and_escape_are_not_blocked_by_preferred_margin(self):
        env = self.env
        # Legal clearance .21 m is inside the preferred .23 m DWA buffer.
        lidar = jnp.ones(env.n_rays).at[env.n_rays // 2].set(.21 / env.max_lidar_range)
        free = jnp.ones((env.grid_h, env.grid_w), bool)
        edges = jnp.zeros((*free.shape, 4), bool)
        safe = jax.jit(lambda command: command_is_safe(
            env, jnp.array([1.25, 1.25]), jnp.float32(0.), command, lidar, free, edges))
        self.assertTrue(bool(safe(jnp.array([0., .5]))))
        self.assertTrue(bool(safe(jnp.array([.2, 0.]))))  # moves away from the hit behind it
        action = jax.jit(lambda: dwa(env, jnp.array([1.25, 1.25]), jnp.float32(0.),
                                     jnp.zeros(2), lidar, free, jnp.array([1.25, 2.25]), edges))()
        self.assertGreater(abs(float(action[1])), 0.)

    def test_diff_drive_keeps_translation_for_tiny_angular_velocity(self):
        env = self.env
        omega = jnp.array([0., 1.01e-6, -1.75e-6, .5])
        heading = jnp.full(4, 2 * np.pi - 1e-6)
        pos = jnp.ones((4, 2))
        result, _ = jax.jit(env._diff_drive)(pos, heading, jnp.full(4, .4), omega)
        half = .5 * np.asarray(omega, dtype=np.float64) * env.dt
        expected = np.asarray(pos) + (.4 * env.dt * np.sinc(half / np.pi))[:, None] * np.stack(
            [np.cos(np.asarray(heading) + half), np.sin(np.asarray(heading) + half)], axis=1)
        np.testing.assert_allclose(result, expected, atol=1e-7)
        self.assertTrue(np.all(np.linalg.norm(np.asarray(result - pos), axis=1) > .039))

    def test_shared_covered_target_is_replanned_without_activation_fee(self):
        state = self.state().replace(fallback_active=jnp.array([True]),
                                    fallback_goal=jnp.array([2 * self.env.grid_w + 7]))
        _, goal, used, _ = jax.jit(self.env._recovery_actions)(state, jnp.array([[-1., 0.]]))
        self.assertTrue(bool(used[0]))
        self.assertEqual(int(goal[0]), 2 * self.env.grid_w + 8)

    def test_dwa_stops_when_lidar_has_no_safe_trajectory(self):
        env = self.env
        action = jax.jit(lambda: dwa(env, jnp.array([1.25, 1.25]), jnp.float32(0.),
                                    jnp.array([1., 0.]), jnp.full(env.n_rays, .01),
                                    jnp.ones((env.grid_h, env.grid_w), bool), jnp.array([2., 1.25]),
                                    jnp.zeros((env.grid_h, env.grid_w, 4), bool)))()
        np.testing.assert_array_equal(action, [-1., 0.])

    def test_memory_or_strict_range_and_legacy_recovery_memory(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'num_robots': 2, 'obs_mode': 'legacy',
                                     'max_lidar_range': .1})
        state = env.reset(jax.random.PRNGKey(2))
        # Robot 0 knows a covered cell, robot 1 a free one and an occupied one.
        mem = (jnp.zeros_like(state.mem_state).at[0, 10, 10].set(COVERED)
               .at[1, 10, 11].set(FREE).at[1, 10, 12].set(OCCUPIED))
        state = state.replace(robot_positions=jnp.array([[1., 1.], [3.99, 1.]]), mem_state=mem)
        merged = env._refresh_memory(state, jnp.zeros(2, bool))
        # In range the dictionaries are merged; covered wins over free/unknown.
        for robot in range(2):
            np.testing.assert_array_equal(merged.mem_state[robot, 10, 10:13], [COVERED, FREE, OCCUPIED])
        merged = env._refresh_memory(merged.replace(
            mem_state=merged.mem_state.at[1, 10, 11].set(COVERED)), jnp.zeros(2, bool))
        np.testing.assert_array_equal(merged.mem_state[:, 10, 11], COVERED)
        separated = env._refresh_memory(state.replace(robot_positions=jnp.array([[1., 1.], [4., 1.]])),
                                        jnp.zeros(2, bool))
        np.testing.assert_array_equal(separated.mem_state[:, 10, 10:13], mem[:, 10, 10:13])
        self.assertTrue(env.track_memory)

    def test_lidar_stores_seen_cells_with_their_label(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'wall_cells': 1})
        state = env.reset(jax.random.PRNGKey(1))
        mem = np.asarray(state.mem_state[0])
        labels = np.asarray(env._cell_labels[0]).reshape(mem.shape)
        seen = mem != UNKNOWN
        self.assertTrue(np.any(mem == OCCUPIED) and np.any(mem >= FREE))
        # Seen walls are occupied, seen floor is free or covered; nothing else is stored.
        np.testing.assert_array_equal(mem[seen & (labels == OCCUPIED)], OCCUPIED)
        self.assertTrue(np.all(mem[seen & (labels == FREE)] >= FREE))
        self.assertEqual(state.mem_state.dtype, jnp.int8)

    def test_vec_autoreset_clears_recovery_and_preserves_diagnostics(self):
        vec = VecEnv(2, {**CONFIG, 'max_steps': 1})
        state, _, _, _ = vec.reset(jax.random.PRNGKey(3))
        state = state.replace(fallback_count=jnp.full((2, 1), 4), revisit_streak=jnp.full((2, 1), 6),
                              fallback_sequence=jnp.ones((2, 1), bool),
                              fallback_command_count=jnp.full((2, 1), 10),
                              fallback_commands=jnp.ones_like(state.fallback_commands))
        state, _, _, _, done, info, _ = vec.step(state, jnp.zeros((2, 1, 2)))
        np.testing.assert_array_equal(done, True)
        np.testing.assert_array_equal(info['fallback_count'], 4)
        np.testing.assert_array_equal(state.fallback_count, 0)
        np.testing.assert_array_equal(state.revisit_streak, 0)
        np.testing.assert_array_equal(state.fallback_active, False)
        np.testing.assert_array_equal(state.fallback_sequence, False)
        np.testing.assert_array_equal(state.fallback_command_count, 0)
        np.testing.assert_array_equal(state.fallback_commands, 0.)

    def test_critic_contains_full_map_and_joint_poses_velocities(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'num_robots': 2})
        state = env.reset(jax.random.PRNGKey(4)).replace(
            robot_positions=jnp.array([[1., 2.], [7., 6.]]),
            robot_headings=jnp.array([0., np.pi / 2]),
            robot_velocities=jnp.array([[.2, -.3], [.7, .8]]))
        global_state = env.get_global_state(state)
        grid, vector = env.critic_inputs(global_state)
        np.testing.assert_array_equal(grid[:, 0], jnp.broadcast_to(env.wall_grids[state.map_id], grid[:, 0].shape))
        np.testing.assert_array_equal(grid[:, 1], jnp.broadcast_to(state.coverage_grid, grid[:, 1].shape))
        np.testing.assert_allclose(vector[:, 6:], jnp.broadcast_to(global_state.kinematics.reshape(-1), (2, 12)))
        np.testing.assert_allclose(global_state.kinematics[:, 4:], state.robot_velocities)


class TrainingMaskTests(unittest.TestCase):
    def test_recovery_rollout_and_all_forced_actor_update(self):
        vec = VecEnv(1, CONFIG)
        env = vec.env
        actor = Actor(vec_dim=env.obs_vec_dim, n_rays=env.n_rays, tail_dim=env.patch_dim,
                      hidden_size=8, lidar_embed=8)
        algo = MAPPO(actor, Critic(hidden_size=8, map_embed=8), vec, dict(n_epochs=1))
        actor_state, critic_state = algo.create_train_states(jax.random.PRNGKey(9))
        carry = algo.init_carry(jax.random.PRNGKey(10))
        state = carry.env_state.replace(fallback_active=jnp.ones((1, 1), bool))
        carry, trajectory, value = algo.rollout(actor_state.params, critic_state.params,
                                               carry._replace(env_state=state), 3, jax.random.PRNGKey(11))
        self.assertEqual(float(trajectory.policy_mask[0, 0, 0]), 0.)
        advantages, returns = compute_gae(trajectory, value, .99, .95)
        # Establish Adam momentum, then ensure a fully forced batch changes
        # neither actor parameters nor its optimizer state.
        actor_state, critic_state, _ = algo.update(
            actor_state, critic_state, trajectory._replace(policy_mask=jnp.ones_like(advantages)),
            advantages, returns, .001, .001)
        before = jax.tree_util.tree_map(lambda x: np.array(x), actor_state)
        actor_state, critic_state, metrics = algo.update(
            actor_state, critic_state, trajectory._replace(policy_mask=jnp.zeros_like(advantages)),
            advantages, returns, .001, .001)
        for old, new in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(actor_state)):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(float(metrics['actor_loss']), 0.)
        self.assertEqual(float(metrics['entropy']), 0.)
        self.assertTrue(np.isfinite(float(metrics['critic_loss'])))


if __name__ == '__main__':
    unittest.main()
