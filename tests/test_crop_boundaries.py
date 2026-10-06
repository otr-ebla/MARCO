"""Cells beyond the floor must never look like unexplored free space."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.coverage_vector_env import FREE, OCCUPIED, MultiRobotCoverageEnv


class CropBoundaryTests(unittest.TestCase):
    def make_env(self, mode):
        env = MultiRobotCoverageEnv(dict(
            num_maps=1, num_robots=9, num_humans=0, n_rays=4,
            obs_mode=mode, local_coverage_size=5, comm_slots=0,
            k_teammates=0, use_full_memory=False, observation_stack=1,
        ))
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        # All four corners, all four edge midpoints, and an interior control.
        cells = np.array([
            (col, row)
            for row in (0, env.grid_h // 2, env.grid_h - 1)
            for col in (0, env.grid_w // 2, env.grid_w - 1)
        ])
        positions = (cells + .5) * env.cell_size
        # Empty boundary cells ensure padding cannot pass by copying a wall.
        env.wall_grids = jnp.zeros_like(env.wall_grids)
        env._cell_labels = jnp.full_like(env._cell_labels, FREE)
        state = state.replace(robot_positions=jnp.asarray(positions),
                              coverage_grid=jnp.zeros_like(state.coverage_grid),
                              mem_state=jnp.where(state.mem_state == OCCUPIED, FREE,
                                                  state.mem_state).astype(jnp.int8))
        return env, state, positions

    def outside_mask(self, env, positions, heading=0.):
        size = env.local_coverage_size
        outside = np.zeros((env.num_robots, size, size), dtype=bool)
        for robot, (x, y) in enumerate(positions):
            for row in range(size):
                for col in range(size):
                    dx = (col - size // 2) * env.cell_size
                    dy = (row - size // 2) * env.cell_size
                    sample_x = x + np.cos(heading) * dx - np.sin(heading) * dy
                    sample_y = y + np.sin(heading) * dx + np.cos(heading) * dy
                    outside[robot, row, col] = not (
                        0 <= sample_x < env.map_layout.width
                        and 0 <= sample_y < env.map_layout.height
                    )
        return outside

    def test_legacy_edges_and_corners_in_rotated_crops(self):
        env, state, positions = self.make_env('legacy')
        get_obs = jax.jit(env.get_obs)
        for heading in (0., np.pi / 2, np.pi / 4, np.pi):
            with self.subTest(heading=heading):
                obs = get_obs(state.replace(
                    robot_headings=jnp.full((env.num_robots,), heading)))
                crop = np.asarray(obs[:, env.frame_norm_dim:]).reshape(
                    env.num_robots, env.local_coverage_size, env.local_coverage_size)
                np.testing.assert_array_equal(
                    crop, self.outside_mask(env, positions, heading).astype(np.float32))

    def test_memory_padding_is_occupied_known_and_not_covered(self):
        env, state, positions = self.make_env('memory_comm')
        get_obs = jax.jit(env.get_obs)
        outside = self.outside_mask(env, positions)
        for known in (0., 1.):
            with self.subTest(interior_known=known):
                obs = get_obs(state.replace(
                    mem_state=env.belief_map(0, jnp.full(state.mem_state.shape, known),
                                             jnp.full(state.mem_state.shape, known)),
                    # Memory crops stay grid-aligned regardless of heading.
                    robot_headings=jnp.full((env.num_robots,), np.pi / 4)))
                crop = np.asarray(obs[:, env.frame_norm_dim:]).reshape(
                    env.num_robots, 3, env.local_coverage_size, env.local_coverage_size)
                np.testing.assert_array_equal(crop[:, 0], outside.astype(np.float32))
                np.testing.assert_array_equal(crop[:, 1], np.where(outside, 0., known))
                np.testing.assert_array_equal(crop[:, 2], np.where(outside, 1., known))


class CropSummaryTests(unittest.TestCase):
    """The ring around the crop averages everything beyond it in each direction."""

    def make_env(self, mode):
        env = MultiRobotCoverageEnv(dict(
            num_maps=1, num_robots=9, num_humans=0, n_rays=4, obs_mode=mode,
            local_coverage_size=5, comm_slots=0, k_teammates=0, crop_summary=True,
            use_full_memory=False, observation_stack=1, sweep_obs=False,
        ))
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        cells = np.array([(col, row) for row in (0, 5, env.grid_h - 1)
                          for col in (0, 7, env.grid_w - 1)])
        rng = np.random.default_rng(0)
        state = state.replace(
            robot_positions=jnp.asarray((cells + .5) * env.cell_size),
            coverage_grid=jnp.asarray(rng.random((env.grid_h, env.grid_w)) < .4, jnp.float32),
            mem_state=env.belief_map(0, rng.random(state.mem_state.shape) < .6,
                                     rng.random(state.mem_state.shape) < .3))
        return env, state, cells

    def expected_memory_ring(self, env, state, cells):
        """Brute force: slice each robot's memory into strips and quadrants."""
        h, s = env.local_coverage_size // 2, env.local_coverage_size
        mem = np.asarray(state.mem_state)
        out = np.zeros((env.num_robots, 4, s + 2, s + 2), np.float32)
        for robot, (col, row) in enumerate(cells):
            maps = [mem[robot] == 1, mem[robot] == 3, mem[robot] != 0]
            spans = {0: (0, max(row - h, 0)), s + 1: (min(row + h + 1, env.grid_h), env.grid_h)}
            col_spans = {0: (0, max(col - h, 0)), s + 1: (min(col + h + 1, env.grid_w), env.grid_w)}
            for er in range(s + 2):
                for ec in range(s + 2):
                    if er not in spans and ec not in col_spans:
                        continue
                    r0, r1 = spans.get(er, (row + er - 1 - h, row + er - h))
                    c0, c1 = col_spans.get(ec, (col + ec - 1 - h, col + ec - h))
                    r0, c0 = max(r0, 0), max(c0, 0)
                    r1, c1 = min(max(r1, r0), env.grid_h), min(max(c1, c0), env.grid_w)
                    count = max(r1 - r0, 0) * max(c1 - c0, 0)
                    for ch, (m, pad) in enumerate(zip(maps, (1., 0., 1.))):
                        out[robot, ch, er, ec] = m[r0:r1, c0:c1].mean() if count else pad
                    corner = er in spans and ec in col_spans
                    out[robot, 3, er, ec] = count / (env.num_cells if corner
                                                     else max(env.grid_h, env.grid_w))
        return out

    def test_memory_ring_matches_brute_force(self):
        env, state, cells = self.make_env('memory_comm')
        s = env.local_coverage_size
        self.assertEqual(env.patch_dim, 4 * (s + 2) ** 2)
        obs = np.asarray(jax.jit(env.get_obs)(state))
        crop = obs[:, env.frame_norm_dim:].reshape(env.num_robots, 4, s + 2, s + 2)
        expected = self.expected_memory_ring(env, state, cells)
        ring = np.ones((s + 2, s + 2), bool)
        ring[1:-1, 1:-1] = False
        np.testing.assert_allclose(crop[:, :, ring], expected[:, :, ring], atol=1e-6)
        # The centre is the unchanged crop plus its in-map flag.
        plain = MultiRobotCoverageEnv(dict(
            num_maps=1, num_robots=9, num_humans=0, n_rays=4, obs_mode='memory_comm',
            local_coverage_size=5, comm_slots=0, k_teammates=0,
            use_full_memory=False, observation_stack=1, sweep_obs=False))
        base = np.asarray(jax.jit(plain.get_obs)(state))[:, plain.frame_norm_dim:]
        base = base.reshape(env.num_robots, 3, s, s)
        np.testing.assert_array_equal(crop[:, :3, 1:-1, 1:-1], base)

    def test_legacy_ring_follows_heading_and_reads_memory(self):
        env, state, cells = self.make_env('legacy')
        s = env.local_coverage_size
        self.assertEqual(env.patch_dim, 3 * (s + 2) ** 2)
        get_obs = jax.jit(env.get_obs)
        at = lambda st, heading: np.asarray(get_obs(st.replace(
            robot_headings=jnp.full((env.num_robots,), heading))))[
                :, env.frame_norm_dim:].reshape(env.num_robots, 3, s + 2, s + 2)
        east, north = at(state, 0.), at(state, np.pi / 2)
        # The ring averages each robot's memory, not the global coverage grid.
        robot, (col, row) = 4, cells[4]
        mem = np.asarray(state.mem_state[robot])
        known = (mem != 0).astype(float)
        source = ((mem == 3) | (mem == 1)).astype(float)
        h = s // 2
        # Facing north, the crop's +x side is the world's +y side.
        np.testing.assert_allclose(east[robot, 0, h + 1, -1], source[row, col + h + 1:].mean(), atol=1e-6)
        np.testing.assert_allclose(north[robot, 0, h + 1, -1], source[row + h + 1:, col].mean(), atol=1e-6)
        np.testing.assert_allclose(north[robot, 1, h + 1, -1], known[row + h + 1:, col].mean(), atol=1e-6)
        np.testing.assert_allclose(north[robot, 2, h + 1, -1],
                                   (env.grid_h - row - h - 1) / max(env.grid_h, env.grid_w), atol=1e-6)
        # No global knowledge: the global coverage grid never reaches the crop,
        # and a wall counts only once the robot's memory knows the cell.
        hidden = at(state.replace(coverage_grid=1. - state.coverage_grid), 0.)
        np.testing.assert_array_equal(hidden, east)
        blind = at(state.replace(mem_state=jnp.zeros_like(state.mem_state)), 0.)
        inside = blind[:, 2, 1:-1, 1:-1] == 1.
        np.testing.assert_array_equal(blind[:, 0, 1:-1, 1:-1][inside], 0.)
        # A robot in the corner sees nothing beyond the map edge.
        np.testing.assert_array_equal(east[0, :, 0, :],
                                      np.stack([np.ones(s + 2), np.ones(s + 2), np.zeros(s + 2)]))

    def test_critic_reads_every_actor_crop(self):
        for mode in ('legacy', 'memory_comm'):
            with self.subTest(mode=mode):
                env = MultiRobotCoverageEnv(dict(
                    num_maps=1, num_robots=3, num_humans=0, n_rays=4, obs_mode=mode,
                    local_coverage_size=5, comm_slots=0, k_teammates=0, crop_summary=True,
                    critic_crops=True, use_full_memory=False, observation_stack=1,
                    sweep_obs=False))
                state = jax.jit(env.reset)(jax.random.PRNGKey(1))
                obs = np.asarray(env.get_obs(state))
                crops = obs[:, env.frame_norm_dim:env.frame_norm_dim + env.crop_dim]
                grid, vec = env.critic_inputs(env.get_global_state(state))
                vec = np.asarray(vec)
                self.assertEqual(vec.shape, (env.num_robots, env.critic_vec_dim))
                tail = vec[:, -(1 + env.num_robots) * env.crop_dim:]
                np.testing.assert_array_equal(tail[:, :env.crop_dim], crops)
                for robot in range(env.num_robots):
                    np.testing.assert_array_equal(tail[robot, env.crop_dim:], crops.reshape(-1))

    def test_sweep_open_ahead_ignores_unseen_walls(self):
        env = MultiRobotCoverageEnv(dict(
            num_maps=1, num_robots=1, num_humans=0, n_rays=4, obs_mode='memory_comm',
            comm_slots=0, sweep_obs=True, use_full_memory=False, observation_stack=1))
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        state = state.replace(last_discovery=jnp.array([[2, 2]]),
                              sweep_direction=jnp.array([[1, 0]]),
                              lane_return=jnp.zeros((1, 2), jnp.int32),
                              mem_state=jnp.zeros_like(state.mem_state))
        # A wall at the cell ahead counts only once it is in the belief map.
        unseen = state
        seen = state.replace(mem_state=state.mem_state.at[0, 2, 3].set(1))
        self.assertTrue(bool(env._sweep_preference(unseen)[3][0]))
        self.assertFalse(bool(env._sweep_preference(seen)[3][0]))


if __name__ == '__main__':
    unittest.main()
