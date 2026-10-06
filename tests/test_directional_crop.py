import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.coverage_vector_env import MultiRobotCoverageEnv

CONFIG = dict(num_maps=1, wall_cells=0, num_robots=2, n_rays=12, obs_mode='memory_comm',
              comm_slots=1, comm_radius=3.0, crop_mode='directional', max_steps=10)


class DirectionalCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MultiRobotCoverageEnv(CONFIG)
        cls.state = cls.env.reset(jax.random.PRNGKey(0))

    def place(self, cells, known, covered=None):
        """Robots at (row, col) cells with the given memory, shared by both robots."""
        env = self.env
        pos = jnp.array([[(c + .5) * env.cell_size, (r + .5) * env.cell_size] for r, c in cells])
        known = jnp.broadcast_to(known, (len(cells), env.grid_h, env.grid_w))
        covered = None if covered is None else jnp.broadcast_to(covered, known.shape)
        return self.state.replace(robot_positions=pos, mem_state=env.belief_map(0, known, covered),
                                  robot_alive=jnp.ones(len(cells), bool))

    def crop(self, state):
        return np.asarray(self.env._actor_crops(state)).reshape(-1, *self.env.crop_shape)

    def test_shape_and_layout(self):
        self.assertEqual(self.env.crop_shape, (3, 7, 7))
        self.assertEqual(self.env.patch_dim, 3 * 49)
        self.assertFalse(self.env.crop_binary)

    def test_ring_holds_per_channel_means_beyond_each_side(self):
        env = self.env
        h, w = env.grid_h, env.grid_w
        r0, c0 = h // 2, w // 2
        rng = np.random.default_rng(0)
        # A random belief map read straight from memory: 0 unknown, 1 occupied,
        # 2 free, 3 covered.
        mem = jnp.asarray(rng.integers(0, 4, (h, w)), jnp.int8)
        state = self.place([(r0, c0), (r0 + 4, c0 + 1)], jnp.zeros((1, h, w)))
        state = state.replace(mem_state=jnp.broadcast_to(mem, state.mem_state.shape))
        crop = self.crop(state)[0]
        # Brute-force means over the cells assigned to each side.
        maps = np.stack([np.asarray(mem == 1), np.asarray(mem == 3), np.zeros((h, w))]).astype(float)
        maps[2, r0 + 4, c0 + 1] = 1.                              # teammate in comm range
        dr, dc = np.meshgrid(np.arange(h) - r0, np.arange(w) - c0, indexing='ij')
        beyond = np.maximum(abs(dr), abs(dc)) > 2
        vertical = abs(dr) >= abs(dc)
        sides = [vertical & (dr < 0), vertical & (dr > 0), ~vertical & (dc < 0), ~vertical & (dc > 0)]
        ring = [crop[:, 0, 1:-1], crop[:, -1, 1:-1], crop[:, 1:-1, 0], crop[:, 1:-1, -1]]
        for side, values in zip(sides, ring):
            expected = maps[:, side & beyond].mean(axis=1)
            np.testing.assert_allclose(values, np.repeat(expected[:, None], 5, 1), rtol=1e-6)
        # Channels differ: the ring is no longer shared across them.
        self.assertFalse(np.allclose(crop[0, -1, 1:-1], crop[1, -1, 1:-1]))
        self.assertGreater(crop[2, -1, 3], 0.)
        np.testing.assert_allclose(crop[:, [0, 0, -1, -1], [0, -1, 0, -1]], 0.)
        # Ground truth coverage is never read.
        truth = self.crop(state.replace(coverage_grid=jnp.ones_like(state.coverage_grid)))
        np.testing.assert_array_equal(self.crop(state), truth)

    def test_side_beyond_the_floor_reads_as_obstacle(self):
        env = self.env
        known = jnp.ones((1, env.grid_h, env.grid_w))
        crop = self.crop(self.place([(1, 1), (6, 6)], known))[0]
        np.testing.assert_allclose(crop[:, 0, 1:-1], [[1.] * 5, [0.] * 5, [0.] * 5])   # -row side
        np.testing.assert_allclose(crop[:, 1:-1, 0], [[1.] * 5, [0.] * 5, [0.] * 5])   # -col side

    def test_core_channels_and_empty_ring(self):
        env = self.env
        h, w = env.grid_h, env.grid_w
        r0, c0 = h // 2, w // 2
        known = jnp.zeros((h, w)).at[r0 - 2:r0 + 3, c0 - 2:c0 + 3].set(1.)
        covered = jnp.zeros((h, w)).at[r0, c0].set(1.)
        state = self.place([(r0, c0), (r0 + 1, c0 + 2)], known[None], covered[None])
        crop = self.crop(state)
        # Nothing seen or covered beyond the core, and the teammate is inside it.
        ring = np.ones((7, 7), bool)
        ring[1:-1, 1:-1] = False
        np.testing.assert_array_equal(crop[0][:, ring], 0.)
        self.assertEqual(crop[0, 1, 3, 3], 1.)                  # covered under the robot
        self.assertEqual(crop[0, 1].sum(), 1.)
        self.assertEqual(crop[0, 2, 3 + 1, 3 + 2], 1.)          # teammate at +1 row, +2 col
        self.assertEqual(crop[0, 2].sum(), 1.)
        self.assertEqual(crop[1, 2, 3 - 1, 3 - 2], 1.)          # and seen from the teammate
        # Out of comm range the teammate is not marked.
        far = self.place([(r0, c0), (r0 + 1, c0 + 2)], known[None], covered[None])
        far = far.replace(robot_alive=jnp.array([True, False]))
        self.assertEqual(self.crop(far)[0, 2].sum(), 0.)

    def test_floor_edge_reads_as_obstacle(self):
        state = self.place([(0, 0), (5, 5)], jnp.zeros((1, self.env.grid_h, self.env.grid_w)))
        core = self.crop(state)[0, :, 1:-1, 1:-1]
        np.testing.assert_array_equal(core[0, :2, :], 1.)       # rows below the floor
        np.testing.assert_array_equal(core[0, :, :2], 1.)       # columns left of the floor
        np.testing.assert_array_equal(core[1], 0.)

    def test_observation_and_step(self):
        state = self.env.reset(jax.random.PRNGKey(3))
        obs = self.env.get_obs(state)
        self.assertEqual(obs.shape, (2, self.env.obs_dim))
        state, _, _, _ = jax.jit(self.env.step)(state, jnp.zeros((2, 2)))
        self.assertTrue(np.all(np.isfinite(np.asarray(self.env.get_obs(state)))))


if __name__ == '__main__':
    unittest.main()
