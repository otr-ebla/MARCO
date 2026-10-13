"""Optimal-path teacher: fixed spawn cells, the offline plan bank and the
tracker that follows it in the environment."""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.envs.bcd_expert import expert_steps, first_episode_summary
from src.envs.optimal_coverage import OPTIMAL, validate_path
from src.envs.optimal_expert import OptimalExpert, OptimalPlanBank, build_teacher
from src.envs.vec_env import VecEnv
from src.train_marl import build_env_config
from src.utils.config_parser import load_config

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'single_robot.yaml')


def small_env_config(spawn_cells=3):
    """Single-robot config on 4 m x 3 m rooms (8 x 6 cells): plans take milliseconds."""
    config = load_config(CONFIG)
    config['env'].update(map_width=4.0, map_height=3.0, map_depth=1, spawn_cells=spawn_cells,
                         max_steps=2000)
    env_cfg, _ = build_env_config(config, num_maps=3)
    env_cfg['fallback_enabled'] = False
    return env_cfg


class OptimalExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory()
        cls.vec_env = VecEnv(6, small_env_config())
        cls.env = cls.vec_env.env
        cls.plan_cfg = {'time_limit': 60, 'workers': 2, 'cache_dir': cls.cache.name}
        cls.bank = OptimalPlanBank(cls.env, cls.plan_cfg, verbose=False)
        cls.expert = OptimalExpert(cls.env, cls.bank)

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def test_robots_spawn_only_on_the_fixed_cells(self):
        state, *_ = self.vec_env.reset(jax.random.PRNGKey(3))
        for _ in range(5):
            state, *_ = self.vec_env.reset(state.key[0])
            cells = np.asarray(state.robot_positions[:, 0, ::-1] // self.env.cell_size, int)
            for m, cell in zip(np.asarray(state.map_id), cells):
                self.assertIn(tuple(cell), {tuple(c) for c in self.env.spawn_cells_np[m]})

    def test_plans_are_valid_optimal_paths_from_every_spawn_cell(self):
        w = self.env.grid_w
        for m in range(self.env.num_maps):
            grid = self.env.free_mask_np[m] > .5
            for k, start in enumerate(self.env.spawn_cells_np[m]):
                path = self.bank.path[m, k]
                path = path[path >= 0]
                cells = list(zip(path // w, path % w))
                self.assertTrue(validate_path(cells, grid, tuple(start)).valid)
                self.assertEqual(self.bank.status[m, k], OPTIMAL)
                order = self.bank.order[m, k]
                order = order[order >= 0]
                self.assertEqual(order[0], self.bank.start[m, k])
                np.testing.assert_array_equal(np.sort(order), np.flatnonzero(grid.ravel()))

    def test_bank_is_cached_on_disk(self):
        again = OptimalPlanBank(self.env, self.plan_cfg, verbose=False)
        self.assertEqual(again.path_file, self.bank.path_file)
        np.testing.assert_array_equal(again.path, self.bank.path)

    def test_episode_start_picks_the_plan_of_the_spawn_cell(self):
        state, *_ = self.vec_env.reset(jax.random.PRNGKey(0))
        chunk = jax.vmap(self.expert.assign)(state)[:, 0]
        cells = np.asarray(state.robot_positions[:, 0, 1] // self.env.cell_size, int) * self.env.grid_w \
            + np.asarray(state.robot_positions[:, 0, 0] // self.env.cell_size, int)
        np.testing.assert_array_equal(self.bank.start[np.asarray(state.map_id), np.asarray(chunk)],
                                      cells)
        # The first target is the second cell of the plan.
        target, _ = jax.vmap(self.expert.targets)(state, chunk[:, None])
        np.testing.assert_array_equal(
            np.asarray(target[:, 0]), self.bank.order[np.asarray(state.map_id), np.asarray(chunk), 1])

    def test_expert_covers_every_map_without_contacts(self):
        state, *_ = self.vec_env.reset(jax.random.PRNGKey(1), jnp.arange(6) % self.env.num_maps)
        run = expert_steps(self.expert, self.vec_env, self.env.max_steps)
        _, stats = run((state, self.expert.init_chunks((self.vec_env.E,))))
        summary = first_episode_summary(jax.device_get(stats), self.env.max_steps)
        self.assertTrue(np.all(summary['complete']))
        self.assertEqual(float(summary['wall'].sum()), 0.0)

    def test_build_teacher_needs_spawn_cells(self):
        env = VecEnv(1, small_env_config(spawn_cells=0)).env
        with self.assertRaises(ValueError):
            build_teacher(env, {'teacher': 'optimal', 'optimal': self.plan_cfg})
        self.assertEqual(build_teacher(env, {'teacher': 'bcd'}).target_rule, 'local')


if __name__ == '__main__':
    unittest.main()
