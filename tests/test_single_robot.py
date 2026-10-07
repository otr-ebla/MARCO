"""Single-robot problem: configurable room, observation without teammates,
and the BCD expert / baseline running on it."""

import os
import unittest

import jax
import numpy as np

from src.envs.bcd_expert import BCDExpert
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

    def test_bcd_tour_visits_every_free_cell_and_acts(self):
        cfg = {**single_robot_env_config(), 'fallback_enabled': False}
        vec_env = VecEnv(2, cfg)
        expert = BCDExpert(vec_env.env, {'target_rule': 'tour'})
        free = vec_env.env.free_mask_np.reshape(vec_env.env.num_maps, -1) > .5
        for m in range(vec_env.env.num_maps):
            tour = expert.tour_np[m][expert.tour_np[m] >= 0]
            self.assertEqual(sorted(tour.tolist()), np.flatnonzero(free[m]).tolist())

        state, *_ = vec_env.reset(jax.random.PRNGKey(0))
        actions, chunk, _ = jax.vmap(expert.act)(state, expert.init_chunks((2,)))
        self.assertEqual(actions.shape, (2, 1, 2))
        self.assertTrue(np.all(np.abs(np.asarray(actions)) <= 1.))
        np.testing.assert_array_equal(np.asarray(chunk), 0)   # one robot owns the whole tour


if __name__ == '__main__':
    unittest.main()
