import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mrcpp-matplotlib')

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.models.actor_critic import Actor, egocentric_memory, latest_memory_map
from src.envs.coverage_vector_env import MultiRobotCoverageEnv

CONFIG = dict(num_maps=2, num_robots=3, n_rays=70, obs_mode='memory_comm', use_full_memory=True,
              memory_map_obs=True,  # legacy layout read by SpatialMemoryRead
              history_cell='previous_visit', reward_mode='sequential', max_steps=200)


class EgocentricMemoryTests(unittest.TestCase):
    def test_robot_cell_is_the_centre_and_offsets_are_preserved(self):
        h, w = 4, 6
        maps = np.zeros((1, 5, h, w), np.float32)
        maps[0, 2] = 1.                 # all known, free, uncovered
        maps[0, 3, 1, 2] = 1.           # robot at row 1, col 2
        maps[0, 1, 3, 5] = 1.           # one covered cell, 2 rows up and 3 columns right
        ego = np.asarray(egocentric_memory(jnp.asarray(maps)))
        self.assertEqual(ego.shape, (1, 2 * h - 1, 2 * w - 1, 6))
        centre = (h - 1, w - 1)
        self.assertEqual(ego[0, centre[0], centre[1], 3], 1.)            # uncovered under the robot
        self.assertEqual(ego[0, centre[0] + 2, centre[1] + 3, 1], 1.)    # covered, same offset
        self.assertEqual(ego[0, centre[0] + 2, centre[1] + 3, 3], 0.)
        # Row -2 relative to the robot is beyond the floor: a known wall, outside.
        np.testing.assert_array_equal(ego[0, centre[0] - 2, centre[1]], [1., 0., 1., 0., 0., 0.])
        self.assertEqual(ego[..., 5].sum(), h * w)


class MemoryActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MultiRobotCoverageEnv(CONFIG)

    def actor(self, **kwargs):
        env = self.env
        return Actor(recurrent=True, vec_dim=env.obs_vec_dim, n_rays=env.n_rays,
                     tail_dim=env.patch_dim, memory_map_shape=env.memory_map_shape,
                     observation_stack=env.observation_stack, hidden_size=64,
                     map_encoder='spatial_memory', map_embed=32, **kwargs)

    def test_latest_map_matches_the_environment_map(self):
        env = MultiRobotCoverageEnv({**CONFIG, 'observation_stack': 3})
        state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
        obs = env.get_obs(state)
        maps = latest_memory_map(obs, env.obs_vec_dim, env.n_rays, env.patch_dim,
                                 env.memory_map_shape, env.observation_stack)
        frame = env._get_frame_obs(state)
        expected = frame[:, -int(np.prod(env.memory_map_shape)):].reshape(-1, *env.memory_map_shape)
        np.testing.assert_array_equal(maps, expected)

    def test_recurrent_spatial_actor_runs_and_reads_the_map(self):
        env = self.env
        actor = self.actor()
        state = env.reset(jax.random.PRNGKey(1), jnp.int32(0))
        obs = env.get_obs(state)
        memory = jnp.zeros((env.num_robots, 64))
        params = actor.init(jax.random.PRNGKey(2), obs, memory)
        self.assertIn('spatial_memory', params['params'])
        mean, log_std, memory = actor.apply(params, obs, memory)
        self.assertEqual(mean.shape, (env.num_robots, 2))
        self.assertEqual(memory.shape, (env.num_robots, 64))
        # Marking far-away cells covered must change the action: the map is read.
        map_dim = int(np.prod(env.memory_map_shape))
        maps = obs[:, -map_dim:].reshape(-1, *env.memory_map_shape)
        changed = maps.at[:, 1].set(1. - maps[:, 1]).reshape(env.num_robots, -1)
        mean2, _, _ = actor.apply(params, obs.at[:, -map_dim:].set(changed), jnp.zeros_like(memory))
        mean1, _, _ = actor.apply(params, obs, jnp.zeros_like(memory))
        self.assertGreater(float(jnp.abs(mean2 - mean1).max()), 0.)


if __name__ == '__main__':
    unittest.main()


class SequenceReplayTests(unittest.TestCase):
    def test_windows_are_ordered_and_start_from_the_stored_memory(self):
        from src.pretrain_bc import ReplayBuffer, sample_windows
        t, e, n, d, hidden, length = 8, 3, 2, 4, 5, 4
        template = {'obs': jax.ShapeDtypeStruct((t, e, n, d), jnp.float32),
                    'label': jax.ShapeDtypeStruct((t, e, n, 2), jnp.float32),
                    'weight': jax.ShapeDtypeStruct((t, e, n), jnp.float32),
                    'change': jax.ShapeDtypeStruct((t, e, n), jnp.float32),
                    'memory': jax.ShapeDtypeStruct((t, e * n, hidden), jnp.float32)}
        buffer = ReplayBuffer(2, template, d, binary_tail=False, sequence_length=length)
        tt, ee, nn = np.meshgrid(np.arange(t), np.arange(e), np.arange(n), indexing='ij')
        for slot in range(2):
            code = 1000 * slot + 100 * tt + 10 * ee + nn                       # (T, E, N)
            obs = np.broadcast_to(code[..., None], (t, e, n, d)).astype(np.float32)
            memory = np.broadcast_to(code.reshape(t, e * n)[..., None], (t, e * n, hidden))
            buffer.add({'obs': jnp.asarray(obs), 'label': jnp.zeros((t, e, n, 2)),
                        'weight': jnp.ones((t, e, n)), 'change': jnp.zeros((t, e, n)),
                        'done': jnp.asarray((tt[..., 0] == 5) & (ee[..., 0] == 1), jnp.float32),
                        'memory': jnp.asarray(memory, jnp.float32)})
        obs, label, weight, change, done, memory = sample_windows(
            buffer.data, 2, jax.random.PRNGKey(0), (t, e, n), length, 6, buffer.observations)
        self.assertEqual(obs.shape, (length, 6, n, d))
        code = np.asarray(obs[..., 0]).astype(int)
        slot, step, env, robot = code // 1000, code // 100 % 10, code // 10 % 10, code % 10
        np.testing.assert_array_equal(np.diff(step, axis=0), 1)               # consecutive steps
        self.assertTrue(np.all(step[0] % length == 0))                        # window boundary
        for x in (slot, env):
            self.assertTrue(np.all(x == x[:1]))                               # one rollout, one env
        np.testing.assert_array_equal(robot, np.broadcast_to(np.arange(n), robot.shape))
        np.testing.assert_array_equal(np.asarray(memory)[:, 0], code[0].reshape(-1))
        np.testing.assert_array_equal(done, (step[..., 0] == 5) & (env[..., 0] == 1))
