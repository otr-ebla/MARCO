import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.models.actor_critic import Actor
from src.algorithms.mappo import recurrent_actor_sequence, compute_gae, Transition


class RecurrentActorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor = Actor(vec_dim=2, n_rays=8, tail_dim=9, hidden_size=16,
                          lidar_embed=8, recurrent=True)
        cls.obs = jax.random.normal(jax.random.PRNGKey(0), (4, 2, 2, 19))
        cls.initial = jnp.zeros((4, 16))
        cls.params = cls.actor.init(jax.random.PRNGKey(1), cls.obs[0].reshape(4, 19))

    def test_sequence_matches_step_execution_and_resets_only_done_env(self):
        done = jnp.array([[True, False], [False, False], [False, True], [False, False]])
        memory, (mean, _) = recurrent_actor_sequence(
            self.actor, self.params, self.obs, self.initial, done)
        h = self.initial
        means = []
        for t in range(4):
            mu, _, h = self.actor.apply(self.params, self.obs[t].reshape(4, 19), h)
            means.append(mu)
            h = jnp.where(jnp.repeat(done[t], 2)[:, None], 0., h)
        np.testing.assert_allclose(memory, h, atol=1e-6)
        np.testing.assert_allclose(mean, jnp.stack(means).reshape(-1, 2), atol=1e-6)
        fresh, _, _ = self.actor.apply(self.params, self.obs[1, 0], jnp.zeros((2, 16)))
        np.testing.assert_allclose(mean.reshape(4, 2, 2, 2)[1, 0], fresh, atol=1e-6)

    def test_history_changes_action_for_same_current_observation(self):
        observation = self.obs[0].reshape(4, 19)
        _, _, history = self.actor.apply(self.params, observation, self.initial)
        fresh, _, _ = self.actor.apply(self.params, observation, self.initial)
        remembered, _, _ = self.actor.apply(self.params, observation, history)
        self.assertGreater(float(jnp.max(jnp.abs(fresh-remembered))), 1e-7)

    def test_sequence_backpropagates_through_gru(self):
        def loss(params):
            _, (mean, _) = recurrent_actor_sequence(
                self.actor, params, self.obs, self.initial, jnp.zeros((4, 2)))
            return jnp.sum(mean[-4:] ** 2)
        grads = jax.grad(loss)(self.params)['params']['memory']
        leaves = jax.tree_util.tree_leaves(grads)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in leaves))
        self.assertGreater(sum(float(jnp.sum(jnp.abs(x))) for x in leaves), 0.)

    def test_chunk_boundary_retains_memory(self):
        done = jnp.zeros((4, 2))
        final, (whole, _) = recurrent_actor_sequence(self.actor, self.params, self.obs, self.initial, done)
        h, (first, _) = recurrent_actor_sequence(self.actor, self.params, self.obs[:2], self.initial, done[:2])
        h, (second, _) = recurrent_actor_sequence(self.actor, self.params, self.obs[2:], h, done[2:])
        np.testing.assert_allclose(jnp.concatenate([first, second]), whole, atol=1e-6)
        np.testing.assert_allclose(h, final, atol=1e-6)

    def test_recurrent_returns_do_not_cross_timeout(self):
        traj = Transition(**{name: jnp.zeros((2, 1, 1)) for name in Transition._fields})
        traj = traj._replace(reward=jnp.array([[[1.]], [[100.]]]), done=jnp.ones((2, 1)),
                             term=jnp.zeros((2, 1)), memory=self.initial)
        _, returns = compute_gae(traj, jnp.array([[1000.]]), .99, .95)
        np.testing.assert_allclose(returns, traj.reward)


class RecurrentRolloutTest(unittest.TestCase):
    def test_ppo_replay_reproduces_rollout_likelihoods_across_resets(self):
        from src.algorithms.bosco_guide import make_guides
        from src.algorithms.mappo import MAPPO, _tanh_normal_log_prob
        from src.envs.coverage_vector_env import E2E_REWARD_DEFAULTS
        from src.envs.vec_env import VecEnv
        from src.models.actor_critic import Critic
        from src.train_bosco import GuidedRollout

        env = VecEnv(2, {**E2E_REWARD_DEFAULTS, 'num_maps': 1, 'num_robots': 2,
                         'max_steps': 3, 'n_rays': 8, 'actor_bosco_guidance': False,
                         'bosco_reward_guidance': False})
        actor = Actor(vec_dim=env.env.obs_vec_dim, n_rays=8,
                      tail_dim=env.env.patch_dim, hidden_size=16,
                      lidar_embed=8, recurrent=True)
        mappo = MAPPO(actor, Critic(hidden_size=16, map_embed=8), env, {'n_epochs': 1})
        a, c = mappo.create_train_states(jax.random.PRNGKey(2))
        rollout = GuidedRollout(mappo, env, make_guides(env.env, 2), 0.)
        carry = rollout.start(jax.random.PRNGKey(3))
        carry = carry._replace(env_state=carry.env_state.replace(step_count=jnp.array([0, 1])))
        final, traj, _, _ = rollout.run(a.params, c.params, carry, 4, jax.random.PRNGKey(4))
        memory, (mean, log_std) = recurrent_actor_sequence(actor, a.params, traj.obs, traj.memory[0], traj.done)
        log_prob = _tanh_normal_log_prob(traj.z.reshape(-1, 2), mean,
                                         jnp.exp(log_std), traj.action.reshape(-1, 2))
        np.testing.assert_allclose(log_prob, traj.log_prob.reshape(-1), atol=2e-5)
        np.testing.assert_allclose(memory, final.memory, atol=1e-6)
        self.assertFalse(np.array_equal(np.asarray(traj.done[:, 0]), np.asarray(traj.done[:, 1])))
        for t in range(3):
            for e in range(2):
                if bool(traj.done[t, e]):
                    np.testing.assert_array_equal(traj.memory[t+1, e*2:(e+1)*2], 0.)
