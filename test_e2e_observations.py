import jax
import jax.numpy as jnp
import numpy as np

from src.envs.coverage_vector_env import MultiRobotCoverageEnv


def test_e2e_actor_and_rewards_ignore_planner_but_critic_retains_it():
    env = MultiRobotCoverageEnv({
        'num_maps': 1, 'actor_bosco_guidance': False,
        'bosco_reward_guidance': False,
    })
    state = env.reset(jax.random.PRNGKey(7))
    changed = state.replace(
        bosco_targets=state.bosco_targets + 2.0,
        cell_assignments=jnp.ones_like(state.cell_assignments),
    )
    obs = jax.jit(env.get_obs)(state)
    np.testing.assert_array_equal(obs, jax.jit(env.get_obs)(changed))
    assert obs.shape == (env.num_robots, env.obs_dim)
    assert env.obs_vec_dim == 2 + 2 * env.k_teammates
    assert not np.array_equal(env.get_global_state(state).bosco_targets,
                              env.get_global_state(changed).bosco_targets)
    action = jnp.zeros((env.num_robots, 2))
    result = jax.jit(env.step)(state, action)
    changed_result = jax.jit(env.step)(changed, action)
    np.testing.assert_array_equal(result[1], changed_result[1])


def test_guided_observation_keeps_target_channels():
    env = MultiRobotCoverageEnv({'num_maps': 1})
    state = env.reset(jax.random.PRNGKey(7))
    changed = state.replace(bosco_targets=state.bosco_targets + 2.0)
    assert not np.array_equal(env.get_obs(state), env.get_obs(changed))
    assert env.obs_vec_dim == 4 + 2 * env.k_teammates
