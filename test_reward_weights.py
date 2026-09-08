from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.envs.coverage_vector_env import E2E_REWARD_DEFAULTS, MultiRobotCoverageEnv
from src.utils.config_parser import load_config


CONFIG = Path(__file__).parent / "config" / "mappo_baseline.yaml"


def test_baseline_reward_weights_are_loaded_by_the_environment():
    config = load_config(CONFIG)
    env_config = config["env"]

    assert env_config["num_maps"] == 1
    assert env_config["alpha"] == 10.0
    assert env_config["coverage_reward_growth"] == 2.0
    assert env_config["beta"] == 0.5
    assert env_config["kappa"] == 50.0
    assert env_config["wall_kappa"] == 30.0
    assert env_config["human_kappa"] == 150.0
    assert env_config["tau"] == 0.05
    assert env_config["psi"] == 2.0
    assert env_config["bosco_gamma"] == 10.0
    assert env_config["bosco_distance_penalty"] == 1.0
    assert env_config["terminate_on_collision"] is False
    assert env_config["safe_dist_factor"] == 3.0
    assert env_config["velocity_cost"] == 0.05
    assert env_config["angular_cost"] == 0.01
    assert env_config["action_smoothness_cost"] == 0.01
    assert "bosco_gamma" not in config["wandb"]


def test_end_to_end_reward_prioritises_late_discoveries_and_completion():
    config = load_config(CONFIG)

    assert config["e2e_num_maps"] == 64
    assert E2E_REWARD_DEFAULTS["coverage_reward_growth"] == 2.0
    assert E2E_REWARD_DEFAULTS["completion_bonus"] == 200.0
    assert E2E_REWARD_DEFAULTS["beta"] == 2.0
    assert config["e2e_reward"]["coverage_reward_growth"] == 2.0
    assert config["e2e_reward"]["completion_bonus"] == 200.0
    assert config["e2e_reward"]["beta"] == 2.0


def test_redundancy_may_be_penalised_more_than_elapsed_time():
    env = MultiRobotCoverageEnv({"beta": 0.5, "tau": 0.05, "num_maps": 1})

    assert env.beta > env.tau


def test_negative_reward_weight_is_rejected():
    with pytest.raises(ValueError, match="reward weights must be non-negative: beta"):
        MultiRobotCoverageEnv({"beta": -0.1, "num_maps": 1})


def test_bosco_distance_penalty_grows_quadratically():
    env = MultiRobotCoverageEnv({
        "num_robots": 1,
        "num_maps": 1,
        "alpha": 0.0,
        "beta": 0.0,
        "tau": 0.0,
        "bosco_gamma": 0.0,
        "bosco_distance_penalty": 1.0,
        "velocity_cost": 0.0,
        "angular_cost": 0.0,
        "action_smoothness_cost": 0.0,
        "room_completion_threshold": 2.0,
        "room_completion_bonus": 0.0,
        "completion_bonus": 0.0,
    })
    state = env.reset(jax.random.PRNGKey(7))
    cols, rows = env._pos_to_cell(state.robot_positions)
    state = state.replace(
        coverage_grid=state.coverage_grid.at[rows[0], cols[0]].set(1.0),
        cell_assignments=state.cell_assignments.at[0, rows[0], cols[0]].set(1.0),
    )
    action = jnp.asarray([[-1.0, 0.0]], jnp.float32)
    near_target = state.robot_positions + jnp.asarray([[env.cell_size, 0.0]])
    far_target = state.robot_positions + jnp.asarray([[2.0 * env.cell_size, 0.0]])

    _, near_reward, _, _ = env.step(
        state.replace(bosco_targets=near_target), action
    )
    _, far_reward, _, _ = env.step(
        state.replace(bosco_targets=far_target), action
    )

    assert near_reward[0] == pytest.approx(-1.0)
    assert far_reward[0] == pytest.approx(-4.0)


def test_actor_observation_contains_binary_local_coverage_patch():
    env = MultiRobotCoverageEnv({
        "num_robots": 1, "num_maps": 1, "local_coverage_size": 5,
    })
    state = env.reset(jax.random.PRNGKey(4))
    obs = env.get_obs(state)

    assert env.patch_dim == 25
    assert env.obs_dim == env.norm_dim + 25
    assert obs.shape == (1, env.obs_dim)
    assert jnp.all((obs[:, -25:] == 0.0) | (obs[:, -25:] == 1.0))


def test_even_local_coverage_patch_is_rejected():
    with pytest.raises(ValueError, match="positive odd integer"):
        MultiRobotCoverageEnv({"local_coverage_size": 4, "num_maps": 1})


def test_assigned_new_cell_reward_is_four_times_other_new_cell_reward():
    env = MultiRobotCoverageEnv({
        "num_robots": 1,
        "num_maps": 1,
        "alpha": 8.0,
        "beta": 0.0,
        "tau": 0.0,
        "bosco_gamma": 0.0,
        "bosco_distance_penalty": 0.0,
        "room_completion_threshold": 2.0,
        "room_completion_bonus": 0.0,
        "completion_bonus": 0.0,
        "velocity_cost": 0.0,
        "angular_cost": 0.0,
        "action_smoothness_cost": 0.0,
    })
    state = env.reset(jax.random.PRNGKey(0)).replace(
        robot_headings=jnp.zeros((1,), jnp.float32)
    )
    cols, rows = env._pos_to_cell(state.robot_positions)
    assignments = state.cell_assignments.at[0, rows[0], cols[0]].set(1.0)
    action = jnp.asarray([[1.0, 0.0]], jnp.float32)

    _, assigned_reward, _, _ = env.step(
        state.replace(cell_assignments=assignments), action
    )
    _, other_reward, _, _ = env.step(state, action)

    assert assigned_reward[0] == pytest.approx(8.0)
    assert other_reward[0] == pytest.approx(2.0)


def test_new_cell_reward_increases_with_existing_coverage():
    env = MultiRobotCoverageEnv({
        "num_robots": 1,
        "num_maps": 1,
        "alpha": 8.0,
        "coverage_reward_growth": 2.0,
        "beta": 0.0,
        "tau": 0.0,
        "bosco_gamma": 0.0,
        "room_completion_threshold": 2.0,
        "room_completion_bonus": 0.0,
        "completion_bonus": 0.0,
    })
    state = env.reset(jax.random.PRNGKey(2)).replace(
        robot_headings=jnp.zeros((1,), jnp.float32)
    )
    cols, rows = env._pos_to_cell(state.robot_positions)
    assignments = state.cell_assignments.at[0, rows[0], cols[0]].set(1.0)
    state = state.replace(cell_assignments=assignments)
    action = jnp.asarray([[1.0, 0.0]], jnp.float32)

    _, early_reward, _, _ = env.step(state, action)
    # Keep half of the free cells covered while leaving the destination cell free.
    free = env.wall_grids[state.map_id] == 0
    free_rank = jnp.cumsum(free.reshape(-1)).reshape(free.shape)
    half_covered = (
        free & (free_rank <= env.free_totals[state.map_id] // 2)
    ).astype(jnp.float32)
    half_covered = half_covered.at[rows[0], cols[0]].set(0.0)
    _, late_reward, _, _ = env.step(
        state.replace(coverage_grid=half_covered), action
    )

    assert late_reward[0] > early_reward[0]


def test_local_coverage_reward_applies_late_discovery_growth():
    env = MultiRobotCoverageEnv({
        **E2E_REWARD_DEFAULTS,
        "num_robots": 1,
        "num_maps": 1,
        "actor_bosco_guidance": False,
        "bosco_reward_guidance": False,
        "alpha": 8.0,
        "beta": 0.0,
        "tau": 0.0,
        "completion_bonus": 0.0,
    })
    state = env.reset(jax.random.PRNGKey(21)).replace(
        robot_headings=jnp.zeros((1,), jnp.float32)
    )
    cols, rows = env._pos_to_cell(state.robot_positions)
    action = jnp.asarray([[1.0, 0.0]], jnp.float32)

    _, early_reward, _, _ = env.step(state, action)
    free = env.free_masks[state.map_id] > 0
    free_rank = jnp.cumsum(free.reshape(-1)).reshape(free.shape)
    half_covered = (
        free & (free_rank <= env.free_totals[state.map_id] // 2)
    ).astype(jnp.float32)
    half_covered = half_covered.at[rows[0], cols[0]].set(0.0)
    _, late_reward, _, _ = env.step(
        state.replace(coverage_grid=half_covered), action
    )

    assert late_reward[0] > early_reward[0] * 1.9


def test_global_state_exposes_agent_cell_assignments():
    env = MultiRobotCoverageEnv({"num_robots": 2, "num_maps": 1})
    state = env.reset(jax.random.PRNGKey(1))
    assignments = state.cell_assignments.at[0, 2, 3].set(1.0)
    grid, _ = env.critic_inputs(env.get_global_state(
        state.replace(cell_assignments=assignments)
    ))

    assert env.critic_channels == 6
    assert grid.shape == (2, 6, env.grid_h, env.grid_w)
    assert grid[0, 4, 2, 3] == pytest.approx(1.0)
    assert grid[1, 5, 2, 3] == pytest.approx(1.0)
