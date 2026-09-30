#!/usr/bin/env python3
"""Train an end-to-end multi-robot coverage policy with MAPPO or IPPO (CTDE).

Actors act only on their own observation and command velocities; every
environment samples a layout independently from a procedural map bank.

    python -m src.train_marl --policy-mode end-to-end --save-dir checkpoints/e2e

`--policy-mode end-to-end-memory` adds a GRU to the local actor with the same
observation width and reward as end-to-end. Memory resets at episode boundaries;
PPO backpropagates ordered sequences over each rollout.

Train comparison policies from scratch; actor architectures differ between modes.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.ippo import IPPO
from src.algorithms.mappo import (
    MAPPO,
    RunningMeanStd,
    Transition,
    _tanh_normal_log_prob,
    compute_gae,
    rms_init,
    rms_normalize,
    rms_update,
)
from src.envs.coverage_vector_env import E2E_REWARD_DEFAULTS
from src.envs.vec_env import VecEnv
from src.models.actor_critic import Actor, Critic, LocalCritic
from src.train_simple import (
    _COLLISION,
    _SUCCESS,
    _TIMEOUT,
    _WINDOW,
    init_wandb,
    linear_lr_decay,
    wandb,
)
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device
from src.utils.human_curriculum import ghost_robot_probability


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, update: int, actor_state, critic_state, rms,
                    tail_dim: int,
                    policy_mode: str = "end-to-end", reward_weights: dict | None = None,
                    obs_config: dict | None = None, algo: str = "mappo",
                    actor_config: dict | None = None) -> None:
    """Save parameters, normalizer and the actor/reward regime for evaluation."""
    payload = {
        'update':        update,
        'actor_params':  jax.device_get(actor_state.params),
        'critic_params': jax.device_get(critic_state.params),
        'actor_opt':     jax.device_get(actor_state.opt_state),
        'critic_opt':    jax.device_get(critic_state.opt_state),
        'obs_rms':       jax.device_get(rms),
        'tail_dim':      int(tail_dim),
        'policy_mode': policy_mode,
        'actor_recurrent': policy_mode == 'end-to-end-memory',
        'reward_weights': dict(reward_weights or {}),
        # Env keys that fix the actor's input layout, restored by evaluators.
        'obs_config': dict(obs_config or {'obs_mode': 'legacy'}),
        'algo': algo,
        'actor_config': dict(actor_config or {}),
    }
    # Publish only a fully-written pickle so evaluators can safely load it while
    # training continues. os.replace is atomic when source and target share a
    # directory/filesystem.
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def _trunk_key(params, in_dim: int) -> str | None:
    """Name of the actor layer whose kernel takes `in_dim` inputs, if unique."""
    hits = [k for k, v in params['params'].items()
            if isinstance(v, dict) and 'kernel' in v
            and np.ndim(v['kernel']) == 2 and np.shape(v['kernel'])[0] == in_dim]
    return hits[0] if len(hits) == 1 else None


def _tree_shapes_match(a, b) -> bool:
    a_leaves, a_tree = jax.tree_util.tree_flatten(a)
    b_leaves, b_tree = jax.tree_util.tree_flatten(b)
    return (a_tree == b_tree and len(a_leaves) == len(b_leaves)
            and all(np.shape(x) == np.shape(y)
                    for x, y in zip(a_leaves, b_leaves)))


def load_checkpoint(path: str, actor_state, critic_state, device,
                    _trunk_in: int) -> tuple[object, object, RunningMeanStd, int]:
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)

    params = ckpt['actor_params']
    if not _tree_shapes_match(actor_state.params, params):
        raise SystemExit(
            f"'{path}' has an incompatible actor architecture. Use a checkpoint "
            "from the same policy mode and model configuration, or train from scratch."
        )

    params = jax.device_put(params, device)
    actor_state = actor_state.replace(
        params=params,
        opt_state=jax.device_put(ckpt['actor_opt'], device),
    )
    if _tree_shapes_match(critic_state.params, ckpt['critic_params']):
        critic_state = critic_state.replace(
            params=jax.device_put(ckpt['critic_params'], device),
            opt_state=jax.device_put(ckpt['critic_opt'], device),
        )
    else:
        print(
            "Checkpoint critic shape differs from the current environment "
            "(for example, the number of humans changed): keeping a freshly "
            "initialised critic while transferring the actor."
        )
    rms = RunningMeanStd(*jax.device_put(tuple(ckpt['obs_rms']), device))
    return actor_state, critic_state, rms, int(ckpt['update'])


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

class RolloutCarry(NamedTuple):
    """Rollout state threaded between updates."""

    env_state: object
    obs:       jax.Array      # actor inputs for the selected policy mode
    gstate:    object
    rms:       object
    episode_stats: object
    smoothed_col_rate: jax.Array
    smoothed_coverage_rate: jax.Array
    memory: object = None


class DeviceEpisodeStats(NamedTuple):
    """Episode accumulators and a fixed-size recent-history ring on device."""

    reward:    jax.Array
    length:    jax.Array
    wall:      jax.Array
    robot:     jax.Array
    human:     jax.Array
    recent_reward:   jax.Array
    recent_coverage: jax.Array
    recent_recoverage: jax.Array
    recent_length:   jax.Array
    recent_wall:     jax.Array
    recent_robot:    jax.Array
    recent_human:    jax.Array
    recent_outcome:  jax.Array
    ring_pos:   jax.Array
    ring_count: jax.Array
    total_count: jax.Array


def _episode_stats_init(num_envs: int) -> DeviceEpisodeStats:
    zeros_e = jnp.zeros((num_envs,), jnp.float32)
    zeros_w = jnp.zeros((_WINDOW,), jnp.float32)
    return DeviceEpisodeStats(
        reward=zeros_e,
        length=jnp.zeros((num_envs,), jnp.int32),
        wall=zeros_e,
        robot=zeros_e,
        human=zeros_e,
        recent_reward=zeros_w,
        recent_coverage=zeros_w,
        recent_recoverage=jnp.ones((_WINDOW,), jnp.float32),
        recent_length=zeros_w,
        recent_wall=zeros_w,
        recent_robot=zeros_w,
        recent_human=zeros_w,
        recent_outcome=jnp.zeros((_WINDOW,), jnp.int32),
        ring_pos=jnp.int32(0),
        ring_count=jnp.int32(0),
        total_count=jnp.int32(0),
    )


def _episode_stats_step(stats: DeviceEpisodeStats, trans: Transition,
                        reward_scale: float, recoverage: jax.Array) -> DeviceEpisodeStats:
    """Consume one vectorised transition without copying trajectory data to host."""
    stats = stats._replace(
        reward=stats.reward + jnp.mean(trans.reward, axis=-1) / reward_scale,
        length=stats.length + 1,
        wall=stats.wall + trans.wall_hit,
        robot=stats.robot + trans.robot_hit,
        human=stats.human + trans.human_hit,
    )

    def one_env(s, xs):
        done, coverage, recov, complete, timeout, env_idx = xs

        def finish(x):
            pos = x.ring_pos
            outcome = jnp.where(
                complete > 0.5, jnp.int32(_SUCCESS),
                jnp.where(timeout > 0.5, jnp.int32(_TIMEOUT), jnp.int32(_COLLISION)),
            )
            x = x._replace(
                recent_reward=x.recent_reward.at[pos].set(x.reward[env_idx]),
                recent_coverage=x.recent_coverage.at[pos].set(coverage),
                recent_recoverage=x.recent_recoverage.at[pos].set(recov),
                recent_length=x.recent_length.at[pos].set(x.length[env_idx]),
                recent_wall=x.recent_wall.at[pos].set(x.wall[env_idx]),
                recent_robot=x.recent_robot.at[pos].set(x.robot[env_idx]),
                recent_human=x.recent_human.at[pos].set(x.human[env_idx]),
                recent_outcome=x.recent_outcome.at[pos].set(outcome),
                reward=x.reward.at[env_idx].set(0.0),
                length=x.length.at[env_idx].set(0),
                wall=x.wall.at[env_idx].set(0.0),
                robot=x.robot.at[env_idx].set(0.0),
                human=x.human.at[env_idx].set(0.0),
                ring_pos=(pos + 1) % _WINDOW,
                ring_count=jnp.minimum(x.ring_count + 1, _WINDOW),
                total_count=x.total_count + 1,
            )
            return x

        return jax.lax.cond(done > 0.5, finish, lambda x: x, s), None

    env_idx = jnp.arange(trans.done.shape[0], dtype=jnp.int32)
    stats, _ = jax.lax.scan(
        one_env, stats,
        (trans.done, trans.coverage, recoverage, trans.complete, trans.timeout, env_idx),
    )
    return stats


class Rollout:
    def __init__(self, mappo: MAPPO, vec_env: VecEnv):
        self.mappo = mappo
        self.env = vec_env
        e, n = vec_env.E, vec_env.num_robots

        @jax.jit
        def act(actor_params, critic_params, rms, obs, gstate, key, memory):
            rms = rms_update(rms, obs.reshape(e * n, -1))
            obs_n = rms_normalize(rms, obs)

            if mappo.actor.recurrent:
                mean, log_std, memory = mappo.actor.apply(
                    actor_params, obs_n.reshape(e * n, -1), memory)
            else:
                mean, log_std = mappo.actor.apply(actor_params, obs_n.reshape(e * n, -1))
            std = jnp.exp(log_std)
            z = mean + std * jax.random.normal(key, mean.shape)
            action = jnp.tanh(z)
            log_prob = _tanh_normal_log_prob(z, mean, std, action).reshape(e, n)
            value = mappo._values(critic_params, obs_n, gstate)
            return (obs_n, action.reshape(e, n, -1), z.reshape(e, n, -1),
                    log_prob, value, rms, memory)

        self._act = act
        # Bootstrap value on the post-rollout observation, normalised with the
        # final statistics (IPPO's critic reads it; MAPPO's ignores it).
        self._value_fn = jax.jit(
            lambda params, rms, obs, gstate: mappo._values(
                params, rms_normalize(rms, obs), gstate)
        )
        
        def jitted_run(actor_params, critic_params, carry, keys):
            def scan_step(c, k):
                (state, obs, gstate, rms, episode_stats,
                 smoothed_col_rate, smoothed_coverage_rate, memory) = c
                
                obs_n, action, z, log_prob, value, rms, next_memory = act(
                    actor_params, critic_params, rms, obs, gstate, k, memory
                )
                (next_state, next_obs, reward, term, done, info,
                 next_gstate) = vec_env.step(state, action)

                if mappo.actor.recurrent:
                    next_memory = jnp.where(jnp.repeat(done, n)[:, None], 0., next_memory)
                trans = Transition(
                    memory=memory,
                    obs=obs_n,
                    gstate=gstate,
                    action=action,
                    z=z,
                    log_prob=log_prob,
                    reward=reward * self.mappo.reward_scale,
                    value=value,
                    # A stacked observation resets at timeout too. Treat its
                    # step budget as terminal, as for the recurrent actor.
                    term=(done if vec_env.env.observation_stack > 1 else term).astype(jnp.float32),
                    done=done.astype(jnp.float32),
                    coverage=info['coverage_ratio'],
                    wall_hit=info['wall_collision_rate'],
                    robot_hit=info['robot_collision_rate'],
                    human_hit=info['human_collision_rate'],
                    complete=info['complete'],
                    timeout=info['timeout'],
                    policy_mask=(~info['fallback_used']).astype(jnp.float32),
                    sequence_used=info['fallback_sequence_used'].astype(jnp.float32),
                    safety_override=info['fallback_safety_override'].astype(jnp.float32),
                    teacher_action=info['executed_action'],
                    teacher_mask=info['teacher_mask'],
                )

                episode_stats = _episode_stats_step(
                    episode_stats, trans, self.mappo.reward_scale, info['recoverage']
                )
                
                next_carry = RolloutCarry(
                    next_state, next_obs, next_gstate, rms,
                    episode_stats, smoothed_col_rate, smoothed_coverage_rate, next_memory,
                )
                return next_carry, trans
                
            return jax.lax.scan(scan_step, carry, keys)

        self._jitted_run = jax.jit(jitted_run)

    def start(self, key: jax.Array) -> RolloutCarry:
        state, obs, gstate, _ = self.env.reset(key)
        E, N = self.env.E, self.env.num_robots
        memory = (
            jnp.zeros((E * N, self.mappo.actor.hidden_size), jnp.float32)
            if self.mappo.actor.recurrent else None
        )
        return RolloutCarry(
            state, obs, gstate, rms_init(self.env.norm_dim),
            _episode_stats_init(E), jnp.float32(0.0), jnp.float32(0.0),
            memory,
        )

    def run(self, actor_params, critic_params, carry: RolloutCarry, num_steps: int, key: jax.Array):
        keys = jax.random.split(key, num_steps)
        final_carry, traj = self._jitted_run(actor_params, critic_params, carry, keys)
        last_value = self._value_fn(critic_params, final_carry.rms,
                                    final_carry.obs, final_carry.gstate)
        return final_carry, traj, last_value

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def policy_checkpoint_score(completion, coverage, contacts, reward):
    """Rank completed-episode metrics for checkpoint selection. Coverage comes
    before contacts so low-motion policies are never preferred."""
    return (completion, coverage, -contacts, reward)


def train(config_path: str, save_dir: str, resume: str | None,
          backend: str | None = None,
          wandb_overrides: dict | None = None, num_humans: int = 0,
          num_envs: int | None = None, policy_mode: str = "end-to-end",
          num_minibatches: int | None = None,
          num_maps: int | None = None,
          additional_updates: int | None = None,
          obs_mode: str | None = None, algo: str = "mappo"):
    if policy_mode not in ('end-to-end', 'end-to-end-memory'):
        raise ValueError(f'Unknown policy mode: {policy_mode}')
    if algo not in ('mappo', 'ippo'):
        raise ValueError(f'Unknown algo: {algo}')
    config = load_config(config_path)
    if obs_mode is not None:
        config.setdefault('env', {})['obs_mode'] = obs_mode
    device = select_device(backend)
    print(f"Device: {describe(device)}  |  requested: {backend or 'auto'}")

    env_cfg = config.setdefault('env', {})
    env_cfg['num_maps'] = int(
        num_maps if num_maps is not None
        else config.get('e2e_num_maps', env_cfg.get('num_maps', 16))
    )
    reward_weights = {**E2E_REWARD_DEFAULTS, **config.get('e2e_reward', {})}
    unknown = set(reward_weights) - set(E2E_REWARD_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown e2e_reward keys: {sorted(unknown)}')
    env_cfg.update(reward_weights)
    checkpoint_name = 'checkpoint_e2e.pkl'
    latest_name = 'checkpoint_e2e_latest.pkl'
    log_name = 'training_log_e2e.csv'
    if policy_mode == 'end-to-end-memory':
        checkpoint_name = 'checkpoint_e2e_memory.pkl'
        latest_name = 'checkpoint_e2e_memory_latest.pkl'
        log_name = 'training_log_e2e_memory.csv'
    if num_humans > 0:
        env_cfg['num_humans'] = num_humans
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})
    
    if num_envs is not None:
        train_cfg['num_envs'] = num_envs
    if num_minibatches is not None:
        train_cfg['num_minibatches'] = num_minibatches

    vec_env    = VecEnv(train_cfg.get('num_envs', 4), env_cfg)
    env        = vec_env.env
    E          = vec_env.E
    N          = vec_env.num_robots
    action_dim = vec_env.action_dim
    tail_dim = env.patch_dim
    obs_dim = env.obs_dim
    obs_config = {'obs_mode': env.obs_mode, 'use_full_memory': env.use_full_memory,
                  'observation_stack': env.observation_stack,
                  'num_robots': env.num_robots, 'n_rays': env.n_rays,
                  'cell_size': env.cell_size, 'k_teammates': env.k_teammates,
                  'local_coverage_size': env.local_coverage_size,
                  'use_local_coverage_obs': env.use_local_coverage_obs,
                  'sweep_obs': env.sweep_obs, 'crop_summary': env.crop_summary,
                  'history_cell': env.history_cell,
                  'critic_context': env.critic_context, 'critic_stack': env.critic_stack,
                  'critic_crops': env.critic_crops,
                  'known_coverage_obs': env.known_coverage_obs,
                  'critic_coverage': env.critic_coverage,
                  'wall_cells': env.wall_cells}
    # Persist recovery control settings alongside the observation regime so
    # evaluation/visualisation reproduce the training controller.
    obs_config.update({name: getattr(env, name) for name in (
        'fallback_enabled', 'fallback_revisit_threshold', 'fallback_stall_steps',
        'fallback_linear_accel', 'fallback_angular_accel', 'fallback_dwa_steps',
        'fallback_dwa_stall_steps', 'fallback_sequence_steps', 'fallback_sequence_speed')})
    obs_config.update(comm_radius=env.comm_radius)
    if env.use_memory:
        obs_config.update(comm_radius=env.comm_radius, comm_slots=env.comm_slots,
                          local_coverage_size=env.local_coverage_size)

    lidar_embed = model_cfg.get('lidar_embed',  64)
    hidden_size = model_cfg.get('hidden_size', 128)
    actor_config = dict(lidar_embed=lidar_embed, hidden_size=hidden_size,
                        log_std_min=model_cfg.get('log_std_min', -5.0),
                        log_std_max=model_cfg.get('log_std_max', 1.0))
    trunk_in    = lidar_embed + env.obs_vec_dim + tail_dim

    print(f"Parallel envs: {E}  |  robots/env: {N}  |  obs_dim: {obs_dim} "
          f"({env.obs_dim} env)  |  critic map: "
          f"{vec_env.critic_channels}x{vec_env.grid_h}x{vec_env.grid_w}"
          f" + {vec_env.critic_vec_dim}")
    free_totals = np.asarray(env.free_totals)
    if env.num_maps == 1:
        print(f"Coverable cells: {int(free_totals[0])} / {env.num_cells} "
              f"({free_totals[0] / env.num_cells:.1%} of the grid)")
    else:
        print(f"Map bank: {env.num_maps} layouts | coverable cells: "
              f"{int(free_totals.min())}-{int(free_totals.max())} / {env.num_cells}")
    print(f"Policy: {policy_mode}")

    print(f"Observation: {obs_config}")
    print(f"Reward mode: {env.reward_mode}; weights: {reward_weights}")
    if env.fallback_enabled:
        print('Training coverage includes recovery; use evaluate_policies --compare-policy-only to measure autonomous coverage.')

    actor = Actor(
        recurrent=policy_mode == "end-to-end-memory",
        action_dim=action_dim,
        vec_dim=env.obs_vec_dim,
        n_rays=env.n_rays,
        tail_dim=tail_dim,
        memory_map_shape=env.memory_map_shape,
        observation_stack=env.observation_stack,
        **actor_config,
    )
    if algo == 'ippo':
        critic = LocalCritic(
            vec_dim=env.obs_vec_dim,
            n_rays=env.n_rays,
            tail_dim=tail_dim,
            memory_map_shape=env.memory_map_shape,
            observation_stack=env.observation_stack,
            lidar_embed=lidar_embed,
            hidden_size=model_cfg.get('critic_hidden', 256),
        )
    else:
        critic = Critic(
            hidden_size=model_cfg.get('critic_hidden',    256),
            map_embed=model_cfg.get('critic_map_embed', 128),
        )
    mappo = (IPPO if algo == 'ippo' else MAPPO)(
        actor, critic, vec_env, train_cfg, device=device
    )
    mappo.env = vec_env
    print(f"Algorithm: {algo.upper()} ("
          f"{'local critic V(o_i)' if algo == 'ippo' else 'centralised critic V_i(s)'}"
          f", shared actor and critic parameters)")

    T             = train_cfg.get('rollout_steps',  256)
    total_updates = train_cfg.get('total_updates',  3000)
    log_interval  = train_cfg.get('log_interval',   10)
    gamma         = train_cfg.get('gamma',          0.99)
    gae_lambda    = train_cfg.get('gae_lambda',     0.95)
    normalize_obs = train_cfg.get('normalize_obs',  True)
    lr_decay      = train_cfg.get('lr_decay',       True)
    lr_actor_0    = train_cfg.get('lr_actor',       3e-4)
    lr_critic_0   = train_cfg.get('lr_critic',      1e-3)

    key = jax.random.PRNGKey(train_cfg.get('seed', 0))
    key, init_key, reset_key = jax.random.split(key, 3)

    actor_state, critic_state = mappo.create_train_states(init_key)

    print(f"Maps: {env.grid_h}x{env.grid_w} cells, sampled independently on reset")
    rollout = Rollout(mappo, vec_env)
    carry = rollout.start(reset_key)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_name)

    start_update = 1
    if resume:
        actor_state, critic_state, rms, last = load_checkpoint(
            resume, actor_state, critic_state, device, trunk_in
        )
        carry = carry._replace(rms=rms)
        start_update = last + 1
        if additional_updates is not None:
            if additional_updates <= 0:
                raise ValueError('additional_updates must be positive')
            total_updates = last + additional_updates
        print(f"Resumed from {resume}, continuing at update {start_update}")
        if additional_updates is not None:
            print(f"Fine-tuning for {additional_updates} additional updates "
                  f"(through update {total_updates})")
    elif additional_updates is not None:
        raise ValueError('additional_updates requires a resume checkpoint')

    run = init_wandb(
        config,
        wandb_overrides or {},
        extra={
            'device':           describe(device),
            'num_envs':         E,
            'num_robots':       N,
            'obs_dim':          obs_dim,
            'action_dim':       action_dim,
            'coverable_cells':  int(env.free_totals[0]),
            'steps_per_update': T * E,
            'policy_mode':      policy_mode,
            'obs_mode':         env.obs_mode,
            'algo':             algo,
            
        },
    )
    if run is not None:
        print(f"W&B run: {run.url or run.dir}")

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['update', 'episodes', 'env_steps',
                                'mean_ep_reward', 'mean_ep_coverage', 'mean_ep_recoverage',
                                'coverage_ratio', 'mean_ep_length',
                                'completion_rate', 'timeout_rate',
                                'collision_end_rate',
                                'wall_collision_rate', 'robot_collision_rate',
                                'human_collision_rate',
                                'wall_collisions_per_episode',
                                'robot_collisions_per_episode',
                                'human_collisions_per_episode',
                                'actor_loss', 'critic_loss', 'entropy', 'std', 'approx_kl', 'clip_fraction', 'actor_update_fraction',
                                'policy_fraction', 'sequence_fraction', 'sequence_override_fraction',
                                'recovery_imitation_loss', 'teacher_fraction'])

    best_policy_score = None

    for update in range(start_update, total_updates + 1):
        lr_a = linear_lr_decay(lr_actor_0,  update, total_updates) if lr_decay else lr_actor_0
        lr_c = linear_lr_decay(lr_critic_0, update, total_updates) if lr_decay else lr_critic_0

        ghost_prob = ghost_robot_probability(
            carry.smoothed_col_rate, carry.smoothed_coverage_rate
        )
        carry = carry._replace(env_state=vec_env.update_ghost_robot_prob(
            carry.env_state, jnp.full((E,), ghost_prob, dtype=jnp.float32)
        ))

        key, rollout_key = jax.random.split(key)
        prev_rms = carry.rms
        carry, traj, last_value = rollout.run(
            actor_state.params, critic_state.params, carry, T, rollout_key
        )
        total_col_rate = (jnp.mean(traj.wall_hit)
                          + jnp.mean(traj.robot_hit)
                          + jnp.mean(traj.human_hit))
        carry = carry._replace(
            smoothed_col_rate=0.9 * carry.smoothed_col_rate + 0.1 * total_col_rate,
            smoothed_coverage_rate=(
                0.9 * carry.smoothed_coverage_rate + 0.1 * jnp.mean(traj.coverage)
            ),
        )
        if not normalize_obs:
            carry = carry._replace(rms=prev_rms)

        advantages, returns = compute_gae(traj, last_value, gamma, gae_lambda)

        actor_state, critic_state, metrics = mappo.update(
            actor_state, critic_state, traj, advantages, returns, lr_a, lr_c
        )

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if update % log_interval == 0:
            s = carry.episode_stats
            compact = jax.device_get((
                s.recent_reward, s.recent_coverage, s.recent_recoverage, s.recent_length,
                s.recent_wall, s.recent_robot, s.recent_human,
                s.recent_outcome, s.ring_count, s.total_count,
                jnp.mean(traj.coverage[-1]), jnp.mean(traj.wall_hit),
                jnp.mean(traj.robot_hit), jnp.mean(traj.human_hit), metrics,
                jnp.mean(traj.policy_mask), jnp.mean(traj.sequence_used), jnp.mean(traj.safety_override),
            ))
            (recent_reward, recent_coverage, recent_recoverage, recent_length, recent_wall,
             recent_robot, recent_human, recent_outcome, ring_count,
             ep_count, last_cov, wall_rate, robot_rate, human_rate,
             losses, policy_fraction, sequence_fraction, override_fraction) = compact
            count = int(ring_count)
            ep_count = int(ep_count)
            valid = slice(0, count)

            def recent_mean(values):
                return float(np.mean(np.asarray(values)[valid])) if count else 0.0

            mean_ep_r = recent_mean(recent_reward)
            mean_ep_cov = recent_mean(recent_coverage)
            mean_ep_recov = recent_mean(recent_recoverage) if count else 1.0
            mean_ep_len = recent_mean(recent_length)
            ep_wall_mean = recent_mean(recent_wall)
            ep_robot_mean = recent_mean(recent_robot)
            ep_human_mean = recent_mean(recent_human)
            policy_fraction = float(policy_fraction)
            sequence_fraction = float(sequence_fraction)
            override_fraction = float(override_fraction)
            last_cov = float(last_cov)
            wall_rate = float(wall_rate)
            robot_rate = float(robot_rate)
            human_rate = float(human_rate)
            outcomes = np.asarray(recent_outcome)[valid]
            if outcomes.size:
                completion_rate    = float(np.mean(outcomes == _SUCCESS))
                timeout_rate       = float(np.mean(outcomes == _TIMEOUT))
                collision_end_rate = float(np.mean(outcomes == _COLLISION))
            else:
                completion_rate = timeout_rate = collision_end_rate = 0.0
            env_steps = update * T * E
            print(
                f"Update {update:5d}/{total_updates} | "
                f"episodes={ep_count:6d} | "
                f"mean_ep_r={mean_ep_r:8.3f} | "
                f"ep_cov={mean_ep_cov:6.2%} | "
                f"recoverage={mean_ep_recov:.3f} | "
                f"complete={completion_rate:6.2%} | "
                f"coverage={last_cov:.2%} | "
                f"actor={float(losses['actor_loss']):7.4f} | "
                f"critic={float(losses['critic_loss']):7.4f} | "
                f"entropy={float(losses['entropy']):6.4f} | "
                f"std={float(losses['std']):5.3f} | "
                f"kl={float(losses['approx_kl']):.4f} | clip={float(losses['clip_fraction']):.1%} | "
                f"actor_updates={float(losses['actor_update_fraction']):.1%} | "
                f"imitation={float(losses['recovery_imitation_loss']):.4f} | "
                f"teacher={float(losses['teacher_fraction']):.1%} | "
                f"policy={policy_fraction:.1%} | sequence={sequence_fraction:.1%} | "
                f"seq_blocked={override_fraction:.1%} | "
                f"rr={robot_rate:6.2%} | "
                f"rh={human_rate:6.2%} | "
                f"rw={wall_rate:6.2%} | "
                f"timeout={timeout_rate:6.2%}",
                flush=True,
            )
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    update, ep_count, env_steps,
                    round(mean_ep_r,   4), round(mean_ep_cov, 4),
                    round(mean_ep_recov, 4),
                    round(last_cov,    4), round(mean_ep_len, 1),
                    round(completion_rate,    4),
                    round(timeout_rate,       4),
                    round(collision_end_rate, 4),
                    round(wall_rate,      6),
                    round(robot_rate,     6),
                    round(human_rate,     6),
                    round(ep_wall_mean,   4),
                    round(ep_robot_mean,  4),
                    round(ep_human_mean,  4),
                    round(float(losses['actor_loss']),  4),
                    round(float(losses['critic_loss']), 4),
                    round(float(losses['entropy']),     4),
                    round(float(losses['std']),         4),
                    round(float(losses['approx_kl']), 6),
                    round(float(losses['clip_fraction']), 6),
                    round(float(losses['actor_update_fraction']), 6),
                    round(policy_fraction, 6), round(sequence_fraction, 6), round(override_fraction, 6),
                    round(float(losses['recovery_imitation_loss']), 6),
                    round(float(losses['teacher_fraction']), 6),
                ])
            if run is not None:
                wandb.log({
                    'env_steps':                      env_steps,
                    'update':                         update,
                    'episode/count':                  ep_count,
                    'episode/mean_reward':            mean_ep_r,
                    'episode/mean_coverage':          mean_ep_cov,
                    'episode/mean_recoverage':        mean_ep_recov,
                    'episode/mean_length':            mean_ep_len,
                    'episode/coverage_last_step':     last_cov,
                    'rate/completion':                completion_rate,
                    'rate/timeout':                   timeout_rate,
                    'rate/collision_end':             collision_end_rate,
                    'collision/wall_per_robot_step':  wall_rate,
                    'collision/robot_per_robot_step': robot_rate,
                    'collision/human_per_robot_step': human_rate,
                    'collision/wall_per_episode':     ep_wall_mean,
                    'collision/robot_per_episode':    ep_robot_mean,
                    'collision/human_per_episode':    ep_human_mean,
                    'loss/actor':                     float(losses['actor_loss']),
                    'loss/critic':                    float(losses['critic_loss']),
                    'loss/recovery_imitation':        float(losses['recovery_imitation_loss']),
                    'control/teacher_fraction':       float(losses['teacher_fraction']),
                    'loss/entropy':                   float(losses['entropy']),
                    'policy/std':                     float(losses['std']),
                    'policy/approx_kl':               float(losses['approx_kl']),
                    'policy/clip_fraction':           float(losses['clip_fraction']),
                    'policy/actor_update_fraction':   float(losses['actor_update_fraction']),
                    'control/policy_fraction':        policy_fraction,
                    'control/sequence_fraction':      sequence_fraction,
                    'control/sequence_override_fraction': override_fraction,
                    'lr/actor':                       lr_a,
                    'lr/critic':                      lr_c,
                }, step=update)
            policy_score = policy_checkpoint_score(
                completion_rate, mean_ep_cov,
                ep_wall_mean + ep_robot_mean + ep_human_mean, mean_ep_r,
            )
            if ep_count > 0 and (
                best_policy_score is None or policy_score > best_policy_score
            ):
                best_policy_score = policy_score
                save_checkpoint(os.path.join(save_dir, checkpoint_name),
                                update, actor_state, critic_state, carry.rms,
                                tail_dim, policy_mode, reward_weights,
                                obs_config, algo, actor_config)
                print(
                    f"  → best policy saved (complete={completion_rate:.2%}, "
                    f"coverage={mean_ep_cov:.2%}, contacts/ep="
                    f"{ep_wall_mean + ep_robot_mean + ep_human_mean:.2f})"
                )
                if run is not None:
                    run.summary['best_mean_ep_reward'] = mean_ep_r
                    run.summary['best_update'] = update

    save_checkpoint(os.path.join(save_dir, latest_name),
                    total_updates, actor_state, critic_state, carry.rms,
                    tail_dim, policy_mode, reward_weights,
                    obs_config, algo, actor_config)
    if run is not None:
        run.finish()
    return actor_state, critic_state, carry.rms


if __name__ == '__main__':
    _default_cfg  = os.path.join(os.path.dirname(__file__), '..', 'config',
                                 'mappo_baseline.yaml')
    _default_save = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

    parser = argparse.ArgumentParser(
        description='Train a feed-forward or recurrent end-to-end coverage policy')
    parser.add_argument('--policy-mode', choices=['end-to-end', 'end-to-end-memory'],
                        default='end-to-end',
                        help='end-to-end-memory uses a GRU actor')
    parser.add_argument('--config',   default=_default_cfg,
                        help='Path to YAML config file')
    parser.add_argument('--save-dir', default=_default_save,
                        help='Directory for checkpoints and training log')
    parser.add_argument('--resume',   default=None,
                        help='Checkpoint with matching actor architecture to resume from')
    parser.add_argument('--backend',  default='auto',
                        choices=['auto', 'metal', 'cuda', 'gpu', 'cpu'],
                        help='Force a JAX backend. Default "auto": Metal on Apple '
                             'Silicon, else CUDA when an NVIDIA GPU is present, else CPU')
    parser.add_argument('--wandb', dest='wandb_enabled', action='store_true',
                        default=None, help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', dest='wandb_enabled', action='store_false',
                        default=None, help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', default=None, help='W&B project name')
    parser.add_argument('--wandb-entity',  default=None, help='W&B team / user')
    parser.add_argument('--wandb-name',    default=None, help='W&B run name')
    parser.add_argument('--wandb-group',   default=None, help='W&B run group')
    parser.add_argument('--wandb-mode',    default=None,
                        choices=['online', 'offline', 'disabled'],
                        help='W&B mode; "offline" logs locally with no network')
    parser.add_argument('--humans', nargs='?', type=int, const=3, default=0, help='Number of humans')
    parser.add_argument('--envs', type=int, default=None, help='Number of parallel environments (overrides config, defaults to 64 on GPU if config uses <=16)')
    parser.add_argument('--minibatches', type=int, default=None,
                        help='PPO minibatches per epoch, split by environment '
                             '(overrides train.num_minibatches; must divide --envs)')
    parser.add_argument('--maps', type=int, default=None,
                        help='Procedural map-bank size')
    parser.add_argument('--algo', choices=['mappo', 'ippo'], default='mappo',
                        help='mappo: centralised critic on the global state; '
                             'ippo: independent critic on the local observation')
    parser.add_argument('--obs-mode', choices=['legacy', 'memory_comm'], default=None,
                        help='Actor observation (overrides env.obs_mode). memory_comm: '
                             'lidar-built per-robot map memory shared within '
                             'comm_radius')
    parser.add_argument('--additional-updates', type=int, default=None,
                        help='When resuming, run exactly this many extra updates')
    args = parser.parse_args()
    train(args.config, args.save_dir, args.resume,
          None if args.backend == 'auto' else args.backend,
          wandb_overrides={
              'enabled': args.wandb_enabled,
              'project': args.wandb_project,
              'entity':  args.wandb_entity,
              'name':    args.wandb_name,
              'group':   args.wandb_group,
              'mode':    args.wandb_mode,
          }, num_humans=args.humans, num_envs=args.envs,
          num_minibatches=args.minibatches,
          policy_mode=args.policy_mode, num_maps=args.maps,
          additional_updates=args.additional_updates,
          obs_mode=args.obs_mode, algo=args.algo)
