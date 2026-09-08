#!/usr/bin/env python3
"""Train MAPPO with BOSCO guidance or an end-to-end actor under CTDE.

`--policy-mode guided` preserves target deltas in actor observations and BOSCO
reward shaping. `--policy-mode end-to-end` removes those two input channels and
all planner-dependent rewards, including assignment-weighted discoveries.
BOSCO targets and assignments remain available only to the centralized critic.
Both modes share the device-side rollout, PPO, local coverage inputs and config.

    python -m src.train_bosco --policy-mode end-to-end --save-dir checkpoints/e2e

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

from src.algorithms.bosco_guide import make_guides
from src.algorithms.jax_bosco import JaxGuideState, jax_guide_step
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
from src.models.actor_critic import Actor, Critic
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
                    guide_dim: int, guide_bonus: float = 2.0,
                    policy_mode: str = "guided", reward_weights: dict | None = None) -> None:
    """Save parameters, normalizer and the actor/reward regime for evaluation."""
    payload = {
        'update':        update,
        'actor_params':  jax.device_get(actor_state.params),
        'critic_params': jax.device_get(critic_state.params),
        'actor_opt':     jax.device_get(actor_state.opt_state),
        'critic_opt':    jax.device_get(critic_state.opt_state),
        'obs_rms':       jax.device_get(rms),
        'guide_dim':     int(guide_dim),
        'bosco_guided':  policy_mode == 'guided',
        'policy_mode': policy_mode,
        'actor_recurrent': policy_mode == 'end-to-end-memory',
        'actor_bosco_guidance': policy_mode == 'guided',
        'bosco_reward_guidance': policy_mode == 'guided',
        'guide_bonus':   float(guide_bonus),
        'reward_weights': dict(reward_weights or {}),
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
# Guided rollout
# ---------------------------------------------------------------------------

class GuidedCarry(NamedTuple):
    """Rollout state threaded between updates."""

    env_state: object
    obs:       jax.Array      # actor inputs for the selected policy mode
    gstate:    object
    rms:       object
    guide_state: JaxGuideState
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
    hits:      jax.Array
    recent_reward:   jax.Array
    recent_coverage: jax.Array
    recent_length:   jax.Array
    recent_wall:     jax.Array
    recent_robot:    jax.Array
    recent_human:    jax.Array
    recent_hits:     jax.Array
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
        hits=zeros_e,
        recent_reward=zeros_w,
        recent_coverage=zeros_w,
        recent_length=zeros_w,
        recent_wall=zeros_w,
        recent_robot=zeros_w,
        recent_human=zeros_w,
        recent_hits=zeros_w,
        recent_outcome=jnp.zeros((_WINDOW,), jnp.int32),
        ring_pos=jnp.int32(0),
        ring_count=jnp.int32(0),
        total_count=jnp.int32(0),
    )


def _episode_stats_step(stats: DeviceEpisodeStats, trans: Transition,
                        reached: jax.Array, reward_scale: float) -> DeviceEpisodeStats:
    """Consume one vectorised transition without copying trajectory data to host."""
    stats = stats._replace(
        reward=stats.reward + jnp.mean(trans.reward, axis=-1) / reward_scale,
        length=stats.length + 1,
        wall=stats.wall + trans.wall_hit,
        robot=stats.robot + trans.robot_hit,
        human=stats.human + trans.human_hit,
        hits=stats.hits + jnp.mean(reached, axis=-1),
    )

    def one_env(s, xs):
        done, coverage, complete, timeout, env_idx = xs

        def finish(x):
            pos = x.ring_pos
            outcome = jnp.where(
                complete > 0.5, jnp.int32(_SUCCESS),
                jnp.where(timeout > 0.5, jnp.int32(_TIMEOUT), jnp.int32(_COLLISION)),
            )
            x = x._replace(
                recent_reward=x.recent_reward.at[pos].set(x.reward[env_idx]),
                recent_coverage=x.recent_coverage.at[pos].set(coverage),
                recent_length=x.recent_length.at[pos].set(x.length[env_idx]),
                recent_wall=x.recent_wall.at[pos].set(x.wall[env_idx]),
                recent_robot=x.recent_robot.at[pos].set(x.robot[env_idx]),
                recent_human=x.recent_human.at[pos].set(x.human[env_idx]),
                recent_hits=x.recent_hits.at[pos].set(x.hits[env_idx]),
                recent_outcome=x.recent_outcome.at[pos].set(outcome),
                reward=x.reward.at[env_idx].set(0.0),
                length=x.length.at[env_idx].set(0),
                wall=x.wall.at[env_idx].set(0.0),
                robot=x.robot.at[env_idx].set(0.0),
                human=x.human.at[env_idx].set(0.0),
                hits=x.hits.at[env_idx].set(0.0),
                ring_pos=(pos + 1) % _WINDOW,
                ring_count=jnp.minimum(x.ring_count + 1, _WINDOW),
                total_count=x.total_count + 1,
            )
            return x

        return jax.lax.cond(done > 0.5, finish, lambda x: x, s), None

    env_idx = jnp.arange(trans.done.shape[0], dtype=jnp.int32)
    stats, _ = jax.lax.scan(
        one_env, stats,
        (trans.done, trans.coverage, trans.complete, trans.timeout, env_idx),
    )
    return stats


class GuidedRollout:
    def __init__(self, mappo: MAPPO, vec_env: VecEnv, guides, guide_bonus: float):
        self.mappo = mappo
        self.env = vec_env
        self.guides = guides
        self.graph = guides[0].graph
        self.bonus = float(guide_bonus)
        self.graph_neighbors = jnp.asarray(self.graph.neighbors, jnp.int32)
        self.graph_free = jnp.asarray(self.graph.free, jnp.bool_)
        self.graph_components = jnp.asarray(self.graph.component, jnp.int32)
        self.graph_centers = jnp.asarray(self.graph.centers, jnp.float32)
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
            value = mappo._values(critic_params, gstate)
            return (obs_n, action.reshape(e, n, -1), z.reshape(e, n, -1),
                    log_prob, value, rms, memory)

        self._act = act
        self._value_fn = jax.jit(mappo._values)
        
        def jitted_run(actor_params, critic_params, carry, keys):
            def scan_step(c, k):
                (state, obs, gstate, rms, guide_state, episode_stats,
                 smoothed_col_rate, smoothed_coverage_rate, memory) = c
                
                obs_n, action, z, log_prob, value, rms, next_memory = act(
                    actor_params, critic_params, rms, obs, gstate, k, memory
                )
                next_state, _, reward, term, done, info, _ = vec_env.step(state, action)
                
                new_guide_state, waypoint, reached = jax_guide_step(
                    guide_state,
                    next_state.robot_positions,
                    next_state.coverage_grid,
                    done,
                    self.graph_neighbors,
                    vec_env.grid_w,
                    vec_env.grid_h,
                    vec_env.env.cell_size,
                    free_cells=self.graph_free,
                    graph_components=self.graph_components,
                    previous_coverage_grid=state.coverage_grid,
                )
                
                valid = waypoint >= 0
                safe_targets = jnp.maximum(waypoint, 0)
                coords = self.graph_centers[safe_targets]
                target_coords = jnp.where(valid[..., None], coords, next_state.robot_positions)
                
                next_state, next_obs, next_gstate = vec_env.update_bosco(next_state, target_coords)
                
                if mappo.actor.recurrent:
                    next_memory = jnp.where(jnp.repeat(done, n)[:, None], 0., next_memory)
                trans = Transition(
                    memory=memory,
                    obs=obs_n,
                    gstate=gstate,
                    action=action,
                    z=z,
                    log_prob=log_prob,
                    reward=(reward + self.bonus * jnp.asarray(reached, jnp.float32)) * self.mappo.reward_scale,
                    value=value,
                    term=term.astype(jnp.float32),
                    done=done.astype(jnp.float32),
                    coverage=info['coverage_ratio'],
                    wall_hit=info['wall_collision_rate'],
                    robot_hit=info['robot_collision_rate'],
                    human_hit=info['human_collision_rate'],
                    complete=info['complete'],
                    timeout=info['timeout'],
                )

                episode_stats = _episode_stats_step(
                    episode_stats, trans, reached, self.mappo.reward_scale
                )
                
                next_carry = GuidedCarry(
                    next_state, next_obs, next_gstate, rms, new_guide_state,
                    episode_stats, smoothed_col_rate, smoothed_coverage_rate, next_memory,
                )
                return next_carry, (trans, reached)
                
            return jax.lax.scan(scan_step, carry, keys)

        self._jitted_run = jax.jit(jitted_run)

    @staticmethod
    def _host(state, done=None):
        payload = (state.robot_positions, state.robot_headings, state.coverage_grid)
        return jax.device_get(payload if done is None else payload + (done,))

    def start(self, key: jax.Array) -> GuidedCarry:
        state, obs, gstate, _ = self.env.reset(key)
        pos, hdg, cov = self._host(state)
        E, N = self.env.E, self.env.num_robots
        
        MAX_TOUR_LEN = 2048
        tours = np.full((E, N, MAX_TOUR_LEN), -1, dtype=np.int32)
        tour_lens = np.zeros((E, N), dtype=np.int32)
        targets = np.full((E, N), -1, dtype=np.int32)
        assignments = np.zeros(
            (E, N, self.env.grid_h, self.env.grid_w), dtype=np.float32
        )
        
        for e in range(E):
            self.guides[e].reset(pos[e])
            owner = self.guides[e].owner.reshape(self.env.grid_h, self.env.grid_w)
            assignments[e] = np.stack([owner == r for r in range(N)]).astype(np.float32)
            for r in range(N):
                tour = self.guides[e].tours[r]
                t_len = min(len(tour), MAX_TOUR_LEN)
                tour_lens[e, r] = t_len
                tours[e, r, :t_len] = tour[:MAX_TOUR_LEN]
            targets[e], _ = self.guides[e].update(pos[e], cov[e])
            
        valid = targets >= 0
        safe_targets = np.maximum(targets, 0)
        coords = self.graph.centers[safe_targets]
        target_coords = np.where(valid[..., None], coords, pos)
        
        state, obs, gstate = self.env.update_bosco(
            state, jnp.asarray(target_coords, dtype=jnp.float32)
        )
        state, obs, gstate = self.env.update_cell_assignments(
            state, jnp.asarray(assignments)
        )
        
        cell_size = self.env.env.cell_size
        col = np.clip((pos[..., 0] / cell_size).astype(np.int32), 0, self.env.grid_w - 1)
        row = np.clip((pos[..., 1] / cell_size).astype(np.int32), 0, self.env.grid_h - 1)
        cell = row * self.env.grid_w + col
        
        guide_state = JaxGuideState(
            target=jnp.asarray(targets, jnp.int32),
            prev_cell=jnp.asarray(cell, jnp.int32),
            fail_cov=jnp.full((E, N), -1, dtype=jnp.int32),
            idx=jnp.zeros((E, N), dtype=jnp.int32),
            tours=jnp.asarray(tours, jnp.int32),
            tour_lens=jnp.asarray(tour_lens, jnp.int32)
        )
        
        return GuidedCarry(
            state, obs, gstate, rms_init(self.env.norm_dim), guide_state,
            _episode_stats_init(E), jnp.float32(0.0), jnp.float32(0.0),
            (jnp.zeros((E * N, self.mappo.actor.hidden_size), jnp.float32)
             if self.mappo.actor.recurrent else None),
        )

    def run(self, actor_params, critic_params, carry: GuidedCarry, num_steps: int, key: jax.Array):
        keys = jax.random.split(key, num_steps)
        final_carry, (traj, hits) = self._jitted_run(actor_params, critic_params, carry, keys)
        last_value = self._value_fn(critic_params, final_carry.gstate)
        return final_carry, traj, last_value, hits

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Kept apart from `train_simple`'s own outputs: the two policies are not
# interchangeable — a guided actor rejects a bare environment observation — so
# writing both to `checkpoint_final.pkl` would silently destroy the baseline.
CHECKPOINT_NAME = 'checkpoint_bosco.pkl'
LATEST_CHECKPOINT_NAME = 'checkpoint_bosco_latest.pkl'
LOG_NAME        = 'training_log_bosco.csv'


def policy_checkpoint_score(policy_mode, completion, coverage, contacts, reward):
    """Rank completed-episode metrics for checkpoint selection."""
    if policy_mode in ("end-to-end", "end-to-end-memory"):
        return (completion, coverage, -contacts, reward)
    return (completion, -contacts, coverage, reward)


def train(config_path: str, save_dir: str, resume: str | None,
          backend: str | None = None, guide_bonus: float = 10.0,
          wandb_overrides: dict | None = None, num_humans: int = 0,
          num_envs: int | None = None, policy_mode: str = "guided"):
    if policy_mode not in ('guided', 'end-to-end', 'end-to-end-memory'):
        raise ValueError(f'Unknown policy mode: {policy_mode}')
    config = load_config(config_path)
    device = select_device(backend)
    print(f"Device: {describe(device)}  |  requested: {backend or 'auto'}")

    env_cfg = config.setdefault('env', {})
    env_cfg['actor_bosco_guidance'] = policy_mode == 'guided'
    env_cfg['bosco_reward_guidance'] = policy_mode == 'guided'
    if policy_mode != 'guided':
        guide_bonus = 0.0
        reward_weights = {**E2E_REWARD_DEFAULTS, **config.get('e2e_reward', {})}
        unknown = set(reward_weights) - set(E2E_REWARD_DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown e2e_reward keys: {sorted(unknown)}')
        env_cfg.update(reward_weights)
    else:
        env_cfg['reward_mode'] = 'legacy'
        reward_weights = {name: env_cfg[name] for name in ('wall_kappa', 'beta')
                          if name in env_cfg}
    checkpoint_name = CHECKPOINT_NAME if policy_mode == 'guided' else 'checkpoint_e2e.pkl'
    latest_name = LATEST_CHECKPOINT_NAME if policy_mode == 'guided' else 'checkpoint_e2e_latest.pkl'
    log_name = LOG_NAME if policy_mode == 'guided' else 'training_log_e2e.csv'
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

    vec_env    = VecEnv(train_cfg.get('num_envs', 4), env_cfg)
    env        = vec_env.env
    if env.num_maps != 1:
        raise ValueError(
            "BOSCO-guided training currently requires env.num_maps=1: the "
            "accelerator rollout shares one traversability graph across all "
            "environments. Multiple maps would produce invalid waypoints."
        )
    E          = vec_env.E
    N          = vec_env.num_robots
    action_dim = vec_env.action_dim
    tail_dim = env.patch_dim
    obs_dim = env.obs_dim

    lidar_embed = model_cfg.get('lidar_embed',  64)
    hidden_size = model_cfg.get('hidden_size', 128)
    trunk_in    = lidar_embed + env.obs_vec_dim + tail_dim

    print(f"Parallel envs: {E}  |  robots/env: {N}  |  obs_dim: {obs_dim} "
          f"({env.obs_dim} env)  |  critic map: "
          f"{vec_env.critic_channels}x{vec_env.grid_h}x{vec_env.grid_w}"
          f" + {vec_env.critic_vec_dim}")
    print(f"Coverable cells: {int(env.free_totals[0])} / {env.num_cells} "
          f"({env.free_totals[0] / env.num_cells:.1%} of the grid)")
    print(f"Policy: {policy_mode}; BOSCO arrival bonus {guide_bonus} "
          f"(discovery alpha={env.alpha}, coverage growth="
          f"{env.coverage_reward_growth})")

    print(f"Reward mode: {env.reward_mode}; weights: wall_kappa={env.wall_kappa:g}, beta={env.beta:g}")

    actor = Actor(
        recurrent=policy_mode == "end-to-end-memory",
        action_dim=action_dim,
        vec_dim=env.obs_vec_dim,
        n_rays=env.n_rays,
        tail_dim=tail_dim,
        lidar_embed=lidar_embed,
        hidden_size=hidden_size,
    )
    critic = Critic(
        hidden_size=model_cfg.get('critic_hidden',    256),
        map_embed=model_cfg.get('critic_map_embed', 128),
    )
    mappo = MAPPO(actor, critic, vec_env, train_cfg, device=device)
    mappo.env = vec_env

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

    guides = make_guides(env, E)
    print(f"Map: {env.grid_h}x{env.grid_w} cells, "
          f"{int(guides[0]._isolated.sum())} unreachable by the robot disc")
    rollout = GuidedRollout(mappo, vec_env, guides, guide_bonus)
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
        print(f"Resumed from {resume}, continuing at update {start_update}")

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
            'guide_bonus':      guide_bonus,
            'policy_mode':      policy_mode,
            
        },
    )
    if run is not None:
        print(f"W&B run: {run.url or run.dir}")

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['update', 'episodes', 'env_steps',
                                'mean_ep_reward', 'mean_ep_coverage',
                                'coverage_ratio', 'mean_ep_length',
                                'completion_rate', 'timeout_rate',
                                'collision_end_rate',
                                'wall_collision_rate', 'robot_collision_rate',
                                'human_collision_rate',
                                'wall_collisions_per_episode',
                                'robot_collisions_per_episode',
                                'human_collisions_per_episode',
                                'guide_hits_per_episode', 'guide_hit_rate',
                                'actor_loss', 'critic_loss', 'entropy', 'std'])

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
        carry, traj, last_value, hits = rollout.run(
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
                s.recent_reward, s.recent_coverage, s.recent_length,
                s.recent_wall, s.recent_robot, s.recent_human, s.recent_hits,
                s.recent_outcome, s.ring_count, s.total_count,
                jnp.mean(traj.coverage[-1]), jnp.mean(traj.wall_hit),
                jnp.mean(traj.robot_hit), jnp.mean(traj.human_hit),
                jnp.mean(hits), metrics,
            ))
            (recent_reward, recent_coverage, recent_length, recent_wall,
             recent_robot, recent_human, recent_hits, recent_outcome, ring_count,
             ep_count, last_cov, wall_rate, robot_rate, human_rate, guide_rate,
             losses) = compact
            count = int(ring_count)
            ep_count = int(ep_count)
            valid = slice(0, count)

            def recent_mean(values):
                return float(np.mean(np.asarray(values)[valid])) if count else 0.0

            mean_ep_r = recent_mean(recent_reward)
            mean_ep_cov = recent_mean(recent_coverage)
            mean_ep_len = recent_mean(recent_length)
            ep_wall_mean = recent_mean(recent_wall)
            ep_robot_mean = recent_mean(recent_robot)
            ep_human_mean = recent_mean(recent_human)
            ep_hit_mean = recent_mean(recent_hits)
            last_cov = float(last_cov)
            wall_rate = float(wall_rate)
            robot_rate = float(robot_rate)
            human_rate = float(human_rate)
            # Fraction of robot-steps that ended on the waypoint. A robot needs
            # ~5 steps to cross a 0.5 m cell at v_max, so 0.2 is the arithmetic
            # ceiling, but BOSCO itself only reaches 0.095 on this map — a 90°
            # lane change costs ~16 steps of turning in place, which buys no
            # waypoint. That measured 0.095, not the ceiling, is what a policy
            # tracking the tour as well as the planner does looks like.
            guide_rate = float(guide_rate)
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
                f"complete={completion_rate:6.2%} | "
                f"coverage={last_cov:.2%} | "
                f"guide={guide_rate:6.2%} | "
                f"actor={float(losses['actor_loss']):7.4f} | "
                f"critic={float(losses['critic_loss']):7.4f} | "
                f"entropy={float(losses['entropy']):6.4f} | "
                f"std={float(losses['std']):5.3f} | "
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
                    round(ep_hit_mean,    4),
                    round(guide_rate,     6),
                    round(float(losses['actor_loss']),  4),
                    round(float(losses['critic_loss']), 4),
                    round(float(losses['entropy']),     4),
                    round(float(losses['std']),         4),
                ])
            if run is not None:
                wandb.log({
                    'env_steps':                      env_steps,
                    'update':                         update,
                    'episode/count':                  ep_count,
                    'episode/mean_reward':            mean_ep_r,
                    'episode/mean_coverage':          mean_ep_cov,
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
                    'guide/hit_per_robot_step':       guide_rate,
                    'guide/hits_per_episode':         ep_hit_mean,
                    'loss/actor':                     float(losses['actor_loss']),
                    'loss/critic':                    float(losses['critic_loss']),
                    'loss/entropy':                   float(losses['entropy']),
                    'policy/std':                     float(losses['std']),
                    'lr/actor':                       lr_a,
                    'lr/critic':                      lr_c,
                }, step=update)
            # End-to-end coverage comes before contacts to avoid selecting
            # low-motion policies. Preserve the guided mode's existing ranking.
            policy_score = policy_checkpoint_score(
                policy_mode, completion_rate, mean_ep_cov,
                ep_wall_mean + ep_robot_mean + ep_human_mean, mean_ep_r,
            )
            if ep_count > 0 and (
                best_policy_score is None or policy_score > best_policy_score
            ):
                best_policy_score = policy_score
                save_checkpoint(os.path.join(save_dir, checkpoint_name),
                                update, actor_state, critic_state, carry.rms,
                                tail_dim, guide_bonus, policy_mode, reward_weights)
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
                    tail_dim, guide_bonus, policy_mode, reward_weights)
    if run is not None:
        run.finish()
    return actor_state, critic_state, carry.rms


if __name__ == '__main__':
    _default_cfg  = os.path.join(os.path.dirname(__file__), '..', 'config',
                                 'mappo_baseline.yaml')
    _default_save = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

    parser = argparse.ArgumentParser(
        description='Train guided, feed-forward end-to-end, or recurrent MAPPO')
    parser.add_argument('--policy-mode', choices=['guided', 'end-to-end', 'end-to-end-memory'],
                        default='guided', help='Policy architecture and guidance regime; '
                             'end-to-end-memory uses a GRU actor')
    parser.add_argument('--config',   default=_default_cfg,
                        help='Path to YAML config file')
    parser.add_argument('--save-dir', default=_default_save,
                        help='Directory for checkpoints and training log')
    parser.add_argument('--resume',   default=None,
                        help='Checkpoint with matching actor architecture to resume from')
    parser.add_argument('--guide-bonus', type=float, default=10.0,
                        help='Reward paid to a robot that enters the cell BOSCO '
                             'pointed it at. Keep it below the environment\'s '
                             'alpha (10.0): the waypoint is usually an '
                             'undiscovered cell, so the two are collected '
                             'together and a larger bonus would make following '
                             'the tour worth more than covering the map')
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
    args = parser.parse_args()
    train(args.config, args.save_dir, args.resume,
          None if args.backend == 'auto' else args.backend,
          args.guide_bonus,
          wandb_overrides={
              'enabled': args.wandb_enabled,
              'project': args.wandb_project,
              'entity':  args.wandb_entity,
              'name':    args.wandb_name,
              'group':   args.wandb_group,
              'mode':    args.wandb_mode,
          }, num_humans=args.humans, num_envs=args.envs, policy_mode=args.policy_mode)
