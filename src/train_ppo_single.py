#!/usr/bin/env python3
"""Recurrent PPO for single-robot coverage, fully jitted in JAX.

One update = a `lax.scan` rollout over `num_envs` vmapped environments, GAE,
and minibatch epochs that replay full rollout sequences through the GRU.

    python -m src.train_ppo_single --config config/single_robot_ppo.yaml --save-dir checkpoints/ppo_rl
    python -m src.train_ppo_single --resume checkpoints/ppo_rl/latest.pkl --save-dir checkpoints/ppo_rl_tv
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import time
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training.train_state import TrainState

from src.envs.coverage_vector_env import MultiRobotCoverageEnv
from src.envs.curriculum_maps import NUM_STAGES, curriculum_wall_bank
from src.envs.vec_env import _select

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Grid and lidar encoders, fused with kinematics and pose, into a GRU core."""

    grid_dim: int
    n_rays: int
    grid_embed: int = 64
    lidar_embed: int = 64
    hidden_size: int = 128
    action_dim: int = 2
    log_std_min: float = -5.0
    log_std_max: float = 1.0

    @nn.compact
    def __call__(self, h, obs):
        g, r = self.grid_dim, self.grid_dim + self.n_rays
        ortho = nn.initializers.orthogonal
        grid = nn.relu(nn.Dense(self.grid_embed, kernel_init=ortho(np.sqrt(2)))(obs[..., :g]))
        lidar = nn.relu(nn.Dense(self.lidar_embed, kernel_init=ortho(np.sqrt(2)))(obs[..., g:r]))
        x = jnp.concatenate([grid, lidar, obs[..., r:]], axis=-1)   # + [v, omega, x, y, theta]
        h, y = nn.GRUCell(features=self.hidden_size)(h, x)
        out = nn.Dense(2 * self.action_dim, kernel_init=ortho(0.01))(y)
        mean, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        value = nn.Dense(1, kernel_init=ortho(1.0))(y)[..., 0]
        return h, mean, log_std, value


def gaussian_log_prob(a, mean, log_std):
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * _LOG_2PI, axis=-1)


def env_action(u, squash: bool):
    """Gaussian sample (or mean) -> env command in [-1, 1].

    squash=True: tanh, so the policy is a distribution over the commands the
    robot executes. squash=False (older checkpoints): clip, under which every
    sample beyond +-1 is the same command, so the mean can drift far outside
    and the std grow for free under the entropy bonus.
    """
    return jnp.tanh(u) if squash else jnp.clip(u, -1.0, 1.0)


def log_tanh_jacobian(u):
    """sum log(1 - tanh(u)^2), numerically stable."""
    return jnp.sum(2.0 * (np.log(2.0) - u - jax.nn.softplus(-2.0 * u)), axis=-1)


# ---------------------------------------------------------------------------
# Environment adapter (single env; vmapped by the caller)
# ---------------------------------------------------------------------------

def make_env_fns(env: MultiRobotCoverageEnv, maps_per_stage: int, stage_max_steps=None):
    """Return `reset(key, stage)` and `step(state, action, stage)` producing the flat observation.

    Stage s draws its maps from bank rows s*K:(s+1)*K (K = maps_per_stage).
    stage_max_steps: per-stage episode limit, applied by the stage of the episode's map.
    """
    stage_limit = jnp.asarray(stage_max_steps or [env.max_steps], jnp.int32)
    kin_scale = jnp.array([env.v_max, env.omega_max], jnp.float32)
    pos_scale = jnp.array([env.map_layout.width, env.map_layout.height], jnp.float32)

    def observe(s):
        theta = jnp.arctan2(jnp.sin(s.robot_headings[0]), jnp.cos(s.robot_headings[0])) / jnp.pi
        return jnp.concatenate([
            env._actor_crops(s)[0],                  # 7x7 grid channels, flattened
            s.lidar[0],                              # normalised ranges
            s.robot_velocities[0] / kin_scale,       # [v, omega]
            s.robot_positions[0] / pos_scale,        # [x, y]
            theta[None],                             # [theta]
        ])

    def sample(key, stage):
        map_key, reset_key = jax.random.split(key)
        map_id = stage * maps_per_stage + jax.random.randint(map_key, (), 0, maps_per_stage)
        return env._reset_state(reset_key, map_id)

    def reset(key, stage):
        s = env._refresh_memory(sample(key, stage), jnp.zeros((1,), bool))
        return s, observe(s)

    def step(s, action, stage):
        s, reward, term, trunc, cover = env._step_core(s, action[None])
        done = term | trunc | (s.step_count >= stage_limit[s.map_id // maps_per_stage])
        free_total = env.free_totals[s.map_id]
        covered = jnp.sum(s.coverage_grid)
        info = {
            'coverage': covered / free_total,
            # cell entries of the robot centre per covered cell; 1 = no cell passed twice
            'recoverage': jnp.where(covered > 0, s.cell_entries / jnp.maximum(covered, 1.0), 1.0),
            'complete': (jnp.sum(s.coverage_grid) >= free_total - 0.5).astype(jnp.float32),
            'collision': s.wall_hits[0],
            'length': s.step_count.astype(jnp.float32),
            'stage': s.map_id // maps_per_stage,
            # ended by the time limit, not by completion (or a collision)
            'truncated': done & ~jnp.any(term),
        }
        key, reset_key = jax.random.split(s.key)
        s = jax.tree_util.tree_map(lambda f, c: _select(done, f, c),
                                   sample(reset_key, stage), s.replace(key=key))
        s = env._refresh_memory(s, cover & ~done)
        return s, observe(s), reward[0], done, info

    return reset, step, observe


def build_env(cfg):
    """Env with the curriculum map bank when enabled; returns (env, maps_per_stage).

    Fills cfg['curriculum'] with the stage count (and neutral values when disabled).
    """
    env_cfg = {**cfg['env'], **cfg['reward']}
    c = cfg.setdefault('curriculum', {'enabled': False})
    if c['enabled']:
        cs = env_cfg['cell_size']
        env_cfg['wall_bank'], env_cfg['floor_bank'] = curriculum_wall_bank(
            c['maps_per_stage'], round(env_cfg['map_width'] / cs), round(env_cfg['map_height'] / cs),
            cs, env_cfg['robot_radius'], c['max_walls'], env_cfg.get('map_seed', 0))
        maps_per_stage, c['num_stages'] = c['maps_per_stage'], NUM_STAGES
    else:
        maps_per_stage = env_cfg['num_maps']
        c.update(num_stages=1, start_stage=0, window=1, success_coverage=0.95, max_recoverage=1.2, max_contacts=0,
                 success_rate=1.0)
    env = MultiRobotCoverageEnv(env_cfg)
    if env.num_robots != 1:
        raise SystemExit('train_ppo_single supports num_robots: 1 only')
    return env, maps_per_stage


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs: jax.Array
    reset: jax.Array      # episode started at this step: zero the GRU state first
    action: jax.Array     # Gaussian sample before tanh / clip
    eps: jax.Array        # its standard-normal noise, for the reparameterised entropy
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array     # scaled
    done: jax.Array
    ep_return: jax.Array  # raw return, valid where done
    ep_contacts: jax.Array  # wall contacts this episode, valid where done
    info: dict


class Runner(NamedTuple):
    train_state: TrainState
    env_state: object
    obs: jax.Array
    h: jax.Array
    done: jax.Array
    ep_return: jax.Array
    ep_contacts: jax.Array
    key: jax.Array
    stage: jax.Array      # current curriculum stage
    success: jax.Array    # (window,) ring of recent current-stage episode outcomes
    success_pos: jax.Array
    success_count: jax.Array


def make_update(net, env_step, cfg):
    p = cfg['ppo']
    E, T, M = p['num_envs'], p['rollout_steps'], p['num_minibatches']
    gamma, lam = p['gamma'], p['gae_lambda']
    c = cfg['curriculum']
    W, last_stage = c['window'], c['num_stages'] - 1
    squash = cfg['model'].get('squash', False)
    bootstrap = p.get('bootstrap_truncation', False)
    v_step = jax.vmap(env_step, in_axes=(0, 0, None))

    def rollout_step(r: Runner, _):
        key, k = jax.random.split(r.key)
        h = jnp.where(r.done[:, None], 0.0, r.h)
        h, mean, log_std, value = net.apply(r.train_state.params, h, r.obs)
        eps = jax.random.normal(k, mean.shape)
        action = mean + jnp.exp(log_std) * eps
        # The tanh Jacobian depends on the sample only, so it cancels in the
        # PPO ratio: the Gaussian log-probability of the pre-tanh sample suffices.
        log_prob = gaussian_log_prob(action, mean, log_std)
        env_state, obs, reward, done, info = v_step(r.env_state, env_action(action, squash), r.stage)
        ep_return = r.ep_return + reward
        ep_contacts = r.ep_contacts + info['collision']
        train_reward = reward * p['reward_scale']
        if bootstrap:
            # A time limit is not part of the task, and the observation carries
            # no clock: cut the trajectory but bootstrap as if it went on
            # (Pardo et al. 2018). V(s_t) stands in for V(s_t+1), one step apart.
            train_reward += gamma * jax.lax.stop_gradient(value) * info['truncated']
        t = Transition(r.obs, r.done, action, eps, log_prob, value,
                       train_reward, done, ep_return, ep_contacts, info)
        # Append finished episodes of the current stage to the success ring;
        # success = enough coverage, few wall contacts, few passes per cell.
        valid = done & (info['stage'] == r.stage)
        slot = jnp.where(valid, (r.success_pos + jnp.cumsum(valid) - 1) % W, W)
        n = valid.sum()
        r = r._replace(
            env_state=env_state, obs=obs, h=h, done=done, key=key,
            ep_return=jnp.where(done, 0.0, ep_return),
            ep_contacts=jnp.where(done, 0.0, ep_contacts),
            success=r.success.at[slot].set((info['coverage'] >= c['success_coverage'])
                                           & (ep_contacts <= c['max_contacts'])
                                           & (info['recoverage'] <= c['max_recoverage']), mode='drop'),
            success_pos=(r.success_pos + n) % W,
            success_count=jnp.minimum(r.success_count + n, W))
        return r, t

    def loss_fn(params, h0, mb, adv, target):
        def seq_step(h, x):
            obs, reset = x
            h, mean, log_std, value = net.apply(params, jnp.where(reset[:, None], 0.0, h), obs)
            return h, (mean, log_std, value)

        _, (mean, log_std, value) = jax.lax.scan(seq_step, h0, (mb.obs, mb.reset))
        log_prob = gaussian_log_prob(mb.action, mean, log_std)
        log_ratio = log_prob - mb.log_prob
        ratio = jnp.exp(log_ratio)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_loss = -jnp.minimum(ratio * adv,
                               jnp.clip(ratio, 1 - p['clip_eps'], 1 + p['clip_eps']) * adv).mean()
        v_loss = 0.5 * jnp.mean((value - target) ** 2)
        entropy = jnp.sum(log_std + 0.5 * (1.0 + _LOG_2PI), axis=-1)
        if squash:
            # Entropy of the executed command: Gaussian entropy + E[log |d tanh/du|],
            # estimated on the stored noise so the gradient reaches mean and std.
            entropy += log_tanh_jacobian(mean + jnp.exp(log_std) * mb.eps)
        entropy = jnp.mean(entropy)
        loss = pg_loss + p['vf_coef'] * v_loss - p['ent_coef'] * entropy
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
        return loss, (pg_loss, v_loss, entropy, approx_kl)

    def update(r: Runner):
        h0 = r.h
        r, traj = jax.lax.scan(rollout_step, r, None, T)

        _, _, _, last_value = net.apply(r.train_state.params,
                                        jnp.where(r.done[:, None], 0.0, r.h), r.obs)

        def gae_step(carry, t):
            gae, next_value = carry
            not_done = 1.0 - t.done
            delta = t.reward + gamma * next_value * not_done - t.value
            gae = delta + gamma * lam * not_done * gae
            return (gae, t.value), gae

        _, adv = jax.lax.scan(gae_step, (jnp.zeros(E), last_value), traj, reverse=True)
        target = adv + traj.value

        def epoch(carry, _):
            train_state, key = carry
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, E)
            # Minibatches over the env axis: (M, T, E/M, ...) sequences, (M, E/M, H) initial states.
            split_t = lambda x: jnp.swapaxes(x[:, perm].reshape(T, M, E // M, *x.shape[2:]), 0, 1)
            data = (h0[perm].reshape(M, E // M, -1),
                    jax.tree_util.tree_map(split_t, (traj, adv, target)))

            def minibatch(ts, batch):
                mb_h0, (mb, mb_adv, mb_target) = batch
                grads, aux = jax.grad(loss_fn, has_aux=True)(ts.params, mb_h0, mb, mb_adv, mb_target)
                return ts.apply_gradients(grads=grads), aux

            train_state, aux = jax.lax.scan(minibatch, train_state, data)
            return (train_state, key), aux

        (train_state, key), aux = jax.lax.scan(epoch, (r.train_state, r.key), None, p['update_epochs'])

        success_rate = r.success.mean()
        promote = (r.success_count >= W) & (success_rate >= c['success_rate']) & (r.stage < last_stage)

        n_done = traj.done.sum()
        # mean over episodes that ended in this rollout; nan when none ended
        at_done = lambda x: jnp.where(n_done > 0, jnp.sum(x * traj.done) / jnp.maximum(n_done, 1), jnp.nan)
        metrics = {
            'episodes': traj.done.sum(),
            'return': at_done(traj.ep_return),
            'coverage': at_done(traj.info['coverage']),
            'completion_rate': at_done(traj.info['complete']),
            'contacts': at_done(traj.ep_contacts),   # wall contacts per episode
            'length': at_done(traj.info['length']),
            'recoverage': at_done(traj.info['recoverage']),
            'stage': r.stage,
            'stage_success': jnp.where(r.success_count >= W, success_rate, jnp.nan),
            'pg_loss': aux[0].mean(),
            'v_loss': aux[1].mean(),
            'entropy': aux[2].mean(),
            'approx_kl': aux[3].mean(),
        }
        r = r._replace(train_state=train_state, key=key, stage=r.stage + promote,
                       success_count=jnp.where(promote, 0, r.success_count))
        return r, metrics

    return update


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config/single_robot_ppo.yaml')
    ap.add_argument('--save-dir', default='checkpoints/ppo_rl')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num-envs', type=int)
    ap.add_argument('--total-timesteps', type=float)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--resume', help='checkpoint (.pkl) to fine-tune: loads its network weights and '
                    'curriculum stage; optimiser, LR schedule and update count start afresh')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg['ppo']
    if args.num_envs:
        p['num_envs'] = args.num_envs
    if args.total_timesteps:
        p['total_timesteps'] = args.total_timesteps
    E, T, M = p['num_envs'], p['rollout_steps'], p['num_minibatches']
    if E % M:
        raise SystemExit('num_envs must be divisible by num_minibatches')
    num_updates = int(float(p['total_timesteps']) // (E * T))   # PyYAML reads 1e9 (no exponent sign) as a string

    env, maps_per_stage = build_env(cfg)
    c = cfg['curriculum']
    reset, env_step, _ = make_env_fns(env, maps_per_stage, c.get('max_steps'))

    m = cfg['model']
    net = ActorCritic(grid_dim=env.crop_dim, n_rays=env.n_rays, grid_embed=m['grid_embed'],
                      lidar_embed=m['lidar_embed'], hidden_size=m['hidden_size'],
                      log_std_min=m['log_std_min'], log_std_max=m['log_std_max'])
    obs_dim = env.crop_dim + env.n_rays + 5

    key = jax.random.PRNGKey(args.seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    h = jnp.zeros((E, m['hidden_size']))
    params = net.init(init_key, h, jnp.zeros((E, obs_dim)))
    steps_per_update = p['update_epochs'] * M
    lr = (optax.linear_schedule(p['lr'], 0.0, num_updates * steps_per_update)
          if p['anneal_lr'] else p['lr'])
    tx = optax.chain(optax.clip_by_global_norm(p['max_grad_norm']), optax.adam(lr, eps=1e-5))
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    if args.resume:
        with open(args.resume, 'rb') as f:
            ckpt = pickle.load(f)
        if ckpt['obs_dim'] != obs_dim:
            raise SystemExit(f"{args.resume}: obs_dim {ckpt['obs_dim']} != {obs_dim}")
        train_state = train_state.replace(params=jax.device_put(ckpt['params']))
        c['start_stage'] = ckpt['stage']
        cfg['resumed_from'] = f"{args.resume} (update {ckpt['update']})"   # recorded in the run's config.yaml
        print(f"Resumed {args.resume} (update {ckpt['update']}, stage {ckpt['stage']})")
    stage = jnp.int32(c['start_stage'])
    env_state, obs = jax.vmap(reset, in_axes=(0, None))(jax.random.split(reset_key, E), stage)
    # Random elapsed steps so first episodes end at different times; otherwise
    # fixed-length episodes keep every env resetting on the same step.
    first_limit = (c.get('max_steps') or [env.max_steps])[c['start_stage']]
    env_state = env_state.replace(step_count=jax.random.randint(key, (E,), 0, first_limit))
    runner = Runner(train_state, env_state, obs, h, jnp.ones(E, bool), jnp.zeros(E), jnp.zeros(E), key,
                    stage, jnp.zeros(c['window'], bool), jnp.int32(0), jnp.int32(0))
    update = jax.jit(make_update(net, env_step, cfg), donate_argnums=0)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)
    log_path = os.path.join(args.save_dir, 'training_log.csv')
    log_file, writer = open(log_path, 'w', newline=''), None
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=cfg.get('wandb', {}).get('project', 'mrcpp-single'), config=cfg)

    def save(name, update_idx):
        payload = {'params': jax.device_get(runner.train_state.params), 'config': cfg,
                   'obs_dim': obs_dim, 'update': update_idx, 'stage': int(runner.stage)}
        path = os.path.join(args.save_dir, name)
        with open(path + '.tmp', 'wb') as f:
            pickle.dump(payload, f)
        os.replace(path + '.tmp', path)

    print(f'{jax.devices()[0]} | {E} envs x {T} steps = {E * T} steps/update | '
          f'{num_updates} updates | obs_dim {obs_dim}')
    t0, last_steps = time.time(), 0
    for i in range(1, num_updates + 1):
        runner, metrics = update(runner)
        if i % p['log_interval'] == 0 or i == num_updates:
            metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
            env_steps = i * E * T
            sps = (env_steps - last_steps) / (time.time() - t0)   # this window only
            t0, last_steps = time.time(), env_steps
            row = {'update': i, 'env_steps': env_steps, 'sps': sps, **metrics}
            print(f"upd {i:5d} | stage {int(metrics['stage'])} ({metrics['stage_success']:.2f}) | "
                  f"steps {env_steps:.3e} | sps {sps:,.0f} | ret {metrics['return']:8.2f} | "
                  f"cov {metrics['coverage']:.3f} | done {metrics['completion_rate']:.3f} | "
                  f"col {metrics['contacts']:.2f} | rec {metrics['recoverage']:.2f} | len {metrics['length']:6.0f} | "
                  f"ent {metrics['entropy']:.2f} | kl {metrics['approx_kl']:.4f}", flush=True)
            if writer is None:
                writer = csv.DictWriter(log_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log_file.flush()
            if run is not None:
                run.log(row, step=env_steps)
        if i % p['save_interval'] == 0:
            save('latest.pkl', i)
    save('final.pkl', num_updates)
    log_file.close()
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
