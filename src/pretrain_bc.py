#!/usr/bin/env python3
"""Imitation pre-training of the end-to-end actor, then a critic warm-up.

    python -m src.pretrain_bc --config config/mappo_sequential.yaml --save-dir checkpoints/bc
    python -m src.pretrain_bc --config config/mappo_sequential.yaml --expert-only
    python -m src.train_marl  --config config/mappo_sequential.yaml \
        --resume checkpoints/bc/checkpoint_bc.pkl --save-dir checkpoints/bc_rl

Phase 1, DAgger. Each iteration rolls out the team with every robot executing
the BCD expert's action with probability beta and the actor's deterministic
action otherwise; the expert labels every visited state. Beta decays from 1
to 0, so later data comes from the states the actor itself reaches. The actor
mean is regressed on the labels (MSE on tanh(mean), the recovery imitation
loss) over a replay of the last rollouts. Labels that differ from the
previous command (turns, lane shifts, stops) are up-weighted: the actor sees
its own velocity, and without the weight it can fit most steps by repeating
it. The environment's recovery fallback is off; the expert replaces it. The
log std is not trained, so RL starts with its usual exploration noise.

With `--policy-mode end-to-end-memory` the GRU actor is cloned on ordered
sequences: the replay keeps each rollout's memory every `sequence_length`
steps, and every gradient step replays windows of that length from their
stored memory (truncated BPTT, memory reset at episode ends), so the GRU
learns what to keep instead of seeing isolated states.

Phase 2, critic warm-up. With the actor frozen, the RL setup (reward,
fallback, discount, GAE) collects rollouts of the BC policy and only the
critic is fitted, so PPO starts from V of the policy it updates rather than
from a random critic. The observation normaliser is frozen at its BC value.

The final checkpoint has the train_marl format with update 0, so RL resumes
from it. Set train.bc_coef to keep a decaying BC term during RL.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.algorithms.mappo import (compute_gae, recurrent_actor_sequence, rms_init, rms_normalize,
                                  rms_update)
from src.envs.bcd_expert import BCDExpert, expert_steps, first_episode_summary
from src.envs.vec_env import VecEnv
from src.train_marl import (Rollout, build_env_config, build_learner, observation_config,
                            save_checkpoint)
from src.train_simple import init_wandb, wandb
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device

PRETRAIN_DEFAULTS = {
    'num_envs': 32,
    'bc_iterations': 300,
    'rollout_steps': 128,
    'beta_decay_iterations': 150,  # DAgger mixing probability decays 1 -> 0
    'replay_rollouts': 32,         # aggregated DAgger data; ~37 MB per rollout, bit-packed
    'grad_steps': 48,              # actor steps per iteration
    'minibatch_size': 1024,
    'lr': 3.0e-4,
    'max_grad_norm': 10.0,
    'decision_weight': 2.0,        # extra weight x min(|label - previous command|, 1)
    'decision_threshold': 0.2,     # label change counted as a decision point in the logs
    'eval_interval': 25,           # iterations between closed-loop evaluations
    'eval_envs': 16,
    'critic_warmup_updates': 100,
    'local_labels': True,          # imitate only decisions whose target lies inside the actor crop
    'rl_init_std': 0.3,            # pre-tanh exploration std handed to RL (actor init: 1.0)
    'sequence_length': 32,         # recurrent actor: BPTT window, must divide rollout_steps
    'expert': {},
}


def with_policy_std(params, std: float, log_std_min: float, log_std_max: float):
    """Set the state-independent std; see Actor's sigmoid-bounded log_std."""
    fraction = (np.log(std) - log_std_min) / (log_std_max - log_std_min)
    if not 0. < fraction < 1.:
        raise ValueError(f'rl_init_std {std} is outside the actor std bounds')
    raw = jnp.full_like(params['params']['log_std_raw'], np.log(fraction / (1. - fraction)))
    return {**params, 'params': {**params['params'], 'log_std_raw': raw}}


def _map_ids(num_envs: int, num_maps: int) -> jnp.ndarray:
    return jnp.asarray(np.linspace(0, num_maps - 1, num_envs).round().astype(np.int32))


# ---------------------------------------------------------------------------
# Phase 1: DAgger data collection and actor regression
# ---------------------------------------------------------------------------

class BCCarry(NamedTuple):
    state: object
    obs: jax.Array
    rms: object
    chunk: jax.Array
    memory: object = None   # (E*N, H) GRU state, recurrent actor only


def actor_step(actor, params, obs: jax.Array, memory):
    """Mean action for a flat (B, D) batch; returns (mean, next memory or None)."""
    if actor.recurrent:
        mean, _, memory = actor.apply(params, obs, memory)
        return mean, memory
    mean, _ = actor.apply(params, obs)
    return mean, None


def reset_memory(memory, done: jax.Array, n: int):
    """Zero the GRU state of every robot in an environment that just ended."""
    if memory is None:
        return None
    return jnp.where(jnp.repeat(done, n)[:, None], 0., memory)


def make_collect(actor, vec_env: VecEnv, expert: BCDExpert, num_steps: int, decision_weight: float,
                 local_labels: bool):
    """Jitted DAgger rollout: (params, carry, beta, key) -> (carry, data).

    With `local_labels`, a label counts only when the expert's target lies in
    the actor crop: choosing a distant target means planning over the whole
    map, which is left to RL and the recovery fallback.
    """
    e, n = vec_env.E, vec_env.num_robots
    env = vec_env.env
    expert_act = jax.vmap(expert.act)
    expert_local = jax.vmap(expert.local)

    def step(params, beta, carry: BCCarry, key):
        state, obs, rms, chunk, memory = carry
        rms = rms_update(rms, obs.reshape(e * n, -1))
        mean, next_memory = actor_step(actor, params, rms_normalize(rms, obs).reshape(e * n, -1),
                                       memory)
        policy = jnp.tanh(mean).reshape(e, n, -1)
        label, chunk, filtered = expert_act(state, chunk)
        local = expert_local(state, chunk)
        use_expert = jax.random.uniform(key, (e, n)) < beta
        executed = jnp.where(use_expert[..., None], label, policy)
        previous = jnp.stack([2. * state.robot_velocities[..., 0] / env.v_max - 1.,
                              state.robot_velocities[..., 1] / env.omega_max], axis=-1)
        change = jnp.linalg.norm(label - previous, axis=-1)
        alive = state.robot_alive.astype(jnp.float32)
        if local_labels:
            alive = alive * local
        state, obs_next, _, _, done, info, _ = vec_env.step(state, executed)
        data = dict(
            obs=obs, label=label, change=change,
            weight=alive * (1. + decision_weight * jnp.minimum(change, 1.)),
            coverage=info['coverage_ratio'], done=done.astype(jnp.float32),
            complete=info['complete'], wall=info['wall_collision_rate'],
            robot=info['robot_collision_rate'],
            filtered=filtered.astype(jnp.float32), expert_used=use_expert.astype(jnp.float32),
            local=local.astype(jnp.float32),
        )
        if memory is not None:
            data['memory'] = memory    # state before this step, for sequence replay
        return BCCarry(state, obs_next, rms, chunk, reset_memory(next_memory, done, n)), data

    @jax.jit
    def collect(params, carry, beta, key):
        return jax.lax.scan(partial(step, params, beta), carry, jax.random.split(key, num_steps))
    return collect


def make_evaluate(actor, vec_env: VecEnv, num_steps: int):
    """Jitted closed-loop run of the deterministic actor, statistics only."""
    e, n = vec_env.E, vec_env.num_robots

    def step(params, rms, carry, _):
        state, obs, memory = carry
        mean, memory = actor_step(actor, params, rms_normalize(rms, obs).reshape(e * n, -1), memory)
        state, obs, _, _, done, info, _ = vec_env.step(state, jnp.tanh(mean).reshape(e, n, -1))
        return (state, obs, reset_memory(memory, done, n)), dict(coverage=info['coverage_ratio'], done=done,
                                  complete=info['complete'], wall=info['wall_collision_rate'],
                                  robot=info['robot_collision_rate'],
                                  filtered=jnp.zeros((e,), jnp.float32))

    @jax.jit
    def run(params, rms, carry):
        return jax.lax.scan(partial(step, params, rms), carry, None, length=num_steps)
    return run


def run_first_episodes(run_chunk, carry, total_steps: int, chunk_steps: int) -> dict:
    stats = []
    for _ in range(-(-total_steps // chunk_steps)):
        carry, out = run_chunk(carry)
        stats.append(jax.device_get(out))
    stacked = {k: np.concatenate([s[k] for s in stats])[:total_steps] for k in stats[0]}
    return first_episode_summary(stacked, total_steps)


def summarise(label: str, s: dict) -> dict:
    result = {'coverage': float(s['coverage'].mean()), 'complete': float(s['complete'].mean()),
              'steps': float(s['steps'].mean()),
              'contacts': float((s['wall'] + s['robot']).mean()),
              'filtered': float(s['filtered'].mean())}
    print(f"  {label:7s} coverage={result['coverage']:.2%} (min {s['coverage'].min():.2%}) "
          f"complete={result['complete']:.2%} steps={result['steps']:.0f} "
          f"contacts/ep={result['contacts']:.2f} safety_filter={result['filtered']:.2%}", flush=True)
    return result


class ReplayBuffer:
    """The last `slots` rollouts on device, as flat (sample, ...) arrays.

    Observations are split at `split`: the continuous prefix stays float32
    and the binary tail (crops and personal maps) is bit-packed, about 14x
    smaller at the default observation, so many more rollouts fit.

    Sample i of slot s is step t, env e, robot r with
    i = s * per_slot + (t * E + e) * N + r, so ordered sequences can be read
    back. With a 'memory' template the GRU state is kept every
    `sequence_length` steps, and 'done' per sample for the memory resets.
    """

    def __init__(self, slots: int, template: dict, split: int, binary_tail: bool,
                 sequence_length: int = 0):
        self.slots = slots
        self.split = split
        self.shape = template['label'].shape[:-1]                # (T, E, N)
        self.per_slot = int(np.prod(self.shape))
        self.sequence_length = sequence_length
        obs_dim = template['obs'].shape[-1]
        self.tail = obs_dim - split if binary_tail else 0
        size = slots * self.per_slot
        self.data = {
            'head': jnp.zeros((size, obs_dim - self.tail), jnp.float32),
            'bits': jnp.zeros((size, -(-self.tail // 8)), jnp.uint8),
            **{k: jnp.zeros((size, *template[k].shape[3:]), template[k].dtype)
               for k in ('label', 'weight', 'change')},
        }
        self.keys = ('obs', 'label', 'weight', 'change')
        if 'memory' in template:
            t, e, n = self.shape
            if not sequence_length or t % sequence_length:
                raise ValueError('sequence_length must divide rollout_steps')
            hidden = template['memory'].shape[-1]
            self.data['done'] = jnp.zeros((size,), jnp.float32)
            self.data['memory'] = jnp.zeros((slots, t // sequence_length, e * n, hidden), jnp.float32)
            self.keys += ('done', 'memory')
        self.next = 0
        self.filled = 0
        head_dim = obs_dim - self.tail

        @partial(jax.jit, donate_argnums=(0,))
        def insert(data, rollout, start, slot):
            obs = rollout['obs'].reshape(-1, obs_dim)
            packed = {'head': obs[:, :head_dim],
                      'bits': jnp.packbits(obs[:, head_dim:] > .5, axis=-1),
                      **{k: rollout[k].reshape(-1, *data[k].shape[1:])
                         for k in ('label', 'weight', 'change')}}
            if 'memory' in data:
                packed['done'] = jnp.broadcast_to(rollout['done'][..., None], self.shape).reshape(-1)
            data = {**data, **{k: jax.lax.dynamic_update_slice_in_dim(data[k], v, start, axis=0)
                               for k, v in packed.items()}}
            if 'memory' in data:
                data['memory'] = data['memory'].at[slot].set(
                    rollout['memory'][::self.sequence_length])
            return data
        self._insert = insert

    def observations(self, data: dict, idx: jax.Array) -> jax.Array:
        """Unpacked float32 observations for sample indices (traceable)."""
        if not self.tail:
            return data['head'][idx]
        bits = jnp.unpackbits(data['bits'][idx], axis=-1, count=self.tail)
        return jnp.concatenate([data['head'][idx], bits.astype(jnp.float32)], axis=-1)

    def add(self, rollout: dict) -> None:
        self.data = self._insert(self.data, {k: rollout[k] for k in self.keys},
                                 self.next * self.per_slot, self.next)
        self.next = (self.next + 1) % self.slots
        self.filled = min(self.filled + 1, self.slots)


def sample_windows(data: dict, count, key, shape: tuple, length: int, num_seq: int, unpack):
    """Random ordered windows from a ReplayBuffer's data, for the recurrent actor.

    Each window is `length` consecutive steps of one rollout slot (< count)
    and one environment, all N robots, starting at a stored-memory boundary.
    Returns obs (L, S, N, D), label (L, S, N, 2), weight and change (L, S, N),
    done (L, S) and the GRU state at the window start (S*N, H).
    """
    t_steps, e_envs, n = shape
    k_slot, k_window, k_env = jax.random.split(key, 3)
    slot = jax.random.randint(k_slot, (num_seq,), 0, count)
    window = jax.random.randint(k_window, (num_seq,), 0, t_steps // length)
    env = jax.random.randint(k_env, (num_seq,), 0, e_envs)
    t = window[None, :] * length + jnp.arange(length)[:, None]                     # (L, S)
    idx = ((slot * t_steps * e_envs * n)[None, :, None]
           + ((t * e_envs + env[None, :]) * n)[..., None]
           + jnp.arange(n)[None, None, :])                                          # (L, S, N)
    flat = idx.reshape(-1)
    obs = unpack(data, flat).reshape(*idx.shape, -1)
    memory = data['memory'][slot[:, None], window[:, None],
                            env[:, None] * n + jnp.arange(n)[None, :]]               # (S, N, H)
    return (obs, *(data[k][flat].reshape(*idx.shape, *data[k].shape[1:])
                   for k in ('label', 'weight', 'change')),
            data['done'][idx[..., 0]], memory.reshape(num_seq * n, -1))


def make_regress(actor, tx, grad_steps: int, batch: int, decision_threshold: float, unpack,
                 buffer_shape: tuple = (), sequence_length: int = 0):
    """Jitted actor regression on replay samples; `unpack(data, idx)` gives observations.

    Feed-forward: `count` is the number of filled samples, drawn independently.
    Recurrent: `count` is the number of filled rollouts; each step replays
    windows of `sequence_length` ordered steps (all robots of an environment)
    from the GRU state stored at the window start.
    """
    if actor.recurrent:
        num_seq = max(batch // (sequence_length * buffer_shape[2]), 1)

    def losses(params, rms, obs, label, weight, change):
        mean, _ = actor.apply(params, rms_normalize(rms, obs))
        return mean_losses(mean, label, weight, change)

    def sequence_losses(params, rms, obs, label, weight, change, done, memory):
        """obs (L, S, N, D), done (L, S), memory (S*N, H): one window per S."""
        _, (mean, _) = recurrent_actor_sequence(actor, params, rms_normalize(rms, obs), memory, done)
        return mean_losses(mean, *(x.reshape(mean.shape[0], *x.shape[3:])
                                   for x in (label, weight, change)))

    def mean_losses(mean, label, weight, change):
        error = jnp.mean((jnp.tanh(mean) - label) ** 2, axis=-1)
        loss = jnp.sum(error * weight) / jnp.maximum(jnp.sum(weight), 1e-6)
        valid = (weight > 0).astype(jnp.float32)
        decision = valid * (change > decision_threshold)
        steady = valid - decision
        return loss, (jnp.sum(error * decision) / jnp.maximum(decision.sum(), 1.),
                      jnp.sum(error * steady) / jnp.maximum(steady.sum(), 1.))

    @partial(jax.jit, donate_argnums=(0, 1))
    def regress(params, opt_state, data, count, rms, key):
        def body(carry, k):
            params, opt_state = carry
            if actor.recurrent:
                loss_fn = sequence_losses
                args = sample_windows(data, count, k, buffer_shape, sequence_length, num_seq, unpack)
            else:
                idx = jax.random.randint(k, (batch,), 0, count)
                loss_fn = losses
                args = (unpack(data, idx), data['label'][idx], data['weight'][idx],
                        data['change'][idx])
            (loss, (decision, steady)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, rms, *args)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), (loss, decision, steady)
        (params, opt_state), out = jax.lax.scan(body, (params, opt_state),
                                                jax.random.split(key, grad_steps))
        return params, opt_state, jax.tree_util.tree_map(jnp.mean, out)

    @jax.jit
    def fresh_loss(params, rollout, rms, key):
        """Loss on a sample of a rollout the actor has not been trained on yet."""
        if actor.recurrent:
            loss, (decision, steady) = sequence_losses(
                params, rms, *(rollout[k] for k in ('obs', 'label', 'weight', 'change', 'done')),
                rollout['memory'][0])
            return loss, decision, steady
        flat = {k: rollout[k].reshape(-1, *rollout[k].shape[3:])
                for k in ('obs', 'label', 'weight', 'change')}
        idx = jax.random.randint(key, (2048,), 0, flat['label'].shape[0])
        loss, (decision, steady) = losses(params, rms, *(flat[k][idx] for k in (
            'obs', 'label', 'weight', 'change')))
        return loss, decision, steady
    return regress, fresh_loss


def rollout_episodes(data: dict) -> tuple[int, float, float]:
    """Episodes ending in a DAgger rollout: count, mean final coverage, completion rate."""
    done = np.asarray(data['done']) > .5
    if not done.any():
        return 0, float('nan'), float('nan')
    return (int(done.sum()), float(np.asarray(data['coverage'])[done].mean()),
            float(np.asarray(data['complete'])[done].mean()))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def pretrain(config_path: str, save_dir: str, backend: str | None = None,
             num_envs: int | None = None, num_maps: int | None = None,
             bc_iterations: int | None = None, critic_updates: int | None = None,
             expert_only: bool = False, wandb_overrides: dict | None = None,
             algo: str = 'mappo', policy_mode: str = 'end-to-end') -> None:
    if policy_mode not in ('end-to-end', 'end-to-end-memory'):
        raise ValueError(f'Unknown policy mode: {policy_mode}')
    config = load_config(config_path)
    pcfg = {**PRETRAIN_DEFAULTS, **config.get('pretrain', {})}
    unknown = set(pcfg) - set(PRETRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown pretrain keys: {sorted(unknown)}')
    if num_envs is not None:
        pcfg['num_envs'] = num_envs
    if bc_iterations is not None:
        pcfg['bc_iterations'] = bc_iterations
    if critic_updates is not None:
        pcfg['critic_warmup_updates'] = critic_updates
    device = select_device(backend)
    print(f"Device: {describe(device)}")

    # Same observation and reward as RL; the expert replaces the fallback.
    bc_config = copy.deepcopy(config)
    env_cfg, reward_weights = build_env_config(bc_config, num_maps)
    env_cfg['fallback_enabled'] = False
    E = int(pcfg['num_envs'])
    vec_env = VecEnv(E, env_cfg)
    env = vec_env.env
    t0 = time.time()
    expert = BCDExpert(env, pcfg['expert'])
    print(f"BCD plans for {env.num_maps} maps in {time.time() - t0:.1f}s | "
          f"obs_dim {env.obs_dim} | envs {E} x robots {env.num_robots}")

    eval_env = VecEnv(min(int(pcfg['eval_envs']), env.num_maps), env_cfg)
    rl_config = copy.deepcopy(config)
    rl_env_cfg, _ = build_env_config(rl_config, num_maps)
    rl_eval_env = VecEnv(eval_env.E, rl_env_cfg)
    eval_maps = _map_ids(eval_env.E, env.num_maps)
    eval_key = jax.random.PRNGKey(12345)
    chunk_steps = 500

    def eval_reset(venv=eval_env):
        state, obs, _, _ = venv.reset(eval_key, eval_maps)
        return state, obs

    expert_run = expert_steps(expert, eval_env, chunk_steps)
    state, _ = eval_reset()
    print(f"Expert alone, first episode on {eval_env.E} maps:")
    expert_summary = summarise('expert', run_first_episodes(
        expert_run, (state, expert.init_chunks((eval_env.E,))), env.max_steps, chunk_steps))
    if expert_only:
        full = VecEnv(env.num_maps, env_cfg)
        state, _, _, _ = full.reset(eval_key, jnp.arange(env.num_maps))
        print(f"Expert alone on all {env.num_maps} maps:")
        summarise('expert', run_first_episodes(
            expert_steps(expert, full, chunk_steps),
            (state, expert.init_chunks((env.num_maps,))), env.max_steps, chunk_steps))
        return

    model_cfg = config.get('model', {})
    train_cfg = dict(config.get('train', {}))
    train_cfg['num_envs'] = E
    if E % int(train_cfg.get('num_minibatches', 1)):
        train_cfg['num_minibatches'] = 1
    mappo, actor_config = build_learner(vec_env, model_cfg, train_cfg, policy_mode, algo, device)
    key = jax.random.PRNGKey(int(train_cfg.get('seed', 0)))
    key, init_key, reset_key = jax.random.split(key, 3)
    actor_state, critic_state = mappo.create_train_states(init_key)
    actor = mappo.actor
    params = actor_state.params

    tx = optax.chain(optax.clip_by_global_norm(float(pcfg['max_grad_norm'])),
                     optax.adam(float(pcfg['lr'])))
    opt_state = tx.init(params)
    T = int(pcfg['rollout_steps'])
    collect = make_collect(actor, vec_env, expert, T, float(pcfg['decision_weight']),
                           bool(pcfg['local_labels']))
    evaluate = make_evaluate(actor, eval_env, chunk_steps)
    evaluate_rl = make_evaluate(actor, rl_eval_env, chunk_steps)

    def initial_memory(envs):
        return (jnp.zeros((envs * env.num_robots, actor.hidden_size), jnp.float32)
                if actor.recurrent else None)

    state, obs, _, _ = vec_env.reset(reset_key)
    carry = BCCarry(state, obs, rms_init(vec_env.norm_dim), expert.init_chunks((E,)),
                    initial_memory(E))
    # The observation tail after the normalised prefix holds 0/1 maps unless the
    # crop summary ring (means) is enabled.
    template = {'obs': jax.ShapeDtypeStruct((T, E, env.num_robots, env.obs_dim), jnp.float32),
                'label': jax.ShapeDtypeStruct((T, E, env.num_robots, 2), jnp.float32),
                'weight': jax.ShapeDtypeStruct((T, E, env.num_robots), jnp.float32),
                'change': jax.ShapeDtypeStruct((T, E, env.num_robots), jnp.float32)}
    sequence_length = int(pcfg['sequence_length']) if actor.recurrent else 0
    if actor.recurrent:
        template['memory'] = jax.ShapeDtypeStruct((T, E * env.num_robots, actor.hidden_size),
                                                  jnp.float32)
    buffer = ReplayBuffer(int(pcfg['replay_rollouts']), template, vec_env.norm_dim,
                          binary_tail=env.crop_binary, sequence_length=sequence_length)
    regress, fresh_loss = make_regress(actor, tx, int(pcfg['grad_steps']),
                                       int(pcfg['minibatch_size']), float(pcfg['decision_threshold']),
                                       buffer.observations, (T, E, env.num_robots), sequence_length)
    if actor.recurrent:
        print(f"Recurrent actor: BPTT windows of {sequence_length} steps from stored GRU state")
    print(f"Replay: {buffer.slots} rollouts, {buffer.slots * buffer.per_slot} samples, "
          f"{sum(v.nbytes for v in buffer.data.values()) / 2**30:.2f} GiB")

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'pretrain_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['phase', 'iteration', 'beta', 'loss', 'decision_loss', 'steady_loss',
                                'fresh_loss', 'fresh_decision_loss', 'episodes', 'episode_coverage',
                                'episode_completion', 'safety_filter', 'eval_coverage',
                                'eval_completion', 'eval_contacts', 'critic_loss',
                                'explained_variance'])
    run = init_wandb(config, wandb_overrides or {}, extra={'phase': 'pretrain_bc', **pcfg})

    def log(row: dict) -> None:
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(k, '') for k in (
                'phase', 'iteration', 'beta', 'loss', 'decision_loss', 'steady_loss',
                'fresh_loss', 'fresh_decision_loss', 'episodes', 'episode_coverage',
                'episode_completion', 'safety_filter', 'eval_coverage', 'eval_completion',
                'eval_contacts', 'critic_loss', 'explained_variance')])
        if run is not None:
            wandb.log({f"{row['phase']}/{k}": v for k, v in row.items()
                       if k != 'phase' and isinstance(v, (int, float))}
                      | {'env_steps': row.get('env_steps', 0)})

    iterations = int(pcfg['bc_iterations'])
    beta_decay = max(int(pcfg['beta_decay_iterations']), 1)
    best = None
    print(f"\nPhase 1: DAgger behaviour cloning, {iterations} iterations of {T} steps")
    for it in range(1, iterations + 1):
        beta = max(0.0, 1.0 - (it - 1) / beta_decay)
        key, k_collect, k_train, k_fresh = jax.random.split(key, 4)
        carry, data = collect(params, carry, jnp.float32(beta), k_collect)
        rms = carry.rms
        fresh = fresh_loss(params, data, rms, k_fresh) if it > 1 else (jnp.nan,) * 3
        buffer.add(data)
        params, opt_state, (loss, decision, steady) = regress(
            params, opt_state, buffer.data,
            buffer.filled if actor.recurrent else buffer.filled * buffer.per_slot, rms, k_train)

        episodes, ep_cov, ep_done = rollout_episodes(data)
        row = dict(phase='bc', iteration=it, beta=beta, loss=float(loss),
                   decision_loss=float(decision), steady_loss=float(steady),
                   fresh_loss=float(fresh[0]), fresh_decision_loss=float(fresh[1]),
                   episodes=episodes, episode_coverage=ep_cov, episode_completion=ep_done,
                   safety_filter=float(jnp.mean(data['filtered'])),
                   local_fraction=float(jnp.mean(data['local'])),
                   env_steps=it * T * E)
        if it % int(pcfg['eval_interval']) == 0 or it == iterations:
            print(f"Iteration {it} closed-loop evaluation (beta=0):")
            state, eval_obs = eval_reset()
            summarise('actor', run_first_episodes(lambda c: evaluate(params, rms, c),
                                                  (state, eval_obs, initial_memory(eval_env.E)),
                                                  env.max_steps, chunk_steps))
            # As RL will run it: recovery fallback as configured. Selects the best actor.
            state, eval_obs = eval_reset(rl_eval_env)
            s = summarise('actor+rl', run_first_episodes(lambda c: evaluate_rl(params, rms, c),
                                                         (state, eval_obs, initial_memory(rl_eval_env.E)),
                                                         env.max_steps, chunk_steps))
            row.update(eval_coverage=s['coverage'], eval_completion=s['complete'],
                       eval_contacts=s['contacts'])
            score = (s['complete'], s['coverage'], -s['contacts'])
            if best is None or score > best[0]:
                best = (score, jax.device_get(params), jax.device_get(rms))
        print(f"BC {it:4d}/{iterations} | beta={beta:.2f} | loss={row['loss']:.4f} "
              f"(decision {row['decision_loss']:.4f}, steady {row['steady_loss']:.4f}) | "
              f"fresh={row['fresh_loss']:.4f} | episodes={episodes} cov={ep_cov:.2%} "
              f"done={ep_done:.2%} | local={row['local_fraction']:.1%} | "
              f"filter={row['safety_filter']:.2%}", flush=True)
        log(row)

    # Continue from the best closed-loop actor and its normaliser.
    _, best_params, best_rms = best
    params = jax.device_put(best_params, device)
    rms = jax.device_put(best_rms, device)
    # The critic is fitted to, and PPO starts from, this exploration noise.
    params = with_policy_std(params, float(pcfg['rl_init_std']),
                             actor_config['log_std_min'], actor_config['log_std_max'])
    actor_state = actor_state.replace(params=params, opt_state=actor_state.tx.init(params), step=0)
    print(f"Best BC actor: completion={best[0][0]:.2%} coverage={best[0][1]:.2%} "
          f"(expert: completion={expert_summary['complete']:.2%} "
          f"coverage={expert_summary['coverage']:.2%})")

    # Phase 2 runs in the RL environment: fallback and reward as configured.
    rl_train_cfg = dict(rl_config.get('train', {}))
    rl_E = int(rl_train_cfg.get('num_envs', E))
    rl_env = VecEnv(rl_E, rl_env_cfg)
    rl_mappo, _ = build_learner(rl_env, model_cfg, rl_train_cfg, policy_mode, algo, device)
    obs_config = observation_config(rl_env.env)
    tail_dim = rl_env.env.patch_dim
    save_checkpoint(os.path.join(save_dir, 'checkpoint_bc_actor.pkl'), 0, actor_state,
                    critic_state, rms, tail_dim, policy_mode, reward_weights, obs_config,
                    algo, actor_config)

    updates = int(pcfg['critic_warmup_updates'])
    rl_T = int(rl_train_cfg.get('rollout_steps', 256))
    gamma = float(rl_train_cfg.get('gamma', 0.99))
    gae_lambda = float(rl_train_cfg.get('gae_lambda', 0.95))
    lr_critic = float(rl_train_cfg.get('lr_critic', 1e-3))
    print(f"\nPhase 2: critic warm-up, {updates} updates of {rl_T} steps x {rl_E} envs, actor frozen")
    rollout = Rollout(rl_mappo, rl_env)
    key, reset_key = jax.random.split(key)
    rl_carry = rollout.start(reset_key)._replace(rms=rms)
    for u in range(1, updates + 1):
        key, k = jax.random.split(key)
        rl_carry, traj, last_value = rollout.run(actor_state.params, critic_state.params,
                                                 rl_carry, rl_T, k)
        rl_carry = rl_carry._replace(rms=rms)
        advantages, returns = compute_gae(traj, last_value, gamma, gae_lambda)
        residual = jnp.var(returns - traj.value)
        explained = float(1. - residual / jnp.maximum(jnp.var(returns), 1e-8))
        # A zero actor rate leaves the parameters unchanged; its optimiser
        # moments are reset below so RL starts from a clean Adam state.
        actor_state, critic_state, metrics = rl_mappo.update(
            actor_state, critic_state, traj, advantages, returns, 0.0, lr_critic)
        row = dict(phase='critic', iteration=u, critic_loss=float(metrics['critic_loss']),
                   explained_variance=explained, env_steps=u * rl_T * rl_E,
                   episode_coverage=float(jnp.mean(traj.coverage[-1])))
        if u % 10 == 0 or u == updates:
            print(f"Critic {u:4d}/{updates} | loss={row['critic_loss']:.4f} | "
                  f"explained_variance={explained:.3f} | coverage={row['episode_coverage']:.2%}",
                  flush=True)
        log(row)

    # update() donates its inputs; the returned actor parameters are unchanged.
    actor_state = actor_state.replace(opt_state=actor_state.tx.init(actor_state.params), step=0)
    path = os.path.join(save_dir, 'checkpoint_bc.pkl')
    save_checkpoint(path, 0, actor_state, critic_state, rms, tail_dim, policy_mode,
                    reward_weights, obs_config, algo, actor_config)
    print(f"\nSaved {path}. Resume RL with:\n  python -m src.train_marl --config {config_path} "
          f"--policy-mode {policy_mode} --resume {path} --save-dir <dir>")
    if run is not None:
        run.finish()


if __name__ == '__main__':
    _default_cfg = os.path.join(os.path.dirname(__file__), '..', 'config', 'mappo_sequential.yaml')
    parser = argparse.ArgumentParser(description='BCD imitation pre-training and critic warm-up')
    parser.add_argument('--config', default=_default_cfg)
    parser.add_argument('--save-dir', default=os.path.join(os.path.dirname(__file__), '..',
                                                           'checkpoints', 'bc'))
    parser.add_argument('--backend', default='auto',
                        choices=['auto', 'metal', 'cuda', 'gpu', 'cpu'])
    parser.add_argument('--envs', type=int, default=None, help='DAgger environments')
    parser.add_argument('--maps', type=int, default=None, help='Procedural map-bank size')
    parser.add_argument('--bc-iterations', type=int, default=None)
    parser.add_argument('--critic-updates', type=int, default=None)
    parser.add_argument('--algo', choices=['mappo', 'ippo'], default='mappo')
    parser.add_argument('--policy-mode', choices=['end-to-end', 'end-to-end-memory'],
                        default='end-to-end', help='end-to-end-memory clones a GRU actor on sequences')
    parser.add_argument('--expert-only', action='store_true',
                        help='Only evaluate the expert on every map and exit')
    parser.add_argument('--wandb', dest='wandb_enabled', action='store_true', default=None)
    parser.add_argument('--no-wandb', dest='wandb_enabled', action='store_false', default=None)
    parser.add_argument('--wandb-name', default=None)
    parser.add_argument('--wandb-group', default=None)
    args = parser.parse_args()
    pretrain(args.config, args.save_dir, None if args.backend == 'auto' else args.backend,
             args.envs, args.maps, args.bc_iterations, args.critic_updates, args.expert_only,
             {'enabled': args.wandb_enabled, 'name': args.wandb_name, 'group': args.wandb_group},
             args.algo, args.policy_mode)
