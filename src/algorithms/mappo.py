"""
Multi-Agent PPO in JAX: centralised critic, parameter-shared feed-forward actor.

The whole rollout is a `lax.scan` over the vectorised environment, so a full
(T steps x E envs) batch is collected in one device call with no Python in the
inner loop. The PPO update is likewise a `lax.scan` over epochs.

Credit assignment is per agent end to end: the environment pays each robot for
its own discoveries, the critic produces an agent-specific V_i(s) from the
agent-centred global map, and GAE runs independently on every (env, agent)
stream. Nothing is averaged across agents, so a robot that free-rides on the
team's coverage sees its own advantage drop.

Shapes
------
obs       : (T, E, N, obs_dim)   — continuous prefix normalised, binary tail raw
gstate    : GlobalState pytree with leading (T, E)
action    : (T, E, N, action_dim)
log_prob  : (T, E, N)
reward    : (T, E, N)           — already multiplied by `reward_scale`
value     : (T, E, N)
term      : (T, E)               — hard termination (collision, completion)
done      : (T, E)               — terminated OR truncated

`term` and `done` are stored as float32 0/1 masks rather than bools: jax-metal
0.1.1 returns all-False for any bool-dtype `lax.scan` output, which would
silently erase every episode boundary from the trajectory (see `rollout`).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from src.utils.jax_device import init_device as _pick_init_device

_LOG_2PI = float(np.log(2.0 * np.pi))
_GH_NODES, _GH_WEIGHTS = (
    tuple(x.tolist()) for x in np.polynomial.hermite.hermgauss(8)
)


# ---------------------------------------------------------------------------
# Observation normalisation (Welford, as a JAX pytree)
# ---------------------------------------------------------------------------

class RunningMeanStd(NamedTuple):
    mean:  jax.Array
    var:   jax.Array
    count: jax.Array


def rms_init(dim: int, eps: float = 1e-4) -> RunningMeanStd:
    return RunningMeanStd(
        jnp.zeros((dim,), jnp.float32),
        jnp.ones((dim,), jnp.float32),
        jnp.float32(eps),
    )


def rms_update(rms: RunningMeanStd, x: jax.Array) -> RunningMeanStd:
    """x : (B, obs_dim) — a flat batch of observations.

    Only the continuous prefix is tracked: `rms.mean` is shorter than the
    observation whenever the environment appends a binary block.
    """
    x = x[:, : rms.mean.shape[-1]]
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = x.shape[0]

    total = rms.count + batch_count
    delta = batch_mean - rms.mean
    mean = rms.mean + delta * batch_count / total
    m2 = rms.var * rms.count + batch_var * batch_count \
        + delta ** 2 * rms.count * batch_count / total
    return RunningMeanStd(mean, m2 / total, total)


def rms_normalize(rms: RunningMeanStd, x: jax.Array) -> jax.Array:
    """Normalise the continuous prefix of x : (..., obs_dim), leading dims kept.

    The binary tail (local coverage patch) is passed through untouched. Dividing
    a 0/1 cell by its own standard deviation turns a rarely-flipping cell into a
    large spike and a constant cell into pure numerical noise, so those channels
    are better left as the indicator they already are.
    """
    n = rms.mean.shape[-1]
    head = (x[..., :n] - rms.mean) / jnp.sqrt(rms.var + 1e-8)
    head = jnp.clip(head, -10.0, 10.0)
    if x.shape[-1] == n:
        return head
    return jnp.concatenate([head, x[..., n:]], axis=-1)


# ---------------------------------------------------------------------------
# Rollout data
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:      jax.Array
    gstate:   object      # GlobalState pytree
    action:   jax.Array   # tanh(z), policy proposal (masked when recovery overrides it)
    z:        jax.Array   # pre-squash Gaussian sample, see _update
    log_prob: jax.Array
    reward:   jax.Array
    value:    jax.Array
    term:     jax.Array   # float32 0/1, not bool — see module docstring
    done:     jax.Array   # float32 0/1, not bool — see module docstring
    coverage: jax.Array
    # Diagnostics only; never read by the update. All (T, E) float32.
    wall_hit:  jax.Array  # fraction of the team that hit a wall this step
    robot_hit: jax.Array  # fraction of the team that hit another robot
    human_hit: jax.Array  # fraction of the team that hit a human this step
    complete:  jax.Array  # 1.0 when the map was fully covered on this step
    timeout:   jax.Array  # 1.0 when the step hit the truncation horizon
    memory:    object = None  # pre-observation GRU state (E*N, H), recurrent only
    policy_mask: object = None  # (T, E, N), 0 for actions overridden by recovery
    sequence_used: object = None # (T, E, N), recovery sequence executed
    safety_override: object = None # (T, E, N), sequence interrupted by live scan
    teacher_action: object = None # executed recovery command, never used as a PPO sample
    teacher_mask: object = None   # collision-free, nonstationary recovery steps


class RolloutCarry(NamedTuple):
    """State threaded across rollout steps and between successive updates."""

    env_state: object
    obs:       jax.Array
    gstate:    object      # GlobalState pytree
    rms:       RunningMeanStd


def recurrent_actor_sequence(actor, params, obs, initial_memory, done):
    """Replay ordered (T,E,N,D) inputs; reset each environment after done.

    Gradients span one rollout, with the starting memory treated as constant.
    """
    t, e, n, d = obs.shape
    def step(memory, inputs):
        observation, ended = inputs
        mean, log_std, memory = actor.apply(params, observation, memory)
        reset = jnp.repeat(ended, n)[:, None]
        memory = jnp.where(reset, 0., memory)
        return memory, (mean, log_std)
    memory, (mean, log_std) = jax.lax.scan(
        step, jax.lax.stop_gradient(initial_memory),
        (obs.reshape(t, e*n, d), done.astype(jnp.bool_)),
    )
    return memory, (mean.reshape(t*e*n, -1), log_std.reshape(t*e*n, -1))


def _tanh_normal_log_prob(
    z: jax.Array, mean: jax.Array, std: jax.Array, action: jax.Array
) -> jax.Array:
    """Log density of a tanh-squashed Normal, summed over the action dimension."""
    log_prob = -0.5 * ((z - mean) / std) ** 2 - jnp.log(std) - 0.5 * _LOG_2PI
    log_prob = log_prob - jnp.log(1.0 - action ** 2 + 1e-6)
    return jnp.sum(log_prob, axis=-1)


def _tanh_normal_entropy(mean: jax.Array, std: jax.Array) -> jax.Array:
    """Differentiable entropy of tanh(N(mean, std)) via Gauss-Hermite quadrature."""
    nodes = jnp.asarray(_GH_NODES, dtype=mean.dtype)
    weights = jnp.asarray(_GH_WEIGHTS, dtype=mean.dtype) / jnp.sqrt(jnp.pi)
    z = mean[..., None] + jnp.sqrt(2.0) * std[..., None] * nodes
    # log(1 - tanh(z)^2), written without forming a saturated tanh.
    log_jacobian = 2.0 * (jnp.log(2.0) - z - jax.nn.softplus(-2.0 * z))
    expected_log_jacobian = jnp.sum(log_jacobian * weights, axis=-1)
    normal_entropy = jnp.log(std) + 0.5 * (1.0 + _LOG_2PI)
    return jnp.sum(normal_entropy + expected_log_jacobian, axis=-1)


def policy_update_diagnostics(log_ratio, mask, clip_eps):
    """Sample KL estimate and clipping fraction, excluding overridden actions."""
    log_ratio = jnp.where(mask > 0, log_ratio, 0.)
    ratio = jnp.exp(log_ratio)
    denominator = jnp.maximum(mask.sum(), 1.)
    approx_kl = jnp.sum((jnp.expm1(log_ratio) - log_ratio) * mask) / denominator
    clip_fraction = jnp.sum((jnp.abs(ratio - 1.) > clip_eps) * mask) / denominator
    return approx_kl, clip_fraction


def recovery_imitation_loss(mean, target_action, mask):
    """Fit the deterministic policy to valid executed recovery commands."""
    error = jnp.mean((jnp.tanh(mean) - jax.lax.stop_gradient(target_action)) ** 2, axis=-1)
    return jnp.sum(error * mask) / jnp.maximum(mask.sum(), 1.)


@jax.jit
def compute_gae(
    traj: Transition, last_value: jax.Array, gamma: float, gae_lambda: float
) -> tuple[jax.Array, jax.Array]:
    """
    GAE-lambda advantages and discounted returns, independent per env AND agent.

    Every array below carries a trailing agent axis and the recursion is
    elementwise over it, so the scan is an implicit vmap over the (E, N) agent
    slots at no extra cost: agent i's advantage depends only on its own reward
    and value streams. Averaging rewards across agents here would undo the
    difference reward the environment computes.

    Feed-forward training masks only hard termination and bootstraps at a time
    limit. Recurrent training also masks time limits because the environment and
    actor memory both reset there; propagating an advantage across that boundary
    would connect two unrelated episodes. The team mask broadcasts over agents.
    """
    values = jnp.concatenate([traj.value, last_value[None]], axis=0)   # (T+1, E, N)
    reward = traj.reward                                              # (T, E, N)
    # Recurrent coverage treats the configured episode budget as a finite
    # horizon. Never bootstrap from an auto-reset episode's value or carry its
    # advantages backward across a memory reset. Preserve legacy FF semantics.
    ended = traj.done if traj.memory is not None else traj.term
    mask = (1.0 - ended.astype(jnp.float32))[:, :, None]              # (T, E, 1)

    def body(gae, xs):
        reward, value, next_value, m = xs
        delta = reward + gamma * next_value * m - value
        gae = delta + gamma * gae_lambda * m * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        body,
        jnp.zeros_like(last_value),
        (reward, values[:-1], values[1:], jnp.broadcast_to(mask, reward.shape)),
        reverse=True,
    )
    return advantages, advantages + values[:-1]


def _apply_gradients(state: TrainState, grads, lr: jax.Array) -> TrainState:
    """Adam step with an explicitly supplied (traceable) learning rate.

    `tx` holds gradient clipping and Adam's moment rescaling; multiplying by
    -lr afterwards is exactly what optax.adam's final scaling does, but keeps
    the rate a traced value so it can be decayed without rebuilding the state.
    """
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    updates = jax.tree_util.tree_map(lambda u: -lr * u, updates)
    return state.replace(
        step=state.step + 1,
        params=optax.apply_updates(state.params, updates),
        opt_state=opt_state,
    )


class MAPPO:
    """
    Multi-Agent PPO with a centralised critic and parameter-shared actor.

    Supports E parallel environments: all actor and critic forward passes are
    batched across (E x N) agent-slots in a single call, so data collection is
    as fast as a single vectorised network pass.
    """

    def __init__(
        self,
        actor,
        critic,
        vec_env,
        config: dict,
        device: jax.Device | None = None,
    ):
        self.actor = actor
        self.critic = critic
        self.env = vec_env

        self.device = device or jax.devices()[0]
        self.init_device = _pick_init_device(self.device)

        self.clip_eps      = float(config.get('clip_eps',      0.2))
        self.entropy_coef  = float(config.get('entropy_coef',  0.01))
        self.recovery_imitation_coef = float(config.get('recovery_imitation_coef', 0.0))
        if self.recovery_imitation_coef < 0:
            raise ValueError('recovery_imitation_coef must be non-negative')
        self.target_kl = float(config.get('target_kl', 0.0))
        if self.target_kl < 0:
            raise ValueError('target_kl must be non-negative (0 disables the guard)')
        self.max_grad_norm = float(config.get('max_grad_norm', 10.0))
        self.n_epochs      = int(config.get('n_epochs',      10))
        # Each epoch splits the environments into this many disjoint groups and
        # takes one gradient step per group. Splitting along E keeps every
        # rollout time-ordered, as the recurrent replay requires, and bounds the
        # update's peak memory by E / num_minibatches instead of E.
        self.num_minibatches = int(config.get('num_minibatches', 1))
        if vec_env.E % self.num_minibatches:
            raise ValueError(
                f"num_envs ({vec_env.E}) must be divisible by "
                f"num_minibatches ({self.num_minibatches})"
            )
        self.gamma         = float(config.get('gamma',         0.99))
        self.gae_lambda    = float(config.get('gae_lambda',    0.95))
        # Static, stateless alternative to return normalisation: a constant
        # factor keeps the whole pipeline a pure function of the rollout, with
        # no running statistics to thread through the scan.
        self.reward_scale  = float(config.get('reward_scale',  1.0))
        self.huber_delta   = float(config.get('huber_delta',   1.0))

        # TrainState buffers are replaced by every update. Donation lets XLA
        # reuse them instead of holding a second full optimiser/parameter copy.
        self._update_fn = jax.jit(self._update, donate_argnums=(0, 1))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def create_train_states(self, key: jax.Array) -> tuple[TrainState, TrainState]:
        """Initialise parameters (on CPU when the backend lacks QR) and optimisers."""
        actor_key, critic_key = jax.random.split(key)

        dummy_obs = jnp.zeros((1, self.env.obs_dim), jnp.float32)

        with jax.default_device(self.init_device):
            a_key, c_key, o, c_args = jax.device_put(
                (actor_key, critic_key, dummy_obs, self._dummy_critic_args()),
                self.init_device,
            )
            actor_params = self.actor.init(a_key, o)
            critic_params = self.critic.init(c_key, *c_args)

        actor_params = jax.device_put(actor_params, self.device)
        critic_params = jax.device_put(critic_params, self.device)

        # Adam's learning rate is applied in _apply_gradients so it stays traceable.
        def make_tx():
            return optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.scale_by_adam(eps=1e-5),
            )

        actor_state = TrainState.create(
            apply_fn=self.actor.apply, params=actor_params, tx=make_tx()
        )
        critic_state = TrainState.create(
            apply_fn=self.critic.apply, params=critic_params, tx=make_tx()
        )
        return actor_state, critic_state

    def init_carry(self, key: jax.Array) -> RolloutCarry:
        env_state, obs, gstate, _ = self.env.reset(key)
        # Statistics cover only the continuous prefix of the observation.
        return RolloutCarry(env_state, obs, gstate, rms_init(self.env.norm_dim))

    # The critic's input is the only difference between MAPPO and IPPO; the
    # two hooks below are what a subclass overrides.

    def _dummy_critic_args(self) -> tuple:
        env = self.env
        return (
            jnp.zeros((1, env.critic_channels, env.grid_h, env.grid_w), jnp.float32),
            jnp.zeros((1, env.critic_vec_dim), jnp.float32),
        )

    def _critic_args(self, obs_n: jax.Array, gstate) -> tuple:
        """Flat critic inputs for obs_n : (..., N, obs_dim) and matching gstate.

        MAPPO: the agent-centred global state V_i(s); obs_n is unused.
        """
        grid, vec = self.env.critic_inputs(gstate)
        return (grid.reshape(-1, *grid.shape[-3:]), vec.reshape(-1, vec.shape[-1]))

    def _values(self, critic_params, obs_n: jax.Array, gstate) -> jax.Array:
        """Value of every agent slot, shaped like obs_n without its last axis."""
        value = self.critic.apply(critic_params, *self._critic_args(obs_n, gstate))
        return value.reshape(obs_n.shape[:-1])

    # ------------------------------------------------------------------
    # Rollout collection (single device call for T steps)
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0, 4))
    def rollout(
        self,
        actor_params,
        critic_params,
        carry: RolloutCarry,
        num_steps: int,
        key: jax.Array,
    ) -> tuple[RolloutCarry, Transition, jax.Array]:
        """Collect `num_steps` transitions across all E envs.

        Takes raw parameters rather than TrainStates so the optimiser state is
        not passed through the jit boundary (jax-metal rejects the resulting
        signature).

        Returns the updated carry, the stacked trajectory, and the bootstrap
        value V(s_T) for each env.
        """
        e, n = self.env.E, self.env.num_robots

        def step(carry: RolloutCarry, step_key: jax.Array):
            # Statistics are refreshed on the raw observation before it is
            # normalised, matching the order used during evaluation.
            rms = rms_update(carry.rms, carry.obs.reshape(e * n, -1))
            obs_n = rms_normalize(rms, carry.obs)

            mean, log_std = self.actor.apply(actor_params, obs_n.reshape(e * n, -1))
            std = jnp.exp(log_std)
            z = mean + std * jax.random.normal(step_key, mean.shape)
            action = jnp.tanh(z)
            log_prob = _tanh_normal_log_prob(z, mean, std, action).reshape(e, n)
            action = action.reshape(e, n, -1)
            z = z.reshape(e, n, -1)

            value = self._values(critic_params, obs_n, carry.gstate)

            env_state, next_obs, reward, term, done, info, next_gstate = self.env.step(
                carry.env_state, action
            )

            # Emitted as float32 masks: jax-metal 0.1.1 returns all-False for
            # bool-dtype scan outputs, which would drop every episode boundary.
            transition = Transition(
                obs=obs_n,
                gstate=carry.gstate,
                action=action,
                z=z,
                log_prob=log_prob,
                # Scaled once, here: everything downstream (GAE, returns, critic
                # targets) inherits the smaller range, and the environment keeps
                # its reward weights in physically meaningful units. Callers that
                # report episode reward must divide by the same factor.
                reward=reward * self.reward_scale,
                value=value,
                term=term.astype(jnp.float32),
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
            return RolloutCarry(env_state, next_obs, next_gstate, rms), transition

        carry, traj = jax.lax.scan(step, carry, jax.random.split(key, num_steps))
        last_value = self._values(
            critic_params, rms_normalize(carry.rms, carry.obs), carry.gstate
        )
        return carry, traj, last_value

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def _update(
        self,
        actor_state: TrainState,
        critic_state: TrainState,
        traj: Transition,
        advantages: jax.Array,
        returns: jax.Array,
        lr_actor: jax.Array,
        lr_critic: jax.Array,
    ) -> tuple[TrainState, TrainState, dict]:
        t, e, n = traj.obs.shape[0], traj.obs.shape[1], traj.obs.shape[2]
        num_mb = self.num_minibatches
        e_mb = e // num_mb
        flat = t * e_mb * n

        # Per-agent advantage; the batch statistics are shared, the values are
        # not. Normalised over the whole (T, E, N) batch, once, before the
        # epoch scan: every epoch must see the same targets.
        policy_mask = jnp.ones_like(advantages) if traj.policy_mask is None else traj.policy_mask
        count = jnp.maximum(policy_mask.sum(), 1.)
        mean_adv = (advantages * policy_mask).sum() / count
        var_adv = (((advantages - mean_adv) ** 2) * policy_mask).sum() / count
        adv = (advantages - mean_adv) / (jnp.sqrt(var_adv) + 1e-8)

        # Only the fields the losses read; memory is (T, E*N, H), so it is
        # regrouped by environment before slicing and flattened back after.
        batch = (
            traj.obs, traj.gstate, traj.action, traj.z, traj.log_prob,
            traj.done, adv, returns, policy_mask,
            None if traj.memory is None else traj.memory.reshape(t, e, n, -1),
            jnp.zeros_like(traj.action) if traj.teacher_action is None else traj.teacher_action,
            jnp.zeros_like(policy_mask) if traj.teacher_mask is None else traj.teacher_mask,
        )

        def select(env_idx):
            if num_mb == 1:
                return batch
            return jax.tree_util.tree_map(
                lambda x: jnp.take(x, env_idx, axis=1), batch)

        def minibatch(carry, env_idx):
            a_state, c_state, actor_enabled = carry
            (obs, gstate, action, z, log_prob, done, adv_b, returns_b, mask_b,
             memory, teacher_action, teacher_mask) = select(env_idx)

            # Feed-forward actors consume this flat batch directly. Recurrent
            # actors use the ordered trajectory and flatten only their outputs
            # so PPO ratios stay aligned with these action arrays.
            obs_f = obs.reshape(flat, -1)
            act_f = action.reshape(flat, -1)
            # The pre-squash sample is replayed from the rollout instead of being
            # recovered with arctanh(action): arctanh needs the action clipped
            # away from +-1 first, and that clip silently rewrites z for exactly
            # the saturated samples that dominate once sigma grows, corrupting
            # the ratio.
            z_f = z.reshape(flat, -1)
            old_log_prob = log_prob.reshape(flat)
            adv_f = adv_b.reshape(flat)
            returns_f = returns_b.reshape(flat)
            # The per-agent channel stack is the largest tensor in the update;
            # it is expanded per minibatch so its size scales with E / num_mb.
            critic_args = self._critic_args(obs, gstate)
            initial_memory = None if memory is None else memory[0].reshape(e_mb * n, -1)

            def actor_loss_fn(params):
                if self.actor.recurrent:
                    _, (mean, log_std) = recurrent_actor_sequence(
                        self.actor, params, obs, initial_memory, done)
                else:
                    mean, log_std = self.actor.apply(params, obs_f)
                loss, diagnostics = ppo_loss(mean, log_std, z_f, act_f, old_log_prob, adv_f, mask_b.reshape(flat))
                imitation = recovery_imitation_loss(mean, teacher_action.reshape(flat, -1), teacher_mask.reshape(flat))
                return loss + self.recovery_imitation_coef * imitation, (*diagnostics, imitation)

            def critic_loss_fn(params):
                values = self.critic.apply(params, *critic_args).squeeze(-1)
                # Huber rather than MSE: the completion bonuses are sparse and
                # large, so a single unpredicted bonus produces an error the
                # squared loss amplifies into a gradient that wipes out the value
                # head. Huber is quadratic within delta and linear beyond it,
                # which caps the per-sample gradient at delta — gradient clipping
                # that acts per sample instead of on the summed batch norm.
                # optax.huber_loss already carries the 0.5 factor in the
                # quadratic branch, so no extra scaling here.
                return jnp.mean(
                    optax.huber_loss(values, returns_f, delta=self.huber_delta)
                )

            (_, (a_loss, entropy, std, approx_kl, clip_fraction, imitation)), a_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(a_state.params)
            actor_enabled = actor_enabled & ((self.target_kl <= 0)
                                             | (approx_kl <= self.target_kl))
            actor_updated = actor_enabled & (jnp.any(mask_b > 0)
                | ((self.recovery_imitation_coef > 0) & jnp.any(teacher_mask > 0)))
            a_state = jax.lax.cond(actor_updated,
                                   lambda: _apply_gradients(a_state, a_grads, lr_actor),
                                   lambda: a_state)

            c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(c_state.params)
            c_state = _apply_gradients(c_state, c_grads, lr_critic)
            return (a_state, c_state, actor_enabled), (
                a_loss, c_loss, entropy, std, approx_kl, clip_fraction,
                actor_updated.astype(jnp.float32), imitation)

        def ppo_loss(mean, log_std, z_f, act_f, old_log_prob, adv_f, mask):
            std = jnp.exp(log_std)
            log_prob = _tanh_normal_log_prob(z_f, mean, std, act_f)

            ratio = jnp.exp(log_prob - old_log_prob)
            surr1 = ratio * adv_f
            surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_f
            denominator = jnp.maximum(mask.sum(), 1.)
            loss = -jnp.sum(jnp.minimum(surr1, surr2) * mask) / denominator

            # This must be evaluated under the *current* policy. Reusing fixed
            # rollout samples gives a cross-entropy gradient which incorrectly
            # drives std upward. Quadrature keeps the tanh Jacobian differentiable
            # and makes the bonus actively pull saturated actions back inward.
            entropy = jnp.sum(_tanh_normal_entropy(mean, std) * mask) / denominator
            approx_kl, clip_fraction = policy_update_diagnostics(
                log_prob - old_log_prob, mask, self.clip_eps)
            return loss - self.entropy_coef * entropy, (
                loss, entropy, jnp.mean(std), approx_kl, clip_fraction)

        def epoch(carry, key):
            # A fresh environment partition every epoch; with a single
            # minibatch the permutation is unused and the update is full-batch.
            groups = jax.random.permutation(key, e).reshape(num_mb, e_mb)
            return jax.lax.scan(minibatch, carry, groups)

        # Derived from the optimiser step so successive updates reshuffle
        # without threading a key through the training loop.
        keys = jax.random.split(
            jax.random.fold_in(jax.random.PRNGKey(0), actor_state.step), self.n_epochs)
        (actor_state, critic_state, _), (a_losses, c_losses, entropies, stds, kls, clip_fractions, actor_updates, imitation_losses) = jax.lax.scan(
            epoch, (actor_state, critic_state, jnp.bool_(True)), keys
        )
        metrics = {
            'actor_loss':  jnp.mean(a_losses),
            'critic_loss': jnp.mean(c_losses),
            'entropy':     jnp.mean(entropies),
            # Logged explicitly: sigma is the quantity that actually diverged,
            # and reading it off the entropy is guesswork once tanh is involved.
            'std':         stds[-1, -1],
            'approx_kl':   jnp.mean(kls),
            'clip_fraction': jnp.mean(clip_fractions),
            'actor_update_fraction': jnp.mean(actor_updates),
            'recovery_imitation_loss': jnp.mean(imitation_losses),
            'teacher_fraction': jnp.mean(batch[-1]),
        }
        return actor_state, critic_state, metrics

    def update(self, actor_state, critic_state, traj, advantages, returns,
               lr_actor, lr_critic):
        return self._update_fn(
            actor_state, critic_state, traj, advantages, returns,
            jnp.float32(lr_actor), jnp.float32(lr_critic),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def act_deterministic(self, actor_params, obs: jax.Array) -> jax.Array:
        """Greedy action tanh(mean) for evaluation. obs is (B, obs_dim)."""
        mean, _ = self.actor.apply(actor_params, obs)
        return jnp.tanh(mean)
