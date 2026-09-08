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
    action:   jax.Array   # tanh(z), what the environment consumed
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
        self.max_grad_norm = float(config.get('max_grad_norm', 10.0))
        self.n_epochs      = int(config.get('n_epochs',      10))
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

        env = self.env
        dummy_obs  = jnp.zeros((1, env.obs_dim), jnp.float32)
        dummy_grid = jnp.zeros(
            (1, env.critic_channels, env.grid_h, env.grid_w), jnp.float32
        )
        dummy_vec  = jnp.zeros((1, env.critic_vec_dim), jnp.float32)

        with jax.default_device(self.init_device):
            args = jax.device_put(
                (actor_key, critic_key, dummy_obs, dummy_grid, dummy_vec),
                self.init_device,
            )
            a_key, c_key, o, g, v = args
            actor_params = self.actor.init(a_key, o)
            critic_params = self.critic.init(c_key, g, v)

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

    def _values(self, critic_params, gstate) -> jax.Array:
        """V_i(s) for every agent slot. gstate has leading (E,) -> (E, N)."""
        e, n = self.env.E, self.env.num_robots
        grid, vec = self.env.critic_inputs(gstate)
        value = self.critic.apply(
            critic_params,
            grid.reshape(e * n, *grid.shape[-3:]),
            vec.reshape(e * n, -1),
        )
        return value.reshape(e, n)

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

            value = self._values(critic_params, carry.gstate)

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
            )
            return RolloutCarry(env_state, next_obs, next_gstate, rms), transition

        carry, traj = jax.lax.scan(step, carry, jax.random.split(key, num_steps))
        last_value = self._values(critic_params, carry.gstate)
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
        flat = t * e * n

        # Feed-forward actors consume this flat batch directly. Recurrent
        # actors use the original ordered trajectory below and flatten only
        # their outputs so PPO ratios stay aligned with these action arrays.
        obs_f = traj.obs.reshape(flat, -1)
        act_f = traj.action.reshape(flat, -1)
        # The pre-squash sample is replayed from the rollout instead of being
        # recovered with arctanh(action): arctanh needs the action clipped away
        # from +-1 first, and that clip silently rewrites z for exactly the
        # saturated samples that dominate once sigma grows, corrupting the ratio.
        z_f = traj.z.reshape(flat, -1)
        old_log_prob = traj.log_prob.reshape(flat)

        # Per-agent advantage; the batch statistics are shared, the values are
        # not. Normalised over the whole (T, E, N) batch, once, before the
        # epoch scan: every epoch must see the same targets.
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_f = adv.reshape(flat)

        # Expanded once here rather than inside the epoch scan: the per-agent
        # channel stack is the largest tensor in the update.
        grid, vec = self.env.critic_inputs(traj.gstate)
        grid_f = grid.reshape(flat, *grid.shape[-3:])
        vec_f = vec.reshape(flat, -1)
        returns_f = returns.reshape(flat)

        def actor_loss_fn(params):
            if self.actor.recurrent:
                _, (mean, log_std) = recurrent_actor_sequence(
                    self.actor, params, traj.obs, traj.memory[0], traj.done)
            else:
                mean, log_std = self.actor.apply(params, obs_f)
            std = jnp.exp(log_std)
            log_prob = _tanh_normal_log_prob(z_f, mean, std, act_f)

            ratio = jnp.exp(log_prob - old_log_prob)
            surr1 = ratio * adv_f
            surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_f
            loss = -jnp.mean(jnp.minimum(surr1, surr2))

            # This must be evaluated under the *current* policy. Reusing fixed
            # rollout samples gives a cross-entropy gradient which incorrectly
            # drives std upward. Quadrature keeps the tanh Jacobian differentiable
            # and makes the bonus actively pull saturated actions back inward.
            entropy = jnp.mean(_tanh_normal_entropy(mean, std))
            return loss - self.entropy_coef * entropy, (loss, entropy, jnp.mean(std))

        def critic_loss_fn(params):
            values = self.critic.apply(params, grid_f, vec_f).squeeze(-1)
            # Huber rather than MSE: the completion bonuses are sparse and large,
            # so a single unpredicted bonus produces an error the squared loss
            # amplifies into a gradient that wipes out the value head. Huber is
            # quadratic within delta and linear beyond it, which caps the
            # per-sample gradient at delta — gradient clipping that acts per
            # sample instead of on the summed batch norm.
            # optax.huber_loss already carries the 0.5 factor in the quadratic
            # branch, so no extra scaling here.
            return jnp.mean(
                optax.huber_loss(values, returns_f, delta=self.huber_delta)
            )

        def epoch(carry, _):
            a_state, c_state = carry
            (_, (a_loss, entropy, std)), a_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(a_state.params)
            a_state = _apply_gradients(a_state, a_grads, lr_actor)

            c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(c_state.params)
            c_state = _apply_gradients(c_state, c_grads, lr_critic)
            return (a_state, c_state), (a_loss, c_loss, entropy, std)

        (actor_state, critic_state), (a_losses, c_losses, entropies, stds) = jax.lax.scan(
            epoch, (actor_state, critic_state), None, length=self.n_epochs
        )
        metrics = {
            'actor_loss':  jnp.mean(a_losses),
            'critic_loss': jnp.mean(c_losses),
            'entropy':     jnp.mean(entropies),
            # Logged explicitly: sigma is the quantity that actually diverged,
            # and reading it off the entropy is guesswork once tanh is involved.
            'std':         stds[-1],
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
