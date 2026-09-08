"""Flax actor and critic networks for CTDE.

Actor (decentralised execution): a 1D-CNN compresses the lidar scan, the
embedding is concatenated with odometry, relative neighbours and the binary
local-coverage patch, and a small MLP emits the action distribution. There is
no recurrence: the observation already carries the local coverage state, so a
GRU only added a serial dependency and a hidden state to carry around.

Critic (centralised training): a light 2D-CNN reads the per-agent multi-channel
map [walls, coverage, self, teammates], and the flattened features are
concatenated with the joint kinematic vector before the value head.

Orthogonal initialisers require a QR factorisation, which the Metal backend
does not implement. Callers should evaluate `init` on CPU and transfer the
resulting parameters to the accelerator (see `src.utils.jax_device`).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import flax.linen as nn

_RELU_GAIN = 2.0 ** 0.5

# The policy is tanh-squashed, so an unbounded log_std is never useful: much
# past sigma ~ 2 every sample saturates to +-1 and the policy turns bang-bang.
# The raw parameter is therefore mapped through a sigmoid into
# [_LOG_STD_MIN, _LOG_STD_MAX]. A jnp.clip would bound sigma too, but its
# gradient is exactly zero outside the interval: once log_std touched the
# ceiling it could never come back down. The sigmoid keeps a nonzero gradient
# everywhere while making divergence impossible.
_LOG_STD_MIN = -5.0     # sigma = 0.0067
_LOG_STD_MAX = 1.0      # sigma = 2.72
# sigmoid(_LOG_STD_RAW_INIT) maps to log_std = 0, i.e. sigma = 1.
_LOG_STD_RAW_INIT = math.log(-_LOG_STD_MIN / _LOG_STD_MAX)


def _dense(features: int, gain: float) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=nn.initializers.orthogonal(gain),
        bias_init=nn.initializers.zeros,
    )


def _conv(features: int, kernel, strides, padding, gain: float = _RELU_GAIN) -> nn.Conv:
    return nn.Conv(
        features,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        kernel_init=nn.initializers.orthogonal(gain),
        bias_init=nn.initializers.zeros,
    )


class Actor(nn.Module):
    """
    Shared-parameter actor, optionally with a per-robot GRU memory.

    The observation layout is fixed by the environment:
        [continuous vector (vec_dim) | lidar (n_rays)]

    Actions are sampled from a tanh-squashed Normal distribution:
        z ~ N(mean, exp(log_std)),  a = tanh(z)  in (-1, 1)^action_dim
    """

    action_dim: int = 2
    vec_dim: int = 8
    n_rays: int = 36
    tail_dim: int = 0
    lidar_embed: int = 64
    hidden_size: int = 128
    recurrent: bool = False

    @nn.compact
    def __call__(self, obs: jax.Array, memory=None):
        """
        obs : (B, obs_dim)
        ->  mean    (B, action_dim)  — unbounded, used as mu of underlying Gaussian
            log_std (action_dim,) for feed-forward, (B, action_dim) for recurrent
            memory  (B, hidden_size), recurrent only
        """
        vec   = obs[:, : self.vec_dim]
        lidar = obs[:, self.vec_dim : self.vec_dim + self.n_rays]
        tail  = obs[:, self.vec_dim + self.n_rays : self.vec_dim + self.n_rays + self.tail_dim]

        # A lidar scan is a ring, so circular padding keeps the two ends of the
        # array adjacent and makes the features rotation-equivariant.
        x = lidar[:, :, None]                                      # (B, n_rays, 1)
        x = nn.relu(_conv(16, (5,), (2,), 'CIRCULAR')(x))
        x = nn.relu(_conv(32, (3,), (2,), 'CIRCULAR')(x))
        x = nn.relu(_dense(self.lidar_embed, _RELU_GAIN)(x.reshape(x.shape[0], -1)))

        h = jnp.concatenate([x, vec, tail], axis=-1)
        h = nn.tanh(_dense(self.hidden_size, _RELU_GAIN)(h))
        h = nn.tanh(_dense(self.hidden_size, _RELU_GAIN)(h))

        if self.recurrent:
            if memory is None:
                memory = jnp.zeros((obs.shape[0], self.hidden_size), obs.dtype)
            memory, h = nn.GRUCell(features=self.hidden_size, name='memory')(memory, h)

        mean = _dense(self.action_dim, 0.01)(h)

        # State-independent spread: the MLP predicts only the mean, so no
        # observation can drive sigma. Bounded by construction, see the
        # _LOG_STD_* constants.
        raw = self.param(
            'log_std_raw',
            nn.initializers.constant(_LOG_STD_RAW_INIT),
            (self.action_dim,),
        )
        log_std = _LOG_STD_MIN + (_LOG_STD_MAX - _LOG_STD_MIN) * nn.sigmoid(raw)
        if self.recurrent:
            return mean, jnp.broadcast_to(log_std, mean.shape), memory
        return mean, log_std


class Critic(nn.Module):
    """
    Centralised critic: V_i(s) from the agent-centred global state.

    The map channels [walls, coverage, self, teammates, my assignment,
    teammates' assignments] make the estimate
    agent-specific, which is what lets each robot get its own advantage while
    the value still sees the whole team.
    """

    hidden_size: int = 256
    map_embed: int = 128

    @nn.compact
    def __call__(self, grid: jax.Array, vec: jax.Array) -> jax.Array:
        """grid : (B, C, H, W), vec : (B, vec_dim) -> value : (B, 1)"""
        x = jnp.transpose(grid, (0, 2, 3, 1))                      # NCHW -> NHWC
        x = nn.relu(_conv(16, (3, 3), (2, 2), 'SAME')(x))
        x = nn.relu(_conv(32, (3, 3), (2, 2), 'SAME')(x))
        x = nn.relu(_dense(self.map_embed, _RELU_GAIN)(x.reshape(x.shape[0], -1)))

        h = jnp.concatenate([x, vec], axis=-1)
        h = nn.tanh(_dense(self.hidden_size, _RELU_GAIN)(h))
        h = nn.tanh(_dense(self.hidden_size, _RELU_GAIN)(h))
        return _dense(1, 1.0)(h)
