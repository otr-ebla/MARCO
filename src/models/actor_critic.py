"""Flax actor and critic networks for CTDE.

Actor (decentralised execution): a 1D-CNN compresses the lidar scan, the
embedding is concatenated with odometry, relative neighbours and the binary
local-coverage patch, and a small MLP emits the action distribution. Optional
personal-map CNN, ordered observation stack and GRU provide spatial and temporal
context without exposing the critic's privileged map. With
map_encoder='spatial_memory' the personal map is read by SpatialMemoryRead:
re-centred on the robot, summarised by a deep CNN and queried by attention, so
the policy can locate uncovered areas it left behind anywhere on the floor.

Critic (centralised training): a light 2D-CNN reads the per-agent global
map, optionally stacking ordered frames with each robot's visit-count map.
Frame vectors include joint kinematics and previous visited cells when stacked.

LocalCritic (IPPO): the actor's encoder on the agent's own observation, V(o_i).

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


def _local_features(obs: jax.Array, vec_dim: int, n_rays: int, tail_dim: int,
                    lidar_embed: int, memory_map_shape: tuple = (),
                    observation_stack: int = 1, encode_maps: bool = True,
                    crop_shape: tuple = (), crop_encoder: str = 'flat') -> jax.Array:
    """Shared local encoder: [vec | lidar | tail] -> [lidar embedding, vec, tail].

    Called from inside a compact module, so the layers it creates belong to
    the caller (names Conv_0, Conv_1, Dense_0 are unchanged for the actor).
    With encode_maps=False the personal maps are dropped from every frame;
    the caller reads the latest one itself (see latest_memory_map).
    With crop_encoder='cnn' the local crop at the start of the tail, (C, S, S)
    with one channel per map layer (occupied, covered, known, ...), is read
    by a 2D CNN instead of being passed on flat ('flat', older checkpoints).
    """
    batch = obs.shape[0]
    if observation_stack > 1:
        split = observation_stack * (vec_dim + n_rays)
        head = obs[:, :split].reshape(batch, observation_stack, vec_dim + n_rays)
        tail = obs[:, split:].reshape(batch, observation_stack, tail_dim)
        obs = jnp.concatenate([head, tail], axis=-1).reshape(
            batch * observation_stack, vec_dim + n_rays + tail_dim)
    vec   = obs[:, :vec_dim]
    lidar = obs[:, vec_dim : vec_dim + n_rays]
    tail  = obs[:, vec_dim + n_rays : vec_dim + n_rays + tail_dim]

    # A lidar scan is a ring, so circular padding keeps the two ends of the
    # array adjacent and makes the features rotation-equivariant.
    x = lidar[:, :, None]                                      # (B, n_rays, 1)
    x = nn.relu(_conv(16, (5,), (2,), 'CIRCULAR')(x))
    x = nn.relu(_conv(32, (3,), (2,), 'CIRCULAR')(x))
    x = nn.relu(_dense(lidar_embed, _RELU_GAIN)(x.reshape(x.shape[0], -1)))
    if crop_encoder == 'cnn' and crop_shape:
        # Full resolution: on a 5x5 crop a stride would merge the cell under
        # the robot with its neighbours.
        crop_dim = math.prod(crop_shape)
        crop = jnp.transpose(tail[:, :crop_dim].reshape(-1, *crop_shape), (0, 2, 3, 1))
        crop = nn.relu(_conv(16, (3, 3), (1, 1), 'SAME')(crop))
        crop = nn.relu(_conv(32, (3, 3), (1, 1), 'SAME')(crop))
        crop = nn.relu(_dense(64, _RELU_GAIN)(crop.reshape(crop.shape[0], -1)))
        tail = jnp.concatenate([crop, tail[:, crop_dim:]], axis=-1)
    elif crop_encoder != 'flat':
        raise ValueError(f'Unknown crop_encoder: {crop_encoder}')
    if memory_map_shape and not encode_maps:
        tail = tail[:, :-math.prod(memory_map_shape)]
    elif memory_map_shape:
        channels, height, width = memory_map_shape
        map_dim = channels * height * width
        maps = tail[:, -map_dim:].reshape(-1, channels, height, width)
        maps = jnp.transpose(maps, (0, 2, 3, 1))
        maps = nn.relu(nn.Conv(16, (3, 3), padding='SAME', name='map_conv_0')(maps))
        maps = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2), padding='SAME', name='map_conv_1')(maps))
        maps = nn.relu(nn.Dense(64, name='map_embed')(maps.reshape(maps.shape[0], -1)))
        tail = jnp.concatenate([tail[:, :-map_dim], maps], axis=-1)
    features = jnp.concatenate([x, vec, tail], axis=-1)
    return features.reshape(batch, -1)


def latest_memory_map(obs: jax.Array, vec_dim: int, n_rays: int, tail_dim: int,
                      memory_map_shape: tuple, observation_stack: int = 1) -> jax.Array:
    """The newest frame's personal map, (B, C, H, W).

    Layout: [stack x (vec | lidar) | stack x tail], frames oldest first, with
    the map at the end of each tail. The map is cumulative, so older frames
    hold no extra spatial information.
    """
    start = observation_stack * (vec_dim + n_rays) + observation_stack * tail_dim
    map_dim = math.prod(memory_map_shape)
    return obs[:, start - map_dim:start].reshape(-1, *memory_map_shape)


def egocentric_memory(maps: jax.Array) -> jax.Array:
    """(B, 5, H, W) personal map -> (B, 2H-1, 2W-1, 6), robot cell at the centre.

    Output channels: occupied, covered, known, uncovered (known free and not
    covered), teammates, inside the floor. Beyond the floor every cell reads
    as a known wall, as in the actor crop.
    """
    b, _, h, w = maps.shape
    occupied, covered, known, me, peers = (maps[:, k] for k in range(5))
    uncovered = known * (1. - occupied) * (1. - covered)
    frame = jnp.stack([occupied, covered, known, uncovered, peers,
                       jnp.ones_like(known)], axis=-1)                # (B, H, W, 6)
    outside = jnp.array([1., 0., 1., 0., 0., 0.], jnp.float32)
    padded = jnp.broadcast_to(outside, (b, 3 * h - 2, 3 * w - 2, 6))
    padded = padded.at[:, h - 1:2 * h - 1, w - 1:2 * w - 1].set(frame)
    cell = jnp.argmax(me.reshape(b, -1), axis=-1)
    return jax.vmap(lambda p, r, c: jax.lax.dynamic_slice(
        p, (r, c, 0), (2 * h - 1, 2 * w - 1, 6)))(padded, cell // w, cell % w)


class SpatialMemoryRead(nn.Module):
    """Neural-Map-style read of the robot's persistent belief map.

    The environment writes the memory (lidar marks cells known, driving marks
    them covered, teammates in range OR their maps in), so it never forgets an
    uncovered area. This module only has to *read* it well:

    1. Egocentric frame: the (H, W) map is shifted so the robot's cell is the
       centre of a (2H-1, 2W-1) window; cells beyond the floor read as known
       walls. Every location keeps a fixed meaning relative to the robot, so
       convolutions learn "uncovered cells to my left" once, everywhere.
    2. Global read: a strided CNN over the whole window -> `embed` features.
    3. Context read: multi-head attention from the robot's current features
       (query) over every 2x2 block of the window, with its relative offset
       and distance as keys -> where the relevant uncovered space is.

    maps: (B, 5, H, W) channels occupied, covered, known, self, teammates.
    """

    embed: int = 256
    heads: int = 4
    key_dim: int = 16

    @nn.compact
    def __call__(self, maps: jax.Array, query: jax.Array) -> jax.Array:
        b, _, h, w = maps.shape
        ego = egocentric_memory(maps)
        x = nn.relu(_conv(16, (3, 3), (1, 1), 'SAME')(ego))
        x = nn.relu(_conv(32, (3, 3), (2, 2), 'SAME')(x))             # (B, h', w', 32)
        gh, gw = x.shape[1], x.shape[2]

        # Context read over the stride-2 grid. Block j sits about 2j - (H-1)
        # cells from the robot along each axis.
        dy = (2. * jnp.arange(gh) - (h - 1)) / h
        dx = (2. * jnp.arange(gw) - (w - 1)) / w
        pos = jnp.stack(jnp.meshgrid(dy, dx, indexing='ij'), axis=-1)  # (h', w', 2)
        pos = jnp.concatenate([pos, jnp.linalg.norm(pos, axis=-1, keepdims=True)], axis=-1)
        cells = jnp.concatenate([x, jnp.broadcast_to(pos, (b, gh, gw, 3))], axis=-1)
        cells = cells.reshape(b, gh * gw, -1)
        inside = nn.max_pool(ego[..., 5:], (2, 2), strides=(2, 2), padding='SAME')
        inside = inside.reshape(b, gh * gw) > .5

        y = nn.relu(_conv(64, (3, 3), (2, 2), 'SAME')(x))
        y = nn.relu(_conv(64, (3, 3), (2, 2), 'SAME')(y))
        g = nn.relu(_dense(self.embed, _RELU_GAIN)(y.reshape(b, -1)))

        d = self.heads * self.key_dim
        keys = _dense(d, 1.0)(cells).reshape(b, gh * gw, self.heads, self.key_dim)
        values = _dense(d, 1.0)(cells).reshape(b, gh * gw, self.heads, self.key_dim)
        q = _dense(d, 1.0)(jnp.concatenate([query, g], axis=-1)).reshape(b, self.heads, self.key_dim)
        scores = jnp.einsum('bhk,bnhk->bhn', q, keys) / math.sqrt(self.key_dim)
        scores = jnp.where(inside[:, None, :], scores, -1e9)
        attention = jax.nn.softmax(scores, axis=-1)
        read = jnp.einsum('bhn,bnhk->bhk', attention, values).reshape(b, d)
        return jnp.concatenate([g, read], axis=-1)


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
    memory_map_shape: tuple = ()
    observation_stack: int = 1
    log_std_min: float = _LOG_STD_MIN
    log_std_max: float = _LOG_STD_MAX
    crop_shape: tuple = ()          # (C, S, S) local crop, read by crop_encoder
    crop_encoder: str = 'flat'      # 'cnn' or 'flat' (older checkpoints)
    map_encoder: str = 'cnn'        # 'cnn' (legacy) or 'spatial_memory'
    map_embed: int = 256            # spatial_memory: global read width
    attention_heads: int = 4        # spatial_memory: context read heads
    attention_dim: int = 16         # spatial_memory: per-head key/value width

    @nn.compact
    def __call__(self, obs: jax.Array, memory=None):
        """
        obs : (B, obs_dim)
        ->  mean    (B, action_dim)  — unbounded, used as mu of underlying Gaussian
            log_std (action_dim,) for feed-forward, (B, action_dim) for recurrent
            memory  (B, hidden_size), recurrent only
        """
        if self.map_encoder == 'spatial_memory':
            if not self.memory_map_shape:
                raise ValueError("map_encoder='spatial_memory' needs env.use_full_memory")
            h = _local_features(obs, self.vec_dim, self.n_rays, self.tail_dim,
                                self.lidar_embed, self.memory_map_shape,
                                self.observation_stack, encode_maps=False,
                                crop_shape=self.crop_shape, crop_encoder=self.crop_encoder)
            maps = latest_memory_map(obs, self.vec_dim, self.n_rays, self.tail_dim,
                                     self.memory_map_shape, self.observation_stack)
            read = SpatialMemoryRead(self.map_embed, self.attention_heads,
                                     self.attention_dim, name='spatial_memory')(maps, h)
            h = jnp.concatenate([h, read], axis=-1)
        elif self.map_encoder == 'cnn':
            h = _local_features(obs, self.vec_dim, self.n_rays, self.tail_dim,
                                self.lidar_embed, self.memory_map_shape, self.observation_stack,
                                crop_shape=self.crop_shape, crop_encoder=self.crop_encoder)
        else:
            raise ValueError(f'Unknown map_encoder: {self.map_encoder}')
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
        if not self.log_std_min < 0.0 < self.log_std_max:
            raise ValueError('log_std_min < 0 < log_std_max is required')
        raw = self.param(
            'log_std_raw',
            nn.initializers.constant(math.log(-self.log_std_min / self.log_std_max)),
            (self.action_dim,),
        )
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * nn.sigmoid(raw)
        if self.recurrent:
            return mean, jnp.broadcast_to(log_std, mean.shape), memory
        return mean, log_std


class Critic(nn.Module):
    """
    Centralised critic: V_i(s) from the agent-centred global state.

    Agent-centred map channels make each value estimate specific to that
    robot while still exposing the whole team's state and visit history.
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


class LocalCritic(nn.Module):
    """
    Decentralised critic for IPPO: V(o_i) from the agent's own observation.

    Same encoder as the actor but separate parameters, so value regression
    cannot distort the policy features. Shared across agents like the actor.
    """

    vec_dim: int = 8
    n_rays: int = 36
    tail_dim: int = 0
    lidar_embed: int = 64
    hidden_size: int = 256
    memory_map_shape: tuple = ()
    observation_stack: int = 1
    crop_shape: tuple = ()
    crop_encoder: str = 'flat'

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """obs : (B, obs_dim) -> value : (B, 1)"""
        h = _local_features(obs, self.vec_dim, self.n_rays, self.tail_dim,
                            self.lidar_embed, self.memory_map_shape, self.observation_stack,
                            crop_shape=self.crop_shape, crop_encoder=self.crop_encoder)
        h = nn.tanh(_dense(self.hidden_size, _RELU_GAIN)(h))
        h = nn.tanh(_dense(self.hidden_size, _RELU_GAIN)(h))
        return _dense(1, 1.0)(h)
