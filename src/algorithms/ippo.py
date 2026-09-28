"""
Independent PPO (IPPO) in JAX, parameter-shared actor and critic.

IPPO is MAPPO with a decentralised critic: V(o_i) sees only what the actor
sees, instead of the agent-centred global state V_i(s). Every other piece is
inherited unchanged: the shared tanh-Normal actor, the clipped surrogate, the
per-agent GAE over (E, N) streams, the Huber value loss and the scanned
rollout. Because the critic reads no privileged information, training is fully
decentralised too; the other robots are simply part of each agent's
environment.

The critic consumes the same normalised observation the actor does (the
`obs` field of `Transition`), so no extra state is stored per step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.algorithms.mappo import MAPPO


class IPPO(MAPPO):
    """Independent PPO: MAPPO's machinery with a local critic V(o_i)."""

    def _dummy_critic_args(self) -> tuple:
        return (jnp.zeros((1, self.env.obs_dim), jnp.float32),)

    def _critic_args(self, obs_n: jax.Array, gstate) -> tuple:
        return (obs_n.reshape(-1, obs_n.shape[-1]),)
