"""
VecEnv: E independent MultiRobotCoverageEnv instances vectorised with `jax.vmap`.

The JAX environment is a pure function, so parallelism comes from mapping over a
leading environment axis on the accelerator rather than from worker processes.
Every batched state is a single pytree with leading dimension E.

`step` auto-resets: environments that reach done are replaced by a fresh episode
before the observation and global state are computed, so the returned obs/state
already belong to the new episode while `rewards`/`terms` still describe the
step that ended it.

It returns both `terms` (hard collision only) and `dones` (term | trunc) so
callers can use the right signal for GAE masking vs hidden-state reset.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.envs.coverage_vector_env import EnvState, MultiRobotCoverageEnv


def _select(done: jax.Array, fresh: jax.Array, current: jax.Array) -> jax.Array:
    """`jnp.where(done, fresh, current)`, skipping zero-sized leaves.

    An empty leaf (e.g. `human_positions` when num_humans == 0) holds no data,
    so the select is an identity — but jax-metal's MPSGraph backend aborts when
    it compiles a select over a 0-element tensor inside `lax.scan`
    ("Incompatible shape for parameter"). Returning the leaf untouched keeps the
    semantics and keeps the Metal backend usable.
    """
    if current.size == 0:
        return current
    return jnp.where(done, fresh, current)


class VecEnv:
    """Vectorised wrapper owning E environment states as one batched pytree.

    Both `reset` and `step` are pure and jitted: they take and return state
    rather than mutating it, which lets callers scan them over time.
    """

    def __init__(self, num_envs: int, env_config: dict):
        self.E = int(num_envs)
        self.env = MultiRobotCoverageEnv(env_config)

        self.reset = jax.jit(self._reset)
        self.step = jax.jit(self._step)
        self.update_ghost_robot_prob = jax.jit(self._update_ghost_robot_prob)

    @property
    def num_robots(self) -> int:
        return self.env.num_robots

    @property
    def obs_dim(self) -> int:
        return self.env.obs_dim

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def norm_dim(self) -> int:
        return self.env.norm_dim

    @property
    def grid_h(self) -> int:
        return self.env.grid_h

    @property
    def grid_w(self) -> int:
        return self.env.grid_w

    @property
    def critic_channels(self) -> int:
        return self.env.critic_channels

    @property
    def critic_vec_dim(self) -> int:
        return self.env.critic_vec_dim

    def critic_inputs(self, gstate):
        """Per-agent critic tensors; broadcasts over the leading env axis."""
        return self.env.critic_inputs(gstate)

    # ------------------------------------------------------------------

    def _reset(self, key: jax.Array):
        """Reset all environments. Returns (state, obs, global_states, infos)."""
        keys = jax.random.split(key, self.E)
        state = jax.vmap(self.env.reset)(keys)
        obs = jax.vmap(self.env.get_obs)(state)
        gstate = jax.vmap(self.env.get_global_state)(state)
        infos = jax.vmap(self.env.get_info)(state)
        return state, obs, gstate, infos

    def _step(self, state: EnvState, actions: jax.Array):
        """
        state   : batched EnvState (leading dim E)
        actions : (E, N, 2) in [-1, 1]

        Returns
        -------
        state   : batched EnvState — already reset where the episode ended
        obs     : (E, N, obs_dim)  — from new episode when done
        rewards : (E, N)
        terms   : (E,) bool        — True on hard termination (collision or a
                                     fully covered map)
        dones   : (E,) bool        — True on terminated OR truncated
        infos   : dict of (E, ...) arrays — from the episode that just stepped,
                  i.e. pre-reset, so terminal coverage stays visible
        gstates : GlobalState with leading (E,) — from new episode when done

        Use `terms` for GAE bootstrap masking (termination = no bootstrap).
        """

        def one(s: EnvState, a: jax.Array):
            s, reward, term, trunc, cover = self.env._step_core(s, a)
            done = term | trunc

            # Diagnostics describe the step just taken, like `reward`, so they
            # are read before the auto-reset. Reading them afterwards would
            # replace the final coverage of a finished episode with the fresh
            # episode's empty grid, making end-of-episode coverage unobservable.
            info = self.env.get_info(s)
            # Preserve post-step positions across auto-reset. Evaluation uses
            # them to count entries into cells that were already covered.
            info['robot_positions'] = s.robot_positions

            key, reset_key = jax.random.split(s.key)
            fresh = self.env._reset_state(reset_key)
            s = jax.tree_util.tree_map(
                lambda f, c: _select(done, f, c), fresh, s.replace(key=key)
            )
            if self.env.use_memory:
                # One scan per step for both branches: a fresh episode starts
                # with an empty memory and an uncovered spawn cell, exactly
                # as `env.reset` would produce.
                s = self.env._refresh_memory(s, cover & ~done)

            return (s, self.env.get_obs(s), reward, term, done,
                    info, self.env.get_global_state(s))

        return jax.vmap(one)(state, actions)

    def _update_ghost_robot_prob(self, state: EnvState, probs: jax.Array):
        def one(s: EnvState, p: jax.Array):
            return self.env.set_ghost_robot_prob(s, p)
        return jax.vmap(one)(state, probs)
