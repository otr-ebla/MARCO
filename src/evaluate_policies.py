#!/usr/bin/env python3
"""Benchmark trained MARL coverage policies and save result data.

    python -m src.evaluate_policies --checkpoint checkpoints/e2e/checkpoint_e2e.pkl
    python -m src.evaluate_policies --config config/single_robot.yaml --bcd tour \
        --checkpoint checkpoints/single/rl/checkpoint_e2e_memory.pkl --label RL

`--bcd tour` adds classical centralised Boustrophedon Cellular Decomposition
(planned on the known map) as a baseline, run on the same seeds and maps.

Each checkpoint is evaluated for `--episodes` episodes in compiled accelerator
batches. Raw episode data, aggregate statistics and run metadata are written to
the output directory. Figures are generated separately by
``python -m src.plot_evaluation_results``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mrcpp-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.mappo import RunningMeanStd, rms_normalize
from src.envs.vec_env import VecEnv
from src.models.actor_critic import Actor
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "mappo_baseline.yaml"

CONTROL_FIELDS = [
    "policy_fraction", "fallback_fraction", "sequence_fraction", "sequence_blocked_fraction",
    "fallback_stationary_fraction", "fallback_contact_fraction",
    "blocked_boundary_fraction", "blocked_unknown_fraction", "blocked_edge_fraction",
    "blocked_lidar_fraction", "blocked_empty_queue_fraction",
]

FIELDS = [
    "policy", "episode", "seed", "num_humans", "steps",
    "completion_time_steps", "completion_time_seconds", "coverage_rate",
    "covered_cells", "completion", "timeout", "collision_end", "team_return",
    "wall_collisions", "robot_collisions", "human_collisions", "all_collisions",
    "wall_collision_rate", "robot_collision_rate", "human_collision_rate",
    "all_collision_rate", "revisits", "cell_entries", "revisit_rate",
    "sweep_efficiency",
] + CONTROL_FIELDS


@dataclass(frozen=True)
class PolicySpec:
    name: str
    humans: int
    checkpoint: Path | None = None
    bcd_rule: str | None = None   # BCD expert instead of a checkpoint: 'tour' or 'local'


def _env_config(path: Path, humans: int, max_steps: int | None,
                maps: int | None = None, map_seed: int | None = None) -> tuple[dict, dict]:
    config = load_config(str(path))
    env_cfg = dict(config.get("env", {}))
    env_cfg["num_humans"] = humans
    if max_steps is not None:
        env_cfg["max_steps"] = max_steps
    if maps is not None:
        env_cfg["num_maps"] = maps
    if map_seed is not None:
        env_cfg["map_seed"] = map_seed
    return config, env_cfg


def _bcd_policy(spec: PolicySpec, config: dict, env_cfg: dict, batch: int):
    """BCD expert on the evaluation environment: (vec_env, act, init_extra).

    'tour' is classical centralised BCD: the decomposition and the tour are
    planned offline on the true map and the robot follows the first cell of
    the tour it has not covered. 'local' is the memory-driven variant (nearest
    uncovered cell). The expert has its own safety filter, so the recovery
    fallback is off.
    """
    from src.envs.bcd_expert import BCDExpert
    from src.envs.coverage_vector_env import E2E_REWARD_DEFAULTS
    env_cfg.update({**E2E_REWARD_DEFAULTS, **config.get("e2e_reward", {})})
    env_cfg["fallback_enabled"] = False
    vec_env = VecEnv(batch, env_cfg)
    expert_cfg = {**config.get("pretrain", {}).get("expert", {}), "target_rule": spec.bcd_rule}
    expert = BCDExpert(vec_env.env, expert_cfg)
    act = jax.vmap(expert.act)

    def policy(state, obs, chunk, key):
        actions, chunk, _ = act(state, chunk)
        return actions, chunk

    return vec_env, policy, expert.init_chunks((vec_env.E,))


def _empty_accumulators(n: int) -> dict[str, np.ndarray]:
    return {
        "steps": np.zeros(n, np.int64),
        "return": np.zeros(n, np.float64),
        "wall": np.zeros(n, np.float64),
        "robot": np.zeros(n, np.float64),
        "human": np.zeros(n, np.float64),
        "revisits": np.zeros(n, np.float64),
        "entries": np.zeros(n, np.float64),
    }


def _record(name: str, episode: int, seed: int, humans: int, steps: int,
            coverage: float, covered_cells: float, complete: float, timeout: float,
            team_return: float, wall: float, robot: float, human: float,
            n_robots: int, dt: float, collision_end: float = 0.0,
            revisits: float = np.nan,
            entries: float = np.nan, efficiency: float = np.nan) -> dict:
    denom = max(steps * n_robots, 1)
    total = wall + robot + human
    return {
        "policy": name, "episode": episode, "seed": seed,
        "num_humans": humans, "steps": steps,
        "completion_time_steps": float(steps) if complete else np.nan,
        "completion_time_seconds": float(steps * dt) if complete else np.nan,
        "coverage_rate": coverage,
        "covered_cells": covered_cells, "completion": complete, "timeout": timeout,
        "collision_end": collision_end, "team_return": team_return,
        "wall_collisions": wall, "robot_collisions": robot,
        "human_collisions": human, "all_collisions": total,
        "wall_collision_rate": wall / denom,
        "robot_collision_rate": robot / denom,
        "human_collision_rate": human / denom,
        "all_collision_rate": total / denom,
        "revisits": revisits, "cell_entries": entries,
        "revisit_rate": revisits / entries if entries > 0 else 0.0,
        "sweep_efficiency": efficiency,
    }


def _checkpoint_policy(spec: PolicySpec, config: dict, env_cfg: dict, batch: int,
                       stochastic: bool, device, policy_only: bool):
    """Trained actor from a checkpoint: (vec_env, act, init_extra)."""
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint}")
    with spec.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    recurrent = checkpoint.get("actor_recurrent", False)
    env_cfg.update(checkpoint.get("reward_weights", {}))
    env_cfg.update({"use_full_memory": False, "observation_stack": 1, "sweep_obs": False,
                    "crop_summary": False, "crop_mode": "memory", "critic_crops": False,
                           "known_coverage_obs": False, "critic_coverage": False,
                    "goal_obs": False, "wall_cells": 0,
                           "history_cell": "last_discovery", "critic_context": False,
                           "critic_stack": 1})
    obs_config = checkpoint.get("obs_config", {"obs_mode": "legacy"})
    env_cfg.update(obs_config)
    # Checkpoints older than memory_map_obs read the full map with use_full_memory.
    env_cfg["memory_map_obs"] = bool(obs_config.get("memory_map_obs", obs_config.get("use_full_memory", False)))
    if policy_only:
        env_cfg['fallback_enabled'] = False
    vec_env = VecEnv(batch, env_cfg)
    env = vec_env.env
    model_cfg = config.get("model", {})
    actor_cfg = {"lidar_embed": model_cfg.get("lidar_embed", 64),
                 "hidden_size": model_cfg.get("hidden_size", 128)}
    actor_cfg.update(checkpoint.get("actor_config", {}))
    actor = Actor(recurrent=recurrent, action_dim=env.action_dim, vec_dim=env.obs_vec_dim,
                  n_rays=env.n_rays, tail_dim=env.patch_dim,
                  memory_map_shape=env.memory_map_shape, crop_shape=env.crop_shape,
                  observation_stack=env.observation_stack, **actor_cfg)
    params = jax.device_put(checkpoint["actor_params"], device)
    rms = RunningMeanStd(*jax.device_put(tuple(checkpoint["obs_rms"]), device))

    def policy(state, obs, memory, action_key):
        normalized = rms_normalize(rms, obs)
        if recurrent:
            mean, log_std, memory = actor.apply(
                params, normalized.reshape(-1, env.obs_dim), memory)
        else:
            mean, log_std = actor.apply(params, normalized.reshape(-1, env.obs_dim))
        if stochastic:
            z = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
            actions = jnp.tanh(z)
        else:
            actions = jnp.tanh(mean)
        return actions.reshape(vec_env.E, env.num_robots, env.action_dim), memory

    memory = jnp.zeros((vec_env.E * env.num_robots, actor.hidden_size)) if recurrent else None
    return vec_env, policy, memory


def evaluate_marl(spec: PolicySpec, config_path: Path, episodes: int, seed: int,
                  max_steps: int | None, batch_size: int, chunk_steps: int,
                  stochastic: bool, device, progress_every: int,
                  policy_only: bool = False, recovery_trace: Path | None = None,
                  maps: int | None = None, map_seed: int | None = None) -> list[dict]:
    config, env_cfg = _env_config(config_path, spec.humans, max_steps, maps, map_seed)
    batch = min(batch_size, episodes)
    if spec.bcd_rule is not None:
        vec_env, policy, extra = _bcd_policy(spec, config, env_cfg, batch)
        recurrent = False
    else:
        vec_env, policy, extra = _checkpoint_policy(
            spec, config, env_cfg, batch, stochastic, device, policy_only)
        recurrent = extra is not None
    env = vec_env.env
    state, obs, _, _ = vec_env.reset(jax.random.PRNGKey(seed))

    def run_chunk(carry, keys):
        def one(c, action_key):
            state, obs, extra = c
            actions, extra = policy(state, obs, extra, action_key)
            next_state, next_obs, rewards, term, done, info, _ = vec_env.step(state, actions)
            previous_col = jnp.clip(
                (state.robot_positions[..., 0] / env.cell_size).astype(jnp.int32),
                0, env.grid_w - 1,
            )
            previous_row = jnp.clip(
                (state.robot_positions[..., 1] / env.cell_size).astype(jnp.int32),
                0, env.grid_h - 1,
            )
            current_col = jnp.clip(
                (info["robot_positions"][..., 0] / env.cell_size).astype(jnp.int32),
                0, env.grid_w - 1,
            )
            current_row = jnp.clip(
                (info["robot_positions"][..., 1] / env.cell_size).astype(jnp.int32),
                0, env.grid_h - 1,
            )
            previous_cell = previous_row * env.grid_w + previous_col
            current_cell = current_row * env.grid_w + current_col
            entered = current_cell != previous_cell
            covered_before = state.coverage_grid.reshape(vec_env.E, -1)
            env_ids = jnp.arange(vec_env.E)[:, None]
            revisited = entered & (covered_before[env_ids, current_cell] > 0.5)
            fallback = info['fallback_used']
            stationary = jnp.linalg.norm(info['robot_positions'] - state.robot_positions, axis=-1) < 1e-5
            contacts = (info['wall_hits'] + info['robot_hits'] + info['human_hits']) > 0
            flags = info['fallback_rejection_flags']
            control = jnp.stack([
                jnp.mean(~fallback, axis=-1), jnp.mean(fallback, axis=-1),
                jnp.mean(info['fallback_sequence_used'], axis=-1),
                jnp.mean(info['fallback_safety_override'], axis=-1),
                jnp.mean(fallback & stationary, axis=-1),
                jnp.mean(fallback & contacts, axis=-1),
                *[jnp.mean((flags & bit) != 0, axis=-1) for bit in (1, 2, 4, 8, 16)],
            ], axis=-1)
            trace = None if recovery_trace is None else (
                flags, state.robot_positions, state.robot_headings, state.robot_velocities,
                state.lidar, state.fallback_goal, state.fallback_stagnation, state.step_count)
            output = (rewards.sum(axis=-1), done, term, info["coverage_ratio"],
                      info["covered_cells"], info["complete"], info["timeout"],
                      info["wall_collision_rate"], info["robot_collision_rate"],
                      info["human_collision_rate"], jnp.sum(revisited, axis=-1),
                      jnp.sum(entered, axis=-1), control, trace)
            if recurrent:
                extra = jnp.where(jnp.repeat(done, env.num_robots)[:, None], 0., extra)
            return (next_state, next_obs, extra), output
        return jax.lax.scan(one, carry, keys)

    run_chunk = jax.jit(run_chunk)
    accum = _empty_accumulators(vec_env.E)
    rows: list[dict] = []
    control_totals = np.zeros((vec_env.E, len(CONTROL_FIELDS)), np.float64)
    trace_count = 0
    key = jax.random.PRNGKey(seed + 1_000_003)
    started = time.time()
    print(f"{spec.name}: starting {episodes} episodes "
          f"({vec_env.E} parallel JAX environments on {describe(device)})", flush=True)
    while len(rows) < episodes:
        key, chunk_key = jax.random.split(key)
        keys = jax.random.split(chunk_key, chunk_steps)
        (state, obs, extra), outputs = run_chunk((state, obs, extra), keys)
        arrays = jax.device_get(outputs)
        (rewards, dones, terms, coverage, covered, complete, timeout, walls,
         robots, humans, revisits, entries, control, trace) = arrays
        for t in range(chunk_steps):
            control_totals += control[t]
            # A bounded sample of blocked poses, at most one per robot per chunk.
            if trace is not None and trace_count < 200 and t == 0:
                flags, positions, headings, velocities, lidar, goals, stagnation, steps = trace
                with recovery_trace.open('a') as handle:
                    for e in range(vec_env.E):
                        for robot in range(env.num_robots):
                            blocked = np.flatnonzero(flags[:, e, robot])
                            if not blocked.size or trace_count >= 200:
                                continue
                            tick = int(blocked[0])
                            record = dict(policy=spec.name, env=e, robot=robot,
                                          step=int(steps[tick, e]), flags=int(flags[tick, e, robot]),
                                          position=positions[tick, e, robot].tolist(),
                                          heading=float(headings[tick, e, robot]),
                                          velocity=velocities[tick, e, robot].tolist(),
                                          lidar_metres=(lidar[tick, e, robot] * env.max_lidar_range).tolist(),
                                          goal=int(goals[tick, e, robot]),
                                          stagnation=int(stagnation[tick, e, robot]))
                            handle.write(json.dumps(record) + '\n')
                            trace_count += 1
            accum["steps"] += 1
            accum["return"] += rewards[t]
            accum["wall"] += walls[t] * env.num_robots
            accum["robot"] += robots[t] * env.num_robots
            accum["human"] += humans[t] * env.num_robots
            accum["revisits"] += revisits[t]
            accum["entries"] += entries[t]
            for e in np.flatnonzero(dones[t]):
                if len(rows) >= episodes:
                    break
                ep = len(rows)
                rows.append(_record(
                    spec.name, ep, seed, spec.humans, int(accum["steps"][e]),
                    float(coverage[t, e]), float(covered[t, e]), float(complete[t, e]),
                    float(timeout[t, e] and not complete[t, e]), float(accum["return"][e]),
                    float(accum["wall"][e]), float(accum["robot"][e]),
                    float(accum["human"][e]), env.num_robots, env.dt,
                    float(terms[t, e] and not complete[t, e]),
                    float(accum["revisits"][e]), float(accum["entries"][e]),
                    float((accum["entries"][e] - accum["revisits"][e])
                          / max(accum["entries"][e], 1.0)),
                ))
                rows[-1].update(zip(CONTROL_FIELDS,
                                    (control_totals[e] / accum['steps'][e]).tolist()))
                control_totals[e] = 0
                for values in accum.values():
                    values[e] = 0
            if len(rows) >= episodes:
                break
        if progress_every and (len(rows) == episodes or len(rows) // progress_every
                               != max(0, len(rows) - vec_env.E) // progress_every):
            elapsed = time.time() - started
            eta = elapsed / len(rows) * (episodes - len(rows)) if rows else float("nan")
            print(f"{spec.name}: {len(rows)}/{episodes} ({100 * len(rows) / episodes:.1f}%) "
                  f"| elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _exclusive_outcome(row: dict) -> str:
    """Assign exactly one episode outcome for rates that must total 100%."""
    if row["completion"] > 0.5:
        return "success"
    collisions = {
        "rhcr": row["human_collisions"],
        "rrcr": row["robot_collisions"],
        "rwcr": row["wall_collisions"],
    }
    largest = max(collisions, key=collisions.get)
    return largest if collisions[largest] > 0 else "tor"


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for policy in dict.fromkeys(row["policy"] for row in rows):
        group = [row for row in rows if row["policy"] == policy]
        outcomes = [_exclusive_outcome(row) for row in group]
        n = len(group)
        completion_times = np.asarray(
            [row["completion_time_seconds"] for row in group], np.float64
        )
        completion_times = completion_times[np.isfinite(completion_times)]
        out = {
            "policy": policy,
            "episodes": n,
            "successful_episodes": outcomes.count("success"),
            "success_coverage_rate_pct": 100.0 * outcomes.count("success") / n,
            "rrcr_pct": 100.0 * outcomes.count("rrcr") / n,
            "rwcr_pct": 100.0 * outcomes.count("rwcr") / n,
            "rhcr_pct": 100.0 * outcomes.count("rhcr") / n,
            "tor_pct": 100.0 * outcomes.count("tor") / n,
            "avg_success_completion_time_seconds": (
                float(completion_times.mean()) if completion_times.size else np.nan
            ),
        }
        for field in FIELDS[4:]:
            values = np.asarray([row[field] for row in group], np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                out[f"{field}_mean"] = float(finite.mean())
                out[f"{field}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
                out[f"{field}_ci95"] = float(1.96 * finite.std(ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0
        result.append(out)
    return result


def plot_results(rows: list[dict], output_base: Path) -> None:
    policies = list(dict.fromkeys(row["policy"] for row in rows))
    colors = plt.get_cmap("tab10").colors[:len(policies)]
    groups = [[row for row in rows if row["policy"] == p] for p in policies]
    fig, axes = plt.subplots(
        3, 3, figsize=(17, 12), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.35)},
    )

    def box(ax, field, title, scale=1.0):
        data = []
        for group in groups:
            values = np.asarray([r[field] for r in group], float) * scale
            values = values[np.isfinite(values)]
            # Matplotlib propagates a single NaN through every box statistic.
            # An empty finite sample remains explicit without hiding other
            # policies that do have observations for this metric.
            data.append(values if values.size else np.asarray([np.nan]))
        artists = ax.boxplot(data, tick_labels=policies, patch_artist=True, showfliers=False)
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.65)
        ax.set_title(title); ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=.25)

    box(axes[0, 0], "coverage_rate", "Final coverage rate", 100)
    box(axes[0, 1], "completion_time_seconds", "Completion time (simulated seconds)")
    box(axes[0, 2], "team_return", "Team return")
    x = np.arange(len(policies))
    outcome_names = (("success", "Success"), ("rrcr", "RRCR"), ("rwcr", "RWCR"),
                     ("rhcr", "RHCR"), ("tor", "TOR"))
    bottom = np.zeros(len(policies))
    for outcome, label in outcome_names:
        values = np.asarray([
            100 * np.mean([_exclusive_outcome(r) == outcome for r in g]) for g in groups
        ])
        axes[1, 0].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[1, 0].set_xticks(x, policies, rotation=18); axes[1, 0].set_title("Episode outcomes (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(axis="y", alpha=.25)
    box(axes[1, 1], "revisit_rate", "Already-covered cell entries (%)", 100)
    box(axes[1, 2], "all_collision_rate", "All-collision rate per robot-step", 100)
    for g, policy, color in zip(groups, policies, colors):
        y = np.asarray([r["coverage_rate"] for r in g]) * 100
        axes[2, 0].plot(np.arange(1, len(y) + 1), np.cumsum(y) / np.arange(1, len(y) + 1), label=policy, color=color)
    axes[2, 0].set_title("Running mean coverage (%)"); axes[2, 0].set_xlabel("Episodes")
    axes[2, 0].grid(alpha=.25); axes[2, 0].legend(fontsize=8)
    for field, label in (("wall_collision_rate", "Wall"), ("robot_collision_rate", "Robot"),
                         ("human_collision_rate", "Human")):
        axes[2, 1].plot(policies, [np.mean([r[field] for r in g]) * 100 for g in groups], "o-", label=label)
    axes[2, 1].set_title("Collision rate (%)"); axes[2, 1].tick_params(axis="x", rotation=18)
    axes[2, 1].grid(alpha=.25); axes[2, 1].legend(fontsize=8)
    axes[2, 2].axis("off")
    table_data = []
    for p, g in zip(policies, groups):
        outcomes = [_exclusive_outcome(r) for r in g]
        rates = [100 * outcomes.count(name) / len(g)
                 for name in ("success", "rrcr", "rwcr", "rhcr", "tor")]
        times = np.asarray([r["completion_time_seconds"] for r in g], float)
        times = times[np.isfinite(times)]
        # A mean over an empty success set is undefined. State the reason in
        # the table instead of leaving the cell blank or inventing a zero time.
        avg_time = f"{times.mean():.1f}" if times.size else f"N/A (0/{len(g)})"
        table_data.append([p, *(f"{rate:.1f}%" for rate in rates), avg_time])
    table = axes[2, 2].table(cellText=table_data,
                             colLabels=["Policy", "Success/\nCov", "RRCR", "RWCR",
                                        "RHCR", "TOR", "Avg success\ntime (s)"], loc="center")
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.65)
    # Policy labels are considerably longer than the numeric summary values.
    # Give their cells enough room while keeping the table within its subplot.
    column_widths = (0.42, 0.12, 0.075, 0.075, 0.075, 0.075, 0.16)
    for (row, column), cell in table.get_celld().items():
        cell.set_width(column_widths[column])
    axes[2, 2].set_title("Summary")
    fig.suptitle("mrCPP policy benchmark", fontsize=16)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="policy checkpoint to evaluate; repeat to compare several")
    parser.add_argument("--label", action="append", default=None,
                        help="display name per --checkpoint (default: parent/file name)")
    parser.add_argument("--bcd", choices=("tour", "local"), action="append", default=[],
                        help="also evaluate the BCD expert: 'tour' = centralised BCD planned "
                             "on the known map, 'local' = nearest uncovered cell in memory")
    parser.add_argument("--maps", type=int, default=None,
                        help="evaluation map-bank size (overrides env.num_maps)")
    parser.add_argument("--map-seed", type=int, default=None,
                        help="evaluation map-bank seed (overrides env.map_seed); a seed "
                             "other than the training one gives unseen layouts")
    parser.add_argument("--humans", type=int, default=8,
                        help="humans during evaluation (default: 8)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_results")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--chunk-steps", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "metal"), default="auto")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using tanh(policy mean)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--policy-only", action="store_true", help="disable recovery after restoring checkpoint settings")
    mode.add_argument("--compare-policy-only", action="store_true", help="evaluate with and without recovery using identical seeds")
    parser.add_argument("--recovery-trace", type=Path, help="write up to 200 blocked-pose JSONL samples per policy")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.batch_size <= 0 or args.chunk_steps <= 0 or args.humans < 0:
        raise SystemExit("episodes, batch-size and chunk-steps must be positive; humans cannot be negative")
    device = select_device(None if args.backend == "auto" else args.backend)
    print(f"Device: {describe(device)}")
    if not args.checkpoint and not args.bcd:
        raise SystemExit("Give at least one --checkpoint or --bcd")
    labels = args.label or [f"{c.parent.name}/{c.name}" for c in args.checkpoint]
    if len(labels) != len(args.checkpoint):
        raise SystemExit("Give one --label per --checkpoint, or none")
    specs = [PolicySpec(label, args.humans, path)
             for label, path in zip(labels, args.checkpoint)]
    specs += [PolicySpec(f"BCD ({rule})", args.humans, bcd_rule=rule)
              for rule in dict.fromkeys(args.bcd)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.recovery_trace is not None:
        args.recovery_trace.parent.mkdir(parents=True, exist_ok=True)
        args.recovery_trace.write_text('')
    started = time.time()
    rows = []
    for spec in specs:
        # The BCD expert never uses the recovery fallback: evaluate it once.
        modes = ([None] if spec.bcd_rule is not None
                 else [False, True] if args.compare_policy_only else [args.policy_only])
        for policy_only in modes:
            suffix = ('' if policy_only is None
                      else '/policy-only' if policy_only else '/with-recovery')
            named = PolicySpec(spec.name + suffix, spec.humans, spec.checkpoint, spec.bcd_rule)
            rows.extend(evaluate_marl(
                named, args.config, args.episodes, args.seed, args.max_steps,
                args.batch_size, args.chunk_steps, args.stochastic, device,
                args.progress_every, bool(policy_only), args.recovery_trace,
                args.maps, args.map_seed,
            ))
    raw_path = args.output_dir / "episodes.csv"
    write_csv(raw_path, rows, FIELDS)
    summary = summarize(rows)
    summary_fields = list(dict.fromkeys(k for row in summary for k in row))
    write_csv(args.output_dir / "summary.csv", summary, summary_fields)
    metadata = {
        "episodes_per_policy": args.episodes, "seed": args.seed,
        "evaluation_humans": args.humans,
        "config": str(args.config.resolve()), "backend": describe(device),
        "policy_only": args.policy_only, "compare_policy_only": args.compare_policy_only,
        "stochastic": args.stochastic, "elapsed_seconds": time.time() - started,
        "maps": args.maps, "map_seed": args.map_seed,
        "policies": [s.__dict__ | {"checkpoint": str(s.checkpoint.resolve()) if s.checkpoint else None}
                     for s in specs],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved raw data, summary and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
