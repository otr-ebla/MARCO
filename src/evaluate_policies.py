#!/usr/bin/env python3
"""Benchmark BOSCO and BOSCO-guided MARL policies and save result data.

The default experiment evaluates 1,000 episodes for each of:
  * the learning-free BOSCO route controller without humans,
  * the learning-free BOSCO route controller with eight humans,
  * BOSCO-guided MARL trained without humans, and
  * BOSCO-guided MARL trained with eight humans.

Raw episode data, aggregate statistics and run metadata are written to the output
directory. MARL environments run in compiled accelerator batches; plain BOSCO
uses the same host-side expert controller as the visual test. Figures are generated
separately by ``python -m src.plot_evaluation_results``.
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

from src.algorithms.bosco_guide import make_guides
from src.algorithms.bosco import BoscoExpert
from src.algorithms.jax_bosco import JaxGuideState, jax_guide_step
from src.algorithms.mappo import RunningMeanStd, rms_normalize
from src.envs.vec_env import VecEnv
from src.envs.coverage_vector_env import MultiRobotCoverageEnv
from src.models.actor_critic import Actor
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "mappo_baseline.yaml"
DEFAULT_NO_HUMANS = ROOT / "checkpoints/archive/no_humans_update1970/checkpoint_bosco.pkl"
DEFAULT_HUMANS8 = ROOT / "checkpoints/archive/humans8_perfect_update4440/checkpoint_bosco.pkl"

FIELDS = [
    "policy", "episode", "seed", "num_humans", "steps",
    "completion_time_steps", "completion_time_seconds", "coverage_rate",
    "covered_cells", "completion", "timeout", "collision_end", "team_return",
    "wall_collisions", "robot_collisions", "human_collisions", "all_collisions",
    "wall_collision_rate", "robot_collision_rate", "human_collision_rate",
    "all_collision_rate", "revisits", "cell_entries", "revisit_rate",
    "sweep_efficiency",
]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str
    humans: int
    checkpoint: Path | None = None


def _env_config(path: Path, humans: int, max_steps: int | None) -> tuple[dict, dict]:
    config = load_config(str(path))
    env_cfg = dict(config.get("env", {}))
    env_cfg["num_humans"] = humans
    if max_steps is not None:
        env_cfg["max_steps"] = max_steps
    return config, env_cfg


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


def _guided_initial_state(vec_env: VecEnv, key: jax.Array):
    state, obs, _, _ = vec_env.reset(key)
    guides = make_guides(vec_env.env, vec_env.E)
    pos = np.asarray(state.robot_positions)
    cov = np.asarray(state.coverage_grid)
    e_count, n_robots = vec_env.E, vec_env.num_robots
    max_tour_len = 2048
    tours = np.full((e_count, n_robots, max_tour_len), -1, np.int32)
    lens = np.zeros((e_count, n_robots), np.int32)
    indices = np.zeros((e_count, n_robots), np.int32)
    targets = np.full((e_count, n_robots), -1, np.int32)
    assignments = np.zeros((e_count, n_robots, vec_env.grid_h, vec_env.grid_w), np.float32)
    for e, guide in enumerate(guides):
        guide.reset(pos[e])
        owner = guide.owner.reshape(vec_env.grid_h, vec_env.grid_w)
        assignments[e] = np.stack([owner == r for r in range(n_robots)])
        targets[e], _ = guide.update(pos[e], cov[e])
        for r, tour in enumerate(guide.tours):
            length = min(len(tour), max_tour_len)
            lens[e, r] = length
            tours[e, r, :length] = tour[:length]
            indices[e, r] = min(int(guide.idx[r]), length)
    graph = guides[0].graph
    coords = graph.centers[np.maximum(targets, 0)]
    target_coords = np.where((targets >= 0)[..., None], coords, pos)
    state, obs, _ = vec_env.update_bosco(state, jnp.asarray(target_coords))
    state, obs, _ = vec_env.update_cell_assignments(state, jnp.asarray(assignments))
    cell_size = vec_env.env.cell_size
    col = np.clip((pos[..., 0] / cell_size).astype(np.int32), 0, vec_env.grid_w - 1)
    row = np.clip((pos[..., 1] / cell_size).astype(np.int32), 0, vec_env.grid_h - 1)
    guide_state = JaxGuideState(
        target=jnp.asarray(targets), prev_cell=jnp.asarray(row * vec_env.grid_w + col),
        fail_cov=jnp.full((e_count, n_robots), -1, jnp.int32),
        idx=jnp.asarray(indices), tours=jnp.asarray(tours),
        tour_lens=jnp.asarray(lens),
    )
    return state, obs, guide_state, graph


def evaluate_marl(spec: PolicySpec, config_path: Path, episodes: int, seed: int,
                  max_steps: int | None, batch_size: int, chunk_steps: int,
                  stochastic: bool, device, progress_every: int) -> list[dict]:
    if spec.checkpoint is not None and not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint}")
    config, env_cfg = _env_config(config_path, spec.humans, max_steps)
    actor = params = rms = None
    checkpoint = {}
    if spec.kind == "marl":
        assert spec.checkpoint is not None
        with spec.checkpoint.open("rb") as handle:
            checkpoint = pickle.load(handle)
    guided = checkpoint.get("actor_bosco_guidance", True)
    env_cfg["actor_bosco_guidance"] = guided
    env_cfg["bosco_reward_guidance"] = checkpoint.get("bosco_reward_guidance", True)
    vec_env = VecEnv(min(batch_size, episodes), env_cfg)
    env = vec_env.env
    if spec.kind == "marl":
        model_cfg = config.get("model", {})
        actor = Actor(action_dim=env.action_dim, vec_dim=env.obs_vec_dim,
                      n_rays=env.n_rays, tail_dim=env.patch_dim,
                      lidar_embed=model_cfg.get("lidar_embed", 64),
                      hidden_size=model_cfg.get("hidden_size", 128))
        params = jax.device_put(checkpoint["actor_params"], device)
        rms = RunningMeanStd(*jax.device_put(tuple(checkpoint["obs_rms"]), device))
    if guided:
        state, obs, guide_state, graph = _guided_initial_state(
            vec_env, jax.random.PRNGKey(seed)
        )
        neighbors = jnp.asarray(graph.neighbors, jnp.int32)
        free = jnp.asarray(graph.free, jnp.bool_)
        components = jnp.asarray(graph.component, jnp.int32)
        centers = jnp.asarray(graph.centers, jnp.float32)
    else:
        state, obs, _, _ = vec_env.reset(jax.random.PRNGKey(seed))
        guide_state = None


    def run_chunk(carry, keys):
        def one(c, action_key):
            state, obs, guide_state = c
            if spec.kind == "marl":
                normalized = rms_normalize(rms, obs)
                mean, log_std = actor.apply(params, normalized.reshape(-1, env.obs_dim))
                if stochastic:
                    z = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
                    actions = jnp.tanh(z)
                else:
                    actions = jnp.tanh(mean)
                actions = actions.reshape(vec_env.E, env.num_robots, env.action_dim)
            else:
                valid = guide_state.target >= 0
                target = centers[jnp.maximum(guide_state.target, 0)]
                anchor_valid = guide_state.prev_cell >= 0
                anchor = centers[jnp.maximum(guide_state.prev_cell, 0)]
                anchor = jnp.where(anchor_valid[..., None], anchor,
                                   state.robot_positions)
                leg = target - anchor
                leg_length = jnp.maximum(jnp.linalg.norm(leg, axis=-1), 1e-6)
                direction = leg / leg_length[..., None]
                progress = jnp.clip(
                    jnp.sum((state.robot_positions - anchor) * direction, axis=-1),
                    0.0, leg_length,
                )
                aim = anchor + direction * jnp.minimum(
                    leg_length, progress + 0.25
                )[..., None]
                delta = aim - state.robot_positions
                desired = jnp.arctan2(delta[..., 1], delta[..., 0])
                angle = (desired - state.robot_headings + jnp.pi) % (2 * jnp.pi) - jnp.pi
                turn = jnp.clip(3.0 * angle / env.omega_max, -1.0, 1.0)
                # A cell-centre waypoint often changes direction by 90 degrees.
                # Driving during that turn gives this differential-drive model a
                # one-metre turning radius, much wider than a half-metre cell,
                # and used to wedge the nominal BOSCO baseline into walls.  Make
                # the adapter rotate first, then cross the next graph edge.
                distance = jnp.maximum(
                    leg_length - progress,
                    jnp.linalg.norm(target - state.robot_positions, axis=-1),
                )
                speed = jnp.where(
                    jnp.abs(angle) <= 0.10,
                    jnp.clip(distance / (env.v_max * env.dt), 0.0, 1.0),
                    0.0,
                )
                actions = jnp.stack([2.0 * speed - 1.0, turn], axis=-1)
                actions = jnp.where(valid[..., None], actions,
                                    jnp.array([-1.0, 0.0], jnp.float32))
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
            next_guide = None
            if guided:
                next_guide, waypoint, _ = jax_guide_step(
                    guide_state, next_state.robot_positions, next_state.coverage_grid, done,
                    neighbors, vec_env.grid_w, vec_env.grid_h, env.cell_size,
                    free_cells=free, graph_components=components,
                    previous_coverage_grid=state.coverage_grid,
                )
                valid = waypoint >= 0
                coords = centers[jnp.maximum(waypoint, 0)]
                target_coords = jnp.where(valid[..., None], coords, next_state.robot_positions)
                next_state, next_obs, _ = vec_env.update_bosco(next_state, target_coords)
            output = (rewards.sum(axis=-1), done, term, info["coverage_ratio"],
                      info["covered_cells"], info["complete"], info["timeout"],
                      info["wall_collision_rate"], info["robot_collision_rate"],
                      info["human_collision_rate"], jnp.sum(revisited, axis=-1),
                      jnp.sum(entered, axis=-1))
            return (next_state, next_obs, next_guide), output
        return jax.lax.scan(one, carry, keys)

    run_chunk = jax.jit(run_chunk)
    accum = _empty_accumulators(vec_env.E)
    rows: list[dict] = []
    key = jax.random.PRNGKey(seed + 1_000_003)
    started = time.time()
    print(f"{spec.name}: starting {episodes} episodes "
          f"({vec_env.E} parallel JAX environments on {describe(device)})", flush=True)
    while len(rows) < episodes:
        key, chunk_key = jax.random.split(key)
        keys = jax.random.split(chunk_key, chunk_steps)
        (state, obs, guide_state), outputs = run_chunk((state, obs, guide_state), keys)
        arrays = [np.asarray(x) for x in jax.device_get(outputs)]
        (rewards, dones, terms, coverage, covered, complete, timeout, walls,
         robots, humans, revisits, entries) = arrays
        for t in range(chunk_steps):
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


def evaluate_bosco(spec: PolicySpec, config_path: Path, episodes: int, seed: int,
                   max_steps: int | None, progress_every: int) -> list[dict]:
    """Evaluate the same host-side BOSCO expert used by ``test_visual``."""
    _, env_cfg = _env_config(config_path, spec.humans, max_steps)
    env = MultiRobotCoverageEnv(env_cfg)
    expert = BoscoExpert(env)
    step = jax.jit(env.step)
    rows = []
    started = time.time()
    print(f"{spec.name}: starting {episodes} episodes (visual BOSCO controller)",
          flush=True)
    for episode in range(episodes):
        state = env.reset(jax.random.PRNGKey(seed + episode))
        expert.reset(np.asarray(state.robot_positions),
                     map_id=int(np.asarray(state.map_id)))
        steps = 0
        team_return = wall = robot = human = revisits = entries = 0.0
        previous_cell = expert.graph.cell_of(np.asarray(state.robot_positions))
        terminated = truncated = False
        while not (terminated or truncated):
            actions = expert.act(
                np.asarray(state.robot_positions),
                np.asarray(state.robot_headings),
                np.asarray(state.coverage_grid),
            )
            covered_before = np.asarray(state.coverage_grid).reshape(-1) > 0.5
            state, rewards, terminated_array, truncated_array = step(
                state, jnp.asarray(actions)
            )
            terminated = bool(terminated_array)
            truncated = bool(truncated_array)
            current_cell = expert.graph.cell_of(np.asarray(state.robot_positions))
            entered = current_cell != previous_cell
            entries += float(np.sum(entered))
            revisits += float(np.sum(entered & covered_before[current_cell]))
            previous_cell = current_cell
            team_return += float(np.sum(np.asarray(rewards)))
            wall += float(np.sum(np.asarray(state.wall_hits)))
            robot += float(np.sum(np.asarray(state.robot_hits)))
            human += float(np.sum(np.asarray(state.human_hits)))
            steps += 1

        info = env.get_info(state)
        complete = float(info["complete"])
        rows.append(_record(
            spec.name, episode, seed, spec.humans, steps,
            float(info["coverage_ratio"]), float(info["covered_cells"]), complete,
            float(bool(truncated) and not complete), team_return, wall, robot, human,
            env.num_robots, env.dt, float(bool(terminated) and not complete),
            revisits, entries, (entries - revisits) / max(entries, 1.0),
        ))
        if progress_every and ((episode + 1) % progress_every == 0
                               or episode + 1 == episodes):
            elapsed = time.time() - started
            done_count = episode + 1
            eta = elapsed / done_count * (episodes - done_count)
            print(f"{spec.name}: {done_count}/{episodes} "
                  f"({100 * done_count / episodes:.1f}%) | "
                  f"elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min",
                  flush=True)
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
    parser.add_argument("--humans", type=int, default=8,
                        help="humans in dynamic-obstacle evaluations (default: 8); an additional plain BOSCO baseline always uses 0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-humans-checkpoint", type=Path, default=DEFAULT_NO_HUMANS)
    parser.add_argument("--humans8-checkpoint", type=Path, default=DEFAULT_HUMANS8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_results")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--chunk-steps", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "metal"), default="auto")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using tanh(policy mean)")
    parser.add_argument("--e2e-checkpoint", type=Path, default=None,
                        help="Add an end-to-end MARL policy to the comparison")
    parser.add_argument("--skip-bosco", action="store_true")
    parser.add_argument("--skip-marl", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.batch_size <= 0 or args.chunk_steps <= 0 or args.humans < 0:
        raise SystemExit("episodes, batch-size and chunk-steps must be positive; humans cannot be negative")
    device = select_device(None if args.backend == "auto" else args.backend)
    print(f"Device: {describe(device)}")
    specs = [
        PolicySpec("Plain BOSCO (0 humans)", "bosco", 0),
        PolicySpec(f"Plain BOSCO ({args.humans} humans)", "bosco", args.humans),
        PolicySpec("BOSCO MARL (trained: 0 humans)", "marl", args.humans,
                   args.no_humans_checkpoint),
        PolicySpec("BOSCO MARL (trained: 8 humans)", "marl", args.humans,
                   args.humans8_checkpoint),
    ]
    if args.e2e_checkpoint is not None:
        specs.append(PolicySpec("End-to-end MARL (CTDE)", "marl", args.humans,
                                args.e2e_checkpoint))
    if args.skip_bosco:
        specs = [s for s in specs if s.kind != "bosco"]
    if args.skip_marl:
        specs = [s for s in specs if s.kind != "marl"]
    if not specs:
        raise SystemExit("No policies selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    for spec in specs:
        if spec.kind == "bosco":
            rows.extend(evaluate_bosco(
                spec, args.config, args.episodes, args.seed, args.max_steps,
                args.progress_every,
            ))
        else:
            rows.extend(evaluate_marl(
                spec, args.config, args.episodes, args.seed, args.max_steps,
                args.batch_size, args.chunk_steps, args.stochastic, device,
                args.progress_every,
            ))
    raw_path = args.output_dir / "episodes.csv"
    write_csv(raw_path, rows, FIELDS)
    summary = summarize(rows)
    summary_fields = list(dict.fromkeys(k for row in summary for k in row))
    write_csv(args.output_dir / "summary.csv", summary, summary_fields)
    metadata = {
        "episodes_per_policy": args.episodes, "seed": args.seed,
        "dynamic_obstacle_evaluation_humans": args.humans,
        "plain_bosco_evaluation_humans": [0, args.humans],
        "config": str(args.config.resolve()), "backend": describe(device),
        "stochastic": args.stochastic, "elapsed_seconds": time.time() - started,
        "policies": [s.__dict__ | {"checkpoint": str(s.checkpoint.resolve()) if s.checkpoint else None} for s in specs],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved raw data, summary and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
