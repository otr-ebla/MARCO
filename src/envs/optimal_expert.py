"""Optimal-path expert: the teacher for imitation in the single-robot study.

The exact minimum-cost coverage path (src.envs.optimal_coverage, MILP with
HiGHS) depends on the start cell and takes seconds to minutes per map, so it
cannot run inside the jitted rollout. The environment therefore spawns the
robot only on `env.spawn_cells` fixed cells per map, and

    OptimalPlanBank   plans every (map, spawn cell) once, in parallel worker
                      processes, and caches the result on disk keyed by the
                      maps, the spawn cells and the planner settings.
    OptimalExpert     the BCD tracker (src.envs.bcd_expert) with the optimal
                      path as its tour: the target is the first cell of the
                      path, in first-visit order, that the robot's memory
                      does not hold as covered; the waypoint, the feedback law
                      and the safety filter are the same. So it labels any
                      visited state, including the ones DAgger reaches off the
                      path, with a differential-drive (v, omega) command.

The plan is picked at step 0 of each episode (the one whose start is the
spawn cell); it is the only state kept between steps, carried as the `chunk`
of the BCD expert interface so callers can use either teacher.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import jax.numpy as jnp
import numpy as np

from .bcd_expert import BCDExpert, open_edges
from .optimal_coverage import OPTIMAL, CostWeights, Grid, plan_coverage

PLAN_DEFAULTS = {
    'alpha': 1.0,                 # weight of the moves T
    'beta': 1.0,                  # weight of the revisits R
    'time_limit': 300.0,          # seconds per plan; TIME_LIMIT keeps the best path and its gap
    'workers': 0,                 # planning processes; 0 = half the CPUs
    'cache_dir': 'checkpoints/optimal_plans',
}
_CACHE_VERSION = 1


def _solve(job):
    """Worker: one plan as plain arrays (picklable)."""
    free, start, open_moves, alpha, beta, time_limit = job
    result = plan_coverage(Grid(free), start, CostWeights(alpha, beta),
                           time_limit=time_limit, open_moves=open_moves)
    if result.path is None:
        raise RuntimeError(f'No coverage path from {start}: {result.message}')
    return dict(path=np.asarray(result.path, np.int32), status=result.status,
                cost=result.cost, moves=result.moves, revisits=result.revisits,
                lower_bound=result.lower_bound, seconds=result.solve_seconds)


class OptimalPlanBank:
    """Optimal coverage paths for every (map, spawn cell) of an environment.

    Arrays, M maps x K spawn cells, cells flat (row * W + col), -1 padded:
    `start` (M, K), `path` (M, K, P) the walk, `order` (M, K, L) its cells in
    first-visit order; `status`, `moves`, `revisits`, `cost`, `lower_bound`,
    `seconds` (M, K).
    """

    def __init__(self, env, config: dict | None = None, verbose: bool = True):
        cfg = {**PLAN_DEFAULTS, **(config or {})}
        unknown = set(cfg) - set(PLAN_DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown optimal planner keys: {sorted(unknown)}')
        if getattr(env, 'spawn_cells', 0) <= 0:
            raise ValueError('The optimal expert needs env.spawn_cells > 0 '
                             '(plans are computed offline per spawn cell)')
        walls = np.asarray(env.walls)
        free = env.free_mask_np > .5
        moves = np.stack([open_edges(free[m], walls[m], env.cell_size) for m in range(env.num_maps)])
        spawns = env.spawn_cells_np                                       # (M, K, 2)
        time_limit = float(cfg['time_limit']) or None
        alpha, beta = float(cfg['alpha']), float(cfg['beta'])

        digest = hashlib.sha1()
        for part in (free, moves, spawns):
            digest.update(np.ascontiguousarray(part).tobytes())
        digest.update(repr((_CACHE_VERSION, alpha, beta, time_limit)).encode())
        path = os.path.join(cfg['cache_dir'], f'optimal_plans_{digest.hexdigest()[:16]}.npz')
        if os.path.exists(path):
            data = dict(np.load(path))
            if verbose:
                print(f'Optimal plans: loaded {path}')
        else:
            data = self._plan(free, moves, spawns, alpha, beta, time_limit,
                              int(cfg['workers']), verbose)
            os.makedirs(cfg['cache_dir'], exist_ok=True)
            np.savez_compressed(path, **data)
            if verbose:
                print(f'Optimal plans: saved {path}')
        self.path_file = path
        for name, value in data.items():
            setattr(self, name, value)
        if verbose:
            print(self.describe())

    @staticmethod
    def _plan(free, moves, spawns, alpha, beta, time_limit, workers, verbose) -> dict:
        m_maps, k_starts, _ = spawns.shape
        w = free.shape[2]
        jobs = [(free[m], tuple(int(v) for v in spawns[m, k]), moves[m], alpha, beta, time_limit)
                for m in range(m_maps) for k in range(k_starts)]
        workers = workers or max(1, (os.cpu_count() or 2) // 2)
        began = time.time()
        if verbose:
            print(f'Optimal plans: {len(jobs)} (map, start) pairs on {workers} processes, '
                  f'time limit {time_limit} s each ...', flush=True)
        results = []
        # 'spawn': forked workers would inherit the parent's JAX runtime.
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=get_context('spawn')) as pool:
            for i, result in enumerate(pool.map(_solve, jobs)):
                results.append(result)
                if verbose and ((i + 1) % 16 == 0 or i + 1 == len(jobs)):
                    print(f'  {i + 1}/{len(jobs)} plans, {time.time() - began:.0f} s', flush=True)

        def first_visits(walk):
            _, index = np.unique(walk, return_index=True)
            return walk[np.sort(index)]

        flat = [r['path'][:, 0] * w + r['path'][:, 1] for r in results]
        orders = [first_visits(p) for p in flat]

        def stack(rows):
            size = max(len(r) for r in rows)
            out = np.stack([np.pad(r, (0, size - len(r)), constant_values=-1) for r in rows])
            return out.reshape(m_maps, k_starts, size)

        def field(name, dtype):
            return np.asarray([r[name] for r in results], dtype).reshape(m_maps, k_starts)
        return dict(
            start=(spawns[..., 0] * w + spawns[..., 1]).astype(np.int32),
            path=stack(flat), order=stack(orders),
            status=field('status', 'U16'), moves=field('moves', np.int32),
            revisits=field('revisits', np.int32), cost=field('cost', np.float64),
            lower_bound=field('lower_bound', np.float64), seconds=field('seconds', np.float64))

    def describe(self) -> str:
        statuses, counts = np.unique(self.status, return_counts=True)
        gap = (self.cost - self.lower_bound) / np.maximum(self.cost, 1e-12)
        return (f'Optimal plans: {self.status.size} | '
                + ', '.join(f'{s} {c}' for s, c in zip(statuses, counts))
                + f' | moves {self.moves.mean():.1f} | revisits {self.revisits.mean():.2f} '
                f'| max gap {gap.max():.2%} | solve median {np.median(self.seconds):.1f} s')

    @property
    def all_optimal(self) -> bool:
        return bool(np.all(self.status == OPTIMAL))


class OptimalExpert(BCDExpert):
    """BCD tracker following the optimal path of the robot's spawn cell. Single robot."""

    def __init__(self, env, bank: OptimalPlanBank, config: dict | None = None):
        if env.num_robots != 1:
            raise ValueError('The optimal expert is single-robot')
        super().__init__(env, {**(config or {}), 'target_rule': 'tour'})
        self.target_rule = 'optimal'
        self.bank = bank
        self.plan_start = jnp.asarray(bank.start)                     # (M, K)
        self.plan_order = jnp.asarray(bank.order)                     # (M, K, L)

    def assign(self, state):
        """(N,) the plan whose start is nearest to the spawn (the spawn itself)."""
        cells, _, _ = self._cells(state)
        m = state.map_id
        cost = self.dist[m][cells[:, None], self.plan_start[m][None, :]]
        return jnp.argmin(cost, axis=1).astype(jnp.int32)

    def targets(self, state, chunk):
        """The first cell of the plan, in first-visit order, that memory does
        not hold as covered; past the end of the plan, the nearest uncovered cell."""
        cells, known_covered, reach, other, active = self.remaining(state)
        order = self.plan_order[state.map_id, chunk]                   # (N, L)
        flat = jnp.maximum(order, 0)
        pending = ((order >= 0) & ~jnp.take_along_axis(known_covered, flat, axis=1)
                   & (flat != cells[:, None]))
        planned = flat[self.env._robot_ids, jnp.argmax(pending, axis=1)]
        nearest = jnp.argmin(jnp.where(other, reach, jnp.iinfo(jnp.int32).max), axis=1)
        return jnp.where(jnp.any(pending, axis=1), planned, nearest), active


def build_teacher(env, pretrain_cfg: dict, teacher: str | None = None):
    """The imitation teacher named by `teacher` or pretrain.teacher: 'bcd' or 'optimal'."""
    teacher = teacher or pretrain_cfg.get('teacher', 'bcd')
    expert_cfg = pretrain_cfg.get('expert', {})
    if teacher == 'bcd':
        return BCDExpert(env, expert_cfg)
    if teacher == 'optimal':
        return OptimalExpert(env, OptimalPlanBank(env, pretrain_cfg.get('optimal', {})), expert_cfg)
    raise ValueError(f"teacher must be 'bcd' or 'optimal', not {teacher!r}")


def main() -> None:
    """Plan (or load) the optimal-path bank of a config's maps and test the expert on it."""
    import argparse

    import jax

    from src.envs.bcd_expert import expert_steps, first_episode_summary
    from src.envs.curriculum_maps import with_stage_maps
    from src.envs.vec_env import VecEnv
    from src.train_marl import build_env_config
    from src.utils.config_parser import load_config

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--config', default='config/single_robot.yaml')
    parser.add_argument('--maps', type=int, default=None, help='map-bank size (env.num_maps)')
    parser.add_argument('--map-seed', type=int, default=None,
                        help='map-bank seed; the evaluation bank uses its own (e.g. 1000)')
    parser.add_argument('--stage', type=int, default=None,
                        help='plan on the maps of this curriculum stage (src.envs.curriculum_maps)')
    parser.add_argument('--no-rollout', action='store_true',
                        help='only plan; skip running the expert in the environment')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.map_seed is not None:
        config['env']['map_seed'] = args.map_seed
    env_cfg, _ = build_env_config(config, args.maps)
    env_cfg['fallback_enabled'] = False
    if args.stage is not None:
        env_cfg = with_stage_maps(env_cfg, args.stage)
    pretrain_cfg = config.get('pretrain', {})
    vec_env = VecEnv(1, env_cfg)
    env = vec_env.env
    expert = build_teacher(env, pretrain_cfg, 'optimal')
    if args.no_rollout:
        return
    # Episodes from random spawn cells: the differential-drive time of the plans.
    m_maps, k_starts = expert.bank.start.shape
    vec_env = VecEnv(m_maps * k_starts, env_cfg)
    expert = OptimalExpert(vec_env.env, expert.bank, pretrain_cfg.get('expert', {}))
    state, *_ = vec_env.reset(jax.random.PRNGKey(0), jnp.repeat(jnp.arange(m_maps), k_starts))
    _, stats = expert_steps(expert, vec_env, env.max_steps)((state, expert.init_chunks((vec_env.E,))))
    s = first_episode_summary(jax.device_get(stats), env.max_steps)
    print(f'Optimal expert, {vec_env.E} episodes: complete {s["complete"].mean():.2%} '
          f'| steps {s["steps"].mean():.0f} ({s["steps"].mean() * env.dt:.1f} s) '
          f'| plan moves {expert.bank.moves.mean():.1f} '
          f'({expert.bank.moves.mean() * env.cell_size / env.v_max:.1f} s holonomic at v_max) '
          f'| contacts/ep {(s["wall"] + s["robot"]).mean():.2f}')


if __name__ == '__main__':
    main()
