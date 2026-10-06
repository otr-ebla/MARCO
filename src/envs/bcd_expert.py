"""Boustrophedon decomposition expert, used as the teacher for imitation.

Planning happens once per map of the static bank, in numpy:

    decomposition  free cells -> rectangles, largest first; BSP rooms and
                   doorways come out as exact rectangles.
    lanes          each rectangle is swept boustrophedon along its longer side.
    tour           rectangles are chained greedily by geodesic distance into
                   one ordered list of every free cell.
    chunks         the tour is cut into N contiguous pieces of equal cost
                   (cells plus transit), one per robot.

At an episode start each robot takes the chunk whose start is nearest to its
spawn (exhaustive matching). Tracking is a stateless feedback law in JAX:

    target     the first cell of the robot's chunk that its own memory does
               not hold as covered, so the decision uses only coverage the
               actor can observe in its crop and personal map. A finished
               chunk falls back to the nearest uncovered cell in memory.
    waypoint   a neighbour on a shortest path to the target, preferring the
               current heading. Lanes therefore run into their end cell
               before shifting, which covers the corners.
    control    omega = omega_max * tanh(k * e),
               v = v_cruise * max(cos e, 0)^p * clearance(lidar),
               continuous in the observation, with no controller memory.
    safety     the nominal command passes command_safety_flags; a rejected
               command is replaced by DWA towards the same waypoint.

The chunk assignment is the only state kept between steps; it is recomputed
whenever `state.step_count == 0`, which is exactly the first observation of
every (auto-reset) episode.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .coverage_vector_env import COVERED
from .recovery import command_safety_flags, dwa

_FAR = 1_000_000   # geodesic distance between disconnected cells
# Neighbour order matches recovery's blocked edges: S (row-1), N (row+1), W (col-1), E (col+1).
_DR = np.array([-1, 1, 0, 0])
_DC = np.array([0, 0, -1, 1])
EXPERT_DEFAULTS = {
    'v_cruise': 0.8,          # m/s on open lanes
    'k_omega': 2.0,           # omega = omega_max * tanh(k_omega * heading error)
    'heading_power': 3.0,     # v scales with max(cos e, 0) ** heading_power
    'stop_margin': 0.05,      # m of free straight travel at which v reaches zero
    'clearance_ramp': 0.4,    # m over which v recovers to full speed
    'lookahead': 0.3,         # m, pursuit distance along the cell centreline
    'yield_radius': 0.9,      # m, a blocked robot gives way to a lower-id teammate
    'target_rule': 'local',   # 'local': nearest uncovered in memory, tour order breaks ties
}


# ---------------------------------------------------------------------------
# Offline planning (numpy)
# ---------------------------------------------------------------------------

def open_edges(free: np.ndarray, walls: np.ndarray, cell_size: float) -> np.ndarray:
    """(H, W, 4) bool: both cells free and no wall between their centres."""
    h, w = free.shape
    rows, cols = np.indices((h, w))
    edges = np.zeros((h, w, 4), bool)
    x0, y0, x1, y1 = (walls[:, k] for k in range(4))
    for k in range(4):
        nr, nc = rows + _DR[k], cols + _DC[k]
        inside = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        ok = free & inside & free[np.clip(nr, 0, h - 1), np.clip(nc, 0, w - 1)]
        # Axis-aligned segment between the two centres against every wall.
        ax = (cols + .5) * cell_size
        ay = (rows + .5) * cell_size
        bx = (np.clip(nc, 0, w - 1) + .5) * cell_size
        by = (np.clip(nr, 0, h - 1) + .5) * cell_size
        lo_x, hi_x = np.minimum(ax, bx)[..., None], np.maximum(ax, bx)[..., None]
        lo_y, hi_y = np.minimum(ay, by)[..., None], np.maximum(ay, by)[..., None]
        if k < 2:   # vertical segment at constant x
            cross = (x0 < ax[..., None]) & (ax[..., None] < x1) & (y0 < hi_y) & (y1 > lo_y)
        else:       # horizontal segment at constant y
            cross = (y0 < ay[..., None]) & (ay[..., None] < y1) & (x0 < hi_x) & (x1 > lo_x)
        edges[..., k] = ok & ~np.any(cross, axis=-1)
    return edges


def geodesic_table(edges: np.ndarray) -> np.ndarray:
    """(C, C) int32 four-connected path length between cells; _FAR if disconnected."""
    h, w, _ = edges.shape
    rows, cols = np.indices((h, w))
    src, dst = [], []
    for k in range(4):
        ok = edges[..., k]
        src.append((rows * w + cols)[ok])
        dst.append(((rows + _DR[k]) * w + cols + _DC[k])[ok])
    src, dst = np.concatenate(src), np.concatenate(dst)
    graph = csr_matrix((np.ones(src.size), (src, dst)), shape=(h * w, h * w))
    dist = shortest_path(graph, method='D', unweighted=True)
    return np.where(np.isfinite(dist), dist, _FAR).astype(np.int32)


def rectangles(free: np.ndarray, edges: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedy partition of the free cells into internally connected rectangles.

    Returns (r0, c0, r1, c1), inclusive. Thin strips (a side under three cells)
    are taken last so a doorway column cannot slice two rooms apart.
    """
    h, w = free.shape
    remaining = free.copy()
    result = []
    while remaining.any():
        best, best_score = None, -1.0
        for r0 in range(h):
            column = np.ones(w, bool)      # rows r0..r1 usable in this column
            joined = np.ones(w, bool)      # rows r0..r1 open towards column + 1
            for r1 in range(r0, h):
                column &= remaining[r1]
                if r1 > r0:
                    column &= edges[r1 - 1, :, 1]
                joined &= edges[r1, :, 3]
                if not column.any():
                    break
                c = 0
                while c < w:
                    if not column[c]:
                        c += 1
                        continue
                    start = c
                    while c + 1 < w and column[c + 1] and joined[c]:
                        c += 1
                    height, width = r1 - r0 + 1, c - start + 1
                    score = height * width * (1.0 if min(height, width) >= 3 else .25)
                    if score > best_score:
                        best, best_score = (r0, start, r1, c), score
                    c += 1
        r0, c0, r1, c1 = best
        remaining[r0:r1 + 1, c0:c1 + 1] = False
        result.append(best)
    return result


def sweep(rect: tuple[int, int, int, int], corner: int) -> list[tuple[int, int]]:
    """Boustrophedon (row, col) order over a rectangle, lanes along its longer side.

    corner: bit 0 starts at the high row, bit 1 at the high column.
    """
    r0, c0, r1, c1 = rect
    rows = list(range(r0, r1 + 1))[::-1 if corner & 1 else 1]
    cols = list(range(c0, c1 + 1))[::-1 if corner & 2 else 1]
    order = []
    if c1 - c0 >= r1 - r0:          # lanes are rows
        for k, r in enumerate(rows):
            order.extend((r, c) for c in (cols if k % 2 == 0 else cols[::-1]))
    else:                            # lanes are columns
        for k, c in enumerate(cols):
            order.extend((r, c) for r in (rows if k % 2 == 0 else rows[::-1]))
    return order


def coverage_tour(free: np.ndarray, edges: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Flat indices of every free cell, rectangles chained nearest-first."""
    w = free.shape[1]
    regions = rectangles(free, edges)
    tour = sweep(regions.pop(0), 0)
    while regions:
        here = tour[-1][0] * w + tour[-1][1]
        options = [(dist[here, s[0][0] * w + s[0][1]], i, s)
                   for i, rect in enumerate(regions)
                   for s in (sweep(rect, corner) for corner in range(4))]
        _, index, order = min(options, key=lambda o: (o[0], o[1]))
        regions.pop(index)
        tour.extend(order)
    return np.array([r * w + c for r, c in tour], np.int32)


def split_tour(tour: np.ndarray, dist: np.ndarray, parts: int) -> np.ndarray:
    """(parts + 1,) boundaries giving each piece an equal share of tour cost."""
    step = np.concatenate([[1], np.minimum(dist[tour[:-1], tour[1:]], tour.size)])
    cost = np.cumsum(step)
    cuts = [int(np.searchsorted(cost, cost[-1] * k / parts, side='right')) for k in range(1, parts)]
    bounds = np.array([0, *cuts, tour.size], np.int32)
    return np.maximum.accumulate(bounds)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class BCDExpert:
    """Per-map plans for an environment's map bank and a jittable tracker.

    `act` handles one environment; vmap it over a VecEnv batch.
    """

    def __init__(self, env, config: dict | None = None):
        cfg = {**EXPERT_DEFAULTS, **(config or {})}
        unknown = set(cfg) - set(EXPERT_DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown expert keys: {sorted(unknown)}')
        self.env = env
        self.v_cruise = min(float(cfg['v_cruise']), env.v_max)
        self.k_omega = float(cfg['k_omega'])
        self.heading_power = float(cfg['heading_power'])
        self.stop_margin = float(cfg['stop_margin'])
        self.clearance_ramp = float(cfg['clearance_ramp'])
        self.lookahead = float(cfg['lookahead'])
        self.yield_radius = float(cfg['yield_radius'])
        self.target_rule = cfg['target_rule']
        if self.target_rule not in ('local', 'tour'):
            raise ValueError("target_rule must be 'local' or 'tour'")
        if min(self.v_cruise, self.k_omega, self.heading_power, self.clearance_ramp,
               self.lookahead, self.yield_radius) <= 0:
            raise ValueError('Expert speed, gain, power, ramp and lookahead must be positive')

        n = env.num_robots
        walls = np.asarray(env.walls)
        tours, bounds, dists, edge_list = [], [], [], []
        for m in range(env.num_maps):
            free = env.free_mask_np[m] > .5
            edges = open_edges(free, walls[m], env.cell_size)
            dist = geodesic_table(edges)
            tour = coverage_tour(free, edges, dist)
            if tour.size != int(free.sum()):
                raise RuntimeError(f'Map {m}: tour covers {tour.size} of {int(free.sum())} cells')
            tours.append(tour)
            bounds.append(split_tour(tour, dist, n))
            dists.append(dist)
            edge_list.append(edges)
        length = max(t.size for t in tours)
        self.tour_np = np.stack([np.pad(t, (0, length - t.size), constant_values=-1) for t in tours])
        self.bounds_np = np.stack(bounds)
        self.tour = jnp.asarray(self.tour_np)
        self.bounds = jnp.asarray(self.bounds_np)
        rank = np.full((env.num_maps, env.num_cells), -1, np.int32)
        for m, tour in enumerate(tours):
            rank[m, tour] = np.arange(tour.size)
        self.rank = jnp.asarray(rank)                    # position of each cell in its tour
        self.dist = jnp.asarray(np.stack(dists))
        self.edges = jnp.asarray(np.stack(edge_list))
        self.free = jnp.asarray(env.free_mask_np.reshape(env.num_maps, -1) > .5)
        # Exhaustive matching stays cheap up to five robots; beyond, robot i takes chunk i.
        perms = (list(itertools.permutations(range(n))) if n <= 5 else [tuple(range(n))])
        self._perms = jnp.asarray(perms, jnp.int32)

    def init_chunks(self, batch: tuple[int, ...] = ()) -> jax.Array:
        return jnp.zeros((*batch, self.env.num_robots), jnp.int32)

    def _cells(self, state):
        col, row = self.env._pos_to_cell(state.robot_positions)
        return row * self.env.grid_w + col, row, col

    def assign(self, state) -> jax.Array:
        """(N,) chunk per robot minimising total spawn-to-chunk-start distance."""
        m = state.map_id
        cells, _, _ = self._cells(state)
        starts = self.tour[m, self.bounds[m, :-1]]
        cost = self.dist[m][cells[:, None], starts[None, :]]               # (N, N)
        total = jnp.sum(cost[self.env._robot_ids[None], self._perms], axis=1)
        return self._perms[jnp.argmin(total)]

    def targets(self, state, chunk):
        """(N,) target cell (flat) and whether any work remains for each robot.

        'local': the nearest cell the robot's memory holds as uncovered, ties
        broken by the smallest tour-order gap to the robot's own cell (the
        next cell of the lane, or of the next lane at a lane end). Everything
        it depends on is in the actor's observation or fixed by the geometry.
        'tour': the first uncovered cell of the robot's assigned chunk, then
        the nearest uncovered cell. Faster as a team, but the chunk itself is
        invisible to the actor.
        """
        env = self.env
        m = state.map_id
        n = env.num_robots
        cells, _, _ = self._cells(state)
        known_covered = (state.mem_state == COVERED).reshape(n, -1)
        reach = self.dist[m][cells]                                         # (N, C)
        other = (self.free[m][None] & ~known_covered & (reach < _FAR)
                 & (jnp.arange(env.num_cells)[None] != cells[:, None]))
        active = jnp.any(other, axis=1)
        if self.target_rule == 'local':
            rank = self.rank[m]
            gap = jnp.abs(rank[None, :] - jnp.maximum(rank[cells], 0)[:, None])
            score = reach * (2 * env.num_cells) + gap
            return jnp.argmin(jnp.where(other, score, jnp.iinfo(jnp.int32).max), axis=1), active

        tour = self.tour[m]
        index = jnp.arange(tour.size)
        lo = self.bounds[m, chunk]
        hi = self.bounds[m, chunk + 1]
        flat_tour = jnp.maximum(tour, 0)
        pending = ((index >= lo[:, None]) & (index < hi[:, None]) & (tour >= 0)[None]
                   & ~known_covered[:, flat_tour] & (flat_tour[None] != cells[:, None]))
        in_chunk = jnp.any(pending, axis=1)
        chunk_target = flat_tour[jnp.argmax(pending, axis=1)]
        nearest = jnp.argmin(jnp.where(other, reach, _FAR + 1), axis=1)
        return jnp.where(in_chunk, chunk_target, nearest), active

    def waypoints(self, state, target):
        """(N, 2) metres: pursuit point on the line from the own cell centre to
        the next cell on a shortest path (heading-aligned among equal paths),
        `lookahead` ahead of the robot's projection on it. Converging to the
        cell centreline keeps door jambs and walls outside the robot's lane."""
        env = self.env
        m = state.map_id
        cells, row, col = self._cells(state)
        dr, dc = jnp.asarray(_DR), jnp.asarray(_DC)
        nr = jnp.clip(row[:, None] + dr, 0, env.grid_h - 1)
        nc = jnp.clip(col[:, None] + dc, 0, env.grid_w - 1)
        neighbours = nr * env.grid_w + nc                                    # (N, 4)
        here = self.dist[m][cells, target]
        ahead = self.dist[m][neighbours, target[:, None]]
        ok = self.edges[m][row, col] & (ahead == here[:, None] - 1) & (here < _FAR)[:, None]
        align = (jnp.cos(state.robot_headings)[:, None] * dc
                 + jnp.sin(state.robot_headings)[:, None] * dr)
        best = jnp.argmax(jnp.where(ok, align, -jnp.inf), axis=1)
        routed = jnp.any(ok, axis=1)
        nxt = jnp.where(routed, neighbours[env._robot_ids, best], target)

        def centre(flat):
            return (jnp.stack([flat % env.grid_w, flat // env.grid_w], axis=-1) + .5) * env.cell_size
        start, goal = centre(cells), centre(nxt)
        direction = jnp.stack([dc[best], dr[best]], axis=-1).astype(jnp.float32)
        offset = state.robot_positions - start
        along = jnp.sum(offset * direction, axis=-1)
        lateral = jnp.abs(offset[:, 0] * direction[:, 1] - offset[:, 1] * direction[:, 0])
        # The lookahead shrinks with the distance from the centreline: after a
        # turn the robot first rejoins the line (turning near the cell centre)
        # instead of cutting diagonally past a wall end. Never aim past the far
        # side of the next cell.
        shrink = jnp.clip(1. - lateral / (.4 * env.cell_size), 0., 1.)
        reach = jnp.minimum(along + self.lookahead * shrink, 1.5 * env.cell_size)
        pursuit = start + direction * reach[:, None]
        return jnp.where(routed[:, None], pursuit, goal)

    def free_travel(self, state) -> jax.Array:
        """(N,) metres the disk can drive straight ahead before touching a lidar hit."""
        env = self.env
        ranges = state.lidar * env.max_lidar_range                         # (N, R)
        forward = ranges * jnp.cos(env._ray_angles)[None]
        lateral = ranges * jnp.sin(env._ray_angles)[None]
        hit = (state.lidar < 1.) & (forward > 0.) & (jnp.abs(lateral) < env.robot_radius)
        travel = forward - jnp.sqrt(jnp.maximum(env.robot_radius ** 2 - lateral ** 2, 0.))
        return jnp.min(jnp.where(hit, travel, env.max_lidar_range), axis=1)

    def yield_points(self, state, point):
        """Right of way by id, from positions only (the actor sees teammate ids
        and offsets in its communication slots).

        Robot i yields when a lower-id teammate j is within `yield_radius` and
        i's pursuit point would bring it closer to j; it then heads for its own
        or a neighbouring cell centre, whichever is farthest from j. Returns
        (N,) yielding and (N, 2) aside points.
        """
        env = self.env
        _, row, col = self._cells(state)
        pos = state.robot_positions
        d2 = env._pairwise_sq_dist(pos)
        ids = env._robot_ids
        near = ((d2 < self.yield_radius ** 2) & (ids[None, :] < ids[:, None])
                & state.robot_alive[None, :])
        other = jnp.argmin(jnp.where(near, d2, jnp.inf), axis=1)
        gap = jnp.linalg.norm(pos - pos[other], axis=-1)
        closing = jnp.linalg.norm(point - pos[other], axis=-1) < gap - .05
        yielding = jnp.any(near, axis=1) & closing & state.robot_alive

        dr = jnp.asarray(np.concatenate([[0], _DR]))
        dc = jnp.asarray(np.concatenate([[0], _DC]))
        nr = jnp.clip(row[:, None] + dr, 0, env.grid_h - 1)
        nc = jnp.clip(col[:, None] + dc, 0, env.grid_w - 1)
        centres = (jnp.stack([nc, nr], axis=-1) + .5) * env.cell_size        # (N, 5, 2)
        ok = jnp.concatenate([jnp.ones((env.num_robots, 1), bool),
                              self.edges[state.map_id][row, col]], axis=1)
        away = jnp.linalg.norm(centres - pos[other][:, None, :], axis=-1)
        best = jnp.argmax(jnp.where(ok, away, -jnp.inf), axis=1)
        return yielding, centres[ids, best]

    def control(self, state, point, active):
        """Nominal physical (v, omega) towards `point`, continuous in the observation."""
        env = self.env
        delta = point - state.robot_positions
        error = jnp.arctan2(delta[:, 1], delta[:, 0]) - state.robot_headings
        error = jnp.arctan2(jnp.sin(error), jnp.cos(error))
        omega = env.omega_max * jnp.tanh(self.k_omega * error)
        clearance = jnp.clip((self.free_travel(state) - self.stop_margin) / self.clearance_ramp, 0., 1.)
        v = self.v_cruise * jnp.maximum(jnp.cos(error), 0.) ** self.heading_power * clearance
        return jnp.where(active, v, 0.), jnp.where(active, omega, 0.)

    def local(self, state, chunk) -> jax.Array:
        """(N,) the target cell lies inside the robot's actor crop, so the
        decision is visible in its local observation."""
        env = self.env
        target, active = self.targets(state, chunk)
        _, row, col = self._cells(state)
        half = env.local_coverage_size // 2
        return (active & (jnp.abs(target // env.grid_w - row) <= half)
                & (jnp.abs(target % env.grid_w - col) <= half))

    def act(self, state, chunk):
        """One environment: (actions (N, 2) in [-1, 1], chunk (N,), safety-filtered (N,))."""
        env = self.env
        chunk = jnp.where(state.step_count == 0, self.assign(state), chunk)
        target, active = self.targets(state, chunk)
        point = self.waypoints(state, target)
        yielding, aside = self.yield_points(state, point)
        point = jnp.where((yielding & active)[:, None], aside, point)
        v, omega = self.control(state, point, active)
        command = jnp.stack([v, omega], axis=-1)

        free = self.free[state.map_id].reshape(env.grid_h, env.grid_w)
        # The planner's true wall edges (the belief map holds no edges).
        edges = jnp.broadcast_to(~self.edges[state.map_id], (env.num_robots, env.grid_h, env.grid_w, 4))

        def safe_command(pos, heading, velocity, lidar, cmd, goal, blocked):
            flags = command_safety_flags(env, pos, heading, cmd, lidar, free, blocked)
            fallback = dwa(env, pos, heading, velocity, lidar, free, goal, blocked)
            physical = jnp.array([(fallback[0] + 1.) * .5 * env.v_max, fallback[1] * env.omega_max])
            return jnp.where(flags == 0, cmd, physical), flags != 0

        command, filtered = jax.vmap(safe_command)(
            state.robot_positions, state.robot_headings, state.robot_velocities,
            state.lidar, command, point, edges)
        filtered = filtered & active
        actions = jnp.stack([2. * command[:, 0] / env.v_max - 1.,
                             command[:, 1] / env.omega_max], axis=-1)
        return jnp.clip(actions, -1., 1.), chunk, filtered


def expert_steps(expert: BCDExpert, vec_env, num_steps: int):
    """Jitted scan running the expert alone on a VecEnv: (carry, key) -> (carry, stats).

    carry = (state, chunk); stats per step: coverage (E,), done (E,), complete
    (E,), wall/robot hits (E,), filtered fraction (E,).
    """
    act = jax.vmap(expert.act)

    def step(carry, _):
        state, chunk = carry
        actions, chunk, filtered = act(state, chunk)
        state, _, _, _, done, info, _ = vec_env.step(state, actions)
        return (state, chunk), dict(
            coverage=info['coverage_ratio'], done=done, complete=info['complete'],
            wall=info['wall_collision_rate'], robot=info['robot_collision_rate'],
            filtered=jnp.mean(filtered.astype(jnp.float32), axis=-1))

    @jax.jit
    def run(carry):
        return jax.lax.scan(step, carry, None, length=num_steps)
    return run


def first_episode_summary(stats: dict, max_steps: int) -> dict:
    """Per-environment results of the first episode in a stack of expert_steps stats."""
    done = np.asarray(stats['done']).astype(bool)                       # (T, E)
    ended = done.any(axis=0)
    last = np.where(ended, done.argmax(axis=0), done.shape[0] - 1)
    envs = np.arange(done.shape[1])
    upto = np.arange(done.shape[0])[:, None] <= last[None]
    return dict(
        coverage=np.asarray(stats['coverage'])[last, envs],
        complete=np.asarray(stats['complete'])[last, envs] > .5,
        steps=last + 1,
        ended=ended,
        wall=(np.asarray(stats['wall']) * upto).sum(axis=0),
        robot=(np.asarray(stats['robot']) * upto).sum(axis=0),
        filtered=(np.asarray(stats['filtered']) * upto).sum(axis=0) / (last + 1),
    )
