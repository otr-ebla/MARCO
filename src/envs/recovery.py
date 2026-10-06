"""JIT-compatible recovery using only a robot's map memory and current lidar."""

import math

import jax
import jax.numpy as jnp


def run_if(active, function, default):
    """Skip recovery work when all vmapped environments are inactive.

    A vmapped lax.cond evaluates both branches. A one-iteration while loop
    avoids running the planner and trajectory search on ordinary policy steps.
    """
    return jax.lax.while_loop(lambda carry: carry[0],
                              lambda _: (jnp.bool_(False), function()),
                              (active, default))[1]


def astar(free, targets, start, blocked_edges=None, *, return_path=False):
    """Multi-goal A*: nearest reachable target by four-connected path length.

    Returns flat indices (goal, next waypoint), or (-1, start) if unreachable.
    With return_path=True, also returns a padded start-to-goal route and length.
    Unknown cells must be excluded from `free` by the caller.
    """
    height, width = free.shape
    if blocked_edges is None:
        blocked_edges = jnp.zeros((*free.shape, 4), bool)
    size = height * width
    ids = jnp.arange(size)
    rows, cols = ids // width, ids % width
    targets = targets.reshape(-1) & free.reshape(-1)
    distance = jnp.abs(rows[:, None] - rows) + jnp.abs(cols[:, None] - cols)
    heuristic = jnp.min(jnp.where(targets[None], distance, size + 1), axis=1)
    costs = jnp.full(size, jnp.inf).at[start].set(0.)
    opened = jnp.zeros(size, bool).at[start].set(jnp.any(targets))
    parents = jnp.full(size, -1, jnp.int32)

    def condition(carry):
        _, opened, _, goal, count = carry
        return jnp.any(opened) & (goal < 0) & (count < size)

    def expand(carry):
        costs, opened, parents, goal, count = carry
        current = jnp.argmin(jnp.where(opened, costs + heuristic, jnp.inf))
        opened = opened.at[current].set(False)
        goal = jnp.where(targets[current], current, goal)
        r = current // width + jnp.array([-1, 1, 0, 0])
        c = current % width + jnp.array([0, 0, -1, 1])
        valid = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        nb = jnp.clip(r, 0, height - 1) * width + jnp.clip(c, 0, width - 1)

        def relax(k, values):
            costs, opened, parents = values
            index = nb[k]
            improve = (valid[k] & ~blocked_edges.reshape(-1, 4)[current, k]
                       & free.reshape(-1)[index] & (costs[current] + 1 < costs[index]))
            costs = costs.at[index].set(jnp.where(improve, costs[current] + 1, costs[index]))
            opened = opened.at[index].set(opened[index] | improve)
            parents = parents.at[index].set(jnp.where(improve, current, parents[index]))
            return costs, opened, parents

        costs, opened, parents = jax.lax.fori_loop(0, 4, relax, (costs, opened, parents))
        return costs, opened, parents, goal, count + 1

    _, _, parents, goal, _ = jax.lax.while_loop(
        condition, expand, (costs, opened, parents, jnp.int32(-1), jnp.int32(0)))

    if return_path:
        def trace(carry):
            node, route, length = carry
            return parents[node], route.at[length].set(node), length + 1

        _, reverse, length = jax.lax.while_loop(
            lambda carry: (carry[0] >= 0) & (carry[2] < size), trace,
            (goal, jnp.full(size, -1, jnp.int32), jnp.int32(0)))
        route = jnp.where(ids < length, reverse[jnp.maximum(length - 1 - ids, 0)], -1)
        waypoint = jnp.where(length > 1, route[1], start)
        return goal, waypoint, route, length

    def backtrack(carry):
        node, count = carry
        return (node != start) & (parents[node] >= 0) & (parents[node] != start) & (count < size)

    waypoint, _ = jax.lax.while_loop(
        backtrack, lambda x: (parents[x[0]], x[1] + 1),
        (jnp.where(goal >= 0, goal, start), jnp.int32(0)))
    return goal, waypoint


def lidar_points(env, position, heading, lidar):
    angles = heading + env._ray_angles
    return position + (lidar * env.max_lidar_range)[:, None] * jnp.stack(
        [jnp.cos(angles), jnp.sin(angles)], axis=-1)


def local_route_free(env, free, position, heading, lidar):
    """Temporary lidar obstacles for sequence replanning; never written to memory."""
    rows, cols = jnp.indices(free.shape)
    centres = (jnp.stack([cols, rows], axis=-1) + .5) * env.cell_size
    hits = lidar_points(env, position, heading, lidar)
    distance = jnp.linalg.norm(centres[..., None, :] - hits, axis=-1)
    blocked = jnp.any((lidar < 1.) & (distance < env.robot_radius + .02), axis=-1)
    col = jnp.clip(jnp.floor(position[0] / env.cell_size).astype(jnp.int32), 0, env.grid_w - 1)
    row = jnp.clip(jnp.floor(position[1] / env.cell_size).astype(jnp.int32), 0, env.grid_h - 1)
    return (free & ~blocked).at[row, col].set(free[row, col])


def velocity_sequence(env, position, heading, velocity, route, length):
    """Synthesize a bounded prefix of turn/brake/drive commands along an A* route.

    Unlike DWA's constant-velocity candidates, this rollout changes controls
    over time and commits to cell centres before turning around corners.
    Commands are physical (v, omega); a live safety check guards execution.
    """
    def step(carry, _):
        pos, hdg, vel, cursor = carry
        cell = jnp.maximum(route[jnp.minimum(cursor, route.size - 1)], 0)
        point = (jnp.array([cell % env.grid_w, cell // env.grid_w]) + .5) * env.cell_size
        cursor = cursor + ((cursor < length) & (jnp.linalg.norm(point - pos) < .02))
        cell = jnp.maximum(route[jnp.minimum(cursor, route.size - 1)], 0)
        point = (jnp.array([cell % env.grid_w, cell // env.grid_w]) + .5) * env.cell_size
        delta = point - pos
        distance = jnp.linalg.norm(delta)
        error = jnp.arctan2(delta[1], delta[0]) - hdg
        error = jnp.arctan2(jnp.sin(error), jnp.cos(error))
        a_dt = env.fallback_linear_accel * env.dt
        w_dt = env.fallback_angular_accel * env.dt
        # Include one control interval in the braking distance.
        speed = jnp.minimum(env.fallback_sequence_speed,
                            jnp.sqrt(a_dt ** 2 + 2 * env.fallback_linear_accel * distance) - a_dt)
        v = jnp.minimum(speed, vel[0] + a_dt)
        turn = (jnp.abs(error) > .04) | (jnp.abs(vel[1]) > w_dt)
        v = jnp.where(turn, 0., v)
        turn_speed = jnp.sqrt(w_dt ** 2 + 2 * env.fallback_angular_accel * jnp.abs(error)) - w_dt
        omega = jnp.sign(error) * jnp.minimum(turn_speed, env.omega_max)
        omega = jnp.clip(omega, vel[1] - w_dt, vel[1] + w_dt)
        command = jnp.where(cursor < length, jnp.array([v, omega]), jnp.zeros(2))
        next_pos, next_hdg = env._diff_drive(pos[None], hdg[None], command[:1], command[1:])
        return (next_pos[0], next_hdg[0], command, cursor), (command, cursor < length)

    start = jnp.maximum(route[0], 0)
    next_cell = jnp.maximum(route[1], 0)
    centre = (jnp.array([start % env.grid_w, start // env.grid_w]) + .5) * env.cell_size
    direction = jnp.array([next_cell % env.grid_w - start % env.grid_w,
                           next_cell // env.grid_w - start // env.grid_w], jnp.float32)
    offset = position - centre
    lateral = jnp.abs(offset[0] * direction[1] - offset[1] * direction[0])
    # A replenished queue must not turn back to a centre already passed on
    # this segment. Retain that centre when a corner still needs alignment.
    skip_start = (length > 1) & (jnp.dot(offset, direction) >= -.02) & (lateral < .02)
    _, (commands, valid) = jax.lax.scan(
        step, (position, heading, velocity, skip_start.astype(jnp.int32)), None,
        length=env.fallback_sequence_steps)
    return commands, jnp.sum(valid.astype(jnp.int32))


def command_safety_flags(env, position, heading, command, lidar, free, blocked_edges=None):
    """Rejection bitmask: 1 boundary, 2 unknown/occupied, 4 wall edge, 8 lidar.

    Check the next command and its braking trajectory against the latest scan.
    blocked_edges (H, W, 4) is optional: the belief map stores walls as
    occupied cells, so only planners on the true geometry pass edges.

    Directional braking checks let a robot rotate or move away from a nearby
    obstacle even when it starts inside DWA's preferred clearance margin.
    """
    if blocked_edges is None:
        blocked_edges = jnp.zeros((*free.shape, 4), bool)
    hits = lidar_points(env, position, heading, lidar)
    initial_clearance = jnp.min(jnp.where(lidar < 1., lidar * env.max_lidar_range, env.max_lidar_range))
    # A nearby wall is sampled at discrete points. Parallel motion changes
    # distance to those samples even when true wall clearance is constant.
    # Inside the preferred buffer, enforce physical clearance instead of
    # requiring the sampled minimum to never decrease.
    # Inside robot_radius already, accept commands that do not get closer;
    # otherwise rotation in place is rejected too and the robot freezes.
    clearance_limit = jnp.minimum(jnp.where(initial_clearance >= env.robot_radius + .01,
                                            env.robot_radius + .01, env.robot_radius),
                                  initial_clearance)

    def step(carry, index):
        pos, hdg, speed, flags = carry
        speed = jnp.where(index == 0, speed, jnp.maximum(0., speed - env.fallback_linear_accel * env.dt))
        new_pos, new_hdg = env._diff_drive(pos[None], hdg[None], speed[None], command[1:])
        new_pos, new_hdg = new_pos[0], new_hdg[0]
        old_c, old_r = env._pos_to_cell(pos[None])
        c = jnp.floor(new_pos[0] / env.cell_size).astype(jnp.int32)
        r = jnp.floor(new_pos[1] / env.cell_size).astype(jnp.int32)
        inside = (c >= 0) & (c < env.grid_w) & (r >= 0) & (r < env.grid_h)
        known_free = free[jnp.clip(r, 0, env.grid_h - 1), jnp.clip(c, 0, env.grid_w - 1)]
        edges = blocked_edges[old_r[0], old_c[0]]
        wall = ((edges[0] & (r < old_r[0])) | (edges[1] & (r > old_r[0]))
                | (edges[2] & (c < old_c[0])) | (edges[3] & (c > old_c[0])))
        nearest = jnp.min(jnp.where(lidar < 1., jnp.linalg.norm(new_pos - hits, axis=-1), env.max_lidar_range))
        clear = nearest >= clearance_limit - 1e-5
        rejected = ((~inside).astype(jnp.int32)
                    | ((~known_free).astype(jnp.int32) * 2)
                    | (wall.astype(jnp.int32) * 4)
                    | ((~clear).astype(jnp.int32) * 8))
        return (new_pos, new_hdg, speed, flags | rejected), None

    # One command interval followed by enough intervals to come to rest.
    steps = math.ceil(env.v_max / (env.fallback_linear_accel * env.dt)) + 1
    (_, _, _, flags), _ = jax.lax.scan(
        step, (position, heading, command[0], jnp.int32(0)), jnp.arange(steps))
    return flags


def command_is_safe(env, position, heading, command, lidar, free, blocked_edges=None):
    return command_safety_flags(env, position, heading, command, lidar, free, blocked_edges) == 0


def dwa(env, position, heading, velocity, lidar, free, waypoint, blocked_edges=None):
    """Acceleration-limited trajectory search, with lidar braking clearance.

    A zero-speed emergency stop is always available. Unknown map cells and
    lidar hits (including people and robots) block candidate trajectories.
    """
    if blocked_edges is None:
        blocked_edges = jnp.zeros((*free.shape, 4), bool)
    dv = env.fallback_linear_accel * env.dt
    dw = env.fallback_angular_accel * env.dt
    vs = jnp.linspace(jnp.maximum(0., velocity[0] - dv),
                      jnp.minimum(env.v_max, velocity[0] + dv), 5)
    ws = jnp.linspace(jnp.maximum(-env.omega_max, velocity[1] - dw),
                      jnp.minimum(env.omega_max, velocity[1] + dw), 9)
    v, w = jnp.meshgrid(vs, ws, indexing='ij')
    v, w = v.reshape(-1), w.reshape(-1)
    # Include braking/rotation even when the current forward speed is high.
    v = jnp.concatenate([v, jnp.zeros(9)])
    w = jnp.concatenate([w, ws])
    hits = lidar_points(env, position, heading, lidar)
    initial_clearance = jnp.min(jnp.where(
        lidar < 1., lidar * env.max_lidar_range, env.max_lidar_range))
    initial = (jnp.broadcast_to(position, (v.size, 2)),
               jnp.full(v.shape, heading), jnp.ones(v.shape, bool),
               jnp.full(v.shape, env.max_lidar_range))

    def simulate(carry, _):
        pos, hdg, valid, clearance = carry
        old_col = jnp.clip(jnp.floor(pos[:, 0] / env.cell_size).astype(jnp.int32), 0, env.grid_w - 1)
        old_row = jnp.clip(jnp.floor(pos[:, 1] / env.cell_size).astype(jnp.int32), 0, env.grid_h - 1)
        pos, hdg = env._diff_drive(pos, hdg, v, w)
        col = jnp.floor(pos[:, 0] / env.cell_size).astype(jnp.int32)
        row = jnp.floor(pos[:, 1] / env.cell_size).astype(jnp.int32)
        inside = (col >= 0) & (col < env.grid_w) & (row >= 0) & (row < env.grid_h)
        known_free = free[jnp.clip(row, 0, env.grid_h - 1), jnp.clip(col, 0, env.grid_w - 1)]
        edges = blocked_edges[old_row, old_col]
        crossed_wall = ((edges[:, 0] & (row < old_row)) | (edges[:, 1] & (row > old_row))
                        | (edges[:, 2] & (col < old_col)) | (edges[:, 3] & (col > old_col)))
        hit_dist = jnp.linalg.norm(pos[:, None] - hits[None], axis=-1)
        nearest = jnp.min(jnp.where(lidar[None] < 1., hit_dist, env.max_lidar_range), axis=1)
        clearance = jnp.minimum(clearance, nearest)
        margin = jnp.where(v > 1e-6, .03 + v ** 2 / (2 * env.fallback_linear_accel), 0.)
        preferred = env.robot_radius + margin
        # Inside the preferred buffer, do not spend the last measured sliver
        # of clearance. Sparse lidar hits can overestimate distance to the
        # wall between rays; allowing approach down to exactly robot_radius
        # caused grazing collisions. Rotation and motion away remain valid.
        # Already inside robot_radius (a person stopped next to the robot):
        # rotating or moving away must stay valid, or both wait forever.
        clearance_limit = jnp.minimum(preferred, initial_clearance)
        safe = nearest >= clearance_limit - 1e-5
        return (pos, hdg, valid & inside & known_free & ~crossed_wall & safe, clearance), None

    (end, hdg, valid, clearance), _ = jax.lax.scan(
        simulate, initial, None, length=env.fallback_dwa_steps)
    delta = waypoint - end
    angle = jnp.arctan2(delta[:, 1], delta[:, 0]) - hdg
    heading_error = jnp.abs(jnp.arctan2(jnp.sin(angle), jnp.cos(angle)))
    score = (-4. * jnp.linalg.norm(delta, axis=-1) - 0.5 * heading_error
             + 0.1 * v + 0.05 * jnp.minimum(clearance, 1.))
    best = jnp.argmax(jnp.where(valid, score, -jnp.inf))
    safe_v = jnp.where(jnp.any(valid), v[best], 0.)
    safe_w = jnp.where(jnp.any(valid), w[best], 0.)
    return jnp.array([2. * safe_v / env.v_max - 1., safe_w / env.omega_max])
