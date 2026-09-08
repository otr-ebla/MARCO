from typing import NamedTuple

import jax
import jax.numpy as jnp


class JaxGuideState(NamedTuple):
    target: jax.Array        # (E, N)
    prev_cell: jax.Array     # (E, N)
    fail_cov: jax.Array      # (E, N)
    idx: jax.Array           # (E, N)
    tours: jax.Array         # (E, N, MAX_TOUR_LEN)
    tour_lens: jax.Array     # (E, N)

def jax_bellman_ford_to_target(targets, cost, neighbors, n_cells):
    """
    Computes shortest path distances from all cells to the `targets`.
    targets: (E, N)
    cost: (E, C)
    neighbors: (C, 4)
    Returns dist: (E, N, C)
    """
    E, N = targets.shape
    C = n_cells
    
    # Pad cost to handle -1 neighbors
    padded_cost = jnp.pad(cost, ((0, 0), (0, 1)), constant_values=jnp.inf) # (E, C+1)
    
    def init_fn():
        dist = jnp.full((E, N, C + 1), jnp.inf)
        e_idx = jnp.arange(E)[:, None]
        n_idx = jnp.arange(N)[None, :]
        
        # Valid targets are >= 0
        valid = targets >= 0
        safe_targets = jnp.maximum(targets, 0)
        
        dist = jnp.where(valid[..., None], dist.at[e_idx, n_idx, safe_targets].set(0.0), dist)
        return dist
        
    def step_fn(dist):
        padded_neighbors = jnp.where(neighbors >= 0, neighbors, C)
        
        # To compute D(c) = min_{nb} (cost[nb] + D(nb))
        # For each cell c, its neighbors are padded_neighbors[c]
        # We need dist of neighbors: dist[:, :, padded_neighbors] -> (E, N, C, 4)
        # We need cost of neighbors: padded_cost[:, padded_neighbors] -> (E, C, 4)
        
        d_nb = dist[:, :, padded_neighbors] # (E, N, C, 4)
        c_nb = padded_cost[:, None, padded_neighbors] # (E, 1, C, 4)
        
        d_from_c_via_nb = c_nb + d_nb
        
        best_d = jnp.min(d_from_c_via_nb, axis=-1) # (E, N, C)
        
        current_d = dist[:, :, :-1]
        improved = best_d < current_d
        new_dist = jnp.where(improved, best_d, current_d)
        
        next_dist = dist.at[:, :, :-1].set(new_dist)
        changed = jnp.any(improved)
        
        return next_dist, changed

    def cond_fn(carry):
        _, changed = carry
        return changed
        
    def body_fn(carry):
        dist, _ = carry
        return step_fn(dist)
        
    initial_dist = init_fn()
    final_dist, _ = jax.lax.while_loop(cond_fn, body_fn, (initial_dist, jnp.bool_(True)))
    
    return final_dist[:, :, :-1]


def get_next_waypoint(cell, target_dists, cost, neighbors, n_cells):
    """
    Given the distance field `target_dists` (E, N, C) and current `cell` (E, N),
    find the best neighbor to step into.
    """
    E, N = cell.shape
    C = n_cells
    
    padded_cost = jnp.pad(cost, ((0, 0), (0, 1)), constant_values=jnp.inf)
    padded_dist = jnp.pad(target_dists, ((0, 0), (0, 0), (0, 1)), constant_values=jnp.inf)
    padded_neighbors = jnp.where(neighbors >= 0, neighbors, C)
    
    e_idx = jnp.arange(E)[:, None, None]
    n_idx = jnp.arange(N)[None, :, None]
    
    # Neighbors of current cell: (E, N, 4)
    c_neighbors = padded_neighbors[cell] 
    
    # Cost to enter those neighbors: (E, N, 4)
    # cost is (E, C+1). We need cost[e, c_neighbors[e, n, i]]
    c_nb_cost = padded_cost[jnp.arange(E)[:, None, None], c_neighbors]
    
    # Distance from those neighbors to target: (E, N, 4)
    c_nb_dist = padded_dist[e_idx, n_idx, c_neighbors]
    
    total_cost = c_nb_cost + c_nb_dist
    
    best_idx = jnp.argmin(total_cost, axis=-1) # (E, N)
    
    best_neighbor = jnp.take_along_axis(c_neighbors, best_idx[..., None], axis=-1).squeeze(-1)
    
    # If cell is already target, return target
    best_neighbor = jnp.where(target_dists[jnp.arange(E)[:, None], jnp.arange(N)[None, :], cell] == 0.0, cell, best_neighbor)
    
    # If unreachable (all inf), return -1
    best_neighbor = jnp.where(jnp.min(total_cost, axis=-1) == jnp.inf, -1, best_neighbor)
    
    return best_neighbor


def select_mopup_targets(cell, covered, free_cells, components, w):
    """Choose one reachable uncovered goal per robot after its tour ends.

    Pending cells are split by current Manhattan distance so robots that finish
    at the same time do not all converge on the same hole.  Connectivity is
    checked with the graph's static component labels; the weighted distance
    field below still decides the actual route and prices revisits.
    """
    E, N = cell.shape
    C = covered.shape[-1]

    ids = jnp.arange(C, dtype=jnp.int32)
    cell_row, cell_col = cell // w, cell % w
    rows, cols = ids // w, ids % w
    distance = (
        jnp.abs(cell_row[..., None] - rows)
        + jnp.abs(cell_col[..., None] - cols)
    ).astype(jnp.float32)

    component = components[cell]
    reachable = components[None, None, :] == component[..., None]
    pending = free_cells[None, None, :] & ~covered[:, None, :] & reachable

    robot_for_cell = jnp.argmin(jnp.where(pending, distance, jnp.inf), axis=1)
    assigned = robot_for_cell[:, None, :] == jnp.arange(N)[None, :, None]
    assigned_pending = pending & assigned

    # A robot may own no pending cell in the dynamic split.  Let it help with
    # any reachable remainder instead of idling; duplicate goals are preferable
    # to abandoning coverable ground.
    candidates = jnp.where(
        jnp.any(assigned_pending, axis=-1, keepdims=True),
        assigned_pending,
        pending,
    )
    score = jnp.where(candidates, distance, jnp.inf)
    target = jnp.argmin(score, axis=-1)
    return jnp.where(jnp.isfinite(jnp.min(score, axis=-1)), target, -1)


def advance_cursors(tours, tour_lens, idx, cell, covered, snap_window=8):
    """Advance every ``(environment, robot)`` cursor in parallel."""
    tour_capacity = tours.shape[-1]
    offsets = jnp.arange(snap_window, dtype=idx.dtype)
    candidates = idx[..., None] + offsets
    in_tour = (candidates < tour_lens[..., None]) & (candidates < tour_capacity)
    safe_candidates = jnp.clip(candidates, 0, tour_capacity - 1)
    candidate_cells = jnp.take_along_axis(tours, safe_candidates, axis=-1)
    matches = in_tour & (candidate_cells == cell[..., None])
    idx = jnp.max(jnp.where(matches, candidates + 1, idx[..., None]), axis=-1)

    def tour_cells(cursor):
        safe = jnp.clip(cursor, 0, tour_capacity - 1)
        return jnp.take_along_axis(tours, safe[..., None], axis=-1)[..., 0]

    def skip_while(cursor, predicate):
        def active(c):
            return (c < tour_lens) & predicate(tour_cells(c))

        return jax.lax.while_loop(
            lambda c: jnp.any(active(c)),
            lambda c: jnp.where(active(c), c + 1, c),
            cursor,
        )

    idx = skip_while(idx, lambda tour_cell: tour_cell == cell)
    env_ids = jnp.arange(covered.shape[0], dtype=jnp.int32)[:, None]
    idx = skip_while(
        idx,
        lambda tour_cell: covered[env_ids, jnp.maximum(tour_cell, 0)],
    )
    return skip_while(idx, lambda tour_cell: tour_cell == cell)


def jax_guide_step(state: JaxGuideState, positions, coverage_grid, done,
                   graph_neighbors, w, h, cell_size, guides=None,
                   free_cells=None, graph_components=None, revisit_penalty=0.5,
                   previous_coverage_grid=None, target_arrived=None):
    """Advance the fixed sweep and emit a reactive one-cell waypoint.

    The remaining sweep is retained after a deviation, while its shortest-path
    prefix is recomputed from the robot's actual cell.  Once that sweep is
    exhausted, a dynamic nearest-cell split keeps every robot mopping up
    reachable uncovered ground until coverage is complete.
    """
    del guides  # Backward-compatible keyword; no host objects enter the hot path.
    E, N = state.idx.shape
    C = w * h

    # Tours are immutable accelerator data.  A fresh episode only resets its
    # cursors; the distance field below reconnects the new spawn to the tour.
    idx = jnp.where(done[:, None], 0, state.idx)
    target = jnp.where(done[:, None], -1, state.target)
    prev_cell = jnp.where(done[:, None], -1, state.prev_cell)
    fail_cov = jnp.where(done[:, None], -1, state.fail_cov)
    state = state._replace(
        idx=idx,
        target=target,
        prev_cell=prev_cell,
        fail_cov=fail_cov,
    )
    
    # 2. Update logic
    col = jnp.clip((positions[..., 0] / cell_size).astype(jnp.int32), 0, w - 1)
    row = jnp.clip((positions[..., 1] / cell_size).astype(jnp.int32), 0, h - 1)
    cell = row * w + col
    
    covered = (coverage_grid > 0.5).reshape(E, C)
    previous_covered = (
        covered if previous_coverage_grid is None
        else (previous_coverage_grid > 0.5).reshape(E, C)
    )

    e_idx = jnp.arange(E)[:, None]
    valid_target = state.target >= 0
    safe_target = jnp.maximum(state.target, 0)
    target_covered = valid_target & covered[e_idx, safe_target]
    target_newly_covered = (
        target_covered & ~previous_covered[e_idx, safe_target]
    )

    # The arrival bonus represents coverage, not mere occupancy.  Likewise a
    # waypoint stays stable until its cell has actually been covered; otherwise
    # a robot that only enters/approaches a cell can receive a new objective.
    reached = (~done[:, None]) & (cell == state.target) & target_newly_covered
    # A guide used as an observation may advance as soon as its target cell is
    # covered: the learned actor is free to cut the corner.  A guide used as the
    # actual BOSCO controller must keep the target stable until the robot reaches
    # the cell centre, otherwise the target flips while the robot is still on the
    # preceding leg and successive 90-degree turns get cut into walls.
    if target_arrived is None:
        target_arrived = target_covered
    can_advance = done[:, None] | ~valid_target | (target_covered & target_arrived)
    
    # Derive candidates in parallel, then retain the previous per-robot guide
    # state wherever ``can_advance`` is false.
    new_idx = advance_cursors(state.tours, state.tour_lens, state.idx, cell, covered)
    
    is_finished = new_idx >= state.tour_lens
    
    n_idx = jnp.arange(N)[None, :]
    safe_idx = jnp.minimum(new_idx, state.tour_lens - 1)
    tour_target = state.tours[e_idx, n_idx, safe_idx]
    tour_target = jnp.where(is_finished, -1, tour_target)

    if graph_components is None:
        # Primarily useful for small standalone callers.  Production passes the
        # graph's real connected-component labels so goals across walls cannot
        # be selected merely because they look close in grid coordinates.
        graph_components = jnp.zeros((C,), dtype=jnp.int32)
    if free_cells is None:
        free_cells = jnp.ones((C,), dtype=jnp.bool_)
    free_cells = jnp.asarray(free_cells, dtype=jnp.bool_)
    graph_components = jnp.asarray(graph_components, dtype=jnp.int32)
    mopup_target = select_mopup_targets(
        cell, covered, free_cells, graph_components, w
    )
    goal = jnp.where(is_finished, mopup_target, tour_target)
    
    cost = jnp.where(covered, 1.0 + revisit_penalty, 1.0)
    dist = jax_bellman_ford_to_target(goal, cost, graph_neighbors, C)
    
    waypoint = get_next_waypoint(cell, dist, cost, graph_neighbors, C)
    waypoint = jnp.where(goal >= 0, waypoint, -1)

    n_covered = jnp.sum(covered, axis=-1, dtype=jnp.int32)
    fail_cov = jnp.where(waypoint < 0, n_covered[:, None], -1)
    
    waypoint = jnp.where(can_advance, waypoint, state.target)
    new_state = state._replace(
        target=waypoint,
        prev_cell=jnp.where(can_advance, cell, state.prev_cell),
        fail_cov=jnp.where(can_advance, fail_cov, state.fail_cov),
        idx=jnp.where(can_advance, new_idx, state.idx),
    )
    
    return new_state, waypoint, reached
