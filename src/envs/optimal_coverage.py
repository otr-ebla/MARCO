"""Exact Complete Coverage Path Planning (CCPP) on a 2D grid, single robot.

The reference planner of the single-robot study: centralised, it knows the
whole map and returns the ordered sequence of cells that covers every free
cell at minimum cost. The robot is holonomic and its heading is not modelled;
how the sequence is executed is left to the caller.

Pipeline (one section of this file each):

    Grid representation      Grid: free / obstacle cells
            |
    Graph construction       CoverageGraph: G = (V, E), 4-connected
            |
    MILP model construction  DirectedMILP / UndirectedMILP
            |
    HiGHS optimisation       scipy.optimize.milp (HiGHS), relative gap 0 by default
            |
    Subtour detection / cuts components of the solution not connected to the start
            |
    Path extraction          Euler path of the solution's multigraph from the start
            |
    Ordered cell sequence    CoverageResult.path = [(r0, c0), (r1, c1), ...]

Cost (s = start cell, V = free cells, v_i = visits of cell i, start included)

    J(P) = alpha T(P) + beta R(P),   T = moves,   R = sum_i max(0, v_i - 1).

Directed MILP (formulation='directed')

    x_ij in {0, 1, 2, ...}   times the transition i -> j is used
    e_i  in {0, 1}           the path ends in i;  sum_i e_i = 1
    z_i  in {0, 1}           i is visited; fixed to 1 for every i in V
    r_i  in {0, 1, 2, ...}   extra visits of i

    flow       sum_j x_ji - sum_j x_ij = e_i - [i = s]     (end free; it may be s)
    coverage   sum_j x_ji + [i = s] >= z_i = 1
    revisits   r_i >= v_i - 1,  v_i = sum_j x_ji + [i = s]
    cut        sum_{i notin S, j in S} x_ij >= 1           every S with s notin S
    objective  min alpha sum_ij x_ij + beta sum_i r_i

Undirected MILP (formulation='undirected', the default)

The same problem on y_ij = x_ij + x_ji, one variable per edge {i, j}:

    y_ij in {0, 1, 2}        times the edge is used, in either direction (an
                             optimal walk never needs a third pass: dropping
                             two copies keeps degrees even and the walk connected)
    e_i, z_i, r_i            as above;  p_i integer = v_i, the visits of i
    degree     sum_j y_ij + [i = s] + e_i = 2 p_i          (Euler: the walk
               s -> ... -> end enters and leaves every visit once)
    coverage   p_i >= z_i = 1
    revisits   r_i >= p_i - 1
    cut        y(delta(S)) + e(S) >= 2                     every S with s notin S
               (the walk enters and leaves S, or enters once and ends in it)
    objective  min alpha sum_ij y_ij + beta sum_i r_i

Every directed solution projects onto an undirected one with the same cost,
and every undirected solution is oriented into a walk by an Euler path, which
gives back x_ij. The directed form keeps both orientations of every cycle as
distinct solutions, and HiGHS did not finish a 175-cell map of the
single-robot study in 25 minutes with it; the undirected form solves the same
maps in seconds to minutes. Both are exact and the tests check that they
agree.

Subtour elimination. Without the cuts, solutions can contain closed cycles
detached from the walk. The cuts are exponentially many and added lazily:
solve, take the components of the used arcs/edges that do not contain s, add
one cut per component, solve again. scipy's HiGHS interface has no callback,
so "lazy" means re-solving with the cuts added. The cuts of every adjacent
pair and 2 x 2 block of cells (without s) are added from the start, since
those small cycles otherwise make up most of the rounds. Every intermediate
model is a relaxation of the full problem: the first connected optimum is a
global optimum and each solve's dual bound is a valid lower bound.

Two exact shortcuts come first:

* Lower bound. Every cell is visited, so R = T + 1 - |V| for every coverage
  path and J = (alpha + beta) T - beta (|V| - 1); a bound on T bounds J.
  T >= |V| - 1, and the grid is bipartite (colour (r + c) % 2): the walk
  alternates colours, so with A cells of the start's colour and B of the
  other, T >= 2A - 2 and T >= 2B - 1. The bound is a row of the MILP.
* When the bound is |V| - 1, a depth-first search for a Hamiltonian path is
  tried; a path found meets the bound and is optimal without the MILP.

As a consequence of R = T + 1 - |V|, any alpha, beta >= 0 (not both 0) rank
coverage paths the same way: the shortest path is also the one with the
fewest revisits, and minimising J minimises both at once. The weights will
matter once moves have different durations (heading, diagonals).

Status:
    OPTIMAL     proven optimal: MILP solved with relative gap 0 and connected,
                or a Hamiltonian path meeting the lower bound.
    FEASIBLE    a valid path from a MILP solved with a positive mip_gap: not
                proven optimal; `gap` bounds the loss.
    TIME_LIMIT  the time limit stopped the proof. `path` is still a valid
                coverage path (the interrupted solution with its detached
                components joined, or a nearest-unvisited walk) and
                `lower_bound` / `gap` certify its distance from the optimum.
    INFEASIBLE  some free cell is unreachable from the start; no path.

References: Dantzig, Fulkerson and Johnson 1954 (subtour elimination); Itai,
Papadimitriou and Szwarcfiter 1982 (Hamiltonian paths in grid graphs are
NP-complete, so no polynomial sweep such as boustrophedon/BCD or Spanning Tree
Coverage is optimal in general); Arkin et al. 2005 and Fekete and Krupke 2019
(integer programs for grid covering tours); Huangfu and Hall 2018 (HiGHS).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, shortest_path

Cell = tuple[int, int]

OPTIMAL, FEASIBLE, TIME_LIMIT, INFEASIBLE = 'OPTIMAL', 'FEASIBLE', 'TIME_LIMIT', 'INFEASIBLE'

# 4-connectivity, (dr, dc): S, N, W, E. Matches bcd_expert.open_edges.
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ---------------------------------------------------------------------------
# Grid representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """(H, W) occupancy: True = free. Cells are (row, col)."""

    free: np.ndarray

    @classmethod
    def from_strings(cls, rows: Sequence[str], obstacle: str = '#') -> 'Grid':
        """'.' (or anything but `obstacle`) is free, `obstacle` is occupied."""
        if not rows or len({len(r) for r in rows}) != 1:
            raise ValueError('rows must be non-empty and of equal length')
        return cls(np.array([[ch != obstacle for ch in row] for row in rows], bool))

    @classmethod
    def coerce(cls, grid: 'Grid | Sequence[str] | np.ndarray') -> 'Grid':
        if isinstance(grid, Grid):
            return grid
        if isinstance(grid, (list, tuple)) and grid and isinstance(grid[0], str):
            return cls.from_strings(grid)
        return cls(np.asarray(grid))

    def __post_init__(self):
        free = np.asarray(self.free, bool)
        if free.ndim != 2:
            raise ValueError('grid must be 2D')
        object.__setattr__(self, 'free', free)

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def is_free(self, cell: Cell) -> bool:
        return self.inside(cell) and bool(self.free[cell])

    def free_cells(self) -> list[Cell]:
        return [(int(r), int(c)) for r, c in zip(*np.nonzero(self.free))]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageGraph:
    """G = (V, E) over the free cells. cells[k] is vertex k; edge k joins
    edge_u[k] and edge_v[k]; arcs 2k and 2k + 1 are its two directions."""

    cells: list[Cell]
    index: dict[Cell, int]
    edge_u: np.ndarray
    edge_v: np.ndarray

    @property
    def n(self) -> int:
        return len(self.cells)

    @property
    def tail(self) -> np.ndarray:
        return np.stack([self.edge_u, self.edge_v], axis=1).ravel()

    @property
    def head(self) -> np.ndarray:
        return np.stack([self.edge_v, self.edge_u], axis=1).ravel()

    def adjacency(self) -> csr_matrix:
        return csr_matrix((np.ones(2 * self.edge_u.size), (self.tail, self.head)),
                          shape=(self.n, self.n))

    def reachable(self, start: int) -> np.ndarray:
        """(n,) bool: vertices connected to `start`."""
        order = breadth_first_order(self.adjacency(), start, directed=True,
                                    return_predecessors=False)
        mask = np.zeros(self.n, bool)
        mask[order] = True
        return mask


def build_graph(grid: Grid, open_moves: np.ndarray | None = None) -> CoverageGraph:
    """Edges between 4-adjacent free cells.

    open_moves: optional (H, W, 4) bool, False where a move is forbidden even
    though both cells are free (a thin wall between them), in MOVES order.
    """
    cells = grid.free_cells()
    index = {cell: k for k, cell in enumerate(cells)}
    us, vs = [], []
    for (r, c), k in index.items():
        for m in (1, 3):                                  # N and E: each edge once
            dr, dc = MOVES[m]
            other = (r + dr, c + dc)
            back = m - 1                                  # S and W from `other`
            if (grid.is_free(other) and (open_moves is None
                                         or (open_moves[r, c, m] and open_moves[other][back]))):
                us.append(k)
                vs.append(index[other])
    return CoverageGraph(cells, index, np.array(us, np.int64), np.array(vs, np.int64))


# ---------------------------------------------------------------------------
# MILP model construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostWeights:
    """J(P) = alpha * T(P) + beta * R(P)."""

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self):
        if self.alpha < 0 or self.beta < 0 or self.alpha + self.beta == 0:
            raise ValueError('alpha and beta must be non-negative and not both zero')

    def cost(self, moves: int, revisits: int) -> float:
        return self.alpha * moves + self.beta * revisits


class _CoverageModel:
    """Shared layout: [moves (arcs or edges) | e | per-model | z | r] and rows."""

    directed: bool
    cut_rhs: float

    def __init__(self, graph: CoverageGraph, start: int, weights: CostWeights,
                 n_moves: int, n_extra: int, min_moves: int):
        n = graph.n
        self.graph, self.start, self.weights = graph, start, weights
        self.sm, self.se = 0, n_moves
        self.sk = n_moves + n                              # model-specific block
        self.sz = self.sk + n_extra
        self.sr = self.sz + n
        self.n_var = self.sr + n
        self.c = np.zeros(self.n_var)
        self.c[:n_moves] = weights.alpha
        self.c[self.sr:] = weights.beta
        self.lower = np.zeros(self.n_var)
        self.upper = np.full(self.n_var, np.inf)
        self.upper[self.se:self.sk] = 1.0                  # e binary
        self.lower[self.sz:self.sr] = 1.0                  # z_i = 1: full coverage
        self.upper[self.sz:self.sr] = 1.0
        end = np.zeros((1, self.n_var))
        end[0, self.se:self.sk] = 1.0
        self.rows = [LinearConstraint(end, 1, 1)]
        if min_moves > 0:
            moves = np.zeros((1, self.n_var))
            moves[0, :n_moves] = 1.0
            self.rows.append(LinearConstraint(moves, min_moves, np.inf))
        self.cuts: list[np.ndarray] = []
        self.first = (np.arange(n) == start).astype(float)  # [i = s]

    def column(self, offset: int, sign: float = 1.0) -> coo_matrix:
        n = self.graph.n
        return coo_matrix((np.full(n, sign), (np.arange(n), offset + np.arange(n))),
                          shape=(n, self.n_var))

    def add_cut(self, members) -> None:
        inside = np.zeros(self.graph.n, bool)
        inside[np.asarray(members, np.int64)] = True
        if inside[self.start]:
            raise ValueError('a subtour cut set must not contain the start')
        self.cuts.append(self.cut_row(inside))

    def constraints(self) -> list[LinearConstraint]:
        extra = ([LinearConstraint(csr_matrix(np.array(self.cuts)), self.cut_rhs, np.inf)]
                 if self.cuts else [])
        return self.rows + extra

    def moves(self, solution: np.ndarray) -> np.ndarray:
        """Multiplicity of each arc (directed) or edge (undirected)."""
        return np.rint(solution[self.sm:self.se]).astype(np.int64)

    def end(self, solution: np.ndarray) -> int:
        return int(np.argmax(solution[self.se:self.sk]))


class DirectedMILP(_CoverageModel):
    """x_ij on arcs (graph.tail -> graph.head); flow conservation."""

    directed, cut_rhs = True, 1.0

    def __init__(self, graph: CoverageGraph, start: int, weights: CostWeights,
                 min_moves: int = 0):
        tail, head = graph.tail, graph.head
        a = tail.size
        super().__init__(graph, start, weights, a, 0, min_moves)
        n_var, arcs = self.n_var, np.arange(a)
        inflow = coo_matrix((np.ones(a), (head, arcs)), shape=(graph.n, n_var))
        outflow = coo_matrix((np.ones(a), (tail, arcs)), shape=(graph.n, n_var))
        first = self.first
        self.rows += [
            # in - out - e = -[i = s]
            LinearConstraint((inflow - outflow - self.column(self.se)).tocsr(), -first, -first),
            # in - z >= -[i = s]
            LinearConstraint((inflow - self.column(self.sz)).tocsr(), -first, np.inf),
            # r - in >= [i = s] - 1
            LinearConstraint((self.column(self.sr) - inflow).tocsr(), first - 1, np.inf),
        ]
        self._tail, self._head = tail, head

    def cut_row(self, inside: np.ndarray) -> np.ndarray:
        """sum_{i notin S, j in S} x_ij."""
        row = np.zeros(self.n_var)
        row[:self.se] = ~inside[self._tail] & inside[self._head]
        return row


class UndirectedMILP(_CoverageModel):
    """y_ij on edges; degree parity through p_i = visits of i."""

    directed, cut_rhs = False, 2.0

    def __init__(self, graph: CoverageGraph, start: int, weights: CostWeights,
                 min_moves: int = 0):
        n, m = graph.n, graph.edge_u.size
        super().__init__(graph, start, weights, m, n, min_moves)
        self.upper[:m] = 2.0
        self.lower[self.sk:self.sz] = 1.0                  # p_i >= 1
        edges = np.arange(m)
        degree = coo_matrix((np.ones(2 * m), (np.concatenate([graph.edge_u, graph.edge_v]),
                                              np.concatenate([edges, edges]))),
                            shape=(n, self.n_var))
        first = self.first
        self.rows += [
            # sum y + e - 2 p = -[i = s]
            LinearConstraint((degree + self.column(self.se) - self.column(self.sk, 2.0)).tocsr(),
                             -first, -first),
            # p - z >= 0
            LinearConstraint((self.column(self.sk) - self.column(self.sz)).tocsr(), 0, np.inf),
            # r - p >= -1
            LinearConstraint((self.column(self.sr) - self.column(self.sk)).tocsr(), -1, np.inf),
        ]

    def cut_row(self, inside: np.ndarray) -> np.ndarray:
        """y(delta(S)) + e(S)."""
        row = np.zeros(self.n_var)
        row[:self.se] = inside[self.graph.edge_u] != inside[self.graph.edge_v]
        row[self.se:self.sk] = inside
        return row


FORMULATIONS = {'directed': DirectedMILP, 'undirected': UndirectedMILP}


# ---------------------------------------------------------------------------
# HiGHS optimisation
# ---------------------------------------------------------------------------

def solve_highs(model: _CoverageModel, time_limit: float | None = None, mip_gap: float = 0.0):
    """One HiGHS branch-and-bound solve of the current model and cuts."""
    options = dict(disp=False, mip_rel_gap=mip_gap)
    if time_limit is not None:
        options['time_limit'] = max(time_limit, 1e-3)
    return milp(model.c, constraints=model.constraints(), integrality=np.ones(model.n_var),
                bounds=Bounds(model.lower, model.upper), options=options)


# ---------------------------------------------------------------------------
# Subtour detection / cuts
# ---------------------------------------------------------------------------

def _ends(graph: CoverageGraph, directed: bool) -> tuple[np.ndarray, np.ndarray]:
    return (graph.tail, graph.head) if directed else (graph.edge_u, graph.edge_v)


def detached_components(graph: CoverageGraph, moves: np.ndarray, start: int,
                        directed: bool) -> list[np.ndarray]:
    """Vertex sets of the solution's components that do not contain the start."""
    a, b = _ends(graph, directed)
    used = moves > 0
    support = csr_matrix((np.ones(used.sum()), (a[used], b[used])), shape=(graph.n, graph.n))
    _, label = connected_components(support, directed=False)
    return [np.flatnonzero(label == k) for k in np.unique(label) if k != label[start]]


def seed_cuts(graph: CoverageGraph, start: int) -> list[list[int]]:
    """Adjacent pairs and 2 x 2 blocks without the start: the cycles that
    otherwise fill the first rounds of the lazy loop."""
    sets = [[int(u), int(v)] for u, v in zip(graph.edge_u, graph.edge_v) if start not in (u, v)]
    for (r, c), k in graph.index.items():
        block = [graph.index.get(cell) for cell in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1))]
        if None not in block and start not in block:
            sets.append(block)
    return sets


def moves_lower_bound(graph: CoverageGraph, start: int) -> int:
    """Fewest moves of any coverage walk from `start`.

    T >= |V| - 1, and the 4-connected grid is bipartite (colour (r + c) % 2):
    the walk p_0..p_T alternates colours, so the start's colour fills the
    floor(T / 2) + 1 even positions and the other colour the ceil(T / 2) odd
    ones. With A cells of the start's colour and B of the other, every cell
    present needs T >= 2A - 2 and T >= 2B - 1.
    """
    colour = np.array([(r + c) % 2 for r, c in graph.cells])
    same = int(np.sum(colour == colour[start]))
    return max(graph.n - 1, 2 * same - 2, 2 * (graph.n - same) - 1)


def hamiltonian_path(graph: CoverageGraph, start: int, budget: int = 200_000,
                     restarts: int = 20, seed: int = 0) -> list[int] | None:
    """Path visiting every vertex once from `start`, or None.

    Randomised restarts of a depth-first search sharing `budget` extensions.
    Order: fewest unvisited neighbours first (Warnsdorff), then straight on,
    then random. A partial path is pruned when an unvisited cell can no longer
    be entered, when two cells could only be entered as the final one, when
    the unvisited cells are not all connected to the path's end, or when
    their colours cannot alternate from it.
    """
    rng = np.random.default_rng(seed)
    share = max(budget // max(restarts, 1), 1)
    for attempt in range(max(restarts, 1)):
        noise = None if attempt == 0 else rng.random(graph.n)
        path = _hamiltonian_search(graph, start, share, noise)
        if path is not None:
            return path
    return None


def _hamiltonian_search(graph: CoverageGraph, start: int, budget: int,
                        noise: np.ndarray | None) -> list[int] | None:
    n = graph.n
    neighbours = [[] for _ in range(n)]
    for t, h in zip(graph.tail.tolist(), graph.head.tolist()):
        neighbours[t].append(h)
    adjacent = [set(nb) for nb in neighbours]
    cells = graph.cells
    open_degree = [len(nb) for nb in neighbours]       # unvisited neighbours
    visited = [False] * n
    forced: set[int] = set()                           # unvisited, at most one way in
    colour = [(r + c) % 2 for r, c in cells]
    unvisited_colour = [colour.count(0), colour.count(1)]

    def reach(u, end):
        return open_degree[u] + (end in adjacent[u])

    def refresh(region, end):
        for u in region:
            if not visited[u] and reach(u, end) <= 1:
                forced.add(u)
            else:
                forced.discard(u)

    def connected(end, remaining):
        seen = {end}
        frontier = [end]
        while frontier:
            v = frontier.pop()
            for u in neighbours[v]:
                if not visited[u] and u not in seen:
                    seen.add(u)
                    frontier.append(u)
        return len(seen) - 1 == remaining

    def balanced(end):
        # The next cell has the other colour: those must equal or exceed the
        # end's colour by one among the unvisited cells.
        surplus = unvisited_colour[1 - colour[end]] - unvisited_colour[colour[end]]
        return 0 <= surplus <= 1

    def candidates(v, before):
        options = [u for u in neighbours[v] if not visited[u]]
        straight = None
        if before is not None:
            straight = (2 * cells[v][0] - cells[before][0], 2 * cells[v][1] - cells[before][1])
        return iter(sorted(options, key=lambda u: (open_degree[u], cells[u] != straight,
                                                   0.0 if noise is None else noise[u])))

    def visit(u):
        visited[u] = True
        unvisited_colour[colour[u]] -= 1
        for w in neighbours[u]:
            open_degree[w] -= 1

    def leave(u):
        visited[u] = False
        unvisited_colour[colour[u]] += 1
        for w in neighbours[u]:
            open_degree[w] += 1

    visit(start)
    path = [start]
    refresh(range(n), start)
    stack = [candidates(start, None)]
    extensions = 0
    while stack:
        if len(path) == n:
            return path
        nxt = next(stack[-1], None)
        if nxt is None:
            stack.pop()
            if len(path) == 1:
                break
            u = path.pop()
            leave(u)
            refresh(set(neighbours[u]) | set(neighbours[path[-1]]) | {u}, path[-1])
            continue
        if visited[nxt]:
            continue
        extensions += 1
        if extensions > budget:
            return None
        before = path[-1]
        visit(nxt)
        path.append(nxt)
        region = set(neighbours[before]) | set(neighbours[nxt]) | {nxt}
        refresh(region, nxt)
        if (len(forced) > 1 or any(reach(u, nxt) == 0 for u in forced)
                or not balanced(nxt) or not connected(nxt, n - len(path))):
            path.pop()
            leave(nxt)
            refresh(region, before)
            continue
        stack.append(candidates(nxt, before))
    return None


def _shortest_paths(graph: CoverageGraph):
    dist, pred = shortest_path(graph.adjacency(), unweighted=True, return_predecessors=True)
    return dist, pred


def _route(pred, source: int, target: int) -> list[int]:
    """Vertices of a shortest path source -> target, both included."""
    out = [int(target)]
    while out[-1] != source:
        out.append(int(pred[source, out[-1]]))
    return out[::-1]


def join_components(graph: CoverageGraph, moves: np.ndarray, start: int,
                    directed: bool) -> np.ndarray:
    """Make an interrupted solution executable: attach each detached component
    by a shortest path there and back (flow balance and parity are kept)."""
    moves = moves.copy()
    a, b = _ends(graph, directed)
    index = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(a, b))}
    if not directed:
        index.update({(v, u): k for (u, v), k in list(index.items())})
    dist, pred = _shortest_paths(graph)
    while True:
        components = detached_components(graph, moves, start, directed)
        if not components:
            return moves
        attached = np.ones(graph.n, bool)
        for members in components:
            attached[members] = False
        members = components[0]
        i, j = np.unravel_index(np.argmin(np.where(attached[:, None], dist[:, members], np.inf)),
                                (graph.n, members.size))
        route = _route(pred, int(i), int(members[j]))
        for u, v in zip(route, route[1:]):
            moves[index[u, v]] += 1
            moves[index[v, u]] += 1


def greedy_walk(graph: CoverageGraph, start: int) -> list[int]:
    """Nearest-unvisited walk from the start (fallback without a solution)."""
    dist, pred = _shortest_paths(graph)
    visited = np.zeros(graph.n, bool)
    walk = [start]
    visited[start] = True
    while not visited.all():
        target = int(np.argmin(np.where(visited, np.inf, dist[walk[-1]])))
        route = _route(pred, walk[-1], target)
        visited[route] = True
        walk.extend(route[1:])
    return walk


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

def euler_path(graph: CoverageGraph, moves: np.ndarray, start: int, directed: bool) -> list[int]:
    """Hierholzer on the solution's multigraph (arcs or edges), from the start.

    The walk ends at the cell with the extra unit of inflow (directed) or
    odd degree (undirected). Among the unused moves of a vertex the one
    continuing the current direction is taken first, so lanes stay straight
    where the optimum allows it.
    """
    a, b = _ends(graph, directed)
    incident = defaultdict(list)
    for k in np.flatnonzero(moves):
        incident[int(a[k])].extend([int(k)] * int(moves[k]))
        if not directed:
            incident[int(b[k])].extend([int(k)] * int(moves[k]))
    remaining = Counter({int(k): int(moves[k]) for k in np.flatnonzero(moves)})
    cells = graph.cells

    def other(k, v):
        return int(b[k]) if int(a[k]) == v else int(a[k])

    stack, came, path = [start], [None], []
    while stack:
        v = stack[-1]
        options = incident[v]
        while options and remaining[options[-1]] == 0:
            options.pop()
        if not options:
            path.append(stack.pop())
            came.pop()
            continue
        pick = len(options) - 1
        if came[-1] is not None:
            ahead = (2 * cells[v][0] - cells[came[-1]][0], 2 * cells[v][1] - cells[came[-1]][1])
            for i in range(len(options) - 1, -1, -1):
                if remaining[options[i]] and cells[other(options[i], v)] == ahead:
                    pick = i
                    break
        k = options.pop(pick)
        if remaining[k] == 0:
            continue
        remaining[k] -= 1
        stack.append(other(k, v))
        came.append(v)
    path.reverse()
    if len(path) != int(moves.sum()) + 1 or path[0] != start:
        raise RuntimeError('the solution is not a single walk from the start')
    return path


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

@dataclass
class CoverageResult:
    status: str
    path: list[Cell] | None
    cost: float = float('nan')
    moves: int = 0
    revisits: int = 0
    lower_bound: float = float('nan')
    solve_seconds: float = 0.0
    iterations: int = 0          # MILP solves
    cuts: int = 0                # subtour cuts in the final model, seeds included
    message: str = ''
    weights: CostWeights = field(default_factory=CostWeights)

    @property
    def gap(self) -> float:
        """(cost - lower bound) / cost; 0 when OPTIMAL, nan without a path."""
        if self.path is None:
            return float('nan')
        if self.status == OPTIMAL:
            return 0.0
        return max(self.cost - self.lower_bound, 0.0) / max(abs(self.cost), 1e-12)

    @property
    def transitions(self) -> Counter:
        """x_ij: times the path moves from cell i to cell j."""
        return Counter(zip(self.path, self.path[1:])) if self.path else Counter()


def plan_coverage(grid: Grid | Sequence[str] | np.ndarray, start: Cell,
                  weights: CostWeights = CostWeights(), *, formulation: str = 'undirected',
                  time_limit: float | None = None, mip_gap: float = 0.0,
                  open_moves: np.ndarray | None = None, search_budget: int = 200_000,
                  seed: bool = True, verbose: bool = False) -> CoverageResult:
    """Minimum-J coverage path from `start` (row, col) over every free cell.

    formulation: 'undirected' (default, fast) or 'directed' (x_ij per arc).
    search_budget: extensions of the Hamiltonian-path search, tried first
    when the lower bound allows a path without revisits (0 skips it).
    seed: start the lazy loop with the pair and 2 x 2 block cuts.
    """
    began = time.time()
    if formulation not in FORMULATIONS:
        raise ValueError(f'formulation must be one of {sorted(FORMULATIONS)}')
    grid = Grid.coerce(grid)
    start = (int(start[0]), int(start[1]))
    if not grid.is_free(start):
        raise ValueError(f'start {start} is not a free cell')
    graph = build_graph(grid, open_moves)
    s = graph.index[start]
    reach = graph.reachable(s)
    if not reach.all():
        missing = [graph.cells[k] for k in np.flatnonzero(~reach)]
        return CoverageResult(INFEASIBLE, None, solve_seconds=time.time() - began,
                              weights=weights,
                              message=f'{len(missing)} free cells unreachable from {start}, '
                                      f'e.g. {missing[:5]}')

    def result(status, walk, bound, iterations, cuts, message=''):
        path = [graph.cells[k] for k in walk]
        moves = len(path) - 1
        revisits = sum(n - 1 for n in Counter(path).values())
        cost = weights.cost(moves, revisits)
        bound = cost if status == OPTIMAL else min(bound, cost)
        return CoverageResult(status, path, cost, moves, revisits, bound,
                              time.time() - began, iterations, cuts, message, weights)

    if graph.n == 1:
        return result(OPTIMAL, [s], 0.0, 0, 0)

    # Every free cell is visited, so R = T + 1 - |V| and a bound on T bounds J.
    min_moves = moves_lower_bound(graph, s)
    bound = weights.cost(min_moves, min_moves + 1 - graph.n)
    if min_moves == graph.n - 1 and search_budget > 0:
        order = hamiltonian_path(graph, s, search_budget)
        if order is not None:
            return result(OPTIMAL, order, bound, 0, 0,
                          'Hamiltonian path: meets the lower bound T >= |V| - 1')
        if verbose:
            print(f'  no Hamiltonian path within {search_budget} extensions')

    deadline = None if time_limit is None else began + time_limit
    model = FORMULATIONS[formulation](graph, s, weights, min_moves)
    if seed:
        for members in seed_cuts(graph, s):
            model.add_cut(members)

    iterations = 0
    while True:
        left = None if deadline is None else deadline - time.time()
        if left is not None and left <= 0:
            return result(TIME_LIMIT, greedy_walk(graph, s), bound, iterations,
                          len(model.cuts), 'time limit before a MILP solution; '
                                           'nearest-unvisited walk')
        iterations += 1
        sol = solve_highs(model, left, mip_gap)
        dual = getattr(sol, 'mip_dual_bound', None)
        if dual is not None and np.isfinite(dual):
            bound = max(bound, float(dual))
        if sol.x is None:
            if sol.status == 1:      # time limit without an incumbent
                return result(TIME_LIMIT, greedy_walk(graph, s), bound, iterations,
                              len(model.cuts), 'time limit before a MILP solution; '
                                               'nearest-unvisited walk')
            raise RuntimeError(f'HiGHS failed: {sol.message}')
        moves = model.moves(sol.x)
        components = detached_components(graph, moves, s, model.directed)
        if verbose:
            print(f'  MILP {iterations}: J = {sol.fun:.3f}, {len(components)} detached '
                  f'components, {len(model.cuts)} cuts, {time.time() - began:.1f} s')
        if sol.status == 1:          # time limit with an incumbent
            moves = join_components(graph, moves, s, model.directed)
            return result(TIME_LIMIT, euler_path(graph, moves, s, model.directed), bound,
                          iterations, len(model.cuts), 'time limit; incumbent made connected')
        if not components:
            walk = euler_path(graph, moves, s, model.directed)
            if mip_gap == 0.0:
                return result(OPTIMAL, walk, float(sol.fun), iterations, len(model.cuts))
            return result(FEASIBLE, walk, bound, iterations, len(model.cuts),
                          f'MILP solved with mip_gap={mip_gap}')
        for members in components:
            model.add_cut(members)


# ---------------------------------------------------------------------------
# Validation (independent of the MILP)
# ---------------------------------------------------------------------------

@dataclass
class PathReport:
    valid: bool
    errors: list[str]
    moves: int
    revisits: int                # sum_i max(0, v_i - 1)
    revisited_cells: int         # cells visited more than once
    cost: float


def validate_path(path: Sequence[Cell], grid: Grid | Sequence[str] | np.ndarray,
                  start: Cell, weights: CostWeights = CostWeights()) -> PathReport:
    """Check a coverage path against the grid alone and recount its cost."""
    grid = Grid.coerce(grid)
    errors = []
    path = [tuple(int(v) for v in cell) for cell in path]
    if not path:
        errors.append('empty path')
    elif path[0] != tuple(start):
        errors.append(f'path starts at {path[0]}, not at {tuple(start)}')
    for k, cell in enumerate(path):
        if not grid.is_free(cell):
            errors.append(f'step {k}: {cell} is an obstacle or outside the grid')
    for k, (a, b) in enumerate(zip(path, path[1:])):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            errors.append(f'step {k}: {a} -> {b} is not a 4-adjacent move')
    counts = Counter(path)
    missing = [cell for cell in grid.free_cells() if cell not in counts]
    if missing:
        errors.append(f'{len(missing)} free cells not covered, e.g. {missing[:5]}')
    moves = max(len(path) - 1, 0)
    revisits = sum(max(0, n - 1) for n in counts.values())
    return PathReport(not errors, errors, moves, revisits,
                      sum(n > 1 for n in counts.values()), weights.cost(moves, revisits))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

class OptimalCoveragePlanner:
    """plan_coverage on an environment's map bank, cached per (map, start cell).

    Cells are (row, col) of the environment grid: row = floor(y / cell_size),
    col = floor(x / cell_size). Moves through thin walls are excluded.
    """

    def __init__(self, env, weights: CostWeights = CostWeights(),
                 time_limit: float | None = None, **options):
        from .bcd_expert import open_edges
        self.env = env
        self.weights = weights
        self.time_limit = time_limit
        self.options = options
        walls = np.asarray(env.walls)
        self.grids = [Grid(env.free_mask_np[m] > .5) for m in range(env.num_maps)]
        self.open_moves = [open_edges(g.free, walls[m], env.cell_size)
                           for m, g in enumerate(self.grids)]
        self._cache: dict[tuple[int, Cell], CoverageResult] = {}

    def cell(self, position) -> Cell:
        col = int(np.clip(np.floor(position[0] / self.env.cell_size), 0, self.env.grid_w - 1))
        row = int(np.clip(np.floor(position[1] / self.env.cell_size), 0, self.env.grid_h - 1))
        return row, col

    def plan(self, map_id: int, position) -> CoverageResult:
        key = (int(map_id), self.cell(position))
        if key not in self._cache:
            self._cache[key] = plan_coverage(
                self.grids[key[0]], key[1], self.weights, time_limit=self.time_limit,
                open_moves=self.open_moves[key[0]], **self.options)
        return self._cache[key]

    def waypoints(self, path: Sequence[Cell]) -> np.ndarray:
        """(L + 1, 2) cell centres (x, y) in metres."""
        cells = np.asarray(path, np.float64)
        return (cells[:, ::-1] + .5) * self.env.cell_size


def main() -> None:
    """Optimal coverage paths on a config's map bank: one CSV row per (map, start)."""
    import argparse
    import csv
    import sys

    from src.envs.coverage_vector_env import MultiRobotCoverageEnv
    from src.utils.config_parser import load_config

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--config', default='config/single_robot.yaml')
    parser.add_argument('--maps', type=int, default=8, help='layouts in the bank')
    parser.add_argument('--map-seed', type=int, default=None)
    parser.add_argument('--starts', type=int, default=4, help='random start cells per map')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of the moves T')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of the revisits R')
    parser.add_argument('--formulation', choices=sorted(FORMULATIONS), default='undirected')
    parser.add_argument('--time-limit', type=float, default=None, help='seconds per plan')
    parser.add_argument('--output', default=None, help='CSV path (default: stdout)')
    args = parser.parse_args()

    env_cfg = {**load_config(args.config).get('env', {}), 'num_maps': args.maps,
               'num_humans': 0}
    if args.map_seed is not None:
        env_cfg['map_seed'] = args.map_seed
    env = MultiRobotCoverageEnv(env_cfg)
    planner = OptimalCoveragePlanner(env, CostWeights(args.alpha, args.beta), args.time_limit,
                                     formulation=args.formulation)
    rng = np.random.default_rng(args.seed)
    fields = ['map', 'start_row', 'start_col', 'cells', 'status', 'cost', 'moves', 'revisits',
              'time_s', 'lower_bound', 'gap', 'iterations', 'cuts', 'solve_seconds']
    handle = open(args.output, 'w', newline='') if args.output else sys.stdout
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for m, grid in enumerate(planner.grids):
        cells = grid.free_cells()
        for k in rng.choice(len(cells), size=min(args.starts, len(cells)), replace=False):
            row, col = cells[k]
            plan = planner.plan(m, ((col + .5) * env.cell_size, (row + .5) * env.cell_size))
            writer.writerow(dict(
                map=m, start_row=row, start_col=col, cells=len(cells), status=plan.status,
                cost=plan.cost, moves=plan.moves, revisits=plan.revisits,
                time_s=round(plan.moves * env.cell_size / env.v_max, 4),
                lower_bound=plan.lower_bound, gap=round(plan.gap, 6),
                iterations=plan.iterations, cuts=plan.cuts,
                solve_seconds=round(plan.solve_seconds, 2)))
            handle.flush()
    if args.output:
        handle.close()


if __name__ == '__main__':
    main()
