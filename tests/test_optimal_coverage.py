"""Exact CCPP planner: validity, optimality against brute force, statuses."""

import unittest
from collections import deque

import numpy as np

from src.envs.optimal_coverage import (FEASIBLE, INFEASIBLE, OPTIMAL, TIME_LIMIT, CostWeights,
                                       Grid, build_graph, hamiltonian_path, moves_lower_bound,
                                       plan_coverage, validate_path)

OPEN = ["....", "....", "....", "...."]
OBSTACLES = ["....", ".##.", "....", "...."]
CORRIDOR = ["........"]
DEAD_ENDS = [".......", "...#...", "...#...", "......."]


def brute_force_moves(rows, start):
    """Fewest moves visiting every free cell: BFS over (cell, visited set)."""
    grid = Grid.from_strings(rows)
    cells = grid.free_cells()
    index = {cell: k for k, cell in enumerate(cells)}
    full = (1 << len(cells)) - 1
    first = (index[start], 1 << index[start])
    seen = {first}
    queue = deque([(first, 0)])
    while queue:
        (k, mask), depth = queue.popleft()
        if mask == full:
            return depth
        r, c = cells[k]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = index.get((r + dr, c + dc))
            if j is not None:
                state = (j, mask | 1 << j)
                if state not in seen:
                    seen.add(state)
                    queue.append((state, depth + 1))
    return None


class OptimalCoverageTest(unittest.TestCase):

    def check(self, rows, start, weights=CostWeights()):
        """Plan with the default settings, and with the MILP alone in both
        formulations; require OPTIMAL and the same cost from all three, and
        recount everything with the validator."""
        result = plan_coverage(rows, start, weights)
        self.assertEqual(result.status, OPTIMAL)
        for formulation in ('undirected', 'directed'):
            alone = plan_coverage(rows, start, weights, formulation=formulation,
                                  search_budget=0)
            self.assertEqual(alone.status, OPTIMAL, formulation)
            self.assertEqual(alone.moves, result.moves, formulation)
            self.assertTrue(validate_path(alone.path, rows, start, weights).valid, formulation)
        report = validate_path(result.path, rows, start, weights)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(result.path[0], start)
        self.assertEqual((report.moves, report.revisits), (result.moves, result.revisits))
        self.assertAlmostEqual(report.cost, result.cost)
        self.assertAlmostEqual(result.lower_bound, result.cost)
        self.assertEqual(result.gap, 0.0)
        return result

    def test_open_grid_is_a_hamiltonian_path(self):
        result = self.check(OPEN, (0, 0))
        self.assertEqual((result.moves, result.revisits), (15, 0))

    def test_obstacles_match_brute_force(self):
        for start in [(0, 0), (1, 0), (2, 2), (3, 3)]:
            self.assertEqual(self.check(OBSTACLES, start).moves,
                             brute_force_moves(OBSTACLES, start))

    def test_corridor_goes_to_the_nearer_end_first(self):
        for col in range(8):
            result = self.check(CORRIDOR, (0, col))
            self.assertEqual(result.moves, 7 + min(col, 7 - col))
            self.assertEqual(result.revisits, result.moves - 7)

    def test_dead_ends(self):
        result = self.check(DEAD_ENDS, (1, 2))
        self.assertEqual((result.moves, result.revisits), (25, 0))

    def test_every_start_is_optimal(self):
        for rows in (["...", "...", "..."], ["....", "....", "...."], OBSTACLES):
            for start in Grid.from_strings(rows).free_cells():
                with self.subTest(rows=rows, start=start):
                    self.assertEqual(self.check(rows, start).moves,
                                     brute_force_moves(rows, start))

    def test_parity_forces_a_revisit(self):
        # 3 x 3: five cells of one colour, four of the other. From the minority
        # colour no Hamiltonian path exists, so one revisit is optimal.
        result = self.check(["...", "...", "..."], (0, 1))
        self.assertEqual((result.moves, result.revisits), (9, 1))

    def test_disconnected_map_is_infeasible(self):
        result = plan_coverage(["..#..", "..#.."], (0, 0))
        self.assertEqual(result.status, INFEASIBLE)
        self.assertIsNone(result.path)
        self.assertIn('unreachable', result.message)

    def test_start_on_obstacle_raises(self):
        with self.assertRaises(ValueError):
            plan_coverage(OBSTACLES, (1, 1))

    def test_weights_scale_the_cost(self):
        weights = CostWeights(alpha=2.0, beta=5.0)
        result = self.check(CORRIDOR, (0, 2), weights)
        self.assertAlmostEqual(result.cost, 2.0 * result.moves + 5.0 * result.revisits)
        with self.assertRaises(ValueError):
            CostWeights(0.0, 0.0)

    def test_numpy_grid_and_single_cell(self):
        free = np.zeros((3, 3), bool)
        free[1, 1] = True
        result = plan_coverage(free, (1, 1))
        self.assertEqual(result.status, OPTIMAL)
        self.assertEqual((result.path, result.moves), ([(1, 1)], 0))

    def test_non_optimal_statuses_still_return_valid_paths(self):
        rows = ["." * 10] * 8
        limited = plan_coverage(rows, (0, 3), time_limit=1e-6, search_budget=0)
        self.assertEqual(limited.status, TIME_LIMIT)
        self.assertTrue(validate_path(limited.path, rows, (0, 3)).valid)
        self.assertLessEqual(limited.lower_bound, limited.cost)
        self.assertGreaterEqual(limited.gap, 0.0)
        loose = plan_coverage(OBSTACLES, (0, 0), mip_gap=0.5, search_budget=0)
        self.assertIn(loose.status, (FEASIBLE, TIME_LIMIT))
        self.assertTrue(validate_path(loose.path, OBSTACLES, (0, 0)).valid)

    def test_lower_bound_and_search(self):
        grid = Grid.from_strings(["...", "...", "..."])
        graph = build_graph(grid)
        self.assertEqual(moves_lower_bound(graph, graph.index[(0, 0)]), 8)   # majority colour
        self.assertEqual(moves_lower_bound(graph, graph.index[(0, 1)]), 9)   # minority colour
        self.assertIsNone(hamiltonian_path(graph, graph.index[(0, 1)]))
        order = hamiltonian_path(graph, graph.index[(1, 1)])
        self.assertEqual(sorted(order), list(range(9)))
        found = plan_coverage(OPEN, (0, 0))
        self.assertIn('Hamiltonian', found.message)
        self.assertEqual(found.iterations, 0)

    def test_transitions_count_the_moves(self):
        result = plan_coverage(CORRIDOR, (0, 3))
        x = result.transitions
        self.assertEqual(sum(x.values()), result.moves)
        self.assertTrue(all(abs(i[1] - j[1]) == 1 for i, j in x))

    def test_unseeded_lazy_loop(self):
        result = plan_coverage(OBSTACLES, (1, 0), search_budget=0, seed=False)
        self.assertEqual(result.status, OPTIMAL)
        self.assertEqual(result.moves, brute_force_moves(OBSTACLES, (1, 0)))

    def test_validator_rejects_bad_paths(self):
        rows = ["...", "..."]
        good = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
        self.assertTrue(validate_path(good, rows, (0, 0)).valid)
        self.assertFalse(validate_path(good, rows, (0, 1)).valid)                 # wrong start
        self.assertFalse(validate_path(good[:-1], rows, (0, 0)).valid)            # cell missing
        self.assertFalse(validate_path([(0, 0), (1, 1)] + good[1:], rows, (0, 0)).valid)  # diagonal
        self.assertFalse(validate_path(good + [(2, 0)], rows, (0, 0)).valid)      # outside
        report = validate_path(good + [(1, 1), (1, 0)], rows, (0, 0))
        self.assertTrue(report.valid)
        self.assertEqual((report.moves, report.revisits, report.revisited_cells), (7, 2, 2))


if __name__ == '__main__':
    unittest.main()
