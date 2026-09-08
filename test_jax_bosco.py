import jax.numpy as jnp

from src.algorithms.jax_bosco import JaxGuideState, jax_guide_step


def _line_neighbors(n):
    return jnp.asarray([
        [cell - 1 if cell > 0 else -1,
         cell + 1 if cell + 1 < n else -1,
         -1,
         -1]
        for cell in range(n)
    ], dtype=jnp.int32)


def test_finished_tour_replans_towards_uncovered_cell():
    tours = jnp.full((1, 1, 4), -1, dtype=jnp.int32).at[0, 0, 0].set(0)
    state = JaxGuideState(
        target=jnp.asarray([[1]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=tours,
        tour_lens=jnp.asarray([[1]], dtype=jnp.int32),
    )

    new_state, waypoint, reached = jax_guide_step(
        state=state,
        positions=jnp.asarray([[[0.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=jnp.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32),
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(4),
        w=4,
        h=1,
        cell_size=1.0,
        guides=[object()],
        free_cells=jnp.ones((4,), dtype=jnp.bool_),
        graph_components=jnp.zeros((4,), dtype=jnp.int32),
    )

    assert waypoint.tolist() == [[1]]
    assert new_state.target.tolist() == [[1]]
    assert reached.tolist() == [[False]]


def test_finished_tour_stops_when_every_free_cell_is_covered():
    tours = jnp.full((1, 1, 4), -1, dtype=jnp.int32).at[0, 0, 0].set(0)
    state = JaxGuideState(
        target=jnp.asarray([[1]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=tours,
        tour_lens=jnp.asarray([[1]], dtype=jnp.int32),
    )

    _, waypoint, _ = jax_guide_step(
        state=state,
        positions=jnp.asarray([[[0.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=jnp.ones((1, 1, 4), dtype=jnp.float32),
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(4),
        w=4,
        h=1,
        cell_size=1.0,
        guides=[object()],
        free_cells=jnp.ones((4,), dtype=jnp.bool_),
        graph_components=jnp.zeros((4,), dtype=jnp.int32),
    )

    assert waypoint.tolist() == [[-1]]


def test_target_and_reward_wait_until_target_cell_is_covered():
    state = JaxGuideState(
        target=jnp.asarray([[1]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=jnp.asarray([[[0, 1, 2]]], dtype=jnp.int32),
        tour_lens=jnp.asarray([[3]], dtype=jnp.int32),
    )
    previous = jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float32)

    waiting_state, waiting_waypoint, waiting_reward = jax_guide_step(
        state,
        positions=jnp.asarray([[[0.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=previous,
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(3),
        w=3,
        h=1,
        cell_size=1.0,
        free_cells=jnp.ones((3,), dtype=jnp.bool_),
        graph_components=jnp.zeros((3,), dtype=jnp.int32),
        previous_coverage_grid=previous,
    )

    assert waiting_waypoint.tolist() == [[1]]
    assert waiting_state.idx.tolist() == [[1]]
    assert waiting_reward.tolist() == [[False]]

    covered = previous.at[0, 0, 1].set(1.0)
    advanced_state, advanced_waypoint, coverage_reward = jax_guide_step(
        waiting_state,
        positions=jnp.asarray([[[1.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=covered,
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(3),
        w=3,
        h=1,
        cell_size=1.0,
        free_cells=jnp.ones((3,), dtype=jnp.bool_),
        graph_components=jnp.zeros((3,), dtype=jnp.int32),
        previous_coverage_grid=previous,
    )

    assert coverage_reward.tolist() == [[True]]
    assert advanced_waypoint.tolist() == [[2]]
    assert advanced_state.idx.tolist() == [[2]]


def test_entering_an_already_covered_target_does_not_pay_coverage_reward():
    state = JaxGuideState(
        target=jnp.asarray([[1]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=jnp.asarray([[[0, 1, 2]]], dtype=jnp.int32),
        tour_lens=jnp.asarray([[3]], dtype=jnp.int32),
    )
    covered = jnp.asarray([[[1.0, 1.0, 0.0]]], dtype=jnp.float32)

    _, _, coverage_reward = jax_guide_step(
        state,
        positions=jnp.asarray([[[1.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=covered,
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(3),
        w=3,
        h=1,
        cell_size=1.0,
        free_cells=jnp.ones((3,), dtype=jnp.bool_),
        graph_components=jnp.zeros((3,), dtype=jnp.int32),
        previous_coverage_grid=covered,
    )

    assert coverage_reward.tolist() == [[False]]


def test_controller_keeps_covered_target_until_geometric_arrival():
    state = JaxGuideState(
        target=jnp.asarray([[1]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=jnp.asarray([[[0, 1, 2]]], dtype=jnp.int32),
        tour_lens=jnp.asarray([[3]], dtype=jnp.int32),
    )
    covered = jnp.asarray([[[1.0, 1.0, 0.0]]], dtype=jnp.float32)

    waiting, waypoint, _ = jax_guide_step(
        state,
        positions=jnp.asarray([[[0.9, 0.5]]], dtype=jnp.float32),
        coverage_grid=covered,
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(3),
        w=3,
        h=1,
        cell_size=1.0,
        free_cells=jnp.ones((3,), dtype=jnp.bool_),
        graph_components=jnp.zeros((3,), dtype=jnp.int32),
        target_arrived=jnp.asarray([[False]]),
    )

    assert waypoint.tolist() == [[1]]
    assert waiting.idx.tolist() == [[1]]

    advanced, waypoint, _ = jax_guide_step(
        waiting,
        positions=jnp.asarray([[[1.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=covered,
        done=jnp.asarray([False]),
        graph_neighbors=_line_neighbors(3),
        w=3,
        h=1,
        cell_size=1.0,
        free_cells=jnp.ones((3,), dtype=jnp.bool_),
        graph_components=jnp.zeros((3,), dtype=jnp.int32),
        target_arrived=jnp.asarray([[True]]),
    )

    assert waypoint.tolist() == [[2]]
    assert advanced.idx.tolist() == [[2]]


def test_mopup_ignores_closer_uncovered_cell_in_an_unreachable_component():
    tours = jnp.full((1, 1, 5), -1, dtype=jnp.int32).at[0, 0, 0].set(0)
    state = JaxGuideState(
        target=jnp.asarray([[0]], dtype=jnp.int32),
        prev_cell=jnp.asarray([[0]], dtype=jnp.int32),
        fail_cov=jnp.asarray([[-1]], dtype=jnp.int32),
        idx=jnp.asarray([[1]], dtype=jnp.int32),
        tours=tours,
        tour_lens=jnp.asarray([[1]], dtype=jnp.int32),
    )
    # Cells 1 and 2 are geometrically close but disconnected.  Cell 4 is the
    # only pending target in the robot's component, reached through cell 3.
    neighbors = jnp.asarray([
        [3, -1, -1, -1],
        [2, -1, -1, -1],
        [1, -1, -1, -1],
        [0, 4, -1, -1],
        [3, -1, -1, -1],
    ], dtype=jnp.int32)

    _, waypoint, _ = jax_guide_step(
        state=state,
        positions=jnp.asarray([[[0.5, 0.5]]], dtype=jnp.float32),
        coverage_grid=jnp.asarray([[[1.0, 0.0, 0.0, 1.0, 0.0]]], dtype=jnp.float32),
        done=jnp.asarray([False]),
        graph_neighbors=neighbors,
        w=5,
        h=1,
        cell_size=1.0,
        guides=[object()],
        free_cells=jnp.ones((5,), dtype=jnp.bool_),
        graph_components=jnp.asarray([0, 1, 1, 0, 0], dtype=jnp.int32),
    )

    assert waypoint.tolist() == [[3]]
