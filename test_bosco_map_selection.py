from types import SimpleNamespace

import numpy as np

from src.algorithms.bosco_guide import BoscoGuide
from src.envs.map_layouts import create_map_bank


def _two_map_env():
    outer = [
        [-0.2, -0.2, 3.2, 0.0],
        [-0.2, 2.0, 3.2, 2.2],
        [-0.2, -0.2, 0.0, 2.2],
        [3.0, -0.2, 3.2, 2.2],
    ]
    padding = [[-10.0, -10.0, -10.0, -10.0]] * 2
    open_map = np.asarray(outer + padding, dtype=np.float32)
    split_map = np.asarray(
        outer + [[0.95, 0.0, 1.05, 2.0]] + padding[:1], dtype=np.float32
    )
    free = np.ones((2, 2, 3), dtype=np.float32)
    return SimpleNamespace(
        grid_h=2,
        grid_w=3,
        cell_size=1.0,
        robot_radius=0.2,
        walls=np.stack([open_map, split_map]),
        free_mask_np=free,
        room_masks=free[:, None],
        num_robots=1,
        dt=0.1,
        v_max=1.0,
        omega_max=1.0,
    )


def test_bosco_guide_rebuilds_partition_for_the_selected_map():
    guide = BoscoGuide(_two_map_env())

    guide.reset(np.asarray([[0.5, 0.5]], dtype=np.float32), map_id=1)

    assert guide.graph.map_id == 1
    # The full-height wall makes the cells to its right unreachable from the
    # spawn. They must remain unowned instead of inheriting map 0's partition.
    assert np.all(guide.owner.reshape(2, 3)[:, 1:] == -1)


def test_procedural_map_bank_is_reproducible_and_seeded_locally():
    first = create_map_bank(3, seed=17)
    same = create_map_bank(3, seed=17)
    different = create_map_bank(3, seed=18)

    for actual, expected in zip(first, same):
        np.testing.assert_array_equal(actual.get_walls(), expected.get_walls())
    assert any(
        not np.array_equal(a.get_walls(), b.get_walls())
        for a, b in zip(first, different)
    )
