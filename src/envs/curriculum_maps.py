"""Curriculum map banks: wall rectangles (x0, y0, x1, y1) on one fixed floor.

Stages, all on the same floor of `wc` x `hc` cells:
  0  small empty room (6-8 cells per side); the rest of the floor is solid
  1  full floor, empty
  2  full floor, 3-5 random 1x1 / 2x2 obstacles
  3  BSP rooms joined by doorways (depth 1 or 2), plus 2-4 obstacles

Cells cut off from the largest free component are dropped by the env's free
mask, so every map is fully coverable. Each map also has a floor mask (the
small room in stage 0, the whole grid otherwise); the env marks cells outside
it as known obstacles from the start.
"""

from __future__ import annotations

import random

import numpy as np

from .map_layouts import ProceduralMapLayout

NUM_STAGES = 4
_PAD = [-10.0, -10.0, -10.0, -10.0]   # zero-area wall far outside the floor


def _outer(wc, hc, cs):
    w, h = wc * cs, hc * cs
    return [[-cs, -cs, w + cs, 0.0], [-cs, h, w + cs, h + cs],
            [-cs, -cs, 0.0, h + cs], [w, -cs, w + cs, h + cs]]


def _block(c, r, w, h, cs):
    return [c * cs, r * cs, (c + w) * cs, (r + h) * cs]


def _obstacles(rng, wc, hc, n, cs):
    """n square obstacles of 1 or 2 cells, at least one cell from the outer wall."""
    out = []
    for _ in range(n):
        s = int(rng.integers(1, 3))
        out.append(_block(int(rng.integers(1, wc - s)), int(rng.integers(1, hc - s)), s, s, cs))
    return out


def _stage_walls(stage, rng, wc, hc, cs, robot_radius):
    """Wall rectangles and the (hc, wc) bool floor mask of one map."""
    walls, floor = _outer(wc, hc, cs), np.ones((hc, wc), bool)
    if stage == 0:
        w, h = (int(v) for v in rng.integers(6, 9, size=2))
        c0, r0 = int(rng.integers(0, wc - w + 1)), int(rng.integers(0, hc - h + 1))
        solid = [(0, 0, c0, hc), (c0 + w, 0, wc - c0 - w, hc),
                 (c0, 0, w, r0), (c0, r0 + h, w, hc - r0 - h)]
        walls += [_block(*b, cs) for b in solid if b[2] > 0 and b[3] > 0]
        floor[:] = False
        floor[r0:r0 + h, c0:c0 + w] = True
    elif stage == 2:
        walls += _obstacles(rng, wc, hc, int(rng.integers(3, 6)), cs)
    elif stage == 3:
        layout = ProceduralMapLayout(width=wc * cs, height=hc * cs, cell_size=cs,
                                     robot_radius=robot_radius, wall_cells=1,
                                     max_depth=int(rng.integers(1, 3)),
                                     rng=random.Random(int(rng.integers(2 ** 31))))
        walls = [w for w in layout.walls if w[0] > -5.0]
        walls += _obstacles(rng, wc, hc, int(rng.integers(2, 5)), cs)
    return walls, floor


def curriculum_wall_bank(maps_per_stage, wc, hc, cell_size, robot_radius,
                         max_walls=16, seed=0):
    """Walls (S*K, max_walls, 4) float32 and floors (S*K, hc, wc) bool; stage s owns rows s*K:(s+1)*K."""
    rng = np.random.default_rng(seed)
    bank = np.empty((NUM_STAGES * maps_per_stage, max_walls, 4), np.float32)
    floors = np.empty((NUM_STAGES * maps_per_stage, hc, wc), bool)
    for stage in range(NUM_STAGES):
        for k in range(maps_per_stage):
            walls, floor = _stage_walls(stage, rng, wc, hc, cell_size, robot_radius)
            if len(walls) > max_walls:
                raise ValueError(f'stage {stage} map has {len(walls)} walls > max_walls={max_walls}')
            bank[stage * maps_per_stage + k] = walls + [_PAD] * (max_walls - len(walls))
            floors[stage * maps_per_stage + k] = floor
    return bank, floors


def with_stage_maps(env_cfg, stage, max_walls=16):
    """Copy of `env_cfg` on the env.num_maps maps of one curriculum stage.

    The bank matches train_ppo_single's with maps_per_stage = num_maps (same
    map_seed), so stage s here is stage s there.
    """
    if not 0 <= stage < NUM_STAGES:
        raise ValueError(f'stage must be in [0, {NUM_STAGES}), not {stage}')
    k, cs = int(env_cfg['num_maps']), env_cfg['cell_size']
    walls, floors = curriculum_wall_bank(
        k, round(env_cfg['map_width'] / cs), round(env_cfg['map_height'] / cs),
        cs, env_cfg['robot_radius'], max_walls, env_cfg.get('map_seed', 0))
    rows = slice(stage * k, (stage + 1) * k)
    return {**env_cfg, 'wall_bank': walls[rows], 'floor_bank': floors[rows]}
