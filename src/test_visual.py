#!/usr/bin/env python3
"""Visualise a coverage controller in real time using pygame (JAX).

Two controllers are available: the trained MAPPO policy, and BOSCO, the
deterministic boustrophedon expert used to generate imitation-learning data.
Under BOSCO each cell is painted in the colour of the robot that owns it after
the initial partition, so the division of the map is visible at a glance.

Usage (from the project root):
    python -m src.test_visual
    python -m src.test_visual --checkpoint checkpoints/checkpoint_final.pkl --episodes 10
    python -m src.test_visual --policy bosco

Playback controls: RIGHT or R skips the current episode, SPACE pauses, L toggles
LiDAR, O toggles ownership, S changes speed, and ESC quits.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import pygame

# Ensure the project root is on sys.path so that "from src.xxx import ..."
# works regardless of where the script is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp

from src.algorithms.bosco import BoscoExpert
from src.algorithms.bosco_guide import BoscoGuide
from src.algorithms.jax_bosco import JaxGuideState, jax_guide_step
from src.algorithms.mappo import RunningMeanStd, rms_normalize
from src.envs.coverage_vector_env import MultiRobotCoverageEnv
from src.models.actor_critic import Actor
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------
SCALE      = 80   # pixels per metre
MARGIN     = 40   # pixel border around the map
HUD_HEIGHT = 50   # pixels for the stats bar at the bottom
FPS        = 30
POPUP_DURATION_MS = 1300

# Speed levels (FPS) cycled by pressing 'S' during playback.
# Each step doubles the previous; the first entry is the default speed.
SPEED_LEVELS  = [30, 60, 120, 240, 0]   # 0 = uncapped (run as fast as possible)
_SPEED_LABELS = ['1×', '2×', '4×', '8×', '∞']

# Step budget BOSCO needs to finish the default map; see src/algorithms/bosco.py.
_BOSCO_MIN_STEPS = 3000

# One colour per robot (cycles if more than 4)
_ROBOT_COLORS = [
    (220,  60,  60),
    ( 60, 120, 220),
    ( 50, 180,  50),
    (220, 160,   0),
]

COLORS = {
    'bg':        (240, 240, 235),
    'covered':     (140, 220, 140),
    'uncovered':   (215, 215, 210),
    'unreachable': (170, 170, 165),
    'wall':      ( 55,  55,  55),
    'border':    ( 30,  30,  30),
    'grid':      (180, 180, 175),
    'lidar':     (255, 165,   0),
    'dead':      (120, 120, 120),
    'hud_bg':    ( 25,  25,  25),
    'hud_text':  (230, 230, 230),
    'success':   ( 35, 190,  80),
    'rr_hit':    (235,  45,  60),
    'rw_hit':    (155,  25,  35),
    'rh_hit':    (190,  90,  20),
    'timeout':   (210,  35, 180),
    'popup_bg':  ( 22,  22,  22),
}


def _to_px(x: float, y: float, map_h: float) -> tuple[int, int]:
    """World metres → pygame pixel (top-left origin, y-axis flipped)."""
    return (int(x * SCALE) + MARGIN,
            int((map_h - y) * SCALE) + MARGIN)


def _ownership_palette(num_robots: int) -> np.ndarray:
    """(R+1, 2, 3) uint8 lookup: [owner, covered] → cell colour.

    Row `num_robots` is the unowned fallback, so the table can be indexed with
    the raw owner id after mapping -1 onto it — no per-cell branching while
    drawing. Owned cells drop the shared green entirely: a robot's territory
    reads as a pale wash of its own colour while pending, and as a strong tint
    of it once swept, which keeps "who owns this" and "is it done" legible in
    the same square.
    """
    white     = np.array((255, 255, 255), dtype=float)
    uncovered = np.array(COLORS['uncovered'], dtype=float)
    lut = np.empty((num_robots + 1, 2, 3), dtype=float)
    lut[num_robots, 0] = uncovered
    lut[num_robots, 1] = COLORS['covered']
    for r in range(num_robots):
        c = np.array(_ROBOT_COLORS[r % len(_ROBOT_COLORS)], dtype=float)
        lut[r, 0] = 0.80 * uncovered + 0.20 * c
        lut[r, 1] = 0.30 * white     + 0.70 * c
    return lut.astype(np.uint8)


def _snapshot(env: MultiRobotCoverageEnv, state, want_lidar: bool,
              controller=None) -> dict:
    """Pull the arrays needed for one frame back to the host in one go.

    Device→host transfers are the only per-frame cost of rendering, so every
    field is fetched with a single `jax.device_get` on a packed tuple.
    """
    info = env.get_info(state)
    payload = (
        state.robot_positions,
        state.robot_headings,
        state.human_positions if env.num_humans > 0 else jnp.zeros((0, 2), jnp.float32),
        state.human_headings if env.num_humans > 0 else jnp.zeros((0,), jnp.float32),
        state.coverage_grid,
        state.bosco_targets,
        state.robot_alive,
        info['step'],
        info['coverage_ratio'],
        info['covered_cells'],
        info['total_cells'],
        info['timeout'],
        state.wall_hits,
        state.robot_hits,
        state.human_hits,
        env._cast_lidar_all(state)
        if want_lidar else jnp.zeros((0,), jnp.float32),
    )
    (pos, hdg, human_pos, human_hdg, grid, bosco_targets, alive, step, cov_ratio,
     covered, total, timeout, wall_hits, robot_hits, human_hits,
     lidar) = jax.device_get(payload)
    snap = {
        'positions':      np.asarray(pos),
        'headings':       np.asarray(hdg),
        'human_positions': np.asarray(human_pos),
        'human_headings': np.asarray(human_hdg),
        'coverage_grid':  np.asarray(grid),
        'bosco_targets':  np.asarray(bosco_targets),
        'alive':          np.asarray(alive),
        'step':           int(step),
        'coverage_ratio': float(cov_ratio),
        'covered_cells':  int(covered),
        'total_cells':    int(total),
        'timeout':        bool(timeout),
        'wall_hits':      np.asarray(wall_hits),
        'robot_hits':     np.asarray(robot_hits),
        'human_hits':     np.asarray(human_hits),
        'lidar':          np.asarray(lidar),
    }
    if controller is not None and hasattr(controller, 'bosco_goal_centers'):
        snap['bosco_goal_centers'] = controller.bosco_goal_centers(
            snap['coverage_grid'], snap['positions']
        )
    if controller is not None and hasattr(controller, 'bosco_target_centers'):
        # Plain BOSCO is a host-side planner and does not write its selected
        # waypoints into EnvState.  Pull them from the planner itself instead
        # of displaying EnvState.bosco_targets, which is merely initialised to
        # the spawn positions in this mode.
        snap['bosco_targets'] = controller.bosco_target_centers()
    return snap


def _star_points(center: tuple[int, int], outer: int, inner: int) -> list[tuple[int, int]]:
    """Return the ten vertices of a five-point star."""
    cx, cy = center
    angles = -np.pi / 2.0 + np.arange(10) * np.pi / 5.0
    radii = np.where(np.arange(10) % 2 == 0, outer, inner)
    return [
        (int(round(cx + radius * np.cos(angle))),
         int(round(cy + radius * np.sin(angle))))
        for angle, radius in zip(angles, radii)
    ]


def _episode_outcome(snap: dict) -> tuple[str, tuple[int, int, int]] | None:
    """Return the terminal banner, with collisions taking priority over success."""
    if np.any(np.asarray(snap['robot_hits']) > 0.0):
        return 'RR-COLLISION', COLORS['rr_hit']
    if np.any(np.asarray(snap['wall_hits']) > 0.0):
        return 'RW-COLLISION', COLORS['rw_hit']
    if np.any(np.asarray(snap['human_hits']) > 0.0):
        return 'RH-COLLISION', COLORS['rh_hit']
    if snap['covered_cells'] >= snap['total_cells']:
        return 'SUCCESS', COLORS['success']
    if snap['timeout']:
        return 'TIMEOUT', COLORS['timeout']
    return None


def _draw_popup(surface: pygame.Surface, popup: dict | None,
                font: pygame.font.Font) -> None:
    if popup is None:
        return
    label = font.render(popup['label'], True, popup['color'])
    box = label.get_rect()
    box.inflate_ip(48, 28)
    box.center = (surface.get_width() // 2,
                  (surface.get_height() - HUD_HEIGHT) // 2)
    shadow = box.move(5, 5)
    pygame.draw.rect(surface, (0, 0, 0), shadow, border_radius=12)
    pygame.draw.rect(surface, COLORS['popup_bg'], box, border_radius=12)
    pygame.draw.rect(surface, popup['color'], box, width=4, border_radius=12)
    surface.blit(label, label.get_rect(center=box.center))


def _draw_frame(
    surface: pygame.Surface,
    env: MultiRobotCoverageEnv,
    walls: np.ndarray,
    free: np.ndarray,
    snap: dict,
    font: pygame.font.Font,
    ep_reward: float,
    show_lidar: bool,
    owner: np.ndarray | None = None,
    palette: np.ndarray | None = None,
    label: str = '',
    speed_label: str = '1×',
    popup: dict | None = None,
    popup_font: pygame.font.Font | None = None,
) -> None:
    mw = env.grid_w * env.cell_size
    mh = env.grid_h * env.cell_size
    cs = env.cell_size

    surface.fill(COLORS['bg'])

    # -- Coverage cells --
    # Unreachable cells are drawn apart from pending ones: they are excluded
    # from the coverage denominator, so leaving them "uncovered" would suggest
    # work that can never be done.
    cell_px = int(cs * SCALE)
    grid = snap['coverage_grid']
    show_owner = owner is not None and palette is not None
    for row in range(env.grid_h):
        for col in range(env.grid_w):
            if free[row, col] == 0.0:
                color = COLORS['unreachable']
            elif show_owner:
                color = palette[owner[row, col], int(grid[row, col] > 0.0)]
            else:
                color = COLORS['covered'] if grid[row, col] > 0.0 else COLORS['uncovered']
            px, py = _to_px(col * cs, (row + 1) * cs, mh)
            pygame.draw.rect(surface, color, pygame.Rect(px, py, cell_px, cell_px))

    # -- Grid cell lines --
    map_px_w = int(mw * SCALE)
    map_px_h = int(mh * SCALE)
    for col in range(env.grid_w + 1):
        x = MARGIN + int(col * cell_px)
        pygame.draw.line(surface, COLORS['grid'], (x, MARGIN), (x, MARGIN + map_px_h))
    for row in range(env.grid_h + 1):
        y = MARGIN + int(row * cell_px)
        pygame.draw.line(surface, COLORS['grid'], (MARGIN, y), (MARGIN + map_px_w, y))

    # -- Walls --
    for x0, y0, x1, y1 in walls:
        px, py = _to_px(x0, y1, mh)
        w = int((x1 - x0) * SCALE)
        h = int((y1 - y0) * SCALE)
        pygame.draw.rect(surface, COLORS['wall'], pygame.Rect(px, py, w, h))

    # -- Map border --
    pygame.draw.rect(
        surface, COLORS['border'],
        pygame.Rect(MARGIN, MARGIN, map_px_w, map_px_h),
        2,
    )

    positions = snap['positions']
    headings  = snap['headings']

    # The star marks the immediate BOSCO waypoint. It can lie on an already
    # covered transit cell while the robot travels towards pending coverage.
    targets = snap['bosco_targets']
    valid_targets = (np.all(np.isfinite(targets), axis=1)
                     & (np.linalg.norm(targets - positions, axis=1) > 1e-5))
    if targets is not None:
        star_outer = max(6, int(cell_px * 0.28))
        star_inner = max(3, int(star_outer * 0.45))
        for i in np.flatnonzero(valid_targets):
            gx, gy = _to_px(targets[i, 0], targets[i, 1], mh)
            color = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
            pygame.draw.polygon(
                surface,
                COLORS['border'],
                _star_points((gx, gy), star_outer + 2, star_inner + 1),
            )
            pygame.draw.polygon(
                surface,
                color,
                _star_points((gx, gy), star_outer, star_inner),
            )

    # -- Next still-uncovered BOSCO cells --
    # Draw the dot after the star so both remain visible when the immediate
    # waypoint is also the next pending coverage objective.
    target_radius = max(3, int(cell_px * 0.12))
    goal_centers = snap.get('bosco_goal_centers')
    if goal_centers is not None:
        for i, goal in enumerate(goal_centers):
            if not np.all(np.isfinite(goal)):
                continue
            tx, ty = _to_px(goal[0], goal[1], mh)
            color = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
            pygame.draw.circle(surface, COLORS['border'], (tx, ty), target_radius + 2)
            pygame.draw.circle(surface, color, (tx, ty), target_radius)

    # -- Lidar rays (toggle with 'L') --
    if show_lidar and snap['lidar'].size:
        angles_rel = np.linspace(0.0, 2.0 * np.pi, env.n_rays, endpoint=False)
        for i in range(env.num_robots):
            ray_color = (_ROBOT_COLORS[i % len(_ROBOT_COLORS)] if snap['alive'][i]
                         else COLORS['dead'])
            pos    = positions[i]
            angles = headings[i] + angles_rel
            dists  = snap['lidar'][i] * env.max_lidar_range   # stored normalised
            cx, cy = _to_px(pos[0], pos[1], mh)
            ex = pos[0] + dists * np.cos(angles)
            ey = pos[1] + dists * np.sin(angles)
            for x, y in zip(ex, ey):
                tip_px, tip_py = _to_px(x, y, mh)
                pygame.draw.line(surface, ray_color, (cx, cy), (tip_px, tip_py), 1)
                pygame.draw.circle(surface, ray_color, (tip_px, tip_py), 3)

    # -- Robots --
    r_px = max(5, int(env.robot_radius * SCALE))
    for i in range(env.num_robots):
        cx, cy = _to_px(positions[i, 0], positions[i, 1], mh)
        hdg    = headings[i]
        color  = (_ROBOT_COLORS[i % len(_ROBOT_COLORS)] if snap['alive'][i]
                  else COLORS['dead'])
        pygame.draw.circle(surface, color, (cx, cy), r_px)
        # Against its own territory tint a bare disc loses its edge, so it keeps
        # a dark rim whenever the ownership overlay is on.
        if show_owner:
            pygame.draw.circle(surface, COLORS['border'], (cx, cy), r_px, 2)
        tip_x = cx + int(r_px * 1.8 * np.cos(hdg))
        tip_y = cy - int(r_px * 1.8 * np.sin(hdg))
        pygame.draw.line(surface, (255, 255, 255), (cx, cy), (tip_x, tip_y), 2)
        # robot index label
        lbl = font.render(str(i), True, (255, 255, 255))
        surface.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

    # -- Humans --
    human_positions = snap['human_positions']
    human_headings = snap['human_headings']
    
    if human_positions.shape[0] > 0:
        ry, rx = int(r_px * 1.2), max(2, int(r_px * 0.6))
        h_surf = pygame.Surface((2 * ry, 2 * ry), pygame.SRCALPHA)
        pygame.draw.ellipse(h_surf, (170, 170, 170), (ry - rx, 0, 2 * rx, 2 * ry))
        dot_r = max(2, int(rx / 2))
        pygame.draw.circle(h_surf, (0, 0, 0), (ry + rx - dot_r, ry), dot_r)

        for i in range(human_positions.shape[0]):
            cx, cy = _to_px(human_positions[i, 0], human_positions[i, 1], mh)
            hdg = human_headings[i]
            rot_surf = pygame.transform.rotate(h_surf, math.degrees(hdg))
            rect = rot_surf.get_rect(center=(cx, cy))
            surface.blit(rot_surf, rect)

    # -- HUD --
    win_h = surface.get_height()
    pygame.draw.rect(
        surface, COLORS['hud_bg'],
        pygame.Rect(0, win_h - HUD_HEIGHT, surface.get_width(), HUD_HEIGHT),
    )
    # Kept to ~107 monospace columns so the key hints survive at the default
    # window width (map width * SCALE + margins).
    text = (f"  {label} Step {snap['step']:4d}/{env.max_steps} | "
            f"Coverage {snap['coverage_ratio']:5.1%} "
            f"({snap['covered_cells']}/{snap['total_cells']}) | "
            f"Reward {ep_reward:8.2f} | "
            f"Speed {speed_label} | "
            f"[ESC] quit [RIGHT/R] next [SPACE] pause "
            f"[L] lidar [O] owners [S] speed")
    rendered = font.render(text, True, COLORS['hud_text'])
    line_h = rendered.get_height()
    top = win_h - HUD_HEIGHT + (HUD_HEIGHT - (2 * line_h if show_owner else line_h)) // 2
    surface.blit(rendered, (8, top))

    # Second line: how much of each robot's own region it has swept. Region
    # sizes differ — the wavefront balances area but connectivity bounds it —
    # so the per-robot fraction says more than the team total alone.
    if show_owner:
        done = grid > 0.0
        x = 8
        parts = font.render("  Regions:", True, COLORS['hud_text'])
        surface.blit(parts, (x, top + line_h))
        x += parts.get_width()
        for i in range(env.num_robots):
            mine = owner == i
            total = int(mine.sum())
            swept = int((mine & done).sum())
            frac = swept / total if total else 1.0
            chunk = font.render(f"  {i}: {swept:3d}/{total:3d} ({frac:4.0%})", True,
                                _ROBOT_COLORS[i % len(_ROBOT_COLORS)])
            surface.blit(chunk, (x, top + line_h))
            x += chunk.get_width()

    _draw_popup(surface, popup, popup_font or font)


class MappoController:
    """Trained policy. One jitted call per frame: normalise → act → step.

    Fusing the policy and the environment transition keeps a single device
    round-trip per rendered frame; only the render snapshot comes back.
    """

    label = 'MAPPO '
    owner = None

    def __init__(self, env: MultiRobotCoverageEnv, actor: Actor, params,
                 obs_rms: RunningMeanStd | None, guided: bool = False,
                 guide_bonus: float = 2.0):
        self.env, self.params, self.obs_rms = env, params, obs_rms
        self.guided = guided
        self.guide_bonus = float(guide_bonus)
        self.expert = (BoscoGuide(env) if guided else
                       BoscoExpert(env) if env.actor_bosco_guidance else None)
        self.guide_state = None

        self.graph_neighbors = self.graph_free = None
        self.graph_components = self.graph_centers = None
        if self.expert is not None:
            graph = self.expert.graph
            self.graph_neighbors = jnp.asarray(graph.neighbors, jnp.int32)
            self.graph_free = jnp.asarray(graph.free, jnp.bool_)
            self.graph_components = jnp.asarray(graph.component, jnp.int32)
            self.graph_centers = jnp.asarray(graph.centers, jnp.float32)

        @jax.jit
        def policy_step(params, rms, state, obs, guide_state,
                        graph_neighbors, graph_free, graph_components,
                        graph_centers):
            obs_n = rms_normalize(rms, obs) if rms is not None else obs
            mean, _ = actor.apply(params, obs_n)
            action = jnp.tanh(mean)                   # deterministic
            next_state, rewards, terminated, truncated = env.step(state, action)

            if guided:
                done = (terminated | truncated)[None]
                guide_state, waypoint, reached = jax_guide_step(
                    guide_state,
                    next_state.robot_positions[None],
                    next_state.coverage_grid[None],
                    done,
                    graph_neighbors,
                    env.grid_w,
                    env.grid_h,
                    env.cell_size,
                    free_cells=graph_free,
                    graph_components=graph_components,
                    previous_coverage_grid=state.coverage_grid[None],
                )
                valid = waypoint[0] >= 0
                safe = jnp.maximum(waypoint[0], 0)
                coords = graph_centers[safe]
                targets = jnp.where(
                    valid[:, None], coords, next_state.robot_positions
                )
                next_state = env.set_bosco_targets(next_state, targets)
                rewards = rewards + self.guide_bonus * reached[0].astype(jnp.float32)

            return (next_state, env.get_obs(next_state), guide_state,
                    rewards, terminated, truncated)

        self._fn = policy_step

    def reset(self, state):
        if self.expert is None:
            self.owner = np.full((self.env.grid_h, self.env.grid_w),
                                 self.env.num_robots, dtype=np.int32)
            self.obs = self.env.get_obs(state)
            return state
        current_map_id = int(jax.device_get(state.map_id))
        positions = np.asarray(jax.device_get(state.robot_positions))
        if self.guided:
            self.expert.reset(positions, map_id=current_map_id)
            graph = self.expert.graph
            self.graph_neighbors = jnp.asarray(graph.neighbors, jnp.int32)
            self.graph_free = jnp.asarray(graph.free, jnp.bool_)
            self.graph_components = jnp.asarray(graph.component, jnp.int32)
            self.graph_centers = jnp.asarray(graph.centers, jnp.float32)
        else:
            self.expert.reset(positions, map_id=current_map_id)
        owner = self.expert.owner.copy()
        owner[owner < 0] = self.env.num_robots
        self.owner = owner.reshape(self.env.grid_h, self.env.grid_w)

        if self.guided:
            coverage = np.asarray(jax.device_get(state.coverage_grid))
            targets, _ = self.expert.update(positions, coverage)
            valid = targets >= 0
            safe = np.maximum(targets, 0)
            coords = self.expert.graph.centers[safe]
            target_coords = np.where(valid[:, None], coords, positions)
            state = self.env.set_bosco_targets(
                state, jnp.asarray(target_coords, jnp.float32)
            )

            max_tour_len = 2048
            tours = np.full(
                (1, self.env.num_robots, max_tour_len), -1, dtype=np.int32
            )
            tour_lens = np.zeros((1, self.env.num_robots), dtype=np.int32)
            for robot, tour in enumerate(self.expert.tours):
                length = min(len(tour), max_tour_len)
                tours[0, robot, :length] = tour[:length]
                tour_lens[0, robot] = length

            cell_size = self.env.cell_size
            col = np.clip(
                (positions[:, 0] / cell_size).astype(np.int32),
                0, self.env.grid_w - 1,
            )
            row = np.clip(
                (positions[:, 1] / cell_size).astype(np.int32),
                0, self.env.grid_h - 1,
            )
            cell = row * self.env.grid_w + col
            self.guide_state = JaxGuideState(
                target=jnp.asarray(targets[None], jnp.int32),
                prev_cell=jnp.asarray(cell[None], jnp.int32),
                fail_cov=jnp.full((1, self.env.num_robots), -1, jnp.int32),
                idx=jnp.zeros((1, self.env.num_robots), jnp.int32),
                tours=jnp.asarray(tours, jnp.int32),
                tour_lens=jnp.asarray(tour_lens, jnp.int32),
            )

        self.obs = self.env.get_obs(state)
        return state

    def step(self, state, snap: dict):
        state, self.obs, self.guide_state, rewards, terminated, truncated = self._fn(
            self.params, self.obs_rms, state, self.obs, self.guide_state,
            self.graph_neighbors, self.graph_free, self.graph_components,
            self.graph_centers,
        )
        return state, rewards, terminated, truncated

    def bosco_goal_centers(self, coverage_grid: np.ndarray,
                           positions: np.ndarray) -> np.ndarray:
        """Centre of the first uncovered tour cell remaining for each robot."""
        goals = np.full((self.env.num_robots, 2), np.nan, dtype=np.float32)
        if not self.guided or self.guide_state is None:
            return goals

        tours, idx, lens = jax.device_get((
            self.guide_state.tours[0],
            self.guide_state.idx[0],
            self.guide_state.tour_lens[0],
        ))
        covered = np.asarray(coverage_grid).reshape(-1) > 0.5
        centers = np.asarray(jax.device_get(self.graph_centers))
        needs_mopup = np.zeros(self.env.num_robots, dtype=bool)
        for robot in range(self.env.num_robots):
            remaining = np.asarray(tours[robot, idx[robot]:lens[robot]], dtype=np.int64)
            pending = remaining[(remaining >= 0) & ~covered[np.maximum(remaining, 0)]]
            if pending.size:
                goals[robot] = centers[pending[0]]
            else:
                needs_mopup[robot] = True

        # Once a fixed tour is exhausted, JAX dynamically splits all reachable
        # uncovered cells among the robots.  Reproduce that split for rendering
        # so the star continues to show the final mop-up objective even when the
        # immediate dot is a covered transit waypoint.
        if np.any(needs_mopup):
            graph_free = np.asarray(jax.device_get(self.graph_free), dtype=bool)
            components = np.asarray(jax.device_get(self.graph_components))
            cols = np.clip(
                (positions[:, 0] / self.env.cell_size).astype(np.int32),
                0, self.env.grid_w - 1,
            )
            rows = np.clip(
                (positions[:, 1] / self.env.cell_size).astype(np.int32),
                0, self.env.grid_h - 1,
            )
            cells = rows * self.env.grid_w + cols
            ids = np.arange(covered.size)
            cell_rows, cell_cols = cells // self.env.grid_w, cells % self.env.grid_w
            grid_rows, grid_cols = ids // self.env.grid_w, ids % self.env.grid_w
            distance = (
                np.abs(cell_rows[:, None] - grid_rows[None, :])
                + np.abs(cell_cols[:, None] - grid_cols[None, :])
            )
            reachable = components[None, :] == components[cells, None]
            pending = graph_free[None, :] & ~covered[None, :] & reachable
            owner = np.argmin(np.where(pending, distance, np.inf), axis=0)
            for robot in np.flatnonzero(needs_mopup):
                assigned = pending[robot] & (owner == robot)
                candidates = assigned if np.any(assigned) else pending[robot]
                if np.any(candidates):
                    score = np.where(candidates, distance[robot], np.inf)
                    goals[robot] = centers[int(np.argmin(score))]
        return goals


class BoscoController:
    """Deterministic boustrophedon expert, driven from the render snapshot.

    `BoscoExpert` is a host-side numpy planner, and the snapshot already pulls
    exactly the three arrays it reads — poses and the coverage grid — back for
    drawing, so feeding it the snapshot adds no device round-trip of its own.
    """

    label = 'BOSCO '

    def __init__(self, env: MultiRobotCoverageEnv):
        self.env = env
        self.expert = BoscoExpert(env)
        self._step = jax.jit(env.step)
        self.owner = None

    def reset(self, state):
        current_map_id = int(jax.device_get(state.map_id))
        info = self.expert.reset(np.asarray(state.robot_positions), map_id=current_map_id)
        # -1 (unowned) folds onto the palette's trailing fallback row.
        owner = self.expert.owner.copy()
        owner[owner < 0] = self.env.num_robots
        self.owner = owner.reshape(self.env.grid_h, self.env.grid_w)
        print(f"  partition {info['region_sizes'].tolist()}", end='', flush=True)
        return state

    def step(self, state, snap: dict):
        actions = self.expert.act(
            snap['positions'], snap['headings'], snap['coverage_grid']
        )
        state, rewards, terminated, truncated = self._step(state, jnp.asarray(actions))
        return state, rewards, terminated, truncated

    def bosco_target_centers(self) -> np.ndarray:
        """Current committed leg endpoint for each plain-BOSCO robot."""
        targets = np.full((self.env.num_robots, 2), np.nan, dtype=np.float32)
        for robot, tour in enumerate(self.expert.tours):
            target_idx = int(self.expert.tgt[robot])
            # tgt is a cursor into the mutable tour, not a flat cell id.  A
            # value behind idx belongs to a retired/replanned leg and must not
            # linger on screen during recovery.
            if (not self.expert.done[robot]
                    and self.expert.spin[robot] == 0
                    and int(self.expert.idx[robot]) <= target_idx < len(tour)):
                targets[robot] = self.expert.graph.centers[tour[target_idx]]
        return targets

    def bosco_goal_centers(self, coverage_grid: np.ndarray,
                           positions: np.ndarray) -> np.ndarray:
        """First still-uncovered cell in each robot's live BOSCO plan."""
        del positions  # kept for the same snapshot hook used by MAPPO
        goals = np.full((self.env.num_robots, 2), np.nan, dtype=np.float32)
        covered = np.asarray(coverage_grid).reshape(-1) > 0.5
        centers = self.expert.graph.centers
        for robot, tour in enumerate(self.expert.tours):
            start = int(self.expert.idx[robot])
            remaining = np.asarray(tour[start:], dtype=np.int64)
            if remaining.size == 0:
                continue
            pending = remaining[~covered[remaining]]
            if pending.size:
                goals[robot] = centers[pending[0]]
        return goals


def run_episode(
    env: MultiRobotCoverageEnv,
    controller,
    key: jax.Array,
    surface: pygame.Surface,
    clock: pygame.time.Clock,
    font: pygame.font.Font,
    popup_font: pygame.font.Font,
    palette: np.ndarray,
    fps: int,
    view_state: dict,
    map_id: int,
) -> tuple[float, bool] | None:
    """Run one episode; the bool requests a different layout next time."""
    state = env.reset(key, jnp.int32(map_id))
    state = controller.reset(state)

    ep_reward = 0.0
    
    # Extract the map_id for this episode and fetch the specific walls/free mask
    current_map_id = int(jax.device_get(state.map_id))
    current_walls = np.asarray(env.walls[current_map_id])
    current_free = np.asarray(env.free_mask_np[current_map_id])

    snap = _snapshot(env, state, view_state['show_lidar'], controller)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_SPACE:
                    view_state['paused'] = not view_state.get('paused', False)
                if event.key == pygame.K_r:
                    return ep_reward, False  # new spawn, same layout
                if event.key == pygame.K_RIGHT:
                    return ep_reward, True   # new spawn and next procedural layout
                if event.key == pygame.K_l:
                    view_state['show_lidar'] = not view_state['show_lidar']
                if event.key == pygame.K_o:
                    view_state['show_owner'] = not view_state['show_owner']
                if event.key == pygame.K_s:
                    view_state['speed_idx'] = (
                        view_state['speed_idx'] + 1) % len(SPEED_LEVELS)

        speed_idx   = view_state['speed_idx']
        current_fps = SPEED_LEVELS[speed_idx]
        speed_label = _SPEED_LABELS[speed_idx]
        if view_state.get('paused', False):
            speed_label = 'PAUSED'

        owner = controller.owner if view_state['show_owner'] else None
        popup = view_state.get('popup')
        if popup is not None and pygame.time.get_ticks() >= popup['until_ms']:
            popup = None
            view_state['popup'] = None
        
        # USA current_walls INVECE DI walls
        _draw_frame(surface, env, current_walls, current_free, snap, font,
                    ep_reward, view_state['show_lidar'], owner, palette,
                    controller.label, speed_label, popup, popup_font)
        pygame.display.flip()
        
        if view_state.get('paused', False):
            clock.tick(60) # keep responsive
            continue
            
        clock.tick(current_fps)

        state, rewards, terminated, truncated = controller.step(state, snap)
        ep_reward += float(jnp.mean(rewards))
        snap = _snapshot(env, state, view_state['show_lidar'], controller)

        if bool(terminated) or bool(truncated):
            outcome = _episode_outcome(snap)
            if outcome is not None:
                label, color = outcome
                view_state['popup'] = {
                    'label': label,
                    'color': color,
                    'until_ms': pygame.time.get_ticks() + POPUP_DURATION_MS,
                }
            _draw_frame(surface, env, current_walls, current_free, snap, font,
                        ep_reward, view_state['show_lidar'], owner, palette,
                        controller.label, speed_label, view_state.get('popup'),
                        popup_font)
            pygame.display.flip()
            if current_fps > 0:
                clock.tick(current_fps)
            return ep_reward, True


def _load_checkpoint(
    path: str, device: jax.Device, env_config: dict | None = None
) -> tuple[dict, RunningMeanStd | None, int, bool, float]:
    """Read a JAX training checkpoint and place its arrays on `device`."""
    exc = None
    for attempt in range(5):
        try:
            with open(path, 'rb') as f:
                ckpt = pickle.load(f)
            break
        except (OSError, pickle.UnpicklingError, UnicodeDecodeError, EOFError) as error:
            exc = error
            if attempt < 4:
                time.sleep(0.1)
    else:
        raise SystemExit(
            f"Cannot read '{path}' as a JAX checkpoint ({exc}).\n"
            "PyTorch-era '.pt' checkpoints are not loadable by the JAX policy: "
            "retrain with `python -m src.train_simple` to produce "
            "'checkpoints/checkpoint_final.pkl'."
        ) from exc

    if 'actor_params' not in ckpt:
        raise SystemExit(
            f"'{path}' has no 'actor_params' entry — it is not a JAX checkpoint "
            "written by src.train_simple."
        )

    if env_config is not None:
        for flag in ('actor_bosco_guidance', 'bosco_reward_guidance'):
            if flag in ckpt:
                env_config[flag] = bool(ckpt[flag])

    params = jax.device_put(ckpt['actor_params'], device)
    rms = None
    if ckpt.get('obs_rms') is not None:
        rms = RunningMeanStd(*jax.device_put(tuple(ckpt['obs_rms']), device))
    # Older guided checkpoints wrote guide_dim=0 after the waypoint moved into
    # the continuous observation prefix. Preserve compatibility with the
    # canonical BOSCO filename; new checkpoints carry an explicit flag.
    guided = bool(ckpt.get(
        'bosco_guided', os.path.basename(path) == 'checkpoint_bosco.pkl'
    ))
    return (params, rms, int(ckpt.get('update', 0)), guided,
            float(ckpt.get('guide_bonus', 2.0)))


def main() -> None:
    _default_cfg  = os.path.join(_ROOT, 'config', 'mappo_baseline.yaml')
    # The current environment/actor layout matches the BOSCO-guided checkpoint.
    # checkpoint_final.pkl predates the current 70-ray observation and fails on
    # its first forward pass with an incompatible Dense_0 input shape.
    _default_ckpt = os.path.join(_ROOT, 'checkpoints', 'checkpoint_bosco.pkl')

    parser = argparse.ArgumentParser(description='Visualise a coverage controller with pygame')
    parser.add_argument('--policy',     default='mappo', choices=['mappo', 'bosco'],
                        help='Controller to run: the trained MAPPO policy, or the '
                             'deterministic BOSCO expert (default: mappo)')
    parser.add_argument('--max-steps',  type=int, default=0,
                        help='Override the episode step limit; 0 = take it from the '
                             'config (default: 0)')
    parser.add_argument('--checkpoint', default=_default_ckpt,
                        help='Path to .pkl checkpoint (default: checkpoints/checkpoint_bosco.pkl)')
    parser.add_argument('--config',     default=_default_cfg,
                        help='Path to YAML config file')
    parser.add_argument('--episodes',   type=int, default=0,
                        help='Episodes to run; 0 = loop forever (default: 0)')
    parser.add_argument('--layouts',    type=int, default=16,
                        help='Procedural layouts kept in the visualisation bank; '
                             'episodes cycle through them (default: 16)')
    parser.add_argument('--fps',        type=int, default=FPS,
                        help=f'Rendering FPS (default: {FPS})')
    parser.add_argument('--seed',       type=int, default=0,
                        help='PRNG seed for episode resets (default: 0)')
    parser.add_argument('--no-obs-norm', action='store_true',
                        help='Disable observation normalisation')
    parser.add_argument('--bosco-guided', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Enable/disable BOSCO waypoint observations. By default '
                             'this is detected from checkpoint metadata/name.')
    parser.add_argument('--guide-bonus', type=float, default=None,
                        help='Override BOSCO arrival bonus used for displayed reward; '
                             'default reads checkpoint metadata (2.0 for old files).')
    parser.add_argument('--backend',    default='cpu',
                        choices=['auto', 'metal', 'cuda', 'gpu', 'cpu'],
                        help='JAX backend. Default "cpu": a single-env rollout is '
                             'tiny, so the CPU beats accelerator launch overhead')
    parser.add_argument('--humans', nargs='?', type=int, const=3, default=0, help='Number of humans')
    parser.add_argument('--terminate-on-collision',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='Override collision termination; default uses the '
                             'training config.')
    args = parser.parse_args()

    if args.layouts < 2:
        parser.error('--layouts must be at least 2 so RIGHT can select a new layout')

    config    = load_config(args.config)
    env_cfg   = config.get('env',   {})
    # Guided training uses one map, but visual evaluation rebuilds the BOSCO
    # graph at every reset and can therefore cycle through a larger map bank.
    env_cfg = {**env_cfg, 'num_maps': args.layouts}
    if args.humans > 0:
        env_cfg['num_humans'] = args.humans
    if args.terminate_on_collision is not None:
        env_cfg['terminate_on_collision'] = args.terminate_on_collision
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    # Must run before any array is created so implicit placement follows it.
    device = select_device(None if args.backend == 'auto' else args.backend)
    print(f"Device: {describe(device)}")

    if args.max_steps > 0:
        env_cfg = {**env_cfg, 'max_steps': args.max_steps}
    elif args.policy == 'bosco' and env_cfg.get('max_steps', 500) < _BOSCO_MIN_STEPS:
        # A full BOSCO sweep of the default map takes ~2100 steps; the training
        # config truncates long before that, which would look like the expert
        # failing rather than the clock running out.
        print(f"Raising max_steps {env_cfg.get('max_steps')} → {_BOSCO_MIN_STEPS} "
              "so the sweep can finish (override with --max-steps).")
        env_cfg = {**env_cfg, 'max_steps': _BOSCO_MIN_STEPS}

    if args.policy == "mappo":
        params, obs_rms, update, checkpoint_guided, checkpoint_guide_bonus = _load_checkpoint(
            args.checkpoint, device, env_cfg
        )
        print(f"Loaded: {args.checkpoint}  (update {update})")

    env   = MultiRobotCoverageEnv(env_cfg)
    #walls = np.asarray(env.walls)

    if args.policy == 'bosco':
        controller = BoscoController(env)
        print(f"Controller: BOSCO (deterministic, {env.num_robots} robots)")
    else:
        actor = Actor(
            action_dim=env.action_dim,
            vec_dim=env.obs_vec_dim,
            n_rays=env.n_rays,
            tail_dim=env.patch_dim,
            lidar_embed=model_cfg.get('lidar_embed',  64),
            hidden_size=model_cfg.get('hidden_size', 128),
        )


        if args.no_obs_norm or not train_cfg.get('normalize_obs', True):
            obs_rms = None
            print("Observation normalisation: disabled.")
        elif obs_rms is None:
            print("Warning: checkpoint has no obs_rms — running without normalisation.")
        else:
            print("Observation normalisation: loaded from checkpoint.")

        guided = checkpoint_guided if args.bosco_guided is None else args.bosco_guided
        guide_bonus = (checkpoint_guide_bonus if args.guide_bonus is None
                       else args.guide_bonus)
        if guided:
            print(f"Guidance: JAX BOSCO waypoints enabled; bonus={guide_bonus:g} "
                  "(matches training).")
        controller = MappoController(
            env, actor, params, obs_rms, guided=guided,
            guide_bonus=guide_bonus,
        )

    # -- Pygame setup --
    # -- Pygame setup --
    mw    = env.grid_w * env.cell_size
    mh    = env.grid_h * env.cell_size
    win_w = int(mw * SCALE) + 2 * MARGIN
    win_h = int(mh * SCALE) + 2 * MARGIN + HUD_HEIGHT

    pygame.init()
    surface = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f'{controller.label.strip()} Coverage — Visual Test')
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont('monospace', 15)
    popup_font = pygame.font.SysFont('monospace', 34, bold=True)

    palette = _ownership_palette(env.num_robots)
    # The overlay only has something to say once a partition exists.
    view_state = {'show_lidar': False, 'show_owner': True,
                  'speed_idx': 0, 'popup': None}
    key = jax.random.PRNGKey(args.seed)

    ep_num = 0
    map_id = 0
    try:
        while args.episodes == 0 or ep_num < args.episodes:
            key, ep_key = jax.random.split(key)
            print(f"Episode {ep_num + 1} (layout {map_id + 1}/{env.num_maps}) ...",
                  end='', flush=True)
            ret = run_episode(env, controller, ep_key, surface, clock, font,
                               popup_font, palette, args.fps, view_state, map_id)
            if ret is None:
                print("  (quit)")
                break
            reward, advance_layout = ret
            print(f"  total reward = {reward:.2f}")
            ep_num += 1
            if advance_layout:
                map_id = (map_id + 1) % env.num_maps
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
