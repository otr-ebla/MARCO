#!/usr/bin/env python3
"""Visualise a trained coverage policy in real time using pygame (JAX).

Usage (from the project root):
    python -m src.visualize_policy
    python -m src.visualize_policy --checkpoint checkpoints/checkpoint_e2e.pkl --episodes 10
    python -m src.visualize_policy --config config/mappo_sequential.yaml --expert [--target-rule tour]

Playback controls: RIGHT or R skips the current episode, SPACE pauses, L toggles
LiDAR, S changes speed, and ESC quits.
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
    'fallback':  (120, 120, 120),
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


def _robot_color(snap: dict, index: int) -> tuple[int, int, int]:
    if not snap['alive'][index]:
        return COLORS['dead']
    if snap['fallback_active'][index]:
        return COLORS['fallback']
    return _ROBOT_COLORS[index % len(_ROBOT_COLORS)]


def _to_px(x: float, y: float, map_h: float) -> tuple[int, int]:
    """World metres → pygame pixel (top-left origin, y-axis flipped)."""
    return (int(x * SCALE) + MARGIN,
            int((map_h - y) * SCALE) + MARGIN)


def _snapshot(env: MultiRobotCoverageEnv, state, want_lidar: bool) -> dict:
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
        state.robot_alive,
        state.fallback_active,
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
    (pos, hdg, human_pos, human_hdg, grid, alive, fallback_active, step, cov_ratio,
     covered, total, timeout, wall_hits, robot_hits, human_hits,
     lidar) = jax.device_get(payload)
    snap = {
        'positions':      np.asarray(pos),
        'headings':       np.asarray(hdg),
        'human_positions': np.asarray(human_pos),
        'human_headings': np.asarray(human_hdg),
        'coverage_grid':  np.asarray(grid),
        'alive':          np.asarray(alive),
        'fallback_active': np.asarray(fallback_active),
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
    return snap


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
    for row in range(env.grid_h):
        for col in range(env.grid_w):
            if free[row, col] == 0.0:
                color = COLORS['unreachable']
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

    # -- Lidar rays (toggle with 'L') --
    if show_lidar and snap['lidar'].size:
        angles_rel = np.linspace(0.0, 2.0 * np.pi, env.n_rays, endpoint=False)
        for i in range(env.num_robots):
            ray_color = _robot_color(snap, i)
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
        color  = _robot_color(snap, i)
        pygame.draw.circle(surface, color, (cx, cy), r_px)
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
            f"[L] lidar [S] speed")
    rendered = font.render(text, True, COLORS['hud_text'])
    top = win_h - HUD_HEIGHT + (HUD_HEIGHT - rendered.get_height()) // 2
    surface.blit(rendered, (8, top))

    _draw_popup(surface, popup, popup_font or font)


class MappoController:
    """Trained policy. One jitted call per frame: normalise → act → step.

    Fusing the policy and the environment transition keeps a single device
    round-trip per rendered frame; only the render snapshot comes back.
    """

    label = 'MAPPO '

    def __init__(self, env: MultiRobotCoverageEnv, actor: Actor, params,
                 obs_rms: RunningMeanStd | None):
        self.env, self.params, self.obs_rms = env, params, obs_rms
        self.recurrent = actor.recurrent
        self.memory_size = actor.hidden_size
        self.memory = None

        @jax.jit
        def policy_step(params, rms, state, obs, memory):
            obs_n = rms_normalize(rms, obs) if rms is not None else obs
            if actor.recurrent:
                mean, _, memory = actor.apply(params, obs_n, memory)
            else:
                mean, _ = actor.apply(params, obs_n)
            action = jnp.tanh(mean)                   # deterministic
            next_state, rewards, terminated, truncated = env.step(state, action)
            return (next_state, env.get_obs(next_state),
                    rewards, terminated, truncated, memory)

        self._fn = policy_step

    def reset(self, state):
        self.memory = (jnp.zeros((self.env.num_robots, self.memory_size), jnp.float32)
                       if self.recurrent else None)
        self.obs = self.env.get_obs(state)
        return state

    def step(self, state):
        state, self.obs, rewards, terminated, truncated, self.memory = self._fn(
            self.params, self.obs_rms, state, self.obs, self.memory,
        )
        return state, rewards, terminated, truncated


class ExpertController:
    """BCD expert (src.envs.bcd_expert) driving every robot; no checkpoint needed."""

    def __init__(self, env: MultiRobotCoverageEnv, expert_config: dict):
        from src.envs.bcd_expert import BCDExpert
        self.expert = BCDExpert(env, expert_config)
        self.label = f'BCD expert ({self.expert.target_rule}) '

        @jax.jit
        def expert_step(state, chunk):
            action, chunk, _ = self.expert.act(state, chunk)
            next_state, rewards, terminated, truncated = env.step(state, action)
            return next_state, chunk, rewards, terminated, truncated

        self._fn = expert_step

    def reset(self, state):
        self.chunk = self.expert.init_chunks()   # reassigned at step 0
        return state

    def step(self, state):
        state, self.chunk, rewards, terminated, truncated = self._fn(state, self.chunk)
        return state, rewards, terminated, truncated


def run_episode(
    env: MultiRobotCoverageEnv,
    controller,
    key: jax.Array,
    surface: pygame.Surface,
    clock: pygame.time.Clock,
    font: pygame.font.Font,
    popup_font: pygame.font.Font,
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

    snap = _snapshot(env, state, view_state['show_lidar'])

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
                if event.key == pygame.K_s:
                    view_state['speed_idx'] = (
                        view_state['speed_idx'] + 1) % len(SPEED_LEVELS)

        speed_idx   = view_state['speed_idx']
        current_fps = SPEED_LEVELS[speed_idx]
        speed_label = _SPEED_LABELS[speed_idx]
        if view_state.get('paused', False):
            speed_label = 'PAUSED'

        popup = view_state.get('popup')
        if popup is not None and pygame.time.get_ticks() >= popup['until_ms']:
            popup = None
            view_state['popup'] = None
        
        _draw_frame(surface, env, current_walls, current_free, snap, font,
                    ep_reward, view_state['show_lidar'],
                    controller.label, speed_label, popup, popup_font)
        pygame.display.flip()
        
        if view_state.get('paused', False):
            clock.tick(60) # keep responsive
            continue
            
        clock.tick(current_fps)

        state, rewards, terminated, truncated = controller.step(state)
        ep_reward += float(jnp.mean(rewards))
        snap = _snapshot(env, state, view_state['show_lidar'])

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
                        ep_reward, view_state['show_lidar'],
                        controller.label, speed_label, view_state.get('popup'),
                        popup_font)
            pygame.display.flip()
            if current_fps > 0:
                clock.tick(current_fps)
            return ep_reward, True


def _load_checkpoint(
    path: str, device: jax.Device, env_config: dict | None = None
) -> tuple[dict, RunningMeanStd | None, int]:
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
            "retrain with `python -m src.train_marl`."
        ) from exc

    if 'actor_params' not in ckpt:
        raise SystemExit(
            f"'{path}' has no 'actor_params' entry — it is not a JAX checkpoint "
            "written by src.train_marl or src.train_simple."
        )

    if env_config is not None:
        env_config.update(ckpt.get("reward_weights", {}))
        env_config.update({"use_full_memory": False, "observation_stack": 1, "sweep_obs": False,
                           "crop_summary": False, "crop_mode": "memory", "critic_crops": False,
                           "known_coverage_obs": False, "critic_coverage": False,
                           "goal_obs": False, "wall_cells": 0,
                           "history_cell": "last_discovery", "critic_context": False,
                           "critic_stack": 1})
        obs_config = ckpt.get("obs_config", {"obs_mode": "legacy"})
        env_config.update(obs_config)
        # Checkpoints older than memory_map_obs read the full map with use_full_memory.
        env_config["memory_map_obs"] = bool(obs_config.get("memory_map_obs", obs_config.get("use_full_memory", False)))
        env_config["actor_recurrent"] = bool(ckpt.get("actor_recurrent", False))
        env_config["actor_config"] = ckpt.get("actor_config", {})

    params = jax.device_put(ckpt['actor_params'], device)
    rms = None
    if ckpt.get('obs_rms') is not None:
        rms = RunningMeanStd(*jax.device_put(tuple(ckpt['obs_rms']), device))
    return params, rms, int(ckpt.get('update', 0))


def main() -> None:
    _default_cfg  = os.path.join(_ROOT, 'config', 'mappo_baseline.yaml')
    _default_ckpt = os.path.join(_ROOT, 'checkpoints', 'checkpoint_e2e.pkl')

    parser = argparse.ArgumentParser(description='Visualise a trained coverage policy with pygame')
    parser.add_argument('--max-steps',  type=int, default=0,
                        help='Override the episode step limit; 0 = take it from the '
                             'config (default: 0)')
    parser.add_argument('--checkpoint', default=_default_ckpt,
                        help='Path to .pkl checkpoint (default: checkpoints/checkpoint_e2e.pkl)')
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
    parser.add_argument('--policy-only', action='store_true',
                        help='Disable recovery after restoring checkpoint settings')
    parser.add_argument('--wall-cells', type=int, default=None,
                        help='Override checkpoint geometry: 0 = original thin walls, 1 = full-cell walls')
    parser.add_argument('--no-obs-norm', action='store_true',
                        help='Disable observation normalisation')
    parser.add_argument('--backend',    default='cpu',
                        choices=['auto', 'metal', 'cuda', 'gpu', 'cpu'],
                        help='JAX backend. Default "cpu": a single-env rollout is '
                             'tiny, so the CPU beats accelerator launch overhead')
    parser.add_argument('--humans', nargs='?', type=int, const=3, default=0, help='Number of humans')
    parser.add_argument('--expert', action='store_true',
                        help='Drive the robots with the BCD expert (config pretrain.expert) '
                             'instead of a checkpoint; recovery is disabled')
    parser.add_argument('--target-rule', choices=['local', 'tour'], default=None,
                        help='With --expert: override pretrain.expert.target_rule')
    parser.add_argument('--terminate-on-collision',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='Override collision termination; default uses the '
                             'training config.')
    args = parser.parse_args()

    if args.layouts < 2:
        parser.error('--layouts must be at least 2 so RIGHT can select a new layout')
    if args.wall_cells is not None and args.wall_cells < 0:
        parser.error('--wall-cells must be non-negative')

    config    = load_config(args.config)
    env_cfg   = config.get('env',   {})
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

    if args.expert:
        from src.envs.coverage_vector_env import E2E_REWARD_DEFAULTS
        # The training environment: e2e reward weights, the expert in place of recovery.
        env_cfg.update({**E2E_REWARD_DEFAULTS, **config.get('e2e_reward', {})})
        if args.wall_cells is not None:
            env_cfg['wall_cells'] = args.wall_cells
        env_cfg['fallback_enabled'] = False
        env = MultiRobotCoverageEnv(env_cfg)
        expert_cfg = dict(config.get('pretrain', {}).get('expert', {}))
        if args.target_rule is not None:
            expert_cfg['target_rule'] = args.target_rule
        controller = ExpertController(env, expert_cfg)
        print(f"Controller: {controller.label.strip()}")
    else:
        env, controller = _policy_controller(args, env_cfg, model_cfg, train_cfg, device)
    _run_viewer(args, env, controller)


def _policy_controller(args, env_cfg, model_cfg, train_cfg, device):
    params, obs_rms, update = _load_checkpoint(args.checkpoint, device, env_cfg)
    if args.wall_cells is not None:
        env_cfg['wall_cells'] = args.wall_cells
    if args.policy_only:
        env_cfg['fallback_enabled'] = False
    print(f"Loaded: {args.checkpoint}  (update {update})")

    env = MultiRobotCoverageEnv(env_cfg)
    actor_cfg = dict(lidar_embed=model_cfg.get('lidar_embed', 64),
                     hidden_size=model_cfg.get('hidden_size', 128))
    actor_cfg.update(env_cfg.get('actor_config', {}))
    actor = Actor(
        recurrent=env_cfg.get("actor_recurrent", False),
        action_dim=env.action_dim,
        vec_dim=env.obs_vec_dim,
        n_rays=env.n_rays,
        tail_dim=env.patch_dim,
        memory_map_shape=env.memory_map_shape,
        observation_stack=env.observation_stack,
        crop_shape=env.crop_shape,
        **actor_cfg,
    )

    if args.no_obs_norm or not train_cfg.get('normalize_obs', True):
        obs_rms = None
        print("Observation normalisation: disabled.")
    elif obs_rms is None:
        print("Warning: checkpoint has no obs_rms — running without normalisation.")
    else:
        print("Observation normalisation: loaded from checkpoint.")
    return env, MappoController(env, actor, params, obs_rms)


def _run_viewer(args, env, controller) -> None:
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

    view_state = {'show_lidar': False, 'speed_idx': 0, 'popup': None}
    key = jax.random.PRNGKey(args.seed)

    ep_num = 0
    map_id = 0
    try:
        while args.episodes == 0 or ep_num < args.episodes:
            key, ep_key = jax.random.split(key)
            print(f"Episode {ep_num + 1} (layout {map_id + 1}/{env.num_maps}) ...",
                  end='', flush=True)
            ret = run_episode(env, controller, ep_key, surface, clock, font,
                               popup_font, args.fps, view_state, map_id)
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
