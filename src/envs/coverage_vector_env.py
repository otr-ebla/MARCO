"""
JAX multi-robot indoor coverage environment.

The environment is a pure function of an immutable `EnvState` pytree, so
`reset` / `step` / `get_obs` can be jitted, vmapped over parallel environments,
and scanned over time without leaving the accelerator.
"""

from __future__ import annotations

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .map_layouts import ProceduralMapLayout, create_map_bank

_TWO_PI = 2.0 * np.pi
_BIG = 1.0e9
_FAR = 1.0e6   # geodesic distance of a cell that cannot reach any target

# Training defaults; checkpoints persist this complete reward specification.
# 'local_coverage_v1' is kept only so older checkpoints still evaluate.
E2E_REWARD_DEFAULTS = {
    'reward_mode': 'progress',
    'alpha': 10.0,                  # per newly covered cell
    'coverage_reward_growth': 2.0,  # late cells are worth up to 3x alpha
    'progress_weight': 2.0,         # per cell of geodesic progress towards work
    'loiter_cost': 1.0,             # per step without discovery or progress
    'spread_weight': 0.2,           # per step and teammate at zero distance
    'spread_radius': 2.5,           # metres; keep <= comm_radius to stay observable
    'tau': 0.02,
    'wall_kappa': 2.0,
    'kappa': 5.0,
    'human_kappa': 10.0,
    'completion_bonus': 200.0,
    'completion_time_bonus': 1.0,   # instant completion pays (1 + this) x bonus
}


@struct.dataclass
class EnvState:
    """Complete environment state. All fields are JAX arrays (a valid pytree)."""
    robot_positions:  jax.Array   # (N, 2)   float32
    robot_headings:   jax.Array   # (N,)     float32
    robot_velocities: jax.Array   # (N, 2)   float32  — [v, omega] commanded
    human_positions:  jax.Array   # (M, 2)   float32
    human_headings:   jax.Array   # (M,)     float32
    human_dists:      jax.Array   # (M,)     float32
    coverage_grid:    jax.Array   # (H, W)   float32  — 0.0 / 1.0
    room_completed:   jax.Array   # (R,)     bool
    robot_alive:      jax.Array   # (N,)     bool
    step_count:       jax.Array   # ()       int32
    map_id:           jax.Array   # ()       int32    — Active map for this episode
    key:              jax.Array   # PRNG key carried for auto-reset
    wall_hits:        jax.Array   # (N,)     float32  — 0.0 / 1.0
    robot_hits:       jax.Array   # (N,)     float32  — 0.0 / 1.0
    human_hits:       jax.Array   # (N,)     float32  — 0.0 / 1.0
    ghost_robot_prob: jax.Array   # ()       float32 — humans ignore robots with this probability
    # Decentralised per-robot memory ('memory_comm' obs mode; zeros otherwise).
    # Dense bitmaps are the jit/vmap-friendly form of a sparse cell dictionary.
    # Occupancy is not stored: a known cell's label is the static ground truth,
    # so occupied = mem_known & wall.
    mem_known:        jax.Array   # (N, H, W) float32 — 0.0 / 1.0, observed by lidar or shared
    mem_covered:      jax.Array   # (N, H, W) float32 — 0.0 / 1.0, covered as far as robot i knows
    lidar:            jax.Array   # (N, R)    float32 — latest normalised scan


@struct.dataclass
class GlobalState:
    """Centralised critic state, kept as small spatial tensors."""
    coverage:        jax.Array   # (H, W)     float32
    occupancy:       jax.Array   # (N, H, W)  float32, one-hot per robot
    kinematics:      jax.Array   # (N, 6)     float32, normalised
    human_positions: jax.Array   # (M, 2)     float32, normalised
    map_id:          jax.Array   # ()         int32


class MultiRobotCoverageEnv:
    def __init__(self, config: dict | None = None):
        cfg = config or {}

        # -- Environment parameters --
        self.num_robots      = int(cfg.get('num_robots',      3))
        self.num_humans      = int(cfg.get('num_humans',      0))
        self.k_teammates     = int(cfg.get('k_teammates',     2))
        self.m_humans        = int(cfg.get('m_humans',        1))
        self.n_rays          = int(cfg.get('n_rays',          36))
        self.max_lidar_range = float(cfg.get('max_lidar_range', 5.0))
        self.cell_size       = float(cfg.get('cell_size',       0.5))
        self.sensing_radius  = float(cfg.get('sensing_radius',  5.0))
        self.robot_radius    = float(cfg.get('robot_radius',    0.20))
        self.dt              = float(cfg.get('dt',              0.1))
        self.max_steps       = int(cfg.get('max_steps',       500))
        self.v_max           = float(cfg.get('v_max',           1.0))
        self.omega_max       = float(cfg.get('omega_max',       1.0))
        self.terminate_on_collision = bool(cfg.get('terminate_on_collision', False))
        self.reward_mode = cfg.get('reward_mode', 'legacy')
        if self.reward_mode not in ('legacy', 'local_coverage_v1', 'progress'):
            raise ValueError(f'Unknown reward_mode: {self.reward_mode}')
        self.use_local_coverage_obs = bool(cfg.get('use_local_coverage_obs', True))
        self.local_coverage_size = int(cfg.get('local_coverage_size', 5))
        if self.local_coverage_size <= 0 or self.local_coverage_size % 2 == 0:
            raise ValueError("local_coverage_size must be a positive odd integer")

        # 'legacy': velocity / teammates / lidar / coverage patch.
        # 'memory_comm': per-robot lidar-built map memory, merged (OR) with
        # teammates inside comm_radius; see _get_memory_obs for the layout.
        self.obs_mode = cfg.get('obs_mode', 'legacy')
        if self.obs_mode not in ('legacy', 'memory_comm'):
            raise ValueError(f'Unknown obs_mode: {self.obs_mode}')
        self.use_memory = self.obs_mode == 'memory_comm'
        self.comm_radius = float(cfg.get('comm_radius', 3.0))
        self.comm_slots = int(cfg.get('comm_slots', max(self.num_robots - 1, 0)))
        if self.comm_radius < 0.0 or self.comm_slots < 0:
            raise ValueError('comm_radius and comm_slots must be non-negative')

        # -- Reward weights --
        self.alpha       = float(cfg.get('alpha',       10.0))
        self.coverage_reward_growth = float(
            cfg.get('coverage_reward_growth', 2.0)
        )
        self.beta        = float(cfg.get('beta',         0.5))
        self.kappa       = float(cfg.get('kappa',        5.0))
        self.wall_kappa  = float(cfg.get('wall_kappa', self.kappa))
        self.human_kappa = float(cfg.get('human_kappa', self.kappa))
        self.tau         = float(cfg.get('tau',          0.05))
        self.psi         = float(cfg.get('psi',          2.0))
        self.velocity_cost = float(cfg.get('velocity_cost', 0.05))
        self.angular_cost = float(cfg.get('angular_cost', 0.01))
        self.action_smoothness_cost = float(
            cfg.get('action_smoothness_cost', 0.01)
        )
        self.progress_weight = float(cfg.get('progress_weight', 2.0))
        self.loiter_cost = float(cfg.get('loiter_cost', 1.0))
        self.spread_weight = float(cfg.get('spread_weight', 0.2))
        self.spread_radius = float(cfg.get('spread_radius', 2.5))
        self.completion_time_bonus = float(cfg.get('completion_time_bonus', 1.0))
        if self.spread_radius <= 0.0:
            raise ValueError('spread_radius must be positive')
        weights = {
            'alpha': self.alpha,
            'coverage_reward_growth': self.coverage_reward_growth,
            'beta': self.beta,
            'kappa': self.kappa,
            'wall_kappa': self.wall_kappa,
            'human_kappa': self.human_kappa,
            'tau': self.tau,
            'psi': self.psi,
            'velocity_cost': self.velocity_cost,
            'angular_cost': self.angular_cost,
            'action_smoothness_cost': self.action_smoothness_cost,
            'progress_weight': self.progress_weight,
            'loiter_cost': self.loiter_cost,
            'spread_weight': self.spread_weight,
            'completion_time_bonus': self.completion_time_bonus,
        }
        negative = [name for name, value in weights.items() if value < 0.0]
        if negative:
            raise ValueError(
                f"reward weights must be non-negative: {', '.join(negative)}"
            )
        self._safe_dist = float(cfg.get('safe_dist_factor', 5.0)) * self.robot_radius
        self.human_robot_stop_distance = float(
            cfg.get('human_robot_stop_distance', self._safe_dist)
        )
        self.room_completion_bonus     = float(cfg.get('room_completion_bonus',     50.0))
        self.room_completion_threshold = float(cfg.get('room_completion_threshold', 0.85))
        self.completion_bonus          = float(cfg.get('completion_bonus',         200.0))

        # -- Map Bank Precomputation --
        self.num_maps = int(cfg.get('num_maps', 16))
        layouts = create_map_bank(self.num_maps,
                                   seed=int(cfg.get('map_seed', 0)),
                                   cell_size=self.cell_size,
                                   robot_radius=self.robot_radius)
        
        # We keep the first layout purely for static dimension extraction
        self.map_layout = layouts[0]
        self.grid_w = int(np.ceil(self.map_layout.width  / self.cell_size))
        self.grid_h = int(np.ceil(self.map_layout.height / self.cell_size))
        self.num_cells = self.grid_h * self.grid_w

        # Stack walls for all generated maps (M, max_walls, 4)
        walls_np = np.stack([l.get_walls() for l in layouts])
        self.walls = jnp.asarray(walls_np)
        self._wall_x0 = self.walls[..., 0]
        self._wall_y0 = self.walls[..., 1]
        self._wall_x1 = self.walls[..., 2]
        self._wall_y1 = self.walls[..., 3]

        # Stack free space masks for all maps (M, H, W)
        free_masks_np = np.stack([self._compute_free_mask(w) for w in walls_np])
        self.free_mask_np = free_masks_np                            # host copy
        self.free_masks   = jnp.asarray(free_masks_np)
        self._free_flat   = self.free_masks.reshape(self.num_maps, -1)
        self.free_totals  = jnp.sum(self.free_masks, axis=(1, 2))
        self.wall_grids   = 1.0 - self.free_masks
        
        if np.any(self.free_totals < self.num_robots):
            raise RuntimeError("Free space is too small in one of the procedural maps.")

        # -- Room definitions (Simplified for procedural maps) --
        # We treat the entire connected free space as 1 single room per map.
        self.num_rooms = 1
        self.room_masks = jnp.expand_dims(self.free_masks, axis=1)   # (M, 1, H, W)
        self.room_totals = jnp.sum(self.room_masks, axis=(2, 3))     # (M, 1)

        # -- Derived dims --
        if self.use_memory:
            # pose (x, y, cos, sin) + (v, omega) + own id + (id, dx, dy) per slot
            self.obs_vec_dim = 4 + 2 + 1 + 3 * self.comm_slots
            # [occupied, covered, known] crop from the robot's memory
            self.patch_dim   = 3 * self.local_coverage_size ** 2
        else:
            self.obs_vec_dim = 2 + self.k_teammates * 2
            self.patch_dim   = (
                self.local_coverage_size ** 2 if self.use_local_coverage_obs else 0
            )
        self.norm_dim      = self.obs_vec_dim + self.n_rays
        self.obs_dim       = self.norm_dim + self.patch_dim
        self.action_dim    = 2
        self.critic_channels = 4
        self.critic_vec_dim  = 6 + 6 * self.num_robots + 2 * self.num_humans
        self._ray_angles = jnp.asarray(
            np.linspace(0.0, _TWO_PI, self.n_rays, endpoint=False, dtype=np.float32)
        )
        patch_axis = (
            np.arange(self.local_coverage_size, dtype=np.float32)
            - self.local_coverage_size // 2
        ) * self.cell_size
        patch_x, patch_y = np.meshgrid(patch_axis, patch_axis)
        self._local_patch_offsets = jnp.asarray(
            np.stack([patch_x, patch_y], axis=-1)
        )
        # Grid-aligned crop: integer cell offsets around the robot's own cell.
        self._crop_offsets = jnp.arange(
            self.local_coverage_size, dtype=jnp.int32
        ) - self.local_coverage_size // 2
        # Lidar cell marking: samples every half cell along each ray, so no
        # cell crossed by a beam is skipped.
        ray_step = 0.5 * self.cell_size
        self._ray_samples = jnp.asarray(
            np.arange(int(np.ceil(self.max_lidar_range / ray_step)) + 1,
                      dtype=np.float32) * ray_step
        )

        # -- Spawn candidates --
        xv, yv = self._cell_centers()
        centers = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)
        
        # Extract candidates per map and pad them to the max shape so they stack
        cands_list = [centers[fm.ravel() > 0.0] for fm in free_masks_np]
        max_cands = max(c.shape[0] for c in cands_list)
        
        padded_cands = []
        for c in cands_list:
            pad_size = max_cands - c.shape[0]
            if pad_size > 0:
                c = np.concatenate([c, np.tile(c[0], (pad_size, 1))], axis=0)
            padded_cands.append(c)
            
        self._spawn_candidates = jnp.asarray(np.stack(padded_cands))
        self._num_candidates   = max_cands
        self._spawn_clearance = 2.0 * self.robot_radius + 0.05
        self._spawn_needs_greedy = self.cell_size < self._spawn_clearance

        # Progress potential: own cell plus its 4-neighbours, whose distance
        # values are blended with the sub-cell offset so the potential moves
        # continuously while crossing a cell instead of jumping at borders.
        self._nb_dr = jnp.array([0, 1, -1, 0, 0], jnp.int32)
        self._nb_dc = jnp.array([0, 0, 0, 1, -1], jnp.int32)
        # Cells gained per step when driving straight at v_max.
        self._full_progress = max(self.v_max * self.dt / self.cell_size, 1e-6)

        self._k_eff = min(self.k_teammates, max(self.num_robots - 1, 0))
        self._m_eff = min(self.m_humans, self.num_humans)
        self._robot_ids = jnp.arange(self.num_robots, dtype=jnp.int32)

    def _cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = (np.arange(self.grid_w) + 0.5) * self.cell_size
        ys = (np.arange(self.grid_h) + 0.5) * self.cell_size
        return np.meshgrid(xs, ys)

    def _compute_free_mask(self, walls: np.ndarray) -> np.ndarray:
        xv, yv = self._cell_centers()
        centers = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)

        cx = np.clip(centers[:, 0:1], walls[None, :, 0], walls[None, :, 2])
        cy = np.clip(centers[:, 1:2], walls[None, :, 1], walls[None, :, 3])
        d2 = (centers[:, 0:1] - cx) ** 2 + (centers[:, 1:2] - cy) ** 2
        clear = (~np.any(d2 < self.robot_radius ** 2, axis=1))
        clear = clear.reshape(self.grid_h, self.grid_w)

        seen = np.zeros_like(clear)
        best = np.zeros_like(clear)
        best_size = 0
        for r0 in range(self.grid_h):
            for c0 in range(self.grid_w):
                if not clear[r0, c0] or seen[r0, c0]:
                    continue
                comp = np.zeros_like(clear)
                queue = deque([(r0, c0)])
                seen[r0, c0] = True
                size = 0
                while queue:
                    r, c = queue.popleft()
                    comp[r, c] = True
                    size += 1
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if (0 <= nr < self.grid_h and 0 <= nc < self.grid_w
                                and clear[nr, nc] and not seen[nr, nc]):
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                if size > best_size:
                    best, best_size = comp, size
        return best.astype(np.float32)

    def _wall_collision(self, pos: jax.Array, map_id: jax.Array) -> jax.Array:
        wx0 = self._wall_x0[map_id]
        wx1 = self._wall_x1[map_id]
        wy0 = self._wall_y0[map_id]
        wy1 = self._wall_y1[map_id]
        cx = jnp.clip(pos[:, 0:1], wx0[None, :], wx1[None, :])
        cy = jnp.clip(pos[:, 1:2], wy0[None, :], wy1[None, :])
        d2 = (pos[:, 0:1] - cx) ** 2 + (pos[:, 1:2] - cy) ** 2
        return jnp.any(d2 < self.robot_radius ** 2, axis=1)

    @staticmethod
    def _pairwise_sq_dist(pos: jax.Array) -> jax.Array:
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = jnp.sum(diff * diff, axis=-1)
        n = pos.shape[0]
        return d2 + jnp.eye(n, dtype=d2.dtype) * _BIG

    def _robot_human_surface_distance(
        self, robot_pos: jax.Array, human_pos: jax.Array, human_hdg: jax.Array
    ) -> jax.Array:
        """Signed edge-to-edge distance for circular robots and elliptical humans."""
        delta = robot_pos[:, None, :] - human_pos[None, :, :]
        center_dist = jnp.linalg.norm(delta, axis=-1)
        direction = delta / jnp.maximum(center_dist[..., None], 1e-8)

        c = jnp.cos(human_hdg)[None, :]
        s = jnp.sin(human_hdg)[None, :]
        along = direction[..., 0] * c + direction[..., 1] * s
        across = -direction[..., 0] * s + direction[..., 1] * c

        semi_along = self.robot_radius * 0.6
        semi_across = self.robot_radius * 1.2
        human_edge_radius = (semi_along * semi_across) / jnp.sqrt(
            (semi_across * along) ** 2 + (semi_along * across) ** 2
        )
        return center_dist - self.robot_radius - human_edge_radius

    def _pos_to_cell(self, pos: jax.Array) -> tuple[jax.Array, jax.Array]:
        col = jnp.clip(jnp.floor(pos[:, 0] / self.cell_size), 0, self.grid_w - 1)
        row = jnp.clip(jnp.floor(pos[:, 1] / self.cell_size), 0, self.grid_h - 1)
        return col.astype(jnp.int32), row.astype(jnp.int32)

    def _geodesic_distance(self, targets: jax.Array, free: jax.Array) -> jax.Array:
        """Multi-source 4-connected distance, in cells, from every free cell to
        the nearest target cell. Cells that reach no target hold `_FAR`.

        targets, free : (..., H, W) bool
        """
        far = jnp.float32(_FAR)
        lead = [(0, 0)] * (free.ndim - 2)

        def relax(d):
            pad = jnp.pad(d, lead + [(1, 1), (1, 1)], constant_values=far)
            nb = jnp.minimum(
                jnp.minimum(pad[..., :-2, 1:-1], pad[..., 2:, 1:-1]),
                jnp.minimum(pad[..., 1:-1, :-2], pad[..., 1:-1, 2:]),
            )
            return jnp.where(free, jnp.minimum(d, nb + 1.0), far)

        def cond(carry):
            _, changed, it = carry
            return changed & (it < self.num_cells)

        def body(carry):
            d, _, it = carry
            nd = relax(d)
            return nd, jnp.any(nd != d), it + 1

        d0 = jnp.where(targets & free, 0.0, far).astype(jnp.float32)
        d, _, _ = jax.lax.while_loop(cond, body, (d0, jnp.bool_(True), jnp.int32(0)))
        return d

    def _work_distance(self, dist: jax.Array, pos: jax.Array) -> jax.Array:
        """Continuous distance, in cells, from each robot to its nearest target.

        dist : (N, H, W) per-robot geodesic field; pos : (N, 2) metres.
        The value is min over the own cell and its 4-neighbours c of
        dist[c] + |pos - centre(c)| / cell_size, so it decreases smoothly as
        the robot drives towards a neighbour lying on a shorter path.
        """
        col, row = self._pos_to_cell(pos)
        r = row[:, None] + self._nb_dr[None, :]
        c = col[:, None] + self._nb_dc[None, :]
        inside = (r >= 0) & (r < self.grid_h) & (c >= 0) & (c < self.grid_w)
        r = jnp.clip(r, 0, self.grid_h - 1)
        c = jnp.clip(c, 0, self.grid_w - 1)
        d = jnp.where(inside, dist[self._robot_ids[:, None], r, c], _FAR)
        centre = (jnp.stack([c, r], axis=-1).astype(jnp.float32) + 0.5) * self.cell_size
        offset = jnp.linalg.norm(pos[:, None, :] - centre, axis=-1) / self.cell_size
        return jnp.min(d + offset, axis=1)

    def _diff_drive(
        self, pos: jax.Array, heading: jax.Array, v: jax.Array, omega: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        straight = jnp.abs(omega) < 1e-6
        omega_safe = jnp.where(straight, 1.0, omega)
        turn_heading = heading + omega * self.dt
        radius       = v / omega_safe

        pos_straight = pos + jnp.stack(
            [v * jnp.cos(heading), v * jnp.sin(heading)], axis=-1
        ) * self.dt
        pos_turn = pos + radius[:, None] * jnp.stack([
            jnp.sin(turn_heading) - jnp.sin(heading),
            jnp.cos(heading) - jnp.cos(turn_heading),
        ], axis=-1)

        new_pos     = jnp.where(straight[:, None], pos_straight, pos_turn)
        new_heading = jnp.where(straight, heading, turn_heading)
        return new_pos, jnp.mod(new_heading, _TWO_PI)

    def _cast_lidar_single(
        self, pos: jax.Array, heading: jax.Array,
        all_pos: jax.Array, other_mask: jax.Array, map_id: jax.Array,
        human_pos: jax.Array, human_hdg: jax.Array
    ) -> jax.Array:
        angles = heading + self._ray_angles
        dx = jnp.cos(angles)
        dy = jnp.sin(angles)

        wx0 = self._wall_x0[map_id]
        wx1 = self._wall_x1[map_id]
        wy0 = self._wall_y0[map_id]
        wy1 = self._wall_y1[map_id]
        
        tx0 = (wx0[None, :] - pos[0]) / dx[:, None]
        tx1 = (wx1[None, :] - pos[0]) / dx[:, None]
        ty0 = (wy0[None, :] - pos[1]) / dy[:, None]
        ty1 = (wy1[None, :] - pos[1]) / dy[:, None]

        t_near = jnp.maximum(jnp.minimum(tx0, tx1), jnp.minimum(ty0, ty1))
        t_far  = jnp.minimum(jnp.maximum(tx0, tx1), jnp.maximum(ty0, ty1))

        valid = (t_near <= t_far + 1e-9) & (t_far > 1e-6)
        t_hit = jnp.where(valid, jnp.maximum(t_near, 1e-6), jnp.inf)
        t_min = jnp.min(t_hit, axis=1)

        dirs = jnp.stack([dx, dy], axis=1)
        w    = pos[None, :] - all_pos
        b    = 2.0 * (w @ dirs.T)
        c    = jnp.sum(w * w, axis=1, keepdims=True) - self.robot_radius ** 2
        disc = b ** 2 - 4.0 * c
        t_r  = (-b - jnp.sqrt(jnp.maximum(disc, 0.0))) * 0.5
        t_r  = jnp.where((disc >= 0) & (t_r > 1e-6) & other_mask[:, None], t_r, jnp.inf)
        t_min = jnp.minimum(t_min, jnp.min(t_r, axis=0))

        if self.num_humans > 0:
            a = self.robot_radius * 0.6  # Semi-axis along heading
            b_ax = self.robot_radius * 1.2  # Semi-axis perpendicular to heading
            a2 = a ** 2
            b2 = b_ax ** 2
            
            h_cos = jnp.cos(human_hdg)
            h_sin = jnp.sin(human_hdg)
            
            w_h = pos[None, :] - human_pos
            ox = w_h[:, 0] * h_cos + w_h[:, 1] * h_sin
            oy = -w_h[:, 0] * h_sin + w_h[:, 1] * h_cos
            
            dx_loc = dx[None, :] * h_cos[:, None] + dy[None, :] * h_sin[:, None]
            dy_loc = -dx[None, :] * h_sin[:, None] + dy[None, :] * h_cos[:, None]
            
            A = (dx_loc**2) / a2 + (dy_loc**2) / b2
            B = 2.0 * ((ox[:, None] * dx_loc) / a2 + (oy[:, None] * dy_loc) / b2)
            C = jnp.expand_dims((ox**2) / a2 + (oy**2) / b2 - 1.0, axis=-1)
            
            disc_h = B**2 - 4.0 * A * C
            t_h = (-B - jnp.sqrt(jnp.maximum(disc_h, 0.0))) / (2.0 * A)
            t_h = jnp.where((disc_h >= 0) & (t_h > 1e-6), t_h, jnp.inf)
            
            t_min = jnp.minimum(t_min, jnp.min(t_h, axis=0))

        dist = jnp.clip(t_min, 0.0, self.max_lidar_range)
        return dist / self.max_lidar_range

    def _cast_lidar_all(self, state: EnvState) -> jax.Array:
        n = self.num_robots
        not_self = ~jnp.eye(n, dtype=bool)
        return jax.vmap(
            self._cast_lidar_single, in_axes=(0, 0, None, 0, None, None, None)
        )(state.robot_positions, state.robot_headings, state.robot_positions, not_self, state.map_id, state.human_positions, state.human_headings)

    def _refresh_memory(self, state: EnvState, cover_mask: jax.Array) -> EnvState:
        """Scan, mark observed/covered cells, then exchange maps within comm range.

        cover_mask : (N,) bool — robot i covered the cell it now stands on.
        """
        n = self.num_robots
        ids = self._robot_ids
        pos = state.robot_positions
        alive = state.robot_alive
        lidar = self._cast_lidar_all(state)                          # (N, R)

        # Every sample strictly before the first hit is a seen cell, plus the
        # cell just behind the hit point (the obstacle's own cell). Labels are
        # static ground truth, so marking "known" is all the memory needs.
        dist = lidar * self.max_lidar_range
        ts = jnp.broadcast_to(self._ray_samples, (*dist.shape, self._ray_samples.shape[0]))
        ts = jnp.concatenate([ts, dist[..., None] + 1e-3], axis=-1)  # (N, R, K+1)
        valid = jnp.concatenate(
            [ts[..., :-1] < dist[..., None], (lidar < 1.0)[..., None]], axis=-1
        )
        angles = state.robot_headings[:, None] + self._ray_angles[None, :]
        px = pos[:, None, None, 0] + ts * jnp.cos(angles)[..., None]
        py = pos[:, None, None, 1] + ts * jnp.sin(angles)[..., None]
        cols = jnp.floor(px / self.cell_size).astype(jnp.int32)
        rows = jnp.floor(py / self.cell_size).astype(jnp.int32)
        valid = (valid & (cols >= 0) & (cols < self.grid_w)
                 & (rows >= 0) & (rows < self.grid_h) & alive[:, None, None])
        flat = (jnp.clip(rows, 0, self.grid_h - 1) * self.grid_w
                + jnp.clip(cols, 0, self.grid_w - 1)).reshape(n, -1)
        known = state.mem_known.reshape(n, -1).at[ids[:, None], flat].max(
            valid.reshape(n, -1).astype(jnp.float32)
        )

        own_c, own_r = self._pos_to_cell(pos)
        covered = state.mem_covered.reshape(n, -1).at[ids, own_r * self.grid_w + own_c].max(
            cover_mask.astype(jnp.float32)
        )

        # Single-hop exchange: OR over direct neighbours (and self). Multi-hop
        # spread still happens over successive steps.
        in_range = self._pairwise_sq_dist(pos) <= self.comm_radius ** 2
        adj = (in_range & alive[:, None] & alive[None, :]) | jnp.eye(n, dtype=bool)
        adj = adj.astype(jnp.float32)
        known = jnp.minimum(adj @ known, 1.0)
        covered = jnp.minimum(adj @ covered, 1.0)

        shape = (n, self.grid_h, self.grid_w)
        return state.replace(
            mem_known=known.reshape(shape),
            mem_covered=covered.reshape(shape),
            lidar=lidar,
        )

    def _sample_spawns(self, key: jax.Array, map_id: jax.Array, num_spawns: int) -> jax.Array:
        cands = self._spawn_candidates[map_id]
        if not self._spawn_needs_greedy:
            idx = jax.random.permutation(key, self._num_candidates)[: num_spawns]
            return cands[idx]

        shuffled = jax.random.permutation(key, cands, axis=0)
        slots = jnp.arange(num_spawns)

        def body(carry, cand):
            chosen, count = carry
            d = jnp.sqrt(jnp.sum((chosen - cand[None, :]) ** 2, axis=1))
            active = slots < count
            clear = jnp.all(jnp.where(active, d >= self._spawn_clearance, True))
            take = clear & (count < num_spawns)
            write = take & (slots == count)
            chosen = jnp.where(write[:, None], cand[None, :], chosen)
            return (chosen, count + take.astype(count.dtype)), None

        init = (jnp.zeros((num_spawns, 2), jnp.float32), jnp.int32(0))
        (chosen, _), _ = jax.lax.scan(body, init, shuffled)
        return chosen

    def reset(self, key: jax.Array, map_id: jax.Array | None = None) -> EnvState:
        state = self._reset_state(key, map_id)
        if self.use_memory:
            # The spawn cell is not covered, matching the global coverage grid.
            state = self._refresh_memory(
                state, jnp.zeros((self.num_robots,), bool)
            )
        return state

    def step(
        self, state: EnvState, joint_actions: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:
        state, rewards, terminated, truncated, cover = self._step_core(
            state, joint_actions
        )
        if self.use_memory:
            state = self._refresh_memory(state, cover)
        return state, rewards, terminated, truncated

    # `_reset_state` / `_step_core` leave the memory and lidar untouched so
    # VecEnv can auto-reset first and then scan once per step: under vmap a
    # scanning reset would be evaluated for every env on every step, doubling
    # the dominant lidar cost.

    def _reset_state(self, key: jax.Array, map_id: jax.Array | None = None) -> EnvState:
        key, map_key, spawn_key, r_hdg_key, h_hdg_key, h_dist_key = jax.random.split(key, 6)
        if map_id is None:
            map_id = jax.random.randint(map_key, (), 0, self.num_maps)
        else:
            map_id = jnp.asarray(map_id, dtype=jnp.int32)
        
        total_spawns = self.num_robots + self.num_humans
        spawns = self._sample_spawns(spawn_key, map_id, total_spawns)
        
        robot_positions = spawns[:self.num_robots]
        human_positions = spawns[self.num_robots:]
        
        return EnvState(
            robot_positions  = robot_positions,
            robot_headings   = jax.random.uniform(
                r_hdg_key, (self.num_robots,), minval=0.0, maxval=_TWO_PI
            ),
            robot_velocities = jnp.zeros((self.num_robots, 2), jnp.float32),
            human_positions  = human_positions,
            human_headings   = jax.random.uniform(
                h_hdg_key, (self.num_humans,), minval=0.0, maxval=_TWO_PI
            ),
            human_dists      = jax.random.uniform(
                h_dist_key, (self.num_humans,), minval=0.5, maxval=5.0
            ),
            coverage_grid    = jnp.zeros((self.grid_h, self.grid_w), jnp.float32),
            room_completed   = jnp.zeros((self.num_rooms,), bool),
            robot_alive      = jnp.ones((self.num_robots,), bool),
            step_count       = jnp.int32(0),
            map_id           = map_id,
            key              = key,
            wall_hits        = jnp.zeros((self.num_robots,), jnp.float32),
            robot_hits       = jnp.zeros((self.num_robots,), jnp.float32),
            human_hits       = jnp.zeros((self.num_robots,), jnp.float32),
            # Evaluation is safe by default: humans always react to robots.
            # Training explicitly overrides this value with its curriculum.
            ghost_robot_prob = jnp.float32(0.0),
            mem_known        = jnp.zeros(
                (self.num_robots, self.grid_h, self.grid_w), jnp.float32
            ),
            mem_covered      = jnp.zeros(
                (self.num_robots, self.grid_h, self.grid_w), jnp.float32
            ),
            lidar            = jnp.zeros((self.num_robots, self.n_rays), jnp.float32),
        )

    def _step_core(
        self, state: EnvState, joint_actions: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Physics, coverage and reward; the last output is the (N,) bool mask
        of robots that covered their current cell, for `_refresh_memory`."""
        alive     = state.robot_alive
        v_cmds    = (joint_actions[:, 0] + 1.0) * 0.5 * self.v_max
        omega_cmds = joint_actions[:, 1] * self.omega_max
        prev_grid = state.coverage_grid

        cand_pos, cand_hdg = self._diff_drive(
            state.robot_positions, state.robot_headings, v_cmds, omega_cmds
        )
        wall_hit = self._wall_collision(cand_pos, state.map_id) & alive

        blocked  = wall_hit | ~alive
        next_pos = jnp.where(blocked[:, None], state.robot_positions, cand_pos)
        next_hdg = jnp.where(alive, cand_hdg, state.robot_headings)

        d2       = self._pairwise_sq_dist(next_pos)
        pair_ok  = (~wall_hit)[:, None] & (~wall_hit)[None, :]
        close    = (d2 < (2.0 * self.robot_radius) ** 2) & pair_ok
        robot_hit = jnp.any(close, axis=1) & alive

        key, h_hdg_key, h_dist_key, ghost_key = jax.random.split(state.key, 4)
        if self.num_humans > 0:
            h_v = 0.5
            dist_step = h_v * self.dt
            h_dx = dist_step * jnp.cos(state.human_headings)
            h_dy = dist_step * jnp.sin(state.human_headings)
            h_cand_pos = state.human_positions + jnp.stack([h_dx, h_dy], axis=-1)
            h_wall_hit = self._wall_collision(h_cand_pos, state.map_id)
            
            new_h_dists = state.human_dists - dist_step
            need_new = h_wall_hit | (new_h_dists <= 0)
            
            new_headings = jax.random.uniform(h_hdg_key, (self.num_humans,), minval=0.0, maxval=_TWO_PI)
            new_dists = jax.random.uniform(h_dist_key, (self.num_humans,), minval=0.5, maxval=5.0)
            
            final_h_headings = jnp.where(need_new, new_headings, state.human_headings)
            final_h_dists = jnp.where(need_new, new_dists, new_h_dists)
            
            # A non-ghost robot is a dynamic obstacle for a human.  Humans are
            # deliberately simple: they stop instead of planning a detour.
            # Sampling per human avoids making the whole crowd ghost at once.
            diff_hr = h_cand_pos[:, None, :] - next_pos[None, :, :]
            robot_near = jnp.any(
                jnp.sum(diff_hr * diff_hr, axis=-1)
                < self.human_robot_stop_distance ** 2,
                axis=1,
            )
            robot_is_ghost = jax.random.uniform(
                ghost_key, (self.num_humans,)
            ) < state.ghost_robot_prob
            human_stops = robot_near & ~robot_is_ghost

            new_human_pos = jnp.where(
                (h_wall_hit | human_stops)[:, None],
                state.human_positions,
                h_cand_pos,
            )

            surface_distance = self._robot_human_surface_distance(
                next_pos, new_human_pos, final_h_headings
            )
            # Contact counts: zero clearance means the two body edges touch.
            robot_hit_human = jnp.any(surface_distance <= 0.0, axis=1) & alive
        else:
            new_human_pos = state.human_positions
            final_h_headings = state.human_headings
            final_h_dists = state.human_dists
            robot_hit_human = jnp.zeros((self.num_robots,), dtype=bool)

        collided = (wall_hit | robot_hit | robot_hit_human) & alive
        alive_next = alive & ~collided if self.terminate_on_collision else alive

        moved   = alive & ~collided
        new_pos = jnp.where(moved[:, None], next_pos, state.robot_positions)
        new_hdg = next_hdg

        new_vel = jnp.stack([
            jnp.where(moved, v_cmds,     0.0),
            jnp.where(alive, omega_cmds, 0.0),
        ], axis=-1)

        cols, rows = self._pos_to_cell(new_pos)
        flat       = rows * self.grid_w + cols
        coverable  = self._free_flat[state.map_id, flat] > 0.0
        already    = prev_grid[rows, cols] > 0.0
        eligible   = moved & coverable & ~already
        
        ids   = self._robot_ids
        claim = jnp.zeros((self.num_cells,), jnp.int32).at[flat].max(
            jnp.where(eligible, ids + 1, 0)
        )
        discovered = eligible & (claim[flat] == ids + 1)
        redundant  = moved & ~discovered
        travelled = jnp.linalg.norm(new_pos - state.robot_positions, axis=-1)
        nominal_step = max(self.v_max * self.dt, 1e-6)
        redundant_travel = redundant * travelled / nominal_step
        if self.reward_mode == 'local_coverage_v1':
            old_cols, old_rows = self._pos_to_cell(state.robot_positions)
            entered = (cols != old_cols) | (rows != old_rows)
            # Crossing within the same cell is free. Simultaneous discovery
            # losers do not pay a revisit penalty for a previously unseen cell.
            redundant_travel = (moved & entered & already & coverable).astype(jnp.float32)

        new_grid = prev_grid.at[rows, cols].max(
            jnp.where(moved & coverable, 1.0, 0.0)
        )

        covered   = jnp.sum(new_grid[None, :, :] * self.room_masks[state.map_id], axis=(1, 2))
        ratio     = covered / jnp.maximum(self.room_totals[state.map_id], 1.0)
        newly     = (~state.room_completed) & (self.room_totals[state.map_id] > 0) \
                    & (ratio >= self.room_completion_threshold)
        room_completed = state.room_completed | newly

        free_total = self.free_totals[state.map_id]
        # Make late discoveries increasingly valuable.  Using the coverage
        # before this step gives every simultaneous discovery the same weight
        # and keeps the first cell worth exactly `alpha`.
        coverage_before = jnp.sum(prev_grid) / jnp.maximum(free_total, 1.0)
        discovery_multiplier = 1.0 + self.coverage_reward_growth * coverage_before
        complete   = jnp.sum(new_grid) >= free_total - 0.5
        team_bonus = (self.room_completion_bonus * jnp.sum(newly)
                      + self.completion_bonus * complete)

        if self.reward_mode == 'local_coverage_v1':
            team_bonus = self.completion_bonus * complete

        dist     = jnp.sqrt(self._pairwise_sq_dist(new_pos))
        pen      = self.psi * (1.0 - dist / self._safe_dist) * (dist < self._safe_dist)
        prox_pen = jnp.sum(pen, axis=1) * alive

        v_norm = v_cmds / self.v_max
        omega_norm = omega_cmds / self.omega_max
        prev_v_norm = state.robot_velocities[:, 0] / self.v_max
        prev_omega_norm = state.robot_velocities[:, 1] / self.omega_max
        control_cost = (
            self.velocity_cost * v_norm ** 2
            + self.angular_cost * omega_norm ** 2
            + self.action_smoothness_cost
            * ((v_norm - prev_v_norm) ** 2
               + (omega_norm - prev_omega_norm) ** 2)
        )

        if self.reward_mode == 'progress':
            rewards = self._progress_reward(
                state, new_pos, prev_grid, discovered, discovery_multiplier,
                wall_hit, robot_hit, robot_hit_human, complete,
            )
        else:
            rewards = jnp.where(
                alive,
                self.alpha * discovery_multiplier * discovered
                - self.beta * redundant_travel
                - self.tau
                - self.wall_kappa * wall_hit
                - self.kappa * robot_hit
                - self.human_kappa * robot_hit_human
                - prox_pen
                + team_bonus
                - control_cost,
                0.0,
            ).astype(jnp.float32)

        step_count = state.step_count + 1
        truncated  = step_count >= self.max_steps
        terminated = complete | (jnp.any(collided) if self.terminate_on_collision else jnp.bool_(False))

        next_state = state.replace(
            robot_positions  = new_pos,
            robot_headings   = new_hdg,
            robot_velocities = new_vel,
            human_positions  = new_human_pos,
            human_headings   = final_h_headings,
            human_dists      = final_h_dists,
            coverage_grid    = new_grid,
            room_completed   = room_completed,
            robot_alive      = alive_next,
            step_count       = step_count,
            key              = key,
            wall_hits        = wall_hit.astype(jnp.float32),
            robot_hits       = robot_hit.astype(jnp.float32),
            human_hits       = robot_hit_human.astype(jnp.float32),
        )
        return next_state, rewards, terminated, truncated, moved & coverable

    def _progress_reward(
        self, state: EnvState, new_pos: jax.Array, prev_grid: jax.Array,
        discovered: jax.Array, discovery_multiplier: jax.Array,
        wall_hit: jax.Array, robot_hit: jax.Array, robot_hit_human: jax.Array,
        complete: jax.Array,
    ) -> jax.Array:
        """Per-robot reward for reward_mode='progress', shape (N,).

        discovery   alpha * (1 + growth * coverage) per newly covered cell.
        progress    progress_weight per cell of geodesic approach to the
                    nearest cell the robot believes uncovered. The belief is
                    its own memory (own coverage + what teammates shared in
                    comm range), so the implicit target uses only knowledge
                    the robot has. Transit over covered cells towards
                    remaining work is therefore paid, not punished.
        loiter      loiter_cost per step without discovery, scaled by the
                    missing fraction of full-speed progress: parking,
                    spinning and wandering over covered cells cost the full
                    amount, driving straight at v_max towards work costs 0.
        spread      spread_weight * max(0, 1 - d / spread_radius) per teammate,
                    a constant pressure to keep the team apart.
        completion  shared bonus when the map is covered, larger when early.

        Both potentials of a step use the same pre-step field, so covering the
        last cell of an area does not look like a sudden loss of progress, and
        the progress term telescopes: closed loops earn exactly zero.
        """
        alive = state.robot_alive
        free = self.free_masks[state.map_id] > 0.5                       # (H, W)
        if self.use_memory:
            believed = state.mem_covered > 0.5                            # (N, H, W)
        else:
            believed = jnp.broadcast_to(prev_grid > 0.5, (self.num_robots, *free.shape))
        targets = free[None] & ~believed
        dist = self._geodesic_distance(targets, jnp.broadcast_to(free, targets.shape))

        before = self._work_distance(dist, state.robot_positions)
        after = self._work_distance(dist, new_pos)
        progress = jnp.where(before < 0.5 * _FAR, before - after, 0.0)

        idle = jnp.clip(1.0 - progress / self._full_progress, 0.0, 1.0)
        loiter = self.loiter_cost * idle * ~discovered

        d = jnp.sqrt(self._pairwise_sq_dist(new_pos))
        crowd = jnp.maximum(0.0, 1.0 - d / self.spread_radius) * alive[None, :]
        spread = self.spread_weight * jnp.sum(crowd, axis=1)

        time_left = jnp.maximum(1.0 - (state.step_count + 1) / self.max_steps, 0.0)
        completion = (self.completion_bonus * complete
                      * (1.0 + self.completion_time_bonus * time_left))

        reward = (
            self.alpha * discovery_multiplier * discovered
            + self.progress_weight * progress
            - loiter
            - spread
            - self.tau
            - self.wall_kappa * wall_hit
            - self.kappa * robot_hit
            - self.human_kappa * robot_hit_human
            + completion
        )
        return jnp.where(alive, reward, 0.0).astype(jnp.float32)

    def set_ghost_robot_prob(self, state: EnvState, prob: jax.Array) -> EnvState:
        return state.replace(ghost_robot_prob=jnp.clip(prob, 0.0, 1.0))

    def _get_memory_obs(self, state: EnvState) -> jax.Array:
        """Per-robot observation for obs_mode='memory_comm', shape (N, obs_dim).

        [ x/W, y/H, cos(h), sin(h)                      pose, 4
          v/v_max, omega/omega_max                      velocity, 2
          id                                            own id in 1..N, 1
          (id_j, dx_j, dy_j) x comm_slots               nearest teammates in comm
                                                        range, world-frame metres;
                                                        id 0 marks an empty slot
          lidar                                         n_rays, normalised
          occupied | covered | known                    3 x S x S memory crop,
                                                        grid-aligned, rows = +y ]
        Cells outside the map read as known and occupied.
        """
        n = self.num_robots
        ids = self._robot_ids
        pos = state.robot_positions
        alive = state.robot_alive

        pose = jnp.stack([
            pos[:, 0] / self.map_layout.width,
            pos[:, 1] / self.map_layout.height,
            jnp.cos(state.robot_headings),
            jnp.sin(state.robot_headings),
        ], axis=-1)
        vel = state.robot_velocities / jnp.array([self.v_max, self.omega_max])
        parts = [pose, vel, (ids + 1).astype(jnp.float32)[:, None]]

        k = min(self.comm_slots, n - 1)
        if k > 0:
            d2 = self._pairwise_sq_dist(pos)
            in_range = (d2 <= self.comm_radius ** 2) & alive[:, None] & alive[None, :]
            neg_d2, idx = jax.lax.top_k(-jnp.where(in_range, d2, _BIG), k)
            valid = (-neg_d2 < _BIG).astype(jnp.float32)[..., None]
            rel = pos[idx] - pos[:, None, :]                          # (N, k, 2)
            slot = jnp.concatenate([(idx + 1).astype(jnp.float32)[..., None], rel], axis=-1)
            parts.append((slot * valid).reshape(n, 3 * k))
        if self.comm_slots > k:
            parts.append(jnp.zeros((n, 3 * (self.comm_slots - k)), jnp.float32))

        parts.append(state.lidar)

        cols, rows = self._pos_to_cell(pos)
        r = rows[:, None, None] + self._crop_offsets[None, :, None]  # (N, S, 1)
        c = cols[:, None, None] + self._crop_offsets[None, None, :]  # (N, 1, S)
        inside = (r >= 0) & (r < self.grid_h) & (c >= 0) & (c < self.grid_w)
        r = jnp.clip(r, 0, self.grid_h - 1)
        c = jnp.clip(c, 0, self.grid_w - 1)
        i = ids[:, None, None]
        known = state.mem_known[i, r, c]
        covered = jnp.where(inside, state.mem_covered[i, r, c], 0.0)
        occupied = jnp.where(inside, known * self.wall_grids[state.map_id][r, c], 1.0)
        known = jnp.where(inside, known, 1.0)
        crop = jnp.stack([occupied, covered, known], axis=1)         # (N, 3, S, S)
        parts.append(crop.reshape(n, self.patch_dim))

        return jnp.concatenate(parts, axis=1).astype(jnp.float32)

    def get_obs(self, state: EnvState) -> jax.Array:
        if self.use_memory:
            return self._get_memory_obs(state)
        n     = self.num_robots

        parts = [state.robot_velocities]

        rel = state.robot_positions[None, :, :] - state.robot_positions[:, None, :]
        if self._k_eff > 0:
            d2 = self._pairwise_sq_dist(state.robot_positions)
            neg_d2, idx = jax.lax.top_k(-d2, self._k_eff)
            near = jnp.take_along_axis(rel, idx[:, :, None], axis=1)
            visible = (-neg_d2) <= self.sensing_radius ** 2
            near = jnp.where(visible[:, :, None], near, 0.0)
            parts.append(near.reshape(n, self._k_eff * 2))
        pad_k = self.k_teammates - self._k_eff
        if pad_k > 0:
            parts.append(jnp.zeros((n, pad_k * 2), jnp.float32))

        parts.append(self._cast_lidar_all(state))

        if self.use_local_coverage_obs:
            offsets = self._local_patch_offsets
            c = jnp.cos(state.robot_headings)[:, None, None]
            s = jnp.sin(state.robot_headings)[:, None, None]
            local_x = offsets[None, :, :, 0]
            local_y = offsets[None, :, :, 1]
            sample_x = state.robot_positions[:, None, None, 0] + c * local_x - s * local_y
            sample_y = state.robot_positions[:, None, None, 1] + s * local_x + c * local_y
            cols = jnp.floor(sample_x / self.cell_size).astype(jnp.int32)
            rows = jnp.floor(sample_y / self.cell_size).astype(jnp.int32)
            inside = ((cols >= 0) & (cols < self.grid_w)
                      & (rows >= 0) & (rows < self.grid_h))
            cols = jnp.clip(cols, 0, self.grid_w - 1)
            rows = jnp.clip(rows, 0, self.grid_h - 1)
            source = jnp.maximum(state.coverage_grid, self.wall_grids[state.map_id])
            patch = jnp.where(inside, source[rows, cols], 1.0)
            parts.append(patch.reshape(n, self.patch_dim))

        return jnp.concatenate(parts, axis=1).astype(jnp.float32)

    def get_global_state(self, state: EnvState) -> GlobalState:
        cols, rows = self._pos_to_cell(state.robot_positions)
        occupancy = jnp.zeros(
            (self.num_robots, self.grid_h, self.grid_w), jnp.float32
        ).at[self._robot_ids, rows, cols].set(state.robot_alive.astype(jnp.float32))

        kinematics = jnp.stack([
            state.robot_positions[:, 0] / self.map_layout.width,
            state.robot_positions[:, 1] / self.map_layout.height,
            jnp.cos(state.robot_headings),
            jnp.sin(state.robot_headings),
            state.robot_velocities[:, 0] / self.v_max,
            state.robot_velocities[:, 1] / self.omega_max,
        ], axis=-1)

        human_norm = state.human_positions / jnp.array([self.map_layout.width, self.map_layout.height])

        return GlobalState(
            coverage=state.coverage_grid, 
            occupancy=occupancy, 
            kinematics=kinematics, 
            human_positions=human_norm,
            map_id=state.map_id
        )

    def critic_inputs(self, gs: GlobalState) -> tuple[jax.Array, jax.Array]:
        occ  = gs.occupancy
        me   = occ[..., :, None, :, :]
        rest = (occ.sum(axis=-3, keepdims=True) - occ)[..., :, None, :, :]
        cov  = jnp.broadcast_to(gs.coverage[..., None, None, :, :], me.shape)
        
        # Broadcast the specific map's wall grid along all batch dimensions
        wall = self.wall_grids[gs.map_id]                           # (..., H, W)
        wall = jnp.expand_dims(wall, axis=(-3, -4))                 # (..., 1, 1, H, W)
        wall = jnp.broadcast_to(wall, me.shape)                     # (..., N, 1, H, W)
        
        grid = jnp.concatenate([wall, cov, me, rest], axis=-3)

        joint = gs.kinematics.reshape(*gs.kinematics.shape[:-2], 1, -1)
        
        flat_humans = gs.human_positions.reshape(*gs.human_positions.shape[:-2], 1, -1)

        joint_ext = jnp.concatenate([joint, flat_humans], axis=-1)

        vec = jnp.concatenate(
            [gs.kinematics, jnp.broadcast_to(joint_ext, (*gs.kinematics.shape[:-1], joint_ext.shape[-1]))],
            axis=-1,
        )
        return grid, vec

    def get_info(self, state: EnvState) -> dict:
        covered = jnp.sum(state.coverage_grid)
        room_cov = jnp.sum(state.coverage_grid[None, :, :] * self.room_masks[state.map_id], axis=(1, 2))
        free_total = self.free_totals[state.map_id]
        
        info = {
            'coverage_ratio':       covered / free_total,
            'covered_cells':        covered,
            'total_cells':          jnp.float32(free_total),
            'step':                 state.step_count,
            'robots_alive':         state.robot_alive,
            'num_robots_alive':     jnp.sum(state.robot_alive),
            'wall_collision_rate':  jnp.mean(state.wall_hits),
            'robot_collision_rate': jnp.mean(state.robot_hits),
            'human_collision_rate': jnp.mean(state.human_hits),
            'complete':             (covered >= free_total - 0.5).astype(jnp.float32),
            'timeout':              (state.step_count >= self.max_steps).astype(jnp.float32),
        }
        for ri in range(self.num_rooms):
            info[f'room_{ri}_ratio'] = room_cov[ri] / jnp.maximum(self.room_totals[state.map_id, ri], 1.0)
        return info
