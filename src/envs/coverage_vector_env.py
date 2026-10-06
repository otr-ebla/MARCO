"""
JAX multi-robot indoor coverage environment.

The environment is a pure function of an immutable `EnvState` pytree, so
`reset` / `step` / `get_obs` can be jitted, vmapped over parallel environments,
and scanned over time without leaving the accelerator.
"""

from __future__ import annotations

import warnings
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .recovery import (astar, command_safety_flags, dwa, local_route_free,
                       run_if, velocity_sequence)
from .map_layouts import ProceduralMapLayout, create_map_bank

_TWO_PI = 2.0 * np.pi
_BIG = 1.0e9
_FAR = 1.0e6   # geodesic distance of a cell that cannot reach any target

# Belief-map cell states, ordered so that merging two maps is a cell-wise max:
# an unseen cell never overrides a seen one and covered overrides uncovered.
UNKNOWN, OCCUPIED, FREE, COVERED = 0, 1, 2, 3

# Training defaults; checkpoints persist this complete reward specification.
# 'local_coverage_v1' is kept only so older checkpoints still evaluate.
E2E_REWARD_DEFAULTS = {
    'reward_mode': 'progress',
    'alpha': 10.0,                  # per newly covered cell
    'coverage_reward_growth': 2.0,  # late cells are worth up to 3x alpha
    'axis_alignment_bonus': 0.0,   # compatibility with earlier checkpoints
    'axis_alignment_cost': 0.5,    # max cost at 45 degrees and full-speed travel
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
    'sequential_bonus': 2.0,        # adjacent discoveries, sequential mode only
    'straight_bonus': 1.0,          # continuing the same sweep direction
    'boustrophedon_bonus': 0.0,     # reversal after shifting to an adjacent lane
    'revisit_cost': 1.0,            # multiplied by consecutive redundant cell entries
    'revisit_end_fraction': 1.0,   # 1 preserves legacy constant cost
    'revisit_decay_power': 2.0,
    'revisit_streak_cap': 0,       # 0 preserves legacy unbounded streaks
    'sweep_straight_bonus': 3.0,    # new cell straight ahead in the lane, any reward mode
    'sweep_turn_bonus': 3.0,        # lane shift once the cell ahead is blocked or covered
    'sweep_break_cost': 5.0,        # leaving the lane head while the cell ahead is open
    'fallback_cost': 10.0,         # once on activation, charged to the triggering action
    'los_spread_weight': 0.0,      # per step and teammate in line of sight at zero distance
    'los_spread_radius': 5.0,      # metres; = max_lidar_range, so the lidar sees who it pays for
    'los_spread_decay_steps': 1500, # episode steps over which the weight decays
    'los_spread_end_fraction': 0.1, # weight fraction left after the decay
    'discovery_streak_bonus': 0.0, # bonus x streak length per consecutive new cell
    'discovery_streak_cap': 10,    # streak length at which the bonus stops growing
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
    # Decentralised per-robot memory (also maintained for legacy-mode recovery).
    # Per-robot belief map: a cell dictionary {(row, col): state} stored as a
    # fixed-shape int8 grid so it can be jitted and vmapped. UNKNOWN marks a
    # key not yet in the dictionary; OCCUPIED / FREE / COVERED are written
    # when a cell is first seen (lidar) or covered (footprint), never erased.
    mem_state:        jax.Array   # (N, H, W) int8 — UNKNOWN / OCCUPIED / FREE / COVERED
    lidar:            jax.Array   # (N, R)    float32 — latest normalised scan
    last_discovery:   jax.Array   # (N, 2) int32 — last new (col, row), -1 at reset
    sweep_direction:  jax.Array   # (N, 2) int32 — last adjacent discovery direction
    obs_history:      jax.Array   # (N, K, frame_dim), oldest first; empty for K=1
    previous_visit:   jax.Array   # (N,2), cell occupied before the latest cell entry
    visit_counts:     jax.Array   # (N,H,W), accepted cell entries by each robot
    global_coverage_history: jax.Array  # (K,H,W), oldest first
    global_occupancy_history: jax.Array # (K,N,H,W)
    global_visit_history: jax.Array     # (K,N,H,W)
    global_kinematics_history: jax.Array # (K,N,6)
    global_humans_history: jax.Array    # (K,M,2)
    global_context_history: jax.Array   # (K,N,12) when critic_context
    global_previous_visit_history: jax.Array # (K,N,2)
    global_history_valid: jax.Array     # (K,), zero for padded frames
    sweep_run_length: jax.Array   # (N,) consecutive new-cell edges in one direction
    lane_return:      jax.Array   # (N, 2) expected reversal after a one-cell lane shift
    last_visit:       jax.Array   # (N, 2) last accepted cell, independent of discovery
    cell_entries:     jax.Array   # () total team visits, including first visits
    revisit_streak:   jax.Array   # (N,) consecutive covered-cell entries
    discovery_streak: jax.Array   # (N,) new cells since the last covered-cell entry
    no_progress_steps: jax.Array # (N,) physics steps without a new cell
    fallback_active:  jax.Array   # (N,) recovery owns the next control step
    fallback_goal:    jax.Array   # (N,) flat cell index, -1 when inactive
    fallback_used:    jax.Array   # (N,) last action came from either recovery controller
    fallback_activated: jax.Array # (N,) activation event on the last step
    fallback_count:   jax.Array   # (N,) activation count this episode
    fallback_stagnation: jax.Array # (N,) steps without improving remaining route distance
    fallback_best_distance: jax.Array # (N,) best remaining route distance for current goal
    fallback_observed_cells: jax.Array # (N,) known-cell count for the progress baseline
    fallback_sequence: jax.Array  # (N,) sequence mode after DWA stalls
    fallback_commands: jax.Array # (N, K, 2) queued physical (v, omega) commands
    fallback_command_index: jax.Array # (N,) next queued command
    fallback_command_count: jax.Array # (N,) valid commands in queue
    fallback_sequence_used: jax.Array # (N,) last command was from the sequence
    fallback_safety_override: jax.Array # (N,) live scan interrupted the sequence
    fallback_rejection_flags: jax.Array # (N,) safety bitmask; 16 = empty queue


@struct.dataclass
class GlobalState:
    """Centralised critic state, kept as small spatial tensors."""
    coverage:        jax.Array   # (H, W)     float32
    occupancy:       jax.Array   # (N, H, W)  float32, one-hot per robot
    kinematics:      jax.Array   # (N, 6)     float32, normalised
    human_positions: jax.Array   # (M, 2)     float32, normalised
    map_id:          jax.Array   # ()         int32
    task_context:    jax.Array   # (N,D), reward/control history and episode time
    visit_counts:    jax.Array   # (N,H,W), cumulative own cell entries
    previous_visit:  jax.Array   # (N,2), prior cell for every robot
    coverage_history: jax.Array
    occupancy_history: jax.Array
    visit_history: jax.Array
    kinematics_history: jax.Array
    humans_history: jax.Array
    context_history: jax.Array
    previous_visit_history: jax.Array
    crops:           jax.Array   # (N, crop_dim) actor crops, or (N, 0)
    history_valid: jax.Array


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
        self.wall_cells = int(cfg.get('wall_cells', 1))
        if self.wall_cells < 0 or self.wall_cells != cfg.get('wall_cells', 1):
            raise ValueError('wall_cells must be a non-negative integer')
        self.sensing_radius  = float(cfg.get('sensing_radius',  5.0))
        self.robot_radius    = float(cfg.get('robot_radius',    0.20))
        # A neighbouring cell counts as covered once the disk enters it by
        # this fraction of the radius; the centre cell is always covered.
        self.coverage_overlap = float(cfg.get('coverage_overlap', 0.8))
        if not 0.0 <= self.coverage_overlap <= 1.0:
            raise ValueError('coverage_overlap must be in [0, 1]')
        self.dt              = float(cfg.get('dt',              0.1))
        self.max_steps       = int(cfg.get('max_steps',       500))
        self.v_max           = float(cfg.get('v_max',           1.0))
        self.omega_max       = float(cfg.get('omega_max',       1.0))
        self.terminate_on_collision = bool(cfg.get('terminate_on_collision', False))
        self.reward_mode = cfg.get('reward_mode', 'legacy')
        if self.reward_mode not in ('legacy', 'local_coverage_v1', 'progress', 'sequential'):
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
        if cfg.get('goal_obs', False):
            raise ValueError('Waypoint observations have been removed; use history_cell=previous_visit')
        self.history_cell = cfg.get('history_cell', 'last_discovery')
        if self.history_cell not in ('last_discovery', 'previous_visit'):
            raise ValueError('history_cell must be last_discovery or previous_visit')
        self.critic_context = bool(cfg.get('critic_context', False))
        self.fallback_enabled = bool(cfg.get('fallback_enabled', True))
        self.track_memory = self.use_memory or self.fallback_enabled
        if self.track_memory and int(cfg.get('wall_cells', 1)) == 0:
            # The belief map stores cells only: a thin wall lying between two
            # free cells is invisible to it, so planning would cross it.
            warnings.warn('wall_cells=0: thin walls are not represented in the belief map')
        self.fallback_revisit_threshold = int(cfg.get('fallback_revisit_threshold', 7))
        self.fallback_stall_steps = int(cfg.get('fallback_stall_steps', 70))
        self.fallback_linear_accel = float(cfg.get('fallback_linear_accel', 1.0))
        self.fallback_angular_accel = float(cfg.get('fallback_angular_accel', 2.0))
        self.fallback_dwa_steps = int(cfg.get('fallback_dwa_steps', 12))
        self.fallback_dwa_stall_steps = int(cfg.get('fallback_dwa_stall_steps', 30))
        self.fallback_sequence_steps = int(cfg.get('fallback_sequence_steps', 64))
        self.fallback_sequence_speed = float(cfg.get('fallback_sequence_speed', min(.4, self.v_max)))
        if not 0 < self.fallback_sequence_speed <= self.v_max:
            raise ValueError('fallback_sequence_speed must be in (0, v_max]')
        if min(self.fallback_revisit_threshold, self.fallback_stall_steps,
               self.fallback_linear_accel, self.fallback_angular_accel,
               self.fallback_dwa_steps, self.fallback_dwa_stall_steps,
               self.fallback_sequence_steps) <= 0:
            raise ValueError('Fallback thresholds, accelerations and horizon must be positive')
        # Past-cell history (previous visit, sweep direction) in the actor vector.
        self.use_full_memory = bool(cfg.get('use_full_memory', False))
        # The whole personal map as actor input. Off: the map lives in the robot
        # and reaches the policy only through the local crop. On only to
        # evaluate checkpoints trained with it.
        self.memory_map_obs = bool(cfg.get('memory_map_obs', False))
        # Preferred lane direction and whether its next cell is open work;
        # memory_comm only, the legacy observation layout is left unchanged.
        self.sweep_obs = bool(cfg.get('sweep_obs', False)) and self.use_memory
        # One-cell ring around the actor crop summarising every cell beyond it
        # in each direction, plus an extent channel; see _add_crop_summary.
        self.crop_summary = bool(cfg.get('crop_summary', False)) and (
            self.use_memory or self.use_local_coverage_obs)
        if self.use_local_coverage_obs and not self.track_memory:
            raise ValueError('The actor crop reads robot memory: needs memory_comm or fallback_enabled')
        # 'directional': [obstacle, covered, robots] S x S crop inside a ring
        # holding each channel's mean beyond each side; see
        # _directional_crops. 'memory': [occupied, covered, known], optionally
        # with crop_summary (checkpoints trained before 'directional').
        self.crop_mode = cfg.get('crop_mode', 'memory')
        if self.crop_mode not in ('memory', 'directional'):
            raise ValueError(f'Unknown crop_mode: {self.crop_mode}')
        if self.crop_mode == 'directional' and (not self.use_memory or self.crop_summary):
            raise ValueError("crop_mode='directional' needs memory_comm and replaces crop_summary")
        # Bit-packable crop: only binary values (see pretrain_bc's replay buffer).
        self.crop_binary = not self.crop_summary and self.crop_mode == 'memory'
        # CTDE: the critic also reads every robot's actor crop (see critic_inputs).
        self.critic_crops = bool(cfg.get('critic_crops', False)) and (
            self.use_memory or self.use_local_coverage_obs)
        # Actor: covered fraction of the free cells in the robot's own memory.
        # Critic: true global coverage fraction. Both change the network inputs.
        self.known_coverage_obs = bool(cfg.get('known_coverage_obs', False))
        if self.known_coverage_obs and not self.track_memory:
            raise ValueError('known_coverage_obs reads robot memory: needs memory_comm or fallback_enabled')
        self.critic_coverage = bool(cfg.get('critic_coverage', False))
        self.observation_stack = int(cfg.get('observation_stack', 1))
        if self.observation_stack < 1:
            raise ValueError('observation_stack must be positive')
        self.critic_stack = int(cfg.get('critic_stack', 1))
        if self.critic_stack < 1:
            raise ValueError('critic_stack must be positive')
        if (self.use_full_memory or self.memory_map_obs or self.reward_mode == 'sequential') \
                and not self.use_memory:
            raise ValueError('Full memory and sequential reward require obs_mode=memory_comm')
        if self.memory_map_obs and not self.use_full_memory:
            raise ValueError('memory_map_obs requires use_full_memory')
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
        self.axis_alignment_cost = float(cfg.get('axis_alignment_cost', 0.0))
        self.axis_alignment_bonus = float(cfg.get('axis_alignment_bonus', 0.0))
        self.progress_weight = float(cfg.get('progress_weight', 2.0))
        self.loiter_cost = float(cfg.get('loiter_cost', 1.0))
        self.spread_weight = float(cfg.get('spread_weight', 0.2))
        self.spread_radius = float(cfg.get('spread_radius', 2.5))
        self.completion_time_bonus = float(cfg.get('completion_time_bonus', 1.0))
        self.sequential_bonus = float(cfg.get('sequential_bonus', 2.0))
        self.straight_bonus = float(cfg.get('straight_bonus', 1.0))
        self.boustrophedon_bonus = float(cfg.get('boustrophedon_bonus', 0.0))
        self.revisit_cost = float(cfg.get('revisit_cost', 1.0))
        self.revisit_end_fraction = float(cfg.get('revisit_end_fraction', 1.0))
        self.revisit_decay_power = float(cfg.get('revisit_decay_power', 2.0))
        self.revisit_streak_cap = int(cfg.get('revisit_streak_cap', 0))
        if not 0 <= self.revisit_end_fraction <= 1 or self.revisit_decay_power <= 0 or self.revisit_streak_cap < 0:
            raise ValueError('Invalid revisit decay fraction, power or streak cap')
        self.sweep_straight_bonus = float(cfg.get('sweep_straight_bonus', 0.0))
        self.sweep_turn_bonus = float(cfg.get('sweep_turn_bonus', 0.0))
        self.sweep_break_cost = float(cfg.get('sweep_break_cost', 0.0))
        self.fallback_cost = float(cfg.get('fallback_cost', 10.0))
        self.los_spread_weight = float(cfg.get('los_spread_weight', 0.0))
        self.los_spread_radius = float(cfg.get('los_spread_radius', 5.0))
        self.los_spread_decay_steps = int(cfg.get('los_spread_decay_steps', 1500))
        self.los_spread_end_fraction = float(cfg.get('los_spread_end_fraction', 0.1))
        self.discovery_streak_bonus = float(cfg.get('discovery_streak_bonus', 0.0))
        self.discovery_streak_cap = int(cfg.get('discovery_streak_cap', 10))
        if self.spread_radius <= 0.0:
            raise ValueError('spread_radius must be positive')
        if (self.los_spread_radius <= 0.0 or self.los_spread_decay_steps <= 0
                or not 0.0 <= self.los_spread_end_fraction <= 1.0
                or self.discovery_streak_cap <= 0):
            raise ValueError('Invalid line-of-sight spread or discovery streak parameters')
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
            'axis_alignment_cost': self.axis_alignment_cost,
            'axis_alignment_bonus': self.axis_alignment_bonus,
            'progress_weight': self.progress_weight,
            'loiter_cost': self.loiter_cost,
            'spread_weight': self.spread_weight,
            'completion_time_bonus': self.completion_time_bonus,
            'sequential_bonus': self.sequential_bonus,
            'straight_bonus': self.straight_bonus,
            'boustrophedon_bonus': self.boustrophedon_bonus,
            'revisit_cost': self.revisit_cost,
            'sweep_straight_bonus': self.sweep_straight_bonus,
            'sweep_turn_bonus': self.sweep_turn_bonus,
            'sweep_break_cost': self.sweep_break_cost,
            'fallback_cost': self.fallback_cost,
            'los_spread_weight': self.los_spread_weight,
            'discovery_streak_bonus': self.discovery_streak_bonus,
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
                                   robot_radius=self.robot_radius, wall_cells=self.wall_cells)
        
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
        # Label a seen cell enters the belief map with: (M, H*W) int8.
        self._cell_labels = jnp.where(self._free_flat > 0.0, FREE, OCCUPIED).astype(jnp.int8)
        
        if np.any(self.free_totals < self.num_robots):
            raise RuntimeError("Free space is too small in one of the procedural maps.")

        # -- Room definitions (Simplified for procedural maps) --
        # We treat the entire connected free space as 1 single room per map.
        self.num_rooms = 1
        self.room_masks = jnp.expand_dims(self.free_masks, axis=1)   # (M, 1, H, W)
        self.room_totals = jnp.sum(self.room_masks, axis=(2, 3))     # (M, 1)

        # -- Derived dims --
        ringed = self.crop_summary or self.crop_mode == 'directional'
        crop_side = self.local_coverage_size + (2 if ringed else 0)
        # Summary adds an extent channel, and a known channel to the legacy patch.
        crop_extra = (1 if self.use_memory else 2) if self.crop_summary else 0
        if self.use_memory:
            # pose (x, y, cos, sin) + (v, omega) + own id + (id, dx, dy) per slot
            self.obs_vec_dim = 4 + 2 + 1 + 3 * self.comm_slots
            # [occupied, covered, known] or [obstacle, covered, robots] crop
            self.crop_dim    = (3 + crop_extra) * crop_side ** 2
            self.crop_shape  = (3 + crop_extra, crop_side, crop_side)
        else:
            self.obs_vec_dim = 2 + self.k_teammates * 2
            self.crop_dim    = (
                (1 + crop_extra) * crop_side ** 2 if self.use_local_coverage_obs else 0
            )
            self.crop_shape  = ((1 + crop_extra, crop_side, crop_side)
                                if self.use_local_coverage_obs else ())
        self.patch_dim = self.crop_dim
        self.norm_dim      = self.obs_vec_dim + self.n_rays
        # Full personal map (occupied, covered, known, self, visible teammates),
        # only with memory_map_obs; see __init__.
        self.memory_map_shape = (5, self.grid_h, self.grid_w) if self.memory_map_obs else ()
        if self.use_full_memory:
            self.obs_vec_dim += 5  # past cell: valid, dx, dy, historical sweep dx/dy
            self.norm_dim = self.obs_vec_dim + self.n_rays
        if self.memory_map_obs:
            self.patch_dim += 5 * self.num_cells
        if self.sweep_obs:
            self.obs_vec_dim += 3  # preferred lane dx/dy, next lane cell open
            self.norm_dim = self.obs_vec_dim + self.n_rays
        if self.known_coverage_obs:
            self.obs_vec_dim += 1  # covered / known free cells in own memory
            self.norm_dim = self.obs_vec_dim + self.n_rays
        self.frame_norm_dim = self.norm_dim
        self.frame_dim = self.frame_norm_dim + self.patch_dim
        self.norm_dim *= self.observation_stack
        self.obs_dim = self.frame_dim * self.observation_stack
        self.action_dim    = 2
        self._critic_frame_channels = 4 + (self.num_robots if self.critic_stack > 1 else 0)
        self._critic_frame_vec_dim = (6 + 6 * self.num_robots + 2 * self.num_humans
                                      + (12 if self.critic_context else 0)
                                      + (3 * self.num_robots if self.critic_stack > 1 else 0))
        self.critic_channels = self.critic_stack * self._critic_frame_channels
        self.critic_vec_dim = self.critic_stack * self._critic_frame_vec_dim
        if self.critic_crops:
            self.critic_vec_dim += (1 + self.num_robots) * self.crop_dim  # own + every robot's
        if self.critic_coverage:
            self.critic_vec_dim += 1  # true global coverage fraction
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
        if self.crop_summary:
            self._crop_ring = self._build_crop_ring()
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
        self._spawn_counts = jnp.asarray([len(c) for c in cands_list], jnp.int32)
        if min(len(c) for c in cands_list) < self.num_robots + self.num_humans:
            raise ValueError('Not enough distinct free cells to spawn all robots and humans')
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
        span = int(np.ceil(self.robot_radius / self.cell_size))
        offsets = np.arange(-span, span + 1, dtype=np.int32)
        self._footprint_dc = jnp.asarray(np.tile(offsets, offsets.size))
        self._footprint_dr = jnp.asarray(np.repeat(offsets, offsets.size))

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

    def _footprint_cells(self, pos: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Cells the robot disk covers: its centre cell, plus every other cell
        it enters by at least `coverage_overlap` of its radius.

        Returns (N, K) clipped cols and rows over the block around the centre
        cell, whether the disk overlaps each one, and its squared distance
        from the centre (0 for the centre cell).
        """
        col, row = self._pos_to_cell(pos)
        cols = col[:, None] + self._footprint_dc[None, :]
        rows = row[:, None] + self._footprint_dr[None, :]
        x0 = cols.astype(jnp.float32) * self.cell_size
        y0 = rows.astype(jnp.float32) * self.cell_size
        dx = pos[:, 0:1] - jnp.clip(pos[:, 0:1], x0, x0 + self.cell_size)
        dy = pos[:, 1:2] - jnp.clip(pos[:, 1:2], y0, y0 + self.cell_size)
        d2 = dx * dx + dy * dy
        inside = (cols >= 0) & (cols < self.grid_w) & (rows >= 0) & (rows < self.grid_h)
        reach = (1.0 - self.coverage_overlap) * self.robot_radius
        centre = (self._footprint_dc == 0) & (self._footprint_dr == 0)
        touch = inside & ((d2 < reach ** 2) | centre[None, :])
        return (jnp.clip(cols, 0, self.grid_w - 1), jnp.clip(rows, 0, self.grid_h - 1),
                touch, d2)

    def _geodesic_distance(self, targets: jax.Array, free: jax.Array) -> jax.Array:
        """Multi-source 4-connected distance, in cells, from every free cell to
        the nearest target cell. Cells that reach no target hold `_FAR`.

        targets, free : (..., H, W) bool
        """
        far = jnp.float32(_FAR)
        lead = [(0, 0)] * (free.ndim - 2)

        def relax(d):
            pad = jnp.pad(d, lead + [(1, 1), (1, 1)], constant_values=far)
            neighbors = jnp.stack([pad[..., :-2, 1:-1], pad[..., 2:, 1:-1],
                                   pad[..., 1:-1, :-2], pad[..., 1:-1, 2:]], axis=-1)
            nb = jnp.min(neighbors, axis=-1)
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
        # Exact constant-twist integration without subtracting nearly equal
        # sines. The old v/omega form could produce zero motion at tiny omega.
        half_turn = .5 * omega * self.dt
        travel = v * self.dt * jnp.sinc(half_turn / jnp.pi)
        middle = heading + half_turn
        new_pos = pos + travel[:, None] * jnp.stack([jnp.cos(middle), jnp.sin(middle)], axis=-1)
        return new_pos, jnp.mod(heading + 2 * half_turn, _TWO_PI)

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

    def belief_map(self, map_id, known, covered=None) -> jax.Array:
        """(..., H, W) int8 belief map from boolean seen / covered masks.

        Seen cells take their label (OCCUPIED or FREE); covered free cells
        become COVERED and count as seen. For tests and tooling.
        """
        labels = self._cell_labels[map_id].reshape(self.grid_h, self.grid_w)
        seen = jnp.asarray(known) > 0.5
        mem = jnp.where(seen, labels, UNKNOWN)
        if covered is not None:
            mem = jnp.where((jnp.asarray(covered) > 0.5) & (labels == FREE), COVERED, mem)
        return mem.astype(jnp.int8)

    @staticmethod
    def _known(state: EnvState) -> jax.Array:
        """(N, H, W) bool: the cell is in robot i's map."""
        return state.mem_state != UNKNOWN

    @staticmethod
    def _occupied(state: EnvState) -> jax.Array:
        return state.mem_state == OCCUPIED

    @staticmethod
    def _free(state: EnvState) -> jax.Array:
        """Seen free cells, covered or not."""
        return state.mem_state >= FREE

    @staticmethod
    def _covered(state: EnvState) -> jax.Array:
        return state.mem_state == COVERED

    @staticmethod
    def _uncovered(state: EnvState) -> jax.Array:
        """Seen free cells not yet covered: the remaining work robot i knows of."""
        return state.mem_state == FREE

    def _refresh_memory(self, state: EnvState, cover_mask: jax.Array) -> EnvState:
        """Scan, mark observed/covered cells, then exchange maps within comm range.

        cover_mask : (N,) bool — robot i covered the coverable cells its disk overlaps.
        """
        n = self.num_robots
        ids = self._robot_ids
        pos = state.robot_positions
        alive = state.robot_alive
        lidar = self._cast_lidar_all(state)                          # (N, R)

        # Every sample strictly before the first hit is a seen cell, plus the
        # cell just behind the hit point (the obstacle's own cell). A seen cell
        # enters the map with its label: OCCUPIED for a wall cell, else FREE.
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
        label = self._cell_labels[state.map_id][flat]                  # (N, R*(K+1))
        mem = state.mem_state.reshape(n, -1).at[ids[:, None], flat].max(
            jnp.where(valid.reshape(n, -1), label, jnp.int8(UNKNOWN)))

        foot_c, foot_r, touch, _ = self._footprint_cells(pos)
        foot_flat = foot_r * self.grid_w + foot_c
        foot_cover = (cover_mask[:, None] & touch
                      & (self._free_flat[state.map_id, foot_flat] > 0.0))
        mem = mem.at[ids[:, None], foot_flat].max(
            jnp.where(foot_cover, jnp.int8(COVERED), jnp.int8(UNKNOWN)))

        # Single-hop exchange: cell-wise max over direct neighbours (and self),
        # i.e. the union of their dictionaries. Multi-hop spread still happens
        # over successive steps.
        in_range = self._pairwise_sq_dist(pos) < self.comm_radius ** 2
        adj = (in_range & alive[:, None] & alive[None, :]) | jnp.eye(n, dtype=bool)
        mem = jnp.max(jnp.where(adj[:, :, None], mem[None], jnp.int8(UNKNOWN)), axis=1)

        state = state.replace(
            mem_state=mem.reshape(n, self.grid_h, self.grid_w),
            lidar=lidar,
        )
        return state

    def _sample_spawns(self, key: jax.Array, map_id: jax.Array, num_spawns: int) -> jax.Array:
        cands = self._spawn_candidates[map_id]
        order = jax.random.permutation(key, self._num_candidates)
        # Map sizes now differ. Padding repeats a real cell and must never
        # participate in sampling without replacement.
        order = order[jnp.argsort(order >= self._spawn_counts[map_id], stable=True)]
        if not self._spawn_needs_greedy:
            return cands[order[:num_spawns]]

        shuffled = cands[order]
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
        if self.track_memory:
            # The spawn cell is not covered, matching the global coverage grid.
            state = self._refresh_memory(
                state, jnp.zeros((self.num_robots,), bool)
            )
        return self._push_observation(state)

    def step(
        self, state: EnvState, joint_actions: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:
        state, rewards, terminated, truncated, cover = self._step_core(
            state, joint_actions
        )
        if self.track_memory:
            state = self._refresh_memory(state, cover)
        return self._push_observation(state), rewards, terminated, truncated

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
            mem_state        = jnp.full(
                (self.num_robots, self.grid_h, self.grid_w), UNKNOWN, jnp.int8
            ),
            lidar            = jnp.zeros((self.num_robots, self.n_rays), jnp.float32),
            last_discovery   = jnp.full((self.num_robots, 2), -1, jnp.int32),
            sweep_direction  = jnp.zeros((self.num_robots, 2), jnp.int32),
            sweep_run_length = jnp.zeros((self.num_robots,), jnp.int32),
            lane_return      = jnp.zeros((self.num_robots, 2), jnp.int32),
            last_visit       = jnp.full((self.num_robots, 2), -1, jnp.int32),
            cell_entries     = jnp.float32(0),
            revisit_streak   = jnp.zeros(self.num_robots, jnp.int32),
            discovery_streak = jnp.zeros(self.num_robots, jnp.int32),
            no_progress_steps = jnp.zeros(self.num_robots, jnp.int32),
            fallback_active = jnp.zeros(self.num_robots, bool),
            fallback_goal   = jnp.full(self.num_robots, -1, jnp.int32),
            fallback_used   = jnp.zeros(self.num_robots, bool),
            fallback_activated = jnp.zeros(self.num_robots, bool),
            fallback_count  = jnp.zeros(self.num_robots, jnp.int32),
            fallback_stagnation = jnp.zeros(self.num_robots, jnp.int32),
            fallback_best_distance = jnp.full(self.num_robots, jnp.inf),
            fallback_observed_cells = jnp.zeros(self.num_robots, jnp.int32),
            fallback_sequence = jnp.zeros(self.num_robots, bool),
            fallback_commands = jnp.zeros((self.num_robots, self.fallback_sequence_steps, 2), jnp.float32),
            fallback_command_index = jnp.zeros(self.num_robots, jnp.int32),
            fallback_command_count = jnp.zeros(self.num_robots, jnp.int32),
            fallback_sequence_used = jnp.zeros(self.num_robots, bool),
            fallback_safety_override = jnp.zeros(self.num_robots, bool),
            fallback_rejection_flags = jnp.zeros(self.num_robots, jnp.int32),
            obs_history      = jnp.zeros((self.num_robots,
                                         self.observation_stack if self.observation_stack > 1 else 0,
                                         self.frame_dim), jnp.float32),
            previous_visit = jnp.full((self.num_robots, 2), -1, jnp.int32),
            visit_counts = jnp.zeros((self.num_robots, self.grid_h, self.grid_w), jnp.float32),
            global_coverage_history = jnp.zeros((self.critic_stack if self.critic_stack > 1 else 0,
                                                  self.grid_h, self.grid_w), jnp.float32),
            global_occupancy_history = jnp.zeros((self.critic_stack if self.critic_stack > 1 else 0,
                                                   self.num_robots, self.grid_h, self.grid_w), jnp.float32),
            global_visit_history = jnp.zeros((self.critic_stack if self.critic_stack > 1 else 0,
                                               self.num_robots, self.grid_h, self.grid_w), jnp.float32),
            global_kinematics_history = jnp.zeros((self.critic_stack if self.critic_stack > 1 else 0,
                                                    self.num_robots, 6), jnp.float32),
            global_humans_history = jnp.zeros((self.critic_stack if self.critic_stack > 1 else 0,
                                               self.num_humans, 2), jnp.float32),
            global_context_history = jnp.zeros((self.critic_stack if self.critic_stack > 1 else 0,
                                                 self.num_robots, 12 if self.critic_context else 0), jnp.float32),
            global_previous_visit_history = jnp.full((self.critic_stack if self.critic_stack > 1 else 0,
                                                       self.num_robots, 2), -1, jnp.int32),
            global_history_valid = jnp.zeros((self.critic_stack if self.critic_stack > 1 else 0,), jnp.float32),
        )

    def _step_core(
        self, state: EnvState, joint_actions: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Physics, coverage and reward; the last output is the (N,) bool mask
        of robots that covered their footprint cells, for `_refresh_memory`."""
        joint_actions, fallback_goal, fallback_used, control = self._recovery_actions(state, joint_actions)
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
            # Only steps towards a nearby robot are refused: a human stopped
            # next to a waiting robot could otherwise never walk away.
            diff_hr = h_cand_pos[:, None, :] - next_pos[None, :, :]
            cand_d2 = jnp.sum(diff_hr * diff_hr, axis=-1)
            curr_hr = state.human_positions[:, None, :] - next_pos[None, :, :]
            robot_near = jnp.any(
                (cand_d2 < self.human_robot_stop_distance ** 2)
                & (cand_d2 < jnp.sum(curr_hr * curr_hr, axis=-1)),
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

        # A cell is covered as soon as any part of the robot disk overlaps it.
        foot_c, foot_r, touch, foot_d2 = self._footprint_cells(new_pos)
        foot_flat  = foot_r * self.grid_w + foot_c
        foot_cover = (moved[:, None] & touch
                      & (self._free_flat[state.map_id, foot_flat] > 0.0))
        eligible   = foot_cover & (prev_grid[foot_r, foot_c] <= 0.0)

        ids   = self._robot_ids
        claim = jnp.zeros((self.num_cells,), jnp.int32).at[foot_flat].max(
            jnp.where(eligible, ids[:, None] + 1, 0)
        )
        new_cells  = eligible & (claim[foot_flat] == ids[:, None] + 1)
        num_new    = jnp.sum(new_cells, axis=1).astype(jnp.float32)
        discovered = num_new > 0.0
        # The new cell nearest the centre represents this step's discovery in
        # the lane/sweep bookkeeping, which works on single cells.
        nearest = jnp.argmin(jnp.where(new_cells, foot_d2, jnp.inf), axis=1)
        disc_cell = jnp.stack([foot_c[ids, nearest], foot_r[ids, nearest]], axis=-1)
        cell = jnp.stack([cols, rows], axis=-1)
        visit = moved & coverable & jnp.any(cell != state.last_visit, axis=-1)
        # Includes overlapping simultaneous claims, but never counts dwelling
        # within a cell, rotation on the spot or rejected collision attempts.
        revisited = visit & ~discovered
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

        new_grid = prev_grid.at[foot_r, foot_c].max(foot_cover.astype(jnp.float32))

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

        if self.reward_mode in ('progress', 'sequential'):
            rewards = self._progress_reward(
                state, new_pos, prev_grid, num_new, disc_cell, discovery_multiplier,
                wall_hit, robot_hit, robot_hit_human, complete,
            )
        else:
            rewards = jnp.where(
                alive,
                self.alpha * discovery_multiplier * num_new
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

        rewards -= self._axis_motion_cost(new_pos - state.robot_positions)
        rewards += jnp.where(alive, self._sweep_reward(state, disc_cell, discovered, visit, cell), 0.0)

        revisit_streak = jnp.where(discovered, 0, state.revisit_streak + revisited.astype(jnp.int32))
        no_progress_steps = jnp.where(discovered, 0, state.no_progress_steps + alive.astype(jnp.int32))
        streak_cost = (jnp.minimum(revisit_streak, self.revisit_streak_cap)
                       if self.revisit_streak_cap else revisit_streak)
        rewards -= self._revisit_weight(state) * streak_cost * revisited

        # Consecutive new cells pay an increasing bonus; any covered-cell entry
        # restarts the count.
        discovery_streak = jnp.where(
            revisited, 0, state.discovery_streak + discovered.astype(jnp.int32))
        rewards += (self.discovery_streak_bonus * discovered
                    * jnp.minimum(discovery_streak, self.discovery_streak_cap))
        rewards -= jnp.where(alive, self._los_spread_cost(state, new_pos), 0.0)

        step_count = state.step_count + 1
        sweep_run_length, lane_return = self._next_sweep_pattern(
            state, disc_cell, discovered, revisited)
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
            last_discovery   = jnp.where(discovered[:, None], disc_cell, state.last_discovery),
            sweep_direction  = self._next_sweep_direction(state, disc_cell, discovered),
            sweep_run_length = sweep_run_length,
            lane_return      = lane_return,
            previous_visit   = jnp.where(visit[:, None], state.last_visit, state.previous_visit),
            last_visit       = jnp.where(visit[:, None], cell, state.last_visit),
            visit_counts     = state.visit_counts.at[ids, rows, cols].add(visit.astype(jnp.float32)),
            cell_entries     = state.cell_entries + jnp.sum(visit.astype(jnp.float32)),
            revisit_streak   = revisit_streak,
            discovery_streak = discovery_streak,
            no_progress_steps = no_progress_steps,
            fallback_goal   = fallback_goal,
            fallback_used   = fallback_used,
            **control,
        )
        # Recovery keeps control until this robot's own footprint covers its
        # goal; a teammate covering it or cells discovered along the route do
        # not release it (a goal known covered is replanned, not released).
        # It also ends on an uncoverable goal or when no known uncovered cell
        # is reachable any more (goal -1, so fallback_used is false).
        own_cover = jnp.any(foot_cover & (foot_flat == fallback_goal[:, None]), axis=1)
        reached = fallback_used & (own_cover | ((flat == fallback_goal) & ~coverable))
        continuing = fallback_used & ~reached & alive_next & ~terminated & ~truncated
        request = (self.fallback_enabled & ~state.fallback_active & alive_next
                   & ~terminated & ~truncated
                   & ((revisit_streak >= self.fallback_revisit_threshold)
                      | (no_progress_steps >= self.fallback_stall_steps)))
        # Include this step's own coverage before planning; shared memory is
        # refreshed once by step/VecEnv after physics (including auto-reset).
        planning = next_state.replace(mem_state=state.mem_state.at[ids[:, None], foot_r, foot_c].max(
            jnp.where(foot_cover, jnp.int8(COVERED), jnp.int8(UNKNOWN))))
        goals = self._recovery_goals(planning, request)
        activated = request & (goals >= 0)
        rewards -= self.fallback_cost * activated
        next_state = next_state.replace(
            fallback_active=continuing | activated,
            fallback_goal=jnp.where(activated, goals, jnp.where(continuing, fallback_goal, -1)),
            fallback_activated=activated,
            fallback_count=state.fallback_count + activated.astype(jnp.int32),
            fallback_stagnation=jnp.where(continuing, next_state.fallback_stagnation, 0),
            fallback_best_distance=jnp.where(continuing, next_state.fallback_best_distance, jnp.inf),
            fallback_observed_cells=jnp.where(continuing, next_state.fallback_observed_cells, 0),
            fallback_sequence=continuing & next_state.fallback_sequence,
            fallback_commands=jnp.where(continuing[:, None, None], next_state.fallback_commands, 0.),
            fallback_command_index=jnp.where(continuing & ~collided, next_state.fallback_command_index, 0),
            fallback_command_count=jnp.where(continuing & ~collided, next_state.fallback_command_count, 0),
            revisit_streak=jnp.where(reached, 0, revisit_streak),
            no_progress_steps=jnp.where(reached, 0, no_progress_steps),
        )
        return next_state, rewards, terminated, truncated, moved

    def _recovery_goals(self, state, requested):
        if not self.fallback_enabled:
            return jnp.full(self.num_robots, -1, jnp.int32)
        cols, rows = self._pos_to_cell(state.robot_positions)

        def plan(i):
            free = state.mem_state[i] >= FREE
            targets = (state.mem_state[i] == FREE) & requested[i]
            return run_if(requested[i],
                          lambda: astar(free, targets, rows[i] * self.grid_w + cols[i])[0],
                          jnp.int32(-1))
        return jax.lax.map(plan, self._robot_ids)

    def _recovery_actions(self, state, actions):
        fields = ('fallback_stagnation', 'fallback_best_distance', 'fallback_observed_cells', 'fallback_sequence',
                  'fallback_commands', 'fallback_command_index', 'fallback_command_count',
                  'fallback_sequence_used', 'fallback_safety_override', 'fallback_rejection_flags')
        cols, rows = self._pos_to_cell(state.robot_positions)

        def control(i):
            empty = (jnp.int32(0), jnp.float32(jnp.inf), jnp.int32(0), jnp.bool_(False),
                     jnp.zeros_like(state.fallback_commands[i]), jnp.int32(0), jnp.int32(0),
                     jnp.bool_(False), jnp.bool_(False), jnp.int32(0))

            def recover():
                pos, heading = state.robot_positions[i], state.robot_headings[i]
                velocity, lidar = state.robot_velocities[i], state.lidar[i]
                start = rows[i] * self.grid_w + cols[i]
                free = state.mem_state[i] >= FREE
                all_targets = (state.mem_state[i] == FREE) & state.fallback_active[i]
                goal = state.fallback_goal[i]
                keep = (goal >= 0) & all_targets.reshape(-1)[jnp.maximum(goal, 0)]
                targets = jnp.where(keep, jnp.arange(self.num_cells).reshape(free.shape) == goal, all_targets)
                planned = astar(free, targets, start, return_path=True)
                goal, waypoint, route, length = jax.lax.cond(
                    planned[0] < 0,
                    lambda: astar(free, all_targets, start, return_path=True),
                    lambda: planned)
                point = (jnp.array([waypoint % self.grid_w, waypoint // self.grid_w]) + .5) * self.cell_size
                distance = jnp.maximum(length - 2, 0) * self.cell_size + jnp.linalg.norm(pos - point)
                changed = goal != state.fallback_goal[i]
                cell_count = jnp.sum(state.mem_state[i] != UNKNOWN, dtype=jnp.int32)
                improved = (changed | (cell_count != state.fallback_observed_cells[i])
                            | (distance < state.fallback_best_distance[i] - .02))
                stagnation = jnp.where(improved, 0, state.fallback_stagnation[i] + 1)
                best = jnp.where(improved, distance, state.fallback_best_distance[i])
                sequence = ((state.fallback_sequence[i] & ~changed)
                            | (stagnation >= self.fallback_dwa_stall_steps)) & (goal >= 0)
                dwa_action = dwa(self, pos, heading, velocity, lidar, free, point)

                def execute_sequence():
                    def generate():
                        # Replan around currently sensed obstructions without
                        # storing moving people/robots as permanent walls.
                        local_free = local_route_free(self, free, pos, heading, lidar)
                        target = jnp.arange(self.num_cells).reshape(free.shape) == goal
                        local = astar(local_free, target, start, return_path=True)
                        selected_route = jnp.where(local[0] >= 0, local[2], route)
                        selected_length = jnp.where(local[0] >= 0, local[3], length)
                        commands, count = velocity_sequence(
                            self, pos, heading, velocity, selected_route, selected_length)
                        return commands, jnp.int32(0), count

                    commands, index, count = run_if(
                        changed | ~state.fallback_sequence[i]
                        | (state.fallback_command_index[i] >= state.fallback_command_count[i]),
                        generate, (state.fallback_commands[i], state.fallback_command_index[i],
                                   state.fallback_command_count[i]))
                    command = commands[jnp.minimum(index, self.fallback_sequence_steps - 1)]
                    flags = jnp.where(index < count, command_safety_flags(
                        self, pos, heading, command, lidar, free), jnp.int32(16))
                    safe = flags == 0
                    action = jnp.array([2 * command[0] / self.v_max - 1., command[1] / self.omega_max])
                    # DWA remains responsible for reactive avoidance. Any
                    # interruption invalidates the queue; replan from actual pose.
                    return (jnp.where(safe, action, dwa_action), commands,
                            jnp.where(safe, index + 1, 0), jnp.where(safe, count, 0), safe, ~safe, flags)

                action, commands, index, count, sequence_used, overridden, flags = run_if(
                    sequence, execute_sequence,
                    (dwa_action, empty[4], jnp.int32(0), jnp.int32(0), jnp.bool_(False), jnp.bool_(False), jnp.int32(0)))
                return (jnp.where(goal >= 0, action, actions[i]), goal, goal >= 0,
                        stagnation, best, cell_count, sequence, commands, index, count, sequence_used, overridden, flags)

            default = (actions[i], jnp.int32(-1), jnp.bool_(False), *empty)
            if not self.fallback_enabled:
                return default
            return run_if(state.fallback_active[i] & state.robot_alive[i], recover, default)

        output = jax.lax.map(control, self._robot_ids)
        return (*output[:3], dict(zip(fields, output[3:])))

    def _discovery_direction(self, state, disc_cell):
        delta = disc_cell - state.last_discovery
        adjacent = (jnp.all(state.last_discovery >= 0, axis=-1)
                    & (jnp.sum(jnp.abs(delta), axis=-1) == 1))
        return delta, adjacent

    def _next_sweep_direction(self, state, disc_cell, discovered):
        delta, adjacent = self._discovery_direction(state, disc_cell)
        direction = jnp.where(adjacent[:, None], delta, 0)
        return jnp.where(discovered[:, None], direction, state.sweep_direction)

    def _sequential_discovery_reward(self, state, disc_cell, discovered):
        delta, adjacent = self._discovery_direction(state, disc_cell)
        straight = jnp.all(delta == state.sweep_direction, axis=-1)
        reversal = (jnp.any(state.lane_return != 0, axis=-1)
                    & jnp.all(delta == state.lane_return, axis=-1))
        return discovered * adjacent * (self.sequential_bonus + self.straight_bonus * straight
                                         + self.boustrophedon_bonus * reversal)

    def _next_sweep_pattern(self, state, disc_cell, discovered, revisited):
        """Recognise a straight run, a perpendicular cell, then reverse on the next lane.

        At least two straight new-cell edges must precede the lane shift. Any
        redundant entry breaks the pattern; turning in place does not.
        """
        delta, adjacent = self._discovery_direction(state, disc_cell)
        straight = jnp.all(delta == state.sweep_direction, axis=-1)
        run = jnp.where(adjacent, jnp.where(straight, state.sweep_run_length + 1, 1), 0)
        shift = (adjacent & (state.sweep_run_length >= 2)
                 & (jnp.sum(delta * state.sweep_direction, axis=-1) == 0))
        lane = jnp.where(shift[:, None], -state.sweep_direction, 0)
        run = jnp.where(revisited, 0, jnp.where(discovered, run, state.sweep_run_length))
        lane = jnp.where(revisited[:, None], 0,
                         jnp.where(discovered[:, None], lane, state.lane_return))
        return run, lane

    def _sweep_preference(self, state):
        """Boustrophedon lane direction and whether its next cell is open work.

        The preferred direction is the reversal after a lane shift, otherwise
        the current sweep direction. The cell ahead of the last discovery is
        open when the robot's own memory holds neither an occupied nor a
        covered cell there; unknown cells count as open.
        """
        shifted = jnp.any(state.lane_return != 0, axis=-1)
        pref = jnp.where(shifted[:, None], state.lane_return, state.sweep_direction)
        has_pref = (jnp.all(state.last_discovery >= 0, axis=-1)
                    & jnp.any(pref != 0, axis=-1))
        ahead = state.last_discovery + pref                               # (col, row)
        inside = ((ahead[:, 0] >= 0) & (ahead[:, 0] < self.grid_w)
                  & (ahead[:, 1] >= 0) & (ahead[:, 1] < self.grid_h))
        c = jnp.clip(ahead[:, 0], 0, self.grid_w - 1)
        r = jnp.clip(ahead[:, 1], 0, self.grid_h - 1)
        # Decentralised: only the robot's memory decides; an unseen wall is open.
        cell = state.mem_state[self._robot_ids, r, c]
        open_ahead = has_pref & inside & (cell != OCCUPIED) & (cell != COVERED)
        return pref, has_pref, ahead, open_ahead

    def _sweep_reward(self, state, disc_cell, discovered, visit, cell):
        """Boustrophedon shaping on the cell grid, in every reward mode.

        straight  sweep_straight_bonus for a new cell straight ahead in the
                  preferred direction (continuing a lane, or reversing on the
                  next lane after a one-cell shift).
        turn      sweep_turn_bonus for a new perpendicular neighbour once the
                  cell ahead is a wall, covered or outside: the lane shift.
        break     sweep_break_cost for leaving the lane head into any other
                  cell while the cell ahead is still open work.
        """
        pref, has_pref, ahead, open_ahead = self._sweep_preference(state)
        delta, adjacent = self._discovery_direction(state, disc_cell)
        new_adjacent = discovered & adjacent & has_pref
        straight = new_adjacent & jnp.all(delta == pref, axis=-1)
        turn = (new_adjacent & ~open_ahead
                & (jnp.sum(delta * pref, axis=-1) == 0))
        at_head = jnp.all(state.last_visit == state.last_discovery, axis=-1)
        broke = (visit & at_head & open_ahead
                 & jnp.any(cell != ahead, axis=-1))
        return (self.sweep_straight_bonus * straight
                + self.sweep_turn_bonus * turn
                - self.sweep_break_cost * broke)

    def _known_work_field(self, state):
        """Routes through observed free cells only; covered frontiers allow exploration."""
        free = self._free(state)
        # Outside the map is not an exploration frontier.
        unknown = jnp.pad(~self._known(state), ((0, 0), (1, 1), (1, 1)), constant_values=False)
        frontier = free & (unknown[:, :-2, 1:-1] | unknown[:, 2:, 1:-1]
                           | unknown[:, 1:-1, :-2] | unknown[:, 1:-1, 2:])
        targets = jnp.stack([self._uncovered(state), frontier])
        fields = self._geodesic_distance(targets, jnp.broadcast_to(free, targets.shape))
        cols, rows = self._pos_to_cell(state.robot_positions)
        has_work = fields[0, self._robot_ids, rows, cols] < 0.5 * _FAR
        # A covered frontier must not distract the robot from reachable known work.
        return jnp.where(has_work[:, None, None], fields[0], fields[1])

    def _line_of_sight(self, pos: jax.Array, map_id: jax.Array) -> jax.Array:
        """(N, N) bool: the segment between two robot centres crosses no wall.

        Slab test of each segment against every axis-aligned wall rectangle;
        the diagonal is False.
        """
        start = pos[:, None, None, :]                                    # (N,1,1,2)
        d = (pos[None, :, :] - pos[:, None, :])[:, :, None, :]           # (N,N,1,2)
        lo = jnp.stack([self._wall_x0[map_id], self._wall_y0[map_id]], axis=-1)
        hi = jnp.stack([self._wall_x1[map_id], self._wall_y1[map_id]], axis=-1)
        parallel = jnp.abs(d) < 1e-9
        safe = jnp.where(parallel, 1.0, d)
        t0 = (lo - start) / safe
        t1 = (hi - start) / safe
        inside = (start >= lo) & (start <= hi)
        t_min = jnp.where(parallel, jnp.where(inside, -jnp.inf, jnp.inf), jnp.minimum(t0, t1))
        t_max = jnp.where(parallel, jnp.where(inside, jnp.inf, -jnp.inf), jnp.maximum(t0, t1))
        t_near = jnp.max(t_min, axis=-1)                                 # (N,N,W)
        t_far = jnp.min(t_max, axis=-1)
        blocked = jnp.any((t_near <= t_far) & (t_far >= 0.0) & (t_near <= 1.0), axis=-1)
        return ~blocked & ~jnp.eye(self.num_robots, dtype=bool)

    def _los_spread_cost(self, state: EnvState, new_pos: jax.Array) -> jax.Array:
        """Repulsion from teammates in line of sight, strongest early in the episode.

        Per teammate: max(0, 1 - d / los_spread_radius); robots separated by a
        wall pay nothing. The weight decays linearly from los_spread_weight to
        los_spread_end_fraction of it over los_spread_decay_steps.
        """
        if self.los_spread_weight == 0.0 or self.num_robots < 2:
            return jnp.zeros(self.num_robots, jnp.float32)
        d = jnp.sqrt(self._pairwise_sq_dist(new_pos))
        visible = self._line_of_sight(new_pos, state.map_id) & state.robot_alive[None, :]
        crowd = jnp.maximum(0.0, 1.0 - d / self.los_spread_radius) * visible
        phase = jnp.clip(state.step_count / self.los_spread_decay_steps, 0.0, 1.0)
        weight = self.los_spread_weight * (
            1.0 - (1.0 - self.los_spread_end_fraction) * phase)
        return weight * jnp.sum(crowd, axis=1)

    def _revisit_weight(self, state):
        """Decay with episode time or coverage, whichever is further along."""
        covered = jnp.sum(state.coverage_grid) / jnp.maximum(self.free_totals[state.map_id], 1.)
        phase = jnp.clip(jnp.maximum(covered, state.step_count / self.max_steps), 0., 1.)
        return self.revisit_cost * (self.revisit_end_fraction
            + (1. - self.revisit_end_fraction) * (1. - phase) ** self.revisit_decay_power)

    def _task_context(self, state):
        """Observable time and history needed to interpret control and rewards."""
        return jnp.column_stack([
            jnp.full(self.num_robots, state.step_count / self.max_steps),
            jnp.minimum(state.revisit_streak / self.fallback_revisit_threshold, 2.),
            jnp.minimum(state.no_progress_steps / self.fallback_stall_steps, 2.),
            state.fallback_active.astype(jnp.float32),
            jnp.minimum(state.sweep_run_length / max(self.grid_h, self.grid_w), 1.),
            state.lane_return,
        ])

    def _visible_teammates(self, state):
        """Exactly the teammates represented by communication slots in the actor input."""
        n = self.num_robots
        d2 = self._pairwise_sq_dist(state.robot_positions)
        valid = ((d2 < self.comm_radius ** 2) & ~jnp.eye(n, dtype=bool)
                 & state.robot_alive[:, None] & state.robot_alive[None, :])
        k = min(self.comm_slots, n - 1)
        selected = jnp.zeros((n, n), bool)
        if k:
            _, idx = jax.lax.top_k(-jnp.where(valid, d2, _BIG), k)
            selected = selected.at[self._robot_ids[:, None], idx].set(True)
        return selected & valid

    def _axis_motion_cost(self, displacement):
        """Squared nearest-cardinal angular error, weighted by actual travel speed.

        30 and 60 degrees both have a 30-degree error; 45 is the maximum. Squaring
        keeps the cost smooth at the axes, so small drifts are barely penalized.
        Rotation in place, blocked translation and dead robots pay no cost.
        """
        components = jnp.abs(displacement)
        error = jnp.arctan2(jnp.min(components, axis=-1),
                            jnp.max(components, axis=-1))
        speed = jnp.clip(jnp.linalg.norm(displacement, axis=-1)
                         / (self.v_max * self.dt), 0.0, 1.0)
        return self.axis_alignment_cost * (error / (jnp.pi / 4)) ** 2 * speed

    def _axis_motion_reward(self, displacement, progress):
        """Prefer cardinal travel towards work; stationary/blocked motion earns zero."""
        squared = displacement ** 2
        distance_sq = jnp.sum(squared, axis=-1)
        # cos(2 * direction)^2: 1 on either axis, 0 on diagonals.
        alignment = ((squared[:, 0] - squared[:, 1])
                     / jnp.maximum(distance_sq, 1e-12)) ** 2
        speed = jnp.clip(jnp.sqrt(distance_sq) / (self.v_max * self.dt), 0.0, 1.0)
        approach = jnp.clip(progress / self._full_progress, 0.0, 1.0)
        return self.axis_alignment_bonus * alignment * jnp.minimum(speed, approach)

    def _progress_reward(
        self, state: EnvState, new_pos: jax.Array, prev_grid: jax.Array,
        num_new: jax.Array, disc_cell: jax.Array, discovery_multiplier: jax.Array,
        wall_hit: jax.Array, robot_hit: jax.Array, robot_hit_human: jax.Array,
        complete: jax.Array,
    ) -> jax.Array:
        """Per-robot progress reward, with observable routes/bonuses in sequential mode.

        discovery   alpha * (1 + growth * coverage) per newly covered cell.
        progress    progress_weight per cell of geodesic approach to the
                    nearest cell the robot believes uncovered. The belief is
                    its own memory (own coverage + what teammates shared in
                    comm range), so the implicit target uses only knowledge
                    the robot has. Covered-cell entry streak penalties are
                    applied separately in `_step_core`.
        loiter      loiter_cost per step without discovery, scaled by the
                    missing fraction of full-speed progress: parking,
                    spinning and wandering over covered cells cost the full
                    amount, driving straight at v_max towards work costs 0.
        spread      spread_weight * max(0, 1 - d / spread_radius) per teammate,
                    a constant pressure to keep the team apart.
        completion  shared bonus when the map is covered, larger when early.

        Both potentials of a step use the same pre-step field, so covering the
        last cell of an area does not look like a sudden loss of progress, and
        the progress term telescopes for loops while that field remains unchanged.
        """
        alive = state.robot_alive
        discovered = num_new > 0.0
        free = self.free_masks[state.map_id] > 0.5                       # (H, W)
        if self.use_memory:
            believed = self._covered(state)                               # (N, H, W)
        else:
            believed = jnp.broadcast_to(prev_grid > 0.5, (self.num_robots, *free.shape))
        targets = free[None] & ~believed
        if self.reward_mode == 'sequential':
            dist = self._known_work_field(state)
        else:
            dist = self._geodesic_distance(targets, jnp.broadcast_to(free, targets.shape))

        before = self._work_distance(dist, state.robot_positions)
        after = self._work_distance(dist, new_pos)
        progress = jnp.where((before < 0.5 * _FAR) & (after < 0.5 * _FAR), before - after, 0.0)

        idle = jnp.clip(1.0 - progress / self._full_progress, 0.0, 1.0)
        loiter = self.loiter_cost * idle * ~discovered

        d = jnp.sqrt(self._pairwise_sq_dist(new_pos))
        crowd = jnp.maximum(0.0, 1.0 - d / self.spread_radius) * alive[None, :]
        if self.reward_mode == 'sequential':
            crowd = crowd * self._visible_teammates(state)
        spread = self.spread_weight * jnp.sum(crowd, axis=1)

        time_left = jnp.maximum(1.0 - (state.step_count + 1) / self.max_steps, 0.0)
        completion = (self.completion_bonus * complete
                      * (1.0 + self.completion_time_bonus * time_left))

        reward = (
            self.alpha * discovery_multiplier * num_new
            + self.progress_weight * progress
            + self._axis_motion_reward(new_pos - state.robot_positions, progress)
            - loiter
            - spread
            - self.tau
            - self.wall_kappa * wall_hit
            - self.kappa * robot_hit
            - self.human_kappa * robot_hit_human
            + completion
        )
        if self.reward_mode == 'sequential':
            reward += self._sequential_discovery_reward(state, disc_cell, discovered)
        return jnp.where(alive, reward, 0.0).astype(jnp.float32)

    def set_ghost_robot_prob(self, state: EnvState, prob: jax.Array) -> EnvState:
        return state.replace(ghost_robot_prob=jnp.clip(prob, 0.0, 1.0))

    def _build_crop_ring(self):
        """Sample offsets for the summary ring around an S x S crop.

        The crop grows to (S+2) x (S+2) in crop coordinates [row = dy, col = dx].
        A side ring cell covers the strip of cells beyond the crop in line with
        its crop row/column, out to the map edge; a corner ring cell covers the
        quadrant beyond both crop edges. Offsets reach hypot(H, W) cells so the
        heading-rotated legacy crop also spans the whole map.

        Returns two groups (sides, corners), each
        (offsets (R, K, 2) as (dx, dy) cells, ring rows (R,), ring cols (R,),
         extent normaliser).
        """
        size, half = self.local_coverage_size, self.local_coverage_size // 2
        far = np.arange(half + 1, half + 1 + int(np.ceil(np.hypot(self.grid_h, self.grid_w))))
        beyond = {0: -far, size + 1: far}
        groups = {False: ([], [], []), True: ([], [], [])}
        for er in range(size + 2):
            for ec in range(size + 2):
                dy = beyond.get(er, np.array([er - 1 - half]))
                dx = beyond.get(ec, np.array([ec - 1 - half]))
                if len(dy) == 1 and len(dx) == 1:
                    continue  # the crop itself
                gx, gy = np.meshgrid(dx, dy)
                offsets, rows, cols = groups[len(dx) > 1 and len(dy) > 1]
                offsets.append(np.stack([gx.ravel(), gy.ravel()], axis=-1))
                rows.append(er)
                cols.append(ec)
        norms = {False: float(max(self.grid_h, self.grid_w)), True: float(self.num_cells)}
        return [(jnp.asarray(np.stack(o), jnp.int32), jnp.asarray(r), jnp.asarray(c), norms[k])
                for k, (o, r, c) in groups.items()]

    def _crop_cells(self, state: EnvState, offsets: jax.Array):
        """Grid cells under crop offsets (..., 2) = (dx, dy) cells, per robot.

        Memory crops are grid-aligned around the robot's cell; legacy crops are
        rotated by the heading and sampled from the robot position, exactly as
        the legacy patch itself. Returns rows, cols (clipped) and inside flags,
        each (N, ...).
        """
        shape = (self.num_robots,) + (1,) * (offsets.ndim - 1)
        if self.use_memory:
            cols, rows = self._pos_to_cell(state.robot_positions)
            c = cols.reshape(shape) + offsets[..., 0]
            r = rows.reshape(shape) + offsets[..., 1]
        else:
            local = offsets.astype(jnp.float32) * self.cell_size
            cos = jnp.cos(state.robot_headings).reshape(shape)
            sin = jnp.sin(state.robot_headings).reshape(shape)
            x = state.robot_positions[:, 0].reshape(shape) + cos * local[..., 0] - sin * local[..., 1]
            y = state.robot_positions[:, 1].reshape(shape) + sin * local[..., 0] + cos * local[..., 1]
            c = jnp.floor(x / self.cell_size).astype(jnp.int32)
            r = jnp.floor(y / self.cell_size).astype(jnp.int32)
        inside = (r >= 0) & (r < self.grid_h) & (c >= 0) & (c < self.grid_w)
        return jnp.clip(r, 0, self.grid_h - 1), jnp.clip(c, 0, self.grid_w - 1), inside

    def _add_crop_summary(self, state: EnvState, crop: jax.Array, inside: jax.Array,
                          sources: jax.Array, outside) -> jax.Array:
        """Wrap an (N, C, S, S) crop in a one-cell summary ring.

        Ring cells hold the mean of each robot's memory channel over the in-map cells
        beyond the crop in that direction (a strip on the sides, a quadrant in
        the corners); `outside` (C,) fills a ring cell with no in-map cell,
        matching the crop's beyond-the-floor convention. An extra extent
        channel holds how many in-map cells were averaged, divided by
        max(H, W) on the sides and H * W in the corners; inside the crop it is
        the in-map flag. sources: (N, H, W, C) per-robot memory maps (own lidar
        OR'd with teammates in comm range), never the global grid.
        Returns (N, C + 1, S + 2, S + 2).
        """
        n, ch, size, _ = crop.shape
        outside = jnp.asarray(outside, jnp.float32)
        full = jnp.zeros((n, ch + 1, size + 2, size + 2), jnp.float32)
        full = full.at[:, :ch, 1:-1, 1:-1].set(crop)
        full = full.at[:, ch, 1:-1, 1:-1].set(inside.astype(jnp.float32))
        robot = self._robot_ids[:, None, None]
        for offsets, ring_r, ring_c, norm in self._crop_ring:
            r, c, ins = self._crop_cells(state, offsets)           # (N, R, K)
            weight = ins.astype(jnp.float32)
            count = weight.sum(-1)                                  # (N, R)
            total = jnp.einsum('nrk,nrkc->nrc', weight, sources[robot, r, c])
            mean = jnp.where(count[..., None] > 0,
                             total / jnp.maximum(count, 1.0)[..., None], outside)
            full = full.at[:, :ch, ring_r, ring_c].set(jnp.swapaxes(mean, 1, 2))
            full = full.at[:, ch, ring_r, ring_c].set(count / norm)
        return full

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
        With crop_mode='directional' the crop is instead
        obstacle | covered | robots, 3 x (S+2) x (S+2), see _directional_crops.
        With full memory enabled, five history values precede lidar: a valid
        flag and relative past-cell position (previous visit, or last discovery
        for old checkpoints), followed by the historical sweep direction.
        Only with memory_map_obs (old checkpoints) the five-channel personal
        map follows the crop. With sweep_obs, the
        preferred lane direction and whether its next cell is open follow them.
        Crop cells beyond the floor boundary are occupied=1, covered=0, known=1.
        With crop_summary the crop is 4 x (S+2) x (S+2): a ring averaging each
        channel over everything beyond the crop, plus an extent channel
        (see _add_crop_summary).
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
            in_range = (d2 < self.comm_radius ** 2) & alive[:, None] & alive[None, :]
            neg_d2, idx = jax.lax.top_k(-jnp.where(in_range, d2, _BIG), k)
            valid = (-neg_d2 < _BIG).astype(jnp.float32)[..., None]
            rel = pos[idx] - pos[:, None, :]                          # (N, k, 2)
            slot = jnp.concatenate([(idx + 1).astype(jnp.float32)[..., None], rel], axis=-1)
            parts.append((slot * valid).reshape(n, 3 * k))
        if self.comm_slots > k:
            parts.append(jnp.zeros((n, 3 * (self.comm_slots - k)), jnp.float32))

        if self.use_full_memory:
            history = state.previous_visit if self.history_cell == 'previous_visit' else state.last_discovery
            valid = jnp.all(history >= 0, axis=-1)
            centre = (history + 0.5) * self.cell_size
            rel = (centre - pos) / jnp.array([self.map_layout.width, self.map_layout.height])
            parts.extend([valid[:, None].astype(jnp.float32),
                          jnp.where(valid[:, None], rel, 0.0), state.sweep_direction])
        if self.sweep_obs:
            pref, _, _, open_ahead = self._sweep_preference(state)
            parts.extend([pref.astype(jnp.float32), open_ahead[:, None].astype(jnp.float32)])
        if self.known_coverage_obs:
            parts.append(self._known_coverage(state)[:, None])
        parts.append(state.lidar)

        parts.append(self._actor_crops(state))
        if self.memory_map_obs:
            cols, rows = self._pos_to_cell(pos)
            own = jnp.zeros((n, self.grid_h, self.grid_w), jnp.float32)
            own = own.at[ids, rows, cols].set(alive.astype(jnp.float32))
            peers = jnp.einsum('ij,jhw->ihw', self._visible_teammates(state).astype(jnp.float32), own)
            maps = jnp.stack([
                self._occupied(state), self._covered(state), self._known(state),
                own, jnp.minimum(peers, 1.0),
            ], axis=1).astype(jnp.float32)
            parts.append(maps.reshape(n, -1))

        return jnp.concatenate(parts, axis=1).astype(jnp.float32)

    def _push_observation(self, state: EnvState) -> EnvState:
        if self.observation_stack > 1:
            frame = self._get_frame_obs(state)
            state = state.replace(obs_history=jnp.concatenate(
                [state.obs_history[:, 1:], frame[:, None]], axis=1))
        if self.critic_stack > 1:
            gs = self.get_global_state(state)
            def push(old, frame):
                return jnp.concatenate([old[1:], frame[None]], axis=0)
            state = state.replace(
                global_coverage_history=push(state.global_coverage_history, gs.coverage),
                global_occupancy_history=push(state.global_occupancy_history, gs.occupancy),
                global_visit_history=push(state.global_visit_history, gs.visit_counts),
                global_kinematics_history=push(state.global_kinematics_history, gs.kinematics),
                global_humans_history=push(state.global_humans_history, gs.human_positions),
                global_context_history=push(state.global_context_history, gs.task_context),
                global_previous_visit_history=push(state.global_previous_visit_history, gs.previous_visit),
                global_history_valid=push(state.global_history_valid, jnp.float32(1.)),
            )
        return state

    def get_obs(self, state: EnvState) -> jax.Array:
        if self.observation_stack == 1:
            return self._get_frame_obs(state)
        # Keep all continuous values in the RMS prefix and binary maps outside it.
        history = state.obs_history
        return jnp.concatenate([
            history[:, :, :self.frame_norm_dim].reshape(self.num_robots, -1),
            history[:, :, self.frame_norm_dim:].reshape(self.num_robots, -1),
        ], axis=-1)

    def _get_frame_obs(self, state: EnvState) -> jax.Array:
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
        if self.known_coverage_obs:
            parts.append(self._known_coverage(state)[:, None])

        parts.append(self._cast_lidar_all(state))

        if self.use_local_coverage_obs:
            parts.append(self._actor_crops(state))

        return jnp.concatenate(parts, axis=1).astype(jnp.float32)

    def _known_coverage(self, state: EnvState) -> jax.Array:
        """(N,) covered fraction of the free cells robot i has in memory."""
        known_free = jnp.sum(self._free(state), axis=(1, 2)).astype(jnp.float32)
        covered = jnp.sum(self._covered(state), axis=(1, 2)).astype(jnp.float32)
        return covered / jnp.maximum(known_free, 1.0)

    def _actor_crops(self, state: EnvState) -> jax.Array:
        """The local crop each actor observes, (N, crop_dim).

        Decentralised: every value comes from the robot's belief map (lidar
        adds seen cells as occupied or free, driving over a cell marks it
        covered, and maps are merged with teammates in comm range).
        memory_comm: [occupied, covered, known], grid-aligned; cells beyond the
        floor are occupied=1, covered=0, known=1.
        legacy: max(covered, known wall) sampled in the heading-rotated frame;
        cells beyond the floor are 1. With crop_summary the legacy patch gains
        a known channel (1 beyond the floor), and both modes are wrapped in the
        summary ring of _add_crop_summary.
        """
        if self.crop_mode == 'directional':
            return self._directional_crops(state)
        n, ids = self.num_robots, self._robot_ids
        i = ids[:, None, None]
        occupied_map = self._occupied(state).astype(jnp.float32)
        covered_map = self._covered(state).astype(jnp.float32)
        known_map = self._known(state).astype(jnp.float32)
        if self.use_memory:
            cols, rows = self._pos_to_cell(state.robot_positions)
            r = rows[:, None, None] + self._crop_offsets[None, :, None]  # (N, S, 1)
            c = cols[:, None, None] + self._crop_offsets[None, None, :]  # (N, 1, S)
            inside = (r >= 0) & (r < self.grid_h) & (c >= 0) & (c < self.grid_w)
            r = jnp.clip(r, 0, self.grid_h - 1)
            c = jnp.clip(c, 0, self.grid_w - 1)
            occupied = jnp.where(inside, occupied_map[i, r, c], 1.0)
            covered = jnp.where(inside, covered_map[i, r, c], 0.0)
            known = jnp.where(inside, known_map[i, r, c], 1.0)
            crop = jnp.stack([occupied, covered, known], axis=1)     # (N, 3, S, S)
            if self.crop_summary:
                sources = jnp.stack([occupied_map, covered_map, known_map], axis=-1)
                crop = self._add_crop_summary(state, crop, inside, sources, (1., 0., 1.))
            return crop.reshape(n, -1)

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
        source = jnp.maximum(covered_map, occupied_map)                # (N, H, W)
        # Virtual cells beyond the floor use the same value as obstacles.
        patch = jnp.where(inside, source[i, rows, cols], 1.0)[:, None]  # (N, 1, S, S)
        if self.crop_summary:
            known = jnp.where(inside, known_map[i, rows, cols], 1.0)
            patch = jnp.concatenate([patch, known[:, None]], axis=1)
            sources = jnp.stack([source, known_map], axis=-1)
            patch = self._add_crop_summary(state, patch, inside, sources, (1., 1.))
        return patch.reshape(n, -1)

    def _directional_crops(self, state: EnvState) -> jax.Array:
        """[obstacle, covered, robots] S x S crop with a directional ring, (N, crop_dim).

        Core S x S, robot in the centre cell, grid-aligned (row = +y), binary:
          obstacle  a wall cell the robot has seen; 1 beyond the floor
          covered   1 if covered in the robot's memory, else 0
          robots    a teammate in communication range stands in the cell
        Ring of thickness 1: each side holds, per channel, the mean of that
        channel over the in-map cells beyond that side, so the obstacle ring
        is the seen-wall fraction, the covered ring the covered fraction and
        the robots ring the fraction of cells holding a teammate in comm
        range. A cell outside the S x S core belongs to the side of its larger
        offset (rows on ties). A side with no in-map cell takes the
        beyond-the-floor values (obstacle 1, covered 0, robots 0). Corners
        are 0. Everything comes from the robot's own memory, never the global
        grid.
        """
        n, ids = self.num_robots, self._robot_ids
        i = ids[:, None, None]
        half, size = self.local_coverage_size // 2, self.local_coverage_size
        cols, rows = self._pos_to_cell(state.robot_positions)
        r = rows[:, None, None] + self._crop_offsets[None, :, None]      # (N, S, 1)
        c = cols[:, None, None] + self._crop_offsets[None, None, :]      # (N, 1, S)
        inside = (r >= 0) & (r < self.grid_h) & (c >= 0) & (c < self.grid_w)
        r = jnp.clip(r, 0, self.grid_h - 1)
        c = jnp.clip(c, 0, self.grid_w - 1)
        # Full-grid channel maps from the robot's memory: (N, 3, H, W).
        own = jnp.zeros((n, self.grid_h, self.grid_w), jnp.float32)
        own = own.at[ids, rows, cols].set(state.robot_alive.astype(jnp.float32))
        seen = self._visible_teammates(state).astype(jnp.float32)          # (N, N)
        maps = jnp.stack([
            self._occupied(state).astype(jnp.float32), self._covered(state).astype(jnp.float32),
            jnp.minimum(jnp.einsum('nm,mhw->nhw', seen, own), 1.0),
        ], axis=1)
        core = jnp.where(inside[:, None], maps[i[:, None], jnp.arange(3)[None, :, None, None],
                                                r[:, None], c[:, None]],
                         jnp.array([1.0, 0.0, 0.0])[None, :, None, None])  # (N, 3, S, S)

        # Per-channel mean beyond each side: (N, 4, 3) for -row, +row, -col, +col.
        dr = jnp.arange(self.grid_h)[None, :, None] - rows[:, None, None]  # (N, H, 1)
        dc = jnp.arange(self.grid_w)[None, None, :] - cols[:, None, None]  # (N, 1, W)
        beyond = jnp.maximum(jnp.abs(dr), jnp.abs(dc)) > half
        vertical = jnp.abs(dr) >= jnp.abs(dc)
        sides = jnp.stack([vertical & (dr < 0), vertical & (dr > 0),
                           ~vertical & (dc < 0), ~vertical & (dc > 0)], axis=1)
        sides = (sides & beyond[:, None]).astype(jnp.float32)              # (N, 4, H, W)
        count = jnp.sum(sides, axis=(2, 3))[..., None]                     # (N, 4, 1)
        total = jnp.einsum('nshw,nchw->nsc', sides, maps)
        mean = jnp.where(count > 0, total / jnp.maximum(count, 1.0), jnp.array([1.0, 0.0, 0.0]))

        crop = jnp.zeros((n, 3, size + 2, size + 2), jnp.float32)
        crop = crop.at[:, :, 0, 1:-1].set(mean[:, 0, :, None])
        crop = crop.at[:, :, -1, 1:-1].set(mean[:, 1, :, None])
        crop = crop.at[:, :, 1:-1, 0].set(mean[:, 2, :, None])
        crop = crop.at[:, :, 1:-1, -1].set(mean[:, 3, :, None])
        crop = crop.at[:, :, 1:-1, 1:-1].set(core)
        return crop.reshape(n, -1)

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
            map_id=state.map_id,
            task_context=(jnp.concatenate([
                self._task_context(state), state.sweep_direction,
                jnp.where(state.fallback_goal[:, None] >= 0,
                          (jnp.stack([state.fallback_goal % self.grid_w,
                                      state.fallback_goal // self.grid_w], axis=-1) + .5)
                          * self.cell_size - state.robot_positions, 0.),
                jnp.full((self.num_robots, 1), self._revisit_weight(state)),
            ], axis=-1) if self.critic_context else jnp.zeros((self.num_robots, 0))),
            visit_counts=state.visit_counts,
            previous_visit=state.previous_visit,
            coverage_history=state.global_coverage_history,
            occupancy_history=state.global_occupancy_history,
            visit_history=state.global_visit_history,
            kinematics_history=state.global_kinematics_history,
            humans_history=state.global_humans_history,
            context_history=state.global_context_history,
            previous_visit_history=state.global_previous_visit_history,
            history_valid=state.global_history_valid,
            crops=(self._actor_crops(state) if self.critic_crops
                   else jnp.zeros((self.num_robots, 0), jnp.float32)),
        )

    def critic_inputs(self, gs: GlobalState) -> tuple[jax.Array, jax.Array]:
        grid, vec = self._critic_inputs(gs)
        if self.critic_crops:
            # Every V_i reads its own actor crop, then all robots' crops in id
            # order: the same memory-based view the actors act on.
            joint = gs.crops.reshape(*gs.crops.shape[:-2], 1, -1)
            vec = jnp.concatenate([vec, gs.crops, jnp.broadcast_to(
                joint, (*gs.crops.shape[:-1], joint.shape[-1]))], axis=-1)
        if self.critic_coverage:
            ratio = (jnp.sum(gs.coverage, axis=(-2, -1))
                     / jnp.maximum(self.free_totals[gs.map_id], 1.0))
            vec = jnp.concatenate([vec, jnp.broadcast_to(
                ratio[..., None, None], (*vec.shape[:-1], 1))], axis=-1)
        return grid, vec

    def _critic_inputs(self, gs: GlobalState) -> tuple[jax.Array, jax.Array]:
        if self.critic_stack > 1:
            # Leading dimensions may be (E,) or (T,E). Keep time ordered,
            # then flatten history into channels/features for the value CNN.
            occ = gs.occupancy_history                     # (..., K, N, H, W)
            me = occ[..., :, :, None, :, :]
            rest = (occ.sum(axis=-3, keepdims=True) - occ)[..., :, :, None, :, :]
            wall = self.wall_grids[gs.map_id][..., None, None, None, :, :]
            wall = jnp.broadcast_to(wall, me.shape)
            cov = jnp.broadcast_to(gs.coverage_history[..., :, None, None, :, :], me.shape)
            # Each recipient sees a separate count map for every robot.
            counts = jnp.log1p(gs.visit_history) / np.log1p(self.max_steps)
            counts = jnp.broadcast_to(counts[..., :, None, :, :, :],
                (*occ.shape[:-4], self.critic_stack, self.num_robots,
                 self.num_robots, self.grid_h, self.grid_w))
            grid = jnp.concatenate([wall, cov, me, rest, counts], axis=-3)
            grid = grid * gs.history_valid[..., :, None, None, None, None]
            grid = jnp.swapaxes(grid, -5, -4).reshape(
                *occ.shape[:-4], self.num_robots, self.critic_channels,
                self.grid_h, self.grid_w)

            kin = gs.kinematics_history
            joint = kin.reshape(*kin.shape[:-2], 1, 6 * self.num_robots)
            humans = gs.humans_history.reshape(*kin.shape[:-2], 1, 2 * self.num_humans)
            joint = jnp.broadcast_to(jnp.concatenate([joint, humans], axis=-1),
                (*kin.shape[:-1], 6 * self.num_robots + 2 * self.num_humans))
            previous = gs.previous_visit_history
            valid = jnp.all(previous >= 0, axis=-1, keepdims=True)
            coords = (previous.astype(jnp.float32) + .5) / jnp.array([self.grid_w, self.grid_h])
            previous = jnp.concatenate([valid.astype(jnp.float32),
                                        jnp.where(valid, coords, 0.)], axis=-1)
            previous = previous.reshape(*kin.shape[:-2], 1, 3 * self.num_robots)
            previous = jnp.broadcast_to(previous, (*kin.shape[:-1], 3 * self.num_robots))
            vec = jnp.concatenate([kin, joint, gs.context_history, previous], axis=-1)
            vec = vec * gs.history_valid[..., :, None, None]
            vec = jnp.swapaxes(vec, -3, -2).reshape(
                *kin.shape[:-3], self.num_robots, self.critic_vec_dim)
            return grid, vec

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
        if self.critic_context:
            vec = jnp.concatenate([vec, gs.task_context], axis=-1)
        return grid, vec

    def get_info(self, state: EnvState) -> dict:
        covered = jnp.sum(state.coverage_grid)
        room_cov = jnp.sum(state.coverage_grid[None, :, :] * self.room_masks[state.map_id], axis=(1, 2))
        free_total = self.free_totals[state.map_id]
        
        info = {
            'executed_action': jnp.stack([2. * state.robot_velocities[:, 0] / self.v_max - 1.,
                                          state.robot_velocities[:, 1] / self.omega_max], axis=-1),
            'teacher_mask': (state.fallback_used & state.robot_alive
                             & (state.wall_hits + state.robot_hits + state.human_hits == 0)
                             & jnp.any(jnp.abs(state.robot_velocities) > 1e-3, axis=-1)).astype(jnp.float32),
            'coverage_ratio':       covered / free_total,
            'recoverage':           jnp.where(covered > 0, state.cell_entries / jnp.maximum(covered, 1.0), 1.0),
            'cell_entries':         state.cell_entries,
            'fallback_active':      state.fallback_active,
            'fallback_used':        state.fallback_used,
            'fallback_activated':   state.fallback_activated,
            'fallback_count':       state.fallback_count,
            'fallback_stagnation':  state.fallback_stagnation,
            'fallback_sequence_used': state.fallback_sequence_used,
            'fallback_safety_override': state.fallback_safety_override,
            'fallback_rejection_flags': state.fallback_rejection_flags,
            'revisit_streak':       state.revisit_streak,
            'discovery_streak':     state.discovery_streak,
            'covered_cells':        covered,
            'total_cells':          jnp.float32(free_total),
            'step':                 state.step_count,
            'robots_alive':         state.robot_alive,
            'num_robots_alive':     jnp.sum(state.robot_alive),
            'wall_hits':            state.wall_hits,
            'robot_hits':           state.robot_hits,
            'human_hits':           state.human_hits,
            'wall_collision_rate':  jnp.mean(state.wall_hits),
            'robot_collision_rate': jnp.mean(state.robot_hits),
            'human_collision_rate': jnp.mean(state.human_hits),
            'complete':             (covered >= free_total - 0.5).astype(jnp.float32),
            'timeout':              (state.step_count >= self.max_steps).astype(jnp.float32),
        }
        for ri in range(self.num_rooms):
            info[f'room_{ri}_ratio'] = room_cov[ri] / jnp.maximum(self.room_totals[state.map_id, ri], 1.0)
        return info
