# MARCO: Multi-Agent Reinforcement Learning for Multi-Robot Coverage Path Planning with Dynamic Obstacles

[![JAX](https://img.shields.io/badge/JAX-Enabled-orange?style=flat-square&logo=google)](https://github.com/google/jax)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-zone)

![Simulated Environment](simulated_environment.gif)

**mrCPP** addresses **multi-robot coverage path planning (CPP)**: a team of robots must jointly visit every reachable cell of an unknown or partially known environment as efficiently as possible, while avoiding collisions with static structures and **dynamic obstacles** (e.g., pedestrians moving through the workspace).

The project explores and compares modern machine-learning approaches to this control problem—ranging from standard model-free MARL to advanced latent world-model methods—to identify optimal architectures for decentralized, scalable, obstacle-aware coverage.

---

## 🧭 Task Definition & Decentralized Execution

Each robot operates under strict **decentralized execution**, relying exclusively on local, egocentric observations:
* **2D LiDAR Scan:** Local range readings capturing static walls and dynamic humans.
* **Ego Velocity:** Linear and angular velocity of the individual robot.
* **Local Coverage Sub-Grid:** A local window of the global coverage map marking visited vs. unvisited cells.
* **Local Teammate Matrix:** Relative state (position/velocity) of nearby teammates within sensing range.

### Environment & Dynamics
* **Unknown Floor Plans:** Episodes take place in randomly generated 2D indoor layouts unknown to the robots at initialization.
* **Dynamic Obstacles:** Simulated using the **Headed Social Force Model (HSFM)**, producing realistic, reactive pedestrian trajectories (goal attraction, inter-agent and wall repulsion) rather than scripted paths.

### Centralized Training, Decentralized Execution (CTDE)
During training, the central critic leverages privileged global state (full robot states, pedestrian trajectories, and the complete environment coverage grid) to provide accurate credit assignment without requiring global observations at test time.

---

## 🧠 Proposed Approach & Novelty

1. **End-to-end MARL via MAPPO:**
   Robots learn velocity control with **MAPPO** (Multi-Agent Proximal Policy Optimization) under CTDE. The sequential actor observes its personal map and previously visited cell, and chooses where to go itself. Recovery demonstrations provide a training-only imitation loss; no next-cell target or waypoint is supplied to the actor. The actor commands linear and angular velocity; recovery can temporarily take control when it gets stuck. Evaluate with recovery disabled to measure the learned policy's independent performance.

2. **Coordination from the reward alone:**
   Spreading and avoiding redundant sweeps are learning objectives, encouraged by the reward (see *Reward* below). Actors choose velocities directly; successful coordination must be measured in evaluation.

3. **Novelty — Dynamic Human Obstacles:**
   The environment includes dynamic, unpredictable obstacles—specifically modeled as **humans moving through the workspace**. Traditional coverage algorithms assume static environments. Our MARL agents must learn to balance total environment coverage with the safety-critical need to avoid moving pedestrians.

---

## ⚙️ Implementation Constraints

* **Framework:** 100% implemented in **JAX**, optimized for high-performance **CUDA** execution.
* **Backend selection:** `train_simple.py` picks its device automatically — Apple **Metal** (via `jax-metal`) on Apple Silicon, otherwise **CUDA** when an NVIDIA GPU is visible, otherwise **CPU**. Override with `--backend {auto,metal,cuda,cpu}`.

> **Note on Apple Silicon:** `jax-metal` is experimental. On an M1 Pro the Metal
> backend measures ~2× *slower* than the CPU backend for the baseline config,
> because the networks are small and the rollout is dominated by many tiny
> kernel dispatches. Prefer `--backend cpu` on Mac unless you scale the model up.

* **Monitoring:** training streams to **Weights & Biases** (configured in the
  `wandb:` block of the YAML, overridable with `--wandb-project`, `--wandb-name`,
  `--wandb-mode {online,offline,disabled}`, or switched off with `--no-wandb`).
  Run `wandb login` once first. The same metrics are always written to
  `<save-dir>/training_log.csv`, and a failed W&B init never stops training.

  Logged per `log_interval`: episode reward / coverage / length, the end-cause
  rates (`rate/completion`, `rate/timeout`, `rate/collision_end`), collision
  diagnostics split by cause (`collision/wall_*`, `collision/robot_*`, both as a
  per-robot-step rate and as a per-episode count), the PPO losses, policy sigma
  and the decayed learning rates.

### Policy benchmark

Evaluate one or more trained checkpoints for 1,000 episodes each. Scenarios use
eight humans by default (`--humans N` to change it):

```bash
python -m src.evaluate_policies --episodes 1000 --output-dir evaluation_results \
  --checkpoint checkpoints/e2e/checkpoint_e2e.pkl --label "E2E" \
  --checkpoint checkpoints/e2e_memory/checkpoint_e2e_memory.pkl --label "E2E + GRU"
```

The summary table reports mutually exclusive episode percentages: success /
coverage rate, robot-robot collision rate (RRCR), robot-wall collision rate
(RWCR), robot-human collision rate (RHCR), and timeout rate (TOR). These columns
sum to 100% for each policy. An unsuccessful episode with multiple collision
types is assigned to its most frequent collision type (ties prefer human, then
robot, then wall); an unsuccessful episode without contact is a timeout. The
table also reports mean simulated completion time over successful episodes only.

Completion time is recorded in both simulation steps and simulated seconds
(`NaN` for episodes that do not complete). The command writes per-episode data
(`episodes.csv`), means/standard deviations/95% confidence intervals
(`summary.csv`) and run metadata. Use `--help` for accelerator, batching, seed,
and horizon overrides.

Evaluation rollouts use 50 parallel environments by default (`--batch-size 50`).
Environment stepping, policy inference and episode transitions run in compiled
JAX scans on CUDA when available; CSV serialization happens on the CPU after
each rollout chunk.

---

## 🚀 Getting Started

Clone the repository and install dependencies:

```bash
git clone https://github.com/otr-ebla/mrCPP.git
cd mrCPP
pip install -r requirements.txt
```

---

### Training (CTDE)

```bash
python -m src.train_marl --policy-mode end-to-end --obs-mode memory_comm \
  --save-dir checkpoints/e2e
```

Each environment samples its layout independently from a procedural map bank
(`e2e_num_maps`, or `--maps`). Use `--obs-mode memory_comm` so that every robot
builds its own lidar map, shares it with teammates in `comm_radius`, and sees
those teammates: the spreading term of the reward is only learnable when the
teammates it charges for are observable. Outputs are `checkpoint_e2e.pkl`
(best), `checkpoint_e2e_latest.pkl` and `training_log_e2e.csv`.

#### Reward

Training uses `reward_mode: progress`, configured under `e2e_reward` in
`config/mappo_baseline.yaml`. Per robot and step:

| Term | Value |
|---|---:|
| Newly covered cell | +10 × (1 + 2 × team coverage) |
| Progress towards work | +2 per cell of approach |
| Travel off the horizontal/vertical axes | −0.5 × (error / 45°)² × speed / v_max |
| Loitering (no discovery) | −1 × (1 − progress / full-speed progress) |
| Teammate within 2.5 m | −0.2 × (1 − d / 2.5 m), per teammate |
| Each timestep | −0.02 |
| Wall / robot / human contact | −2 / −5 / −10 |
| Team completes the map | +200 × (1 + remaining time fraction) |

*Work* is the nearest free cell the robot **believes** uncovered: its own memory,
merged with what teammates shared within communication range. Distances are
geodesic over free cells, so progress is only paid along real paths. The
implicit target lives only in the reward; the actor never observes it.

* **Transit is paid, wandering is not.** Driving at full speed towards work over
  covered cells earns about +0.58 per step when aligned with an axis; parking or spinning costs −1.02;
  driving away costs −1.42. A robot whose area is done is therefore pushed to
  where teammates still have work.
* **The progress potential telescopes.** Both potentials of a step use the same field, so
  progress telescopes to zero over any loop and covering a cell never looks like
  a loss of progress.
* **Spreading** is a constant pressure against staying close; who yields on a
  meeting is learned, not scripted.

Old checkpoints keep the reward they were trained with (`local_coverage_v1`),
which is stored in the checkpoint and restored by both evaluators.

### Actor with memory

#### Sequential coverage with five-frame observations

`config/mappo_sequential.yaml` addresses the case where the local 5×5 crop is
already covered but work remains elsewhere. It adds:

* A CNN over the robot's **complete personal map**: known obstacles, believed
  coverage, known cells, self and currently observable teammates. Unknown map
  geometry and distant teammate positions are not exposed to the actor.
* Grid-aligned walls with `env.wall_cells: 1`: every interior wall occupies a
  full cell band (0.5 m here), so physics, lidar, occupancy and planning agree.
  Doorways and their approaches remain open during recursive room generation.
  `wall_cells: 0` retains the old thin-wall layouts for older checkpoints.
* `env.history_cell: previous_visit`: a valid flag and relative position of
  the cell visited immediately before the current cell, including revisits.
  This history changes only on accepted cell entries, persists while dwelling,
  and resets at episode boundaries. The actor also keeps its past sweep direction.
  It receives no planned next cell, target bearing or route distance: destination
  selection must be learned from map memory and history. `critic_context: true`
  supplies time and control/reward history to the centralized critic only.
* `env.critic_stack: 5`: the centralized critic receives five ordered global
  frames. Each frame contains walls, team coverage, each robot's position,
  and a separate visit-count map for **every** robot. The vector contains all
  robots' kinematics and their previous visited cells, including a validity
  marker for cells that have not been visited yet. Zero padding marks missing
  frames at episode start; completion and timeout clear the history. These
  visit-count maps and stacked global frames are critic inputs only.
* An ordered stack of the **last five observations**, including the current one
  (`env.observation_stack: 5`). With `dt: 0.1`, the oldest frame is 0.4 seconds
  behind the newest. The encoder shares weights across frames, then concatenates
  their features in time order. Missing frames at episode start are zero-filled;
  auto-reset clears only the finished environment's history. The stack works
  with either the feed-forward actor or the GRU and supplies short-term context,
  not a guarantee of long-term memory or successful coverage. Stacked training
  treats timeout as a terminal horizon for GAE, so values never cross a reset.
* `reward_mode: sequential`: +6 for a newly covered cell adjacent to that
  robot's previous discovery, another +4 when continuing the same grid direction.
  A boustrophedon lane change earns an extra +8: after at least two straight
  new-cell edges, discover one perpendicular adjacent cell and then a new cell
  in the opposite direction to the original run. A revisit breaks this pattern;
  turning in place preserves it but earns no discovery bonus.
* Revisit costs start high and decrease within each episode. With
  `phase = max(coverage_fraction, elapsed_steps / max_steps)`, the per-entry
  weight is `6 * (0.025 + 0.975 * (1 - phase)^2)`: 6 initially, about 1.61 at
  phase 0.5, and 0.15 at completion or timeout. Multiply by the consecutive
  revisit streak, capped at 3. New discoveries reset the streak; dwelling and
  rejected moves do not charge another entry. This discourages early overlap
  while allowing late transit through already cleaned rooms.
* Progress routes computed only through known free cells, toward reachable
  believed-uncovered cells, falling back to exploration frontiers when needed.
  Spreading is charged only for
  teammates represented in communication slots, with weight 0.5 within 2.5 m.
* A configurable pre-tanh standard-deviation floor of about 0.10. This prevents
  Gaussian sigma collapsing to 0.009; tanh saturation can still reduce action noise.
* `train.recovery_imitation_coef: 0.5`: a separate supervised loss fits
  `tanh(policy_mean)` to executed recovery commands. Only collision-free,
  nonstationary recovery commands are used. PPO ratios and entropy still use
  policy-controlled samples only; recovery actions are never relabelled as
  policy samples. Set the coefficient to zero to disable imitation. Logs include
  `recovery_imitation_loss` and `teacher_fraction`.

For the updated geometry, reward and previous-cell semantics, start a **new training run**:

```bash
python -m src.train_marl --config config/mappo_sequential.yaml \
  --policy-mode end-to-end-memory --backend cuda --no-wandb \
  --save-dir checkpoints/sequential_navigation
```

This configuration starts with 32 environments, 128 rollout steps and 8
minibatches. Five complete maps per observation substantially increase memory
use; GPU capacity has to be checked before scaling back to 512 environments.
These settings also collect fewer environment steps per update than the old
512-environment run: compare training budgets in environment steps, not updates.
The new reward encourages contiguous sweeps; it does not prescribe a complete
coverage path or guarantee map completion. Compare completion rate, revisits,
sweep efficiency and collisions on held-out map seeds after training.

```bash
python -m src.visualize_policy \
  --config config/mappo_sequential.yaml \
  --checkpoint checkpoints/sequential_navigation/checkpoint_e2e_memory_latest.pkl \
  --policy-only
```

Measure both assisted and independent coverage on the same evaluation seeds:

```bash
python -m src.evaluate_policies --config config/mappo_sequential.yaml \
  --checkpoint checkpoints/sequential_navigation/checkpoint_e2e_memory_latest.pkl \
  --humans 0 --episodes 100 --seed 123 --compare-policy-only
```

Training coverage and checkpoint rankings include recovery whenever enabled.
Judge progress using policy-only completion, contacts, revisits and recovery
usage, not assisted coverage alone. The full-memory actor retains the original
observation width, but previous-cell semantics and map geometry differ from the
old run. Existing checkpoints remain loadable with their original observations.

To visualize an old policy, use its existing checkpoint:

```bash
python -m src.visualize_policy \
  --checkpoint checkpoints/sequential_stack5/checkpoint_e2e_memory.pkl \
  --config config/mappo_sequential.yaml
```

This restores the old policy's observation semantics (last newly discovered cell)
and thin walls. Append `--wall-cells 1` to run those same weights on full-cell
wall layouts instead; this changes the environment, not the trained policy.
New checkpoints record `history_cell` and wall geometry explicitly.

Observation stack, personal-map settings, encoder sizes and exploration bounds
are saved in checkpoints and restored by both evaluators. Old checkpoints without
these settings retain their original single-frame actor. `checkpoint_e2e_memory.pkl`
is the selected **best** checkpoint; `_latest.pkl` contains the final update. A
file reporting update 140 is evaluating that saved policy, even after training
has reached update 2500.

The training line also reports `recoverage`, averaged over the same recent
completed episodes as `ep_cov`. For each episode it is **total team cell visits /
distinct covered cells**: 1.000 means no repeated coverage, 1.500 means 150 visits
over 100 distinct cells, 2.000 means two visits per covered cell on average.
The first accepted occupancy counts as a visit; subsequent visits require leaving
and entering a cell again. Standing still, rotation within a cell and blocked
collision attempts do not increase the count. Before any completed episode (or
with no covered cells), the neutral value is 1.000. Low recoverage should always
be read alongside coverage and completion, since an idle robot can also have 1.
CSV stores `mean_ep_recoverage`; W&B stores `episode/mean_recoverage`.

The stronger rewards and this metric preserve the stack-5 actor architecture,
so an existing stack-5 checkpoint can be resumed with the updated config. New
reward weights are saved in subsequent checkpoints; older checkpoints retain
their saved weights when evaluated. An already running process must be restarted
to use these code/config changes.

#### Spatial-memory actor (IL + RL)

`config/mappo_memory.yaml` gives the actor a two-level memory so it can return
to uncovered areas it left behind:

* **Spatial memory, read Neural-Map style** (`map_encoder: spatial_memory`).
  The robot's personal map already stores, exactly and permanently, every cell
  it knows and has covered (own lidar and coverage, OR-merged with teammates in
  range). `SpatialMemoryRead` re-centres it on the robot's cell in a
  `(2H-1) x (2W-1)` window, with derived *uncovered* (known free, not covered)
  and *inside-floor* channels; cells beyond the floor read as known walls. A
  strided CNN gives a global read (`map_embed`), and multi-head attention
  (`attention_heads` x `attention_dim`) from the robot's current features over
  every 2x2 block, keyed by its relative offset and distance, gives a context
  read: where the relevant uncovered space is. The old two-layer map CNN read
  absolute coordinates into 64 features.
* **Temporal memory**: a 256-unit GRU (`--policy-mode end-to-end-memory`)
  replaces the five-frame stack (`observation_stack: 1`) and keeps the
  robot's current intention between steps.

The BCD imitation phase supports the GRU: the replay keeps the rollout memory
every `pretrain.sequence_length` (32) steps and clones the actor on ordered
windows from that stored state, resetting it at episode ends (truncated BPTT).
Train a new actor; old checkpoints keep their architecture:

```bash
python -m src.pretrain_bc --config config/mappo_memory.yaml \
  --policy-mode end-to-end-memory --save-dir checkpoints/bc_memory
python -m src.train_marl --config config/mappo_memory.yaml \
  --policy-mode end-to-end-memory --resume checkpoints/bc_memory/checkpoint_bc.pkl \
  --save-dir checkpoints/bc_memory_rl
```

The actor has about 1.1 M parameters (0.4 M for the sequential actor). The
encoder settings are stored in the checkpoint, so `evaluate_policies` and
`visualize_policy` rebuild the same network.

#### Recurrent actor

Select `--policy-mode end-to-end-memory` to add a GRU (width `model.hidden_size`,
128 by default) after the local observation encoder.

```bash
python -m src.train_marl --policy-mode end-to-end-memory --obs-mode memory_comm \
  --save-dir checkpoints/e2e_memory_humans8 --humans 8 --envs 8 --maps 64 \
  --backend cuda --no-wandb
```

To run more environments than fit in one PPO update, split each epoch into
minibatches of environments with `--minibatches` (must divide `--envs`). On a
10 GB GPU, 64 environments per minibatch fit:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python -m src.train_marl \
  --policy-mode end-to-end-memory --obs-mode memory_comm \
  --envs 512 --minibatches 8 --backend cuda --save-dir checkpoints/e2e_memory_E512
```

Train from scratch: feed-forward checkpoints have incompatible actor parameters.
Each robot has independent memory, preserved between rollouts and reset on
completion or timeout. PPO replays complete ordered rollouts and differentiates
through the GRU over `train.rollout_steps` timesteps. Starting memory is detached
for truncated backpropagation. The critic stays feed-forward and receives
privileged global state during training.

The recurrent mode treats the episode step budget as a terminal horizon for GAE,
preventing bootstrapping or advantage propagation across an environment reset.
A resumed checkpoint restarts environments and memory; it does not restore a
mid-episode simulator state.

Outputs: `checkpoint_e2e_memory.pkl` (best),
`checkpoint_e2e_memory_latest.pkl` (end of training), and
`training_log_e2e_memory.csv`. Both the visualizer and the evaluator detect
recurrent checkpoints:

```bash
python -m src.visualize_policy \
  --checkpoint checkpoints/e2e_memory_humans8/checkpoint_e2e_memory.pkl \
  --humans 8

python -m src.evaluate_policies \
  --checkpoint checkpoints/e2e_memory_humans8/checkpoint_e2e_memory.pkl
```

Resume a trained actor for a fixed fine-tuning phase with
`--additional-updates`. The reward weights are persisted in every checkpoint.


### Memory-based recovery (training and evaluation)

The shared environment switches each robot to A* plus local DWA after seven
consecutive entries into covered cells, or 70 physics steps without discovering
a new cell (including stopping or spinning). Both thresholds are configurable
under `env` with `fallback_revisit_threshold` and `fallback_stall_steps`.
`fallback_enabled: false` disables recovery for ablations.

A* chooses the closest reachable remembered, uncovered cell by four-connected
path length. It routes through known free cells and respects remembered wall
edges, including thin walls between free cell centres. DWA follows successive
waypoints using acceleration-limited velocity samples, a 12-step prediction
horizon, and current lidar clearance. Its speed and turn acceleration limits are
`fallback_linear_accel` and `fallback_angular_accel`; the horizon is
`fallback_dwa_steps`. Recovery retains its target until reached, replans if
shared coverage or new wall observations invalidate it, and returns control to
the actor after reaching work. With no reachable remembered work, the actor
continues without an activation charge. The controller does not use the critic's
global coverage or unseen obstacle map. Recovery is a local heuristic; moving
obstacles can still delay it.

If DWA goes `fallback_dwa_stall_steps` (default 30) without reducing its
remaining A* route distance by at least 2 cm, recovery switches to a generated
sequence of physical `(linear_velocity, angular_velocity)` commands. The sequence
turns toward cell centres, drives, and brakes before turns. It buffers
`fallback_sequence_steps` commands (default 64), replenishes them along the route,
and caps forward speed at `fallback_sequence_speed` (default 0.4 m/s). New wall
observations reset the progress baseline so discovering a necessary detour does
not itself count as getting stuck.

Before each queued command, a guard checks that command and its braking motion
against the latest lidar, known free cells, and remembered wall edges. If the
sequence becomes unsafe, DWA supplies reactive avoidance or a stop, the queue is
discarded, and the sequence is replanned from the actual pose. Currently sensed
obstacles can cause a temporary route detour; they are not added to permanent
wall memory. A changed target, collision, or episode reset also invalidates queued
commands. This keeps execution responsive to moving obstacles, although persistent
blockages can still prevent arrival. Escalating from DWA to sequence control adds
no second activation charge. Both controllers remain excluded from the PPO
policy-gradient loss; valid commands can train the actor through the separate
recovery imitation loss described above.

Training logs report `policy` (fraction of robot-steps controlled by RL),
`sequence` (fraction executing queued commands), and `seq_blocked` (fraction where
the live guard interrupted a sequence). DWA's fraction is `1 - policy - sequence`.
The same fractions are saved to CSV and W&B. New controller settings are saved
with checkpoints; a running process needs restarting to load code changes.

`e2e_reward.fallback_cost` defaults to 10 and is charged once on the policy step
that triggers a successful activation. `revisit_cost` defaults to 1: consecutive
covered-cell entries incur -1, -2, -3, and so on, in every reward mode. Discovering
a new cell resets the streak. Dwelling and rejected moves do not count as cell
entries. The sequential configuration overrides this with the decaying, capped
schedule above; defaults (`revisit_end_fraction: 1`, `revisit_streak_cap: 0`)
preserve the old constant-weight behavior. PPO excludes overridden actions from
its policy-gradient loss and entropy bonus;
the critic still learns from all transitions and recovery costs propagate to
preceding policy actions through GAE.

Known cells, covered cells, and observed blocked edges merge by logical OR when
robots are strictly less than `comm_radius` (default 3 m) apart. Memory uses dense
JAX bitmaps with the same cell-key union semantics as a sparse dictionary;
exchange is simultaneous and single-hop per physics step. Recovery maintains
this memory even for legacy actor observations. Controller settings and reward
weights are saved in new training checkpoints and restored by evaluation and
visualisation. Old checkpoints without controller settings use the supplied
configuration and these environment defaults.

MAPPO implements CTDE. The sequential configuration stacks five global frames
for each critic, including all robots' visit counts and previous visited cells.
The actor and recovery controller use their execution observations and
local/shared memory. IPPO deliberately uses a local critic instead.

The `axis_alignment_bonus` rewards actual horizontal or vertical displacement in
progress and sequential reward modes. It varies smoothly from zero on diagonals
to its maximum on either axis, scaled by speed and positive progress towards work.
Stopping, spinning, blocked moves and moving away earn no alignment bonus. Older
checkpoints without this weight retain a zero bonus; train with the updated
configuration to learn the preference. Existing lane-reversal bonuses still apply.

The `axis_alignment_cost` penalizes the angular error between each robot's actual
displacement and the nearest horizontal or vertical direction, in every reward mode.
Moving at 30° or 60° is a 30° error, and 45° is the maximum. The cost is
`axis_alignment_cost * (error / 45°)² * speed / v_max`: squaring keeps it smooth at
the axes, so small drifts are barely penalized, and rotating in place, blocked
moves and dead robots pay nothing.

Boustrophedon sweeps are shaped on the cell grid in every reward mode. The
preferred direction is the current lane direction, or the reversal right after
a one-cell lane shift; its next cell is *open* when the robot's own memory
holds no wall, wall edge or coverage there:

| Event | Reward |
|---|---:|
| New cell straight ahead in the preferred direction | +`sweep_straight_bonus` (3) |
| New perpendicular neighbour once the next cell is not open (lane shift) | +`sweep_turn_bonus` (3) |
| Leaving the lane head into any other cell while the next cell is open | −`sweep_break_cost` (5) |

With `sweep_obs: true` (memory_comm only) the actor also observes the preferred
direction and whether its next cell is open, three values before lidar.

A recovery activation keeps control until its goal cell is covered, by the robot
or a teammate; uncovered cells discovered on the way no longer release it.

In the visualizer, robots and their lidar rays turn gray while fallback is active
and return to their individual colors when policy control resumes.
