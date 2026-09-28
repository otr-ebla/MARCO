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
   Robots learn coverage from scratch with **MAPPO** (Multi-Agent Proximal Policy Optimization) under CTDE. There is no planner in the loop: each actor sees only its own observation and commands its own linear and angular velocity, so the same policy runs on any floor plan without knowing the layout in advance.

2. **Coordination from the reward alone:**
   Spreading, handing over remaining work and avoiding redundant sweeps are not scripted. They emerge from a layout-agnostic reward built on what each robot knows (see *Reward* below).

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
  covered cells earns about +0.38 per step; parking or spinning costs −1.02;
  driving away costs −1.42. A robot whose area is done is therefore pushed to
  where teammates still have work.
* **Closed loops earn nothing.** Both potentials of a step use the same field, so
  progress telescopes to zero over any loop and covering a cell never looks like
  a loss of progress.
* **Spreading** is a constant pressure against staying close; who yields on a
  meeting is learned, not scripted.

Old checkpoints keep the reward they were trained with (`local_coverage_v1`),
which is stored in the checkpoint and restored by both evaluators.

### Actor with memory

Select `--policy-mode end-to-end-memory` to add a GRU (width `model.hidden_size`,
128 by default) after the local observation encoder.

```bash
python -m src.train_marl --policy-mode end-to-end-memory --obs-mode memory_comm \
  --save-dir checkpoints/e2e_memory_humans8 --humans 8 --envs 8 --maps 64 \
  --backend cuda --no-wandb
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
`training_log_e2e_memory.csv`. Both evaluators detect recurrent checkpoints:

```bash
python -m src.test_visual \
  --checkpoint checkpoints/e2e_memory_humans8/checkpoint_e2e_memory.pkl \
  --humans 8 --backend cpu

python -m src.evaluate_policies \
  --checkpoint checkpoints/e2e_memory_humans8/checkpoint_e2e_memory.pkl
```

Resume a trained actor for a fixed fine-tuning phase with
`--additional-updates`. The reward weights are persisted in every checkpoint.
