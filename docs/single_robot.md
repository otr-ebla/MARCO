# Single-robot coverage: learned policies vs. Boustrophedon Cellular Decomposition

A simplified version of the multi-robot problem, used to study the learning
question in isolation before adding teammates and communication.

## Problem

* **One robot** (differential drive, radius 0.20 m, `v_max` 1 m/s, `omega_max` 1 rad/s).
* **Smaller rooms:** procedural 8 m × 6 m layouts (16 × 12 cells of 0.5 m), BSP depth 2,
  so up to four rooms joined by doorways (≈175 free cells). The multi-robot setup uses
  12 m × 8 m and depth 3. Set with `env.map_width`, `env.map_height`, `env.map_depth`.
* **Goal:** cover every free cell in the shortest time, without knowing the global map.
  An episode succeeds when every free cell is covered, and times out after 4000 steps
  (400 s).
* **Observation:** the same local observation as the multi-robot memory actor, without
  the communication slots (`comm_slots: 0`):
  pose and heading, `(v, omega)`, previous visited cell and sweep direction, a 70-ray
  lidar, and a 7 × 7 directional crop of the robot's own lidar-built belief map
  (`[obstacle, covered, robots]` inside a ring that summarises each side beyond the
  crop). The global map is never an actor input. The MAPPO critic is centralised
  (privileged global state) during training only.
* **Reward:** the sequential coverage reward of `mappo_memory.yaml` with the teammate
  terms (`spread_weight`, `los_spread_weight`) set to zero.

Everything is in `config/single_robot.yaml`.

## Compared methods

| Name | What it is | Map knowledge | How to get it |
|---|---|---|---|
| **BCD** | Classical Boustrophedon Cellular Decomposition: free cells split into rectangles, each swept in lanes, rectangles chained by geodesic distance into one tour, tracked by a feedback controller with a DWA safety filter. | Full map known in advance (centralised, offline plan) | `evaluate_policies --bcd tour` |
| **IL** | Actor trained only by imitation of BCD (DAgger, β decaying 1 → 0). | Local observation only | `pretrain_bc`, file `checkpoint_bc_actor.pkl` |
| **RL** | Actor trained only by MAPPO (one agent: PPO with a centralised critic). | Local observation only | `train_marl` from scratch |
| **IL+RL** | IL actor, then PPO fine-tuning with a BC term towards BCD that decays linearly to 0 over 300 updates. | Local observation only | `train_marl --resume …/checkpoint_bc.pkl --bc-coef 0.5` |

The IL teacher is the same BCD plan used as the baseline (`pretrain.expert.target_rule: tour`).
It acts on the true map, so DAgger only imitates steps where the target cell lies inside
the actor's 5 × 5 crop (`local_labels: true`), which is about 82% of an expert episode
(≈39% in the first 200 steps, which are mostly the transit to the tour start, ≈87% afterwards).
The memory-only alternative (`target_rule: local`, nearest uncovered cell) is a poor
teacher with one robot: it keeps reversing and times out at ≈76% coverage.

RL and IL+RL use the same number of PPO updates (`train.total_updates`). IL+RL resumes
from update 0 with a critic already fitted to the IL policy.

## Running

All four methods and the comparison in one go:

```bash
bash scripts/single_robot_experiments.sh --backend cuda
```

or step by step (GRU actor; use `--policy-mode end-to-end` for the feed-forward one):

```bash
python -m src.train_marl  --config config/single_robot.yaml --policy-mode end-to-end-memory \
    --save-dir checkpoints/single/rl
python -m src.pretrain_bc --config config/single_robot.yaml --policy-mode end-to-end-memory \
    --save-dir checkpoints/single/il
python -m src.train_marl  --config config/single_robot.yaml --policy-mode end-to-end-memory \
    --resume checkpoints/single/il/checkpoint_bc.pkl --bc-coef 0.5 \
    --save-dir checkpoints/single/il_rl

python -m src.evaluate_policies --config config/single_robot.yaml --humans 0 --policy-only \
    --episodes 1000 --map-seed 1000 --output-dir evaluation_results/single_robot --bcd tour \
    --checkpoint checkpoints/single/il/checkpoint_bc_actor.pkl          --label IL \
    --checkpoint checkpoints/single/rl/checkpoint_e2e_memory.pkl        --label RL \
    --checkpoint checkpoints/single/il_rl/checkpoint_e2e_memory.pkl     --label IL+RL
```

Evaluation notes:

* `--map-seed 1000` draws a map bank different from training (`map_seed: 0`), so the
  comparison is on unseen layouts; drop it to evaluate on the training maps.
* `--policy-only` disables the recovery fallback, so the learned policy acts alone
  (BCD never uses it). `--compare-policy-only` reports both.
* `--humans 0` keeps the static problem; dynamic pedestrians can be added later with `--humans N`.
* Watch BCD with `python -m src.visualize_policy --config config/single_robot.yaml --expert --target-rule tour`.

Outputs: `episodes.csv` (one row per episode), `summary.csv` (success rate, completion
time over successful episodes, coverage, revisit rate, collision counts, with std and
95% CI), `metadata.json`. Plots: `python -m src.plot_evaluation_results`.

## Reference: BCD on this setup

32 episodes on unseen layouts (`--map-seed 1000`), CPU:

| Policy | Success | Coverage | Completion time | Wall contacts |
|---|---|---|---|---|
| BCD (tour) | 100% | 100% | 203.7 s (≈2040 steps) | 0 |
| BCD (local, memory-only) | 0% (timeout) | 76% | — | 0 |

The learned policies have not been trained yet; their rows come from the commands above.
