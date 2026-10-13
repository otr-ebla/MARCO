# Single-robot coverage: optimal method, IL, RL and IL+RL

A simplified version of the multi-robot problem, used to study the learning
question in isolation before adding teammates and communication.

## Problem

* **One robot** (differential drive, radius 0.20 m, `v_max` 1 m/s, `omega_max` 1 rad/s).
* **Smaller rooms:** procedural 8 m × 6 m layouts (16 × 12 cells of 0.5 m), BSP depth 2,
  so up to four rooms joined by doorways (≈175 free cells). Set with `env.map_width`,
  `env.map_height`, `env.map_depth`.
* **Start cells:** the robot spawns on one of `env.spawn_cells: 8` fixed cells per map
  (drawn from `map_seed`), so the optimal path of every episode is planned offline.
* **Goal:** cover every free cell in the shortest time, without knowing the global map.
  An episode succeeds when every free cell is covered, and times out after 4000 steps
  (400 s). Wall contact does not end the episode.
* **Observation:** the multi-robot memory actor's observation without the
  communication slots (`comm_slots: 0`): pose and heading, `(v, omega)`, previous
  visited cell and sweep direction, a 70-ray lidar, and a 7 × 7 directional crop of the
  robot's own lidar-built belief map. The global map is never an actor input; the
  MAPPO critic is centralised (privileged global state) during training only.
* **No recovery fallback** (`fallback_enabled: false`): the policy alone drives the robot,
  in training and in evaluation.

Everything is in `config/single_robot.yaml`.

## Reward

`reward_mode: simple`, per step:

    r = alpha * (new cells covered) - revisit_cost * (re-entries into visited cells)
        - tau - wall_kappa * (wall contact) + completion_bonus * (map complete)

| Term | Value | Role |
|---|---|---|
| `alpha` | 1.0 per new cell | the task: cover cells |
| `revisit_cost` | 0.5 per entry into a cell the robot's centre has already visited | do not re-cover |
| `tau` | 0.01 per step | finish sooner |
| `wall_kappa` | 0.1 per step in contact | do not push into walls |
| `completion_bonus` | 10 once | finish the whole map |

None of the multi-robot shaping terms (progress, loiter, sweep and boustrophedon bonuses,
revisit streaks, axis alignment, spread, fallback cost) is used. `train.reward_scale` is 1.

## Compared methods

| Name | What it is | Map knowledge | How to get it |
|---|---|---|---|
| **Optimal** | The exact optimal coverage path of the spawn cell (MILP below), tracked by a differential-drive controller (waypoint on the next cell, feedback law, lidar safety filter). | Full map known in advance | `evaluate_policies --optimal` |
| **IL** | Actor trained only by DAgger imitation of Optimal (β decaying 1 → 0). | Local observation only | `pretrain_bc`, file `checkpoint_bc_actor.pkl` |
| **RL** | Actor trained only by MAPPO (one agent: PPO with a centralised critic). | Local observation only | `train_marl` from scratch |
| **IL+RL** | IL actor, then MAPPO with a BC term towards Optimal decaying linearly to 0 over 300 updates. | Local observation only | `train_marl --resume …/checkpoint_bc.pkl --bc-coef 0.5` |

RL and IL+RL use the same number of PPO updates (`train.total_updates`). IL+RL resumes
from update 0 with a critic already fitted to the IL policy.

### The optimal teacher (`src/envs/optimal_expert.py`)

* The optimal path depends on the start cell and takes seconds to minutes to solve, so it
  cannot run inside the jitted rollout. `OptimalPlanBank` solves every (map, spawn cell)
  once, in parallel processes, and caches the paths in `pretrain.optimal.cache_dir`
  (keyed by the maps, the spawn cells and the planner settings).
* `OptimalExpert` tracks the path: its target is the first cell of the path, in
  first-visit order, that the robot's memory does not hold as covered. It therefore labels
  every state DAgger visits, including states off the path, with a `(v, omega)` command.
* With `local_labels: true`, a label is used only when its target lies inside the actor's
  crop.

## Running

```bash
bash scripts/single_robot_experiments.sh --backend cuda
```

or step by step:

```bash
python -m src.envs.optimal_expert --config config/single_robot.yaml                   # training plans
python -m src.envs.optimal_expert --config config/single_robot.yaml --map-seed 1000   # evaluation plans
python -m src.train_marl  --config config/single_robot.yaml --policy-mode end-to-end-memory \
    --save-dir checkpoints/single/rl
python -m src.pretrain_bc --config config/single_robot.yaml --policy-mode end-to-end-memory \
    --save-dir checkpoints/single/il
python -m src.train_marl  --config config/single_robot.yaml --policy-mode end-to-end-memory \
    --resume checkpoints/single/il/checkpoint_bc.pkl --bc-coef 0.5 --save-dir checkpoints/single/il_rl

python -m src.evaluate_policies --config config/single_robot.yaml --humans 0 \
    --episodes 1000 --map-seed 1000 --output-dir evaluation_results/single_robot --optimal \
    --checkpoint checkpoints/single/il/checkpoint_bc_actor.pkl      --label IL \
    --checkpoint checkpoints/single/rl/checkpoint_e2e_memory.pkl    --label RL \
    --checkpoint checkpoints/single/il_rl/checkpoint_e2e_memory.pkl --label IL+RL
```

* `--map-seed 1000` draws a map bank different from training (`map_seed: 0`), so the
  comparison is on unseen layouts.
* Watch a policy: `python -m src.visualize_policy --config config/single_robot.yaml
  --checkpoint checkpoints/single/il_rl/checkpoint_e2e_memory.pkl`; the optimal expert:
  `--expert --target-rule optimal` instead of `--checkpoint`.

Outputs: `episodes.csv` (one row per episode), `summary.csv` (success rate, completion
time over successful episodes, coverage, revisit rate, collision counts, with std and
95% CI), `metadata.json`. Plots: `python -m src.plot_evaluation_results`.

## The optimal method: exact coverage path (MILP)

`src/envs/optimal_coverage.py` computes the **optimal** cell sequence for one robot
that knows the whole map. It is the lower reference for every other method: no
method can cover the map in fewer moves from the same start cell.

Model: the robot is holonomic and point-like for coverage (a cell is covered when the
robot's centre is in it); heading and dynamics are not modelled. A path starts at the
robot's cell, moves between 4-adjacent free cells, visits every free cell, and may end
anywhere. Its cost is

    J = alpha * T + beta * R,    T = moves,  R = sum_i max(0, visits_i - 1).

Every free cell is visited at least once, so `R = T + 1 - |V|` for every coverage
path: with any `alpha, beta >= 0` the shortest path is also the one with the fewest
re-coverages, and one optimum minimises both.

Method: the integer program for grid covering walks (DFJ subtour elimination; Arkin
et al. 2005; Fekete & Krupke 2019), solved exactly with HiGHS:

* `y_ij` in {0, 1, 2} traversals of each edge, `e_i` end cell, `p_i` visits of each
  cell, `r_i >= p_i - 1` re-coverages; degree parity
  `sum_j y_ij + [i = s] + e_i = 2 p_i` makes the edges one walk from the start;
  `p_i >= 1` covers every cell.
* Connectivity cuts `y(delta(S)) + e(S) >= 2` (start not in S), added lazily: solve,
  find the components detached from the start, cut them, solve again. The pair and
  2 × 2 block cuts are added up front.
* `formulation='directed'` keeps one integer variable per transition `x_ij` with flow
  conservation and cuts `sum_{i not in S, j in S} x_ij >= 1`. It gives the same optimum
  but HiGHS is much slower on 175-cell maps (both orientations of every cycle are
  separate solutions); it is kept for checking.
* Lower bound: the grid is bipartite and a walk alternates colours, so
  `T >= max(|V| - 1, 2A - 2, 2B - 1)` (A cells of the start's colour, B of the
  other). When the bound is `|V| - 1`, a pruned depth-first search for a Hamiltonian
  path runs first; a path found meets the bound and is optimal without the MILP.

Each result carries a status: `OPTIMAL` (proven), `FEASIBLE` (MILP solved with a
positive `mip_gap`), `TIME_LIMIT` (a valid path and a certified lower bound, so
`gap` is the worst-case loss) or `INFEASIBLE` (free cells unreachable from the start),
plus cost, moves, re-coverages, solver time and gap. `validate_path` checks any path
against the grid alone (start, free cells, adjacency, full coverage, cost); the
tests use it and compare every small case with a brute-force search.

```python
from src.envs.optimal_coverage import CostWeights, plan_coverage, validate_path

grid = [".......",
        "...#...",
        "...#...",
        "......."]
result = plan_coverage(grid, (1, 2), CostWeights(alpha=1.0, beta=1.0), time_limit=300)
result.status, result.moves, result.revisits, result.path[:4]
# ('OPTIMAL', 25, 0, [(1, 2), ...])
assert validate_path(result.path, grid, (1, 2)).valid
```

On the environment's maps (cells as (row, col) = (floor(y / 0.5), floor(x / 0.5))):

```bash
# optimal plans for 8 layouts x 4 random starts, one CSV row each
python -m src.envs.optimal_coverage --config config/single_robot.yaml --maps 8 --starts 4 \
    --time-limit 300 --output evaluation_results/single_robot/optimal_paths.csv

# watch the robot follow the optimal path (holonomic, v_max along the cell centres)
python -m src.visualize_policy --config config/single_robot.yaml --optimal
```

Reference run (8 training layouts × 4 random starts, `--time-limit 300`, CPU): all
32 plans **OPTIMAL**, 174 moves on average (≈87 s at 1 m/s), 0 re-coverages in 8
cases, 1 in 19, 2 in 5; solver time median 26 s, max 59 s (3 settled instantly by
the Hamiltonian search). These times are for the holonomic model, not for the
differential-drive robot (see below).

In the viewer the planned path is drawn under the robot (the travelled part
darker) and the status appears in the HUD. The robot moves `v_max · dt` per step
along the cell centres, so an episode takes `5 · T - 2` steps at 0.5 m cells and
1 m/s (the last cell counts as covered as soon as the centre enters it).

### Optimal path on the differential-drive robot

Plan banks (8 spawn cells × 64 maps, `time_limit` 300 s, 32 processes):

| Bank | Plans | Status | Mean moves | Mean revisits | Solve median | Wall time |
|---|---|---|---|---|---|---|
| training (`map_seed` 0) | 512 | all OPTIMAL | 173.9 | 0.81 | 16.6 s | 8.6 min |
| evaluation (`map_seed` 1000) | 512 | all OPTIMAL | 173.9 | 0.80 | 21.2 s | 11.5 min |

The MILP minimises moves for a holonomic point. The differential-drive robot
(`omega_max` 1 rad/s) slows down to turn, and the optimal paths turn often (106–122
turns on a map where a boustrophedon sweep needs 32–40). Tracked in the environment,
the optimal expert covers every map without contacts in ≈247–250 s on average,
against 87 s for the holonomic plan. A turn-aware cost (`alpha T + beta R + gamma turns`,
which needs the heading in the state) would make it optimal for this robot.

## Results

Pending: produced by `scripts/single_robot_experiments.sh`.
