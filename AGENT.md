# mrCPP — Multi-Robot Coverage Path Planning with Dynamic Obstacles

## Intent

mrCPP addresses **multi-robot coverage path planning (CPP)**: a team of robots must
jointly visit every reachable cell of an unknown or partially known environment as
efficiently as possible, while avoiding collisions with static structure and with
**dynamic obstacles** (people moving through the workspace). The project explores and
compares several modern machine-learning approaches to this control problem, from
standard model-free MARL to latent world-model methods, in order to identify the best
architecture for decentralized, scalable, obstacle-aware coverage.

## Task Definition

### Per-robot observation (actor input, decentralized)

Each robot receives a local, egocentric observation composed of:

- **2D LiDAR scan** — local range readings around the robot, encoding nearby static
  structure and dynamic obstacles (people).
- **Ego velocity** — the robot's own linear/angular velocity.
- **Local coverage sub-grid** — a local window of the global coverage map centered on
  the robot, marking visited vs. unvisited cells.
- **Local teammate matrix** — relative state (e.g. position/velocity) of nearby
  teammate robots within sensing range, zero-filled where no teammate is present.

This observation must be sufficient for decentralized execution: at deployment, each
robot acts only on what it locally perceives.

## Environment

Each episode takes place in a **randomly generated 2D indoor floor plan** (rooms,
corridors, walls). The layout, including wall placement, is **unknown to the robots at
the start of the episode** — coverage and obstacle avoidance must be performed under
partial observability, relying on the onboard LiDAR and the incrementally built local
coverage sub-grid rather than a prior map.

Dynamic obstacles are **humans** moving through the floor plan, simulated with the
**Headed Social Force Model (HSFM)**: each pedestrian has a body orientation coupled to
its motion, and moves under social-force dynamics (goal-attraction, inter-agent
repulsion, wall repulsion) rather than following a scripted or purely random path. This
produces realistic, locally reactive human trajectories that robots must perceive via
LiDAR and avoid while covering the space.

### Centralized critic input (training only)

Under CTDE (Centralized Training, Decentralized Execution), the critic has access to
privileged global state unavailable at execution time:

- Positions and velocities of **all** robots.
- Positions/trajectories of **all** dynamic obstacles (people).
- The **full coverage grid** (covered vs. uncovered cells) for the entire environment.

This global view lets the critic assign accurate credit to individual robot actions
without requiring any robot to observe the full environment at runtime.

## Candidate Methods

The project evaluates multiple modern approaches on this task, in increasing order of
architectural ambition:

1. **MAPPO with CTDE** — baseline model-free multi-agent PPO: decentralized actors,
   centralized critic, standard on-policy policy-gradient training.
2. **World-model-based methods (Dreamer-style)** — learn a recurrent latent dynamics
   model per agent and train actor/critic via imagined rollouts in latent space.
3. **Mamba-JEPA (candidate best solution)** — a CTDE, purely latent (non-generative)
   world model built on a Mamba structured state-space model (SSM):
   - A JEPA-style (Joint Embedding Predictive Architecture) encoder compresses LiDAR
     and local sub-grid observations into a latent state — no pixel/observation-space
     reconstruction.
   - The world model is trained only to predict the **future latent state**, minimizing
     the gap (e.g. MSE or contrastive loss) between the predicted and the actually
     encoded next latent state.
   - The recurrent core (RSSM-style recurrence) is replaced with **Mamba**: as a
     modern SSM, it maintains long historical context without vanishing gradients,
     integrating each robot's full sequence of past local observations and actions into
     a single latent "mental map."
   - During training, the world model has access to the local observations and actions
     of **all** robots, learning a **shared latent map** of how the environment is
     being explored jointly.
   - The actor and critic are trained by **imagining trajectories** inside this shared
     latent world model (as in Dreamer/TD-MPC2), rather than only on real environment
     rollouts.

## Implementation Constraints

- All modeling and training code (encoder, Mamba SSM world model, actor, critic,
  imagination rollouts) must be implemented in **JAX**, targeting **CUDA** execution, if the current machine has an NVIDIA GPU on, **JAX-METAL** if the code is running on a macbook with apple silicon M1/M2/M3/M4, **CPU** in all other cases. All code implementations are intended to run as fast as possible on the machine used.
- Code must be lean: no unnecessary comments, no dead code, no speculative
  abstractions ahead of what a given experiment needs.

## Test scripts
