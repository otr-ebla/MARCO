#!/usr/bin/env bash
# Single-robot coverage: the optimal method, IL, RL and IL+RL, compared on the
# same seeds and on layouts unseen during training. No recovery fallback.
#
#   bash scripts/single_robot_experiments.sh [extra train/pretrain args, e.g. --backend cuda]
#
# Environment overrides: CONFIG, OUT, MODE, BC_COEF, EPISODES, EVAL_MAP_SEED.
set -euo pipefail

CONFIG=${CONFIG:-config/single_robot.yaml}
OUT=${OUT:-checkpoints/single}
MODE=${MODE:-end-to-end-memory}        # or end-to-end (feed-forward actor)
BC_COEF=${BC_COEF:-0.5}                # decaying BC weight during IL+RL
EPISODES=${EPISODES:-1000}
EVAL_MAP_SEED=${EVAL_MAP_SEED:-1000}   # training uses env.map_seed (0)

ckpt=checkpoint_e2e.pkl
[ "$MODE" = end-to-end-memory ] && ckpt=checkpoint_e2e_memory.pkl

# 0. Optimal plans for every (map, spawn cell), training and evaluation banks
#    (cached in pretrain.optimal.cache_dir; later steps load them).
python -m src.envs.optimal_expert --config "$CONFIG"
python -m src.envs.optimal_expert --config "$CONFIG" --map-seed "$EVAL_MAP_SEED"

# 1. RL from scratch.
python -m src.train_marl --config "$CONFIG" --policy-mode "$MODE" \
    --save-dir "$OUT/rl" --wandb-name single-rl --no-live-plot "$@"

# 2. IL: DAgger behaviour cloning of the optimal expert (+ critic warm-up for step 3).
python -m src.pretrain_bc --config "$CONFIG" --policy-mode "$MODE" \
    --save-dir "$OUT/il" --wandb-name single-il "$@"

# 3. IL+RL: PPO from the cloned actor, BC term towards the optimal expert decaying to zero.
python -m src.train_marl --config "$CONFIG" --policy-mode "$MODE" \
    --resume "$OUT/il/checkpoint_bc.pkl" --bc-coef "$BC_COEF" \
    --save-dir "$OUT/il_rl" --wandb-name single-il-rl --no-live-plot "$@"

# 4. Comparison of the four policies (humans off: the static problem).
python -m src.evaluate_policies --config "$CONFIG" --humans 0 \
    --episodes "$EPISODES" --map-seed "$EVAL_MAP_SEED" \
    --output-dir evaluation_results/single_robot \
    --optimal \
    --checkpoint "$OUT/il/checkpoint_bc_actor.pkl" --label IL \
    --checkpoint "$OUT/rl/$ckpt"                   --label RL \
    --checkpoint "$OUT/il_rl/$ckpt"                --label IL+RL
