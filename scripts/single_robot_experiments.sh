#!/usr/bin/env bash
# Single-robot coverage: train RL, IL and IL+RL, then compare them with
# centralised BCD on the same seeds and on layouts unseen during training.
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

# 1. RL from scratch.
python -m src.train_marl --config "$CONFIG" --policy-mode "$MODE" \
    --save-dir "$OUT/rl" --wandb-name single-rl --no-live-plot "$@"

# 2. IL: DAgger behaviour cloning of the BCD expert (+ critic warm-up for step 3).
python -m src.pretrain_bc --config "$CONFIG" --policy-mode "$MODE" \
    --save-dir "$OUT/il" --wandb-name single-il "$@"

# 3. IL+RL: PPO from the cloned actor, BC term decaying to zero.
python -m src.train_marl --config "$CONFIG" --policy-mode "$MODE" \
    --resume "$OUT/il/checkpoint_bc.pkl" --bc-coef "$BC_COEF" \
    --save-dir "$OUT/il_rl" --wandb-name single-il-rl --no-live-plot "$@"

# 4. Comparison. --policy-only: no recovery fallback, the policy acts alone
#    (BCD never uses it). Humans off: the static problem.
python -m src.evaluate_policies --config "$CONFIG" --humans 0 --policy-only \
    --episodes "$EPISODES" --map-seed "$EVAL_MAP_SEED" \
    --output-dir evaluation_results/single_robot \
    --bcd tour \
    --checkpoint "$OUT/il/checkpoint_bc_actor.pkl" --label IL \
    --checkpoint "$OUT/rl/$ckpt"                   --label RL \
    --checkpoint "$OUT/il_rl/$ckpt"                --label IL+RL
