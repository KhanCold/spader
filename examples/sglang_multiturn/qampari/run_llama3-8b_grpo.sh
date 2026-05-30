#!/bin/bash

CONDA_PATH=$(which conda)
if [ -n "$CONDA_PATH" ]; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda activate verl-sglang

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

WAND_PROJECT=${WAND_PROJECT:-"SPADER_QAMPARI"}
BASE_MODEL=${BASE_MODEL:-"$PROJECT_DIR/models/Llama-3.1-8B-Instruct"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qampari-llama3-8b-grpo-$(date +%m%d_%H%M%S)"}

export PROJECT_DIR

if [ -z "${TRAIN_DATA:-}" ] || [ -z "${VAL_DATA:-}" ]; then
    echo "Please set TRAIN_DATA and VAL_DATA to parquet files." >&2
    echo "Example: TRAIN_DATA=/path/to/train.parquet VAL_DATA=/path/to/val.parquet $0" >&2
    exit 1
fi
export TRAIN_DATA
export VAL_DATA

OUTPUT_ROOT=${OUTPUT_ROOT:-"$PROJECT_DIR/output"}
RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints" "${RUN_DIR}/eval" "${RUN_DIR}/wandb" "${RUN_DIR}/rollouts"
export WANDB_DIR="${RUN_DIR}/wandb"
export RUN_DIR

export PYTHONUNBUFFERED=1

export BASE_MODEL
export EXPERIMENT_NAME
export WAND_PROJECT

echo "BASE_MODEL: ${BASE_MODEL}"
echo "TRAIN_DATA: ${TRAIN_DATA}"
echo "VAL_DATA: ${VAL_DATA}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"

LOG_FILE="${RUN_DIR}/logs/train.log"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='qampari_llama3_8b_grpo' \
    hydra.run.dir="$RUN_DIR" \
    hydra.job.chdir=true \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    reward_model.reward_kwargs.cost_free_steps=200 \
    reward_model.reward_kwargs.tool_round_cost=0.01 \
    "$@" \
    2>&1 | tee "$LOG_FILE"
