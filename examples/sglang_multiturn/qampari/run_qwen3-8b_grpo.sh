#!/bin/bash

set -x

__conda_setup="$('/opt/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/miniconda/etc/profile.d/conda.sh" ]; then
        . "/opt/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate verl-sglang

ulimit -n 65535

for lib_dir in "$HOME"/.local/lib/python3.12/site-packages/nvidia/*/lib; do
    if [ -d "$lib_dir" ]; then
        export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
    fi
done

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

WAND_PROJECT=${WAND_PROJECT:-"SPADER_QAMPARI"}
BASE_MODEL=${BASE_MODEL:-"$PROJECT_DIR/models/Qwen3-8B"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qampari-qwen3-8b-grpo-$(date +%m%d_%H%M%S)"}

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
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints" "${RUN_DIR}/eval" "${RUN_DIR}/wandb"
export WANDB_DIR="${RUN_DIR}/wandb"
export RUN_DIR

# Disable python output buffering to see logs immediately in train.log
export PYTHONUNBUFFERED=1

export BASE_MODEL
export EXPERIMENT_NAME
export WAND_PROJECT

echo "BASE_MODEL: ${BASE_MODEL}"
echo "TRAIN_DATA: ${TRAIN_DATA}"
echo "VAL_DATA: ${VAL_DATA}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "RUN_DIR: ${RUN_DIR}"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='qampari_qwen3_8b_grpo' \
    hydra.run.dir="$RUN_DIR" \
    hydra.job.chdir=true \
    "$@" \
    2>&1 | tee "${RUN_DIR}/logs/train.log"
