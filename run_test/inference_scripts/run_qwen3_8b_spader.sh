#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"models/qwen3-8b-spader"}
SERVED_NAME=${SERVED_NAME:-"qwen3-8b-spader"}

PORT=${PORT:-8001}
TensorParallel=${TensorParallel:-4}
GPU_IDS=${GPU_IDS:-"0,1,2,3"}

LOG_FILE="vllm_server_${SERVED_NAME}_${PORT}_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES=$GPU_IDS nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $SERVED_NAME \
    --host 127.0.0.1 \
    --port $PORT \
    --dtype bfloat16 \
    --tensor-parallel-size $TensorParallel \
    --gpu-memory-utilization 0.5 \
    --trust-remote-code > "$LOG_FILE" 2>&1 &
