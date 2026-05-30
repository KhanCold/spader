#!/bin/bash

python run_test/run_eval.py \
    --dataset "${DATASET:-all}" \
    --model llama3-8b-grpo \
    --api_base "${API_BASE:-http://localhost:8001/v1}" \
    --tool_url "${TOOL_URL:-http://127.0.0.1:8000/retrieve}" \
    --concurrency "${CONCURRENCY:-20}" \
    --analyze \
    --exp_name "${EXP_NAME:-llama3-8b-grpo}"
