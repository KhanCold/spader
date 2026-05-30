#!/usr/bin/env bash
set -euo pipefail

# Script to launch retrieval server using the QAMPARI BM25 index with BGE Reranker

# Fix GLIBCXX version mismatch: use conda's newer libstdc++ instead of system's
if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local dir="$1"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/pyproject.toml" || -d "$dir/.git" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  echo "$1"
}

REPO_ROOT="$(find_repo_root "$SCRIPT_DIR")"
SERVER_PY="$SCRIPT_DIR/retrieval_server.py"

RETRIEVER_NAME="bm25"
INDEX_PATH="$REPO_ROOT/data/QAMPARI_wikipedia_2021/qampari_bm25_index"
RERANKER_MODEL="$REPO_ROOT/models/bge-reranker-v2-m3"

if [[ ! -d "$INDEX_PATH" ]]; then
  echo "Index not found at: $INDEX_PATH" >&2
  echo "Please run examples/sglang_multiturn/search_r1_like/local_dense_retriever/build_qampari_index.sh first." >&2
  exit 2
fi

if [[ ! -d "$RERANKER_MODEL" ]]; then
    echo "Warning: Reranker model path not found at $RERANKER_MODEL. It might be downloaded automatically from Hugging Face if you have internet access."
fi

echo "Running Retrieval Server using QAMPARI index + BM25 + Reranker..."
echo "  GPUs will be auto-detected by the server (control via CUDA_VISIBLE_DEVICES)."
nohup python -u "$SERVER_PY" \
  --retriever_name "$RETRIEVER_NAME" \
  --index_path "$INDEX_PATH" \
  --use_reranker \
  --reranker_model "$RERANKER_MODEL" \
  "$@" > retrieval_server.log 2>&1 &

echo "Server started in background (PID: $!). Logs are being written to retrieval_server.log"

