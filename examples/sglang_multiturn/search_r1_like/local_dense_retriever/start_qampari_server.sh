#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
# Script to launch retrieval server using the QAMPARI BM25 index

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

if [[ ! -d "$INDEX_PATH" ]]; then
  echo "Index not found at: $INDEX_PATH" >&2
  echo "Please run examples/sglang_multiturn/search_r1_like/local_dense_retriever/build_qampari_index.sh first." >&2
  exit 2
fi

echo "Running Retrieval Server using QAMPARI index..."
# We rely on the index containing raw documents (--storeRaw), so no corpus path needed.
nohup python -u "$SERVER_PY" \
  --retriever_name "$RETRIEVER_NAME" \
  --index_path "$INDEX_PATH" \
  "$@" > retrieval_server.log 2>&1 &

echo "Server started in background (PID: $!). Logs are being written to retrieval_server.log"
