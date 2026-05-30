#!/usr/bin/env bash
set -euo pipefail

# Minimal nohup-friendly launcher for retrieval_server.py
# Usage:
#   INDEX_PATH=/path/to/bm25_index CORPUS_PATH=/path/to/wiki-18.jsonl \
#   RETRIEVER_NAME=bm25 TOPK=3 \
#   bash start_retrieval_server.sh

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

RETRIEVER_NAME="${RETRIEVER_NAME:-bm25}"
DEFAULT_LUCENE_INDEX="$REPO_ROOT/data/pyserini_indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6"
INDEX_PATH="${INDEX_PATH:-$DEFAULT_LUCENE_INDEX}"
if [[ ! -e "$INDEX_PATH" ]]; then
  echo "INDEX_PATH not found: $INDEX_PATH" >&2
  echo "Set INDEX_PATH to your Lucene/FAISS index path." >&2
  exit 2
fi

CORPUS_PATH="${CORPUS_PATH:-}"
TOPK="${TOPK:-3}"
USE_FAISS_GPU="${USE_FAISS_GPU:-0}"

cmd=(python -u "$SERVER_PY" \
  --retriever_name "$RETRIEVER_NAME" \
  --index_path "$INDEX_PATH" \
  --topk "$TOPK")

if [[ -n "$CORPUS_PATH" ]]; then
  cmd+=(--corpus_path "$CORPUS_PATH")
fi

if [[ "$USE_FAISS_GPU" == "1" ]]; then
  cmd+=(--faiss_gpu)
fi

echo "Running: ${cmd[*]}" >&2
exec "${cmd[@]}"
