#!/bin/bash
set -euo pipefail

# This script builds the BM25 index for the QAMPARI dataset.
# It handles extraction, indexing, and cleanup of raw files.

# 1. Determine paths
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
DATA_ROOT="$REPO_ROOT/data/QAMPARI_wikipedia_2021"
TAR_FILE="$DATA_ROOT/chunked_wikipedia.tar.gz"
EXTRACT_DIR="$DATA_ROOT"
# The tarball contains: wikipedia_chunks/chunks_v5/wikipedia_chunks_*.jsonl
# So extracting to EXTRACT_DIR results in: $DATA_ROOT/wikipedia_chunks/chunks_v5
CORPUS_DIR="$DATA_ROOT/wikipedia_chunks/chunks_v5"
INDEX_DIR="$DATA_ROOT/qampari_bm25_index"

# 2. Check prerequisites
if [[ ! -f "$TAR_FILE" ]]; then
    echo "Error: Tarball not found at $TAR_FILE"
    exit 1
fi

if ! python -c "import pyserini" &> /dev/null; then
    echo "Pyserini not found. Installing..."
    pip install pyserini
fi

# 3. Extract Corpus
echo "[1/3] Extracting corpus..."
# -C changes directory before extracting
tar -zxf "$TAR_FILE" -C "$EXTRACT_DIR"

if [[ ! -d "$CORPUS_DIR" ]]; then
    echo "Error: Expected extracted directory $CORPUS_DIR not found!"
    exit 1
fi

# 4. Build Index
echo "[2/3] Building Pyserini BM25 Index..."
# Note: --storeRaw stores the JSON content in the index, allowing us to delete the source files later.
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$CORPUS_DIR" \
  --index "$INDEX_DIR" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw

# 5. Cleanup
echo "[3/3] Cleaning up extracted JSONL files..."
rm -rf "$DATA_ROOT/wikipedia_chunks"

echo "Success! Index created at: $INDEX_DIR"
echo "Raw JSONL files have been removed to save space."
