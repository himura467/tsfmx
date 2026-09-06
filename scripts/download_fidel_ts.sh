#!/usr/bin/env bash

# Download one Fidel-TS sub-dataset from the Hugging Face Hub.
#
# The raw_data/ archives hold the unprocessed source dumps that the dataset's own cleaning
# scripts already consumed, and are several times larger than everything else. Pre-computed
# report embeddings are skipped too: this repository encodes the text itself, with the encoder
# named by the model config. Both appear at the top level in some sub-datasets and nested under
# a location in others, so the patterns match at any depth.

set -euo pipefail

DATASET="${1:-Bear_room}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
TARGET_DIR="$REPO_DIR/data/Fidel-TS/$DATASET"

# A download that fails partway leaves the directory behind, so completion is judged by a series
# having arrived rather than by the directory existing. Sub-datasets disagree on where those live
# — Bear_room uses time_series, Germany_Renewable_Energy_Grid impute_data — so look for the files.
if [[ -d "$TARGET_DIR" ]] && [[ -n "$(find "$TARGET_DIR" -name '*.parquet' -print -quit)" ]]; then
  echo "Fidel-TS $DATASET already exists at $TARGET_DIR, skipping download."
  exit 0
fi
if [[ -d "$TARGET_DIR" ]]; then
  echo "Removing incomplete download at $TARGET_DIR."
  rm -rf "$TARGET_DIR"
fi

mkdir -p "$REPO_DIR/data/Fidel-TS"
uv run hf download "fidel-ts/$DATASET" \
  --repo-type dataset \
  --local-dir "$TARGET_DIR" \
  --exclude "*raw_data/*" \
  --exclude "*scripts/*" \
  --exclude "*report_embedding/*" \
  --exclude "*.pkl"

echo "Fidel-TS $DATASET downloaded to $TARGET_DIR."
