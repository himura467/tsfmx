#!/usr/bin/env bash

# Download one Fidel-TS sub-dataset from the Hugging Face Hub.
#
# The raw_data/ archives hold the unprocessed source dumps that the dataset's own cleaning
# scripts already consumed, and are several times larger than everything else. They are
# excluded, so only the processed time series, the heterogeneous context, and the metadata
# are fetched.

set -euo pipefail

DATASET="${1:-Bear_room}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
TARGET_DIR="$REPO_DIR/data/Fidel-TS/$DATASET"

# A download that fails partway leaves the directory behind, so completion is judged by the
# time series having arrived rather than by the directory existing.
if [[ -d "$TARGET_DIR/time_series" ]]; then
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
  --exclude "raw_data/*" \
  --exclude "scripts/*"

echo "Fidel-TS $DATASET downloaded to $TARGET_DIR."
