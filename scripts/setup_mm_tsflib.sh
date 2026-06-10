#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
MM_DIR="$REPO_DIR/third_party/MM-TSFlib"

if [[ -d "$MM_DIR" ]]; then
  echo "MM-TSFlib already exists at $MM_DIR, skipping clone."
else
  mkdir -p "$REPO_DIR/third_party"
  git clone --depth 1 https://github.com/AdityaLab/MM-TSFlib.git "$MM_DIR"
  echo "MM-TSFlib cloned to $MM_DIR."
fi

RAR="$MM_DIR/data/Environment/NewYork_AQI_Day.rar"
CSV="$MM_DIR/data/Environment/NewYork_AQI_Day.csv"
if [[ -f "$RAR" && ! -f "$CSV" ]]; then
  bsdtar -xf "$RAR" -C "$MM_DIR/data/Environment/"
  echo "Extracted Environment data."
fi
