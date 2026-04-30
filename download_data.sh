#!/usr/bin/env bash
set -e

# This script is intentionally lightweight.
# Its purpose is to download SMALL public artefacts that are too large or inconvenient to store in Git:
#   1) a tiny demo bundle (recommended)
#   2) an optional pretrained checkpoint
#   3) an optional processed subset for evaluation
# It should NOT download private robot XML files.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT_DIR/data/demo" "$ROOT_DIR/checkpoints"

DEMO_URL="${DEMO_URL:-}"
CHECKPOINT_URL="${CHECKPOINT_URL:-}"
PROCESSED_URL="${PROCESSED_URL:-}"

fetch_file () {
  url="$1"
  out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Need curl or wget to download files."
    exit 1
  fi
}

if [ -n "$DEMO_URL" ]; then
  echo "Downloading demo bundle..."
  fetch_file "$DEMO_URL" "$ROOT_DIR/data/demo/demo_bundle.zip"
  unzip -o "$ROOT_DIR/data/demo/demo_bundle.zip" -d "$ROOT_DIR/data/demo"
else
  echo "DEMO_URL is not set. Skipping demo download."
fi

if [ -n "$CHECKPOINT_URL" ]; then
  echo "Downloading checkpoint..."
  fetch_file "$CHECKPOINT_URL" "$ROOT_DIR/checkpoints/best.pt"
else
  echo "CHECKPOINT_URL is not set. Skipping checkpoint download."
fi

if [ -n "$PROCESSED_URL" ]; then
  echo "Downloading optional processed subset..."
  fetch_file "$PROCESSED_URL" "$ROOT_DIR/data/processed_subset.zip"
else
  echo "PROCESSED_URL is not set. Skipping processed subset download."
fi

echo "Done. Read data/README.md for the expected folder structure."
