#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-/tmp/factorminer-quickstart-mock}"

mkdir -p "$OUTPUT_DIR"

uv run factorminer --cpu -o "$OUTPUT_DIR" mine \
  --mock \
  -n 2 \
  -b 6 \
  -t 2
