#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-/tmp/factorminer-quickstart-real}"

mkdir -p "$OUTPUT_DIR"
bash "${ROOT_DIR}/validate-data.sh" "${ROOT_DIR}/sample_market_data.csv" >/dev/null

uv run factorminer --cpu -c "${ROOT_DIR}/quickstart.yaml" -o "$OUTPUT_DIR" mine \
  --data "${ROOT_DIR}/sample_market_data.csv" \
  -n 3 \
  -b 8 \
  -t 3
