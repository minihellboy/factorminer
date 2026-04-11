#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${1:-${ROOT_DIR}/sample_market_data.csv}"

uv run factorminer --cpu validate-data "$DATA_PATH"
