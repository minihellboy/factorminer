#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-/tmp/factorminer-quickstart-real}"
LIBRARY_PATH="${2:-${OUTPUT_DIR}/factor_library.json}"

LIBRARY_STATUS="$(
  uv run python - "$LIBRARY_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"MISSING:{path}")
    raise SystemExit(0)

data = json.loads(path.read_text())
if len(data.get("factors", [])) == 0:
    print("EMPTY")
    raise SystemExit(0)

print("HAS_FACTORS")
PY
)"

case "$LIBRARY_STATUS" in
  MISSING:*)
    printf '%s\n' "$LIBRARY_STATUS" >&2
    exit 1
    ;;
  EMPTY)
    printf '%s\n' "No admitted factors in ${LIBRARY_PATH}; evaluate after running a larger mining pass."
    exit 0
    ;;
esac

uv run factorminer --cpu -c "${ROOT_DIR}/quickstart.yaml" evaluate "$LIBRARY_PATH" \
  --data "${ROOT_DIR}/sample_market_data.csv" \
  --period both \
  --top-k 5
