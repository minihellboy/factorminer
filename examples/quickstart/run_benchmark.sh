#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_LIVE="false"
if [[ "${1:-}" == "--run" ]]; then
  RUN_LIVE="true"
  shift
fi
OUTPUT_DIR="${1:-/tmp/factorminer-quickstart-benchmark}"

mkdir -p "$OUTPUT_DIR"

if [[ "$RUN_LIVE" == "true" ]]; then
  uv run factorminer --cpu -o "$OUTPUT_DIR" benchmark table1 --mock --baseline factor_miner
else
  cat <<EOF
Benchmark quickstart dry run
============================

Intended command:
  bash ${ROOT_DIR}/run_benchmark.sh --run "$OUTPUT_DIR"

Equivalent CLI:
  uv run factorminer --cpu -o "$OUTPUT_DIR" benchmark table1 --mock --baseline factor_miner

Why dry-run by default:
  The canonical table1 benchmark intentionally builds a large mock panel and can take
  materially longer than the other quickstart scripts. Run it explicitly with --run
  when you want to exercise the full benchmark path.
EOF
fi

cat <<EOF

Expected artifacts:
  - ${OUTPUT_DIR}/benchmark/table1/factor_miner.json
  - ${OUTPUT_DIR}/benchmark/table1/factor_miner_manifest.json
  - ${OUTPUT_DIR}/benchmark/table1/factor_miner/runtime/

See:
  - ${ROOT_DIR}/README.md
  - ${ROOT_DIR}/expected/artifacts.md
EOF
