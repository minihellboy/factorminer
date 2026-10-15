#!/usr/bin/env bash
# Run a queue of run_qlib_baselines.py fits in parallel; finished runs (metrics.json) are skipped.
# usage: [PAR=2] [LGB_THREADS=12] [UNIVERSE=csi300] [OUT=output/qlib_cn] fit_queue.sh <jobs file>
# Each non-comment line: <protocol> <run name> <run_qlib_baselines.py args...>;
# the run goes to $OUT/$UNIVERSE/<protocol>/<run name>, its log next to it.
set -u
cd "$(dirname "$0")/../.."
export QPY=${QPY:-$HOME/fm-data/qlib-venv/bin/python} PROVIDER=${PROVIDER:-$HOME/fm-data/qlib_cn}
export UNIVERSE=${UNIVERSE:-csi300} OUT=${OUT:-output/qlib_cn} LGB_THREADS=${LGB_THREADS:-12}
export MLFLOW_DISABLE_AGENT_HINT=1
fit() {
  set -- $1; local p=$1 name=$2; shift 2
  local d=$OUT/$UNIVERSE/$p/$name
  [ -f "$d/metrics.json" ] && return 0
  mkdir -p "$d"
  "$QPY" -I scripts/qlib_cn/run_qlib_baselines.py --provider "$PROVIDER" --universe "$UNIVERSE" \
    --protocol "$p" "$@" --out "$d" > "$d.log" 2>&1 && echo "done $p $name" || echo "FAIL $p $name"
}
export -f fit
grep -vE '^\s*(#|$)' "$1" | xargs -P "${PAR:-2}" -I{} bash -c 'fit "{}"'
echo "queue $1 finished"
