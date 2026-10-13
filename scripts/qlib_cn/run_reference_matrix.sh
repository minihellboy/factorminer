#!/usr/bin/env bash
# Run the Qlib reference workflows for every universe/protocol pair.
# usage: run_reference_matrix.sh <qlib python> <provider> <output root>
set -u
PY=$1; PROVIDER=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
export MLFLOW_DISABLE_AGENT_HINT=1
run() {  # universe protocol handler model seed
  local dir="$OUT/$1/$2/$3_$4_s$5"
  [ -f "$dir/metrics.json" ] && return 0
  mkdir -p "$dir"
  "$PY" -I "$HERE/run_qlib_baselines.py" --provider "$PROVIDER" --universe "$1" --protocol "$2" \
    --handler "$3" --model "$4" --seed "$5" --out "$dir" > "$dir/run.log" 2>&1 \
    && echo "ok   $dir" || echo "FAIL $dir"
}
for universe in csi300 csi500; do
  for protocol in classic recent; do
    for seed in 0 1 2; do
      run "$universe" "$protocol" alpha158 lgb "$seed"
      run "$universe" "$protocol" alpha360 lgb "$seed"
    done
    run "$universe" "$protocol" alpha158 linear 0
  done
done
