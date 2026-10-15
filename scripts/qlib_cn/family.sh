#!/usr/bin/env bash
# Judge books in every universe and pooled across universes (the standard comparison).
# usage: [YEARS="2018 2020 2022 2024"] [UNIVERSES="csi300 csi500 csi1000"] [DRAWS=4] [JOBS=12] [EXTRA="book_eval args"] family.sh <tag> <baseline> name=glob [name=glob ...]
# Each universe: book_eval.py over the walk-forward windows -> output/qlib_cn/family/<tag>_<universe>.json
# (+ .daily.parquet); then cross_universe.py pools the day-averaged paired differences
# -> output/qlib_cn/family/<tag>_pooled.json.
set -eu
cd "$(dirname "$0")/../.."
QPY=${QPY:-$HOME/fm-data/qlib-venv/bin/python} PROVIDER=${PROVIDER:-$HOME/fm-data/qlib_cn}
OUT=${OUT:-output/qlib_cn} UNIVERSES=${UNIVERSES:-csi300 csi500 csi1000}
tag=$1 baseline=$2; shift 2
books=(); for b in "$@"; do books+=(--book "$b"); done
mkdir -p "$OUT/family"
daily=()
for u in $UNIVERSES; do
  windows=(); for y in ${YEARS:-2018 2020 2022 2024}; do windows+=(--window "wf$y=$OUT/$u/wf$y"); done
  "$QPY" -I scripts/qlib_cn/book_eval.py --provider "$PROVIDER" --universe "$u" \
    --label "$HOME/fm-data/labels/$u.parquet" "${windows[@]}" "${books[@]}" --baseline "$baseline" \
    --tie-draws "${DRAWS:-4}" --jobs "${JOBS:-12}" ${EXTRA:-} --out "$OUT/family/${tag}_$u.json" 2>&1 \
    | grep -E '^[^ ]+ \{' | sed "s/^/$u /" | cut -c1-160
  daily+=("$OUT/family/${tag}_$u.daily.parquet")
done
"$QPY" -I scripts/qlib_cn/cross_universe.py --baseline "$baseline" --out "$OUT/family/${tag}_pooled.json" "${daily[@]}"
