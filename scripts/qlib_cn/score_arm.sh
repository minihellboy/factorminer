#!/usr/bin/env bash
# Score mined campaign arms on the adopted model: export each window's library, append it to
# Alpha158 under the 5-day label (7 seeds), then judge the books on CSI300.
# usage: [YEARS="2018 2020 2022 2024"] [SEEDS=0-6] score_arm.sh <tag> <arm> [<arm> ...]
# Per arm and window: campaigns/wf<y>_<arm> -> csi300/wf<y>/wf<y>_<arm>/factor_values.csv
# (evaluate_library --marginal) -> csi300/wf<y>/lib_<arm>_h5_lgb_s<seed> (fast_fit).
# Books: h5 baseline, old-flash marginal (lib_marginal), and each arm -> family/<tag>_csi300.json.
set -eu
cd "$(dirname "$0")/../.."
QPY=${QPY:-$HOME/fm-data/qlib-venv/bin/python} PROVIDER=${PROVIDER:-$HOME/fm-data/qlib_cn}
FPY=${FPY:-.venv/bin/python} PANEL=${PANEL:-$HOME/fm-data/panels/csi300_2007_2026.csv}
OUT=${OUT:-output/qlib_cn} C=$OUT/campaigns YEARS=${YEARS:-2018 2020 2022 2024}
tag=$1; shift
for arm in "$@"; do
  protocols=()
  for y in $YEARS; do
    w=wf$y dst=$OUT/csi300/$w/${w}_$arm
    protocols+=("$w")
    [ -s "$C/${w}_$arm/factor_library.json" ] || { echo "missing library $C/${w}_$arm" >&2; exit 1; }
    [ -s "$dst/factor_values.csv" ] || "$FPY" scripts/qlib_cn/evaluate_library.py -c "$C/${w}_$arm.yaml" \
      --data "$PANEL" --library "$C/${w}_$arm/factor_library.json" --out "$dst" --backend gpu --marginal \
      > "$dst.log" 2>&1
  done
  ( IFS=,; "$QPY" -I scripts/qlib_cn/fast_fit.py fit --provider "$PROVIDER" --universe csi300 \
      --protocols "${protocols[*]}" --horizons 5 --seeds "${SEEDS:-0-6}" --name "lib_${arm}_h{h}_lgb_s{s}" \
      --factors "$OUT/csi300/{p}/{p}_$arm/factor_values.csv" --out "$OUT" ) > "$OUT/family/${tag}_fit_$arm.log" 2>&1
done
books=(h5='alpha158_h5_lgb_s[0-6]' marginal='lib_marginal_h5_lgb_s[0-6]')
for arm in "$@"; do books+=("$arm=lib_${arm}_h5_lgb_s[0-6]"); done
YEARS="$YEARS" UNIVERSES=csi300 EXTRA="--tradable" scripts/qlib_cn/family.sh "$tag" h5 "${books[@]}"
