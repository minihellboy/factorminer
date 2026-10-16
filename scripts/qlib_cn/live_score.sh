#!/usr/bin/env bash
# Score a registered procedure on the prospective holdout (docs/qlib-cn-protocol.md, "Prospective holdout").
# usage: [TAG=2026-11-02] live_score.sh docs/registrations/<name>.json
#
# 1. Fetches the investment_data provider release TAG (default: latest) into ~/fm-data/live/<tag>/qlib_bin,
#    never touching the research provider ~/fm-data/qlib_cn (provider rebuilds can revise history).
# 2. register.py verify: the boosters are the frozen ones.
# 3. Predicts with every registered booster (run_qlib_baselines.py --protocol live --load-model)
#    into output/qlib_cn/<universe>/live_pred/<tag>/<run>.
# 4. book_eval.py --window live=... over the registration's books and baseline (registered runs only)
#    -> output/qlib_cn/live/<name>_<tag>.json, with the provider's last date and calendar hash.
set -eu
cd "$(dirname "$0")/../.."
QPY=${QPY:-$HOME/fm-data/qlib-venv/bin/python} LIVE=${LIVE:-$HOME/fm-data/live}
reg=$1
tag=${TAG:-$(gh release list -R chenditc/investment_data -L 1 --json tagName -q '.[0].tagName')}
provider="$LIVE/$tag/qlib_bin"
if [ ! -d "$provider" ]; then
  mkdir -p "$LIVE/$tag"
  gh release download "$tag" -R chenditc/investment_data -p 'qlib_bin*' -D "$LIVE/$tag" --clobber
  tar xzf "$LIVE/$tag/qlib_bin.tar.gz" -C "$LIVE/$tag" && rm "$LIVE/$tag/qlib_bin.tar.gz"
fi
last=$(tail -1 "$provider/calendars/day.txt")
start=$("$QPY" -I -c "import sys; sys.path.insert(0, 'scripts/qlib_cn'); from run_qlib_baselines import PROTOCOLS; print(PROTOCOLS['live']['test'][0])")
if [[ "$last" < "$start" ]]; then
  echo "provider $tag ends $last, before the holdout starts ($start): nothing to score"; exit 0
fi

"$QPY" -I scripts/qlib_cn/register.py verify "$reg"
read -r name universe root baseline < <("$QPY" -I -c "
import json, sys; r = json.load(open(sys.argv[1])); s = r['spec']
print(s['name'], s.get('universe', 'csi300'), r['runs_root'], s['baseline'])" "$reg")
pred="output/qlib_cn/$universe/live_pred/$tag"
mkdir -p "$pred" output/qlib_cn/live
for run in $("$QPY" -I -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['runs']))" "$reg"); do
  [ -f "$pred/$run/predictions.parquet" ] && continue
  read -r horizon seed < <("$QPY" -I -c "
import json, sys; m = json.load(open(sys.argv[1])); print(m.get('horizon', 1), m.get('seed', 0))" "$root/$run/metrics.json")
  "$QPY" -I scripts/qlib_cn/run_qlib_baselines.py --provider "$provider" --universe "$universe" --protocol live \
    --handler alpha158 --load-model "$root/$run/model.txt" --horizon "$horizon" --seed "$seed" \
    --out "$pred/$run" > "$pred/$run.log" 2>&1
done

label="$LIVE/$tag/labels_$universe.parquet"
[ -f "$label" ] || "$QPY" -I -c "
import sys, qlib; from qlib.data import D
qlib.init(provider_uri=sys.argv[1], region='cn')
df = D.features(D.instruments(sys.argv[2]), ['Ref(\$close, -2)/Ref(\$close, -1) - 1'], start_time=sys.argv[3])
df.columns = ['label']; df.reset_index().to_parquet(sys.argv[4])" "$provider" "$universe" "$start" "$label"

books=()
while read -r book; do books+=(--book "$book"); done < <("$QPY" -I -c "
import json, sys; [print(f'{k}={v}') for k, v in json.load(open(sys.argv[1]))['spec']['books'].items()]" "$reg")
out="output/qlib_cn/live/${name}_$tag.json"
"$QPY" -I scripts/qlib_cn/book_eval.py --provider "$provider" --universe "$universe" --label "$label" \
  --window "live=$pred" "${books[@]}" --baseline "$baseline" --tie-draws 4 --out "$out"
"$QPY" -I -c "
import hashlib, json, sys
out, provider, tag = sys.argv[1:4]
record = json.load(open(out))
calendar = open(f'{provider}/calendars/day.txt', 'rb').read()
record['provider'] = {'release_tag': tag, 'last_date': calendar.split()[-1].decode(),
                      'calendar_sha256': hashlib.sha256(calendar).hexdigest()}
json.dump(record, open(out, 'w'), indent=2)
print(f'{out}: provider {tag}, last date', record['provider']['last_date'])" "$out" "$provider" "$tag"
