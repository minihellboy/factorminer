"""Write campaign configs for new Qlib CN arms from an existing campaign's config.

Run in the FactorMiner environment::

    uv run python scripts/qlib_cn/make_campaigns.py --campaigns output/qlib_cn/campaigns \\
        --arm top='{"admission": {"min_top_t": 2.0, "min_top_lift_t": 0.0}}' \\
        --arm fund='{"data": {"features": "+fund"}}' \\
        --protocol classic:3 --protocol wf2018 --protocol wf2020 --protocol wf2022 --protocol wf2024

Each arm is a JSON override merged into the ``<protocol>_marginal[_c1]`` config
of that protocol (``classic:3`` writes three replicates ``_c1.._c3``). The
``"+fund"`` feature shorthand appends the BaoStock valuation/size/liquidity
leaves (``merge_panel_features.py``) with their prompt descriptions; ``"+fq"``
adds, on top of those, the point-in-time quarterly report and industry leaves
of ``data_factors.py --panel-out``. Output
files are ``<campaigns>/<protocol>_<arm>[_cN].yaml``; the campaign's
``output_dir`` and ``benchmark.seed`` follow the template's convention.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

FUND_FEATURES = {
    "$turnover": "daily turnover rate, % of float shares traded (liquidity / attention; "
                 "compare to its own history or rank across stocks)",
    "$pe_ttm": "trailing twelve-month price/earnings, point in time from the latest published "
               "report; negative when earnings are negative, so prefer 1/$pe_ttm (earnings yield)",
    "$pb": "price/book from the most recent quarterly report (value: 1/$pb is book yield)",
    "$ps_ttm": "trailing twelve-month price/sales (1/$ps_ttm is sales yield)",
    "$pcf_ttm": "trailing twelve-month price/net cash flow (1/$pcf_ttm is cash-flow yield)",
    "$mcap": "float market capitalisation in CNY (size; use Log or CsRank)",
    "$st": "1 when the stock is under special treatment (ST) risk warning, else 0",
    "$index_close": "benchmark index close (CSI300/CSI500), the same for every stock on a date; "
                    "use it to strip market moves (relative strength, beta, residual momentum)",
}

FQ_FEATURES = {
    "$roe": "return on equity of the latest published report, %, year to date as reported "
            "(Q1 ~ a quarter's ROE, annual ~ a year's; compare across stocks or vs the same quarter a year ago)",
    "$gp_margin": "gross margin of the latest published report, % (pricing power / profitability)",
    "$np_margin": "net income / revenue of the latest published report (net profitability)",
    "$yoy_ni": "year-on-year net income growth of the latest published report, % (earnings growth; "
               "heavy-tailed, prefer CsRank)",
    "$yoy_rev": "year-on-year revenue growth of the latest published report, % (heavy-tailed, prefer CsRank)",
    "$cfo_to_book": "operating cash flow per share / book value per share, latest report (earnings quality)",
    "$ind_return": "that day's equal-weight return of the stock's industry among index members "
                   "(industry momentum / reversal; stock return minus it is the industry-relative move)",
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand_features(cfg: dict) -> None:
    data = cfg["data"]
    extra = {"+fund": FUND_FEATURES, "+fq": {**FUND_FEATURES, **FQ_FEATURES}}.get(data.get("features"))
    if extra is not None:
        base = ["$open", "$high", "$low", "$close", "$volume", "$amt", "$vwap", "$returns"]
        data["features"] = base + list(extra)
        data["feature_descriptions"] = dict(extra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaigns", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, help="name=JSON override")
    parser.add_argument("--protocol", action="append", required=True, help="name[:replicates]")
    parser.add_argument("--template", default="marginal", help="template arm (default marginal)")
    args = parser.parse_args()

    arms = {name: json.loads(spec) for name, spec in (a.split("=", 1) for a in args.arm)}
    for protocol_spec in args.protocol:
        protocol, _, replicates = protocol_spec.partition(":")
        replicates = int(replicates or 1)
        for n in range(1, replicates + 1):
            suffix = f"_c{n}" if protocol == "classic" else ""
            template = args.campaigns / f"{protocol}_{args.template}{suffix}.yaml"
            base = yaml.safe_load(template.read_text())
            for arm, override in arms.items():
                cfg = _merge(base, override)
                _expand_features(cfg)
                name = f"{protocol}_{arm}{suffix}"
                cfg["output_dir"] = str(args.campaigns.resolve() / name)
                target = args.campaigns / f"{name}.yaml"
                target.write_text(yaml.safe_dump(cfg, sort_keys=False))
                print(target)


if __name__ == "__main__":
    main()
