"""Point-in-time fundamental/industry leaves for mining, and a fixed data-feature set for Qlib.

Run in an environment with ``pandas`` and ``pyarrow``::

    python -I scripts/qlib_cn/data_factors.py --panel ~/fm-data/panels/csi300_2007_2026_fund.parquet \\
        --fundamentals ~/fm-data/eastmoney --panel-out ~/fm-data/panels/csi300_2007_2026_fq \\
        --factors-out output/qlib_cn/data_factors/csi300_fq.parquet

``--panel-out`` writes the panel plus new leaves for the miner:

* quarterly report figures (``fetch_em_fundamentals.py``), taken point in
  time: a report counts from the first trading day *after* its ``pubDate``
  (announcements often come after the close) and holds until the next one.
  ``roe`` (year to date, as reported), ``gp_margin``, ``np_margin`` (net
  income / revenue), ``yoy_ni`` and ``yoy_rev`` (year-on-year net-income and
  revenue growth, %), ``cfo_to_book`` (operating cash flow per share / book
  value per share);
* ``ind_return``: the equal-weight mean daily return of the other index
  members in the stock's Eastmoney industry that day (leave-one-out; NaN with
  fewer than three peers). The industry is the one of the stock's latest
  report; before its first labelled report it is unknown. Peers are that
  day's members only, never stocks selected by later membership.

``--factors-out`` writes, for universe-member rows, a fixed set of standard
data features ranked cross-sectionally to [-1, 1] (``datetime, instrument,
...``), to append to Alpha158 without any mining: value (earnings, book,
sales and cash-flow yield), size, liquidity (turnover level, its 20-day mean
and its jump over that mean), 60-day beta and 20-day idiosyncratic volatility
against the index, industry momentum and the industry-relative 5-day
return, and the quarterly figures above plus the year-on-year ROE change.
It measures what the data alone add, the bar a mined use of it must clear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUARTERLY = ["roe", "gp_margin", "np_margin", "yoy_ni", "yoy_rev", "cfo_to_book"]
MIN_PEERS = 3


def _quarterly(directory: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per (asset, trading day) the latest report published strictly before that day."""
    q = pd.read_parquet(directory / "quarterly.parquet")
    q["pub"] = pd.to_datetime(q["pubDate"])
    q["stat"] = pd.to_datetime(q["statDate"])
    q["np_margin"] = q["net_income"] / q["revenue"].where(q["revenue"] > 0)
    q["cfo_to_book"] = q["cfps"] / q["bps"].where(q["bps"] > 0)
    q = q[["asset_id", "pub", "stat", "industry", *QUARTERLY]]
    # Same quarter one year earlier, for the year-on-year ROE change (YTD figures compare like for like).
    prior = q[["asset_id", "stat", "roe"]].assign(stat=lambda f: f["stat"] + pd.DateOffset(years=1))
    q = q.merge(prior.rename(columns={"roe": "roe_prior"}).drop_duplicates(["asset_id", "stat"]),
                on=["asset_id", "stat"], how="left")
    q["roe_yoy"] = q["roe"] - q["roe_prior"]
    # Effective from the next trading day after publication.
    position = dates.searchsorted(q["pub"].to_numpy(), side="right")
    q = q[position < len(dates)].assign(datetime=dates[position[position < len(dates)]])
    # Several reports on one day (e.g. annual + Q1 in late April): the newest statement wins.
    q = (q.sort_values(["asset_id", "datetime", "stat"]).drop_duplicates(["asset_id", "datetime"], keep="last")
         .drop(columns=["pub", "stat", "roe_prior"]))
    # A field a report leaves blank keeps the asset's last published value.
    fields = [*QUARTERLY, "roe_yoy", "industry"]
    q[fields] = q.groupby("asset_id")[fields].ffill()
    return q


def _cs_rank(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ranked = frame.groupby("datetime")[columns].rank(pct=True)
    return ranked * 2 - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fundamentals", type=Path, default=None,
                        help="directory with fetch_em_fundamentals.py's quarterly.parquet")
    parser.add_argument("--panel-out", type=Path, default=None)
    parser.add_argument("--factors-out", type=Path, default=None)
    parser.add_argument("--industry-out", type=Path, default=None,
                        help="parquet of (datetime, instrument, industry) for index members: the Eastmoney "
                             "industry of the latest report published before each day")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel = panel.sort_values(["asset_id", "datetime"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(np.sort(panel["datetime"].unique()))
    by_asset = panel.groupby("asset_id")
    # A daily return only across consecutive sessions: after a suspension gap it spans several days.
    previous = pd.Series(dates[np.maximum(dates.searchsorted(panel["datetime"]) - 1, 0)], index=panel.index)
    ret = by_asset["close"].pct_change(fill_method=None).where(by_asset["datetime"].shift(1) == previous)
    panel["ret"] = ret
    added: list[str] = []

    have_q = args.fundamentals is not None and (args.fundamentals / "quarterly.parquet").exists()
    if have_q:
        q = _quarterly(args.fundamentals, dates)
        # Backward as-of per asset: a report effective on a suspended day applies from the next row.
        panel = pd.merge_asof(panel.sort_values("datetime"), q.sort_values("datetime"), on="datetime",
                              by="asset_id", direction="backward")
        panel = panel.sort_values(["asset_id", "datetime"]).reset_index(drop=True)
        ret = panel["ret"]
        added += QUARTERLY + ["roe_yoy"]
        member = (panel["in_universe"] == 1) & panel["ret"].notna()
        key = [panel["datetime"], panel["industry"]]
        total = panel["ret"].where(member).groupby(key).transform("sum")
        count = member.groupby(key).transform("sum")
        peers = count - member.astype(int)
        own = panel["ret"].where(member, 0.0)
        panel["ind_return"] = ((total - own) / peers).where(peers >= MIN_PEERS)
        added.append("ind_return")

    if args.industry_out is not None:
        if not have_q:
            raise SystemExit("--industry-out needs --fundamentals")
        ind = panel.loc[(panel["in_universe"] == 1) & panel["industry"].notna(),
                        ["datetime", "asset_id", "industry"]].rename(columns={"asset_id": "instrument"})
        args.industry_out.parent.mkdir(parents=True, exist_ok=True)
        ind.to_parquet(args.industry_out, index=False)
        covered = float(panel.loc[panel["in_universe"] == 1, "industry"].notna().mean())
        print("industry", args.industry_out, len(ind), "industries", ind["industry"].nunique(),
              "member coverage", round(covered, 3))

    if args.panel_out is not None:
        leaves = [c for c in added if c != "roe_yoy"]
        out = panel.drop(columns=["ret", "industry", "roe_yoy"], errors="ignore")
        out = out.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
        for c in leaves:
            out[c] = out[c].astype("float32")
        args.panel_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.panel_out.with_suffix(".parquet"), index=False)
        out.to_csv(args.panel_out.with_suffix(".csv"), index=False)
        members = out[out["in_universe"] == 1]
        args.panel_out.with_suffix(".manifest.json").write_text(json.dumps({
            "panel": str(args.panel), "rows": int(len(out)),
            "added_columns": leaves, "member_coverage": {c: float(members[c].notna().mean()) for c in leaves},
            "fundamentals": str(args.fundamentals),
        }, indent=2) + "\n")
        print("panel", args.panel_out, {c: round(float(members[c].notna().mean()), 3) for c in leaves})

    if args.factors_out is not None:
        f = pd.DataFrame({"datetime": panel["datetime"], "instrument": panel["asset_id"],
                          "member": panel["in_universe"] == 1})
        for name, column in {"ep": "pe_ttm", "bp": "pb", "sp": "ps_ttm", "cfp": "pcf_ttm"}.items():
            f[name] = 1.0 / panel[column].where(panel[column] != 0)
        f["size"] = np.log(panel["mcap"].where(panel["mcap"] > 0))
        turn = panel["turnover"].where(panel["turnover"] > 0)
        turn_ma20 = turn.groupby(panel["asset_id"]).transform(lambda s: s.rolling(20, min_periods=10).mean())
        f["turn"] = turn
        f["turn_ma20"] = turn_ma20
        f["turn_jump"] = turn / turn_ma20
        index_ret = panel["index_close"] / panel.groupby("asset_id")["index_close"].shift(1) - 1
        roll = panel.assign(r=ret, m=index_ret, rm=ret * index_ret, mm=index_ret ** 2).groupby("asset_id")
        mean = {c: roll[c].transform(lambda s: s.rolling(60, min_periods=40).mean()) for c in ("r", "m", "rm", "mm")}
        beta = (mean["rm"] - mean["r"] * mean["m"]) / (mean["mm"] - mean["m"] ** 2)
        f["beta60"] = beta
        resid = ret - beta * index_ret
        f["ivol20"] = resid.groupby(panel["asset_id"]).transform(lambda s: s.rolling(20, min_periods=15).std())
        f["st"] = panel["st"].astype(float)
        if have_q:
            # Missing days stay missing: windows need most of their observations.
            log_ind = np.log1p(panel["ind_return"]).groupby(panel["asset_id"])
            f["ind_mom20"] = log_ind.transform(lambda s: s.rolling(20, min_periods=15).sum())
            ret5 = np.log1p(ret).groupby(panel["asset_id"]).transform(lambda s: s.rolling(5, min_periods=4).sum())
            f["ind_rel_ret5"] = ret5 - log_ind.transform(lambda s: s.rolling(5, min_periods=4).sum())
        if have_q:
            for c in QUARTERLY + ["roe_yoy"]:
                f[c] = panel[c]
        f = f[f["member"]].drop(columns="member")
        columns = [c for c in f.columns if c not in {"datetime", "instrument"}]
        f = f.replace([np.inf, -np.inf], np.nan)
        f[columns] = _cs_rank(f, columns).astype("float32")
        args.factors_out.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(args.factors_out, index=False)
        print("factors", args.factors_out, len(f), {c: round(float(f[c].notna().mean()), 3) for c in columns})


if __name__ == "__main__":
    main()
