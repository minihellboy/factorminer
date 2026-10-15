"""Diagnostics of a book's signal: alpha decay by lag, and alpha net of size exposure.

    $QPY -I scripts/qlib_cn/book_diagnostics.py decay --universe csi300 --book h5='alpha158_h5_lgb_s[0-6]'
    $QPY -I scripts/qlib_cn/book_diagnostics.py size --universe csi1000 \\
        --daily output/qlib_cn/family/ew90_csi1000.daily.parquet --baseline h1

``decay``: for the book's walk-forward score on day t, the mean daily rank IC
and top-50 spread (bps, against the universe's equal-weight mean) with the
k-th daily return after entry, r_{t+k} = close(t+k+1)/close(t+k) - 1, for
k = 1..20. The profile is the predictability curve a training label trades
off: later lags add return noise for whatever signal they still carry.

``size``: each book's daily after-cost excess (over the cap-weighted index)
regressed on the equal-weight universe minus the index. The intercept is the
book's alpha net of its small-cap exposure; for books against ``--baseline``
the paired difference gets the same adjustment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_eval import DAYS, _nw_t, _runs, _signal  # noqa: E402
from run_qlib_baselines import BENCHMARK_INDEX, TOPK  # noqa: E402

WINDOWS = (2018, 2020, 2022, 2024)
LAGS = 20


def _score(root: Path, universe: str, patterns: str) -> pd.Series:
    return pd.concat([_signal(_runs(root / universe / f"wf{y}", patterns), 0) for y in WINDOWS]).sort_index()


def decay(args) -> None:
    from qlib.data import D

    exprs = [f"Ref($close, -{k + 1})/Ref($close, -{k}) - 1" for k in range(1, LAGS + 1)]
    returns = D.features(D.instruments(args.universe), exprs, start_time=f"{WINDOWS[0]}-01-01",
                         end_time=f"{WINDOWS[-1] + 1}-12-31").swaplevel().sort_index()
    returns.columns = range(1, LAGS + 1)
    for spec in args.book:
        name, patterns = spec.split("=", 1)
        frame = returns.join(_score(args.out, args.universe, patterns).rename("s"), how="inner")
        ranks = frame.groupby(level="datetime").rank(pct=True)
        size = frame.groupby(level="datetime")["s"].transform("size")
        top = ranks["s"] > 1 - TOPK / size
        ics, spreads = [], []
        for k in range(1, LAGS + 1):
            pair = ranks[["s", k]].dropna()
            ics.append(pair.groupby(level="datetime").apply(lambda d: d["s"].corr(d[k])).mean())
            day = frame[k].groupby(level="datetime")
            spreads.append(float((frame[k][top].groupby(level="datetime").mean() - day.mean()).mean() * 1e4))
        print(f"{name} RankIC x1e3 by lag: {np.round(np.array(ics) * 1e3, 1).tolist()}")
        print(f"{name} top-{TOPK} spread bps by lag: {np.round(spreads, 2).tolist()}"
              f"  sum 1-5 {sum(spreads[:5]):.1f}, 6-{LAGS} {sum(spreads[5:]):.1f}")


def size(args) -> None:
    from qlib.data import D

    daily = pd.read_parquet(args.daily)["excess"].unstack("book")
    span = dict(start_time=str(daily.index.min().date()), end_time=str(daily.index.max().date()))
    expr = ["$close/Ref($close, 1) - 1"]
    ew = D.features(D.instruments(args.universe), expr, **span).iloc[:, 0].groupby(level="datetime").mean()
    index = D.features([BENCHMARK_INDEX[args.universe]], expr, **span).iloc[:, 0].droplevel(0)
    frame = daily.join((ew - index).rename("_size"), how="inner").dropna()
    x = frame.pop("_size").to_numpy()
    design = np.c_[np.ones_like(x), x]
    print(f"{args.universe}: equal-weight minus index {x.mean() * DAYS:.1%}/yr")

    def report(label: str, y: np.ndarray) -> None:
        beta = np.linalg.lstsq(design, y, rcond=None)[0][1]
        resid = y - beta * x
        print(f"  {label:14s} raw {y.mean() * DAYS:6.1%}/yr IR {y.mean() / y.std() * DAYS ** .5:5.2f} t {_nw_t(y, 7):5.2f}"
              f" | beta {beta:5.2f} | net of size {resid.mean() * DAYS:6.1%}/yr"
              f" IR {resid.mean() / resid.std() * DAYS ** .5:5.2f} t {_nw_t(resid, 7):5.2f}")

    for book in frame.columns:
        report(book, frame[book].to_numpy())
    if args.baseline:
        for book in frame.columns.drop(args.baseline):
            report(f"{book}-{args.baseline}", (frame[book] - frame[args.baseline]).to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["decay", "size"])
    parser.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    parser.add_argument("--universe", required=True)
    parser.add_argument("--book", action="append", default=[], help="decay: name=glob[,glob...]")
    parser.add_argument("--out", type=Path, default=Path("output/qlib_cn"), help="decay: runs root")
    parser.add_argument("--daily", type=Path, help="size: a book_eval .daily.parquet")
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()

    import qlib

    qlib.init(provider_uri=str(args.provider.expanduser()), region="cn")
    {"decay": decay, "size": size}[args.command](args)


if __name__ == "__main__":
    main()
