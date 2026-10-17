"""Diagnostics of a book's signal: alpha decay by lag, and alpha net of size exposure.

    $QPY -I scripts/qlib_cn/book_diagnostics.py decay --universe csi300 --book h5='alpha158_h5_lgb_s[0-6]'
    $QPY -I scripts/qlib_cn/book_diagnostics.py size --universe csi1000 \\
        --daily output/qlib_cn/family/ew90_csi1000.daily.parquet --baseline h1
    $QPY -I scripts/qlib_cn/book_diagnostics.py curve --universe csi500 --book h5='alpha158_h5_lgb_s[0-6]'
    $QPY -I scripts/qlib_cn/book_diagnostics.py reversal --universe csi500
    $QPY -I scripts/qlib_cn/book_diagnostics.py screen --universe csi500 --book h5='alpha158_h5_lgb_s[0-6]' \\
        --expr 'ovn20=Sum(Log($open/Ref($close, 1)), 20)'

``decay``: for the book's walk-forward score on day t, the mean daily rank IC
and top-50 spread (bps, against the universe's equal-weight mean) with the
k-th daily return after entry, r_{t+k} = close(t+k+1)/close(t+k) - 1, for
k = 1..20. The profile is the predictability curve a training label trades
off: later lags add return noise for whatever signal they still carry.

``curve``: the cross-sectional shape of the signal. Mean 5-day excess return
(lags 1-5, against the equal-weight mean) by score ventile, then frictionless
long-only books rebalanced daily: equal-weight top-k for several k and the
universe minus its bottom q. For each book, the mean daily excess (lag 1), its
IR, the one-sided daily membership turnover and the break-even one-way cost
(mean excess / turnover / 2). The gap between these IRs and book_eval's is
what costs, the n_drop limit and execution take.

``reversal``: by calendar year, the mean 5-day excess return (lags 1-5) of
the extreme deciles of the past 5-day return, and the reversal RankIC
(-corr(past 5-day return, next 5-day excess)). Separates the long side of
short-term reversal (buying recent losers) from its short side (avoiding
recent winners), which a long-only book cannot trade.

``screen``: long-side value of candidate features (Qlib expressions, or
``--feature`` columns of a (datetime, instrument) parquet, known at close t)
for a top-of-book. For each feature, the 5-day excess (lags 1-5,
against the equal-weight mean) of its top and bottom deciles, and the spread
between the feature's upper and lower half among the book's candidates (the
score's top 20%, the names a long-only book can buy): the information it adds
where the book trades. t uses the daily series with the variance inflated by
the five-day overlap.

``size``: each book's daily after-cost excess (over the cap-weighted index)
regressed on the equal-weight universe minus the index. The intercept is the
book's alpha net of its small-cap exposure; for books against ``--baseline``
the paired difference gets the same adjustment, and the IR difference of the
size residuals gets a paired block-bootstrap interval per universe and pooled
(day-averaged residuals). Several universes: ``--universe a,b`` with one
``--daily`` each.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_eval import (  # noqa: E402
    DAYS,
    UNIVERSE_SIZE,
    _ir,
    _ir_diff_interval,
    _nw_t,
    _runs,
    _signal,
)
from run_qlib_baselines import BENCHMARK_INDEX, TOPK  # noqa: E402

WINDOWS = (2018, 2020, 2022, 2024)
LAGS = 20


def _score(root: Path, universe: str, patterns: str) -> pd.Series:
    return pd.concat([_signal(_runs(root / universe / f"wf{y}", patterns), 0) for y in WINDOWS]).sort_index()


def _span() -> dict[str, str]:
    return dict(start_time=f"{WINDOWS[0]}-01-01", end_time=f"{WINDOWS[-1] + 1}-12-31")


def decay(args) -> None:
    from qlib.data import D

    exprs = [f"Ref($close, -{k + 1})/Ref($close, -{k}) - 1" for k in range(1, LAGS + 1)]
    returns = D.features(D.instruments(args.universe), exprs, **_span()).swaplevel().sort_index()
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


def curve(args) -> None:
    from qlib.data import D

    exprs = [f"Ref($close, -{k + 1})/Ref($close, -{k}) - 1" for k in range(1, 6)]
    returns = D.features(D.instruments(args.universe), exprs, **_span()).swaplevel().sort_index()
    returns.columns = range(1, 6)
    n = UNIVERSE_SIZE[args.universe]
    for spec in args.book:
        name, patterns = spec.split("=", 1)
        frame = returns.join(_score(args.out, args.universe, patterns).rename("s"), how="inner")
        frame = frame[frame[1].notna()]
        frame["r5"] = frame[list(range(1, 6))].sum(axis=1, min_count=5)
        day = frame.groupby(level="datetime")
        pct = day["s"].rank(pct=True)
        rank = day["s"].rank(ascending=False, method="first")
        x1 = frame[1] - day[1].transform("mean")
        x5 = frame["r5"] - day["r5"].transform("mean")
        bucket = np.minimum((pct * 20).astype(int), 19)
        curve = x5.groupby(bucket).mean() * 1e4
        print(f"{name} 5-day excess bps by score ventile (low->high): {np.round(curve.to_numpy(), 1).tolist()}")
        books = {f"top{k}": rank <= k for k in (10, 25, 50, 100, n // 5, n // 2)}
        books.update({f"ex-bottom{q}%": pct > q / 100 for q in (10, 20, 40)})
        for label, member in books.items():
            daily = x1[member].groupby(level="datetime").mean()
            held = member.unstack("instrument", fill_value=False).astype(float)
            turnover = (held.diff().clip(lower=0).sum(axis=1) / held.sum(axis=1)).iloc[1:].mean()
            print(f"  {label:13s} {daily.mean() * 1e4:6.2f} bps/day  IR {daily.mean() / daily.std() * DAYS ** .5:5.2f}"
                  f"  turnover {turnover:5.1%}/day  break-even {daily.mean() / turnover / 2 * 1e4:5.1f} bps one-way")


def reversal(args) -> None:
    from qlib.data import D

    exprs = ["Ref($close, -6)/Ref($close, -1) - 1", "$close/Ref($close, 5) - 1"]
    frame = D.features(D.instruments(args.universe), exprs, start_time="2010-01-01", end_time="2025-12-31")
    frame.columns = ["r5", "past5"]
    frame = frame.swaplevel().sort_index().dropna()
    day = frame.groupby(level="datetime")
    frame["x5"] = frame["r5"] - day["r5"].transform("mean")
    frame["decile"] = np.minimum((day["past5"].rank(pct=True) * 10).astype(int), 9)
    year = frame.index.get_level_values("datetime").year
    table = frame.groupby([year, "decile"])["x5"].mean().unstack() * 1e4
    ic = -frame.groupby(level="datetime").apply(lambda d: d["past5"].corr(d["x5"], method="spearman"))
    table = pd.DataFrame({"losers bps": table[0], "winners bps": table[9],
                          "RankIC x100": ic.groupby(ic.index.year).mean() * 100})
    print(f"{args.universe}: next 5-day excess of past-5-day return deciles, and reversal RankIC")
    print(table.round(1).to_string())


def screen(args) -> None:
    from qlib.data import D

    pairs = [e.split("=", 1) for e in args.expr]
    names, exprs = [n for n, _ in pairs], [e for _, e in pairs]
    ret = "Ref($close, -6)/Ref($close, -1) - 1"
    frame = D.features(D.instruments(args.universe), [ret, *exprs], **_span())
    frame.columns = ["r5", *names]
    frame = frame.swaplevel().sort_index()
    for spec in args.feature:
        name, source = spec.split("=", 1)
        path, column = source.rsplit(":", 1)
        frame[name] = pd.read_parquet(Path(path).expanduser(), columns=[column])[column].reindex(frame.index)
        names.append(name)
    name, patterns = args.book[0].split("=", 1)
    frame = frame.join(_score(args.out, args.universe, patterns).rename("s"), how="inner")
    frame = frame[frame["r5"].notna()]
    day = frame.groupby(level="datetime")
    frame["x5"] = frame["r5"] - day["r5"].transform("mean")
    if args.cand:  # the two-stage candidate set: top N names by score
        pool = day["s"].rank(ascending=False, method="first") <= args.cand
        where = f"the top {args.cand} names by {name}"
    else:
        pool = day["s"].rank(pct=True) > 0.8
        where = f"the top 20% by {name}"

    def t(daily: pd.Series) -> float:
        daily = daily.dropna()
        return daily.mean() / daily.std() * (len(daily) / 5) ** .5

    print(f"{args.universe} {WINDOWS}: 5-day excess bps; 'in book pool' = upper minus lower half of the feature "
          f"within {where}")
    for feature in names:
        pct = day[feature].rank(pct=True)
        top = frame["x5"][pct > 0.9].groupby(level="datetime").mean() * 1e4
        bottom = frame["x5"][pct <= 0.1].groupby(level="datetime").mean() * 1e4
        inner = frame[pool & pct.notna()]
        upper = inner.groupby(level="datetime")[feature].transform("median") < inner[feature]
        split = (inner["x5"][upper].groupby(level="datetime").mean()
                 - inner["x5"][~upper].groupby(level="datetime").mean()) * 1e4
        rho = frame[[feature, "s"]].dropna().groupby(level="datetime").apply(
            lambda d: d[feature].corr(d["s"], method="spearman")).mean()
        rho_pool = inner[[feature, "s"]].dropna().groupby(level="datetime").apply(
            lambda d: d[feature].corr(d["s"], method="spearman")).mean()
        coverage = float(frame.loc[pool, feature].notna().mean())
        print(f"  {feature:10s} top decile {top.mean():6.1f} (t {t(top):5.2f})  bottom decile {bottom.mean():6.1f}"
              f" (t {t(bottom):5.2f})  in book pool {split.mean():6.1f} (t {t(split):5.2f})  rho with score {rho:5.2f}"
              f" (in pool {rho_pool:5.2f})  pool coverage {coverage:.2f}")


def _size_residuals(universe: str, path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Daily after-cost excess of every book in ``path`` and the universe's equal-weight-minus-index return."""
    from qlib.data import D

    daily = pd.read_parquet(path)["excess"].unstack("book")
    span = dict(start_time=str(daily.index.min().date()), end_time=str(daily.index.max().date()))
    expr = ["$close/Ref($close, 1) - 1"]
    ew = D.features(D.instruments(universe), expr, **span).iloc[:, 0].groupby(level="datetime").mean()
    index = D.features([BENCHMARK_INDEX[universe]], expr, **span).iloc[:, 0].droplevel(0)
    frame = daily.join((ew - index).rename("_size"), how="inner").dropna()
    return frame.drop(columns="_size"), frame["_size"]


def _residual(y: pd.Series, x: pd.Series) -> tuple[pd.Series, float]:
    design = np.c_[np.ones(len(x)), x.to_numpy()]
    beta = float(np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0][1])
    return y - beta * x, beta


def size(args) -> None:
    universes = args.universe.split(",")
    if len(universes) != len(args.daily):
        raise SystemExit("size: one --daily per --universe entry")
    resid = {}
    for universe, path in zip(universes, args.daily):
        books, x = _size_residuals(universe, path)
        print(f"{universe}: equal-weight minus index {x.mean() * DAYS:.1%}/yr")

        def report(label: str, y: pd.Series) -> None:
            r, beta = _residual(y, x)
            print(f"  {label:20s} raw {y.mean() * DAYS:6.1%}/yr IR {y.mean() / y.std() * DAYS ** .5:5.2f} t {_nw_t(y.to_numpy(), 7):5.2f}"
                  f" | beta {beta:5.2f} | net of size {r.mean() * DAYS:6.1%}/yr"
                  f" IR {r.mean() / r.std() * DAYS ** .5:5.2f} t {_nw_t(r.to_numpy(), 7):5.2f}")

        for book in books.columns:
            report(book, books[book])
        resid[universe] = pd.DataFrame({b: _residual(books[b], x)[0] for b in books.columns})
        if args.baseline:
            for book in books.columns.drop(args.baseline):
                report(f"{book}-{args.baseline}", books[book] - books[args.baseline])
    if not args.baseline:
        return
    print(f"size-adjusted IR difference against {args.baseline} (90% paired block-bootstrap interval)")
    pooled = pd.concat(resid.values(), axis=1, keys=resid).T.groupby(level=1).mean().T.dropna()
    for book in pooled.columns.drop(args.baseline):
        cells = []
        for universe, r in [*resid.items(), ("pooled", pooled)]:
            a, b = r[book].to_numpy(), r[args.baseline].to_numpy()
            lo, hi = _ir_diff_interval(a, b)
            cells.append(f"{universe} {_ir(a[None])[0] - _ir(b[None])[0]:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        print(f"  {book:20s} " + "  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["decay", "curve", "reversal", "screen", "size"])
    parser.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    parser.add_argument("--universe", required=True)
    parser.add_argument("--book", action="append", default=[], help="decay/curve: name=glob[,glob...]")
    parser.add_argument("--expr", action="append", default=[], help="screen: name=Qlib expression")
    parser.add_argument("--feature", action="append", default=[],
                        help="screen: name=path.parquet:column, a (datetime, instrument) feature known at close")
    parser.add_argument("--years", default=None, help="decay/curve: walk-forward window years (default 2018-2024)")
    parser.add_argument("--cand", type=int, default=0,
                        help="screen: judge features within the top N names by score (0 = the top 20%)")
    parser.add_argument("--out", type=Path, default=Path("output/qlib_cn"), help="decay: runs root")
    parser.add_argument("--daily", type=Path, action="append", default=[],
                        help="size: a book_eval .daily.parquet per --universe entry (comma-separated)")
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()

    global WINDOWS
    if args.years:
        WINDOWS = tuple(int(y) for y in args.years.split(","))
    import qlib

    qlib.init(provider_uri=str(args.provider.expanduser()), region="cn")
    {"decay": decay, "curve": curve, "reversal": reversal, "screen": screen, "size": size}[args.command](args)


if __name__ == "__main__":
    main()
