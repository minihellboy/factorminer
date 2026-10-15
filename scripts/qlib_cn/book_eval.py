"""Score portfolio books built from saved predictions: seed/model ensembles, score smoothing, n_drop.

Run with the Qlib interpreter::

    qlib-venv/bin/python -I scripts/qlib_cn/book_eval.py --provider ~/fm-data/qlib_cn --universe csi300 \\
        --label ~/fm-data/labels/csi300.parquet --window wf2018=output/qlib_cn/csi300/wf2018 ... \\
        --book mtop='alpha158r_mtop_lgb_s0' --book mtop3='alpha158r_mtop_lgb_s*' \\
        --book blend='alpha158r_mtop_lgb_s*,alpha158r_alstm_s*' --smooth 0,3 --baseline mtop \\
        --out output/qlib_cn/book_mtop.json

A book is a comma-separated list of run-directory globs under each window's
root. Its runs' predictions are ranked per day and averaged, each *pattern*
weighing equally (seeds within a pattern share its weight), so ``a_s*,b_s*``
is a 50/50 blend of two models. ``--smooth h`` replaces the score with its
exponential moving average over trading days (half-life ``h``), which keeps
names near the top from being dropped and re-bought on noise. Each (book,
smoothing, n_drop) is backtested with the benchmark's TopkDropout book and
costs in every window; daily after-cost excess returns of the disjoint test
windows are concatenated. Against ``--baseline`` (unsmoothed), the paired
daily difference of after-cost excess return gets a Newey-West t, which
removes the market noise both books share and is far tighter than comparing
IRs across seeds. Windows must not overlap, every run of a book must cover
its window's test days, and the daily series of every book are saved next to
``--out`` (``<out>.daily.parquet``) for bootstraps and attribution. The IR
difference gets a paired circular block bootstrap (20-day blocks, the same
resampled days for both books) 90% interval, and the mean-difference t is
also reported at Newey-West bandwidths of 20 and 60 days.

``--sim continuous`` (default) runs one portfolio through all windows: each
window's models score its own test days, and holdings carry over from one
window's model to the next under the same TopkDropout rule, so no window
starts from cash. Windows must then be contiguous. ``--sim reset`` restarts
each window from cash (the earlier convention).

``--select name=bookA,bookB,...`` adds a procedure book that, in each window,
trades the listed book with the highest after-cost information ratio on that
window's validation segment (scored from each run's
``valid_predictions.parquet``; validation books always start from cash). Its
test result is an honest estimate of choosing among those books, unlike the
best book's own test result. Over all books, a paired max-statistic block
bootstrap (White's reality check) gives the p-value of the best book's mean
improvement over the baseline, corrected for having tried every book.

``--combine zscore`` averages each run's per-day z-scored predictions instead
of percentile ranks, so a model's conviction (how far a name sits from the
day's mean) carries into a blend. ``--costs stamp2023`` scales each day's
cost by 0.75 from 2023-08-28, when the sell-side stamp tax fell from 0.10% to
0.05% (close cost 0.15% -> 0.10%; buys and sells are about equal in a
TopkDropout book, so cost scales by (0.05+0.10)/(0.05+0.15)).
``--neutral industry.parquet[:w,...]`` adds, for each book, a book whose daily
score has ``w`` times its (day, industry) mean removed before ranking
(``w`` 1 is industry-neutral scoring: each industry of five or more names
centres at zero, so the top 50 no longer carries the model's bets on the
large sectors; smaller groups and names without an industry keep their
rank). The map is ``data_factors.py --industry-out``: the Eastmoney industry
of the latest report published before each day.

Tied scores are broken by a stable pseudo-random key (``_untie``), so a
backtest is reproducible; ``--tie-draws K`` averages each book's daily returns
over K tie-breaking keys (the expected book under random tie-breaking) and
reports the IR spread across draws and the share of tied scores.
Every evaluated book is also appended to ``registry.jsonl`` next to
``--out``, so the number of books ever tried on these windows is on record.
"""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_lift import _hac_t  # noqa: E402
from run_qlib_baselines import PROTOCOLS, TOPK, _backtest  # noqa: E402

DAYS = 238  # Qlib risk_analysis annualisation for daily returns
BLOCK, DRAWS = 20, 2000
UNIVERSE_SIZE = {"csi300": 300, "csi500": 500, "csi1000": 1000}


def _nw_t(x: np.ndarray, lags: int) -> float:
    e = x - x.mean()
    variance = e @ e / len(x)
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / len(x)
    return float(x.mean() / math.sqrt(variance / len(x))) if variance > 0 else float("nan")


def _ir(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1) / x.std(axis=-1) * math.sqrt(DAYS)


def _ir_diff_interval(a: np.ndarray, b: np.ndarray, seed: int = 0) -> list[float]:
    """90% paired circular block-bootstrap interval of IR(a) - IR(b)."""
    n = len(a)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(DRAWS, -(-n // BLOCK)))
    index = ((starts[:, :, None] + np.arange(BLOCK)) % n).reshape(DRAWS, -1)[:, :n]
    diff = _ir(a[index]) - _ir(b[index])
    return [float(q) for q in np.quantile(diff, [0.05, 0.95])]


def _runs(root: Path, patterns: str, file: str = "predictions.parquet") -> list[list[Path]]:
    groups = []
    for pattern in patterns.split(","):
        runs = sorted(p for p in root.iterdir() if fnmatch.fnmatch(p.name, pattern)
                      and (p / file).exists())
        if not runs:
            raise SystemExit(f"no runs match {root}/{pattern}")
        groups.append(runs)
    return groups


COMBINE = "rank"  # --combine: rank (percentile) or zscore
STAMP_CUT, STAMP_SCALE = pd.Timestamp("2023-08-28"), 0.75
COSTS = "fixed"  # --costs: fixed (Qlib's) or stamp2023
INDUSTRY: pd.Series | None = None  # --neutral: industry per (datetime, instrument)
MIN_GROUP = 5  # smaller (day, industry) groups are centred on the day's mean instead


def _neutralize(score: pd.Series, weight: float) -> pd.Series:
    """Remove ``weight`` times the (day, industry) mean of ``score``.

    The Eastmoney industries are fine (about 80 a day among 300 names, median
    two names each), so only groups of at least ``MIN_GROUP`` names are centred
    on their own mean; smaller groups and names without an industry are centred
    on the day's mean, which leaves their ranks unchanged.
    """
    if weight == 0:
        return score
    industry = INDUSTRY.reindex(score.index).fillna("_none").to_numpy()
    keys = [score.index.get_level_values("datetime"), industry]
    grouped = score.groupby(keys)
    own = (industry != "_none") & (grouped.transform("size").to_numpy() >= MIN_GROUP)
    centre = grouped.transform("mean").where(own, score.groupby(level="datetime").transform("mean"))
    return (score - weight * centre).rename("score")


def _rank(path: Path, file: str = "predictions.parquet") -> pd.Series:
    score = pd.read_parquet(path / file).set_index(["datetime", "instrument"])["score"]
    day = score.groupby(level="datetime")
    if COMBINE == "zscore":
        return ((score - day.transform("mean")) / day.transform("std")).clip(-5, 5)
    return day.rank(pct=True)


def _signal(groups: list[list[Path]], smooth: float, file: str = "predictions.parquet") -> pd.Series:
    ranks = [[_rank(r, file) for r in runs] for runs in groups]
    days = [{r.name: s.index.get_level_values("datetime").unique() for r, s in zip(runs, rs)}
            for runs, rs in zip(groups, ranks)]
    spans = {name: (d.min(), d.max(), len(d)) for g in days for name, d in g.items()}
    if len({v for v in spans.values()}) > 1:
        raise SystemExit(f"runs cover different test days: {spans}")
    parts = [pd.concat(rs, axis=1).mean(axis=1) for rs in ranks]
    score = pd.concat(parts, axis=1).mean(axis=1).rename("score")
    if smooth > 0:
        wide = score.unstack("instrument")
        ema = wide.ewm(halflife=smooth, ignore_na=True).mean().where(wide.notna())
        score = ema.stack().rename("score")
    return score.sort_index()


def _daily_top(score: pd.Series, label: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"s": score}).join(label.rename("y"), how="inner").dropna()

    def top(day: pd.DataFrame) -> float:
        return (day.nlargest(TOPK, "s")["y"].mean() - day["y"].mean()) * 1e4
    return frame.groupby(level="datetime").apply(top)


def _daily_ric(score: pd.Series, label: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"s": score}).join(label.rename("y"), how="inner").dropna()
    return frame.groupby(level="datetime").apply(lambda g: g["s"].corr(g["y"], method="spearman"))


def _reality_check(diffs: pd.DataFrame, seed: int = 0) -> dict[str, float]:
    """White's reality check: p-value of the best mean of ``diffs`` columns against all-zero means."""
    x = diffs.dropna().to_numpy()
    n = len(x)
    stat = np.sqrt(n) * x.mean(axis=0)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(DRAWS, -(-n // BLOCK)))
    index = ((starts[:, :, None] + np.arange(BLOCK)) % n).reshape(DRAWS, -1)[:, :n]
    boot = np.sqrt(n) * (x[index].mean(axis=1) - x.mean(axis=0))
    best = int(stat.argmax())
    return {"best": str(diffs.columns[best]), "books": int(x.shape[1]),
            "p_value": float((boot.max(axis=1) >= stat[best]).mean())}


def _frame(report: pd.DataFrame) -> pd.DataFrame:
    cost = report["cost"]
    if COSTS == "stamp2023":
        cost = cost.where(report.index < STAMP_CUT, cost * STAMP_SCALE)
    return pd.DataFrame({"gross": report["return"] - report["bench"],
                         "excess": report["return"] - report["bench"] - cost,
                         "turnover": report["turnover"]})


def _simulate(scores: dict[str, pd.Series], universe: str, n_drop: int, sim: str, salt: int,
              band: float | None = None, topk: int = TOPK) -> pd.DataFrame:
    if sim == "reset":
        return pd.concat([_frame(_backtest(scores[p], universe, PROTOCOLS[p], n_drop=n_drop, return_report=True,
                                           tie_salt=salt, band=band, topk=topk)[1]) for p in scores]).sort_index()
    spans = sorted(PROTOCOLS[p]["test"] for p in scores)
    score = pd.concat(scores.values()).sort_index()
    _, report = _backtest(score, universe, {"test": (spans[0][0], spans[-1][1])}, n_drop=n_drop,
                          return_report=True, tie_salt=salt, band=band, topk=topk)
    return _frame(report)


_SCORES: dict[str, tuple] = {}  # key -> (window scores, n_drop, band, topk); read by forked workers
JOBS = 1  # --jobs


def _simulate_key(key: str, universe: str, sim: str, salt: int) -> pd.DataFrame:
    scores, n_drop, band, topk = _SCORES[key]
    return _simulate(scores, universe, n_drop, sim, salt, band, topk)


def _map(fn, tasks: list[tuple]) -> list:
    """``fn(*task)`` for every task, in ``JOBS`` forked processes that share the parent's cached Exchange."""
    if JOBS <= 1 or len(tasks) <= 1:
        return [fn(*t) for t in tasks]
    import multiprocessing

    with multiprocessing.get_context("fork").Pool(min(JOBS, len(tasks))) as pool:
        return pool.starmap(fn, tasks, chunksize=1)


def _books(universe: str, windows: dict[str, str], specs: dict[str, tuple], sim: str,
           draws: int = 1) -> dict[str, tuple[pd.DataFrame, pd.Series, dict[str, float]]]:
    """Daily book returns over all windows for every ``key -> (patterns, smooth, n_drop, neutral, band, topk)``,
    each averaged over ``draws`` tie-breaking keys; all (book, draw) backtests run in parallel."""
    from run_qlib_baselines import _exchange

    _SCORES.clear()
    for key, (patterns, smooth, n_drop, neutral, band, topk) in specs.items():
        _SCORES[key] = ({p: _neutralize(_signal(_runs(Path(root), patterns[p]), smooth), neutral)
                         for p, root in windows.items()}, n_drop, band, topk)
    _exchange(universe)  # built once, before forking
    tasks = [(key, universe, sim, salt) for key in specs for salt in range(draws)]
    reports = dict(zip([(t[0], t[3]) for t in tasks], _map(_simulate_key, tasks)))
    out = {}
    for key in specs:
        runs = [reports[key, salt] for salt in range(draws)]
        frame = sum(runs) / draws
        score = pd.concat(_SCORES[key][0].values()).sort_index()
        ties = {"tie_share": float(score.groupby(level="datetime").apply(lambda d: d.duplicated().mean()).mean()),
                "tie_draw_ir": [float(r["excess"].mean() / r["excess"].std() * math.sqrt(DAYS)) for r in runs]}
        frame["window"] = None
        for p in windows:
            start, end = PROTOCOLS[p]["test"]
            frame.loc[start:end, "window"] = p
        out[key] = (frame, score, ties)
    return out


def _valid_ir(universe: str, root: Path, patterns: str, split: dict, draws: int = 1) -> float:
    score = _signal(_runs(root, patterns, "valid_predictions.parquet"), 0, "valid_predictions.parquet")
    excess = sum(_frame(_backtest(score, universe, {"test": split["valid"]}, return_report=True, tie_salt=salt)[1])
                 ["excess"] for salt in range(draws)) / draws
    return float(excess.mean() / excess.std() * math.sqrt(DAYS))


def _summary(daily: pd.DataFrame) -> dict[str, float]:
    excess = daily["excess"].to_numpy()
    return {
        "dates": int(len(daily)),
        "annualized_excess": float(excess.mean() * DAYS),
        "information_ratio": float(excess.mean() / excess.std() * math.sqrt(DAYS)),
        "gross_information_ratio": float(daily["gross"].mean() / daily["gross"].std() * math.sqrt(DAYS)),
        "turnover_mean": float(daily["turnover"].mean()),
        "top_spread_bps": float(daily["top"].mean()),
        "RankIC": float(daily["ric"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--window", action="append", required=True, help="protocol=root")
    parser.add_argument("--book", action="append", required=True, help="name=glob[,glob...]")
    parser.add_argument("--smooth", default="0", help="comma-separated EMA half-lives (0 = none)")
    parser.add_argument("--n-drop", default="5", help="comma-separated TopkDropout n_drop values")
    parser.add_argument("--size", default=None,
                        help="comma-separated book sizes as fractions of the universe: also trade each book with "
                             "topk = f x N and n_drop scaled by the same factor (n_drop/topk kept)")
    parser.add_argument("--band", default=None,
                        help="comma-separated rank bands (x topk): also trade each book with the band rule")
    parser.add_argument("--select", action="append", default=[],
                        help="name=book,book,...: per window, the book with the best validation IR")
    parser.add_argument("--sim", choices=["continuous", "reset"], default="continuous")
    parser.add_argument("--combine", choices=["rank", "zscore"], default="rank")
    parser.add_argument("--tie-draws", type=int, default=1, help="average over this many tie-breaking keys")
    parser.add_argument("--costs", choices=["fixed", "stamp2023"], default="fixed")
    parser.add_argument("--neutral", default=None,
                        help="industry.parquet[:w,w,...]: also score each book with w x its (day, industry) "
                             "mean removed (default weights 1)")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--jobs", type=int, default=8, help="parallel backtest processes")
    parser.add_argument("--tradable", action="store_true",
                        help="TopkDropout only_tradable: buy the next-ranked name instead of a suspended/limit one")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global COMBINE, COSTS, INDUSTRY, JOBS
    COMBINE, COSTS, JOBS = args.combine, args.costs, args.jobs
    import run_qlib_baselines

    run_qlib_baselines.ONLY_TRADABLE = args.tradable
    neutrals = [0.0]
    if args.neutral:
        path, _, weights = args.neutral.partition(":")
        INDUSTRY = pd.read_parquet(path).set_index(["datetime", "instrument"])["industry"]
        INDUSTRY = INDUSTRY[~INDUSTRY.index.duplicated()]
        neutrals += [float(w) for w in (weights or "1").split(",")]

    import qlib

    qlib.init(provider_uri=str(args.provider.expanduser()), region="cn")
    label = pd.read_parquet(args.label)
    label["instrument"] = label["instrument"].str.upper()
    label = label.set_index(["datetime", "instrument"])["label"]
    windows = dict(w.split("=", 1) for w in args.window)
    tests = sorted((PROTOCOLS[p]["test"], p) for p in windows)
    for (a, pa), (b, pb) in zip(tests, tests[1:]):
        if b[0] <= a[1]:
            raise SystemExit(f"test windows overlap: {pa} {a} and {pb} {b}")
        if args.sim == "continuous" and pd.Timestamp(b[0]) - pd.Timestamp(a[1]) > pd.Timedelta(days=7):
            raise SystemExit(f"--sim continuous needs contiguous windows: {pa} {a} then {pb} {b}")
    books = dict(b.split("=", 1) for b in args.book)
    selections = {}
    for spec in args.select:
        name, members = spec.split("=", 1)
        valid = {p: {m: _valid_ir(args.universe, Path(root), books[m], PROTOCOLS[p], args.tie_draws)
                     for m in members.split(",")}
                 for p, root in windows.items()}
        selections[name] = {p: {"valid_ir": v, "chosen": max(v, key=v.get)} for p, v in valid.items()}
        print(name, json.dumps(selections[name]), flush=True)
    smooths = [float(h) for h in args.smooth.split(",")]
    drops = [int(n) for n in args.n_drop.split(",")]
    bands = [None] + [float(b) for b in args.band.split(",")] if args.band else [None]
    sizes = [None] + [float(f) for f in args.size.split(",")] if args.size else [None]

    per_book = {name: {p: patterns for p in windows} for name, patterns in books.items()}
    per_book.update({name: {p: books[c["chosen"]] for p, c in chosen.items()} for name, chosen in selections.items()})
    specs = {}
    for name, patterns in per_book.items():
        for h, n_drop, w, band, size in itertools.product(smooths, drops, neutrals, bands, sizes):
            key = (name + (f"@ema{h:g}" if h else "") + (f"@drop{n_drop}" if n_drop != 5 else "")
                   + (f"@neutral{w:g}" if w else "") + (f"@band{band:g}" if band else "")
                   + (f"@size{size:g}" if size else ""))
            topk = round(size * UNIVERSE_SIZE[args.universe]) if size else TOPK
            specs[key] = (patterns, h, max(1, round(topk * n_drop / TOPK)) if size else n_drop, w, band, topk)
    daily: dict[str, pd.DataFrame] = {}
    ties: dict[str, dict] = {}
    for key, (frame, score, ties[key]) in _books(args.universe, windows, specs, args.sim, args.tie_draws).items():
        frame = frame.join(_daily_top(score, label).rename("top")).join(
            _daily_ric(score, label).rename("ric"))
        daily[key] = frame
        assert daily[key].index.is_unique
        print(key, json.dumps(_summary(daily[key])), flush=True)

    result = {"windows": list(windows), "books": books, "sim": args.sim, "combine": args.combine,
              "tradable": args.tradable,
              "tie_draws": args.tie_draws,
              "costs": args.costs, "selections": selections,
              "results": {}}
    if args.baseline and args.baseline not in daily:
        raise SystemExit(f"baseline {args.baseline} is not among the evaluated books {sorted(daily)}")
    base = daily.get(args.baseline) if args.baseline else None
    for key, frame in daily.items():
        entry = {**_summary(frame), **ties[key]}
        entry["by_window"] = {w: _summary(g) for w, g in frame.groupby("window")}
        if base is not None and key != args.baseline:
            both = frame.join(base, rsuffix="_base", how="inner")
            diff = (both["excess"] - both["excess_base"]).to_numpy()
            top = (both["top"] - both["top_base"]).to_numpy()
            ex, ex_base = both["excess"].to_numpy(), both["excess_base"].to_numpy()
            entry["vs_baseline"] = {
                "annualized_excess_diff": float(diff.mean() * DAYS), "excess_diff_t": _hac_t(diff),
                "excess_diff_t_nw20": _nw_t(diff, 20), "excess_diff_t_nw60": _nw_t(diff, 60),
                "excess_diff_drop10_best": float(np.sort(diff)[:-10].mean() * DAYS),
                "turnover_ratio": float(both["turnover"].mean() / both["turnover_base"].mean()),
                "ir_diff": float(_ir(ex) - _ir(ex_base)), "ir_diff_90ci": _ir_diff_interval(ex, ex_base),
                "top_spread_diff_bps": float(np.nanmean(top)), "top_spread_diff_t": _hac_t(top),
                "RankIC_diff": float((both["ric"] - both["ric_base"]).mean()),
                "RankIC_diff_t": _hac_t((both["ric"] - both["ric_base"]).to_numpy()),
            }
        result["results"][key] = entry
    if base is not None and len(daily) > 1:
        diffs = pd.DataFrame({k: f["excess"] - base["excess"] for k, f in daily.items() if k != args.baseline})
        result["reality_check"] = _reality_check(diffs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    pd.concat(daily, names=["book", "datetime"]).to_parquet(args.out.with_suffix(".daily.parquet"))
    with open(args.out.parent / "registry.jsonl", "a") as registry:
        for key, entry in result["results"].items():
            registry.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "out": str(args.out), "book": key,
                "runs": per_book[key.split("@")[0]], "sim": args.sim, "combine": args.combine, "costs": args.costs, "windows": list(windows),
                "information_ratio": entry["information_ratio"],
                "annualized_excess": entry["annualized_excess"]}) + "\n")


if __name__ == "__main__":
    main()
