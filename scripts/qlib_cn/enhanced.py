"""Benchmark-relative (index-enhancement) books: float-cap index weights tilted by the score.

    $QPY -I scripts/qlib_cn/enhanced.py --universe csi500 --book 'alpha158_h5_lgb_s[0-6]' \\
        --tilt excl0.2 --tilt exp0.5 --rate 1,0.3 --compare output/qlib_cn/family/decay_csi500.daily.parquet:h5

A top-k long-only book can only collect the score's winners; the signal now
sits in its losers (docs/qlib-cn-benchmark.md, Round 4). A book that starts
from the index's own weights can harvest the losers as underweights, down to
zero weight. Target weight on day t, from the previous close's float caps
(``float_cap.py``) and the score dated t-1 (Qlib's signal shift):

* ``excl<q>``: cap x 1[score percentile > q], renormalised (drop the bottom q);
* ``exp<l>``: cap x exp(l x z), z the normal quantile of the score percentile;
* ``excl<q>+exp<l>``: both.

Execution mirrors the Qlib exchange the TopkDropout books use: trades at the
close (deal price close), buys cost 5 bps and sells 15 bps, no trade in a
suspended name (no volume) and no buy at +9.5% or sell at -9.5% (Qlib's
limit_threshold on $change). Each day the book moves a fraction ``--rate`` of
the way from its drifted weights to the target (partial trading towards an
aim portfolio, Garleanu-Pedersen), except names already within
``--threshold`` x their target weight of it (a no-trade region against the
noise of the float-share estimate); untradable names keep their weight and the
tradable ones absorb the difference. Weights drift with close-to-close returns.

Reported per book: after-cost excess over the index (annualised mean, IR,
tracking error), its excess over the float-cap replica (``cap`` book, the same
simulation without a tilt), and one-sided turnover. The CSI indices are price
indices while the holdings' returns (Qlib's adjusted close) include
dividends, so even the replica beats its index by about the dividend yield;
excess over the replica is the honest alpha, for the ``--compare`` books too. ``--compare path:book`` adds a ``book_eval`` daily series
(e.g. the top-50 book) and the paired block-bootstrap interval of the
difference in replica-relative IR. (IR over the price index favours
low-tracking-error books: about 2%/yr of dividends on 2.5% tracking error is
0.8 of IR for nothing.)
Daily series go to ``--out`` (.daily.parquet next to the .json) for
``cross_universe.py``-style pooling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_eval import DAYS, _ir_diff_interval, _runs, _signal  # noqa: E402
from run_qlib_baselines import BENCHMARK_INDEX  # noqa: E402

BUY_COST, SELL_COST, LIMIT = 0.0005, 0.0015, 0.095


def _tilt(spec: str, pct: np.ndarray) -> np.ndarray:
    mult = np.ones_like(pct)
    for part in spec.split("+"):
        if part == "cap":
            continue
        if part.startswith("excl"):
            mult = mult * (pct > float(part[4:]))
        elif part.startswith("exp"):
            z = norm.ppf(np.clip(pct - 0.5 / max(len(pct), 1), 1e-4, 1 - 1e-4))
            mult = mult * np.exp(float(part[3:]) * z)
        else:
            raise SystemExit(f"unknown tilt {part!r}")
    return mult


def simulate(spec: str, rate: float, threshold: float, score: pd.DataFrame, cap: pd.DataFrame, ret: pd.DataFrame,
             member: pd.DataFrame, buyable: pd.DataFrame, sellable: pd.DataFrame) -> pd.DataFrame:
    """Daily (gross, cost, turnover) of the tilted index book. All frames: dates x instruments, aligned.

    Row t of ``score`` and ``cap`` is what is known at the close of t-1; trades happen at the close of t;
    ``ret`` row t+1 is earned on the weights held after trading at t.
    """
    dates, names = score.index, score.columns
    score, cap, ret, member, buyable, sellable = (f.to_numpy() for f in (score, cap, ret, member, buyable, sellable))
    w = np.zeros(len(names))
    rows = []
    for i in range(len(dates) - 1):
        m = member[i].astype(bool) & np.isfinite(cap[i])
        target = np.zeros(len(names))
        if m.any():
            s = score[i]
            pct = np.full(len(names), 0.5)
            scored = m & np.isfinite(s)
            if scored.any():
                pct[scored] = pd.Series(s[scored]).rank(pct=True).to_numpy()
            target[m] = cap[i][m] * _tilt(spec, pct[m])
            target /= target.sum()
        if w.sum() == 0:
            new = target.copy()
        else:
            new = w + rate * (target - w) if rate < 1 else target.copy()
            # no-trade region: leave a name whose weight is within threshold x its target of the target
            hold = np.abs(target - w) <= threshold * target
            new[hold] = w[hold]
            frozen = ~(buyable[i] | sellable[i])
            no_buy = (new > w) & ~buyable[i]
            no_sell = (new < w) & ~sellable[i]
            fixed = frozen | no_buy | no_sell
            new[fixed] = w[fixed]
            free = ~fixed & (new > 0)
            room = 1 - new[fixed].sum()
            if free.any() and new[free].sum() > 0:
                new[free] *= room / new[free].sum()
        trade = new - w
        cost = trade.clip(min=0).sum() * BUY_COST - trade.clip(max=0).sum() * SELL_COST
        r = np.nan_to_num(ret[i + 1])
        gross = float(new @ r)
        rows.append((dates[i + 1], gross, cost, trade.clip(min=0).sum()))
        w = new * (1 + r)
        w = w / w.sum() if w.sum() > 0 else w
    return pd.DataFrame(rows, columns=["datetime", "gross", "cost", "turnover"]).set_index("datetime")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    parser.add_argument("--caps", type=Path, default=Path("~/fm-data/float_cap.parquet"))
    parser.add_argument("--universe", required=True)
    parser.add_argument("--runs", type=Path, default=Path("output/qlib_cn"))
    parser.add_argument("--years", default="2018,2020,2022,2024", help="walk-forward window years")
    parser.add_argument("--book", required=True, help="run glob(s) of the score, book_eval syntax")
    parser.add_argument("--tilt", action="append", default=[], help="excl<q>, exp<l> or both joined by +")
    parser.add_argument("--rate", default="1", help="comma-separated daily trading rates towards the target")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="no-trade region: skip a name within this fraction of its target weight")
    parser.add_argument("--compare", action="append", default=[], help="book_eval .daily.parquet:book")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(args.provider.expanduser()), region="cn")
    years = [int(y) for y in args.years.split(",")]
    score = pd.concat([_signal(_runs(args.runs / args.universe / f"wf{y}", args.book), 0) for y in years])
    start, end = str(score.index.get_level_values("datetime").min().date()), f"{years[-1] + 1}-12-31"
    spans = D.list_instruments(D.instruments(args.universe), start_time=start, end_time=end, as_list=False)
    names = sorted(spans)
    # every name's own bars, also outside its membership: a name leaving the index still earns its return
    fields = D.features(names, ["$close/Ref($close, 1) - 1", "$volume", "$change"], start_time=start, end_time=end)
    fields.columns = ["ret", "volume", "change"]
    fields = fields.swaplevel().sort_index()
    wide = {c: fields[c].unstack("instrument").reindex(columns=names) for c in fields}
    dates = wide["ret"].index
    member = pd.DataFrame(False, index=dates, columns=names)
    for name, periods in spans.items():
        for first, last in periods:
            member.loc[pd.Timestamp(first):pd.Timestamp(last), name] = True
    live = wide["volume"].fillna(0) > 0
    buyable = live & (wide["change"].fillna(0) < LIMIT)
    sellable = live & (wide["change"].fillna(0) > -LIMIT)
    caps = pd.read_parquet(args.caps.expanduser())["float_cap"].unstack("instrument")
    cap = caps.reindex(columns=names).reindex(dates, method="ffill").shift(1)
    sig = score.unstack("instrument").reindex(index=dates, columns=names).shift(1)
    tested = sig.notna().any(axis=1)
    sl = slice(tested.idxmax(), None)
    frames = {k: v.loc[sl] for k, v in dict(sig=sig, cap=cap, ret=wide["ret"], member=member,
                                             buyable=buyable, sellable=sellable).items()}
    index = D.features([BENCHMARK_INDEX[args.universe]], ["$close/Ref($close, 1) - 1"], start_time=start,
                       end_time=end).iloc[:, 0].droplevel(0)

    rates = [float(r) for r in args.rate.split(",")]
    specs = [("cap", 1.0)] + [(t, r) for t in args.tilt for r in rates]
    daily = {}
    for spec, rate in specs:
        sim = simulate(spec, rate, args.threshold, frames["sig"], frames["cap"], frames["ret"], frames["member"],
                       frames["buyable"], frames["sellable"])
        key = spec if spec == "cap" else f"{spec}@r{rate:g}"
        daily[key] = sim.assign(excess=sim["gross"] - sim["cost"] - index.reindex(sim.index))
        print(f"{key} done", file=sys.stderr, flush=True)
    excess = pd.DataFrame({k: v["excess"] for k, v in daily.items()}).dropna()
    for item in args.compare:
        path, book = item.rsplit(":", 1)
        other = pd.read_parquet(path)["excess"].unstack("book")[book]
        excess[f"cmp:{book}"] = other
    excess = excess.dropna()

    def ir(x: pd.Series) -> float:
        return float(x.mean() / x.std() * DAYS ** .5)

    results = {}
    print(f"{args.universe} {excess.index.min().date()}..{excess.index.max().date()} ({len(excess)} days)")
    for key in excess:
        x = excess[key]
        row = {"annualized_excess": float(x.mean() * DAYS), "information_ratio": ir(x),
               "tracking_error": float(x.std() * DAYS ** .5)}
        if key in daily:
            row.update(turnover=float(daily[key]["turnover"].mean()), cost=float(daily[key]["cost"].mean() * DAYS))
        if key != "cap":
            vs_cap = x - excess["cap"]
            row.update(over_replica=float(vs_cap.mean() * DAYS), over_replica_ir=ir(vs_cap))
        for other in [c for c in excess if c.startswith("cmp:")]:
            if key not in (other, "cap"):
                a, b = (x - excess["cap"]).to_numpy(), (excess[other] - excess["cap"]).to_numpy()
                lo, hi = _ir_diff_interval(a, b)
                row[f"dIR_vs_{other[4:]}"] = [round(ir(x - excess["cap"]) - ir(excess[other] - excess["cap"]), 3),
                                              round(lo, 3), round(hi, 3)]
        results[key] = row
        print(f"  {key:22s} " + " ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"universe": args.universe, "years": years, "book": args.book,
                                        "results": results}, indent=2) + "\n")
        excess.rename_axis("book", axis=1).stack().rename("excess").to_frame().to_parquet(
            args.out.with_suffix(".daily.parquet"))


if __name__ == "__main__":
    main()
