"""Pool paired book differences across universes from book_eval daily files.

    python -I scripts/qlib_cn/cross_universe.py --baseline h1 \\
        output/qlib_cn/book_wf_horizon.daily.parquet output/qlib_cn/book_csi500_horizon.daily.parquet

For every book present in all files, the daily after-cost excess difference
to the baseline is averaged across universes day by day (the universes trade
the same days, so their differences are not independent draws; the average
is one combined book), then summarised with the Newey-West t at bandwidths
7, 20 and 60 days, the pooled IR difference and its paired block-bootstrap
90% interval, and Stouffer's combination of the per-universe t values for
comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_lift import _hac_t  # noqa: E402
from book_eval import _ir, _ir_diff_interval, _nw_t  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", type=Path, default=None, help="also write the table as JSON")
    args = parser.parse_args()
    frames = [pd.read_parquet(f)["excess"].unstack("book") for f in args.files]
    books = sorted(set.intersection(*(set(f.columns) for f in frames)) - {args.baseline})
    table = {}
    print(f"{'book':10s} {'t_nw7':>6s} {'t_nw20':>6s} {'t_nw60':>6s} {'dIR':>6s} {'ci90':>16s} {'stouffer':>8s} | per-universe t")
    for book in books:
        diffs = [f[book] - f[args.baseline] for f in frames]
        pooled = pd.concat(diffs, axis=1).mean(axis=1).dropna().to_numpy()
        a = pd.concat([f[book] for f in frames], axis=1).mean(axis=1).dropna().to_numpy()
        b = pd.concat([f[args.baseline] for f in frames], axis=1).mean(axis=1).dropna().to_numpy()
        ts = [_hac_t(d.dropna().to_numpy()) for d in diffs]
        table[book] = {"t_nw7": _hac_t(pooled), "t_nw20": _nw_t(pooled, 20), "t_nw60": _nw_t(pooled, 60),
                       "ir": float(_ir(a)), "ir_base": float(_ir(b)), "ir_diff": float(_ir(a) - _ir(b)),
                       "ir_diff_90ci": _ir_diff_interval(a, b), "stouffer": sum(ts) / math.sqrt(len(ts)),
                       "per_universe_t": dict(zip(map(str, args.files), ts))}
        print(f"{book:10s} {_hac_t(pooled):6.2f} {_nw_t(pooled, 20):6.2f} {_nw_t(pooled, 60):6.2f} "
              f"{float(_ir(a) - _ir(b)):6.2f} {str([round(c, 2) for c in _ir_diff_interval(a, b)]):>16s} "
              f"{sum(ts) / math.sqrt(len(ts)):8.2f} | " + " ".join(f"{t:.2f}" for t in ts))
    if args.out:
        args.out.write_text(json.dumps({"baseline": args.baseline, "files": list(map(str, args.files)),
                                        "books": table}, indent=2) + "\n")


if __name__ == "__main__":
    main()
