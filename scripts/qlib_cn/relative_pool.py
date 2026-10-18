"""Put several enhanced.py runs on one score basis so cross_universe.py can pool them.

    $QPY -I scripts/qlib_cn/relative_pool.py --tilt add0.5@r0.1 --out output/qlib_cn/enhanced/pool_csi300.daily.parquet \\
        h5=output/qlib_cn/enhanced/pivot_h5_csi300.json shrink30=output/qlib_cn/enhanced/pivot_rp_shrink30_csi300.json

``enhanced.py`` writes one daily file per score with the books of that score
(``cap``, ``add0.5@r0.1``, ...). This takes one tilt from each run, subtracts
the run's replica (``cap``) so the series is the replica-relative excess, and
writes a ``book_eval``-layout daily file (``datetime, book`` index, ``excess``)
with one book per run, named as given. ``cross_universe.py --baseline h5``
then pools the paired differences across universes exactly as it does for
top-k books.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="name=path to an enhanced.py .json (its .daily.parquet is read)")
    parser.add_argument("--tilt", default="add0.5@r0.1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    columns = {}
    for item in args.runs:
        name, path = item.split("=", 1)
        daily = pd.read_parquet(Path(path).with_suffix(".daily.parquet"))["excess"].unstack("book")
        columns[name] = daily[args.tilt] - daily["cap"]
    frame = pd.DataFrame(columns).dropna()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.rename_axis("book", axis=1).stack().rename("excess").to_frame().to_parquet(args.out)
    print(f"{args.out}: {frame.shape[0]} days, books {list(frame.columns)}")


if __name__ == "__main__":
    main()
