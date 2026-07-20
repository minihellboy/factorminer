"""Import market data from Qlib-style per-instrument CSV dumps.

Qlib's CSV-dump interchange format (the input shape expected by
``qlib/scripts/dump_bin.py`` before it is compiled into Qlib's binary
provider) is one CSV file per instrument, named ``<instrument>.csv``, with
at least ``date,open,high,low,close,volume`` columns and no per-row
instrument identifier -- the instrument is implied by the filename.

This module reads a directory of such dumps and reshapes it into
FactorMiner's canonical long-format schema (see
:mod:`factorminer.data.loader`), so it can be used anywhere a regular
``load_market_data`` result is used (``mine``, ``evaluate``,
``validate-data``, ...). It is an *import* adapter only: FactorMiner does
not write Qlib's binary provider format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from factorminer.data.loader import COLUMN_ALIASES, _coerce_types, _validate_columns

logger = logging.getLogger(__name__)


def _find_column(df: pd.DataFrame, name: str, aliases: tuple[str, ...]) -> str | None:
    """Return the actual column name in *df* matching *name* or one of *aliases*.

    Matching is case-insensitive, mirroring
    :func:`factorminer.data.loader._validate_columns`.
    """
    lower_map = {c.lower().strip(): c for c in df.columns}
    for candidate in (name, *aliases):
        found = lower_map.get(candidate.lower().strip())
        if found is not None:
            return found
    return None


def load_qlib_dump(
    directory: str | Path,
    *,
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    """Load a directory of Qlib-style per-instrument CSV dumps.

    Each file is expected to be named ``<instrument>.csv`` and to contain at
    least ``date,open,high,low,close,volume`` columns (Qlib's common CSV
    dump interchange shape; the standard column aliases in
    :data:`factorminer.data.loader.COLUMN_ALIASES` are also accepted, e.g.
    ``timestamp`` for ``date``). The instrument identifier is *not* read
    from inside the file -- it is derived from the filename stem.

    Qlib dumps rarely carry a turnover/notional column. When neither
    ``amount`` nor one of its known aliases (``amt``, ``turnover``,
    ``value``, ``traded_amount``) is present, ``amount`` is *approximated*
    as ``close * volume``. This is a standard proxy for traded notional, not
    a substitute for true turnover data -- callers that need exact turnover
    must supply a dump that already includes it. This mirrors the honesty
    convention used for other derived columns in
    ``docs/reproducibility.md``.

    Parameters
    ----------
    directory : str or Path
        Directory containing one ``<instrument>.csv`` file per asset.
    instruments : list of str, optional
        If given, only files whose stem matches one of these identifiers
        (case-insensitive) are loaded. By default every ``*.csv`` file in
        the directory is loaded.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same canonical column set, ordering, and
        dtypes as :func:`factorminer.data.loader.load_market_data`
        (``datetime``, ``asset_id``, ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``amount``), sorted by ``(datetime, asset_id)``.

    Raises
    ------
    NotADirectoryError
        If *directory* does not exist or is not a directory.
    FileNotFoundError
        If no matching per-instrument CSV files are found.
    ValueError
        If a file is missing required OHLCV columns (and amount cannot be
        derived because ``close``/``volume`` are also missing).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Qlib dump directory not found: {directory}")

    csv_paths = sorted(directory.glob("*.csv"))
    if instruments is not None:
        wanted = {i.strip().upper() for i in instruments}
        csv_paths = [p for p in csv_paths if p.stem.upper() in wanted]

    if not csv_paths:
        suffix = f" matching instruments={instruments}" if instruments else ""
        raise FileNotFoundError(
            f"No per-instrument CSV files found in {directory}{suffix}"
        )

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        raw = pd.read_csv(path)
        raw = raw.copy()
        raw["asset_id"] = path.stem

        amount_col = _find_column(raw, "amount", tuple(COLUMN_ALIASES["amount"]))
        if amount_col is None:
            close_col = _find_column(raw, "close", tuple(COLUMN_ALIASES["close"]))
            volume_col = _find_column(raw, "volume", tuple(COLUMN_ALIASES["volume"]))
            if close_col is not None and volume_col is not None:
                close_vals = pd.to_numeric(raw[close_col], errors="coerce")
                volume_vals = pd.to_numeric(raw[volume_col], errors="coerce")
                raw["amount"] = close_vals * volume_vals
                logger.info(
                    "%s: no turnover/amount column found; approximating "
                    "amount = close * volume",
                    path.name,
                )

        frames.append(_validate_columns(raw, path))

    df = pd.concat(frames, ignore_index=True)
    df = _coerce_types(df)
    df = df.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    logger.info(
        "Loaded Qlib dump from %s: %d rows, %d instruments",
        directory, len(df), df["asset_id"].nunique(),
    )
    return df
