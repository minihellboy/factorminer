"""Tests for the Qlib data adapter and anonymized/redacted export mode.

Covers:
    - ``load_qlib_dump``: per-instrument CSV dumps -> canonical schema.
    - ``export_formulas_qlib``: honest best-effort DSL -> Qlib translation.
    - ``export_anonymized``: formula-free redacted export.
"""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import export_anonymized
from factorminer.data.loader import REQUIRED_COLUMNS
from factorminer.data.qlib_library import export_formulas_qlib
from factorminer.data.qlib_source import load_qlib_dump

# ---------------------------------------------------------------------------
# load_qlib_dump
# ---------------------------------------------------------------------------

def _write_instrument_csv(directory, stem: str, *, with_amount: bool = False) -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = {
        "date": dates.strftime("%Y-%m-%d"),
        "open": [10.0, 10.5, 11.0, 10.8, 11.2],
        "high": [10.6, 11.0, 11.4, 11.1, 11.6],
        "low": [9.8, 10.2, 10.7, 10.5, 10.9],
        "close": [10.4, 10.9, 11.1, 10.9, 11.5],
        "volume": [1000.0, 1200.0, 900.0, 1100.0, 1300.0],
    }
    if with_amount:
        rows["turnover"] = [v * c for v, c in zip(rows["volume"], rows["close"])]
    df = pd.DataFrame(rows)
    df.to_csv(directory / f"{stem}.csv", index=False)


def test_load_qlib_dump_produces_canonical_loader_compatible_frame(tmp_path):
    """Three synthetic per-instrument CSVs load into the canonical schema."""
    _write_instrument_csv(tmp_path, "AAA")
    _write_instrument_csv(tmp_path, "BBB")
    _write_instrument_csv(tmp_path, "CCC", with_amount=True)

    df = load_qlib_dump(tmp_path)

    # Canonical columns present with the exact loader dtypes.
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
    assert pd.api.types.is_string_dtype(df["asset_id"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        assert pd.api.types.is_numeric_dtype(df[col])

    # Asset ids derived from filename stems.
    assert set(df["asset_id"].unique()) == {"AAA", "BBB", "CCC"}
    assert len(df) == 15  # 3 instruments x 5 rows

    # Sorted by (datetime, asset_id), matching load_market_data.
    assert list(df["datetime"]) == sorted(df["datetime"])

    # amount approximated as close * volume for the two files with no
    # turnover column; the file with an explicit turnover column keeps it.
    aaa = df[df["asset_id"] == "AAA"].sort_values("datetime")
    expected_amount = (aaa["close"] * aaa["volume"]).to_numpy()
    assert aaa["amount"].to_numpy() == pytest.approx(expected_amount)

    ccc = df[df["asset_id"] == "CCC"].sort_values("datetime")
    expected_ccc_amount = (ccc["close"] * ccc["volume"]).to_numpy()
    assert ccc["amount"].to_numpy() == pytest.approx(expected_ccc_amount)


def test_load_qlib_dump_filters_by_instruments(tmp_path):
    _write_instrument_csv(tmp_path, "AAA")
    _write_instrument_csv(tmp_path, "BBB")

    df = load_qlib_dump(tmp_path, instruments=["aaa"])  # case-insensitive

    assert set(df["asset_id"].unique()) == {"AAA"}


def test_load_qlib_dump_missing_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_qlib_dump(tmp_path / "does_not_exist")


def test_load_qlib_dump_no_matching_files_raises(tmp_path):
    _write_instrument_csv(tmp_path, "AAA")
    with pytest.raises(FileNotFoundError):
        load_qlib_dump(tmp_path, instruments=["ZZZ"])


# ---------------------------------------------------------------------------
# export_formulas_qlib
# ---------------------------------------------------------------------------

def _library_with_formulas(formulas: dict[str, str]) -> FactorLibrary:
    library = FactorLibrary()
    for i, (name, formula) in enumerate(formulas.items()):
        factor = Factor(
            id=i,
            name=name,
            formula=formula,
            category="test",
            ic_mean=0.05,
            icir=0.5,
            ic_win_rate=0.55,
            max_correlation=0.1,
            batch_number=1,
        )
        library.factors[factor.id] = factor
    return library


def test_export_formulas_qlib_translates_mapped_operators(tmp_path):
    """A formula using only mapped operators translates cleanly to Qlib syntax."""
    library = _library_with_formulas({
        "mapped": "Mean($close, 10)",
        "comparison": "Greater($close, $open)",
    })
    out_path = tmp_path / "library_qlib.json"

    export_formulas_qlib(library, out_path)

    payload = json.loads(out_path.read_text())
    entries = {e["name"]: e for e in payload["factors"]}

    mapped = entries["mapped"]
    assert mapped["qlib_translatable"] is True
    assert mapped["unsupported_operators"] == []
    assert mapped["qlib_expression"] == "Mean($close, 10)"

    # FactorMiner's comparison "Greater" must map to Qlib's boolean "Gt",
    # NOT Qlib's own "Greater" (which means element-wise max in Qlib).
    comparison = entries["comparison"]
    assert comparison["qlib_translatable"] is True
    assert comparison["qlib_expression"] == "Gt($close, $open)"


def test_export_formulas_qlib_flags_unmapped_operator(tmp_path):
    """A formula using an operator with no verified Qlib equivalent is flagged."""
    library = _library_with_formulas({
        "unmapped": "CsRank($close)",
    })
    out_path = tmp_path / "library_qlib.json"

    export_formulas_qlib(library, out_path)

    payload = json.loads(out_path.read_text())
    entry = payload["factors"][0]

    assert entry["qlib_translatable"] is False
    assert "CsRank" in entry["unsupported_operators"]
    # Never silently wrong: the entry is still emitted with best-effort output.
    assert entry["qlib_expression"] is not None


def test_export_formulas_qlib_flags_semantically_divergent_operators(tmp_path):
    """Regression test: operators whose semantics diverge from their
    Qlib namesake must be flagged unsupported, never silently
    mistranslated as if verified equivalent.

    Prior entries claimed 1:1 equivalence for these despite real
    divergence: TsRank (different percentile convention than Qlib's
    Rank), TsArgMax/TsArgMin (0-based vs Qlib's 1-based IdxMax/IdxMin),
    Log (FactorMiner's sign-preserving log1p vs Qlib's raw np.log),
    And/Or/Not (logical 0/1 ops vs Qlib's bitwise ops), and IfElse
    (FactorMiner's NaN-safe positive-only truthiness vs Qlib's raw
    np.where truthiness, which treats NaN/negative as true).
    """
    library = _library_with_formulas({
        "ts_rank": "TsRank($close, 20)",
        "ts_argmax": "TsArgMax($close, 10)",
        "ts_argmin": "TsArgMin($close, 10)",
        "log": "Log($close)",
        "logical_not": "Not(Greater($close, $open))",
        "logical_and": "And(Greater($close, $open), Greater($high, $low))",
        "logical_or": "Or(Greater($close, $open), Greater($high, $low))",
        "if_else": "IfElse(Greater($close, $open), $close, $open)",
    })
    out_path = tmp_path / "library_qlib.json"

    export_formulas_qlib(library, out_path)

    payload = json.loads(out_path.read_text())
    entries = {e["name"]: e for e in payload["factors"]}

    expected_unsupported_op = {
        "ts_rank": "TsRank",
        "ts_argmax": "TsArgMax",
        "ts_argmin": "TsArgMin",
        "log": "Log",
        "logical_not": "Not",
        "logical_and": "And",
        "logical_or": "Or",
        "if_else": "IfElse",
    }
    for name, op in expected_unsupported_op.items():
        entry = entries[name]
        assert entry["qlib_translatable"] is False, (
            f"{name} ({op}) must not be claimed Qlib-translatable"
        )
        assert op in entry["unsupported_operators"]
        # Still emitted (never silently wrong), just under its FM name.
        assert op in entry["qlib_expression"]


def test_export_formulas_qlib_mixed_formula_flags_only_unmapped_part(tmp_path):
    library = _library_with_formulas({
        "mixed": "Sub(CsRank($close), Mean($close, 5))",
    })
    out_path = tmp_path / "library_qlib.json"

    export_formulas_qlib(library, out_path)

    entry = json.loads(out_path.read_text())["factors"][0]
    assert entry["qlib_translatable"] is False
    assert entry["unsupported_operators"] == ["CsRank"]
    assert "Mean($close, 5)" in entry["qlib_expression"]


# ---------------------------------------------------------------------------
# export_anonymized
# ---------------------------------------------------------------------------

def test_export_anonymized_never_contains_raw_formula(tmp_path):
    library = _library_with_formulas({
        "f0": "Neg(CsRank(Div(Sub($close, $vwap), $vwap)))",
        "f1": "Mean($close, 10)",
    })
    out_path = tmp_path / "library_anon.csv"

    export_anonymized(library, out_path, fmt="csv")

    raw_text = out_path.read_text()
    assert "$close" not in raw_text
    assert "$vwap" not in raw_text
    assert "Neg(" not in raw_text
    assert "CsRank(" not in raw_text

    with open(out_path, newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 2
    for row in rows:
        assert "formula" not in row
        assert "formula_hash" in row
        assert len(row["formula_hash"]) == 16
        assert "family" in row


def test_export_anonymized_identical_formulas_hash_identically(tmp_path):
    library = _library_with_formulas({
        "f0": "Mean($close, 10)",
        "f1": "Mean($close, 10)",
        "f2": "Std($close, 10)",
    })
    out_path = tmp_path / "library_anon.csv"

    export_anonymized(library, out_path, fmt="csv")

    with open(out_path, newline="") as fp:
        rows = {row["name"]: row for row in csv.DictReader(fp)}

    assert rows["f0"]["formula_hash"] == rows["f1"]["formula_hash"]
    assert rows["f0"]["formula_hash"] != rows["f2"]["formula_hash"]


def test_export_anonymized_json_format(tmp_path):
    library = _library_with_formulas({"f0": "Std($close, 10)"})
    out_path = tmp_path / "library_anon.json"

    export_anonymized(library, out_path, fmt="json")

    payload = json.loads(out_path.read_text())
    assert "Std(" not in json.dumps(payload)
    assert payload[0]["family"] == "Volatility"


def test_export_anonymized_rejects_unknown_format(tmp_path):
    library = _library_with_formulas({"f0": "Mean($close, 10)"})
    with pytest.raises(ValueError):
        export_anonymized(library, tmp_path / "out.xml", fmt="xml")
