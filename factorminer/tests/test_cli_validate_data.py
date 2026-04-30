"""CLI tests for the `validate-data` command."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from factorminer.cli import main


def _write_csv(tmp_path, name: str, df: pd.DataFrame) -> str:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_validate_data_accepts_aliases_and_emits_json(tmp_path):
    """Alias-heavy vendor columns should validate and map to canonical names."""
    path = _write_csv(
        tmp_path,
        "alias_data.csv",
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2025-01-01 09:30:00", "2025-01-01 09:40:00"]
                ),
                "code": ["600519.SH", "600519.SH"],
                "open": [10.0, 10.2],
                "high": [10.3, 10.4],
                "low": [9.9, 10.1],
                "close": [10.1, 10.3],
                "volume": [1000.0, 1200.0],
                "amt": [10100.0, 12360.0],
            }
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--cpu",
            "validate-data",
            path,
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "valid"
    assert payload["valid_schema"] is True
    assert payload["canonical_mapping"]["datetime"] == "timestamp"
    assert payload["canonical_mapping"]["asset_id"] == "code"
    assert payload["canonical_mapping"]["amount"] == "amt"
    assert payload["duplicate_key_count"] == 0
    assert payload["warning_count"] == 0


def test_validate_data_reports_missing_required_columns(tmp_path):
    """Missing schema columns should fail with a nonzero exit code."""
    path = _write_csv(
        tmp_path,
        "invalid_data.csv",
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2025-01-01 09:30:00"]),
                "open": [10.0],
                "high": [10.3],
                "low": [9.9],
                "close": [10.1],
                "volume": [1000.0],
                "amount": [10100.0],
            }
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--cpu",
            "validate-data",
            path,
        ],
    )

    assert result.exit_code == 1, result.output
    assert "Missing required columns" in result.output
    assert "asset_id" in result.output
    assert "Accepted aliases" in result.output
    assert "Next command" in result.output


def test_validate_data_strict_fails_on_warnings(tmp_path):
    """Warnings should become failures under strict mode."""
    path = _write_csv(
        tmp_path,
        "warning_data.csv",
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2025-01-01 09:30:00", "2025-01-01 09:30:00"]
                ),
                "asset_id": ["A", "A"],
                "open": [10.0, 10.1],
                "high": [10.3, 10.4],
                "low": [9.9, 10.0],
                "close": [10.1, 10.2],
                "volume": [1000.0, 1100.0],
                "amount": [10100.0, 11220.0],
                "return_next": [0.1, 0.2],
            }
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--cpu",
            "validate-data",
            path,
            "--strict",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "Status: invalid (strict)" in result.output
    assert "duplicate asset/timestamp" in result.output
    assert "Potential leakage-risk column names detected" in result.output


def test_validate_data_text_includes_next_steps_and_split_warnings(tmp_path):
    path = _write_csv(
        tmp_path,
        "valid_old_data.csv",
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2023-01-01 09:30:00", "2023-01-01 09:40:00"]
                ),
                "asset_id": ["A", "A"],
                "open": [10.0, 10.2],
                "high": [10.3, 10.4],
                "low": [9.9, 10.1],
                "close": [10.1, 10.3],
                "volume": [1000.0, 1200.0],
                "amount": [10100.0, 12360.0],
            }
        ),
    )

    runner = CliRunner()
    result = runner.invoke(main, ["--cpu", "validate-data", path])

    assert result.exit_code == 0, result.output
    assert "Accepted aliases" in result.output
    assert "asset_id: asset_id, ticker, symbol, code" in result.output
    assert "Derived fields" in result.output
    assert "vwap: derived as amount / volume" in result.output
    assert "Configured split coverage" in result.output
    assert "WARN: configured train split is empty" in result.output
    assert "WARN: configured test split is empty" in result.output
    assert f"uv run factorminer -o output mine --data {path}" in result.output


def test_resample_data_converts_5m_to_10min_candles(tmp_path):
    path = _write_csv(
        tmp_path,
        "binance_5m.csv",
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2025-01-01 00:00:00",
                        "2025-01-01 00:05:00",
                        "2025-01-01 00:10:00",
                        "2025-01-01 00:15:00",
                        "2025-01-01 00:00:00",
                        "2025-01-01 00:05:00",
                    ]
                ),
                "asset_id": ["BTC", "BTC", "BTC", "BTC", "ETH", "ETH"],
                "open": [100.0, 101.0, 103.0, 102.0, 10.0, 11.0],
                "high": [102.0, 104.0, 105.0, 103.0, 12.0, 13.0],
                "low": [99.0, 100.0, 101.0, 100.0, 9.0, 10.0],
                "close": [101.0, 103.0, 102.0, 101.0, 11.0, 12.0],
                "volume": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "amount": [101.0, 206.0, 306.0, 404.0, 55.0, 72.0],
            }
        ),
    )
    output = tmp_path / "binance_10m.csv"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--cpu", "resample-data", path, str(output), "--rule", "10min"],
    )

    assert result.exit_code == 0, result.output
    assert "Rows:   6 -> 3" in result.output
    resampled = pd.read_csv(output, parse_dates=["datetime"])
    btc = resampled[resampled["asset_id"] == "BTC"].sort_values("datetime")
    first = btc.iloc[0]
    assert first["datetime"] == pd.Timestamp("2025-01-01 00:00:00")
    assert first["open"] == 100.0
    assert first["high"] == 104.0
    assert first["low"] == 99.0
    assert first["close"] == 103.0
    assert first["volume"] == 3.0
    assert first["amount"] == 307.0
    assert first["vwap"] == pytest.approx(307.0 / 3.0)
