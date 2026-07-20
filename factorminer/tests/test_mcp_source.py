"""Tests for MCP-sourced market-data ingestion."""

from __future__ import annotations

import pandas as pd
import pytest

from factorminer.data.mcp_source import (
    CANONICAL_COLUMNS,
    MCPDataSourceConfig,
    _dig,
    _records_to_frame,
    known_mcp_connectors,
    load_mcp_source_config,
)


def _field_mapping() -> dict[str, str]:
    return {
        "datetime": "date",
        "asset_id": "ticker",
        "open": "px_open",
        "high": "px_high",
        "low": "px_low",
        "close": "px_close",
        "volume": "shares",
        "amount": "turnover",
    }


def test_mcp_source_config_requires_full_loader_schema() -> None:
    mapping = _field_mapping()
    mapping.pop("amount")

    with pytest.raises(ValueError, match="amount"):
        MCPDataSourceConfig(
            transport="http",
            url="https://example.com/mcp",
            tool="get_prices",
            field_mapping=mapping,
        )


def test_records_to_frame_maps_and_sorts_rows() -> None:
    config = MCPDataSourceConfig(
        transport="http",
        url="https://example.com/mcp",
        tool="get_prices",
        field_mapping=_field_mapping(),
    )
    records = [
        {
            "date": "2024-01-02",
            "ticker": "MSFT",
            "px_open": "100",
            "px_high": "102",
            "px_low": "99",
            "px_close": "101",
            "shares": "2000",
            "turnover": "202000",
        },
        {
            "date": "2024-01-01",
            "ticker": "AAPL",
            "px_open": "10",
            "px_high": "12",
            "px_low": "9",
            "px_close": "11",
            "shares": "1000",
            "turnover": "11000",
        },
    ]

    frame = _records_to_frame(records, config)

    assert list(frame.columns) == list(CANONICAL_COLUMNS)
    assert frame.loc[0, "asset_id"] == "AAPL"
    assert str(frame.loc[0, "datetime"].date()) == "2024-01-01"
    assert frame.loc[1, "asset_id"] == "MSFT"


def test_load_mcp_source_config_expands_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_MCP_TOKEN", "secret-token")
    config_path = tmp_path / "source.yaml"
    config_path.write_text(
        """
transport: http
url: https://example.com/mcp
headers:
  Authorization: "Bearer ${FAKE_MCP_TOKEN}"
tool: get_prices
field_mapping:
  datetime: date
  asset_id: ticker
  open: px_open
  high: px_high
  low: px_low
  close: px_close
  volume: shares
  amount: turnover
""",
        encoding="utf-8",
    )

    config = load_mcp_source_config(config_path)

    assert config.headers["Authorization"] == "Bearer secret-token"


def test_dig_returns_nested_records() -> None:
    assert _dig({"data": {"prices": [1, 2]}}, "data.prices") == [1, 2]


def test_known_mcp_connectors_include_upstream_fsi_endpoints() -> None:
    connectors = {connector["name"]: connector for connector in known_mcp_connectors()}

    assert "factset" in connectors
    assert "lseg" in connectors
    assert connectors["factset"]["url"] == "https://mcp.factset.com/mcp"


def test_records_to_frame_handles_positional_rows_with_constants_and_derived_amount() -> None:
    config = MCPDataSourceConfig(
        transport="stdio",
        command="npx",
        args=["-y", "@mcpfun/mcp-server-ccxt"],
        tool="get-ohlcv",
        arguments={"exchange": "binance", "symbol": "BTC/USDT", "timeframe": "5m"},
        field_mapping={
            "datetime": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        columns_order=["timestamp", "open", "high", "low", "close", "volume"],
        constant_columns={"asset_id": "BTC/USDT"},
        derive_amount_from_close_volume=True,
        datetime_unit="ms",
    )
    records = [
        [1704067200000, 100.0, 102.0, 99.0, 101.0, 10.0],
        [1704067500000, 101.0, 103.0, 100.0, 102.0, 20.0],
    ]

    frame = _records_to_frame(records, config)

    assert list(frame.columns) == list(CANONICAL_COLUMNS)
    assert (frame["asset_id"] == "BTC/USDT").all()
    assert frame.loc[0, "amount"] == 101.0 * 10.0
    assert frame.loc[0, "datetime"] == pd.Timestamp("2024-01-01 00:00:00")


def test_config_allows_amount_via_derive_flag_without_field_mapping() -> None:
    mapping = _field_mapping()
    mapping.pop("amount")

    config = MCPDataSourceConfig(
        transport="http",
        url="https://example.com/mcp",
        tool="get_prices",
        field_mapping=mapping,
        derive_amount_from_close_volume=True,
    )

    assert "amount" not in config.field_mapping


def test_config_allows_amount_via_constant_columns() -> None:
    mapping = _field_mapping()
    mapping.pop("amount")

    config = MCPDataSourceConfig(
        transport="http",
        url="https://example.com/mcp",
        tool="get_prices",
        field_mapping=mapping,
        constant_columns={"amount": "0"},
    )

    assert config.constant_columns["amount"] == "0"


def test_columns_order_rejects_unmapped_field_mapping_targets() -> None:
    with pytest.raises(ValueError, match="not present in columns_order"):
        MCPDataSourceConfig(
            transport="stdio",
            command="npx",
            tool="get-ohlcv",
            field_mapping={
                "datetime": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "asset_id": "not_in_columns_order",
                "amount": "not_in_columns_order_either",
            },
            columns_order=["timestamp", "open", "high", "low", "close", "volume"],
        )


def test_known_mcp_connectors_include_ccxt() -> None:
    connectors = {connector["name"]: connector for connector in known_mcp_connectors()}

    assert "ccxt" in connectors
    assert "mcp-server-ccxt" in connectors["ccxt"]["url"]
