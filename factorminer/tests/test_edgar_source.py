"""Tests for SEC EDGAR XBRL fundamentals connector (point-in-time)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factorminer.core.types import get_features, reset_features
from factorminer.data.edgar_source import (
    DEFAULT_USER_AGENT,
    EDGAR_FEATURE_LEAVES,
    EdgarClient,
    EdgarConfig,
    _parse_companyfacts_payload,
    attach_edgar_to_panel,
    build_mock_companyfacts,
    facts_to_asof_frame,
    load_edgar_fundamentals,
    register_edgar_features,
    sanitize_cik,
)


@pytest.fixture(autouse=True)
def _reset_features():
    reset_features()
    yield
    reset_features()


def test_default_user_agent_identifies_caller_with_contact():
    """SEC fair-access requires a descriptive UA with contact email."""
    assert "FactorMiner" in DEFAULT_USER_AGENT
    assert "@" in DEFAULT_USER_AGENT
    cfg = EdgarConfig()
    assert cfg.user_agent == DEFAULT_USER_AGENT
    assert cfg.max_requests_per_second <= 10.0


def test_sanitize_cik_rejects_path_traversal():
    assert sanitize_cik(320193) == "0000320193"
    assert sanitize_cik("320193") == "0000320193"
    with pytest.raises(ValueError, match="path-like"):
        sanitize_cik("../etc/passwd")
    with pytest.raises(ValueError, match="path-like"):
        sanitize_cik("123/456")


def test_register_edgar_features_extends_registry():
    register_edgar_features()
    feats = get_features()
    for leaf in EDGAR_FEATURE_LEAVES:
        assert leaf in feats


def test_parse_companyfacts_and_point_in_time_asof():
    """Concrete PIT proof: filed-after-period-end does not leak early."""
    # Fact covers FY2023 (end 2023-12-31) but is *filed* on 2024-02-15.
    payload = build_mock_companyfacts(
        cik="0000320193",
        facts=[
            {
                "concept": "EarningsPerShareDiluted",
                "val": 6.13,
                "end": "2023-12-31",
                "filed": "2024-02-15",
                "form": "10-K",
                "fy": 2023,
                "fp": "FY",
                "unit": "USD/shares",
            },
            {
                "concept": "Revenues",
                "val": 383_285_000_000.0,
                "end": "2023-12-31",
                "filed": "2024-02-15",
                "form": "10-K",
                "fy": 2023,
                "unit": "USD",
            },
            {
                "concept": "StockholdersEquity",
                "val": 62_146_000_000.0,
                "end": "2023-12-31",
                "filed": "2024-02-15",
                "unit": "USD",
            },
            {
                "concept": "CommonStockSharesOutstanding",
                "val": 15_550_000_000.0,
                "end": "2023-12-31",
                "filed": "2024-02-15",
                "unit": "shares",
            },
            # Later quarterly update
            {
                "concept": "EarningsPerShareDiluted",
                "val": 1.53,
                "end": "2024-03-31",
                "filed": "2024-05-02",
                "form": "10-Q",
                "fy": 2024,
                "fp": "Q1",
                "unit": "USD/shares",
            },
        ],
    )

    facts = _parse_companyfacts_payload(payload, EdgarConfig().concepts)
    assert facts["eps"]
    assert facts["eps"][0].value == pytest.approx(6.13)
    assert facts["eps"][0].filed == pd.Timestamp("2024-02-15")

    bar_dates = pd.to_datetime(
        [
            "2023-12-29",  # after period end, BEFORE filed → must be NaN
            "2024-02-14",  # day before filed → NaN
            "2024-02-15",  # filed day → 6.13
            "2024-03-01",  # still 6.13
            "2024-05-01",  # day before Q1 filed → still 6.13
            "2024-05-02",  # Q1 filed → 1.53
            "2024-06-01",
        ]
    )
    asof = facts_to_asof_frame(facts, bar_dates)

    # --- Point-in-time assertions (no look-ahead) ---
    assert np.isnan(asof.loc[pd.Timestamp("2023-12-29"), "eps"])
    assert np.isnan(asof.loc[pd.Timestamp("2024-02-14"), "eps"])
    assert asof.loc[pd.Timestamp("2024-02-15"), "eps"] == pytest.approx(6.13)
    assert asof.loc[pd.Timestamp("2024-03-01"), "eps"] == pytest.approx(6.13)
    assert asof.loc[pd.Timestamp("2024-05-01"), "eps"] == pytest.approx(6.13)
    assert asof.loc[pd.Timestamp("2024-05-02"), "eps"] == pytest.approx(1.53)
    assert asof.loc[pd.Timestamp("2024-06-01"), "eps"] == pytest.approx(1.53)

    # Revenue follows the same filed-date discipline
    assert np.isnan(asof.loc[pd.Timestamp("2024-02-14"), "revenue"])
    assert asof.loc[pd.Timestamp("2024-02-15"), "revenue"] == pytest.approx(383_285_000_000.0)


def test_attach_edgar_to_panel_offline_fixture():
    payload = build_mock_companyfacts(
        facts=[
            {
                "concept": "EarningsPerShareDiluted",
                "val": 2.5,
                "end": "2024-01-31",
                "filed": "2024-02-10",
                "unit": "USD/shares",
            }
        ]
    )
    panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-02-01", "2024-02-10", "2024-02-12"] * 2
            ),
            "asset_id": ["AAPL"] * 3 + ["MSFT"] * 3,
            "open": [100.0] * 6,
            "high": [101.0] * 6,
            "low": [99.0] * 6,
            "close": [100.5] * 6,
            "volume": [1e6] * 6,
            "amount": [1e8] * 6,
        }
    )
    joined = attach_edgar_to_panel(
        panel,
        {"AAPL": "0000320193", "MSFT": "0000789019"},
        config=EdgarConfig(register_leaves=True),
        offline_payloads={"AAPL": payload, "MSFT": payload},
    )
    assert "eps" in joined.columns
    aapl = joined[joined["asset_id"] == "AAPL"].sort_values("datetime")
    assert np.isnan(aapl.iloc[0]["eps"])  # 2024-02-01 before filed
    assert aapl.iloc[1]["eps"] == pytest.approx(2.5)  # filed day
    assert aapl.iloc[2]["eps"] == pytest.approx(2.5)


def test_client_uses_cache_and_offline_payload(tmp_path):
    payload = build_mock_companyfacts(
        facts=[
            {
                "concept": "Revenues",
                "val": 100.0,
                "filed": "2024-01-15",
                "end": "2023-12-31",
            }
        ]
    )
    cfg = EdgarConfig(cache_dir=str(tmp_path / "cache"))
    client = EdgarClient(cfg)
    out = client.fetch_companyfacts("320193", offline_payload=payload)
    assert out["entityName"] == "Mock Co"

    # Write cache manually and ensure fetch hits it without network
    cik10 = sanitize_cik("320193")
    cache_file = Path(cfg.cache_dir) / f"companyfacts_CIK{cik10}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    cached = client.fetch_companyfacts("320193")
    assert cached["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["val"] == 100.0


def test_load_edgar_fundamentals_frame_shape():
    payload = build_mock_companyfacts(
        facts=[
            {
                "concept": "EarningsPerShareDiluted",
                "val": 3.0,
                "filed": "2024-03-01",
                "end": "2023-12-31",
                "unit": "USD/shares",
            }
        ]
    )
    bars = pd.bdate_range("2024-01-01", periods=80)
    frame = load_edgar_fundamentals(
        {"AAA": "0000000001"},
        bars,
        offline_payloads={"AAA": payload},
    )
    assert not frame.empty
    assert set(["datetime", "asset_id"]).issubset(frame.columns)
    assert "eps" in frame.columns
    # Before filed date all NaN
    early = frame[frame["datetime"] < pd.Timestamp("2024-03-01")]
    assert early["eps"].isna().all()
    late = frame[frame["datetime"] >= pd.Timestamp("2024-03-01")]
    assert (late["eps"] == 3.0).all()


def test_malformed_payload_fail_closed():
    facts = _parse_companyfacts_payload({"not_facts": True}, EdgarConfig().concepts)
    assert facts["eps"] == []
    facts2 = _parse_companyfacts_payload(
        {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {
                        "units": {
                            "USD/shares": [
                                {"val": "not-a-number", "filed": "2024-01-01"},
                                {"val": 1.0},  # missing filed
                                {"val": 2.0, "filed": "2024-02-01"},
                            ]
                        }
                    }
                }
            }
        },
        EdgarConfig().concepts,
    )
    assert len(facts2["eps"]) == 1
    assert facts2["eps"][0].value == pytest.approx(2.0)


def test_rate_limiter_config_rejects_above_sec_ceiling():
    with pytest.raises(ValueError, match="10"):
        EdgarConfig(max_requests_per_second=50).validate()
