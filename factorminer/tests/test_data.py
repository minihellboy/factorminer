"""Tests for the data pipeline: mock data generation, preprocessing, tensor building."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorminer.data.loader import load_market_data
from factorminer.data.mock_data import MockConfig, generate_mock_data, generate_with_halts

# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------

class TestMockDataGeneration:
    """Test the synthetic market data generator."""

    @pytest.fixture
    def small_config(self):
        return MockConfig(
            num_assets=10,
            num_periods=100,
            frequency="1d",
            seed=42,
        )

    @pytest.fixture
    def small_df(self, small_config):
        return generate_mock_data(small_config)

    def test_returns_dataframe(self, small_df):
        assert isinstance(small_df, pd.DataFrame)

    def test_required_columns(self, small_df):
        for col in ["datetime", "asset_id", "open", "high", "low", "close", "volume", "amount"]:
            assert col in small_df.columns, f"Missing column: {col}"

    def test_correct_shape(self, small_config, small_df):
        expected_rows = small_config.num_assets * small_config.num_periods
        assert len(small_df) == expected_rows

    def test_unique_assets(self, small_config, small_df):
        n_unique = small_df["asset_id"].nunique()
        assert n_unique == small_config.num_assets

    def test_periods_per_asset(self, small_config, small_df):
        counts = small_df.groupby("asset_id").size()
        assert (counts == small_config.num_periods).all()


# ---------------------------------------------------------------------------
# OHLC consistency
# ---------------------------------------------------------------------------

class TestOHLCConsistency:
    """Test that generated data maintains OHLC invariants."""

    @pytest.fixture
    def df(self):
        config = MockConfig(num_assets=20, num_periods=200, seed=123)
        return generate_mock_data(config)

    def test_low_le_high(self, df):
        assert (df["low"] <= df["high"] + 1e-8).all(), "Found low > high"

    def test_open_within_range(self, df):
        assert (df["open"] >= df["low"] - 1e-8).all(), "Found open < low"
        assert (df["open"] <= df["high"] + 1e-8).all(), "Found open > high"

    def test_close_within_range(self, df):
        assert (df["close"] >= df["low"] - 1e-8).all(), "Found close < low"
        assert (df["close"] <= df["high"] + 1e-8).all(), "Found close > high"

    def test_positive_prices(self, df):
        for col in ["open", "high", "low", "close"]:
            assert (df[col] > 0).all(), f"Found non-positive {col}"

    def test_positive_volume(self, df):
        assert (df["volume"] >= 0).all(), "Found negative volume"

    def test_positive_amount(self, df):
        assert (df["amount"] >= 0).all(), "Found negative amount"


# ---------------------------------------------------------------------------
# Trading halts
# ---------------------------------------------------------------------------

class TestHaltGeneration:
    """Test synthetic data with trading halts."""

    def test_generate_with_halts(self):
        config = MockConfig(num_assets=10, num_periods=100, seed=42)
        df = generate_with_halts(config, halt_fraction=0.05)
        assert isinstance(df, pd.DataFrame)
        # Should have some zero-volume bars
        assert (df["volume"] == 0).any()

    def test_halt_bars_have_flat_ohlc(self):
        config = MockConfig(num_assets=10, num_periods=100, seed=42)
        df = generate_with_halts(config, halt_fraction=0.05)
        halted = df[df["volume"] == 0]
        if len(halted) > 0:
            # Open = High = Low = Close for halted bars
            np.testing.assert_array_almost_equal(halted["open"], halted["close"])
            np.testing.assert_array_almost_equal(halted["high"], halted["close"])
            np.testing.assert_array_almost_equal(halted["low"], halted["close"])


# ---------------------------------------------------------------------------
# Different frequencies
# ---------------------------------------------------------------------------

class TestFrequencies:
    """Test data generation at different frequencies."""

    @pytest.mark.parametrize("freq", ["10min", "30min", "1h", "1d"])
    def test_frequency(self, freq):
        config = MockConfig(num_assets=5, num_periods=50, frequency=freq, seed=42)
        df = generate_mock_data(config)
        assert len(df) > 0
        assert "datetime" in df.columns


# ---------------------------------------------------------------------------
# MockConfig defaults
# ---------------------------------------------------------------------------

class TestMockConfig:
    """Test MockConfig defaults and overrides."""

    def test_default_config(self):
        config = MockConfig()
        assert config.num_assets == 50
        assert config.num_periods == 1000
        assert config.frequency == "10min"
        assert config.seed == 42

    def test_config_with_universe(self):
        config = MockConfig(num_assets=5, num_periods=20, universe="CSI300")
        df = generate_mock_data(config)
        assert "universe" in df.columns
        assert (df["universe"] == "CSI300").all()

    def test_config_no_planted_alpha(self):
        config = MockConfig(num_assets=5, num_periods=20, plant_alpha=False)
        df = generate_mock_data(config)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Feature computation (basic checks with preprocessor if available)
# ---------------------------------------------------------------------------

class TestFeatureComputation:
    """Test derived feature computation."""

    def test_vwap_computable(self):
        config = MockConfig(num_assets=5, num_periods=50, seed=42)
        df = generate_mock_data(config)
        # VWAP can be approximated from high, low, close
        vwap = (df["high"] + df["low"] + df["close"]) / 3
        assert len(vwap) == len(df)
        assert (vwap > 0).all()

    def test_returns_computable(self):
        config = MockConfig(num_assets=5, num_periods=50, seed=42)
        df = generate_mock_data(config)
        # Returns per asset
        df = df.sort_values(["asset_id", "datetime"])
        df["returns"] = df.groupby("asset_id")["close"].pct_change()
        # First bar per asset should be NaN
        first_bar_per_asset = df.groupby("asset_id").head(1)
        assert first_bar_per_asset["returns"].isna().all()
        # Rest should be finite
        rest = df.dropna(subset=["returns"])
        assert np.isfinite(rest["returns"]).all()


# ---------------------------------------------------------------------------
# Tensor builder integration
# ---------------------------------------------------------------------------

class TestTensorBuilder:
    """Test tensor construction from mock data (if modules available)."""

    def test_build_pipeline_import(self):
        """Verify we can import the tensor builder."""
        from factorminer.data.tensor_builder import TensorConfig
        config = TensorConfig()
        assert config.backend == "numpy"
        assert "close" in config.features

    def test_temporal_split_import(self):
        """Verify temporal_split is importable."""
        from factorminer.data.tensor_builder import temporal_split
        assert callable(temporal_split)


# ---------------------------------------------------------------------------
# Loader schema compatibility
# ---------------------------------------------------------------------------

class TestLoaderSchemaCompatibility:
    """Test common market-data schema variants accepted by the loader."""

    def test_accepts_common_column_aliases(self, tmp_path):
        path = tmp_path / "alias_data.csv"
        df = pd.DataFrame(
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
        )
        df.to_csv(path, index=False)

        loaded = load_market_data(path)

        assert list(loaded.columns[:8]) == [
            "datetime",
            "asset_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
        assert loaded.loc[0, "asset_id"] == "600519.SH"
        assert loaded.loc[1, "amount"] == 12360.0

    def test_missing_asset_id_still_raises_clear_error(self, tmp_path):
        path = tmp_path / "missing_asset_id.csv"
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2025-01-01 09:30:00"]),
                "open": [10.0],
                "high": [10.3],
                "low": [9.9],
                "close": [10.1],
                "volume": [1000.0],
                "amount": [10100.0],
            }
        )
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing required columns: \\['asset_id'\\]"):
            load_market_data(path)
