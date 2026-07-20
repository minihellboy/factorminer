"""Tests for continuous-contract futures lane and capacity extension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorminer.agent.prompt_builder import build_system_prompt
from factorminer.core.parser import parse
from factorminer.core.types import get_features, reset_features
from factorminer.data.futures_source import (
    FUTURES_FEATURE_LEAVES,
    FuturesConfig,
    backward_adjust_continuous,
    build_continuous_futures_panel,
    compute_basis_leaves,
    generate_mock_futures_panel,
    notional_volume,
    register_futures_features,
)
from factorminer.data.tensor_builder import TensorConfig, build_tensor
from factorminer.evaluation.capacity import CapacityConfig, MarketImpactModel


@pytest.fixture(autouse=True)
def _reset_features():
    reset_features()
    yield
    reset_features()


def test_register_futures_features():
    register_futures_features()
    feats = get_features()
    for leaf in FUTURES_FEATURE_LEAVES:
        assert leaf in feats


def test_generate_mock_futures_panel_has_basis_leaves():
    panel = generate_mock_futures_panel(num_assets=4, num_periods=60, seed=1)
    for col in ("basis", "spot", "premium", "roll_yield", "oi", "close", "volume"):
        assert col in panel.columns
    assert panel["asset_id"].nunique() == 4
    assert len(panel) == 4 * 60
    # basis ~= close - spot
    np.testing.assert_allclose(
        panel["basis"].to_numpy(),
        (panel["close"] - panel["spot"]).to_numpy(),
        rtol=1e-10,
        atol=1e-8,
    )
    assert panel["oi"].gt(0).all()


def test_compute_basis_leaves_premium_definition():
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "asset_id": ["ES", "ES"],
            "close": [5100.0, 5120.0],
            "spot": [5000.0, 5050.0],
            "oi": [1000.0, 1100.0],
        }
    )
    out = compute_basis_leaves(df)
    assert out.loc[0, "basis"] == pytest.approx(100.0)
    assert out.loc[0, "premium"] == pytest.approx(5100.0 / 5000.0 - 1.0)


def test_backward_adjust_removes_roll_gap():
    # Synthetic: price jumps +10 at roll solely due to contract switch; adj factor
    # encodes the ratio so backward adjust flattens the gap.
    df = pd.DataFrame(
        {
            "datetime": pd.bdate_range("2024-01-02", periods=5),
            "asset_id": ["CL"] * 5,
            "open": [70.0, 71.0, 82.0, 83.0, 84.0],
            "high": [71.0, 72.0, 83.0, 84.0, 85.0],
            "low": [69.0, 70.0, 81.0, 82.0, 83.0],
            "close": [70.5, 71.5, 82.5, 83.5, 84.5],
            "volume": [1000.0] * 5,
            "amount": [70500.0] * 5,
            "adjustment_factor": [1.0, 1.0, 1.0 + 11.0 / 71.5, 1.0 + 11.0 / 71.5, 1.0 + 11.0 / 71.5],
        }
    )
    adj = backward_adjust_continuous(df)
    # After backward adjust, last close equals original last close
    assert adj.iloc[-1]["close"] == pytest.approx(df.iloc[-1]["close"])
    # Pre-roll closes are scaled
    assert adj.iloc[0]["close"] != pytest.approx(df.iloc[0]["close"])


def test_build_continuous_panel_registers_and_derives():
    raw = pd.DataFrame(
        {
            "datetime": list(pd.bdate_range("2024-01-02", periods=3)) * 2,
            "asset_id": ["F1"] * 3 + ["F2"] * 3,
            "open": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            "high": [101.0, 102.0, 103.0, 201.0, 202.0, 203.0],
            "low": [99.0, 100.0, 101.0, 199.0, 200.0, 201.0],
            "close": [100.5, 101.5, 102.5, 200.5, 201.5, 202.5],
            "volume": [1000.0] * 6,
            "spot": [99.0, 100.0, 101.0, 198.0, 199.0, 200.0],
            "oi": [5000.0] * 6,
            "adjustment_factor": [1.0] * 6,
        }
    )
    panel = build_continuous_futures_panel(raw, config=FuturesConfig())
    assert "$basis" in get_features()
    assert "basis" in panel.columns
    assert "premium" in panel.columns
    assert "amount" in panel.columns


def test_tensor_builder_with_futures_extras():
    panel = generate_mock_futures_panel(num_assets=3, num_periods=40, seed=2)
    # Ensure derived columns present for DEFAULT_FEATURES
    cfg = TensorConfig(
        extra_features=["basis", "spot", "premium", "roll_yield", "oi"],
    )
    ds = build_tensor(panel, cfg)
    for name in ("basis", "spot", "premium", "oi"):
        assert name in ds.feature_names
    register_futures_features()
    tree = parse("CsRank(Div($basis, $spot))")
    data = {
        f"${n}" if not n.startswith("$") else n: np.asarray(ds.data)[:, :, i]
        for i, n in enumerate(ds.feature_names)
    }
    # normalize keys to DSL
    data_dict = {}
    for k, v in data.items():
        key = k if k.startswith("$") else f"${k}"
        if key == "$amount":
            key = "$amt"
        data_dict[key] = v
    signals = tree.evaluate(data_dict)
    assert signals.shape[0] == 3
    assert np.isfinite(signals).any()


def test_prompt_framing_for_futures_asset_class():
    register_futures_features()
    prompt = build_system_prompt(asset_class="futures", features=get_features())
    assert "futures" in prompt.lower()
    assert "$basis" in prompt
    assert "$oi" in prompt
    assert "stock selection" not in prompt.lower()


def test_capacity_contracts_volume_mode():
    rng = np.random.default_rng(0)
    M, T = 20, 50
    signals = rng.normal(size=(M, T))
    contract_volume = rng.integers(100, 1000, size=(M, T)).astype(np.float64)
    prices = rng.uniform(50, 150, size=(M, T))

    equity_cfg = CapacityConfig(volume_mode="dollar")
    fut_cfg = CapacityConfig(volume_mode="contracts", contract_multiplier=50.0)

    eq_model = MarketImpactModel(equity_cfg)
    fut_model = MarketImpactModel(fut_cfg)

    # Dollar notional = price * contracts * multiplier
    dollar_vol = notional_volume(prices, contract_volume, multiplier=50.0)
    eq_imp = eq_model.estimate_impact(signals, dollar_vol, capital=1e8)

    # Futures path with pre-multiplied price*contracts and multiplier on config
    # should be in the same ballpark as equity dollar path when volumes match.
    fut_imp2 = fut_model.estimate_impact(signals, contract_volume * prices, capital=1e8)
    assert fut_imp2.avg_impact_bps >= 0
    assert eq_imp.avg_impact_bps >= 0
    # Higher capital => higher impact still holds under contracts mode
    low = fut_model.estimate_impact(signals, contract_volume * prices, capital=1e6)
    high = fut_model.estimate_impact(signals, contract_volume * prices, capital=5e9)
    assert high.avg_impact_bps >= low.avg_impact_bps


def test_capacity_rejects_bad_volume_mode():
    with pytest.raises(ValueError, match="volume_mode"):
        MarketImpactModel(CapacityConfig(volume_mode="shares"))
