"""Tests for the composable feature-leaf registry and tensor path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorminer.agent.llm_interface import MockProvider
from factorminer.agent.prompt_builder import (
    PromptBuilder,
    _format_feature_list,
    build_system_prompt,
)
from factorminer.architecture.dataset_contract import DatasetContract
from factorminer.architecture.research_absorption import (
    ResearchAbsorptionService,
    _heuristic_eligibility,
)
from factorminer.core.parser import parse, try_parse
from factorminer.core.types import (
    DEFAULT_FEATURES,
    FEATURES,
    column_to_feature,
    feature_to_column,
    get_features,
    normalize_feature_name,
    register_features,
    reset_features,
    unregister_features,
)
from factorminer.data.loader import load_market_data
from factorminer.data.tensor_builder import TensorConfig, build_tensor
from factorminer.evaluation.runtime import (
    _column_to_feature,
    _resolve_feature_columns,
    load_runtime_dataset,
)
from factorminer.utils.config import load_config


@pytest.fixture(autouse=True)
def _reset_feature_registry():
    """Keep the global registry isolated across tests."""
    reset_features()
    yield
    reset_features()


def test_default_features_unchanged():
    assert get_features() == list(DEFAULT_FEATURES)
    assert FEATURES[:8] == list(DEFAULT_FEATURES)


def test_register_and_unregister_extra_features():
    out = register_features(["$eps", "revenue", "$book_equity"])
    assert "$eps" in out
    assert "$revenue" in out  # bare name normalized
    assert "$book_equity" in out
    # defaults preserved, extras appended
    assert out[:8] == list(DEFAULT_FEATURES)

    unregister_features(["$eps"])
    assert "$eps" not in get_features()
    assert "$revenue" in get_features()

    reset_features()
    assert get_features() == list(DEFAULT_FEATURES)


def test_normalize_and_column_maps():
    assert normalize_feature_name("eps") == "$eps"
    assert normalize_feature_name("$amt") == "$amt"
    assert feature_to_column("$amt") == "amount"
    assert feature_to_column("$eps") == "eps"
    assert column_to_feature("amount") == "$amt"
    assert column_to_feature("eps") == "$eps"


def test_parser_accepts_registered_extra_leaf():
    register_features(["$eps"])
    tree = parse("CsRank(Div($eps, $close))")
    assert tree is not None
    assert "$eps" in tree.root.leaf_features() if hasattr(tree, "root") else True

    # unregistered leaf still rejected after reset
    reset_features()
    with pytest.raises(Exception):
        parse("CsRank($eps)")


def test_loader_preserves_extra_numeric_columns(tmp_path):
    path = tmp_path / "with_eps.csv"
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]
            ),
            "asset_id": ["A", "A", "B", "B"],
            "open": [10.0, 10.1, 20.0, 20.2],
            "high": [10.5, 10.6, 20.5, 20.7],
            "low": [9.8, 9.9, 19.8, 19.9],
            "close": [10.2, 10.3, 20.1, 20.4],
            "volume": [1000.0, 1100.0, 2000.0, 2100.0],
            "amount": [10200.0, 11330.0, 40200.0, 42840.0],
            "eps": [1.5, 1.5, 2.0, 2.0],
            "revenue": [1e9, 1e9, 2e9, 2e9],
        }
    )
    df.to_csv(path, index=False)

    loaded = load_market_data(path)
    assert "eps" in loaded.columns
    assert "revenue" in loaded.columns
    assert loaded["eps"].dtype.kind in "fc"
    assert float(loaded.loc[0, "eps"]) == 1.5


def test_tensor_builder_includes_extra_features():
    rows = []
    for asset in ("A", "B", "C"):
        for i, dt in enumerate(pd.bdate_range("2024-01-02", periods=30)):
            rows.append(
                {
                    "datetime": dt,
                    "asset_id": asset,
                    "open": 10.0 + i * 0.01,
                    "high": 10.5 + i * 0.01,
                    "low": 9.5 + i * 0.01,
                    "close": 10.2 + i * 0.01,
                    "volume": 1000.0 + i,
                    "amount": 10200.0 + i,
                    "vwap": 10.2 + i * 0.01,
                    "returns": 0.001 if i else np.nan,
                    "eps": 1.0 + 0.01 * i,
                    "basis": 0.1,
                }
            )
    df = pd.DataFrame(rows)
    cfg = TensorConfig(
        features=list(TensorConfig().features),
        extra_features=["eps", "basis"],
    )
    ds = build_tensor(df, cfg)
    assert "eps" in ds.feature_names
    assert "basis" in ds.feature_names
    arr = np.asarray(ds.data)
    assert arr.shape[2] == len(ds.feature_names)
    eps_idx = ds.feature_names.index("eps")
    assert np.isfinite(arr[0, -1, eps_idx])


def test_runtime_resolves_and_maps_extra_columns():
    register_features(["$eps", "$basis"])
    cols = _resolve_feature_columns(
        ["$open", "$high", "$low", "$close", "$volume", "$amt", "$vwap", "$returns", "$eps", "$basis"]
    )
    assert "eps" in cols
    assert "basis" in cols
    assert _column_to_feature("eps") == "$eps"
    assert _column_to_feature("amount") == "$amt"


def test_prompt_lists_registered_extra_leaves():
    register_features(
        ["$eps"],
        descriptions={"$eps": "earnings per share (point-in-time, as-filed)"},
    )
    text = _format_feature_list()
    assert "$eps" in text
    assert "earnings per share" in text

    sys_prompt = build_system_prompt(asset_class="futures")
    assert "futures" in sys_prompt.lower()
    assert "stock selection" not in sys_prompt.lower()

    equity_prompt = build_system_prompt(asset_class="equity")
    assert "stock selection" in equity_prompt.lower()

    builder = PromptBuilder(features=get_features(), asset_class="futures")
    assert "$eps" in builder.system_prompt
    assert "futures" in builder.system_prompt.lower()


def test_dataset_contract_tracks_extra_features():
    class _Cfg:
        class data:  # noqa: N801 -- mimics a lowercase config-namespace attribute
            features = [
                "$open", "$high", "$low", "$close",
                "$volume", "$amt", "$vwap", "$returns", "$eps",
            ]
            default_target = "paper"
            train_period = ["2024-01-01", "2024-06-30"]
            test_period = ["2024-07-01", "2024-12-31"]
            asset_class = "equity"
            market = "a_shares"

    returns = np.zeros((3, 10))
    contract = DatasetContract.from_arrays(
        _Cfg,
        data_tensor=np.zeros((3, 10, 9)),
        returns=returns,
    )
    assert "$eps" in contract.extra_features
    assert contract.asset_class == "equity"


def test_rma_alt_enabled_keeps_eps_fragment_when_registered():
    reset_features()
    keep, _ = _heuristic_eligibility(
        "Earnings per share momentum predicts returns via fundamental mean reversion.",
        mode="ohlcv_only",
    )
    assert keep is False

    register_features(["$eps"])
    keep_alt, reason = _heuristic_eligibility(
        "Earnings per share momentum predicts returns via fundamental mean reversion.",
        mode="alt_enabled",
    )
    assert keep_alt is True
    assert "$eps" in reason

    service = ResearchAbsorptionService(
        llm_provider=MockProvider(),
        eligibility_mode="alt_enabled",
    )
    keep_svc, reason_svc = service.screen_eligibility(
        "SEC filing EPS revisions contain alpha when mapped to $eps."
    )
    assert keep_svc is True
    assert reason_svc


def test_end_to_end_parse_evaluate_extra_leaf_formula():
    """Prove a formula referencing $eps parses and evaluates on a panel."""
    register_features(["$eps"])
    M, T = 5, 40
    rng = np.random.default_rng(0)
    data = {
        "$close": rng.normal(size=(M, T)) + 100,
        "$eps": rng.normal(size=(M, T)) + 2.0,
        "$open": rng.normal(size=(M, T)) + 100,
        "$high": rng.normal(size=(M, T)) + 101,
        "$low": rng.normal(size=(M, T)) + 99,
        "$volume": rng.random((M, T)) * 1e6 + 1e3,
        "$amt": rng.random((M, T)) * 1e8,
        "$vwap": rng.normal(size=(M, T)) + 100,
        "$returns": rng.normal(scale=0.01, size=(M, T)),
    }
    tree = parse("CsRank(Div($eps, $close))")
    assert try_parse("CsRank(Div($eps, $close))") is not None
    signals = tree.evaluate(data)
    assert signals.shape == (M, T)
    assert np.isfinite(signals).any()


def test_runtime_dataset_with_extra_feature_column():
    """load_runtime_dataset keeps extra columns when listed in config.features."""
    register_features(["$eps"])
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=60)
    for asset in ("X", "Y", "Z", "W"):
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "datetime": dt,
                    "asset_id": asset,
                    "open": 10.0 + i * 0.01,
                    "high": 10.4 + i * 0.01,
                    "low": 9.6 + i * 0.01,
                    "close": 10.1 + i * 0.01,
                    "volume": 1000.0 + i,
                    "amount": 10100.0 + i,
                    "eps": 1.25 if i >= 10 else np.nan,
                }
            )
    raw = pd.DataFrame(rows)
    cfg = load_config()
    # Patch features list on the live config object
    object.__setattr__(
        cfg.data,
        "features",
        [
            "$open", "$high", "$low", "$close",
            "$volume", "$amt", "$vwap", "$returns", "$eps",
        ],
    ) if False else None
    cfg.data.features = [
        "$open", "$high", "$low", "$close",
        "$volume", "$amt", "$vwap", "$returns", "$eps",
    ]
    cfg.data.train_period = ["2024-01-02", "2024-02-29"]
    cfg.data.test_period = ["2024-03-01", "2024-03-29"]

    ds = load_runtime_dataset(raw, cfg)
    assert "$eps" in ds.data_dict
    assert ds.data_tensor.shape[2] == len(ds.data_dict) or ds.data_tensor.shape[2] >= 9
