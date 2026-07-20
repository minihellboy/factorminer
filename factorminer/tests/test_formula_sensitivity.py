"""Tests for formula AST sensitivity / ablation analysis."""

from __future__ import annotations

import numpy as np

from factorminer.evaluation.formula_sensitivity import (
    FormulaSensitivityConfig,
    analyze_formula_sensitivity,
    sensitivity_table,
)
from factorminer.utils.tearsheet import format_sensitivity_panel


def _data(seed: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    m, t = 20, 80
    close = 100 + np.cumsum(rng.normal(0, 0.5, (m, t)), axis=1)
    open_ = close + rng.normal(0, 0.05, (m, t))
    high = np.maximum(close, open_) + 0.2
    low = np.minimum(close, open_) - 0.2
    volume = np.abs(rng.normal(1e6, 2e5, (m, t)))
    vwap = (high + low + close) / 3
    amt = volume * vwap
    returns = np.zeros((m, t))
    returns[:, 1:] = np.diff(close, axis=1) / np.maximum(close[:, :-1], 1e-8)
    # Plant a weak close-based signal so ablations of $close move IC.
    signals_proxy = -(close / np.nanmean(close, axis=0, keepdims=True) - 1.0)
    returns = returns + 0.15 * np.roll(signals_proxy, -1, axis=1)
    data = {
        "$open": open_,
        "$high": high,
        "$low": low,
        "$close": close,
        "$volume": volume,
        "$amt": amt,
        "$vwap": vwap,
        "$returns": returns,
    }
    return data, returns


def test_leaf_and_subtree_ablation_produce_delta_ic() -> None:
    data, returns = _data()
    formula = "Neg(CsRank(Delta($close, 5)))"
    result = analyze_formula_sensitivity(
        formula,
        data,
        returns,
        factor_name="mom_rev",
        config=FormulaSensitivityConfig(include_explanation=True),
    )
    assert result.formula == formula
    assert np.isfinite(result.baseline_ic)
    assert result.leaf_ablations, "expected leaf ablations for $close"
    assert any(row.target == "$close" for row in result.leaf_ablations)
    # At least one leaf ablation should change IC (zero or permute).
    assert any(abs(row.delta_ic) > 1e-8 for row in result.leaf_ablations)
    assert result.subtree_ablations, "expected operator subtree leave-one-out rows"
    assert result.parameter_sensitivity, "expected window/parameter sensitivity rows"
    assert result.explanation
    assert result.explanation_source == "template"

    table = sensitivity_table(result)
    assert table
    assert "delta_ic" in table[0]


def test_sensitivity_panel_text_for_tearsheet() -> None:
    data, returns = _data(seed=1)
    result = analyze_formula_sensitivity(
        "CsZScore(Div(Sub($volume, Mean($volume, 20)), Std($volume, 20)))",
        data,
        returns,
        factor_name="vol_surprise",
    )
    panel = format_sensitivity_panel(result)
    assert "Formula Sensitivity" in panel
    assert "vol_surprise" in panel
    assert "Baseline paper IC" in panel


def test_unparseable_formula_fails_closed() -> None:
    data, returns = _data()
    result = analyze_formula_sensitivity("NotAReal(Formula", data, returns)
    assert not np.isfinite(result.baseline_ic)
    assert "parse" in result.explanation.lower() or "failed" in result.explanation.lower()


def test_parameter_ablation_respects_operator_declared_range() -> None:
    """Regression test: non-window parameters must clamp to their own
    declared range, not a blanket window-length floor.

    A prior bug applied ``max(config.min_window, ...)`` (default 2.0) to
    every numeric parameter regardless of what it represents, crushing
    e.g. ``Quantile.q`` (a probability in [0, 1]) to 2.0 on every
    perturbation -- an invalid, meaningless quantile.
    """
    data, returns = _data(seed=2)
    result = analyze_formula_sensitivity(
        "Quantile($close, 20, 0.5)",
        data,
        returns,
        config=FormulaSensitivityConfig(include_explanation=False),
    )
    q_rows = [row for row in result.parameter_sensitivity if row.target.startswith("q@")]
    assert q_rows, "expected sensitivity rows for the q parameter"
    for row in q_rows:
        perturbed_q = float(row.detail.split("->")[1].split("(")[0].strip())
        assert 0.0 <= perturbed_q <= 1.0, (
            f"q must stay within its declared [0, 1] range, got {perturbed_q}"
        )

    # Power's exponent (declared range [0, 10]) must likewise not be
    # crushed to the window floor.
    result2 = analyze_formula_sensitivity(
        "Power($close, 2)",
        data,
        returns,
        config=FormulaSensitivityConfig(include_explanation=False),
    )
    exp_rows = [
        row for row in result2.parameter_sensitivity if row.target.startswith("exponent@")
    ]
    assert exp_rows
    for row in exp_rows:
        perturbed_exp = float(row.detail.split("->")[1].split("(")[0].strip())
        assert 0.0 <= perturbed_exp <= 10.0

    # A genuine window parameter must still clamp to config.min_window.
    window_rows = [
        row for row in result.parameter_sensitivity if row.target.startswith("window@")
    ]
    assert window_rows
    for row in window_rows:
        perturbed_window = float(row.detail.split("->")[1].split("(")[0].strip())
        assert perturbed_window >= 2.0
