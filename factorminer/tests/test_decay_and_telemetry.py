"""Tests for factor decay/half-life analysis and lifecycle telemetry.

Covers:
- ``factorminer.evaluation.decay.compute_factor_decay_curve`` half-life
  estimation and classification thresholds.
- ``factorminer.evaluation.decay.build_decay_report`` extraction from a
  ``FactorLifecycleStore``.
- ``factorminer.architecture.lifecycle.FactorLifecycleStore.telemetry_summary``
  per-status/per-iteration aggregation.
- ``factorminer.architecture.lifecycle.FactorLifecycleStore.load`` JSONL
  round-tripping.
- The text section renderers in ``factorminer.utils.reporting``.
"""

from __future__ import annotations

import pytest

from factorminer.architecture.lifecycle import FactorLifecycleStore
from factorminer.evaluation.decay import (
    DecayCurveResult,
    build_decay_report,
    compute_factor_decay_curve,
)
from factorminer.utils.reporting import render_decay_report, render_telemetry_report

# ---------------------------------------------------------------------------
# compute_factor_decay_curve
# ---------------------------------------------------------------------------

def test_decay_curve_linear_halving_series_matches_half_life():
    """A series that linearly halves after N iterations should report a
    half_life_iterations within 1 of N, classified as 'decaying'."""
    N = 10
    ic0 = 0.10
    series = [ic0 - (ic0 / 2.0) / N * t for t in range(15)]

    result = compute_factor_decay_curve(series, admission_iteration=0, factor_id="F1")

    assert isinstance(result, DecayCurveResult)
    assert result.half_life_iterations is not None
    assert abs(result.half_life_iterations - N) <= 1.0
    assert result.classification == "decaying"
    assert result.trend_slope < 0


def test_decay_curve_flat_series_is_stable():
    """A flat IC series has no decay and should be classified 'stable' with
    no half-life."""
    series = [0.05] * 12
    result = compute_factor_decay_curve(series, admission_iteration=0, factor_id="F2")

    assert result.classification == "stable"
    assert result.half_life_iterations is None
    assert result.trend_slope == pytest.approx(0.0, abs=1e-9)


def test_decay_curve_growing_series_is_stable():
    """A growing IC series (the factor is getting *better*) should also be
    classified 'stable', with no half-life."""
    series = [0.02 + 0.01 * t for t in range(10)]
    result = compute_factor_decay_curve(series, admission_iteration=0, factor_id="F3")

    assert result.classification == "stable"
    assert result.half_life_iterations is None
    assert result.trend_slope > 0


def test_decay_curve_decayed_series_hits_zero():
    """A series that decays past zero should be classified 'decayed'."""
    series = [0.08 - 0.008 * t for t in range(12)]
    result = compute_factor_decay_curve(series, admission_iteration=0, factor_id="F4")

    assert result.classification == "decayed"
    assert result.ic_current <= 0.02


def test_decay_curve_short_series_is_insufficient_data():
    """Too few observations should not produce a spurious trend."""
    series = [0.05, 0.04]
    result = compute_factor_decay_curve(series, admission_iteration=0, factor_id="F5")

    assert result.classification == "insufficient_data"
    assert result.half_life_iterations is None


def test_decay_curve_handles_nan_observations():
    """NaN entries (missed iterations) should be skipped, not crash the fit."""
    ic0 = 0.10
    N = 10
    series = [ic0 - (ic0 / 2.0) / N * t for t in range(15)]
    series[3] = float("nan")
    series[7] = float("nan")

    result = compute_factor_decay_curve(series, admission_iteration=0, factor_id="F6")

    assert result.classification == "decaying"
    assert result.half_life_iterations is not None
    assert abs(result.half_life_iterations - N) <= 2.0


# ---------------------------------------------------------------------------
# build_decay_report
# ---------------------------------------------------------------------------

def test_build_decay_report_extracts_per_factor_curves():
    store = FactorLifecycleStore(output_dir=None)
    ic0 = 0.09
    N = 6
    for t in range(10):
        ic = ic0 - (ic0 / 2.0) / N * t
        store.record(
            t,
            "decaying_factor",
            "f(decaying_factor)",
            stage="fast_screened",
            status="passed",
            details={"ic_mean": ic},
        )
    store.record(
        0,
        "one_shot_factor",
        "f(one_shot_factor)",
        stage="admitted",
        status="passed",
        details={"ic_mean": 0.02, "icir": 0.5, "replaced": None},
    )

    report = build_decay_report(store)
    by_id = {row["factor_id"]: row for row in report}

    assert "decaying_factor" in by_id
    assert by_id["decaying_factor"]["classification"] == "decaying"
    assert abs(by_id["decaying_factor"]["half_life_iterations"] - N) <= 1.0

    assert "one_shot_factor" in by_id
    assert by_id["one_shot_factor"]["classification"] == "insufficient_data"
    assert by_id["one_shot_factor"]["observations"] == 1


# ---------------------------------------------------------------------------
# FactorLifecycleStore.telemetry_summary
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_store() -> FactorLifecycleStore:
    """A store with a deliberate mix of statuses across two iterations:

    Iteration 0 (3 candidates):
      - A: parse error
      - B: parses, fast-screens, then rejected as an intra-batch duplicate
      - C: parses, fast-screens, admitted
    Iteration 1 (2 candidates):
      - D: parses but never reaches fast-screen (implicit IC-screen reject)
      - C: re-admitted (replaces itself, for count purposes)
    """
    store = FactorLifecycleStore(output_dir=None)

    store.record(0, "A", "f(A)", stage="proposed", status="seen")
    store.record(
        0, "A", "f(A)", stage="parsed", status="failed",
        details={"reason": "Parse failure"},
    )

    store.record(0, "B", "f(B)", stage="proposed", status="seen")
    store.record(0, "B", "f(B)", stage="parsed", status="passed")
    store.record(
        0, "B", "f(B)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.05},
    )
    store.record(
        0, "B", "f(B)", stage="correlation_rejected", status="failed",
        details={"reason": "Intra-batch deduplication (correlated with higher-IC batch member)"},
    )

    store.record(0, "C", "f(C)", stage="proposed", status="seen")
    store.record(0, "C", "f(C)", stage="parsed", status="passed")
    store.record(
        0, "C", "f(C)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.08},
    )
    store.record(
        0, "C", "f(C)", stage="admitted", status="passed",
        details={"ic_mean": 0.08, "icir": 1.2, "replaced": None},
    )

    store.record(1, "D", "f(D)", stage="proposed", status="seen")
    store.record(1, "D", "f(D)", stage="parsed", status="passed")

    store.record(1, "C", "f(C)", stage="proposed", status="seen")
    store.record(1, "C", "f(C)", stage="parsed", status="passed")
    store.record(
        1, "C", "f(C)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.06},
    )
    store.record(
        1, "C", "f(C)", stage="admitted", status="passed",
        details={"ic_mean": 0.06, "icir": 1.0, "replaced": None},
    )
    return store


def test_telemetry_summary_per_status_counts(populated_store):
    summary = populated_store.telemetry_summary()

    assert summary["iterations"] == [0, 1]
    assert summary["total_candidates"] == 5
    assert summary["stage_status_totals"]["proposed:seen"] == 5
    assert summary["stage_status_totals"]["parsed:failed"] == 1
    assert summary["stage_status_totals"]["admitted:passed"] == 2

    it0 = next(row for row in summary["per_iteration"] if row["iteration"] == 0)
    assert it0["candidates_seen"] == 3
    assert it0["parse_errors"] == 1
    assert it0["duplicate_rejected"] == 1
    assert it0["correlation_rejected"] == 0
    assert it0["admitted"] == 1
    assert it0["total_rejected"] == 2
    assert it0["rejection_rate"] == pytest.approx(2 / 3)

    it1 = next(row for row in summary["per_iteration"] if row["iteration"] == 1)
    assert it1["candidates_seen"] == 2
    assert it1["ic_screen_rejected"] == 1
    assert it1["admitted"] == 1
    assert it1["total_rejected"] == 1
    assert it1["rejection_rate"] == pytest.approx(0.5)

    assert summary["rejection_reason_totals"]["duplicate"] == 1
    assert summary["rejection_reason_totals"]["parse_error"] == 1
    assert summary["rejection_reason_totals"]["ic_below_threshold"] == 1
    assert summary["total_rejected"] == 3
    assert summary["overall_rejection_rate"] == pytest.approx(3 / 5)


def test_telemetry_summary_rejection_rate_trend(populated_store):
    summary = populated_store.telemetry_summary()
    trend = summary["rejection_rate_trend"]

    assert [row["iteration"] for row in trend] == [0, 1]
    assert trend[0]["rejection_rate"] == pytest.approx(2 / 3)
    assert trend[1]["rejection_rate"] == pytest.approx(0.5)


def test_telemetry_summary_empty_store_is_zeroed():
    store = FactorLifecycleStore(output_dir=None)
    summary = store.telemetry_summary()

    assert summary["iterations"] == []
    assert summary["per_iteration"] == []
    assert summary["total_candidates"] == 0
    assert summary["overall_rejection_rate"] == 0.0


# ---------------------------------------------------------------------------
# FactorLifecycleStore.load
# ---------------------------------------------------------------------------

def test_lifecycle_store_load_round_trips_jsonl(tmp_path):
    store = FactorLifecycleStore(output_dir=tmp_path)
    store.record(0, "X", "f(X)", stage="proposed", status="seen")
    store.record(
        0, "X", "f(X)", stage="parsed", status="passed", details={"note": "ok"}
    )

    reloaded_dir = FactorLifecycleStore.load(tmp_path)
    assert len(reloaded_dir.events) == 2
    assert reloaded_dir.events[0].factor_name == "X"
    assert reloaded_dir.events[1].details == {"note": "ok"}

    reloaded_file = FactorLifecycleStore.load(tmp_path / "factor_lifecycle.jsonl")
    assert len(reloaded_file.events) == 2


def test_lifecycle_store_load_missing_path_is_empty(tmp_path):
    store = FactorLifecycleStore.load(tmp_path / "does_not_exist")
    assert store.events == []


# ---------------------------------------------------------------------------
# utils.reporting section renderers
# ---------------------------------------------------------------------------

def test_render_decay_report_lists_factors_and_breakdown():
    results = [
        compute_factor_decay_curve(
            [0.10 - 0.005 * t for t in range(15)], 0, factor_id="AlphaDecay"
        ),
        compute_factor_decay_curve([0.05] * 10, 0, factor_id="StableFactor"),
    ]
    text = render_decay_report(results)

    assert "FACTOR DECAY REPORT" in text
    assert "AlphaDecay" in text
    assert "StableFactor" in text
    assert "decaying" in text
    assert "stable" in text


def test_render_decay_report_handles_empty_input():
    text = render_decay_report([])
    assert "No decay observations recorded." in text


def test_render_telemetry_report_shows_breakdown(populated_store):
    summary = populated_store.telemetry_summary()
    text = render_telemetry_report(summary)

    assert "MINING TELEMETRY REPORT" in text
    assert "Total candidates" in text
    assert "duplicate" in text
    assert "parse_error" in text
    assert "Per-Iteration Breakdown" in text
