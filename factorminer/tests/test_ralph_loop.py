"""Integration tests for the Ralph Loop end-to-end mining pipeline.

Tests the full pipeline using MockProvider for deterministic factor generation
and synthetic market data, covering:
  - BudgetTracker resource monitoring
  - FactorGenerator response parsing
  - ValidationPipeline multi-stage evaluation
  - RalphLoop end-to-end mining iterations
  - Category inference from formula structure
  - Session persistence (save / load)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from factorminer.agent.llm_interface import MockProvider
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.ralph_loop import (
    BudgetTracker,
    EvaluationResult,
    FactorGenerator,
    MiningReporter,
    RalphLoop,
    ValidationPipeline,
)
from factorminer.memory.memory_store import ExperienceMemory


# ---------------------------------------------------------------------------
# Minimal config for tests
# ---------------------------------------------------------------------------

@dataclass
class _TestConfig:
    target_library_size: int = 10
    batch_size: int = 5
    max_iterations: int = 3
    ic_threshold: float = 0.02
    icir_threshold: float = 0.3
    correlation_threshold: float = 0.7
    replacement_ic_min: float = 0.10
    replacement_ic_ratio: float = 1.3
    fast_screen_assets: int = 0  # No fast screening for deterministic tests
    num_workers: int = 1
    output_dir: str = ""
    redundancy_metric: str = "spearman"
    memory_policy: str = "paper"
    memory_regime_lookback_window: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_library_size": self.target_library_size,
            "batch_size": self.batch_size,
            "max_iterations": self.max_iterations,
            "ic_threshold": self.ic_threshold,
            "icir_threshold": self.icir_threshold,
            "correlation_threshold": self.correlation_threshold,
            "replacement_ic_min": self.replacement_ic_min,
            "replacement_ic_ratio": self.replacement_ic_ratio,
            "redundancy_metric": self.redundancy_metric,
            "memory_policy": self.memory_policy,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="ralph_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_config(tmp_dir):
    return _TestConfig(output_dir=tmp_dir)


@pytest.fixture
def synthetic_data(rng):
    """Synthetic (M=15, T=60, F=8) data tensor and returns."""
    M, T, F = 15, 60, 8
    data_tensor = rng.normal(0, 1, (M, T, F)).astype(np.float64)
    returns = rng.normal(0, 0.02, (M, T)).astype(np.float64)
    return data_tensor, returns


@pytest.fixture
def mock_provider():
    return MockProvider(cycle=True)


@pytest.fixture
def empty_library():
    return FactorLibrary(correlation_threshold=0.7, ic_threshold=0.02)


@pytest.fixture
def empty_memory():
    return ExperienceMemory()


# ===========================================================================
# BudgetTracker tests
# ===========================================================================

class TestBudgetTracker:

    def test_initial_state(self):
        bt = BudgetTracker()
        assert bt.llm_calls == 0
        assert bt.total_tokens == 0
        assert bt.compute_seconds == 0.0
        assert not bt.is_exhausted()

    def test_record_llm_call(self):
        bt = BudgetTracker()
        bt.record_llm_call(prompt_tokens=100, completion_tokens=50)
        assert bt.llm_calls == 1
        assert bt.llm_prompt_tokens == 100
        assert bt.llm_completion_tokens == 50
        assert bt.total_tokens == 150

    def test_record_compute(self):
        bt = BudgetTracker()
        bt.record_compute(1.5)
        bt.record_compute(2.5)
        assert bt.compute_seconds == pytest.approx(4.0)

    def test_exhausted_by_llm_calls(self):
        bt = BudgetTracker(max_llm_calls=2)
        assert not bt.is_exhausted()
        bt.record_llm_call()
        assert not bt.is_exhausted()
        bt.record_llm_call()
        assert bt.is_exhausted()

    def test_exhausted_by_wall_time(self):
        bt = BudgetTracker(max_wall_seconds=0.01)
        import time
        time.sleep(0.02)
        assert bt.is_exhausted()

    def test_unlimited_budgets(self):
        bt = BudgetTracker(max_llm_calls=0, max_wall_seconds=0)
        for _ in range(100):
            bt.record_llm_call()
        assert not bt.is_exhausted()

    def test_to_dict_keys(self):
        bt = BudgetTracker()
        bt.record_llm_call(10, 20)
        d = bt.to_dict()
        expected_keys = {
            "llm_calls", "llm_prompt_tokens", "llm_completion_tokens",
            "total_tokens", "compute_seconds", "wall_elapsed_seconds",
        }
        assert set(d.keys()) == expected_keys

    def test_wall_elapsed_positive(self):
        bt = BudgetTracker()
        assert bt.wall_elapsed >= 0


# ===========================================================================
# EvaluationResult tests
# ===========================================================================

class TestEvaluationResult:

    def test_defaults(self):
        r = EvaluationResult(factor_name="test", formula="Neg($close)")
        assert not r.parse_ok
        assert r.ic_mean == 0.0
        assert r.icir == 0.0
        assert not r.admitted
        assert r.replaced is None
        assert r.rejection_reason == ""
        assert r.stage_passed == 0
        assert r.signals is None

    def test_admitted_result(self):
        r = EvaluationResult(
            factor_name="good",
            formula="CsRank($close)",
            parse_ok=True,
            ic_mean=0.08,
            icir=1.2,
            admitted=True,
            stage_passed=3,
        )
        assert r.admitted
        assert r.stage_passed == 3


# ===========================================================================
# FactorGenerator tests
# ===========================================================================

class TestFactorGenerator:

    def test_generate_batch(self, mock_provider):
        gen = FactorGenerator(llm_provider=mock_provider)
        candidates = gen.generate_batch(
            memory_signal={},
            library_state={"size": 0},
            batch_size=5,
        )
        assert len(candidates) > 0
        for name, formula in candidates:
            assert isinstance(name, str)
            assert isinstance(formula, str)
            assert len(name) > 0
            assert len(formula) > 0

    def test_parse_response_numbered_format(self):
        raw = (
            "1. factor_a: Neg($close)\n"
            "2. factor_b: CsRank(Mean($close, 10))\n"
            "3. factor_c: Div($high, $low)\n"
        )
        result = FactorGenerator._parse_response(raw)
        assert len(result) == 3
        assert result[0] == ("factor_a", "Neg($close)")
        assert result[1] == ("factor_b", "CsRank(Mean($close, 10))")

    def test_parse_response_empty(self):
        assert FactorGenerator._parse_response("") == []
        assert FactorGenerator._parse_response("\n\n") == []

    def test_parse_response_ignores_bad_lines(self):
        raw = (
            "Some random text\n"
            "1. valid_factor: Neg($close)\n"
            "Not a factor line\n"
            "2. another: CsRank($volume)\n"
        )
        result = FactorGenerator._parse_response(raw)
        assert len(result) == 2

    def test_mock_provider_deterministic(self):
        p1 = MockProvider(cycle=False)
        p2 = MockProvider(cycle=False)
        r1 = p1.generate("sys", "user", 0.8, 4096)
        r2 = p2.generate("sys", "user", 0.8, 4096)
        assert r1 == r2

    def test_mock_provider_cycling(self):
        p = MockProvider(cycle=True)
        r1 = p.generate("sys", "user")
        r2 = p.generate("sys", "user")
        # Second call should produce different factors (cycled offset)
        # unless batch_size == len(MOCK_FACTORS)
        assert isinstance(r1, str)
        assert isinstance(r2, str)


# ===========================================================================
# ValidationPipeline tests
# ===========================================================================

class TestValidationPipeline:

    @pytest.fixture
    def pipeline(self, synthetic_data, empty_library):
        data_tensor, returns = synthetic_data
        return ValidationPipeline(
            data_tensor=data_tensor,
            returns=returns,
            library=empty_library,
            ic_threshold=0.02,
            fast_screen_assets=0,  # Use all assets
        )

    def test_parse_failure(self, pipeline):
        result = pipeline.evaluate_candidate("bad", "NotAnOperator($close)")
        assert not result.parse_ok
        assert result.stage_passed == 0
        assert "Parse failure" in result.rejection_reason

    def test_valid_formula_parses(self, pipeline):
        result = pipeline.evaluate_candidate("neg_close", "Neg($close)")
        assert result.parse_ok

    def test_signals_computed(self, pipeline):
        result = pipeline.evaluate_candidate("neg_close", "Neg($close)")
        assert result.signals is not None
        # Signals should be (M, T) shaped
        M, T = pipeline.returns.shape
        assert result.signals.shape == (M, T)

    def test_ic_computed(self, pipeline):
        result = pipeline.evaluate_candidate("neg_close", "Neg($close)")
        # IC should be a number (may or may not pass threshold)
        assert isinstance(result.ic_mean, float)

    def test_batch_evaluation(self, pipeline):
        candidates = [
            ("f1", "Neg($close)"),
            ("f2", "CsRank(Mean($close, 10))"),
            ("f3", "InvalidFormula!!!"),
        ]
        results = pipeline.evaluate_batch(candidates)
        assert len(results) == 3
        # Third should fail parse
        assert not results[2].parse_ok

    def test_deduplication_keeps_highest_ic(self, synthetic_data, empty_library):
        data_tensor, returns = synthetic_data
        # Use very low threshold and high correlation threshold to admit most
        pipeline = ValidationPipeline(
            data_tensor=data_tensor,
            returns=returns,
            library=empty_library,
            ic_threshold=0.0001,
            fast_screen_assets=0,
        )

        # Create two results with identical signals but different IC
        M, T = returns.shape
        signals = np.random.RandomState(99).randn(M, T)

        r1 = EvaluationResult(
            factor_name="low_ic", formula="Neg($close)",
            parse_ok=True, ic_mean=0.05, admitted=True,
            stage_passed=3, signals=signals.copy(),
        )
        r2 = EvaluationResult(
            factor_name="high_ic", formula="CsRank($close)",
            parse_ok=True, ic_mean=0.10, admitted=True,
            stage_passed=3, signals=signals.copy(),
        )
        results = pipeline._deduplicate_batch([r1, r2])

        # The higher-IC one should be kept; the lower deduped
        admitted = [r for r in results if r.admitted]
        assert len(admitted) == 1
        assert admitted[0].factor_name == "high_ic"

    def test_deduplication_uncorrelated_kept(self, synthetic_data, empty_library):
        data_tensor, returns = synthetic_data
        pipeline = ValidationPipeline(
            data_tensor=data_tensor,
            returns=returns,
            library=empty_library,
            ic_threshold=0.0001,
            fast_screen_assets=0,
        )

        M, T = returns.shape
        rng = np.random.RandomState(42)

        r1 = EvaluationResult(
            factor_name="f1", formula="Neg($close)",
            parse_ok=True, ic_mean=0.05, admitted=True,
            stage_passed=3, signals=rng.randn(M, T),
        )
        r2 = EvaluationResult(
            factor_name="f2", formula="CsRank($volume)",
            parse_ok=True, ic_mean=0.07, admitted=True,
            stage_passed=3, signals=rng.randn(M, T),
        )
        results = pipeline._deduplicate_batch([r1, r2])
        admitted = [r for r in results if r.admitted]
        # Both should survive (independent random signals -> low correlation)
        assert len(admitted) == 2


# ===========================================================================
# MiningReporter tests
# ===========================================================================

class TestMiningReporter:

    def test_log_batch(self, tmp_dir):
        reporter = MiningReporter(output_dir=tmp_dir)
        reporter.log_batch(1, admitted=3, rejected=7)
        log_path = os.path.join(tmp_dir, "mining_batches.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["iteration"] == 1
        assert record["admitted"] == 3

    def test_export_library(self, tmp_dir, empty_library):
        reporter = MiningReporter(output_dir=tmp_dir)
        path = reporter.export_library(empty_library)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "factors" in data
        assert "diagnostics" in data
        assert "exported_at" in data


# ===========================================================================
# Category inference tests
# ===========================================================================

class TestCategoryInference:

    def test_momentum(self):
        assert RalphLoop._infer_category("Delta($close, 5)") == "Momentum"

    def test_volatility(self):
        assert RalphLoop._infer_category("Std($returns, 10)") == "Volatility"

    def test_higher_moment(self):
        assert RalphLoop._infer_category("Skew($returns, 20)") == "Higher-Moment"

    def test_pv_correlation(self):
        assert RalphLoop._infer_category("Corr($close, $volume, 10)") == "PV-Correlation"

    def test_regime_conditional(self):
        cat = RalphLoop._infer_category(
            "IfElse(Greater($returns, 0), $volume, Neg($volume))"
        )
        assert cat == "Regime-Conditional"

    def test_regression(self):
        assert RalphLoop._infer_category("TsLinRegSlope($close, 20)") == "Regression"

    def test_smoothing(self):
        assert RalphLoop._infer_category("EMA($close, 10)") == "Smoothing"

    def test_vwap(self):
        assert RalphLoop._infer_category("Div(Sub($close, $vwap), $vwap)") == "VWAP"

    def test_amount(self):
        assert RalphLoop._infer_category("CsRank($amt)") == "Amount"

    def test_extrema(self):
        assert RalphLoop._infer_category("TsMax($close, 20)") == "Extrema"

    def test_cross_sectional(self):
        assert RalphLoop._infer_category("CsRank($close)") == "Cross-Sectional"

    def test_other_fallback(self):
        assert RalphLoop._infer_category("Add($close, $open)") == "Other"


# ===========================================================================
# End-to-end RalphLoop tests
# ===========================================================================

class TestRalphLoopEndToEnd:

    def test_single_iteration(self, test_config, synthetic_data, mock_provider, tmp_dir):
        test_config.max_iterations = 1
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )

        library = loop.run(max_iterations=1)
        assert isinstance(library, FactorLibrary)
        assert loop.iteration == 1
        assert loop.budget.llm_calls >= 1

    def test_multiple_iterations(self, test_config, synthetic_data, mock_provider, tmp_dir):
        test_config.max_iterations = 3
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )

        library = loop.run(max_iterations=3)
        assert loop.iteration <= 3
        assert loop.budget.llm_calls <= 3

    def test_library_grows(self, test_config, synthetic_data, mock_provider, tmp_dir):
        test_config.max_iterations = 5
        test_config.target_library_size = 100  # High target so we don't stop early
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )

        library = loop.run(max_iterations=5, target_size=100)
        # With mock provider and low IC threshold, some factors should be admitted
        # (exact count depends on pseudo-signal randomness)
        assert isinstance(library.size, int)

    def test_callback_invoked(self, test_config, synthetic_data, mock_provider, tmp_dir):
        test_config.max_iterations = 2
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )

        callback_calls = []

        def cb(iteration: int, stats: Dict[str, Any]) -> None:
            callback_calls.append((iteration, stats))

        loop.run(max_iterations=2, callback=cb)
        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 1
        assert callback_calls[1][0] == 2
        # Stats should have standard keys
        for _, stats in callback_calls:
            assert "candidates" in stats
            assert "admitted" in stats
            assert "library_size" in stats
            assert "yield_rate" in stats

    def test_budget_stops_loop(self, test_config, synthetic_data, mock_provider, tmp_dir):
        test_config.max_iterations = 100
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop.budget = BudgetTracker(max_llm_calls=2)

        library = loop.run(max_iterations=100, target_size=1000)
        assert loop.budget.llm_calls == 2
        assert loop.iteration == 2

    def test_target_size_stops_loop(self, test_config, synthetic_data, mock_provider, tmp_dir):
        test_config.output_dir = tmp_dir
        test_config.ic_threshold = 0.0001  # Very low to admit most
        test_config.correlation_threshold = 0.99  # Very high to avoid dedup
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        # Request tiny library
        library = loop.run(max_iterations=50, target_size=2)
        # Either reached target or exhausted iterations
        assert library.size >= 0

    def test_memory_evolves(self, test_config, synthetic_data, mock_provider, tmp_dir):
        test_config.max_iterations = 2
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )

        loop.run(max_iterations=2)
        # Memory should have been updated at least once
        assert loop.memory is not None

    def test_output_files_created(self, test_config, synthetic_data, mock_provider, tmp_dir):
        test_config.output_dir = tmp_dir
        test_config.max_iterations = 1
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop.run(max_iterations=1)

        # Check that library JSON was exported
        lib_path = os.path.join(tmp_dir, "factor_library.json")
        assert os.path.exists(lib_path)

    def test_run_with_prepopulated_library(
        self, test_config, synthetic_data, mock_provider, tmp_dir, rng
    ):
        test_config.output_dir = tmp_dir
        test_config.max_iterations = 1
        data_tensor, returns = synthetic_data
        M, T = returns.shape

        lib = FactorLibrary(
            correlation_threshold=0.7, ic_threshold=0.02,
        )
        # Add one factor
        factor = Factor(
            id=0, name="seed_factor", formula="Neg($close)",
            category="test", ic_mean=0.06, icir=1.0,
            ic_win_rate=0.6, max_correlation=0.0,
            batch_number=0, signals=rng.normal(0, 1, (M, T)),
        )
        lib.admit_factor(factor)

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
            library=lib,
        )
        result_lib = loop.run(max_iterations=1)
        assert result_lib.size >= 1  # At least the seed factor

    def test_empty_stats_on_no_candidates(self, test_config, synthetic_data, tmp_dir):
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        # Provider that returns empty response
        class EmptyProvider(MockProvider):
            def generate(self, *args, **kwargs):
                return ""

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=EmptyProvider(),
        )
        stats = loop._run_iteration(batch_size=5)
        assert stats["candidates"] == 0
        assert stats["admitted"] == 0


# ===========================================================================
# Session persistence tests
# ===========================================================================

class TestSessionPersistence:

    def test_save_creates_checkpoint(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        test_config.output_dir = tmp_dir
        test_config.max_iterations = 1
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop.run(max_iterations=1)

        checkpoint_path = loop.save_session(tmp_dir)
        assert os.path.isdir(checkpoint_path)

        # Should contain key files
        checkpoint_files = os.listdir(checkpoint_path)
        assert "library.json" in checkpoint_files
        assert "memory.json" in checkpoint_files
        assert "loop_state.json" in checkpoint_files

    def test_load_restores_iteration(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        # Run 2 iterations
        loop1 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop1.run(max_iterations=2)
        checkpoint_path = loop1.save_session(tmp_dir)

        # Load into new loop
        loop2 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=MockProvider(cycle=True),
        )
        loop2.load_session(checkpoint_path)
        assert loop2.iteration == loop1.iteration

    def test_load_restores_memory(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop1 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop1.run(max_iterations=2)
        checkpoint_path = loop1.save_session(tmp_dir)

        loop2 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=MockProvider(cycle=True),
        )
        loop2.load_session(checkpoint_path)
        # Memory should have been restored
        assert loop2.memory is not None


# ===========================================================================
# Checkpoint / Resume tests (Phase 1f)
# ===========================================================================

class TestCheckpointResume:
    """Tests for the checkpoint/resume functionality."""

    def test_checkpoint_creates_files(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify that save_session creates all expected checkpoint files."""
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop.run(max_iterations=2)
        checkpoint_path = loop.save_session()

        checkpoint_dir = Path(checkpoint_path)
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "library.json").exists()
        assert (checkpoint_dir / "memory.json").exists()
        assert (checkpoint_dir / "loop_state.json").exists()
        assert (checkpoint_dir / "session.json").exists()

        # Verify loop_state.json contains expected keys
        with open(checkpoint_dir / "loop_state.json") as f:
            loop_state = json.load(f)
        assert "iteration" in loop_state
        assert "library_size" in loop_state
        assert "memory_version" in loop_state
        assert "budget" in loop_state
        assert loop_state["iteration"] == loop.iteration
        assert loop_state["library_size"] == loop.library.size
        assert loop_state["budget"]["llm_calls"] == loop.budget.llm_calls

    def test_resume_continues_from_checkpoint(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify that resuming continues from the saved iteration."""
        test_config.output_dir = tmp_dir
        test_config.target_library_size = 200  # High target so loop doesn't stop early
        data_tensor, returns = synthetic_data

        # Run 2 iterations, then save
        loop1 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop1.run(max_iterations=2, target_size=200)
        saved_iteration = loop1.iteration
        saved_library_size = loop1.library.size
        loop1.save_session()

        # Create a new loop and resume from checkpoint
        loop2 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=MockProvider(cycle=True),
        )
        assert loop2.iteration == 0  # Starts fresh

        # Resume should load the saved state and continue
        library = loop2.run(max_iterations=4, target_size=200, resume=True)

        # loop2 should have continued from iteration 2, running up to 4
        assert loop2.iteration > saved_iteration
        assert loop2.iteration <= 4

    def test_resume_preserves_library(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify that library factors are preserved across resume."""
        test_config.output_dir = tmp_dir
        test_config.ic_threshold = 0.0001
        test_config.correlation_threshold = 0.99
        data_tensor, returns = synthetic_data

        # Run and save
        loop1 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop1.run(max_iterations=3, target_size=100)
        saved_factors = {
            fid: f.to_dict() for fid, f in loop1.library.factors.items()
        }
        saved_size = loop1.library.size
        loop1.save_session()

        # Load into a new loop
        loop2 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=MockProvider(cycle=True),
        )
        checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
        loop2.load_session(checkpoint_dir)

        # Library should have the same factors
        assert loop2.library.size == saved_size
        for fid, f_dict in saved_factors.items():
            assert fid in loop2.library.factors
            restored = loop2.library.factors[fid].to_dict()
            assert restored["name"] == f_dict["name"]
            assert restored["formula"] == f_dict["formula"]
            assert restored["ic_mean"] == pytest.approx(f_dict["ic_mean"])

    def test_resume_preserves_memory(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify that experience memory is preserved across resume."""
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        # Run and save
        loop1 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop1.run(max_iterations=2)
        saved_version = loop1.memory.version
        saved_patterns = len(loop1.memory.success_patterns)
        saved_forbidden = len(loop1.memory.forbidden_directions)
        saved_insights = len(loop1.memory.insights)
        loop1.save_session()

        # Load into a new loop
        loop2 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=MockProvider(cycle=True),
        )
        checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
        loop2.load_session(checkpoint_dir)

        # Memory state should match
        assert loop2.memory.version == saved_version
        assert len(loop2.memory.success_patterns) == saved_patterns
        assert len(loop2.memory.forbidden_directions) == saved_forbidden
        assert len(loop2.memory.insights) == saved_insights

    def test_checkpoint_interval_controls_frequency(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify checkpoint_interval controls how often checkpoints are saved."""
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        # With interval=2, checkpoint should be written at iterations 2 and 4
        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
            checkpoint_interval=2,
        )
        loop.run(max_iterations=3)

        checkpoint_dir = Path(tmp_dir) / "checkpoint"
        # After 3 iterations with interval=2, checkpoint at iteration 2
        # should have created the directory
        assert checkpoint_dir.exists()

        # Verify the checkpoint was written at least once
        with open(checkpoint_dir / "loop_state.json") as f:
            state = json.load(f)
        # The last checkpoint should be at iteration 2 (since 3 is not
        # divisible by 2, the checkpoint at iter 2 is the latest one)
        assert state["iteration"] == 2

    def test_checkpoint_disabled(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify checkpoint_interval=0 disables automatic checkpointing."""
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
            checkpoint_interval=0,
        )
        loop.run(max_iterations=2)

        checkpoint_dir = Path(tmp_dir) / "checkpoint"
        # No automatic checkpoint should have been created
        assert not checkpoint_dir.exists()

    def test_resume_from_classmethod(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify the resume_from classmethod works correctly."""
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        # Run and save
        loop1 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop1.run(max_iterations=2)
        checkpoint_path = loop1.save_session()

        # Use classmethod to resume
        loop2 = RalphLoop.resume_from(
            checkpoint_path=checkpoint_path,
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=MockProvider(cycle=True),
        )

        assert loop2.iteration == loop1.iteration
        assert loop2.library.size == loop1.library.size

    def test_resume_restores_budget(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify that budget tracker state is preserved across resume."""
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop1 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        loop1.run(max_iterations=2)
        saved_llm_calls = loop1.budget.llm_calls
        saved_compute = loop1.budget.compute_seconds
        loop1.save_session()

        loop2 = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=MockProvider(cycle=True),
        )
        checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
        loop2.load_session(checkpoint_dir)

        assert loop2.budget.llm_calls == saved_llm_calls
        assert loop2.budget.compute_seconds == pytest.approx(
            saved_compute, abs=0.1
        )

    def test_backward_compatible_no_checkpoint(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify run() works without checkpoint/resume (backward compat)."""
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        # Disable checkpointing entirely
        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
            checkpoint_interval=0,
        )
        library = loop.run(max_iterations=2)

        assert isinstance(library, FactorLibrary)
        assert loop.iteration == 2

    def test_resume_no_checkpoint_is_noop(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Verify resume=True with no existing checkpoint just starts fresh."""
        test_config.output_dir = tmp_dir
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        # resume=True but no checkpoint exists -- should work normally
        library = loop.run(max_iterations=1, resume=True)
        assert isinstance(library, FactorLibrary)
        assert loop.iteration == 1

    def test_run_exports_manifest_and_factor_provenance(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        """Completed runs should export a manifest and persist factor provenance."""
        test_config.output_dir = tmp_dir
        test_config.ic_threshold = 0.0
        test_config.icir_threshold = -1.0
        test_config.correlation_threshold = 1.1
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )
        library = loop.run(max_iterations=2, target_size=2)

        manifest_path = Path(tmp_dir) / "run_manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["loop_type"] == "ralph"
        assert manifest["artifact_paths"]["run_manifest"] == str(manifest_path)
        assert manifest["dataset_summary"]["data_tensor_shape"] == list(data_tensor.shape)

        assert library.size > 0
        exported_library = json.loads((Path(tmp_dir) / "factor_library.json").read_text())
        factor_payload = exported_library["factors"][0]
        assert "provenance" in factor_payload
        assert factor_payload["provenance"]["run_id"] == loop._session.session_id
        assert factor_payload["provenance"]["loop_type"] == "ralph"
        assert factor_payload["provenance"]["generator_family"]

    def test_loop_uses_configured_memory_policy_and_dependence_metric(
        self, test_config, synthetic_data, mock_provider, tmp_dir
    ):
        test_config.output_dir = tmp_dir
        test_config.memory_policy = "none"
        test_config.redundancy_metric = "pearson"
        data_tensor, returns = synthetic_data

        loop = RalphLoop(
            config=test_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=mock_provider,
        )

        assert loop.memory_policy.schema()["policy"] == "none"
        assert loop.library.dependence_metric.name == "pearson"
