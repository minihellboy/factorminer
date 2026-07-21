"""Evaluation APIs exposed without importing every diagnostic implementation."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_EXPORTS = {
    **{
        name: "factorminer.evaluation.metrics"
        for name in (
            "METRIC_VERSION",
            "compute_ic",
            "compute_ic_vectorized",
            "compute_icir",
            "compute_ic_mean",
            "compute_ic_paper_mean",
            "compute_ic_abs_mean",
            "compute_ic_paper_icir",
            "compute_ic_win_rate",
            "compute_pairwise_correlation",
            "compute_factor_stats",
            "compute_quintile_returns",
            "compute_turnover",
        )
    },
    **{
        name: "factorminer.evaluation.correlation"
        for name in (
            "batch_spearman_correlation",
            "batch_spearman_pairwise",
            "compute_correlation_batch",
            "IncrementalCorrelationMatrix",
        )
    },
    **{
        name: "factorminer.evaluation.admission"
        for name in ("check_admission", "check_replacement", "AdmissionDecision", "StockThresholds")
    },
    **{
        name: "factorminer.evaluation.model_zoo"
        for name in (
            "ModelZooConfig",
            "ModelZooEvaluator",
            "ModelCoOptimizationReport",
            "FactorContributionSummary",
        )
    },
    **{
        name: "factorminer.evaluation.pipeline"
        for name in (
            "CandidateFactor",
            "EvaluationResult",
            "FactorLibraryView",
            "PipelineConfig",
            "ValidationPipeline",
            "run_evaluation_pipeline",
        )
    },
    "FactorCombiner": "factorminer.evaluation.combination",
    "FactorSelector": "factorminer.evaluation.selection",
    "PortfolioBacktester": "factorminer.evaluation.portfolio",
    **{
        name: "factorminer.evaluation.backtest"
        for name in (
            "SplitWindow",
            "DrawdownResult",
            "train_test_split",
            "rolling_splits",
            "compute_ic_series",
            "compute_rolling_ic",
            "compute_cumulative_ic",
            "compute_ic_stats",
            "factor_return_attribution",
            "compute_drawdown",
            "compute_sharpe_ratio",
            "compute_calmar_ratio",
        )
    },
    **{
        name: "factorminer.evaluation.regime"
        for name in (
            "MarketRegime",
            "RegimeConfig",
            "RegimeClassification",
            "RegimeDetector",
            "RegimeICResult",
            "RegimeAwareEvaluator",
        )
    },
    **{
        name: "factorminer.evaluation.capacity"
        for name in (
            "CapacityConfig",
            "CapacityEstimate",
            "CapacityEstimator",
            "MarketImpactEstimate",
            "MarketImpactModel",
            "NetCostResult",
        )
    },
    **{
        name: "factorminer.evaluation.causal"
        for name in ("CausalConfig", "CausalTestResult", "CausalValidator")
    },
    **{
        name: "factorminer.evaluation.significance"
        for name in (
            "BootstrapCIResult",
            "BootstrapICTester",
            "DeflatedSharpeCalculator",
            "DeflatedSharpeResult",
            "FDRController",
            "FDRResult",
            "SignificanceConfig",
            "check_significance",
        )
    },
    **{
        name: "factorminer.evaluation.research"
        for name in (
            "FactorGeometryDiagnostics",
            "FactorScoreVector",
            "compute_factor_geometry",
            "build_score_vector",
            "passes_research_admission",
            "run_research_model_suite",
        )
    },
    **{
        name: "factorminer.evaluation.cross_validation"
        for name in (
            "CombinatorialPurgedCV",
            "CPCVSplit",
            "CrossValidationConfig",
            "PBOResult",
            "ProbabilityOfBacktestOverfitting",
        )
    },
    **{
        name: "factorminer.evaluation.risk_portfolio"
        for name in (
            "HRPOptimizer",
            "RiskParityOptimizer",
            "CVaRPortfolioOptimizer",
            "RiskPortfolioConfig",
            "RiskPortfolioResult",
            "construct_portfolio",
        )
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
