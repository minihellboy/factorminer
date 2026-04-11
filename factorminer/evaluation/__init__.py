"""Multi-stage factor evaluation and validation pipeline."""

from factorminer.evaluation.admission import (
    AdmissionDecision,
    StockThresholds,
    check_admission,
    check_replacement,
)
from factorminer.evaluation.backtest import (
    DrawdownResult,
    SplitWindow,
    compute_calmar_ratio,
    compute_cumulative_ic,
    compute_drawdown,
    compute_ic_series,
    compute_ic_stats,
    compute_rolling_ic,
    compute_sharpe_ratio,
    factor_return_attribution,
    rolling_splits,
    train_test_split,
)
from factorminer.evaluation.capacity import (
    CapacityConfig,
    CapacityEstimate,
    CapacityEstimator,
    MarketImpactEstimate,
    MarketImpactModel,
    NetCostResult,
)
from factorminer.evaluation.causal import (
    CausalConfig,
    CausalTestResult,
    CausalValidator,
)
from factorminer.evaluation.combination import FactorCombiner
from factorminer.evaluation.correlation import (
    IncrementalCorrelationMatrix,
    batch_spearman_correlation,
    batch_spearman_pairwise,
    compute_correlation_batch,
)
from factorminer.evaluation.metrics import (
    compute_factor_stats,
    compute_ic,
    compute_ic_mean,
    compute_ic_vectorized,
    compute_ic_win_rate,
    compute_icir,
    compute_pairwise_correlation,
    compute_quintile_returns,
    compute_turnover,
)
from factorminer.evaluation.pipeline import (
    CandidateFactor,
    EvaluationResult,
    FactorLibraryView,
    PipelineConfig,
    ValidationPipeline,
    run_evaluation_pipeline,
)
from factorminer.evaluation.portfolio import PortfolioBacktester
from factorminer.evaluation.regime import (
    MarketRegime,
    RegimeAwareEvaluator,
    RegimeClassification,
    RegimeConfig,
    RegimeDetector,
    RegimeICResult,
)
from factorminer.evaluation.research import (
    FactorGeometryDiagnostics,
    FactorScoreVector,
    build_score_vector,
    compute_factor_geometry,
    passes_research_admission,
    run_research_model_suite,
)
from factorminer.evaluation.selection import FactorSelector
from factorminer.evaluation.significance import (
    BootstrapCIResult,
    BootstrapICTester,
    DeflatedSharpeCalculator,
    DeflatedSharpeResult,
    FDRController,
    FDRResult,
    SignificanceConfig,
    check_significance,
)

__all__ = [
    # metrics
    "compute_ic",
    "compute_ic_vectorized",
    "compute_icir",
    "compute_ic_mean",
    "compute_ic_win_rate",
    "compute_pairwise_correlation",
    "compute_factor_stats",
    "compute_quintile_returns",
    "compute_turnover",
    # correlation
    "batch_spearman_correlation",
    "batch_spearman_pairwise",
    "compute_correlation_batch",
    "IncrementalCorrelationMatrix",
    # admission
    "check_admission",
    "check_replacement",
    "AdmissionDecision",
    "StockThresholds",
    # pipeline
    "CandidateFactor",
    "EvaluationResult",
    "FactorLibraryView",
    "PipelineConfig",
    "ValidationPipeline",
    "run_evaluation_pipeline",
    # combination / selection / backtest
    "FactorCombiner",
    "FactorSelector",
    "PortfolioBacktester",
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
    # regime
    "MarketRegime",
    "RegimeConfig",
    "RegimeClassification",
    "RegimeDetector",
    "RegimeICResult",
    "RegimeAwareEvaluator",
    # capacity
    "CapacityConfig",
    "CapacityEstimate",
    "CapacityEstimator",
    "MarketImpactEstimate",
    "MarketImpactModel",
    "NetCostResult",
    # causal
    "CausalConfig",
    "CausalTestResult",
    "CausalValidator",
    # significance
    "BootstrapCIResult",
    "BootstrapICTester",
    "DeflatedSharpeCalculator",
    "DeflatedSharpeResult",
    "FDRController",
    "FDRResult",
    "SignificanceConfig",
    "check_significance",
    # research
    "FactorGeometryDiagnostics",
    "FactorScoreVector",
    "compute_factor_geometry",
    "build_score_vector",
    "passes_research_admission",
    "run_research_model_suite",
]
