"""Seed knowledge used by the paper-faithful memory policy."""

from __future__ import annotations

from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    MiningState,
    StrategicInsight,
    SuccessPattern,
)


def default_success_patterns() -> list[SuccessPattern]:
    """Initial success patterns from FactorMiner Table 4."""
    return [
        SuccessPattern(
            name="Higher Moment Regimes",
            description=(
                "Use Skew/Kurt as IfElse conditions to route between different "
                "factor computations. High-moment regime switching captures "
                "non-linear market states effectively."
            ),
            template="IfElse(Skew($close, 20), <factor_a>, <factor_b>)",
            success_rate="High",
            example_factors=["HMR_001", "HMR_002"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="PV Corr Interaction",
            description=(
                "Price-volume correlation interaction: use rolling Corr($close, $volume) "
                "as a signal or conditioning variable. Captures supply-demand imbalance "
                "through price-volume divergence."
            ),
            template="CsRank(Corr($close, $volume, 20))",
            success_rate="High",
            example_factors=["PVC_001", "PVC_002"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Robust Efficiency",
            description=(
                "Use Median for noise filtering instead of Mean. Rolling median "
                "is more robust to outliers in intraday data, producing factors "
                "with higher ICIR."
            ),
            template="CsRank(Div(Median($close, 10), Median($close, 60)))",
            success_rate="High",
            example_factors=["RE_001"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Smoothed Efficiency Rank",
            description=(
                "Combine EMA smoothing with CsRank cross-sectional normalization. "
                "EMA reduces noise while CsRank ensures cross-sectional comparability."
            ),
            template="CsRank(EMA(Div($close, Mean($close, 20)), 10))",
            success_rate="High",
            example_factors=["SER_001", "SER_002"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Trend Regression Adaptive",
            description=(
                "Use TsLinRegSlope, TsLinRegResid, or rolling R-squared to capture "
                "trend strength and mean reversion. Regression residuals identify "
                "deviations from local trends."
            ),
            template="CsRank(TsLinRegSlope($close, 20))",
            success_rate="High",
            example_factors=["TRA_001", "TRA_002"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Logical Or Extreme Regimes",
            description=(
                "Use Or/And with Greater/Less to combine multiple extreme-value "
                "conditions. Captures compound regime states that single indicators miss."
            ),
            template="IfElse(Or(Greater(Skew($returns, 20), 1), Less(Kurt($returns, 20), -1)), <a>, <b>)",
            success_rate="Medium",
            example_factors=["LOR_001"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Kurtosis Regime",
            description=(
                "Use rolling kurtosis to detect fat-tail regimes and switch "
                "factor behavior accordingly. High kurtosis indicates regime "
                "changes and trend breaks."
            ),
            template="IfElse(Kurt($returns, 20), CsRank(Std($returns, 10)), CsRank(Mean($returns, 10)))",
            success_rate="Medium",
            example_factors=["KR_001"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Amt Efficiency Rank Interaction",
            description=(
                "Combine $amt (turnover) with efficiency ratios and CsRank. "
                "Amount-weighted efficiency captures liquidity-adjusted momentum."
            ),
            template="CsRank(Div(EMA($amt, 5), EMA($amt, 20)))",
            success_rate="Medium",
            example_factors=["AER_001"],
            occurrence_count=0,
        ),
    ]


def default_forbidden_directions() -> list[ForbiddenDirection]:
    """Initial forbidden directions from FactorMiner Table 5."""
    return [
        ForbiddenDirection(
            name="Standardized Returns/Amount",
            description=(
                "CsZScore or Std-normalized $returns and $amt variants. "
                "These produce a cluster of highly correlated factors."
            ),
            correlated_factors=["std_ret_cluster"],
            typical_correlation=0.6,
            reason="Standardized return/amount variants cluster with rho > 0.6",
            occurrence_count=0,
        ),
        ForbiddenDirection(
            name="VWAP Deviation variants",
            description=(
                "Factors based on deviation from VWAP (Sub($close, $vwap) or "
                "Delta($vwap)). All VWAP deviation variants converge to the "
                "same signal."
            ),
            correlated_factors=["vwap_dev_cluster"],
            typical_correlation=0.5,
            reason="VWAP deviation variants produce highly correlated factors (rho > 0.5)",
            occurrence_count=0,
        ),
        ForbiddenDirection(
            name="Simple Delta Reversal",
            description=(
                "Simple price-change reversal factors using Delta($close) or "
                "Neg(Return($close)). These are well-known and already "
                "saturated in most factor libraries."
            ),
            correlated_factors=["delta_rev_cluster"],
            typical_correlation=0.5,
            reason="Simple delta-based reversal factors are redundant (rho > 0.5)",
            occurrence_count=0,
        ),
        ForbiddenDirection(
            name="WMA/EMA Smoothed Efficiency",
            description=(
                "Smoothing the same base signal with WMA, EMA, SMA, DEMA "
                "produces nearly identical factors. Different smoothing methods "
                "on the same input do not add diversity."
            ),
            correlated_factors=["smoothed_eff_cluster"],
            typical_correlation=0.9,
            reason="WMA/EMA/SMA smoothed efficiency variants nearly identical (rho > 0.9)",
            occurrence_count=0,
        ),
    ]


def default_insights() -> list[StrategicInsight]:
    """Initial strategic insights from the paper."""
    return [
        StrategicInsight(
            insight="Non-linear transformations (IfElse, Skew, Kurt) outperform linear ones",
            evidence="Paper finding: regime-switching factors consistently achieve higher IC",
            batch_source=0,
        ),
        StrategicInsight(
            insight="Cross-sectional ranking (CsRank) as final layer improves factor stability",
            evidence="CsRank normalization reduces outlier sensitivity and improves ICIR",
            batch_source=0,
        ),
        StrategicInsight(
            insight="Combining operators from different categories produces more diverse factors",
            evidence="Multi-category composition (e.g., Statistical + Logical + CrossSectional) "
            "reduces correlation with existing library members",
            batch_source=0,
        ),
    ]


# ---------------------------------------------------------------------------


def create_default_memory() -> ExperienceMemory:
    """Create an independent, version-zero paper memory state."""
    return ExperienceMemory(
        state=MiningState(),
        success_patterns=default_success_patterns(),
        forbidden_directions=default_forbidden_directions(),
        insights=default_insights(),
        version=0,
    )
