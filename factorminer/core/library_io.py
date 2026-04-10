"""Serialization and I/O for the FactorLibrary.

Provides save/load to JSON + optional binary signal cache (.npz),
CSV export, formula export, and import of the 110 factors from the paper.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from factorminer.core.factor_library import Factor, FactorLibrary

logger = logging.getLogger(__name__)


# ======================================================================
# Save / Load
# ======================================================================

def save_library(
    library: FactorLibrary,
    path: Union[str, Path],
    save_signals: bool = True,
) -> None:
    """Save a FactorLibrary to disk.

    Creates two files:
    - ``<path>.json`` -- factor metadata and library configuration
    - ``<path>_signals.npz`` -- binary signal cache (if save_signals=True
      and any factors have signals)

    Parameters
    ----------
    library : FactorLibrary
    path : str or Path
        Base path (without extension). E.g. ``"output/my_library"`` produces
        ``output/my_library.json`` and ``output/my_library_signals.npz``.
    save_signals : bool
        Whether to write the binary signal cache.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # -- Metadata JSON --
    meta = {
        "correlation_threshold": library.correlation_threshold,
        "ic_threshold": library.ic_threshold,
        "dependence_metric": library.dependence_metric.name,
        "next_id": library._next_id,
        "factors": [f.to_dict() for f in library.list_factors()],
    }
    if library.correlation_matrix is not None:
        meta["correlation_matrix"] = library.correlation_matrix.tolist()
    meta["id_to_index"] = {str(k): v for k, v in library._id_to_index.items()}

    json_path = path.with_suffix(".json")
    with open(json_path, "w") as fp:
        json.dump(meta, fp, indent=2)
    logger.info("Saved library metadata to %s (%d factors)", json_path, library.size)

    # -- Binary signal cache --
    if save_signals:
        signal_arrays: Dict[str, np.ndarray] = {}
        for f in library.list_factors():
            if f.signals is not None:
                signal_arrays[f"factor_{f.id}"] = f.signals

        if signal_arrays:
            npz_path = Path(str(path) + "_signals.npz")
            np.savez_compressed(npz_path, **signal_arrays)
            logger.info(
                "Saved signal cache to %s (%d arrays)",
                npz_path, len(signal_arrays),
            )


def load_library(path: Union[str, Path]) -> FactorLibrary:
    """Load a FactorLibrary from disk.

    Parameters
    ----------
    path : str or Path
        Base path (without extension). Will look for ``<path>.json`` and
        optionally ``<path>_signals.npz``.

    Returns
    -------
    FactorLibrary
    """
    path = Path(path)
    json_path = path.with_suffix(".json")

    with open(json_path, "r") as fp:
        meta = json.load(fp)

    library = FactorLibrary(
        correlation_threshold=meta.get("correlation_threshold", 0.5),
        ic_threshold=meta.get("ic_threshold", 0.04),
        dependence_metric=meta.get("dependence_metric", "spearman"),
    )
    library._next_id = meta.get("next_id", 1)

    # Restore factors
    for fd in meta.get("factors", []):
        factor = Factor.from_dict(fd)
        library.factors[factor.id] = factor

    # Restore correlation matrix
    if "correlation_matrix" in meta and meta["correlation_matrix"] is not None:
        library.correlation_matrix = np.array(
            meta["correlation_matrix"], dtype=np.float64
        )

    # Restore id-to-index mapping
    if "id_to_index" in meta:
        library._id_to_index = {
            int(k): v for k, v in meta["id_to_index"].items()
        }

    # Load signal cache if present
    npz_path = Path(str(path) + "_signals.npz")
    if npz_path.exists():
        data = np.load(npz_path)
        for f in library.factors.values():
            key = f"factor_{f.id}"
            if key in data:
                f.signals = data[key]
        data.close()
        logger.info("Loaded signal cache from %s", npz_path)

    logger.info(
        "Loaded library from %s (%d factors)", json_path, library.size
    )
    return library


# ======================================================================
# Export utilities
# ======================================================================

def export_csv(library: FactorLibrary, path: Union[str, Path]) -> None:
    """Export the factor table to CSV.

    Columns: ID, Name, Formula, Category, IC_Mean, ICIR, IC_Win_Rate,
    Max_Correlation, Batch, Admission_Date
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ID", "Name", "Formula", "Category", "IC_Mean", "ICIR",
        "IC_Win_Rate", "Max_Correlation", "Batch", "Admission_Date",
    ]

    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for f in library.list_factors():
            writer.writerow({
                "ID": f.id,
                "Name": f.name,
                "Formula": f.formula,
                "Category": f.category,
                "IC_Mean": f"{f.ic_mean:.6f}",
                "ICIR": f"{f.icir:.6f}",
                "IC_Win_Rate": f"{f.ic_win_rate:.4f}",
                "Max_Correlation": f"{f.max_correlation:.4f}",
                "Batch": f.batch_number,
                "Admission_Date": f.admission_date,
            })

    logger.info("Exported %d factors to %s", library.size, path)


def export_formulas(library: FactorLibrary, path: Union[str, Path]) -> None:
    """Export just the formulas for reproduction.

    One formula per line, prefixed with the factor ID and name.
    Format: ``ID | Name | Formula``
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as fp:
        fp.write("# FactorMiner Library Formulas\n")
        fp.write("# ID | Name | Formula\n")
        fp.write(f"# Total: {library.size} factors\n")
        fp.write("#" + "-" * 78 + "\n")
        for f in library.list_factors():
            fp.write(f"{f.id:04d} | {f.name} | {f.formula}\n")

    logger.info("Exported %d formulas to %s", library.size, path)


# ======================================================================
# Paper factor catalog (110 factors from Appendix P)
# ======================================================================

# Representative subset of the 110 factors discovered by FactorMiner.
# Each entry: (name, formula, category)
PAPER_FACTORS: List[Dict[str, str]] = [
    # Factor 001
    {
        "name": "Intraday Range Position",
        "formula": "Neg(CsRank(Div(Sub($close, TsMin($close, 48)), Add(Sub(TsMax($close, 48), TsMin($close, 48)), 1e-8))))",
        "category": "Mean-reversion",
    },
    # Factor 002
    {
        "name": "Volume-Weighted Momentum",
        "formula": "Neg(CsRank(Mul(Return($close, 5), Div($volume, Mean($volume, 20)))))",
        "category": "Momentum",
    },
    # Factor 003
    {
        "name": "Residual Volatility",
        "formula": "Neg(CsRank(Std(Sub($close, EMA($close, 10)), 20)))",
        "category": "Volatility",
    },
    # Factor 004
    {
        "name": "Intraday Amplitude Ratio",
        "formula": "Neg(CsRank(Div(Sub($high, $low), Add($close, 1e-8))))",
        "category": "Volatility",
    },
    # Factor 005
    {
        "name": "Volume Surprise",
        "formula": "Neg(CsRank(Div(Sub($volume, Mean($volume, 20)), Add(Std($volume, 20), 1e-8))))",
        "category": "Volume",
    },
    # Factor 006
    {
        "name": "VWAP Deviation",
        "formula": "Neg(Div(Sub($close, $vwap), $vwap))",
        "category": "VWAP",
    },
    # Factor 007
    {
        "name": "Short-term Reversal",
        "formula": "Neg(CsRank(Return($close, 3)))",
        "category": "Mean-reversion",
    },
    # Factor 008
    {
        "name": "Turnover Momentum",
        "formula": "Neg(CsRank(Delta(Div($amt, Add($volume, 1e-8)), 5)))",
        "category": "Turnover",
    },
    # Factor 009
    {
        "name": "High-Low Midpoint Reversion",
        "formula": "Neg(CsRank(Sub($close, Div(Add($high, $low), 2))))",
        "category": "Mean-reversion",
    },
    # Factor 010
    {
        "name": "Rolling Beta Residual",
        "formula": "Neg(CsRank(Resid($returns, Mean($returns, 20), 20)))",
        "category": "Risk",
    },
    # Factor 011
    {
        "name": "VWAP Slope",
        "formula": "Neg(CsRank(TsLinRegSlope(Div(Sub($close, $vwap), $vwap), 10)))",
        "category": "VWAP",
    },
    # Factor 012
    {
        "name": "Accumulation-Distribution",
        "formula": "Neg(CsRank(Sum(Mul(Div(Sub(Mul(2, $close), Add($high, $low)), Add(Sub($high, $low), 1e-8)), $volume), 10)))",
        "category": "Volume",
    },
    # Factor 013
    {
        "name": "Relative Strength Index Deviation",
        "formula": "Neg(CsRank(Sub(Mean(Max(Delta($close, 1), 0), 14), Mean(Abs(Min(Delta($close, 1), 0)), 14))))",
        "category": "Momentum",
    },
    # Factor 014
    {
        "name": "Price-Volume Correlation",
        "formula": "Neg(Corr($close, $volume, 10))",
        "category": "Volume",
    },
    # Factor 015
    {
        "name": "Skewness of Returns",
        "formula": "Neg(CsRank(Skew($returns, 20)))",
        "category": "Higher-moment",
    },
    # Factor 016
    {
        "name": "Kurtosis of Returns",
        "formula": "Neg(CsRank(Kurt($returns, 20)))",
        "category": "Higher-moment",
    },
    # Factor 017
    {
        "name": "Volume-Weighted Return",
        "formula": "Neg(CsRank(Div(Sum(Mul($returns, $volume), 10), Add(Sum($volume, 10), 1e-8))))",
        "category": "Volume",
    },
    # Factor 018
    {
        "name": "Close-to-High Ratio",
        "formula": "Neg(CsRank(Div(Sub($high, $close), Add($high, 1e-8))))",
        "category": "Mean-reversion",
    },
    # Factor 019
    {
        "name": "Delayed Correlation Shift",
        "formula": "Neg(CsRank(Sub(Corr($close, $volume, 10), Corr(Delay($close, 5), $volume, 10))))",
        "category": "Volume",
    },
    # Factor 020
    {
        "name": "Exponential Momentum",
        "formula": "Neg(CsRank(Sub($close, EMA($close, 20))))",
        "category": "Momentum",
    },
    # Factor 021
    {
        "name": "Range-Adjusted Volume",
        "formula": "Neg(CsRank(Div($volume, Add(Sub($high, $low), 1e-8))))",
        "category": "Volume",
    },
    # Factor 022
    {
        "name": "Cumulative Return Rank",
        "formula": "Neg(CsRank(Sum($returns, 10)))",
        "category": "Momentum",
    },
    # Factor 023
    {
        "name": "VWAP Momentum",
        "formula": "Neg(CsRank(Return($vwap, 5)))",
        "category": "VWAP",
    },
    # Factor 024
    {
        "name": "Bollinger Band Position",
        "formula": "Neg(CsRank(Div(Sub($close, Mean($close, 20)), Add(Std($close, 20), 1e-8))))",
        "category": "Mean-reversion",
    },
    # Factor 025
    {
        "name": "Volume Decay Weighted",
        "formula": "Neg(CsRank(Decay($volume, 10)))",
        "category": "Volume",
    },
    # Factor 026
    {
        "name": "Overnight Return",
        "formula": "Neg(CsRank(Div(Sub($open, Delay($close, 1)), Add(Delay($close, 1), 1e-8))))",
        "category": "Overnight",
    },
    # Factor 027
    {
        "name": "Intraday Return",
        "formula": "Neg(CsRank(Div(Sub($close, $open), Add($open, 1e-8))))",
        "category": "Intraday",
    },
    # Factor 028
    {
        "name": "Max Drawdown",
        "formula": "Neg(CsRank(Div(Sub($close, TsMax($close, 20)), Add(TsMax($close, 20), 1e-8))))",
        "category": "Risk",
    },
    # Factor 029
    {
        "name": "Hurst Exponent Proxy",
        "formula": "Neg(CsRank(Div(Std($returns, 20), Add(Std($returns, 5), 1e-8))))",
        "category": "Volatility",
    },
    # Factor 030
    {
        "name": "Volume Imbalance",
        "formula": "Neg(CsRank(Sub(Mean($volume, 5), Mean($volume, 20))))",
        "category": "Volume",
    },
    # Factor 031
    {
        "name": "Weighted Close Position",
        "formula": "Neg(CsRank(Div(Sub(Mul(2, $close), Add($high, $low)), Add(Sub($high, $low), 1e-8))))",
        "category": "Mean-reversion",
    },
    # Factor 032
    {
        "name": "Trend Intensity",
        "formula": "Neg(CsRank(Div(Abs(Delta($close, 10)), Add(Sum(Abs(Delta($close, 1)), 10), 1e-8))))",
        "category": "Trend",
    },
    # Factor 033
    {
        "name": "Return Dispersion",
        "formula": "Neg(CsRank(Std($returns, 5)))",
        "category": "Volatility",
    },
    # Factor 034
    {
        "name": "VWAP Relative Strength",
        "formula": "Neg(CsRank(Div(Sub(Mean($close, 5), $vwap), Add($vwap, 1e-8))))",
        "category": "VWAP",
    },
    # Factor 035
    {
        "name": "Rank Reversal",
        "formula": "Neg(CsRank(Sub(TsRank($close, 10), TsRank($close, 30))))",
        "category": "Mean-reversion",
    },
    # Factor 036
    {
        "name": "Money Flow Index",
        "formula": "Neg(CsRank(Div(Sum(Mul(Max(Delta($close, 1), 0), $volume), 14), Add(Sum(Mul(Abs(Delta($close, 1)), $volume), 14), 1e-8))))",
        "category": "Volume",
    },
    # Factor 037
    {
        "name": "Adaptive Momentum",
        "formula": "Neg(CsRank(Mul(Return($close, 10), Div(Std($returns, 5), Add(Std($returns, 20), 1e-8)))))",
        "category": "Momentum",
    },
    # Factor 038
    {
        "name": "Volume Trend",
        "formula": "Neg(CsRank(TsLinRegSlope($volume, 10)))",
        "category": "Volume",
    },
    # Factor 039
    {
        "name": "Price Acceleration",
        "formula": "Neg(CsRank(Sub(Delta($close, 5), Delta(Delay($close, 5), 5))))",
        "category": "Momentum",
    },
    # Factor 040
    {
        "name": "Realized Volatility Ratio",
        "formula": "Neg(CsRank(Div(Std($returns, 10), Add(Std($returns, 30), 1e-8))))",
        "category": "Volatility",
    },
    # Factor 041
    {
        "name": "Amount Concentration",
        "formula": "Neg(CsRank(Div(TsMax($amt, 5), Add(Mean($amt, 20), 1e-8))))",
        "category": "Turnover",
    },
    # Factor 042
    {
        "name": "Cross-Sectional Volume Rank",
        "formula": "Neg(CsRank(Div($volume, Add(Mean($volume, 60), 1e-8))))",
        "category": "Volume",
    },
    # Factor 043
    {
        "name": "Gap Momentum",
        "formula": "Neg(CsRank(Sum(Div(Sub($open, Delay($close, 1)), Add(Delay($close, 1), 1e-8)), 5)))",
        "category": "Overnight",
    },
    # Factor 044
    {
        "name": "VWAP Distance Decay",
        "formula": "Neg(CsRank(Decay(Div(Sub($close, $vwap), Add($vwap, 1e-8)), 10)))",
        "category": "VWAP",
    },
    # Factor 045
    {
        "name": "Tail Risk Indicator",
        "formula": "Neg(CsRank(Div(TsMin($returns, 20), Add(Std($returns, 20), 1e-8))))",
        "category": "Risk",
    },
    # Factor 046
    {
        "name": "Volatility-Regime Reversal Divergence",
        "formula": "IfElse(Greater(Std($returns, 12), Mean(Std($returns, 12), 48)), Neg(CsRank(Delta($close, 3))), Neg(CsRank(Div(Sub($close, $low), Add(Sub($high, $low), 0.0001)))))",
        "category": "Regime-switching",
    },
    # Factor 047
    {
        "name": "Regime Volume Signal",
        "formula": "IfElse(Greater($volume, Mean($volume, 20)), Neg(CsRank($returns)), Neg(CsRank(Return($close, 5))))",
        "category": "Regime-switching",
    },
    # Factor 048
    {
        "name": "Liquidity-Adjusted Reversal",
        "formula": "Neg(CsRank(Mul(Return($close, 3), Div($volume, Add(Mean($volume, 20), 1e-8)))))",
        "category": "Mean-reversion",
    },
    # Factor 049
    {
        "name": "Cross-Sectional Volatility Rank",
        "formula": "Neg(CsRank(CsRank(Std($returns, 10))))",
        "category": "Volatility",
    },
    # Factor 050
    {
        "name": "VWAP Bollinger",
        "formula": "Neg(CsRank(Div(Sub($vwap, Mean($vwap, 20)), Add(Std($vwap, 20), 1e-8))))",
        "category": "VWAP",
    },
    # Factor 051
    {
        "name": "Smoothed Return Reversal",
        "formula": "Neg(CsRank(EMA($returns, 5)))",
        "category": "Mean-reversion",
    },
    # Factor 052
    {
        "name": "Volume-Price Divergence",
        "formula": "Neg(CsRank(Sub(TsRank($volume, 10), TsRank($close, 10))))",
        "category": "Volume",
    },
    # Factor 053
    {
        "name": "Decay Weighted Momentum",
        "formula": "Neg(CsRank(Decay($returns, 20)))",
        "category": "Momentum",
    },
    # Factor 054
    {
        "name": "Range Percentile",
        "formula": "Neg(CsRank(Div(Sub($close, TsMin($close, 20)), Add(Sub(TsMax($close, 20), TsMin($close, 20)), 1e-8))))",
        "category": "Mean-reversion",
    },
    # Factor 055
    {
        "name": "Volume Skewness",
        "formula": "Neg(CsRank(Skew($volume, 20)))",
        "category": "Volume",
    },
    # Factor 056
    {
        "name": "Residual Momentum",
        "formula": "Neg(CsRank(TsLinRegResid($close, 20)))",
        "category": "Momentum",
    },
    # Factor 057
    {
        "name": "VWAP Trend",
        "formula": "Neg(CsRank(Delta(Div(Sub($close, $vwap), $vwap), 5)))",
        "category": "VWAP",
    },
    # Factor 058
    {
        "name": "Return Autocorrelation",
        "formula": "Neg(CsRank(Corr($returns, Delay($returns, 1), 10)))",
        "category": "Mean-reversion",
    },
    # Factor 059
    {
        "name": "Price Efficiency",
        "formula": "Neg(CsRank(Div(Abs(Sum($returns, 10)), Add(Sum(Abs($returns), 10), 1e-8))))",
        "category": "Trend",
    },
    # Factor 060
    {
        "name": "Relative Volume Change",
        "formula": "Neg(CsRank(Return($volume, 5)))",
        "category": "Volume",
    },
    # Factor 061
    {
        "name": "Weighted VWAP Position",
        "formula": "Neg(CsRank(WMA(Div(Sub($close, $vwap), $vwap), 10)))",
        "category": "VWAP",
    },
    # Factor 062
    {
        "name": "Regime Momentum Flip",
        "formula": "IfElse(Greater(Mean($returns, 5), 0), Neg(CsRank(Return($close, 10))), CsRank(Return($close, 3)))",
        "category": "Regime-switching",
    },
    # Factor 063
    {
        "name": "High-Low Volatility",
        "formula": "Neg(CsRank(Mean(Div(Sub($high, $low), Add($close, 1e-8)), 10)))",
        "category": "Volatility",
    },
    # Factor 064
    {
        "name": "Opening Gap Reversal",
        "formula": "Neg(CsRank(Div(Sub($open, Delay($close, 1)), Add(Std($returns, 10), 1e-8))))",
        "category": "Overnight",
    },
    # Factor 065
    {
        "name": "Volume Momentum Spread",
        "formula": "Neg(CsRank(Sub(Mean($volume, 5), Mean($volume, 40))))",
        "category": "Volume",
    },
    # Factor 066
    {
        "name": "Regime Volume Reversal",
        "formula": "IfElse(Greater(Div($volume, Add(Mean($volume, 20), 1e-8)), 1.5), Neg(CsRank($returns)), Neg(CsRank(Return($close, 10))))",
        "category": "Regime-switching",
    },
    # Factor 067
    {
        "name": "Slope Reversal",
        "formula": "Neg(CsRank(TsLinRegSlope($close, 5)))",
        "category": "Mean-reversion",
    },
    # Factor 068
    {
        "name": "VWAP Momentum Decay",
        "formula": "Neg(CsRank(Decay(Return($vwap, 3), 10)))",
        "category": "VWAP",
    },
    # Factor 069
    {
        "name": "Turnover Rate Change",
        "formula": "Neg(CsRank(Delta(Div($amt, Add($volume, 1e-8)), 10)))",
        "category": "Turnover",
    },
    # Factor 070
    {
        "name": "Return Quantile Signal",
        "formula": "Neg(CsRank(Quantile($returns, 20, 0.75)))",
        "category": "Higher-moment",
    },
    # Factor 071
    {
        "name": "Double EMA Crossover",
        "formula": "Neg(CsRank(Sub(EMA($close, 5), EMA($close, 20))))",
        "category": "Trend",
    },
    # Factor 072
    {
        "name": "Conditional Volatility Return",
        "formula": "Neg(CsRank(Div($returns, Add(Std($returns, 10), 1e-8))))",
        "category": "Risk",
    },
    # Factor 073
    {
        "name": "Amplitude Trend",
        "formula": "Neg(CsRank(TsLinRegSlope(Div(Sub($high, $low), Add($close, 1e-8)), 10)))",
        "category": "Volatility",
    },
    # Factor 074
    {
        "name": "Volume-Weighted Range",
        "formula": "Neg(CsRank(Mean(Mul(Div(Sub($high, $low), Add($close, 1e-8)), $volume), 10)))",
        "category": "Volume",
    },
    # Factor 075
    {
        "name": "Intraday Efficiency Ratio",
        "formula": "Neg(CsRank(Div(Abs(Sub($close, $open)), Add(Sub($high, $low), 1e-8))))",
        "category": "Intraday",
    },
    # Factor 076
    {
        "name": "Cumulative Volume Signal",
        "formula": "Neg(CsRank(Div(Sum(Mul($returns, $volume), 20), Add(Sum($volume, 20), 1e-8))))",
        "category": "Volume",
    },
    # Factor 077
    {
        "name": "VWAP Cross-Sectional Momentum",
        "formula": "Neg(CsRank(CsRank(Return($vwap, 10))))",
        "category": "VWAP",
    },
    # Factor 078
    {
        "name": "Mean-Reversion Indicator",
        "formula": "Neg(CsRank(Div(Sub($close, SMA($close, 10)), Add(SMA($close, 10), 1e-8))))",
        "category": "Mean-reversion",
    },
    # Factor 079
    {
        "name": "Volume Regime Indicator",
        "formula": "Neg(CsRank(Div(Std($volume, 5), Add(Std($volume, 20), 1e-8))))",
        "category": "Volume",
    },
    # Factor 080
    {
        "name": "Return Persistence",
        "formula": "Neg(CsRank(Mul(Sign(Delta($close, 1)), Sign(Delta($close, 5)))))",
        "category": "Momentum",
    },
    # Factor 081
    {
        "name": "Regime Trend Strength",
        "formula": "IfElse(Greater(Abs(TsLinRegSlope($close, 20)), Std($close, 20)), Neg(CsRank(TsLinRegSlope($close, 5))), Neg(CsRank(Return($close, 3))))",
        "category": "Regime-switching",
    },
    # Factor 082
    {
        "name": "VWAP Dispersion",
        "formula": "Neg(CsRank(Std(Div(Sub($close, $vwap), $vwap), 10)))",
        "category": "VWAP",
    },
    # Factor 083
    {
        "name": "Smart Money Flow",
        "formula": "Neg(CsRank(Sum(Mul(IfElse(Greater($close, Delay($close, 1)), $volume, Neg($volume)), Div(Sub($high, $low), Add($close, 1e-8))), 10)))",
        "category": "Volume",
    },
    # Factor 084
    {
        "name": "Return Rank Dispersion",
        "formula": "Neg(CsRank(Sub(TsRank($returns, 5), TsRank($returns, 20))))",
        "category": "Mean-reversion",
    },
    # Factor 085
    {
        "name": "Volume Acceleration",
        "formula": "Neg(CsRank(Sub(Delta($volume, 5), Delta(Delay($volume, 5), 5))))",
        "category": "Volume",
    },
    # Factor 086
    {
        "name": "Close-Low Ratio Trend",
        "formula": "Neg(CsRank(Mean(Div(Sub($close, $low), Add(Sub($high, $low), 1e-8)), 5)))",
        "category": "Mean-reversion",
    },
    # Factor 087
    {
        "name": "Hull MA Deviation",
        "formula": "Neg(CsRank(Div(Sub($close, HMA($close, 10)), Add(Std($close, 10), 1e-8))))",
        "category": "Trend",
    },
    # Factor 088
    {
        "name": "DEMA Momentum Signal",
        "formula": "Neg(CsRank(Sub(DEMA($close, 5), DEMA($close, 20))))",
        "category": "Momentum",
    },
    # Factor 089
    {
        "name": "Volume Profile Skew",
        "formula": "Neg(CsRank(Skew(Div($volume, Add(Mean($volume, 20), 1e-8)), 10)))",
        "category": "Volume",
    },
    # Factor 090
    {
        "name": "Conditional VWAP Signal",
        "formula": "IfElse(Greater($close, $vwap), Neg(CsRank(Div(Sub($close, $vwap), $vwap))), CsRank(Div(Sub($vwap, $close), $vwap)))",
        "category": "VWAP",
    },
    # Factor 091
    {
        "name": "Extreme Volume Reversal",
        "formula": "Neg(CsRank(Mul(IfElse(Greater($volume, Mul(2, Mean($volume, 20))), 1, 0), $returns)))",
        "category": "Volume",
    },
    # Factor 092
    {
        "name": "Range Expansion Signal",
        "formula": "Neg(CsRank(Div(Sub($high, $low), Add(Mean(Sub($high, $low), 20), 1e-8))))",
        "category": "Volatility",
    },
    # Factor 093
    {
        "name": "Short-Term IC Momentum",
        "formula": "Neg(CsRank(Sum(Mul(Sign($returns), Abs($returns)), 5)))",
        "category": "Momentum",
    },
    # Factor 094
    {
        "name": "VWAP Curvature",
        "formula": "Neg(CsRank(Sub(Div(Sub($vwap, Delay($vwap, 5)), Add(Delay($vwap, 5), 1e-8)), Div(Sub(Delay($vwap, 5), Delay($vwap, 10)), Add(Delay($vwap, 10), 1e-8)))))",
        "category": "VWAP",
    },
    # Factor 095
    {
        "name": "Relative Strength",
        "formula": "Neg(CsRank(Div(Return($close, 5), Add(Return($close, 20), 1e-8))))",
        "category": "Momentum",
    },
    # Factor 096
    {
        "name": "Volume-Correlated Return",
        "formula": "Neg(CsRank(Cov($returns, $volume, 10)))",
        "category": "Volume",
    },
    # Factor 097
    {
        "name": "Regime Volatility Band",
        "formula": "IfElse(Greater(Std($returns, 5), Mul(1.5, Std($returns, 20))), Neg(CsRank(Return($close, 1))), Neg(CsRank(Return($close, 10))))",
        "category": "Regime-switching",
    },
    # Factor 098
    {
        "name": "Open-Close Spread Momentum",
        "formula": "Neg(CsRank(Mean(Div(Sub($close, $open), Add($open, 1e-8)), 5)))",
        "category": "Intraday",
    },
    # Factor 099
    {
        "name": "Volatility-Scaled Reversal",
        "formula": "Neg(CsRank(Div(Return($close, 5), Add(Std($returns, 20), 1e-8))))",
        "category": "Mean-reversion",
    },
    # Factor 100
    {
        "name": "VWAP Time-Weighted Signal",
        "formula": "Neg(CsRank(WMA(Div(Sub($close, $vwap), Add($vwap, 1e-8)), 20)))",
        "category": "VWAP",
    },
    # Factor 101
    {
        "name": "Covariance Structure Shift",
        "formula": "Neg(CsRank(Sub(Cov($returns, $volume, 5), Cov($returns, $volume, 20))))",
        "category": "Volume",
    },
    # Factor 102
    {
        "name": "Quadratic Regression Residual",
        "formula": "Neg(CsRank(TsLinRegResid(Square($returns), 20)))",
        "category": "Higher-moment",
    },
    # Factor 103
    {
        "name": "VWAP Mean-Reversion Strength",
        "formula": "Neg(CsRank(Mul(Div(Sub($close, $vwap), $vwap), Div($volume, Add(Mean($volume, 20), 1e-8)))))",
        "category": "VWAP",
    },
    # Factor 104
    {
        "name": "Multi-Scale Momentum",
        "formula": "Neg(CsRank(Add(Return($close, 5), Return($close, 20))))",
        "category": "Momentum",
    },
    # Factor 105
    {
        "name": "Relative High Position",
        "formula": "Neg(CsRank(Div(Sub(TsMax($high, 20), $close), Add(TsMax($high, 20), 1e-8))))",
        "category": "Mean-reversion",
    },
    # Factor 106
    {
        "name": "Turnover Volatility",
        "formula": "Neg(CsRank(Std(Div($amt, Add($volume, 1e-8)), 10)))",
        "category": "Turnover",
    },
    # Factor 107
    {
        "name": "Regime Correlation Signal",
        "formula": "IfElse(Greater(Abs(Corr($close, $volume, 10)), 0.5), Neg(CsRank(Return($close, 3))), Neg(CsRank(Return($close, 10))))",
        "category": "Regime-switching",
    },
    # Factor 108
    {
        "name": "Intraday Momentum Reversal",
        "formula": "Neg(CsRank(Div(Sub($close, $open), Add(Sub($high, $low), 1e-8))))",
        "category": "Intraday",
    },
    # Factor 109
    {
        "name": "Volume-Weighted Slope",
        "formula": "Neg(CsRank(TsLinRegSlope(Mul($returns, $volume), 10)))",
        "category": "Volume",
    },
    # Factor 110
    {
        "name": "Adaptive Range Reversal",
        "formula": "IfElse(Greater(Std($returns, 10), Mean(Std($returns, 10), 40)), Neg(CsRank(Div(Sub($close, TsMin($close, 10)), Add(Sub(TsMax($close, 10), TsMin($close, 10)), 1e-8)))), Neg(CsRank(Return($close, 5))))",
        "category": "Regime-switching",
    },
]


def import_from_paper(
    path: Optional[Union[str, Path]] = None,
) -> FactorLibrary:
    """Import the 110 factors from the paper's Appendix P.

    If *path* is given and points to a JSON file with a ``"factors"`` list,
    those entries are loaded instead of the built-in catalog.  Each entry
    must have ``"name"``, ``"formula"``, and ``"category"`` keys.

    Parameters
    ----------
    path : str or Path, optional
        Optional JSON file to load factors from.

    Returns
    -------
    FactorLibrary
        A new library pre-populated with the paper's factors. Since no
        market data is provided, signals are ``None`` and the correlation
        matrix is not computed.
    """
    if path is not None:
        path = Path(path)
        with open(path, "r") as fp:
            raw = json.load(fp)
        entries = raw if isinstance(raw, list) else raw.get("factors", [])
    else:
        entries = PAPER_FACTORS

    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)

    for i, entry in enumerate(entries):
        factor = Factor(
            id=0,  # Will be assigned by admit_factor
            name=entry["name"],
            formula=entry["formula"],
            category=entry["category"],
            ic_mean=entry.get("ic_mean", 0.0),
            icir=entry.get("icir", 0.0),
            ic_win_rate=entry.get("ic_win_rate", 0.0),
            max_correlation=entry.get("max_correlation", 0.0),
            batch_number=entry.get("batch_number", 0),
            admission_date=entry.get("admission_date", ""),
            signals=None,
        )
        library.admit_factor(factor)

    logger.info(
        "Imported %d factors from %s",
        library.size,
        path if path else "built-in paper catalog",
    )
    return library
