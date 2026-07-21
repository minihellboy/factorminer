# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 FactorMiner Team

"""Composition service for auditable factor evidence reports.

Inputs use FactorMiner's canonical ``(assets, periods)`` orientation. The
caller remains responsible for point-in-time data, universe construction,
corporate actions, and train/test freeze discipline; those properties cannot
be inferred from numerical panels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from factorminer.evaluation.evidence.inference import (
    compute_hac_mean_test,
    compute_ic_bundle,
)
from factorminer.evaluation.evidence.models import (
    INDUSTRY_EVIDENCE_VERSION,
    IndustryEvidenceConfig,
    IndustryEvidenceReport,
)
from factorminer.evaluation.evidence.portfolio import aligned_panels, compute_cost_stress
from factorminer.evaluation.evidence.risk import (
    residualize_against_risk_exposures,
)


def _coverage(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def evaluate_industry_evidence(
    factor_name: str,
    signals: np.ndarray,
    forward_returns: np.ndarray,
    *,
    config: IndustryEvidenceConfig | None = None,
    risk_exposures: np.ndarray | None = None,
    risk_weights: np.ndarray | None = None,
    exposure_names: Sequence[str] | None = None,
    family_ic_series: Mapping[str, np.ndarray] | None = None,
    n_trials: int = 1,
    pbo_performance_matrix: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    capacity_config: Any | None = None,
) -> IndustryEvidenceReport:
    """Build a factor evidence report from aligned signal/return panels.

    Optional inputs activate otherwise-unmeasurable gates:

    - ``risk_exposures``: static ``(M, K)`` or point-in-time ``(M, T, K)``
      style/industry exposures;
    - ``family_ic_series``: IC series for the complete tried family, used for
      Benjamini-Hochberg FDR with HAC p-values;
    - ``pbo_performance_matrix``: ``(trials, paths)`` CPCV performance matrix;
    - ``volume``: dollar-volume panel used by FactorMiner's square-root
      capacity estimator.

    Split/freeze construction belongs upstream and remains explicit in
    ``validation_coverage``.
    """
    cfg = config or IndustryEvidenceConfig()
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    signal_panel, return_panel = aligned_panels(signals, forward_returns)
    warnings: list[str] = []

    raw_metrics, _, rank_series = compute_ic_bundle(signal_panel, return_panel, cfg)
    raw_portfolio, raw_net_paths = compute_cost_stress(signal_panel, return_panel, cfg)

    from factorminer.evaluation.significance import (
        BootstrapICTester,
        DeflatedSharpeCalculator,
        FDRController,
        SignificanceConfig,
    )

    significance_config = SignificanceConfig(
        bootstrap_n_samples=cfg.bootstrap_n_samples,
        bootstrap_block_size=cfg.bootstrap_block_size,
        bootstrap_confidence=cfg.bootstrap_confidence,
        fdr_level=cfg.fdr_level,
        seed=cfg.seed,
    )
    bootstrap_tester = BootstrapICTester(significance_config)
    bootstrap_result = bootstrap_tester.compute_ci(factor_name, rank_series)
    primary_net = raw_net_paths[float(cfg.primary_cost_bps)]
    dsr_result = DeflatedSharpeCalculator(significance_config).compute(
        factor_name,
        primary_net,
        n_trials=n_trials,
        annualization_factor=cfg.periods_per_year,
    )
    significance: dict[str, Any] = {
        "rank_ic_block_bootstrap": asdict(bootstrap_result),
        "deflated_sharpe": {
            **asdict(dsr_result),
            "cost_bps": cfg.primary_cost_bps,
            "implementation_note": (
                "FactorMiner's DSR uses an expected-maximum null approximation "
                "parameterized by n_trials; retain the complete trial return panel "
                "for independent replication."
            ),
        },
        "fdr": None,
        "pbo": None,
    }

    if family_ic_series is not None:
        family = {
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in family_ic_series.items()
        }
        family[factor_name] = rank_series
        p_values = {
            name: compute_hac_mean_test(
                values,
                lags=cfg.hac_lags,
                confidence=cfg.hac_confidence,
            ).p_value
            for name, values in family.items()
        }
        fdr = FDRController(significance_config).apply_fdr(p_values)
        significance["fdr"] = {
            **asdict(fdr),
            "p_value_method": "two-sided Newey-West mean test",
            "factor_adjusted_p_value": fdr.adjusted_p_values[factor_name],
            "factor_significant": fdr.significant[factor_name],
        }
        warnings.append(
            "Benjamini-Hochberg is most defensible under independent or positively "
            "dependent tests; strongly dependent factor families may need BY or "
            "resampling-based control."
        )

    if pbo_performance_matrix is not None:
        from factorminer.evaluation.cross_validation import (
            ProbabilityOfBacktestOverfitting,
        )

        pbo_result = ProbabilityOfBacktestOverfitting(seed=cfg.seed).compute(pbo_performance_matrix)
        significance["pbo"] = asdict(pbo_result)

    risk_payload: dict[str, Any] | None = None
    risk_portfolio: dict[str, Any] | None = None
    risk_periods = 0
    if risk_exposures is not None:
        risk_result = residualize_against_risk_exposures(
            signal_panel,
            risk_exposures,
            weights=risk_weights,
            exposure_names=exposure_names,
        )
        risk_periods = risk_result.periods_residualized
        residual_metrics, _, _ = compute_ic_bundle(
            risk_result.residual_signals,
            return_panel,
            cfg,
        )
        risk_portfolio, _ = compute_cost_stress(
            risk_result.residual_signals,
            return_panel,
            cfg,
        )
        risk_payload = {
            "method": "point-in-time cross-sectional weighted least squares residual",
            "diagnostics": risk_result.to_dict(include_signals=False),
            "metrics": residual_metrics,
        }

    capacity_payload: dict[str, Any] | None = None
    if volume is not None:
        volume_panel = np.asarray(volume, dtype=np.float64)
        if volume_panel.shape != signal_panel.shape:
            raise ValueError(
                f"volume must have shape {signal_panel.shape}; got {volume_panel.shape}"
            )
        from factorminer.evaluation.capacity import CapacityConfig, CapacityEstimator

        resolved_capacity_config = capacity_config or CapacityConfig()
        estimator = CapacityEstimator(return_panel, volume_panel, resolved_capacity_config)
        capacity_estimate = estimator.estimate(factor_name, signal_panel)
        net_cost = estimator.net_cost_evaluation(factor_name, signal_panel)
        capacity_payload = {
            "model": "square-root market-impact scenario",
            "estimate": asdict(capacity_estimate),
            "base_capital_net_evaluation": asdict(net_cost),
        }
        warnings.append(
            "Capacity is a square-root impact scenario using long-leg liquidity as the "
            "shared two-leg proxy, not an order-book replay or live fill study."
        )

    risk_status = "not_supplied"
    risk_detail = "Supply style and one-hot industry/country exposures; no exposure was inferred."
    if risk_exposures is not None:
        risk_status = "measured" if risk_periods else "partial"
        risk_detail = (
            "Signal was neutralized against supplied point-in-time exposures."
            if risk_periods
            else "Exposures were supplied, but no period had enough valid assets to fit."
        )

    rank_periods = int(raw_metrics["rank_ic"]["n_periods"])
    pearson_periods = int(raw_metrics["pearson_ic"]["n_periods"])
    gross_periods = int(np.isfinite(raw_portfolio["gross_return_series"]).sum())
    turnover_periods = int(np.isfinite(raw_portfolio["turnover_series"]).sum())
    ic_status = "measured" if rank_periods and pearson_periods else "partial"
    serial_status = "measured" if rank_periods >= 3 else "partial"
    portfolio_status = "measured" if gross_periods and turnover_periods else "partial"

    validation_coverage = {
        "ic_rankic": _coverage(
            ic_status,
            (
                "Pearson IC and Spearman RankIC are separate, with exact series and definitions."
                if ic_status == "measured"
                else "No period had enough finite signal/return pairs for both IC definitions."
            ),
        ),
        "serial_dependence": _coverage(
            serial_status,
            (
                "Newey-West inference and a circular block-bootstrap CI are reported."
                if serial_status == "measured"
                else "At least three valid RankIC periods are required for mean inference."
            ),
        ),
        "turnover_cost_stress": _coverage(
            portfolio_status,
            (
                "Long/short target-weight turnover and configurable linear cost stress."
                if portfolio_status == "measured"
                else "Valid selected returns and at least two eligible portfolio periods are required."
            ),
        ),
        "capacity": _coverage(
            "measured" if volume is not None else "not_supplied",
            (
                "Square-root impact capacity evaluated from the supplied dollar-volume panel."
                if volume is not None
                else "Supply a point-in-time dollar-volume panel to activate capacity stress."
            ),
        ),
        "multiple_testing": _coverage(
            ("measured" if family_ic_series is not None and rank_periods >= 3 else "partial"),
            (
                "Family-wide HAC p-values receive Benjamini-Hochberg adjustment; DSR "
                "also uses the declared trial count."
                if family_ic_series is not None
                else "DSR uses n_trials, but family-wide FDR requires every tried "
                "factor's IC series."
            ),
        ),
        "selection_overfit": _coverage(
            "measured" if pbo_performance_matrix is not None else "not_supplied",
            (
                "PBO evaluated from the supplied CPCV path matrix."
                if pbo_performance_matrix is not None
                else "Supply the complete trial-by-CPCV-path matrix to estimate PBO."
            ),
        ),
        "risk_residualization": _coverage(risk_status, risk_detail),
        "walk_forward_freeze": _coverage(
            "external_required",
            "This report scores supplied panels; construct train/validation/test freezes upstream.",
        ),
        "point_in_time_data": _coverage(
            "external_required",
            "Dataset timestamps, survivorship, corporate actions, and vendor revisions "
            "need provenance.",
        ),
        "independent_execution_replay": _coverage(
            "external_required",
            "Replay frozen weights in an event-driven engine or broker simulator "
            "before deployment.",
        ),
    }
    if risk_exposures is None:
        warnings.append("Raw predictive power may be an unmeasured style or industry exposure.")
    if family_ic_series is None:
        warnings.append("A single-factor p-value does not control false discoveries.")
    if not rank_periods:
        warnings.append("No period had enough finite observations to estimate RankIC.")

    return IndustryEvidenceReport(
        protocol_version=INDUSTRY_EVIDENCE_VERSION,
        factor_name=factor_name,
        metric_contract={
            "panel_orientation": "assets_by_periods",
            "forward_return_alignment": "caller_supplied",
            "legacy_ic_alias": "spearman_rank_ic",
            "icir": "mean(IC_t) / sample_std(IC_t), unannualized",
            "annualized_icir": "icir * sqrt(periods_per_year)",
            "independent_t_stat": "icir * sqrt(n_periods)",
            "hac_t_stat": "mean(IC_t) / Newey-West standard_error(mean)",
            "periods_per_year": cfg.periods_per_year,
        },
        raw_signal=raw_metrics,
        risk_residual=risk_payload,
        portfolio={
            "raw_signal": raw_portfolio,
            "risk_residual_signal": risk_portfolio,
        },
        significance=significance,
        capacity=capacity_payload,
        validation_coverage=validation_coverage,
        warnings=tuple(warnings),
    )
