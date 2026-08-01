"""Composed Phase-2 validation policy for the Helix loop."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from factorminer.application.mining_budget import EvaluationResult
from factorminer.architecture.trial_ledger import canonical_formula_identity

logger = logging.getLogger(__name__)


class HelixValidationService:
    """Apply optional validation capabilities and revoke failed admissions."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loop, name)

    def validate(
        self,
        results: list[EvaluationResult],
        admitted_results: list[EvaluationResult],
    ) -> int:
        """Stage 4 extended VALIDATE: causal + regime + capacity + significance.

        Runs Phase 2 validation on admitted candidates and revokes admission
        for those that fail. Returns the number of Phase 2 rejections.
        """
        if not admitted_results:
            self._no_admission_streak += 1
            return 0

        rejected = 0

        # Collect admitted results that still have signals for extended checks
        to_check = [r for r in admitted_results if r.signals is not None]
        if not to_check:
            self._no_admission_streak = (
                0 if any(r.admitted for r in admitted_results) else self._no_admission_streak + 1
            )
            return 0

        # -- Causal validation --
        if self._causal_config is not None:
            rejected += self._validate_causal(to_check, results)

        # -- Regime validation --
        if self._regime_evaluator is not None:
            rejected += self._validate_regime(to_check, results)

        # -- Capacity validation --
        if self._capacity_estimator is not None:
            rejected += self._validate_capacity(to_check, results)

        # -- Significance testing (batch-level FDR) --
        if self._bootstrap_tester is not None and self._fdr_controller is not None:
            rejected += self._validate_significance(to_check, results)

        if rejected > 0:
            logger.info(
                "Helix: Phase 2 validation rejected %d/%d admitted candidates",
                rejected,
                len(admitted_results),
            )

        if any(r.admitted for r in admitted_results):
            self._no_admission_streak = 0
        else:
            self._no_admission_streak += 1

        return rejected

    def _validate_causal(
        self,
        to_check: list[EvaluationResult],
        all_results: list[EvaluationResult],
    ) -> int:
        """Run causal validation (Granger + intervention) on admitted candidates."""
        CausalValidatorCls = self._causal_validator
        if CausalValidatorCls is None:
            return 0

        # Collect library signals for controls
        library_signals: dict[str, np.ndarray] = {}
        for f in self.library.list_factors():
            if f.signals is not None:
                library_signals[f.name] = f.signals

        try:
            validator = CausalValidatorCls(
                returns=self.returns,
                data_tensor=self.data_tensor,
                library_signals=library_signals,
                config=self._causal_config,
            )
        except Exception as exc:
            logger.warning("Helix: causal validator creation failed: %s", exc)
            return 0

        rejected = 0
        threshold = getattr(self._causal_config, "robustness_threshold", 0.4)

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = validator.validate(r.factor_name, r.signals)
                if not result.passes:
                    self.revoke(
                        r,
                        all_results,
                        f"Causal: robustness_score={result.robustness_score:.3f} < {threshold}",
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: causal rejection for '%s' (score=%.3f)",
                        r.factor_name,
                        result.robustness_score,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: causal validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_regime(
        self,
        to_check: list[EvaluationResult],
        all_results: list[EvaluationResult],
    ) -> int:
        """Run regime-aware IC evaluation on admitted candidates."""
        if self._regime_evaluator is None:
            return 0

        rejected = 0
        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = self._regime_evaluator.evaluate(r.factor_name, r.signals)
                if not result.passes:
                    self.revoke(
                        r,
                        all_results,
                        f"Regime: only {result.n_regimes_passing} regimes passing "
                        f"(need {getattr(self._regime_config, 'min_regimes_passing', 2)})",
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: regime rejection for '%s' (%d regimes passing)",
                        r.factor_name,
                        result.n_regimes_passing,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: regime validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_capacity(
        self,
        to_check: list[EvaluationResult],
        all_results: list[EvaluationResult],
    ) -> int:
        """Run capacity-aware cost evaluation on admitted candidates."""
        if self._capacity_estimator is None:
            return 0

        rejected = 0
        net_icir_threshold = getattr(self._capacity_config, "net_icir_threshold", 0.3)

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = self._capacity_estimator.net_cost_evaluation(
                    factor_name=r.factor_name,
                    signals=r.signals,
                )
                if not result.passes_net_threshold:
                    self.revoke(
                        r,
                        all_results,
                        f"Capacity: net_icir={result.net_icir:.3f} < {net_icir_threshold}",
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: capacity rejection for '%s' (net_icir=%.3f)",
                        r.factor_name,
                        result.net_icir,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: capacity validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_significance(
        self,
        to_check: list[EvaluationResult],
        all_results: list[EvaluationResult],
    ) -> int:
        """Run bootstrap CI + batch-level FDR correction on admitted candidates."""
        if self._bootstrap_tester is None or self._fdr_controller is None:
            return 0

        # Resolve admitted candidates to canonical identities, then test them
        # against every canonical trial that has touched this dataset.
        result_map: dict[str, EvaluationResult] = {}

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                canonical_hash = canonical_formula_identity(r.formula)[1]
                result_map[canonical_hash] = r
            except Exception as exc:
                logger.warning(
                    "Helix: canonical identity error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        if not result_map:
            return 0

        try:
            family = self.trial_ledger.build_inference_family(
                dataset_id=self.trial_dataset_id,
                target_name=self.dataset_contract.default_target,
                periods=int(self.returns.shape[1]),
                estimate_effective=False,
            )
            ic_series_map = family.as_arrays()
            fdr_result = self._fdr_controller.batch_evaluate(ic_series_map, self._bootstrap_tester)
        except Exception as exc:
            reason = f"Significance: complete global trial family unavailable ({exc})"
            logger.warning("Helix: %s", reason)
            rejected = 0
            for result in result_map.values():
                if result.admitted:
                    self.revoke(result, all_results, reason)
                    rejected += 1
            return rejected

        rejected = 0
        for canonical_hash, result in result_map.items():
            is_sig = fdr_result.significant.get(canonical_hash, False)
            if not is_sig:
                if result.admitted:
                    adj_p = fdr_result.adjusted_p_values.get(canonical_hash, 1.0)
                    self.revoke(
                        result,
                        all_results,
                        f"Significance: FDR-adjusted p={adj_p:.4f} > "
                        f"{getattr(self._significance_config, 'fdr_level', 0.05)}",
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: significance rejection for '%s' (adj_p=%.4f)",
                        result.factor_name,
                        adj_p,
                    )

        if bool(getattr(self._significance_config, "deflated_sharpe_enabled", False)):
            from factorminer.evaluation.portfolio import PortfolioBacktester
            from factorminer.evaluation.significance import DeflatedSharpeCalculator

            backtester = PortfolioBacktester()
            dsr = DeflatedSharpeCalculator(self._significance_config)
            for result in result_map.values():
                if not result.admitted or result.signals is None:
                    continue
                try:
                    portfolio = backtester.quintile_backtest(
                        result.signals.T,
                        self.returns.T,
                    )
                    ls_returns = np.asarray(portfolio["ls_net_series"], dtype=np.float64)
                    dsr_result = dsr.compute(
                        result.factor_name,
                        ls_returns,
                        n_trials=family.dsr_trial_count,
                    )
                except Exception as exc:
                    self.revoke(
                        result,
                        all_results,
                        f"Significance: ledger-fed DSR unavailable ({exc})",
                    )
                    rejected += 1
                    continue
                if not dsr_result.passes:
                    self.revoke(
                        result,
                        all_results,
                        "Significance: ledger-fed DSR failed "
                        f"(DSR={dsr_result.deflated_sharpe:.3f}, "
                        f"p={dsr_result.p_value:.4f}, "
                        f"effective_trials={family.dsr_trial_count:.3f}, "
                        f"raw_trials={family.raw_trial_count})",
                    )
                    rejected += 1

        return rejected

    def revoke(
        self,
        result: EvaluationResult,
        all_results: list[EvaluationResult],
        reason: str,
    ) -> None:
        """Revoke a previously admitted candidate from the library.

        Updates the EvaluationResult and removes the factor from the library.
        """
        result.admitted = False
        result.rejection_reason = reason

        # Find and remove from library by name+formula match
        try:
            for factor in self.library.list_factors():
                if factor.name == result.factor_name and factor.formula == result.formula:
                    self.library.remove_factor(factor.id)
                    self._remove_semantic_artifacts(result.factor_name)
                    logger.debug(
                        "Helix: revoked factor '%s' (id=%d): %s",
                        result.factor_name,
                        factor.id,
                        reason,
                    )
                    return
        except Exception as exc:
            logger.warning(
                "Helix: failed to revoke factor '%s': %s",
                result.factor_name,
                exc,
            )

        self._remove_semantic_artifacts(result.factor_name)

    # ------------------------------------------------------------------
