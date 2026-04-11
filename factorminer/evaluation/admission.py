"""Admission rules for the factor library.

Implements the decision logic for whether a candidate factor should be
admitted to the library, replace an existing factor, or be rejected.

Admission Rule (Eq. 10):
    Admit alpha if |IC(alpha)| >= tau_IC  AND  max_{g in L} |rho(alpha, g)| < theta

Replacement Rule (Eq. 11):
    Replace g with alpha if:
        |IC(alpha)| >= 0.10  AND
        |IC(alpha)| >= 1.3 * |IC(g)|  AND
        |{g in L : |rho(alpha, g)| >= theta}| == 1
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdmissionDecision:
    """Result of an admission check for a candidate factor."""

    admitted: bool
    replaced_factor_id: str | None = None
    rejection_reason: str | None = None
    max_correlation: float = 0.0
    correlated_with: str | None = None
    decision_type: str = "rejected"  # "admitted", "replacement", "rejected"


def check_admission(
    ic_abs: float,
    max_corr: float,
    correlated_with: str | None,
    ic_threshold: float = 0.04,
    correlation_threshold: float = 0.5,
) -> AdmissionDecision:
    """Standard admission check (Eq. 10).

    Parameters
    ----------
    ic_abs : float
        Absolute IC of the candidate.
    max_corr : float
        Maximum absolute correlation with any library factor.
    correlated_with : str or None
        ID of the most correlated library factor.
    ic_threshold : float
        Minimum |IC| for admission (tau_IC).
    correlation_threshold : float
        Maximum allowed correlation (theta).

    Returns
    -------
    AdmissionDecision
    """
    if ic_abs < ic_threshold:
        return AdmissionDecision(
            admitted=False,
            rejection_reason=f"IC too low: |IC|={ic_abs:.4f} < {ic_threshold}",
            max_correlation=max_corr,
            correlated_with=correlated_with,
            decision_type="rejected",
        )

    if max_corr >= correlation_threshold:
        return AdmissionDecision(
            admitted=False,
            rejection_reason=(
                f"Too correlated: max|rho|={max_corr:.4f} >= {correlation_threshold} "
                f"(with {correlated_with})"
            ),
            max_correlation=max_corr,
            correlated_with=correlated_with,
            decision_type="rejected",
        )

    return AdmissionDecision(
        admitted=True,
        max_correlation=max_corr,
        correlated_with=correlated_with,
        decision_type="admitted",
    )


def check_replacement(
    candidate_ic_abs: float,
    max_corr: float,
    correlated_with: str | None,
    library_ic_map: dict[str, float],
    correlation_map: dict[str, float],
    replacement_ic_min: float = 0.10,
    replacement_ic_ratio: float = 1.3,
    correlation_threshold: float = 0.5,
) -> AdmissionDecision:
    """Replacement admission check (Eq. 11).

    A candidate that failed the standard correlation check may still
    replace an existing library factor if it is sufficiently stronger
    and only conflicts with exactly one factor.

    Parameters
    ----------
    candidate_ic_abs : float
        Absolute IC of the candidate.
    max_corr : float
        Max absolute correlation with any library factor.
    correlated_with : str or None
        ID of the most correlated library factor.
    library_ic_map : dict
        Mapping from library factor ID to its absolute IC.
    correlation_map : dict
        Mapping from library factor ID to correlation with the candidate.
    replacement_ic_min : float
        Minimum |IC| for replacement consideration.
    replacement_ic_ratio : float
        Required ratio IC(candidate) / IC(existing).
    correlation_threshold : float
        Correlation threshold (theta) for determining conflicts.

    Returns
    -------
    AdmissionDecision
    """
    # Must meet minimum IC for replacement
    if candidate_ic_abs < replacement_ic_min:
        return AdmissionDecision(
            admitted=False,
            rejection_reason=(
                f"IC too low for replacement: |IC|={candidate_ic_abs:.4f} < {replacement_ic_min}"
            ),
            max_correlation=max_corr,
            correlated_with=correlated_with,
            decision_type="rejected",
        )

    # Find all factors above the correlation threshold
    conflicting: list[str] = [
        fid for fid, corr in correlation_map.items()
        if abs(corr) >= correlation_threshold
    ]

    # Must conflict with exactly one factor
    if len(conflicting) != 1:
        return AdmissionDecision(
            admitted=False,
            rejection_reason=(
                f"Replacement requires exactly 1 correlated factor, found {len(conflicting)}"
            ),
            max_correlation=max_corr,
            correlated_with=correlated_with,
            decision_type="rejected",
        )

    target_id = conflicting[0]
    target_ic = library_ic_map.get(target_id, 0.0)

    # Candidate must be sufficiently stronger
    if target_ic > 0 and candidate_ic_abs < replacement_ic_ratio * target_ic:
        return AdmissionDecision(
            admitted=False,
            rejection_reason=(
                f"Not strong enough to replace {target_id}: "
                f"|IC|={candidate_ic_abs:.4f} < {replacement_ic_ratio} * {target_ic:.4f}"
            ),
            max_correlation=max_corr,
            correlated_with=correlated_with,
            decision_type="rejected",
        )

    return AdmissionDecision(
        admitted=True,
        replaced_factor_id=target_id,
        max_correlation=max_corr,
        correlated_with=correlated_with,
        decision_type="replacement",
    )


# ---------------------------------------------------------------------------
# Stock-level thresholds (configurable)
# ---------------------------------------------------------------------------

@dataclass
class StockThresholds:
    """Default thresholds for A-share stock factor evaluation."""

    ic_abs_min: float = 0.05
    icir_abs_min: float = 0.5
    ic_win_rate_min: float = 0.50
    max_turnover: float = 0.8
    min_monotonicity: float = 0.0

    def passes(
        self,
        ic_abs: float,
        icir_abs: float,
        ic_win_rate: float = 1.0,
        turnover: float = 0.0,
        monotonicity: float = 1.0,
    ) -> tuple[bool, str | None]:
        """Check if a factor meets all stock-level thresholds.

        Returns
        -------
        tuple of (passes, rejection_reason)
        """
        if ic_abs < self.ic_abs_min:
            return False, f"|IC|={ic_abs:.4f} < {self.ic_abs_min}"
        if icir_abs < self.icir_abs_min:
            return False, f"|ICIR|={icir_abs:.4f} < {self.icir_abs_min}"
        if ic_win_rate < self.ic_win_rate_min:
            return False, f"IC win rate={ic_win_rate:.4f} < {self.ic_win_rate_min}"
        if turnover > self.max_turnover:
            return False, f"Turnover={turnover:.4f} > {self.max_turnover}"
        if monotonicity < self.min_monotonicity:
            return False, f"Monotonicity={monotonicity:.4f} < {self.min_monotonicity}"
        return True, None
