"""Factor IC decay / half-life analysis for FactorMiner.

Admitted factors typically start with a promising in-sample IC that erodes
as the underlying anomaly gets arbitraged away or the market regime shifts.
This module turns a per-iteration IC history into a compact decay curve --
an estimated half-life, a linear trend, and a coarse classification -- so
that mining sessions and reports can flag factors that are quietly losing
their edge.

Reuses :func:`factorminer.evaluation.backtest.compute_ic_stats` for the
windowed mean/std bookkeeping rather than reimplementing it; only the trend
and half-life estimation are new logic specific to decay analysis.

Hyperbolic crowding extension (Lee 2025 / landscape §10 item 4):
    Alongside the linear half-life fit, :func:`fit_hyperbolic_decay` estimates
    ``α(t) = K / (1 + λ t)`` on an IC series. Combined with
    ``architecture.families.mechanism_family``, this feeds a *research risk
    label* (not a mining objective or trade-timing signal) that flags
    mechanical Trend/Momentum and Reversal families as more crowding-
    vulnerable than judgment-like families. Crowding-based timing does not
    beat buy-and-hold on the mean in the source literature -- treat the
    label as a risk annotation only.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from factorminer.evaluation.backtest import compute_ic_stats

if TYPE_CHECKING:  # pragma: no cover - typing only
    from factorminer.architecture.lifecycle import FactorLifecycleStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (documented here so callers/tests can reference the same
# constants instead of hard-coding magic numbers).
# ---------------------------------------------------------------------------

#: Minimum number of finite IC observations required to fit a decay trend.
#: Below this, a linear fit is too noisy to trust, so the curve is reported
#: as 'insufficient_data' instead of a spurious classification.
MIN_OBSERVATIONS_FOR_TREND = 4

#: Number of observations averaged at each end of the series (via
#: ``compute_ic_stats``) to smooth single-iteration noise when estimating
#: ``ic_at_admission`` / ``ic_current``. Capped at ``n_finite // 2`` so the
#: two windows never overlap.
EDGE_WINDOW = 3

#: ``|ic_current| / |ic_at_admission|`` at or above this ratio -- or a
#: non-negative trend slope -- is classified 'stable': the factor's edge has
#: not meaningfully eroded.
STABLE_RATIO = 0.8

#: ``|ic_current| / |ic_at_admission|`` at or below this ratio (or a sign
#: flip from positive-at-admission to non-positive-now) is classified
#: 'decayed': the factor has lost essentially all of its edge.
DECAYED_RATIO = 0.2

#: Mechanism families treated as more crowding-vulnerable under the
#: hyperbolic taxonomy (mechanical / widely-arbitraged styles).
CROWDING_VULNERABLE_FAMILIES: frozenset[str] = frozenset(
    {
        "Trend/Momentum",
        "Reversal/Mean-Reversion",
    }
)


@dataclass
class DecayCurveResult:
    """Summary of how one factor's IC has evolved since admission.

    Attributes
    ----------
    factor_id : str
        Identifier of the factor this curve describes; empty string if the
        caller did not supply one.
    ic_at_admission : float
        Smoothed IC near the start of the observed series.
    ic_current : float
        Smoothed IC near the end of the observed series.
    half_life_iterations : float | None
        Estimated number of iterations for ``|IC|`` to fall to half of
        ``ic_at_admission``, assuming the fitted linear trend continues.
        ``None`` when the trend is flat/increasing or ``ic_at_admission`` is
        ~0 (no meaningful half-life exists).
    trend_slope : float
        Slope (IC per iteration) of an ordinary-least-squares fit of IC
        against iteration index.
    classification : str
        One of ``'stable'``, ``'decaying'``, ``'decayed'``,
        ``'insufficient_data'``. See module docstring / constants above for
        the exact thresholds.
    """

    factor_id: str
    ic_at_admission: float
    ic_current: float
    half_life_iterations: float | None
    trend_slope: float
    classification: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation."""
        return {
            "factor_id": self.factor_id,
            "ic_at_admission": self.ic_at_admission,
            "ic_current": self.ic_current,
            "half_life_iterations": self.half_life_iterations,
            "trend_slope": self.trend_slope,
            "classification": self.classification,
        }


@dataclass(frozen=True)
class HyperbolicDecayFit:
    """Hyperbolic IC decay fit ``α(t) = K / (1 + λ t)``.

    Attributes
    ----------
    k : float
        Level parameter (estimated |IC| at t=0).
    lambda_ : float
        Decay-rate parameter (≥ 0). Larger λ → faster decay.
    half_life : float | None
        Time at which α falls to K/2, i.e. ``t = 1/λ``. ``None`` when λ≈0
        (no meaningful hyperbolic half-life).
    r_squared : float
        Coefficient of determination of the fit against |IC(t)|.
    n_obs : int
        Number of finite observations used.
    """

    k: float
    lambda_: float
    half_life: float | None
    r_squared: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "lambda": self.lambda_,
            "half_life": self.half_life,
            "r_squared": self.r_squared,
            "n_obs": self.n_obs,
        }


@dataclass(frozen=True)
class CrowdingDecayTaxonomy:
    """Research risk label from hyperbolic decay + mechanism family.

    This is explicitly a **research risk annotation**, not a mining objective
    or trade-timing signal. Source papers show crowding-based timing does not
    beat buy-and-hold on the mean.
    """

    mechanism_family: str
    fine_family: str
    hyperbolic: HyperbolicDecayFit
    crowding_vulnerable: bool
    risk_label: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_family": self.mechanism_family,
            "fine_family": self.fine_family,
            "hyperbolic": self.hyperbolic.to_dict(),
            "crowding_vulnerable": self.crowding_vulnerable,
            "risk_label": self.risk_label,
            "rationale": self.rationale,
        }


def compute_factor_decay_curve(
    ic_by_iteration: Sequence[float],
    admission_iteration: int,
    *,
    factor_id: str = "",
) -> DecayCurveResult:
    """Estimate a factor's IC decay curve from a per-iteration IC series.

    Parameters
    ----------
    ic_by_iteration : Sequence[float]
        IC observations for one factor, one entry per iteration since (and
        including) admission, in iteration order. May contain NaN for
        iterations where an IC could not be computed.
    admission_iteration : int
        Loop iteration number of the first observation. The decay math only
        depends on the series' own relative order (shifting the x-axis by a
        constant does not change a linear slope), so this value is used for
        labeling/logging by callers, not for the fit itself.
    factor_id : str, optional
        Identifier stamped onto the returned result.

    Returns
    -------
    DecayCurveResult
    """
    series = np.asarray(ic_by_iteration, dtype=float)
    finite_mask = np.isfinite(series)
    n_finite = int(finite_mask.sum())

    if n_finite < MIN_OBSERVATIONS_FOR_TREND:
        finite_vals = series[finite_mask]
        ic_at_admission = float(finite_vals[0]) if finite_vals.size else 0.0
        ic_current = float(finite_vals[-1]) if finite_vals.size else 0.0
        logger.debug(
            "decay curve for factor_id=%s admission_iteration=%d: only %d finite "
            "observations (< %d), reporting insufficient_data",
            factor_id, admission_iteration, n_finite, MIN_OBSERVATIONS_FOR_TREND,
        )
        return DecayCurveResult(
            factor_id=factor_id,
            ic_at_admission=ic_at_admission,
            ic_current=ic_current,
            half_life_iterations=None,
            trend_slope=0.0,
            classification="insufficient_data",
        )

    finite_idx = np.flatnonzero(finite_mask)
    edge = min(EDGE_WINDOW, n_finite // 2)
    admission_window = series[finite_idx[:edge]]
    current_window = series[finite_idx[-edge:]]
    ic_at_admission = compute_ic_stats(admission_window)["ic_mean"]
    ic_current = compute_ic_stats(current_window)["ic_mean"]

    x = finite_idx.astype(float)
    y = series[finite_idx]
    trend_slope = float(np.polyfit(x, y, 1)[0])

    half_life_iterations: float | None = None
    if trend_slope < -1e-9 and abs(ic_at_admission) > 1e-9:
        half_life_iterations = abs(ic_at_admission) / (2.0 * abs(trend_slope))

    if abs(ic_at_admission) > 1e-9:
        decay_ratio = abs(ic_current) / abs(ic_at_admission)
    else:
        decay_ratio = 1.0 if abs(ic_current) <= 1e-9 else 0.0

    sign_flip = ic_at_admission > 0 and ic_current <= 0

    if trend_slope >= 0 or decay_ratio >= STABLE_RATIO:
        classification = "stable"
    elif decay_ratio <= DECAYED_RATIO or sign_flip:
        classification = "decayed"
    else:
        classification = "decaying"

    return DecayCurveResult(
        factor_id=factor_id,
        ic_at_admission=float(ic_at_admission),
        ic_current=float(ic_current),
        half_life_iterations=half_life_iterations,
        trend_slope=trend_slope,
        classification=classification,
    )


def fit_hyperbolic_decay(
    ic_by_iteration: Sequence[float],
    *,
    min_obs: int = MIN_OBSERVATIONS_FOR_TREND,
) -> HyperbolicDecayFit:
    """Fit ``α(t) = K / (1 + λ t)`` to the absolute IC series.

    Uses a linearized OLS form on ``1/|IC|`` vs ``t`` when |IC| stays
    strictly positive, with a non-negative λ clamp. Falls closed to a flat
    (λ=0) fit when data are insufficient or degenerate -- never invents a
    half-life from noise.
    """
    series = np.asarray(ic_by_iteration, dtype=float)
    finite_mask = np.isfinite(series)
    n_finite = int(finite_mask.sum())
    if n_finite < min_obs:
        return HyperbolicDecayFit(
            k=0.0, lambda_=0.0, half_life=None, r_squared=0.0, n_obs=n_finite
        )

    t = np.flatnonzero(finite_mask).astype(float)
    y = np.abs(series[finite_mask])
    # Drop near-zero |IC| points that would blow up 1/y linearization.
    positive = y > 1e-9
    if int(positive.sum()) < min_obs:
        k0 = float(np.mean(y)) if y.size else 0.0
        return HyperbolicDecayFit(
            k=k0, lambda_=0.0, half_life=None, r_squared=0.0, n_obs=n_finite
        )

    t_pos = t[positive]
    y_pos = y[positive]
    inv_y = 1.0 / y_pos
    # inv_y ≈ (1/K) + (λ/K) t  →  slope = λ/K, intercept = 1/K
    slope, intercept = np.polyfit(t_pos, inv_y, 1)
    if intercept <= 1e-12:
        k = float(np.mean(y_pos))
        lambda_ = 0.0
    else:
        k = float(1.0 / intercept)
        lambda_ = float(max(0.0, slope * k))

    # Goodness of fit on the original scale.
    pred = k / (1.0 + lambda_ * t_pos)
    ss_res = float(np.sum((y_pos - pred) ** 2))
    ss_tot = float(np.sum((y_pos - np.mean(y_pos)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    r_squared = float(max(0.0, min(1.0, r_squared)))

    half_life = (1.0 / lambda_) if lambda_ > 1e-12 else None
    return HyperbolicDecayFit(
        k=k,
        lambda_=lambda_,
        half_life=half_life,
        r_squared=r_squared,
        n_obs=n_finite,
    )


def classify_crowding_decay_risk(
    ic_by_iteration: Sequence[float],
    *,
    formula: str = "",
    fine_family: str | None = None,
    mechanism: str | None = None,
) -> CrowdingDecayTaxonomy:
    """Build a crowding-vulnerability research risk label.

    Mechanical families (Trend/Momentum, Reversal/Mean-Reversion) are flagged
    as more crowding-vulnerable than judgment-like families. Fast hyperbolic
    decay (high λ / short half-life) raises the risk tier within that frame.

    Parameters
    ----------
    ic_by_iteration:
        Per-iteration IC series (may contain NaN).
    formula:
        Optional formula string used to infer family when ``fine_family`` is
        not supplied.
    fine_family / mechanism:
        Optional pre-computed family labels; when omitted they are inferred
        from ``formula`` via ``infer_family`` / ``mechanism_family``.
    """
    from factorminer.architecture.families import infer_family, mechanism_family

    resolved_fine = fine_family or (infer_family(formula) if formula else "Other")
    resolved_mech = mechanism or mechanism_family(resolved_fine)
    hyp = fit_hyperbolic_decay(ic_by_iteration)
    vulnerable = resolved_mech in CROWDING_VULNERABLE_FAMILIES

    # Risk tiers: research labels only.
    fast_decay = hyp.half_life is not None and hyp.half_life < 20.0
    if vulnerable and fast_decay:
        risk_label = "high_crowding_decay_risk"
        rationale = (
            f"Mechanical family '{resolved_mech}' with fast hyperbolic decay "
            f"(λ={hyp.lambda_:.4f}, half_life={hyp.half_life:.1f}). "
            "Research risk label only — not a trade timer."
        )
    elif vulnerable:
        risk_label = "elevated_crowding_decay_risk"
        rationale = (
            f"Mechanical family '{resolved_mech}' is crowding-vulnerable by "
            "taxonomy even without a short measured half-life. "
            "Research risk label only — not a trade timer."
        )
    elif fast_decay:
        risk_label = "decay_watch"
        rationale = (
            f"Non-mechanical family '{resolved_mech}' but hyperbolic half-life "
            f"is short ({hyp.half_life:.1f}). Monitor; not auto-crowding."
        )
    else:
        risk_label = "low_crowding_decay_risk"
        rationale = (
            f"Family '{resolved_mech}' is not in the mechanical crowding set "
            "and hyperbolic decay is not fast."
        )

    return CrowdingDecayTaxonomy(
        mechanism_family=resolved_mech,
        fine_family=resolved_fine,
        hyperbolic=hyp,
        crowding_vulnerable=vulnerable,
        risk_label=risk_label,
        rationale=rationale,
    )


def build_decay_report(store: FactorLifecycleStore) -> list[dict[str, Any]]:
    """Build decay curves for every factor with an IC history in a
    :class:`~factorminer.architecture.lifecycle.FactorLifecycleStore`.

    Groups lifecycle events carrying an ``ic_mean`` detail (recorded at the
    ``fast_screened``/``admitted``/``replaced`` stages) by
    ``(factor_name, formula)``, orders them by iteration, and runs
    :func:`compute_factor_decay_curve` per factor.

    Parameters
    ----------
    store : FactorLifecycleStore

    Returns
    -------
    list of dict
        JSON-safe rows: ``factor_id``, ``formula``, ``ic_at_admission``,
        ``ic_current``, ``half_life_iterations``, ``trend_slope``,
        ``classification``, ``observations``. Sorted by ``factor_id``.
    """
    series_by_factor: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for event in store.events:
        ic_mean = event.details.get("ic_mean")
        if ic_mean is None:
            continue
        series_by_factor[(event.factor_name, event.formula)].append(
            (event.iteration, float(ic_mean))
        )

    rows: list[dict[str, Any]] = []
    for (factor_name, formula), points in series_by_factor.items():
        points.sort(key=lambda p: p[0])
        ics = [ic for _, ic in points]
        admission_iteration = points[0][0]
        curve = compute_factor_decay_curve(ics, admission_iteration, factor_id=factor_name)
        row = curve.to_dict()
        row["formula"] = formula
        row["observations"] = len(ics)
        # Additive hyperbolic / crowding taxonomy (research risk label).
        taxonomy = classify_crowding_decay_risk(ics, formula=formula)
        row["hyperbolic"] = taxonomy.hyperbolic.to_dict()
        row["crowding_risk_label"] = taxonomy.risk_label
        row["mechanism_family"] = taxonomy.mechanism_family
        rows.append(row)

    rows.sort(key=lambda row: row["factor_id"])
    return rows
