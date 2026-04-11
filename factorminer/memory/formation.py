"""Memory Formation operator F(M, tau).

Analyzes a mining trajectory tau (batch of evaluated candidates with IC,
correlation, admission results) and extracts new experience:
- Successful patterns from admitted factors
- Forbidden directions from high-correlation rejections
- Strategic insights about what works across the batch
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    MiningState,
    StrategicInsight,
    SuccessPattern,
)

# ---------------------------------------------------------------------------
# Operator-pattern detection helpers
# ---------------------------------------------------------------------------

# Maps operator substrings to pattern categories
_PATTERN_SIGNATURES: dict[str, list[str]] = {
    "Higher Moment Regimes": ["Skew", "Kurt", "IfElse"],
    "PV Corr Interaction": ["Corr", "$close", "$volume"],
    "Robust Efficiency": ["Med", "Median"],
    "Smoothed Efficiency Rank": ["EMA", "CsRank"],
    "Trend Regression Adaptive": ["Rsquare", "Slope", "Resi", "TsLinReg"],
    "Logical Or Extreme Regimes": ["Or", "Greater", "Less"],
    "Kurtosis Regime": ["Kurt", "IfElse"],
    "Amt Efficiency Rank Interaction": ["$amt", "CsRank"],
}

_FORBIDDEN_SIGNATURES: dict[str, dict[str, Any]] = {
    "Standardized Returns/Amount": {
        "keywords": ["CsZScore", "$returns", "$amt", "Std"],
        "typical_corr": 0.6,
        "reason": "Standardized return/amount variants cluster with rho > 0.6",
    },
    "VWAP Deviation variants": {
        "keywords": ["$vwap", "Delta", "Sub", "$close"],
        "typical_corr": 0.5,
        "reason": "VWAP deviation variants produce highly correlated factors (rho > 0.5)",
    },
    "Simple Delta Reversal": {
        "keywords": ["Delta", "$close", "Neg", "Return"],
        "typical_corr": 0.5,
        "reason": "Simple delta-based reversal factors are redundant (rho > 0.5)",
    },
    "WMA/EMA Smoothed Efficiency": {
        "keywords": ["WMA", "EMA", "SMA"],
        "typical_corr": 0.9,
        "reason": "WMA/EMA smoothed efficiency variants nearly identical (rho > 0.9)",
    },
}


def _extract_operators(formula: str) -> list[str]:
    """Extract operator names from a DSL formula string."""
    return re.findall(r"([A-Z][a-zA-Z]+)\(", formula)


def _extract_features(formula: str) -> list[str]:
    """Extract feature references from a DSL formula string."""
    return re.findall(r"\$[a-z]+", formula)


def _matches_pattern(formula: str, signature_keywords: list[str]) -> bool:
    """Check if a formula matches a pattern based on keyword presence."""
    formula_upper = formula.upper()
    ops = _extract_operators(formula)
    feats = _extract_features(formula)
    all_tokens = [o.upper() for o in ops] + [f.upper() for f in feats]
    match_count = sum(
        1 for kw in signature_keywords
        if any(kw.upper() in token for token in all_tokens)
        or kw.upper() in formula_upper
    )
    # Require at least 2 keyword matches (or all if fewer than 2 keywords)
    threshold = min(2, len(signature_keywords))
    return match_count >= threshold


def _classify_success_pattern(formula: str) -> str | None:
    """Try to classify a formula into a known success pattern category."""
    for pattern_name, keywords in _PATTERN_SIGNATURES.items():
        if _matches_pattern(formula, keywords):
            return pattern_name
    return None


def _classify_forbidden_direction(formula: str) -> str | None:
    """Try to classify a formula into a known forbidden direction."""
    for direction_name, info in _FORBIDDEN_SIGNATURES.items():
        if _matches_pattern(formula, info["keywords"]):
            return direction_name
    return None


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------

def _analyze_admissions(
    trajectory: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split trajectory into admitted and rejected candidates."""
    admitted = []
    rejected = []
    for candidate in trajectory:
        if candidate.get("admitted", False):
            admitted.append(candidate)
        else:
            rejected.append(candidate)
    return admitted, rejected


def _extract_success_patterns(
    admitted: list[dict],
    existing_patterns: list[SuccessPattern],
) -> list[SuccessPattern]:
    """Extract new or reinforced success patterns from admitted factors."""
    pattern_map: dict[str, SuccessPattern] = {
        p.name: SuccessPattern(
            name=p.name,
            description=p.description,
            template=p.template,
            success_rate=p.success_rate,
            example_factors=list(p.example_factors),
            occurrence_count=p.occurrence_count,
        )
        for p in existing_patterns
    }

    for candidate in admitted:
        formula = candidate.get("formula", "")
        factor_id = candidate.get("factor_id", formula[:60])
        ic = candidate.get("ic", 0.0)

        pattern_name = _classify_success_pattern(formula)
        if pattern_name is None:
            # Novel pattern: create a generic entry based on operators used
            ops = _extract_operators(formula)
            if len(ops) >= 2:
                pattern_name = f"Novel: {'+'.join(ops[:3])}"
            else:
                continue

        if pattern_name in pattern_map:
            pat = pattern_map[pattern_name]
            pat.occurrence_count += 1
            if factor_id not in pat.example_factors:
                pat.example_factors.append(factor_id)
                # Keep example list bounded
                if len(pat.example_factors) > 10:
                    pat.example_factors = pat.example_factors[-10:]
            # Upgrade success rate if consistently passing
            if pat.occurrence_count >= 5 and pat.success_rate == "Medium":
                pat.success_rate = "High"
        else:
            pattern_map[pattern_name] = SuccessPattern(
                name=pattern_name,
                description=f"Pattern derived from admitted factor with IC={ic:.4f}",
                template=formula,
                success_rate="Low",
                example_factors=[factor_id],
                occurrence_count=1,
            )

    return list(pattern_map.values())


def _extract_forbidden_directions(
    rejected: list[dict],
    existing_forbidden: list[ForbiddenDirection],
) -> list[ForbiddenDirection]:
    """Extract new or reinforced forbidden directions from rejections."""
    direction_map: dict[str, ForbiddenDirection] = {
        f.name: ForbiddenDirection(
            name=f.name,
            description=f.description,
            correlated_factors=list(f.correlated_factors),
            typical_correlation=f.typical_correlation,
            reason=f.reason,
            occurrence_count=f.occurrence_count,
        )
        for f in existing_forbidden
    }

    for candidate in rejected:
        formula = candidate.get("formula", "")
        candidate.get("factor_id", formula[:60])
        rejection_reason = candidate.get("rejection_reason", "")
        max_corr = candidate.get("max_correlation", 0.0)
        correlated_with = candidate.get("correlated_with", "")

        # Only track correlation-based rejections
        if max_corr < 0.4 and "correlation" not in rejection_reason.lower():
            continue

        direction_name = _classify_forbidden_direction(formula)
        if direction_name is None:
            # Detect generic high-correlation cluster
            if max_corr >= 0.5:
                ops = _extract_operators(formula)
                feats = _extract_features(formula)
                direction_name = f"HighCorr: {'+'.join(ops[:2])}({','.join(feats[:2])})"
            else:
                continue

        if direction_name in direction_map:
            d = direction_map[direction_name]
            d.occurrence_count += 1
            if correlated_with and correlated_with not in d.correlated_factors:
                d.correlated_factors.append(correlated_with)
                if len(d.correlated_factors) > 10:
                    d.correlated_factors = d.correlated_factors[-10:]
            # Update typical correlation as running average
            if max_corr > 0:
                d.typical_correlation = (
                    d.typical_correlation * (d.occurrence_count - 1) + max_corr
                ) / d.occurrence_count
        else:
            direction_map[direction_name] = ForbiddenDirection(
                name=direction_name,
                description=f"Rejected due to: {rejection_reason}",
                correlated_factors=[correlated_with] if correlated_with else [],
                typical_correlation=max_corr,
                reason=rejection_reason or f"High correlation (rho={max_corr:.2f})",
                occurrence_count=1,
            )

    return list(direction_map.values())


def _derive_insights(
    admitted: list[dict],
    rejected: list[dict],
    batch_number: int,
) -> list[StrategicInsight]:
    """Derive higher-level strategic insights from a batch."""
    insights: list[StrategicInsight] = []
    if not admitted and not rejected:
        return insights

    total = len(admitted) + len(rejected)
    admission_rate = len(admitted) / total if total > 0 else 0.0

    # Insight: overall batch success rate
    if total >= 5:
        if admission_rate > 0.3:
            insights.append(StrategicInsight(
                insight="Current direction is productive with high admission rate",
                evidence=f"Batch {batch_number}: {len(admitted)}/{total} admitted ({admission_rate:.0%})",
                batch_source=batch_number,
            ))
        elif admission_rate < 0.05:
            insights.append(StrategicInsight(
                insight="Current direction is exhausted, need to pivot to new operator combinations",
                evidence=f"Batch {batch_number}: only {len(admitted)}/{total} admitted ({admission_rate:.0%})",
                batch_source=batch_number,
            ))

    # Insight: operator frequency analysis
    admitted_ops = Counter()
    rejected_ops = Counter()
    for c in admitted:
        for op in _extract_operators(c.get("formula", "")):
            admitted_ops[op] += 1
    for c in rejected:
        for op in _extract_operators(c.get("formula", "")):
            rejected_ops[op] += 1

    # Find operators that appear disproportionately in admitted vs rejected
    for op, count in admitted_ops.most_common(5):
        rej_count = rejected_ops.get(op, 0)
        if count >= 3 and (rej_count == 0 or count / max(rej_count, 1) > 2.0):
            insights.append(StrategicInsight(
                insight=f"Operator '{op}' is highly productive in current search",
                evidence=f"Appeared in {count} admitted vs {rej_count} rejected factors",
                batch_source=batch_number,
            ))

    # Insight: feature analysis
    admitted_feats = Counter()
    for c in admitted:
        for feat in _extract_features(c.get("formula", "")):
            admitted_feats[feat] += 1

    if admitted_feats:
        top_feat, top_count = admitted_feats.most_common(1)[0]
        if top_count >= 3:
            insights.append(StrategicInsight(
                insight=f"Feature '{top_feat}' appears frequently in successful factors",
                evidence=f"Present in {top_count}/{len(admitted)} admitted factors",
                batch_source=batch_number,
            ))

    # Insight: non-linear vs linear
    nonlinear_ops = {"IfElse", "Skew", "Kurt", "Square", "Pow", "Log", "Or", "And"}
    admitted_nonlinear = sum(
        1 for c in admitted
        if any(op in nonlinear_ops for op in _extract_operators(c.get("formula", "")))
    )
    if len(admitted) >= 3 and admitted_nonlinear / len(admitted) > 0.6:
        insights.append(StrategicInsight(
            insight="Non-linear transformations outperform linear ones in current regime",
            evidence=f"{admitted_nonlinear}/{len(admitted)} admitted factors use non-linear operators",
            batch_source=batch_number,
        ))

    return insights


# ---------------------------------------------------------------------------
# Public API: Memory Formation
# ---------------------------------------------------------------------------

def form_memory(
    memory: ExperienceMemory,
    trajectory: list[dict],
    batch_number: int = 0,
) -> ExperienceMemory:
    """Memory Formation operator F(M, tau).

    Analyzes the mining trajectory tau and forms new experience memories
    to be merged into the existing memory via the evolution operator.

    Parameters
    ----------
    memory : ExperienceMemory
        Current memory state (used for context, not modified in place).
    trajectory : list[dict]
        Batch of evaluated candidates. Each dict should contain:
        - formula: str - the DSL formula
        - factor_id: str - unique identifier
        - ic: float - information coefficient
        - icir: float - IC information ratio
        - max_correlation: float - max correlation with existing factors
        - correlated_with: str - ID of most correlated existing factor
        - admitted: bool - whether the factor was admitted to the library
        - rejection_reason: str - reason for rejection (if rejected)
    batch_number : int
        Current batch/iteration number.

    Returns
    -------
    ExperienceMemory
        A *new* ExperienceMemory containing only the newly formed entries
        (to be merged by the evolution operator).
    """
    admitted, rejected = _analyze_admissions(trajectory)

    # Extract patterns
    new_success = _extract_success_patterns(admitted, memory.success_patterns)
    new_forbidden = _extract_forbidden_directions(rejected, memory.forbidden_directions)
    new_insights = _derive_insights(admitted, rejected, batch_number)

    # Build updated mining state
    new_state = MiningState(
        library_size=memory.state.library_size + len(admitted),
        recent_admissions=[
            {
                "factor_id": c.get("factor_id", ""),
                "formula": c.get("formula", ""),
                "ic": c.get("ic", 0.0),
                "batch": batch_number,
            }
            for c in admitted
        ],
        recent_rejections=[
            {
                "factor_id": c.get("factor_id", ""),
                "formula": c.get("formula", ""),
                "reason": c.get("rejection_reason", ""),
                "max_correlation": c.get("max_correlation", 0.0),
                "batch": batch_number,
            }
            for c in rejected[-20:]  # Keep only last 20 rejections
        ],
        domain_saturation=_compute_domain_saturation(
            memory.state.domain_saturation, admitted, rejected
        ),
        admission_log=memory.state.admission_log + [
            {
                "batch": batch_number,
                "admitted": len(admitted),
                "rejected": len(rejected),
                "admission_rate": len(admitted) / max(len(trajectory), 1),
            }
        ],
    )

    return ExperienceMemory(
        state=new_state,
        success_patterns=new_success,
        forbidden_directions=new_forbidden,
        insights=new_insights,
        version=memory.version,
    )


def _compute_domain_saturation(
    existing_saturation: dict[str, float],
    admitted: list[dict],
    rejected: list[dict],
) -> dict[str, float]:
    """Compute per-category domain saturation metrics.

    Saturation increases when many candidates in a category are rejected
    due to high correlation (the domain is "full").
    """
    saturation = dict(existing_saturation)

    # Count category attempts and rejections
    category_attempts: dict[str, int] = defaultdict(int)
    category_rejections: dict[str, int] = defaultdict(int)

    for candidate in admitted + rejected:
        formula = candidate.get("formula", "")
        category = _classify_success_pattern(formula) or "Other"
        category_attempts[category] += 1

    for candidate in rejected:
        formula = candidate.get("formula", "")
        max_corr = candidate.get("max_correlation", 0.0)
        if max_corr >= 0.4:
            category = _classify_success_pattern(formula) or "Other"
            category_rejections[category] += 1

    # Update saturation with exponential moving average
    alpha = 0.3
    for category, attempts in category_attempts.items():
        if attempts > 0:
            batch_saturation = category_rejections.get(category, 0) / attempts
            old = saturation.get(category, 0.0)
            saturation[category] = (1 - alpha) * old + alpha * batch_saturation

    return saturation
