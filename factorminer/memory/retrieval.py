"""Memory Retrieval operator R(M, L).

Context-dependent retrieval of experience memory, producing a structured
memory signal m for injection into the LLM generation prompt.

The retrieval considers the current library state (domain saturation,
recent rejections) to select the most relevant patterns and insights.
"""

from __future__ import annotations

from typing import Any

from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    MiningState,
    StrategicInsight,
    SuccessPattern,
)


def _score_success_pattern(
    pattern: SuccessPattern,
    domain_saturation: dict[str, float],
    saturated_threshold: float = 0.7,
) -> float:
    """Score a success pattern for relevance given current library state.

    Patterns in saturated domains score lower; high success-rate patterns
    with many occurrences score higher.
    """
    base_score = 1.0

    # Success rate bonus
    rate_bonus = {"High": 2.0, "Medium": 1.0, "Low": 0.5}
    base_score *= rate_bonus.get(pattern.success_rate, 1.0)

    # Occurrence count bonus (log scale to avoid runaway)
    if pattern.occurrence_count > 0:
        import math
        base_score *= 1.0 + math.log1p(pattern.occurrence_count)

    # Domain saturation penalty
    saturation = domain_saturation.get(pattern.name, 0.0)
    if saturation >= saturated_threshold:
        base_score *= 0.2  # Heavily penalize saturated domains
    elif saturation >= 0.5:
        base_score *= 0.6

    return base_score


def _score_forbidden_direction(
    direction: ForbiddenDirection,
    recent_rejection_reasons: list[str],
) -> float:
    """Score a forbidden direction for relevance.

    Directions matching recent rejection reasons score higher (more
    important to communicate to the LLM).
    """
    base_score = 1.0

    # Higher correlation = more important to avoid
    base_score *= 1.0 + direction.typical_correlation

    # Occurrence count: frequently encountered = important warning
    if direction.occurrence_count > 0:
        import math
        base_score *= 1.0 + math.log1p(direction.occurrence_count)

    # Boost if matching recent rejections
    direction_lower = direction.name.lower()
    for reason in recent_rejection_reasons:
        if any(
            word in reason.lower()
            for word in direction_lower.split()
            if len(word) > 3
        ):
            base_score *= 1.5
            break

    return base_score


def _select_relevant_success(
    patterns: list[SuccessPattern],
    domain_saturation: dict[str, float],
    max_patterns: int = 8,
) -> list[SuccessPattern]:
    """Select the most relevant success patterns for the current context."""
    if not patterns:
        return []

    scored = [
        (pat, _score_success_pattern(pat, domain_saturation))
        for pat in patterns
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pat for pat, _ in scored[:max_patterns]]


def _select_relevant_forbidden(
    directions: list[ForbiddenDirection],
    recent_rejections: list[dict],
    max_directions: int = 10,
) -> list[ForbiddenDirection]:
    """Select the most relevant forbidden directions for the current context."""
    if not directions:
        return []

    recent_reasons = [
        r.get("reason", "") for r in recent_rejections
    ]
    scored = [
        (d, _score_forbidden_direction(d, recent_reasons))
        for d in directions
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:max_directions]]


def _format_library_state(state: MiningState) -> dict[str, Any]:
    """Format mining state as structured context for LLM prompt."""
    # Identify saturated domains
    saturated = {
        domain: sat
        for domain, sat in state.domain_saturation.items()
        if sat >= 0.5
    }

    # Recent admission rate trend
    recent_logs = state.admission_log[-5:] if state.admission_log else []
    avg_rate = 0.0
    if recent_logs:
        avg_rate = sum(log.get("admission_rate", 0) for log in recent_logs) / len(recent_logs)

    return {
        "library_size": state.library_size,
        "recent_admission_rate": round(avg_rate, 3),
        "saturated_domains": saturated,
        "recent_admissions_count": len(state.recent_admissions),
        "recent_rejections_count": len(state.recent_rejections),
    }


def _format_for_prompt(
    success_patterns: list[SuccessPattern],
    forbidden_directions: list[ForbiddenDirection],
    insights: list[StrategicInsight],
    library_state: dict[str, Any],
) -> str:
    """Format the memory signal as structured text for LLM injection.

    Produces a human-readable prompt section that can be inserted into
    the factor generation prompt to guide the LLM.
    """
    sections = []

    # Library state
    sections.append("=== CURRENT LIBRARY STATE ===")
    sections.append(f"Library size: {library_state['library_size']} factors")
    sections.append(f"Recent admission rate: {library_state['recent_admission_rate']:.1%}")
    if library_state.get("saturated_domains"):
        sections.append("Saturated domains (avoid):")
        for domain, sat in library_state["saturated_domains"].items():
            sections.append(f"  - {domain}: {sat:.0%} saturated")
    sections.append("")

    # Recommended directions
    if success_patterns:
        sections.append("=== RECOMMENDED DIRECTIONS (P_succ) ===")
        for i, pat in enumerate(success_patterns, 1):
            sections.append(f"{i}. {pat.name} [{pat.success_rate}]")
            sections.append(f"   {pat.description}")
            sections.append(f"   Template: {pat.template}")
            if pat.example_factors:
                sections.append(f"   Examples: {', '.join(pat.example_factors[:3])}")
        sections.append("")

    # Forbidden directions
    if forbidden_directions:
        sections.append("=== FORBIDDEN DIRECTIONS (P_fail) ===")
        sections.append("DO NOT generate factors using these patterns:")
        for i, fd in enumerate(forbidden_directions, 1):
            sections.append(f"{i}. {fd.name} (rho > {fd.typical_correlation:.2f})")
            sections.append(f"   Reason: {fd.reason}")
            if fd.correlated_factors:
                sections.append(f"   Correlated with: {', '.join(fd.correlated_factors[:3])}")
        sections.append("")

    # Strategic insights
    if insights:
        sections.append("=== STRATEGIC INSIGHTS ===")
        for insight in insights:
            sections.append(f"- {insight.insight}")
            sections.append(f"  Evidence: {insight.evidence}")
        sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public API: Memory Retrieval
# ---------------------------------------------------------------------------

def retrieve_memory(
    memory: ExperienceMemory,
    library_state: dict[str, Any] | None = None,
    max_success: int = 8,
    max_forbidden: int = 10,
    max_insights: int = 10,
) -> dict[str, Any]:
    """Memory Retrieval operator R(M, L).

    Performs context-dependent retrieval matching against the current
    library state, returning a memory signal m suitable for LLM prompt
    injection.

    Parameters
    ----------
    memory : ExperienceMemory
        The experience memory to retrieve from.
    library_state : dict, optional
        Current library diagnostics. If None, uses the state from memory.
        Expected keys: library_size, domain_saturation, etc.
    max_success : int
        Maximum number of success patterns to include.
    max_forbidden : int
        Maximum number of forbidden directions to include.
    max_insights : int
        Maximum number of insights to include.

    Returns
    -------
    dict
        Memory signal m with keys:
        - recommended_directions: list of success pattern dicts
        - forbidden_directions: list of forbidden direction dicts
        - insights: list of insight dicts
        - library_state: dict of library state info
        - prompt_text: str - formatted text for LLM prompt injection
    """
    # Use provided library state or fall back to memory's state
    if library_state is not None:
        # Update memory state with external library info
        state = MiningState(
            library_size=library_state.get("library_size", memory.state.library_size),
            recent_admissions=memory.state.recent_admissions,
            recent_rejections=memory.state.recent_rejections,
            domain_saturation=library_state.get(
                "domain_saturation", memory.state.domain_saturation
            ),
            admission_log=memory.state.admission_log,
        )
    else:
        state = memory.state

    # Select relevant patterns
    relevant_success = _select_relevant_success(
        memory.success_patterns, state.domain_saturation, max_success
    )
    relevant_forbidden = _select_relevant_forbidden(
        memory.forbidden_directions, state.recent_rejections, max_forbidden
    )

    # Select most recent insights (up to limit)
    sorted_insights = sorted(
        memory.insights, key=lambda i: i.batch_source, reverse=True
    )
    relevant_insights = sorted_insights[:max_insights]

    # Format library state
    lib_state_info = _format_library_state(state)

    # Format as prompt text
    prompt_text = _format_for_prompt(
        relevant_success, relevant_forbidden, relevant_insights, lib_state_info
    )

    return {
        "recommended_directions": [p.to_dict() for p in relevant_success],
        "forbidden_directions": [f.to_dict() for f in relevant_forbidden],
        "insights": [i.to_dict() for i in relevant_insights],
        "library_state": lib_state_info,
        "prompt_text": prompt_text,
    }
