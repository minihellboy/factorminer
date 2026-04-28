"""Memory Evolution operator E(M, M_form).

Consolidates newly formed experience into the existing memory:
- Merges redundant success/failure patterns
- Discards low-utility entries
- Reclassifies patterns that have changed behavior
- Caps memory size according to configuration limits
"""

from __future__ import annotations

from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    StrategicInsight,
    SuccessPattern,
)


def _merge_success_patterns(
    existing: list[SuccessPattern],
    new: list[SuccessPattern],
) -> list[SuccessPattern]:
    """Merge new success patterns into existing ones.

    Patterns with the same name are consolidated by combining examples
    and updating occurrence counts. Novel patterns are appended.
    """
    merged: dict[str, SuccessPattern] = {}

    for pat in existing:
        merged[pat.name] = SuccessPattern(
            name=pat.name,
            description=pat.description,
            template=pat.template,
            success_rate=pat.success_rate,
            example_factors=list(pat.example_factors),
            occurrence_count=pat.occurrence_count,
            confidence=getattr(pat, "confidence", 1.0),
        )

    for pat in new:
        if pat.name in merged:
            existing_pat = merged[pat.name]
            existing_pat.occurrence_count += pat.occurrence_count
            # Merge example factors, dedup
            seen = set(existing_pat.example_factors)
            for ex in pat.example_factors:
                if ex not in seen:
                    existing_pat.example_factors.append(ex)
                    seen.add(ex)
            # Cap examples
            if len(existing_pat.example_factors) > 10:
                existing_pat.example_factors = existing_pat.example_factors[-10:]
            # Update description if new one is more informative
            if len(pat.description) > len(existing_pat.description):
                existing_pat.description = pat.description
            # Promote success rate based on accumulated evidence
            if existing_pat.occurrence_count >= 10:
                existing_pat.success_rate = "High"
            elif existing_pat.occurrence_count >= 5:
                existing_pat.success_rate = "Medium"
        else:
            merged[pat.name] = SuccessPattern(
                name=pat.name,
                description=pat.description,
                template=pat.template,
                success_rate=pat.success_rate,
                example_factors=list(pat.example_factors),
                occurrence_count=pat.occurrence_count,
                confidence=getattr(pat, "confidence", 1.0),
            )

    return list(merged.values())


def _merge_forbidden_directions(
    existing: list[ForbiddenDirection],
    new: list[ForbiddenDirection],
) -> list[ForbiddenDirection]:
    """Merge new forbidden directions into existing ones."""
    merged: dict[str, ForbiddenDirection] = {}

    for fd in existing:
        merged[fd.name] = ForbiddenDirection(
            name=fd.name,
            description=fd.description,
            correlated_factors=list(fd.correlated_factors),
            typical_correlation=fd.typical_correlation,
            reason=fd.reason,
            occurrence_count=fd.occurrence_count,
        )

    for fd in new:
        if fd.name in merged:
            existing_fd = merged[fd.name]
            existing_fd.occurrence_count += fd.occurrence_count
            # Merge correlated factors
            seen = set(existing_fd.correlated_factors)
            for cf in fd.correlated_factors:
                if cf not in seen:
                    existing_fd.correlated_factors.append(cf)
                    seen.add(cf)
            if len(existing_fd.correlated_factors) > 10:
                existing_fd.correlated_factors = existing_fd.correlated_factors[-10:]
            # Update correlation as weighted average
            total_count = existing_fd.occurrence_count
            if total_count > 0 and fd.typical_correlation > 0:
                old_weight = (total_count - fd.occurrence_count) / total_count
                new_weight = fd.occurrence_count / total_count
                existing_fd.typical_correlation = (
                    old_weight * existing_fd.typical_correlation
                    + new_weight * fd.typical_correlation
                )
            if len(fd.reason) > len(existing_fd.reason):
                existing_fd.reason = fd.reason
        else:
            merged[fd.name] = ForbiddenDirection(
                name=fd.name,
                description=fd.description,
                correlated_factors=list(fd.correlated_factors),
                typical_correlation=fd.typical_correlation,
                reason=fd.reason,
                occurrence_count=fd.occurrence_count,
            )

    return list(merged.values())


def _merge_insights(
    existing: list[StrategicInsight],
    new: list[StrategicInsight],
) -> list[StrategicInsight]:
    """Merge new insights into existing, deduplicating similar ones.

    Insights with substantially overlapping text are consolidated.
    """
    merged: list[StrategicInsight] = list(existing)

    for new_insight in new:
        is_duplicate = False
        new_lower = new_insight.insight.lower()
        for i, existing_insight in enumerate(merged):
            existing_lower = existing_insight.insight.lower()
            # Simple similarity: check if core words overlap significantly
            new_words = set(new_lower.split())
            existing_words = set(existing_lower.split())
            if len(new_words) > 0 and len(existing_words) > 0:
                overlap = len(new_words & existing_words)
                max_len = max(len(new_words), len(existing_words))
                if overlap / max_len > 0.6:
                    # Keep the one from the more recent batch
                    if new_insight.batch_source > existing_insight.batch_source:
                        merged[i] = new_insight
                    is_duplicate = True
                    break
        if not is_duplicate:
            merged.append(new_insight)

    return merged


def _reclassify_patterns(
    success_patterns: list[SuccessPattern],
    forbidden_directions: list[ForbiddenDirection],
    failure_threshold: int = 5,
) -> tuple[list[SuccessPattern], list[ForbiddenDirection]]:
    """Reclassify patterns that have changed behavior.

    If a success pattern consistently appears in forbidden directions
    (e.g., VWAP variant with rho=0.82), move it from success to forbidden.
    """
    {fd.name for fd in forbidden_directions}

    remaining_success: list[SuccessPattern] = []
    new_forbidden: list[ForbiddenDirection] = []

    for pat in success_patterns:
        # Check if this pattern name overlaps with forbidden directions
        should_reclassify = False
        matching_forbidden: ForbiddenDirection | None = None

        for fd in forbidden_directions:
            # Check for name overlap or keyword overlap
            if _names_overlap(pat.name, fd.name):
                if fd.occurrence_count >= failure_threshold:
                    should_reclassify = True
                    matching_forbidden = fd
                    break

        if should_reclassify and matching_forbidden is not None:
            # Demote: success -> forbidden
            new_forbidden.append(ForbiddenDirection(
                name=pat.name,
                description=f"Reclassified from success: {pat.description}",
                correlated_factors=matching_forbidden.correlated_factors,
                typical_correlation=matching_forbidden.typical_correlation,
                reason=f"Initially promising but consistently produces correlated factors "
                       f"(rho={matching_forbidden.typical_correlation:.2f})",
                occurrence_count=matching_forbidden.occurrence_count,
            ))
        else:
            remaining_success.append(pat)

    all_forbidden = forbidden_directions + new_forbidden
    return remaining_success, all_forbidden


def _names_overlap(name_a: str, name_b: str) -> bool:
    """Check if two pattern names refer to the same concept."""
    a_words = set(name_a.lower().replace("/", " ").replace("_", " ").split())
    b_words = set(name_b.lower().replace("/", " ").replace("_", " ").split())
    # Remove common filler words
    filler = {"the", "a", "an", "of", "in", "with", "for", "and", "or"}
    a_words -= filler
    b_words -= filler
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    return overlap >= min(2, min(len(a_words), len(b_words)))


def _prune_low_utility(
    success_patterns: list[SuccessPattern],
    forbidden_directions: list[ForbiddenDirection],
    insights: list[StrategicInsight],
    min_occurrences: int = 1,
) -> tuple[list[SuccessPattern], list[ForbiddenDirection], list[StrategicInsight]]:
    """Remove entries with too few occurrences to be reliable.

    Initial knowledge base entries (occurrence_count=0) are preserved.
    """
    pruned_success = [
        p for p in success_patterns
        if p.occurrence_count >= min_occurrences or p.occurrence_count == 0
    ]
    pruned_forbidden = [
        f for f in forbidden_directions
        if f.occurrence_count >= min_occurrences or f.occurrence_count == 0
    ]
    # Insights are lightweight, keep all
    return pruned_success, pruned_forbidden, insights


def _cap_memory_size(
    success_patterns: list[SuccessPattern],
    forbidden_directions: list[ForbiddenDirection],
    insights: list[StrategicInsight],
    max_success: int = 50,
    max_forbidden: int = 100,
    max_insights: int = 30,
) -> tuple[list[SuccessPattern], list[ForbiddenDirection], list[StrategicInsight]]:
    """Enforce maximum memory sizes by keeping the most useful entries."""
    # Sort success patterns by occurrence count (most useful first)
    if len(success_patterns) > max_success:
        success_patterns = sorted(
            success_patterns, key=lambda p: p.occurrence_count, reverse=True
        )[:max_success]

    # Sort forbidden directions by occurrence count
    if len(forbidden_directions) > max_forbidden:
        forbidden_directions = sorted(
            forbidden_directions, key=lambda f: f.occurrence_count, reverse=True
        )[:max_forbidden]

    # Keep most recent insights
    if len(insights) > max_insights:
        insights = sorted(
            insights, key=lambda i: i.batch_source, reverse=True
        )[:max_insights]

    return success_patterns, forbidden_directions, insights


# ---------------------------------------------------------------------------
# Public API: Memory Evolution
# ---------------------------------------------------------------------------

def evolve_memory(
    memory: ExperienceMemory,
    formed_memory: ExperienceMemory,
    max_success_patterns: int = 50,
    max_failure_patterns: int = 100,
    max_insights: int = 30,
) -> ExperienceMemory:
    """Memory Evolution operator E(M, M_form).

    Consolidates newly formed experience into the existing memory.

    Parameters
    ----------
    memory : ExperienceMemory
        Current persistent memory.
    formed_memory : ExperienceMemory
        Newly formed memory from the latest batch (output of form_memory).
    max_success_patterns : int
        Maximum number of success patterns to retain.
    max_failure_patterns : int
        Maximum number of forbidden directions to retain.
    max_insights : int
        Maximum number of strategic insights to retain.

    Returns
    -------
    ExperienceMemory
        Updated memory with consolidated experience.
    """
    # 1. Merge patterns
    merged_success = _merge_success_patterns(
        memory.success_patterns, formed_memory.success_patterns
    )
    merged_forbidden = _merge_forbidden_directions(
        memory.forbidden_directions, formed_memory.forbidden_directions
    )
    merged_insights = _merge_insights(memory.insights, formed_memory.insights)

    # 2. Reclassify patterns that have changed behavior
    merged_success, merged_forbidden = _reclassify_patterns(
        merged_success, merged_forbidden
    )

    # 3. Prune low-utility entries
    merged_success, merged_forbidden, merged_insights = _prune_low_utility(
        merged_success, merged_forbidden, merged_insights
    )

    # 4. Cap memory size
    merged_success, merged_forbidden, merged_insights = _cap_memory_size(
        merged_success, merged_forbidden, merged_insights,
        max_success=max_success_patterns,
        max_forbidden=max_failure_patterns,
        max_insights=max_insights,
    )

    # 5. Update state
    new_state = formed_memory.state

    return ExperienceMemory(
        state=new_state,
        success_patterns=merged_success,
        forbidden_directions=merged_forbidden,
        insights=merged_insights,
        version=memory.version + 1,
    )


# ---------------------------------------------------------------------------
# Phase 2: Online confidence decay helpers (added for HelixFactor)
# ---------------------------------------------------------------------------

def apply_confidence_decay(
    memory: ExperienceMemory,
    decay_factor: float = 0.99,
    min_confidence: float = 0.05,
) -> ExperienceMemory:
    """Return new ExperienceMemory with decayed pattern confidences.

    Seed patterns (occurrence_count == 0) are immune to decay.
    Patterns below min_confidence are pruned.

    Parameters
    ----------
    memory:
        Input ExperienceMemory (not mutated — immutable-style).
    decay_factor:
        Multiplicative decay per call (e.g. 0.99 = 1% decay per iteration).
    min_confidence:
        Patterns with confidence < this threshold after decay are removed.

    Returns
    -------
    ExperienceMemory
        New instance with decayed/pruned patterns.
    """
    import dataclasses

    new_patterns = []
    for p in memory.success_patterns:
        if p.occurrence_count == 0:
            # seed pattern — never decay
            new_patterns.append(p)
            continue
        new_conf = getattr(p, "confidence", 1.0) * decay_factor
        if new_conf >= min_confidence:
            try:
                new_patterns.append(dataclasses.replace(p, confidence=new_conf))
            except TypeError:
                new_patterns.append(p)

    return dataclasses.replace(memory, success_patterns=new_patterns)


def _keyword_list(keywords: list[str] | str) -> list[str]:
    """Normalize a single keyword or keyword list for confidence helpers."""
    if isinstance(keywords, str):
        return [keywords]
    return list(keywords)


def bump_pattern_confidence(
    memory: ExperienceMemory,
    keywords: list[str] | str,
    boost: float = 0.05,
    max_confidence: float = 1.0,
) -> ExperienceMemory:
    """Return new ExperienceMemory with confidence boosted for matching patterns.

    Patterns whose description or template contain any of the keywords receive
    a confidence boost.

    Parameters
    ----------
    memory:
        Input ExperienceMemory (not mutated).
    keywords:
        List of strings to match against pattern descriptions.
    boost:
        Additive confidence increase for matching patterns.
    max_confidence:
        Confidence cap.

    Returns
    -------
    ExperienceMemory
        New instance with boosted pattern confidences.
    """
    import dataclasses

    normalized_keywords = _keyword_list(keywords)
    new_patterns = []
    for p in memory.success_patterns:
        desc = (getattr(p, "description", "") or "").lower()
        tmpl = (getattr(p, "template", "") or "").lower()
        matched = any(
            kw.lower() in desc or kw.lower() in tmpl
            for kw in normalized_keywords
        )
        if matched:
            new_conf = min(getattr(p, "confidence", 1.0) + boost, max_confidence)
            try:
                new_patterns.append(dataclasses.replace(p, confidence=new_conf))
            except TypeError:
                new_patterns.append(p)
        else:
            new_patterns.append(p)

    return dataclasses.replace(memory, success_patterns=new_patterns)


def penalise_pattern_confidence(
    memory: ExperienceMemory,
    keywords: list[str] | str,
    penalty: float = 0.15,
    min_confidence: float = 0.05,
) -> ExperienceMemory:
    """Return new ExperienceMemory with confidence penalised for matching patterns.

    Parameters
    ----------
    memory:
        Input ExperienceMemory (not mutated).
    keywords:
        List of strings to match against pattern descriptions.
    penalty:
        Multiplicative penalty factor (confidence *= (1 - penalty)).
    min_confidence:
        Patterns below this threshold after penalty are pruned.

    Returns
    -------
    ExperienceMemory
        New instance with penalised/pruned pattern confidences.
    """
    import dataclasses

    normalized_keywords = _keyword_list(keywords)
    new_patterns = []
    for p in memory.success_patterns:
        desc = (getattr(p, "description", "") or "").lower()
        tmpl = (getattr(p, "template", "") or "").lower()
        matched = any(
            kw.lower() in desc or kw.lower() in tmpl
            for kw in normalized_keywords
        )
        if matched:
            new_conf = getattr(p, "confidence", 1.0) * (1.0 - penalty)
            if new_conf < min_confidence:
                continue  # prune
            try:
                new_patterns.append(dataclasses.replace(p, confidence=new_conf))
            except TypeError:
                new_patterns.append(p)
        else:
            new_patterns.append(p)

    return dataclasses.replace(memory, success_patterns=new_patterns)
