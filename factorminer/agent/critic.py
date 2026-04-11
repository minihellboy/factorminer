"""Critic agent that multi-dimensionally scores candidate factors.

The ``CriticAgent`` pre-filters proposals from specialist agents along six
dimensions before any expensive backtesting occurs.  Only the top-scoring
fraction proceeds to IC evaluation, dramatically reducing wasted compute.

Scoring pipeline:
1. Structural heuristics (complexity, operator diversity) -- O(1) per factor.
2. Novelty scoring via string-level edit-distance and token overlap -- O(n).
3. Pattern alignment against success memory -- keyword matching -- O(n).
4. LLM scoring of top candidates for economic intuition -- one API call.
5. Composite score computation and ranking.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from factorminer.agent.llm_interface import LLMProvider
from factorminer.agent.output_parser import CandidateFactor
from factorminer.agent.prompt_builder import normalize_factor_references

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CriticScore dataclass -- multi-dimensional
# ---------------------------------------------------------------------------

@dataclass
class CriticScore:
    """Multi-dimensional scored review of a single candidate factor.

    Attributes
    ----------
    factor_name : str
        Name of the candidate factor being reviewed.
    formula : str
        DSL formula string of the candidate.
    source_specialist : str
        Name of the specialist that proposed this candidate.
    scores : dict
        Per-dimension scores, each in [0, 1]:
        - ``novelty``: structural distinctiveness from existing library.
        - ``economic_intuition``: economic meaningfulness (LLM-assessed).
        - ``complexity_penalty``: complexity fitness (1 = optimal depth/ops).
        - ``operator_diversity``: uses diverse operator categories.
        - ``pattern_alignment``: aligns with known success patterns in memory.
        - ``regime_appropriateness``: appropriate for current market regime.
    composite_score : float
        Weighted average of dimension scores.
    keep : bool
        Whether this factor should proceed to expensive IC evaluation.
    critique : str
        Natural-language explanation from the critic.
    """

    factor_name: str
    formula: str
    source_specialist: str
    scores: dict[str, float] = field(default_factory=lambda: {
        "novelty": 0.5,
        "economic_intuition": 0.5,
        "complexity_penalty": 0.5,
        "operator_diversity": 0.5,
        "pattern_alignment": 0.5,
        "regime_appropriateness": 0.5,
    })
    composite_score: float = 0.5
    keep: bool = True
    critique: str = ""

    # --- Backward-compatible convenience properties ---

    @property
    def novelty_score(self) -> float:
        return self.scores.get("novelty", 0.5)

    @property
    def quality_score(self) -> float:
        return self.scores.get("economic_intuition", 0.5)

    @property
    def diversity_bonus(self) -> float:
        return self.scores.get("operator_diversity", 0.5)

    @property
    def critic_rationale(self) -> str:
        return self.critique

    @property
    def final_score(self) -> float:
        return self.composite_score


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

_SCORE_WEIGHTS: dict[str, float] = {
    "novelty": 0.25,
    "economic_intuition": 0.30,
    "complexity_penalty": 0.15,
    "operator_diversity": 0.10,
    "pattern_alignment": 0.10,
    "regime_appropriateness": 0.10,
}

# Pre-filter: keep this fraction by composite score before expensive eval
_PREFILTER_FRACTION = 0.60

# LLM scoring: only send this many top candidates to the LLM for economic
# intuition scoring (reduces API cost while covering the promising ones)
_LLM_SCORING_TOP_K = 40


# ---------------------------------------------------------------------------
# Formula-level feature extraction helpers
# ---------------------------------------------------------------------------

# Operator categories for diversity measurement
_OP_CATEGORIES: dict[str, str] = {
    "Add": "arithmetic", "Sub": "arithmetic", "Mul": "arithmetic",
    "Div": "arithmetic", "Neg": "arithmetic", "Abs": "arithmetic",
    "Square": "arithmetic", "Sqrt": "arithmetic", "Log": "arithmetic",
    "Pow": "arithmetic", "Sign": "arithmetic",
    "Std": "statistical", "Var": "statistical", "Mean": "statistical",
    "Sum": "statistical", "Skew": "statistical", "Kurt": "statistical",
    "Median": "statistical", "Quantile": "statistical", "Max": "statistical",
    "Min": "statistical",
    "Delta": "timeseries", "Delay": "timeseries", "TsRank": "timeseries",
    "TsMax": "timeseries", "TsMin": "timeseries", "TsArgMax": "timeseries",
    "TsArgMin": "timeseries", "TsLinRegSlope": "timeseries",
    "Return": "timeseries", "LogReturn": "timeseries", "CumSum": "timeseries",
    "EMA": "smoothing", "SMA": "smoothing", "WMA": "smoothing",
    "HMA": "smoothing", "DEMA": "smoothing", "KAMA": "smoothing",
    "Decay": "smoothing",
    "CsRank": "cross_sectional", "CsZScore": "cross_sectional",
    "CsDemean": "cross_sectional", "CsScale": "cross_sectional",
    "CsNeutralize": "cross_sectional", "CsQuantile": "cross_sectional",
    "Corr": "regression", "Cov": "regression", "Beta": "regression",
    "Resi": "regression", "Rsquare": "regression", "Resid": "regression",
    "IfElse": "logical", "Greater": "logical", "Less": "logical",
    "GreaterEqual": "logical", "LessEqual": "logical", "Equal": "logical",
}

_OPERATOR_PATTERN = re.compile(r"([A-Z][a-zA-Z0-9]*)\s*\(")


def _extract_operators(formula: str) -> list[str]:
    """Extract all operator names from a formula string."""
    return _OPERATOR_PATTERN.findall(formula)


def _formula_depth(formula: str) -> int:
    """Estimate nesting depth by counting maximum parenthesis depth."""
    max_depth = 0
    depth = 0
    for ch in formula:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def _tokenize_formula(formula: str) -> set[str]:
    """Tokenize a formula into its operator and feature tokens."""
    tokens: set[str] = set()
    tokens.update(_OPERATOR_PATTERN.findall(formula))
    for feat in re.findall(r"\$[a-z]+", formula):
        tokens.add(feat)
    return tokens


def _edit_distance_normalized(a: str, b: str, max_len: int = 200) -> float:
    """Compute normalized edit distance between two formula strings.

    Returns 0.0 for identical strings, 1.0 for completely different.
    """
    a = a[:max_len]
    b = b[:max_len]
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[lb] / max(la, lb)


def _token_idf_similarity(formula: str, existing: list[str]) -> float:
    """Compute TF-IDF-inspired token overlap similarity.

    Returns a value in [0, 1] where 1 means very similar to existing,
    0 means completely novel.
    """
    if not existing:
        return 0.0

    query_tokens = _tokenize_formula(formula)
    if not query_tokens:
        return 0.0

    df: Counter = Counter()
    for ex in existing:
        for tok in _tokenize_formula(ex):
            df[tok] += 1

    n_docs = len(existing)
    score = 0.0
    for tok in query_tokens:
        if tok in df:
            idf = math.log(n_docs / df[tok]) if df[tok] < n_docs else 0.0
            score += (1.0 + idf)

    max_score = sum(1.0 for _ in query_tokens)
    if max_score == 0:
        return 0.0
    return min(1.0, score / (max_score * math.log(n_docs + 1) + 1.0))


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class CriticAgent:
    """LLM-powered multi-dimensional critic for candidate factor pre-filtering.

    Evaluates candidates along 6 dimensions before any expensive IC evaluation.
    Uses structural heuristics for fast pre-scoring, then sends top-K to the
    LLM for economic intuition scoring.  Only the top fraction by composite
    score is marked as ``keep=True`` for downstream evaluation.

    Parameters
    ----------
    llm_provider : LLMProvider
        LLM backend for economic intuition scoring.
    temperature : float
        Sampling temperature for the critic's LLM calls.
    max_tokens : int
        Max response tokens for the critic review.
    prefilter_fraction : float
        Fraction of candidates to keep after scoring (0.0-1.0).
    llm_scoring_top_k : int
        How many top candidates (by heuristic score) to send to LLM
        for economic intuition scoring.
    """

    _SYSTEM_PROMPT = (
        "You are a rigorous quantitative research critic specialising in "
        "formulaic alpha factors.  Your job is to evaluate candidate factor "
        "expressions on their economic intuition: does the factor make sense "
        "as a predictor of cross-sectional stock returns?  Be rigorous, "
        "concise, and return structured JSON only."
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        prefilter_fraction: float = _PREFILTER_FRACTION,
        llm_scoring_top_k: int = _LLM_SCORING_TOP_K,
    ) -> None:
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prefilter_fraction = prefilter_fraction
        self.llm_scoring_top_k = llm_scoring_top_k

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def score_batch(
        self,
        candidates: list[str],
        existing_factors: list[str] | None = None,
        memory_signal: str | None = None,
        regime_context: str = "",
        specialist_map: dict[str, str] | None = None,
    ) -> list[CriticScore]:
        """Score a flat list of candidate formula strings.

        Parameters
        ----------
        candidates : list[str]
            Formula strings to evaluate.
        existing_factors : list[str] or None
            Formulas already in the library (for novelty scoring).
        memory_signal : str or None
            Free-text memory context (success patterns, etc.).
        regime_context : str
            Current market regime description.
        specialist_map : dict or None
            Mapping formula -> specialist name for attribution.

        Returns
        -------
        list[CriticScore]
            Scores in the same order as ``candidates``.
        """
        existing_factors = existing_factors or []
        specialist_map = specialist_map or {}

        from factorminer.agent.output_parser import _try_build_candidate
        cf_list: list[CandidateFactor] = []
        for i, formula in enumerate(candidates):
            cf = _try_build_candidate(f"candidate_{i}", formula)
            cf_list.append(cf)

        proposals: dict[str, list[CandidateFactor]] = {}
        for cf in cf_list:
            src = specialist_map.get(cf.formula, "unknown")
            proposals.setdefault(src, []).append(cf)

        return self._score_proposals(
            proposals=proposals,
            existing_factors=existing_factors,
            memory_signal=memory_signal or "",
            regime_context=regime_context,
        )

    def review_candidates(
        self,
        proposals: dict[str, list[CandidateFactor]],
        library_state: dict[str, Any] | None = None,
        memory_signal: dict[str, Any] | None = None,
        top_k: int = 40,
    ) -> list[CriticScore]:
        """Review all specialist proposals and return ranked scores.

        This is the primary interface used by ``DebateGenerator``.

        Parameters
        ----------
        proposals : dict[str, list[CandidateFactor]]
            Mapping from specialist name to its list of candidates.
        library_state : dict or None
            Current factor library state for context.
        memory_signal : dict or None
            Memory priors for context.
        top_k : int
            Number of top-scoring candidates to return.

        Returns
        -------
        list[CriticScore]
            Top-K candidates sorted by ``composite_score`` descending.
        """
        library_state = library_state or {}
        memory_signal = memory_signal or {}

        existing_factors: list[str] = normalize_factor_references(
            library_state.get("recent_admissions", [])
        )
        mem_str = self._memory_signal_to_str(memory_signal)
        regime_context = memory_signal.get("regime_context", "")

        scores = self._score_proposals(
            proposals=proposals,
            existing_factors=existing_factors,
            memory_signal=mem_str,
            regime_context=str(regime_context),
        )

        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Internal scoring pipeline
    # ------------------------------------------------------------------

    def _score_proposals(
        self,
        proposals: dict[str, list[CandidateFactor]],
        existing_factors: list[str],
        memory_signal: str,
        regime_context: str,
    ) -> list[CriticScore]:
        """Full multi-dimensional scoring pipeline."""
        all_pairs: list[tuple[str, CandidateFactor]] = []
        for spec_name, candidates in proposals.items():
            for c in candidates:
                all_pairs.append((spec_name, c))

        if not all_pairs:
            return []

        # Phase 1: Heuristic scoring
        partial_scores: list[CriticScore] = []
        for spec_name, candidate in all_pairs:
            scores_dict = self._heuristic_score(
                formula=candidate.formula,
                existing_factors=existing_factors,
                memory_signal=memory_signal,
                regime_context=regime_context,
            )
            scores_dict["economic_intuition"] = 0.5  # LLM will fill in

            composite = self._compute_composite(scores_dict)
            critique = self._brief_heuristic_critique(scores_dict, candidate.formula)

            partial_scores.append(CriticScore(
                factor_name=candidate.name,
                formula=candidate.formula,
                source_specialist=spec_name,
                scores=scores_dict,
                composite_score=composite,
                keep=True,
                critique=critique,
            ))

        # Phase 2: LLM economic intuition for top candidates
        partial_scores.sort(key=lambda s: s.composite_score, reverse=True)
        top_for_llm = partial_scores[:self.llm_scoring_top_k]

        llm_econ_scores = self._llm_economic_intuition(
            candidates=top_for_llm,
            existing_factors=existing_factors,
            memory_signal=memory_signal,
        )

        # Phase 3: Recompute composite with LLM scores
        for score_obj in partial_scores:
            if score_obj.factor_name in llm_econ_scores:
                econ, rationale = llm_econ_scores[score_obj.factor_name]
                score_obj.scores["economic_intuition"] = econ
                score_obj.composite_score = self._compute_composite(score_obj.scores)
                if rationale:
                    score_obj.critique = rationale

        # Phase 4: Diversity-aware re-ranking
        partial_scores.sort(key=lambda s: s.composite_score, reverse=True)
        partial_scores = self._apply_diversity_adjustment(partial_scores)

        # Phase 5: Pre-filter -- mark keep/discard
        n_keep = max(1, int(len(partial_scores) * self.prefilter_fraction))
        for i, score_obj in enumerate(partial_scores):
            score_obj.keep = i < n_keep

        return partial_scores

    def _heuristic_score(
        self,
        formula: str,
        existing_factors: list[str],
        memory_signal: str,
        regime_context: str,
    ) -> dict[str, float]:
        """Compute heuristic dimension scores without LLM call."""
        operators = _extract_operators(formula)
        depth = _formula_depth(formula)
        unique_ops = list(dict.fromkeys(operators))
        n_unique = len(unique_ops)

        novelty = self._score_novelty(formula, existing_factors)
        complexity = self._score_complexity(depth, n_unique)
        op_diversity = self._score_operator_diversity(unique_ops)
        pattern_align = self._score_pattern_alignment(formula, memory_signal)
        regime_score = self._score_regime_appropriateness(formula, regime_context)

        return {
            "novelty": novelty,
            "economic_intuition": 0.5,
            "complexity_penalty": complexity,
            "operator_diversity": op_diversity,
            "pattern_alignment": pattern_align,
            "regime_appropriateness": regime_score,
        }

    def _score_novelty(self, formula: str, existing_factors: list[str]) -> float:
        """Novelty: 1.0 = completely novel, 0.0 = exact duplicate."""
        if not existing_factors:
            return 0.8

        token_sim = _token_idf_similarity(formula, existing_factors)
        sample = existing_factors[-20:]
        edit_dists = [_edit_distance_normalized(formula, ex) for ex in sample]
        avg_edit = sum(edit_dists) / len(edit_dists) if edit_dists else 1.0

        novelty = 0.5 * (1.0 - token_sim) + 0.5 * avg_edit
        return float(max(0.0, min(1.0, novelty)))

    def _score_complexity(self, depth: int, n_unique_ops: int) -> float:
        """Complexity fitness: 1.0 = optimal (depth 3-7, 3-5 unique ops)."""
        if 3 <= depth <= 7:
            depth_score = 1.0
        elif depth < 3:
            depth_score = depth / 3.0
        else:
            depth_score = max(0.0, 1.0 - 0.15 * (depth - 7))

        if 3 <= n_unique_ops <= 5:
            op_score = 1.0
        elif n_unique_ops < 3:
            op_score = n_unique_ops / 3.0
        else:
            op_score = max(0.0, 1.0 - 0.1 * (n_unique_ops - 5))

        return float(0.6 * depth_score + 0.4 * op_score)

    def _score_operator_diversity(self, unique_ops: list[str]) -> float:
        """Operator diversity: how many distinct operator categories appear?"""
        categories = {_OP_CATEGORIES.get(op, "other") for op in unique_ops}
        n_categories = len(categories)
        diversity_map = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.8}
        return float(diversity_map.get(n_categories, 1.0))

    def _score_pattern_alignment(self, formula: str, memory_signal: str) -> float:
        """Pattern alignment: do formula tokens appear in known success patterns?"""
        if not memory_signal:
            return 0.5

        formula_lower = formula.lower()
        signal_lower = memory_signal.lower()

        formula_tokens = set(re.findall(r"[a-z]+", formula_lower))
        signal_tokens = set(re.findall(r"[a-z]+", signal_lower))

        stopwords = {"the", "a", "is", "in", "of", "to", "and", "or", "for",
                     "as", "by", "on", "it", "be", "at", "an", "up"}
        formula_tokens -= stopwords
        signal_tokens -= stopwords

        if not formula_tokens:
            return 0.5

        overlap = formula_tokens & signal_tokens
        alignment = len(overlap) / len(formula_tokens)
        return float(0.3 + 0.7 * min(1.0, alignment * 2))

    def _score_regime_appropriateness(
        self, formula: str, regime_context: str
    ) -> float:
        """Does this formula suit the stated regime context?"""
        if not regime_context:
            return 0.7

        regime_lower = regime_context.lower()

        momentum_kw = {"momentum", "trend", "trending", "breakout"}
        volatility_kw = {"volatile", "volatility", "risk-off", "vix"}
        reversal_kw = {"reversal", "mean-reversion", "oversold", "overbought"}
        liquidity_kw = {"illiquid", "liquidity", "volume"}

        regime_is_momentum = any(k in regime_lower for k in momentum_kw)
        regime_is_volatile = any(k in regime_lower for k in volatility_kw)
        regime_is_reversal = any(k in regime_lower for k in reversal_kw)
        regime_is_illiquid = any(k in regime_lower for k in liquidity_kw)

        formula_has_momentum = any(
            op in formula for op in ("Delta", "TsRank", "EMA", "Return", "TsLinReg")
        )
        formula_has_vol = any(op in formula for op in ("Std", "Kurt", "Skew", "Var"))
        formula_has_reversal = "Neg" in formula or any(
            op in formula for op in ("Mean", "SMA", "TsRank")
        )
        formula_has_volume = any(f in formula for f in ("$volume", "$amt"))

        matches = 0
        total_signals = 0
        if regime_is_momentum:
            total_signals += 1
            matches += int(formula_has_momentum)
        if regime_is_volatile:
            total_signals += 1
            matches += int(formula_has_vol)
        if regime_is_reversal:
            total_signals += 1
            matches += int(formula_has_reversal)
        if regime_is_illiquid:
            total_signals += 1
            matches += int(formula_has_volume)

        if total_signals == 0:
            return 0.7
        return float(0.4 + 0.6 * (matches / total_signals))

    @staticmethod
    def _compute_composite(scores: dict[str, float]) -> float:
        """Compute weighted composite score from dimension scores."""
        total = 0.0
        weight_sum = 0.0
        for dim, weight in _SCORE_WEIGHTS.items():
            val = scores.get(dim, 0.5)
            total += weight * val
            weight_sum += weight
        if weight_sum == 0:
            return 0.5
        return float(total / weight_sum)

    def _brief_heuristic_critique(
        self, scores: dict[str, float], formula: str
    ) -> str:
        """Generate a brief human-readable critique from heuristic scores."""
        parts = []
        depth = _formula_depth(formula)
        ops = _extract_operators(formula)
        n_unique = len(set(ops))

        novelty = scores.get("novelty", 0.5)
        if novelty < 0.3:
            parts.append("closely resembles existing library factors")
        elif novelty > 0.7:
            parts.append("structurally novel")

        complexity = scores.get("complexity_penalty", 0.5)
        if complexity < 0.4:
            if depth < 3:
                parts.append(f"too shallow (depth={depth})")
            elif depth > 8:
                parts.append(f"overly deep (depth={depth})")
            if n_unique < 3:
                parts.append(f"low operator diversity ({n_unique} unique ops)")

        op_div = scores.get("operator_diversity", 0.5)
        if op_div >= 0.8:
            cats = {_OP_CATEGORIES.get(op, "other") for op in set(ops)}
            parts.append(f"good operator variety ({', '.join(sorted(cats))})")

        if not parts:
            parts.append("passes heuristic checks")

        return "; ".join(parts) + "."

    # ------------------------------------------------------------------
    # LLM economic intuition scoring
    # ------------------------------------------------------------------

    def _llm_economic_intuition(
        self,
        candidates: list[CriticScore],
        existing_factors: list[str],
        memory_signal: str,
    ) -> dict[str, tuple[float, str]]:
        """Send top candidates to LLM for economic intuition scoring."""
        if not candidates:
            return {}

        prompt = self._build_llm_scoring_prompt(
            candidates=candidates,
            existing_factors=existing_factors,
            memory_signal=memory_signal,
        )

        try:
            raw = self.llm_provider.generate(
                system_prompt=self._SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self._parse_llm_scoring_response(raw, candidates)
        except Exception as exc:
            logger.warning(
                "Critic LLM economic intuition scoring failed: %s. "
                "Keeping heuristic scores.",
                exc,
            )
            return {}

    def _build_llm_scoring_prompt(
        self,
        candidates: list[CriticScore],
        existing_factors: list[str],
        memory_signal: str,
    ) -> str:
        """Build the structured scoring prompt for LLM economic intuition."""
        sections: list[str] = []

        if existing_factors:
            sections.append("## EXISTING LIBRARY SAMPLE (last 10)")
            for f in existing_factors[-10:]:
                sections.append(f"  - {f}")

        if memory_signal:
            sections.append(f"\n## MEMORY CONTEXT\n{memory_signal[:800]}")

        sections.append("\n## CANDIDATES FOR ECONOMIC INTUITION SCORING")
        sections.append(
            "Score each on economic_intuition (0.0-1.0): does this formula "
            "capture a plausible, economically meaningful cross-sectional "
            "return predictor?  Consider:\n"
            "  - Is there a coherent economic story?\n"
            "  - Is the complexity level appropriate (depth 3-7 is best)?\n"
            "  - Does it avoid trivial reformulations of simple momentum/reversal?\n"
            "  - Does it use features in a semantically coherent way?\n"
        )

        for cs in candidates:
            sections.append(
                f"  Factor: {cs.factor_name}  "
                f"[Specialist: {cs.source_specialist}]\n"
                f"  Formula: {cs.formula}"
            )

        sections.append(
            "\n## OUTPUT FORMAT\n"
            "Return one JSON object per line, exactly:\n"
            '{"name": "<factor_name>", "economic_intuition": <0.0-1.0>, '
            '"rationale": "<one sentence>"}\n'
            "Output ONLY the JSON lines. No markdown, no explanations."
        )

        return "\n".join(sections)

    def _parse_llm_scoring_response(
        self,
        raw: str,
        candidates: list[CriticScore],
    ) -> dict[str, tuple[float, str]]:
        """Parse LLM scoring response into economic intuition scores."""
        valid_names = {cs.factor_name for cs in candidates}
        results: dict[str, tuple[float, str]] = {}
        json_pattern = re.compile(r"\{[^{}]+\}")

        for match in json_pattern.findall(raw):
            try:
                obj = json.loads(match)
            except json.JSONDecodeError:
                continue
            name = obj.get("name", "")
            if name not in valid_names:
                continue
            econ = float(max(0.0, min(1.0, obj.get("economic_intuition", 0.5))))
            rationale = str(obj.get("rationale", ""))
            results[name] = (econ, rationale)

        logger.debug(
            "LLM economic intuition: scored %d/%d candidates",
            len(results),
            len(candidates),
        )
        return results

    # ------------------------------------------------------------------
    # Diversity adjustment
    # ------------------------------------------------------------------

    def _apply_diversity_adjustment(
        self, scores: list[CriticScore]
    ) -> list[CriticScore]:
        """Slightly boost underrepresented specialists to maintain balance."""
        if not scores:
            return scores

        specialist_counts: Counter = Counter()
        n_specialists = len({s.source_specialist for s in scores})
        ideal_frac = 1.0 / max(n_specialists, 1)

        adjusted = []
        for cs in scores:
            specialist_counts[cs.source_specialist] += 1
            total_so_far = sum(specialist_counts.values())
            actual_frac = specialist_counts[cs.source_specialist] / total_so_far
            diversity_adj = (ideal_frac - actual_frac) * 0.1
            diversity_adj = max(-0.05, min(0.05, diversity_adj))
            adjusted_score = float(
                max(0.0, min(1.0, cs.composite_score + diversity_adj))
            )
            new_scores = dict(cs.scores)
            new_scores["operator_diversity"] = float(
                max(0.0, min(1.0,
                    cs.scores.get("operator_diversity", 0.5) + diversity_adj
                ))
            )
            adjusted.append(CriticScore(
                factor_name=cs.factor_name,
                formula=cs.formula,
                source_specialist=cs.source_specialist,
                scores=new_scores,
                composite_score=adjusted_score,
                keep=cs.keep,
                critique=cs.critique,
            ))

        adjusted.sort(key=lambda s: s.composite_score, reverse=True)
        return adjusted

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _memory_signal_to_str(memory_signal: dict[str, Any]) -> str:
        """Flatten a memory signal dict to a compact string for embedding."""
        parts: list[str] = []
        for key in (
            "recommended_directions", "strategic_insights",
            "complementary_patterns", "prompt_text",
        ):
            val = memory_signal.get(key)
            if isinstance(val, list):
                parts.extend(str(v) for v in val)
            elif isinstance(val, str) and val:
                parts.append(val)
        return " ".join(parts)

    @staticmethod
    def _fallback_uniform_scores(
        proposals: dict[str, list[CandidateFactor]],
    ) -> list[CriticScore]:
        """Generate uniform scores when all scoring mechanisms fail."""
        default_composite = 0.5
        scores: list[CriticScore] = []
        for specialist_name, candidates in proposals.items():
            for c in candidates:
                scores.append(CriticScore(
                    factor_name=c.name,
                    formula=c.formula,
                    source_specialist=specialist_name,
                    scores={
                        "novelty": 0.5,
                        "economic_intuition": 0.5,
                        "complexity_penalty": 0.5,
                        "operator_diversity": 0.5,
                        "pattern_alignment": 0.5,
                        "regime_appropriateness": 0.5,
                    },
                    composite_score=default_composite,
                    keep=True,
                    critique="Fallback uniform score (critic unavailable).",
                ))
        return scores
