"""XAlpha-style Report-to-Memory Absorption (RMA) service.

Implements a proportionate subset of the RMA layer described in XAlpha (Liu et al.,
2026, "XALPHA: A Memory-Driven AI Quant Researcher for Hypothesis-to-Code Alpha
Discovery", arXiv:2607.08332, Sec. 3.1): a three-layer A/B/C taxonomy that turns
external research fragments into structured, retrievable knowledge instead of
dumping raw report text into generation prompts.

* **A-layer (OHLCV eligibility gate)** -- ``screen_eligibility``: KEEP/DROP a
  fragment based on whether its core mechanism can be observed, inferred, or
  stably proxied from daily open/high/low/close/volume bars. Fragments that
  fundamentally depend on unavailable information (analyst estimates, fundamentals,
  order-book microstructure, news/sentiment text, macro releases) are DROPped.
* **B-layer (mechanism family)** -- reuses
  ``factorminer.architecture.families.mechanism_family``, the modest 6-entry
  grouping layer that already sits above ``infer_family``'s 11 fine-grained
  categories (``families.MECHANISM_FAMILIES``); this module adds no separate
  taxonomy of its own.
* **C-layer (research archetype)** -- ``classify_mechanism``: specializes a KEEP
  fragment into a ``ResearchArchetype`` record -- a structured research cue
  (``research_paths``), not a factor formula -- for downstream hypothesis
  generation via ``PromptContextBuilder``.

This module deliberately implements only the RMA absorption boundary. XAlpha's
48-archetype taxonomy and its Macro/Micro/Cross-Brain research loop are out of
scope here; ``ResearchAbsorptionService.absorb`` is the batch entry point that
screens and classifies a list of report fragments into ``ResearchArchetype``
records ready to feed generation prompts.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from factorminer.agent.llm_interface import LLMProvider
from factorminer.architecture.families import MECHANISM_FAMILIES, mechanism_family

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# B-layer: mechanism families. This module does NOT define its own taxonomy --
# it reuses `factorminer.architecture.families.mechanism_family`, the modest
# grouping layer (`MECHANISM_FAMILIES`) that already sits above the 11
# `infer_family` categories, so RMA and family-aware macro-cycle routing share
# one taxonomy instead of two.
# ---------------------------------------------------------------------------

#: Keyword cues used to infer a fine-grained family from free-text research
#: fragments (as opposed to `infer_family`, which parses formula operators).
_FINE_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Momentum": (
        "momentum", "trend continuation", "breakout", "moving average crossover",
        "return persistence", "price acceleration", "trend following",
    ),
    "Smoothing": (
        "smoothed", "moving average", "denoise", "exponential average", "filtered price",
    ),
    "Regression": (
        "regression", "residual", "linear fit", "slope", "trendline deviation",
    ),
    "VWAP": (
        "vwap", "volume weighted average price",
    ),
    "Amount": (
        "turnover value", "dollar volume", "yuan volume", "amount traded", "traded value",
    ),
    "Volatility": (
        "volatility", "realized vol", "variance", "risk premium", "vol spike", "vol clustering",
    ),
    "Higher-Moment": (
        "skew", "kurtosis", "tail risk", "fat tail", "return asymmetry",
    ),
    "Extrema": (
        "reversal", "mean reversion", "selling exhaustion", "buying exhaustion",
        "oversold", "overbought", "52-week high", "52-week low", "new low", "new high",
        "range breakout", "bounce",
    ),
    "Cross-Sectional": (
        "relative to peers", "sector rank", "percentile among stocks", "cross-sectional",
        "relative strength versus", "peer comparison",
    ),
    "PV-Correlation": (
        "price-volume correlation", "volume confirms price", "volume divergence",
        "price-volume divergence", "volume precedes price", "volume leads price",
    ),
    "Regime-Conditional": (
        "regime", "bull market", "bear market", "conditional on volatility state",
        "market state", "high-volatility regime", "low-volatility regime",
    ),
}

#: Keyword cues that make a fragment's mechanism OHLCV-representable (A-layer KEEP).
_ELIGIBLE_KEYWORDS: tuple[str, ...] = (
    "price", "volume", "close", "open price", "high", "low", "candle", "bar", "ohlcv",
    "reversal", "momentum", "breakout", "volatility", "turnover", "selling", "buying",
    "exhaustion", "trend", "support", "resistance", "bounce", "rally", "drawdown",
    "range", "gap", "vwap", "return persistence",
)

#: Keyword cues that make a fragment's mechanism OHLCV-ineligible (A-layer DROP):
#: fundamentals/analyst data, order-book microstructure, news/sentiment, macro.
#: In ``alt_enabled`` mode, fundamentals/SEC-filing cues that map onto a
#: registered non-OHLCV leaf are reclassified as KEEP (see
#: :data:`_ALT_FEATURE_KEYWORDS`).
_INELIGIBLE_KEYWORDS: tuple[str, ...] = (
    "eps", "earnings", "analyst", "estimate revision", "consensus estimate", "guidance",
    "fundamental", "order book", "order-book", "limit order", "book depth",
    "sec filing", "10-k", "10-q", "balance sheet", "sentiment", "news headline",
    "macroeconomic", "gdp", "cpi", "interest rate decision", "fed statement",
    "dividend announcement", "buyback announcement", "insider transaction",
)

#: Keywords that become eligible under ``alt_enabled`` when the matching leaf
#: is present in the live feature registry (Phase A/B extensible leaves).
_ALT_FEATURE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "$eps": ("eps", "earnings per share", "earnings"),
    "$revenue": ("revenue", "sales", "top-line", "top line"),
    "$book_equity": ("book equity", "book value", "stockholders equity", "balance sheet"),
    "$shares_out": ("shares outstanding", "share count", "diluted shares"),
    "$basis": ("basis", "futures basis", "cash and carry"),
    "$spot": ("spot price", "spot market"),
    "$premium": ("futures premium", "contango", "backwardation"),
    "$roll_yield": ("roll yield", "roll return"),
    "$oi": ("open interest", "oi "),
}

_VERDICT_PATTERN = re.compile(r"\b(KEEP|DROP)\b\s*[:\-]\s*(.+)", re.IGNORECASE)
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

_A_LAYER_SYSTEM_PROMPT = (
    "You are the OHLCV-eligibility gate (RMA A-layer) of a formulaic alpha-mining "
    "research assistant. Decide whether the core mechanism described in a research "
    "fragment can be directly observed, inferred, or stably proxied using only daily "
    "open, high, low, close, and volume (OHLCV) bars for a single tradable asset. "
    "KEEP fragments whose mechanism is representable from price/volume action alone "
    "(e.g. momentum, reversal, volatility, volume/turnover dynamics, price-volume "
    "co-movement). DROP fragments that fundamentally require information OHLCV bars "
    "cannot provide or proxy, such as analyst estimates/EPS revisions, fundamentals "
    "(earnings, balance sheet, guidance), order-book microstructure, news/sentiment "
    "text, or macroeconomic releases.\n"
    "Respond with exactly one line: 'KEEP: <one-sentence reason>' or "
    "'DROP: <one-sentence reason>'."
)

_C_LAYER_SYSTEM_PROMPT = (
    "You are the mechanism-family and archetype layer (RMA B/C-layer) of a "
    "formulaic alpha-mining research assistant. You receive a research fragment "
    "that has already passed the OHLCV-eligibility gate. Respond with a single "
    "JSON object of the form "
    '{"archetype_name": "<short_snake_case_slug>", '
    '"mechanism_role": "<one sentence on how the mechanism would manifest in '
    'OHLCV factor code>", "research_paths": ["<hypothesis cue 1>", "<cue 2>"]}. '
    "Use at most 3 research_paths. Respond with the JSON object only."
)


def _infer_fine_family_from_text(text: str) -> str:
    """Infer one of the 11 `families.infer_family` categories from free text."""
    lowered = text.lower()
    best_family = "Other"
    best_hits = 0
    for family, keywords in _FINE_FAMILY_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits > best_hits:
            best_hits = hits
            best_family = family
    return best_family


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "archetype"


def _heuristic_eligibility(
    text: str,
    *,
    mode: str = "ohlcv_only",
    registered_features: frozenset[str] | None = None,
) -> tuple[bool, str]:
    """Deterministic keyword-based A-layer fallback.

    Used whenever the LLM provider's raw response carries no parseable
    KEEP/DROP verdict -- in particular this is what the mock LLM provider
    resolves to for this call type, since `MockProvider.generate` always
    returns its canned factor-formula listing regardless of prompt content.

    Parameters
    ----------
    mode :
        ``"ohlcv_only"`` (default, backward compatible) hard-drops
        fundamentals/SEC/on-chain cues. ``"alt_enabled"`` keeps a fragment
        when its mechanism maps to a registered non-OHLCV feature leaf.
    registered_features :
        Optional explicit set of DSL leaves. Defaults to the union of the
        live feature registry and the known alt-data leaf catalog
        (:data:`_ALT_FEATURE_KEYWORDS`), so ``alt_enabled`` screening is
        meaningful even when called standalone (e.g. from `ingest-research`)
        before any EDGAR/futures loader has registered leaves in-process.
        Pass an explicit set to gate strictly on what is actually live in a
        given session instead (e.g. from inside a mining loop).
    """
    lowered = text.lower()
    ineligible_hits = [keyword for keyword in _INELIGIBLE_KEYWORDS if keyword in lowered]

    if mode == "alt_enabled" and ineligible_hits:
        if registered_features is not None:
            active = registered_features
        else:
            from factorminer.core.types import get_feature_set

            active = get_feature_set() | frozenset(_ALT_FEATURE_KEYWORDS)
        for leaf, keywords in _ALT_FEATURE_KEYWORDS.items():
            if leaf not in active:
                continue
            for kw in keywords:
                if kw in lowered:
                    return True, (
                        f"Mechanism maps to registered alt-data leaf '{leaf}' "
                        f"(matched '{kw}'); kept under alt_enabled eligibility."
                    )

    if ineligible_hits:
        return False, (
            f"Mechanism depends on '{ineligible_hits[0]}', which cannot be observed "
            "or stably proxied from daily OHLCV bars."
        )
    eligible_hits = [keyword for keyword in _ELIGIBLE_KEYWORDS if keyword in lowered]
    if eligible_hits:
        return True, (
            f"Mechanism is grounded in '{eligible_hits[0]}', directly representable "
            "from daily OHLCV bars."
        )
    return False, "No OHLCV-representable price/volume mechanism was identified in the fragment."


def _heuristic_archetype(text: str, fine_family: str) -> tuple[str, str, list[str]]:
    """Deterministic archetype construction fallback (mock-provider behavior)."""
    words = text.strip().split()
    snippet = " ".join(words[:12]) + ("..." if len(words) > 12 else "")
    name = f"{_slugify(fine_family)}_archetype"
    mechanism_role = (
        f"Represent the {fine_family.lower()} mechanism described in the fragment "
        f"as an OHLCV-derived factor: \"{snippet}\""
    )
    lowered = text.lower()
    hits = [keyword for keyword in _FINE_FAMILY_KEYWORDS.get(fine_family, ()) if keyword in lowered]
    research_paths = [f"Proxy '{hit}' via an OHLCV-derived operator sequence." for hit in hits[:3]]
    if not research_paths:
        research_paths = [f"Explore OHLCV formulations consistent with: {snippet}"]
    return name, mechanism_role, research_paths


def _parse_archetype_json(raw: str) -> tuple[str, str, list[str]] | None:
    match = _JSON_OBJECT_PATTERN.search(raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    name = str(obj.get("archetype_name") or obj.get("name") or "").strip()
    mechanism_role = str(obj.get("mechanism_role") or "").strip()
    raw_paths = obj.get("research_paths")
    if not name or not isinstance(raw_paths, list):
        return None
    research_paths = [str(path) for path in raw_paths if str(path).strip()]
    return name, mechanism_role, research_paths


@dataclass(frozen=True)
class ResearchNote:
    """An external research fragment submitted for RMA absorption.

    Parameters
    ----------
    text : str
        The raw report/paper fragment text.
    source : str
        Provenance label (file path, report title, or "unspecified").
    """

    text: str
    source: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchArchetype:
    """RMA C-layer output: a retrievable research cue, not a factor formula.

    Mirrors XAlpha's (arXiv:2607.08332, Sec. 3.1) C-layer archetype record,
    scoped down to a single mechanism role plus a short list of hypothesis
    cues (``research_paths``) rather than the paper's full 48-entry taxonomy.

    Attributes
    ----------
    name : str
        Short archetype slug, e.g. ``"extrema_archetype"``.
    mechanism_family : str
        One of `MECHANISM_FAMILIES` (RMA B-layer).
    fine_family : str
        One of the 11 `factorminer.architecture.families.infer_family`
        categories used to derive `mechanism_family`.
    mechanism_role : str
        One-sentence description of how the mechanism would manifest as an
        OHLCV-derived factor.
    research_paths : list[str]
        Reusable hypothesis cues for Macro-Brain-style routing / prompt injection.
    source_text : str
        The originating (KEEP) fragment text.
    eligibility_reason : str
        The A-layer KEEP reason recorded when this archetype was constructed.
    """

    name: str
    mechanism_family: str
    fine_family: str
    mechanism_role: str
    research_paths: list[str] = field(default_factory=list)
    source_text: str = ""
    eligibility_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_research_note(path: str | Path) -> ResearchNote:
    """Read a plain-text research fragment from disk into a `ResearchNote`."""
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    return ResearchNote(text=text, source=str(resolved))


class ResearchAbsorptionService:
    """XAlpha-style Report-to-Memory Absorption (RMA) service.

    Screens external research fragments for OHLCV-representability (A-layer),
    then classifies KEEP fragments into `ResearchArchetype` records carrying a
    broad `mechanism_family` (B-layer) and reusable `research_paths` hypothesis
    cues (C-layer). See module docstring for the paper reference and scope.

    Parameters
    ----------
    llm_provider : LLMProvider
        The LLM backend used for eligibility screening and mechanism
        classification. Pass `factorminer.agent.llm_interface.MockProvider()`
        for offline/deterministic use (matches the repo's `--mock` convention).
    max_research_paths : int
        Cap on the number of `research_paths` cues per archetype.
    eligibility_mode : str
        ``"ohlcv_only"`` (default) preserves the historical hard-drop of
        fundamentals/SEC/on-chain fragments. ``"alt_enabled"`` keeps a
        fragment when its mechanism maps to a registered non-OHLCV feature
        leaf (see Phase A/B extensible feature registry).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_research_paths: int = 3,
        *,
        eligibility_mode: str = "ohlcv_only",
    ) -> None:
        if eligibility_mode not in {"ohlcv_only", "alt_enabled"}:
            raise ValueError(
                "eligibility_mode must be 'ohlcv_only' or 'alt_enabled', "
                f"got {eligibility_mode!r}"
            )
        self.llm_provider = llm_provider
        self.max_research_paths = max_research_paths
        self.eligibility_mode = eligibility_mode

    def screen_eligibility(self, text: str) -> tuple[bool, str]:
        """A-layer: decide KEEP/DROP based on OHLCV / alt-data representability.

        Returns
        -------
        tuple[bool, str]
            ``(keep, reason)`` -- ``keep`` is True for KEEP, False for DROP.
        """
        raw = self.llm_provider.generate(
            system_prompt=_A_LAYER_SYSTEM_PROMPT,
            user_prompt=text,
            temperature=0.0,
            max_tokens=128,
        )
        match = _VERDICT_PATTERN.search(raw)
        if match:
            verdict = match.group(1).upper()
            reason = match.group(2).strip()
            if reason:
                # Even when the LLM returns DROP, alt_enabled may override for
                # registered fundamental leaves (deterministic, mock-safe).
                if verdict == "DROP" and self.eligibility_mode == "alt_enabled":
                    keep_alt, alt_reason = _heuristic_eligibility(
                        text, mode="alt_enabled"
                    )
                    if keep_alt:
                        return True, alt_reason
                return verdict == "KEEP", reason
        return _heuristic_eligibility(text, mode=self.eligibility_mode)

    def classify_mechanism(self, text: str) -> ResearchArchetype:
        """B/C-layer: classify a KEEP fragment into a `ResearchArchetype`.

        Callers are expected to have already confirmed KEEP via
        `screen_eligibility`; this method does not re-run the gate.
        """
        fine_family = _infer_fine_family_from_text(text)
        mechanism_family_name = mechanism_family(fine_family)
        assert mechanism_family_name in MECHANISM_FAMILIES, (
            f"mechanism_family() returned {mechanism_family_name!r}, expected one of "
            f"{MECHANISM_FAMILIES}"
        )

        raw = self.llm_provider.generate(
            system_prompt=_C_LAYER_SYSTEM_PROMPT,
            user_prompt=text,
            temperature=0.0,
            max_tokens=256,
        )
        parsed = _parse_archetype_json(raw)
        if parsed is not None:
            name, mechanism_role, research_paths = parsed
        else:
            name, mechanism_role, research_paths = _heuristic_archetype(text, fine_family)

        return ResearchArchetype(
            name=name,
            mechanism_family=mechanism_family_name,
            fine_family=fine_family,
            mechanism_role=mechanism_role,
            research_paths=research_paths[: self.max_research_paths],
            source_text=text,
        )

    def absorb(self, notes: Sequence[str]) -> list[ResearchArchetype]:
        """Batch RMA pipeline: A-layer screen then B/C-layer classify each note.

        DROPped notes are logged and excluded from the result. Returns at most
        one `ResearchArchetype` per eligible note, in input order.
        """
        archetypes: list[ResearchArchetype] = []
        for note in notes:
            keep, reason = self.screen_eligibility(note)
            if not keep:
                logger.info("RMA A-layer DROP (%s): %s", self.llm_provider.provider_name, reason)
                continue
            archetype = self.classify_mechanism(note)
            archetypes.append(
                ResearchArchetype(
                    name=archetype.name,
                    mechanism_family=archetype.mechanism_family,
                    fine_family=archetype.fine_family,
                    mechanism_role=archetype.mechanism_role,
                    research_paths=archetype.research_paths,
                    source_text=archetype.source_text,
                    eligibility_reason=reason,
                )
            )
        return archetypes
