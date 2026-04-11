"""Specialist agent configurations for domain-focused factor generation.

Each specialist focuses on a particular alpha factor domain with a distinct
cognitive style, preferred operators, domain hypotheses, and historical
success tracking.  ``SpecialistAgent`` wraps a config with per-domain memory
and proposal logic.  ``SpecialistPromptBuilder`` extends the base
``PromptBuilder`` to inject domain-specific directives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from factorminer.agent.llm_interface import LLMProvider
from factorminer.agent.output_parser import parse_llm_output
from factorminer.agent.prompt_builder import (
    SYSTEM_PROMPT,
    PromptBuilder,
    normalize_factor_references,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpecialistConfig:
    """Configuration for a domain-specialist factor generator.

    Attributes
    ----------
    name : str
        Human-readable specialist name (e.g. ``"MomentumMiner"``).
    domain : str
        Domain description used in prompt directives.
    preferred_operators : list[str]
        Operator names this specialist should emphasise.
    preferred_features : list[str]
        Raw features this specialist should lean towards.
    hypothesis : str
        Core economic hypothesis driving this specialist's approach.
    example_factors : list[str]
        Example formulas to ground the specialist in concrete patterns.
    avoid : list[str]
        Structural patterns this specialist should steer clear of.
    temperature : float
        Sampling temperature for LLM calls.
    system_prompt_suffix : str
        Extra paragraph appended to the system prompt.
    provider_config : dict
        Optional provider-level overrides (model, max_tokens, etc.).
    """

    name: str
    domain: str
    preferred_operators: list[str]
    preferred_features: list[str]
    hypothesis: str = ""
    example_factors: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    temperature: float = 0.8
    system_prompt_suffix: str = ""
    provider_config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pre-defined specialist constants
# ---------------------------------------------------------------------------

MOMENTUM_SPECIALIST = SpecialistConfig(
    name="MomentumMiner",
    domain="price momentum and trend following",
    preferred_operators=["TsRank", "Delta", "EMA", "SMA", "TsLinRegSlope", "Return"],
    preferred_features=["$close", "$returns", "$vwap"],
    hypothesis=(
        "Short-term momentum and trend reversals contain predictive signal. "
        "Serial correlation in returns and time-series rank dynamics reveal "
        "persistent directional biases exploitable at the cross-section."
    ),
    example_factors=[
        "Neg(TsRank(Delta($close, 5), 20))",
        "CsRank(TsLinRegSlope($close, 10))",
        "Neg(CsRank(EMA($returns, 8)))",
    ],
    avoid=[
        "volume-only factors without price context",
        "pure cross-sectional without time component",
        "very long windows (>60) on returns",
    ],
    temperature=0.85,
    system_prompt_suffix=(
        "You are the MOMENTUMMINER specialist.  Your cognitive style is "
        "directional and trend-aware.  Focus on price persistence, serial "
        "correlation in returns, and time-series rank dynamics.  Prefer "
        "directional operators (Delta, Return, TsLinRegSlope, EMA, TsRank) "
        "to capture price trajectory information.  Explore both short-term "
        "reversal (1-5 day) and medium-term momentum (10-30 day) regimes.  "
        "Hypothesis: recent price trends contain exploitable signal that "
        "cross-sectional ranking amplifies."
    ),
)

VOLATILITY_SPECIALIST = SpecialistConfig(
    name="VolatilityMiner",
    domain="volatility regimes and higher-moment signals",
    preferred_operators=["Std", "Skew", "Kurt", "TsRank", "IfElse", "Greater"],
    preferred_features=["$returns", "$high", "$low", "$close"],
    hypothesis=(
        "Volatility clustering and moment anomalies predict near-term returns. "
        "Stocks with anomalous higher moments (excess kurtosis, negative skew) "
        "exhibit predictable subsequent return patterns via risk-aversion channels."
    ),
    example_factors=[
        "IfElse(Greater(Std($returns,12), Mean(Std($returns,12),48)), "
        "Neg(CsRank(Delta($close,3))), CsRank(Skew($returns,20)))",
        "Neg(CsRank(Kurt($returns, 20)))",
        "CsRank(Div(Std($returns,5), Std($returns,20)))",
    ],
    avoid=[
        "simple momentum without vol conditioning",
        "long window trends > 40 bars",
        "volume-only volatility without returns",
    ],
    temperature=0.9,
    system_prompt_suffix=(
        "You are the VOLATILITYMINER specialist.  Your cognitive style is "
        "regime-aware and risk-focused.  Combine statistical operators "
        "(Std, Var, Kurt, Skew) with logical branching (IfElse, Greater, Less) "
        "to capture asymmetric behaviour in volatility regimes.  "
        "Explore vol-of-vol, vol regime transitions, and higher-moment "
        "cross-sectional anomalies.  Condition momentum signals on vol "
        "regimes -- high-vol vs low-vol stocks behave very differently.  "
        "Hypothesis: volatility clustering and skewness anomalies carry "
        "cross-sectional predictive power beyond simple momentum."
    ),
)

LIQUIDITY_SPECIALIST = SpecialistConfig(
    name="LiquidityMiner",
    domain="volume, liquidity, and microstructure signals",
    preferred_operators=["Corr", "TsRank", "CsRank", "EMA", "Delta"],
    preferred_features=["$volume", "$amt", "$vwap", "$close"],
    hypothesis=(
        "Volume-price divergence and liquidity dynamics predict order flow "
        "imbalances.  Stocks with abnormal volume relative to price movement "
        "signal informed trading; VWAP deviations capture intraday microstructure."
    ),
    example_factors=[
        "CsRank(Corr($volume, $close, 10))",
        "Neg(CsRank(EMA(Div(Sub($close,$vwap),Add($vwap,1e-4)),5)))",
        "CsZScore(Delta(Mean($amt, 5), 5))",
    ],
    avoid=[
        "volume in isolation without price context",
        "close/open ratio without volume normalization",
        "microstructure without cross-sectional ranking",
    ],
    temperature=0.85,
    system_prompt_suffix=(
        "You are the LIQUIDITYMINER specialist.  Your cognitive style is "
        "microstructure-focused and flow-aware.  Focus on cross-sectional "
        "liquidity patterns: volume-price divergence, turnover anomalies, "
        "and VWAP-based microstructure signals.  Use correlation/covariance "
        "operators to capture relative volume-price alignment.  Explore "
        "amount (dollar volume) signals -- $amt is often underused.  "
        "Condition signals on whether volume is confirming or diverging "
        "from price direction.  Hypothesis: volume-price divergence "
        "and liquidity imbalances predict short-term order flow reversals."
    ),
)

REGIME_SPECIALIST = SpecialistConfig(
    name="RegimeMiner",
    domain="cross-sectional dispersion and regime classification",
    preferred_operators=["CsRank", "CsZScore", "Std", "TsLinRegSlope", "Rsquare", "Resi"],
    preferred_features=["$close", "$returns", "$vwap", "$amt"],
    hypothesis=(
        "Cross-sectional dispersion and regression residuals capture "
        "regime-independent signals.  Stocks that deviate from their "
        "predicted cross-sectional position contain mean-reversion signal "
        "that is robust across bull and bear markets."
    ),
    example_factors=[
        "Mul(CsRank(Rsquare($close, 24)), CsRank(Delta($close, 3)))",
        "CsRank(Resi($close, $vwap, 20))",
        "CsZScore(CsRank(TsLinRegSlope($returns, 15)))",
    ],
    avoid=[
        "single-feature factors without statistical operators",
        "arithmetic without cross-sectional normalization",
        "momentum without regime conditioning",
    ],
    temperature=0.85,
    system_prompt_suffix=(
        "You are the REGINEMINER specialist.  Your cognitive style is "
        "cross-sectional and regression-oriented.  Focus on dispersion, "
        "residual signals, and regime-robust patterns.  Use regression "
        "operators (Rsquare, Resi, TsLinRegSlope) to decompose price "
        "behaviour into systematic and idiosyncratic components.  "
        "Cross-sectional normalization is essential -- every factor should "
        "be comparable across stocks.  Explore cross-asset dispersion "
        "patterns that persist regardless of market direction.  "
        "Hypothesis: cross-sectional regression residuals and R-squared "
        "signals capture regime-independent structural mispricings."
    ),
)

DEFAULT_SPECIALISTS: list[SpecialistConfig] = [
    MOMENTUM_SPECIALIST,
    VOLATILITY_SPECIALIST,
    LIQUIDITY_SPECIALIST,
    REGIME_SPECIALIST,
]

# Map from specialist name to config for convenience
SPECIALIST_CONFIGS: dict[str, SpecialistConfig] = {
    spec.name: spec for spec in DEFAULT_SPECIALISTS
}


# ---------------------------------------------------------------------------
# SpecialistDomainMemory -- per-specialist admission tracking
# ---------------------------------------------------------------------------

@dataclass
class SpecialistDomainMemory:
    """Tracks admission/rejection history for a single specialist.

    Parameters
    ----------
    specialist_name : str
        The name of the specialist this memory belongs to.
    """

    specialist_name: str
    admitted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def total_proposed(self) -> int:
        return len(self.admitted) + len(self.rejected)

    @property
    def success_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return len(self.admitted) / self.total_proposed

    def record_admitted(self, formulas: list[str]) -> None:
        self.admitted.extend(formulas)

    def record_rejected(self, formulas: list[str], reasons: list[str]) -> None:
        self.rejected.extend(formulas)
        self.rejection_reasons.extend(reasons)

    def get_summary(self) -> str:
        """Human-readable summary of domain performance."""
        from collections import Counter
        lines = [
            f"Specialist: {self.specialist_name}",
            f"  Proposed: {self.total_proposed}  Admitted: {len(self.admitted)}  "
            f"Rejected: {len(self.rejected)}",
            f"  Success rate: {self.success_rate:.1%}",
        ]
        if self.admitted:
            lines.append("  Best admitted (last 3):")
            for f in self.admitted[-3:]:
                lines.append(f"    + {f}")
        if self.rejection_reasons:
            counts = Counter(self.rejection_reasons)
            top = counts.most_common(3)
            lines.append("  Top rejection reasons:")
            for reason, count in top:
                lines.append(f"    - {reason} (x{count})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SpecialistAgent -- proposal generation with domain memory
# ---------------------------------------------------------------------------

class SpecialistAgent:
    """Domain-specialist factor proposer with memory and success tracking.

    Each specialist has a unique cognitive style, a preferred operator
    toolkit, and maintains per-domain memory of what has worked and failed.
    Proposals are generated by building a rich context-aware prompt and
    calling the shared LLM provider.

    Parameters
    ----------
    config : SpecialistConfig
        Configuration defining this specialist's domain and style.
    llm : LLMProvider
        LLM backend shared across all specialists.
    base_system_prompt : str or None
        Override for the base system prompt.
    """

    def __init__(
        self,
        config: SpecialistConfig,
        llm: LLMProvider,
        base_system_prompt: str | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self._memory = SpecialistDomainMemory(specialist_name=config.name)

        # Build the specialist prompt builder (extends base PromptBuilder)
        self._prompt_builder = SpecialistPromptBuilder(
            specialist_config=config,
            base_system_prompt=base_system_prompt,
        )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def success_rate(self) -> float:
        """Fraction of this specialist's proposals that were admitted."""
        return self._memory.success_rate

    def generate_proposals(
        self,
        n_proposals: int,
        memory_signal: dict[str, Any] | None = None,
        library_diagnostics: dict[str, Any] | None = None,
        regime_context: str = "",
        forbidden_patterns: list[str] | None = None,
        existing_factors: list[str] | None = None,
    ) -> list[str]:
        """Generate formula string proposals from this specialist.

        Builds a rich domain-aware prompt injecting memory, diagnostics,
        regime context, and forbidden patterns, then calls the LLM and
        parses the response into formula strings.

        Parameters
        ----------
        n_proposals : int
            Number of factor formulas to request.
        memory_signal : dict or None
            Experience memory priors (recommended/forbidden directions, etc.).
        library_diagnostics : dict or None
            Current library state (size, saturation, recent admissions, etc.).
        regime_context : str
            Current market regime description for conditioning.
        forbidden_patterns : list[str] or None
            Structural patterns to explicitly avoid.
        existing_factors : list[str] or None
            Formula strings already in the library (to avoid duplicates).

        Returns
        -------
        list[str]
            List of formula strings proposed by this specialist.
        """
        memory_signal = memory_signal or {}
        library_diagnostics = library_diagnostics or {}
        forbidden_patterns = forbidden_patterns or []
        existing_factors = normalize_factor_references(existing_factors)

        enriched_signal = self._enrich_memory_signal(
            memory_signal, forbidden_patterns, regime_context
        )

        system_prompt = self._prompt_builder.system_prompt
        user_prompt = self._prompt_builder.build_user_prompt(
            memory_signal=enriched_signal,
            library_state=library_diagnostics,
            batch_size=n_proposals,
        )

        logger.debug(
            "Specialist %s generating %d proposals (provider=%s)",
            self.name,
            n_proposals,
            self.llm.provider_name,
        )

        try:
            raw = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
                max_tokens=4096,
            )
        except Exception as exc:
            logger.warning(
                "Specialist %s LLM call failed: %s. Returning empty list.",
                self.name,
                exc,
            )
            return []

        candidates, _ = parse_llm_output(raw)
        valid = [c for c in candidates if c.is_valid]

        if existing_factors:
            existing_set = set(existing_factors)
            valid = [c for c in valid if c.formula not in existing_set]

        formulas = [c.formula for c in valid]
        logger.debug(
            "Specialist %s produced %d valid proposals",
            self.name,
            len(formulas),
        )
        return formulas

    def update_domain_memory(
        self,
        admitted: list[str],
        rejected: list[str],
        reasons: list[str] | None = None,
    ) -> None:
        """Update this specialist's domain memory after evaluation.

        Parameters
        ----------
        admitted : list[str]
            Formulas from this specialist that were admitted to the library.
        rejected : list[str]
            Formulas that were rejected.
        reasons : list[str] or None
            Rejection reasons (parallel to ``rejected``).
        """
        reasons = reasons or ["unknown"] * len(rejected)
        if len(reasons) < len(rejected):
            reasons = reasons + ["unknown"] * (len(rejected) - len(reasons))
        self._memory.record_admitted(admitted)
        self._memory.record_rejected(rejected, reasons[:len(rejected)])

    def get_domain_performance_summary(self) -> str:
        """Human-readable summary of what this specialist has discovered."""
        return self._memory.get_summary()

    def _enrich_memory_signal(
        self,
        base_signal: dict[str, Any],
        forbidden_patterns: list[str],
        regime_context: str,
    ) -> dict[str, Any]:
        """Merge base memory signal with domain-specific context."""
        enriched = dict(base_signal)

        base_forbidden = list(enriched.get("forbidden_directions", []))
        enriched["forbidden_directions"] = base_forbidden + [
            f"[{self.name} domain] Avoid: {p}" for p in self.config.avoid
        ] + forbidden_patterns

        if self.config.example_factors:
            existing_insights = list(enriched.get("strategic_insights", []))
            existing_insights.append(
                f"As {self.name}, your reference examples are: "
                + " | ".join(self.config.example_factors[:3])
            )
            enriched["strategic_insights"] = existing_insights

        if regime_context:
            existing_prompt = enriched.get("prompt_text", "")
            regime_note = f"[Regime context] {regime_context}"
            enriched["prompt_text"] = (
                regime_note + "\n" + existing_prompt
                if existing_prompt
                else regime_note
            )

        if self._memory.total_proposed > 0:
            perf_note = (
                f"[{self.name} history] Success rate: {self.success_rate:.1%} "
                f"({len(self._memory.admitted)} admitted, "
                f"{len(self._memory.rejected)} rejected)."
            )
            existing_insights = list(enriched.get("strategic_insights", []))
            existing_insights.append(perf_note)
            enriched["strategic_insights"] = existing_insights

        return enriched


# ---------------------------------------------------------------------------
# SpecialistPromptBuilder -- extends PromptBuilder with domain directives
# ---------------------------------------------------------------------------

class SpecialistPromptBuilder(PromptBuilder):
    """Prompt builder that injects domain-specific specialist directives.

    Extends the base system prompt with a specialist suffix and biases
    the user prompt towards the specialist's preferred operators, features,
    hypothesis, and example factors.

    Parameters
    ----------
    specialist_config : SpecialistConfig
        The specialist configuration to use.
    base_system_prompt : str or None
        Override for the base system prompt.  Defaults to the global
        ``SYSTEM_PROMPT`` from :mod:`factorminer.agent.prompt_builder`.
    """

    def __init__(
        self,
        specialist_config: SpecialistConfig,
        base_system_prompt: str | None = None,
    ) -> None:
        base = base_system_prompt or SYSTEM_PROMPT
        suffix = specialist_config.system_prompt_suffix
        hypothesis_block = ""
        if specialist_config.hypothesis:
            hypothesis_block = (
                f"\n\n## DOMAIN HYPOTHESIS\n"
                f"{specialist_config.hypothesis}"
            )
        modified_system = (
            f"{base}\n\n"
            f"## SPECIALIST DOMAIN DIRECTIVE\n"
            f"{suffix}"
            f"{hypothesis_block}"
        )
        super().__init__(system_prompt=modified_system)
        self._specialist = specialist_config

    @property
    def specialist_config(self) -> SpecialistConfig:
        """Return the underlying specialist configuration."""
        return self._specialist

    def build_user_prompt(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        batch_size: int = 40,
    ) -> str:
        """Build user prompt with specialist operator/feature bias.

        Calls the base ``PromptBuilder.build_user_prompt`` and appends a
        directive asking the specialist to focus roughly 60% of its
        candidates on its preferred operators and features, plus injects
        example factors for grounding.

        Parameters
        ----------
        memory_signal : dict
            Memory priors (recommended/forbidden directions, etc.).
        library_state : dict
            Current library state (size, saturation, etc.).
        batch_size : int
            Number of candidates to generate.

        Returns
        -------
        str
            Assembled user prompt with specialist bias section.
        """
        base_prompt = super().build_user_prompt(
            memory_signal=memory_signal,
            library_state=library_state,
            batch_size=batch_size,
        )

        spec = self._specialist
        ops = ", ".join(spec.preferred_operators)
        feats = ", ".join(spec.preferred_features)

        specialist_section = (
            f"\n## SPECIALIST FOCUS [{spec.name}]\n"
            f"As the {spec.domain} specialist, focus ~60% of candidates on "
            f"{{{ops}}} operators applied to {{{feats}}} features.\n"
            f"The remaining ~40% should explore creative cross-domain "
            f"combinations to maintain diversity.\n"
        )

        if spec.example_factors:
            specialist_section += (
                "\n## DOMAIN REFERENCE EXAMPLES (structure to emulate, not copy)\n"
                + "\n".join(f"  - {ex}" for ex in spec.example_factors)
                + "\n"
            )

        if spec.avoid:
            specialist_section += (
                "\n## DOMAIN-SPECIFIC AVOIDANCES\n"
                + "\n".join(f"  X {av}" for av in spec.avoid)
                + "\n"
            )

        return base_prompt + specialist_section
