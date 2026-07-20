"""Main factor generation agent using LLM guided by memory priors.

Orchestrates the prompt construction, LLM invocation, output parsing,
and retry logic for a single batch of factor candidates.

When the LLM backend is a :class:`~factorminer.agent.llm_interface.CascadeProvider`,
the initial draft uses the cheap local model; the repair path always
escalates to the frontier provider.  Escalation is driven by FactorMiner's
deterministic DSL parser — a free routing signal most generic agentic-cost
papers do not have.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from factorminer.agent.llm_interface import CascadeProvider, LLMProvider
from factorminer.agent.output_parser import CandidateFactor, parse_llm_output
from factorminer.agent.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class FactorGenerator:
    """LLM-based factor generation agent.

    Generates batches of candidate factors by constructing prompts that
    inject experience memory priors, calling an LLM provider, and parsing
    the output into validated CandidateFactor objects.

    Parameters
    ----------
    llm_provider : LLMProvider
        The LLM backend to use for text generation.
    prompt_builder : PromptBuilder
        Builds system and user prompts.
    temperature : float
        Default sampling temperature.
    max_tokens : int
        Default max response tokens.
    cacheable_prefix : str or None
        Optional stable system-prompt prefix shared with peer generators
        (e.g. debate specialists).  Forwarded to the provider so parallel
        callers hit the same prompt-cache breakpoint.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_builder: PromptBuilder | None = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        cacheable_prefix: str | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Prefer an explicit prefix; otherwise use the full system prompt so
        # a lone FactorGenerator still marks a stable cache breakpoint.
        self.cacheable_prefix = (
            cacheable_prefix
            if cacheable_prefix is not None
            else self.prompt_builder.system_prompt
        )
        self._generation_count = 0

    @staticmethod
    def _unwrap_cascade_provider(provider: LLMProvider) -> CascadeProvider | None:
        """Walk through known provider wrappers to find an inner CascadeProvider.

        ``PrefixedCacheProvider`` (used by ``DebateGenerator`` so parallel
        specialists share one cache breakpoint) does not itself expose
        ``generate_frontier``, so a naive
        ``isinstance(provider, CascadeProvider)`` check silently fails once
        the cascade is wrapped -- ``force_frontier`` then becomes a no-op:
        repair re-enters the cheap local draft instead of forcing the
        frontier model, exactly when quality-safety repair matters most.
        """
        seen: set[int] = set()
        current: Any = provider
        while current is not None and id(current) not in seen:
            if isinstance(current, CascadeProvider):
                return current
            seen.add(id(current))
            current = getattr(current, "inner", None)
        return None

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        force_frontier: bool = False,
    ) -> str:
        """Invoke the provider, optionally forcing cascade frontier repair."""
        kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cacheable_prefix": self.cacheable_prefix,
        }
        if force_frontier:
            cascade = self._unwrap_cascade_provider(self.llm_provider)
            if cascade is not None:
                return cascade.generate_frontier(**kwargs)
        return self.llm_provider.generate(**kwargs)

    def generate_batch(
        self,
        memory_signal: dict[str, Any] | None = None,
        library_state: dict[str, Any] | None = None,
        batch_size: int = 40,
    ) -> list[CandidateFactor]:
        """Generate a batch of candidate factors using LLM guided by memory priors.

        Steps:
        1. Build prompt with memory signal injection.
        2. Call LLM to generate candidates.
        3. Parse and validate each candidate.
        4. Retry failed parses if any (frontier path under cascade).
        5. Return list of valid CandidateFactor objects.

        Parameters
        ----------
        memory_signal : dict or None
            Memory priors to inject into the prompt. Keys:
            - ``"recommended_directions"`` : list[str]
            - ``"forbidden_directions"`` : list[str]
            - ``"strategic_insights"`` : list[str]
            - ``"recent_rejections"`` : list[dict]
        library_state : dict or None
            Current library state. Keys:
            - ``"size"`` : int
            - ``"target_size"`` : int
            - ``"recent_admissions"`` : list[str]
            - ``"domain_saturation"`` : dict[str, float]
        batch_size : int
            Number of candidates to request per batch.

        Returns
        -------
        list[CandidateFactor]
            All valid candidate factors (those with successfully parsed
            expression trees).
        """
        memory_signal = memory_signal or {}
        library_state = library_state or {}

        self._generation_count += 1
        batch_id = self._generation_count

        logger.info(
            "Generating batch #%d: size=%d, provider=%s",
            batch_id,
            batch_size,
            self.llm_provider.provider_name,
        )

        # 1. Build prompts
        system_prompt = self.prompt_builder.system_prompt
        user_prompt = self.prompt_builder.build_user_prompt(
            memory_signal=memory_signal,
            library_state=library_state,
            batch_size=batch_size,
        )

        # 2. Call LLM (cascade: cheap draft first; parser decides escalate)
        t0 = time.monotonic()
        raw_output = self._call_llm(
            system_prompt,
            user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            force_frontier=False,
        )
        elapsed = time.monotonic() - t0
        logger.info("LLM response received in %.1fs (%d chars)", elapsed, len(raw_output))

        # 3. Parse output — free deterministic routing signal for cascade
        candidates, failed_lines = parse_llm_output(raw_output)

        valid = [c for c in candidates if c.is_valid]
        invalid = [c for c in candidates if not c.is_valid]

        logger.info(
            "Batch #%d initial parse: %d valid, %d invalid, %d unparseable lines",
            batch_id,
            len(valid),
            len(invalid),
            len(failed_lines),
        )

        # 4. Retry failed parses (always frontier under CascadeProvider)
        if failed_lines or invalid:
            retry_input = failed_lines + [c.formula for c in invalid if c.formula]
            retried = self._retry_failed_parses(retry_input, attempts=2)
            if retried:
                # Deduplicate by formula
                existing_formulas = {c.formula for c in valid}
                for c in retried:
                    if c.formula not in existing_formulas:
                        valid.append(c)
                        existing_formulas.add(c.formula)
                logger.info(
                    "Batch #%d after retry: %d total valid candidates",
                    batch_id,
                    len(valid),
                )

        # 5. Log summary
        if valid:
            categories = {}
            for c in valid:
                categories[c.category] = categories.get(c.category, 0) + 1
            logger.info(
                "Batch #%d categories: %s",
                batch_id,
                ", ".join(f"{k}={v}" for k, v in sorted(categories.items())),
            )

        return valid

    def _retry_failed_parses(
        self,
        failed: list[str],
        attempts: int = 2,
    ) -> list[CandidateFactor]:
        """Retry parsing failed outputs with a repair prompt.

        Asks the LLM to fix malformed formulas by providing the broken
        expressions and asking for corrected versions.  Under a cascade
        provider the repair path always uses the frontier model — the cheap
        draft already failed the deterministic DSL parse signal.

        Parameters
        ----------
        failed : list[str]
            Original text lines or formulas that failed to parse.
        attempts : int
            Max number of retry rounds.

        Returns
        -------
        list[CandidateFactor]
            Successfully parsed candidates from retries.
        """
        if not failed:
            return []

        # Limit retries to avoid excessive API calls
        failed = failed[:15]
        recovered: list[CandidateFactor] = []

        for attempt in range(1, attempts + 1):
            if not failed:
                break

            repair_prompt = (
                "The following factor formulas failed to parse. "
                "Fix each one so it uses ONLY valid operators and features "
                "from the library. Return them in the same numbered format:\n"
                "<number>. <name>: <corrected_formula>\n\n"
                "Broken formulas:\n"
                + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(failed))
                + "\n\nFix all syntax errors, unknown operators, and invalid "
                "feature names. Every formula must be a valid nested function "
                "call using only operators from the library."
            )

            try:
                raw = self._call_llm(
                    self.prompt_builder.system_prompt,
                    repair_prompt,
                    temperature=max(0.3, self.temperature - 0.3),
                    max_tokens=self.max_tokens,
                    force_frontier=True,
                )
            except Exception as e:
                logger.warning("Retry attempt %d failed: %s", attempt, e)
                break

            candidates, still_failed = parse_llm_output(raw)
            new_valid = [c for c in candidates if c.is_valid]
            recovered.extend(new_valid)

            # Update failed list for next attempt
            failed = still_failed + [c.formula for c in candidates if not c.is_valid]

            logger.debug(
                "Retry attempt %d: recovered %d, still failing %d",
                attempt,
                len(new_valid),
                len(failed),
            )

        return recovered
