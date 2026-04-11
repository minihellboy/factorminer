"""Main factor generation agent using LLM guided by memory priors.

Orchestrates the prompt construction, LLM invocation, output parsing,
and retry logic for a single batch of factor candidates.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from factorminer.agent.llm_interface import LLMProvider
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
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_builder: PromptBuilder | None = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> None:
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._generation_count = 0

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
        4. Retry failed parses if any.
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

        # 2. Call LLM
        t0 = time.monotonic()
        raw_output = self.llm_provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed = time.monotonic() - t0
        logger.info("LLM response received in %.1fs (%d chars)", elapsed, len(raw_output))

        # 3. Parse output
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

        # 4. Retry failed parses
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
        expressions and asking for corrected versions.

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
                raw = self.llm_provider.generate(
                    system_prompt=self.prompt_builder.system_prompt,
                    user_prompt=repair_prompt,
                    temperature=max(0.3, self.temperature - 0.3),
                    max_tokens=self.max_tokens,
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
