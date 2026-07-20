"""Abstract LLM interface supporting multiple providers.

Provides a unified API for generating text completions across OpenAI,
Anthropic, Google (Gemini), OpenAI-compatible local engines, a cheap-first
cascade router, and a deterministic mock provider for testing.

Prompt caching (default ON)
---------------------------
* Anthropic: marks the stable system prefix with ``cache_control`` /
  ephemeral breakpoints so parallel debate specialists reuse one cached prefix.
* OpenAI: sets ``prompt_cache_key`` derived from the stable system prefix so
  automatic prefix caching routes related calls together.

Local cascade routing (default OFF)
-----------------------------------
``CascadeProvider`` drafts on a cheap local/OpenAI-compatible engine, then
uses FactorMiner's **deterministic DSL parser** as a free routing signal:
escalate to the frontier provider only when the draft fails to parse into
valid factor formulas.  This is viable specifically *because* FactorMiner
has a deterministic expression parser — a free, high-precision routing
signal most generic agentic-cost papers do not have.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class MissingAPIKeyError(ValueError):
    """Raised when a non-mock provider is used without a configured API key."""


class LLMProvider(ABC):
    """Abstract base for LLM text-generation providers."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        """Generate a text completion.

        Parameters
        ----------
        system_prompt : str
            System-level instructions (role, rules, operator library, etc.).
        user_prompt : str
            Per-iteration user prompt (memory signal, library state, etc.).
        temperature : float
            Sampling temperature; higher = more creative.
        max_tokens : int
            Maximum tokens in the response.
        cacheable_prefix : str or None
            Optional stable byte-prefix of ``system_prompt`` shared across
            related calls (e.g. debate specialists).  Providers that support
            prompt caching place the cache breakpoint on this prefix so
            parallel callers hit the same cached content.

        Returns
        -------
        str
            Raw text response from the model.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""


def _prompt_cache_key(prefix: str) -> str:
    """Stable OpenAI ``prompt_cache_key`` from a shared system prefix."""
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    # OpenAI accepts up to 64 chars; hex sha256 is exactly 64.
    return digest


def _split_system_for_cache(
    system_prompt: str,
    cacheable_prefix: str | None,
) -> tuple[str, str | None]:
    """Return (cached_block, uncached_suffix_or_none)."""
    if cacheable_prefix and system_prompt.startswith(cacheable_prefix):
        suffix = system_prompt[len(cacheable_prefix) :]
        # Keep leading newlines out of the uncached block noise.
        if suffix.startswith("\n"):
            suffix = suffix.lstrip("\n")
            if suffix:
                return cacheable_prefix, suffix
            return cacheable_prefix, None
        if suffix:
            return cacheable_prefix, suffix
        return cacheable_prefix, None
    return system_prompt, None


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-4o, etc.) with optional prompt caching."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        prompt_cache: bool = True,
        base_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.prompt_cache = bool(prompt_cache)
        # Optional base_url lets callers point at OpenAI-compatible endpoints
        # via the openai provider key. Prefer OpenAICompatibleProvider for
        # local engines so frontier keys are never forwarded.
        self.base_url = base_url
        self.timeout_s = float(timeout_s)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError(
                    "OpenAI provider requires an API key. "
                    "Set OPENAI_API_KEY or configure llm.api_key. "
                    "For local testing, use --mock."
                )
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install with: pip install openai"
                )
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout_s,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def build_request_kwargs(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Construct the chat.completions payload (no network I/O).

        Used by unit tests to assert cache-key shape without a live API call.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.prompt_cache:
            prefix = cacheable_prefix if cacheable_prefix is not None else system_prompt
            kwargs["prompt_cache_key"] = _prompt_cache_key(prefix)
        return kwargs

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        client = self._get_client()
        kwargs = self.build_request_kwargs(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            cacheable_prefix=cacheable_prefix,
        )
        logger.debug(
            "OpenAI request: model=%s temp=%.2f cache=%s",
            self.model,
            temperature,
            "prompt_cache_key" in kwargs,
        )
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        logger.debug("OpenAI response: %d chars", len(text))
        return text

    @property
    def provider_name(self) -> str:
        return f"openai/{self.model}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider with adaptive thinking + prompt caching."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        use_thinking: bool = True,
        effort: str = "max",
        prompt_cache: bool = True,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.use_thinking = use_thinking
        self.effort = effort
        self.prompt_cache = bool(prompt_cache)
        self.timeout_s = float(timeout_s)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError(
                    "Anthropic provider requires an API key. "
                    "Set ANTHROPIC_API_KEY or configure llm.api_key. "
                    "For local testing, use --mock."
                )
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install with: pip install anthropic"
                )
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout_s,
            )
        return self._client

    def build_request_kwargs(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1,
        max_tokens: int = 32000,
        *,
        cacheable_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Construct the messages.create payload (no network I/O).

        When ``prompt_cache`` is enabled the stable system prefix is emitted
        as a content block with ``cache_control: {type: ephemeral}`` so
        subsequent calls (and parallel debate specialists sharing the same
        prefix) reuse Anthropic's prompt cache.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
        }

        if self.prompt_cache:
            cached, suffix = _split_system_for_cache(system_prompt, cacheable_prefix)
            blocks: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": cached,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if suffix:
                blocks.append({"type": "text", "text": suffix})
            kwargs["system"] = blocks
        else:
            kwargs["system"] = system_prompt

        if self.use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["temperature"] = 1  # Required for thinking mode
            kwargs["output_config"] = {"effort": self.effort}
        else:
            kwargs["temperature"] = temperature

        return kwargs

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1,
        max_tokens: int = 32000,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        client = self._get_client()
        kwargs = self.build_request_kwargs(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            cacheable_prefix=cacheable_prefix,
        )
        logger.debug(
            "Anthropic request: model=%s thinking=%s effort=%s cache=%s",
            self.model,
            self.use_thinking,
            self.effort,
            self.prompt_cache,
        )

        response = client.messages.create(**kwargs)

        # Extract text from response, skipping thinking blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        text = "\n".join(text_parts) if text_parts else ""
        logger.debug("Anthropic response: %d chars", len(text))
        return text

    @property
    def provider_name(self) -> str:
        return f"anthropic/{self.model}"


class GoogleProvider(LLMProvider):
    """Google Gemini API provider (paper uses Gemini 3.0 Flash)."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        prompt_cache: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        # Gemini has no Anthropic-style cache_control; flag kept for parity
        # and is a harmless no-op on the request path.
        self.prompt_cache = bool(prompt_cache)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise MissingAPIKeyError(
                    "Google provider requires an API key. "
                    "Set GOOGLE_API_KEY or configure llm.api_key. "
                    "For local testing, use --mock."
                )
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required for GoogleProvider. "
                    "Install with: pip install google-generativeai"
                )
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                self.model,
                generation_config={"max_output_tokens": 8192},
            )
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        del cacheable_prefix  # unused — no Gemini cache_control equivalent here
        client = self._get_client()
        logger.debug("Google request: model=%s temp=%.2f", self.model, temperature)
        combined = f"{system_prompt}\n\n---\n\n{user_prompt}"
        response = client.generate_content(
            combined,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = response.text if response.text else ""
        logger.debug("Google response: %d chars", len(text))
        return text

    @property
    def provider_name(self) -> str:
        return f"google/{self.model}"


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible HTTP provider for local engines (Ollama/vLLM/SGLang).

    Security
    --------
    * ``base_url`` MUST come only from local YAML / trusted config — never
      from remote or user-controlled input (SSRF consideration: a hostile
      base_url could make the process issue HTTP to internal addresses).
    * NEVER forward a real frontier-provider API key (``OPENAI_API_KEY`` /
      Anthropic / Google) to a custom ``base_url``.  Local engines get an
      explicit local key (default ``"local"``) supplied in config only.
    * Every request uses an explicit timeout.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://127.0.0.1:11434/v1",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        prompt_cache: bool = False,
    ) -> None:
        # SECURITY (SSRF): base_url is trusted local config only. Do not accept
        # URLs from prompts, remote manifests, or untrusted tool input.
        if not base_url or not isinstance(base_url, str):
            raise ValueError("OpenAICompatibleProvider requires a base_url from local config")
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Intentionally do NOT fall back to OPENAI_API_KEY — never forward
        # frontier credentials to a custom/local endpoint.
        self.api_key = api_key if api_key is not None else "local"
        self.timeout_s = float(timeout_s)
        self.prompt_cache = bool(prompt_cache)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAICompatibleProvider. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_s,
            )
        return self._client

    def build_request_kwargs(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Construct the chat.completions payload (no network I/O)."""
        del cacheable_prefix
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        client = self._get_client()
        kwargs = self.build_request_kwargs(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            cacheable_prefix=cacheable_prefix,
        )
        logger.debug(
            "OpenAI-compatible request: model=%s base_url=%s temp=%.2f timeout=%.1fs",
            self.model,
            self.base_url,
            temperature,
            self.timeout_s,
        )
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        logger.debug("OpenAI-compatible response: %d chars", len(text))
        return text

    @property
    def provider_name(self) -> str:
        return f"openai_compatible/{self.model}"


class CascadeProvider(LLMProvider):
    """Cheap-first cascade: local draft → DSL-parse signal → frontier escalate.

    Creative AI / cost-routing angle
    --------------------------------
    This cascade is viable specifically *because* FactorMiner's factor DSL
    has a **deterministic parser** (``parse_llm_output`` / ``try_parse``).
    Parse success/failure is a free, high-precision routing signal that most
    generic agentic-cost papers do not have: if a cheap local model already
    emits well-formed formulas, there is no need to spend frontier tokens;
    only unparseable drafts escalate.

    Parameters
    ----------
    draft :
        Cheap local / small-model provider (e.g. ``OpenAICompatibleProvider``).
    frontier :
        Expensive frontier provider used only on parse failure (or when the
        draft returns zero valid candidates).
    escalate_on_parse_failure :
        When True (default), any fully-failed parse escalates.  Partial
        success (at least one valid candidate) is accepted without escalate.
    """

    def __init__(
        self,
        draft: LLMProvider,
        frontier: LLMProvider,
        *,
        escalate_on_parse_failure: bool = True,
    ) -> None:
        self.draft = draft
        self.frontier = frontier
        self.escalate_on_parse_failure = bool(escalate_on_parse_failure)
        self.draft_calls = 0
        self.frontier_calls = 0
        self.escalations = 0

    def _draft_is_acceptable(self, raw_text: str) -> bool:
        """Return True when the deterministic DSL parser accepts the draft."""
        # Local import keeps the provider module free of hard agent cycles
        # at import time for callers that only need the ABC / mock.
        from factorminer.agent.output_parser import parse_llm_output

        candidates, failed_lines = parse_llm_output(raw_text)
        valid = [c for c in candidates if c.is_valid]
        if valid:
            return True
        if not self.escalate_on_parse_failure:
            return bool(candidates) and not failed_lines
        return False

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        self.draft_calls += 1
        draft_text = self.draft.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            cacheable_prefix=cacheable_prefix,
        )
        if self._draft_is_acceptable(draft_text):
            logger.debug(
                "Cascade: draft accepted (provider=%s, %d chars)",
                self.draft.provider_name,
                len(draft_text),
            )
            return draft_text

        self.escalations += 1
        self.frontier_calls += 1
        logger.info(
            "Cascade: escalating to frontier after draft parse failure "
            "(draft=%s frontier=%s)",
            self.draft.provider_name,
            self.frontier.provider_name,
        )
        return self.frontier.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            cacheable_prefix=cacheable_prefix,
        )

    def generate_frontier(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        """Force a frontier call (used by factor-generator repair path)."""
        self.frontier_calls += 1
        return self.frontier.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            cacheable_prefix=cacheable_prefix,
        )

    @property
    def provider_name(self) -> str:
        return f"cascade[{self.draft.provider_name}->{self.frontier.provider_name}]"


class MockProvider(LLMProvider):
    """Deterministic provider for testing without API calls.

    Returns predefined factor formulas that exercise diverse operator
    combinations.  Useful for unit tests and integration testing.
    Prompt-cache / cascade flags are accepted as no-ops so ``--mock``
    keeps working with caching/cascade config toggles enabled.
    """

    MOCK_FACTORS = [
        ("momentum_reversal", "Neg(CsRank(Delta($close, 5)))"),
        ("volume_surprise", "CsZScore(Div(Sub($volume, Mean($volume, 20)), Std($volume, 20)))"),
        ("price_range_ratio", "Div(Sub($high, $low), Add($high, $low))"),
        ("vwap_deviation", "CsRank(Div(Sub($close, $vwap), $vwap))"),
        ("return_skew", "Neg(Skew($returns, 20))"),
        ("intraday_momentum", "CsRank(Div(Sub($close, $open), Sub($high, $low)))"),
        ("volume_price_corr", "Neg(Corr($volume, $close, 10))"),
        ("amt_acceleration", "CsZScore(Delta(Mean($amt, 5), 5))"),
        ("close_high_ratio", "CsRank(Sub(Div($close, TsMax($high, 20)), 1))"),
        ("smooth_return", "Neg(CsRank(EMA($returns, 10)))"),
        ("volatility_ratio", "Div(Std($returns, 5), Std($returns, 20))"),
        ("mean_reversion", "Neg(CsZScore(Div(Sub($close, SMA($close, 20)), SMA($close, 20))))"),
        ("volume_trend", "CsRank(TsLinRegSlope($volume, 20))"),
        ("price_position", "CsRank(Div(Sub($close, TsMin($close, 20)), Sub(TsMax($close, 20), TsMin($close, 20))))"),
        ("amt_volume_div", "CsRank(Neg(Corr(CsRank($amt), CsRank($volume), 10)))"),
        ("weighted_return", "CsZScore(WMA($returns, 10))"),
        ("high_low_decay", "Neg(Decay(Div(Sub($high, $low), $close), 10))"),
        ("residual_vol", "CsRank(Std(Resid($close, $volume, 20), 10))"),
        ("open_gap", "CsZScore(Div(Sub($open, Delay($close, 1)), Delay($close, 1)))"),
        ("log_turnover", "Neg(CsRank(Log(Div($amt, $volume))))"),
        ("beta_momentum", "CsRank(Mul(Beta($returns, $volume, 20), Delta($close, 10)))"),
        ("rank_reversal", "Neg(CsRank(Sum($returns, 5)))"),
        ("kurtosis_signal", "CsZScore(Neg(Kurt($returns, 20)))"),
        ("vwap_trend", "CsRank(TsLinRegSlope(Div($close, $vwap), 20))"),
        ("adaptive_mean", "CsRank(Div(Sub($close, KAMA($close, 10)), Std($close, 10)))"),
        ("cumulative_flow", "CsZScore(CsRank(Delta(CumSum(Mul($volume, Sign(Delta($close, 1)))), 5)))"),
        ("range_breakout", "CsRank(Div(Sub($close, TsMin($low, 10)), Std($close, 10)))"),
        ("hull_deviation", "Neg(CsRank(Div(Sub($close, HMA($close, 20)), $close)))"),
        ("conditional_vol", "CsZScore(IfElse(Greater($returns, 0), Std($returns, 10), Neg(Std($returns, 10))))"),
        ("dema_crossover", "CsRank(Sub(DEMA($close, 5), DEMA($close, 20)))"),
        ("ts_rank_volume", "Neg(CsRank(TsRank($volume, 20)))"),
        ("median_price", "CsZScore(Div(Sub($close, Median($close, 20)), Median($close, 20)))"),
        ("argmax_timing", "CsRank(Neg(TsArgMax($close, 20)))"),
        ("log_return_sum", "Neg(CsRank(Sum(LogReturn($close, 1), 10)))"),
        ("price_cov", "CsZScore(Neg(Cov($close, $volume, 20)))"),
        ("inv_volatility", "CsRank(Inv(Std($returns, 20)))"),
        ("squared_return", "Neg(CsRank(Mean(Square($returns), 10)))"),
        ("abs_return_ratio", "CsRank(Div(Abs(Delta($close, 1)), Mean(Abs(Delta($close, 1)), 20)))"),
        ("quantile_signal", "CsZScore(Quantile($returns, 20, 0.75))"),
        ("neutralized_mom", "CsNeutralize(Delta($close, 10))"),
    ]

    def __init__(self, cycle: bool = True, prompt_cache: bool = True) -> None:
        self._cycle = cycle
        self._call_count = 0
        # Accepted so caching config toggles are harmless under --mock.
        self.prompt_cache = bool(prompt_cache)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        del system_prompt, temperature, max_tokens, cacheable_prefix
        # Parse batch_size from user_prompt if present
        batch_size = 40
        for line in user_prompt.split("\n"):
            if "generate" in line.lower() and "candidate" in line.lower():
                for word in line.split():
                    if word.isdigit():
                        batch_size = int(word)
                        break

        batch_size = min(batch_size, len(self.MOCK_FACTORS))

        start = self._call_count * batch_size
        if self._cycle:
            indices = [
                (start + i) % len(self.MOCK_FACTORS)
                for i in range(batch_size)
            ]
        else:
            indices = list(range(min(batch_size, len(self.MOCK_FACTORS))))

        self._call_count += 1

        lines = []
        for idx, factor_idx in enumerate(indices, 1):
            name, formula = self.MOCK_FACTORS[factor_idx]
            lines.append(f"{idx}. {name}: {formula}")

        return "\n".join(lines)

    @property
    def provider_name(self) -> str:
        return "mock"


class PrefixedCacheProvider(LLMProvider):
    """Wrapper that injects a stable ``cacheable_prefix`` on every call.

    Debate runs multiple specialists in parallel against the *same* base
    system prompt.  Wrapping the shared provider with this class guarantees
    every specialist call passes an identical byte-stable prefix for the
    provider-layer cache breakpoint, even when specialist suffixes differ.
    """

    def __init__(self, inner: LLMProvider, cacheable_prefix: str) -> None:
        self._inner = inner
        self._cacheable_prefix = cacheable_prefix

    @property
    def cacheable_prefix(self) -> str:
        return self._cacheable_prefix

    @property
    def inner(self) -> LLMProvider:
        return self._inner

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        prefix = cacheable_prefix if cacheable_prefix is not None else self._cacheable_prefix
        return self._inner.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            cacheable_prefix=prefix,
        )

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "mock": MockProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "local": OpenAICompatibleProvider,
}


def _build_single_provider(config: dict[str, Any]) -> LLMProvider:
    """Instantiate one non-cascade provider from a config dict."""
    provider_name = config.get("provider", "mock")
    cls = _PROVIDER_MAP.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Available: {sorted(_PROVIDER_MAP.keys())}"
        )

    prompt_cache = bool(config.get("prompt_cache", True))
    kwargs: dict[str, Any] = {}

    if provider_name == "mock":
        kwargs["prompt_cache"] = prompt_cache
        return cls(**kwargs)

    if "model" in config:
        kwargs["model"] = config["model"]

    if provider_name in ("openai_compatible", "local"):
        # SECURITY (SSRF): base_url exclusively from local config dict.
        base_url = config.get("base_url") or config.get("draft_base_url")
        if not base_url:
            raise ValueError(
                "openai_compatible/local provider requires llm.base_url "
                "(or cascade.draft_base_url) from local YAML config"
            )
        kwargs["base_url"] = base_url
        # Never pull OPENAI_API_KEY — only an explicit local key.
        if "api_key" in config and config["api_key"]:
            kwargs["api_key"] = config["api_key"]
        else:
            kwargs["api_key"] = config.get("local_api_key", "local")
        if "timeout_s" in config:
            kwargs["timeout_s"] = config["timeout_s"]
        kwargs["prompt_cache"] = bool(config.get("prompt_cache", False))
        return cls(**kwargs)

    # Frontier providers
    if "api_key" in config and config["api_key"]:
        kwargs["api_key"] = config["api_key"]
    kwargs["prompt_cache"] = prompt_cache

    if provider_name == "openai":
        # Optional base_url on OpenAIProvider is supported but discouraged for
        # local engines (use openai_compatible so keys are not forwarded).
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        if "timeout_s" in config:
            kwargs["timeout_s"] = config["timeout_s"]
    elif provider_name == "anthropic":
        if "use_thinking" in config:
            kwargs["use_thinking"] = config["use_thinking"]
        if "effort" in config:
            kwargs["effort"] = config["effort"]
        if "timeout_s" in config:
            kwargs["timeout_s"] = config["timeout_s"]

    logger.info(
        "Creating LLM provider: %s (kwargs=%s)",
        provider_name,
        list(kwargs.keys()),
    )
    return cls(**kwargs)


def create_provider(config: dict[str, Any]) -> LLMProvider:
    """Factory function to instantiate an LLM provider from config.

    Parameters
    ----------
    config : dict
        Must contain ``"provider"`` key (one of "openai", "anthropic",
        "google", "mock", "openai_compatible", "local").  Additional keys:
        - ``"model"`` : model identifier
        - ``"api_key"`` : API key (overrides env var) — frontier only
        - ``"prompt_cache"`` : bool, default True (no-op under mock)
        - ``"base_url"`` : local OpenAI-compatible endpoint (YAML only)
        - ``"timeout_s"`` : request timeout seconds
        - ``"cascade"`` : optional nested dict enabling cheap-first routing:
            ``enabled``, ``draft_provider``, ``draft_model``,
            ``draft_base_url``, ``draft_api_key``, ``timeout_s``,
            ``escalate_on_parse_failure``

    Returns
    -------
    LLMProvider
    """
    cascade_cfg = config.get("cascade") or {}
    cascade_enabled = bool(
        config.get("cascade_enabled", False)
        or (isinstance(cascade_cfg, dict) and cascade_cfg.get("enabled", False))
    )

    if cascade_enabled:
        # Frontier = the configured primary provider, without cascade recursion.
        frontier_cfg = {
            k: v
            for k, v in config.items()
            if k not in ("cascade", "cascade_enabled")
        }
        # Strip local-only keys that must not leak onto the frontier client.
        frontier_cfg.pop("base_url", None)
        frontier = _build_single_provider(frontier_cfg)

        draft_provider = cascade_cfg.get("draft_provider", "openai_compatible")
        draft_cfg: dict[str, Any] = {
            "provider": draft_provider,
            "model": cascade_cfg.get(
                "draft_model",
                cascade_cfg.get("model", "llama3.2"),
            ),
            # SECURITY (SSRF): draft base_url only from local cascade YAML.
            "base_url": cascade_cfg.get(
                "draft_base_url",
                cascade_cfg.get("base_url", "http://127.0.0.1:11434/v1"),
            ),
            # Explicit local key only — never the frontier api_key.
            "api_key": cascade_cfg.get(
                "draft_api_key",
                cascade_cfg.get("local_api_key", "local"),
            ),
            "timeout_s": cascade_cfg.get(
                "timeout_s",
                config.get("timeout_s", 60.0),
            ),
            "prompt_cache": False,
        }
        draft = _build_single_provider(draft_cfg)
        escalate = bool(cascade_cfg.get("escalate_on_parse_failure", True))
        logger.info(
            "Creating cascade provider: draft=%s frontier=%s",
            draft.provider_name,
            frontier.provider_name,
        )
        return CascadeProvider(
            draft=draft,
            frontier=frontier,
            escalate_on_parse_failure=escalate,
        )

    return _build_single_provider(config)
