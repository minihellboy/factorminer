"""Abstract LLM interface supporting multiple providers.

Provides a unified API for generating text completions across OpenAI,
Anthropic, Google (Gemini), and a deterministic mock provider for testing.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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

        Returns
        -------
        str
            Raw text response from the model.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-4o, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
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
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> str:
        client = self._get_client()
        logger.debug("OpenAI request: model=%s temp=%.2f", self.model, temperature)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        logger.debug("OpenAI response: %d chars", len(text))
        return text

    @property
    def provider_name(self) -> str:
        return f"openai/{self.model}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider with adaptive thinking support."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        use_thinking: bool = True,
        effort: str = "max",
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.use_thinking = use_thinking
        self.effort = effort
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
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1,
        max_tokens: int = 32000,
    ) -> str:
        client = self._get_client()
        logger.debug("Anthropic request: model=%s thinking=%s effort=%s",
                      self.model, self.use_thinking, self.effort)

        kwargs: dict = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
        }

        if self.use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["temperature"] = 1  # Required for thinking mode
            kwargs["output_config"] = {"effort": self.effort}
        else:
            kwargs["temperature"] = temperature

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
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
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
    ) -> str:
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


class MockProvider(LLMProvider):
    """Deterministic provider for testing without API calls.

    Returns predefined factor formulas that exercise diverse operator
    combinations.  Useful for unit tests and integration testing.
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

    def __init__(self, cycle: bool = True) -> None:
        self._cycle = cycle
        self._call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> str:
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_MAP: Dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "mock": MockProvider,
}


def create_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to instantiate an LLM provider from config.

    Parameters
    ----------
    config : dict
        Must contain ``"provider"`` key (one of "openai", "anthropic",
        "google", "mock").  Additional keys are passed as kwargs to the
        provider constructor:
        - ``"model"`` : model identifier
        - ``"api_key"`` : API key (overrides env var)

    Returns
    -------
    LLMProvider
    """
    provider_name = config.get("provider", "mock")
    cls = _PROVIDER_MAP.get(provider_name)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Available: {sorted(_PROVIDER_MAP.keys())}"
        )

    kwargs: Dict[str, Any] = {}
    if "model" in config and provider_name != "mock":
        kwargs["model"] = config["model"]
    if "api_key" in config and provider_name != "mock":
        kwargs["api_key"] = config["api_key"]

    logger.info("Creating LLM provider: %s (kwargs=%s)", provider_name, list(kwargs.keys()))
    return cls(**kwargs)
