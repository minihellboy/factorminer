"""Tests for provider-layer prompt caching payload construction.

These tests inspect the *constructed request payload* via
``build_request_kwargs`` — they do NOT call a live Anthropic/OpenAI API.
That is intentional: cache_control / prompt_cache_key shape is fully
determined client-side before the HTTP call.
"""

from __future__ import annotations

import hashlib

from factorminer.agent.debate import DebateGenerator
from factorminer.agent.llm_interface import (
    AnthropicProvider,
    MockProvider,
    OpenAIProvider,
    PrefixedCacheProvider,
    _prompt_cache_key,
    create_provider,
)
from factorminer.agent.prompt_builder import SYSTEM_PROMPT

BASE_SYSTEM = "You are FactorMiner.\n## OPERATORS\nCsRank, Delta, Neg"
SPECIALIST_A = BASE_SYSTEM + "\n\n## SPECIALIST DOMAIN DIRECTIVE\nMomentum focus."
SPECIALIST_B = BASE_SYSTEM + "\n\n## SPECIALIST DOMAIN DIRECTIVE\nVolatility focus."


def test_anthropic_cache_control_ephemeral_when_enabled():
    provider = AnthropicProvider(
        model="claude-sonnet-4-6",
        api_key="test-key",
        use_thinking=False,
        prompt_cache=True,
    )
    kwargs = provider.build_request_kwargs(
        system_prompt=SPECIALIST_A,
        user_prompt="Generate 2 factors.",
        temperature=0.5,
        max_tokens=256,
        cacheable_prefix=BASE_SYSTEM,
    )

    system = kwargs["system"]
    assert isinstance(system, list)
    assert len(system) >= 1
    first = system[0]
    assert first["type"] == "text"
    assert first["text"] == BASE_SYSTEM
    assert first["cache_control"] == {"type": "ephemeral"}
    # Specialist suffix lands in a separate uncached block.
    assert any(
        block.get("type") == "text" and "Momentum focus" in block.get("text", "")
        for block in system
    )


def test_anthropic_no_cache_control_when_disabled():
    provider = AnthropicProvider(
        model="claude-sonnet-4-6",
        api_key="test-key",
        use_thinking=False,
        prompt_cache=False,
    )
    kwargs = provider.build_request_kwargs(
        system_prompt=SPECIALIST_A,
        user_prompt="Generate 2 factors.",
        cacheable_prefix=BASE_SYSTEM,
    )
    assert kwargs["system"] == SPECIALIST_A
    assert isinstance(kwargs["system"], str)


def test_openai_prompt_cache_key_when_enabled():
    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="test-key",
        prompt_cache=True,
    )
    kwargs = provider.build_request_kwargs(
        system_prompt=SPECIALIST_A,
        user_prompt="Generate 2 factors.",
        cacheable_prefix=BASE_SYSTEM,
    )
    assert "prompt_cache_key" in kwargs
    expected = hashlib.sha256(BASE_SYSTEM.encode("utf-8")).hexdigest()
    assert kwargs["prompt_cache_key"] == expected
    assert kwargs["prompt_cache_key"] == _prompt_cache_key(BASE_SYSTEM)


def test_openai_no_prompt_cache_key_when_disabled():
    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="test-key",
        prompt_cache=False,
    )
    kwargs = provider.build_request_kwargs(
        system_prompt=SPECIALIST_A,
        user_prompt="Generate 2 factors.",
        cacheable_prefix=BASE_SYSTEM,
    )
    assert "prompt_cache_key" not in kwargs


def test_parallel_specialists_share_identical_cache_prefix():
    """Debate specialists must share one byte-stable cacheable prefix."""
    provider = AnthropicProvider(
        model="claude-test",
        api_key="test-key",
        use_thinking=False,
        prompt_cache=True,
    )

    payloads = []
    for system in (SPECIALIST_A, SPECIALIST_B):
        payloads.append(
            provider.build_request_kwargs(
                system_prompt=system,
                user_prompt="batch",
                cacheable_prefix=BASE_SYSTEM,
            )
        )

    cached_a = payloads[0]["system"][0]["text"]
    cached_b = payloads[1]["system"][0]["text"]
    assert cached_a == cached_b == BASE_SYSTEM
    assert payloads[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payloads[1]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_prefixed_cache_provider_injects_prefix():
    recorded: list[dict] = []

    class RecordingProvider(MockProvider):
        def generate(self, system_prompt, user_prompt, temperature=0.8, max_tokens=4096, *, cacheable_prefix=None):
            recorded.append(
                {
                    "system_prompt": system_prompt,
                    "cacheable_prefix": cacheable_prefix,
                }
            )
            return super().generate(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                cacheable_prefix=cacheable_prefix,
            )

    inner = RecordingProvider()
    wrapped = PrefixedCacheProvider(inner, cacheable_prefix=BASE_SYSTEM)
    wrapped.generate(SPECIALIST_A, "Generate 1 candidate factor.")
    wrapped.generate(SPECIALIST_B, "Generate 1 candidate factor.")

    assert len(recorded) == 2
    assert recorded[0]["cacheable_prefix"] == BASE_SYSTEM
    assert recorded[1]["cacheable_prefix"] == BASE_SYSTEM


def test_debate_generator_wraps_provider_with_shared_prefix():
    gen = DebateGenerator(llm_provider=MockProvider(), debate_config=None)
    assert isinstance(gen.llm_provider, PrefixedCacheProvider)
    # Default base is the module SYSTEM_PROMPT (or PromptBuilder default).
    assert gen._cacheable_prefix == SYSTEM_PROMPT
    assert gen.llm_provider.cacheable_prefix == SYSTEM_PROMPT


def test_create_provider_prompt_cache_default_on():
    p = create_provider({"provider": "openai", "model": "gpt-4o", "api_key": "k"})
    assert isinstance(p, OpenAIProvider)
    assert p.prompt_cache is True

    a = create_provider(
        {
            "provider": "anthropic",
            "model": "claude-test",
            "api_key": "k",
            "prompt_cache": False,
        }
    )
    assert isinstance(a, AnthropicProvider)
    assert a.prompt_cache is False


def test_mock_provider_accepts_cacheable_prefix_kwarg():
    """--mock must keep working with the extended generate() signature."""
    mock = MockProvider(prompt_cache=True)
    out = mock.generate(
        system_prompt="sys",
        user_prompt="Please generate 2 candidate factors.",
        cacheable_prefix="sys",
    )
    assert "1." in out
    assert mock.provider_name == "mock"
