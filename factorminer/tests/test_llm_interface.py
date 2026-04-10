"""Tests for LLM provider configuration failures."""

from __future__ import annotations

import pytest

from factorminer.agent.llm_interface import (
    AnthropicProvider,
    GoogleProvider,
    MissingAPIKeyError,
    OpenAIProvider,
)


@pytest.mark.parametrize(
    ("provider_cls", "env_name"),
    [
        (OpenAIProvider, "OPENAI_API_KEY"),
        (AnthropicProvider, "ANTHROPIC_API_KEY"),
        (GoogleProvider, "GOOGLE_API_KEY"),
    ],
)
def test_provider_without_api_key_fails_fast(monkeypatch, provider_cls, env_name):
    monkeypatch.delenv(env_name, raising=False)
    provider = provider_cls(api_key="")

    with pytest.raises(MissingAPIKeyError):
        provider._get_client()
