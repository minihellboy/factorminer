"""Tests for LLM provider configuration failures."""

from __future__ import annotations

import pytest

from factorminer.agent.llm_interface import (
    AnthropicProvider,
    DeepSeekProvider,
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
        (DeepSeekProvider, "DEEPSEEK_API_KEY"),
    ],
)
def test_provider_without_api_key_fails_fast(monkeypatch, provider_cls, env_name):
    monkeypatch.delenv(env_name, raising=False)
    provider = provider_cls(api_key="")

    with pytest.raises(MissingAPIKeyError):
        provider._get_client()


def test_deepseek_factory_uses_own_credential_and_fixed_endpoint(monkeypatch):
    from factorminer.agent.llm_interface import create_provider
    from factorminer.utils.config import load_config

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    cfg = load_config("factorminer/configs/research_actions_deepseek.yaml")
    provider = create_provider({**vars(cfg.llm), "base_url": "https://untrusted.invalid"})
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.api_key == "test-deepseek"
    assert provider.provider_name == "deepseek/deepseek-flash"
    assert provider.build_request_kwargs("system", "user")["model"] == "deepseek-flash"
    assert cfg.mining.max_iterations == 0
    assert cfg.research.planner.enabled
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(MissingAPIKeyError):
        create_provider({"provider": "deepseek"})._get_client()
