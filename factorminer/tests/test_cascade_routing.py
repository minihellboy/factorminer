"""Tests for cheap-first local→frontier cascade routing.

The cascade uses FactorMiner's deterministic DSL parser
(``parse_llm_output``) as a free routing signal: escalate to the frontier
provider only when the local draft fails to produce any valid formula.
This is viable specifically because FactorMiner has a deterministic
expression parser — a free signal most generic agentic-cost papers lack.
"""

from __future__ import annotations

import pytest

from factorminer.agent.factor_generator import FactorGenerator
from factorminer.agent.llm_interface import (
    CascadeProvider,
    MockProvider,
    OpenAICompatibleProvider,
    PrefixedCacheProvider,
    create_provider,
)
from factorminer.agent.prompt_builder import PromptBuilder


class StubProvider:
    """Minimal LLMProvider double that records calls and returns scripted text."""

    def __init__(self, responses: list[str], name: str = "stub") -> None:
        self._responses = list(responses)
        self._name = name
        self.calls: list[dict] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "cacheable_prefix": cacheable_prefix,
            }
        )
        if not self._responses:
            return ""
        # Cycle the last response if exhausted.
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    @property
    def provider_name(self) -> str:
        return self._name


VALID_DRAFT = (
    "1. mom_rev: Neg(CsRank(Delta($close, 5)))\n"
    "2. vol_ratio: Div(Std($returns, 5), Std($returns, 20)))"
)

# Second formula intentionally has an extra ')', but first is valid — draft
# still acceptable because at least one candidate parses.
VALID_DRAFT_CLEAN = (
    "1. mom_rev: Neg(CsRank(Delta($close, 5)))\n"
    "2. vol_ratio: Div(Std($returns, 5), Std($returns, 20))"
)

UNPARSEABLE_DRAFT = (
    "Sure! Here are some ideas:\n"
    "- momentum something with close prices maybe\n"
    "I hope this helps!"
)

# Line-shaped but DSL-invalid formulas: parser yields invalid candidates
# (triggers FactorGenerator repair) and zero valid (triggers cascade escalate).
BROKEN_FORMULAS = (
    "1. bad_alpha: FooBar($close, 5)\n"
    "2. worse_beta: NotReal(1, 2)"
)

FRONTIER_VALID = (
    "1. frontier_mom: Neg(CsRank(Delta($close, 10)))\n"
    "2. frontier_vol: CsZScore(Std($returns, 20))"
)


def test_cascade_accepts_parseable_draft_without_escalation():
    draft = StubProvider([VALID_DRAFT_CLEAN], name="local-draft")
    frontier = StubProvider([FRONTIER_VALID], name="frontier")
    cascade = CascadeProvider(draft=draft, frontier=frontier)

    out = cascade.generate("sys", "Generate 2 candidate factors.")

    assert out == VALID_DRAFT_CLEAN
    assert cascade.draft_calls == 1
    assert cascade.frontier_calls == 0
    assert cascade.escalations == 0
    assert len(draft.calls) == 1
    assert len(frontier.calls) == 0


def test_cascade_escalates_only_on_parse_failure():
    draft = StubProvider([UNPARSEABLE_DRAFT], name="local-draft")
    frontier = StubProvider([FRONTIER_VALID], name="frontier")
    cascade = CascadeProvider(draft=draft, frontier=frontier)

    out = cascade.generate("sys", "Generate 2 candidate factors.")

    assert out == FRONTIER_VALID
    assert cascade.draft_calls == 1
    assert cascade.frontier_calls == 1
    assert cascade.escalations == 1
    assert len(draft.calls) == 1
    assert len(frontier.calls) == 1


def test_cascade_escalation_count_across_mixed_calls():
    """Two unparseable drafts + one good draft => exactly two escalations."""
    draft = StubProvider(
        [UNPARSEABLE_DRAFT, VALID_DRAFT_CLEAN, UNPARSEABLE_DRAFT],
        name="local-draft",
    )
    frontier = StubProvider(
        [FRONTIER_VALID, FRONTIER_VALID],
        name="frontier",
    )
    cascade = CascadeProvider(draft=draft, frontier=frontier)

    r1 = cascade.generate("sys", "u1")
    r2 = cascade.generate("sys", "u2")
    r3 = cascade.generate("sys", "u3")

    assert r1 == FRONTIER_VALID
    assert r2 == VALID_DRAFT_CLEAN
    assert r3 == FRONTIER_VALID
    assert cascade.draft_calls == 3
    assert cascade.frontier_calls == 2
    assert cascade.escalations == 2
    assert len(frontier.calls) == 2


def test_cascade_generate_frontier_forces_escalation():
    draft = StubProvider([VALID_DRAFT_CLEAN], name="local-draft")
    frontier = StubProvider([FRONTIER_VALID], name="frontier")
    cascade = CascadeProvider(draft=draft, frontier=frontier)

    out = cascade.generate_frontier("sys", "repair these formulas")
    assert out == FRONTIER_VALID
    assert cascade.frontier_calls == 1
    assert cascade.draft_calls == 0
    assert len(draft.calls) == 0


def test_factor_generator_repair_uses_frontier_under_cascade():
    """Repair path must force frontier under CascadeProvider."""
    # Draft emits line-shaped but DSL-invalid formulas:
    # - cascade escalates (zero valid)
    # - FactorGenerator still sees invalid candidates → repair
    # - repair uses generate_frontier (force_frontier=True)
    draft = StubProvider([BROKEN_FORMULAS], name="local-draft")
    frontier = StubProvider(
        [BROKEN_FORMULAS, FRONTIER_VALID],  # escalate + repair
        name="frontier",
    )
    cascade = CascadeProvider(draft=draft, frontier=frontier)

    gen = FactorGenerator(
        llm_provider=cascade,
        prompt_builder=PromptBuilder(system_prompt="You are a factor miner."),
        temperature=0.5,
        max_tokens=512,
    )
    result = gen.generate_batch(batch_size=2)

    assert cascade.draft_calls == 1
    # Initial generate escalates (1) + repair force_frontier (1+)
    assert cascade.frontier_calls >= 2
    assert len(frontier.calls) >= 2
    assert any(c.is_valid for c in result)


def test_factor_generator_repair_uses_frontier_when_cascade_is_wrapped():
    """Repair path must still force frontier when the cascade is wrapped
    by PrefixedCacheProvider (the DebateGenerator/specialist wiring).

    Regression test: a prior bug used
    ``isinstance(self.llm_provider, CascadeProvider)`` directly, which is
    False once wrapped -- ``force_frontier`` silently became a no-op and
    repair re-entered the cheap local draft again instead of escalating,
    exactly when quality-safety repair matters most.
    """
    draft = StubProvider([BROKEN_FORMULAS], name="local-draft")
    frontier = StubProvider(
        [BROKEN_FORMULAS, FRONTIER_VALID],  # escalate + repair
        name="frontier",
    )
    cascade = CascadeProvider(draft=draft, frontier=frontier)
    wrapped = PrefixedCacheProvider(cascade, cacheable_prefix="You are a factor miner.")

    gen = FactorGenerator(
        llm_provider=wrapped,
        prompt_builder=PromptBuilder(system_prompt="You are a factor miner."),
        temperature=0.5,
        max_tokens=512,
    )
    result = gen.generate_batch(batch_size=2)

    # Repair must reach CascadeProvider.generate_frontier through the
    # wrapper, not silently fall back to cascade.generate() (cheap-first)
    # again on repair.
    assert cascade.frontier_calls >= 2
    assert any(c.is_valid for c in result)


def test_unwrap_cascade_provider_finds_inner_cascade():
    """Direct unit test of the unwrap helper across wrapper chains."""
    draft = StubProvider([], name="d")
    frontier = StubProvider([], name="f")
    cascade = CascadeProvider(draft=draft, frontier=frontier)
    wrapped = PrefixedCacheProvider(cascade, cacheable_prefix="base")
    double_wrapped = PrefixedCacheProvider(wrapped, cacheable_prefix="base")

    assert FactorGenerator._unwrap_cascade_provider(cascade) is cascade
    assert FactorGenerator._unwrap_cascade_provider(wrapped) is cascade
    assert FactorGenerator._unwrap_cascade_provider(double_wrapped) is cascade
    assert FactorGenerator._unwrap_cascade_provider(draft) is None


def test_create_provider_cascade_from_config():
    provider = create_provider(
        {
            "provider": "mock",
            "cascade": {
                "enabled": True,
                "draft_provider": "openai_compatible",
                "draft_model": "tiny-local",
                "draft_base_url": "http://127.0.0.1:9999/v1",
                "draft_api_key": "local-only-key",
                "timeout_s": 5.0,
            },
        }
    )
    assert isinstance(provider, CascadeProvider)
    assert isinstance(provider.draft, OpenAICompatibleProvider)
    assert provider.draft.base_url == "http://127.0.0.1:9999/v1"
    # Must NOT forward a frontier env key — explicit local key only.
    assert provider.draft.api_key == "local-only-key"
    assert isinstance(provider.frontier, MockProvider)


def test_openai_compatible_does_not_read_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-frontier-secret")
    local = OpenAICompatibleProvider(
        model="llama",
        base_url="http://127.0.0.1:11434/v1",
    )
    assert local.api_key == "local"
    assert local.api_key != "sk-frontier-secret"


def test_openai_compatible_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleProvider(model="x", base_url="")  # type: ignore[arg-type]


def test_create_provider_cascade_default_off():
    p = create_provider({"provider": "mock"})
    assert isinstance(p, MockProvider)
    assert not isinstance(p, CascadeProvider)


def test_cascade_provider_name():
    draft = StubProvider([], name="d")
    frontier = StubProvider([], name="f")
    cascade = CascadeProvider(draft=draft, frontier=frontier)
    assert cascade.provider_name == "cascade[d->f]"
