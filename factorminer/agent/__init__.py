"""LLM generation APIs exposed without initializing providers eagerly."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_EXPORTS = {
    "FactorGenerator": "factorminer.agent.factor_generator",
    **{
        name: "factorminer.agent.llm_interface"
        for name in (
            "LLMProvider",
            "OpenAIProvider",
            "OpenAICompatibleProvider",
            "AnthropicProvider",
            "GoogleProvider",
            "CascadeProvider",
            "PrefixedCacheProvider",
            "MockProvider",
            "create_provider",
        )
    },
    "CandidateFactor": "factorminer.agent.output_parser",
    "parse_llm_output": "factorminer.agent.output_parser",
    **{
        name: "factorminer.agent.prompt_builder"
        for name in (
            "PromptBuilder",
            "build_specialist_prompt",
            "build_critic_scoring_prompt",
            "build_debate_synthesis_prompt",
        )
    },
    **{
        name: "factorminer.agent.specialists"
        for name in (
            "SpecialistConfig",
            "SpecialistAgent",
            "SpecialistDomainMemory",
            "SpecialistPromptBuilder",
            "MOMENTUM_SPECIALIST",
            "VOLATILITY_SPECIALIST",
            "LIQUIDITY_SPECIALIST",
            "REGIME_SPECIALIST",
            "DEFAULT_SPECIALISTS",
            "SPECIALIST_CONFIGS",
        )
    },
    "CriticAgent": "factorminer.agent.critic",
    "CriticScore": "factorminer.agent.critic",
    **{
        name: "factorminer.agent.debate"
        for name in (
            "DebateGenerator",
            "DebateConfig",
            "DebateOrchestrator",
            "DebateResult",
            "DebateMemory",
        )
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
