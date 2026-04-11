"""LLM agent integration for factor generation."""

from factorminer.agent.critic import CriticAgent, CriticScore
from factorminer.agent.debate import (
    DebateConfig,
    DebateGenerator,
    DebateMemory,
    DebateOrchestrator,
    DebateResult,
)
from factorminer.agent.factor_generator import FactorGenerator
from factorminer.agent.llm_interface import (
    AnthropicProvider,
    GoogleProvider,
    LLMProvider,
    MockProvider,
    OpenAIProvider,
    create_provider,
)
from factorminer.agent.output_parser import CandidateFactor, parse_llm_output
from factorminer.agent.prompt_builder import (
    PromptBuilder,
    build_critic_scoring_prompt,
    build_debate_synthesis_prompt,
    build_specialist_prompt,
)
from factorminer.agent.specialists import (
    DEFAULT_SPECIALISTS,
    LIQUIDITY_SPECIALIST,
    MOMENTUM_SPECIALIST,
    REGIME_SPECIALIST,
    SPECIALIST_CONFIGS,
    VOLATILITY_SPECIALIST,
    SpecialistAgent,
    SpecialistConfig,
    SpecialistDomainMemory,
    SpecialistPromptBuilder,
)

__all__ = [
    # Generator
    "FactorGenerator",
    # LLM providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MockProvider",
    "create_provider",
    # Parsing
    "CandidateFactor",
    "parse_llm_output",
    # Prompt
    "PromptBuilder",
    "build_specialist_prompt",
    "build_critic_scoring_prompt",
    "build_debate_synthesis_prompt",
    # Specialists
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
    # Critic
    "CriticAgent",
    "CriticScore",
    # Debate
    "DebateGenerator",
    "DebateConfig",
    "DebateOrchestrator",
    "DebateResult",
    "DebateMemory",
]
