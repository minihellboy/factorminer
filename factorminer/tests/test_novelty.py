"""Novelty controls: skip formulas a campaign already evaluated, show the model what it tried."""

from __future__ import annotations

import numpy as np

from factorminer.agent.llm_interface import MockProvider
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.core.ralph_loop import RalphLoop
from factorminer.utils.config import load_config

FORMULAS = [
    "Neg(CsRank(Delta($close, 5)))",
    "CsRank(Div(Sub($close, $open), $open))",
    "Neg(CsRank(Std($returns, 10)))",
    "CsRank(Corr($close, $volume, 10))",
]


class RepeatingProvider(MockProvider):
    """Returns the same batch every call (plus one in-batch repeat) and keeps the prompts."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    def generate(self, system_prompt, user_prompt, temperature=0.8, max_tokens=4096, **kwargs):
        self.prompts.append(user_prompt)
        lines = [f"{i + 1}. f{i}: {f}" for i, f in enumerate(FORMULAS)]
        lines.append(f"5. f0_renamed: {FORMULAS[0]}")
        return "\n".join(lines)


def _loop(tmp_path, **mining):
    cfg = load_config(overrides={"mining": {"target_library_size": 50, "batch_size": 5, **mining}})
    cfg.output_dir = str(tmp_path)
    rng = np.random.default_rng(0)
    provider = RepeatingProvider()
    loop = RalphLoop(config=cfg, data_tensor=rng.normal(size=(12, 60, 8)),
                     returns=rng.normal(size=(12, 60)), llm_provider=provider)
    return loop, provider


def test_skip_tried_drops_repeats_within_and_across_batches(tmp_path):
    loop, _ = _loop(tmp_path, skip_tried=True)
    first = loop._loop_services.execute_iteration(5)
    assert first["tried_skipped"] == 1  # the renamed in-batch repeat
    evaluated = len(loop.trial_ledger.observations)
    assert evaluated == len(FORMULAS)
    loop._loop_services.execute_iteration(5)
    assert len(loop.trial_ledger.observations) == evaluated  # nothing re-evaluated


def test_default_reevaluates_repeats(tmp_path):
    loop, _ = _loop(tmp_path)
    loop._loop_services.execute_iteration(5)
    loop._loop_services.execute_iteration(5)
    assert len(loop.trial_ledger.observations) == 2 * (len(FORMULAS) + 1)


def test_tried_in_prompt_lists_rejected_formulas(tmp_path):
    loop, provider = _loop(tmp_path, tried_in_prompt=3)
    loop._loop_services.execute_iteration(5)
    loop._loop_services.execute_iteration(5)
    assert "ALREADY TRIED" not in provider.prompts[0]
    rejected = loop._recently_tried(3)
    assert 0 < len(rejected) <= 3 and len(set(rejected)) == len(rejected)
    assert "ALREADY TRIED" in provider.prompts[-1]
    assert all(formula in provider.prompts[-1] for formula in rejected)


def test_prompt_builder_renders_recently_tried():
    prompt = PromptBuilder().build_user_prompt({"recently_tried": ["Neg($close)"]}, {}, batch_size=4)
    assert "ALREADY TRIED" in prompt and "  - Neg($close)" in prompt
    assert "ALREADY TRIED" not in PromptBuilder().build_user_prompt({}, {}, batch_size=4)


def test_prompt_builder_renders_a_task_brief_instead_of_the_generic_directive():
    brief = "False positives in the book have higher 20-day volume growth than true positives."
    prompt = PromptBuilder().build_user_prompt({"task_brief": brief}, {}, batch_size=4)
    assert "RESEARCH TASK" in prompt and brief in prompt and "ORTHOGONALITY" not in prompt
    assert "ORTHOGONALITY" in PromptBuilder().build_user_prompt({}, {}, batch_size=4)
