"""Boundary from structured memory state to generation prompt context."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from factorminer.architecture.families import FactorFamilyDiscovery
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.architecture.research_planner import ResearchCyclePlanner, RoutingMode


def _research_archetype_prompt_text(archetypes: Sequence[Any]) -> str:
    """Render RMA `ResearchArchetype`-like records into prompt-facing text.

    Duck-typed on `.to_dict()` (or plain dict) so this module never needs to
    import `factorminer.architecture.research_absorption`.
    """
    lines = ["=== RESEARCH ARCHETYPE CONTEXT ==="]
    for archetype in archetypes:
        data = archetype.to_dict() if hasattr(archetype, "to_dict") else dict(archetype)
        name = data.get("name", "archetype")
        family = data.get("mechanism_family", "")
        role = data.get("mechanism_role", "")
        paths = data.get("research_paths", []) or []
        lines.append(f"- {name} [{family}]" if family else f"- {name}")
        if role:
            lines.append(f"  role: {role}")
        for path in paths:
            lines.append(f"  path: {path}")
    return "\n".join(lines)


@dataclass
class PromptContextBuilder:
    """Builds the prompt-facing context payload for factor generation."""

    protocol: PaperProtocol
    family_discovery: FactorFamilyDiscovery | None = None
    cycle_planner: ResearchCyclePlanner | None = None

    def build(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        *,
        batch_size: int,
        extras: dict[str, Any] | None = None,
        cycle_mode: RoutingMode = "memory_driven",
        cycle_fixed_family: str | None = None,
        cycle_coarse_hint: str | None = None,
        research_archetypes: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(memory_signal)
        payload["library_state"] = dict(library_state)
        payload["protocol_contract"] = self.protocol.runtime_contract()
        payload["generation_batch_size"] = int(batch_size)
        if self.family_discovery is not None:
            family_context = self.family_discovery.summarize(
                library_state=dict(library_state),
                memory_signal=dict(memory_signal),
            )
            payload["family_context"] = family_context
            payload["family_prompt_text"] = family_context.get("prompt_text", "")
        if self.cycle_planner is not None:
            cycle_plan = self.cycle_planner.plan_cycle(
                dict(library_state),
                dict(memory_signal),
                mode=cycle_mode,
                fixed_family=cycle_fixed_family,
                coarse_hint=cycle_coarse_hint,
            )
            payload["research_cycle_theme"] = cycle_plan.to_dict()
            payload["research_cycle_prompt_text"] = cycle_plan.prompt_text
        if research_archetypes:
            payload["research_archetypes"] = [
                archetype.to_dict() if hasattr(archetype, "to_dict") else dict(archetype)
                for archetype in research_archetypes
            ]
            payload["research_prompt_text"] = _research_archetype_prompt_text(research_archetypes)
        if extras:
            payload.update(extras)
        return payload
