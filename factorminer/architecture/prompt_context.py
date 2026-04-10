"""Boundary from structured memory state to generation prompt context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from factorminer.architecture.families import FactorFamilyDiscovery
from factorminer.architecture.paper_protocol import PaperProtocol


@dataclass
class PromptContextBuilder:
    """Builds the prompt-facing context payload for factor generation."""

    protocol: PaperProtocol
    family_discovery: FactorFamilyDiscovery | None = None

    def build(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        *,
        batch_size: int,
        extras: dict[str, Any] | None = None,
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
        if extras:
            payload.update(extras)
        return payload
