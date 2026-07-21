"""Retrieval, proposal, debate, and synthesis capabilities for Helix."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from factorminer.agent.output_parser import candidate_pairs
from factorminer.core.parser import try_parse
from factorminer.memory.retrieval import retrieve_memory

logger = logging.getLogger(__name__)


class HelixGenerationService:
    """Compose optional retrieval, debate, and deduplication capabilities."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loop, name)

    def _helix_retrieve(self, library_state: dict[str, Any]) -> dict[str, Any]:
        """Stage 1 RETRIEVE: KG + embeddings + flat memory hybrid retrieval.

        Falls back to standard retrieve_memory if no KG/embedder is available.
        """
        (retrieve_enhanced_fn,) = self._component_factory.resolve(
            "factorminer.memory.kg_retrieval",
            "retrieve_memory_enhanced",
        )

        if retrieve_enhanced_fn is not None and (
            self._kg is not None or self._embedder is not None
        ):
            try:
                return retrieve_enhanced_fn(
                    memory=self.memory,
                    library_state=library_state,
                    kg=self._kg,
                    embedder=self._embedder,
                )
            except Exception as exc:
                logger.warning("Helix: enhanced retrieval failed, falling back: %s", exc)

        return retrieve_memory(self.memory, library_state=library_state)

    # ------------------------------------------------------------------
    # Stage 2: Debate or standard proposal
    # ------------------------------------------------------------------

    def _helix_propose(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        batch_size: int,
    ) -> list[tuple[str, str]]:
        """Stage 2 PROPOSE: Use debate generator or standard generator.

        Returns list of (name, formula) tuples compatible with the
        validation pipeline.
        """
        if self._debate_generator is not None:
            try:
                debate_candidates = self._debate_generator.generate_batch(
                    memory_signal=memory_signal,
                    library_state=library_state,
                    batch_size=batch_size,
                )
                self._record_debate_round()
                tuples = candidate_pairs(debate_candidates)
                if tuples:
                    logger.info(
                        "Helix: debate generator produced %d candidates",
                        len(tuples),
                    )
                    return tuples
                logger.warning(
                    "Helix: debate generator returned 0 candidates, "
                    "falling back to standard generator"
                )
            except Exception as exc:
                logger.warning("Helix: debate generation failed, falling back: %s", exc)

        # Standard generation uses the same validated generator as RalphLoop.
        candidates = self.generator.generate_batch(
            memory_signal=memory_signal,
            library_state=library_state,
            batch_size=batch_size,
        )
        return candidate_pairs(candidates)

    def _record_debate_round(self) -> None:
        """Serialize the latest debate/critic result into the run directory.

        Writes ``debate_log.json`` under the configured output directory so an
        external planner (MCP ``inspect_debate``) can inspect specialist
        proposals, critic scores, and shortlist reasoning after the process
        exits. Failures are swallowed -- debate telemetry must never abort mining.
        """
        generator = self._debate_generator
        if generator is None:
            return
        result = getattr(generator, "last_debate_result", None)
        if result is None:
            return

        critic_scores: list[dict[str, Any]] = []
        for score in getattr(result, "critic_scores", None) or []:
            critic_scores.append(
                {
                    "factor_name": getattr(score, "factor_name", ""),
                    "formula": getattr(score, "formula", ""),
                    "source_specialist": getattr(score, "source_specialist", ""),
                    "scores": dict(getattr(score, "scores", {}) or {}),
                    "composite_score": float(getattr(score, "composite_score", 0.0) or 0.0),
                    "keep": bool(getattr(score, "keep", False)),
                    "critique": getattr(score, "critique", "") or "",
                }
            )

        round_payload: dict[str, Any] = {
            "iteration": int(getattr(self, "iteration", 0) or 0),
            "all_proposals": list(getattr(result, "all_proposals", None) or []),
            "after_dedup": list(getattr(result, "after_dedup", None) or []),
            "after_critic": list(getattr(result, "after_critic", None) or []),
            "specialist_proposals": {
                str(name): list(formulas)
                for name, formulas in (getattr(result, "specialist_proposals", None) or {}).items()
            },
            "specialist_success_rates": {
                str(name): float(rate)
                for name, rate in (getattr(result, "specialist_success_rates", None) or {}).items()
            },
            "critic_scores": critic_scores,
            "debate_stats": dict(getattr(result, "debate_stats", None) or {}),
        }

        leaderboard = None
        get_leaderboard = getattr(generator, "get_specialist_leaderboard", None)
        if callable(get_leaderboard):
            try:
                leaderboard = get_leaderboard()
            except Exception:  # pragma: no cover - defensive
                leaderboard = None
        if leaderboard is not None:
            round_payload["specialist_leaderboard"] = leaderboard

        self._debate_rounds.append(round_payload)

        output_dir = Path(self.settings.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "debate_log.json"
            payload = {
                "ok": True,
                "research_artifact_only": True,
                "n_rounds": len(self._debate_rounds),
                "rounds": self._debate_rounds,
            }
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk errors should not kill the loop
            logger.warning("Helix: failed to persist debate_log.json: %s", exc)

    # ------------------------------------------------------------------
    # Stage 3: Canonicalization + deduplication
    # ------------------------------------------------------------------

    def _canonicalize_and_dedup(
        self, candidates: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], int, int]:
        """Stage 3 SYNTHESIZE: Remove mathematically equivalent candidates.

        Uses SymPy-based canonicalization to detect algebraic duplicates
        before evaluation, saving compute.

        Returns
        -------
        tuple of (deduplicated_candidates, n_canonical_duplicates_removed,
        n_semantic_duplicates_removed)
        """
        if self._canonicalizer is None and self._embedder is None:
            return candidates, 0, 0

        seen_hashes: dict[str, str] = {}  # hash -> first factor name
        unique: list[tuple[str, str]] = []
        n_canon_dupes = 0
        n_semantic_dupes = 0

        for name, formula in candidates:
            tree = try_parse(formula)
            if tree is not None and self._canonicalizer is not None:
                try:
                    canon_hash = self._canonicalizer.canonicalize(tree)
                except Exception as exc:
                    logger.debug("Helix: canonicalization failed for '%s': %s", name, exc)
                else:
                    if canon_hash in seen_hashes:
                        n_canon_dupes += 1
                        logger.debug(
                            "Helix: canonical duplicate '%s' matches '%s'",
                            name,
                            seen_hashes[canon_hash],
                        )
                        continue
                    seen_hashes[canon_hash] = name

            semantic_match = self._semantic_duplicate_target(formula)
            if semantic_match is not None:
                n_semantic_dupes += 1
                logger.debug(
                    "Helix: semantic duplicate '%s' matches library factor '%s'",
                    name,
                    semantic_match,
                )
                continue

            unique.append((name, formula))

        if n_canon_dupes > 0:
            logger.info(
                "Helix: canonicalization removed %d/%d duplicate candidates",
                n_canon_dupes,
                len(candidates),
            )

        if n_semantic_dupes > 0:
            logger.info(
                "Helix: embedding screen removed %d/%d library-adjacent candidates",
                n_semantic_dupes,
                len(candidates),
            )

        return unique, n_canon_dupes, n_semantic_dupes

    # ------------------------------------------------------------------
    def _semantic_duplicate_target(self, formula: str) -> str | None:
        """Return the matched library factor if embeddings flag a near-duplicate."""
        if self._embedder is None or self.library.size == 0:
            return None

        try:
            return self._embedder.is_semantic_duplicate(formula)
        except Exception as exc:
            logger.debug("Helix: semantic duplicate check failed: %s", exc)
            return None
