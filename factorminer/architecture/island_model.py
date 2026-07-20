"""Island-model mining mode: independent sub-populations with periodic migration.

Mirrors the AlphaEvolve/FunSearch search-diversity mechanic (DeepMind): several
sub-populations ("islands") evolve independently -- each with its own
`FactorLibrary` and its own generation bias -- and periodically exchange top
performers ("migration"). This is a population-diversity mechanism, not a new
mining algorithm: it reuses the existing `RalphLoop`/`HelixLoop` mining loop
unchanged (one loop instance per island) and the existing
`FactorAdmissionService` admission/replacement logic (one destination-library
gate per migrant). See `docs/landscape-and-extensions.md` §2 ("AlphaEvolve /
FunSearch") for the motivating gap: FactorMiner's family-aware memory policy
reranks *one* shared library by family gap; it has no multi-population /
periodic-migration mechanism to prevent one dominant factor family from
crowding out the search.

Composition, not a new stage or loop branch:

- `IslandBiasDescriptor` -- a per-island steering profile (preferred families
  plus, optionally, a regime focus) that both nudges a real LLM's prompt
  (via `recommended_directions` hints fed through the existing
  `PromptContextBuilder` path) and deterministically filters/reorders
  whatever candidates the underlying generator proposes, so the bias has a
  reproducible effect under `MockProvider` too.
- `IslandPopulation` -- one island: an owned `RalphLoop`/`HelixLoop` instance
  (and therefore its own `FactorLibrary`, `ExperienceMemory`, and memory
  policy), wrapped so its generator applies that island's bias.
- `IslandMiningCoordinator` -- drives N `IslandPopulation`s for
  `migration_interval` iterations at a time, then migrates each island's
  top-K factors (by `ic_paper_mean`, deduplicated by formula) into every
  other island via `FactorAdmissionService.admit_migrant` -- migrated
  factors must still clear the destination's own admission bar.

Optional Hypothesis-Redundancy gate (landscape §10 item 5, default **off**):
when `enable_llm_jump_gate=True`, the coordinator records a
:class:`~factorminer.architecture.geometry.JumpWorthAssessment` per island
at each migration boundary. Existing island-model behaviour is unchanged
unless the gate is explicitly enabled; the assessment is advisory for
exploration-schedule consumers (prefer local AST edit vs. frontier LLM).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from factorminer.architecture.families import FactorFamilyDiscovery, infer_family
from factorminer.core.factor_library import Factor, FactorLibrary

if TYPE_CHECKING:
    from factorminer.core.helix_loop import HelixLoop
    from factorminer.core.ralph_loop import RalphLoop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bias descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IslandBiasDescriptor:
    """Steering profile for one island's generation process.

    Parameters
    ----------
    name : str
        Human-readable label for this bias (typically the island name).
    preferred_families : tuple[str, ...]
        `families.infer_family` category names this island should favor
        (e.g. ``("Momentum", "Smoothing")``). Empty means unbiased.
    family_bias_strength : float
        Fraction in ``[0, 1]`` of proposed candidates *outside*
        `preferred_families` to filter out before evaluation. ``1.0`` keeps
        only preferred-family candidates (falling back to the unfiltered
        batch if that would empty it); ``0.0`` disables filtering entirely,
        leaving only the prompt-text nudge in effect.
    regime_focus : str, optional
        Optional regime label surfaced in the prompt hint text, for parity
        with `RegimeAwareMemoryPolicy`-style steering.
    """

    name: str
    preferred_families: tuple[str, ...] = ()
    family_bias_strength: float = 1.0
    regime_focus: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.family_bias_strength <= 1.0:
            raise ValueError("family_bias_strength must be in [0, 1]")

    def recommended_direction_hints(self) -> list[str]:
        """Prompt-facing hints steering a real LLM toward this island's bias."""
        hints: list[str] = []
        if self.preferred_families:
            hints.append(
                f"[island:{self.name}] Favor factor families: "
                + ", ".join(self.preferred_families)
            )
        if self.regime_focus:
            hints.append(f"[island:{self.name}] Focus on regime: {self.regime_focus}")
        return hints

    def filter_candidates(
        self, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Deterministically bias a generated candidate batch toward this island.

        Splits `candidates` into those whose formula's inferred family is in
        `preferred_families` and those that aren't, then keeps all of the
        former plus a `1 - family_bias_strength` fraction of the latter.
        With no `preferred_families` (or zero bias strength) this is a no-op.
        """
        if not self.preferred_families or self.family_bias_strength <= 0.0:
            return list(candidates)

        preferred: list[tuple[str, str]] = []
        other: list[tuple[str, str]] = []
        for name, formula in candidates:
            bucket = preferred if infer_family(formula) in self.preferred_families else other
            bucket.append((name, formula))

        if self.family_bias_strength >= 1.0:
            return preferred or list(candidates)

        keep_other = round(len(other) * (1.0 - self.family_bias_strength))
        return preferred + other[:keep_other]


# ---------------------------------------------------------------------------
# Generator wrapper (composition, not a ralph_loop.py change)
# ---------------------------------------------------------------------------


class _BiasedFactorGenerator:
    """Wraps an island's `FactorGenerator`, applying its `IslandBiasDescriptor`.

    Delegates generation entirely to the wrapped generator (never
    reimplements it); only adds prompt hints before and filters candidates
    after. Duck-types the `FactorGenerator.generate_batch` contract so it can
    be swapped onto `RalphLoop.generator`/`HelixLoop.generator` in place.
    """

    def __init__(self, inner: Any, bias: IslandBiasDescriptor) -> None:
        self._inner = inner
        self.bias = bias

    def generate_batch(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        batch_size: int = 40,
    ) -> list[tuple[str, str]]:
        biased_signal = dict(memory_signal)
        hints = list(biased_signal.get("recommended_directions", []) or [])
        biased_signal["recommended_directions"] = hints + self.bias.recommended_direction_hints()
        candidates = self._inner.generate_batch(
            memory_signal=biased_signal,
            library_state=library_state,
            batch_size=batch_size,
        )
        return self.bias.filter_candidates(candidates)

    def __getattr__(self, item: str) -> Any:
        # Forward anything else (e.g. `.llm_provider`) to the wrapped generator.
        return getattr(self._inner, item)


# ---------------------------------------------------------------------------
# Family diversity snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyDiversitySnapshot:
    """A single island's factor-family composition at one point in time."""

    island: str
    family_counts: dict[str, int]
    total_factors: int
    distinct_families: int
    shannon_entropy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "island": self.island,
            "family_counts": dict(self.family_counts),
            "total_factors": self.total_factors,
            "distinct_families": self.distinct_families,
            "shannon_entropy": self.shannon_entropy,
        }

    @classmethod
    def from_counts(cls, island: str, family_counts: dict[str, int]) -> FamilyDiversitySnapshot:
        counts = {name: count for name, count in family_counts.items() if count > 0}
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return cls(
            island=island,
            family_counts=counts,
            total_factors=total,
            distinct_families=len(counts),
            shannon_entropy=entropy,
        )


# ---------------------------------------------------------------------------
# Island population
# ---------------------------------------------------------------------------


class IslandPopulation:
    """One island: an independent mining loop, library, and generation bias.

    Wraps a fully-constructed `RalphLoop`/`HelixLoop` instance -- the mining
    loop itself is reused unchanged; this class only owns identity (`name`,
    `bias`) and installs a `_BiasedFactorGenerator` in front of the loop's
    generator so this island's proposals skew toward its bias.
    """

    def __init__(
        self,
        name: str,
        loop: RalphLoop | HelixLoop,
        bias: IslandBiasDescriptor | None = None,
    ) -> None:
        self.name = name
        self.loop = loop
        self.bias = bias or IslandBiasDescriptor(name=name)
        if not isinstance(loop.generator, _BiasedFactorGenerator):
            loop.generator = _BiasedFactorGenerator(loop.generator, self.bias)

    @property
    def library(self) -> FactorLibrary:
        return self.loop.library

    @property
    def admission_service(self):  # -> FactorAdmissionService
        return self.loop.admission_service

    def run_iterations(self, upto_iteration: int, *, target_size: int = 10_000) -> FactorLibrary:
        """Advance this island's loop up to (absolute) iteration `upto_iteration`."""
        return self.loop.run(target_size=target_size, max_iterations=upto_iteration)

    def family_diversity(
        self, discovery: FactorFamilyDiscovery | None = None
    ) -> FamilyDiversitySnapshot:
        """Snapshot this island's current family composition."""
        discovery = discovery or FactorFamilyDiscovery()
        entries = [
            {"formula": f.formula, "ic_mean": f.ic_mean, "admitted": True}
            for f in self.library.list_factors()
        ]
        families = discovery.discover(entries)
        counts = {family.name: family.count for family in families}
        return FamilyDiversitySnapshot.from_counts(self.name, counts)


# ---------------------------------------------------------------------------
# Migration bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationOutcome:
    """Result of attempting to migrate one factor into one destination island."""

    source: str
    destination: str
    factor_name: str
    formula: str
    family: str
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "destination": self.destination,
            "factor_name": self.factor_name,
            "formula": self.formula,
            "family": self.family,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MigrationRound:
    """All migration attempts triggered at one `migration_interval` boundary."""

    epoch: int
    iteration: int
    outcomes: list[MigrationOutcome] = field(default_factory=list)

    @property
    def accepted(self) -> list[MigrationOutcome]:
        return [o for o in self.outcomes if o.accepted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class IslandMiningCoordinator:
    """Owns N `IslandPopulation`s and drives an island-model mining run.

    Each island keeps running its own `RalphLoop`/`HelixLoop` (own memory,
    own library); the coordinator's only job is scheduling `run_iterations`
    calls in `migration_interval`-sized blocks and, between blocks, migrating
    each island's top-K factors into every other island's library through
    `FactorAdmissionService.admit_migrant` -- the existing admission bar, not
    a force-insert.
    """

    def __init__(
        self,
        islands: list[IslandPopulation],
        *,
        migration_interval: int = 5,
        migration_top_k: int = 3,
        allow_replacement: bool = False,
        enable_llm_jump_gate: bool = False,
        llm_jump_threshold: float = 0.45,
    ) -> None:
        if len(islands) < 2:
            raise ValueError("island-model mining requires at least 2 islands")
        names = [island.name for island in islands]
        if len(set(names)) != len(names):
            raise ValueError("island names must be unique")
        if migration_interval < 1:
            raise ValueError("migration_interval must be >= 1")
        if migration_top_k < 1:
            raise ValueError("migration_top_k must be >= 1")

        self.islands = list(islands)
        self.migration_interval = migration_interval
        self.migration_top_k = migration_top_k
        self.allow_replacement = allow_replacement
        # Optional geometric gate: default off — existing behaviour unchanged.
        self.enable_llm_jump_gate = bool(enable_llm_jump_gate)
        self.llm_jump_threshold = float(llm_jump_threshold)
        self.migration_rounds: list[MigrationRound] = []
        self.pre_migration_diversity: dict[str, FamilyDiversitySnapshot] | None = None
        self.post_migration_diversity: dict[str, FamilyDiversitySnapshot] | None = None
        self.jump_worth_log: list[dict[str, Any]] = []

    def island(self, name: str) -> IslandPopulation:
        for population in self.islands:
            if population.name == name:
                return population
        raise KeyError(f"no island named {name!r}")

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    def diversity_snapshot(self) -> dict[str, FamilyDiversitySnapshot]:
        """Current per-island family-diversity snapshot."""
        return {population.name: population.family_diversity() for population in self.islands}

    def diversity_report(self) -> dict[str, dict[str, Any]]:
        """Per-island family diversity before vs. after the mining run's migrations.

        Raises
        ------
        RuntimeError
            If `run` has not completed yet.
        """
        if self.pre_migration_diversity is None or self.post_migration_diversity is None:
            raise RuntimeError("run() must complete before requesting a diversity report")
        return {
            name: {
                "before": self.pre_migration_diversity[name].to_dict(),
                "after": self.post_migration_diversity[name].to_dict(),
            }
            for name in self.pre_migration_diversity
        }


    # ------------------------------------------------------------------
    # Optional LLM-jump exploration gate (default off)
    # ------------------------------------------------------------------

    def assess_exploration_jump(
        self,
        population: IslandPopulation,
        *,
        candidate_signals: Any | None = None,
        target: Any | None = None,
    ) -> Any:
        """Run the Hypothesis-Redundancy gate on one island's library span.

        Returns a :class:`~factorminer.architecture.geometry.JumpWorthAssessment`.
        When no ``candidate_signals`` are supplied, a unit probe orthogonal
        to the first library column is synthesised so the call remains a pure
        span diagnostic (useful at migration boundaries).
        """
        import numpy as np

        from factorminer.architecture.geometry import (
            assess_llm_jump_worth,
            collect_library_span_matrix,
        )

        span = collect_library_span_matrix(population.library)
        if candidate_signals is None:
            if span.size == 0:
                probe = np.ones(8, dtype=np.float64)
            else:
                # Deterministic probe: residual of a linear trend vs span.
                d = span.shape[0]
                probe = np.linspace(-1.0, 1.0, d, dtype=np.float64)
        else:
            probe = candidate_signals
        return assess_llm_jump_worth(
            span if span.size else population.library,
            probe,
            target=target,
            threshold=self.llm_jump_threshold,
        )

    def _record_jump_assessments(self, *, epoch: int, iteration: int) -> None:
        """Snapshot per-island jump-worth when the gate is enabled."""
        if not self.enable_llm_jump_gate:
            return
        for population in self.islands:
            assessment = self.assess_exploration_jump(population)
            entry = {
                "epoch": epoch,
                "iteration": iteration,
                "island": population.name,
                **assessment.to_dict(),
            }
            self.jump_worth_log.append(entry)
            logger.info(
                "Island %s jump-worth gate @ epoch=%d iter=%d: "
                "worth=%.3f recommend_llm=%s",
                population.name,
                epoch,
                iteration,
                assessment.jump_worth,
                assessment.recommend_llm_jump,
            )

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def try_migrate_factor(
        self,
        factor: Factor,
        destination: IslandPopulation,
        *,
        source_name: str,
        iteration: int,
    ) -> MigrationOutcome:
        """Attempt to migrate a single factor into `destination`'s library.

        Reuses `FactorAdmissionService.admit_migrant` -- the migrant must
        clear the destination's own admission (or, if `allow_replacement`,
        replacement) bar; it is never force-inserted.
        """
        admitted, reason = destination.admission_service.admit_migrant(
            factor,
            iteration=iteration,
            allow_replacement=self.allow_replacement,
        )
        return MigrationOutcome(
            source=source_name,
            destination=destination.name,
            factor_name=factor.name,
            formula=factor.formula,
            family=infer_family(factor.formula),
            accepted=admitted,
            reason=reason,
        )

    def _top_k_unique(self, population: IslandPopulation) -> list[Factor]:
        """This island's top-K factors by `ic_paper_mean`, deduplicated by formula."""
        ranked = sorted(
            population.library.list_factors(),
            key=lambda f: float(f.ic_paper_mean if f.ic_paper_mean is not None else abs(f.ic_mean)),
            reverse=True,
        )
        seen: set[str] = set()
        unique: list[Factor] = []
        for factor in ranked:
            if factor.formula in seen:
                continue
            seen.add(factor.formula)
            unique.append(factor)
            if len(unique) >= self.migration_top_k:
                break
        return unique

    def migrate(self, *, epoch: int, iteration: int) -> MigrationRound:
        """Migrate each island's top-K factors into every other island.

        Top-K selections are snapshotted for every island up front, before
        any destination library is mutated, so a factor migrated into island
        B this round can't immediately re-qualify for re-migration into
        island C within the same round.
        """
        top_by_island = {
            population.name: self._top_k_unique(population) for population in self.islands
        }
        outcomes: list[MigrationOutcome] = []
        for source in self.islands:
            for factor in top_by_island[source.name]:
                for destination in self.islands:
                    if destination is source:
                        continue
                    outcomes.append(
                        self.try_migrate_factor(
                            factor,
                            destination,
                            source_name=source.name,
                            iteration=iteration,
                        )
                    )
        migration_round = MigrationRound(epoch=epoch, iteration=iteration, outcomes=outcomes)
        self.migration_rounds.append(migration_round)
        logger.info(
            "Island migration epoch=%d iteration=%d: %d/%d accepted",
            epoch,
            iteration,
            len(migration_round.accepted),
            len(outcomes),
        )
        return migration_round

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, total_iterations: int, *, target_size: int = 10_000) -> dict[str, IslandPopulation]:
        """Drive all islands for `total_iterations`, migrating every `migration_interval`.

        `target_size` is passed through to each island's loop as a very high
        library-size ceiling by default, so the schedule here (not the
        loop's own target-size stopping condition) governs how long each
        island runs -- islands still stop early if their own loop reaches
        `target_size` first.

        Returns
        -------
        dict[str, IslandPopulation]
            Islands keyed by name, in their final state.
        """
        if total_iterations < 1:
            raise ValueError("total_iterations must be >= 1")

        iteration = 0
        epoch = 0
        while iteration < total_iterations:
            block_end = min(total_iterations, iteration + self.migration_interval)
            for population in self.islands:
                population.run_iterations(block_end, target_size=target_size)
            iteration = block_end

            if self.pre_migration_diversity is None:
                self.pre_migration_diversity = self.diversity_snapshot()

            epoch += 1
            # Optional advisory gate (no-op when enable_llm_jump_gate is False).
            self._record_jump_assessments(epoch=epoch, iteration=iteration)
            self.migrate(epoch=epoch, iteration=iteration)

        self.post_migration_diversity = self.diversity_snapshot()
        return {population.name: population for population in self.islands}


# ---------------------------------------------------------------------------
# Construction helper
# ---------------------------------------------------------------------------


def build_island(
    name: str,
    config: Any,
    data_tensor: Any,
    returns: Any,
    bias: IslandBiasDescriptor | None = None,
    *,
    llm_provider: Any = None,
    library: FactorLibrary | None = None,
    memory: Any = None,
    loop_cls: type | None = None,
    checkpoint_interval: int = 0,
) -> IslandPopulation:
    """Construct one `IslandPopulation` around a fresh mining loop.

    A thin convenience wrapper: builds `loop_cls(config, data_tensor,
    returns, ...)` (defaulting to `RalphLoop`) and wraps it in an
    `IslandPopulation`. Each island needs its own `config.output_dir` (the
    loop persists session/checkpoint state there); this helper does not
    infer one for you.

    Parameters
    ----------
    name : str
        Island name; also the default bias label.
    config : Any
        Mining config, e.g. `core.config.MiningConfig`, with a distinct
        `output_dir` per island.
    data_tensor, returns : np.ndarray
        Shared market data this island searches over.
    bias : IslandBiasDescriptor, optional
        This island's steering profile. Defaults to unbiased.
    llm_provider, library, memory : optional
        Forwarded to the loop constructor; see `RalphLoop.__init__`.
    loop_cls : type, optional
        `RalphLoop` (default) or `HelixLoop`.
    checkpoint_interval : int
        Forwarded to the loop constructor; defaults to 0 (disabled) since
        island-model runs create many short-lived loops and per-iteration
        checkpointing of every island is rarely wanted.
    """
    if loop_cls is None:
        from factorminer.core.ralph_loop import RalphLoop as loop_cls  # noqa: N813

    loop = loop_cls(
        config=config,
        data_tensor=data_tensor,
        returns=returns,
        llm_provider=llm_provider,
        memory=memory,
        library=library,
        checkpoint_interval=checkpoint_interval,
    )
    return IslandPopulation(name=name, loop=loop, bias=bias or IslandBiasDescriptor(name=name))
