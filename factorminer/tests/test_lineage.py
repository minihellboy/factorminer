"""Tests for parent_formula lineage wiring into mining trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from factorminer.agent.llm_interface import MockProvider
from factorminer.architecture.memory_policy import EditAwareMemoryPolicy, extract_edit_motif
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.provenance import detect_edit_type, infer_parent_lineage
from factorminer.core.ralph_loop import EvaluationResult, RalphLoop
from factorminer.memory.memory_store import ExperienceMemory


@dataclass
class _Cfg:
    target_library_size: int = 5
    batch_size: int = 4
    max_iterations: int = 2
    ic_threshold: float = 0.0
    icir_threshold: float = 0.0
    correlation_threshold: float = 0.95
    replacement_ic_min: float = 0.10
    replacement_ic_ratio: float = 1.3
    fast_screen_assets: int = 0
    num_workers: int = 1
    output_dir: str = ""
    redundancy_metric: str = "spearman"
    memory_policy: str = "paper"
    memory_regime_lookback_window: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_library_size": self.target_library_size,
            "batch_size": self.batch_size,
            "max_iterations": self.max_iterations,
            "ic_threshold": self.ic_threshold,
            "memory_policy": self.memory_policy,
        }


def _panel(m: int = 12, t: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tensor = rng.normal(0, 1, (m, t, 8)).astype(np.float64)
    returns = rng.normal(0, 0.02, (m, t)).astype(np.float64)
    return tensor, returns


def _factor(
    fid: int,
    name: str,
    formula: str,
    *,
    signals: np.ndarray,
    ic: float = 0.05,
) -> Factor:
    return Factor(
        id=fid,
        name=name,
        formula=formula,
        category="Momentum",
        ic_mean=ic,
        icir=0.8,
        ic_win_rate=0.55,
        max_correlation=0.0,
        batch_number=0,
        ic_paper_mean=abs(ic),
        ic_paper_icir=0.8,
        signals=signals,
    )


def test_infer_parent_lineage_picks_similar_library_formula() -> None:
    library_state = {
        "recent_admissions": [
            {"formula": "Mean($close, 5)", "ic_paper_mean": 0.04, "name": "parent"},
            {"formula": "CsRank($volume)", "ic_paper_mean": 0.02, "name": "other"},
        ]
    }
    lineage = infer_parent_lineage("Mean($close, 20)", library_state)
    assert lineage["parent_formula"] == "Mean($close, 5)"
    assert lineage["edit_type"] in {"mutation", "crossover", "fresh", "unknown"}
    assert extract_edit_motif("Mean($close, 5)", "Mean($close, 20)") == "window_rescale"


def test_detect_edit_type_fresh_without_parent() -> None:
    assert detect_edit_type("Mean($close, 5)", None) == "fresh"
    assert detect_edit_type("Mean($close, 20)", "Mean($close, 5)") == "mutation"


def test_annotate_result_lineage_and_trajectory_carry_parent(tmp_path) -> None:
    tensor, returns = _panel()
    library = FactorLibrary(correlation_threshold=0.95, ic_threshold=0.0)
    parent = _factor(1, "parent_mean", "Mean($close, 5)", signals=returns.copy(), ic=0.05)
    library.admit_factor(parent)

    cfg = _Cfg(output_dir=str(tmp_path / "out"))
    loop = RalphLoop(
        config=cfg,
        data_tensor=tensor,
        returns=returns,
        llm_provider=MockProvider(),
        library=library,
    )

    results = [
        EvaluationResult(
            factor_name="child_mean",
            formula="Mean($close, 20)",
            parse_ok=True,
            ic_mean=0.03,
            ic_paper_mean=0.03,
            icir=0.5,
            ic_paper_icir=0.5,
            admitted=True,
            stage_passed=4,
        ),
        EvaluationResult(
            factor_name="fresh_vol",
            formula="CsZScore($volume)",
            parse_ok=True,
            ic_mean=0.01,
            ic_paper_mean=0.01,
            admitted=False,
            stage_passed=1,
        ),
    ]
    library_state = library.get_state_summary()
    loop._annotate_result_lineage(results, library_state)

    assert results[0].parent_formula == "Mean($close, 5)"
    assert results[0].edit_type in {"mutation", "crossover", "fresh", "unknown"}
    assert extract_edit_motif(results[0].parent_formula, results[0].formula) == "window_rescale"
    assert results[0].parent_ic_paper_mean == 0.05

    trajectory = loop._build_trajectory(results)
    child_entries = [e for e in trajectory if e.get("formula") == "Mean($close, 20)"]
    assert child_entries
    assert child_entries[0].get("parent_formula") == "Mean($close, 5)"
    assert child_entries[0].get("parent_ic_paper_mean") == 0.05
    assert child_entries[0].get("ic_paper_mean") == 0.03


def test_two_iteration_run_emits_non_null_parent_on_admitted_factor(tmp_path) -> None:
    """Synthetic 2-step mining produces trajectory entries with parent_formula."""
    tensor, returns = _panel(m=15, t=50, seed=7)
    cfg = _Cfg(output_dir=str(tmp_path / "mine"), memory_policy="paper")
    loop = RalphLoop(
        config=cfg,
        data_tensor=tensor,
        returns=returns,
        llm_provider=MockProvider(),
    )

    seed = _factor(
        1,
        "seed_close",
        "CsRank(Delta($close, 5))",
        signals=returns.copy(),
        ic=0.06,
    )
    loop.library.admit_factor(seed)

    loop.iteration = 1
    r1 = EvaluationResult(
        factor_name="mut_child",
        formula="Neg(CsRank(Delta($close, 5)))",
        parse_ok=True,
        ic_mean=0.05,
        ic_paper_mean=0.05,
        icir=0.9,
        ic_paper_icir=0.9,
        admitted=True,
        stage_passed=4,
        signals=returns.copy(),
    )
    loop._annotate_result_lineage([r1], loop.library.get_state_summary())
    assert r1.parent_formula == "CsRank(Delta($close, 5))"
    assert r1.edit_type == "mutation"

    child = _factor(
        2,
        r1.factor_name,
        r1.formula,
        signals=r1.signals if r1.signals is not None else returns.copy(),
        ic=r1.ic_paper_mean,
    )
    loop.library.admit_factor(child)

    traj = loop._build_trajectory([r1])
    assert any(e.get("parent_formula") for e in traj), traj

    # edit_aware policy observes the edge when parent_formula is present.
    from factorminer.utils.config import load_config

    protocol = PaperProtocol.from_config(load_config())
    policy = EditAwareMemoryPolicy(protocol)
    memory = ExperienceMemory()
    policy.form(memory, traj, iteration=1)
    assert len(policy._motif_stats) >= 1
