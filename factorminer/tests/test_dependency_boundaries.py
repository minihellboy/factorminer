"""Executable tests for architectural dependency direction."""

from __future__ import annotations

from scripts.check_architecture import check_repository, layer_for, violations_for_source


def test_repository_respects_dependency_rules() -> None:
    assert check_repository() == []


def test_layer_assignment_is_explicit_for_representative_modules() -> None:
    assert layer_for("factorminer.domain.dependence") == "domain"
    assert layer_for("factorminer.core.factor_library") == "domain"
    assert layer_for("factorminer.agent.factor_generator") == "adapter"
    assert layer_for("factorminer.core.ralph_loop") == "application"
    assert layer_for("factorminer.benchmark.runtime") == "interface"


def test_domain_to_evaluation_import_is_rejected() -> None:
    violations = violations_for_source(
        "factorminer.domain.example",
        "from factorminer.evaluation.metrics import compute_ic\n",
    )
    assert len(violations) == 1
    assert violations[0].rule == "domain must not depend on higher layers"


def test_adapter_to_cli_import_is_rejected() -> None:
    violations = violations_for_source(
        "factorminer.agent.example",
        "from factorminer.cli import main\n",
    )
    assert len(violations) == 1
    assert "adapter must not depend" in violations[0].rule


def test_internal_compatibility_import_is_rejected() -> None:
    violations = violations_for_source(
        "factorminer.architecture.geometry",
        "from factorminer.architecture.dependence import DependenceMetric\n",
    )
    assert len(violations) == 1
    assert "compatibility import" in violations[0].rule
