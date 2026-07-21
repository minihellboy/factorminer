"""Import boundary regressions for package-level exports."""

from __future__ import annotations

import json
import subprocess
import sys


def _fresh_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )


def test_evaluation_runtime_import_is_cycle_free_in_fresh_process() -> None:
    result = _fresh_python(
        "import factorminer.evaluation.runtime as runtime; "
        "print(runtime.SignalComputationError.__name__)"
    )

    assert "SignalComputationError" in result.stdout


def test_package_import_is_lightweight_in_fresh_process() -> None:
    result = _fresh_python(
        "import json, sys, factorminer; "
        "heavy = {'numpy', 'pandas', 'scipy', 'torch', 'matplotlib', 'seaborn', 'xgboost'}; "
        "print(json.dumps(sorted(heavy.intersection(sys.modules))))"
    )

    assert json.loads(result.stdout) == []


def test_configuration_export_does_not_load_plotting_or_ml_stacks() -> None:
    result = _fresh_python(
        "import json, sys; from factorminer import Config; "
        "heavy = {'torch', 'matplotlib', 'seaborn', 'xgboost'}; "
        "print(Config.__name__); "
        "print(json.dumps(sorted(heavy.intersection(sys.modules))))"
    )
    lines = result.stdout.splitlines()

    assert lines[0] == "Config"
    assert json.loads(lines[1]) == []


def test_package_namespaces_do_not_eagerly_load_implementations() -> None:
    result = _fresh_python(
        "import importlib, json, sys; "
        "packages = ["
        "'factorminer.agent', 'factorminer.architecture', 'factorminer.benchmark', "
        "'factorminer.core', 'factorminer.data', 'factorminer.domain', "
        "'factorminer.evaluation', 'factorminer.memory', 'factorminer.operators', "
        "'factorminer.utils']; "
        "[importlib.import_module(name) for name in packages]; "
        "implementations = [name for name in sys.modules "
        "if name.startswith('factorminer.') "
        "and name not in packages "
        "and name != 'factorminer._lazy_exports']; "
        "print(json.dumps(sorted(implementations)))"
    )

    assert json.loads(result.stdout) == []


def test_lazy_public_exports_remain_compatible() -> None:
    result = _fresh_python(
        "from factorminer.agent import MockProvider; "
        "from factorminer.core import FactorLibrary; "
        "from factorminer.data import TensorConfig; "
        "from factorminer.domain import SpearmanDependenceMetric; "
        "from factorminer.evaluation import compute_ic; "
        "print(MockProvider.__name__, FactorLibrary.__name__, TensorConfig.__name__, "
        "SpearmanDependenceMetric.__name__, compute_ic.__name__)"
    )

    assert result.stdout.strip() == (
        "MockProvider FactorLibrary TensorConfig SpearmanDependenceMetric compute_ic"
    )
