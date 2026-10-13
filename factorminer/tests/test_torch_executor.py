"""Parity of the torch plan executor with the NumPy expression evaluator."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from factorminer.core.expression_plan import compile_batch  # noqa: E402
from factorminer.core.expression_tree import _dispatch_operator  # noqa: E402
from factorminer.core.parser import parse  # noqa: E402
from factorminer.core.types import OPERATOR_REGISTRY  # noqa: E402
from factorminer.evaluation import runtime  # noqa: E402
from factorminer.operators.torch_executor import (  # noqa: E402
    TorchPlanExecutor,
    _pairwise_sum,
    torch_kernel_names,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CUDA and the host libm round these transcendental functions independently.
ROUNDING_OPERATORS = frozenset({"Exp", "Log", "LogReturn", "Pow", "Skew", "Kurt", "Tanh"})
if DEVICE == "cpu":
    # Torch's vectorized CPU sqrt is not correctly rounded; CUDA's is.
    ROUNDING_OPERATORS |= {"Sqrt", "Std", "Corr"}
WINDOWS = (1, 2, 3, 5, 8, 9, 20, 130, 300)


def _panel(rng: np.random.Generator, assets: int = 23, periods: int = 157) -> np.ndarray:
    x = rng.standard_normal((assets, periods)).cumsum(axis=1)
    x[rng.random((assets, periods)) < 0.15] = np.nan
    x[1, :] = np.nan  # an asset that never trades
    x[2, 30:70] = np.nan  # a suspension longer than most windows
    x[3, ::2] = 1.0  # ties within windows
    x[4, :] = 2.0  # a constant series
    x[5, 11] = np.inf
    x[:, 40] = 0.5  # a cross-section of ties
    return x


@pytest.fixture(scope="module")
def executor() -> TorchPlanExecutor:
    # A small block budget exercises the row/period blocking on small panels.
    return TorchPlanExecutor(DEVICE, block_elements=997)


@pytest.fixture(scope="module")
def inputs() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [_panel(rng) for _ in range(3)]


def _cases():
    for name in sorted(torch_kernel_names()):
        spec = OPERATOR_REGISTRY[name]
        windows = WINDOWS if "window" in spec.param_names else (None,)
        for window in windows:
            yield name, window


@pytest.mark.parametrize(("name", "window"), list(_cases()))
def test_device_kernel_matches_numpy_dispatch(executor, inputs, name, window):
    spec = OPERATOR_REGISTRY[name]
    params: dict[str, float] = {} if window is None else {"window": float(window)}
    if name == "Quantile":
        params["q"] = 0.3
    children = inputs[: spec.arity]
    if name == "IfElse":
        children = [inputs[0] - np.nanmean(inputs[0]), inputs[1], inputs[2]]

    expected = _dispatch_operator(spec, children, params)
    actual = executor.to_numpy(
        executor.operator(spec, [executor.from_numpy(child) for child in children], params)
    )

    assert actual.shape == expected.shape
    if name in ROUNDING_OPERATORS:
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True)
    else:
        np.testing.assert_array_equal(actual, expected)


HOST_OPERATORS = ("EMA", "DEMA", "KAMA", "CumSum", "CumProd", "CsZScore", "CsDemean",
                  "CsNeutralize", "CsScale", "Scale")


def test_registry_operators_without_kernels_use_the_exact_host_path(executor, inputs):
    # Other tests may register custom operators; check the built-in host set.
    assert not set(HOST_OPERATORS) & torch_kernel_names()
    for name in HOST_OPERATORS:
        spec = OPERATOR_REGISTRY[name]
        params = {"window": 10.0} if "window" in spec.param_names else {}
        children = inputs[: spec.arity]
        expected = _dispatch_operator(spec, children, params)
        actual = executor.to_numpy(
            executor.operator(spec, [executor.from_numpy(c) for c in children], params)
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 127, 128, 129, 300, 1000])
def test_pairwise_sum_reproduces_numpy_summation_order(length):
    rng = np.random.default_rng(length)
    values = rng.standard_normal((50, length)) * rng.choice([1e-3, 1.0, 1e6], size=(50, 1))
    total = _pairwise_sum(torch.as_tensor(values, device=DEVICE)).cpu().numpy()
    np.testing.assert_array_equal(total, values.sum(axis=1))


@pytest.mark.skipif(DEVICE == "cpu", reason="bit parity is a CUDA property (CPU sqrt rounds)")
def test_batch_plan_on_device_matches_numpy_with_shared_subexpressions(executor, inputs):
    data = {"$close": inputs[0], "$volume": np.abs(inputs[1]), "$open": inputs[2]}
    formulas = [
        "CsRank(Div(Sub($close, Mean($close, 10)), Std($close, 10)))",
        "Neg(CsRank(Corr(CsRank($close), CsRank($volume), 10)))",
        "CsRank(Sub(TsRank($volume, 10), TsRank($close, 10)))",
        "IfElse(Greater(Std($close, 10), 1), TsLinRegSlope($close, 10), EMA($open, 5))",
        "CsZScore(Resid($close, $open, 20))",
    ]
    plan = compile_batch([parse(formula) for formula in formulas])
    assert plan.reused_steps > 0
    host = list(plan.iter_outputs(data))
    device = list(plan.iter_outputs(data, executor))
    for (index, expected), (_, actual) in zip(host, device, strict=True):
        assert isinstance(actual, np.ndarray), formulas[index]
        np.testing.assert_array_equal(actual, expected, err_msg=formulas[index])


def test_device_failures_propagate_like_numpy(executor, inputs):
    plan = compile_batch([parse("Mean($close, 5)"), parse("Mean($volume, 5)")])
    outputs = list(plan.iter_outputs({"$close": inputs[0]}, executor))
    assert isinstance(outputs[0][1], np.ndarray)
    assert isinstance(outputs[1][1], KeyError)


def test_plan_executor_selection(monkeypatch):
    assert runtime.plan_executor(None) is None
    assert runtime.plan_executor("numpy") is None
    with pytest.raises(ValueError, match="Unknown evaluation backend"):
        runtime.plan_executor("tpu")
    monkeypatch.setattr(runtime, "_PLAN_EXECUTORS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        runtime.plan_executor("gpu")


def test_device_out_of_memory_falls_back_to_the_exact_host_path(monkeypatch, inputs):
    from factorminer.operators import torch_executor

    executor = TorchPlanExecutor(DEVICE)
    real = torch_executor._KERNELS["Mean"]

    def exhausted(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated: device held by another process")

    monkeypatch.setitem(torch_executor._KERNELS, "Mean", exhausted)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    data = {"$close": inputs[0]}
    plan = compile_batch([parse("CsRank(Sub($close, Mean($close, 10)))")])
    (_, value), = plan.iter_outputs(data, executor)
    (_, expected), = plan.iter_outputs(data)
    np.testing.assert_array_equal(value, expected)
    assert executor.host_fallbacks == 1
    monkeypatch.setitem(torch_executor._KERNELS, "Mean", real)


def test_leaves_stay_on_the_host_when_the_device_is_full(monkeypatch, inputs):
    executor = TorchPlanExecutor(DEVICE)

    def full(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated")

    monkeypatch.setattr(executor, "_upload", full)
    monkeypatch.setattr(executor, "_to_device", full)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    data = {"$close": inputs[0]}
    plan = compile_batch([parse("Add(CsRank(Mean($close, 5)), 1)")])
    (_, value), = plan.iter_outputs(data, executor)
    (_, expected), = plan.iter_outputs(data)
    np.testing.assert_array_equal(value, expected)


def test_fresh_views_of_one_panel_share_one_upload(inputs):
    executor = TorchPlanExecutor(DEVICE)
    tensor = np.stack(inputs, axis=2)  # (M, T, F) as the mining loop holds it
    for _ in range(5):  # each batch slices new views of the same buffer
        for i in range(tensor.shape[2]):
            executor.leaf("$x", {"$x": tensor[:, :, i]})
    assert len(executor._uploads) == tensor.shape[2]


def test_opaque_results_stay_on_the_host_when_the_device_is_full(monkeypatch, inputs):
    executor = TorchPlanExecutor(DEVICE)

    def full(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated")

    monkeypatch.setattr(executor, "_to_device", full)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert executor.from_numpy(inputs[0]) is inputs[0]
    assert executor.host_fallbacks == 1
