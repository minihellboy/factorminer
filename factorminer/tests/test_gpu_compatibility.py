"""Exercise real CUDA tensors when available, with CPU numerical oracles.

Set FACTORMINER_REQUIRE_CUDA=1 on a GPU runner to make missing CUDA a failure.
"""

from __future__ import annotations

import inspect
import os
import warnings

import numpy as np
import pytest

from factorminer.operators.registry import OPERATOR_REGISTRY, execute_operator


@pytest.fixture(params=["cpu", "cuda"])
def tensor_device(request):
    try:
        import torch
    except ImportError:
        if os.environ.get("FACTORMINER_REQUIRE_CUDA") == "1":
            pytest.fail("CUDA validation requires PyTorch")
        pytest.skip("PyTorch is optional")
    if request.param == "cuda" and not torch.cuda.is_available():
        if os.environ.get("FACTORMINER_REQUIRE_CUDA") == "1":
            pytest.fail("CUDA validation requires an available CUDA device")
        pytest.skip("CUDA device unavailable")
    return torch.device(request.param)


_OPERATORS = [
    name for name, (_, numpy_fn, torch_fn) in OPERATOR_REGISTRY.items()
    if numpy_fn is not None and torch_fn is not None
]


def test_expression_ranks_break_ties_by_asset_order():
    from factorminer.core.parser import parse

    values = np.array([[2., np.nan], [2., 1.], [1., np.nan], [np.nan, np.nan]])
    result = parse("CsRank($close)").evaluate({"$close": values})
    expected = np.array([[2/3, np.nan], [1., 1.], [1/3, np.nan], [np.nan, np.nan]])
    np.testing.assert_allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize("operator", _OPERATORS)
@pytest.mark.parametrize("scenario", ["dense", "missing", "ties", "short"])
def test_registered_operator_matches_numpy_on_device(tensor_device, operator, scenario):
    import torch

    rng = np.random.default_rng(20260913)
    arrays = [rng.normal(size=(9, 3 if scenario == "short" else 16)) for _ in range(3)]
    if scenario == "missing":
        for i, arr in enumerate(arrays):
            arr[:, :6] = np.nan
            arr[i, :] = np.nan
            arr[:, -1] = np.nan
            arr[-1, -1] = 1.0
            arr[(i + 3) % 9, 9] = np.nan
    elif scenario == "ties":
        arrays = [np.round(arr) for arr in arrays]
    spec, numpy_fn, _ = OPERATOR_REGISTRY[operator]
    inputs = arrays[:spec.arity]
    params = {"window": 6} if "window" in inspect.signature(numpy_fn).parameters else {}
    if operator == "Quantile":
        params["q"] = 0.35
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = execute_operator(operator, *inputs, params=params, backend="numpy")
    tensors = [torch.tensor(x, device=tensor_device, dtype=torch.float64) for x in inputs]
    actual = execute_operator(operator, *tensors, params=params, backend="torch")
    assert actual.device.type == tensor_device.type
    assert actual.shape == expected.shape
    result = actual.detach().cpu().numpy()
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6, equal_nan=True)


@pytest.mark.parametrize("operator", ["Add", "Mul", "Div", "Inv", "Mean", "Std", "Corr"])
def test_operator_gradients_match_cpu(tensor_device, operator):
    import torch

    spec, numpy_fn, _ = OPERATOR_REGISTRY[operator]
    rng = np.random.default_rng(71)
    arrays = [rng.uniform(0.5, 2, size=(5, 12)) for _ in range(spec.arity)]
    params = {"window": 4} if "window" in inspect.signature(numpy_fn).parameters else {}
    gradients = []
    for device in [torch.device("cpu"), tensor_device]:
        inputs = [torch.tensor(x, dtype=torch.float64, device=device, requires_grad=True)
                  for x in arrays]
        result = execute_operator(operator, *inputs, params=params, backend="torch")
        torch.nan_to_num(result).square().sum().backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs)
        gradients.append([x.grad.cpu().numpy() for x in inputs])
    for expected, actual in zip(*gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    # An independent NumPy finite-difference oracle also checks the derivative,
    # rather than merely reproducing the same Torch implementation on CPU.
    directions = [rng.normal(size=x.shape) for x in arrays]
    losses = []
    for step in [-1e-5, 1e-5]:
        shifted = [x + step * d for x, d in zip(arrays, directions, strict=True)]
        result = execute_operator(operator, *shifted, params=params, backend="numpy")
        losses.append(np.nansum(result ** 2))
    expected = (losses[1] - losses[0]) / 2e-5
    actual = sum(np.sum(g * d) for g, d in zip(gradients[-1], directions, strict=True))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_neural_training_and_portable_checkpoint(tensor_device, tmp_path):
    import torch

    from factorminer.operators.neuro_symbolic import (
        NeuralLeaf,
        NeuralLeafRegistry,
        train_neural_leaf,
    )

    rng = np.random.default_rng(22)
    features = rng.normal(size=(16, 48, 2)).astype(np.float32)
    returns = features[:, :, 0] * 0.02 + rng.normal(0, 0.01, size=(16, 48))
    features[0, 10, 0] = np.nan
    torch.manual_seed(22)
    initial = NeuralLeaf(window_size=3, n_features=2, hidden_dim=8)
    torch.manual_seed(22)
    leaf = train_neural_leaf("cuda_check", features, returns, window_size=3,
                             hidden_dim=8, n_epochs=3, batch_size=256, device=tensor_device)
    assert leaf is not None
    assert next(leaf.parameters()).device.type == tensor_device.type
    assert all(torch.isfinite(p).all() for p in leaf.parameters())
    assert any(not torch.equal(p.detach().cpu(), q)
               for p, q in zip(leaf.parameters(), initial.parameters(), strict=True))
    prediction = leaf.evaluate(features)
    expected_nan = np.zeros((16, 48), dtype=bool)
    expected_nan[:, :2] = True
    expected_nan[0, 10:13] = True
    np.testing.assert_array_equal(np.isnan(prediction), expected_nan)

    registry = NeuralLeafRegistry(storage_dir=str(tmp_path))
    registry.register("cuda_check", leaf)
    checkpoint = registry.save("cuda_check")
    loaded = NeuralLeafRegistry(storage_dir=str(tmp_path)).load("reloaded", checkpoint)
    assert next(loaded.parameters()).device.type == "cpu"
    np.testing.assert_allclose(loaded.evaluate(features), prediction,
                               rtol=2e-5, atol=2e-6, equal_nan=True)
    np.testing.assert_allclose(loaded.evaluate(features, device=tensor_device), prediction,
                               rtol=2e-5, atol=2e-6, equal_nan=True)
    assert next(loaded.parameters()).device.type == tensor_device.type


def test_cupy_kernel_and_torch_interchange(tensor_device):
    import torch

    if tensor_device.type != "cuda":
        pytest.skip("CuPy requires CUDA")
    # A missing CuPy install on the CUDA runner is an error, not a skip.
    import cupy as cp

    values = np.linspace(-2, 2, 1024, dtype=np.float32)
    tensor = torch.tensor(values, device=tensor_device)
    shared = cp.from_dlpack(tensor)
    assert shared.data.ptr == tensor.data_ptr()
    kernel = cp.ElementwiseKernel("float32 x", "float32 y", "y = sinf(x) + x*x;",
                                   "factorminer_runtime_check")
    output = kernel(shared)
    restored = torch.from_dlpack(output)
    torch.cuda.synchronize()
    np.testing.assert_allclose(restored.cpu().numpy(), np.sin(values) + values**2,
                               rtol=2e-6, atol=2e-6)
    assert restored.device.type == "cuda"
    assert restored.data_ptr() == output.data.ptr


@pytest.mark.skipif(os.environ.get("FACTORMINER_TEST_EMBEDDINGS") != "1",
                    reason="Real embedding download is opt-in")
def test_real_formula_embeddings(tensor_device):
    from sentence_transformers import SentenceTransformer

    from factorminer.memory.embeddings import DEFAULT_MODEL_NAME, FormulaEmbedder

    embedder = FormulaEmbedder(use_faiss=False)
    embedder._model = SentenceTransformer(DEFAULT_MODEL_NAME, device=str(tensor_device))
    assert embedder._model.device.type == tensor_device.type
    formula = "Mean($close, 5)"
    vector = embedder.embed("mean", formula)
    embedder.embed("volume", "Std($volume, 20)")
    assert vector.shape == (384,)
    assert np.isfinite(vector).all()
    np.testing.assert_allclose(np.linalg.norm(vector), 1, atol=1e-6)
    assert embedder.find_nearest(formula, k=1)[0][0] == "mean"
    embedder._model.to("cpu")
    embedder.clear()
    np.testing.assert_allclose(embedder.embed("mean", formula), vector, rtol=2e-4, atol=2e-6)


def test_masked_division_has_finite_gradients(tensor_device):
    import torch

    from factorminer.operators.arithmetic import div_torch, inv_torch

    x = torch.tensor([2., 3., 4.], device=tensor_device, requires_grad=True)
    y = torch.tensor([2., 0., 1e-20], device=tensor_device, requires_grad=True)
    result = div_torch(x, y)
    torch.nan_to_num(result).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.5, 0., 0.], device=tensor_device))
    torch.testing.assert_close(y.grad, torch.tensor([-0.5, 0., 0.], device=tensor_device))
    y.grad = None
    result = inv_torch(y)
    torch.nan_to_num(result).sum().backward()
    torch.testing.assert_close(y.grad, torch.tensor([-0.25, 0., 0.], device=tensor_device))


def test_correlation_backend_preserves_ties_and_missing_periods(tensor_device, monkeypatch):
    import torch

    from factorminer.evaluation.correlation import compute_correlation_batch

    # Force CPU fallback as a distinct case even on a CUDA-equipped machine.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: tensor_device.type == "cuda")
    rng = np.random.default_rng(19)
    candidate = np.round(rng.normal(size=(12, 8)))
    library = np.stack([candidate.copy(), -candidate, np.ones_like(candidate)])
    candidate[:, 0] = np.nan
    candidate[4:, 1] = np.nan
    library[0, :3, 2] = np.nan
    library[1, :5, 3] = np.nan
    expected = compute_correlation_batch(candidate, library, backend="numpy")
    actual = compute_correlation_batch(candidate, library, backend="gpu")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert actual[2] == 0.0  # A constant factor has no ranking information.


def test_factor_workflow_matches_cpu_admission(tensor_device, monkeypatch):
    import torch

    from factorminer.core.parser import parse
    from factorminer.evaluation import pipeline

    monkeypatch.setattr(torch.cuda, "is_available", lambda: tensor_device.type == "cuda")
    rng = np.random.default_rng(54)
    data = {"$close": rng.normal(size=(24, 40)), "$volume": rng.normal(size=(24, 40))}
    returns = data["$close"] + data["$volume"] * 0.7
    library = pipeline.FactorLibraryView(
        factor_ids=["existing"], signals={"existing": data["$close"]}, ic_map={"existing": 1.0},
    )
    calls = []
    compute = pipeline.compute_correlation_batch

    def tracked_correlation(candidate, signals, backend="numpy"):
        calls.append(backend)
        return compute(candidate, signals, backend=backend)

    monkeypatch.setattr(pipeline, "compute_correlation_batch", tracked_correlation)
    outcomes = []
    for backend in ["numpy", "gpu"]:
        validator = pipeline.ValidationPipeline(
            returns, library,
            pipeline.PipelineConfig(backend=backend, num_workers=1, ic_threshold=0.01,
                                    icir_threshold=0.0, correlation_threshold=0.6),
            compute_signals_fn=lambda candidate, panel: parse(candidate.formula).evaluate(panel),
            data=data,
        )
        results = validator.evaluate_batch([
            pipeline.CandidateFactor("duplicate", "$close"),
            pipeline.CandidateFactor("new", "$volume"),
        ])
        outcomes.append({r.factor_name: r for r in results})
    assert calls.count("gpu") == 2  # A populated library must reach the GPU path.
    for name in ["duplicate", "new"]:
        expected, actual = (outcome[name] for outcome in outcomes)
        assert actual.admitted == expected.admitted
        assert actual.stage_passed == expected.stage_passed
        assert actual.rejection_reason == expected.rejection_reason
        np.testing.assert_allclose(actual.ic_series, expected.ic_series, equal_nan=True)
        np.testing.assert_allclose(actual.max_correlation, expected.max_correlation, atol=1e-10)
    assert not outcomes[1]["duplicate"].admitted
    assert outcomes[1]["new"].admitted
