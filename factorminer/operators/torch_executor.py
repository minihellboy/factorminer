"""Torch execution of compiled expression plans (``evaluation.backend: gpu``).

:class:`TorchPlanExecutor` runs each :class:`~factorminer.core.expression_plan.PlanStep`
on a torch device, mirroring ``core.expression_tree._dispatch_operator``
operator for operator: the same NaN handling, epsilon guards, warm-up NaNs,
tie order and window definitions, in float64. Window sums follow NumPy's
pairwise summation order and divisions are true IEEE divisions, so arithmetic
operators, rolling moments, regressions, order statistics and ranks are
bit-identical to the NumPy evaluator. Transcendental functions (``exp``,
``log1p``, ``tanh``, non-square ``pow``, hence Skew and Kurt) use CUDA's
libraries and can differ in the last unit of precision. Operators without a
device kernel run through the NumPy dispatch on host copies, so every formula
the NumPy evaluator accepts is accepted here. Signal caches key results by
backend.

Rolling windows are unfolded in blocks of at most ``block_elements`` window
cells, bounding device memory per operator independently of panel size.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np

from factorminer.core.expression_tree import _EPS, _dispatch_operator
from factorminer.core.types import OperatorSpec

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without the gpu extra
    torch = None  # type: ignore[assignment]

# Window cells per unfolded block: ~64 MiB per float64 temporary, so several
# campaigns can share one device.
_MAX_UPLOADS = 64
_DEFAULT_BLOCK_ELEMENTS = 1 << 23
_NAN = float("nan")


def _device_full(exc: BaseException) -> bool:
    """CUDA out-of-memory, from the caching allocator or the runtime itself."""
    if torch is not None and isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def torch_executor_available() -> bool:
    return torch is not None


class TorchPlanExecutor:
    """Execute plan steps on a torch device and hand results back as NumPy."""

    backend = "gpu"

    def __init__(
        self,
        device: str | None = None,
        *,
        block_elements: int = _DEFAULT_BLOCK_ELEMENTS,
    ) -> None:
        if torch is None:
            raise RuntimeError("evaluation.backend 'gpu' requires PyTorch (install the gpu extra)")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.block_elements = int(block_elements)
        self._uploads: OrderedDict[tuple, tuple[np.ndarray, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.host_fallbacks = 0  # steps run on the host after a device OOM

    # -- executor protocol ---------------------------------------------------

    def leaf(self, name: str, data: Mapping[str, np.ndarray]) -> Any:
        if name not in data:
            raise KeyError(f"Feature '{name}' not found in data. Available: {sorted(data.keys())}")
        try:
            return self._upload(data[name])
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _device_full(exc):
                raise
            torch.cuda.empty_cache()
            return data[name].astype(np.float64, copy=False)  # consumers fall back to the host

    def constant(self, value: float, data: Mapping[str, np.ndarray]) -> Any:
        for panel in data.values():
            try:
                return torch.full(panel.shape, value, dtype=torch.float64, device=self.device)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not _device_full(exc):
                    raise
                torch.cuda.empty_cache()
                return np.full(panel.shape, value, dtype=np.float64)
        raise ValueError("Cannot evaluate ConstantNode with empty data dict.")

    def operator(self, spec: OperatorSpec, inputs: list[Any], params: dict[str, float]) -> Any:
        kernel = _KERNELS.get(spec.name)
        if kernel is None:
            return self._host_fallback(spec, inputs, params)
        try:
            # An input left on the host by an earlier fallback moves back here.
            inputs = [self._to_device(v) if isinstance(v, np.ndarray) else v for v in inputs]
            return kernel(self, inputs, params)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _device_full(exc):
                raise
            # Another process holds the device: release cached blocks and run
            # this step on the exact host path instead of failing the formula.
            torch.cuda.empty_cache()
            with self._lock:
                self.host_fallbacks += 1
            return self._host_fallback(spec, inputs, params)

    def _host_fallback(self, spec: OperatorSpec, inputs: list[Any], params: dict[str, float]) -> Any:
        host = [self.to_numpy(value) for value in inputs]
        result = _dispatch_operator(spec, host, params)
        try:
            return self._to_device(result)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _device_full(exc):
                raise
            torch.cuda.empty_cache()
            return result  # stays on the host; later steps accept NumPy inputs

    def release(self) -> None:
        """Return cached device blocks so processes sharing the card fit."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def from_numpy(self, array: np.ndarray) -> Any:
        try:
            return self._to_device(array)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _device_full(exc):
                raise
            torch.cuda.empty_cache()
            with self._lock:
                self.host_fallbacks += 1
            return array  # stays on the host; later steps accept NumPy inputs

    def to_numpy(self, value: Any) -> np.ndarray:
        """A host float64 copy the caller owns."""
        if isinstance(value, np.ndarray):
            return np.array(value, dtype=np.float64, copy=True)
        if not hasattr(value, "detach"):
            raise TypeError(f"unexpected step value {type(value).__name__}")
        result: np.ndarray = value.detach().to("cpu", torch.float64).numpy()
        return result

    # -- helpers -------------------------------------------------------------

    def _upload(self, array: np.ndarray) -> Any:
        """Device copy of a feature panel, uploaded once per host buffer.

        Keyed by the memory a panel views rather than the array object:
        callers slice ``data_tensor[:, :, i]`` afresh for every batch, and
        those views share one upload. The entry holds the array so its buffer
        cannot be freed and its address reused; the oldest entries are dropped
        beyond ``_MAX_UPLOADS``.
        """
        interface = array.__array_interface__
        key = (interface["data"][0], array.shape, array.strides, array.dtype.str)
        with self._lock:
            cached = self._uploads.get(key)
            if cached is not None:
                self._uploads.move_to_end(key)
                return cached[1]
            tensor = self._to_device(array)
            self._uploads[key] = (array, tensor)
            while len(self._uploads) > _MAX_UPLOADS:
                self._uploads.popitem(last=False)
            return tensor

    def _to_device(self, array: np.ndarray) -> Any:
        host = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float64))
        return host.to(self.device, non_blocking=False)

    def window_blocks(self, rows: int, periods: int, window: int) -> Iterator[tuple[slice, int, int]]:
        """``(rows, first, last)`` output blocks whose unfolded windows fit the budget."""
        outputs = periods - window + 1
        if window < 1 or outputs < 1:
            return
        span = max(1, min(outputs, self.block_elements // window))
        block = max(1, self.block_elements // (span * window))
        for start in range(0, rows, block):
            for first in range(0, outputs, span):
                yield slice(start, start + block), first, min(outputs, first + span)

    def rolling(
        self,
        x: Any,
        window: int,
        reduce: Callable[..., Any],
        y: Any = None,
    ) -> Any:
        """Apply ``reduce`` to every full window; the first ``window - 1`` periods are NaN."""
        rows, periods = x.shape
        out = torch.full((rows, periods), _NAN, dtype=torch.float64, device=x.device)
        for row_block, first, last in self.window_blocks(rows, periods, window):
            columns = slice(first, last + window - 1)
            wx = x[row_block, columns].unfold(1, window, 1)
            if y is None:
                value = reduce(wx)
            else:
                value = reduce(wx, y[row_block, columns].unfold(1, window, 1))
            out[row_block, window - 1 + first:window - 1 + last] = value
        return out


# ---------------------------------------------------------------------------
# Kernels.  Each mirrors the NumPy function of the same name in
# core.expression_tree; window reductions act on the last axis.
# ---------------------------------------------------------------------------

def _div(x: Any, divisor: float) -> Any:
    """``x / divisor`` with IEEE division.

    CUDA divides by a host scalar as multiplication by its reciprocal, which
    rounds differently from NumPy (7 / 10 -> 0.7000000000000001); a device
    tensor divisor keeps true division.
    """
    return x / torch.tensor(float(divisor), dtype=torch.float64, device=x.device)


_PAIRWISE_BLOCK = 128


def _pairwise_sum(x: Any) -> Any:
    """Sum over the last axis in NumPy's pairwise order.

    NumPy reduces a contiguous axis with eight interleaved accumulators per
    block of up to 128 elements and splits longer runs in half. Reproducing
    that order, with IEEE additions, gives sums equal to NumPy's bit for bit.
    """
    n = x.shape[-1]
    if n < 8:
        total = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for i in range(n):
            total = total + x[..., i]
        return total
    if n <= _PAIRWISE_BLOCK:
        lanes = x[..., :8].clone()
        limit = n - n % 8
        for i in range(8, limit, 8):
            lanes += x[..., i:i + 8]
        r = [lanes[..., j] for j in range(8)]
        total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for i in range(limit, n):
            total = total + x[..., i]
        return total
    half = n // 2
    half -= half % 8
    return _pairwise_sum(x[..., :half]) + _pairwise_sum(x[..., half:])


def _nansum(x: Any) -> Any:
    return _pairwise_sum(torch.where(torch.isnan(x), 0.0, x))


def _sequential_prod(x: Any) -> Any:
    total = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
    for i in range(x.shape[-1]):
        total = total * x[..., i]
    return total


def _safe_div(a: Any, b: Any) -> Any:
    a, b = torch.broadcast_tensors(a, b)
    fallback = torch.where(torch.isnan(b), _NAN, 0.0).to(a.dtype)
    return torch.where(b.abs() > _EPS, a / b, fallback)


def _sign(x: Any) -> Any:
    return torch.where(torch.isnan(x), x, torch.sign(x))


def _nanmean(sx: Any, keepdim: bool = False) -> Any:
    counts = (~torch.isnan(sx)).sum(-1, keepdim=keepdim)
    total = _nansum(sx)
    if keepdim:
        total = total.unsqueeze(-1)
    return torch.where(counts > 0, total / counts.clamp_min(1), torch.full_like(total, _NAN))


def _nanvar(sx: Any, ddof: int = 1, keepdim: bool = False) -> Any:
    means = _nanmean(sx, keepdim=True)
    counts = (~torch.isnan(sx)).sum(-1, keepdim=keepdim)
    denominator = counts - ddof
    total = _nansum((sx - means) ** 2)
    if keepdim:
        total = total.unsqueeze(-1)
    return torch.where(
        denominator > 0, total / denominator.clamp_min(1), torch.full_like(total, _NAN)
    )


def _masked(sx: Any, value: Any) -> Any:
    """NaN where a window has no observations."""
    empty = torch.isnan(sx).all(-1)
    return torch.where(empty, torch.full_like(value, _NAN), value)


def _ts_max(sx: Any) -> Any:
    return _masked(sx, torch.where(torch.isnan(sx), -math.inf, sx).amax(-1))


def _ts_min(sx: Any) -> Any:
    return _masked(sx, torch.where(torch.isnan(sx), math.inf, sx).amin(-1))


def _ts_argmax(sx: Any) -> Any:
    return _masked(sx, torch.where(torch.isnan(sx), -math.inf, sx).argmax(-1).double())


def _ts_argmin(sx: Any) -> Any:
    return _masked(sx, torch.where(torch.isnan(sx), math.inf, sx).argmin(-1).double())


def _window_order_stat(sx: Any, q: float, *, median: bool) -> Any:
    """``nanmedian`` / ``nanquantile(method="linear")`` of every window."""
    window = sx.shape[-1]
    ordered = torch.sort(sx, dim=-1).values  # NaN sorts last
    count = window - torch.isnan(sx).sum(-1)
    if median:
        upper = torch.clamp(count // 2, 0, window - 1)
        lower = torch.clamp((count - 1) // 2, 0, window - 1)
        a = torch.gather(ordered, -1, lower.unsqueeze(-1)).squeeze(-1)
        b = torch.gather(ordered, -1, upper.unsqueeze(-1)).squeeze(-1)
        value = torch.where(count % 2 == 1, b, (a + b) / 2)
    else:
        virtual = (count - 1).double() * q
        lower_f = torch.floor(virtual)
        gamma = virtual - lower_f
        lower_idx = torch.clamp(lower_f.long(), 0, window - 1)
        upper_idx = torch.minimum(lower_idx + 1, torch.clamp(count - 1, min=0))
        below = torch.gather(ordered, -1, lower_idx.unsqueeze(-1)).squeeze(-1)
        above = torch.gather(ordered, -1, upper_idx.unsqueeze(-1)).squeeze(-1)
        diff = above - below
        value = torch.where(gamma >= 0.5, above - diff * (1 - gamma), below + diff * gamma)
    return torch.where(count == 0, torch.full_like(value, _NAN), value)


def _moment(sx: Any, power: int) -> Any:
    m = _nanmean(sx, keepdim=True)
    s = torch.sqrt(_nanvar(sx, keepdim=True))
    s = torch.where(s > _EPS, s, torch.ones_like(s))
    return _nanmean(((sx - m) / s) ** power)


def _ts_skew(sx: Any) -> Any:
    n = sx.shape[-1]
    return _div(_moment(sx, 3) * n**2, max((n - 1) * (n - 2), 1))


def _ts_rank(sx: Any) -> Any:
    latest = sx[..., -1:]
    rank = _div((sx <= latest).sum(-1).double(), sx.shape[-1])
    return torch.where(torch.isnan(latest[..., 0]), _NAN, rank)


def _ts_corr(sx: Any, sy: Any) -> Any:
    dx = sx - _nanmean(sx, keepdim=True)
    dy = sy - _nanmean(sy, keepdim=True)
    cov = _nanmean(dx * dy)
    denom = torch.sqrt(_nanvar(sx)) * torch.sqrt(_nanvar(sy))
    fallback = torch.where(torch.isnan(denom), _NAN, 0.0).to(cov.dtype)
    return torch.where(denom > _EPS, cov / denom, fallback)


def _ts_cov(sx: Any, sy: Any) -> Any:
    return _nanmean((sx - _nanmean(sx, keepdim=True)) * (sy - _nanmean(sy, keepdim=True)))


def _ts_beta(sx: Any, sy: Any) -> Any:
    mean_x, mean_y = _nanmean(sx, keepdim=True), _nanmean(sy, keepdim=True)
    dy = sy - mean_y
    var_y = _nansum(dy**2)
    cov_xy = _nansum((sx - mean_x) * dy)
    empty = torch.isnan(mean_x[..., 0]) | torch.isnan(mean_y[..., 0])
    fallback = torch.where(empty, _NAN, 0.0).to(cov_xy.dtype)
    return torch.where(var_y > _EPS, cov_xy / var_y, fallback)


def _ts_resid(sx: Any, sy: Any) -> Any:
    beta = _ts_beta(sx, sy)
    predicted = _nanmean(sx) + beta * (sy[..., -1] - _nanmean(sy))
    return sx[..., -1] - predicted


def _time_index(window: int, like: Any) -> tuple[Any, float, float]:
    t_vals = torch.arange(window, dtype=torch.float64, device=like.device)
    t_mean = (window - 1) / 2.0
    t_var = float(np.sum((np.arange(window, dtype=np.float64) - t_mean) ** 2))
    return t_vals, t_mean, t_var


def _linreg_slope(sx: Any) -> Any:
    t_vals, t_mean, t_var = _time_index(sx.shape[-1], sx)
    cov = _nansum((sx - _nanmean(sx, keepdim=True)) * (t_vals - t_mean))
    return _masked(sx, _div(cov, max(t_var, _EPS)))


def _linreg_rsquare(sx: Any) -> Any:
    t_vals, t_mean, t_var = _time_index(sx.shape[-1], sx)
    x_mean = _nanmean(sx, keepdim=True)
    slope = _div(_nansum((sx - x_mean) * (t_vals - t_mean)), max(t_var, _EPS))
    intercept = x_mean[..., 0] - slope * t_mean
    fitted = slope.unsqueeze(-1) * t_vals + intercept.unsqueeze(-1)
    ss_res = _nansum((sx - fitted) ** 2)
    ss_tot = _nansum((sx - x_mean) ** 2)
    ratio = torch.where(ss_tot > _EPS, ss_res / ss_tot, torch.full_like(ss_tot, _NAN))
    return 1.0 - ratio


def _shift(x: Any, w: int, combine: Callable[[Any, Any], Any]) -> Any:
    out = torch.full_like(x, _NAN)
    if w < x.shape[1]:
        out[:, w:] = combine(x[:, w:], x[:, :-w] if w > 0 else x)
    return out


def _weighted(ex: TorchPlanExecutor, x: Any, weights: np.ndarray) -> Any:
    w = torch.as_tensor(weights, dtype=torch.float64, device=x.device)
    return ex.rolling(x, len(weights), lambda sx: _pairwise_sum(sx * w))


def _wma(ex: TorchPlanExecutor, x: Any, window: int) -> Any:
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights /= weights.sum()
    return _weighted(ex, x, weights)


def _decay(ex: TorchPlanExecutor, x: Any, window: int) -> Any:
    alpha = 2.0 / (window + 1)
    weights = np.array([alpha * (1 - alpha) ** i for i in range(window)][::-1])
    return _weighted(ex, x, weights)


def _cs_rank(x: Any) -> Any:
    rows = x.shape[0]
    order = torch.sort(x, dim=0, stable=True).indices  # NaN last, ties in row order
    positions = torch.empty_like(order)
    arange = torch.arange(rows, device=x.device).unsqueeze(1).expand_as(order)
    positions.scatter_(0, order, arange)
    valid = ~torch.isnan(x)
    counts = valid.sum(0, keepdim=True)
    ranked = (positions + 1).double() / counts.clamp_min(1)
    return torch.where(valid, ranked, torch.full_like(ranked, _NAN))


def _linreg_parts(ex: TorchPlanExecutor, x: Any, w: int) -> tuple[Any, Any]:
    _t_vals, t_mean, _t_var = _time_index(max(w, 1), x)
    slope = ex.rolling(x, w, _linreg_slope)
    intercept = ex.rolling(x, w, _nanmean) - slope * t_mean
    return slope, intercept


def _linreg_fitted(ex: TorchPlanExecutor, x: Any, w: int) -> Any:
    slope, intercept = _linreg_parts(ex, x, w)
    return intercept + slope * float(w - 1)


def _linreg_resid(ex: TorchPlanExecutor, x: Any, w: int) -> Any:
    fitted = _linreg_fitted(ex, x, w)
    out = torch.full_like(x, _NAN)
    if 0 < w <= x.shape[1]:
        out[:, w - 1:] = x[:, w - 1:] - fitted[:, w - 1:]
    return out


def _hma(ex: TorchPlanExecutor, x: Any, w: int) -> Any:
    half_w = max(w // 2, 1)
    sqrt_w = max(int(math.sqrt(w)), 1)
    diff = 2 * _wma(ex, x, half_w) - _wma(ex, x, w)
    return _wma(ex, diff, sqrt_w)


def _mask_missing(values: Any, source: Any) -> Any:
    return torch.where(torch.isnan(source), _NAN, values)


def _window(params: dict[str, float]) -> int:
    return int(params.get("window", 0))


def _rolling_kernel(reduce: Callable[..., Any]) -> Callable[..., Any]:
    def run(ex: TorchPlanExecutor, xs: list[Any], params: dict[str, float]) -> Any:
        w = _window(params)
        if w < 1:
            raise _DeferToHostError
        return ex.rolling(xs[0], w, reduce, xs[1] if len(xs) > 1 else None)
    return run


class _DeferToHostError(Exception):
    """Raised by a kernel to defer to the NumPy dispatch."""


def _cmp(fn: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    return lambda ex, xs, p: fn(xs[0], xs[1]).double()


def _lag(combine: Callable[[Any, Any], Any]) -> Callable[..., Any]:
    def run(ex: TorchPlanExecutor, xs: list[Any], params: dict[str, float]) -> Any:
        w = _window(params)
        if w < 1:
            raise _DeferToHostError
        return _shift(xs[0], w, combine)
    return run


def _log_return(cur: Any, prev: Any) -> Any:
    keep = torch.isnan(prev) | (prev.abs() > _EPS)
    denominator = torch.where(keep, prev, torch.ones_like(prev))
    return torch.log(_safe_div(cur, denominator).abs() + _EPS)


def _signed_power(ex: TorchPlanExecutor, xs: list[Any], params: dict[str, float]) -> Any:
    return _sign(xs[0]) * xs[0].abs() ** float(params.get("exponent", 2.0))


_KERNELS: dict[str, Callable[[TorchPlanExecutor, list[Any], dict[str, float]], Any]] = {
    "Add": lambda ex, xs, p: xs[0] + xs[1],
    "Sub": lambda ex, xs, p: xs[0] - xs[1],
    "Mul": lambda ex, xs, p: xs[0] * xs[1],
    "Div": lambda ex, xs, p: _safe_div(xs[0], xs[1]),
    "Neg": lambda ex, xs, p: -xs[0],
    "Abs": lambda ex, xs, p: xs[0].abs(),
    "Sign": lambda ex, xs, p: _sign(xs[0]),
    "Log": lambda ex, xs, p: _sign(xs[0]) * torch.log1p(xs[0].abs()),
    "Sqrt": lambda ex, xs, p: _sign(xs[0]) * torch.sqrt(xs[0].abs()),
    "Square": lambda ex, xs, p: xs[0] * xs[0],
    "Pow": lambda ex, xs, p: _sign(xs[0]) * xs[0].abs() ** xs[1],
    "SignedPower": _signed_power,
    "Power": lambda ex, xs, p: torch.pow(xs[0], float(p.get("exponent", 2.0))),
    "Exp": lambda ex, xs, p: torch.exp(torch.clamp(xs[0], -50.0, 50.0)),
    "Tanh": lambda ex, xs, p: torch.tanh(xs[0]),
    "Max": lambda ex, xs, p: torch.maximum(xs[0], xs[1]),
    "Max2": lambda ex, xs, p: torch.maximum(xs[0], xs[1]),
    "Min": lambda ex, xs, p: torch.minimum(xs[0], xs[1]),
    "Min2": lambda ex, xs, p: torch.minimum(xs[0], xs[1]),
    "Clip": lambda ex, xs, p: torch.clamp(
        xs[0], float(p.get("lower", -3.0)), float(p.get("upper", 3.0))
    ),
    "Inv": lambda ex, xs, p: _safe_div(torch.ones_like(xs[0]), xs[0]),
    # Rolling statistics
    "Mean": _rolling_kernel(_nanmean),
    "SMA": _rolling_kernel(_nanmean),
    "Std": _rolling_kernel(lambda sx: torch.sqrt(_nanvar(sx))),
    "Var": _rolling_kernel(_nanvar),
    "Skew": _rolling_kernel(_ts_skew),
    "Kurt": _rolling_kernel(lambda sx: _moment(sx, 4) - 3.0),
    "Median": _rolling_kernel(lambda sx: _window_order_stat(sx, 0.5, median=True)),
    "Med": _rolling_kernel(lambda sx: _window_order_stat(sx, 0.5, median=True)),
    "Sum": _rolling_kernel(lambda sx: _masked(sx, _nansum(sx))),
    "Prod": _rolling_kernel(
        lambda sx: _masked(sx, _sequential_prod(torch.where(torch.isnan(sx), 1.0, sx)))
    ),
    "Product": _rolling_kernel(
        lambda sx: _masked(sx, _sequential_prod(torch.where(torch.isnan(sx), 1.0, sx)))
    ),
    "TsMax": _rolling_kernel(_ts_max),
    "TsMin": _rolling_kernel(_ts_min),
    "TsArgMax": _rolling_kernel(_ts_argmax),
    "TsArgMin": _rolling_kernel(_ts_argmin),
    "TsRank": _rolling_kernel(_ts_rank),
    "CountNaN": _rolling_kernel(lambda sx: torch.isnan(sx).sum(-1).double()),
    "CountNotNaN": _rolling_kernel(lambda sx: (~torch.isnan(sx)).sum(-1).double()),
    "Corr": _rolling_kernel(_ts_corr),
    "Cov": _rolling_kernel(_ts_cov),
    "Beta": _rolling_kernel(_ts_beta),
    "Resid": _rolling_kernel(_ts_resid),
    # Time series
    "Delta": _lag(lambda cur, prev: cur - prev),
    "Delay": _lag(lambda cur, prev: prev),
    "Return": _lag(lambda cur, prev: _safe_div(cur - prev, prev)),
    "LogReturn": _lag(_log_return),
    "WMA": lambda ex, xs, p: _wma(ex, xs[0], _window(p)),
    "Decay": lambda ex, xs, p: _decay(ex, xs[0], _window(p)),
    "TsDecay": lambda ex, xs, p: _decay(ex, xs[0], _window(p)),
    "CumMax": lambda ex, xs, p: _mask_missing(
        torch.cummax(torch.nan_to_num(xs[0], nan=-math.inf), 1).values, xs[0]
    ),
    "CumMin": lambda ex, xs, p: _mask_missing(
        torch.cummin(torch.nan_to_num(xs[0], nan=math.inf), 1).values, xs[0]
    ),
    "HMA": lambda ex, xs, p: _hma(ex, xs[0], _window(p)),
    # Cross-sectional
    "CsRank": lambda ex, xs, p: _cs_rank(xs[0]),
    "CsQuantile": lambda ex, xs, p: torch.clamp(
        torch.ceil(_cs_rank(xs[0]) * int(p.get("n_bins", 5))), 1, int(p.get("n_bins", 5))
    ),
    # Regression
    "TsLinReg": lambda ex, xs, p: _linreg_fitted(ex, xs[0], _window(p)),
    "TsLinRegSlope": _rolling_kernel(_linreg_slope),
    "Slope": _rolling_kernel(_linreg_slope),
    "TsLinRegIntercept": lambda ex, xs, p: _linreg_parts(ex, xs[0], _window(p))[1],
    "TsLinRegResid": lambda ex, xs, p: _linreg_resid(ex, xs[0], _window(p)),
    "Resi": lambda ex, xs, p: _linreg_resid(ex, xs[0], _window(p)),
    "Rsquare": _rolling_kernel(_linreg_rsquare),
    "Quantile": _rolling_kernel(lambda sx: sx),  # replaced below: needs q
    # Logical
    "IfElse": lambda ex, xs, p: torch.where(xs[0] > 0, xs[1], xs[2]),
    "Greater": _cmp(lambda a, b: a > b),
    "GreaterEqual": _cmp(lambda a, b: a >= b),
    "Less": _cmp(lambda a, b: a < b),
    "LessEqual": _cmp(lambda a, b: a <= b),
    "Equal": _cmp(lambda a, b: (a - b).abs() < _EPS),
    "Eq": _cmp(lambda a, b: (a - b).abs() < _EPS),
    "Ne": _cmp(lambda a, b: (a - b).abs() >= _EPS),
    "And": _cmp(lambda a, b: (a > 0) & (b > 0)),
    "Or": _cmp(lambda a, b: (a > 0) | (b > 0)),
    "Not": lambda ex, xs, p: (xs[0] <= 0).double(),
}


def _quantile(ex: TorchPlanExecutor, xs: list[Any], params: dict[str, float]) -> Any:
    w = _window(params)
    if w < 1:
        raise _DeferToHostError
    q = float(params.get("q", 0.5))
    return ex.rolling(xs[0], w, lambda sx: _window_order_stat(sx, q, median=False))


_KERNELS["Quantile"] = _quantile
# Host fallback (exact, and cheap on the host): EMA, DEMA, KAMA, CumSum and
# CumProd are sequential recurrences that device scans would reassociate, and
# CsZScore, CsDemean, CsNeutralize and CsScale reduce over assets in NumPy's
# column order.


def _with_fallback(
    name: str, kernel: Callable[[TorchPlanExecutor, list[Any], dict[str, float]], Any]
) -> Callable[[TorchPlanExecutor, list[Any], dict[str, float]], Any]:
    """Defer degenerate windows (< 1), whose NumPy loop semantics are irregular."""
    def run(ex: TorchPlanExecutor, xs: list[Any], params: dict[str, float]) -> Any:
        try:
            if "window" in params and int(params["window"]) < 1:
                raise _DeferToHostError
            return kernel(ex, xs, params)
        except _DeferToHostError:
            from factorminer.core.types import OPERATOR_REGISTRY

            return ex._host_fallback(OPERATOR_REGISTRY[name], xs, params)
    return run


_KERNELS = {name: _with_fallback(name, kernel) for name, kernel in _KERNELS.items()}


def torch_kernel_names() -> frozenset[str]:
    """Operators executed natively on the device (others use the host fallback)."""
    return frozenset(_KERNELS)
