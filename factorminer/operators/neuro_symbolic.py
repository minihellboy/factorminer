"""Hybrid neural-symbolic operators for HelixFactor.

WHY THIS MODULE EXISTS
----------------------
Symbolic expression trees give us interpretability and generalizability, but
they are limited by the vocabulary of hand-coded operators.  Neural leaves
bridge that gap: a tiny MLP trained on historical market data can discover
non-linear interaction patterns (e.g. volume-price divergence under high
intraday volatility) that no single hand-written formula captures.

The workflow is:
  1. Train a NeuralLeaf on historical data to maximise IC with next-period
     returns.  The leaf sees a rolling window of all available features.
  2. Insert the trained leaf into an expression tree as a NeuralLeafNode.
     It behaves like any other operator: (M, T) in -> (M, T) out.
  3. After validation, run distill_to_symbolic() to find the symbolic
     formula from the existing operator library that best approximates
     the neural leaf.  This restores interpretability while keeping the
     discovered signal.
  4. Replace NeuralLeafNode with the distilled formula for production.

Architecture constraints
------------------------
- Each NeuralLeaf has < 5 000 parameters (fits on CPU, fast inference).
- 2-layer MLP: input -> 32 hidden -> 1, with LayerNorm and GELU.
- Input: flattened rolling window of F features over the last W time steps.
- Output: scalar signal per (asset, time) pair, shape (M, T).
- Training uses a differentiable Pearson-IC proxy loss.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import — graceful degradation
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch is not installed.  NeuralLeaf training and inference will be "
        "unavailable.  Install torch to enable neuro-symbolic operators.",
        ImportWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical feature order — must match factorminer.core.types.FEATURES
_DEFAULT_FEATURES: list[str] = [
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$amt",
    "$vwap",
    "$returns",
]

_HIDDEN_DIM: int = 32  # keeps param count ~2 000 for window=10, F=8


# ===========================================================================
# NeuralLeaf — the learnable micro-model
# ===========================================================================

if _TORCH_AVAILABLE:

    class NeuralLeaf(nn.Module):
        """Tiny MLP operating on a rolling window of market features.

        Parameters
        ----------
        window_size : int
            Number of look-back time steps fed to the model.
        n_features : int
            Number of input feature channels (e.g. 8 for the standard OHLCV set).
        hidden_dim : int
            Width of the single hidden layer (default: 32).
        name : str
            Human-readable identifier used in DSL strings and logging.

        Input / Output shapes
        ---------------------
        ``forward`` expects a tensor of shape ``(M * T_valid, window_size * n_features)``
        where rows where the window is fully available have been pre-selected.
        It returns a tensor of shape ``(M * T_valid,)``.

        The public ``evaluate()`` method handles the full (M, T) -> (M, T) pipeline
        including NaN masking and output assembly.

        Parameter count
        ---------------
        With defaults (window=10, F=8, hidden=32):
            input_dim  = 10 * 8 = 80
            layer 1    = 80 * 32 + 32 = 2 592
            layer 2    = 32 * 1  + 1  = 33
            LayerNorm  = 2 * 80 + 2 * 32 = 224
            total      ≈ 2 849  (well under 5 000)
        """

        def __init__(
            self,
            window_size: int = 10,
            n_features: int = 8,
            hidden_dim: int = _HIDDEN_DIM,
            name: str = "NeuralLeaf",
        ) -> None:
            super().__init__()
            self.window_size = window_size
            self.n_features = n_features
            self.hidden_dim = hidden_dim
            self.name = name

            input_dim = window_size * n_features

            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
            )
            # Xavier init for stability
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        # ------------------------------------------------------------------
        # Core PyTorch forward
        # ------------------------------------------------------------------

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Map ``(N, window_size * n_features)`` -> ``(N,)``.

            Parameters
            ----------
            x : torch.Tensor, shape (N, window_size * n_features)

            Returns
            -------
            torch.Tensor, shape (N,)
            """
            return self.net(x).squeeze(-1)

        # ------------------------------------------------------------------
        # High-level evaluation: (M, T, F) -> (M, T) with NaN handling
        # ------------------------------------------------------------------

        def evaluate(
            self,
            features_3d: np.ndarray,
            device: torch.device | None = None,
        ) -> np.ndarray:
            """Evaluate the leaf on a full (M, T, F) market tensor.

            For each (asset, time) pair where a full window is available,
            the flattened window is fed through the MLP.  Positions where
            the window is not yet complete (the first ``window_size - 1``
            time steps) are filled with NaN.

            Parameters
            ----------
            features_3d : np.ndarray, shape (M, T, F)
                Stack of feature arrays, F channels, in the order given at
                construction time.
            device : torch.device, optional
                Where to place tensors.  Defaults to CPU.

            Returns
            -------
            np.ndarray, shape (M, T)
            """
            if not _TORCH_AVAILABLE:
                return np.full(features_3d.shape[:2], np.nan)

            device = device or torch.device("cpu")
            M, T, F = features_3d.shape
            W = self.window_size

            out = np.full((M, T), np.nan, dtype=np.float64)

            if T < W:
                return out

            # Build input matrix: (M, T - W + 1, W * F)
            # stride-trick to avoid copies
            X_windows = _build_windows_np(features_3d, W)  # (M, T-W+1, W*F)

            # Reshape to (M * (T-W+1), W*F)
            n_windows = T - W + 1
            X_flat = X_windows.reshape(M * n_windows, W * F).astype(np.float32)

            # Mask out rows that contain any NaN
            nan_mask = np.isnan(X_flat).any(axis=1)  # (M * n_windows,)

            X_valid = X_flat[~nan_mask]
            if X_valid.shape[0] == 0:
                return out

            self.eval()
            with torch.no_grad():
                x_tensor = torch.from_numpy(X_valid).to(device)
                preds = self.forward(x_tensor).cpu().numpy().astype(np.float64)

            # Scatter predictions back
            result_flat = np.full(M * n_windows, np.nan, dtype=np.float64)
            result_flat[~nan_mask] = preds
            result_2d = result_flat.reshape(M, n_windows)

            # Place into the last T - W + 1 columns of the output
            out[:, W - 1 :] = result_2d

            return out

        # ------------------------------------------------------------------
        # Utilities
        # ------------------------------------------------------------------

        def param_count(self) -> int:
            """Return the total number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def __repr__(self) -> str:
            return (
                f"NeuralLeaf(name={self.name!r}, window={self.window_size}, "
                f"features={self.n_features}, params={self.param_count()})"
            )

else:
    # Stub when torch is unavailable so type annotations still resolve.
    class NeuralLeaf:  # type: ignore[no-redef]
        """Stub NeuralLeaf (PyTorch unavailable)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.window_size = kwargs.get("window_size", 10)
            self.n_features = kwargs.get("n_features", 8)
            self.name = kwargs.get("name", "NeuralLeaf")

        def evaluate(self, features_3d: np.ndarray, **kwargs: Any) -> np.ndarray:
            return np.full(features_3d.shape[:2], np.nan)

        def param_count(self) -> int:
            return 0


# ===========================================================================
# Window construction helper (NumPy, no copy when F is contiguous)
# ===========================================================================

def _build_windows_np(x: np.ndarray, window: int) -> np.ndarray:
    """Create sliding windows from a (M, T, F) array.

    Returns
    -------
    np.ndarray, shape (M, T - window + 1, window * F)
        Each row is the flattened window of shape (window, F).
    """
    M, T, F = x.shape
    n = T - window + 1
    # Use stride tricks for zero-copy view
    s_m, s_t, s_f = x.strides
    shape = (M, n, window, F)
    strides = (s_m, s_t, s_t, s_f)
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return windows.reshape(M, n, window * F)


# ===========================================================================
# IC loss helpers (differentiable Pearson proxy)
# ===========================================================================

def _pearson_ic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative Pearson cross-sectional IC averaged over time steps.

    Both tensors must be shape ``(M,)`` (one time slice) or ``(N,)``
    (flattened batch).  The loss is ``-IC`` so gradient descent maximises IC.

    Parameters
    ----------
    pred : torch.Tensor, shape (N,)
    target : torch.Tensor, shape (N,)
    eps : float
        Denominator stabiliser.

    Returns
    -------
    torch.Tensor scalar
    """
    pred_m = pred - pred.mean()
    tgt_m = target - target.mean()
    cov = (pred_m * tgt_m).mean()
    denom = pred_m.std(unbiased=False).clamp(min=eps) * tgt_m.std(unbiased=False).clamp(min=eps)
    ic = cov / denom
    return -ic


def _l2_regularisation(model: NeuralLeaf, lam: float = 1e-4) -> torch.Tensor:
    """Compute L2 weight penalty (excludes bias and LayerNorm params)."""
    reg = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if "weight" in name and "norm" not in name:
            reg = reg + param.pow(2).sum()
    return lam * reg


# ===========================================================================
# Training procedure
# ===========================================================================

def train_neural_leaf(
    name: str,
    features: np.ndarray,
    returns: np.ndarray,
    window_size: int = 10,
    n_epochs: int = 100,
    lr: float = 1e-3,
    hidden_dim: int = _HIDDEN_DIM,
    val_fraction: float = 0.2,
    l2_lambda: float = 1e-4,
    batch_size: int = 2048,
    patience: int = 15,
    device: torch.device | None = None,
    verbose: bool = False,
) -> NeuralLeaf | None:
    """Train a NeuralLeaf to maximise cross-sectional IC with next-period returns.

    The leaf receives a rolling window of F features per (asset, time) pair
    and learns to output a signal that is cross-sectionally correlated with
    next-period returns.  Training uses time-based train/validation splits
    (no look-ahead: validation set = later time steps).

    Parameters
    ----------
    name : str
        Human-readable name for the leaf (e.g. ``"NeuralMomentum"``).
    features : np.ndarray, shape (M, T, F)
        Market feature tensor.  F must match the ``_DEFAULT_FEATURES`` list
        or be explicitly sized for the model.
    returns : np.ndarray, shape (M, T)
        Forward returns aligned to the same (M, T) grid.
    window_size : int
        Number of look-back bars for the rolling window.
    n_epochs : int
        Maximum training epochs.
    lr : float
        Adam learning rate.
    hidden_dim : int
        Width of the hidden layer.
    val_fraction : float
        Fraction of time steps reserved for validation (tail of the series).
    l2_lambda : float
        L2 regularisation coefficient.
    batch_size : int
        Mini-batch size over the flattened (asset, time) dimension.
    patience : int
        Early stopping patience (epochs without val IC improvement).
    device : torch.device, optional
        Computation device.  Defaults to CPU.
    verbose : bool
        Whether to log training progress at DEBUG level.

    Returns
    -------
    NeuralLeaf or None
        The trained leaf, or None if torch is unavailable or training fails.
    """
    if not _TORCH_AVAILABLE:
        logger.warning("train_neural_leaf: PyTorch unavailable, returning None.")
        return None

    device = device or torch.device("cpu")
    M, T, F = features.shape

    if T <= window_size:
        logger.warning(
            "train_neural_leaf(%s): T=%d <= window=%d, cannot train.", name, T, window_size
        )
        return None

    # ------------------------------------------------------------------
    # Build full flat dataset: (M * n_windows, window * F) and target (M * n_windows,)
    # ------------------------------------------------------------------
    n_windows = T - window_size + 1
    X_all = _build_windows_np(features, window_size)  # (M, n_windows, W*F)
    X_flat = X_all.reshape(M * n_windows, window_size * F).astype(np.float32)

    # Target: forward return at the LAST time step of each window (t = W-1 + k)
    # features[:, k : k+W, :] -> return at time k + W - 1
    # We align the return index to the last step in the window.
    ret_aligned = returns[:, window_size - 1 :]  # (M, n_windows)
    y_flat = ret_aligned.reshape(M * n_windows).astype(np.float32)

    # ------------------------------------------------------------------
    # Remove NaN rows (both in X and y)
    # ------------------------------------------------------------------
    valid_mask = (~np.isnan(X_flat).any(axis=1)) & (~np.isnan(y_flat))
    X_flat = X_flat[valid_mask]
    y_flat = y_flat[valid_mask]

    N = X_flat.shape[0]
    if N < 100:
        logger.warning(
            "train_neural_leaf(%s): only %d valid samples after NaN removal.", name, N
        )
        return None

    # ------------------------------------------------------------------
    # Temporal train / val split: preserve time ordering.
    # The valid_mask does not preserve temporal ordering in general, so
    # we use a simple head/tail split on the original time dimension.
    # ------------------------------------------------------------------
    # We rebuild from scratch with explicit temporal indexing to ensure
    # the val set is always strictly later in time.

    T_val_start = int(T * (1.0 - val_fraction))
    T_val_start = max(T_val_start, window_size)  # ensure at least one val window

    # Train windows: windows whose last time index < T_val_start
    # Last time index of window k = window_size - 1 + k  (0-indexed over n_windows)
    # => k < T_val_start - window_size + 1
    k_split = T_val_start - window_size + 1  # exclusive upper bound for train
    k_split = max(1, min(k_split, n_windows - 1))

    X_train_raw = _build_windows_np(features[:, :T_val_start, :], window_size)
    X_train_raw = X_train_raw.reshape(-1, window_size * F).astype(np.float32)
    y_train_raw = returns[:, window_size - 1 : T_val_start].reshape(-1).astype(np.float32)

    X_val_raw = _build_windows_np(features[:, T_val_start - window_size + 1 :, :], window_size)
    X_val_raw = X_val_raw.reshape(-1, window_size * F).astype(np.float32)
    y_val_raw = returns[:, T_val_start:].reshape(-1).astype(np.float32)

    def _clean(X: np.ndarray, y: np.ndarray):
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        return X[mask], y[mask]

    X_train, y_train = _clean(X_train_raw, y_train_raw)
    X_val, y_val = _clean(X_val_raw, y_val_raw)

    if X_train.shape[0] < 50:
        logger.warning(
            "train_neural_leaf(%s): too few training samples (%d).", name, X_train.shape[0]
        )
        return None

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    leaf = NeuralLeaf(
        window_size=window_size,
        n_features=F,
        hidden_dim=hidden_dim,
        name=name,
    ).to(device)

    optimizer = optim.Adam(leaf.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    N_train = X_train_t.shape[0]

    best_val_ic: float = -np.inf
    best_state: dict[str, Any] | None = None
    no_improve: int = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(n_epochs):
        leaf.train()
        # Shuffle each epoch
        perm = torch.randperm(N_train, device=device)
        X_shuf = X_train_t[perm]
        y_shuf = y_train_t[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N_train, batch_size):
            xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]
            if xb.shape[0] < 4:
                continue  # skip tiny last batch

            optimizer.zero_grad()
            pred = leaf(xb)
            ic_loss = _pearson_ic_loss(pred, yb)
            reg = _l2_regularisation(leaf, l2_lambda)
            loss = ic_loss + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(leaf.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # ------------------------------------------------------------------
        # Validation IC (no gradient)
        # ------------------------------------------------------------------
        leaf.eval()
        with torch.no_grad():
            val_pred = leaf(X_val_t)
            val_ic = -_pearson_ic_loss(val_pred, y_val_t).item()  # positive = good

        if val_ic > best_val_ic + 1e-5:
            best_val_ic = val_ic
            best_state = {k: v.clone() for k, v in leaf.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.debug(
                "Epoch %d/%d  train_loss=%.5f  val_IC=%.4f  best_val_IC=%.4f",
                epoch + 1,
                n_epochs,
                avg_loss,
                val_ic,
                best_val_ic,
            )

        if no_improve >= patience:
            logger.info(
                "train_neural_leaf(%s): early stopping at epoch %d (val_IC=%.4f).",
                name,
                epoch + 1,
                best_val_ic,
            )
            break

    # Restore best weights
    if best_state is not None:
        leaf.load_state_dict(best_state)

    logger.info(
        "Trained NeuralLeaf '%s': params=%d, best_val_IC=%.4f",
        name,
        leaf.param_count(),
        best_val_ic,
    )
    leaf.eval()
    return leaf


# ===========================================================================
# Symbolic Distillation
# ===========================================================================

@dataclass
class DistillationResult:
    """Result of distilling a neural leaf to a symbolic approximation.

    Attributes
    ----------
    formula : str
        The best-matching symbolic formula string (DSL notation).
    correlation : float
        Pearson correlation between the neural leaf output and the
        best symbolic approximation (over all valid positions).
    rank_correlation : float
        Spearman rank correlation (more relevant for factor quality).
    candidate_scores : dict
        Full mapping of formula -> correlation for all candidates tried.
    """

    formula: str
    correlation: float
    rank_correlation: float
    candidate_scores: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"DistillationResult(formula={self.formula!r}, "
            f"r={self.correlation:.4f}, rho={self.rank_correlation:.4f})"
        )


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two flat arrays, ignoring NaN."""
    from scipy.stats import spearmanr as _spearman  # local import to keep scipy optional

    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return 0.0
    rho, _ = _spearman(a[mask], b[mask])
    return float(rho)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flat arrays, ignoring NaN."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return 0.0
    am, bm = a[mask], b[mask]
    num = np.mean((am - am.mean()) * (bm - bm.mean()))
    denom = am.std() * bm.std()
    if denom < 1e-10:
        return 0.0
    return float(num / denom)


def _evaluate_symbolic_candidate(formula_fn, data: dict[str, np.ndarray]) -> np.ndarray | None:
    """Safely evaluate a symbolic candidate, returning None on failure."""
    try:
        result = formula_fn(data)
        if not isinstance(result, np.ndarray):
            return None
        if result.shape != next(iter(data.values())).shape:
            return None
        return result
    except Exception as exc:  # noqa: BLE001
        logger.debug("Symbolic candidate failed: %s", exc)
        return None


def _build_symbolic_candidates(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Generate all symbolic candidate outputs from the hand-coded operator library.

    Returns a dict mapping formula string -> (M, T) array.
    """
    # Import operators lazily to avoid circular imports
    from factorminer.core.expression_tree import (  # type: ignore[attr-defined]  # type: ignore[attr-defined]
        _ema,
        _rolling_apply,
        _ts_mean,
        _ts_rank,
        _ts_std,
    )

    candidates: dict[str, np.ndarray] = {}

    close = data.get("$close")
    volume = data.get("$volume")
    returns = data.get("$returns")
    high = data.get("$high")
    low = data.get("$low")
    data.get("$amt")
    vwap = data.get("$vwap")

    def _safe_add(name: str, arr: np.ndarray | None) -> None:
        if arr is not None and isinstance(arr, np.ndarray):
            candidates[name] = arr

    # EMA variants
    if close is not None:
        for w in (3, 5, 10, 20):
            _safe_add(f"EMA($close, {w})", _ema(close, w))
        # Delta (momentum)
        for w in (1, 3, 5, 10):
            M, T = close.shape
            out = np.full_like(close, np.nan, dtype=np.float64)
            if w < T:
                out[:, w:] = close[:, w:] - close[:, :-w]
            _safe_add(f"Delta($close, {w})", out)
        # Rolling return
        for w in (1, 3, 5, 10):
            M, T = close.shape
            out = np.full_like(close, np.nan, dtype=np.float64)
            if w < T:
                prev = close[:, :-w]
                ok = np.abs(prev) > 1e-10
                out[: , w:][ok] = close[:, w:][ok] / prev[ok] - 1.0
            _safe_add(f"Return($close, {w})", out)
        # TsRank
        for w in (5, 10, 20):
            _safe_add(f"TsRank($close, {w})", _rolling_apply(close, w, _ts_rank))
        # Rolling std
        for w in (5, 10, 20):
            _safe_add(f"Std($close, {w})", _rolling_apply(close, w, _ts_std))
        # Rolling mean
        for w in (5, 10, 20):
            _safe_add(f"Mean($close, {w})", _rolling_apply(close, w, _ts_mean))

    if volume is not None:
        for w in (5, 10, 20):
            _safe_add(f"TsRank($volume, {w})", _rolling_apply(volume, w, _ts_rank))
            _safe_add(f"EMA($volume, {w})", _ema(volume, w))
        for w in (1, 3, 5):
            M, T = volume.shape
            out = np.full_like(volume, np.nan, dtype=np.float64)
            if w < T:
                out[:, w:] = volume[:, w:] - volume[:, :-w]
            _safe_add(f"Delta($volume, {w})", out)

    if returns is not None:
        for w in (5, 10, 20):
            _safe_add(f"Std($returns, {w})", _rolling_apply(returns, w, _ts_std))
            _safe_add(f"Mean($returns, {w})", _rolling_apply(returns, w, _ts_mean))
            _safe_add(f"TsRank($returns, {w})", _rolling_apply(returns, w, _ts_rank))

    # VWAP-close spread (price-quality signal)
    if close is not None and vwap is not None:
        spread = close - vwap
        _safe_add("Sub($close, $vwap)", spread)
        for w in (5, 10):
            _safe_add(f"EMA(Sub($close,$vwap),{w})", _ema(spread, w))

    # High-low range (volatility proxy)
    if high is not None and low is not None:
        hl_range = high - low
        _safe_add("Sub($high, $low)", hl_range)
        for w in (5, 10, 20):
            _safe_add(f"Mean(Sub($high,$low),{w})", _rolling_apply(hl_range, w, _ts_mean))

    return candidates


def distill_to_symbolic(
    leaf: NeuralLeaf,
    data: dict[str, np.ndarray],
    feature_order: list[str] | None = None,
) -> DistillationResult:
    """Find the symbolic formula that best approximates the neural leaf.

    Evaluates the leaf on *data*, then computes the Pearson and Spearman
    correlation between the leaf output and every formula in a curated
    candidate set.  The candidate with the highest absolute Pearson
    correlation is chosen as the distillation target.

    Parameters
    ----------
    leaf : NeuralLeaf
        A trained neural leaf.
    data : dict[str, np.ndarray]
        Market data dict mapping feature name -> (M, T) array.
    feature_order : list of str, optional
        Order of features in the (M, T, F) stack passed to the leaf.
        Defaults to ``_DEFAULT_FEATURES``.

    Returns
    -------
    DistillationResult
    """
    feature_order = feature_order or _DEFAULT_FEATURES

    # Build feature tensor (M, T, F)
    ref_arr = next(iter(data.values()))
    M, T = ref_arr.shape
    len(feature_order)
    features_3d = np.stack(
        [data.get(f, np.full((M, T), np.nan)) for f in feature_order],
        axis=-1,
    )  # (M, T, F)

    # Evaluate neural leaf -> (M, T)
    leaf_output = leaf.evaluate(features_3d)

    # Flatten for correlation computation
    leaf_flat = leaf_output.ravel()

    # Build symbolic candidates
    candidates = _build_symbolic_candidates(data)

    scores: dict[str, float] = {}
    for formula, arr in candidates.items():
        r = _pearson_corr(leaf_flat, arr.ravel())
        scores[formula] = abs(r)  # rank by |r|

    if not scores:
        logger.warning("distill_to_symbolic: no symbolic candidates available.")
        return DistillationResult(
            formula="NeuralLeaf(no_candidates)",
            correlation=0.0,
            rank_correlation=0.0,
            candidate_scores={},
        )

    best_formula = max(scores, key=lambda k: scores[k])
    best_arr = candidates[best_formula]
    best_r = _pearson_corr(leaf_flat, best_arr.ravel())

    try:
        best_rho = _spearman_corr(leaf_flat, best_arr.ravel())
    except ImportError:
        best_rho = 0.0
        logger.debug("distill_to_symbolic: scipy not available, Spearman correlation skipped.")

    logger.info(
        "Distillation: best formula='%s', Pearson r=%.4f, Spearman rho=%.4f",
        best_formula,
        best_r,
        best_rho,
    )

    return DistillationResult(
        formula=best_formula,
        correlation=best_r,
        rank_correlation=best_rho,
        candidate_scores={k: v for k, v in sorted(scores.items(), key=lambda x: -x[1])},
    )


# ===========================================================================
# Expression Tree Integration
# ===========================================================================

class NeuralLeafNode:
    """A node that wraps a NeuralLeaf for use inside expression trees.

    Implements the same interface as ``factorminer.core.expression_tree.Node``
    so it can be dropped into any tree position that expects a (M, T) output.

    Crucially, this node does NOT inherit from ``Node`` to avoid coupling to
    the abstract base class, but it exposes the same public methods so that
    ExpressionTree machinery works without modification.

    Parameters
    ----------
    leaf : NeuralLeaf
        The trained (or untrained) neural leaf.
    feature_order : list of str, optional
        Feature channels fed to the leaf, in order.
        Defaults to ``_DEFAULT_FEATURES``.
    distilled_formula : str, optional
        If set, ``to_string()`` returns this formula instead of the neural
        leaf name.  Used after distillation for interpretable serialisation.
    """

    def __init__(
        self,
        leaf: NeuralLeaf,
        feature_order: list[str] | None = None,
        distilled_formula: str | None = None,
    ) -> None:
        self._leaf = leaf
        self._feature_order = feature_order or _DEFAULT_FEATURES
        self._distilled_formula = distilled_formula

    # ------------------------------------------------------------------
    # Node interface
    # ------------------------------------------------------------------

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Compute the leaf signal on market data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Maps feature names to (M, T) arrays.

        Returns
        -------
        np.ndarray, shape (M, T)
        """
        ref = next(iter(data.values()))
        M, T = ref.shape
        len(self._feature_order)
        features_3d = np.stack(
            [data.get(f, np.full((M, T), np.nan)) for f in self._feature_order],
            axis=-1,
        )
        return self._leaf.evaluate(features_3d)

    def to_string(self) -> str:
        """DSL serialisation.  Returns distilled formula when available."""
        if self._distilled_formula:
            return self._distilled_formula
        return f"NeuralLeaf({self._leaf.name})"

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1

    def clone(self) -> NeuralLeafNode:
        return NeuralLeafNode(
            leaf=self._leaf,  # shared reference — leaf weights are shared
            feature_order=list(self._feature_order),
            distilled_formula=self._distilled_formula,
        )

    def leaf_features(self) -> list[str]:
        return sorted(self._feature_order)

    def __repr__(self) -> str:
        return self.to_string()

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    @property
    def neural_leaf(self) -> NeuralLeaf:
        return self._leaf

    def set_distilled_formula(self, formula: str) -> None:
        """Pin the distilled formula used by ``to_string()``."""
        self._distilled_formula = formula


# ===========================================================================
# SymbolicShell — presents a neural leaf as a typed operator
# ===========================================================================

class SymbolicShell:
    """Wraps a NeuralLeaf as a callable operator compatible with the DSL.

    After distillation, the internal NeuralLeafNode can be replaced with
    its symbolic approximation by calling ``replace_with_symbolic()``.

    Parameters
    ----------
    name : str
        Operator name used in the registry and DSL strings.
    leaf_node : NeuralLeafNode
        The node wrapping the trained leaf.

    Usage
    -----
    ::

        shell = SymbolicShell("NeuralMomentum", leaf_node)
        signal = shell(data)                    # (M, T) array
        distilled = shell.distill(data)         # DistillationResult
        shell.replace_with_symbolic(distilled.formula)
        print(shell.formula_string)             # "EMA($close, 10)"
    """

    def __init__(self, name: str, leaf_node: NeuralLeafNode) -> None:
        self.name = name
        self._node = leaf_node
        self._is_distilled = False

    def __call__(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the operator on market data."""
        return self._node.evaluate(data)

    @property
    def formula_string(self) -> str:
        """Current DSL formula (neural or distilled)."""
        return self._node.to_string()

    @property
    def is_distilled(self) -> bool:
        return self._is_distilled

    def distill(
        self,
        data: dict[str, np.ndarray],
        feature_order: list[str] | None = None,
    ) -> DistillationResult:
        """Run distillation and return the result without modifying state."""
        return distill_to_symbolic(
            self._node.neural_leaf,
            data,
            feature_order=feature_order,
        )

    def replace_with_symbolic(self, formula: str) -> None:
        """Pin a distilled symbolic formula to this shell.

        After calling this, ``formula_string`` and ``to_string()`` return
        *formula* instead of the neural leaf name.

        Parameters
        ----------
        formula : str
            Symbolic formula string (DSL notation).
        """
        self._node.set_distilled_formula(formula)
        self._is_distilled = True
        logger.info("SymbolicShell '%s' replaced with symbolic formula: %s", self.name, formula)

    def __repr__(self) -> str:
        state = "distilled" if self._is_distilled else "neural"
        return f"SymbolicShell({self.name!r}, {state}, formula={self.formula_string!r})"


# ===========================================================================
# NeuralLeafRegistry
# ===========================================================================

class NeuralLeafRegistry:
    """Registry of named, trained NeuralLeaf models.

    Provides named storage, persistence, and lookup of NeuralLeaf instances.
    Trained weights are persisted via ``torch.save`` / ``torch.load``.

    Parameters
    ----------
    storage_dir : str, optional
        Directory where weights are saved.  Defaults to the system temp dir.

    Example
    -------
    ::

        registry = NeuralLeafRegistry(storage_dir="/tmp/neural_leaves")
        leaf = train_neural_leaf("NeuralMomentum", features, returns)
        registry.register("NeuralMomentum", leaf)
        registry.save("NeuralMomentum")

        # Later:
        registry.load("NeuralMomentum")
        leaf = registry.get("NeuralMomentum")
    """

    def __init__(self, storage_dir: str | None = None) -> None:
        import tempfile

        self._storage_dir = storage_dir or os.path.join(tempfile.gettempdir(), "neural_leaves")
        os.makedirs(self._storage_dir, exist_ok=True)
        self._leaves: dict[str, NeuralLeaf] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, name: str, leaf: NeuralLeaf) -> None:
        """Register a trained leaf under *name*."""
        self._leaves[name] = leaf
        logger.info("NeuralLeafRegistry: registered '%s'.", name)

    def get(self, name: str) -> NeuralLeaf | None:
        """Return the leaf registered under *name*, or None."""
        return self._leaves.get(name)

    def remove(self, name: str) -> None:
        """Remove a leaf from the in-memory registry."""
        self._leaves.pop(name, None)

    def available(self) -> list[str]:
        """Return sorted list of registered leaf names."""
        return sorted(self._leaves.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _path(self, name: str) -> str:
        safe_name = name.replace("/", "_").replace("\\", "_")
        return os.path.join(self._storage_dir, f"{safe_name}.pt")

    def save(self, name: str) -> str:
        """Save a registered leaf's weights to disk.

        Returns
        -------
        str
            Path where the file was saved.

        Raises
        ------
        KeyError
            If *name* is not registered.
        RuntimeError
            If PyTorch is unavailable.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available; cannot save NeuralLeaf.")
        leaf = self._leaves.get(name)
        if leaf is None:
            raise KeyError(f"NeuralLeafRegistry: no leaf named '{name}'.")
        path = self._path(name)
        torch.save(
            {
                "name": leaf.name,
                "window_size": leaf.window_size,
                "n_features": leaf.n_features,
                "hidden_dim": leaf.hidden_dim,
                "state_dict": leaf.state_dict(),
            },
            path,
        )
        logger.info("Saved NeuralLeaf '%s' to %s", name, path)
        return path

    def load(self, name: str, path: str | None = None) -> NeuralLeaf:
        """Load a leaf from disk and register it.

        Parameters
        ----------
        name : str
            Registry name to assign (may differ from the file's embedded name).
        path : str, optional
            Explicit file path.  If omitted, uses the default storage path.

        Returns
        -------
        NeuralLeaf
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available; cannot load NeuralLeaf.")
        file_path = path or self._path(name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NeuralLeaf weights not found at '{file_path}'.")
        ckpt = torch.load(file_path, map_location="cpu", weights_only=True)
        leaf = NeuralLeaf(
            window_size=ckpt["window_size"],
            n_features=ckpt["n_features"],
            hidden_dim=ckpt["hidden_dim"],
            name=ckpt["name"],
        )
        leaf.load_state_dict(ckpt["state_dict"])
        leaf.eval()
        self._leaves[name] = leaf
        logger.info("Loaded NeuralLeaf '%s' from %s", name, file_path)
        return leaf

    def save_all(self) -> dict[str, str]:
        """Save all registered leaves.  Returns name -> path mapping."""
        return {name: self.save(name) for name in self._leaves}

    def load_all(self) -> list[str]:
        """Load all .pt files from the storage directory.  Returns loaded names."""
        loaded = []
        for fname in os.listdir(self._storage_dir):
            if fname.endswith(".pt"):
                name = fname[:-3]
                try:
                    self.load(name)
                    loaded.append(name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load '%s': %s", name, exc)
        return loaded


# ===========================================================================
# NeuralOperatorIntegration — high-level orchestration
# ===========================================================================

@dataclass
class NeuralLeafConfig:
    """Configuration for a single named neural leaf.

    Attributes
    ----------
    name : str
        Registry name (e.g. ``"NeuralMomentum"``).
    window_size : int
        Rolling window size.
    n_epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    hidden_dim : int
        Hidden layer width.
    description : str
        Human-readable description.
    """

    name: str
    window_size: int = 10
    n_epochs: int = 100
    lr: float = 1e-3
    hidden_dim: int = _HIDDEN_DIM
    description: str = ""


class NeuralOperatorIntegration:
    """Orchestrates training, distillation, and persistence of neural leaves.

    This is the main entry point for integrating neural leaves into a
    HelixFactor workflow.

    Parameters
    ----------
    registry : NeuralLeafRegistry, optional
        Shared registry.  A new one is created if not provided.
    feature_order : list of str, optional
        Feature channels expected in the (M, T, F) input tensor.

    Example
    -------
    ::

        integration = NeuralOperatorIntegration()
        configs = [
            NeuralLeafConfig("NeuralMomentum", window_size=10),
            NeuralLeafConfig("NeuralReversal", window_size=5),
            NeuralLeafConfig("NeuralVolume",   window_size=10),
        ]
        integration.train_all_leaves(features_3d, returns, configs)
        distilled = integration.distill_all(data_dict)
        integration.save("/tmp/my_leaves")
    """

    def __init__(
        self,
        registry: NeuralLeafRegistry | None = None,
        feature_order: list[str] | None = None,
    ) -> None:
        self._registry = registry or NeuralLeafRegistry()
        self._feature_order = feature_order or _DEFAULT_FEATURES
        self._distillation_results: dict[str, DistillationResult] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_all_leaves(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        leaf_configs: list[NeuralLeafConfig],
        device: torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        """Train all listed neural leaves and register them.

        Parameters
        ----------
        features : np.ndarray, shape (M, T, F)
            Market feature tensor in the order given by ``feature_order``.
        returns : np.ndarray, shape (M, T)
            Forward returns for training targets.
        leaf_configs : list of NeuralLeafConfig
            One entry per leaf to train.
        device : torch.device, optional
        verbose : bool
            Pass through to training loop for debug logging.
        """
        for cfg in leaf_configs:
            logger.info("Training NeuralLeaf '%s'…", cfg.name)
            leaf = train_neural_leaf(
                name=cfg.name,
                features=features,
                returns=returns,
                window_size=cfg.window_size,
                n_epochs=cfg.n_epochs,
                lr=cfg.lr,
                hidden_dim=cfg.hidden_dim,
                device=device,
                verbose=verbose,
            )
            if leaf is not None:
                self._registry.register(cfg.name, leaf)
            else:
                logger.warning("Training failed for '%s', skipping.", cfg.name)

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    def distill_all(
        self,
        data: dict[str, np.ndarray],
    ) -> dict[str, str]:
        """Distill all registered leaves and return name -> best formula.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Market data dict (same format used for expression tree evaluation).

        Returns
        -------
        dict
            Maps leaf name to its best symbolic approximation formula string.
        """
        results: dict[str, str] = {}
        for name in self._registry.available():
            leaf = self._registry.get(name)
            if leaf is None:
                continue
            distilled = distill_to_symbolic(
                leaf, data, feature_order=self._feature_order
            )
            self._distillation_results[name] = distilled
            results[name] = distilled.formula
            logger.info(
                "Distilled '%s' -> '%s' (r=%.4f, rho=%.4f)",
                name,
                distilled.formula,
                distilled.correlation,
                distilled.rank_correlation,
            )
        return results

    # ------------------------------------------------------------------
    # Registry accessors
    # ------------------------------------------------------------------

    def get_available_leaves(self) -> list[str]:
        """Return names of all registered leaves."""
        return self._registry.available()

    def get_leaf(self, name: str) -> NeuralLeaf | None:
        """Return the NeuralLeaf registered under *name*, or None."""
        return self._registry.get(name)

    def get_distillation_result(self, name: str) -> DistillationResult | None:
        """Return the stored DistillationResult for *name*, or None."""
        return self._distillation_results.get(name)

    def as_node(self, name: str) -> NeuralLeafNode | None:
        """Return a NeuralLeafNode ready for use in an expression tree.

        If distillation has been run, the formula string is automatically set
        on the returned node.

        Parameters
        ----------
        name : str

        Returns
        -------
        NeuralLeafNode or None
        """
        leaf = self._registry.get(name)
        if leaf is None:
            return None
        distilled_formula = None
        if name in self._distillation_results:
            distilled_formula = self._distillation_results[name].formula
        return NeuralLeafNode(
            leaf=leaf,
            feature_order=self._feature_order,
            distilled_formula=distilled_formula,
        )

    def as_shell(self, name: str) -> SymbolicShell | None:
        """Return a SymbolicShell for *name*, or None if unknown."""
        node = self.as_node(name)
        if node is None:
            return None
        shell = SymbolicShell(name=name, leaf_node=node)
        if name in self._distillation_results:
            shell.replace_with_symbolic(self._distillation_results[name].formula)
        return shell

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all registered leaves to *path* (directory).

        Parameters
        ----------
        path : str
            Target directory.  Will be created if it does not exist.
        """
        os.makedirs(path, exist_ok=True)
        old_dir = self._registry._storage_dir
        self._registry._storage_dir = path
        self._registry.save_all()
        self._registry._storage_dir = old_dir
        logger.info("Saved %d neural leaves to %s", len(self._registry.available()), path)

    def load(self, path: str) -> None:
        """Load all .pt files from *path* into the registry.

        Parameters
        ----------
        path : str
            Directory containing .pt weight files.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"NeuralOperatorIntegration.load: '{path}' is not a directory.")
        old_dir = self._registry._storage_dir
        self._registry._storage_dir = path
        loaded = self._registry.load_all()
        self._registry._storage_dir = old_dir
        logger.info("Loaded %d neural leaves from %s", len(loaded), path)


# ===========================================================================
# Registry hook — exposes neural leaves to the operator registry
# ===========================================================================

# Global singleton, populated lazily when neural leaves are trained/loaded.
_GLOBAL_REGISTRY: NeuralLeafRegistry | None = None


def get_global_neural_registry() -> NeuralLeafRegistry:
    """Return (and lazily create) the global NeuralLeafRegistry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = NeuralLeafRegistry()
    return _GLOBAL_REGISTRY


def register_neural_leaves_in_operator_registry() -> None:
    """Expose registered neural leaves to the main operator OPERATOR_REGISTRY.

    This function should be called AFTER leaves have been trained / loaded.
    Each leaf is added to the registry with:
      - A synthetic OperatorSpec (category AUTO_INVENTED, arity 0 — the leaf
        takes the full data dict rather than individual array inputs).
      - A numpy_fn that calls ``leaf.evaluate(features_3d)`` after assembling
        the feature tensor from the data dict.
      - No PyTorch fn (the leaf already uses PyTorch internally).

    This allows the broader HelixFactor system to treat neural leaves as
    first-class operators that can appear in search spaces and fitness
    evaluation loops.
    """
    try:
        from factorminer.core.types import OperatorSpec, OperatorType, SignatureType
        from factorminer.operators.registry import OPERATOR_REGISTRY  # type: ignore[attr-defined]
    except ImportError:
        logger.debug("register_neural_leaves_in_operator_registry: operator registry not available.")
        return

    registry = get_global_neural_registry()
    for name in registry.available():
        if name in OPERATOR_REGISTRY:
            continue  # already registered

        leaf = registry.get(name)
        if leaf is None:
            continue

        feature_order = _DEFAULT_FEATURES

        # Capture leaf in closure
        def _make_np_fn(captured_leaf, captured_order):
            def _np_fn(data: dict[str, np.ndarray]) -> np.ndarray:
                ref = next(iter(data.values()))
                M, T = ref.shape
                len(captured_order)
                features_3d = np.stack(
                    [data.get(f, np.full((M, T), np.nan)) for f in captured_order],
                    axis=-1,
                )
                return captured_leaf.evaluate(features_3d)

            return _np_fn

        np_fn = _make_np_fn(leaf, feature_order)

        spec = OperatorSpec(
            name=name,
            arity=0,  # special: takes data dict, not individual arrays
            category=OperatorType.AUTO_INVENTED,
            signature=SignatureType.TIME_SERIES_TO_TIME_SERIES,
            description=f"NeuralLeaf: {name}",
        )
        OPERATOR_REGISTRY[name] = (spec, np_fn, None)
        logger.info("Registered neural leaf '%s' in OPERATOR_REGISTRY.", name)


# ===========================================================================
# Convenience: build standard leaves from mock data
# ===========================================================================

def build_default_neural_leaves(
    num_assets: int = 20,
    num_periods: int = 500,
    window_size: int = 10,
    n_epochs: int = 50,
    seed: int = 42,
    verbose: bool = False,
) -> NeuralOperatorIntegration:
    """Train the three standard neural leaves on synthetic mock data.

    Intended for quick experimentation and testing.  Uses
    ``factorminer.data.mock_data.generate_mock_data`` internally.

    The three leaves are:
    - ``NeuralMomentum``: captures price trend and momentum patterns.
    - ``NeuralReversal``: captures short-term mean-reversion signals.
    - ``NeuralVolume``: captures volume-price interaction signals.

    Parameters
    ----------
    num_assets : int
    num_periods : int
    window_size : int
    n_epochs : int
    seed : int
    verbose : bool

    Returns
    -------
    NeuralOperatorIntegration
        Fully initialised integration with trained leaves.
    """
    from factorminer.data.mock_data import MockConfig, generate_mock_data

    config = MockConfig(
        num_assets=num_assets,
        num_periods=num_periods,
        seed=seed,
        plant_alpha=True,
        alpha_strength=0.03,
    )
    df = generate_mock_data(config)

    # Pivot to (M, T) arrays
    df_sorted = df.sort_values(["asset_id", "datetime"])
    assets = sorted(df_sorted["asset_id"].unique())
    len(assets)
    T = df_sorted.groupby("asset_id").size().min()

    def _pivot(col: str) -> np.ndarray:
        return np.array(
            [df_sorted[df_sorted["asset_id"] == a][col].values[:T] for a in assets],
            dtype=np.float64,
        )

    close = _pivot("close")
    high = _pivot("high")
    low = _pivot("low")
    open_ = _pivot("open")
    volume = _pivot("volume")
    amount = _pivot("amount")
    # Derive returns and vwap
    ret = np.full_like(close, np.nan)
    ret[:, 1:] = close[:, 1:] / np.where(close[:, :-1] > 1e-10, close[:, :-1], np.nan) - 1.0
    vwap = (high + low + close) / 3.0

    data_dict: dict[str, np.ndarray] = {
        "$open": open_,
        "$high": high,
        "$low": low,
        "$close": close,
        "$volume": volume,
        "$amt": amount,
        "$vwap": vwap,
        "$returns": ret,
    }

    # Stack features in the canonical order
    features_3d = np.stack(
        [data_dict[f] for f in _DEFAULT_FEATURES],
        axis=-1,
    )  # (M, T, F)

    # Forward returns: shift by 1
    fwd_returns = np.full_like(close, np.nan)
    fwd_returns[:, :-1] = ret[:, 1:]

    configs = [
        NeuralLeafConfig(
            "NeuralMomentum",
            window_size=window_size,
            n_epochs=n_epochs,
            description="Learns price-momentum patterns from OHLCV windows",
        ),
        NeuralLeafConfig(
            "NeuralReversal",
            window_size=max(5, window_size // 2),
            n_epochs=n_epochs,
            description="Learns short-term mean-reversion signals",
        ),
        NeuralLeafConfig(
            "NeuralVolume",
            window_size=window_size,
            n_epochs=n_epochs,
            description="Learns volume-price interaction patterns",
        ),
    ]

    integration = NeuralOperatorIntegration(feature_order=_DEFAULT_FEATURES)
    integration.train_all_leaves(
        features=features_3d,
        returns=fwd_returns,
        leaf_configs=configs,
        verbose=verbose,
    )

    # Distill to symbolic
    integration.distill_all(data_dict)

    return integration


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    # Core classes
    "NeuralLeaf",
    "NeuralLeafNode",
    "NeuralLeafRegistry",
    "SymbolicShell",
    # Training
    "train_neural_leaf",
    "NeuralLeafConfig",
    # Distillation
    "distill_to_symbolic",
    "DistillationResult",
    # Orchestration
    "NeuralOperatorIntegration",
    # Registry integration
    "get_global_neural_registry",
    "register_neural_leaves_in_operator_registry",
    # Convenience
    "build_default_neural_leaves",
    # Constants
    "_DEFAULT_FEATURES",
    "_TORCH_AVAILABLE",
]
