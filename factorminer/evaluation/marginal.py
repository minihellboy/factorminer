"""Marginal value of a candidate factor beyond a reference basis and the library.

Standalone IC rewards candidates that restate signal a model already has.
:class:`MarginalBasis` measures what a candidate adds: on every period it
rank-transforms the cross-section, residualizes the candidate and the target
on a basis of reference factors (principal components of, e.g., Alpha158) and
the admitted library by ridge least squares, and reports

* ``partial_ic``: the mean daily correlation of the two residuals (a partial
  rank IC), with a Newey-West t-statistic;
* ``residual_share``: the share of the candidate's cross-sectional variance
  the basis does not explain (``1 - R^2``);
* optionally, the partial IC over the most recent ``recent_periods`` of the
  panel, so a candidate whose value has already decayed can be refused;
* long-only economics of the oriented residual under the benchmark's
  TopkDropout rule (hold the top ``topk_fraction`` of members, replace at most
  ``drop_fraction`` of the book per day): excess return over the universe
  mean, daily turnover, and the excess net of round-trip costs on it;
* top-of-book value, judged where a long-only book trades. A *base score*
  is the basis's pooled ridge prediction of the target (one coefficient
  vector over all periods). ``top_partial_ic`` is the partial IC restricted
  to each period's *contention zone*, the best ``zone_fraction`` of members
  by base score, where the book picks its names. ``top_lift_bps`` is how
  much adding the candidate's residual to the base score, at its pooled
  regression weight, changes the daily top-``topk_fraction`` spread (mean
  return of the top names minus the period's mean).

Ranks are average ranks among a period's members (finite targets) mapped to
``(-0.5, 0.5)``; a member without a value takes the neutral rank 0. Everything
is computed on the panel it is given, so callers pass the discovery split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]

from factorminer.evaluation.evidence.inference import compute_hac_mean_test


@dataclass(frozen=True)
class MarginalConfig:
    """How the basis is built and how marginal value is judged."""

    reference_components: int = 16  # principal components kept; 0 keeps every feature
    ridge: float = 1e-3  # ridge, relative to the mean basis variance per period
    min_assets: int = 30  # periods with fewer members are skipped
    topk_fraction: float = 1 / 6  # long-only book size (Qlib: top 50 of CSI300)
    drop_fraction: float = 0.1  # most of the book replaced per day (Qlib: 5 of 50)
    cost_bps: float = 20.0  # round-trip cost charged on book turnover
    min_t: float = 3.0  # admission: Newey-West t of the partial IC
    min_partial_ic: float = 0.005  # admission: |mean partial IC|
    min_residual_share: float = 0.2  # admission: novelty beyond the basis
    min_excess_bps: float | None = 0.0  # admission: daily gross book excess; None disables
    min_net_bps: float | None = None  # admission: daily book excess net of costs
    recent_periods: int = 0  # decay check over the panel's last periods; 0 disables
    min_recent_t: float = 1.0  # admission: same-sign partial IC t over recent_periods
    zone_fraction: float = 1 / 3  # contention zone: best share of members by base score
    min_top_t: float | None = None  # admission: same-sign partial IC t in the zone
    min_top_lift_bps: float | None = None  # admission: top-spread lift over the base score
    min_top_lift_t: float | None = None  # admission: Newey-West t of that lift
    # Out-of-sample-aware admission (2026-10-17). Pooled train-window t selects on noise and on the
    # pre-2017 regime: admitted factors kept 27% of their train partial IC in walk-forward tests, and
    # the strongest (|t| >= 6) shrank the most. Three statistics make the decay part of the rule.
    blocks: int = 0  # contiguous time blocks of the panel; 0 disables the block statistics
    block_purge: int = 6  # periods dropped between blocks (a 5-day label spans 6 closes)
    min_block_t: float | None = None  # admission: min over blocks of the same-sign partial IC t
    min_block_mean_minus_sd: float | None = None  # admission: mean - sd of the block t values
    retention: tuple[tuple[float, float], ...] = ()  # (|t| from, expected test/train ratio) steps
    min_shrunk_partial_ic: float | None = None  # admission: |partial IC| x retention(|t|)
    regime_periods: int = 0  # the panel's last periods form the current regime; 0 disables
    regime_weight: float = 1.0  # weight of those periods in the partial IC mean and t
    # Top-of-book admission against a frozen stage-1 score (``MarginalBasis(zone_score=...)``): the
    # zone is the best ``zone_fraction`` by that score, the score joins the basis, and with
    # ``zone_residual`` the zone statistics residualise candidate and target on zone rows only.
    zone_residual: bool = False

    def retention_at(self, t: float) -> float:
        """Expected out-of-sample share of a train partial IC with HAC statistic ``t``."""
        share = 1.0
        for threshold, value in sorted(self.retention):
            if abs(t) >= threshold:
                share = value
        return share

    def validate(self) -> None:
        if self.reference_components < 0:
            raise ValueError("reference_components must be >= 0")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if not 0.0 < self.topk_fraction <= 0.5:
            raise ValueError("topk_fraction must be in (0, 0.5]")
        if not 0.0 < self.drop_fraction <= 1.0:
            raise ValueError("drop_fraction must be in (0, 1]")
        if self.min_assets < 5:
            raise ValueError("min_assets must be >= 5")
        if self.recent_periods < 0:
            raise ValueError("recent_periods must be >= 0")
        if not self.topk_fraction <= self.zone_fraction <= 1.0:
            raise ValueError("zone_fraction must be in [topk_fraction, 1]")
        if self.blocks < 0 or self.blocks == 1:
            raise ValueError("blocks must be 0 (off) or at least 2")
        if self.block_purge < 0:
            raise ValueError("block_purge must be >= 0")
        if any(not 0 <= share <= 1 for _, share in self.retention):
            raise ValueError("retention shares must be in [0, 1]")
        if self.regime_periods < 0 or self.regime_weight <= 0:
            raise ValueError("regime_periods must be >= 0 and regime_weight positive")


@dataclass(frozen=True)
class MarginalStats:
    """Marginal-value summary of one candidate against one basis state."""

    partial_ic: float
    partial_ic_t: float
    partial_icir: float
    periods: int
    residual_share: float
    direction: float
    top_excess_bps: float
    top_turnover: float
    top_net_bps: float
    top_net_t: float
    basis_size: int
    recent_partial_ic: float = 0.0  # signed in the full-panel direction
    recent_partial_ic_t: float = 0.0
    top_partial_ic: float = 0.0  # in the contention zone, signed in the full-panel direction
    top_partial_ic_t: float = 0.0
    top_lift_bps: float = 0.0  # daily top-spread change from adding the candidate
    top_lift_t: float = 0.0
    block_t_min: float = 0.0  # min over contiguous blocks of the same-sign partial IC t
    block_t_mean_minus_sd: float = 0.0
    shrunk_partial_ic: float = 0.0  # |partial IC| x retention(|t|)

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def passes_marginal_admission(stats: MarginalStats, config: MarginalConfig) -> tuple[bool, str]:
    """Apply the marginal-value admission rule."""
    failures = []
    if not np.isfinite(stats.partial_ic_t) or stats.partial_ic_t < config.min_t:
        failures.append(f"partial IC t {stats.partial_ic_t:.2f} < {config.min_t:g}")
    if not abs(stats.partial_ic) >= config.min_partial_ic:
        failures.append(f"|partial IC| {abs(stats.partial_ic):.4f} < {config.min_partial_ic:g}")
    if not stats.residual_share >= config.min_residual_share:
        failures.append(
            f"residual share {stats.residual_share:.3f} < {config.min_residual_share:g}"
        )
    if config.min_excess_bps is not None and not stats.top_excess_bps > config.min_excess_bps:
        failures.append(f"book excess {stats.top_excess_bps:.2f} bps <= {config.min_excess_bps:g}")
    if config.min_net_bps is not None and not stats.top_net_bps >= config.min_net_bps:
        failures.append(f"book net {stats.top_net_bps:.2f} bps < {config.min_net_bps:g}")
    if config.recent_periods and not stats.recent_partial_ic_t >= config.min_recent_t:
        failures.append(
            f"recent partial IC t {stats.recent_partial_ic_t:.2f} < {config.min_recent_t:g}"
        )
    if config.min_top_t is not None and not stats.top_partial_ic_t >= config.min_top_t:
        failures.append(f"zone partial IC t {stats.top_partial_ic_t:.2f} < {config.min_top_t:g}")
    if config.min_top_lift_bps is not None and not stats.top_lift_bps > config.min_top_lift_bps:
        failures.append(f"top lift {stats.top_lift_bps:.2f} bps <= {config.min_top_lift_bps:g}")
    if config.min_top_lift_t is not None and not stats.top_lift_t >= config.min_top_lift_t:
        failures.append(f"top lift t {stats.top_lift_t:.2f} < {config.min_top_lift_t:g}")
    if config.min_block_t is not None and not stats.block_t_min >= config.min_block_t:
        failures.append(f"block partial IC t min {stats.block_t_min:.2f} < {config.min_block_t:g}")
    if (config.min_block_mean_minus_sd is not None
            and not stats.block_t_mean_minus_sd >= config.min_block_mean_minus_sd):
        failures.append(
            f"block t mean - sd {stats.block_t_mean_minus_sd:.2f} < {config.min_block_mean_minus_sd:g}"
        )
    if (config.min_shrunk_partial_ic is not None
            and not stats.shrunk_partial_ic >= config.min_shrunk_partial_ic):
        failures.append(
            f"shrunk partial IC {stats.shrunk_partial_ic:.4f} < {config.min_shrunk_partial_ic:g}"
        )
    if failures:
        return False, "Marginal value: " + "; ".join(failures)
    return True, (
        f"Marginal value: partial IC {stats.partial_ic:+.4f} (t {stats.partial_ic_t:.2f}), "
        f"residual share {stats.residual_share:.2f}, net {stats.top_net_bps:.2f} bps, "
        f"top lift {stats.top_lift_bps:+.2f} bps (t {stats.top_lift_t:.2f})"
    )


class MarginalBasis:
    """Per-period ridge basis of reference components and library factors.

    Parameters
    ----------
    returns:
        ``(M, T)`` target panel; finite cells define each period's members.
    reference:
        ``(M, T)`` reference factor panels (e.g. Alpha158 features); an
        iterator is consumed one panel at a time.
    """

    def __init__(
        self,
        returns: np.ndarray,
        reference: Iterable[np.ndarray],
        config: MarginalConfig | None = None,
        zone_score: np.ndarray | None = None,
    ) -> None:
        self.config = config or MarginalConfig()
        self.config.validate()
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim != 2:
            raise ValueError("returns must be a 2-D (assets, periods) panel")
        self.shape = returns.shape
        member = np.isfinite(returns)
        counts = member.sum(axis=0)
        self._periods = np.flatnonzero(counts >= self.config.min_assets)
        width = int(counts[self._periods].max()) if self._periods.size else 0
        # rows[p, j]: asset index of the j-th member in period p (-1 pads).
        self._rows = np.full((self._periods.size, width), -1, dtype=np.int64)
        for p, t in enumerate(self._periods):
            members = np.flatnonzero(member[:, t])
            self._rows[p, : members.size] = members
        self._active = self._rows >= 0
        self._n = self._active.sum(axis=1)
        self._raw_returns = self._gather(returns)
        self._y = self._ranks(self._raw_returns)

        reference_ranks = [self._ranks(self._gather(panel)) for panel in reference]
        self.reference_components, self.reference_explained = self._principal_components(
            reference_ranks
        )
        # A frozen stage-1 score (e.g. the cross-fitted h5 model): a permanent basis column that
        # also defines the contention zone and the base score the top lift is measured against.
        self._zone_ranks = None if zone_score is None else self._ranks(self._gather(zone_score))
        self._library: dict[object, np.ndarray] = {}
        self._factor()

    # -- layout -----------------------------------------------------------

    def _gather(self, panel: np.ndarray) -> np.ndarray:
        panel = np.asarray(panel, dtype=np.float64)
        if panel.shape != self.shape:
            raise ValueError(f"panel shape {panel.shape} != {self.shape}")
        values = panel[np.where(self._active, self._rows, 0), self._periods[:, None]]
        return np.where(self._active, values, np.nan)

    def _ranks(self, gathered: np.ndarray) -> np.ndarray:
        """Centered percentile ranks among members; missing members rank 0."""
        finite = np.isfinite(gathered) & self._active
        ranks = rankdata(np.where(finite, gathered, np.nan), axis=1, nan_policy="omit")
        valid = finite.sum(axis=1, keepdims=True)
        centered = np.where(finite, (ranks - 0.5) / np.maximum(valid, 1) - 0.5, 0.0)
        # Members only; pads stay exactly zero so sums over rows ignore them.
        centered = np.where(self._active, centered, 0.0)
        centered -= self._active * (centered.sum(axis=1, keepdims=True)
                                    / np.maximum(self._n, 1)[:, None])
        return centered

    def _principal_components(self, ranks: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Leading principal components of the pooled reference ranks."""
        if not ranks:
            return np.zeros((*self._rows.shape, 0)), 0.0
        stacked = np.stack(ranks, axis=-1)  # (P, N, F)
        keep = self.config.reference_components
        if keep == 0 or keep >= stacked.shape[-1]:
            return stacked, 1.0
        pooled = stacked[self._active]
        covariance = pooled.T @ pooled / max(len(pooled) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:keep]
        explained = float(eigenvalues[order].sum() / max(eigenvalues.sum(), 1e-300))
        return stacked @ eigenvectors[:, order], explained

    # -- basis maintenance ------------------------------------------------

    @property
    def library_keys(self) -> tuple[object, ...]:
        return tuple(self._library)

    @property
    def size(self) -> int:
        return int(self._basis.shape[-1])

    def add(self, key: object, signals: np.ndarray) -> None:
        self._library[key] = self._ranks(self._gather(signals))
        self._factor()

    def remove(self, key: object) -> None:
        if self._library.pop(key, None) is not None:
            self._factor()

    def sync(self, library: Mapping[object, np.ndarray]) -> None:
        """Make the library part of the basis equal to ``library``."""
        stale = [key for key in self._library if key not in library]
        fresh = [key for key in library if key not in self._library]
        for key in stale:
            self._library.pop(key)
        for key in fresh:
            self._library[key] = self._ranks(self._gather(library[key]))
        if stale or fresh:
            self._factor()

    def _factor(self) -> None:
        columns = [self.reference_components]
        if self._zone_ranks is not None:
            columns.append(self._zone_ranks[..., None])
        if self._library:
            columns.append(np.stack(list(self._library.values()), axis=-1))
        self._basis = np.concatenate(columns, axis=-1)
        k = self._basis.shape[-1]
        if k == 0:
            self._chol = None
            self._y_resid = self._y
            self._set_base_score(np.zeros(self._y.shape))
            return
        gram = np.einsum("pnk,pnj->pkj", self._basis, self._basis)
        scale = np.trace(gram, axis1=1, axis2=2) / k
        ridge = self.config.ridge * np.maximum(scale, 1e-12)
        pooled = gram.sum(axis=0)
        pooled[np.arange(k), np.arange(k)] += ridge.sum()
        beta = np.linalg.solve(pooled, np.einsum("pnk,pn->k", self._basis, self._y))
        gram[:, np.arange(k), np.arange(k)] += ridge[:, None]
        self._chol = np.linalg.cholesky(gram)
        self._y_resid = self._residualize(self._y)
        self._set_base_score(self._basis @ beta if self._zone_ranks is None else self._zone_ranks)

    def _set_base_score(self, score: np.ndarray) -> None:
        """Contention zone and top-book returns of the basis's pooled prediction."""
        self._base_score = score
        self._zone = self._top_mask(score, self.config.zone_fraction)
        self._base_top = self._top_return(score)

    def _top_mask(self, score: np.ndarray, fraction: float) -> np.ndarray:
        """Members in each period's best ``fraction`` by ``score`` (ties: lower index)."""
        ordered = np.argsort(np.where(self._active, -score, np.inf), axis=1, kind="stable")
        position = np.empty_like(ordered)
        np.put_along_axis(position, ordered, np.arange(ordered.shape[1])[None, :], axis=1)
        size = np.maximum(1, np.rint(fraction * self._n)).astype(np.int64)
        return np.asarray(self._active & (position < size[:, None]))

    def _top_return(self, score: np.ndarray) -> np.ndarray:
        """Mean raw return of each period's top-``topk_fraction`` names minus the mean."""
        top = self._top_mask(score, self.config.topk_fraction)
        returns = np.where(self._active, self._raw_returns, 0.0)
        top_mean = (returns * top).sum(axis=1) / np.maximum(top.sum(axis=1), 1)
        return np.asarray(top_mean - returns.sum(axis=1) / np.maximum(self._n, 1))

    def _residualize(self, values: np.ndarray) -> np.ndarray:
        if self._chol is None:
            return values
        rhs = np.einsum("pnk,pn->pk", self._basis, values)[..., None]
        beta = np.linalg.solve(
            np.swapaxes(self._chol, 1, 2), np.linalg.solve(self._chol, rhs)
        )[..., 0]
        residual: np.ndarray = values - np.einsum("pnk,pk->pn", self._basis, beta)
        return residual

    def _residualize_within(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Ridge residual of ``values`` on the basis fitted on ``mask`` rows only, centred within them.

        The relationship between features and returns at the top of a score differs from the
        one over the whole cross-section; a residual fitted on all rows can leave the zone with
        a component the zone's own fit would remove.
        """
        count = np.maximum(mask.sum(axis=1), 1)[:, None]
        v = np.where(mask, values, 0.0)
        v = np.where(mask, v - v.sum(axis=1, keepdims=True) / count, 0.0)
        if self._chol is None:
            return v
        basis = np.where(mask[..., None], self._basis, 0.0)
        basis = np.where(mask[..., None], basis - basis.sum(axis=1, keepdims=True) / count[..., None], 0.0)
        k = basis.shape[-1]
        gram = np.einsum("pnk,pnj->pkj", basis, basis)
        scale = np.trace(gram, axis1=1, axis2=2) / k
        gram[:, np.arange(k), np.arange(k)] += self.config.ridge * np.maximum(scale, 1e-12)[:, None]
        beta = np.linalg.solve(gram, np.einsum("pnk,pn->pk", basis, v)[..., None])[..., 0]
        return np.asarray(v - np.einsum("pnk,pk->pn", basis, beta))

    # -- statistics -------------------------------------------------------

    def evaluate(self, signals: np.ndarray, config: MarginalConfig | None = None) -> MarginalStats:
        """Marginal statistics of ``signals`` against the current basis.

        ``config`` (default: the basis's own) sets the statistics' options: the recent
        window, the regime weights, the block split and the retention table. The basis
        itself (components, ridge, zone) is fixed at construction.
        """
        cfg = config or self.config
        candidate = self._ranks(self._gather(signals))
        residual = self._residualize(candidate)
        num = np.sum(residual * self._y_resid, axis=1)
        den = np.sqrt(np.sum(residual**2, axis=1) * np.sum(self._y_resid**2, axis=1))
        usable = den > 1e-12
        series = num[usable] / den[usable]
        recent = self._periods[usable] >= self.shape[1] - cfg.recent_periods
        total = float(np.sum(candidate**2))
        residual_share = float(np.sum(residual**2) / total) if total > 1e-12 else 0.0
        if series.size < 3:
            return MarginalStats(0.0, 0.0, 0.0, int(series.size), residual_share, 1.0,
                                 0.0, 0.0, 0.0, 0.0, self.size)
        weights = self._regime_weights(usable, cfg)
        mean = float(np.average(series, weights=weights))
        std = float(series.std(ddof=1))
        hac = compute_hac_mean_test(series * weights)  # a weighted mean test: E[w x] = 0
        direction = 1.0 if mean >= 0 else -1.0
        block_min, block_mms = self._block_t(direction * series, usable, cfg)
        t_abs = float(abs(hac.t_stat)) if np.isfinite(hac.t_stat) else 0.0
        excess, turnover, net, net_t = self._bucket(direction * residual)
        top_ic, top_t, lift, lift_t = self._top_of_book(residual, direction, usable, candidate, cfg)
        recent_ic = recent_t = 0.0
        if cfg.recent_periods and recent.sum() >= 3:
            tail = direction * series[recent]
            recent_ic = float(tail.mean())
            tail_t = compute_hac_mean_test(tail).t_stat
            recent_t = float(tail_t) if np.isfinite(tail_t) else 0.0
        return MarginalStats(
            partial_ic=mean,
            partial_ic_t=t_abs,
            partial_icir=mean / std if std > 0 else 0.0,
            periods=int(series.size),
            residual_share=residual_share,
            direction=direction,
            top_excess_bps=excess,
            top_turnover=turnover,
            top_net_bps=net,
            top_net_t=net_t,
            basis_size=self.size,
            recent_partial_ic=recent_ic,
            recent_partial_ic_t=recent_t,
            top_partial_ic=top_ic,
            top_partial_ic_t=top_t,
            top_lift_bps=lift,
            top_lift_t=lift_t,
            block_t_min=block_min,
            block_t_mean_minus_sd=block_mms,
            shrunk_partial_ic=abs(mean) * cfg.retention_at(t_abs),
        )

    def _regime_weights(self, usable: np.ndarray, cfg: MarginalConfig) -> np.ndarray:
        """Per-period weights (mean 1): ``regime_weight`` on the panel's last ``regime_periods``."""
        weights = np.ones(int(usable.sum()))
        if cfg.regime_periods and cfg.regime_weight != 1.0:
            recent = self._periods[usable] >= self.shape[1] - cfg.regime_periods
            weights[recent] = cfg.regime_weight
            weights /= weights.mean()
        return weights

    def _block_t(self, series: np.ndarray, usable: np.ndarray, cfg: MarginalConfig) -> tuple[float, float]:
        """Same-sign HAC t of the partial IC in each of ``blocks`` contiguous blocks: (min, mean - sd).

        Blocks are equal spans of the panel's periods with ``block_purge`` periods dropped
        at the start of every block after the first, so a label window cannot straddle two
        blocks. A factor that is strong in one regime and absent in another has a low
        minimum even when its pooled t is high; the pooled statistic cannot tell them apart.
        """
        if cfg.blocks < 2:
            return 0.0, 0.0
        periods = self._periods[usable]
        edges = np.linspace(0, self.shape[1], cfg.blocks + 1)
        values = []
        for b in range(cfg.blocks):
            lo = edges[b] + (cfg.block_purge if b else 0)
            inside = (periods >= lo) & (periods < edges[b + 1])
            if inside.sum() < 3:
                values.append(0.0)
                continue
            t = compute_hac_mean_test(series[inside]).t_stat
            values.append(float(t) if np.isfinite(t) else 0.0)
        block_t = np.asarray(values)
        return float(block_t.min()), float(block_t.mean() - block_t.std(ddof=1))

    def _top_of_book(
        self, residual: np.ndarray, direction: float, usable: np.ndarray,
        candidate: np.ndarray | None = None, cfg: MarginalConfig | None = None,
    ) -> tuple[float, float, float, float]:
        """Zone partial IC (mean, t) and top-spread lift (bps, t) of a residual."""
        zone = self._zone
        cfg = cfg or self.config
        if cfg.zone_residual and candidate is not None:
            x = self._residualize_within(candidate, zone)
            y = self._residualize_within(self._y, zone)
        else:
            x = np.where(zone, residual, 0.0)
            y = np.where(zone, self._y_resid, 0.0)
            count = np.maximum(zone.sum(axis=1), 1)
            x = np.where(zone, x - x.sum(axis=1, keepdims=True) / count[:, None], 0.0)
            y = np.where(zone, y - y.sum(axis=1, keepdims=True) / count[:, None], 0.0)
        den = np.sqrt(np.sum(x**2, axis=1) * np.sum(y**2, axis=1))
        ok = usable & (den > 1e-12)
        zone_series = direction * np.sum(x * y, axis=1)[ok] / den[ok]
        # Pooled regression weight of the residual, as a linear model would fit it.
        weight = float(np.sum(residual * self._y_resid) / max(np.sum(residual**2), 1e-300))
        lift_series = (self._top_return(self._base_score + weight * residual) - self._base_top) * 1e4
        out = []
        for series in (zone_series, lift_series):
            if series.size < 3:
                out += [0.0, 0.0]
                continue
            t = compute_hac_mean_test(series).t_stat
            out += [float(series.mean()), float(t) if np.isfinite(t) else 0.0]
        return out[0], out[1], out[2], out[3]

    def _bucket(self, score: np.ndarray) -> tuple[float, float, float, float]:
        """TopkDropout book on ``score``: excess, turnover, net (bps/day), net t."""
        cfg = self.config
        held = np.zeros(self.shape[0], dtype=bool)
        excess = np.zeros(self._periods.size)
        turnover = np.zeros(self._periods.size)
        for p in range(self._periods.size):
            members = self._rows[p, : self._n[p]]
            values = score[p, : self._n[p]]
            returns = self._raw_returns[p, : self._n[p]]
            k = max(1, int(round(cfg.topk_fraction * members.size)))
            drop = max(1, int(round(cfg.drop_fraction * k)))
            order = np.argsort(-values, kind="stable")
            is_held = held[members]
            kept = int(is_held.sum())
            # Rank the book against the best non-held names, as TopkDropout does.
            fresh = order[~is_held[order]][: drop + max(k - kept, 0)]
            pool = np.concatenate([order[is_held[order]], fresh])
            pool = pool[np.argsort(-values[pool], kind="stable")]
            # Sell held names in the bottom ``drop`` of held + fresh, so a book
            # never sells a name to buy a worse one.
            sell = {int(i) for i in pool[-drop:] if is_held[i]}
            buys = list(fresh[: len(sell) + max(k - kept, 0)])
            book = [i for i in np.flatnonzero(is_held) if i not in sell] + buys
            dropped_out = int(held.sum()) - kept  # holdings that left the universe
            held[:] = False
            held[members[book]] = True
            turnover[p] = (len(sell) + dropped_out) / k if p else 1.0
            excess[p] = returns[book].mean() - returns.mean()
        net = excess - turnover * cfg.cost_bps / 1e4
        hac = compute_hac_mean_test(net)
        return (
            float(excess.mean() * 1e4),
            float(turnover[1:].mean()) if turnover.size > 1 else 1.0,
            float(net.mean() * 1e4),
            float(hac.t_stat) if np.isfinite(hac.t_stat) else 0.0,
        )
