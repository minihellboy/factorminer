"""Combinatorial Purged Cross-Validation and Probability of Backtest Overfitting.

These two tools are the standard companions to the Deflated Sharpe Ratio
(`factorminer.evaluation.significance.DeflatedSharpeCalculator`), all from the
same research lineage:

- **Combinatorially Purged Cross-Validation (CPCV)**: Lopez de Prado,
  *Advances in Financial Machine Learning* (2018), ch. 7 and 12. Ordinary
  k-fold CV leaks information across financial time series because labels
  are built from *future* returns (so a training sample's "label window"
  can overlap the test set) and because market data is serially correlated
  (so samples immediately after a test block are still contaminated by it).
  CPCV addresses both: it *purges* any training sample whose label window
  overlaps a test block, and *embargoes* a short window immediately after
  each test block.
- **Probability of Backtest Overfitting (PBO)**: Bailey, Borwein, Lopez de
  Prado & Zhu, "The Probability of Backtest Overfitting", *Journal of
  Computational Finance* 20(4) (2017). Deflated Sharpe asks "is *this one*
  strategy's Sharpe ratio significant once we account for how many trials
  were tried". PBO asks the complementary question: "how likely is it that
  the *selection process itself* -- picking whichever trial looked best
  in-sample -- chose a strategy that will underperform out-of-sample". The
  paper's estimator is combinatorially symmetric cross-validation (CSCV):
  split many independent in-sample/out-of-sample performance paths in half
  every possible way, and see how often the in-sample winner ends up below
  the out-of-sample median.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CrossValidationConfig:
    """Configuration shared by `CombinatorialPurgedCV` and `ProbabilityOfBacktestOverfitting`.

    Parameters
    ----------
    n_groups : int
        Number of contiguous, time-ordered groups the sample index is split
        into before combinatorial test-set selection.
    n_test_groups : int
        Number of groups held out as the test set in each CPCV split. The
        total number of splits (and therefore PBO "paths") is
        ``C(n_groups, n_test_groups)``.
    embargo_fraction : float
        Fraction of the full sample length embargoed (dropped from training)
        immediately after each test group, guarding against leakage from
        serial correlation.
    min_paths_for_pbo : int
        Minimum number of usable performance paths required before
        `ProbabilityOfBacktestOverfitting.compute` will produce an estimate.
        Below this, the IS/OOS split combinatorics are too thin to trust.
    """

    n_groups: int = 10
    n_test_groups: int = 2
    embargo_fraction: float = 0.01
    min_paths_for_pbo: int = 8


# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPCVSplit:
    """One combinatorial purged train/test partition.

    Attributes
    ----------
    train_indices : np.ndarray
        Sorted sample indices safe to train on for this split (test group
        indices, purged overlaps, and embargoed indices all excluded).
    test_indices : np.ndarray
        Sorted sample indices making up this split's test set.
    path_id : int
        Index of this split among all `C(n_groups, n_test_groups)`
        combinations; used by `ProbabilityOfBacktestOverfitting` to line up
        one performance column per path.
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    path_id: int


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (Lopez de Prado, 2018, ch. 7/12).

    The sample index ``[0, n_samples)`` is cut into `n_groups` contiguous,
    time-ordered groups. Every way of choosing `n_test_groups` of those
    groups as the test set forms one split (so there are
    ``C(n_groups, n_test_groups)`` splits in total). For each split:

    - any training sample whose label window ``[i, i + label_horizon)``
      intersects the test set is *purged* (dropped from training), because
      its label was built from information the test set is supposed to
      hold out;
    - any sample within `embargo_fraction * n_samples` positions
      immediately after a test group is also dropped, guarding against
      leakage through serial correlation in the underlying return series.

    Each split's `path_id` is later used directly as one "path" of the
    Probability of Backtest Overfitting estimator -- treating every
    combinatorial test-set choice as an independent out-of-sample sample of
    strategy performance, per Bailey, Borwein, Lopez de Prado & Zhu (2017).

    Parameters
    ----------
    config : CrossValidationConfig, optional
    """

    def __init__(self, config: CrossValidationConfig | None = None) -> None:
        self._config = config or CrossValidationConfig()

    @property
    def config(self) -> CrossValidationConfig:
        return self._config

    @property
    def n_paths(self) -> int:
        """Number of CPCV splits (== PBO paths) this configuration produces."""
        cfg = self._config
        return math.comb(cfg.n_groups, cfg.n_test_groups)

    def split(self, n_samples: int, label_horizon: int = 1) -> list[CPCVSplit]:
        """Build every combinatorial purged train/test split.

        Parameters
        ----------
        n_samples : int
            Length of the time-ordered sample index to split.
        label_horizon : int
            Number of forward samples each label depends on. A training
            sample at position ``i`` is purged if any of
            ``[i, i + label_horizon)`` falls inside the test set.

        Returns
        -------
        list[CPCVSplit]
            One entry per combination of `n_test_groups` out of `n_groups`,
            ordered by `itertools.combinations`.
        """
        cfg = self._config
        if cfg.n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not (1 <= cfg.n_test_groups < cfg.n_groups):
            raise ValueError("n_test_groups must be in [1, n_groups - 1]")
        if n_samples < cfg.n_groups:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_groups ({cfg.n_groups})"
            )
        if label_horizon < 0:
            raise ValueError("label_horizon must be >= 0")

        group_indices = np.array_split(np.arange(n_samples), cfg.n_groups)
        embargo = max(int(round(cfg.embargo_fraction * n_samples)), 0)

        splits: list[CPCVSplit] = []
        for path_id, test_group_ids in enumerate(
            itertools.combinations(range(cfg.n_groups), cfg.n_test_groups)
        ):
            is_test = np.zeros(n_samples, dtype=bool)
            is_embargo = np.zeros(n_samples, dtype=bool)
            for group_id in test_group_ids:
                idx = group_indices[group_id]
                if idx.size == 0:
                    continue
                is_test[idx] = True
                embargo_start = int(idx[-1]) + 1
                embargo_end = min(embargo_start + embargo, n_samples)
                if embargo_end > embargo_start:
                    is_embargo[embargo_start:embargo_end] = True

            # Purge: drop any train sample whose label window [i, i+h) overlaps
            # the test set. Vectorized via a prefix sum over the test mask.
            cum_test = np.concatenate([[0], np.cumsum(is_test)])
            window_end = np.minimum(np.arange(n_samples) + label_horizon, n_samples)
            overlaps_test = (cum_test[window_end] - cum_test[np.arange(n_samples)]) > 0

            train_mask = ~is_test & ~is_embargo & ~overlaps_test
            splits.append(
                CPCVSplit(
                    train_indices=np.nonzero(train_mask)[0].astype(np.int64),
                    test_indices=np.nonzero(is_test)[0].astype(np.int64),
                    path_id=path_id,
                )
            )

        return splits


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting
# ---------------------------------------------------------------------------


@dataclass
class PBOResult:
    """Result of the combinatorially-symmetric CV Probability of Backtest Overfitting estimate.

    Attributes
    ----------
    pbo : float
        Fraction of IS/OOS path-splits in which the in-sample-best trial's
        out-of-sample performance fell at or below the out-of-sample median
        (logit <= 0). This is the paper's PBO estimate.
    n_combinations : int
        Number of IS/OOS path-splits actually evaluated (full
        ``C(n_paths, n_paths // 2)`` enumeration, or a deterministic sample
        of it -- see `ProbabilityOfBacktestOverfitting`).
    logit_values : list[float]
        The relative-rank logit ``ln(omega / (1 - omega))`` computed for
        each evaluated split, where `omega` is the in-sample winner's
        out-of-sample rank percentile. Negative/zero values are the
        overfitting occurrences that make up `pbo`.
    passes : bool
        ``pbo < 0.5``.
    """

    pbo: float
    n_combinations: int
    logit_values: list[float]
    passes: bool


class ProbabilityOfBacktestOverfitting:
    """Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu, 2017).

    Implements the paper's combinatorially symmetric cross-validation
    (CSCV) estimator. Given an ``(n_trials, n_paths)`` matrix of performance
    values -- one column per independent in-sample/out-of-sample path (e.g.
    one `CombinatorialPurgedCV` split each) -- the algorithm:

    1. Enumerates ways of splitting the `n_paths` columns into two halves,
       one playing the role of the in-sample (IS) set and the other the
       out-of-sample (OOS) set, for that combination.
    2. For each split, finds the trial with the best *IS* average
       performance (the one a naive researcher would have picked).
    3. Locates that trial's rank among all trials' *OOS* average
       performance, converted to a relative rank ``omega in (0, 1)``.
    4. Converts the relative rank to a logit,
       ``lambda = ln(omega / (1 - omega))``; ``lambda <= 0`` means the
       IS-best trial did not even reach the OOS median, i.e. this split
       counts as an overfitting occurrence.

    PBO is the fraction of splits flagged as overfitting occurrences.

    Why the 0.5 pass threshold
    ---------------------------
    Under the null hypothesis that in-sample performance carries no genuine
    information about out-of-sample performance, the IS-best trial's OOS
    rank is, by symmetry, uniformly distributed across all trials -- so it
    falls at or below the OOS median in exactly half of all splits. PBO is
    therefore centered at 0.5 for a selection process with *no* skill:
    ``PBO < 0.5`` means the "pick the in-sample winner" procedure beats
    that chance baseline at finding an above-median OOS performer, while
    ``PBO >= 0.5`` means it does no better (or actively worse) than
    guessing -- the paper's own criterion for calling a backtest overfit
    (Bailey et al., 2017, sec. 4 and 8: "if PBO > 0.5, it is more likely
    than not that the model selected as 'best' in-sample will underperform
    ... out-of-sample").

    Parameters
    ----------
    config : CrossValidationConfig, optional
        Only `min_paths_for_pbo` is consulted here.
    max_combinations : int
        Full enumeration is ``C(n_paths, n_paths // 2)``, which explodes
        quickly (e.g. 45 CPCV paths -> ~3.9e12 splits). Above this many
        combinations, a deterministic random sample of exactly
        `max_combinations` distinct half-splits is used instead.
    seed : int
        Seed for the deterministic sampling above.
    """

    def __init__(
        self,
        config: CrossValidationConfig | None = None,
        max_combinations: int = 2000,
        seed: int = 42,
    ) -> None:
        self._config = config or CrossValidationConfig()
        self._max_combinations = max(int(max_combinations), 1)
        self._seed = seed

    def compute(self, is_oos_matrix: np.ndarray) -> PBOResult:
        """Estimate the Probability of Backtest Overfitting.

        Parameters
        ----------
        is_oos_matrix : np.ndarray, shape (n_trials, n_paths)
            Performance value (e.g. mean IC, mean long-short return, or
            Sharpe) of each of `n_trials` candidate strategies on each of
            `n_paths` independent performance paths.

        Returns
        -------
        PBOResult
        """
        matrix = np.asarray(is_oos_matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("is_oos_matrix must be 2-D (n_trials, n_paths)")
        n_trials, n_paths = matrix.shape
        if n_trials < 2:
            raise ValueError("PBO requires at least 2 trials")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("is_oos_matrix must not contain NaN/inf values")
        if n_paths < self._config.min_paths_for_pbo:
            raise ValueError(
                f"is_oos_matrix has {n_paths} paths; PBO requires at least "
                f"{self._config.min_paths_for_pbo} (config.min_paths_for_pbo)"
            )

        half = n_paths // 2
        if half < 1:
            raise ValueError("n_paths too small to split into IS/OOS halves")

        total_combinations = math.comb(n_paths, half)
        is_combos = self._select_combinations(n_paths, half, total_combinations)

        logit_values: list[float] = []
        overfit_count = 0
        is_mask = np.zeros(n_paths, dtype=bool)
        for is_paths in is_combos:
            is_mask[:] = False
            is_mask[list(is_paths)] = True

            is_perf = matrix[:, is_mask].mean(axis=1)
            oos_perf = matrix[:, ~is_mask].mean(axis=1)

            best_trial = int(np.argmax(is_perf))
            # Relative rank of the IS-best trial's OOS performance, kept
            # strictly inside (0, 1) so the logit is always finite.
            ranks = rankdata(oos_perf, method="average")
            omega = ranks[best_trial] / (n_trials + 1.0)
            omega = min(max(omega, 1e-9), 1.0 - 1e-9)
            logit = math.log(omega / (1.0 - omega))
            logit_values.append(logit)
            if logit <= 0.0:
                overfit_count += 1

        n_combinations = len(logit_values)
        pbo = overfit_count / n_combinations if n_combinations else 1.0

        return PBOResult(
            pbo=pbo,
            n_combinations=n_combinations,
            logit_values=logit_values,
            passes=pbo < 0.5,
        )

    def _select_combinations(
        self, n_paths: int, half: int, total_combinations: int
    ) -> list[tuple[int, ...]]:
        """Return the IS-path-index combinations to evaluate."""
        if total_combinations <= self._max_combinations:
            return list(itertools.combinations(range(n_paths), half))

        rng = np.random.default_rng(self._seed)
        seen: set[tuple[int, ...]] = set()
        combos: list[tuple[int, ...]] = []
        while len(combos) < self._max_combinations:
            choice = tuple(sorted(rng.choice(n_paths, size=half, replace=False).tolist()))
            if choice in seen:
                continue
            seen.add(choice)
            combos.append(choice)
        return combos
