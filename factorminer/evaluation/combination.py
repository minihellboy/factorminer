"""Factor combination strategies for building composite signals.

Implements Equal-Weight, IC-Weighted, and Orthogonal combination methods
for merging multiple alpha factors into a single composite signal, following
the methodology described in the FactorMiner paper.
"""

from __future__ import annotations

import numpy as np


class FactorCombiner:
    """Combine multiple factor signals into a single composite signal.

    Each factor signal is a 2-D array of shape (T, N) where T is the number
    of time steps and N is the number of assets.  Factor IDs are arbitrary
    integers used as dictionary keys.
    """

    # ------------------------------------------------------------------
    # Public combination methods
    # ------------------------------------------------------------------

    def equal_weight(self, factor_signals: dict[int, np.ndarray]) -> np.ndarray:
        """Equal-Weight (EW): simple average of cross-sectionally standardized factors.

        Paper results: IC Mean=0.1451, ICIR=1.2053, IC Win Rate=85.0%.

        Parameters
        ----------
        factor_signals : dict[int, ndarray]
            Mapping from factor ID to (T, N) signal array.

        Returns
        -------
        ndarray of shape (T, N)
            Composite signal (average of z-scored factors).
        """
        if not factor_signals:
            raise ValueError("factor_signals must not be empty")

        standardized = [
            self._cross_sectional_standardize(sig)
            for sig in factor_signals.values()
        ]
        stacked = np.stack(standardized, axis=0)  # (K, T, N)
        # Average over factors, ignoring NaNs
        return np.nanmean(stacked, axis=0)

    def ic_weighted(
        self,
        factor_signals: dict[int, np.ndarray],
        ic_values: dict[int, float],
    ) -> np.ndarray:
        """IC-Weighted (ICW): weight factors proportionally by their historical IC.

        Paper results: IC Mean=0.1496, ICIR=1.2430, Cumulative Return=26.67
        (12.4% over EW).

        Parameters
        ----------
        factor_signals : dict[int, ndarray]
            Mapping from factor ID to (T, N) signal array.
        ic_values : dict[int, float]
            Mapping from factor ID to its historical Information Coefficient.
            Factors with non-positive IC are excluded.

        Returns
        -------
        ndarray of shape (T, N)
            Composite signal.
        """
        if not factor_signals:
            raise ValueError("factor_signals must not be empty")

        ids = list(factor_signals.keys())
        weights: dict[int, float] = {}
        for fid in ids:
            ic = ic_values.get(fid, 0.0)
            if np.isfinite(ic) and ic > 0.0:
                weights[fid] = ic

        if not weights:
            # Fall back to equal weight if all ICs are non-positive
            return self.equal_weight(factor_signals)

        total_weight = sum(weights.values())
        ref_shape = next(iter(factor_signals.values())).shape
        composite = np.zeros(ref_shape, dtype=np.float64)

        for fid, w in weights.items():
            z = self._cross_sectional_standardize(factor_signals[fid])
            composite += (w / total_weight) * np.where(np.isnan(z), 0.0, z)

        return composite

    def orthogonal(self, factor_signals: dict[int, np.ndarray]) -> np.ndarray:
        """Orthogonal: Gram-Schmidt orthogonalization before averaging.

        Removes cross-factor collinearity by projecting each factor onto the
        subspace orthogonal to all previously processed factors, then averages
        the orthogonalized residuals.

        Paper results: IC Mean=0.1400, ICIR=1.1933.

        Parameters
        ----------
        factor_signals : dict[int, ndarray]
            Mapping from factor ID to (T, N) signal array.

        Returns
        -------
        ndarray of shape (T, N)
            Composite signal (average of orthogonalized z-scored factors).
        """
        if not factor_signals:
            raise ValueError("factor_signals must not be empty")

        standardized = [
            self._cross_sectional_standardize(sig)
            for sig in factor_signals.values()
        ]

        orthogonalized = self._gram_schmidt(standardized)
        stacked = np.stack(orthogonalized, axis=0)  # (K, T, N)
        return np.nanmean(stacked, axis=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cross_sectional_standardize(self, signals: np.ndarray) -> np.ndarray:
        """Standardize signals cross-sectionally (across assets) at each time step.

        z_score = (x - mean) / std  per cross-section (row).

        Parameters
        ----------
        signals : ndarray of shape (T, N)

        Returns
        -------
        ndarray of shape (T, N)
            Cross-sectionally standardized values.  Rows where std == 0
            are set to 0.
        """
        signals = np.asarray(signals, dtype=np.float64)
        cs_mean = np.nanmean(signals, axis=1, keepdims=True)
        cs_std = np.nanstd(signals, axis=1, keepdims=True)
        # Avoid division by zero
        cs_std = np.where(cs_std == 0.0, 1.0, cs_std)
        return (signals - cs_mean) / cs_std

    @staticmethod
    def _gram_schmidt(factors: list[np.ndarray]) -> list[np.ndarray]:
        """Modified Gram-Schmidt orthogonalization on flattened factor vectors.

        Each factor is a (T, N) array.  We flatten to 1-D, orthogonalize,
        then reshape back.  NaN values are treated as zero during projection
        and restored afterward.

        Parameters
        ----------
        factors : list of ndarray, each (T, N)

        Returns
        -------
        list of ndarray, each (T, N) -- orthogonalized factors.
        """
        if len(factors) <= 1:
            return list(factors)

        shape = factors[0].shape
        # Replace NaN with 0 for linear algebra, track NaN mask
        nan_masks = [np.isnan(f) for f in factors]
        vecs = [np.where(m, 0.0, f).ravel() for f, m in zip(factors, nan_masks)]

        ortho: list[np.ndarray] = []
        for i, v in enumerate(vecs):
            u = v.copy()
            for prev in ortho:
                denom = np.dot(prev, prev)
                if denom > 1e-12:
                    u -= (np.dot(u, prev) / denom) * prev
            ortho.append(u)

        result = []
        for u, mask in zip(ortho, nan_masks):
            arr = u.reshape(shape)
            arr[mask] = np.nan
            result.append(arr)
        return result
