"""Factor crowding diagnostics for FactorMiner.

Three complementary research-risk signals (landscape §10 item 4):

1. **Consensus-factor novelty screen** — correlate a candidate's long-short
   return series against a free public consensus panel (Ken French Data
   Library CSV/ZIP layout). High ``max|ρ|`` means the formula rediscovers a
   well-known style rather than a novel edge.
2. **Lou–Polk CoMetric (CoMOM)** — within long and short legs separately,
   average pairwise *residual* return correlation over a rolling window.
   ``CoMOM = 0.5 * (CoMOM_long + CoMOM_short)``. This is **within-leg**
   residual comovement of one factor's own holdings, distinct from the
   library's cross-factor dependence metrics in
   ``architecture/dependence.py``. Two residualization modes
   (:func:`compute_cometric`'s ``residual_mode``):

   - ``"cross_sectional"`` (default) — per-bar cross-sectional demeaning
     (subtract the equal-weighted cross-sectional mean each bar,
     equivalent to a single-factor, equal-beta market model). Fast, needs
     no external data, but leaves heterogeneous-beta assets' own market
     exposure in the "residual" (cross-sectional demeaning only removes
     each bar's *average* level, not each asset's *own* loading).
   - ``"factor_regression"`` (opt-in) — the paper's actual methodology:
     regress each asset's excess return on Mkt-RF/SMB/HML within each
     window (beta assumed stable only within-window, per Lewellen & Nagel
     2006) and use the OLS residual, which is orthogonal to the factors
     by construction. Requires a non-empty ``ConsensusFactorPanel`` (the
     same Ken French data already used for the overlap screen above); one
     omission from the paper remains even in this mode: no industry
     adjustment (Fama-French 30 industries), since FactorMiner's universe
     spans asset classes -- crypto, futures, multi-asset OHLCV -- where a
     US-equity SIC-code industry taxonomy doesn't generalize. Falls back
     to ``"cross_sectional"`` (logged, and reported on the result) when
     the panel is unavailable -- never fabricates a factor fit.
3. **Hyperbolic decay taxonomy** — composed from the already-built
   :func:`factorminer.evaluation.decay.fit_hyperbolic_decay` /
   :func:`factorminer.evaluation.decay.classify_crowding_decay_risk`
   (Lee 2025). Do not re-fit here; import and attach.

All three are **research risk annotations**, not mining objectives or
trade-timing signals. Source papers show crowding means are largely priced
in; use the labels to flag rediscovery / capacity-crowding risk only.

Security for remote Ken French fetches
--------------------------------------
HTTPS only, socket timeout, response size cap, content validation, and
fail-closed behaviour on malformed/truncated payloads. A failed load never
crashes the caller and never invents a reassuring overlap score — the
panel is empty and overlap is reported as unavailable.
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from factorminer.evaluation.decay import (
    CrowdingDecayTaxonomy,
    classify_crowding_decay_risk,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Bundled offline fixture (synthetic Ken French FF3 daily layout).
DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "ken_french_ff3_fixture.csv"
)

#: Public Ken French Fama/French 3 Factors daily ZIP (HTTPS).
DEFAULT_KEN_FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024  # 4 MiB
DEFAULT_USER_AGENT = "FactorMiner Research Bot 1.0 (contact@factorminer.local)"

#: Default rolling window for Lou–Polk CoMetric (~63 daily / ~52 weekly bars).
DEFAULT_COMETRIC_WINDOW = 63

#: Minimum assets in a leg to compute pairwise residual correlations.
MIN_LEG_ASSETS = 3

#: Minimum overlapping finite observations for a pairwise residual correlation.
MIN_PAIR_OBS = 10

#: Labels for consensus-overlap magnitude.
OVERLAP_HIGH = 0.70
OVERLAP_MODERATE = 0.40


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CrowdingConfig:
    """Configuration for crowding diagnostics."""

    cometric_window: int = DEFAULT_COMETRIC_WINDOW
    leg_fraction: float = 0.2
    consensus_url: str = DEFAULT_KEN_FRENCH_URL
    cache_dir: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    user_agent: str = DEFAULT_USER_AGENT
    # Percent → decimal scale for Ken French (values are typically x100).
    scale_percent_to_decimal: bool = True
    # "cross_sectional" (default, no external data) or "factor_regression"
    # (Lou & Polk's actual methodology: FF3-regression residuals, reusing
    # the same consensus panel already fetched for overlap scoring below).
    cometric_residual_mode: str = "cross_sectional"

    def __post_init__(self) -> None:
        if self.cometric_window < 5:
            raise ValueError("cometric_window must be >= 5")
        if not 0.05 <= self.leg_fraction <= 0.5:
            raise ValueError("leg_fraction must be in [0.05, 0.5]")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be >= 1024")
        if self.cometric_residual_mode not in ("cross_sectional", "factor_regression"):
            raise ValueError(
                "cometric_residual_mode must be 'cross_sectional' or 'factor_regression'"
            )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusOverlapResult:
    """Correlation of a candidate L/S series against a consensus panel."""

    max_abs_rho: float
    best_factor: str
    label: str
    correlations: dict[str, float]
    available: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_rho": self.max_abs_rho,
            "best_factor": self.best_factor,
            "label": self.label,
            "correlations": dict(self.correlations),
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CoMetricResult:
    """Lou–Polk within-leg residual comovement score."""

    comom: float
    comom_long: float
    comom_short: float
    n_windows: int
    window: int
    available: bool
    detail: str = ""
    residual_mode: str = "cross_sectional"

    def to_dict(self) -> dict[str, Any]:
        return {
            "comom": self.comom,
            "comom_long": self.comom_long,
            "comom_short": self.comom_short,
            "n_windows": self.n_windows,
            "window": self.window,
            "available": self.available,
            "detail": self.detail,
            "residual_mode": self.residual_mode,
        }


@dataclass(frozen=True)
class CrowdingScore:
    """Composite research-risk crowding annotation for one factor.

    Composes consensus novelty (a), Lou–Polk CoMetric (b), and the salvaged
    hyperbolic decay taxonomy from ``evaluation/decay.py`` (c).
    """

    factor_id: str
    consensus: ConsensusOverlapResult
    cometric: CoMetricResult
    decay_taxonomy: CrowdingDecayTaxonomy | None
    composite_label: str
    novelty_modulation: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "consensus": self.consensus.to_dict(),
            "cometric": self.cometric.to_dict(),
            "decay_taxonomy": (
                self.decay_taxonomy.to_dict() if self.decay_taxonomy is not None else None
            ),
            "composite_label": self.composite_label,
            "novelty_modulation": self.novelty_modulation,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Consensus panel loader (Ken French CSV/ZIP)
# ---------------------------------------------------------------------------


def _looks_like_date_token(token: str) -> bool:
    token = token.strip().strip(",")
    if not token:
        return False
    # YYYYMM or YYYYMMDD (Ken French daily/monthly)
    if token.isdigit() and len(token) in {6, 8}:
        return True
    return False


def _parse_ken_french_text(text: str) -> dict[str, np.ndarray]:
    """Parse Ken French factor CSV text into name → return arrays.

    Fail-closed: returns ``{}`` on any structural problem (no header, no
    numeric rows, inconsistent widths). Never raises for content issues.
    """
    if not text or not text.strip():
        logger.warning("Ken French payload empty; fail-closed empty panel")
        return {}

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header_idx: int | None = None
    header_cols: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("This file"):
            continue
        # Header typically looks like: ,Mkt-RF,SMB,HML,RF  or  Date,Mkt-RF,...
        parts = [p.strip() for p in stripped.split(",")]
        non_empty = [p for p in parts if p]
        # A header has factor-ish names and is not a pure date row.
        if len(non_empty) >= 2 and not _looks_like_date_token(non_empty[0]):
            # Prefer rows that mention a known factor token.
            joined = ",".join(non_empty).upper()
            if any(tok in joined for tok in ("MKT", "SMB", "HML", "RF", "MOM", "RMW", "CMA")):
                header_idx = i
                # Drop leading empty date column name if present.
                header_cols = [p for p in parts if p] if parts[0] else [p for p in parts[1:] if p]
                if parts[0] and not _looks_like_date_token(parts[0]):
                    # First cell might be "Date" — drop non-factor label.
                    if parts[0].lower() in {"", "date", "dates"}:
                        header_cols = [p for p in parts[1:] if p]
                    else:
                        header_cols = [p for p in parts if p]
                break
        # Also accept a bare header with empty first cell + factor names.
        if (
            len(parts) >= 3
            and parts[0] == ""
            and any("MKT" in p.upper() or "SMB" in p.upper() for p in parts[1:])
        ):
            header_idx = i
            header_cols = [p for p in parts[1:] if p]
            break

    if header_idx is None or not header_cols:
        logger.warning("Ken French CSV: no factor header found; fail-closed")
        return {}

    n_cols = len(header_cols)
    columns: dict[str, list[float]] = {name: [] for name in header_cols}

    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            # Blank line often separates annual section; stop at first break
            # after we have data (Ken French daily files append annual block).
            if any(columns[name] for name in header_cols):
                break
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if not parts:
            continue
        # Skip copyright / notes tails.
        if not _looks_like_date_token(parts[0]):
            if any(columns[name] for name in header_cols):
                break
            continue
        values = parts[1:] if len(parts) > n_cols else parts[1:]
        # Align to header width.
        if len(parts) >= n_cols + 1:
            values = parts[1 : n_cols + 1]
        elif len(parts) == n_cols and not _looks_like_date_token(parts[0]):
            values = parts
        else:
            values = parts[1:]
        if len(values) != n_cols:
            # Truncated / ragged row — fail this row, keep going.
            continue
        parsed: list[float] = []
        ok = True
        for raw in values:
            if raw == "" or raw.upper() == "NA":
                ok = False
                break
            try:
                parsed.append(float(raw))
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        for name, val in zip(header_cols, parsed, strict=True):
            columns[name].append(val)

    series: dict[str, np.ndarray] = {}
    for name, vals in columns.items():
        if len(vals) < 2:
            continue
        # Skip pure risk-free column from overlap scoring utility later;
        # still keep it available for residualization consumers.
        series[name] = np.asarray(vals, dtype=np.float64)

    if not series:
        logger.warning("Ken French CSV: no numeric factor rows; fail-closed")
        return {}
    return series


def _extract_text_from_bytes(raw: bytes) -> str | None:
    """Decode CSV bytes or first CSV member of a ZIP. Fail-closed → None."""
    if not raw:
        return None
    # ZIP magic
    if raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = [
                    n
                    for n in zf.namelist()
                    if n.lower().endswith(".csv") and not n.endswith("/")
                ]
                if not names:
                    # Fall back to any non-directory member.
                    names = [n for n in zf.namelist() if not n.endswith("/")]
                if not names:
                    logger.warning("Ken French ZIP has no members; fail-closed")
                    return None
                payload = zf.read(names[0])
        except zipfile.BadZipFile:
            logger.warning("Ken French ZIP malformed; fail-closed")
            return None
        except Exception:  # pragma: no cover - defensive
            logger.warning("Ken French ZIP read failed; fail-closed", exc_info=True)
            return None
    else:
        payload = raw

    # Reject obvious truncation of UTF-8 mid-stream by requiring decodable text.
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return payload.decode("latin-1")
        except Exception:
            logger.warning("Ken French payload not decodable; fail-closed")
            return None


class ConsensusFactorPanel:
    """Loader/cache for free public consensus factor-return series.

    Prefer the bundled fixture for tests/CI (``load_fixture`` /
    ``from_fixture``). Remote fetches use HTTPS + timeout + size cap and
    fail closed on malformed data.
    """

    def __init__(
        self,
        series: Mapping[str, np.ndarray] | None = None,
        *,
        config: CrowdingConfig | None = None,
        source: str = "empty",
    ) -> None:
        self.config = config or CrowdingConfig()
        self._series: dict[str, np.ndarray] = {
            str(k): np.asarray(v, dtype=np.float64).reshape(-1)
            for k, v in (series or {}).items()
        }
        self.source = source

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_fixture(
        cls,
        path: str | Path | None = None,
        *,
        config: CrowdingConfig | None = None,
    ) -> ConsensusFactorPanel:
        """Load the bundled (or caller-supplied) offline Ken French fixture."""
        cfg = config or CrowdingConfig()
        fixture = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
        try:
            text = fixture.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Consensus fixture unreadable (%s); fail-closed", exc)
            return cls(series={}, config=cfg, source=f"fixture-error:{fixture}")
        series = _parse_ken_french_text(text)
        if cfg.scale_percent_to_decimal and series:
            series = {k: v / 100.0 for k, v in series.items()}
        return cls(series=series, config=cfg, source=f"fixture:{fixture}")

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        config: CrowdingConfig | None = None,
        source: str = "bytes",
    ) -> ConsensusFactorPanel:
        """Parse a CSV/ZIP byte payload. Fail-closed → empty panel."""
        cfg = config or CrowdingConfig()
        text = _extract_text_from_bytes(raw)
        if text is None:
            return cls(series={}, config=cfg, source=f"{source}:undecodable")
        series = _parse_ken_french_text(text)
        if cfg.scale_percent_to_decimal and series:
            series = {k: v / 100.0 for k, v in series.items()}
        return cls(series=series, config=cfg, source=source)

    @classmethod
    def fetch(
        cls,
        url: str | None = None,
        *,
        config: CrowdingConfig | None = None,
        cache_path: str | Path | None = None,
    ) -> ConsensusFactorPanel:
        """Fetch-and-cache a Ken French CSV/ZIP with security hygiene.

        Fail-closed on network errors, non-HTTPS URLs, oversized responses,
        and malformed content. Never raises to callers for those cases.
        """
        cfg = config or CrowdingConfig()
        target = url or cfg.consensus_url

        # HTTPS only — reject anything else closed.
        if not target.lower().startswith("https://"):
            logger.warning(
                "Consensus fetch refused non-HTTPS URL %r; fail-closed", target
            )
            return cls(series={}, config=cfg, source="refused-non-https")

        # Cache hit.
        cache_file: Path | None = None
        if cache_path is not None:
            cache_file = Path(cache_path)
        elif cfg.cache_dir:
            cache_dir = Path(cfg.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Stable filename from URL tail.
            tail = target.rstrip("/").rsplit("/", 1)[-1] or "consensus.bin"
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in tail)
            cache_file = cache_dir / safe

        if cache_file is not None and cache_file.is_file():
            try:
                raw = cache_file.read_bytes()
                if 0 < len(raw) <= cfg.max_response_bytes:
                    panel = cls.from_bytes(raw, config=cfg, source=f"cache:{cache_file}")
                    if panel.factor_names:
                        return panel
                    logger.warning(
                        "Cached consensus payload malformed; re-fetching %s", target
                    )
            except OSError as exc:
                logger.warning("Consensus cache read failed (%s); re-fetching", exc)

        request = Request(
            target,
            headers={"User-Agent": cfg.user_agent, "Accept": "application/zip,text/csv,*/*"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=cfg.timeout_s) as resp:  # noqa: S310 — HTTPS-only guarded above
                raw = resp.read(cfg.max_response_bytes + 1)
        except HTTPError as exc:
            logger.warning(
                "Consensus fetch HTTP %s for %s: %s; fail-closed",
                exc.code,
                target,
                exc.reason,
            )
            return cls(series={}, config=cfg, source=f"http-error:{exc.code}")
        except URLError as exc:
            logger.warning(
                "Consensus fetch URL error for %s: %s; fail-closed", target, exc.reason
            )
            return cls(series={}, config=cfg, source="url-error")
        except TimeoutError:
            logger.warning("Consensus fetch timed out for %s; fail-closed", target)
            return cls(series={}, config=cfg, source="timeout")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Consensus fetch failed for %s: %s; fail-closed", target, exc
            )
            return cls(series={}, config=cfg, source="fetch-error")

        if len(raw) > cfg.max_response_bytes:
            logger.warning(
                "Consensus response exceeded max_response_bytes=%d; fail-closed",
                cfg.max_response_bytes,
            )
            return cls(series={}, config=cfg, source="oversized")

        panel = cls.from_bytes(raw, config=cfg, source=f"fetch:{target}")
        if panel.factor_names and cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(raw)
            except OSError as exc:
                logger.warning("Consensus cache write failed: %s", exc)
        return panel

    # -- accessors ----------------------------------------------------------

    @property
    def factor_names(self) -> list[str]:
        return list(self._series.keys())

    @property
    def empty(self) -> bool:
        return not self._series

    def get(self, name: str) -> np.ndarray | None:
        arr = self._series.get(name)
        return None if arr is None else arr.copy()

    def as_matrix(
        self, names: Sequence[str] | None = None
    ) -> tuple[list[str], np.ndarray]:
        """Return ``(names, T×K matrix)`` aligned on the shortest series length."""
        use = list(names) if names is not None else self.factor_names
        use = [n for n in use if n in self._series]
        if not use:
            return [], np.zeros((0, 0), dtype=np.float64)
        length = min(self._series[n].shape[0] for n in use)
        if length < 2:
            return [], np.zeros((0, 0), dtype=np.float64)
        mat = np.column_stack([self._series[n][-length:] for n in use])
        return use, mat


# ---------------------------------------------------------------------------
# Consensus overlap
# ---------------------------------------------------------------------------


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation on finite pairwise overlap; 0.0 if degenerate."""
    n = min(a.shape[0], b.shape[0])
    if n < 3:
        return 0.0
    x = np.asarray(a[-n:], dtype=np.float64)
    y = np.asarray(b[-n:], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-18:
        return 0.0
    return float(np.sum(x * y) / denom)


def _overlap_label(max_abs_rho: float) -> str:
    if max_abs_rho >= OVERLAP_HIGH:
        return "high_consensus_overlap"
    if max_abs_rho >= OVERLAP_MODERATE:
        return "moderate_consensus_overlap"
    return "low_consensus_overlap"


def consensus_overlap_score(
    candidate_ls_returns: np.ndarray,
    panel: ConsensusFactorPanel,
    *,
    skip_names: Sequence[str] = ("RF", "rf", "RiskFree"),
) -> ConsensusOverlapResult:
    """Correlate a candidate long-short return series against the panel.

    Returns an unavailable result (not a fabricated low score) when the
    panel is empty or the candidate series is unusable — fail closed.
    """
    series = np.asarray(candidate_ls_returns, dtype=np.float64).reshape(-1)
    finite = series[np.isfinite(series)]
    if finite.size < 3:
        return ConsensusOverlapResult(
            max_abs_rho=0.0,
            best_factor="",
            label="unavailable",
            correlations={},
            available=False,
            detail="candidate long-short series too short or non-finite",
        )
    if panel.empty:
        return ConsensusOverlapResult(
            max_abs_rho=0.0,
            best_factor="",
            label="unavailable",
            correlations={},
            available=False,
            detail=f"consensus panel empty (source={panel.source})",
        )

    skip = {s.lower() for s in skip_names}
    correlations: dict[str, float] = {}
    for name in panel.factor_names:
        if name.lower() in skip:
            continue
        ref = panel.get(name)
        if ref is None:
            continue
        rho = _safe_pearson(series, ref)
        correlations[name] = rho

    if not correlations:
        return ConsensusOverlapResult(
            max_abs_rho=0.0,
            best_factor="",
            label="unavailable",
            correlations={},
            available=False,
            detail="no comparable consensus factors after filters",
        )

    best_factor = max(correlations, key=lambda k: abs(correlations[k]))
    max_abs = float(abs(correlations[best_factor]))
    return ConsensusOverlapResult(
        max_abs_rho=max_abs,
        best_factor=best_factor,
        label=_overlap_label(max_abs),
        correlations=correlations,
        available=True,
        detail=f"max|ρ|={max_abs:.3f} vs {best_factor}",
    )


# ---------------------------------------------------------------------------
# Lou–Polk CoMetric (within-leg residual comovement)
# ---------------------------------------------------------------------------


def _cross_sectional_residuals(returns: np.ndarray) -> np.ndarray:
    """Demean returns cross-sectionally each bar → residual panel (M, T)."""
    r = np.asarray(returns, dtype=np.float64)
    out = np.full_like(r, np.nan, dtype=np.float64)
    m, t = r.shape
    for j in range(t):
        col = r[:, j]
        mask = np.isfinite(col)
        if int(mask.sum()) < 2:
            continue
        mu = float(np.mean(col[mask]))
        out[mask, j] = col[mask] - mu
    return out


#: Fama-French 3-factor names as produced by :func:`_parse_ken_french_text`.
#: Regression uses whichever of these (plus ``RF``) are present in the
#: supplied panel -- Mkt-RF alone still gives a real market-model residual
#: even if SMB/HML are unavailable.
_FF3_FACTOR_NAMES: tuple[str, ...] = ("Mkt-RF", "SMB", "HML")


def _factor_regression_residuals_window(
    returns_window: np.ndarray,
    factor_window: np.ndarray,
) -> np.ndarray:
    """OLS residuals of ``returns_window`` (M, W) against ``factor_window`` (W, K).

    One regression per asset over the WHOLE window (beta assumed stable only
    within this window, following Lewellen & Nagel 2006 / Lou & Polk 2022 --
    not a single full-sample beta). Adds an intercept column internally.
    Assets/columns with insufficient finite overlap get an all-NaN residual
    row rather than a degenerate fit.
    """
    r = np.asarray(returns_window, dtype=np.float64)
    f = np.asarray(factor_window, dtype=np.float64)
    m, w = r.shape
    out = np.full((m, w), np.nan, dtype=np.float64)
    if w < f.shape[0]:
        f = f[-w:, :]
    elif f.shape[0] < w:
        return out
    design = np.column_stack([np.ones(w, dtype=np.float64), f])
    k = design.shape[1]
    factor_finite = np.all(np.isfinite(design), axis=1)
    if int(factor_finite.sum()) < k + 3:
        # Not enough factor-observation overlap to identify the regression
        # at all this window -- fail closed rather than fit on noise.
        return out
    for i in range(m):
        row = r[i]
        obs_mask = factor_finite & np.isfinite(row)
        if int(obs_mask.sum()) < k + 3:
            continue
        x = design[obs_mask]
        y = row[obs_mask]
        try:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = x @ coef
        out[i, obs_mask] = y - fitted
    return out


def _align_factor_matrix_to_window(
    consensus_panel: ConsensusFactorPanel,
    window_len: int,
) -> np.ndarray | None:
    """Return the trailing ``window_len`` rows of the FF3 factor matrix, or
    ``None`` if the panel doesn't have enough usable factor history.

    Excess-return construction: caller's ``returns`` should already be raw
    asset returns; ``RF`` (if present) is subtracted from the dependent
    variable, not included as a regressor -- Mkt-RF/SMB/HML are the
    regressors, matching the Fama-French convention.
    """
    names, mat = consensus_panel.as_matrix(
        [n for n in _FF3_FACTOR_NAMES if n in consensus_panel.factor_names]
    )
    if not names or mat.shape[0] < window_len:
        return None
    return mat[-window_len:, :]



def _mean_pairwise_corr(block: np.ndarray) -> float:
    """Mean off-diagonal correlation of rows in ``block`` (n_assets, n_time)."""
    n = block.shape[0]
    if n < 2:
        return float("nan")
    # Pairwise only on columns where both finite.
    corrs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a = block[i]
            b = block[j]
            mask = np.isfinite(a) & np.isfinite(b)
            if int(mask.sum()) < MIN_PAIR_OBS:
                continue
            rho = _safe_pearson(a[mask], b[mask])
            if np.isfinite(rho):
                corrs.append(rho)
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def _leg_indices(signal_t: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (long_idx, short_idx) for one cross-section."""
    valid = np.flatnonzero(np.isfinite(signal_t))
    n = int(valid.size)
    if n < max(2 * MIN_LEG_ASSETS, 5):
        empty = np.asarray([], dtype=int)
        return empty, empty
    k = max(MIN_LEG_ASSETS, int(np.floor(n * fraction)))
    k = min(k, n // 2)
    if k < MIN_LEG_ASSETS:
        empty = np.asarray([], dtype=int)
        return empty, empty
    order = valid[np.argsort(signal_t[valid])]
    short_idx = order[:k]
    long_idx = order[-k:]
    return long_idx, short_idx


def compute_cometric(
    signals: np.ndarray,
    returns: np.ndarray,
    *,
    window: int = DEFAULT_COMETRIC_WINDOW,
    leg_fraction: float = 0.2,
    residual_mode: str = "cross_sectional",
    consensus_panel: ConsensusFactorPanel | None = None,
) -> CoMetricResult:
    """Lou–Polk CoMOM: mean within-leg residual pairwise correlation.

    Parameters
    ----------
    signals, returns:
        Shape ``(M, T)`` — assets × time (FactorMiner convention).
    window:
        Rolling lookback for pairwise residual correlations. Also the
        per-window regression window in ``factor_regression`` mode
        (following Lewellen & Nagel 2006 / Lou & Polk 2022: betas are
        assumed stable only *within* the regression window, not
        full-sample).
    leg_fraction:
        Fraction of the cross-section in each of the long/short legs.
    residual_mode:
        ``"cross_sectional"`` (default): fast, no external data, subtracts
        the equal-weighted cross-sectional mean each bar (a single-factor,
        equal-beta market-model proxy). ``"factor_regression"``: the paper's
        actual methodology — regress each asset's excess return on
        Mkt-RF/SMB/HML within each window and use the OLS residual, using
        ``consensus_panel`` (Ken French data). Requires ``consensus_panel``
        to be non-empty and cover at least ``window`` observations; falls
        back to ``cross_sectional`` (with a logged warning and
        ``residual_mode="cross_sectional"`` on the returned result) when
        that data isn't available — never silently fabricates factor
        loadings from missing data.
    consensus_panel:
        Required for ``residual_mode="factor_regression"``. See
        :class:`ConsensusFactorPanel`.
    """
    sig = np.asarray(signals, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    if sig.ndim != 2 or ret.ndim != 2 or sig.shape != ret.shape:
        return CoMetricResult(
            comom=0.0,
            comom_long=0.0,
            comom_short=0.0,
            n_windows=0,
            window=window,
            available=False,
            detail=f"shape mismatch signals={getattr(sig, 'shape', None)} "
            f"returns={getattr(ret, 'shape', None)}",
        )

    m, t = sig.shape
    if t < window or m < 2 * MIN_LEG_ASSETS:
        return CoMetricResult(
            comom=0.0,
            comom_long=0.0,
            comom_short=0.0,
            n_windows=0,
            window=window,
            available=False,
            detail="insufficient time or assets for CoMetric window",
        )

    factor_matrix: np.ndarray | None = None
    rf_series: np.ndarray | None = None
    effective_mode = "cross_sectional"
    if residual_mode == "factor_regression":
        if consensus_panel is not None and not consensus_panel.empty:
            factor_matrix = _align_factor_matrix_to_window(consensus_panel, t)
            if "RF" in consensus_panel.factor_names:
                rf_full = consensus_panel.get("RF")
                if rf_full is not None and rf_full.shape[0] >= t:
                    rf_series = rf_full[-t:]
        if factor_matrix is not None:
            effective_mode = "factor_regression"
        else:
            logger.warning(
                "CoMetric factor_regression mode requested but consensus_panel "
                "is unavailable/too short (< %d obs); falling back to "
                "cross_sectional residuals",
                t,
            )
    elif residual_mode != "cross_sectional":
        raise ValueError(f"Unsupported residual_mode: {residual_mode!r}")

    excess_ret = ret if rf_series is None else ret - rf_series[np.newaxis, :]
    resid: np.ndarray | None = None
    if effective_mode == "cross_sectional":
        resid = _cross_sectional_residuals(ret)

    long_scores: list[float] = []
    short_scores: list[float] = []

    # Evaluate at the end of each full window (stride = max(1, window // 4)
    # to keep this cheap on long panels).
    stride = max(1, window // 4)
    for end in range(window, t + 1, stride):
        start = end - window
        if effective_mode == "factor_regression":
            window_resid = _factor_regression_residuals_window(
                excess_ret[:, start:end], factor_matrix[start:end, :]  # type: ignore[index]
            )
        else:
            window_resid = resid[:, start:end]  # type: ignore[index]
        # Leg membership from the last bar of the window (point-in-time holdings).
        long_idx, short_idx = _leg_indices(sig[:, end - 1], leg_fraction)
        if long_idx.size >= MIN_LEG_ASSETS:
            val = _mean_pairwise_corr(window_resid[long_idx, :])
            if np.isfinite(val):
                long_scores.append(val)
        if short_idx.size >= MIN_LEG_ASSETS:
            val = _mean_pairwise_corr(window_resid[short_idx, :])
            if np.isfinite(val):
                short_scores.append(val)

    if not long_scores and not short_scores:
        return CoMetricResult(
            comom=0.0,
            comom_long=0.0,
            comom_short=0.0,
            n_windows=0,
            window=window,
            available=False,
            detail="no valid within-leg residual correlations",
            residual_mode=effective_mode,
        )

    comom_long = float(np.mean(long_scores)) if long_scores else 0.0
    comom_short = float(np.mean(short_scores)) if short_scores else 0.0
    if long_scores and short_scores:
        comom = 0.5 * (comom_long + comom_short)
    elif long_scores:
        comom = comom_long
    else:
        comom = comom_short
    return CoMetricResult(
        comom=float(comom),
        comom_long=float(comom_long),
        comom_short=float(comom_short),
        n_windows=max(len(long_scores), len(short_scores)),
        window=window,
        available=True,
        detail=f"CoMOM={comom:.3f} (L={comom_long:.3f}, S={comom_short:.3f})",
        residual_mode=effective_mode,
    )


# ---------------------------------------------------------------------------
# Long-short return series from signals (for consensus overlap)
# ---------------------------------------------------------------------------


def long_short_returns(
    signals: np.ndarray,
    returns: np.ndarray,
    *,
    leg_fraction: float = 0.2,
) -> np.ndarray:
    """Per-bar Q-top minus Q-bottom mean return; shape ``(T,)``.

    ``signals`` / ``returns`` are ``(M, T)``.
    """
    sig = np.asarray(signals, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    if sig.shape != ret.shape or sig.ndim != 2:
        return np.zeros(0, dtype=np.float64)
    _m, t = sig.shape
    out = np.full(t, np.nan, dtype=np.float64)
    for j in range(t):
        long_idx, short_idx = _leg_indices(sig[:, j], leg_fraction)
        if long_idx.size == 0 or short_idx.size == 0:
            continue
        r_long = ret[long_idx, j]
        r_short = ret[short_idx, j]
        if not (np.isfinite(r_long).any() and np.isfinite(r_short).any()):
            continue
        out[j] = float(np.nanmean(r_long) - np.nanmean(r_short))
    return out


# ---------------------------------------------------------------------------
# Composite crowding score
# ---------------------------------------------------------------------------


def _composite_label(
    consensus: ConsensusOverlapResult,
    cometric: CoMetricResult,
    taxonomy: CrowdingDecayTaxonomy | None,
) -> tuple[str, float, str]:
    """Return (label, novelty_modulation in [0,1], rationale).

    ``novelty_modulation`` is a soft multiplier for geometry novelty_score:
    1.0 = no penalty (novel), lower = more crowded / consensus-like.
    """
    parts: list[str] = []
    # Start from full novelty; subtract penalties.
    modulation = 1.0

    if consensus.available:
        parts.append(f"consensus={consensus.label}(max|ρ|={consensus.max_abs_rho:.2f})")
        # High overlap → strong novelty penalty.
        modulation *= float(max(0.0, 1.0 - consensus.max_abs_rho))
    else:
        parts.append("consensus=unavailable")

    if cometric.available:
        parts.append(f"CoMOM={cometric.comom:.2f}")
        # CoMOM in ~[0,1]; higher within-leg residual corr → more crowded.
        c = float(np.clip(cometric.comom, 0.0, 1.0))
        modulation *= float(max(0.0, 1.0 - 0.5 * c))
    else:
        parts.append("CoMOM=unavailable")

    decay_label = None
    if taxonomy is not None:
        decay_label = taxonomy.risk_label
        parts.append(f"decay={decay_label}")
        if taxonomy.risk_label == "high_crowding_decay_risk":
            modulation *= 0.7
        elif taxonomy.risk_label == "elevated_crowding_decay_risk":
            modulation *= 0.85

    modulation = float(np.clip(modulation, 0.0, 1.0))

    # Composite ordinal label (research risk only).
    high_consensus = consensus.available and consensus.max_abs_rho >= OVERLAP_HIGH
    high_comom = cometric.available and cometric.comom >= 0.35
    high_decay = decay_label in {
        "high_crowding_decay_risk",
        "elevated_crowding_decay_risk",
    }

    if high_consensus and (high_comom or high_decay):
        label = "high_crowding_risk"
    elif high_consensus or high_comom:
        label = "elevated_crowding_risk"
    elif high_decay:
        label = "decay_crowding_watch"
    elif not consensus.available and not cometric.available:
        label = "insufficient_crowding_data"
    else:
        label = "low_crowding_risk"

    rationale = (
        f"{label}: " + "; ".join(parts) + ". Research risk label only — not a trade timer."
    )
    return label, modulation, rationale


def score_factor_crowding(
    *,
    signals: np.ndarray,
    returns: np.ndarray,
    panel: ConsensusFactorPanel | None = None,
    ic_by_iteration: Sequence[float] | None = None,
    formula: str = "",
    factor_id: str = "",
    config: CrowdingConfig | None = None,
    ls_returns: np.ndarray | None = None,
) -> CrowdingScore:
    """Compose consensus overlap + CoMetric + hyperbolic taxonomy."""
    cfg = config or CrowdingConfig()
    resolved_panel = panel if panel is not None else ConsensusFactorPanel.from_fixture(
        config=cfg
    )

    if ls_returns is None:
        ls = long_short_returns(signals, returns, leg_fraction=cfg.leg_fraction)
    else:
        ls = np.asarray(ls_returns, dtype=np.float64).reshape(-1)

    consensus = consensus_overlap_score(ls, resolved_panel)
    cometric = compute_cometric(
        signals,
        returns,
        window=cfg.cometric_window,
        leg_fraction=cfg.leg_fraction,
        residual_mode=cfg.cometric_residual_mode,
        consensus_panel=resolved_panel if cfg.cometric_residual_mode == "factor_regression" else None,
    )

    taxonomy: CrowdingDecayTaxonomy | None = None
    if ic_by_iteration is not None:
        taxonomy = classify_crowding_decay_risk(
            ic_by_iteration, formula=formula
        )

    label, modulation, rationale = _composite_label(consensus, cometric, taxonomy)
    return CrowdingScore(
        factor_id=factor_id,
        consensus=consensus,
        cometric=cometric,
        decay_taxonomy=taxonomy,
        composite_label=label,
        novelty_modulation=modulation,
        rationale=rationale,
    )


def build_crowding_report(
    items: Sequence[Mapping[str, Any]],
    *,
    panel: ConsensusFactorPanel | None = None,
    config: CrowdingConfig | None = None,
) -> list[dict[str, Any]]:
    """Build crowding rows for a sequence of factor dicts.

    Each item may supply: ``factor_id``, ``formula``, ``signals``, ``returns``,
    ``ls_returns``, ``ic_by_iteration``.
    """
    cfg = config or CrowdingConfig()
    resolved_panel = panel if panel is not None else ConsensusFactorPanel.from_fixture(
        config=cfg
    )
    rows: list[dict[str, Any]] = []
    for item in items:
        signals = item.get("signals")
        returns = item.get("returns")
        if signals is None or returns is None:
            continue
        score = score_factor_crowding(
            signals=np.asarray(signals),
            returns=np.asarray(returns),
            panel=resolved_panel,
            ic_by_iteration=item.get("ic_by_iteration"),
            formula=str(item.get("formula", "")),
            factor_id=str(item.get("factor_id", item.get("name", ""))),
            config=cfg,
            ls_returns=item.get("ls_returns"),
        )
        rows.append(score.to_dict())
    rows.sort(key=lambda r: r.get("factor_id", ""))
    return rows
