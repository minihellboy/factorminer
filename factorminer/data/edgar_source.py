"""SEC EDGAR XBRL fundamentals connector (point-in-time as-filed).

Fetches ``us-gaap`` company facts from the official SEC JSON API
(``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json``) and maps sparse
filing facts onto an existing ``(asset_id, datetime)`` bar grid via
**forward-fill from each fact's filed date**.

Point-in-time discipline
------------------------
The single most important correctness property of this module is that a fact
becomes visible only on/after its *filed* timestamp (the as-reported filing
date), never the covered period end. Using period-end would leak future
information into the panel. Tests assert NaN before the filed date and the
correct value after.

Security / fair-access
----------------------
SEC fair-access policy (see https://www.sec.gov/os/accessing-edgar-data and
https://www.sec.gov/developer):

* Identify the caller with a descriptive ``User-Agent`` that includes a contact
  email (SEC rejects bare library defaults and may throttle anonymous scrapers).
* Stay at or under **10 requests per second** across the SEC estate.
* Prefer HTTPS endpoints (``data.sec.gov``).

This module:

* uses HTTPS only
* sets ``User-Agent: FactorMiner Research Bot 1.0 (contact@factorminer.local)``
  by default (override via :class:`EdgarConfig.user_agent`)
* enforces a minimum inter-request interval (default 0.12s ≈ 8 req/s, under the
  10 req/s ceiling) plus an explicit socket timeout
* caps response body size before JSON parse
* validates ``Content-Type`` and fail-closes on malformed payloads
* caches by sanitized CIK under a local directory (no path traversal)
* never ``eval``/``exec``s fetched content
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorminer.core.types import register_features

logger = logging.getLogger(__name__)

# SEC fair-access: descriptive UA with contact. Override in config for production.
DEFAULT_USER_AGENT = "FactorMiner Research Bot 1.0 (contact@factorminer.local)"
DEFAULT_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"
DEFAULT_MAX_RPS = 8.0  # stay under SEC's documented 10 req/s ceiling
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024  # 8 MiB

# DSL leaves produced by this connector.
EDGAR_FEATURE_LEAVES: tuple[str, ...] = (
    "$eps",
    "$revenue",
    "$book_equity",
    "$shares_out",
)

EDGAR_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "$eps": "earnings per share (point-in-time, as-filed from SEC EDGAR XBRL)",
    "$revenue": "total revenue (point-in-time, as-filed from SEC EDGAR XBRL)",
    "$book_equity": (
        "book equity / stockholders' equity (point-in-time, as-filed from SEC EDGAR XBRL)"
    ),
    "$shares_out": "shares outstanding (point-in-time, as-filed from SEC EDGAR XBRL)",
}

# Map internal leaf -> preferred us-gaap concept names (first hit wins).
_CONCEPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "eps": (
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
        "EarningsPerShareBasicAndDiluted",
    ),
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "book_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
    ),
    "shares_out": (
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ),
}

_CIK_RE = re.compile(r"[^0-9]")


@dataclass(frozen=True)
class EdgarConfig:
    """Configuration for the SEC EDGAR XBRL fundamentals connector."""

    user_agent: str = DEFAULT_USER_AGENT
    base_url: str = DEFAULT_BASE_URL
    cache_dir: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_requests_per_second: float = DEFAULT_MAX_RPS
    register_leaves: bool = True
    concepts: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: v for k, v in _CONCEPT_CANDIDATES.items()}
    )

    def validate(self) -> None:
        if not self.user_agent or "bot" in self.user_agent.lower() and "@" not in self.user_agent:
            # Soft check: SEC requires identifying UA with contact; we only warn
            # via logger when '@' is missing.
            pass
        if "@" not in self.user_agent:
            logger.warning(
                "EdgarConfig.user_agent should include a contact email per SEC "
                "fair-access policy; got %r",
                self.user_agent,
            )
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be >= 1024")
        if self.max_requests_per_second <= 0 or self.max_requests_per_second > 10:
            raise ValueError(
                "max_requests_per_second must be in (0, 10]; SEC fair-access "
                "ceiling is 10 requests per second"
            )


@dataclass(frozen=True)
class EdgarFact:
    """One as-filed XBRL fact observation."""

    concept: str
    value: float
    filed: pd.Timestamp
    end: pd.Timestamp | None = None
    form: str = ""
    fy: int | None = None
    fp: str = ""


class _RateLimiter:
    """Simple minimum-interval rate limiter (thread-safe)."""

    def __init__(self, max_rps: float) -> None:
        self._min_interval = 1.0 / float(max_rps)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._min_interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


def sanitize_cik(cik: str | int) -> str:
    """Return a zero-padded 10-digit CIK, rejecting path-like input."""
    raw = str(cik).strip()
    if ".." in raw or "/" in raw or "\\" in raw:
        raise ValueError(f"Refusing CIK with path-like characters: {cik!r}")
    digits = _CIK_RE.sub("", raw)
    if not digits:
        raise ValueError(f"CIK has no digits: {cik!r}")
    if len(digits) > 10:
        raise ValueError(f"CIK too long: {cik!r}")
    return digits.zfill(10)


def register_edgar_features() -> list[str]:
    """Register EDGAR fundamental leaves on the global feature registry."""
    return register_features(
        list(EDGAR_FEATURE_LEAVES),
        descriptions=EDGAR_FEATURE_DESCRIPTIONS,
    )


def _cache_path(cache_dir: Path, cik10: str) -> Path:
    # cik10 is digits-only from sanitize_cik — safe as a single path segment.
    return cache_dir / f"companyfacts_CIK{cik10}.json"


def _parse_companyfacts_payload(
    payload: Mapping[str, Any],
    concepts: Mapping[str, Sequence[str]],
) -> dict[str, list[EdgarFact]]:
    """Extract as-filed facts for the requested concept groups.

    Fail-closed: malformed units/facts are skipped with a warning.
    """
    facts_root = payload.get("facts")
    if not isinstance(facts_root, dict):
        logger.warning("EDGAR payload missing 'facts' object; returning empty")
        return {key: [] for key in concepts}

    gaap = facts_root.get("us-gaap")
    if not isinstance(gaap, dict):
        # Some entities put entity-level facts under `dei` only.
        gaap = {}
    dei = facts_root.get("dei") if isinstance(facts_root.get("dei"), dict) else {}

    out: dict[str, list[EdgarFact]] = {}
    for leaf_key, candidates in concepts.items():
        collected: list[EdgarFact] = []
        for concept in candidates:
            block = gaap.get(concept) if concept in gaap else dei.get(concept)
            if not isinstance(block, dict):
                continue
            units = block.get("units")
            if not isinstance(units, dict):
                continue
            # Prefer USD / shares unit bags; otherwise take the first unit list.
            unit_keys = list(units.keys())
            preferred = [k for k in unit_keys if k.upper() in {"USD", "SHARES", "PURE"}]
            ordered_keys = preferred + [k for k in unit_keys if k not in preferred]
            for ukey in ordered_keys:
                rows = units.get(ukey)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    fact = _row_to_fact(concept, row)
                    if fact is not None:
                        collected.append(fact)
            if collected:
                break  # first matching concept wins
        # Sort by filed ascending for asof/ffill
        collected.sort(key=lambda f: f.filed)
        out[leaf_key] = collected
    return out


def _row_to_fact(concept: str, row: Any) -> EdgarFact | None:
    if not isinstance(row, dict):
        return None
    filed_raw = row.get("filed")
    if not filed_raw:
        return None
    try:
        filed = pd.Timestamp(filed_raw)
    except (ValueError, TypeError):
        logger.warning("Skipping fact with unparseable filed=%r", filed_raw)
        return None
    if pd.isna(filed):
        return None

    val = row.get("val")
    try:
        value = float(val)
    except (TypeError, ValueError):
        logger.warning("Skipping fact with non-numeric val=%r for %s", val, concept)
        return None
    if not np.isfinite(value):
        return None

    end: pd.Timestamp | None
    try:
        end = pd.Timestamp(row["end"]) if row.get("end") else None
    except (ValueError, TypeError):
        end = None

    fy_raw = row.get("fy")
    try:
        fy = int(fy_raw) if fy_raw is not None else None
    except (TypeError, ValueError):
        fy = None

    return EdgarFact(
        concept=concept,
        value=value,
        filed=filed.normalize() if hasattr(filed, "normalize") else pd.Timestamp(filed.date()),
        end=end,
        form=str(row.get("form") or ""),
        fy=fy,
        fp=str(row.get("fp") or ""),
    )


def facts_to_asof_frame(
    facts_by_leaf: Mapping[str, Sequence[EdgarFact]],
    bar_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a date-indexed frame of point-in-time fundamental values.

    For each bar date ``t``, the value is the latest fact with ``filed <= t``
    (forward-filled from the filed date). Dates before the first filing are NaN.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(list(bar_dates))).normalize().unique().sort_values()
    data: dict[str, np.ndarray] = {}
    for leaf_key, facts in facts_by_leaf.items():
        col = np.full(len(dates), np.nan, dtype=np.float64)
        if not facts:
            data[leaf_key] = col
            continue
        filed_ts = np.array([pd.Timestamp(f.filed).normalize() for f in facts])
        values = np.array([f.value for f in facts], dtype=np.float64)
        # For each bar date, rightmost fact with filed <= date
        idxs = np.searchsorted(filed_ts, dates.values, side="right") - 1
        valid = idxs >= 0
        col[valid] = values[idxs[valid]]
        data[leaf_key] = col
    return pd.DataFrame(data, index=dates)


class EdgarClient:
    """HTTPS client for SEC companyfacts with rate limiting and disk cache."""

    def __init__(self, config: EdgarConfig | None = None) -> None:
        self.config = config or EdgarConfig()
        self.config.validate()
        self._limiter = _RateLimiter(self.config.max_requests_per_second)
        self._cache_dir = (
            Path(self.config.cache_dir).expanduser()
            if self.config.cache_dir
            else None
        )
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_companyfacts(
        self,
        cik: str | int,
        *,
        offline_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the companyfacts JSON object for *cik*.

        Parameters
        ----------
        cik :
            Central Index Key (int or digit string).
        offline_payload :
            When provided, skip the network entirely and use this mapping
            (tests / fixtures). Still runs through the same parser path.
        """
        cik10 = sanitize_cik(cik)
        if offline_payload is not None:
            if not isinstance(offline_payload, Mapping):
                raise TypeError("offline_payload must be a mapping")
            return dict(offline_payload)

        if self._cache_dir is not None:
            cached = _cache_path(self._cache_dir, cik10)
            if cached.is_file():
                try:
                    text = cached.read_text(encoding="utf-8")
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        return payload
                    logger.warning("Ignoring non-object cache payload at %s", cached)
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    logger.warning("Cache read failed for %s: %s", cached, exc)

        url = f"{self.config.base_url.rstrip('/')}/CIK{cik10}.json"
        payload = self._http_get_json(url)
        if self._cache_dir is not None and payload:
            try:
                _cache_path(self._cache_dir, cik10).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Failed to write EDGAR cache: %s", exc)
        return payload

    def _http_get_json(self, url: str) -> dict[str, Any]:
        if not url.lower().startswith("https://"):
            raise ValueError(f"Refusing non-HTTPS EDGAR URL: {url!r}")

        self._limiter.wait()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.config.user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as resp:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "json" not in content_type and "javascript" not in content_type:
                    # SEC occasionally omits charset; still require json-ish type.
                    logger.warning(
                        "Unexpected Content-Type %r from %s; refusing parse",
                        content_type,
                        url,
                    )
                    return {}
                raw = resp.read(self.config.max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            logger.warning("EDGAR HTTP error %s for %s: %s", exc.code, url, exc.reason)
            return {}
        except urllib.error.URLError as exc:
            logger.warning("EDGAR network error for %s: %s", url, exc.reason)
            return {}
        except TimeoutError:
            logger.warning("EDGAR timeout for %s", url)
            return {}

        if len(raw) > self.config.max_response_bytes:
            logger.warning(
                "EDGAR response exceeded max_response_bytes=%d; discarding",
                self.config.max_response_bytes,
            )
            return {}

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            logger.warning("EDGAR JSON parse failed for %s: %s", url, exc)
            return {}

        if not isinstance(payload, dict):
            logger.warning("EDGAR payload is not a JSON object; discarding")
            return {}
        return payload


def load_edgar_fundamentals(
    cik_map: Mapping[str, str | int],
    bar_index: pd.DatetimeIndex | Sequence[pd.Timestamp],
    *,
    config: EdgarConfig | None = None,
    offline_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Load point-in-time fundamentals for many assets.

    Parameters
    ----------
    cik_map :
        ``asset_id -> CIK`` mapping.
    bar_index :
        Union of bar timestamps (will be normalized to dates).
    config :
        Connector configuration.
    offline_payloads :
        Optional ``asset_id -> companyfacts dict`` for offline/mock mode.

    Returns
    -------
    pd.DataFrame
        Long panel with columns ``datetime``, ``asset_id``, and fundamental
        columns ``eps``, ``revenue``, ``book_equity``, ``shares_out``.
    """
    cfg = config or EdgarConfig()
    cfg.validate()
    if cfg.register_leaves:
        register_edgar_features()

    client = EdgarClient(cfg)
    dates = pd.DatetimeIndex(pd.to_datetime(list(bar_index))).normalize().unique().sort_values()
    frames: list[pd.DataFrame] = []
    offline_payloads = offline_payloads or {}

    for asset_id, cik in cik_map.items():
        payload = None
        if asset_id in offline_payloads:
            payload = offline_payloads[asset_id]
        elif str(cik) in offline_payloads:
            payload = offline_payloads[str(cik)]
        try:
            raw = client.fetch_companyfacts(cik, offline_payload=payload)
        except ValueError as exc:
            logger.warning("Skipping asset %s: %s", asset_id, exc)
            continue
        if not raw:
            logger.warning("No EDGAR payload for asset %s (CIK=%s)", asset_id, cik)
            continue
        facts = _parse_companyfacts_payload(raw, cfg.concepts)
        asof = facts_to_asof_frame(facts, dates)
        if asof.empty:
            continue
        piece = asof.reset_index().rename(columns={"index": "datetime"})
        if "datetime" not in piece.columns:
            # pandas may name the index column differently
            piece = asof.copy()
            piece.index.name = "datetime"
            piece = piece.reset_index()
        piece["asset_id"] = str(asset_id)
        frames.append(piece)

    if not frames:
        cols = ["datetime", "asset_id", "eps", "revenue", "book_equity", "shares_out"]
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["asset_id"] = out["asset_id"].astype(str)
    return out.sort_values(["datetime", "asset_id"]).reset_index(drop=True)


def attach_edgar_to_panel(
    panel: pd.DataFrame,
    cik_map: Mapping[str, str | int],
    *,
    config: EdgarConfig | None = None,
    offline_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Left-join point-in-time EDGAR fundamentals onto an OHLCV panel.

    The panel must contain ``datetime`` and ``asset_id``. Fundamental columns
    are attached as bare names (``eps``, ``revenue``, ...) matching tensor
    builder column conventions; register DSL leaves via
    :func:`register_edgar_features`.
    """
    if panel.empty:
        return panel.copy()
    work = panel.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    work["asset_id"] = work["asset_id"].astype(str)

    fund = load_edgar_fundamentals(
        cik_map,
        work["datetime"],
        config=config,
        offline_payloads=offline_payloads,
    )
    if fund.empty:
        for col in ("eps", "revenue", "book_equity", "shares_out"):
            if col not in work.columns:
                work[col] = np.nan
        return work

    # Align on calendar date for asof join within each asset.
    work = work.copy()
    work["_date"] = work["datetime"].dt.normalize()
    fund = fund.copy()
    fund["_date"] = pd.to_datetime(fund["datetime"]).dt.normalize()
    value_cols = [c for c in ("eps", "revenue", "book_equity", "shares_out") if c in fund.columns]
    merged = work.merge(
        fund[["asset_id", "_date", *value_cols]],
        on=["asset_id", "_date"],
        how="left",
        suffixes=("", "_edgar"),
    )
    merged = merged.drop(columns=["_date"])
    return merged.sort_values(["datetime", "asset_id"]).reset_index(drop=True)


def build_mock_companyfacts(
    *,
    cik: str = "0000320193",
    entity_name: str = "Mock Co",
    facts: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal companyfacts-shaped payload for offline tests.

    Each entry in *facts* is a dict with keys:
    ``concept``, ``val``, ``filed``, and optional ``end``, ``form``, ``fy``,
    ``fp``, ``unit`` (default ``USD`` or ``shares`` for share concepts).
    """
    us_gaap: dict[str, Any] = {}
    for row in facts or ():
        concept = str(row["concept"])
        unit = str(row.get("unit") or ("shares" if "Share" in concept else "USD"))
        entry = {
            "val": row["val"],
            "filed": row["filed"],
            "end": row.get("end", row["filed"]),
            "form": row.get("form", "10-K"),
            "fy": row.get("fy"),
            "fp": row.get("fp", "FY"),
        }
        block = us_gaap.setdefault(concept, {"label": concept, "units": {}})
        block["units"].setdefault(unit, []).append(entry)

    return {
        "cik": int(sanitize_cik(cik)),
        "entityName": entity_name,
        "facts": {"us-gaap": us_gaap},
    }


__all__ = [
    "DEFAULT_USER_AGENT",
    "EDGAR_FEATURE_DESCRIPTIONS",
    "EDGAR_FEATURE_LEAVES",
    "EdgarClient",
    "EdgarConfig",
    "EdgarFact",
    "attach_edgar_to_panel",
    "build_mock_companyfacts",
    "facts_to_asof_frame",
    "load_edgar_fundamentals",
    "register_edgar_features",
    "sanitize_cik",
]
