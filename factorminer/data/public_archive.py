"""Pinned public-archive ingestion for reproducible evidence releases.

This module deliberately separates an editable source specification from a
fully resolved lock file. A benchmark may only consume the lock: every remote
archive URL and SHA-256 must already be frozen before any model is evaluated.
"""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

PUBLIC_DATASET_LOCK_VERSION = 2
PUBLIC_DATASET_MANIFEST_VERSION = 2
DEFAULT_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_EXTRACTED_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class PublicArchive:
    asset_id: str
    period: str
    url: str
    sha256: str
    parser: str = "binance_spot_kline_zip"
    checksum_source: str = "provider_sidecar"


@dataclass(frozen=True)
class PublicDatasetLock:
    schema_version: int
    dataset_id: str
    provider: str
    asset_class: str
    frequency: str
    periods_per_year: float
    liquidity_evidence: bool
    interval: str
    start: str
    end: str
    assets: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    license: dict[str, Any]
    universe: dict[str, Any]
    availability_lag: str
    point_in_time_limitations: tuple[str, ...]
    transformations: tuple[str, ...]
    target: dict[str, Any]
    archives: tuple[PublicArchive, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["transformations"]:
            payload.pop("transformations")
        for archive in payload["archives"]:
            if archive["checksum_source"] == "provider_sidecar":
                archive.pop("checksum_source")
        return payload


@dataclass(frozen=True)
class PublicDatasetVerification:
    passed: bool
    dataset_id: str
    rows: int
    assets: int
    mismatches: tuple[str, ...] = ()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hex_digest(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")


def _validate_remote_url(url: str, allowed_hosts: tuple[str, ...]) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"public archive URL must use https: {url}")
    if parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError(f"public archive URL contains forbidden authority components: {url}")
    if parsed.hostname.lower() not in {host.lower() for host in allowed_hosts}:
        raise ValueError(f"public archive host is not allow-listed: {parsed.hostname}")


def load_public_dataset_lock(path: str | Path) -> PublicDatasetLock:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != PUBLIC_DATASET_LOCK_VERSION:
        raise ValueError(
            f"unsupported public dataset lock version: {payload.get('schema_version')}"
        )
    archives = tuple(PublicArchive(**item) for item in payload.get("archives", []))
    lock = PublicDatasetLock(
        schema_version=int(payload["schema_version"]),
        dataset_id=str(payload["dataset_id"]),
        provider=str(payload["provider"]),
        asset_class=str(payload["asset_class"]),
        frequency=str(payload["frequency"]),
        periods_per_year=float(payload["periods_per_year"]),
        liquidity_evidence=bool(payload["liquidity_evidence"]),
        interval=str(payload["interval"]),
        start=str(payload["start"]),
        end=str(payload["end"]),
        assets=tuple(str(item) for item in payload["assets"]),
        allowed_hosts=tuple(str(item) for item in payload["allowed_hosts"]),
        license=dict(payload["license"]),
        universe=dict(payload["universe"]),
        availability_lag=str(payload["availability_lag"]),
        point_in_time_limitations=tuple(str(item) for item in payload["point_in_time_limitations"]),
        transformations=tuple(str(item) for item in payload.get("transformations", [])),
        target=dict(payload["target"]),
        archives=archives,
    )
    if lock.asset_class not in {"crypto", "equity", "futures", "fx", "multi_asset", "other"}:
        raise ValueError(f"unsupported public dataset asset_class: {lock.asset_class}")
    if not math.isfinite(lock.periods_per_year) or lock.periods_per_year <= 0:
        raise ValueError("public dataset periods_per_year must be finite and positive")
    if not lock.dataset_id or not lock.assets or not lock.archives:
        raise ValueError("public dataset lock requires a dataset_id, assets, and archives")
    if lock.target.get("price_pair") not in {
        "close_to_close",
        "close_to_open",
        "open_to_close",
        "open_to_open",
    }:
        raise ValueError("public dataset target has an unsupported price_pair")
    if not str(lock.target.get("name", "")).strip():
        raise ValueError("public dataset target requires a name")
    if tuple(sorted(set(lock.assets))) != lock.assets:
        raise ValueError("public dataset assets must be unique and sorted")
    expected_pairs = {(archive.asset_id, archive.period) for archive in lock.archives}
    if len(expected_pairs) != len(lock.archives):
        raise ValueError("public dataset lock contains duplicate asset/period archives")
    for archive in lock.archives:
        if archive.asset_id not in lock.assets:
            raise ValueError(f"archive references an undeclared asset: {archive.asset_id}")
        _validate_remote_url(archive.url, lock.allowed_hosts)
        _validate_hex_digest(archive.sha256, field=f"archive {archive.url}")
        if archive.parser not in {"binance_spot_kline_zip", "ecb_exchange_rate_csv"}:
            raise ValueError(f"unsupported public archive parser: {archive.parser}")
        if archive.checksum_source not in {"provider_sidecar", "content_sha256"}:
            raise ValueError(f"unsupported checksum source: {archive.checksum_source}")
    redistribution = lock.license.get("redistribution_status")
    if redistribution not in {"allowed", "not_asserted"}:
        raise ValueError("license.redistribution_status must be allowed or not_asserted")
    if not lock.license.get("source_url") or not lock.license.get("terms_url"):
        raise ValueError("license source_url and terms_url are required")
    license_class = lock.license.get("data_license_class")
    if license_class == "redistributable_with_attribution":
        if redistribution != "allowed" or not lock.license.get("attribution"):
            raise ValueError(
                "redistributable_with_attribution requires allowed redistribution and attribution"
            )
    if license_class == "publicly_retrievable" and redistribution != "not_asserted":
        raise ValueError("publicly_retrievable data must not assert redistribution permission")
    return lock


def _month_range(start: str, end: str) -> list[str]:
    start_dt = datetime.strptime(start, "%Y-%m").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m").replace(tzinfo=UTC)
    if end_dt < start_dt:
        raise ValueError("end_month must be on or after start_month")
    months: list[str] = []
    cursor = start_dt
    while cursor <= end_dt:
        months.append(cursor.strftime("%Y-%m"))
        year = cursor.year + (1 if cursor.month == 12 else 0)
        month = 1 if cursor.month == 12 else cursor.month + 1
        cursor = cursor.replace(year=year, month=month)
    return months


def _read_url_bytes(
    url: str,
    *,
    allowed_hosts: tuple[str, ...],
    max_bytes: int,
    timeout: float = 30.0,
) -> bytes:
    _validate_remote_url(url, allowed_hosts)
    request = urllib.request.Request(url, headers={"User-Agent": "FactorMiner-evidence/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        _validate_remote_url(response.geturl(), allowed_hosts)
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError(f"remote object exceeds size limit: {url}")
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"remote object exceeds size limit: {url}")
    return content


def lock_public_dataset_spec(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    timeout: float = 30.0,
) -> Path:
    """Resolve an editable YAML source spec into a checksum-pinned JSON lock."""
    spec = yaml.safe_load(Path(spec_path).read_text())
    if not isinstance(spec, dict):
        raise ValueError("public dataset specification must be a mapping")
    assets = tuple(sorted(set(str(item).upper() for item in spec["assets"])))
    allowed_hosts = tuple(str(item) for item in spec["allowed_hosts"])
    periods = (
        [str(item) for item in spec["periods"]]
        if "periods" in spec
        else _month_range(str(spec["start_month"]), str(spec["end_month"]))
    )
    url_template = str(spec["url_template"])
    checksum_suffix = str(spec.get("checksum_suffix", ".CHECKSUM"))
    checksum_mode = str(spec.get("checksum_mode", "provider_sidecar"))
    if checksum_mode not in {"provider_sidecar", "content_sha256"}:
        raise ValueError(f"unsupported checksum_mode: {checksum_mode}")
    parser = str(spec.get("parser", "binance_spot_kline_zip"))
    archives: list[PublicArchive] = []
    for asset in assets:
        for period in periods:
            url = url_template.format(
                asset=asset,
                period=period,
                interval=spec["interval"],
                start=spec["start"],
                end=spec["end"],
            )
            if checksum_mode == "provider_sidecar":
                checksum_url = url + checksum_suffix
                checksum_text = _read_url_bytes(
                    checksum_url,
                    allowed_hosts=allowed_hosts,
                    max_bytes=4096,
                    timeout=timeout,
                ).decode("utf-8")
                digest = checksum_text.strip().split()[0]
                _validate_hex_digest(digest, field=checksum_url)
            else:
                content = _read_url_bytes(
                    url,
                    allowed_hosts=allowed_hosts,
                    max_bytes=int(spec.get("max_source_bytes", DEFAULT_MAX_ARCHIVE_BYTES)),
                    timeout=timeout,
                )
                digest = sha256(content).hexdigest()
            archives.append(
                PublicArchive(
                    asset_id=asset,
                    period=period,
                    url=url,
                    sha256=digest.lower(),
                    parser=parser,
                    checksum_source=checksum_mode,
                )
            )

    lock = PublicDatasetLock(
        schema_version=PUBLIC_DATASET_LOCK_VERSION,
        dataset_id=str(spec["dataset_id"]),
        provider=str(spec["provider"]),
        asset_class=str(spec["asset_class"]),
        frequency=str(spec["frequency"]),
        periods_per_year=float(spec["periods_per_year"]),
        liquidity_evidence=bool(spec["liquidity_evidence"]),
        interval=str(spec["interval"]),
        start=str(spec["start"]),
        end=str(spec["end"]),
        assets=assets,
        allowed_hosts=allowed_hosts,
        license=dict(spec["license"]),
        universe=dict(spec["universe"]),
        availability_lag=str(spec["availability_lag"]),
        point_in_time_limitations=tuple(str(item) for item in spec["point_in_time_limitations"]),
        transformations=tuple(str(item) for item in spec.get("transformations", [])),
        target=dict(spec["target"]),
        archives=tuple(archives),
    )
    output = Path(output_path)
    content = _canonical_json_bytes(lock.to_dict())
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite divergent public dataset lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return output


def _download_archive(
    archive: PublicArchive,
    *,
    cache_dir: Path,
    allowed_hosts: tuple[str, ...],
    offline: bool,
    max_archive_bytes: int,
) -> Path:
    filename = Path(urllib.parse.urlparse(archive.url).path).name
    destination = cache_dir / archive.sha256 / filename
    if destination.is_file() and _file_sha256(destination) == archive.sha256:
        return destination
    if offline:
        raise FileNotFoundError(f"archive is not present in the verified cache: {archive.url}")
    content = _read_url_bytes(
        archive.url,
        allowed_hosts=allowed_hosts,
        max_bytes=max_archive_bytes,
    )
    if sha256(content).hexdigest() != archive.sha256:
        raise ValueError(f"archive checksum mismatch: {archive.url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(destination)
    return destination


def _timestamp_to_iso(raw: str) -> str:
    value = int(raw)
    unit = 1_000_000 if value >= 10**15 else 1_000
    timestamp = datetime.fromtimestamp(value / unit, tz=UTC)
    return timestamp.isoformat().replace("+00:00", "Z")


def _validate_decimal(raw: str, *, field: str) -> str:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal in {field}: {raw}") from exc
    if not value.is_finite():
        raise ValueError(f"non-finite decimal in {field}: {raw}")
    return raw


def _validate_bar(
    *, open_: str, high: str, low: str, close: str, volume: str, amount: str, field: str
) -> None:
    open_value, high_value, low_value, close_value, volume_value, amount_value = map(
        Decimal, (open_, high, low, close, volume, amount)
    )
    if high_value < max(open_value, close_value) or low_value > min(open_value, close_value):
        raise ValueError(f"invalid OHLC bounds in {field}")
    if high_value < low_value:
        raise ValueError(f"high is below low in {field}")
    if volume_value < 0 or amount_value < 0:
        raise ValueError(f"negative volume or amount in {field}")


def _rows_from_binance_archive(
    path: Path, *, asset_id: str, max_extracted_bytes: int
) -> list[tuple[str, ...]]:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
            raise ValueError(f"expected exactly one CSV member in {path}")
        member = members[0]
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"unsafe archive member path: {member.filename}")
        if member.file_size > max_extracted_bytes:
            raise ValueError(f"archive member exceeds extracted size limit: {path}")
        content = archive.read(member)
    rows: list[tuple[str, ...]] = []
    for line_number, row in enumerate(csv.reader(io.StringIO(content.decode("utf-8"))), 1):
        if len(row) < 12:
            raise ValueError(f"invalid Binance kline row at {path}:{line_number}")
        if line_number == 1 and not row[0].isdigit():
            continue
        open_, high, low, close = (
            _validate_decimal(row[index], field=f"{path}:{line_number}") for index in (1, 2, 3, 4)
        )
        volume = _validate_decimal(row[5], field=f"{path}:{line_number}")
        amount = _validate_decimal(row[7], field=f"{path}:{line_number}")
        _validate_bar(
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            amount=amount,
            field=f"{path}:{line_number}",
        )
        rows.append((_timestamp_to_iso(row[0]), asset_id, open_, high, low, close, volume, amount))
    return rows


def _decimal_text(value: Decimal) -> str:
    rendered = format(value.normalize(), "f")
    return "0" if rendered in {"", "-0"} else rendered


def _rows_from_ecb_exchange_rate(
    path: Path, *, asset_id: str, max_extracted_bytes: int
) -> list[tuple[str, ...]]:
    if path.stat().st_size > max_extracted_bytes:
        raise ValueError(f"ECB CSV exceeds extracted size limit: {path}")
    rows: list[tuple[str, ...]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "FREQ",
            "CURRENCY",
            "CURRENCY_DENOM",
            "EXR_TYPE",
            "EXR_SUFFIX",
            "TIME_PERIOD",
            "OBS_VALUE",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"ECB CSV schema is missing required columns: {path}")
        for line_number, row in enumerate(reader, 2):
            if (
                row["FREQ"] != "D"
                or row["CURRENCY"] != asset_id
                or row["CURRENCY_DENOM"] != "EUR"
                or row["EXR_TYPE"] != "SP00"
                or row["EXR_SUFFIX"] != "A"
            ):
                raise ValueError(f"unexpected ECB series identity at {path}:{line_number}")
            raw_rate = _validate_decimal(row["OBS_VALUE"], field=f"{path}:{line_number}:OBS_VALUE")
            rate = Decimal(raw_rate)
            if rate <= 0:
                raise ValueError(f"ECB exchange rate must be positive at {path}:{line_number}")
            close = _decimal_text(Decimal(1) / rate)
            timestamp = f"{row['TIME_PERIOD']}T00:00:00Z"
            rows.append((timestamp, asset_id, close, close, close, close, "1", close))
    if not rows:
        raise ValueError(f"ECB CSV contains no observations: {path}")
    return rows


def prepare_public_dataset(
    lock_path: str | Path,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
    max_workers: int = 8,
) -> Path:
    """Download, verify, and deterministically normalize a locked dataset."""
    lock_file = Path(lock_path)
    lock = load_public_dataset_lock(lock_file)
    root = Path(output_dir)
    root.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        Path(cache_dir)
        if cache_dir is not None
        else root.parent / ".public-data-cache" / lock.dataset_id
    )
    staging = Path(tempfile.mkdtemp(prefix=".public-data-", dir=root.parent))
    rows: list[tuple[str, ...]] = []
    try:

        def load_archive(archive: PublicArchive) -> list[tuple[str, ...]]:
            local_archive = _download_archive(
                archive,
                cache_dir=cache,
                allowed_hosts=lock.allowed_hosts,
                offline=offline,
                max_archive_bytes=max_archive_bytes,
            )
            if archive.parser == "binance_spot_kline_zip":
                return _rows_from_binance_archive(
                    local_archive,
                    asset_id=archive.asset_id,
                    max_extracted_bytes=max_extracted_bytes,
                )
            if archive.parser == "ecb_exchange_rate_csv":
                return _rows_from_ecb_exchange_rate(
                    local_archive,
                    asset_id=archive.asset_id,
                    max_extracted_bytes=max_extracted_bytes,
                )
            raise ValueError(f"unsupported public archive parser: {archive.parser}")

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for archive_rows in executor.map(load_archive, lock.archives):
                rows.extend(archive_rows)
        rows.sort(key=lambda item: (item[0], item[1]))
        if len({(row[0], row[1]) for row in rows}) != len(rows):
            raise ValueError("public archives contain duplicate datetime/asset rows")
        start = datetime.fromisoformat(lock.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(lock.end.replace("Z", "+00:00"))
        rows = [
            row
            for row in rows
            if start <= datetime.fromisoformat(row[0].replace("Z", "+00:00")) <= end
        ]
        observed_assets = tuple(sorted({row[1] for row in rows}))
        if observed_assets != lock.assets:
            raise ValueError(
                f"prepared dataset assets differ from lock: expected {lock.assets}, "
                f"observed {observed_assets}"
            )
        data_path = staging / "market_data.csv"
        with data_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                ["datetime", "asset_id", "open", "high", "low", "close", "volume", "amount"]
            )
            writer.writerows(rows)
        shutil.copyfile(lock_file, staging / "dataset.lock.json")
        manifest = {
            "schema_version": PUBLIC_DATASET_MANIFEST_VERSION,
            "dataset_id": lock.dataset_id,
            "provider": lock.provider,
            "asset_class": lock.asset_class,
            "frequency": lock.frequency,
            "periods_per_year": lock.periods_per_year,
            "liquidity_evidence": lock.liquidity_evidence,
            "interval": lock.interval,
            "start": lock.start,
            "end": lock.end,
            "rows": len(rows),
            "assets": list(lock.assets),
            "data_path": "market_data.csv",
            "data_sha256": _file_sha256(data_path),
            "lock_path": "dataset.lock.json",
            "lock_sha256": _file_sha256(lock_file),
            "license": lock.license,
            "universe": lock.universe,
            "availability_lag": lock.availability_lag,
            "point_in_time_limitations": list(lock.point_in_time_limitations),
            "transformations": list(lock.transformations),
            "target": lock.target,
            "source_archives": len(lock.archives),
        }
        (staging / "dataset_manifest.json").write_bytes(_canonical_json_bytes(manifest))
        if root.exists():
            existing = verify_prepared_public_dataset(root)
            if (
                existing.passed
                and _file_sha256(root / "market_data.csv") == manifest["data_sha256"]
            ):
                shutil.rmtree(staging)
                return root
            raise FileExistsError(f"refusing to overwrite divergent dataset directory: {root}")
        staging.replace(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


def verify_prepared_public_dataset(path: str | Path) -> PublicDatasetVerification:
    root = Path(path).resolve()
    mismatches: list[str] = []
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return PublicDatasetVerification(
            passed=False,
            dataset_id="unknown",
            rows=0,
            assets=0,
            mismatches=("dataset_manifest.json is missing",),
        )
    manifest = json.loads(manifest_path.read_text())
    dataset_id = str(manifest.get("dataset_id", "unknown"))
    if manifest.get("schema_version") != PUBLIC_DATASET_MANIFEST_VERSION:
        mismatches.append("unsupported dataset manifest version")
    try:
        lock_path = _resolve_dataset_member(root, str(manifest.get("lock_path", "")))
        data_path = _resolve_dataset_member(root, str(manifest.get("data_path", "")))
    except ValueError as exc:
        return PublicDatasetVerification(
            passed=False,
            dataset_id=dataset_id,
            rows=0,
            assets=0,
            mismatches=(str(exc),),
        )
    lock: PublicDatasetLock | None = None
    if not lock_path.is_file():
        mismatches.append("locked source manifest is missing")
    elif _file_sha256(lock_path) != manifest.get("lock_sha256"):
        mismatches.append("locked source manifest hash mismatch")
    else:
        try:
            lock = load_public_dataset_lock(lock_path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            mismatches.append(f"locked source manifest is invalid: {exc}")
        else:
            if lock.dataset_id != dataset_id:
                mismatches.append("dataset ID differs between lock and manifest")
            locked_manifest_fields = {
                "provider": lock.provider,
                "asset_class": lock.asset_class,
                "frequency": lock.frequency,
                "periods_per_year": lock.periods_per_year,
                "liquidity_evidence": lock.liquidity_evidence,
                "interval": lock.interval,
                "start": lock.start,
                "end": lock.end,
                "assets": list(lock.assets),
                "license": lock.license,
                "universe": lock.universe,
                "availability_lag": lock.availability_lag,
                "point_in_time_limitations": list(lock.point_in_time_limitations),
                "transformations": list(lock.transformations),
                "target": lock.target,
                "source_archives": len(lock.archives),
            }
            for field, expected in locked_manifest_fields.items():
                if manifest.get(field) != expected:
                    mismatches.append(f"{field} differs between lock and manifest")
    rows = 0
    assets: set[str] = set()
    observations: set[tuple[str, str]] = set()
    if not data_path.is_file():
        mismatches.append("normalized market data is missing")
    else:
        if _file_sha256(data_path) != manifest.get("data_sha256"):
            mismatches.append("normalized market data hash mismatch")
        with data_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"datetime", "asset_id", "open", "high", "low", "close", "volume", "amount"}
            if set(reader.fieldnames or ()) != required:
                mismatches.append("normalized market data schema mismatch")
            for line_number, row in enumerate(reader, 2):
                rows += 1
                asset_id = str(row.get("asset_id", ""))
                timestamp_text = str(row.get("datetime", ""))
                assets.add(asset_id)
                observation = (timestamp_text, asset_id)
                if observation in observations:
                    mismatches.append(
                        f"normalized market data contains a duplicate at line {line_number}"
                    )
                observations.add(observation)
                try:
                    timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
                    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                        raise ValueError("timestamp has no timezone")
                    if lock is not None:
                        start = datetime.fromisoformat(lock.start.replace("Z", "+00:00"))
                        end = datetime.fromisoformat(lock.end.replace("Z", "+00:00"))
                        if not start <= timestamp <= end:
                            raise ValueError("timestamp is outside the locked period")
                        if asset_id not in lock.assets:
                            raise ValueError("asset is outside the locked universe")
                    values = {
                        field: _validate_decimal(
                            str(row.get(field, "")),
                            field=f"market_data.csv:{line_number}:{field}",
                        )
                        for field in ("open", "high", "low", "close", "volume", "amount")
                    }
                    _validate_bar(
                        open_=values["open"],
                        high=values["high"],
                        low=values["low"],
                        close=values["close"],
                        volume=values["volume"],
                        amount=values["amount"],
                        field=f"market_data.csv:{line_number}",
                    )
                except (TypeError, ValueError) as exc:
                    mismatches.append(f"invalid normalized market data row {line_number}: {exc}")
        if rows != int(manifest.get("rows", -1)):
            mismatches.append("normalized market data row count mismatch")
        if sorted(assets) != list(manifest.get("assets", [])):
            mismatches.append("normalized market data asset set mismatch")
    return PublicDatasetVerification(
        passed=not mismatches,
        dataset_id=dataset_id,
        rows=rows,
        assets=len(assets),
        mismatches=tuple(mismatches),
    )


def _resolve_dataset_member(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("dataset manifest paths must be non-empty relative paths")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("dataset manifest paths must not contain symbolic links")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("dataset manifest path escaped the dataset directory") from exc
    return resolved
