"""Fail-closed coverage for checksum-locked public evidence data."""

from __future__ import annotations

import json
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from factorminer.data.public_archive import (
    PublicArchive,
    PublicDatasetLock,
    load_public_dataset_lock,
    lock_public_dataset_spec,
    prepare_public_dataset,
    verify_prepared_public_dataset,
)


def _archive_bytes(tmp_path: Path, *, unsafe: bool = False, invalid_bar: bool = False) -> bytes:
    path = tmp_path / "source.zip"
    member = "../escape.csv" if unsafe else "BTCUSDT-1d-2024-01.csv"
    rows = (
        "1704067200000,42000,43000,41000,42500,100,1704153599999,4250000,10,50,2125000,0\n"
        "1704153600000,42500,44000,42000,43800,110,1704239999999,4800000,12,55,2400000,0\n"
    )
    if invalid_bar:
        rows = rows.replace("42000,43000,41000,42500", "42000,41000,43000,42500")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(member, rows)
    return path.read_bytes()


def _ecb_csv_bytes() -> bytes:
    return (
        b"FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE\n"
        b"D,USD,EUR,SP00,A,2024-01-01,1.25\n"
        b"D,USD,EUR,SP00,A,2024-01-02,1.20\n"
    )


def _lock(tmp_path: Path, content: bytes) -> Path:
    digest = sha256(content).hexdigest()
    lock = PublicDatasetLock(
        schema_version=2,
        dataset_id="fixture-v1",
        provider="fixture",
        asset_class="crypto",
        frequency="daily",
        periods_per_year=365,
        liquidity_evidence=True,
        interval="1d",
        start="2024-01-01T00:00:00Z",
        end="2024-01-02T23:59:59Z",
        assets=("BTCUSDT",),
        allowed_hosts=("data.example.test",),
        license={
            "data_license_class": "publicly_retrievable",
            "redistribution_status": "not_asserted",
            "source_url": "https://data.example.test/",
            "terms_url": "https://data.example.test/terms",
        },
        universe={"selection": "frozen_fixture"},
        availability_lag="one day",
        point_in_time_limitations=("fixture",),
        transformations=(),
        target={
            "name": "next_close",
            "entry_delay_bars": 0,
            "holding_bars": 1,
            "price_pair": "close_to_close",
            "return_transform": "simple",
        },
        archives=(
            PublicArchive(
                asset_id="BTCUSDT",
                period="2024-01",
                url="https://data.example.test/BTCUSDT-1d-2024-01.zip",
                sha256=digest,
            ),
        ),
    )
    lock_path = tmp_path / "dataset.lock.json"
    lock_path.write_text(json.dumps(lock.to_dict(), sort_keys=True))
    return lock_path


def _seed_cache(tmp_path: Path, lock_path: Path, content: bytes) -> Path:
    lock = load_public_dataset_lock(lock_path)
    archive = lock.archives[0]
    cache = tmp_path / "cache"
    destination = cache / archive.sha256 / Path(archive.url).name
    destination.parent.mkdir(parents=True)
    destination.write_bytes(content)
    return cache


def test_prepare_and_verify_locked_archive_offline(tmp_path: Path) -> None:
    content = _archive_bytes(tmp_path)
    lock_path = _lock(tmp_path, content)
    cache = _seed_cache(tmp_path, lock_path, content)
    output = prepare_public_dataset(lock_path, tmp_path / "prepared", cache_dir=cache, offline=True)

    result = verify_prepared_public_dataset(output)
    assert result.passed is True, result.mismatches
    assert result.rows == 2
    assert result.assets == 1
    data = (output / "market_data.csv").read_text()
    assert "2024-01-01T00:00:00Z,BTCUSDT" in data


def test_prepared_dataset_tampering_fails_verification(tmp_path: Path) -> None:
    content = _archive_bytes(tmp_path)
    lock_path = _lock(tmp_path, content)
    cache = _seed_cache(tmp_path, lock_path, content)
    output = prepare_public_dataset(lock_path, tmp_path / "prepared", cache_dir=cache, offline=True)
    with (output / "market_data.csv").open("a") as handle:
        handle.write("tampered\n")

    result = verify_prepared_public_dataset(output)
    assert result.passed is False
    assert "normalized market data hash mismatch" in result.mismatches


def test_prepared_manifest_cannot_override_locked_evidence_metadata(tmp_path: Path) -> None:
    content = _archive_bytes(tmp_path)
    lock_path = _lock(tmp_path, content)
    cache = _seed_cache(tmp_path, lock_path, content)
    output = prepare_public_dataset(lock_path, tmp_path / "prepared", cache_dir=cache, offline=True)
    manifest_path = output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["liquidity_evidence"] = False
    manifest["license"] = {
        "data_license_class": "public_domain",
        "redistribution_status": "allowed",
        "source_url": "https://example.test",
        "terms_url": "https://example.test",
    }
    manifest_path.write_text(json.dumps(manifest))

    result = verify_prepared_public_dataset(output)
    assert result.passed is False
    assert "liquidity_evidence differs between lock and manifest" in result.mismatches
    assert "license differs between lock and manifest" in result.mismatches


def test_prepare_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    content = _archive_bytes(tmp_path, unsafe=True)
    lock_path = _lock(tmp_path, content)
    cache = _seed_cache(tmp_path, lock_path, content)
    with pytest.raises(ValueError, match="unsafe archive member"):
        prepare_public_dataset(lock_path, tmp_path / "prepared", cache_dir=cache, offline=True)
    assert not (tmp_path / "escape.csv").exists()


def test_lock_spec_resolves_and_pins_upstream_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _archive_bytes(tmp_path)
    digest = sha256(content).hexdigest()
    spec = {
        "dataset_id": "fixture-v1",
        "provider": "fixture",
        "asset_class": "crypto",
        "frequency": "daily",
        "periods_per_year": 365,
        "liquidity_evidence": True,
        "interval": "1d",
        "target": {
            "name": "next_close",
            "entry_delay_bars": 0,
            "holding_bars": 1,
            "price_pair": "close_to_close",
            "return_transform": "simple",
        },
        "start_month": "2024-01",
        "end_month": "2024-01",
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-31T23:59:59Z",
        "assets": ["BTCUSDT"],
        "allowed_hosts": ["data.example.test"],
        "url_template": "https://data.example.test/{asset}-{interval}-{period}.zip",
        "license": {
            "data_license_class": "publicly_retrievable",
            "redistribution_status": "not_asserted",
            "source_url": "https://data.example.test/",
            "terms_url": "https://data.example.test/terms",
        },
        "universe": {"selection": "frozen_fixture"},
        "availability_lag": "one day",
        "point_in_time_limitations": ["fixture"],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(
        "factorminer.data.public_archive._read_url_bytes",
        lambda *args, **kwargs: f"{digest}  archive.zip\n".encode(),
    )

    output = lock_public_dataset_spec(spec_path, tmp_path / "dataset.lock.json")
    lock = load_public_dataset_lock(output)
    assert lock.archives[0].sha256 == digest
    assert lock.archives[0].url.endswith("BTCUSDT-1d-2024-01.zip")


def test_lock_rejects_unapproved_host(tmp_path: Path) -> None:
    content = _archive_bytes(tmp_path)
    lock_path = _lock(tmp_path, content)
    payload = json.loads(lock_path.read_text())
    payload["archives"][0]["url"] = "https://metadata.internal/source.zip"
    lock_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not allow-listed"):
        load_public_dataset_lock(lock_path)


def test_prepare_rejects_invalid_ohlc_bounds(tmp_path: Path) -> None:
    content = _archive_bytes(tmp_path, invalid_bar=True)
    lock_path = _lock(tmp_path, content)
    cache = _seed_cache(tmp_path, lock_path, content)
    with pytest.raises(ValueError, match="OHLC bounds"):
        prepare_public_dataset(lock_path, tmp_path / "prepared", cache_dir=cache, offline=True)


def test_verifier_rejects_manifest_path_escape(tmp_path: Path) -> None:
    content = _archive_bytes(tmp_path)
    lock_path = _lock(tmp_path, content)
    cache = _seed_cache(tmp_path, lock_path, content)
    output = prepare_public_dataset(lock_path, tmp_path / "prepared", cache_dir=cache, offline=True)
    manifest_path = output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data_path"] = "../../outside.csv"
    manifest_path.write_text(json.dumps(manifest))

    result = verify_prepared_public_dataset(output)
    assert result.passed is False
    assert "relative paths" in result.mismatches[0]


def test_redirect_to_unapproved_host_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from factorminer.data.public_archive import _read_url_bytes

    class RedirectedResponse:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self) -> str:
            return "https://metadata.internal/secret"

        def read(self, size: int) -> bytes:
            return b"secret"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: RedirectedResponse())
    with pytest.raises(ValueError, match="not allow-listed"):
        _read_url_bytes(
            "https://data.example.test/archive.zip",
            allowed_hosts=("data.example.test",),
            max_bytes=100,
        )


def test_ecb_content_lock_and_transformation_are_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = _ecb_csv_bytes()
    spec = {
        "dataset_id": "ecb-fixture-v1",
        "provider": "ECB fixture",
        "asset_class": "fx",
        "frequency": "daily",
        "periods_per_year": 252,
        "liquidity_evidence": False,
        "interval": "1d",
        "target": {
            "name": "next_close",
            "entry_delay_bars": 0,
            "holding_bars": 1,
            "price_pair": "close_to_close",
            "return_transform": "simple",
        },
        "periods": ["2024"],
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-02T23:59:59Z",
        "assets": ["USD"],
        "allowed_hosts": ["data.example.test"],
        "url_template": "https://data.example.test/{asset}.csv",
        "checksum_mode": "content_sha256",
        "parser": "ecb_exchange_rate_csv",
        "license": {
            "data_license_class": "redistributable_with_attribution",
            "redistribution_status": "allowed",
            "source_url": "https://data.example.test/",
            "terms_url": "https://data.example.test/terms",
            "attribution": "Source: fixture",
        },
        "universe": {"selection": "fixture"},
        "availability_lag": "one day",
        "transformations": ["invert rate"],
        "point_in_time_limitations": ["fixture"],
    }
    spec_path = tmp_path / "ecb.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(
        "factorminer.data.public_archive._read_url_bytes",
        lambda *args, **kwargs: content,
    )
    lock_path = lock_public_dataset_spec(spec_path, tmp_path / "ecb.lock.json")
    lock = load_public_dataset_lock(lock_path)
    assert lock.archives[0].sha256 == sha256(content).hexdigest()
    assert lock.archives[0].checksum_source == "content_sha256"
    cache = _seed_cache(tmp_path, lock_path, content)
    output = prepare_public_dataset(
        lock_path,
        tmp_path / "ecb-prepared",
        cache_dir=cache,
        offline=True,
    )
    result = verify_prepared_public_dataset(output)
    assert result.passed is True, result.mismatches
    prepared = (output / "market_data.csv").read_text()
    assert "2024-01-01T00:00:00Z,USD,0.8,0.8,0.8,0.8,1,0.8" in prepared
