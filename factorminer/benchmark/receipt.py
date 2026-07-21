"""Assembles, publishes, and offline-verifies ExternalResearchReceipt artifacts."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from factorminer.architecture.research_receipt import (
    DatasetCommitment,
    EvidenceTier,
    ExternalResearchReceipt,
    RunStatus,
    SourceManifestRef,
    derive_release_id,
    derive_release_id_from_payload,
)
from factorminer.benchmark.reporting import file_sha256
from factorminer.core.provenance import stable_digest


def _git_sha(repo_root: Path) -> str:
    """Best-effort commit identity, including the exact dirty-tree content."""
    try:
        rev = subprocess.run(  # noqa: S603,S607 - fixed argv, no shell, no user input
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if rev.returncode != 0:
            return "unknown"
        sha = rev.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode != 0 or not status.stdout.strip():
            return sha

        digest = sha256()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=repo_root,
            capture_output=True,
            timeout=10,
            check=False,
        )
        digest.update(diff.stdout)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_root,
            capture_output=True,
            timeout=10,
            check=False,
        )
        for relative_raw in untracked.stdout.split(b"\0"):
            if not relative_raw:
                continue
            relative = relative_raw.decode("utf-8", errors="surrogateescape")
            path = repo_root / relative
            digest.update(relative_raw)
            if path.is_file():
                digest.update(path.read_bytes())
        return f"{sha}-dirty-{digest.hexdigest()}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_research_receipt(
    *,
    phase2_manifest: dict[str, Any],
    phase2_manifest_path: Path,
    evidence_tier: EvidenceTier,
    run_status: RunStatus,
    seed: int,
    config_path: Path,
    data_path: Path | None,
    dataset_descriptor: dict[str, Any],
    protocol_admission_contract: dict[str, float],
    protocol_replacement_contract: dict[str, float],
    memory_policy_schema: dict[str, Any] | None,
    ic_metric: str,
    metric_version: str,
    walk_forward_contract: dict[str, Any] | None = None,
    stress_contract: dict[str, Any] | None = None,
    data_license_class: str = "synthetic",
    commitment_key: str | None = None,
    supersedes_release_id: str | None = None,
) -> ExternalResearchReceipt:
    """Assemble a receipt from an already-written phase2 manifest.

    Pure with respect to the filesystem beyond reading `config_path`,
    `data_path`, `uv.lock`, and every path in
    `phase2_manifest["artifact_paths"]` -- callers are responsible for
    having already produced those files before calling this.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env_lock_path = repo_root / "uv.lock"

    if data_path is not None and data_path.exists():
        plain_digest = file_sha256(data_path)
        if evidence_tier == EvidenceTier.PRIVATE_PARTNER_OBSERVED:
            if not commitment_key:
                raise ValueError("private_partner_observed requires a 32-byte commitment key")
            key = bytes.fromhex(commitment_key)
            if len(key) < 32:
                raise ValueError("commitment key must contain at least 32 bytes")
            digest = hmac.new(key, bytes.fromhex(plain_digest), sha256).hexdigest()
            dataset_commitment = DatasetCommitment(scheme="hmac-sha256", digest=digest)
        else:
            dataset_commitment = DatasetCommitment(scheme="sha256", digest=plain_digest)
    else:
        dataset_commitment = DatasetCommitment(scheme="none", digest="mock")

    artifact_inventory = _artifact_inventory(phase2_manifest)
    missing = [name for name, path in artifact_inventory.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "receipt cannot be built with missing artifacts: " + ", ".join(sorted(missing))
        )
    artifact_sha256s = {name: file_sha256(path) for name, path in artifact_inventory.items()}

    runtime_refs = list(phase2_manifest.get("runtime_manifest_refs", []))
    baseline_provenance = {
        str(ref.get("baseline") or f"runtime_{index}"): dict(ref.get("baseline_provenance") or {})
        for index, ref in enumerate(runtime_refs)
    }
    factor_library_sha256 = stable_digest(
        {
            "runtime_manifests": [str(ref.get("sha256", "")) for ref in runtime_refs],
            "runtime_topk": artifact_sha256s.get("phase2/runtime_topk", ""),
        }
    )

    draft = ExternalResearchReceipt(
        release_id="",
        evidence_tier=evidence_tier,
        run_status=run_status,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        code_sha=_git_sha(repo_root),
        config_sha256=file_sha256(config_path),
        environment_lock_sha256=(
            file_sha256(env_lock_path) if env_lock_path.exists() else "unknown"
        ),
        seed=seed,
        dataset_descriptor=dict(dataset_descriptor),
        dataset_commitment=dataset_commitment,
        data_license_class=data_license_class,
        factor_library_sha256=factor_library_sha256,
        baseline_provenance=baseline_provenance,
        split_and_freeze_contract=dict(walk_forward_contract or {}),
        cost_and_capacity_assumptions=dict(stress_contract or {}),
        metric_definitions={"ic_metric_version": metric_version, "ic_metric": ic_metric},
        selection_policy={
            "memory_policy": dict(memory_policy_schema or {}),
            "admission_contract": dict(protocol_admission_contract),
            "replacement_contract": dict(protocol_replacement_contract),
        },
        source_manifest=SourceManifestRef(
            path=str(phase2_manifest_path), sha256=file_sha256(phase2_manifest_path)
        ),
        artifact_sha256s=artifact_sha256s,
        supersedes_release_id=supersedes_release_id,
    )
    return replace(draft, release_id=derive_release_id(draft))


def _resolve_bundle_path(
    path: str | Path, *, base_dir: Path | None, allow_absolute: bool = True
) -> Path:
    resolved = Path(path)
    if resolved.is_absolute() and not allow_absolute:
        raise ValueError("portable bundle paths must be relative")
    if not resolved.is_absolute() and base_dir is not None:
        if not resolved.parts or ".." in resolved.parts:
            raise ValueError("bundle paths must be non-empty relative paths")
        root = base_dir.resolve()
        resolved = (root / resolved).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("bundle path escaped its release directory") from exc
    return resolved


def _artifact_inventory(
    phase2_manifest: dict[str, Any],
    *,
    base_dir: Path | None = None,
    allow_absolute: bool = True,
) -> dict[str, Path]:
    """Return every top-level and referenced-runtime artifact with stable keys.

    Historical manifests contain absolute producer-machine paths. Portable
    release manifests contain paths relative to the manifest itself. Supporting
    both here keeps old receipts verifiable while making new bundles relocatable.
    """
    inventory = {
        f"phase2/{name}": _resolve_bundle_path(
            path, base_dir=base_dir, allow_absolute=allow_absolute
        )
        for name, path in dict(phase2_manifest.get("artifact_paths", {})).items()
    }
    for index, ref in enumerate(phase2_manifest.get("runtime_manifest_refs", [])):
        for name, path in dict(ref.get("artifact_paths", {})).items():
            inventory[f"runtime/{index}/{name}"] = _resolve_bundle_path(
                path, base_dir=base_dir, allow_absolute=allow_absolute
            )
    return inventory


def _portable_artifact_path(name: str, source: Path) -> Path:
    """Map an inventory key to a safe, deterministic bundle-relative path."""
    safe_parts = []
    for part in Path(name).parts:
        clean = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in part)
        safe_parts.append(clean or "artifact")
    return Path("artifacts", *safe_parts, source.name)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _rewrite_manifest_for_bundle(
    phase2_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Path], dict[Path, bytes]]:
    """Create a portable manifest and in-memory normalized runtime manifests."""
    portable = json.loads(json.dumps(phase2_manifest))
    source_inventory = _artifact_inventory(phase2_manifest)
    destination_by_name = {
        name: _portable_artifact_path(name, source) for name, source in source_inventory.items()
    }
    if len(set(destination_by_name.values())) != len(destination_by_name):
        raise ValueError("artifact names collide after portable-path normalization")

    portable["artifact_paths"] = {
        name: destination_by_name[f"phase2/{name}"].as_posix()
        for name in dict(phase2_manifest.get("artifact_paths", {}))
    }

    generated_files: dict[Path, bytes] = {}
    source_refs = list(phase2_manifest.get("runtime_manifest_refs", []))
    portable_refs = list(portable.get("runtime_manifest_refs", []))
    for index, (source_ref, portable_ref) in enumerate(
        zip(source_refs, portable_refs, strict=True)
    ):
        portable_ref["artifact_paths"] = {
            name: destination_by_name[f"runtime/{index}/{name}"].as_posix()
            for name in dict(source_ref.get("artifact_paths", {}))
        }
        source_runtime_path = Path(str(source_ref.get("path", "")))
        if not source_runtime_path.is_file():
            raise FileNotFoundError(f"runtime manifest {index} is missing: {source_runtime_path}")

        runtime_payload = json.loads(source_runtime_path.read_text())
        if isinstance(runtime_payload, dict) and "artifact_paths" in runtime_payload:
            runtime_payload["artifact_paths"] = dict(portable_ref["artifact_paths"])
        runtime_rel = Path("manifests", f"runtime_{index}.json")
        runtime_bytes = _json_bytes(runtime_payload)
        generated_files[runtime_rel] = runtime_bytes
        portable_ref["source_sha256"] = str(
            source_ref.get("sha256") or file_sha256(source_runtime_path)
        )
        portable_ref["path"] = runtime_rel.as_posix()
        portable_ref["sha256"] = sha256(runtime_bytes).hexdigest()

    portable["portable_bundle"] = {
        "schema_version": 1,
        "path_semantics": "relative_to_release_root",
    }
    if "output_dir" in portable:
        portable["output_dir"] = "."
    if "runtime_manifest_root" in portable:
        portable["runtime_manifest_root"] = "."
    return portable, source_inventory, generated_files


def publish_portable_bundle(
    receipt: ExternalResearchReceipt,
    *,
    phase2_manifest: dict[str, Any],
    releases_root: Path,
    commitment_input: Path | None = None,
    include_commitment_input: bool = False,
) -> Path:
    """Publish a self-contained, relocatable evidence release.

    The existing :func:`write_receipt` intentionally preserves producer paths
    for compatibility. This publication path copies every declared artifact,
    rewrites paths relative to the release root, and optionally includes a
    publicly redistributable committed dataset. Private partner data can never
    be included through this API.
    """
    portable_manifest, source_inventory, generated_files = _rewrite_manifest_for_bundle(
        phase2_manifest
    )
    missing = [name for name, path in source_inventory.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "portable bundle cannot be built with missing artifacts: " + ", ".join(sorted(missing))
        )
    source_hashes = {name: file_sha256(path) for name, path in source_inventory.items()}
    if source_hashes != receipt.artifact_sha256s:
        raise ValueError("artifact inventory no longer matches the receipt")

    descriptor = dict(receipt.dataset_descriptor)
    dataset_copy: tuple[Path, Path] | None = None
    if include_commitment_input:
        if receipt.evidence_tier != EvidenceTier.PUBLIC_REPRODUCIBLE:
            raise ValueError("only public_reproducible receipts may bundle committed data")
        if receipt.data_license_class not in {
            "public_domain",
            "redistributable_with_attribution",
            "vendor_redistributable_sample",
        }:
            raise ValueError("the receipt data license class does not permit public bundling")
        if commitment_input is None or not commitment_input.is_file():
            raise FileNotFoundError("include_commitment_input requires an existing dataset file")
        if file_sha256(commitment_input) != receipt.dataset_commitment.digest:
            raise ValueError("commitment input does not match the receipt dataset commitment")
        dataset_rel = Path("dataset", commitment_input.name)
        descriptor["bundle_path"] = dataset_rel.as_posix()
        dataset_copy = (commitment_input, dataset_rel)

    manifest_bytes = _json_bytes(portable_manifest)
    normalized_artifacts: dict[str, bytes] = {}
    if "phase2/manifest" in source_inventory:
        normalized_artifacts["phase2/manifest"] = manifest_bytes
    for index, _ in enumerate(portable_manifest.get("runtime_manifest_refs", [])):
        name = f"runtime/{index}/manifest"
        generated_path = Path("manifests", f"runtime_{index}.json")
        if name in source_inventory and generated_path in generated_files:
            normalized_artifacts[name] = generated_files[generated_path]
    portable_hashes = dict(source_hashes)
    portable_hashes.update(
        {name: sha256(content).hexdigest() for name, content in normalized_artifacts.items()}
    )
    portable_draft = replace(
        receipt,
        release_id="",
        dataset_descriptor=descriptor,
        artifact_sha256s=portable_hashes,
        source_manifest=SourceManifestRef(
            path="manifest.json", sha256=sha256(manifest_bytes).hexdigest()
        ),
    )
    portable_receipt = replace(portable_draft, release_id=derive_release_id(portable_draft))
    release_dir = releases_root / portable_receipt.release_id
    receipt_path = release_dir / "receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if existing != portable_receipt.to_dict():
            raise FileExistsError(
                f"{receipt_path} exists with divergent content; refusing to overwrite"
            )
        return receipt_path

    releases_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".receipt-", dir=releases_root))
    try:
        (staging / "manifest.json").write_bytes(manifest_bytes)
        for relative, content in generated_files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        for name, source in source_inventory.items():
            destination = staging / _portable_artifact_path(name, source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name in normalized_artifacts:
                destination.write_bytes(normalized_artifacts[name])
            else:
                shutil.copyfile(source, destination)
        if dataset_copy is not None:
            source, relative = dataset_copy
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        (staging / "receipt.json").write_bytes(_json_bytes(portable_receipt.to_dict()))
        os.replace(staging, release_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    (releases_root / "LATEST").write_text(portable_receipt.release_id)
    return receipt_path


def generate_commitment_key() -> str:
    """32 bytes of randomness, hex-encoded. Never stored inside receipt.json."""
    return secrets.token_hex(32)


generate_commitment_nonce = generate_commitment_key


def seal_private_commitment(
    receipt: ExternalResearchReceipt, *, nonce: str
) -> ExternalResearchReceipt:
    """Re-key an existing sha256 dataset_commitment with a withheld nonce.

    Call only when evidence_tier == PRIVATE_PARTNER_OBSERVED. Returns a new
    receipt; caller must re-derive release_id (already done here) and persist
    `nonce` itself -- write_receipt's caller writes it to a sidecar, never
    into receipt.json.
    """
    plain_digest = receipt.dataset_commitment.digest
    keyed = hmac.new(bytes.fromhex(nonce), bytes.fromhex(plain_digest), sha256).hexdigest()
    sealed = replace(
        receipt,
        evidence_tier=EvidenceTier.PRIVATE_PARTNER_OBSERVED,
        data_license_class="proprietary_licensed",
        dataset_commitment=DatasetCommitment(scheme="hmac-sha256", digest=keyed, nonce=None),
    )
    return replace(sealed, release_id=derive_release_id(sealed))


def write_receipt(receipt: ExternalResearchReceipt, releases_root: Path) -> Path:
    """Content-addressed, append-only publish. Never overwrites divergent content."""
    derived_release_id = derive_release_id(receipt)
    if receipt.release_id != derived_release_id:
        raise ValueError(
            f"receipt release_id is stale: claimed {receipt.release_id}, "
            f"derived {derived_release_id}"
        )
    release_dir = releases_root / receipt.release_id
    receipt_path = release_dir / "receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if existing != receipt.to_dict():
            raise FileExistsError(
                f"{receipt_path} exists with divergent content; refusing to overwrite"
            )
        return receipt_path  # identical content already published; idempotent no-op
    release_dir.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    (releases_root / "LATEST").write_text(receipt.release_id)
    return receipt_path


@dataclass(frozen=True)
class ReceiptVerificationResult:
    release_id: str
    passed: bool
    mismatches: list[str] = field(default_factory=list)


def verify_research_receipt(
    release_dir: Path,
    *,
    commitment_nonce: str | None = None,
    commitment_key: str | None = None,
    commitment_input: Path | None = None,
) -> ReceiptVerificationResult:
    """Offline: re-hash every recorded artifact and re-derive release_id."""
    receipt_path = release_dir / "receipt.json"
    try:
        payload = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return ReceiptVerificationResult(
            release_id=release_dir.name,
            passed=False,
            mismatches=[f"receipt.json cannot be read as JSON: {exc}"],
        )
    if not isinstance(payload, dict):
        return ReceiptVerificationResult(
            release_id=release_dir.name,
            passed=False,
            mismatches=["receipt.json must contain a JSON object"],
        )
    mismatches: list[str] = []

    claimed_release_id = payload.get("release_id", "")
    try:
        parsed_receipt = ExternalResearchReceipt.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return ReceiptVerificationResult(
            release_id=str(claimed_release_id),
            passed=False,
            mismatches=[f"receipt schema validation failed: {exc}"],
        )
    else:
        if parsed_receipt.to_dict() != payload:
            mismatches.append("receipt schema round-trip changed the serialized payload")
    if claimed_release_id != release_dir.name:
        mismatches.append(
            f"release_id in receipt.json ({claimed_release_id}) does not match "
            f"directory name ({release_dir.name})"
        )
    derived_release_id = derive_release_id_from_payload(payload)
    if claimed_release_id != derived_release_id:
        mismatches.append(
            f"release_id content mismatch: claimed {claimed_release_id}, "
            f"derived {derived_release_id}"
        )

    # artifact_sha256s is keyed by artifact NAME -> digest; paths live in the
    # referenced source_manifest, so re-open it to resolve each name to a path.
    source_manifest = payload.get("source_manifest") or {}
    try:
        manifest_path = _resolve_bundle_path(source_manifest.get("path", ""), base_dir=release_dir)
    except ValueError as exc:
        mismatches.append(f"invalid source_manifest path: {exc}")
        manifest_path = None
    if manifest_path is not None and manifest_path.is_file():
        manifest_digest = file_sha256(manifest_path)
        if manifest_digest != source_manifest.get("sha256"):
            mismatches.append(f"source_manifest at {manifest_path} has changed since release")
        try:
            manifest_payload = json.loads(manifest_path.read_text())
            if not isinstance(manifest_payload, dict):
                raise ValueError("source manifest must contain a JSON object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            mismatches.append(f"source_manifest cannot be read as JSON: {exc}")
            manifest_payload = {}
            inventory = {}
        else:
            try:
                inventory = _artifact_inventory(
                    manifest_payload,
                    base_dir=manifest_path.parent,
                    allow_absolute=not bool(manifest_payload.get("portable_bundle")),
                )
            except ValueError as exc:
                mismatches.append(f"invalid artifact path: {exc}")
                inventory = {}
        expected_artifacts = dict(payload.get("artifact_sha256s", {}))
        missing_records = sorted(set(inventory) - set(expected_artifacts))
        unexpected_records = sorted(set(expected_artifacts) - set(inventory))
        if missing_records:
            mismatches.append("artifact hashes missing from receipt: " + ", ".join(missing_records))
        if unexpected_records:
            mismatches.append(
                "receipt contains unknown artifact hashes: " + ", ".join(unexpected_records)
            )
        for name, expected_digest in expected_artifacts.items():
            path = inventory.get(name)
            if path is None or not path.is_file():
                mismatches.append(f"artifact '{name}' is missing (expected at {path})")
                continue
            actual_digest = file_sha256(path)
            if actual_digest != expected_digest:
                mismatches.append(f"artifact '{name}' hash changed: {path}")

        for index, ref in enumerate(manifest_payload.get("runtime_manifest_refs", [])):
            try:
                nested_path = _resolve_bundle_path(
                    str(ref.get("path", "")),
                    base_dir=manifest_path.parent,
                    allow_absolute=not bool(manifest_payload.get("portable_bundle")),
                )
            except ValueError as exc:
                mismatches.append(f"runtime manifest {index} path is invalid: {exc}")
                continue
            if not nested_path.is_file():
                mismatches.append(f"runtime manifest {index} is missing: {nested_path}")
                continue
            if file_sha256(nested_path) != ref.get("sha256"):
                mismatches.append(f"runtime manifest {index} hash changed: {nested_path}")
    elif manifest_path is not None:
        mismatches.append(f"source_manifest missing: {manifest_path}")

    commitment = payload.get("dataset_commitment", {})
    scheme = commitment.get("scheme")
    if scheme == "sha256":
        bundled_input = payload.get("dataset_descriptor", {}).get("bundle_path")
        if commitment_input is None and bundled_input:
            try:
                candidate = _resolve_bundle_path(
                    bundled_input,
                    base_dir=release_dir,
                    allow_absolute=False,
                )
            except ValueError as exc:
                mismatches.append(f"dataset bundle path is invalid: {exc}")
            else:
                if candidate.is_file():
                    commitment_input = candidate
        if commitment_input is None or not commitment_input.is_file():
            mismatches.append("dataset_commitment: sha256 verification requires --commitment-input")
        elif file_sha256(commitment_input) != commitment.get("digest"):
            mismatches.append("dataset_commitment: SHA-256 mismatch")
    elif scheme == "hmac-sha256":
        key_hex = commitment_key or commitment_nonce
        if not key_hex or commitment_input is None or not commitment_input.is_file():
            mismatches.append(
                "dataset_commitment: HMAC verification requires a key and --commitment-input"
            )
        else:
            try:
                key = bytes.fromhex(key_hex)
                local_digest = file_sha256(commitment_input)
                recomputed = hmac.new(key, bytes.fromhex(local_digest), sha256).hexdigest()
            except ValueError:
                mismatches.append("dataset_commitment: invalid hexadecimal HMAC key")
            else:
                if recomputed != commitment.get("digest"):
                    mismatches.append(
                        "dataset_commitment: HMAC mismatch against supplied commitment-input"
                    )

    return ReceiptVerificationResult(
        release_id=claimed_release_id, passed=not mismatches, mismatches=mismatches
    )
