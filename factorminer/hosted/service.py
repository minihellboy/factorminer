"""Tenant isolation, quotas, and safe job contracts for the hosted pilot."""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from factorminer.benchmark.reporting import file_sha256
from factorminer.hosted.models import (
    ALLOWED_SCOPES,
    CredentialIssue,
    JobKind,
    JobRecord,
    Principal,
    TenantQuota,
    TenantStatus,
)
from factorminer.hosted.store import HostedStore

BENCHMARK_MODES = frozenset(
    {"table1", "ablation-memory", "ablation-strategy", "cost-pressure", "efficiency", "suite"}
)
REPORT_FORMATS = frozenset({"markdown", "html"})
MINING_LOOPS = frozenset({"ralph", "helix"})


def _bounded_integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


class HostedPilotService:
    """Application boundary for all hosted-pilot state and filesystem access."""

    def __init__(self, store: HostedStore, storage_root: str | Path) -> None:
        self.store = store
        self.storage_root = Path(storage_root).resolve()
        if self.storage_root == Path(self.storage_root.anchor):
            raise ValueError("hosted storage root must not be a filesystem root")

    def initialize(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.store.initialize()

    def tenant_root(self, tenant_id: str) -> Path:
        candidate = self.storage_root / tenant_id
        if candidate.is_symlink():
            raise ValueError("tenant roots must not be symbolic links")
        root = candidate.resolve()
        try:
            root.relative_to(self.storage_root)
        except ValueError as exc:
            raise ValueError("tenant path escaped the hosted storage root") from exc
        return root

    def create_tenant(
        self,
        tenant_id: str,
        *,
        display_name: str,
        quota: TenantQuota | None = None,
        retention_days: int = 30,
    ) -> dict[str, Any]:
        tenant = self.store.create_tenant(
            tenant_id,
            display_name=display_name,
            quota=quota,
            retention_days=retention_days,
        )
        root = self.tenant_root(tenant_id)
        for directory in ("inputs", "jobs", "exports"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        return tenant

    def issue_credential(
        self,
        tenant_id: str,
        *,
        label: str,
        scopes: tuple[str, ...] = tuple(sorted(ALLOWED_SCOPES)),
        expires_at: datetime | None = None,
    ) -> CredentialIssue:
        return self.store.issue_credential(
            tenant_id,
            label=label,
            scopes=scopes,
            expires_at=expires_at,
        )

    def authenticate(self, token: str) -> Principal | None:
        return self.store.verify_token(token)

    def quota_for(self, tenant_id: str) -> TenantQuota:
        tenant = self.store.get_tenant(tenant_id)
        if tenant["status"] != TenantStatus.ACTIVE.value:
            raise PermissionError("tenant is not active")
        return TenantQuota.from_dict(tenant["quota"])

    def _record_request(self, principal: Principal, operation: str) -> TenantQuota:
        quota = self.quota_for(principal.tenant_id)
        self.store.record_rate_limited_request(
            principal.tenant_id,
            operation=operation,
            limit=quota.requests_per_minute,
            since=datetime.now(UTC) - timedelta(minutes=1),
            metadata={"credential_id": principal.credential_id},
        )
        return quota

    def resolve_tenant_path(
        self,
        tenant_id: str,
        relative_path: str | Path,
        *,
        must_exist: bool = False,
        file_only: bool = False,
    ) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("hosted paths must be non-empty tenant-relative paths")
        root = self.tenant_root(tenant_id)
        candidate = root / relative
        current = root
        for part in relative.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ValueError("symbolic links are forbidden in hosted tenant paths")
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("hosted path escaped the tenant root") from exc
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"tenant path does not exist: {relative.as_posix()}")
        if file_only and resolved.exists() and not resolved.is_file():
            raise ValueError(f"tenant path must be a file: {relative.as_posix()}")
        return resolved

    def _directory_size(self, root: Path) -> int:
        total = 0
        if not root.exists():
            return 0
        for directory, subdirs, files in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            subdirs[:] = [name for name in subdirs if not (directory_path / name).is_symlink()]
            for name in files:
                path = directory_path / name
                if not path.is_symlink():
                    total += path.stat().st_size
        return total

    def tenant_storage_bytes(self, tenant_id: str) -> int:
        """Return regular-file storage usage without following symbolic links."""
        return self._directory_size(self.tenant_root(tenant_id))

    def assert_safe_artifact_tree(self, tenant_id: str, job_id: str) -> None:
        """Reject worker output containing links or paths outside the job root."""
        job_root = self.resolve_tenant_path(tenant_id, Path("jobs") / job_id, must_exist=True)
        for path in job_root.rglob("*"):
            if path.is_symlink():
                raise ValueError("symbolic links are forbidden in hosted job artifacts")
            try:
                path.resolve(strict=True).relative_to(job_root)
            except ValueError as exc:
                raise ValueError("hosted job artifact escaped its job root") from exc

    def put_input(
        self,
        tenant_id: str,
        source: str | Path,
        *,
        destination_name: str | None = None,
    ) -> dict[str, Any]:
        """Operator-side input provisioning; never accepts a server path from MCP."""
        source_path = Path(source)
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError("input source must be an existing regular file")
        quota = self.quota_for(tenant_id)
        size = source_path.stat().st_size
        if size > quota.max_input_bytes:
            raise ValueError("input exceeds tenant input-size quota")
        name = destination_name or source_path.name
        destination = self.resolve_tenant_path(tenant_id, Path("inputs") / name, must_exist=False)
        if destination.exists():
            if file_sha256(destination) == file_sha256(source_path):
                return self.input_descriptor(
                    tenant_id, destination.relative_to(self.tenant_root(tenant_id))
                )
            raise FileExistsError("refusing to overwrite a divergent tenant input")
        projected = self._directory_size(self.tenant_root(tenant_id)) + size
        if projected > quota.max_storage_bytes:
            raise RuntimeError("tenant storage quota exceeded")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.resolve_tenant_path(
            tenant_id,
            destination.relative_to(self.tenant_root(tenant_id)).with_suffix(
                destination.suffix + ".uploading"
            ),
            must_exist=False,
        )
        created_temporary = False
        try:
            with source_path.open("rb") as source_handle, temporary.open("xb") as target_handle:
                created_temporary = True
                shutil.copyfileobj(source_handle, target_handle)
            temporary.replace(destination)
        except Exception:
            if created_temporary and temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
            raise
        self.store.record_usage(
            tenant_id,
            operation="input.provisioned",
            quantity=float(size),
            unit="byte",
            metadata={"path": destination.relative_to(self.tenant_root(tenant_id)).as_posix()},
        )
        self.store.record_audit(
            tenant_id,
            action="input.provisioned",
            subject=destination.relative_to(self.tenant_root(tenant_id)).as_posix(),
            metadata={"bytes": size, "sha256": file_sha256(destination)},
        )
        return self.input_descriptor(
            tenant_id, destination.relative_to(self.tenant_root(tenant_id))
        )

    def input_descriptor(self, tenant_id: str, relative_path: str | Path) -> dict[str, Any]:
        path = self.resolve_tenant_path(tenant_id, relative_path, must_exist=True, file_only=True)
        return {
            "path": path.relative_to(self.tenant_root(tenant_id)).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    def _validate_job_parameters(
        self, tenant_id: str, kind: JobKind, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        supplied = dict(parameters)
        if kind == JobKind.VALIDATE_DATA:
            allowed = {"input_path", "strict", "hdf_key"}
            required = {"input_path"}
            normalized = {
                "input_path": str(supplied["input_path"]),
                "strict": bool(supplied.get("strict", True)),
                "hdf_key": str(supplied.get("hdf_key", "data")),
            }
        elif kind == JobKind.MINE:
            allowed = {"input_path", "loop", "iterations", "batch_size", "target"}
            required = {"input_path", "loop"}
            loop = str(supplied["loop"])
            if loop not in MINING_LOOPS:
                raise ValueError(f"unsupported mining loop: {loop}")
            normalized = {
                "input_path": str(supplied["input_path"]),
                "loop": loop,
                "iterations": _bounded_integer(
                    supplied.get("iterations", 20),
                    field="iterations",
                    minimum=1,
                    maximum=200,
                ),
                "batch_size": _bounded_integer(
                    supplied.get("batch_size", 20),
                    field="batch_size",
                    minimum=1,
                    maximum=100,
                ),
                "target": _bounded_integer(
                    supplied.get("target", 10),
                    field="target",
                    minimum=1,
                    maximum=100,
                ),
            }
        elif kind == JobKind.BENCHMARK:
            allowed = {"input_path", "mode"}
            required = {"input_path", "mode"}
            mode = str(supplied["mode"])
            if mode not in BENCHMARK_MODES or mode == "efficiency":
                raise ValueError("hosted benchmark mode must consume an explicit dataset")
            normalized = {"input_path": str(supplied["input_path"]), "mode": mode}
        elif kind == JobKind.GENERATE_REPORT:
            allowed = {"library_path", "session_log", "benchmarks", "format"}
            required = {"library_path"}
            report_format = str(supplied.get("format", "markdown"))
            if report_format not in REPORT_FORMATS:
                raise ValueError(f"unsupported report format: {report_format}")
            normalized = {
                "library_path": str(supplied["library_path"]),
                "session_log": (
                    str(supplied["session_log"]) if supplied.get("session_log") else None
                ),
                "benchmarks": [str(item) for item in supplied.get("benchmarks", [])],
                "format": report_format,
            }
        elif kind == JobKind.VERIFY_RECEIPT:
            allowed = {"release_dir", "commitment_input"}
            required = {"release_dir"}
            normalized = {
                "release_dir": str(supplied["release_dir"]),
                "commitment_input": (
                    str(supplied["commitment_input"]) if supplied.get("commitment_input") else None
                ),
            }
        else:  # pragma: no cover - enum blocks this
            raise ValueError(f"unsupported hosted job kind: {kind}")
        unknown = set(supplied) - allowed
        missing = required - set(supplied)
        if unknown:
            raise ValueError(f"unknown {kind.value} parameters: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing {kind.value} parameters: {sorted(missing)}")

        path_fields = {
            "input_path",
            "library_path",
            "session_log",
            "release_dir",
            "commitment_input",
        }
        for field in path_fields:
            value = normalized.get(field)
            if value:
                path = self.resolve_tenant_path(tenant_id, value, must_exist=True)
                if field != "release_dir" and not path.is_file():
                    raise ValueError(f"{field} must reference a regular tenant file")
                normalized[field] = path.relative_to(self.tenant_root(tenant_id)).as_posix()
        benchmark_paths: list[str] = []
        for value in normalized.get("benchmarks", []):
            path = self.resolve_tenant_path(tenant_id, value, must_exist=True, file_only=True)
            benchmark_paths.append(path.relative_to(self.tenant_root(tenant_id)).as_posix())
        if "benchmarks" in normalized:
            normalized["benchmarks"] = benchmark_paths
        return normalized

    def submit_job(
        self,
        principal: Principal,
        *,
        kind: JobKind | str,
        parameters: dict[str, Any],
        idempotency_key: str,
        timeout_seconds: int | None = None,
    ) -> JobRecord:
        principal.require("jobs:submit")
        quota = self._record_request(principal, "job.submit")
        selected_kind = JobKind(kind)
        normalized = self._validate_job_parameters(principal.tenant_id, selected_kind, parameters)
        return self.store.submit_job(
            principal.tenant_id,
            kind=selected_kind,
            parameters=normalized,
            idempotency_key=idempotency_key,
            timeout_seconds=timeout_seconds or quota.max_runtime_seconds,
            quota=quota,
        )

    def get_job(self, principal: Principal, job_id: str) -> JobRecord:
        principal.require("jobs:read")
        self._record_request(principal, "job.read")
        return self.store.get_job(principal.tenant_id, job_id)

    def list_jobs(self, principal: Principal, *, limit: int = 100) -> list[JobRecord]:
        principal.require("jobs:read")
        self._record_request(principal, "job.list")
        return self.store.list_jobs(principal.tenant_id, limit=limit)

    def cancel_job(self, principal: Principal, job_id: str) -> JobRecord:
        principal.require("jobs:cancel")
        self._record_request(principal, "job.cancel")
        return self.store.cancel_job(principal.tenant_id, job_id)

    def list_artifacts(self, principal: Principal, job_id: str) -> list[dict[str, Any]]:
        principal.require("artifacts:read")
        self._record_request(principal, "artifact.list")
        self.store.get_job(principal.tenant_id, job_id)
        job_root = self.resolve_tenant_path(
            principal.tenant_id, Path("jobs") / job_id, must_exist=False
        )
        artifacts: list[dict[str, Any]] = []
        if not job_root.exists():
            return artifacts
        for path in sorted(job_root.rglob("*")):
            if path.is_symlink():
                raise ValueError("symbolic links are forbidden in hosted job artifacts")
            if path.is_file():
                artifacts.append(
                    {
                        "path": path.relative_to(self.tenant_root(principal.tenant_id)).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
        return artifacts

    def read_text_artifact(
        self, principal: Principal, job_id: str, relative_path: str, *, max_bytes: int = 1_000_000
    ) -> dict[str, Any]:
        principal.require("artifacts:read")
        self._record_request(principal, "artifact.read")
        self.store.get_job(principal.tenant_id, job_id)
        required_prefix = Path("jobs") / job_id
        requested = Path(relative_path)
        if tuple(requested.parts[:2]) != tuple(required_prefix.parts):
            raise PermissionError("artifact path must belong to the requested job")
        path = self.resolve_tenant_path(
            principal.tenant_id, requested, must_exist=True, file_only=True
        )
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError("artifact exceeds inline read limit")
        content = path.read_text(encoding="utf-8")
        self.store.record_usage(
            principal.tenant_id,
            operation="artifact.download",
            quantity=float(size),
            unit="byte",
            metadata={"job_id": job_id, "path": relative_path},
        )
        self.store.record_audit(
            principal.tenant_id,
            action="artifact.downloaded",
            subject=relative_path,
            metadata={"job_id": job_id, "bytes": size},
        )
        return {"path": relative_path, "bytes": size, "sha256": file_sha256(path), "text": content}

    def usage(self, principal: Principal, *, limit: int = 1000) -> list[dict[str, Any]]:
        principal.require("usage:read")
        self._record_request(principal, "usage.read")
        return self.store.list_usage(principal.tenant_id, limit=limit)

    def set_consent(
        self,
        principal: Principal,
        *,
        purpose: str,
        granted: bool,
        policy_version: str,
    ) -> None:
        principal.require("consent:write")
        self._record_request(principal, "consent.write")
        self.store.set_consent(
            principal.tenant_id,
            purpose=purpose,
            granted=granted,
            policy_version=policy_version,
        )

    def record_aggregate_observation(
        self,
        principal: Principal,
        *,
        purpose: str,
        metric_name: str,
        metric_value: float,
        context: dict[str, Any],
        source_receipt_id: str,
    ) -> str:
        principal.require("consent:write")
        self._record_request(principal, "aggregate.write")
        return self.store.record_aggregate_observation(
            principal.tenant_id,
            purpose=purpose,
            metric_name=metric_name,
            metric_value=metric_value,
            context=context,
            source_receipt_id=source_receipt_id,
        )

    def export_usage(self, tenant_id: str, output: Path) -> Path:
        records = list(reversed(self.store.list_usage(tenant_id, limit=10000)))
        content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        if output.exists() and output.read_text() != content:
            raise FileExistsError("refusing to overwrite divergent usage export")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        return output

    def delete_tenant(self, tenant_id: str, *, confirmation: str) -> None:
        if confirmation != tenant_id:
            raise ValueError("tenant deletion confirmation must exactly match tenant_id")
        root = self.tenant_root(tenant_id)
        self.store.delete_tenant_records(tenant_id)
        if root.exists():
            if root.is_symlink():
                raise ValueError("refusing to delete a symlinked tenant root")
            shutil.rmtree(root)

    def retention_candidates(self, tenant_id: str) -> list[Path]:
        tenant = self.store.get_tenant(tenant_id)
        cutoff = datetime.now(UTC) - timedelta(days=int(tenant["retention_days"]))
        candidates: list[Path] = []
        for job in self.store.list_retention_jobs(
            tenant_id, finished_before=cutoff, limit=1000
        ):
            path = self.resolve_tenant_path(
                tenant_id, Path("jobs") / job.job_id, must_exist=False
            )
            if path.exists():
                candidates.append(path)
        return sorted(candidates)

    def purge_retention(self, tenant_id: str, *, apply: bool = False) -> list[str]:
        """List or remove expired terminal-job artifacts for exactly one tenant."""
        root = self.tenant_root(tenant_id)
        candidates = self.retention_candidates(tenant_id)
        relative_paths = [path.relative_to(root).as_posix() for path in candidates]
        if not apply:
            return relative_paths
        for path, relative_path in zip(candidates, relative_paths, strict=True):
            if path.is_symlink() or path.parent != root / "jobs":
                raise ValueError("refusing unsafe retention target")
            bytes_removed = self._directory_size(path)
            shutil.rmtree(path)
            self.store.record_usage(
                tenant_id,
                operation="retention.purged",
                quantity=float(bytes_removed),
                unit="byte",
                metadata={"path": relative_path},
            )
            self.store.record_audit(
                tenant_id,
                action="retention.purged",
                subject=relative_path,
                metadata={"bytes_removed": bytes_removed},
            )
        return relative_paths
