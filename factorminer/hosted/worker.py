"""Crash-recoverable subprocess worker for allow-listed hosted jobs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from factorminer.hosted.models import JobKind, JobRecord, JobState
from factorminer.hosted.service import HostedPilotService

ALLOWED_PROVIDER_SECRET_ENV = frozenset({"ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"})


class HostedWorker:
    def __init__(
        self,
        service: HostedPilotService,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 30,
        provider_secret_env: tuple[str, ...] = (),
    ) -> None:
        self.service = service
        self.worker_id = worker_id or f"worker-{uuid.uuid4()}"
        self.lease_seconds = max(10, lease_seconds)
        unknown_secrets = set(provider_secret_env) - ALLOWED_PROVIDER_SECRET_ENV
        if unknown_secrets:
            raise ValueError(
                f"unsupported provider secret environment names: {sorted(unknown_secrets)}"
            )
        self.provider_secret_env = tuple(sorted(set(provider_secret_env)))

    def _job_directory(self, job: JobRecord) -> Path:
        path = self.service.resolve_tenant_path(
            job.tenant_id, Path("jobs") / job.job_id, must_exist=False
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _tenant_path(self, job: JobRecord, value: str) -> Path:
        return self.service.resolve_tenant_path(job.tenant_id, value, must_exist=True)

    def _command(self, job: JobRecord, job_dir: Path) -> list[str]:
        parameters = job.parameters
        prefix = [sys.executable, "-m", "factorminer.cli"]
        if job.kind == JobKind.VALIDATE_DATA:
            command = [
                *prefix,
                "validate-data",
                str(self._tenant_path(job, parameters["input_path"])),
                "--hdf-key",
                str(parameters["hdf_key"]),
                "--json",
            ]
            if parameters["strict"]:
                command.append("--strict")
            return command
        if job.kind == JobKind.MINE:
            output = job_dir / "output"
            output.mkdir(exist_ok=True)
            command_name = "helix" if parameters["loop"] == "helix" else "mine"
            return [
                *prefix,
                "-o",
                str(output),
                command_name,
                "--data",
                str(self._tenant_path(job, parameters["input_path"])),
                "--iterations",
                str(parameters["iterations"]),
                "--batch-size",
                str(parameters["batch_size"]),
                "--target",
                str(parameters["target"]),
            ]
        if job.kind == JobKind.BENCHMARK:
            output = job_dir / "output"
            output.mkdir(exist_ok=True)
            return [
                *prefix,
                "-o",
                str(output),
                "benchmark",
                str(parameters["mode"]),
                "--data",
                str(self._tenant_path(job, parameters["input_path"])),
            ]
        if job.kind == JobKind.GENERATE_REPORT:
            extension = "html" if parameters["format"] == "html" else "md"
            output = job_dir / "output" / f"report.{extension}"
            output.parent.mkdir(exist_ok=True)
            command = [
                *prefix,
                "report",
                str(self._tenant_path(job, parameters["library_path"])),
                "--format",
                str(parameters["format"]),
                "--output",
                str(output),
            ]
            if parameters.get("session_log"):
                command.extend(
                    ["--session-log", str(self._tenant_path(job, parameters["session_log"]))]
                )
            for benchmark in parameters.get("benchmarks", []):
                command.extend(["--benchmark", str(self._tenant_path(job, benchmark))])
            return command
        if job.kind == JobKind.VERIFY_RECEIPT:
            command = [
                *prefix,
                "verify-receipt",
                str(self._tenant_path(job, parameters["release_dir"])),
            ]
            if parameters.get("commitment_input"):
                command.extend(
                    [
                        "--commitment-input",
                        str(self._tenant_path(job, parameters["commitment_input"])),
                    ]
                )
            return command
        raise ValueError(f"unsupported hosted job kind: {job.kind.value}")

    @staticmethod
    def _terminate(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - Windows CI not used here
                process.terminate()
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover
                    process.kill()

    def execute(self, job: JobRecord) -> JobRecord:
        quota = self.service.quota_for(job.tenant_id)
        job_dir = self._job_directory(job)
        command = self._command(job, job_dir)
        (job_dir / "job.json").write_text(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "kind": job.kind.value,
                    "parameters": job.parameters,
                    "attempt": job.attempt,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        started = time.monotonic()
        next_heartbeat = started + self.lease_seconds / 3
        terminal_state = JobState.FAILED
        error: str | None = None
        returncode: int | None = None
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "PYTHONUNBUFFERED": "1",
        }
        for name in self.provider_secret_env:
            if value := os.environ.get(name):
                env[name] = value
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(  # noqa: S603 - argv is built from an allow-list
                command,
                cwd=self.service.tenant_root(job.tenant_id),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            while True:
                returncode = process.poll()
                if returncode is not None:
                    terminal_state = JobState.SUCCEEDED if returncode == 0 else JobState.FAILED
                    if returncode != 0:
                        error = f"subprocess exited with status {returncode}"
                    break
                current = time.monotonic()
                elapsed = current - started
                current_job = self.service.store.get_job(job.tenant_id, job.job_id)
                if current_job.cancel_requested:
                    terminal_state = JobState.CANCELED
                    error = "canceled by tenant"
                    self._terminate(process)
                    returncode = process.returncode
                    break
                if elapsed > job.timeout_seconds:
                    terminal_state = JobState.FAILED
                    error = f"job exceeded timeout of {job.timeout_seconds} seconds"
                    self._terminate(process)
                    returncode = process.returncode
                    break
                log_bytes = sum(
                    path.stat().st_size for path in (stdout_path, stderr_path) if path.exists()
                )
                if log_bytes > quota.max_log_bytes:
                    terminal_state = JobState.FAILED
                    error = "job exceeded log-size quota"
                    self._terminate(process)
                    returncode = process.returncode
                    break
                if self.service.tenant_storage_bytes(job.tenant_id) > quota.max_storage_bytes:
                    terminal_state = JobState.FAILED
                    error = "job exceeded tenant storage quota"
                    self._terminate(process)
                    returncode = process.returncode
                    break
                if current >= next_heartbeat:
                    if not self.service.store.heartbeat_job(
                        job.job_id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                    ):
                        self._terminate(process)
                        raise RuntimeError("worker lost the job lease")
                    next_heartbeat = current + self.lease_seconds / 3
                time.sleep(0.1)

        elapsed_seconds = max(0.0, time.monotonic() - started)
        try:
            self.service.assert_safe_artifact_tree(job.tenant_id, job.job_id)
        except ValueError as exc:
            terminal_state = JobState.FAILED
            error = str(exc)
        if self.service.tenant_storage_bytes(job.tenant_id) > quota.max_storage_bytes:
            terminal_state = JobState.FAILED
            error = "job exceeded tenant storage quota"
        self.service.store.record_usage(
            job.tenant_id,
            operation="job.compute",
            quantity=elapsed_seconds,
            unit="compute_second",
            metadata={"job_id": job.job_id, "kind": job.kind.value},
        )
        result = {
            "returncode": returncode,
            "elapsed_seconds": elapsed_seconds,
            "stdout": (Path("jobs") / job.job_id / "stdout.log").as_posix(),
            "stderr": (Path("jobs") / job.job_id / "stderr.log").as_posix(),
            "output": (Path("jobs") / job.job_id / "output").as_posix(),
        }
        return self.service.store.finish_job(
            job.job_id,
            worker_id=self.worker_id,
            state=terminal_state,
            result=result,
            error=error,
        )

    def run_once(self) -> JobRecord | None:
        self.service.store.recover_expired_leases()
        job = self.service.store.claim_job(
            worker_id=self.worker_id, lease_seconds=self.lease_seconds
        )
        if job is None:
            return None
        try:
            return self.execute(job)
        except Exception as exc:
            try:
                return self.service.store.finish_job(
                    job.job_id,
                    worker_id=self.worker_id,
                    state=JobState.FAILED,
                    error=f"worker failure: {type(exc).__name__}: {exc}",
                )
            except KeyError:
                raise exc

    def run_forever(self, *, poll_seconds: float = 1.0) -> None:
        while True:
            if self.run_once() is None:
                time.sleep(max(0.1, poll_seconds))
