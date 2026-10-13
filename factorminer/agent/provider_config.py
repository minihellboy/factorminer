"""Provider roles, credential resolution, and typed model-call failures.

A mining run can call two models: the *primary* (frontier) model and an
optional *draft* model used by the cheap-first cascade. Each role resolves
its own credential, so a draft never inherits the primary's explicit key and
a hosted draft never receives a local placeholder key.

:class:`GuardedProvider` wraps a provider so every failed call raises a
:class:`ProviderCallError` that names the role, provider, model, and failure
kind, and keeps a bounded log of recent failures for run manifests.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from factorminer.agent.llm_interface import LLMProvider

PRIMARY = "primary"
DRAFT = "draft"

HOSTED_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
LOCAL_PROVIDERS = frozenset({"openai_compatible", "local"})
DRAFT_KEY_ENV = "FACTORMINER_DRAFT_API_KEY"


@dataclass(frozen=True)
class ProviderCredential:
    """Where one role's API key came from; the key itself is never serialized."""

    role: str
    provider: str
    source: str  # "config", "env:<NAME>", "local", "none"
    api_key: str = ""

    @property
    def fingerprint(self) -> str:
        if not self.api_key or self.source in ("local", "none"):
            return ""
        return hashlib.sha256(self.api_key.encode()).hexdigest()[:12]

    def describe(self) -> dict[str, str]:
        return {
            "role": self.role,
            "provider": self.provider,
            "source": self.source,
            "fingerprint": self.fingerprint,
        }


def resolve_credential(role: str, provider: str, config: dict[str, Any]) -> ProviderCredential:
    """Resolve the credential for ``role`` from its own config block and environment.

    Primary: ``api_key`` in config, else the provider's standard env var.
    Draft: ``draft_api_key`` in the cascade block, else ``FACTORMINER_DRAFT_API_KEY``,
    else the provider's standard env var for hosted providers. Local engines
    use an explicit local key or the ``"local"`` placeholder and never read
    hosted-provider env vars.
    """
    if provider == "mock":
        return ProviderCredential(role, provider, "none")
    explicit_key = "api_key" if role == PRIMARY else "draft_api_key"
    explicit = config.get(explicit_key)
    if provider in LOCAL_PROVIDERS:
        key = explicit or config.get("local_api_key")
        return ProviderCredential(role, provider, "config" if key else "local", key or "local")
    if explicit:
        return ProviderCredential(role, provider, "config", str(explicit))
    env_names = ([DRAFT_KEY_ENV] if role == DRAFT else []) + [HOSTED_KEY_ENV.get(provider, "")]
    for name in env_names:
        if name and os.environ.get(name):
            return ProviderCredential(role, provider, f"env:{name}", os.environ[name])
    return ProviderCredential(role, provider, "none")


# ---------------------------------------------------------------------------
# Typed call failures
# ---------------------------------------------------------------------------

ERROR_KINDS = (
    "auth", "rate_limit", "timeout", "connection", "bad_request", "server",
    "missing_dependency", "truncated", "unknown",
)
_RETRYABLE = frozenset({"rate_limit", "timeout", "connection", "server"})
_NAME_KINDS = (
    ("truncatedresponse", "truncated"), ("importerror", "missing_dependency"), ("modulenotfound", "missing_dependency"),
    ("authentication", "auth"), ("permissiondenied", "auth"), ("missingapikey", "auth"),
    ("unauthenticated", "auth"), ("ratelimit", "rate_limit"), ("resourceexhausted", "rate_limit"),
    ("timeout", "timeout"), ("deadlineexceeded", "timeout"), ("connection", "connection"),
    ("badrequest", "bad_request"), ("invalidargument", "bad_request"),
    ("unprocessable", "bad_request"), ("notfound", "bad_request"),
    ("internalserver", "server"), ("serviceunavailable", "server"), ("overloaded", "server"),
)


def classify_provider_error(exc: BaseException) -> str:
    """Map an SDK exception onto a stable failure kind without importing SDKs."""
    for klass in type(exc).__mro__:
        name = klass.__name__.lower().replace("_", "")
        for needle, kind in _NAME_KINDS:
            if needle in name:
                return kind
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        if status in (401, 403):
            return "auth"
        if status == 429:
            return "rate_limit"
        if status in (408, 504):
            return "timeout"
        if 400 <= status < 500:
            return "bad_request"
        if status >= 500:
            return "server"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection"
    return "unknown"


class ProviderCallError(RuntimeError):
    """A model call failed; carries the role, provider, model, and failure kind."""

    def __init__(
        self,
        *,
        role: str,
        provider: str,
        model: str,
        kind: str,
        message: str,
    ) -> None:
        self.role = role
        self.provider = provider
        self.model = model
        self.kind = kind
        self.retryable = kind in _RETRYABLE
        self.message = message
        super().__init__(f"{role} model call failed ({provider}/{model}, {kind}): {message}")


@dataclass(frozen=True)
class CallFailure:
    """Serializable record of one failed call."""

    role: str
    provider: str
    model: str
    kind: str
    retryable: bool
    message: str
    at: float


class GuardedProvider(LLMProvider):
    """Wrap a provider so failures are typed, attributed to a role, and recorded."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        role: str,
        credential: ProviderCredential | None = None,
        history: int = 50,
    ) -> None:
        self._inner = inner
        self.role = role
        self.credential = credential
        self.calls = 0
        self.failures: deque[CallFailure] = deque(maxlen=history)
        self.failure_count = 0
        self._lock = threading.Lock()

    @property
    def inner(self) -> LLMProvider:
        return self._inner

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "") or "")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        *,
        cacheable_prefix: str | None = None,
    ) -> str:
        with self._lock:
            self.calls += 1
        try:
            text = self._inner.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                cacheable_prefix=cacheable_prefix,
            )
        except ProviderCallError as exc:
            error = self._redact_error(exc)
            self._record(error)
            if error is exc:
                raise
            raise error from exc
        except Exception as exc:  # noqa: BLE001 - converted to a typed error
            error = ProviderCallError(
                role=self.role,
                provider=self.provider_name,
                model=self.model,
                kind=classify_provider_error(exc),
                message=self._redact(f"{type(exc).__name__}: {exc}"),
            )
            self._record(error)
            raise error from exc
        return text

    def _redact(self, message: str) -> str:
        key = self.credential.api_key if self.credential else ""
        return message.replace(key, "[REDACTED]") if key and key != "local" else message

    def _redact_error(self, error: ProviderCallError) -> ProviderCallError:
        message = self._redact(error.message)
        if message == error.message:
            return error
        return ProviderCallError(
            role=error.role, provider=error.provider, model=error.model,
            kind=error.kind, message=message,
        )

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the wrapper lacks (model settings,
        # request builders, counters); delegate so callers keep working.
        if name.startswith("__") or name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _record(self, error: ProviderCallError) -> None:
        with self._lock:
            self.failure_count += 1
            self.failures.append(CallFailure(
                role=error.role, provider=error.provider, model=error.model, kind=error.kind,
                retryable=error.retryable, message=error.message[:500], at=time.time(),
            ))

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "role": self.role,
                "provider": self.provider_name,
                "model": self.model,
                "credential": self.credential.describe() if self.credential else None,
                "calls": self.calls,
                "failures": self.failure_count,
                "recent_failures": [asdict(failure) for failure in self.failures],
            }


def describe_providers(provider: Any) -> list[dict[str, Any]]:
    """Describe every guarded provider reachable through known wrappers."""
    found: list[dict[str, Any]] = []
    seen: set[int] = set()
    stack = [provider]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, GuardedProvider):
            found.append(current.describe())
        for attribute in ("inner", "frontier", "draft", "llm_provider"):
            stack.append(getattr(current, attribute, None))
    return sorted(found, key=lambda item: item["role"])
