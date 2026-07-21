"""Small helper for stable package APIs without eager dependency loading."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

type ExportTarget = str | tuple[str, str]


def resolve_export(
    package: ModuleType | dict[str, object],
    name: str,
    exports: dict[str, ExportTarget],
) -> object:
    """Resolve and cache a lazily declared package attribute.

    A string target means the source attribute has the same name.  A tuple is
    used for aliases such as ``CoreMiningConfig``.
    """
    try:
        target = exports[name]
    except KeyError as exc:
        package_name = (
            str(package.get("__name__", ""))
            if isinstance(package, dict)
            else package.__name__
        )
        raise AttributeError(
            f"module {package_name!r} has no attribute {name!r}"
        ) from exc

    module_name, attribute_name = target if isinstance(target, tuple) else (target, name)
    value = getattr(import_module(module_name), attribute_name)
    namespace = package if isinstance(package, dict) else vars(package)
    namespace[name] = value
    return value


def public_dir(namespace: dict[str, object], exports: dict[str, ExportTarget]) -> list[str]:
    """Return deterministic names for interactive discovery and documentation."""
    return sorted(set(namespace) | set(exports))
