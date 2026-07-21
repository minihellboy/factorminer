"""Operator registry and execution APIs loaded only when requested."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_EXPORTS = {
    **{
        name: "factorminer.operators.registry"
        for name in (
            "OPERATOR_REGISTRY",
            "execute_operator",
            "get_impl",
            "get_operator",
            "implemented_operators",
            "list_operators",
        )
    },
    **{
        name: "factorminer.operators.gpu_backend"
        for name in (
            "DeviceManager",
            "batch_execute",
            "device_manager",
            "to_numpy",
            "to_tensor",
            "torch_available",
        )
    },
    **{
        name: "factorminer.operators.auto_inventor"
        for name in ("OperatorInventor", "ProposedOperator", "ValidationResult")
    },
    "CustomOperator": "factorminer.operators.custom",
    "CustomOperatorStore": "factorminer.operators.custom",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
