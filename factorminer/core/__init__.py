"""Core DSL, factor model, session, and loop APIs with lazy exports."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_EXPORTS = {
    "FormulaCanonicalizer": "factorminer.core.canonicalizer",
    "CoreMiningConfig": ("factorminer.core.config", "MiningConfig"),
    **{
        name: "factorminer.core.expression_tree"
        for name in ("Node", "LeafNode", "ConstantNode", "OperatorNode", "ExpressionTree")
    },
    "Factor": "factorminer.core.factor_library",
    "FactorLibrary": "factorminer.core.factor_library",
    **{
        name: "factorminer.core.library_io"
        for name in (
            "save_library",
            "load_library",
            "export_csv",
            "export_formulas",
            "import_from_paper",
        )
    },
    "parse": "factorminer.core.parser",
    "try_parse": "factorminer.core.parser",
    "MiningSession": "factorminer.core.session",
    "RalphLoop": "factorminer.core.ralph_loop",
    "HelixLoop": "factorminer.core.helix_loop",
    **{
        name: "factorminer.core.types"
        for name in (
            "OperatorSpec",
            "OperatorType",
            "SignatureType",
            "DEFAULT_FEATURES",
            "FEATURES",
            "FEATURE_SET",
            "FEATURE_DESCRIPTIONS",
            "OPERATOR_REGISTRY",
            "get_operator",
            "get_features",
            "get_feature_set",
            "get_feature_descriptions",
            "normalize_feature_name",
            "feature_to_column",
            "column_to_feature",
            "register_features",
            "unregister_features",
            "reset_features",
        )
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
