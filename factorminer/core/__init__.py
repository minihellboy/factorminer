"""FactorMiner core: expression trees, types, factor DSL parser, and Ralph Loop."""

from importlib import import_module

from factorminer.core.canonicalizer import FormulaCanonicalizer
from factorminer.core.config import MiningConfig as CoreMiningConfig
from factorminer.core.expression_tree import (
    ConstantNode,
    ExpressionTree,
    LeafNode,
    Node,
    OperatorNode,
)
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import (
    export_csv,
    export_formulas,
    import_from_paper,
    load_library,
    save_library,
)
from factorminer.core.parser import parse, try_parse
from factorminer.core.session import MiningSession
from factorminer.core.types import (
    DEFAULT_FEATURES,
    FEATURE_DESCRIPTIONS,
    FEATURE_SET,
    FEATURES,
    OPERATOR_REGISTRY,
    OperatorSpec,
    OperatorType,
    SignatureType,
    column_to_feature,
    feature_to_column,
    get_feature_descriptions,
    get_feature_set,
    get_features,
    get_operator,
    normalize_feature_name,
    register_features,
    reset_features,
    unregister_features,
)

_LAZY_EXPORTS = {
    "RalphLoop": ("factorminer.core.ralph_loop", "RalphLoop"),
    "HelixLoop": ("factorminer.core.helix_loop", "HelixLoop"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    # Expression tree
    "Node",
    "LeafNode",
    "ConstantNode",
    "OperatorNode",
    "ExpressionTree",
    # Factor library
    "Factor",
    "FactorLibrary",
    "save_library",
    "load_library",
    "export_csv",
    "export_formulas",
    "import_from_paper",
    # Parser
    "parse",
    "try_parse",
    # Loops
    "RalphLoop",
    "HelixLoop",
    "MiningSession",
    "CoreMiningConfig",
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
    # Canonicalizer
    "FormulaCanonicalizer",
]
