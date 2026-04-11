"""Financial operators for factor expression evaluation.

Exports the central registry and all operator category modules.
"""

from factorminer.operators.auto_inventor import (
    OperatorInventor,
    ProposedOperator,
    ValidationResult,
)
from factorminer.operators.custom import (
    CustomOperator,
    CustomOperatorStore,
)
from factorminer.operators.gpu_backend import (
    DeviceManager,
    batch_execute,
    device_manager,
    to_numpy,
    to_tensor,
    torch_available,
)
from factorminer.operators.registry import (
    OPERATOR_REGISTRY,
    execute_operator,
    get_impl,
    get_operator,
    implemented_operators,
    list_operators,
)

__all__ = [
    # Registry
    "OPERATOR_REGISTRY",
    "execute_operator",
    "get_impl",
    "get_operator",
    "implemented_operators",
    "list_operators",
    # GPU
    "DeviceManager",
    "batch_execute",
    "device_manager",
    "to_numpy",
    "to_tensor",
    "torch_available",
    # Auto-inventor
    "OperatorInventor",
    "ProposedOperator",
    "ValidationResult",
    # Custom operators
    "CustomOperator",
    "CustomOperatorStore",
]
