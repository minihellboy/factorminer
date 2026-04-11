"""Tests for auto-operator invention (operators/auto_inventor.py)."""

from __future__ import annotations

import numpy as np

from factorminer.operators.auto_inventor import (
    OperatorInventor,
    ProposedOperator,
)
from factorminer.operators.custom import CustomOperatorStore

# -----------------------------------------------------------------------
# _compile_safely: valid numpy code -> callable
# -----------------------------------------------------------------------

def test_compile_safely_valid_code():
    """Valid numpy code defining compute() should return a callable."""
    code = "def compute(x):\n    return np.nanmean(x, axis=1, keepdims=True) * np.ones_like(x)"

    # Use OperatorInventor._compile_safely as a static-like test
    data = np.random.default_rng(42).normal(0, 1, (10, 50))
    inventor = OperatorInventor(
        llm_provider=_mock_provider(),
        data_tensor=data.reshape(10, 50, 1),
        returns=data,
    )
    fn = inventor._compile_safely(code)
    assert fn is not None
    assert callable(fn)
    result = fn(data)
    assert isinstance(result, np.ndarray)


# -----------------------------------------------------------------------
# _compile_safely: os.system -> returns None (SECURITY)
# -----------------------------------------------------------------------

def test_compile_safely_blocks_os_system():
    """Code containing os.system should be blocked."""
    code = "import os\ndef compute(x):\n    os.system('echo hacked')\n    return x"
    inventor = _make_inventor()
    fn = inventor._compile_safely(code)
    assert fn is None


# -----------------------------------------------------------------------
# _compile_safely: import os -> returns None (SECURITY)
# -----------------------------------------------------------------------

def test_compile_safely_blocks_import_os():
    """Code with 'import ' token should be blocked."""
    code = "import os\ndef compute(x):\n    return x"
    inventor = _make_inventor()
    fn = inventor._compile_safely(code)
    assert fn is None


def test_compile_safely_blocks_eval():
    """Code with eval() should be blocked."""
    code = "def compute(x):\n    return eval('x + 1')"
    inventor = _make_inventor()
    fn = inventor._compile_safely(code)
    assert fn is None


# -----------------------------------------------------------------------
# CustomOperatorStore: register and list
# -----------------------------------------------------------------------

def test_custom_operator_store_register_and_list(tmp_path):
    store = CustomOperatorStore(store_dir=str(tmp_path / "ops"))

    from factorminer.core.types import OperatorSpec, OperatorType, SignatureType
    spec = OperatorSpec(
        name="TestOp",
        arity=1,
        category=OperatorType.AUTO_INVENTED,
        signature=SignatureType.ELEMENT_WISE,
        description="test operator",
    )
    from factorminer.operators.custom import CustomOperator
    op = CustomOperator(
        name="TestOp",
        spec=spec,
        numpy_code="def compute(x): return x * 2",
        numpy_fn=lambda x: x * 2,
        validation_ic=0.05,
    )
    store.register(op)
    assert "TestOp" in store.list_operators()
    assert store.get_operator("TestOp") is not None


# -----------------------------------------------------------------------
# ProposedOperator dataclass
# -----------------------------------------------------------------------

def test_proposed_operator_dataclass():
    op = ProposedOperator(
        name="TestOp",
        arity=1,
        description="A test operator",
        numpy_code="def compute(x): return x",
    )
    assert op.name == "TestOp"
    assert op.arity == 1
    assert op.param_names == ()
    assert op.based_on == []


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _mock_provider():
    from factorminer.agent.llm_interface import MockProvider
    return MockProvider()


def _make_inventor():
    data = np.random.default_rng(42).normal(0, 1, (10, 50))
    return OperatorInventor(
        llm_provider=_mock_provider(),
        data_tensor=data.reshape(10, 50, 1),
        returns=data,
    )
