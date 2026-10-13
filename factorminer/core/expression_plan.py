"""Compiled, immutable execution plans for expression trees.

An :class:`ExpressionPlan` flattens one formula into a topologically ordered,
deduplicated list of steps. A :class:`BatchPlan` merges several formulas so a
subexpression shared by any of them is evaluated once per batch, then released
after its last consumer.

Each step runs exactly the code the recursive tree evaluator runs: leaves and
constants call their node's ``evaluate`` and operators call the same NumPy
dispatch with the same parameters. Plan outputs are therefore bit-identical to
``ExpressionTree.evaluate``, including NaNs, ties, and warm-up periods.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from factorminer.core.expression_tree import (
    ConstantNode,
    ExpressionTree,
    LeafNode,
    OperatorNode,
    _dispatch_operator,
)
from factorminer.core.types import OperatorSpec, SignatureType

# Bump when any operator's numerical semantics change; it is part of every
# formula digest so cached signals never outlive the semantics that made them.
# v2: missing (NaN) inputs never produce values -- NaN denominators, empty
# windows and missing latest values give NaN, and EMA-type recurrences skip
# gaps instead of turning NaN for the rest of the series.
OPERATOR_SEMANTICS_VERSION = "numpy-expression-v2"

# Operators whose output at t depends on inputs t-(window-1)..t.
_ROLLING_WINDOW_OPERATORS = frozenset(
    {
        "Mean", "Std", "Var", "Skew", "Kurt", "Median", "Med", "Sum", "Prod",
        "Product", "TsMax", "TsMin", "TsArgMax", "TsArgMin", "TsRank", "Quantile",
        "CountNaN", "CountNotNaN", "Corr", "Cov", "Beta", "Resid", "SMA", "WMA",
        "Decay", "TsDecay", "TsLinReg", "TsLinRegSlope", "Slope",
        "TsLinRegIntercept", "TsLinRegResid", "Resi", "Rsquare",
    }
)
# Operators whose output at t depends on inputs t-window and t.
_LAG_OPERATORS = frozenset({"Delta", "Delay", "Return", "LogReturn"})
# Operators with unbounded memory: output at t depends on all prior inputs.
_UNBOUNDED_OPERATORS = frozenset(
    {"EMA", "DEMA", "KAMA", "CumSum", "CumProd", "CumMax", "CumMin"}
)


class PlanExecutor(Protocol):
    """Runs plan steps on another array backend (see ``operators.torch_executor``).

    Values passed between steps are the executor's own arrays; ``to_numpy``
    converts a formula's output back to a NumPy panel the caller owns.
    """

    backend: str

    def leaf(self, name: str, data: Mapping[str, np.ndarray]) -> Any: ...

    def constant(self, value: float, data: Mapping[str, np.ndarray]) -> Any: ...

    def operator(self, spec: OperatorSpec, inputs: list[Any], params: dict[str, float]) -> Any: ...

    def from_numpy(self, array: np.ndarray) -> Any: ...

    def to_numpy(self, value: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class PlanStep:
    """One deduplicated node in a compiled plan.

    ``lookback`` is the number of prior periods whose inputs can affect the
    output at a period, or ``None`` when the dependence is unbounded or unknown.
    """

    digest: str
    kind: str  # "leaf" | "constant" | "operator" | "opaque"
    inputs: tuple[int, ...]
    lookback: int | None
    cross_sectional: bool
    node: Any = None  # Only opaque nodes retain a mutable evaluator.
    operator: OperatorSpec | None = None
    params: tuple[tuple[str, float], ...] = ()
    leaf_name: str = ""
    constant: float = 0.0

    def run(
        self,
        data: Mapping[str, np.ndarray],
        inputs: list[Any],
        executor: PlanExecutor | None = None,
    ) -> Any:
        if executor is not None and self.kind != "opaque":
            if self.kind == "operator":
                assert self.operator is not None
                return executor.operator(self.operator, inputs, dict(self.params))
            if self.kind == "leaf":
                return executor.leaf(self.leaf_name, data)
            return executor.constant(self.constant, data)
        if executor is not None:
            # Opaque nodes evaluate on host panels; hand the result to the executor.
            return executor.from_numpy(self.node.evaluate(data))
        if self.kind == "operator":
            assert self.operator is not None
            return _dispatch_operator(self.operator, inputs, dict(self.params))
        if self.kind == "leaf":
            if self.leaf_name not in data:
                raise KeyError(
                    f"Feature '{self.leaf_name}' not found in data. "
                    f"Available: {sorted(data.keys())}"
                )
            return data[self.leaf_name].astype(np.float64, copy=False)
        if self.kind == "constant":
            for panel in data.values():
                return np.full_like(panel, self.constant, dtype=np.float64)
            raise ValueError("Cannot evaluate ConstantNode with empty data dict.")
        result: np.ndarray = self.node.evaluate(data)
        return result


def _hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def _operator_lookback(node: OperatorNode, child_lookbacks: Sequence[int | None]) -> int | None:
    if any(lookback is None for lookback in child_lookbacks):
        return None
    base = max((int(lookback) for lookback in child_lookbacks), default=0)  # type: ignore[arg-type]
    name = node.operator.name
    window = int(node.params.get("window", 0))
    if name in _ROLLING_WINDOW_OPERATORS:
        return base + max(window - 1, 0)
    if name in _LAG_OPERATORS:
        return base + max(window, 0)
    if name == "HMA":
        sqrt_window = max(int(np.sqrt(window)), 1)
        return base + max(window - 1, 0) + sqrt_window - 1
    if name in _UNBOUNDED_OPERATORS:
        return None
    if node.operator.signature in (
        SignatureType.ELEMENT_WISE,
        SignatureType.CROSS_SECTION_TO_CROSS_SECTION,
    ):
        return base
    return None


class _Compiler:
    """Accumulate deduplicated steps for one or more trees."""

    def __init__(self) -> None:
        self.steps: list[PlanStep] = []
        self.index: dict[str, int] = {}

    def add(self, node: Any) -> int:
        if isinstance(node, LeafNode):
            return self._intern(
                _hash("leaf", node.feature_name),
                lambda digest: PlanStep(
                    digest, "leaf", (), 0, False, leaf_name=node.feature_name
                ),
            )
        if isinstance(node, ConstantNode):
            return self._intern(
                _hash("constant", float(node.value).hex()),
                lambda digest: PlanStep(
                    digest, "constant", (), 0, False, constant=float(node.value)
                ),
            )
        if isinstance(node, OperatorNode):
            inputs = tuple(self.add(child) for child in node.children)
            params = tuple(sorted((str(k), float(v)) for k, v in node.params.items()))
            digest = _hash(
                "operator",
                node.operator.name,
                *(f"{name}={value.hex()}" for name, value in params),
                *(self.steps[i].digest for i in inputs),
            )
            lookback = _operator_lookback(node, [self.steps[i].lookback for i in inputs])
            cross_sectional = (
                node.operator.signature is SignatureType.CROSS_SECTION_TO_CROSS_SECTION
                or any(self.steps[i].cross_sectional for i in inputs)
            )
            return self._intern(
                digest,
                lambda digest: PlanStep(
                    digest, "operator", inputs, lookback, cross_sectional,
                    operator=node.operator, params=params,
                ),
            )
        # Duck-typed nodes (e.g. neural leaves) are opaque: evaluated as a unit,
        # never shared, and treated as needing full history and every asset.
        return self._intern(
            _hash("opaque", str(id(node))),
            lambda digest: PlanStep(digest, "opaque", (), None, True, node),
        )

    def _intern(self, digest: str, build: Any) -> int:
        existing = self.index.get(digest)
        if existing is not None:
            return existing
        self.steps.append(build(digest))
        self.index[digest] = len(self.steps) - 1
        return len(self.steps) - 1


def _root(tree: ExpressionTree | Any) -> Any:
    return tree.root if isinstance(tree, ExpressionTree) else tree


@dataclass(frozen=True)
class ExpressionPlan:
    """Immutable compiled form of one formula."""

    formula: str
    steps: tuple[PlanStep, ...]
    output: int
    required_features: frozenset[str]
    max_lookback: int | None
    cross_sectional: bool
    digest: str

    @property
    def operator_order(self) -> tuple[str, ...]:
        return tuple(step.operator.name for step in self.steps if step.operator is not None)

    def evaluate(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate the formula; identical to ``ExpressionTree.evaluate``."""
        batch = BatchPlan(plans=(self,), steps=self.steps, outputs=(self.output,))
        (_, value), = batch.iter_outputs(data)
        if isinstance(value, BaseException):
            raise value
        result: np.ndarray = value
        return result


def compile_tree(tree: ExpressionTree | Any) -> ExpressionPlan:
    """Compile one expression tree (or root node) into an :class:`ExpressionPlan`."""
    root = _root(tree)
    compiler = _Compiler()
    output = compiler.add(root)
    return _plan_from_steps(root, tuple(compiler.steps), output)


def _plan_from_steps(root: Any, steps: tuple[PlanStep, ...], output: int) -> ExpressionPlan:
    reachable: list[int] = []
    seen: set[int] = set()
    stack = [output]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        reachable.append(index)
        stack.extend(steps[index].inputs)
    ordered = sorted(reachable)
    remap = {old: new for new, old in enumerate(ordered)}
    local = tuple(
        PlanStep(
            step.digest, step.kind, tuple(remap[i] for i in step.inputs), step.lookback,
            step.cross_sectional, step.node, step.operator, step.params,
            step.leaf_name, step.constant,
        )
        for step in (steps[i] for i in ordered)
    )
    out = local[remap[output]]
    return ExpressionPlan(
        formula=root.to_string(),
        steps=local,
        output=remap[output],
        required_features=frozenset(
            step.leaf_name for step in local if step.kind == "leaf"
        ),
        max_lookback=out.lookback,
        cross_sectional=out.cross_sectional,
        digest=_hash(OPERATOR_SEMANTICS_VERSION, out.digest),
    )


@dataclass(frozen=True)
class BatchPlan:
    """Several formulas compiled into one shared, deduplicated step list."""

    plans: tuple[ExpressionPlan, ...]
    steps: tuple[PlanStep, ...]
    outputs: tuple[int, ...]

    @property
    def required_features(self) -> frozenset[str]:
        return frozenset().union(*(plan.required_features for plan in self.plans))

    @property
    def reused_steps(self) -> int:
        """Node evaluations saved relative to evaluating each formula separately."""
        return sum(len(plan.steps) for plan in self.plans) - len(self.steps)

    def iter_outputs(
        self,
        data: Mapping[str, np.ndarray],
        executor: PlanExecutor | None = None,
    ) -> Iterator[tuple[int, np.ndarray | BaseException]]:
        """Yield ``(formula_index, signals_or_exception)`` in formula order.

        Intermediate arrays are dropped after their last consumer, so peak
        memory is bounded by live shared subexpressions rather than batch size.
        A step failure is propagated to every dependent formula, which receives
        the same exception a recursive evaluation would have raised. With an
        *executor*, steps run on its backend and outputs come back as NumPy.
        """
        remaining = [0] * len(self.steps)
        for step in self.steps:
            for index in step.inputs:
                remaining[index] += 1
        for output in self.outputs:
            remaining[output] += 1

        values: dict[int, np.ndarray | BaseException] = {}
        cursor = 0
        for formula_index, output in enumerate(self.outputs):
            while cursor <= output:
                if cursor not in values:
                    values[cursor] = self._run_step(cursor, data, values, executor)
                    for index in self.steps[cursor].inputs:
                        self._release(index, remaining, values)
                cursor += 1
            value = values[output]
            remaining[output] -= 1
            if remaining[output] == 0:
                del values[output]
            if executor is not None and not isinstance(value, BaseException):
                try:
                    value = executor.to_numpy(value)
                except Exception as exc:  # noqa: BLE001 - reported per formula
                    value = exc
            elif remaining[output] > 0 and isinstance(value, np.ndarray):
                # Still needed by a later formula: hand out a private copy.
                value = value.copy()
            yield formula_index, value

    def _run_step(
        self,
        index: int,
        data: Mapping[str, np.ndarray],
        values: dict[int, Any],
        executor: PlanExecutor | None = None,
    ) -> Any:
        step = self.steps[index]
        inputs = [values[i] for i in step.inputs]
        for value in inputs:
            if isinstance(value, BaseException):
                return value
        try:
            return step.run(data, inputs, executor)
        except Exception as exc:  # noqa: BLE001 - propagated to dependents
            return exc

    @staticmethod
    def _release(
        index: int, remaining: list[int], values: dict[int, np.ndarray | BaseException]
    ) -> None:
        remaining[index] -= 1
        if remaining[index] == 0:
            values.pop(index, None)


def compile_batch(trees: Sequence[ExpressionTree | Any]) -> BatchPlan:
    """Compile several trees into one plan that evaluates shared nodes once."""
    compiler = _Compiler()
    roots = [_root(tree) for tree in trees]
    outputs = tuple(compiler.add(root) for root in roots)
    steps = tuple(compiler.steps)
    plans = tuple(
        _plan_from_steps(root, steps, output) for root, output in zip(roots, outputs, strict=True)
    )
    return BatchPlan(plans=plans, steps=steps, outputs=outputs)
