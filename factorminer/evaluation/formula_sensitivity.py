"""Formula AST sensitivity / ablation analysis.

Proves interpretability of formulaic alphas by measuring how IC changes when
individual leaves, operator subtrees, or numeric parameters are perturbed.
Uses ``core/expression_tree.py`` tree-walk APIs read-only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from factorminer.core.expression_tree import (
    ConstantNode,
    ExpressionTree,
    LeafNode,
    Node,
    OperatorNode,
)
from factorminer.core.parser import try_parse
from factorminer.evaluation.metrics import (
    compute_ic,
    compute_ic_paper_icir,
    compute_ic_paper_mean,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FormulaSensitivityConfig:
    """Configuration for formula AST sensitivity analysis."""

    permute_seed: int = 42
    window_deltas: tuple[float, ...] = (-0.25, 0.25)
    min_window: float = 2.0
    max_operator_subtrees: int = 12
    include_explanation: bool = True


@dataclass
class AblationRow:
    """One row of a sensitivity / ablation table."""

    kind: str  # leaf | subtree | parameter
    target: str
    baseline_ic: float
    perturbed_ic: float
    delta_ic: float
    baseline_icir: float
    perturbed_icir: float
    delta_icir: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FormulaSensitivityResult:
    """Full sensitivity report for one formula."""

    formula: str
    factor_name: str
    baseline_ic: float
    baseline_icir: float
    leaf_ablations: list[AblationRow] = field(default_factory=list)
    subtree_ablations: list[AblationRow] = field(default_factory=list)
    parameter_sensitivity: list[AblationRow] = field(default_factory=list)
    explanation: str = ""
    explanation_source: str = "template"  # template | llm

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "factor_name": self.factor_name,
            "baseline_ic": self.baseline_ic,
            "baseline_icir": self.baseline_icir,
            "leaf_ablations": [row.to_dict() for row in self.leaf_ablations],
            "subtree_ablations": [row.to_dict() for row in self.subtree_ablations],
            "parameter_sensitivity": [row.to_dict() for row in self.parameter_sensitivity],
            "explanation": self.explanation,
            "explanation_source": self.explanation_source,
        }

    @property
    def all_rows(self) -> list[AblationRow]:
        return [
            *self.leaf_ablations,
            *self.subtree_ablations,
            *self.parameter_sensitivity,
        ]


def _ic_pair(signals: np.ndarray, returns: np.ndarray) -> tuple[float, float]:
    ic_series = compute_ic(signals, returns)
    return float(compute_ic_paper_mean(ic_series)), float(compute_ic_paper_icir(ic_series))


def _safe_eval(tree: ExpressionTree, data: Mapping[str, np.ndarray]) -> np.ndarray | None:
    try:
        out = tree.evaluate(dict(data))
    except Exception:  # noqa: BLE001 - ablations must fail closed
        logger.debug("sensitivity eval failed for %s", tree.to_string(), exc_info=True)
        return None
    if out is None or not isinstance(out, np.ndarray):
        return None
    if out.shape != next(iter(data.values())).shape:
        # Some operators can reduce shape unexpectedly; reject rather than crash.
        if out.ndim != 2:
            return None
    return np.asarray(out, dtype=np.float64)


def _clone_replace_leaves(node: Node, feature: str, mode: str, rng: np.random.RandomState) -> Node:
    """Deep-copy *node*, ablating every LeafNode matching *feature*."""
    if isinstance(node, LeafNode):
        if node.feature_name != feature:
            return node.clone()
        if mode == "zero":
            return ConstantNode(0.0)
        # permute: replaced at evaluate-time via data dict; keep leaf identity
        return node.clone()
    if isinstance(node, ConstantNode):
        return node.clone()
    assert isinstance(node, OperatorNode)
    return OperatorNode(
        operator=node.operator,
        children=[_clone_replace_leaves(child, feature, mode, rng) for child in node.children],
        params=dict(node.params),
    )


def _node_path_replace(
    node: Node,
    path: Sequence[int],
    replacement: Node,
) -> Node:
    """Return a clone of *node* with the subtree at *path* replaced."""
    if not path:
        return replacement.clone() if hasattr(replacement, "clone") else replacement
    if not isinstance(node, OperatorNode):
        return node.clone()
    idx, *rest = path
    children = []
    for i, child in enumerate(node.children):
        if i == idx:
            children.append(_node_path_replace(child, rest, replacement))
        else:
            children.append(child.clone())
    return OperatorNode(
        operator=node.operator,
        children=children,
        params=dict(node.params),
    )


def _collect_operator_paths(node: Node, prefix: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], OperatorNode]]:
    """Collect (path, operator_node) pairs for every operator subtree."""
    found: list[tuple[tuple[int, ...], OperatorNode]] = []
    if isinstance(node, OperatorNode):
        # Skip the root itself for leave-one-out (ablating whole formula is meaningless).
        if prefix:
            found.append((prefix, node))
        for i, child in enumerate(node.children):
            found.extend(_collect_operator_paths(child, prefix + (i,)))
    return found


def _collect_param_sites(
    node: Node, prefix: tuple[int, ...] = ()
) -> list[tuple[tuple[int, ...], str, float, tuple[float, float] | None]]:
    """Collect ``(path, pname, pval, param_range)`` sites for every operator param.

    ``param_range`` is the operator's own declared ``(lo, hi)`` bound for
    that parameter name (from ``OperatorSpec.param_ranges``), or ``None``
    if undeclared. Perturbation must clamp to this operator-specific
    range, not a single blanket floor shared by every parameter -- a
    window-length floor is meaningless (and destructive) applied to e.g.
    ``Quantile.q`` (a probability in ``[0, 1]``) or ``Power.exponent``.
    """
    sites: list[tuple[tuple[int, ...], str, float, tuple[float, float] | None]] = []
    if isinstance(node, OperatorNode):
        ranges = node.operator.param_ranges
        for pname, pval in node.params.items():
            try:
                sites.append((prefix, str(pname), float(pval), ranges.get(pname)))
            except (TypeError, ValueError):
                continue
        for i, child in enumerate(node.children):
            sites.extend(_collect_param_sites(child, prefix + (i,)))
    return sites


def _set_param_at_path(node: Node, path: Sequence[int], pname: str, value: float) -> Node:
    if not isinstance(node, OperatorNode):
        return node.clone()
    if not path:
        params = dict(node.params)
        params[pname] = float(value)
        return OperatorNode(
            operator=node.operator,
            children=[child.clone() for child in node.children],
            params=params,
        )
    idx, *rest = path
    children = []
    for i, child in enumerate(node.children):
        if i == idx:
            children.append(_set_param_at_path(child, rest, pname, value))
        else:
            children.append(child.clone())
    return OperatorNode(
        operator=node.operator,
        children=children,
        params=dict(node.params),
    )


def _template_explanation(result: FormulaSensitivityResult) -> str:
    rows = sorted(result.all_rows, key=lambda r: abs(r.delta_ic), reverse=True)
    if not rows:
        return (
            f"No successful ablations for `{result.formula}`; "
            "sensitivity could not isolate a dominant subexpression."
        )
    top = rows[0]
    direction = "reduces" if top.delta_ic < 0 else "increases"
    return (
        f"Baseline paper IC for `{result.factor_name or result.formula}` is "
        f"{result.baseline_ic:.4f}. The largest absolute IC change comes from "
        f"{top.kind} `{top.target}` (ΔIC={top.delta_ic:+.4f}), which {direction} "
        f"predictive strength when ablated/perturbed. "
        f"{top.detail or 'This subexpression appears to drive most of the measured signal.'}"
    )


def draft_sensitivity_explanation(
    result: FormulaSensitivityResult,
    *,
    llm_provider: Any | None = None,
    use_llm: bool = False,
) -> tuple[str, str]:
    """Return ``(explanation, source)`` with mock-safe template default."""
    template = _template_explanation(result)
    if not use_llm or llm_provider is None:
        return template, "template"

    top = sorted(result.all_rows, key=lambda r: abs(r.delta_ic), reverse=True)[:5]
    table = "\n".join(
        f"- {row.kind} {row.target}: delta_ic={row.delta_ic:+.4f}, detail={row.detail}"
        for row in top
    )
    system_prompt = (
        "You explain formulaic factor ablation tables in plain English for a "
        "research memo. Do not claim compliance, guaranteed alpha, or that the "
        "factor is validated. Return one short paragraph only."
    )
    user_prompt = (
        f"Formula: {result.formula}\n"
        f"Baseline IC: {result.baseline_ic:.4f}\n"
        f"Top ablations:\n{table}\n"
        "Explain which subexpression appears to drive the factor and why, "
        "in one paragraph."
    )
    try:
        raw = str(
            llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=250,
            )
        ).strip()
    except Exception:  # noqa: BLE001
        logger.warning("sensitivity explanation LLM draft failed", exc_info=True)
        return template, "template"
    if not raw or raw.lstrip().startswith("1."):
        # MockProvider returns factor lists; treat as unusable.
        return template, "template"
    return raw, "llm"


def analyze_formula_sensitivity(
    formula: str,
    data: Mapping[str, np.ndarray],
    returns: np.ndarray,
    *,
    factor_name: str = "",
    config: FormulaSensitivityConfig | None = None,
    llm_provider: Any | None = None,
    use_llm_explanation: bool = False,
) -> FormulaSensitivityResult:
    """Run leaf / subtree / parameter sensitivity for one formula.

    Parameters
    ----------
    formula:
        DSL formula string.
    data:
        Feature dict mapping ``$close``-style names to ``(M, T)`` arrays.
    returns:
        Forward-return panel ``(M, T)``.
    """
    cfg = config or FormulaSensitivityConfig()
    tree = try_parse(formula)
    if tree is None:
        return FormulaSensitivityResult(
            formula=formula,
            factor_name=factor_name,
            baseline_ic=float("nan"),
            baseline_icir=float("nan"),
            explanation="Formula failed to parse; sensitivity skipped.",
            explanation_source="template",
        )

    baseline_signals = _safe_eval(tree, data)
    if baseline_signals is None:
        return FormulaSensitivityResult(
            formula=formula,
            factor_name=factor_name,
            baseline_ic=float("nan"),
            baseline_icir=float("nan"),
            explanation="Baseline evaluation failed; sensitivity skipped.",
            explanation_source="template",
        )

    baseline_ic, baseline_icir = _ic_pair(baseline_signals, returns)
    rng = np.random.RandomState(cfg.permute_seed)
    result = FormulaSensitivityResult(
        formula=formula,
        factor_name=factor_name or formula,
        baseline_ic=baseline_ic,
        baseline_icir=baseline_icir,
    )

    # --- (a) leaf-feature ablation: zero and permute ---
    for feature in tree.leaf_features():
        # zero
        zero_root = _clone_replace_leaves(tree.root, feature, "zero", rng)
        zero_tree = ExpressionTree(zero_root)
        zero_signals = _safe_eval(zero_tree, data)
        if zero_signals is not None:
            z_ic, z_icir = _ic_pair(zero_signals, returns)
            result.leaf_ablations.append(
                AblationRow(
                    kind="leaf",
                    target=feature,
                    baseline_ic=baseline_ic,
                    perturbed_ic=z_ic,
                    delta_ic=z_ic - baseline_ic,
                    baseline_icir=baseline_icir,
                    perturbed_icir=z_icir,
                    delta_icir=z_icir - baseline_icir,
                    detail="zero ablation",
                )
            )

        # permute feature panel across assets within each time (breaks CS signal)
        if feature in data:
            perm_data = dict(data)
            arr = np.array(data[feature], dtype=np.float64, copy=True)
            for t in range(arr.shape[1]):
                rng.shuffle(arr[:, t])
            perm_data[feature] = arr
            perm_signals = _safe_eval(tree, perm_data)
            if perm_signals is not None:
                p_ic, p_icir = _ic_pair(perm_signals, returns)
                result.leaf_ablations.append(
                    AblationRow(
                        kind="leaf",
                        target=feature,
                        baseline_ic=baseline_ic,
                        perturbed_ic=p_ic,
                        delta_ic=p_ic - baseline_ic,
                        baseline_icir=baseline_icir,
                        perturbed_icir=p_icir,
                        delta_icir=p_icir - baseline_icir,
                        detail="cross-sectional permute ablation",
                    )
                )

    # --- (b) operator-subtree leave-one-out ---
    op_paths = _collect_operator_paths(tree.root)
    # Prefer larger / deeper subtrees first, then cap.
    op_paths.sort(key=lambda item: item[1].size(), reverse=True)
    for path, op_node in op_paths[: cfg.max_operator_subtrees]:
        replaced = ExpressionTree(
            _node_path_replace(tree.root, path, ConstantNode(0.0))
        )
        signals = _safe_eval(replaced, data)
        if signals is None:
            continue
        s_ic, s_icir = _ic_pair(signals, returns)
        label = op_node.to_string()
        if len(label) > 80:
            label = label[:77] + "..."
        result.subtree_ablations.append(
            AblationRow(
                kind="subtree",
                target=f"{op_node.operator.name}:{label}",
                baseline_ic=baseline_ic,
                perturbed_ic=s_ic,
                delta_ic=s_ic - baseline_ic,
                baseline_icir=baseline_icir,
                perturbed_icir=s_icir,
                delta_icir=s_icir - baseline_icir,
                detail=f"leave-one-out zero at path {list(path)}",
            )
        )

    # --- (c) window / parameter local sensitivity ---
    for path, pname, pval, prange in _collect_param_sites(tree.root):
        lo, hi = prange if prange is not None else (cfg.min_window, float("inf"))
        for delta in cfg.window_deltas:
            new_val = min(hi, max(lo, pval * (1.0 + delta)))
            if abs(new_val - pval) < 1e-12:
                step = (hi - lo) * 0.05 if hi > lo else 1.0
                step = step if delta >= 0 else -step
                new_val = min(hi, max(lo, pval + step))
            perturbed = ExpressionTree(_set_param_at_path(tree.root, path, pname, new_val))
            signals = _safe_eval(perturbed, data)
            if signals is None:
                continue
            p_ic, p_icir = _ic_pair(signals, returns)
            result.parameter_sensitivity.append(
                AblationRow(
                    kind="parameter",
                    target=f"{pname}@{list(path) or ['root']}",
                    baseline_ic=baseline_ic,
                    perturbed_ic=p_ic,
                    delta_ic=p_ic - baseline_ic,
                    baseline_icir=baseline_icir,
                    perturbed_icir=p_icir,
                    delta_icir=p_icir - baseline_icir,
                    detail=f"{pname}: {pval:g} -> {new_val:g} ({delta:+.0%})",
                )
            )

    if cfg.include_explanation:
        text, source = draft_sensitivity_explanation(
            result,
            llm_provider=llm_provider,
            use_llm=use_llm_explanation,
        )
        result.explanation = text
        result.explanation_source = source

    return result


def sensitivity_table(result: FormulaSensitivityResult) -> list[dict[str, Any]]:
    """Flatten a sensitivity result into report-friendly rows."""
    rows = [row.to_dict() for row in result.all_rows]
    rows.sort(key=lambda r: abs(float(r.get("delta_ic") or 0.0)), reverse=True)
    return rows
