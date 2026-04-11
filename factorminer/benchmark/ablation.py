"""Runtime ablation study for HelixFactor Phase 2 components.

This module now drives ablations through the real loop path:
- HelixLoop execution on a training slice
- runtime recomputation of the admitted library
- freeze/top-k selection and combo evaluation on a held-out slice
- optional memory suppression via temporary monkeypatching

Supported ablations:
  full             - all components enabled
  no_debate        - disable specialist debate
  no_causal        - disable causal validation
  no_canonicalize  - disable SymPy deduplication
  no_regime        - disable regime-aware evaluation
  no_online_memory - disable memory retrieval / formation / evolution hooks
  no_capacity      - disable capacity estimation
  no_significance  - disable significance filtering
  no_memory        - disable memory-guided generation and updates
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

import factorminer.core.helix_loop as helix_loop_module
import factorminer.core.ralph_loop as ralph_loop_module
from factorminer.agent.debate import DebateConfig as RuntimeDebateConfig
from factorminer.agent.llm_interface import MockProvider
from factorminer.benchmark.helix_benchmark import AblationResult, MethodResult
from factorminer.benchmark.runtime import (
    build_benchmark_library,
    evaluate_frozen_set,
    select_frozen_top_k,
)
from factorminer.core.config import MiningConfig
from factorminer.core.factor_library import FactorLibrary
from factorminer.core.helix_loop import HelixLoop
from factorminer.evaluation.capacity import CapacityConfig as RuntimeCapacityConfig
from factorminer.evaluation.causal import CausalConfig as RuntimeCausalConfig
from factorminer.evaluation.regime import RegimeConfig as RuntimeRegimeConfig
from factorminer.evaluation.runtime import (
    DatasetSplit,
    EvaluationDataset,
    evaluate_factors,
)
from factorminer.evaluation.significance import (
    SignificanceConfig as RuntimeSignificanceConfig,
)
from factorminer.memory.memory_store import ExperienceMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation configuration registry
# ---------------------------------------------------------------------------

_FULL_CFG = {
    "debate": True,
    "causal": True,
    "canonicalize": True,
    "regime": True,
    "online_memory": True,
    "capacity": True,
    "significance": True,
    "memory": True,
}

ABLATION_CONFIGS: dict[str, dict[str, bool]] = {
    "full": dict(_FULL_CFG),
    "no_debate": {**_FULL_CFG, "debate": False},
    "no_causal": {**_FULL_CFG, "causal": False},
    "no_canonicalize": {**_FULL_CFG, "canonicalize": False},
    "no_regime": {**_FULL_CFG, "regime": False},
    "no_online_memory": {**_FULL_CFG, "online_memory": False},
    "no_capacity": {**_FULL_CFG, "capacity": False},
    "no_significance": {**_FULL_CFG, "significance": False},
    "no_memory": {**_FULL_CFG, "memory": False, "debate": False},
}

ABLATION_LABELS: dict[str, str] = {
    "full": "HelixFactor (Full)",
    "no_debate": "w/o Debate",
    "no_causal": "w/o Causal",
    "no_canonicalize": "w/o Canonicalization",
    "no_regime": "w/o Regime",
    "no_online_memory": "w/o Online Memory",
    "no_capacity": "w/o Capacity",
    "no_significance": "w/o Significance",
    "no_memory": "w/o Memory (≈ FactorMiner NM)",
}

EXPECTED_CONTRIBUTION_SIGN: dict[str, int] = {
    "debate": +1,
    "causal": +1,
    "canonicalize": +1,
    "regime": +1,
    "online_memory": +1,
    "capacity": +1,
    "significance": +1,
    "memory": +1,
}

_FEATURE_KEYS = [
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$amt",
    "$vwap",
    "$returns",
]


def _merge_slices(train_data: dict, test_data: dict) -> dict:
    """Concatenate train/test slices into one runtime evaluation dictionary."""
    merged: dict[str, np.ndarray] = {}
    for key in sorted(set(train_data) | set(test_data)):
        if key not in train_data or key not in test_data:
            continue
        left = np.asarray(train_data[key], dtype=np.float64)
        right = np.asarray(test_data[key], dtype=np.float64)
        if left.ndim == 2 and right.ndim == 2 and left.shape[0] == right.shape[0]:
            merged[key] = np.concatenate([left, right], axis=1)
        else:
            merged[key] = np.asarray(left)
    return merged


def _slice_data(data: dict, start: int, end: int) -> dict:
    """Slice all 2-D benchmark arrays to a column range."""
    return {
        key: value[:, start:end]
        for key, value in data.items()
        if isinstance(value, np.ndarray) and value.ndim >= 2
    }


def _build_runtime_dataset(data: dict) -> EvaluationDataset:
    """Build a minimal runtime dataset from the benchmark dictionary format."""
    feature_keys = [key for key in _FEATURE_KEYS if key in data]
    if "forward_returns" not in data:
        raise ValueError("Runtime ablation requires 'forward_returns' in the data dict")
    if not feature_keys:
        raise ValueError("Runtime ablation requires at least one market feature array")

    arrays = [np.asarray(data[key], dtype=np.float64) for key in feature_keys]
    data_tensor = np.stack(arrays, axis=2)
    returns = np.asarray(data["forward_returns"], dtype=np.float64)
    timestamps = np.arange(returns.shape[1])
    asset_ids = np.arange(returns.shape[0])
    full_split = DatasetSplit(
        name="full",
        indices=np.arange(returns.shape[1]),
        timestamps=timestamps,
        returns=returns,
        target_returns={"target": returns},
        default_target="target",
    )

    # The caller populates train/test splits by passing a merged train+test view.
    return EvaluationDataset(
        data_dict={key: np.asarray(data[key], dtype=np.float64) for key in feature_keys},
        data_tensor=data_tensor,
        returns=returns,
        timestamps=timestamps,
        asset_ids=asset_ids,
        splits={"full": full_split},
        processed_df=pd.DataFrame(),
        target_panels={"target": returns},
        default_target="target",
    )


def _build_split_dataset(data: dict, split_name: str) -> EvaluationDataset:
    """Create a single-split runtime dataset from one benchmark slice."""
    dataset = _build_runtime_dataset(data)
    split = DatasetSplit(
        name=split_name,
        indices=np.arange(dataset.returns.shape[1]),
        timestamps=dataset.timestamps,
        returns=dataset.returns,
        target_returns={"target": dataset.returns},
        default_target="target",
    )
    dataset.splits = {split_name: split}
    return dataset


def _build_combined_dataset(train_data: dict, test_data: dict) -> EvaluationDataset:
    """Create a train/test runtime dataset from sliced benchmark inputs."""
    merged = _merge_slices(train_data, test_data)
    dataset = _build_runtime_dataset(merged)
    train_len = np.asarray(train_data["forward_returns"]).shape[1]
    test_len = np.asarray(test_data["forward_returns"]).shape[1]
    timestamps = np.arange(train_len + test_len)
    returns = np.asarray(merged["forward_returns"], dtype=np.float64)

    dataset.timestamps = timestamps
    dataset.returns = returns
    dataset.target_panels = {"target": returns}
    dataset.default_target = "target"
    dataset.splits = {
        "train": DatasetSplit(
            name="train",
            indices=np.arange(0, train_len),
            timestamps=timestamps[:train_len],
            returns=returns[:, :train_len],
            target_returns={"target": returns[:, :train_len]},
            default_target="target",
        ),
        "test": DatasetSplit(
            name="test",
            indices=np.arange(train_len, train_len + test_len),
            timestamps=timestamps[train_len:],
            returns=returns[:, train_len:],
            target_returns={"target": returns[:, train_len:]},
            default_target="target",
        ),
        "full": DatasetSplit(
            name="full",
            indices=np.arange(train_len + test_len),
            timestamps=timestamps,
            returns=returns,
            target_returns={"target": returns},
            default_target="target",
        ),
    }
    return dataset


def _build_mining_config(
    *,
    output_dir: str,
    target_library_size: int,
    batch_size: int,
    max_iterations: int,
    ic_threshold: float,
    correlation_threshold: float,
) -> MiningConfig:
    """Create a loop config tailored for a single runtime ablation."""
    cfg = MiningConfig(
        target_library_size=target_library_size,
        batch_size=batch_size,
        max_iterations=max_iterations,
        ic_threshold=ic_threshold,
        icir_threshold=0.5,
        correlation_threshold=correlation_threshold,
        replacement_ic_min=max(ic_threshold * 2.5, ic_threshold + 0.05),
        replacement_ic_ratio=1.3,
        fast_screen_assets=100,
        num_workers=1,
        output_dir=output_dir,
        backend="numpy",
        signal_failure_policy="reject",
    )
    cfg.benchmark_mode = "paper"
    cfg.research = None
    cfg.target_panels = None
    cfg.target_horizons = None
    return cfg


def _build_phase2_configs(flags: dict[str, bool]) -> dict[str, Any]:
    """Translate ablation flags into real HelixLoop runtime configs."""
    return {
        "debate_config": RuntimeDebateConfig() if flags.get("debate", True) else None,
        "causal_config": RuntimeCausalConfig(enabled=True) if flags.get("causal", True) else None,
        "regime_config": RuntimeRegimeConfig(enabled=True) if flags.get("regime", True) else None,
        "capacity_config": RuntimeCapacityConfig(enabled=True) if flags.get("capacity", True) else None,
        "significance_config": (
            RuntimeSignificanceConfig(enabled=True)
            if flags.get("significance", True)
            else None
        ),
        "canonicalize": flags.get("canonicalize", True),
    }


@contextmanager
def _patched_memory_hooks(enabled: bool):
    """Disable memory retrieval and learning when a no-memory ablation is requested."""
    if enabled:
        yield
        return

    def _empty_signal(*_args, **_kwargs) -> dict[str, Any]:
        return {
            "recommended_directions": [],
            "forbidden_directions": [],
            "insights": [],
            "library_state": {
                "library_size": 0,
                "recent_admission_rate": 0.0,
                "saturated_domains": {},
                "recent_admissions_count": 0,
                "recent_rejections_count": 0,
            },
            "prompt_text": "",
        }

    def _identity_memory(memory, *args, **kwargs):
        return memory

    patch_targets = [
        (ralph_loop_module, "retrieve_memory", _empty_signal),
        (ralph_loop_module, "form_memory", _identity_memory),
        (ralph_loop_module, "evolve_memory", _identity_memory),
        (helix_loop_module, "retrieve_memory", _empty_signal),
        (helix_loop_module, "form_memory", _identity_memory),
        (helix_loop_module, "evolve_memory", _identity_memory),
    ]

    originals = []
    for module, attr, replacement in patch_targets:
        originals.append((module, attr, getattr(module, attr)))
        setattr(module, attr, replacement)

    try:
        yield
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)


def _compute_avg_abs_rho(artifacts) -> float:
    if len(artifacts) < 2:
        return 0.0

    corr = np.abs(
        np.corrcoef([artifact.split_signals["train"].reshape(-1) for artifact in artifacts])
    )
    if corr.ndim != 2:
        return 0.0
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    return float(np.mean(upper)) if upper.size else 0.0


def _runtime_payload_to_result(
    *,
    method: str,
    payload: dict[str, Any],
    benchmark_library_size: int,
    benchmark_succeeded: int,
    elapsed_seconds: float,
    run_id: int,
) -> MethodResult:
    """Convert runtime benchmark output into a MethodResult."""
    library = payload.get("library", {})
    combinations = payload.get("combinations", {})
    selections = payload.get("selections", {})

    result = MethodResult(
        method=method,
        library_ic=float(library.get("ic", 0.0)),
        library_icir=float(library.get("icir", 0.0)),
        avg_abs_rho=float(library.get("avg_abs_rho", 0.0)),
        ew_ic=float(combinations.get("equal_weight", {}).get("ic", 0.0)),
        ew_icir=float(combinations.get("equal_weight", {}).get("icir", 0.0)),
        icw_ic=float(combinations.get("ic_weighted", {}).get("ic", 0.0)),
        icw_icir=float(combinations.get("ic_weighted", {}).get("icir", 0.0)),
        lasso_ic=float(selections.get("lasso", {}).get("ic", 0.0)),
        lasso_icir=float(selections.get("lasso", {}).get("icir", 0.0)),
        xgb_ic=float(selections.get("xgboost", {}).get("ic", 0.0)),
        xgb_icir=float(selections.get("xgboost", {}).get("icir", 0.0)),
        n_factors=benchmark_library_size,
        admission_rate=benchmark_library_size / max(benchmark_succeeded, 1),
        elapsed_seconds=elapsed_seconds,
        ic_series=None,
        run_id=run_id,
    )
    result.runtime_payload = payload
    return result


def _evaluate_runtime_library(
    library,
    dataset: EvaluationDataset,
    cfg: MiningConfig,
    *,
    target_library_size: int,
    cost_bps: list[float] | None = None,
) -> tuple[MethodResult, dict[str, Any], int, int]:
    """Recompute a mined library using the runtime benchmark contract."""
    if cost_bps is None:
        cost_bps = [1.0, 4.0, 7.0, 10.0, 11.0]

    factors = library.list_factors()
    artifacts = evaluate_factors(factors, dataset, signal_failure_policy="reject")
    [artifact for artifact in artifacts if artifact.succeeded]
    benchmark_library, benchmark_stats = build_benchmark_library(
        artifacts,
        cfg,
        split_name="train",
        ic_threshold=cfg.ic_threshold,
        correlation_threshold=cfg.correlation_threshold,
    )
    frozen = select_frozen_top_k(
        artifacts,
        benchmark_library,
        top_k=target_library_size,
        split_name="train",
    )
    payload = evaluate_frozen_set(
        frozen,
        dataset,
        split_name="test",
        fit_split="train",
        cost_bps=cost_bps,
    )
    payload["benchmark"] = {
        "admitted": benchmark_stats.get("admitted", 0),
        "succeeded": benchmark_stats.get("succeeded", 0),
        "replaced": benchmark_stats.get("replaced", 0),
        "threshold_rejections": benchmark_stats.get("threshold_rejections", 0),
        "correlation_rejections": benchmark_stats.get("correlation_rejections", 0),
        "freeze_library_size": benchmark_library.size,
        "frozen_top_k": [
            {
                "name": artifact.name,
                "formula": artifact.formula,
                "category": artifact.category,
                "train_ic": artifact.split_stats["train"]["ic_abs_mean"],
                "train_icir": abs(artifact.split_stats["train"]["icir"]),
            }
            for artifact in frozen
        ],
    }
    result = _runtime_payload_to_result(
        method="helix_phase2",
        payload=payload,
        benchmark_library_size=benchmark_library.size,
        benchmark_succeeded=max(int(benchmark_stats.get("succeeded", 0)), 1),
        elapsed_seconds=0.0,
        run_id=0,
    )
    result.n_factors = benchmark_library.size
    result.admission_rate = benchmark_library.size / max(benchmark_stats.get("succeeded", 0), 1)
    result.avg_abs_rho = _compute_avg_abs_rho(frozen)
    return result, payload, benchmark_library.size, int(benchmark_stats.get("succeeded", 0))


class AblatedMethodRunner:
    """Run one ablation variant through the real HelixLoop benchmark path."""

    def __init__(
        self,
        cfg: dict[str, bool],
        ic_threshold: float = 0.02,
        correlation_threshold: float = 0.5,
        seed: int = 42,
        llm_provider: Any | None = None,
        benchmark_mode: str = "paper",
    ) -> None:
        self._cfg = dict(cfg)
        self.ic_threshold = ic_threshold
        self.correlation_threshold = correlation_threshold
        self.seed = seed
        self.llm_provider = llm_provider
        self.benchmark_mode = benchmark_mode

    def _run_loop(
        self,
        *,
        train_data: dict,
        n_factors: int,
    ) -> tuple[HelixLoop, MiningConfig]:
        """Instantiate and run the real HelixLoop on the training slice."""
        phase2 = _build_phase2_configs(self._cfg)
        target_library_size = max(int(n_factors), 1)
        max_iterations = max(target_library_size * 4, 4)
        batch_size = max(4, min(target_library_size, 40))
        loop_dataset = _build_runtime_dataset(train_data)
        with tempfile.TemporaryDirectory(prefix="factorminer_ablation_") as tmp:
            mining_cfg = _build_mining_config(
                output_dir=tmp,
                target_library_size=target_library_size,
                batch_size=batch_size,
                max_iterations=max_iterations,
                ic_threshold=self.ic_threshold,
                correlation_threshold=self.correlation_threshold,
            )
            mining_cfg.benchmark_mode = self.benchmark_mode
            if self._cfg.get("memory", True):
                memory = ExperienceMemory()
            else:
                memory = ExperienceMemory()

            loop = HelixLoop(
                config=mining_cfg,
                data_tensor=loop_dataset.data_tensor,
                returns=np.asarray(train_data["forward_returns"], dtype=np.float64),
                llm_provider=self.llm_provider or MockProvider(),
                memory=memory,
                library=FactorLibrary(
                    correlation_threshold=self.correlation_threshold,
                    ic_threshold=self.ic_threshold,
                ),
                debate_config=phase2["debate_config"],
                enable_knowledge_graph=False,
                enable_embeddings=False,
                enable_auto_inventor=False,
                auto_invention_interval=10,
                canonicalize=phase2["canonicalize"],
                forgetting_lambda=0.95,
                causal_config=phase2["causal_config"],
                regime_config=phase2["regime_config"],
                capacity_config=phase2["capacity_config"],
                significance_config=phase2["significance_config"],
                volume=np.asarray(train_data.get("$amt", train_data["forward_returns"]), dtype=np.float64)
                if "$amt" in train_data
                else None,
            )
            with _patched_memory_hooks(self._cfg.get("memory", True) and self._cfg.get("online_memory", True)):
                loop.run(
                    target_size=target_library_size,
                    max_iterations=max_iterations,
                    resume=False,
                )
            return loop, mining_cfg

    def run(
        self,
        data: dict,
        test_data: dict,
        n_factors: int = 40,
    ) -> MethodResult:
        """Run this ablation variant using the real loop + runtime contract."""
        t0 = time.perf_counter()
        train_dataset = _build_split_dataset(data, "train")
        benchmark_dataset = _build_combined_dataset(data, test_data)

        loop, mining_cfg = self._run_loop(train_data=data, n_factors=n_factors)
        result, payload, benchmark_library_size, benchmark_succeeded = _evaluate_runtime_library(
            loop.library,
            benchmark_dataset,
            mining_cfg,
            target_library_size=n_factors,
        )
        elapsed = time.perf_counter() - t0

        result.elapsed_seconds = elapsed
        result.method = "helix_phase2"
        result.run_id = self.seed
        result.runtime_payload = {
            **payload,
            "train_split": {
                "train_length": train_dataset.returns.shape[1],
                "benchmark_library_size": benchmark_library_size,
                "benchmark_succeeded": benchmark_succeeded,
            },
            "ablation": {
                "name": self._cfg,
                "seed": self.seed,
            },
        }
        return result


class AblationStudy:
    """Run real-loop ablations and summarize component contribution."""

    def __init__(
        self,
        ic_threshold: float = 0.02,
        correlation_threshold: float = 0.5,
        seed: int = 42,
        configs: dict[str, dict[str, bool]] | None = None,
        llm_provider: Any | None = None,
        benchmark_mode: str = "paper",
    ) -> None:
        self.ic_threshold = ic_threshold
        self.correlation_threshold = correlation_threshold
        self.seed = seed
        self.configs = configs or ABLATION_CONFIGS
        self.llm_provider = llm_provider
        self.benchmark_mode = benchmark_mode

    def run_ablation(
        self,
        data: dict,
        train_period: tuple[int, int],
        test_period: tuple[int, int],
        n_factors: int = 40,
        configs_to_run: list[str] | None = None,
    ) -> AblationResult:
        """Run one or more ablation variants on the real loop pipeline."""
        configs_to_run = configs_to_run or list(self.configs.keys())
        train_data = _slice_data(data, *train_period)
        test_data = _slice_data(data, *test_period)

        config_results: dict[str, MethodResult] = {}
        for cfg_name in configs_to_run:
            cfg = self.configs.get(cfg_name)
            if cfg is None:
                logger.warning("Unknown ablation config: %s", cfg_name)
                continue

            label = ABLATION_LABELS.get(cfg_name, cfg_name)
            logger.info("Running ablation: %s", label)
            t0 = time.perf_counter()
            try:
                runner = AblatedMethodRunner(
                    cfg=cfg,
                    ic_threshold=self.ic_threshold,
                    correlation_threshold=self.correlation_threshold,
                    seed=self.seed,
                    llm_provider=self.llm_provider,
                    benchmark_mode=self.benchmark_mode,
                )
                result = runner.run(
                    data=train_data,
                    test_data=test_data,
                    n_factors=n_factors,
                )
                result.method = cfg_name
                config_results[cfg_name] = result
            except Exception as exc:
                logger.warning("Ablation %s failed: %s", cfg_name, exc)
                config_results[cfg_name] = MethodResult(method=cfg_name)

            elapsed = time.perf_counter() - t0
            ic = config_results[cfg_name].library_ic
            logger.info("  %s: IC=%.4f  elapsed=%.1fs", cfg_name, ic, elapsed)

        ablation = AblationResult(
            configs=configs_to_run,
            results=config_results,
        )
        ablation.contributions = self.summarize_contributions(ablation)
        return ablation

    def summarize_contributions(self, result: AblationResult) -> pd.DataFrame:
        """Summarize component contributions relative to the full runtime run."""
        full = result.results.get("full")
        if full is None:
            logger.warning("No 'full' config in ablation results; cannot summarize")
            return pd.DataFrame()

        rows = []
        component_map = {
            "no_debate": "debate",
            "no_causal": "causal",
            "no_canonicalize": "canonicalize",
            "no_regime": "regime",
            "no_online_memory": "online_memory",
            "no_capacity": "capacity",
            "no_significance": "significance",
            "no_memory": "memory",
        }

        for ablation_key, component in component_map.items():
            ablated = result.results.get(ablation_key)
            if ablated is None:
                continue

            ic_contrib = full.library_ic - ablated.library_ic
            icir_contrib = full.library_icir - ablated.library_icir
            adm_delta = full.admission_rate - ablated.admission_rate

            expected_sign = EXPECTED_CONTRIBUTION_SIGN.get(component, +1)
            actual_sign = np.sign(ic_contrib) if ic_contrib != 0 else 0
            if abs(ic_contrib) < 0.0005:
                interpretation = "Negligible"
            elif actual_sign == expected_sign:
                pct = abs(ic_contrib) / max(full.library_ic, 1e-6) * 100
                interpretation = f"Helps (+{pct:.1f}% IC)"
            else:
                interpretation = "Hurts (unexpected direction)"

            rows.append({
                "component": component,
                "ablation_config": ablation_key,
                "ic_full": full.library_ic,
                "ic_ablated": ablated.library_ic,
                "ic_contribution": ic_contrib,
                "ic_contribution_pct": ic_contrib / max(full.library_ic, 1e-6) * 100,
                "icir_full": full.library_icir,
                "icir_ablated": ablated.library_icir,
                "icir_contribution": icir_contrib,
                "admission_rate_delta": adm_delta,
                "interpretation": interpretation,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ic_contribution", ascending=False).reset_index(drop=True)
        return df

    def to_latex_table(self, result: AblationResult) -> str:
        """Generate a LaTeX ablation study table."""
        df = result.contributions
        if df is None or df.empty:
            return "% No ablation data available"

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{HelixFactor Ablation Study: Component Contributions}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccccl}",
            r"\toprule",
            r"Component & IC (Full) & IC (Ablated) & $\Delta$IC & $\Delta$IC\% & Interpretation \\",
            r"\midrule",
        ]

        for _, row in df.iterrows():
            lines.append(
                f"{row['component'].replace('_', r' ')} & "
                f"{row['ic_full']:.4f} & "
                f"{row['ic_ablated']:.4f} & "
                f"{row['ic_contribution']:+.4f} & "
                f"{row['ic_contribution_pct']:+.1f}\\% & "
                f"{row['interpretation']} \\\\"
            )

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def print_summary(self, result: AblationResult) -> None:
        """Print a human-readable ablation summary."""
        df = result.contributions
        if df is None or df.empty:
            print("  No ablation summary available.")
            return

        print("\n" + "=" * 70)
        print("  Ablation Study: Component Contributions")
        print("=" * 70)

        full = result.results.get("full")
        if full:
            print(f"\n  FULL System: IC={full.library_ic:.4f}  ICIR={full.library_icir:.3f}")
            print()

        header = (
            f"  {'Component':<22} {'IC Full':>8} {'IC Ablated':>10} "
            f"{'Delta IC':>10} {'Delta%':>8}  Interpretation"
        )
        print(header)
        print("  " + "-" * 80)

        for _, row in df.iterrows():
            comp = row["component"].replace("_", " ")
            print(
                f"  {comp:<22} {row['ic_full']:>8.4f} {row['ic_ablated']:>10.4f} "
                f"{row['ic_contribution']:>+10.4f} {row['ic_contribution_pct']:>+7.1f}%  "
                f"{row['interpretation']}"
            )

        print()


def run_full_ablation_study(
    n_assets: int = 100,
    n_periods: int = 500,
    n_factors: int = 40,
    seed: int = 42,
    configs_to_run: list[str] | None = None,
    verbose: bool = True,
) -> AblationResult:
    """Run the full runtime ablation study on mock data."""
    if verbose:
        print("\nGenerating mock data for ablation study...")

    from factorminer.benchmark.helix_benchmark import _build_mock_data_dict

    data = _build_mock_data_dict(n_assets=n_assets, n_periods=n_periods, seed=seed)
    T = list(data.values())[0].shape[1]
    train_end = int(T * 0.7)

    if verbose:
        print(f"  Data: M={n_assets}, T={T}, train=0:{train_end}, test={train_end}:{T}")
        cfgs = configs_to_run or list(ABLATION_CONFIGS.keys())
        print(f"  Running {len(cfgs)} ablation configurations through real loops...")

    study = AblationStudy(seed=seed, llm_provider=MockProvider())
    result = study.run_ablation(
        data=data,
        train_period=(0, train_end),
        test_period=(train_end, T),
        n_factors=n_factors,
        configs_to_run=configs_to_run,
    )

    if verbose:
        study.print_summary(result)

    return result
