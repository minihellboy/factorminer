"""Click-based CLI for FactorMiner."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from dataclasses import fields
from importlib.util import find_spec
from pathlib import Path

import click
import numpy as np
import yaml

from factorminer.configs import DEFAULT_CONFIG_PATH, load_default_yaml
from factorminer.utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    """Configure root logger for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not verbose:
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0 for slice.",
            category=RuntimeWarning,
        )


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge two plain dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config_data(config_path: str | None) -> dict:
    """Load raw YAML including top-level fields not mapped to dataclasses."""
    raw = load_default_yaml()
    if config_path:
        with open(config_path) as f:
            user_raw = yaml.safe_load(f) or {}
        if isinstance(user_raw, dict):
            raw = _deep_merge_dict(raw, user_raw)
    return raw


def _load_data(cfg, data_path: str | None, mock: bool):
    """Load market data from file or generate mock data.

    Returns
    -------
    pd.DataFrame
        Market data with columns: datetime, asset_id, open, high, low,
        close, volume, amount.
    """
    raw_cfg = getattr(cfg, "_raw", {})
    configured_path = raw_cfg.get("data_path")

    if mock:
        click.echo("Generating mock market data...")
        from factorminer.data.mock_data import MockConfig, generate_mock_data

        mock_cfg = MockConfig(
            num_assets=50,
            num_periods=500,
            frequency="1d",
            plant_alpha=True,
        )
        return generate_mock_data(mock_cfg)

    # Try data_path argument, then config top-level data_path
    path = data_path
    if path is None:
        path = configured_path

    if path is None:
        click.echo("No data path specified. Use --data or --mock flag.")
        raise click.Abort()

    click.echo(f"Loading market data from: {path}")
    from factorminer.data.loader import load_market_data

    return load_market_data(path)


def _prepare_data_arrays(df):
    """Convert a market DataFrame to numpy arrays for the mining loop.

    Returns
    -------
    data_tensor : np.ndarray, shape (M, T, F)
        Market data tensor.
    returns : np.ndarray, shape (M, T)
        Forward returns.
    """
    asset_ids = sorted(df["asset_id"].unique())
    dates = sorted(df["datetime"].unique())
    M = len(asset_ids)
    T = len(dates)

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "returns",
    ]
    F = len(feature_cols)

    data_tensor = np.full((M, T, F), np.nan, dtype=np.float64)
    returns = np.full((M, T), np.nan, dtype=np.float64)

    asset_to_idx = {a: i for i, a in enumerate(asset_ids)}
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for _, row in df.iterrows():
        ai = asset_to_idx[row["asset_id"]]
        ti = date_to_idx[row["datetime"]]
        for fi, col in enumerate(feature_cols[:6]):
            data_tensor[ai, ti, fi] = row[col]

        if "vwap" in row.index and not np.isnan(row["vwap"]):
            data_tensor[ai, ti, 6] = row["vwap"]
        elif (
            not np.isnan(row["volume"])
            and abs(row["volume"]) > 1e-12
            and not np.isnan(row["amount"])
        ):
            data_tensor[ai, ti, 6] = row["amount"] / row["volume"]

        if "returns" in row.index and not np.isnan(row["returns"]):
            data_tensor[ai, ti, 7] = row["returns"]

    close_idx = feature_cols.index("close")
    amount_idx = feature_cols.index("amount")
    vwap_idx = feature_cols.index("vwap")
    feature_returns_idx = feature_cols.index("returns")

    # Fill derived VWAP where the source file did not provide it.
    volume = data_tensor[:, :, feature_cols.index("volume")]
    amount = data_tensor[:, :, amount_idx]
    derived_vwap = np.divide(
        amount,
        volume,
        out=np.full_like(amount, np.nan),
        where=np.abs(volume) > 1e-12,
    )
    missing_vwap = np.isnan(data_tensor[:, :, vwap_idx])
    data_tensor[:, :, vwap_idx] = np.where(
        missing_vwap,
        np.where(np.isnan(derived_vwap), data_tensor[:, :, close_idx], derived_vwap),
        data_tensor[:, :, vwap_idx],
    )

    # Compute bar returns feature from close prices where missing.
    for i in range(M):
        close = data_tensor[i, :, close_idx]
        asset_returns = np.full(T, np.nan, dtype=np.float64)
        asset_returns[1:] = (close[1:] - close[:-1]) / np.where(
            close[:-1] == 0, np.nan, close[:-1]
        )
        missing_feature_returns = np.isnan(data_tensor[i, :, feature_returns_idx])
        data_tensor[i, :, feature_returns_idx] = np.where(
            missing_feature_returns,
            asset_returns,
            data_tensor[i, :, feature_returns_idx],
        )

        # Simple 1-period forward return target.
        returns[i, :-1] = (close[1:] - close[:-1]) / np.where(
            close[:-1] == 0, np.nan, close[:-1]
        )

    return data_tensor, returns


def _create_llm_provider(cfg, mock: bool):
    """Create an LLM provider from config or use mock."""
    from factorminer.agent.llm_interface import MissingAPIKeyError, MockProvider, create_provider

    if mock:
        click.echo("Using mock LLM provider (no API calls).")
        return MockProvider()

    llm_config = {
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
    }
    # Use api_key from config if set
    if hasattr(cfg, "_raw") and cfg._raw.get("llm", {}).get("api_key"):
        llm_config["api_key"] = cfg._raw["llm"]["api_key"]

    click.echo(f"Using LLM provider: {cfg.llm.provider}/{cfg.llm.model}")
    try:
        return create_provider(llm_config)
    except MissingAPIKeyError as exc:
        click.echo(f"LLM configuration error: {exc}")
        raise click.Abort() from exc


def _build_core_mining_config(cfg, output_dir: Path, mock: bool = False):
    """Create the flat mining config expected by RalphLoop/HelixLoop."""
    from factorminer.core.config import MiningConfig as CoreMiningConfig

    signal_failure_policy = (
        "synthetic" if mock else cfg.evaluation.signal_failure_policy
    )
    memory_cfg = getattr(cfg, "memory", None)

    mining_cfg = CoreMiningConfig(
        target_library_size=cfg.mining.target_library_size,
        batch_size=cfg.mining.batch_size,
        max_iterations=cfg.mining.max_iterations,
        ic_threshold=cfg.mining.ic_threshold,
        icir_threshold=cfg.mining.icir_threshold,
        correlation_threshold=cfg.mining.correlation_threshold,
        replacement_ic_min=cfg.mining.replacement_ic_min,
        replacement_ic_ratio=cfg.mining.replacement_ic_ratio,
        fast_screen_assets=cfg.evaluation.fast_screen_assets,
        num_workers=cfg.evaluation.num_workers,
        output_dir=str(output_dir),
        backend=cfg.evaluation.backend,
        gpu_device=cfg.evaluation.gpu_device,
        redundancy_metric=getattr(cfg.evaluation, "redundancy_metric", "spearman"),
        signal_failure_policy=signal_failure_policy,
        memory_policy=getattr(memory_cfg, "policy", "paper"),
        memory_regime_lookback_window=getattr(memory_cfg, "regime_lookback_window", 60),
    )
    mining_cfg.research = getattr(cfg, "research", None)
    benchmark_cfg = getattr(cfg, "benchmark", None)
    mining_cfg.benchmark_mode = getattr(benchmark_cfg, "mode", "paper")
    mining_cfg.target_panels = None
    mining_cfg.target_horizons = None
    return mining_cfg


def _attach_runtime_targets(mining_config, dataset) -> None:
    """Attach multi-horizon runtime metadata for research-mode mining."""
    mining_config.target_panels = dataset.target_panels
    mining_config.target_horizons = {
        name: max(getattr(spec, "holding_bars", 1), 1)
        for name, spec in dataset.target_specs.items()
    }


def _save_result_library(library, output_dir: Path) -> Path:
    """Persist a factor library to the standard output location."""
    from factorminer.core.library_io import save_library

    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = output_dir / "factor_library"
    save_library(library, lib_path)
    return lib_path.with_suffix(".json")


def _filter_dataclass_kwargs(source, target_cls):
    """Copy shared dataclass fields from one config object to another."""
    target_fields = {f.name for f in fields(target_cls)}
    source_fields = getattr(source, "__dataclass_fields__", {})
    return {
        name: getattr(source, name)
        for name in source_fields
        if name in target_fields
    }


def _build_debate_config(cfg):
    """Build the runtime debate config from YAML config settings."""
    if not cfg.phase2.debate.enabled:
        return None

    from factorminer.agent.debate import DebateConfig as RuntimeDebateConfig
    from factorminer.agent.specialists import DEFAULT_SPECIALISTS

    available = len(DEFAULT_SPECIALISTS)
    requested = cfg.phase2.debate.num_specialists
    selected = list(DEFAULT_SPECIALISTS[:requested])
    if requested > available:
        logger.warning(
            "Requested %d specialists but only %d are available; using all defaults.",
            requested,
            available,
        )

    return RuntimeDebateConfig(
        specialists=selected,
        enable_critic=cfg.phase2.debate.enable_critic,
        candidates_per_specialist=cfg.phase2.debate.candidates_per_specialist,
        top_k_after_critic=cfg.phase2.debate.top_k_after_critic,
        critic_temperature=cfg.phase2.debate.critic_temperature,
    )


def _build_phase2_runtime_configs(cfg):
    """Instantiate evaluation/runtime configs for the Helix loop."""
    from factorminer.evaluation.capacity import CapacityConfig as RuntimeCapacityConfig
    from factorminer.evaluation.causal import CausalConfig as RuntimeCausalConfig
    from factorminer.evaluation.regime import RegimeConfig as RuntimeRegimeConfig
    from factorminer.evaluation.significance import (
        SignificanceConfig as RuntimeSignificanceConfig,
    )

    causal_config = None
    if cfg.phase2.causal.enabled:
        causal_config = RuntimeCausalConfig(
            **_filter_dataclass_kwargs(cfg.phase2.causal, RuntimeCausalConfig)
        )

    regime_config = None
    if cfg.phase2.regime.enabled:
        regime_config = RuntimeRegimeConfig(
            **_filter_dataclass_kwargs(cfg.phase2.regime, RuntimeRegimeConfig)
        )

    capacity_config = None
    if cfg.phase2.capacity.enabled:
        capacity_config = RuntimeCapacityConfig(
            **_filter_dataclass_kwargs(cfg.phase2.capacity, RuntimeCapacityConfig)
        )

    significance_config = None
    if cfg.phase2.significance.enabled:
        significance_config = RuntimeSignificanceConfig(
            **_filter_dataclass_kwargs(cfg.phase2.significance, RuntimeSignificanceConfig)
        )

    return {
        "debate_config": _build_debate_config(cfg),
        "causal_config": causal_config,
        "regime_config": regime_config,
        "capacity_config": capacity_config,
        "significance_config": significance_config,
    }


def _extract_capacity_volume(data_tensor: np.ndarray) -> np.ndarray | None:
    """Prefer dollar volume (`amount`) and fall back to raw volume if needed."""
    if data_tensor.ndim != 3 or data_tensor.shape[2] == 0:
        return None

    amount_idx = 5
    volume_idx = 4

    if data_tensor.shape[2] > amount_idx:
        amount = data_tensor[:, :, amount_idx]
        if not np.all(np.isnan(amount)):
            return amount

    if data_tensor.shape[2] > volume_idx:
        volume = data_tensor[:, :, volume_idx]
        if not np.all(np.isnan(volume)):
            return volume

    return None


def _active_phase2_features(cfg) -> list[str]:
    """Describe the effective Helix feature set for CLI output."""
    features: list[str] = []

    if cfg.phase2.causal.enabled:
        features.append("causal")
    if cfg.phase2.regime.enabled:
        features.append("regime")
    if cfg.phase2.capacity.enabled:
        features.append("capacity")
    if cfg.phase2.significance.enabled:
        features.append("significance")
    if cfg.phase2.debate.enabled:
        features.append("debate")
    if cfg.phase2.auto_inventor.enabled:
        features.append("auto-inventor")
    if cfg.phase2.helix.enabled and cfg.phase2.helix.enable_canonicalization:
        features.append("canonicalization")
    if cfg.phase2.helix.enabled and cfg.phase2.helix.enable_knowledge_graph:
        features.append("knowledge-graph")
    if cfg.phase2.helix.enabled and cfg.phase2.helix.enable_embeddings:
        features.append("embeddings")

    return features


def _load_runtime_dataset_for_analysis(cfg, data_path: str | None, mock: bool):
    """Load, preprocess, split, and tensorize data for analysis commands."""
    from factorminer.evaluation.runtime import load_runtime_dataset

    raw_df = _load_data(cfg, data_path, mock)
    return load_runtime_dataset(raw_df, cfg)


def _recompute_analysis_artifacts(library, dataset, signal_failure_policy: str):
    """Recompute library factors on the canonical analysis dataset."""
    from factorminer.evaluation.runtime import evaluate_factors

    return evaluate_factors(
        library.list_factors(),
        dataset,
        signal_failure_policy=signal_failure_policy,
    )


def _report_artifact_failures(artifacts, header: str) -> list[str]:
    """Print a concise recomputation failure summary and return failure texts."""
    from factorminer.evaluation.runtime import summarize_failures

    failures = summarize_failures(artifacts)
    if not failures:
        return []

    click.echo(f"{header}: {len(failures)} factor(s) failed to recompute.")
    for failure in failures[:10]:
        click.echo(f"  - {failure}")
    if len(failures) > 10:
        click.echo(f"  ... and {len(failures) - 10} more")

    return failures


def _artifact_map_by_id(artifacts):
    return {artifact.factor_id: artifact for artifact in artifacts}


def _select_artifacts_for_ids(artifacts, factor_ids: tuple[int, ...]):
    if not factor_ids:
        return [artifact for artifact in artifacts if artifact.succeeded]

    artifact_map = _artifact_map_by_id(artifacts)
    selected = []
    failed = []
    missing = []
    for factor_id in factor_ids:
        artifact = artifact_map.get(factor_id)
        if artifact is None:
            missing.append(str(factor_id))
        elif not artifact.succeeded:
            failed.append(artifact)
        else:
            selected.append(artifact)

    if missing:
        click.echo(f"Missing recomputed factors for ids: {', '.join(missing)}")
        raise click.Abort()
    if failed:
        click.echo("Requested factors failed to recompute:")
        for artifact in failed:
            click.echo(f"  - {artifact.factor_id}: {artifact.name} ({artifact.error})")
        raise click.Abort()

    return selected


def _analysis_output_path(output_dir: Path, stem: str, split_name: str, fmt: str) -> str:
    return str(output_dir / f"{stem}_{split_name}.{fmt}")


def _print_benchmark_summary(title: str, payload: dict) -> None:
    """Emit a concise benchmark summary for CLI runs."""
    click.echo("=" * 60)
    click.echo(title)
    click.echo("=" * 60)
    if not payload:
        click.echo("No benchmark results produced.")
        return

    if all(isinstance(value, dict) and "universes" in value for value in payload.values()):
        for baseline, result in payload.items():
            click.echo(f"Baseline: {baseline}")
            click.echo(
                f"  Freeze library: {result.get('freeze_library_size', 0)} "
                f"| Frozen Top-K: {len(result.get('frozen_top_k', []))}"
            )
            for universe, metrics in result.get("universes", {}).items():
                library = metrics.get("library", {})
                click.echo(
                    f"  {universe}: library IC={library.get('ic', 0.0):.4f}, "
                    f"ICIR={library.get('icir', 0.0):.4f}, "
                    f"Avg|rho|={library.get('avg_abs_rho', 0.0):.4f}"
                )
    else:
        click.echo(json.dumps(payload, indent=2))


def _print_recomputed_factor_table(artifacts, split_name: str) -> None:
    click.echo(
        f"{'ID':>4s}  {'Name':<35s}  {'IC Mean':>8s}  {'Paper IC':>8s}  "
        f"{'Abs IC':>8s}  {'Paper ICIR':>10s}  {'Win%':>6s}  {'Turn':>6s}"
    )
    click.echo("-" * 108)

    for artifact in artifacts:
        stats = artifact.split_stats[split_name]
        click.echo(
            f"{artifact.factor_id:4d}  {artifact.name:<35s}  "
            f"{stats['ic_mean']:8.4f}  "
            f"{stats.get('ic_paper_mean', abs(stats['ic_mean'])):8.4f}  "
            f"{stats['ic_abs_mean']:8.4f}  "
            f"{stats.get('ic_paper_icir', abs(stats['icir'])):10.3f}  "
            f"{stats['ic_win_rate'] * 100:5.1f}%  "
            f"{stats['turnover']:6.3f}"
        )


def _print_split_summary(artifacts, split_name: str) -> None:
    if not artifacts:
        click.echo("  No successful factor recomputations.")
        return

    ic_values = [artifact.split_stats[split_name]["ic_mean"] for artifact in artifacts]
    paper_ic_values = [
        artifact.split_stats[split_name].get(
            "ic_paper_mean",
            abs(artifact.split_stats[split_name]["ic_mean"]),
        )
        for artifact in artifacts
    ]
    abs_ic_values = [artifact.split_stats[split_name]["ic_abs_mean"] for artifact in artifacts]
    paper_icir_values = [
        artifact.split_stats[split_name].get(
            "ic_paper_icir",
            abs(artifact.split_stats[split_name]["icir"]),
        )
        for artifact in artifacts
    ]
    click.echo("-" * 108)
    click.echo(f"  Total factors:    {len(artifacts)}")
    click.echo(f"  Mean IC:          {np.mean(ic_values):.4f}")
    click.echo(f"  Mean paper IC:    {np.mean(paper_ic_values):.4f}")
    click.echo(f"  Mean abs IC:      {np.mean(abs_ic_values):.4f}")
    click.echo(f"  Mean paper ICIR:  {np.mean(paper_icir_values):.3f}")
    click.echo(f"  Max paper IC:     {max(paper_ic_values):.4f}")
    click.echo(f"  Min paper IC:     {min(paper_ic_values):.4f}")


def _load_library_from_path(library_path: str):
    """Load a factor library, handling both .json extension and base path.

    Returns
    -------
    FactorLibrary
    """
    from factorminer.core.library_io import load_library

    path = Path(library_path)
    # load_library expects the base path (without .json extension)
    # but also works with .json since it calls path.with_suffix(".json")
    if path.suffix == ".json":
        base_path = path.with_suffix("")
    else:
        base_path = path

    try:
        library = load_library(base_path)
        click.echo(f"Loaded factor library: {library.size} factors")
        return library
    except FileNotFoundError:
        click.echo(f"Error: Factor library not found at {library_path}")
        click.echo(f"  Tried: {base_path}.json")
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error loading library: {e}")
        raise click.Abort()


def _json_safe(value):
    """Convert common runtime values to JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _check(status: str, name: str, detail: str, **extra) -> dict:
    return {"status": status, "name": name, "detail": detail, **extra}


def _doctor_checks(cfg, raw: dict, output_dir: Path) -> list[dict]:
    """Collect local install and runtime readiness checks."""
    checks: list[dict] = []

    default_data = load_default_yaml()
    if DEFAULT_CONFIG_PATH.exists() and default_data:
        checks.append(_check("ok", "packaged_config", f"Found {DEFAULT_CONFIG_PATH}"))
    else:
        checks.append(_check("error", "packaged_config", "Missing packaged default.yaml"))

    backend = cfg.evaluation.backend
    checks.append(_check("ok", "effective_backend", backend))
    if backend == "gpu":
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
            status = "ok" if cuda_ok else "error"
            detail = "CUDA is available" if cuda_ok else "GPU backend requested but CUDA is unavailable"
            checks.append(_check(status, "cuda", detail))
        except Exception as exc:
            checks.append(_check("error", "cuda", f"GPU backend requested but torch failed: {exc}"))

    optional_modules = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google-generativeai": "google.generativeai",
        "sentence-transformers": "sentence_transformers",
        "faiss-cpu": "faiss",
    }
    for package, module in optional_modules.items():
        if find_spec(module) is None:
            checks.append(_check("warning", f"optional:{package}", "Not installed"))
        else:
            checks.append(_check("ok", f"optional:{package}", "Installed"))

    provider = cfg.llm.provider
    api_key = raw.get("llm", {}).get("api_key") if isinstance(raw.get("llm"), dict) else None
    env_names = {
        "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
    }
    if provider == "mock":
        checks.append(_check("ok", "llm", "Mock provider selected; no API key required"))
    elif api_key or any(os.getenv(name) for name in env_names.get(provider, ())):
        checks.append(_check("ok", "llm", f"{provider} credentials available"))
    else:
        checks.append(_check("warning", "llm", f"{provider} selected but no API key was found"))

    data_path = raw.get("data_path")
    if data_path:
        path = Path(data_path).expanduser()
        status = "ok" if path.exists() else "error"
        detail = f"Configured data path exists: {path}" if path.exists() else f"Configured data path is missing: {path}"
        checks.append(_check(status, "data_path", detail))
    else:
        checks.append(_check("warning", "data_path", "No data_path configured; use --mock or --data"))

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".factorminer-doctor-", dir=output_dir, delete=True):
            pass
        checks.append(_check("ok", "output_dir", f"Writable: {output_dir}"))
    except Exception as exc:
        checks.append(_check("error", "output_dir", f"Not writable: {output_dir} ({exc})"))

    return checks


def _print_doctor_report(checks: list[dict]) -> None:
    click.echo("FactorMiner Doctor")
    click.echo("=" * 60)
    for item in checks:
        label = item["status"].upper()
        click.echo(f"[{label:<7}] {item['name']}: {item['detail']}")


def _starter_config() -> dict:
    """Return a CPU-safe, mock-friendly starter config."""
    config = load_default_yaml()
    config = _deep_merge_dict(config, {
        "output_dir": "./output",
        "data_path": None,
        "mining": {
            "target_library_size": 5,
            "batch_size": 8,
            "max_iterations": 3,
            "ic_threshold": 0.0001,
            "icir_threshold": 0.0001,
            "correlation_threshold": 1.0,
            "replacement_ic_min": 0.001,
            "replacement_ic_ratio": 1.0,
        },
        "evaluation": {
            "backend": "numpy",
            "signal_failure_policy": "synthetic",
        },
        "llm": {
            "provider": "mock",
            "model": "mock",
            "batch_candidates": 8,
        },
    })
    return config


def _accepted_alias_lines() -> list[str]:
    """Describe loader-supported column aliases for data validation output."""
    from factorminer.data.loader import COLUMN_ALIASES, REQUIRED_COLUMNS

    lines = []
    for canonical in REQUIRED_COLUMNS:
        aliases = COLUMN_ALIASES.get(canonical, [])
        accepted = ", ".join([canonical, *aliases])
        lines.append(f"  {canonical}: {accepted}")
    return lines


def _split_row_counts_for_validation(report, cfg, hdf_key: str) -> dict[str, int] | None:
    """Best-effort row counts for configured train/test periods."""
    datetime_source = report.canonical_mapping.get("datetime")
    if not datetime_source:
        return None

    try:
        import pandas as pd

        from factorminer.data.validation import _read_raw_frame

        df = _read_raw_frame(Path(report.path), report.fmt, hdf_key=hdf_key)
        timestamps = pd.to_datetime(df[datetime_source], errors="coerce")
        train_start, train_end = [pd.Timestamp(value) for value in cfg.data.train_period]
        test_start, test_end = [pd.Timestamp(value) for value in cfg.data.test_period]
        return {
            "train": int(((timestamps >= train_start) & (timestamps <= train_end)).sum()),
            "test": int(((timestamps >= test_start) & (timestamps <= test_end)).sum()),
        }
    except Exception:
        return None


def _render_validation_next_steps(report, cfg, path: str, hdf_key: str) -> str:
    """Render actionable guidance after the validator's structural report."""
    lines: list[str] = []
    lines.append("")
    lines.append("Accepted aliases")
    lines.append("-" * 60)
    lines.extend(_accepted_alias_lines())
    lines.append("")
    lines.append("Derived fields")
    lines.append("-" * 60)
    lines.append("  vwap: derived as amount / volume when missing, falling back to close if needed")
    lines.append("  returns: derived from close-to-close prices when the feature column is missing")

    split_counts = _split_row_counts_for_validation(report, cfg, hdf_key)
    lines.append("")
    lines.append("Configured split coverage")
    lines.append("-" * 60)
    if split_counts is None:
        lines.append("  Could not compute split coverage from the datetime column.")
    else:
        lines.append(
            f"  train {cfg.data.train_period[0]}..{cfg.data.train_period[1]}: "
            f"{split_counts['train']} rows"
        )
        lines.append(
            f"  test  {cfg.data.test_period[0]}..{cfg.data.test_period[1]}: "
            f"{split_counts['test']} rows"
        )
        if split_counts["train"] == 0:
            lines.append("  WARN: configured train split is empty for this file.")
        if split_counts["test"] == 0:
            lines.append("  WARN: configured test split is empty for this file.")

    lines.append("")
    lines.append("Next command")
    lines.append("-" * 60)
    lines.append(f"  uv run factorminer -o output mine --data {path}")
    return "\n".join(lines)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {"value": payload}
    except Exception as exc:
        return {"_error": str(exc)}


def _inspect_session_dir(output_dir: Path) -> dict:
    """Summarize a FactorMiner output directory."""
    artifacts = {
        "run_manifest": output_dir / "run_manifest.json",
        "session": output_dir / "session.json",
        "session_log": output_dir / "session_log.json",
        "factor_library": output_dir / "factor_library.json",
    }
    payloads = {name: _read_json(path) for name, path in artifacts.items()}
    warnings_out: list[str] = []
    for name, path in artifacts.items():
        payload = payloads[name]
        if payload is None:
            warnings_out.append(f"Missing {name}: {path}")
        elif "_error" in payload:
            warnings_out.append(f"Could not parse {name}: {payload['_error']}")

    manifest = payloads.get("run_manifest") or {}
    session_payload = payloads.get("session") or {}
    session_log = payloads.get("session_log") or {}
    library = payloads.get("factor_library") or {}

    factors = library.get("factors")
    if not isinstance(factors, list):
        factors = []
    library_count = len(factors)
    session_final_size = session_payload.get("last_library_size")
    manifest_size = manifest.get("library_size")
    if session_final_size is not None and session_final_size != library_count:
        warnings_out.append(
            f"session.json last_library_size={session_final_size} but factor_library.json has {library_count}"
        )
    if manifest_size is not None and manifest_size != library_count:
        warnings_out.append(
            f"run_manifest.json library_size={manifest_size} but factor_library.json has {library_count}"
        )

    summary = session_log.get("summary", {}) if isinstance(session_log, dict) else {}
    iterations = session_payload.get("total_iterations") or len(session_log.get("iterations", []))
    total_candidates = summary.get("total_candidates", 0)
    total_admitted = summary.get("total_admitted", 0)
    yield_rate = summary.get("overall_yield_rate")
    if yield_rate is None and total_candidates:
        yield_rate = total_admitted / total_candidates

    config_summary = session_payload.get("config") or manifest.get("config_summary", {})
    dataset_summary = manifest.get("dataset_summary", {})

    return {
        "output_dir": output_dir,
        "status": session_payload.get("status", "unknown"),
        "library_size": library_count,
        "session_final_library_size": session_final_size,
        "manifest_library_size": manifest_size,
        "iterations": iterations,
        "yield_rate": yield_rate,
        "backend": config_summary.get("backend"),
        "data_tensor_shape": dataset_summary.get("data_tensor_shape"),
        "returns_shape": dataset_summary.get("returns_shape"),
        "warnings": warnings_out,
        "artifacts": {
            name: {"path": path, "present": payloads[name] is not None}
            for name, path in artifacts.items()
        },
    }


def _print_session_inspection(payload: dict) -> None:
    click.echo("FactorMiner Session")
    click.echo("=" * 60)
    click.echo(f"Output dir: {payload['output_dir']}")
    click.echo(f"Status: {payload['status']}")
    click.echo(f"Library size: {payload['library_size']}")
    click.echo(f"Iterations: {payload['iterations']}")
    if payload.get("yield_rate") is not None:
        click.echo(f"Yield: {payload['yield_rate'] * 100:.1f}%")
    if payload.get("backend"):
        click.echo(f"Backend: {payload['backend']}")
    if payload.get("data_tensor_shape"):
        click.echo(f"Data tensor: {payload['data_tensor_shape']}")
    if payload.get("returns_shape"):
        click.echo(f"Returns: {payload['returns_shape']}")

    if payload["warnings"]:
        click.echo("\nWarnings:")
        for warning in payload["warnings"]:
            click.echo(f"  - {warning}")

    click.echo("\nArtifacts:")
    for name, info in payload["artifacts"].items():
        status = "present" if info["present"] else "missing"
        click.echo(f"  - {name}: {status} ({info['path']})")


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config file (merges with defaults).",
)
@click.option(
    "--gpu/--cpu",
    default=None,
    help="Override evaluation backend. Omit to use the configured backend.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug-level logging.")
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False),
    default="output",
    help="Directory for all output artifacts.",
)
@click.version_option(package_name="factorminer")
@click.pass_context
def main(
    ctx: click.Context,
    config: str | None,
    gpu: bool | None,
    verbose: bool,
    output_dir: str,
) -> None:
    """FactorMiner -- LLM-powered quantitative factor mining."""
    _setup_logging(verbose)

    overrides: dict = {}
    if gpu is True:
        overrides.setdefault("evaluation", {})["backend"] = "gpu"
    elif gpu is False:
        overrides.setdefault("evaluation", {})["backend"] = "numpy"

    try:
        cfg = load_config(config_path=config, overrides=overrides if overrides else None)
    except Exception as e:
        click.echo(f"Error loading config: {e}")
        raise click.Abort()

    # Stash raw YAML for top-level fields like data_path/output_dir.
    raw_config = _load_raw_config_data(config)
    setattr(cfg, "_raw", raw_config)

    if output_dir == "output":
        output_dir = raw_config.get("output_dir", output_dir)

    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose
    ctx.obj["output_dir"] = Path(output_dir)


# ---------------------------------------------------------------------------
# doctor / init-config / session inspect
# ---------------------------------------------------------------------------

@main.command("doctor")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--config",
    "doctor_config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config file.",
)
@click.pass_context
def doctor(ctx: click.Context, json_output: bool, doctor_config: str | None) -> None:
    """Check local install, config, optional dependencies, and output paths."""
    cfg = ctx.obj["config"]
    raw = getattr(cfg, "_raw", {})
    if doctor_config:
        cfg = load_config(config_path=doctor_config)
        raw = _load_raw_config_data(doctor_config)
    output_dir = ctx.obj["output_dir"]
    checks = _doctor_checks(cfg, raw, output_dir)
    payload = {
        "ok": not any(item["status"] == "error" for item in checks),
        "checks": checks,
    }
    if json_output:
        click.echo(json.dumps(_json_safe(payload), indent=2))
    else:
        _print_doctor_report(checks)
    if not payload["ok"]:
        ctx.exit(1)


@main.command("init-config")
@click.argument(
    "path",
    required=False,
    type=click.Path(dir_okay=False),
    default="factorminer.local.yaml",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def init_config(path: str, force: bool) -> None:
    """Write a CPU-safe starter YAML config."""
    output_path = Path(path)
    if output_path.exists() and not force:
        raise click.ClickException(f"{output_path} already exists. Pass --force to overwrite.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(_starter_config(), f, sort_keys=False)
    click.echo(f"Wrote starter config to {output_path}")


@main.group("session")
def session_group() -> None:
    """Inspect FactorMiner session artifacts."""


@session_group.command("inspect")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
def session_inspect(output_dir: str, json_output: bool) -> None:
    """Summarize run artifacts in an output directory."""
    payload = _inspect_session_dir(Path(output_dir))
    if json_output:
        click.echo(json.dumps(_json_safe(payload), indent=2))
    else:
        _print_session_inspection(payload)


# ---------------------------------------------------------------------------
# validate-data
# ---------------------------------------------------------------------------

@main.command("validate-data")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--strict",
    is_flag=True,
    help="Treat warnings as failures.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit machine-readable JSON instead of text.",
)
@click.option(
    "--hdf-key",
    default="data",
    show_default=True,
    help="HDF5 key to read when validating .h5/.hdf5 files.",
)
@click.pass_context
def validate_data(
    ctx: click.Context,
    path: str,
    strict: bool,
    json_output: bool,
    hdf_key: str,
) -> None:
    """Validate a market-data file before mining."""
    from factorminer.data.validation import render_validation_report, validate_market_data

    try:
        report = validate_market_data(path, hdf_key=hdf_key)
    except Exception as exc:  # noqa: BLE001 - surfaced to CLI
        click.echo(f"Validation error: {exc}")
        raise click.Abort() from exc

    if json_output:
        click.echo(json.dumps(report.to_dict(strict=strict), indent=2, sort_keys=True))
    else:
        click.echo(render_validation_report(report, strict=strict))
        click.echo(_render_validation_next_steps(report, ctx.obj["config"], path, hdf_key))

    code = report.exit_code(strict=strict)
    if code != 0:
        ctx.exit(code)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--session-log",
    type=click.Path(exists=True),
    default=None,
    help="Optional session_log.json path.",
)
@click.option(
    "--benchmark",
    "benchmark_paths",
    type=click.Path(exists=True),
    multiple=True,
    help="Optional benchmark JSON path. May be passed multiple times.",
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["markdown", "html"]),
    default="markdown",
    show_default=True,
    help="Static report format.",
)
@click.option(
    "--output",
    "-o",
    "report_output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the report to this path instead of stdout.",
)
def report(
    library_path: str,
    session_log: str | None,
    benchmark_paths: tuple[str, ...],
    report_format: str,
    report_output: str | None,
) -> None:
    """Generate a static report from FactorMiner artifacts."""
    from factorminer.evaluation.report_viewer import generate_report

    rendered = generate_report(
        library_path,
        session_log_source=session_log,
        benchmark_sources=benchmark_paths,
        format=report_format,
        output_path=report_output,
    )

    if report_output is None:
        click.echo(rendered)
    else:
        click.echo(f"Report written to: {report_output}")


# ---------------------------------------------------------------------------
# quickstart
# ---------------------------------------------------------------------------

@main.command("quickstart")
@click.option(
    "--output-dir",
    "quickstart_output_dir",
    type=click.Path(file_okay=False),
    default="/tmp/factorminer-quickstart",
    show_default=True,
    help="Directory for quickstart artifacts.",
)
@click.option("--iterations", "-n", type=int, default=2, show_default=True)
@click.option("--batch-size", "-b", type=int, default=8, show_default=True)
@click.option("--target", "-t", type=int, default=2, show_default=True)
@click.pass_context
def quickstart(
    ctx: click.Context,
    quickstart_output_dir: str,
    iterations: int,
    batch_size: int,
    target: int,
) -> None:
    """Run a mock end-to-end mining session and generate a static report."""
    output_dir = Path(quickstart_output_dir)
    starter = _starter_config()
    starter["output_dir"] = str(output_dir)
    starter["mining"]["target_library_size"] = target
    starter["mining"]["batch_size"] = batch_size
    starter["mining"]["max_iterations"] = iterations
    starter["llm"]["batch_candidates"] = batch_size

    cfg = load_config(overrides=starter)
    setattr(cfg, "_raw", starter)

    original_cfg = ctx.obj["config"]
    original_output_dir = ctx.obj["output_dir"]
    ctx.obj["config"] = cfg
    ctx.obj["output_dir"] = output_dir

    click.echo("Running doctor with mock quickstart settings...")
    checks = _doctor_checks(cfg, starter, output_dir)
    _print_doctor_report(checks)
    if any(item["status"] == "error" for item in checks):
        ctx.obj["config"] = original_cfg
        ctx.obj["output_dir"] = original_output_dir
        raise click.Abort()

    try:
        ctx.invoke(
            mine,
            iterations=iterations,
            batch_size=batch_size,
            target=target,
            resume=None,
            mock=True,
            data_path=None,
        )
    finally:
        ctx.obj["config"] = original_cfg
        ctx.obj["output_dir"] = original_output_dir

    library_path = output_dir / "factor_library.json"
    session_log_path = output_dir / "session_log.json"
    report_path = output_dir / "quickstart_report.html"
    if library_path.exists():
        from factorminer.evaluation.report_viewer import generate_report

        generate_report(
            library_path,
            session_log_source=session_log_path if session_log_path.exists() else None,
            format="html",
            output_path=report_path,
        )
        click.echo(f"Static report written to: {report_path}")
    else:
        click.echo("Quickstart completed without a factor_library.json artifact.")

    click.echo("")
    click.echo("Next real-data commands")
    click.echo("-" * 60)
    click.echo("uv run factorminer validate-data path/to/market_data.csv")
    click.echo("uv run factorminer init-config factorminer.local.yaml")
    click.echo(
        "uv run factorminer -c factorminer.local.yaml -o output-real "
        "mine --data path/to/market_data.csv"
    )


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------

@main.command()
@click.option("--iterations", "-n", type=int, default=None, help="Override max_iterations.")
@click.option("--batch-size", "-b", type=int, default=None, help="Override batch_size.")
@click.option("--target", "-t", type=int, default=None, help="Override target_library_size.")
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from a saved library.")
@click.option("--mock", is_flag=True, help="Use mock data and mock LLM provider (for testing).")
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.pass_context
def mine(
    ctx: click.Context,
    iterations: int | None,
    batch_size: int | None,
    target: int | None,
    resume: str | None,
    mock: bool,
    data_path: str | None,
) -> None:
    """Run a factor mining session."""
    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]

    if iterations is not None:
        cfg.mining.max_iterations = iterations
    if batch_size is not None:
        cfg.mining.batch_size = batch_size
    if target is not None:
        cfg.mining.target_library_size = target

    try:
        cfg.validate()
    except ValueError as e:
        click.echo(f"Configuration error: {e}")
        raise click.Abort()

    click.echo("=" * 60)
    click.echo("FactorMiner -- Mining Session")
    click.echo("=" * 60)
    click.echo(f"  Target library size: {cfg.mining.target_library_size}")
    click.echo(f"  Batch size:          {cfg.mining.batch_size}")
    click.echo(f"  Max iterations:      {cfg.mining.max_iterations}")
    click.echo(f"  IC threshold:        {cfg.mining.ic_threshold}")
    click.echo(f"  Correlation limit:   {cfg.mining.correlation_threshold}")
    click.echo(f"  Output directory:    {output_dir}")
    click.echo("-" * 60)

    # Load data
    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    click.echo(
        f"  Data loaded: {len(dataset.asset_ids)} assets x "
        f"{len(dataset.timestamps)} periods"
    )
    click.echo("  Preparing data tensors...")
    data_tensor = dataset.data_tensor
    returns = dataset.returns

    # Create LLM provider
    llm_provider = _create_llm_provider(cfg, mock)

    # Load existing library for resume
    library = None
    if resume:
        click.echo(f"  Resuming from: {resume}")
        library = _load_library_from_path(resume)

    # Create and configure MiningConfig for the RalphLoop
    mining_config = _build_core_mining_config(cfg, output_dir, mock=mock)
    _attach_runtime_targets(mining_config, dataset)

    # Create and run the Ralph Loop
    from factorminer.core.ralph_loop import RalphLoop

    click.echo("-" * 60)
    click.echo("Starting Ralph Loop...")

    def _progress_callback(iteration: int, stats: dict) -> None:
        """Print progress after each iteration."""
        lib_size = stats.get("library_size", 0)
        admitted = stats.get("admitted", 0)
        yield_rate = stats.get("yield_rate", 0) * 100
        click.echo(
            f"  Iteration {iteration:3d}: "
            f"Library={lib_size}, "
            f"Admitted={admitted}, "
            f"Yield={yield_rate:.1f}%"
        )

    try:
        loop = RalphLoop(
            config=mining_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=llm_provider,
            library=library,
        )
        result_library = loop.run(callback=_progress_callback)
    except KeyboardInterrupt:
        click.echo("\nMining interrupted by user.")
        return
    except Exception as e:
        click.echo(f"Mining error: {e}")
        logger.exception("Mining failed")
        raise click.Abort()

    # Save results
    lib_path = _save_result_library(result_library, output_dir)

    click.echo("=" * 60)
    click.echo(f"Mining complete! Library size: {result_library.size}")
    click.echo(f"Library saved to: {lib_path}")
    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for evaluation.")
@click.option("--period", type=click.Choice(["train", "test", "both"]), default="test", help="Evaluation period.")
@click.option("--top-k", type=int, default=None, help="Evaluate only the top-K factors by IC.")
@click.pass_context
def evaluate(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    period: str,
    top_k: int | None,
) -> None:
    """Evaluate a factor library on historical data."""
    cfg = ctx.obj["config"]
    signal_failure_policy = cfg.evaluation.signal_failure_policy

    click.echo("=" * 60)
    click.echo("FactorMiner -- Factor Evaluation")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    click.echo(f"  Period: {period} | Backend: {cfg.evaluation.backend}")
    click.echo(
        f"  Data: {len(dataset.asset_ids)} assets x {len(dataset.timestamps)} periods"
    )

    artifacts = _recompute_analysis_artifacts(library, dataset, signal_failure_policy)
    failures = _report_artifact_failures(artifacts, header="Evaluation warnings")

    from factorminer.evaluation.runtime import analysis_split_names, select_top_k

    split_names = analysis_split_names(period)
    selection_split = "train" if period == "both" else split_names[0]
    selected = select_top_k(artifacts, selection_split, top_k)
    if not selected:
        click.echo("No factors successfully recomputed for evaluation.")
        if signal_failure_policy == "reject" and failures:
            raise click.Abort()
        raise click.Abort()

    if top_k is not None and top_k < len([a for a in artifacts if a.succeeded]):
        if period == "both":
            click.echo(
                f"  Evaluating top {top_k} factors by train paper IC for train/test comparison"
            )
        else:
            click.echo(f"  Evaluating top {top_k} factors by {selection_split} paper IC")

    for split_name in split_names:
        click.echo("-" * 60)
        click.echo(f"Split: {split_name}")
        _print_recomputed_factor_table(selected, split_name)
        _print_split_summary(selected, split_name)

    if period == "both" and selected:
        click.echo("-" * 60)
        click.echo("Decay summary (train -> test)")
        click.echo(
            f"{'ID':>4s}  {'Name':<35s}  {'Train Paper IC':>14s}  "
            f"{'Test Paper IC':>13s}  {'Delta':>8s}"
        )
        click.echo("-" * 80)
        for artifact in selected:
            train_ic = artifact.split_stats["train"].get(
                "ic_paper_mean",
                artifact.split_stats["train"]["ic_abs_mean"],
            )
            test_ic = artifact.split_stats["test"].get(
                "ic_paper_mean",
                artifact.split_stats["test"]["ic_abs_mean"],
            )
            click.echo(
                f"{artifact.factor_id:4d}  {artifact.name:<35s}  "
                f"{train_ic:10.4f}  {test_ic:9.4f}  {test_ic - train_ic:8.4f}"
            )

    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for combination.")
@click.option(
    "--fit-period",
    type=click.Choice(["train", "test", "both"]),
    default="train",
    help="Split used for top-k selection and model/weight fitting.",
)
@click.option(
    "--eval-period",
    type=click.Choice(["train", "test", "both"]),
    default="test",
    help="Split used to evaluate the combined signal.",
)
@click.option(
    "--method", "-m",
    type=click.Choice(["equal-weight", "ic-weighted", "orthogonal", "all"]),
    default="all",
    help="Factor combination method.",
)
@click.option(
    "--selection", "-s",
    type=click.Choice(["lasso", "stepwise", "xgboost", "none"]),
    default="none",
    help="Factor selection method to run before combination.",
)
@click.option("--top-k", type=int, default=None, help="Select top-K factors before combining.")
@click.pass_context
def combine(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    fit_period: str,
    eval_period: str,
    method: str,
    selection: str,
    top_k: int | None,
) -> None:
    """Run factor combination and selection methods."""
    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]

    click.echo("=" * 60)
    click.echo("FactorMiner -- Factor Combination")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    from factorminer.evaluation.runtime import (
        resolve_split_for_fit_eval,
        select_top_k,
    )

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    artifacts = _recompute_analysis_artifacts(
        library,
        dataset,
        cfg.evaluation.signal_failure_policy,
    )
    failures = _report_artifact_failures(artifacts, header="Combination warnings")

    fit_split = resolve_split_for_fit_eval(fit_period)
    eval_split = resolve_split_for_fit_eval(eval_period)

    selected_artifacts = select_top_k(artifacts, fit_split, top_k)
    if not selected_artifacts:
        click.echo("No factors successfully recomputed for combination.")
        if cfg.evaluation.signal_failure_policy == "reject" and failures:
            raise click.Abort()
        raise click.Abort()

    if top_k is not None and top_k < len([a for a in artifacts if a.succeeded]):
        click.echo(
            f"  Pre-selected top {len(selected_artifacts)} factors by {fit_split} paper IC"
        )

    click.echo(f"  Fit split:  {fit_split}")
    click.echo(f"  Eval split: {eval_split}")
    click.echo(f"  Combining {len(selected_artifacts)} factors")
    click.echo("-" * 60)

    # Run selection if requested
    selected_ids = [artifact.factor_id for artifact in selected_artifacts]
    fit_returns_tn = dataset.get_split(fit_split).returns.T
    fit_factor_signals = {
        artifact.factor_id: artifact.split_signals[fit_split].T
        for artifact in selected_artifacts
    }

    if selection != "none":
        click.echo(f"\n  Running {selection} selection...")
        from factorminer.evaluation.selection import FactorSelector

        selector = FactorSelector()

        try:
            if selection == "lasso":
                results = selector.lasso_selection(fit_factor_signals, fit_returns_tn)
            elif selection == "stepwise":
                results = selector.forward_stepwise(fit_factor_signals, fit_returns_tn)
            elif selection == "xgboost":
                results = selector.xgboost_selection(fit_factor_signals, fit_returns_tn)
            else:
                results = []

            if results:
                selected_ids = [factor_id for factor_id, _ in results]
                click.echo(f"\n  {selection.capitalize()} selection results:")
                click.echo(f"  {'Factor ID':>10s}  {'Score':>10s}")
                click.echo("  " + "-" * 25)
                for fid, score in results[:20]:  # Show top 20
                    click.echo(f"  {fid:10d}  {score:10.4f}")
                click.echo(f"  Total selected: {len(selected_ids)}")
            else:
                click.echo(f"  {selection} selection returned no factors.")
        except ImportError as e:
            click.echo(f"  Selection method '{selection}' requires additional packages: {e}")
        except Exception as e:
            click.echo(f"  Selection error: {e}")
            logger.exception("Selection failed")

    # Run combination methods
    from factorminer.evaluation.combination import FactorCombiner
    from factorminer.evaluation.portfolio import PortfolioBacktester

    combiner = FactorCombiner()
    backtester = PortfolioBacktester()
    artifact_map = _artifact_map_by_id(selected_artifacts)
    eval_factor_signals = {
        factor_id: artifact_map[factor_id].split_signals[eval_split].T
        for factor_id in selected_ids
        if factor_id in artifact_map
    }
    ic_values = {
        factor_id: artifact_map[factor_id].split_stats[fit_split]["ic_mean"]
        for factor_id in eval_factor_signals
    }
    eval_returns_tn = dataset.get_split(eval_split).returns.T

    methods_to_run = []
    if method == "all":
        methods_to_run = ["equal-weight", "ic-weighted", "orthogonal"]
    else:
        methods_to_run = [method]

    for m in methods_to_run:
        click.echo(f"\n  {m.upper()} combination:")
        try:
            if m == "equal-weight":
                composite = combiner.equal_weight(eval_factor_signals)
            elif m == "ic-weighted":
                composite = combiner.ic_weighted(eval_factor_signals, ic_values)
            elif m == "orthogonal":
                composite = combiner.orthogonal(eval_factor_signals)
            else:
                continue

            stats = backtester.quintile_backtest(composite, eval_returns_tn)
            click.echo(f"    IC Mean:      {stats['ic_mean']:.4f}")
            click.echo(f"    Paper IC:     {abs(stats['ic_mean']):.4f}")
            click.echo(f"    ICIR:         {stats['icir']:.4f}")
            click.echo(f"    Long-Short:   {stats['ls_return']:.4f}")
            click.echo(f"    Monotonicity: {stats['monotonicity']:.2f}")
            click.echo(f"    Avg Turnover: {stats['avg_turnover']:.4f}")
        except Exception as e:
            click.echo(f"    Error: {e}")
            logger.exception("Combination method %s failed", m)

    if cfg.research.enabled and cfg.benchmark.mode == "research":
        click.echo("\n  Research model suite:")
        try:
            from factorminer.evaluation.research import run_research_model_suite

            research_reports = run_research_model_suite(
                eval_factor_signals,
                eval_returns_tn,
                cfg.research,
            )
            research_path = output_dir / "research_model_suite.json"
            research_path.write_text(json.dumps(research_reports, indent=2))
            for model_name, report in research_reports.items():
                if not report.get("available", True):
                    click.echo(f"    {model_name}: unavailable ({report.get('error', 'unknown error')})")
                    continue
                click.echo(
                    f"    {model_name}: "
                    f"net IR={report.get('mean_test_net_ir', 0.0):.4f}, "
                    f"ICIR={report.get('mean_test_icir', 0.0):.4f}, "
                    f"stability={report.get('selection_stability', 0.0):.3f}"
                )
            click.echo(f"    Saved: {research_path}")
        except Exception as e:
            click.echo(f"    Research suite error: {e}")
            logger.exception("Research model suite failed")

    click.echo("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for visualization.")
@click.option("--period", type=click.Choice(["train", "test", "both"]), default="test", help="Evaluation split to visualize.")
@click.option("--factor-id", "factor_ids", type=int, multiple=True, help="Specific factor ID(s) to visualize.")
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Top-K factors by split paper IC for set-level plots.",
)
@click.option("--tearsheet", is_flag=True, help="Generate a full factor tear sheet.")
@click.option("--correlation", is_flag=True, help="Plot factor correlation heatmap.")
@click.option("--ic-timeseries", is_flag=True, help="Plot IC time series.")
@click.option("--quintile", is_flag=True, help="Plot quintile returns.")
@click.option("--format", "fmt", type=click.Choice(["png", "pdf", "svg"]), default="png", help="Output format.")
@click.pass_context
def visualize(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    period: str,
    factor_ids: tuple[int, ...],
    top_k: int | None,
    tearsheet: bool,
    correlation: bool,
    ic_timeseries: bool,
    quintile: bool,
    fmt: str,
) -> None:
    """Generate plots and tear sheets for a factor library."""
    output_dir = ctx.obj["output_dir"]
    cfg = ctx.obj["config"]

    click.echo("=" * 60)
    click.echo("FactorMiner -- Visualization")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    # Determine what to plot
    plot_all = not (tearsheet or correlation or ic_timeseries or quintile)
    if plot_all:
        click.echo("No specific plots requested; generating all available.")
        correlation = True
        ic_timeseries = True
        quintile = True

    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"  Output format: {fmt}")
    click.echo(f"  Output dir:    {output_dir}")
    click.echo(f"  Period:        {period}")
    click.echo("-" * 60)

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    artifacts = _recompute_analysis_artifacts(
        library,
        dataset,
        cfg.evaluation.signal_failure_policy,
    )
    failures = _report_artifact_failures(artifacts, header="Visualization warnings")

    from factorminer.evaluation.runtime import (
        analysis_split_names,
        compute_correlation_matrix,
        select_top_k,
    )
    from factorminer.utils.tearsheet import FactorTearSheet
    from factorminer.utils.visualization import (
        plot_correlation_heatmap,
        plot_ic_timeseries,
        plot_quintile_returns,
    )

    split_names = analysis_split_names(period)
    explicit_artifacts = _select_artifacts_for_ids(artifacts, factor_ids)
    if not explicit_artifacts and factor_ids:
        if cfg.evaluation.signal_failure_policy == "reject" and failures:
            raise click.Abort()
        raise click.Abort()

    for split_name in split_names:
        split = dataset.get_split(split_name)
        click.echo(f"  Split: {split_name}")

        if correlation:
            if factor_ids:
                corr_artifacts = explicit_artifacts
            else:
                corr_artifacts = select_top_k(artifacts, split_name, top_k)

            if corr_artifacts:
                click.echo("    Generating correlation heatmap...")
                corr_matrix = compute_correlation_matrix(corr_artifacts, split_name)
                save_path = _analysis_output_path(output_dir, "correlation_heatmap", split_name, fmt)
                plot_correlation_heatmap(
                    corr_matrix,
                    [artifact.name[:20] for artifact in corr_artifacts],
                    title=f"Factor Correlation Heatmap ({split_name})",
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")
            else:
                click.echo("    Skipped: no successfully recomputed factors for correlation heatmap.")

        factor_artifacts = explicit_artifacts
        if not factor_ids and (ic_timeseries or quintile or tearsheet):
            factor_artifacts = select_top_k(artifacts, split_name, 1)
            if factor_artifacts:
                click.echo(
                    f"    Defaulted to factor #{factor_artifacts[0].factor_id} "
                    f"{factor_artifacts[0].name} for factor-specific plots."
                )

        if ic_timeseries:
            click.echo("    Generating IC time series plot(s)...")
            for artifact in factor_artifacts:
                stats = artifact.split_stats[split_name]
                dates = [str(ts)[:10] for ts in split.timestamps]
                save_path = _analysis_output_path(
                    output_dir,
                    f"ic_timeseries_factor_{artifact.factor_id}",
                    split_name,
                    fmt,
                )
                plot_ic_timeseries(
                    stats["ic_series"],
                    dates,
                    title=f"{artifact.name} IC Time Series ({split_name})",
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")

        if quintile:
            click.echo("    Generating quintile return plot(s)...")
            for artifact in factor_artifacts:
                stats = artifact.split_stats[split_name]
                save_path = _analysis_output_path(
                    output_dir,
                    f"quintile_returns_factor_{artifact.factor_id}",
                    split_name,
                    fmt,
                )
                plot_quintile_returns(
                    {
                        f"Q{i}": stats[f"Q{i}"] for i in range(1, 6)
                    }
                    | {
                        "long_short": stats["long_short"],
                        "monotonicity": stats["monotonicity"],
                    },
                    title=f"{artifact.name} Quintile Returns ({split_name})",
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")

        if tearsheet:
            click.echo("    Generating tear sheet(s)...")
            ts = FactorTearSheet()
            dates = [str(ts_)[:10] for ts_ in split.timestamps]
            for artifact in factor_artifacts:
                save_path = _analysis_output_path(
                    output_dir,
                    f"tearsheet_factor_{artifact.factor_id}",
                    split_name,
                    fmt,
                )
                ts.generate(
                    factor_id=artifact.factor_id,
                    factor_name=artifact.name,
                    formula=artifact.formula,
                    signals=artifact.split_signals[split_name],
                    returns=split.returns,
                    dates=dates,
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")

    click.echo("=" * 60)
    click.echo("Visualization complete.")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@main.command(name="export")
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "csv", "formulas"]),
    default="json",
    help="Export format.",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.pass_context
def export_cmd(ctx: click.Context, library_path: str, fmt: str, output: str | None) -> None:
    """Export a factor library to various formats."""
    output_dir = ctx.obj["output_dir"]

    click.echo("=" * 60)
    click.echo("FactorMiner -- Export")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    # Determine output path
    if output is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if fmt == "formulas":
            output = str(output_dir / "library_formulas.txt")
        else:
            output = str(output_dir / f"library.{fmt}")

    click.echo(f"  Format:  {fmt}")
    click.echo(f"  Output:  {output}")
    click.echo("-" * 60)

    try:
        from factorminer.core.library_io import export_csv, export_formulas, save_library

        if fmt == "json":
            # save_library expects base path without extension
            out_path = Path(output)
            if out_path.suffix == ".json":
                base = out_path.with_suffix("")
            else:
                base = out_path
            save_library(library, base, save_signals=False)
            click.echo(f"  Exported {library.size} factors to {base}.json")

        elif fmt == "csv":
            export_csv(library, output)
            click.echo(f"  Exported {library.size} factors to {output}")

        elif fmt == "formulas":
            export_formulas(library, output)
            click.echo(f"  Exported {library.size} formulas to {output}")

    except Exception as e:
        click.echo(f"Export error: {e}")
        logger.exception("Export failed")
        raise click.Abort()

    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@main.group()
def benchmark() -> None:
    """Run strict paper/research benchmark workflows."""


def _benchmark_common_options(fn):
    fn = click.option(
        "--data",
        "data_path",
        type=click.Path(exists=True),
        default=None,
        help="Path to market data file.",
    )(fn)
    fn = click.option(
        "--mock",
        is_flag=True,
        help="Use mock data for benchmark execution.",
    )(fn)
    fn = click.option(
        "--factor-miner-library",
        type=click.Path(exists=True),
        default=None,
        help="Optional saved library for the FactorMiner baseline.",
    )(fn)
    fn = click.option(
        "--factor-miner-no-memory-library",
        type=click.Path(exists=True),
        default=None,
        help="Optional saved library for the FactorMiner No Memory baseline.",
    )(fn)
    return click.pass_context(fn)


@benchmark.command("table1")
@click.option("--baseline", "baselines", multiple=True, help="Restrict to one or more baseline ids.")
@_benchmark_common_options
def benchmark_table1(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
    baselines: tuple[str, ...],
) -> None:
    """Run the Top-K freeze benchmark across configured universes."""
    from factorminer.benchmark.runtime import run_table1_benchmark

    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]
    payload = run_table1_benchmark(
        cfg,
        output_dir,
        data_path=data_path,
        mock=mock,
        baseline_names=list(baselines) if baselines else None,
        factor_miner_library_path=factor_miner_library,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library,
    )
    _print_benchmark_summary("FactorMiner -- Benchmark Table 1", payload)


@benchmark.command("ablation-memory")
@_benchmark_common_options
def benchmark_ablation_memory(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
) -> None:
    """Run the experience-memory ablation benchmark."""
    from factorminer.benchmark.runtime import run_ablation_memory_benchmark

    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]
    payload = run_ablation_memory_benchmark(
        cfg,
        output_dir,
        data_path=data_path,
        mock=mock,
        factor_miner_library_path=factor_miner_library,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library,
    )
    _print_benchmark_summary("FactorMiner -- Memory Ablation", payload)


@benchmark.command("ablation-strategy")
@click.option("--baseline", default="factor_miner", help="Runtime baseline id to evaluate.")
@_benchmark_common_options
def benchmark_ablation_strategy(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
    baseline: str,
) -> None:
    """Run runtime ablations across memory policy, dependence metric, and backend."""
    from factorminer.benchmark.runtime import run_ablation_strategy_benchmark

    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]
    payload = run_ablation_strategy_benchmark(
        cfg,
        output_dir,
        baseline=baseline,
        data_path=data_path,
        mock=mock,
    )
    _print_benchmark_summary("FactorMiner -- Strategy Ablation", payload)


@benchmark.command("cost-pressure")
@click.option("--baseline", default="factor_miner", help="Baseline id to evaluate.")
@_benchmark_common_options
def benchmark_cost_pressure(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
    baseline: str,
) -> None:
    """Run transaction-cost pressure testing."""
    from factorminer.benchmark.runtime import run_cost_pressure_benchmark

    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]
    payload = run_cost_pressure_benchmark(
        cfg,
        output_dir,
        baseline=baseline,
        data_path=data_path,
        mock=mock,
        factor_miner_library_path=factor_miner_library,
    )
    _print_benchmark_summary("FactorMiner -- Cost Pressure", payload)


@benchmark.command("efficiency")
@click.pass_context
def benchmark_efficiency(ctx: click.Context) -> None:
    """Run operator-level and factor-level efficiency benchmarks."""
    from factorminer.benchmark.runtime import run_efficiency_benchmark

    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]
    payload = run_efficiency_benchmark(cfg, output_dir)
    _print_benchmark_summary("FactorMiner -- Efficiency Benchmark", payload)


@benchmark.command("suite")
@_benchmark_common_options
def benchmark_suite(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
) -> None:
    """Run the full benchmark suite."""
    from factorminer.benchmark.runtime import run_benchmark_suite

    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]
    payload = run_benchmark_suite(
        cfg,
        output_dir,
        data_path=data_path,
        mock=mock,
        factor_miner_library_path=factor_miner_library,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library,
    )
    _print_benchmark_summary("FactorMiner -- Benchmark Suite", payload)


# ---------------------------------------------------------------------------
# helix
# ---------------------------------------------------------------------------

@main.command()
@click.option("--iterations", "-n", type=int, default=None, help="Override max_iterations.")
@click.option("--batch-size", "-b", type=int, default=None, help="Override batch_size.")
@click.option("--target", "-t", type=int, default=None, help="Override target_library_size.")
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from a saved library.")
@click.option("--causal/--no-causal", default=None, help="Enable/disable causal validation.")
@click.option("--regime/--no-regime", default=None, help="Enable/disable regime-conditional evaluation.")
@click.option("--debate/--no-debate", default=None, help="Enable/disable multi-specialist debate generation.")
@click.option("--canonicalize/--no-canonicalize", default=None, help="Enable/disable SymPy canonicalization.")
@click.option("--mock", is_flag=True, help="Use mock data and mock LLM provider (for testing).")
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.pass_context
def helix(
    ctx: click.Context,
    iterations: int | None,
    batch_size: int | None,
    target: int | None,
    resume: str | None,
    causal: bool | None,
    regime: bool | None,
    debate: bool | None,
    canonicalize: bool | None,
    mock: bool,
    data_path: str | None,
) -> None:
    """Run the enhanced Helix Loop with Phase 2 features."""
    cfg = ctx.obj["config"]

    if iterations is not None:
        cfg.mining.max_iterations = iterations
    if batch_size is not None:
        cfg.mining.batch_size = batch_size
    if target is not None:
        cfg.mining.target_library_size = target

    if causal is not None:
        cfg.phase2.causal.enabled = causal
    if regime is not None:
        cfg.phase2.regime.enabled = regime
    if debate is not None:
        cfg.phase2.debate.enabled = debate
    if canonicalize is not None:
        if canonicalize:
            cfg.phase2.helix.enabled = True
        cfg.phase2.helix.enable_canonicalization = canonicalize

    try:
        cfg.validate()
    except ValueError as e:
        click.echo(f"Configuration error: {e}")
        raise click.Abort()

    output_dir = ctx.obj["output_dir"]
    enabled_features = _active_phase2_features(cfg)

    click.echo("HelixFactor Phase 2 mining engine.")
    click.echo(f"  Target: {cfg.mining.target_library_size} | "
               f"Batch: {cfg.mining.batch_size} | "
               f"Max iterations: {cfg.mining.max_iterations}")
    click.echo(f"  Output directory: {output_dir}")

    if enabled_features:
        click.echo(f"  Active Phase 2 features: {', '.join(enabled_features)}")
    else:
        click.echo("  No Phase 2 features enabled. Configure phase2.* in your config to enable features.")

    if resume:
        click.echo(f"  Resuming from: {resume}")

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    click.echo("  Preparing data tensors...")
    data_tensor = dataset.data_tensor
    returns = dataset.returns
    llm_provider = _create_llm_provider(cfg, mock)

    library = None
    if resume:
        library = _load_library_from_path(resume)

    mining_config = _build_core_mining_config(cfg, output_dir, mock=mock)
    _attach_runtime_targets(mining_config, dataset)
    phase2_configs = _build_phase2_runtime_configs(cfg)
    volume = _extract_capacity_volume(data_tensor) if cfg.phase2.capacity.enabled else None

    from factorminer.core.helix_loop import HelixLoop

    click.echo("-" * 60)
    click.echo("Starting Helix Loop...")

    def _progress_callback(iteration: int, stats: dict) -> None:
        message = (
            f"  Iteration {iteration:3d}: "
            f"Library={stats.get('library_size', 0)}, "
            f"Admitted={stats.get('admitted', 0)}, "
            f"Yield={stats.get('yield_rate', 0) * 100:.1f}%"
        )
        canon_removed = stats.get("canonical_duplicates_removed", 0)
        phase2_rejections = stats.get("phase2_rejections", 0)
        if canon_removed:
            message += f", CanonDupes={canon_removed}"
        if phase2_rejections:
            message += f", Phase2Reject={phase2_rejections}"
        click.echo(message)

    try:
        loop = HelixLoop(
            config=mining_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=llm_provider,
            library=library,
            debate_config=phase2_configs["debate_config"],
            enable_knowledge_graph=(
                cfg.phase2.helix.enabled and cfg.phase2.helix.enable_knowledge_graph
            ),
            enable_embeddings=(
                cfg.phase2.helix.enabled and cfg.phase2.helix.enable_embeddings
            ),
            enable_auto_inventor=cfg.phase2.auto_inventor.enabled,
            auto_invention_interval=cfg.phase2.auto_inventor.invention_interval,
            canonicalize=(
                cfg.phase2.helix.enabled and cfg.phase2.helix.enable_canonicalization
            ),
            forgetting_lambda=cfg.phase2.helix.forgetting_lambda,
            causal_config=phase2_configs["causal_config"],
            regime_config=phase2_configs["regime_config"],
            capacity_config=phase2_configs["capacity_config"],
            significance_config=phase2_configs["significance_config"],
            volume=volume,
        )
        result_library = loop.run(callback=_progress_callback)
    except KeyboardInterrupt:
        click.echo("\nHelix mining interrupted by user.")
        return
    except Exception as e:
        click.echo(f"Helix mining error: {e}")
        logger.exception("Helix loop failed")
        raise click.Abort()

    lib_path = _save_result_library(result_library, output_dir)

    click.echo("=" * 60)
    click.echo(f"Helix mining complete! Library size: {result_library.size}")
    click.echo(f"Library saved to: {lib_path}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
