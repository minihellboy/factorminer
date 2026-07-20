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
        # Caching/cascade toggles are no-ops under mock; accept signature parity.
        return MockProvider(prompt_cache=bool(getattr(cfg.llm, "prompt_cache", True)))

    llm_cfg = cfg.llm
    llm_config: dict = {
        "provider": llm_cfg.provider,
        "model": llm_cfg.model,
        "prompt_cache": bool(getattr(llm_cfg, "prompt_cache", True)),
        "timeout_s": float(getattr(llm_cfg, "timeout_s", 120.0)),
    }
    # Use api_key from config if set (frontier providers only)
    raw_llm = {}
    if hasattr(cfg, "_raw") and isinstance(cfg._raw.get("llm"), dict):
        raw_llm = cfg._raw["llm"]
    if raw_llm.get("api_key"):
        llm_config["api_key"] = raw_llm["api_key"]

    # Optional base_url from local YAML only (SSRF-sensitive).
    base_url = getattr(llm_cfg, "base_url", None) or raw_llm.get("base_url")
    if base_url:
        llm_config["base_url"] = base_url

    # Cascade: prefer nested raw llm.cascade block; fall back to flat fields.
    cascade_raw = raw_llm.get("cascade") if isinstance(raw_llm.get("cascade"), dict) else {}
    cascade_enabled = bool(
        cascade_raw.get("enabled", getattr(llm_cfg, "cascade_enabled", False))
    )
    if cascade_enabled:
        llm_config["cascade"] = {
            "enabled": True,
            "draft_provider": cascade_raw.get(
                "draft_provider",
                getattr(llm_cfg, "cascade_draft_provider", "openai_compatible"),
            ),
            "draft_model": cascade_raw.get(
                "draft_model",
                getattr(llm_cfg, "cascade_draft_model", "llama3.2"),
            ),
            # SECURITY (SSRF): draft_base_url from local YAML/config only.
            "draft_base_url": cascade_raw.get(
                "draft_base_url",
                getattr(
                    llm_cfg,
                    "cascade_draft_base_url",
                    "http://127.0.0.1:11434/v1",
                ),
            ),
            "draft_api_key": cascade_raw.get(
                "draft_api_key",
                cascade_raw.get("local_api_key", "local"),
            ),
            "timeout_s": cascade_raw.get(
                "timeout_s",
                getattr(llm_cfg, "cascade_timeout_s", 60.0),
            ),
            "escalate_on_parse_failure": cascade_raw.get(
                "escalate_on_parse_failure",
                getattr(llm_cfg, "cascade_escalate_on_parse_failure", True),
            ),
        }

    click.echo(
        f"Using LLM provider: {llm_cfg.provider}/{llm_cfg.model}"
        f" (prompt_cache={llm_config['prompt_cache']}"
        f", cascade={cascade_enabled})"
    )
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
    mining_cfg.model_co_optimize = getattr(cfg, "_raw", {}).get("model_co_optimize")
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


def _run_session_sensitivity(output_dir: Path, factor_ref: str) -> dict:
    """Run offline formula sensitivity for one library factor using mock data."""
    import numpy as np

    from factorminer.evaluation.formula_sensitivity import analyze_formula_sensitivity

    library = _read_json(output_dir / "factor_library.json") or {}
    factors = library.get("factors") if isinstance(library, dict) else None
    if not isinstance(factors, list):
        return {"error": "factor_library.json missing or has no factors", "factor_ref": factor_ref}

    target = None
    for factor in factors:
        if not isinstance(factor, dict):
            continue
        if str(factor.get("id", "")) == str(factor_ref) or str(factor.get("name", "")) == str(
            factor_ref
        ):
            target = factor
            break
    if target is None:
        return {"error": f"factor not found: {factor_ref}", "factor_ref": factor_ref}

    formula = str(target.get("formula", "") or "")
    if not formula:
        return {"error": "factor has empty formula", "factor_ref": factor_ref}

    # Deterministic offline panel — never hits the network.
    rng = np.random.default_rng(42)
    m, t = 20, 60
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, (m, t)), axis=1)
    open_ = close + rng.normal(0, 0.1, (m, t))
    high = np.maximum(close, open_) + np.abs(rng.normal(0, 0.2, (m, t)))
    low = np.minimum(close, open_) - np.abs(rng.normal(0, 0.2, (m, t)))
    low = np.maximum(low, 1.0)
    volume = np.abs(rng.normal(1e6, 1e5, (m, t)))
    vwap = (high + low + close) / 3.0
    amt = volume * vwap
    returns = np.zeros((m, t))
    returns[:, 1:] = np.diff(close, axis=1) / close[:, :-1]
    data = {
        "$open": open_,
        "$high": high,
        "$low": low,
        "$close": close,
        "$volume": volume,
        "$amt": amt,
        "$vwap": vwap,
        "$returns": returns,
    }
    result = analyze_formula_sensitivity(
        formula,
        data,
        returns,
        factor_name=str(target.get("name", factor_ref)),
    )
    return result.to_dict()



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


def _load_lifecycle_telemetry(output_dir: Path) -> dict:
    """Load factor_lifecycle.jsonl (if present) and summarize mining telemetry."""
    from factorminer.architecture.lifecycle import FactorLifecycleStore

    store = FactorLifecycleStore.load(output_dir)
    return store.telemetry_summary()


def _print_telemetry_summary(telemetry: dict) -> None:
    click.echo("\nMining Telemetry")
    click.echo("=" * 60)
    click.echo(f"Iterations tracked: {len(telemetry['iterations'])}")
    click.echo(f"Total candidates:   {telemetry['total_candidates']}")
    click.echo(f"Total rejected:     {telemetry['total_rejected']}")
    click.echo(f"Overall rejection rate: {telemetry['overall_rejection_rate'] * 100:.1f}%")

    if telemetry["rejection_reason_totals"]:
        click.echo("\nRejection reasons:")
        for reason, count in sorted(
            telemetry["rejection_reason_totals"].items(), key=lambda item: -item[1]
        ):
            click.echo(f"  - {reason:<20} {count:>5}")

    if telemetry["per_iteration"]:
        click.echo("\nPer-iteration:")
        click.echo(
            f"  {'Iter':>5}  {'Cand':>5}  {'Parse':>6}  {'ICScr':>6}  "
            f"{'Dup':>4}  {'Corr':>5}  {'Admit':>6}  {'Reject%':>8}"
        )
        for row in telemetry["per_iteration"]:
            click.echo(
                f"  {row['iteration']:>5}  {row['candidates_seen']:>5}  {row['parse_errors']:>6}  "
                f"{row['ic_screen_rejected']:>6}  {row['duplicate_rejected']:>4}  "
                f"{row['correlation_rejected']:>5}  {row['admitted']:>6}  "
                f"{row['rejection_rate'] * 100:>7.1f}%"
            )


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
@click.option(
    "--telemetry",
    is_flag=True,
    help=(
        "Include per-round mining telemetry (parse errors, duplicates, "
        "rejection reasons) parsed from factor_lifecycle.jsonl."
    ),
)
@click.option(
    "--sensitivity",
    "sensitivity_factor_id",
    default=None,
    help=(
        "Run formula AST sensitivity/ablation for a factor id or name from the "
        "session library (uses mock panel data; offline/reproducible)."
    ),
)
def session_inspect(
    output_dir: str,
    json_output: bool,
    telemetry: bool,
    sensitivity_factor_id: str | None,
) -> None:
    """Summarize run artifacts in an output directory."""
    output_path = Path(output_dir)
    payload = _inspect_session_dir(output_path)
    telemetry_payload = _load_lifecycle_telemetry(output_path) if telemetry else None
    if telemetry_payload is not None:
        payload["telemetry"] = telemetry_payload

    sensitivity_payload = None
    if sensitivity_factor_id:
        sensitivity_payload = _run_session_sensitivity(output_path, sensitivity_factor_id)
        payload["sensitivity"] = sensitivity_payload

    if json_output:
        click.echo(json.dumps(_json_safe(payload), indent=2))
    else:
        _print_session_inspection(payload)
        if telemetry_payload is not None:
            _print_telemetry_summary(telemetry_payload)
        if sensitivity_payload is not None:
            from factorminer.utils.tearsheet import format_sensitivity_panel

            click.echo("")
            click.echo(format_sensitivity_panel(sensitivity_payload))



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
# resample-data
# ---------------------------------------------------------------------------

@main.command("resample-data")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
@click.option("--rule", default="10min", show_default=True, help="Pandas resample rule.")
@click.option(
    "--hdf-key",
    default="data",
    show_default=True,
    help="HDF5 key to read/write for .h5/.hdf5 files.",
)
def resample_data(input_path: str, output_path: str, rule: str, hdf_key: str) -> None:
    """Resample canonical OHLCV market data, e.g. Binance 5m bars to 10min bars."""
    from factorminer.data.loader import load_market_data, resample_market_data

    source = load_market_data(input_path, hdf_key=hdf_key)
    resampled = resample_market_data(source, rule=rule)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    if suffix == ".csv":
        resampled.to_csv(output, index=False)
    elif suffix in {".parquet", ".pq"}:
        resampled.to_parquet(output, index=False)
    elif suffix in {".h5", ".hdf5"}:
        resampled.to_hdf(output, key=hdf_key, index=False)
    else:
        raise click.ClickException(
            "Could not infer output format. Use .csv, .parquet, .pq, .h5, or .hdf5."
        )

    click.echo("FactorMiner -- Data Resample")
    click.echo("=" * 60)
    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Rule:   {rule}")
    click.echo(
        f"Rows:   {len(source)} -> {len(resampled)} | "
        f"Assets: {source['asset_id'].nunique()} -> {resampled['asset_id'].nunique()}"
    )


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
@click.option(
    "--mrm-pack/--no-mrm-pack",
    default=False,
    show_default=True,
    help=(
        "Include an MRM validation pack (model inventory, conceptual soundness, "
        "outcomes analysis, ongoing monitoring). Evidence for a qualified reviewer "
        "only — not a compliance determination."
    ),
)
@click.option(
    "--attest-rationale",
    "attest_factor_ids",
    multiple=True,
    type=str,
    help=(
        "Human attestation: mark economic rationale for this factor id/name as "
        "attested. Repeatable. Never set automatically by generation code."
    ),
)
def report(
    library_path: str,
    session_log: str | None,
    benchmark_paths: tuple[str, ...],
    report_format: str,
    report_output: str | None,
    mrm_pack: bool,
    attest_factor_ids: tuple[str, ...],
) -> None:
    """Generate a static report from FactorMiner artifacts."""
    import json
    from pathlib import Path

    from factorminer.core.provenance import attest_economic_rationale
    from factorminer.evaluation.report_viewer import generate_report

    # Optional human attestation mutates a working copy of the library JSON only
    # when writing an output report path alongside --attest-rationale.
    library_source: str | Path | dict = library_path
    if attest_factor_ids:
        path = Path(library_path)
        if path.is_dir():
            path = path / "factor_library.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        wanted = {str(x) for x in attest_factor_ids}
        for factor in payload.get("factors", []):
            fid = str(factor.get("id", ""))
            name = str(factor.get("name", ""))
            if fid not in wanted and name not in wanted:
                continue
            prov = factor.setdefault("provenance", {})
            rationale = prov.get("economic_rationale") or factor.get("economic_rationale") or {}
            prov["economic_rationale"] = attest_economic_rationale(rationale, attestor="cli-human")
        library_source = payload
        if report_output:
            attested_path = Path(report_output).with_suffix(".attested_library.json")
            attested_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            click.echo(f"Attested library snapshot written to: {attested_path}")

    rendered = generate_report(
        library_source,
        session_log_source=session_log,
        benchmark_sources=benchmark_paths,
        format=report_format,
        output_path=report_output,
        include_mrm_pack=mrm_pack,
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
    type=click.Choice(["equal-weight", "ic-weighted", "orthogonal", "temporal-reweight", "all"]),
    default="all",
    help="Factor combination method.",
)
@click.option(
    "--lookback",
    type=int,
    default=60,
    help="Trailing window size (periods) for --method temporal-reweight.",
)
@click.option(
    "--rebalance-every",
    type=int,
    default=20,
    help="Periods between weight recomputations for --method temporal-reweight.",
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
    lookback: int,
    rebalance_every: int,
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
        methods_to_run = ["equal-weight", "ic-weighted", "orthogonal", "temporal-reweight"]
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
            elif m == "temporal-reweight":
                composite = combiner.temporal_reweight(
                    eval_factor_signals,
                    eval_returns_tn,
                    lookback=lookback,
                    rebalance_every=rebalance_every,
                    method="ic_weighted",
                )
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
# portfolio-construct
# ---------------------------------------------------------------------------

@main.command("portfolio-construct")
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for portfolio construction.")
@click.option(
    "--method", "-m",
    type=click.Choice(["hrp", "risk_parity", "cvar"]),
    default="hrp",
    help="Risk-based portfolio construction method.",
)
@click.option(
    "--top-k", type=int, default=None,
    help="Select top-K factors (by paper IC on the chosen split) before constructing the portfolio.",
)
@click.option(
    "--alpha", type=float, default=0.95,
    help="CVaR confidence level (only used with --method cvar).",
)
@click.option(
    "--period",
    type=click.Choice(["train", "test", "both"]),
    default="test",
    help="Split used to build per-factor return proxies and construct the portfolio.",
)
@click.pass_context
def portfolio_construct(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    method: str,
    top_k: int | None,
    alpha: float,
    period: str,
) -> None:
    """Construct risk-based portfolio weights over a factor library's strategies.

    Each selected factor's own quintile long-short return series is used as
    an asset-level return proxy; HRP / naive risk parity / CVaR-optimal
    weights are then computed across those proxies (research artifact only,
    not a trade recommendation).
    """
    cfg = ctx.obj["config"]

    click.echo("=" * 60)
    click.echo("FactorMiner -- Risk-Based Portfolio Construction")
    click.echo("=" * 60)

    library = _load_library_from_path(library_path)

    from factorminer.evaluation.runtime import resolve_split_for_fit_eval, select_top_k

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
    _report_artifact_failures(artifacts, header="Portfolio construction warnings")

    split = resolve_split_for_fit_eval(period)
    selected_artifacts = select_top_k(artifacts, split, top_k)
    if len(selected_artifacts) < 2:
        click.echo("Need at least 2 successfully recomputed factors for portfolio construction.")
        raise click.Abort()

    click.echo(f"  Split:   {split}")
    click.echo(f"  Method:  {method}")
    click.echo(f"  Assets (factor strategies): {len(selected_artifacts)}")
    click.echo("-" * 60)

    from factorminer.evaluation.portfolio import PortfolioBacktester
    from factorminer.evaluation.risk_portfolio import RiskPortfolioConfig, construct_portfolio

    backtester = PortfolioBacktester()
    split_returns_tn = dataset.get_split(split).returns.T

    asset_ids = []
    return_series = []
    for artifact in selected_artifacts:
        signal_tn = artifact.split_signals[split].T
        stats = backtester.quintile_backtest(signal_tn, split_returns_tn)
        asset_ids.append(artifact.factor_id)
        return_series.append(stats["ls_net_series"])

    returns_matrix = np.column_stack(return_series)
    valid_mask = np.all(np.isfinite(returns_matrix), axis=1)
    returns_matrix = returns_matrix[valid_mask]
    if returns_matrix.shape[0] < 2:
        click.echo("Not enough overlapping valid periods across selected factors.")
        raise click.Abort()

    config = RiskPortfolioConfig(cvar_alpha=alpha)
    try:
        result = construct_portfolio(
            returns_matrix, method=method, asset_ids=asset_ids, config=config
        )
    except Exception as e:
        click.echo(f"Portfolio construction failed: {e}")
        logger.exception("Portfolio construction failed")
        raise click.Abort()

    click.echo(f"  {'Factor ID':>10s}  {'Weight':>8s}")
    click.echo("  " + "-" * 22)
    for factor_id, weight in zip(result.asset_ids, result.weights):
        click.echo(f"  {factor_id:10d}  {weight:8.4f}")

    click.echo("-" * 60)
    click.echo(f"  Method:         {result.method}")
    click.echo(f"  Realized vol:   {result.realized_vol:.6f}")
    click.echo(f"  Realized CVaR:  {result.realized_cvar:.6f}")
    click.echo(f"  Effective N:    {result.effective_n:.2f} (of {len(result.asset_ids)} assets)")
    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# crowding
# ---------------------------------------------------------------------------

@main.command("crowding")
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for crowding diagnostics.")
@click.option(
    "--fixture",
    "fixture_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Offline Ken French-format CSV fixture (default: bundled FF3 fixture).",
)
@click.option(
    "--fetch-consensus/--no-fetch-consensus",
    default=False,
    show_default=True,
    help="Fetch live Ken French panel over HTTPS (fail-closed). Default uses fixture.",
)
@click.option("--top-k", type=int, default=None, help="Score only the top-K factors by IC.")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--window",
    type=int,
    default=63,
    show_default=True,
    help="Rolling window for Lou-Polk CoMetric.",
)
@click.option(
    "--cometric-residual-mode",
    "cometric_residual_mode",
    type=click.Choice(["cross_sectional", "factor_regression"]),
    default="cross_sectional",
    show_default=True,
    help=(
        "CoMetric residualization: 'cross_sectional' (fast, no external data) "
        "or 'factor_regression' (Lou & Polk's actual FF3-regression residuals; "
        "requires the consensus panel to be non-empty, falls back with a "
        "warning otherwise)."
    ),
)
@click.pass_context
def crowding_cmd(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    fixture_path: str | None,
    fetch_consensus: bool,
    top_k: int | None,
    json_output: bool,
    window: int,
    cometric_residual_mode: str,
) -> None:
    """Score library factors for consensus-overlap / CoMetric crowding risk.

    Research risk annotations only — not a trade timer or mining objective.
    Composes consensus novelty, Lou-Polk CoMetric, and hyperbolic decay
    taxonomy from evaluation/decay.py.
    """
    from factorminer.evaluation.crowding import (
        ConsensusFactorPanel,
        CrowdingConfig,
        score_factor_crowding,
    )

    cfg = ctx.obj["config"]
    signal_failure_policy = cfg.evaluation.signal_failure_policy

    click.echo("=" * 60)
    click.echo("FactorMiner -- Factor Crowding Diagnostics")
    click.echo("=" * 60)

    library = _load_library_from_path(library_path)

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    crowding_cfg = CrowdingConfig(
        cometric_window=window, cometric_residual_mode=cometric_residual_mode
    )
    if fetch_consensus:
        panel = ConsensusFactorPanel.fetch(config=crowding_cfg)
        click.echo(f"  Consensus panel: fetch ({panel.source}) factors={panel.factor_names}")
    else:
        panel = ConsensusFactorPanel.from_fixture(fixture_path, config=crowding_cfg)
        click.echo(f"  Consensus panel: fixture factors={panel.factor_names}")

    if panel.empty:
        click.echo(
            "  WARNING: consensus panel empty (fail-closed). "
            "Overlap scores will be unavailable; CoMetric still runs."
        )

    artifacts = _recompute_analysis_artifacts(library, dataset, signal_failure_policy)
    failures = _report_artifact_failures(artifacts, header="Crowding warnings")
    from factorminer.evaluation.runtime import select_top_k

    selected = select_top_k(artifacts, "test", top_k)
    if not selected:
        click.echo("No factors successfully recomputed for crowding.")
        if signal_failure_policy == "reject" and failures:
            raise click.Abort()
        raise click.Abort()

    ret_full = np.asarray(dataset.returns, dtype=np.float64)

    rows: list[dict] = []
    for artifact in selected:
        signals = artifact.signals_full
        if signals is None:
            signals = artifact.split_signals.get("test") or artifact.split_signals.get("full")
        if signals is None:
            continue

        sig = np.asarray(signals, dtype=np.float64)
        ret = ret_full
        # Align returns time axis to signals when using a split.
        if sig.shape != ret.shape and sig.ndim == 2 and ret.ndim == 2:
            if sig.shape[0] == ret.shape[0] and sig.shape[1] < ret.shape[1]:
                split = dataset.splits.get("test") or dataset.splits.get("full")
                if split is not None:
                    ret = np.asarray(split.returns, dtype=np.float64)
            elif sig.shape == ret.T.shape:
                ret = ret.T

        score = score_factor_crowding(
            signals=sig,
            returns=ret,
            panel=panel,
            formula=artifact.formula or "",
            factor_id=str(artifact.factor_id),
            config=crowding_cfg,
        )
        rows.append(score.to_dict())

    if json_output:
        click.echo(json.dumps({"crowding": rows}, indent=2, default=str))
        return

    click.echo("-" * 60)
    click.echo(
        f"{'ID':>6s}  {'Label':<28s}  {'max|ρ|':>7s}  {'CoMOM':>6s}  "
        f"{'NovMod':>6s}  Detail"
    )
    click.echo("-" * 100)
    for row in rows:
        cons = row.get("consensus") or {}
        com = row.get("cometric") or {}
        max_rho = cons.get("max_abs_rho", 0.0) if cons.get("available") else float("nan")
        comom = com.get("comom", 0.0) if com.get("available") else float("nan")
        click.echo(
            f"{str(row.get('factor_id', '')):>6s}  "
            f"{row.get('composite_label', ''):<28s}  "
            f"{max_rho:7.3f}  {comom:6.3f}  "
            f"{row.get('novelty_modulation', 0.0):6.3f}  "
            f"{(row.get('rationale') or '')[:60]}"
        )
    click.echo("=" * 60)
    click.echo(f"Scored {len(rows)} factor(s). Research risk labels only.")


# ---------------------------------------------------------------------------
# jump-worth (Hypothesis-Redundancy geometric gate)
# ---------------------------------------------------------------------------

@main.command("jump-worth")
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--threshold",
    type=float,
    default=0.45,
    show_default=True,
    help="Recommend LLM jump when jump_worth >= threshold.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def jump_worth_cmd(
    ctx: click.Context,
    library_path: str,
    threshold: float,
    json_output: bool,
) -> None:
    """Assess whether a non-local LLM jump is worth its cost for this library.

    Implements the Hypothesis-Redundancy geometric gate (arXiv:2606.14386)
    as JumpWorthAssessment: spectral compression × orthogonal escape
    (× residual alignment when a target is available). Advisory only.
    """
    from factorminer.architecture.geometry import (
        assess_llm_jump_worth,
        collect_library_span_matrix,
    )

    library = _load_library_from_path(library_path)
    span = collect_library_span_matrix(library)

    click.echo("=" * 60)
    click.echo("FactorMiner -- LLM Jump-Worth Gate")
    click.echo("=" * 60)
    click.echo(f"  Library size: {library.size}  span shape: {span.shape}")

    if span.size == 0:
        # Probe against empty span.
        probe = np.ones(16, dtype=np.float64)
    else:
        # Default probe: a direction partially outside the span (linear trend).
        probe = np.linspace(-1.0, 1.0, span.shape[0], dtype=np.float64)

    assessment = assess_llm_jump_worth(span, probe, threshold=threshold)
    payload = assessment.to_dict()

    if json_output:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"  jump_worth:           {assessment.jump_worth:.4f}")
    click.echo(f"  spectral_compression: {assessment.spectral_compression:.4f}")
    click.echo(f"  orthogonal_escape:    {assessment.orthogonal_escape:.4f}")
    click.echo(f"  residual_alignment:   {assessment.residual_alignment:.4f}")
    click.echo(f"  library_rank:         {assessment.library_rank}/{assessment.library_size}")
    click.echo(f"  recommend_llm_jump:   {assessment.recommend_llm_jump}")
    click.echo(f"  rationale:            {assessment.rationale}")
    click.echo("=" * 60)



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
    type=click.Choice(["json", "csv", "formulas", "qlib"]),
    default="json",
    help="Export format.",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option(
    "--anonymize",
    is_flag=True,
    help="Emit a redacted factor table (formula replaced by a hash) instead of the raw --format export.",
)
@click.pass_context
def export_cmd(
    ctx: click.Context, library_path: str, fmt: str, output: str | None, anonymize: bool,
) -> None:
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
        if anonymize:
            anon_ext = "json" if fmt in ("json", "formulas", "qlib") else fmt
            output = str(output_dir / f"library_anonymized.{anon_ext}")
        elif fmt == "formulas":
            output = str(output_dir / "library_formulas.txt")
        elif fmt == "qlib":
            output = str(output_dir / "library_qlib.json")
        else:
            output = str(output_dir / f"library.{fmt}")

    click.echo(f"  Format:  {fmt}{' (anonymized)' if anonymize else ''}")
    click.echo(f"  Output:  {output}")
    click.echo("-" * 60)

    try:
        from factorminer.core.library_io import (
            export_anonymized,
            export_csv,
            export_formulas,
            export_formulas_qlib,
            save_library,
        )

        if anonymize:
            # The redacted export is a fixed row-table shape; --format only
            # picks its container (csv, or json for the non-tabular formats).
            anon_fmt = "json" if fmt in ("json", "formulas", "qlib") else fmt
            export_anonymized(library, output, fmt=anon_fmt)
            click.echo(f"  Exported {library.size} anonymized factors to {output}")

        elif fmt == "json":
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

        elif fmt == "qlib":
            export_formulas_qlib(library, output)
            click.echo(f"  Exported {library.size} Qlib-translated formulas to {output}")

    except Exception as e:
        click.echo(f"Export error: {e}")
        logger.exception("Export failed")
        raise click.Abort()

    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# export-rft-dataset
# ---------------------------------------------------------------------------

@main.command(name="export-rft-dataset")
@click.argument(
    "lifecycle_path",
    type=click.Path(exists=True),
    required=False,
    default=None,
)
@click.option(
    "--output", "-o", "output_path",
    type=click.Path(),
    default=None,
    help="Destination JSONL path (default: <output_dir>/rft_dataset.jsonl).",
)
@click.option(
    "--data", "data_path",
    type=click.Path(exists=True),
    default=None,
    help="Optional market data file used only for regime-aware task bucketing.",
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use mock returns for regime-aware task bucketing (no network).",
)
@click.option(
    "--include-failed-parses",
    is_flag=True,
    help="Keep parse-failed candidates in the exported trajectory.",
)
@click.pass_context
def export_rft_dataset_cmd(
    ctx: click.Context,
    lifecycle_path: str | None,
    output_path: str | None,
    data_path: str | None,
    mock: bool,
    include_failed_parses: bool,
) -> None:
    """Export a reward-annotated offline RFT trajectory dataset as JSONL.

    Exports a reward-annotated training dataset for external reinforcement
    fine-tuning (e.g. GRPO via Verl/vLLM on a GPU host). This command does
    NOT train a model -- policy-weight training requires external GPU
    infrastructure not available in this environment.

    Reads ``factor_lifecycle.jsonl`` from a mining session output directory
    (or a direct path to that file) and writes one JSON object per candidate
    with the documented ``rft_v1`` schema:
    ``(state, action/formula, reward, regime_context)``.
    """
    from factorminer.architecture.rft_export import (
        RFT_EXPORT_HONESTY,
        RFTExportConfig,
        export_rft_dataset,
    )

    session_output = ctx.obj["output_dir"]
    cfg = ctx.obj["config"]

    # Resolve lifecycle source: explicit arg, else the session output dir.
    source = lifecycle_path or str(session_output)
    if output_path is None:
        session_output.mkdir(parents=True, exist_ok=True)
        output_path = str(Path(session_output) / "rft_dataset.jsonl")

    click.echo("=" * 60)
    click.echo("FactorMiner -- Export RFT Dataset (offline only)")
    click.echo("=" * 60)
    click.echo(RFT_EXPORT_HONESTY)
    click.echo("-" * 60)
    click.echo(f"  Lifecycle source: {source}")
    click.echo(f"  Output JSONL:     {output_path}")

    returns = None
    if mock or data_path is not None:
        try:
            dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
            returns = dataset.returns
            click.echo(
                f"  Regime bucketing: enabled "
                f"({len(dataset.asset_ids)} assets x {len(dataset.timestamps)} periods)"
            )
        except Exception as e:
            click.echo(f"  Regime bucketing skipped (data load failed: {e})")
            returns = None
    else:
        click.echo("  Regime bucketing: skipped (pass --mock or --data to enable)")

    export_cfg = RFTExportConfig(include_failed_parses=include_failed_parses)
    try:
        result = export_rft_dataset(
            source,
            output_path,
            returns=returns,
            config=export_cfg,
        )
    except Exception as e:
        click.echo(f"RFT export error: {e}")
        logger.exception("RFT dataset export failed")
        raise click.Abort()

    click.echo("-" * 60)
    click.echo(f"  Records:       {result.n_records}")
    click.echo(f"  Iterations:    {result.n_iterations}")
    click.echo(f"  Schema:        {result.schema_version}")
    click.echo(
        f"  Reward mean/std/min/max: "
        f"{result.reward_mean:.6f} / {result.reward_std:.6f} / "
        f"{result.reward_min:.6f} / {result.reward_max:.6f}"
    )
    if result.regime_task_counts:
        mix = ", ".join(f"{k}={v}" for k, v in sorted(result.regime_task_counts.items()))
        click.echo(f"  Regime tasks:  {mix}")
    if result.manifest_path:
        click.echo(f"  Manifest:      {result.manifest_path}")
    click.echo(f"  Wrote:         {result.path}")
    click.echo("-" * 60)
    click.echo("Reminder: this command does NOT train a model.")
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


@benchmark.command("cpcv")
@click.option("--baseline", default="factor_miner", help="Baseline id to evaluate.")
@_benchmark_common_options
def benchmark_cpcv(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
    baseline: str,
) -> None:
    """Run Combinatorial Purged CV + Probability of Backtest Overfitting diagnostics."""
    from factorminer.benchmark.runtime import run_cpcv_benchmark

    cfg = ctx.obj["config"]
    payload = run_cpcv_benchmark(
        cfg,
        data_path=data_path,
        mock=mock,
        baseline=baseline,
    )
    _print_benchmark_summary("FactorMiner -- CPCV / PBO", payload)


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


# ---------------------------------------------------------------------------
# retrieval-smoke
# ---------------------------------------------------------------------------

@main.command("retrieval-smoke")
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Include dense embedder ranks (hash/TF-IDF fallback; no forced download).",
)
@click.option(
    "--rerank/--no-rerank",
    default=False,
    help="Enable lightweight cross-encoder-style rerank over the fused top pool.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
def retrieval_smoke(embeddings: bool, rerank: bool, json_output: bool) -> None:
    """Run hybrid BM25+dense retrieval quality smoke on a synthetic labeled set.

    Verifies Reciprocal Rank Fusion prefers historically-successful patterns
    over weak/forbidden ones. No network I/O; safe under --mock/CI.
    """
    from factorminer.memory.retrieval import (
        HybridRetrievalConfig,
        retrieval_quality_smoke,
    )

    embedder = None
    if embeddings:
        from factorminer.memory.embeddings import FormulaEmbedder

        embedder = FormulaEmbedder(use_faiss=False)

    cfg = HybridRetrievalConfig(enabled=True, enable_rerank=rerank)
    result = retrieval_quality_smoke(embedder=embedder, hybrid_config=cfg)

    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
        if not result.get("passed"):
            raise click.Abort()
        return

    click.echo("FactorMiner -- Retrieval Quality Smoke")
    click.echo("=" * 60)
    click.echo(f"  Passed:            {result.get('passed')}")
    click.echo(f"  Hybrid ranking:    {result.get('hybrid_ranking')}")
    click.echo(f"  Heuristic ranking: {result.get('heuristic_ranking')}")
    click.echo(f"  Criterion:         {result.get('criterion')}")
    click.echo("=" * 60)
    if not result.get("passed"):
        raise click.ClickException("retrieval quality smoke failed")


# ---------------------------------------------------------------------------
# mcp-serve
# ---------------------------------------------------------------------------

@main.command("mcp-serve")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"], case_sensitive=False),
    default="stdio",
    show_default=True,
    help="MCP transport. 'stdio' is process-local (no auth). "
    "'http' enables streamable-HTTP on --host/--port and requires a bearer token.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host for --transport http. Default is loopback only (never 0.0.0.0).",
)
@click.option(
    "--port",
    type=int,
    default=8765,
    show_default=True,
    help="Bind port for --transport http.",
)
@click.option(
    "--auth-token-env",
    default="FACTORMINER_MCP_TOKEN",
    show_default=True,
    help="Env var holding the bearer token required when --transport http is selected. "
    "The server refuses to start if the variable is unset/empty under HTTP.",
)
def mcp_serve(transport: str, host: str, port: int, auth_token_env: str) -> None:
    """Run the FactorMiner MCP server (stdio by default, optional HTTP).

    Exposes FactorMiner's mining, evaluation, backtesting, benchmark, and
    reporting workflows as Model Context Protocol tools. Register it with a
    Claude client (Claude Code, Cowork, or a Managed Agent) through an
    .mcp.json entry. Nothing is written to stdout besides the MCP protocol
    stream when using stdio, so this command must not be combined with other
    output on that transport.

    HTTP mode binds to HOST:PORT (default 127.0.0.1:8765) and requires a
    non-empty bearer token in the environment variable named by
    --auth-token-env (default FACTORMINER_MCP_TOKEN).
    """
    try:
        from factorminer.mcp.server import run_server
    except ModuleNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        run_server(
            transport=transport.lower(),  # type: ignore[arg-type]
            host=host,
            port=port,
            auth_token_env=auth_token_env,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


# ---------------------------------------------------------------------------
# mcp-connectors
# ---------------------------------------------------------------------------

@main.command("mcp-connectors")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
def mcp_connectors(json_output: bool) -> None:
    """List financial-services MCP connector endpoints bundled with the plugin."""
    from factorminer.data.mcp_source import known_mcp_connectors

    connectors = known_mcp_connectors()
    if json_output:
        click.echo(json.dumps({"connectors": connectors}, indent=2, sort_keys=True))
        return

    click.echo("FactorMiner -- FSI MCP Connectors")
    click.echo("=" * 60)
    for connector in connectors:
        click.echo(f"  {connector['name']:<12s} {connector['url']}")
        click.echo(f"  {'':<12s} {connector['best_for']}")
    click.echo("=" * 60)
    click.echo("Use these endpoints in plugin .mcp.json or an MCP-source YAML config.")


# ---------------------------------------------------------------------------
# fetch-data
# ---------------------------------------------------------------------------

@main.command("fetch-data")
@click.option(
    "--mcp-config",
    "mcp_config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an MCP-source YAML config (connector URL, tool, field mapping).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Destination market-data file (.csv, .parquet, or .h5).",
)
def fetch_data_cmd(mcp_config: str, output_path: str) -> None:
    """Fetch market data from an external MCP connector (FactSet, Daloopa, ...)."""
    from factorminer.data.mcp_source import fetch_to_file, load_mcp_source_config

    click.echo("FactorMiner -- MCP Data Fetch")
    click.echo("=" * 60)
    click.echo(f"  Config: {mcp_config}")
    try:
        config = load_mcp_source_config(mcp_config)
        click.echo(f"  Connector: {config.transport} -> tool '{config.tool}'")
        written = fetch_to_file(config, output_path)
    except ModuleNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to CLI
        click.echo(f"Fetch error: {exc}")
        raise click.Abort() from exc

    click.echo(f"  Wrote: {written}")
    click.echo("=" * 60)
    click.echo(f"Next: uv run factorminer validate-data {written}")


# ---------------------------------------------------------------------------
# attach-edgar
# ---------------------------------------------------------------------------

@main.command("attach-edgar")
@click.option(
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="OHLCV market-data file to join fundamentals onto.",
)
@click.option(
    "--cik-map",
    "cik_map_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML/JSON mapping of asset_id -> CIK.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Destination panel with eps/revenue/book_equity/shares_out columns.",
)
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Local cache directory for companyfacts JSON (keyed by sanitized CIK).",
)
@click.option(
    "--user-agent",
    default=None,
    help="SEC-compliant User-Agent (must include contact email). "
    "Default: 'FactorMiner Research Bot 1.0 (contact@factorminer.local)'.",
)
@click.option(
    "--fixture",
    "fixture_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Offline companyfacts JSON fixture (skips network; for tests/CI).",
)
def attach_edgar_cmd(
    data_path: str,
    cik_map_path: str,
    output_path: str,
    cache_dir: str | None,
    user_agent: str | None,
    fixture_path: str | None,
) -> None:
    """Join point-in-time SEC EDGAR XBRL fundamentals onto an OHLCV panel."""
    import json
    from pathlib import Path as _Path

    import yaml

    from factorminer.data.edgar_source import (
        DEFAULT_USER_AGENT,
        EdgarConfig,
        attach_edgar_to_panel,
        register_edgar_features,
    )
    from factorminer.data.loader import load_market_data

    click.echo("FactorMiner -- Attach EDGAR Fundamentals")
    click.echo("=" * 60)

    panel = load_market_data(data_path)
    raw_map = _Path(cik_map_path).read_text(encoding="utf-8")
    if cik_map_path.endswith((".yaml", ".yml")):
        cik_map = yaml.safe_load(raw_map) or {}
    else:
        cik_map = json.loads(raw_map)
    if not isinstance(cik_map, dict) or not cik_map:
        raise click.ClickException("cik-map must be a non-empty asset_id -> CIK mapping")

    offline = None
    if fixture_path is not None:
        fixture = json.loads(_Path(fixture_path).read_text(encoding="utf-8"))
        # Accept either a single payload applied to all assets or asset_id-keyed dict.
        if isinstance(fixture, dict) and "facts" in fixture:
            offline = {str(a): fixture for a in cik_map}
        elif isinstance(fixture, dict):
            offline = fixture
        else:
            raise click.ClickException("fixture must be a companyfacts object or asset-keyed map")

    cfg = EdgarConfig(
        user_agent=user_agent or DEFAULT_USER_AGENT,
        cache_dir=cache_dir,
    )
    register_edgar_features()
    joined = attach_edgar_to_panel(
        panel,
        {str(k): v for k, v in cik_map.items()},
        config=cfg,
        offline_payloads=offline,
    )

    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".parquet" or suffix == ".pq":
        joined.to_parquet(out, index=False)
    elif suffix in {".h5", ".hdf5"}:
        joined.to_hdf(out, key="data", mode="w")
    else:
        joined.to_csv(out, index=False)

    click.echo(f"  Assets: {joined['asset_id'].nunique()}")
    click.echo(f"  Rows:   {len(joined)}")
    for col in ("eps", "revenue", "book_equity", "shares_out"):
        if col in joined.columns:
            n = int(joined[col].notna().sum())
            click.echo(f"  {col}: {n} non-null values (point-in-time as-filed)")
    click.echo(f"  Wrote:  {out}")
    click.echo("=" * 60)
    click.echo(
        "Security: HTTPS data.sec.gov, descriptive User-Agent, "
        "<=10 req/s rate limit, local CIK cache, fail-closed JSON parse."
    )


# ---------------------------------------------------------------------------
# build-futures
# ---------------------------------------------------------------------------

@main.command("build-futures")
@click.option(
    "--data",
    "data_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Raw futures panel CSV/Parquet. Omit with --mock.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Destination continuous futures panel with basis/spot/premium/oi.",
)
@click.option("--mock", is_flag=True, help="Generate a deterministic synthetic futures panel.")
@click.option(
    "--multiplier",
    default=1.0,
    show_default=True,
    type=float,
    help="Contract multiplier used for notional amount.",
)
def build_futures_cmd(
    data_path: str | None,
    output_path: str,
    mock: bool,
    multiplier: float,
) -> None:
    """Build a roll-adjusted continuous futures panel with basis leaves."""
    from pathlib import Path as _Path

    from factorminer.data.futures_source import (
        FuturesConfig,
        build_continuous_futures_panel,
        generate_mock_futures_panel,
        register_futures_features,
    )
    from factorminer.data.loader import load_market_data

    click.echo("FactorMiner -- Build Continuous Futures Panel")
    click.echo("=" * 60)

    cfg = FuturesConfig(contract_multiplier=float(multiplier))
    register_futures_features()

    if mock or data_path is None:
        if data_path is not None and not mock:
            raise click.ClickException("Pass --mock or provide --data")
        click.echo("  Generating mock continuous futures panel...")
        panel = generate_mock_futures_panel(config=cfg)
    else:
        raw = load_market_data(data_path)
        panel = build_continuous_futures_panel(raw, config=cfg)

    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        panel.to_parquet(out, index=False)
    elif suffix in {".h5", ".hdf5"}:
        panel.to_hdf(out, key="data", mode="w")
    else:
        panel.to_csv(out, index=False)

    click.echo(f"  Assets: {panel['asset_id'].nunique()}")
    click.echo(f"  Rows:   {len(panel)}")
    for col in ("basis", "spot", "premium", "roll_yield", "oi"):
        if col in panel.columns:
            click.echo(f"  leaf ${col}: present")
    click.echo(f"  Wrote:  {out}")
    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# ingest-research
# ---------------------------------------------------------------------------

@main.command("ingest-research")
@click.argument("note_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--mock", is_flag=True, help="Use mock LLM provider (no API calls).")
@click.option(
    "--eligibility-mode",
    type=click.Choice(["ohlcv_only", "alt_enabled"]),
    default="ohlcv_only",
    show_default=True,
    help="A-layer gate: ohlcv_only (default) or alt_enabled (keep fragments "
    "that map onto registered non-OHLCV leaves such as $eps).",
)
@click.pass_context
def ingest_research(
    ctx: click.Context,
    note_path: str,
    mock: bool,
    eligibility_mode: str,
) -> None:
    """Absorb a research report fragment via Report-to-Memory Absorption (RMA).

    Screens the fragment for OHLCV-representability (A-layer); if KEPT,
    classifies it into a mechanism family and prints reusable research-path
    hypothesis cues (B/C-layer). See the `research-ingestion` skill.
    """
    from factorminer.architecture.research_absorption import (
        ResearchAbsorptionService,
        read_research_note,
    )

    cfg = ctx.obj["config"]
    provider = _create_llm_provider(cfg, mock)
    service = ResearchAbsorptionService(
        llm_provider=provider,
        eligibility_mode=eligibility_mode,
    )
    note = read_research_note(note_path)

    click.echo("FactorMiner -- Research Ingestion (RMA)")
    click.echo("=" * 60)
    click.echo(f"Source: {note.source}")

    keep, reason = service.screen_eligibility(note.text)
    click.echo(f"A-layer verdict: {'KEEP' if keep else 'DROP'}")
    click.echo(f"Reason:          {reason}")

    if not keep:
        click.echo("=" * 60)
        click.echo("Fragment dropped -- not OHLCV-representable.")
        return

    archetype = service.classify_mechanism(note.text)
    click.echo(f"Mechanism family: {archetype.mechanism_family}")
    click.echo(f"Fine family:      {archetype.fine_family}")
    click.echo(f"Archetype name:   {archetype.name}")
    click.echo(f"Mechanism role:   {archetype.mechanism_role}")
    click.echo("Research paths:")
    for path in archetype.research_paths:
        click.echo(f"  - {path}")
    click.echo("=" * 60)



# ---------------------------------------------------------------------------
# sealed-search (Agora multi-evaluator research mode)
# ---------------------------------------------------------------------------

@main.command("sealed-search")
@click.option(
    "--agreement-rule",
    type=click.Choice(["majority", "unanimous", "all_but_one", "threshold"]),
    default="majority",
    show_default=True,
    help="Multi-evaluator agreement rule for promotion eligibility.",
)
@click.option(
    "--min-agree",
    type=int,
    default=2,
    show_default=True,
    help="Minimum evaluator passes when --agreement-rule=threshold.",
)
@click.option(
    "--no-llm-judge",
    is_flag=True,
    help="Disable the optional LLM-as-judge persona (numeric panel only).",
)
@click.option(
    "--demo",
    is_flag=True,
    help="Run the built-in synthetic disagreement demo (no library/data needed).",
)
@click.option("--mock", is_flag=True, help="Use mock LLM provider if LLM judge is enabled.")
@click.pass_context
def sealed_search(
    ctx: click.Context,
    agreement_rule: str,
    min_agree: int,
    no_llm_judge: bool,
    demo: bool,
    mock: bool,
) -> None:
    """Opt-in Agora sealed multi-evaluator promotion (research mode).

    Runs differently-biased evaluators with sealed internals, promotes only
    under multi-evaluator agreement, and reports disagreement diagnostics.
    Does NOT replace EvaluationKernel default admission. Paper caveat
    (arXiv:2606.29194): single-seed variance is real — not a proven default.
    """
    from factorminer.architecture.sealed_joint_search import (
        RESEARCH_MODE_CAVEAT,
        AgreementRule,
        CandidateObservation,
        SealedJointSearchConfig,
        SealedJointSearchEngine,
    )

    cfg = ctx.obj["config"]
    provider = None
    if not no_llm_judge:
        # Always mock for the research demo unless a future library path needs a real judge.
        provider = _create_llm_provider(cfg, True)

    engine = SealedJointSearchEngine(
        SealedJointSearchConfig(
            enabled=True,
            agreement_rule=AgreementRule(agreement_rule),
            min_agree=min_agree,
            include_llm_judge=not no_llm_judge,
            retain_internal_scores=True,
        ),
        llm_provider=provider,
    )

    click.echo("=" * 60)
    click.echo("FactorMiner -- Sealed Joint Search (research mode)")
    click.echo("=" * 60)
    click.echo(f"Agreement rule: {agreement_rule}")
    click.echo(f"Evaluators:     {', '.join(engine.evaluator_ids)}")
    click.echo(RESEARCH_MODE_CAVEAT)
    click.echo("-" * 60)

    if not demo:
        # Default path is the synthetic demo so the command is always runnable
        # without a library; future library-path wiring stays opt-in.
        demo = True
        click.echo("No library input supplied — running built-in synthetic demo.")

    observations = [
        CandidateObservation(
            name="high_ic_brittle",
            formula="CsRank(Delta($close, 1))",
            ic_paper_mean=0.08,
            ic_mean=0.08,
            ic_std=0.12,
            icir=0.67,
            ic_win_rate=0.62,
            intervention_robustness=0.15,
            cpcv_ic_std=0.10,
            cpcv_ic_mean=0.08,
            max_library_dependence=0.25,
            novelty_score=0.75,
        ),
        CandidateObservation(
            name="high_ic_crowded",
            formula="CsRank(Delta($close, 5))",
            ic_paper_mean=0.07,
            ic_mean=0.07,
            ic_std=0.03,
            icir=2.3,
            ic_win_rate=0.70,
            intervention_robustness=0.80,
            cpcv_ic_std=0.015,
            cpcv_ic_mean=0.07,
            max_library_dependence=0.92,
            novelty_score=0.08,
        ),
        CandidateObservation(
            name="balanced_solid",
            formula="Neg(CsZScore(Div(Sub($close, SMA($close, 20)), SMA($close, 20))))",
            ic_paper_mean=0.045,
            ic_mean=0.045,
            ic_std=0.02,
            icir=2.25,
            ic_win_rate=0.60,
            intervention_robustness=0.75,
            cpcv_ic_std=0.018,
            cpcv_ic_mean=0.045,
            max_library_dependence=0.20,
            novelty_score=0.80,
        ),
        CandidateObservation(
            name="weak_noise",
            formula="CsRank($volume)",
            ic_paper_mean=0.005,
            ic_mean=0.005,
            ic_std=0.08,
            icir=0.06,
            ic_win_rate=0.48,
            intervention_robustness=0.20,
            cpcv_ic_std=0.09,
            cpcv_ic_mean=0.005,
            max_library_dependence=0.85,
            novelty_score=0.15,
        ),
    ]

    report = engine.evaluate_batch(observations)
    click.echo(f"Candidates:     {report.n_candidates}")
    click.echo(f"Promoted:       {report.promoted_names()}")
    click.echo(f"Rejected:       {report.rejected_names()}")
    click.echo(f"Disagreement:   {report.disagreement_rate:.2%}")
    click.echo(f"Mean agreement: {report.mean_agreement_fraction:.2%}")
    click.echo("-" * 60)
    for decision in report.decisions:
        fb = decision.feedback
        status = "PROMOTED" if decision.promoted else "held"
        click.echo(
            f"  [{status:8s}] {decision.observation.name:18s} "
            f"passed {decision.n_passed}/{decision.n_evaluators} "
            f"rank={decision.batch_rank} "
            f"disagree={decision.disagreement}"
        )
        if fb is not None:
            click.echo(f"             personas+={list(fb.passed_personas)} "
                       f"personas-={list(fb.failed_personas)}")
    click.echo("=" * 60)
    click.echo("Prompt-safe sealed feedback (no raw evaluator scores):")
    for payload in report.sealed_feedback_batch():
        click.echo(
            f"  {payload['candidate_name']}: "
            f"{payload['n_passed']}/{payload['n_evaluators']} "
            f"promoted={payload['promoted']}"
        )
    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# model-co-optimize (RD-Agent(Q)-style downstream model zoo diagnostic)
# ---------------------------------------------------------------------------


@main.command("model-co-optimize")
@click.option(
    "--model-kind",
    type=click.Choice(["ridge", "lasso", "xgboost", "corr_graphsage"]),
    default="ridge",
    show_default=True,
    help="Downstream model family to fit on the factor library signals.",
)
@click.option(
    "--train-objective",
    type=click.Choice(["mse", "margin_pairwise", "listnet", "bpr"]),
    default="mse",
    show_default=True,
    help="Training objective. Ranking losses re-fit linear models; xgboost uses rank:pairwise.",
)
@click.option("--alpha", type=float, default=1.0, show_default=True, help="L2/L1 strength (linear models).")
@click.option(
    "--train-fraction",
    type=float,
    default=0.7,
    show_default=True,
    help="Fraction of periods used for training (rest held out).",
)
@click.option(
    "--graph-corr-threshold",
    type=float,
    default=0.3,
    show_default=True,
    help="Absolute return-correlation threshold for corr_graphsage edges.",
)
@click.option(
    "--graph-hidden-dim",
    type=int,
    default=8,
    show_default=True,
    help="Hidden width of the corr_graphsage encoder.",
)
@click.option(
    "--permutation-repeats",
    type=int,
    default=10,
    show_default=True,
    help="Repeats for held-out permutation importance.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="RNG seed.")
@click.option("--mock", is_flag=True, help="Run against a built-in synthetic factor panel (no library needed).")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit the co-optimization report as JSON.",
)
def model_co_optimize_cmd(
    model_kind: str,
    train_objective: str,
    alpha: float,
    train_fraction: float,
    graph_corr_threshold: float,
    graph_hidden_dim: int,
    permutation_repeats: int,
    seed: int,
    mock: bool,
    json_output: bool,
) -> None:
    """Fit a downstream model on factor signals and rank factor contributions.

    RD-Agent(Q)-style diagnostic: ridge/lasso/xgboost on flattened samples, or
    corr_graphsage (optional torch) on a per-date asset correlation graph.
    Defaults to a synthetic mock panel so the command is always runnable.
    """
    import json as json_lib

    from factorminer.evaluation.model_zoo import ModelZooConfig, ModelZooEvaluator

    if not mock:
        mock = True
        click.echo("No library input supplied — running built-in synthetic mock panel.")

    rng = __import__("numpy").random.default_rng(seed)
    assets, periods = 20, 80
    strong = rng.standard_normal((assets, periods))
    weak = rng.standard_normal((assets, periods))
    noise_a = rng.standard_normal((assets, periods))
    noise_b = rng.standard_normal((assets, periods))
    returns = 0.75 * strong + 0.15 * weak + 0.4 * rng.standard_normal((assets, periods))
    factor_signals = {1: strong, 2: weak, 3: noise_a, 4: noise_b}
    factor_names = {1: "alpha_strong", 2: "alpha_weak", 3: "noise_a", 4: "noise_b"}

    config = ModelZooConfig(
        model_kind=model_kind,
        train_objective=train_objective,
        alpha=alpha,
        train_fraction=train_fraction,
        graph_corr_threshold=graph_corr_threshold,
        graph_hidden_dim=graph_hidden_dim,
        permutation_repeats=permutation_repeats,
        seed=seed,
        xgb_n_estimators=40,
        xgb_max_depth=3,
    )

    click.echo("=" * 60)
    click.echo("FactorMiner -- Model Co-Optimize (downstream zoo)")
    click.echo("=" * 60)
    click.echo(f"Model kind:       {model_kind}")
    click.echo(f"Train objective:  {train_objective}")
    click.echo(f"Train fraction:   {train_fraction}")
    click.echo(f"Panel:            {assets} assets x {periods} periods (mock)")
    click.echo("-" * 60)

    try:
        report = ModelZooEvaluator().evaluate(
            factor_signals, factor_names, returns, config=config, iteration=0
        )
    except RuntimeError as exc:
        click.echo(f"Error: {exc}")
        raise click.Abort() from exc

    if json_output:
        click.echo(json_lib.dumps(report.to_dict(), indent=2, default=str))
        return

    click.echo(f"Held-out IC:      {report.held_out_ic:.4f}")
    click.echo(f"Held-out R^2:     {report.held_out_r2:.4f}")
    click.echo(f"Held-out Sharpe:  {report.held_out_sharpe:.4f}")
    click.echo(f"Baseline EQ IC:   {report.baseline_equal_weight_ic:.4f}")
    click.echo(f"Train/test n:     {report.n_train_samples}/{report.n_test_samples}")
    if report.neighbor_influence_summary:
        click.echo(f"Neighbors:        {report.neighbor_influence_summary}")
    click.echo("-" * 60)
    click.echo(f"{'Rank':>4s}  {'Factor':<20s}  {'PermImp':>10s}  {'Coef':>10s}  {'dIC':>8s}")
    for c in report.contributions:
        coef = f"{c.coefficient:.4f}" if c.coefficient is not None else "n/a"
        click.echo(
            f"{c.rank:4d}  {c.factor_name:<20s}  "
            f"{c.permutation_importance_mean:10.4f}  {coef:>10s}  "
            f"{c.ensemble_marginal_delta_ic:8.4f}"
        )
    click.echo("=" * 60)

if __name__ == "__main__":
    main()
