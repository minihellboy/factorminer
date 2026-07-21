"""FactorMiner CLI root and command registration."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import fields
from importlib.util import find_spec
from pathlib import Path

import click
import numpy as np

from factorminer.cli.context import (
    deep_merge_dict as _deep_merge_dict,
)
from factorminer.cli.context import (
    load_market_frame as _load_data,
)
from factorminer.cli.context import main as main
from factorminer.configs import DEFAULT_CONFIG_PATH, load_default_yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
