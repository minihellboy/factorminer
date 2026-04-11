"""Static report generation for FactorMiner artifacts.

This module turns persisted mining artifacts into a local markdown or HTML
report without introducing a web dashboard. It understands the canonical
factor library JSON, optional session logs, and optional benchmark payloads.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

JSONSource = str | Path | Mapping[str, Any]


@dataclass(frozen=True)
class ReportSection:
    """One benchmark payload section in the final report."""

    label: str
    payload: dict[str, Any]
    source: str


def _resolve_json_path(source: JSONSource, *, default_filename: str | None = None) -> Path:
    if isinstance(source, Mapping):
        raise TypeError("mapping sources do not resolve to a filesystem path")

    path = Path(source)
    if path.is_dir():
        if default_filename is None:
            raise FileNotFoundError(f"{path} is a directory; expected a JSON file")
        path = path / default_filename
    elif not path.exists() and not path.suffix:
        candidate = path.with_suffix(".json")
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Could not find JSON artifact at {path}")
    return path


def _load_json_source(
    source: JSONSource | None, *, default_filename: str | None = None
) -> dict[str, Any] | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return dict(source)

    path = _resolve_json_path(source, default_filename=default_filename)
    return json.loads(path.read_text(encoding="utf-8"))


def _json_label(source: JSONSource, fallback: str) -> str:
    if isinstance(source, Mapping):
        return fallback
    path = Path(source)
    if path.is_dir():
        return path.name
    return path.stem


def _first_non_empty(*values: Any, default: str = "unknown") -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and value is not False:
            return str(value)
    return default


def _fmt_num(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{precision}f}"
    return str(value)


def _table_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _factor_rows(library_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in library_payload.get("factors", []):
        if not isinstance(factor, Mapping):
            continue
        provenance = factor.get("provenance") or {}
        research_metrics = factor.get("research_metrics") or {}
        family_hint = _first_non_empty(
            provenance.get("family_hint"),
            provenance.get("family"),
            provenance.get("group"),
            research_metrics.get("family_hint"),
            research_metrics.get("family"),
            research_metrics.get("group"),
            factor.get("category"),
        )
        rows.append(
            {
                "id": factor.get("id", "-"),
                "name": factor.get("name", "-"),
                "category": factor.get("category", "-"),
                "ic_mean": factor.get("ic_mean"),
                "icir": factor.get("icir"),
                "ic_win_rate": factor.get("ic_win_rate"),
                "max_correlation": factor.get("max_correlation"),
                "batch_number": factor.get("batch_number", "-"),
                "family_hint": family_hint,
                "provenance_loop": provenance.get("loop_type", "-"),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["ic_mean"] or 0.0),
            int(row["batch_number"]) if str(row["batch_number"]).isdigit() else 0,
            int(row["id"]) if str(row["id"]).isdigit() else 0,
        )
    )
    return rows


def _session_counts(session_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not session_payload:
        return None

    iterations = session_payload.get("iterations", [])
    factors = session_payload.get("factors", [])
    summary = session_payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}

    total_candidates = summary.get("total_candidates")
    if total_candidates is None:
        total_candidates = sum(
            int(it.get("candidates_generated", it.get("candidates", 0)) or 0)
            for it in iterations
            if isinstance(it, Mapping)
        )

    total_admitted = summary.get("total_admitted")
    if total_admitted is None:
        total_admitted = sum(
            int(it.get("admitted", 0) or 0) for it in iterations if isinstance(it, Mapping)
        )

    total_rejected = sum(
        int(it.get("rejected", 0) or 0) for it in iterations if isinstance(it, Mapping)
    )
    total_replaced = sum(
        int(it.get("replaced", 0) or 0) for it in iterations if isinstance(it, Mapping)
    )

    rejection_reasons = Counter(
        str(factor.get("rejection_reason", "unknown")).strip() or "unknown"
        for factor in factors
        if isinstance(factor, Mapping) and not factor.get("admitted", False)
    )

    return {
        "iterations": summary.get("total_iterations", len(iterations)),
        "candidates": total_candidates,
        "admitted": total_admitted,
        "rejected": total_rejected,
        "replaced": total_replaced,
        "yield_rate": summary.get(
            "overall_yield_rate",
            (float(total_admitted) / float(total_candidates)) if total_candidates else 0.0,
        ),
        "final_library_size": summary.get(
            "final_library_size",
            iterations[-1].get("library_size", 0) if iterations else 0,
        ),
        "factor_records": len(factors),
        "admitted_factors": sum(
            1 for factor in factors if isinstance(factor, Mapping) and factor.get("admitted", False)
        ),
        "rejection_reasons": rejection_reasons.most_common(5),
        "iteration_rows": [
            {
                "iteration": it.get("iteration", it.get("batch_num", "-")),
                "generated": it.get("candidates_generated", it.get("candidates", 0)),
                "admitted": it.get("admitted", 0),
                "rejected": it.get("rejected", 0),
                "replaced": it.get("replaced", 0),
                "library_size": it.get("library_size", 0),
                "best_ic": it.get("best_ic", 0.0),
                "mean_ic": it.get("mean_ic", 0.0),
                "yield_rate": it.get("yield_rate", 0.0),
                "elapsed_seconds": it.get("elapsed_seconds", 0.0),
            }
            for it in iterations
            if isinstance(it, Mapping)
        ],
    }


def _benchmark_sections(benchmark_source: JSONSource) -> list[ReportSection]:
    raw = _load_json_source(benchmark_source)
    if raw is None:
        return []

    source_label = _json_label(benchmark_source, "benchmark")
    if _looks_like_suite_payload(raw):
        sections: list[ReportSection] = []
        for label, payload in raw.items():
            if isinstance(payload, Mapping):
                sections.append(
                    ReportSection(
                        label=str(label),
                        payload=dict(payload),
                        source=source_label,
                    )
                )
        return sections

    return [
        ReportSection(
            label=str(raw.get("baseline") or raw.get("benchmark_name") or source_label),
            payload=dict(raw),
            source=source_label,
        )
    ]


def _looks_like_suite_payload(raw: Mapping[str, Any]) -> bool:
    single_keys = {
        "baseline",
        "benchmark_name",
        "freeze_universe",
        "freeze_library_size",
        "frozen_top_k",
        "universes",
    }
    if any(key in raw for key in single_keys):
        return False
    if not raw:
        return False
    return all(isinstance(value, Mapping) for value in raw.values())


def _benchmark_universe_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    universes = section.get("universes", {})
    if not isinstance(universes, Mapping):
        return rows

    for universe_name, universe_payload in universes.items():
        if not isinstance(universe_payload, Mapping):
            continue
        library_metrics = universe_payload.get("library", {})
        if not isinstance(library_metrics, Mapping):
            library_metrics = {}
        selection_metrics = universe_payload.get("selection", {})
        if not isinstance(selection_metrics, Mapping):
            selection_metrics = {}
        combination_metrics = universe_payload.get("combinations", {})
        if not isinstance(combination_metrics, Mapping):
            combination_metrics = {}
        factor_metrics = universe_payload.get("factors", {})
        if not isinstance(factor_metrics, Mapping):
            factor_metrics = {}

        rows.append(
            {
                "universe": universe_name,
                "library_ic": _first_numeric(
                    library_metrics, "ic", "ic_pct", "library_ic", "mean_ic"
                ),
                "library_icir": _first_numeric(library_metrics, "icir", "library_icir"),
                "avg_abs_rho": _first_numeric(library_metrics, "avg_abs_rho", "avg_rho", "rho"),
                "selected_ic": _first_numeric(
                    selection_metrics, "ic", "ic_pct", "best_ic", "mean_ic"
                ),
                "selected_icir": _first_numeric(selection_metrics, "icir", "best_icir"),
                "avg_turnover": _first_numeric(selection_metrics, "avg_turnover", "turnover"),
                "combo_ic": _first_numeric(combination_metrics, "ew_ic", "icw_ic", "ic", "mean_ic"),
                "combo_icir": _first_numeric(combination_metrics, "ew_icir", "icw_icir", "icir"),
                "factor_count": _first_numeric(
                    factor_metrics, "factor_count", "n_factors", "count"
                ),
            }
        )

    return rows


def _first_numeric(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def build_report_payload(
    library_source: JSONSource,
    *,
    session_log_source: JSONSource | None = None,
    benchmark_sources: Sequence[JSONSource] | None = None,
) -> dict[str, Any]:
    """Load and normalize report inputs into a structured payload."""

    library = _load_json_source(library_source, default_filename="factor_library.json")
    if library is None:
        raise ValueError("library_source is required")

    session_log = _load_json_source(session_log_source, default_filename="session_log.json")
    benchmark_sources = list(benchmark_sources or [])
    benchmark_sections: list[ReportSection] = []
    for benchmark_source in benchmark_sources:
        benchmark_sections.extend(_benchmark_sections(benchmark_source))

    factor_rows = _factor_rows(library)
    family_counts = Counter(row["family_hint"] for row in factor_rows)
    category_counts = Counter(row["category"] for row in factor_rows)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "library": {
            "source": _json_label(library_source, "factor_library"),
            "count": len(factor_rows),
            "correlation_threshold": library.get("correlation_threshold"),
            "ic_threshold": library.get("ic_threshold"),
            "dependence_metric": library.get("dependence_metric", "unknown"),
            "factors": factor_rows,
            "family_counts": family_counts.most_common(10),
            "category_counts": category_counts.most_common(10),
        },
        "session": {
            "source": _json_label(session_log_source, "session_log")
            if session_log_source
            else None,
            "counts": _session_counts(session_log),
        }
        if session_log is not None
        else None,
        "benchmarks": [
            {
                "label": section.label,
                "source": section.source,
                "payload": section.payload,
                "universe_rows": _benchmark_universe_rows(section.payload),
                "freeze_top_k": [
                    {
                        "name": item.get("name", "-"),
                        "formula": item.get("formula", "-"),
                        "category": item.get("category", "-"),
                        "train_ic": item.get("train_ic"),
                        "train_icir": item.get("train_icir"),
                    }
                    for item in section.payload.get("frozen_top_k", [])
                    if isinstance(item, Mapping)
                ],
            }
            for section in benchmark_sections
        ],
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_No data available._"
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(_table_cell(cell) for cell in row) + " |" for row in rows)
    return "\n".join([header_line, separator, body])


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "<p>No data available.</p>"
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_html_cell_value(cell))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _html_cell_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# FactorMiner Static Report")
    lines.append(f"Generated at: {payload.get('generated_at', '-')}")
    lines.append("")

    library = payload.get("library", {})
    lines.append("## Library Summary")
    lines.append(
        f"- Source: `{library.get('source', '-')}`  \n"
        f"- Factors: `{library.get('count', 0)}`  \n"
        f"- IC threshold: `{_fmt_num(library.get('ic_threshold'))}`  \n"
        f"- Correlation threshold: `{_fmt_num(library.get('correlation_threshold'))}`  \n"
        f"- Dependence metric: `{library.get('dependence_metric', 'unknown')}`"
    )
    lines.append("")
    lines.append(
        _markdown_table(
            [
                "ID",
                "Name",
                "Category",
                "IC",
                "ICIR",
                "Win Rate",
                "Max Corr",
                "Batch",
                "Family Hint",
            ],
            [
                [
                    row["id"],
                    row["name"],
                    row["category"],
                    row["ic_mean"],
                    row["icir"],
                    row["ic_win_rate"],
                    row["max_correlation"],
                    row["batch_number"],
                    row["family_hint"],
                ]
                for row in library.get("factors", [])
            ],
        )
    )

    lines.append("")
    lines.append("## Family and Category Hints")
    hint_rows = library.get("family_counts", [])
    if hint_rows:
        lines.append(_markdown_table(["Hint", "Count"], hint_rows))
    else:
        lines.append("_No family hints available._")
    lines.append("")
    category_rows = library.get("category_counts", [])
    if category_rows:
        lines.append(_markdown_table(["Category", "Count"], category_rows))
    else:
        lines.append("_No category hints available._")

    session = payload.get("session")
    if session and session.get("counts"):
        counts = session["counts"]
        lines.append("")
        lines.append("## Lifecycle and Admission")
        lines.append(
            f"- Iterations: `{counts.get('iterations', 0)}`  \n"
            f"- Candidates: `{counts.get('candidates', 0)}`  \n"
            f"- Admitted: `{counts.get('admitted', 0)}`  \n"
            f"- Rejected: `{counts.get('rejected', 0)}`  \n"
            f"- Replaced: `{counts.get('replaced', 0)}`  \n"
            f"- Yield rate: `{_fmt_num(counts.get('yield_rate'), 3)}`  \n"
            f"- Final library size: `{counts.get('final_library_size', 0)}`"
        )
        if counts.get("rejection_reasons"):
            lines.append("")
            lines.append(
                _markdown_table(["Rejection reason", "Count"], counts["rejection_reasons"])
            )
        if counts.get("iteration_rows"):
            lines.append("")
            lines.append(
                _markdown_table(
                    [
                        "Iteration",
                        "Generated",
                        "Admitted",
                        "Rejected",
                        "Replaced",
                        "Library Size",
                        "Yield",
                        "Best IC",
                        "Mean IC",
                    ],
                    [
                        [
                            row["iteration"],
                            row["generated"],
                            row["admitted"],
                            row["rejected"],
                            row["replaced"],
                            row["library_size"],
                            row["yield_rate"],
                            row["best_ic"],
                            row["mean_ic"],
                        ]
                        for row in counts["iteration_rows"]
                    ],
                )
            )

    benchmarks = payload.get("benchmarks", [])
    lines.append("")
    lines.append("## Benchmarks")
    if not benchmarks:
        lines.append("_No benchmark JSON provided._")
    else:
        for section in benchmarks:
            lines.append("")
            lines.append(f"### {section['label']}")
            lines.append(f"- Source: `{section['source']}`")
            section_payload = section.get("payload", {})
            if section_payload.get("baseline") is not None:
                lines.append(f"- Baseline: `{section_payload.get('baseline')}`")
            if section_payload.get("freeze_universe") is not None:
                lines.append(f"- Freeze universe: `{section_payload.get('freeze_universe')}`")
            if section_payload.get("freeze_library_size") is not None:
                lines.append(
                    f"- Freeze library size: `{section_payload.get('freeze_library_size')}`"
                )
            if section.get("freeze_top_k"):
                lines.append("")
                lines.append(
                    _markdown_table(
                        ["Name", "Category", "Train IC", "Train ICIR"],
                        [
                            [item["name"], item["category"], item["train_ic"], item["train_icir"]]
                            for item in section["freeze_top_k"]
                        ],
                    )
                )
            if section.get("universe_rows"):
                lines.append("")
                lines.append(
                    _markdown_table(
                        [
                            "Universe",
                            "Library IC",
                            "Library ICIR",
                            "Avg |rho|",
                            "Selected IC",
                            "Selected ICIR",
                            "Avg Turnover",
                            "Combo IC",
                            "Combo ICIR",
                            "Factor Count",
                        ],
                        [
                            [
                                row["universe"],
                                row["library_ic"],
                                row["library_icir"],
                                row["avg_abs_rho"],
                                row["selected_ic"],
                                row["selected_icir"],
                                row["avg_turnover"],
                                row["combo_ic"],
                                row["combo_icir"],
                                row["factor_count"],
                            ]
                            for row in section["universe_rows"]
                        ],
                    )
                )

    return "\n".join(lines).rstrip() + "\n"


def render_html_report(payload: Mapping[str, Any]) -> str:
    library = payload.get("library", {})
    session = payload.get("session")
    benchmarks = payload.get("benchmarks", [])

    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>FactorMiner Static Report</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;line-height:1.5;margin:32px;max-width:1400px;color:#111;background:#fafafa}",
        "h1,h2,h3{line-height:1.2}",
        (
            "section{margin:28px 0;padding:20px;background:#fff;"
            "border:1px solid #e5e7eb;border-radius:14px}"
        ),
        "table{border-collapse:collapse;width:100%;font-size:14px;margin-top:12px}",
        "th,td{border:1px solid #d1d5db;padding:8px 10px;text-align:left;vertical-align:top}",
        "th{background:#f3f4f6}",
        ".meta{color:#4b5563;font-size:14px}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}",
        ".card{border:1px solid #e5e7eb;border-radius:12px;padding:12px;background:#fcfcfc}",
        ".muted{color:#6b7280}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>FactorMiner Static Report</h1>",
        f'<p class="meta">Generated at {html.escape(str(payload.get("generated_at", "-")))}</p>',
        "<section>",
        "<h2>Library Summary</h2>",
        '<div class="cards">',
        _card("Source", library.get("source", "-")),
        _card("Factors", library.get("count", 0)),
        _card("IC threshold", _fmt_num(library.get("ic_threshold"))),
        _card("Correlation threshold", _fmt_num(library.get("correlation_threshold"))),
        _card("Dependence metric", library.get("dependence_metric", "unknown")),
        "</div>",
        _html_table(
            [
                "ID",
                "Name",
                "Category",
                "IC",
                "ICIR",
                "Win Rate",
                "Max Corr",
                "Batch",
                "Family Hint",
            ],
            [
                [
                    row["id"],
                    row["name"],
                    row["category"],
                    row["ic_mean"],
                    row["icir"],
                    row["ic_win_rate"],
                    row["max_correlation"],
                    row["batch_number"],
                    row["family_hint"],
                ]
                for row in library.get("factors", [])
            ],
        ),
        "</section>",
        "<section>",
        "<h2>Family and Category Hints</h2>",
        _html_table(["Hint", "Count"], library.get("family_counts", [])),
        _html_table(["Category", "Count"], library.get("category_counts", [])),
        "</section>",
    ]

    if session and session.get("counts"):
        counts = session["counts"]
        parts.extend(
            [
                "<section>",
                "<h2>Lifecycle and Admission</h2>",
                '<div class="cards">',
                _card("Iterations", counts.get("iterations", 0)),
                _card("Candidates", counts.get("candidates", 0)),
                _card("Admitted", counts.get("admitted", 0)),
                _card("Rejected", counts.get("rejected", 0)),
                _card("Replaced", counts.get("replaced", 0)),
                _card("Yield rate", _fmt_num(counts.get("yield_rate"), 3)),
                _card("Final library size", counts.get("final_library_size", 0)),
                "</div>",
                _html_table(["Rejection reason", "Count"], counts.get("rejection_reasons", [])),
                _html_table(
                    [
                        "Iteration",
                        "Generated",
                        "Admitted",
                        "Rejected",
                        "Replaced",
                        "Library Size",
                        "Yield",
                        "Best IC",
                        "Mean IC",
                    ],
                    [
                        [
                            row["iteration"],
                            row["generated"],
                            row["admitted"],
                            row["rejected"],
                            row["replaced"],
                            row["library_size"],
                            row["yield_rate"],
                            row["best_ic"],
                            row["mean_ic"],
                        ]
                        for row in counts.get("iteration_rows", [])
                    ],
                ),
                "</section>",
            ]
        )

    parts.append("<section><h2>Benchmarks</h2>")
    if not benchmarks:
        parts.append('<p class="muted">No benchmark JSON provided.</p>')
    else:
        for section in benchmarks:
            section_payload = section.get("payload", {})
            parts.append(f"<h3>{html.escape(str(section['label']))}</h3>")
            parts.append(f'<p class="muted">Source: {html.escape(str(section["source"]))}</p>')
            if section_payload.get("baseline") is not None:
                baseline = html.escape(str(section_payload.get("baseline")))
                parts.append(f"<p>Baseline: <code>{baseline}</code></p>")
            if section_payload.get("freeze_universe") is not None:
                freeze_universe = html.escape(str(section_payload.get("freeze_universe")))
                parts.append(f"<p>Freeze universe: <code>{freeze_universe}</code></p>")
            if section_payload.get("freeze_library_size") is not None:
                freeze_size = html.escape(str(section_payload.get("freeze_library_size")))
                parts.append(f"<p>Freeze library size: <code>{freeze_size}</code></p>")
            if section.get("freeze_top_k"):
                parts.append(
                    _html_table(
                        ["Name", "Category", "Train IC", "Train ICIR"],
                        [
                            [item["name"], item["category"], item["train_ic"], item["train_icir"]]
                            for item in section["freeze_top_k"]
                        ],
                    )
                )
            if section.get("universe_rows"):
                parts.append(
                    _html_table(
                        [
                            "Universe",
                            "Library IC",
                            "Library ICIR",
                            "Avg |rho|",
                            "Selected IC",
                            "Selected ICIR",
                            "Avg Turnover",
                            "Combo IC",
                            "Combo ICIR",
                            "Factor Count",
                        ],
                        [
                            [
                                row["universe"],
                                row["library_ic"],
                                row["library_icir"],
                                row["avg_abs_rho"],
                                row["selected_ic"],
                                row["selected_icir"],
                                row["avg_turnover"],
                                row["combo_ic"],
                                row["combo_icir"],
                                row["factor_count"],
                            ]
                            for row in section["universe_rows"]
                        ],
                    )
                )
    parts.extend(["</section>", "</body>", "</html>"])
    return "\n".join(parts)


def _card(label: str, value: Any) -> str:
    return (
        '<div class="card">'
        f'<div class="muted">{html.escape(str(label))}</div>'
        f"<div><strong>{html.escape(_html_cell_value(value))}</strong></div>"
        "</div>"
    )


def generate_report(
    library_source: JSONSource,
    *,
    session_log_source: JSONSource | None = None,
    benchmark_sources: Sequence[JSONSource] | None = None,
    format: str = "markdown",
    output_path: str | Path | None = None,
) -> str:
    """Generate a static report and optionally persist it to disk."""

    payload = build_report_payload(
        library_source,
        session_log_source=session_log_source,
        benchmark_sources=benchmark_sources,
    )
    if format == "markdown":
        report = render_markdown_report(payload)
    elif format == "html":
        report = render_html_report(payload)
    else:
        raise ValueError("format must be 'markdown' or 'html'")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")

    return report


def write_report(
    library_source: JSONSource,
    output_path: str | Path,
    *,
    session_log_source: JSONSource | None = None,
    benchmark_sources: Sequence[JSONSource] | None = None,
    format: str = "markdown",
) -> Path:
    """Generate a report and save it to ``output_path``."""

    generate_report(
        library_source,
        session_log_source=session_log_source,
        benchmark_sources=benchmark_sources,
        format=format,
        output_path=output_path,
    )
    return Path(output_path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for local report generation."""

    parser = argparse.ArgumentParser(description="Generate a static FactorMiner report")
    parser.add_argument("library", help="Path to factor_library.json or its base path")
    parser.add_argument(
        "--session-log", dest="session_log", default=None, help="Optional session_log.json path"
    )
    parser.add_argument(
        "--benchmark",
        dest="benchmarks",
        action="append",
        default=[],
        help="Optional benchmark JSON path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "html"),
        default="markdown",
        help="Report output format.",
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Write the report to this path instead of stdout."
    )
    args = parser.parse_args(argv)

    report = generate_report(
        args.library,
        session_log_source=args.session_log,
        benchmark_sources=args.benchmarks,
        format=args.format,
        output_path=args.output,
    )
    if args.output is None:
        print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
