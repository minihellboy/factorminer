"""Deterministic baseline formula catalogs for benchmark workflows."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from factorminer.core.library_io import PAPER_FACTORS


@dataclass(frozen=True)
class CandidateEntry:
    """One benchmark candidate formula."""

    name: str
    formula: str
    category: str


ALPHA101_CLASSIC: tuple[CandidateEntry, ...] = (
    CandidateEntry(
        "alpha101_close_return_rank",
        "Neg(CsRank(Return($close, 5)))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_intraday_position",
        "CsRank(Div(Sub($close, $open), Add(Sub($high, $low), 1e-8)))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_volume_reversal",
        "Neg(CsRank(Mul(Return($close, 5), Div($volume, Mean($volume, 20)))))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_vwap_gap",
        "Neg(CsRank(Div(Sub($close, $vwap), Add($vwap, 1e-8))))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_price_volume_corr",
        "Neg(CsRank(Corr(CsRank($close), CsRank($volume), 10)))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_range_volatility",
        "Neg(CsRank(Std(Div(Sub($high, $low), Add($close, 1e-8)), 20)))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_close_vs_mean",
        "Neg(CsRank(Div(Sub($close, Mean($close, 10)), Add(Std($close, 10), 1e-8))))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_turnover_rank",
        "Neg(CsRank(Div($amt, Add(Mean($amt, 20), 1e-8))))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_return_skew",
        "Neg(CsRank(Skew(Return($close, 1), 20)))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_trend_strength",
        "CsRank(TsRank(Return($close, 1), 20))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_volume_std",
        "Neg(CsRank(Div(Std($volume, 20), Add(Mean($volume, 20), 1e-8))))",
        "Alpha101 Classic",
    ),
    CandidateEntry(
        "alpha101_amount_momentum",
        "CsRank(Mul(Return($close, 10), Div($amt, Add(Mean($amt, 20), 1e-8))))",
        "Alpha101 Classic",
    ),
)

_WINDOW_PATTERN = re.compile(r"\b(5|10|20|30)\b")


def build_alpha101_adapted() -> list[CandidateEntry]:
    """Expand the classic catalog into frequency-adapted window variants."""
    variants: list[CandidateEntry] = []
    windows = (3, 6, 12, 24, 48)
    for entry in ALPHA101_CLASSIC:
        for window in windows:
            formula = _WINDOW_PATTERN.sub(str(window), entry.formula)
            variants.append(
                CandidateEntry(
                    name=f"{entry.name}_w{window}",
                    formula=formula,
                    category="Alpha101 Adapted",
                )
            )
    return variants


def build_random_exploration(seed: int, count: int = 160) -> list[CandidateEntry]:
    """Generate deterministic random-formula candidates from safe templates."""
    rng = np.random.RandomState(seed)
    unary_templates = [
        "Neg(CsRank(Return({feat}, {w1})))",
        "CsRank(TsRank({feat}, {w1}))",
        "Neg(CsRank(Div(Sub({feat}, Mean({feat}, {w1})), Add(Std({feat}, {w1}), 1e-8))))",
        "CsRank(Div(Std({feat}, {w1}), Add(Mean({feat}, {w2}), 1e-8)))",
        "Neg(CsRank(Skew({feat}, {w1})))",
    ]
    binary_templates = [
        "Neg(CsRank(Corr(CsRank({feat_a}), CsRank({feat_b}), {w1})))",
        "CsRank(Div(Sub({feat_a}, {feat_b}), Add(Std({feat_b}, {w1}), 1e-8)))",
        "Neg(CsRank(Mul(Return({feat_a}, {w1}), Div({feat_b}, Add(Mean({feat_b}, {w2}), 1e-8)))))",
        "CsRank(Cov({feat_a}, {feat_b}, {w1}))",
        "Neg(CsRank(Div(Sub(EMA({feat_a}, {w1}), EMA({feat_b}, {w2})), Add(Std({feat_a}, {w1}), 1e-8))))",
    ]
    features = ("$open", "$high", "$low", "$close", "$volume", "$amt", "$vwap", "$returns")
    windows = (3, 5, 10, 20, 30, 48)

    entries: list[CandidateEntry] = []
    for idx in range(count):
        use_binary = bool(rng.randint(0, 2))
        if use_binary:
            template = binary_templates[rng.randint(0, len(binary_templates))]
            feat_a, feat_b = rng.choice(features, size=2, replace=False)
            formula = template.format(
                feat_a=feat_a,
                feat_b=feat_b,
                w1=int(rng.choice(windows)),
                w2=int(rng.choice(windows)),
            )
        else:
            template = unary_templates[rng.randint(0, len(unary_templates))]
            formula = template.format(
                feat=rng.choice(features),
                w1=int(rng.choice(windows)),
                w2=int(rng.choice(windows)),
            )
        entries.append(
            CandidateEntry(
                name=f"random_exploration_{idx:03d}",
                formula=formula,
                category="Random Exploration",
            )
        )
    return entries


def build_gplearn_style(seed: int, count: int = 160) -> list[CandidateEntry]:
    """Build deeper deterministic mutation chains that mimic GP search."""
    base = build_random_exploration(seed + 17, count=max(count, 64))
    rng = np.random.RandomState(seed + 23)
    entries: list[CandidateEntry] = []
    for idx in range(count):
        left = base[idx % len(base)].formula
        right = base[rng.randint(0, len(base))].formula
        if idx % 3 == 0:
            formula = f"Neg(CsRank(Add({left}, {right})))"
        elif idx % 3 == 1:
            formula = f"CsRank(Div(Sub({left}, {right}), Add(Std($close, 10), 1e-8)))"
        else:
            formula = f"Neg(CsRank(Mul({left}, {right})))"
        entries.append(
            CandidateEntry(
                name=f"gplearn_style_{idx:03d}",
                formula=formula,
                category="GPLearn",
            )
        )
    return entries


def build_alphaforge_style() -> list[CandidateEntry]:
    """Reuse a diverse subset of the paper catalog for dynamic-combine baselines."""
    entries: list[CandidateEntry] = []
    for idx, factor in enumerate(PAPER_FACTORS[::2][:80]):
        entries.append(
            CandidateEntry(
                name=f"alphaforge_style_{idx:03d}",
                formula=factor["formula"],
                category="AlphaForge-style",
            )
        )
    return entries


def build_alphaagent_style() -> list[CandidateEntry]:
    """Reuse an alternate paper-catalog slice for LLM-style baseline proposals."""
    entries: list[CandidateEntry] = []
    for idx, factor in enumerate(PAPER_FACTORS[1::2][:80]):
        entries.append(
            CandidateEntry(
                name=f"alphaagent_style_{idx:03d}",
                formula=factor["formula"],
                category="AlphaAgent-style",
            )
        )
    return entries


def build_factor_miner_catalog() -> list[CandidateEntry]:
    """Expose the full paper factor catalog as benchmark candidates."""
    return [
        CandidateEntry(
            name=f"factor_miner_{idx + 1:03d}",
            formula=factor["formula"],
            category=factor["category"],
        )
        for idx, factor in enumerate(PAPER_FACTORS)
    ]


def entries_from_library(library) -> list[CandidateEntry]:
    """Convert a saved FactorLibrary into benchmark candidate entries."""
    return [
        CandidateEntry(name=factor.name, formula=factor.formula, category=factor.category)
        for factor in library.list_factors()
    ]


def dedupe_entries(entries: Iterable[CandidateEntry]) -> list[CandidateEntry]:
    """Remove duplicate formulas while preserving order."""
    seen: set[str] = set()
    unique: list[CandidateEntry] = []
    for entry in entries:
        if entry.formula in seen:
            continue
        seen.add(entry.formula)
        unique.append(entry)
    return unique
