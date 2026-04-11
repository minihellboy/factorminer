#!/usr/bin/env python3
"""HelixFactor End-to-End Demo

Demonstrates the complete system on synthetic data:
1. Generate realistic mock market data with planted alpha
2. Evaluate the paper's 110 factors on this data
3. Run the mining loop with MockProvider
4. Show factor combination and selection
5. Demonstrate Phase 2 features (causal, regime, significance, canonicalization)

No API keys needed - uses MockProvider for LLM generation.
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
import numpy as np

np.random.seed(42)

from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import PAPER_FACTORS
from factorminer.core.parser import try_parse
from factorminer.data.mock_data import MockConfig, generate_mock_data
from factorminer.data.preprocessor import preprocess
from factorminer.evaluation.combination import FactorCombiner
from factorminer.evaluation.metrics import (
    compute_ic,
    compute_ic_mean,
    compute_ic_win_rate,
    compute_icir,
)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    # ================================================================
    # STEP 1: Generate Mock Market Data
    # ================================================================
    section("STEP 1: Generate Mock Market Data")

    config = MockConfig(
        num_assets=100,
        num_periods=500,  # ~21 trading days of 10-min bars
        frequency="10min",
        plant_alpha=True,
        alpha_strength=0.03,
        alpha_assets_frac=0.3,
        seed=42,
    )
    raw_data = generate_mock_data(config)
    print(f"  Generated: {raw_data.shape[0]:,} rows")
    print(f"  Assets: {raw_data['asset_id'].nunique()}")
    print(f"  Periods: {raw_data.groupby('asset_id').size().iloc[0]}")
    print(f"  Columns: {list(raw_data.columns)}")
    print(f"  Date range: {raw_data['datetime'].min()} to {raw_data['datetime'].max()}")

    # Preprocess
    processed = preprocess(raw_data)
    print(f"  After preprocessing: {processed.shape[0]:,} rows")

    # Build data dict for expression tree evaluation
    assets = sorted(processed["asset_id"].unique())
    M = len(assets)
    T = processed.groupby("asset_id").size().min()

    # Pivot to (M, T) arrays
    data_dict = {}
    feature_map = {
        "$open": "open", "$high": "high", "$low": "low", "$close": "close",
        "$volume": "volume", "$amt": "amount", "$vwap": "vwap", "$returns": "returns",
    }
    for feat_name, col_name in feature_map.items():
        if col_name in processed.columns:
            pivot = processed.pivot(index="asset_id", columns="datetime", values=col_name)
            pivot = pivot.loc[assets].iloc[:, :T]
            data_dict[feat_name] = pivot.values.astype(np.float64)

    # Compute forward returns (target)
    close = data_dict["$close"]
    forward_returns = np.roll(close, -1, axis=1) / close - 1
    forward_returns[:, -1] = np.nan  # last period unknown

    print(f"  Data tensor: M={M} assets, T={T} periods, F={len(data_dict)} features")
    print(f"  Forward returns: shape={forward_returns.shape}")

    # ================================================================
    # STEP 2: Evaluate Paper's 110 Factors
    # ================================================================
    section("STEP 2: Evaluate Paper's 110 Factors on Mock Data")

    results = []
    parse_failures = 0
    eval_failures = 0

    t0 = time.time()
    for idx, factor_info in enumerate(PAPER_FACTORS):
        fid = idx + 1
        fname = factor_info["name"]
        formula = factor_info["formula"]
        category = factor_info["category"]

        tree = try_parse(formula)
        if tree is None:
            parse_failures += 1
            continue

        try:
            signals = tree.evaluate(data_dict)
            ic_series = compute_ic(signals, forward_returns)
            ic_mean = compute_ic_mean(ic_series)
            icir = compute_icir(ic_series)
            win_rate = compute_ic_win_rate(ic_series)

            results.append({
                "id": fid,
                "name": fname,
                "formula": formula,
                "category": category,
                "ic_mean": ic_mean,
                "icir": icir,
                "win_rate": win_rate,
                "signals": signals,
                "ic_series": ic_series,
            })
        except Exception:
            eval_failures += 1

    elapsed = time.time() - t0
    print(f"  Evaluated {len(results)} factors in {elapsed:.1f}s")
    print(f"  Parse failures: {parse_failures}, Eval failures: {eval_failures}")

    # Sort by |IC|
    results.sort(key=lambda x: abs(x["ic_mean"]), reverse=True)

    print("\n  Top 20 Factors by |IC|:")
    print(f"  {'ID':<5} {'Name':<40} {'Cat':<15} {'IC':>8} {'ICIR':>8} {'Win%':>6}")
    print(f"  {'-'*5} {'-'*40} {'-'*15} {'-'*8} {'-'*8} {'-'*6}")
    for r in results[:20]:
        print(f"  {r['id']:<5} {r['name'][:40]:<40} {r['category'][:15]:<15} "
              f"{r['ic_mean']:>8.4f} {r['icir']:>8.3f} {r['win_rate']:>5.1%}")

    # Category breakdown
    print("\n  Category Breakdown:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(abs(r["ic_mean"]))
    for cat, ics in sorted(categories.items(), key=lambda x: -np.mean(x[1])):
        print(f"    {cat:<25} {len(ics):>3} factors  avg|IC|={np.mean(ics):.4f}")

    # ================================================================
    # STEP 3: Build Factor Library with Admission Rules
    # ================================================================
    section("STEP 3: Build Factor Library (IC > 0.02, corr < 0.5)")

    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.02)
    admitted = 0
    rejected_ic = 0
    rejected_corr = 0

    for r in results:
        ic_abs = abs(r["ic_mean"])
        if ic_abs < 0.02:
            rejected_ic += 1
            continue

        # Check correlation with existing library
        can_admit = True
        max_corr = 0.0
        for existing_id, existing_factor in library.factors.items():
            if existing_factor.signals is not None:
                corr = library.compute_correlation(r["signals"], existing_factor.signals)
                max_corr = max(max_corr, abs(corr))
                if abs(corr) >= 0.5:
                    can_admit = False
                    break

        if not can_admit:
            rejected_corr += 1
            continue

        factor = Factor(
            id=library._next_id,
            name=r["name"],
            formula=r["formula"],
            category=r["category"],
            ic_mean=r["ic_mean"],
            icir=r["icir"],
            ic_win_rate=r["win_rate"],
            max_correlation=max_corr,
            batch_number=1,
            admission_date="2024-01-01",
            signals=r["signals"],
        )
        library.admit_factor(factor)
        admitted += 1

    print(f"  Admitted: {admitted}")
    print(f"  Rejected (IC < 0.02): {rejected_ic}")
    print(f"  Rejected (correlation >= 0.5): {rejected_corr}")
    print(f"  Library size: {library.size}")

    if library.size > 0:
        diag = library.get_diagnostics()
        print(f"  Avg |rho|: {diag.get('avg_correlation', 0):.4f}")

    # ================================================================
    # STEP 4: Factor Combination
    # ================================================================
    if library.size >= 3:
        section("STEP 4: Factor Combination Methods")

        factor_signals = {}
        ic_values = {}
        for fid, factor in library.factors.items():
            if factor.signals is not None:
                factor_signals[fid] = factor.signals
                ic_values[fid] = factor.ic_mean

        combiner = FactorCombiner()

        # Equal weight
        ew = combiner.equal_weight(factor_signals)
        ew_ic = compute_ic(ew, forward_returns)
        print(f"  Equal-Weight:  IC={compute_ic_mean(ew_ic):.4f}, "
              f"ICIR={compute_icir(ew_ic):.3f}, "
              f"Win={compute_ic_win_rate(ew_ic):.1%}")

        # IC-weighted
        icw = combiner.ic_weighted(factor_signals, ic_values)
        icw_ic = compute_ic(icw, forward_returns)
        print(f"  IC-Weighted:   IC={compute_ic_mean(icw_ic):.4f}, "
              f"ICIR={compute_icir(icw_ic):.3f}, "
              f"Win={compute_ic_win_rate(icw_ic):.1%}")

        # Orthogonal
        try:
            ortho = combiner.orthogonal(factor_signals)
            ortho_ic = compute_ic(ortho, forward_returns)
            print(f"  Orthogonal:    IC={compute_ic_mean(ortho_ic):.4f}, "
                  f"ICIR={compute_icir(ortho_ic):.3f}, "
                  f"Win={compute_ic_win_rate(ortho_ic):.1%}")
        except Exception as e:
            print(f"  Orthogonal:    skipped ({e})")

    # ================================================================
    # STEP 5: Phase 2 - Regime Detection
    # ================================================================
    section("STEP 5: Phase 2 - Regime-Aware Analysis")

    from factorminer.evaluation.regime import RegimeAwareEvaluator, RegimeConfig, RegimeDetector

    regime_config = RegimeConfig(lookback_window=30, min_regime_ic=0.01, min_regimes_passing=2)
    detector = RegimeDetector(regime_config)
    classification = detector.classify(forward_returns)

    from factorminer.evaluation.regime import MarketRegime
    for regime in MarketRegime:
        mask = classification.periods[regime]
        n = int(mask.sum())
        stats = classification.stats[regime]
        print(f"  {regime.value:>10}: {n:>4} periods "
              f"(avg_ret={stats['mean_return']:.4f}, vol={stats['volatility']:.4f})")

    if library.size > 0:
        evaluator = RegimeAwareEvaluator(forward_returns, classification, regime_config)
        best_factor = list(library.factors.values())[0]
        if best_factor.signals is not None:
            regime_result = evaluator.evaluate(best_factor.name, best_factor.signals)
            print(f"\n  Top factor '{best_factor.name}' regime analysis:")
            for regime, ic_val in regime_result.regime_ic.items():
                print(f"    {regime.value:>10}: IC={ic_val:.4f}")
            print(f"    Regimes passing: {regime_result.n_regimes_passing}, Passes: {regime_result.passes}")

    # ================================================================
    # STEP 6: Phase 2 - Statistical Significance
    # ================================================================
    section("STEP 6: Phase 2 - Statistical Significance Testing")

    from factorminer.evaluation.significance import (
        BootstrapICTester,
        FDRController,
        SignificanceConfig,
    )

    sig_config = SignificanceConfig(bootstrap_n_samples=500, bootstrap_block_size=10)
    bootstrap = BootstrapICTester(sig_config)
    fdr = FDRController(sig_config)

    if library.size > 0:
        # Bootstrap CI for top 5 factors
        print("  Bootstrap 95% CI for top factors:")
        p_values = {}
        for fid, factor in list(library.factors.items())[:5]:
            ic_series = compute_ic(factor.signals, forward_returns)
            ci = bootstrap.compute_ci(factor.name, ic_series)
            p_val = bootstrap.compute_p_value(ic_series)
            p_values[factor.name] = p_val
            print(f"    {factor.name[:35]:<35} IC={ci.ic_mean:.4f} "
                  f"CI=[{ci.ci_lower:.4f}, {ci.ci_upper:.4f}] "
                  f"{'*' if ci.ci_excludes_zero else ' '} p={p_val:.4f}")

        # FDR correction
        if len(p_values) >= 2:
            fdr_result = fdr.apply_fdr(p_values)
            print(f"\n  FDR Correction (BH at {sig_config.fdr_level}):")
            print(f"    Significant discoveries: {fdr_result.n_discoveries}/{len(p_values)}")

    # ================================================================
    # STEP 7: Phase 2 - SymPy Canonicalization
    # ================================================================
    section("STEP 7: Phase 2 - SymPy Formula Canonicalization")

    from factorminer.core.canonicalizer import FormulaCanonicalizer

    canon = FormulaCanonicalizer()

    test_pairs = [
        ("Neg(Neg($close))", "$close", True),
        ("Add($close, $open)", "Add($open, $close)", True),
        ("CsRank(Neg($close))", "Neg(CsRank($close))", False),
        ("Mul($close, Div($open, $close))", "$open", True),
    ]

    print("  Equivalence Detection:")
    for f1, f2, expected in test_pairs:
        t1 = try_parse(f1)
        t2 = try_parse(f2)
        if t1 and t2:
            h1 = canon.canonicalize(t1)
            h2 = canon.canonicalize(t2)
            is_dup = h1 == h2
            status = "CORRECT" if is_dup == expected else "WRONG"
            sym = "==" if is_dup else "!="
            print(f"    {f1:>35} {sym} {f2:<35} [{status}]")

    # ================================================================
    # STEP 8: Phase 2 - Knowledge Graph
    # ================================================================
    section("STEP 8: Phase 2 - Knowledge Graph Memory")

    import re

    from factorminer.memory.knowledge_graph import FactorKnowledgeGraph, FactorNode

    kg = FactorKnowledgeGraph()

    for fid, factor in list(library.factors.items())[:20]:
        operators = re.findall(r"([A-Z][a-zA-Z]+)\(", factor.formula)
        features = re.findall(r"\$[a-z]+", factor.formula)
        node = FactorNode(
            factor_id=str(fid),
            formula=factor.formula,
            ic_mean=factor.ic_mean,
            category=factor.category,
            operators=list(set(operators)),
            features=list(set(features)),
            batch_number=1,
            admitted=True,
        )
        kg.add_factor(node)

    # Add correlation edges
    factor_list = list(library.factors.values())[:20]
    for i in range(len(factor_list)):
        for j in range(i + 1, len(factor_list)):
            if factor_list[i].signals is not None and factor_list[j].signals is not None:
                corr = library.compute_correlation(factor_list[i].signals, factor_list[j].signals)
                if abs(corr) > 0.3:
                    kg.add_correlation_edge(str(factor_list[i].id), str(factor_list[j].id), abs(corr), threshold=0.3)

    print(f"  Knowledge Graph: {kg.get_factor_count()} factor nodes, {kg.get_edge_count()} edges")

    saturated = kg.find_saturated_regions(threshold=0.3)
    print(f"  Saturated clusters (rho > 0.3): {len(saturated)}")
    for i, cluster in enumerate(saturated[:3]):
        print(f"    Cluster {i+1}: {len(cluster)} factors")

    cooccur = kg.get_operator_cooccurrence()
    if cooccur:
        top_pairs = sorted(cooccur.items(), key=lambda x: -x[1])[:5]
        print("  Top operator co-occurrences:")
        for (op1, op2), count in top_pairs:
            print(f"    ({op1}, {op2}): {count} times")

    # ================================================================
    # STEP 9: Mining Loop Demo (3 iterations)
    # ================================================================
    section("STEP 9: Mining Loop Demo (3 iterations with MockProvider)")

    from factorminer.agent.llm_interface import MockProvider
    from factorminer.core.config import MiningConfig
    from factorminer.core.ralph_loop import RalphLoop

    mining_config = MiningConfig(
        target_library_size=20,
        batch_size=10,
        max_iterations=3,
        ic_threshold=0.02,
        correlation_threshold=0.5,
        fast_screen_assets=50,
        num_workers=1,
        signal_failure_policy="synthetic",
    )

    loop = RalphLoop(
        config=mining_config,
        data_tensor=np.stack(list(data_dict.values()), axis=-1),  # (M, T, F)
        returns=forward_returns,
        llm_provider=MockProvider(),
    )

    print("  Running 3 mining iterations...")
    t0 = time.time()
    result_library = loop.run(target_size=20, max_iterations=3)
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Library size: {result_library.size}")
    if result_library.size > 0:
        print("\n  Admitted factors:")
        for fid, f in result_library.factors.items():
            print(f"    [{fid}] {f.name[:50]} IC={f.ic_mean:.4f} ICIR={f.icir:.3f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    section("SUMMARY")

    print(f"  Mock data: {M} assets x {T} periods")
    print(f"  Paper factors evaluated: {len(results)}/110")
    print(f"  Factors with |IC| > 0.02: {sum(1 for r in results if abs(r['ic_mean']) > 0.02)}")
    print(f"  Library (admission-filtered): {library.size} factors")
    print(f"  Mining loop (3 iter): {result_library.size} factors discovered")
    print("")
    print("  Phase 2 Features Demonstrated:")
    print("    Regime detection:     3 regimes classified")
    print("    Statistical testing:  Bootstrap CI + FDR")
    print("    Canonicalization:     Neg(Neg(x))==x detected")
    print(f"    Knowledge graph:      {kg.get_factor_count()} nodes, {kg.get_edge_count()} edges")
    print("")
    print("  To run with real LLM (e.g., Claude):")
    print("    export ANTHROPIC_API_KEY=sk-ant-...")
    print("    factorminer mine --config factorminer/configs/default.yaml")
    print("")
    print("  To run with real market data:")
    print("    Place CSV with [datetime,asset_id,open,high,low,close,volume,amount]")
    print("    at data/market.csv and update configs/default.yaml")


if __name__ == "__main__":
    main()
