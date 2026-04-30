# Paper Claims Matrix

This page maps the paper claims to the current public repository. It is meant
to make the boundary between implemented, reproducible, proxy, and
user-provided-data work explicit.

Status legend:

- `implemented`: available in the repo and covered by normal commands/tests.
- `partial`: available, but not yet a complete reproduction of the paper
  artifact or benchmark.
- `proxy`: a deterministic local stand-in, not the original external method.
- `requires data`: needs user-provided market data that is not bundled.
- `not bundled`: paper artifact or private data is not included in the repo.

## Core System

| Paper claim | Repo status | Notes |
| --- | --- | --- |
| Typed formula DSL over OHLCV-style features | implemented | Features are `$open`, `$high`, `$low`, `$close`, `$volume`, `$amt`, `$vwap`, and `$returns`. |
| 60+ financial operators | implemented | The registry includes canonical operators plus paper-style DSL names such as `SignedPower`, `Med`, `Rsquare`, `Slope`, `Resi`, `Eq`, `Min2`, `Max2`, `TsDecay`, and `Scale`. |
| Paper-style operator syntax from Appendix P | implemented | Users can now paste representative paper formulas without manually renaming operators. Built-in paper factors remain normalized to the repo's canonical DSL. |
| Ralph mining loop | implemented | `RalphLoop` is the paper-facing loop. |
| Experience memory with success, failure, and insight retrieval | implemented | Default memory patterns are seeded from the paper's success and forbidden-direction examples. |
| Multi-stage validation: IC screen, correlation screen, replacement, dedup, full validation | implemented | The evaluation pipeline and library admission code implement these gates. |
| Paper IC semantics | implemented | Paper-mode admission and selection use `ic_paper_mean = abs(mean(IC_t))` and `ic_paper_icir = abs(mean(IC_t)) / std(IC_t)`. See [Metric Semantics](metrics.md). |
| 110 discovered factors | implemented | `PAPER_FACTORS` contains the built-in 110-factor catalog, normalized to the repo DSL. |

## Data And Reproduction

| Paper claim | Repo status | Notes |
| --- | --- | --- |
| A-share 10-minute bars for CSI500, CSI1000, and HS300 | requires data | The repo does not bundle the paper's A-share data. Users must provide compatible OHLCV/amount data. |
| Crypto 10-minute bars for 64 Binance assets | requires data | The bundled Binance file is a small 5-minute sample for onboarding, not the full paper benchmark dataset. |
| 2024 train and held-out 2025 test protocol | partial | Configs expose paper-style split fields. Exact reproduction depends on user-provided data and universe membership. |
| Next 10-minute open-to-close target | implemented | The tensor builder supports the paper-style target contract. |
| Table 1-style evaluation | partial | The benchmark runner supports the flow, but paper-level numeric reproduction requires paper-equivalent data and real baseline implementations. |

## Benchmarks And Baselines

| Paper baseline | Repo status | Notes |
| --- | --- | --- |
| FactorMiner | implemented | Uses the built-in paper catalog or a saved library/runtime run, depending on command inputs. |
| FactorMiner without memory | partial/proxy | Runtime loop support exists, but the catalog fallback is a synthetic no-memory proxy unless a saved/runtime run is supplied. |
| Alpha101 Classic | partial | Current catalog is a deterministic subset, not the full 101-factor implementation. |
| Alpha101 Adapted | partial | Current variants expand the subset over window choices. Full paper fidelity needs all Alpha101 formulas and the paper's adaptation protocol. |
| Random Formula Exploration | partial | Current generator is deterministic and safe, but not yet a full typed random expression-tree search. |
| GPLearn | proxy | Current `gplearn` baseline is a GP-style deterministic catalog, not the external GPLearn algorithm. |
| AlphaForge | proxy | Current `alphaforge_style` baseline uses catalog-derived stand-ins. |
| AlphaAgent | proxy | Current `alphaagent_style` baseline uses catalog-derived stand-ins. |

See [Benchmark Baselines](baselines.md) for the detailed provenance rules.

## Performance Claims

| Paper claim | Repo status | Notes |
| --- | --- | --- |
| NumPy/Pandas, C/Bottleneck, and GPU/PyTorch-style backends | partial | NumPy, C fallback, and GPU registry paths exist, but some runtime recomputation paths still need stricter backend routing before making paper-level speed claims. |
| A100 performance table | not bundled | The repo does not ship A100 hardware measurements. Benchmark outputs should be interpreted on the hardware used locally. |
| 1000-factor evaluation efficiency comparison | partial | Efficiency benchmark commands exist, but reproducing paper numbers requires paper-scale data and comparable hardware. |

## Known Next Fidelity Work

1. Replace Alpha101 subset with the full 101-formula catalog.
2. Replace deterministic random templates with a typed bounded-depth random tree sampler.
3. Add optional real GPLearn adapter when the dependency is installed.
4. Add stricter `paper-audit` data checks for 10-minute frequency, 2024/2025 splits, universe size, and target construction.
5. Route all runtime recomputation through backend-aware operator execution before publishing GPU speed claims.
