# Factor formula DSL — operator reference

A FactorMiner factor is an **expression tree**: operators applied to market-data
leaves. Leaves are the per-(asset, time) features `open`, `high`, `low`,
`close`, `volume`, `amount`, `vwap`, and `returns`. The engine ships **84
operators** in 7 categories; the operator registry
(`factorminer/operators/registry.py`) is the single source of truth.

## Categories

**Arithmetic (21)** — `Abs Add Clip Div Exp Inv Log Max Max2 Min Min2 Mul Neg Pow Power Sign SignedPower Sqrt Square Sub Tanh`

**Time-series (15)** — `Beta Corr Cov CumMax CumMin CumProd CumSum Decay Delay Delta LogReturn Resid Return TsDecay WMA`

**Cross-sectional (7)** — `CsDemean CsNeutralize CsQuantile CsRank CsScale CsZScore Scale`

**Statistical (18)** — `CountNaN CountNotNaN Kurt Mean Med Median Prod Product Quantile Skew Std Sum TsArgMax TsArgMin TsMax TsMin TsRank Var`

**Smoothing (5)** — `DEMA EMA HMA KAMA SMA`

**Regression (7)** — `Resi Rsquare Slope TsLinReg TsLinRegIntercept TsLinRegResid TsLinRegSlope`

**Logical (11)** — `And Eq Equal Greater GreaterEqual IfElse Less LessEqual Ne Not Or`

## Reading a formula

- **Cross-sectional** operators (`Cs*`, `Scale`) rank or normalize *across assets*
  at each timestamp — the basis of most alpha signals.
- **Time-series** operators (`Ts*`, `Delay`, `Delta`, `Decay`, windowed stats)
  take a lookback window and act *along time* per asset.
- A typical momentum factor: `CsRank(Sub(close, Delay(close, 20)))` — rank the
  20-bar price change across the universe.
- A typical reversal factor: `Neg(CsRank(Return(close, 5)))`.

## Why this matters for mining

- The LLM proposes formulas in this DSL; the parser
  (`factorminer/core/parser.py`) and expression tree validate them.
- The Helix loop's **canonicalization** stage uses SymPy to fold formulas that
  are mathematically identical (e.g. `Add(a,b)` vs `Add(b,a)`) so the library
  does not fill with disguised duplicates.
- **Auto-inventor** (a Phase 2 feature) lets the LLM propose *new* operators,
  which are sandbox-validated before joining the registry.

When reporting a factor, always quote its formula verbatim — it is the
reproducible definition of the signal.
