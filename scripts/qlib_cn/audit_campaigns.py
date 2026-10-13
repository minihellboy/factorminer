"""Correct each campaign's admitted factors for the whole search that produced them.

    uv run python scripts/qlib_cn/audit_campaigns.py \\
        --campaigns output/qlib_cn/campaigns/classic_memory_c1 ... \\
        --scores output/qlib_cn/csi300/classic --out output/qlib_cn/audit.json

Every candidate a campaign evaluated is in its tamper-evident trial ledger
(verified on load), with its train-split daily RankIC series. Per campaign:

* each trial's Newey-West t of mean RankIC and two-sided p-value; Benjamini-
  Hochberg q-values over *all* trials, not just the admitted ones;
* the effective number of independent trials (Bailey & Lopez de Prado
  average-correlation estimate) and two Deflated Sharpe hurdles for each
  admitted factor's oriented IC series: ``dsr_null`` uses the null sampling
  variance 1/(T-1) (does it beat the best of N noise-only trials?), ``dsr``
  the observed cross-trial variance of signed per-period Sharpe ratios
  (does it beat the best of N draws from this search's own spread, real
  effects included? -- the strict reading, which a search whose trials
  are mostly genuine but similar cannot pass by construction);
* for marginal-admission campaigns, BH q-values of the partial-IC t over all
  trials that carry marginal statistics;
* if ``evaluate_library.py`` scored the library (``<scores>/<campaign>/
  factors.csv``), each admitted factor's test RankIC under its train sign.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from factorminer.architecture.trial_ledger import TrialLedger, canonical_formula_identity
from factorminer.evaluation.evidence.inference import compute_hac_mean_test
from factorminer.evaluation.significance import (
    DeflatedSharpeCalculator,
    FDRController,
    SignificanceConfig,
    estimate_effective_trials,
)


def _bh(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    result = FDRController(SignificanceConfig()).apply_fdr(p_values)
    return {key: float(value) for key, value in result.adjusted_p_values.items()}


def _trials(ledger_path: Path) -> dict[str, dict]:
    """One record per canonical formula, preferring evaluated ones."""
    TrialLedger.load(ledger_path)  # raises if the hash chain was altered
    trials: dict[str, dict] = {}
    for line in ledger_path.read_text().splitlines():
        record = json.loads(line)
        key = record["canonical_formula_hash"]
        if key not in trials or (record.get("ic_series") and not trials[key].get("ic_series")):
            trials[key] = record
    return trials


def _audit(campaign: Path, scores: Path | None) -> dict:
    trials = _trials(campaign / "global_trial_ledger.jsonl")
    series = {}
    for key, record in trials.items():
        values = np.array([np.nan if v is None else v for v in record.get("ic_series") or []],
                          dtype=np.float64)
        if np.isfinite(values).sum() >= 30:
            series[key] = values
    stats = {}
    for key, values in series.items():
        finite = values[np.isfinite(values)]
        t = compute_hac_mean_test(finite).t_stat
        sd = finite.std(ddof=1)
        stats[key] = {"t": float(t), "p": float(2 * norm.sf(abs(t))) if np.isfinite(t) else 1.0,
                      "sr": float(finite.mean() / sd) if sd > 0 else 0.0}
    q = _bh({key: row["p"] for key, row in stats.items()})
    lengths = {len(v) for v in series.values()}
    effective = estimate_effective_trials(np.stack(list(series.values()))) if len(lengths) == 1 else None
    n_eff = effective.effective_trials if effective else float(len(series))
    sr_variance = float(np.var([row["sr"] for row in stats.values()], ddof=1)) if len(stats) > 1 else None

    marginal_p = {}
    for key, record in trials.items():
        marginal = (record.get("metadata") or {}).get("marginal")
        if marginal:
            marginal_p[key] = float(2 * norm.sf(abs(marginal["partial_ic_t"])))
    marginal_q = _bh(marginal_p)

    oos = {}
    path = scores / campaign.name / "factors.csv" if scores is not None else None
    if path is not None and path.exists() and path.stat().st_size > 1:
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            if isinstance(row.get("formula"), str) and pd.notna(row.get("test_rank_ic")):
                oos[row["formula"]] = (float(row["test_rank_ic"]), float(row["test_hac_t"]))

    library = json.loads((campaign / "factor_library.json").read_text())["factors"]
    dsr = DeflatedSharpeCalculator(SignificanceConfig())
    factors = []
    for factor in library:
        _, key = canonical_formula_identity(factor["formula"])
        row = {"name": factor["name"], "formula": factor["formula"]}
        if key in series:
            values = series[key][np.isfinite(series[key])]
            oriented = values * (1.0 if values.mean() >= 0 else -1.0)
            result = dsr.compute(factor["name"], oriented, n_trials=n_eff,
                                 trial_sharpe_variance=sr_variance)
            null = dsr.compute(factor["name"], oriented, n_trials=n_eff)
            row.update(train_t=stats[key]["t"], q=q.get(key, 1.0), dsr_z=result.deflated_sharpe,
                       dsr_p=result.p_value, dsr_null_p=null.p_value)
        if key in marginal_q:
            row["marginal_q"] = marginal_q[key]
        if factor["formula"] in oos:
            row["test_rank_ic"], row["test_t"] = oos[factor["formula"]]
        factors.append(row)

    def share(predicate, rows):
        rows = list(rows)
        return float(np.mean([predicate(r) for r in rows])) if rows else float("nan")

    scored = [r for r in factors if "test_rank_ic" in r]
    summary = {
        "campaign": campaign.name,
        "trials": len(trials), "trials_with_series": len(series),
        "effective_trials": n_eff,
        "average_trial_correlation": effective.average_pairwise_correlation if effective else None,
        "admitted": len(factors),
        "admitted_q_below_0.05": share(lambda r: r.get("q", 1) < 0.05, factors),
        "trial_sharpe_sd": math.sqrt(sr_variance) if sr_variance is not None else None,
        "admitted_dsr_null_p_below_0.05": share(lambda r: r.get("dsr_null_p", 1) < 0.05, factors),
        "admitted_dsr_p_below_0.05": share(lambda r: r.get("dsr_p", 1) < 0.05, factors),
        "admitted_marginal_q_below_0.05": share(lambda r: r.get("marginal_q", 1) < 0.05,
                                                [r for r in factors if "marginal_q" in r]),
        "test_sign_held": share(lambda r: r["test_rank_ic"] > 0, scored),
        "test_sign_held_dsr_null_pass": share(lambda r: r["test_rank_ic"] > 0,
                                              [r for r in scored if r.get("dsr_null_p", 1) < 0.05]),
        "test_sign_held_dsr_null_fail": share(lambda r: r["test_rank_ic"] > 0,
                                              [r for r in scored if r.get("dsr_null_p", 1) >= 0.05]),
        "test_mean_rank_ic": float(np.mean([r["test_rank_ic"] for r in scored])) if scored else math.nan,
    }
    return {"summary": summary, "factors": factors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaigns", type=Path, nargs="+", required=True)
    parser.add_argument("--scores", type=Path, help="root holding <campaign>/factors.csv")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    audits = [_audit(campaign, args.scores) for campaign in args.campaigns]
    if args.out:
        args.out.write_text(json.dumps(audits, indent=2) + "\n")
    keys = ("trials", "effective_trials", "admitted", "admitted_q_below_0.05",
            "admitted_dsr_null_p_below_0.05", "admitted_dsr_p_below_0.05", "admitted_marginal_q_below_0.05", "test_sign_held",
            "test_mean_rank_ic")
    print(pd.DataFrame([a["summary"] for a in audits]).set_index("campaign")[list(keys)]
          .round(3).to_string())


if __name__ == "__main__":
    main()
