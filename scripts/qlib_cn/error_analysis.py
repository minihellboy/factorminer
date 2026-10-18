"""Error analysis of the book: which features separate its false positives from its true positives.

    $QPY -I scripts/qlib_cn/error_analysis.py --universe csi300 --protocol wf2020 \\
        --out output/qlib_cn/error_analysis/csi300_wf2020

On every day of the window's train+valid span (never its test days), the
stage-1 score s1 (the frozen cross-fitted panel written by ``two_stage.py``)
defines the candidate set C_t (top ``--cand`` names) and the book (top
``--topk``). A held name is a *false positive* when its 5-day return is below
the C_t median, a *true positive* otherwise. Every Alpha158 feature is
z-scored within C_t per day; the report gives, per feature, the mean z of the
false positives minus that of the true positives, its HAC t over days, and
the same contrast before and after ``--regime`` (2017-01-01, when short-term
reversal lost its long side). Features are grouped into Alpha158 families by
name prefix.

Outputs ``<out>.json`` (the full table) and ``<out>.md``, a research brief for
the mining prompt (``mining.task_brief``): instead of "find orthogonal alpha",
the model is asked to explain this residual, with the contrasts as evidence.
A feature that is higher among false positives is a feature the book should
be penalised for at the top; the mined factor is judged by its partial IC
within C_t after s1 and the Alpha158 components are removed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_lift import _hac_t  # noqa: E402
from fast_fit import _load, _segment  # noqa: E402
from run_qlib_baselines import PROTOCOLS, TOPK  # noqa: E402

FAMILIES = {
    "K": "candle shape (KMID, KLEN, KUP, KLOW, KSFT: body, range, shadows of today's bar)",
    "OPEN": "today's open relative to close", "HIGH": "today's high relative to close",
    "LOW": "today's low relative to close", "VWAP": "today's vwap relative to close",
    "ROC": "rate of change (close / close d days ago)", "MA": "moving average of close relative to close",
    "STD": "volatility of close", "BETA": "slope of close on time", "RSQR": "R-squared of close on time",
    "RESI": "residual of close on time", "MAX": "highest high relative to close", "MIN": "lowest low relative to close",
    "QTLU": "upper quantile of close", "QTLD": "lower quantile of close", "RANK": "rank of today's close in the window",
    "RSV": "position of close in the high-low range", "IMAX": "days since the window high", "IMIN": "days since the window low",
    "IMXD": "days since high minus days since low", "CORR": "correlation of close and log volume",
    "CORD": "correlation of close change and volume change", "CNTP": "share of up days", "CNTN": "share of down days",
    "CNTD": "up-day share minus down-day share", "SUMP": "share of gains in total absolute change",
    "SUMN": "share of losses in total absolute change", "SUMD": "gain share minus loss share",
    "VMA": "volume relative to its moving average", "VSTD": "volume volatility relative to volume",
    "WVMA": "volume-weighted price change volatility", "VSUMP": "share of volume rises", "VSUMN": "share of volume falls",
    "VSUMD": "volume rise share minus fall share",
}


def _family(feature: str) -> str:
    if feature.startswith("K"):
        return "K"
    return re.sub(r"\d+$", "", feature)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("~/fm-data/cache"))
    parser.add_argument("--universe", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--stage1-oof", type=Path, default=None,
                        help="frozen s1 panel (default: the window's _stage1_alpha158_h5_lgb_s06_oof.parquet)")
    parser.add_argument("--runs", type=Path, default=Path("output/qlib_cn"))
    parser.add_argument("--cand", type=int, default=int(2.5 * TOPK))
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--regime", default="2017-01-01")
    parser.add_argument("--top", type=int, default=15, help="features listed in the brief")
    parser.add_argument("--out", type=Path, required=True, help="path stem for .json and .md")
    args = parser.parse_args()

    split = PROTOCOLS[args.protocol]
    features, labels = _load(args.cache, args.universe, split["start"])
    span = (split["start"], split["valid"][1])
    x = _segment(features, *span)
    r5 = _segment(labels["h5"], *span).astype(np.float64)
    oof = args.stage1_oof or args.runs / args.universe / args.protocol / "_stage1_alpha158_h5_lgb_s06_oof.parquet"
    s1 = pd.read_parquet(oof).set_index(["datetime", "instrument"])["s1"].reindex(x.index)
    day = s1.groupby(level="datetime")
    rank = day.rank(ascending=False, method="first")
    cand = (rank <= args.cand) & r5.notna()
    x, r5, rank = x[cand], r5[cand], rank[cand]
    median = r5.groupby(level="datetime").transform("median")
    book = rank <= args.topk
    fp, tp = book & (r5 < median), book & (r5 >= median)
    z = x.groupby(level="datetime").transform(lambda f: (f - f.mean()) / f.std()).clip(-5, 5)
    regime = pd.Timestamp(args.regime)

    def contrast(mask_a: pd.Series, mask_b: pd.Series) -> pd.DataFrame:
        a = z[mask_a].groupby(level="datetime").mean()
        b = z[mask_b].groupby(level="datetime").mean()
        return (a - b).dropna(how="all")

    daily = contrast(fp, tp)
    rows = []
    for feature in z.columns:
        d = daily[feature].dropna()
        pre, post = d[d.index < regime], d[d.index >= regime]
        rows.append({"feature": feature, "family": _family(feature), "fp_minus_tp_z": float(d.mean()),
                     "t": _hac_t(d.to_numpy()), "pre_regime": float(pre.mean()) if len(pre) else None,
                     "pre_t": _hac_t(pre.to_numpy()) if len(pre) > 10 else None,
                     "post_regime": float(post.mean()) if len(post) else None,
                     "post_t": _hac_t(post.to_numpy()) if len(post) > 10 else None})
    table = pd.DataFrame(rows).sort_values("t", key=np.abs, ascending=False)
    family = (table.assign(abs_t=table["t"].abs()).groupby("family")
              .agg(features=("feature", "size"), mean_contrast=("fp_minus_tp_z", "mean"), max_abs_t=("abs_t", "max"))
              .sort_values("max_abs_t", ascending=False))
    hit = float((tp.groupby(level="datetime").sum() / book.groupby(level="datetime").sum()).mean())
    spread = float(((r5[book] - median[book]).groupby(level="datetime").mean() * 1e4).mean())
    summary = {"universe": args.universe, "protocol": args.protocol, "span": span, "days": int(daily.shape[0]),
               "cand": args.cand, "topk": args.topk, "true_positive_share": hit, "book_minus_cand_median_bps": spread,
               "regime": args.regime, "features": table.to_dict(orient="records"),
               "families": family.reset_index().to_dict(orient="records")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    lines = [
        f"# Residual-explanation task: {args.universe} {args.protocol}, {span[0]}..{span[1]}",
        "",
        f"A LightGBM score on Alpha158 ranks the universe each day. The top {args.cand} names are the candidate set "
        f"and the top {args.topk} are the book. A held name is a FALSE POSITIVE when its next-5-day return is below the "
        f"candidate-set median, a TRUE POSITIVE otherwise. Over {daily.shape[0]} days the book was right "
        f"{hit:.0%} of the time and earned {spread:+.1f} bps per 5 days over the candidate median.",
        "",
        "Your task: propose factors (in the DSL below, from the available leaves) that are HIGHER for the true "
        "positives than for the false positives within the candidate set, i.e. information the score does not use "
        "that separates winners from losers among the names it already likes. A factor is judged by its partial "
        "rank IC with the 5-day return WITHIN the candidate set, after the stage-1 score and the Alpha158 "
        "principal components are removed. Do not restate the score, Alpha158, or short-term reversal.",
        "",
        "Evidence: mean z-score (within the candidate set) of false positives minus true positives, with HAC t over "
        f"days, overall and before/after {args.regime}. A positive value means the book's losers score higher on it.",
        "",
        "| feature | family | FP - TP (z) | t | pre | t | post | t |", "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.head(args.top).iterrows():
        fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"  # noqa: E731
        fmt_t = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"  # noqa: E731
        lines.append(f"| {r['feature']} | {r['family']} | {fmt(r['fp_minus_tp_z'])} | {fmt_t(r['t'])} | "
                     f"{fmt(r['pre_regime'])} | {fmt_t(r['pre_t'])} | {fmt(r['post_regime'])} | {fmt_t(r['post_t'])} |")
    lines += ["", "Family view (largest |t| per Alpha158 family):", ""]
    for fam, r in family.head(10).iterrows():
        lines.append(f"- {fam} ({FAMILIES.get(fam, fam)}): {r['features']} features, mean contrast {r['mean_contrast']:+.3f}, "
                     f"max |t| {r['max_abs_t']:.1f}")
    lines += ["", "Hypotheses to test are about the residual: what distinguishes a liked name that goes on to win from one "
              "that goes on to lose? Think of persistence of the move (does the pattern that put it in the top continue?), "
              "crowding and exhaustion (volume, range, gaps on the way up), the regime dependence shown above, and "
              "interactions between families. Name each factor after its hypothesis."]
    Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))
    print(table.head(args.top).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"-> {args.out}.json, {args.out}.md")


if __name__ == "__main__":
    main()
