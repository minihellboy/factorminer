"""Replay a campaign's candidate stream under other admission rules, without the LLM.

Run in the FactorMiner environment::

    uv run python scripts/qlib_cn/replay_admission.py \\
        -c output/qlib_cn/configs/csi300_classic.yaml --data panels/csi300.csv \\
        --campaign output/qlib_cn/campaigns/classic_marginal_c1 \\
        --rule marginal='{}' --rule top='{"min_top_t": 2.0, "min_top_lift_t": 0.0}' \\
        --out output/qlib_cn/replay/classic_marginal_c1 --backend gpu

The trial ledger records every formula the LLM proposed, in order. The replay
walks the ledger iteration by iteration, as the mining loop did: each
iteration's new formulas are judged against the reference basis plus the
library admitted so far, passing candidates are taken greedily by partial-IC
t, and each pick joins the basis. A rule is ``MarginalConfig`` gate overrides
on top of the campaign's own ``admission`` settings, so ``'{}'`` reproduces the
campaign's rule. The candidate stream itself was generated under the
campaign's original feedback; only the admission decision changes.

Outputs, per rule, ``<rule>/library.json`` (``evaluate_library.py --formulas``),
and ``candidates.csv``: each candidate's statistics against the reference
basis alone on the discovery panel (``train_*``) and on the validation split
(``valid_*``, oriented by the train sign), for calibrating gates without
touching the test split. When ``candidates.csv`` already exists in ``--out``,
it is reused: only the prefiltered candidates' discovery signals are
recomputed, and new rules are added to the existing ``replay.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from factorminer.application.marginal_admission import zone_panel
from factorminer.application.research_actions import discovery_dataset
from factorminer.core.parser import try_parse
from factorminer.core.reference_catalogs import reference_catalog
from factorminer.data.loader import load_market_data
from factorminer.evaluation.marginal import MarginalBasis, MarginalConfig, passes_marginal_admission
from factorminer.evaluation.runtime import compute_batch_signals, load_runtime_dataset
from factorminer.utils.config import load_config

STATS = ("partial_ic", "partial_ic_t", "residual_share", "top_excess_bps", "top_partial_ic",
         "top_partial_ic_t", "top_lift_bps", "top_lift_t")


def _stream(campaign: Path) -> list[tuple[int, str, str]]:
    """(iteration, name, formula) of each distinct canonical formula, first occurrence."""
    seen, out = set(), []
    with open(campaign / "global_trial_ledger.jsonl") as handle:
        rows = sorted((json.loads(line) for line in handle), key=lambda r: r["sequence"])
    for row in rows:
        key = row["canonical_formula_hash"]
        if key not in seen:
            seen.add(key)
            out.append((int(row["iteration"]), row["factor_name"], row["formula"]))
    return out


def _signals(trees, dataset, backend: str) -> list[np.ndarray | None]:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        outcomes = compute_batch_signals(trees, dataset.data_dict, dataset.returns.shape,
                                         backend=backend)
        return [None if error is not None or panel is None
                else np.asarray(panel, dtype=np.float64) for _, panel, error in outcomes]


def _basis(dataset, returns: np.ndarray, config: MarginalConfig, backend: str,
           admission=None) -> MarginalBasis:
    trees = [tree for _, formula in reference_catalog("alpha158")
             if (tree := try_parse(formula)) is not None]
    panels = (panel for panel in _signals(trees, dataset, backend) if panel is not None)
    zone = zone_panel(admission, dataset.data_dict) if admission is not None else None
    return MarginalBasis(returns, panels, config, zone_score=zone)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--rule", action="append", required=True, help="name=JSON gate overrides")
    parser.add_argument("--prefilter-t", type=float, default=2.0,
                        help="skip candidates whose partial-IC t against the reference alone is below")
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="recompute candidates.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_config = MarginalConfig(**cfg.admission.marginal_settings())
    rules = {name: replace(base_config, **json.loads(spec))
             for name, spec in (r.split("=", 1) for r in args.rule)}
    full = load_runtime_dataset(load_market_data(args.data), cfg)
    discovery = discovery_dataset(full)
    stream = _stream(args.campaign)
    trees = [try_parse(formula) for _, _, formula in stream]

    # Statistics against the reference basis alone: discovery panel and validation split.
    train_basis = _basis(discovery, discovery.returns, base_config, args.backend, cfg.admission)
    cached = args.out / "candidates.csv"
    if cached.exists() and not args.force:
        kept = _cached_signals(cached, stream, trees, discovery, args)
        rows = None
    else:
        rows, kept = _score_candidates(full, discovery, stream, trees, train_basis, base_config, args, cfg.admission)
    args.out.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        pd.DataFrame(rows).to_csv(cached, index=False)

    # Greedy replay per rule, iteration by iteration.
    iterations = sorted({it for it, _, _ in stream})
    report = {"campaign": str(args.campaign), "candidates": len(stream),
              "prefiltered": len(kept), "rules": {}}
    if (args.out / "replay.json").exists():
        report["rules"] = json.loads((args.out / "replay.json").read_text()).get("rules", {})
    for rule_name, rule in rules.items():
        train_basis.sync({})
        library: list[int] = []
        for iteration in iterations:
            batch = []
            for i in (i for i, (it, _, _) in enumerate(stream) if it == iteration and i in kept):
                stats = train_basis.evaluate(kept[i], rule)  # the rule's statistics options too
                if passes_marginal_admission(stats, rule)[0]:
                    batch.append((stats.partial_ic_t, i))
            for position, (_, i) in enumerate(sorted(batch, reverse=True)):
                if position and not passes_marginal_admission(train_basis.evaluate(kept[i], rule), rule)[0]:
                    continue
                train_basis.add(i, kept[i])
                library.append(i)
        out = args.out / rule_name
        out.mkdir(exist_ok=True)
        (out / "library.json").write_text(json.dumps({
            "rule": {k: v for k, v in asdict(rule).items()},
            "factors": [{"name": f"{stream[i][1]}_{i}", "formula": stream[i][2],
                         "iteration": stream[i][0]} for i in library],
        }, indent=2) + "\n")
        report["rules"][rule_name] = {"admitted": len(library),
                                      "gates": json.loads(dict(r.split("=", 1) for r in args.rule)[rule_name])}
        print(f"{args.campaign.name} {rule_name}: {len(library)} admitted", flush=True)
    (args.out / "replay.json").write_text(json.dumps(report, indent=2) + "\n")


def _cached_signals(cached: Path, stream, trees, discovery, args) -> dict[int, np.ndarray]:
    """Discovery signals of the candidates ``candidates.csv`` says pass the prefilter."""
    table = pd.read_csv(cached)
    if len(table) != len(stream) or (table["formula"].to_numpy() != [f for _, _, f in stream]).any():
        raise SystemExit(f"{cached} does not match the campaign ledger; rerun with --force")
    wanted = [int(i) for i in table.index[table["train_partial_ic_t"] >= args.prefilter_t]
              if trees[i] is not None]
    kept = {}
    chunk = 64
    for start in range(0, len(wanted), chunk):
        members = wanted[start:start + chunk]
        for i, signal in zip(members, _signals([trees[i] for i in members], discovery, args.backend),
                             strict=True):
            if signal is not None:
                kept[i] = signal
    print(f"{args.campaign.name}: reused {cached}, {len(kept)} prefiltered signals", flush=True)
    return kept


def _score_candidates(full, discovery, stream, trees, train_basis, base_config, args, admission=None):
    valid_returns = np.full_like(full.returns, np.nan)
    valid_idx = full.get_split("validation").indices
    valid_returns[:, valid_idx] = full.returns[:, valid_idx]
    valid_basis = _basis(full, valid_returns, base_config, args.backend, admission)
    rows, kept = [], {}
    chunk = 64  # full-panel signals of a chunk at a time
    for start in range(0, len(stream), chunk):
        members = [i for i in range(start, min(start + chunk, len(stream))) if trees[i] is not None]
        train_signals = _signals([trees[i] for i in members], discovery, args.backend)
        valid_signals = _signals([trees[i] for i in members], full, args.backend)
        signals = {i: (t, v) for i, t, v in zip(members, train_signals, valid_signals, strict=True)}
        for i in range(start, min(start + chunk, len(stream))):
            iteration, name, formula = stream[i]
            row = {"index": i, "iteration": iteration, "name": name, "formula": formula}
            train_signal, valid_signal = signals.get(i, (None, None))
            if train_signal is not None and valid_signal is not None:
                train = train_basis.evaluate(train_signal)
                valid = valid_basis.evaluate(valid_signal)
                row.update({f"train_{k}": getattr(train, k) for k in STATS})
                # Validation statistics in the train direction; top statistics refit on
                # validation are only comparable when the sign held.
                held = train.direction == valid.direction
                row.update({"valid_sign_held": held,
                            "valid_partial_ic": train.direction * valid.partial_ic,
                            "valid_partial_ic_t": (1 if held else -1) * valid.partial_ic_t,
                            **{f"valid_{k}": getattr(valid, k) for k in
                               ("top_partial_ic", "top_partial_ic_t", "top_lift_bps",
                                "top_lift_t", "top_excess_bps")}})
                if train.partial_ic_t >= args.prefilter_t:
                    kept[i] = train_signal
            rows.append(row)
        print(f"{args.campaign.name}: scored {min(start + chunk, len(stream))}/{len(stream)}",
              flush=True)
    return rows, kept


if __name__ == "__main__":
    main()
