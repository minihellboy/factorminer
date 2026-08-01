"""Canonical global-trial ledger and inference-family tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from factorminer.architecture.trial_ledger import (
    TRIAL_LEDGER_FILENAME,
    TrialLedger,
    canonical_formula_identity,
)


def test_canonical_identity_deduplicates_algebraic_aliases():
    left = canonical_formula_identity("Add($close, $open)")
    right = canonical_formula_identity("Add($open, $close)")
    assert left == right
    assert len(left[1]) == 64


def test_ledger_builds_complete_deduplicated_family_and_reloads(tmp_path):
    ledger = TrialLedger(tmp_path, campaign_id="campaign-1")
    common = {
        "dataset_id": "dataset-sha",
        "target_name": "forward-1d",
        "iteration": 1,
        "stage": "full_validation",
    }
    ledger.record_data_contact(
        factor_name="sum-a",
        formula="Add($close, $open)",
        status="evaluated",
        ic_series=np.array([0.1, 0.2, np.nan, 0.0]),
        **common,
    )
    ledger.record_data_contact(
        factor_name="sum-alias",
        formula="Add($open, $close)",
        status="evaluated",
        ic_series=np.array([0.1, 0.2, np.nan, 0.0]),
        **common,
    )
    ledger.record_data_contact(
        factor_name="failed-signal",
        formula="Mean($volume, 5)",
        status="screened_or_failed",
        **common,
    )

    family = ledger.build_inference_family(
        dataset_id="dataset-sha",
        target_name="forward-1d",
        periods=4,
    )
    assert len(ledger.observations) == 3
    assert family.raw_trial_count == 2
    assert len(family.ic_series) == 2
    assert family.metadata["missing_series_filled_with_zero"] == 1
    assert any(values == (0.0, 0.0, 0.0, 0.0) for values in family.ic_series.values())
    assert 1.0 <= family.dsr_trial_count <= 2.0

    reloaded = TrialLedger.load(tmp_path, campaign_id="campaign-1")
    assert reloaded.observations == ledger.observations
    assert reloaded.build_inference_family(
        dataset_id="dataset-sha",
        target_name="forward-1d",
        periods=4,
    ).raw_trial_count == 2


def test_ledger_rejects_tampering(tmp_path):
    ledger = TrialLedger(tmp_path, campaign_id="campaign-1")
    ledger.record_data_contact(
        factor_name="original",
        formula="$close",
        dataset_id="dataset-sha",
        target_name="paper",
        iteration=0,
        stage="screen",
        status="evaluated",
        ic_series=[0.1, 0.2],
    )
    path = tmp_path / TRIAL_LEDGER_FILENAME
    payload = json.loads(path.read_text().strip())
    payload["factor_name"] = "tampered"
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="record hash mismatch"):
        TrialLedger.load(path)


def test_batch_recording_counts_data_contacts_but_not_parse_failures(tmp_path):
    ledger = TrialLedger(tmp_path, campaign_id="campaign-1")
    results = [
        SimpleNamespace(
            factor_name="parsed",
            formula="$close",
            parse_ok=True,
            target_stats={"paper": {"rank_ic_series": np.array([0.1, 0.2, 0.3])}},
            stage_passed=1,
            admitted=False,
            rejection_reason="",
        ),
        SimpleNamespace(
            factor_name="bad",
            formula="NotAFormula(",
            parse_ok=False,
            target_stats={},
            stage_passed=0,
            admitted=False,
            rejection_reason="Parse failure",
        ),
    ]
    recorded = ledger.record_batch_results(
        iteration=2,
        results=results,
        dataset_id="dataset-sha",
        target_name="paper",
    )
    assert len(recorded) == 1
    assert recorded[0].factor_name == "parsed"


def test_inference_family_rejects_mixed_period_lengths():
    ledger = TrialLedger(campaign_id="campaign-1")
    for formula, series in (("$close", [0.1, 0.2]), ("$open", [0.1, 0.2, 0.3])):
        ledger.record_data_contact(
            factor_name=formula,
            formula=formula,
            dataset_id="dataset-sha",
            target_name="paper",
            iteration=0,
            stage="screen",
            status="evaluated",
            ic_series=series,
        )
    with pytest.raises(ValueError, match="common IC-series length"):
        ledger.build_inference_family(dataset_id="dataset-sha", target_name="paper")
