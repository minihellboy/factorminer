"""Public API for FactorMiner's auditable evidence protocol."""

from factorminer.evaluation.evidence.inference import compute_hac_mean_test
from factorminer.evaluation.evidence.models import (
    INDUSTRY_EVIDENCE_VERSION,
    HACMeanTestResult,
    ICMetricSummary,
    IndustryEvidenceConfig,
    IndustryEvidenceReport,
    RiskResidualizationResult,
)
from factorminer.evaluation.evidence.report import evaluate_industry_evidence
from factorminer.evaluation.evidence.risk import residualize_against_risk_exposures

__all__ = [
    "INDUSTRY_EVIDENCE_VERSION",
    "HACMeanTestResult",
    "ICMetricSummary",
    "IndustryEvidenceConfig",
    "IndustryEvidenceReport",
    "RiskResidualizationResult",
    "compute_hac_mean_test",
    "evaluate_industry_evidence",
    "residualize_against_risk_exposures",
]
