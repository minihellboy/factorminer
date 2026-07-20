"""Tests for factor crowding diagnostics (evaluation/crowding.py).

Covers consensus-factor novelty screen, Lou-Polk CoMetric, composite
CrowdingScore (composing salvaged hyperbolic taxonomy from decay.py),
fail-closed Ken French parsing, and discrimination proofs.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.evaluation.crowding import (
    DEFAULT_FIXTURE_PATH,
    ConsensusFactorPanel,
    CrowdingConfig,
    CrowdingScore,
    _factor_regression_residuals_window,
    _parse_ken_french_text,
    build_crowding_report,
    compute_cometric,
    consensus_overlap_score,
    long_short_returns,
    score_factor_crowding,
)
from factorminer.evaluation.decay import classify_crowding_decay_risk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ff_panel() -> ConsensusFactorPanel:
    return ConsensusFactorPanel.from_fixture()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


# ---------------------------------------------------------------------------
# ConsensusFactorPanel loader
# ---------------------------------------------------------------------------


def test_fixture_panel_loads_and_has_core_factors(ff_panel: ConsensusFactorPanel):
    assert not ff_panel.empty
    names_upper = {n.upper() for n in ff_panel.factor_names}
    assert any("MKT" in n for n in names_upper)
    assert any(n == "SMB" for n in names_upper)
    assert any(n == "HML" for n in names_upper)
    # Scaled percent → decimal: fixture values were O(1) percent points.
    mkt = ff_panel.get("Mkt-RF")
    assert mkt is not None
    assert float(np.max(np.abs(mkt))) < 1.0  # decimals, not raw percent


def test_parse_malformed_csv_fail_closed():
    """Truncated / garbage CSV must never crash or invent scores."""
    assert _parse_ken_french_text("") == {}
    assert _parse_ken_french_text("not a ken french file\nfoo,bar\n") == {}
    assert _parse_ken_french_text(",Mkt-RF,SMB\n20200102,0.1\n") == {}  # ragged
    # Header only, no rows.
    assert _parse_ken_french_text(",Mkt-RF,SMB,HML,RF\n") == {}


def test_from_bytes_truncated_zip_fail_closed():
    panel = ConsensusFactorPanel.from_bytes(b"PK\x03\x04truncated")
    assert panel.empty
    # Overlap against empty panel is unavailable (not a fake low score label).
    result = consensus_overlap_score(np.array([0.1, -0.2, 0.05, 0.0, 0.03]), panel)
    assert result.available is False
    assert result.label == "unavailable"


def test_fetch_refuses_non_https():
    panel = ConsensusFactorPanel.fetch(url="http://example.com/factors.csv")
    assert panel.empty
    assert "non-https" in panel.source or panel.source.startswith("refused")


def test_default_fixture_path_exists():
    assert DEFAULT_FIXTURE_PATH.is_file()


# ---------------------------------------------------------------------------
# Consensus overlap discrimination
# ---------------------------------------------------------------------------


def test_consensus_overlap_high_for_mimic_low_for_noise(ff_panel: ConsensusFactorPanel):
    """A synthetic factor that mimics SMB scores high; orthogonal noise scores low."""
    smb = ff_panel.get("SMB")
    assert smb is not None and smb.size >= 20

    # Near-perfect mimic of SMB (+ tiny proportional noise).
    rng = np.random.default_rng(1)
    scale = float(np.std(smb)) + 1e-12
    mimic = smb + 0.01 * scale * rng.normal(size=smb.shape)
    high = consensus_overlap_score(mimic, ff_panel)
    assert high.available
    assert high.max_abs_rho > 0.85
    assert high.best_factor.upper() == "SMB"
    assert high.label == "high_consensus_overlap"

    # Orthogonal-ish noise (different seed, demeaned).
    noise = rng.normal(size=smb.shape)
    noise = noise - noise.mean()
    low = consensus_overlap_score(noise, ff_panel)
    assert low.available
    assert low.max_abs_rho < 0.40
    assert low.label == "low_consensus_overlap"
    assert high.max_abs_rho > low.max_abs_rho + 0.4


# ---------------------------------------------------------------------------
# Lou-Polk CoMetric
# ---------------------------------------------------------------------------


def _make_crowded_panel(rng: np.random.Generator, m: int = 40, t: int = 120):
    """Returns + signals where top/bottom legs share a common residual factor."""
    market = rng.normal(0.0, 0.01, size=t)
    # Idiosyncratic noise.
    idio = rng.normal(0.0, 0.01, size=(m, t))
    # Shared residual factor that will load heavily on a subset.
    common = rng.normal(0.0, 0.02, size=t)
    loadings = np.zeros(m)
    loadings[: m // 5] = 1.0  # long-leg assets share common factor
    loadings[-m // 5 :] = 1.0  # short-leg assets also share (crowded both legs)
    returns = market[None, :] + idio + loadings[:, None] * common[None, :]

    # Signal ranks assets so top/bottom match the common-loading groups.
    base_signal = loadings[:, None] + 0.05 * rng.normal(size=(m, t))
    # Add mild time variation so legs aren't static-identical every bar.
    signals = base_signal + 0.01 * rng.normal(size=(m, t))
    return signals, returns


def _make_uncrowded_panel(rng: np.random.Generator, m: int = 40, t: int = 120):
    """Independent assets — within-leg residual corr should be near 0."""
    returns = rng.normal(0.0, 0.01, size=(m, t))
    signals = rng.normal(0.0, 1.0, size=(m, t))
    return signals, returns


def test_cometric_higher_for_crowded_legs(rng: np.random.Generator):
    sig_c, ret_c = _make_crowded_panel(rng)
    sig_u, ret_u = _make_uncrowded_panel(rng)

    crowded = compute_cometric(sig_c, ret_c, window=40, leg_fraction=0.2)
    uncrowded = compute_cometric(sig_u, ret_u, window=40, leg_fraction=0.2)

    assert crowded.available
    assert uncrowded.available
    # Crowded legs share a residual factor → higher CoMOM.
    assert crowded.comom > uncrowded.comom + 0.05
    assert crowded.comom > 0.15


def test_cometric_shape_mismatch_unavailable():
    result = compute_cometric(np.zeros((5, 10)), np.zeros((6, 10)))
    assert result.available is False


def test_cometric_factor_regression_residual_orthogonal_to_market(
    rng: np.random.Generator,
):
    """Decisive proof of the Lou-Polk gap this mode closes.

    OLS residuals are mathematically guaranteed orthogonal to the
    regressors used to fit them. Cross-sectional demeaning has no such
    guarantee: it only removes each bar's average level, not each asset's
    OWN market-factor exposure, so heterogeneous-beta assets retain real
    correlation with the market factor after demeaning.
    """
    m, window = 30, 60
    mkt = rng.normal(0.0, 1.0, window)
    smb = rng.normal(0.0, 0.5, window)
    hml = rng.normal(0.0, 0.5, window)
    true_beta = rng.uniform(0.5, 2.5, size=m)
    idio = rng.normal(0.0, 0.3, size=(m, window))
    returns = true_beta[:, None] * mkt[None, :] + idio
    factor_window = np.column_stack([mkt, smb, hml])

    fr_resid = _factor_regression_residuals_window(returns, factor_window)
    from factorminer.evaluation.crowding import _cross_sectional_residuals
    cs_resid = _cross_sectional_residuals(returns)

    fr_corr = [abs(float(np.corrcoef(fr_resid[i], mkt)[0, 1])) for i in range(m)]
    cs_corr = [abs(float(np.corrcoef(cs_resid[i], mkt)[0, 1])) for i in range(m)]

    assert max(fr_corr) < 1e-8, "OLS residuals must be orthogonal to the regressor"
    assert np.mean(cs_corr) > 0.05, (
        "cross-sectional residuals should retain real market exposure "
        "given heterogeneous betas -- otherwise this fixture isn't testing anything"
    )


def test_cometric_factor_regression_mode_wired_end_to_end(
    ff_panel: ConsensusFactorPanel, rng: np.random.Generator
):
    sig, ret = _make_uncrowded_panel(rng, m=20, t=80)
    result = compute_cometric(
        sig, ret, window=40, residual_mode="factor_regression", consensus_panel=ff_panel
    )
    assert result.available
    assert result.residual_mode == "factor_regression"
    assert np.isfinite(result.comom)


def test_cometric_factor_regression_falls_back_without_panel(
    rng: np.random.Generator,
):
    """No consensus_panel supplied -> fail-closed fallback to cross_sectional,
    never a crash and never a silently-fabricated factor fit."""
    sig, ret = _make_uncrowded_panel(rng, m=20, t=80)
    result = compute_cometric(
        sig, ret, window=40, residual_mode="factor_regression", consensus_panel=None
    )
    assert result.available
    assert result.residual_mode == "cross_sectional"


def test_cometric_factor_regression_falls_back_on_empty_panel(
    rng: np.random.Generator,
):
    sig, ret = _make_uncrowded_panel(rng, m=20, t=80)
    empty_panel = ConsensusFactorPanel(series={}, source="empty")
    result = compute_cometric(
        sig, ret, window=40, residual_mode="factor_regression", consensus_panel=empty_panel
    )
    assert result.residual_mode == "cross_sectional"


def test_cometric_invalid_residual_mode_raises(rng: np.random.Generator):
    sig, ret = _make_uncrowded_panel(rng, m=20, t=80)
    with pytest.raises(ValueError, match="residual_mode"):
        compute_cometric(sig, ret, window=40, residual_mode="bogus")


def test_long_short_returns_shape(rng: np.random.Generator):
    sig, ret = _make_uncrowded_panel(rng, m=20, t=50)
    ls = long_short_returns(sig, ret)
    assert ls.shape == (50,)
    assert np.isfinite(ls).sum() > 10


# ---------------------------------------------------------------------------
# Composite CrowdingScore (+ salvaged hyperbolic taxonomy)
# ---------------------------------------------------------------------------


def test_score_factor_crowding_composes_decay_taxonomy(
    ff_panel: ConsensusFactorPanel, rng: np.random.Generator
):
    sig, ret = _make_crowded_panel(rng)
    # Mechanical family formula + fast-decaying IC series.
    ic_series = [0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.008]
    score = score_factor_crowding(
        signals=sig,
        returns=ret,
        panel=ff_panel,
        ic_by_iteration=ic_series,
        formula="TsRank($close, 20)",  # Trend/Momentum-ish
        factor_id="mom1",
        config=CrowdingConfig(cometric_window=40),
    )
    assert isinstance(score, CrowdingScore)
    assert score.decay_taxonomy is not None
    assert score.decay_taxonomy.risk_label  # salvaged classify_crowding_decay_risk
    assert 0.0 <= score.novelty_modulation <= 1.0
    assert score.composite_label
    d = score.to_dict()
    assert "consensus" in d and "cometric" in d and "decay_taxonomy" in d


def test_discrimination_mimic_vs_orthogonal_on_both_axes(
    ff_panel: ConsensusFactorPanel, rng: np.random.Generator
):
    """End-to-end: consensus-mimic scores high on overlap; noise scores low.

    CoMetric is exercised on crowded vs uncrowded synthetic panels.
    """
    smb = ff_panel.get("SMB")
    assert smb is not None

    # Build a (M,T) panel whose L/S series tracks SMB closely by construction:
    # assign assets random exposures; L/S return ≈ SMB.
    m, t = 30, int(smb.shape[0])
    # Use first t bars of smb.
    factor = smb[:t]
    loadings = rng.normal(size=m)
    returns = loadings[:, None] * factor[None, :] + 0.001 * rng.normal(size=(m, t))
    signals = np.broadcast_to(loadings[:, None], (m, t)).astype(np.float64)
    # Add tiny jitter so ranking is stable but not identical every bar.
    signals = signals + 1e-6 * rng.normal(size=(m, t))

    mimic_score = score_factor_crowding(
        signals=signals,
        returns=returns,
        panel=ff_panel,
        factor_id="mimic",
        config=CrowdingConfig(cometric_window=40),
    )

    # Orthogonal noise factor.
    noise_sig = rng.normal(size=(m, t))
    noise_ret = rng.normal(0.0, 0.01, size=(m, t))
    noise_score = score_factor_crowding(
        signals=noise_sig,
        returns=noise_ret,
        panel=ff_panel,
        factor_id="noise",
        config=CrowdingConfig(cometric_window=40),
    )

    assert mimic_score.consensus.available
    assert noise_score.consensus.available
    assert mimic_score.consensus.max_abs_rho > noise_score.consensus.max_abs_rho + 0.3

    # Crowded vs uncrowded CoMetric discrimination (separate synthetic panels).
    sig_c, ret_c = _make_crowded_panel(rng)
    sig_u, ret_u = _make_uncrowded_panel(rng)
    c_score = score_factor_crowding(
        signals=sig_c,
        returns=ret_c,
        panel=ff_panel,
        factor_id="crowded",
        config=CrowdingConfig(cometric_window=40),
    )
    u_score = score_factor_crowding(
        signals=sig_u,
        returns=ret_u,
        panel=ff_panel,
        factor_id="uncrowded",
        config=CrowdingConfig(cometric_window=40),
    )
    assert c_score.cometric.available and u_score.cometric.available
    assert c_score.cometric.comom > u_score.cometric.comom


def test_build_crowding_report(ff_panel: ConsensusFactorPanel, rng: np.random.Generator):
    sig, ret = _make_uncrowded_panel(rng, m=20, t=80)
    rows = build_crowding_report(
        [
            {
                "factor_id": "a",
                "formula": "Rank($volume)",
                "signals": sig,
                "returns": ret,
                "ic_by_iteration": [0.05, 0.05, 0.04, 0.05, 0.05],
            }
        ],
        panel=ff_panel,
        config=CrowdingConfig(cometric_window=30),
    )
    assert len(rows) == 1
    assert rows[0]["factor_id"] == "a"


def test_crowding_novelty_modulation_softens_geometry_novelty(
    ff_panel: ConsensusFactorPanel,
):
    """Soft novelty modulation extends LibraryGeometry.novelty_score."""
    from factorminer.architecture.geometry import LibraryGeometry
    from factorminer.core.factor_library import FactorLibrary

    library = FactorLibrary(correlation_threshold=0.7, ic_threshold=0.01)
    # Empty library → base novelty 1.0; modulation 0.5 → 0.5.
    geo = LibraryGeometry(library)
    cand = np.random.default_rng(0).normal(size=(5, 20))
    g0 = geo.candidate_geometry(cand)
    assert g0.novelty_score == pytest.approx(1.0)
    g1 = geo.candidate_geometry(cand, crowding_novelty_modulation=0.5)
    assert g1.novelty_score == pytest.approx(0.5)


def test_salvaged_hyperbolic_still_importable_from_decay():
    """Credit check: crowding composes decay.py, does not reimplement it."""
    tax = classify_crowding_decay_risk(
        [0.1, 0.08, 0.05, 0.03, 0.02, 0.01],
        formula="TsRank($close, 10)",
    )
    assert tax.hyperbolic.n_obs >= 4
    assert tax.risk_label
