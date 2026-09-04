r"""Tests for ``CalibratedHallucinationDetector``.

Targets ``spectramr.infrastructure.calibration.chd`` — ULF radiologist-
acceptance roadmap PR-21 (R2).

Plan acceptance criterion (§"Tests"):

- *"empirical FPR ≤ β + ε with ε from the DKW bound"* — covered
  by the empirical-FPR property test below.

Other contract tests:

- ``β`` / ``δ`` outside ``(0, 1)`` rejected
- ``predict`` before ``fit`` raises
- Calibration set must be non-empty
- Threshold equals empirical ``(1-β)``-quantile
- ``bound_is_tight`` reports ``True`` when slack ≤ β
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.calibration.chd import (
    CalibratedHallucinationDetector,
    dkw_slack,
)


# ---------------------------------------------------------------------------
# DKW slack
# ---------------------------------------------------------------------------


def test_dkw_slack_closed_form() -> None:
    """``dkw_slack`` matches the closed-form ``√(ln(2/δ) / (2 n))``."""
    n = 1000
    delta = 1e-3
    expected = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    assert pytest.approx(dkw_slack(n, delta)) == expected


def test_dkw_slack_decreases_with_n() -> None:
    """More calibration samples → tighter bound."""
    s_small = dkw_slack(100, 1e-3)
    s_big = dkw_slack(10000, 1e-3)
    assert s_big < s_small


def test_dkw_slack_invalid_n_rejected() -> None:
    """``n ≤ 0`` rejected."""
    with pytest.raises(ValueError, match="n must be"):
        dkw_slack(0, 1e-3)


def test_dkw_slack_invalid_delta_rejected() -> None:
    """``δ`` outside ``(0, 1)`` rejected."""
    with pytest.raises(ValueError, match="delta"):
        dkw_slack(100, 0.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_invalid_beta_rejected() -> None:
    """``β`` outside ``(0, 1)`` rejected."""
    with pytest.raises(ValueError, match="beta"):
        CalibratedHallucinationDetector(beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        CalibratedHallucinationDetector(beta=1.0)


def test_invalid_delta_rejected() -> None:
    """``δ`` outside ``(0, 1)`` rejected."""
    with pytest.raises(ValueError, match="delta"):
        CalibratedHallucinationDetector(beta=0.05, delta=1.5)


def test_default_delta_is_documented_value() -> None:
    """Default δ = 1e-3."""
    chd = CalibratedHallucinationDetector(beta=0.05)
    assert chd.delta == 1e-3


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises() -> None:
    """``predict`` before ``fit`` raises."""
    chd = CalibratedHallucinationDetector(beta=0.05)
    with pytest.raises(RuntimeError, match="not been fitted"):
        chd.predict(torch.zeros(4))


def test_threshold_before_fit_raises() -> None:
    """``threshold`` is undefined before ``fit``."""
    chd = CalibratedHallucinationDetector(beta=0.05)
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = chd.threshold


def test_fit_with_empty_calibration_raises() -> None:
    """Empty calibration set → ``ValueError``."""
    chd = CalibratedHallucinationDetector(beta=0.05)
    with pytest.raises(ValueError, match="empty"):
        chd.fit(torch.tensor([]))


def test_fit_with_zero_dim_input_raises() -> None:
    """0-D scalar input is rejected."""
    chd = CalibratedHallucinationDetector(beta=0.05)
    with pytest.raises(ValueError, match="at least 1-D"):
        chd.fit(torch.tensor(0.5))


def test_threshold_is_empirical_quantile() -> None:
    """Threshold equals the empirical ``(1-β)``-quantile."""
    torch.manual_seed(0)
    scores = torch.linspace(0.0, 1.0, 1000)
    chd = CalibratedHallucinationDetector(beta=0.05).fit(scores)
    expected = torch.quantile(scores.double(), 0.95).item()
    assert pytest.approx(chd.threshold, abs=1e-3) == expected


def test_predict_returns_bool_mask() -> None:
    """``predict`` returns a boolean tensor of the same shape."""
    chd = CalibratedHallucinationDetector(beta=0.1)
    chd.fit(torch.linspace(0.0, 1.0, 1000))
    test = torch.tensor([0.5, 0.95, 0.99])
    out = chd.predict(test)
    assert out.dtype == torch.bool
    assert out.shape == test.shape


def test_fit_returns_self_for_chaining() -> None:
    """``fit`` is fluent."""
    chd = CalibratedHallucinationDetector(beta=0.05)
    out = chd.fit(torch.linspace(0.0, 1.0, 100))
    assert out is chd


# ---------------------------------------------------------------------------
# Plan acceptance criterion: empirical FPR within DKW bound
# ---------------------------------------------------------------------------


def test_empirical_fpr_within_dkw_bound() -> None:
    """Empirical FPR on a fresh non-hallucinated test split ≤ β + DKW slack."""
    torch.manual_seed(0)
    beta = 0.1
    delta = 1e-3
    n_cal, n_test = 5000, 50000

    # Non-hallucinated scores ~ N(0, 1). All H = 0.
    cal_scores = torch.randn(n_cal)
    test_scores = torch.randn(n_test)

    chd = CalibratedHallucinationDetector(beta=beta, delta=delta).fit(cal_scores)
    fpr = chd.empirical_fpr(test_scores)
    bound = chd.fpr_upper_bound

    # The bound holds with probability 1 - δ; under iid Gaussians the
    # empirical FPR should comfortably sit under the bound.
    assert fpr <= bound, f"empirical FPR {fpr:.4f} exceeded bound {bound:.4f}"


# ---------------------------------------------------------------------------
# Bound tightness audit
# ---------------------------------------------------------------------------


def test_bound_is_tight_with_large_n() -> None:
    """Large n → DKW slack ≤ β."""
    chd = CalibratedHallucinationDetector(beta=0.1, delta=1e-3)
    chd.fit(torch.randn(100_000))
    assert chd.bound_is_tight is True


def test_bound_not_tight_with_small_n() -> None:
    """Small n → DKW slack exceeds β; audit gate condition fires."""
    chd = CalibratedHallucinationDetector(beta=0.05, delta=1e-3)
    chd.fit(torch.randn(50))  # too few samples
    assert chd.bound_is_tight is False


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


def test_n_calibration_recorded() -> None:
    """Calibration sample count is exposed for the audit gate."""
    chd = CalibratedHallucinationDetector(beta=0.05)
    chd.fit(torch.randn(1234))
    assert chd.n_calibration == 1234
