r"""Tests for K-RCPS entrywise conformal risk control (SOTA plan Phase C).

K-RCPS (Teneggi et al. 2023) controls the *entrywise* (per-pixel) risk of a
reconstruction prediction set, rather than a single scalar interval. This
implementation calibrates the smallest scale ``β`` over a per-entry width profile
``w`` so the marginal CRC bound ``(∑ ℓ_i(β) + B)/(n+1) ≤ α`` holds — reusing the
scalar CRC machinery. Corner case: uniform widths ⇒ scalar CRC.
"""

from __future__ import annotations

import pytest

from mriforge.core.metrics.meta_evaluation.conformal import select_conformal_threshold
from mriforge.core.metrics.meta_evaluation.krcps import KRCPSResult, krcps_calibrate


def _calib(n: int = 20, d: int = 4, scale: float = 1.0):
    """Deterministic per-entry residuals: entry j has |error| ≈ (j+1)·0.1·scale."""
    return [[(j + 1) * 0.1 * scale * (1.0 + 0.01 * i) for j in range(d)] for i in range(n)]


def test_uniform_widths_reduce_to_scalar_crc():
    """Corner case: w_d ≡ 1 ⇒ K-RCPS β equals the scalar conformal threshold."""
    residuals = _calib()
    d = len(residuals[0])
    res = krcps_calibrate(residuals, widths=[1.0] * d, alpha=0.2, n_grid=200)
    # Reproduce the scalar selection over the same β grid + miscoverage loss.
    ratios = sorted({abs(r) for row in residuals for r in row})
    beta_max = ratios[-1]
    betas = [beta_max * k / 199 for k in range(200)]
    loss_at = [
        [sum(abs(r) > b for r in row) / d for row in residuals] for b in betas
    ]
    scalar = select_conformal_threshold(betas, loss_at, alpha=0.2)
    assert abs(res.beta - scalar.threshold) < 1e-9
    assert res.n_groups == 1


def test_entrywise_widths_give_varying_intervals():
    residuals = _calib()
    widths = [0.5, 1.0, 2.0, 4.0]
    res = krcps_calibrate(residuals, widths=widths, alpha=0.3)
    iw = res.interval_widths
    assert len(iw) == 4
    # interval width tracks the nominal width profile (entrywise, not constant).
    assert iw[0] < iw[-1]
    assert res.n_groups == 4


def test_controlled_when_alpha_feasible():
    residuals = _calib()
    res = krcps_calibrate(residuals, widths=[1.0, 1.0, 1.0, 1.0], alpha=0.5)
    assert res.controlled
    # The finite-sample risk bound is the CRC (n+1)α/n.
    assert res.risk_bound == pytest.approx((len(residuals) + 1) * 0.5 / len(residuals))


def test_looser_alpha_gives_tighter_intervals():
    """Monotonicity: larger α (more risk allowed) ⇒ smaller calibrated β."""
    residuals = _calib()
    widths = [1.0, 1.0, 1.0, 1.0]
    beta_tight = krcps_calibrate(residuals, widths=widths, alpha=0.1).beta
    beta_loose = krcps_calibrate(residuals, widths=widths, alpha=0.5).beta
    assert beta_loose <= beta_tight


def test_result_type_and_interval_scaling():
    residuals = _calib()
    res = krcps_calibrate(residuals, widths=[2.0, 2.0, 2.0, 2.0], alpha=0.3)
    assert isinstance(res, KRCPSResult)
    assert all(abs(w - res.beta * 2.0) < 1e-9 for w in res.interval_widths)


def test_validation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        krcps_calibrate([], widths=[1.0], alpha=0.2)
    with pytest.raises(ValueError):
        krcps_calibrate([[0.1, 0.2]], widths=[1.0], alpha=0.2)  # width/entry mismatch
    with pytest.raises(ValueError):
        krcps_calibrate([[0.1, 0.2]], widths=[1.0, 0.0], alpha=0.2)  # non-positive width
    with pytest.raises(ValueError):
        krcps_calibrate([[0.1, 0.2]], widths=[1.0, 1.0], alpha=1.5)  # alpha out of range
