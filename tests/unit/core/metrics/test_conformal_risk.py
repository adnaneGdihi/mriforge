"""Unit tests for conformal risk control on quantitative parameter maps (qCRC).

qCRC lifts the Conformal Risk Control procedure (Angelopoulos et al.,
arXiv:2208.02814) onto the geodesic error functional of T1/T2 parameter
maps. The certified quantity is a geodesic tolerance ``lambda_hat`` such
that the expected miscoverage of the nested geodesic prediction set is at
most ``alpha``, with the finite-sample CRC correction ``(B - alpha) / n``.

The core algorithm (``conformal_risk_lambda`` / ``conformal_coverage``) is
pure tensor numerics and is tested exhaustively here; the registered
metric classes are smoke-tested against the Bloch relaxation manifold.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.quantitative.conformal_risk import (
    conformal_coverage,
    conformal_risk_lambda,
)
from mriforge.core.metrics.registry import MetricsRegistry, get_metric

# --------------------------------------------------------------------- #
# Pure CRC core
# --------------------------------------------------------------------- #


def test_core_lambda_controls_empirical_miscoverage():
    """The calibrated radius keeps empirical miscoverage at or below alpha."""
    torch.manual_seed(0)
    residuals = torch.rand(50, 64)  # n=50 scans, V=64 voxels, scores in [0,1)
    alpha = 0.1
    lam = conformal_risk_lambda(residuals, alpha=alpha)
    miscoverage = (residuals > lam).float().mean().item()
    assert miscoverage <= alpha + 1e-9


def test_core_lambda_zero_when_perfect():
    """Zero residuals (perfect reconstruction) certify a zero-radius set."""
    residuals = torch.zeros(10, 16)
    assert conformal_risk_lambda(residuals, alpha=0.1) == 0.0


def test_core_lambda_monotone_nondecreasing_in_error():
    """Scaling every residual up cannot shrink the certified radius."""
    torch.manual_seed(1)
    residuals = torch.rand(40, 32)
    lam_small = conformal_risk_lambda(residuals, alpha=0.1)
    lam_large = conformal_risk_lambda(residuals * 3.0, alpha=0.1)
    assert lam_large >= lam_small


def test_core_coverage_meets_target():
    """Realized coverage at the calibrated radius is at least 1 - alpha."""
    torch.manual_seed(2)
    residuals = torch.rand(60, 50)
    alpha = 0.15
    assert conformal_coverage(residuals, alpha=alpha) >= 1.0 - alpha


def test_core_finite_sample_correction_tightens_with_alpha_zero():
    """alpha=0 forces full coverage: the radius is the largest residual."""
    torch.manual_seed(3)
    residuals = torch.rand(20, 10)
    lam = conformal_risk_lambda(residuals, alpha=0.0)
    assert lam == pytest.approx(residuals.max().item(), abs=1e-6)


# --------------------------------------------------------------------- #
# Registered metric classes (manifold-backed)
# --------------------------------------------------------------------- #


def test_metrics_registered():
    assert MetricsRegistry.is_registered("conformal_risk_control")
    assert MetricsRegistry.is_registered("qmap_conformal_coverage")


def test_metric_contract_directions():
    risk = get_metric("conformal_risk_control")
    cov = get_metric("qmap_conformal_coverage")
    assert risk.name == "conformal_risk_control"
    assert risk.higher_is_better is False
    assert cov.higher_is_better is True


def _qmaps(b: int = 8, hw: int = 2) -> torch.Tensor:
    """A small batch of physically plausible (M0, T1, T2) maps [B, 3, H, W]."""
    torch.manual_seed(0)
    m0 = torch.rand(b, 1, hw, hw) * 0.5 + 0.8
    t1 = torch.rand(b, 1, hw, hw) * 400.0 + 800.0
    t2 = torch.rand(b, 1, hw, hw) * 40.0 + 60.0
    return torch.cat([m0, t1, t2], dim=1)


def test_perfect_prediction_certifies_zero_radius():
    """pred == target ⇒ all geodesic residuals vanish ⇒ lambda_hat = 0."""
    risk = get_metric("conformal_risk_control", alpha=0.1)
    maps = _qmaps()
    assert risk(maps, maps) == pytest.approx(0.0, abs=1e-6)


def test_coverage_certificate_meets_target_on_real_maps():
    """The manifold-backed coverage metric honours the 1 - alpha guarantee."""
    cov = get_metric("qmap_conformal_coverage", alpha=0.2)
    pred = _qmaps()
    target = pred + torch.randn_like(pred) * torch.tensor([0.05, 20.0, 5.0]).reshape(1, 3, 1, 1)
    value = cov(pred, target)
    assert isinstance(value, float)
    assert value >= 1.0 - 0.2 - 1e-9


def test_shape_mismatch_raises():
    risk = get_metric("conformal_risk_control")
    with pytest.raises(ValueError):
        risk(torch.rand(2, 3, 4, 4), torch.rand(2, 3, 5, 5))
