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

from spectramr.core.metrics.quantitative.conformal_risk import (
    conformal_coverage,
    conformal_risk_lambda,
)
from spectramr.core.metrics.registry import MetricsRegistry, get_metric

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


# --------------------------------------------------------------------- #
# Image-domain empirical coverage (the validation ensemble, 2026-09)
#
# ``coverage_fraction`` is the one owner of "fraction of residuals inside a
# radius"; the CRC coverage above and the registered ``empirical_coverage``
# both go through it, and the first test below is what keeps that true.
# --------------------------------------------------------------------- #


def test_coverage_fraction_is_the_arithmetic_conformal_coverage_uses():
    """One owner: the CRC coverage equals ``coverage_fraction`` at the calibrated radius."""
    from spectramr.core.metrics.quantitative.conformal_risk import coverage_fraction

    torch.manual_seed(4)
    residuals = torch.rand(30, 40)
    alpha = 0.1
    lam = conformal_risk_lambda(residuals, alpha=alpha)
    assert conformal_coverage(residuals, alpha=alpha) == pytest.approx(
        float(coverage_fraction(residuals.reshape(-1), lam))
    )


def test_coverage_fraction_with_a_per_pixel_radius_counts_the_boundary():
    from spectramr.core.metrics.quantitative.conformal_risk import coverage_fraction

    residuals = torch.tensor([0.0, 1.0, 2.0, 3.0])
    radius = torch.tensor([0.5, 0.5, 2.0, 2.5])
    # inside: 0.0 <= 0.5, 2.0 <= 2.0 (boundary counts); outside: 1.0, 3.0
    assert float(coverage_fraction(residuals, radius)) == pytest.approx(0.5)


def test_coverage_fraction_refuses_an_empty_tensor():
    from spectramr.core.metrics.quantitative.conformal_risk import coverage_fraction

    with pytest.raises(ValueError, match="empty"):
        coverage_fraction(torch.zeros(0), 1.0)


def test_empirical_coverage_is_registered_and_declares_its_need():
    metric = get_metric("empirical_coverage")
    assert MetricsRegistry.is_registered("empirical_coverage")
    assert metric.name == "empirical_coverage"
    assert metric.higher_is_better is True
    assert MetricsRegistry.needs("empirical_coverage") == ("ensemble_std",)
    assert MetricsRegistry.needs_context("empirical_coverage") is True


def _coverage_fixture():
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[0.5, 1.5], [2.5, 0.0]]]])
    std = torch.ones(1, 1, 2, 2)
    return pred, target, std


def test_empirical_coverage_counts_target_pixels_inside_k_std():
    from spectramr.core.metrics.context import MetricContext
    from spectramr.core.metrics.quantitative.conformal_risk import EmpiricalCoverageMetric

    pred, target, std = _coverage_fixture()
    ctx = MetricContext(ensemble_std=std)
    # residuals 0.5, 1.5, 2.5, 0.0 against k * 1
    assert EmpiricalCoverageMetric(k=1.0)(pred, target, context=ctx) == pytest.approx(0.5)
    assert EmpiricalCoverageMetric(k=2.0)(pred, target, context=ctx) == pytest.approx(0.75)
    assert EmpiricalCoverageMetric(k=3.0)(pred, target, context=ctx) == pytest.approx(1.0)


def test_empirical_coverage_accepts_the_loose_kwarg_form():
    """The legacy engine passes context fields flat; ``resolve_context`` collects them."""
    from spectramr.core.metrics.quantitative.conformal_risk import EmpiricalCoverageMetric

    pred, target, std = _coverage_fixture()
    assert EmpiricalCoverageMetric(k=2.0)(pred, target, ensemble_std=std) == pytest.approx(0.75)


def test_empirical_coverage_without_the_std_is_declared_not_applicable():
    """Not a ValueError (the computer re-raises those as crashes) and not a number."""
    from spectramr.core.metrics.outcome import MetricNotApplicableError, NotApplicableReason
    from spectramr.core.metrics.quantitative.conformal_risk import EmpiricalCoverageMetric

    pred, target, _ = _coverage_fixture()
    with pytest.raises(MetricNotApplicableError) as excinfo:
        EmpiricalCoverageMetric()(pred, target)
    assert excinfo.value.reason is NotApplicableReason.MISSING_MEASUREMENT_CONTEXT


def test_empirical_coverage_through_the_computer_is_nan_not_a_crash():
    """An arm listing it in ``metrics.compute`` on a single-sample validation gets the
    declared N/A: NaN in the dict and a reason on ``last_not_applicable``."""
    import math

    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec(name="empirical_coverage")], primary_metric="empirical_coverage"
        ),
        device="cpu",
    )
    out = computer.compute(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4))
    assert math.isnan(out["empirical_coverage"])
    assert "empirical_coverage" in computer.last_not_applicable


def test_empirical_coverage_refuses_a_mismatched_std_and_a_bad_k():
    from spectramr.core.metrics.context import MetricContext
    from spectramr.core.metrics.quantitative.conformal_risk import EmpiricalCoverageMetric

    pred, target, _ = _coverage_fixture()
    with pytest.raises(ValueError, match="ensemble_std"):
        EmpiricalCoverageMetric()(pred, target, context=MetricContext(ensemble_std=torch.ones(1)))
    with pytest.raises(ValueError, match="shape mismatch"):
        EmpiricalCoverageMetric()(
            pred,
            torch.zeros(1, 1, 3, 3),
            context=MetricContext(ensemble_std=torch.ones(1, 1, 2, 2)),
        )
    with pytest.raises(ValueError, match="coverage k"):
        EmpiricalCoverageMetric(k=0.0)
