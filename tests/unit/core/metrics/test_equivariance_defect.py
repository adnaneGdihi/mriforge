"""Unit tests for the equivariance-defect conformal score (B-1 / Idea 1).

Direct-import tests (no registry / factory resolution): instantiate the metric
and call the standalone functions on tiny tensors. Covers shape, no-NaN, the
key property (perfect equivariance => defect ~ 0), deterministic seeding,
multi-coil handling, AMP-safety, and split-conformal coverage at 1 - alpha.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.equivariance_defect import (
    SUPPORTED_GROUPS,
    EquivarianceDefectMetric,
    conformal_quantile,
    equivariance_defect_score,
    _rotate_image,
)


def _identity(y: torch.Tensor) -> torch.Tensor:
    return y


def test_perfect_equivariance_identity_is_zero():
    """Identity reconstructor commutes with any rotation => defect == 0."""
    y = torch.randn(2, 1, 32, 32)
    s = equivariance_defect_score(_identity, y, num_rotations=6, group="SO2", seed=0)
    assert s.shape == (2,)
    assert torch.all(s >= 0)
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)


def test_equivariant_linear_op_is_near_zero():
    """A rotation-equivariant op (global scale) has near-zero defect."""
    scale = lambda t: 2.5 * t  # noqa: E731 — commutes with rotation
    y = torch.randn(1, 1, 32, 32)
    s = equivariance_defect_score(scale, y, num_rotations=8, group="SO2", seed=1)
    assert float(s.max()) < 1e-4


def test_large_defect_detected_for_non_equivariant_op():
    """A spatially-biased op (multiply by a horizontal ramp) breaks equivariance."""
    h = w = 32
    ramp = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w)

    def non_equivariant(t: torch.Tensor) -> torch.Tensor:
        return t * ramp.to(t.device)

    y = torch.randn(1, 1, h, w)
    s_bad = equivariance_defect_score(
        non_equivariant, y, num_rotations=8, group="SO2", seed=2
    )
    s_good = equivariance_defect_score(_identity, y, num_rotations=8, group="SO2", seed=2)
    assert float(s_bad.mean()) > float(s_good.mean())
    assert float(s_bad.mean()) > 1e-3
    assert not torch.isnan(s_bad).any()


def test_deterministic_seeding():
    """Same seed -> identical score; different seed -> (generally) different."""
    ramp = torch.linspace(-1.0, 1.0, 32).view(1, 1, 1, 32)
    op = lambda t: t * ramp.to(t.device)  # noqa: E731
    y = torch.randn(1, 1, 32, 32)
    a = equivariance_defect_score(op, y, num_rotations=8, group="SO2", seed=7)
    b = equivariance_defect_score(op, y, num_rotations=8, group="SO2", seed=7)
    c = equivariance_defect_score(op, y, num_rotations=8, group="SO2", seed=8)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_multi_coil_complex_handling():
    """Multi-coil complex64 input is accepted; output is per-sample, finite."""
    y = torch.randn(3, 4, 24, 24, dtype=torch.complex64)
    s = equivariance_defect_score(_identity, y, num_rotations=4, group="SO2", seed=0)
    assert s.shape == (3,)
    assert torch.isfinite(s).all()
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)


def test_se2_group_supported_and_finite():
    y = torch.randn(2, 1, 32, 32)
    s = equivariance_defect_score(
        _identity, y, num_rotations=4, group="SE2", seed=3, max_shift_px=3
    )
    # Identity still commutes with the (roll-based) SE2 action exactly.
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-5)
    assert torch.isfinite(s).all()


def test_unknown_group_raises():
    y = torch.randn(1, 1, 16, 16)
    try:
        equivariance_defect_score(_identity, y, group="SO3")
    except ValueError as exc:
        assert "SO3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown group")


def test_bad_num_rotations_raises():
    y = torch.randn(1, 1, 16, 16)
    try:
        equivariance_defect_score(_identity, y, num_rotations=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for num_rotations < 1")


def test_non_4d_input_raises():
    try:
        equivariance_defect_score(_identity, torch.randn(16, 16))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-4D input")


def test_rotate_image_2pi_is_identity():
    """Rotating by 2*pi returns (numerically) the same image."""
    x = torch.randn(1, 1, 32, 32)
    r = _rotate_image(x, 2.0 * torch.pi)
    assert torch.allclose(r, x, atol=1e-4)


def test_amp_safe_no_nan_under_autocast():
    """Score is finite when computed under CPU autocast (AMP-safe path)."""
    y = torch.randn(2, 1, 32, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        s = equivariance_defect_score(_identity, y, num_rotations=4, group="SO2", seed=0)
    assert torch.isfinite(s).all()


def test_metric_imetric_contract():
    """The registered metric is callable and exposes name/higher_is_better."""
    metric = EquivarianceDefectMetric(num_rotations=4, group="SO2", seed=0)
    y = torch.randn(2, 1, 32, 32)
    val = metric(y, y)  # identity recon_fn default => 0
    assert isinstance(val, float)
    assert val == 0.0 or abs(val) < 1e-5
    assert metric.name == "equivariance_defect"
    assert metric.higher_is_better is False
    assert metric.group in SUPPORTED_GROUPS


def test_metric_forward_alias_matches_score():
    metric = EquivarianceDefectMetric(num_rotations=4, group="SO2", seed=5)
    op = lambda t: t * 1.3  # noqa: E731
    y = torch.randn(1, 1, 24, 24)
    a = metric.score(op, y)
    b = metric.forward(op, y)
    assert torch.allclose(a, b)


def test_conformal_quantile_basic():
    """Quantile is finite, non-negative, and conservative for tiny n."""
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    q = conformal_quantile(scores, alpha=0.2)
    assert q >= 0.0
    assert q <= float(scores.max()) + 1e-9
    # Tiny-n / large-(1-alpha) -> returns max (infinite-interval conservative).
    q_small = conformal_quantile(torch.tensor([1.0, 2.0]), alpha=0.1)
    assert q_small == 2.0


def test_conformal_quantile_bad_alpha_raises():
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            conformal_quantile(torch.tensor([1.0, 2.0, 3.0]), alpha=bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for alpha={bad}")


def test_conformal_coverage_at_one_minus_alpha():
    """Empirical coverage of the conformal threshold matches 1 - alpha (MC band).

    With exchangeable scores (drawn i.i.d. here), the fraction of fresh test
    scores below the calibration quantile should be ~ 1 - alpha.
    """
    torch.manual_seed(0)
    alpha = 0.1
    n_cal, n_test, trials = 500, 500, 40
    covered_fracs = []
    for t in range(trials):
        g = torch.Generator().manual_seed(t)
        cal = torch.rand(n_cal, generator=g)
        test = torch.rand(n_test, generator=g)
        q = conformal_quantile(cal, alpha=alpha)
        covered_fracs.append(float((test <= q).float().mean().item()))
    mean_cov = sum(covered_fracs) / len(covered_fracs)
    # Split-conformal guarantees coverage >= 1 - alpha marginally; allow a
    # generous Monte-Carlo band around the nominal 0.90.
    assert 0.85 <= mean_cov <= 0.95
