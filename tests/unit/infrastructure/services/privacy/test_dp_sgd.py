"""Tests for the DP-SGD primitives.

Targets ``mriforge.infrastructure.services.privacy.dp_sgd``.

Categories:

- ``clip_per_sample_grad``: norm bound enforced per row;
  in-bound rows untouched; out-of-bound rows scaled down only;
  validation
- ``add_gaussian_noise``: noise std matches ``σ · C / B`` formula;
  ``noise_multiplier=0`` is a no-op; deterministic with a seeded
  ``torch.Generator``
- ``compose_dp_sgd_update``: composite output shape matches one
  per-sample gradient (averaged across batch)
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.services.privacy.dp_sgd import (
    add_gaussian_noise,
    clip_per_sample_grad,
    compose_dp_sgd_update,
)


# ---------------------------------------------------------------------------
# clip_per_sample_grad
# ---------------------------------------------------------------------------


def test_clip_caps_l2_norm() -> None:
    """Each per-sample gradient L2 norm ≤ ``max_grad_norm`` after clipping."""
    grads = torch.randn(8, 4, 4) * 100.0  # huge, certainly above C=1
    clipped = clip_per_sample_grad(grads, max_grad_norm=1.0)
    norms = clipped.flatten(1).norm(dim=1)
    assert (norms <= 1.0 + 1e-5).all()


def test_clip_does_not_grow_small_gradients() -> None:
    """Per-sample gradients already below the cap are unchanged."""
    small = torch.full((4, 2, 2), 0.01)  # L2 norm ≈ 0.04
    clipped = clip_per_sample_grad(small, max_grad_norm=1.0)
    assert torch.allclose(clipped, small)


def test_clip_zero_max_grad_norm_raises() -> None:
    """``max_grad_norm <= 0`` rejected (no scaling rule)."""
    with pytest.raises(ValueError, match="max_grad_norm"):
        clip_per_sample_grad(torch.zeros(4, 2), max_grad_norm=0.0)


def test_clip_empty_batch_raises() -> None:
    """Empty per-sample-grad tensor rejected."""
    with pytest.raises(ValueError, match="empty batch"):
        clip_per_sample_grad(torch.zeros(0, 2), max_grad_norm=1.0)


# ---------------------------------------------------------------------------
# add_gaussian_noise
# ---------------------------------------------------------------------------


def test_zero_noise_is_no_op() -> None:
    """``noise_multiplier=0`` returns the input unchanged."""
    grad = torch.randn(2, 3)
    out = add_gaussian_noise(grad, noise_multiplier=0.0, max_grad_norm=1.0, batch_size=4)
    assert torch.equal(out, grad)


def test_noise_std_matches_formula() -> None:
    """Empirical noise std ≈ ``σ · C / B`` over a large draw."""
    # Build a zero-grad target so the noise we observe = the noise added.
    base = torch.zeros(1024)
    sigma = 1.5
    C = 0.7
    B = 16
    expected_std = sigma * C / B
    out = add_gaussian_noise(base, noise_multiplier=sigma, max_grad_norm=C, batch_size=B)
    measured_std = out.std(unbiased=False).item()
    # Sample-std fluctuation is ~ expected_std / sqrt(2N).
    assert abs(measured_std - expected_std) < expected_std * 0.1


def test_seeded_generator_reproducible() -> None:
    """Same generator seed → identical noise."""
    base = torch.zeros(64)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = add_gaussian_noise(base, 1.0, 1.0, 4, generator=g1)
    out2 = add_gaussian_noise(base, 1.0, 1.0, 4, generator=g2)
    assert torch.equal(out1, out2)


def test_negative_noise_multiplier_rejected() -> None:
    """``σ < 0`` rejected."""
    with pytest.raises(ValueError, match="noise_multiplier"):
        add_gaussian_noise(torch.zeros(4), -1.0, 1.0, 4)


def test_zero_batch_size_rejected() -> None:
    """``batch_size = 0`` rejected (would divide by zero)."""
    with pytest.raises(ValueError, match="batch_size"):
        add_gaussian_noise(torch.zeros(4), 1.0, 1.0, 0)


def test_zero_max_grad_norm_rejected() -> None:
    """``C = 0`` is undefined (would yield zero std for σ > 0)."""
    with pytest.raises(ValueError, match="max_grad_norm"):
        add_gaussian_noise(torch.zeros(4), 1.0, 0.0, 4)


# ---------------------------------------------------------------------------
# compose_dp_sgd_update
# ---------------------------------------------------------------------------


def test_compose_output_shape_matches_param_shape() -> None:
    """Composite collapses the batch dim — result has the per-param shape."""
    per_sample_grads = torch.randn(8, 3, 4)
    out = compose_dp_sgd_update(
        per_sample_grads, max_grad_norm=1.0, noise_multiplier=0.5
    )
    assert out.shape == (3, 4)


def test_compose_zero_noise_equals_clipped_mean() -> None:
    """With ``σ = 0`` the composite is exactly ``mean(clipped)``."""
    grads = torch.randn(8, 4)
    expected = clip_per_sample_grad(grads, 1.0).mean(dim=0)
    out = compose_dp_sgd_update(grads, max_grad_norm=1.0, noise_multiplier=0.0)
    assert torch.allclose(out, expected)


def test_compose_changes_under_noise() -> None:
    """Non-zero noise produces a different result than the noiseless one."""
    torch.manual_seed(0)
    grads = torch.randn(8, 16)
    g = torch.Generator().manual_seed(123)
    deterministic = compose_dp_sgd_update(grads, 1.0, 0.0)
    noisy = compose_dp_sgd_update(grads, 1.0, 0.5, generator=g)
    assert not torch.allclose(deterministic, noisy)
