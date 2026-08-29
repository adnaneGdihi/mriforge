"""Tests for the real acquisition-asset MetricContext builder.

CPU-only, deterministic. Builds a synthetic multi-coil fully-sampled acquisition
(smooth phantom x unit-RSS sensitivity maps) and asserts
``prepare_metric_context`` returns the REAL measurement assets — not a
fabricated stand-in — and that the recovered k-space + SENSE reconstructor
round-trip back to the phantom.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.context import MetricContext
from mriforge.infrastructure.physics.asset_preparation import (
    estimate_noise_covariance,
    prepare_metric_context,
)


def _phantom(h: int = 32, w: int = 32) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    return torch.exp(-(yy**2 + xx**2) / 0.3).clamp(0, 1)[None, None].float()


def _unit_rss_smaps(c: int = 4, h: int = 32, w: int = 32) -> torch.Tensor:
    """Smooth complex maps normalised to sum_c |S_c|^2 == 1 (Roemer-exact)."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
    )
    centres = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)][:c]
    maps = []
    for cy, cx in centres:
        bump = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 0.8)
        phase = torch.exp(1j * (0.5 * xx + 0.3 * yy))
        maps.append((bump * phase).to(torch.complex64))
    s = torch.stack(maps, dim=0)[None]  # [1, C, H, W]
    rss = torch.sqrt((s.abs() ** 2).sum(dim=1, keepdim=True)).clamp_min(1e-6)
    return s / rss


def _multicoil():
    phantom = _phantom()
    smaps = _unit_rss_smaps()
    coil_images = phantom.to(torch.complex64) * smaps  # [1, C, H, W]
    return phantom, smaps, coil_images


def test_context_carries_real_assets_not_stand_in():
    phantom, smaps, coil_images = _multicoil()
    ctx = prepare_metric_context(coil_images=coil_images, smaps=smaps, p99=1.0)

    assert isinstance(ctx, MetricContext)
    # Real coil maps threaded verbatim (identity, not a birdcage synthesis).
    assert ctx.coil_maps is smaps
    assert ctx.acceleration == 1.0
    assert ctx.mask is None  # fully sampled reference
    assert ctx.y_kspace is not None and ctx.y_kspace.is_complex()
    assert ctx.y_kspace.shape == coil_images.shape
    assert ctx.foreground_mask is not None
    assert ctx.foreground_mask.shape[-2:] == phantom.shape[-2:]
    assert callable(ctx.reconstructor)
    # Prior/encoder are genuinely unavailable -> left unset (honest not_applicable).
    assert ctx.prior_model is None
    assert ctx.encoder is None


def test_kspace_and_reconstructor_round_trip():
    phantom, smaps, coil_images = _multicoil()
    ctx = prepare_metric_context(coil_images=coil_images, smaps=smaps, p99=1.0)
    # SENSE-adjoint of the fully-sampled real k-space with unit-RSS maps recovers
    # the phantom (Roemer-exact), confirming y_kspace is the true measurement.
    recon = ctx.reconstructor(ctx.y_kspace).abs()
    assert torch.allclose(recon.squeeze(), phantom.squeeze(), atol=1e-4)


def test_p99_scaling_matches_graded_image_scale():
    phantom, smaps, coil_images = _multicoil()
    p99 = 7.5
    ctx = prepare_metric_context(coil_images=coil_images, smaps=smaps, p99=p99)
    recon = ctx.reconstructor(ctx.y_kspace).abs()
    # k-space was divided by p99, so the reconstruction lands on phantom / p99.
    assert torch.allclose(recon.squeeze(), (phantom / p99).squeeze(), atol=1e-4)


def test_noise_covariance_shape_and_hermitian():
    _, _, coil_images = _multicoil()
    cov = estimate_noise_covariance(coil_images)
    c = coil_images.shape[1]
    assert cov.shape == (c, c)
    assert cov.is_complex()
    # Sample covariance is Hermitian.
    assert torch.allclose(cov, cov.conj().transpose(0, 1), atol=1e-5)


def test_accepts_chw_layout():
    _, smaps, coil_images = _multicoil()
    ctx = prepare_metric_context(coil_images=coil_images[0], smaps=smaps[0], p99=1.0)
    assert ctx.y_kspace.shape == coil_images.shape
