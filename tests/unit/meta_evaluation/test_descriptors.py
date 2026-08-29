"""Descriptor invariances and shapes."""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.meta_evaluation.descriptors import (
    DescriptorConfig,
    haar_pyramid_energies,
    radial_spectral_profile,
    residual_descriptor,
    scattering_orientation_energies,
)


def test_radial_profile_sums_to_one() -> None:
    r = torch.randn(1, 32, 32)
    p = radial_spectral_profile(r, n_bins=12)
    assert p.shape == (12,)
    assert abs(float(p.sum().item()) - 1.0) < 1e-6


def test_radial_profile_energy_invariance() -> None:
    r = torch.randn(1, 32, 32)
    p1 = radial_spectral_profile(r, n_bins=8)
    p2 = radial_spectral_profile(5.0 * r, n_bins=8)
    # Same shape, same normalized profile.
    assert torch.allclose(p1, p2, atol=1e-6)


def test_haar_pyramid_shape_and_normalization() -> None:
    r = torch.randn(1, 32, 32)
    coefs = haar_pyramid_energies(r, levels=3)
    assert coefs.shape == (9,)
    # All non-negative.
    assert (coefs >= 0).all().item()


def test_scattering_shape_and_invariance() -> None:
    r = torch.randn(1, 32, 32)
    s1 = scattering_orientation_energies(r, n_orientations=4, n_scales=2)
    s2 = scattering_orientation_energies(3.0 * r, n_orientations=4, n_scales=2)
    assert s1.shape == (8,)
    assert torch.allclose(s1, s2, atol=1e-5)


def test_residual_descriptor_concatenates_parts() -> None:
    cfg = DescriptorConfig(
        n_radial_bins=4,
        wavelet_levels=2,
        scattering_orientations=3,
        scattering_scales=2,
    )
    z = residual_descriptor(torch.randn(1, 32, 32), cfg)
    expected = 4 + 2 * 3 + 2 * 3  # radial + wavelet + scattering
    assert z.shape == (expected,)


def test_radial_profile_is_device_consistent_for_direct_calls() -> None:
    """``radial_spectral_profile`` builds its grid + scatter_add accumulators
    on the SAME device as the input (regression: a CUDA residual previously
    mixed a CPU ``self``/index with a CUDA ``src`` in ``scatter_add_`` and
    raised a device-mismatch RuntimeError). On CPU this is the identity path.
    """
    r = torch.randn(1, 32, 32)
    out = radial_spectral_profile(r, n_bins=8)
    assert out.device == r.device


@pytest.mark.gpu
@torch.no_grad()
def test_residual_descriptor_accepts_cuda_residual() -> None:
    """The sim2rank cdscr ranker feeds ``s.residual`` (often on CUDA) to
    ``residual_descriptor`` and then ``torch.stack``-es the results, so the
    composed descriptor must come back on a single, stack-compatible device.
    Regression for the 2026-05-21 sim2rank crash in ``radial_spectral_profile``.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = DescriptorConfig(
        n_radial_bins=4,
        wavelet_levels=2,
        scattering_orientations=3,
        scattering_scales=2,
    )
    z = residual_descriptor(torch.randn(1, 32, 32, device="cuda"), cfg)
    assert z.shape == (4 + 2 * 3 + 2 * 3,)
    # All descriptors are stacked together downstream — a single device is
    # required. CPU is the float64-native home chosen by the SSOT entry.
    assert not z.is_cuda
