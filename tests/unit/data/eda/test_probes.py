"""Tests for the EDA physics + statistical probes."""
from __future__ import annotations

import torch
from pytest import approx

from mriforge.data.eda import probes
from mriforge.infrastructure.physics.fft_ops import fft2c


def test_radial_energy_decays_for_smooth_image():
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    img = torch.exp(-(xx ** 2 + yy ** 2) / 0.1)
    ksp = fft2c(img.unsqueeze(0))
    _radii, energy, cumulative = probes.radial_energy(ksp)
    assert energy[0] > energy[-1]                     # smooth → low-frequency dominated
    assert cumulative[-1] == approx(1.0)


def test_detect_sampling_full_vs_undersampled():
    ksp = torch.randn(1, 32, 32) + 1j * torch.randn(1, 32, 32)
    assert probes.detect_sampling(ksp)["fully_sampled"] is True
    ksp[:, ::2, :] = 0
    us = probes.detect_sampling(ksp)
    assert us["fully_sampled"] is False
    assert us["acceleration"] > 1.5


def test_normalized_distributions_has_four_schemes():
    imgs = [torch.rand(16, 16) * 100]
    dist = probes.normalized_distributions(imgs)
    assert set(dist) >= {"raw", "max", "pclip", "zscore"}
    assert all(len(v) > 0 for v in dist.values())
    assert dist["max"].max() <= 1.0 + 1e-6


def test_percentiles_monotone():
    imgs = [torch.rand(32, 32)]
    p = probes.percentiles(imgs)
    assert p["p0.1"] <= p["p50"] <= p["p99.9"] <= p["p100"]


def test_coil_maps_singlecoil_returns_none():
    assert probes.coil_maps(torch.randn(1, 16, 16) + 0j) is None


def test_bounded_kspace_caps_coils_and_crops_fov():
    """Unbounded ESPIRiT allocates O(H·W·C²) and OOM-kills the EDA process on a large
    multicoil slice. ``_bounded_kspace`` caps the coils and center-crops the FOV first."""
    k = torch.zeros(20, 300, 256, dtype=torch.complex64)
    b = probes._bounded_kspace(k, max_coils=8, max_fov=128)
    assert b.shape == (8, 128, 128)


def test_bounded_kspace_leaves_small_input_untouched():
    k = torch.zeros(4, 64, 64, dtype=torch.complex64)
    b = probes._bounded_kspace(k, max_coils=8, max_fov=128)
    assert b.shape == (4, 64, 64)


def test_coil_maps_bounds_oversized_multicoil_input():
    """coil_maps must bound its ESPIRiT input so the per-pixel eigendecomposition cannot
    OOM-kill the suite — the result is the (smooth) low-res, coil-capped estimate."""
    torch.manual_seed(0)
    big = torch.randn(10, 144, 144) + 1j * torch.randn(10, 144, 144)
    csm = probes.coil_maps(big, max_coils=8, max_fov=128)
    assert csm is not None
    assert csm.shape[0] <= 8            # coils capped
    assert max(csm.shape[-2:]) <= 128   # FOV cropped
