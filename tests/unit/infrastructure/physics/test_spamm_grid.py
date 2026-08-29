"""Tests for SPAMM tagging injection and the Dirac notch filter.

Targets ``mriforge.infrastructure.physics.spamm_grid``. The injector applies a
``cos(2πfX)·cos(2πfY)`` tag pattern on ``X, Y ∈ [-1, 1]`` (FOV width 2), so
each tag harmonic sits ``2f`` k-space bins from the centre — at the four
diagonal corners ``(±2f, ±2f)``. The notch filter must null exactly those
bins.

Regression (2026-07): the notch was placed at ``±round(f)`` and therefore
missed the ``±2f`` tags entirely, so the advertised k-space erasure was a
no-op.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.physics.spamm_grid import (
    DiracNotchFilter,
    SPAMMGridInjector,
)


def _tag_kspace(H: int, W: int, f: float) -> torch.Tensor:
    """Centred k-space of a tagged flat image ``[1, 1, H, W]`` complex."""
    injector = SPAMMGridInjector(spatial_freq_x=f, spatial_freq_y=f, tag_contrast=0.8)
    tagged = injector.inject(torch.ones(1, 1, H, W))
    return torch.fft.fftshift(torch.fft.fft2(tagged), dim=(-2, -1))


def test_tag_harmonics_appear_at_two_f_corners() -> None:
    """The four dominant non-DC peaks are at (±2f, ±2f)."""
    H = W = 64
    f = 8.0
    kspace = _tag_kspace(H, W, f)[0, 0]
    cy, cx = H // 2, W // 2
    mag = kspace.abs().clone()
    mag[cy, cx] = 0.0  # ignore DC
    idx = torch.topk(mag.flatten(), 4).indices
    offsets = sorted((int(i // W) - cy, int(i % W) - cx) for i in idx.tolist())
    o = int(round(2 * f))
    assert offsets == sorted(
        [(-o, -o), (-o, o), (o, -o), (o, o)]
    ), f"tag peaks not at ±2f corners: {offsets}"


def test_notch_nulls_the_tag_harmonics() -> None:
    """The notch zeroes the ±2f corners and leaves DC intact."""
    H = W = 64
    f = 8.0
    kspace = _tag_kspace(H, W, f)
    cy, cx = H // 2, W // 2
    o = int(round(2 * f))
    notch = DiracNotchFilter(
        im_size=(H, W), spatial_freq_x=f, spatial_freq_y=f, notch_radius=2
    )
    filtered = notch.apply_mask(kspace)
    for sy in (-o, o):
        for sx in (-o, o):
            assert filtered[0, 0, cy + sy, cx + sx].abs().item() < 1e-6
    # DC is untouched.
    assert filtered[0, 0, cy, cx].abs().item() == kspace[0, 0, cy, cx].abs().item()


def test_notch_reduces_total_tag_energy() -> None:
    """Applying the notch removes energy (it is not a no-op — the old bug)."""
    H = W = 64
    f = 8.0
    kspace = _tag_kspace(H, W, f)
    notch = DiracNotchFilter(
        im_size=(H, W), spatial_freq_x=f, spatial_freq_y=f, notch_radius=2
    )
    filtered = notch.apply_mask(kspace)
    assert filtered.abs().sum().item() < kspace.abs().sum().item()
