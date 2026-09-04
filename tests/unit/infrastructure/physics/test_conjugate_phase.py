"""Tests for ``ConjugatePhaseReconstructor``.

Targets ``spectramr.infrastructure.physics.conjugate_phase``. Conjugate Phase
Reconstruction (CPR) corrects for B₀-induced phase distortion via
multi-frequency interpolation. The reconstructor is a deterministic
algorithm with no learnable parameters — easy to test against the
known identity case (B0 = 0).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.conjugate_phase import (
    ConjugatePhaseReconstructor,
)


def _small_recon(n: int = 16) -> ConjugatePhaseReconstructor:
    """Cheap reconstructor for fast tests."""
    return ConjugatePhaseReconstructor(
        freq_range_hz=(-1000.0, 1000.0),
        freq_step_hz=100.0,
        bw_hz=10_000.0,
        n_readout=n,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction_produces_expected_bin_count() -> None:
    """``n_bins`` follows the documented (max - min) / step + 1 formula."""
    cpr = ConjugatePhaseReconstructor(
        freq_range_hz=(-100.0, 100.0), freq_step_hz=50.0
    )
    # 200 / 50 + 1 = 5 bins
    assert cpr.n_bins == 5


def test_dwell_time_is_one_over_bandwidth() -> None:
    cpr = ConjugatePhaseReconstructor(bw_hz=20_000.0)
    assert pytest.approx(cpr.dwell, rel=1e-9) == 1.0 / 20_000.0


# ---------------------------------------------------------------------------
# Reconstruction — identity case
# ---------------------------------------------------------------------------


def test_zero_b0_recon_runs_without_error() -> None:
    """B0 = 0 → frequency bin index is the centre — no phase distortion path."""
    cpr = _small_recon(n=16)
    H = W = 16
    kspace = torch.randn(1, 1, H, W, dtype=torch.complex64)
    b0 = torch.zeros(1, 1, H, W)
    out = cpr.reconstruct(kspace, b0)
    assert out.shape == (1, 1, H, W)
    assert torch.is_complex(out)
    assert torch.isfinite(out).all()


def test_t_shift_zero_default_works() -> None:
    """``t_shift=0`` default is the documented unshifted readout."""
    cpr = _small_recon(n=8)
    H = W = 8
    kspace = torch.randn(1, 1, H, W, dtype=torch.complex64)
    b0 = torch.zeros(1, 1, H, W)
    out = cpr.reconstruct(kspace, b0, t_shift=0.0)
    assert out.shape == (1, 1, H, W)


def test_recon_handles_nonzero_b0_map() -> None:
    """Non-zero B0 produces a finite reconstruction (correction applied)."""
    cpr = _small_recon(n=16)
    H = W = 16
    kspace = torch.randn(1, 1, H, W, dtype=torch.complex64)
    # B0 within the configured range so all voxels find a valid bin.
    b0 = torch.randn(1, 1, H, W) * 200.0  # ±200 Hz
    out = cpr.reconstruct(kspace, b0)
    assert torch.isfinite(out.real).all()
    assert torch.isfinite(out.imag).all()


def test_per_voxel_interpolation_is_spatially_resolved() -> None:
    """The vectorized interpolation must gather each voxel's OWN B0 bin.

    Regression for the per-voxel Python loop (two ``.item()`` syncs each) that
    was vectorized: a spatially-varying B0 must yield a spatially-varying recon,
    and the gather must index per voxel (top/bottom halves differ).
    """
    cpr = _small_recon(n=16)
    torch.manual_seed(0)
    kspace = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
    b0 = torch.zeros(1, 1, 16, 16)
    b0[..., :8, :] = 500.0  # top half +500 Hz
    b0[..., 8:, :] = -500.0  # bottom half -500 Hz
    out = cpr.reconstruct(kspace, b0)
    assert not torch.allclose(out[..., :8, :], out[..., 8:, :], atol=1e-4)
    out0 = cpr.reconstruct(kspace, torch.zeros_like(b0))
    assert not torch.allclose(out, out0, atol=1e-4)


def test_b0_clamped_to_bin_range() -> None:
    """Out-of-range B0 values are clamped (via ``clamp(0, n_bins-2)``).

    This is a defensive contract: a B0 well outside the configured range
    still produces a finite reconstruction (clamped to nearest bin).
    """
    cpr = _small_recon(n=8)
    H = W = 8
    kspace = torch.randn(1, 1, H, W, dtype=torch.complex64)
    # Way outside [-1000, 1000] Hz.
    b0 = torch.full((1, 1, H, W), 50_000.0)
    out = cpr.reconstruct(kspace, b0)
    assert torch.isfinite(out.real).all()


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h,w",
    [(8, 8), (16, 16), (8, 16)],
    ids=lambda v: f"{v}",
)
def test_recon_shape_matrix(h: int, w: int) -> None:
    """Output shape matches input k-space spatial dims."""
    cpr = ConjugatePhaseReconstructor(
        freq_range_hz=(-500.0, 500.0),
        freq_step_hz=100.0,
        bw_hz=10_000.0,
        n_readout=w,
    )
    kspace = torch.randn(1, 1, h, w, dtype=torch.complex64)
    b0 = torch.zeros(1, 1, h, w)
    out = cpr.reconstruct(kspace, b0)
    assert out.shape == (1, 1, h, w)
