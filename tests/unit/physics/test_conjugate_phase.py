"""Canary / parametrized / edge-case / expected-failure tests for conjugate_phase.py.

Covers ConjugatePhaseReconstructor.reconstruct().

We use a narrow frequency range and large step to keep the number of MFI bins
small (CPU-tractable). The module's per-voxel loop is intentionally small-N only.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _kspace(n_pe: int, n_ro: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, n_pe, n_ro, dtype=torch.complex64, generator=g)


def _b0_map(n_pe: int, n_ro: int, hz: float = 0.0) -> torch.Tensor:
    return torch.full((1, 1, n_pe, n_ro), hz)


def _cpr_small():
    """Return a CPR instance with a tiny frequency range for fast CPU tests."""
    from spectramr.infrastructure.physics.conjugate_phase import ConjugatePhaseReconstructor
    return ConjugatePhaseReconstructor(
        freq_range_hz=(-100.0, 100.0),
        freq_step_hz=50.0,     # 5 bins only
        bw_hz=10_000.0,
        n_readout=8,
    )


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_cpr_import():
    """Module imports and ConjugatePhaseReconstructor can be instantiated."""
    from spectramr.infrastructure.physics.conjugate_phase import ConjugatePhaseReconstructor
    cpr = ConjugatePhaseReconstructor()
    assert cpr is not None


@pytest.mark.physics
def test_canary_cpr_reconstruct_trivial():
    """reconstruct() on an 8×8 k-space returns [1, 1, 8, 8] complex image."""
    cpr = _cpr_small()
    ks = _kspace(8, 8)
    b0 = _b0_map(8, 8, hz=50.0)
    out = cpr.reconstruct(ks, b0)
    assert out.shape == (1, 1, 8, 8)
    assert torch.is_complex(out)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "n_pe, n_ro",
    [
        (8, 8),
        (8, 16),    # rectangular
        (16, 8),
        (16, 16),
    ],
    ids=["8x8", "8x16", "16x8", "16x16"],
)
def test_cpr_reconstruct_shape(n_pe, n_ro):
    """Output shape equals [1, 1, H, W] = [1, 1, n_pe, n_ro]."""
    from spectramr.infrastructure.physics.conjugate_phase import ConjugatePhaseReconstructor
    cpr = ConjugatePhaseReconstructor(
        freq_range_hz=(-100.0, 100.0),
        freq_step_hz=50.0,
        bw_hz=10_000.0,
        n_readout=n_ro,
    )
    ks = _kspace(n_pe, n_ro)
    b0 = _b0_map(n_pe, n_ro, hz=0.0)
    out = cpr.reconstruct(ks, b0)
    assert out.shape == (1, 1, n_pe, n_ro), f"Expected [1,1,{n_pe},{n_ro}], got {tuple(out.shape)}"


@pytest.mark.physics
@pytest.mark.parametrize("b0_hz", [-100.0, 0.0, 50.0, 100.0])
def test_cpr_various_b0(b0_hz):
    """CPR runs without NaN across different uniform B0 offsets."""
    cpr = _cpr_small()
    ks = _kspace(8, 8)
    b0 = _b0_map(8, 8, hz=b0_hz)
    out = cpr.reconstruct(ks, b0)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_cpr_t_shift():
    """Non-zero t_shift produces output without error."""
    cpr = _cpr_small()
    ks = _kspace(8, 8)
    b0 = _b0_map(8, 8, hz=50.0)
    out = cpr.reconstruct(ks, b0, t_shift=1e-5)
    assert out.shape == (1, 1, 8, 8)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_cpr_edge_zero_kspace():
    """All-zero k-space → zero (or near-zero) image."""
    cpr = _cpr_small()
    ks = torch.zeros(1, 1, 8, 8, dtype=torch.complex64)
    b0 = _b0_map(8, 8)
    out = cpr.reconstruct(ks, b0)
    assert torch.allclose(out.abs(), torch.zeros_like(out.abs()), atol=1e-6)


@pytest.mark.physics
def test_cpr_edge_uniform_b0_zero():
    """With B0=0, CPR should match plain IFFT up to numeric tolerance."""
    from spectramr.infrastructure.physics.fft_ops import ifft2c
    from spectramr.infrastructure.physics.conjugate_phase import ConjugatePhaseReconstructor
    # Very tight frequency grid around 0 Hz — minimal demodulation
    cpr = ConjugatePhaseReconstructor(
        freq_range_hz=(0.0, 0.0),
        freq_step_hz=1.0,
        bw_hz=10_000.0,
        n_readout=8,
    )
    ks = _kspace(8, 8, seed=5)
    b0 = _b0_map(8, 8, hz=0.0)
    cpr_out = cpr.reconstruct(ks, b0)
    ifft_out = ifft2c(ks)
    # They should be close-ish (same demodulation freq 0)
    assert torch.allclose(cpr_out.abs(), ifft_out.abs(), atol=0.1)


@pytest.mark.physics
def test_cpr_edge_single_bin():
    """Single-bin frequency range (freq_step larger than range) still runs."""
    from spectramr.infrastructure.physics.conjugate_phase import ConjugatePhaseReconstructor
    cpr = ConjugatePhaseReconstructor(
        freq_range_hz=(0.0, 0.0),
        freq_step_hz=10.0,
        bw_hz=5_000.0,
        n_readout=8,
    )
    ks = _kspace(8, 8)
    b0 = _b0_map(8, 8)
    out = cpr.reconstruct(ks, b0)
    assert out.shape == (1, 1, 8, 8)
