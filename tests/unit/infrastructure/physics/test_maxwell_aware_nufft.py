"""Oracle tests for the Maxwell-aware concomitant NUFFT (A-3.4 MA-NUFFT)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.maxwell_aware_nufft import (
    MaxwellAwareNUFFT,
    maxwell_aware_nufft_forward,
    maxwell_concomitant_phase,
)


def _cartesian_ktraj(h: int, w: int) -> torch.Tensor:
    # Integer (k_y, k_x) on the grid -> NDFT reproduces the standard 2-D DFT.
    ky, kx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([kx.reshape(-1).float(), ky.reshape(-1).float()], dim=1)  # [H*W, 2] (kx, ky)


# --------------------------------------------------------------------------- #
# NDFT correctness: on the Cartesian grid, MA-NUFFT (no concomitant) == DFT
# --------------------------------------------------------------------------- #


def test_ndft_on_grid_matches_dft() -> None:
    torch.manual_seed(0)
    img = torch.rand(1, 1, 8, 8)
    ktraj = _cartesian_ktraj(8, 8)
    y = maxwell_aware_nufft_forward(img, ktraj, concomitant_phase=None)
    ref = torch.fft.fft2(img[:, 0].to(torch.complex128)).reshape(1, -1)
    assert torch.allclose(y.to(torch.complex128), ref, atol=1e-4)


def test_zero_concomitant_phase_equals_plain_nufft() -> None:
    torch.manual_seed(1)
    img = torch.rand(2, 1, 6, 6)
    ktraj = torch.rand(20, 2) * 6.0
    plain = maxwell_aware_nufft_forward(img, ktraj, concomitant_phase=None)
    zero = maxwell_aware_nufft_forward(img, ktraj, concomitant_phase=torch.zeros(6, 6))
    assert torch.allclose(plain, zero, atol=1e-5)


# --------------------------------------------------------------------------- #
# Concomitant phase: the physics oracles (1/B0 and G^2 scaling)
# --------------------------------------------------------------------------- #


def test_concomitant_phase_scales_inverse_with_field() -> None:
    # Halving B0 doubles the concomitant phase (f_c ~ 1/B0).
    p_low = maxwell_concomitant_phase((16, 16), gradient_amp_mt_per_m=20.0, field_strength_t=0.1)
    p_high = maxwell_concomitant_phase((16, 16), gradient_amp_mt_per_m=20.0, field_strength_t=0.2)
    ratio = p_low.abs().max() / (p_high.abs().max() + 1e-12)
    assert float(ratio) == pytest.approx(2.0, rel=1e-3)


def test_concomitant_phase_scales_quadratically_with_gradient() -> None:
    # Doubling the gradient quadruples the concomitant phase (f_c ~ G^2).
    p1 = maxwell_concomitant_phase((16, 16), gradient_amp_mt_per_m=10.0, field_strength_t=0.1)
    p2 = maxwell_concomitant_phase((16, 16), gradient_amp_mt_per_m=20.0, field_strength_t=0.1)
    ratio = p2.abs().max() / (p1.abs().max() + 1e-12)
    assert float(ratio) == pytest.approx(4.0, rel=1e-3)


def test_zero_gradient_gives_zero_concomitant_phase() -> None:
    p = maxwell_concomitant_phase((16, 16), gradient_amp_mt_per_m=0.0, field_strength_t=0.1)
    assert float(p.abs().max()) == pytest.approx(0.0, abs=1e-9)


def test_high_field_concomitant_negligible_vs_low_field() -> None:
    p_ulf = maxwell_concomitant_phase((16, 16), gradient_amp_mt_per_m=20.0, field_strength_t=0.064)
    p_hf = maxwell_concomitant_phase((16, 16), gradient_amp_mt_per_m=20.0, field_strength_t=3.0)
    assert float(p_hf.abs().max()) < float(p_ulf.abs().max()) / 40.0


# --------------------------------------------------------------------------- #
# Concomitant phase changes the measured signal; centre voxel is unaffected
# --------------------------------------------------------------------------- #


def test_concomitant_phase_changes_signal() -> None:
    torch.manual_seed(2)
    img = torch.rand(1, 1, 16, 16)
    ktraj = torch.rand(30, 2) * 16.0
    phase = maxwell_concomitant_phase((16, 16), gradient_amp_mt_per_m=50.0, field_strength_t=0.064)
    y_plain = maxwell_aware_nufft_forward(img, ktraj, concomitant_phase=None)
    y_maxwell = maxwell_aware_nufft_forward(img, ktraj, concomitant_phase=phase)
    assert not torch.allclose(y_plain, y_maxwell, atol=1e-3)


def test_concomitant_phase_zero_at_isocentre() -> None:
    # The Maxwell term vanishes at iso-centre (r=0); for an even grid the centre is
    # between voxels, so the minimum |phase| is near the middle, not the corners.
    p = maxwell_concomitant_phase((17, 17), gradient_amp_mt_per_m=20.0, field_strength_t=0.1)
    assert float(p[8, 8]) == pytest.approx(0.0, abs=1e-9)  # exact centre of a 17x17 grid
    assert float(p[0, 0]) > 0.0  # corners carry the most phase


# --------------------------------------------------------------------------- #
# Operator wrapper + provenance
# --------------------------------------------------------------------------- #


def test_operator_metadata_reports_activity() -> None:
    phase = maxwell_concomitant_phase((8, 8), gradient_amp_mt_per_m=20.0, field_strength_t=0.1)
    op = MaxwellAwareNUFFT(concomitant_phase=phase)
    meta = op.metadata()
    assert meta["concomitant_active"] is True
    assert meta["max_concomitant_phase_rad"] > 0.0
    assert MaxwellAwareNUFFT(None).metadata()["concomitant_active"] is False


def test_field_strength_must_be_positive() -> None:
    with pytest.raises(ValueError, match="field_strength_t"):
        maxwell_concomitant_phase((8, 8), gradient_amp_mt_per_m=20.0, field_strength_t=0.0)
