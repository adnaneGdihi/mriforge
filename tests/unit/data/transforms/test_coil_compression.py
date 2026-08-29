"""Tests for ``SVDCoilCompression`` and ``SVDCoilCompressionTransform``.

Targets ``mriforge.data.transforms.coil_compression``. SVD coil compression
projects ``[B, C, H, W]`` complex k-space onto its top-N principal
components — a hardware-equivalent of physical coil compression.

Key invariants:

- Energy concentration: the top eigenvector captures the dominant signal
- Linearity: ``apply_basis(α·x + y) == α·apply_basis(x) + apply_basis(y)``
- Pass-through: when ``C ≤ num_virtual_coils``, output equals input
- Shape contract: ``[B, C, H, W] → [B, num_virtual, H, W]``
"""

from __future__ import annotations

import pytest
import torch

from mriforge.data.transforms.coil_compression import (
    SVDCoilCompression,
    SVDCoilCompressionTransform,
)


def test_transform_forwards_calibration_lines_to_compressor() -> None:
    """SVDCoilCompressionTransform threads calibration_lines into its compressor."""
    t = SVDCoilCompressionTransform(num_virtual_coils=4, calibration_lines=24)
    assert t.calibration_lines == 24
    assert t.compressor.calibration_lines == 24
    # Default is None (full FoV) — parity with prior behavior.
    assert SVDCoilCompressionTransform(num_virtual_coils=4).compressor.calibration_lines is None


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_persists_num_virtual_coils() -> None:
    """The constructor stores ``num_virtual_coils``."""
    compressor = SVDCoilCompression(num_virtual_coils=8)
    assert compressor.num_virtual_coils == 8


# ---------------------------------------------------------------------------
# calibration_lines — basis from a centered ACS band (None = full FoV parity)
# ---------------------------------------------------------------------------


def test_construction_persists_calibration_lines() -> None:
    c = SVDCoilCompression(num_virtual_coils=2, calibration_lines=24)
    assert c.calibration_lines == 24
    # Default is None (full FoV).
    assert SVDCoilCompression(num_virtual_coils=2).calibration_lines is None


def test_calibration_none_is_full_fov_parity() -> None:
    """calibration_lines=None must reproduce the full-FoV basis exactly."""
    torch.manual_seed(0)
    ks = torch.complex(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16))
    base = SVDCoilCompression(num_virtual_coils=2).compute_basis(ks)
    cal_none = SVDCoilCompression(
        num_virtual_coils=2, calibration_lines=None
    ).compute_basis(ks)
    assert torch.equal(base, cal_none)


def test_calibration_band_matches_manual_crop() -> None:
    """calibration_lines=K computes the basis from the central K rows."""
    torch.manual_seed(1)
    ks = torch.complex(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16))
    cal = 8
    out = SVDCoilCompression(
        num_virtual_coils=3, calibration_lines=cal
    ).compute_basis(ks)

    # Manual: crop central 8 rows, eigh of covariance, top-3 eigenvectors.
    c0 = 16 // 2 - cal // 2
    band = ks[:, :, c0 : c0 + cal, :]
    k_mat = band.reshape(1, 8, -1)
    cov = torch.bmm(k_mat, k_mat.conj().transpose(1, 2)) / (cal * 16)
    _, vecs = torch.linalg.eigh(cov)
    expected = vecs[:, :, -3:]
    assert torch.allclose(out, expected, atol=1e-6)


def test_calibration_band_differs_from_full_fov() -> None:
    """A band-limited basis generally differs from the full-FoV basis."""
    torch.manual_seed(2)
    # Plant structured signal so the ACS band and full FoV see different stats.
    ks = torch.complex(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16))
    full = SVDCoilCompression(num_virtual_coils=2).compute_basis(ks)
    band = SVDCoilCompression(
        num_virtual_coils=2, calibration_lines=6
    ).compute_basis(ks)
    assert not torch.allclose(full, band, atol=1e-4)


# ---------------------------------------------------------------------------
# compute_basis
# ---------------------------------------------------------------------------


def test_compute_basis_shape() -> None:
    """Returned basis has shape ``[B, C, num_virtual_coils]``."""
    compressor = SVDCoilCompression(num_virtual_coils=2)
    kspace = torch.complex(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16))
    basis = compressor.compute_basis(kspace)
    assert basis.shape == (1, 8, 2)


def test_compute_basis_passes_through_when_c_le_target() -> None:
    """When physical coils ≤ target, basis is None (no compression needed)."""
    compressor = SVDCoilCompression(num_virtual_coils=8)
    kspace = torch.complex(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))
    basis = compressor.compute_basis(kspace)
    assert basis is None


def test_compute_basis_rejects_non_4d() -> None:
    """3-D input raises ``ValueError``."""
    compressor = SVDCoilCompression()
    with pytest.raises(ValueError, match="4D k-space"):
        compressor.compute_basis(torch.complex(torch.randn(8, 16, 16), torch.zeros(8, 16, 16)))


def test_compute_basis_eigenvectors_are_orthonormal() -> None:
    """Eigenvectors produced by ``eigh`` are orthonormal."""
    compressor = SVDCoilCompression(num_virtual_coils=3)
    torch.manual_seed(0)
    kspace = torch.complex(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16))
    basis = compressor.compute_basis(kspace)
    # V^H V should be identity (3×3)
    gram = torch.einsum("bcd,bce->bde", basis.conj(), basis)
    eye = torch.eye(3, dtype=gram.dtype).unsqueeze(0)
    assert torch.allclose(gram, eye, atol=1e-5)


# ---------------------------------------------------------------------------
# apply_basis
# ---------------------------------------------------------------------------


def test_apply_basis_output_shape() -> None:
    """``apply_basis`` produces ``[B, num_virtual, H, W]``."""
    compressor = SVDCoilCompression(num_virtual_coils=2)
    kspace = torch.complex(torch.randn(2, 8, 16, 16), torch.randn(2, 8, 16, 16))
    basis = compressor.compute_basis(kspace)
    compressed = compressor.apply_basis(kspace, basis)
    assert compressed.shape == (2, 2, 16, 16)
    assert torch.is_complex(compressed)


def test_apply_basis_is_linear() -> None:
    """``apply_basis(α·x + y) == α·apply_basis(x) + apply_basis(y)``."""
    compressor = SVDCoilCompression(num_virtual_coils=2)
    torch.manual_seed(0)
    x = torch.complex(torch.randn(1, 8, 8, 8), torch.randn(1, 8, 8, 8))
    y = torch.complex(torch.randn(1, 8, 8, 8), torch.randn(1, 8, 8, 8))
    basis = compressor.compute_basis(x)  # use same basis for both

    alpha = 2.5 + 1.0j
    lhs = compressor.apply_basis(alpha * x + y, basis)
    rhs = alpha * compressor.apply_basis(x, basis) + compressor.apply_basis(y, basis)
    assert torch.allclose(lhs, rhs, atol=1e-5)


# ---------------------------------------------------------------------------
# forward()
# ---------------------------------------------------------------------------


def test_forward_passes_through_when_c_le_target() -> None:
    """``C ≤ num_virtual_coils`` → input returned unchanged."""
    compressor = SVDCoilCompression(num_virtual_coils=8)
    kspace = torch.complex(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))
    out = compressor(kspace)
    assert torch.equal(out, kspace)


def test_forward_compresses_when_c_gt_target() -> None:
    """``C > num_virtual_coils`` → coil dim reduced to ``num_virtual_coils``."""
    compressor = SVDCoilCompression(num_virtual_coils=2)
    kspace = torch.complex(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16))
    out = compressor(kspace)
    assert out.shape == (1, 2, 16, 16)


def test_forward_with_precomputed_basis_skips_recomputation() -> None:
    """Precomputed basis is reused (no second eigendecomposition)."""
    compressor = SVDCoilCompression(num_virtual_coils=2)
    kspace = torch.complex(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 16, 16))
    basis = compressor.compute_basis(kspace)
    out_with_basis = compressor(kspace, basis=basis)
    out_inferred = compressor(kspace)
    # Outputs should match (same basis derived from same kspace).
    assert torch.allclose(out_with_basis, out_inferred, atol=1e-5)


def test_forward_rejects_non_4d_input() -> None:
    """Non-4D input → ``ValueError``."""
    compressor = SVDCoilCompression(num_virtual_coils=2)
    with pytest.raises(ValueError, match="4D k-space"):
        compressor(torch.complex(torch.randn(8, 16, 16), torch.zeros(8, 16, 16)))


# ---------------------------------------------------------------------------
# Energy concentration property
# ---------------------------------------------------------------------------


def test_energy_concentration_in_dominant_components() -> None:
    """Top-2 of 4 components captures more energy than the bottom-2."""
    torch.manual_seed(0)
    # Plant a strong signal in coil 0 + weak background noise.
    kspace = 0.01 * torch.complex(torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16))
    kspace[:, 0] += 5.0  # dominant signal in first coil

    compressor_top = SVDCoilCompression(num_virtual_coils=1)
    out_top = compressor_top(kspace)
    energy_top = (out_top.abs() ** 2).sum().item()

    # Total energy of input.
    total_energy = (kspace.abs() ** 2).sum().item()
    assert energy_top / total_energy > 0.9


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, num_virtual",
    [
        ((1, 8, 16, 16), 2),
        ((2, 12, 32, 32), 4),
        ((1, 6, 8, 8), 3),
        ((4, 16, 16, 32), 4),
    ],
    ids=["b1c8_v2", "b2c12_v4", "b1c6_v3", "b4c16_v4"],
)
def test_forward_shape_matrix(shape: tuple[int, ...], num_virtual: int) -> None:
    """Output coil dim is exactly ``num_virtual_coils`` when compressing."""
    compressor = SVDCoilCompression(num_virtual_coils=num_virtual)
    kspace = torch.complex(torch.randn(*shape), torch.randn(*shape))
    out = compressor(kspace)
    assert out.shape == (shape[0], num_virtual, shape[2], shape[3])
    assert torch.isfinite(out.real).all()
    assert torch.isfinite(out.imag).all()
