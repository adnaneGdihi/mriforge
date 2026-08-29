"""Tests for ``EncodingMatrixBuilder``.

Targets ``mriforge.infrastructure.physics.encoding_matrix``. Builds the
explicit MRI encoding matrix and provides matrix-free forward / adjoint
operators that incorporate per-readout B₀ phase.

Categories:

- Unit: shape contracts, B₀=0 reduces to plain DFT
- Property: adjoint test ``<E·m, y> == <m, E†·y>`` (within tolerance)
- Property: forward / adjoint round-trip with zero B₀ matches DFT round-trip
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_persists_geometry() -> None:
    """Constructor stores ``n_ro``, ``n_pe``, ``fov``, ``bw``, ``dwell``."""
    builder = EncodingMatrixBuilder(n_readout=16, n_phase=16, fov_m=0.2, bw_hz=10000.0)
    assert builder.n_ro == 16
    assert builder.n_pe == 16
    assert builder.fov == 0.2
    assert builder.bw == 10000.0
    assert builder.dwell == 1e-4


# ---------------------------------------------------------------------------
# Explicit matrix
# ---------------------------------------------------------------------------


def test_build_explicit_shape() -> None:
    """Output matrix has shape ``[N_pe * N_ro, H * W]`` and complex dtype."""
    builder = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    b0 = torch.zeros(1, 1, 8, 8)
    E = builder.build_explicit(b0)
    assert E.shape == (64, 64)
    assert E.dtype == torch.complex64


def test_build_explicit_zero_b0_unit_modulus() -> None:
    """With B₀ = 0, every entry has unit modulus (pure DFT phase)."""
    builder = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    E = builder.build_explicit(torch.zeros(8, 8))
    mods = E.abs()
    assert torch.allclose(mods, torch.ones_like(mods), atol=1e-5)


# ---------------------------------------------------------------------------
# Matrix-free forward
# ---------------------------------------------------------------------------


def test_forward_output_shape() -> None:
    """``forward`` returns ``[1, 1, N_pe, N_ro]`` complex."""
    builder = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    image = torch.rand(8, 8)
    b0 = torch.zeros(8, 8)
    out = builder.forward(image, b0)
    assert out.shape == (1, 1, 8, 8)
    assert out.dtype == torch.complex64


def test_forward_matches_explicit_matrix_multiply() -> None:
    """Matrix-free ``forward`` matches ``E @ m`` to within tolerance."""
    torch.manual_seed(0)
    builder = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    image = torch.randn(8, 8) + 1j * torch.randn(8, 8)
    b0 = 50.0 * torch.randn(8, 8)  # 50 Hz off-resonance

    E = builder.build_explicit(b0)
    expected = (E @ image.reshape(-1).to(torch.complex64)).reshape(8, 8)
    actual = builder.forward(image, b0).squeeze()
    assert torch.allclose(actual, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Matrix-free adjoint
# ---------------------------------------------------------------------------


def test_adjoint_output_shape() -> None:
    """``adjoint`` returns ``[1, 1, H, W]`` complex."""
    builder = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    kspace = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    b0 = torch.zeros(8, 8)
    out = builder.adjoint(kspace, b0)
    assert out.shape == (1, 1, 8, 8)
    assert out.dtype == torch.complex64


def test_adjoint_inner_product_property() -> None:
    """``<E·m, y> ≈ <m, E†·y>`` (within complex-precision tolerance)."""
    torch.manual_seed(0)
    builder = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    image = torch.randn(8, 8) + 1j * torch.randn(8, 8)
    kspace = torch.randn(1, 1, 8, 8) + 1j * torch.randn(1, 1, 8, 8)
    b0 = torch.zeros(8, 8)

    Em = builder.forward(image, b0).flatten()
    Ehy = builder.adjoint(kspace, b0).flatten()

    lhs = (Em * kspace.flatten().conj()).sum()
    rhs = (image.flatten() * Ehy.conj()).sum()

    rel_err = (lhs - rhs).abs().item() / (lhs.abs().item() + 1e-9)
    assert rel_err < 1e-3, f"adjoint test failed: rel_err = {rel_err:.3e}"


# ---------------------------------------------------------------------------
# B₀ phase modulation
# ---------------------------------------------------------------------------


def test_nonzero_b0_changes_kspace() -> None:
    """Non-zero B₀ produces a different k-space than zero B₀."""
    torch.manual_seed(0)
    builder = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    image = torch.randn(8, 8)
    out_zero = builder.forward(image, torch.zeros(8, 8))
    out_b0 = builder.forward(image, 100.0 * torch.ones(8, 8))
    diff = (out_zero - out_b0).abs().max().item()
    assert diff > 1e-3


def test_t_shift_affects_b0_phase() -> None:
    """Non-zero ``t_shift`` shifts the readout window → different output."""
    torch.manual_seed(0)
    builder = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    image = torch.randn(8, 8)
    b0 = 50.0 * torch.ones(8, 8)
    out_t0 = builder.forward(image, b0, t_shift=0.0)
    out_t1 = builder.forward(image, b0, t_shift=1e-3)
    assert not torch.allclose(out_t0, out_t1)
