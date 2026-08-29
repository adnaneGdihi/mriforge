"""Property tests for SVD-based coil compression.

Targets ``mriforge.infrastructure.physics.coil_compression``. The compression matrix
is built from the dominant eigenvectors of the coil-channel Gram matrix, so:

- For a *low-rank* synthetic k-space (signal lives in a small coil subspace),
  compressing to V coils where V >= true_rank should preserve nearly all
  energy.
- For full-rank random data, compression to V virtual coils should preserve at
  least V/C of the energy.
- The compression matrix should have orthonormal rows (V x C with VV^H = I).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.coil_compression import (
    SVDCoilCompressor,
    apply_compression_matrix,
    apply_svd_compression,
    estimate_compression_matrix,
    fit_svd_basis,
)


@pytest.fixture
def synthetic_lowrank_kspace() -> torch.Tensor:
    """Build an 8-coil k-space whose signal lives in a 3-dim coil subspace.

    Constructs ``y = U x``, where ``U`` is an 8x3 *real* mixing matrix and
    ``x`` is a 3-channel complex k-space "source". A real ``U`` ensures
    ``column_span(U) == column_span(U.conj())``, which matters because the
    SVD's top right-singular vectors span ``column_span(U.conj())``. With a
    complex ``U`` the two spans diverge and the test would (correctly)
    report only partial energy retention.
    """
    torch.manual_seed(0)
    b, h, w = 1, 32, 32
    true_rank = 3
    src = torch.randn(b, true_rank, h, w, dtype=torch.complex64)
    mixing_real = torch.randn(8, true_rank) / (8 ** 0.5)
    mixing = mixing_real.to(torch.complex64)
    return torch.einsum("vc,bchw->bvhw", mixing, src)


def _energy(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return (x.conj() * x).real.sum()
    return (x * x).sum()


def test_lowrank_compression_preserves_energy(
    synthetic_lowrank_kspace: torch.Tensor,
) -> None:
    """Compressing 8→3 should retain ≥99% of energy on a rank-3 source."""
    kspace = synthetic_lowrank_kspace
    matrix = estimate_compression_matrix(kspace, num_virtual_coils=3, acs_size=24)
    compressed = apply_compression_matrix(kspace, matrix)
    ratio = (_energy(compressed) / _energy(kspace)).item()
    assert ratio >= 0.99, f"Expected ≥99% energy retention on rank-3 source, got {ratio:.4f}"


def test_compression_matrix_has_orthonormal_rows() -> None:
    """V x C compression matrix must satisfy VV^H = I_V."""
    torch.manual_seed(1)
    kspace = torch.randn(1, 8, 24, 24, dtype=torch.complex64)
    V = 4
    matrix = estimate_compression_matrix(kspace, num_virtual_coils=V)
    gram = matrix @ matrix.conj().t()
    eye = torch.eye(V, dtype=gram.dtype, device=gram.device)
    err = (gram - eye).abs().max().item()
    assert err < 1e-4, f"Rows are not orthonormal; max ||VV^H - I|| = {err:.2e}"


def test_compression_preserves_lower_bound_for_random_input() -> None:
    """Random full-rank input: V/C virtual coils retain at least V/C of energy.

    SVD picks the top-V eigenvectors, so the retained energy fraction equals
    ``sum(top-V eigenvalues) / sum(all eigenvalues)`` ≥ V/C (uniform spectrum
    is the worst case; real eigenvalue concentration only helps).
    """
    torch.manual_seed(2)
    kspace = torch.randn(2, 8, 32, 32, dtype=torch.complex64)
    V = 4
    matrix = estimate_compression_matrix(kspace, num_virtual_coils=V, acs_size=24)
    compressed = apply_compression_matrix(kspace, matrix)
    ratio = (_energy(compressed) / _energy(kspace)).item()
    assert ratio >= V / 8 * 0.9, (
        f"Expected ≥{V/8*0.9:.2f} energy retention for V={V}/C=8, got {ratio:.4f}"
    )


def test_invalid_num_virtual_coils_raises() -> None:
    """Asking for V > C must raise ValueError."""
    kspace = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    with pytest.raises(ValueError, match="num_virtual_coils"):
        estimate_compression_matrix(kspace, num_virtual_coils=5)


def test_svd_coil_compressor_module_forward() -> None:
    """End-to-end: the nn.Module wrapper produces correctly-shaped output."""
    torch.manual_seed(3)
    kspace = torch.randn(2, 8, 32, 32, dtype=torch.complex64)
    compressor = SVDCoilCompressor(num_virtual_coils=4, acs_size=16, per_batch=True)
    out = compressor(kspace)
    assert out.shape == (2, 4, 32, 32), f"Unexpected output shape: {out.shape}"
    assert torch.is_complex(out)


# --- Plan Task 1: fit_svd_basis / apply_svd_compression primitives ---


def _multicoil(b=1, c=8, h=32, w=32):
    torch.manual_seed(0)
    return torch.complex(torch.randn(b, c, h, w), torch.randn(b, c, h, w))


def test_lossless_when_k_equals_ncoils():
    ks = _multicoil(c=8)
    basis = fit_svd_basis(ks, num_virtual_coils=8)
    out = apply_svd_compression(ks, basis)
    assert out.shape == (1, 8, 32, 32)
    # K == C is an orthonormal rotation → energy preserved
    assert torch.allclose(out.abs().pow(2).sum(), ks.abs().pow(2).sum(), rtol=1e-4)


def test_reduces_coil_count():
    ks = _multicoil(c=8)
    basis = fit_svd_basis(ks, num_virtual_coils=4)
    out = apply_svd_compression(ks, basis)
    assert out.shape == (1, 4, 32, 32)
    # virtual coils are energy-ordered: retained energy is the top-4 share
    assert out.abs().pow(2).sum() <= ks.abs().pow(2).sum() * (1 + 1e-4)


def test_k_greater_than_ncoils_raises():
    ks = _multicoil(c=4)
    with pytest.raises(ValueError, match="num_virtual_coils"):
        fit_svd_basis(ks, num_virtual_coils=8)


def test_apply_svd_compression_basis_coil_mismatch_raises():
    """A basis whose leading dim != C must raise a clear ValueError.

    Regression: previously a mismatched basis surfaced a cryptic einsum
    broadcast RuntimeError instead of a contract-violation message (mirrors
    the guard in ``apply_compression_matrix``).
    """
    ks = _multicoil(c=8)
    basis = fit_svd_basis(_multicoil(c=4), num_virtual_coils=2)  # basis.shape[0] == 4
    with pytest.raises(ValueError, match="incompatible with C=8"):
        apply_svd_compression(ks, basis)


def test_fit_and_apply_real_interleaved_layout():
    """Real-interleaved (B, 2*C, H, W) input is routed through _to_complex.

    Regression: previously the 2*C real channels were silently treated as
    2*C independent real coils, mixing real and imaginary parts and returning
    a physically wrong result. The functions must instead detect the layout
    via ``torch.is_complex`` (consistent with the rest of the module) and
    produce the SAME virtual coils as the equivalent complex input, returned
    in real-interleaved layout.
    """
    torch.manual_seed(7)
    c = 8
    complex_ks = torch.complex(
        torch.randn(1, c, 16, 16), torch.randn(1, c, 16, 16)
    )
    # Interleaved real layout: [real_0, imag_0, real_1, imag_1, ...] → (1, 2C, H, W).
    interleaved = torch.stack(
        [complex_ks.real, complex_ks.imag], dim=2
    ).flatten(1, 2)
    assert interleaved.shape == (1, 2 * c, 16, 16)
    assert not torch.is_complex(interleaved)

    k = 4
    basis_real = fit_svd_basis(interleaved, num_virtual_coils=k)
    basis_complex = fit_svd_basis(complex_ks, num_virtual_coils=k)
    # The detected complex layout must yield the same C-dim basis, not a 2C one.
    assert basis_real.shape == (c, k)
    assert torch.is_complex(basis_real)
    assert torch.allclose(basis_real, basis_complex, atol=1e-5)

    out_real = apply_svd_compression(interleaved, basis_real)
    out_complex = apply_svd_compression(complex_ks, basis_complex)
    # Real-interleaved input → real-interleaved output of width 2K.
    assert out_real.shape == (1, 2 * k, 16, 16)
    assert not torch.is_complex(out_real)
    # And it must equal the complex result, just re-interleaved.
    out_real_as_complex = torch.complex(out_real[:, 0::2], out_real[:, 1::2])
    assert torch.allclose(out_real_as_complex, out_complex, atol=1e-5)


def test_fit_svd_basis_does_not_cancel_across_batch_phase() -> None:
    """Batch samples with opposite global phase must not cancel the coil basis.

    Regression (2026-07): ``fit_svd_basis`` averaged k-space over the batch
    before the SVD, so two samples carrying opposite phase (``+field`` and
    ``-field`` of the same rank-1 coil signal) summed to ~zero and the recovered
    basis was arbitrary. Concatenating samples along the sample axis preserves
    the dominant coil direction ``v``.
    """
    torch.manual_seed(0)
    c, h, w = 6, 8, 8
    v = torch.complex(torch.randn(c), torch.randn(c))
    v = v / v.norm()  # unit coil direction
    field = torch.complex(torch.randn(h, w), torch.randn(h, w))
    s0 = v.view(c, 1, 1) * field.view(1, h, w)
    ksp = torch.stack([s0, -s0], dim=0)  # [2, C, H, W]; batch mean == 0

    basis = fit_svd_basis(ksp, num_virtual_coils=1)  # (C, 1)
    align = (v.conj() * basis[:, 0]).sum().abs()  # |<v, u0>|, ≈1 if aligned
    assert align.item() > 0.99
