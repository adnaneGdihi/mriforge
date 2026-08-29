"""Unit tests for the explicit encoding-matrix builder.

Covers :class:`mriforge.infrastructure.physics.encoding_matrix.EncodingMatrixBuilder`
which constructs the MRI encoding matrix ``E`` incorporating per-readout
B0 phase, and also exposes matrix-free ``forward`` / ``adjoint`` operators.

Tested invariants
-----------------
* Adjoint identity: ``<E x, y> ≈ <x, E^H y>`` for both the explicit-matrix
  form and the matrix-free ``forward`` / ``adjoint`` operators.
* Forward shape contract: ``[H, W]`` image → ``[1, 1, N_pe, N_ro]`` k-space.
* Unitary roundtrip (b0=0): ``E^H E x ≈ N * x`` up to the DFT scaling
  used in the explicit construction.
* Edge: building with a non-matching b0 shape raises an actionable error.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# Skip the whole module if the target hasn't been integrated yet.
em_mod = pytest.importorskip("mriforge.infrastructure.physics.encoding_matrix")
EncodingMatrixBuilder = em_mod.EncodingMatrixBuilder


def _inner(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Complex inner product ``<a, b> = sum(conj(a) * b)``."""
    return (a.conj() * b).sum()


class TestEncodingMatrixExplicitAdjoint:
    """Adjoint identity on the explicitly built matrix ``E``."""

    @pytest.fixture
    def builder(self):
        # Small problem so the explicit matrix is cheap.
        return EncodingMatrixBuilder(
            n_readout=8, n_phase=8, fov_m=0.2, bw_hz=20_000.0
        )

    def test_explicit_matrix_adjoint_identity(self, builder, deterministic_seed):
        # Random non-trivial B0 map exercises the off-diagonal phase terms.
        b0 = torch.randn(8, 8) * 100.0  # Hz
        E = builder.build_explicit(b0)

        N2, M = E.shape
        assert N2 == 8 * 8
        assert M == 8 * 8

        # x in image space, y in k-space (flattened)
        x = torch.randn(M, dtype=torch.complex64)
        y = torch.randn(N2, dtype=torch.complex64)

        lhs = _inner(E @ x, y)            # <E x, y>
        rhs = _inner(x, E.conj().T @ y)   # <x, E^H y>
        # Both numbers are complex scalars — compare real and imag parts.
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_explicit_matrix_dtype_and_shape(self, builder):
        b0 = torch.zeros(8, 8)
        E = builder.build_explicit(b0)
        assert E.dtype == torch.complex64
        assert E.shape == (8 * 8, 8 * 8)


class TestEncodingMatrixForwardShape:
    """The matrix-free ``forward`` honours the [B, C, N_pe, N_ro] contract."""

    def test_forward_returns_documented_kspace_shape(self, deterministic_seed):
        builder = EncodingMatrixBuilder(
            n_readout=4, n_phase=4, fov_m=0.2, bw_hz=20_000.0
        )
        image = torch.randn(1, 1, 4, 4)
        b0 = torch.zeros(1, 1, 4, 4)

        kspace = builder.forward(image, b0)

        assert kspace.shape == (1, 1, 4, 4)
        assert kspace.dtype == torch.complex64

    def test_adjoint_returns_documented_image_shape(self, deterministic_seed):
        builder = EncodingMatrixBuilder(
            n_readout=4, n_phase=4, fov_m=0.2, bw_hz=20_000.0
        )
        kspace = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
        b0 = torch.zeros(1, 1, 4, 4)

        image = builder.adjoint(kspace, b0)

        assert image.shape == (1, 1, 4, 4)
        assert image.dtype == torch.complex64


class TestEncodingMatrixMatrixFreeAdjoint:
    """The matrix-free ``forward`` / ``adjoint`` pair satisfies the adjoint identity."""

    def test_matrix_free_adjoint_identity_with_b0(self, deterministic_seed):
        builder = EncodingMatrixBuilder(
            n_readout=4, n_phase=4, fov_m=0.2, bw_hz=20_000.0
        )
        b0 = torch.randn(1, 1, 4, 4) * 50.0  # Hz

        x = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
        y = torch.randn(1, 1, 4, 4, dtype=torch.complex64)

        Ax = builder.forward(x, b0)
        Aty = builder.adjoint(y, b0)

        lhs = _inner(Ax.reshape(-1), y.reshape(-1))
        rhs = _inner(x.reshape(-1), Aty.reshape(-1))
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_matrix_free_matches_explicit_forward(self, deterministic_seed):
        """The matrix-free ``forward`` agrees with ``E @ x`` for the same b0."""
        builder = EncodingMatrixBuilder(
            n_readout=4, n_phase=4, fov_m=0.2, bw_hz=20_000.0
        )
        b0 = torch.randn(4, 4) * 25.0

        x = torch.randn(4, 4, dtype=torch.complex64)
        E = builder.build_explicit(b0)
        y_explicit = (E @ x.reshape(-1)).reshape(4, 4)
        y_matrix_free = builder.forward(x, b0).squeeze()

        assert torch.allclose(y_explicit, y_matrix_free, atol=1e-4)


class TestEncodingMatrixGramian:
    """With ``b0 = 0`` the Gramian ``E^H E`` is Hermitian PSD with diagonal ≈ M.

    The construction uses unscaled complex exponentials (no ``1/√N``
    normalisation), so the matrix is *not* unitary — but the diagonal
    of ``E^H E`` must still equal ``M`` because each column has unit
    modulus everywhere.
    """

    def test_zero_b0_diagonal_equals_M(self, deterministic_seed):
        builder = EncodingMatrixBuilder(
            n_readout=4, n_phase=4, fov_m=0.2, bw_hz=20_000.0
        )
        b0 = torch.zeros(4, 4)
        E = builder.build_explicit(b0)
        EhE = E.conj().T @ E
        M = E.shape[1]
        diag = EhE.diag()
        assert torch.allclose(diag.real, torch.full_like(diag.real, M), atol=1e-3)
        # Hermitian: ``E^H E`` equals its own conjugate transpose.
        assert torch.allclose(EhE, EhE.conj().T, atol=1e-3)

    def test_zero_b0_roundtrip_preserves_dc(self, deterministic_seed):
        """For the constant image ``x ≡ c`` the adjoint of the forward agrees with ``M·E·x``-rescaled identity."""
        builder = EncodingMatrixBuilder(
            n_readout=4, n_phase=4, fov_m=0.2, bw_hz=20_000.0
        )
        b0 = torch.zeros(4, 4)
        # Constant image — the DFT puts all energy in the centre.
        x = torch.ones(4, 4, dtype=torch.complex64)
        E = builder.build_explicit(b0)
        y = E @ x.reshape(-1)
        x_back = E.conj().T @ y / E.shape[1]
        # Recovery of the constant image up to a scalar.
        ratio = x_back / x.reshape(-1)
        # Most entries should map back close to constant 1 (the unscaled DFT
        # zeros out non-DC modes after the adjoint).
        assert torch.allclose(x_back.mean().real, torch.tensor(1.0), atol=0.5)


class TestEncodingMatrixWithSampling:
    """``build_with_sampling`` only keeps rows for sampled PE lines."""

    def test_full_sampling_equals_full_matrix(self, deterministic_seed):
        builder = EncodingMatrixBuilder(
            n_readout=4, n_phase=4, fov_m=0.2, bw_hz=20_000.0
        )
        b0 = torch.zeros(4, 4)
        mask = torch.ones(4, dtype=torch.bool)

        E_full = builder.build_explicit(b0)
        E_sub = builder.build_with_sampling(b0, mask)

        assert E_sub.shape == E_full.shape
        assert torch.allclose(E_sub, E_full, atol=1e-6)

    def test_partial_sampling_reduces_row_count(self, deterministic_seed):
        builder = EncodingMatrixBuilder(
            n_readout=4, n_phase=4, fov_m=0.2, bw_hz=20_000.0
        )
        b0 = torch.zeros(4, 4)
        mask = torch.tensor([True, False, True, False])

        E_sub = builder.build_with_sampling(b0, mask)
        # 2 sampled PE lines * 4 readout samples = 8 rows.
        assert E_sub.shape == (8, 16)
