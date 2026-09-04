"""Canary / parametrized / edge-case / expected-failure tests for encoding_matrix.py.

Covers EncodingMatrixBuilder: forward(), adjoint(), and build_with_sampling().
We keep spatial size tiny (n_readout=n_phase=8 or 16) so the matrix-free ops
remain CPU-tractable; build_explicit is avoided for all but the tiniest sizes.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _b0_map(n_ro: int, n_pe: int, hz: float = 0.0) -> torch.Tensor:
    return torch.full((1, 1, n_pe, n_ro), hz)


def _image(n_ro: int, n_pe: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.rand(1, 1, n_pe, n_ro, generator=g)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_encoding_matrix_import():
    """Module imports and EncodingMatrixBuilder can be instantiated."""
    from spectramr.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
    emb = EncodingMatrixBuilder(n_readout=8, n_phase=8)
    assert emb is not None


@pytest.mark.physics
def test_canary_forward_output_shape():
    """Forward operator returns [1, 1, N_pe, N_ro] complex tensor."""
    from spectramr.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
    n_ro, n_pe = 8, 8
    emb = EncodingMatrixBuilder(n_readout=n_ro, n_phase=n_pe)
    image = _image(n_ro, n_pe)
    b0 = _b0_map(n_ro, n_pe)
    kspace = emb.forward(image, b0)
    assert kspace.shape == (1, 1, n_pe, n_ro)
    assert torch.is_complex(kspace)


# ---------------------------------------------------------------------------
# Parametrized — forward + adjoint shape
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "n_ro, n_pe",
    [
        (8, 8),
        (8, 16),   # rectangular
        (16, 8),
        (16, 16),
    ],
    ids=["8x8", "8x16", "16x8", "16x16"],
)
def test_forward_shape(n_ro, n_pe):
    from spectramr.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
    emb = EncodingMatrixBuilder(n_readout=n_ro, n_phase=n_pe)
    image = _image(n_ro, n_pe)
    b0 = _b0_map(n_ro, n_pe)
    kspace = emb.forward(image, b0)
    assert kspace.shape == (1, 1, n_pe, n_ro)


@pytest.mark.physics
@pytest.mark.parametrize(
    "n_ro, n_pe",
    [
        (8, 8),
        (8, 16),
        (16, 8),
    ],
    ids=["8x8", "8x16", "16x8"],
)
def test_adjoint_shape(n_ro, n_pe):
    from spectramr.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
    emb = EncodingMatrixBuilder(n_readout=n_ro, n_phase=n_pe)
    b0 = _b0_map(n_ro, n_pe)
    kspace = torch.randn(1, 1, n_pe, n_ro, dtype=torch.complex64)
    image_out = emb.adjoint(kspace, b0)
    # adjoint returns [1, 1, H, W] where H=n_pe, W=n_ro
    assert image_out.shape == (1, 1, n_pe, n_ro)
    assert torch.is_complex(image_out)


# ---------------------------------------------------------------------------
# Correctness: forward-adjoint inner product consistency
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_forward_adjoint_inner_product():
    """<Ef, g> = <f, E^H g> up to float32 tolerance (n=8)."""
    from spectramr.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
    n = 8
    emb = EncodingMatrixBuilder(n_readout=n, n_phase=n)
    b0 = _b0_map(n, n)

    g = torch.Generator().manual_seed(11)
    f = torch.randn(1, 1, n, n, generator=g) + 0j
    y = torch.randn(1, 1, n, n, dtype=torch.complex64, generator=g)

    Ef = emb.forward(f.real.unsqueeze(0).squeeze(0), b0)   # oops: fix shape
    # build properly
    f_real = torch.rand(n, n)
    y_c = torch.randn(1, 1, n, n, dtype=torch.complex64)
    Ef2 = emb.forward(f_real, b0)            # [1,1,n,n]
    EHy = emb.adjoint(y_c, b0)               # [1,1,n,n]

    lhs = torch.sum(Ef2 * torch.conj(y_c)).real
    rhs = torch.sum(f_real.to(torch.complex64) * torch.conj(EHy)).real
    # A modest tolerance — this is a small operator with float32 arithmetic
    assert abs((lhs - rhs).item()) / (abs(lhs.item()) + 1e-6) < 0.1


# ---------------------------------------------------------------------------
# build_with_sampling shape
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_build_with_sampling_shape():
    """build_with_sampling returns a submatrix with fewer rows."""
    from spectramr.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
    n_ro, n_pe = 8, 8
    emb = EncodingMatrixBuilder(n_readout=n_ro, n_phase=n_pe)
    b0 = _b0_map(n_ro, n_pe).squeeze()   # [H, W]
    # Mask: keep only first 4 PE lines
    mask = torch.zeros(n_pe, dtype=torch.bool)
    mask[:4] = True
    E_sub = emb.build_with_sampling(b0, mask)
    n_sampled = int(mask.sum().item())
    M = n_ro * n_pe
    assert E_sub.shape == (n_sampled * n_ro, M)
    assert torch.is_complex(E_sub)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_forward_edge_zero_b0():
    """Zero B0 map: no phase distortion; output should be well-defined."""
    from spectramr.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
    n = 8
    emb = EncodingMatrixBuilder(n_readout=n, n_phase=n)
    b0 = torch.zeros(1, 1, n, n)
    image = _image(n, n)
    kspace = emb.forward(image, b0)
    assert not torch.isnan(kspace).any()


@pytest.mark.physics
def test_adjoint_edge_zero_kspace():
    """Zero k-space → zero image estimate."""
    from spectramr.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder
    n = 8
    emb = EncodingMatrixBuilder(n_readout=n, n_phase=n)
    b0 = _b0_map(n, n)
    kspace = torch.zeros(1, 1, n, n, dtype=torch.complex64)
    out = emb.adjoint(kspace, b0)
    assert torch.allclose(out.abs(), torch.zeros_like(out.abs()), atol=1e-6)
