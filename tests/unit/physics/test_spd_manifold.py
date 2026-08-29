"""Regression tests for the affine-invariant SPD manifold.

Covers the doc-contract clarification on
:meth:`mriforge.infrastructure.physics.manifolds.spd_manifold.SPDManifold.metric_tensor`:
it returns the ``[..., n, n]`` affine-invariant metric *factor* ``C^{-1}``,
**not** the protocol's ``[..., dim, dim]`` (``dim = n(n+1)/2``) Gram matrix. The
fix adds explicit :meth:`inner_product` / :meth:`squared_norm` helpers that
realise the affine-invariant inner product ``<A, B>_C = tr(C^{-1} A C^{-1} B)``,
so callers no longer have to (mis)interpret ``metric_tensor`` as a Gram matrix.
These tests pin the corrected contract.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.physics.manifolds.spd_manifold import SPDManifold


def _batched_spd(batch: int, n: int, *, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    X = torch.randn(batch, n, n)
    return X @ X.transpose(-1, -2) + n * torch.eye(n)


def test_metric_tensor_returns_nxn_factor_not_gram_matrix() -> None:
    """``metric_tensor`` returns the ``n x n`` factor ``C^{-1}`` (not ``dim x dim``)."""
    M = SPDManifold(n=4)
    C = _batched_spd(3, 4)
    G = M.metric_tensor(C)
    # The intrinsic dimension differs from n, so the protocol Gram shape would
    # have been (dim, dim) -- this returns (n, n) instead.
    assert M.dim != M.n
    assert G.shape[-2:] == (M.n, M.n)
    assert torch.allclose(G, torch.linalg.inv(C), atol=1e-5)


def test_inner_product_matches_trace_formula() -> None:
    """``inner_product`` equals ``tr(C^{-1} A C^{-1} B)``."""
    M = SPDManifold(n=4)
    C = _batched_spd(3, 4, seed=0)
    torch.manual_seed(7)
    A = torch.randn(3, 4, 4)
    A = 0.5 * (A + A.transpose(-1, -2))
    B = torch.randn(3, 4, 4)
    B = 0.5 * (B + B.transpose(-1, -2))

    Cinv = torch.linalg.inv(C)
    expected = (Cinv @ A @ Cinv @ B).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    got = M.inner_product(C, A, B)
    assert got.shape == (3,)
    assert torch.allclose(got, expected, atol=1e-5)


def test_inner_product_is_symmetric() -> None:
    """The affine-invariant inner product is symmetric in A, B."""
    M = SPDManifold(n=3)
    C = _batched_spd(2, 3, seed=1)
    torch.manual_seed(11)
    A = torch.randn(2, 3, 3)
    A = 0.5 * (A + A.transpose(-1, -2))
    B = torch.randn(2, 3, 3)
    B = 0.5 * (B + B.transpose(-1, -2))
    assert torch.allclose(M.inner_product(C, A, B), M.inner_product(C, B, A), atol=1e-5)


def test_squared_norm_matches_inner_product_self_and_is_nonneg() -> None:
    """``squared_norm(C, A) == inner_product(C, A, A)`` and is non-negative (PD metric)."""
    M = SPDManifold(n=4)
    C = _batched_spd(3, 4, seed=2)
    torch.manual_seed(13)
    A = torch.randn(3, 4, 4)
    A = 0.5 * (A + A.transpose(-1, -2))
    sn = M.squared_norm(C, A)
    assert torch.allclose(sn, M.inner_product(C, A, A), atol=1e-6)
    # Affine-invariant metric is positive-definite -> norm >= 0.
    assert torch.all(sn >= -1e-6)


def test_affine_invariance_of_inner_product() -> None:
    """<A,B>_C is invariant under congruence: g.A.g^T at g.C.g^T equals <A,B>_C."""
    M = SPDManifold(n=3)
    C = _batched_spd(2, 3, seed=3)
    torch.manual_seed(17)
    A = torch.randn(2, 3, 3)
    A = 0.5 * (A + A.transpose(-1, -2))
    B = torch.randn(2, 3, 3)
    B = 0.5 * (B + B.transpose(-1, -2))
    g = torch.randn(2, 3, 3)  # GL(n) congruence
    gt = g.transpose(-1, -2)
    base = M.inner_product(C, A, B)
    transformed = M.inner_product(g @ C @ gt, g @ A @ gt, g @ B @ gt)
    assert torch.allclose(base, transformed, atol=1e-4)
