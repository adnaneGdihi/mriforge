"""SPD manifold with affine-invariant metric (fMRI §4 — dynamic FC).

Implements the affine-invariant Riemannian structure on
:math:`\\mathrm{Sym}^+(n)`, the closed-form exponential and log maps,
and the Cayley conformal compactification used to handle the
boundary numerically.

References:
    [18] X. Pennec et al., "A Riemannian framework for tensor
         computing", *Int. J. Comput. Vis.*, 66(1), 2006, 41-66.
    [20] R. Bhatia, *Positive Definite Matrices*, Princeton, 2007.
    [22] J. Faraut & A. Korányi, *Analysis on Symmetric Cones*,
         Oxford, 1994.
"""

from __future__ import annotations

import torch


def _sym_matrix_log(C: torch.Tensor) -> torch.Tensor:
    """Matrix log of an SPD matrix via eigendecomposition."""
    eigvals, eigvecs = torch.linalg.eigh(C)
    log_eig = eigvals.clamp_min(1e-8).log()
    return eigvecs @ torch.diag_embed(log_eig) @ eigvecs.transpose(-1, -2)


def _sym_matrix_exp(S: torch.Tensor) -> torch.Tensor:
    """Matrix exp of a symmetric matrix."""
    eigvals, eigvecs = torch.linalg.eigh(S)
    exp_eig = eigvals.exp()
    return eigvecs @ torch.diag_embed(exp_eig) @ eigvecs.transpose(-1, -2)


def _sym_matrix_pow(C: torch.Tensor, p: float) -> torch.Tensor:
    """Symmetric matrix power for SPD matrices."""
    eigvals, eigvecs = torch.linalg.eigh(C)
    pow_eig = eigvals.clamp_min(1e-8).pow(p)
    return eigvecs @ torch.diag_embed(pow_eig) @ eigvecs.transpose(-1, -2)


class SPDManifold(torch.nn.Module):
    """Affine-invariant SPD manifold ``Sym^+(n)``.

    Operates on batched SPD matrices of shape ``[..., n, n]``.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.dim = n * (n + 1) // 2
        self.n = n

    def exp_map(self, C: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Affine-invariant geodesic exp at base point ``C``.

        :math:`\\exp_C(V) = C^{1/2}\\,\\mathrm{exp}_M(C^{-1/2} V C^{-1/2})\\,C^{1/2}`.
        """
        sqrt_C = _sym_matrix_pow(C, 0.5)
        inv_sqrt_C = _sym_matrix_pow(C, -0.5)
        inner = inv_sqrt_C @ V @ inv_sqrt_C
        return sqrt_C @ _sym_matrix_exp(inner) @ sqrt_C

    def log_map(self, C0: torch.Tensor, C1: torch.Tensor) -> torch.Tensor:
        """Affine-invariant geodesic log.

        :math:`\\log_{C_0}(C_1) = C_0^{1/2}\\,\\log_M(C_0^{-1/2} C_1 C_0^{-1/2})\\,C_0^{1/2}`.
        """
        sqrt_C = _sym_matrix_pow(C0, 0.5)
        inv_sqrt_C = _sym_matrix_pow(C0, -0.5)
        inner = inv_sqrt_C @ C1 @ inv_sqrt_C
        return sqrt_C @ _sym_matrix_log(inner) @ sqrt_C

    def geodesic_distance(self, C0: torch.Tensor, C1: torch.Tensor) -> torch.Tensor:
        """Affine-invariant geodesic distance ``‖log_M(C0^{-1/2} C1 C0^{-1/2})‖_F``."""
        inv_sqrt_C = _sym_matrix_pow(C0, -0.5)
        inner = inv_sqrt_C @ C1 @ inv_sqrt_C
        return _sym_matrix_log(inner).pow(2).sum(dim=(-1, -2)).sqrt()

    def project_tangent(self, C: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Tangent space at C is the symmetric matrices — symmetrise V."""
        return 0.5 * (V + V.transpose(-1, -2))

    def metric_tensor(self, C: torch.Tensor) -> torch.Tensor:
        """Return the affine-invariant metric *factor* ``C^{-1}`` of shape ``[..., n, n]``.

        .. note::
            This **does not** follow the ``RiemannianManifold`` protocol's
            ``[..., dim, dim]`` Gram-matrix convention (here ``dim = n(n+1)/2``).
            On matrix manifolds the affine-invariant inner product factorises as
            ``⟨A, B⟩_C = tr(C^{-1} A C^{-1} B)``, so the natural object is the
            ``n×n`` factor ``C^{-1}`` (equivalently the Kronecker factor of
            ``C^{-1} ⊗ C^{-1}``), **not** a dense ``dim×dim`` Gram matrix. Callers
            must therefore use :meth:`inner_product`/:meth:`squared_norm` rather
            than treating the return value as a ``[dim, dim]`` Gram matrix.
        """
        return torch.linalg.inv(C)

    def inner_product(self, C: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Affine-invariant inner product ``⟨A, B⟩_C = tr(C^{-1} A C^{-1} B)``.

        Args:
            C: SPD base point, shape ``[..., n, n]``.
            A, B: Symmetric tangent vectors at ``C``, shape ``[..., n, n]``.

        Returns:
            Scalar inner products, shape ``[...]``.
        """
        Cinv = torch.linalg.inv(C)
        M = Cinv @ A @ Cinv @ B
        return M.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    def squared_norm(self, C: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Affine-invariant squared norm ``‖A‖_C^2 = tr((C^{-1} A)^2)``."""
        return self.inner_product(C, A, A)


def cayley_transform(C: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    """Cayley transform from SPD cone to the unit ball of symmetric matrices.

    :math:`\\Psi(C) = (C - I)(C + I)^{-1}`.
    Conformal at the identity; provides a bounded chart for numerical
    integration near the boundary of the cone.
    """
    n = C.shape[-1]
    I = torch.eye(n, dtype=C.dtype, device=C.device).expand_as(C)
    return (C - I) @ torch.linalg.inv(C + I + eps * I)


def inverse_cayley_transform(W: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    """Inverse of :func:`cayley_transform`."""
    n = W.shape[-1]
    I = torch.eye(n, dtype=W.dtype, device=W.device).expand_as(W)
    return (I + W) @ torch.linalg.inv(I - W + eps * I)


__all__ = [
    "SPDManifold",
    "cayley_transform",
    "inverse_cayley_transform",
]
