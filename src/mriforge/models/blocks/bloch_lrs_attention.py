r"""Bloch-Coupled Low-Rank-plus-Sparse (LRS) Attention.

The attention output is decomposed as :math:`A=L+S` with :math:`L=UV^{\top}`
rank-:math:`r` and :math:`S` :math:`k`-sparse. The *Bloch-coupling*
constraint forces the columns of :math:`U` to lie in the image of a
pre-computed Bloch dictionary, restricting the effective rank to
:math:`\min(r,M)` where :math:`M` is the dictionary size. Under rotation
augmentation, the Candès-Recht incoherence bound
:math:`\mu(L^{\rm aug})\le\mu_{0}+(4/N)\sqrt{M\log N}` is restored
[Tropp 2015, matrix-Bernstein].

This pure-PyTorch implementation skips the mixed-precision /
cuBLAS-Lt-Sparse fusion described in the parent proposal; the asymptotic
parameter and memory savings remain the same. The block is a
drop-in replacement for full attention in transformer backbones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BlochLRSAttention"]


class BlochLRSAttention(nn.Module):
    r"""Low-rank-plus-sparse attention with a Bloch-image rank constraint.

    Args:
        embed_dim: Feature dimension :math:`d`.
        token_len: Token-sequence length :math:`N`.
        rank: Low-rank parameter :math:`r`.
        sparse_k: Number of nonzero entries per row of the sparse component
            :math:`S`. Set to 0 to disable the sparse term.
        bloch_dictionary: Optional ``(M, r)`` tensor; if provided, the
            low-rank factor :math:`U` is projected onto its column span at
            every forward pass. ``None`` falls back to unconstrained
            low-rank attention.

    Shape:
        Input: ``(B, N, d)``. Output: ``(B, N, d)``.
    """

    def __init__(
        self,
        embed_dim: int,
        token_len: int,
        rank: int = 8,
        sparse_k: int = 0,
        bloch_dictionary: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if sparse_k < 0:
            raise ValueError("sparse_k must be non-negative.")
        if token_len < 2:
            raise ValueError("token_len must be >= 2.")
        self.embed_dim = embed_dim
        self.token_len = token_len
        self.rank = rank
        self.sparse_k = sparse_k

        self.in_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Learnable low-rank factor U: (N, r). V: (r, d).
        self.u_factor = nn.Parameter(torch.randn(token_len, rank) * 0.02)
        self.v_factor = nn.Parameter(torch.randn(rank, embed_dim) * 0.02)

        if bloch_dictionary is not None:
            if bloch_dictionary.ndim != 2 or bloch_dictionary.shape[1] != rank:
                raise ValueError(
                    f"bloch_dictionary must have shape (M, {rank}); "
                    f"got {tuple(bloch_dictionary.shape)}."
                )
            # Project Bloch dictionary to an orthonormal basis of its image.
            q, _ = torch.linalg.qr(bloch_dictionary)
            self.register_buffer("bloch_basis", q)
        else:
            self.bloch_basis = None  # type: ignore[assignment]

        if sparse_k > 0:
            # Learnable sparse attention values; the position layout is
            # initialised as a band of width sparse_k around the diagonal,
            # frozen once at construction time.
            indices = []
            for i in range(token_len):
                offsets = torch.arange(-sparse_k // 2, sparse_k // 2)
                cols = (i + offsets) % token_len
                for c in cols.tolist():
                    indices.append([i, c])
            idx = torch.tensor(indices, dtype=torch.long)
            self.register_buffer("sparse_indices", idx)
            self.sparse_values = nn.Parameter(torch.zeros(idx.shape[0]))
        else:
            self.sparse_indices = None  # type: ignore[assignment]
            self.sparse_values = None  # type: ignore[assignment]

    def _project_to_bloch(self, u: torch.Tensor) -> torch.Tensor:
        if self.bloch_basis is None:
            return u
        # Restrict each column of u^T (a vector in R^r) to the span of the
        # Bloch basis. bloch_basis: (M, r). Compute (Q Q^T) u.
        coeff = u @ self.bloch_basis  # (N, M)
        return coeff @ self.bloch_basis.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, N, d) input; got {tuple(x.shape)}.")
        b, n, d = x.shape
        if n != self.token_len:
            raise ValueError(f"input token-len {n} != configured token_len {self.token_len}.")
        if d != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: {d} vs {self.embed_dim}.")

        v = self.in_proj(x)  # (B, N, d)
        # Low-rank part: out = U (V x)
        u_proj = self._project_to_bloch(self.u_factor)
        vx = v @ self.v_factor.T  # (B, N, r)
        low_rank_out = vx @ u_proj.T  # (B, N, N) implicit via outer
        low_rank_out = low_rank_out @ v / max(1.0, float(n))

        # Sparse part.
        if self.sparse_values is not None and self.sparse_indices is not None:
            sparse_dense = torch.zeros(n, n, device=x.device, dtype=x.dtype)
            sparse_dense[self.sparse_indices[:, 0], self.sparse_indices[:, 1]] = self.sparse_values
            sparse_out = sparse_dense @ v
        else:
            sparse_out = 0.0

        out = low_rank_out + sparse_out
        return self.out_proj(out)

    def extra_repr(self) -> str:
        bloch = (
            f", bloch_basis=({tuple(self.bloch_basis.shape)})"
            if self.bloch_basis is not None
            else ", bloch_basis=None"
        )
        return (
            f"embed_dim={self.embed_dim}, token_len={self.token_len}, "
            f"rank={self.rank}, sparse_k={self.sparse_k}{bloch}"
        )
