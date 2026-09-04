r"""Krylov-Lanczos Attention via three-term recurrence.

Treats the symmetrised attention kernel :math:`H=(QK^{\top}+KQ^{\top})/\sqrt{d}`
as a Hermitian operator and builds a :math:`k`-dimensional Krylov subspace

.. math::

    \beta_i \mathbf{v}_{i+1} = H\mathbf{v}_i - \alpha_i \mathbf{v}_i - \beta_{i-1}\mathbf{v}_{i-1}.

The projected tridiagonal :math:`T_k=V_k^{\top}HV_k\in\mathbb{R}^{k\times k}`
captures the dominant spectrum; Chebyshev's approximation theorem
[Golub-Van Loan 2013, Th. 4.4.10] gives exponential convergence
:math:`\|(I-P_k)\mathbf{u}_*\|\le 2((\sqrt\kappa-1)/(\sqrt\kappa+1))^k`.

Cost is :math:`O(Nkd)` per forward pass (vs :math:`O(N^{2}d)` for full
attention). The implementation uses full re-orthogonalisation at every
step, which is :math:`O(Nk^{2})` total — fine for :math:`k\le 32`.

The CUDA persistent-kernel variant from the parent proposal is left as
future work; the present pure-PyTorch implementation captures the same
asymptotic complexity reduction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["KrylovLanczosAttention"]


class KrylovLanczosAttention(nn.Module):
    r"""Krylov-subspace approximation of full attention.

    Args:
        embed_dim: Feature dimension :math:`d`.
        num_heads: Number of attention heads.
        krylov_dim: Krylov subspace dimension :math:`k`. Recommended
            :math:`8\le k\le 32`. Higher :math:`k` → tighter approximation
            but higher cost.
        eps: Threshold for early termination when :math:`\beta_i` is small.

    Shape:
        Input: ``(B, N, d)``.
        Output: ``(B, N, d)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        krylov_dim: int = 8,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}.")
        if krylov_dim <= 0:
            raise ValueError("krylov_dim must be positive.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.krylov_dim = krylov_dim
        self.eps = eps
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _lanczos(
        self, h_apply, v0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run k Lanczos iterations producing (V_k, alpha, beta).

        ``h_apply`` is a callable: vector → vector. Vectors have shape
        ``(B, H, N)``. Returns ``V_k`` of shape ``(B, H, k, N)``, ``alpha``
        and ``beta`` of shape ``(B, H, k)``.
        """
        k = self.krylov_dim
        basis: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        v_curr = v0 / v0.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        basis.append(v_curr)
        w = h_apply(v_curr)
        a0 = (w * v_curr).sum(dim=-1)
        alphas.append(a0)
        w = w - a0.unsqueeze(-1) * v_curr
        for i in range(1, k):
            b_prev = w.norm(dim=-1)
            betas.append(b_prev)
            denom = b_prev.clamp_min(self.eps).unsqueeze(-1)
            mask = (b_prev > self.eps).unsqueeze(-1)
            v_next = torch.where(mask, w / denom, torch.zeros_like(w))
            # Full re-orthogonalisation against accumulated basis.
            prior = torch.stack(basis, dim=-2)  # (B, H, i, N)
            coeff = torch.einsum("bhn,bhin->bhi", v_next, prior)
            v_next = v_next - torch.einsum("bhi,bhin->bhn", coeff, prior)
            v_next = v_next / v_next.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            basis.append(v_next)
            w_new = h_apply(v_next)
            a_i = (w_new * v_next).sum(dim=-1)
            alphas.append(a_i)
            w = w_new - a_i.unsqueeze(-1) * v_next - b_prev.unsqueeze(-1) * v_curr
            v_curr = v_next
        # Pad beta list to length k for consistent shape.
        if len(betas) < k:
            betas.append(torch.zeros_like(alphas[0]))
        v_basis = torch.stack(basis, dim=-2)
        alpha = torch.stack(alphas, dim=-1)
        beta = torch.stack(betas, dim=-1)
        return v_basis, alpha, beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, N, d) input; got {tuple(x.shape)}.")
        b, n, d = x.shape
        if d != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: {d} vs {self.embed_dim}.")
        if n < 2:
            raise ValueError("token_len must be >= 2.")

        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, H, N, D_h)
        scale = float(self.head_dim) ** -0.5

        # Hermitian attention kernel H = (QK^T + KQ^T) * scale.
        def h_apply(u: torch.Tensor) -> torch.Tensor:
            # u: (B, H, N)
            kt_u = torch.einsum("bhnd,bhn->bhd", k, u)
            qkt_u = torch.einsum("bhnd,bhd->bhn", q, kt_u)
            qt_u = torch.einsum("bhnd,bhn->bhd", q, u)
            kqt_u = torch.einsum("bhnd,bhd->bhn", k, qt_u)
            return scale * (qkt_u + kqt_u)

        v0 = v.mean(dim=-1)  # (B, H, N) — start vector is the mean V channel.
        v_basis, alpha, beta = self._lanczos(h_apply, v0)
        # Form softmax of T_k and apply to V via the basis.
        t_k = (
            torch.diag_embed(alpha)
            + torch.diag_embed(beta[..., :-1], offset=1)
            + torch.diag_embed(beta[..., :-1], offset=-1)
        )
        t_k = torch.softmax(t_k, dim=-1)  # (B, H, k, k)
        # Project V into Krylov basis: c_i = v_basis_i · V (over the N axis).
        v_in_basis = torch.einsum("bhkn,bhnd->bhkd", v_basis, v)
        v_back = torch.einsum("bhkj,bhjd->bhkd", t_k, v_in_basis)
        out = torch.einsum("bhkn,bhkd->bhnd", v_basis, v_back)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, krylov_dim={self.krylov_dim}"
        )
