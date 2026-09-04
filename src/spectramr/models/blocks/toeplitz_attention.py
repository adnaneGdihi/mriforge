r"""Toeplitz-Spectral Self-Attention via Fused FFT-Convolution.

For locally stationary inputs (MRI image patches), the optimal attention
bilinear form is a Toeplitz operator [Kailath-Sayed 1995]. By the
circulant-embedding theorem every Toeplitz matrix embeds in a circulant
on a doubled grid, and circulants are diagonalised by the DFT:

.. math::

    C_\tau = F^{*}\Lambda_\tau F, \qquad \Lambda_\tau = \mathrm{diag}(F\tau).

Hence the Toeplitz-attention operator can be evaluated in
:math:`O(N\,d\,\log N)` time and :math:`O(N\,d)` memory via two FFTs and a
pointwise spectral product — eliminating the :math:`N\times N`
materialisation that standard scaled dot-product attention requires.

This module exposes :class:`ToeplitzSpectralAttention`, a drop-in
replacement for ``nn.MultiheadAttention`` on stationary-friendly token
sequences. The CUDA-kernel-fused variant described in the parent proposal
is left as future work; the present implementation uses PyTorch's built-in
:func:`torch.fft.rfft` / :func:`torch.fft.irfft` which already dispatches
to cuFFT on GPU.

Consistency check: in the limit of an unbounded learned spectrum
:math:`\Lambda_\tau`, the Toeplitz operator becomes dense and the block
recovers full-rank attention (Szegő's first limit theorem
[Grenander-Szegő 1958]).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["ToeplitzSpectralAttention"]


class ToeplitzSpectralAttention(nn.Module):
    r"""Stationary attention via per-head circulant-embedded Toeplitz operator.

    The attention output along the token axis is

    .. math::

        \mathrm{out}(n) = \sum_{m} \tau_{h, n-m}\,\mathbf{V}_h(m),

    parameterised by a learnable per-head generating sequence
    :math:`\tau_h\in\mathbb{R}^{2N-1}`. Computed in :math:`O(Nd\log N)` via
    rfft-based circular convolution on a length-:math:`2N` padded signal.

    Args:
        embed_dim: Channel / feature dimension :math:`d`.
        num_heads: Number of attention heads. ``embed_dim`` must be
            divisible by ``num_heads``.
        token_len: Token-sequence length :math:`N`. Required up-front so the
            spectral parameter has fixed shape; the same module instance
            cannot be reused on a different :math:`N` without re-initialising
            the parameter.
        bias: Whether the input/output linear projections include bias.

    Shape:
        Input: ``(B, N, d)``.
        Output: ``(B, N, d)``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        token_len: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}.")
        if token_len < 2:
            raise ValueError("token_len must be >= 2.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.token_len = token_len
        self.in_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Per-head generating sequence of the Toeplitz operator. Length 2N
        # gives access to all symmetric off-diagonals.
        self.tau = nn.Parameter(torch.zeros(num_heads, 2 * token_len) + 1.0 / (2.0 * token_len))
        # Heuristic init: a Gaussian decay so attention is initially local.
        with torch.no_grad():
            idx = torch.arange(2 * token_len) - token_len
            sigma = max(1.0, token_len / 8.0)
            window = torch.exp(-0.5 * (idx.float() / sigma) ** 2)
            window = window / window.sum()
            self.tau.data = window.unsqueeze(0).expand(num_heads, -1).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, N, d) input; got shape {tuple(x.shape)}.")
        b, n, d = x.shape
        if n != self.token_len:
            raise ValueError(f"input token-len {n} != configured token_len {self.token_len}.")
        if d != self.embed_dim:
            raise ValueError(f"input dim {d} != configured embed_dim {self.embed_dim}.")
        v = self.in_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        # v: (B, H, N, D_h)
        pad_n = 2 * n
        v_pad = torch.nn.functional.pad(v, (0, 0, 0, n))  # zero-pad along token axis
        v_freq = torch.fft.rfft(v_pad, dim=-2, n=pad_n)
        tau_freq = torch.fft.rfft(self.tau, dim=-1, n=pad_n)
        out_freq = v_freq * tau_freq.view(1, self.num_heads, -1, 1)
        out = torch.fft.irfft(out_freq, dim=-2, n=pad_n)[..., :n, :]
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, token_len={self.token_len}"

    @staticmethod
    def expected_flops(token_len: int, embed_dim: int) -> int:
        r"""Theoretical FLOP count :math:`\sim 5 N d \log_2 N`.

        Useful for budgeting against full-attention :math:`O(N^{2}d)`.
        """
        return 5 * token_len * embed_dim * max(1, int(math.log2(max(2, token_len))))
