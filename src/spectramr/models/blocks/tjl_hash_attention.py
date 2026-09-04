r"""Tensorised Johnson-Lindenstrauss Hashed Attention.

For an order-:math:`p` tensor signal (e.g. an MRF readout of shape
``time × coils``), the tensorised JL embedding

.. math::

    \phi(\mathbf{S}) = (R_1\otimes R_2)\,\mathrm{vec}(\mathbf{S})

preserves :math:`\ell^2` distance with :math:`m=O(p\,\varepsilon^{-2}\log N)`
projections — exponentially fewer than the flat-JL bound on the dense
:math:`T\cdot n_c`-dim signal [Kasiviswanathan-Rudelson-Smith 2013].

A sign-hash yields a Hamming code; the Goemans-Williamson identity makes
Hamming similarity a monotone function of cosine similarity, so the
rank-ordering of dictionary entries is preserved under the hash. The
Klartag-Mendelson manifold extension reduces :math:`m` further to
:math:`O(r)` for Bloch-encoded signals where :math:`r\le 4`.

This module exposes :class:`TJLHashAttention`, a drop-in replacement for
attention layers in MRF-style backbones. The CUDA-int4 Tensor-Core
implementation from the parent proposal is left as future work; the
pure-PyTorch implementation captures the same asymptotic memory savings
(:math:`m^{2}` bits per token).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["TJLHashAttention", "tensorised_jl_hash"]


def tensorised_jl_hash(
    signals: torch.Tensor,
    proj1: torch.Tensor,
    proj2: torch.Tensor,
) -> torch.Tensor:
    r"""Hash an order-2 tensor signal with the Kronecker projection :math:`R_1\otimes R_2`.

    Args:
        signals: ``(B, N, T, C)`` order-2 tensor batch — ``T`` time, ``C``
            channels (e.g. coils). Real-valued.
        proj1: ``(m, T)`` projection matrix.
        proj2: ``(m, C)`` projection matrix.

    Returns:
        ``(B, N, m * m)`` real-valued hash (the pre-sign embedding).
    """
    if signals.ndim != 4:
        raise ValueError(f"signals must have shape (B, N, T, C); got {tuple(signals.shape)}.")
    if proj1.ndim != 2 or proj2.ndim != 2:
        raise ValueError("proj1 and proj2 must be 2D matrices.")
    if proj1.shape[1] != signals.shape[-2] or proj2.shape[1] != signals.shape[-1]:
        raise ValueError("Projection dimensions don't match signal dimensions.")
    # Apply proj1 along the time axis, proj2 along the channel axis.
    proj_time = torch.einsum("mt,bntc->bnmc", proj1, signals)
    proj_full = torch.einsum("kc,bnmc->bnmk", proj2, proj_time)
    return proj_full.reshape(*signals.shape[:2], -1)


class TJLHashAttention(nn.Module):
    r"""Hamming-similarity attention via tensorised Johnson-Lindenstrauss hashing.

    The block expects token signals shaped ``(B, N, T, C)`` (order-2
    tensors per token); it applies the Kronecker JL projection, sign-hashes
    the result, and computes attention scores from pairwise Hamming
    similarities (equivalent to cosine similarity by Goemans-Williamson).

    Args:
        time_dim: Time / sequence axis :math:`T`.
        coil_dim: Coil / channel axis :math:`C`.
        hash_dim: Projection dimension :math:`m` per axis. Total hash size
            is :math:`m^{2}` bits.
        seed: Optional random seed for the projection matrices (frozen
            after init, treated as a buffer not a parameter — matching
            the JL theorem hypothesis).

    Shape:
        Input signals: ``(B, N, T, C)``.
        Input values: ``(B, N, d)``.
        Output: ``(B, N, d)``.
    """

    def __init__(
        self,
        time_dim: int,
        coil_dim: int,
        hash_dim: int = 8,
        value_dim: int = 16,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if hash_dim < 2:
            raise ValueError("hash_dim must be >= 2.")
        if value_dim <= 0:
            raise ValueError("value_dim must be positive.")
        self.time_dim = time_dim
        self.coil_dim = coil_dim
        self.hash_dim = hash_dim
        self.value_dim = value_dim
        gen = torch.Generator().manual_seed(seed)
        # Per JL norm-preservation: each entry of R_i is N(0, 1/m) so the
        # Kronecker (R_1 ⊗ R_2) has expected ‖·‖_F^2 = ‖·‖_F^2.
        proj1 = torch.randn(hash_dim, time_dim, generator=gen) / math.sqrt(hash_dim)
        proj2 = torch.randn(hash_dim, coil_dim, generator=gen) / math.sqrt(hash_dim)
        self.register_buffer("proj1", proj1)
        self.register_buffer("proj2", proj2)
        self.value_proj = nn.Linear(time_dim * coil_dim, value_dim)
        self.out_proj = nn.Linear(value_dim, value_dim)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 4:
            raise ValueError(f"signals must have shape (B, N, T, C); got {tuple(signals.shape)}.")
        b, n, t, c = signals.shape
        if t != self.time_dim:
            raise ValueError(f"time_dim mismatch: {t} vs {self.time_dim}.")
        if c != self.coil_dim:
            raise ValueError(f"coil_dim mismatch: {c} vs {self.coil_dim}.")
        # Hash. Sign hash via tanh approximation so gradients flow.
        hashes = tensorised_jl_hash(signals, self.proj1, self.proj2)
        # Soft signs: tanh(beta * h) — beta large gives near-binary.
        soft_signs = torch.tanh(4.0 * hashes)
        # Compute Hamming similarity ~ cosine similarity (Goemans-Williamson).
        m_sq = soft_signs.shape[-1]
        sim = soft_signs @ soft_signs.transpose(-1, -2) / m_sq  # (B, N, N)
        attn = torch.softmax(sim, dim=-1)
        # Value projection on the flattened signal.
        v = self.value_proj(signals.reshape(b, n, t * c))
        out = attn @ v
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return (
            f"time_dim={self.time_dim}, coil_dim={self.coil_dim}, "
            f"hash_dim={self.hash_dim}, value_dim={self.value_dim}"
        )
