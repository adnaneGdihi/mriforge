"""SFCWindowedAttention — local-window attention along an SFC ordering.

Implements the building block from the IMPLEMENTATION_SPEC.md §14.4:
self-attention restricted to fixed-size windows along an arbitrary 1-D
SFC ordering (Hilbert by default), augmented with ``n_global`` learned
tokens that participate in every window's attention pool.

The motivation is from the cross-domain backbone (Phase 5 / XDTA): full
attention on Hilbert-tokenised k-space at 320×320 resolution costs
``T = HW / window_size`` token positions which is intractable globally,
but window-local attention along a *locality-preserving* curve captures
99 % of the relevant context with O(W²) cost per window. The global
tokens give every window a constant-size summary of the whole sequence.

Tested against:

* Spec §14.4 reference snippet (tensor shapes, einsum layout)
* Locality preservation under Hilbert ordering (Moon et al. 2001)

References
----------
* IMPLEMENTATION_SPEC.md §11 (XDTA architecture) and §14.4 (reference
  PyTorch snippet) — *Traversal-Based Computational MRI*.
* B. Moon et al., "Analysis of the clustering properties of the Hilbert
  space-filling curve," IEEE TKDE 13(1), 2001.
* A. Vaswani et al., "Attention is all you need," NeurIPS 2017
  (scaled-dot-product attention).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFCWindowedAttention(nn.Module):
    r"""Local-window self-attention along a 1-D SFC ordering.

    The module assumes the input has already been linearised by an
    :class:`HilbertOrder` (or any other SFC). It splits the token
    sequence into windows of size ``window``, prepends ``n_global``
    learned tokens to each window, runs per-window scaled-dot-product
    attention, and concatenates back.

    Args:
        d_model:  Embedding dimension.
        n_heads:  Number of attention heads.
        window:   Tokens per window (typically 256 per spec §11.1).
        n_global: Learned global tokens prepended to every window
                  (typically 32 per spec §11.1).
        dropout:  Optional attention dropout (default 0).

    Shape
    -----
    Input:   ``(B, T, D)`` already in SFC order.
    Output:  ``(B, T, D)`` same length.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window: int = 256,
        n_global: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if window < 1 or n_global < 0:
            raise ValueError("window must be >= 1 and n_global must be >= 0")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.n_global = n_global
        self.dropout = float(dropout)

        # Learned global tokens — broadcast across batch and windows
        self.global_tokens = (
            nn.Parameter(torch.randn(n_global, d_model) * 0.02) if n_global > 0 else None
        )

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        if self.d_model != D:
            raise ValueError(f"Expected D={self.d_model}, got {D}")

        # ── 1. Right-pad to a multiple of window ────────────────────────
        pad = (-T) % self.window
        x_pad = F.pad(x, (0, 0, 0, pad))
        T_pad = T + pad
        n_win = T_pad // self.window

        # ── 2. Reshape into per-window batches (B*n_win, W, D) ──────────
        x_w = x_pad.view(B, n_win, self.window, D).reshape(B * n_win, self.window, D)

        # ── 3. Prepend global tokens (constant context across windows) ──
        if self.n_global > 0:
            g = self.global_tokens.unsqueeze(0).expand(B * n_win, -1, -1)
            x_w = torch.cat([g, x_w], dim=1)  # (B*n_win, n_global + W, D)
        L = x_w.shape[1]  # L = n_global + W

        # ── 4. QKV projection + multi-head reshape ──────────────────────
        qkv = self.qkv(x_w)  # (B*n_win, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B * n_win, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * n_win, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * n_win, L, self.n_heads, self.head_dim).transpose(1, 2)

        # ── 5. Scaled-dot-product attention ─────────────────────────────
        # Scale by sqrt(head_dim) implicitly inside SDPA
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B*n_win, n_heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B * n_win, L, D)

        # ── 6. Drop global tokens; restore (B, T_pad, D); slice back ────
        if self.n_global > 0:
            out = out[:, self.n_global :, :]  # (B*n_win, W, D)
        out = out.reshape(B, n_win * self.window, D)  # reshape: SDPA may
        # return non-contiguous
        out = out[:, :T, :]  # crop pad

        return self.out_proj(out)


class CrossDomainBlock(nn.Module):
    """One block of the XDTA cross-domain stack (Spec §11.1, §14.5).

    Two parallel SFC-windowed self-attention streams (k-space + image)
    plus optional cross-stream attention every few blocks.

    Args:
        d_model:  Embedding dimension.
        n_heads:  Heads per stream.
        win_attn: Window size along the SFC ordering.
        n_global: Global tokens per window.
        cross:    If True, run cross-stream attention after self.
        ff_mult:  Hidden dimension multiplier in the feed-forward MLP.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        win_attn: int = 256,
        n_global: int = 32,
        cross: bool = False,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.k_attn = SFCWindowedAttention(d_model, n_heads, win_attn, n_global)
        self.x_attn = SFCWindowedAttention(d_model, n_heads, win_attn, n_global)
        self.cross = cross
        if cross:
            self.cross_attn_kx = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.cross_attn_xk = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_k_self = nn.LayerNorm(d_model)
        self.norm_x_self = nn.LayerNorm(d_model)
        self.norm_k_cross = nn.LayerNorm(d_model) if cross else None
        self.norm_x_cross = nn.LayerNorm(d_model) if cross else None
        self.norm_k_ff = nn.LayerNorm(d_model)
        self.norm_x_ff = nn.LayerNorm(d_model)
        self.ff_k = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ff_x = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(
        self,
        z_k: torch.Tensor,
        z_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm + residual self-attention
        z_k = z_k + self.k_attn(self.norm_k_self(z_k))
        z_x = z_x + self.x_attn(self.norm_x_self(z_x))

        if self.cross:
            qk = self.norm_k_cross(z_k)
            qx = self.norm_x_cross(z_x)
            z_k2, _ = self.cross_attn_kx(qk, qx, qx)
            z_x2, _ = self.cross_attn_xk(qx, qk, qk)
            z_k = z_k + z_k2
            z_x = z_x + z_x2

        # Pre-norm + residual MLP
        z_k = z_k + self.ff_k(self.norm_k_ff(z_k))
        z_x = z_x + self.ff_x(self.norm_x_ff(z_x))
        return z_k, z_x


__all__ = ["CrossDomainBlock", "SFCWindowedAttention"]
