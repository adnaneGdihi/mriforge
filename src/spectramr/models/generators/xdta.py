r"""XDTA — Cross-Domain Traversal Attention backbone.

Implements ``IMPLEMENTATION_SPEC.md`` §11 (Phase 5): a unified
dual-stream backbone serving reconstruction, contrast translation,
and low-field-to-high-field synthesis at inference time, switched by
conditioning tokens.

Architecture (Spec §11.1)::

    k-space stream (Hilbert-tokenised k_obs)        image stream (Hilbert-tokenised x_init)
            │                                                         │
            ▼                                                         ▼
        in_proj (ComplexLinear)                               in_proj (Linear)
            │              + cond tokens                              │       + cond tokens
            ▼                                                         ▼
                   ┌──────── CrossDomainBlock × N ────────┐
                   │       (cross-stream every 4 blocks)  │
                   ▼                                       ▼
              k-out head                                x-out head
              (ComplexLinear)                          (Linear)
                   │                                       │
                   ▼                                       ▼
               κ_hat ∈ ℂ^(B,1,H,W)                    x_hat ∈ ℝ^(B,1,H,W)

Conditioning vocabulary (Spec §11.2): 32 tokens covering 5 contrasts +
4 field strengths + a growing set of task tokens. The full vocab is
:data:`COND_VOCAB`.

References
----------
* IMPLEMENTATION_SPEC.md §11 — Phase 5 XDTA architecture and §11.2
  conditioning vocabulary.
* :class:`spectramr.models.blocks.sfc_windowed_attention.CrossDomainBlock` —
  the building block.
* :class:`spectramr.models.blocks.hilbert_order.HilbertOrder` — Hilbert
  permutation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spectramr.models.blocks.hilbert_order import HilbertOrder
from spectramr.models.blocks.sfc_windowed_attention import CrossDomainBlock
from spectramr.models.layers.complex_layers import ComplexLinear
from spectramr.models.registry import register_model

# ──────────────────────────────────────────────────────────────────────
# Conditioning vocabulary (Spec §11.2)
# ──────────────────────────────────────────────────────────────────────


COND_VOCAB: dict[str, int] = {
    # Contrasts (0–4)
    "T1": 0,
    "T1ce": 1,
    "T2": 2,
    "FLAIR": 3,
    "PD": 4,
    # Field strength (5–8)
    "0.05T": 5,
    "0.5T": 6,
    "1.5T": 7,
    "3T": 8,
    # Task tokens (9–12)
    "task=recon": 9,
    "task=c2c": 10,
    "task=lf2hf": 11,
    "task=sr": 12,
    # Reserved 13–31 for DWI / SWI / MRA / future
}
N_COND_TOKENS = 32


# ──────────────────────────────────────────────────────────────────────
# XDTA backbone
# ──────────────────────────────────────────────────────────────────────


@register_model(name="xdta", training_mode="reconstruction")
class XDTA(nn.Module):
    """Cross-Domain Traversal Attention backbone (Spec §11.1).

    Args:
        image_size:    Spatial side ``H = W`` of the input grid.
        win_tok:       Tokens per window along the SFC ordering
                       (default 32).
        d_model:       Embedding dimension (default 256 for memory;
                       spec uses 512 at full resolution).
        n_blocks:      Number of stacked :class:`CrossDomainBlock`s.
        n_heads:       Attention heads per stream.
        win_attn:      Token-window size used by :class:`SFCWindowedAttention`.
        n_global:      Learned global tokens per window.
        cross_every:   Run cross-stream attention every N blocks.
        n_cond_tokens: Size of the conditioning embedding table.
    """

    def __init__(
        self,
        image_size: int = 320,
        win_tok: int = 32,
        d_model: int = 256,
        n_blocks: int = 6,
        n_heads: int = 8,
        win_attn: int = 256,
        n_global: int = 32,
        cross_every: int = 4,
        n_cond_tokens: int = N_COND_TOKENS,
    ) -> None:
        super().__init__()
        if image_size % win_tok != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by win_tok ({win_tok})")
        self.image_size = image_size
        self.win_tok = win_tok
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.hilbert = HilbertOrder(shape=(image_size, image_size), mode="hilbert")

        # Per-window token feature dim:
        #   k-stream: F_k = win_tok * (1 complex value)  →  ComplexLinear maps F_k → d_model
        #   image stream: F_x = win_tok * 1 real value  →  Linear maps F_x → d_model
        # Mask tokens are concatenated to the k-stream: F_k_with_mask = 2 * win_tok
        self.k_in_proj = ComplexLinear(win_tok, d_model)
        self.k_mask_proj = nn.Linear(win_tok, d_model)
        self.x_in_proj = nn.Linear(win_tok, d_model)

        # Conditioning embeddings (Spec §11.2)
        self.cond_embed = nn.Embedding(n_cond_tokens, d_model)

        self.blocks = nn.ModuleList(
            [
                CrossDomainBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    win_attn=win_attn,
                    n_global=n_global,
                    cross=((i + 1) % cross_every == 0),
                )
                for i in range(n_blocks)
            ]
        )

        # Output heads
        self.k_norm = nn.LayerNorm(d_model)
        self.x_norm = nn.LayerNorm(d_model)
        self.k_out = ComplexLinear(d_model, win_tok)
        self.x_out = nn.Linear(d_model, win_tok)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _tokenise_kspace(
        self,
        k: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, 1, H, W) complex + (B, 1, H, W) bool`` → tokens.

        Returns:
            k_tokens:    ``(B, T, d_model)`` — embedded k-space tokens
            mask_tokens: ``(B, T, d_model)`` — embedded mask tokens
            (added to k_tokens; the model interprets the sum)
        """
        B = k.shape[0]
        # Hilbert-linearise + window
        seq_k = self.hilbert.linearize(k).squeeze(1)  # (B, N) complex
        seq_m = self.hilbert.linearize(mask.float()).squeeze(1)  # (B, N) real
        N = seq_k.shape[-1]
        T = N // self.win_tok
        k_win = seq_k.reshape(B, T, self.win_tok)  # (B, T, F) complex
        m_win = seq_m.reshape(B, T, self.win_tok)
        k_emb = self.k_in_proj(k_win).real + self.k_in_proj(k_win).imag  # complex → real for attn
        # Note: ComplexLinear preserves complex; for the attention stack we
        # collapse to magnitude+phase encoding by summing real/imag after the
        # linear projection. This keeps the downstream attention real-valued
        # while preserving phase coherence in the projection weights.
        m_emb = self.k_mask_proj(m_win)
        return k_emb + m_emb, m_emb

    def _tokenise_image(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, H, W)`` real → ``(B, T, d_model)`` tokens."""
        B = x.shape[0]
        seq_x = self.hilbert.linearize(x).squeeze(1)  # (B, N)
        T = seq_x.shape[-1] // self.win_tok
        x_win = seq_x.reshape(B, T, self.win_tok)
        return self.x_in_proj(x_win)

    def _detokenise_kspace(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model)`` → ``(B, 1, H, W)`` complex k-space."""
        B = tokens.shape[0]
        out = self.k_out(self.k_norm(tokens).to(torch.complex64))
        # ComplexLinear expects complex input; norm output is real, so we cast
        # via complex(real, 0) and let the projection re-introduce phase.
        seq = out.reshape(B, -1)  # (B, N) complex
        return self.hilbert.unflatten(
            seq.unsqueeze(1),
            (self.image_size, self.image_size),
        )

    def _detokenise_image(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        out = self.x_out(self.x_norm(tokens))  # (B, T, win_tok)
        seq = out.reshape(B, -1)  # (B, N) real
        return self.hilbert.unflatten(
            seq.unsqueeze(1),
            (self.image_size, self.image_size),
        )

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        k_obs: torch.Tensor,
        mask: torch.Tensor,
        x_init: torch.Tensor,
        cond: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Run the dual-stream cross-domain attention backbone.

        Args:
            k_obs:  ``(B, 1, H, W)`` complex64 — observed k-space
                    (zero-filled where unsampled)
            mask:   ``(B, 1, H, W)`` bool — sampling mask
            x_init: ``(B, 1, H, W)`` real — initial image (e.g. zero-
                    filled IFFT)
            cond:   ``(B, n_cond)`` long — conditioning token indices,
                    or a sequence resolved via :data:`COND_VOCAB`. None
                    means unconditional.

        Returns:
            k_hat: ``(B, 1, H, W)`` complex — refined k-space
            x_hat: ``(B, 1, H, W)`` real    — refined image
        """
        B = k_obs.shape[0]

        # Cast to required dtypes
        if not torch.is_complex(k_obs):
            k_obs = torch.complex(k_obs, torch.zeros_like(k_obs))
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            mask = (
                mask.bool()
                if isinstance(mask, torch.Tensor)
                else torch.ones_like(k_obs, dtype=torch.bool)
            )

        # Tokenise both streams
        z_k, _m_emb = self._tokenise_kspace(k_obs, mask)
        z_x = self._tokenise_image(x_init)

        # Conditioning — broadcast to every token as a learned bias
        if cond is not None:
            if not isinstance(cond, torch.Tensor):
                cond = torch.tensor(
                    [COND_VOCAB[c] if isinstance(c, str) else int(c) for c in cond],
                    device=z_k.device,
                    dtype=torch.long,
                )
            cond_vec = self.cond_embed(cond).mean(dim=1, keepdim=True)  # (B, 1, d_model)
            z_k = z_k + cond_vec
            z_x = z_x + cond_vec

        # Run blocks
        for blk in self.blocks:
            z_k, z_x = blk(z_k, z_x)

        # Detokenise
        k_hat = self._detokenise_kspace(z_k)
        x_hat = self._detokenise_image(z_x)
        return k_hat, x_hat


__all__ = ["COND_VOCAB", "N_COND_TOKENS", "XDTA"]
