r"""Hypernetwork-conditioned generator for cross-contrast translation.

Implements the hypernetwork branch of §2 / §8 — instead of injecting
contrast information through FiLM modulation of activation statistics,
the conditioning vector :math:`\mathbf{e}_c` *generates the decoder
parameters themselves*:

.. math::

   \theta_c \;=\; H(\mathbf{e}_c), \qquad G(\mathbf{x};\,\theta_c).

Empirically this is the strongest conditioning route when the contrast
distributions are physically distant (e.g. SWI vs. T1) — the entire
computation, not just per-feature scale/shift, can change with the
contrast. The trade-off is parameter count: the hypernetwork head adds
:math:`O(D_c \cdot |\theta_{\text{decoder}}|)` parameters, so we restrict
the hypernet to producing per-layer scale/shift modulation weights for
the decoder rather than full conv kernels — the SIREN-style "weight as
modulation" approximation. This recovers most of the expressivity at a
fraction of the cost.

Architecture (per pixel of decoder output):

1. Encoder: standard conv U-Net descending to a bottleneck.
2. Hypernet: MLP over
   :class:`~spectramr.models.blocks.acquisition_conditioning.AcquisitionEmbedding`
   output → (γ, β) for each decoder block.
3. Decoder: conv blocks whose group-norm scales/shifts are *replaced*
   by the hypernet's (γ, β) on each forward pass.

Tagged with ``supports_contrast_conditioning=True`` so it shows up as
a Pattern-C-capable model in the audit guard.

References
----------
[1]  E. M. Haacke *et al.*, *Magnetic Resonance Imaging: Physical
     Principles and Sequence Design*, Wiley, 2014.
[4]  X. Huang *et al.*, "Multimodal unsupervised image-to-image
     translation (MUNIT)," ECCV 2018.
[17] E. Perez *et al.*, "FiLM: Visual reasoning with a general
     conditioning layer," AAAI 2018.

§2 / §8 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.acquisition_conditioning import AcquisitionEmbedding
from spectramr.models.registry import register_model

__all__ = ["HypernetworkContrastGenerator"]


class _ConvBlock(nn.Module):
    """Conv → modulated GroupNorm → SiLU → Conv → modulated GroupNorm → SiLU.

    The GroupNorm's ``affine`` parameters are *replaced* on every forward
    pass by the hypernetwork's (γ, β) so the same backbone learns
    contrast-specific feature statistics.
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        g_out = min(groups, out_ch) if out_ch >= groups else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(g_out, out_ch, affine=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(g_out, out_ch, affine=False)
        self.out_ch = out_ch

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        # gamma, beta: (B, 2*out_ch). Split into halves for the two GNs.
        g1, g2 = gamma.chunk(2, dim=-1)
        b1, b2 = beta.chunk(2, dim=-1)
        h = self.gn1(self.conv1(x))
        h = h * g1.view(-1, self.out_ch, 1, 1) + b1.view(-1, self.out_ch, 1, 1)
        h = F.silu(h)
        h = self.gn2(self.conv2(h))
        h = h * g2.view(-1, self.out_ch, 1, 1) + b2.view(-1, self.out_ch, 1, 1)
        return F.silu(h)


@register_model(
    name="hypernetwork_contrast",
    training_mode="reconstruction",
    supports_contrast_conditioning=True,
)
class HypernetworkContrastGenerator(nn.Module):
    r"""U-Net whose decoder normalisation parameters are produced by a
    hypernetwork driven by the per-sample acquisition embedding.

    Args:
        in_channels: Input image channels.
        out_channels: Output image channels.
        base_channels: Encoder/decoder base width.
        depth: Number of down/up steps. ``depth=3`` → 3 encoder blocks,
            1 bottleneck, 3 decoder blocks.
        contrast_dim: AcquisitionEmbedding output dimension.
        n_contrasts: Cohort cardinality forwarded to AcquisitionEmbedding.
        hyper_hidden_mult: Hidden width multiplier inside the hypernet MLP.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        contrast_dim: int = 32,
        n_contrasts: int = 4,
        hyper_hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        self.depth = depth

        widths = [base_channels * (2**i) for i in range(depth + 1)]

        # ---- encoder ----
        encoder = []
        prev = in_channels
        for w in widths[:-1]:
            encoder.append(_ConvBlock(prev, w))
            prev = w
        self.encoder_blocks = nn.ModuleList(encoder)
        self.bottleneck = _ConvBlock(prev, widths[-1])

        # ---- decoder ----
        self.up_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        prev = widths[-1]
        for w in reversed(widths[:-1]):
            self.up_layers.append(nn.ConvTranspose2d(prev, w, kernel_size=2, stride=2))
            self.decoder_blocks.append(_ConvBlock(w * 2, w))
            prev = w
        self.head = nn.Conv2d(prev, out_channels, kernel_size=1)

        # ---- conditioning + hypernet ----
        self.acquisition_embedding = AcquisitionEmbedding(
            embed_dim=contrast_dim,
            n_freqs=6,
            n_contrasts=n_contrasts,
        )
        # Each decoder block needs (gamma, beta) of size 2*out_ch each.
        # Encoder blocks use static GN — only the decoder is hypernet-conditioned.
        decoder_widths = list(reversed(widths[:-1]))
        self._decoder_widths = decoder_widths
        total_modulation = sum(4 * w for w in decoder_widths)  # 2 GN × (γ+β) × out_ch
        hidden = max(contrast_dim * hyper_hidden_mult, total_modulation // 4)
        self.hypernet = nn.Sequential(
            nn.Linear(contrast_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, total_modulation),
        )
        # Initialise final linear toward (γ=1, β=0) — start as identity GN.
        with torch.no_grad():
            self.hypernet[-1].weight.zero_()
            self.hypernet[-1].bias.zero_()
            offset = 0
            for w in decoder_widths:
                # γ block first (2*w entries), then β block (2*w entries)
                self.hypernet[-1].bias[offset : offset + 2 * w] = 1.0
                offset += 4 * w

    def _split_modulation(self, mod: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for w in self._decoder_widths:
            chunk = mod[:, offset : offset + 4 * w]  # (B, 4*w)
            gamma, beta = chunk[:, : 2 * w], chunk[:, 2 * w :]
            out.append((gamma, beta))
            offset += 4 * w
        return out

    def forward(
        self,
        x: torch.Tensor,
        acquisition_params: dict[str, torch.Tensor] | None = None,
        contrast_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]

        # --- conditioning embedding ---
        if acquisition_params is None:
            zero = {
                k: torch.zeros(B, device=x.device, dtype=x.dtype)
                for k in ("TE", "TR", "TI", "FA", "B0")
            }
            c_idx = (
                contrast_idx
                if contrast_idx is not None
                else torch.zeros(B, dtype=torch.long, device=x.device)
            )
            cond = self.acquisition_embedding(zero, contrast_id=c_idx)
        else:
            params = {k: v.to(x.device).to(x.dtype) for k, v in acquisition_params.items()}
            c_idx = (
                contrast_idx
                if contrast_idx is not None
                else torch.zeros(B, dtype=torch.long, device=x.device)
            )
            cond = self.acquisition_embedding(params, contrast_id=c_idx)

        modulation_flat = self.hypernet(cond)
        per_block = self._split_modulation(modulation_flat)

        # --- encoder ---
        skips = []
        h = x
        # Encoder blocks use identity (γ=1, β=0) modulation.
        for block in self.encoder_blocks:
            ones = h.new_ones((B, 2 * block.out_ch))
            zeros = h.new_zeros((B, 2 * block.out_ch))
            h = block(h, ones, zeros)
            skips.append(h)
            h = F.avg_pool2d(h, 2)
        ones = h.new_ones((B, 2 * self.bottleneck.out_ch))
        zeros = h.new_zeros((B, 2 * self.bottleneck.out_ch))
        h = self.bottleneck(h, ones, zeros)

        # --- decoder ---
        for up, block, skip, (gamma, beta) in zip(
            self.up_layers, self.decoder_blocks, reversed(skips), per_block
        ):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h, gamma, beta)
        return self.head(h)
