r"""Hierarchical (two-level) VQ-VAE (``hierarchical_vq_vae``).

VQ-VAE-2 of Razavi *et al.* [1]: a coarse *top* discrete latent captures
global structure and a fine *bottom* discrete latent captures local
detail conditioned on the top. Both are quantised against learned
codebooks via :class:`spectramr.models.vq.quantizer.VectorQuantizer`
(the existing single-level quantizer is reused at both levels).

Objective::

    L = ||x - D(z_q_top, z_q_bot)||^2 + sum_level vq_loss_level

where each ``vq_loss`` already folds the codebook + commitment terms with
straight-through gradients (see the reused quantizer).

Reference:
    [1] A. Razavi, A. van den Oord, O. Vinyals, "Generating diverse
        high-fidelity images with VQ-VAE-2," NeurIPS 2019.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.registry import register_model
from spectramr.models.vq.quantizer import VectorQuantizer


def _down(in_ch: int, out_ch: int) -> nn.Sequential:
    """Stride-2 downsample block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


def _up(in_ch: int, out_ch: int) -> nn.Sequential:
    """Stride-2 upsample block."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


@register_model(
    name="hierarchical_vq_vae",
    training_mode="vqvae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class HierarchicalVQVAE(nn.Module):
    r"""Two-level VQ-VAE-2.

    Args:
        in_channels: Input image channels.
        out_channels: Reconstruction channels.
        codebook_size_top: Number of top-level codebook entries.
        codebook_size_bot: Number of bottom-level codebook entries.
        embed_dim: Embedding dimension shared by both codebooks.
        commitment_beta: Commitment-loss weight.
        hidden: Encoder/decoder channel width.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        codebook_size_top: int = 512,
        codebook_size_bot: int = 512,
        embed_dim: int = 64,
        commitment_beta: float = 0.25,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.hidden = hidden

        # Bottom encoder: x -> H/4 features.
        self.encoder_bot = nn.Sequential(
            _down(in_channels, hidden // 2),
            _down(hidden // 2, hidden),
            nn.Conv2d(hidden, embed_dim, 3, 1, 1),
        )
        # Top encoder: bottom features -> H/16 features.
        self.encoder_top = nn.Sequential(
            _down(embed_dim, hidden),
            _down(hidden, hidden),
            nn.Conv2d(hidden, embed_dim, 3, 1, 1),
        )

        self.vq_top = VectorQuantizer(
            num_embeddings=codebook_size_top,
            embedding_dim=embed_dim,
            commitment_cost=commitment_beta,
        )
        self.vq_bot = VectorQuantizer(
            num_embeddings=codebook_size_bot,
            embedding_dim=embed_dim,
            commitment_cost=commitment_beta,
        )

        # Top decoder produces conditioning at the bottom resolution.
        self.decoder_top = nn.Sequential(
            _up(embed_dim, hidden),
            _up(hidden, embed_dim),
        )
        # Project concatenated (bottom features + top conditioning) back to
        # embed_dim before bottom quantization.
        self.pre_quant_bot = nn.Conv2d(embed_dim * 2, embed_dim, 1)

        # Bottom decoder: (top conditioning upsampled + bottom quant) -> x.
        self.decoder_bot = nn.Sequential(
            _up(embed_dim * 2, hidden),
            _up(hidden, hidden // 2),
            nn.Conv2d(hidden // 2, out_channels, 3, 1, 1),
        )

    @property
    def name(self) -> str:
        return "hierarchical_vq_vae"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _match(self, x: Tensor, ref: Tensor) -> Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
        return x

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Returns:
            dict with ``x_hat`` (reconstruction matching input shape),
            ``vq_loss`` (sum of both levels, finite scalar), and
            ``idx_top`` / ``idx_bot`` (codebook indices in ``[0, K)``).
        """
        h_bot = self.encoder_bot(x)
        h_top = self.encoder_top(h_bot)

        z_q_top, vq_loss_top, idx_top = self.vq_top(h_top)

        # Condition the bottom encoder output on the (decoded) top latent.
        cond = self.decoder_top(z_q_top)
        cond = self._match(cond, h_bot)
        z_e_bot = self.pre_quant_bot(torch.cat([h_bot, cond], dim=1))
        z_q_bot, vq_loss_bot, idx_bot = self.vq_bot(z_e_bot)

        # Decode from both latents.
        top_up = self._match(self.decoder_top(z_q_top), z_q_bot)
        dec_in = torch.cat([z_q_bot, top_up], dim=1)
        x_hat = self.decoder_bot(dec_in)
        if x_hat.shape[-2:] != x.shape[-2:]:
            x_hat = F.interpolate(x_hat, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # "reconstruction" is the key VQVAETrainingStrategy reads; "x_hat" is
        # kept as the model's own descriptive alias.
        return {
            "reconstruction": x_hat,
            "x_hat": x_hat,
            "vq_loss": vq_loss_top + vq_loss_bot,
            "idx_top": idx_top,
            "idx_bot": idx_bot,
        }


__all__ = ["HierarchicalVQVAE"]
