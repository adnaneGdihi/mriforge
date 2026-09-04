"""Latent-diffusion generator with a pixel-space high-frequency residual head.

Latent diffusion compresses the image into a small latent before running
the reverse process. That compression is cheap and gives easier
conditioning, but it discards high-frequency content -- small lesions
and fine textures lose sharpness after the autoencoder round-trip.

This wrapper restores the missing detail with a *small* pixel-space
UNet residual head that takes the decoded image (plus an optional
contrast-conditioning embedding) and predicts a residual added back to
the decoded output:

.. math::

    \\hat{x}_{\\text{final}} = D(z_t)
        + h_\\theta\\bigl(D(z_t), c_{\\text{contrast}}\\bigr)

The head is trained against the high-frequency component of the target
(``target - lowpass(target)``) via the registered
``high_frequency_residual`` loss, so it learns to recover detail rather
than copy the target.

Multi-contrast Item 2 (``TODO/backlog_multi_contrast_remaining.md``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _ResidualHeadUNet(nn.Module):
    """Small UNet operating in pixel space on the decoded image.

    Three configurable hyperparameters: ``base_channels``, ``depth``
    (number of down/up blocks), and ``contrast_dim`` (size of the
    contrast embedding consumed via FiLM-style conditioning).

    Intentionally small -- the spec calls for a "lightweight" head; we
    sanity-check the parameter ratio in
    :meth:`LatentDiffusionWithResidualHead.residual_head_param_ratio`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        contrast_dim: int = 0,
    ) -> None:
        super().__init__()
        self.contrast_dim = contrast_dim

        # Encoder path
        encoders: list[nn.Module] = []
        ch = in_channels
        chs: list[int] = []
        for d in range(depth):
            out_ch = base_channels * (2**d)
            encoders.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(),
                )
            )
            chs.append(out_ch)
            ch = out_ch
        self.encoders = nn.ModuleList(encoders)

        # Bottleneck FiLM modulation for contrast conditioning
        if contrast_dim > 0:
            self.film = nn.Linear(contrast_dim, ch * 2)
        else:
            self.film = None

        # Decoder path (mirror of encoder)
        decoders: list[nn.Module] = []
        for d in reversed(range(depth)):
            in_ch = chs[d] * 2 if d < depth - 1 else chs[d]
            out_ch = chs[d - 1] if d > 0 else base_channels
            decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(),
                )
            )
        self.decoders = nn.ModuleList(decoders)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Zero-init the final conv so the head starts as a no-op
        # (identity residual). The backbone's decoded image is preserved
        # at t=0 and the head only gradually contributes high-frequency
        # detail as it trains -- avoids destabilising the backbone.
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(
        self, x: torch.Tensor, contrast_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < len(self.encoders) - 1:
                h = F.avg_pool2d(h, 2)

        if self.film is not None and contrast_embedding is not None:
            gamma_beta = self.film(contrast_embedding)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            h = h * (1.0 + gamma) + beta

        for i, dec in enumerate(self.decoders):
            if i > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                skip = skips[len(skips) - 1 - i]
                # Crop / pad to skip's spatial shape (handles odd dims).
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return self.out_conv(h)


@register_model(
    name="latent_diffusion_residual_head",
    training_mode="diffusion",
    supports_contrast_conditioning=True,
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class LatentDiffusionWithResidualHead(nn.Module, IGenerator):
    """Latent-diffusion backbone composed with a pixel-space residual head.

    The backbone is the existing ``latent_gan_generator`` registered as
    :class:`LatentDiffusionGenerator`. The residual head receives the
    decoded image (and an optional contrast embedding) and predicts a
    high-frequency residual that is added back to the decoded output.

    Args:
        backbone: Mapping of kwargs forwarded to the
            :class:`LatentDiffusionGenerator` constructor (``type`` key
            is stripped before forwarding).
        residual_head: Mapping of kwargs for the residual UNet
            (``base_channels``, ``depth``, ``contrast_conditioning``).
        in_channels: Image input channels (default 1).
        out_channels: Image output channels (default 1).
        num_contrasts: Number of distinct contrasts for the conditioning
            embedding when ``residual_head.contrast_conditioning`` is True.
    """

    def __init__(
        self,
        backbone: dict[str, Any] | None = None,
        residual_head: dict[str, Any] | None = None,
        in_channels: int = 1,
        out_channels: int = 1,
        num_contrasts: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Lazy import to avoid circular registration (the backbone's
        # @register_model fires when latent_diffusion_generator is
        # imported, and that module pulls in a fairly large tree).
        from spectramr.models.generators.latent_diffusion_generator import (
            LatentDiffusionGenerator,
        )

        backbone_cfg = dict(backbone or {})
        backbone_cfg.pop("type", None)  # YAML carries the dispatch key
        backbone_cfg.setdefault("in_channels", in_channels)
        backbone_cfg.setdefault("out_channels", out_channels)
        backbone_cfg.setdefault("num_contrasts", num_contrasts)
        self.backbone = LatentDiffusionGenerator(**backbone_cfg)

        rh_cfg = dict(residual_head or {})
        use_contrast = bool(rh_cfg.pop("contrast_conditioning", False))
        contrast_dim = num_contrasts if use_contrast else 0
        self.residual_head = _ResidualHeadUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(rh_cfg.get("base_channels", 32)),
            depth=int(rh_cfg.get("depth", 3)),
            contrast_dim=contrast_dim,
        )
        if use_contrast:
            self.contrast_embedding = nn.Embedding(num_contrasts, num_contrasts)
        else:
            self.contrast_embedding = None

    def forward(
        self,
        x: torch.Tensor,
        contrast_idx: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compose backbone reconstruction with residual head correction.

        Args:
            x: Input image batch ``[B, C, H, W]``.
            contrast_idx: Optional ``[B]`` long tensor of contrast indices.
            **kwargs: Forwarded to the backbone (timestep, conditioning).

        Returns:
            ``[B, out_channels, H, W]`` corrected image.
        """
        backbone_out = self.backbone(x, **kwargs)
        if isinstance(backbone_out, tuple):
            # Some training modes return (image, aux) -- only correct the image.
            decoded = backbone_out[0]
        else:
            decoded = backbone_out

        embed = None
        if self.contrast_embedding is not None and contrast_idx is not None:
            embed = self.contrast_embedding(contrast_idx)

        residual = self.residual_head(decoded, embed)
        return decoded + residual

    def residual_head_param_ratio(self) -> float:
        """Return ``|head| / |backbone|``.

        Used by the acceptance test in
        ``tests/unit/generators/test_latent_diffusion_with_residual_head.py``
        to verify the head stays lightweight (< 10 % of backbone params).
        """
        head_params = sum(p.numel() for p in self.residual_head.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        if backbone_params == 0:
            return float("inf")
        return head_params / backbone_params


__all__ = ["LatentDiffusionWithResidualHead"]
