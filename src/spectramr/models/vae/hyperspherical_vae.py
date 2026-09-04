r"""Hyperspherical VAE (``hyperspherical_vae``).

VAE with a von Mises-Fisher posterior over a latent on the unit sphere
:math:`S^{d-1}` and a uniform spherical prior, following Davidson *et
al.* [1]. Because the prior is on a compact manifold, the posterior KL
is bounded above and the Gaussian-VAE collapse mode (KL → 0 for small
``d``) disappears.

Reference:
    [1] T. R. Davidson *et al.*, "Hyperspherical variational
        auto-encoders," UAI 2018.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.blocks.vae.von_mises_fisher import VonMisesFisher
from spectramr.models.registry import register_model


@register_model(
    name="hyperspherical_vae",
    training_mode="vae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class HypersphericalVAE(nn.Module):
    r"""vMF-prior VAE.

    The encoder maps an image to ``(mu, kappa)``; ``mu`` is normalised to
    the sphere and ``kappa = softplus(.) + 1`` is strictly positive. A
    reparameterised vMF sample is decoded back to image space.

    Args:
        in_channels: Input image channels.
        out_channels: Reconstruction channels.
        latent_dim: Sphere dimension ``d`` (the latent lives on
            :math:`S^{d-1}`). Bessel numerics are well behaved for
            ``2 <= d <= 64``.
        hidden: Trunk channel width.
        pooled_size: Spatial size pooled to before the latent heads.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 16,
        hidden: int = 128,
        pooled_size: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim < 2:
            raise ValueError(f"hyperspherical_vae requires latent_dim >= 2, got {latent_dim}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.pooled_size = pooled_size
        feature_dim = hidden * pooled_size * pooled_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, 4, 2, 1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.mu_head = nn.Linear(feature_dim, latent_dim)
        self.kappa_head = nn.Linear(feature_dim, 1)

        self.dec_fc = nn.Linear(latent_dim, feature_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden // 2, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, out_channels, 3, 1, 1),
        )

    @property
    def name(self) -> str:
        return "hyperspherical_vae"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mu, kappa)`` with ``mu`` on the sphere and ``kappa > 0``."""
        h = self.pool(self.encoder(x))
        h = torch.flatten(h, start_dim=1)
        mu = F.normalize(self.mu_head(h), dim=-1)
        kappa = F.softplus(self.kappa_head(h)) + 1.0  # (B, 1), > 0
        return mu, kappa

    def decode(self, z: Tensor, hw: tuple[int, int]) -> Tensor:
        h = self.dec_fc(z)
        h = h.view(-1, self.hidden, self.pooled_size, self.pooled_size)
        x_hat = self.decoder(h)
        if x_hat.shape[-2:] != hw:
            x_hat = F.interpolate(x_hat, size=hw, mode="bilinear", align_corners=False)
        return x_hat

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Returns a dict consumed by ``VAETrainingStrategy``: ``reconstruction``
        (``x_hat``) and ``kl_override`` (the batch-mean closed-form vMF-vs-
        uniform KL, a non-negative scalar). The vMF posterior has no Gaussian
        (mu, logvar), so KL is supplied through the override channel.
        """
        mu, kappa = self.encode(x)
        q = VonMisesFisher(mu, kappa, validate=False)
        z = q.rsample()
        x_hat = self.decode(z, x.shape[-2:])
        kl = q.kl_uniform().mean()
        return {"reconstruction": x_hat, "kl_override": kl}


__all__ = ["HypersphericalVAE"]
