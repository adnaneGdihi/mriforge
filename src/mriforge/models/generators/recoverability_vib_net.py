"""Deep variational information-bottleneck restorer (MICCAI MRIxFields2026, B-2.3).

A convolutional VIB (Alemi 2017) for Task-2 restoration (degraded -> clean). A stochastic
encoder ``q(z|x) = N(mu(x), diag(exp(logvar(x))))`` over a SPATIAL latent feeds a decoder; the
KL to the standard-normal prior is the rate ``I(X;Z)`` upper bound — the **recoverability
budget** (nats/sample). Training the IB Lagrangian ``recon + beta * rate`` and sweeping ``beta``
traces the rate-distortion curve, whose low-``beta`` (high-rate) limit is the recoverability
CEILING: the best restoration achievable given the degraded input's information content.

Magnitude-only (1->1). ``forward`` is the DETERMINISTIC posterior mean (uses ``mu``, no sampling)
so the Tier-2 probe and validation are stable; the strategy draws the stochastic ``z`` for
training via :meth:`encode` + :meth:`reparameterize` + :meth:`decode`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mriforge.models.registry import register_model


@register_model(
    name="recoverability_vib_net",
    training_mode="recoverability_vib",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
)
class RecoverabilityVIBNet(nn.Module):
    """Convolutional variational information bottleneck restorer (B-2.3)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        latent_channels: int = 16,
        n_downsample: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"RecoverabilityVIBNet is magnitude-only (1->1); got in={in_channels}, "
                f"out={out_channels}."
            )
        if n_downsample < 1:
            raise ValueError(f"n_downsample must be >= 1; got {n_downsample}.")
        if latent_channels < 1:
            raise ValueError(f"latent_channels must be >= 1; got {latent_channels}.")
        self.width = int(width)
        self.latent_channels = int(latent_channels)
        self.n_downsample = int(n_downsample)
        pad = kernel_size // 2

        enc: list[nn.Module] = [nn.Conv2d(1, width, kernel_size, padding=pad), nn.SiLU()]
        for _ in range(n_downsample):
            enc += [
                nn.Conv2d(width, width, kernel_size, stride=2, padding=pad),
                nn.GroupNorm(min(8, width), width),
                nn.SiLU(),
            ]
        self.encoder = nn.Sequential(*enc)
        self.to_mu = nn.Conv2d(width, latent_channels, 1)
        self.to_logvar = nn.Conv2d(width, latent_channels, 1)

        dec: list[nn.Module] = [
            nn.Conv2d(latent_channels, width, kernel_size, padding=pad),
            nn.SiLU(),
        ]
        for _ in range(n_downsample):
            dec += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(width, width, kernel_size, padding=pad),
                nn.GroupNorm(min(8, width), width),
                nn.SiLU(),
            ]
        dec += [nn.Conv2d(width, 1, 1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_complex(x):
            raise ValueError("RecoverabilityVIBNet expects magnitude (real) input.")
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        recon = self.decoder(z)
        if recon.shape[-2:] != out_hw:  # guard against odd-size rounding in up/downsample
            recon = nn.functional.interpolate(
                recon, size=out_hw, mode="bilinear", align_corners=False
            )
        return recon

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        # Deterministic posterior mean (no sampling) — stable for the probe + validation.
        mu, _ = self.encode(x)
        return self.decode(mu, x.shape[-2:])
