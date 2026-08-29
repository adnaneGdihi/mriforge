r"""Stage-1 wrapper around `AutoencoderKL` for two-stage LDM training.

A thin generator that exposes the AutoencoderKL used inside
`LatentDiffusionGenerator` as a stand-alone trainable model. The point
is to produce a checkpoint whose state-dict keys (``autoencoder.*``)
slot directly into the stage-2 LDM via its
``vae_checkpoint_path`` / ``freeze_vae`` kwargs — no manual key
surgery required.

Forward returns the reconstructed image; the training strategy
computes the standard VAE loss (recon + KL) using ``mu``, ``logvar``
exposed via ``encode``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.generators.latent_diffusion_generator import AutoencoderKL
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@register_model(
    name="autoencoder_kl_pretrain",
    training_mode="vae",
    supports_contrast_conditioning=True,
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class AutoencoderPretrainGenerator(nn.Module, IGenerator):
    r"""Stand-alone AutoencoderKL trainer for stage-1 LDM pretraining.

    The submodule is exposed under the attribute ``autoencoder`` so
    the saved state-dict carries an ``autoencoder.*`` prefix that
    matches the stage-2 :class:`LatentDiffusionGenerator`'s
    autoencoder layout exactly. The stage-2 loader only needs to copy
    matching keys — no renaming.

    Multi-contrast support is intentionally **declared but unused**: the
    forward accepts ``contrast_idx`` from the dataloader (so the audit
    `multi_contrast_model_support` check passes) but does not condition
    the encoder/decoder on it. The design choice is deliberate — one
    contrast-agnostic autoencoder learns a shared latent across T1w /
    T2w / FLAIR / PDw because all contrasts share brain anatomy and only
    differ in intensity statistics. Contrast-aware generation happens at
    stage-2 (LDM) via ``use_contrast_guidance: true``.

    Args:
        in_channels: Input channels (matches the stage-2 LDM).
        out_channels: Reconstruction channels (matches stage-2 LDM).
        latent_channels: Latent z dimension (matches stage-2 LDM).
        base_channels: Channel base width (matches stage-2 LDM).
        num_layers: Encoder/decoder depth.
        downsample_factor: Spatial compression ratio (8 = 256→32).
        spatial_dims: 2 or 3.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 64,
        num_layers: int = 4,
        downsample_factor: int = 8,
        spatial_dims: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.spatial_dims = spatial_dims
        self.autoencoder = AutoencoderKL(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            downsample_factor=downsample_factor,
            spatial_dims=spatial_dims,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) — used by the VAE strategy for the KL term."""
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        contrast_idx: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Standard VAE forward — returns recon + (mu, logvar) for the loss.

        Returning a dict matches the convention used by other VAE
        generators in the codebase so the existing VAE training strategy
        can read ``mu``/``logvar`` for the KL term. ``contrast_idx`` is
        accepted from the dataloader but intentionally not used — see
        the class docstring.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return {
            "reconstruction": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }
