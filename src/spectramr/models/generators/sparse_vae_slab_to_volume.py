"""Sparse VAE that generates a full HF volume from a single slice or slab.

This module addresses the May 2026 LDM-stage request: a variant that
accepts a 2D slice (or thin slab of N adjacent slices) of ULF / HF input
and reconstructs the *entire* high-field volume. The 2D-mode
AutoencoderKL in :mod:`spectramr.models.generators.latent_diffusion_generator`
cannot do this — its decoder is 2D and produces a same-resolution
output. Here the decoder explicitly upsamples a depth dimension from
``D_in`` (slab thickness) to ``D_out`` (full volume).

Architecture (all 3D-aware):

* ``SparseVolumetricEncoder``: ``[B, C_in, H, W, D_in]`` → ``(mu, logvar)``
  in latent ``[B, latent_channels, H', W', D_lat]``. Uses ``nn.Conv3d``
  with stride-2 downsampling on H/W and no downsampling on D when
  ``D_in == 1``, so a single-slice input flows through cleanly.
* ``SparseVolumetricDecoder``: latent → ``[B, C_out, H, W, D_out]``.
  Upsamples depth via ``nn.ConvTranspose3d`` with stride
  ``(1, 1, D_out // D_lat)`` so the network learns the slice→volume
  mapping explicitly, not by tiling.
* Sparsity: a learnable per-channel mask on the latent (TRELLIS-style)
  pruned via straight-through estimator — same regularization
  philosophy as :class:`SparseVAE` but in the volumetric setting.

The constructor accepts both ``D_in`` (input slab thickness, default 1)
and ``D_out`` (output volume depth) so the same class supports
slice→volume and slab→volume.

References
----------
* Bansal et al., "Sparse Latent Spaces in Variational Autoencoders"
  (sparsity regularization).
* TRELLIS: Sparse-structure latent for fast sampling
  (channel-mask sparsity inspiration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@dataclass
class SparseVAESlabToVolumeConfig:
    """Configuration for the slab→volume sparse VAE."""

    in_channels: int = 1
    out_channels: int = 1
    d_in: int = 1
    """Input slab thickness (1 = single slice)."""
    d_out: int = 64
    """Output volume depth — what the decoder reconstructs."""
    base_channels: int = 32
    latent_channels: int = 4
    num_layers: int = 3
    """Spatial downsampling levels (factor 2^num_layers on H, W)."""
    sparsity_lambda: float = 0.001
    sparsity_target: float = 0.05
    beta_kl: float = 1.0e-4


def _conv_norm_act(
    in_ch: int, out_ch: int, kernel: int = 3, stride: int | tuple[int, ...] = 1
) -> nn.Sequential:
    """3D Conv → GroupNorm → SiLU. ``stride`` may be int or 3-tuple."""
    if isinstance(stride, int):
        stride_tup = (stride, stride, stride)
    else:
        stride_tup = tuple(stride)
    pad = kernel // 2
    groups = 8 if out_ch % 8 == 0 else max(1, out_ch // 2)
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride_tup, padding=pad),
        nn.GroupNorm(groups, out_ch, eps=1e-6, affine=True),
        nn.SiLU(),
    )


class SparseVolumetricEncoder(nn.Module):
    """Encoder: ``[B, C_in, H, W, D_in]`` → mu, logvar."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        latent_channels: int,
        num_layers: int,
        d_in: int,
    ) -> None:
        super().__init__()
        self.d_in = d_in

        layers: list[nn.Module] = [_conv_norm_act(in_channels, base_channels, kernel=3, stride=1)]
        ch = base_channels
        # Downsample H and W by 2^num_layers; preserve D so a single-slice
        # input never hits stride-2 on a length-1 axis (which would
        # collapse the dim and confuse the decoder's later upsample).
        for i in range(num_layers):
            next_ch = ch * 2
            layers.append(_conv_norm_act(ch, next_ch, kernel=3, stride=(2, 2, 1)))
            layers.append(_conv_norm_act(next_ch, next_ch, kernel=3, stride=1))
            ch = next_ch
        # Project to 2 * latent_channels (mu and logvar concatenated).
        layers.append(nn.Conv3d(ch, latent_channels * 2, kernel_size=1))
        self.body = nn.Sequential(*layers)
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # If user supplied 4D ``[B, C, H, W]`` (single slice), inject a
        # depth axis so the 3D convs work uniformly. Decoder will still
        # reconstruct the full volume.
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        h = self.body(x)
        mu, logvar = torch.split(h, self.latent_channels, dim=1)
        return mu, logvar


class SparseVolumetricDecoder(nn.Module):
    """Decoder: latent ``[B, latent, H', W', D_lat]`` → ``[B, C_out, H, W, D_out]``."""

    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        latent_channels: int,
        num_layers: int,
        d_in: int,
        d_out: int,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # Depth upsampling factor — applied as a single ConvTranspose3d
        # at the entrance so the H/W decoder stack stays vanilla. When
        # ``d_in == d_out`` (no upsampling) this collapses to a 1x1x1
        # projection.
        self.depth_upsample_factor = max(1, d_out // max(1, d_in))

        layers: list[nn.Module] = []
        ch = base_channels * (2**num_layers)
        # Initial projection from latent → first decoder block. We bake
        # the depth upsample into a ConvTranspose3d here so a single-slice
        # latent (D=1) is expanded to the target depth in one go.
        layers.append(
            nn.ConvTranspose3d(
                latent_channels,
                ch,
                kernel_size=(3, 3, self.depth_upsample_factor),
                stride=(1, 1, self.depth_upsample_factor),
                padding=(1, 1, 0),
            )
        )
        layers.append(nn.GroupNorm(min(8, ch), ch, eps=1e-6, affine=True))
        layers.append(nn.SiLU())

        for i in range(num_layers):
            next_ch = max(base_channels, ch // 2)
            layers.append(
                nn.ConvTranspose3d(
                    ch,
                    next_ch,
                    kernel_size=4,
                    stride=(2, 2, 1),
                    padding=1,
                )
            )
            layers.append(_conv_norm_act(next_ch, next_ch, kernel=3, stride=1))
            ch = next_ch

        layers.append(nn.Conv3d(ch, out_channels, kernel_size=1))
        self.body = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.unsqueeze(-1)
        return self.body(z)


@register_model(
    name="sparse_vae_slab_to_volume",
    training_mode="vae",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class SparseVAESlabToVolumeGenerator(nn.Module, IGenerator):
    r"""Slice/slab → HF volume sparse VAE.

    Forward returns the reconstructed full volume; ``encode`` exposes the
    ``(mu, logvar)`` pair for the VAE strategy's KL term, and
    ``sparsity_loss()`` exposes the L1-on-mask term for the strategy to
    add to the total objective.

    Args:
        in_channels: Channels of the slab tensor (e.g. 1 for magnitude
            ULF / HF, 2 for real-stacked complex).
        out_channels: Channels of the reconstructed volume — matches
            ``in_channels`` for the standard "fill the volume" task.
        d_in: Input slab thickness. ``1`` = single slice (the user's
            simplest variant); larger values give the encoder more
            cross-slice context.
        d_out: Output volume depth. The decoder upsamples
            ``d_in → d_out`` along the depth axis via a single
            ``ConvTranspose3d`` at the entrance.
        base_channels / latent_channels / num_layers: standard VAE
            capacity knobs; defaults match the configured stage-1 LDM.
        sparsity_lambda / sparsity_target / beta_kl: regularization
            weights — same semantics as :class:`SparseVAE`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_in: int = 1,
        d_out: int = 64,
        base_channels: int = 32,
        latent_channels: int = 4,
        num_layers: int = 3,
        sparsity_lambda: float = 0.001,
        sparsity_target: float = 0.05,
        beta_kl: float = 1.0e-4,
        **_: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_in = d_in
        self.d_out = d_out
        self.latent_channels = latent_channels
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target
        self.beta_kl = beta_kl

        self.encoder = SparseVolumetricEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            num_layers=num_layers,
            d_in=d_in,
        )
        self.decoder = SparseVolumetricDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            num_layers=num_layers,
            d_in=d_in,
            d_out=d_out,
        )

        # Per-channel sparsity mask — straight-through estimator.
        # Initialised at 1 (no pruning) so training starts behaving like
        # a plain VAE; the sparsity_loss() penalty drives mask values
        # toward zero on unused channels.
        self.sparsity_mask = nn.Parameter(torch.ones(latent_channels))

    @property
    def name(self) -> str:
        return "sparse_vae_slab_to_volume"

    # ── Standard VAE building blocks ──────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, apply_sparsity: bool = True) -> torch.Tensor:
        if apply_sparsity:
            mask = torch.sigmoid(self.sparsity_mask).view(1, -1, 1, 1, 1)
            z = z * mask
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        apply_sparsity: bool = True,
        **_: Any,
    ) -> torch.Tensor:
        """slab/slice → reconstructed full volume."""
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z, apply_sparsity=apply_sparsity)
        # If the decoder under/overshoots the requested depth (typically
        # due to a non-divisible ``d_out / d_in`` ratio), interpolate to
        # the exact target shape so the loss path doesn't break.
        if recon.shape[-1] != self.d_out:
            recon = F.interpolate(
                recon,
                size=(*recon.shape[2:-1], self.d_out),
                mode="trilinear",
                align_corners=False,
            )
        return recon

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.decode(z, apply_sparsity=True)

    # ── Loss-side helpers ────────────────────────────────────────────

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def sparsity_loss(self) -> torch.Tensor:
        """L1 on the *active* fraction of the mask, anchored at the
        configured sparsity target. Encourages the mask to converge to
        a sparse but non-degenerate solution."""
        active_fraction = torch.sigmoid(self.sparsity_mask).mean()
        return (active_fraction - (1.0 - self.sparsity_target)).abs()

    # ── Interface boilerplate ────────────────────────────────────────

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        b = input_shape[0]
        h, w = input_shape[2], input_shape[3]
        return (b, self.out_channels, h, w, self.d_out)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
