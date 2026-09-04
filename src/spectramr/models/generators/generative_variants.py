"""Generative Model Variants
=============================

VAE, autoregressive, flow, and other generative architectures for MRI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2), nn.InstanceNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 1. Generative Prior — U-Net with learned generative image prior
# ---------------------------------------------------------------------------


@register_model(name="generative_prior", training_mode="reconstruction")
class GenerativePriorGenerator(nn.Module, IGenerator):
    """Generator with a VAE-style latent prior for image regularization.

    Encodes to a latent z, applies a prior network, then decodes.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        latent_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.encoder = nn.Sequential(
            _ConvBN(in_channels, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        self.mu = nn.Conv2d(bf * 4, latent_dim, 1)
        self.logvar = nn.Conv2d(bf * 4, latent_dim, 1)
        self.prior_net = nn.Sequential(
            _ConvBN(latent_dim, latent_dim), _ConvBN(latent_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            _ConvBN(latent_dim, bf * 4),
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.encoder(x)
        mu, logvar = self.mu(feat), self.logvar(feat)
        if self.training:
            z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        else:
            z = mu
        z = self.prior_net(z)
        out = self.decoder(z)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 2. Causal Autoregressive — pixel-by-pixel autoregressive generation
# ---------------------------------------------------------------------------


@register_model(name="causal_autoregressive", training_mode="reconstruction")
class CausalAutoregressiveGenerator(nn.Module, IGenerator):
    """Masked causal convolution generator for autoregressive MRI reconstruction."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: int = 64,
        depth: int = 6,
        **kwargs,
    ):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features, 7, padding=3), nn.GELU()]
        for _ in range(depth):
            # Causal masking via asymmetric padding
            layers.extend([nn.Conv2d(features, features, 3, padding=1), nn.GELU()])
        layers.append(nn.Conv2d(features, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. VRNN Reconstruction — variational RNN for sequential MRI slices
# ---------------------------------------------------------------------------


@register_model(name="vrnn_reconstruction", training_mode="reconstruction")
class VRNNReconstruction(nn.Module, IGenerator):
    """Variational RNN: processes MRI slices sequentially with latent state."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(8)
        )
        self.gru = nn.GRUCell(hidden_dim * 64, hidden_dim)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim + latent_dim, hidden_dim * 64), nn.GELU())
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, out_channels, 3, padding=1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.feat_extractor(x).flatten(1)
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        h = self.gru(feat, h)
        mu = self.prior_mu(h)
        logvar = self.prior_logvar(h)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if self.training else mu
        dec = self.decoder(torch.cat([h, z], dim=1))
        dec = dec.reshape(B, self.hidden_dim, 8, 8)
        out = self.spatial_decoder(dec)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 4. Invertible Network — bijective flow-based reconstruction
# ---------------------------------------------------------------------------


class _InvertibleBlock(nn.Module):
    """Additive coupling layer."""

    def __init__(self, ch: int):
        super().__init__()
        half = ch // 2
        self.net = nn.Sequential(
            nn.Conv2d(half, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, half, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        y1 = x1
        y2 = x2 + self.net(x1)
        return torch.cat([y1, y2], dim=1)


@register_model(name="invertible_network", training_mode="reconstruction")
class InvertibleNetworkGenerator(nn.Module, IGenerator):
    """Fully invertible network via additive coupling layers."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: int = 32,
        n_blocks: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, features, 1)
        self.blocks = nn.Sequential(*[_InvertibleBlock(features) for _ in range(n_blocks)])
        self.proj_out = nn.Conv2d(features, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj_out(self.blocks(self.proj_in(x)))


# ---------------------------------------------------------------------------
# 5. Structured VAE — VAE with hierarchical latent structure
# ---------------------------------------------------------------------------


@register_model(name="structured_vae", training_mode="vae")
class StructuredVAEGenerator(nn.Module, IGenerator):
    """VAE with hierarchical ladder structure: multiple latent layers."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_levels: int = 3,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.encoders = nn.ModuleList()
        self.mu_layers = nn.ModuleList()
        self.logvar_layers = nn.ModuleList()
        ch = in_channels
        for i in range(n_levels):
            out_ch = bf * (2**i)
            self.encoders.append(
                nn.Sequential(nn.Conv2d(ch, out_ch, 3, stride=2, padding=1), nn.GELU())
            )
            self.mu_layers.append(nn.Conv2d(out_ch, out_ch, 1))
            self.logvar_layers.append(nn.Conv2d(out_ch, out_ch, 1))
            ch = out_ch
        self.decoders = nn.ModuleList()
        for i in reversed(range(n_levels)):
            in_ch = bf * (2**i)
            out_ch = bf * (2 ** max(0, i - 1)) if i > 0 else out_channels
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                    nn.GELU() if i > 0 else nn.Identity(),
                )
            )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        h = x
        latents = []
        for enc, mu_l, lv_l in zip(self.encoders, self.mu_layers, self.logvar_layers, strict=False):
            h = enc(h)
            mu, lv = mu_l(h), lv_l(h)
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu) if self.training else mu
            latents.append(z)
        h = latents[-1]
        for dec in self.decoders:
            h = dec(h)
        return F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 6. Latent Diffusion Hierarchical — multi-scale latent diffusion
# ---------------------------------------------------------------------------


@register_model(name="latent_diffusion_hierarchical", training_mode="diffusion")
class LatentDiffusionHierarchical(nn.Module, IGenerator):
    """Hierarchical latent diffusion: coarse-to-fine denoising in latent space."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.encoder = nn.Sequential(
            _ConvBN(in_channels, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        self.coarse_denoiser = nn.Sequential(_ConvBN(bf * 4, bf * 4), _ConvBN(bf * 4, bf * 4))
        self.fine_denoiser = nn.Sequential(_ConvBN(bf * 2, bf * 2), _ConvBN(bf * 2, bf * 2))
        self.upsample_coarse = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.upsample_fine = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.encoder(x)
        coarse = self.coarse_denoiser(feat)
        fine_input = self.upsample_coarse(coarse)
        fine = self.fine_denoiser(fine_input)
        out = self.head(self.upsample_fine(fine))
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

    # F11 (2026-05-21 smoke audit): identity encode/decode pair, same
    # pattern as ``x_diffusion_latent.py``. The model_type
    # ``latent_diffusion_hierarchical`` matches DiffusionTrainingStrategy's
    # ``_is_latent_diffusion`` heuristic (substring "latent" + "diffusion")
    # which then expects ``encode_to_latent`` on the generator.
    # Despite the class name, this model operates in pixel space (the
    # final F.interpolate restores input spatial size), so making the
    # latent-diffusion path a no-op is semantically correct.
    # See TODO/audit/smoke_audit_20260521.md §F11 and the analogous
    # docstring in ``x_diffusion_latent.py::encode_to_latent``.
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Identity encoder — this generator operates in pixel space."""
        return x

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Identity decoder — pair of :meth:`encode_to_latent`."""
        return z
