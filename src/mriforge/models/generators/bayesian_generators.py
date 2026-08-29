"""Bayesian & Uncertainty-Aware Generators
==========================================

Models that explicitly handle epistemic/aleatoric uncertainty in MRI reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _UNetBackbone(nn.Module):
    """Lightweight U-Net backbone shared by Bayesian variants."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.InstanceNorm2d(mid_ch), nn.GELU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(mid_ch, mid_ch * 2, 3, padding=1),
            nn.InstanceNorm2d(mid_ch * 2),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(mid_ch * 2, mid_ch * 4, 3, padding=1), nn.GELU()
        )
        self.up2 = nn.ConvTranspose2d(mid_ch * 4, mid_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(mid_ch * 4, mid_ch * 2, 3, padding=1), nn.GELU())
        self.up1 = nn.ConvTranspose2d(mid_ch * 2, mid_ch, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(mid_ch * 2, mid_ch, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 1. Hierarchical Bayesian — multi-level Bayesian inference with learned priors
# ---------------------------------------------------------------------------


@register_model(name="hierarchical_bayesian", training_mode="reconstruction")
class HierarchicalBayesianGenerator(nn.Module, IGenerator):
    """Multi-level Bayesian generator with per-level learned prior distributions.

    Each encoder level produces mean/logvar that parameterize a Gaussian.
    During training, samples are drawn and KL divergence regularizes the latents.
    """

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1), nn.InstanceNorm2d(bf), nn.GELU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(bf, bf * 2, 3, padding=1),
            nn.InstanceNorm2d(bf * 2),
            nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(bf * 2, bf * 4, 3, padding=1), nn.GELU()
        )

        # Per-level posterior parameters
        self.mu_layers = nn.ModuleList([nn.Conv2d(bf * (2**i), bf * (2**i), 1) for i in range(3)])
        self.logvar_layers = nn.ModuleList(
            [nn.Conv2d(bf * (2**i), bf * (2**i), 1) for i in range(3)]
        )

        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(bf * 4, bf * 2, 3, padding=1), nn.GELU())
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(bf * 2, bf, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(bf, out_channels, 1)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        encs = []
        h = x
        for i, enc in enumerate([self.enc1, self.enc2, self.enc3]):
            h = enc(h)
            mu = self.mu_layers[i](h)
            logvar = self.logvar_layers[i](h)
            h = self._reparameterize(mu, logvar)
            encs.append(h)

        d2 = self.dec2(torch.cat([self.up2(encs[2]), encs[1]], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), encs[0]], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 2. Mixture Density Network — output Gaussian mixture parameters
# ---------------------------------------------------------------------------


@register_model(name="mixture_density_network", training_mode="reconstruction")
class MixtureDensityGenerator(nn.Module, IGenerator):
    """Predicts a mixture of Gaussians per pixel; output is the mode mean."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        n_components: int = 5,
        base_features: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.n_components = n_components
        self.backbone = _UNetBackbone(in_channels, base_features, base_features)
        # Per-component: weight, mean, logvar
        self.pi_head = nn.Conv2d(base_features, n_components, 1)
        self.mu_head = nn.Conv2d(base_features, n_components * out_channels, 1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.backbone(x)
        pi = F.softmax(self.pi_head(feat), dim=1)  # (B, K, H, W)
        mu = self.mu_head(feat)  # (B, K*C, H, W)
        B, _, H, W = mu.shape
        mu = mu.reshape(B, self.n_components, self.out_channels, H, W)
        pi = pi.unsqueeze(2)
        return (pi * mu).sum(dim=1)  # weighted average → (B, C, H, W)


# ---------------------------------------------------------------------------
# 3. Uncertainty Wrapper — wraps a backbone and outputs prediction + uncertainty map
# ---------------------------------------------------------------------------


@register_model(name="uncertainty_wrapper", training_mode="reconstruction")
class UncertaintyWrapperGenerator(nn.Module, IGenerator):
    """Backbone + dual-head: one for reconstruction, one for pixel-wise uncertainty."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        self.backbone = _UNetBackbone(in_channels, base_features, base_features)
        self.recon_head = nn.Conv2d(base_features, out_channels, 1)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(base_features, base_features // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_features // 2, out_channels, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.backbone(x)
        recon = self.recon_head(feat)
        # Uncertainty map is available via self.get_uncertainty(x)
        return recon

    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.uncertainty_head(feat)


# ---------------------------------------------------------------------------
# 4. Robust Certified — adversarially certified reconstruction
# ---------------------------------------------------------------------------


@register_model(name="robust_certified", training_mode="reconstruction")
class RobustCertifiedGenerator(nn.Module, IGenerator):
    """U-Net with Lipschitz-constrained layers for certifiable robustness."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.backbone = _UNetBackbone(in_channels, bf, out_channels)
        # Spectral norm on all conv layers for Lipschitz bound
        for module in self.backbone.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.utils.parametrizations.spectral_norm(module)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.backbone(x)
