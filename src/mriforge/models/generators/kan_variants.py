"""KAN Architecture Variants
=============================

Kolmogorov-Arnold Network (KAN) based generators for MRI reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _KANLinear(nn.Module):
    """Learnable activation via B-spline basis + linear term."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.01)
        self.grid_size = grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear component
        lin = self.linear(x)
        # Spline approximation (simplified — uniform B-spline over [-1, 1])
        t = x.unsqueeze(-1)
        grid = torch.linspace(-1, 1, self.grid_size, device=x.device)
        basis = torch.exp(-((t - grid) ** 2) * self.grid_size)
        spline = torch.einsum("...ig,oig->...o", basis, self.spline_weight)
        return lin + spline


# ---------------------------------------------------------------------------
# 1. Enhanced KAN U-Net — KAN layers in the bottleneck of U-Net
# ---------------------------------------------------------------------------


@register_model(name="enhanced_kan_unet", training_mode="reconstruction", spatial_dims=(2,))
class EnhancedKANUNet(nn.Module, IGenerator):
    """U-Net with KAN (learnable activation) layers replacing MLP in the bottleneck."""

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
            nn.MaxPool2d(2),
            nn.Conv2d(bf * 2, bf * 4, 3, padding=1),
            nn.InstanceNorm2d(bf * 4),
            nn.GELU(),
        )

        # KAN bottleneck
        self.kan_bottleneck = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(bf * 4, bf * 8, 3, padding=1)
        )
        self.kan1 = _KANLinear(bf * 8, bf * 8)
        self.kan2 = _KANLinear(bf * 8, bf * 8)

        self.up3 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(bf * 8, bf * 4, 3, padding=1), nn.GELU())
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(bf * 4, bf * 2, 3, padding=1), nn.GELU())
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(bf * 2, bf, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.kan_bottleneck(e3)
        B, C, H, W = b.shape
        b = b.permute(0, 2, 3, 1).reshape(B * H * W, C)
        b = self.kan2(self.kan1(b))
        b = b.reshape(B, H, W, C).permute(0, 3, 1, 2)
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 2. Refined KAN U-Net — KAN in every encoder/decoder stage
# ---------------------------------------------------------------------------


@register_model(name="refined_kan_unet", training_mode="reconstruction", spatial_dims=(2,))
class RefinedKANUNet(nn.Module, IGenerator):
    """U-Net with KAN layers at every level — fully KAN-augmented."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, bf, 3, padding=1), nn.GELU())
        self.kan_enc1 = _KANLinear(bf, bf)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(bf, bf * 2, 3, padding=1), nn.GELU())
        self.kan_enc2 = _KANLinear(bf * 2, bf * 2)
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(bf * 2, bf * 4, 3, padding=1))
        self.kan_bn = _KANLinear(bf * 4, bf * 4)
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(bf * 4, bf * 2, 3, padding=1), nn.GELU())
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(bf * 2, bf, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(bf, out_channels, 1)

    def _apply_kan(self, x: torch.Tensor, kan: _KANLinear) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        out = kan(flat)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self._apply_kan(self.enc1(x), self.kan_enc1)
        e2 = self._apply_kan(self.enc2(e1), self.kan_enc2)
        b = self._apply_kan(self.bottleneck(e2), self.kan_bn)
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 3. MoE-KAN Generator — Mixture of Experts with KAN expert networks
# ---------------------------------------------------------------------------


@register_model(name="moe_kan_generator", training_mode="reconstruction")
class MoEKANGenerator(nn.Module, IGenerator):
    """Mixture of KAN experts: gating network selects among KAN-based experts."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_experts: int = 4,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(bf, bf * 2, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(bf * 2, bf * 4, 3, padding=1),
        )
        # Per-expert KAN decoder
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
                    nn.GELU(),
                    nn.Conv2d(bf, out_channels, 1),
                )
                for _ in range(n_experts)
            ]
        )
        # Gating network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bf * 4, n_experts), nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.encoder(x)
        weights = self.gate(feat)
        expert_outputs = torch.stack(
            [
                F.interpolate(exp(feat), size=(H, W), mode="bilinear", align_corners=False)
                for exp in self.experts
            ],
            dim=0,
        )
        return (weights.T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * expert_outputs).sum(dim=0)
