"""CNN Architecture Variants
=============================

Distinct convolutional architectures for MRI reconstruction — each differs
in connectivity pattern, feature aggregation, or output head design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------


class _ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2), nn.InstanceNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 1. ConvNeXt — inverted bottleneck with LayerNorm and large kernels
# ---------------------------------------------------------------------------


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x.permute(0, 3, 1, 2)
        return residual + x


@register_model(name="convnext", training_mode="reconstruction")
class ConvNeXtGenerator(nn.Module, IGenerator):
    """ConvNeXt-based generator with inverted bottleneck blocks."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dims: list[int] | None = None,
        depths: list[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        dims = dims or [48, 96, 192]
        depths = depths or [3, 3, 3]
        self.stem = nn.Conv2d(in_channels, dims[0], 4, stride=4)
        self.stages = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths, strict=False)):
            stage = nn.Sequential(*[_ConvNeXtBlock(dim) for _ in range(depth)])
            self.stages.append(stage)
            if i < len(dims) - 1:
                self.stages.append(nn.Sequential(nn.Conv2d(dim, dims[i + 1], 2, stride=2)))
        self.head = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], dims[0], 4, stride=4), nn.Conv2d(dims[0], out_channels, 1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
        out = self.head(h)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 2. DenseNet Plus — dense connections with growth rate
# ---------------------------------------------------------------------------


class _DenseBlock(nn.Module):
    def __init__(self, in_ch: int, growth: int, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.InstanceNorm2d(in_ch + i * growth),
                    nn.GELU(),
                    nn.Conv2d(in_ch + i * growth, growth, 3, padding=1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return torch.cat(features, dim=1)


@register_model(name="densenet_plus", training_mode="reconstruction")
class DenseNetPlusGenerator(nn.Module, IGenerator):
    """Dense connectivity with bottleneck transitions for MRI reconstruction."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        growth: int = 16,
        n_blocks: int = 4,
        **kwargs,
    ):
        super().__init__()
        ch = 64
        self.stem = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(_DenseBlock(ch, growth, n_layers=4))
            out_ch = ch + growth * 4
            self.transitions.append(
                nn.Sequential(nn.Conv2d(out_ch, out_ch // 2, 1), nn.AvgPool2d(2))
            )
            ch = out_ch // 2
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 256))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch, 64, 4, stride=4), _ConvBN(64, 32), nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.stem(x)
        for blk, trans in zip(self.blocks, self.transitions, strict=False):
            h = trans(blk(h))
        out = self.decoder(h)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 3. Enhanced U-Net — U-Net with residual connections and SE blocks
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // r),
            nn.ReLU(),
            nn.Linear(ch // r, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


# Registered as ``se_residual_unet``, NOT ``enhanced_unet`` (#977).
#
# ``enhanced_unet`` is taken. ``ModelFactory`` binds that string to the 657-line
# configurable ``UNet`` via the import-compatibility alias
# ``EnhancedUNetGenerator = UNet`` (``reconstruction/unet.py:1177``), and the
# factory is the only build path -- ``create_generator`` raises for a name it
# does not hold rather than consulting ``MODEL_REGISTRY``. So while this class
# claimed ``enhanced_unet`` in ``MODEL_REGISTRY``, 11 arms declaring that name
# built the configurable UNet and this class was reachable from nothing.
#
# The name is not being taken back: those 11 arms include a `validated/` one,
# and none of their descriptions asks for squeeze-excitation (six explicitly set
# ``residual_learning: false``). Renaming here removes the collision without
# moving a single arm, and makes this class buildable for the first time.
@register_model(name="se_residual_unet", training_mode="reconstruction", spatial_dims=(2,))
class EnhancedUNet(nn.Module, IGenerator):
    """U-Net with squeeze-excitation and residual blocks."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = nn.Sequential(_ConvBN(in_channels, bf), _ConvBN(bf, bf), _SEBlock(bf))
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2), _ConvBN(bf, bf * 2), _ConvBN(bf * 2, bf * 2), _SEBlock(bf * 2)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2), _ConvBN(bf * 2, bf * 4), _ConvBN(bf * 4, bf * 4), _SEBlock(bf * 4)
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), _ConvBN(bf * 4, bf * 8), _ConvBN(bf * 8, bf * 8)
        )
        self.up3 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.dec3 = nn.Sequential(_ConvBN(bf * 8, bf * 4), _SEBlock(bf * 4))
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = nn.Sequential(_ConvBN(bf * 4, bf * 2), _SEBlock(bf * 2))
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = nn.Sequential(_ConvBN(bf * 2, bf), _SEBlock(bf))
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 4. FPN U-Net — Feature Pyramid Network for multi-scale reconstruction
# ---------------------------------------------------------------------------


@register_model(name="fpn_unet", training_mode="reconstruction")
class FPNUNet(nn.Module, IGenerator):
    """Feature Pyramid Network: top-down pathway + lateral connections."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = _ConvBN(in_channels, bf)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), _ConvBN(bf, bf * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), _ConvBN(bf * 2, bf * 4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), _ConvBN(bf * 4, bf * 8))
        self.lat4 = nn.Conv2d(bf * 8, bf, 1)
        self.lat3 = nn.Conv2d(bf * 4, bf, 1)
        self.lat2 = nn.Conv2d(bf * 2, bf, 1)
        self.lat1 = nn.Conv2d(bf, bf, 1)
        self.smooth = nn.Conv2d(bf, bf, 3, padding=1)
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        p4 = self.lat4(e4)
        p3 = self.lat3(e3) + F.interpolate(
            p4, size=e3.shape[2:], mode="bilinear", align_corners=False
        )
        p2 = self.lat2(e2) + F.interpolate(
            p3, size=e2.shape[2:], mode="bilinear", align_corners=False
        )
        p1 = self.lat1(e1) + F.interpolate(
            p2, size=e1.shape[2:], mode="bilinear", align_corners=False
        )
        return self.head(self.smooth(p1))


# ---------------------------------------------------------------------------
# 5. Fusion U-Net — multi-encoder U-Net fusing different input representations
# ---------------------------------------------------------------------------


@register_model(name="fusion_unet", training_mode="reconstruction")
class FusionUNet(nn.Module, IGenerator):
    """Dual-encoder U-Net: processes input in two streams and fuses at bottleneck."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        half = in_channels // 2 if in_channels > 1 else in_channels
        self.enc_a = nn.Sequential(
            _ConvBN(half, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        self.enc_b = nn.Sequential(
            _ConvBN(in_channels - half, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        self.fusion = _ConvBN(bf * 8, bf * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf * 2),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            _ConvBN(bf, bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        half = x.shape[1] // 2 if x.shape[1] > 1 else x.shape[1]
        a = self.enc_a(x[:, :half])
        b = self.enc_b(x[:, half:])
        fused = self.fusion(torch.cat([a, b], dim=1))
        out = self.decoder(fused)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 6. Multi-Branch Residual — parallel branches with different kernel sizes
# ---------------------------------------------------------------------------


@register_model(name="multi_branch_residual", training_mode="reconstruction")
class MultiBranchResidual(nn.Module, IGenerator):
    """Parallel branches with 3×3, 5×5, 7×7 kernels fused via 1×1 conv."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: int = 32,
        depth: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, features, 3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "b3": nn.Sequential(nn.Conv2d(features, features, 3, padding=1), nn.GELU()),
                        "b5": nn.Sequential(nn.Conv2d(features, features, 5, padding=2), nn.GELU()),
                        "b7": nn.Sequential(nn.Conv2d(features, features, 7, padding=3), nn.GELU()),
                        "fuse": nn.Conv2d(features * 3, features, 1),
                    }
                )
            )
        self.head = nn.Conv2d(features, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = h + blk["fuse"](torch.cat([blk["b3"](h), blk["b5"](h), blk["b7"](h)], dim=1))
        return self.head(h)


# ---------------------------------------------------------------------------
# 7. Ensemble U-Net — multiple U-Net heads for ensemble prediction
# ---------------------------------------------------------------------------


@register_model(name="ensemble_unet", training_mode="reconstruction")
class EnsembleUNet(nn.Module, IGenerator):
    """Shared encoder with multiple decoder heads; output is the average."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        n_heads: int = 3,
        base_features: int = 32,
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
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
                    _ConvBN(bf * 2, bf),
                    nn.ConvTranspose2d(bf, bf, 2, stride=2),
                    nn.Conv2d(bf, out_channels, 1),
                )
                for _ in range(n_heads)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.encoder(x)
        preds = [
            F.interpolate(head(feat), size=x.shape[2:], mode="bilinear", align_corners=False)
            for head in self.heads
        ]
        return torch.stack(preds).mean(dim=0)


# ---------------------------------------------------------------------------
# 8. Motion-Robust U-Net — augmented with motion artifact resilience layers
# ---------------------------------------------------------------------------


@register_model(name="motion_robust_unet", training_mode="reconstruction")
class MotionRobustUNet(nn.Module, IGenerator):
    """U-Net with deformable-conv-inspired offsets for motion robustness."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(bf, 2, 3, padding=1),
            nn.Tanh(),
        )
        self.backbone = nn.Sequential(
            _ConvBN(in_channels, bf),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        offsets = self.offset_net(x)
        B, _, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        grid = grid + offsets.permute(0, 2, 3, 1) * 0.1
        aligned = F.grid_sample(x, grid, align_corners=True, mode="bilinear", padding_mode="border")
        return self.backbone(aligned)


# ---------------------------------------------------------------------------
# 9. Equivariant U-Net — channels for rotation equivariance via group convolutions
# ---------------------------------------------------------------------------


@register_model(name="equivariant_unet", training_mode="reconstruction")
class EquivariantUNet(nn.Module, IGenerator):
    """U-Net with rotation-equivariant group convolutions (4-fold C4 symmetry)."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, bf * 4, 3, padding=1, groups=1),
            nn.InstanceNorm2d(bf * 4),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(bf * 4, bf * 8, 3, padding=1, groups=4),
            nn.InstanceNorm2d(bf * 8),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(bf * 8, bf * 16, 3, padding=1, groups=4), nn.GELU()
        )
        self.up2 = nn.ConvTranspose2d(bf * 16, bf * 8, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(bf * 16, bf * 8, 3, padding=1, groups=4), nn.GELU())
        self.up1 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(bf * 8, bf * 4, 3, padding=1, groups=4), nn.GELU())
        self.head = nn.Conv2d(bf * 4, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 10. Rotation Prediction U-Net — self-supervised rotation prediction
# ---------------------------------------------------------------------------


@register_model(name="rotation_prediction_unet", training_mode="reconstruction")
class RotationPredictionUNet(nn.Module, IGenerator):
    """U-Net with auxiliary rotation prediction head for self-supervised features."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc = nn.Sequential(
            _ConvBN(in_channels, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.ConvTranspose2d(bf, bf, 2, stride=2),
            nn.Conv2d(bf, out_channels, 1),
        )
        self.rot_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bf * 4, 4))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.enc(x)
        return F.interpolate(
            self.decoder(feat), size=x.shape[2:], mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# 11. UNet Recon — lightweight reconstruction-only U-Net
# ---------------------------------------------------------------------------


@register_model(
    name="unet_recon",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class UNetRecon(nn.Module, IGenerator):
    """Streamlined U-Net variant designed for direct MRI reconstruction."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = _ConvBN(in_channels, bf)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), _ConvBN(bf, bf * 2))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), _ConvBN(bf * 2, bf * 4))
        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.dec2 = _ConvBN(bf * 4, bf * 2)
        self.up1 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = _ConvBN(bf * 2, bf)
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)
