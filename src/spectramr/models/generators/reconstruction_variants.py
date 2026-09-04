"""Reconstruction-Specific Variants
=====================================

Specialized architectures for MRI reconstruction tasks not covered by other
category files. Includes VarNet ensembles, Mamba variants, Fourier-augmented
networks, and topological/geometric approaches.
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
# 1. Ensemble VarNet Diffusion — VarNet + diffusion ensemble
# ---------------------------------------------------------------------------


@register_model(name="ensemble_varnet_diffusion", training_mode="reconstruction")
class EnsembleVarNetDiffusion(nn.Module, IGenerator):
    """Ensemble of VarNet-style cascades with diffusion refinement."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_cascades: int = 4,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.cascades = nn.ModuleList(
            [
                nn.Sequential(
                    _ConvBN(in_channels if i == 0 else out_channels, bf),
                    _ConvBN(bf, bf),
                    nn.Conv2d(bf, out_channels, 1),
                )
                for i in range(n_cascades)
            ]
        )
        self.refiner = nn.Sequential(
            _ConvBN(out_channels, bf), _ConvBN(bf, bf), nn.Conv2d(bf, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for cascade in self.cascades:
            h = (
                cascade(h) + h[:, : self.cascades[-1][-1].out_channels]
                if h.shape[1] == self.cascades[-1][-1].out_channels
                else cascade(h)
            )
        return self.refiner(h)


# ---------------------------------------------------------------------------
# 2. Meta VarNet — VarNet with meta-learned sensitivity maps
# ---------------------------------------------------------------------------


# Renamed from ``meta_varnet`` to break the collision with
# ``meta_learning/meta_varnet.py:MetaVarNet`` (the canonical MAML-aware
# implementation that experiments/training/umr/exp_meta_varnet.yaml
# expects). This is the generic reconstruction-mode wrapper kept for
# legacy callers. See TODO/audit/09_models_registry_generators.md §3.3.
@register_model(name="meta_varnet_recon_variant", training_mode="reconstruction")
class MetaVarNetGenerator(nn.Module, IGenerator):
    """VarNet variant with meta-learned sensitivity estimation."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_cascades: int = 6,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.sens_net = nn.Sequential(
            _ConvBN(in_channels, bf), _ConvBN(bf, bf), nn.Conv2d(bf, in_channels, 1), nn.Sigmoid()
        )
        self.cascades = nn.ModuleList(
            [
                nn.Sequential(
                    _ConvBN(in_channels, bf), _ConvBN(bf, bf), nn.Conv2d(bf, in_channels, 1)
                )
                for _ in range(n_cascades)
            ]
        )
        self.head = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        sens = self.sens_net(x)
        h = x * sens
        for cascade in self.cascades:
            h = h + cascade(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# 3. Vision Mamba Local-Global — Mamba with local + global scan
# ---------------------------------------------------------------------------


class _SimpleMambaBlock(nn.Module):
    """Simplified Mamba-style SSM block: 1D conv → gated projection."""

    def __init__(self, dim: int, conv_size: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.conv = nn.Conv1d(dim, dim, conv_size, padding=conv_size - 1, groups=dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm(x)
        xz = self.proj(h)
        x1, z = xz.chunk(2, dim=-1)
        x1 = self.conv(x1.transpose(1, 2))[:, :, :N].transpose(1, 2)
        return x + self.out_proj(x1 * F.silu(z))


@register_model(name="vision_mamba_local_global", training_mode="reconstruction")
class VisionMambaLocalGlobal(nn.Module, IGenerator):
    """Vision Mamba with alternating local (row) and global (column) SSM scans."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, dim: int = 64, depth: int = 6, **kwargs
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.local_blocks = nn.ModuleList([_SimpleMambaBlock(dim) for _ in range(depth)])
        self.global_blocks = nn.ModuleList([_SimpleMambaBlock(dim) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 4, stride=4),
            nn.GELU(),
            nn.Conv2d(dim // 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        for local_blk, global_blk in zip(self.local_blocks, self.global_blocks, strict=False):
            # Local scan (row-wise)
            row_tokens = patches.flatten(2).permute(0, 2, 1)
            row_tokens = local_blk(row_tokens)
            patches = row_tokens.permute(0, 2, 1).reshape(B, -1, pH, pW)
            # Global scan (column-wise)
            col_tokens = patches.permute(0, 1, 3, 2).flatten(2).permute(0, 2, 1)
            col_tokens = global_blk(col_tokens)
            patches = col_tokens.permute(0, 2, 1).reshape(B, -1, pW, pH).permute(0, 1, 3, 2)
        out = self.head(patches)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 4. Fourier-Augmented — spectral convolution + spatial convolution
# ---------------------------------------------------------------------------


@register_model(name="fourier_augmented", training_mode="reconstruction")
class FourierAugmentedGenerator(nn.Module, IGenerator):
    """Dual-path: spatial convolution + Fourier-domain spectral convolution."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.spatial = nn.Sequential(
            _ConvBN(in_channels, bf), _ConvBN(bf, bf * 2), _ConvBN(bf * 2, bf)
        )
        # Spectral path: learnable frequency-domain filter
        self.spectral_weight = nn.Parameter(torch.randn(1, bf, 1, 1) * 0.01)
        self.spectral_proj = nn.Conv2d(in_channels, bf, 1)
        self.fusion = nn.Conv2d(bf * 2, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        spatial_out = self.spatial(x)
        spec_in = self.spectral_proj(x)
        spec_fft = torch.fft.rfft2(spec_in, norm="ortho")
        spec_fft = spec_fft * self.spectral_weight
        spec_out = torch.fft.irfft2(spec_fft, s=x.shape[2:], norm="ortho")
        return self.fusion(torch.cat([spatial_out, spec_out], dim=1))


# ---------------------------------------------------------------------------
# 5. Geometric MRI — Riemannian manifold-aware reconstruction
# ---------------------------------------------------------------------------


@register_model(name="geometric_mri", training_mode="reconstruction")
class GeometricMRIGenerator(nn.Module, IGenerator):
    """MRI reconstruction on Riemannian manifold via learned metric tensor."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.metric_net = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1), nn.GELU(), nn.Conv2d(bf, 4, 1)
        )
        self.backbone = nn.Sequential(
            _ConvBN(in_channels + 4, bf),
            _ConvBN(bf, bf * 2),
            _ConvBN(bf * 2, bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        metric = self.metric_net(x)
        return self.backbone(torch.cat([x, metric], dim=1))


# ---------------------------------------------------------------------------
# 6. Riemannian Geometric — explicit geodesic-aware convolutions
# ---------------------------------------------------------------------------


@register_model(name="riemannian_geometric", training_mode="reconstruction")
class RiemannianGeometricGenerator(nn.Module, IGenerator):
    """Convolutions weighted by estimated Riemannian distance."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.dist_estimator = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1), nn.GELU(), nn.Conv2d(bf, 1, 1), nn.Sigmoid()
        )
        self.net = nn.Sequential(
            _ConvBN(in_channels, bf),
            _ConvBN(bf, bf * 2),
            _ConvBN(bf * 2, bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        dist_weight = self.dist_estimator(x)
        return self.net(x) * dist_weight


# ---------------------------------------------------------------------------
# 7. SSAD MRI — self-supervised anomaly detection
# ---------------------------------------------------------------------------


@register_model(name="ssad_mri", training_mode="reconstruction")
class SSADMRIGenerator(nn.Module, IGenerator):
    """Self-Supervised Anomaly Detection for MRI reconstruction."""

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
        self.recon_decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.ConvTranspose2d(bf, bf, 2, stride=2),
            nn.Conv2d(bf, out_channels, 1),
        )
        self.anomaly_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bf * 4, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.encoder(x)
        return F.interpolate(
            self.recon_decoder(feat), size=x.shape[2:], mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# 8. TDA Informed — topological data analysis guided reconstruction
# ---------------------------------------------------------------------------


@register_model(name="tda_informed", training_mode="reconstruction")
class TDAInformedGenerator(nn.Module, IGenerator):
    """Reconstruction guided by persistent homology features (topological regularization)."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.topo_extractor = nn.Sequential(
            nn.Conv2d(in_channels, bf, 5, padding=2), nn.GELU(), nn.Conv2d(bf, 1, 1), nn.Sigmoid()
        )
        self.backbone = nn.Sequential(
            _ConvBN(in_channels + 1, bf),
            _ConvBN(bf, bf * 2),
            _ConvBN(bf * 2, bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        topo = self.topo_extractor(x)
        return self.backbone(torch.cat([x, topo], dim=1))


# ---------------------------------------------------------------------------
# 9. Tensor Decomposition Net — CP/Tucker decomposition-based generator
# ---------------------------------------------------------------------------


@register_model(name="tensor_decomposition_net", training_mode="reconstruction")
class TensorDecompositionNetGenerator(nn.Module, IGenerator):
    """Low-rank tensor decomposition: factored convolutions for efficiency."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        rank: int = 8,
        base_features: int = 32,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.proj_in = nn.Conv2d(in_channels, bf, 1)
        self.factor_h = nn.Conv2d(bf, rank, (3, 1), padding=(1, 0))
        self.factor_w = nn.Conv2d(rank, bf, (1, 3), padding=(0, 1))
        self.body = nn.Sequential(_ConvBN(bf, bf * 2), _ConvBN(bf * 2, bf))
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.proj_in(x)
        h = self.factor_w(self.factor_h(h)) + h  # low-rank factored conv + skip
        return self.head(self.body(h))


# ---------------------------------------------------------------------------
# 10. DVS (Differentiable Volume Sampler) — differentiable slice selection
# ---------------------------------------------------------------------------


@register_model(name="dvs", training_mode="reconstruction")
class DVSGenerator(nn.Module, IGenerator):
    """Differentiable volume sampler: learns optimal slice selection weights."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1), nn.GELU(), nn.Conv2d(bf, 1, 1), nn.Sigmoid()
        )
        self.backbone = nn.Sequential(
            _ConvBN(in_channels, bf),
            _ConvBN(bf, bf * 2),
            _ConvBN(bf * 2, bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        attn = self.attention(x)
        return self.backbone(x * attn)


# ---------------------------------------------------------------------------
# 11. Invariance-Equivariance — learns invariant + equivariant features
# ---------------------------------------------------------------------------


@register_model(name="invariance_equivariance", training_mode="reconstruction")
class InvarianceEquivarianceGenerator(nn.Module, IGenerator):
    """Dual-branch: invariant features (global pool) + equivariant features (local)."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.invariant = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, bf),
            nn.GELU(),
            nn.Linear(bf, bf),
        )
        self.equivariant = nn.Sequential(_ConvBN(in_channels, bf), _ConvBN(bf, bf))
        self.fusion = nn.Sequential(_ConvBN(bf * 2, bf), nn.Conv2d(bf, out_channels, 1))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        inv = self.invariant(x).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        equi = self.equivariant(x)
        return self.fusion(torch.cat([inv, equi], dim=1))


# ---------------------------------------------------------------------------
# 12. NIHS — non-iterative half-Fourier synthesis
# ---------------------------------------------------------------------------


@register_model(name="nihs", training_mode="reconstruction")
class NIHSGenerator(nn.Module, IGenerator):
    """Non-Iterative Half-Fourier Synthesis for partial-Fourier reconstruction."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.net = nn.Sequential(
            _ConvBN(in_channels, bf),
            _ConvBN(bf, bf * 2),
            _ConvBN(bf * 2, bf * 2),
            _ConvBN(bf * 2, bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 13. Monai Swin UNETR — UNETR with Swin Transformer encoder
# ---------------------------------------------------------------------------


@register_model(name="monai_swin_unetr", training_mode="reconstruction")
class MonaiSwinUNETR(nn.Module, IGenerator):
    """UNETR: Transformer encoder + CNN decoder with skip connections."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, dim: int = 64, depth: int = 4, **kwargs
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 4, batch_first=True),
            num_layers=depth,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 4, stride=4),
            _ConvBN(dim // 2, dim // 2),
            nn.Conv2d(dim // 2, out_channels, 1),
        )
        self.skip_proj = nn.Conv2d(in_channels, dim // 2, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # TorchIO yields 5-D ``[B, C, H, W, D]``. Two cases:
        #   * D == 1 (2-D-sliced data) → squeeze the trailing singleton.
        #   * D > 1 (volumetric, e.g. ``baseline_swin_unetr_3d`` with
        #     ``patch_size: [H, W, 16]``) → fold depth into batch so the
        #     2-D backbone runs slice-by-slice, then unfold the output
        #     back to 5-D. This 2-D model has no cross-slice receptive
        #     field — drop in Conv3d/ConvTranspose3d for a true 3-D
        #     baseline.
        squeezed_depth: int | None = None
        while x.ndim > 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim == 5:
            B0, C, H, W, D = x.shape
            squeezed_depth = D
            # ``permute`` puts depth right after batch so reshape is contiguous.
            x = x.permute(0, 4, 1, 2, 3).reshape(B0 * D, C, H, W)
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        tokens = patches.flatten(2).permute(0, 2, 1)
        tokens = self.encoder(tokens)
        feat = tokens.permute(0, 2, 1).reshape(B, -1, pH, pW)
        dec = self.decoder(feat)
        skip = self.skip_proj(x)
        out = dec + F.interpolate(skip, size=dec.shape[2:], mode="bilinear", align_corners=False)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        if squeezed_depth is not None:
            B0 = B // squeezed_depth
            out = out.reshape(B0, squeezed_depth, out.shape[1], H, W).permute(0, 2, 3, 4, 1)
        return out


# ---------------------------------------------------------------------------
# 14. KIDOT — Knowledge-Informed Dynamic Optimal Transport generator
# ---------------------------------------------------------------------------


@register_model(name="kidot", training_mode="reconstruction")
class KIDOTGenerator(nn.Module, IGenerator):
    """Neural ODE transport generator: learns velocity field for domain translation.

    Wraps the KIDOT concept (Benamou-Brenier formulation) as an IGenerator.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 64,
        n_steps: int = 5,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.n_steps = n_steps
        # Velocity field network: predicts displacement at each Euler step
        self.velocity = nn.Sequential(
            nn.Conv2d(in_channels + 1, bf, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(bf, bf, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(bf, in_channels, 3, padding=1),
        )
        self.head = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            t_val = step * dt
            t_map = torch.full((h.shape[0], 1, h.shape[2], h.shape[3]), t_val, device=h.device)
            v = self.velocity(torch.cat([h, t_map], dim=1))
            h = h + dt * v
        return self.head(h)
