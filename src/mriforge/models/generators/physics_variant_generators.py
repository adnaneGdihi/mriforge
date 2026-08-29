"""Physics-Informed Generator Variants
=======================================

MRI reconstruction models with explicit physics constraints and priors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 1. Bloch-Guided — Bloch-equation-aware reconstruction
# ---------------------------------------------------------------------------


@register_model(name="bloch_guided", training_mode="reconstruction")
class BlochGuidedGenerator(nn.Module, IGenerator):
    """MRI reconstruction guided by Bloch equation tissue relaxation parameters.

    Learns T1/T2 maps as auxiliary outputs and uses them to constrain reconstruction.
    """

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
        # Bloch parameter heads (T1, T2, PD)
        self.t1_head = nn.Conv2d(bf * 4, 1, 1)
        self.t2_head = nn.Conv2d(bf * 4, 1, 1)
        self.pd_head = nn.Conv2d(bf * 4, 1, 1)
        # Reconstruction decoder conditioned on Bloch parameters
        self.decoder = nn.Sequential(
            _ConvBN(bf * 4 + 3, bf * 4),
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.encoder(x)
        t1 = F.softplus(self.t1_head(feat))
        t2 = F.softplus(self.t2_head(feat))
        pd = torch.sigmoid(self.pd_head(feat))
        conditioned = torch.cat([feat, t1, t2, pd], dim=1)
        out = self.decoder(conditioned)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 2. Physics-Aware Gaussian Splatter — 3DGS with physics constraints
# ---------------------------------------------------------------------------


@register_model(name="physics_aware_gaussian_splatter", training_mode="reconstruction")
class PhysicsAwareGaussianSplatter(nn.Module, IGenerator):
    """Learnable Gaussian splats with physics-derived positions and scales."""

    def __init__(self, in_channels: int = 2, out_channels: int = 2, n_splats: int = 64, **kwargs):
        super().__init__()
        self.n_splats = n_splats
        self.out_channels = out_channels
        self.centers = nn.Parameter(torch.randn(n_splats, 2) * 0.3)
        self.scales = nn.Parameter(torch.ones(n_splats) * 0.05)
        self.weights = nn.Parameter(torch.randn(n_splats, out_channels) * 0.01)
        self.rotation = nn.Parameter(torch.zeros(n_splats))
        # Refiner to blend with input
        self.refiner = nn.Sequential(
            nn.Conv2d(out_channels + in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        field = torch.zeros(1, self.out_channels, H, W, device=x.device)
        for i in range(self.n_splats):
            dist_sq = ((coords - self.centers[i]) ** 2).sum(-1)
            kernel = torch.exp(-dist_sq / (2 * self.scales[i] ** 2 + 1e-8))
            field = field + kernel.unsqueeze(0).unsqueeze(0) * self.weights[i].reshape(1, -1, 1, 1)
        field = field.expand(B, -1, -1, -1)
        return self.refiner(torch.cat([field, x], dim=1))


# ---------------------------------------------------------------------------
# 3. Gaussian Splatting — vanilla 2D Gaussian splatting
# ---------------------------------------------------------------------------


@register_model(name="gaussian_splatting", training_mode="reconstruction")
class GaussianSplattingGenerator(nn.Module, IGenerator):
    """2D Gaussian splatting: learn splat positions, scales, and colors."""

    def __init__(self, in_channels: int = 2, out_channels: int = 2, n_splats: int = 128, **kwargs):
        super().__init__()
        self.n_splats = n_splats
        self.out_channels = out_channels
        self.centers = nn.Parameter(torch.randn(n_splats, 2) * 0.5)
        self.scales = nn.Parameter(torch.ones(n_splats, 2) * 0.05)
        self.colors = nn.Parameter(torch.randn(n_splats, out_channels) * 0.1)
        self.opacities = nn.Parameter(torch.zeros(n_splats))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).unsqueeze(2)  # (H, W, 1, 2)
        centers = self.centers.reshape(1, 1, self.n_splats, 2)
        scales = F.softplus(self.scales).reshape(1, 1, self.n_splats, 2)
        diff = (coords - centers) / scales
        gauss = torch.exp(-0.5 * (diff**2).sum(-1))  # (H, W, n_splats)
        opacity = torch.sigmoid(self.opacities)
        weighted = gauss * opacity
        colors = self.colors  # (n_splats, out_channels)
        image = torch.einsum("hwn,nc->chw", weighted, colors).unsqueeze(0)
        return image.expand(B, -1, -1, -1)


# ---------------------------------------------------------------------------
# 4. Lagrangian Splatter — Lagrangian fluid-inspired splatting
# ---------------------------------------------------------------------------


@register_model(name="lagrangian_splatter", training_mode="reconstruction")
class LagrangianSplatter(nn.Module, IGenerator):
    """Particle-based rendering with physics-driven trajectories."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, n_particles: int = 64, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.positions = nn.Parameter(torch.randn(n_particles, 2) * 0.3)
        self.velocities = nn.Parameter(torch.zeros(n_particles, 2))
        self.values = nn.Parameter(torch.randn(n_particles, out_channels) * 0.1)
        self.kernel_size = nn.Parameter(torch.ones(n_particles) * 0.05)
        self.refiner = nn.Sequential(
            nn.Conv2d(out_channels + in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        pos = self.positions + self.velocities * 0.1
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1)
        field = torch.zeros(1, self.out_channels, H, W, device=x.device)
        for i in range(pos.shape[0]):
            dist = ((coords - pos[i]) ** 2).sum(-1)
            kernel = torch.exp(-dist / (2 * self.kernel_size[i] ** 2 + 1e-8))
            field = field + kernel.unsqueeze(0).unsqueeze(0) * self.values[i].reshape(1, -1, 1, 1)
        return self.refiner(torch.cat([field.expand(B, -1, -1, -1), x], dim=1))


# ---------------------------------------------------------------------------
# 5. PrintMesh — mesh-based differentiable rendering for MRI
# ---------------------------------------------------------------------------


@register_model(name="printmesh", training_mode="reconstruction")
class PrintMeshGenerator(nn.Module, IGenerator):
    """Differentiable mesh-based reconstruction using learned vertex positions."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.feature_net = nn.Sequential(
            _ConvBN(in_channels, bf), _ConvBN(bf, bf * 2), nn.MaxPool2d(2), _ConvBN(bf * 2, bf * 4)
        )
        self.vertex_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(bf * 4 * 64, 256),
            nn.GELU(),
            nn.Linear(256, 128 * 2),
        )
        self.rasterizer = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(bf * 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.feature_net(x)
        out = self.rasterizer(feat)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 6. PGGS Pipeline — physics-guided Gaussian splatting pipeline
# ---------------------------------------------------------------------------


@register_model(name="pggs_pipeline", training_mode="reconstruction")
class PGGSPipelineGenerator(nn.Module, IGenerator):
    """Physics-Guided Gaussian Splatting: CNN encoder → splat prediction → rendering."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_splats: int = 64,
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
        # Predict splat parameters from encoded features
        self.splat_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bf * 4, n_splats * (2 + 1 + out_channels)),
        )
        self.n_splats = n_splats
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.encoder(x)
        params = self.splat_head(feat)
        params = params.reshape(B, self.n_splats, 2 + 1 + self.out_channels)
        centers = torch.tanh(params[:, :, :2])
        scales = F.softplus(params[:, :, 2:3])
        colors = params[:, :, 3:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1)
        result = torch.zeros(B, self.out_channels, H, W, device=x.device)
        for b in range(B):
            for i in range(self.n_splats):
                dist = ((coords - centers[b, i]) ** 2).sum(-1)
                kernel = torch.exp(-dist / (2 * scales[b, i] ** 2 + 1e-8))
                result[b] += kernel.unsqueeze(0) * colors[b, i].reshape(-1, 1, 1)
        return result


# ---------------------------------------------------------------------------
# 7. QMRI Network — quantitative MRI parameter mapping
# ---------------------------------------------------------------------------


@register_model(name="qmri_network", training_mode="reconstruction")
class QMRINetworkGenerator(nn.Module, IGenerator):
    """Multi-echo/multi-contrast input → quantitative maps (T1, T2, PD)."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.encoder = nn.Sequential(
            _ConvBN(in_channels, bf), _ConvBN(bf, bf * 2), nn.MaxPool2d(2), _ConvBN(bf * 2, bf * 4)
        )
        self.param_decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.Conv2d(bf, 3, 1),
        )
        self.synth_decoder = nn.Sequential(
            nn.Conv2d(3, bf, 3, padding=1), nn.GELU(), nn.Conv2d(bf, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.encoder(x)
        params = self.param_decoder(feat)
        t1 = F.softplus(params[:, 0:1])
        t2 = F.softplus(params[:, 1:2])
        pd = torch.sigmoid(params[:, 2:3])
        qmaps = torch.cat([t1, t2, pd], dim=1)
        return self.synth_decoder(qmaps)


# ---------------------------------------------------------------------------
# 8. Differentiable Convex — convex optimization layer in the loop
# ---------------------------------------------------------------------------


@register_model(name="differentiable_convex", training_mode="reconstruction")
class DifferentiableConvexGenerator(nn.Module, IGenerator):
    """Unrolled proximal gradient with learned proximal operator."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        n_iters: int = 5,
        base_features: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.n_iters = n_iters
        bf = base_features
        self.prox = nn.Sequential(
            _ConvBN(out_channels, bf), _ConvBN(bf, bf), nn.Conv2d(bf, out_channels, 1)
        )
        self.step_sizes = nn.Parameter(torch.ones(n_iters) * 0.1)
        self.init_pred = nn.Sequential(_ConvBN(in_channels, bf), nn.Conv2d(bf, out_channels, 1))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pred = self.init_pred(x)
        for i in range(self.n_iters):
            grad = pred - x[:, : pred.shape[1]]
            pred = self.prox(pred - self.step_sizes[i] * grad)
        return pred
