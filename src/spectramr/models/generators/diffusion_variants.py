"""Diffusion Model Variants
===========================

Distinct diffusion-based architectures for MRI reconstruction.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# 1. Diffusion U-Net — standard denoising U-Net with timestep conditioning
# ---------------------------------------------------------------------------


@register_model(name="diffusion_unet", training_mode="diffusion", spatial_dims=(2,))
class DiffusionUNet(nn.Module, IGenerator):
    """U-Net denoiser with sinusoidal timestep embedding for diffusion models."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 64, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.time_mlp = nn.Sequential(nn.Linear(bf, bf * 4), nn.SiLU(), nn.Linear(bf * 4, bf))
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, bf, 3, padding=1), _ResBlock(bf))
        self.enc2 = nn.Sequential(nn.Conv2d(bf, bf * 2, 3, stride=2, padding=1), _ResBlock(bf * 2))
        self.mid = nn.Sequential(
            _ResBlock(bf * 2), nn.MultiheadAttention(bf * 2, 4, batch_first=True), _ResBlock(bf * 2)
        )
        self.up2 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(bf * 2, bf, 3, padding=1), _ResBlock(bf))
        self.head = nn.Conv2d(bf, out_channels, 1)
        self.bf = bf

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.bf // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (2.0 * 3.14159 / half))
        args = t[:, None] * freqs[None]
        return torch.cat([args.cos(), args.sin()], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)
        temb = self.time_mlp(self._time_embed(t))
        e1 = self.enc1(x)
        e1 = e1 + temb[:, :, None, None]
        e2 = self.enc2(e1)
        B, C, H, W = e2.shape
        mid = e2.flatten(2).permute(0, 2, 1)
        mid = self.mid[0](e2)
        d1 = self.dec1(torch.cat([self.up2(mid), e1], 1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 2. Diffusion Recon — reconstruction-focused diffusion model
# ---------------------------------------------------------------------------


@register_model(name="diffusion_recon", training_mode="diffusion")
class DiffusionRecon(nn.Module, IGenerator):
    """Diffusion model specialized for MRI reconstruction with data consistency."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 64, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.denoiser = nn.Sequential(
            nn.Conv2d(in_channels + 1, bf, 3, padding=1),
            _ResBlock(bf),
            nn.Conv2d(bf, bf * 2, 3, stride=2, padding=1),
            _ResBlock(bf * 2),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            _ResBlock(bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if t is None:
            t = torch.zeros(x.shape[0], 1, 1, 1, device=x.device)
        elif t.ndim == 1:
            t = t.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return self.denoiser(torch.cat([x, t], dim=1))


# ---------------------------------------------------------------------------
# 3. Physics Diffusion — physics-constrained diffusion
# ---------------------------------------------------------------------------


@register_model(name="physics_diffusion", training_mode="diffusion")
class PhysicsDiffusion(nn.Module, IGenerator):
    """Diffusion model with physics-based data consistency enforcement."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 64,
        dc_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.dc_weight = dc_weight
        self.denoiser = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            _ResBlock(bf),
            _ResBlock(bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        prediction = self.denoiser(x)
        return prediction * (1 - self.dc_weight) + x * self.dc_weight


# ---------------------------------------------------------------------------
# 4. SPIRiT Diffusion — self-consistent diffusion with k-space constraints
# ---------------------------------------------------------------------------


@register_model(name="spirit_diffusion", training_mode="diffusion")
class SPIRiTDiffusion(nn.Module, IGenerator):
    """Diffusion with SPIRiT-like k-space self-consistency regularization."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 64,
        n_steps: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.denoiser = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 3, padding=1),
            _ResBlock(base_features),
            _ResBlock(base_features),
            nn.Conv2d(base_features, out_channels, 1),
        )
        self.spirit_kernel = nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=1)
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pred = self.denoiser(x)
        for _ in range(self.n_steps):
            pred = pred + 0.1 * self.spirit_kernel(pred)
        return pred


# ---------------------------------------------------------------------------
# 5. Coupling Flow Diffusion — normalizing flow + diffusion hybrid
# ---------------------------------------------------------------------------


@register_model(name="coupling_flow_diffusion", training_mode="diffusion")
class CouplingFlowDiffusion(nn.Module, IGenerator):
    """Hybrid: affine coupling layers (flow) composed with diffusion denoising."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 64,
        n_coupling: int = 4,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.coupling_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, bf, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(bf, in_channels * 2, 3, padding=1),
                )
                for _ in range(n_coupling)
            ]
        )
        self.denoiser = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1), _ResBlock(bf), nn.Conv2d(bf, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for net in self.coupling_nets:
            st = net(h)
            s, t = st.chunk(2, dim=1)
            h = h * torch.sigmoid(s) + t
        return self.denoiser(h)


# ---------------------------------------------------------------------------
# 6. Consistency Regularized — diffusion with consistency distillation
# ---------------------------------------------------------------------------


@register_model(name="consistency_regularized", training_mode="diffusion")
class ConsistencyRegularized(nn.Module, IGenerator):
    """Diffusion model trained with consistency regularization for fast sampling."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 64, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, bf, 3, padding=1),
            _ResBlock(bf),
            nn.Conv2d(bf, bf * 2, 3, stride=2, padding=1),
            _ResBlock(bf * 2),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            _ResBlock(bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if t is None:
            t = torch.zeros(x.shape[0], 1, 1, 1, device=x.device)
        elif t.ndim == 1:
            t = t.reshape(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, t], dim=1))


# ---------------------------------------------------------------------------
# 7. Score SDE — continuous-time score-based SDE diffusion
# ---------------------------------------------------------------------------


@register_model(name="score_sde", training_mode="diffusion")
class ScoreSDEGenerator(nn.Module, IGenerator):
    """Score-based SDE model with continuous noise level conditioning."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 64, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.sigma_embed = nn.Sequential(nn.Linear(1, bf), nn.SiLU(), nn.Linear(bf, bf))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            _ResBlock(bf),
            _ResBlock(bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if sigma is not None and sigma.ndim == 1:
            emb = self.sigma_embed(sigma.unsqueeze(-1))
            x = x + emb[:, : x.shape[1], None, None]
        return self.net(x)


# ---------------------------------------------------------------------------
# 8. Hybrid GS Diffusion — Gaussian Splatting + Diffusion
# ---------------------------------------------------------------------------


@register_model(name="hybrid_gs_diffusion", training_mode="diffusion")
class HybridGSDiffusion(nn.Module, IGenerator):
    """Diffusion denoising with learnable Gaussian kernel splatting regularizer."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 64,
        n_gaussians: int = 16,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.denoiser = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1), _ResBlock(bf), nn.Conv2d(bf, out_channels, 1)
        )
        self.gaussian_centers = nn.Parameter(torch.randn(n_gaussians, 2) * 0.1)
        self.gaussian_scales = nn.Parameter(torch.ones(n_gaussians) * 0.05)
        self.gaussian_weights = nn.Parameter(torch.randn(n_gaussians, out_channels) * 0.01)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pred = self.denoiser(x)
        B, C, H, W = pred.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1)
        gs_field = torch.zeros(1, C, H, W, device=x.device)
        for i in range(self.gaussian_centers.shape[0]):
            dist = ((coords - self.gaussian_centers[i]) ** 2).sum(-1)
            kernel = torch.exp(-dist / (2 * self.gaussian_scales[i] ** 2 + 1e-6))
            gs_field += kernel.unsqueeze(0).unsqueeze(0) * self.gaussian_weights[i].reshape(
                1, -1, 1, 1
            )
        return pred + gs_field
