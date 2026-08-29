"""Neural Field & INR Variants
===============================

Implicit neural representation architectures for continuous MRI reconstruction.
"""

import math

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

# ---------------------------------------------------------------------------
# 1. Implicit Representation — SIREN-based continuous MRI field
# ---------------------------------------------------------------------------


class _SirenLayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, omega: float = 30.0, is_first: bool = False
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega
        bound = 1 / in_features if is_first else math.sqrt(6 / in_features) / omega
        nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


@register_model(name="implicit_representation", training_mode="reconstruction")
class ImplicitRepresentationGenerator(nn.Module, IGenerator):
    """SIREN-based implicit representation that maps (x, y) coordinates to pixel values."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 256,
        depth: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        layers = [_SirenLayer(2, hidden_dim, is_first=True)]
        for _ in range(depth - 2):
            layers.append(_SirenLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_channels))
        self.net = nn.Sequential(*layers)

    def _make_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
        return grid.reshape(-1, 2)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        coords = self._make_grid(H, W, x.device).unsqueeze(0).expand(B, -1, -1)
        values = self.net(coords)
        return values.permute(0, 2, 1).reshape(B, self.out_channels, H, W)


# ---------------------------------------------------------------------------
# 2. InFusion — implicit fusion of multi-scale features
# ---------------------------------------------------------------------------


@register_model(name="infusion", training_mode="reconstruction")
class InFusionGenerator(nn.Module, IGenerator):
    """Feature-conditioned INR: SIREN layers modulated by CNN feature maps."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, hidden_dim: int = 128, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
        )
        self.inr = nn.Sequential(
            _SirenLayer(2 + hidden_dim, hidden_dim, is_first=True),
            _SirenLayer(hidden_dim, hidden_dim),
            _SirenLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        features = self.feature_net(x)  # (B, hidden_dim, H, W)
        y = torch.linspace(-1, 1, H, device=x.device)
        xc = torch.linspace(-1, 1, W, device=x.device)
        grid = (
            torch.stack(torch.meshgrid(y, xc, indexing="ij"), dim=-1)
            .reshape(1, H * W, 2)
            .expand(B, -1, -1)
        )
        feat_flat = features.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)
        inp = torch.cat([grid, feat_flat], dim=-1)
        values = self.inr(inp)
        return values.permute(0, 2, 1).reshape(B, self.out_channels, H, W)


# ---------------------------------------------------------------------------
# 3. Motion INR — temporal-aware INR for motion-resolved MRI
# ---------------------------------------------------------------------------


@register_model(name="motion_inr", training_mode="reconstruction")
class MotionINRGenerator(nn.Module, IGenerator):
    """INR conditioned on spatial + temporal coordinates for motion-resolved MRI."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, hidden_dim: int = 256, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        # 3D input: (x, y, t)
        self.net = nn.Sequential(
            _SirenLayer(3, hidden_dim, is_first=True),
            _SirenLayer(hidden_dim, hidden_dim),
            _SirenLayer(hidden_dim, hidden_dim),
            _SirenLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x: torch.Tensor, t: float = 0.0, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        y_coords = torch.linspace(-1, 1, H, device=x.device)
        x_coords = torch.linspace(-1, 1, W, device=x.device)
        grid = torch.stack(torch.meshgrid(y_coords, x_coords, indexing="ij"), dim=-1).reshape(-1, 2)
        t_coord = torch.full((grid.shape[0], 1), t, device=x.device)
        coords = torch.cat([grid, t_coord], dim=-1).unsqueeze(0).expand(B, -1, -1)
        values = self.net(coords)
        return values.permute(0, 2, 1).reshape(B, self.out_channels, H, W)


# ---------------------------------------------------------------------------
# 4. Neural Process — conditional neural process for few-shot reconstruction
# ---------------------------------------------------------------------------


@register_model(name="neural_process", training_mode="reconstruction")
class NeuralProcessGenerator(nn.Module, IGenerator):
    """Conditional Neural Process: aggregates context set into a global latent."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.context_encoder = nn.Sequential(
            nn.Linear(2 + in_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.aggregator = nn.Linear(latent_dim, latent_dim)  # mean aggregation + projection
        self.decoder = nn.Sequential(
            nn.Linear(2 + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        y = torch.linspace(-1, 1, H, device=x.device)
        xc = torch.linspace(-1, 1, W, device=x.device)
        grid = (
            torch.stack(torch.meshgrid(y, xc, indexing="ij"), dim=-1)
            .reshape(1, H * W, 2)
            .expand(B, -1, -1)
        )
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        context = torch.cat([grid, x_flat], dim=-1)
        r = self.context_encoder(context).mean(dim=1, keepdim=True)  # (B, 1, latent_dim)
        r = self.aggregator(r).expand(-1, H * W, -1)
        target_input = torch.cat([grid, r], dim=-1)
        values = self.decoder(target_input)
        return values.permute(0, 2, 1).reshape(B, self.out_channels, H, W)


# ---------------------------------------------------------------------------
# 5. PIN-NeRF — physics-informed neural radiance field for MRI
# ---------------------------------------------------------------------------


@register_model(name="pin_nerf", training_mode="reconstruction")
class PINNeRFGenerator(nn.Module, IGenerator):
    """Physics-Informed NeRF: combines positional encoding with physics constraints."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 256,
        n_freqs: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.n_freqs = n_freqs
        input_dim = 2 * (2 * n_freqs + 1)  # positional encoding of 2D coords
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def _positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(self.n_freqs, device=coords.device) * math.pi
        enc = [coords]
        for f in freqs:
            enc.extend([torch.sin(f * coords), torch.cos(f * coords)])
        return torch.cat(enc, dim=-1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        y = torch.linspace(-1, 1, H, device=x.device)
        xc = torch.linspace(-1, 1, W, device=x.device)
        grid = (
            torch.stack(torch.meshgrid(y, xc, indexing="ij"), dim=-1)
            .reshape(1, -1, 2)
            .expand(B, -1, -1)
        )
        encoded = self._positional_encoding(grid)
        values = self.net(encoded)
        return values.permute(0, 2, 1).reshape(B, self.out_channels, H, W)
