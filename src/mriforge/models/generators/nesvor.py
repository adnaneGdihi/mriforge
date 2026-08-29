"""NeSVoR: Neural Slice-to-Volume Reconstruction.

Implicit Neural Representation for Fetal MRI.
f(x, y, z, t) -> Intensity
"""

import torch
import torch.nn as nn

from mriforge.models.registry import register_model


@register_model(name="nesvor", training_mode="reconstruction")
class NeSVoR(nn.Module):
    """NeSVoR Implicit Network."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 256,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            hidden_dim (int): Description.
        """
        super().__init__()
        # Blueprint: Hash Encoding (InstantNGP style) or Siren layers recommended for speed
        # But for basics, we use linear layers for encoding as per snippet.
        # "self.spatial_encoding = nn.Linear(3, hidden_dim)"

        # Coordinate encoding
        self.spatial_encoding = nn.Linear(3, hidden_dim)
        self.time_encoding = nn.Linear(1, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Added depth for better capacity
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels),  # Output Intensity
        )

    def forward(
        self, coords: torch.Tensor, time: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass.

        Supports two calling conventions:
        1. Coordinate-based: coords=(B, N, 3), time=(B, 1)
        2. Image-based (pipeline): coords=(B, C, H, W) — generates grid internally.

        Args:
            coords: 3D coordinates (B, N, 3) or image tensor (B, C, H, W).
            time: Time index (B, 1). Defaults to zeros.
        """
        # If coords is a 4D image tensor, convert to coordinate grid
        if coords.ndim == 4:
            B, C, H, W = coords.shape
            # Create normalized grid [-1, 1]
            gy = torch.linspace(-1, 1, H, device=coords.device)
            gx = torch.linspace(-1, 1, W, device=coords.device)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            grid_z = torch.zeros_like(grid_x)
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (H, W, 3)
            grid = grid.reshape(1, H * W, 3).expand(B, -1, -1)  # (B, N, 3)
            coords = grid
            if time is None:
                time = torch.zeros(B, 1, device=coords.device)
            # Return as image: (B, out_channels, H, W)
            sp_emb = self.spatial_encoding(coords)
            t_emb = self.time_encoding(time).unsqueeze(1)
            feat = sp_emb + t_emb
            out = self.mlp(feat)  # (B, N, out_channels)
            return out.permute(0, 2, 1).reshape(B, -1, H, W)

        if time is None:
            time = torch.zeros(coords.shape[0], 1, device=coords.device)

        sp_emb = self.spatial_encoding(coords)  # (B, N, H)
        t_emb = self.time_encoding(time).unsqueeze(1)  # (B, 1, H)

        # Feature combination
        feat = sp_emb + t_emb

        # MLP
        return self.mlp(feat)
