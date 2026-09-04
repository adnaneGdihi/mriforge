import torch.nn as nn

from spectramr.models.registry import register_model

from .wavelet_kan import WaveletKANLinear


@register_model(name="implicit_kan", training_mode="experimental")
class ImplicitKAN(nn.Module):
    """
    Paradigm 4.2: KAN-based Implicit Neural Representation.
    Maps continuous spatial coordinates to signal intensity.
    Useful for continuous-resolution MRI reconstruction.
    """

    def __init__(self, spatial_dim=3, hidden_dim=64, num_layers=4, out_channels=1):
        """__init__.

        Args:
            spatial_dim (Any): Description.
            hidden_dim (Any): Description.
            num_layers (Any): Description.
            out_channels (Any): Description.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim

        layers = []
        # Input layer
        layers.append(WaveletKANLinear(spatial_dim, hidden_dim))
        layers.append(nn.SiLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(WaveletKANLinear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        # Output layer
        layers.append(WaveletKANLinear(hidden_dim, out_channels))

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        """
        Args:
            coords: [B, N, D] or [B, D, H, W] - Batch of spatial coordinates
        Returns:
            [B, N, out_channels] or [B, out_channels, H, W] - Signal intensity
        """
        is_2d_grid = coords.ndim == 4
        if is_2d_grid:
            B, D, H, W = coords.shape
            # [B, D, H, W] -> [B, D, H*W] -> [B, H*W, D]
            coords = coords.view(B, D, -1).transpose(1, 2)

        B, N, D = coords.shape
        x = coords.view(-1, D)

        out = self.net(x)
        out = out.view(B, N, -1)

        if is_2d_grid:
            # [B, H*W, C] -> [B, C, H*W] -> [B, C, H, W]
            out = out.transpose(1, 2).view(B, -1, H, W)

        return out


from spectramr.models.blocks.embeddings import (
    CoordinatePositionalEncoding as PositionalEncoding,
)


@register_model(name="implicit_mri_field", training_mode="experimental")
class ImplicitMRIField(nn.Module):
    """
    Full INR model for MRI with positional encoding.
    Can reconstruct at arbitrary resolution.
    """

    def __init__(self, spatial_dim=2, hidden_dim=128, num_layers=4, num_frequencies=6):
        """__init__.

        Args:
            spatial_dim (Any): Description.
            hidden_dim (Any): Description.
            num_layers (Any): Description.
            num_frequencies (Any): Description.
        """
        super().__init__()
        # CoordinatePositionalEncoding expects [..., D] format for encoding.
        # We will handle the shape inside forward.
        self.encoding = PositionalEncoding(num_frequencies)
        encoded_dim = spatial_dim + 2 * spatial_dim * num_frequencies
        self.net = ImplicitKAN(encoded_dim, hidden_dim, num_layers)

    def forward(self, coords):
        """forward.

        Args:
            coords: [B, D, H, W] or [B, N, D]
        """
        is_2d_grid = coords.ndim == 4
        if is_2d_grid:
            B, D, H, W = coords.shape
            # [B, D, H, W] -> [B, H, W, D] for encoding
            coords_reshaped = coords.permute(0, 2, 3, 1)
        else:
            coords_reshaped = coords

        encoded = self.encoding(coords_reshaped)

        if is_2d_grid:
            # [B, H, W, encoded_D] -> [B, encoded_D, H, W]
            encoded = encoded.permute(0, 3, 1, 2)

        return self.net(encoded)
