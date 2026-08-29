"""Spatial Anatomy Network (SAN) for CPT-4DMR
=============================================

SIREN-based Implicit Neural Representation that encodes the canonical
(motion-free) anatomical volume at a reference cardiac/respiratory phase.

Key Features:
- Sine activation functions for better high-frequency learning
- Positional encoding for capturing fine details
- Resolution-independent: query at any spatial coordinate
"""

import math

import torch
import torch.nn as nn


class SineActivation(nn.Module):
    """Sine activation function with learnable frequency."""

    def __init__(self, omega_0: float = 30.0):
        """__init__.

        Args:
            omega_0 (float): Description.
        """
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SineActivation.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.sin(self.omega_0 * x)


class SIRENLayer(nn.Module):
    """Single SIREN layer with proper initialization.

    SIREN uses sine activations and requires special weight initialization
    to maintain signal magnitude through the network.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        omega_0: Frequency factor for sine activation
        is_first: Whether this is the first layer (different init)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            omega_0 (float): Description.
            is_first (bool): Description.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features)
        self.activation = SineActivation(omega_0)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in_features, 1/in_features]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/in_features)/omega_0, ...]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SIRENLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.activation(self.linear(x))


class SIREN(nn.Module):
    """Sinusoidal Representation Network (SIREN).

    A coordinate-based MLP with sine activations that can represent
    continuous signals with high-frequency details.

    Args:
        in_features: Input coordinate dimension (2 for 2D, 3 for 3D)
        hidden_features: Hidden layer dimension
        hidden_layers: Number of hidden layers
        out_features: Output dimension (1 for intensity, 3 for RGB)
        omega_0: Frequency factor for first layer
        omega_hidden: Frequency factor for hidden layers
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        out_features: int = 1,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            hidden_features (int): Description.
            hidden_layers (int): Description.
            out_features (int): Description.
            omega_0 (float): Description.
            omega_hidden (float): Description.
        """
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Build network
        layers = []

        # First layer
        layers.append(SIRENLayer(in_features, hidden_features, omega_0=omega_0, is_first=True))

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(
                SIRENLayer(
                    hidden_features,
                    hidden_features,
                    omega_0=omega_hidden,
                    is_first=False,
                )
            )

        self.layers = nn.Sequential(*layers)

        # Output layer (linear, no activation)
        self.output_layer = nn.Linear(hidden_features, out_features)
        self._init_output_weights()

    def _init_output_weights(self) -> None:
        """Initialize output layer weights."""
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.hidden_features) / 30.0
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            coords: Coordinates [..., in_features]

        Returns:
            Signal values [..., out_features]

        forward method for SIREN.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.layers(coords)
        return self.output_layer(x)


from mriforge.models.blocks.embeddings import (
    CoordinatePositionalEncoding as PositionalEncoding,
)


class SpatialAnatomyNet(nn.Module):
    """Spatial Anatomy Network for CPT-4DMR.

    Encodes the canonical anatomical volume using a SIREN network.
    This represents the heart/organ at a reference phase (e.g., end-expiration).

    Args:
        spatial_dim: Spatial dimension (2 for 2D slices, 3 for 3D volumes)
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        out_channels: Output channels (typically 1 for MRI)
        use_positional_encoding: Use Fourier positional encoding
        num_frequencies: Number of frequency bands for encoding
        omega_0: SIREN frequency parameter
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        out_channels: int = 1,
        use_positional_encoding: bool = True,
        num_frequencies: int = 6,
        omega_0: float = 30.0,
    ):
        """__init__.

        Args:
            spatial_dim (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
            out_channels (int): Description.
            use_positional_encoding (bool): Description.
            num_frequencies (int): Description.
            omega_0 (float): Description.
        """
        super().__init__()

        self.spatial_dim = spatial_dim
        self.out_channels = out_channels
        self.use_positional_encoding = use_positional_encoding

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(num_frequencies)
            input_dim = self.pos_encoding.output_dim(spatial_dim)
        else:
            self.pos_encoding = None
            input_dim = spatial_dim

        # SIREN backbone
        self.siren = SIREN(
            in_features=input_dim,
            hidden_features=hidden_dim,
            hidden_layers=num_layers,
            out_features=out_channels,
            omega_0=omega_0,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Query anatomical intensity at given coordinates.

        Args:
            coords: Spatial coordinates [B, N, D] or [B, H, W, D]
                    Values should be normalized to [-1, 1]

        Returns:
            Intensity values [B, N, C] or [B, H, W, C]

        forward method for SpatialAnatomyNet.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        input_shape = coords.shape[:-1]
        D = coords.shape[-1]

        # Flatten to [B*..., D] - use reshape for non-contiguous tensors
        coords_flat = coords.reshape(-1, D)

        # Apply positional encoding
        if self.pos_encoding is not None:
            coords_flat = self.pos_encoding(coords_flat)

        # Query SIREN
        out = self.siren(coords_flat)

        # Reshape to original spatial dimensions
        return out.reshape(*input_shape, self.out_channels)

    def generate_grid(
        self,
        resolution: tuple[int, ...],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Generate a coordinate grid for reconstruction.

        Args:
            resolution: Output resolution (H, W) or (D, H, W)
            device: Target device

        Returns:
            Coordinate grid [1, *resolution, spatial_dim]
        """
        device = device or next(self.parameters()).device

        # Create normalized coordinates [-1, 1]
        axes = [torch.linspace(-1, 1, r, device=device) for r in resolution]

        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return grid.unsqueeze(0)  # Add batch dimension
