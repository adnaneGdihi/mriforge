"""Log-Polar k-Space Hash Grid (k-Hash) for Efficient k-Space INR.

Innovation VIII from SOTA Roadmap: Memory-efficient k-space representation.

Key Insight:
    k-space energy follows 1/f distribution - most energy is at center (DC).
    Uniform hash grids waste resolution on high-frequency regions with little signal.
    Log-polar mapping allocates more hash entries where energy is highest.

Mathematical Foundation:
    Transform: (k_x, k_y) → (log(ρ), θ)
        ρ = sqrt(k_x² + k_y²)
        θ = atan2(k_y, k_x)

    Multi-resolution hash encoding (InstantNGP-style):
        γ(x) = [h₁(x), h₂(x), ..., h_L(x)]

    where h_l are hash table lookups at different resolutions.

Benefits:
    - O(1) memory per k-space sample (vs O(N²) for grid)
    - Resolution matches energy distribution
    - Fast training/inference via hash table

References:
    - [33] InstantNGP multi-resolution hash
    - [20] Implicit k-space representations

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectramr.models.registry import register_model


class MultiResolutionHashEncoder(nn.Module):
    """Multi-resolution hash encoding for continuous coordinates.

    Based on InstantNGP, encodes coordinates using multiple hash tables
    at different spatial resolutions.

    Args:
        input_dim: Input coordinate dimension
        num_levels: Number of resolution levels
        level_dim: Feature dimension per level
        base_resolution: Resolution at level 0
        finest_resolution: Resolution at highest level
        hash_table_size: Size of each hash table (power of 2)
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_levels: int = 16,
        level_dim: int = 2,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        hash_table_size: int = 2**14,
    ) -> None:
        """__init__.

        Args:
            input_dim (int): Description.
            num_levels (int): Description.
            level_dim (int): Description.
            base_resolution (int): Description.
            finest_resolution (int): Description.
            hash_table_size (int): Description.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.hash_table_size = hash_table_size

        # Compute resolution at each level (geometric progression)
        self.per_level_scale = (finest_resolution / base_resolution) ** (1 / (num_levels - 1))
        self.resolutions = [
            int(base_resolution * (self.per_level_scale**l)) for l in range(num_levels)
        ]

        # Hash tables (one per level)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(hash_table_size, level_dim) for _ in range(num_levels)]
        )

        # Initialize with small values
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

        # Prime numbers for hashing
        self.primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode coordinates using multi-resolution hashing.

        Args:
            x: Coordinates (batch, input_dim) in [0, 1]

        Returns:
            Encoded features (batch, num_levels * level_dim)

        forward method for MultiResolutionHashEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        outputs = []

        for level, (resolution, embedding) in enumerate(
            zip(self.resolutions, self.embeddings, strict=False)
        ):
            # Scale coordinates to grid
            scaled = x * resolution

            # Get corner indices
            corner_indices, weights = self._get_interpolation(scaled, resolution)

            # Hash and lookup
            features = self._hash_lookup(corner_indices, embedding)

            # Interpolate
            interpolated = self._interpolate(features, weights)
            outputs.append(interpolated)

        return torch.cat(outputs, dim=-1)

    def _get_interpolation(
        self,
        scaled: torch.Tensor,
        resolution: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get corner indices and interpolation weights for bilinear interpolation."""
        floor = torch.floor(scaled).long()
        ceil = floor + 1
        weights = scaled - floor.float()

        # Clamp to valid range
        floor = torch.clamp(floor, 0, resolution - 1)
        ceil = torch.clamp(ceil, 0, resolution - 1)

        # All 4 corners for 2D (or 8 for 3D)
        if self.input_dim == 2:
            corners = torch.stack(
                [
                    torch.stack([floor[:, 0], floor[:, 1]], dim=-1),  # (0,0)
                    torch.stack([ceil[:, 0], floor[:, 1]], dim=-1),  # (1,0)
                    torch.stack([floor[:, 0], ceil[:, 1]], dim=-1),  # (0,1)
                    torch.stack([ceil[:, 0], ceil[:, 1]], dim=-1),  # (1,1)
                ],
                dim=1,
            )  # (batch, 4, 2)
        else:
            # Simplified for 2D only in this implementation
            raise NotImplementedError("Only 2D supported")

        return corners, weights

    def _hash_lookup(
        self,
        indices: torch.Tensor,
        embedding: nn.Embedding,
    ) -> torch.Tensor:
        """Hash coordinates and lookup in embedding table."""
        batch, num_corners, dim = indices.shape

        # Spatial hashing
        primes = self.primes[:dim].to(indices.device)
        hashed = (indices * primes).sum(dim=-1) % self.hash_table_size

        # Lookup
        return embedding(hashed)  # (batch, num_corners, level_dim)

    def _interpolate(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinear interpolation of corner features."""
        # weights: (batch, 2) for x and y
        w_x, w_y = weights[:, 0:1], weights[:, 1:2]

        # Bilinear weights for 4 corners
        w00 = (1 - w_x) * (1 - w_y)
        w10 = w_x * (1 - w_y)
        w01 = (1 - w_x) * w_y
        w11 = w_x * w_y

        weights_4 = torch.stack([w00, w10, w01, w11], dim=1)  # (batch, 4, 1)

        return (features * weights_4).sum(dim=1)


class LogPolarTransform(nn.Module):
    """Transform k-space coordinates to log-polar.

    Maps (k_x, k_y) → (log(ρ), θ/2π) ∈ [0, 1]²
    """

    def __init__(self, rho_min: float = 1e-3, rho_max: float = 1.0) -> None:
        """__init__.

        Args:
            rho_min (float): Description.
            rho_max (float): Description.
        """
        super().__init__()
        self.log_rho_min = math.log(rho_min)
        self.log_rho_max = math.log(rho_max)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Transform to log-polar.

        Args:
            k: k-space coordinates (batch, 2) in [-1, 1]

        Returns:
            Log-polar coordinates (batch, 2) in [0, 1]

        forward method for LogPolarTransform.

        Executes PyTorch tensor operations.

        Args:
            k (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        rho = torch.sqrt(k[:, 0] ** 2 + k[:, 1] ** 2).clamp(min=1e-6)
        theta = torch.atan2(k[:, 1], k[:, 0])

        # Normalize to [0, 1]
        log_rho = (torch.log(rho) - self.log_rho_min) / (self.log_rho_max - self.log_rho_min)
        log_rho = log_rho.clamp(0, 1)

        theta_norm = (theta + math.pi) / (2 * math.pi)

        return torch.stack([log_rho, theta_norm], dim=-1)


@register_model(name="k_hash", training_mode="reconstruction")
class KSpaceHashINR(nn.Module):
    """k-Space INR with log-polar hash encoding.

    Represents k-space as a continuous function using log-polar
    coordinate transform and multi-resolution hashing.

    Args:
        num_levels: Hash encoding levels
        level_dim: Features per level
        hidden_dims: MLP hidden dimensions
        out_channels: Output channels (2 for complex)
    """

    def __init__(
        self,
        num_levels: int = 16,
        level_dim: int = 2,
        hidden_dims: tuple[int, ...] = (64, 64),
        out_channels: int = 2,
    ) -> None:
        """__init__.

        Args:
            num_levels (int): Description.
            level_dim (int): Description.
            hidden_dims (tuple[int, ...]): Description.
            out_channels (int): Description.
        """
        super().__init__()

        # Log-polar transform
        self.log_polar = LogPolarTransform()

        # Hash encoder
        self.encoder = MultiResolutionHashEncoder(
            input_dim=2,
            num_levels=num_levels,
            level_dim=level_dim,
        )

        # MLP decoder
        encoded_dim = num_levels * level_dim
        layers = []
        in_dim = encoded_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, out_channels))

        self.decoder = nn.Sequential(*layers)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Evaluate k-space at given coordinates.

        Args:
            k: k-space coordinates (batch, 2) in [-1, 1]

        Returns:
            k-space values (batch, out_channels)

        forward method for KSpaceHashINR.

        Executes PyTorch tensor operations.

        Args:
            k (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Transform to log-polar
        k_lp = self.log_polar(k)

        # Hash encode
        encoded = self.encoder(k_lp)

        # Decode to k-space value
        return self.decoder(encoded)

    def render_kspace(
        self,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Render full k-space grid.

        Args:
            H, W: Grid dimensions
            device: Torch device

        Returns:
            k-space (1, out_channels, H, W)
        """
        # Create coordinate grid
        ky = torch.linspace(-1, 1, H, device=device)
        kx = torch.linspace(-1, 1, W, device=device)
        grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing="ij")

        coords = torch.stack([grid_kx.flatten(), grid_ky.flatten()], dim=-1)

        # Evaluate
        values = self(coords)

        # Reshape
        return values.reshape(1, H, W, -1).permute(0, 3, 1, 2)


def create_kspace_hash(
    num_levels: int = 16,
    level_dim: int = 2,
    hidden_dims: tuple[int, ...] = (64, 64),
) -> KSpaceHashINR:
    """Factory function for k-Hash INR.

    Args:
        num_levels: Hash encoding levels
        level_dim: Features per level
        hidden_dims: MLP dimensions

    Returns:
        Configured k-Hash INR model
    """
    return KSpaceHashINR(
        num_levels=num_levels,
        level_dim=level_dim,
        hidden_dims=hidden_dims,
    )


__all__ = [
    "KSpaceHashINR",
    "LogPolarTransform",
    "MultiResolutionHashEncoder",
    "create_kspace_hash",
]
