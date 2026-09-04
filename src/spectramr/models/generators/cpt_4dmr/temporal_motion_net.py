"""Temporal Motion Network (TMN) for CPT-4DMR
=============================================

MLP that predicts Deformation Vector Fields (DVF) conditioned on:
- Spatial coordinates (x, y, z)
- Time (cardiac/respiratory phase)
- Respiratory surrogate signal

The DVF warps the canonical anatomy to match the current motion state.

Key constraint: Jacobian determinant penalty for diffeomorphism.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMotionNet(nn.Module):
    """Temporal Motion Network for CPT-4DMR.

    Predicts a continuous Deformation Vector Field (DVF) that warps
    the canonical anatomy to match the current motion state.

    DVF(x, t, resp) = displacement to add to coordinates

    Args:
        spatial_dim: Spatial dimension (2 or 3)
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        time_embed_dim: Dimension of time embedding
        use_respiratory: Whether to condition on respiratory signal
        max_displacement: Maximum displacement magnitude (for clipping)
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_embed_dim: int = 32,
        use_respiratory: bool = True,
        max_displacement: float = 0.5,
    ):
        """__init__.

        Args:
            spatial_dim (int): Description.
            hidden_dim (int): Description.
            num_layers (int): Description.
            time_embed_dim (int): Description.
            use_respiratory (bool): Description.
            max_displacement (float): Description.
        """
        super().__init__()

        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.use_respiratory = use_respiratory
        self.max_displacement = max_displacement

        # Time embedding (sinusoidal)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Respiratory embedding (if used)
        if use_respiratory:
            self.resp_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            conditioning_dim = time_embed_dim * 2
        else:
            self.resp_embed = None
            conditioning_dim = time_embed_dim

        # Input: spatial coords + conditioning
        input_dim = spatial_dim + conditioning_dim

        # Build MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        self.backbone = nn.Sequential(*layers)

        # Output: DVF (same dimension as spatial)
        self.dvf_head = nn.Linear(hidden_dim, spatial_dim)

        # Initialize output to near-zero for identity transform
        self._init_output()

    def _init_output(self) -> None:
        """Initialize output layer for near-identity transform."""
        with torch.no_grad():
            self.dvf_head.weight.fill_(0.0)
            self.dvf_head.bias.fill_(0.0)

    def forward(
        self,
        coords: torch.Tensor,
        time: torch.Tensor,
        resp_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict deformation vector field.

        Args:
            coords: Spatial coordinates [B, N, D]
            time: Normalized time [B] or [B, 1], range [0, 1]
            resp_signal: Respiratory surrogate [B] or [B, 1], optional

        Returns:
            DVF: Displacement field [B, N, D]

        forward method for TemporalMotionNet.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            resp_signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, D = coords.shape
        device = coords.device

        # Ensure time is [B, 1]
        if time.dim() == 1:
            time = time.unsqueeze(1)

        # Time embedding [B, time_embed_dim]
        time_emb = self.time_embed(time)

        # Respiratory embedding (use zeros if not provided but network expects it)
        if self.use_respiratory:
            if resp_signal is not None:
                if resp_signal.dim() == 1:
                    resp_signal = resp_signal.unsqueeze(1)
            else:
                # Provide zeros if resp_signal not given
                resp_signal = torch.zeros(B, 1, device=device)
            resp_emb = self.resp_embed(resp_signal)
            conditioning = torch.cat([time_emb, resp_emb], dim=1)  # [B, 2*time_embed_dim]
        else:
            conditioning = time_emb  # [B, time_embed_dim]

        # Expand conditioning to match spatial points
        conditioning = conditioning.unsqueeze(1).expand(-1, N, -1)  # [B, N, C]

        # Concatenate coords and conditioning
        x = torch.cat([coords, conditioning], dim=-1)  # [B, N, D + C]

        # Flatten and process
        x = x.view(B * N, -1)
        x = self.backbone(x)
        dvf = self.dvf_head(x)

        # Reshape
        dvf = dvf.view(B, N, D)

        # Clip to max displacement
        dvf = torch.tanh(dvf) * self.max_displacement

        return dvf

    def compute_jacobian_loss(
        self,
        coords: torch.Tensor,
        time: torch.Tensor,
        resp_signal: torch.Tensor | None = None,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Compute Jacobian determinant regularization.

        Penalizes deformations with det(J) < 0 (folding) or det(J) too far from 1.

        Args:
            coords: Spatial coordinates [B, N, D]
            time: Normalized time [B]
            resp_signal: Respiratory surrogate [B], optional
            eps: Finite difference step size

        Returns:
            Jacobian loss (scalar)
        """
        B, N, D = coords.shape
        device = coords.device

        # Compute DVF at coords
        dvf = self.forward(coords, time, resp_signal)

        # Compute Jacobian via finite differences
        jacobians = []

        for d in range(D):
            # Perturb in direction d
            coords_plus = coords.clone()
            coords_plus[..., d] += eps

            coords_minus = coords.clone()
            coords_minus[..., d] -= eps

            dvf_plus = self.forward(coords_plus, time, resp_signal)
            dvf_minus = self.forward(coords_minus, time, resp_signal)

            # Gradient in direction d
            grad_d = (dvf_plus - dvf_minus) / (2 * eps)  # [B, N, D]
            jacobians.append(grad_d)

        # Stack to form Jacobian matrix [B, N, D, D]
        jacobian = torch.stack(jacobians, dim=-1)

        # Add identity (since total transform is x + dvf)
        identity = torch.eye(D, device=device).unsqueeze(0).unsqueeze(0)
        jacobian = jacobian + identity

        # Compute determinant
        if D == 2:
            det = (
                jacobian[..., 0, 0] * jacobian[..., 1, 1]
                - jacobian[..., 0, 1] * jacobian[..., 1, 0]
            )
        elif D == 3:
            det = torch.linalg.det(jacobian)
        else:
            det = torch.linalg.det(jacobian)

        # Penalize: |det - 1|^2 (want volume-preserving)
        # Also penalize negative determinants heavily
        loss = ((det - 1) ** 2).mean()
        loss = loss + F.relu(-det).mean() * 10.0  # Heavy penalty for folding

        return loss


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding similar to transformers.

    Args:
        dim: Embedding dimension (must be even)
    """

    def __init__(self, dim: int):
        """__init__.

        Args:
            dim (int): Description.
        """
        super().__init__()
        self.dim = dim

        # Pre-compute frequency bands
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim).float() / half_dim)
        self.register_buffer("freqs", freqs)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Embed time values.

        Args:
            time: Time values [B, 1] in range [0, 1]

        Returns:
            Embeddings [B, dim]

        forward method for SinusoidalTimeEmbedding.

        Executes PyTorch tensor operations.

        Args:
            time (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Scale time to larger range for better frequency coverage
        time = time * 1000.0

        # Compute embeddings
        args = time * self.freqs.unsqueeze(0)  # [B, half_dim]
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embeddings
