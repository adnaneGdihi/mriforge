"""PINNeRF: 4D Cardiac Physics-Informed NeRF
=========================================

Implementation of Experiment 4.2 from the Research Roadmap.
Uses 4D INR (x, y, z, t) for dynamic cardiac MRI reconstruction with physics constraints.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class SpatialEncoder(nn.Module):
    """Encode spatial coordinates with positional encoding."""

    def __init__(self, num_freqs: int = 6):
        """__init__.

        Args:
            num_freqs (int): Description.
        """
        super().__init__()
        self.num_freqs = num_freqs
        # Output dim: 3 + 2 * 3 * num_freqs (original + sin/cos per freq)
        self.out_dim = 3 + 2 * 3 * num_freqs

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, N, 3] spatial coords
        Returns:
            encoded: [B, N, out_dim]

        forward method for SpatialEncoder.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        encoded = [coords]
        for i in range(self.num_freqs):
            freq = 2**i * torch.pi
            encoded.append(torch.sin(freq * coords))
            encoded.append(torch.cos(freq * coords))
        return torch.cat(encoded, dim=-1)


class CardiacODEFunc(nn.Module):
    """Cardiac motion dynamics (velocity field)."""

    def __init__(self, hidden_dim: int = 128):
        """__init__.

        Args:
            hidden_dim (int): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),  # (x, y, z, t)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),  # Output: velocity (dx/dt, dy/dt, dz/dt)
        )

    def forward(self, t: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [1] current time
            coords: [B*N, 3] spatial coordinates
        Returns:
            velocities: [B*N, 3]

        forward method for CardiacODEFunc.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        t_expanded = t.expand(coords.shape[0], 1)
        coords_t = torch.cat([coords, t_expanded], dim=-1)
        return self.net(coords_t)


class DynamicNeRF(nn.Module):
    """4D Implicit Neural Representation with Neural ODE for cardiac motion."""

    def __init__(self, hidden_dim: int = 256, out_channels: int = 1):
        """__init__.

        Args:
            hidden_dim (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.spatial_encoder = SpatialEncoder(num_freqs=6)

        # Static appearance network
        self.appearance_net = nn.Sequential(
            nn.Linear(self.spatial_encoder.out_dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

        # Cardiac motion dynamics (ODE function)
        self.ode_func = CardiacODEFunc(hidden_dim=128)

    def forward(self, coords_t: torch.Tensor, _use_ode: bool = False) -> torch.Tensor:
        """Query 4D INR.

        Args:
            coords_t: [B, N, 4] (x, y, z, t)
            use_ode: Whether to integrate ODE (for physics regularization)

        Returns:
            Intensity [B, N, out_channels]

        forward method for DynamicNeRF.

        Executes PyTorch tensor operations.

        Args:
            coords_t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            _use_ode (bool): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        coords, t = coords_t[..., :3], coords_t[..., 3:4]

        # Encode spatial coordinates
        coords_encoded = self.spatial_encoder(coords)

        # Concatenate with time
        features = torch.cat([coords_encoded, t], dim=-1)

        # Predict intensity
        return self.appearance_net(features)

    def compute_divergence(self, coords_t: torch.Tensor) -> torch.Tensor:
        """Compute divergence of velocity field for incompressibility constraint.

        Args:
            coords_t: [B, N, 4]
        Returns:
            divergence: [B, N, 1]
        """
        coords = coords_t[..., :3]
        t = coords_t[..., 3:4].mean()  # Scalar time

        coords.requires_grad_(True)
        velocities = self.ode_func(t, coords.reshape(-1, 3))
        velocities = velocities.contiguous().view(*coords.shape)

        # Compute divergence: sum of diagonal elements of Jacobian
        # div(v) = dv_x/dx + dv_y/dy + dv_z/dz
        div = torch.zeros(coords.shape[:-1], device=coords.device)

        for i in range(3):
            # Compute d(velocity_i) / d(coord_i)
            grad_outputs = torch.zeros_like(velocities)
            grad_outputs[..., i] = 1
            grads = torch.autograd.grad(
                outputs=velocities,
                inputs=coords,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            div += grads[..., i]

        return div.unsqueeze(-1)


@register_model(name="pin_inr", training_mode="reconstruction")
class PININR(nn.Module, IGenerator):
    """Physics-Informed Implicit Neural Representation for Cardiac MRI.

    Formerly mislabeled as PINNeRF (no ray marching involved).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 256,
        _temporal_dim: int = 64,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            hidden_dim (int): Description.
            _temporal_dim (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.nerf = DynamicNeRF(hidden_dim=hidden_dim, out_channels=out_channels)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "PININR"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate output from spatio-temporal coordinates."""
        return self.forward(z, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Spatio-temporal coordinates [B, N, 4] or [B, C, T, H, W] or [B, C, H, W]

        Returns:
            Predicted intensity

        forward method for PININR.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() == 5:  # [B, C, T, H, W] - 4D cardiac data
            B, C, T, H, W = x.shape
            device = x.device

            # Create coordinate grid using meshgrid (vectorized)
            t_coords = torch.linspace(0, 1, T, device=device)
            y_coords = torch.linspace(-1, 1, H, device=device)
            x_coords_vec = torch.linspace(-1, 1, W, device=device)

            # Create 4D meshgrid [T, H, W] per dimension
            grid_t, grid_y, grid_x = torch.meshgrid(t_coords, y_coords, x_coords_vec, indexing="ij")

            # Stack to [T*H*W, 4] - (x, y, z=0, t)
            coords_t = (
                torch.stack(
                    [
                        grid_x.flatten(),
                        grid_y.flatten(),
                        torch.zeros(T * H * W, device=device),  # z = 0 for 2D+t
                        grid_t.flatten(),
                    ],
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(B, -1, -1)
            )  # [B, T*H*W, 4]

            values = self.nerf(coords_t)  # [B, T*H*W, out_channels]
            return values.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        elif x.dim() == 4:  # [B, C, H, W] - 2D static MRI (ADDED)
            B, C, H, W = x.shape
            device = x.device

            # Create 2D coordinate grid
            y_coords = torch.linspace(-1, 1, H, device=device)
            x_coords_vec = torch.linspace(-1, 1, W, device=device)

            grid_y, grid_x = torch.meshgrid(y_coords, x_coords_vec, indexing="ij")

            # Stack to [H*W, 4] with z=0, t=0 (static)
            coords_t = (
                torch.stack(
                    [
                        grid_x.flatten(),
                        grid_y.flatten(),
                        torch.zeros(H * W, device=device),  # z = 0
                        torch.zeros(H * W, device=device),  # t = 0 (static)
                    ],
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(B, -1, -1)
            )  # [B, H*W, 4]

            values = self.nerf(coords_t)  # [B, H*W, out_channels]
            return values.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        elif x.dim() == 3 and x.shape[-1] == 4:  # Already coordinates [B, N, 4]
            return self.nerf(x)

        else:
            raise ValueError(
                f"PINNeRF expects [B, C, H, W], [B, C, T, H, W], or [B, N, 4] but got {x.shape}"
            )

    def compute_physics_loss(self, coords_t: torch.Tensor) -> torch.Tensor:
        """Compute physics-informed loss (incompressibility constraint).

        Args:
            coords_t: [B, N, 4]

        Returns:
            Loss value (divergence should be near zero)
        """
        divergence = self.nerf.compute_divergence(coords_t)
        return (divergence**2).mean()
