"""KIDOT: Knowledge-Informed Dynamic Optimal Transport.

Neural ODE-based transport with physics-informed cost function for
unpaired MRI domain translation (e.g., VLF → HF).

Based on:
- Benamou-Brenier dynamic optimal transport formulation
- Neural ODE (Chen et al., 2018)
- Physics-informed reconstruction constraints
"""

import math
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torchdiffeq import odeint, odeint_adjoint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None
    odeint_adjoint = None


class VelocityField(nn.Module):
    """Neural network parameterizing the velocity field v(x, t).

    The velocity field defines how probability mass moves in the
    transport from source distribution to target distribution.
    """

    def __init__(
        self,
        spatial_dim: int = 2,  # H x W flattened or image channels
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        """
        Args:
            spatial_dim: Dimension of spatial features
            hidden_dim: Hidden layer width
            num_layers: Number of hidden layers
            time_embed_dim: Dimension of time embedding
        """
        super().__init__()

        self.time_embed_dim = time_embed_dim

        # Sinusoidal time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main velocity network
        layers = [nn.Linear(spatial_dim + hidden_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, spatial_dim))

        self.net = nn.Sequential(*layers)

        # Initialize final layer to small outputs (near-identity transport)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _time_embedding(self, t: Tensor, device: torch.device) -> Tensor:
        """Sinusoidal time embedding."""
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        if t.dim() == 0:
            t = t.unsqueeze(0)

        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            t: [1] or [] time in [0, 1]
            x: [B, D] state at time t

        Returns:
            v: [B, D] velocity at (x, t)

        forward method for VelocityField.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Time embedding
        t_emb = self._time_embedding(t, x.device)  # [1, time_embed_dim]
        t_emb = self.time_mlp(t_emb)  # [1, hidden_dim]

        # Broadcast to batch
        if t_emb.size(0) == 1 and x.size(0) > 1:
            t_emb = t_emb.expand(x.size(0), -1)

        # Concatenate and compute velocity
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)


class ConvVelocityField(nn.Module):
    """Convolutional velocity field for image-based transport.

    Works directly on image tensors rather than flattened features.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        time_embed_dim: int = 128,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
            time_embed_dim (int): Description.
        """
        super().__init__()

        self.time_embed_dim = time_embed_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.enc3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)

        # Decoder
        self.dec1 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.dec2 = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

        self.act = nn.SiLU()

        # Initialize output to zero
        nn.init.zeros_(self.dec2.weight)
        nn.init.zeros_(self.dec2.bias)

    def _time_embedding(self, t: Tensor, device: torch.device) -> Tensor:
        """Sinusoidal time embedding."""
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        if t.dim() == 0:
            t = t.unsqueeze(0)

        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        Args:
            t: [] or [1] time in [0, 1]
            x: [B, C, H, W] image state

        Returns:
            v: [B, C, H, W] velocity field

        forward method for ConvVelocityField.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Time embedding
        t_emb = self._time_embedding(t, x.device)  # [1, time_embed_dim]
        t_emb = self.time_mlp(t_emb)  # [1, hidden_channels]
        t_emb = t_emb.view(-1, t_emb.size(-1), 1, 1).expand(B, -1, H, W)

        # Encoder
        h = self.act(self.enc1(x))
        h = h + t_emb  # Add time embedding
        h = self.act(self.enc2(h))
        h = self.act(self.enc3(h))

        # Decoder
        h = self.act(self.dec1(h))
        v = self.dec2(h)

        return v


class PhysicsInformedCost(nn.Module):
    """Physics-informed transport cost function.

    Cost(x, y) = ||x - y||^2 + λ * ||y - A(x)||^2

    Where A is the forward model (e.g., k-space encoding).
    The second term ensures the transported image is consistent
    with the measured data.
    """

    def __init__(
        self,
        forward_model: Callable[[Tensor], Tensor] | None = None,
        lambda_physics: float = 0.1,
    ):
        """
        Args:
            forward_model: A(x) -> y forward operator
            lambda_physics: Weight for physics consistency term
        """
        super().__init__()
        self.forward_model = forward_model
        self.lambda_physics = lambda_physics

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        measurements: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            source: [B, ...] source distribution samples
            target: [B, ...] transported samples
            measurements: [B, ...] measured data (e.g., undersampled k-space)

        Returns:
            cost: [B] transport cost per sample

        forward method for PhysicsInformedCost.

        Executes PyTorch tensor operations.

        Args:
            source (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measurements (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Wasserstein term: transport distance
        wasserstein = torch.sum((source - target) ** 2, dim=tuple(range(1, source.dim())))

        # Physics term: data consistency
        if self.forward_model is not None and measurements is not None:
            predicted = self.forward_model(source)
            physics = torch.sum(
                (measurements - predicted) ** 2, dim=tuple(range(1, measurements.dim()))
            )
            cost = wasserstein + self.lambda_physics * physics
        else:
            cost = wasserstein

        return cost


class ODEFunc(nn.Module):
    """Wrapper for velocity field to work with torchdiffeq."""

    def __init__(self, velocity_field: nn.Module):
        """__init__.

        Args:
            velocity_field (nn.Module): Description.
        """
        super().__init__()
        self.velocity_field = velocity_field
        self.nfe = 0  # Number of function evaluations

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """forward.

        Args:
            t (Tensor): Description.
            x (Tensor): Description.
        Returns:
            Tensor: Description.

        forward method for ODEFunc.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        self.nfe += 1
        return self.velocity_field(t, x)


class KIDOTTransport(nn.Module):
    """Knowledge-Informed Dynamic Optimal Transport.

    Uses Neural ODE to learn the transport map between source and target
    distributions with physics-informed constraints.

    The transport is defined as:
        dx/dt = v(x, t)
        x(0) = source
        x(1) = target

    Training minimizes:
        L = ||x(1) - target||^2 + λ * ∫₀¹ ||v(x,t)||^2 dt + μ * Phys(x(1))
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        use_conv: bool = True,
        num_steps: int = 10,
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        adjoint: bool = True,
        forward_model: Callable | None = None,
        lambda_physics: float = 0.1,
        lambda_kinetic: float = 0.01,
    ):
        """
        Args:
            in_channels: Number of image channels
            hidden_channels: Hidden channels in velocity network
            use_conv: Use convolutional velocity field
            num_steps: Number of integration steps for Euler (ignored for adaptive)
            method: ODE solver method ('dopri5', 'euler', 'rk4')
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
            adjoint: Use adjoint sensitivity for memory efficiency
            forward_model: Physics forward model A(x)
            lambda_physics: Physics consistency weight
            lambda_kinetic: Kinetic energy regularization weight
        """
        super().__init__()

        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError(
                "torchdiffeq required for KIDOT. Install with: pip install torchdiffeq"
            )

        self.num_steps = num_steps
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint
        self.lambda_kinetic = lambda_kinetic

        # Velocity field
        if use_conv:
            velocity = ConvVelocityField(in_channels, hidden_channels)
        else:
            # For flattened inputs
            velocity = VelocityField(spatial_dim=in_channels, hidden_dim=hidden_channels)

        self.ode_func = ODEFunc(velocity)

        # Physics cost
        self.physics_cost = PhysicsInformedCost(forward_model, lambda_physics)

    def transport(
        self,
        source: Tensor,
        t_span: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Transport source to target via ODE integration.

        Args:
            source: [B, C, H, W] source samples
            t_span: Time points for integration
            return_trajectory: Return full trajectory

        Returns:
            target: [B, C, H, W] transported samples at t=1
            trajectory: [T, B, C, H, W] full trajectory if requested
        """
        if t_span is None:
            t_span = torch.linspace(0, 1, self.num_steps, device=source.device)

        # Reset NFE counter
        self.ode_func.nfe = 0

        # Choose solver
        solver = odeint_adjoint if self.adjoint else odeint

        # Integrate ODE
        trajectory = solver(
            self.ode_func,
            source,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        # trajectory: [T, B, C, H, W]
        target = trajectory[-1]  # Final state

        if return_trajectory:
            return target, trajectory
        return target, None

    def forward(
        self,
        source: Tensor,
        target_gt: Tensor | None = None,
        measurements: Tensor | None = None,
    ) -> dict:
        """Forward pass with loss computation.

        Args:
            source: [B, C, H, W] source distribution
            target_gt: [B, C, H, W] ground truth target (for supervised training)
            measurements: [B, ...] measured data for physics consistency

        Returns:
            Dictionary with transported image and losses

        forward method for KIDOTTransport.

        Executes PyTorch tensor operations.

        Args:
            source (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_gt (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measurements (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Transport
        transported, trajectory = self.transport(source, return_trajectory=True)

        result = {"transported": transported, "nfe": self.ode_func.nfe}

        # Reconstruction loss (supervised case)
        if target_gt is not None:
            result["loss_recon"] = torch.nn.functional.mse_loss(transported, target_gt)

        # Physics consistency loss
        if measurements is not None:
            physics_cost = self.physics_cost(source, transported, measurements)
            result["loss_physics"] = physics_cost.mean()

        # Kinetic energy regularization (encourages short paths)
        if trajectory is not None and self.lambda_kinetic > 0:
            # Compute velocity at intermediate points
            velocities = trajectory[1:] - trajectory[:-1]  # Finite difference
            kinetic = torch.sum(velocities**2) / velocities.numel()
            result["loss_kinetic"] = self.lambda_kinetic * kinetic

        # Total loss
        total_loss = 0.0
        for key, value in result.items():
            if key.startswith("loss_"):
                total_loss = total_loss + value
        result["loss"] = total_loss

        return result

    def inverse_transport(self, target: Tensor) -> Tensor:
        """Inverse transport from target to source.

        Useful for cycle consistency or bidirectional training.
        """
        t_span = torch.linspace(1, 0, self.num_steps, device=target.device)

        solver = odeint_adjoint if self.adjoint else odeint
        trajectory = solver(
            self.ode_func,
            target,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        return trajectory[-1]


def create_kidot_transport(
    in_channels: int = 1,
    hidden_channels: int = 64,
    use_conv: bool = True,
    forward_model: Callable | None = None,
    lambda_physics: float = 0.1,
    **kwargs,
) -> KIDOTTransport:
    """Factory function for KIDOT transport module."""
    return KIDOTTransport(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        use_conv=use_conv,
        forward_model=forward_model,
        lambda_physics=lambda_physics,
        **kwargs,
    )
