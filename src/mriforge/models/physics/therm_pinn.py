"""Thermal Safety Physics-Informed Neural Network (Therm-PINN).

Innovation XIV from SOTA Roadmap: SAR-optimal 7T scanning.

Key Insight:
    High-field MRI (7T+) has strict SAR limits due to RF heating.
    Therm-PINN embeds the Pennes Bio-Heat equation into the neural network,
    enabling real-time temperature prediction and safety-constrained optimization.

Mathematical Foundation:
    Pennes Bio-Heat equation:
        ρc ∂T/∂t = k∇²T - ρb·cb·ω(T-Tb) + SAR·ρ

    where:
        T: Temperature
        k: Thermal conductivity
        ω: Blood perfusion rate
        SAR: Specific Absorption Rate

    PINN residual loss:
        L_physics = ||ρc ∂T/∂t - k∇²T + ρb·cb·ω(T-Tb) - SAR·ρ||²

    Safety constraint: T < T_limit (hard constraint via penalty)

Benefits:
    - Real-time temperature monitoring
    - Optimal RF power usage
    - Patient-specific safety margins

References:
    - [53] PINNs for thermal modeling
    - [55] SAR management in 7T MRI

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.registry import register_model


class TemperatureNetwork(nn.Module):
    """Neural network predicting temperature field.

    Takes spatial coordinates and time as input, outputs temperature.

    Args:
        hidden_dims: Hidden layer dimensions
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 64, 64, 64),
    ) -> None:
        """__init__.

        Args:
            hidden_dims (tuple[int, ...]): Description.
        """
        super().__init__()

        layers = []
        in_dim = 4  # (x, y, z, t)

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.Tanh(),
                ]
            )
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))  # Temperature output

        self.net = nn.Sequential(*layers)

        # Initialize output to body temperature
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 37.0)

    def forward(self, coords: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict temperature.

        Args:
            coords: Spatial coordinates (batch, 3) in normalized space
            t: Time (batch, 1) in normalized space

        Returns:
            Temperature (batch, 1) in Celsius

        forward method for TemperatureNetwork.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = torch.cat([coords, t], dim=-1)
        return self.net(x)


@register_model(name="therm_pinn", training_mode="reconstruction")
class ThermPINN(nn.Module):
    """Therm-PINN for SAR-constrained 7T imaging.

    Uses physics-informed loss to enforce bio-heat equation.

    Args:
        hidden_dims: Network hidden dimensions
        rho: Tissue density (kg/m³)
        c: Specific heat capacity (J/kg/K)
        k: Thermal conductivity (W/m/K)
        rho_b: Blood density (kg/m³)
        c_b: Blood specific heat (J/kg/K)
        omega: Blood perfusion rate (1/s)
        T_blood: Blood temperature (°C)
        T_limit: Safety temperature limit (°C)
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 64, 64, 64),
        rho: float = 1050.0,
        c: float = 3600.0,
        k: float = 0.5,
        rho_b: float = 1050.0,
        c_b: float = 3600.0,
        omega: float = 0.0036,
        T_blood: float = 37.0,
        T_limit: float = 40.0,
    ) -> None:
        """__init__.

        Args:
            hidden_dims (tuple[int, ...]): Description.
            rho (float): Description.
            c (float): Description.
            k (float): Description.
            rho_b (float): Description.
            c_b (float): Description.
            omega (float): Description.
            T_blood (float): Description.
            T_limit (float): Description.
        """
        super().__init__()

        self.T_net = TemperatureNetwork(hidden_dims)

        # Physical parameters
        self.register_buffer("rho", torch.tensor(rho))
        self.register_buffer("c", torch.tensor(c))
        self.register_buffer("k", torch.tensor(k))
        self.register_buffer("rho_b", torch.tensor(rho_b))
        self.register_buffer("c_b", torch.tensor(c_b))
        self.register_buffer("omega", torch.tensor(omega))
        self.register_buffer("T_blood", torch.tensor(T_blood))
        self.register_buffer("T_limit", torch.tensor(T_limit))

    def forward(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict temperature field.

        Args:
            coords: (batch, 3) spatial coordinates
            t: (batch, 1) time

        Returns:
            Temperature (batch, 1)

        forward method for ThermPINN.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.T_net(coords, t)

    def compute_pde_residual(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        SAR: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Pennes equation residual.

        Args:
            coords: (batch, 3) with requires_grad=True
            t: (batch, 1) with requires_grad=True
            SAR: (batch, 1) local SAR values (W/kg)

        Returns:
            PDE residual (batch, 1)
        """
        coords = coords.requires_grad_(True)
        t = t.requires_grad_(True)

        T = self.T_net(coords, t)

        # Compute gradients
        # ∂T/∂t
        dT_dt = torch.autograd.grad(T.sum(), t, create_graph=True, retain_graph=True)[0]

        # ∂T/∂x, ∂T/∂y, ∂T/∂z
        dT_dx = torch.autograd.grad(T.sum(), coords, create_graph=True, retain_graph=True)[0]

        # ∂²T/∂x², etc. (Laplacian)
        laplacian = torch.zeros_like(T)
        for i in range(3):
            d2T_dxi2 = torch.autograd.grad(
                dT_dx[:, i].sum(), coords, create_graph=True, retain_graph=True
            )[0][:, i : i + 1]
            laplacian = laplacian + d2T_dxi2

        # Pennes equation residual
        # ρc ∂T/∂t = k∇²T - ρb·cb·ω(T-Tb) + SAR·ρ
        lhs = self.rho * self.c * dT_dt
        diffusion = self.k * laplacian
        perfusion = self.rho_b * self.c_b * self.omega * (T - self.T_blood)
        source = SAR * self.rho

        residual = lhs - diffusion + perfusion - source

        return residual

    def compute_loss(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        SAR: torch.Tensor,
        T_measured: torch.Tensor | None = None,
        physics_weight: float = 1.0,
        safety_weight: float = 10.0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total PINN loss.

        Args:
            coords: Collocation points (batch, 3)
            t: Time points (batch, 1)
            SAR: SAR values (batch, 1)
            T_measured: Optional temperature measurements
            physics_weight: Weight for PDE residual
            safety_weight: Weight for safety constraint

        Returns:
            (total_loss, loss_dict)
        """
        losses = {}

        # PDE residual loss
        residual = self.compute_pde_residual(coords, t, SAR)
        loss_pde = (residual**2).mean()
        losses["pde"] = loss_pde.item()

        # Data loss (if measurements available)
        if T_measured is not None:
            T_pred = self(coords.detach(), t.detach())
            loss_data = F.mse_loss(T_pred, T_measured)
            losses["data"] = loss_data.item()
        else:
            loss_data = torch.tensor(0.0, device=coords.device)

        # Safety constraint: max(0, T - T_limit)²
        T_pred = self(coords.detach(), t.detach())
        violation = F.relu(T_pred - self.T_limit)
        loss_safety = (violation**2).mean()
        losses["safety"] = loss_safety.item()

        total = loss_data + physics_weight * loss_pde + safety_weight * loss_safety
        losses["total"] = total.item()

        return total, losses

    def predict_max_safe_SAR(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        SAR_range: tuple[float, float] = (0.1, 10.0),
        num_samples: int = 20,
    ) -> torch.Tensor:
        """Find maximum SAR that keeps T < T_limit.

        Args:
            coords: Spatial location (1, 3)
            t: Time (1, 1)
            SAR_range: Range to search
            num_samples: Number of SAR values to try

        Returns:
            Maximum safe SAR (scalar)
        """
        SARs = torch.linspace(SAR_range[0], SAR_range[1], num_samples, device=coords.device)

        max_safe = SARs[0]
        for sar in SARs:
            with torch.no_grad():
                T = self(coords, t)
                if T.item() < self.T_limit.item():
                    max_safe = sar
                else:
                    break

        return max_safe


def create_therm_pinn(
    hidden_dims: tuple[int, ...] = (64, 64, 64),
    T_limit: float = 40.0,
) -> ThermPINN:
    """Factory function for Therm-PINN.

    Args:
        hidden_dims: Network dimensions
        T_limit: Safety temperature limit

    Returns:
        Configured Therm-PINN model
    """
    return ThermPINN(
        hidden_dims=hidden_dims,
        T_limit=T_limit,
    )


__all__ = [
    "TemperatureNetwork",
    "ThermPINN",
    "create_therm_pinn",
]
