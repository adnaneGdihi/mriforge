"""Flow MRI Quality Metrics.

Implements metrics for phase-contrast and 4D flow MRI:
- Consistency of Mass (mass conservation)
- Divergence (velocity field quality)

Follows unit-test.md rules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spectramr.config.schemas.enums import Regime
from spectramr.core.metrics.registry import register_metric

try:
    from beartype import beartype
    from jaxtyping import Float, jaxtyped

    JAXTYPING_AVAILABLE = True
except ImportError:
    JAXTYPING_AVAILABLE = False

    def jaxtyped(fn):
        """jaxtyped.

        Args:
            fn (Any): Description.
        Returns:
            Any: Description.
        """
        return fn

    def beartype(fn):
        """beartype.

        Args:
            fn (Any): Description.
        Returns:
            Any: Description.
        """
        return fn


@register_metric(
    "mass_conservation", aliases=["MassConservation"], workflows=frozenset({Regime.FLOW})
)
class ConsistencyOfMass(nn.Module):
    """Mass Conservation metric for flow data.

    For incompressible flow, mass must be conserved:

    .. math::

        \\int \rho \\mathbf{v} \\cdot d\\mathbf{A} = 0 \\quad (\text{through closed surface})

    Or for multiple inlets/outlets:

    .. math::

        \\sum (\text{inlet flows}) = \\sum (\text{outlet flows})

    Deviation from this indicates measurement error or artifacts.
    """

    def __init__(
        self,
        tolerance: float = 0.05,  # 5% tolerance
    ) -> None:
        """__init__.

        Args:
            tolerance (float): Description.
        """
        super().__init__()
        self.tolerance = tolerance

    @jaxtyped
    @beartype
    def forward(
        self,
        inlet_flows: Float[Tensor, "batch num_inlets"],
        outlet_flows: Float[Tensor, "batch num_outlets"],
    ) -> dict:
        """Compute mass conservation error.

        Args:
            inlet_flows: Flow rates through inlet planes [ml/s]
            outlet_flows: Flow rates through outlet planes [ml/s]

        Returns:
            Dictionary with:
                - error: Absolute mass conservation error
                - percent_error: Percentage error relative to inlet
                - passes: Boolean if within tolerance

        forward method for ConsistencyOfMass.

        Executes PyTorch tensor operations.

        Args:
            inlet_flows (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            outlet_flows (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Sum flows
        total_inlet = inlet_flows.sum(dim=1)
        total_outlet = outlet_flows.sum(dim=1)

        # Mass conservation error
        error = (total_inlet - total_outlet).abs()
        percent_error = error / (total_inlet.abs() + 1e-8) * 100.0

        # Check tolerance
        passes = percent_error < (self.tolerance * 100)

        return {
            "error": error.mean(),
            "percent_error": percent_error.mean(),
            "passes": passes.float().mean(),
        }

    @jaxtyped
    @beartype
    def compute_from_velocity(
        self,
        velocity: Float[Tensor, "batch 3 depth height width"],
        plane_normal: Float[Tensor, 3],
        plane_mask: Float[Tensor, "batch 1 depth height width"],
        voxel_area: float = 1.0,
    ) -> Float[Tensor, ""]:
        """Compute flow through a plane from velocity field.

        Args:
            velocity: 3D velocity field [B, 3, D, H, W]
            plane_normal: Normal vector to plane
            plane_mask: Binary mask of plane voxels
            voxel_area: Area per voxel [mm²]

        Returns:
            Total flow through plane [ml/s]
        """
        # Dot product with normal
        flow_component = (velocity * plane_normal.view(1, 3, 1, 1, 1)).sum(dim=1, keepdim=True)

        # Integrate over plane
        total_flow = (flow_component * plane_mask).sum() * voxel_area

        return total_flow


@register_metric("divergence", aliases=["Divergence"], workflows=frozenset({Regime.FLOW}))
class Divergence(nn.Module):
    """Velocity Field Divergence metric.

    For incompressible flow:

    .. math::

        \nabla \\cdot \\mathbf{v} = \frac{\\partial v_x}{\\partial x} + \frac{\\partial v_y}{\\partial y} + \frac{\\partial v_z}{\\partial z} = 0

    Non-zero divergence indicates:
    1. Measurement noise
    2. Partial volume effects
    3. Flow artifacts
    """

    def __init__(
        self,
        voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """
        Args:
            voxel_size: Voxel dimensions (dx, dy, dz) in mm
        """
        super().__init__()
        self.voxel_size = voxel_size

    def _compute_gradient(
        self,
        field: Tensor,
        dim: int,
        spacing: float,
    ) -> Tensor:
        """Compute gradient using central differences."""
        # Pad for central difference
        if dim == 2:  # D dimension
            padded = torch.nn.functional.pad(field, (0, 0, 0, 0, 1, 1), mode="replicate")
            grad = (padded[:, :, 2:, :, :] - padded[:, :, :-2, :, :]) / (2 * spacing)
        elif dim == 3:  # H dimension
            padded = torch.nn.functional.pad(field, (0, 0, 1, 1, 0, 0), mode="replicate")
            grad = (padded[:, :, :, 2:, :] - padded[:, :, :, :-2, :]) / (2 * spacing)
        else:  # dim == 4, W dimension
            padded = torch.nn.functional.pad(field, (1, 1, 0, 0, 0, 0), mode="replicate")
            grad = (padded[:, :, :, :, 2:] - padded[:, :, :, :, :-2]) / (2 * spacing)

        return grad

    @jaxtyped
    @beartype
    def forward(
        self,
        velocity: Float[Tensor, "batch 3 depth height width"],
        mask: Float[Tensor, "batch 1 depth height width"] | None = None,
    ) -> dict:
        """Compute divergence of velocity field.

        Args:
            velocity: 3D velocity field [B, 3, D, H, W]
                velocity[:, 0] = vx
                velocity[:, 1] = vy
                velocity[:, 2] = vz
            mask: Optional mask for ROI (e.g., vessel lumen)

        Returns:
            Dictionary with:
                - divergence_mean: Mean absolute divergence
                - divergence_max: Maximum absolute divergence
                - divergence_map: Full divergence field

        forward method for Divergence.

        Executes PyTorch tensor operations.

        Args:
            velocity (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        dx, dy, dz = self.voxel_size

        # Compute partial derivatives
        dvx_dx = self._compute_gradient(velocity[:, 0:1], dim=4, spacing=dx)
        dvy_dy = self._compute_gradient(velocity[:, 1:2], dim=3, spacing=dy)
        dvz_dz = self._compute_gradient(velocity[:, 2:3], dim=2, spacing=dz)

        # Divergence
        divergence = dvx_dx + dvy_dy + dvz_dz

        # Apply mask if provided
        if mask is not None:
            divergence_masked = divergence * mask
            valid_voxels = mask.sum()
            mean_div = divergence_masked.abs().sum() / (valid_voxels + 1e-8)
            max_div = (divergence_masked.abs() * mask).max()
        else:
            mean_div = divergence.abs().mean()
            max_div = divergence.abs().max()

        return {
            "divergence_mean": mean_div,
            "divergence_max": max_div,
            "divergence_map": divergence,
        }


@register_metric("vnr", aliases=["VelocityNoiseRatio"], workflows=frozenset({Regime.FLOW}))
class VelocityNoiseRatio(nn.Module):
    """Velocity-to-Noise Ratio (VNR) for phase-contrast MRI.

    Analogous to SNR but for velocity data.
    """

    @jaxtyped
    @beartype
    def forward(
        self,
        velocity: Float[Tensor, "batch 3 depth height width"],
        signal_mask: Float[Tensor, "batch 1 depth height width"],
        noise_mask: Float[Tensor, "batch 1 depth height width"] | None = None,
    ) -> Float[Tensor, ""]:
        """Compute velocity-to-noise ratio.

        Args:
            velocity: Velocity field
            signal_mask: Mask of region with flow
            noise_mask: Mask of static region (background)

        Returns:
            VNR (higher is better)

        forward method for VelocityNoiseRatio.

        Executes PyTorch tensor operations.

        Args:
            velocity (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            signal_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            noise_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Signal: mean velocity magnitude in flow region
        vel_magnitude = velocity.norm(dim=1, keepdim=True)
        signal = (vel_magnitude * signal_mask).sum() / (signal_mask.sum() + 1e-8)

        # Noise: velocity std in static region
        if noise_mask is not None:
            noise_region = vel_magnitude * noise_mask
            noise = (
                noise_region[noise_mask.bool()].std()
                if noise_mask.sum() > 0
                else vel_magnitude.std() * 0.1
            )
        else:
            # Use background (low velocity) as noise estimate
            threshold = vel_magnitude.median()
            background = vel_magnitude < threshold
            noise = vel_magnitude[background].std()

        vnr = signal / (noise + 1e-8)

        return vnr


@register_metric("velocity_rmse", aliases=["VelocityRMSE"], workflows=frozenset({Regime.FLOW}))
class VelocityRMSE(nn.Module):
    """Root-mean-square error between predicted and reference velocity fields."""

    def forward(self, v_pred: Tensor, v_target: Tensor) -> Tensor:
        """Compute velocity RMSE over all components/voxels."""
        return torch.sqrt(torch.mean((v_pred - v_target) ** 2))


@register_metric(
    "peak_velocity_error", aliases=["PeakVelocityError"], workflows=frozenset({Regime.FLOW})
)
class PeakVelocityError(nn.Module):
    """Absolute error in peak velocity magnitude (a key clinical PC-MRI readout)."""

    def forward(self, v_pred: Tensor, v_target: Tensor) -> Tensor:
        """Compute |max|v_pred| - max|v_target||."""
        peak_pred = v_pred.norm(dim=1).amax()
        peak_target = v_target.norm(dim=1).amax()
        return (peak_pred - peak_target).abs()


@register_metric("net_flow_error", aliases=["NetFlowError"], workflows=frozenset({Regime.FLOW}))
class NetFlowError(nn.Module):
    """Absolute error in net through-plane flow (integral of through-plane velocity)."""

    def forward(
        self,
        v_through_plane_pred: Tensor,
        v_through_plane_target: Tensor,
        plane_mask: Tensor | None = None,
        voxel_area: float = 1.0,
    ) -> Tensor:
        """Compute |net_flow_pred - net_flow_target| over the plane."""
        if plane_mask is not None:
            v_through_plane_pred = v_through_plane_pred * plane_mask
            v_through_plane_target = v_through_plane_target * plane_mask
        flow_pred = v_through_plane_pred.sum() * voxel_area
        flow_target = v_through_plane_target.sum() * voxel_area
        return (flow_pred - flow_target).abs()
