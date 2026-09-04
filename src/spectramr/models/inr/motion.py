"""Dual-INR framework for Dynamic MRI (Motion/Flow).

This module implements the Time-Varying INR described in Pillar I of the UMR framework.
Key concept: The volume at time t is a warped version of a canonical volume.
I(x, t) = Spatial(x + Flow(x, t))
"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from spectramr.models.registry import register_model

from .hypernetwork import Sine
from .local_inr import LocalINR


class StandardSIREN(nn.Module):
    """Standard SIREN with internal learnable parameters."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        omega_0: float = 30.0,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            hidden_features (int): Description.
            hidden_layers (int): Description.
            out_features (int): Description.
            omega_0 (float): Description.
        """
        super().__init__()
        self.net = nn.Sequential()

        # Input layer
        self.net.add_module("linear_0", nn.Linear(in_features, hidden_features))
        self.net.add_module("sine_0", Sine(omega_0))

        # Hidden layers
        for i in range(hidden_layers):
            self.net.add_module(f"linear_{i + 1}", nn.Linear(hidden_features, hidden_features))
            self.net.add_module(f"sine_{i + 1}", Sine(omega_0))

        # Output layer (Linear, no activation for flow)
        self.net.add_module("linear_out", nn.Linear(hidden_features, out_features))

        self._init_weights()

    def _init_weights(self):
        """_init_weights.

        Returns:
            Any: Description.
        """
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    # SIREN initialization paper scheme would be better,
                    # but Kaiming is acceptable for this prototype or
                    # uniform (-1/fan_in, 1/fan_in) for first layer etc.
                    # Simplest robust init for Sine:
                    fan_in = m.weight.size(1)
                    limit = calculate_siren_limit(fan_in)
                    m.weight.uniform_(-limit, limit)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        Args:
            x (Tensor): Description.
        Returns:
            Tensor: Description.

        forward method for StandardSIREN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


def calculate_siren_limit(fan_in: int, omega: float = 30.0):
    """calculate_siren_limit.

    Args:
        fan_in (int): Description.
        omega (float): Description.
    Returns:
        Any: Description.
    """
    return (6 / fan_in) ** 0.5 / omega


class TemporalFlowNet(nn.Module):
    """Predicts deformation field (dx, dy, dz) from (x, y, z, t)."""

    def __init__(self, hidden_features: int = 64, hidden_layers: int = 3, omega_0: float = 30.0):
        """__init__.

        Args:
            hidden_features (int): Description.
            hidden_layers (int): Description.
            omega_0 (float): Description.
        """
        super().__init__()
        # Input: 4 (x, y, z, t)
        # Output: 3 (dx, dy, dz)
        self.siren = StandardSIREN(
            in_features=4,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=3,
            omega_0=omega_0,
        )

    def forward(
        self, coords: Float[Tensor, "B N 3"], times: Float[Tensor, "B N 1"]
    ) -> Float[Tensor, "B N 3"]:
        # Concatenate spatial and temporal coordinates
        """forward.

        Args:
            coords (Float[Tensor, 'B N 3']): Description.
            times (Float[Tensor, 'B N 1']): Description.
        Returns:
            Float[Tensor, 'B N 3']: Description.

        forward method for TemporalFlowNet.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            times (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B N 3']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        xt = torch.cat([coords, times], dim=-1)  # [B, N, 4]
        flow = self.siren(xt)
        return flow


@register_model(name="dual_motion_inr", training_mode="reconstruction")
class DualMotionINR(nn.Module):
    """Combines Spatial Canonical INR with Temporal Flow INR."""

    def __init__(
        self,
        spatial_inr: LocalINR,
        temporal_features: int = 64,
    ):
        """__init__.

        Args:
            spatial_inr (LocalINR): Description.
            temporal_features (int): Description.
        """
        super().__init__()
        self.spatial_inr = spatial_inr
        self.temporal_inr = TemporalFlowNet(hidden_features=temporal_features)

    def forward(
        self,
        canonical_volume: Float[Tensor, "B C D H W"],
        coords: Float[Tensor, "B N 3"],
        times: Float[Tensor, "B N 1"],
    ) -> tuple[Float[Tensor, "B N 1"], Float[Tensor, "B N 3"]]:
        """
        Args:
            canonical_volume: The reference anatomy (z_geo source).
            coords: Query coordinates in [-1, 1].
            times: Time value for each query point.

        Returns:
            intensity: Predicted MRI signal.
            flow: Deformation vector used.

        forward method for DualMotionINR.

        Executes PyTorch tensor operations.

        Args:
            canonical_volume (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            times (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[Float[Tensor, 'B N 1'], Float[Tensor, 'B N 3']]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Predict Flow
        flow = self.temporal_inr(coords, times)

        # 2. Warp Coordinates
        # warping: x_canonical = x_t + flow(x_t, t)
        # OR x_t = x_canonical + flow
        # In Neural Volumes paper, usually: Canonical = Deformed + Flow
        # Here we follow: I(x, t) = Spatial(x + flow)
        warped_coords = coords + flow

        # Clip to ensure valid range for LocalINR?
        # SIREN can handle unbounded, but usually we want to stay in [-1, 1]
        # or rely on the SIREN's periodicity.
        # Ideally, we clamp or tanh the flow. Currently unconstrained.

        # 3. Query Spatial INR
        # Note: We pass canonical_volume as both 'local_volume' (geometry)
        # and 'contrast_volume' (physics) unless we want to disentangle contrast too.
        # Assuming canonical volume carries the target contrast physics for now.
        intensity = self.spatial_inr(local_volume=canonical_volume, coords=warped_coords)

        return intensity, flow
