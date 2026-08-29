"""Dynamic 3DGS: Motion-Decoupled 3D Gaussian Splatting.

Innovation VIII (SOTA 2.0): Temporal 3D Reconstruction.

Mathematical Foundation:
    - Canonical Space: t=0, G_0 = {mu_0, q_0, s_0, alpha_0, c_0}
    - Deformation Field: D(mu_0, t) -> (delta_mu, delta_q)
    - Deformed Gaussians:
        mu(t) = mu_0 + delta_mu
        q(t) = q_0 * delta_q (quaternion multiplication or addition)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.registry import register_model
from mriforge.models.volumetric.gen_3dgs import GaussianParameters


class DeformationField(nn.Module):
    """Predicts deformation (delta_pos, delta_rot) from Position and Time."""

    def __init__(self, hidden_dim=64):
        """__init__.

        Args:
            hidden_dim (Any): Description.
        """
        super().__init__()
        # Simple coordinate-based MLP
        # Input: 3 (pos) + 1 (time) = 4
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            # Output: 3 for pos, 4 for rotation
            nn.Linear(hidden_dim, 7),
        )

        # Zero initialization for deformation (start from identity)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Canonical positions [B, N, 3]
            t: Time scalar [B, 1]

        Returns:
            delta: [B, N, 7]

        forward method for DeformationField.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, _ = x.shape
        t_exp = t.view(B, 1, 1).expand(B, N, 1)

        inp = torch.cat([x, t_exp], dim=-1)
        return self.net(inp)


@register_model(name="dynamic_3dgs", training_mode="reconstruction")
class Dynamic3DGS(nn.Module):
    """Dynamic 3D Gaussian Splatting Model."""

    def __init__(self, canonical_model: nn.Module = None):
        """__init__.

        Args:
            canonical_model (nn.Module): Description.
        """
        super().__init__()
        self.deformation_net = DeformationField()
        self.canonical_model = canonical_model  # e.g. Gen3DGS

    def forward(self, x_static: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x_static: Canonical representation input (e.g. MRI volume or params)
            t: Time query

        Returns:
            Deformed GaussianParameters

        forward method for Dynamic3DGS.

        Executes PyTorch tensor operations.

        Args:
            x_static (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Get Canonical Gaussians
        # If canonical_model is Gen3DGS, x_static is the volume
        if self.canonical_model is not None:
            _, canonical_params = self.canonical_model(x_static)
        else:
            # Assume x_static IS the parameters (raw tensor)
            canonical_params = GaussianParameters(x_static)

        # 2. Get Canonical Positions
        mu_0 = canonical_params.xyz  # [B, 3, D, H, W]
        B, C, D, H, W = mu_0.shape

        # Flatten for deformation net
        mu_0_flat = mu_0.permute(0, 2, 3, 4, 1).reshape(B, -1, 3)  # [B, N, 3]

        # 3. Predict Deformation
        delta = self.deformation_net(mu_0_flat, t)
        delta_xyz = delta[..., :3]
        delta_rot = delta[..., 3:]

        # 4. Apply Deformation
        # Position: Additive
        mu_t_flat = mu_0_flat + delta_xyz

        # Rotation: Additive or Composition?
        # 3DGS usually uses quaternion composition, but additive is stable for small deform
        # q_0 = canonical_params.rotation (flattened)
        # Let's keep it simple: We return the *deltas* usually used by the splatting engine,
        # or return the final parameters.

        # Here we return a dictionary of deformed properties (flat)
        # to be reshaped or rasterized.

        return {
            "xyz": mu_t_flat,
            "delta_rot": delta_rot,  # Apply later
            "canonical": canonical_params,
        }


def create_dynamic_3dgs() -> Dynamic3DGS:
    """create_dynamic_3dgs.

    Returns:
        Dynamic3DGS: Description.
    """
    return Dynamic3DGS()
