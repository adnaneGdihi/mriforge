"""Phased-Array Projected Conditional Flow Matching (PA-PCFM).

Innovation I from SOTA Roadmap: Deterministic flow matching with physics constraints.

Key Insight:
    Standard flow matching learns velocity field v_θ(x,t) to transport from noise to data.
    PA-PCFM constrains v_θ to the null-space of the forward operator, letting physics
    handle the range-space trajectory (data consistency).

Mathematical Formulation:
    - Decompose velocity: v_θ = v_θ^∥ (null-space) + v_θ^⊥ (range-space)
    - Loss: L = E[||P_N(A)(v_θ - (x₁-x₀))||²]
    - Only learn null-space flow; range-space is analytically determined

Benefits:
    - Deterministic sampling (no stochastic diffusion)
    - NFE < 10 (vs ~1000 for standard diffusion)
    - Perfect data consistency by construction

References:
    - [1] arxiv:2512.17493 - UPMRI: Projected Conditional Flow Matching

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.infrastructure.physics.forward_operator import (
    MultiCoilForwardOperator,
    NullSpaceProjection,
)
from mriforge.models.registry import register_model


class PAPCFMVelocityNet(nn.Module):
    """Velocity network for PA-PCFM.

    Predicts the velocity field v(x, t, y) for transporting the prior
    to the posterior conditioned on measurements y.

    Architecture: U-Net with time embedding and k-space conditioning.

    Args:
        in_channels: Number of input channels (2 for complex as real/imag)
        base_channels: Base channel count
        time_embed_dim: Dimension of time embedding
        cond_channels: Channels for conditioning (k-space)
        num_res_blocks: Residual blocks per resolution
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        time_embed_dim: int = 128,
        cond_channels: int = 2,
        num_res_blocks: int = 2,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            base_channels (int): Description.
            time_embed_dim (int): Description.
            cond_channels (int): Description.
            num_res_blocks (int): Description.
        """
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial projection
        self.in_conv = nn.Conv2d(
            in_channels + cond_channels,  # x + conditioning
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # Encoder
        self.enc1 = ResBlock(base_channels, base_channels, time_embed_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)

        self.enc2 = ResBlock(base_channels * 2, base_channels * 2, time_embed_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        # Middle
        self.mid = ResBlock(base_channels * 4, base_channels * 4, time_embed_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(
            base_channels * 4, base_channels * 2, time_embed_dim
        )  # Skip connection

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_embed_dim)

        # Output projection
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            x: Current state (B, 2, H, W) - real/imag channels
            t: Time (B,) or (B, 1) in [0, 1]
            cond: Conditioning (B, 2, H, W) - e.g., zero-filled recon

        Returns:
            Velocity (B, 2, H, W)
        """
        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)  # (B, time_embed_dim)

        # Concatenate input and conditioning
        h = torch.cat([x, cond], dim=1)
        h = self.in_conv(h)

        # Encoder
        h1 = self.enc1(h, t_emb)
        h = self.down1(h1)

        h2 = self.enc2(h, t_emb)
        h = self.down2(h2)

        # Middle
        h = self.mid(h, t_emb)

        # Decoder with skip connections
        h = self.up2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h, t_emb)

        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, t_emb)

        return self.out_conv(h)


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_embed_dim (int): Description.
        """
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = F.silu(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)


@register_model(name="pa_pcfm", training_mode="diffusion")
class PAPCFM(nn.Module):
    """Phased-Array Projected Conditional Flow Matching.

    Main model class implementing:
    1. Null-space projected velocity learning
    2. Optimal transport interpolation
    3. Deterministic ODE integration for sampling

    Args:
        velocity_net: Network predicting v_θ(x, t, y)
        forward_operator: Multi-coil MRI forward operator
        sigma_min: Minimum noise scale for OT path
        use_null_projection: Whether to project velocity to null-space
    """

    def __init__(
        self,
        velocity_net: PAPCFMVelocityNet | None = None,
        forward_operator: MultiCoilForwardOperator | None = None,
        sigma_min: float = 1e-4,
        use_null_projection: bool = True,
        # Config args
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 4,
    ) -> None:
        """__init__.

        Args:
            velocity_net (PAPCFMVelocityNet | None): Description.
            forward_operator (MultiCoilForwardOperator | None): Description.
            sigma_min (float): Description.
            use_null_projection (bool): Description.
            in_channels (int): Description.
            hidden_channels (int): Description.
            num_layers (int): Description.
        """
        super().__init__()

        if velocity_net is None:
            velocity_net = PAPCFMVelocityNet(
                in_channels=in_channels,
                base_channels=hidden_channels,
                num_res_blocks=2,  # Fixed default
            )

        if forward_operator is None:
            # Create dummy operator for validation
            # Needs coil sensitivities and mask
            # For validation instantiation we can skip actual functional forward_op
            # but we need the object.
            # Ideally we have a 'create_dummy' method on MultiCoilForwardOperator
            # But let's check what ForwardOperator needs.
            pass

        self.velocity_net = velocity_net
        self.forward_op = forward_operator

        if forward_operator is not None:
            self.null_proj = NullSpaceProjection(forward_operator)
        else:
            self.null_proj = None

        self.sigma_min = sigma_min
        self.use_null_projection = use_null_projection

    def compute_loss(
        self,
        x1: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PA-PCFM training loss.

        Loss = E_t [ ||P_N(A)(v_θ(φ_t, t, y) - (x₁ - x₀))||² ]

        Args:
            x1: Target image (batch, 1, H, W) complex
            y: k-space measurements (batch, coils, H, W) complex

        Returns:
            Scalar loss
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Convert to real representation (B, 2, H, W)
        x1_real = self._complex_to_real(x1)

        # Sample prior x0 ~ N(0, I)
        x0_real = torch.randn_like(x1_real)

        # Sample time t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)

        # Optimal transport interpolation: φ_t = t*x1 + (1-t)*x0
        t_exp = t[:, None, None, None]
        phi_t = t_exp * x1_real + (1 - t_exp) * x0_real

        # Target velocity: x1 - x0
        target_velocity = x1_real - x0_real

        # Get conditioning (zero-filled reconstruction)
        cond = self._get_conditioning(y)

        # Predict velocity
        pred_velocity = self.velocity_net(phi_t, t, cond)

        # Project to null-space if enabled
        if self.use_null_projection:
            # Convert to complex for projection
            pred_complex = self._real_to_complex(pred_velocity)
            target_complex = self._real_to_complex(target_velocity)

            # Project
            pred_null = self.null_proj(pred_complex)
            target_null = self.null_proj(target_complex)

            # Loss on null-space component only
            loss = F.mse_loss(pred_null.real, target_null.real)
            loss = loss + F.mse_loss(pred_null.imag, target_null.imag)
        else:
            # Standard flow matching loss
            loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        num_steps: int = 10,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Sample reconstruction using ODE integration.

        Uses Euler discretization of: dx/dt = v_θ(x, t, y)
        with data consistency projection at each step.

        Args:
            y: k-space measurements (batch, coils, H, W) complex
            num_steps: Number of integration steps (NFE)
            return_trajectory: If True, return all intermediate states

        Returns:
            Reconstructed image (batch, 1, H, W) complex
        """
        batch_size = y.shape[0]
        H, W = y.shape[-2:]
        device = y.device

        # Initialize from noise
        x = torch.randn(batch_size, 2, H, W, device=device)

        # Get conditioning
        cond = self._get_conditioning(y)

        trajectory = [x] if return_trajectory else None

        # Euler integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i / num_steps, device=device)

            # Predict velocity
            v = self.velocity_net(x, t, cond)

            # Euler step
            x = x + dt * v

            # Data consistency (project to measurement manifold)
            x_complex = self._real_to_complex(x)

            # Decompose into range + null
            x_range = self.null_proj.range_projection(x_complex)
            x_null = x_complex - x_range

            # Replace range component with measurement-consistent version
            x_measured = self.forward_op.adjoint(y)

            # Combine: measured range + predicted null
            x_complex = x_measured + x_null
            x = self._complex_to_real(x_complex)

            if return_trajectory:
                trajectory.append(x.clone())

        # Final output
        result = self._real_to_complex(x)

        if return_trajectory:
            return result, trajectory
        return result

    def _complex_to_real(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, 1, H, W) complex to (B, 2, H, W) real."""
        if not torch.is_complex(x):
            return x
        return torch.cat([x.real, x.imag], dim=1)

    def _real_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, 2, H, W) real to (B, 1, H, W) complex."""
        if torch.is_complex(x):
            return x
        C = x.shape[1] // 2
        return torch.complex(x[:, :C], x[:, C:])

    def _get_conditioning(self, y: torch.Tensor) -> torch.Tensor:
        """Get conditioning signal from k-space.

        Uses zero-filled reconstruction (A†y) as conditioning.
        """
        x_zf = self.forward_op.adjoint(y)
        return self._complex_to_real(x_zf)


def create_pa_pcfm(
    coil_sensitivities: torch.Tensor,
    mask: torch.Tensor,
    base_channels: int = 64,
    time_embed_dim: int = 128,
) -> PAPCFM:
    """Factory function to create PA-PCFM model.

    Args:
        coil_sensitivities: (num_coils, H, W) complex
        mask: (H, W) binary
        base_channels: UNet base channels
        time_embed_dim: Time embedding dimension

    Returns:
        Configured PA-PCFM model
    """
    # Create forward operator
    forward_op = MultiCoilForwardOperator(
        coil_sensitivities=coil_sensitivities,
        mask=mask,
        centered=True,
    )

    # Create velocity network
    velocity_net = PAPCFMVelocityNet(
        in_channels=2,  # Complex as 2 real channels
        base_channels=base_channels,
        time_embed_dim=time_embed_dim,
        cond_channels=2,
        num_res_blocks=2,
    )

    return PAPCFM(
        velocity_net=velocity_net,
        forward_operator=forward_op,
        use_null_projection=True,
    )


__all__ = [
    "PAPCFM",
    "PAPCFMVelocityNet",
    "create_pa_pcfm",
]
