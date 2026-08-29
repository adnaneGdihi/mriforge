"""Cross-Contrast Homotopic Transport for MRI Synthesis.

Innovation XIX from SOTA Roadmap: Geometry-preserving contrast synthesis.

Key Insight:
    Standard image translation (pix2pix, CycleGAN) can distort lesion geometry.
    Homotopic transport uses optimal transport to preserve topology:
    - Same lesion should appear in both contrasts
    - Anatomical boundaries must be preserved

Mathematical Foundation:
    Wasserstein gradient flow:
        ∂ρ/∂t = ∇·(ρ∇ψ)

    where ρ is the image intensity distribution and ψ is the transport potential.

    Geometric regularization:
        ||∇T(x) - I|| → 0  (near-identity Jacobian)

    This prevents folding/distortion of anatomical structures.

Benefits:
    - Lesion-localization preservation
    - Topology-preserving transport
    - Interpretable transformation

References:
    - [59] Optimal transport for medical imaging
    - [60] Diffeomorphic registration

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.registry import register_model


class TransportPotentialNet(nn.Module):
    """Network predicting transport potential ψ.

    The potential defines the transport map via its gradient.

    Args:
        in_channels: Input channels
        hidden_channels: Feature channels
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, 3, padding=1),  # Source + target
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),  # Scalar potential
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Predict transport potential.

        Args:
            source: Source contrast (B, C, H, W)
            target: Target contrast (B, C, H, W)

        Returns:
            Potential field ψ (B, 1, H, W)
        """
        x = torch.cat([source, target], dim=1)
        return self.net(x)


class IntensityTransformer(nn.Module):
    """Transform intensity values between contrasts.

    Learns the nonlinear mapping of intensities (T1→T2, etc.).

    Args:
        hidden_channels: Feature channels
    """

    def __init__(self, hidden_channels: int = 64) -> None:
        """__init__.

        Args:
            hidden_channels (int): Description.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform intensities.

        forward method for IntensityTransformer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


@register_model(name="homotopic_transport", training_mode="reconstruction")
class HomotopicTransport(nn.Module):
    """Homotopic Transport for cross-contrast synthesis.

    Uses optimal transport with geometric regularization to
    synthesize one MRI contrast from another while preserving anatomy.

    Args:
        in_channels: Input channels (2 for complex as real/imag)
        hidden_channels: Feature channels
        num_steps: Transport flow steps
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_steps: int = 10,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
            num_steps (int): Description.
        """
        super().__init__()

        self.num_steps = num_steps

        # Transport potential network
        self.potential_net = TransportPotentialNet(in_channels, hidden_channels)

        # Intensity transformer
        self.intensity_net = IntensityTransformer(hidden_channels)

    def forward(
        self,
        source: torch.Tensor,
        return_flow: bool = False,
    ) -> torch.Tensor:
        """Synthesize target contrast from source.

        Args:
            source: Source contrast (B, C, H, W)
            return_flow: If True, return transport field

        Returns:
            Synthesized target (B, C, H, W)
        """
        B, C, H, W = source.shape
        device = source.device

        # Initial estimate
        target_init = self.intensity_net(source)

        # Initialize flow as identity
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        flow = torch.zeros(B, 2, H, W, device=device)

        # Iterative transport
        current = source.clone()
        for step in range(self.num_steps):
            # Interpolation parameter
            t = step / self.num_steps

            # Get potential
            psi = self.potential_net(current, target_init)

            # Compute gradient of potential (transport velocity)
            grad_psi_x = psi[:, :, :, 1:] - psi[:, :, :, :-1]
            grad_psi_y = psi[:, :, 1:, :] - psi[:, :, :-1, :]

            # Pad to original size
            grad_psi_x = F.pad(grad_psi_x, (0, 1, 0, 0))
            grad_psi_y = F.pad(grad_psi_y, (0, 0, 0, 1))

            # Update flow
            velocity = torch.cat([grad_psi_x, grad_psi_y], dim=1)
            flow = flow + velocity / self.num_steps

            # Apply flow
            sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            sample_grid = sample_grid.expand(B, -1, -1, -1)
            sample_grid = sample_grid + flow.permute(0, 2, 3, 1)

            current = F.grid_sample(source, sample_grid, mode="bilinear", align_corners=True)

        # Final intensity adjustment
        output = self.intensity_net(current)

        if return_flow:
            return output, flow
        return output

    def compute_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        jacobian_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict]:
        """Compute transport loss with geometric regularization.

        Args:
            source: Source contrast
            target: Ground truth target
            jacobian_weight: Weight for Jacobian regularization

        Returns:
            (total_loss, loss_dict)
        """
        output, flow = self(source, return_flow=True)

        losses = {}

        # Reconstruction loss
        loss_recon = F.mse_loss(output, target)
        losses["recon"] = loss_recon.item()

        # Structural loss to prevent hallucinations
        loss_l1 = F.l1_loss(output, target)
        losses["l1"] = loss_l1.item()

        # Jacobian regularization: ||∇T - I|| should be small
        # Encourages near-identity deformation
        dx = flow[:, 0, :, 1:] - flow[:, 0, :, :-1]
        dy = flow[:, 1, 1:, :] - flow[:, 1, :-1, :]

        # Identity Jacobian would have dx=0, dy=0
        loss_jac = (dx**2).mean() + (dy**2).mean()
        losses["jacobian"] = loss_jac.item()

        total = loss_recon + loss_l1 + jacobian_weight * loss_jac
        losses["total"] = total.item()

        return total, losses


def create_homotopic_transport(
    hidden_channels: int = 64,
    num_steps: int = 10,
) -> HomotopicTransport:
    """Factory function for Homotopic Transport.

    Args:
        hidden_channels: Feature channels
        num_steps: Transport steps

    Returns:
        Configured model
    """
    return HomotopicTransport(
        in_channels=2,
        hidden_channels=hidden_channels,
        num_steps=num_steps,
    )


__all__ = [
    "HomotopicTransport",
    "IntensityTransformer",
    "TransportPotentialNet",
    "create_homotopic_transport",
]
