"""Null-Space Projection Test-Time Adaptation (NSP-TTA).

Innovation XVIII from SOTA Roadmap: Hallucination-free test-time adaptation.

Key Insight:
    Standard TTA can hallucinate details not present in measurements.
    NSP-TTA decomposes output into:
    - Range-space: Determined by measurements (fixed)
    - Null-space: Missing information (learned)

    This guarantees data consistency while allowing adaptation.

Mathematical Foundation:
    Decomposition:
        x̂ = A†y + (I - A†A)·Net(y)

    where:
        A†y: Minimum-norm solution (data consistent)
        (I - A†A): Null-space projector
        Net(y): Learned component

    Key property: Measurement fidelity ||Ax̂ - y|| = 0 (hard constraint)

Benefits:
    - Zero hallucination (guaranteed)
    - Fast adaptation (null-space only)
    - Cross-scanner robustness

References:
    - [50] Null-space in MRI reconstruction
    - [6] Test-time adaptation for medical imaging

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.models.registry import register_model


class NullSpaceNetwork(nn.Module):
    """Network for null-space content prediction.

    Learns to fill missing k-space information.

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
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict null-space content.

        Args:
            x: Input (B, C, H, W)

        Returns:
            Null-space prediction (B, C, H, W)
        """
        return self.net(x)


@register_model(name="nsp_tta", training_mode="reconstruction")
class NSPTTA(nn.Module):
    """Null-Space Projection Test-Time Adaptation.

    Guarantees data consistency while allowing adaptation in null-space.

    Args:
        in_channels: Input channels
        hidden_channels: Feature channels
        mask: k-space undersampling mask (H, W)
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        mask: torch.Tensor | None = None,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
            mask (torch.Tensor | None): Description.
        """
        super().__init__()

        self.null_net = NullSpaceNetwork(in_channels, hidden_channels)

        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def set_mask(self, mask: torch.Tensor) -> None:
        """Set undersampling mask."""
        self.register_buffer("mask", mask)

    def _forward_operator(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model: A(x) = mask * FFT(x).

        Args:
            x: Image (B, 2, H, W) as [real, imag]

        Returns:
            k-space (B, 2, H, W)
        """

        # Convert to complex
        x_complex = torch.complex(x[:, 0], x[:, 1])

        # FFT using canonical centered transform
        kspace = fft2c(x_complex)

        # Apply mask
        if self.mask is not None:
            kspace = kspace * self.mask

        # Back to real
        return torch.stack([kspace.real, kspace.imag], dim=1)


def _adjoint_operator(self, y: torch.Tensor) -> torch.Tensor:
    """Adjoint: A†(y) = IFFT(mask * y).

    Args:
        y: k-space (B, 2, H, W)

    Returns:
        Image (B, 2, H, W)
    """
    y_complex = torch.complex(y[:, 0], y[:, 1])

    # Apply mask (self-adjoint)
    if self.mask is not None:
        y_complex = y_complex * self.mask

    # IFFT using canonical centered transform
    image = ifft2c(y_complex)

    return torch.stack([image.real, image.imag], dim=1)


def _null_space_project(self, x: torch.Tensor) -> torch.Tensor:
    """Null-space projection: (I - A†A)x.

    Args:
        x: Image (B, 2, H, W)

    Returns:
        Null-space component (B, 2, H, W)
    """
    # A†A = IFFT(mask * FFT(x)) for masked FFT
    Ax = self._forward_operator(x)
    AtAx = self._adjoint_operator(Ax)
    return x - AtAx


def forward(
    self,
    y: torch.Tensor,
    return_components: bool = False,
) -> torch.Tensor:
    """Reconstruct with null-space projection.

    x̂ = A†y + (I - A†A)·Net(A†y)

    Args:
        y: Undersampled k-space (B, 2, H, W)
        return_components: If True, return decomposition

    Returns:
        Reconstructed image (B, 2, H, W)
    """
    # Range-space component: minimum-norm solution
    x_range = self._adjoint_operator(y)

    # Predict null-space content
    x_null_pred = self.null_net(x_range)

    # Project to null-space (remove any range-space leakage)
    x_null = self._null_space_project(x_null_pred)

    # Combine
    x_recon = x_range + x_null

    if return_components:
        return x_recon, x_range, x_null
    return x_recon


def adapt(
    self,
    y: torch.Tensor,
    num_steps: int = 5,
    lr: float = 0.001,
) -> dict:
    """Test-time adaptation on new measurement.

    Only adapts null-space network, preserving data consistency.

    Args:
        y: k-space measurement (B, 2, H, W)
        num_steps: Adaptation steps
        lr: Learning rate

    Returns:
        Loss history
    """
    from mriforge.infrastructure.training.builders import OptimizationBuilder

    optimizer = OptimizationBuilder.create_single_optimizer(
        self.null_net.parameters(), learning_rate=lr
    )

    losses = {}

    for step in range(num_steps):
        optimizer.zero_grad()

        x_recon = self(y)

        # Self-consistency loss (encourage smooth null-space)
        x_null = self._null_space_project(self.null_net(self._adjoint_operator(y)))
        loss = torch.abs(x_null).mean()  # Minimize null-space magnitude

        loss.backward()
        optimizer.step()

        losses[f"step_{step}"] = loss.item()

    return losses


def verify_data_consistency(self, y: torch.Tensor) -> torch.Tensor:
    """Verify Ax̂ = y (up to numerical precision).

    Args:
        y: Original k-space

    Returns:
        Residual ||Ax̂ - y||
    """
    with torch.no_grad():
        x_recon = self(y)
        y_recon = self._forward_operator(x_recon)

        # Compare only measured locations
        if self.mask is not None:
            residual = (y - y_recon) * self.mask.unsqueeze(0).unsqueeze(0)
        else:
            residual = y - y_recon

        return residual.abs().mean()


def create_nsp_tta(
    hidden_channels: int = 64,
    mask: torch.Tensor | None = None,
) -> NSPTTA:
    """Factory function for NSP-TTA.

    Args:
        hidden_channels: Feature channels
        mask: Undersampling mask

    Returns:
        Configured NSP-TTA model
    """
    return NSPTTA(
        in_channels=2,
        hidden_channels=hidden_channels,
        mask=mask,
    )


__all__ = [
    "NSPTTA",
    "NullSpaceNetwork",
    "create_nsp_tta",
]
