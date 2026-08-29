#!/usr/bin/env python3
"""Edge Detection Losses
====================

Modern edge detection loss functions for medical image super-resolution.
Replaces legacy Sobel-based losses with improved implementations.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.losses.registry import register_loss


class SobelOperator(nn.Module):
    r"""Modern Sobel edge detection operator.

    Computes horizontal and vertical gradients using Sobel kernels.
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{Sobel}(x) = \sqrt{(G_x * x)^2 + (G_y * x)^2}"""

    def __init__(self):
        """__init__."""
        super().__init__()
        # Define Sobel kernels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )

        # Combine kernels for both directions
        kernels = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)  # (2, 1, 3, 3)

        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sobel operator to input tensor.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Gradient tensor of shape (N, C, 2, H, W)
            where the second dimension contains horizontal and vertical gradients

        forward method for SobelOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        num_channels = x.shape[1]

        if num_channels == 1:
            # Single channel case - use original kernels
            grad = F.conv2d(x, self.kernels, padding=1)  # (N, 2, H, W)
            grad = grad.unsqueeze(1)  # (N, 1, 2, H, W)
        else:
            # Multi-channel case - apply same kernels to each channel
            # For grouped convolution: weight shape should be
            # (out_channels, in_channels/groups, kH, kW)
            # We want out_channels = 2*C (horizontal and vertical for each channel)
            # and in_channels/groups = 1, so weight should be (2*C, 1, 3, 3)
            kernels_replicated = self.kernels.repeat(num_channels, 1, 1, 1)
            grad = F.conv2d(x, kernels_replicated, groups=num_channels, padding=1)
            grad = grad.view(x.shape[0], x.shape[1], 2, x.shape[2], x.shape[3])

        return grad


@register_loss(name="sobel_edge", aliases=["SobelLoss"], domain="image")
class SobelLoss(nn.Module):
    r"""Sobel-based edge detection loss.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 1, H, W] or [B, C, H, W] image-space intensity
    **3D Support**: No (2D Sobel operator)

    Computes MSE loss between Sobel gradients of predicted and target images.
    Do NOT use with k-space [B, 2, H, W].
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{SobelLoss} = \| \mathcal{O}_{Sobel}(\hat{y}) - \mathcal{O}_{Sobel}(y) \|_2^2"""

    def __init__(self, reduction: str = "mean"):
        """__init__.

        Args:
            reduction (str): Description.
        """
        super().__init__()
        self.sobel = SobelOperator()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        lesion_coordinates: torch.Tensor = None,
        lesion_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute Sobel loss between predicted and target images.

        Args:
            y_pred: Predicted images (N, C, H, W)
            y_target: Target images (N, C, H, W)
            lesion_coordinates: Optional lesion coordinates for weighted loss
            lesion_weight: Weight for lesion regions

        Returns:
            Sobel loss value

        forward method for SobelLoss.

        Executes PyTorch tensor operations.

        Args:
            y_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y_target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            lesion_coordinates (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            lesion_weight (float): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Compute Sobel gradients
        pred_grad = self.sobel(y_pred)
        target_grad = self.sobel(y_target)

        # Compute MSE loss between gradients
        if self.mse_loss.reduction == "none":
            # Compute element-wise MSE, then reduce to per-sample
            loss = F.mse_loss(pred_grad, target_grad, reduction="none")
            # Average over all dimensions except batch
            loss = loss.mean(dim=[1, 2, 3, 4])  # (N,)
        else:
            loss = self.mse_loss(pred_grad, target_grad)

        # Apply lesion weighting if provided
        if lesion_coordinates is not None:
            # This would require implementation of lesion-aware weighting
            # For now, just apply uniform weighting
            loss = loss * lesion_weight

        return loss


@register_loss(name="edge_consistency", aliases=["EdgePreservationLoss"], domain="image")
class EdgePreservationLoss(nn.Module):
    r"""Edge preservation loss combining multiple edge detection methods.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 1, H, W] or [B, C, H, W] image-space intensity
    **3D Support**: No (2D edge operators)

    Combines Sobel, Laplacian, and gradient magnitude for robust edge preservation.
    Do NOT use with k-space [B, 2, H, W].
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{EdgePreservation} = \lambda_1 \mathcal{L}_{Sobel} + \lambda_2 \mathcal{L}_{Laplacian} + \lambda_3 \mathcal{L}_{GradMag}"""

    def __init__(self, weights: dict = None):
        """__init__.

        Args:
            weights (dict): Description.
        """
        super().__init__()
        self.sobel_loss = SobelLoss()

        # Default weights for different edge detection methods
        self.weights = weights or {
            "sobel": 1.0,
            "laplacian": 0.5,
            "gradient_magnitude": 0.3,
        }

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """Compute edge preservation loss.

        Args:
            y_pred: Predicted images
            y_target: Target images

        Returns:
            Combined edge preservation loss

        forward method for EdgePreservationLoss.

        Executes PyTorch tensor operations.

        Args:
            y_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y_target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Accumulate as a graph-connected tensor. Using ``0.0`` + ``.item()``
        # would return a gradient-free Python float, silently zeroing this loss.
        total_loss = y_pred.new_zeros(())

        # Sobel-based loss
        if self.weights["sobel"] > 0:
            total_loss = total_loss + self.weights["sobel"] * self.sobel_loss(y_pred, y_target)

        # Laplacian edge detection
        if self.weights["laplacian"] > 0:
            pred_laplacian = self._laplacian_edge_detection(y_pred)
            target_laplacian = self._laplacian_edge_detection(y_target)
            laplacian_loss = F.mse_loss(pred_laplacian, target_laplacian)
            total_loss = total_loss + self.weights["laplacian"] * laplacian_loss

        # Gradient magnitude loss
        if self.weights["gradient_magnitude"] > 0:
            pred_grad_mag = self._gradient_magnitude(y_pred)
            target_grad_mag = self._gradient_magnitude(y_target)
            grad_loss = F.mse_loss(pred_grad_mag, target_grad_mag)
            total_loss = total_loss + self.weights["gradient_magnitude"] * grad_loss

        return total_loss

    def _laplacian_edge_detection(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Laplacian edge detection."""
        # Laplacian kernel
        laplacian_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=x.device,
        )
        laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        # Apply to each channel
        edges = []
        for c in range(x.shape[1]):
            channel = x[:, c : c + 1, :, :]
            edge = F.conv2d(channel, laplacian_kernel, padding=1)
            edges.append(edge)

        return torch.cat(edges, dim=1)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude."""
        # Compute gradients
        sobel_grad = self.sobel_loss.sobel(x)

        # Compute magnitude
        if x.shape[1] == 1:
            # Single channel
            grad_x = sobel_grad[:, 0, 0, :, :]  # Horizontal gradient
            grad_y = sobel_grad[:, 0, 1, :, :]  # Vertical gradient
            magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            magnitude = magnitude.unsqueeze(1)  # Add channel dimension back
        else:
            # Multi-channel - average across channels
            grad_x = sobel_grad[:, :, 0, :, :].mean(dim=1, keepdim=True)
            grad_y = sobel_grad[:, :, 1, :, :].mean(dim=1, keepdim=True)
            magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        return magnitude


@register_loss(name="explicit_gradient", aliases=["GradientLoss"])
class GradientLoss(nn.Module):
    """Explicit First-Order Edge Penalty using Sobel operators.

    Penalizes the L1 difference between the gradients of the synthesized
    and target images in the x and y directions.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{Gradient} = \\| \nabla_x \\hat{y} - \nabla_x y \\|_1 + \\| \nabla_y \\hat{y} - \nabla_y y \\|_1"""

    def __init__(self):
        """__init__."""
        super().__init__()
        # Define Sobel kernels
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
            1, 1, 3, 3
        )
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(
            1, 1, 3, 3
        )
        self.register_buffer("weight_x", kernel_x)
        self.register_buffer("weight_y", kernel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Gradient loss between predicted and target images.

        forward method for GradientLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if pred.shape[1] > 1:
            # Convert to grayscale broadly if needed, but usually it's 1 channel in MRI
            # Just apply on all channels independently
            pass

        import torch.nn.functional as F

        # Expand kernels to match input channels if necessary
        C = pred.shape[1]
        weight_x = self.weight_x.expand(C, 1, 3, 3)
        weight_y = self.weight_y.expand(C, 1, 3, 3)

        pred_grad_x = F.conv2d(pred, weight_x, padding=1, groups=C)
        pred_grad_y = F.conv2d(pred, weight_y, padding=1, groups=C)
        target_grad_x = F.conv2d(target, weight_x, padding=1, groups=C)
        target_grad_y = F.conv2d(target, weight_y, padding=1, groups=C)

        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
