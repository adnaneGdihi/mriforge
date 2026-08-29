"""Complex-valued Neural Network Layers.

Provides complex-valued layers including batch normalization and activations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexBatchNorm2d(nn.Module):
    """Complex-valued Batch Normalization for 2D spatial data.

    Handles Interleaved [R1, I1, R2, I2, ...] layout.
    Normalizes complex features by their magnitude while preserving phase.
    IMPORTANT: Does NOT subtract the mean to preserve the DC component in k-space.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """__init__.

        Args:
            num_features (int): Description.
            eps (float): Description.
            momentum (float): Description.
            affine (bool): Description.
            track_running_stats (bool): Description.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.gamma_rr = nn.Parameter(torch.ones(num_features))
            self.gamma_ri = nn.Parameter(torch.zeros(num_features))
            self.gamma_ii = nn.Parameter(torch.ones(num_features))
            self.beta_real = nn.Parameter(torch.zeros(num_features))
            self.beta_imag = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma_rr", None)
            self.register_parameter("gamma_ri", None)
            self.register_parameter("gamma_ii", None)
            self.register_parameter("beta_real", None)
            self.register_parameter("beta_imag", None)

        if track_running_stats:
            self.register_buffer("running_mean_real", torch.zeros(num_features))
            self.register_buffer("running_mean_imag", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
        else:
            self.register_buffer("running_mean_real", None)
            self.register_buffer("running_mean_imag", None)
            self.register_buffer("running_var", None)

    def forward(
        self,
        input_real: torch.Tensor,
        input_imag: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Parse input format
        """forward.

        Args:
            input_real (torch.Tensor): Description.
            input_imag (torch.Tensor | None): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Description.

        forward method for ComplexBatchNorm2d.

        Executes PyTorch tensor operations.

        Args:
            input_real (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            input_imag (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if torch.is_complex(input_real):
            x_real = input_real.real
            x_imag = input_real.imag
            return_complex = True
            return_split = False
        elif input_imag is None:
            # Assume Interleaved [R1, I1, ...]
            x_real = input_real[:, 0::2, ...]
            x_imag = input_real[:, 1::2, ...]
            return_complex = False
            return_split = False
        else:
            x_real = input_real
            x_imag = input_imag
            return_complex = False
            return_split = True

        # Compute statistics
        if self.training or not self.track_running_stats:
            # Magnitude variance E[|z|^2] (No mean subtraction)
            mean_real = x_real.mean(dim=(0, 2, 3))
            mean_imag = x_imag.mean(dim=(0, 2, 3))
            var = (x_real.pow(2) + x_imag.pow(2)).mean(dim=(0, 2, 3))

            if self.track_running_stats:
                self.running_mean_real.mul_(1 - self.momentum).add_(mean_real, alpha=self.momentum)
                self.running_mean_imag.mul_(1 - self.momentum).add_(mean_imag, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
        else:
            var = self.running_var

        # Normalize
        inv_std = torch.rsqrt(var + self.eps).view(1, -1, 1, 1)
        x_real_norm = x_real * inv_std
        x_imag_norm = x_imag * inv_std

        # Apply Affine
        if self.affine:
            grr = self.gamma_rr.view(1, -1, 1, 1)
            gri = self.gamma_ri.view(1, -1, 1, 1)
            gii = self.gamma_ii.view(1, -1, 1, 1)
            br = self.beta_real.view(1, -1, 1, 1)
            bi = self.beta_imag.view(1, -1, 1, 1)
            out_real = grr * x_real_norm + gri * x_imag_norm + br
            out_imag = gri * x_real_norm + gii * x_imag_norm + bi
        else:
            out_real, out_imag = x_real_norm, x_imag_norm

        if return_complex:
            return torch.complex(out_real, out_imag)
        if return_split:
            return out_real, out_imag

        # Return Interleaved
        return torch.stack([out_real, out_imag], dim=2).flatten(1, 2)


class ComplexLinear(nn.Module):
    """Complex-valued Linear Layer.

    Implements (x_real + i*x_imag) * (w_real + i*w_imag) + (b_real + i*b_imag).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            input_real (torch.Tensor): Description.
            input_imag (torch.Tensor | None): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Description.

        forward method for ComplexLinear.

        Executes PyTorch tensor operations.

        Args:
            input_real (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            input_imag (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if torch.is_complex(input_real):
            x_real = input_real.real
            x_imag = input_real.imag
            return_complex = True
        else:
            x_real = input_real
            x_imag = input_imag
            return_complex = False

        # (x_r + i*x_i)(w_r + i*w_i) = (x_r*w_r - x_i*w_i) + i(x_r*w_i + x_i*w_r)
        res_real = F.linear(x_real, self.weight_real) - F.linear(x_imag, self.weight_imag)
        res_imag = F.linear(x_real, self.weight_imag) + F.linear(x_imag, self.weight_real)

        if self.bias_real is not None:
            res_real += self.bias_real
            res_imag += self.bias_imag

        if return_complex:
            return torch.complex(res_real, res_imag)
        return res_real, res_imag


class ComplexReLU(nn.Module):
    """Phase-preserving modReLU activation.

    modReLU(z) = ReLU(|z| - b) * (z / |z|)
    """

    def __init__(self, learnable_bias: bool = False, num_features: int | None = None) -> None:
        """__init__.

        Args:
            learnable_bias (bool): Description.
            num_features (int | None): Description.
        """
        super().__init__()
        if learnable_bias:
            if num_features is None:
                raise ValueError("num_features required for learnable bias")
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("bias", None)

    def forward(
        self, input_real: torch.Tensor, input_imag: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            input_real (torch.Tensor): Description.
            input_imag (torch.Tensor | None): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Description.

        forward method for ComplexReLU.

        Executes PyTorch tensor operations.

        Args:
            input_real (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            input_imag (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if torch.is_complex(input_real):
            x_real = input_real.real
            x_imag = input_real.imag
            return_complex = True
        else:
            x_real = input_real
            x_imag = input_imag
            return_complex = False

        magnitude = torch.sqrt(x_real.pow(2) + x_imag.pow(2) + 1e-8)

        if self.bias is not None:
            # Broadcast bias
            bias = self.bias
            if x_real.ndim == 4:
                bias = bias.view(1, -1, 1, 1)
            elif x_real.ndim == 2:
                bias = bias.view(1, -1)

            scale = F.relu(magnitude - bias) / magnitude
        else:
            scale = F.relu(magnitude) / magnitude

        out_real = x_real * scale
        out_imag = x_imag * scale

        if return_complex:
            return torch.complex(out_real, out_imag)
        return out_real, out_imag


class WirtingerGradient:
    """Utilities for Wirtinger calculus in complex-valued networks."""

    @staticmethod
    def conjugate_gradient(grad_real: torch.Tensor, grad_imag: torch.Tensor) -> torch.Tensor:
        """Compute the conjugate gradient ∂L/∂z*.

        ∂L/∂z* = 1/2 * (∂L/∂x + i * ∂L/∂y)
        """
        return 0.5 * torch.complex(grad_real, grad_imag)
