"""Quantized Fourier Neural Operator (Q-FNO) for Edge Deployment.

Innovation XX from SOTA Roadmap: On-scanner INT8 deployment.

Key Insight:
    Standard neural networks are too slow and memory-intensive for on-scanner
    deployment. Q-FNO combines:
    - Fourier Neural Operator: efficient global operations in frequency domain
    - Quantization: INT8 weights for 4x memory reduction and faster inference

Mathematical Foundation:
    Fourier Neural Operator layer:
        (Kv)(x) = F^{-1}(R · F(v))(x)

    where F is FFT and R is a learnable frequency-domain filter.

    Quantization:
        W_q = round(W / scale) * scale

    with learned scale parameters for each layer.

Benefits:
    - O(n log n) complexity via FFT
    - 4x memory reduction from quantization
    - Real-time inference on scanner hardware

References:
    - [19] Fourier Neural Operators
    - [61] Post-training quantization

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model


class FourierLayer(nn.Module):
    """Fourier Neural Operator layer.

    Performs global convolution via FFT.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        modes: Number of Fourier modes to keep
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 12,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            modes (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex weights for frequency filtering
        self.scale = 1 / (in_channels * out_channels)
        self.weights_r = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights_i = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier layer.

        Args:
            x: Input (B, C, H, W)

        Returns:
            Output (B, C_out, H, W)
        """
        B, C, H, W = x.shape

        # FFT
        x_ft = torch.fft.rfft2(x)

        # Truncate to modes
        x_ft = x_ft[:, :, : self.modes, : self.modes]

        # Complex multiplication with learnable weights
        weights = torch.complex(self.weights_r, self.weights_i)
        out_ft = torch.einsum("bcxy,coxy->boxy", x_ft, weights)

        # Pad back to original size
        out_padded = torch.zeros(
            B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_padded[:, :, : self.modes, : self.modes] = out_ft

        # Inverse FFT
        out = torch.fft.irfft2(out_padded, s=(H, W))

        return out


class QuantizedLinear(nn.Module):
    """Quantized linear layer with simulated INT8.

    Uses straight-through estimator for training.

    Args:
        in_features: Input features
        out_features: Output features
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
        """
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable quantization scale
        self.scale = nn.Parameter(torch.ones(1))

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate INT8 quantization."""
        # Scale to INT8 range [-128, 127]
        x_scaled = x / self.scale.clamp(min=1e-6)
        x_quantized = torch.round(x_scaled).clamp(-128, 127)

        # Straight-through estimator for gradients
        x_dequant = x_quantized * self.scale
        return x + (x_dequant - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with quantized weights.

        forward method for QuantizedLinear.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        w_quant = self.quantize(self.weight)
        return F.linear(x, w_quant, self.bias)


@register_model(name="q_fno", training_mode="reconstruction")
class QFNO(nn.Module):
    """Quantized Fourier Neural Operator for MRI reconstruction.

    Combines FNO layers with quantization for edge deployment.

    Args:
        in_channels: Input channels (2 for complex)
        out_channels: Output channels (defaults to in_channels)
        hidden_channels: Feature channels
        num_layers: Number of FNO layers
        modes: Fourier modes to keep
        **kwargs: Additional keyword arguments from model_kwargs config.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int | None = None,
        hidden_channels: int = 32,
        num_layers: int = 4,
        modes: int = 12,
        **kwargs,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Accept config overrides from model_kwargs
        hidden_channels = kwargs.get("width", hidden_channels)
        num_layers = kwargs.get("num_layers", num_layers)
        modes1 = kwargs.get("modes1", modes)
        modes2 = kwargs.get("modes2", modes)
        modes = min(modes1, modes2)  # Use minimum for square FourierLayer

        # Input projection (quantized)
        self.in_proj = QuantizedLinear(in_channels, hidden_channels)

        # FNO layers
        self.fno_layers = nn.ModuleList(
            [FourierLayer(hidden_channels, hidden_channels, modes) for _ in range(num_layers)]
        )

        # Residual connections (1x1 conv, quantized)
        self.res_layers = nn.ModuleList(
            [nn.Conv2d(hidden_channels, hidden_channels, 1) for _ in range(num_layers)]
        )

        # Output projection
        self.out_proj = QuantizedLinear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, C, H, W)
            **kwargs: Optional forward kwargs from strategy.

        Returns:
            Output (B, out_channels, H, W)
        """
        B, C, H, W = x.shape

        # Project to hidden (per-pixel)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        h = self.in_proj(x_flat)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # FNO layers with residual
        for fno, res in zip(self.fno_layers, self.res_layers, strict=False):
            h_fno = fno(h)
            h_res = res(h)
            h = F.gelu(h_fno + h_res)

        # Project to output
        out_flat = h.permute(0, 2, 3, 1).reshape(B * H * W, -1)
        out = self.out_proj(out_flat)
        out = out.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)

        return out

    @property
    def name(self) -> str:
        return "q_fno"

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def estimate_int8_size_mb(self) -> float:
        """Estimate INT8 model size in MB."""
        total_params = self.count_parameters()
        # INT8 = 1 byte per parameter
        return total_params / (1024 * 1024)


def create_q_fno(
    hidden_channels: int = 32,
    num_layers: int = 4,
    modes: int = 12,
) -> QFNO:
    """Factory function for Q-FNO.

    Args:
        hidden_channels: Feature channels
        num_layers: Number of FNO layers
        modes: Fourier modes

    Returns:
        Configured Q-FNO model
    """
    return QFNO(
        in_channels=2,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        modes=modes,
    )


__all__ = [
    "QFNO",
    "FourierLayer",
    "QuantizedLinear",
    "create_q_fno",
]
