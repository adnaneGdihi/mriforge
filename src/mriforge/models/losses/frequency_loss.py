import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.infrastructure.physics.fft_ops import fft2c
from mriforge.models.losses.registry import register_loss


@register_loss(name="frequency", aliases=["FrequencyLoss", "freq"], domain="agnostic")
class FrequencyLoss(nn.Module):
    r"""Frequency-domain reconstruction loss via FFT.

    **DOMAIN**: AGNOSTIC (works on both k-space and image)
    **Input**: [B, 2, H, W] complex or [B, 1, H, W] image
    **3D Support**: No (2D FFT only)

    Computes L1 loss between magnitude spectra of FFTs. Converts to magnitude
    image first, then FFT. Works on both domains via magnitude conversion.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{Frequency} = \| |\mathcal{F}(\hat{y})| - |\mathcal{F}(y)| \|_1"""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss in Fourier domain (Magnitude Spectrum of Magnitude Image).

        forward method for FrequencyLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Convert to Magnitude Image
        pred_mag = self._to_magnitude(pred)
        target_mag = self._to_magnitude(target)

        # 2. FFT using canonical centered transform
        pred_fft = fft2c(pred_mag)
        target_fft = fft2c(target_mag)

        # 3. Compare Magnitude of Spectrum
        loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        return loss

    def _to_magnitude(self, t: torch.Tensor) -> torch.Tensor:
        """Convert input to magnitude image."""
        if torch.is_complex(t):
            return torch.abs(t)

        # Handle multi-channel (real/imag)
        if t.dim() == 4 and t.shape[1] % 2 == 0:
            B, C, H, W = t.shape
            t_reshaped = t.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            t_complex = torch.view_as_complex(t_reshaped).permute(0, 3, 1, 2)
            return torch.sqrt(torch.sum(t_complex.abs() ** 2, dim=1, keepdim=True) + 1e-8)

        return t
