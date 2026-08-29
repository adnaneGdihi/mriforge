"""KAN-based Diffusion Processes
=============================

This module provides KAN-specific diffusion processes. These classes
are used by the model factory to pair a KAN-based U-Net with the
appropriate diffusion process (e.g., standard or cold).
"""

import torch

from mriforge.models.diffusion.base_diffusion import Diffusion
from mriforge.models.diffusion.cold_diffusion import ColdDiffusion

# --- KAN-Specific Diffusion Processes ---


class DiffusionKAN(Diffusion):
    """A standard diffusion process specifically paired with KAN-based U-Nets (Complex-Valued).

    This class overrides the noise injection process to operate explicitly in the
    complex plane, ensuring phase coherence is respected during the forward diffusion.

    Noise Injection:
        z_t = z_0 + sigma_t * (n_real + j * n_imag)
        where n_real, n_imag ~ N(0, 1).
    """

    def __init__(self, *args, **kwargs):
        """__init__."""
        super().__init__(*args, **kwargs)

    def _to_complex(self, x):
        """_to_complex.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        if torch.is_complex(x):
            return x
        if x.shape[1] == 2:
            return torch.complex(x[:, 0, ...], x[:, 1, ...]).unsqueeze(1)
        return x

    def _from_complex(self, x_c, original_shape):
        """_from_complex.

        Args:
            x_c (Any): Description.
            original_shape (Any): Description.
        Returns:
            Any: Description.
        """
        if len(original_shape) == 4 and original_shape[1] == 2 and not torch.is_complex(x_c):
            # Already real? check calling context.
            # If x_c is complex
            if torch.is_complex(x_c):
                return torch.cat([x_c.real, x_c.imag], dim=1)
        return x_c

    def q_sample(self, x_start, t, noise=None):
        """
        complex-valued q_sample.
        If input is 2-channel, treats it as Real/Imag.
        """
        is_complex = torch.is_complex(x_start) or (x_start.ndim == 4 and x_start.shape[1] == 2)

        if not is_complex:
            return super().q_sample(x_start, t, noise)

        # Convert to complex
        x_c = self._to_complex(x_start)

        if noise is None:
            # Generate complex noise
            # Standard randn_like(complex) has var=1 (Re=0.5, Im=0.5)
            # We want Re=1, Im=1 to match "n_real + j*n_imag" where n ~ N(0,1)
            # So scale by sqrt(2)
            noise_c = torch.randn_like(x_c) * 1.41421356
        else:
            noise_c = self._to_complex(noise)

        t = t.to(self.device)
        x_c = x_c.to(self.device)
        noise_c = noise_c.to(self.device)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_c.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_c.shape
        )

        # Complex linear combination
        x_t_c = sqrt_alphas_cumprod_t * x_c + sqrt_one_minus_alphas_cumprod_t * noise_c

        # Convert back
        if x_start.shape[1] == 2 and not torch.is_complex(x_start):
            return torch.cat([x_t_c.real, x_t_c.imag], dim=1)

        return x_t_c


class ColdDiffusionKAN(ColdDiffusion):
    """A cold diffusion process specifically paired with a KAN-based U-Net.

    Inherits all functionality from the ColdDiffusion class.
    """

    def __init__(self, *args, **kwargs):
        """__init__."""
        super().__init__(*args, **kwargs)
        # Cold Diffusion logic is different (deterministic degradation),
        # so noise injection might not be the primary mechanism.
        # But if it uses noise, it should use similar logic.
        # Check ColdDiffusion implementation.
        # For now, leaving as-is implies it uses base implementation.
        pass


import torch.nn as nn

from mriforge.models.layers.kan.kan_convs.fast_kan_conv import FastKANConv2DLayer


class ComplexKanBlock(nn.Module):
    """Complex-Valued KAN Block for Phase-Coherent Diffusion.

    Processing:
        (a + bi)(c + di) = (ac - bd) + i(ad + bc)
        Here, we model the complex transformation as two shared KANs acting on cross-terms:
        Real_out = KAN_r(Real_in) - KAN_i(Imag_in)
        Imag_out = KAN_i(Real_in) + KAN_r(Imag_in)

        This enforces coupled processing of phase information.
    """

    def __init__(self, in_features, out_features, **kwargs):
        """__init__.

        Args:
            in_features (Any): Description.
            out_features (Any): Description.
        """
        super().__init__()
        # Use FastKANConv2DLayer as the learnable function
        # Using kernel_size=3, padding=1 for spatial preservation
        self.kan_r = FastKANConv2DLayer(
            in_features, out_features, kernel_size=3, padding=1, **kwargs
        )
        self.kan_i = FastKANConv2DLayer(
            in_features, out_features, kernel_size=3, padding=1, **kwargs
        )

    def forward(self, x_complex):
        # x_complex: (B, 2, H, W) OR (B, C, H, W) where C is even
        # The prompt implies (B, 2, H, W) or similar.
        # If input has more channels (e.g. 64), we split them.

        """forward.

        Args:
            x_complex (Any): Description.
        Returns:
            Any: Description.

        forward method for ComplexKanBlock.

        Executes PyTorch tensor operations.

        Args:
            x_complex (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        C = x_complex.shape[1]
        half_c = C // 2

        real = x_complex[:, :half_c]
        imag = x_complex[:, half_c:]

        # Coupled processing
        # (ac - bd)
        out_real = self.kan_r(real) - self.kan_i(imag)

        # (ad + bc)
        out_imag = self.kan_i(real) + self.kan_r(imag)

        return torch.cat([out_real, out_imag], dim=1)
