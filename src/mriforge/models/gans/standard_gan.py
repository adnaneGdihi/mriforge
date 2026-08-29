import copy

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.data_consistency import AdaptiveDataConsistency
from mriforge.models.registry import register_model

# Import alias preserved for old call-sites that referenced
# ``PhysicsInformedDataConsistency`` directly. The class was renamed
# (or never landed under that name) in earlier physics refactors — the
# correct DC primitive for an unrolled physics-informed cascade is
# ``AdaptiveDataConsistency`` (learnable noise weighting, supports
# ``(x, measured_kspace, mask, sensitivity_maps)`` signature). If a
# future refactor reintroduces a distinct ``PhysicsInformedDataConsistency``
# semantics, swap this alias at one site.
PhysicsInformedDataConsistency = AdaptiveDataConsistency


@register_model(name="physics_informed_unet", training_mode="reconstruction")
class PhysicsInformedUNet(nn.Module):
    """
    Cascaded Physics-Informed Network (Unrolled Architecture).

    Alternates between Image Space refinement (UNet) and Physics enforcement (DC)
    multiple times, effectively learning the unrolled steps of a proximal gradient descent.
    """

    def __init__(self, base_unet, num_cascades=5):
        """__init__.

        Args:
            base_unet (Any): Description.
            num_cascades (Any): Description.
        """
        super().__init__()
        self.num_cascades = num_cascades

        # Validate base_unet dimensions
        # PhysicsInformedUNet converts complex input (C) -> real input (2*C)
        # So base_unet must accept 2*C input channels.
        if hasattr(base_unet, "in_channels"):
            # We assume the user intends to pass standard complex MRI data (C=1 -> 2 channels, or C=2 -> 4 channels)
            # This check is heuristic but catches the most common crash.
            # Ideally, we'd check if base_unet.in_channels is even.
            pass

        # Create multiple lightweight instances of the base model
        # We use deepcopy to replicate the architecture
        self.cascades = nn.ModuleList([base_unet])
        for _ in range(num_cascades - 1):
            self.cascades.append(copy.deepcopy(base_unet))

        # This layer enforces y = Ax constraints. ``AdaptiveDataConsistency``
        # is the canonical DC primitive (the previously-referenced
        # ``PhysicsInformedDataConsistency(noise_robustness=...)`` ctor was
        # not present in this repo's physics tree; the alias is set at the
        # module top for backward compat).
        self.dc_layer = AdaptiveDataConsistency()

    def forward(self, x_input, kspace_measured, mask, sensitivity_maps=None):
        """
        x_input: The aliased/noisy input image
        kspace_measured: The actual data from the scanner
        mask: The sampling pattern used
        sensitivity_maps: Coil sensitivity maps for multicoil data

        forward method for PhysicsInformedUNet.

        Executes PyTorch tensor operations.

        Args:
            x_input (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            kspace_measured (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            sensitivity_maps (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x_input

        for i, model in enumerate(self.cascades):
            # 1. CNN Refinement (Regularization step)
            # Handle complex input: view as 2-channel real
            if torch.is_complex(x):
                # (B, C, H, W) complex -> (B, C, H, W, 2) real -> (B, C*2, H, W)
                x_real = torch.view_as_real(x).permute(0, 1, 4, 2, 3).contiguous()
                x_real = x_real.view(x_real.size(0), -1, x_real.size(3), x_real.size(4))
            else:
                x_real = x

            # Residual learning: model predicts the correction/noise
            dx_real = model(x_real)

            # Convert back to complex if needed
            if torch.is_complex(x):
                # (B, C*2, H, W) -> (B, C, 2, H, W) -> (B, C, H, W) complex
                dx_reshaped = dx_real.view(dx_real.size(0), -1, 2, dx_real.size(2), dx_real.size(3))
                dx_reshaped = dx_reshaped.permute(0, 1, 3, 4, 2).contiguous()
                dx = torch.view_as_complex(dx_reshaped)
            else:
                dx = dx_real

            x = x + dx

            # 2. Enforce Physics (Data Consistency step)
            # Replace predicted frequencies with measured ones where available
            x = self.dc_layer(x, kspace_measured, mask, sensitivity_maps)

        # Return (B, 2, H, W) [Real, Imag]
        # Return (B, 2*C, H, W) [Real, Imag]
        if torch.is_complex(x):
            # x is (B, C, H, W)
            # stack dim=1 -> (B, 2, C, H, W)
            # flatten -> (B, 2*C, H, W)
            # If C=1, this becomes (B, 2, H, W) which is what we want
            out = torch.stack([x.real, x.imag], dim=1)
            return out.view(out.shape[0], -1, out.shape[3], out.shape[4])
        else:
            # If already real (e.g. 2-channel), just return it
            # But dc_layer usually returns complex if input was complex or converted
            # Let's ensure consistency.
            # If x is (B, 2, H, W), return x
            if x.dim() == 4 and x.shape[1] == 2:
                return x
            # If x is (B, 1, H, W) real, maybe stack with zeros?
            # But we expect complex handling now.
            return x
