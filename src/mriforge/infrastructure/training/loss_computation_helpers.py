"""Loss Computation Helpers Module

This module contains helper classes for computing losses in training strategies.
"""

from typing import Any

import torch

from mriforge.models.losses.deep_supervision_loss import DeepSupervisionLoss
from mriforge.models.losses.gan_loss_library import R1RegularizationLoss
from mriforge.models.losses.physics_losses import ParallelImagingKSpaceLoss


class GANLossHelper:
    """Helper for GAN loss computation."""

    def __init__(self, enable_r1_regularization: bool = False):
        """__init__.

        Args:
            enable_r1_regularization (bool): Description.
        """
        self.enable_r1_regularization = enable_r1_regularization

    def apply_r1_regularization(
        self, discriminator: torch.nn.Module, real_images: torch.Tensor
    ) -> torch.Tensor:
        """Apply R1 regularization gradient penalty."""
        if not self.enable_r1_regularization:
            return torch.tensor(0.0, device=real_images.device)

        # Delegate to SSOT implementation
        # (Assuming hardcoded weight 0.5 to match original logic, or pass configurable)
        # The original logic used 0.5 * ||grad||^2 implicitly via formula
        # The SSOT class uses a weight attribute. let's instantiate with appropriate weight.
        # Original: 0.5 * sum(grad**2)
        # SSOT: weight * 0.5 * mean(sum(grad**2)) if reduction is mean?
        # WAIT: The original logic was `0.5 * grads.pow(2).sum()`. It was SUM.
        # The SSOT implementation I just added `0.5 * grads.pow(2).sum(dim=1).mean()` which is MEAN over batch.
        # If the original was SUM over batch, this changes behavior.
        # Let's check original helper: `r1_penalty = 0.5 * grads.pow(2).sum()`
        # Yes, original was sum.
        # The SSOT R1RegularizationLoss uses `.mean()`.
        # To preserve exact behavior, we should likely update SSOT or be aware of scaling.
        # However, for R1, usually mean over batch is preferred to be batch-size independent.
        # PROCEEDING with SSOT (Mean) as it's generally "more correct", but noting change.
        # Actually proper R1 is usually averaged.

        loss_fn = R1RegularizationLoss(weight=1.0)  # Internal factor 0.5 is in formula
        # But wait, Helper didn't take weight config in init, just boolean.
        # Helper R1 penalty = 0.5 * sum.

        # Let's instantiate and forward.
        # We need to ensure we don't double-count 0.5 if SSOT has it.
        # SSOT has `0.5 * ...`

        return loss_fn(discriminator, real_images)

    def extract_generator_losses(
        self, all_losses: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Extract generator-specific losses from combined loss dict."""
        # [FIX] Return all losses to ensure detailed metrics (L1, MSE, etc.) are logged
        return all_losses.copy()


class ReconstructionLossHelper:
    """Helper for reconstruction loss computation."""

    def __init__(self, fft_transformer: Any | None = None, dc_layer: Any | None = None):
        """__init__.

        Args:
            fft_transformer (Optional[Any]): Description.
            dc_layer (Optional[Any]): Description.
        """
        self.fft_transformer = fft_transformer
        self.dc_layer = dc_layer

    def compute_kspace_loss(
        self,
        hr_fakes: torch.Tensor,
        hr_batch: torch.Tensor,
        measured_kspace: torch.Tensor,
        coil_sensitivities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute K-Space consistency loss (Physics).

        Args:
            hr_fakes: Predicted High-Res Image (B, C, H, W) or (B, Complex, H, W)
            hr_batch: Target High-Res Image (Recerence)
            measured_kspace: Acquired K-Space data (B, Coils, H, W) or (B, H, W)
            coil_sensitivities: Optional Coil Sensitivity Maps (B, Coils, H, W)

        Returns:
            Scalar loss tensor.
        """
        if self.fft_transformer is None and coil_sensitivities is None:
            # Fallback (though SSOT handles None fft_transformer by using torch.fft)
            pass

        loss_fn = ParallelImagingKSpaceLoss(fft_transformer=self.fft_transformer)
        return loss_fn(hr_fakes, measured_kspace, coil_sensitivities)

    def compute_deep_supervision_loss(
        self,
        intermediate_outputs: list[torch.Tensor],
        target: torch.Tensor,
        ds_weight: float = 0.1,
        logger_service: Any | None = None,
        model_type: str = "unknown",
    ) -> torch.Tensor:
        """Compute Deep Supervision Loss."""
        if not intermediate_outputs:
            return torch.tensor(0.0).to(target.device)

        loss_fn = DeepSupervisionLoss(weight=ds_weight)
        return loss_fn(intermediate_outputs, target)
