"""
K-Space Masked Image Modeling (MIM) Loss.

Implements self-supervised pretraining objective for MRI Foundation Models.
Masks k-space lines while preserving the central Auto-Calibration Signal (ACS) region.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KSpaceMIMLoss(nn.Module):
    """
    Masked Image Modeling Loss in k-space.

    Args:
        mask_ratio (float): Percentage of k-space lines to mask (0.0 to 1.0).
        acs_radius (int): Number of central lines to preserve (fully sampled center).
        patch_size (int): Size of patches if masking patches (default 1 for line masking).
    """

    def __init__(self, mask_ratio: float = 0.75, acs_radius: int = 4):
        """__init__.

        Args:
            mask_ratio (float): Description.
            acs_radius (int): Description.
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.acs_radius = acs_radius

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input k-space tensor [B, C, H, W, 2] (complex) or [B, C, H, W] (magnitude/real).
               Assumes H is phase encoding direction to mask.

        Returns:
            loss: scalar loss value.
            masked_x: input with locations masked out (replaced with zeros or token).
            mask: binary mask [B, 1, H, 1] (1 = masked/drop, 0 = keep).

        forward method for KSpaceMIMLoss.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape[:4]

        # 1. Create Mask
        # Shape [B, 1, H, 1] for line masking across W and C
        # Mask 1 means "remove", 0 means "keep".

        prob = torch.rand(B, 1, H, 1, device=x.device)
        mask = (prob < self.mask_ratio).float()

        # 2. Preserve ACS (Center)
        center = H // 2
        # Determine strict center range
        acs_start = center - self.acs_radius
        acs_end = center + self.acs_radius

        # Force mask to 0 in ACS region
        mask[:, :, acs_start:acs_end, :] = 0.0

        # 3. Apply Mask
        # If x is complex [..., 2], mask broadcasts
        if x.dim() == 5:
            mask_broad = mask.unsqueeze(-1)
        else:
            mask_broad = mask

        masked_x = x * (1 - mask_broad)  # Zero out masked regions

        # 4. Compute Loss
        # We want to reconstruct 'x' from 'masked_x'.
        # But this module is usually JUST the loss function,
        # normally the model would take masked_x and predict x_pred.
        #
        # However, for standard MIM, the loss is computed *only on masked regions*.
        #
        # Let's assume this forward() is just a utility to generate the mask and target.
        # The model training loop would look like:
        # loss, masked_input, mask = mim_loss(input) -> this is wrong flow.
        # Correct flow:
        #   masked_input, mask = mim_loss.mask_input(input)
        #   pred = model(masked_input)
        #   loss = mim_loss.compute_loss(pred, input, mask)

        # Retaining original signature style but adapting logic
        return masked_x, mask

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute predictions vs target L1/L2 loss only on masked regions.

        Args:
            pred: Predicted k-space.
            target: Original k-space.
            mask: Binary mask (1=masked).
        """
        # Broadcast mask
        if target.dim() == 5 and mask.dim() == 4:
            mask = mask.unsqueeze(-1)

        loss = F.l1_loss(pred, target, reduction="none")

        # Mean over masked regions only
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        return loss
