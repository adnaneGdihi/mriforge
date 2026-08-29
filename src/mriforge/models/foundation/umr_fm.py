"""
Universal MRI Foundation Model (UMR-FM).

Integrates Swin-Mamba-KAN backbone with Masked Image Modeling (MIM)
for large-scale self-supervised pretraining on multi-modal MRI data.
"""

import torch
import torch.nn as nn

from mriforge.models.registry import register_model

from ..topological.swin_mamba_kan import SwinMambaKAN
from .mim_loss import KSpaceMIMLoss


@register_model(name="umr_fm", training_mode="reconstruction")
class UniversalFoundationModel(nn.Module):
    """
    UMR-FM: Universal MRI Foundation Model.

    Architecture:
        - Patch Embed
        - Swin-Mamba-KAN Backbone (Encoder-Decoder or Unet)
        - Reconstruction Head

    Pretraining Objective:
        - K-Space Masked Image Modeling (MIM)
    """

    def __init__(
        self,
        in_chans: int = 2,
        embed_dim: int = 64,
        depths: list[int] = [2, 2, 6, 2],
        num_heads: list[int] = [2, 4, 8, 16],
        window_size: int = 8,
        mask_ratio: float = 0.75,
        acs_radius: int = 4,
    ):
        """__init__.

        Args:
            in_chans (int): Description.
            embed_dim (int): Description.
            depths (list[int]): Description.
            num_heads (list[int]): Description.
            window_size (int): Description.
            mask_ratio (float): Description.
            acs_radius (int): Description.
        """
        super().__init__()

        # 1. Backbone: Unified Swin-Mamba-KAN
        self.backbone = SwinMambaKAN(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
        )

        # 2. MIM Logic
        self.mim_loss = KSpaceMIMLoss(mask_ratio=mask_ratio, acs_radius=acs_radius)

        # 3. Head (Simple linear projection if backbone outputs features,
        #    but SwinMambaKAN currently outputs image-size tensor [B, C, H, W])
        #    So we might not need an extra head if backbone is UNet-like.
        #    Let's assume backbone returns reconstructed image directly.

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: Input k-space [B, C, H, W]
            mask: Optional pre-defined mask. If None, generated randomly.

        Returns:
            loss: MIM loss (if training)
            x_rec: Reconstructed k-space
            mask: The mask utilized

        forward method for UniversalFoundationModel.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Masking
        if mask is None:
            # Generate mask and masked input internally
            # Note: We need to clone x to avoid modifying ground truth for loss
            x_masked, mask = self.mim_loss(x.clone())
        else:
            x_masked = x * (1 - mask)

        # 2. Forward Pass
        x_rec = self.backbone(x_masked)

        # 3. Loss
        if self.training:
            loss = self.mim_loss.compute_loss(x_rec, x, mask)
            return loss, x_rec, mask
        else:
            return x_rec, mask


def create_umr_fm(pretrained: bool = False, **kwargs):
    """create_umr_fm.

    Args:
        pretrained (bool): If True, load pretrained weights. Not yet available --
            raises NotImplementedError rather than silently returning a
            random-init model (pitfall #15: unread knob = silent no-op).
    Returns:
        UniversalFoundationModel: A (random-init) foundation model.
    """
    if pretrained:
        raise NotImplementedError("create_umr_fm: pretrained weights are not available")
    model = UniversalFoundationModel(**kwargs)
    return model
