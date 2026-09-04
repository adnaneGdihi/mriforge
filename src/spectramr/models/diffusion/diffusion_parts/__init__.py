__all__ = ["EnhancedDeepDiffusion"]

"""Expose diffusion part implementations with compatibility shims."""

from spectramr.models.diffusion.architectures.enhanced_deep_unet import (
    EnhancedDeepDiffusionUNet as EnhancedDeepDiffusion,
)
