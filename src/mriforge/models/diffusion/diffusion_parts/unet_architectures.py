"""U-Net Architectures Module
==========================

This module serves as a central point for accessing various U-Net architectures
and their building blocks. It imports components from the refactored
`unet_variants` sub-package to maintain backward compatibility with other parts
of the codebase that import from this file.

The actual implementations are now located in:
- `src/models/gans/diffusion_parts/unet_variants/`

This refactoring improves code organization and maintainability.
"""

from mriforge.models.diffusion.architectures.transformer_hybrid_unets import (
    SwinHybridUNet,
    ViTHybridUNet,
)

__all__ = ["SwinHybridUNet", "ViTHybridUNet"]
