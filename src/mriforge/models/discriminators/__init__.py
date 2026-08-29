"""Discriminator Implementations Package
====================================

This package contains the essential discriminator implementations.
Only PatchGAN and Balanced discriminators are included for simplicity.
"""

from .balanced_discriminators import (
    RealESRGANDiscriminator,
    create_balanced_discriminator,
)
from .patchgan_discriminator import PatchGANDiscriminator
from .stargan_v2_discriminator import (
    StarGANv2Discriminator,
)

__all__ = [
    "PatchGANDiscriminator",
    "RealESRGANDiscriminator",
    "StarGANv2Discriminator",
    "create_balanced_discriminator",
]
