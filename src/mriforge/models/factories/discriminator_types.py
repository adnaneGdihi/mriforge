"""Discriminator model type constants for type-safe model creation.

This module provides the DiscriminatorTypes class which acts as a single source of truth
for all available discriminator model types in the MRIForge framework.

Usage:
    from mriforge.models.factories.discriminator_types import DiscriminatorTypes

    # Type-safe access to discriminator types
    factory.create_discriminator(DiscriminatorTypes.PATCHGAN, ...)
"""

from __future__ import annotations


class DiscriminatorTypes:
    """Registry of all available discriminator model types in MRIForge.

    This class provides constants for all supported discriminator architectures.
    Each constant corresponds to a registered discriminator in the ModelFactory.
    """

    # GAN discriminators
    PATCHGAN = "patchgan"
    REAL_ESRGAN = "real_esrgan"

    @classmethod
    def all_types(cls) -> list[str]:
        """Return list of all available discriminator types.

        Returns:
            List of all registered discriminator type strings
        """
        return [
            cls.PATCHGAN,
            cls.REAL_ESRGAN,
        ]

    @classmethod
    def is_valid(cls, discriminator_type: str) -> bool:
        """Check if a discriminator type is valid and registered.

        Args:
            discriminator_type: The discriminator type string to validate

        Returns:
            True if the discriminator type is registered, False otherwise
        """
        return discriminator_type in cls.all_types()


__all__ = ["DiscriminatorTypes"]
