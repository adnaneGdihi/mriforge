"""Generator model type constants for type-safe model creation and configuration.

This module provides the GeneratorTypes class which acts as a single source of truth
for all available generator model types in the spectraMR framework. Using these constants
instead of magic strings improves code maintainability, enables IDE autocomplete,
and reduces the likelihood of typos.

Usage:
    from spectramr.models.factories.generator_types import GeneratorTypes

    # Type-safe access to generator types
    factory.create_generator(GeneratorTypes.STANDARD_UNET, ...)

    # Validate user input
    user_input = "standard_unet"
    if user_input not in GeneratorTypes.all_types():
        raise ValueError(f"Unknown generator type: {user_input}")
"""

from __future__ import annotations


class GeneratorTypes:
    """Registry of all available generator model types in spectraMR.

    This class provides constants for all 41+ supported generator architectures,
    organized by category for clarity. Each constant corresponds to a registered
    generator in the ModelFactory.
    """

    # Diffusion-based generators
    CHI_SQUARE_DIFFUSION = "chi_square_diffusion"
    COLD_DIFFUSION = "cold_diffusion"
    HYBRID_TRANSFORMER_DIFFUSION = "hybrid_transformer_diffusion"
    KSPACE_COLD_DIFFUSION = "kspace_cold_diffusion"
    LAPLACE_DIFFUSION = "laplace_diffusion"
    LATENT_DIFFUSION = "latent_diffusion"
    RICIAN_DIFFUSION = "rician_diffusion"
    SCORE_BASED_DIFFUSION = "score_based_diffusion"

    # Super-Resolution generators
    CYCLESR = "cyclesr"
    DIFFUSION_SR = "diffusion_sr"
    EDSR = "edsr"
    ELAN = "elan"
    IMDN = "imdn"
    RCAN = "rcan"
    RESTORMER = "restormer"
    SWIN_IR = "swin_ir"

    # Hybrid/Transformer-based generators
    HYBRID_TRANSFORMER_DIFFUSION = "hybrid_transformer_diffusion"
    VIT_HYBRID_UNET = "vit_hybrid_unet"
    VIT_KAN = "vit_kan"
    SWIN_TRANSFORMER_KAN = "swin_transformer_kan"

    # KAN (Kolmogorov-Arnold Network) generators
    KAN = "kan"
    KAN_GAN = "kan_gan"
    STYLE_KAN_GAN = "style_kan_gan"

    # U-Net variants
    STANDARD_UNET = "standard_unet"
    ENHANCED_UNET = "enhanced_unet"
    ENHANCED_DEEP_UNET = "enhanced_deep_unet"
    UNET = "unet"
    UNET_2_5D = "unet_2_5d"
    UNET3D = "unet3d"

    # Attention-based generators
    HAN = "han"
    NAFNet = "nafnet"

    # VAE-based generators
    VAE = "vae"
    SPARSE_VAE = "sparse_vae"
    VQGAN = "vqgan"

    # TRELLIS generators (3D volumetric)
    TRELLIS_3D = "trellis_3d"
    TRELLIS_GAUSSIAN_VAE = "trellis_gaussian_vae"
    TRELLIS_MESH_VAE = "trellis_mesh_vae"

    # Mamba-based generators
    MAMBA_UNET = "mamba_unet"
    MAMBA_RECONSTRUCTION = "mamba_reconstruction"

    # Additional 3D and volumetric generators
    VNET = "vnet"
    UNETR = "unetr"
    VOLUMETRIC_UNET = "volumetric_unet"
    VOLUMETRIC_DIFFUSION = "volumetric_diffusion"
    VOLUMETRIC_VAE = "volumetric_vae"

    # Multi-contrast and specialized generators
    MULTICONTRAST_FUSION = "multicontrast_fusion"
    SLICE_TO_VOLUME = "slice_to_volume"
    UNROLLED_RECONSTRUCTION = "unrolled_reconstruction"
    VARIATIONAL_NETWORK = "variational_network"

    @classmethod
    def all_types(cls) -> list[str]:
        """Return list of all available generator types for validation.

        Returns:
            List of all registered generator type strings

        Example:
            >>> all_gens = GeneratorTypes.all_types()
            >>> if user_type in all_gens:
            ...     generator = factory.create_generator(user_type, ...)
        """
        return [
            # Diffusion-based
            cls.CHI_SQUARE_DIFFUSION,
            cls.COLD_DIFFUSION,
            cls.HYBRID_TRANSFORMER_DIFFUSION,
            cls.KSPACE_COLD_DIFFUSION,
            cls.LAPLACE_DIFFUSION,
            cls.LATENT_DIFFUSION,
            cls.RICIAN_DIFFUSION,
            cls.SCORE_BASED_DIFFUSION,
            # Super-Resolution
            cls.CYCLESR,
            cls.DIFFUSION_SR,
            cls.EDSR,
            cls.ELAN,
            cls.IMDN,
            cls.RCAN,
            cls.RESTORMER,
            cls.SWIN_IR,
            # Hybrid/Transformer
            cls.VIT_HYBRID_UNET,
            cls.VIT_KAN,
            cls.SWIN_TRANSFORMER_KAN,
            # KAN-based
            cls.KAN,
            cls.KAN_GAN,
            cls.STYLE_KAN_GAN,
            # U-Net variants
            cls.STANDARD_UNET,
            cls.ENHANCED_UNET,
            cls.ENHANCED_DEEP_UNET,
            cls.UNET,
            cls.UNET_2_5D,
            cls.UNET3D,
            # Attention-based
            cls.HAN,
            cls.NAFNet,
            # VAE-based
            cls.VAE,
            cls.SPARSE_VAE,
            cls.VQGAN,
            # TRELLIS
            cls.TRELLIS_3D,
            cls.TRELLIS_GAUSSIAN_VAE,
            cls.TRELLIS_MESH_VAE,
            # Mamba-based
            cls.MAMBA_UNET,
            cls.MAMBA_RECONSTRUCTION,
            # Additional 3D
            cls.VNET,
            cls.UNETR,
            cls.VOLUMETRIC_UNET,
            cls.VOLUMETRIC_DIFFUSION,
            cls.VOLUMETRIC_VAE,
            # Multi-contrast and specialized
            cls.MULTICONTRAST_FUSION,
            cls.SLICE_TO_VOLUME,
            cls.UNROLLED_RECONSTRUCTION,
            cls.VARIATIONAL_NETWORK,
        ]

    @classmethod
    def diffusion_types(cls) -> list[str]:
        """Return list of diffusion-based generator types."""
        return [
            cls.CHI_SQUARE_DIFFUSION,
            cls.COLD_DIFFUSION,
            cls.HYBRID_TRANSFORMER_DIFFUSION,
            cls.KSPACE_COLD_DIFFUSION,
            cls.LAPLACE_DIFFUSION,
            cls.LATENT_DIFFUSION,
            cls.RICIAN_DIFFUSION,
            cls.SCORE_BASED_DIFFUSION,
        ]

    @classmethod
    def sr_types(cls) -> list[str]:
        """Return list of super-resolution generator types."""
        return [
            cls.CYCLESR,
            cls.DIFFUSION_SR,
            cls.EDSR,
            cls.ELAN,
            cls.IMDN,
            cls.RCAN,
            cls.RESTORMER,
            cls.SWIN_IR,
        ]

    @classmethod
    def vae_types(cls) -> list[str]:
        """Return list of VAE-based generator types."""
        return [
            cls.VAE,
            cls.SPARSE_VAE,
            cls.VQGAN,
            cls.TRELLIS_GAUSSIAN_VAE,
            cls.TRELLIS_MESH_VAE,
        ]

    @classmethod
    def unet_types(cls) -> list[str]:
        """Return list of U-Net variant types."""
        return [
            cls.STANDARD_UNET,
            cls.ENHANCED_UNET,
            cls.ENHANCED_DEEP_UNET,
            cls.UNET,
            cls.UNET_2_5D,
            cls.UNET3D,
            cls.VOLUMETRIC_UNET,
        ]

    @classmethod
    def kan_types(cls) -> list[str]:
        """Return list of KAN-based generator types."""
        return [
            cls.KAN,
            cls.KAN_GAN,
            cls.STYLE_KAN_GAN,
            cls.VIT_KAN,
            cls.SWIN_TRANSFORMER_KAN,
        ]

    @classmethod
    def is_valid(cls, generator_type: str) -> bool:
        """Check if a generator type is valid and registered.

        Args:
            generator_type: The generator type string to validate

        Returns:
            True if the generator type is registered, False otherwise

        Example:
            >>> if GeneratorTypes.is_valid(user_input):
            ...     generator = factory.create_generator(user_input, ...)
        """
        return generator_type in cls.all_types()


__all__ = ["GeneratorTypes"]
