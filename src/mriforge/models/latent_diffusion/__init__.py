#!/usr/bin/env python3
"""Latent Diffusion Models Package
==============================

This package contains implementations of latent diffusion models including
VAE, VQ-VAE, and latent diffusion models for super-resolution tasks.
"""

from mriforge.models.losses.latent_losses import (
    LatentDiffusionLoss,
    LatentRegularizationLoss,
    PerceptualLatentLoss,
    VQLoss,
    create_annealed_kl_loss,
    create_beta_kl_loss,
    create_latent_diffusion_loss,
    create_vq_loss,
)

from .latent_discriminator import (
    LatentDiscriminator,
    MultiScaleLatentDiscriminator,
    PatchLatentDiscriminator,
    create_latent_discriminator,
    create_multiscale_latent_discriminator,
    create_patch_latent_discriminator,
    create_standard_latent_discriminator,
)
from .latent_unet import LatentUNet, create_latent_unet

__all__ = [
    "LatentDiffusionLoss",
    "LatentDiscriminator",
    "LatentRegularizationLoss",
    "LatentUNet",
    "MultiScaleLatentDiscriminator",
    "PatchLatentDiscriminator",
    "PerceptualLatentLoss",
    "VQLoss",
    "create_annealed_kl_loss",
    "create_beta_kl_loss",
    "create_latent_diffusion_loss",
    "create_latent_discriminator",
    "create_latent_unet",
    "create_multiscale_latent_discriminator",
    "create_patch_latent_discriminator",
    "create_standard_latent_discriminator",
    "create_vq_loss",
]
