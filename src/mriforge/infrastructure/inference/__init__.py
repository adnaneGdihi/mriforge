"""Inference Strategies Module

This module contains inference strategies for different training paradigms,
including GAN, diffusion, VAE, VQVAE, disentangled, physics-driven, and
reconstruction inference strategies.
"""

from typing import Any

import torch

from .base_inference_strategy import BaseInferenceStrategy
from .cold_diffusion_inference_strategy import ColdDiffusionInferenceStrategy
from .diffusion_inference_strategy import DiffusionInferenceStrategy
from .disentangled_inference_strategy import DisentangledInferenceStrategy
from .domain_adaptation_inference_strategy import DomainAdaptationInferenceStrategy
from .gan_inference_strategy import GANInferenceStrategy
from .latent_diffusion_inference_strategy import LatentDiffusionInferenceStrategy
from .latent_gan_inference_strategy import LatentGANInferenceStrategy
from .mae_inference_strategy import MAEInferenceStrategy
from .physics_driven_inference_strategy import PhysicsDrivenInferenceStrategy
from .reconstruction_inference_strategy import ReconstructionInferenceStrategy
from .ssl_inference_strategy import SSLInferenceStrategy
from .vae_inference_strategy import VAEInferenceStrategy
from .vqvae_inference_strategy import VQVAEInferenceStrategy

__all__ = [
    "BaseInferenceStrategy",
    "ColdDiffusionInferenceStrategy",
    "DiffusionInferenceStrategy",
    "DisentangledInferenceStrategy",
    "DomainAdaptationInferenceStrategy",
    "GANInferenceStrategy",
    "LatentDiffusionInferenceStrategy",
    "LatentGANInferenceStrategy",
    "MAEInferenceStrategy",
    "PhysicsDrivenInferenceStrategy",
    "ReconstructionInferenceStrategy",
    "SSLInferenceStrategy",
    "VAEInferenceStrategy",
    "VQVAEInferenceStrategy",
    "create_inference_strategy",
]


def create_inference_strategy(
    training_mode: str,
    model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> BaseInferenceStrategy:
    """Factory function to create the appropriate inference strategy.

    Args:
        training_mode: Training paradigm (e.g., 'gan', 'diffusion', 'vae',
                       'vqvae', 'disentangled', 'physics_driven', 'ssl',
                       'domain_adaptation', 'latent_gan', 'latent_diffusion', 'mae')
        model: The trained model
        device: Device to run inference on
        config: Configuration dictionary for the strategy

    Returns:
        Configured inference strategy instance

    Raises:
        ValueError: If training_mode is not supported
    """
    from mriforge.config.schemas.enums import TrainingModeTypes

    try:
        mode_enum = TrainingModeTypes(training_mode.lower())
    except ValueError:
        supported_modes = [mode.value for mode in TrainingModeTypes]
        raise ValueError(
            f"Unsupported training mode: {training_mode}. Supported modes: {supported_modes}"
        ) from None

    strategy_map = {
        TrainingModeTypes.GAN: GANInferenceStrategy,
        TrainingModeTypes.DIFFUSION: DiffusionInferenceStrategy,
        TrainingModeTypes.VAE: VAEInferenceStrategy,
        TrainingModeTypes.RECONSTRUCTION: ReconstructionInferenceStrategy,
        TrainingModeTypes.SSL: SSLInferenceStrategy,
        TrainingModeTypes.DISENTANGLED: DisentangledInferenceStrategy,
        TrainingModeTypes.DOMAIN_ADAPTATION: DomainAdaptationInferenceStrategy,
        TrainingModeTypes.LATENT_GAN: LatentGANInferenceStrategy,
        TrainingModeTypes.LATENT_DIFFUSION: LatentDiffusionInferenceStrategy,
        TrainingModeTypes.MAE: MAEInferenceStrategy,
    }

    # Detect cold diffusion from config if mode is diffusion
    if mode_enum == TrainingModeTypes.DIFFUSION and config:
        # Check for cold type or sampler
        diff_cfg = config.get("diffusion") or config.get("training", {}).get("diffusion", {})
        is_cold = diff_cfg and (
            diff_cfg.get("type") == "cold" or "cold" in diff_cfg.get("sampler", "")
        )
        if is_cold:
            return ColdDiffusionInferenceStrategy(model, device, config)

    strategy_class = strategy_map.get(mode_enum)
    if strategy_class is None:
        raise ValueError(f"No inference strategy available for training mode: {training_mode}")

    return strategy_class(model, device, config)
