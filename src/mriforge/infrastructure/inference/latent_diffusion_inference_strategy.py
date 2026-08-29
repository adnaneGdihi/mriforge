"""Latent Diffusion Inference Strategy

Inference strategy for Latent Diffusion Models (LDM).
Handles:
1. Latent Sampling: Random Noise -> Latent Code (via Diffusion Process)
2. Decoding: Latent Code -> Image (via VQVAE/VAE Decoder)
"""

import logging
from typing import Any

import torch
from torch import nn

from mriforge.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class LatentDiffusionInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for Latent Diffusion Models.

    Performs iterative denoising in latent space and decodes result.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize latent diffusion inference strategy.

        Args:
            model: The LDM model (contains unet, first_stage_model, cond_stage_model)
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)

        self.ldm_config = self.config.get("latent_diffusion", {})
        self.num_inference_steps = self.ldm_config.get("num_inference_steps", 50)
        self.guidance_scale = self.ldm_config.get("guidance_scale", 1.0)

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.LATENT_DIFFUSION

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input (conditioning or noise).

        Args:
            input_tensor: Conditioning tensor or initial noise
            **kwargs: Additional parameters

        Returns:
            Preprocessed tensor
        """
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)
        return input_tensor.detach()

    def run_inference(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run latent diffusion inference.

        Args:
            input_tensor: Conditioning input (e.g., class label or image embedding)
            **kwargs: Additional parameters
                - num_inference_steps: Override steps
                - guidance_scale: Override guidance
                - decode: Boolean, whether to decode (default True)

        Returns:
            Generated image or tuple with metadata
        """
        steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        scale = kwargs.get("guidance_scale", self.guidance_scale)
        decode_output = kwargs.get("decode", True)

        with torch.no_grad():
            # Step 1: Sample in Latent Space
            # Assuming model has a 'sample' method for inference (standard in LDMs)
            # If not, we might need to manually run the loop here.
            # For now, we assume the model exposes a high-level generate/sample API.

            if hasattr(self.model, "sample"):
                # Common API
                latents = self.model.sample(
                    cond=input_tensor,
                    steps=steps,
                    guidance_scale=scale,
                )
            elif hasattr(self.model, "generate"):
                latents = self.model.generate(input_tensor, num_steps=steps, scale=scale)
            else:
                # Fallback: assume model forward does sampling if in inference mode?
                # Or raise descriptive error.
                logger.warning("Model does not have 'sample' or 'generate' method. using forward()")
                latents = self.model(input_tensor)

            metadata = {"latents": latents.detach().cpu()}

            # Step 2: Decode Latents
            if decode_output and hasattr(self.model, "decode_first_stage"):
                output = self.model.decode_first_stage(latents)
            elif decode_output and hasattr(self.model, "first_stage_model"):
                output = self.model.first_stage_model.decode(latents)
            else:
                output = latents

            return output, metadata

    def postprocess_output(
        self,
        output_tensor: torch.Tensor | tuple[torch.Tensor, dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Postprocess output.

        Args:
            output_tensor: Raw model output
            **kwargs: Additional parameters

        Returns:
            Postprocessed output
        """
        if isinstance(output_tensor, tuple):
            output, metadata = output_tensor
        else:
            output = output_tensor
            metadata = {}

        output = output.detach().cpu()

        # Normalize images
        if output.ndim == 4:
            if output.min() < 0:
                output = (output + 1.0) / 2.0
            output = torch.clamp(output, 0.0, 1.0)

        return output, metadata
