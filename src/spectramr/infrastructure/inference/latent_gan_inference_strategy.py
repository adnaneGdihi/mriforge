"""Latent GAN Inference Strategy

Inference strategy for Latent GAN models.
Handles two-stage inference:
1. Latent Generation: Z -> Latent Code (via Generator)
2. Decoding: Latent Code -> Image (via Pretrained Encoder/Decoder)
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class LatentGANInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for Latent GAN models.

    Operates in a compressed latent space provided by a VAE/VQGAN.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize latent GAN inference strategy.

        Args:
            model: The latent GAN model (typically contains generator and first_stage_model)
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)

        self.latent_config = self.config.get("latent_gan", {})
        self.latent_dim = self.latent_config.get("latent_dim", None)

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.LATENT_GAN

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input (usually noise for GANs, or condition).

        Args:
            input_tensor: Input tensor (noise input Z or conditioning image)
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
        """Run latent GAN inference.

        Args:
            input_tensor: Input noise or condition
            **kwargs: Additional parameters
                - decode: Boolean, whether to decode latent to image (default True)

        Returns:
            Generated image or tuple with metadata
        """
        decode_output = kwargs.get("decode", True)

        with torch.no_grad():
            # Step 1: Generate Latent Code
            # Model forward typically generates the latent code
            latent_code = self.model(input_tensor)

            metadata = {"latent_code": latent_code.detach().cpu()}

            # Step 2: Decode Latent Code -> Image
            if decode_output and hasattr(self.model, "decode"):
                output = self.model.decode(latent_code)
            elif decode_output and hasattr(self.model, "first_stage_model"):
                # Use first_stage_model decoder
                output = self.model.first_stage_model.decode(latent_code)
            else:
                # Return latent code if decoding not possible or requested
                output = latent_code

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

        # Check if output is image-like (4D) or latent (maybe 2D/4D but different scale)
        # Assuming image output is normalized [-1, 1] or [0, 1]
        if output.ndim == 4:
            if output.min() < 0:
                output = (output + 1.0) / 2.0
            output = torch.clamp(output, 0.0, 1.0)

        return output, metadata
