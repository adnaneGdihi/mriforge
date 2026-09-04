"""MAE Inference Strategy

Inference strategy for Masked Autoencoder (MAE) models.
Handles reconstruction of masked images and visualization of masking.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class MAEInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for Masked Autoencoders.

    Handles reconstruction from masked inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize MAE inference strategy.

        Args:
            model: The MAE model
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)

        self.mae_config = self.config.get("mae", {})
        self.mask_ratio = self.mae_config.get("mask_ratio", 0.75)

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.MAE

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input tensor.

        Args:
            input_tensor: Full image tensor
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
        """Run MAE inference.

        Args:
            input_tensor: Input image
            **kwargs: Additional parameters
                - mask_ratio: Override mask ratio

        Returns:
            Reconstructed image or tuple with metadata (mask, patches)
        """
        mask_ratio = kwargs.get("mask_ratio", self.mask_ratio)

        with torch.no_grad():
            # MAE forward typically returns loss, prediction, mask
            # But during inference we want prediction.
            # Many implementations (like ViT-MAE) have a forward_encoder / forward_decoder split
            # or a specific method for inference.

            if hasattr(self.model, "forward_inference"):
                # Custom inference method
                output = self.model.forward_inference(input_tensor, mask_ratio=mask_ratio)
                # Assume output is (pred, mask) or similar
                if isinstance(output, tuple):
                    pred = output[0]
                    metadata = {"mask": output[1] if len(output) > 1 else None}
                else:
                    pred = output
                    metadata = {}
            else:
                # Standard forward might return loss too
                # Using standard forward and inspecting output
                try:
                    output = self.model(input_tensor, mask_ratio=mask_ratio)
                except TypeError:
                    # Fallback if mask_ratio argument not accepted
                    output = self.model(input_tensor)

                # Check output format
                if isinstance(output, tuple):
                    # Common: (loss, pred, mask) or (pred, mask)
                    if len(output) == 3:
                        pred = output[1]
                        mask = output[2]
                        metadata = {"mask": mask, "loss": output[0]}
                    elif len(output) == 2:
                        pred = output[0]
                        mask = output[1]
                        metadata = {"mask": mask}
                    else:
                        pred = output[0]
                        metadata = {}
                elif isinstance(output, dict):
                    pred = output.get(
                        "pred", output.get("reconstruction", list(output.values())[0])
                    )
                    metadata = {k: v for k, v in output.items() if k != "pred"}
                else:
                    pred = output
                    metadata = {}

            # If pred is patches, we might need to unpatchify
            # This logic depends heavily on model implementation.
            # Assuming model returns reconstructed image for now.
            if hasattr(self.model, "unpatchify") and pred.ndim == 3:
                # (B, N, C*P*P) -> (B, C, H, W)
                pred = self.model.unpatchify(pred)

            return pred, metadata

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

        # MAE often normalizes pixels.
        # If output is normalized, we might need to denormalize using ImageNet stats
        # if the model included that.
        # Here we just clamp to [0, 1] assuming standard normalization.

        # If min < 0, map to [0, 1]
        if output.min() < 0:
            output = (output + 1.0) / 2.0

        output = torch.clamp(output, 0.0, 1.0)

        return output, metadata
