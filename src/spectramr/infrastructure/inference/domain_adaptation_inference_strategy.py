"""Domain Adaptation Inference Strategy

Inference strategy for domain adaptation models (e.g., CycleGAN, UNIT).
Handles cross-domain translation between different MRI modalities or field strengths.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class DomainAdaptationInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for Domain Adaptation models.

    Handles translation between source and target domains (e.g., 0.55T -> 3T).

    Inference Modes:
    1. A_to_B: Translate source domain A to target domain B
    2. B_to_A: Translate target domain B to source domain A
    3. Cycle: A -> B -> A (Verify cycle consistency)
    """

    # Phase 2: single-pass forward through the adapted generator —
    # compatible with tiling.
    supports_grid_tiling: bool = True

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize domain adaptation inference strategy.

        Args:
            model: The domain adaptation model (typically has generators G_A2B, G_B2A)
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)

        # Extract domain adaptation config
        self.da_config = self.config.get("domain_adaptation", {})

        # Default direction
        self.direction = self.da_config.get("direction", "AtoB")  # AtoB or BtoA

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.DOMAIN_ADAPTATION

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input tensor.

        Args:
            input_tensor: Raw input tensor
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor
        """
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)
        return input_tensor.detach()

    def run_inference(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run domain adaptation inference.

        Args:
            input_tensor: Preprocessed input tensor
            **kwargs: Additional parameters
                - direction: 'AtoB' or 'BtoA' (override default)
                - return_cycle: If True, return cycle reconstruction (A->B->A)

        Returns:
            Translated image or tuple with metadata
        """
        direction = kwargs.get("direction", self.direction)
        return_cycle = kwargs.get("return_cycle", False)

        with torch.no_grad():
            if direction == "AtoB":
                # A -> B
                if hasattr(self.model, "netG_A2B"):
                    fake_B = self.model.netG_A2B(input_tensor)
                elif hasattr(self.model, "generator_ab"):
                    fake_B = self.model.generator_ab(input_tensor)
                else:
                    # Assume model is just the generator for this direction
                    fake_B = self.model(input_tensor)

                if return_cycle:
                    # B -> A (Consistency)
                    if hasattr(self.model, "netG_B2A"):
                        rec_A = self.model.netG_B2A(fake_B)
                    elif hasattr(self.model, "generator_ba"):
                        rec_A = self.model.generator_ba(fake_B)
                    else:
                        raise ValueError(
                            "Model does not have reverse generator for cycle consistency"
                        )
                    return fake_B, {"cycle_reconstruction": rec_A.detach().cpu()}

                return fake_B, {}

            else:  # BtoA
                # B -> A
                if hasattr(self.model, "netG_B2A"):
                    fake_A = self.model.netG_B2A(input_tensor)
                elif hasattr(self.model, "generator_ba"):
                    fake_A = self.model.generator_ba(input_tensor)
                else:
                    # Assume model is just the generator for this direction
                    fake_A = self.model(input_tensor)

                if return_cycle:
                    # A -> B (Consistency)
                    if hasattr(self.model, "netG_A2B"):
                        rec_B = self.model.netG_A2B(fake_A)
                    elif hasattr(self.model, "generator_ab"):
                        rec_B = self.model.generator_ab(fake_A)
                    else:
                        raise ValueError(
                            "Model does not have reverse generator for cycle consistency"
                        )
                    return fake_A, {"cycle_reconstruction": rec_B.detach().cpu()}

                return fake_A, {}

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

        # Unnormalize [-1, 1] -> [0, 1] if typically used in CycleGAN
        if self.da_config.get("unnormalize", True):
            if output.min() < 0:
                output = (output + 1.0) / 2.0

        output = torch.clamp(output, 0.0, 1.0)

        return output, metadata

    @torch.no_grad()
    def infer_single(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run inference on a single input tensor."""
        return self.infer(input_data, **kwargs)

    def infer_batch(self, input_data_list: list, batch_size: int = 4, **kwargs) -> list:
        """Run inference on a batch of input tensors."""
        results = []
        for input_data in input_data_list:
            result = self.infer_single(input_data, **kwargs)
            results.append(result)
        return results
