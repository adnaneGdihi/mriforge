"""Physics-Driven Inference Strategy

Inference strategy for physics-informed MRI reconstruction models.
Handles data consistency and k-space operations during inference.
"""

import logging
from typing import Any

import torch
from torch import nn

from mriforge.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class PhysicsDrivenInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for physics-driven reconstruction models.

    Applies data consistency between model predictions and measured k-space.
    Supports iterative refinement for unrolled architectures.

    Inference Modes:
    1. Direct: Single forward pass through reconstruction network
    2. Iterative: Multiple refinement steps with data consistency
    3. Cascade: Sequential processing through cascade blocks
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize physics-driven inference strategy.

        Args:
            model: The physics-driven reconstruction model
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)

        # Extract physics config
        self.physics_config = self.config.get("physics", {})
        self.dc_config = self.physics_config.get("data_consistency", {})

        # Data consistency parameters
        self.dc_enabled = self.dc_config.get("enabled", True)
        self.dc_method = self.dc_config.get("method", "hard")  # hard or soft
        self.dc_weight = self.dc_config.get("weight", 1.0)

        # Iterative refinement
        self.num_iterations = self.physics_config.get("num_iterations", 1)

        # Initialize FFT transformer
        self._fft = None
        self._init_fft()

    def _init_fft(self) -> None:
        """Initialize FFT transformer for k-space operations."""
        try:
            from mriforge.infrastructure.physics.fft_ops import FFTTransformer

            self._fft = FFTTransformer(device=self.device)
            logger.info("Initialized FFTTransformer for physics inference")
        except ImportError:
            logger.warning("FFTTransformer not available, DC disabled")
            self.dc_enabled = False

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.RECONSTRUCTION

    def preprocess_input(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Preprocess input for physics-driven inference.

        Args:
            input_tensor: Raw input (undersampled image or k-space)
            **kwargs: Additional parameters
                - kspace: Measured k-space data
                - mask: Undersampling mask
                - is_kspace: If True, input is k-space data

        Returns:
            Tuple of (preprocessed tensor, preprocessing metadata)
        """
        # Ensure tensor is on correct device
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        input_tensor = input_tensor.detach()

        # Extract k-space and mask
        kspace = kwargs.get("kspace")
        mask = kwargs.get("mask")
        is_kspace = kwargs.get("is_kspace", False)

        metadata = {}

        # If input is k-space, convert to image
        if is_kspace and self._fft is not None:
            if kspace is None:
                kspace = input_tensor.clone()
            input_image = self._fft.ifftnc(input_tensor, dims=(-2, -1))
            metadata["original_kspace"] = kspace
        else:
            input_image = input_tensor
            if kspace is not None:
                metadata["original_kspace"] = kspace

        # Store mask for data consistency
        if mask is not None:
            if mask.device != self.device:
                mask = mask.to(self.device)
            metadata["mask"] = mask

        return input_image, metadata

    def run_inference(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run physics-driven inference.

        Args:
            input_tensor: Preprocessed input (undersampled image)
            **kwargs: Additional inference parameters
                - preprocess_metadata: Metadata from preprocessing
                - num_iterations: Override number of refinement iterations

        Returns:
            Reconstructed output or tuple of (output, metadata)
        """
        preprocess_meta = kwargs.get("preprocess_metadata", {})
        num_iterations = kwargs.get("num_iterations", self.num_iterations)

        kspace = preprocess_meta.get("original_kspace")
        mask = preprocess_meta.get("mask")

        with torch.no_grad():
            current = input_tensor

            # Iterative refinement with data consistency
            for i in range(num_iterations):
                # Model forward pass
                if hasattr(self.model, "forward_step"):
                    # Unrolled model with per-step forward
                    output = self.model.forward_step(current, step=i)
                else:
                    # Standard model
                    output = self.model(current)

                # Handle different output types
                if isinstance(output, dict):
                    output = output.get(
                        "reconstruction", output.get("output", list(output.values())[0])
                    )

                # Apply data consistency
                if self.dc_enabled and kspace is not None and mask is not None:
                    output = self._apply_data_consistency(output, kspace, mask)

                current = output

            metadata = {
                "num_iterations": num_iterations,
                "dc_enabled": self.dc_enabled,
                "dc_method": self.dc_method,
            }

            return current, metadata

    def _apply_data_consistency(
        self,
        image: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply data consistency projection.

        Args:
            image: Current reconstruction estimate
            measured_kspace: Original measured k-space
            mask: Undersampling mask

        Returns:
            Data-consistent reconstruction
        """
        if self._fft is None:
            return image

        # Transform to k-space
        pred_kspace = self._fft.fftnc(image, dims=(-2, -1))

        # Apply data consistency
        if self.dc_method == "hard":
            # Hard DC: Replace measured positions with measured values
            dc_kspace = pred_kspace * (1 - mask) + measured_kspace * mask
        else:
            # Soft DC: Weighted combination
            w = self.dc_weight
            dc_kspace = pred_kspace * (1 - w * mask) + measured_kspace * (w * mask)

        # Transform back to image
        return self._fft.ifftnc(dc_kspace, dims=(-2, -1))

    def postprocess_output(
        self,
        output_tensor: torch.Tensor | tuple[torch.Tensor, dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Postprocess physics-driven output.

        Args:
            output_tensor: Raw model output
            **kwargs: Additional postprocessing parameters

        Returns:
            Postprocessed output
        """
        if isinstance(output_tensor, tuple):
            output, metadata = output_tensor
        else:
            output = output_tensor
            metadata = {}

        output = output.detach().cpu()

        # Handle complex output
        if self.physics_config.get("output_complex", False):
            # Keep complex representation
            pass
        elif output.shape[1] == 2:
            # Convert real/imag to magnitude
            output = torch.sqrt(output[:, 0:1] ** 2 + output[:, 1:2] ** 2 + 1e-8)

        # Normalize to [0, 1]
        if self.physics_config.get("normalize_output", True):
            output = output - output.min()
            output = output / (output.max() + 1e-8)
            output = torch.clamp(output, 0.0, 1.0)

        return output, metadata

    def reconstruct_with_dc(
        self,
        undersampled: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        num_iterations: int | None = None,
    ) -> torch.Tensor:
        """Full reconstruction pipeline with data consistency.

        Args:
            undersampled: Undersampled input image
            kspace: Measured k-space
            mask: Undersampling mask
            num_iterations: Number of refinement iterations

        Returns:
            Final reconstruction
        """
        processed, preprocess_meta = self.preprocess_input(undersampled, kspace=kspace, mask=mask)
        result, _ = self.run_inference(
            processed,
            preprocess_metadata=preprocess_meta,
            num_iterations=num_iterations,
        )
        return self.postprocess_output(result)[0]

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this physics-driven inference strategy."""
        info = super().get_strategy_info()
        info.update(
            {
                "physics_config": self.physics_config,
                "dc_enabled": self.dc_enabled,
                "dc_method": self.dc_method,
                "dc_weight": self.dc_weight,
                "num_iterations": self.num_iterations,
                "has_fft": self._fft is not None,
            }
        )
        return info

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
