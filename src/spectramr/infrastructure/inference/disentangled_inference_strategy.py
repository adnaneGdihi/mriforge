"""Disentangled MRI Inference Strategy

Inference strategy for Disentangled MRI models (Experiment 32a, 50).
Handles content/style disentanglement and optional Bloch synthesis.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class DisentangledInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for Disentangled MRI models.

    DisentangledMRI separates anatomy (content) from contrast (style):
    - ContentEncoder: Extracts structural information (anatomy)
    - StyleEncoder: Extracts physics/contrast information (style)
    - Generator: Combines content + style via AdaIN

    Inference Modes:
    1. Reconstruction: Encode input → Decode with same style
    2. Style Transfer: Encode input → Decode with target style
    3. Tissue Params: Output tissue parameters (ρ, T1, T2) + Bloch synthesis
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize Disentangled inference strategy.

        Args:
            model: The DisentangledMRI model
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)

        # Extract disentangled-specific config
        self.disentangled_config = self.config.get("disentangled", {})

        # Inference mode: 'reconstruction', 'style_transfer', 'tissue_params'
        self.inference_mode = self.disentangled_config.get("inference_mode", "reconstruction")

        # Whether model outputs tissue parameters
        self.output_tissue_params = self.disentangled_config.get("output_tissue_params", False)

        # Fallback: Check model config if not found in disentangled section
        if not self.output_tissue_params:
            model_conf = self.config.get("model", {}).get("model_kwargs", {})
            if (
                model_conf.get("output_mode") == "tissue_params"
                or model_conf.get("enable_bloch_synthesis") is True
            ):
                self.output_tissue_params = True
                logger.info("Enabled tissue param output based on model configuration")

        # Target style vector for style transfer (optional)
        self.target_style = None

        # Bloch layer for tissue param → image synthesis
        self._bloch_layer = None
        if self.output_tissue_params:
            self._init_bloch_layer()

    def _init_bloch_layer(self) -> None:
        """Initialize Bloch layer for tissue parameter synthesis."""
        try:
            from spectramr.infrastructure.physics.multi_physics_bloch import (
                MultiPhysicsBlochLayer,
            )

            self._bloch_layer = MultiPhysicsBlochLayer(
                default_flip_angle_deg=10.0,
                learnable_flip_angle=False,  # Fixed during inference
                eps=1e-6,
            ).to(self.device)
            logger.info("Initialized MultiPhysicsBlochLayer for inference")
        except ImportError as e:
            logger.warning(
                f"MultiPhysicsBlochLayer not available: {e}. Tissue param synthesis disabled."
            )
            # Log traceback for debugging
            import traceback

            logger.debug(traceback.format_exc())

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.DISENTANGLED

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input tensor for disentangled inference.

        Args:
            input_tensor: Raw input tensor
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor ready for encoding
        """
        # Ensure tensor is on correct device
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        input_tensor = input_tensor.detach()

        # Handle channel adaptation
        expected_channels = getattr(self.model, "in_channels", 1)
        if input_tensor.shape[1] != expected_channels:
            if input_tensor.shape[1] == 2 and expected_channels == 1:
                # Complex (real/imag) → Magnitude
                input_tensor = torch.sqrt(
                    input_tensor[:, 0:1] ** 2 + input_tensor[:, 1:2] ** 2 + 1e-6
                )
            elif input_tensor.shape[1] == 1 and expected_channels == 2:
                # Single channel → Repeat for complex
                input_tensor = input_tensor.repeat(1, 2, 1, 1)

        return input_tensor

    def run_inference(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run disentangled inference.

        Args:
            input_tensor: Preprocessed input tensor
            **kwargs: Additional inference parameters
                - target_style: Optional style vector for transfer
                - target_image: Optional image to extract style from
                - output_mode: 'image', 'tissue_params', or 'both'

        Returns:
            Output tensor or tuple of (output, metadata)
        """
        target_style = kwargs.get("target_style", self.target_style)
        target_image = kwargs.get("target_image")
        output_mode = kwargs.get("output_mode", "image")

        with torch.no_grad():
            # 1. Encode content (anatomy)
            content = self.model.enc_c(input_tensor)

            # 2. Get style vector
            if target_style is not None:
                # Use provided style vector
                style = target_style
                if style.device != self.device:
                    style = style.to(self.device)
            elif target_image is not None:
                # Extract style from target image
                if target_image.device != self.device:
                    target_image = target_image.to(self.device)
                style = self.model.enc_s(target_image)
            else:
                # Use input's own style (reconstruction mode)
                style = self.model.enc_s(input_tensor)

            # 3. Generate output
            decoder_output = self.model.gen(content, style)

            # 4. Handle output based on mode
            metadata = {
                "content_shape": content.shape,
                "style_shape": style.shape,
            }

            if isinstance(decoder_output, dict):
                # Tissue parameters mode
                tissue_params = decoder_output
                metadata["tissue_params"] = {k: v.detach().cpu() for k, v in tissue_params.items()}

                if output_mode == "tissue_params":
                    return tissue_params, metadata
                elif output_mode == "both":
                    # Synthesize image via Bloch
                    image = self._synthesize_via_bloch(tissue_params, kwargs)
                    return (image, tissue_params), metadata
                else:
                    # Default: synthesize image
                    image = self._synthesize_via_bloch(tissue_params, kwargs)
                    return image, metadata
            else:
                # Direct image output
                return decoder_output, metadata

    def _synthesize_via_bloch(
        self, tissue_params: dict[str, torch.Tensor], kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Synthesize MRI image from tissue parameters via Bloch equation.

        Args:
            tissue_params: Dict with 'rho', 't1', 't2' tensors
            kwargs: Additional parameters (TR, TE, TI, sequence_type)
        Returns:
            Synthesized MRI image tensor
        """
        if self._bloch_layer is None:
            # Fallback: just return rho (proton density)
            logger.warning("Bloch layer not available, returning proton density as output")
            return tissue_params["rho"]

        # Get sequence parameters
        tr = kwargs.get("tr", 2000.0)  # ms
        te = kwargs.get("te", 30.0)  # ms
        ti = kwargs.get("ti")  # ms (for IR sequences)
        sequence_type = kwargs.get("sequence_type", "se")  # se, ir, gre

        # Construct metadata for MultiPhysicsBlochLayer
        metadata = {
            "TR": tr,
            "TE": te,
            "TI": ti if ti is not None else 0.0,
            "contrast_type": sequence_type,
        }

        # Synthesize via Bloch
        synthesized = self._bloch_layer(
            tissue_params=tissue_params,
            metadata=metadata,
        )

        return synthesized

    def postprocess_output(
        self,
        output_tensor: torch.Tensor | tuple[torch.Tensor, dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Postprocess disentangled output.

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

        # Handle both tensor and dict outputs
        if isinstance(output, dict):
            # Tissue params - just move to CPU
            output = {k: v.detach().cpu() for k, v in output.items()}
        elif isinstance(output, tuple):
            # (image, tissue_params) tuple
            image, tissue_params = output
            image = image.detach().cpu()
            tissue_params = {k: v.detach().cpu() for k, v in tissue_params.items()}
            output = (image, tissue_params)
        else:
            # Single tensor
            output = output.detach().cpu()

            # Normalize to [0, 1] if needed
            if self.disentangled_config.get("normalize_output", True):
                # Handle tanh output [-1, 1] → [0, 1]
                if output.min() < 0:
                    output = (output + 1.0) / 2.0
                output = torch.clamp(output, 0.0, 1.0)

        return output, metadata

    def extract_content(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract content (anatomy) encoding from input.

        Args:
            input_tensor: Input image tensor

        Returns:
            Content vector/feature map
        """
        processed_input = self.preprocess_input(input_tensor)
        with torch.no_grad():
            content = self.model.enc_c(processed_input)
        return content.detach().cpu()

    def extract_style(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract style (contrast/physics) encoding from input.

        Args:
            input_tensor: Input image tensor

        Returns:
            Style vector
        """
        processed_input = self.preprocess_input(input_tensor)
        with torch.no_grad():
            style = self.model.enc_s(processed_input)
        return style.detach().cpu()

    def style_transfer(
        self, content_image: torch.Tensor, style_image: torch.Tensor
    ) -> torch.Tensor:
        """Perform style transfer (anatomy A + contrast B).

        Args:
            content_image: Image providing anatomy (content)
            style_image: Image providing contrast (style)

        Returns:
            Synthesized image with content A + style B
        """
        content_input = self.preprocess_input(content_image)
        style_input = self.preprocess_input(style_image)

        with torch.no_grad():
            content = self.model.enc_c(content_input)
            style = self.model.enc_s(style_input)
            output = self.model.gen(content, style)

            if isinstance(output, dict):
                # Tissue params mode - synthesize
                output = self._synthesize_via_bloch(output, {})

        return self.postprocess_output(output)[0]

    def set_target_style(self, style_vector: torch.Tensor) -> None:
        """Set a fixed target style for all subsequent inferences.

        Args:
            style_vector: Style vector to use for decoding
        """
        self.target_style = style_vector.to(self.device)
        logger.info(f"Set target style vector with shape {style_vector.shape}")

    def clear_target_style(self) -> None:
        """Clear the fixed target style (return to reconstruction mode)."""
        self.target_style = None
        logger.info("Cleared target style, returning to reconstruction mode")

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this disentangled inference strategy."""
        info = super().get_strategy_info()
        info.update(
            {
                "disentangled_config": self.disentangled_config,
                "inference_mode": self.inference_mode,
                "output_tissue_params": self.output_tissue_params,
                "has_target_style": self.target_style is not None,
                "bloch_layer_available": self._bloch_layer is not None,
            }
        )
        return info

    @torch.no_grad()
    def infer_single(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run inference on a single input tensor.

        Args:
            input_data: Single input tensor
            **kwargs: Additional parameters

        Returns:
            Output tensor
        """
        return self.infer(input_data, **kwargs)

    @torch.no_grad()
    def infer_batch(self, input_data_list: list, batch_size: int = 4, **kwargs) -> list:
        """Run inference on a batch of input tensors.

        Args:
            input_data_list: List of input tensors
            batch_size: Batch size for processing
            **kwargs: Additional parameters

        Returns:
            List of output tensors
        """
        results = []
        for input_data in input_data_list:
            result = self.infer_single(input_data, **kwargs)
            results.append(result)
        return results
