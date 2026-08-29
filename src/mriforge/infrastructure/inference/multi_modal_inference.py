"""Multi-Modal Inference Support

Enhanced inference capabilities for multi-modal inputs including
text-to-image, image-to-image, and cross-modal generation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


class MultiModalConditioning:
    """Handles multi-modal conditioning for inference."""

    def __init__(self, device: torch.device):
        """Initialize multi-modal conditioning handler.

        Args:
            device: Device for tensor operations
        """
        self.device = device
        self.conditioning_modules = {}

    def register_conditioning_module(
        self,
        modality: str,
        module: nn.Module,
        preprocessor: Callable | None = None,
    ):
        """Register a conditioning module for a specific modality.

        Args:
            modality: Type of conditioning ('text', 'image', 'mask', etc.)
            module: Conditioning module (encoder/embedder)
            preprocessor: Optional preprocessing function
        """
        self.conditioning_modules[modality] = {
            "module": module,
            "preprocessor": preprocessor,
        }

    def encode_conditioning(
        self,
        conditioning_data: dict[str, Any],
        target_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Encode multi-modal conditioning data with dimension validation.

        Args:
            conditioning_data: Dictionary with modality keys and data
            target_shape: Target shape for conditioning tensor

        Returns:
            Encoded conditioning tensor

        Raises:
            ValueError: If conditioning data has invalid dimensions or modalities
            RuntimeError: If encoding fails due to shape mismatch
        """
        if not isinstance(conditioning_data, dict):
            raise ValueError(f"conditioning_data must be dict, got {type(conditioning_data)}")

        conditioning_tensors = []

        for modality, data in conditioning_data.items():
            if modality not in self.conditioning_modules:
                raise ValueError(
                    f"Unknown conditioning modality: {modality}. "
                    f"Registered: {list(self.conditioning_modules.keys())}"
                )

            module_info = self.conditioning_modules[modality]
            module = module_info["module"]
            preprocessor = module_info["preprocessor"]

            # Preprocess if needed
            if preprocessor is not None:
                data = preprocessor(data)

            # Validate tensor
            if not isinstance(data, torch.Tensor):
                raise ValueError(
                    f"Conditioning data for {modality} must be tensor after preprocessing, "
                    f"got {type(data)}"
                )

            # Validate shape - ensure at least 2D (batch, features)
            if data.ndim < 2:
                raise ValueError(
                    f"Conditioning data for {modality} must be at least 2D, got shape {data.shape}"
                )

            # Encode
            with torch.no_grad():
                if data.device != self.device:
                    data = data.to(self.device)

                try:
                    encoded = module(data)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to encode {modality} conditioning with shape {data.shape}: {e!s}"
                    )

            # Validate encoded output
            if not isinstance(encoded, torch.Tensor):
                raise TypeError(
                    f"Conditioning module for {modality} must return tensor, got {type(encoded)}"
                )

            if encoded.ndim < 2:
                raise ValueError(
                    f"Encoded {modality} conditioning must be at least 2D, got shape {encoded.shape}"
                )

            conditioning_tensors.append(encoded)

        if not conditioning_tensors:
            # Return zero tensor if no conditioning
            if target_shape:
                if not isinstance(target_shape, (tuple, list)):
                    raise ValueError(
                        f"target_shape must be tuple or list, got {type(target_shape)}"
                    )
                return torch.zeros(target_shape, device=self.device)
            else:
                return torch.zeros(1, 512, device=self.device)  # Default shape

        # Validate all tensors have compatible batch dimension
        batch_sizes = [t.shape[0] for t in conditioning_tensors]
        if len(set(batch_sizes)) > 1:
            raise ValueError(
                f"All conditioning tensors must have same batch size, "
                f"got: {dict(zip(conditioning_data.keys(), batch_sizes, strict=False))}"
            )

        # Validate feature dimensions before concatenation
        try:
            combined_conditioning = torch.cat(conditioning_tensors, dim=-1)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to concatenate conditioning tensors. "
                f"Shapes: {[t.shape for t in conditioning_tensors]}. "
                f"Error: {e!s}"
            )

        # Resize if target shape specified
        if target_shape:
            if not isinstance(target_shape, (tuple, list)):
                raise ValueError(f"target_shape must be tuple or list, got {type(target_shape)}")

            if combined_conditioning.shape != target_shape:
                try:
                    # Validate target shape dimensions
                    if len(target_shape) < 2:
                        raise ValueError(
                            f"target_shape must be at least 2D, got {len(target_shape)}D"
                        )

                    # Simple resizing - in practice, you might want more sophisticated methods
                    combined_conditioning = torch.nn.functional.interpolate(
                        combined_conditioning.unsqueeze(0),
                        size=target_shape[1:],
                        mode="bilinear" if len(target_shape) == 3 else "trilinear",
                        align_corners=False,
                    ).squeeze(0)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to resize conditioning tensor from {combined_conditioning.shape} "
                        f"to {target_shape}: {e!s}"
                    )

        return combined_conditioning


class TextToImageConditioning:
    """Text-to-image conditioning using CLIP or similar text encoders."""

    def __init__(self, text_encoder: nn.Module, tokenizer: Any, device: torch.device):
        """Initialize text-to-image conditioning.

        Args:
            text_encoder: Text encoder model
            tokenizer: Text tokenizer
            device: Device for operations
        """
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        """Encode text prompt(s) to conditioning tensor with validation.

        Args:
            text: Text prompt or list of prompts

        Returns:
            Text conditioning tensor

        Raises:
            TypeError: If text is not string or list of strings
            ValueError: If text is empty or encoding fails
        """
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise TypeError(f"text must be str or list of str, got {type(text)}")

        if not text:
            raise ValueError("text cannot be empty")

        # Validate all items are strings
        if not all(isinstance(t, str) for t in text):
            raise TypeError("All items in text list must be strings")

        try:
            # Tokenize
            tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(
                self.device
            )
        except Exception as e:
            raise ValueError(f"Failed to tokenize text: {e!s}")

        # Encode
        with torch.no_grad():
            try:
                text_embeddings = self.text_encoder(**tokens).last_hidden_state
            except Exception as e:
                raise RuntimeError(
                    f"Failed to encode text with shape {tokens['input_ids'].shape}: {e!s}"
                )

        # Validate output
        if not isinstance(text_embeddings, torch.Tensor):
            raise TypeError(f"Text encoder must return tensor, got {type(text_embeddings)}")

        if text_embeddings.shape[0] != len(text):
            raise ValueError(
                f"Text encoder output batch size {text_embeddings.shape[0]} "
                f"doesn't match input batch size {len(text)}"
            )

        # Pool to get single embedding per prompt
        text_embeddings = text_embeddings.mean(dim=1)  # Average pooling

        return text_embeddings


class ImageToImageConditioning:
    """Image-to-image conditioning for editing and translation."""

    def __init__(self, image_encoder: nn.Module, device: torch.device):
        """Initialize image-to-image conditioning.

        Args:
            image_encoder: Image encoder model
            device: Device for operations
        """
        self.image_encoder = image_encoder.to(device)
        self.device = device

    def encode_image(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode source image for image-to-image translation with validation.

        Args:
            image: Source image tensor (B, C, H, W)
            mask: Optional mask for inpainting (B, 1, H, W)

        Returns:
            Image conditioning tensor

        Raises:
            TypeError: If inputs are not tensors
            ValueError: If tensor shapes are invalid
            RuntimeError: If encoding fails
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image must be tensor, got {type(image)}")

        if image.ndim != 4:
            raise ValueError(f"image must be 4D (B, C, H, W), got shape {image.shape}")

        if image.shape[0] == 0:
            raise ValueError("image batch size cannot be 0")

        if image.device != self.device:
            image = image.to(self.device)

        # Validate and process mask if provided
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"mask must be tensor, got {type(mask)}")

            if mask.ndim != 4:
                raise ValueError(f"mask must be 4D (B, 1, H, W), got shape {mask.shape}")

            if mask.shape[0] != image.shape[0]:
                raise ValueError(
                    f"mask batch size {mask.shape[0]} doesn't match image batch size {image.shape[0]}"
                )

            if mask.shape[2:] != image.shape[2:]:
                raise ValueError(
                    f"mask spatial size {mask.shape[2:]} doesn't match image spatial size {image.shape[2:]}"
                )

            if mask.device != self.device:
                mask = mask.to(self.device)

        try:
            with torch.no_grad():
                image_embedding = self.image_encoder(image)
        except Exception as e:
            raise RuntimeError(f"Failed to encode image with shape {image.shape}: {e!s}")

        if not isinstance(image_embedding, torch.Tensor):
            raise TypeError(f"Image encoder must return tensor, got {type(image_embedding)}")

        # Concatenate with mask if provided
        if mask is not None:
            # Resize mask to match embedding dimensions if needed
            if mask.shape[-2:] != image_embedding.shape[-2:]:
                try:
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0) if mask.ndim == 3 else mask,
                        size=image_embedding.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to resize mask from {mask.shape} to {image_embedding.shape[-2:]}: {e!s}"
                    )

            if mask.ndim == 5:
                mask = mask.squeeze(0)

            if mask.shape[1] != 1:
                raise ValueError(f"mask must have 1 channel, got {mask.shape[1]}")

            try:
                image_embedding = torch.cat([image_embedding, mask], dim=1)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to concatenate image and mask. "
                    f"Image shape: {image_embedding.shape}, Mask shape: {mask.shape}. "
                    f"Error: {e!s}"
                )

        return image_embedding


class MultiModalInferenceMixin:
    """Mixin class providing multi-modal inference capabilities."""

    def __init__(self, *args, **kwargs):
        """__init__."""
        super().__init__(*args, **kwargs)
        self.multi_modal_conditioning = MultiModalConditioning(self.device)

    def setup_text_to_image(
        self,
        text_encoder: nn.Module,
        tokenizer: Any,
    ):
        """Setup text-to-image conditioning.

        Args:
            text_encoder: Text encoder model
            tokenizer: Text tokenizer
        """
        self.text_conditioning = TextToImageConditioning(text_encoder, tokenizer, self.device)
        self.multi_modal_conditioning.register_conditioning_module(
            "text",
            self.text_conditioning.text_encoder,
            lambda x: self.text_conditioning.encode_text(x),
        )

    def setup_image_to_image(self, image_encoder: nn.Module):
        """Setup image-to-image conditioning.

        Args:
            image_encoder: Image encoder model
        """
        self.image_conditioning = ImageToImageConditioning(image_encoder, self.device)
        self.multi_modal_conditioning.register_conditioning_module(
            "image",
            self.image_conditioning.image_encoder,
            lambda x: self.image_conditioning.encode_image(x[0], x[1] if len(x) > 1 else None),
        )

    def prepare_multi_modal_conditioning(
        self,
        conditioning_data: dict[str, Any],
        target_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Prepare multi-modal conditioning tensor.

        Args:
            conditioning_data: Dictionary with conditioning data
            target_shape: Target shape for conditioning

        Returns:
            Combined conditioning tensor
        """
        return self.multi_modal_conditioning.encode_conditioning(conditioning_data, target_shape)

    def text_to_image_inference(self, text_prompt: str | list[str], **kwargs) -> torch.Tensor:
        """Run text-to-image inference.

        Args:
            text_prompt: Text prompt(s) for generation
            **kwargs: Additional inference parameters

        Returns:
            Generated image tensor
        """
        # Encode text conditioning
        text_conditioning = self.text_conditioning.encode_text(text_prompt)

        # Run inference with text conditioning
        return self.run_inference(text_conditioning, **kwargs)

    def image_to_image_inference(
        self,
        source_image: torch.Tensor,
        mask: torch.Tensor | None = None,
        strength: float = 0.8,
        **kwargs,
    ) -> torch.Tensor:
        """Run image-to-image inference.

        Args:
            source_image: Source image for editing
            mask: Optional mask for inpainting
            strength: Strength of conditioning (0-1)
            **kwargs: Additional inference parameters

        Returns:
            Edited image tensor
        """
        # Encode image conditioning
        image_conditioning = self.image_conditioning.encode_image(source_image, mask)

        # Add noise to source image based on strength
        if strength < 1.0:
            noise = torch.randn_like(source_image)
            # Add noise inversely proportional to strength
            noise_level = 1.0 - strength
            noisy_image = source_image * strength + noise * noise_level
        else:
            noisy_image = source_image

        # Run inference with image conditioning
        return self.run_inference(image_conditioning, initial_noise=noisy_image, **kwargs)
