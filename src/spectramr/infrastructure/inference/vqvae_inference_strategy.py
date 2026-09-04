"""VQ-VAE Inference Strategy

Inference strategy for Vector Quantized VAE models.
Handles discrete codebook encoding/decoding for latent space compression.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class VQVAEInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for VQ-VAE models.

    VQ-VAE uses discrete codebook quantization:
    - Encoder: Maps input to continuous latent space
    - Vector Quantizer: Maps continuous latent to discrete codebook indices
    - Decoder: Reconstructs from quantized latent vectors

    Inference Modes:
    1. Reconstruction: Encode → Quantize → Decode
    2. Latent Analysis: Extract codebook indices for analysis
    3. Latent Editing: Modify codebook indices and decode
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize VQ-VAE inference strategy.

        Args:
            model: The VQ-VAE model
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)

        # Extract VQ-VAE-specific config
        self.vqvae_config = self.config.get("vqvae", {})

        # Codebook parameters
        self.num_embeddings = self.vqvae_config.get("num_embeddings", 512)
        self.embedding_dim = self.vqvae_config.get("embedding_dim", 64)

        # Whether to return codebook indices
        self.return_indices = self.vqvae_config.get("return_indices", False)

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.VAE  # VQ-VAE falls under VAE category

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input tensor for VQ-VAE inference.

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
                # Single channel → Repeat
                input_tensor = input_tensor.repeat(1, 2, 1, 1)

        return input_tensor

    def run_inference(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run VQ-VAE inference.

        Args:
            input_tensor: Preprocessed input tensor
            **kwargs: Additional inference parameters
                - return_indices: If True, return codebook indices
                - latent_only: If True, return only latent representation

        Returns:
            Reconstructed output or tuple of (output, metadata)
        """
        return_indices = kwargs.get("return_indices", self.return_indices)
        latent_only = kwargs.get("latent_only", False)

        with torch.no_grad():
            # Encode to continuous latent
            if hasattr(self.model, "encode"):
                z_e = self.model.encode(input_tensor)
            else:
                z_e = self.model.encoder(input_tensor)

            # Vector Quantization
            if hasattr(self.model, "quantize"):
                z_q, indices, commit_loss = self.model.quantize(z_e)
            elif hasattr(self.model, "vq"):
                z_q, indices, commit_loss = self.model.vq(z_e)
            else:
                # Try direct model call (handles quantization internally)
                output = self.model(input_tensor)
                if isinstance(output, dict):
                    z_q = output.get("z_q", output.get("quantized"))
                    indices = output.get("indices", output.get("encoding_indices"))
                    reconstruction = output.get(
                        "reconstruction", output.get("output", output.get("x_recon"))
                    )
                    return reconstruction, {"indices": indices, "z_q": z_q}
                return output, {}

            metadata = {
                "z_e_shape": z_e.shape,
                "z_q_shape": z_q.shape,
            }

            if return_indices:
                metadata["indices"] = indices.detach().cpu()
                metadata["z_e"] = z_e.detach().cpu()
                metadata["z_q"] = z_q.detach().cpu()

            if latent_only:
                return z_q, metadata

            # Decode from quantized latent
            if hasattr(self.model, "decode"):
                reconstruction = self.model.decode(z_q)
            else:
                reconstruction = self.model.decoder(z_q)

            return reconstruction, metadata

    def postprocess_output(
        self,
        output_tensor: torch.Tensor | tuple[torch.Tensor, dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Postprocess VQ-VAE output.

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

        # Normalize to [0, 1] if needed
        if self.vqvae_config.get("normalize_output", True):
            # Handle tanh output [-1, 1] → [0, 1]
            if output.min() < 0:
                output = (output + 1.0) / 2.0
            output = torch.clamp(output, 0.0, 1.0)

        return output, metadata

    def encode_to_indices(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Encode input to discrete codebook indices.

        Args:
            input_tensor: Input image tensor

        Returns:
            Codebook indices tensor
        """
        processed_input = self.preprocess_input(input_tensor)
        _, metadata = self.run_inference(processed_input, return_indices=True, latent_only=True)
        return metadata.get("indices")

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices.

        Args:
            indices: Codebook indices tensor

        Returns:
            Decoded output tensor
        """
        if indices.device != self.device:
            indices = indices.to(self.device)

        with torch.no_grad():
            # Get embeddings from codebook
            if hasattr(self.model, "quantize"):
                codebook = self.model.quantize.embedding.weight
            elif hasattr(self.model, "vq"):
                codebook = self.model.vq.embedding.weight
            else:
                # Try to find codebook in model
                codebook = getattr(self.model, "codebook", getattr(self.model, "embedding", None))
                if codebook is not None:
                    codebook = codebook.weight

            if codebook is None:
                raise RuntimeError("Could not find codebook in model")

            # Index into codebook
            z_q = codebook[indices.flatten()]
            z_q = z_q.view(*indices.shape, -1)

            # Rearrange to (B, C, H, W)
            if z_q.ndim == 4:
                z_q = z_q.permute(0, 3, 1, 2)

            # Decode
            if hasattr(self.model, "decode"):
                reconstruction = self.model.decode(z_q)
            else:
                reconstruction = self.model.decoder(z_q)

        return self.postprocess_output(reconstruction)[0]

    def get_codebook_usage(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Analyze codebook usage for input.

        Args:
            input_tensor: Input image tensor

        Returns:
            Dict with usage statistics
        """
        indices = self.encode_to_indices(input_tensor)
        unique_indices = torch.unique(indices)

        return {
            "indices": indices,
            "unique_count": len(unique_indices),
            "unique_indices": unique_indices,
            "utilization": len(unique_indices) / self.num_embeddings,
        }

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this VQ-VAE inference strategy."""
        info = super().get_strategy_info()
        info.update(
            {
                "vqvae_config": self.vqvae_config,
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "return_indices": self.return_indices,
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
