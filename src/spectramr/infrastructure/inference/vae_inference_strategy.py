"""VAE Inference Strategy

Inference strategy for Variational Autoencoders.
Handles latent space encoding/decoding and KL divergence.
"""

from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes

from .base_inference_strategy import BaseInferenceStrategy


class VAEInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for VAE models.

    Handles encoding to latent space and decoding for reconstruction.
    Supports both deterministic and stochastic decoding.
    """

    # Phase 2: encode/decode is single-pass per tile — compatible with
    # tiling. Stochastic sampling adds noise but the noise is
    # independent across tiles, so Hann aggregation still smooths
    # tile borders correctly.
    supports_grid_tiling: bool = True

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize VAE inference strategy.

        Args:
            model: The VAE model (encoder-decoder)
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)
        self.vae_config = self.config.get("vae", {})

        # VAE-specific parameters
        self.latent_dim = self.vae_config.get("latent_dim", 128)
        self.stochastic_decoding = self.vae_config.get("stochastic_decoding", False)
        self.kl_weight = self.vae_config.get("kl_weight", 1.0)

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.vae

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input tensor for VAE inference.

        Args:
            input_tensor: Raw input tensor
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor ready for VAE encoding
        """
        # Ensure tensor is on correct device
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        input_tensor = input_tensor.detach()

        # Apply any encoder-specific preprocessing
        if hasattr(self.model, "preprocess_input"):
            input_tensor = self.model.preprocess_input(input_tensor)

        return input_tensor

    def run_inference(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Run VAE inference (encoding + decoding).

        Args:
            input_tensor: Preprocessed input tensor
            **kwargs: Additional inference parameters

        Returns:
            Reconstructed output tensor, optionally with latent info
        """
        with torch.no_grad():
            # Encode to latent space
            if hasattr(self.model, "encode"):
                # Model has separate encode method
                latent_params = self.model.encode(input_tensor)

                if isinstance(latent_params, tuple):
                    # VAE returns (mean, log_var)
                    mu, log_var = latent_params

                    if self.stochastic_decoding:
                        # Sample from latent distribution
                        std = torch.exp(0.5 * log_var)
                        eps = torch.randn_like(std)
                        z = mu + eps * std
                    else:
                        # Use mean for deterministic decoding
                        z = mu

                    # Store latent info for metadata
                    latent_info = {
                        "mu": mu.detach().cpu(),
                        "log_var": log_var.detach().cpu(),
                        "z": z.detach().cpu(),
                    }
                else:
                    # Model returns latent directly
                    z = latent_params
                    latent_info = {"z": z.detach().cpu()}
            else:
                # Model handles encoding internally
                z = self.model.encode_to_latent(input_tensor)
                latent_info = {"z": z.detach().cpu()}

            # Decode from latent space
            if hasattr(self.model, "decode"):
                reconstruction = self.model.decode(z)
            else:
                reconstruction = self.model.decode_from_latent(z)

            # Handle different model output formats
            if isinstance(reconstruction, dict):
                if "reconstruction" in reconstruction:
                    reconstruction = reconstruction["reconstruction"]
                elif "output" in reconstruction:
                    reconstruction = reconstruction["output"]
                else:
                    reconstruction = next(iter(reconstruction.values()))

        # Return reconstruction with latent metadata
        return reconstruction, latent_info

    def postprocess_output(
        self,
        output_tensor: torch.Tensor | tuple[torch.Tensor, dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Postprocess VAE output.

        Args:
            output_tensor: Raw VAE output (reconstruction or tuple)
            **kwargs: Additional postprocessing parameters

        Returns:
            Postprocessed output with latent metadata
        """
        if isinstance(output_tensor, tuple):
            reconstruction, latent_info = output_tensor
        else:
            reconstruction = output_tensor
            latent_info = {}

        # Ensure reconstruction is detached
        reconstruction = reconstruction.detach().cpu()

        # Apply decoder-specific postprocessing
        if hasattr(self.model, "postprocess_output"):
            reconstruction = self.model.postprocess_output(reconstruction)

        # Apply final activation if specified
        final_activation = self.vae_config.get("final_activation", "")
        if final_activation.lower() == "tanh":
            reconstruction = torch.tanh(reconstruction)
        elif final_activation.lower() == "sigmoid":
            reconstruction = torch.sigmoid(reconstruction)

        # Normalize to [0, 1] range
        if self.vae_config.get("normalize_output", True):
            reconstruction = (reconstruction + 1.0) / 2.0

        # Clamp outputs to valid range
        reconstruction = torch.clamp(reconstruction, 0.0, 1.0)

        return reconstruction, latent_info

    def encode_to_latent(
        self, input_tensor: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Encode input to latent space only.

        Args:
            input_tensor: Input tensor to encode
            **kwargs: Additional encoding parameters

        Returns:
            Tuple of (latent_vector, metadata)
        """
        processed_input = self.preprocess_input(input_tensor, **kwargs)

        with torch.no_grad():
            if hasattr(self.model, "encode"):
                latent_params = self.model.encode(processed_input)

                if isinstance(latent_params, tuple):
                    mu, log_var = latent_params
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    z = mu + eps * std

                    latent_info = {
                        "mu": mu.detach().cpu(),
                        "log_var": log_var.detach().cpu(),
                        "std": std.detach().cpu(),
                        "z": z.detach().cpu(),
                    }
                else:
                    z = latent_params
                    latent_info = {"z": z.detach().cpu()}
            else:
                z = self.model.encode_to_latent(processed_input)
                latent_info = {"z": z.detach().cpu()}

        return z.detach().cpu(), latent_info

    def decode_from_latent(self, latent_vector: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode from latent space.

        Args:
            latent_vector: Latent vector to decode
            **kwargs: Additional decoding parameters

        Returns:
            Decoded output tensor
        """
        if latent_vector.device != self.device:
            latent_vector = latent_vector.to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "decode"):
                reconstruction = self.model.decode(latent_vector)
            else:
                reconstruction = self.model.decode_from_latent(latent_vector)

            # Handle different model output formats
            if isinstance(reconstruction, dict):
                reconstruction = reconstruction.get(
                    "reconstruction", reconstruction.get("output", reconstruction)
                )

        # Postprocess the reconstruction
        return self.postprocess_output(reconstruction, **kwargs)[0]

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this VAE inference strategy."""
        info = super().get_strategy_info()
        info.update(
            {
                "vae_config": self.vae_config,
                "latent_dim": self.latent_dim,
                "stochastic_decoding": self.stochastic_decoding,
                "kl_weight": self.kl_weight,
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
    def infer_batch(
        self, input_data_list: list[torch.Tensor], batch_size: int = 4, **kwargs
    ) -> list[torch.Tensor]:
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
