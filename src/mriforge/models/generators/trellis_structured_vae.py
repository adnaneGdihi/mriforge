"""TRELLIS Structured Latent VAE Generators.

These classes wrap the reference TRELLIS Structured Latent VAE (SLat VAE)
implementations so they can be created via the shared :class:`ModelFactory`.
The wrapper focuses on clean construction and data-shape validation while
leaving heavy lifting to the upstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mriforge.infrastructure.integration.trellis_adapter import (
    TrellisImportError,
    ensure_trellis_path,
    import_trellis_module,
)
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@dataclass
class TrellisStructuredVAEOutput:
    """Container returned by the forward pass."""

    representation: Any
    latent: Any
    mean: torch.Tensor
    logvar: torch.Tensor


class TrellisStructuredVAEBase(nn.Module, IGenerator):
    """Base wrapper for Structured Latent VAE variants."""

    decoder_module: str = ""
    decoder_class_name: str = ""
    representation_key: str = "representation"

    def __init__(
        self,
        encoder_config: dict[str, Any] | None = None,
        decoder_config: dict[str, Any] | None = None,
        trainer_config: dict[str, Any] | None = None,
        sample_posterior: bool = True,
        trellis_root: str | None = None,
        **_: Any,
    ) -> None:
        """__init__.

        Args:
            encoder_config (dict[str, Any] | None): Description.
            decoder_config (dict[str, Any] | None): Description.
            trainer_config (dict[str, Any] | None): Description.
            sample_posterior (bool): Description.
            trellis_root (str | None): Description.
        """
        super().__init__()

        encoder_config = encoder_config or {}
        decoder_config = decoder_config or {}
        self.trainer_config = trainer_config or {}
        self.sample_posterior_default = sample_posterior

        trellis_path = Path(trellis_root) if trellis_root else None

        try:
            ensure_trellis_path(trellis_path)
            sparse_module = import_trellis_module("trellis.modules.sparse")
            encoder_module = import_trellis_module("trellis.models.structured_latent_vae.encoder")
            decoder_module = import_trellis_module(self.decoder_module)

            self._sparse = sparse_module
            encoder_cls = encoder_module.ElasticSLatEncoder
            decoder_cls = getattr(decoder_module, self.decoder_class_name)
            self.encoder = encoder_cls(**encoder_config)
            self.decoder = decoder_cls(**decoder_config)
            self._dependencies_available = True

        except TrellisImportError as exc:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"TRELLIS dependencies missing: {exc}. "
                "Model initialized in zombie mode (will crash on forward)."
            )
            self.encoder = None
            self.decoder = None
            self._sparse = None
            self._dependencies_available = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def forward(
        self,
        sparse_tensor: Any,
        *,
        sample_posterior: bool | None = None,
        return_output: bool = True,
    ) -> TrellisStructuredVAEOutput | Any:
        """Encode and decode a structured sparse tensor.

        Args:
            sparse_tensor: Expected to be a
                ``trellis.modules.sparse.SparseTensor`` instance.
            sample_posterior: Override the default reparameterisation flag.
            return_output: When ``False`` only the decoded representation is
                returned (useful when the caller is not interested in latent
                statistics).

        forward method for TrellisStructuredVAEBase.

        Executes PyTorch tensor operations.

        Args:
            sparse_tensor (Any): Expected input tensor.

        Returns:
            TrellisStructuredVAEOutput | Any: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""

        if not self._dependencies_available:
            raise RuntimeError("TRELLIS dependencies are required for forward pass.")

        expected_type = self._sparse.SparseTensor
        if not isinstance(sparse_tensor, expected_type):
            msg = (
                "TRELLIS structured VAEs expect inputs in sparse tensor "
                "format. "
                f"Received type: {type(sparse_tensor)!r}."
            )
            raise TypeError(msg)

        use_posterior = (
            sample_posterior if sample_posterior is not None else self.sample_posterior_default
        )

        latent, mean, logvar = self.encoder(
            sparse_tensor,
            sample_posterior=use_posterior,
            return_raw=True,
        )
        representation = self.decoder(latent)

        if not return_output:
            return representation

        return TrellisStructuredVAEOutput(
            representation=representation,
            latent=latent,
            mean=mean,
            logvar=logvar,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def encode(self, sparse_tensor: Any, *, sample_posterior: bool = True) -> Any:
        """encode.

        Args:
            sparse_tensor (Any): Description.
            sample_posterior (bool): Description.
        Returns:
            Any: Description.
        """
        if not self._dependencies_available:
            raise RuntimeError("TRELLIS dependencies are required for encode.")
        return self.encoder(
            sparse_tensor,
            sample_posterior=sample_posterior,
            return_raw=False,
        )

    def decode(self, latent: Any) -> Any:
        """decode.

        Args:
            latent (Any): Description.
        Returns:
            Any: Description.
        """
        if not self._dependencies_available:
            raise RuntimeError("TRELLIS dependencies are required for decode.")
        return self.decoder(latent)

    def generate(self, latent: Any, **_: Any) -> Any:  # pragma: no cover - thin wrapper
        """generate.

        Args:
            latent (Any): Description.
        Returns:
            Any: Description.
        """
        return self.decode(latent)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape.

        For TRELLIS, the output shape depends on the chosen renderer decoder.
        It doesn't strictly follow standard (B, C, H, W) paradigms but outputs
        specific primitive counts like Gaussians or Vertices across sparse voxels.
        """
        if hasattr(self, "decoder") and hasattr(self.decoder, "output_dim"):
            # Usually SLat decoders output (N, output_dim) where N is number of sparse voxels
            return (input_shape[0], -1, self.decoder.output_dim)

        # Heuristics based on model type
        if self.representation_key == "gaussian":
            return (input_shape[0], -1, 14)  # 14 params per gaussian splat
        elif self.representation_key == "mesh":
            return (input_shape[0], -1, 3)  # 3D vertices

        return input_shape  # Fallback

    def get_parameter_count(self) -> int:
        """Returns total parameter count."""
        if not self._dependencies_available:
            return 0
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        return enc_params + dec_params

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return f"trellis_{self.representation_key}_vae"


@register_model(name="trellis_gaussian_vae", training_mode="vae")
class TrellisStructuredGaussianVAEGenerator(TrellisStructuredVAEBase):
    """TrellisStructuredGaussianVAEGenerator class."""

    decoder_module = "trellis.models.structured_latent_vae.decoder_gs"
    decoder_class_name = "ElasticSLatGaussianDecoder"
    representation_key = "gaussian"


@register_model(name="trellis_mesh_vae", training_mode="vae")
class TrellisStructuredMeshVAEGenerator(TrellisStructuredVAEBase):
    """TrellisStructuredMeshVAEGenerator class."""

    decoder_module = "trellis.models.structured_latent_vae.decoder_mesh"
    decoder_class_name = "ElasticSLatMeshDecoder"
    representation_key = "mesh"
