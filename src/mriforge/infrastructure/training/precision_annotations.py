"""
Precision Type Annotations for AMP Training

Provides type hints for tensor precision to enable compile-time detection
of precision mismatches and runtime logging of all precision transitions.

This solves the silent casting problem in VAE/VQVAE training where .float()
and .half() calls accumulate without visibility, causing numerical instability.

Phase 4b-1: Explicit precision annotations with logging enable debugging
precision flow and catching mismatches at code review time.

Usage:
    from mriforge.infrastructure.training.precision_annotations import Tensor_FP16, Tensor_FP32

    mu_fp16: Tensor_FP16 = encoder_output  # Type hint - encoder assumed FP16
    mu_fp32: Tensor_FP32 = cast_to_fp32(mu_fp16, reason="kl_divergence_stability")
"""

import logging
from typing import Annotated

import torch

logger = logging.getLogger(__name__)

# Type aliases for precision-specific tensors
# These don't enforce anything at runtime but provide IDE hints and documentation
Tensor_FP16 = Annotated[torch.Tensor, "precision=fp16"]
Tensor_FP32 = Annotated[torch.Tensor, "precision=fp32"]
Tensor_FP64 = Annotated[torch.Tensor, "precision=fp64"]


class PrecisionAwareCaster:
    """Explicit precision caster with logging for numerical stability."""

    def __init__(self, model_name: str = "model"):
        """Initialize caster with model name for logging context.

        Args:
            model_name: Name of model for logging (e.g., "vae_encoder")
        """
        self.model_name = model_name

    def cast_to_fp32(
        self,
        tensor: torch.Tensor,
        reason: str = "numerical_stability",
        log_stats: bool = False,
    ) -> Tensor_FP32:
        """Cast tensor to FP32 with explicit logging.

        Phase 4b-1: Explicit casts prevent silent precision errors that
        accumulate over 50+ epochs in VAE training.

        Args:
            tensor: Input tensor (any precision)
            reason: Human-readable reason for cast (logged)
            log_stats: Whether to log tensor statistics before cast

        Returns:
            FP32 tensor with same values

        Example:
            mu_fp32 = caster.cast_to_fp32(mu, reason="kl_divergence_computation")
        """
        # The stat f-strings eagerly call ``tensor.min()/.max()/.mean()`` which
        # ``.item()``-sync the GPU; gate on the effective log level so a
        # disabled DEBUG logger pays nothing (this runs per training step for
        # VAE/VQVAE).
        if log_stats and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[{self.model_name}] Pre-FP32 cast stats: "
                f"min={tensor.min():.6f}, max={tensor.max():.6f}, "
                f"mean={tensor.mean():.6f}, dtype={tensor.dtype}"
            )

        logger.debug(f"[{self.model_name}] Casting {tensor.dtype} → FP32 ({reason})")
        result = tensor.float()

        if log_stats and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[{self.model_name}] Post-FP32 cast stats: "
                f"min={result.min():.6f}, max={result.max():.6f}, "
                f"mean={result.mean():.6f}"
            )

        return result

    def cast_to_fp16(
        self,
        tensor: torch.Tensor,
        reason: str = "memory_efficiency",
        log_stats: bool = False,
    ) -> Tensor_FP16:
        """Cast tensor to FP16 with explicit logging.

        Args:
            tensor: Input tensor (any precision)
            reason: Human-readable reason for cast (logged)
            log_stats: Whether to log tensor statistics before cast

        Returns:
            FP16 tensor with same values
        """
        if log_stats:
            logger.debug(
                f"[{self.model_name}] Pre-FP16 cast stats: "
                f"min={tensor.min():.6f}, max={tensor.max():.6f}, "
                f"mean={tensor.mean():.6f}, dtype={tensor.dtype}"
            )

        logger.debug(f"[{self.model_name}] Casting {tensor.dtype} → FP16 ({reason})")
        result = tensor.half()

        if log_stats:
            logger.debug(
                f"[{self.model_name}] Post-FP16 cast stats: "
                f"min={result.min():.6f}, max={result.max():.6f}, "
                f"mean={result.mean():.6f}"
            )

        return result

    def validate_precision(
        self,
        tensor: torch.Tensor,
        expected_dtype: torch.dtype,
        operation_name: str,
    ) -> None:
        """Validate tensor precision matches expectation.

        Raises ValueError if mismatch detected (fail-fast).

        Args:
            tensor: Tensor to validate
            expected_dtype: Expected data type
            operation_name: Name of operation for error message

        Raises:
            ValueError: If precision mismatch detected
        """
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"[{self.model_name}] Precision mismatch in {operation_name}: "
                f"expected {expected_dtype}, got {tensor.dtype}"
            )


class VAEPrecisionManager:
    """Manages precision for VAE training (encoder/decoder/KL computation).

    Typical VAE precision flow:
    1. Input: FP32
    2. Encoder output: FP16 (if using AMP)
    3. Latent (mu, logvar): FP16 → FP32 for KL (numerically stable)
    4. Decoder: FP16
    5. Loss computation: FP16 reconstruction + FP32 KL
    6. Backprop: FP32 total loss through FP16 activations (AMP handles this)
    """

    def __init__(self, device: torch.device):
        """Initialize VAE precision manager.

        Args:
            device: Training device (cuda or cpu)
        """
        self.device = device
        self.caster_encoder = PrecisionAwareCaster("vae_encoder")
        self.caster_decoder = PrecisionAwareCaster("vae_decoder")
        self.caster_latent = PrecisionAwareCaster("vae_latent")

    def encode_with_precision(
        self,
        x: Tensor_FP32,
        encoder: torch.nn.Module,
    ) -> tuple[Tensor_FP16, Tensor_FP16]:
        """Encode input through encoder, maintaining precision tracking.

        Args:
            x: FP32 input
            encoder: Encoder module (output will be FP16 if AMP enabled)

        Returns:
            (mu, logvar) both in FP16 from encoder
        """
        # Encoder output will be FP16 if using autocast in training
        mu, logvar = encoder(x)
        return mu, logvar

    def prepare_latent_for_kl_computation(
        self,
        mu: Tensor_FP16,
        logvar: Tensor_FP16,
    ) -> tuple[Tensor_FP32, Tensor_FP32]:
        """Cast latent parameters to FP32 for numerically stable KL computation.

        KL divergence involves log() operations which are sensitive to
        precision loss. Computing in FP32 prevents 2-5% convergence degradation
        observed in FP16-only computations.

        Args:
            mu: FP16 mean from encoder
            logvar: FP16 log-variance from encoder

        Returns:
            (mu_fp32, logvar_fp32) ready for KL computation
        """
        mu_fp32 = self.caster_latent.cast_to_fp32(
            mu,
            reason="kl_divergence_numerical_stability",
            log_stats=False,
        )
        logvar_fp32 = self.caster_latent.cast_to_fp32(
            logvar,
            reason="kl_divergence_numerical_stability",
            log_stats=False,
        )
        return mu_fp32, logvar_fp32

    def compute_kl_divergence(
        self,
        mu: Tensor_FP32,
        logvar: Tensor_FP32,
    ) -> Tensor_FP32:
        """Compute KL divergence in FP32 for numerical stability.

        Args:
            mu: FP32 mean
            logvar: FP32 log-variance

        Returns:
            KL divergence loss in FP32
        """
        # Standard VAE KL: KL(N(μ, σ²) || N(0, I))
        # = 0.5 * Σ(1 + log(σ²) - μ² - σ²)
        # Sensitive to log() precision, must be FP32
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl


class VQVAEPrecisionManager:
    """Manages precision for VQ-VAE training (encoder/codebook/decoder).

    VQ-VAE precision challenges:
    1. Codebook: Usually FP32 (requires stable discrete assignments)
    2. Encoder/Decoder: Can be FP16 for efficiency
    3. VQ loss: Must be FP32 for stable gradient flow
    """

    def __init__(self, device: torch.device):
        """Initialize VQ-VAE precision manager.

        Args:
            device: Training device
        """
        self.device = device
        self.caster_encoder = PrecisionAwareCaster("vqvae_encoder")
        self.caster_codebook = PrecisionAwareCaster("vqvae_codebook")
        self.caster_vq_loss = PrecisionAwareCaster("vqvae_vq_loss")

    def validate_vq_loss_precision(
        self,
        vq_loss: torch.Tensor,
    ) -> Tensor_FP32:
        """Ensure VQ loss is in FP32 for gradient stability.

        VQ loss computation involves discrete selections that can become
        unstable in FP16. Enforcing FP32 prevents training collapse.

        Args:
            vq_loss: VQ loss from codebook

        Returns:
            VQ loss as FP32 tensor
        """
        if vq_loss.dtype != torch.float32:
            logger.warning(f"VQ loss is {vq_loss.dtype}, casting to FP32 for stability")
            vq_loss = self.caster_vq_loss.cast_to_fp32(
                vq_loss,
                reason="vq_loss_gradient_stability",
            )

        return vq_loss

    def validate_codebook_precision(
        self,
        codebook: torch.nn.Parameter,
    ) -> None:
        """Validate codebook is FP32 (required for stable discrete assignments).

        Raises:
            ValueError: If codebook is not FP32
        """
        if codebook.dtype != torch.float32:
            raise ValueError(
                f"Codebook must be FP32 for discrete assignments, got {codebook.dtype}"
            )
