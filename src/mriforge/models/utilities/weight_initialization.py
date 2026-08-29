#!/usr/bin/env python
"""Weight Initialization Manager
=============================

This module provides a centralized manager for applying various weight
initialization schemes to PyTorch models. It supports specialized initialization
for different model architectures like standard CNNs, Transformers, KANs, and
Quantizers with SSOT (Single Source of Truth) registry.
"""

import logging

import torch
from torch import nn
from torch.nn import init

logger = logging.getLogger(__name__)


# Utility function for trunc_normal_ initialization (fallback implementation)
def trunc_normal_(
    tensor: "torch.Tensor",
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> "torch.Tensor":
    """Truncated normal initialization from timm (fallback implementation).

    Args:
        tensor: Tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        a: Lower bound of truncation (in standard deviations)
        b: Upper bound of truncation (in standard deviations)
    """
    import math

    import torch

    def norm_cdf(x):
        """norm_cdf.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf(a)
        u = norm_cdf(b)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

    return tensor


class _WeightInitializationManager:
    """Internal class to manage and apply different weight initialization schemes.
    This is not intended to be instantiated directly. Use the
    `apply_weight_initialization` function instead.

    Registry-based approach (SSOT):
    - Maps model types and layer types to appropriate initialization strategies
    - Supports: CNNs (kaiming), Transformers (trunc_normal), Quantizers (uniform), KANs (custom)
    - All initialization happens through centralized apply_weight_initialization()
    """

    @staticmethod
    def _initialize_generic(
        module: nn.Module,
        init_type: str,
        gain: float,
        activation: str,
    ):
        """Initializes weights for standard Conv/Linear layers."""
        classname = module.__class__.__name__
        if hasattr(module, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            # Registry pattern for O(1) initialization lookup
            INIT_REGISTRY = {
                "he_normal": lambda w: init.kaiming_normal_(
                    w, a=0, mode="fan_in", nonlinearity=activation
                ),
                "he_uniform": lambda w: init.kaiming_uniform_(
                    w, a=0, mode="fan_in", nonlinearity=activation
                ),
                "xavier_normal": lambda w: init.xavier_normal_(w, gain=gain),
                "xavier_uniform": lambda w: init.xavier_uniform_(w, gain=gain),
                "orthogonal": lambda w: init.orthogonal_(w, gain=gain),
                "normal": lambda w: init.normal_(w, 0, 0.02),
            }

            init_fn = INIT_REGISTRY.get(init_type)
            if init_fn is None:
                logger.warning(
                    f"Unknown generic init type: '{init_type}'. Skipping initialization for {classname}.",
                )
                return

            init_fn(module.weight)

            if module.bias is not None:
                init.constant_(module.bias, 0.0)

        elif classname.find("BatchNorm") != -1:
            if hasattr(module, "weight") and module.weight is not None:
                init.normal_(module.weight, 1.0, 0.02)
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias, 0.0)

    @staticmethod
    def _initialize_transformer(module: nn.Module, std: float = 0.02):
        """Initializes transformer weights with truncated normal for positional embeddings and weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.weight, 1.0)
            init.constant_(module.bias, 0.0)
        # Special handling for embeddings (positional, cls tokens, mask tokens)
        elif isinstance(module, nn.Parameter):
            trunc_normal_(module, std=std)

    @staticmethod
    def _initialize_quantizer(module: nn.Module):
        """Initializes VQ/Quantizer-specific layers with uniform distribution."""
        classname = module.__class__.__name__

        # Quantizer codebook: uniform distribution
        if hasattr(module, "embedding"):
            if hasattr(module.embedding, "weight"):
                init.uniform_(module.embedding.weight, -1.0, 1.0)

        # Quantizer parameters: uniform distribution
        if hasattr(module, "codebook") and module.codebook is not None:
            init.uniform_(module.codebook, -1.0, 1.0)

        # Quantizer grid/splines: uniform distribution
        if hasattr(module, "grid") and module.grid is not None:
            init.uniform_(module.grid, -1.0, 1.0)

        # Fall back to generic for conv/linear in quantizer
        if hasattr(module, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                init.constant_(module.bias, 0.0)

    @staticmethod
    def _initialize_kan(module: nn.Module, std: float = 0.1):
        """Initializes KAN-related weights with moderate std."""
        if hasattr(module, "weight") and module.weight is not None:
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
        elif hasattr(module, "grid") and module.grid is not None:
            init.uniform_(module.grid, -1, 1)
        elif hasattr(module, "spline_weight") and module.spline_weight is not None:
            init.normal_(module.spline_weight, mean=0.0, std=std)

    @classmethod
    def initialize_weights(
        cls,
        model: nn.Module,
        init_type: str,
        gain: float = 1.0,
        activation: str = "relu",
    ):
        """Applies the generic weight initialization to a model."""

        def init_closure(m):
            """init_closure.

            Args:
                m (Any): Description.
            Returns:
                Any: Description.
            """
            cls._initialize_generic(m, init_type, gain, activation)

        model.apply(init_closure)

    @classmethod
    def initialize_transformer_weights(cls, model: nn.Module, std: float = 0.02):
        """Applies transformer-specific weight initialization."""

        def init_closure(m):
            """init_closure.

            Args:
                m (Any): Description.
            Returns:
                Any: Description.
            """
            cls._initialize_transformer(m, std)

        model.apply(init_closure)

    @classmethod
    def initialize_quantizer_weights(cls, model: nn.Module):
        """Applies quantizer-specific weight initialization (uniform for codebooks)."""

        def init_closure(m):
            """init_closure.

            Args:
                m (Any): Description.
            Returns:
                Any: Description.
            """
            cls._initialize_quantizer(m)

        model.apply(init_closure)

    @classmethod
    def initialize_kan_weights(cls, model: nn.Module, std: float = 0.1):
        """Applies KAN-specific weight initialization."""

        def init_closure(m):
            """init_closure.

            Args:
                m (Any): Description.
            Returns:
                Any: Description.
            """
            cls._initialize_kan(m, std)

        model.apply(init_closure)


def apply_weight_initialization(
    model: nn.Module,
    init_type: str = "he_normal",
    model_type: str = "standard",
    gain: float = 1.0,
    activation: str = "relu",
):
    """Apply weight initialization to a model based on its type (SSOT registry).

    This is the PRIMARY entry point for all weight initialization. All embedded
    initialization code in model implementations should be removed and replaced
    with calls to this function.

    Args:
        model: PyTorch model to initialize
        init_type: Initialization strategy (he_normal, xavier_normal, etc.)
        model_type: Model architecture type (determines specialization):
            - "standard", "unet", "gan", "patchgan": use generic (he_normal)
            - "transformer", "vit", "swin", "mae": use trunc_normal
            - "kan", "kanu": use KAN-specific
            - "vae", "vq", "vqgan", "vqvae": use generic + quantizer special handling
            - "diffusion": use transformer initialization
        gain: Gain parameter for Xavier/orthogonal initialization
        activation: Activation function for He initialization (relu, tanh, sigmoid, etc.)

    Returns:
        bool: True if initialization succeeded, False otherwise
    """
    try:
        msg = f"Applying weight initialization for model_type='{model_type}'"
        msg += f" with method='{init_type}', activation='{activation}'..."
        logger.info(msg)
        model_type_lower = model_type.lower()

        # REGISTRY: Map model types to initialization strategies (SSOT)
        if (
            "transformer" in model_type_lower
            or "vit" in model_type_lower
            or "swin" in model_type_lower
            or "mae" in model_type_lower
            or "diffusion" in model_type_lower
        ):
            # Transformer-based: use trunc_normal
            _WeightInitializationManager.initialize_transformer_weights(model)
            msg = "[OK] Applied Transformer-specific (trunc_normal) initialization to "
            msg += f"'{model.__class__.__name__}'."
            logger.info(msg)
        elif "kan" in model_type_lower:
            # KAN-specific initialization
            _WeightInitializationManager.initialize_kan_weights(model)
            msg = "[OK] Applied KAN-specific initialization to "
            msg += f"'{model.__class__.__name__}'."
            logger.info(msg)
        elif "vq" in model_type_lower or "quantizer" in model_type_lower:
            # VQ/Quantizer: use uniform for codebooks, kaiming for convs
            _WeightInitializationManager.initialize_quantizer_weights(model)
            msg = "[OK] Applied Quantizer-specific initialization to "
            msg += f"'{model.__class__.__name__}'."
            logger.info(msg)
        elif "vae" in model_type_lower:
            # VAE: use generic kaiming for encoder/decoder, but also check for quantizers
            def init_closure(m):
                # First try quantizer-specific
                """init_closure.

                Args:
                    m (Any): Description.
                Returns:
                    Any: Description.
                """
                _WeightInitializationManager._initialize_quantizer(m)
                # Then fall back to generic
                _WeightInitializationManager._initialize_generic(m, init_type, gain, activation)

            model.apply(init_closure)
            msg = "[OK] Applied VAE initialization (generic + quantizer handling) to "
            msg += f"'{model.__class__.__name__}'."
            logger.info(msg)
        else:
            # Default: generic initialization (CNNs, GANs, UNets, etc.)
            def init_closure(m):
                """init_closure.

                Args:
                    m (Any): Description.
                Returns:
                    Any: Description.
                """
                _WeightInitializationManager._initialize_generic(m, init_type, gain, activation)

            model.apply(init_closure)
            msg = f"[OK] Applied generic '{init_type}' initialization to "
            msg += f"'{model.__class__.__name__}' with activation='{activation}'."
            logger.info(msg)

        return True

    except Exception as e:
        logger.warning(f"⚠️ Weight initialization failed for {model_type}: {e}")
        return False
