#!/usr/bin/env python3
"""Embedding Dimension Validator

This module validates embedding dimensions in transformer models to ensure
they meet mathematical constraints (e.g., embed_dim % num_heads == 0).

Key validations:
- embed_dim must be divisible by num_heads in MultiHeadAttention layers
- Default num_heads is 12, so embed_dim should be divisible by 12
- Validates both explicit configs and defaults
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingDimensionValidator:
    """Validates embedding dimensions in neural network configs."""

    # Default num_heads for various model types
    DEFAULT_HEADS: dict[str, int] = {
        "transformer": 12,
        "vit": 12,
        "swin": 8,
        "vit_kan": 12,
        "swin_transformer_kan": 8,
        "kan": 8,
        "spectral_graph": 4,
        "multihead": 12,  # Generic MultiHeadAttention default
    }

    # Model types that require embedding dimension validation
    TRANSFORMER_MODELS = {
        "transformer",
        "vit",
        "swin",
        "vit_kan",
        "swin_transformer_kan",
        "kan",
        "spectral_graph",
        "recon_former",
        "linformer",
        "sparse_transformer",
        "cross_attention_fusion",
        "contrastive_learning",
        "vit_adapter",
    }

    @staticmethod
    def validate_model_config(
        model_type: str,
        model_kwargs: dict[str, Any],
        raise_on_error: bool = True,
    ) -> tuple[bool, str | None]:
        """
        Validate embedding dimensions in model config.

        Args:
            model_type: Type of model (e.g., "vit", "transformer")
            model_kwargs: Model kwargs dict containing embed_dim, num_heads, etc.
            raise_on_error: If True, raise ValueError on validation failure

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passes
            - error_message: None if valid, error string if invalid
        """
        # Skip validation for non-transformer models
        if model_type not in EmbeddingDimensionValidator.TRANSFORMER_MODELS:
            return True, None

        # Extract embedding dimension
        embed_dim = model_kwargs.get("embed_dim")
        if embed_dim is None:
            return True, None  # No embedding dimension specified

        if not isinstance(embed_dim, int) or embed_dim <= 0:
            error = f"Invalid embed_dim: {embed_dim} (must be positive integer)"
            if raise_on_error:
                raise ValueError(error)
            return False, error

        # Determine num_heads to validate against
        num_heads = model_kwargs.get("num_heads")
        if num_heads is None:
            # Use default for model type
            num_heads = EmbeddingDimensionValidator.DEFAULT_HEADS.get(model_type, 12)
        elif isinstance(num_heads, list):
            # For hierarchical models (e.g., Swin), validate each head count
            errors = []
            for i, heads in enumerate(num_heads):
                if embed_dim % heads != 0:
                    errors.append(
                        f"Layer {i}: embed_dim={embed_dim} not divisible by num_heads={heads}"
                    )
            if errors:
                error = "; ".join(errors)
                if raise_on_error:
                    raise ValueError(error)
                return False, error
            return True, None

        if not isinstance(num_heads, int) or num_heads <= 0:
            error = f"Invalid num_heads: {num_heads} (must be positive integer)"
            if raise_on_error:
                raise ValueError(error)
            return False, error

        # Validate divisibility
        if embed_dim % num_heads != 0:
            error = (
                f"Model '{model_type}': embed_dim ({embed_dim}) is not divisible "
                f"by num_heads ({num_heads}). This will cause an assertion error "
                f"in MultiHeadSelfAttention. Suggested fixes: "
                f"1) Change embed_dim to a multiple of {num_heads} "
                f"(e.g., {num_heads * (embed_dim // num_heads + 1)}), "
                f"2) Change num_heads to a divisor of {embed_dim} "
                f"(e.g., {EmbeddingDimensionValidator._find_divisor(embed_dim)})"
            )
            if raise_on_error:
                raise ValueError(error)
            return False, error

        return True, None

    @staticmethod
    def validate_training_config(
        config: dict[str, Any],
        raise_on_error: bool = True,
    ) -> tuple[bool, str | None]:
        """
        Validate embedding dimensions in complete training config.

        Args:
            config: TrainingSettings dict or similar config object
            raise_on_error: If True, raise ValueError on validation failure

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Extract model config
        model_config = config.get("model", {})
        if not isinstance(model_config, dict):
            # Handle Pydantic model
            try:
                model_config = model_config.model_dump()
            except AttributeError:
                model_config = {}

        model_type = model_config.get("model_type")
        model_kwargs = model_config.get("model_kwargs", {})

        if not model_type:
            return True, None

        return EmbeddingDimensionValidator.validate_model_config(
            model_type, model_kwargs, raise_on_error
        )

    @staticmethod
    def _find_divisor(n: int, preferred_range: tuple[int, int] = (4, 16)) -> int:
        """Find a good divisor for n in the preferred range."""
        for d in range(preferred_range[1], preferred_range[0] - 1, -1):
            if n % d == 0:
                return d
        return 1

    @staticmethod
    def suggest_fixes(embed_dim: int, num_heads: int) -> str:
        """Generate suggestions for fixing embedding dimension mismatch."""
        suggestions = []

        # Suggestion 1: Adjust embed_dim
        next_multiple = num_heads * ((embed_dim // num_heads) + 1)
        suggestions.append(f"Set embed_dim={next_multiple} (multiple of {num_heads})")

        # Suggestion 2: Adjust num_heads
        good_divisors = []
        for d in range(1, min(17, embed_dim + 1)):
            if embed_dim % d == 0 and 4 <= d <= 16:
                good_divisors.append(d)

        if good_divisors:
            suggestions.append(f"Set num_heads to one of: {good_divisors}")

        # Suggestion 3: Round embed_dim
        for target in [64, 128, 256, 512, 768, 1024]:
            if abs(target - embed_dim) < embed_dim * 0.2:  # Within 20%
                if target % num_heads == 0:
                    suggestions.append(
                        f"Set embed_dim={target} (close to current, divisible by {num_heads})"
                    )
                    break

        return "\n".join(f"  - {s}" for s in suggestions)
