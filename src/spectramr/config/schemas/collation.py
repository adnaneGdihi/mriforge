"""Collation configuration schema.

This module defines the configuration for batch collation strategy selection and parameters.
Collation strategies control how individual samples are assembled into batches during data loading.

Author: spectraMR Team
Date: 2026-02-15
"""

import typing
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["CollationConfigSchema"]


class CollationConfigSchema(BaseModel):
    """Configuration for batch collation strategy selection and parameters.

    This schema controls how data samples are collated (assembled) into batches
    during training, validation, and testing. Different strategies handle different
    data formats (images, k-space, graphs, sequences, etc.) and provide different
    padding/stacking behaviors.

    Available Strategies:
        - 'image': General-purpose image tensor collation with padding/stacking
        - 'slab': 2.5D slab mode (flattens depth to channels)
        - 'physics': Physics-aware collation for non-Cartesian k-space
        - 'robust': Fallback strategy with None filtering
        - 'graph': PyTorch Geometric graph batching
        - 'sequence': Variable-length sequence padding
        - 'mixed': Multi-modal data collation

    Example YAML:
        ```yaml
        data:
          collation:
            strategy: "image"           # Explicit strategy choice
            padding_mode: "zeros"       # Pad with zeros
            padding_value: 0.0          # Value for padding
            allow_variable_shapes: false
            squeeze_depth_dim: null     # Auto-derive from patch_size
            validate_nans: true         # Safety check
            log_strategy_selection: true
        ```

    Attributes:
        strategy: Collation strategy name. If None, auto-selected from dataset_type.
        padding_mode: Padding mode for ImageCollateStrategy ('zeros', 'reflect', 'replicate').
        padding_value: Value to use for padding (default 0.0).
        allow_variable_shapes: Allow variable tensor shapes in batch (no padding to max).
        squeeze_depth_dim: Squeeze depth dimension if D=1. If None, auto-derived from patch_size.
        validate_nans: Validate and replace NaN values in collated tensors (default True).
        log_strategy_selection: Log which strategy was selected and parameters (default True).
    """

    model_config = ConfigDict(
        frozen=True,  # Immutability (config cannot be modified after creation)
        extra="forbid",  # Strict: reject unknown fields
        protected_namespaces=(),  # Allow "model_" field names
    )

    # ========== Strategy Selection ==========
    strategy: (
        Literal[
            "image",
            "slab",
            "graph",
            "sequence",
            "mixed",
            "robust",
            "physics",
            # "universal" removed 2026-08-09. CollateStrategyFactory has no such
            # class and COMPATIBILITY_MAP dropped it (audit A8), but this Literal
            # -- the vocabulary a YAML is actually validated against -- kept
            # advertising it. Zero arms declared it, so it was latent; the
            # three-way guard in strategy_selector now keeps the schema, the
            # compatibility map and the factory in lockstep.
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Collation strategy name. Options: 'image' (general tensor collation), "
            "'slab' (2.5D mode), 'physics' (non-Cartesian), "
            "'robust' (fallback), 'graph' (PyG), 'sequence' (variable-length), "
            "'mixed' (multi-modal). If None, auto-selected based on dataset_type and enable_slab_mode."
        ),
    )

    # ========== ImageCollateStrategy Parameters ==========
    # These apply when strategy='image' or strategy='slab' (which extends image)

    padding_mode: Literal["zeros", "reflect", "replicate"] = Field(
        default="zeros",
        description=(
            "Padding mode for ImageCollateStrategy: "
            "'zeros' (pad with constant value), "
            "'reflect' (reflect edges), "
            "'replicate' (repeat edge values). "
            "Only used when strategy='image' or 'slab'."
        ),
    )

    slab_target_mode: Literal["middle", "flatten", "keep"] = Field(
        default="middle",
        description=(
            "For strategy='slab': what the TARGET is. 'middle' keeps the centre "
            "depth slice (predict the centre from a slab of context), 'flatten' "
            "folds depth into channels so every slice is supervised, 'keep' "
            "leaves the 5-D target untouched for a volumetric decoder. "
            "CollationStrategySelector hardcoded 'middle' with no config route, "
            "so an arm could not choose among the three the strategy already "
            "implemented (audit B6). The default is unchanged, so no existing "
            "run moves."
        ),
    )

    padding_value: float = Field(
        default=0.0,
        description=(
            "Value for constant padding when padding_mode='zeros'. "
            "Default 0.0. Only used when strategy='image' or 'slab'."
        ),
    )

    allow_variable_shapes: bool = Field(
        default=False,
        description=(
            "Allow variable tensor shapes in batch (no padding to max shape). "
            "If False, all tensors padded to max shape across batch. "
            "If True, batch may contain tensors of different sizes. "
            "Only used when strategy='image' or 'slab'."
        ),
    )

    squeeze_depth_dim: bool | None = Field(
        default=None,
        description=(
            "Squeeze depth dimension if D=1 (convert 5D to 4D for 2D models). "
            "TorchIO produces 5D tensors (B, C, H, W, D) even for 2D data (D=1). "
            "If True, squeeze to (B, C, H, W). "
            "If False, keep 5D. "
            "If None, auto-derived from patch_size: True if patch_size[-1]==1. "
            "Only used when strategy='image' or 'slab'."
        ),
    )

    # ========== Validation & Safety ==========

    validate_nans: bool = Field(
        default=True,
        description=(
            "Whether to validate for NaN values in collated tensors and replace with padding_value. "
            "If True, NaN values are detected and replaced (safe). "
            "If False, NaN values pass through (may cause training failure). "
            "Recommended: True for safety."
        ),
    )

    log_strategy_selection: bool = Field(
        default=True,
        description=(
            "Whether to log which collation strategy was selected and its parameters. "
            "If True, INFO-level logs show strategy name, selection reason, and kwargs. "
            "If False, selection happens silently. "
            "Recommended: True for transparency and debugging."
        ),
    )

    # ========== Validators ==========

    @field_validator("strategy")
    @classmethod
    def validate_strategy_name(cls, v: str | None) -> str | None:
        """Validate collation strategy name.

        Args:
            v: Strategy name or None

        Returns:
            Validated strategy name

        Raises:
            ValueError: If strategy name is invalid
        """
        if v is None:
            # None is valid (triggers auto-selection)
            return v

        # DERIVED from the field's own Literal, not restated. This was a
        # hand-kept set duplicating the annotation eight strings above it, and
        # the two had already drifted apart from the factory -- three copies of
        # one vocabulary is how "universal" stayed advertised after it stopped
        # being buildable. A validator that re-lists its field's options can
        # only ever agree with them by luck.
        valid_strategies = {
            a
            for a in typing.get_args(
                typing.get_args(CollationConfigSchema.model_fields["strategy"].annotation)[0]
            )
            if isinstance(a, str)
        }

        if v not in valid_strategies:
            raise ValueError(
                f"Invalid collation strategy: '{v}'. Valid options: {sorted(valid_strategies)}"
            )

        return v

    @field_validator("padding_mode")
    @classmethod
    def validate_padding_mode(cls, v: str) -> str:
        """Validate padding mode.

        Args:
            v: Padding mode string

        Returns:
            Validated padding mode

        Raises:
            ValueError: If padding mode is invalid
        """
        valid_modes = {"zeros", "reflect", "replicate"}

        if v not in valid_modes:
            raise ValueError(f"Invalid padding_mode: '{v}'. Valid options: {sorted(valid_modes)}")

        return v

    @field_validator("padding_value")
    @classmethod
    def validate_padding_value(cls, v: float) -> float:
        """Validate padding value is finite.

        Args:
            v: Padding value

        Returns:
            Validated padding value

        Raises:
            ValueError: If padding value is NaN or Inf
        """
        import math

        if not math.isfinite(v):
            raise ValueError(
                f"padding_value must be finite, got {v}. NaN or Inf padding values are not allowed."
            )

        return v
