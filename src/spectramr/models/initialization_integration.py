"""Integration module for weight initialization with model factory.

This module bridges configuration-based weight initialization with model creation,
removing any model-specific initialization methods that conflict.
"""

import logging
from typing import TYPE_CHECKING

from torch import nn

from spectramr.models.utilities.weight_initialization import apply_weight_initialization

if TYPE_CHECKING:
    from spectramr.config.schemas.model import ModelConfigSchema

logger = logging.getLogger(__name__)


def remove_builtin_initialization(model: nn.Module) -> None:
    """Remove built-in initialization methods from models to avoid conflicts.

    This disables model-specific _init_weights and initialize_weights methods
    that would override centralized initialization.

    Args:
        model: PyTorch model to process
    """
    # Methods to remove/disable
    init_methods = ["_init_weights", "initialize_weights", "init_weights"]

    for method_name in init_methods:
        if hasattr(model, method_name):
            # Replace with no-op function
            setattr(model, method_name, lambda self=model: None)
            logger.debug(f"Disabled {method_name} on {model.__class__.__name__}")


def apply_initialization_from_config(
    model: nn.Module,
    model_config: "ModelConfigSchema",
) -> None:
    """Apply centralized weight initialization based on configuration.

    Args:
        model: PyTorch model to initialize
        model_config: Model configuration schema with initialization parameters
    """
    if not model_config.apply_weight_init:
        logger.debug("Weight initialization disabled by config")
        return

    try:
        # Determine model type for specialized initialization
        model_type = model_config.model_type.lower()

        # Apply centralized initialization
        apply_weight_initialization(
            model=model,
            init_type=model_config.weight_init_type,
            model_type=model_type,
            gain=model_config.weight_init_gain,
            activation=model_config.weight_init_activation,
        )

        logger.info(
            f"Applied weight initialization: type={model_config.weight_init_type}, "
            f"model_type={model_type}, gain={model_config.weight_init_gain}"
        )
    except Exception as e:
        logger.warning(
            f"Failed to apply weight initialization: {e}. Model will use default initialization."
        )


def initialize_model_weights(
    model: nn.Module,
    model_config: "ModelConfigSchema",
    disable_builtin: bool = True,
) -> None:
    """Initialize model weights with centralized configuration.

    This is the primary entry point for model initialization integration.

    Args:
        model: PyTorch model to initialize
        model_config: Model configuration with initialization parameters
        disable_builtin: Whether to disable model-specific initialization methods
    """
    if disable_builtin:
        remove_builtin_initialization(model)

    apply_initialization_from_config(model, model_config)


__all__ = [
    "apply_initialization_from_config",
    "initialize_model_weights",
    "remove_builtin_initialization",
]
