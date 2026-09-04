"""Training utilities package.

This package contains utility modules for training strategies,
including k-space operations, transformations, and context management.
"""

from .diffusion_mixin import DiffusionStrategyMixin
from .domain_inference import infer_output_domain, needs_ifft_for_visualization
from .kspace_masks import KSpaceMaskGenerator
from .strategy_context import StrategyContext, create_strategy_context
from .transform_ops import ComplexTensorHandler, FFTTransformer

# Import commonly used functions and classes from the main utils module
try:
    from .training_utils import (
        LimitedDataLoader,
        ProfilingState,
        SafeGradScaler,
        call_generator_model,
        clamp_to_range,
        feature_matching_loss_fn,
        handle_training_error,
        initialize_device,
        initialize_profiler,
    )

    _utils_available = True
except ImportError:
    _utils_available = False
    # Create dummy classes/functions for fallback
    SafeGradScaler = None  # type: ignore
    call_generator_model = None  # type: ignore
    clamp_to_range = None  # type: ignore
    LimitedDataLoader = None  # type: ignore
    ProfilingState = None  # type: ignore
    initialize_profiler = None  # type: ignore
    initialize_device = None  # type: ignore
    feature_matching_loss_fn = None  # type: ignore
    handle_training_error = None  # type: ignore

__all__ = [
    "ComplexTensorHandler",
    "DiffusionStrategyMixin",
    "FFTTransformer",
    "KSpaceMaskGenerator",
    "StrategyContext",
    "create_strategy_context",
    "infer_output_domain",
    "needs_ifft_for_visualization",
]

if _utils_available:
    __all__.extend(
        [
            "LimitedDataLoader",
            "ProfilingState",
            "SafeGradScaler",
            "call_generator_model",
            "clamp_to_range",
            "feature_matching_loss_fn",
            "handle_training_error",
            "initialize_device",
            "initialize_profiler",
        ],
    )
