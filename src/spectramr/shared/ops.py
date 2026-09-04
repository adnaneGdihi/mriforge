"""Gradient Checkpointing Configuration System

This module provides configurable gradient checkpointing with
per-module class filters, allowing fine-grained control over
which parts of the model use gradient checkpointing for memory
efficiency.
"""

import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class GradientCheckpointingConfig:
    """Configuration for gradient checkpointing behavior"""

    # Enable/disable gradient checkpointing globally
    enabled: bool = False

    # Per-module class filters - modules to apply checkpointing to
    checkpoint_modules: set[str] = field(
        default_factory=lambda: {
            "TransformerBlock",
            "AttentionBlock",
            "ResBlock",
            "UNetBlock",
        },
    )

    # Modules to exclude from checkpointing
    exclude_modules: set[str] = field(default_factory=set)

    # Use torch.utils.checkpoint instead of model-specific methods
    use_torch_checkpoint: bool = True

    # Checkpointing strategy
    strategy: str = "selective"  # "selective", "all", "none"

    # Memory threshold for automatic enabling (GB)
    memory_threshold_gb: float = 8.0

    # Recompute strategy for torch checkpoint
    use_reentrant: bool = True

    # Custom checkpoint function
    custom_checkpoint_fn: Callable | None = None


class GradientCheckpointingManager:
    """Manager for configurable gradient checkpointing with per-module filters.

    This class provides:
    - Per-module class filtering for selective checkpointing
    - Memory-aware automatic enabling
    - Integration with existing model architectures
    - Context manager for temporary checkpointing control
    """

    def __init__(self, config: GradientCheckpointingConfig | None = None):
        """__init__.

        Args:
            config (Optional[GradientCheckpointingConfig]): Description.
        """
        self.config = config or GradientCheckpointingConfig()
        self._original_forwards: dict[nn.Module, Callable] = {}
        self._checkpointed_modules: set[nn.Module] = set()

    def should_checkpoint_module(self, module: nn.Module) -> bool:
        """Determine if a module should use gradient checkpointing
        based on filters.

        Args:
            module: The module to check

        Returns:
            True if the module should use gradient checkpointing

        """
        if not self.config.enabled:
            return False

        module_class = module.__class__.__name__

        # Check exclusions first
        if module_class in self.config.exclude_modules:
            return False

        # Check inclusions
        if self.config.strategy == "all":
            return True
        if self.config.strategy == "none":
            return False
        if self.config.strategy == "selective":
            return module_class in self.config.checkpoint_modules

        return False

    def apply_checkpointing_to_module(self, module: nn.Module) -> None:
        """Apply gradient checkpointing to a specific module.

        Args:
            module: The module to apply checkpointing to

        """
        if not self.should_checkpoint_module(module):
            return

        # Store original forward method
        if module not in self._original_forwards:
            self._original_forwards[module] = module.forward

        # Apply checkpointing based on configuration
        if self.config.use_torch_checkpoint:
            self._apply_torch_checkpoint(module)
        else:
            self._apply_model_checkpoint(module)

        self._checkpointed_modules.add(module)
        logger.debug(
            f"Applied gradient checkpointing to {module.__class__.__name__}",
        )

    def _apply_torch_checkpoint(self, module: nn.Module) -> None:
        """Apply torch.utils.checkpoint to the module"""
        original_forward = module.forward

        def checkpointed_forward(*args, **kwargs):
            # Use custom checkpoint function if provided
            """checkpointed_forward.

            Returns:
                Any: Description.
            """
            if self.config.custom_checkpoint_fn:
                return self.config.custom_checkpoint_fn(
                    original_forward,
                    *args,
                    **kwargs,
                )

            # Use torch checkpoint with reentrant option
            return torch.utils.checkpoint.checkpoint(
                original_forward,
                *args,
                use_reentrant=self.config.use_reentrant,
                **kwargs,
            )

        module.forward = checkpointed_forward

    def _apply_model_checkpoint(self, module: nn.Module) -> None:
        """Apply model-specific gradient checkpointing methods"""
        # Try model-specific methods first
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
        elif hasattr(module, "set_grad_checkpointing"):
            module.set_grad_checkpointing(True)
        else:
            logger.warning(
                f"Module {module.__class__.__name__} does not support "
                "model-specific gradient checkpointing, "
                "falling back to torch checkpoint",
            )
            self._apply_torch_checkpoint(module)

    def remove_checkpointing_from_module(self, module: nn.Module) -> None:
        """Remove gradient checkpointing from a specific module.

        Args:
            module: The module to remove checkpointing from

        """
        if module in self._original_forwards:
            module.forward = self._original_forwards[module]
            del self._original_forwards[module]
            self._checkpointed_modules.discard(module)
        logger.debug(
            f"Removed gradient checkpointing from {module.__class__.__name__}",
        )

    def apply_to_model(self, model: nn.Module) -> None:
        """Apply gradient checkpointing to all applicable modules in a model.

        Args:
            model: The model to apply checkpointing to

        """
        # Check memory usage for automatic enabling
        should_enable = self.config.enabled or self._should_enable_automatically()

        if not should_enable:
            logger.info("Gradient checkpointing is disabled")
            return

        # If auto-enabling due to memory, update the config
        if self._should_enable_automatically() and not self.config.enabled:
            logger.info(
                "Automatically enabling gradient checkpointing due to high memory usage",
            )
            # Create a new config with enabled=True
            updated_config = GradientCheckpointingConfig(**self.config.__dict__)
            updated_config.enabled = True
            self.config = updated_config

        checkpointed_count = 0
        for module in model.modules():
            if module != model:  # Skip root module
                self.apply_checkpointing_to_module(module)
                if module in self._checkpointed_modules:
                    checkpointed_count += 1

        logger.info(
            f"Applied gradient checkpointing to {checkpointed_count} modules",
        )

    def remove_from_model(self, model: nn.Module) -> None:
        """Remove gradient checkpointing from all modules in a model.

        Args:
            model: The model to remove checkpointing from

        """
        for module in list(self._checkpointed_modules):
            self.remove_checkpointing_from_module(module)

        logger.info("Removed gradient checkpointing from all modules")

    def _should_enable_automatically(self) -> bool:
        """Check if gradient checkpointing should be enabled
        automatically based on memory
        """
        if not torch.cuda.is_available():
            return False

        try:
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            return allocated_gb > self.config.memory_threshold_gb
        except (RuntimeError, AttributeError, ValueError):
            return False

    @contextmanager
    def temporary_config(self, **config_updates):
        """Context manager for temporary configuration changes.

        Args:
            **config_updates: Configuration updates to apply temporarily

        """
        original_config = self.config
        temp_config = GradientCheckpointingConfig(**original_config.__dict__)
        temp_config.__dict__.update(config_updates)
        self.config = temp_config

        try:
            yield
        finally:
            self.config = original_config

    def get_checkpointed_modules_info(self) -> dict[str, Any]:
        """Get information about currently checkpointed modules"""
        return {
            "enabled": self.config.enabled,
            "checkpointed_count": len(self._checkpointed_modules),
            "checkpointed_modules": [m.__class__.__name__ for m in self._checkpointed_modules],
            "strategy": self.config.strategy,
            "memory_threshold_gb": self.config.memory_threshold_gb,
        }


# Global instance
_checkpointing_manager = None


def get_checkpointing_manager() -> GradientCheckpointingManager:
    """Get or create global gradient checkpointing manager instance"""
    global _checkpointing_manager
    if _checkpointing_manager is None:
        _checkpointing_manager = GradientCheckpointingManager()
    return _checkpointing_manager


def configure_gradient_checkpointing(
    enabled: bool = True,
    checkpoint_modules: list[str] | None = None,
    exclude_modules: list[str] | None = None,
    strategy: str = "selective",
    memory_threshold_gb: float = 8.0,
) -> GradientCheckpointingManager:
    """Configure gradient checkpointing with the specified parameters.

    Args:
        enabled: Whether to enable gradient checkpointing
        checkpoint_modules: list of module class names to checkpoint
        exclude_modules: list of module class names to exclude
        strategy: Checkpointing strategy ("selective", "all", "none")
        memory_threshold_gb: Memory threshold for automatic enabling

    Returns:
        Configured GradientCheckpointingManager instance

    """
    config = GradientCheckpointingConfig(
        enabled=enabled,
        checkpoint_modules=set(checkpoint_modules or []),
        exclude_modules=set(exclude_modules or []),
        strategy=strategy,
        memory_threshold_gb=memory_threshold_gb,
    )

    manager = get_checkpointing_manager()
    manager.config = config
    return manager


@contextmanager
def gradient_checkpointing_context(
    enabled: bool = True,
    checkpoint_modules: list[str] | None = None,
    **config_kwargs,
):
    """Context manager for temporary gradient checkpointing configuration.

    Args:
        enabled: Whether to enable checkpointing in this context
        checkpoint_modules: Module classes to checkpoint
        **config_kwargs: Additional configuration parameters

    """
    manager = get_checkpointing_manager()
    config_updates = {"enabled": enabled}
    if checkpoint_modules:
        config_updates["checkpoint_modules"] = set(checkpoint_modules)
    config_updates.update(config_kwargs)

    with manager.temporary_config(**config_updates):
        yield


def apply_gradient_checkpointing_to_model(
    model: nn.Module,
    enabled: bool = True,
    checkpoint_modules: list[str] | None = None,
    **config_kwargs,
) -> nn.Module:
    """Convenience function to apply gradient checkpointing to a model.

    Args:
        model: The model to apply checkpointing to
        enabled: Whether to enable checkpointing
        checkpoint_modules: Module classes to checkpoint
        **config_kwargs: Additional configuration parameters

    Returns:
        The model with checkpointing applied

    """
    manager = configure_gradient_checkpointing(
        enabled=enabled,
        checkpoint_modules=checkpoint_modules,
        **config_kwargs,
    )
    manager.apply_to_model(model)
    return model


if __name__ == "__main__":
    # Example usage
    logger.debug("Testing Gradient Checkpointing Configuration System...")

    # Create a simple test model
    class TestBlock(nn.Module):
        """TestBlock class."""

        def __init__(self):
            """__init__."""
            super().__init__()
            self.conv = nn.Conv2d(64, 64, 3, padding=1)

        def forward(self, x):
            """forward.

                Args:
                    x (Any): Description.
                Returns:
                    Any: Description.

            forward method for TestBlock.

            Executes PyTorch tensor operations.

            Args:
                x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

            Returns:
                torch.Tensor: Output tensor with shape matching the operation.

            Hardware/Device Context:
                Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
            return self.conv(x)

    model = nn.Sequential(TestBlock(), TestBlock())

    # Configure and apply checkpointing
    manager = configure_gradient_checkpointing(
        enabled=True,
        checkpoint_modules=["TestBlock"],
        strategy="selective",
    )

    logger.debug("Before checkpointing: %s", manager.get_checkpointed_modules_info())
    manager.apply_to_model(model)
    logger.debug("After checkpointing: %s", manager.get_checkpointed_modules_info())

    # Test context manager
    with gradient_checkpointing_context(enabled=False):
        logger.debug("In context (disabled): %s", manager.get_checkpointed_modules_info())

    logger.debug("After context: %s", manager.get_checkpointed_modules_info())

    logger.debug("Gradient checkpointing test complete!")
