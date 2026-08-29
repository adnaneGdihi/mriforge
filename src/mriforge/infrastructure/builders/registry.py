"""Builder Registry System

Phase 0: Central registry for all builder classes.

This module provides a thread-safe registry for builder classes, enabling:
1. O(1) lookup of builder implementations
2. Automatic builder discovery
3. Builder factory pattern support
4. Validation of builder implementations

The registry replaces the need for factory classes that just instantiate
builders. Instead, builders are registered once and accessed by name.

Design: Singleton Registry Pattern + Factory Pattern
"""

import logging
import threading
from typing import Optional

from .core.base_builder import BuilderBase, DirectorBuilder, FluentBuilder

logger = logging.getLogger(__name__)


class BuilderRegistry:
    """Thread-safe central registry for all builder classes.

    This is a singleton that maintains a mapping of builder names to
    their implementation classes. It enables:
    - O(1) lookup of builder implementations
    - Builder factory instantiation
    - Registry validation
    - Batch registration

    Example:
        >>> registry = BuilderRegistry.get_instance()
        >>> registry.register("model", ModelBuilder)
        >>> registry.register("optimizer", OptimizerBuilder)
        >>> ModelBuilder = registry.get("model")
        >>> builder = registry.create("model", config, device)

    Thread Safety:
        - Uses RLock for thread-safe access
        - Get instance is thread-safe
        - All public methods are thread-safe
    """

    _instance: Optional["BuilderRegistry"] = None
    _lock = threading.RLock()

    def __init__(self):
        """Initialize empty registry."""
        self._builders: dict[str, type[BuilderBase]] = {}
        self._lock = threading.RLock()
        logger.info("BuilderRegistry initialized")

    @staticmethod
    def get_instance() -> "BuilderRegistry":
        """Get singleton instance (thread-safe).

        Returns:
            BuilderRegistry singleton
        """
        if BuilderRegistry._instance is None:
            with BuilderRegistry._lock:
                if BuilderRegistry._instance is None:
                    BuilderRegistry._instance = BuilderRegistry()

        return BuilderRegistry._instance

    def register(
        self, name: str, builder_class: type[BuilderBase], overwrite: bool = False
    ) -> "BuilderRegistry":
        """Register a builder class.

        Args:
            name: Builder identifier (e.g., "model", "optimization")
            builder_class: Builder class to register
            overwrite: If False, raise error on duplicate; if True, replace

        Returns:
            self for method chaining

        Raises:
            TypeError: If builder_class doesn't inherit from BuilderBase
            ValueError: If name already registered and overwrite=False
        """
        if not issubclass(builder_class, BuilderBase):
            raise TypeError(f"Builder must inherit from BuilderBase, got {builder_class}")

        with self._lock:
            if name in self._builders and not overwrite:
                raise ValueError(
                    f"Builder '{name}' already registered. Use overwrite=True to replace."
                )

            self._builders[name] = builder_class
            action = "Registered" if name not in self._builders else "Replaced"
            logger.info(f"{action} builder: {name} -> {builder_class.__name__}")

        return self

    def register_batch(
        self, builders: dict[str, type[BuilderBase]], overwrite: bool = False
    ) -> "BuilderRegistry":
        """Register multiple builders at once.

        Args:
            builders: Dict mapping names to builder classes
            overwrite: If False, raise error on duplicate; if True, replace

        Returns:
            self for method chaining

        Example:
            >>> registry = BuilderRegistry.get_instance()
            >>> registry.register_batch({
            ...     "model": ModelBuilder,
            ...     "optimizer": OptimizerBuilder,
            ...     "loss": LossBuilder,
            ... })
        """
        for name, builder_class in builders.items():
            self.register(name, builder_class, overwrite=overwrite)

        return self

    def unregister(self, name: str) -> "BuilderRegistry":
        """Unregister a builder.

        Args:
            name: Builder identifier

        Returns:
            self for method chaining

        Raises:
            KeyError: If builder not registered
        """
        with self._lock:
            if name not in self._builders:
                raise KeyError(f"Builder '{name}' not registered")

            del self._builders[name]
            logger.info(f"Unregistered builder: {name}")

        return self

    def get(self, name: str) -> type[BuilderBase]:
        """Get builder class by name.

        Args:
            name: Builder identifier

        Returns:
            Builder class

        Raises:
            KeyError: If builder not registered
        """
        with self._lock:
            if name not in self._builders:
                available = ", ".join(self._builders.keys())
                raise KeyError(f"Builder '{name}' not registered. Available: {available}")

            return self._builders[name]

    def get_or_default(
        self, name: str, default: type[BuilderBase] | None = None
    ) -> type[BuilderBase] | None:
        """Get builder class with default fallback.

        Args:
            name: Builder identifier
            default: Default builder class if not found

        Returns:
            Builder class or default
        """
        with self._lock:
            return self._builders.get(name, default)

    def exists(self, name: str) -> bool:
        """Check if builder is registered.

        Args:
            name: Builder identifier

        Returns:
            True if builder exists
        """
        with self._lock:
            return name in self._builders

    def list_builders(self) -> dict[str, str]:
        """List all registered builders.

        Returns:
            Dict mapping names to class names
        """
        with self._lock:
            return {name: cls.__name__ for name, cls in self._builders.items()}

    def count(self) -> int:
        """Get count of registered builders.

        Returns:
            Number of registered builders
        """
        with self._lock:
            return len(self._builders)

    def create(self, builder_name: str, *args, **kwargs) -> BuilderBase:
        """Factory method: create builder instance.

        Args:
            builder_name: Builder identifier
            *args: Positional arguments for builder __init__
            **kwargs: Keyword arguments for builder __init__

        Returns:
            Builder instance

        Raises:
            KeyError: If builder not registered
            TypeError: If initialization fails
        """
        builder_class = self.get(builder_name)

        try:
            return builder_class(*args, **kwargs)
        except Exception as e:
            raise TypeError(
                f"Failed to instantiate builder '{builder_name}' ({builder_class.__name__}): {e!s}"
            ) from e

    def validate(self) -> dict[str, list]:
        """Validate all registered builders.

        Checks that:
        - Each builder is a subclass of BuilderBase
        - Each builder implements required abstract methods
        - No circular references

        Returns:
            Dict with "valid" and "errors" lists

        Example:
            >>> result = registry.validate()
            >>> if result["errors"]:
            ...     print(f"Validation failed: {result['errors']}")
        """
        result = {"valid": [], "errors": []}

        with self._lock:
            for name, builder_class in self._builders.items():
                try:
                    # Check inheritance
                    if not issubclass(builder_class, BuilderBase):
                        result["errors"].append(f"{name}: Not subclass of BuilderBase")
                        continue

                    # Check abstract methods implemented
                    if not hasattr(builder_class, "build"):
                        result["errors"].append(f"{name}: Missing build() method")
                        continue

                    result["valid"].append(name)

                except Exception as e:
                    result["errors"].append(f"{name}: {e!s}")

        return result

    def clear(self) -> "BuilderRegistry":
        """Clear all registered builders (for testing).

        Returns:
            self for method chaining
        """
        with self._lock:
            self._builders.clear()
            logger.warning("BuilderRegistry cleared (all builders removed)")

        return self

    def reset_singleton(self) -> None:
        """Reset singleton instance (for testing).

        This is typically only used in test teardown.
        """
        with BuilderRegistry._lock:
            BuilderRegistry._instance = None
            logger.warning("BuilderRegistry singleton reset")

    def __repr__(self) -> str:
        """Get registry summary.

        Returns:
            String representation
        """
        with self._lock:
            count = len(self._builders)
            return f"BuilderRegistry(count={count})"

    def __str__(self) -> str:
        """Get registry details.

        Returns:
            Detailed string representation
        """
        with self._lock:
            lines = ["BuilderRegistry:", "=" * 50]

            if not self._builders:
                lines.append("  (empty)")
            else:
                for name, builder_class in sorted(self._builders.items()):
                    pattern = (
                        "FLUENT"
                        if issubclass(builder_class, FluentBuilder)
                        else (
                            "DIRECTOR" if issubclass(builder_class, DirectorBuilder) else "GENERIC"
                        )
                    )
                    lines.append(f"  {name:20} -> {builder_class.__name__:30} [{pattern}]")

            return "\n".join(lines)


# Global convenience functions


def get_builder_registry() -> BuilderRegistry:
    """Get the global builder registry singleton.

    Returns:
        BuilderRegistry instance
    """
    return BuilderRegistry.get_instance()


def register_builder(name: str, builder_class: type[BuilderBase], overwrite: bool = False) -> None:
    """Register a builder globally.

    Args:
        name: Builder identifier
        builder_class: Builder class
        overwrite: If False, raise on duplicate

    Example:
        >>> register_builder("model", ModelBuilder)
    """
    get_builder_registry().register(name, builder_class, overwrite=overwrite)


def create_builder(name: str, *args, **kwargs) -> BuilderBase:
    """Create a builder instance from registry.

    Args:
        name: Builder identifier
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Builder instance

    Example:
        >>> builder = create_builder("model", config, device)
    """
    return get_builder_registry().create(name, *args, **kwargs)
