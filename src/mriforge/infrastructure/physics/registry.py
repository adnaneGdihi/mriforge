"""Forward Operator Registry and Interfaces
========================================

This module provides a unified registry system for forward operators in MRI reconstruction.
It defines interfaces for forward operators and provides a plugin-based registry for
different operator implementations (FFT, NUFFT, etc.).

Key Components:
- Uses unified IPhysicsOperator interface from interfaces.py
- OperatorRegistry: Plugin-based registry for operator types
- Factory functions for creating operators by type
"""

from __future__ import annotations

from typing import Any

# Import unified interface
from .interfaces import IPhysicsOperator, IPhysicsOperatorABC

# Use unified interface (backward compatibility alias)
IForwardOperator = IPhysicsOperator
BaseForwardOperator = IPhysicsOperatorABC


class OperatorRegistry:
    r"""Registry for forward operator implementations.

    Provides plugin-based registration and instantiation of different
    operator types (FFT, NUFFT, etc.).
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{OperatorRegistry} = \emptyset"""

    def __init__(self):
        """__init__."""
        self._operators: dict[str, type[BaseForwardOperator]] = {}
        self._operator_configs: dict[str, dict[str, Any]] = {}

    def register_operator(
        self,
        name: str,
        operator_class: type[BaseForwardOperator],
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register a new operator type.

        Args:
            name: Unique name for the operator type
            operator_class: Class implementing BaseForwardOperator
            config: Optional default configuration for the operator

        """
        if not issubclass(operator_class, BaseForwardOperator):
            raise TypeError(
                f"Operator class {operator_class} must inherit from BaseForwardOperator",
            )

        self._operators[name] = operator_class
        self._operator_configs[name] = config or {}

    def get_operator_class(self, name: str) -> type[BaseForwardOperator]:
        """Get operator class by name.

        Args:
            name: Registered operator name

        Returns:
            Operator class

        Raises:
            KeyError: If operator name not found

        """
        if name not in self._operators:
            available = list(self._operators.keys())
            raise KeyError(
                f"Operator '{name}' not found. Available operators: {available}",
            )
        return self._operators[name]

    def get_operator_config(self, name: str) -> dict[str, Any]:
        """Get default configuration for operator.

        Args:
            name: Registered operator name

        Returns:
            Configuration dictionary

        """
        return self._operator_configs.get(name, {}).copy()

    def list_operators(self) -> dict[str, dict[str, Any]]:
        """list all registered operators with their configurations.

        Returns:
            Dictionary mapping operator names to their metadata

        """
        return {
            name: {
                "class": cls.__name__,
                "config": self._operator_configs.get(name, {}),
            }
            for name, cls in self._operators.items()
        }

    def create_operator(
        self,
        name: str,
        **kwargs: Any,
    ) -> BaseForwardOperator:
        """Create an operator instance.

        Args:
            name: Registered operator name
            **kwargs: Additional arguments for operator initialization

        Returns:
            Configured operator instance

        """
        operator_class = self.get_operator_class(name)
        config = self.get_operator_config(name)

        # Merge default config with provided kwargs
        merged_config = {**config, **kwargs}

        try:
            return operator_class(**merged_config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create operator '{name}' with config {merged_config}: {e}",
            ) from e


# Global registry instance
_operator_registry = OperatorRegistry()


def register_operator(
    name: str,
    operator_class: type[BaseForwardOperator],
    config: dict[str, Any] | None = None,
) -> None:
    """Global function to register an operator type.

    Args:
        name: Unique name for the operator type
        operator_class: Class implementing BaseForwardOperator
        config: Optional default configuration

    """
    _operator_registry.register_operator(name, operator_class, config)


def get_operator_class(name: str) -> type[BaseForwardOperator]:
    """Get operator class by name from global registry.

    Args:
        name: Registered operator name

    Returns:
        Operator class

    """
    return _operator_registry.get_operator_class(name)


def get_operator_config(name: str) -> dict[str, Any]:
    """Get default configuration for operator from global registry.

    Args:
        name: Registered operator name

    Returns:
        Configuration dictionary

    """
    return _operator_registry.get_operator_config(name)


def list_operators() -> dict[str, dict[str, Any]]:
    """list all registered operators from global registry.

    Returns:
        Dictionary mapping operator names to their metadata

    """
    return _operator_registry.list_operators()


def create_operator(name: str, **kwargs: Any) -> BaseForwardOperator:
    """Create an operator instance from global registry.

    Args:
        name: Registered operator name
        **kwargs: Additional arguments for operator initialization

    Returns:
        Configured operator instance

    """
    return _operator_registry.create_operator(name, **kwargs)


def get_registry() -> OperatorRegistry:
    """Get the global operator registry instance.

    Returns:
        Global OperatorRegistry instance

    """
    return _operator_registry


__all__ = [
    "BaseForwardOperator",
    "IForwardOperator",
    "OperatorRegistry",
    "create_operator",
    "get_operator_class",
    "get_operator_config",
    "get_registry",
    "list_operators",
    "register_operator",
]
