__all__ = [
    "Any",
    "IConfig",
    "IConfigLoader",
    "IConfigValidator",
    "IDataConfig",
    "IModelConfig",
    "ITrainingConfig",
    "Path",
    "Protocol",
]

"""Core Configuration Interfaces
===========================

This module defines the core interfaces for configuration management.
These interfaces follow SOLID principles, particularly Interface Segregation.
"""

from pathlib import Path
from typing import Any, Protocol


class IConfigLoader(Protocol):
    """Interface for configuration loading."""

    def load_from_file(self, path: Path) -> "IConfig":
        """Loads configuration from file."""
        ...

    def load_from_dict(self, config_dict: dict[str, Any]) -> "IConfig":
        """Loads configuration from dictionary."""
        ...


class IConfigValidator(Protocol):
    """Interface for configuration validation."""

    def validate(self, config: "IConfig") -> bool:
        """Validates configuration."""
        ...

    def get_validation_errors(self) -> list[str]:
        """Returns validation errors."""
        ...


class ITrainingConfig(Protocol):
    """Protocol for training configuration."""

    @property
    def epochs(self) -> int:
        """Number of training epochs."""
        ...

    @property
    def batch_size(self) -> int:
        """Batch size for training."""
        ...

    @property
    def learning_rate(self) -> float:
        """Learning rate for training."""
        ...

    @property
    def model_type(self) -> str:
        """Type of model to train."""
        ...


class IModelConfig(Protocol):
    """Protocol for model configuration."""

    @property
    def in_channels(self) -> int:
        """Number of input channels."""
        ...

    @property
    def out_channels(self) -> int:
        """Number of output classes."""
        ...

    @property
    def model_kwargs(self) -> dict[str, Any]:
        """Additional model parameters."""
        ...


class IDataConfig(Protocol):
    """Protocol for data configuration."""

    @property
    def input_lr_dir(self) -> str:
        """Directory for low-resolution input data."""
        ...

    @property
    def input_hr_dir(self) -> str:
        """Directory for high-resolution input data."""
        ...

    @property
    def batch_size(self) -> int:
        """Batch size for data loading."""
        ...


class IConfig(
    IConfigLoader,
    IConfigValidator,
    ITrainingConfig,
    IModelConfig,
    IDataConfig,
    Protocol,
):
    """Main configuration interface combining all aspects."""

    def save_to_file(self, path: Path) -> None:
        """Saves configuration to file."""
        ...

    def update_from_dict(self, updates: dict[str, Any]) -> None:
        """Updates configuration from dictionary."""
        ...

    def get_section(self, section_name: str) -> dict[str, Any]:
        """Gets a configuration section."""
        ...
