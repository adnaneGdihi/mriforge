"""Configuration preset wrapper class."""

from pathlib import Path
from typing import Any

import yaml

from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
from spectramr.config.settings import TrainingSettings


class ConfigPreset:
    """Wraps a YAML configuration as a reusable preset.

    Presets provide named, versioned, documented training configurations
    that can be loaded and customized.

    Example:
        >>> preset = ConfigPreset(
        ...     name="baseline_gan",
        ...     category="gan",
        ...     description="Standard GAN baseline for reconstruction",
        ...     yaml_path="presets/baseline_gan.yaml",
        ...     version="6.0"
        ... )
        >>> config = preset.to_settings(epochs=200)
    """

    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        yaml_path: str,
        version: str = CANONICAL_CONFIG_VERSION,
        tags: list[str] | None = None,
        parent_presets: list[str] | None = None,
    ):
        """Initialize a configuration preset.

        Args:
            name: Unique preset name (e.g., "baseline_gan")
            category: Preset category (e.g., "gan", "diffusion", "reconstruction")
            description: Human-readable description
            yaml_path: Path to YAML configuration file
            version: Configuration schema version (default: the canonical one)
            tags: Optional tags for categorization (e.g., ["stable", "recommended"])
            parent_presets: Optional list of parent preset names to inherit from
        """
        self.name = name
        self.category = category
        self.description = description
        self.yaml_path = Path(yaml_path)
        self.version = version
        self.tags = tags or []
        self.parent_presets = parent_presets or []

    def load_config(self) -> dict[str, Any]:
        """Load the YAML configuration file.

        Returns:
            Dictionary of configuration data

        Raises:
            FileNotFoundError: If YAML file not found
            ValueError: If version doesn't match
        """
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Preset config file not found: {self.yaml_path}")

        with open(self.yaml_path) as f:
            data = yaml.safe_load(f)

        if data.get("config_version") != self.version:
            raise ValueError(
                f"Preset {self.name} has version {data.get('config_version')}, "
                f"expected {self.version}"
            )

        return data

    def to_settings(self, **overrides: Any) -> TrainingSettings:
        """Convert preset to TrainingSettings with optional overrides.

        Args:
            **overrides: Field overrides (nested access supported)
                Example: epochs=200, optimization={'learning_rate': 1e-4}

        Returns:
            TrainingSettings configured from preset + overrides

        Raises:
            FileNotFoundError: If preset YAML not found
            ValueError: If configuration is invalid
        """
        config_data = self.load_config()

        # Apply overrides
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config_data:
                # Merge nested dicts
                if isinstance(config_data[key], dict):
                    config_data[key].update(value)
                else:
                    config_data[key] = value
            else:
                config_data[key] = value

        return TrainingSettings(**config_data)

    def __repr__(self) -> str:
        """__repr__.

        Returns:
            str: Description.
        """
        return (
            f"ConfigPreset(name={self.name!r}, category={self.category!r}, "
            f"description={self.description!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Export preset metadata as dictionary.

        Returns:
            Dictionary of preset metadata
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "yaml_path": str(self.yaml_path),
            "version": self.version,
            "tags": self.tags,
            "parent_presets": self.parent_presets,
        }
