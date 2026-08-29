"""CLI utilities for preset-based configuration loading.

This module provides utilities for loading presets via command-line interface,
supporting both preset selection and field-level overrides.

Example:
    >>> from mriforge.config.presets.cli import load_preset_or_config
    >>> # Load preset with overrides
    >>> config = load_preset_or_config(
    ...     preset_name="baseline_gan",
    ...     config_path=None,
    ...     overrides={"epochs": 200, "learning_rate": 1e-4}
    ... )
    >>> # Load config file (no preset)
    >>> config = load_preset_or_config(
    ...     preset_name=None,
    ...     config_path="experiments/training/experiment_01.yaml",
    ...     overrides={"batch_size": 64}
    ... )
"""

from pathlib import Path
from typing import Any

from mriforge.config.presets import (
    PresetRegistry,
    discover_and_register_presets,
    get_baseline_presets,
)
from mriforge.config.settings import TrainingSettings


def discover_presets(force_refresh: bool = False) -> None:
    """Discover and register all available presets.

    Args:
        force_refresh: If True, clear existing registry and re-discover.
                      Useful for testing or when new presets added at runtime.

    Raises:
        RuntimeError: If preset discovery fails.
    """
    if force_refresh:
        registry = PresetRegistry()
        registry._presets.clear()
        registry._categories.clear()
        registry._tags.clear()

    try:
        discover_and_register_presets()
    except Exception as e:
        raise RuntimeError(f"Failed to discover presets: {e}") from e


def list_available_presets(category: str | None = None) -> dict[str, Any]:
    """List all available presets, optionally filtered by category.

    Args:
        category: If provided, only list presets from this category
                 (e.g., 'gan', 'diffusion', 'reconstruction').

    Returns:
        Dictionary mapping preset names to their metadata.

    Example:
        >>> presets = list_available_presets(category="gan")
        >>> print(f"Available GAN presets: {list(presets.keys())}")
    """
    registry = PresetRegistry()

    if category:
        presets = registry.filter(category=category)
    else:
        presets = registry.list_all()

    return {p.name: p.to_dict() for p in presets}


def get_preset_categories() -> list[str]:
    """Get all available preset categories.

    Returns:
        List of category names (e.g., ['gan', 'diffusion', 'reconstruction']).
    """
    registry = PresetRegistry()
    return registry.list_categories()


def get_baseline_preset(category: str | None = None) -> dict[str, Any] | None:
    """Get baseline preset for a category, or any baseline if no category specified.

    Args:
        category: If provided, get baseline for this specific category.

    Returns:
        Preset metadata dict, or None if no baseline found.
    """
    registry = PresetRegistry()

    if category:
        presets = registry.filter(category=category, tag="baseline")
        return presets[0].to_dict() if presets else None

    baselines = get_baseline_presets()
    return baselines[0].to_dict() if baselines else None


def load_preset_or_config(
    preset_name: str | None = None,
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainingSettings:
    """Load configuration from preset or YAML file with optional overrides.

    Exactly one of preset_name or config_path must be provided.

    Args:
        preset_name: Name of registered preset to load (e.g., 'baseline_gan').
        config_path: Path to YAML configuration file.
        overrides: Dictionary of field overrides to apply
                  (e.g., {'epochs': 200, 'learning_rate': 1e-4}).

    Returns:
        Loaded and validated TrainingSettings instance.

    Raises:
        ValueError: If neither or both preset_name and config_path provided.
        KeyError: If preset_name not found in registry.
        FileNotFoundError: If config_path does not exist.
        ValidationError: If configuration is invalid.

    Example:
        >>> config = load_preset_or_config(
        ...     preset_name="baseline_gan",
        ...     overrides={"epochs": 200}
        ... )
        >>> # OR
        >>> config = load_preset_or_config(
        ...     config_path="experiments/training/experiment_01.yaml",
        ...     overrides={"batch_size": 64}
        ... )
    """
    if not overrides:
        overrides = {}

    # Validate inputs
    if (preset_name and config_path) or (not preset_name and not config_path):
        raise ValueError(
            "Exactly one of preset_name or config_path must be provided. "
            f"Got preset_name={preset_name}, config_path={config_path}"
        )

    if preset_name:
        registry = PresetRegistry()
        try:
            preset = registry.get(preset_name)
            return preset.to_settings(**overrides)
        except KeyError:
            available = [p.name for p in registry.list_all()]
            raise KeyError(
                f"Preset '{preset_name}' not found. "
                f"Available presets: {', '.join(sorted(available))}"
            ) from None

    else:
        # Load from YAML config file
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        settings = TrainingSettings.from_yaml(str(config_file))

        # Apply overrides (if any)
        if overrides:
            config_dict = settings.model_dump()
            _deep_update(config_dict, overrides)
            settings = TrainingSettings(**config_dict)

        return settings


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """Recursively update nested dictionary.

    Modifies target dict in-place by applying updates. Supports nested keys
    separated by dots (e.g., 'optimization.optimizer.learning_rate': 0.001).

    Args:
        target: Dictionary to update (modified in-place).
        updates: Dictionary of updates to apply.
    """
    for key, value in updates.items():
        if "." in key:
            # Handle nested keys like "optimization.optimizer.learning_rate"
            parts = key.split(".", 1)
            root = parts[0]
            rest = parts[1]

            if root not in target:
                target[root] = {}

            _deep_update(target[root], {rest: value})
        else:
            target[key] = value


if __name__ == "__main__":
    # Example usage
    discover_presets()

    print("Available preset categories:")
    for category in get_preset_categories():
        presets = list_available_presets(category=category)
        print(f"  {category}: {list(presets.keys())}")

    print("\nBaseline presets:")
    for baseline in get_baseline_presets():
        print(f"  {baseline.name} ({baseline.category})")

    print("\nLoading baseline_gan with overrides...")
    config = load_preset_or_config(
        preset_name="baseline_gan",
        overrides={"epochs": 200, "learning_rate": 0.001},
    )
    print(f"  Loaded config: {config.experiment.name}")
    print(f"  Epochs: {config.optimization.epochs}")
    print(f"  Learning rate: {config.optimization.optimizer.learning_rate}")
