"""Configuration presets and templates for quick experiment setup.

This module provides a unified preset registry system that allows users to:
1. Select pre-configured templates for common training paradigms
2. Combine multiple presets for advanced configurations
3. Override specific fields via CLI or programmatically

Example:
    >>> from spectramr.config.presets import PresetRegistry, discover_and_register_presets
    >>> discover_and_register_presets()
    >>> registry = PresetRegistry()
    >>> gan_preset = registry.get("experiment_01_baseline_gan")
    >>> config = gan_preset.to_settings(epochs=200, learning_rate=1e-4)

    >>> # List all available presets
    >>> all_presets = registry.list_all()
    >>> print(f"Registered {len(all_presets)} presets")

    >>> # Filter by category
    >>> gan_presets = registry.filter(category="gan")
    >>> baseline_presets = registry.filter(tag="baseline")
"""

from .discovery import discover_and_register_presets, get_baseline_presets
from .preset import ConfigPreset
from .registry import PresetRegistry

__all__ = [
    "ConfigPreset",
    "PresetRegistry",
    "discover_and_register_presets",
    "get_baseline_presets",
]
