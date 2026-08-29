"""Unified preset registry for configuration management."""

import logging
from typing import Optional

from .preset import ConfigPreset

logger = logging.getLogger(__name__)


class PresetRegistry:
    """Central registry for all configuration presets.

    Provides O(1) lookup of presets by name and filtering by category/tags.
    Singleton pattern ensures consistent registry across application.

    Example:
        >>> registry = PresetRegistry()
        >>> registry.register(ConfigPreset(...))
        >>> preset = registry.get("baseline_gan")
        >>> all_gan_presets = registry.filter(category="gan")
        >>> presets_with_tag = registry.filter(tag="recommended")
    """

    _instance: Optional["PresetRegistry"] = None
    _presets: dict[str, ConfigPreset] = {}
    _categories: dict[str, set[str]] = {}
    _tags: dict[str, set[str]] = {}

    def __new__(cls) -> "PresetRegistry":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, preset: ConfigPreset) -> None:
        """Register a new preset.

        Args:
            preset: ConfigPreset instance to register

        Raises:
            ValueError: If preset name already registered
        """
        instance = cls()

        if preset.name in instance._presets:
            raise ValueError(f"Preset {preset.name!r} already registered")

        instance._presets[preset.name] = preset

        # Update category index
        if preset.category not in instance._categories:
            instance._categories[preset.category] = set()
        instance._categories[preset.category].add(preset.name)

        # Update tags index
        for tag in preset.tags:
            if tag not in instance._tags:
                instance._tags[tag] = set()
            instance._tags[tag].add(preset.name)

        logger.debug(f"Registered preset {preset.name!r} (category={preset.category!r})")

    @classmethod
    def get(cls, name: str) -> ConfigPreset:
        """Get preset by name.

        Args:
            name: Preset name

        Returns:
            ConfigPreset instance

        Raises:
            KeyError: If preset not found
        """
        instance = cls()

        if name not in instance._presets:
            available = ", ".join(sorted(instance._presets.keys()))
            raise KeyError(f"Preset {name!r} not found. Available: [{available}]")

        return instance._presets[name]

    @classmethod
    def list_all(cls) -> list[ConfigPreset]:
        """List all registered presets.

        Returns:
            List of ConfigPreset instances sorted by name
        """
        instance = cls()
        return sorted(instance._presets.values(), key=lambda p: p.name)

    @classmethod
    def filter(
        cls,
        category: str | None = None,
        tag: str | None = None,
        name_pattern: str | None = None,
    ) -> list[ConfigPreset]:
        """Filter presets by category, tag, or name pattern.

        Args:
            category: Filter by category
            tag: Filter by tag
            name_pattern: Filter by name substring (case-insensitive)

        Returns:
            Filtered list of ConfigPreset instances
        """
        instance = cls()
        results = set(instance._presets.keys())

        if category:
            results &= instance._categories.get(category, set())

        if tag:
            results &= instance._tags.get(tag, set())

        if name_pattern:
            pattern_lower = name_pattern.lower()
            results = {name for name in results if pattern_lower in name.lower()}

        return sorted(
            [instance._presets[name] for name in results],
            key=lambda p: p.name,
        )

    @classmethod
    def list_categories(cls) -> list[str]:
        """List all available categories.

        Returns:
            Sorted list of category names
        """
        instance = cls()
        return sorted(instance._categories.keys())

    @classmethod
    def list_tags(cls) -> list[str]:
        """List all available tags.

        Returns:
            Sorted list of tag names
        """
        instance = cls()
        return sorted(instance._tags.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered presets (for testing).

        WARNING: This clears all presets. Use only in tests.
        """
        instance = cls()
        instance._presets.clear()
        instance._categories.clear()
        instance._tags.clear()
        logger.warning("PresetRegistry cleared")

    @classmethod
    def get_info(cls, name: str) -> dict:
        """Get full preset information.

        Args:
            name: Preset name

        Returns:
            Dictionary with preset metadata and info

        Raises:
            KeyError: If preset not found
        """
        preset = cls.get(name)
        return {
            **preset.to_dict(),
            "num_parent_presets": len(preset.parent_presets),
        }

    @classmethod
    def list_info(cls) -> list[dict]:
        """Get information on all presets.

        Returns:
            List of preset information dictionaries
        """
        return [PresetRegistry.get_info(p.name) for p in cls.list_all()]
