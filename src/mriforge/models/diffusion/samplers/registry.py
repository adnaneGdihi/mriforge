"""``@register_sampler`` decorator + lookup helpers.

Mirrors the existing ``mriforge.models.losses.registry`` and
``mriforge.core.metrics.registry`` patterns so that any code expecting one
of those finds an analogous surface here. Aliases dispatch case-
insensitively to a canonical name.

Implements ``CC-1`` from
``TODO/backlog_paradigm_expansion_roadmap.md``.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class SamplerRegistry:
    """Class-level singleton mapping sampler name → constructor."""

    _samplers: ClassVar[dict[str, type]] = {}
    _aliases: ClassVar[dict[str, str]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        aliases: list[str] | None = None,
    ):
        """Decorator: register a sampler class under ``name``.

        Args:
            name: Canonical (lowercased) sampler key.
            aliases: Optional alternative spellings (e.g. ``["DDIM", "ddim_v1"]``).

        Returns:
            The decorator.
        """
        canonical = name.lower()

        def _decorator(sampler_cls: type) -> type:
            existing = cls._samplers.get(canonical)
            if existing is not None and existing is not sampler_cls:
                # Same name, different class — silently overwrite (matches the
                # existing metric registry's behaviour during reloads).
                logger.debug(
                    "Sampler %r being re-registered with a different class",
                    canonical,
                )
            cls._samplers[canonical] = sampler_cls
            if aliases:
                for alias in aliases:
                    cls._aliases[alias.lower()] = canonical
            return sampler_cls

        return _decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> Any:
        """Instantiate a sampler by canonical name or alias.

        Raises:
            KeyError: when ``name`` is not registered. The error message
                includes the available canonical names — fail-loud per
                CLAUDE.md pitfall #9.
        """
        lookup = name.lower()
        canonical = cls._aliases.get(lookup, lookup)
        if canonical not in cls._samplers:
            available = ", ".join(sorted(cls._samplers.keys())) or "<none>"
            raise KeyError(f"Unknown sampler {name!r}. Available: [{available}]")
        return cls._samplers[canonical](**kwargs)

    @classmethod
    def get_class(cls, name: str) -> type:
        """Return the sampler class without instantiating."""
        lookup = name.lower()
        canonical = cls._aliases.get(lookup, lookup)
        if canonical not in cls._samplers:
            available = ", ".join(sorted(cls._samplers.keys())) or "<none>"
            raise KeyError(f"Unknown sampler {name!r}. Available: [{available}]")
        return cls._samplers[canonical]

    @classmethod
    def list_available(cls) -> list[str]:
        """Return the sorted canonical keys."""
        return sorted(cls._samplers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """True if ``name`` is a canonical key or a registered alias."""
        lookup = name.lower()
        return lookup in cls._samplers or lookup in cls._aliases


# Convenience module-level handles mirroring the metrics/loss registries.
SAMPLER_REGISTRY = SamplerRegistry


def register_sampler(name: str, aliases: list[str] | None = None):
    """Free-function decorator wrapping ``SamplerRegistry.register``."""
    return SamplerRegistry.register(name, aliases)


def get_sampler(name: str, **kwargs: Any) -> Any:
    """Instantiate a sampler by name."""
    return SamplerRegistry.get(name, **kwargs)


def list_available() -> list[str]:
    """Return the sorted canonical sampler names."""
    return SamplerRegistry.list_available()
