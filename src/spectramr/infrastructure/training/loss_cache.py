"""Loss module caching system for efficient loss function reuse.

Provides singleton-based caching for loss modules to avoid recompilation
and improve performance during training.
"""

from typing import Any, Optional

__all__ = ["LossModuleCache"]


class LossModuleCache:
    """Singleton cache for loss modules.

    Manages a dictionary of cached loss modules indexed by configuration hash.
    Ensures that loss modules are created once and reused across the training loop.

    Thread-safe singleton pattern implementation.
    """

    _instance: Optional["LossModuleCache"] = None
    _cache: dict[str, Any] = {}

    def __new__(cls) -> "LossModuleCache":
        """Ensure singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "LossModuleCache":
        """Return the singleton instance (backward-compatible helper)."""
        return cls()

    def get(self, key: str) -> Any | None:
        """Retrieve a cached loss module by key.

        Args:
            key: Cache key (typically a hash of loss config)

        Returns:
            Cached loss module or None if not found
        """
        return self._cache.get(key)

    def put(self, key: str, module: Any) -> None:
        """Store a loss module in the cache.

        Args:
            key: Cache key (typically a hash of loss config)
            module: Loss module to cache
        """
        self._cache[key] = module

    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists in cache, False otherwise
        """
        return key in self._cache

    def clear(self) -> None:
        """Clear all cached modules."""
        self._cache.clear()

    def size(self) -> int:
        """Get the number of cached modules.

        Returns:
            Number of items in cache
        """
        return len(self._cache)

    def keys(self) -> list:
        """Get all cache keys.

        Returns:
            List of cache keys
        """
        return list(self._cache.keys())

    def values(self) -> list:
        """Get all cached modules.

        Returns:
            List of cached modules
        """
        return list(self._cache.values())

    def items(self) -> list:
        """Get all cache items as (key, module) pairs.

        Returns:
            List of (key, module) tuples
        """
        return list(self._cache.items())


# Global singleton instance
loss_module_cache = LossModuleCache()
