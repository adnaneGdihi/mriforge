"""Unified public API for dataset creation.

.. note::

   ``register_dataset`` used to live here and **could never be called**. It did
   ``registry.register(name, dataset_class)`` against
   :meth:`DatasetRegistry.register`, whose signature is ``(entity: DatasetEntity)``
   -- one argument, and an entity rather than a (name, class) pair. Every call
   raised ``TypeError``. It survived because the only test touching this module
   asserted that an import STRING appeared in a source file, so it never
   executed either function.

   Registration lives in :func:`spectramr.data.datasets.registry.register_dataset`
   -- the registry ``DataPipelineDirector`` actually dispatches ``dataset_type``
   on (non-negotiable 17: one owner).

This module exposes canonical helpers so downstream code can refer to a
single namespace when constructing datasets.

.. mermaid::

    classDiagram
        class API {
            +create_dataset(name)
            +register_dataset(name, cls)
            +get_registry()
        }
        class DatasetRegistry {
            +create_dataset()
            +register()
        }

        API --> DatasetRegistry : delegates
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from spectramr.config.settings import TrainingSettings
from spectramr.data.datasets.factory import initialize_dataset_registry

# Initialize the registry bridge
initialize_dataset_registry()


def get_registry() -> Any:
    """Return the shared :class:`DatasetRegistry` instance."""
    from spectramr.domain.entities.data.dataset_registry import get_dataset_registry

    return get_dataset_registry()


def create_dataset(name: str, config: TrainingSettings, **kwargs: Any) -> Dataset[Any]:
    """Create a dataset by name.

    Args:
        name: Name of the dataset to create
        config: Training configuration object
        **kwargs: Arguments to pass to the dataset constructor

    Returns:
        Dataset instance
    """
    registry = get_registry()
    return registry.create_dataset(name, config=config, **kwargs)


__all__ = [
    "create_dataset",
    "get_registry",
]
