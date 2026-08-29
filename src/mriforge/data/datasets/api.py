"""Unified public API for dataset creation and registration.

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

from mriforge.config.settings import TrainingSettings
from mriforge.data.datasets.factory import initialize_dataset_registry

# Initialize the registry bridge
initialize_dataset_registry()


def get_registry() -> Any:
    """Return the shared :class:`DatasetRegistry` instance."""
    from mriforge.domain.entities.data.dataset_registry import get_dataset_registry

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


def register_dataset(name: str, dataset_class: type[Dataset[Any]]) -> None:
    """Register a dataset implementation.

    Args:
        name: Name to register the dataset under
        dataset_class: Dataset class to register
    """
    registry = get_registry()
    registry.register(name, dataset_class)


__all__ = [
    "create_dataset",
    "get_registry",
    "register_dataset",
]
