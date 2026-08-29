"""Interfaces for MRI Dataset Components
=====================================

This module defines interfaces for the MRI dataset components following
SOLID principles. Each interface represents a single responsibility.

.. mermaid::

    classDiagram
        class IDataLoader {
            <<interface>>
            +load_image_pair()
        }
        class IImageTransformer {
            <<interface>>
            +transform_to_tensor()
        }
        class IAugmentationStrategy {
            <<interface>>
            +apply_augmentations()
        }
        class ICacheManager {
            <<interface>>
            +get_cached_item()
            +cache_item()
        }
        class IFileDiscoverer {
            <<interface>>
            +find_image_files()
        }
"""

from abc import ABC, abstractmethod

import torch
from PIL import Image


class IDataLoader(ABC):
    """Interface for loading image data."""

    @abstractmethod
    def load_image_pair(
        self,
        lr_path: str,
        hr_path: str,
    ) -> tuple[Image.Image, Image.Image]:
        """Load a single LR/HR image pair from disk."""


class IImageTransformer(ABC):
    """Interface for transforming images to tensors."""

    @abstractmethod
    def transform_to_tensor(
        self,
        lr_img: Image.Image,
        hr_img: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert PIL images to tensors with standard preprocessing."""


class IAugmentationStrategy(ABC):
    """Interface for applying data augmentations."""

    @abstractmethod
    def apply_augmentations(
        self,
        lr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to image tensors."""


class ICacheManager(ABC):
    """Interface for managing data caching."""

    @abstractmethod
    def get_cached_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get cached item if available."""

    @abstractmethod
    def cache_item(self, idx: int, data: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Cache an item."""

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if caching is enabled."""


class IFileDiscoverer(ABC):
    """Interface for discovering image files."""

    @abstractmethod
    def find_image_files(self, directory: str) -> list[str]:
        """Find all supported image files in a directory."""
