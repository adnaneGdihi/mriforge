"""Dataset Service Interface
=========================

This module defines the interface for the dataset service.
"""

from abc import ABC, abstractmethod
from typing import Any


class IDatasetService(ABC):
    """Interface for dataset loading and management.

    ### Dataset Service Contract
    ```mermaid
    classDiagram
        class IDatasetService {
            <<interface>>
            +load_data(config)
        }
        class IDataLoader {
            <<interface>>
            +load_image_pair()
        }
        IDatasetService ..> IDataLoader : uses
    ```
    """

    @abstractmethod
    def load_data(self, config: Any) -> Any:
        """Load data based on the provided configuration."""
