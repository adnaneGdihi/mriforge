"""Model Card Service Interface
============================

This module defines the interface for the model card service.
"""

from abc import ABC, abstractmethod
from typing import Any


class IModelCardService(ABC):
    """Interface for creating and managing model cards.

    ### Model Card Contract
    ```mermaid
    classDiagram
        class IModelCardService {
            <<interface>>
            +create_model_card(model_info)
            +save_model_card(model_card, filename)
        }
    ```
    """

    @abstractmethod
    def create_model_card(self, model_info: dict[str, Any]) -> Any:
        """Create a ModelCard instance."""

    @abstractmethod
    def save_model_card(self, model_card: Any, filename: str) -> None:
        """Save a model card to a file."""
