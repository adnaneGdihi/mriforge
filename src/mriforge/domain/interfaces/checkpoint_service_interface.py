from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class ICheckpointService(ABC):
    """Interface for model checkpoint management.

    ### Checkpoint Interface Hierarchy
    ```mermaid
    classDiagram
        class ICheckpointService {
            <<interface>>
            +save_checkpoint(model, optimizer, epoch, loss, path)
            +load_checkpoint(model, optimizer, path)
            +find_latest_checkpoint(dir)
        }
        class CheckpointServiceImpl {
            +save_checkpoint()
            +load_checkpoint()
        }
        ICheckpointService <|.. CheckpointServiceImpl
    ```
    """

    @abstractmethod
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        epoch: int,
        loss: float,
        file_path: str,
        **kwargs: Any,
    ) -> None:
        """Save a model checkpoint."""

    @abstractmethod
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Any,
        file_path: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Load a model checkpoint."""

    @abstractmethod
    def find_latest_checkpoint(self, checkpoint_dir: str) -> str | None:
        """Find the latest checkpoint in a directory."""
