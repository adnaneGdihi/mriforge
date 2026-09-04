from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any


class IProfilingService(ABC):
    """Abstract interface for a profiling service.

    ### Profiling Service Contract
    ```mermaid
    classDiagram
        class IProfilingService {
            <<interface>>
            +bool enabled
            +enable()
            +disable()
            +reset()
            +profile(name, metadata) context
            +record_event(name, duration, metadata)
            +snapshot() list
            +flush_metrics(prefix) dict
        }
    ```
    """

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether profiling is enabled."""

    @abstractmethod
    def enable(self) -> None:
        """Enable profiling."""

    @abstractmethod
    def disable(self) -> None:
        """Disable profiling."""

    @abstractmethod
    def reset(self) -> None:
        """Reset profiling data."""

    @abstractmethod
    def profile(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[None]:
        """Context manager for profiling a code block.

        Args:
            name: Name of the profiling event
            metadata: Optional metadata for the event

        Yields:
            None
        """

    @abstractmethod
    def record_event(
        self,
        name: str,
        *,
        duration_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a profiling event manually.

        Args:
            name: Name of the event
            duration_s: Duration in seconds
            metadata: Optional metadata
        """

    @abstractmethod
    def snapshot(self) -> list[Any]:
        """Get a snapshot of profiling events.

        Returns:
            List of profiling events
        """

    @abstractmethod
    def flush_metrics(self, *, prefix: str = "profile") -> dict[str, float]:
        """Flush profiling metrics.

        Args:
            prefix: Prefix for metric keys

        Returns:
            Dictionary of profiling metrics
        """

    @abstractmethod
    def extend_metrics(
        self,
        target: dict[str, float],
        *,
        prefix: str = "profile",
    ) -> dict[str, float]:
        """Extend target dictionary with profiling metrics.

        Args:
            target: Dictionary to extend
            prefix: Prefix for metric keys

        Returns:
            Extended dictionary
        """
