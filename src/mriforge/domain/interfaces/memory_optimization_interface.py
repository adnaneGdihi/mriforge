"""Memory Optimization Service Interface
=====================================

This module defines the interface for the memory optimization service.
"""

from abc import ABC, abstractmethod


class IMemoryOptimizationService(ABC):
    """Interface for memory optimization techniques.

    ### Memory Optimization Contract
    ```mermaid
    classDiagram
        class IMemoryOptimizationService {
            <<interface>>
            +optimize_memory()
            +clear_cache()
            +get_memory_stats()
        }
    ```
    """

    @abstractmethod
    def optimize_memory(self) -> None:
        """Apply memory optimization techniques."""
