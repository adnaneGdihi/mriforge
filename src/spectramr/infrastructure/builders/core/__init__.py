"""Core Builder Infrastructure

Phase 0: Foundational builder base classes for migration.

Public API:
    - BuilderBase: Abstract base class for all builders
    - FluentBuilder: Base for simple component builders
    - DirectorBuilder: Base for complex orchestration
    - ComposableBuilder: Builders that work with directors
    - BuilderRegistry: Central registry for all builders
    - get_builder_registry: Get the global registry
    - register_builder: Register a builder globally
    - create_builder: Create a builder from registry

Example:
    >>> from spectramr.infrastructure.builders.core import FluentBuilder, DirectorBuilder
    >>> from spectramr.infrastructure.builders.registry import BuilderRegistry
    >>>
    >>> class ModelBuilder(FluentBuilder[Dict]):
    ...     def build(self) -> Dict:
    ...         self.validate()
    ...         return self._models
    >>>
    >>> registry = BuilderRegistry.get_instance()
    >>> registry.register("model", ModelBuilder)
    >>> builder = registry.create("model", config, device)
    >>> models = builder.build()
"""

from .base_builder import BuilderBase, ComposableBuilder, DirectorBuilder, FluentBuilder

__all__ = [
    "BuilderBase",
    "ComposableBuilder",
    "DirectorBuilder",
    "FluentBuilder",
]
