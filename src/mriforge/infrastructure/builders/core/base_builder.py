"""Base Builder Classes for Unified Builder Pattern Architecture

Phase 0: Foundation infrastructure for builder pattern transition.

This module provides two base builder classes that form the foundation
for the entire builder migration:

1. FluentBuilder: For simple component builders (<5 responsibilities)
   - Uses fluent/chainable API
   - Stores state in private attributes
   - Minimal validation overhead
   - Examples: ModelBuilder, OptimizerBuilder, LossBuilder

2. DirectorBuilder: For complex orchestration (>5 sub-builders)
   - Manages multiple sub-builders
   - Controls composition order
   - Provides structured validation
   - Examples: TrainingEnvironmentBuilder, InferenceEnvironmentBuilder

Both inherit from BuilderBase which establishes the builder contract.

Design Pattern: Builder Pattern + Template Method Pattern
Compatibility: Works with existing Builder ABC in base.py
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")  # Generic type for built product


class BuilderBase(ABC):
    """Abstract base class that establishes the builder contract.

    All builders must implement build() and validate() methods.
    This maintains compatibility with existing src/infrastructure/training/builders/base.py.

    Design Intent:
        - Enforce strict contract for all builder implementations
        - Support method chaining through fluent API
        - Enable validation before build completion
        - Support both simple (fluent) and complex (director) patterns

    Attributes:
        _product: The built product (set by concrete builders)
        _validated: Whether validation has passed before build()
    """

    def __init__(self):
        """Initialize base builder."""
        self._product: Any | None = None
        self._validated: bool = False

    @abstractmethod
    def build(self) -> Any:
        """Build and return the product.

        Must be implemented by all concrete builders.
        Should return the final constructed object(s).

        Returns:
            Any: The built product (dict, tuple, or single object)

        Raises:
            ValueError: If required components are missing or invalid
        """
        pass

    def validate(self) -> "BuilderBase":
        """Validate builder state before build.

        Override in subclasses to add specific validation logic.
        This is called automatically by build() for non-fluent builders,
        or manually by fluent builders before final build() call.

        Returns:
            self: To enable method chaining

        Raises:
            ValueError: If validation fails
        """
        self._validated = True
        return self

    def is_validated(self) -> bool:
        """Check if builder has been validated.

        Returns:
            True if validate() has been called
        """
        return self._validated


class FluentBuilder(BuilderBase, Generic[T]):
    """Base class for simple component builders using fluent API.

    Use this for builders with <5 responsibilities that need chainable,
    readable API. Examples: ModelBuilder, OptimizerBuilder, LossBuilder.

    Design Pattern: Fluent Builder Pattern

    Features:
        - Method chaining for readability: builder.method1().method2().build()
        - Each method returns self
        - Clean, expressive API
        - Simple state management in private attributes
        - Minimal validation overhead

    Example:
        >>> class ModelBuilder(FluentBuilder[Dict]):
        ...     def __init__(self, config, device):
        ...         super().__init__()
        ...         self._config = config
        ...         self._device = device
        ...         self._models = {}
        ...
        ...     def with_generator(self, gen: nn.Module) -> "ModelBuilder":
        ...         self._models["generator"] = gen
        ...         return self
        ...
        ...     def with_discriminator(self, disc: nn.Module) -> "ModelBuilder":
        ...         self._models["discriminator"] = disc
        ...         return self
        ...
        ...     def build(self) -> Dict[str, nn.Module]:
        ...         self.validate()
        ...         return self._models

    Guidelines:
        - All instance attributes should be private (start with _)
        - All builder methods should return self
        - Validation should check required attributes
        - Final build() should return the product
    """

    def __init__(self):
        """Initialize fluent builder."""
        super().__init__()

    @abstractmethod
    def build(self) -> T:
        """Build and return the product.

        Subclasses must implement this. It should:
        1. Call self.validate() once to check state
        2. Set self._product
        3. Return self._product (or typed equivalent)

        Returns:
            T: The built product

        Raises:
            ValueError: If validation fails
        """
        pass

    def with_config(self, config: Any) -> "FluentBuilder[T]":
        """Optional: Add config to builder (can be overridden).

        Args:
            config: Configuration object

        Returns:
            self for chaining
        """
        if not hasattr(self, "_config"):
            self._config = config
        return self

    def with_device(self, device: Any) -> "FluentBuilder[T]":
        """Optional: Add device to builder (can be overridden).

        Args:
            device: Device specification (torch.device or str)

        Returns:
            self for chaining
        """
        if not hasattr(self, "_device"):
            self._device = device
        return self

    def summary(self) -> str:
        """Get human-readable summary of builder state.

        Override in subclasses to provide specific details.

        Returns:
            Summary string
        """
        return (
            f"{self.__class__.__name__} "
            f"(validated={self._validated}, "
            f"product={'set' if self._product else 'not set'})"
        )


class DirectorBuilder(BuilderBase, Generic[T]):
    """Base class for complex orchestration using Director Pattern.

    Use this for builders with >5 responsibilities that coordinate
    multiple sub-builders. Examples: TrainingEnvironmentBuilder,
    InferenceEnvironmentBuilder.

    Design Pattern: Director Pattern (with Builder sub-pattern)

    Features:
        - Manages multiple sub-builders
        - Controls composition order
        - Structured validation at multiple levels
        - Clear separation of concerns
        - Easier to test and debug

    Example:
        >>> class TrainingEnvironmentBuilder(DirectorBuilder[TrainingEnvironment]):
        ...     def __init__(self, config: TrainingSettings):
        ...         super().__init__()
        ...         self._config = config
        ...         self._sub_builders = {}
        ...
        ...     def with_model_builder(self, builder: ModelBuilder) -> "TrainingEnvironmentBuilder":
        ...         self._sub_builders["model"] = builder
        ...         return self
        ...
        ...     def build(self) -> TrainingEnvironment:
        ...         self.validate()
        ...         # Build in correct order
        ...         models = self._sub_builders["model"].build()
        ...         optimizers = self._sub_builders["optimization"].build()
        ...         # ... more sub-builders
        ...         return TrainingEnvironment(models, optimizers, ...)

    Guidelines:
        - Define sub-builders as private attributes
        - Use with_*_builder() methods to add sub-builders
        - Implement build() to orchestrate sub-builders in correct order
        - Call self.validate() once before building sub-products
        - Level 1: Validate director state (all sub-builders present)
        - Level 2: Call validate() on each sub-builder
        - Level 3: Build each sub-product in correct order
    """

    def __init__(self):
        """Initialize director builder."""
        super().__init__()
        self._sub_builders: dict[str, BuilderBase] = {}

    def add_sub_builder(self, name: str, builder: BuilderBase) -> "DirectorBuilder[T]":
        """Register a sub-builder.

        Args:
            name: Identifier for sub-builder (e.g., "model", "optimization")
            builder: Sub-builder instance

        Returns:
            self for chaining

        Raises:
            TypeError: If builder doesn't inherit from BuilderBase
        """
        if not isinstance(builder, BuilderBase):
            raise TypeError(f"Expected BuilderBase, got {type(builder)}")

        self._sub_builders[name] = builder
        return self

    def has_sub_builder(self, name: str) -> bool:
        """Check if sub-builder is registered.

        Args:
            name: Sub-builder identifier

        Returns:
            True if sub-builder exists
        """
        return name in self._sub_builders

    def get_sub_builder(self, name: str) -> BuilderBase | None:
        """Get registered sub-builder.

        Args:
            name: Sub-builder identifier

        Returns:
            Sub-builder instance or None

        Raises:
            KeyError: If sub-builder not found
        """
        return self._sub_builders.get(name)

    def validate_sub_builders(self) -> "DirectorBuilder[T]":
        """Validate all sub-builders.

        Calls validate() on each registered sub-builder.
        This is typically called in director's validate() method.

        Returns:
            self for chaining

        Raises:
            ValueError: If any sub-builder fails validation
        """
        for name, builder in self._sub_builders.items():
            try:
                builder.validate()
            except ValueError as e:
                raise ValueError(f"Sub-builder '{name}' validation failed: {e!s}") from e

        return self

    def build_sub_products(self) -> dict[str, Any]:
        """Build all sub-products.

        Calls build() on each sub-builder in registration order.

        Returns:
            Dict mapping sub-builder names to their products

        Raises:
            ValueError: If any sub-builder build fails
        """
        products = {}

        for name, builder in self._sub_builders.items():
            try:
                products[name] = builder.build()
            except Exception as e:
                raise ValueError(f"Building sub-product '{name}' failed: {e!s}") from e

        return products

    @abstractmethod
    def build(self) -> T:
        """Build and coordinate all sub-builders.

        Subclasses must implement this. Standard pattern:
        1. Call self.validate()
        2. Call self.validate_sub_builders()
        3. Call self.build_sub_products()
        4. Assemble final product from sub-products
        5. Set self._product
        6. Return self._product

        Returns:
            T: The composed product

        Raises:
            ValueError: If any validation or build fails
        """
        pass

    def summary(self) -> str:
        """Get summary of director and sub-builders.

        Returns:
            Summary string listing all sub-builders
        """
        sub_builder_summary = ", ".join(
            [
                f"{name}({builder.__class__.__name__})"
                for name, builder in self._sub_builders.items()
            ]
        )
        return (
            f"{self.__class__.__name__} "
            f"(validated={self._validated}, "
            f"sub_builders=[{sub_builder_summary}])"
        )


class ComposableBuilder(FluentBuilder[T]):
    """Optional: Base for builders that can be composed into directors.

    This combines fluent API with ability to be added to a director.
    Not required; use FluentBuilder or DirectorBuilder directly.

    Example:
        >>> class ModelBuilder(ComposableBuilder[Dict]):
        ...     def __init__(self, config, device):
        ...         super().__init__()
        ...         ...
    """

    def compose_into(self, director: DirectorBuilder, name: str) -> "ComposableBuilder[T]":
        """Add this builder to a director.

        Args:
            director: Director to compose into
            name: Name for this builder in director

        Returns:
            self for chaining

        Example:
            >>> model_builder = ModelBuilder(config, device)
            >>> director.add_sub_builder("model", model_builder)
        """
        director.add_sub_builder(name, self)
        return self
