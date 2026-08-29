"""Builder Test Fixtures and Utilities

Phase 0: Testing infrastructure for builder pattern.

Provides:
1. Mock builder implementations for testing
2. Builder test utilities (validation, comparison, etc.)
3. Common test patterns and fixtures
4. Builder registration helpers for tests

This ensures builders can be thoroughly tested in isolation
before integration into the training pipeline.
"""

from typing import Any

import pytest

from mriforge.infrastructure.builders.core.base_builder import (
    BuilderBase,
    DirectorBuilder,
    FluentBuilder,
)
from mriforge.infrastructure.builders.registry import BuilderRegistry

# ============================================================================
# Mock Builders for Testing
# ============================================================================


class MockFluentBuilder(FluentBuilder[dict[str, str]]):
    """Mock fluent builder for testing FluentBuilder functionality."""

    def __init__(self, name: str = "test_builder"):
        """Initialize mock fluent builder.

        Args:
            name: Builder name for identification
        """
        super().__init__()
        self._name = name
        self._values: dict[str, str] = {}
        self._required_fields = {"key1"}

    def with_key(self, key: str, value: str) -> "MockFluentBuilder":
        """Add key-value pair.

        Args:
            key: Key name
            value: Key value

        Returns:
            self for chaining
        """
        self._values[key] = value
        return self

    def with_multiple(self, values: dict[str, str]) -> "MockFluentBuilder":
        """Add multiple key-value pairs.

        Args:
            values: Dict of key-value pairs

        Returns:
            self for chaining
        """
        self._values.update(values)
        return self

    def validate(self) -> "MockFluentBuilder":
        """Validate that required fields are present.

        Returns:
            self for chaining

        Raises:
            ValueError: If required fields missing
        """
        super().validate()

        missing = self._required_fields - set(self._values.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return self

    def build(self) -> dict[str, str]:
        """Build product.

        Returns:
            Dictionary of key-value pairs
        """
        self.validate()
        self._product = dict(self._values)
        return self._product

    def summary(self) -> str:
        """Get summary.

        Returns:
            Summary string
        """
        return (
            f"MockFluentBuilder(name={self._name}, "
            f"values={len(self._values)}, validated={self._validated})"
        )


class MockDirectorBuilder(DirectorBuilder[dict[str, Any]]):
    """Mock director builder for testing DirectorBuilder functionality."""

    def __init__(self, name: str = "test_director"):
        """Initialize mock director builder.

        Args:
            name: Builder name for identification
        """
        super().__init__()
        self._name = name
        self._required_sub_builders = {"component1", "component2"}

    def validate(self) -> "MockDirectorBuilder":
        """Validate director state.

        Returns:
            self for chaining

        Raises:
            ValueError: If required sub-builders missing
        """
        super().validate()

        missing = self._required_sub_builders - set(self._sub_builders.keys())
        if missing:
            raise ValueError(f"Missing required sub-builders: {missing}")

        return self

    def build(self) -> dict[str, Any]:
        """Build product by orchestrating sub-builders.

        Returns:
            Dictionary with all sub-products
        """
        self.validate()
        self.validate_sub_builders()

        products = self.build_sub_products()

        self._product = {"director_name": self._name, "components": products}
        return self._product

    def summary(self) -> str:
        """Get summary.

        Returns:
            Summary string
        """
        return (
            f"MockDirectorBuilder(name={self._name}, "
            f"sub_builders={len(self._sub_builders)}, validated={self._validated})"
        )


# ============================================================================
# Builder Comparison and Validation Utilities
# ============================================================================


class BuilderTestUtils:
    """Utility functions for testing builders."""

    @staticmethod
    def compare_products(product1: Any, product2: Any) -> bool:
        """Compare two builder products for equality.

        Note: This is simplified for testing. Real comparison depends
        on product types (torch tensors, nested dicts, etc).

        Args:
            product1: First product
            product2: Second product

        Returns:
            True if products are equivalent
        """
        if type(product1) != type(product2):
            return False

        if isinstance(product1, dict):
            if set(product1.keys()) != set(product2.keys()):
                return False

            for key in product1:
                if not BuilderTestUtils.compare_products(product1[key], product2[key]):
                    return False

            return True

        if isinstance(product1, (list, tuple)):
            if len(product1) != len(product2):
                return False

            for item1, item2 in zip(product1, product2, strict=False):
                if not BuilderTestUtils.compare_products(item1, item2):
                    return False

            return True

        # For scalars, use simple equality
        return product1 == product2

    @staticmethod
    def assert_builder_state_valid(builder: BuilderBase) -> None:
        """Assert builder is in valid state for building.

        Args:
            builder: Builder instance

        Raises:
            AssertionError: If builder state is invalid
        """
        assert builder is not None, "Builder is None"
        assert hasattr(builder, "validate"), "Builder missing validate method"
        assert hasattr(builder, "build"), "Builder missing build method"
        assert hasattr(builder, "_validated"), "Builder missing _validated attribute"

    @staticmethod
    def assert_product_structure(
        product: Any, expected_keys: set, expected_type: type = dict
    ) -> None:
        """Assert product has expected structure.

        Args:
            product: Built product
            expected_keys: Expected keys/attributes
            expected_type: Expected product type (default: dict)

        Raises:
            AssertionError: If structure is incorrect
        """
        assert isinstance(
            product, expected_type
        ), f"Expected {expected_type}, got {type(product)}"

        if isinstance(product, dict):
            actual_keys = set(product.keys())
            assert (
                actual_keys == expected_keys
            ), f"Expected keys {expected_keys}, got {actual_keys}"

    @staticmethod
    def validate_fluent_chaining(builder: FluentBuilder, methods: list[str]) -> bool:
        """Validate that fluent builder methods return self.

        Args:
            builder: Fluent builder instance
            methods: List of method names to test

        Returns:
            True if all methods support chaining

        Raises:
            AssertionError: If any method fails chaining
        """
        for method_name in methods:
            assert hasattr(
                builder, method_name
            ), f"Builder missing method: {method_name}"

            method = getattr(builder, method_name)
            result = method()

            assert result is builder, (
                f"Method {method_name} did not return self "
                f"(fluent API requirement failed)"
            )

        return True


# ============================================================================
# Registry Test Utilities
# ============================================================================


class RegistryTestUtils:
    """Utility functions for testing builder registry."""

    @staticmethod
    def register_mock_builders(
        registry: BuilderRegistry | None = None,
    ) -> BuilderRegistry:
        """Register mock builders for testing.

        Args:
            registry: Registry instance (uses singleton if None)

        Returns:
            Populated registry
        """
        if registry is None:
            registry = BuilderRegistry.get_instance()

        registry.register_batch(
            {
                "mock_fluent": MockFluentBuilder,
                "mock_director": MockDirectorBuilder,
            }
        )

        return registry

    @staticmethod
    def assert_registry_state(
        registry: BuilderRegistry, expected_count: int, expected_names: set
    ) -> None:
        """Assert registry has expected state.

        Args:
            registry: Registry instance
            expected_count: Expected number of registered builders
            expected_names: Expected builder names

        Raises:
            AssertionError: If registry state is unexpected
        """
        assert (
            registry.count() == expected_count
        ), f"Expected {expected_count} builders, got {registry.count()}"

        actual_names = set(registry.list_builders().keys())
        assert (
            actual_names == expected_names
        ), f"Expected {expected_names}, got {actual_names}"

    @staticmethod
    def assert_builder_creation(
        registry: BuilderRegistry, name: str, *args, **kwargs
    ) -> BuilderBase:
        """Assert builder can be created from registry.

        Args:
            registry: Registry instance
            name: Builder name
            *args: Creation args
            **kwargs: Creation kwargs

        Returns:
            Created builder instance

        Raises:
            AssertionError: If creation fails
        """
        assert registry.exists(name), f"Builder '{name}' not in registry"

        builder = registry.create(name, *args, **kwargs)

        assert builder is not None, f"Failed to create builder '{name}'"
        assert isinstance(builder, BuilderBase), "Created object is not a BuilderBase"

        return builder


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def clean_registry() -> BuilderRegistry:
    """Fixture: Clean registry for isolated testing.

    Yields:
        Fresh, empty BuilderRegistry

    Teardown:
        Resets singleton instance
    """
    # Save original instance
    original = BuilderRegistry._instance

    # Create fresh instance
    BuilderRegistry._instance = None
    registry = BuilderRegistry.get_instance()

    yield registry

    # Restore original
    BuilderRegistry._instance = original


@pytest.fixture
def populated_registry(clean_registry: BuilderRegistry) -> BuilderRegistry:
    """Fixture: Registry populated with mock builders.

    Args:
        clean_registry: Clean registry fixture

    Yields:
        Populated BuilderRegistry
    """
    RegistryTestUtils.register_mock_builders(clean_registry)
    return clean_registry


@pytest.fixture
def mock_fluent_builder() -> MockFluentBuilder:
    """Fixture: Mock fluent builder instance.

    Yields:
        MockFluentBuilder ready for testing
    """
    return MockFluentBuilder()


@pytest.fixture
def mock_director_builder() -> MockDirectorBuilder:
    """Fixture: Mock director builder instance.

    Yields:
        MockDirectorBuilder ready for testing
    """
    return MockDirectorBuilder()


@pytest.fixture
def initialized_fluent_builder() -> MockFluentBuilder:
    """Fixture: Initialized mock fluent builder with required fields.

    Yields:
        MockFluentBuilder ready to build()
    """
    return MockFluentBuilder().with_key("key1", "value1")


@pytest.fixture
def initialized_director_builder(
    mock_fluent_builder: MockFluentBuilder,
) -> MockDirectorBuilder:
    """Fixture: Initialized mock director with sub-builders.

    Args:
        mock_fluent_builder: Mock fluent builder fixture

    Yields:
        MockDirectorBuilder ready to build()
    """
    director = MockDirectorBuilder()
    director.add_sub_builder("component1", MockFluentBuilder("sub1"))
    director.add_sub_builder("component2", MockFluentBuilder("sub2"))
    return director
