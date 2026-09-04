"""Tests for Phase 0 Core Builder Infrastructure

Tests Phase 0 foundation infrastructure:
1. FluentBuilder base class functionality
2. DirectorBuilder orchestration
3. BuilderRegistry operations
4. Builder composition
5. Error handling and validation
"""

import pytest

from spectramr.infrastructure.builders.core import BuilderBase, DirectorBuilder, FluentBuilder
from spectramr.infrastructure.builders.registry import (
    BuilderRegistry,
    create_builder,
    get_builder_registry,
    register_builder,
)
from tests.fixtures.builders import MockDirectorBuilder, MockFluentBuilder


class TestFluentBuilder:
    """Tests for FluentBuilder base class."""

    def test_fluent_builder_inheritance(self):
        """Test FluentBuilder inherits from BuilderBase."""
        builder = MockFluentBuilder()
        assert isinstance(builder, FluentBuilder)
        assert isinstance(builder, BuilderBase)

    def test_fluent_builder_chaining(self):
        """Test fluent API method chaining."""
        builder = MockFluentBuilder()
        result = builder.with_key("k1", "v1").with_key("k2", "v2")
        assert result is builder

    def test_fluent_builder_build(self):
        """Test building simple product."""
        builder = MockFluentBuilder().with_key("key1", "value1")
        product = builder.build()

        assert isinstance(product, dict)
        assert "key1" in product
        assert product["key1"] == "value1"

    def test_fluent_builder_validation_required_fields(self):
        """Test validation of required fields."""
        builder = MockFluentBuilder()
        with pytest.raises(ValueError, match="Missing required fields"):
            builder.build()

        # Should pass with required field
        builder = MockFluentBuilder().with_key("key1", "value1")
        builder.build()

    def test_fluent_builder_state_tracking(self):
        """Test builder state tracking."""
        builder = MockFluentBuilder().with_key("key1", "value1")

        assert not builder.is_validated()
        builder.validate()
        assert builder.is_validated()

    def test_fluent_builder_multiple_products(self):
        """Test building multiple products in sequence."""
        builder1 = MockFluentBuilder("builder1").with_key("key1", "value1")
        product1 = builder1.build()

        builder2 = MockFluentBuilder("builder2").with_key("key1", "value2")
        product2 = builder2.build()

        assert product1["key1"] != product2["key1"]


class TestDirectorBuilder:
    """Tests for DirectorBuilder orchestration."""

    def test_director_builder_inheritance(self):
        """Test DirectorBuilder inherits from BuilderBase."""
        builder = MockDirectorBuilder()
        assert isinstance(builder, DirectorBuilder)
        assert isinstance(builder, BuilderBase)

    def test_director_builder_add_sub_builder(self):
        """Test adding sub-builders."""
        director = MockDirectorBuilder()
        sub1 = MockFluentBuilder("sub1")
        sub2 = MockFluentBuilder("sub2")

        director.add_sub_builder("component1", sub1)
        director.add_sub_builder("component2", sub2)

        assert director.has_sub_builder("component1")
        assert director.has_sub_builder("component2")

    def test_director_builder_get_sub_builder(self):
        """Test retrieving sub-builders."""
        director = MockDirectorBuilder()
        sub = MockFluentBuilder()
        director.add_sub_builder("component", sub)

        retrieved = director.get_sub_builder("component")
        assert retrieved is sub

    def test_director_builder_sub_builder_type_check(self):
        """Test type checking for sub-builders."""
        director = MockDirectorBuilder()

        with pytest.raises(TypeError):
            director.add_sub_builder("invalid", "not a builder")

    def test_director_builder_build(self):
        """Test orchestrating sub-builders."""
        director = MockDirectorBuilder()
        director.add_sub_builder(
            "component1", MockFluentBuilder("sub1").with_key("key1", "value1")
        )
        director.add_sub_builder(
            "component2", MockFluentBuilder("sub2").with_key("key1", "value2")
        )

        product = director.build()

        assert "director_name" in product
        assert "components" in product
        assert "component1" in product["components"]
        assert "component2" in product["components"]

    def test_director_builder_missing_required_sub_builders(self):
        """Test validation of required sub-builders."""
        director = MockDirectorBuilder()

        with pytest.raises(ValueError, match="Missing required sub-builders"):
            director.build()

    def test_director_builder_chaining(self):
        """Test director builder method chaining."""
        director = MockDirectorBuilder()
        result = (
            director.add_sub_builder("component1", MockFluentBuilder("sub1"))
            .add_sub_builder("component2", MockFluentBuilder("sub2"))
            .validate()
        )
        assert result is director


class TestBuilderRegistry:
    """Tests for BuilderRegistry."""

    def test_registry_singleton(self, clean_registry):
        """Test registry is singleton."""
        registry1 = BuilderRegistry.get_instance()
        registry2 = BuilderRegistry.get_instance()
        assert registry1 is registry2

    def test_registry_register_builder(self, clean_registry):
        """Test registering a builder."""
        clean_registry.register("test", MockFluentBuilder)
        assert clean_registry.exists("test")

    def test_registry_get_builder(self, clean_registry):
        """Test retrieving a registered builder class."""
        clean_registry.register("test", MockFluentBuilder)
        builder_class = clean_registry.get("test")
        assert builder_class is MockFluentBuilder

    def test_registry_get_nonexistent_builder(self, clean_registry):
        """Test retrieving non-existent builder raises error."""
        with pytest.raises(KeyError):
            clean_registry.get("nonexistent")

    def test_registry_duplicate_registration(self, clean_registry):
        """Test duplicate registration raises error."""
        clean_registry.register("test", MockFluentBuilder)

        with pytest.raises(ValueError):
            clean_registry.register("test", MockDirectorBuilder)

    def test_registry_duplicate_registration_with_overwrite(self, clean_registry):
        """Test overwriting registration."""
        clean_registry.register("test", MockFluentBuilder)
        clean_registry.register("test", MockDirectorBuilder, overwrite=True)

        assert clean_registry.get("test") is MockDirectorBuilder

    def test_registry_batch_register(self, clean_registry):
        """Test batch registration."""
        clean_registry.register_batch(
            {
                "fluent": MockFluentBuilder,
                "director": MockDirectorBuilder,
            }
        )

        assert clean_registry.exists("fluent")
        assert clean_registry.exists("director")
        assert clean_registry.count() == 2

    def test_registry_unregister(self, clean_registry):
        """Test unregistering a builder."""
        clean_registry.register("test", MockFluentBuilder)
        clean_registry.unregister("test")
        assert not clean_registry.exists("test")

    def test_registry_list_builders(self, populated_registry):
        """Test listing all builders."""
        builders = populated_registry.list_builders()
        assert "mock_fluent" in builders
        assert "mock_director" in builders

    def test_registry_create_builder(self, clean_registry):
        """Test factory creation of builders."""
        clean_registry.register("test", MockFluentBuilder)
        builder = clean_registry.create("test", "test_name")

        assert isinstance(builder, MockFluentBuilder)
        assert builder._name == "test_name"

    def test_registry_create_with_args_kwargs(self, clean_registry):
        """Test builder creation with arguments."""
        clean_registry.register("test", MockFluentBuilder)
        builder = clean_registry.create("test", name="my_builder")

        assert builder._name == "my_builder"

    def test_registry_validate(self, populated_registry):
        """Test registry validation."""
        result = populated_registry.validate()

        assert "valid" in result
        assert "errors" in result
        assert len(result["valid"]) > 0
        assert len(result["errors"]) == 0

    def test_registry_count(self, clean_registry):
        """Test counting registered builders."""
        assert clean_registry.count() == 0

        clean_registry.register("test1", MockFluentBuilder)
        assert clean_registry.count() == 1

        clean_registry.register("test2", MockDirectorBuilder)
        assert clean_registry.count() == 2

    def test_registry_clear(self, populated_registry):
        """Test clearing registry."""
        assert populated_registry.count() > 0
        populated_registry.clear()
        assert populated_registry.count() == 0

    def test_registry_get_or_default(self, clean_registry):
        """Test get with default fallback."""
        result = clean_registry.get_or_default("nonexistent", None)
        assert result is None

        result = clean_registry.get_or_default("nonexistent", MockFluentBuilder)
        assert result is MockFluentBuilder


class TestBuilderComposition:
    """Tests for builder composition patterns."""

    def test_fluent_into_director(self):
        """Test composing fluent builders into director."""
        director = MockDirectorBuilder()
        fluent1 = MockFluentBuilder("f1").with_key("key1", "value1")
        fluent2 = MockFluentBuilder("f2").with_key("key1", "value2")

        director.add_sub_builder("component1", fluent1)
        director.add_sub_builder("component2", fluent2)

        product = director.build()

        assert "components" in product
        assert "component1" in product["components"]
        assert "component2" in product["components"]

    def test_director_in_registry_creation(self, clean_registry):
        """Test creating director with sub-builders from registry."""
        clean_registry.register("fluent", MockFluentBuilder)
        clean_registry.register("director", MockDirectorBuilder)

        fluent1 = clean_registry.create("fluent", "f1")
        fluent1.with_key("key1", "value1")

        fluent2 = clean_registry.create("fluent", "f2")
        fluent2.with_key("key1", "value2")

        director = clean_registry.create("director")
        director.add_sub_builder("component1", fluent1)
        director.add_sub_builder("component2", fluent2)

        product = director.build()
        assert "components" in product


class TestBuilderErrorHandling:
    """Tests for error handling in builders."""

    def test_builder_validation_failure(self):
        """Test builder validation failure."""
        builder = MockFluentBuilder()
        with pytest.raises(ValueError):
            builder.build()

    def test_director_sub_builder_validation_propagation(self):
        """Test validation errors propagate from sub-builders."""
        director = MockDirectorBuilder()
        director.add_sub_builder("component1", MockFluentBuilder("invalid"))
        director.add_sub_builder(
            "component2", MockFluentBuilder().with_key("key1", "v")
        )

        # Should fail because component1 doesn't have required "key1"
        with pytest.raises(ValueError, match="validation failed"):
            director.build()

    def test_registry_type_error_on_invalid_builder_class(self, clean_registry):
        """Test registry rejects non-BuilderBase classes."""
        with pytest.raises(TypeError):
            clean_registry.register("invalid", str)

    def test_registry_creation_error_propagation(self, clean_registry):
        """Test registry creation errors are propagated."""
        clean_registry.register("test", MockFluentBuilder)

        # MockFluentBuilder requires a name argument
        with pytest.raises(TypeError):
            clean_registry.create("test", "too", "many", "args")


class TestGlobalConvenienceFunctions:
    """Tests for global convenience functions."""

    def test_get_builder_registry(self, clean_registry):
        """Test getting global registry."""
        registry = get_builder_registry()
        assert isinstance(registry, BuilderRegistry)

    def test_register_builder_globally(self, clean_registry):
        """Test global registration function."""
        register_builder("test", MockFluentBuilder)
        registry = get_builder_registry()
        assert registry.exists("test")

    def test_create_builder_globally(self, clean_registry):
        """Test global creation function."""
        register_builder("test", MockFluentBuilder)
        builder = create_builder("test", "test_name")

        assert isinstance(builder, MockFluentBuilder)
        assert builder._name == "test_name"


# ============================================================================
# Helper for tests that need ModelBuilder
# ============================================================================


class MockModelBuilder(FluentBuilder[dict]):
    """Simple mock model builder for tests."""

    def __init__(self, config=None, device=None):
        """Initialize."""
        super().__init__()
        self._config = config
        self._device = device
        self._models = {}

    def build(self) -> dict:
        """Build."""
        self.validate()
        return self._models


@pytest.fixture
def setup_mock_model_builder(clean_registry):
    """Setup for tests using ModelBuilder."""
    clean_registry.register("model", MockModelBuilder)
    return clean_registry


# Import this for convenience in tests
ModelBuilder = MockModelBuilder
