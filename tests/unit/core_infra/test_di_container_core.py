"""Unit tests for the thread-safe Dependency Injection container."""

import pytest

from mriforge.infrastructure.di.di_container import (
    DIContainer,
    get_global_container,
    init_container,
    register_service,
    resolve_service,
)


class BaseLogger:
    pass


class ConcreteLogger(BaseLogger):
    pass


class DummyService:
    def __init__(self, logger: BaseLogger):
        self.logger = logger


class OptionalDepService:
    def __init__(self, config: dict | None = None):
        self.config = config


class CircularA:
    def __init__(self, b: "CircularB"):
        self.b = b


class CircularB:
    def __init__(self, a: "CircularA"):
        self.a = a


class PrimitiveDepService:
    def __init__(self, count: int):
        self.count = count


class TestDIContainer:
    """Tests for DIContainer instance logic."""

    @pytest.fixture
    def container(self):
        """Provide a fresh DIContainer for each test."""
        return DIContainer()

    def test_register_and_resolve_concrete_implementation(self, container):
        """Test resolving an explicitly registered implementation."""
        logger_impl = ConcreteLogger()
        container.register(BaseLogger, implementation=logger_impl)

        resolved = container.resolve(BaseLogger)
        assert resolved is logger_impl

    def test_register_and_resolve_provider(self, container):
        """Test resolving via a factory provider."""
        container.register(BaseLogger, provider=lambda: ConcreteLogger())

        resolved = container.resolve(BaseLogger)
        assert isinstance(resolved, ConcreteLogger)

    def test_singleton_persistence(self, container):
        """Test that singletons return the same instance across resolutions."""
        container.register(BaseLogger, implementation=ConcreteLogger())

        # Resolve twice
        res1 = container.resolve(BaseLogger)
        res2 = container.resolve(BaseLogger)

        assert res1 is res2

    def test_transient_persistence(self, container):
        """Test that transient services return a new instance each time."""
        container.register(
            BaseLogger, provider=lambda: ConcreteLogger(), singleton=False
        )

        res1 = container.resolve(BaseLogger)
        res2 = container.resolve(BaseLogger)

        assert res1 is not res2

    def test_autowire_concrete_dependencies(self, container):
        """Test automatic injection of constructor dependencies."""
        container.register(BaseLogger, implementation=ConcreteLogger())

        # Auto-wires DummyService by inspecting its constructor
        service = container.resolve(DummyService)

        assert isinstance(service, DummyService)
        assert isinstance(service.logger, ConcreteLogger)

    def test_optional_dependency_resolution(self, container):
        """Test resolving Optional[T] handles un-registered types gracefully."""
        # Not registered dict, but wrapped in Optional
        service = container.resolve(OptionalDepService)

        assert isinstance(service, OptionalDepService)
        assert service.config is None

    def test_circular_dependency_detection(self, container):
        """Test that circular dependencies are caught explicitly."""
        # A depends on B, B depends on A
        with pytest.raises(RuntimeError, match="Circular dependency"):
            container.resolve(CircularA)

    def test_primitive_dependency_rejection(self, container):
        """Test that attempting to auto-wire primitives fails-fast."""
        with pytest.raises(RuntimeError, match="Cannot auto-wire"):
            container.resolve(PrimitiveDepService)

    def test_prevent_duplicate_registration(self, container):
        """Test that registering the same type twice throws."""
        container.register(BaseLogger, implementation=ConcreteLogger())

        with pytest.raises(ValueError, match="already registered"):
            container.register(BaseLogger, implementation=ConcreteLogger())


class TestGlobalDIContainer:
    """Tests for the global container state module variables."""

    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Clear the global container state before and after each test."""
        # Reset the module level variable
        import mriforge.infrastructure.di.di_container as di_module

        di_module._global_container = None
        yield
        di_module._global_container = None

    def test_uninitialized_global_container(self):
        """Test getting the container before init raises an error."""
        with pytest.raises(RuntimeError, match="Container not initialized"):
            get_global_container()

    def test_initialized_global_container(self):
        """Test module-level container access."""
        container = init_container()
        assert get_global_container() is container

    def test_global_service_registration_and_resolution(self):
        """Test global helpers."""
        init_container()

        # Test the module-level helper functions
        register_service(BaseLogger, implementation=ConcreteLogger())
        resolved = resolve_service(BaseLogger)

        assert isinstance(resolved, ConcreteLogger)
