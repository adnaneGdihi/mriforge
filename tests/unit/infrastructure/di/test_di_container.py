import threading
from typing import Optional, Protocol, Union, runtime_checkable

import pytest
import torch

from spectramr.infrastructure.di.di_container import (
    DIContainer,
    ServiceResolutionError,
    get_global_container,
    init_container,
    register_service,
    resolve_service,
)


class MockServiceA:
    def __init__(self):
        pass


class MockServiceB:
    def __init__(self, a: MockServiceA):
        self.a = a


class MockServiceC:
    def __init__(self, b: MockServiceB, value: int = 42):
        self.b = b
        self.value = value


class MockServiceCircularA:
    def __init__(self, b: "MockServiceCircularB"):
        self.b = b


class MockServiceCircularB:
    def __init__(self, a: MockServiceCircularA):
        self.a = a


class MockAbstractService:
    def do_something(self):
        raise NotImplementedError


class MockConcreteService(MockAbstractService):
    def do_something(self):
        return "done"


class MockPrimitiveWithoutDefault:
    def __init__(self, a: int):
        self.a = a


def test_register_and_resolve_concrete():
    """Test registering and resolving a concrete instance."""
    container = DIContainer()
    instance = MockServiceA()
    container.register(MockServiceA, implementation=instance)

    resolved = container.resolve(MockServiceA)
    assert resolved is instance


def test_provider_resolution():
    """Test registering and resolving using a provider."""
    container = DIContainer()
    container.register(MockServiceA, provider=lambda: MockServiceA())

    resolved = container.resolve(MockServiceA)
    assert isinstance(resolved, MockServiceA)


def test_singleton_behavior():
    """Test singleton vs non-singleton behavior."""
    container = DIContainer()

    # Singleton provider
    container.register(MockServiceA, provider=lambda: MockServiceA(), singleton=True)
    inst1 = container.resolve(MockServiceA)
    inst2 = container.resolve(MockServiceA)
    assert inst1 is inst2

    # Non-singleton provider
    container.register(
        MockServiceB, provider=lambda: MockServiceB(MockServiceA()), singleton=False
    )
    inst3 = container.resolve(MockServiceB)
    inst4 = container.resolve(MockServiceB)
    assert inst3 is not inst4


def test_autowire_dependencies():
    """Test auto-wiring dependencies recursively."""
    container = DIContainer()

    # We don't even need to register A, it should auto-wire.
    resolved_c = container.resolve(MockServiceC)

    assert isinstance(resolved_c, MockServiceC)
    assert isinstance(resolved_c.b, MockServiceB)
    assert isinstance(resolved_c.b.a, MockServiceA)
    assert resolved_c.value == 42


def test_circular_dependency():
    """Test circular dependency detection."""
    container = DIContainer()

    with pytest.raises(RuntimeError, match="Circular dependency"):
        container.resolve(MockServiceCircularA)


def test_primitive_without_default():
    """Test primitive type without default raises RuntimeError."""
    container = DIContainer()
    with pytest.raises(RuntimeError, match="is a primitive.*with no default value"):
        container.resolve(MockPrimitiveWithoutDefault)


def test_union_optional_resolution():
    """Test Optional and Union resolutions."""
    container = DIContainer()

    container.register(MockServiceA, provider=lambda: MockServiceA())
    resolved = container.resolve(Optional[MockServiceA])
    assert isinstance(resolved, MockServiceA)

    # Resolving Optional for non-existent service shouldn't raise if we catch exceptions?
    # Wait, DIContainer says if len(args) == 1 and it fails, return None
    class UnregisteredMock:
        def __init__(self, x: int):
            self.x = x

    assert container.resolve(Optional[UnregisteredMock]) is None

    with pytest.raises(RuntimeError, match="Cannot resolve Union"):
        container.resolve(Union[MockServiceA, MockServiceB])


class ServiceWithPep604OptionalDep:
    """Constructor with a PEP-604 ``X | None`` dependency and no default.

    Exercises the ``types.UnionType`` arm of ``_instantiate`` — the container
    must resolve the non-None arm rather than skip it.
    """

    def __init__(self, a: "MockServiceA | None"):
        self.a = a


class ServiceWithPep604OptionalDefault:
    """PEP-604 optional dep WITH a default — container must keep the default."""

    def __init__(self, a: "MockServiceA | None" = None):
        self.a = a


def test_pep604_optional_param_without_default_autowires():
    """``X | None`` with no default → container resolves the non-None arm.

    Regression for the audit cleanup that hoisted the per-iteration
    ``import types`` out of the auto-wiring loop: the ``types.UnionType``
    branch must still resolve ``MockServiceA`` instead of leaving it unset.
    """
    container = DIContainer()
    resolved = container.resolve(ServiceWithPep604OptionalDep)
    assert isinstance(resolved.a, MockServiceA)


def test_pep604_optional_param_with_default_keeps_default():
    """``X | None = None`` → container honours the default, does not auto-wire."""
    container = DIContainer()
    resolved = container.resolve(ServiceWithPep604OptionalDefault)
    assert resolved.a is None


def test_abstract_class_resolution():
    """Test resolving an unregistered abstract class."""
    container = DIContainer()

    # Simulate an abstract class manually since abc is not used
    class AbstractManual:
        pass

    AbstractManual.__abstractmethods__ = frozenset(["test"])

    with pytest.raises(ValueError, match="not registered and is abstract"):
        container.resolve(AbstractManual)


def test_global_container_functions():
    """Test global container utility functions."""
    # Ensure init
    container = init_container()
    assert container is get_global_container()

    # Clear for tests
    container.clear()

    register_service(MockServiceA, implementation=MockServiceA())
    resolved = resolve_service(MockServiceA)
    assert isinstance(resolved, MockServiceA)


def test_gradient_flow_audit_dummy_torch():
    """Gradient Flow Audit: Ensure DI container handles resolving torch Modules properly."""
    container = DIContainer()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

    container.register(DummyModel, singleton=True)

    model = container.resolve(DummyModel)
    assert isinstance(model, torch.nn.Module)

    # Check requires grad is intact
    assert model.linear.weight.requires_grad

    x = torch.randn(1, 10)
    out = model.linear(x)
    loss = out.sum()
    loss.backward()

    assert model.linear.weight.grad is not None
    assert not torch.isnan(model.linear.weight.grad).any()


def test_thread_safety_stress():
    """Integration/Edge: Test container thread-safety with concurrent resolutions."""
    container = DIContainer()
    container.register(MockServiceA, provider=lambda: MockServiceA(), singleton=True)

    results = []

    def worker():
        try:
            res = container.resolve(MockServiceA)
            results.append(res)
        except Exception as e:
            results.append(e)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should resolve the same singleton instance
    assert len(results) == 50
    first_inst = results[0]
    for res in results:
        assert res is first_inst


# ---------------------------------------------------------------------------
# Task IV.E — 5 property tests (CLAUDE.md §9: fail-loud, no silent fallbacks)
# Protocols defined here so tests are fully self-contained.
# ---------------------------------------------------------------------------

@runtime_checkable
class IGreeter(Protocol):
    def greet(self) -> str:
        ...


class ConcreteGreeter:
    def greet(self) -> str:
        return "hello"


@runtime_checkable
class IParser(Protocol):
    def parse(self, text: str) -> list[str]:
        ...


class ConcreteParser:
    def parse(self, text: str) -> list[str]:
        return text.split()


class ServiceWithDep:
    """Concrete service that depends on IGreeter — auto-wiring target."""

    def __init__(self, greeter: ConcreteGreeter) -> None:
        self.greeter = greeter


# (a) register_service + resolve_service round-trip via Protocol key
@pytest.mark.unit
def test_protocol_register_resolve_round_trip() -> None:
    """Registering an implementation under a Protocol key and resolving it returns
    the same instance (identity check — not just isinstance)."""
    container = DIContainer()
    impl = ConcreteGreeter()
    container.register(IGreeter, implementation=impl)

    resolved = container.resolve(IGreeter)
    assert resolved is impl, "resolved instance must be the exact registered object"
    assert isinstance(resolved, IGreeter)


# (b) singleton returns same instance; transient returns new instance
@pytest.mark.unit
def test_singleton_same_instance_transient_new_instance() -> None:
    """Singleton scope → same object on repeated resolves.
    Transient scope → distinct objects on every resolve."""
    container = DIContainer()

    container.register(IGreeter, provider=lambda: ConcreteGreeter(), singleton=True)
    s1 = container.resolve(IGreeter)
    s2 = container.resolve(IGreeter)
    assert s1 is s2, "singleton must return the same object"

    container.register(IParser, provider=lambda: ConcreteParser(), singleton=False)
    t1 = container.resolve(IParser)
    t2 = container.resolve(IParser)
    assert t1 is not t2, "transient must return a fresh object on each resolve"


# (c) resolving an unregistered Protocol raises — no silent None (CLAUDE.md pitfall #9)
@pytest.mark.unit
def test_unregistered_protocol_raises_not_silent_none() -> None:
    """Resolving a Protocol that was never registered must raise, not return None.

    CLAUDE.md pitfall #9: 'If a model receives an attention_type it doesn't know,
    raise instead of falling back.'  The same rule applies to the DI container —
    a missing service must be an explicit error, not a silent None.
    """

    @runtime_checkable
    class IUnregistered(Protocol):
        def do(self) -> None:
            ...

    container = DIContainer()
    # IUnregistered is a Protocol (has __abstractmethods__ or __protocol_attrs__).
    # The container path for an unregistered, non-concrete type must raise.
    # Protocol classes appear abstract to inspect.isabstract() only when they
    # carry abstractmethods; set the frozenset explicitly to ensure that path.
    IUnregistered.__abstractmethods__ = frozenset({"do"})

    with pytest.raises((ValueError, RuntimeError)):
        container.resolve(IUnregistered)


# (d) init_container is idempotent — multi-call safe
@pytest.mark.unit
def test_init_container_idempotent() -> None:
    """Calling init_container() multiple times must return the same container object.

    The global container singleton must not be re-created on subsequent calls,
    which would silently drop all registered services.
    """
    import spectramr.infrastructure.di.di_container as di_module

    # Reset to known-clean state for this test only
    original = di_module._global_container
    di_module._global_container = None
    try:
        first = init_container()
        second = init_container()
        third = init_container()
        assert first is second, "second call must return the same container"
        assert first is third, "third call must return the same container"
    finally:
        di_module._global_container = original


# (e) type-hint-driven auto-wiring resolves a nested dependency
@pytest.mark.unit
def test_auto_wiring_resolves_nested_dependency() -> None:
    """Container auto-wires ServiceWithDep by inspecting its constructor type hints.

    ServiceWithDep.__init__(self, greeter: ConcreteGreeter) — the container must
    recursively resolve ConcreteGreeter without any explicit registration.
    """
    container = DIContainer()

    service = container.resolve(ServiceWithDep)

    assert isinstance(service, ServiceWithDep)
    assert isinstance(service.greeter, ConcreteGreeter)
    assert service.greeter.greet() == "hello"


@pytest.mark.unit
def test_singleton_provider_constructs_exactly_once() -> None:
    """A ``singleton=True`` provider's factory must run exactly once across
    repeated resolves. The old code always invoked the provider/constructor and
    only de-duplicated at cache-insert time, so the singleton's whole dependency
    subtree was rebuilt-and-discarded on every resolve (wasted construction +
    leaked resources for torch-module singletons)."""
    calls = {"n": 0}

    class Service:
        pass

    def provider() -> Service:
        calls["n"] += 1
        return Service()

    container = DIContainer()
    container.register(Service, provider=provider, singleton=True)

    a = container.resolve(Service)
    b = container.resolve(Service)
    c = container.resolve(Service)

    assert a is b is c
    assert calls["n"] == 1, f"provider ran {calls['n']}× for a singleton"


@pytest.mark.unit
def test_auto_wired_singleton_subtree_constructs_once() -> None:
    """An auto-wired concrete singleton must not re-run its dependency subtree
    on every resolve."""
    counter = {"dep": 0}

    class Dep:
        def __init__(self) -> None:
            counter["dep"] += 1

    class Root:
        def __init__(self, dep: Dep) -> None:
            self.dep = dep

    container = DIContainer()
    r1 = container.resolve(Root)
    r2 = container.resolve(Root)

    assert r1 is r2
    assert counter["dep"] == 1, f"dependency ran {counter['dep']}× for a singleton"
@pytest.mark.unit
def test_service_resolution_error_matches_legacy_types() -> None:
    """``ServiceResolutionError`` satisfies both legacy raise-site contracts.

    The container historically raised a mix of ``ValueError`` and
    ``RuntimeError`` for wiring misses; callers match on either. The dedicated
    error type must remain catchable as both.
    """
    assert issubclass(ServiceResolutionError, ValueError)
    assert issubclass(ServiceResolutionError, RuntimeError)


@pytest.mark.unit
def test_optional_propagates_provider_failure() -> None:
    """A provider that *fails* must not be silently absorbed into ``None``.

    Regression for the fail-loud fix (CLAUDE.md pitfall #9): the Optional arm
    previously did ``except Exception: return None``, converting real
    constructor failures into a silent missing dependency. Only wiring misses
    (``ServiceResolutionError``) may degrade to ``None``.
    """

    class Exploding:
        pass

    def bad_provider() -> Exploding:
        raise OSError("backing store unavailable")

    container = DIContainer()
    container.register(Exploding, provider=bad_provider)

    with pytest.raises(OSError, match="backing store unavailable"):
        container.resolve(Optional[Exploding])


@pytest.mark.unit
def test_optional_propagates_circular_dependency() -> None:
    """A circular dependency inside an Optional resolve is a failure, not
    absence — it must propagate instead of degrading to ``None``."""
    container = DIContainer()

    with pytest.raises(RuntimeError, match="Circular dependency"):
        container.resolve(Optional[MockServiceCircularA])


@pytest.mark.unit
def test_optional_wiring_miss_still_returns_none() -> None:
    """The legitimate Optional degrade path survives the fail-loud fix: a
    genuine wiring miss (primitive without default) still yields ``None``."""

    class Unwirable:
        def __init__(self, x: int):
            self.x = x

    container = DIContainer()
    assert container.resolve(Optional[Unwirable]) is None


@pytest.mark.unit
def test_none_returning_singleton_provider_runs_once() -> None:
    """A singleton provider that legitimately returns ``None`` must run once.

    Regression for the ``_MISSING`` sentinel: the cache-hit test was
    ``if cached is not None``, so a cached ``None`` looked like a miss and the
    provider re-ran on every resolve.
    """
    calls = {"n": 0}

    class MaybeAbsent:
        pass

    def provider() -> "MaybeAbsent | None":
        calls["n"] += 1
        return None

    container = DIContainer()
    container.register(MaybeAbsent, provider=provider, singleton=True)

    assert container.resolve(MaybeAbsent) is None
    assert container.resolve(MaybeAbsent) is None
    assert calls["n"] == 1, f"provider ran {calls['n']}x for a cached-None singleton"


@pytest.mark.unit
def test_concurrent_singleton_constructs_exactly_once() -> None:
    """Racing threads must not each build the singleton's dependency subtree.

    Regression for the construction race: the lock used to be dropped between
    the cache check and the insert, so concurrent first-resolves all
    constructed and ``_cache_singleton`` silently discarded the extras (a
    resource leak for torch-module singletons). ``test_thread_safety_stress``
    only asserts instance identity, which the old code also satisfied — this
    test pins the construction *count*.
    """
    calls = {"n": 0}
    release = threading.Event()

    class Slow:
        pass

    def slow_provider() -> Slow:
        calls["n"] += 1
        release.wait(timeout=5)  # hold construction open so threads pile up
        return Slow()

    container = DIContainer()
    container.register(Slow, provider=slow_provider, singleton=True)

    results: list[Slow] = []

    def worker() -> None:
        results.append(container.resolve(Slow))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    release.set()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(r is results[0] for r in results)
    assert calls["n"] == 1, f"singleton constructed {calls['n']}x under contention"


def test_typing_localns_cached_at_module_load():
    """The auto-wire typing namespace is computed once, not per-resolve."""
    from spectramr.infrastructure.di import di_container as dic

    assert isinstance(dic._TYPING_LOCALNS, dict)
    # Sanity: it carries the public typing names used to resolve string
    # annotations like ``Optional[...]`` / ``Union[...]``.
    for name in ("Optional", "Union", "List", "Dict"):
        assert name in dic._TYPING_LOCALNS
    assert all(not n.startswith("_") for n in dic._TYPING_LOCALNS)


def test_autowire_still_resolves_with_cached_typing_namespace():
    """Regression: caching the typing namespace must not break auto-wiring."""
    container = DIContainer()
    c = container.resolve(MockServiceC)
    assert isinstance(c, MockServiceC)
    assert isinstance(c.b, MockServiceB)
    assert isinstance(c.b.a, MockServiceA)
    assert c.value == 42
