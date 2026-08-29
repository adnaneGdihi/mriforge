import inspect
import logging
import threading
import types
import typing
from collections.abc import Callable
from typing import (
    Any,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Primitive types that don't need dependency injection
_SKIP_TYPES = (str, int, float, bool, type(None), Any)

# ``get_type_hints`` needs the ``typing`` namespace as ``localns`` so that
# string annotations referencing ``Optional``/``Union``/etc. resolve.
# Computed ONCE at module load — previously this dict comprehension over
# ``dir(typing)`` was rebuilt on every ``_instantiate`` (auto-wire) call,
# which is pure per-resolve waste. The typing module is immutable in
# practice, so a module-level snapshot is safe.
_TYPING_LOCALNS = {n: getattr(typing, n) for n in dir(typing) if not n.startswith("_")}

# Sentinel distinguishing "not cached" from "cached None" in the singleton
# cache. A provider may legitimately return ``None``; ``dict.get(key)`` would
# treat that as a miss and re-run the provider on every resolve.
_MISSING = object()


class ServiceResolutionError(ValueError, RuntimeError):
    """The container cannot *wire* a service.

    Raised for wiring misses only: unregistered abstract type, missing type
    hint, primitive parameter without a default, or a union with no resolvable
    arm. Exceptions raised *inside* a provider or constructor are real
    failures and are never wrapped in this type.

    The distinction is what makes ``Optional[T]`` resolution honest: a wiring
    miss on ``T`` degrades to ``None`` (the dependency is genuinely optional),
    while a provider/constructor error or a circular dependency propagates —
    no silent fallbacks (CLAUDE.md pitfall #9).

    Subclasses both ``ValueError`` and ``RuntimeError`` because the historical
    raise sites used a mix of the two and existing callers match on either.
    """


class DIContainer:
    """Minimal dependency injection container with automatic wiring.

    [FORENSIC FIX]: Thread-safe with per-thread circular dependency detection.

    .. mermaid::

        sequenceDiagram
            participant Client
            participant Container
            participant Registry
            participant Resolver

            Client->>Container: resolve(ServiceType)
            Container->>Resolver: _do_resolve(ServiceType)
            Resolver->>Registry: get(ServiceType)

            alt Is Registered Provider
                Registry-->>Resolver: Provider
                Resolver->>Resolver: Invoke Provider
            else Is Concrete Type
                Resolver->>Resolver: Inspect __init__
                Resolver->>Container: Recursive resolve(Dependencies)
            end

            Resolver-->>Container: Instance
            Container-->>Client: Instance
    """

    def __init__(self) -> None:
        """__init__."""
        self._registrations: dict[Any, Any] = {}
        self._singletons: dict[Any, Any] = {}
        # [FORENSIC FIX] Use thread-local storage for resolution stack
        # Prevents false circular dependency errors in concurrent environments
        self._local = threading.local()
        self._lock = threading.RLock()

    @property
    def _resolving(self) -> set[Any]:
        """Get thread-local resolving set."""
        if not hasattr(self._local, "resolving"):
            self._local.resolving = set()
        return self._local.resolving

    def register(
        self,
        service_type: Any,
        implementation: Any | None = None,
        provider: Callable[[], T] | None = None,
        singleton: bool = True,
    ) -> None:
        """Register a service: implementation, provider, or auto-wire by type."""
        with self._lock:
            if service_type in self._registrations:
                raise ValueError(f"Service {service_type} already registered")
            self._registrations[service_type] = {
                "impl": implementation,
                "provider": provider,
                "singleton": singleton,
            }

    def resolve(self, service_type: Any) -> Any:
        """Resolve a service with all dependencies."""
        # Handle Optional[T] = Union[T, None]
        origin = get_origin(service_type)
        if origin is Union:
            args = [arg for arg in get_args(service_type) if arg is not type(None)]
            if len(args) == 1:
                try:
                    return self.resolve(args[0])
                except ServiceResolutionError:
                    # Wiring miss on the optional dependency → genuinely
                    # absent. Provider/constructor errors and circular
                    # dependencies are failures, not absence — they propagate.
                    return None
            raise ServiceResolutionError(f"Cannot resolve Union {service_type}")

        service_name = getattr(service_type, "__name__", repr(service_type))
        if service_type in self._resolving:
            raise RuntimeError(f"Circular dependency: {service_name}")

        self._resolving.add(service_type)
        try:
            return self._do_resolve(service_type)
        finally:
            self._resolving.discard(service_type)

    def _do_resolve(self, service_type: Any) -> Any:
        """Internal resolution logic."""
        # The whole check→construct→insert sequence runs under the RLock: a
        # second thread that misses the singleton cache blocks until the first
        # finishes constructing, then hits the cache. Previously the lock was
        # dropped during construction, so two racing threads could both build
        # the full dependency subtree and ``_cache_singleton`` silently
        # discarded the second instance — wasted work and a resource leak for
        # torch-module singletons. The RLock is reentrant, so the recursive
        # ``resolve()`` calls made while auto-wiring dependencies on the same
        # thread cannot deadlock.
        with self._lock:
            cached = self._singletons.get(service_type, _MISSING)
            if cached is not _MISSING:
                return cached
            return self._construct(service_type)

    def _construct(self, service_type: Any) -> Any:
        """Build a service after a singleton-cache miss (caller holds the lock)."""
        registration = self._registrations.get(service_type)

        # Provider takes precedence
        if registration and registration["provider"]:
            return self._cache_singleton(
                service_type,
                registration["provider"](),
                registration["singleton"],
            )

        # Explicit implementation
        if registration and registration["impl"] is not None:
            impl = registration["impl"]
            if isinstance(impl, type):
                return self._cache_singleton(
                    service_type,
                    self._instantiate(impl),
                    registration["singleton"],
                )
            # Already an instance
            return self._cache_singleton(
                service_type,
                impl,
                registration["singleton"],
            )

        # Auto-wire concrete types
        if isinstance(service_type, type):
            # Abstract types need explicit registration (fail-fast)
            if inspect.isabstract(service_type):
                raise ServiceResolutionError(
                    f"Service {service_type.__name__} not registered and is abstract. "
                    f"Must be explicitly registered via register() before resolution."
                )

            # Concrete type: auto-wire
            return self._cache_singleton(
                service_type,
                self._instantiate(service_type),
                True,
            )

        raise ServiceResolutionError(f"Cannot resolve {service_type}")

    def _instantiate(self, concrete_type: type[T]) -> T:
        """Auto-wire a concrete type by inspecting constructor."""
        try:
            sig = inspect.signature(concrete_type.__init__)
            module = inspect.getmodule(concrete_type)
            globalns = getattr(module, "__dict__", {})

            # Provide typing module context (cached at module load — see
            # ``_TYPING_LOCALNS``).
            hints = get_type_hints(
                concrete_type.__init__, globalns=globalns, localns=_TYPING_LOCALNS
            )
        except (TypeError, AttributeError, NameError):
            return concrete_type()

        # Resolve dependencies
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "__pydantic_self__"):
                continue

            # ``*args`` / ``**kwargs`` carry no single type to wire and are
            # optional by nature. They must be skipped, not type-resolved —
            # otherwise any class without an explicit ``__init__`` (whose
            # constructor is the inherited ``object.__init__(self, *args,
            # **kwargs)``) fails auto-wiring with "No type hint for args".
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            param_type = hints.get(param_name)
            if not param_type:
                if param.default is inspect.Parameter.empty:
                    raise ServiceResolutionError(
                        f"No type hint for {param_name} in {concrete_type.__name__}"
                    )
                continue

            # [FORENSIC FIX] Handle primitives gracefully
            # If a primitive has no default value, we cannot auto-wire it.
            if param_type in _SKIP_TYPES:
                if param.default is inspect.Parameter.empty:
                    raise ServiceResolutionError(
                        f"Cannot auto-wire {concrete_type.__name__}: argument '{param_name}' "
                        f"is a primitive ({param_type.__name__ if hasattr(param_type, '__name__') else param_type}) "
                        f"with no default value. Register with explicit instance or use provider."
                    )
                continue  # Has default, use it

            # Handle Optional[T] / T | None / Union[T, None]: when the parameter
            # has a default value, prefer it over forcing the container to
            # resolve a Union type it has no implementation for.
            origin = get_origin(param_type)
            if origin is Union or origin is types.UnionType:
                if param.default is not inspect.Parameter.empty:
                    continue  # use default
                # Else: try to resolve a non-None arm.
                args = [a for a in get_args(param_type) if a is not type(None)]
                resolved = False
                for arm in args:
                    try:
                        kwargs[param_name] = self.resolve(arm)
                        resolved = True
                        break
                    except ServiceResolutionError:
                        # Wiring miss on this arm — try the next one. Real
                        # construction failures propagate (pitfall #9).
                        continue
                if not resolved:
                    raise ServiceResolutionError(
                        f"Cannot auto-wire {concrete_type.__name__}: union-typed "
                        f"argument '{param_name}' ({param_type}) has no default and "
                        f"no arm could be resolved by the container."
                    )
                continue

            if hasattr(param_type, "__module__") and "typing" in str(param_type.__module__):
                continue

            try:
                kwargs[param_name] = self.resolve(param_type)
            except Exception as e:
                logger.error(
                    "Failed to resolve %s of type %s for %s: %s",
                    param_name,
                    param_type,
                    concrete_type.__name__,
                    e,
                )
                raise

        return concrete_type(**kwargs)

    def _cache_singleton(self, service_type: Any, instance: T, is_singleton: bool) -> T:
        """Cache singleton instances."""
        if is_singleton:
            with self._lock:
                if service_type not in self._singletons:
                    self._singletons[service_type] = instance
                return self._singletons[service_type]
        return instance

    def has_service(self, service_type: Any) -> bool:
        """Check if a service is registered."""
        with self._lock:
            return service_type in self._registrations

    def clear(self) -> None:
        """Clear registrations and singletons."""
        with self._lock:
            self._registrations.clear()
            self._singletons.clear()

    def shutdown(self) -> None:
        """Shutdown container."""
        self.clear()
        logger.info("DIContainer shutdown")


# Global container instance
_global_container: DIContainer | None = None
_global_lock = threading.Lock()


def init_container() -> DIContainer:
    """Initialize the global DI container."""
    global _global_container
    with _global_lock:
        if _global_container is None:
            _global_container = DIContainer()
        return _global_container


def get_global_container() -> DIContainer:
    """Get the global DI container, fail if not initialized."""
    if _global_container is None:
        raise RuntimeError("Container not initialized. Call init_container() first.")
    return _global_container


def register_service(
    service_type: Any,
    implementation: Any | None = None,
    provider: Callable[[], T] | None = None,
    singleton: bool = True,
) -> None:
    """Register a service in the global container."""
    get_global_container().register(service_type, implementation, provider, singleton)


def resolve_service(service_type: Any) -> Any:
    """Resolve a service from the global container."""
    return get_global_container().resolve(service_type)
