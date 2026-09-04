"""Canonical builder context + a non-breaking compat adapter (cached-cascade WS-D).

The recon census found builder ``__init__`` signatures split into three
competing shapes: ``(config)`` [11 builders], ``(config, device)`` [12, with
``device`` typed inconsistently as ``Any`` / ``torch.device`` / ``str="cuda"``],
and ``(config, <domain-object>)`` [4 -- ``models`` / ``params`` / ``dataset`` /
``optimizer``]. ``BuilderContext`` carries ``config`` + ``device`` + the optional
already-built dependencies, collapsing all three onto one frozen shape (the
*with-what* of a build; distinct from ``ResolvedExperimentContext``, the *what*).

This is the **freeze**: the dataclass + the :func:`accepts_builder_context`
adapter are landed so component workstreams (Phase B) migrate builders to
``def __init__(self, ctx: BuilderContext)`` while their legacy ``(config, device)``
/ ``(config, models)`` call sites keep working. Silent by default (the repo
promotes ``spectramr.*`` ``DeprecationWarning`` to a test error); ``warn=True`` is
the flagged removal-pass escalation.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BuilderContext", "accepts_builder_context"]


@dataclass(frozen=True)
class BuilderContext:
    """Everything a component builder needs to build, in one frozen object.

    ``config`` is the ``TrainingSettings`` SSOT (typed ``Any`` to avoid an
    infrastructure->config hard import). The remaining fields are the optional
    already-built dependencies the Group-C builders take positionally today.
    """

    config: Any
    device: Any = None  # torch.device | str | None
    models: Mapping[str, Any] | None = None
    dataset: Any = None
    optimizer: Any = None
    params: Any = None
    #: anything paradigm-specific that does not warrant a named field
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(cls, config: Any, device: Any = None, **deps: Any) -> BuilderContext:
        """Build a context from the legacy ``(config, device, **deps)`` shape."""
        known = {"models", "dataset", "optimizer", "params"}
        named = {k: deps.pop(k) for k in list(deps) if k in known}
        return cls(config=config, device=device, extras=deps, **named)


def accepts_builder_context(
    func: Callable[..., Any] | None = None, *, warn: bool = False
) -> Callable[..., Any]:
    """Decorator for a builder ``__init__`` to accept either shape.

    Wrap ``def __init__(self, ctx: BuilderContext)``; the wrapper also accepts
    the legacy ``__init__(self, config, device=..., models=..., ...)`` form and
    normalizes it into a ``BuilderContext`` before calling the new body. Legacy
    calls are silent unless ``warn=True``.
    """

    def _decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if args and isinstance(args[0], BuilderContext) and not kwargs:
                return fn(self, args[0])
            if "ctx" in kwargs and isinstance(kwargs["ctx"], BuilderContext):
                return fn(self, kwargs["ctx"])
            # Legacy path: (config, device=?, **deps) -> BuilderContext
            if warn:
                warnings.warn(
                    f"{fn.__qualname__} called with legacy (config, device, ...) "
                    "args; pass a BuilderContext instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            ctx = BuilderContext.from_args(*args, **kwargs)
            return fn(self, ctx)

        # Record the wrapped signature so the fitness check can tell a migrated
        # builder (takes BuilderContext) from a legacy one. (setattr: function
        # objects carry arbitrary attrs; direct assignment trips the type-checker)
        setattr(wrapper, "__accepts_builder_context__", True)  # noqa: B010
        setattr(wrapper, "__wrapped_signature__", inspect.signature(fn))  # noqa: B010
        return wrapper

    if func is not None:
        return _decorate(func)
    return _decorate
