"""Optimizer registry + the canonical scheduler-registry surface (WS-D).

Recon found optimizer selection split across two places -- a static ``if/elif``
in ``optimization_builder.create_single_optimizer`` (keys ``adam``/``adamw``/
``sgd``/``rmsprop``) and a reflective ``getattr(torch.optim, ...)`` in the leaf
``OptimizerBuilder``. This module introduces the genuinely-new
:class:`OptimizerRegistry` (``@register_optimizer`` / :func:`create_optimizer`)
so the Tier-1 WS-D conversion can route both through one explicit, no-silent-
fallback dispatch table.

Schedulers already HAVE a registry -- ``SCHEDULER_REGISTRY`` + ``@register_scheduler``
in :mod:`mriforge.infrastructure.training.scheduler_system`. Per the repo's
one-canonical-home rule (pitfall #12) we do **not** create a competing
``SchedulerRegistry``; instead we re-export the existing one here so callers have
a single import surface, and add :func:`create_scheduler` (which **raises** on an
unknown name, tightening the silent ``logger.warning + continue`` the current
builder does -- a pitfall-#10 fix the Tier-1 conversion will adopt).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import torch
from torch import optim

# Re-export the EXISTING scheduler registry as the canonical surface (SSOT lives
# in scheduler_system.py -- do not duplicate the storage here).
from mriforge.infrastructure.training.scheduler_system import (
    SCHEDULER_REGISTRY,
    register_scheduler,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEDULER_REGISTRY",
    "OptimizerRegistry",
    "accepted_optimizer_kwargs",
    "create_optimizer",
    "create_scheduler",
    "register_optimizer",
    "register_scheduler",
]


class OptimizerRegistry:
    """Explicit name -> optimizer-class dispatch table (no silent fallback).

    Mirrors ``LossRegistry``: a class-level dict plus a ``register`` decorator.
    The standard ``torch.optim`` optimizers are pre-registered; custom ones opt
    in via :func:`register_optimizer`.
    """

    _registry: ClassVar[dict[str, type[optim.Optimizer]]] = {}
    #: Explicitly-declared accepted kwargs, for classes whose ``__init__``
    #: signature cannot be introspected usefully (see :func:`register_optimizer`).
    _accepted: ClassVar[dict[str, frozenset[str]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        aliases: Iterable[str] = (),
        params: Iterable[str] | None = None,
    ) -> Callable[[type[optim.Optimizer]], type[optim.Optimizer]]:
        def decorator(opt_cls: type[optim.Optimizer]) -> type[optim.Optimizer]:
            for key in (name, *aliases):
                lk = key.lower()
                existing = cls._registry.get(lk)
                if existing is not None and existing is not opt_cls:
                    raise ValueError(
                        f"Optimizer '{key}' already registered to "
                        f"{existing.__name__}; refusing to overwrite with "
                        f"{opt_cls.__name__}."
                    )
                cls._registry[lk] = opt_cls
                if params is not None:
                    cls._accepted[lk] = frozenset(params)
            return opt_cls

        return decorator

    @classmethod
    def register_class(
        cls,
        name: str,
        opt_cls: type[optim.Optimizer],
        *,
        aliases: Iterable[str] = (),
        params: Iterable[str] | None = None,
    ) -> None:
        """Imperatively register an existing optimizer class (e.g. torch's)."""
        cls.register(name, aliases=aliases, params=params)(opt_cls)

    @classmethod
    def accepted_kwargs(cls, name: str) -> frozenset[str]:
        """Constructor kwargs ``name`` accepts, besides ``params``.

        Signature-derived by default. A ``**kwargs`` in the signature would make
        that set "everything", turning the caller's no-silent-drop guarantee into
        a rubber stamp, so such a class MUST declare its set explicitly at
        registration time; failing to is an error here rather than a permissive
        default.
        """
        lk = name.lower()
        explicit = cls._accepted.get(lk)
        if explicit is not None:
            return explicit

        opt_cls = cls._registry.get(lk)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer: {name!r}. Available: {sorted(cls._registry)}")

        signature = inspect.signature(opt_cls.__init__)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            raise ValueError(
                f"Optimizer {name!r} ({opt_cls.__name__}) takes **kwargs, so its "
                "accepted set cannot be introspected. Declare it explicitly: "
                "@register_optimizer(name, params=frozenset({...}))."
            )
        return frozenset(signature.parameters) - {"self", "params"}

    @classmethod
    def create(cls, name: str, params: Any, **kwargs: Any) -> optim.Optimizer:
        lk = name.lower()
        opt_cls = cls._registry.get(lk)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer: {name!r}. Available: {sorted(cls._registry)}")
        return opt_cls(params, **kwargs)

    @classmethod
    def get(cls, name: str) -> type[optim.Optimizer] | None:
        """Look up a registered optimizer class without instantiating it.

        Needed by callers that must validate declared kwargs against the
        optimizer's signature before building (see
        ``OptimizationBuilder._unconsumed_optimizer_kwargs``). Returns ``None``
        for an unregistered name so the caller can decide whether that is fatal;
        :meth:`create` remains the raising path.
        """
        return cls._registry.get(name.lower())

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._registry)


def register_optimizer(
    name: str, *, aliases: Iterable[str] = (), params: Iterable[str] | None = None
) -> Callable[[type[optim.Optimizer]], type[optim.Optimizer]]:
    """Decorator to register a custom optimizer class (mirrors register_loss).

    ``params`` declares the accepted constructor kwargs. Required when the class
    takes ``**kwargs`` (e.g. a wrapper optimizer forwarding to a base), because
    signature introspection would otherwise report "accepts everything" and the
    resolver's raise-on-unaccepted-knob guarantee would silently evaporate.
    """
    return OptimizerRegistry.register(name, aliases=aliases, params=params)


def accepted_optimizer_kwargs(name: str) -> frozenset[str]:
    """Constructor kwargs ``name`` accepts (besides ``params``)."""
    return OptimizerRegistry.accepted_kwargs(name)


def create_optimizer(name: str, params: Any, **kwargs: Any) -> optim.Optimizer:
    """Create an optimizer by registered name (raises on unknown)."""
    return OptimizerRegistry.create(name, params, **kwargs)


def create_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    scheduler_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Create an LR scheduler via the canonical ``SCHEDULER_REGISTRY``.

    Raises ``ValueError`` on an unknown name -- explicitly tighter than the
    builder's current silent ``logger.warning + continue`` (pitfall #10).
    """
    key = "cosine_annealing" if name == "cosine" else name
    factory = SCHEDULER_REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"Unknown scheduler: {name!r}. Available: {sorted(SCHEDULER_REGISTRY)}")
    return factory(optimizer, scheduler_kwargs or {})


# Pre-register the torch.optim tier of ``OptimizerType``.
#
# This used to stop at seven names while the live leaf builder resolved ANY
# public attribute of ``torch.optim`` reflectively — so the set the registry
# knew and the set that actually worked disagreed, and
# ``_unconsumed_optimizer_kwargs`` skipped validation entirely for every name in
# the gap (adagrad, lbfgs, adafactor, ...). Registering the full tier closes
# that: one table, and it is the one that dispatches.
#
# ``lbfgs`` and ``sparseadam`` are here deliberately. They accept no
# ``weight_decay``, which is precisely why the old unconditional forward
# TypeError'd on them; naming them makes the drop a designed, tested path.
for _name, _cls in (
    ("adam", torch.optim.Adam),
    ("adamw", torch.optim.AdamW),
    ("sgd", torch.optim.SGD),
    ("rmsprop", torch.optim.RMSprop),
    ("adamax", torch.optim.Adamax),
    ("nadam", torch.optim.NAdam),
    ("radam", torch.optim.RAdam),
    ("adagrad", torch.optim.Adagrad),
    ("adadelta", torch.optim.Adadelta),
    ("asgd", torch.optim.ASGD),
    ("rprop", torch.optim.Rprop),
    ("lbfgs", torch.optim.LBFGS),
    ("sparseadam", torch.optim.SparseAdam),
):
    OptimizerRegistry.register_class(_name, _cls)

# torch >= 2.5 only; the enum advertises it, so register it when present rather
# than letting the name resolve to nothing on an older wheel.
if hasattr(torch.optim, "Adafactor"):  # pragma: no branch - version dependent
    OptimizerRegistry.register_class("adafactor", torch.optim.Adafactor)


def _register_builtin_optimizers() -> None:
    """Import the optimizer package so its ``@register_optimizer`` decorators run.

    Deferred to the end of this module on purpose: ``optimizers/*.py`` import
    ``register_optimizer`` back from here, so this must run only once that name
    is bound on the partially-initialised module object. Moving this call any
    earlier turns the package into a circular-import failure.
    """
    from mriforge.infrastructure.training import optimizers as _optimizers  # noqa: F401


_register_builtin_optimizers()


def _assert_registry_covers_the_advertised_vocabulary() -> None:
    """Every ``OptimizerType`` member must be registered.

    Without this, a ``@register_optimizer`` sitting in a module that nothing
    imports is silently dead: the name vanishes from ``list_available()`` and a
    YAML naming it reports "unknown optimizer", which reads as a typo rather than
    a missing import. That is pitfall #15 wearing an enum's clothes — the enum
    advertises, and nothing checks that the advertisement is honoured.

    ``lars`` and ``lamb`` were members of that enum with no implementation
    anywhere for the whole life of the enum, and only a test xfail recorded it.
    """
    from mriforge.config.schemas.enums import OPTIMIZER_NAMES

    missing = OPTIMIZER_NAMES - set(OptimizerRegistry._registry)
    if missing:
        raise ImportError(
            f"OptimizerType advertises {sorted(missing)} but nothing registered "
            "them. A @register_optimizer in a module no __init__ imports is dead "
            "(CLAUDE.md #6) — add the import to "
            "mriforge/infrastructure/training/optimizers/__init__.py, or remove the "
            "enum member."
        )


_assert_registry_covers_the_advertised_vocabulary()
