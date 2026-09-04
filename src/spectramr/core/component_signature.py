"""What a component's constructor will actually accept — one reading, shared.

Every family that builds a component from YAML has to answer the same question:
does this declared key reach the constructor? Several places answered it
separately (``model_factory._filter_kwargs_to_signature``,
``optimizer_registry``, ``core/metrics/registry``) and the loss builder did not
answer it at all — which is how ``sobolev_order: 1`` sat declared-and-dead in 56
arms for weeks (issues #560, #615).

**Ownership comes from the MRO, never from ``inspect.signature(cls)``.** A
:class:`torch.nn.Module` subclass with no ``__init__`` of its own reports
``nn.Module``'s signature, which promises to accept things it will drop on the
floor. Reading that as a contract is what left 8 registered metrics silently
dead until PR #718. :func:`owned_init` therefore walks ``__mro__`` for the class
that has ``__init__`` in its own ``__dict__``, and treats ``nn.Module`` /
``object`` as "defines nothing".

Lives in ``core/`` because both ``models/`` and ``infrastructure/`` consume it
and neither may be imported from here (the layering rule points inward only).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

__all__ = ["SignatureContract", "owned_init", "signature_contract"]


@dataclass(frozen=True, slots=True)
class SignatureContract:
    """The parameters a component's constructor names, and whether it takes more."""

    #: Named parameters the owned ``__init__`` accepts, excluding ``self``.
    accepted: frozenset[str]
    #: ``True`` when the owned ``__init__`` declares ``**kwargs``. Such a class
    #: accepts every key and typically reads a subset via ``kwargs.get(...)``, so
    #: acceptance here is **not** evidence of consumption — see issue #878.
    accepts_var_kwargs: bool
    #: Name of the class that owns ``__init__``; ``""`` when none does.
    owner: str


def _rootless_bases() -> tuple[type, ...]:
    """Classes whose ``__init__`` declares no contract for a component.

    ``torch`` is imported lazily: ``conftest.py`` installs a ``MagicMock`` shim
    when torch is unavailable, and a module-level import here would bind the
    mock's attributes into the tuple.
    """
    try:
        import torch.nn as nn

        return (nn.Module, object)
    except Exception:  # torch absent (CPU/CI shim) — object is still a root
        return (object,)


def owned_init(cls: type) -> type | None:
    """The class in ``cls.__mro__`` that actually defines ``__init__``.

    Returns ``None`` when only ``nn.Module`` / ``object`` define it, i.e. the
    component declares no constructor contract of its own.
    """
    rootless = _rootless_bases()
    for klass in cls.__mro__:
        if "__init__" in klass.__dict__:
            return None if klass in rootless else klass
    return None


def signature_contract(cls: type) -> SignatureContract:
    """Resolve ``cls`` into the parameter set its constructor will accept."""
    owner = owned_init(cls)
    if owner is None:
        return SignatureContract(frozenset(), False, "")

    try:
        # Read the function out of the owner's own ``__dict__`` rather than via
        # attribute access. ``owned_init`` already proved the key is there, and
        # ``owner.__init__`` would go through the descriptor protocol — which
        # mypy correctly calls unsound, since the attribute could resolve to an
        # incompatible subclass's. The whole point of this module is to read the
        # function that class actually defines.
        params = inspect.signature(owner.__dict__["__init__"]).parameters
    except (ValueError, TypeError, KeyError):
        # Signature inspection failed (e.g. a C extension). Refusing to guess:
        # report a permissive contract so no caller drops a key on a reading it
        # could not make. A caller that must be strict checks ``owner`` first.
        return SignatureContract(frozenset(), True, owner.__name__)

    accepted = frozenset(
        name
        for name, p in params.items()
        if name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    )
    var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    return SignatureContract(accepted, var_kw, owner.__name__)
