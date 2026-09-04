"""Tests for the physics-operator interfaces.

Targets ``spectramr.infrastructure.physics.interfaces``. Three coordinate
abstractions:

- ``IPhysicsOperator`` (Protocol) — duck-typed forward / adjoint pair
- ``IPhysicsOperatorABC`` (ABC) — same surface, but inheritable
- ``IDataConsistency`` (ABC) — single ``enforce_consistency`` method

Categories:

- Backward-compat aliases ``PhysicsOperator`` / ``BasePhysicsOperator``
  point at the canonical interfaces
- ABC: cannot instantiate without overriding all abstract methods
- ABC: subclass implementing all methods can be instantiated
- ``IDataConsistency`` ABC: same instantiation constraint
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.interfaces import (
    BasePhysicsOperator,
    IDataConsistency,
    IPhysicsOperator,
    IPhysicsOperatorABC,
    PhysicsOperator,
)


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_physics_operator_alias() -> None:
    """``PhysicsOperator`` is an alias for ``IPhysicsOperator``."""
    assert PhysicsOperator is IPhysicsOperator


def test_base_physics_operator_alias() -> None:
    """``BasePhysicsOperator`` is an alias for ``IPhysicsOperatorABC``."""
    assert BasePhysicsOperator is IPhysicsOperatorABC


# ---------------------------------------------------------------------------
# IPhysicsOperatorABC abstract methods
# ---------------------------------------------------------------------------


def test_abc_cannot_be_instantiated_directly() -> None:
    """``IPhysicsOperatorABC`` has abstract methods → cannot construct."""
    with pytest.raises(TypeError, match="abstract"):
        IPhysicsOperatorABC()  # type: ignore[abstract]


def test_partial_subclass_cannot_be_instantiated() -> None:
    """A subclass that overrides only ``forward`` still abstract."""

    class _Partial(IPhysicsOperatorABC):
        def forward(self, img, **kwargs):
            return img

    with pytest.raises(TypeError, match="abstract"):
        _Partial()  # type: ignore[abstract]


def test_full_subclass_can_be_instantiated() -> None:
    """Override all four abstract members → instantiable."""

    class _Full(IPhysicsOperatorABC):
        def forward(self, img, **kwargs):
            return img

        def adjoint(self, data, **kwargs):
            return data

        @property
        def img_size(self):
            return (128, 128)

        def get_operator_type(self):
            return "identity"

    op = _Full()
    x = torch.randn(1, 1, 4, 4)
    assert torch.equal(op.forward(x), x)
    assert torch.equal(op.adjoint(x), x)
    assert op.img_size == (128, 128)
    assert op.get_operator_type() == "identity"


# ---------------------------------------------------------------------------
# IDataConsistency
# ---------------------------------------------------------------------------


def test_data_consistency_cannot_be_instantiated() -> None:
    """``IDataConsistency`` is abstract — direct construction fails."""
    with pytest.raises(TypeError, match="abstract"):
        IDataConsistency()  # type: ignore[abstract]


def test_data_consistency_subclass_works() -> None:
    """Concrete subclass overriding ``enforce_consistency`` is instantiable."""

    class _DC(IDataConsistency):
        def enforce_consistency(self, _current_img, _measured_data, mask=None):
            return _current_img

    dc = _DC()
    x = torch.randn(1, 1, 4, 4)
    assert torch.equal(dc.enforce_consistency(x, x), x)
