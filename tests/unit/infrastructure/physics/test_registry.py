"""Tests for the forward-operator registry.

Targets ``mriforge.infrastructure.physics.registry``. Dispatches forward-
operator implementations (FFT, NUFFT, …) by name with optional default
configs.

Categories:

- Construction: empty registry has no operators
- ``register_operator`` rejects classes not inheriting ``BaseForwardOperator``
- ``get_operator_class`` raises ``KeyError`` with available list on miss
- ``create_operator`` merges default config + kwargs
- ``list_operators`` returns metadata for every registered class
- Global helpers (``register_operator``, ``create_operator``, …) target
  the module-level singleton
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.interfaces import IPhysicsOperatorABC
from mriforge.infrastructure.physics.registry import (
    OperatorRegistry,
    create_operator,
    get_operator_class,
    get_registry,
    list_operators,
    register_operator,
)


# ---------------------------------------------------------------------------
# Test dummy operator
# ---------------------------------------------------------------------------


class _DummyOp(IPhysicsOperatorABC):
    """Identity operator for registry tests."""

    def __init__(self, scale: float = 1.0, im_size: tuple[int, int] = (8, 8)) -> None:
        self.scale = scale
        self._im_size = im_size

    def forward(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        return img * self.scale

    def adjoint(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return data / self.scale

    @property
    def img_size(self) -> tuple[int, int]:
        return self._im_size

    def get_operator_type(self) -> str:
        return "dummy"


# ---------------------------------------------------------------------------
# OperatorRegistry instance
# ---------------------------------------------------------------------------


def test_register_and_lookup() -> None:
    """Registering a class makes it retrievable by name."""
    registry = OperatorRegistry()
    registry.register_operator("dummy", _DummyOp, config={"scale": 2.0})
    assert registry.get_operator_class("dummy") is _DummyOp


def test_register_non_subclass_raises() -> None:
    """``register_operator`` rejects classes not inheriting ``BaseForwardOperator``."""

    class _NotAnOp:
        pass

    registry = OperatorRegistry()
    with pytest.raises(TypeError, match="must inherit"):
        registry.register_operator("bad", _NotAnOp)  # type: ignore[arg-type]


def test_get_unknown_raises_keyerror() -> None:
    """``get_operator_class`` raises ``KeyError`` listing available operators."""
    registry = OperatorRegistry()
    with pytest.raises(KeyError, match="not found"):
        registry.get_operator_class("missing")


def test_default_config_returned_as_copy() -> None:
    """``get_operator_config`` returns a defensive copy."""
    registry = OperatorRegistry()
    registry.register_operator("dummy", _DummyOp, config={"scale": 2.0})
    cfg = registry.get_operator_config("dummy")
    cfg["scale"] = 999.0  # mutating local copy must not affect registry
    cfg2 = registry.get_operator_config("dummy")
    assert cfg2["scale"] == 2.0


def test_create_operator_merges_default_and_kwargs() -> None:
    """Default config is merged with explicit kwargs at creation time."""
    registry = OperatorRegistry()
    registry.register_operator("dummy", _DummyOp, config={"scale": 2.0})
    op = registry.create_operator("dummy", im_size=(16, 16))
    assert isinstance(op, _DummyOp)
    assert op.scale == 2.0  # from default config
    assert op.img_size == (16, 16)  # from kwargs


def test_create_operator_kwargs_override_default() -> None:
    """Explicit kwargs win over default config on key collision."""
    registry = OperatorRegistry()
    registry.register_operator("dummy", _DummyOp, config={"scale": 2.0})
    op = registry.create_operator("dummy", scale=5.0)
    assert op.scale == 5.0


def test_create_operator_constructor_failure_wrapped() -> None:
    """Constructor errors are wrapped in ``RuntimeError`` with context."""
    registry = OperatorRegistry()
    registry.register_operator("dummy", _DummyOp)
    with pytest.raises(RuntimeError, match="Failed to create operator"):
        registry.create_operator("dummy", unknown_arg=1)


def test_list_operators_metadata_shape() -> None:
    """``list_operators`` returns ``{name: {class, config}}`` per entry."""
    registry = OperatorRegistry()
    registry.register_operator("dummy", _DummyOp, config={"scale": 2.0})
    listed = registry.list_operators()
    assert "dummy" in listed
    assert listed["dummy"]["class"] == "_DummyOp"
    assert listed["dummy"]["config"] == {"scale": 2.0}


# ---------------------------------------------------------------------------
# Module-level helpers (global singleton)
# ---------------------------------------------------------------------------


def test_global_helpers_target_singleton() -> None:
    """``register_operator``/``create_operator`` work against ``get_registry()``."""
    name = "__test_dummy_for_registry__"
    register_operator(name, _DummyOp, config={"scale": 3.0})
    try:
        cls = get_operator_class(name)
        assert cls is _DummyOp
        assert name in list_operators()
        op = create_operator(name)
        assert op.scale == 3.0
        assert get_registry() is not None
    finally:
        # Clean up so other tests don't see the dummy.
        registry = get_registry()
        registry._operators.pop(name, None)
        registry._operator_configs.pop(name, None)
