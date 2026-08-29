"""Tests for the OptimizerRegistry + canonical scheduler surface (WS-D)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from mriforge.infrastructure.training.optimizer_registry import (
    SCHEDULER_REGISTRY,
    OptimizerRegistry,
    create_optimizer,
    create_scheduler,
    register_optimizer,
)


@pytest.fixture
def _isolated_registry() -> Iterator[None]:
    snap = dict(OptimizerRegistry._registry)
    try:
        yield
    finally:
        OptimizerRegistry._registry.clear()
        OptimizerRegistry._registry.update(snap)


def _params() -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(torch.zeros(2))]


def test_standard_optimizers_preregistered() -> None:
    avail = OptimizerRegistry.list_available()
    for name in ("adam", "adamw", "sgd", "rmsprop"):
        assert name in avail


def test_create_optimizer_instantiates() -> None:
    opt = create_optimizer("adamw", _params(), lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)


def test_unknown_optimizer_raises_no_silent_fallback() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer("__nope__", _params())


def test_register_custom_optimizer(_isolated_registry: None) -> None:
    @register_optimizer("wsd_custom_sgd", aliases=["wsd_csgd"])
    class _CustomSGD(torch.optim.SGD):
        pass

    opt = create_optimizer("wsd_csgd", _params(), lr=0.1)
    assert isinstance(opt, _CustomSGD)


def test_duplicate_registration_conflict_raises(_isolated_registry: None) -> None:
    @register_optimizer("wsd_dup")
    class _A(torch.optim.SGD):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register_optimizer("wsd_dup")
        class _B(torch.optim.SGD):
            pass


def test_scheduler_registry_is_reexported() -> None:
    # Single SSOT: this is the SAME dict object as scheduler_system's.
    from mriforge.infrastructure.training import scheduler_system

    assert SCHEDULER_REGISTRY is scheduler_system.SCHEDULER_REGISTRY


def test_create_scheduler_unknown_raises() -> None:
    opt = create_optimizer("sgd", _params(), lr=0.1)
    with pytest.raises(ValueError, match="Unknown scheduler"):
        create_scheduler("__nope__", opt)


def test_create_scheduler_cosine_alias() -> None:
    opt = create_optimizer("sgd", _params(), lr=0.1)
    # "cosine" aliases "cosine_annealing" — must resolve without raising.
    sched = create_scheduler("cosine", opt, {"T_max": 10})
    assert sched is not None


def test_get_returns_the_class_without_instantiating():
    """``get`` exists so callers can validate kwargs against the signature."""
    import torch

    from mriforge.infrastructure.training.optimizer_registry import OptimizerRegistry

    assert OptimizerRegistry.get("adamw") is torch.optim.AdamW
    assert OptimizerRegistry.get("AdamW") is torch.optim.AdamW


def test_get_returns_none_for_unregistered_name():
    from mriforge.infrastructure.training.optimizer_registry import OptimizerRegistry

    assert OptimizerRegistry.get("no_such_optimizer") is None
