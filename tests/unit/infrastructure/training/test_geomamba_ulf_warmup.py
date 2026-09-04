"""Unit test for the GeoMamba-ULF P3 topology-loss warmup schedule.

The warmup factor is a tiny piece of state on
:class:`GeoMambaULFStrategy`. We pin its math directly without
instantiating the full strategy (which needs a config + DI
container) by binding the unbound method onto a minimal stub.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # the strategy module imports torch at top level

from spectramr.infrastructure.training.strategies.geomamba_ulf_strategy import (  # noqa: E402
    GeoMambaULFStrategy,
)


class _Stub:
    """Minimal duck-type holder for ``_topology_warmup_epochs``."""

    _topology_warmup_factor = GeoMambaULFStrategy._topology_warmup_factor

    def __init__(self, warmup_epochs: int) -> None:
        self._topology_warmup_epochs = warmup_epochs


def test_warmup_disabled_returns_one() -> None:
    s = _Stub(warmup_epochs=0)
    assert s._topology_warmup_factor(0) == 1.0
    assert s._topology_warmup_factor(100) == 1.0


def test_warmup_first_epoch_is_inverse_of_warmup_epochs() -> None:
    s = _Stub(warmup_epochs=10)
    # epoch=0 -> (0 + 1) / 10 = 0.1
    assert s._topology_warmup_factor(0) == pytest.approx(0.1)


def test_warmup_reaches_one_at_warmup_epoch_boundary() -> None:
    s = _Stub(warmup_epochs=10)
    # epoch=9 -> (9 + 1) / 10 = 1.0
    assert s._topology_warmup_factor(9) == pytest.approx(1.0)


def test_warmup_clamps_at_one_after_boundary() -> None:
    s = _Stub(warmup_epochs=5)
    assert s._topology_warmup_factor(50) == 1.0
    assert s._topology_warmup_factor(1000) == 1.0


def test_warmup_handles_negative_epoch_gracefully() -> None:
    s = _Stub(warmup_epochs=10)
    # max(epoch + 1, 0) -> 0 for epoch <= -1, so factor is 0/10 = 0.
    assert s._topology_warmup_factor(-5) == 0.0
