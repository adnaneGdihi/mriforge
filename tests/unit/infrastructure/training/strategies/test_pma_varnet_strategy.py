"""PMA-VarNet must not re-resolve the device the environment already resolved.

The constructor used to run ``self.device = device or torch.device("cuda" if
torch.cuda.is_available() else "cpu")``. The factory never passes ``device``
(``strategy_factory.py`` constructs with ``env=`` and ``logging_service=``
only), so the right-hand side ran unconditionally and silently relocated the
run from ``env.device`` -- to ``cuda:0`` on a multi-GPU node, or to CPU on a
GPU-less one, which is the silent fallback non-negotiable 9b forbids.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from mriforge.infrastructure.training.strategies import pma_varnet_strategy
from mriforge.infrastructure.training.strategies.pma_varnet_strategy import (
    ConcretePMAVarNetStrategy,
)


def _env(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        device=device, config=MagicMock(), models={}, step=0, optimizers={}
    )


def test_strategy_does_not_overwrite_the_device_the_base_resolved(
    monkeypatch,
) -> None:
    """Isolated to the seam under test.

    The full base constructor builds optimizers and services, so it is stubbed
    down to the one thing that matters here -- it publishes ``env.device`` as
    ``self.device``. Anything the subclass then does to ``self.device`` is the
    behaviour this test exists to pin.
    """
    from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy

    def fake_init(self, env, device=None, **kwargs):
        self.device = env.device

    monkeypatch.setattr(BaseTrainingStrategy, "__init__", fake_init)

    env = _env(torch.device("cuda", 1))
    strategy = ConcretePMAVarNetStrategy(env)
    assert strategy.device == torch.device("cuda", 1), (
        "the strategy re-resolved the device and lost the environment's choice"
    )


def test_the_constructor_no_longer_re_resolves_the_device() -> None:
    """Source-level: the removed expression must not come back."""
    src = inspect.getsource(ConcretePMAVarNetStrategy.__init__)
    for node in ast.walk(ast.parse(inspect.cleandoc(src))):
        if isinstance(node, ast.IfExp):
            assert "cuda.is_available" not in ast.unparse(node)


def test_module_has_no_availability_probe_at_all() -> None:
    source = inspect.getsource(pma_varnet_strategy)
    assert "torch.cuda.is_available()" not in source
