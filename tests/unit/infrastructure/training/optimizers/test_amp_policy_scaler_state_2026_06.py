"""``AMPPolicy`` must not hard-depend on GradScaler's private state.

``backward_and_step`` reads ``scaler._per_optimizer_states`` (a private
torch attribute) to detect the "unscale_ recorded no inf-checks" corner
case. A torch upgrade that renames the attribute would crash every AMP
step with ``AttributeError``. The access now goes through tolerant
helpers: when the private API is unavailable, the policy falls back to
the standard public pattern (``scaler.step`` + ``scaler.update``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from mriforge.infrastructure.training.optimizers.amp_policy import AMPPolicy

pytestmark = pytest.mark.unit


class _ScalerWithoutPrivateState:
    """Mimics a future GradScaler whose private state attr was renamed."""

    def __init__(self) -> None:
        self.stepped = False
        self.updated = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer) -> None:
        pass

    def step(self, optimizer) -> None:
        self.stepped = True

    def update(self) -> None:
        self.updated = True

    def __getattr__(self, name: str):
        if name == "_per_optimizer_states":
            raise AttributeError(name)
        raise AttributeError(name)


def test_backward_and_step_survives_missing_private_scaler_state() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.randn(4, 2)).sum()
    scaler = _ScalerWithoutPrivateState()

    policy = AMPPolicy()
    # Must not raise AttributeError; must fall back to the public pattern.
    policy.backward_and_step(
        loss=loss,
        optimizer=optimizer,
        model=model,
        model_name="g",
        epoch=0,
        model_type="reconstruction",
        scaler=scaler,
    )

    assert scaler.stepped, "fallback must use the standard scaler.step path"
    assert scaler.updated, "fallback must use the standard scaler.update path"


def test_backward_and_step_normal_scaler_still_works() -> None:
    """The torch GradScaler path (private API present) keeps working."""
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.randn(4, 2)).sum()
    scaler = MagicMock()
    scaler.scale.side_effect = lambda x: x
    scaler._per_optimizer_states = {
        id(optimizer): {"found_inf_per_device": {"cuda:0": torch.zeros(1)}}
    }

    AMPPolicy().backward_and_step(
        loss=loss,
        optimizer=optimizer,
        model=model,
        model_name="g",
        epoch=0,
        model_type="reconstruction",
        scaler=scaler,
    )

    scaler.step.assert_called_once_with(optimizer)
    scaler.update.assert_called_once()
