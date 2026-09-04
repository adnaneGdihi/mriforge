"""Tests for :meth:`TrainingEnvironment.from_components`.

The config-driven path builds a :class:`TrainingEnvironment` via the
``TrainingEnvironmentDirector``. The scripting path (``spectramr.api.fit``) needs
to build the *same* immutable container from loose, hand-supplied objects so it
can drive the *same* loop. ``from_components`` is that adapter: it maps a user's
model/optimizer/losses/loaders onto the exact attribute contract the loop and
strategies read (``models["generator"]``, ``optimizers["opt_g"]``, …), filling
absent collections with empty dicts and resolving the device.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment


def _cfg() -> TrainingSettings:
    return TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {},
            "optimization": {},
            "logging": {},
        }
    )


def test_from_components_minimal_satisfies_contract():
    cfg = _cfg()
    model = nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters())

    env = TrainingEnvironment.from_components(
        config=cfg,
        models={"generator": model},
        optimizers={"opt_g": opt},
        device="cpu",
    )

    # The canonical accessors the loop + strategies read.
    assert env.generator is model
    assert env.opt_g is opt
    assert env.config is cfg
    # Absent collections default to empty dicts (never None — the loop indexes them).
    assert env.losses == {}
    assert env.physics == {}
    assert env.data_loaders == {}
    assert env.metrics == {}
    assert env.schedulers == {}
    assert env.scaler is None
    # The vestigial frozen ``step`` field was removed (WS-3 follow-up): the live
    # iteration SSOT is ``strategy.loop_state.iteration``, not env.step. Guard
    # that the inert seam is not reintroduced.
    assert not hasattr(env, "step")


def test_from_components_resolves_string_device():
    env = TrainingEnvironment.from_components(
        config=_cfg(),
        models={"generator": nn.Linear(2, 2)},
        optimizers={"opt_g": torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)},
        device="cpu",
    )
    assert isinstance(env.device, torch.device)
    assert env.device.type == "cpu"


def test_from_components_passes_losses_and_loaders():
    cfg = _cfg()
    model = nn.Linear(4, 4)
    loss = nn.L1Loss()
    loader = torch.utils.data.DataLoader([torch.zeros(4)], batch_size=1)

    env = TrainingEnvironment.from_components(
        config=cfg,
        models={"generator": model},
        optimizers={"opt_g": torch.optim.Adam(model.parameters())},
        losses={"l1": loss},
        data_loaders={"train": loader},
        device="cpu",
    )
    assert env.losses["l1"] is loss
    assert env.data_loaders["train"] is loader


def test_from_components_returns_frozen_instance():
    env = TrainingEnvironment.from_components(
        config=_cfg(),
        models={"generator": nn.Linear(2, 2)},
        optimizers={"opt_g": torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)},
        device="cpu",
    )
    # frozen dataclass — mutation raises.
    import dataclasses

    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        env.ema = 5  # type: ignore[misc]
