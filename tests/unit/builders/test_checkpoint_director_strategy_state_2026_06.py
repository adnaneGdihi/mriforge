"""CheckpointDirector ↔ strategy-owned state round-trip (2026-06, section R).

Proves the director saves ``strategy.strategy_state_dict()`` under the
``strategy_state`` key and restores it on ``load_from`` — exercising the REAL
``BaseTrainingStrategy`` seam (not a stub), so this doubles as the end-to-end
acceptance test for the corrupt-resume fix.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from mriforge.infrastructure.builders.directors.checkpoint_director import (
    CheckpointDirector,
)
from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy


class _Strat(BaseTrainingStrategy):
    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - stub
        return {}


def _strategy_with_owned_sfc(seed: int) -> _Strat:
    """A strategy that owns an ``sfc`` module (not reachable from env)."""
    s = object.__new__(_Strat)
    s.env = SimpleNamespace(generator=nn.Linear(4, 4), discriminator=None, losses={})
    torch.manual_seed(seed)
    s.sfc = nn.Linear(3, 3)
    return s


def _fake_pipeline(generator: nn.Module):
    opt = torch.optim.SGD(generator.parameters(), lr=0.1)
    return SimpleNamespace(
        generator=generator,
        discriminator=None,
        optimizer_g=opt,
        optimizer_d=None,
        scheduler_g=None,
        scheduler_d=None,
        scaler=None,
        device=torch.device("cpu"),
    )


def test_director_round_trips_strategy_owned_state(tmp_path) -> None:
    ckpt_dir = str(tmp_path / "checkpoints")

    # --- save: a strategy whose owned sfc head has distinctive weights ---
    save_strat = _strategy_with_owned_sfc(seed=0)
    with torch.no_grad():
        save_strat.sfc.weight.add_(2.5)  # make the weights non-default
    save_pipeline = _fake_pipeline(nn.Linear(4, 4))

    director = CheckpointDirector(MagicMock())
    path = (
        director.with_checkpoint_dir(ckpt_dir)
        .with_pipeline(save_pipeline)
        .with_strategy(save_strat)
        .with_epoch(1)
        .with_global_step(100)
        .validate()
        .save()
    )

    # The owned state is persisted under the sibling key (not inside "generator").
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert "strategy_state" in blob
    assert "sfc" in blob["strategy_state"]["modules"]
    assert "sfc.weight" not in blob["generator"]  # not merged into the generator

    # --- resume: a FRESH strategy with a differently-initialised sfc ---
    load_strat = _strategy_with_owned_sfc(seed=999)
    assert not torch.allclose(load_strat.sfc.weight, save_strat.sfc.weight)

    load_pipeline = _fake_pipeline(nn.Linear(4, 4))
    resume = CheckpointDirector(MagicMock())
    ok = (
        resume.with_checkpoint_dir(ckpt_dir)
        .with_pipeline(load_pipeline)
        .with_strategy(load_strat)
        .load_from(str(path))
    )
    assert ok is True
    # The owned sfc weights are restored bit-for-bit (the corrupt-resume fix).
    assert torch.allclose(load_strat.sfc.weight, save_strat.sfc.weight)
    assert torch.allclose(load_strat.sfc.bias, save_strat.sfc.bias)


def test_director_without_strategy_writes_no_strategy_state(tmp_path) -> None:
    # Backward compatibility: no .with_strategy(...) → no strategy_state key,
    # and load_from on such a checkpoint is a clean no-op for owned state.
    director = CheckpointDirector(MagicMock())
    path = (
        director.with_checkpoint_dir(str(tmp_path / "ckpt"))
        .with_pipeline(_fake_pipeline(nn.Linear(4, 4)))
        .with_epoch(0)
        .with_global_step(0)
        .validate()
        .save()
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert "strategy_state" not in blob


def test_director_skips_strategy_state_when_nothing_owned(tmp_path) -> None:
    # A strategy that owns no modules/params must not add an empty key.
    s = object.__new__(_Strat)
    s.env = SimpleNamespace(generator=nn.Linear(4, 4), discriminator=None, losses={})
    s.scalar = 0.5  # not a module/param

    director = CheckpointDirector(MagicMock())
    path = (
        director.with_checkpoint_dir(str(tmp_path / "ckpt"))
        .with_pipeline(_fake_pipeline(nn.Linear(4, 4)))
        .with_strategy(s)
        .with_epoch(0)
        .with_global_step(0)
        .validate()
        .save()
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert "strategy_state" not in blob
