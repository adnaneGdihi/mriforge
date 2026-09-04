"""Tests for the canonical training-step I/O + compat adapter (WS-D)."""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from spectramr.infrastructure.training.step_io import (
    TrainingStepInput,
    TrainingStepOutput,
    accepts_step_io,
)


class _Strategy:
    @accepts_step_io
    def train_step(self, batch, epoch, input_batch=None, target_batch=None, **kw):
        return {
            "batch": batch,
            "epoch": epoch,
            "input_batch": input_batch,
            "target_batch": target_batch,
            "extras": kw,
        }

    @accepts_step_io
    def narrow_step(self, batch, epoch):  # no **kwargs — projects onto declared names
        return (batch, epoch)


def test_legacy_kwargs_call_passes_through() -> None:
    out = _Strategy().train_step("B", 2, input_batch="I", target_batch="T")
    assert out["batch"] == "B" and out["epoch"] == 2
    assert out["input_batch"] == "I" and out["target_batch"] == "T"
    assert out["extras"] == {}


def test_dataclass_call_is_exploded() -> None:
    sio = TrainingStepInput(batch="B", epoch=2, input_batch="I", iteration=7)
    out = _Strategy().train_step(sio)
    assert out["batch"] == "B" and out["input_batch"] == "I"
    # train_step has **kwargs, so the rest of the union lands in extras
    assert out["extras"]["iteration"] == 7


def test_projection_onto_narrow_signature() -> None:
    sio = TrainingStepInput(batch="B", epoch=9, iteration=3, scaler=object())
    # narrow_step declares only (batch, epoch) and NO **kwargs -> only those bind
    assert _Strategy().narrow_step(sio) == ("B", 9)


def test_explicit_kwargs_win_over_dataclass() -> None:
    sio = TrainingStepInput(batch="B", epoch=2)
    out = _Strategy().train_step(sio, epoch=99)
    assert out["epoch"] == 99


def test_marker_attribute_present() -> None:
    assert getattr(_Strategy.train_step, "__accepts_step_io__", False) is True


def test_silent_on_legacy_by_default() -> None:
    # The repo promotes spectramr DeprecationWarning to an error; the shim MUST be
    # silent on the legacy path by default.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _Strategy().train_step("B", 1)  # must not raise


def test_warn_opt_in() -> None:
    class _W:
        @accepts_step_io(warn=True)
        def step(self, batch, epoch):
            return batch

    with pytest.warns(DeprecationWarning):
        _W().step("B", 0)


def test_output_constructors_and_frozen() -> None:
    out = TrainingStepOutput.from_records([{"loss": 1.0}])
    assert out.to_records() == [{"loss": 1.0}]
    vout = TrainingStepOutput.from_metrics({"psnr": 30.0})
    assert vout.metrics["psnr"] == 30.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.total_loss = 1  # type: ignore[misc]


def test_input_is_frozen() -> None:
    sio = TrainingStepInput(batch="B")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sio.epoch = 5  # type: ignore[misc]
